"""Sentinel-2 methane screening — clean version.

FIXES:
  • Removed ternary with st calls (caused DeltaGenerator crash)
  • Removed nested columns inside columns (Streamlit cursor bug)
  • Removed nested-quote f-strings
  • Removed binary_opening (was killing signal)
  • All filter params exposed in UI
"""
from __future__ import annotations

import io
import json
import math
import re
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
from folium.plugins import Draw
from scipy.ndimage import binary_dilation, gaussian_filter, label, convolve
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from skimage.measure import regionprops
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
BANDS = ["B03", "B04", "B08", "B11", "B12"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".sentinel_methane_cache"
RESULT_DIR = Path.home() / ".sentinel_methane_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

S5P_COLLECTION = "sentinel-5p-l2"
DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

PARAMS = {
    "b03_quantile": 0.05,
    "swir_saturation": 1.0,
    "ndwi_threshold": 0.20,
    "ndvi_threshold": 0.30,
    "ndbi_threshold": 0.20,
    "ndsi_threshold": 0.42,
    "lrad_dilation": 2,
    "n_reference_scenes": 6,
    "gaussian_sigma": 1.5,
    "threshold_sigma": 2.0,
    "spatial_coherence_min": 4,
    "min_component_pixels": 10,
    "final_dilation": 2,
    "min_solidity": 0.15,
    "b12_epsilon": 1e-6,
    "min_valid_ref_pixels": 100,
}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def as_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return dict(item)


def get_properties(item):
    return as_dict(item).get("properties", {}) or {}


def get_datetime(item) -> Optional[datetime]:
    data = as_dict(item)
    props = get_properties(data)
    value = props.get("datetime") or props.get("start_datetime")
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    match = re.search(r"_(\d{8}T\d{6})_", data.get("id", "").upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data = as_dict(item)
    props = get_properties(data)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key):
            return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", data.get("id", "").upper())
    return match.group(1) if match else None


def get_cloud(item):
    props = get_properties(item)
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try:
            return float(props[key])
        except Exception:
            continue
    return 100.0


def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geometries = []
        for feature in obj.get("features", []):
            geometry = normalize_geometry(feature.get("geometry"))
            if geometry:
                geometries.append(shape(geometry))
        return mapping(unary_union(geometries)) if geometries else None
    try:
        geometry = shape(obj)
        return mapping(geometry) if not geometry.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(DEFAULT_AOI)


def search_scenes(aoi, start, end, max_cloud):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "intersects": ensure_aoi(aoi),
        "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
        "limit": 100,
    }
    response = requests.post(f"{STAC_URL}search", json=payload, timeout=120)
    response.raise_for_status()
    features = response.json().get("features", [])
    return features if isinstance(features, list) else list(features)


def authenticate_cdse(username: str, password: str, totp: str = ""):
    username = username.strip()
    password = password.strip()
    totp = totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")
    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    if totp:
        form["totp"] = totp
    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")
    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Copernicus did not return an access token.")
    now = time.time()
    return {
        "access_token": access_token,
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": now + int(data.get("expires_in", 600)),
        "username": username,
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        return None
    data = response.json()
    token = data.get("access_token")
    if not token:
        return None
    auth["access_token"] = token
    auth["refresh_token"] = data.get("refresh_token", refresh_token)
    auth["expires_at"] = time.time() + int(data.get("expires_in", 600))
    return auth


def get_access_token():
    auth = st.session_state.get("cdse_auth")
    if not auth:
        raise RuntimeError("Please log in to Copernicus first.")
    if time.time() < float(auth.get("expires_at", 0)) - 60:
        return auth["access_token"]
    refreshed = refresh_cdse_session(auth)
    if refreshed:
        st.session_state.cdse_auth = refreshed
        return refreshed["access_token"]
    st.session_state.pop("cdse_auth", None)
    raise RuntimeError("Your Copernicus session expired. Please log in again.")


def evalscript():
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03","B04","B08","B11","B12"], units: "REFLECTANCE"}],
    output: {bands: 5, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
}
"""


def s5p_evalscript():
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: ["CH4", "dataMask"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.CH4, 1.0];
}
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=payload, timeout=900)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")
    output_path.write_bytes(response.content)
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path


def download_s5p_scene(aoi, date_from, date_to, access_token):
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps(["s5p_v6", aoi, str(date_from), str(date_to)], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / f"s5p_{cache_id}"
    output_path = folder / "ch4.tif"
    if output_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": S5P_COLLECTION, "dataFilter": {"timeRange": {"from": date_from.strftime("%Y-%m-%dT00:00:00Z"), "to": date_to.strftime("%Y-%m-%dT23:59:59Z")}}}],
        },
        "output": {"width": 256, "height": 256, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": s5p_evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=payload, timeout=300)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"S5P Process API failed ({response.status_code}): {detail}")
    output_path.write_bytes(response.content)
    return output_path


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return {band: array[index] for index, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in BANDS])
    artifact = ((bands["B11"] >= PARAMS["swir_saturation"]) & (bands["B12"] >= PARAMS["swir_saturation"]))
    artifact = artifact | (bands["B03"] <= q_value)
    artifact = artifact | (normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"])
    artifact = artifact | (normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"])
    artifact = artifact | (normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"])
    artifact = artifact | (normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"])
    artifact = artifact | (~finite)
    artifact = binary_dilation(artifact, structure=disk(PARAMS["lrad_dilation"]))
    return finite & (~artifact)


def calculate_c(b11, b12, valid):
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0) & (b12 > 0)
    if use.sum() < 100:
        return 1.0
    x, y = b11[use].astype(np.float64), b12[use].astype(np.float64)
    return float(np.sum(x * y) / max(np.sum(x * x), 1e-20))


def calculate_mbsp(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (np.abs(b12) > PARAMS["b12_epsilon"])
    output[use] = c * (b12[use] - b11[use]) / b12[use]
    return output


def run_algorithm_multi(target, reference_median, valid_lrad):
    c_target = calculate_c(target["B11"], target["B12"], valid_lrad)
    target_mbsp = calculate_mbsp(target["B11"], target["B12"], c_target, valid_lrad)
    reference_mbsp = calculate_mbsp(reference_median["B11"], reference_median["B12"], c_target, valid_lrad)
    relative = target_mbsp - reference_mbsp
    relative[~valid_lrad] = np.nan
    finite = np.isfinite(relative)
    if not finite.any():
        raise RuntimeError("LRAD removed all pixels.")

    values = relative[finite].astype(np.float64)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    gaussian = gaussian_filter(np.where(finite, relative, mean_value), sigma=PARAMS["gaussian_sigma"])
    threshold = mean_value + PARAMS["threshold_sigma"] * std_value

    above = valid_lrad & np.isfinite(gaussian) & (gaussian > threshold)

    neighbor_count = convolve(
        above.astype(np.float32),
        np.ones((3, 3), dtype=np.float32),
        mode="constant",
    )
    neighbor_count = neighbor_count - above.astype(np.float32)

    coherent = above & (neighbor_count >= PARAMS["spatial_coherence_min"])

    labels, _ = label(coherent, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= PARAMS["min_component_pixels"])[0]
    retained = retained[retained != 0]

    if len(retained) > 0:
        keep = set()
        try:
            props = regionprops(labels)
            for prop in props:
                if prop.label not in retained:
                    continue
                if prop.solidity < PARAMS["min_solidity"]:
                    continue
                keep.add(prop.label)
        except Exception:
            keep = set(retained.tolist())
        retained = np.array(sorted(keep)) if keep else np.array([], dtype=int)

    connected = np.isin(labels, retained)
    final = binary_dilation(connected, structure=disk(PARAMS["final_dilation"])) & valid_lrad

    return {
        "relative": relative,
        "gaussian": gaussian,
        "valid": valid_lrad,
        "initial": above,
        "coherent": coherent,
        "connected": connected,
        "final": final,
        "mean": mean_value,
        "std": std_value,
        "threshold": threshold,
        "regions": int(len(retained)),
        "valid_count": int(valid_lrad.sum()),
        "initial_count": int(above.sum()),
        "coherent_count": int(coherent.sum()),
        "final_count": int(final.sum()),
    }


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    output = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map([geometry.centroid.y, geometry.centroid.x], zoom_start=11, tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    return fmap


def image_png(array, mask=False):
    from PIL import Image
    data = np.asarray(array)
    if mask:
        rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        rgb[data.astype(bool)] = [220, 30, 30]
    else:
        finite = np.isfinite(data)
        rgb = np.full((data.shape[0], data.shape[1], 3), 255, dtype=np.uint8)
        if finite.any():
            values = data[finite]
            low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
            if high <= low:
                low, high = float(values.min()), float(values.max())
            if high > low:
                normalized = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
                import matplotlib.pyplot as plt
                rgb = (plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def ch4_anomaly_png(array):
    from PIL import Image
    import matplotlib.pyplot as plt
    data = np.asarray(array).astype(np.float32)
    finite = np.isfinite(data)
    rgb = np.full((data.shape[0], data.shape[1], 3), 255, dtype=np.uint8)
    if finite.any() and finite.sum() > 1:
        values = data[finite]
        mean_val = float(np.mean(values))
        anomaly = data - mean_val
        max_abs = float(np.percentile(np.abs(anomaly[finite]), 98))
        if max_abs < 1e-6:
            max_abs = float(np.max(np.abs(anomaly[finite])))
        if max_abs > 0:
            normalized = np.clip((np.nan_to_num(anomaly, nan=0) + max_abs) / (2.0 * max_abs), 0, 1)
            rgb = (plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255).astype(np.uint8)
            rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def legend_html(kind):
    if kind == "mask":
        rows = [("#dc1e1e", "Candidate"), ("#000000", "Background")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid"), ("#000000", "Invalid")]
    elif kind == "s5p":
        rows = [("#b43232", "Above local mean"), ("#3250b4", "Below local mean"), ("#ffffff", "No data")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(
        '<div class="legend-row"><span class="legend-swatch" style="background:' + c + ';"></span><span>' + t + '</span></div>'
        for c, t in rows
    )
    return '<div class="result-legend"><div class="legend-heading">Legend</div>' + items + '</div>'


def create_png_worldfile(profile, array_shape):
    transform = profile["transform"]
    xres = transform.a
    yres = transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    pgw = str(xres) + "\n0.0\n0.0\n" + str(yres) + "\n" + str(x_center) + "\n" + str(y_center) + "\n"
    crs_text = profile.get("crs")
    prj = crs_text.to_wkt() if crs_text else ""
    return pgw.encode("utf-8"), prj.encode("utf-8")


def georeferenced_png_package(array, profile, mask=False):
    import zipfile
    png_data = image_png(array, mask=mask)
    pgw_data, prj_data = create_png_worldfile(profile, np.asarray(array).shape)
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", png_data)
        archive.writestr("output.pgw", pgw_data)
        if prj_data:
            archive.writestr("output.prj", prj_data)
    return package.getvalue()


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Sentinel-2 Methane", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: #f1faee; }
[data-testid="stHeader"] { background: #f1faee !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.5rem; }
h1, h2, h3, h4, p, label, small, strong, em, li, td, th, div[data-testid="stMarkdownContainer"] * { color: #111111 !important; }
div[data-testid="stDateInput"] input, .stDateInput input,
div[data-testid="stNumberInput"] input, .stNumberInput input {
    background-color: #292a33 !important; color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
input::placeholder { color: #bfc3cc !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] [role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="listbox"] *, div[data-baseweb="popover"] [role="option"] * { color: #ffffff !important; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.85rem; color: #111111 !important; }
.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }
.stButton > button[kind="primary"] * { color: #ffffff !important; }
.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.82rem; }
.legend-swatch { width: 18px; height: 14px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Sentinel-2 Methane Screening")
st.caption("Multi-reference median + tunable spatial filter · Aradkouh landfill default AOI")

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)

# ══════════════════════════════════════════════════════════════════════
#  ROW 1: AOI map + Scene search
# ══════════════════════════════════════════════════════════════════════
map_col, ctrl_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.subheader("01 · Study Area")
    st.caption("Draw or edit the AOI directly on the map.")
    map_data = st_folium(create_map(st.session_state.aoi), height=385, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi:
            st.session_state.aoi = new_aoi

with ctrl_col:
    st.subheader("02 · Scene Search")
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", default_start, key="start_date")
    with d2:
        end_date = st.date_input("End date", default_end, key="end_date")

    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 30.0, key="max_cloud")
    with s2:
        reference_days = st.slider("Reference window (days)", 1, 90, 60, key="reference_days")

    search_clicked = st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True, key="search_btn")

    if search_clicked:
        try:
            with st.spinner("Searching CDSE STAC..."):
                sr = search_scenes(
                    st.session_state.aoi,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    max_cloud,
                )
            if sr is None:
                sr = []
            elif not isinstance(sr, list):
                sr = list(sr)
            st.session_state["scene_results"] = sr
            st.session_state.pop("target", None)
            if len(sr) > 0:
                st.success(str(len(sr)) + " scene(s) found")
            else:
                st.warning("No Sentinel-2 scenes were found.")
        except Exception as e:
            st.session_state["scene_results"] = []
            st.session_state.pop("target", None)
            st.error("Search failed: " + str(e))

    scene_results = st.session_state.get("scene_results", [])

    if len(scene_results) > 0:
        scene_table = pd.DataFrame([
            {"date": get_datetime(s), "tile": get_tile(s), "cloud": get_cloud(s)}
            for s in scene_results
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")

        st.dataframe(
            scene_table,
            use_container_width=True,
            height=112,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        scene_ids = []
        for s in scene_results:
            sid = as_dict(s).get("id")
            if sid:
                scene_ids.append(sid)

        if len(scene_ids) > 0:
            def format_scene(scene_id):
                for c in scene_results:
                    if as_dict(c).get("id") == scene_id:
                        sd = get_datetime(c)
                        dt = sd.strftime("%Y-%m-%d") if sd else "?"
                        return dt + "  |  " + (get_tile(c) or "?") + "  |  cloud " + format(get_cloud(c), ".1f") + "%"
                return str(scene_id)

            selected_id = st.selectbox("Target scene", scene_ids, format_func=format_scene, key="target_scene_select")
            for s in scene_results:
                if as_dict(s).get("id") == selected_id:
                    st.session_state["target"] = s
                    break

# ══════════════════════════════════════════════════════════════════════
#  ROW 2: Parameters + Process
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
param_col, proc_col = st.columns([1.65, 1.0], gap="small")

with param_col:
    st.subheader("03 · Detection Parameters")
    st.caption("Signal detection")
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        PARAMS["threshold_sigma"] = st.number_input("Threshold (σ)", min_value=0.5, max_value=6.0,
                                                     value=float(PARAMS["threshold_sigma"]), step=0.1, key="thr_sigma")
    with a2:
        PARAMS["gaussian_sigma"] = st.number_input("Gaussian σ", min_value=0.5, max_value=4.0,
                                                    value=float(PARAMS["gaussian_sigma"]), step=0.1, key="g_sigma")
    with a3:
        PARAMS["n_reference_scenes"] = st.number_input("Reference count", min_value=2, max_value=20,
                                                        value=int(PARAMS["n_reference_scenes"]), step=1, key="n_refs")

    st.caption("Noise filtering (lower = keep more signal)")
    b1, b2, b3 = st.columns(3, gap="small")
    with b1:
        PARAMS["spatial_coherence_min"] = st.number_input("Coherence (of 8)", min_value=0, max_value=8,
                                                           value=int(PARAMS["spatial_coherence_min"]), step=1, key="coh_min")
    with b2:
        PARAMS["min_component_pixels"] = st.number_input("Min pixels/region", min_value=1, max_value=500,
                                                          value=int(PARAMS["min_component_pixels"]), step=1, key="min_px")
    with b3:
        PARAMS["min_solidity"] = st.number_input("Min solidity", min_value=0.0, max_value=1.0,
                                                  value=float(PARAMS["min_solidity"]), step=0.05, key="min_sol")

    st.caption(
        "Pipeline: threshold → coherence (≥" + str(PARAMS["spatial_coherence_min"]) + "/8) → "
        "size (≥" + str(PARAMS["min_component_pixels"]) + "px) → solidity (≥" + format(PARAMS["min_solidity"], ".2f") + ")."
    )

with proc_col:
    st.subheader("04 · Process")

    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.success("✓ Copernicus connected · " + cdse_auth.get("username", ""))
        if st.button("Log out", use_container_width=True, key="cdse_logout"):
            st.session_state.pop("cdse_auth", None)
            st.rerun()
    else:
        st.info("Copernicus login")
        st.link_button("🌐 Open Copernicus", "https://dataspace.copernicus.eu/", use_container_width=True)
        with st.form("cdse_login_form", clear_on_submit=True):
            u = st.text_input("Email", placeholder="your-email@example.com")
            p = st.text_input("Password", type="password")
            t = st.text_input("2FA (optional)", max_chars=8)
            sub = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
        if sub:
            try:
                with st.spinner("Connecting…"):
                    st.session_state.cdse_auth = authenticate_cdse(u, p, t)
                st.rerun()
            except Exception as e:
                st.error(str(e))

    scene_results = st.session_state.get("scene_results", [])
    has_target = "target" in st.session_state

    if len(scene_results) > 0 and has_target:
        target = st.session_state.get("target")
        if target is not None:
            tdate = get_datetime(target)
            ttile = get_tile(target) or "?"
            tlabel = tdate.strftime("%Y-%m-%d") if tdate else "?"
            st.caption("Target: " + tlabel + " · " + ttile)

            can_detect = bool(st.session_state.get("cdse_auth"))
            clicked = st.button("🛰️  Download & Detect", type="primary",
                                 use_container_width=True, key="detect_button",
                                 disabled=not can_detect)

            if clicked:
                progress = st.progress(0, text="Preparing…")
                status = st.empty()
                try:
                    status.caption("Step 1/5 · Auth…")
                    progress.progress(5)
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    tdate = get_datetime(target)
                    ttile = get_tile(target)
                    if tdate is None:
                        raise RuntimeError("No valid date.")

                    candidates = []
                    for s in scene_results:
                        if as_dict(s).get("id") == as_dict(target).get("id"):
                            continue
                        sd = get_datetime(s)
                        if sd is None:
                            continue
                        if abs((sd - tdate).total_seconds()) / 86400 <= reference_days:
                            candidates.append(s)

                    same_tile = [s for s in candidates if get_tile(s) == ttile]
                    references = same_tile if len(same_tile) > 0 else candidates
                    if len(references) == 0:
                        st.warning("No references found.")
                        st.stop()

                    status.caption("Step 2/5 · Downloading target…")
                    progress.progress(15)
                    target_bands, profile = read_stack(download_scene(target, st.session_state.aoi, access_token))
                    tq = float(np.nanquantile(target_bands["B03"], PARAMS["b03_quantile"]))
                    target_lrad = calculate_lrad(target_bands, tq)

                    status.caption("Step 3/5 · " + str(len(references)) + " references…")
                    ref_rows = []
                    ref_bands_list = []
                    tot = max(1, len(references))
                    for i, ref in enumerate(references, start=1):
                        pct = 20 + int(45 * i / tot)
                        progress.progress(pct, text="Ref " + str(i) + "/" + str(tot))
                        try:
                            rb, _ = read_stack(download_scene(ref, st.session_state.aoi, access_token))
                        except Exception:
                            continue
                        vpx = (
                            target_lrad
                            & np.isfinite(rb["B11"]) & np.isfinite(rb["B12"])
                            & np.isfinite(target_bands["B11"]) & np.isfinite(target_bands["B12"])
                        )
                        vc = int(vpx.sum())
                        corr = np.nan
                        if vc >= PARAMS["min_valid_ref_pixels"]:
                            t11 = target_bands["B11"][vpx]
                            r11 = rb["B11"][vpx]
                            t12 = target_bands["B12"][vpx]
                            r12 = rb["B12"][vpx]
                            if np.std(r11) > 1e-9 and np.std(r12) > 1e-9:
                                try:
                                    c11 = float(np.corrcoef(t11, r11)[0, 1])
                                    c12 = float(np.corrcoef(t12, r12)[0, 1])
                                    corr = (c11 + c12) / 2.0
                                except Exception:
                                    pass
                        ref_rows.append({
                            "id": as_dict(ref).get("id"),
                            "date": get_datetime(ref),
                            "tile": get_tile(ref),
                            "swir_correlation": corr,
                            "valid_pixels": vc,
                        })
                        if np.isfinite(corr):
                            ref_bands_list.append((corr, rb))

                    st.session_state.reference_table = pd.DataFrame(ref_rows)

                    if len(ref_bands_list) == 0:
                        st.error("No usable references (SWIR correlation failed).")
                        st.stop()

                    ref_bands_list.sort(key=lambda x: -x[0])
                    top_k = ref_bands_list[:PARAMS["n_reference_scenes"]]
                    status.caption("Step 4/5 · Median of top-" + str(len(top_k)) + "…")
                    progress.progress(75)

                    b11s = np.stack([rb["B11"] for _, rb in top_k], axis=0)
                    b12s = np.stack([rb["B12"] for _, rb in top_k], axis=0)
                    b03s = np.stack([rb["B03"] for _, rb in top_k], axis=0)
                    b04s = np.stack([rb["B04"] for _, rb in top_k], axis=0)
                    b08s = np.stack([rb["B08"] for _, rb in top_k], axis=0)

                    median_ref = {
                        "B11": np.nanmedian(b11s, axis=0).astype(np.float32),
                        "B12": np.nanmedian(b12s, axis=0).astype(np.float32),
                        "B03": np.nanmedian(b03s, axis=0).astype(np.float32),
                        "B04": np.nanmedian(b04s, axis=0).astype(np.float32),
                        "B08": np.nanmedian(b08s, axis=0).astype(np.float32),
                    }

                    rq = float(np.nanquantile(median_ref["B03"], PARAMS["b03_quantile"]))
                    ref_lrad = calculate_lrad(median_ref, rq)
                    valid_lrad = target_lrad & ref_lrad

                    if valid_lrad.sum() < 1000:
                        st.warning("Only " + str(int(valid_lrad.sum())) + " valid pixels.")
                        st.stop()

                    status.caption("Step 5/5 · Running detection…")
                    progress.progress(90)

                    result = run_algorithm_multi(target_bands, median_ref, valid_lrad)
                    result["date"] = tdate.strftime("%Y-%m-%d")
                    result["b4_correlation"] = float(top_k[0][0]) if len(top_k) > 0 else np.nan

                    out_folder = RESULT_DIR / tdate.strftime("%Y%m%d")
                    out_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("relative", "gaussian", "initial", "coherent", "final", "valid"):
                        paths[key] = out_folder / (key + ".tif")
                        save_raster(paths[key], result[key], profile, key in ("initial", "coherent", "final", "valid"))

                    st.session_state.result = result
                    st.session_state.paths = paths
                    st.session_state.output_profile = profile
                    st.session_state.png_outputs = {
                        "relative": image_png(result["relative"]),
                        "gaussian": image_png(result["gaussian"]),
                        "initial": image_png(result["initial"], mask=True),
                        "coherent": image_png(result["coherent"], mask=True),
                        "final": image_png(result["final"], mask=True),
                        "valid": image_png(result["valid"], mask=True),
                    }
                    progress.progress(100, text="Done")
                    st.success("Complete · median of " + str(len(top_k)) + " references")

                    if result["final_count"] == 0:
                        st.info(
                            "Final mask is empty. Counts: initial=" + str(result["initial_count"])
                            + ", coherent=" + str(result["coherent_count"])
                            + ", final=0. Lower Coherence or Min pixels/region and re-run."
                        )
                except Exception as e:
                    st.error("Detection failed: " + str(e))
    else:
        st.caption("Select a scene first.")

# ══════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════
if "result" in st.session_state:
    result = st.session_state.result
    png_outputs = st.session_state.get("png_outputs", {})
    profile = st.session_state.get("output_profile")

    st.markdown("---")
    st.subheader("05 · Results")

    metrics = st.columns(6, gap="small")
    ct = format(result["b4_correlation"], ".3f") if np.isfinite(result.get("b4_correlation", np.nan)) else "n/a"
    metrics[0].metric("SWIR corr", ct)
    metrics[1].metric("Valid", format(result["valid_count"], ","))
    metrics[2].metric("Initial", format(result["initial_count"], ","))
    metrics[3].metric("Coherent", format(result.get("coherent_count", 0), ","))
    metrics[4].metric("Final", format(result["final_count"], ","))
    metrics[5].metric("Regions", result["regions"])

    # 3-panel top row (no nested columns)
    result_items = [
        ("relative", "Relative MBMP", "Anomaly"),
        ("gaussian", "Smoothed", "Gaussian"),
        ("initial", "Above threshold", "Stage 1"),
    ]
    row1 = st.columns(3, gap="small")
    for col, (key, title, tag) in zip(row1, result_items):
        col.caption(tag + " · " + title)
        col.image(png_outputs[key], use_container_width=True, output_format="PNG")

    result_items2 = [
        ("coherent", "Coherent only", "Stage 2"),
        ("final", "Final mask", "Stage 3"),
        ("valid", "Valid pixels", "Validity"),
    ]
    row2 = st.columns(3, gap="small")
    for col, (key, title, tag) in zip(row2, result_items2):
        col.caption(tag + " · " + title)
        col.image(png_outputs[key], use_container_width=True, output_format="PNG")

    st.caption(
        "Funnel: " + format(result["initial_count"], ",")
        + " above threshold → " + format(result.get("coherent_count", 0), ",")
        + " coherent → " + format(result["final_count"], ",")
        + " final · " + str(result["regions"]) + " regions."
    )

    dl1, dl2 = st.columns([1, 1], gap="small")
    with dl1:
        st.download_button(
            "⬇ Reference table CSV",
            st.session_state.reference_table.to_csv(index=False),
            file_name="references.csv",
            mime="text/csv",
            key="dl_ref_csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇ Final mask GeoTIFF",
            st.session_state.paths["final"].read_bytes(),
            file_name="final_mask.tif",
            mime="image/tiff",
            key="dl_final_tif",
            use_container_width=True,
        )

    # ── Sentinel-5P ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("05b · Sentinel-5P CH4 Context")
    st.caption("TROPOMI CH4 (~5.5 × 7 km). Colormap = anomaly relative to local mean.")

    s5p_c1, s5p_c2 = st.columns([1, 3], gap="small")
    with s5p_c1:
        run_s5p = st.button(
            "🛰️  Fetch S5P CH4",
            type="primary",
            use_container_width=True,
            key="s5p_btn",
            disabled=not bool(st.session_state.get("cdse_auth")),
        )
    with s5p_c2:
        s5p_days = st.slider("S5P window (days)", 1, 30, 15, key="s5p_days")

    if run_s5p:
        try:
            access_token = get_access_token()
            tdate = get_datetime(st.session_state.get("target"))
            if tdate is None:
                raise RuntimeError("No target date.")
            with st.spinner("Fetching S5P…"):
                s5p_path = download_s5p_scene(
                    st.session_state.aoi,
                    tdate - timedelta(days=int(s5p_days)),
                    tdate + timedelta(days=int(s5p_days)),
                    access_token,
                )
            with rasterio.open(s5p_path) as src:
                ch4 = src.read(1).astype(np.float32)
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[ch4 <= 0] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.success("S5P loaded")
        except Exception as e:
            st.error("S5P failed: " + str(e))

    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        v = ch4[np.isfinite(ch4)]
        if v.size < 2:
            ph = float(np.nanmean(v)) if v.size > 0 else 1900.0
            ch4 = np.full_like(ch4, ph)
            v = ch4[np.isfinite(ch4)]
            st.warning("No valid S5P pixels. Placeholder shown.")
        if v.size > 1:
            c1, c2, c3, c4 = st.columns(4, gap="small")
            c1.metric("Mean (ppb)", format(float(np.nanmean(v)), ".1f"))
            c2.metric("Min", format(float(np.nanmin(v)), ".1f"))
            c3.metric("Max", format(float(np.nanmax(v)), ".1f"))
            c4.metric("Range", format(float(np.nanmax(v) - np.nanmin(v)), ".1f"))
            st.image(ch4_anomaly_png(ch4), use_container_width=True, output_format="PNG")
