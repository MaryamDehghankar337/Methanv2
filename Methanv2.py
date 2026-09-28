"""Sentinel-2 methane screening — multi-reference median version.

KEY CHANGE: Instead of comparing to ONE reference, this builds a MEDIAN
reference from the top-K best-correlated scenes (using B11+B12 correlation),
then applies a strict spatial-coherence filter.
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
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    gaussian_filter,
    label,
    convolve,
)
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
    # ── Multi-reference + spatial coherence ──────────────────────────
    "n_reference_scenes": 6,       # use median of top-6 best scenes
    "gaussian_sigma": 2.0,         # stronger smoothing
    "threshold_sigma": 2.5,
    "spatial_coherence_min": 7,    # require 7 of 8 neighbors above threshold
    "min_component_pixels": 40,
    "final_dilation": 2,
    "min_solidity": 0.55,
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
    cache_id = hashlib.sha256(
        json.dumps(["s5p_v4", aoi, str(date_from), str(date_to)], sort_keys=True).encode()
    ).hexdigest()[:24]
    folder = CACHE_DIR / f"s5p_{cache_id}"
    output_path = folder / "ch4.tif"
    if output_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": S5P_COLLECTION,
                "dataFilter": {
                    "timeRange": {
                        "from": date_from.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": date_to.strftime("%Y-%m-%dT23:59:59Z"),
                    },
                },
            }],
        },
        "output": {
            "width": 256,
            "height": 256,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": s5p_evalscript(),
    }
    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
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
    artifact |= bands["B03"] <= q_value
    artifact |= normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"]
    artifact |= normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"]
    artifact |= normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"]
    artifact |= normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"]
    artifact |= ~finite
    artifact = binary_dilation(artifact, structure=disk(PARAMS["lrad_dilation"]))
    return finite & ~artifact


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
    """
    Run MBMP using a MEDIAN reference (dict with B11, B12).
    Applies spatial-coherence filter to kill scattered noise.
    """
    # Compute C from target only
    c_target = calculate_c(target["B11"], target["B12"], valid_lrad)

    target_mbsp = calculate_mbsp(target["B11"], target["B12"], c_target, valid_lrad)
    reference_mbsp = calculate_mbsp(reference_median["B11"], reference_median["B12"], c_target, valid_lrad)

    relative = target_mbsp - reference_mbsp
    relative[~valid_lrad] = np.nan
    finite = np.isfinite(relative)
    if not finite.any():
        raise RuntimeError("LRAD removed all pixels. Reduce artifact thresholds or use a smaller valid AOI.")

    values = relative[finite].astype(np.float64)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))

    # Strong gaussian smoothing
    gaussian = gaussian_filter(np.where(finite, relative, mean_value), sigma=PARAMS["gaussian_sigma"])
    threshold = mean_value + PARAMS["threshold_sigma"] * std_value

    # Initial: pixel above threshold
    above = valid_lrad & np.isfinite(gaussian) & (gaussian > threshold)

    # ── SPATIAL COHERENCE FILTER ─────────────────────────────────────
    # Require at least N of 8 neighbors to also be above threshold
    neighbor_count = convolve(
        above.astype(np.float32),
        np.ones((3, 3), dtype=np.float32),
        mode="constant",
    ) - above.astype(np.float32)  # exclude self

    coherent = above & (neighbor_count >= PARAMS["spatial_coherence_min"])

    # Connected-component + solidity filter
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

    # Morphological opening with disk(2) → kills thin lines
    if connected.any():
        connected = binary_opening(connected, structure=disk(2))

    # Final dilation
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
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[data.astype(bool)] = [220, 30, 30]
    else:
        finite = np.isfinite(data)
        rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
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
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
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
        rows = [("#dc1e1e", "Methane candidate"), ("#000000", "Background / non-candidate")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid pixels"), ("#000000", "Invalid / masked pixels")]
    elif kind == "s5p":
        rows = [("#b43232", "CH4 above local mean"), ("#3250b4", "CH4 below local mean"), ("#ffffff", "No data")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>' for c, t in rows)
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def create_png_worldfile(profile, array_shape):
    transform = profile["transform"]
    xres = transform.a
    yres = transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    pgw = f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
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
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --dark-field: #292a33;
}
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-bottom: 0.9rem; }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp li, .stApp td, .stApp th { color: #111111 !important; }
div[data-testid="stDateInput"] input, .stDateInput input, div[data-testid="stNumberInput"] input, .stNumberInput input { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
input::placeholder, textarea::placeholder { color: #bfc3cc !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] [role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="listbox"] *, div[data-baseweb="popover"] [role="option"] * { color: #ffffff !important; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.78rem; color: #111111 !important; }
.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }
.stButton > button[kind="primary"] * { color: #ffffff !important; }
.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }
.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; margin: 0.2rem 0 0.45rem 0; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.82rem; }
.legend-swatch { width: 18px; height: 14px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
        <div class="app-subtitle">Multi-reference median + spatial coherence detection</div>
    </div>
    <div class="status-pill">20 m processing</div>
</div>
""", unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)

map_col, control_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Draw or edit the study area directly on the map.</div>', unsafe_allow_html=True)
    map_data = st_folium(create_map(st.session_state.aoi), height=385, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi:
            st.session_state.aoi = new_aoi
    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)
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

    if st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC..."):
                search_results = search_scenes(
                    st.session_state.aoi,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    max_cloud,
                )
            if search_results is None:
                search_results = []
            elif not isinstance(search_results, list):
                search_results = list(search_results)
            st.session_state["scene_results"] = search_results
            st.session_state.pop("target", None)
            if search_results:
                st.success(f"{len(search_results)} scene(s) found")
            else:
                st.warning("No Sentinel-2 scenes were found for the selected criteria.")
        except Exception as error:
            st.session_state["scene_results"] = []
            st.session_state.pop("target", None)
            st.error(f"Scene search failed: {error}")

    scene_results = st.session_state.get("scene_results", [])

    if scene_results:
        scene_table = pd.DataFrame([
            {"date": get_datetime(scene), "tile": get_tile(scene), "cloud": get_cloud(scene)}
            for scene in scene_results
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")

        st.dataframe(
            scene_table, use_container_width=True, height=112, hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        scene_ids = [as_dict(scene).get("id") for scene in scene_results if as_dict(scene).get("id")]

        def format_scene(scene_id):
            scene = next((c for c in scene_results if as_dict(c).get("id") == scene_id), None)
            if scene is None:
                return str(scene_id)
            sd = get_datetime(scene)
            dt = sd.strftime("%Y-%m-%d") if sd else "Unknown"
            return f"{dt}  |  {get_tile(scene) or 'Unknown tile'}  |  cloud {get_cloud(scene):.1f}%"

        if scene_ids:
            selected_scene_id = st.selectbox("Target scene", scene_ids, format_func=format_scene, key="target_scene_select")
            selected_scene = next((s for s in scene_results if as_dict(s).get("id") == selected_scene_id), None)
            if selected_scene is not None:
                st.session_state["target"] = selected_scene
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["threshold_sigma"] = st.number_input("Threshold multiplier", min_value=0.1, max_value=6.0, value=float(PARAMS["threshold_sigma"]), step=0.1, key="threshold_sigma")
    with p2:
        PARAMS["min_component_pixels"] = st.number_input("Minimum candidate pixels", min_value=2, max_value=1000, value=int(PARAMS["min_component_pixels"]), step=5, key="min_component_pixels")
    with p3:
        PARAMS["n_reference_scenes"] = st.number_input("Reference scenes (median)", min_value=2, max_value=20, value=int(PARAMS["n_reference_scenes"]), step=1, key="n_refs")
    st.markdown(f'<div class="card-caption">Median of top-{PARAMS["n_reference_scenes"]} scenes by SWIR correlation · spatial coherence: ≥{PARAMS["spatial_coherence_min"]}/8 neighbors.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    scene_results = st.session_state.get("scene_results", [])
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username", "")}</div>', unsafe_allow_html=True)
        if st.button("Log out", use_container_width=True, key="cdse_logout"):
            st.session_state.pop("cdse_auth", None)
            st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8)
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login_submitted:
            try:
                with st.spinner("Connecting…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.rerun()
            except Exception as e:
                st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)

    if scene_results and "target" in st.session_state:
        target = st.session_state.get("target")
        if target is not None:
            target_date = get_datetime(target)
            target_tile = get_tile(target) or "Unknown"
            target_label = target_date.strftime("%Y-%m-%d") if target_date else "Unknown"
            st.markdown(f'<div class="card-title">Ready to detect</div><div class="card-caption">Target: {target_label} · {target_tile}</div>', unsafe_allow_html=True)
            detect_clicked = st.button("🛰️  Download AOI & Detect", type="primary", use_container_width=True, key="detect_button", disabled=not bool(st.session_state.get("cdse_auth")))
            if detect_clicked:
                progress = st.progress(0, text="Preparing…")
                status = st.empty()
                try:
                    status.markdown('<div class="card-caption">Step 1/5 · Auth + target setup…</div>', unsafe_allow_html=True)
                    progress.progress(5)
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    target_date = get_datetime(target)
                    target_tile = get_tile(target)
                    if target_date is None:
                        raise RuntimeError("Target has no valid date.")

                    candidates = [
                        s for s in scene_results
                        if as_dict(s).get("id") != as_dict(target).get("id")
                        and get_datetime(s) is not None
                        and abs((get_datetime(s) - target_date).total_seconds()) / 86400 <= reference_days
                    ]
                    same_tile = [s for s in candidates if get_tile(s) == target_tile]
                    references = same_tile if same_tile else candidates
                    if not references:
                        st.warning("No reference scenes. Widen the date range.")
                        st.stop()

                    status.markdown('<div class="card-caption">Step 2/5 · Downloading target bands…</div>', unsafe_allow_html=True)
                    progress.progress(15)
                    target_bands, profile = read_stack(download_scene(target, st.session_state.aoi, access_token))
                    target_q = float(np.nanquantile(target_bands["B03"], PARAMS["b03_quantile"]))
                    target_lrad = calculate_lrad(target_bands, target_q)

                    # ── Download ALL references, score by B11+B12 correlation
                    status.markdown(f'<div class="card-caption">Step 3/5 · Downloading {len(references)} reference scenes…</div>', unsafe_allow_html=True)
                    ref_rows = []
                    ref_bands_list = []
                    total_refs = max(1, len(references))

                    for i, ref in enumerate(references, start=1):
                        pct = 20 + int(45 * i / total_refs)
                        progress.progress(pct, text=f"Reference {i}/{total_refs}…")
                        try:
                            rb, _ = read_stack(download_scene(ref, st.session_state.aoi, access_token))
                        except Exception:
                            continue
                        valid_px = target_lrad & np.isfinite(rb["B11"]) & np.isfinite(rb["B12"]) & np.isfinite(target_bands["B11"]) & np.isfinite(target_bands["B12"])
                        vc = int(valid_px.sum())
                        corr = np.nan
                        if vc >= PARAMS["min_valid_ref_pixels"]:
                            t11 = target_bands["B11"][valid_px]; r11 = rb["B11"][valid_px]
                            t12 = target_bands["B12"][valid_px]; r12 = rb["B12"][valid_px]
                            if np.std(r11) > 1e-9 and np.std(r12) > 1e-9:
                                try:
                                    c11 = float(np.corrcoef(t11, r11)[0, 1])
                                    c12 = float(np.corrcoef(t12, r12)[0, 1])
                                    corr = (c11 + c12) / 2.0
                                except Exception:
                                    pass
                        ref_rows.append({
                            "id": as_dict(ref).get("id"), "date": get_datetime(ref),
                            "tile": get_tile(ref), "swir_correlation": corr, "valid_pixels": vc,
                        })
                        if np.isfinite(corr):
                            ref_bands_list.append((corr, rb))

                    st.session_state.reference_table = pd.DataFrame(ref_rows)

                    if not ref_bands_list:
                        st.error("No usable reference scene (SWIR correlation could not be computed).")
                        st.stop()

                    # ── Sort by correlation, take top-K, build median stack
                    ref_bands_list.sort(key=lambda x: -x[0])
                    top_k = ref_bands_list[:PARAMS["n_reference_scenes"]]
                    status.markdown(f'<div class="card-caption">Step 4/5 · Building median reference from top-{len(top_k)} scenes…</div>', unsafe_allow_html=True)
                    progress.progress(75)

                    b11_stack = np.stack([rb["B11"] for _, rb in top_k], axis=0)
                    b12_stack = np.stack([rb["B12"] for _, rb in top_k], axis=0)
                    b03_stack = np.stack([rb["B03"] for _, rb in top_k], axis=0)
                    b04_stack = np.stack([rb["B04"] for _, rb in top_k], axis=0)
                    b08_stack = np.stack([rb["B08"] for _, rb in top_k], axis=0)

                    median_ref = {
                        "B11": np.nanmedian(b11_stack, axis=0).astype(np.float32),
                        "B12": np.nanmedian(b12_stack, axis=0).astype(np.float32),
                        "B03": np.nanmedian(b03_stack, axis=0).astype(np.float32),
                        "B04": np.nanmedian(b04_stack, axis=0).astype(np.float32),
                        "B08": np.nanmedian(b08_stack, axis=0).astype(np.float32),
                    }

                    # Common LRAD: target AND median reference
                    ref_q = float(np.nanquantile(median_ref["B03"], PARAMS["b03_quantile"]))
                    ref_lrad = calculate_lrad(median_ref, ref_q)
                    valid_lrad = target_lrad & ref_lrad

                    if valid_lrad.sum() < 1000:
                        st.warning(f"Only {int(valid_lrad.sum()):,} valid pixels after LRAD. Consider a larger AOI.")
                        st.stop()

                    status.markdown('<div class="card-caption">Step 5/5 · Running MBMP + coherence filter…</div>', unsafe_allow_html=True)
                    progress.progress(90)

                    result = run_algorithm_multi(target_bands, median_ref, valid_lrad)
                    result["date"] = target_date.strftime("%Y-%m-%d")
                    # Store best correlation for display
                    result["b4_correlation"] = float(top_k[0][0]) if top_k else np.nan

                    output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                    output_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("relative", "gaussian", "final", "valid"):
                        paths[key] = output_folder / f"{key}.tif"
                        save_raster(paths[key], result[key], profile, key in ("final", "valid"))
                    st.session_state.result = result
                    st.session_state.paths = paths
                    st.session_state.output_profile = profile
                    st.session_state.png_outputs = {
                        "relative": image_png(result["relative"]),
                        "gaussian": image_png(result["gaussian"]),
                        "final": image_png(result["final"], mask=True),
                        "valid": image_png(result["valid"], mask=True),
                    }
                    progress.progress(100, text="Done")
                    st.success(f"Processing complete · median of {len(top_k)} references")

                    # Sanity check
                    ratio = result["final_count"] / max(1, result["valid_count"])
                    if ratio < 0.0005:
                        st.warning(
                            f"⚠️ Signal ratio = {ratio*100:.4f}% ({result['final_count']:,} / {result['valid_count']:,}). "
                            f"This is likely **background noise**, not a real plume. "
                            f"Try `Threshold = 4.0` or a different target/reference pair."
                        )
                except Exception as error:
                    st.error(f"Detection failed: {error}")
    else:
        st.markdown('<div class="card-title">Select scenes first</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════

if "result" in st.session_state:
    result = st.session_state.result
    png_outputs = st.session_state.get("png_outputs", {})
    profile = st.session_state.get("output_profile")
    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)
    metrics = st.columns(6, gap="small")
    corr_text = f"{result['b4_correlation']:.3f}" if np.isfinite(result.get("b4_correlation", np.nan)) else "n/a"
    metrics[0].metric("SWIR correlation", corr_text)
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Initial", f"{result['initial_count']:,}")
    metrics[3].metric("Coherent", f"{result.get('coherent_count', 0):,}")
    metrics[4].metric("Final", f"{result['final_count']:,}")
    metrics[5].metric("Regions", result["regions"])
    result_items = [("relative", "Relative MBMP", "Anomaly"), ("gaussian", "Gaussian filtered", "Smoothed"), ("final", "Methane candidates", "Final mask"), ("valid", "Valid pixels", "Validity mask")]
    result_cols = st.columns(4, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown(f'<div class="card-caption" style="font-weight:700;">{tag} · {title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(png_outputs[key], use_container_width=True, output_format="PNG")
            with legend_col:
                st.markdown(legend_html("mask" if key == "final" else "valid" if key == "valid" else "continuous"), unsafe_allow_html=True)
            path = st.session_state.paths[key]
            fmt = st.selectbox("Download format", ["GeoTIFF", "PNG + World File"], key=f"fmt_{key}")
            if fmt == "GeoTIFF":
                st.download_button("⬇ GeoTIFF", path.read_bytes(), file_name=path.name, mime="image/tiff", key=f"dl_tif_{key}", use_container_width=True)
            else:
                pkg = georeferenced_png_package(result[key], profile, mask=key in ("final", "valid"))
                st.download_button("⬇ PNG package", pkg, file_name=f"{key}_png.zip", mime="application/zip", key=f"dl_png_{key}", use_container_width=True)
    st.markdown(f'<div class="card-caption">Initial (pixel &gt; threshold): <b>{result["initial_count"]:,}</b> → Coherent (≥{PARAMS["spatial_coherence_min"]}/8 neighbors): <b>{result.get("coherent_count", 0):,}</b> → After size/shape filter: <b>{result["final_count"]:,}</b></div>', unsafe_allow_html=True)
    st.download_button("⬇ Reference table CSV", st.session_state.reference_table.to_csv(index=False), file_name="references.csv", mime="text/csv", key="dl_ref_csv", use_container_width=False)
    st.markdown('<div class="card-caption">⚠️ This is a screening tool. Candidate masks are NOT physical methane concentration.</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # 05b · SENTINEL-5P CH4
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05b · SENTINEL-5P CH4 CONTEXT</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">TROPOMI CH4 (~5.5 × 7 km). All pixels shown. Colormap = anomaly relative to local mean.</div>', unsafe_allow_html=True)

    s5p_c1, s5p_c2 = st.columns([1, 3], gap="small")
    with s5p_c1:
        run_s5p = st.button("🛰️  Fetch S5P CH4", type="primary", use_container_width=True, key="s5p_btn", disabled=not bool(st.session_state.get("cdse_auth")))
    with s5p_c2:
        s5p_days = st.slider("S5P temporal window (days)", 1, 30, 15, key="s5p_days")

    if run_s5p:
        try:
            access_token = get_access_token()
            target_date = get_datetime(st.session_state.get("target"))
            if target_date is None:
                raise RuntimeError("Target date missing.")
            with st.spinner("Fetching S5P…"):
                s5p_path = download_s5p_scene(
                    st.session_state.aoi,
                    target_date - timedelta(days=int(s5p_days)),
                    target_date + timedelta(days=int(s5p_days)),
                    access_token,
                )
            with rasterio.open(s5p_path) as src:
                ch4 = src.read(1).astype(np.float32)
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[ch4 <= 0] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.success("S5P loaded")
        except Exception as e:
            st.error(f"S5P failed: {e}")

    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        v = ch4[np.isfinite(ch4)]
        if v.size < 2:
            placeholder = float(np.nanmean(v)) if v.size > 0 else 1900.0
            ch4 = np.full_like(ch4, placeholder)
            v = ch4[np.isfinite(ch4)]
            st.warning("No valid S5P pixels. Showing placeholder at AOI center.")
        if v.size > 1:
            c1, c2, c3, c4 = st.columns(4, gap="small")
            c1.metric("Mean CH4 (ppb)", f"{float(np.nanmean(v)):.1f}")
            c2.metric("Min", f"{float(np.nanmin(v)):.1f}")
            c3.metric("Max", f"{float(np.nanmax(v)):.1f}")
            c4.metric("Range", f"{float(np.nanmax(v) - np.nanmin(v)):.1f}")
            ic, lc = st.columns([3.6, 1.0], gap="small")
            with ic:
                st.image(ch4_anomaly_png(ch4), use_container_width=True, output_format="PNG")
            with lc:
                st.markdown(legend_html("s5p"), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
