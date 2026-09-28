"""Sentinel-2 methane screening app - corrected version.

Main corrections:
- Requests SCL, CLD and dataMask quality layers.
- Masks cloud, cloud shadow, cirrus, snow and no-data pixels.
- Uses AOI-only statistics.
- Selects references using B04, B11 and B12 similarity.
- Uses robust median/MAD thresholding.
- Uses normalized Gaussian filtering without filling invalid pixels by the mean.
- Does not delete small candidates before displaying the raw candidate mask.
- Keeps the final minimum-area filter configurable and conservative.
- Supports dates outside the current search window for reference scenes.

This remains a screening workflow. A red candidate is not proof of methane.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from folium.plugins import Draw
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
RESOLUTION = 20
DATA_BANDS = ["B03", "B04", "B08", "B11", "B12"]
OUTPUT_BANDS = DATA_BANDS + ["SCL", "CLD", "DATA_MASK"]
CACHE_DIR = Path.home() / ".sentinel_methane_cache_v2"
RESULT_DIR = Path.home() / ".sentinel_methane_results_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "cloud_probability": 35.0,
    "threshold_sigma": 2.5,
    "min_component_pixels": 3,
    "final_dilation": 0,
    "gaussian_sigma": 0.85,
    "lrad_dilation": 1,
    "min_reference_correlation": 0.70,
    "max_reference_count": 8,
}


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
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    match = re.search(r"_(\d{8}T\d{6})_", str(data.get("id", "")).upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data = as_dict(item)
    props = get_properties(data)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key):
            return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", str(data.get("id", "")).upper())
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
        if not geometries:
            return None
        merged = unary_union(geometries)
        return mapping(merged) if not merged.is_empty else None
    try:
        geometry = shape(obj)
        return mapping(geometry) if not geometry.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(box(51.25, 35.65, 51.45, 35.80))


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
    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    if totp.strip():
        form["totp"] = totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")
    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            data = response.json()
            detail = data.get("error_description") or data.get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Copernicus did not return an access token.")
    return {
        "access_token": token,
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": time.time() + int(data.get("expires_in", 600)),
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
    input: [{
      bands: ["B03", "B04", "B08", "B11", "B12", "SCL", "CLD", "dataMask"],
      units: "REFLECTANCE"
    }],
    output: {bands: 8, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [
    sample.B03,
    sample.B04,
    sample.B08,
    sample.B11,
    sample.B12,
    sample.SCL,
    sample.CLD,
    sample.dataMask
  ];
}
"""


def download_scene(item, aoi, access_token, cloud_probability=35.0):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")

    cache_key = hashlib.sha256(
        json.dumps(
            [item.get("id"), aoi, RESOLUTION, cloud_probability],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    folder = CACHE_DIR / cache_key
    output_path = folder / "bands_quality.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        return output_path

    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))

    start = acquisition.strftime("%Y-%m-%dT00:00:00Z")
    end = (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": start, "to": end},
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": float(cloud_probability),
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"},
            }],
        },
        "evalscript": evalscript(),
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=900,
    )
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")

    output_path.write_bytes(response.content)
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    if array.shape[0] < len(OUTPUT_BANDS):
        raise RuntimeError(
            f"Expected {len(OUTPUT_BANDS)} output bands, received {array.shape[0]}."
        )
    bands = {name: array[index] for index, name in enumerate(OUTPUT_BANDS)}
    return bands, profile


def normalized_difference(first, second):
    result = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    use = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-8)
    result[use] = (first[use] - second[use]) / denominator[use]
    return result


def quality_mask(bands, cloud_probability=35.0):
    scl = np.rint(bands["SCL"]).astype(np.int16)
    cld = bands["CLD"]
    data_mask = bands["DATA_MASK"] >= 0.5

    bad_scl = np.isin(scl, [0, 1, 3, 7, 8, 9, 10, 11])
    bad_cloud_probability = np.isfinite(cld) & (cld > float(cloud_probability))
    finite = np.logical_and.reduce([
        np.isfinite(bands[name]) for name in DATA_BANDS
    ])
    positive = (bands["B11"] > 0) & (bands["B12"] > 0)

    valid = data_mask & finite & positive & ~bad_scl & ~bad_cloud_probability
    return valid


def calculate_lrad(bands, valid, dilation_radius=1):
    artifact = ~valid

    water_like = normalized_difference(
        bands["B03"], bands["B08"]
    ) > 0.35

    snow_like = normalized_difference(
        bands["B03"], bands["B11"]
    ) > 0.55

    artifact |= water_like | snow_like

    if dilation_radius > 0:
        artifact = binary_dilation(
            artifact,
            structure=disk(int(dilation_radius)),
        )

    return valid & ~artifact


def calculate_c(b11, b12, valid):
    use = valid & np.isfinite(b11) & np.isfinite(b12)
    use &= (b11 > 0) & (b12 > 0)
    if use.sum() < 100:
        return 1.0
    x = b11[use].astype(np.float64)
    y = b12[use].astype(np.float64)
    denominator = max(float(np.sum(x * x)), 1e-20)
    return float(np.sum(x * y) / denominator)


def calculate_mbmp(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12)
    use &= (b11 > 0) & (b12 > 0)
    output[use] = c * np.log(b12[use] / b11[use])
    return output


def robust_quantile(array, valid, q):
    values = array[valid & np.isfinite(array)]
    if values.size == 0:
        return 0.0
    return float(np.nanquantile(values, q))


def robust_threshold(values, sigma):
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite anomaly values were available.")
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1e-8
    return center, scale, center + float(sigma) * scale


def normalized_gaussian(array, valid, sigma):
    numerator = gaussian_filter(
        np.where(valid, np.nan_to_num(array, nan=0.0), 0.0).astype(np.float32),
        sigma=float(sigma),
    )
    denominator = gaussian_filter(
        valid.astype(np.float32),
        sigma=float(sigma),
    )
    result = np.full(array.shape, np.nan, dtype=np.float32)
    use = denominator > 0.25
    result[use] = numerator[use] / denominator[use]
    return result


def run_algorithm(target, reference, params):
    target_quality = quality_mask(target, params["cloud_probability"])
    reference_quality = quality_mask(reference, params["cloud_probability"])
    valid = target_quality & reference_quality

    target_q = robust_quantile(target["B03"], valid, 0.02)
    reference_q = robust_quantile(reference["B03"], valid, 0.02)

    valid = calculate_lrad(
        target,
        valid & (target["B03"] > target_q),
        params["lrad_dilation"],
    )
    valid &= calculate_lrad(
        reference,
        valid & (reference["B03"] > reference_q),
        params["lrad_dilation"],
    )

    if valid.sum() < 100:
        raise RuntimeError(f"Too few valid pixels after masking: {int(valid.sum())}")

    c_target = calculate_c(target["B11"], target["B12"], valid)
    c_reference = calculate_c(reference["B11"], reference["B12"], valid)

    target_mbmp = calculate_mbmp(
        target["B11"], target["B12"], c_target, valid
    )
    reference_mbmp = calculate_mbmp(
        reference["B11"], reference["B12"], c_reference, valid
    )

    relative = target_mbmp - reference_mbmp
    relative[~valid] = np.nan
    finite = np.isfinite(relative)

    center, scale, threshold = robust_threshold(
        relative[finite],
        params["threshold_sigma"],
    )

    smoothed = normalized_gaussian(
        relative,
        finite,
        params["gaussian_sigma"],
    )

    initial = valid & np.isfinite(smoothed) & (smoothed > threshold)

    labels, _ = label(
        initial,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= int(params["min_component_pixels"]))[0]
    retained = retained[retained != 0]
    connected = np.isin(labels, retained)

    final = connected & valid
    dilation = int(params["final_dilation"])
    if dilation > 0:
        final = binary_dilation(
            final,
            structure=disk(dilation),
        ) & valid

    return {
        "relative": relative,
        "gaussian": smoothed,
        "valid": valid,
        "initial": initial,
        "connected": connected,
        "final": final,
        "mean": center,
        "std": scale,
        "threshold": threshold,
        "regions": int(len(retained)),
        "valid_count": int(valid.sum()),
        "initial_count": int(initial.sum()),
        "final_count": int(final.sum()),
    }


def robust_corr(a, b, valid):
    use = valid & np.isfinite(a) & np.isfinite(b)
    if use.sum() < 100:
        return np.nan
    x = a[use].astype(np.float64)
    y = b[use].astype(np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def reference_score(target, reference, cloud_probability):
    target_valid = quality_mask(target, cloud_probability)
    reference_valid = quality_mask(reference, cloud_probability)
    valid = target_valid & reference_valid
    if valid.sum() < 100:
        return np.nan, np.nan, np.nan, np.nan, int(valid.sum())

    c_b4 = robust_corr(target["B04"], reference["B04"], valid)
    c_b11 = robust_corr(target["B11"], reference["B11"], valid)
    c_b12 = robust_corr(target["B12"], reference["B12"], valid)
    score = np.nanmean([c_b4, c_b11, c_b12])
    return score, c_b4, c_b11, c_b12, int(valid.sum())


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8" if mask else "float32",
        nodata=255 if mask else -9999,
        compress="deflate",
        tiled=False,
        BIGTIFF="IF_SAFER",
    )
    if mask:
        output = np.where(np.asarray(array).astype(bool), 1, 0).astype(np.uint8)
    else:
        output = np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map(
        [geometry.centroid.y, geometry.centroid.x],
        zoom_start=11,
        tiles="OpenStreetMap",
    )
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False},
    ).add_to(fmap)
    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(fmap)
    return fmap


def image_png(array, mask=False):
    data = np.asarray(array)
    if mask:
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[data.astype(bool)] = [220, 30, 30]
    else:
        finite = np.isfinite(data)
        rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
        if finite.any():
            values = data[finite]
            low = float(np.percentile(values, 2))
            high = float(np.percentile(values, 98))
            if high <= low:
                low, high = float(values.min()), float(values.max())
            if high > low:
                normalized = np.clip(
                    (np.nan_to_num(data, nan=low) - low) / (high - low),
                    0,
                    1,
                )
                rgb = (
                    plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255
                ).astype(np.uint8)
                rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def create_png_worldfile(profile):
    transform = profile["transform"]
    xres = transform.a
    yres = transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    worldfile = (
        f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n"
        f"{x_center:.12f}\n{y_center:.12f}\n"
    )
    crs = profile.get("crs")
    return worldfile.encode(), crs.to_wkt().encode() if crs else b""


def georeferenced_png_package(array, profile, mask=False):
    package = io.BytesIO()
    png_data = image_png(array, mask=mask)
    worldfile, projection = create_png_worldfile(profile)
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", png_data)
        archive.writestr("output.pgw", worldfile)
        if projection:
            archive.writestr("output.prj", projection)
    return package.getvalue()


def format_scene(scene_id, scenes):
    scene = next(
        (item for item in scenes if as_dict(item).get("id") == scene_id),
        None,
    )
    if scene is None:
        return str(scene_id)
    dt = get_datetime(scene)
    date_text = dt.strftime("%Y-%m-%d") if dt else "Unknown date"
    return f"{date_text} | {get_tile(scene) or 'Unknown tile'} | cloud {get_cloud(scene):.1f}%"


def collect_reference_scenes(target, aoi, target_date, reference_days, max_cloud):
    start = target_date - timedelta(days=int(reference_days))
    end = target_date + timedelta(days=int(reference_days))
    candidates = search_scenes(aoi, start, end, max_cloud)
    target_id = as_dict(target).get("id")
    target_tile = get_tile(target)
    return [
        scene for scene in candidates
        if as_dict(scene).get("id") != target_id
        and get_datetime(scene) is not None
        and get_tile(scene) == target_tile
    ]


st.set_page_config(
    page_title="Sentinel-2 Methane Screening",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root { --bg:#f1faee; --red:#e63946; --blue:#457b9d; --dark:#292a33; --border:#d8e6e8; }
.stApp { background:var(--bg); color:#111 !important; }
[data-testid="stHeader"] { background:var(--bg) !important; }
[data-testid="stSidebar"] { display:none; }
.block-container { max-width:1700px; padding-top:3.9rem !important; padding-left:1.2rem; padding-right:1.2rem; }
.app-header,.app-card { background:#fff; border:1px solid var(--border); border-radius:15px; box-shadow:0 2px 10px rgba(29,53,87,.05); }
.app-header { display:flex; justify-content:space-between; align-items:center; padding:.75rem 1rem; margin-bottom:.9rem; }
.app-title { font-size:1.45rem; font-weight:850; }
.app-subtitle,.card-caption { font-size:.75rem; }
.app-card { padding:.75rem; height:100%; }
.card-title { font-size:1rem; font-weight:800; }
.section-label { display:inline-block; background:#a8dadc; border-radius:999px; padding:.2rem .55rem; font-size:.65rem; font-weight:800; margin-bottom:.35rem; }
.stApp p,.stApp label,.stApp span,.stApp td,.stApp th,.stApp li { color:#111 !important; }
.stButton > button,.stDownloadButton > button { border-radius:9px; min-height:2.15rem; font-weight:750; }
.stButton > button[kind="primary"] { background:var(--red); border-color:var(--red); color:#fff !important; }
.stButton > button[kind="primary"] * { color:#fff !important; }
.stDownloadButton > button { background:#fff; border:1px solid #a8dadc; color:#111 !important; }
.result-legend { background:#fff; border:1px solid #d7e4e7; border-radius:10px; padding:.7rem; min-height:100px; }
.legend-row { display:flex; gap:.4rem; align-items:center; font-size:.78rem; margin-top:.35rem; }
.legend-swatch { width:18px; height:14px; border:1px solid #555; display:inline-block; border-radius:2px; }
footer { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-header">
  <div>
    <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
    <div class="app-subtitle">Quality-masked SWIR anomaly screening at 20 m</div>
  </div>
  <div>20 m processing • v2</div>
</div>
""",
    unsafe_allow_html=True,
)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(box(51.25, 35.65, 51.45, 35.80))

map_col, search_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Draw the landfill or study area on the map.</div>', unsafe_allow_html=True)
    map_data = st_folium(
        create_map(st.session_state.aoi),
        height=385,
        width=1000,
        key="aoi_map",
    )
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({
            "type": "FeatureCollection",
            "features": map_data["all_drawings"],
        })
        if new_aoi:
            st.session_state.aoi = new_aoi
    st.markdown('</div>', unsafe_allow_html=True)

with search_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        start_date = st.date_input(
            "Target search start",
            date(2026, 7, 1),
            key="search_start",
        )
    with right:
        end_date = st.date_input(
            "Target search end",
            date.today(),
            key="search_end",
        )
    left, right = st.columns(2, gap="small")
    with left:
        max_cloud = st.slider(
            "Scene cloud cover (%)",
            0.0,
            100.0,
            50.0,
            key="scene_cloud",
        )
    with right:
        reference_days = st.slider(
            "Reference window (days)",
            7,
            180,
            60,
            key="reference_days",
        )
    if st.button("🔎 Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            scenes = search_scenes(
                st.session_state.aoi,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
                max_cloud,
            )
            st.session_state.scene_results = scenes or []
            st.session_state.pop("target", None)
            st.session_state.pop("result", None)
            st.success(f"{len(st.session_state.scene_results)} scene(s) found")
        except Exception as error:
            st.session_state.scene_results = []
            st.error(f"Scene search failed: {error}")

    scenes = st.session_state.get("scene_results", [])
    if scenes:
        table = pd.DataFrame([
            {
                "date": get_datetime(scene),
                "tile": get_tile(scene),
                "cloud": get_cloud(scene),
            }
            for scene in scenes
        ]).sort_values(["date", "cloud"], na_position="last")
        st.dataframe(table, use_container_width=True, height=112, hide_index=True)
        ids = [as_dict(scene).get("id") for scene in scenes if as_dict(scene).get("id")]
        selected_id = st.selectbox(
            "Target scene",
            ids,
            format_func=lambda value: format_scene(value, scenes),
            key="target_scene",
        )
        target = next(
            (scene for scene in scenes if as_dict(scene).get("id") == selected_id),
            None,
        )
        if target is not None:
            st.session_state.target = target
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
settings_col, process_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        threshold_sigma = st.number_input(
            "Robust threshold multiplier",
            min_value=0.5,
            max_value=6.0,
            value=DEFAULTS["threshold_sigma"],
            step=0.1,
            key="threshold_sigma",
        )
    with p2:
        min_pixels = st.number_input(
            "Minimum candidate pixels",
            min_value=1,
            max_value=1000,
            value=DEFAULTS["min_component_pixels"],
            step=1,
            key="min_pixels",
        )
    with p3:
        dilation = st.number_input(
            "Final dilation radius",
            min_value=0,
            max_value=10,
            value=DEFAULTS["final_dilation"],
            step=1,
            key="final_dilation",
        )
    st.markdown(
        f'<div class="card-caption">Final minimum area: {int(min_pixels) * RESOLUTION * RESOLUTION:,} m². Raw candidates remain visible for small plume inspection.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with process_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    auth = st.session_state.get("cdse_auth")
    if auth:
        st.success(f"Copernicus connected · {auth.get('username', '')}")
        if st.button("Log out", key="logout"):
            st.session_state.pop("cdse_auth", None)
            st.rerun()
    else:
        st.link_button(
            "🌐 Open Copernicus website",
            "https://dataspace.copernicus.eu/",
            use_container_width=True,
        )
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Copernicus email")
            password = st.text_input("Copernicus password", type="password")
            totp = st.text_input("2FA code (optional)")
            login = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login:
            try:
                st.session_state.cdse_auth = authenticate_cdse(username, password, totp)
                st.success("Connected")
                st.rerun()
            except Exception as error:
                st.error(str(error))

    target = st.session_state.get("target")
    if target is not None:
        target_dt = get_datetime(target)
        target_label = target_dt.strftime("%Y-%m-%d") if target_dt else "Unknown date"
        st.markdown(
            f'<div class="card-title">Target ready</div><div class="card-caption">{target_label} · {get_tile(target) or "Unknown tile"}</div>',
            unsafe_allow_html=True,
        )
        disabled = not bool(st.session_state.get("cdse_auth"))
        run = st.button(
            "🛰️ Download AOI & detect methane",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="run_detection",
        )
        if run:
            try:
                token = get_access_token()
                progress = st.progress(0, text="Preparing…")
                target = st.session_state.target
                target_dt = get_datetime(target)
                if target_dt is None:
                    raise RuntimeError("Target scene date is unavailable.")

                progress.progress(10, text="Downloading target bands and quality layers…")
                target_path = download_scene(
                    target,
                    st.session_state.aoi,
                    token,
                    DEFAULTS["cloud_probability"],
                )
                target_bands, profile = read_stack(target_path)

                progress.progress(25, text="Searching the full reference window…")
                references = collect_reference_scenes(
                    target,
                    st.session_state.aoi,
                    target_dt,
                    reference_days,
                    max_cloud,
                )
                if not references:
                    raise RuntimeError(
                        "No same-tile reference scenes found. Increase the reference window or cloud limit."
                    )

                rows = []
                scored = []
                for index, reference in enumerate(references, start=1):
                    progress.progress(
                        25 + int(40 * index / max(1, len(references))),
                        text=f"Comparing reference {index}/{len(references)}…",
                    )
                    reference_path = download_scene(
                        reference,
                        st.session_state.aoi,
                        token,
                        DEFAULTS["cloud_probability"],
                    )
                    reference_bands, _ = read_stack(reference_path)
                    score, corr_b4, corr_b11, corr_b12, valid_count = reference_score(
                        target_bands,
                        reference_bands,
                        DEFAULTS["cloud_probability"],
                    )
                    dt = get_datetime(reference)
                    row = {
                        "id": as_dict(reference).get("id"),
                        "date": dt,
                        "tile": get_tile(reference),
                        "scene_cloud": get_cloud(reference),
                        "score": score,
                        "b4_corr": corr_b4,
                        "b11_corr": corr_b11,
                        "b12_corr": corr_b12,
                        "valid_pixels": valid_count,
                    }
                    rows.append(row)
                    if np.isfinite(score):
                        scored.append((score, reference, reference_bands))

                reference_table = pd.DataFrame(rows).sort_values(
                    "score", ascending=False, na_position="last"
                )
                st.session_state.reference_table = reference_table
                if not scored:
                    raise RuntimeError("No reference had enough valid pixels for comparison.")

                scored.sort(key=lambda item: item[0], reverse=True)
                selected = [
                    item for item in scored
                    if item[0] >= float(DEFAULTS["min_reference_correlation"])
                ][: int(DEFAULTS["max_reference_count"])]
                if not selected:
                    selected = scored[: min(3, len(scored))]

                reference_stack = {}
                for band in OUTPUT_BANDS:
                    reference_stack[band] = np.nanmedian(
                        np.stack([item[2][band] for item in selected]),
                        axis=0,
                    ).astype(np.float32)

                progress.progress(75, text="Running quality-masked anomaly detection…")
                params = {
                    "cloud_probability": DEFAULTS["cloud_probability"],
                    "threshold_sigma": float(threshold_sigma),
                    "min_component_pixels": int(min_pixels),
                    "final_dilation": int(dilation),
                    "gaussian_sigma": DEFAULTS["gaussian_sigma"],
                    "lrad_dilation": DEFAULTS["lrad_dilation"],
                }
                result = run_algorithm(target_bands, reference_stack, params)
                result["reference_count"] = len(selected)
                result["reference_best_score"] = float(selected[0][0])
                result["date"] = target_dt.strftime("%Y-%m-%d")

                output_folder = RESULT_DIR / target_dt.strftime("%Y%m%d")
                output_folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key in ("relative", "gaussian", "initial", "connected", "final", "valid"):
                    paths[key] = output_folder / f"{key}.tif"
                    save_raster(
                        paths[key],
                        result[key],
                        profile,
                        mask=key in ("initial", "connected", "final", "valid"),
                    )

                st.session_state.result = result
                st.session_state.paths = paths
                st.session_state.output_profile = profile
                st.session_state.png_outputs = {
                    "relative": image_png(result["relative"]),
                    "gaussian": image_png(result["gaussian"]),
                    "initial": image_png(result["initial"], mask=True),
                    "connected": image_png(result["connected"], mask=True),
                    "final": image_png(result["final"], mask=True),
                    "valid": image_png(result["valid"], mask=True),
                }
                progress.progress(100, text="Processing completed")
                st.success("Detection completed")
            except Exception as error:
                st.error(f"Detection failed: {error}")
    else:
        st.markdown('<div class="card-caption">Search and select a target scene first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if "result" in st.session_state:
    result = st.session_state.result
    pngs = st.session_state.png_outputs
    profile = st.session_state.output_profile
    st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)
    metrics = st.columns(7, gap="small")
    metrics[0].metric("Best reference", f"{result['reference_best_score']:.3f}")
    metrics[1].metric("References", result["reference_count"])
    metrics[2].metric("Valid", f"{result['valid_count']:,}")
    metrics[3].metric("Raw candidates", f"{result['initial_count']:,}")
    metrics[4].metric("Final", f"{result['final_count']:,}")
    metrics[5].metric("Regions", result["regions"])
    metrics[6].metric("Threshold", f"{result['threshold']:.5f}")

    outputs = [
        ("relative", "Relative MBMP", False),
        ("gaussian", "Smoothed anomaly", False),
        ("initial", "Raw candidates", True),
        ("final", "Final candidates", True),
        ("valid", "Validity mask", True),
    ]
    columns = st.columns(5, gap="small")
    for column, (key, title, is_mask) in zip(columns, outputs):
        with column:
            st.markdown(f"**{title}**")
            st.image(pngs[key], use_container_width=True)
            path = st.session_state.paths[key]
            choice = st.selectbox(
                "Format",
                ["GeoTIFF", "PNG + world file"],
                key=f"format_{key}",
            )
            if choice == "GeoTIFF":
                st.download_button(
                    "⬇ Download GeoTIFF",
                    path.read_bytes(),
                    file_name=path.name,
                    mime="image/tiff",
                    key=f"download_{key}_tif",
                    use_container_width=True,
                )
            else:
                package = georeferenced_png_package(
                    result[key], profile, mask=is_mask
                )
                st.download_button(
                    "⬇ Download PNG package",
                    package,
                    file_name=f"{key}_georeferenced.zip",
                    mime="application/zip",
                    key=f"download_{key}_png",
                    use_container_width=True,
                )

    st.markdown(
        f'<div class="card-caption">Raw candidates: {result["initial_count"]:,} pixels. Final candidates after connected-component filtering: {result["final_count"]:,} pixels. The raw mask is intentionally retained so a small landfill plume is not silently deleted.</div>',
        unsafe_allow_html=True,
    )
    if "reference_table" in st.session_state:
        st.download_button(
            "⬇ Download reference-selection CSV",
            st.session_state.reference_table.to_csv(index=False),
            file_name="reference_selection_v2.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
