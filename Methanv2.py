"""Sentinel-2 methane screening app v3.

This version addresses scattered detections and missing landfill-area detections:
- Correct SCL handling: SCL=2 is retained as a separate dark/cast-shadow class,
  not silently treated as valid methane evidence.
- Uses SCL + dataMask quality screening.
- Uses a stable relative SWIR absorption score based on B11/B12 log-ratios.
- Uses robust per-pixel temporal reference statistics from several scenes.
- Uses a scene-level robust background and a local anomaly score.
- Adds spatial coherence, morphological closing and connected-component filters.
- Reports raw, coherent and final masks separately.
- Does not claim that every red pixel is methane.
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
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    label,
    uniform_filter,
)
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
RESOLUTION = 20
DATA_BANDS = ["B03", "B04", "B08", "B11", "B12"]
OUTPUT_BANDS = DATA_BANDS + ["SCL", "DATA_MASK"]
CACHE_DIR = Path.home() / ".sentinel_methane_cache_v3"
RESULT_DIR = Path.home() / ".sentinel_methane_results_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "cloud_probability": 35.0,
    "threshold_sigma": 2.5,
    "local_sigma": 5.0,
    "min_component_pixels": 4,
    "min_coherent_pixels": 6,
    "final_dilation": 0,
    "gaussian_sigma": 0.8,
    "quality_dilation": 1,
    "max_reference_count": 8,
    "min_reference_score": 0.70,
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
    value = get_properties(data).get("datetime") or get_properties(data).get("start_datetime")
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
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
        geometries = [shape(g) for f in obj.get("features", []) if (g := normalize_geometry(f.get("geometry")))]
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


def authenticate_cdse(username, password, totp=""):
    if not username.strip() or not password.strip():
        raise RuntimeError("Please enter your Copernicus email and password.")
    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username.strip(),
        "password": password.strip(),
    }
    if totp.strip():
        form["totp"] = totp.strip()
    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")
    data = response.json()
    if not data.get("access_token"):
        raise RuntimeError("Copernicus did not return an access token.")
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": time.time() + int(data.get("expires_in", 600)),
        "username": username.strip(),
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None
    response = requests.post(
        TOKEN_URL,
        data={"grant_type": "refresh_token", "client_id": "cdse-public", "refresh_token": refresh_token},
        timeout=90,
    )
    if response.status_code >= 400:
        return None
    data = response.json()
    if not data.get("access_token"):
        return None
    auth["access_token"] = data["access_token"]
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
      bands: ["B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN", "DN"]
    }],
    output: {bands: 7, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12, sample.SCL, sample.dataMask];
}
"""


def download_scene(item, aoi, token, cloud_probability=35.0):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")
    cache_key = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_key
    output = folder / "bands_quality.tif"
    metadata = folder / "metadata.json"
    if output.exists() and metadata.exists():
        return output
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    start = acquisition.strftime("%Y-%m-%dT00:00:00Z")
    end = (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": start, "to": end}, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=payload, timeout=900)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")
    output.write_bytes(response.content)
    metadata.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    if array.shape[0] != len(OUTPUT_BANDS):
        raise RuntimeError(f"Expected {len(OUTPUT_BANDS)} bands, received {array.shape[0]}.")
    return {band: array[index] for index, band in enumerate(OUTPUT_BANDS)}, profile


def nd(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    use = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-8)
    output[use] = (first[use] - second[use]) / denominator[use]
    return output


def quality_mask(bands):
    scl = np.rint(bands["SCL"]).astype(np.int16)
    data_mask = bands["DATA_MASK"] >= 0.5
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in DATA_BANDS])
    positive = (bands["B11"] > 0) & (bands["B12"] > 0)
    # SCL=2 is dark/cast shadow. It is not automatically deleted here;
    # it is separately reported and excluded from methane evidence below.
    bad = np.isin(scl, [0, 1, 3, 7, 8, 9, 10, 11])
    return data_mask & finite & positive & ~bad


def usable_methane_pixels(bands, valid, dilation=1):
    scl = np.rint(bands["SCL"]).astype(np.int16)
    dark_or_shadow = np.isin(scl, [2])
    water = np.isin(scl, [6])
    exclude = dark_or_shadow | water | ~valid
    if dilation > 0:
        exclude = binary_dilation(exclude, structure=disk(int(dilation)))
    return valid & ~exclude


def robust_center_scale(values):
    values = values[np.isfinite(values)]
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.nanstd(values))
    return center, max(scale, 1e-8)


def normalize_gaussian(values, valid, sigma):
    numerator = gaussian_filter(np.where(valid, np.nan_to_num(values, nan=0.0), 0.0).astype(np.float32), sigma=float(sigma))
    denominator = gaussian_filter(valid.astype(np.float32), sigma=float(sigma))
    output = np.full(values.shape, np.nan, dtype=np.float32)
    use = denominator > 0.25
    output[use] = numerator[use] / denominator[use]
    return output


def local_zscore(values, valid, radius=5):
    finite_values = np.where(valid, np.nan_to_num(values, nan=0.0), 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    size = int(radius) * 2 + 1
    mean_num = uniform_filter(finite_values, size=size, mode="nearest")
    weight_mean = uniform_filter(weights, size=size, mode="nearest")
    local_mean = mean_num / np.maximum(weight_mean, 1e-6)
    mean_sq = uniform_filter(finite_values * finite_values, size=size, mode="nearest") / np.maximum(weight_mean, 1e-6)
    local_std = np.sqrt(np.maximum(mean_sq - local_mean * local_mean, 1e-8))
    output = np.full(values.shape, np.nan, dtype=np.float32)
    output[valid] = (values[valid] - local_mean[valid]) / local_std[valid]
    return output


def calculate_anomaly(target, reference, params):
    target_valid = quality_mask(target)
    reference_valid = quality_mask(reference)
    common = target_valid & reference_valid
    target_use = usable_methane_pixels(target, common, params["quality_dilation"])
    reference_use = usable_methane_pixels(reference, common, params["quality_dilation"])
    valid = target_use & reference_use
    if valid.sum() < 100:
        raise RuntimeError(f"Too few usable pixels: {int(valid.sum())}")

    # Relative SWIR absorption proxy. B12 is more methane-sensitive than B11.
    target_ratio = np.full(valid.shape, np.nan, dtype=np.float32)
    reference_ratio = np.full(valid.shape, np.nan, dtype=np.float32)
    target_ratio[valid] = np.log(target["B12"][valid] / target["B11"][valid])
    reference_ratio[valid] = np.log(reference["B12"][valid] / reference["B11"][valid])
    relative = target_ratio - reference_ratio
    relative[~valid] = np.nan

    raw_center, raw_scale = robust_center_scale(relative[valid])
    global_z = (relative - raw_center) / raw_scale
    smoothed = normalize_gaussian(global_z, valid, params["gaussian_sigma"])
    local_z = local_zscore(smoothed, valid, params["local_sigma"])

    scene_threshold = float(params["threshold_sigma"])
    local_threshold = float(params["local_sigma_threshold"])
    raw = valid & np.isfinite(smoothed) & (smoothed > scene_threshold)
    local = valid & np.isfinite(local_z) & (local_z > local_threshold)
    initial = raw & local

    # Require spatial support. Isolated one/two-pixel spikes are not coherent plumes.
    neighbourhood = uniform_filter(initial.astype(np.float32), size=3, mode="nearest") * 9
    coherent = initial & (neighbourhood >= int(params["min_neighbours"]))
    coherent = binary_closing(coherent, structure=disk(1))

    labels, _ = label(coherent, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= int(params["min_component_pixels"]))[0]
    retained = retained[retained != 0]
    connected = np.isin(labels, retained)

    final = connected.copy()
    if params["final_dilation"] > 0:
        final = binary_dilation(final, structure=disk(int(params["final_dilation"]))) & valid

    return {
        "relative": relative,
        "smoothed": smoothed,
        "local_z": local_z,
        "valid": valid,
        "raw": raw,
        "initial": initial,
        "coherent": coherent,
        "connected": connected,
        "final": final,
        "threshold": scene_threshold,
        "local_threshold": local_threshold,
        "valid_count": int(valid.sum()),
        "raw_count": int(raw.sum()),
        "initial_count": int(initial.sum()),
        "coherent_count": int(coherent.sum()),
        "final_count": int(final.sum()),
        "regions": int(len(retained)),
    }


def robust_corr(a, b, valid):
    use = valid & np.isfinite(a) & np.isfinite(b)
    if use.sum() < 100:
        return np.nan
    x, y = a[use].astype(float), b[use].astype(float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def reference_score(target, reference):
    valid = quality_mask(target) & quality_mask(reference)
    if valid.sum() < 100:
        return np.nan, np.nan, np.nan, np.nan, int(valid.sum())
    c4 = robust_corr(target["B04"], reference["B04"], valid)
    c11 = robust_corr(target["B11"], reference["B11"], valid)
    c12 = robust_corr(target["B12"], reference["B12"], valid)
    return np.nanmean([c4, c11, c12]), c4, c11, c12, int(valid.sum())


def save_raster(path, array, profile, mask=False):
    p = profile.copy()
    p.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    data = np.where(np.asarray(array).astype(bool), 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data, 1)


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
            lo, hi = float(np.percentile(values, 2)), float(np.percentile(values, 98))
            if hi <= lo:
                lo, hi = float(values.min()), float(values.max())
            if hi > lo:
                norm = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0, 1)
                rgb = (plt.get_cmap("RdBu_r")(norm)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def png_package(array, profile, mask=False):
    transform = profile["transform"]
    worldfile = f"{transform.a:.12f}\n0\n0\n{transform.e:.12f}\n{transform.c + transform.a / 2:.12f}\n{transform.f + transform.e / 2:.12f}\n"
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", image_png(array, mask))
        archive.writestr("output.pgw", worldfile)
        if profile.get("crs"):
            archive.writestr("output.prj", profile["crs"].to_wkt())
    return package.getvalue()


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map([geometry.centroid.y, geometry.centroid.x], zoom_start=11, tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    return fmap


def scene_label(scene, scenes):
    scene = next((x for x in scenes if as_dict(x).get("id") == scene), None)
    if scene is None:
        return "Unknown"
    dt = get_datetime(scene)
    return f"{dt.strftime('%Y-%m-%d') if dt else 'Unknown'} | {get_tile(scene) or 'Unknown'} | cloud {get_cloud(scene):.1f}%"


def get_references(target, aoi, reference_days, max_cloud):
    target_dt = get_datetime(target)
    if target_dt is None:
        return []
    candidates = search_scenes(aoi, target_dt - timedelta(days=int(reference_days)), target_dt + timedelta(days=int(reference_days)), max_cloud)
    target_id = as_dict(target).get("id")
    target_tile = get_tile(target)
    return [s for s in candidates if as_dict(s).get("id") != target_id and get_tile(s) == target_tile]


st.set_page_config(page_title="Sentinel-2 Methane Screening", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp{background:#f1faee;color:#111!important}.block-container{max-width:1700px;padding-top:3.9rem!important}.app-header,.app-card{background:#fff;border:1px solid #d8e6e8;border-radius:15px;box-shadow:0 2px 10px rgba(29,53,87,.05)}.app-header{display:flex;justify-content:space-between;padding:.75rem 1rem;margin-bottom:.9rem}.app-title{font-size:1.45rem;font-weight:850}.app-card{padding:.75rem;height:100%}.section-label{display:inline-block;background:#a8dadc;border-radius:999px;padding:.2rem .55rem;font-size:.65rem;font-weight:800}.card-title{font-weight:800}.card-caption{font-size:.75rem}.stApp p,.stApp label,.stApp span,.stApp td,.stApp th{color:#111!important}.stButton>button,.stDownloadButton>button{border-radius:9px;font-weight:750}.stButton>button[kind="primary"]{background:#e63946;border-color:#e63946;color:white!important}.stButton>button[kind="primary"] *{color:white!important}.stDownloadButton>button{background:#fff;border:1px solid #a8dadc;color:#111!important}footer{visibility:hidden}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="app-header"><div><div class="app-title">🛰️ Sentinel-2 Methane Screening</div><div class="card-caption">Quality-masked SWIR anomaly screening at 20 m · v3</div></div><div>20 m processing</div></div>""", unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(box(51.25, 35.65, 51.45, 35.80))

map_col, search_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    map_data = st_folium(create_map(st.session_state.aoi), height=385, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi:
            st.session_state.aoi = new_aoi
    st.markdown('</div>', unsafe_allow_html=True)

with search_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Search start", date(2026, 7, 1), key="search_start")
    with c2:
        end_date = st.date_input("Search end", date.today(), key="search_end")
    c1, c2 = st.columns(2)
    with c1:
        max_cloud = st.slider("Scene cloud cover (%)", 0.0, 100.0, 50.0, key="scene_cloud")
    with c2:
        reference_days = st.slider("Reference window (days)", 7, 180, 90, key="reference_days")
    if st.button("🔎 Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            st.session_state.scene_results = search_scenes(st.session_state.aoi, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), max_cloud)
            st.session_state.pop("target", None)
            st.session_state.pop("result", None)
            st.success(f"{len(st.session_state.scene_results)} scene(s) found")
        except Exception as error:
            st.session_state.scene_results = []
            st.error(f"Scene search failed: {error}")
    scenes = st.session_state.get("scene_results", [])
    if scenes:
        table = pd.DataFrame([{"date": get_datetime(s), "tile": get_tile(s), "cloud": get_cloud(s)} for s in scenes]).sort_values(["date", "cloud"], na_position="last")
        st.dataframe(table, use_container_width=True, height=112, hide_index=True)
        ids = [as_dict(s).get("id") for s in scenes if as_dict(s).get("id")]
        selected = st.selectbox("Target scene", ids, format_func=lambda value: scene_label(value, scenes), key="target_scene")
        st.session_state.target = next((s for s in scenes if as_dict(s).get("id") == selected), None)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
settings_col, process_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        threshold_sigma = st.number_input("Scene threshold", 0.5, 6.0, 2.5, 0.1, key="threshold_sigma")
    with p2:
        local_threshold = st.number_input("Local threshold", 0.5, 8.0, 2.0, 0.1, key="local_threshold")
    with p3:
        min_pixels = st.number_input("Minimum component pixels", 1, 1000, 4, 1, key="min_pixels")
    with p4:
        min_neighbours = st.number_input("Neighbour support / 9", 1, 9, 3, 1, key="min_neighbours")
    st.markdown('<div class="card-caption">The app now shows raw, coherent and final masks separately. This prevents small candidates from disappearing silently.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with process_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    auth = st.session_state.get("cdse_auth")
    if auth:
        st.success(f"Copernicus connected · {auth.get('username','')}")
    else:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Copernicus email")
            password = st.text_input("Copernicus password", type="password")
            totp = st.text_input("2FA code (optional)")
            login = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login:
            try:
                st.session_state.cdse_auth = authenticate_cdse(username, password, totp)
                st.rerun()
            except Exception as error:
                st.error(str(error))
    target = st.session_state.get("target")
    if target is not None:
        target_dt = get_datetime(target)
        run = st.button("🛰️ Download AOI & detect methane", type="primary", use_container_width=True, disabled=not bool(st.session_state.get("cdse_auth")))
        if run:
            try:
                token = get_access_token()
                progress = st.progress(0, text="Preparing…")
                target_path = download_scene(target, st.session_state.aoi, token)
                target_bands, profile = read_stack(target_path)
                progress.progress(20, text="Searching reference window…")
                references = get_references(target, st.session_state.aoi, reference_days, max_cloud)
                if not references:
                    raise RuntimeError("No same-tile references found. Increase the reference window.")
                scored = []
                rows = []
                for i, reference in enumerate(references, 1):
                    progress.progress(20 + int(45 * i / max(1, len(references))), text=f"Downloading reference {i}/{len(references)}…")
                    path = download_scene(reference, st.session_state.aoi, token)
                    bands, _ = read_stack(path)
                    score, c4, c11, c12, count = reference_score(target_bands, bands)
                    rows.append({"id": as_dict(reference).get("id"), "date": get_datetime(reference), "tile": get_tile(reference), "cloud": get_cloud(reference), "score": score, "b4": c4, "b11": c11, "b12": c12, "valid": count})
                    if np.isfinite(score):
                        scored.append((score, reference, bands))
                st.session_state.reference_table = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")
                if not scored:
                    raise RuntimeError("No reference could be scored.")
                scored.sort(key=lambda x: x[0], reverse=True)
                selected_refs = [x for x in scored if x[0] >= DEFAULTS["min_reference_score"]][:DEFAULTS["max_reference_count"]] or scored[:3]
                reference_stack = {band: np.nanmedian(np.stack([x[2][band] for x in selected_refs]), axis=0).astype(np.float32) for band in OUTPUT_BANDS}
                progress.progress(75, text="Running coherent anomaly detection…")
                params = {"threshold_sigma": float(threshold_sigma), "local_sigma": DEFAULTS["local_sigma"], "local_sigma_threshold": float(local_threshold), "min_component_pixels": int(min_pixels), "min_neighbours": int(min_neighbours), "final_dilation": DEFAULTS["final_dilation"], "gaussian_sigma": DEFAULTS["gaussian_sigma"], "quality_dilation": DEFAULTS["quality_dilation"]}
                result = calculate_anomaly(target_bands, reference_stack, params)
                result["reference_count"] = len(selected_refs)
                result["reference_best_score"] = float(selected_refs[0][0])
                folder = RESULT_DIR / (target_dt.strftime("%Y%m%d") if target_dt else "unknown")
                folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key in ("relative", "smoothed", "local_z", "valid", "raw", "initial", "coherent", "final"):
                    paths[key] = folder / f"{key}.tif"
                    save_raster(paths[key], result[key], profile, key in ("valid", "raw", "initial", "coherent", "final"))
                st.session_state.result = result
                st.session_state.paths = paths
                st.session_state.profile = profile
                st.session_state.pngs = {key: image_png(result[key], key in ("valid", "raw", "initial", "coherent", "final")) for key in paths}
                progress.progress(100, text="Completed")
                st.success("Detection completed")
            except Exception as error:
                st.error(f"Detection failed: {error}")
    st.markdown('</div>', unsafe_allow_html=True)

if "result" in st.session_state:
    result = st.session_state.result
    st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)
    metrics = st.columns(7)
    metrics[0].metric("Reference score", f"{result['reference_best_score']:.3f}")
    metrics[1].metric("References", result["reference_count"])
    metrics[2].metric("Valid", f"{result['valid_count']:,}")
    metrics[3].metric("Raw", f"{result['raw_count']:,}")
    metrics[4].metric("Coherent", f"{result['coherent_count']:,}")
    metrics[5].metric("Final", f"{result['final_count']:,}")
    metrics[6].metric("Regions", result["regions"])
    outputs = [("relative", "Relative SWIR anomaly", False), ("smoothed", "Smoothed anomaly", False), ("local_z", "Local z-score", False), ("valid", "Validity mask", True), ("raw", "Raw candidates", True), ("coherent", "Coherent candidates", True), ("final", "Final candidates", True)]
    columns = st.columns(7, gap="small")
    for column, (key, title, mask) in zip(columns, outputs):
        with column:
            st.markdown(f"**{title}**")
            st.image(st.session_state.pngs[key], use_container_width=True)
            st.download_button("⬇ GeoTIFF", st.session_state.paths[key].read_bytes(), file_name=st.session_state.paths[key].name, mime="image/tiff", key=f"download_{key}", use_container_width=True)
    st.markdown(f'<div class="card-caption">Valid pixels are quality-screened and usable for the comparison. Raw pixels show spectral anomaly; coherent pixels additionally require local spatial support. Final pixels pass the connected-component filter. This is a screening product, not a physical methane concentration.</div>', unsafe_allow_html=True)
    if "reference_table" in st.session_state:
        st.download_button("⬇ Reference CSV", st.session_state.reference_table.to_csv(index=False), file_name="reference_selection_v3.csv", mime="text/csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
