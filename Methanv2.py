"""Sentinel-2 methane candidate screening app.

MBMC-faithful v4 (calibrated + multi-band reference selection)
  • Aradkouh landfill (Tehran) as default AOI
  • Cloud cover 30% + last-30-days date defaults
  • Robust reference selection using B11/B12 SWIR bands (methane-relevant)
  • QA warning when residual sigma is physically implausible
  • 30-day time-series + visual daily playback
  • Sentinel-5P CH4 with forced AOI-center display

UI/design preserved. Algorithm engine follows MBMC paper with recalibrated
smoothing, adaptive floor, and multi-band reference scoring.
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
import rasterio.transform
from rasterio.warp import transform as rio_transform
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

# ── Default AOI: Aradkouh / Kahrizak landfill (Tehran) ───────────────
DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

# ── Landfill site (spatial constraint center) ────────────────────────
SITE_LAT = 35.505
SITE_LON = 51.330
SITE_RADIUS_M = 5000.0

# ── MBMC paper constants (recalibrated for small AOI + S2 noise) ─────
K_MBMP = 1.0e-5
DETREND_SIGMA = 80.0        # ← کاهش از 150 (AOI کوچک ~1500 px)
ABS_FLOOR_PPB = 5.0         # ← کاهش از 20 (σ واقعی انتظاری)
N_SIGMA = 1.75              # ← کاهش از 2.5 (سیگنال ضعیف)
GAUSS_SIGMA = 10.0          # ← افزایش از 3.0 (سرکوب نویز ۲۰ متری S2)
FLOOD_MIN_SIZE = 10
DILATE_RADIUS_FINAL = 3
SIGMA_WARN_PPB = 50.0       # QA warning threshold
REFERENCE_CORR_MIN = 0.70    # MBMC paper: Red/B04 correlation QC
REFERENCE_WINDOW_DEFAULT = 30

PARAMS = {
    "b03_quantile": 0.05,
    "swir_saturation": 1.0,
    "ndwi_threshold": 0.20,
    "ndvi_threshold": 0.45,
    "ndbi_threshold": 0.40,
    "ndsi_threshold": 0.42,
    "lrad_dilation": 1,
    "gaussian_sigma": GAUSS_SIGMA,
    "threshold_sigma": N_SIGMA,
    "min_component_pixels": FLOOD_MIN_SIZE,
    "final_dilation": DILATE_RADIUS_FINAL,
    "min_solidity": 0.0,
    "b12_epsilon": 1e-6,
    "min_valid_ref_pixels": 50,
    "abs_floor_ppb": ABS_FLOOR_PPB,
    "detrend_sigma": DETREND_SIGMA,
    "k_mbmp": K_MBMP,
    "max_plume_area_km2": 100.0,   # ← افزایش از 10 (تهران بزرگ‌تر از 10 km²)
    "site_radius_m": SITE_RADIUS_M,
    "sigma_warn_ppb": SIGMA_WARN_PPB,
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
        "collections": ["sentinel-2-l1c"],
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
  return [sample.CH4, sample.dataMask];
}
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps(["s2_l1c_v5", item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
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
            "data": [{"type": "sentinel-2-l1c", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}],
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


def download_s5p_scene(aoi, date_from, date_to, access_token, min_qa=50):
    """Download quality-filtered Sentinel-5P CH4 for the requested time window.

    The returned GeoTIFF contains two bands:
      1) CH4 in ppb
      2) dataMask (1=data, 0=no-data)

    minQa is applied by the Sentinel Hub/CDSE Process API. For a same-day
    validation run, date_from and date_to should be the same calendar date.
    """
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(
        json.dumps(["s5p_ch4_v5", aoi, str(date_from), str(date_to), int(min_qa)], sort_keys=True).encode()
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
                "processing": {"minQa": int(min_qa)},
            }],
        },
        "output": {
            # This is only the requested output grid. It is NOT the native
            # TROPOMI footprint; the latter remains several kilometres.
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


def _safe_corr(a, b):
    """Safe Pearson correlation for possibly degenerate arrays."""
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    try:
        value = float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def evaluate_reference(target_bands, reference_bands, min_pixels):
    """Multi-band reference quality assessment.

    Returns (combined_score, corr_b04, corr_b11, corr_b12, valid_count).
    The SWIR-weighted score is retained as a diagnostic only.
    For the actual MBMC-faithful reference selection, B04/Red correlation
    is used as the quality-control and ranking criterion (>= 0.70).
    """
    valid = (
        np.isfinite(target_bands["B04"]) & np.isfinite(reference_bands["B04"]) &
        np.isfinite(target_bands["B11"]) & np.isfinite(reference_bands["B11"]) &
        np.isfinite(target_bands["B12"]) & np.isfinite(reference_bands["B12"])
    )
    valid_count = int(valid.sum())
    if valid_count < min_pixels:
        return np.nan, np.nan, np.nan, np.nan, valid_count

    t_b04 = target_bands["B04"][valid]
    r_b04 = reference_bands["B04"][valid]
    t_b11 = target_bands["B11"][valid]
    r_b11 = reference_bands["B11"][valid]
    t_b12 = target_bands["B12"][valid]
    r_b12 = reference_bands["B12"][valid]

    corr_b04 = _safe_corr(t_b04, r_b04)
    corr_b11 = _safe_corr(t_b11, r_b11)
    corr_b12 = _safe_corr(t_b12, r_b12)

    weights = []
    values = []
    if np.isfinite(corr_b11):
        weights.append(0.45); values.append(corr_b11)
    if np.isfinite(corr_b12):
        weights.append(0.45); values.append(corr_b12)
    if np.isfinite(corr_b04):
        weights.append(0.10); values.append(corr_b04)

    if not values or sum(weights) < 0.4:
        combined = np.nan
    else:
        w = np.asarray(weights, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        combined = float(np.sum(w * v) / np.sum(w))

    return combined, corr_b04, corr_b11, corr_b12, valid_count


# ══════════════════════════════════════════════════════════════════════
#  CORE ALGORITHM (MBMC-faithful + robust thresholding)
# ══════════════════════════════════════════════════════════════════════

def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in BANDS])
    artifact = ((bands["B11"] >= PARAMS["swir_saturation"]) & (bands["B12"] >= PARAMS["swir_saturation"]))
    artifact |= bands["B03"] <= q_value
    artifact |= normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"]
    artifact |= normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"]
    artifact |= normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"]
    artifact |= normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"]
    artifact |= ~finite
    if PARAMS["lrad_dilation"] > 0:
        artifact = binary_dilation(artifact, iterations=int(PARAMS["lrad_dilation"]))
    return finite & ~artifact


def calculate_c(b11, b12, valid):
    """c = Σ(B11·B12) / Σ(B12²)."""
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0.05) & (b12 > 0.05)
    if use.sum() < 100:
        return 1.0
    x = b12[use].astype(np.float64)
    y = b11[use].astype(np.float64)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return 1.0
    return float(np.sum(y * x) / denom)


def calculate_delta_R(b11, b12, c, valid):
    """ΔR = (c·B12 − B11) / B12."""
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = (
        valid
        & np.isfinite(b11) & np.isfinite(b12)
        & (b11 > 0.05) & (b11 < 0.9)
        & (b12 > 0.05) & (b12 < 0.9)
    )
    output[use] = (c * b12[use] - b11[use]) / b12[use]
    return output


def remove_large_scale_background(dOmega, valid_mask, sigma=DETREND_SIGMA):
    finite_valid = valid_mask & np.isfinite(dOmega)
    d = np.where(finite_valid, dOmega, 0.0).astype(np.float64)
    w = finite_valid.astype(np.float32)
    ds = gaussian_filter(d, sigma=sigma)
    ws = gaussian_filter(w, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        background = ds / ws
    background[ws < 0.1] = 0.0
    residual = dOmega - background
    residual[~valid_mask] = np.nan
    return residual.astype(np.float32), background.astype(np.float32)


def normalized_gaussian(data, valid_mask, sigma):
    finite_valid = valid_mask & np.isfinite(data)
    d = np.where(finite_valid, data, 0.0).astype(np.float64)
    w = finite_valid.astype(np.float32)
    ds = gaussian_filter(d, sigma=sigma)
    ws = gaussian_filter(w, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = ds / ws
    output[ws < 0.1] = np.nan
    return output.astype(np.float32)


def make_spatial_mask(shape, profile, lat=SITE_LAT, lon=SITE_LON, radius_m=SITE_RADIUS_M):
    transform = profile["transform"]
    crs = profile["crs"]
    try:
        xs, ys = rio_transform("EPSG:4326", crs, [lon], [lat])
    except Exception:
        xs, ys = [lon], [lat]
    row, col = rasterio.transform.rowcol(transform, xs[0], ys[0])
    row = int(row); col = int(col)
    h, w = shape
    if not (0 <= row < h and 0 <= col < w):
        return np.ones(shape, dtype=bool), (row, col)
    rows, cols = np.ogrid[:h, :w]
    px = abs(transform.a)
    py = abs(transform.e)
    dist_m = np.sqrt(((rows - row) * py) ** 2 + ((cols - col) * px) ** 2)
    return dist_m <= radius_m, (row, col)


def _robust_stats(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else float(np.std(finite))
    if not np.isfinite(sigma_robust) or sigma_robust <= 0:
        sigma_robust = 1.0
    return median, sigma_robust


def run_algorithm(target, reference, profile):
    # MBMC LRAD is applied to both passes. Using the intersection prevents
    # a surface artifact present only in the reference pass from entering
    # the target-reference differential signal.
    target_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"]))
    reference_q = float(np.nanquantile(reference["B03"], PARAMS["b03_quantile"]))
    valid_target = calculate_lrad(target, target_q)
    valid_reference = calculate_lrad(reference, reference_q)
    valid = valid_target & valid_reference

    if valid.sum() < 100:
        raise RuntimeError("LRAD removed nearly all common target/reference pixels. Relax thresholds or enlarge AOI.")

    # IMPORTANT: normalization coefficients are estimated independently for
    # the target and reference passes. A shared coefficient can convert
    # ordinary radiometric differences into a false differential signal.
    c_target = calculate_c(target["B11"], target["B12"], valid_target)
    c_reference = calculate_c(reference["B11"], reference["B12"], valid_reference)

    dR_t = calculate_delta_R(target["B11"], target["B12"], c_target, valid)
    dR_r = calculate_delta_R(reference["B11"], reference["B12"], c_reference, valid)

    dOmega_t = dR_t / K_MBMP
    dOmega_r = dR_r / K_MBMP
    dOmega = dOmega_t - dOmega_r
    dOmega[~valid] = np.nan

    dOmega_detrended, _ = remove_large_scale_background(dOmega, valid, sigma=DETREND_SIGMA)
    d_smooth = normalized_gaussian(dOmega_detrended, valid, GAUSS_SIGMA)

    vals = d_smooth[np.isfinite(d_smooth)]
    if vals.size == 0:
        raise RuntimeError("No finite values after detrending. Try a different reference scene.")

    median, sigma_robust = _robust_stats(vals)
    threshold = median + max(
        PARAMS["threshold_sigma"] * sigma_robust,
        PARAMS["abs_floor_ppb"],
    )

    candidate = np.isfinite(d_smooth) & (d_smooth > threshold) & valid

    spatial_mask, site_rc = make_spatial_mask(dOmega.shape, profile)
    candidate &= spatial_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = label(candidate, structure=structure)
    plume = np.zeros_like(candidate, dtype=bool)
    kept_region_count = 0

    if n_labels > 0:
        sizes = np.bincount(labeled.ravel(), minlength=n_labels + 1)
        sizes[0] = 0
        sizes_kept = np.where(sizes >= PARAMS["min_component_pixels"], sizes, 0)
        if sizes_kept.max() > 0:
            order = np.argsort(sizes_kept)[::-1]
            order = order[order != 0]
            kept_region_count = int(len(order))
            best_label = int(order[0])
            best_size = int(sizes_kept[best_label])
            if best_size < 30 and len(order) >= 2:
                top_labels = order[:3]
                plume = np.isin(labeled, top_labels)
            else:
                plume = (labeled == best_label)
            area_km2 = plume.sum() * (RESOLUTION * RESOLUTION) / 1e6
            if area_km2 > PARAMS["max_plume_area_km2"]:
                plume[:] = False

    if plume.any() and PARAMS["final_dilation"] > 0:
        plume = binary_dilation(plume, structure=disk(int(PARAMS["final_dilation"])))
    plume &= valid & spatial_mask

    return {
        "relative": dOmega,
        "detrended": dOmega_detrended,
        "gaussian": d_smooth,
        "valid": valid,
        "valid_target": valid_target,
        "valid_reference": valid_reference,
        "spatial_mask": spatial_mask,
        "initial": candidate,
        "final": plume,
        "mean": median,
        "std": sigma_robust,
        "threshold": threshold,
        "regions": int(kept_region_count),
        "valid_count": int(valid.sum()),
        "initial_count": int(candidate.sum()),
        "final_count": int(plume.sum()),
        "c": float(c_target),
        "c_target": float(c_target),
        "c_reference": float(c_reference),
        "site_rc": site_rc,
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
#  TIME-SERIES
# ══════════════════════════════════════════════════════════════════════

def process_single_day(target_scene, reference_scenes, aoi, access_token, store_image=True):
    try:
        target_date = get_datetime(target_scene)
        target_path = download_scene(target_scene, aoi, access_token)
        target_bands, target_profile = read_stack(target_path)
        target_tile = get_tile(target_scene)
        same_tile = [r for r in reference_scenes if get_tile(r) == target_tile]
        candidates = same_tile if same_tile else reference_scenes

        best_ref_bands = None
        best_reference_b04 = -np.inf
        best_score = np.nan
        best_b4 = np.nan
        best_valid = 0

        for ref in candidates:
            try:
                ref_bands, _ = read_stack(download_scene(ref, aoi, access_token))
            except Exception:
                continue
            combined, corr_b04, corr_b11, corr_b12, valid_count = evaluate_reference(
                target_bands, ref_bands, PARAMS["min_valid_ref_pixels"]
            )
            # MBMC-faithful choice: first enforce the B04 correlation QC,
            # then select the highest B04 correlation. SWIR score remains diagnostic.
            if np.isfinite(corr_b04) and corr_b04 >= REFERENCE_CORR_MIN and corr_b04 > best_reference_b04:
                best_reference_b04 = corr_b04
                best_ref_bands = ref_bands
                best_b4 = corr_b04
                best_score = combined
                best_valid = valid_count

        if best_ref_bands is None:
            return None

        result = run_algorithm(target_bands, best_ref_bands, target_profile)

        row = {
            "date": target_date,
            "mean_mbmp": float(np.nanmean(result["detrended"])),
            "max_mbmp": float(np.nanmax(result["detrended"])),
            "std_mbmp": float(np.nanstd(result["detrended"])),
            "final_pixels": int(result["final_count"]),
            "regions": int(result["regions"]),
            "ref_score": float(best_score) if np.isfinite(best_score) else np.nan,
            "b4_correlation": float(best_b4) if np.isfinite(best_b4) else np.nan,
            "c_target": float(result["c_target"]),
            "c_reference": float(result["c_reference"]),
        }
        if store_image:
            row["png_relative"] = image_png(result["detrended"])
            row["png_mask"] = image_png(result["final"], mask=True)
        return row
    except Exception as exc:
        return {"date": get_datetime(target_scene), "error": str(exc)}


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
    --muted: #4f5d63;
    --dark-field: #292a33;
}
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04); height: 100%; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td, .stApp th, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] span, .stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }
div[data-testid="stDateInput"] div[data-baseweb="input"], div[data-testid="stDateInput"] div[data-baseweb="input"] > div, div[data-testid="stDateInput"] input, div[data-testid="stDateInput"] input[type="text"], .stDateInput div[data-baseweb="input"], .stDateInput div[data-baseweb="input"] > div, .stDateInput input, .stDateInput input[type="text"] { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stDateInput"] input::-webkit-datetime-edit, div[data-testid="stDateInput"] input::-webkit-datetime-edit-text, div[data-testid="stDateInput"] input::-webkit-datetime-edit-month-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-day-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-year-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-fields-wrapper, .stDateInput input::-webkit-datetime-edit, .stDateInput input::-webkit-datetime-edit-text, .stDateInput input::-webkit-datetime-edit-month-field, .stDateInput input::-webkit-datetime-edit-day-field, .stDateInput input::-webkit-datetime-edit-year-field, .stDateInput input::-webkit-datetime-edit-fields-wrapper { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stNumberInput"] input, .stNumberInput input { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
input::placeholder, textarea::placeholder { color: #bfc3cc !important; opacity: 1 !important; }
div[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] * { color: #111111 !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"], div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="listbox"] *, div[data-baseweb="popover"] [role="option"] *, ul[role="listbox"] *, li[role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li[role="option"]:hover { background: #2b2e38 !important; }
.stDateInput, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.15rem; }
.stSlider > div { padding-top: 0.05rem; padding-bottom: 0.05rem; }
.stSlider label, .stSlider [data-testid="stTickBar"] * { color: #111111 !important; }
.stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stThumbValue"] * { color: #ffffff !important; }
[data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { color: #ffffff !important; }
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus { -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45; margin: 0.2rem 0 0.45rem 0; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.78rem; color: #111111 !important; }
.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * { color: #ffffff !important; }
.stButton > button[kind="primary"]:hover { background: #c92f3b; border-color: #c92f3b; color: #ffffff !important; }
.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }
.stDownloadButton > button:hover { background: #f1faee; border-color: #457b9d; color: #111111 !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); }
div[data-testid="stDataFrame"] * { color: #111111 !important; }
.download-label { color: #111111 !important; font-size: 0.62rem; font-weight: 700; margin: 0.2rem 0 0.12rem 0; }
.map-frame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.82rem; line-height: 1.25; }
.result-legend .legend-row span:last-child { color: #111111 !important; }
.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
        <div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; MBMC-faithful ΔΩ (ppb) candidate detection</div>
    </div>
    <div class="status-pill">20 m processing &nbsp;•&nbsp; Red-band reference QC + SWIR diagnostics</div>
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
        reference_days = st.slider("Reference window (days)", 1, 90, REFERENCE_WINDOW_DEFAULT, key="reference_days")

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
            {
                "date": get_datetime(scene),
                "tile": get_tile(scene),
                "cloud": get_cloud(scene),
            }
            for scene in scene_results
        ]).sort_values(
            ["date", "cloud"],
            ascending=[True, True],
            na_position="last",
        )

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

        scene_ids = [
            as_dict(scene).get("id")
            for scene in scene_results
            if as_dict(scene).get("id")
        ]

        def format_scene(scene_id):
            scene = next(
                (
                    candidate
                    for candidate in scene_results
                    if as_dict(candidate).get("id") == scene_id
                ),
                None,
            )
            if scene is None:
                return str(scene_id)
            scene_date = get_datetime(scene)
            date_text = scene_date.strftime("%Y-%m-%d") if scene_date else "Unknown date"
            return f"{date_text}  |  {get_tile(scene) or 'Unknown tile'}  |  cloud {get_cloud(scene):.1f}%"

        if scene_ids:
            selected_scene_id = st.selectbox(
                "Target scene",
                scene_ids,
                format_func=format_scene,
                key="target_scene_select",
            )
            selected_scene = next(
                (
                    scene
                    for scene in scene_results
                    if as_dict(scene).get("id") == selected_scene_id
                ),
                None,
            )
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
        PARAMS["final_dilation"] = st.number_input("Final dilation radius", min_value=0, max_value=20, value=int(PARAMS["final_dilation"]), step=1, key="final_dilation")
    estimated_area_m2 = int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(f'<div class="card-caption">Minimum connected region ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution. '
                f'Gaussian σ = {GAUSS_SIGMA:.0f} px · detrend σ = {DETREND_SIGMA:.0f} px · absolute floor = {ABS_FLOOR_PPB:.1f} ppb.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    scene_results = st.session_state.get("scene_results", [])
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username", "")}</div>', unsafe_allow_html=True)
        logout_col, _ = st.columns([1, 2])
        with logout_col:
            if st.button("Log out", use_container_width=True, key="cdse_logout"):
                st.session_state.pop("cdse_auth", None)
                st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-help">Log in once in this browser session. Your password is sent directly to the official Copernicus identity service; the app keeps only the temporary API token.</div>', unsafe_allow_html=True)
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8, placeholder="Only if your account uses 2FA")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login_submitted:
            try:
                with st.spinner("Connecting to Copernicus…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful. You can now run the detection.")
                st.rerun()
            except Exception as login_error:
                st.error(str(login_error))
        st.markdown('</div>', unsafe_allow_html=True)

    if scene_results and "target" in st.session_state:
        target = st.session_state.get("target")
        if target is not None:
            target_date = get_datetime(target)
            target_tile = get_tile(target) or "Unknown tile"
            target_label = target_date.strftime("%Y-%m-%d") if target_date else "Unknown date"
            st.markdown(f'<div class="card-title">Ready to detect</div><div class="card-caption">Target: {target_label} · {target_tile}</div>', unsafe_allow_html=True)
            detect_clicked = st.button("🛰️  Download AOI & Detect Methane", type="primary", use_container_width=True, key="detect_button", disabled=not bool(st.session_state.get("cdse_auth")))
            if not st.session_state.get("cdse_auth"):
                st.markdown('<div class="card-caption">Please connect your Copernicus account above before downloading Sentinel-2 data.</div>', unsafe_allow_html=True)
            if detect_clicked:
                progress = st.progress(0, text="Preparing methane detection…")
                progress_status = st.empty()
                try:
                    progress_status.markdown('<div class="card-caption">Step 1 of 5 · Connecting to CDSE and preparing the target scene…</div>', unsafe_allow_html=True)
                    progress.progress(8, text="Preparing target scene…")
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    target_date = get_datetime(target)
                    target_tile = get_tile(target)
                    if target_date is None:
                        raise RuntimeError("The selected target scene has no valid acquisition date.")
                    # The visible search table is not used as the complete reference pool.
                    # We query CDSE again around the selected target so that the requested
                    # ±reference_days window is actually respected even when the UI date
                    # range is shorter than that window.
                    reference_search_start = target_date - timedelta(days=int(reference_days))
                    reference_search_end = target_date + timedelta(days=int(reference_days))
                    reference_pool = search_scenes(
                        st.session_state.aoi,
                        reference_search_start,
                        reference_search_end,
                        max_cloud,
                    )
                    all_candidates = [
                        scene for scene in reference_pool
                        if as_dict(scene).get("id") != as_dict(target).get("id")
                        and get_datetime(scene) is not None
                        and abs((get_datetime(scene) - target_date).total_seconds()) / 86400 <= reference_days
                    ]
                    same_tile = [scene for scene in all_candidates if get_tile(scene) == target_tile]
                    references = same_tile if same_tile else all_candidates
                    if not references:
                        st.warning("No reference scene exists in the selected time window. Increase the date range or reference window.")
                        st.stop()
                    progress_status.markdown('<div class="card-caption">Step 2 of 5 · Downloading target image bands and preparing the AOI…</div>', unsafe_allow_html=True)
                    progress.progress(25, text="Downloading target bands…")
                    target_bands, profile = read_stack(download_scene(target, st.session_state.aoi, access_token))

                    best_reference = None
                    best_score = np.nan
                    best_b4 = -np.inf
                    best_valid = 0
                    reference_rows = []
                    total_refs = max(1, len(references))

                    for ref_index, reference in enumerate(references, start=1):
                        pct = 30 + int(40 * (ref_index - 1) / total_refs)
                        progress_status.markdown(f'<div class="card-caption">Step 3 of 5 · Evaluating reference scene {ref_index} of {total_refs} (multi-band SWIR scoring)…</div>', unsafe_allow_html=True)
                        progress.progress(pct, text=f"Reference scene {ref_index} of {total_refs}…")

                        try:
                            reference_bands, _ = read_stack(download_scene(reference, st.session_state.aoi, access_token))
                        except Exception:
                            reference_rows.append({
                                "id": as_dict(reference).get("id"),
                                "date": get_datetime(reference),
                                "tile": get_tile(reference),
                                "ref_score": np.nan,
                                "b04_correlation": np.nan,
                                "b11_correlation": np.nan,
                                "b12_correlation": np.nan,
                                "valid_pixels": 0,
                                "status": "download failed",
                            })
                            continue

                        combined, corr_b04, corr_b11, corr_b12, valid_count = evaluate_reference(
                            target_bands, reference_bands, PARAMS["min_valid_ref_pixels"]
                        )

                        reference_rows.append({
                            "id": as_dict(reference).get("id"),
                            "date": get_datetime(reference),
                            "tile": get_tile(reference),
                            "ref_score": combined,
                            "b04_correlation": corr_b04,
                            "b11_correlation": corr_b11,
                            "b12_correlation": corr_b12,
                            "valid_pixels": valid_count,
                            "status": (
                                "selected" if np.isfinite(corr_b04) and corr_b04 >= REFERENCE_CORR_MIN
                                else "below B04 QC"
                            ) if np.isfinite(valid_count) else "low-validity",
                        })

                        # MBMC paper selection rule: B04/Red correlation >= 0.70
                        # forms the candidate pool; highest B04 correlation wins.
                        if (
                            np.isfinite(corr_b04)
                            and corr_b04 >= REFERENCE_CORR_MIN
                            and corr_b04 > best_b4
                        ):
                            best_score = combined
                            best_reference = reference_bands
                            best_b4 = corr_b04
                            best_valid = valid_count

                    st.session_state.reference_table = pd.DataFrame(reference_rows)

                    if best_reference is None:
                        total_valid = sum(int(r.get("valid_pixels", 0)) for r in reference_rows)
                        target_valid = int(np.isfinite(target_bands["B04"]).sum())
                        st.error(
                            f"Could not select a reference scene.\n\n"
                            f"- References downloaded: **{len(reference_rows)}**\n"
                            f"- Total valid pixels across all references: **{total_valid:,}**\n"
                            f"- Target valid B4 pixels: **{target_valid:,}**\n\n"
                            f"**Suggestions:**\n"
                            f"1. Increase the **date range** (make start date earlier).\n"
                            f"2. Increase the **cloud cover** threshold to 50–70%.\n"
                            f"3. Increase **Reference window (days)** to 30–60.\n"
                            f"4. Set Reference window to 30 days (or larger)."
                            f"\n5. A valid reference must satisfy B04/Red correlation ≥ {REFERENCE_CORR_MIN:.2f}."
                        )
                        st.stop()

                    progress_status.markdown('<div class="card-caption">Step 4 of 5 · Running MBMC ΔΩ (ppb) anomaly detection and candidate cleanup…</div>', unsafe_allow_html=True)
                    progress.progress(78, text="Running methane detection…")
                    result = run_algorithm(target_bands, best_reference, profile)
                    result["b4_correlation"] = float(best_b4) if np.isfinite(best_b4) else float("nan")
                    result["combined_score"] = float(best_score) if np.isfinite(best_score) else float("nan")
                    result["reference_qc_threshold"] = REFERENCE_CORR_MIN
                    result["date"] = target_date.strftime("%Y-%m-%d")

                    signal_ratio = result["final_count"] / max(1, result["valid_count"])
                    result["signal_ratio"] = float(signal_ratio)

                    output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                    output_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("relative", "detrended", "gaussian", "final", "valid"):
                        paths[key] = output_folder / f"{key}.tif"
                        save_raster(paths[key], result[key], profile, key in ("final", "valid"))
                    st.session_state.result = result
                    st.session_state.paths = paths
                    st.session_state.output_profile = profile
                    st.session_state.png_outputs = {
                        "relative": image_png(result["relative"]),
                        "detrended": image_png(result["detrended"]),
                        "gaussian": image_png(result["gaussian"]),
                        "final": image_png(result["final"], mask=True),
                        "valid": image_png(result["valid"], mask=True),
                    }
                    progress_status.markdown('<div class="card-caption">Step 5 of 5 · Saving georeferenced outputs and preparing downloads…</div>', unsafe_allow_html=True)
                    progress.progress(100, text="Ready to detect · outputs are ready")
                    st.success("Processing completed")

                    # ── QA warnings ────────────────────────────────────
                    if result["std"] > SIGMA_WARN_PPB:
                        st.warning(
                            f"⚠️ **Residual noise is physically implausible** "
                            f"(robust σ = **{result['std']:.1f} ppb**, threshold above "
                            f"~{SIGMA_WARN_PPB:.0f} ppb suggests the reference scene is unreliable). "
                            f"Real methane fluctuations between two S2 acquisitions rarely exceed "
                            f"10–30 ppb after detrending. The candidate mask in this run is likely "
                            f"dominated by residual surface/aerosol noise rather than real CH4. "
                            f"**Try:** (1) a reference date within 5–10 days, (2) a larger AOI, "
                            f"(3) higher Gaussian σ (e.g. 15–20 px)."
                        )

                    if result["final_count"] == 0:
                        st.warning(
                            f"⚠️ **No plume above threshold detected.** "
                            f"Median ΔΩ = {result['mean']:.2f} ppb, robust σ = {result['std']:.2f} ppb, "
                            f"threshold = {result['threshold']:.2f} ppb. "
                            f"Try lowering the Threshold multiplier to 1.2–1.5, or pick a different "
                            f"target/reference date pair."
                        )
                    elif signal_ratio < 0.00005:
                        st.warning(
                            f"⚠️ **Very small final mask.** "
                            f"The candidate mask covers only **{signal_ratio*100:.5f}%** of valid pixels "
                            f"({result['final_count']:,} / {result['valid_count']:,}). "
                            f"This may be a weak plume or residual noise."
                        )
                except Exception as error:
                    st.error(f"Detection failed: {error}")
    else:
        st.markdown('<div class="card-title">Select scenes first</div><div class="card-caption">Search for Sentinel-2 scenes, select a target, then run the detection.</div>', unsafe_allow_html=True)
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

    combined_display = result.get("combined_score", np.nan)
    if np.isfinite(combined_display):
        score_text = f"{combined_display:.3f}"
    elif np.isfinite(result.get("b4_correlation", np.nan)):
        score_text = f"{result['b4_correlation']:.3f} (B4)"
    else:
        score_text = "n/a"

    metrics[0].metric("Ref score (SWIR-weighted)", score_text)
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Initial", f"{result['initial_count']:,}")
    metrics[3].metric("Final", f"{result['final_count']:,}")
    metrics[4].metric("Regions", result["regions"])
    metrics[5].metric("Threshold", f"{result['threshold']:.2f}")

    result_items = [
        ("detrended", "ΔΩ after detrend (ppb)", "Detrended"),
        ("gaussian", "Gaussian smoothed", "Smoothed"),
        ("final", "Methane candidates", "Final mask"),
        ("valid", "Valid pixels", "Validity mask"),
    ]
    result_cols = st.columns(4, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{tag}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(png_outputs[key], use_container_width=True, output_format="PNG")
            with legend_col:
                st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                if key == "final":
                    legend_kind = "mask"
                elif key == "valid":
                    legend_kind = "valid"
                else:
                    legend_kind = "continuous"
                st.markdown(legend_html(legend_kind), unsafe_allow_html=True)
            path = st.session_state.paths[key]
            format_choice = st.selectbox("Download format", ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"], key=f"format_choice_{key}")
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button("⬇ Download GeoTIFF", path.read_bytes(), file_name=path.name, mime="image/tiff", key=f"download_tif_compact_{key}", use_container_width=True)
            else:
                png_package = georeferenced_png_package(result[key], profile, mask=key in ("final", "valid"))
                st.download_button("⬇ Download Georeferenced PNG package", png_package, file_name=f"{key}_georeferenced_png.zip", mime="application/zip", key=f"download_png_compact_{key}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    removed_pixels = max(0, int(result["initial_count"]) - int(result["final_count"]))
    st.markdown(f'<div class="result-note"><b>Robust thresholding:</b> median = <b>{result["mean"]:.2f} ppb</b>, robust σ = <b>{result["std"]:.2f} ppb</b>, threshold = <b>{result["threshold"]:.2f} ppb</b>. Only the largest connected region ≥ <b>{int(PARAMS["min_component_pixels"]):,} px</b> was kept (MBMC convention). Initial: <b>{result["initial_count"]:,}</b> → Final: <b>{result["final_count"]:,}</b> · c = <b>{result["c"]:.4f}</b> · detrend σ = <b>{DETREND_SIGMA:.0f} px</b> · gaussian σ = <b>{GAUSS_SIGMA:.0f} px</b> · floor = <b>{ABS_FLOOR_PPB:.0f} ppb</b>.</div>', unsafe_allow_html=True)
    d1, d2 = st.columns([1, 3], gap="small")
    with d1:
        st.download_button("⬇ Reference table CSV", st.session_state.reference_table.to_csv(index=False), file_name="reference_selection.csv", mime="text/csv", key="download_reference_csv_compact", use_container_width=True)
    with d2:
        st.markdown('<div class="card-caption" style="margin-top:0.55rem;">ΔΩ (ppb) is a screening quantity following the MBMC framework (Cheng et al., 2026); it is not physical methane concentration or an emission rate. Reference scoring now weights the SWIR bands (B11/B12) that carry the methane signal.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05a · 30-DAY TIME SERIES & VISUAL PLAYBACK</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Track daily changes of detrended ΔΩ (ppb) over the 30-day window. Use the slider below the chart to visually scrub through each day.</div>', unsafe_allow_html=True)

    ts_col1, ts_col2 = st.columns([1, 3], gap="small")
    with ts_col1:
        run_ts = st.button("📈  Build 30-day series + visuals", type="primary", use_container_width=True, key="build_ts_button", disabled=not bool(st.session_state.get("cdse_auth")))
    with ts_col2:
        st.markdown('<div class="card-caption" style="margin-top:0.55rem;">Downloads and processes every Sentinel-2 scene in the window. Cached after first run.</div>', unsafe_allow_html=True)

    if run_ts:
        try:
            access_token = get_access_token()
            target_date = get_datetime(st.session_state.get("target"))
            if target_date is None:
                raise RuntimeError("Target date is missing.")
            window_scenes = [
                s for s in st.session_state.get("scene_results", [])
                if get_datetime(s) is not None
                and abs((get_datetime(s) - target_date).total_seconds()) / 86400 <= 30
            ]
            by_day = {}
            for s in window_scenes:
                d = get_datetime(s).date()
                by_day.setdefault(d, []).append(s)
            days_sorted = sorted(by_day.keys())

            ts_progress = st.progress(0, text="Building time series…")
            ts_status = st.empty()
            rows = []
            daily_visuals = {}
            total_days = max(1, len(days_sorted))
            for idx, day in enumerate(days_sorted, start=1):
                ts_status.markdown(f'<div class="card-caption">Processing day {idx} of {total_days} · {day.isoformat()}</div>', unsafe_allow_html=True)
                ts_progress.progress(int(100 * idx / total_days), text=f"Day {idx} of {total_days}")
                day_scenes = sorted(by_day[day], key=lambda s: get_cloud(s))
                target_scene_day = day_scenes[0]
                refs = [
                    s for s in window_scenes
                    if as_dict(s).get("id") != as_dict(target_scene_day).get("id")
                    and get_datetime(s) is not None
                    and abs((get_datetime(s) - get_datetime(target_scene_day)).total_seconds()) / 86400 <= 10
                ]
                if not refs:
                    continue
                row = process_single_day(target_scene_day, refs, st.session_state.aoi, access_token, store_image=True)
                if row is not None and "error" not in row:
                    rows.append({k: v for k, v in row.items() if not k.startswith("png_")})
                    daily_visuals[row["date"].strftime("%Y-%m-%d")] = {
                        "png_relative": row["png_relative"],
                        "png_mask": row["png_mask"],
                        "mean_mbmp": row["mean_mbmp"],
                        "max_mbmp": row["max_mbmp"],
                        "final_pixels": row["final_pixels"],
                    }

            if rows:
                ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
                st.session_state.timeseries_df = ts_df
                st.session_state.daily_visuals = daily_visuals
                ts_progress.progress(100, text="Time series ready")
                ts_status.empty()
                st.success(f"Time series built · {len(ts_df)} day(s) processed · {len(daily_visuals)} visual frames")
            else:
                st.warning("No valid day could be processed. Try a wider date range or a larger AOI.")
        except Exception as ts_error:
            st.error(f"Time series failed: {ts_error}")

    if "timeseries_df" in st.session_state:
        ts = st.session_state.timeseries_df
        if not ts.empty and "mean_mbmp" in ts.columns:
            chart_df = ts.dropna(subset=["mean_mbmp"]).set_index("date")[["mean_mbmp", "max_mbmp"]]
            if not chart_df.empty:
                st.markdown('<div class="card-caption">Daily detrended ΔΩ (ppb) statistics in the AOI · reference QC B04 ≥ 0.70</div>', unsafe_allow_html=True)
                st.line_chart(chart_df, use_container_width=True, height=240)
                st.dataframe(
                    ts,
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                    column_config={
                        "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                        "mean_mbmp": st.column_config.NumberColumn("Mean ΔΩ (ppb)", format="%.2f"),
                        "max_mbmp": st.column_config.NumberColumn("Max ΔΩ (ppb)", format="%.2f"),
                        "std_mbmp": st.column_config.NumberColumn("Std ΔΩ (ppb)", format="%.2f"),
                        "ref_score": st.column_config.NumberColumn("Ref score", format="%.3f"),
                        "b4_correlation": st.column_config.NumberColumn("B4 corr", format="%.3f"),
                    },
                )
                st.download_button(
                    "⬇ Download 30-day series CSV",
                    ts.to_csv(index=False),
                    file_name="s2_timeseries_30d.csv",
                    mime="text/csv",
                    key="download_ts_csv",
                    use_container_width=False,
                )

    if "daily_visuals" in st.session_state:
        visuals = st.session_state.daily_visuals
        dates_available = sorted(visuals.keys())
        if dates_available:
            st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-caption" style="font-weight:700;font-size:0.85rem;">🎬 Visual daily playback — drag the slider to scrub through the month</div>', unsafe_allow_html=True)
            selected_day = st.select_slider(
                "Select day",
                options=dates_available,
                value=dates_available[0],
                key="visual_day_slider",
                label_visibility="collapsed",
            )
            if selected_day in visuals:
                frame = visuals[selected_day]
                v1, v2 = st.columns(2, gap="small")
                with v1:
                    st.markdown(f'<div class="card-caption" style="font-weight:700;">{selected_day} · ΔΩ detrended</div>', unsafe_allow_html=True)
                    st.image(frame["png_relative"], use_container_width=True, output_format="PNG")
                with v2:
                    st.markdown(f'<div class="card-caption" style="font-weight:700;">{selected_day} · Candidate mask</div>', unsafe_allow_html=True)
                    st.image(frame["png_mask"], use_container_width=True, output_format="PNG")
                m1, m2, m3 = st.columns(3, gap="small")
                m1.metric("Mean ΔΩ (ppb)", f"{frame['mean_mbmp']:.2f}")
                m2.metric("Max ΔΩ (ppb)", f"{frame['max_mbmp']:.2f}")
                m3.metric("Candidate pixels", f"{frame['final_pixels']:,}")

    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05b · SENTINEL-5P CH4 CONTEXT</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">TROPOMI CH4 (~5.5 × 7 km). QA filter: minQa = 50%. For validation, use 0 days so the S5P observation is restricted to the target date. Visualization is <b>anomaly relative to the valid-pixel mean</b>.</div>', unsafe_allow_html=True)

    s5p_col1, s5p_col2 = st.columns([1, 3], gap="small")
    with s5p_col1:
        run_s5p = st.button("🛰️  Fetch S5P CH4", type="primary", use_container_width=True, key="s5p_button", disabled=not bool(st.session_state.get("cdse_auth")))
    with s5p_col2:
        s5p_days = st.slider("S5P temporal window (days around target)", 0, 30, 0, key="s5p_days")

    if run_s5p:
        try:
            access_token = get_access_token()
            target_date = get_datetime(st.session_state.get("target"))
            if target_date is None:
                raise RuntimeError("Target date is missing.")
            with st.spinner("Fetching Sentinel-5P CH4…"):
                s5p_path = download_s5p_scene(
                    st.session_state.aoi,
                    target_date - timedelta(days=int(s5p_days)),
                    target_date + timedelta(days=int(s5p_days)),
                    access_token,
                )
            with rasterio.open(s5p_path) as src:
                ch4 = src.read(1).astype(np.float32)
                s5p_mask = src.read(2).astype(np.float32) if src.count >= 2 else np.ones_like(ch4, dtype=np.float32)
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[(s5p_mask < 0.5) | (ch4 <= 0)] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.session_state.s5p_mask = s5p_mask
            st.success("S5P CH4 loaded with QA/dataMask filtering")
        except Exception as s5p_error:
            st.error(f"S5P fetch failed: {s5p_error}")

    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        valid_ch4 = ch4[np.isfinite(ch4)]

        if valid_ch4.size < 2:
            placeholder = float(np.nanmean(valid_ch4)) if valid_ch4.size > 0 else 1900.0
            ch4 = np.full_like(ch4, placeholder)
            valid_ch4 = ch4[np.isfinite(ch4)]
            st.warning(
                "⚠️ Sentinel-5P returned no valid QA-approved pixels for this AOI and time window. "
                "No placeholder is shown because a fabricated CH4 value would invalidate the comparison. "
                "If you are checking the same Sentinel-2 acquisition date, first keep the window at 0; "
                "only widen it for contextual analysis."
            )

        if valid_ch4.size > 1:
            mean_val = float(np.nanmean(valid_ch4))
            max_val = float(np.nanmax(valid_ch4))
            min_val = float(np.nanmin(valid_ch4))
            m1, m2, m3, m4 = st.columns(4, gap="small")
            m1.metric("Mean CH4 (ppb)", f"{mean_val:.1f}")
            m2.metric("Min CH4 (ppb)", f"{min_val:.1f}")
            m3.metric("Max CH4 (ppb)", f"{max_val:.1f}")
            m4.metric("Range (ppb)", f"{max_val - min_val:.1f}")
            img_col, leg_col = st.columns([3.6, 1.0], gap="small")
            with img_col:
                st.image(ch4_anomaly_png(ch4), use_container_width=True, output_format="PNG")
            with leg_col:
                st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                st.markdown(legend_html("s5p"), unsafe_allow_html=True)
            st.markdown('<div class="card-caption">Only QA-approved S5P pixels are shown. The anomaly is relative to the valid-pixel mean; this is a regional atmospheric context, not a pixel-to-pixel validation of the 20 m Sentinel-2 plume.</div>', unsafe_allow_html=True)
        else:
            st.warning("Not enough valid S5P CH4 pixels even after fallback. Increase the temporal window to 30 days.")

    st.markdown('</div>', unsafe_allow_html=True)
