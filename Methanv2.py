"""Sentinel-2 methane candidate screening app.

v6 — Temporal Background Stacking (the MBMC way, done properly)
  • Per-pixel median of ΔR over N reference scenes
  • Conservative Gaussian smoothing (σ = 10 px = 200 m)
  • High-confidence filter: refuses to report unreliable detections
  • Aradkouh landfill (Tehran) as default AOI
  • Sentinel-5P CH4 context
"""
from __future__ import annotations

import io
import json
import math
import re
import hashlib
import time
import warnings
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
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
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
SITE_LAT = 35.505
SITE_LON = 51.330
SITE_RADIUS_M = 5000.0

# ── MBMC constants (v6: conservative, temporal-stacked) ──────────────
K_MBMP = 1.0e-5
DETREND_SIGMA = 80.0
ABS_FLOOR_PPB = 8.0
N_SIGMA = 2.0
GAUSS_SIGMA = 10.0         # 200 m — kill noise, keep plumes > 500 m
FLOOD_MIN_SIZE = 15        # 15 px = 6000 m² core
DILATE_RADIUS_FINAL = 3
SIGMA_WARN_PPB = 100.0

# Temporal background defaults
DEFAULT_N_BACKGROUND = 12
DEFAULT_BG_WINDOW_DAYS = 60
EXCLUDE_NEAR_DAYS = 5

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
    "min_valid_ref_pixels": 50,
    "abs_floor_ppb": ABS_FLOOR_PPB,
    "detrend_sigma": DETREND_SIGMA,
    "k_mbmp": K_MBMP,
    "max_plume_area_km2": 50.0,
    "site_radius_m": SITE_RADIUS_M,
    "sigma_warn_ppb": SIGMA_WARN_PPB,
    "n_background": DEFAULT_N_BACKGROUND,
    "bg_window_days": DEFAULT_BG_WINDOW_DAYS,
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
    r = requests.post(f"{STAC_URL}search", json=payload, timeout=120)
    r.raise_for_status()
    features = r.json().get("features", [])
    return features if isinstance(features, list) else list(features)


def authenticate_cdse(username: str, password: str, totp: str = ""):
    username = username.strip(); password = password.strip(); totp = totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")
    form = {"grant_type": "password", "client_id": "cdse-public",
            "username": username, "password": password}
    if totp:
        form["totp"] = totp
    r = requests.post(TOKEN_URL, data=form, timeout=90)
    if r.status_code >= 400:
        try:
            detail = r.json().get("error_description") or r.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {r.status_code}).")
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Copernicus did not return an access token.")
    return {"access_token": token, "refresh_token": data.get("refresh_token", ""),
            "expires_at": time.time() + int(data.get("expires_in", 600)),
            "username": username}


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None
    r = requests.post(TOKEN_URL, data={"grant_type": "refresh_token",
                                        "client_id": "cdse-public",
                                        "refresh_token": refresh_token}, timeout=90)
    if r.status_code >= 400:
        return None
    data = r.json()
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
  return {input: [{bands: ["B03","B04","B08","B11","B12"], units: "REFLECTANCE"}],
          output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.B03, s.B04, s.B08, s.B11, s.B12]; }
"""


def s5p_evalscript():
    return """//VERSION=3
function setup() {
  return {input: [{bands: ["CH4", "dataMask"]}], output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.CH4, 1.0]; }
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item); aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    lat = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(lat) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acq = get_datetime(item)
    if acq is None:
        raise RuntimeError("Could not read acquisition date.")
    payload = {
        "input": {"bounds": {"bbox": [minx, miny, maxx, maxy],
                             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": acq.strftime("%Y-%m-%dT00:00:00Z"),
                                                          "to": (acq + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")},
                                            "mosaickingOrder": "leastCC"}}]},
        "output": {"width": width, "height": height,
                    "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}",
                                              "Content-Type": "application/json"},
                       json=payload, timeout=900)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({r.status_code}): {detail}")
    output_path.write_bytes(r.content)
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path


def download_s5p_scene(aoi, date_from, date_to, access_token):
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps(["s5p", aoi, str(date_from), str(date_to)],
                                          sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / f"s5p_{cache_id}"
    output_path = folder / "ch4.tif"
    if output_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    payload = {
        "input": {"bounds": {"bbox": [minx, miny, maxx, maxy],
                              "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                   "data": [{"type": S5P_COLLECTION,
                             "dataFilter": {"timeRange": {"from": date_from.strftime("%Y-%m-%dT00:00:00Z"),
                                                          "to": date_to.strftime("%Y-%m-%dT23:59:59Z")}}}]},
        "output": {"width": 256, "height": 256,
                    "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": s5p_evalscript(),
    }
    r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}",
                                              "Content-Type": "application/json"},
                       json=payload, timeout=300)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:1000]
        raise RuntimeError(f"S5P Process API failed ({r.status_code}): {detail}")
    output_path.write_bytes(r.content)
    return output_path


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return {band: array[i] for i, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denom = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denom) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denom[valid]
    return output


def _safe_corr(a, b):
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    try:
        v = float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def evaluate_reference(target_bands, reference_bands, min_pixels):
    valid = (np.isfinite(target_bands["B04"]) & np.isfinite(reference_bands["B04"]) &
             np.isfinite(target_bands["B11"]) & np.isfinite(reference_bands["B11"]) &
             np.isfinite(target_bands["B12"]) & np.isfinite(reference_bands["B12"]))
    vc = int(valid.sum())
    if vc < min_pixels:
        return np.nan, np.nan, np.nan, np.nan, vc
    c04 = _safe_corr(target_bands["B04"][valid], reference_bands["B04"][valid])
    c11 = _safe_corr(target_bands["B11"][valid], reference_bands["B11"][valid])
    c12 = _safe_corr(target_bands["B12"][valid], reference_bands["B12"][valid])
    w, v = [], []
    if np.isfinite(c11): w.append(0.45); v.append(c11)
    if np.isfinite(c12): w.append(0.45); v.append(c12)
    if np.isfinite(c04): w.append(0.10); v.append(c04)
    if not v or sum(w) < 0.4:
        combined = np.nan
    else:
        w = np.asarray(w); v = np.asarray(v)
        combined = float(np.sum(w * v) / np.sum(w))
    return combined, c04, c11, c12, vc


# ══════════════════════════════════════════════════════════════════════
#  CORE ALGORITHM
# ══════════════════════════════════════════════════════════════════════

def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[b]) for b in BANDS])
    artifact = ((bands["B11"] >= PARAMS["swir_saturation"]) &
                (bands["B12"] >= PARAMS["swir_saturation"]))
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
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0.05) & (b12 > 0.05)
    if use.sum() < 100:
        return 1.0
    x = b12[use].astype(np.float64); y = b11[use].astype(np.float64)
    denom = float(np.sum(x * x))
    return 1.0 if denom <= 0 else float(np.sum(y * x) / denom)


def calculate_delta_R(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = (valid & np.isfinite(b11) & np.isfinite(b12)
           & (b11 > 0.05) & (b11 < 0.9) & (b12 > 0.05) & (b12 < 0.9))
    output[use] = (c * b12[use] - b11[use]) / b12[use]
    return output


def build_temporal_background(target_bands, scene_candidates, aoi, access_token,
                               valid, c, target_date, n_scenes, window_days,
                               progress_cb=None):
    """Download up to n_scenes references, compute ΔR for each, return
    per-pixel median and the list of scenes actually used.

    - Filters out scenes within ±EXCLUDE_NEAR_DAYS of the target date
    - Scores each candidate (multi-band SWIR correlation) and skips low-quality ones
    - Median-stacks ΔR → suppresses random atmospheric noise
    """
    dR_stack = []
    used = []
    scored = []
    for idx, scene in enumerate(scene_candidates):
        sdate = get_datetime(scene)
        if sdate is None:
            continue
        if abs((sdate - target_date).days) < EXCLUDE_NEAR_DAYS:
            continue
        if abs((sdate - target_date).days) > window_days:
            continue
        scored.append(scene)

    # Sort by date proximity (closest in time first — best atmospheric match)
    scored.sort(key=lambda s: abs((get_datetime(s) - target_date).total_seconds()))

    for idx, scene in enumerate(scored, start=1):
        if len(used) >= n_scenes:
            break
        if progress_cb:
            progress_cb(idx, min(len(scored), n_scenes * 2), len(used))
        try:
            ref_bands, _ = read_stack(download_scene(scene, aoi, access_token))
        except Exception:
            continue
        combined, _, _, _, vc = evaluate_reference(target_bands, ref_bands, PARAMS["min_valid_ref_pixels"])
        if not np.isfinite(combined) or combined < 0.85:
            continue
        dR = calculate_delta_R(ref_bands["B11"], ref_bands["B12"], c, valid)
        if not np.isfinite(dR).any():
            continue
        dR_stack.append(dR)
        used.append(scene)
        if len(used) >= n_scenes:
            break

    if not dR_stack:
        return None, []

    if len(dR_stack) == 1:
        return dR_stack[0], used

    stack = np.stack(dR_stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        composite = np.nanmedian(stack, axis=0).astype(np.float32)
    return composite, used


def remove_large_scale_background(dOmega, valid_mask, sigma=DETREND_SIGMA):
    finite_valid = valid_mask & np.isfinite(dOmega)
    d = np.where(finite_valid, dOmega, 0.0).astype(np.float64)
    w = finite_valid.astype(np.float32)
    ds = gaussian_filter(d, sigma=sigma); ws = gaussian_filter(w, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        bg = ds / ws
    bg[ws < 0.1] = 0.0
    residual = dOmega - bg
    residual[~valid_mask] = np.nan
    return residual.astype(np.float32), bg.astype(np.float32)


def normalized_gaussian(data, valid_mask, sigma):
    fv = valid_mask & np.isfinite(data)
    d = np.where(fv, data, 0.0).astype(np.float64)
    w = fv.astype(np.float32)
    ds = gaussian_filter(d, sigma=sigma); ws = gaussian_filter(w, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ds / ws
    out[ws < 0.1] = np.nan
    return out.astype(np.float32)


def make_spatial_mask(shape, profile, lat=SITE_LAT, lon=SITE_LON, radius_m=SITE_RADIUS_M):
    transform = profile["transform"]; crs = profile["crs"]
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
    px = abs(transform.a); py = abs(transform.e)
    dist_m = np.sqrt(((rows - row) * py) ** 2 + ((cols - col) * px) ** 2)
    return dist_m <= radius_m, (row, col)


def _robust_stats(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return median, sigma


def run_algorithm(target, dR_r_composite, profile, n_refs_used=1):
    target_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"]))
    valid = calculate_lrad(target, target_q)
    if valid.sum() < 100:
        raise RuntimeError("LRAD removed nearly all pixels.")

    c = calculate_c(target["B11"], target["B12"], valid)
    dR_t = calculate_delta_R(target["B11"], target["B12"], c, valid)

    dOmega = (dR_t - dR_r_composite) / K_MBMP
    dOmega[~valid] = np.nan

    dOmega_detrended, _ = remove_large_scale_background(dOmega, valid, sigma=DETREND_SIGMA)
    d_smooth = normalized_gaussian(dOmega_detrended, valid, GAUSS_SIGMA)

    vals = d_smooth[np.isfinite(d_smooth)]
    if vals.size == 0:
        raise RuntimeError("No finite values after detrending.")

    median, sigma_robust = _robust_stats(vals)
    threshold = median + max(PARAMS["threshold_sigma"] * sigma_robust, PARAMS["abs_floor_ppb"])

    candidate = np.isfinite(d_smooth) & (d_smooth > threshold) & valid
    spatial_mask, site_rc = make_spatial_mask(dOmega.shape, profile)
    candidate &= spatial_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = label(candidate, structure=structure)
    plume = np.zeros_like(candidate, dtype=bool)
    if n_labels > 0:
        sizes = np.bincount(labeled.ravel(), minlength=n_labels + 1)
        sizes[0] = 0
        sizes_kept = np.where(sizes >= PARAMS["min_component_pixels"], sizes, 0)
        if sizes_kept.max() > 0:
            order = np.argsort(sizes_kept)[::-1]
            order = order[order != 0]
            best_label = int(order[0])
            best_size = int(sizes_kept[best_label])
            if best_size < 30 and len(order) >= 2:
                plume = np.isin(labeled, order[:3])
            else:
                plume = (labeled == best_label)
            area_km2 = plume.sum() * (RESOLUTION * RESOLUTION) / 1e6
            if area_km2 > PARAMS["max_plume_area_km2"]:
                plume[:] = False

    if plume.any() and PARAMS["final_dilation"] > 0:
        plume = binary_dilation(plume, structure=disk(int(PARAMS["final_dilation"])))
    plume &= valid & spatial_mask

    # ── Plume properties ─────────────────────────────────────────────
    plume_area_km2 = float(plume.sum() * (RESOLUTION * RESOLUTION) / 1e6)
    if plume.any():
        plume_values = d_smooth[plume]
        peak_ppb = float(np.nanmax(plume_values))
        mean_ppb = float(np.nanmean(plume_values))
    else:
        peak_ppb = float("nan")
        mean_ppb = float("nan")
    snr = float((peak_ppb - median) / sigma_robust) if np.isfinite(peak_ppb) and sigma_robust > 0 else 0.0

    return {
        "relative": dOmega,
        "detrended": dOmega_detrended,
        "gaussian": d_smooth,
        "valid": valid,
        "spatial_mask": spatial_mask,
        "initial": candidate,
        "final": plume,
        "mean": median,
        "std": sigma_robust,
        "threshold": threshold,
        "regions": int(1 if plume.any() else 0),
        "valid_count": int(valid.sum()),
        "initial_count": int(candidate.sum()),
        "final_count": int(plume.sum()),
        "plume_area_km2": plume_area_km2,
        "peak_ppb": peak_ppb,
        "mean_ppb": mean_ppb,
        "snr": snr,
        "c": float(c),
        "site_rc": site_rc,
        "n_refs_used": int(n_refs_used),
    }


def assess_confidence(result):
    """Return (verdict, reasons) based on physical and statistical checks."""
    reasons = []
    ok = True

    if result["std"] > SIGMA_WARN_PPB:
        ok = False
        reasons.append(f"Global σ = {result['std']:.1f} ppb exceeds {SIGMA_WARN_PPB:.0f} ppb — "
                       f"background not clean enough.")
    if result["final_count"] == 0:
        ok = False
        reasons.append("No pixels above threshold.")
    if result["final_count"] > 0:
        if result["plume_area_km2"] < 0.2:
            ok = False
            reasons.append(f"Plume area {result['plume_area_km2']:.2f} km² is below 0.2 km² — "
                           f"likely a residual artifact.")
        if result["plume_area_km2"] > 20.0:
            ok = False
            reasons.append(f"Plume area {result['plume_area_km2']:.1f} km² is above 20 km² — "
                           f"likely over-detection.")
        if np.isfinite(result["snr"]) and result["snr"] < 3.0:
            ok = False
            reasons.append(f"Peak SNR = {result['snr']:.2f} is below 3.0 — signal is weak.")
        if result["regions"] > 5:
            ok = False
            reasons.append(f"{result['regions']} disconnected regions — pattern is noisy.")

    if ok:
        return "high", ["All physical and statistical checks passed."]
    return "low", reasons


def save_raster(path, array, profile, mask=False):
    p = profile.copy()
    p.update(count=1, dtype="uint8" if mask else "float32",
             nodata=255 if mask else -9999, compress="deflate",
             tiled=False, BIGTIFF="IF_SAFER")
    out = np.where(array, 1, 0).astype(np.uint8) if mask else \
          np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map([geometry.centroid.y, geometry.centroid.x], zoom_start=11, tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False,
                                     "marker": False, "circlemarker": False}).add_to(fmap)
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
                n = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
                import matplotlib.pyplot as plt
                rgb = (plt.get_cmap("RdBu_r")(n)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


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
            n = np.clip((np.nan_to_num(anomaly, nan=0) + max_abs) / (2 * max_abs), 0, 1)
            rgb = (plt.get_cmap("RdBu_r")(n)[:, :, :3] * 255).astype(np.uint8)
            rgb[~finite] = 255
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def legend_html(kind):
    if kind == "mask":
        rows = [("#dc1e1e", "Methane candidate"), ("#000000", "Background")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid pixels"), ("#000000", "Invalid / masked")]
    elif kind == "s5p":
        rows = [("#b43232", "CH4 above local mean"), ("#3250b4", "CH4 below local mean"), ("#ffffff", "No data")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>' for c, t in rows)
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def create_png_worldfile(profile):
    transform = profile["transform"]
    xres = transform.a; yres = transform.e
    xc = transform.c + xres / 2.0; yc = transform.f + yres / 2.0
    pgw = f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n{xc:.12f}\n{yc:.12f}\n"
    crs_text = profile.get("crs")
    prj = crs_text.to_wkt() if crs_text else ""
    return pgw.encode("utf-8"), prj.encode("utf-8")


def georeferenced_png_package(array, profile, mask=False):
    import zipfile
    png_data = image_png(array, mask=mask)
    pgw_data, prj_data = create_png_worldfile(profile)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("output.png", png_data)
        z.writestr("output.pgw", pgw_data)
        if prj_data:
            z.writestr("output.prj", prj_data)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Sentinel-2 Methane", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --red: #e63946; --honeydew: #f1faee; --frost: #a8dadc; --blue: #457b9d;
        --navy: #1d3557; --black: #111111; --white: #ffffff; --border: #d8e6e8;
        --muted: #4f5d63; --dark-field: #292a33; }
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem;
                    padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { position: relative; z-index: 10; display: flex; align-items: center;
              justify-content: space-between; background: #ffffff; border: 1px solid var(--border);
              border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem;
              margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc;
               border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem;
               font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px;
            padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04);
            height: 100%; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important;
                 border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem;
                 font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td,
.stApp th, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }
div[data-testid="stDateInput"] input, .stDateInput input { background-color: var(--dark-field) !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-testid="stNumberInput"] input, .stNumberInput input { background-color: var(--dark-field) !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"],
div[data-baseweb="select"] * { color: #111111 !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
.stSlider label { color: #111111 !important; }
.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px;
              padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important;
                border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem;
                font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45;
              margin: 0.2rem 0 0.45rem 0; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem;
    font-weight: 750; font-size: 0.78rem; color: #111111 !important; }
.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }
.stButton > button[kind="primary"] * { color: #ffffff !important; }
.stButton > button[kind="primary"]:hover { background: #c92f3b; border-color: #c92f3b; color: #ffffff !important; }
.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); }
div[data-testid="stDataFrame"] * { color: #111111 !important; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px;
                  padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box;
                  display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem;
                              color: #111111 !important; font-size: 0.82rem; line-height: 1.25; }
.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555;
                  border-radius: 2px; display: inline-block; }
.verdict-high { background: #e8f7ea; border: 2px solid #2a9d3f; border-radius: 12px;
                 padding: 0.7rem 0.9rem; margin: 0.4rem 0; }
.verdict-low { background: #fff4f4; border: 2px solid #e63946; border-radius: 12px;
                padding: 0.7rem 0.9rem; margin: 0.4rem 0; }
.verdict-title { font-size: 0.95rem; font-weight: 850; margin-bottom: 0.25rem; }
.verdict-reason { font-size: 0.78rem; line-height: 1.4; margin: 0.12rem 0; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
        <div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; MBMC ΔΩ (ppb) · Temporal Background Stacking</div>
    </div>
    <div class="status-pill">20 m &nbsp;•&nbsp; High-confidence mode</div>
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
    default_start = default_end - timedelta(days=90)
    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", default_start, key="start_date")
    with d2:
        end_date = st.date_input("End date", default_end, key="end_date")
    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 40.0, key="max_cloud")
    with s2:
        reference_days = st.slider("Max date distance (days)", 5, 120, 60, key="reference_days")

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
            {"date": get_datetime(s), "tile": get_tile(s), "cloud": get_cloud(s)}
            for s in scene_results
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")
        st.dataframe(scene_table, use_container_width=True, height=112, hide_index=True,
                     column_config={
                         "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                         "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
                     })

        scene_ids = [as_dict(s).get("id") for s in scene_results if as_dict(s).get("id")]

        def format_scene(scene_id):
            scene = next((c for c in scene_results if as_dict(c).get("id") == scene_id), None)
            if scene is None:
                return str(scene_id)
            sd = get_datetime(scene)
            dt = sd.strftime("%Y-%m-%d") if sd else "Unknown"
            return f"{dt}  |  {get_tile(scene) or 'Unknown tile'}  |  cloud {get_cloud(scene):.1f}%"

        if scene_ids:
            selected_scene_id = st.selectbox("Target scene", scene_ids,
                                              format_func=format_scene, key="target_scene_select")
            selected_scene = next((s for s in scene_results if as_dict(s).get("id") == selected_scene_id), None)
            if selected_scene is not None:
                st.session_state["target"] = selected_scene
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption"><b>Temporal Background Stacking:</b> the app will download '
                '<b>N reference scenes</b> around the target date and use their per-pixel median ΔR as '
                'the background — suppressing random atmospheric noise without blurring small plumes.</div>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["n_background"] = st.number_input("Background scenes (N)", min_value=3, max_value=30,
                                                   value=int(PARAMS["n_background"]), step=1, key="n_background")
    with p2:
        PARAMS["bg_window_days"] = st.number_input("Background window (days)", min_value=10, max_value=120,
                                                     value=int(PARAMS["bg_window_days"]), step=5, key="bg_window_days")
    with p3:
        PARAMS["threshold_sigma"] = st.number_input("Threshold σ multiplier", min_value=0.5, max_value=5.0,
                                                      value=float(PARAMS["threshold_sigma"]), step=0.1, key="threshold_sigma")
    p4, p5 = st.columns(2, gap="small")
    with p4:
        PARAMS["min_component_pixels"] = st.number_input("Min pixels / component", min_value=2, max_value=1000,
                                                           value=int(PARAMS["min_component_pixels"]), step=1,
                                                           key="min_component_pixels")
    with p5:
        PARAMS["final_dilation"] = st.number_input("Dilation radius (px)", min_value=0, max_value=20,
                                                     value=int(PARAMS["final_dilation"]), step=1, key="final_dilation")
    est = int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(f'<div class="card-caption">Min region ≈ {est:,} m² · gaussian σ = {GAUSS_SIGMA:.0f} px '
                f'(~{GAUSS_SIGMA*RESOLUTION:.0f} m) · detrend σ = {DETREND_SIGMA:.0f} px · floor = {ABS_FLOOR_PPB:.1f} ppb.</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    scene_results = st.session_state.get("scene_results", [])
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username","")}</div>',
                    unsafe_allow_html=True)
        lo, _ = st.columns([1, 2])
        with lo:
            if st.button("Log out", use_container_width=True, key="cdse_logout"):
                st.session_state.pop("cdse_auth", None); st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-help">Log in once per session. Password goes directly to the official Copernicus identity service.</div>',
                    unsafe_allow_html=True)
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8, placeholder="Only if 2FA")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login_submitted:
            try:
                with st.spinner("Connecting to Copernicus…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful.")
                st.rerun()
            except Exception as login_error:
                st.error(str(login_error))
        st.markdown('</div>', unsafe_allow_html=True)

    if scene_results and "target" in st.session_state:
        target = st.session_state.get("target")
        if target is not None:
            td = get_datetime(target)
            tt = get_tile(target) or "Unknown tile"
            tl = td.strftime("%Y-%m-%d") if td else "Unknown date"
            st.markdown(f'<div class="card-title">Ready to detect</div>'
                        f'<div class="card-caption">Target: {tl} · {tt}</div>',
                        unsafe_allow_html=True)
            detect_clicked = st.button("🛰️  Run detection (temporal background)",
                                        type="primary", use_container_width=True,
                                        key="detect_button",
                                        disabled=not bool(st.session_state.get("cdse_auth")))
            if not st.session_state.get("cdse_auth"):
                st.markdown('<div class="card-caption">Please connect your Copernicus account above.</div>',
                            unsafe_allow_html=True)
            if detect_clicked:
                progress = st.progress(0, text="Preparing methane detection…")
                progress_status = st.empty()
                try:
                    progress_status.markdown('<div class="card-caption">Step 1 of 5 · Connecting to CDSE…</div>',
                                              unsafe_allow_html=True)
                    progress.progress(5, text="Connecting…")
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    target_date = get_datetime(target)
                    target_tile = get_tile(target)
                    if target_date is None:
                        raise RuntimeError("Target has no valid date.")

                    progress_status.markdown('<div class="card-caption">Step 2 of 5 · Downloading target bands…</div>',
                                              unsafe_allow_html=True)
                    progress.progress(12, text="Downloading target…")
                    target_bands, profile = read_stack(download_scene(target, st.session_state.aoi, access_token))

                    # Compute LRAD + c on target up front (needed for ΔR of references)
                    target_q = float(np.nanquantile(target_bands["B03"], PARAMS["b03_quantile"]))
                    valid_mask = calculate_lrad(target_bands, target_q)
                    c_value = calculate_c(target_bands["B11"], target_bands["B12"], valid_mask)

                    # Candidate list for temporal background
                    candidates = [s for s in scene_results
                                   if as_dict(s).get("id") != as_dict(target).get("id")
                                   and get_datetime(s) is not None]
                    same_tile = [s for s in candidates if get_tile(s) == target_tile]
                    pool = same_tile if same_tile else candidates
                    # Sort by date proximity
                    pool.sort(key=lambda s: abs((get_datetime(s) - target_date).total_seconds()))

                    if not pool:
                        st.error("No other Sentinel-2 scenes available for temporal background. "
                                 "Widen your date range in the search step.")
                        st.stop()

                    n_bg = int(PARAMS["n_background"])
                    bg_window = int(PARAMS["bg_window_days"])

                    def prog_cb(i, total, used):
                        pct = 15 + int(60 * i / max(total, 1))
                        progress.progress(pct, text=f"Background scene {i} · used {used}/{n_bg}")

                    progress_status.markdown(
                        f'<div class="card-caption">Step 3 of 5 · Downloading and stacking up to <b>{n_bg}</b> '
                        f'background scenes (window ±{bg_window} days)…</div>',
                        unsafe_allow_html=True)
                    progress.progress(15, text="Building temporal background…")

                    dR_r_composite, used_scenes = build_temporal_background(
                        target_bands, pool, st.session_state.aoi, access_token,
                        valid_mask, c_value, target_date,
                        n_scenes=n_bg, window_days=bg_window,
                        progress_cb=prog_cb,
                    )

                    ref_table_rows = [{
                        "id": as_dict(s).get("id"),
                        "date": get_datetime(s),
                        "tile": get_tile(s),
                        "used_in_background": True,
                    } for s in used_scenes]
                    st.session_state.reference_table = pd.DataFrame(ref_table_rows)

                    if dR_r_composite is None or not used_scenes:
                        st.error(
                            f"Could not build a temporal background. "
                            f"Only {len(used_scenes)} scene(s) passed the quality filter (min 1 required).\n\n"
                            f"**Suggestions:**\n"
                            f"1. Widen the **date range** in the search step.\n"
                            f"2. Increase **cloud cover** threshold to 60–80%.\n"
                            f"3. Expand the **background window** to 90–120 days.\n"
                            f"4. Enlarge the AOI."
                        )
                        st.stop()

                    progress_status.markdown(
                        f'<div class="card-caption">Step 4 of 5 · Computing ΔΩ with {len(used_scenes)}-scene '
                        f'temporal background…</div>',
                        unsafe_allow_html=True)
                    progress.progress(80, text="Running MBMC detection…")

                    result = run_algorithm(target_bands, dR_r_composite, profile, n_refs_used=len(used_scenes))
                    result["date"] = target_date.strftime("%Y-%m-%d")
                    result["n_refs_requested"] = n_bg

                    verdict, reasons = assess_confidence(result)
                    result["confidence"] = verdict
                    result["confidence_reasons"] = reasons

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
                    progress_status.markdown('<div class="card-caption">Step 5 of 5 · Saving outputs…</div>',
                                              unsafe_allow_html=True)
                    progress.progress(100, text="Ready")
                    st.success(f"Processing completed · {len(used_scenes)}-scene temporal background")
                except Exception as error:
                    st.error(f"Detection failed: {error}")
    else:
        st.markdown('<div class="card-title">Select scenes first</div>'
                    '<div class="card-caption">Search for Sentinel-2 scenes, select a target, then run detection.</div>',
                    unsafe_allow_html=True)
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

    # Confidence verdict
    verdict = result.get("confidence", "low")
    reasons = result.get("confidence_reasons", [])
    if verdict == "high":
        st.markdown(f'<div class="verdict-high">'
                    f'<div class="verdict-title">✓ High-confidence detection</div>'
                    + "".join(f'<div class="verdict-reason">• {r}</div>' for r in reasons) +
                    '</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-low">'
                    f'<div class="verdict-title">⚠ Low-confidence — no reliable plume confirmed</div>'
                    + "".join(f'<div class="verdict-reason">• {r}</div>' for r in reasons) +
                    '</div>', unsafe_allow_html=True)

    metrics = st.columns(7, gap="small")
    metrics[0].metric("Refs used", result.get("n_refs_used", 1))
    metrics[1].metric("Global σ (ppb)", f"{result['std']:.1f}")
    metrics[2].metric("Plume area (km²)", f"{result['plume_area_km2']:.2f}")
    metrics[3].metric("Peak ΔΩ (ppb)", f"{result['peak_ppb']:.1f}" if np.isfinite(result['peak_ppb']) else "—")
    metrics[4].metric("SNR", f"{result['snr']:.2f}")
    metrics[5].metric("Initial px", f"{result['initial_count']:,}")
    metrics[6].metric("Final px", f"{result['final_count']:,}")

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
                legend_kind = "mask" if key == "final" else ("valid" if key == "valid" else "continuous")
                st.markdown(legend_html(legend_kind), unsafe_allow_html=True)
            path = st.session_state.paths[key]
            format_choice = st.selectbox("Download format",
                                          ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"],
                                          key=f"format_choice_{key}")
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button("⬇ Download GeoTIFF", path.read_bytes(),
                                    file_name=path.name, mime="image/tiff",
                                    key=f"download_tif_compact_{key}", use_container_width=True)
            else:
                pkg = georeferenced_png_package(result[key], profile, mask=key in ("final", "valid"))
                st.download_button("⬇ Download PNG package", pkg,
                                    file_name=f"{key}_georeferenced_png.zip", mime="application/zip",
                                    key=f"download_png_compact_{key}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-note"><b>v6 temporal background:</b> '
        f'median = <b>{result["mean"]:.2f} ppb</b>, global σ = <b>{result["std"]:.2f} ppb</b>, '
        f'threshold = <b>{result["threshold"]:.2f} ppb</b>. '
        f'<b>{result.get("n_refs_used",1)} scene(s)</b> in the temporal stack. '
        f'c = <b>{result["c"]:.4f}</b> · gaussian σ = <b>{GAUSS_SIGMA:.0f} px (~{GAUSS_SIGMA*RESOLUTION:.0f} m)</b> · '
        f'detrend σ = <b>{DETREND_SIGMA:.0f} px</b> · floor = <b>{ABS_FLOOR_PPB:.0f} ppb</b>.</div>',
        unsafe_allow_html=True)
    d1, d2 = st.columns([1, 3], gap="small")
    with d1:
        if "reference_table" in st.session_state:
            st.download_button("⬇ Background scenes CSV",
                                st.session_state.reference_table.to_csv(index=False),
                                file_name="background_scenes.csv", mime="text/csv",
                                key="download_reference_csv_compact", use_container_width=True)
    with d2:
        st.markdown('<div class="card-caption" style="margin-top:0.55rem;">ΔΩ (ppb) is a screening quantity '
                    '(MBMC framework). Not physical CH4 concentration or emission rate.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05b · SENTINEL-5P CH4 CONTEXT</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">TROPOMI CH4 (~5.5 × 7 km per pixel). Anomaly relative to local mean.</div>',
                unsafe_allow_html=True)
    s5p_col1, s5p_col2 = st.columns([1, 3], gap="small")
    with s5p_col1:
        run_s5p = st.button("🛰️  Fetch S5P CH4", type="primary", use_container_width=True,
                             key="s5p_button",
                             disabled=not bool(st.session_state.get("cdse_auth")))
    with s5p_col2:
        s5p_days = st.slider("S5P temporal window (days)", 1, 30, 15, key="s5p_days")

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
                    access_token)
            with rasterio.open(s5p_path) as src:
                ch4 = src.read(1).astype(np.float32)
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[ch4 <= 0] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.success("S5P CH4 loaded")
        except Exception as s5p_error:
            st.error(f"S5P fetch failed: {s5p_error}")

    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        vc4 = ch4[np.isfinite(ch4)]
        if vc4.size < 2:
            placeholder = float(np.nanmean(vc4)) if vc4.size > 0 else 1900.0
            ch4 = np.full_like(ch4, placeholder)
            vc4 = ch4[np.isfinite(ch4)]
            st.warning("⚠️ S5P returned no valid pixels. Placeholder shown. Try 30-day window.")
        if vc4.size > 1:
            mean_val = float(np.nanmean(vc4)); max_val = float(np.nanmax(vc4)); min_val = float(np.nanmin(vc4))
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
            st.markdown('<div class="card-caption">Each pixel is deviation from local mean (red = above, blue = below). '
                        'At ~7 km resolution, a small landfill occupies 1–2 pixels.</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("Not enough valid S5P CH4 pixels.")

    st.markdown('</div>', unsafe_allow_html=True)
