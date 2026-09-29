"""Sentinel-2 Methane Screening App

Integrated extension of the existing MBMC-faithful dashboard.

The existing MBMC workflow, UI style, map, scene search, authentication,
outputs, time series, and Sentinel-5P context are preserved conceptually.
A second, optional research pipeline is added:

    MBMP-guided candidate mask -> SimCLR pretraining -> segmentation

Important scientific limitation:
The 2026 paper does not publish its full dataset, exact AOI, checkpoint,
complete decoder, all hyperparameters, or complete IME constants. Therefore
this file implements a reproducible approximation of the described method,
not the authors' private source code.

Run:
    streamlit run sentinel_methane_app_simclr_integrated.py

Optional packages for the research model:
    pip install torch torchvision scikit-image
"""
from __future__ import annotations

import io
import json
import math
import re
import hashlib
import time
import zipfile
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
from folium.plugins import Draw, MousePosition
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_AVAILABLE = False

try:
    from PIL import Image
except Exception:
    Image = None

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
BANDS = ["B03", "B04", "B08", "B11", "B12", "SCL"]
MODEL_BANDS = ["B11", "B12"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".sentinel_methane_cache"
RESULT_DIR = Path.home() / ".sentinel_methane_results"
MODEL_DIR = RESULT_DIR / "simclr_models"
for directory in (CACHE_DIR, RESULT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

S5P_COLLECTION = "sentinel-5p-l2"
DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)
SITE_LAT = 35.505
SITE_LON = 51.330
SITE_RADIUS_M = 5000.0

K_MBMP = 1.0e-5
DETREND_SIGMA = 200.0
ABS_FLOOR_PPB = 20.0
N_SIGMA = 2.0
GAUSS_SIGMA = 5.0
FLOOD_MIN_SIZE = 10
DILATE_RADIUS_FINAL = 3

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
    "b12_epsilon": 1e-6,
    "min_valid_ref_pixels": 50,
    "abs_floor_ppb": ABS_FLOOR_PPB,
    "detrend_sigma": DETREND_SIGMA,
    "k_mbmp": K_MBMP,
    "max_plume_area_km2": 10.0,
    "site_radius_m": SITE_RADIUS_M,
    "threshold_percentile_cap": 99.0,
    "trim_low_pct": 10.0,
    "trim_high_pct": 90.0,
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


def geocode_location(query: str):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            headers={"User-Agent": "SentinelMethaneScreeningApp/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            first = results[0]
            return float(first["lat"]), float(first["lon"]), first.get("display_name", query)
    except Exception:
        pass
    return None


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
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Copernicus did not return an access token.")
    return {
        "access_token": token,
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
    input: [
      {bands: ["B03","B04","B08","B11","B12"], units: "REFLECTANCE", datasource: "refl"},
      {bands: ["SCL"], datasource: "scl"}
    ],
    output: {bands: 6, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(samples) {
  var refl = samples.refl[0];
  var scl = samples.scl[0];
  return [refl.B03, refl.B04, refl.B08, refl.B11, refl.B12, scl.SCL];
}
"""


def s5p_evalscript():
    return """//VERSION=3
function setup() {
  return {input: [{bands: ["CH4", "dataMask"]}], output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(sample) { return [sample.CH4, 1.0]; }
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION, "v6"], sort_keys=True).encode()).hexdigest()[:24]
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
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")
    time_from = acquisition.strftime("%Y-%m-%dT00:00:00Z")
    time_to = (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [
                {"type": "sentinel-2-l2a", "id": "refl", "dataFilter": {"timeRange": {"from": time_from, "to": time_to}, "mosaickingOrder": "leastCC"}},
                {"type": "sentinel-2-l2a", "id": "scl", "dataFilter": {"timeRange": {"from": time_from, "to": time_to}, "mosaickingOrder": "leastCC"}},
            ],
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
    cache_id = hashlib.sha256(json.dumps(["s5p_ch4_v6", aoi, str(date_from), str(date_to)], sort_keys=True).encode()).hexdigest()[:24]
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
    finite = np.logical_and.reduce([np.isfinite(bands[b]) for b in ["B03", "B04", "B08", "B11", "B12"]])
    scl = bands.get("SCL")
    if scl is not None:
        scl_int = np.round(np.nan_to_num(scl, nan=0.0)).astype(np.int32)
        finite &= ~np.isin(scl_int, [0, 1, 3, 8, 9, 10, 11])
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
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0.05) & (b12 > 0.05)
    if use.sum() < 100:
        return 1.0
    x = b12[use].astype(np.float64)
    y = b11[use].astype(np.float64)
    denom = float(np.sum(x * x))
    return float(np.sum(y * x) / denom) if denom > 0 else 1.0


def calculate_delta_R(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0.05) & (b11 < 0.9) & (b12 > 0.05) & (b12 < 0.9)
    output[use] = (c * b12[use] - b11[use]) / b12[use]
    return output


def remove_large_scale_background(domega, valid_mask, sigma=DETREND_SIGMA):
    finite_valid = valid_mask & np.isfinite(domega)
    data = np.where(finite_valid, domega, 0.0).astype(np.float64)
    weights = finite_valid.astype(np.float32)
    ds = gaussian_filter(data, sigma=sigma)
    ws = gaussian_filter(weights, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        background = ds / ws
    background[ws < 0.1] = 0.0
    residual = domega - background
    residual[~valid_mask] = np.nan
    return residual.astype(np.float32), background.astype(np.float32)


def normalized_gaussian(data, valid_mask, sigma):
    finite_valid = valid_mask & np.isfinite(data)
    values = np.where(finite_valid, data, 0.0).astype(np.float64)
    weights = finite_valid.astype(np.float32)
    ds = gaussian_filter(values, sigma=sigma)
    ws = gaussian_filter(weights, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = ds / ws
    output[ws < 0.1] = np.nan
    return output.astype(np.float32)


def make_spatial_mask(shape_, profile, lat=SITE_LAT, lon=SITE_LON, radius_m=SITE_RADIUS_M):
    transform = profile["transform"]
    crs = profile["crs"]
    try:
        xs, ys = rio_transform("EPSG:4326", crs, [lon], [lat])
    except Exception:
        xs, ys = [lon], [lat]
    row, col = rasterio.transform.rowcol(transform, xs[0], ys[0])
    row, col = int(row), int(col)
    h, w = shape_
    if not (0 <= row < h and 0 <= col < w):
        return np.ones(shape_, dtype=bool), (row, col)
    rows, cols = np.ogrid[:h, :w]
    px, py = abs(transform.a), abs(transform.e)
    dist_m = np.sqrt(((rows - row) * py) ** 2 + ((cols - col) * px) ** 2)
    return dist_m <= radius_m, (row, col)


def robust_stats(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    lo, hi = np.percentile(finite, [PARAMS["trim_low_pct"], PARAMS["trim_high_pct"]])
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    sigma_trim = float(np.std(trimmed)) / 0.7817 if trimmed.size > 10 else 0.0
    mad = float(np.median(np.abs(finite - median)))
    sigma_mad = 1.4826 * mad if mad > 1e-9 else float(np.std(finite))
    sigma = min(sigma_trim, sigma_mad) if sigma_trim > 0 else sigma_mad
    return median, float(sigma if np.isfinite(sigma) and sigma > 0 else 1.0)


def run_mbmc_algorithm(target, reference, profile):
    target_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"]))
    valid = calculate_lrad(target, target_q)
    if valid.sum() < 100:
        raise RuntimeError("LRAD removed nearly all pixels. Relax thresholds or enlarge AOI.")
    c = calculate_c(target["B11"], target["B12"], valid)
    dr_t = calculate_delta_R(target["B11"], target["B12"], c, valid)
    dr_r = calculate_delta_R(reference["B11"], reference["B12"], c, valid)
    domega = dr_t / K_MBMP - dr_r / K_MBMP
    domega[~valid] = np.nan
    detrended, _ = remove_large_scale_background(domega, valid)
    smooth = normalized_gaussian(detrended, valid, GAUSS_SIGMA)
    vals = smooth[np.isfinite(smooth)]
    if vals.size == 0:
        raise RuntimeError("No finite values after detrending.")
    median, sigma = robust_stats(vals)
    threshold_sigma = median + max(PARAMS["threshold_sigma"] * sigma, PARAMS["abs_floor_ppb"])
    threshold_pct = float(np.percentile(vals, PARAMS["threshold_percentile_cap"]))
    threshold = max(min(threshold_sigma, threshold_pct), median + PARAMS["abs_floor_ppb"])
    candidate = np.isfinite(smooth) & (smooth > threshold) & valid
    spatial_mask, site_rc = make_spatial_mask(domega.shape, profile)
    candidate &= spatial_mask
    labeled, n_labels = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    plume = np.zeros_like(candidate, dtype=bool)
    if n_labels:
        sizes = np.bincount(labeled.ravel(), minlength=n_labels + 1)
        sizes[0] = 0
        keep = np.where(sizes >= PARAMS["min_component_pixels"], sizes, 0)
        if keep.max() > 0:
            order = np.argsort(keep)[::-1]
            order = order[order != 0]
            plume = labeled == int(order[0])
            if plume.sum() * RESOLUTION * RESOLUTION / 1e6 > PARAMS["max_plume_area_km2"]:
                plume[:] = False
    if plume.any() and PARAMS["final_dilation"] > 0:
        plume = binary_dilation(plume, structure=disk(int(PARAMS["final_dilation"])))
    plume &= valid & spatial_mask
    return {
        "relative": domega, "detrended": detrended, "gaussian": smooth,
        "valid": valid, "spatial_mask": spatial_mask, "initial": candidate,
        "final": plume, "mean": median, "std": sigma, "threshold": threshold,
        "regions": int(1 if plume.any() else 0), "valid_count": int(valid.sum()),
        "initial_count": int(candidate.sum()), "final_count": int(plume.sum()),
        "c": float(c), "site_rc": site_rc,
        "diagnostics": {"median_ppb": median, "sigma_ppb": sigma, "threshold_final_ppb": threshold,
                         "valid_pixels": int(vals.size), "above_threshold": int((vals > threshold).sum())},
    }


# ------------------------- Research pipeline -------------------------
if TORCH_AVAILABLE:
    class MultiSpectralAugment:
        def __init__(self, crop_size=128, cutout_probability=0.4):
            self.crop_size = crop_size
            self.cutout_probability = cutout_probability

        def __call__(self, x):
            _, h, w = x.shape
            crop = min(self.crop_size, h, w)
            if h > crop:
                top = int(torch.randint(0, h - crop + 1, (1,)).item())
            else:
                top = 0
            if w > crop:
                left = int(torch.randint(0, w - crop + 1, (1,)).item())
            else:
                left = 0
            x = x[:, top:top + crop, left:left + crop]
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            x = torch.rot90(x, k, dims=[1, 2])
            if torch.rand(()) < 0.5:
                shift_y = int(torch.randint(-crop // 8, crop // 8 + 1, (1,)).item())
                shift_x = int(torch.randint(-crop // 8, crop // 8 + 1, (1,)).item())
                x = torch.roll(x, shifts=(shift_y, shift_x), dims=(1, 2))
            if torch.rand(()) < self.cutout_probability:
                ch = max(1, crop // 6)
                cw = max(1, crop // 6)
                yy = int(torch.randint(0, max(1, crop - ch + 1), (1,)).item())
                xx = int(torch.randint(0, max(1, crop - cw + 1), (1,)).item())
                x[:, yy:yy + ch, xx:xx + cw] = 0
            return x

    class ArrayImageDataset(Dataset):
        def __init__(self, arrays, labels=None, augment=None, crop_size=128):
            self.arrays = arrays
            self.labels = labels
            self.augment = augment
            self.crop_size = crop_size

        def __len__(self):
            return len(self.arrays)

        def _tensor(self, value):
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 2:
                value = value[None]
            value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            t = torch.from_numpy(value)
            return t

        def __getitem__(self, index):
            x = self._tensor(self.arrays[index])
            if self.augment is not None:
                return self.augment(x), self.augment(x.clone())
            if self.labels is None:
                return x
            y = torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32))[None]
            if x.shape[-2:] != y.shape[-2:]:
                y = F.interpolate(y[None], size=x.shape[-2:], mode="nearest")[0]
            return x, y

    class TinySpatialEncoder(nn.Module):
        """A dependency-free MobileNet-like encoder for multispectral input."""
        def __init__(self, in_channels=4, width=32):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
                nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
                nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 8), nn.ReLU(inplace=True),
            )
            self.out_dim = width * 8

        def forward(self, x):
            return self.features(x)

    class SimCLRModel(nn.Module):
        def __init__(self, in_channels=4, projection_dim=256):
            super().__init__()
            self.encoder = TinySpatialEncoder(in_channels=in_channels)
            self.pool = nn.AdaptiveAvgPool2d(1)
            d = self.encoder.out_dim
            self.projector = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, projection_dim))

        def forward(self, x):
            feature_map = self.encoder(x)
            h = self.pool(feature_map).flatten(1)
            return self.projector(h), feature_map

    class SegmentationModel(nn.Module):
        def __init__(self, encoder, in_channels=4):
            super().__init__()
            self.encoder = encoder
            c = encoder.out_dim
            self.decoder = nn.Sequential(
                nn.Conv2d(c, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            )

        def forward(self, x):
            z = self.decoder(self.encoder(x))
            return F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)


def nt_xent(z1, z2, temperature=0.1):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    n = z.shape[0]
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-1e9)
    targets = (torch.arange(n, device=z.device) + n // 2) % n
    return F.cross_entropy(logits, targets)


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum()
    dice = (2 * intersection + 1e-6) / (probability.sum() + target.sum() + 1e-6)
    return bce + 1.0 - dice


def make_simclr_input(target, reference):
    values = np.stack([target["B11"], target["B12"], reference["B11"], reference["B12"]], axis=0).astype(np.float32)
    finite = np.isfinite(values)
    values[~finite] = 0.0
    for i in range(values.shape[0]):
        valid = values[i][finite[i]]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            if hi > lo:
                values[i] = np.clip((values[i] - lo) / (hi - lo), 0, 1)
    return values


def train_simclr_on_scene(target, reference, device, epochs=20, batch_size=4, lr=1e-3, temperature=0.1):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Install PyTorch first: pip install torch torchvision")
    x = make_simclr_input(target, reference)
    h, w = x.shape[-2:]
    patch = min(128, h, w)
    patches = []
    stride = max(32, patch // 2)
    for y in range(0, max(1, h - patch + 1), stride):
        for xx in range(0, max(1, w - patch + 1), stride):
            patches.append(x[:, y:y + patch, xx:xx + patch])
    if not patches:
        patches = [x]
    dataset = ArrayImageDataset(patches, augment=MultiSpectralAugment(crop_size=patch))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SimCLRModel(in_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(int(epochs)):
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, _ = model(v1)
            z2, _ = model(v2)
            loss = nt_xent(z1, z2, temperature=temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    checkpoint = MODEL_DIR / "simclr_latest.pt"
    torch.save({"encoder": model.encoder.state_dict(), "in_channels": 4, "epochs": int(epochs), "temperature": float(temperature)}, checkpoint)
    return checkpoint, float(loss.detach().cpu())


def infer_simclr_segmentation(target, reference, device, checkpoint=None, threshold=0.5):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Install PyTorch first: pip install torch torchvision")
    x = make_simclr_input(target, reference)
    model_ssl = SimCLRModel(in_channels=4).to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model_ssl.encoder.load_state_dict(state["encoder"], strict=False)
    model = SegmentationModel(model_ssl.encoder, in_channels=4).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None]).to(device))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = probability >= float(threshold)
    return {"probability": probability, "mask": mask, "threshold": float(threshold), "model": model}


def calculate_binary_metrics(prediction, truth):
    p = np.asarray(prediction).astype(bool)
    y = np.asarray(truth).astype(bool)
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "IoU": iou, "F1": f1}


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    output = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output, 1)


def image_png(array, mask=False):
    if Image is None:
        raise RuntimeError("Pillow is required for PNG previews.")
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
                norm = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
                import matplotlib.pyplot as plt
                rgb = (plt.get_cmap("RdBu_r")(norm)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def create_png_worldfile(profile):
    transform = profile["transform"]
    pgw = f"{transform.a:.12f}\n0.0\n0.0\n{transform.e:.12f}\n{transform.c + transform.a/2:.12f}\n{transform.f + transform.e/2:.12f}\n"
    prj = profile["crs"].to_wkt() if profile.get("crs") else ""
    return pgw.encode(), prj.encode()


def georeferenced_png_package(array, profile, mask=False):
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", image_png(array, mask=mask))
        pgw, prj = create_png_worldfile(profile)
        archive.writestr("output.pgw", pgw)
        if prj:
            archive.writestr("output.prj", prj)
    return package.getvalue()


def create_map(aoi, search_center=None, search_zoom=11):
    geometry = shape(ensure_aoi(aoi))
    center = [float(search_center[0]), float(search_center[1])] if search_center else [geometry.centroid.y, geometry.centroid.x]
    fmap = folium.Map(center, zoom_start=int(search_zoom if search_center else 11), tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    if search_center:
        folium.Marker([float(search_center[0]), float(search_center[1])], tooltip="Searched location").add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon:", num_digits=5).add_to(fmap)
    return fmap


def legend_html(kind):
    if kind == "mask":
        rows = [("#dc1e1e", "Methane candidate"), ("#000000", "Background")]
    elif kind == "comparison":
        rows = [("#dc1e1e", "Existing only"), ("#1f77b4", "SimCLR only"), ("#9b59b6", "Both")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>' for c, t in rows)
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def add_research_ui(target, target_bands, reference_bands, profile, mbmc_result):
    st.markdown('<div class="section-label">04a · PAPER METHOD</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">SimCLR Sentinel-2 segmentation</div>', unsafe_allow_html=True)
    st.caption("The existing MBMC pipeline remains unchanged. This optional path uses the paper-inspired 4-channel two-pass input [B11_t, B12_t, B11_ref, B12_ref].")
    if not TORCH_AVAILABLE:
        st.warning("PyTorch is not installed. Install it with: pip install torch torchvision")
        return None
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        ssl_epochs = st.number_input("SimCLR epochs", 1, 500, 20, 1, key="ssl_epochs")
    with c2:
        ssl_batch = st.number_input("SSL batch", 1, 64, 4, 1, key="ssl_batch")
    with c3:
        ssl_temp = st.number_input("Temperature τ", 0.01, 2.0, 0.10, 0.01, key="ssl_temp")
    with c4:
        model_threshold = st.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.05, key="simclr_threshold")
    action1, action2 = st.columns(2, gap="small")
    with action1:
        train_clicked = st.button("🧠 Train SimCLR + run segmentation", type="primary", use_container_width=True, key="train_simclr_button")
    with action2:
        infer_clicked = st.button("▶ Use cached SimCLR checkpoint", use_container_width=True, key="infer_simclr_button")
    if train_clicked or infer_clicked:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = MODEL_DIR / "simclr_latest.pt"
            loss_value = None
            if train_clicked:
                with st.spinner(f"Training SimCLR on {device}…"):
                    checkpoint, loss_value = train_simclr_on_scene(target_bands, reference_bands, device, int(ssl_epochs), int(ssl_batch), temperature=float(ssl_temp))
            if not checkpoint.exists():
                raise RuntimeError("No cached checkpoint exists. Train SimCLR first.")
            with st.spinner("Running segmentation decoder…"):
                simclr_result = infer_simclr_segmentation(target_bands, reference_bands, device, checkpoint, float(model_threshold))
            simclr_mask = simclr_result["mask"]
            comparison = np.zeros_like(simclr_mask, dtype=np.uint8)
            existing = mbmc_result["final"]
            comparison[existing & ~simclr_mask] = 1
            comparison[~existing & simclr_mask] = 2
            comparison[existing & simclr_mask] = 3
            output_folder = RESULT_DIR / "simclr"
            output_folder.mkdir(parents=True, exist_ok=True)
            simclr_path = output_folder / "simclr_probability.tif"
            mask_path = output_folder / "simclr_mask.tif"
            compare_path = output_folder / "method_comparison.tif"
            save_raster(simclr_path, simclr_result["probability"], profile, False)
            save_raster(mask_path, simclr_mask, profile, True)
            save_raster(compare_path, comparison, profile, False)
            result = {"probability": simclr_result["probability"], "mask": simclr_mask, "comparison": comparison, "paths": {"probability": simclr_path, "mask": mask_path, "comparison": compare_path}, "checkpoint": checkpoint, "loss": loss_value}
            st.session_state.simclr_result = result
            st.success(f"SimCLR segmentation complete · device={device}" + (f" · final SSL loss={loss_value:.4f}" if loss_value is not None else ""))
        except Exception as error:
            st.error(f"SimCLR pipeline failed: {error}")
    result = st.session_state.get("simclr_result")
    if result:
        mask = result["mask"]
        c = st.columns(5, gap="small")
        c[0].metric("SimCLR pixels", f"{int(mask.sum()):,}")
        c[1].metric("Existing pixels", f"{int(mbmc_result['final'].sum()):,}")
        c[2].metric("Intersection", f"{int((mask & mbmc_result['final']).sum()):,}")
        c[3].metric("SimCLR area km²", f"{mask.sum() * RESOLUTION * RESOLUTION / 1e6:.4f}")
        c[4].metric("Checkpoint", result["checkpoint"].name)
        a, b, d = st.columns(3, gap="small")
        with a:
            st.image(image_png(result["probability"]), use_container_width=True)
            st.caption("SimCLR decoder probability")
        with b:
            st.image(image_png(mask, mask=True), use_container_width=True)
            st.caption("SimCLR plume mask")
        with d:
            st.image(image_png(result["comparison"], mask=False), use_container_width=True)
            st.caption("Comparison raster; download GeoTIFF for exact class values")
        for key, label, mime in [("probability", "Download SimCLR probability", "image/tiff"), ("mask", "Download SimCLR mask", "image/tiff"), ("comparison", "Download method comparison", "image/tiff")]:
            path = result["paths"][key]
            st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=f"download_simclr_{key}")
    return result


# ------------------------- CSS and application -------------------------
st.set_page_config(page_title="Sentinel-2 Methane", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root { --red:#e63946; --honeydew:#f1faee; --frost:#a8dadc; --blue:#457b9d; --navy:#1d3557; --border:#d8e6e8; --dark-field:#292a33; }
.stApp { background:#f1faee; color:#111 !important; }
[data-testid="stHeader"] { background:#f1faee !important; height:3.25rem !important; }
[data-testid="stSidebar"] { display:none; }
.block-container { max-width:1700px; padding-top:3.9rem !important; padding-bottom:.8rem; padding-left:1.2rem; padding-right:1.2rem; }
.app-header { display:flex; align-items:center; justify-content:space-between; background:#fff; border:1px solid var(--border); border-radius:16px; padding:.75rem 1rem; margin-top:.15rem; margin-bottom:.9rem; box-shadow:0 2px 10px rgba(29,53,87,.05); }
.app-title { color:#111 !important; font-size:1.45rem; font-weight:850; line-height:1.1; }
.app-subtitle,.card-caption { color:#111 !important; font-size:.78rem; }
.status-pill { background:#f1faee; color:#111 !important; border:1px solid #a8dadc; border-radius:999px; padding:.35rem .7rem; font-size:.72rem; font-weight:750; }
.app-card { background:#fff; border:1px solid var(--border); border-radius:15px; padding:.75rem; box-shadow:0 2px 10px rgba(29,53,87,.04); height:100%; color:#111 !important; }
.card-title { color:#111 !important; font-size:1rem; font-weight:800; margin-bottom:.1rem; }
.section-label { display:inline-block; background:#a8dadc; color:#111 !important; border-radius:999px; padding:.2rem .55rem; font-size:.65rem; font-weight:800; letter-spacing:.03em; margin-bottom:.35rem; }
.stApp p,.stApp label,.stApp small,.stApp strong,.stApp li,.stApp td,.stApp th,.stApp [data-testid="stMarkdownContainer"] { color:#111 !important; }
.stButton>button,.stDownloadButton>button { border-radius:9px; min-height:2.15rem; font-weight:750; font-size:.78rem; color:#111 !important; }
.stButton>button[kind="primary"] { background:#e63946; border-color:#e63946; color:#fff !important; }
.stButton>button[kind="primary"] * { color:#fff !important; }
.stDownloadButton>button { background:#fff; border:1px solid #a8dadc; }
.result-legend { background:#fff; border:1px solid #d7e4e7; border-radius:10px; padding:.75rem .7rem; min-height:96px; }
.legend-heading { color:#111 !important; font-size:.88rem; font-weight:800; }
.legend-row { display:flex; align-items:center; gap:.45rem; color:#111 !important; font-size:.82rem; line-height:1.25; margin-top:.42rem; }
.legend-swatch { width:18px; height:14px; min-width:18px; border:1px solid #555; border-radius:2px; display:inline-block; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header"><div><div class="app-title">🛰️ Sentinel-2 Methane Screening</div><div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; Existing MBMC + optional SimCLR paper pipeline &nbsp;|&nbsp; v6</div></div><div class="status-pill">20 m processing &nbsp;•&nbsp; Light dashboard</div></div>
""", unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)

map_col, control_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Search a location or draw the study area directly on the map.</div>', unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([4, 1], gap="small")
    with search_col1:
        search_query = st.text_input("Location search", placeholder="🔎  Search a place", key="location_search_input", label_visibility="collapsed")
    with search_col2:
        search_clicked = st.button("🔍 Find", use_container_width=True, key="search_location_btn")
    if search_clicked:
        if not (search_query or "").strip():
            st.warning("Please type a location name first.")
        else:
            with st.spinner("Searching location…"):
                geo = geocode_location(search_query.strip())
            if geo:
                st.session_state["search_center"] = [geo[0], geo[1]]
                st.session_state["search_name"] = geo[2]
                st.session_state["search_zoom"] = 12
            else:
                st.warning("Location not found.")
    map_data = st_folium(create_map(st.session_state.aoi, st.session_state.get("search_center"), st.session_state.get("search_zoom", 11)), height=385, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi:
            st.session_state.aoi = new_aoi
    geom = shape(ensure_aoi(st.session_state.aoi))
    st.markdown(f'<div class="card-caption">AOI center: Lat {geom.centroid.y:.5f} · Lon {geom.centroid.x:.5f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)
    end_default = datetime.now().date()
    start_default = end_default - timedelta(days=30)
    d1, d2 = st.columns(2)
    with d1: start_date = st.date_input("Start date", start_default, key="start_date")
    with d2: end_date = st.date_input("End date", end_default, key="end_date")
    s1, s2 = st.columns(2)
    with s1: max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 30.0, key="max_cloud")
    with s2: reference_days = st.slider("Reference window (days)", 1, 90, 60, key="reference_days")
    if st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC…"):
                st.session_state.scene_results = search_scenes(st.session_state.aoi, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), max_cloud)
            st.session_state.pop("target", None)
            st.success(f"{len(st.session_state.scene_results)} scene(s) found")
        except Exception as error:
            st.session_state.scene_results = []
            st.error(f"Scene search failed: {error}")
    scene_results = st.session_state.get("scene_results", [])
    if scene_results:
        table = pd.DataFrame([{"date": get_datetime(s), "tile": get_tile(s), "cloud": get_cloud(s)} for s in scene_results]).sort_values(["date", "cloud"], ascending=[True, True])
        st.dataframe(table, use_container_width=True, height=112, hide_index=True)
        ids = [as_dict(s).get("id") for s in scene_results if as_dict(s).get("id")]
        selected_id = st.selectbox("Target scene", ids, format_func=lambda x: f"{get_datetime(next(s for s in scene_results if as_dict(s).get('id') == x)).strftime('%Y-%m-%d')} | {get_tile(next(s for s in scene_results if as_dict(s).get('id') == x))}", key="target_scene_select")
        st.session_state.target = next(s for s in scene_results if as_dict(s).get("id") == selected_id)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1: PARAMS["threshold_sigma"] = st.number_input("Threshold multiplier", .1, 6.0, float(PARAMS["threshold_sigma"]), .1, key="threshold_sigma")
    with p2: PARAMS["min_component_pixels"] = st.number_input("Minimum candidate pixels", 2, 1000, int(PARAMS["min_component_pixels"]), 5, key="min_component_pixels")
    with p3: PARAMS["final_dilation"] = st.number_input("Final dilation radius", 0, 20, int(PARAMS["final_dilation"]), 1, key="final_dilation")
    st.markdown(f'<div class="card-caption">Minimum connected region ≈ {int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION:,} m² at {RESOLUTION} m.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    auth = st.session_state.get("cdse_auth")
    if auth:
        st.success(f"Copernicus connected · {auth.get('username', '')}")
        if st.button("Log out", key="cdse_logout"):
            st.session_state.pop("cdse_auth", None)
            st.rerun()
    else:
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.caption("Only the temporary API token is kept in this session.")
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)
        if login_submitted:
            try:
                st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful.")
                st.rerun()
            except Exception as error:
                st.error(str(error))
    scene_results = st.session_state.get("scene_results", [])
    if scene_results and st.session_state.get("target") is not None:
        target_scene = st.session_state.target
        target_date = get_datetime(target_scene)
        if st.button("🛰️  Download AOI & Detect Methane", type="primary", use_container_width=True, disabled=not bool(st.session_state.get("cdse_auth")), key="detect_button"):
            try:
                token = get_access_token()
                if target_date is None:
                    raise RuntimeError("Target date is missing.")
                refs = [s for s in scene_results if as_dict(s).get("id") != as_dict(target_scene).get("id") and get_datetime(s) and abs((get_datetime(s)-target_date).total_seconds())/86400 <= reference_days]
                same_tile = [s for s in refs if get_tile(s) == get_tile(target_scene)]
                refs = same_tile if same_tile else refs
                if not refs:
                    raise RuntimeError("No reference scene. Increase date range or reference window.")
                with st.spinner("Downloading target and reference scenes…"):
                    target_bands, profile = read_stack(download_scene(target_scene, st.session_state.aoi, token))
                    best_ref, best_corr, best_valid = None, -np.inf, 0
                    reference_rows = []
                    for ref in refs:
                        try:
                            ref_bands, _ = read_stack(download_scene(ref, st.session_state.aoi, token))
                        except Exception:
                            continue
                        valid_px = np.isfinite(target_bands["B04"]) & np.isfinite(ref_bands["B04"])
                        count = int(valid_px.sum())
                        corr = np.nan
                        if count >= PARAMS["min_valid_ref_pixels"]:
                            a, b = target_bands["B04"][valid_px], ref_bands["B04"][valid_px]
                            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                                corr = float(np.corrcoef(a, b)[0, 1])
                        reference_rows.append({"date": get_datetime(ref), "tile": get_tile(ref), "b4_correlation": corr, "valid_b4_pixels": count})
                        if np.isfinite(corr) and corr > best_corr:
                            best_ref, best_corr, best_valid = ref_bands, corr, count
                        elif best_ref is None and count > best_valid:
                            best_ref, best_valid = ref_bands, count
                if best_ref is None:
                    raise RuntimeError("Could not select reference scene.")
                with st.spinner("Running existing MBMC algorithm…"):
                    mbmc_result = run_mbmc_algorithm(target_bands, best_ref, profile)
                st.session_state.result = mbmc_result
                st.session_state.target_bands = target_bands
                st.session_state.reference_bands = best_ref
                st.session_state.output_profile = profile
                st.session_state.reference_table = pd.DataFrame(reference_rows)
                output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                output_folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key in ("relative", "detrended", "gaussian", "final", "valid"):
                    paths[key] = output_folder / f"{key}.tif"
                    save_raster(paths[key], mbmc_result[key], profile, key in ("final", "valid"))
                st.session_state.paths = paths
                st.success("Existing processing completed. The paper pipeline is available below.")
            except Exception as error:
                st.error(f"Detection failed: {error}")
    else:
        st.caption("Search scenes and select a target scene first.")
    st.markdown('</div>', unsafe_allow_html=True)

if "result" in st.session_state:
    result = st.session_state.result
    profile = st.session_state.output_profile
    st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)
    metrics = st.columns(6)
    metrics[0].metric("B4 correlation", f"{st.session_state.get('reference_table', pd.DataFrame()).b4_correlation.max():.3f}" if not st.session_state.get('reference_table', pd.DataFrame()).empty and np.isfinite(st.session_state.reference_table.b4_correlation.max()) else "n/a")
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Initial", f"{result['initial_count']:,}")
    metrics[3].metric("Final", f"{result['final_count']:,}")
    metrics[4].metric("Regions", result["regions"])
    metrics[5].metric("Threshold", f"{result['threshold']:.2f}")
    st.markdown('<div class="card-caption">Existing MBMC result remains the reference screening output.</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    for col, key, title, is_mask in [(t1, "detrended", "ΔΩ after detrend", False), (t2, "gaussian", "Gaussian smoothed", False), (t3, "final", "Existing methane mask", True), (t4, "valid", "Validity mask", True)]:
        with col:
            st.image(image_png(result[key], mask=is_mask), use_container_width=True)
            st.caption(title)
            path = st.session_state.paths[key]
            st.download_button("⬇ GeoTIFF", path.read_bytes(), file_name=path.name, mime="image/tiff", key=f"download_existing_{key}")
    if "target_bands" in st.session_state and "reference_bands" in st.session_state:
        st.markdown('<div style="height:.3rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        sim_result = add_research_ui(st.session_state.target, st.session_state.target_bands, st.session_state.reference_bands, profile, result)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Keep Sentinel-5P section available and independent.
if "result" in st.session_state:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">06 · SENTINEL-5P CH4 CONTEXT</div>', unsafe_allow_html=True)
    st.caption("This context layer is preserved as an optional independent operation.")
    if st.button("🛰️ Fetch S5P CH4", type="primary", disabled=not bool(st.session_state.get("cdse_auth")), key="s5p_button"):
        try:
            token = get_access_token()
            target_date = get_datetime(st.session_state.target)
            s5p_path = download_s5p_scene(st.session_state.aoi, target_date - timedelta(days=15), target_date + timedelta(days=15), token)
            with rasterio.open(s5p_path) as src:
                ch4 = src.read(1).astype(np.float32)
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[ch4 <= 0] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.success("S5P CH4 loaded")
        except Exception as error:
            st.error(f"S5P fetch failed: {error}")
    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        valid = ch4[np.isfinite(ch4)]
        if valid.size:
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean CH4 (ppb)", f"{np.nanmean(valid):.1f}")
            m2.metric("Min CH4 (ppb)", f"{np.nanmin(valid):.1f}")
            m3.metric("Max CH4 (ppb)", f"{np.nanmax(valid):.1f}")
            st.image(image_png(ch4), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
