"""Complete Streamlit Sentinel-2 methane candidate screening app."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
from folium.plugins import Draw
from rasterio.transform import Affine
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    gaussian_filter,
    generate_binary_structure,
    label,
)
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

PARAMS = {
    "b03_quantile": 0.03,
    "reference_max_days": 90,
    "water_mndwi": 0.12,
    "water_ndwi": 0.28,
    "cloud_ndsi": 0.40,
    "cloud_ndwi": 0.35,
    "cloud_bright_quantile": 0.995,
    "edge_margin_pixels": 8,
    "background_sigma": 30.0,
    "smoothing_sigma": 0.8,
    "threshold_z": 3.0,
    "absolute_floor": 0.0008,
    "min_component_pixels": 12,
    "max_component_pixels": 2500,
    "max_candidate_fraction": 0.01,
    "final_dilation": 1,
}


def as_dict(item):
    if isinstance(item, dict): return item
    if hasattr(item, "to_dict"): return item.to_dict()
    return dict(item)


def get_properties(item): return as_dict(item).get("properties", {}) or {}


def get_datetime(item) -> Optional[datetime]:
    data, props = as_dict(item), get_properties(item)
    value = props.get("datetime") or props.get("start_datetime")
    if value:
        try: return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError: pass
    match = re.search(r"_(\d{8}T\d{6})_", data.get("id", "").upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data, props = as_dict(item), get_properties(item)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key): return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", data.get("id", "").upper())
    return match.group(1) if match else None


def get_cloud(item):
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try: return float(get_properties(item)[key])
        except Exception: pass
    return 100.0


def normalize_geometry(obj):
    if obj is None: return None
    if hasattr(obj, "__geo_interface__"): obj = obj.__geo_interface__
    if not isinstance(obj, dict): return None
    if obj.get("type") == "Feature": return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geoms = [shape(g) for f in obj.get("features", []) if (g := normalize_geometry(f.get("geometry")))]
        return mapping(unary_union(geoms)) if geoms else None
    try:
        geom = shape(obj)
        return mapping(geom) if not geom.is_empty else None
    except Exception: return None


def ensure_aoi(obj): return normalize_geometry(obj) or mapping(box(48.0, 29.0, 49.0, 30.0))


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
    return response.json().get("features", [])


def authenticate_cdse(username, password, totp=""):
    if not username.strip() or not password.strip(): raise RuntimeError("Enter Copernicus email and password.")
    form = {"grant_type": "password", "client_id": "cdse-public", "username": username.strip(), "password": password.strip()}
    if totp.strip(): form["totp"] = totp.strip()
    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try: detail = response.json().get("error_description") or response.json().get("error")
        except Exception: detail = None
        raise RuntimeError(detail or f"Copernicus login failed: HTTP {response.status_code}")
    data = response.json()
    if not data.get("access_token"): raise RuntimeError("No access token returned.")
    return {"access_token": data["access_token"], "refresh_token": data.get("refresh_token", ""), "expires_at": time.time() + int(data.get("expires_in", 600)), "username": username.strip()}


def refresh_cdse_session(auth):
    if not auth.get("refresh_token"): return None
    response = requests.post(TOKEN_URL, data={"grant_type": "refresh_token", "client_id": "cdse-public", "refresh_token": auth["refresh_token"]}, timeout=90)
    if response.status_code >= 400: return None
    data = response.json()
    if not data.get("access_token"): return None
    auth.update({"access_token": data["access_token"], "refresh_token": data.get("refresh_token", auth["refresh_token"]), "expires_at": time.time() + int(data.get("expires_in", 600))})
    return auth


def get_access_token():
    auth = st.session_state.get("cdse_auth")
    if not auth: raise RuntimeError("Please log in to Copernicus first.")
    if time.time() < float(auth.get("expires_at", 0)) - 60: return auth["access_token"]
    auth = refresh_cdse_session(auth)
    if auth:
        st.session_state.cdse_auth = auth
        return auth["access_token"]
    st.session_state.pop("cdse_auth", None)
    raise RuntimeError("Copernicus session expired. Log in again.")


def evalscript():
    return '''//VERSION=3
function setup() {
  return {input: [{bands: ["B03","B04","B08","B11","B12"], units: "REFLECTANCE"}], output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(sample) { return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12]; }
'''


def download_scene(item, aoi, token):
    item, aoi = as_dict(item), ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder, output = CACHE_DIR / cache_id, CACHE_DIR / cache_id / "bands.tif"
    if output.exists(): return output
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    lat = math.radians((miny + maxy) / 2)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(lat) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acquisition = get_datetime(item)
    if acquisition is None: raise RuntimeError("Scene acquisition date is unavailable.")
    payload = {
        "input": {"bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}]},
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=payload, timeout=900)
    if response.status_code >= 400: raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {response.text[:1000]}")
    output.write_bytes(response.content)
    return output


def read_stack(path):
    with rasterio.open(path) as src:
        values, profile = src.read().astype(np.float32), src.profile.copy()
    return {b: values[i] for i, b in enumerate(BANDS)}, profile


def nd(a, b):
    out = np.full(a.shape, np.nan, dtype=np.float32)
    den = a + b
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(den) > 1e-8)
    out[good] = (a[good] - b[good]) / den[good]
    return out


def scene_features(b):
    return {"ndvi": nd(b["B08"], b["B04"]), "ndwi": nd(b["B03"], b["B08"]), "mndwi": nd(b["B03"], b["B11"]), "ndsi": nd(b["B03"], b["B11"])}


def l1c_quality_mask(b):
    shape_ = b["B11"].shape
    finite = np.logical_and.reduce([np.isfinite(b[x]) for x in BANDS])
    for x in BANDS: finite &= (b[x] >= 0) & (b[x] <= 1.2)
    finite &= (b["B03"] > 1e-5) & (b["B11"] > 1e-5)
    f = scene_features(b)
    bright = np.nanmedian(np.stack([b["B03"], b["B04"], b["B08"]]), axis=0)
    bright_values = bright[np.isfinite(bright)]
    bright_limit = np.quantile(bright_values, PARAMS["cloud_bright_quantile"]) if bright_values.size else np.inf
    cloud = (bright >= bright_limit) | (f["ndsi"] > PARAMS["cloud_ndsi"]) | (f["ndwi"] > PARAMS["cloud_ndwi"])
    water = (f["mndwi"] > PARAMS["water_mndwi"]) | (f["ndwi"] > PARAMS["water_ndwi"])
    saturation = (b["B11"] >= 1.0) | (b["B12"] >= 1.0)
    edge = np.ones(shape_, dtype=bool)
    m = PARAMS["edge_margin_pixels"]
    edge[:m] = edge[-m:] = False
    edge[:, :m] = edge[:, -m:] = False
    return finite & ~cloud & ~water & ~saturation & edge


def stable_pair_mask(t, r):
    valid = l1c_quality_mask(t) & l1c_quality_mask(r)
    for name in BANDS:
        ratio = np.abs(t[name] - r[name]) / np.maximum(r[name], 0.02)
        valid &= ratio < 0.75
    return valid


def calculate_c(b11, b12, valid):
    use = valid & (b11 > 0.02) & (b12 > 0.02) & (b11 < 1.0) & (b12 < 1.0)
    if use.sum() < 100: return 1.0
    x, y = b12[use].astype(np.float64), b11[use].astype(np.float64)
    return float(np.sum(x * y) / max(np.sum(x * x), 1e-20))


def mbsp(b11, b12, c, valid):
    out = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (np.abs(b12) > 1e-8)
    out[use] = ((c * b12[use] - b11[use]) / b12[use]).astype(np.float32)
    return out


def background_remove(arr, valid):
    values = np.where(valid & np.isfinite(arr), arr, 0).astype(np.float32)
    weights = (valid & np.isfinite(arr)).astype(np.float32)
    sigma = PARAMS["background_sigma"]
    sv, sw = gaussian_filter(values, sigma), gaussian_filter(weights, sigma)
    bg = np.zeros_like(arr, dtype=np.float32)
    good = sw > 0.05
    bg[good] = sv[good] / sw[good]
    residual = arr - bg
    residual[~valid] = np.nan
    return residual, bg


def run_algorithm(target, reference):
    valid = stable_pair_mask(target, reference)
    if valid.sum() < 100: raise RuntimeError("Too few stable pixels after quality masking.")
    ct, cr = calculate_c(target["B11"], target["B12"], valid), calculate_c(reference["B11"], reference["B12"], valid)
    relative = mbsp(target["B11"], target["B12"], ct, valid) - mbsp(reference["B11"], reference["B12"], cr, valid)
    relative[~valid] = np.nan
    residual, background = background_remove(relative, valid)
    smooth = gaussian_filter(np.where(valid & np.isfinite(residual), residual, 0), PARAMS["smoothing_sigma"])
    smooth[~valid] = np.nan
    values = smooth[valid & np.isfinite(smooth)]
    med = float(np.median(values))
    mad = max(float(np.median(np.abs(values - med))) * 1.4826, 1e-8)
    threshold = max(med + PARAMS["threshold_z"] * mad, PARAMS["absolute_floor"])
    initial = valid & np.isfinite(smooth) & (smooth > threshold)
    structure = generate_binary_structure(2, 2)
    cleaned = binary_opening(initial, structure=structure)
    cleaned = binary_closing(cleaned, structure=structure)
    labels, count = label(cleaned, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.where((sizes >= PARAMS["min_component_pixels"]) & (sizes <= PARAMS["max_component_pixels"]))[0]
    keep = keep[keep != 0]
    final = np.isin(labels, keep) & valid
    if PARAMS["final_dilation"]: final = binary_dilation(final, iterations=PARAMS["final_dilation"]) & valid
    fraction = float(final.sum() / max(valid.sum(), 1))
    if fraction > PARAMS["max_candidate_fraction"]: final[:] = False; keep = np.array([], dtype=int)
    return {"relative": relative, "gaussian": smooth, "background": background, "residual": residual, "valid": valid, "initial": initial, "connected": final, "final": final, "mean": med, "std": mad, "threshold": threshold, "regions": int(len(keep)), "valid_count": int(valid.sum()), "initial_count": int(initial.sum()), "final_count": int(final.sum()), "candidate_fraction": fraction, "c_target": ct, "c_reference": cr, "screening_only": True, "flux_available": False}


def save_raster(path, array, profile, mask=False):
    profile = profile.copy(); profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate")
    data = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst: dst.write(data, 1)


def image_png(array, mask=False):
    from PIL import Image
    data = np.asarray(array); finite = np.isfinite(data)
    if mask:
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8); rgb[data.astype(bool)] = [255, 30, 0]
    else:
        rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
        if finite.any():
            lo, hi = np.percentile(data[finite], [2, 98]); hi = max(hi, lo + 1e-8)
            z = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0, 1)
            rgb = (plt.get_cmap("viridis")(z)[:, :, :3] * 255).astype(np.uint8); rgb[~finite] = 255
    stream = io.BytesIO(); Image.fromarray(rgb).save(stream, format="PNG"); return stream.getvalue()


def image_overlay(anomaly, mask):
    data = np.asarray(anomaly); finite = np.isfinite(data); rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(data[finite], [2, 98]); hi = max(hi, lo + 1e-8)
        z = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0, 1)
        rgb = (plt.get_cmap("viridis")(z)[:, :, :3] * 255).astype(np.uint8); rgb[~finite] = 255
    rgb[np.asarray(mask, dtype=bool)] = [255, 30, 0]
    stream = io.BytesIO(); from PIL import Image; Image.fromarray(rgb).save(stream, format="PNG"); return stream.getvalue()


def create_map(aoi):
    geom = shape(ensure_aoi(aoi)); fmap = folium.Map([geom.centroid.y, geom.centroid.x], zoom_start=7)
    folium.GeoJson(mapping(geom), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    return fmap


def georef_png_zip(array, profile, mask=False):
    png = image_png(array, mask); t = profile["transform"]
    pgw = f"{t.a}\n0\n0\n{t.e}\n{t.c + t.a / 2}\n{t.f + t.e / 2}\n".encode()
    prj = profile["crs"].to_wkt().encode() if profile.get("crs") else b""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("output.png", png); z.writestr("output.pgw", pgw); z.writestr("output.prj", prj)
    return out.getvalue()


st.set_page_config(page_title="Sentinel-2 Methane", page_icon="🛰️", layout="wide")
st.title("🛰️ Sentinel-2 Methane Candidate Screening")
if "aoi" not in st.session_state: st.session_state.aoi = mapping(box(48.0, 29.0, 49.0, 30.0))

left, right = st.columns([1.6, 1.0])
with left:
    st.subheader("Study area")
    map_data = st_folium(create_map(st.session_state.aoi), height=400, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi: st.session_state.aoi = new_aoi
with right:
    st.subheader("Scene search")
    d1, d2 = st.columns(2)
    with d1: start = st.date_input("Start date", datetime.now().date() - timedelta(days=90))
    with d2: end = st.date_input("End date", datetime.now().date())
    cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 50.0)
    ref_days = st.slider("Reference window (days)", 1, 90, 60)
    if st.button("Search scenes", type="primary", use_container_width=True):
        try:
            st.session_state.scene_results = search_scenes(st.session_state.aoi, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.max.time()), cloud)
            st.success(f"{len(st.session_state.scene_results)} scene(s) found")
        except Exception as e: st.error(f"Search failed: {e}")
    scenes = st.session_state.get("scene_results", [])
    if scenes:
        table = pd.DataFrame([{"date": get_datetime(s), "tile": get_tile(s), "cloud": get_cloud(s), "id": as_dict(s).get("id")} for s in scenes]).sort_values("date")
        st.dataframe(table, use_container_width=True, hide_index=True)
        ids = [as_dict(s).get("id") for s in scenes]
        chosen_id = st.selectbox("Target scene", ids, format_func=lambda x: str(x))
        st.session_state.target = next(s for s in scenes if as_dict(s).get("id") == chosen_id)

st.divider()
left, right = st.columns([1.6, 1.0])
with left:
    st.subheader("Detection")
    st.write("The algorithm uses conservative L1C proxy masks because L1C has no SCL layer.")
    threshold_z = st.number_input("Robust threshold multiplier", 2.0, 5.0, float(PARAMS["threshold_z"]), 0.1)
    min_pixels = st.number_input("Minimum component pixels", 4, 1000, int(PARAMS["min_component_pixels"]), 1)
    PARAMS["threshold_z"] = threshold_z; PARAMS["min_component_pixels"] = min_pixels
with right:
    st.subheader("Copernicus")
    auth = st.session_state.get("cdse_auth")
    if not auth:
        with st.form("login"):
            user = st.text_input("Email"); password = st.text_input("Password", type="password"); totp = st.text_input("2FA (optional)")
            submit = st.form_submit_button("Login")
        if submit:
            try: st.session_state.cdse_auth = authenticate_cdse(user, password, totp); st.rerun()
            except Exception as e: st.error(str(e))
    else:
        st.success(f"Connected: {auth.get('username', '')}")
        if st.button("Log out"): st.session_state.pop("cdse_auth", None); st.rerun()
    if st.session_state.get("target") is not None:
        disabled = not bool(st.session_state.get("cdse_auth"))
        if st.button("Download AOI & detect", type="primary", disabled=disabled, use_container_width=True):
            try:
                token = get_access_token(); target = st.session_state.target; target_date = get_datetime(target)
                candidates = [s for s in scenes if as_dict(s).get("id") != as_dict(target).get("id") and get_datetime(s) and abs((get_datetime(s) - target_date).days) <= ref_days]
                same = [s for s in candidates if get_tile(s) == get_tile(target)]; candidates = same or candidates
                if not candidates: raise RuntimeError("No reference scene found.")
                target_bands, profile = read_stack(download_scene(target, st.session_state.aoi, token))
                rows, best, best_corr = [], None, -np.inf
                for ref in candidates:
                    rb, _ = read_stack(download_scene(ref, st.session_state.aoi, token)); valid = np.isfinite(target_bands["B04"]) & np.isfinite(rb["B04"])
                    corr = float(np.corrcoef(target_bands["B04"][valid], rb["B04"][valid])[0, 1]) if valid.sum() > 100 else np.nan
                    rows.append({"id": as_dict(ref).get("id"), "date": get_datetime(ref), "tile": get_tile(ref), "b4_correlation": corr})
                    if np.isfinite(corr) and corr > best_corr: best, best_corr = rb, corr
                if best is None: raise RuntimeError("Reference correlation could not be calculated.")
                result = run_algorithm(target_bands, best); result["b4_correlation"] = best_corr; result["date"] = target_date.strftime("%Y-%m-%d")
                folder = RESULT_DIR / target_date.strftime("%Y%m%d"); folder.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key in ("relative", "gaussian", "background", "residual", "final", "valid"):
                    paths[key] = folder / f"{key}.tif"; save_raster(paths[key], result[key], profile, key in ("final", "valid"))
                st.session_state.result = result; st.session_state.paths = paths; st.session_state.profile = profile; st.session_state.references = pd.DataFrame(rows)
                st.success("Processing completed")
            except Exception as e: st.error(f"Detection failed: {e}")

if "result" in st.session_state:
    result, profile = st.session_state.result, st.session_state.profile
    st.divider(); st.subheader("Results")
    c = st.columns(7)
    c[0].metric("B4 corr", f"{result['b4_correlation']:.3f}"); c[1].metric("Valid", f"{result['valid_count']:,}"); c[2].metric("Initial", f"{result['initial_count']:,}"); c[3].metric("Final", f"{result['final_count']:,}"); c[4].metric("Regions", result["regions"]); c[5].metric("Threshold", f"{result['threshold']:.5f}"); c[6].metric("Candidate %", f"{100 * result['candidate_fraction']:.3f}%")
    outputs = [("relative", "Relative anomaly", False), ("gaussian", "Smoothed anomaly", False), ("residual", "Background-corrected", False), ("final", "Methane candidates", True), ("valid", "Valid mask", True)]
    cols = st.columns(len(outputs))
    for col, (key, title, is_mask) in zip(cols, outputs):
        with col:
            st.caption(title)
            st.image(image_png(result[key], is_mask), use_container_width=True)
            st.download_button("Download GeoTIFF", st.session_state.paths[key].read_bytes(), file_name=st.session_state.paths[key].name, mime="image/tiff", key=f"download_{key}", use_container_width=True)
    st.image(image_overlay(result["residual"], result["final"]), caption="Red = selected candidate pixels", use_container_width=True)
    st.download_button("Download reference table", st.session_state.references.to_csv(index=False), "reference_selection.csv", "text/csv")
    st.warning("This app produces conservative methane candidates from L1C proxy masks. It does not provide a validated emission rate.")
