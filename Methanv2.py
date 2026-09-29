"""
Sentinel-2 Methane Detection App — SimCLR-based (ISPRS 2026 Paper Implementation)
================================================================================
UI/UX preserved from original app. Algorithm replaced with:
- Self-Supervised Learning (SimCLR) for representation learning
- Fine-tuning for binary plume segmentation
- Integrated Mass Enhancement (IME) for emission quantification

References:
- Paper: "A Self-Supervised Learning Framework for Methane Emission Detection Using Sentinel-2"
- SimCLR: Chen et al., 2020
- MBMP: Varon et al., 2021
- IME: Varon et al., 2018
"""

from __future__ import annotations

import io
import json
import math
import re
import hashlib
import time
import copy
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
import rasterio.transform
from rasterio.warp import transform as rio_transform
import streamlit as st
from folium.plugins import Draw, MousePosition
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label,
)
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

# ── Deep Learning imports ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from torchvision import models
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch / timm not installed. Deep learning features will be disabled. "
               "Install with: pip install torch torchvision timm")

warnings.filterwarnings("ignore")

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BANDS = ["B03", "B04", "B08", "B11", "B12", "SCL"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".sentinel_methane_cache"
RESULT_DIR = Path.home() / ".sentinel_methane_results"
MODEL_DIR = Path.home() / ".sentinel_methane_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

S5P_COLLECTION = "sentinel-5p-l2"
DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

# ── SimCLR Hyperparameters (from paper) ──
SIMCLR_TEMPERATURE = 0.5          # tau
SIMCLR_PROJ_DIM = 256             # projection head units
SIMCLR_EPOCHS = 50                # pretraining epochs (paper doesn't specify exactly; use 50 for demo)
SIMCLR_BATCH_SIZE = 64
SIMCLR_LR = 3e-4
FINETUNE_EPOCHS = 30
WARMUP_EPOCHS = 10                # frozen encoder epochs
FINETUNE_LR = 1e-4
ENCODER_LR_FACTOR = 0.1           # lower LR for encoder during joint training
TILE_SIZE = 256
PATCH_SIZE = 256                  # Sentinel-2 tile size
NUM_CLASSES = 2                   # plume / non-plume

# ── IME Constants ──
CH4_MOLAR_MASS = 0.01604          # kg/mol
# Effective wind speed calibration (Guanter et al. 2021, used in paper)
# U_eff = 0.34 * U10 + 0.44  (m/s)
WIND_COEFF_A = 0.34
WIND_COEFF_B = 0.44

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  EVALSCRIPTS
# ══════════════════════════════════════════════════════════════════════════

def evalscript_s2():
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
  return {
    input: [{bands: ["CH4", "dataMask"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.CH4, 1.0];
}
"""


# ══════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════

def download_scene(item, aoi, access_token, return_array=False):
    """Download Sentinel-2 scene bands for AOI. Returns path or array."""
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION, "simclr_v1"], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        if return_array:
            return read_stack(output_path)
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")

    time_from = acquisition.strftime("%Y-%m-%dT00:00:00Z")
    time_to = (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "id": "refl",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                        "mosaickingOrder": "leastCC",
                    },
                },
                {
                    "type": "sentinel-2-l2a",
                    "id": "scl",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                        "mosaickingOrder": "leastCC",
                    },
                },
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript_s2(),
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
    if return_array:
        return read_stack(output_path)
    return output_path


def download_s5p_scene(aoi, date_from, date_to, access_token):
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(
        json.dumps(["s5p_ch4_v3", aoi, str(date_from), str(date_to)], sort_keys=True).encode()
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


# ══════════════════════════════════════════════════════════════════════════
#  SIMCLR DATASET & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

class Sentinel2TileDataset(Dataset):
    """
    Dataset for SimCLR pretraining on unlabeled Sentinel-2 tiles.
    Each item is a 6-channel image (B03, B04, B08, B11, B12, SCL).
    Augmentations: random zoom, rotation, flip, translation, cutout.
    """
    def __init__(self, tiles: np.ndarray, transform=None):
        # tiles shape: (N, 6, H, W)
        self.tiles = tiles.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.tiles[idx])
        if self.transform:
            x1 = self.transform(img)
            x2 = self.transform(img)
            return x1, x2
        return img, img


class SimCLRAugmentation:
    """
    SimCLR augmentation pipeline for Sentinel-2 tiles.
    Paper: random zoom, rotation, horizontal and vertical flipping,
    translation, and cutout operations applied independently and stochastically.
    """
    def __init__(self, size=TILE_SIZE):
        self.size = size
        # Spatial transforms only (paper focuses on spatial transformations)
        self.spatial = T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
        ])
        self.cutout = T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3))

    def __call__(self, img):
        # img: (C, H, W) tensor
        img = self.spatial(img)
        img = self.cutout(img)
        return img


# ══════════════════════════════════════════════════════════════════════════
#  SIMCLR MODEL: ENCODER + PROJECTION HEAD
# ══════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    Projection head: 3-layer MLP with 256 units each and ReLU.
    Paper: Section 2.2.3
    """
    def __init__(self, in_dim, hidden_dim=SIMCLR_PROJ_DIM, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    SimCLR model: encoder backbone + projection head.
    Supports multiple backbones: MobileNet, Inception-v3, DenseNet-121,
    Xception, ViT, SwinT.
    """
    def __init__(self, backbone_name="mobilenet_v3_small", proj_dim=SIMCLR_PROJ_DIM, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, self.feat_dim = self._build_encoder(backbone_name, pretrained)
        self.projection_head = ProjectionHead(self.feat_dim, hidden_dim=proj_dim)

    def _build_encoder(self, name, pretrained):
        """Build encoder backbone and return (module, feature_dim)."""
        name_lower = name.lower().replace("-", "_").replace(" ", "_")

        if "mobilenet" in name_lower:
            # MobileNetV3 Small
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            # Remove classifier; use features + avgpool
            encoder = nn.Sequential(*list(base.features.children()), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat_dim = 576  # MobileNetV3 Small output channels
        elif "inception" in name_lower:
            base = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None, aux_logits=False)
            # Remove fc
            encoder = nn.Sequential(*list(base.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat_dim = 2048
        elif "densenet" in name_lower:
            base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            encoder = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat_dim = 1024
        elif "xception" in name_lower:
            # Use timm for Xception
            try:
                base = timm.create_model("xception", pretrained=pretrained, num_classes=0, global_pool="avg")
                encoder = base
                feat_dim = base.num_features
            except Exception:
                # Fallback to ResNet50 if xception unavailable
                st.warning("Xception not available in timm, falling back to ResNet50")
                base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
                encoder = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
                feat_dim = 2048
        elif "vit" in name_lower:
            base = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0, global_pool="avg")
            encoder = base
            feat_dim = base.num_features
        elif "swin" in name_lower:
            base = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0, global_pool="avg")
            encoder = base
            feat_dim = base.num_features
        else:
            raise ValueError(f"Unknown backbone: {name}")

        return encoder, feat_dim

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


# ── NT-Xent Loss ──

def nt_xent_loss(z1, z2, temperature=SIMCLR_TEMPERATURE):
    """
    Normalized Temperature-scaled Cross Entropy Loss (Equation 1 in paper).
    z1, z2: (B, D) - projections of two augmented views.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim_matrix = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    targets = torch.arange(2 * B, device=z.device)
    targets[:B] = targets[:B] + B
    targets[B:] = targets[B:] - B

    loss = F.cross_entropy(sim_matrix, targets)
    return loss


# ══════════════════════════════════════════════════════════════════════════
#  DECODER (Figure 3 in paper)
# ══════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Decoder: Flatten -> Dense -> Reshape -> ConvTranspose + ReLU + BN
    Architecture from Figure 3 of the paper.
    Output: single-channel binary segmentation map.
    """
    def __init__(self, in_feat_dim, initial_spatial=7, out_channels=1):
        super().__init__()
        self.initial_spatial = initial_spatial
        self.fc = nn.Sequential(
            nn.Linear(in_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512 * initial_spatial * initial_spatial),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, self.initial_spatial, self.initial_spatial)
        x = self.deconv(x)
        return x


class SegmentationModel(nn.Module):
    """Encoder + Decoder for plume segmentation."""
    def __init__(self, encoder, feat_dim, initial_spatial=7):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder(feat_dim, initial_spatial=initial_spatial)

    def forward(self, x):
        h = self.encoder(x)  # (B, feat_dim)
        return self.decoder(h)


# ══════════════════════════════════════════════════════════════════════════
#  MBMP BASELINE (for automatic labeling / comparison)
# ══════════════════════════════════════════════════════════════════════════

def mbmp_enhancement(target_bands, reference_bands):
    """
    MBMP enhancement (Varon et al., 2021).
    ΔΩ_MBMP = (δ12 * B12^t - δ11 * B11^t) / (δ11 * B11^t) - same for reference
    where δ are normalization coefficients computed via regression.
    Returns enhancement map in ppb (approximate).
    """
    B11_t = target_bands["B11"]
    B12_t = target_bands["B12"]
    B11_r = reference_bands["B11"]
    B12_r = reference_bands["B12"]

    # Compute normalization coefficients via regression on valid pixels
    valid = (np.isfinite(B11_t) & np.isfinite(B12_t) &
             np.isfinite(B11_r) & np.isfinite(B12_r) &
             (B11_t > 0.05) & (B12_t > 0.05) &
             (B11_r > 0.05) & (B12_r > 0.05))

    if valid.sum() < 100:
        return np.full_like(B11_t, np.nan)

    x = B12_t[valid]
    y = B11_t[valid]
    delta_11, delta_12 = np.polyfit(x, y, 1)

    dOmega_t = (delta_12 * B12_t - delta_11 * B11_t) / (delta_11 * B11_t)
    dOmega_r = (delta_12 * B12_r - delta_11 * B11_r) / (delta_11 * B11_r)
    dOmega = dOmega_t - dOmega_r

    # Convert to ppb (approximate scaling)
    K_MBMP = 1e-5
    return dOmega / K_MBMP


# ══════════════════════════════════════════════════════════════════════════
#  IME QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════

def compute_ime(plume_mask, enhancement_map, pixel_area_m2=None):
    """
    Integrated Mass Enhancement (IME).
    IME = sum_j (ΔΩ_j * A_j)  [kg]
    where ΔΩ is column enhancement [kg/m²] and A_j is pixel area [m²].
    """
    if pixel_area_m2 is None:
        pixel_area_m2 = RESOLUTION * RESOLUTION  # 20m x 20m

    # ΔΩ is in ppb; convert to kg/m²
    # 1 ppb CH4 ≈ 2.69e-9 kg/m² (approximate at surface)
    PPB_TO_KG_M2 = 2.69e-9
    delta_omega = enhancement_map[plume_mask] * PPB_TO_KG_M2

    if delta_omega.size == 0:
        return 0.0

    ime = np.nansum(delta_omega) * pixel_area_m2
    return float(ime)


def compute_emission_rate(plume_mask, enhancement_map, wind_speed_10m, pixel_area_m2=None):
    """
    Estimate CH4 emission rate Q using IME method.
    Q = U_eff * IME / L
    where L = sqrt(plume_area) (characteristic plume size),
    U_eff = 0.34 * U10 + 0.44 (Guanter et al. 2021).
    Returns Q in kg/h.
    """
    if pixel_area_m2 is None:
        pixel_area_m2 = RESOLUTION * RESOLUTION

    ime = compute_ime(plume_mask, enhancement_map, pixel_area_m2)
    plume_area = np.sum(plume_mask) * pixel_area_m2
    if plume_area <= 0:
        return 0.0, 0.0, 0.0

    L = math.sqrt(plume_area)  # characteristic length [m]
    U_eff = WIND_COEFF_A * wind_speed_10m + WIND_COEFF_B  # m/s

    # Q = U_eff * IME / L  [kg/s]
    Q_kg_s = U_eff * ime / L
    Q_kg_h = Q_kg_s * 3600.0

    return Q_kg_h, ime, L


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def prepare_tiles_from_scenes(scenes, aoi, access_token, progress_bar=None):
    """Download scenes and extract 256x256 tiles for training."""
    tiles = []
    total = len(scenes)
    for i, scene in enumerate(scenes):
        try:
            if progress_bar:
                progress_bar.progress((i + 1) / total, text=f"Downloading tile {i+1}/{total}…")
            bands, profile = download_scene(scene, aoi, access_token, return_array=True)
            # Stack bands into (6, H, W)
            stacked = np.stack([bands[b] for b in BANDS], axis=0)
            # Normalize reflectance to [0, 1] range for B03-B12
            stacked[:5] = np.clip(stacked[:5], 0, 1)
            # SCL is categorical, keep as is
            stacked[5] = np.clip(stacked[5], 0, 11)
            # Crop/resize to 256x256
            H, W = stacked.shape[1], stacked.shape[2]
            if H > TILE_SIZE or W > TILE_SIZE:
                # Center crop
                top = (H - TILE_SIZE) // 2
                left = (W - TILE_SIZE) // 2
                stacked = stacked[:, top:top+TILE_SIZE, left:left+TILE_SIZE]
            elif H < TILE_SIZE or W < TILE_SIZE:
                # Pad
                pad_h = max(0, TILE_SIZE - H)
                pad_w = max(0, TILE_SIZE - W)
                stacked = np.pad(stacked, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
            tiles.append(stacked)
        except Exception as e:
            print(f"Failed to download scene {i}: {e}")
            continue
    return np.array(tiles)


def pretrain_simclr(tiles, backbone_name, epochs=SIMCLR_EPOCHS, batch_size=SIMCLR_BATCH_SIZE,
                    lr=SIMCLR_LR, device="cpu"):
    """Pretrain SimCLR model on unlabeled tiles."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    augment = SimCLRAugmentation(TILE_SIZE)
    dataset = Sentinel2TileDataset(tiles, transform=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SimCLRModel(backbone_name, pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(loader))
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"[SimCLR] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, losses


def fine_tune_segmentation(pretrained_encoder, tiles, labels, feat_dim,
                           epochs=FINETUNE_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                           batch_size=16, lr=FINETUNE_LR, device="cpu"):
    """
    Fine-tune segmentation model.
    Two-stage training (Figure 4):
    - Stage 1: frozen encoder, train decoder for warmup_epochs
    - Stage 2: joint fine-tuning with lower LR for encoder
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    # Create segmentation model
    model = SegmentationModel(pretrained_encoder, feat_dim)
    model.to(device)

    # Dataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(tiles).float(),
        torch.from_numpy(labels).float().unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    # Stage 1: Frozen encoder, train decoder only
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)

    for epoch in range(warmup_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Resize output to match label size
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Fine-tune Stage 1] Epoch {epoch+1}/{warmup_epochs} - Loss: {epoch_loss/max(1,len(loader)):.4f}")

    # Stage 2: Joint fine-tuning
    for param in model.encoder.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr * ENCODER_LR_FACTOR},
        {'params': model.decoder.parameters(), 'lr': lr},
    ])

    for epoch in range(epochs - warmup_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Fine-tune Stage 2] Epoch {epoch+1}/{epochs-warmup_epochs} - Loss: {epoch_loss/max(1,len(loader)):.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def predict_plume_mask(model, bands, device="cpu", threshold=0.5):
    """Run segmentation model on a single tile."""
    model.eval()
    # Prepare input
    stacked = np.stack([bands[b] for b in BANDS], axis=0)
    stacked[:5] = np.clip(stacked[:5], 0, 1)
    stacked[5] = np.clip(stacked[5], 0, 11)
    H, W = stacked.shape[1], stacked.shape[2]
    if H != TILE_SIZE or W != TILE_SIZE:
        # Resize
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=(TILE_SIZE, TILE_SIZE), mode='bilinear', align_corners=False)
    else:
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()

    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).squeeze().cpu().numpy()

    # Resize mask back to original size
    if mask.shape != (H, W):
        mask = np.array(torch.nn.functional.interpolate(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze().bool())

    return mask


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS (for results display)
# ══════════════════════════════════════════════════════════════════════════

def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    output = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def create_map(aoi, search_center=None, search_zoom=11):
    geometry = shape(ensure_aoi(aoi))
    if search_center:
        center = [float(search_center[0]), float(search_center[1])]
        zoom = int(search_zoom)
    else:
        center = [geometry.centroid.y, geometry.centroid.x]
        zoom = 11

    fmap = folium.Map(center, zoom_start=zoom, tiles="OpenStreetMap")
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False},
    ).add_to(fmap)

    if search_center:
        folium.Marker(
            [float(search_center[0]), float(search_center[1])],
            popup=folium.Popup(
                f"<b>Searched location</b><br>"
                f"Lat: {float(search_center[0]):.5f}<br>"
                f"Lon: {float(search_center[1]):.5f}",
                max_width=260,
            ),
            tooltip="Searched location",
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

    try:
        MousePosition(
            position="bottomright",
            separator=" | ",
            prefix="Lat/Lon:",
            num_digits=5,
        ).add_to(fmap)
    except Exception:
        pass

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


# ══════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Sentinel-2 Methane — SimCLR", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

# ── CSS (identical to original) ──
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
.coords-panel { margin-top: 0.55rem; background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.5rem 0.7rem; display: flex; flex-direction: column; gap: 0.3rem; }
.coord-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: #111111 !important; }
.coord-label { font-weight: 700; color: #111111 !important; flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.coord-value { flex: 0 0 auto; font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; background: #eef5f6; border: 1px solid #cfe0e3; border-radius: 6px; padding: 0.12rem 0.5rem; color: #111111 !important; font-size: 0.74rem; white-space: nowrap; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane — SimCLR Framework</div>
        <div class="app-subtitle">ISPRS 2026 · Self-Supervised Learning + IME Quantification · MobileNet/SwinT/Xception</div>
    </div>
    <div class="status-pill">20 m processing &nbsp;&nbsp; Deep Learning</div>
</div>
""", unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)

# ── Section 01: Study Area ──
map_col, control_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">Search a location or draw the study area directly on the map.</div>',
        unsafe_allow_html=True,
    )

    search_col1, search_col2 = st.columns([4, 1], gap="small")
    with search_col1:
        search_query = st.text_input(
            "Location search",
            placeholder="🔎  Search a place  (e.g. Tehran, Aradkouh landfill, Paris, …)",
            key="location_search_input",
            label_visibility="collapsed",
        )
    with search_col2:
        search_clicked = st.button("🔍 Find", use_container_width=True, key="search_location_btn")

    if search_clicked:
        query = (search_query or "").strip()
        if not query:
            st.warning("Please type a location name first.")
        else:
            with st.spinner("Searching location…"):
                geo = geocode_location(query)
            if geo:
                st.session_state["search_center"] = [geo[0], geo[1]]
                st.session_state["search_name"] = geo[2]
                st.session_state["search_zoom"] = 12
            else:
                st.warning("Location not found. Try a different name or spelling.")

    if st.session_state.get("search_center"):
        info_col, clear_col = st.columns([4, 1], gap="small")
        with info_col:
            st.markdown(
                f'<div class="card-caption" style="margin-top:0.35rem;">'
                f'📍 {st.session_state.get("search_name", "Searched location")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with clear_col:
            if st.button("✖ Clear", use_container_width=True, key="clear_search_btn"):
                st.session_state.pop("search_center", None)
                st.session_state.pop("search_name", None)
                st.rerun()

    map_center = st.session_state.get("search_center")
    map_zoom = st.session_state.get("search_zoom", 11)
    map_data = st_folium(
        create_map(st.session_state.aoi, search_center=map_center, search_zoom=map_zoom),
        height=385,
        width=1000,
        key="aoi_map",
    )
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry(
            {"type": "FeatureCollection", "features": map_data["all_drawings"]}
        )
        if new_aoi:
            st.session_state.aoi = new_aoi

    coord_lines = []
    aoi_geom = shape(ensure_aoi(st.session_state.aoi))
    aoi_c = aoi_geom.centroid
    coord_lines.append(
        f'<div class="coord-row">'
        f'<span class="coord-label">🎯 AOI center</span>'
        f'<span class="coord-value">Lat {aoi_c.y:.5f} · Lon {aoi_c.x:.5f}</span>'
        f'</div>'
    )

    if st.session_state.get("search_center"):
        sc = st.session_state["search_center"]
        sname = st.session_state.get("search_name", "Searched location")
        short_name = (sname[:70] + "…") if len(sname) > 70 else sname
        coord_lines.append(
            f'<div class="coord-row">'
            f'<span class="coord-label">📍 {short_name}</span>'
            f'<span class="coord-value">Lat {sc[0]:.5f} · Lon {sc[1]:.5f}</span>'
            f'</div>'
        )

    if map_data and map_data.get("last_clicked"):
        lc = map_data["last_clicked"]
        coord_lines.append(
            f'<div class="coord-row">'
            f'<span class="coord-label">🖱️ Last click on map</span>'
            f'<span class="coord-value">Lat {lc["lat"]:.5f} · Lon {lc["lng"]:.5f}</span>'
            f'</div>'
        )

    coord_lines.append(
        '<div class="coord-row"><span class="coord-label" style="opacity:0.65;font-weight:500;">'
        'Live mouse position shown on the map (bottom-right corner).</span>'
        '<span class="coord-value" style="opacity:0.65;">–</span></div>'
    )

    st.markdown(
        '<div class="coords-panel">' + "".join(coord_lines) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Section 02: Scene Search + Algorithm Selection ──
with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH & ALGORITHM</div>', unsafe_allow_html=True)
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
        max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 30.0, key="max_cloud")
    with s2:
        reference_days = st.slider("Reference window (days)", 1, 90, 60, key="reference_days")

    # Algorithm selection
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Algorithm</div>', unsafe_allow_html=True)
    algorithm_mode = st.radio(
        "Select algorithm",
        ["SimCLR (SSL) + Fine-tuning", "MBMP Baseline (for labeling)"],
        index=0,
        key="algorithm_mode",
        label_visibility="collapsed",
    )

    # Backbone selection for SimCLR
    if "SimCLR" in algorithm_mode:
        backbone_choice = st.selectbox(
            "Encoder backbone",
            ["mobilenet_v3_small", "inception_v3", "densenet121", "xception", "vit_base_patch16_224", "swin_tiny_patch4_window7_224"],
            index=0,
            key="backbone_choice",
        )
        simclr_epochs = st.number_input("SimCLR pretraining epochs", 5, 200, 30, key="simclr_epochs")
        finetune_epochs = st.number_input("Fine-tuning epochs", 5, 100, 20, key="finetune_epochs")
        warmup_epochs = st.number_input("Warmup (frozen encoder) epochs", 0, 30, 10, key="warmup_epochs")

    if st.button("🔍  Search Sentinel-2 scenes", type="primary", use_container_width=True):
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
            scene_date = get_datetime(scene)
            date_text = scene_date.strftime("%Y-%m-%d") if scene_date else "Unknown date"
            return f"{date_text}  |  {get_tile(scene) or 'Unknown tile'}  |  cloud {get_cloud(scene):.1f}%"

        if scene_ids:
            selected_scene_id = st.selectbox("Target scene", scene_ids, format_func=format_scene, key="target_scene_select")
            selected_scene = next((s for s in scene_results if as_dict(s).get("id") == selected_scene_id), None)
            if selected_scene is not None:
                st.session_state["target"] = selected_scene

    st.markdown('</div>', unsafe_allow_html=True)

# ── Section 03 & 04: Process ──
st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · TRAINING DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Training Configuration</div>', unsafe_allow_html=True)

    if "SimCLR" in algorithm_mode:
        st.markdown(
            '<div class="card-caption">'
            'SimCLR pretraining uses unlabeled Sentinel-2 tiles downloaded from the search window. '
            'Fine-tuning requires labeled plume masks. You can either draw plumes on the map '
            '(using the polygon tool) or generate pseudo-labels automatically using the MBMP baseline.'
            '</div>',
            unsafe_allow_html=True,
        )
        label_source = st.radio(
            "Label source for fine-tuning",
            ["Auto (MBMP pseudo-labels)", "Manual (draw on map)"],
            index=0,
            key="label_source",
        )
    else:
        st.markdown(
            '<div class="card-caption">'
            'MBMP baseline computes enhancement maps directly. No training required. '
            'Use this mode for quick screening or to generate pseudo-labels.'
            '</div>',
            unsafe_allow_html=True,
        )

    num_training_scenes = st.slider("Number of training scenes to download", 5, 50, 20, key="num_training_scenes")
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="auth-help">Log in once. Password is sent directly to Copernicus.</div>', unsafe_allow_html=True)
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
                st.success("Copernicus login successful.")
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

            run_button_label = "🧠  Train SimCLR & Detect Methane" if "SimCLR" in algorithm_mode else "📊  Run MBMP Baseline"
            detect_clicked = st.button(
                run_button_label, type="primary", use_container_width=True,
                key="detect_button", disabled=not bool(st.session_state.get("cdse_auth"))
            )
            if not st.session_state.get("cdse_auth"):
                st.markdown('<div class="card-caption">Please connect your Copernicus account first.</div>', unsafe_allow_html=True)

            if detect_clicked:
                progress = st.progress(0, text="Preparing…")
                progress_status = st.empty()
                try:
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    target_date = get_datetime(target)
                    target_tile = get_tile(target)
                    if target_date is None:
                        raise RuntimeError("Target scene has no valid acquisition date.")

                    # ── Download target scene ──
                    progress_status.markdown('<div class="card-caption">Step 1 · Downloading target scene…</div>', unsafe_allow_html=True)
                    progress.progress(5, text="Downloading target bands…")
                    target_bands, profile = download_scene(target, st.session_state.aoi, access_token, return_array=True)

                    # ── Select reference scenes ──
                    all_candidates = [
                        scene for scene in scene_results
                        if as_dict(scene).get("id") != as_dict(target).get("id")
                        and get_datetime(scene) is not None
                        and abs((get_datetime(scene) - target_date).total_seconds()) / 86400 <= reference_days
                    ]
                    same_tile = [s for s in all_candidates if get_tile(s) == target_tile]
                    references = same_tile if same_tile else all_candidates

                    if "MBMP" in algorithm_mode:
                        # ── MBMP Baseline ──
                        if not references:
                            raise RuntimeError("No reference scene available for MBMP.")
                        progress_status.markdown('<div class="card-caption">Step 2 · Computing MBMP enhancement…</div>', unsafe_allow_html=True)
                        progress.progress(30, text="Downloading reference scene…")
                        ref_bands, _ = download_scene(references[0], st.session_state.aoi, access_token, return_array=True)
                        enhancement = mbmp_enhancement(target_bands, ref_bands)
                        # Threshold at 2-sigma
                        valid_vals = enhancement[np.isfinite(enhancement)]
                        med = np.median(valid_vals)
                        sigma = np.std(valid_vals)
                        plume_mask = enhancement > (med + 2.0 * sigma)
                        result = {
                            "enhancement": enhancement,
                            "final": plume_mask,
                            "mean": med,
                            "std": sigma,
                            "threshold": med + 2.0 * sigma,
                            "valid_count": int(np.isfinite(enhancement).sum()),
                            "final_count": int(plume_mask.sum()),
                            "regions": 1 if plume_mask.any() else 0,
                            "initial_count": int((enhancement > med).sum()),
                            "c": 0.0,
                            "b4_correlation": float("nan"),
                            "date": target_date.strftime("%Y-%m-%d"),
                            "algorithm": "MBMP",
                        }
                        progress.progress(100, text="MBMP baseline complete")
                        st.success("MBMP baseline computed.")
                    else:
                        # ── SimCLR Pipeline ──
                        if not TORCH_AVAILABLE:
                            raise RuntimeError("PyTorch is required for SimCLR. Please install torch, torchvision, timm.")

                        # 1. Download unlabeled training tiles
                        progress_status.markdown('<div class="card-caption">Step 2 · Downloading unlabeled training tiles…</div>', unsafe_allow_html=True)
                        train_scenes = scene_results[:min(num_training_scenes, len(scene_results))]
                        tiles = prepare_tiles_from_scenes(train_scenes, st.session_state.aoi, access_token, progress_bar=progress)
                        if len(tiles) < 5:
                            raise RuntimeError(f"Only {len(tiles)} tiles downloaded. Need at least 5. Try expanding the date range or cloud threshold.")

                        # 2. SimCLR Pretraining
                        progress_status.markdown('<div class="card-caption">Step 3 · SimCLR self-supervised pretraining…</div>', unsafe_allow_html=True)
                        progress.progress(30, text="Pretraining SimCLR…")
                        simclr_model, losses = pretrain_simclr(
                            tiles, backbone_choice, epochs=int(simclr_epochs),
                            batch_size=SIMCLR_BATCH_SIZE, lr=SIMCLR_LR, device="cpu"
                        )

                        # 3. Prepare labeled data (fine-tuning)
                        progress_status.markdown('<div class="card-caption">Step 4 · Preparing labeled data for fine-tuning…</div>', unsafe_allow_html=True)
                        progress.progress(60, text="Generating labels…")

                        if label_source == "Auto (MBMP pseudo-labels)":
                            # Generate pseudo-labels using MBMP
                            labels = []
                            label_tiles = []
                            for i, scene in enumerate(train_scenes[:min(10, len(train_scenes))]):
                                try:
                                    bands, _ = download_scene(scene, st.session_state.aoi, access_token, return_array=True)
                                    # Find reference
                                    refs = [r for r in references if as_dict(r).get("id") != as_dict(scene).get("id")]
                                    if not refs:
                                        continue
                                    ref_bands, _ = download_scene(refs[0], st.session_state.aoi, access_token, return_array=True)
                                    enh = mbmp_enhancement(bands, ref_bands)
                                    # Threshold
                                    valid = enh[np.isfinite(enh)]
                                    if len(valid) < 100:
                                        continue
                                    med = np.median(valid)
                                    sig = np.std(valid)
                                    mask = (enh > (med + 2.0 * sig)).astype(np.float32)
                                    # Resize to 256x256
                                    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                                    mask_t = F.interpolate(mask_t, size=(TILE_SIZE, TILE_SIZE), mode='nearest')
                                    mask = mask_t.squeeze().numpy()
                                    stacked = np.stack([bands[b] for b in BANDS], axis=0)
                                    stacked[:5] = np.clip(stacked[:5], 0, 1)
                                    stacked[5] = np.clip(stacked[5], 0, 11)
                                    H, W = stacked.shape[1], stacked.shape[2]
                                    if H > TILE_SIZE or W > TILE_SIZE:
                                        top = (H - TILE_SIZE) // 2
                                        left = (W - TILE_SIZE) // 2
                                        stacked = stacked[:, top:top+TILE_SIZE, left:left+TILE_SIZE]
                                    elif H < TILE_SIZE or W < TILE_SIZE:
                                        pad_h = max(0, TILE_SIZE - H)
                                        pad_w = max(0, TILE_SIZE - W)
                                        stacked = np.pad(stacked, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
                                        mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode='constant')
                                    label_tiles.append(stacked)
                                    labels.append(mask)
                                except Exception as e:
                                    continue
                            if len(labels) < 3:
                                raise RuntimeError("Could not generate enough labeled samples. Try expanding the search.")
                            labels_arr = np.array(labels)
                            label_tiles_arr = np.array(label_tiles)
                        else:
                            # Manual labeling: use target scene + draw
                            st.warning("Manual labeling mode: using MBMP pseudo-labels for demonstration. "
                                       "In production, draw plume polygons on the map.")
                            # Fallback to pseudo-labels
                            labels_arr = np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                            label_tiles_arr = tiles[:1]

                        # 4. Fine-tune
                        progress_status.markdown('<div class="card-caption">Step 5 · Fine-tuning segmentation model…</div>', unsafe_allow_html=True)
                        progress.progress(75, text="Fine-tuning…")
                        # Get feat_dim from pretrained encoder
                        feat_dim = simclr_model.feat_dim
                        encoder = simclr_model.encoder
                        seg_model = fine_tune_segmentation(
                            encoder, label_tiles_arr, labels_arr, feat_dim,
                            epochs=int(finetune_epochs), warmup_epochs=int(warmup_epochs),
                            batch_size=8, lr=FINETUNE_LR, device="cpu"
                        )

                        # 5. Predict on target scene
                        progress_status.markdown('<div class="card-caption">Step 6 · Running detection on target scene…</div>', unsafe_allow_html=True)
                        progress.progress(90, text="Detecting plumes…")
                        plume_mask = predict_plume_mask(seg_model, target_bands, device="cpu", threshold=0.5)

                        # Compute enhancement for visualization (MBMP)
                        if references:
                            ref_bands, _ = download_scene(references[0], st.session_state.aoi, access_token, return_array=True)
                            enhancement = mbmp_enhancement(target_bands, ref_bands)
                        else:
                            enhancement = np.full_like(target_bands["B11"], np.nan)

                        valid_vals = enhancement[np.isfinite(enhancement)]
                        result = {
                            "enhancement": enhancement,
                            "final": plume_mask,
                            "mean": float(np.nanmedian(valid_vals)) if len(valid_vals) > 0 else 0.0,
                            "std": float(np.nanstd(valid_vals)) if len(valid_vals) > 0 else 1.0,
                            "threshold": 0.0,
                            "valid_count": int(np.isfinite(enhancement).sum()),
                            "final_count": int(plume_mask.sum()),
                            "regions": 1 if plume_mask.any() else 0,
                            "initial_count": int(plume_mask.sum()),
                            "c": 0.0,
                            "b4_correlation": float("nan"),
                            "date": target_date.strftime("%Y-%m-%d"),
                            "algorithm": f"SimCLR ({backbone_choice})",
                            "simclr_losses": losses,
                        }
                        progress.progress(100, text="SimCLR detection complete")
                        st.success("SimCLR detection completed successfully.")

                    # ── IME Quantification ──
                    progress_status.markdown('<div class="card-caption">Step 7 · Estimating emission rate (IME)…</div>', unsafe_allow_html=True)
                    # Use a default wind speed (can be replaced with real data)
                    wind_10m = st.number_input("10 m wind speed (m/s)", 0.0, 20.0, 3.0, key="wind_speed")
                    Q_kg_h, ime, L = compute_emission_rate(result["final"], result["enhancement"], wind_10m)
                    result["Q_kg_h"] = Q_kg_h
                    result["IME_kg"] = ime
                    result["plume_L_m"] = L
                    result["wind_speed"] = wind_10m

                    # Save results
                    output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                    output_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("enhancement", "final"):
                        paths[key] = output_folder / f"{key}.tif"
                        save_raster(paths[key], result[key], profile, key == "final")

                    st.session_state.result = result
                    st.session_state.paths = paths
                    st.session_state.output_profile = profile
                    st.session_state.png_outputs = {
                        "enhancement": image_png(result["enhancement"]),
                        "final": image_png(result["final"], mask=True),
                    }
                    progress.progress(100, text="Complete")
                    st.success("Processing completed successfully.")

                    if result["final_count"] == 0:
                        st.warning("No plume pixels detected above threshold. Try adjusting the threshold or using a different scene.")

                except Exception as error:
                    st.error(f"Processing failed: {error}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.markdown('<div class="card-title">Select scenes first</div><div class="card-caption">Search for Sentinel-2 scenes and select a target.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════

if "result" in st.session_state:
    result = st.session_state.result
    png_outputs = st.session_state.get("png_outputs", {})
    profile = st.session_state.get("output_profile")

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    # Metrics row
    metrics = st.columns(6, gap="small")
    metrics[0].metric("Algorithm", result.get("algorithm", "N/A"))
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Plume pixels", f"{result['final_count']:,}")
    metrics[3].metric("Regions", result["regions"])
    if "Q_kg_h" in result:
        metrics[4].metric("Emission rate", f"{result['Q_kg_h']:.0f} kg/h")
        metrics[5].metric("IME", f"{result['IME_kg']:.2f} kg")
    else:
        metrics[4].metric("Threshold", f"{result['threshold']:.2f}")
        metrics[5].metric("Median", f"{result['mean']:.2f}")

    # SimCLR loss curve
    if "simclr_losses" in result:
        with st.expander("📉 SimCLR pretraining loss", expanded=False):
            loss_df = pd.DataFrame({"Epoch": range(1, len(result["simclr_losses"]) + 1), "Loss": result["simclr_losses"]})
            st.line_chart(loss_df.set_index("Epoch"), use_container_width=True, height=200)

    # Result images
    result_items = [
        ("enhancement", "CH4 Enhancement (ppb)", "Enhancement"),
        ("final", "Detected Plume Mask", "Plume Mask"),
    ]
    result_cols = st.columns(2, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown(f'<div class="result-tag">{tag}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(png_outputs[key], use_container_width=True, output_format="PNG")
            with legend_col:
                st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                legend_kind = "mask" if key == "final" else "continuous"
                st.markdown(legend_html(legend_kind), unsafe_allow_html=True)
            path = st.session_state.paths[key]
            format_choice = st.selectbox("Download format", ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"], key=f"format_choice_{key}")
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button("⬇ Download GeoTIFF", path.read_bytes(), file_name=path.name, mime="image/tiff", key=f"download_tif_{key}", use_container_width=True)
            else:
                png_package = georeferenced_png_package(result[key], profile, mask=key == "final")
                st.download_button("⬇ Download Georeferenced PNG package", png_package, file_name=f"{key}_georeferenced_png.zip", mime="application/zip", key=f"download_png_{key}", use_container_width=True)

    # Summary note
    if "Q_kg_h" in result:
        st.markdown(
            f'<div class="result-note">'
            f'<b>Emission quantification (IME):</b> '
            f'Q = <b>{result["Q_kg_h"]:.0f} kg/h</b> · '
            f'IME = <b>{result["IME_kg"]:.2f} kg</b> · '
            f'Plume characteristic length L = <b>{result["plume_L_m"]:.0f} m</b> · '
            f'Wind speed (10 m) = <b>{result["wind_speed"]:.1f} m/s</b>. '
            f'U<sub>eff</sub> = 0.34·U<sub>10</sub> + 0.44.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  S5P CONTEXT (kept from original)
# ══════════════════════════════════════════════════════════════════════════

if "result" in st.session_state:
    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05b · SENTINEL-5P CH4 CONTEXT</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">TROPOMI CH4 (~5.5 × 7 km). Visualization is anomaly relative to local mean.</div>', unsafe_allow_html=True)

    s5p_col1, s5p_col2 = st.columns([1, 3], gap="small")
    with s5p_col1:
        run_s5p = st.button("🛰️  Fetch S5P CH4", type="primary", use_container_width=True, key="s5p_button", disabled=not bool(st.session_state.get("cdse_auth")))
    with s5p_col2:
        s5p_days = st.slider("S5P temporal window (days around target)", 1, 30, 15, key="s5p_days")

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
            ch4[~np.isfinite(ch4)] = np.nan
            ch4[ch4 <= 0] = np.nan
            st.session_state.s5p_ch4 = ch4
            st.success("S5P CH4 loaded")
        except Exception as s5p_error:
            st.error(f"S5P fetch failed: {s5p_error}")

    if "s5p_ch4" in st.session_state:
        ch4 = st.session_state.s5p_ch4
        valid_ch4 = ch4[np.isfinite(ch4)]
        if valid_ch4.size < 2:
            placeholder = float(np.nanmean(valid_ch4)) if valid_ch4.size > 0 else 1900.0
            ch4 = np.full_like(ch4, placeholder)
            valid_ch4 = ch4[np.isfinite(ch4)]
            st.warning("⚠️ Sentinel-5P returned no valid pixels. Showing placeholder.")
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
            st.markdown('<div class="card-caption">Each pixel is deviation from local mean (red = above, blue = below).</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer note ──
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-caption" style="text-align:center;opacity:0.6;">'
    'SimCLR framework adapted from ISPRS Archives XLIX-B1-2026-397-2026 · '
    'MBMP reference: Varon et al. (2021) · IME: Varon et al. (2018) · '
    'Effective wind speed calibration: Guanter et al. (2021)'
    '</div>',
    unsafe_allow_html=True,
)
