"""Research-grade SimCLR + Sentinel-2 plume segmentation module.

This module is designed to be imported into the user's existing Streamlit app.
It deliberately does not modify CSS, map layout, progress bars, search UI,
reference selection, time series, or existing MBMC code.

The paper specifies the following high-level protocol:
- unlabeled Sentinel-2 imagery for SimCLR pretraining;
- two independent spatial augmentations;
- encoder + three-layer 256-unit projection MLP;
- NT-Xent loss;
- remove projection head for segmentation;
- fixed ConvTranspose + BatchNorm decoder;
- freeze encoder for 10 epochs;
- joint fine-tuning with a lower encoder learning rate;
- compare SimCLR initialization with ImageNet-only initialization.

The paper does not provide exact dataset files, exact AOI list, complete
hyperparameters, full decoder dimensions, or complete IME constants. This
module therefore exposes all unspecified choices as configuration parameters
and never claims exact numerical reproduction of the paper.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError as exc:
    raise ImportError("Install PyTorch before using the research module: pip install torch torchvision") from exc


@dataclass
class SimCLRConfig:
    in_channels: int = 4
    crop_size: int = 128
    projection_hidden: int = 256
    projection_dim: int = 256
    temperature: float = 0.1
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 16
    pretrain_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    fine_tune_epochs: int = 100
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    segmentation_batch_size: int = 4
    validation_fraction: float = 0.2
    seed: int = 42
    threshold: float = 0.5
    checkpoint_dir: str = "simclr_checkpoints"


@dataclass
class TrainHistory:
    ssl_loss: list
    train_loss: list
    validation_loss: list
    validation_f1: list
    validation_iou: list


class TwoViewAugmentation:
    """Geometric SimCLR augmentations suitable for multispectral data."""

    def __init__(self, crop_size: int, cutout_probability: float = 0.4):
        self.crop_size = crop_size
        self.cutout_probability = cutout_probability

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        crop = min(self.crop_size, height, width)
        top = random.randint(0, max(0, height - crop))
        left = random.randint(0, max(0, width - crop))
        image = image[:, top:top + crop, left:left + crop]
        if random.random() < 0.5:
            image = image.flip(-1)
        if random.random() < 0.5:
            image = image.flip(-2)
        image = torch.rot90(image, random.randint(0, 3), dims=(-2, -1))
        if random.random() < 0.5:
            max_shift = max(1, crop // 8)
            image = torch.roll(
                image,
                shifts=(random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)),
                dims=(-2, -1),
            )
        if random.random() < self.cutout_probability:
            cut_h = max(1, crop // 6)
            cut_w = max(1, crop // 6)
            y = random.randint(0, max(0, crop - cut_h))
            x = random.randint(0, max(0, crop - cut_w))
            image = image.clone()
            image[:, y:y + cut_h, x:x + cut_w] = 0.0
        return image.contiguous()

    def __call__(self, image: torch.Tensor):
        return self._one(image), self._one(image.clone())


class UnlabeledTileDataset(Dataset):
    def __init__(self, tiles: Iterable[np.ndarray], transform: TwoViewAugmentation):
        self.tiles = list(tiles)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        value = np.asarray(self.tiles[index], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if value.ndim == 2:
            value = value[None]
        tensor = torch.from_numpy(value)
        return self.transform(tensor)


class SegmentationTileDataset(Dataset):
    def __init__(self, tiles: Iterable[np.ndarray], masks: Iterable[np.ndarray]):
        self.tiles = list(tiles)
        self.masks = list(masks)
        if len(self.tiles) != len(self.masks):
            raise ValueError("tiles and masks must have the same length")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        image = np.asarray(self.tiles[index], dtype=np.float32)
        mask = np.asarray(self.masks[index], dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        if image.ndim == 2:
            image = image[None]
        if mask.ndim == 2:
            mask = mask[None]
        return torch.from_numpy(image), torch.from_numpy(mask)


class MobileNetLikeEncoder(nn.Module):
    """Dependency-free convolutional encoder with MobileNet-like role.

    For exact torchvision MobileNetV2 experiments, replace this encoder with
    torchvision.models.mobilenet_v2 and adapt its first convolution to the
    requested number of channels.
    """

    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width * 8), nn.ReLU(inplace=True),
        )
        self.out_channels = width * 8

    def forward(self, x):
        return self.features(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SimCLR(nn.Module):
    def __init__(self, config: SimCLRConfig):
        super().__init__()
        self.encoder = MobileNetLikeEncoder(config.in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = ProjectionHead(self.encoder.out_channels, config.projection_hidden, config.projection_dim)

    def forward(self, x):
        feature_map = self.encoder(x)
        pooled = self.pool(feature_map).flatten(1)
        return self.projector(pooled), feature_map


class SegmentationDecoder(nn.Module):
    def __init__(self, encoder_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(encoder_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, feature_map, output_size):
        logits = self.layers(feature_map)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class PlumeSegmenter(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_channels: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = SegmentationDecoder(encoder_channels)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features, x.shape[-2:])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nt_xent(z1, z2, temperature: float):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    logits = representations @ representations.T / temperature
    logits.fill_diagonal_(-1e9)
    n = z1.shape[0]
    labels = (torch.arange(2 * n, device=z1.device) + n) % (2 * n)
    return F.cross_entropy(logits, labels)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum()
    dice = (2 * intersection + 1e-6) / (probability.sum() + target.sum() + 1e-6)
    return bce + 1.0 - dice


def batch_metrics(logits, target, threshold: float):
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = (prediction & truth).sum().item()
    fp = (prediction & ~truth).sum().item()
    fn = (~prediction & truth).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "iou": iou, "f1": f1}


def pretrain_simclr(tiles, config: SimCLRConfig, device: Optional[str] = None):
    set_seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UnlabeledTileDataset(tiles, TwoViewAugmentation(config.crop_size))
    loader = DataLoader(dataset, batch_size=config.pretrain_batch_size, shuffle=True, drop_last=False)
    model = SimCLR(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr, weight_decay=config.weight_decay)
    history = []
    model.train()
    for _ in range(config.pretrain_epochs):
        losses = []
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            z1, _ = model(view1)
            z2, _ = model(view2)
            loss = nt_xent(z1, z2, config.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "simclr_encoder.pt"
    torch.save({"config": asdict(config), "encoder": model.encoder.state_dict(), "history": history}, path)
    return model.encoder, path, history


def load_encoder(config: SimCLRConfig, checkpoint: str | Path, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MobileNetLikeEncoder(config.in_channels).to(device)
    state = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(state["encoder"], strict=False)
    return encoder


def fine_tune_segmenter(encoder, tiles, masks, config: SimCLRConfig, device: Optional[str] = None):
    set_seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationTileDataset(tiles, masks)
    n_val = max(1, int(len(dataset) * config.validation_fraction)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    if n_val:
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.seed))
    else:
        train_set, val_set = dataset, None
    train_loader = DataLoader(train_set, batch_size=config.segmentation_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.segmentation_batch_size) if val_set is not None else None
    model = PlumeSegmenter(encoder, encoder.out_channels).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    warm_optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config.decoder_lr, weight_decay=config.weight_decay)
    history = TrainHistory([], [], [], [], [])
    for _ in range(config.warmup_epochs):
        model.train()
        losses = []
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)
            loss = bce_dice_loss(logits, target)
            warm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            warm_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.train_loss.append(float(np.mean(losses)) if losses else float("nan"))
        _evaluate(model, val_loader, device, config, history)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_lr},
        {"params": model.decoder.parameters(), "lr": config.decoder_lr},
    ], weight_decay=config.weight_decay)
    for _ in range(config.fine_tune_epochs):
        model.train()
        losses = []
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)
            loss = bce_dice_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.train_loss.append(float(np.mean(losses)) if losses else float("nan"))
        _evaluate(model, val_loader, device, config, history)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "segmenter.pt"
    torch.save({"config": asdict(config), "model": model.state_dict(), "history": asdict(history)}, path)
    return model, path, history


def _evaluate(model, loader, device, config, history):
    if loader is None:
        history.validation_loss.append(float("nan"))
        history.validation_f1.append(float("nan"))
        history.validation_iou.append(float("nan"))
        return
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)
            losses.append(float(bce_dice_loss(logits, target).cpu()))
            metrics.append(batch_metrics(logits, target, config.threshold))
    history.validation_loss.append(float(np.mean(losses)))
    history.validation_f1.append(float(np.mean([x["f1"] for x in metrics])))
    history.validation_iou.append(float(np.mean([x["iou"] for x in metrics])))


def infer(model, image: np.ndarray, config: SimCLRConfig, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    value = np.asarray(image, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if value.ndim == 2:
        value = value[None]
    with torch.no_grad():
        logits = model(torch.from_numpy(value[None]).to(device))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probability, probability >= config.threshold


def save_config(config: SimCLRConfig, path: str | Path):
    Path(path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
