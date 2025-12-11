from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage import color, filters

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = BASE_DIR / ".feature_cache"

HIST_FEATURE_DIM = 107
FFT_FEATURE_DIM = 14


def _open_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _cache_path(image_path: Path, cache_dir: Optional[Path], split: str, name: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    hashed = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()
    cache_file = Path(cache_dir) / name / split / f"{hashed}.npy"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    return cache_file


def load_hist_features(path: Path, cache_dir: Optional[Path] = None, split: str = "train") -> np.ndarray:
    cache_file = _cache_path(path, cache_dir or DEFAULT_CACHE_DIR, split, "hist")
    if cache_file and cache_file.exists():
        return np.load(cache_file)
    array = _compute_hist_features(path)
    if cache_file is not None:
        np.save(cache_file, array)
    return array


def load_fft_features(path: Path, cache_dir: Optional[Path] = None, split: str = "train") -> np.ndarray:
    cache_file = _cache_path(path, cache_dir or DEFAULT_CACHE_DIR, split, "fft")
    if cache_file and cache_file.exists():
        return np.load(cache_file)
    array = _compute_fft_features(path)
    if cache_file is not None:
        np.save(cache_file, array)
    return array


def _compute_hist_features(path: Path) -> np.ndarray:
    rgb = _open_image(path)
    hsv = color.rgb2hsv(rgb)
    ycbcr = color.rgb2ycbcr(rgb)
    h_hist, _ = np.histogram(hsv[..., 0], bins=32, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(hsv[..., 1], bins=32, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(hsv[..., 2], bins=32, range=(0.0, 1.0), density=True)
    ycbcr_mean = ycbcr.reshape(-1, 3).mean(axis=0)
    ycbcr_std = ycbcr.reshape(-1, 3).std(axis=0)
    bright_ratio = float((hsv[..., 2] > 0.8).mean())
    warm_mask = ((hsv[..., 0] < 0.08) | (hsv[..., 0] > 0.92)) & (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.5)
    warm_ratio = float(warm_mask.mean())
    smoke_mask = (hsv[..., 1] < 0.25) & (hsv[..., 2] > 0.35)
    smoke_ratio = float(smoke_mask.mean())
    sobel_edges = filters.sobel(color.rgb2gray(rgb))
    edge_density = float((sobel_edges > 0.1).mean())
    contrast = float(rgb.std())
    vector = np.concatenate(
        [
            h_hist,
            s_hist,
            v_hist,
            ycbcr_mean,
            ycbcr_std,
            np.array([bright_ratio, warm_ratio, smoke_ratio, contrast, edge_density]),
        ]
    ).astype(np.float32)
    return vector


def _compute_fft_features(path: Path) -> np.ndarray:
    rgb = _open_image(path)
    gray = color.rgb2gray(rgb)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft)
    total = magnitude.sum() + 1e-8
    norm = magnitude / total
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = min(cy, cx)
    low = norm[dist <= radius * 0.15].sum()
    mid = norm[(dist > radius * 0.15) & (dist <= radius * 0.35)].sum()
    high = norm[(dist > radius * 0.35)].sum()
    horiz = norm[cy, :].sum()
    vert = norm[:, cx].sum()
    diag1 = np.trace(norm) / norm.shape[0]
    diag2 = np.trace(np.fliplr(norm)) / norm.shape[0]
    percentiles = np.percentile(magnitude, [50, 75, 90]) / (magnitude.max() + 1e-8)
    entropy = float(-np.sum(norm * np.log(norm + 1e-10)))
    sparsity = float(np.mean(norm > norm.mean()))
    high_low_ratio = float(high / (low + 1e-8))
    vector = np.array(
        [
            low,
            mid,
            high,
            horiz,
            vert,
            diag1,
            diag2,
            float(gray.std()),
            percentiles[0],
            percentiles[1],
            percentiles[2],
            entropy,
            sparsity,
            high_low_ratio,
        ],
        dtype=np.float32,
    )
    return vector
