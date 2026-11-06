"""Traditional image features (NO CNN): HOG + LBP + HSV + Hu + shape.
The identical feature extractor is imported by training, CLI, and Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = 256
HOG_SIZE = 96


@dataclass
class SegmentationInfo:
    mask: np.ndarray  # uint8 0/255
    coverage: float
    warning: str
    bg_variation: float


def read_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB').copy()


def to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(ImageOps.exif_transpose(image).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        ))
    a = np.asarray(image)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[-1] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
    return cv2.resize(a, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def segment_leaf(rgb: np.ndarray) -> SegmentationInfo:
    """Foreground via robust border-color distance in LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(6, int(min(h, w) * 0.07))
    outer = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)], axis=0)
    base = np.median(outer, axis=0)
    border_delta = np.linalg.norm(outer - base, axis=1)
    variation = float(np.percentile(border_delta, 80))
    delta = np.linalg.norm(lab - base, axis=2)
    threshold = float(np.clip(np.percentile(border_delta, 90) + 9, 13, 47))
    m = (delta > threshold).astype(np.uint8) * 255

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == biggest, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask_filled = np.zeros_like(m)
        cv2.drawContours(mask_filled, contours, -1, 255, thickness=-1)
        m = mask_filled
    coverage = float(np.mean(m > 0))
    warns = []
    if variation > 26:
        warns.append('Nền ảnh có vẻ không đồng nhất; hãy đặt một lá trên nền trơn.')
    if coverage < .07:
        warns.append('Vùng lá quá nhỏ hoặc không tách được; hãy chụp lá gần hơn.')
    if coverage > .87:
        warns.append('Vùng tách quá lớn; tránh lá chạm mép ảnh hoặc nền nhiều màu.')
    if not contours:
        warns.append('Không phát hiện rõ đối tượng lá trong ảnh.')
    return SegmentationInfo(mask=m, coverage=coverage,
                            warning=' '.join(warns), bg_variation=variation)


def _masked_hist(ch: np.ndarray, mask: np.ndarray, bins: int, channel_range: Tuple[int, int]) -> np.ndarray:
    hist = cv2.calcHist([ch], [0], mask, [bins], list(channel_range)).reshape(-1)
    hist = hist / (float(hist.sum()) + 1e-10)
    return hist.astype(np.float32)


def extract_features(image: Image.Image | np.ndarray, return_preview: bool = False):
    """Return a fixed-length 1D float32 array; optionally return RGB and segmentation."""
    rgb = to_rgb(image)
    seg = segment_leaf(rgb)
    m = seg.mask

    # Nếu không tách được lá khỏi nền, tự động dùng toàn bộ khung ảnh làm mask
    if int((m > 0).sum()) < 16:
        m = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
        seg = SegmentationInfo(
            mask=m,
            coverage=1.0,
            warning='Không tách được lá khỏi nền. Sử dụng toàn bộ khung ảnh.',
            bg_variation=seg.bg_variation
        )

    inside = m > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    color = [_masked_hist(hsv[:, :, 0], m, 18, (0, 180)),
             _masked_hist(hsv[:, :, 1], m, 16, (0, 256)),
             _masked_hist(hsv[:, :, 2], m, 16, (0, 256))]
    hsv_vals = hsv[inside].astype(np.float32)
    rgb_vals = rgb[inside].astype(np.float32)
    stats = np.r_[hsv_vals.mean(axis=0) / [180, 255, 255],
                  hsv_vals.std(axis=0) / [180, 255, 255],
                  rgb_vals.mean(axis=0) / 255,
                  rgb_vals.std(axis=0) / 255]
    H, S, V = (hsv[:, :, i] for i in range(3))
    green = (H >= 32) & (H <= 95) & (S > 38)
    yellow = (H >= 16) & (H < 35) & (S > 34)
    brown = (H >= 4) & (H < 22) & (S > 35) & (V < 200)
    dark = V < 65
    pale = (S < 48) & (V > 130)
    n_inside = int(inside.sum())
    proportions = np.array([((x & inside).sum() / n_inside) for x in
                            (green, yellow, brown, dark, pale)], dtype=np.float32)

    mom = cv2.moments(m, binaryImage=True)
    hu = cv2.HuMoments(mom).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    if cnt is not None:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-9)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-9)
        shape = np.array([seg.coverage, w / (h + 1e-9), solidity,
                          circularity, perimeter / (2 * IMAGE_SIZE)], dtype=np.float32)
    else:
        shape = np.zeros(5, dtype=np.float32)

    gray_masked = np.where(inside, gray, 0).astype(np.uint8)
    lbp = local_binary_pattern(gray_masked, P=8, R=1, method='uniform')
    texture = []
    regions = [(slice(None), slice(None))] + [
        (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
        for i in range(2) for j in range(2)
    ]
    for rs, cs in regions:
        values = lbp[rs, cs][inside[rs, cs]]
        hist = np.bincount(np.clip(values.astype(int), 0, 9), minlength=10).astype(np.float32)
        texture.append(hist / (hist.sum() + 1e-9))

    x, y, w, h = cv2.boundingRect(cnt) if cnt is not None else (0, 0, 256, 256)
    pad = int(max(w, h) * .08)
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, 256), min(y + h + pad, 256)
    crop_gray = gray_masked[y1:y2, x1:x2]
    crop_gray = cv2.resize(crop_gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    hog_vec = hog(crop_gray, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

    vec = np.concatenate([*color, stats, proportions, hu, shape,
                          *texture, hog_vec]).astype(np.float32)
    if not np.isfinite(vec).all():
        raise ValueError('Đặc trưng ảnh không hợp lệ (NaN/Inf).')
    if return_preview:
        return vec, rgb, seg
    return vec


def describe_features():
    sample = Image.new('RGB', (256, 256), (220, 220, 220))
    a = np.array(sample)
    cv2.ellipse(a, (128, 128), (75, 35), 30, 0, 360, (35, 130, 50), -1)
    return len(extract_features(a))