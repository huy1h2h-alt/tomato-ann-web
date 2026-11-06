"""Robust Image Features for Tomato Leaf Classification (HSV + HOG + LBP + Hu)."""
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
    mask: np.ndarray
    coverage: float
    warning: str
    bg_variation: float


def read_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')
            bg = Image.new('RGB', rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            return bg
        return im.convert('RGB')


def to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        im = ImageOps.exif_transpose(image)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')
            bg = Image.new('RGB', rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            im = bg
        else:
            im = im.convert('RGB')
        return np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))
    
    a = np.asarray(image)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[-1] == 4:
        alpha = a[:, :, 3:] / 255.0
        a = (a[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.resize(a, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def segment_leaf(rgb: np.ndarray) -> SegmentationInfo:
    """Tách lá khỏi nền bằng ExG + LAB distance."""
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
    m_lab = (delta > threshold).astype(np.uint8) * 255

    r, g, b_ch = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b_ch
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_exg = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if variation > 15 and (m_exg > 0).sum() > 100:
        m = cv2.bitwise_and(m_lab, m_exg)
        if (m > 0).sum() < 200:
            m = m_exg
    else:
        m = m_lab

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
    return SegmentationInfo(mask=m, coverage=coverage, warning='', bg_variation=variation)


def _masked_hist(ch: np.ndarray, mask: np.ndarray, bins: int, channel_range: Tuple[int, int]) -> np.ndarray:
    hist = cv2.calcHist([ch], [0], mask, [bins], list(channel_range)).reshape(-1)
    hist = hist / (float(hist.sum()) + 1e-10)
    return hist.astype(np.float32)


def extract_features(image: Image.Image | np.ndarray, return_preview: bool = False):
    rgb = to_rgb(image)
    seg = segment_leaf(rgb)
    m = seg.mask

    if int((m > 0).sum()) < 16:
        m = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
        seg = SegmentationInfo(mask=m, coverage=1.0, warning='Dùng toàn bộ khung ảnh.', bg_variation=seg.bg_variation)

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

    # Đã tinh chỉnh: Mở rộng green (S>20) để bao phủ gân lá; Siết pale (S<25, V>180) tránh nhầm lẫn gân lá
    H, S, V = (hsv[:, :, i] for i in range(3))
    green = (H >= 25) & (H <= 95) & (S > 20)
    yellow = (H >= 12) & (H < 25) & (S > 40)
    brown = (H >= 2) & (H < 12) & (S > 40) & (V < 180)
    dark = (V < 45)
    pale = (S < 25) & (V > 180)
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
        w, h = cv2.boundingRect(cnt)[2:]
        solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-9)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-9)
        shape = np.array([seg.coverage, w / (h + 1e-9), solidity, circularity, perimeter / (2 * IMAGE_SIZE)], dtype=np.float32)
    else:
        shape = np.zeros(5, dtype=np.float32)

    gray_masked = np.where(inside, gray, 0).astype(np.uint8)
    lbp = local_binary_pattern(gray_masked, P=8, R=1, method='uniform')
    texture = []
    regions = [(slice(None), slice(None))] + [
        (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128)) for i in range(2) for j in range(2)
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

    vec = np.concatenate([*color, stats, proportions, hu, shape, *texture, hog_vec]).astype(np.float32)
    if not np.isfinite(vec).all():
        raise ValueError('Đặc trưng ảnh chứa giá trị không hợp lệ (NaN/Inf).')
    if return_preview:
        return vec, rgb, seg
    return vec