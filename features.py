"""Traditional computer-vision features for tomato leaves; no CNN/deep image encoder.

Keep this module identical at training and inference time. Any changes to the
feature algorithm REQUIRE a new model trained using the same code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

FEATURE_VERSION = "tomato_hsv_lbp_hog_hu_v3_20261119"
IMAGE_SIZE = 256
HOG_SIZE = 64  # 324 HOG dimensions vs 900 at 96x96; less overfitting on 2k photos
MAX_IMAGE_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass
class SegmentationInfo:
    mask: np.ndarray
    coverage: float
    warning: str
    bg_variation: float
    quality_ok: bool = True
    blur_score: float = 0.0
    leaf_color_ratio: float = 0.0


def read_pil(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return _pil_rgb(im)


def _pil_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Anh qua lon; vui long chon anh duoi 24 megapixel.")
    if image.width < 32 or image.height < 32:
        raise ValueError("Anh qua nho (toi thieu 32 x 32 pixels).")
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))
        return white
    return image.convert("RGB")


def to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        im = _pil_rgb(image)
        return np.asarray(im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)
    a = np.asarray(image)
    if a.ndim == 2:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("Anh phai co 1, 3 hoac 4 kenh mau.")
    if a.shape[0] < 32 or a.shape[1] < 32 or a.shape[0] * a.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Kich thuoc anh khong hop le.")
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[2] == 4:
        alpha = a[:, :, 3:4].astype(np.float32) / 255.0
        a = (a[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.resize(a, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _clean_mask(m: np.ndarray) -> np.ndarray:
    m = np.uint8(m > 0) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        k = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        m = np.uint8(labels == k) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(m)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        m = filled
    return m


def segment_leaf(rgb: np.ndarray) -> SegmentationInfo:
    """Heuristic segmentation: background LAB distance + green ExG.

    This is not a botanical leaf detector; quality checks are conservative hints.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = rgb.shape[:2]
    band = max(6, int(min(h, w) * 0.07))
    border = np.concatenate((lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
                             lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)))
    ref = np.median(border, axis=0)
    border_diff = np.linalg.norm(border - ref, axis=1)
    bg_var = float(np.percentile(border_diff, 80))
    delta = np.linalg.norm(lab - ref, axis=2)
    threshold = float(np.clip(np.percentile(border_diff, 90) + 9.0, 13.0, 48.0))
    m_lab = np.uint8(delta > threshold) * 255

    rgb_f = rgb.astype(np.float32)
    exg = 2 * rgb_f[:, :, 1] - rgb_f[:, :, 0] - rgb_f[:, :, 2]
    exg_8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_exg = cv2.threshold(exg_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bg_var > 15 and int(np.count_nonzero(m_exg)) > 200:
        combined = cv2.bitwise_and(m_lab, m_exg)
        m = combined if np.count_nonzero(combined) >= 250 else m_exg
    else:
        m = m_lab
    m = _clean_mask(m)

    coverage = float(np.mean(m > 0))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    leafish = ((hue >= 12) & (hue <= 100) & (sat > 24) & (val > 25)) | (
        (hue < 12) & (sat > 38) & (val < 170) & (rgb[:, :, 1] > rgb[:, :, 2] * 0.65)
    )
    leaf_ratio = float(np.mean(leafish[m > 0])) if np.any(m > 0) else 0.0
    blur = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    warnings = []
    quality_ok = True
    if coverage < 0.06 or coverage > 0.94:
        warnings.append("Khong tach duoc vung la ro rang; thu chup 1 la tren nen don gian.")
        quality_ok = False
    if leaf_ratio < 0.10:
        warnings.append("Mau sac anh khong giong la ca chua; vui long kiem tra anh.")
        quality_ok = False
    if blur < 18:
        warnings.append("Anh co the bi mo; chup lai voi anh sang tot hon.")
        # A blur check alone is not strong enough to refuse classification.
    if bg_var > 32:
        warnings.append("Nen anh phuc tap; ket qua co the kem tin cay hon anh nen don gian.")
    return SegmentationInfo(m, coverage, " ".join(warnings), bg_var, quality_ok, blur, leaf_ratio)


def _masked_hist(channel: np.ndarray, mask: np.ndarray, bins: int, hi: int) -> np.ndarray:
    vals = cv2.calcHist([channel], [0], mask, [bins], [0, hi]).reshape(-1)
    return (vals / max(1.0, float(vals.sum()))).astype(np.float32)


def extract_features(image: Image.Image | np.ndarray, return_preview: bool = False):
    rgb = to_rgb(image)
    seg = segment_leaf(rgb)
    mask = seg.mask
    # For malformed/no-leaf images, retain a deterministic vector so inference
    # can flag quality issues instead of crashing. Never auto-mark these healthy.
    if np.count_nonzero(mask) < 16:
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
        seg.mask = mask
        seg.quality_ok = False
        seg.warning = (seg.warning + " Phan tich tam toan anh.").strip()
    inside = mask > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    hist = [_masked_hist(hsv[:, :, 0], mask, 18, 180),
            _masked_hist(hsv[:, :, 1], mask, 16, 256),
            _masked_hist(hsv[:, :, 2], mask, 16, 256)]
    hv = hsv[inside].astype(np.float32)
    rv = rgb[inside].astype(np.float32)
    stats = np.concatenate((hv.mean(axis=0) / [180, 255, 255],
                            hv.std(axis=0) / [180, 255, 255],
                            rv.mean(axis=0) / 255.0,
                            rv.std(axis=0) / 255.0)).astype(np.float32)

    H, S, V = cv2.split(hsv)
    green = (H >= 25) & (H <= 95) & (S > 20)
    yellow = (H >= 12) & (H < 25) & (S > 40)
    brown = (H < 12) & (S > 40) & (V < 180)
    dark = V < 45
    pale = (S < 25) & (V > 180)
    color_masks = (green, yellow, brown, dark, pale)
    n = max(1, int(inside.sum()))
    proportions = np.array([np.count_nonzero(z & inside) / n for z in color_masks], dtype=np.float32)
    # Disease coloration varies across the leaf, not just globally.
    regional_colors = []
    for row in range(2):
        for col in range(2):
            r = slice(row * 128, (row + 1) * 128)
            c = slice(col * 128, (col + 1) * 128)
            region = inside[r, c]
            denom = max(1, int(region.sum()))
            regional_colors.extend(np.count_nonzero(z[r, c] & region) / denom for z in color_masks)

    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    if cnt is not None:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, ww, hh = cv2.boundingRect(cnt)
        solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-9)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-9)
        shape = np.array([seg.coverage, ww / max(1, hh), solidity,
                          circularity, perimeter / (2 * IMAGE_SIZE)], dtype=np.float32)
    else:
        x, y, ww, hh = 0, 0, 256, 256
        shape = np.zeros(5, dtype=np.float32)

    gray_masked = np.where(inside, gray, 0).astype(np.uint8)
    lbp = local_binary_pattern(gray_masked, P=8, R=1, method="uniform")
    textures = []
    regions = [(slice(None), slice(None))] + [
        (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
        for i in range(2) for j in range(2)]
    for r, c in regions:
        val_lbp = lbp[r, c][inside[r, c]]
        hist_lbp = np.bincount(np.clip(val_lbp.astype(np.int32), 0, 9), minlength=10).astype(np.float32)
        textures.append(hist_lbp / max(1.0, float(hist_lbp.sum())))

    pad = int(max(ww, hh) * 0.08)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(256, x + ww + pad), min(256, y + hh + pad)
    crop = gray_masked[y1:y2, x1:x2]
    crop = cv2.resize(crop, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    gradient = hog(crop, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

    features = np.concatenate((
        *hist, stats, proportions, np.asarray(regional_colors, dtype=np.float32),
        hu, shape, *textures, gradient.astype(np.float32)
    )).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature vector contains NaN or infinity.")
    if return_preview:
        return features, rgb, seg
    return features


def describe_features() -> dict:
    """Group sizes, useful for the report and the compatibility check."""
    return {"HSV_hist": 50, "color_statistics": 12, "color_ratios": 5,
            "regional_color": 20, "Hu_moments": 7, "shape": 5,
            "LBP_texture": 50, "HOG": 324, "total": 473,
            "version": FEATURE_VERSION}
