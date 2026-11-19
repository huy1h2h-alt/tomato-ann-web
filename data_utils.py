"""Lightweight image duplicate checks to avoid optimistic test metrics."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def image_signature(image: Image.Image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    thumb = np.asarray(rgb.resize((40, 40), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.uint8)
    dhash = (gray[:, 1:] > gray[:, :-1]).tobytes()
    return hashlib.sha256(dhash).hexdigest(), thumb


def near_identical(hash_a: str, thumb_a: np.ndarray, hash_b: str, thumb_b: np.ndarray) -> bool:
    """Conservative near-duplicate test; not a semantic similarity detector."""
    if hash_a != hash_b:
        return False
    err = np.abs(thumb_a.astype(np.int16) - thumb_b.astype(np.int16))
    return float(err.mean()) <= 1.2 and float(np.percentile(err, 90)) <= 4.0


def check_split_leakage(df, max_examples: int = 6):
    """Raise for exact/near-identical photos that cross different dataset splits.

    Rows must have path, split and label columns. A near-duplicate test cannot
    guarantee complete leakage detection, e.g. rotated crops of the same leaf.
    """
    sha_seen = {}
    near_buckets = defaultdict(list)
    errors = []
    duplicates_inside_split = 0
    for row in df.itertuples(index=False):
        path = Path(row.path)
        data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = sha_seen.get(data_hash)
        if previous is not None:
            if previous[1] != row.split or previous[2] != row.label:
                errors.append((str(previous[0]), str(path), "byte-identical"))
            else:
                duplicates_inside_split += 1
        else:
            sha_seen[data_hash] = (path, row.split, row.label)
        with Image.open(path) as im:
            dh, thumb = image_signature(im)
        for old_path, old_split, old_label, old_thumb in near_buckets[dh]:
            if near_identical(dh, thumb, dh, old_thumb):
                if old_split != row.split or old_label != row.label:
                    if len(errors) < max_examples:
                        errors.append((str(old_path), str(path), "near-identical"))
                else:
                    duplicates_inside_split += 1
                break
        near_buckets[dh].append((path, row.split, row.label, thumb))
        if len(errors) >= max_examples:
            break
    if errors:
        details = "\n".join(f"  {a} <> {b} [{why}]" for a, b, why in errors[:max_examples])
        raise ValueError(
            "Co anh trung hoac gan giong nhau giua cac tap train/val/test (hoac hai nhan)!\n"
            + details + "\nHay lam sach va chia lai data truoc khi train."
        )
    return {"cross_split_duplicate_checked": True,
            "within_split_duplicate_warnings": int(duplicates_inside_split)}
