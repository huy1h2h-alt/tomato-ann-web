"""Predict one tomato leaf photo using model's calibrated optimal threshold."""
from pathlib import Path
import joblib
import numpy as np

from features import extract_features, read_pil


def load_model(path: str | Path = 'models/tomato_ann.joblib'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Chưa có file mô hình: {path}')
    return joblib.load(path)


def classify(image, artifact):
    vec, rgb, seg = extract_features(image, return_preview=True)
    expected_len = artifact.get('feature_count', vec.shape[0])
    
    if vec.shape[0] != expected_len:
        raise ValueError(
            f"Kích thước đặc trưng không khớp mô hình ({vec.shape[0]} vs {expected_len}). "
            "Hãy kiểm tra lại file features.py!"
        )

    # Tính xác suất bệnh từ mô hình ANN
    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    
    # Tự động lấy ngưỡng tối ưu (0.45) đã lưu trong file model .joblib
    threshold = float(artifact.get('threshold', 0.45))
    is_diseased = p >= threshold

    return {
        'label': 'CÓ DẤU HIỆU BỆNH' if is_diseased else 'KHÔNG PHÁT HIỆN BỆNH',
        'diseased': bool(is_diseased),
        'prob_disease': p,
        'prob_healthy': 1.0 - p,
        'threshold': threshold,
        'margin': abs(p - threshold),
        'rgb': rgb,
        'segmentation': seg
    }