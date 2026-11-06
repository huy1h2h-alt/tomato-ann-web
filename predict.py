"""Predict one tomato leaf photo with real-world calibrated threshold."""
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
        if vec.shape[0] > expected_len:
            vec = vec[:expected_len]
        else:
            vec = np.pad(vec, (0, expected_len - vec.shape[0]))

    # Xác suất lá bệnh từ mô hình ANN (từ 0.0 đến 1.0)
    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    
    # Nâng ngưỡng báo bệnh lên 0.62 (62%) để loại bỏ báo động giả trên lá khỏe
    DISEASE_THRESHOLD = 0.62
    is_diseased = p >= DISEASE_THRESHOLD

    return {
        'label': 'CÓ DẤU HIỆU BỆNH' if is_diseased else 'KHÔNG PHÁT HIỆN BỆNH',
        'diseased': bool(is_diseased),
        'prob_disease': p,
        'prob_healthy': 1.0 - p,
        'threshold': DISEASE_THRESHOLD,
        'margin': abs(p - DISEASE_THRESHOLD),
        'rgb': rgb,
        'segmentation': seg
    }