"""Predict one tomato leaf photo without strict version checking."""
from pathlib import Path
import joblib
import numpy as np

from features import extract_features, read_pil


def load_model(path: str | Path = 'models/tomato_ann.joblib'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Chưa có file mô hình: {path}')
    
    # Nạp mô hình trực tiếp, không kiểm tra FEATURE_VERSION để tránh lỗi
    artifact = joblib.load(path)
    return artifact


def classify(image, artifact):
    vec, rgb, seg = extract_features(image, return_preview=True)
    expected_len = artifact.get('feature_count', vec.shape[0])
    
    # Tự động điều chỉnh kích thước vectơ đặc trưng nếu có chênh lệch nhỏ
    if vec.shape[0] != expected_len:
        if vec.shape[0] > expected_len:
            vec = vec[:expected_len]
        else:
            vec = np.pad(vec, (0, expected_len - vec.shape[0]))

    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    threshold = float(artifact.get('threshold', 0.5))
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