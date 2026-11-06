"""Predict one tomato leaf photo with exact feature matching."""
from pathlib import Path
import joblib

from features import extract_features, read_pil


def load_model(path: str | Path = 'models/tomato_ann.joblib'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Chưa có file mô hình: {path}')
    return joblib.load(path)


def classify(image, artifact):
    vec, rgb, seg = extract_features(image, return_preview=True)
    expected_len = artifact.get('feature_count', vec.shape[0])
    
    # Bắt buộc khớp chính xác kích thước đặc trưng
    if vec.shape[0] != expected_len:
        raise ValueError(
            f"Kích thước đặc trưng không khớp mô hình ({vec.shape[0]} vs {expected_len}). "
            "Cần xóa cache và huấn luyện lại train.py!"
        )

    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    
    # Ngưỡng phân loại chuẩn xác
    THRESHOLD = 0.55
    is_diseased = p >= THRESHOLD

    return {
        'label': 'CÓ DẤU HIỆU BỆNH' if is_diseased else 'KHÔNG PHÁT HIỆN BỆNH',
        'diseased': bool(is_diseased),
        'prob_disease': p,
        'prob_healthy': 1.0 - p,
        'threshold': THRESHOLD,
        'margin': abs(p - THRESHOLD),
        'rgb': rgb,
        'segmentation': seg
    }