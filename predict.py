"""Predict one tomato leaf photo using the exact robust feature extraction."""
import argparse
from pathlib import Path

import cv2
import joblib

from features import extract_features, read_pil
from train import FEATURE_VERSION


def load_model(path: str | Path = 'models/tomato_ann.joblib'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('Chưa có file mô hình: ' + str(path))
    artifact = joblib.load(path)
    if artifact.get('feature_version') != FEATURE_VERSION:
        raise ValueError(f"Phiên bản đặc trưng không khớp (Model: {artifact.get('feature_version')}, Code: {FEATURE_VERSION}). Hãy chạy train.py lại.")
    return artifact


def classify(image, artifact):
    vec, rgb, seg = extract_features(image, return_preview=True)
    if vec.shape[0] != artifact['feature_count']:
        raise ValueError('Số lượng đặc trưng không khớp mô hình; hãy huấn luyện lại.')
    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    threshold = float(artifact['threshold'])
    is_diseased = p >= threshold
    return {'label': 'CÓ DẤU HIỆU BỆNH' if is_diseased else 'KHÔNG PHÁT HIỆN BỆNH',
            'diseased': bool(is_diseased), 'prob_disease': p,
            'prob_healthy': 1.0 - p, 'threshold': threshold,
            'margin': abs(p - threshold), 'rgb': rgb, 'segmentation': seg}


def main():
    pa = argparse.ArgumentParser(description='Kiểm tra một ảnh lá cà chua bằng ANN.')
    pa.add_argument('--image', required=True)
    pa.add_argument('--model', default='models/tomato_ann.joblib')
    args = pa.parse_args()
    artifact = load_model(args.model)
    result = classify(read_pil(args.image), artifact)
    print('KẾT LUẬN:', result['label'])
    print('Xác suất bệnh:', f'{result["prob_disease"]:.1%}')
    print('Xác suất khỏe:', f'{result["prob_healthy"]:.1%}')


if __name__ == '__main__':
    main()