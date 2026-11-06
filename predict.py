"""Predict one tomato leaf photo using the exact training feature extraction."""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import extract_features, read_pil
from train import FEATURE_VERSION


def load_model(path: str | Path = 'models/tomato_ann.joblib'):
    # joblib is a pickle format: load ONLY models produced locally by this project.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('Chưa có model. Hãy chạy python train.py trước. File cần: ' + str(path))
    artifact = joblib.load(path)
    if artifact.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Model không khớp phiên bản đặc trưng. Hãy train lại.')
    return artifact


def classify(image, artifact):
    vec, rgb, seg = extract_features(image, return_preview=True)
    if vec.shape[0] != artifact['feature_count']:
        raise ValueError('Số lượng đặc trưng không khớp model; hãy train lại.')
    p = float(artifact['model'].predict_proba(vec[None, :])[0, 1])
    threshold = float(artifact['threshold'])
    is_diseased = p >= threshold
    return {'label': 'CÓ DẤU HIỆU BỆNH' if is_diseased else 'KHÔNG PHÁT HIỆN BỆNH',
            'diseased': bool(is_diseased), 'prob_disease': p,
            'prob_healthy': 1.0-p, 'threshold': threshold,
            'margin': abs(p-threshold), 'rgb': rgb, 'segmentation': seg}


def main():
    pa = argparse.ArgumentParser(description='Kiểm tra một ảnh lá cà chua bằng ANN truyền thống.')
    pa.add_argument('--image', required=True, help='Đường dẫn ảnh, ví dụ images/la1.jpg')
    pa.add_argument('--model', default='models/tomato_ann.joblib')
    pa.add_argument('--save-mask', default='outputs/single_mask.jpg')
    args = pa.parse_args()
    artifact = load_model(args.model)
    result = classify(read_pil(args.image), artifact)
    print('Ảnh:', args.image)
    print('KẾT LUẬN:', result['label'])
    print('Xác suất bệnh (ước lượng ANN):', f'{result["prob_disease"]:.1%}')
    print('Xác suất khỏe (ước lượng ANN):', f'{result["prob_healthy"]:.1%}')
    print('Ngưỡng quyết định từ Validation:', f'{result["threshold"]:.3f}')
    print('Tỉ lệ vùng lá:', f'{result["segmentation"].coverage:.1%}')
    if result['segmentation'].warning:
        print('CẢNH BÁO ẢNH:', result['segmentation'].warning)
    if result['margin'] < 0.12:
        print('CẢNH BÁO: Dự đoán gần ngưỡng; nên chụp lại và tham khảo chuyên gia.')
    p = Path(args.save_mask)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), result['segmentation'].mask)
    print('Đã lưu ảnh mặt nạ:', p)


if __name__ == '__main__':
    main()
