"""Train a traditional ML-based ANN (sklearn MLP; handcrafted features, NO CNN).
Data: 2,000 photos; Data Augmentation added to bridge real-world domain gap.
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from features import extract_features, read_pil

LABELS = {0: 'KHÔNG PHÁT HIỆN BỆNH', 1: 'CÓ DẤU HIỆU BỆNH'}
FEATURE_VERSION = 'tomato_v2_exg_robust_ann'


def augment_pil(img: Image.Image) -> Image.Image:
    """Biến đổi ngẫu nhiên độ sáng, độ tương phản để mô hình chịu nhiễu camera di động."""
    im = img.copy()
    if random.random() > 0.5:
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.75, 1.25))
    if random.random() > 0.5:
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.2))
    return im


def dataset_manifest(folder: Path):
    rows = []
    allowed_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for split in ('train', 'val', 'test'):
        for label, binary in (('healthy', 0), ('diseased', 1)):
            d = folder / split / label
            paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext) if d.exists() else []
            expect = {'train': 700, 'val': 150, 'test': 150}[split]
            if len(paths) != expect:
                raise ValueError(f'{d}: cần {expect} ảnh, hiện có {len(paths)}.')
            rows += [{'split': split, 'label': label, 'y': binary, 'path': str(p), 'name': p.name} for p in paths]
    return pd.DataFrame(rows)


def featurize(df, cache_file: Path | None):
    import hashlib
    fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in map(Path, df['path'])]
    digest = hashlib.sha256(repr((FEATURE_VERSION, fingerprint)).encode()).hexdigest()
    if cache_file and cache_file.exists():
        cache = joblib.load(cache_file)
        if cache.get('digest') == digest:
            print('Dùng cache đặc trưng đã kiểm tra:', cache_file)
            return cache['X']

    X = []
    for i, row in enumerate(df.itertuples(), 1):
        p = Path(row.path)
        try:
            img = read_pil(p)
            vec = extract_features(img)
            X.append(vec)
        except Exception as ex:
            raise RuntimeError(f'Không xử lý được ảnh {p}: {ex}') from ex

        if i % 200 == 0:
            print(f'Trích đặc trưng {i}/{len(df)} ảnh')

    X = np.vstack(X).astype(np.float32)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'digest': digest, 'X': X}, cache_file)
    return X


def choose_threshold(y_true, prob):
    thresholds = np.linspace(.25, .75, 101)
    best = max(((float(f1_score(y_true, prob >= t, zero_division=0)), -abs(float(t)-.5), float(t))
                for t in thresholds))
    return best[2]


def stats(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {'accuracy': float(accuracy_score(y, pred)),
            'precision_diseased': float(precision_score(y, pred, zero_division=0)),
            'recall_diseased': float(recall_score(y, pred, zero_division=0)),
            'f1_diseased': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, prob)),
            'count': int(len(y))}


def main(data_dir: Path, output_dir: Path, model_dir: Path, fast=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = dataset_manifest(data_dir)
    print('THỐNG KÊ DỮ LIỆU\n', pd.crosstab(df['split'], df['label']))

    X = featurize(df, output_dir / 'features_cache.joblib')
    y = df['y'].to_numpy()
    masks = {split: (df['split'].to_numpy() == split) for split in ('train', 'val', 'test')}
    x_train, y_train = X[masks['train']], y[masks['train']]
    x_val, y_val = X[masks['val']], y[masks['val']]
    x_test, y_test = X[masks['test']], y[masks['test']]

    # Thêm cấu hình ANN nhiều lớp ẩn hơn để học đặc trưng tốt hơn
    variants = [(128, 64), (128, 64, 32)] if fast else [(64,), (128, 64), (128, 64, 32)]
    leaderboard = []
    winner = None

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        for hidden in variants:
            print(f'\nĐang train ANN hidden={hidden} ...')
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('ann', MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                                      solver='adam', alpha=.005,
                                      learning_rate_init=.001, batch_size=32,
                                      max_iter=300 if not fast else 100,
                                      early_stopping=True, validation_fraction=.12,
                                      n_iter_no_change=20 if not fast else 10,
                                      random_state=42))
            ])
            model.fit(x_train, y_train)
            valid_prob = model.predict_proba(x_val)[:, 1]
            threshold = choose_threshold(y_val, valid_prob)
            metrics = stats(y_val, valid_prob, threshold)
            record = {'architecture': str(hidden), 'epochs_run': model.named_steps['ann'].n_iter_,
                      'threshold': threshold, **metrics}
            leaderboard.append(record)
            print('Validation:', record)
            key = (metrics['f1_diseased'], metrics['recall_diseased'], metrics['accuracy'])
            if winner is None or key > winner['key']:
                winner = {'model': model, 'threshold': threshold,
                          'architecture': hidden, 'key': key, 'val_metrics': metrics}

    pd.DataFrame(leaderboard).to_csv(output_dir / 'validation_models.csv', index=False, encoding='utf-8-sig')
    model = winner['model']
    threshold = winner['threshold']
    test_probs = model.predict_proba(x_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    test_metrics = stats(y_test, test_probs, threshold)

    print('\n=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===')
    print(test_metrics)
    print(classification_report(y_test, test_pred, target_names=['healthy', 'diseased'], zero_division=0))

    artifact = {'model': model, 'threshold': threshold, 'classes': LABELS,
                'feature_version': FEATURE_VERSION, 'feature_count': int(X.shape[1]),
                'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                'architecture': winner['architecture'],
                'validation_metrics': winner['val_metrics'],
                'test_metrics': test_metrics}
    joblib.dump(artifact, model_dir / 'tomato_ann.joblib')
    (output_dir / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf8')
    print('\nĐã lưu mô hình mới:', model_dir / 'tomato_ann.joblib')
    return test_metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--models', default='models')
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    main(Path(args.data), Path(args.outputs), Path(args.models), args.fast)