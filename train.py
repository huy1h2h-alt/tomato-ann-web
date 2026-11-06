"""Train a traditional ML-based ANN (sklearn MLP; handcrafted features, NO CNN).

Data: exactly 2,000 photos, balanced; train 1,400; val 300; test 300.
Model selection uses ONLY validation; test is evaluated once.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from features import extract_features, read_pil

LABELS = {0: 'KHÔNG PHÁT HIỆN BỆNH', 1: 'CÓ DẤU HIỆU BỆNH'}
FEATURE_VERSION = 'tomato_v1_hog_lbp_hsv_hu'


def dataset_manifest(folder: Path):
    # Explicit split structure, never copy/transforms between splits.
    rows = []
    allowed_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for split in ('train', 'val', 'test'):
        for label, binary in (('healthy', 0), ('diseased', 1)):
            d = folder / split / label
            paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext) if d.exists() else []
            expect = {'train': 700, 'val': 150, 'test': 150}[split]
            if len(paths) != expect:
                raise ValueError(f'{d}: cần {expect} ảnh, hiện có {len(paths)}. '
                                 'Hãy chạy download_2000.py hoặc prepare_local.py trước.')
            rows += [{'split': split, 'label': label, 'y': binary, 'path': str(p),
                      'name': p.name} for p in paths]
    return pd.DataFrame(rows)


def featurize(df, cache_file: Path | None):
    fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in map(Path, df['path'])]
    # No reuse if feature logic changes or files change.
    import hashlib
    digest = hashlib.sha256(repr((FEATURE_VERSION, fingerprint)).encode()).hexdigest()
    if cache_file and cache_file.exists():
        cache = joblib.load(cache_file)
        if cache.get('digest') == digest:
            print('Dùng cache đặc trưng đã kiểm tra:', cache_file)
            return cache['X']
    X = []
    for i, p in enumerate(df['path'], 1):
        try:
            X.append(extract_features(read_pil(p)))
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
    # Use validation ONLY; optimize positive-class F1. Disease false negatives
    # are clinically/agronomically important, so F2 can be configured instead.
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
    if (data_dir / 'manifest.csv').exists():
        source_df = pd.read_csv(data_dir / 'manifest.csv')
        if 'leaf_group' in source_df and 'split' in source_df:
            groups = source_df.groupby('leaf_group')['split'].nunique()
            if int((groups > 1).sum()) > 0:
                raise ValueError('DATA LEAKAGE: Có leaf_group nằm ở nhiều split!')

    X = featurize(df, output_dir / 'features_cache.joblib')
    y = df['y'].to_numpy()
    masks = {split: (df['split'].to_numpy() == split) for split in ('train', 'val', 'test')}
    x_train, y_train = X[masks['train']], y[masks['train']]
    x_val, y_val = X[masks['val']], y[masks['val']]
    x_test, y_test = X[masks['test']], y[masks['test']]
    variants = [(64,), (128, 64)] if fast else [(64,), (128,), (128, 64)]
    leaderboard = []
    winner = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        for hidden in variants:
            print(f'\nĐang train ANN hidden={hidden} ...')
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('ann', MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                                      solver='adam', alpha=.01,
                                      learning_rate_init=.001, batch_size=64,
                                      max_iter=250 if not fast else 80,
                                      early_stopping=True, validation_fraction=.12,
                                      n_iter_no_change=17 if not fast else 9,
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
            # Select ONLY on validation F1; test never used in model selection.
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
    print('\n=== TEST CHƯA TỪNG DÙNG ĐỂ CHỌN MODEL ===')
    print(test_metrics)
    print(classification_report(y_test, test_pred, target_names=['healthy', 'diseased'], zero_division=0))

    test_details = df.loc[masks['test'], ['path', 'name', 'label']].copy()
    test_details['du_doan'] = np.where(test_pred == 1, 'diseased', 'healthy')
    test_details['xac_suat_benh'] = test_probs
    test_details['dung_sai'] = np.where(test_pred == y_test, 'ĐÚNG', 'SAI')
    test_details.to_csv(output_dir / 'test_300_chi_tiet.csv', index=False, encoding='utf-8-sig')

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Diseased']).plot(ax=ax, cmap='Greens', values_format='d')
    ax.set_title('Confusion Matrix - 300 holdout test images')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=170)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC AUC = {test_metrics["roc_auc"]:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC - Test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'roc_test.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(model.named_steps['ann'].loss_curve_) + 1),
            model.named_steps['ann'].loss_curve_, label='Train Loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title=f'ANN training loss - {winner["architecture"]}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'training_loss.png', dpi=170)
    plt.close(fig)

    artifact = {'model': model, 'threshold': threshold, 'classes': LABELS,
                'feature_version': FEATURE_VERSION, 'feature_count': int(X.shape[1]),
                'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                'architecture': winner['architecture'],
                'validation_metrics': winner['val_metrics'],
                'test_metrics': test_metrics}
    joblib.dump(artifact, model_dir / 'tomato_ann.joblib')
    (output_dir / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf8')
    print('\nĐã lưu model:', model_dir / 'tomato_ann.joblib')
    print('Đã lưu kết quả:', output_dir)
    if test_metrics['accuracy'] < .8:
        print('LƯU Ý: Test Accuracy dưới 80%. Cần thêm dữ liệu thật/đặc trưng hoặc thay mô hình.')
    else:
        print('Test Accuracy >=80% trên holdout PlantVillage. Không suy ra >=80% ảnh ngoài thực tế.')
    return test_metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train ANN truyền thống phân loại bệnh trên lá cà chua.')
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--models', default='models')
    ap.add_argument('--fast', action='store_true', help='Dùng 2 cấu hình và số epoch ít hơn để test code.')
    args = ap.parse_args()
    main(Path(args.data), Path(args.outputs), Path(args.models), args.fast)
