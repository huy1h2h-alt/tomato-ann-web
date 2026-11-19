"""Train a classical ANN (scikit-learn MLP) on handcrafted leaf features.

NO CNN, NO transfer learning. Data flow:
    file -> HSV/HOG/LBP/Hu -> StandardScaler -> MLPClassifier
Splits: original train only for optimization; validation for model and threshold
selection; test is evaluated just once after selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import importlib.metadata as package_metadata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, fbeta_score,
    log_loss, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import check_split_leakage
from features import FEATURE_VERSION, describe_features, extract_features, read_pil

LABELS = {0: "KHONG PHAT HIEN BENH", 1: "CO DAU HIEU BENH"}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RNG_SEED = 42


def runtime_metadata() -> dict:
    """Record the training runtime so the Cloud environment can be matched."""
    package_names = {
        "numpy": ("numpy",),
        "scipy": ("scipy",),
        "Pillow": ("Pillow",),
        "opencv": ("opencv-python-headless", "opencv-python"),
        "scikit-image": ("scikit-image",),
        "scikit-learn": ("scikit-learn",),
        "joblib": ("joblib",),
    }
    versions = {}
    for label, candidates in package_names.items():
        for candidate in candidates:
            try:
                versions[label] = package_metadata.version(candidate)
                break
            except package_metadata.PackageNotFoundError:
                continue
    return {"python": platform.python_version(), "packages": versions}


def dataset_manifest(folder: Path, strict_counts: bool = False) -> pd.DataFrame:
    rows = []
    expected = {"train": 700, "val": 150, "test": 150}
    for split in ("train", "val", "test"):
        for label, y in (("healthy", 0), ("diseased", 1)):
            d = folder / split / label
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS and p.is_file()) if d.is_dir() else []
            minimum = 10 if split == "train" else 2
            if len(files) < minimum:
                raise ValueError(f"{d}: co {len(files)} anh, can it nhat {minimum} anh.")
            if strict_counts and len(files) != expected[split]:
                raise ValueError(f"{d}: can dung {expected[split]} anh (hien co {len(files)}).")
            rows.extend(dict(split=split, label=label, y=y, path=str(p.resolve()), name=p.name) for p in files)
    return pd.DataFrame(rows)


def _cache_digest(df: pd.DataFrame, salt: str = "") -> str:
    feature_source = Path(__file__).with_name("features.py").read_bytes()
    records = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in map(Path, df.path)]
    return hashlib.sha256(repr((FEATURE_VERSION, hashlib.sha256(feature_source).hexdigest(), records, salt)).encode()).hexdigest()


def featurize(df: pd.DataFrame, cache_path: Path | None = None) -> np.ndarray:
    digest = _cache_digest(df)
    if cache_path and cache_path.exists():
        try:
            cache = joblib.load(cache_path)  # only load trusted local cache files
            if cache.get("digest") == digest and len(cache["X"]) == len(df):
                print(f"[CACHE] Dung dac trung da luu: {cache_path}", flush=True)
                return cache["X"]
        except (EOFError, KeyError, ValueError, OSError):
            print("[CACHE] Cache hong/khong hop le, dang tinh lai.")
    feats = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            feats.append(extract_features(read_pil(row.path)))
        except Exception as exc:
            raise RuntimeError(f"Khong the trich dac trung anh {row.path}: {exc}") from exc
        if i % 200 == 0 or i == len(df):
            print(f"[FEATURES] {i}/{len(df)} anh", flush=True)
    X = np.vstack(feats).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"digest": digest, "X": X}, cache_path, compress=2)
    return X


def augment_pil(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Mild illumination/orientation shifts only; preserve disease lesion cues."""
    im = image.copy()
    if rng.random() < .5:
        im = ImageOps.mirror(im)
    if rng.random() < .3:
        im = ImageOps.flip(im)
    angle = int(rng.choice((0, 90, 180, 270)))
    if angle:
        im = im.rotate(angle, expand=True)
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(.82, 1.18)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(.86, 1.14)))
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(.85, 1.15)))
    return im


def make_augmented(df_train: pd.DataFrame, copies: int, cache_path: Path | None, seed: int):
    """Only call on the inner-fit partition, NEVER validation/inner-monitor/test."""
    if copies <= 0:
        return np.empty((0, describe_features()["total"]), dtype=np.float32)
    digest = _cache_digest(df_train, salt=f"aug-v3-copies={copies};seed={seed}")
    if cache_path and cache_path.exists():
        try:
            cache = joblib.load(cache_path)
            if cache.get("digest") == digest:
                print(f"[CACHE] Dung anh tang cuong da trich: {cache_path}", flush=True)
                return cache["X"]
        except (EOFError, KeyError, ValueError, OSError):
            pass
    rng = np.random.default_rng(seed)
    feats = []
    total = len(df_train) * copies
    for row in df_train.itertuples(index=False):
        original = read_pil(row.path)
        for _ in range(copies):
            feats.append(extract_features(augment_pil(original, rng)))
        if len(feats) % 400 == 0 or len(feats) == total:
            print(f"[AUGMENT] {len(feats)}/{total}", flush=True)
    X = np.vstack(feats).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"digest": digest, "X": X}, cache_path, compress=2)
    return X


def choose_threshold(y_true, prob, metric: str = "f1") -> float:
    """Tune a predeclared, bounded threshold grid on VAL (not on TEST)."""
    thresholds = np.linspace(.25, .75, 101)
    scores = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        score = float(fbeta_score(y_true, pred, beta=(2 if metric == "f2" else 1), zero_division=0))
        scores.append(score)
    # Prefer 0.5 whenever it is within 0.005 of the maximum, improving stability.
    top = max(scores)
    candidates = [(abs(float(t) - .5), -float(scores[i]), float(t))
                  for i, t in enumerate(thresholds) if scores[i] >= top - .005]
    return min(candidates)[2]


def stats(y_true, prob, threshold):
    pred = (np.asarray(prob) >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_diseased": float(precision_score(y_true, pred, zero_division=0)),
        "recall_diseased": float(recall_score(y_true, pred, zero_division=0)),
        "f1_diseased": float(f1_score(y_true, pred, zero_division=0)),
        "f2_diseased": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "count": int(len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, prob))
    return result


def _fit_with_independent_monitor(X_fit, y_fit, X_monitor, y_monitor,
                                  hidden, alpha, max_epochs, patience, seed):
    """Manual early stop monitors ORIGINAL photos never used for fitting.

    This avoids built-in early_stopping leaking augmented versions of a photo
    into the monitoring split.
    """
    scaler = StandardScaler()
    X_train_z = np.clip(scaler.fit_transform(X_fit), -6.0, 6.0).astype(np.float32)
    X_monitor_z = np.clip(scaler.transform(X_monitor), -6.0, 6.0).astype(np.float32)
    # A clip needs to be carried into inference too; use an explicit pipeline
    # transformer that is importable when joblib re-opens the model.
    from model_components import ClipTransformer
    model = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                          alpha=alpha, learning_rate_init=0.001,
                          batch_size=min(64, len(X_train_z)), shuffle=True,
                          random_state=seed)
    best_model = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(X_train_z, y_fit, classes=np.array([0, 1]))
        monitor_probs = np.clip(model.predict_proba(X_monitor_z)[:, 1], 1e-7, 1 - 1e-7)
        loss = float(log_loss(y_monitor, monitor_probs, labels=[0, 1]))
        if loss < best_loss - 0.0003:
            best_loss = loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience and epoch >= max(15, patience):
            break
    assert best_model is not None
    pipeline = Pipeline([("scaler", scaler), ("clip", ClipTransformer(-6.0, 6.0)), ("ann", best_model)])
    return pipeline, {"epochs_run": epoch, "best_epoch": best_epoch,
                      "monitor_log_loss": best_loss}


def _save_test_plots(y_test, prob, threshold, outputs: Path):
    preds = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Diseased"])
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, colorbar=False, values_format="d", cmap="Blues")
    ax.set_title("ANN - Test confusion matrix")
    fig.tight_layout()
    fig.savefig(outputs / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)
    (outputs / "test_classification_report.txt").write_text(
        classification_report(y_test, preds, target_names=["healthy", "diseased"],
                              zero_division=0), encoding="utf-8")


def main(data_dir: Path, output_dir: Path, model_dir: Path,
         fast=False, augment=1, metric="f1", max_epochs=120, patience=16,
         strict_counts=False, skip_duplicate_check=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = dataset_manifest(data_dir, strict_counts=strict_counts)
    print("[DATA] So anh:\n", pd.crosstab(df.split, df.label), flush=True)
    if not skip_duplicate_check:
        integrity = check_split_leakage(df)
        print("[DATA] Kiem tra leakage:", integrity, flush=True)
    else:
        integrity = {"cross_split_duplicate_checked": False}
        warnings.warn("DA BO QUA KIEM TRA TRUNG ANH! Ket qua test co the lac quan sai.")

    X = featurize(df, output_dir / "feature_cache.joblib")
    if X.shape[1] != describe_features()["total"]:
        raise RuntimeError("Feature dimension mismatch; cannot train.")
    y = df.y.to_numpy(dtype=np.int32)
    splits = df.split.to_numpy()
    mask_tr, mask_val, mask_test = (splits == s for s in ("train", "val", "test"))
    X_train, y_train = X[mask_tr], y[mask_tr]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    fit_idx, monitor_idx = train_test_split(
        np.arange(len(y_train)), test_size=.15, random_state=RNG_SEED,
        stratify=y_train
    )
    df_original_train = df.loc[mask_tr].reset_index(drop=True)
    # Reuse the same augmented data across all candidate architectures.
    if fast:
        augment = 0
        max_epochs = min(max_epochs, 35)
        patience = min(patience, 7)
    X_aug = make_augmented(df_original_train.iloc[fit_idx], augment,
                           output_dir / "augment_cache.joblib" if augment else None,
                           seed=RNG_SEED)
    X_fit = np.vstack([X_train[fit_idx], X_aug]) if len(X_aug) else X_train[fit_idx]
    y_fit = np.concatenate([y_train[fit_idx], np.repeat(y_train[fit_idx], augment)]) if augment else y_train[fit_idx]
    X_monitor, y_monitor = X_train[monitor_idx], y_train[monitor_idx]

    variants = [((64, 32), .01)] if fast else [
        ((64,), .01), ((64, 32), .01), ((128, 64), .01),
        ((64, 32), .05), ((128, 64), .05),
    ]
    leaderboard, winner = [], None
    for hidden, alpha in variants:
        print(f"\n[TRAIN] MLP {hidden}, alpha={alpha}, max_epochs={max_epochs}", flush=True)
        pipeline, train_details = _fit_with_independent_monitor(
            X_fit, y_fit, X_monitor, y_monitor,
            hidden, alpha, max_epochs, patience, RNG_SEED)
        prob = pipeline.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, prob, metric)
        metrics = stats(y_val, prob, threshold)
        record = {"architecture": str(hidden), "alpha": alpha,
                  **train_details, "threshold": threshold, **metrics}
        leaderboard.append(record)
        print("[VAL]", record, flush=True)
        metric_key = "f2_diseased" if metric == "f2" else "f1_diseased"
        # Validation controls candidate selection; TEST is never viewed here.
        key = (metrics[metric_key], metrics["balanced_accuracy"],
               -abs(threshold - .5), -train_details["monitor_log_loss"])
        if winner is None or key > winner["key"]:
            winner = {"model": pipeline, "threshold": threshold,
                      "architecture": hidden, "alpha": alpha,
                      "val_metrics": metrics, "key": key,
                      "train_details": train_details}

    pd.DataFrame(leaderboard).to_csv(output_dir / "validation_models.csv", index=False, encoding="utf-8-sig")
    assert winner is not None
    model = winner["model"]
    threshold = winner["threshold"]
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = stats(y_test, test_prob, threshold)
    _save_test_plots(y_test, test_prob, threshold, output_dir)
    test_rows = df.loc[mask_test, ["path", "name", "label", "y"]].copy()
    test_rows["score_diseased"] = test_prob
    test_rows["predicted"] = (test_prob >= threshold).astype(int)
    test_rows["correct"] = test_rows.y == test_rows.predicted
    test_rows.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact = {
        "model": model, "threshold": float(threshold), "classes": LABELS,
        "feature_version": FEATURE_VERSION, "feature_count": int(X.shape[1]),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": tuple(winner["architecture"]), "alpha": winner["alpha"],
        "validation_metrics": winner["val_metrics"], "test_metrics": test_metrics,
        "runtime": runtime_metadata(),
        "training": {"training_originals": int(len(y_train)),
                     "internal_fit_originals": int(len(fit_idx)),
                     "internal_monitor_originals": int(len(monitor_idx)),
                     "augmentation_per_fit_image": int(augment),
                     "criterion": metric, "data_integrity": integrity,
                     **winner["train_details"]},
    }
    model_path = model_dir / "tomato_ann.joblib"
    temporary = model_dir / "tomato_ann.joblib.tmp"
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(model_path)  # atomic on one filesystem
    print("\n[TEST]", test_metrics, "\n[MODEL]", model_path, flush=True)
    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Traditional MLP ANN tomato leaves, no CNN")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--augment", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--metric", choices=("f1", "f2"), default="f1")
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--strict-counts", action="store_true")
    ap.add_argument("--skip-duplicate-check", action="store_true")
    ap.add_argument("--fast", action="store_true", help="1 ANN config, no augmentation for quick smoke test")
    args = ap.parse_args()
    main(args.data, args.outputs, args.models, fast=args.fast, augment=args.augment,
         metric=args.metric, max_epochs=args.max_epochs, patience=args.patience,
         strict_counts=args.strict_counts, skip_duplicate_check=args.skip_duplicate_check)
