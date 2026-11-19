"""Single image prediction using the same handcrafted features as training."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from features import FEATURE_VERSION, describe_features, extract_features, read_pil

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "tomato_ann.joblib"


def load_model(path: str | Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chua co model: {path}. Hay chay 'python train.py' truoc.")
    # IMPORTANT: joblib is a pickle-based format. Load only models you trust.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Model file khong dung cau truc. Hay train lai.")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Model dung phien ban feature {artifact.get('feature_version')!r}, "
            f"nhung code hien tai la {FEATURE_VERSION!r}. "
            "Ban can train lai (python train.py) de tranh du doan sai."
        )
    if artifact.get("feature_count") != describe_features()["total"]:
        raise ValueError("Khong khop so chieu dac trung voi model; vui long train lai.")
    threshold = float(artifact.get("threshold", .5))
    if not np.isfinite(threshold) or not .05 <= threshold <= .95:
        raise ValueError("Nguong du doan trong model khong hop le.")
    artifact["threshold"] = threshold
    return artifact


def classify(image, artifact: dict, uncertainty_band: float = 0.08) -> dict:
    feature_vec, rgb, seg = extract_features(image, return_preview=True)
    if feature_vec.shape[0] != int(artifact.get("feature_count", -1)):
        raise ValueError("Dac trung anh khong khop model (chac chan chua train lai).")
    if artifact.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ValueError("Model cu khong tuong thich features.py. Hay train lai.")
    model = artifact["model"]
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Model co thu tu lop khong phai [healthy=0, diseased=1].")
    score = float(model.predict_proba(feature_vec.reshape(1, -1))[0, 1])
    if not np.isfinite(score):
        raise ValueError("Model tra ve diem du doan khong hop le.")
    threshold = float(artifact.get("threshold", .5))
    margin = abs(score - threshold)
    warnings = [seg.warning] if seg.warning else []
    if not seg.quality_ok:
        decision, label, diseased = "invalid", "ANH CHUA PHU HOP DE KET LUAN", None
    elif margin <= uncertainty_band:
        decision, label, diseased = "uncertain", "KET QUA CHUA RO RANG", None
        warnings.append("Diem ANN gan nguong quyet dinh; nen chup anh khac hoac kiem tra truc tiep.")
    elif score >= threshold:
        decision, label, diseased = "diseased", "LA CO DAU HIEU BENH", True
    else:
        decision, label, diseased = "healthy", "KHONG PHAT HIEN DAU HIEU BENH", False
    return {
        "decision": decision,
        "label": label,
        "diseased": diseased,
        "prob_disease": score,
        "prob_healthy": 1.0 - score,
        "threshold": threshold,
        "margin": float(margin),
        "uncertainty_band": float(uncertainty_band),
        "warnings": warnings,
        "rgb": rgb,
        "segmentation": seg,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args()
    result = classify(read_pil(args.image), load_model(args.model))
    print(f"Ket qua: {result['label']}")
    print(f"Diem ANN cho benh: {result['prob_disease']:.3f} | nguong: {result['threshold']:.3f}")
    if result["warnings"]:
        print("Canh bao:", " ".join(result["warnings"]))
