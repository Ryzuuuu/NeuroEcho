"""Combine biomarker + CNN predictions into a final risk band."""
import logging
import math
import sys
from functools import lru_cache
from pathlib import Path

# Add both ml/ (for config) and ml/src/ (for sibling modules) to sys.path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from ml_config import MODELS_DIR, GREEN_MAX, YELLOW_MAX  # noqa: E402

logger = logging.getLogger(__name__)

# ============================================================================
# CALIBRATION FOR CONSUMER MICROPHONES
# ============================================================================
# The UCI Parkinson's dataset was recorded with professional clinical
# microphones in a sound booth. Consumer laptop/phone microphones produce
# drastically different baseline biomarker values due to:
#   - Ambient noise → inflates jitter/shimmer, crushes HNR
#   - Cheap ADC → adds quantization noise
#   - AGC (auto gain control) → amplitude fluctuations inflate shimmer
#
# UCI clinical baselines (healthy voice, professional mic):
#   Jitter:  0.003 - 0.006  (ratio)
#   Shimmer: 0.015 - 0.035  (ratio)
#   HNR:     19 - 27 dB
#
# Consumer mic baselines (healthy voice, laptop/phone):
#   Jitter:  0.02 - 0.06   (ratio)  ~10x higher
#   Shimmer: 0.10 - 0.25   (ratio)  ~7x higher
#   HNR:     1 - 10 dB              ~15-20 dB lower
#
# Before the features reach the biomarker model they are shifted towards the
# clinical range with the empirical correction below: jitter and shimmer are
# divided by a constant and a constant offset is added to HNR.
# ============================================================================
CONSUMER_CORRECTION = {
    "jitter_div": 10.0,
    "shimmer_div": 10.0,
    "hnr_add": 20.0,
}


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained biomarker model cannot be loaded."""


def safe_float(val, default):
    """Return val if it's a valid finite float, otherwise default."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def band_from_prob(p: float) -> str:
    if p < GREEN_MAX:
        return "GREEN"
    if p < YELLOW_MAX:
        return "YELLOW"
    return "RED"


MESSAGES = {
    "GREEN": (
        "Your speech biomarkers fall within the normal healthy range. "
        "No significant signs of neurological vocal instability were detected."
    ),
    "YELLOW": (
        "Slight irregularities in vocal patterns detected (e.g., elevated jitter or "
        "shimmer). We recommend monitoring and taking a follow-up test in a few weeks."
    ),
    "RED": (
        "Significant irregularities in vocal patterns detected that are often "
        "correlated with neurological decline. We strongly advise consulting a "
        "neurologist for a professional assessment."
    ),
}


@lru_cache(maxsize=1)
def _load_biomarker_model():
    import joblib

    path = MODELS_DIR / "biomarker_model.pkl"
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelNotAvailableError(
            f"Could not load biomarker model from {path}: {exc}. "
            "Train it with `python ml/src/train_biomarker.py`."
        ) from exc


@lru_cache(maxsize=1)
def _load_cnn():
    """Return the CNN, or None when it has not been trained (it is optional)."""
    path = MODELS_DIR / "cnn_model.pt"
    if not path.exists():
        return None
    try:
        import torch
        from train_cnn import NeuroEchoCNN

        cnn = NeuroEchoCNN()
        cnn.load_state_dict(torch.load(path, map_location="cpu"))
        cnn.eval()
        return cnn
    except Exception as exc:
        logger.warning("Could not load CNN model (%s); using biomarker model only", exc)
        return None


def predict(wav_path: str, w=(0.5, 0.5)) -> dict:
    """Run ensemble prediction on an audio file and return risk band + biomarkers."""
    from feature_extraction import extract_biomarkers, wav_to_melspec

    bm = extract_biomarkers(wav_path)

    # Raw values with NaN safety
    jitter_pct = safe_float(bm["jitter_pct"], 0.5)    # percentage
    shimmer_pct = safe_float(bm["shimmer_pct"], 3.0)  # percentage
    hnr_db = safe_float(bm["hnr_db"], 15.0)           # dB

    # Apply empirical corrections to approximate the clinical range
    jitter_cal = (jitter_pct / 100.0) / CONSUMER_CORRECTION["jitter_div"]
    shimmer_cal = (shimmer_pct / 100.0) / CONSUMER_CORRECTION["shimmer_div"]
    hnr_cal = hnr_db + CONSUMER_CORRECTION["hnr_add"]

    # Order must match the 3 gender-neutral features used in train_biomarker.py:
    # ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR"]. Absolute pitch (Fo/Fhi/Flo) is
    # deliberately excluded because it is heavily gender-biased.
    calibrated_features = [jitter_cal, shimmer_cal, hnr_cal]
    logger.debug(
        "Calibrated features: jitter=%.6f shimmer=%.6f hnr=%.3f", *calibrated_features
    )

    p_bm = float(_load_biomarker_model().predict_proba([calibrated_features])[0, 1])
    logger.debug("Biomarker model probability: %.4f", p_bm)

    cnn = _load_cnn()
    if cnn is None:
        p_cnn, w = p_bm, (1.0, 0.0)
    else:
        import torch

        mel = torch.tensor(wav_to_melspec(wav_path)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            p_cnn = torch.sigmoid(cnn(mel)).item()

    p = min(1.0, max(0.0, w[0] * p_bm + w[1] * p_cnn))
    band = band_from_prob(p)
    return {"probability": round(p, 3), "band": band, "biomarkers": bm, "message": MESSAGES[band]}
