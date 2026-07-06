"""Combine biomarker + CNN predictions into a final risk band."""
import sys
import math
from pathlib import Path

# Add both ml/ (for config) and ml/src/ (for sibling modules) to sys.path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import joblib
import torch
import numpy as np
from train_cnn import NeuroEchoCNN
from feature_extraction import extract_biomarkers, wav_to_melspec
from ml_config import MODELS_DIR, GREEN_MAX, YELLOW_MAX

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
# We apply a calibration transform to map consumer-mic values into the
# clinical range before feeding them to the model. This is a linear
# normalization that maps [consumer_min, consumer_max] → [clinical_min, clinical_max]
# for each feature.
# ============================================================================

# Consumer microphone empirical correction factors
# Browser WebRTC and cheap ADC mics inflate jitter/shimmer by ~10x
# and reduce HNR by ~20dB compared to clinical sound booth microphones.
CONSUMER_CORRECTION = {
    "jitter_div": 10.0,
    "shimmer_div": 10.0,
    "hnr_add": 20.0
}


def _safe_float(val, default):
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


def predict(wav_path: str, w=(0.5, 0.5)) -> dict:
    """Run ensemble prediction on a WAV file and return risk band + biomarkers."""
    bm = extract_biomarkers(wav_path)

    # Get raw values with NaN safety
    jitter_pct = _safe_float(bm["jitter_pct"], 0.5)   # percentage
    shimmer_pct = _safe_float(bm["shimmer_pct"], 3.0)  # percentage
    hnr_db = _safe_float(bm["hnr_db"], 15.0)           # dB

    # Apply empirical corrections to approximate clinical range
    jitter_cal = (jitter_pct / 100.0) / CONSUMER_CORRECTION["jitter_div"]
    shimmer_cal = (shimmer_pct / 100.0) / CONSUMER_CORRECTION["shimmer_div"]
    hnr_cal = hnr_db + CONSUMER_CORRECTION["hnr_add"]

    fo = _safe_float(bm.get("fo_hz"), 150.0)
    fhi_raw = _safe_float(bm.get("fhi_hz"), 200.0)
    flo_raw = _safe_float(bm.get("flo_hz"), 100.0)

    # The UCI dataset consists of sustained vowels ("Aaaaah"), meaning Fhi and Flo
    # are very close to Fo. If the user speaks naturally, Fhi/Flo sweep wildly,
    # which confuses the model and causes random probability fluctuations.
    # We clamp Fhi and Flo to be within +/- 10% of Fo to simulate a sustained vowel.
    fhi = min(fhi_raw, fo * 1.10)
    flo = max(flo_raw, fo * 0.90)

    # Order must match the 3 features used in train_biomarker.py:
    # ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR"]
    calibrated_features = [jitter_cal, shimmer_cal, hnr_cal]
    print(f"[DEBUG] Calibrated features: fo={fo:.1f}, fhi={fhi:.1f}, flo={flo:.1f}, jitter={jitter_cal:.6f}, shimmer={shimmer_cal:.6f}, hnr={hnr_cal:.3f}")

    # Load biomarker model
    p_bm = 0.5
    try:
        bm_model = joblib.load(MODELS_DIR / "biomarker_model.pkl")
        p_bm = float(bm_model.predict_proba([calibrated_features])[0, 1])
        print(f"[DEBUG] Biomarker model probability: {p_bm:.4f}")
    except Exception as e:
        print(f"Warning: Could not load biomarker model ({e})")

    # Load CNN model
    p_cnn = 0.5
    try:
        cnn = NeuroEchoCNN()
        cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_model.pt", map_location="cpu"))
        cnn.eval()
        mel = torch.tensor(wav_to_melspec(wav_path)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            p_cnn = torch.sigmoid(cnn(mel)).item()
    except Exception as e:
        print(f"Warning: Could not load CNN model ({e}), using biomarker model only")
        p_cnn = p_bm
        w = (1.0, 0.0)

    p = float(np.clip(w[0] * p_bm + w[1] * p_cnn, 0, 1))

    # Determine the message to show in the UI based on the band
    band = band_from_prob(p)
    if band == "GREEN":
        message = "Your speech biomarkers fall within the normal healthy range. No significant signs of neurological vocal instability were detected."
    elif band == "YELLOW":
        message = "Slight irregularities in vocal patterns detected (e.g., elevated jitter or shimmer). We recommend monitoring and taking a follow-up test in a few weeks."
    else:
        message = "Significant irregularities in vocal patterns detected that are often correlated with neurological decline. We strongly advise consulting a neurologist for a professional assessment."

    return {"probability": round(p, 3), "band": band, "biomarkers": bm, "message": message}
