"""End-to-end check of the inference pipeline on a synthetic voiced signal."""
import math
import wave

import numpy as np
import pytest

pytest.importorskip("parselmouth")
pytest.importorskip("librosa")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from ml.src.ensemble import predict  # noqa: E402
from ml_config import MODELS_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "biomarker_model.pkl").exists(),
    reason="biomarker model not trained (run ml/src/train_biomarker.py)",
)

SAMPLE_RATE = 16000


def _write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _clean_voice(seconds=3, f0=150):
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE * seconds) / SAMPLE_RATE
    voice = (0.5 * np.sin(2 * np.pi * f0 * t)
             + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
             + 0.1 * np.sin(2 * np.pi * 3 * f0 * t))
    return voice + 0.02 * rng.standard_normal(len(t))


def test_predict_on_clean_voice(tmp_path):
    wav = tmp_path / "clean.wav"
    _write_wav(wav, _clean_voice())

    result = predict(str(wav))

    assert set(result) == {"probability", "band", "biomarkers", "message"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["band"] == "GREEN"
    assert result["biomarkers"]["fo_hz"] == pytest.approx(150, abs=5)
    assert all(math.isfinite(v) for v in result["biomarkers"].values())
