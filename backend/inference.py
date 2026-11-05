"""Thin wrapper that exposes the ML ensemble to the API layer."""
import sys
from pathlib import Path

# Make the repository root importable so the `ml` package resolves.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml.src.ensemble import ModelNotAvailableError, predict  # noqa: E402,F401


def run_inference(wav_path: str) -> dict:
    """Run the ensemble on an audio file and return the prediction payload."""
    return predict(wav_path)
