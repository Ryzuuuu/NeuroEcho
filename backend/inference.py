"""Load models once and run inference on uploaded audio."""
import sys
from pathlib import Path

# allow importing the ml package
sys.path.append(str(Path(__file__).parent.parent))
from ml.src.ensemble import predict  # noqa: E402


def run_inference(wav_path: str) -> dict:
    result = predict(wav_path)
    return result

