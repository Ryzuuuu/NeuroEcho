import os
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
# Override with the MODEL_DIR environment variable (used by the Docker image).
MODELS_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))

SAMPLE_RATE = 16000          # smartphone-grade
N_MELS = 64
MAX_LEN = 256
RANDOM_STATE = 42

# Risk thresholds for ensemble probability
GREEN_MAX = 0.60   # Normal healthy variation (accounts for 75% dataset base rate)
YELLOW_MAX = 0.85  # Moderate risk (requires monitoring)
                   # > 85% is high risk (RED)
