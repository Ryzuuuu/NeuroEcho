from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000          # smartphone-grade
N_MELS = 64
MAX_LEN = 256
RANDOM_STATE = 42

# Risk thresholds for ensemble probability
# Risk thresholds for ensemble probability
GREEN_MAX = 0.60   # Normal healthy variation (accounts for 75% dataset base rate)
YELLOW_MAX = 0.85  # Moderate risk (requires monitoring)
                   # > 85% is high risk (RED)
