import os
from pathlib import Path

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "ml" / "models"))
ALLOWED_AUDIO = {".wav", ".m4a", ".mp3", ".webm", ".ogg"}
MAX_AUDIO_SECONDS = 120
