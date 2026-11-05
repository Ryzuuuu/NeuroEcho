"""Runtime settings for the API."""

# Formats Praat/Parselmouth can decode. The bundled web client records WAV.
ALLOWED_AUDIO = {".wav", ".flac", ".mp3"}

# Upload size cap (roughly 2 minutes of 48 kHz stereo 16-bit audio).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
