"""Extract clinically-validated speech biomarkers + mel-spectrograms."""
import sys
from pathlib import Path

# Add ml/ to sys.path for config import
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from ml_config import SAMPLE_RATE, N_MELS, MAX_LEN


def extract_biomarkers(wav_path: str) -> dict:
    """Return a dict of voice biomarkers from a WAV file."""
    snd = parselmouth.Sound(wav_path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, point_process], "Get shimmer (local)",
                   0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)

    # Extract pitch (fundamental frequency) stats
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    fo = call(pitch, "Get mean", 0, 0, "Hertz")
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur = sum((e - s) for s, e in intervals) / sr
    total_dur = len(y) / sr
    pause_ratio = 1 - (speech_dur / total_dur) if total_dur else 0
    articulation_rate = len(intervals) / speech_dur if speech_dur else 0

    return {
        "fo_hz": round(fo, 3) if fo else 0.0,
        "fhi_hz": round(fhi, 3) if fhi else 0.0,
        "flo_hz": round(flo, 3) if flo else 0.0,
        "jitter_pct": round(jitter, 4),
        "shimmer_pct": round(shimmer, 4),
        "hnr_db": round(hnr, 3),
        "pause_ratio": round(pause_ratio, 4),
        "articulation_rate": round(articulation_rate, 3),
    }


def wav_to_melspec(wav_path: str) -> np.ndarray:
    """Convert WAV to a normalized fixed-size mel-spectrogram [N_MELS, MAX_LEN]."""
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]
    return mel.astype(np.float32)
