# neuroecho_prototype.py
# Standalone rule-based demo — run on a laptop with a recorded .wav clip:
#   python neuroecho_prototype.py voice.wav
# Requires: pip install librosa numpy praat-parselmouth scipy

import librosa
import parselmouth
from parselmouth.praat import call


def extract_biomarkers(wav_path):
    """Extract clinically-validated speech biomarkers from a WAV file."""
    snd = parselmouth.Sound(wav_path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    # Voice tremor / instability biomarkers (Parkinson's indicators)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, point_process], "Get shimmer (local)",
                   0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)  # voice quality

    # Speech timing biomarkers (stroke / cognitive indicators)
    y, sr = librosa.load(wav_path)
    intervals = librosa.effects.split(y, top_db=25)   # detect speech vs pause
    speech_dur = sum((e - s) for s, e in intervals) / sr
    total_dur = len(y) / sr
    pause_ratio = 1 - (speech_dur / total_dur)
    articulation_rate = len(intervals) / speech_dur if speech_dur else 0

    return {
        "jitter_%": round(jitter, 3),
        "shimmer_%": round(shimmer, 3),
        "HNR_dB": round(hnr, 2),
        "pause_ratio": round(pause_ratio, 3),
        "articulation_rate": round(articulation_rate, 2),
    }


def risk_score(b):
    """Classify biomarkers into Green/Yellow/Red risk bands."""
    flags = 0
    if b["jitter_%"] > 1.04:       flags += 1     # elevated jitter -> tremor
    if b["shimmer_%"] > 3.81:      flags += 1     # elevated shimmer
    if b["HNR_dB"] < 20:           flags += 1     # poor voice quality
    if b["pause_ratio"] > 0.45:    flags += 1     # excessive pausing -> cognitive/stroke
    if b["articulation_rate"] < 2: flags += 1

    if flags <= 1:
        return "🟢 GREEN  — Low risk. Routine self-check recommended."
    if flags <= 3:
        return "🟡 YELLOW — Some markers elevated. Re-test & consult a doctor."
    return "🔴 RED    — Multiple markers abnormal. See a neurologist promptly."


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Usage: python neuroecho_prototype.py <voice.wav>")
    path = sys.argv[1]
    b = extract_biomarkers(path)
    print("Speech biomarkers:", b)
    print(risk_score(b))
