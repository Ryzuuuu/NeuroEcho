"""End-to-end test: generate a realistic speech-like signal and run through 
the full backend inference pipeline (same code path as the server)."""
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "ml" / "src"))

# Generate a realistic voiced speech signal (150Hz fundamental with harmonics)
sr = 16000
duration = 5  # seconds
t = np.linspace(0, duration, sr * duration)

# Simulate healthy voice: clean harmonics, minimal noise
f0 = 150
voice = (0.5 * np.sin(2 * np.pi * f0 * t) +
         0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
         0.1 * np.sin(2 * np.pi * 3 * f0 * t) +
         0.02 * np.random.randn(len(t)))  # tiny noise
sf.write("test_healthy_voice.wav", voice, sr)

# Simulate unhealthy voice: irregular harmonics, lots of noise, jitter
jitter_signal = np.zeros_like(t)
phase = 0
for i in range(len(t)):
    dt = 1.0 / sr
    # Add frequency perturbation (jitter)
    f_inst = f0 + 15 * np.random.randn()  # large jitter
    phase += 2 * np.pi * f_inst * dt
    jitter_signal[i] = 0.3 * np.sin(phase)
noisy_voice = jitter_signal + 0.2 * np.random.randn(len(t))
sf.write("test_unhealthy_voice.wav", noisy_voice, sr)

from ensemble import predict

print("=" * 60)
print("TEST 1: HEALTHY VOICE (clean harmonics, minimal noise)")
print("=" * 60)
r1 = predict("test_healthy_voice.wav")
for k, v in r1.items():
    print(f"  {k}: {v}")

print()
print("=" * 60)
print("TEST 2: UNHEALTHY VOICE (irregular pitch, lots of noise)")
print("=" * 60)
r2 = predict("test_unhealthy_voice.wav")
for k, v in r2.items():
    print(f"  {k}: {v}")

# Check for NaN in biomarkers
import math
print()
print("=" * 60)
print("NaN CHECK")
print("=" * 60)
for name, result in [("Healthy", r1), ("Unhealthy", r2)]:
    for bk, bv in result["biomarkers"].items():
        if isinstance(bv, float) and math.isnan(bv):
            print(f"  WARNING: {name} -> {bk} is NaN!")
        else:
            print(f"  OK: {name} -> {bk} = {bv}")
