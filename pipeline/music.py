"""Ambient bed, synthesised from scratch (no samples, nothing licensed).

usage: python3 -I music.py seconds out.wav
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

dur, out = float(sys.argv[1]), sys.argv[2]
SR = 48000
n = int(dur * SR)
t = np.arange(n) / SR
rng = np.random.default_rng(3)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


CHORDS = [
    [38, 45, 50, 53, 57, 64],  # D minor add9
    [34, 41, 46, 50, 53, 57],  # Bb major 7
    [41, 48, 53, 57, 60, 64],  # F major 7
    [36, 43, 48, 52, 55, 62],  # C add9
]
seg = dur / len(CHORDS)
xf = 2.6
mix = np.zeros((n, 2))
for ci, chord in enumerate(CHORDS):
    a, b = ci * seg - xf / 2, (ci + 1) * seg + xf / 2
    env = np.clip((t - a) / xf, 0, 1) * np.clip((b - t) / xf, 0, 1)
    env = np.sin(env * np.pi / 2) ** 2
    if ci == 0:
        env = np.where(t < seg / 2, np.maximum(env, np.clip(t / 0.5, 0, 1)), env)
    if ci == len(CHORDS) - 1:
        env = np.where(t > dur - seg / 2, 1.0, env)
    for k, m in enumerate(chord):
        f = hz(m)
        w = 1.0 / (1 + 0.35 * k)
        trem = 1 + 0.14 * np.sin(2 * np.pi * (0.09 + 0.017 * k) * t + rng.uniform(0, 6.28))
        for side, det in ((0, -0.0013), (1, 0.0013)):
            ph = rng.uniform(0, 6.28)
            tone = np.sin(2 * np.pi * f * (1 + det) * t + ph)
            tone += 0.16 * np.sin(2 * np.pi * 2 * f * (1 + det) * t + ph * 1.7)
            tone += 0.05 * np.sin(2 * np.pi * 3 * f * (1 - det) * t + ph * 0.6)
            mix[:, side] += tone * w * trem * env

# reverb: decaying filtered noise as the impulse response
ir_n = int(3.0 * SR)
ti = np.arange(ir_n) / SR
sos = butter(2, 3800, "lowpass", fs=SR, output="sos")
wet = np.zeros_like(mix)
for side in (0, 1):
    ir = sosfilt(sos, rng.standard_normal(ir_n)) * np.exp(-ti / 0.95)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet[:, side] = fftconvolve(mix[:, side], ir)[:n]
y = 0.6 * mix + 0.5 * wet
y = sosfilt(butter(2, 32, "highpass", fs=SR, output="sos"), y, axis=0)
y *= np.clip(t / 0.4, 0, 1)[:, None] * np.clip((dur - t) / 1.4, 0, 1)[:, None]
y = y / np.max(np.abs(y)) * 0.5
sf.write(out, y.astype(np.float32), SR, subtype="PCM_16")
print("music", out, f"{dur:.2f}s rms {20 * np.log10(np.sqrt(np.mean(y ** 2))):.1f} dBFS")
