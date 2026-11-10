"""Voice + music bed -> one stereo track at 48 kHz. Music sits a fixed distance under the voice.

usage: python3 -I mix.py narration.wav music.wav out.wav [music_below_voice_db]
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

voice_p, music_p, out = sys.argv[1:4]
below = float(sys.argv[4]) if len(sys.argv) > 4 else 19.0
v, sr = sf.read(voice_p, dtype="float32")
m, msr = sf.read(music_p, dtype="float32")
assert msr == 48000
v = resample_poly(v, 48000, sr).astype(np.float32)
n = min(len(v), len(m))
v, m = v[:n], m[:n]
frame = 2400
act = [np.sqrt(np.mean(v[i:i + frame] ** 2)) for i in range(0, n - frame, frame)]
act = [a for a in act if a > 0.01]
v_rms = float(np.sqrt(np.mean(np.square(act))))
m_rms = float(np.sqrt(np.mean(m ** 2)))
g = v_rms / m_rms * 10 ** (-below / 20)
y = np.stack([v, v], axis=1) + m * g
peak = float(np.max(np.abs(y)))
if peak > 0.95:
    y *= 0.95 / peak
sf.write(out, y, 48000, subtype="PCM_16")
print(f"voice rms {20 * np.log10(v_rms):.1f} dBFS, music gain {20 * np.log10(g):.1f} dB, peak {peak:.2f}")
