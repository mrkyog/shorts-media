"""Narration + word timing.

Reads an episode JSON, synthesises each line with Kokoro, joins the lines with
the gaps the episode asks for, then gets per-word timings from Whisper and maps
them back onto the caption words.

usage: python3 -I tts_align.py episode.json models_dir build_dir
"""
import difflib
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

episode_path, models_dir, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
ep = json.load(open(episode_path))
os.makedirs(build_dir, exist_ok=True)

TOKEN = re.compile(r"\{([^|}]+)\|([^}]+)\}|(\S+)")


def parse(text):
    """-> list of (say, show, brk).

    {say|show} lets the spoken and written forms differ; a lone "/" starts a new
    caption group and is never spoken.
    """
    out, brk = [], True
    for m in TOKEN.finditer(text):
        if m.group(3) == "/":
            brk = True
            continue
        if m.group(3):
            out.append((m.group(3), m.group(3), brk))
        else:
            out.append((m.group(1), m.group(2), brk))
        brk = False
    return out


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def trim(x, sr, thresh=0.012, pad=0.02):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(x), idx[-1] + int(pad * sr))
    return x[a:b]


# ---------------------------------------------------------------- synthesis
from kokoro_onnx import Kokoro

kokoro = Kokoro(os.path.join(models_dir, "kokoro-v1.0.onnx"), os.path.join(models_dir, "voices-v1.0.bin"))
voice, speed = ep["voice"], ep.get("speed", 1.0)

SR = 24000
clips = []
for line in ep["lines"]:
    pairs = parse(line["text"])
    say = " ".join(p[0] for p in pairs)
    samples, sr = kokoro.create(say, voice=voice, speed=speed, lang="en-us")
    assert sr == SR, sr
    samples = trim(np.asarray(samples, dtype=np.float32), sr)
    clips.append((line, pairs, say, samples))
    print(f"{line['id']:>4} {len(samples) / sr:5.2f}s  {say}")

# ---------------------------------------------------------------- assemble
t = ep.get("lead_in", 0.15)
parts = [np.zeros(int(t * SR), dtype=np.float32)]
placed = []
for line, pairs, say, samples in clips:
    start = t
    parts.append(samples)
    t += len(samples) / SR
    placed.append((line, pairs, say, samples, start, t))
    gap = line.get("gap", 0.25)
    parts.append(np.zeros(int(gap * SR), dtype=np.float32))
    t += int(gap * SR) / SR
parts.append(np.zeros(int(ep.get("tail", 0.8) * SR), dtype=np.float32))
audio = np.concatenate(parts)
peak = float(np.max(np.abs(audio)))
audio = audio / peak * 0.89
sf.write(os.path.join(build_dir, "narration.wav"), audio, SR, subtype="PCM_16")
total = len(audio) / SR
print(f"narration {total:.2f}s  peak {peak:.3f}")

# ---------------------------------------------------------------- alignment
from faster_whisper import WhisperModel

wm = WhisperModel("base.en", device="cpu", compute_type="int8")
PAD = 0.25
timeline = {"id": ep["id"], "duration": round(total, 3), "sr": SR, "lines": []}
report = []
for line, pairs, say, samples, start, end in placed:
    clip = np.concatenate([np.zeros(int(PAD * SR), np.float32), samples, np.zeros(int(PAD * SR), np.float32)])
    # whisper wants 16 kHz mono float32
    n16 = int(len(clip) * 16000 / SR)
    clip16 = np.interp(np.linspace(0, len(clip) - 1, n16), np.arange(len(clip)), clip).astype(np.float32)
    segs, _ = wm.transcribe(clip16, language="en", word_timestamps=True, beam_size=5,
                            condition_on_previous_text=False, vad_filter=False)
    heard = []
    for s in segs:
        for w in s.words or []:
            heard.append((w.word.strip(), max(0.0, w.start - PAD), max(0.0, w.end - PAD)))
    dur = len(samples) / SR

    # spoken tokens, each remembering which caption word it belongs to
    spoken = []
    for ci, (say_w, _show, _brk) in enumerate(pairs):
        for tok in say_w.split():
            spoken.append([tok, ci, None, None])
    a = [norm(s[0]) for s in spoken]
    b = [norm(h[0]) for h in heard]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                spoken[i1 + k][2], spoken[i1 + k][3] = heard[j1 + k][1], heard[j1 + k][2]
        elif tag == "replace":
            t0, t1 = heard[j1][1], heard[j2 - 1][2]
            weights = [max(1, len(a[i])) for i in range(i1, i2)]
            acc, tot = 0, sum(weights)
            for k, i in enumerate(range(i1, i2)):
                spoken[i][2] = t0 + (t1 - t0) * acc / tot
                acc += weights[k]
                spoken[i][3] = t0 + (t1 - t0) * acc / tot
    # fill tokens whisper did not hear by interpolating between neighbours
    i = 0
    while i < len(spoken):
        if spoken[i][2] is None:
            j = i
            while j < len(spoken) and spoken[j][2] is None:
                j += 1
            t0 = spoken[i - 1][3] if i > 0 else 0.0
            t1 = spoken[j][2] if j < len(spoken) else dur
            weights = [max(1, len(a[k])) for k in range(i, j)]
            acc, tot = 0, sum(weights)
            for k in range(i, j):
                spoken[k][2] = t0 + (t1 - t0) * acc / tot
                acc += weights[k - i]
                spoken[k][3] = t0 + (t1 - t0) * acc / tot
            i = j
        else:
            i += 1
    # monotonic, inside the clip
    prev = 0.0
    for s in spoken:
        s[2] = min(max(s[2], prev), dur)
        s[3] = min(max(s[3], s[2] + 0.04), dur)
        prev = s[2]

    words = []
    for ci, (say_w, show, brk) in enumerate(pairs):
        toks = [s for s in spoken if s[1] == ci]
        words.append({"show": show, "say": say_w, "brk": brk,
                      "start": round(start + toks[0][2], 3), "end": round(start + toks[-1][3], 3)})
    timeline["lines"].append({"id": line["id"], "start": round(start, 3), "end": round(end, 3),
                              "say": say, "words": words})
    heard_txt = " ".join(h[0] for h in heard)
    ratio = sm.ratio()
    report.append((line["id"], ratio, say, heard_txt))

json.dump(timeline, open(os.path.join(build_dir, "timeline.json"), "w"), indent=1)
print("\n--- heard back (script vs. transcription of the synthetic voice)")
for lid, ratio, say, heard_txt in report:
    flag = "" if ratio > 0.8 else "   <-- CHECK"
    print(f"{lid:>4} match {ratio:.2f}{flag}\n      say:   {say}\n      heard: {heard_txt}")
