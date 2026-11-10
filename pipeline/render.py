"""Vertical short renderer (1080x1920, 30 fps).

Draws every frame from an episode storyboard + the word timeline, and encodes one
file per shot so a single shot can be redone without touching the others.

usage:
  python3 -I render.py episode.json timeline.json assets_dir build_dir            # all shots
  python3 -I render.py episode.json timeline.json assets_dir build_dir --frames 1.0,7.5
  python3 -I render.py episode.json timeline.json assets_dir build_dir --shots 3,4
"""
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
F_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_XBOLD = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
F_MED = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
GOLD = (255, 200, 61)
WHITE = (255, 255, 255)
EARTH = (111, 168, 255)

CAP_Y, CAP_SIZE, CAP_MAXW, CAP_STROKE = 1300, 104, 900, 11
TAG_X, TAG_Y = 84, 290

episode_path, timeline_path, assets_dir, build_dir = sys.argv[1:5]
opts = sys.argv[5:]
ep = json.load(open(episode_path))
tl = json.load(open(timeline_path))
LINES = {ln["id"]: ln for ln in tl["lines"]}
NFRAMES = int(round(tl["duration"] * FPS))


# ------------------------------------------------------------------ timing
def clean(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def word(lid, text, n=1):
    hits = [w for w in LINES[lid]["words"] if clean(w["show"]) == clean(text)]
    return hits[n - 1]


def when(ref):
    """'0' | seconds | 'L3' (line start) | 'L3:word' | 'L3:word#2' (word start)."""
    try:
        return float(ref)
    except ValueError:
        pass
    if ":" not in ref:
        return max(0.0, LINES[ref]["start"] - 0.06)
    lid, text = ref.split(":", 1)
    n = 1
    if "#" in text:
        text, n = text.split("#")
        n = int(n)
    return word(lid, text, n)["start"]


def ease(u, kind="linear"):
    u = min(1.0, max(0.0, u))
    if kind == "inout":
        return u * u * (3 - 2 * u)
    if kind == "out":
        return 1 - (1 - u) ** 3
    return u


SHOTS = ep["shots"]
for s in SHOTS:
    t = when(s["at"])
    if ":" in s["at"]:
        t -= 0.04
    s["t0"] = t
    s["f0"] = int(round(t * FPS))
for i, s in enumerate(SHOTS):
    s["f1"] = SHOTS[i + 1]["f0"] if i + 1 < len(SHOTS) else NFRAMES
    s["t1"] = s["f1"] / FPS
    s["i"] = i


# ------------------------------------------------------------------ text
_fonts, _patches = {}, {}


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def text_patch(text, path, size, color=WHITE, alpha=255, tracking=0.0, shadow=0.0):
    """RGBA patch of one line of text; cached."""
    key = (text, path, size, color, alpha, tracking, shadow)
    if key in _patches:
        return _patches[key]
    f = font(path, size)
    pad = 28 if shadow else 4
    if tracking:
        adv = [f.getlength(ch) + tracking for ch in text]
        width = int(sum(adv) - tracking) + 1
    else:
        width = int(f.getlength(text)) + 1
    asc, desc = f.getmetrics()
    height = asc + desc
    img = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))

    def draw_on(d, fill):
        if tracking:
            x = pad
            for ch, a in zip(text, adv):
                d.text((x, pad), ch, font=f, fill=fill)
                x += a
        else:
            d.text((pad, pad), text, font=f, fill=fill)

    if shadow:
        sh = Image.new("L", img.size, 0)
        draw_on(ImageDraw.Draw(sh), 255)
        sh = sh.filter(ImageFilter.GaussianBlur(9))
        sh = sh.point(lambda v: int(min(255, v * 1.6) * shadow))
        black = Image.new("RGBA", img.size, (0, 0, 0, 255))
        black.putalpha(sh)
        img = Image.alpha_composite(img, black)
    top = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_on(ImageDraw.Draw(top), color + (alpha,))
    img = Image.alpha_composite(img, top)
    arr = np.array(img)
    _patches[key] = (arr, pad)
    return _patches[key]


def blit(frame, patch, x, y, alpha=1.0):
    """Alpha-composite an RGBA patch with its top-left corner at (x, y)."""
    ph, pw = patch.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + pw), min(H, y + ph)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    p = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    a = p[..., 3:4].astype(np.float32) * (alpha / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + p[..., :3].astype(np.float32) * a).astype(np.uint8)


def put(frame, text, path, size, x, y, color=WHITE, alpha=1.0, tracking=0.0, shadow=0.6, align="l", opacity=255):
    patch, pad = text_patch(text, path, size, color, opacity, tracking, shadow)
    w = patch.shape[1] - 2 * pad
    if align == "r":
        x = x - w
    elif align == "c":
        x = x - w / 2
    blit(frame, patch, x - pad, y - pad, alpha)
    return w


def glow(frame, x, y, radius, color, strength=1.0):
    """Additive soft dot."""
    r = int(radius * 3)
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + r + 1), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * radius * radius)) * strength
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi += g[..., None] * np.array(color, np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ captions
def disp(w):
    text = re.sub(r"[.,:;]+$", "", w["show"]).upper()
    return re.sub(r"(\d)S$", r"\1s", text)  # 1970s, not 1970S


CHUNKS = []
for ln in tl["lines"]:
    cur = None
    for w in ln["words"]:
        if w.get("brk") or cur is None:
            cur = {"words": [], "line": ln["id"]}
            CHUNKS.append(cur)
        cur["words"].append(w)
for i, c in enumerate(CHUNKS):
    nxt = CHUNKS[i + 1] if i + 1 < len(CHUNKS) else None
    c["start"] = 0.0 if i == 0 else c["words"][0]["start"] - 0.05  # words on screen from the first frame
    nxt_start = nxt["words"][0]["start"] - 0.05 if nxt else 1e9
    if nxt and nxt["line"] == c["line"]:
        c["end"] = nxt_start
    else:
        c["end"] = min(LINES[c["line"]]["end"] + 0.22, nxt_start)
    c["i"] = i

_cap_cache = {}


def caption_patch(ci, active):
    key = (ci, active)
    if key in _cap_cache:
        return _cap_cache[key]
    words = [disp(w) for w in CHUNKS[ci]["words"]]
    size = CAP_SIZE
    while True:
        f = font(F_BLACK, size)
        sp = f.getlength(" ") * 0.92
        ws = [f.getlength(t) for t in words]
        total = sum(ws) + sp * (len(words) - 1)
        if total <= CAP_MAXW:
            rows = [list(range(len(words)))]
            break
        best = None
        for k in range(1, len(words)):
            l1 = sum(ws[:k]) + sp * (k - 1)
            l2 = sum(ws[k:]) + sp * (len(words) - k - 1)
            if best is None or max(l1, l2) < best[0]:
                best = (max(l1, l2), k)
        if best and best[0] <= CAP_MAXW:
            rows = [list(range(best[1])), list(range(best[1], len(words)))]
            break
        size -= 6
    asc, desc = f.getmetrics()
    line_h = int(size * 1.10)
    pad = 44
    row_w = [sum(ws[j] for j in r) + sp * (len(r) - 1) for r in rows]
    pw = int(max(row_w)) + 2 * pad
    ph = line_h * len(rows) + 2 * pad + desc
    pos = {}
    for ri, r in enumerate(rows):
        x = pad + (max(row_w) - row_w[ri]) / 2
        for j in r:
            pos[j] = (x, pad + ri * line_h)
            x += ws[j] + sp
    sh = Image.new("L", (pw, ph), 0)
    d = ImageDraw.Draw(sh)
    for j, t in enumerate(words):
        d.text((pos[j][0], pos[j][1] + 8), t, font=f, fill=255, stroke_width=CAP_STROKE, stroke_fill=255)
    sh = sh.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.6))
    img = Image.new("RGBA", (pw, ph), (0, 0, 0, 255))
    img.putalpha(sh)
    top = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(top)
    for j, t in enumerate(words):
        fill = GOLD if j == active else WHITE
        d.text(pos[j], t, font=f, fill=fill + (255,), stroke_width=CAP_STROKE, stroke_fill=(0, 0, 0, 255))
    img = Image.alpha_composite(img, top)
    _cap_cache[key] = np.array(img)
    return _cap_cache[key]


def draw_caption(frame, t):
    for c in CHUNKS:
        if c["start"] <= t < c["end"]:
            active = 0
            for j, w in enumerate(c["words"]):
                if w["start"] <= t + 0.02:
                    active = j
            patch = caption_patch(c["i"], active)
            age = t - c["start"]
            s = 0.88 + 0.12 * ease(age / 0.12, "out")
            a = min(1.0, age / 0.07 + 0.25)
            if c["i"] == 0:  # the opening frame is the cover: no fade on it
                s, a = 1.0, 1.0
            if s < 0.999:
                patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            ph, pw = patch.shape[:2]
            blit(frame, patch, W / 2 - pw / 2, CAP_Y - ph / 2, a)
            return


# ------------------------------------------------------------------ stat tags
def tag_value_size(final_text):
    size = 124
    while font(F_XBOLD, size).getlength(final_text) > 860 and size > 60:
        size -= 4
    return size


def draw_tag(frame, shot, t):
    tags = shot.get("tags") or []
    live = None
    for tg in tags:
        if "t" not in tg:
            tg["t"] = when(tg["at"]) - 0.03
            if "count" in tg:
                tg["final"] = tg["value"].format(v=tg["count"][1])
            else:
                tg["final"] = tg["value"]
            tg["size"] = tag_value_size(tg["final"])
        if t >= tg["t"]:
            live = tg
    if live is None:
        return
    age = t - live["t"]
    a = ease(age / 0.22, "out")
    dx = -28 * (1 - a)
    if "count" in live:
        lo, hi = live["count"]
        p = ease(age / live.get("dur", 0.6), live.get("ease", "out"))
        text = live["value"].format(v=lo + (hi - lo) * p)
    else:
        text = live["final"]
    x, y = TAG_X + dx, TAG_Y
    size = live["size"]
    block_h = 46 + int(size * 1.12)
    # accent bar
    bar = frame[y + 2:y + block_h, int(x):int(x) + 7].astype(np.float32)
    frame[y + 2:y + block_h, int(x):int(x) + 7] = (bar * (1 - a) + np.array(GOLD, np.float32) * a).astype(np.uint8)
    put(frame, live["label"], F_SEMI, 31, x + 30, y, WHITE, a, tracking=2.2, shadow=0.75, opacity=225)
    put(frame, text, F_XBOLD, size, x + 26, y + 40, WHITE, a, shadow=0.75)


# ------------------------------------------------------------------ backgrounds
def star_layer(seed, n, rmin, rmax, extra=760):
    rng = np.random.default_rng(seed)
    hh = H + extra
    img = np.zeros((hh, W), np.float32)
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, hh, n)
    bs = rng.uniform(0.22, 1.0, n) ** 2.4
    rs = rng.integers(rmin, rmax + 1, n)
    for x, y, b, r in zip(xs, ys, bs, rs):
        cv2.circle(img, (int(x), int(y)), int(r), float(b), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) * 1.7
    return np.clip(img, 0, 1)


STARS_FAR = star_layer(7, 900, 1, 1)
STARS_NEAR = star_layer(11, 160, 1, 2)
_y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
SPACE_BG = np.zeros((H, W, 3), np.float32)
for ch, (top, bot) in enumerate(((9, 2), (12, 3), (26, 9))):
    SPACE_BG[..., ch] = top + (bot - top) * _y
SCRIM = 1 - 0.52 * np.clip(1 - _y / 0.30, 0, 1) ** 1.4 - 0.55 * np.clip((_y - 0.50) / 0.30, 0, 1) ** 1.2


def stars(t, gain=1.0):
    out = np.zeros((H, W), np.float32)
    for layer, speed in ((STARS_FAR, 5.0), (STARS_NEAR, 13.0)):
        M = np.float32([[1, 0, 0], [0, 1, -(layer.shape[0] - H) + speed * t]])
        out += cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR)
    return np.clip(out * gain, 0, 1)


def space(t):
    s = stars(t)
    return np.clip(SPACE_BG + s[..., None] * np.array((232, 238, 255), np.float32), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ image shots
_src, _prep = {}, {}


def source(name):
    if name not in _src:
        path = os.path.join(assets_dir, ep["assets"][name])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        _src[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _src[name]


def image_frame(shot, f, t):
    src = source(shot["src"])
    sh, sw = src.shape[:2]
    cover_h = min(sh, sw * H / W)
    k_cover = H / cover_h
    a, b = shot["from"], shot["to"]
    key = shot["i"]
    if key not in _prep:
        kmax = k_cover * max(a["zoom"], b["zoom"])
        fac = min(1.0, kmax * 1.04)
        work = cv2.resize(src, None, fx=fac, fy=fac, interpolation=cv2.INTER_AREA) if fac < 0.999 else src
        _prep[key] = (work, fac)
    work, fac = _prep[key]
    n = max(1, shot["f1"] - shot["f0"] - 1)
    e = ease((f - shot["f0"]) / n, shot.get("ease", "linear"))
    zoom = a["zoom"] * (b["zoom"] / a["zoom"]) ** e
    cx = (a["c"][0] + (b["c"][0] - a["c"][0]) * e) * sw
    cy = (a["c"][1] + (b["c"][1] - a["c"][1]) * e) * sh
    aa, ab = a.get("anchor", [W / 2, H / 2]), b.get("anchor", [W / 2, H / 2])
    ax, ay = aa[0] + (ab[0] - aa[0]) * e, aa[1] + (ab[1] - aa[1]) * e
    k = k_cover * zoom
    if zoom >= 1.0:  # keep a cover crop inside the picture
        cw, chh = W / k, H / k
        cx = min(max(cx, cw / 2), sw - cw / 2)
        cy = min(max(cy, chh / 2), sh - chh / 2)
    kw = k / fac
    M = np.float32([[kw, 0, ax - kw * cx * fac], [0, kw, ay - kw * cy * fac]])
    interp = cv2.INTER_CUBIC if kw > 1.001 else cv2.INTER_LINEAR
    frame = cv2.warpAffine(work, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if shot.get("stars"):
        s = (stars(t, 0.85)[..., None] * np.array((232, 238, 255), np.float32)).astype(np.uint8)
        frame = np.maximum(frame, s)
    m = (SCRIM * shot.get("dim", 0.92) * 256).astype(np.uint16)[..., None]
    frame = ((frame.astype(np.uint16) * m) >> 8).astype(np.uint8)
    draw_tag(frame, shot, t)
    if shot.get("credit"):
        a_c = ease((t - shot["t0"] - 0.6) / 0.4, "out")
        put(frame, shot["credit"], F_MED, 24, W / 2, 1560, WHITE, a_c, tracking=1.6, shadow=0.7, align="c", opacity=170)
    return frame


# ------------------------------------------------------------------ infographics
X0, X1 = 100, 980


def track(frame, y, alpha):
    c = int(70 * alpha)
    cv2.line(frame, (X0, y), (X1, y), (c, c, c + 8), 2, cv2.LINE_AA)
    cv2.circle(frame, (X0, y), 9, tuple(int(v * alpha) for v in EARTH), -1, cv2.LINE_AA)
    cv2.circle(frame, (X1, y), 11, (int(235 * alpha),) * 3, 2, cv2.LINE_AA)


def race_frame(shot, f, t):
    frame = space(t)
    t0 = shot["t0"]
    a_in = ease((t - t0) / 0.2, "out")
    put(frame, "SAME DISTANCE · ONE LIGHT-DAY", F_SEMI, 31, 84, 296, WHITE, a_in, tracking=2.2, opacity=200)

    # Voyager: crawls while the narration says "49 years"
    v0, v1 = t0 + 0.10, word("L6", "years")["end"] + 0.15
    pv = ease((t - v0) / (v1 - v0), "out")
    yv = 600
    put(frame, "VOYAGER 1", F_XBOLD, 64, 84, 456, WHITE, a_in)
    put(frame, f"{int(round(49 * pv))} YEARS", F_XBOLD, 64, 996, 456, GOLD, a_in, align="r")
    track(frame, yv, a_in)
    xv = X0 + (X1 - X0) * pv
    if pv > 0:
        cv2.line(frame, (X0, yv), (int(xv), yv), GOLD, 5, cv2.LINE_AA)
        glow(frame, xv, yv, 13, GOLD, 0.9)
        cv2.circle(frame, (int(xv), yv), 8, (255, 236, 170), -1, cv2.LINE_AA)

    # light: crosses in a blink when the narration says "light"
    l0 = word("L6", "light")["start"] - 0.03
    pl = ease((t - l0) / 0.28, "inout")
    yl = 900
    put(frame, "LIGHT", F_XBOLD, 64, 84, 756, WHITE, a_in)
    track(frame, yl, a_in)
    put(frame, "EARTH", F_MED, 24, 84, yl + 30, WHITE, a_in, tracking=1.6, opacity=150)
    put(frame, "25.9 BILLION KM", F_MED, 24, 996, yl + 30, WHITE, a_in, tracking=1.6, opacity=150, align="r")
    if t >= l0:
        xl = X0 + (X1 - X0) * pl
        cv2.line(frame, (X0, yl), (int(xl), yl), (255, 255, 255), 5, cv2.LINE_AA)
        glow(frame, xl, yl, 20, (255, 255, 255), 1.0)
        cv2.circle(frame, (int(xl), yl), 8, (255, 255, 255), -1, cv2.LINE_AA)
        if pl >= 1.0:
            age = t - (l0 + 0.28)
            flash = max(0.0, 1 - age / 0.35)
            glow(frame, X1, yl, 34, (255, 255, 255), 0.9 * flash)
            put(frame, "24 HOURS", F_XBOLD, 64, 996, 756, GOLD, ease(age / 0.12, "out"), align="r")
    return frame


def scale_frame(shot, f, t):
    frame = space(t)
    t0 = shot["t0"]
    a_in = ease((t - t0) / 0.2, "out")
    put(frame, "ONE LIGHT-YEAR · TO SCALE", F_SEMI, 31, 84, 296, WHITE, a_in, tracking=2.2, opacity=200)

    bx0, bx1, by0, by1 = 84, 996, 640, 684
    grow = ease((t - t0) / 0.45, "out")
    xe = int(bx0 + (bx1 - bx0) * grow)
    cv2.rectangle(frame, (bx0, by0), (xe, by1), (22, 26, 40), -1)
    cv2.rectangle(frame, (bx0, by0), (xe, by1), (120, 126, 146), 2, cv2.LINE_AA)
    put(frame, "1 LIGHT-YEAR", F_MED, 26, 996, by1 + 18, WHITE, ease((t - t0 - 0.3) / 0.25, "out"),
        tracking=1.6, opacity=170, align="r")

    c0 = word("L9", "At")["start"] - 0.05
    a_c = ease((t - c0) / 0.22, "out")
    if a_c > 0:
        put(frame, "VOYAGER 1 AFTER 49 YEARS", F_SEMI, 31, 84, 452, GOLD, a_c, tracking=2.2)
        put(frame, "0.27% OF THE WAY", F_XBOLD, 72, 82, 494, WHITE, a_c)
        g = tuple(int(v * a_c) for v in GOLD)
        cv2.line(frame, (bx0 + 2, 592), (bx0 + 2, by0), g, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (bx0, by0), (bx0 + 4, by1), g, -1)
        glow(frame, bx0 + 2, (by0 + by1) / 2, 12, GOLD, 0.55 * a_c)

    n0 = word("L9", "17,000")["start"] - 0.03
    if t >= n0:
        age = t - n0
        v = 17000 * ease(age / 0.7, "out")
        a_n = ease(age / 0.15, "out")
        put(frame, f"{v:,.0f}+", F_BLACK, 190, 78, 790, WHITE, a_n)
        put(frame, "YEARS TO GO AT THIS SPEED", F_SEMI, 31, 84, 1012, WHITE, a_n, tracking=2.2, opacity=215)
    return frame


KINDS = {"image": image_frame, "race": race_frame, "scale": scale_frame}


def render(f):
    shot = next(s for s in SHOTS if s["f0"] <= f < s["f1"])
    t = f / FPS
    frame = KINDS[shot["kind"]](shot, f, t)
    draw_caption(frame, t)
    return np.ascontiguousarray(frame)


# ------------------------------------------------------------------ main
ENC = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-video_track_timescale", "15360"]

if opts and opts[0] == "--frames":
    out = os.path.join(build_dir, "preview")
    os.makedirs(out, exist_ok=True)
    for ts in opts[1].split(","):
        f = min(NFRAMES - 1, int(round(float(ts) * FPS)))
        Image.fromarray(render(f)).save(os.path.join(out, f"f_{f:04d}.jpg"), quality=92)
        print("frame", f, f"t={f / FPS:.2f}")
else:
    only = None
    if opts and opts[0] == "--shots":
        only = {int(x) for x in opts[1].split(",")}
    out = os.path.join(build_dir, "shots")
    os.makedirs(out, exist_ok=True)
    for s in SHOTS:
        if only is not None and s["i"] not in only:
            continue
        final = os.path.join(out, f"shot_{s['i']:02d}.mp4")
        tmp = os.path.join(out, f"shot_{s['i']:02d}.part.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-",
               "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p"] + ENC + [tmp]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(s["f0"], s["f1"]):
            p.stdin.write(render(f).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"ffmpeg failed on shot {s['i']}")
        os.replace(tmp, final)
        _prep.pop(s["i"], None)
        print(f"shot {s['i']:02d} {s['kind']:6s} frames {s['f0']}-{s['f1'] - 1} ({(s['f1'] - s['f0']) / FPS:.2f}s)", flush=True)
    with open(os.path.join(build_dir, "concat.txt"), "w") as fh:
        for s in SHOTS:
            fh.write(f"file 'shots/shot_{s['i']:02d}.mp4'\n")
    print("frames", NFRAMES, f"({NFRAMES / FPS:.2f}s)")
