#!/usr/bin/env python3
"""
music_viz.py - turn an audio file into a glowing, beat-driven MP4.

Usage:
    python music_viz.py song.mp3
    python music_viz.py song.mp3 -o song_viz.mp4 --size 1280x720 --fps 30

Speed:
    On a machine with an NVIDIA GPU and torch (CUDA build), the frame
    render and the H.264 encode run on the GPU by default
    (--render auto / --encode auto). Without a GPU it falls back to the
    numpy CPU render and libx264.

Structure:
    The song is segmented into musical phases (intro, build, drop,
    sustain, break, outro) from its energy curve. A director (state
    machine) picks a scene for each phase and crossfades between
    scenes at every boundary:

        orbs   - the classic glowing orbs + core + beat rings
        lasers - rotating neon beams + scanline shots on hard punches
        burst  - particles spiral in during a build, then explode
                 at the drop
        grid   - a grid of spinning neon squares that morph between
                 shapes (circle, star, square, diamond, wave)
        stars  - a warp-tunnel starfield that streaks on punch
        poly   - nested rotating neon polygons that kick on the beat

Rendering:
    Every visual is a list of additive "fields" (point, ring, line,
    box). All fields of a frame are batched into one set of kernels on
    the GPU (exp + matmul), tone-mapped with 1 - exp(-x), and a light
    vignette is applied. No shape or graphics libraries are used.

Audio to visuals:
    bass   (20-160 Hz)   -> big, deep elements
    mids   (160-1600 Hz) -> medium elements
    treble (1600-8000 Hz)-> small, bright, fast elements
    punch  (spectral flux, weighted to the lows) -> flashes, kicks,
            explosions, and rings
"""

import argparse
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    sys.exit("Missing dependency. Run:  python -m pip install imageio-ffmpeg numpy")


def ffmpeg_exe():
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_has_nvenc():
    """True if the bundled ffmpeg build contains the h264_nvenc encoder."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"],
                       capture_output=True, text=True)
    return "h264_nvenc" in (p.stdout or "")


def has_audio(path):
    """Return (has_audio, channel_count) parsed from ffprobe-style info."""
    p = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    err = p.stderr or ""
    if "Audio:" not in err:
        return False, 1
    m = re.search(r"Audio:.*?(\d+) ch", err)
    if m:
        nch = int(m.group(1))
    elif "mono" in err:
        nch = 1
    else:
        nch = 2  # stereo is the common case
    return True, nch


def decode_audio(path, sr=44100):
    p = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", path,
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("could not decode audio from: " + path)
    return np.frombuffer(p.stdout, np.float32).copy()


# ------------------------- audio analysis -------------------------

def analyze(samples, sr, fps):
    """One feature value per video frame: bass, mid, treble energy,
    and punch (spectral flux, low-frequency weighted)."""
    hop = sr // fps
    F = 2048
    n = max(1, math.ceil(len(samples) / hop))
    win = np.hanning(F).astype(np.float32)
    hz = np.arange(F // 2 + 1) * (sr / F)
    m_lo = (hz >= 20) & (hz <= 160)
    m_mid = (hz > 160) & (hz <= 1600)
    m_hi = (hz > 1600) & (hz <= 8000)
    m_all = (hz >= 20) & (hz <= 8000)

    mags = np.empty((n, F // 2 + 1), np.float32)
    for i in range(n):
        seg = samples[i * hop: i * hop + F]
        if seg.size < F:
            seg = np.pad(seg, (0, F - seg.size))
        mags[i] = np.abs(np.fft.rfft(seg * win))

    e_lo = np.empty(n, np.float32)
    e_mid = np.empty(n, np.float32)
    e_hi = np.empty(n, np.float32)
    flux = np.empty(n, np.float32)
    flux_lo = np.empty(n, np.float32)
    prev = np.zeros(F // 2 + 1, np.float32)
    for i in range(n):
        m = mags[i]
        e_lo[i] = np.dot(m[m_lo], m[m_lo])
        e_mid[i] = np.dot(m[m_mid], m[m_mid])
        e_hi[i] = np.dot(m[m_hi], m[m_hi])
        d = m[m_all] - prev[m_all]
        np.maximum(d, 0.0, out=d)
        flux[i] = np.dot(d, d)
        d = m[m_lo] - prev[m_lo]
        np.maximum(d, 0.0, out=d)
        flux_lo[i] = np.dot(d, d)
        prev = m

    def norm(x):
        s = float(np.percentile(x, 98))
        return np.clip(x / max(s, 1e-9), 0.0, 3.0)

    return {
        "n": n, "fps": fps,
        "bass": norm(np.sqrt(e_lo)),
        "mid": norm(np.sqrt(e_mid)),
        "treb": norm(np.sqrt(e_hi)),
        "punch": norm((flux + 4.0 * flux_lo).astype(np.float32)),
    }


# ------------------------- phase segmentation -------------------------

def segment_phases(feats, fps):
    """Label every frame with a musical phase.

    Returns a list of [phase, i0, i1] spans covering the whole song.
    Phases: intro, build, drop, sustain, break, outro.

    Method:
      1. Build a fast-attack / slow-release energy envelope, normalized
         with the 5th / 98th percentiles.
      2. Quantize it into quiet / mid / loud states with thresholds
         taken from the song's own 25th / 75th percentiles, with
         hysteresis so the state does not flicker.
      3. Label the state runs:
         - everything before the first loud section -> intro (its last
           stretch is a build)
         - a loud section that rises out of a break (quiet, or a mid
           dip with much lower punch) -> drop
         - loud sections that just continue loud ones -> sustain
         - quiet / low-punch mid dips in the middle -> break
         - a quiet tail at the end -> outro
    """
    n = feats["n"]
    punch = feats["punch"]
    e = (0.5 * feats["bass"] + 0.3 * feats["mid"] + 0.2 * feats["treb"])
    # fast attack / slow release envelope
    E = np.empty(n, np.float32)
    v = 0.0
    for i in range(n):
        ev = float(e[i])
        v += (ev - v) * (0.35 if ev > v else 0.045)
        E[i] = v
    p5 = float(np.percentile(E, 5))
    p98 = float(np.percentile(E, 98))
    Er = np.clip((E - p5) / max(p98 - p5, 1e-6), 0.0, 1.5)

    TQ = float(np.percentile(Er, 25))
    TL = float(np.percentile(Er, 75))
    if TL - TQ < 0.20:
        # little dynamics: one flat state, no phases to detect
        st = np.ones(n, np.int32)
    else:
        st = np.empty(n, np.int32)
        s = 0 if Er[0] < TQ else (2 if Er[0] > TL else 1)
        st[0] = s
        for i in range(1, n):
            x = Er[i]
            if s == 0:
                if x > TL:
                    s = 2
                elif x > TQ + 0.08:
                    s = 1
            elif s == 2:
                if x < TQ:
                    s = 0
                elif x < TL - 0.08:
                    s = 1
            else:
                if x > TL + 0.02:
                    s = 2
                elif x < TQ - 0.02:
                    s = 0
            st[i] = s

    # runs of constant state
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        runs.append([st[i], i, j])
        i = j
    # merge runs shorter than 1.5 s into the previous one
    m = []
    for r in runs:
        if m and r[2] - r[1] < int(1.5 * fps):
            m[-1][2] = r[2]
        else:
            m.append(r)
    runs = m

    loud = [k for k, r in enumerate(runs) if r[0] == 2]
    if not loud:
        return [["sustain", 0, n]]
    first_loud, last_loud = loud[0], loud[-1]
    pm_loud = float(punch[st == 2].mean())

    def mid_is_break(k):
        state, a, b = runs[k]
        if state != 1:
            return False
        return float(punch[a:b].mean()) < 0.6 * max(pm_loud, 1e-6)

    spans = []
    # everything before the first loud section
    pre_end = runs[first_loud][1]
    if pre_end > int(8 * fps):
        bt = pre_end - int(min(4 * fps, pre_end // 2))
        for state, a, b in runs[:first_loud]:
            if b <= bt:
                spans.append(["intro", a, b])
            elif a >= bt:
                spans.append(["build", a, b])
            else:
                spans.append(["intro", a, bt])
                spans.append(["build", bt, b])
    else:
        for state, a, b in runs[:first_loud]:
            spans.append(["intro", a, b])

    # the middle: loud runs and the dips between them
    for k in range(first_loud, last_loud + 1):
        state, a, b = runs[k]
        if state == 2:
            if k > first_loud:
                ps, pa, pb = runs[k - 1]
                prev_break = (ps == 0) or mid_is_break(k - 1)
                spans.append(["drop" if prev_break else "sustain", a, b])
            else:
                spans.append(["drop", a, b])
        elif state == 0:
            spans.append(["break", a, b])
        else:
            spans.append(["break" if mid_is_break(k) else "sustain", a, b])

    # the tail after the last loud section
    for k in range(last_loud + 1, len(runs)):
        state, a, b = runs[k]
        if k == len(runs) - 1 and b == n and state == 0 \
                and b - a > int(4 * fps):
            spans.append(["outro", a, b])
        else:
            spans.append(["break", a, b])

    # merge spans shorter than 2 s into the previous one
    merged = []
    for sp in spans:
        if merged and sp[2] - sp[1] < int(2.0 * fps):
            merged[-1][2] = sp[2]
        else:
            merged.append(list(sp))
    if len(merged) > 1 and merged[0][2] - merged[0][1] < int(2.0 * fps):
        merged[1][1] = merged[0][1]
        del merged[0]
    return merged


def plan_scenes(spans, fps):
    """Turn phase spans into a scene plan.

    - a build followed directly by a drop becomes one "burst" span
      (the scene converges during the build and explodes at the drop);
      the burst covers the build plus 10 s of the drop, the rest of a
      long drop becomes sustain
    - spans longer than 12 s are chunked so scenes vary
    - each phase cycles through its scene list for variety
    - spans shorter than 3 s merge into the previous one
    """
    CHUNK = int(12 * fps)
    plan = []
    i = 0
    while i < len(spans):
        ph, a, b = spans[i]
        if ph == "build" and i + 1 < len(spans) \
                and spans[i + 1][0] == "drop" and spans[i + 1][1] == b:
            drop_end = spans[i + 1][2]
            end = min(drop_end, b + int(10 * fps))
            plan.append({"scene": "burst", "i0": a, "i1": end,
                         "drop": b, "phase": "build+drop"})
            if end < drop_end:
                plan.append({"scene": None, "i0": end, "i1": drop_end,
                             "drop": None, "phase": "sustain"})
            i += 2
            continue
        plan.append({"scene": None, "i0": a, "i1": b, "drop": None, "phase": ph})
        i += 1

    # chunk long spans; the tail of a long drop becomes sustain
    split = []
    for sp in plan:
        if sp["scene"] != "burst" and sp["i1"] - sp["i0"] > CHUNK:
            c = sp["i0"]
            first = True
            while c < sp["i1"]:
                d = min(c + CHUNK, sp["i1"])
                s2 = dict(sp)
                if not first and sp["phase"] == "drop":
                    s2["phase"] = "sustain"
                s2["i0"], s2["i1"] = c, d
                split.append(s2)
                first = False
                c = d
        else:
            split.append(sp)
    plan = split

    cycles = {"drop": ["lasers", "poly"],
              "sustain": ["orbs", "grid"],
              "break": ["grid", "stars"]}
    counters = {"drop": 0, "sustain": 0, "break": 0}
    for sp in plan:
        ph = sp["phase"]
        if sp["scene"] == "burst":
            continue
        if ph == "intro":
            sp["scene"] = "stars"
        elif ph == "outro":
            sp["scene"] = "poly"
        elif ph == "build":
            sp["scene"] = "poly"
        else:
            sp["scene"] = cycles[ph][counters[ph] % len(cycles[ph])]
            counters[ph] += 1

    fin = []
    for sp in plan:
        if fin and sp["i1"] - sp["i0"] < int(3 * fps) and sp["scene"] != "burst":
            fin[-1]["i1"] = sp["i1"]
        else:
            fin.append(sp)
    return fin


# ------------------------- color + field helpers -------------------------

def hsv(h, s, v):
    """hsv -> (r,g,b) in 0..1. Plain math, no color library."""
    h = h % 1.0
    f = h * 6.0
    i = int(f) % 6
    f -= int(f)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return ((v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q))[i]


# field row: (type, a, b, c, d, width/sigma, amp, r, g, b)
FT_POINT, FT_RING, FT_LINE, FT_BOX, FT_BOXO = 0, 1, 2, 3, 4


def point(x, y, sigma, amp, col):
    return (FT_POINT, x, y, 0.0, 0.0, sigma, amp, col[0], col[1], col[2])


def ring(cx, cy, radius, width, amp, col):
    return (FT_RING, cx, cy, radius, 0.0, width, amp, col[0], col[1], col[2])


def line(x1, y1, x2, y2, width, amp, col):
    return (FT_LINE, x1, y1, x2, y2, width, amp, col[0], col[1], col[2])


def box(x, y, rot, half, amp, col):
    """soft filled square, rotated by rot (radians)."""
    return (FT_BOX, x, y, rot, 0.0, half, amp, col[0], col[1], col[2])


def boxo(x, y, rot, half, amp, col):
    """soft square outline, rotated by rot (radians)."""
    return (FT_BOXO, x, y, rot, 0.0, half, amp, col[0], col[1], col[2])


# ------------------------- field kernels (numpy + torch) -------------------------

def _np_shim():
    class NP:
        chunk = 24

        exp = staticmethod(np.exp)
        sqrt = staticmethod(np.sqrt)
        cos = staticmethod(np.cos)
        sin = staticmethod(np.sin)
        abs = staticmethod(np.abs)
        maximum = staticmethod(np.maximum)

        @staticmethod
        def clamp(a, lo, hi):
            return np.clip(a, lo, hi)

        @staticmethod
        def floor(a):
            return np.maximum(a, 0.0)

        @staticmethod
        def clipout(a, lo, hi):
            return np.clip(a, lo, hi, out=a)

        @staticmethod
        def zeros(sh):
            return np.zeros(sh, np.float32)

        @staticmethod
        def asarray(rows):
            return np.asarray(rows, np.float32)

        @staticmethod
        def permute3(F):
            return F.transpose(1, 2, 0)

        @staticmethod
        def arange(n):
            return np.arange(n, dtype=np.float32)

        @staticmethod
        def frame(out):
            return np.ascontiguousarray(out * 255.0).astype(np.uint8).tobytes()

        @staticmethod
        def sync():
            pass

    return NP


def _torch_shim(dev):
    import torch

    class TOR:
        chunk = 64

        exp = staticmethod(torch.exp)
        sqrt = staticmethod(torch.sqrt)
        cos = staticmethod(torch.cos)
        sin = staticmethod(torch.sin)
        abs = staticmethod(torch.abs)
        maximum = staticmethod(torch.maximum)

        @staticmethod
        def clamp(a, lo, hi):
            return torch.clamp(a, lo, hi)

        @staticmethod
        def floor(a):
            return torch.clamp(a, min=0.0)

        @staticmethod
        def clipout(a, lo, hi):
            a.clamp_(lo, hi)
            return a

        @staticmethod
        def zeros(sh):
            return torch.zeros(sh, dtype=torch.float32, device=dev)

        @staticmethod
        def asarray(rows):
            return torch.as_tensor(rows, dtype=torch.float32, device=dev)

        @staticmethod
        def permute3(F):
            return F.permute(1, 2, 0)

        @staticmethod
        def arange(n):
            return torch.arange(n, dtype=torch.float32, device=dev)

        @staticmethod
        def frame(out):
            out.mul_(255.0)
            return out.to(torch.uint8).contiguous().cpu().numpy().tobytes()

        @staticmethod
        def sync():
            if dev.type == "cuda":
                torch.cuda.synchronize()

    return TOR


def field_kernel(S, ft, fx, fy, base, g):
    """Evaluate one chunk (g, 10) of same-type fields -> (k, H, W)."""
    x = g[:, 1]
    y = g[:, 2]
    w = g[:, 5]
    amp = g[:, 6]
    if ft == FT_POINT:
        cs = x * x + y * y
        sg2 = 2.0 * w * w
        d2 = (base[None, :, :]
              - 2.0 * (x[:, None, None] * fx[None, None, :]
                       + y[:, None, None] * fy[None, :, None])
              + cs[:, None, None])
        return S.exp(-d2 / sg2[:, None, None]) * amp[:, None, None]
    if ft == FT_RING:
        r = g[:, 3]
        d2 = (base[None, :, :]
              - 2.0 * (x[:, None, None] * fx[None, None, :]
                       + y[:, None, None] * fy[None, :, None])
              + (x * x + y * y)[:, None, None])
        d = S.sqrt(S.floor(d2))
        ww = 2.0 * w * w
        return S.exp(-((d - r[:, None, None]) ** 2)
                     / ww[:, None, None]) * amp[:, None, None]
    if ft == FT_LINE:
        x2 = g[:, 3]
        y2 = g[:, 4]
        dx = x2 - x
        dy = y2 - y
        l2 = S.floor(dx * dx + dy * dy) + 1e-6
        px = fx[None, None, :]
        py = fy[None, :, None]
        proj = ((px - x[:, None, None]) * dx[:, None, None]
                + (py - y[:, None, None]) * dy[:, None, None]) / l2[:, None, None]
        proj = S.clamp(proj, 0.0, 1.0)
        qx = x[:, None, None] + proj * dx[:, None, None]
        qy = y[:, None, None] + proj * dy[:, None, None]
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        ww = 2.0 * w * w
        return S.exp(-d2 / ww[:, None, None]) * amp[:, None, None]
    # FT_BOX (filled) / FT_BOXO (outline)
    rot = g[:, 3]
    half = w
    c = S.cos(rot)
    sn = S.sin(rot)
    px = fx[None, None, :]
    py = fy[None, :, None]
    ox = px - x[:, None, None]
    oy = py - y[:, None, None]
    lx = c[:, None, None] * ox + sn[:, None, None] * oy
    ly = -sn[:, None, None] * ox + c[:, None, None] * oy
    m = S.maximum(S.abs(lx), S.abs(ly))
    edge = (0.35 * half + 2.0)[:, None, None]
    if ft == FT_BOX:
        q = S.floor(m - half[:, None, None])
    else:
        q = m - half[:, None, None]
    return S.exp(-(q * q) / (2.0 * edge * edge)) * amp[:, None, None]


def fields_to_image(S, fx, fy, base, rows):
    """Sum all field rows into a (H, W, 3) linear-light image."""
    H = base.shape[0]
    W = base.shape[1]
    if not rows:
        return S.zeros((H, W, 3))
    F = S.asarray(rows)
    img3 = S.zeros((H, W, 3))
    for ft in (FT_POINT, FT_RING, FT_LINE, FT_BOX, FT_BOXO):
        G = F[F[:, 0] == ft]
        if G.shape[0] == 0:
            continue
        for c0 in range(0, G.shape[0], S.chunk):
            g = G[c0:c0 + S.chunk]
            Fg = field_kernel(S, ft, fx, fy, base, g)
            img3 = img3 + S.permute3(Fg) @ g[:, 7:10]
    return img3


# ------------------------- scenes -------------------------

class Orb:
    """A glowing orb on a slow Lissajous orbit around the center.
    Punches add acceleration; a spring pulls it back to its orbit."""

    def __init__(self, rng, band, sigma, hue, amp_g, amp_c,
                 oa, ob, wa, wb, phase, punch_k):
        self.band = band
        self.sigma = sigma
        self.hue = hue
        self.amp_g = amp_g
        self.amp_c = amp_c
        self.oa, self.ob = oa, ob
        self.wa, self.wb = wa, wb
        self.phase = phase
        self.punch_k = punch_k
        d = rng.random(2) - 0.5
        self.dir = d / (np.linalg.norm(d) + 1e-9)
        self.off = np.zeros(2, np.float32)
        self.vel = np.zeros(2, np.float32)

    def anchor(self, t, cx, cy):
        return (cx + self.oa * math.cos(self.wa * t + self.phase),
                cy + self.ob * math.sin(self.wb * t + self.phase * 1.7))

    def step(self, punch, rng):
        if punch > 0.35:  # re-roll the kick direction on hard punches
            self.dir += 0.5 * (rng.random(2) - 0.5)
            self.dir /= np.linalg.norm(self.dir) + 1e-9
        self.vel += self.punch_k * punch * self.dir
        self.vel -= self.off * 0.004          # spring back to orbit
        self.vel *= 0.90                      # damping
        self.off += self.vel


def build_orbs(rng, W, H, s):
    # neon palette: magenta/pink bass, cyan mids, yellow-green treble
    bass = [Orb(rng, "bass", (85 + 14 * i) * s, 0.87 + 0.03 * i,
                0.34, 0.34,
                (0.20 + 0.04 * i) * W, (0.14 + 0.03 * i) * H,
                0.18 + 0.05 * i, 0.14 + 0.04 * i,
                float(rng.random() * 6.28), 12.0) for i in range(3)]
    mid = [Orb(rng, "mid", (46 + 10 * i) * s, 0.50 + 0.03 * i,
               0.42, 0.42,
               0.30 * W, 0.22 * H,
               0.45 + 0.15 * i, 0.38 + 0.12 * i,
               float(rng.random() * 6.28), 8.5) for i in range(3)]
    treb = [Orb(rng, "treb", (16 + 5 * i) * s, 0.10 + 0.05 * i,
                0.52, 1.30,
                0.40 * W, 0.32 * H,
                1.0 + 0.30 * i, 0.85 + 0.25 * i,
                float(rng.random() * 6.28), 7.5) for i in range(4)]
    return bass + mid + treb


class Scene:
    """Base scene. tick() returns a list of field rows for one frame."""

    def __init__(self, W, H, s, fps, rng, span):
        self.W, self.H, self.s, self.fps = W, H, s, fps
        self.rng = rng
        self.span = span
        self.cx0, self.cy0 = W / 2.0, H / 2.0
        self.diag = math.hypot(W, H)

    def drift(self, t):
        """slow camera drift, shared by all scenes"""
        return (self.cx0 + 0.015 * self.W * math.sin(t * 0.13),
                self.cy0 + 0.012 * self.H * math.cos(t * 0.11))

    def tick(self, i, t, env, punch, overall):
        raise NotImplementedError


class OrbsScene(Scene):
    """the classic: band orbs + cycling core + expanding beat rings"""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.orbs = build_orbs(rng, W, H, s)
        self.rings = []
        self.last_ring_t = -1.0

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        for orb in self.orbs:
            orb.step(punch, self.rng)
            ax, ay = orb.anchor(t, cx, cy)
            x, y = ax + orb.off[0], ay + orb.off[1]
            e = env[orb.band]
            hue = (orb.hue + 0.02 * t) % 1.0
            if orb.band == "bass":
                sig = orb.sigma * (0.80 + 0.55 * e)
                ag = orb.amp_g * (0.6 + 0.9 * e)
                ac = orb.amp_c * (0.5 + 1.1 * e)
            elif orb.band == "mid":
                sig = orb.sigma * (0.75 + 0.50 * e)
                ag = orb.amp_g * (0.6 + 1.0 * e)
                ac = orb.amp_c * (0.6 + 1.1 * e)
            else:
                sig = orb.sigma * (0.65 + 0.60 * e)
                ag = orb.amp_g * (0.45 + 1.3 * e)
                ac = orb.amp_c * (0.4 + 1.5 * e)
            rows.append(point(x, y, sig * 2.6, ag, hsv(hue, 1.0, 1.0)))
            rows.append(point(x, y, sig, ac, hsv(hue, 0.15, 1.0)))
        core_sig = (26 + 30 * overall + 20 * punch) * self.s
        core_amp = 0.30 + 0.85 * overall + 1.2 * punch
        hue_c = (0.04 * t) % 1.0
        rows.append(point(cx, cy, core_sig * 2.6, core_amp, hsv(hue_c, 1.0, 1.0)))
        rows.append(point(cx, cy, core_sig, core_amp * 1.1, hsv(hue_c, 0.15, 1.0)))
        if punch > 0.40 and t - self.last_ring_t > 0.14:
            self.rings.append([40 * self.s, 10 * self.s, 0.8 + 0.7 * punch,
                               (9.0 + 34.0 * punch) * self.s,
                               (0.10 + 0.05 * t) % 1.0])
            self.last_ring_t = t
        self.rings = [r for r in self.rings
                      if r[2] > 0.02 and r[0] < self.diag]
        for r in self.rings:
            r[0] += r[3]
            r[3] *= 0.97
            r[2] *= 0.935
        self.rings = [r for r in self.rings
                      if r[2] > 0.02 and r[0] < self.diag]
        for r in self.rings:
            rows.append(ring(cx, cy, r[0], r[1], r[2], hsv(r[4], 1.0, 1.0)))
        return rows


class LasersScene(Scene):
    """rotating neon beams from the center + scanline shots on punch"""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.scan = []
        self.last_scan_t = -1.0
        self.bands = ("bass", "mid", "treb")

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        N = 6
        sp = 0.30 + 1.25 * overall
        R = 0.85 * self.diag
        for k in range(N):
            d = 1.0 if k % 2 == 0 else -1.0
            th = d * t * sp * (0.75 + 0.09 * k) + k * math.pi / 3.0
            x2 = cx + R * math.cos(th)
            y2 = cy + R * math.sin(th)
            band = min(float(env[self.bands[k % 3]]), 1.6)
            w = (5.0 + 17.0 * band) * self.s
            hue = (th / (2 * math.pi) + 0.06 * t) % 1.0
            amp = 0.50 + 1.1 * band + 0.7 * punch
            rows.append(line(cx, cy, x2, y2, w, amp, hsv(hue, 1.0, 1.0)))
            rows.append(point(x2, y2, (6.0 + 8.0 * punch) * self.s,
                              amp * 0.8, hsv(hue, 0.12, 1.0)))
        # scanline shots on hard punches
        if punch > 0.45 and t - self.last_scan_t > 0.16:
            self.scan.append({"p": float(self.rng.uniform(-0.38, 0.38)),
                              "h": bool(self.rng.random() < 0.5),
                              "a": 0,
                              "hue": 0.90 if self.rng.random() < 0.5 else 0.52})
            self.last_scan_t = t
        for sc in self.scan:
            sc["a"] += 1
        self.scan = [sc for sc in self.scan if sc["a"] < 7]
        for sc in self.scan:
            fade = 1.0 - sc["a"] / 7.0
            off = sc["p"] * min(self.W, self.H) + sc["a"] * 3.0 * self.s
            w = (4.0 + 12.0 * punch) * self.s
            col = hsv(sc["hue"], 1.0, 1.0)
            if sc["h"]:
                rows.append(line(-20.0, cy + off, self.W + 20.0, cy + off,
                                 w, 1.5 * fade, col))
            else:
                rows.append(line(cx + off, -20.0, cx + off, self.H + 20.0,
                                 w, 1.5 * fade, col))
        hue_c = (0.60 + 0.05 * t) % 1.0
        rows.append(point(cx, cy, (18 + 24 * overall) * self.s,
                          0.5 + 1.4 * punch, hsv(hue_c, 1.0, 1.0)))
        rows.append(point(cx, cy, (8 + 10 * overall) * self.s,
                          0.9 + 2.2 * punch, hsv(hue_c, 0.10, 1.0)))
        return rows


class BurstScene(Scene):
    """build then explode: particles spiral in, detonate at the drop"""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.t0 = span["i0"] / fps
        self.t_exp = span["drop"] / fps
        self.N = 110
        self.R = 0.46 * min(W, H)
        self.th = rng.uniform(0.0, 2 * math.pi, self.N)
        self.r0 = self.R * (0.30 + 0.70 * rng.random(self.N) ** 0.6)
        self.v = self.R * (1.6 + 2.6 * rng.random(self.N))
        self.hue_b = (0.50 + 0.14 * rng.random(self.N)) % 1.0
        self.hue_e = (0.95 + 0.18 * rng.random(self.N)) % 1.0

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        if t < self.t_exp:
            dur = max(self.t_exp - self.t0, 1e-3)
            p = min(max((t - self.t0) / dur, 0.0), 1.0)
            pe = p * p * (3.0 - 2.0 * p)
            r = self.r0 * (1.0 - 0.85 * pe)
            th = self.th + 2.5 * pe
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            amp = 0.30 + 1.05 * pe + 0.25 * overall
            sig = (3.0 + 7.0 * pe) * self.s
            for n in range(self.N):
                rows.append(point(float(x[n]), float(y[n]), sig, amp,
                                  hsv(self.hue_b[n], 1.0, 1.0)))
            rr1 = self.R * (1.0 - (t * 0.35) % 1.0)
            rr2 = self.R * (1.0 - ((t * 0.35) + 0.5) % 1.0)
            rows.append(ring(cx, cy, rr1, 5 * self.s, 0.55, hsv(0.55, 1.0, 1.0)))
            rows.append(ring(cx, cy, rr2, 5 * self.s, 0.40, hsv(0.60, 1.0, 1.0)))
            hue_c = (0.50 + 0.38 * pe) % 1.0
            rows.append(point(cx, cy, (12 + 75 * pe) * self.s,
                              0.4 + 1.9 * pe * pe, hsv(hue_c, 1.0, 1.0)))
            rows.append(point(cx, cy, (6 + 28 * pe) * self.s,
                              0.8 + 2.6 * pe * pe, hsv(hue_c, 0.10, 1.0)))
        else:
            tau = t - self.t_exp
            fade = max(0.0, 1.0 - tau / 0.85)
            r = 0.10 * self.R + self.v * tau
            x = cx + r * np.cos(self.th)
            y = cy + r * np.sin(self.th)
            for n in range(self.N):
                if not (-60 < x[n] < self.W + 60 and -60 < y[n] < self.H + 60):
                    continue
                rows.append(point(float(x[n]), float(y[n]),
                                  (4.0 + 14.0 * tau) * self.s,
                                  0.4 + 1.6 * fade,
                                  hsv(self.hue_e[n], 1.0, 1.0)))
            fl = math.exp(-tau * 7.0)
            if fl > 0.01:
                rows.append(point(cx, cy, (40 + 460 * tau) * self.s, 8.0 * fl,
                                  hsv(0.90, 0.45, 1.0)))
                rows.append(point(cx, cy, (16 + 200 * tau) * self.s, 12.0 * fl,
                                  hsv(0.0, 0.08, 1.0)))
            if tau < 1.3:
                f2 = 1.0 - tau / 1.3
                rows.append(ring(cx, cy, 1500 * self.s * tau, 18 * self.s,
                                 2.6 * f2, hsv(0.93, 1.0, 1.0)))
                rows.append(ring(cx, cy, 1000 * self.s * tau, 13 * self.s,
                                 1.8 * f2, hsv(0.52, 1.0, 1.0)))
                rows.append(ring(cx, cy, 1900 * self.s * tau, 8 * self.s,
                                 1.2 * f2, hsv(0.12, 1.0, 1.0)))
            hue_c = (0.90 + 0.04 * t) % 1.0
            rows.append(point(cx, cy, (16 + 42 * env["bass"]) * self.s,
                              0.35 + 1.1 * env["bass"], hsv(hue_c, 1.0, 1.0)))
            rows.append(point(cx, cy, (8 + 18 * env["bass"]) * self.s,
                              0.6 + 1.4 * env["bass"], hsv(hue_c, 0.10, 1.0)))
        return rows


class GridScene(Scene):
    """a grid of spinning neon squares that morph between shapes"""

    PERIOD = 2.6  # seconds per shape

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.cols, self.nrows = 13, 8
        self.cell = min(W / (self.cols + 1.5), H / (self.nrows + 1.5))

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        idx = int(t / self.PERIOD) % 5
        f = (t / self.PERIOD) % 1.0
        fe = f * f * (3.0 - 2.0 * f)
        pulse = 0.55 + 0.50 * overall + 0.55 * punch
        half_c = (self.cols - 1) / 2.0
        half_r = (self.nrows - 1) / 2.0
        for j in range(self.cols):
            u = (j - half_c) / half_c
            x = cx + (j - half_c) * self.cell
            for k in range(self.nrows):
                v = (k - half_r) / half_r
                y = cy + (k - half_r) * self.cell
                m = self._shape(idx, u, v, t)
                m2 = self._shape((idx + 1) % 5, u, v, t)
                m = min(1.0, max(0.0, m + (m2 - m) * fe))
                amp = 0.07 + 1.7 * m * pulse
                rot = t * (1.1 + 0.5 * ((j + k) % 3)) + j * 0.25 + k * 0.45
                half = self.cell * (0.26 + 0.14 * m) * (1.0 + 0.10 * punch)
                hue = (0.55 + 0.30 * math.sin(2 * math.pi * j / self.cols
                                              - 1.6 * t + 0.9 * k / self.nrows)) % 1.0
                rows.append(box(x, y, rot, half, amp, hsv(hue, 1.0, 1.0)))
        return rows

    @staticmethod
    def _shape(k, u, v, t):
        d = math.hypot(u, v)
        if k == 0:      # ring
            return 1.0 - min(1.0, abs(d - 0.80) / 0.24)
        if k == 1:      # 5-point star
            phi = math.atan2(v, u)
            rs = 0.48 + 0.52 * abs(math.cos(2.5 * phi + 0.4 * t))
            return 1.0 - min(1.0, max(0.0, d - rs) / 0.14)
        if k == 2:      # square
            ds = max(abs(u), abs(v))
            return 1.0 - min(1.0, max(0.0, ds - 0.82) / 0.12)
        if k == 3:      # diamond
            dd = (abs(u) + abs(v)) / 1.30
            return 1.0 - min(1.0, max(0.0, dd - 0.88) / 0.14)
        # travelling wave
        return 0.30 + 0.70 * abs(math.sin(2.4 * math.pi * u * 0.5 + 2.2 * t)) \
            * (0.40 + 0.60 * abs(math.sin(1.8 * math.pi * v - 1.5 * t)))


class StarsScene(Scene):
    """a warp-tunnel starfield that streaks on punch"""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.N = 150
        self.z = rng.uniform(0.07, 1.0, self.N)
        self.th = rng.uniform(0.0, 2 * math.pi, self.N)
        self.sf = rng.uniform(0.6, 1.5, self.N)
        self.hot = rng.random(self.N) < 0.18
        self.hue0 = 0.52 + 0.22 * rng.random(self.N)
        self.R = 0.72 * min(W, H)
        self.limit = self.diag * 0.8

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        dz = 0.0035 + 0.016 * overall + 0.030 * punch
        z = self.z
        for n in range(self.N):
            zn = z[n] - dz * self.sf[n]
            if zn < 0.05:
                zn = 0.9 + 0.1 * self.rng.random()
                self.th[n] = self.rng.random() * 2 * math.pi
            z[n] = zn
            r = self.R * (1.0 - zn) / zn * 0.055
            if r > self.limit:
                continue
            th = self.th[n]
            x = cx + r * math.cos(th)
            y = cy + r * math.sin(th)
            b = 1.0 - zn
            hue = (self.hue0[n] + 0.03 * t) % 1.0
            if self.hot[n]:
                col = hsv(hue, 0.05, 1.0)
                amp = (0.8 + 1.8 * b) * (0.7 + 1.0 * overall)
            else:
                col = hsv(hue, 1.0, 1.0)
                amp = (0.5 + 1.5 * b) * (0.6 + 1.2 * overall)
            if punch > 0.3:
                zp = zn + dz * self.sf[n]
                rp = self.R * (1.0 - zp) / zp * 0.055
                x2 = cx + rp * math.cos(th)
                y2 = cy + rp * math.sin(th)
                rows.append(line(x2, y2, x, y, (1.5 + 2.5 * punch) * self.s,
                                 amp * (0.6 + 0.8 * punch), col))
            else:
                rows.append(point(x, y, (1.5 + 6.0 * b) * self.s, amp, col))
        hue_c = (0.62 + 0.04 * t) % 1.0
        rows.append(point(cx, cy, (18 + 30 * overall) * self.s,
                          0.7 + 1.4 * punch, hsv(hue_c, 1.0, 1.0)))
        rows.append(point(cx, cy, (9 + 14 * overall) * self.s,
                          1.1 + 1.8 * punch, hsv(hue_c, 0.10, 1.0)))
        return rows


class PolyScene(Scene):
    """nested rotating neon polygons that kick on the beat"""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        # (sides, radius fraction of min(W,H), base hue)
        self.pols = ((3, 0.15, 0.92), (4, 0.25, 0.55),
                     (5, 0.36, 0.18), (6, 0.48, 0.33))
        self.theta = [0.0, 0.7, 1.9, 3.1]
        self.kick = 0.0

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        self.kick = max(punch, self.kick * 0.90)
        dt = 1.0 / self.fps
        base_w = 0.35 + 1.15 * overall
        for k, (np_, rf, hue0) in enumerate(self.pols):
            d = 1.0 if k % 2 == 0 else -1.0
            self.theta[k] += d * (base_w * (0.7 + 0.28 * k)
                                  + 1.8 * self.kick) * dt
            R = rf * min(self.W, self.H) * (1.0 + 0.12 * self.kick)
            th0 = self.theta[k]
            hue = (hue0 + 0.02 * t) % 1.0
            lw = (6.0 + 12.0 * min(1.2, 0.4 + overall + self.kick)) * self.s
            amp = 0.90 + 1.35 * min(1.4, 0.55 * overall + self.kick)
            verts = [(cx + R * math.cos(th0 + 2 * math.pi * m / np_),
                      cy + R * math.sin(th0 + 2 * math.pi * m / np_))
                     for m in range(np_)]
            for m in range(np_):
                a = verts[m]
                b = verts[(m + 1) % np_]
                rows.append(line(a[0], a[1], b[0], b[1], lw, amp,
                                 hsv(hue, 1.0, 1.0)))
            for (x, y) in verts:
                rows.append(point(x, y,
                                  (4.0 + 8.0 * min(1.0, 0.4 + overall
                                                    + self.kick))
                                  * self.s,
                                  amp * 1.4, hsv(hue, 0.10, 1.0)))
        hue_c = (0.05 * t) % 1.0
        rows.append(point(cx, cy, (16 + 20 * overall) * self.s,
                          0.75 + 1.2 * punch, hsv(hue_c, 1.0, 1.0)))
        if self.kick > 0.45:
            rows.append(ring(cx, cy,
                             0.15 * min(self.W, self.H)
                             * (1.0 + 0.3 * (1.0 - self.kick)),
                             8 * self.s, (self.kick - 0.45) * 2.4,
                             hsv(0.90, 1.0, 1.0)))
        return rows


SCENES = {
    "orbs": OrbsScene,
    "lasers": LasersScene,
    "burst": BurstScene,
    "grid": GridScene,
    "stars": StarsScene,
    "poly": PolyScene,
}


# ------------------------- director (state machine) -------------------------

def _scale(rows, w):
    if w >= 0.999:
        return rows
    return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6] * w, r[7], r[8], r[9])
            for r in rows]


class Director:
    """Walks the scene plan, keeps prev/current scenes alive, and
    crossfades between them (any scene into any scene) by blending
    their linear-light fields."""

    def __init__(self, plan, W, H, s, fps, rng):
        self.plan = plan
        self.W, self.H, self.s, self.fps = W, H, s, fps
        self.rng = rng
        self.cx0, self.cy0 = W / 2.0, H / 2.0
        self.idx = -1
        self.cur = None
        self.prev = None
        self.switch_i = 0
        self.xf = max(4, int(0.55 * fps))
        self.halo_sig = min(W, H) * 0.55

    def _make(self, sp):
        return SCENES[sp["scene"]](self.W, self.H, self.s, self.fps,
                                   self.rng, sp)

    def fields(self, i, t, env, punch, overall):
        if self.cur is None:
            self.idx = 0
            sp = self.plan[0]
            self.cur = self._make(sp)
            self.prev = None
            self.switch_i = sp["i0"]
        elif i >= self.plan[self.idx]["i1"]:
            self.prev = self.cur
            self.idx = min(self.idx + 1, len(self.plan) - 1)
            sp = self.plan[self.idx]
            self.cur = self._make(sp)
            self.switch_i = sp["i0"]
        w = min(1.0, (i - self.switch_i) / self.xf)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep
        rows = []
        if self.prev is not None and w < 1.0:
            rows += _scale(self.prev.tick(i, t, env, punch, overall), 1.0 - w)
            rows += _scale(self.cur.tick(i, t, env, punch, overall), w)
        else:
            rows += self.cur.tick(i, t, env, punch, overall)
        # global ambient halo, hue drifting over time
        hc = hsv((0.72 + 0.03 * t) % 1.0, 0.95, 1.0)
        a = 0.07 * (0.4 + 0.6 * overall)
        rows.append(point(self.cx0, self.cy0, self.halo_sig, 1.0,
                          (a * hc[0], a * hc[1], a * hc[2])))
        return rows


# ------------------------- render loop -------------------------

def render(feats, plan, W, H, fps, writer, use_gpu):
    if use_gpu:
        S = _torch_shim(_cuda_dev())
    else:
        S = _np_shim()
    n = feats["n"]

    fx = S.arange(W)
    fy = S.arange(H)
    base = fx[None, :] ** 2 + fy[:, None] ** 2
    cx0, cy0 = W / 2.0, H / 2.0
    d2c = (fx[None, :] - cx0) ** 2 + (fy[:, None] - cy0) ** 2
    vr = S.sqrt(d2c) / (0.5 * math.hypot(W, H))
    vig = (1.0 - 0.30 * S.clamp(vr, 0.0, 1.2) ** 2)

    s = min(W, H) / 720.0
    director = Director(plan, W, H, s, fps, np.random.default_rng(20240709))
    e_state = {"bass": 0.0, "mid": 0.0, "treb": 0.0}
    punch_state = 0.0
    exposure = 1.0
    done = n // 12 + 1

    for i in range(n):
        t = i / fps
        for b in ("bass", "mid", "treb"):
            v = float(feats[b][i])
            st = e_state[b]
            e_state[b] = min(st + (v - st) * (0.55 if v > st else 0.10), 1.6)
        punch_state = max(float(feats["punch"][i]), punch_state * 0.80)
        overall = (0.5 * e_state["bass"] + 0.3 * e_state["mid"]
                   + 0.2 * e_state["treb"])

        rows = director.fields(i, t, e_state, punch_state, overall)
        img3 = fields_to_image(S, fx, fy, base, rows)
        out = 1.0 - S.exp(-img3 * exposure)
        out = out * vig[:, :, None]
        S.clipout(out, 0.0, 1.0)
        writer.write(S.frame(out))

        S.sync()
        if i % done == 0:
            print(f"  frame {i + 1}/{n}")
    print(f"  frame {n}/{n}")


def _cuda_dev():
    import torch
    return torch.device("cuda")


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Turn an audio file into a glowing beat-driven MP4.")
    ap.add_argument("audio", help="input audio file "
                    "(default: inputs/<audio> if not found as given)")
    ap.add_argument("-o", "--out", help="output .mp4 path "
                    "(default: outputs/<input>_viz.mp4)")
    ap.add_argument("--size", default="1280x720", help="WxH, default 1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--adelay", type=int, default=40,
                    help="delay the output audio by N ms so the visuals "
                         "lead the sound (default 40, 0 = no delay)")
    ap.add_argument("--render", choices=["auto", "gpu", "cpu"],
                    default="auto",
                    help="render frames on the GPU (torch CUDA) or the CPU "
                         "(numpy). auto: GPU if available (default).")
    ap.add_argument("--encode", choices=["auto", "nvenc", "x264"],
                    default="auto",
                    help="encode H.264 with NVIDIA NVENC (GPU) or libx264 "
                         "(CPU). auto: NVENC if available (default).")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    # convenience: try inputs/ if the file is not found as given
    if not os.path.exists(args.audio) \
            and os.path.exists(os.path.join("inputs", args.audio)):
        args.audio = os.path.join("inputs", args.audio)
    if not os.path.exists(args.audio):
        sys.exit("input file not found: " + args.audio)
    out = args.out or (os.path.join("outputs",
                                    os.path.splitext(
                                        os.path.basename(args.audio))[0]
                                    + "_viz.mp4"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # ---- pick render + encode paths ----
    use_gpu = False
    if args.render in ("auto", "gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
            elif args.render == "gpu":
                print("note: --render gpu requested but no CUDA device is "
                      "available; using the CPU render.")
        except ImportError:
            if args.render == "gpu":
                print("note: --render gpu requested but torch is not "
                      "installed; using the CPU render.")
    use_nvenc = False
    if args.encode in ("auto", "nvenc"):
        if ffmpeg_has_nvenc():
            use_nvenc = True
        elif args.encode == "nvenc":
            print("note: --encode nvenc requested but this ffmpeg build has "
                  "no h264_nvenc; using libx264.")

    print("decoding audio...")
    samples = decode_audio(args.audio)
    print("analyzing beats...")
    feats = analyze(samples, 44100, args.fps)
    spans = segment_phases(feats, args.fps)
    plan = plan_scenes(spans, args.fps)
    secs = feats["n"] / args.fps
    mode = "GPU render + NVENC" if (use_gpu and use_nvenc) else \
           "GPU render" if use_gpu else \
           "NVENC" if use_nvenc else "CPU"
    print(f"rendering {feats['n']} frames ({secs:.1f}s) at {W}x{H}@{args.fps} "
          f"[{mode}]...")
    print("scene plan:")
    for sp in plan:
        print(f"  {sp['i0'] / args.fps:7.1f}-{sp['i1'] / args.fps:7.1f}s  "
              f"{sp['phase']:10s} -> {sp['scene']}")

    a_ok, nch = has_audio(args.audio)
    cmd = [ffmpeg_exe(), "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
           "-r", str(args.fps), "-i", "pipe:0"]
    if a_ok:
        cmd += ["-i", args.audio, "-map", "0:v:0", "-map", "1:a:0"]
    if use_nvenc:
        mbit = max(4, int(round(W * H / 1e6 * 8)))
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5",
                "-rc", "vbr", "-cq", "19", "-b:v", f"{mbit}M",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p"]
    if a_ok:
        if args.adelay > 0:
            cmd += ["-af", "adelay=" + "|".join([str(args.adelay)] * nch)]
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-shortest", out]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        render(feats, plan, W, H, args.fps, proc.stdin, use_gpu)
    except BrokenPipeError:
        proc.wait()
        sys.exit("ffmpeg encoding failed")
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        sys.exit("ffmpeg encoding failed")
    print("wrote " + os.path.abspath(out))


if __name__ == "__main__":
    main()
