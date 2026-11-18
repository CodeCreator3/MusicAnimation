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
    The song is segmented into musical phases (intro, build, suspense,
    release, sustain, break, outro) from its energy curve. A "drop" is
    three phases: the rising build, a last ~2 s of suspense, and the
    release. A director (state machine) picks a scene for each phase
    and crossfades between scenes at every boundary:

        orbs   - the classic glowing orbs + core + beat rings
        lasers - rotating lightsaber beams (bright near-white core with
                 crisp edges, soft colored glow) + scanline shots
        burst  - the full drop arc: particles spiral in and brighten
                 through the build, collapse to a white-hot point at
                 max tension in the suspense, then explode on the
                 release and settle into a pulsing aftermath
        grid   - drone show: 160 small sharp-edged squares with spring
                 physics; on every major beat the swarm forms a new
                 shape (circle, star, square, diamond, wave wall,
                 flower) and the simulation itself runs faster
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


def extract_beats(feats):
    """Pick the punch peaks that drive the visuals, as editable beats.

    Returns a list of [time_sec, intensity] for each local peak of the
    punch curve (prominence + minimum spacing filtered). The UI shows
    these as draggable markers; their position and intensity can be
    edited by ear.
    """
    n = feats["n"]
    fps = feats["fps"]
    p = feats["punch"].astype(np.float64)
    pm = float(np.percentile(p, 90)) if n else 0.0
    beats = []
    min_gap = int(round(0.28 * fps))
    last = -10 ** 9
    for i in range(1, n - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > 0.22 * pm:
            if i - last >= min_gap:
                beats.append([round(i / fps, 3), round(float(min(1.0, p[i])), 3)])
                last = i
    return beats


def punch_from_beats(beats, n, fps, decay=0.80):
    """Build a per-frame punch curve from a list of [time, intensity]
    beats: a sharp spike at each beat that decays like the audio punch."""
    out = np.zeros(n, np.float32)
    for t, inten in beats:
        i = int(round(t * fps))
        i = max(0, min(n - 1, i))
        out[i] = max(out[i], min(3.0, inten))
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] * decay)
    return out


# ------------------------- phase segmentation -------------------------

def segment_phases(feats, fps):
    """Label every frame with a musical phase.

    Returns a list of [phase, i0, i1] spans covering the whole song.
    Phases: intro, build, suspense, release, sustain, break, outro.

    A "drop" is really three phases in a row:
      - build    - energy rising toward the hit
      - suspense - the last ~2 s of tension right before it lands
      - release  - the loud section itself (the drop hitting)

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
           dip with much lower punch) -> release
         - loud sections that just continue loud ones -> sustain
         - quiet / low-punch mid dips in the middle -> break
         - a quiet tail at the end -> outro
      4. Before each release, carve a "suspense" window (last ~2 s)
         and, out of quiet breaks / intros, a "build" window (~4.5 s),
         so every drop has build -> suspense -> release.
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
                spans.append(["release" if prev_break else "sustain", a, b])
            else:
                spans.append(["release", a, b])
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

    # before each release, carve a "suspense" window (last ~2 s) and,
    # out of quiet breaks / intros, a "build" window (~4.5 s), so
    # every drop has build -> suspense -> release
    final = []
    SUS = int(2.0 * fps)
    BLD = int(4.5 * fps)
    for k, sp in enumerate(merged):
        nxt = merged[k + 1] if k + 1 < len(merged) else None
        if not (nxt and nxt[0] == "release" and sp[0] != "release"
                and sp[2] - sp[1] > SUS + int(0.8 * fps)):
            final.append(sp)
            continue
        susp = sp[2] - SUS
        if sp[0] in ("break", "intro") and susp - sp[1] > BLD + int(1.0 * fps):
            bld = susp - BLD
            final.append([sp[0], sp[1], bld])
            final.append(["build", bld, susp])
        else:
            final.append([sp[0], sp[1], susp])
        final.append(["suspense", susp, sp[2]])

    # merge adjacent spans with the same phase (e.g. two intro runs)
    out = []
    for sp in final:
        if out and out[-1][0] == sp[0]:
            out[-1][2] = sp[2]
        else:
            out.append(sp)
    return out


def plan_scenes(spans, fps):
    """Turn phase spans into a scene plan.

    - a drop arc -- an optional build, an optional suspense, and a
      release, with no other phase between them -- becomes one "burst"
      span. So build+suspense+release, build+release (no suspense), and
      suspense+release all work. The scene brightens and condenses
      through the build, collapses at the peak, and detonates exactly
      when the release hits; the burst covers up to 10 s of the release,
      the rest of a long release becomes sustain. A bare release (no
      build or suspense before it) is not an arc and keeps its own scene.
    - spans longer than 12 s are chunked so scenes vary
    - each phase cycles through its scene list for variety
    - spans shorter than 3 s merge into the previous one
    """
    CHUNK = int(12 * fps)
    ARC_PRE = ("build", "build_v2", "suspense", "suspense_v2")
    ARC_REL = ("release", "release_v2")
    plan = []
    i = 0
    while i < len(spans):
        ph, a, b = spans[i]
        # does a drop arc start here? An arc is an optional build, an
        # optional suspense, and a release, with no other phase between
        # them. It can be build+suspense+release, build+release (no
        # suspense), or suspense+release. It always starts at a build or
        # suspense span -- never at the phase that precedes it (intro,
        # break, ...), so that preceding phase keeps its own scene.
        if ph in ARC_PRE:
            arc = []
            k = i
            while k < len(spans) \
                    and spans[k][0] in (ARC_PRE + ARC_REL):
                arc.append(spans[k])
                if spans[k][0] in ARC_REL:
                    break
                k += 1
            if arc and arc[-1][0] in ARC_REL:
                release = arc[-1]
                # explosion moment = release start; collapse (suspense)
                # window = the last suspense span, or the 2 s before the
                # release when there is no separate suspense span
                drop = None
                susp = None
                for x in arc[:-1]:
                    if x[0] in ("suspense", "suspense_v2"):
                        drop = x[2]
                        susp = x[1]
                if drop is None:
                    drop = release[1]
                if susp is None:
                    susp = max(arc[0][1], drop - int(2.0 * fps))
                end = min(release[2], drop + int(10 * fps))
                # v1 arc -> the exploding burst; v2 arc -> the charging laser
                scene = "charge" if release[0] == "release_v2" else "burst"
                tail = "sustain" if release[0] == "release" else "sustain_v2"
                plan.append({"scene": scene, "i0": arc[0][1], "i1": end,
                             "drop": drop, "susp": susp,
                             "phase": "+".join(x[0] for x in arc)})
                if end < release[2]:
                    plan.append({"scene": None, "i0": end, "i1": release[2],
                                 "drop": None, "phase": tail})
                i = k + 1
                continue
        plan.append({"scene": None, "i0": a, "i1": b, "drop": None, "phase": ph})
        i += 1

    # chunk long spans; the tail of a long release becomes sustain
    split = []
    for sp in plan:
        if sp["scene"] not in ("burst", "charge") \
                and sp["i1"] - sp["i0"] > CHUNK:
            c = sp["i0"]
            first = True
            while c < sp["i1"]:
                d = min(c + CHUNK, sp["i1"])
                s2 = dict(sp)
                if not first and sp["phase"] == "release":
                    s2["phase"] = "sustain"
                elif not first and sp["phase"] == "release_v2":
                    s2["phase"] = "sustain_v2"
                s2["i0"], s2["i1"] = c, d
                split.append(s2)
                first = False
                c = d
        else:
            split.append(sp)
    plan = split

    # v1 and v2 phases each have their own scene pool; v2 uses a
    # completely different set of animations (snake, charge, bombs,
    # fractal, vortex). A fixed scene (string) is always kept; a pool
    # (list) cycles for variety.
    pools = {
        "release": ["lasers", "poly"],
        "release_v2": ["charge", "vortex"],
        "suspense": ["burst", "poly"],
        "suspense_v2": ["charge", "vortex"],
        "sustain": ["orbs", "grid"],
        "sustain_v2": ["snake", "vortex"],
        "break": ["grid", "stars"],
        "break_v2": ["bombs", "snake"],
        "build": ["poly", "bombs"],
        "build_v2": ["fractal", "vortex"],
        "intro": "stars",
        "outro": "poly",
    }
    counters = {ph: 0 for ph in pools}
    for sp in plan:
        ph = sp["phase"]
        if sp["scene"] in ("burst", "charge"):
            continue
        pool = pools.get(ph)
        if pool is None:
            continue
        if isinstance(pool, list):
            sp["scene"] = pool[counters[ph] % len(pool)]
            counters[ph] += 1
        else:
            sp["scene"] = pool

    fin = []
    for sp in plan:
        if fin and sp["i1"] - sp["i0"] < int(3 * fps) \
                and sp["scene"] not in ("burst", "charge"):
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
            hue = (th / (2 * math.pi) + 0.06 * t) % 1.0
            # lightsaber: wide soft colored glow + a thin, bright,
            # near-white core with crisp edges
            glow_w = (5.0 + 11.0 * band) * self.s
            core_w = (0.6 + 1.3 * band) * self.s
            glow_amp = 0.50 + 0.95 * band + 0.55 * punch
            core_amp = 1.7 + 1.7 * band + 1.3 * punch
            rows.append(line(cx, cy, x2, y2, glow_w, glow_amp,
                             hsv(hue, 1.0, 1.0)))
            rows.append(line(cx, cy, x2, y2, core_w, core_amp,
                             hsv(hue, 0.12, 1.0)))
            # bright hot tip
            rows.append(point(x2, y2, (5.0 + 8.0 * punch) * self.s,
                              core_amp * 0.9, hsv(hue, 0.08, 1.0)))
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
            col = hsv(sc["hue"], 1.0, 1.0)
            core = hsv(sc["hue"], 0.10, 1.0)
            if sc["h"]:
                rows.append(line(-20.0, cy + off, self.W + 20.0, cy + off,
                                 (5.0 + 10.0 * punch) * self.s, 1.4 * fade, col))
                rows.append(line(-20.0, cy + off, self.W + 20.0, cy + off,
                                 (0.7 + 1.2 * punch) * self.s, 2.4 * fade, core))
            else:
                rows.append(line(cx + off, -20.0, cx + off, self.H + 20.0,
                                 (5.0 + 10.0 * punch) * self.s, 1.4 * fade, col))
                rows.append(line(cx + off, -20.0, cx + off, self.H + 20.0,
                                 (0.7 + 1.2 * punch) * self.s, 2.4 * fade, core))
        hue_c = (0.60 + 0.05 * t) % 1.0
        rows.append(point(cx, cy, (18 + 24 * overall) * self.s,
                          0.5 + 1.4 * punch, hsv(hue_c, 1.0, 1.0)))
        rows.append(point(cx, cy, (8 + 10 * overall) * self.s,
                          0.9 + 2.2 * punch, hsv(hue_c, 0.10, 1.0)))
        return rows


class BurstScene(Scene):
    """the full drop arc: build -> suspense -> release.

    build:    matter condenses into a singularity -- 180 infall streams
              spiral in as motion streaks (accelerating as they near
              the center), an accretion disk tightens, a razor-thin
              photon ring forms, and the dense core swells; white-hot
              pixels appear at max tension
    suspense: everything collapses into one small white-hot point at
              maximum tension, with a held flicker and a final ring
              closing in on it
    release:  white flash, three shockwave rings, debris; then the
              aftermath settles into a bass-driven pulsing core that
              feeds into the next scene (the burst span covers up to
              10 s of the release)
    """

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.t0 = span["i0"] / fps
        self.t_susp = span.get("susp", span["drop"] / fps - 2.0) / fps
        self.t_exp = span["drop"] / fps
        self.N = 130
        self.R = 0.50 * min(W, H)
        self.th = rng.uniform(0.0, 2 * math.pi, self.N)
        # start far outside the screen and spiral in during the build
        self.r0 = self.R * (1.4 + 1.2 * rng.random(self.N))
        self.swirl = rng.uniform(1.0, 2.2, self.N) * \
            rng.choice(np.array([-1.0, 1.0]), self.N)
        self.v = self.R * (1.8 + 2.8 * rng.random(self.N))
        self.hue_b = (0.50 + 0.16 * rng.random(self.N)) % 1.0
        self.hue_e = (0.95 + 0.18 * rng.random(self.N)) % 1.0
        # accreting infall streams (the matter condensing into the singularity)
        self.N_in = 180
        self.in_r = (self.R * (0.20 + 1.00 * rng.random(self.N_in)))
        self.in_th = rng.uniform(0.0, 2 * math.pi, self.N_in)
        self.in_sp = rng.uniform(0.7, 1.5, self.N_in)
        self.in_sw = rng.uniform(0.6, 1.7, self.N_in) * \
            rng.choice(np.array([-1.0, 1.0]), self.N_in)

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        if t < self.t_exp:
            # ---- build + suspense: matter condenses into a singularity ----
            bt = max(self.t_susp - self.t0, 1e-3)
            et = max(self.t_exp - self.t_susp, 1e-3)
            b = min(max((t - self.t0) / bt, 0.0), 1.0)   # build progress
            c = min(max((t - self.t_susp) / et, 0.0), 1.0)  # collapse
            b2 = b * b * (3.0 - 2.0 * b)
            c2 = c * c
            flick = 1.0 + 0.12 * c2 * math.sin(t * 46.0)
            R = self.R

            # --- continuous infall: streams of matter falling in ---
            # they accelerate (in_r shrinks faster as tension builds),
            # swirl up as they near the center, and respawn at the rim so
            # the accretion never stops. Each is drawn as a motion streak
            # from where it was last frame to where it is now.
            dt = 1.0 / self.fps
            fall = R * (0.10 + 0.55 * b2)            # infall speed (px/s)
            sat_i = 1.0 - 0.75 * c2                   # -> white near collapse
            for n in range(self.N_in):
                # swirl speeds up as the stream gets closer (angular momentum)
                near = 1.0 - min(1.0, self.in_r[n] / R)
                sw = self.in_sw[n] * (0.5 + 3.0 * near) * (0.6 + 1.4 * b2)
                # current position
                r = self.in_r[n]
                th = self.in_th[n]
                x = cx + r * math.cos(th)
                y = cy + r * math.sin(th)
                # previous position (where the streak trails from)
                pr = r + fall * dt * self.in_sp[n]
                pth = th - sw * dt
                px = cx + pr * math.cos(pth)
                py = cy + pr * math.sin(pth)
                # streak length: longer as it falls faster
                L = math.hypot(x - px, y - py)
                if L < 1.5:
                    L = 1.5
                ext = min(L * (1.0 + 1.5 * b2), 26.0 * self.s)
                ux, uy = (x - px) / L, (y - py) / L
                tail_x, tail_y = x - ux * ext, y - uy * ext
                amp = (0.5 + 1.1 * b2 + 1.3 * c2) * (0.55 + 0.45 * near)
                hue = (self.hue_b[n % self.N] + 0.04 * t) % 1.0
                col = hsv(hue, sat_i, 1.0)
                rows.append(line(tail_x, tail_y, x, y, 2.0 * self.s, amp,
                                 col))
                rows.append(point(x, y, 2.5 * self.s, amp * 1.2,
                                  hsv(hue, sat_i, 1.0)))
                # advance the stream; respawn at the rim when it hits center
                self.in_r[n] -= fall * dt * self.in_sp[n]
                self.in_th[n] += sw * dt
                if self.in_r[n] < R * 0.02:
                    self.in_r[n] = R * (0.55 + 0.95 * self.rng.random())
                    self.in_th[n] = self.rng.random() * 2 * math.pi

            # --- the original orbiting particles, condensing in ---
            r = self.r0 * (1.0 - 0.65 * b2) * (1.0 - 0.985 * c2)
            th = self.th + self.swirl * (1.8 * b + 5.5 * c2)
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            amp = (0.35 + 1.0 * b2 + 0.4 * overall) * flick
            sig = (2.0 + 4.0 * b2 + 6.0 * c2) * self.s
            for n in range(self.N):
                hue = (self.hue_b[n] + 0.04 * t) % 1.0
                rows.append(point(float(x[n]), float(y[n]), sig, amp,
                                  hsv(hue, sat_i, 1.0)))

            # --- accretion disk: a bright, tightening ring of infall ---
            disk_r = R * (0.30 + 0.45 * b2) * (1.0 - 0.85 * c2)
            rows.append(ring(cx, cy, disk_r, (3.0 + 6.0 * b2) * self.s,
                             (0.5 + 2.2 * b2 + 2.5 * c2) * flick,
                             hsv((0.55 + 0.25 * b2) % 1.0, 1.0 - 0.6 * c2,
                                 1.0)))
            rows.append(ring(cx, cy, disk_r * 0.7,
                             (2.0 + 4.0 * b2) * self.s,
                             (0.35 + 1.5 * b2 + 2.0 * c2) * flick,
                             hsv((0.60 + 0.20 * b2) % 1.0, 1.0 - 0.6 * c2,
                                 1.0)))

            # --- photon ring: a razor-thin hot ring around the core ---
            ph_r = R * (0.10 + 0.10 * b2) * (1.0 - 0.6 * c2)
            rows.append(ring(cx, cy, ph_r, 1.5 * self.s,
                             (1.0 + 3.0 * b2 + 3.0 * c2) * flick,
                             hsv(0.0, 0.2, 1.0)))

            # --- converging shock rings (matter piling onto the core) ---
            if c < 0.999:
                rr = R * (1.0 - 0.98 * c2)
                rows.append(ring(cx, cy, rr, (4.0 + 5.0 * c2) * self.s,
                                 0.35 + 1.3 * c2, hsv(0.60, 1.0, 1.0)))
            spd = 0.5 + 2.2 * b2
            ph = (t * spd) % 1.0
            rows.append(ring(cx, cy, R * (1.0 - ph), 4 * self.s,
                             (0.25 + 0.9 * b2) * (0.4 + 0.6 * ph),
                             hsv(0.55, 1.0, 1.0)))

            # --- the singularity core: dark, dense, growing, then white-hot ---
            core_sig = (10 + 70 * b2 + 26 * c2) * self.s
            core_amp = (0.4 + 1.8 * b2 + 3.2 * c2) * flick
            hue_c = (0.50 + 0.40 * b2) % 1.0
            rows.append(point(cx, cy, core_sig, core_amp,
                              hsv(hue_c, 1.0 - 0.7 * c2, 1.0)))
            rows.append(point(cx, cy, core_sig * 0.4,
                              core_amp * 1.2, hsv(hue_c, 1.0 - 0.85 * c2,
                                                  1.0)))
        else:
            # ---- release: flash, shockwaves, debris, then aftermath ----
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
                rows.append(point(cx, cy, (40 + 460 * tau) * self.s,
                                  8.0 * fl, hsv(0.90, 0.45, 1.0)))
                rows.append(point(cx, cy, (16 + 200 * tau) * self.s,
                                  12.0 * fl, hsv(0.0, 0.08, 1.0)))
            if tau < 1.3:
                f2 = 1.0 - tau / 1.3
                rows.append(ring(cx, cy, 1500 * self.s * tau, 18 * self.s,
                                 2.6 * f2, hsv(0.93, 1.0, 1.0)))
                rows.append(ring(cx, cy, 1000 * self.s * tau, 13 * self.s,
                                 1.8 * f2, hsv(0.52, 1.0, 1.0)))
                rows.append(ring(cx, cy, 1900 * self.s * tau, 8 * self.s,
                                 1.2 * f2, hsv(0.12, 1.0, 1.0)))
            # aftermath: a bass-driven pulsing core that settles into
            # the sustain (the burst span runs up to 10 s past the hit)
            bass = min(1.6, float(env["bass"]))
            settle = min(1.0, tau / 2.0)
            hue_c = (0.90 + 0.04 * t) % 1.0
            rows.append(point(cx, cy, (16 + 42 * bass) * self.s,
                              (0.35 + 1.1 * bass) * (0.6 + 0.4 * settle),
                              hsv(hue_c, 1.0, 1.0)))
            rows.append(point(cx, cy, (8 + 18 * bass) * self.s,
                              (0.6 + 1.4 * bass) * (0.5 + 0.5 * settle),
                              hsv(hue_c, 0.10, 1.0)))
        return rows


class GridScene(Scene):
    """drone show, v2: a swarm of small sharp-edged squares with
    spring physics. On every major beat the swarm forms a new shape
    (circle, star, square, diamond, wave wall, flower); the physics
    simulation itself runs faster on the beat."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.N = 160
        self.R = 0.42 * min(W, H)
        self.pos = np.empty((self.N, 2), np.float32)
        self.pos[:, 0] = rng.uniform(0.0, 1.0, self.N) * W
        self.pos[:, 1] = rng.uniform(0.0, 1.0, self.N) * H
        self.vel = np.zeros((self.N, 2), np.float32)
        self.rot = rng.uniform(0.0, 6.28, self.N)
        self.spin = rng.uniform(-2.5, 2.5, self.N)
        self.hue_i = np.arange(self.N) / self.N
        self.half = min(W, H) * 0.011  # small squares
        self.sim_t = 0.0
        self.form_idx = -1  # first beat (or first frame) forms the shape
        self.last_form_t = -10.0

    def tick(self, i, t, env, punch, overall):
        dt_frame = 1.0 / self.fps
        cx, cy = self.drift(t)

        if self.form_idx < 0:
            self.form_idx = 0
            self.last_form_t = self.sim_t
        elif punch > 0.50 and self.sim_t - self.last_form_t > 1.3:
            # major beat: form a new shape, with a velocity kick so
            # the transition is dynamic
            self.form_idx = (self.form_idx + 1) % 6
            self.last_form_t = self.sim_t
            self.vel += 0.6 * (self._formation(self.form_idx, cx, cy)
                               - self.pos)

        # the simulation runs faster on the beat
        dt = dt_frame * (1.0 + 3.0 * min(2.0, punch))
        P = self._formation(self.form_idx, cx, cy)
        steps = 2
        h = dt / steps
        for _ in range(steps):
            accel = 22.0 * (P - self.pos) - 7.0 * self.vel
            self.vel += accel * h
            self.pos += self.vel * h
        self.sim_t += dt
        self.rot += self.spin * dt

        half = self.half * (1.0 + 0.6 * min(1.5, punch))
        amp_b = 1.0 + 1.1 * overall + 1.1 * punch
        amp_g = 0.22 + 0.5 * overall + 0.5 * punch
        rows = []
        for n in range(self.N):
            x, y = self.pos[n]
            if -40 < x < self.W + 40 and -40 < y < self.H + 40:
                hue = (self.hue_i[n] + 0.03 * t) % 1.0
                col = hsv(hue, 1.0, 1.0)
                rows.append(box(float(x), float(y), float(self.rot[n]),
                                half, amp_b, col))
                rows.append(point(float(x), float(y), half * 2.2,
                                  amp_g, col))
        return rows

    def _formation(self, k, cx, cy):
        """Target positions (N, 2) for formation k."""
        N, R = self.N, self.R
        a = np.arange(N) * (2 * math.pi / N)
        if k == 0:  # circle
            return np.stack([cx + R * np.cos(a),
                             cy + R * np.sin(a)], 1).astype(np.float32)
        if k == 1:  # 5-point star
            r = R * (0.55 + 0.45 * np.abs(np.cos(2.5 * a)))
            return np.stack([cx + r * np.cos(a),
                             cy + r * np.sin(a)], 1).astype(np.float32)
        if k == 2:  # square (evenly spaced along the perimeter)
            A = 0.72 * R
            side_len = 2 * A
            s = np.arange(N) * 8 * A / N
            side = (s // side_len).astype(int) % 4
            off = s % side_len
            P = np.empty((N, 2), np.float32)
            m = side == 0
            P[m, 0] = cx - A + off[m]; P[m, 1] = cy - A
            m = side == 1
            P[m, 0] = cx + A; P[m, 1] = cy - A + off[m]
            m = side == 2
            P[m, 0] = cx + A - off[m]; P[m, 1] = cy + A
            m = side == 3
            P[m, 0] = cx - A; P[m, 1] = cy + A - off[m]
            return P
        if k == 3:  # diamond (perimeter of a 45-deg square)
            D = 0.85 * R
            sl = D * math.sqrt(2)
            s = np.arange(N) * 4 * D * math.sqrt(2) / N
            side = (s // sl).astype(int) % 4
            off = (s % sl) / math.sqrt(2)
            P = np.empty((N, 2), np.float32)
            m = side == 0
            P[m, 0] = cx + off[m]; P[m, 1] = cy - D + off[m]
            m = side == 1
            P[m, 0] = cx + D - off[m]; P[m, 1] = cy + off[m]
            m = side == 2
            P[m, 0] = cx - off[m]; P[m, 1] = cy + D - off[m]
            m = side == 3
            P[m, 0] = cx - D + off[m]; P[m, 1] = cy - off[m]
            return P
        if k == 4:  # wave wall: 16 x 10 grid, ripple frozen in place
            cols, rowsn = 16, 10
            P = np.empty((N, 2), np.float32)
            for j in range(N):
                c_, m_ = j % cols, j // cols
                P[j, 0] = cx + (c_ - (cols - 1) / 2) * 2.2 * R / (cols - 1)
                P[j, 1] = (cy + (m_ - (rowsn - 1) / 2) * 1.0 * R / (rowsn - 1)
                           + 0.14 * R * math.sin(
                               (c_ / (cols - 1) - 0.5) * 4.2 * math.pi))
            return P
        fr = np.arange(N) / N
        r = R * (0.15 + 0.85 * np.sqrt(fr))
        th = np.arange(N) * 2.399963
        return np.stack([cx + r * np.cos(th),
                         cy + r * np.sin(th)], 1).astype(np.float32)


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
        self.t0 = span["i0"] / fps

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        local = max(0.0, t - self.t0)
        dz = (0.0035 + 0.016 * overall + 0.030 * punch) \
            * (1.0 + 0.8 * min(1.0, local / 12.0))  # zoom speeds up
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
            # differential rotation: angular velocity rises with radius,
            # so the outer stars sweep around faster (a vortex)
            th = self.th[n] + local * 0.55 * (0.20 + 1.6 * (r / max(self.R, 1.0)))
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


class SnakeScene(Scene):
    """lanes of sharp squares that light up in sequence and travel across
    the screen, trailing off behind the head (a moving snake)."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.lanes = []
        nl = 5
        for k in range(nl):
            ncells = 22
            spacing = W / (ncells - 1)
            self.lanes.append({
                "y": (0.16 + 0.17 * k) * H, "ncells": ncells,
                "spacing": spacing,
                "speed": (4.0 + 3.0 * float(rng.random()))
                          * (1.0 if k % 2 == 0 else -1.0),
                "off": float(rng.random() * ncells),
                "hue": (0.52 + 0.09 * k) % 1.0, "trail": 7,
                "size": 0.30 * spacing * (1.0 - 0.04 * k),
            })

    def tick(self, i, t, env, punch, overall):
        rows = []
        speed_mul = (0.6 + 0.6 * overall) * (1.0 + 0.8 * punch)
        for ln in self.lanes:
            n = ln["ncells"]
            head = (ln["off"] + t * ln["speed"] * speed_mul) % n
            for j in range(ln["trail"]):
                idx = (int(head) - j) % n
                x = idx * ln["spacing"]
                fade = (1.0 - j / ln["trail"]) ** 1.6
                if j == 0:  # the head: hot, near-white
                    col = hsv(ln["hue"], 0.12, 1.0)
                    amp = (1.5 + 1.3 * punch) * (0.7 + 0.5 * overall)
                    half = ln["size"] * (1.0 + 0.4 * punch)
                else:
                    col = hsv(ln["hue"], 1.0, 1.0)
                    amp = (0.3 + 1.2 * overall + 0.9 * punch) * fade
                    half = ln["size"] * (0.9 + 0.2 * fade)
                rows.append(box(x, ln["y"], 0.0, half, amp, col))
            if punch > 0.5:
                hx = (int(head)) * ln["spacing"]
                rows.append(ring(hx, ln["y"], 14 * punch * self.s, 6 * self.s,
                                 0.8 * punch, hsv(ln["hue"], 0.1, 1.0)))
        return rows


class ChargeScene(Scene):
    """the v2 drop: a laser that charges up through the build/suspense and
    fires at the release. The beam focuses to white-hot, then erupts into a
    full-screen fire with traveling pulses and shockwaves."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        drop = span.get("drop") or span["i1"]
        susp = span.get("susp") or (drop - int(2.0 * fps))
        self.t0 = span["i0"] / fps
        self.t_susp = susp / fps
        self.t_exp = drop / fps
        self.angle = float(rng.uniform(0.0, math.pi))
        self.hue = float(rng.uniform(0.52, 0.62))
        self.R = 0.5 * min(W, H)
        self.last_fire = 0.0

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        R = 0.95 * self.diag
        ang = self.angle + 0.08 * math.sin(t * 0.5)
        dx, dy = math.cos(ang), math.sin(ang)
        x1, y1 = cx - R * dx, cy - R * dy
        x2, y2 = cx + R * dx, cy + R * dy
        dur = max(self.t_exp - self.t0, 1e-3)
        if t < self.t_exp:
            # ---- charging: the beam focuses, energy tightens in ----
            Q = min(max((t - self.t0) / dur, 0.0), 1.0)
            focus = Q * Q * (3.0 - 2.0 * Q)
            jitter = (1.0 - focus) * 4.0 * self.s * math.sin(t * 42.0)
            gx, gy = x2 - jitter * dy, y2 + jitter * dx
            rows.append(line(x1, y1, gx, gy, (26 - 20 * focus) * self.s,
                             0.30 + 1.4 * Q, hsv(self.hue, 1.0, 1.0)))
            rows.append(line(x1, y1, gx, gy, 2.0 * self.s, 0.6 * Q,
                             hsv(self.hue, 0.2, 1.0)))
            # a ring of energy focusing toward the beam
            Np = 44
            for n in range(Np):
                a = n * (2 * math.pi / Np) + t * 0.7
                rr = self.R * (1.0 - 0.9 * focus) * (0.5 + 0.5 * ((n * 0.61) % 1.0))
                px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
                rows.append(point(px, py, (2.0 + 3.0 * focus) * self.s,
                                  0.4 + 1.3 * focus, hsv(self.hue,
                                                         1.0 - 0.7 * focus, 1.0)))
            # the capacitor: a swelling white-hot node at the center
            rows.append(point(cx, cy, (8 + 26 * focus) * self.s,
                              0.4 + 1.8 * focus * focus, hsv(self.hue, 0.1, 1.0)))
        else:
            # ---- fired: full beam, white-hot, pulses and shockwaves ----
            tau = t - self.t_exp
            flash = math.exp(-tau * 7.0)
            bass = min(1.6, float(env["bass"]))
            rows.append(line(x1, y1, x2, y2, (8 + 50 * flash) * self.s,
                             0.8 + 4.0 * flash + 0.7 * bass,
                             hsv(self.hue, 1.0, 1.0)))
            rows.append(line(x1, y1, x2, y2, (1.5 + 8 * flash) * self.s,
                             1.5 + 6.0 * flash + 1.1 * bass,
                             hsv(self.hue, 0.1, 1.0)))
            # traveling pulses along the beam
            for k in range(2):
                pp = (tau * (1.3 + 0.5 * k)) % 1.0
                rows.append(point(x1 + (x2 - x1) * pp, y1 + (y2 - y1) * pp,
                                  (6 + 8 * flash) * self.s, 1.0 + 3.0 * flash,
                                  hsv(self.hue, 0.1, 1.0)))
            # the moment of fire
            if flash > 0.01:
                rows.append(point(cx, cy, (30 + 300 * tau) * self.s, 6.0 * flash,
                                  hsv(0.9, 0.4, 1.0)))
                rows.append(point(cx, cy, (15 + 150 * tau) * self.s, 9.0 * flash,
                                  hsv(0.0, 0.05, 1.0)))
            if tau < 1.3:
                f2 = 1.0 - tau / 1.3
                rows.append(ring(cx, cy, 1400 * self.s * tau, 16 * self.s,
                                 2.5 * f2, hsv(0.95, 1.0, 1.0)))
                rows.append(ring(cx, cy, 900 * self.s * tau, 10 * self.s,
                                 1.6 * f2, hsv(self.hue, 1.0, 1.0)))
            # re-fires on each hard punch
            if punch > 0.4 and tau - self.last_fire > 0.18:
                self.last_fire = tau
                rows.append(ring(cx, cy, 260 * punch * self.s, 12 * self.s,
                                 1.6 * punch, hsv(0.0, 0.1, 1.0)))
                rows.append(point(cx, cy, (20 + 120 * punch) * self.s,
                                  2.5 * punch, hsv(self.hue, 0.1, 1.0)))
        return rows


class BombsScene(Scene):
    """little bombs are thrown across the screen and burst into spiral
    explosions of light."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.bombs = []
        self.expl = []
        self.last_launch = -10.0

    def _launch(self, t):
        W, H = self.W, self.H
        r = self.rng
        side = int(r.integers(0, 4))
        if side == 0: x0, y0 = float(r.random()) * W, -30.0
        elif side == 1: x0, y0 = W + 30.0, float(r.random()) * H
        elif side == 2: x0, y0 = float(r.random()) * W, H + 30.0
        else: x0, y0 = -30.0, float(r.random()) * H
        self.bombs.append({
            "x0": x0, "y0": y0,
            "x1": float(r.uniform(0.18, 0.82)) * W,
            "y1": float(r.uniform(0.18, 0.82)) * H,
            "t0": t, "dur": float(r.uniform(0.5, 1.1)),
            "hue": float(r.random())})

    def tick(self, i, t, env, punch, overall):
        rows = []
        if (punch > 0.45 and t - self.last_launch > 0.12) or \
                (len(self.bombs) < 2 and t - self.last_launch > 0.5):
            self._launch(t)
            self.last_launch = t
        alive = []
        for b in self.bombs:
            p = (t - b["t0"]) / b["dur"]
            if p >= 1.0:
                self.expl.append({"cx": b["x1"], "cy": b["y1"], "t0": t,
                                  "hue": b["hue"],
                                  "spin": float(self.rng.uniform(3.0, 7.0))
                                           * (1.0 if self.rng.random() < 0.5 else -1.0),
                                  "R": min(self.W, self.H)
                                       * float(self.rng.uniform(0.18, 0.4))})
                continue
            x = b["x0"] + (b["x1"] - b["x0"]) * p
            y = b["y0"] + (b["y1"] - b["y0"]) * p - 120 * math.sin(p * math.pi)
            amp = 0.6 + 0.5 * overall
            rows.append(point(x, y, 4 * self.s, amp, hsv(b["hue"], 0.2, 1.0)))
            rows.append(point(x, y, 10 * self.s, 0.4 * amp,
                              hsv(b["hue"], 1.0, 1.0)))
            rows.append(point(x, y - 4 * self.s, 3 * self.s, 0.8 * amp,
                              hsv(0.08, 0.4, 1.0)))
            alive.append(b)
        self.bombs = alive
        life = 0.9
        alive_e = []
        for e in self.expl:
            age = t - e["t0"]
            if age > life:
                continue
            f = 1.0 - age / life
            Rexp = e["R"] * (0.3 + 0.7 * (age / life))
            arms, per = 3, 10
            for a in range(arms):
                for j in range(per):
                    rr = Rexp * ((j + 1) / per) ** 1.4
                    ang = a * (2 * math.pi / arms) + (j / per) * 2.2 \
                          + age * e["spin"]
                    x = e["cx"] + rr * math.cos(ang)
                    y = e["cy"] + rr * math.sin(ang)
                    if -20 < x < self.W + 20 and -20 < y < self.H + 20:
                        hue = (e["hue"] + j * 0.01 + age * 0.1) % 1.0
                        rows.append(point(x, y, (2.0 + 4.0 * f) * self.s,
                                          0.3 + 1.4 * f,
                                          hsv(hue, 1.0 - 0.6 * f, 1.0)))
            if age < 0.25:
                cf = 1.0 - age / 0.25
                rows.append(point(e["cx"], e["cy"],
                                  (10 + 40 * (1 - cf)) * self.s, 3.0 * cf,
                                  hsv(0.0, 0.05, 1.0)))
            alive_e.append(e)
        self.expl = alive_e
        return rows


class FractalScene(Scene):
    """a recursive fractal tree that grows out from its base, with smaller
    satellite fractals growing around the edges."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.t0 = span["i0"] / fps
        self.sway = float(rng.uniform(0.0, 6.28))

    def _tree(self, rows, x, y, angle, length, depth, g, hue, spread, ratio,
              maxd):
        if depth > maxd or length < 5.0 or g < depth * 0.10:
            return
        ex = x + length * math.cos(angle)
        ey = y + length * math.sin(angle)
        lw = max(1.5, length * 0.10)
        lvl = min(1.0, depth / 6.0)
        rows.append(line(x, y, ex, ey, lw, 0.4 + 0.7 * (1.0 - lvl),
                         hsv(hue, 1.0, 1.0)))
        rows.append(line(x, y, ex, ey, lw * 0.35, 0.4 + 0.6 * (1.0 - lvl),
                         hsv(hue, 0.15, 1.0)))
        nl = length * ratio
        self._tree(rows, ex, ey, angle - spread, nl, depth + 1, g,
                   (hue + 0.04) % 1.0, spread, ratio, maxd)
        self._tree(rows, ex, ey, angle + spread, nl, depth + 1, g,
                   (hue + 0.09) % 1.0, spread, ratio, maxd)

    def tick(self, i, t, env, punch, overall):
        rows = []
        W, H = self.W, self.H
        local = max(0.0, t - self.t0)
        grow = min(1.0, 0.35 + 0.65 * (local / 4.0))
        g = min(1.0, max(0.2,
                         grow * (0.85 + 0.12 * math.sin(t * 0.7))
                         + 0.15 * punch))
        sway = self.sway + 0.15 * math.sin(t * 0.6)
        hue0 = (0.30 + 0.04 * local) % 1.0
        # main tree growing up from the bottom
        self._tree(rows, W / 2, H * 0.92, -math.pi / 2 + sway * 0.1,
                   0.28 * H, 0, g, hue0, 0.45, 0.72, 6)
        # satellites around the edges
        for (sx, sy, ang, hue) in (
                (0.13 * W, 0.30 * H, 0.1, (hue0 + 0.30) % 1.0),
                (0.87 * W, 0.30 * H, math.pi - 0.1, (hue0 + 0.55) % 1.0),
                (0.50 * W, 0.16 * H, math.pi / 2, (hue0 + 0.80) % 1.0)):
            self._tree(rows, sx, sy, ang + sway * 0.2, 0.15 * H, 0,
                       min(1.0, g * 1.15), hue, 0.55, 0.70, 4)
        rows.append(point(W / 2, H * 0.92, (10 + 10 * overall) * self.s,
                          0.5 + 0.6 * punch, hsv(hue0, 0.2, 1.0)))
        return rows


class VortexScene(Scene):
    """a spinning galaxy vortex: dense spiral arms of points rotate and
    tighten on the beat around a hot core."""

    def __init__(self, W, H, s, fps, rng, span):
        super().__init__(W, H, s, fps, rng, span)
        self.N = 220
        self.arms = 3
        self.arm = np.floor(rng.random(self.N) * self.arms).astype(int)
        self.dist = rng.power(0.6, self.N)
        self.tw = rng.normal(0.0, 0.15, self.N)
        self.ph = rng.random(self.N)
        self.R = 0.46 * min(W, H)
        self.hue0 = float(rng.uniform(0.5, 0.7))

    def tick(self, i, t, env, punch, overall):
        rows = []
        cx, cy = self.drift(t)
        rot = t * (0.5 + 1.6 * overall + 2.0 * punch)
        tighten = 1.0 - 0.35 * min(1.0, overall + 0.5 * punch)
        R = self.R * tighten
        for n in range(self.N):
            d = self.dist[n]
            r = R * d
            th = (self.arm[n] * (2 * math.pi / self.arms)
                  + d * 3.2 + self.tw[n] + rot * (1.0 - 0.4 * d))
            x = cx + r * math.cos(th)
            y = cy + r * math.sin(th)
            inner = 1.0 - d
            hue = (self.hue0 + 0.25 * d + 0.02 * t) % 1.0
            amp = (0.25 + 1.1 * inner * inner) * (0.6 + 0.9 * overall
                                                  + 0.6 * punch)
            sz = (1.6 + 3.0 * inner) * self.s
            col = hsv(hue, 0.05, 1.0) if self.ph[n] < 0.12 \
                else hsv(hue, 1.0, 1.0)
            rows.append(point(x, y, sz, amp, col))
        rows.append(point(cx, cy, (18 + 22 * overall) * self.s,
                          0.6 + 1.4 * punch, hsv(self.hue0, 1.0, 1.0)))
        rows.append(point(cx, cy, (9 + 10 * overall) * self.s,
                          1.0 + 2.0 * punch, hsv(self.hue0, 0.1, 1.0)))
        if punch > 0.45:
            rows.append(ring(cx, cy, R * (0.4 + 0.6 * punch), 8 * self.s,
                             1.2 * punch, hsv((self.hue0 + 0.3) % 1.0,
                                               1.0, 1.0)))
        return rows


SCENES = {
    "orbs": OrbsScene,
    "lasers": LasersScene,
    "burst": BurstScene,
    "grid": GridScene,
    "stars": StarsScene,
    "poly": PolyScene,
    "snake": SnakeScene,
    "charge": ChargeScene,
    "bombs": BombsScene,
    "fractal": FractalScene,
    "vortex": VortexScene,
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

def render(feats, plan, W, H, fps, writer, use_gpu, progress_cb=None,
           beats=None):
    if use_gpu:
        S = _torch_shim(_cuda_dev())
    else:
        S = _np_shim()
    n = feats["n"]
    # punch curve: from user-edited beats if given, else from the audio
    if beats:
        punch_curve = punch_from_beats(beats, n, fps)
    else:
        punch_curve = feats["punch"]

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
        punch_state = max(float(punch_curve[i]), punch_state * 0.80)
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
            if progress_cb:
                progress_cb(i, n)
    print(f"  frame {n}/{n}")
    if progress_cb:
        progress_cb(n, n)


def _cuda_dev():
    import torch
    return torch.device("cuda")


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def render_video(audio, out, W=1280, H=720, fps=30, adelay=40,
                 use_gpu=False, use_nvenc=False, phase_spans=None,
                 plan=None, progress_cb=None, beats=None):
    """Decode, analyze, segment, and render a video file.

    phase_spans: optional user-edited list of [phase, i0, i1] (frames).
    plan:        optional pre-built scene plan (list of dicts); if given,
                 phase_spans is ignored and no segmentation runs.
    beats:       optional list of [time_sec, intensity]; if given, the
                 punch curve is built from these instead of the audio.
    Returns (plan, spans). Raises RuntimeError if ffmpeg fails.
    """
    print("decoding audio...")
    samples = decode_audio(audio)
    print("analyzing beats...")
    feats = analyze(samples, 44100, fps)
    if plan is None:
        if phase_spans is None:
            spans = segment_phases(feats, fps)
        else:
            spans = [list(sp) for sp in phase_spans]
        plan = plan_scenes(spans, fps)
    else:
        plan = [dict(sp) for sp in plan]
        for sp in plan:
            sp["i0"] = int(sp["i0"])
            sp["i1"] = int(sp["i1"])
        spans = [[sp["phase"], sp["i0"], sp["i1"]] for sp in plan]
    secs = feats["n"] / fps
    mode = "GPU render + NVENC" if (use_gpu and use_nvenc) else \
           "GPU render" if use_gpu else \
           "NVENC" if use_nvenc else "CPU"
    print(f"rendering {feats['n']} frames ({secs:.1f}s) at {W}x{H}@{fps} "
          f"[{mode}]...")
    print("scene plan:")
    for sp in plan:
        print(f"  {sp['i0'] / fps:7.1f}-{sp['i1'] / fps:7.1f}s  "
              f"{sp['phase']:10s} -> {sp['scene']}")

    a_ok, nch = has_audio(audio)
    cmd = [ffmpeg_exe(), "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
           "-r", str(fps), "-i", "pipe:0"]
    if a_ok:
        cmd += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0"]
    if use_nvenc:
        mbit = max(4, int(round(W * H / 1e6 * 8)))
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5",
                "-rc", "vbr", "-cq", "19", "-b:v", f"{mbit}M",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p"]
    if a_ok:
        if adelay > 0:
            cmd += ["-af", "adelay=" + "|".join([str(adelay)] * nch)]
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-shortest", out]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        render(feats, plan, W, H, fps, proc.stdin, use_gpu, progress_cb,
               beats=beats)
    except BrokenPipeError:
        proc.wait()
        raise RuntimeError("ffmpeg encoding failed")
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg encoding failed")
    return plan, spans


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
        if cuda_available():
            use_gpu = True
        elif args.render == "gpu":
            print("note: --render gpu requested but no CUDA device is "
                  "available; using the CPU render.")
    use_nvenc = False
    if args.encode in ("auto", "nvenc"):
        if ffmpeg_has_nvenc():
            use_nvenc = True
        elif args.encode == "nvenc":
            print("note: --encode nvenc requested but this ffmpeg build has "
                  "no h264_nvenc; using libx264.")

    try:
        plan, spans = render_video(
            args.audio, out, W=W, H=H, fps=args.fps, adelay=args.adelay,
            use_gpu=use_gpu, use_nvenc=use_nvenc)
    except RuntimeError as e:
        sys.exit(str(e))

    print("wrote " + os.path.abspath(out))


if __name__ == "__main__":
    main()
