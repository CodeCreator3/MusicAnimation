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
    pure-numpy CPU render and libx264.

No shape or graphics libraries are used. Every glow, orb, and ring is
computed directly from math functions (gaussian fields, hsv colors,
spring physics) on numpy/torch arrays, then piped raw into FFmpeg.

Audio to visuals:
    bass   (20-160 Hz)   -> big, deep, slow orbs that breathe with the kick
    mids   (160-1600 Hz) -> medium orbs
    treble (1600-8000 Hz)-> small, bright, near orbs
    punch  (spectral flux, weighted to the lows) -> acceleration impulses
            for the orbs, flashes for the core, and expanding rings

Rendering:
    additive gaussian glows on a black canvas, tone-mapped with
    1 - exp(-x) so the glow stays soft and clean, plus a light vignette.
"""

import argparse
import math
import os
import re
import subprocess
import sys
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


# ------------------------- drawing helpers -------------------------

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


def add_glow(buf, gx, gy, gx2, gy2, x, y, sigma,
             amp_glow, amp_core, col_glow, col_core, W, H):
    """Add one gaussian glow (wide, tinted) plus a tight bright core.
    d2 is expanded as x^2+y^2 - 2cx*x - 2cy*y + c^2 to reuse the
    precomputed grids."""
    nx = 0.0 if x < 0 else (W - 1) if x > W - 1 else x
    ny = 0.0 if y < 0 else (H - 1) if y > H - 1 else y
    if (x - nx) ** 2 + (y - ny) ** 2 > (7.0 * sigma) ** 2:
        return  # completely off screen
    d2 = gx2 + gy2 - 2.0 * (x * gx + y * gy) + (x * x + y * y)
    sg = sigma * 2.6
    f = np.exp(-d2 / (2.0 * sg * sg)) * amp_glow
    buf[:, :, 0] += f * col_glow[0]
    buf[:, :, 1] += f * col_glow[1]
    buf[:, :, 2] += f * col_glow[2]
    f = np.exp(-d2 / (2.0 * sigma * sigma)) * amp_core
    buf[:, :, 0] += f * col_core[0]
    buf[:, :, 1] += f * col_core[1]
    buf[:, :, 2] += f * col_core[2]


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


# ------------------------- cpu render loop -------------------------

def render_cpu(feats, W, H, fps, writer):
    s = min(W, H) / 720.0
    rng = np.random.default_rng(20240709)
    n = feats["n"]

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    gx2, gy2 = gx * gx, gy * gy
    cx0, cy0 = W / 2.0, H / 2.0

    # precomputed ambient halo and vignette
    d2c = (gx - cx0) ** 2 + (gy - cy0) ** 2
    halo = np.exp(-d2c / (2.0 * (min(W, H) * 0.55) ** 2))
    vr = np.sqrt(d2c) / (0.5 * math.hypot(W, H))
    vig = (1.0 - 0.30 * np.clip(vr, 0.0, 1.2) ** 2)[:, :, None].astype(np.float32)

    orbs = build_orbs(rng, W, H, s)
    e_state = {"bass": 0.0, "mid": 0.0, "treb": 0.0}
    punch_state = 0.0
    rings = []
    last_ring_t = -1.0

    buf = np.zeros((H, W, 3), np.float32)
    frame = np.empty((H, W, 3), np.uint8)
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

        # slow camera drift
        cx = cx0 + 0.015 * W * math.sin(t * 0.13)
        cy = cy0 + 0.012 * H * math.cos(t * 0.11)

        buf[...] = 0

        # faint ambient halo, hue drifting over time
        hc = hsv((0.72 + 0.03 * t) % 1.0, 0.95, 1.0)
        a = 0.07 * (0.4 + 0.6 * overall)
        buf += halo[:, :, None] * np.array(
            [a * hc[0], a * hc[1], a * hc[2]], np.float32)[None, None, :]

        # the orbs
        for orb in orbs:
            orb.step(punch_state, rng)
            ax, ay = orb.anchor(t, cx, cy)
            x, y = ax + orb.off[0], ay + orb.off[1]
            env = e_state[orb.band]
            hue = (orb.hue + 0.02 * t) % 1.0
            if orb.band == "bass":          # big + deep: wide, heavy
                sig = orb.sigma * (0.80 + 0.55 * env)
                ag, ac = orb.amp_g * (0.6 + 0.9 * env), orb.amp_c * (0.5 + 1.1 * env)
            elif orb.band == "mid":
                sig = orb.sigma * (0.75 + 0.50 * env)
                ag, ac = orb.amp_g * (0.6 + 1.0 * env), orb.amp_c * (0.6 + 1.1 * env)
            else:                            # small + near: tight, bright, fast
                sig = orb.sigma * (0.65 + 0.60 * env)
                ag, ac = orb.amp_g * (0.45 + 1.3 * env), orb.amp_c * (0.4 + 1.5 * env)
            add_glow(buf, gx, gy, gx2, gy2, x, y, sig, ag, ac,
                     hsv(hue, 1.0, 1.0), hsv(hue, 0.15, 1.0), W, H)

        # central core, flashes on punch (cycling through the full spectrum)
        core_sig = (26 + 30 * overall + 20 * punch_state) * s
        core_amp = 0.30 + 0.85 * overall + 1.2 * punch_state
        hue_c = (0.04 * t) % 1.0
        add_glow(buf, gx, gy, gx2, gy2, cx, cy, core_sig, core_amp,
                 core_amp * 1.1, hsv(hue_c, 1.0, 1.0), hsv(hue_c, 0.15, 1.0), W, H)

        # expanding rings on hard punches
        if punch_state > 0.40 and t - last_ring_t > 0.14:
            rings.append({"r": 40 * s, "w": 10 * s,
                          "amp": 0.8 + 0.7 * punch_state,
                          "sp": (9.0 + 34.0 * punch_state) * s,
                          "hue": (0.10 + 0.05 * t) % 1.0})
            last_ring_t = t
        rings = [r for r in rings
                 if r["amp"] > 0.02 and r["r"] < math.hypot(W, H)]
        if rings:
            d2r = gx2 + gy2 - 2.0 * (cx * gx + cy * gy) + cx * cx + cy * cy
            d = np.sqrt(np.maximum(d2r, 0.0))
            for r in rings:
                r["r"] += r["sp"]
                r["sp"] *= 0.97
                r["amp"] *= 0.935
                f = np.exp(-((d - r["r"]) ** 2) / (2.0 * r["w"] ** 2)) * r["amp"]
                col = hsv(r["hue"], 1.0, 1.0)
                buf[:, :, 0] += f * col[0]
                buf[:, :, 1] += f * col[1]
                buf[:, :, 2] += f * col[2]
            rings = [r for r in rings
                     if r["amp"] > 0.02 and r["r"] < math.hypot(W, H)]

        # tone map: soft additive glow, never harsh clipping
        img = 1.0 - np.exp(-buf * exposure)
        img *= vig
        np.clip(img, 0.0, 1.0, out=img)
        frame[:] = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        writer.write(frame.tobytes())

        if i % done == 0:
            print(f"  frame {i + 1}/{n}")
    print(f"  frame {n}/{n}")


# ------------------------- gpu render loop -------------------------
#
# Same math as render_cpu, but all full-frame fields are stacked into
# one (K, H, W) tensor and computed with a batched exp + matmul on the
# GPU. Per frame this is a handful of large GEMM-sized kernels instead
# of ~50 python-looped numpy passes.

# point fields per frame: 20 orb fields (glow+core each) + 2 core + 1 halo = 23
_N_POINT = 23
_PTS = _N_POINT * 7        # per field: x, y, sigma, amp, cr, cg, cb
_RTS = 6                   # per ring:  r, w, amp, cr, cg, cb


def render_gpu(feats, W, H, fps, writer, dev):
    import torch

    s = min(W, H) / 720.0
    rng = np.random.default_rng(20240709)
    n = feats["n"]

    fx = torch.arange(W, dtype=torch.float32, device=dev)
    fy = torch.arange(H, dtype=torch.float32, device=dev)
    fx2 = fx[None, :] ** 2            # (1, W)
    fy2 = fy[:, None] ** 2            # (H, 1)
    base = fx2 + fy2                  # (H, W) x^2+y^2, reused by all fields
    cx0, cy0 = W / 2.0, H / 2.0

    # true squared distance from the canvas center (for the vignette)
    d2c = (fx[None, :] - cx0) ** 2 + (fy[:, None] - cy0) ** 2
    vr = d2c.sqrt() / (0.5 * math.hypot(W, H))
    vig = (1.0 - 0.30 * torch.clamp(vr, 0.0, 1.2) ** 2)

    orbs = build_orbs(rng, W, H, s)
    e_state = {"bass": 0.0, "mid": 0.0, "treb": 0.0}
    punch_state = 0.0
    rings = []
    last_ring_t = -1.0
    diag = math.hypot(W, H)
    halo_sig = min(W, H) * 0.55
    exposure = 1.0
    done = n // 12 + 1
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    for i in range(n):
        t = i / fps
        for b in ("bass", "mid", "treb"):
            v = float(feats[b][i])
            st = e_state[b]
            e_state[b] = min(st + (v - st) * (0.55 if v > st else 0.10), 1.6)
        punch_state = max(float(feats["punch"][i]), punch_state * 0.80)
        overall = (0.5 * e_state["bass"] + 0.3 * e_state["mid"]
                   + 0.2 * e_state["treb"])

        # slow camera drift
        cx = cx0 + 0.015 * W * math.sin(t * 0.13)
        cy = cy0 + 0.012 * H * math.cos(t * 0.11)

        # ---- gather point fields (one upload to the GPU) ----
        # row layout: [x, y, sigma, amp, cr, cg, cb]
        pts = np.empty(_PTS, np.float32)
        off = 0
        for orb in orbs:
            orb.step(punch_state, rng)
            ax, ay = orb.anchor(t, cx, cy)
            x, y = ax + orb.off[0], ay + orb.off[1]
            env = e_state[orb.band]
            hue = (orb.hue + 0.02 * t) % 1.0
            if orb.band == "bass":          # big + deep: wide, heavy
                sig = orb.sigma * (0.80 + 0.55 * env)
                ag, ac = orb.amp_g * (0.6 + 0.9 * env), orb.amp_c * (0.5 + 1.1 * env)
            elif orb.band == "mid":
                sig = orb.sigma * (0.75 + 0.50 * env)
                ag, ac = orb.amp_g * (0.6 + 1.0 * env), orb.amp_c * (0.6 + 1.1 * env)
            else:                            # small + near: tight, bright, fast
                sig = orb.sigma * (0.65 + 0.60 * env)
                ag, ac = orb.amp_g * (0.45 + 1.3 * env), orb.amp_c * (0.4 + 1.5 * env)
            col_g = hsv(hue, 1.0, 1.0)
            col_c = hsv(hue, 0.15, 1.0)
            pts[off:off + 7] = (x, y, sig * 2.6, ag, col_g[0], col_g[1], col_g[2])
            off += 7
            pts[off:off + 7] = (x, y, sig, ac, col_c[0], col_c[1], col_c[2])
            off += 7
        # central core, flashes on punch (cycling through the full spectrum)
        core_sig = (26 + 30 * overall + 20 * punch_state) * s
        core_amp = 0.30 + 0.85 * overall + 1.2 * punch_state
        hue_c = (0.04 * t) % 1.0
        col_g = hsv(hue_c, 1.0, 1.0)
        col_c = hsv(hue_c, 0.15, 1.0)
        pts[off:off + 7] = (cx, cy, core_sig * 2.6, core_amp,
                            col_g[0], col_g[1], col_g[2])
        off += 7
        pts[off:off + 7] = (cx, cy, core_sig, core_amp * 1.1,
                            col_c[0], col_c[1], col_c[2])
        off += 7
        # ambient halo, hue drifting over time (full-screen gaussian
        # centered on the canvas center, as in render_cpu)
        hc = hsv((0.72 + 0.03 * t) % 1.0, 0.95, 1.0)
        a = 0.07 * (0.4 + 0.6 * overall)
        pts[off:off + 7] = (cx0, cy0, halo_sig, 1.0,
                            a * hc[0], a * hc[1], a * hc[2])

        # ---- rings (same order of cull / advance / cull as render_cpu) ----
        if punch_state > 0.40 and t - last_ring_t > 0.14:
            rings.append([40 * s, 10 * s, 0.8 + 0.7 * punch_state,
                          (9.0 + 34.0 * punch_state) * s,
                          (0.10 + 0.05 * t) % 1.0])
            last_ring_t = t
        rings = [r for r in rings if r[2] > 0.02 and r[0] < diag]
        for r in rings:
            r[0] += r[3]
            r[3] *= 0.97
            r[2] *= 0.935
        rings = [r for r in rings if r[2] > 0.02 and r[0] < diag]

        # ---- GPU: batched gaussian point fields ----
        P = torch.from_numpy(pts).to(dev, non_blocking=True).view(_N_POINT, 7)
        X, Y, SIG = P[:, 0], P[:, 1], P[:, 2]
        CS = X * X + Y * Y
        SG2 = 2.0 * SIG * SIG
        # d2[k, h, w] = |p - (X[k], Y[k])|^2, one row per field
        d2 = (base[None, :, :]
              - 2.0 * (X[:, None, None] * fx[None, None, :]
                       + Y[:, None, None] * fy[None, :, None])
              + CS[:, None, None])
        F = torch.exp(-d2 / SG2[:, None, None]) * P[:, 3][:, None, None]

        K = _N_POINT
        if rings:
            data = []
            for r in rings:
                col = hsv(r[4], 1.0, 1.0)
                data += [r[0], r[1], r[2], col[0], col[1], col[2]]
            R = torch.tensor(data, dtype=torch.float32, device=dev)
            R = R.view(len(rings), _RTS)
            d2r = base - 2.0 * (cx * fx[None, :] + cy * fy[:, None]) \
                + cx * cx + cy * cy
            d = d2r.clamp_(min=0.0).sqrt()          # (H, W) distance from center
            FR = torch.exp(-((d[None, :, :] - R[:, 0][:, None, None]) ** 2)
                           / (2.0 * R[:, 1][:, None, None] ** 2)) \
                * R[:, 2][:, None, None]            # (Rn, H, W)
            F = torch.cat([F, FR], dim=0)
            K = _N_POINT + len(rings)

        # ---- combine, tone map, send to encoder ----
        # F: (K, H, W), C: (K, 3) -> sum over fields
        C = torch.empty(K, 3, dtype=torch.float32, device=dev)
        C[:_N_POINT] = P[:, 4:]
        if rings:
            C[_N_POINT:] = R[:, 3:]
        img3 = F.permute(1, 2, 0) @ C                # (H, W, 3)
        out = 1.0 - torch.exp(-img3 * exposure)
        out.mul_(vig[:, :, None])
        out.clamp_(0.0, 1.0)
        out.mul_(255.0)
        frame = out.to(torch.uint8).cpu().numpy()
        writer.write(frame.tobytes())

        sync()
        if i % done == 0:
            print(f"  frame {i + 1}/{n}")
    print(f"  frame {n}/{n}")


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Turn an audio file into a glowing beat-driven MP4.")
    ap.add_argument("audio", help="input audio file (mp3, wav, m4a, ...)")
    ap.add_argument("-o", "--out", help="output .mp4 path "
                    "(default: <input>_viz.mp4)")
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
    out = args.out or (os.path.splitext(args.audio)[0] + "_viz.mp4")

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
    secs = feats["n"] / args.fps
    mode = "GPU render + NVENC" if (use_gpu and use_nvenc) else \
           "GPU render" if use_gpu else \
           "NVENC" if use_nvenc else "CPU"
    print(f"rendering {feats['n']} frames ({secs:.1f}s) at {W}x{H}@{args.fps} "
          f"[{mode}]...")

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
        if use_gpu:
            import torch
            render_gpu(feats, W, H, args.fps, proc.stdin,
                       torch.device("cuda"))
        else:
            render_cpu(feats, W, H, args.fps, proc.stdin)
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
