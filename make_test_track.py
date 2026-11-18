import os

import numpy as np, subprocess, imageio_ffmpeg

sr = 44100
n = sr * 15
t = np.arange(n) / sr

out = np.zeros(n)

# kick: punchy transient on a 120 bpm grid
kicks = np.arange(0, 15, 0.5)
for k in kicks:
    idx = int(k * sr)
    seg = np.zeros(n - idx)
    ts = np.arange(len(seg)) / sr
    seg += 0.9 * np.exp(-ts * 9) * np.sin(2 * np.pi * 55 * ts)
    seg += 0.3 * np.exp(-ts * 30) * np.sin(2 * np.pi * 110 * ts)
    out[idx:idx + len(seg)] += seg

# mid: soft pad swells
out += 0.15 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
out += 0.12 * np.sin(2 * np.pi * 330 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 0.07 * t))

# treble: filtered noise shimmer
rng = np.random.default_rng(0)
noise = rng.standard_normal(n)
f = np.fft.rfft(noise)
f[:100] = 0
noise = np.fft.irfft(f, n)
noise /= noise.max()
out += 0.10 * noise * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))

out = out / (np.abs(out).max() + 1e-9) * 0.8
buf = (out * 32767).astype("<i2").tobytes()

os.makedirs("inputs", exist_ok=True)
p = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                    "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    os.path.join("inputs", "test_track.mp3")],
                   input=buf, capture_output=True)
print(p.returncode, p.stderr.decode(errors="replace")[:300])
print("wrote inputs/test_track.mp3")
