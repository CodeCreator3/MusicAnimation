# Music Animation

Turns an audio file into an MP4 with a glowing, beat-driven animation.

## How it works

1. FFmpeg decodes the audio to mono float samples.
2. Every 1/30 s hop, the program computes:
   - band energy (bass 20-160 Hz, mid 160-1600 Hz, treble 1600-8000 Hz)
   - punch = spectral flux (onset strength), weighted 5x toward the lows
3. 10 glowing orbs on slow Lissajous orbits react to the audio:
   - bass orbs: big, dim, deep (slow, heavy)
   - mid orbs: medium
   - treble orbs: small, bright, near (fast, light)
   - each punch adds an acceleration impulse; a spring pulls each orb
     back to its orbit
4. Hard punches fire a core flash and an expanding glowing ring.
5. All glows are hand-rolled gaussian fields (no graphics libraries),
   added on a black canvas, tone-mapped with `1 - exp(-x)` for soft
   additive light, plus a vignette.
6. Raw RGB frames are piped to FFmpeg, which encodes H.264 + the
   original audio into an MP4.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install numpy imageio-ffmpeg
# for GPU rendering (optional but recommended):
.venv\Scripts\pip install torch
```

FFmpeg ships inside imageio-ffmpeg, so no system install is needed.
The GPU path needs a torch CUDA build and an NVIDIA GPU; without
either it falls back to the CPU render automatically.

## Usage

```
.venv\Scripts\python music_viz.py path\to\song.mp3
```

Options:

```
-o OUT         output file (default: <input>_viz.mp4)
--size WxH     default 1280x720
--fps N        default 30
--render M     gpu | cpu | auto (default auto, GPU if available)
--encode M     nvenc | x264 | auto (default auto, NVENC if available)
```

GPU render needs `pip install torch` (a CUDA build). NVENC needs an
ffmpeg with h264_nvenc; the ffmpeg shipped in imageio-ffmpeg has it.

## Speed

With an NVIDIA GPU (this machine: RTX 4070) the render runs on CUDA
via torch and the H.264 encode runs on NVENC. Both are automatic:
- the 15 s test track renders in about 5 s (was 125 s on CPU)
- the full 3:24 song renders in about 40 s (was ~13 min on CPU)

Force one path or the other with `--render gpu|cpu|auto` and
`--encode nvenc|x264|auto` (default `auto`: GPU if available).
The GPU and CPU paths produce identical frames (tested to within
1/255 per pixel).

On a machine without a GPU (or without torch installed), the program
falls back to the numpy CPU render and libx264: about 15x slower than
real time at 720p30.

## Files

- `music_viz.py` - the main program
- `make_test_track.py` - generates `test_track.mp3` (15 s of synthetic
  kick + pad + noise for testing)
- `test_track_viz.mp4` - example output
