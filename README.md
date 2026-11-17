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
```

(Already done in this project: `.venv` exists and both packages are
installed. FFmpeg ships inside imageio-ffmpeg, so no system install
is needed.)

## Usage

```
.venv\Scripts\python music_viz.py path\to\song.mp3
```

Options:

```
-o OUT        output file (default: <input>_viz.mp4)
--size WxH    default 1280x720
--fps N       default 30
```

## Speed

Renders at roughly real time at 720p30 on this machine (15 s of audio
took about 2 minutes). A 3-minute song at 1080p takes a while; for a
quick preview use `--size 1280x720`.

## Files

- `music_viz.py` - the main program
- `make_test_track.py` - generates `test_track.mp3` (15 s of synthetic
  kick + pad + noise for testing)
- `test_track_viz.mp4` - example output
