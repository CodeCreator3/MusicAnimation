# Music Animation

Turns an audio file into an MP4 with a glowing, beat-driven animation.

## How it works

1. FFmpeg decodes the audio to mono float samples.
2. Every 1/30 s hop, the program computes:
   - band energy (bass 20-160 Hz, mid 160-1600 Hz, treble 1600-8000 Hz)
   - punch = spectral flux (onset strength), weighted 5x toward the lows
3. **Phase segmentation**: the energy curve is smoothed into an
   attack/release envelope, quantized into quiet/mid/loud states with
   hysteresis, and each stretch is labeled intro, build, suspense,
   release, sustain, break, or outro from the song's own dynamics
   (adaptive thresholds, so it works on any track). A musical "drop"
   is three phases: the rising **build**, a last ~2 s of **suspense**,
   and the loud **release**.
4. **The director** (a state machine) maps each phase to one of six
   scenes and crossfades between scenes at every boundary, so any
   scene can transition into any other scene smoothly:

   | scene  | look |
   |--------|------|
   | `orbs`   | the classic: 10 neon orbs on Lissajous orbits (magenta bass, cyan mids, yellow-green treble), a spectrum-cycling core, expanding beat rings |
   | `lasers` | six rotating lightsaber beams: a bright near-white core with crisp edges over a soft colored glow, plus full-screen scanline shots on hard punches |
   | `burst`  | 110 particles spiral in as the build rises, then detonate at the drop: white flash, three shockwave rings, debris |
   | `grid`   | drone show: 96 crisp neon squares with spring physics that fly between formations (wave wall, circle, star, square, diamond) every 3 s; punches kick the swarm outward |
   | `stars`  | a warp-tunnel starfield that streaks on punch |
   | `poly`   | nested rotating neon polygons (triangle, square, pentagon, hexagon) that kick on the beat |

5. Every visual is a list of additive "fields" (point, ring, line,
   box) in linear light. All fields of a frame are batched into one
   set of GPU kernels (exp + matmul), tone-mapped with `1 - exp(-x)`
   for soft additive light, plus a vignette. No shape or graphics
   libraries are used.
6. Raw RGB frames are piped to FFmpeg, which encodes H.264 + the
   original audio into an MP4.

Phase -> scene mapping: intro -> stars, build -> poly, suspense ->
burst / poly, release -> lasers / poly (cycled), sustain -> orbs / grid
(cycled), break -> grid / stars (cycled), outro -> poly. A build or
suspense that flows straight into a release becomes one burst whose
explosion lands on the release. Long stretches are chunked at 12 s so
scenes keep varying.

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

## Web UI

```
python webui.py        # then open http://127.0.0.1:8737
```

- choose or upload an input mp3 (uploads land in `inputs/`)
- play the audio and tweak the detected phases by ear: drag the
  boundary handles on the timeline and change phase labels with the
  dropdowns; the scene plan updates live
- pick the output folder, file name, and resolution, then generate
- the result plays on the page with scene markers, a synced playhead,
  and a download link

## Usage (command line)

```
.venv\Scripts\python music_viz.py inputs\song.mp3
```

Input files live in `inputs/`, output videos in `outputs/`:
- if you pass a bare name (`music_viz.py song.mp3`) it is looked up in
  `inputs/` automatically
- the output defaults to `outputs/<input>_viz.mp4`

Options:

```
-o OUT         output file (default: outputs/<input>_viz.mp4)
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
- the full 3:24 song (state machine, 6 scenes) renders in about 2 min
- the same song with a single scene renders in about 40 s
- the 15 s test track renders in about 5-8 s

Force one path or the other with `--render gpu|cpu|auto` and
`--encode nvenc|x264|auto` (default `auto`: GPU if available).
The GPU and CPU paths produce identical frames (tested to within
1-2/255 per pixel).

On a machine without a GPU (or without torch installed), the program
falls back to the numpy CPU render and libx264: about 15x slower than
real time at 720p30.

The scene plan is printed to the console before rendering, so you can
see which phase maps to which scene. To tune the segmentation, edit
`segment_phases` (energy thresholds, hysteresis) and `plan_scenes`
(chunk length, phase->scene cycles).

## Files

- `music_viz.py` - the main program (segmentation, director, 6 scenes,
  GPU + CPU field renderers, and the `render_video()` API used by the CLI
  and the web UI)
- `webui.py` - the local web server (Flask) behind the UI
- `web/index.html` - the web UI (single self-contained page)
- `make_test_track.py` - generates `inputs/test_track.mp3` (15 s of
  synthetic kick + pad + noise for testing)
- `inputs/` - audio files (mp3, wav, m4a, ...)
- `outputs/` - rendered MP4s, e.g. `outputs/test_track_viz.mp4`
  (example) and `outputs/sm_vivid.mp4` (full song with the
  state-machine scenes)
