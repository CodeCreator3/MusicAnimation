#!/usr/bin/env python3
"""Local web UI for music_viz.py.

Run:  python webui.py      (then open http://127.0.0.1:8737)

The page lets you:
  - choose an input mp3 (uploaded into inputs/)
  - choose an output folder and an output file name
  - preview the auto-detected phases and scenes as a timeline, with the
    audio playing so you can tweak boundaries and phase labels by ear
  - render in the background with a progress bar, then play the result
    with scene markers on a synced timeline
"""

import json
import os
import re
import threading
import time
import traceback

from flask import (Flask, Response, jsonify, request, send_file)

import music_viz as M

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(ROOT, "inputs")
WEB = os.path.join(ROOT, "web")
os.makedirs(INPUTS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

# initialize the CUDA context on the main thread so render worker
# threads can use the GPU
CUDA = M.cuda_available()
if CUDA:
    import torch
    torch.zeros(1, device="cuda")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()


def _safe_name(name):
    return os.path.basename(str(name).replace("\\", "/"))


# ------------------------- media serving (Range) -------------------------

def send_media(path, ctype):
    """Serve a media file with HTTP Range support so the browser can
    seek in the audio preview and the result video."""
    if not path or not os.path.exists(path):
        return jsonify(error="file not found: " + str(path)), 404
    size = os.path.getsize(path)
    rng = request.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start >= size:
                return Response(status=416,
                                headers={"Content-Range":
                                         "bytes */%d" % size})
            length = end - start + 1

            def gen():
                with open(path, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(256 * 1024, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        yield chunk

            return Response(gen(), 206, headers={
                "Content-Type": ctype, "Accept-Ranges": "bytes",
                "Content-Range": "bytes %d-%d/%d" % (start, end, size),
                "Content-Length": str(length)})
    return send_file(path, mimetype=ctype, conditional=True)


# ------------------------- endpoints -------------------------

@app.get("/")
def index():
    return send_file(os.path.join(WEB, "index.html"))


@app.get("/api/info")
def api_info():
    return jsonify({
        "root": ROOT,
        "cuda": CUDA,
        "nvenc": M.ffmpeg_has_nvenc(),
        "inputs": sorted(os.listdir(INPUTS)),
    })


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file uploaded"), 400
    name = _safe_name(f.filename)
    if not name:
        return jsonify(error="bad file name"), 400
    f.save(os.path.join(INPUTS, name))
    return jsonify(name=name)


@app.get("/api/analyze")
def api_analyze():
    name = _safe_name(request.args.get("name", ""))
    path = os.path.join(INPUTS, name)
    if not os.path.exists(path):
        return jsonify(error="input not found: " + name), 404
    fps = 30
    feats = M.analyze(M.decode_audio(path), 44100, fps)
    n = feats["n"]
    e = 0.5 * feats["bass"] + 0.3 * feats["mid"] + 0.2 * feats["treb"]
    # decimate curves to ~1500 points
    step = max(1, n // 1500)
    idx = list(range(0, n, step))
    energy = [round(float(e[i]), 4) for i in idx]
    punch = [round(float(feats["punch"][i]), 4) for i in idx]
    spans = M.segment_phases(feats, fps)
    plan = M.plan_scenes(spans, fps)
    return jsonify({
        "name": name, "fps": fps, "n": n,
        "dur": n / fps, "energy": energy, "punch": punch,
        "spans": [[sp[0], sp[1] / fps, sp[2] / fps] for sp in spans],
        "plan": _plan_seconds(plan, fps),
    })


def _plan_seconds(plan, fps):
    return [{"phase": sp["phase"], "scene": sp["scene"],
             "s0": sp["i0"] / fps, "s1": sp["i1"] / fps,
             "drop": (sp["drop"] / fps) if sp.get("drop") is not None
             else None}
            for sp in plan]


@app.post("/api/replan")
def api_replan():
    d = request.get_json(force=True)
    fps = int(d.get("fps", 30))
    n = int(d.get("n", 0))
    spans = d.get("spans", [])
    if not spans:
        return jsonify(error="no spans"), 400
    fs = []
    prev_end = 0
    for sp in spans:
        ph = str(sp[0])
        a = max(int(round(float(sp[1]) * fps)), prev_end)
        b = int(round(float(sp[2]) * fps))
        fs.append([ph, a, max(b, a + 1)])
        prev_end = b
    fs[0][1] = 0
    fs[-1][2] = n
    plan = M.plan_scenes(fs, fps)
    return jsonify(plan=_plan_seconds(plan, fps))


@app.get("/api/resolve_dir")
def api_resolve_dir():
    """Resolve a folder name (from the browser folder picker) to an
    absolute path under the project root, its parent, or the home dir."""
    name = _safe_name(request.args.get("name", ""))
    if not name:
        return jsonify(error="no folder name"), 400
    cands = [os.path.join(ROOT, name),
             os.path.join(os.path.dirname(ROOT), name),
             os.path.join(os.path.expanduser("~"), name)]
    for c in cands:
        if os.path.isdir(c):
            return jsonify(path=os.path.normpath(c))
    return jsonify(error="could not find folder named " + name)


@app.post("/api/render")
def api_render():
    d = request.get_json(force=True)
    name = _safe_name(d.get("name", ""))
    audio = os.path.join(INPUTS, name)
    if not os.path.exists(audio):
        return jsonify(error="input not found: " + name), 400

    out_dir = str(d.get("out_dir", "outputs"))
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)
    out_dir = os.path.normpath(out_dir)
    out_name = _safe_name(d.get("out_name") or
                          (name.rsplit(".", 1)[0] + "_viz.mp4"))
    W, H = (int(v) for v in str(d.get("size", "1280x720")).lower().split("x"))
    fps = int(d.get("fps", 30))
    adelay = int(d.get("adelay", 40))

    spans = d.get("spans") or None
    spans_frames = ([[sp[0], int(round(sp[1] * fps)), int(round(sp[2] * fps))]
                    for sp in spans] if spans else None)

    out = os.path.join(out_dir, out_name)
    use_gpu = CUDA
    use_nvenc = M.ffmpeg_has_nvenc()

    job = "j%d" % int(time.time() * 1000)

    def worker():
        def cb(i, n):
            with jobs_lock:
                jobs[job]["progress"] = i / n

        try:
            os.makedirs(out_dir, exist_ok=True)
            M.render_video(audio, out, W=W, H=H, fps=fps, adelay=adelay,
                           use_gpu=use_gpu, use_nvenc=use_nvenc,
                           phase_spans=spans_frames, progress_cb=cb)
            with jobs_lock:
                jobs[job].update(state="done", out=out)
        except Exception as ex:
            with jobs_lock:
                jobs[job].update(state="error", error=str(ex),
                                 tb=traceback.format_exc())

    with jobs_lock:
        jobs[job] = dict(state="running", progress=0.0, out=out,
                         error=None)
    threading.Thread(target=worker, daemon=True).start()
    return jsonify(job=job, out=out)


@app.get("/api/status")
def api_status():
    job = request.args.get("job", "")
    with jobs_lock:
        j = dict(jobs.get(job, {}))
    if not j:
        return jsonify(error="no such job"), 404
    j.pop("tb", None)
    return jsonify(j)


@app.get("/media")
def api_media():
    name = _safe_name(request.args.get("name", ""))
    return send_media(os.path.join(INPUTS, name), "audio/mpeg")


@app.get("/video")
def api_video():
    path = request.args.get("path", "")
    path = os.path.normpath(path)
    # only serve files under the project root or its parent
    if not (path.startswith(os.path.dirname(ROOT) + os.sep)):
        return jsonify(error="path not allowed"), 403
    return send_media(path, "video/mp4")


def main():
    port = int(os.environ.get("PORT", "8737"))
    print("Music Animation web UI:  http://127.0.0.1:%d" % port)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
