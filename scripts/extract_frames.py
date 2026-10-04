#!/usr/bin/env python3
"""Extract frames once, outside any training loop.

Decoding inside a data loader was measured as the actual wall clock in a
comparable pipeline: it seeks per frame, tests membership against a list, and
converts to an image, per clip and per epoch, with no worker processes. A
three-minute clip at thirty frames per second means over five thousand full
decodes to retrieve a handful of frames.

The backend interfaces accept a frame path and never a video path, so this
step is structurally required rather than merely advised.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import subprocess
import sys


def probe(path):
    """Duration, frame count and rate, via the standard probe tool."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    nb = stream.get("nb_frames")
    rate = stream.get("avg_frame_rate", "0/1")
    num, _, den = rate.partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(stream.get("duration") or 0.0)
    count = int(nb) if nb and nb != "N/A" else int(duration * fps)
    if count <= 0:
        raise ValueError(
            "{} reports no frame count. Refusing to guess: a fallback here "
            "silently samples only the opening of the clip.".format(path))
    return {"duration_s": duration, "container_frame_count": count, "fps": fps}


def extract(path, clip_id, out_dir, fps, quality=2):
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "{}_%06d.jpg".format(clip_id))
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
           "-vf", "fps={}".format(fps), "-q:v", str(quality), pattern]
    subprocess.run(cmd, check=True)
    return sorted(f for f in os.listdir(out_dir) if f.startswith(clip_id + "_"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON with clips: clip_id, session_id, labels, path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--write-manifest",
                    help="write back a manifest with probed durations")
    args = ap.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for clip in data["clips"]:
        info = probe(clip["path"])
        clip.update(info)
        files = extract(clip["path"], clip["clip_id"], args.out_dir, args.fps)
        clip["n_extracted"] = len(files)
        print("{}: {} frames from {:.1f}s".format(
            clip["clip_id"], len(files), info["duration_s"]))

    if args.write_manifest:
        with open(args.write_manifest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
