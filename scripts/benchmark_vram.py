#!/usr/bin/env python3
"""Measure peak detector memory. This is experiment one.

The research could not find peak memory for this detector on a 12 GB card
measured anywhere. The only nearby data points are under 25 GB on a
datacentre card for long multi-class video, and an out-of-memory report on a
24 GB card. So this is the single biggest unknown in the whole plan, and every
cost estimate downstream depends on it.

The schedule and the pass criteria are pure Python and tested. Only the
measuring needs a GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import sys


def build_schedule(max_vram_gb=12.0):
    """Resolutions and frame counts to try, cheapest first.

    Ordered so that the run produces a usable answer even if it fails part
    way through, which it will when it hits the ceiling.
    """
    from vlmlab.frames import ResolutionError, validate_resolution
    rows = []
    for frame_size in (336, 448, 560, 672, 1008):
        for n_frames in (1, 4, 8, 16):
            try:
                tokens = validate_resolution(frame_size, n_frames, max_tokens=100000)
            except ResolutionError:
                continue
            rows.append({"frame_size": frame_size, "n_frames": n_frames,
                         "est_vision_tokens": tokens})
    rows.sort(key=lambda r: r["est_vision_tokens"])
    return rows


def verdict(peak_gb, budget_gb=12.0, headroom=0.85):
    """Pass only with headroom; a configuration at the ceiling will fail later."""
    limit = budget_gb * headroom
    if peak_gb is None:
        return "unknown"
    if peak_gb <= limit:
        return "pass"
    if peak_gb <= budget_gb:
        return "marginal"
    return "fail"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=False, help="path to one extracted frame")
    ap.add_argument("--budget-gb", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule without touching a GPU")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    schedule = build_schedule(args.budget_gb)
    if args.dry_run or not args.frame:
        print(json.dumps({"schedule": schedule,
                          "note": "dry run; no GPU was touched"}, indent=2))
        return 0

    import torch
    from vlmlab.backends.sam3 import Sam3Detector
    from vlmlab.config import Config

    results = []
    for row in schedule:
        cfg = Config()
        cfg.detector.frame_size = row["frame_size"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak = None
        error = None
        try:
            det = Sam3Detector(cfg)
            det.detect(args.frame, ("laboratory instrument",))
            peak = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        except RuntimeError as exc:
            error = str(exc)[:200]
        row = dict(row)
        row.update({"peak_gb": peak, "verdict": verdict(peak, args.budget_gb),
                    "error": error})
        results.append(row)
        print(json.dumps(row))
        if error and "out of memory" in error.lower():
            break

    doc = {"budget_gb": args.budget_gb, "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
