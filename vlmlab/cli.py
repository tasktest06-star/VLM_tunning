"""Command line. Subcommands: preflight, plan, shard, run, protocol, provenance.

``shard`` deserves a note. The two cards are used as two independent
single-GPU processes dividing the clip list, never as one distributed job.
On this hardware there is no peer-to-peer link, so sharding a model across the
bus would be communication-bound, while two independent workers have zero
inter-card traffic and scale almost linearly.
"""
import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

__all__ = ["main", "shard_clips"]


def shard_clips(clip_ids, index, of):
    """Deterministic, disjoint, exhaustive division of the clip list."""
    if of < 1:
        raise ValueError("--of must be at least 1")
    if not (0 <= index < of):
        raise ValueError("--index must be in [0, of)")
    ordered = sorted(clip_ids)
    return tuple(c for i, c in enumerate(ordered) if i % of == index)


def _load_manifest(path):
    from vlmlab.types import ClipRecord
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    clips = []
    for item in data["clips"]:
        clips.append(ClipRecord(
            clip_id=item["clip_id"], session_id=item["session_id"],
            labels=tuple(item["labels"]),
            duration_s=item.get("duration_s"),
            container_frame_count=item.get("container_frame_count"),
            fps=item.get("fps"), room_id=item.get("room_id")))
    return clips


def _load_config(path, frame_size=None):
    from vlmlab.config import Config
    cfg = Config.from_json(path) if path else Config()
    if frame_size:
        cfg.detector.frame_size = frame_size
    return cfg


def _load_registry(path):
    from vlmlab.registry import Registry
    if not path:
        return Registry()
    with open(path, "r", encoding="utf-8") as fh:
        return Registry.from_dict(json.load(fh))


def cmd_preflight(args):
    from vlmlab.preflight import format_report, run_pure_checks
    cfg = _load_config(args.config, args.frame_size)
    plans = []
    if args.manifest:
        reg = _load_registry(args.registry)
        from vlmlab.prompts import build_clip_prompt_plan
        for clip in _load_manifest(args.manifest):
            plans.append(build_clip_prompt_plan(
                clip, reg, max_concepts=cfg.detector.max_concepts_per_clip,
                n_hard_negatives=cfg.cascade.n_hard_negatives))
    findings = run_pure_checks(cfg, os.environ, set(sys.modules), plans)
    print(format_report(findings))
    failed = [f for f in findings if not f.ok and f.severity == "error"]
    return 1 if failed else 0


def cmd_plan(args):
    cfg = _load_config(args.config, args.frame_size)
    cfg.validate()
    clips = _load_manifest(args.manifest)
    reg = _load_registry(args.registry)
    from vlmlab.prompts import build_clip_prompt_plan
    total = 0.0
    kf = cfg.estimated_keyframes_per_clip()
    print("clips={} sessions={} keyframes/clip={} min_chunks/clip={}".format(
        len(clips), len({c.session_id for c in clips}), kf,
        cfg.min_chunks_per_clip()))
    for clip in clips:
        plan = build_clip_prompt_plan(
            clip, reg, max_concepts=cfg.detector.max_concepts_per_clip,
            n_hard_negatives=cfg.cascade.n_hard_negatives)
        total += plan.estimated_seconds(kf)
    print("estimated detector time: {:.1f} h for {} clips ({:.1f} h for a 10x pool)"
          .format(total / 3600.0, len(clips), total * 10 / 3600.0))
    print("note: cost is linear in concepts; per-clip vocabulary restriction is "
          "what keeps this affordable")
    return 0


def cmd_shard(args):
    clips = _load_manifest(args.manifest)
    mine = shard_clips([c.clip_id for c in clips], args.index, args.of)
    for c in mine:
        print(c)
    return 0


def cmd_protocol(args):
    cfg = _load_config(args.config, args.frame_size)
    clips = _load_manifest(args.manifest)
    from vlmlab.eval.protocol import build_protocol
    report = build_protocol(clips, cfg)
    print(json.dumps(report.summary(), indent=2))
    return 0


def cmd_provenance(args):
    cfg = _load_config(args.config, args.frame_size)
    report = cfg.provenance_report()
    for tier in ("verified", "estimated", "unmeasured"):
        rows = report.get(tier, [])
        print("\n{} ({})".format(tier.upper(), len(rows)))
        for path, value, source, note in rows:
            line = "  {:42s} {!r:22s} {}".format(path, value, source)
            print(line)
            if note:
                print("  {:42s} {}".format("", note))
    return 0


def cmd_run(args):
    cfg = _load_config(args.config, args.frame_size)
    cfg.validate()
    clips = _load_manifest(args.manifest)
    reg = _load_registry(args.registry)
    from vlmlab.logging_setup import get_logger, log_realised_config
    logger = get_logger("vlmlab", log_dir=args.log_dir, run_id=args.run_id)
    log_realised_config(logger, cfg)

    if args.backend == "fake":
        from vlmlab.backends.fake import FakeDetector, FakePropagator
        det = FakeDetector(image_size=(cfg.detector.frame_size,
                                       cfg.detector.frame_size))
        prop = FakePropagator()
    else:
        from vlmlab.backends.sam2 import Sam2Propagator
        from vlmlab.backends.sam3 import Sam3Detector
        det = Sam3Detector(cfg)
        prop = Sam2Propagator(cfg)

    from vlmlab.pipeline import run_pipeline
    results, report = run_pipeline(clips, det, prop, reg, cfg,
                                   frame_dir=args.frame_dir, logger=logger)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"report": report,
                       "clips": [r.summary() for r in results]}, fh, indent=2)
    return 0


def _add_common(p):
    """Accepted before or after the subcommand, so entry points stay simple."""
    p.add_argument("--config")
    p.add_argument("--registry")
    p.add_argument("--frame-size", type=int, dest="frame_size")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vlmlab")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("preflight", help="run the checks that need no GPU")
    p.add_argument("--manifest")
    _add_common(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("plan", help="estimate cost before spending it")
    p.add_argument("--manifest", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("shard", help="divide clips across two single-GPU workers")
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--of", type=int, required=True)
    _add_common(p)
    p.set_defaults(func=cmd_shard)

    p = sub.add_parser("protocol", help="resolve folds and the statistical test")
    p.add_argument("--manifest", required=True)
    _add_common(p)
    p.set_defaults(func=cmd_protocol)

    p = sub.add_parser("provenance", help="show how well grounded each value is")
    _add_common(p)
    p.set_defaults(func=cmd_provenance)

    p = sub.add_parser("run", help="run the pipeline")
    p.add_argument("--manifest", required=True)
    p.add_argument("--backend", choices=("fake", "real"), default="fake")
    p.add_argument("--frame-dir")
    p.add_argument("--log-dir")
    p.add_argument("--run-id")
    p.add_argument("--out")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
