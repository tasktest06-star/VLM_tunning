"""Preflight checks. Importable, mostly pure functions, so the logic is testable.

The eleven verified engineering traps are enforced in four tiers. This module
holds the two tiers that can be checked at runtime:

* pure functions over an environment mapping or a configuration, which run and
  are tested in any environment;
* guards over a loaded model object, which need torch and live in
  ``backends/_torch_guards.py``.

The other two tiers are elsewhere by design: frozen configuration defaults in
``config.py``, and structural guarantees in ``backends/base.py`` where the
detect method wraps the implementation in inference mode so a subclass cannot
forget it.
"""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PreflightFailure", "Finding", "check_single_visible_device",
    "check_no_flash_attention", "check_config_frozen_fields",
    "check_concept_budget", "check_frame_grid", "check_provenance",
    "run_pure_checks", "format_report",
]


class PreflightFailure(RuntimeError):
    pass


class Finding(object):
    def __init__(self, name, ok, detail, severity="error"):
        self.name = name
        self.ok = bool(ok)
        self.detail = detail
        self.severity = severity

    def __repr__(self):
        return "Finding({}, ok={}, {})".format(self.name, self.ok, self.severity)


def check_single_visible_device(env):
    """Exactly one device must be visible.

    A launcher switches to distributed training whenever more than one device
    is present, which doubles the effective batch and halves the optimiser
    steps. At this dataset size that changes the schedule far more than the
    throughput gain is worth. Shard clips across two processes instead.
    """
    raw = env.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw == "":
        return Finding("single_visible_device", False,
                       "CUDA_VISIBLE_DEVICES is unset, so every device is visible and "
                       "the launcher will silently run distributed. Pin one device.")
    ids = [p for p in str(raw).split(",") if p.strip() != ""]
    if len(ids) != 1:
        return Finding("single_visible_device", False,
                       "{} devices visible ({}). Pin exactly one and shard clips "
                       "across processes.".format(len(ids), raw))
    return Finding("single_visible_device", True,
                   "one device visible ({})".format(raw))


def check_no_flash_attention(modules, cfg=None):
    """Flash attention must be neither imported nor requested.

    No wheels are published, so installing it means a source build that takes
    hours or is killed for memory. The attention implementation built into the
    tensor library is sufficient at these sequence lengths.
    """
    if "flash_attn" in modules:
        return Finding("no_flash_attention", False,
                       "flash_attn is imported. It cannot be installed from the "
                       "package index and is unnecessary here.")
    if cfg is not None and getattr(cfg.detector, "allow_flash_attention", False):
        return Finding("no_flash_attention", False,
                       "configuration requests flash attention")
    return Finding("no_flash_attention", True, "absent and not requested")


def check_config_frozen_fields(cfg):
    """Every frozen field still holds its required value."""
    from vlmlab.config import FROZEN_PATHS
    problems = []
    expected = {
        "detector.attn_implementation": "sdpa",
        "detector.quant_type": "nf4",
        "detector.quant_double": True,
        "detector.compute_dtype": "bfloat16",
        "detector.allow_flash_attention": False,
        "detector.require_detection_head_boxes": True,
        "eval.batch_size": 1,
        "eval.per_class_thresholds": False,
    }
    for path in sorted(FROZEN_PATHS):
        section, field = path.split(".", 1)
        actual = getattr(getattr(cfg, section), field)
        if path in expected and actual != expected[path]:
            problems.append("{} is {!r}, expected {!r}".format(path, actual,
                                                               expected[path]))
    if problems:
        return Finding("frozen_fields", False, "; ".join(problems))
    return Finding("frozen_fields", True,
                   "{} frozen fields hold".format(len(FROZEN_PATHS)))


def check_concept_budget(plans, max_concepts, per_concept_s=0.31, keyframes=45,
                         n_clips=150):
    """Warn when the prompt budget makes the run unaffordable.

    Cost is linear in concepts because the per-concept decoder pass does not
    amortise, only image encoding does. On a three-minute clip, restricting
    the vocabulary is worth roughly two and a half times.
    """
    if not plans:
        return Finding("concept_budget", True, "no plans to check", "warning")
    worst = max(p.n_concepts for p in plans)
    mean = sum(p.n_concepts for p in plans) / float(len(plans))
    hours = keyframes * (0.03 + mean * per_concept_s) * n_clips / 3600.0
    detail = ("mean {:.1f} concepts per clip, worst {}; about {:.1f} hours for {} clips "
              "at {} keyframes".format(mean, worst, hours, n_clips, keyframes))
    if worst > max_concepts:
        return Finding("concept_budget", False,
                       detail + "; worst exceeds the cap of {}".format(max_concepts))
    if mean > 8:
        return Finding("concept_budget", False, detail +
                       "; above eight concepts per clip the pool becomes days rather "
                       "than nights", "warning")
    return Finding("concept_budget", True, detail)


def check_frame_grid(grid, require_cap_bound=False):
    """The realised grid must match what was asked for."""
    if grid is None:
        return Finding("frame_grid", False, "no frame grid was produced")
    if require_cap_bound and not grid.cap_bound:
        return Finding("frame_grid", False,
                       "realised {} frames but the cap is {}; every downstream memory "
                       "and timing figure would describe a configuration you did not "
                       "run".format(grid.n_frames, grid.cap))
    return Finding("frame_grid", True,
                   "{} frames, cap_bound={}, from {} source frames"
                   .format(grid.n_frames, grid.cap_bound, grid.source_frame_count))


def check_provenance(cfg):
    """No unmeasured value may feed a gate threshold."""
    from vlmlab.config import Provenance
    report = cfg.provenance_report()
    unmeasured = report.get(Provenance.UNMEASURED.value, [])
    gates = [p for p in unmeasured
             if "threshold" in p[0] or "fragmentation" in p[0]]
    if gates:
        return Finding("provenance", False,
                       "unmeasured values feed gates: {}"
                       .format(", ".join(p[0] for p in gates)), "warning")
    return Finding("provenance", True,
                   "{} unmeasured values, none feeding a gate"
                   .format(len(unmeasured)), "warning")


def run_pure_checks(cfg, env, modules, plans=(), grid=None):
    """Every check that needs no GPU. Returns findings, worst first."""
    findings = [
        check_single_visible_device(env),
        check_no_flash_attention(modules, cfg),
        check_config_frozen_fields(cfg),
        check_concept_budget(plans, cfg.detector.max_concepts_per_clip),
        check_provenance(cfg),
    ]
    if grid is not None:
        findings.append(check_frame_grid(grid))
    return tuple(sorted(findings, key=lambda f: (f.ok, f.severity != "error")))


def format_report(findings):
    lines = []
    for f in findings:
        mark = "PASS" if f.ok else ("WARN" if f.severity == "warning" else "FAIL")
        lines.append("  [{}] {}: {}".format(mark, f.name, f.detail))
    failed = [f for f in findings if not f.ok and f.severity == "error"]
    lines.append("  {} checks, {} failed".format(len(findings), len(failed)))
    return "\n".join(lines)
