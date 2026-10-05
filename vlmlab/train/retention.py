"""Retention harness: does fine-tuning keep the generalisation it depends on?

Fine-tuning is in the plan, which reverses an earlier prohibition. The collapse
figure that prohibition rested on came from a thermal-infrared fine-tune
evaluated on colour images, so it measured a modality shift rather than a
vocabulary effect, and this project is colour throughout.

But the good news has a trap inside it, and this module exists for the trap.
In the reference experiment the **mean** across thirty-five held-out domains
fell nine percent while the **median** fell sixty-one. A mean-based retention
metric therefore hides exactly the failure it is meant to detect.

So: hold classes out entirely, report the median, and treat the mean as a
diagnostic only.
"""
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "choose_held_out_classes", "RetentionError", "retention_report",
    "format_retention",
]


class RetentionError(ValueError):
    pass


def choose_held_out_classes(registry, n=6, prefer_with_public_boxes=True,
                            always_keep=()):
    """Pick classes to withhold from fine-tuning entirely.

    Prefers classes that have public boxes, so that withholding them does not
    also waste the scarce hand annotation. Never withholds a class the caller
    insists on keeping, and never withholds so many that nothing is left to
    train on.
    """
    names = list(registry.names)
    if n >= len(names):
        raise RetentionError(
            "cannot hold out {} of {} classes and still have anything to "
            "fine-tune on".format(n, len(names)))
    keep = set(always_keep)
    candidates = [x for x in names if x not in keep]
    if prefer_with_public_boxes:
        candidates.sort(key=lambda x: (not registry.get(x).has_public_boxes, x))
    else:
        candidates.sort()
    chosen = tuple(candidates[:n])
    if len(chosen) < n:
        raise RetentionError("only {} classes available to hold out".format(
            len(chosen)))
    return chosen


def retention_report(before, after, held_out):
    """Compare per-class scores before and after, on withheld classes only.

    ``before`` and ``after`` map class name to a score. Classes missing from
    either side are reported rather than silently skipped, because a missing
    class is usually a wiring error and quietly dropping it would flatter the
    result.
    """
    held = list(held_out)
    if not held:
        raise RetentionError("no held-out classes supplied")
    missing_before = [c for c in held if c not in before]
    missing_after = [c for c in held if c not in after]
    shared = [c for c in held if c in before and c in after]
    if not shared:
        raise RetentionError(
            "none of the held-out classes appear on both sides; missing before "
            "{}, missing after {}".format(missing_before, missing_after))

    b = [float(before[c]) for c in shared]
    a = [float(after[c]) for c in shared]
    med_b = statistics.median(b)
    med_a = statistics.median(a)
    mean_b = statistics.mean(b)
    mean_a = statistics.mean(a)

    per_class = {}
    for c in shared:
        delta = float(after[c]) - float(before[c])
        rel = (delta / float(before[c])) if before[c] else None
        per_class[c] = {"before": float(before[c]), "after": float(after[c]),
                        "delta": delta, "relative": rel}

    median_rel = (med_a - med_b) / med_b if med_b else None
    mean_rel = (mean_a - mean_b) / mean_b if mean_b else None

    warnings = []
    if missing_before or missing_after:
        warnings.append("held-out classes missing from a side: before {}, "
                        "after {}".format(missing_before, missing_after))
    if median_rel is not None and mean_rel is not None:
        if median_rel < -0.2 and mean_rel > -0.1:
            warnings.append(
                "the mean fell {:.0%} while the median fell {:.0%}. This is the "
                "exact pattern a mean-based retention metric hides: report the "
                "median.".format(mean_rel, median_rel))
    if median_rel is not None and median_rel < -0.25:
        warnings.append(
            "median retention fell {:.0%} on withheld classes. Reduce the step "
            "budget, or move further along the weight-average sweep."
            .format(median_rel))

    return {
        "n_held_out": len(shared),
        "held_out": tuple(shared),
        "median_before": med_b, "median_after": med_a,
        "median_delta": med_a - med_b, "median_relative": median_rel,
        "mean_before": mean_b, "mean_after": mean_a,
        "mean_delta": mean_a - mean_b, "mean_relative": mean_rel,
        "per_class": per_class,
        "headline": "median",
        "warnings": warnings,
        "note": ("report the median. In the reference experiment a nine percent "
                 "mean drop concealed a sixty-one percent median collapse."),
    }


def format_retention(report):
    lines = ["retention on {} withheld classes (headline is the MEDIAN)"
             .format(report["n_held_out"])]
    lines.append("  median {:.4f} -> {:.4f}  ({:+.1%})".format(
        report["median_before"], report["median_after"],
        report["median_relative"] if report["median_relative"] is not None else 0.0))
    lines.append("  mean   {:.4f} -> {:.4f}  ({:+.1%})   [diagnostic only]".format(
        report["mean_before"], report["mean_after"],
        report["mean_relative"] if report["mean_relative"] is not None else 0.0))
    for name in report["held_out"]:
        pc = report["per_class"][name]
        lines.append("    {:22s} {:.4f} -> {:.4f}  ({:+.4f})".format(
            name, pc["before"], pc["after"], pc["delta"]))
    for w in report["warnings"]:
        lines.append("  WARNING: {}".format(w))
    return "\n".join(lines)
