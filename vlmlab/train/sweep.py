"""Weight-average sweep: mitigation number one, and free.

In the reference experiment, interpolating between the frozen and fine-tuned
weights at a coefficient of 0.4 scored above **both** endpoints: three points
better in the wild than the frozen model while keeping most of the in-domain
gain. That makes this the first thing to try rather than a fallback, and it
costs one weight average per coefficient with no training.

The sweep is driven here, in pure arithmetic, because choosing the operating
point is a decision about a trade-off and should be auditable. The tensor
arithmetic lives in the adapter.
"""
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["DEFAULT_ALPHAS", "SweepError", "run_sweep", "choose_operating_point",
           "format_sweep"]

DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class SweepError(ValueError):
    pass


def run_sweep(alphas, evaluate, in_domain_key="in_domain",
              retention_key="retention"):
    """Evaluate each coefficient. ``evaluate(alpha)`` returns a mapping.

    Zero is the frozen model and one is the fine-tuned model, so both endpoints
    are always measured and the interpolated points are compared against them
    rather than against an assumption.
    """
    alphas = tuple(sorted(set(float(a) for a in alphas)))
    if not alphas:
        raise SweepError("no coefficients to sweep")
    for a in alphas:
        if not (0.0 <= a <= 1.0):
            raise SweepError("coefficient {} is outside [0, 1]".format(a))
    if 0.0 not in alphas or 1.0 not in alphas:
        raise SweepError(
            "both endpoints are required: 0.0 is the frozen model and 1.0 the "
            "fine-tuned one, and the interpolated points are only meaningful "
            "against them")

    rows = []
    for a in alphas:
        out = dict(evaluate(a))
        if in_domain_key not in out or retention_key not in out:
            raise SweepError(
                "evaluate({}) must return both {!r} and {!r}"
                .format(a, in_domain_key, retention_key))
        out["alpha"] = a
        rows.append(out)
    return tuple(rows)


def choose_operating_point(rows, in_domain_key="in_domain",
                           retention_key="retention",
                           min_retention_fraction=0.90):
    """Pick a coefficient, with the trade-off stated rather than hidden.

    Requires retention to stay within a stated fraction of the frozen model,
    then maximises in-domain score subject to that. Reports whether any point
    dominated both endpoints, because that is the outcome the reference
    experiment found and it is worth noticing when it happens.
    """
    if not rows:
        raise SweepError("no sweep rows")
    by_alpha = {r["alpha"]: r for r in rows}
    frozen = by_alpha.get(0.0)
    tuned = by_alpha.get(1.0)
    if frozen is None or tuned is None:
        raise SweepError("both endpoints must be present")

    floor = frozen[retention_key] * float(min_retention_fraction)
    eligible = [r for r in rows if r[retention_key] >= floor]
    if not eligible:
        # Only reachable when the floor exceeds the frozen model's own
        # retention, which requires min_retention_fraction above one.
        return {
            "chosen": None,
            "reason": ("the retention floor of {:.4f} exceeds even the frozen "
                       "model's {:.4f}, so no point can qualify. Lower "
                       "min_retention_fraction below 1.0."
                       .format(floor, frozen[retention_key])),
            "retention_floor": floor,
            "fell_back_to_frozen": False,
            "rows": rows,
        }

    best = max(eligible, key=lambda r: (r[in_domain_key], -r["alpha"]))
    dominating = [r for r in rows
                  if r[in_domain_key] > max(frozen[in_domain_key],
                                            tuned[in_domain_key])
                  and r[retention_key] > max(frozen[retention_key],
                                             tuned[retention_key])]
    # Choosing the frozen model is a real outcome, not a failure to decide: it
    # means no amount of fine-tuning retained acceptably, and saying so plainly
    # is more useful than reporting no choice at all.
    fell_back = (best["alpha"] == 0.0)
    return {
        "chosen": best["alpha"],
        "chosen_row": best,
        "fell_back_to_frozen": fell_back,
        "reason": ("fine-tuning produced nothing that kept retention within "
                   "{:.0%} of the frozen model, so the frozen model wins. "
                   "Reduce the step budget and sweep again."
                   .format(min_retention_fraction)) if fell_back else "",
        "retention_floor": floor,
        "frozen": frozen, "fine_tuned": tuned,
        "dominates_both_endpoints": tuple(r["alpha"] for r in dominating),
        "gain_over_frozen": best[in_domain_key] - frozen[in_domain_key],
        "retention_cost_vs_frozen": best[retention_key] - frozen[retention_key],
        "rows": rows,
        "note": ("in the reference experiment a coefficient of 0.4 beat both "
                 "endpoints, so check dominates_both_endpoints before assuming "
                 "a trade-off was necessary"),
    }


def format_sweep(decision, in_domain_key="in_domain",
                 retention_key="retention"):
    lines = ["weight-average sweep"]
    lines.append("  {:>6s}  {:>10s}  {:>10s}".format("alpha", "in-domain",
                                                     "retention"))
    for r in decision["rows"]:
        mark = " <-- chosen" if r["alpha"] == decision.get("chosen") else ""
        lines.append("  {:6.2f}  {:10.4f}  {:10.4f}{}".format(
            r["alpha"], r[in_domain_key], r[retention_key], mark))
    if decision.get("chosen") is None:
        lines.append("  no acceptable point: {}".format(decision["reason"]))
    elif decision.get("fell_back_to_frozen"):
        lines.append("  FELL BACK TO THE FROZEN MODEL: {}"
                     .format(decision["reason"]))
    else:
        lines.append("  gain over frozen: {:+.4f}, retention cost: {:+.4f}"
                     .format(decision["gain_over_frozen"],
                             decision["retention_cost_vs_frozen"]))
        if decision["dominates_both_endpoints"]:
            lines.append("  coefficients beating BOTH endpoints: {}".format(
                list(decision["dominates_both_endpoints"])))
    return "\n".join(lines)
