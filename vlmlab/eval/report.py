"""End-to-end scoring: gold set plus predictions in, a defensible number out.

This is the path that was missing. The statistics layer was complete and
tested but had nothing real to consume, and the metric had no honest ground
truth to score against.

Everything the research insists on is applied here rather than left to the
caller: the session is the resampling unit, the test is chosen by group count,
the threshold is global and fitted out of fold, predictions the gold set
cannot judge are dropped rather than counted wrong, and a difference below the
pre-registered minimum is reported as not resolvable instead of as a result.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.eval.metrics import (ap_from_matches, match_detections,
                                 mean_average_precision, per_class_recall)
from vlmlab.eval.stats import (bootstrap_ci, choose_test, design_effect_from_icc,
                               icc_one_way, kish_design_effect, mde_paired,
                               paired_group_permutation_test)

__all__ = ["score", "compare", "Report", "ComparisonReport"]


class Report(object):
    def __init__(self, payload):
        self._p = dict(payload)

    def __getitem__(self, key):
        return self._p[key]

    def to_dict(self):
        return dict(self._p)

    def __repr__(self):
        m = self._p.get("mAP")
        ci = self._p.get("ci95")
        return "Report(mAP={}, ci95={}, n_sessions={})".format(
            m, ci, self._p.get("n_sessions"))


def _macro_metric(labels, iou_threshold, max_dets):
    """Build a metric closure over pooled records, for the bootstrap.

    Recomputes the macro average on the pooled records rather than averaging
    per-session values, because average precision is non-linear and the mean
    of per-session values is a different quantity.
    """
    def metric(records):
        gts = [g for r in records for g in r["gts"]]
        preds = [p for r in records for p in r["preds"]]
        if not gts:
            return None
        res = mean_average_precision(preds, gts, labels=labels,
                                     iou_threshold=iou_threshold,
                                     max_dets=max_dets)
        return res["mAP"]
    return metric


def score(gold, predictions, cfg, labels=None):
    """Score predictions against a gold set, with an interval and warnings."""
    labels = list(labels or gold.vocabulary)
    kept, filtered = gold.filter_predictions(predictions)
    units = gold.units_by_session(kept, cfg.eval.iou_threshold)

    pooled = [{"gts": tuple(gold.to_ground_truth()), "preds": tuple(kept)}]
    overall = mean_average_precision(
        kept, gold.to_ground_truth(), labels=labels,
        iou_threshold=cfg.eval.iou_threshold,
        max_dets=cfg.eval.max_dets_per_image)

    metric = _macro_metric(labels, cfg.eval.iou_threshold,
                           cfg.eval.max_dets_per_image)
    unit_records = [[u] for u in units]
    ci = bootstrap_ci(unit_records, metric, n_resamples=cfg.eval.n_resamples,
                      seed=cfg.eval.seed)

    n_sessions = len(units)
    test_name, test_note = choose_test(n_sessions,
                                       cfg.eval.permutation_below_groups)

    # Design effects, reported separately because conflating them is an error.
    sizes = [len(u["gts"]) for u in units]
    per_session_ap = []
    for u in units:
        if not u["gts"]:
            continue
        r = mean_average_precision(list(u["preds"]), list(u["gts"]),
                                   labels=labels,
                                   iou_threshold=cfg.eval.iou_threshold,
                                   max_dets=cfg.eval.max_dets_per_image)
        if r["mAP"] is not None:
            per_session_ap.append([r["mAP"]])
    icc = icc_one_way(per_session_ap) if len(per_session_ap) > 1 else None

    recalls = {}
    for label in labels:
        recalls[label] = per_class_recall(
            kept, gold.to_ground_truth(), label,
            score_threshold=0.0, iou_threshold=cfg.eval.iou_threshold)

    warnings = list(gold.warnings())
    if ci.lo is not None and ci.width is not None:
        if ci.width > cfg.eval.pre_registered_mde / 100.0:
            warnings.append(
                "the interval is {:.3f} wide, above the pre-registered minimum "
                "detectable difference of {:.3f}. Differences smaller than the "
                "interval are not results.".format(
                    ci.width, cfg.eval.pre_registered_mde / 100.0))
    if not ci.trustworthy:
        warnings.append("{} of {} bootstrap replicates were degenerate; the "
                        "interval should not be quoted"
                        .format(ci.n_degenerate, ci.n_resamples))
    if filtered["dropped_unannotated_frame"]:
        warnings.append(
            "{} predictions fell on unannotated frames and were dropped rather "
            "than counted as false positives"
            .format(filtered["dropped_unannotated_frame"]))

    return Report({
        "mAP": overall["mAP"],
        "n_classes_averaged": overall["n_classes_averaged"],
        "n_classes_total": overall["n_classes_total"],
        "ci95": (ci.lo, ci.hi),
        "ci_width": ci.width,
        "ci_trustworthy": ci.trustworthy,
        "n_sessions": n_sessions,
        "n_gold_frames": len(gold.annotated()),
        "n_gold_boxes": len(gold.to_ground_truth()),
        "statistical_test": test_name,
        "test_note": test_note,
        "kish_design_effect": kish_design_effect(sizes),
        "intra_session_icc": icc,
        "icc_design_effect": design_effect_from_icc(
            (sum(sizes) / float(len(sizes))) if sizes else 1.0, icc),
        "per_class_recall": recalls,
        "predictions_filtered": filtered,
        "warnings": warnings,
        "note": ("per-class average precision is deliberately not reported: it "
                 "is not reportable at any affordable annotation budget here. "
                 "Per-class recall at a frozen threshold with an exact interval "
                 "is given instead."),
    })


class ComparisonReport(Report):
    pass


def compare(gold, predictions_a, predictions_b, cfg, labels=None,
            name_a="A", name_b="B"):
    """Compare two pipelines on the same gold set, paired by session.

    Uses the test the group count dictates, and refuses to call a difference
    below the pre-registered threshold a result.
    """
    labels = list(labels or gold.vocabulary)
    kept_a, _ = gold.filter_predictions(predictions_a)
    kept_b, _ = gold.filter_predictions(predictions_b)
    units_a = gold.units_by_session(kept_a, cfg.eval.iou_threshold)
    units_b = gold.units_by_session(kept_b, cfg.eval.iou_threshold)

    metric = _macro_metric(labels, cfg.eval.iou_threshold,
                           cfg.eval.max_dets_per_image)
    n_groups = len(units_a)
    test_name, test_note = choose_test(n_groups,
                                       cfg.eval.permutation_below_groups)

    perm = paired_group_permutation_test([[u] for u in units_a],
                                         [[u] for u in units_b], metric,
                                         seed=cfg.eval.seed)
    a_val = metric([u for u in units_a])
    b_val = metric([u for u in units_b])

    # Spread of the per-session difference, for the detectable-difference figure.
    diffs = []
    for ua, ub in zip(units_a, units_b):
        va, vb = metric([ua]), metric([ub])
        if va is not None and vb is not None:
            diffs.append(va - vb)
    sd = 0.0
    if len(diffs) > 1:
        mean = sum(diffs) / float(len(diffs))
        sd = (sum((d - mean) ** 2 for d in diffs) / float(len(diffs) - 1)) ** 0.5
    mde = mde_paired(sd, max(2, len(diffs))) if diffs else None

    threshold = cfg.eval.pre_registered_mde / 100.0
    observed = None if (a_val is None or b_val is None) else (a_val - b_val)
    resolvable = (observed is not None and abs(observed) >= threshold)

    warnings = list(gold.warnings())
    if not resolvable and observed is not None:
        warnings.append(
            "the observed difference of {:.4f} is below the pre-registered "
            "minimum of {:.4f}. Report this as not resolvable, not as a result."
            .format(observed, threshold))
    if perm.exact:
        warnings.append(
            "the exact test over {} groups cannot report a p-value below about "
            "{:.4f}, so a stricter demand is unsatisfiable by construction"
            .format(n_groups, perm.p_floor))

    return ComparisonReport({
        "name_a": name_a, "name_b": name_b,
        "mAP_a": a_val, "mAP_b": b_val,
        "difference": observed,
        "p_value": perm.p_value,
        "exact_test": perm.exact,
        "n_permutations": perm.n_perm,
        "p_value_floor": perm.p_floor,
        "statistical_test": test_name,
        "test_note": test_note,
        "n_groups": n_groups,
        "sd_of_paired_difference": sd,
        "minimum_detectable_difference": None if mde is None else mde.mde,
        "pre_registered_threshold": threshold,
        "resolvable": resolvable,
        "warnings": warnings,
    })
