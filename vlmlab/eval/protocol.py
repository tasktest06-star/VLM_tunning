"""Evaluation protocol: grouped folds and one global threshold.

Two rules are enforced rather than recommended.

**One global threshold.** With about 150 clips and a dozen classes, any
held-out split holds two to six positives per class, and a threshold fitted on
that has a standard error wider than the interval being searched. Earlier
research proposed per-class thresholds; the later position forbids them, so
there is deliberately no per-class threshold interface here at all.

**The threshold is fitted out of fold.** Fitting on in-fold scores is
leakage, and it is the kind that produces a number nobody can reproduce. This
module raises rather than allowing it.
"""
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.splits import balance_report, choose_fold_scheme

__all__ = [
    "LeakageError", "ThresholdChoice", "choose_global_threshold",
    "build_protocol", "ProtocolReport",
]


class LeakageError(RuntimeError):
    """A score used for threshold selection came from inside its own fold."""


class ThresholdChoice(object):
    """The single global threshold, and the evidence for it.

    There is no per-class threshold attribute. That absence is the point.
    """

    def __init__(self, threshold, objective, n_scores, curve):
        self.threshold = float(threshold)
        self.objective = objective
        self.n_scores = int(n_scores)
        self.curve = tuple(curve)

    def __repr__(self):
        return ("ThresholdChoice(threshold={:.4f}, objective={}, n_scores={})"
                .format(self.threshold, self.objective, self.n_scores))


def choose_global_threshold(out_of_fold_scores, objective_fn, grid=None,
                            fold_of=None, in_fold_guard=True):
    """Pick one threshold maximising an objective over pooled out-of-fold scores.

    ``out_of_fold_scores`` is a sequence of ``(score, is_positive, fold_id)``.
    When ``in_fold_guard`` is set, every record must carry a fold identifier,
    which is what makes accidental in-fold fitting detectable.
    """
    records = list(out_of_fold_scores)
    if not records:
        raise ValueError("no scores supplied")
    if in_fold_guard:
        for i, rec in enumerate(records):
            if len(rec) < 3 or rec[2] is None:
                raise LeakageError(
                    "record {} has no fold identifier, so it cannot be shown to be "
                    "out of fold. Pass (score, is_positive, fold_id), or disable the "
                    "guard explicitly and say why.".format(i))
        if fold_of is not None:
            for i, rec in enumerate(records):
                if rec[2] == fold_of:
                    raise LeakageError(
                        "record {} belongs to fold {!r}, the very fold whose threshold "
                        "is being fitted. That is leakage.".format(i, fold_of))

    if grid is None:
        uniq = sorted({round(float(r[0]), 4) for r in records})
        grid = uniq if len(uniq) <= 200 else [i / 200.0 for i in range(201)]

    curve = []
    best = None
    for t in grid:
        value = objective_fn(records, t)
        curve.append((float(t), None if value is None else float(value)))
        if value is None:
            continue
        if best is None or value > best[1]:
            best = (float(t), float(value))
    if best is None:
        raise ValueError("objective was undefined at every threshold")
    return ThresholdChoice(best[0], objective_fn.__name__, len(records), curve)


def f1_objective(records, threshold):
    """Balanced objective suitable for a multi-label presence decision."""
    tp = fp = fn = 0
    for score, is_pos, _fold in records:
        pred = score >= threshold
        if pred and is_pos:
            tp += 1
        elif pred and not is_pos:
            fp += 1
        elif not pred and is_pos:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ProtocolReport(object):
    def __init__(self, scheme, folds, balance, n_groups, test_name, test_note):
        self.scheme = scheme
        self.folds = folds
        self.balance = balance
        self.n_groups = n_groups
        self.test_name = test_name
        self.test_note = test_note

    def summary(self):
        return {
            "scheme": self.scheme,
            "n_folds": len(self.folds),
            "n_groups": self.n_groups,
            "statistical_test": self.test_name,
            "test_note": self.test_note,
            "warnings": list(self.balance.get("warnings", ())),
        }


def build_protocol(clips, cfg, labels_of=None):
    """Resolve the fold scheme and the statistical test for this dataset."""
    from vlmlab.eval.stats import choose_test

    group_of = lambda c: c.session_id
    if labels_of is None:
        labels_of = lambda c: c.labels

    folds, scheme = choose_fold_scheme(
        clips, group_of, loso_below=cfg.eval.loso_below_sessions,
        n_folds=cfg.eval.n_folds, seed=cfg.eval.seed)
    balance = balance_report(folds, labels_of)
    n_groups = len({group_of(c) for c in clips})
    test_name, test_note = choose_test(n_groups, cfg.eval.permutation_below_groups)
    return ProtocolReport(scheme, folds, balance, n_groups, test_name, test_note)
