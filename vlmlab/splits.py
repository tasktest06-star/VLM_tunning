"""Session-grouped splitting, and the leakage diagnostic.

Grouping by clip is **not** sufficient. Several clips typically show the same
physical instrument on the same bench under the same lighting, so a clip-level
split still leaks instrument identity and background, and a frozen-feature
model will match the background and score near ceiling. The grouping unit is
the recording session.

Stratifying within groups is deliberately **not** offered. At nine to twenty
sessions and a dozen classes it is not achievable, and pretending otherwise
produces folds that satisfy neither constraint. Grouping is the hard
constraint; class balance is a reported diagnostic that warns.
"""
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "TooFewGroupsError", "group_kfold", "leave_one_group_out",
    "choose_fold_scheme", "balance_report", "grouped_vs_ungrouped_gap",
    "leaky_clip_kfold",
]


class TooFewGroupsError(ValueError):
    pass


def _group_members(items, group_of):
    groups = {}
    for it in items:
        groups.setdefault(group_of(it), []).append(it)
    return groups


def group_kfold(items, group_of, n_splits, seed=0, min_groups=4):
    """Partition into folds so that no group straddles a fold boundary.

    Greedy largest-group-first assignment to the currently smallest fold,
    which keeps fold sizes closer than random assignment at small group
    counts. Fully determined by ``seed``.

    Returns a tuple of ``(train_items, val_items)`` pairs.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = _group_members(items, group_of)
    names = sorted(groups)
    if len(names) < min_groups:
        raise TooFewGroupsError(
            "{} groups is too few to split meaningfully; at least {} are needed. "
            "With this many recording sessions the confidence interval is floored "
            "regardless of how many clips you label.".format(len(names), min_groups))
    if len(names) < n_splits:
        raise TooFewGroupsError(
            "{} groups cannot fill {} folds".format(len(names), n_splits))

    # Shuffle first, then sort by size only, so that equally sized groups keep
    # the shuffled order. Sorting by name as a tiebreak would make the seed a
    # no-op whenever groups are the same size, which is the common case.
    rng = random.Random(seed)
    rng.shuffle(names)
    shuffled_rank = {g: i for i, g in enumerate(names)}
    names.sort(key=lambda g: (-len(groups[g]), shuffled_rank[g]))

    buckets = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for g in names:
        target = min(range(n_splits), key=lambda i: (sizes[i], i))
        buckets[target].append(g)
        sizes[target] += len(groups[g])

    folds = []
    for i in range(n_splits):
        val_groups = set(buckets[i])
        val = [it for g in sorted(val_groups) for it in groups[g]]
        train = [it for g in sorted(set(names) - val_groups) for it in groups[g]]
        folds.append((tuple(train), tuple(val)))
    return tuple(folds)


def leave_one_group_out(items, group_of):
    """One fold per group. The right scheme when groups are few."""
    groups = _group_members(items, group_of)
    names = sorted(groups)
    if len(names) < 2:
        raise TooFewGroupsError("need at least 2 groups")
    folds = []
    for held in names:
        val = tuple(groups[held])
        train = tuple(it for g in names if g != held for it in groups[g])
        folds.append((train, val))
    return tuple(folds)


def choose_fold_scheme(items, group_of, loso_below=12, n_folds=5, seed=0):
    """Leave one session out when sessions are few, otherwise grouped folds.

    The research switched the grouping unit to sessions without fixing a fold
    count; this is the rule being adopted.
    """
    n_groups = len(_group_members(items, group_of))
    if n_groups < loso_below:
        return leave_one_group_out(items, group_of), "leave-one-session-out"
    return group_kfold(items, group_of, n_folds, seed=seed), "grouped-{}-fold".format(n_folds)


def balance_report(folds, labels_of, all_labels=None):
    """Per-fold class counts, plus warnings. A diagnostic, never a constraint."""
    if all_labels is None:
        seen = set()
        for train, val in folds:
            for it in list(train) + list(val):
                seen.update(labels_of(it))
        all_labels = sorted(seen)
    report = {"labels": list(all_labels), "folds": [], "warnings": []}
    for i, (train, val) in enumerate(folds):
        counts = {lab: 0 for lab in all_labels}
        for it in val:
            for lab in labels_of(it):
                if lab in counts:
                    counts[lab] += 1
        report["folds"].append({"fold": i, "n_val": len(val), "counts": counts})
        for lab in all_labels:
            if counts[lab] == 0:
                report["warnings"].append(
                    "fold {} has no validation positives for {!r}, so its per-class "
                    "score is undefined, not zero".format(i, lab))
    # Per-class total across folds, for the unfalsifiability warning.
    totals = {lab: sum(f["counts"][lab] for f in report["folds"]) for lab in all_labels}
    report["totals"] = totals
    for lab, n in sorted(totals.items()):
        if 0 < n < 8:
            report["warnings"].append(
                "class {!r} appears in only {} validation items; per-class average "
                "precision is effectively unfalsifiable below about 8".format(lab, n))
    return report


def leaky_clip_kfold(items, clip_of, n_splits, seed=0):
    """A deliberately leaky splitter, grouped by clip rather than session.

    Exists only as a negative control in the test suite. If the leakage test
    cannot detect this, it has no power and would not detect a real mistake
    either. Do not use it in a pipeline.
    """
    return group_kfold(items, clip_of, n_splits, seed=seed, min_groups=2)


def grouped_vs_ungrouped_gap(items, group_of, clip_of, score_fn, n_splits=5, seed=0):
    """Measure the inflation from splitting on the wrong unit.

    The research names this number as a deliverable: publish it as your
    leakage estimate. A large positive gap means a clip-level or frame-level
    split would have flattered the model.

    ``score_fn(train, val)`` returns a scalar.
    """
    grouped = group_kfold(items, group_of, n_splits, seed=seed, min_groups=2)
    ungrouped = leaky_clip_kfold(items, clip_of, n_splits, seed=seed)
    g = [score_fn(tr, va) for tr, va in grouped]
    u = [score_fn(tr, va) for tr, va in ungrouped]
    mean_g = sum(g) / float(len(g))
    mean_u = sum(u) / float(len(u))
    return {
        "grouped_mean": mean_g,
        "ungrouped_mean": mean_u,
        "gap": mean_u - mean_g,
        "grouped_scores": g,
        "ungrouped_scores": u,
        "n_groups": len(_group_members(items, group_of)),
        "n_clips": len(_group_members(items, clip_of)),
    }
