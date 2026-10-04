"""Clustered statistics, in pure Python.

The research's central warning is that at this sample size most comparisons
are unresolvable, and that the binding constraint is the number of recording
sessions rather than the number of clips. This module is what makes that
visible instead of hoping it is not true.

Three design points are load-bearing.

**The bootstrap resamples units and recomputes on pooled records.** It does
not average per-unit metrics. Average precision is a non-linear functional of
a ranking, so the mean of per-unit values is a different quantity and would
give a confidently wrong interval.

**Below about fifteen groups, use the permutation test.** A bootstrap is
resampling too few exchangeable units to be informative there. The exact
sign-flip test is also honest about its floor: with nine groups it cannot
report a p-value below about 0.004, so demanding less is unsatisfiable.

**Two different design effects are reported, never conflated.** The Kish form
captures unequal cluster sizes and needs no model. The
correlation form captures clustering itself and is the one that produces the
roughly 2.5 figure the research used. Treating them as interchangeable is a
real error.
"""
import itertools
import math
import random
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.eval.dist import normal_ppf, t_ppf
from vlmlab.types import CIResult, MDEResult, PermResult

__all__ = [
    "bootstrap_ci", "paired_group_permutation_test", "kish_design_effect",
    "effective_sample_size", "icc_one_way", "design_effect_from_icc",
    "mde_paired", "post_hoc_power", "choose_test", "spread_vs_depth",
]

_EXACT_LIMIT = 14  # 2**14 = 16384 sign-flip vectors


def bootstrap_ci(units, metric, n_resamples=2000, alpha=0.05, seed=0):
    """Cluster bootstrap percentile interval.

    ``units`` is a sequence of record-sequences, one per cluster, typically
    one per recording session. ``metric`` maps a flat record sequence to a
    scalar or None.
    """
    units = [tuple(u) for u in units]
    n = len(units)
    if n == 0:
        return CIResult(None, None, None, 0, 0, 0, "percentile")
    pooled = [r for u in units for r in u]
    point = metric(pooled)

    rng = random.Random(seed)
    reps = []
    degenerate = 0
    for _ in range(n_resamples):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        flat = [r for u in sample for r in u]
        val = metric(flat)
        if val is None:
            degenerate += 1
            continue
        reps.append(val)

    if not reps:
        return CIResult(point, None, None, n, n_resamples, degenerate, "percentile")

    reps.sort()
    m = len(reps)
    # (m + 1) order-statistic convention.
    lo_idx = max(0, min(m - 1, int(math.floor((m + 1) * (alpha / 2.0))) - 1))
    hi_idx = max(0, min(m - 1, int(math.ceil((m + 1) * (1.0 - alpha / 2.0))) - 1))
    return CIResult(point, reps[lo_idx], reps[hi_idx], n, n_resamples,
                    degenerate, "percentile")


def paired_group_permutation_test(units_a, units_b, metric, n_perm=None,
                                  seed=0):
    """Group-level sign-flip test for a paired comparison of two pipelines.

    ``units_a`` and ``units_b`` are aligned sequences of record-sequences: the
    same groups scored under two systems. Each permutation swaps a subset of
    groups between the systems and recomputes both pooled metrics.
    """
    if len(units_a) != len(units_b):
        raise ValueError("paired test needs the same groups on both sides")
    n = len(units_a)
    if n == 0:
        return PermResult(None, 1.0, 0, True, 0)

    a = [tuple(u) for u in units_a]
    b = [tuple(u) for u in units_b]

    def pooled_diff(flips):
        fa, fb = [], []
        for i, flip in enumerate(flips):
            if flip:
                fa.extend(b[i])
                fb.extend(a[i])
            else:
                fa.extend(a[i])
                fb.extend(b[i])
        va, vb = metric(fa), metric(fb)
        if va is None or vb is None:
            return None
        return va - vb

    observed = pooled_diff([False] * n)
    if observed is None:
        return PermResult(None, 1.0, 0, True, n)

    exact = n <= _EXACT_LIMIT
    nulls = []
    if exact:
        for combo in itertools.product((False, True), repeat=n):
            d = pooled_diff(list(combo))
            if d is not None:
                nulls.append(d)
        used = len(nulls)
    else:
        rng = random.Random(seed)
        used = int(n_perm or 10000)
        for _ in range(used):
            d = pooled_diff([rng.random() < 0.5 for _ in range(n)])
            if d is not None:
                nulls.append(d)

    extreme = sum(1 for d in nulls if abs(d) >= abs(observed))
    p = (1.0 + extreme) / (1.0 + len(nulls))
    return PermResult(observed, p, len(nulls), exact, n)


def kish_design_effect(cluster_sizes):
    """Unequal-weighting design effect. No model, just the sizes.

    Equals exactly 1.0 when every cluster is the same size, whatever that size
    is, because there is then no unequal weighting to penalise. The effective
    count is in units of clusters, so it must be compared against the cluster
    count rather than the record count.
    """
    sizes = [float(s) for s in cluster_sizes if s > 0]
    if not sizes:
        return 1.0
    k = float(len(sizes))
    total = sum(sizes)
    sq = sum(s * s for s in sizes)
    if sq <= 0:
        return 1.0
    n_eff_clusters = (total * total) / sq
    if n_eff_clusters <= 0:
        return 1.0
    return k / n_eff_clusters


def effective_sample_size(cluster_sizes):
    sizes = [float(s) for s in cluster_sizes if s > 0]
    if not sizes:
        return 0.0
    total = sum(sizes)
    return (total * total) / sum(s * s for s in sizes)


def icc_one_way(values_by_cluster):
    """Intra-cluster correlation by one-way random-effects analysis of variance.

    This is the quantity that actually drives the clustering penalty, and the
    research says to measure it on the first twenty clips rather than assume
    it.
    """
    groups = [list(v) for v in values_by_cluster if len(v) > 0]
    k = len(groups)
    if k < 2:
        return None
    n_total = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / float(n_total)
    ms_between = sum(len(g) * (sum(g) / float(len(g)) - grand) ** 2
                     for g in groups) / float(k - 1)
    within_df = n_total - k
    if within_df <= 0:
        return None
    ms_within = sum((x - sum(g) / float(len(g))) ** 2
                    for g in groups for x in g) / float(within_df)
    m_bar = n_total / float(k)
    denom = ms_between + (m_bar - 1.0) * ms_within
    if denom <= 0:
        return 0.0
    icc = (ms_between - ms_within) / denom
    return max(0.0, min(1.0, icc))


def design_effect_from_icc(mean_cluster_size, icc):
    """The clustering design effect, one plus size-minus-one times correlation."""
    if icc is None:
        return None
    return 1.0 + (float(mean_cluster_size) - 1.0) * float(icc)


def mde_paired(sd_diff, n_units, power=0.80, alpha=0.05, n_comparisons=1,
               two_sided=True):
    """Minimum detectable difference for a paired comparison.

    ``n_comparisons`` applies a Bonferroni correction, which is what turns the
    research's roughly 3.5 to 4 point figure into 4.5 to 5 once a realistic
    number of pipeline variants is being compared.
    """
    if n_units < 2:
        raise ValueError("need at least 2 units")
    if sd_diff < 0:
        raise ValueError("sd_diff must be non-negative")
    df = n_units - 1
    a = alpha / float(n_comparisons)
    t_alpha = t_ppf(1.0 - a / 2.0, df) if two_sided else t_ppf(1.0 - a, df)
    t_beta = t_ppf(power, df)
    mde = (t_alpha + t_beta) * sd_diff / math.sqrt(n_units)
    return MDEResult(mde, n_units, sd_diff, power, alpha, n_comparisons)


def post_hoc_power(true_diff, sd_diff, n_units, alpha=0.05, n_comparisons=1):
    """Probability of detecting a difference of the stated size."""
    if sd_diff <= 0:
        return 1.0
    df = n_units - 1
    a = alpha / float(n_comparisons)
    crit = t_ppf(1.0 - a / 2.0, df)
    ncp = abs(true_diff) * math.sqrt(n_units) / sd_diff
    return max(0.0, min(1.0, 1.0 - _approx_t_cdf(crit - ncp, df)))


def _approx_t_cdf(x, df):
    from vlmlab.eval.dist import t_cdf
    return t_cdf(x, df)


def choose_test(n_groups, permutation_below=15):
    """Which test to use, by group count. The research's own rule."""
    if n_groups < permutation_below:
        return "permutation", (
            "with {} groups a bootstrap is resampling too few exchangeable units "
            "to be informative; use the exact sign-flip test and note its p-value "
            "floor of about {:.4f}".format(n_groups, 1.0 / (2 ** n_groups + 1)))
    return "bootstrap", "{} groups is enough for a cluster bootstrap".format(n_groups)


def spread_vs_depth(n_clips, frames_per_clip, icc):
    """Effective sample size for a fixed annotation budget.

    Makes concrete the finding that spreading frames across many clips beats
    densely annotating a few, which is the opposite of the natural instinct.
    """
    total = n_clips * frames_per_clip
    deff = design_effect_from_icc(frames_per_clip, icc)
    if deff is None or deff <= 0:
        return {"total_frames": total, "n_eff": float(total), "deff": 1.0}
    return {"total_frames": total, "n_eff": total / deff, "deff": deff,
            "n_clips": n_clips, "frames_per_clip": frames_per_clip}
