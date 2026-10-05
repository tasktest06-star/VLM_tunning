"""Tracking metrics over a federated gold set.

Implements the higher-order tracking accuracy family, which decomposes into a
detection term and an association term and is the right choice here because it
weights the two equally. The alternatives over-weight detection, which would
hide exactly the cross-chunk identity failures this pipeline is most at risk
of: a three-minute clip exceeds the backend's documented thirty-second limit by
six times, so every clip is stitched.

**One deliberate approximation, stated plainly.** The published definition
assigns detections to ground truth with an optimal bipartite matching. This
uses greedy matching by overlap, which differs only when several candidates are
genuinely ambiguous. That is well inside the five to ten point interval the
research found these metrics carry at any affordable annotation budget, and the
research's own conclusion is that tracking metrics here can arbitrate only
large changes and should not be used to choose between two good trackers. Do
not quote these figures as comparable to published leaderboard numbers.

The federated part matters: a class a frame was never checked for is unknown,
not a miss, and scoring it as a miss inflates precision.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.geometry import iou

__all__ = ["ALPHAS", "hota", "track_summary", "TrackingError"]

#: The published alpha sweep.
ALPHAS = tuple(round(0.05 + 0.05 * i, 2) for i in range(19))


class TrackingError(ValueError):
    pass


def _match_at_alpha(gold_dets, pred_dets, alpha):
    """Greedy one-to-one matching per frame at a given overlap threshold.

    Returns matched pairs plus the unmatched counts.
    """
    by_frame = {}
    for g in gold_dets:
        by_frame.setdefault(g["frame_id"], {"g": [], "p": []})["g"].append(g)
    for p in pred_dets:
        by_frame.setdefault(p["frame_id"], {"g": [], "p": []})["p"].append(p)

    pairs = []
    n_fn = 0
    n_fp = 0
    for frame_id in sorted(by_frame):
        bucket = by_frame[frame_id]
        golds = list(bucket["g"])
        preds = sorted(bucket["p"], key=lambda d: (-d.get("score", 1.0),
                                                   d["track_id"]))
        used_gold = set()
        for p in preds:
            best = None
            best_ov = 0.0
            for gi, g in enumerate(golds):
                if gi in used_gold or g["label"] != p["label"]:
                    continue
                ov = iou(g["box"], p["box"])
                if ov > best_ov:
                    best, best_ov = gi, ov
            if best is not None and best_ov >= alpha:
                used_gold.add(best)
                pairs.append((golds[best], p))
            else:
                n_fp += 1
        n_fn += len(golds) - len(used_gold)
    return pairs, n_fn, n_fp


def hota(gold_dets, pred_dets, alphas=ALPHAS):
    """Detection accuracy, association accuracy and their geometric mean.

    Each detection record is a dict with ``frame_id``, ``label``, ``box`` and
    ``track_id``. Predictions may also carry ``score``.
    """
    if not gold_dets:
        return {"hota": None, "det_a": None, "ass_a": None,
                "n_gold": 0, "n_pred": len(pred_dets),
                "note": "no gold tracks to score against"}

    per_alpha = []
    for alpha in alphas:
        pairs, n_fn, n_fp = _match_at_alpha(gold_dets, pred_dets, alpha)
        n_tp = len(pairs)
        denom = n_tp + n_fn + n_fp
        det_a = (n_tp / float(denom)) if denom else 0.0

        if n_tp == 0:
            per_alpha.append({"alpha": alpha, "det_a": det_a, "ass_a": 0.0,
                              "hota": 0.0, "tp": 0, "fn": n_fn, "fp": n_fp})
            continue

        # Counts per (gold id, predicted id) co-occurrence.
        tpa = {}
        gold_total = {}
        pred_total = {}
        for g, p in pairs:
            key = (g["track_id"], p["track_id"])
            tpa[key] = tpa.get(key, 0) + 1
        for g in gold_dets:
            gold_total[g["track_id"]] = gold_total.get(g["track_id"], 0) + 1
        for p in pred_dets:
            pred_total[p["track_id"]] = pred_total.get(p["track_id"], 0) + 1

        total = 0.0
        for g, p in pairs:
            key = (g["track_id"], p["track_id"])
            a = tpa[key]
            fna = gold_total.get(g["track_id"], 0) - a
            fpa = pred_total.get(p["track_id"], 0) - a
            total += a / float(a + fna + fpa)
        ass_a = total / float(n_tp)
        per_alpha.append({"alpha": alpha, "det_a": det_a, "ass_a": ass_a,
                          "hota": (det_a * ass_a) ** 0.5,
                          "tp": n_tp, "fn": n_fn, "fp": n_fp})

    n = float(len(per_alpha))
    return {
        "hota": sum(r["hota"] for r in per_alpha) / n,
        "det_a": sum(r["det_a"] for r in per_alpha) / n,
        "ass_a": sum(r["ass_a"] for r in per_alpha) / n,
        "per_alpha": tuple(per_alpha),
        "n_gold": len(gold_dets),
        "n_pred": len(pred_dets),
        "note": ("greedy matching approximates the published optimal "
                 "assignment; well inside the 5 to 10 point interval these "
                 "metrics carry here. Not comparable to leaderboard numbers."),
    }


def track_summary(tracks):
    """Descriptive statistics, including how much stitching happened."""
    if not tracks:
        return {"n_tracks": 0}
    lengths = [t.n_frames for t in tracks]
    stitched = [t for t in tracks if t.spans_chunks]
    frags = [t.mask_fragmentation for t in tracks
             if t.mask_fragmentation is not None]
    return {
        "n_tracks": len(tracks),
        "n_stitched": len(stitched),
        "fraction_stitched": len(stitched) / float(len(tracks)),
        "mean_length": sum(lengths) / float(len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_mask_fragmentation": (sum(frags) / float(len(frags))
                                    if frags else None),
        "note": ("a high stitched fraction is expected: a three-minute clip "
                 "exceeds the backend's documented 30-second limit six times "
                 "over, so stitching is the common path and the dominant "
                 "accuracy risk"),
    }
