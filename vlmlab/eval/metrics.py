"""Average precision in pure Python, as the tested reference.

Deliberately not delegating to a compiled library. Three reasons: the research
found that go/no-go gates at this sample size fire on noise, so the metric has
to be auditable rather than a black box; it must run in an environment with no
package manager; and splitting matching from the curve is a hard requirement,
because the permutation test recomputes the metric hundreds of times and only
the curve half should re-run.

An optional cross-check against a compiled implementation lives in
``crosscheck.py`` and is skipped when that library is absent.

The matching rule, precisely:

1. partition by image, sort predictions by descending score, stable, ties
   broken by insertion order;
2. truncate to ``max_dets`` per image **before** matching, which is where the
   default 300-detection cap silently moves rare-class precision;
3. concatenate and sort globally, stable;
4. greedy one-to-one: each prediction takes the highest-overlap *unmatched*
   ground truth in the same image, a true positive at ``>= threshold``,
   inclusive; a consumed ground truth is never rematched;
5. ignored ground truth still matches, but the prediction is then dropped from
   the curve entirely rather than counted as a false positive;
6. positives count only non-ignored ground truth;
7. unmatched ground truth depresses recall and never appears as an entry;
8. precision is made monotone by a right-to-left running maximum;
9. 101-point interpolation;
10. no positives gives ``None``, never ``0.0``.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.geometry import iou
from vlmlab.types import APResult, MatchTable

__all__ = [
    "GroundTruth", "Prediction", "match_detections", "ap_from_matches",
    "average_precision", "mean_average_precision", "per_class_recall",
]


class GroundTruth(object):
    __slots__ = ("image_id", "label", "box", "ignore")

    def __init__(self, image_id, label, box, ignore=False):
        self.image_id = str(image_id)
        self.label = str(label)
        self.box = box
        self.ignore = bool(ignore)


class Prediction(object):
    __slots__ = ("image_id", "label", "box", "score")

    def __init__(self, image_id, label, box, score):
        self.image_id = str(image_id)
        self.label = str(label)
        self.box = box
        self.score = float(score)


def match_detections(preds, gts, iou_threshold=0.5, max_dets=0,
                     area_range=None):
    """Greedy one-to-one matching. ``max_dets=0`` means unlimited."""
    def _tie_key(pred):
        """Deterministic ordering for equal scores, independent of input order.

        Breaking ties on the input index would make the metric depend on the
        order predictions happened to arrive in, so shuffling the same
        predictions would change the reported number.
        """
        return (pred.image_id, pred.label) + pred.box.as_tuple()

    by_img_pred = {}
    for i, p in enumerate(preds):
        by_img_pred.setdefault(p.image_id, []).append((i, p))

    def _ignored(g):
        if g.ignore:
            return True
        if area_range is not None:
            lo, hi = area_range
            if not (lo <= g.box.area <= hi):
                return True
        return False

    by_img_gt = {}
    n_pos = 0
    for g in gts:
        by_img_gt.setdefault(g.image_id, []).append(g)
        if not _ignored(g):
            n_pos += 1

    # Per-image truncation happens before matching.
    kept = []
    for img in sorted(by_img_pred):
        items = by_img_pred[img]
        items.sort(key=lambda t: (-t[1].score,) + _tie_key(t[1]))
        if max_dets and max_dets > 0:
            items = items[:max_dets]
        kept.extend(items)

    kept.sort(key=lambda t: (-t[1].score,) + _tie_key(t[1]))

    consumed = set()
    scores = []
    is_tp = []
    n_ignored = 0
    for idx, p in kept:
        candidates = by_img_gt.get(p.image_id, ())
        best = None
        best_ov = 0.0
        for gi, g in enumerate(candidates):
            if g.label != p.label:
                continue
            key = (p.image_id, gi)
            if key in consumed:
                continue
            ov = iou(p.box, g.box)
            if ov > best_ov:
                best, best_ov = (gi, g), ov
        if best is not None and best_ov >= iou_threshold:
            gi, g = best
            consumed.add((p.image_id, gi))
            if _ignored(g):
                # Matched something we are told to ignore: drop the prediction
                # from the curve entirely rather than penalising it.
                n_ignored += 1
                continue
            scores.append(p.score)
            is_tp.append(True)
        else:
            scores.append(p.score)
            is_tp.append(False)

    return MatchTable(tuple(scores), tuple(is_tp), n_pos, n_ignored)


def ap_from_matches(table, recall_points=101):
    """Interpolated average precision from a match table. The cheap half."""
    if table.n_pos == 0:
        return APResult(None, 0, table.n_dets, table.n_ignored, recall_points)
    if table.n_dets == 0:
        return APResult(0.0, table.n_pos, 0, table.n_ignored, recall_points)

    tp = 0
    fp = 0
    recalls = []
    precisions = []
    for hit in table.is_tp:
        if hit:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / float(table.n_pos))
        precisions.append(tp / float(tp + fp))

    # Right-to-left running maximum makes precision monotone.
    for i in range(len(precisions) - 2, -1, -1):
        if precisions[i] < precisions[i + 1]:
            precisions[i] = precisions[i + 1]

    total = 0.0
    n = recall_points
    for k in range(n):
        r = k / float(n - 1)
        chosen = 0.0
        for i, rec in enumerate(recalls):
            if rec >= r:
                chosen = precisions[i]
                break
        total += chosen
    return APResult(total / float(n), table.n_pos, table.n_dets,
                    table.n_ignored, recall_points)


def average_precision(preds, gts, iou_threshold=0.5, max_dets=0,
                      area_range=None, recall_points=101):
    table = match_detections(preds, gts, iou_threshold, max_dets, area_range)
    return ap_from_matches(table, recall_points)


def mean_average_precision(preds, gts, labels=None, iou_threshold=0.5,
                           max_dets=0, area_range=None):
    """Macro average, reporting how many classes were actually averaged.

    Classes with no positives are excluded rather than scored zero. Dropping
    terms from a macro average raises its variance, so the count matters as
    much as the value.
    """
    if labels is None:
        labels = sorted({g.label for g in gts} | {p.label for p in preds})
    per_class = {}
    for lab in labels:
        pr = [p for p in preds if p.label == lab]
        gt = [g for g in gts if g.label == lab]
        per_class[lab] = average_precision(pr, gt, iou_threshold, max_dets,
                                           area_range)
    scored = [r.ap for r in per_class.values() if r.ap is not None]
    return {
        "mAP": (sum(scored) / float(len(scored))) if scored else None,
        "n_classes_averaged": len(scored),
        "n_classes_total": len(labels),
        "per_class": per_class,
    }


def per_class_recall(preds, gts, label, score_threshold, iou_threshold=0.5):
    """Recall at a frozen threshold, with an exact binomial interval.

    The research commits to this instead of per-class average precision,
    because per-class average precision is not reportable at any affordable
    annotation budget here.
    """
    from vlmlab.eval.dist import clopper_pearson
    pr = [p for p in preds if p.label == label and p.score >= score_threshold]
    gt = [g for g in gts if g.label == label and not g.ignore]
    table = match_detections(pr, gt, iou_threshold)
    hits = sum(1 for t in table.is_tp if t)
    n = table.n_pos
    lo, hi = clopper_pearson(hits, n) if n else (0.0, 1.0)
    return {"label": label, "recall": (hits / float(n)) if n else None,
            "hits": hits, "n_pos": n, "ci95": (lo, hi),
            "score_threshold": score_threshold}
