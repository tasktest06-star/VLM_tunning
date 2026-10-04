"""Temporal-consistency and cross-model-agreement filters.

Both exploit structure that costs nothing to obtain.

**Temporal consistency.** A real instrument sitting on a bench persists across
frames. A spurious detection flickers. So a box appearing in only one frame of
a nearby run is likely a false positive.

**Cross-model agreement.** Two detectors with genuinely different training
mixes provide independent evidence. One model cannot detect its own blind
spot, and a teacher agreeing with a student trained on that teacher's own
output is not evidence at all, which is the confirmation-bias trap the
research found in a sibling pipeline.

A note on scope: the pseudo-box precision gain from requiring two detectors to
agree is **unmeasured in the literature**. It is a half-day experiment and the
result should decide whether this stage earns its compute.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.geometry import iou
from vlmlab.types import Detection, RejectReason, Verdict

__all__ = [
    "temporal_consistency", "cross_model_agreement", "AgreementReport",
    "track_stability",
]


def _frame_order(detections):
    """Stable frame ordering by identifier, with a numeric suffix if present."""
    def key(fid):
        digits = "".join(ch for ch in fid if ch.isdigit())
        return (int(digits) if digits else 0, fid)
    return sorted({d.frame_id for d in detections}, key=key)


def temporal_consistency(detections, min_appearances=2, iou_threshold=0.5,
                         window=3):
    """Keep detections corroborated by a nearby frame.

    A detection survives when at least ``min_appearances`` frames inside a
    sliding window contain an overlapping box of the same label, counting
    itself. Returns verdicts so the drop is attributable.
    """
    if min_appearances < 1:
        raise ValueError("min_appearances must be at least 1")
    frames = _frame_order(detections)
    pos = {fid: i for i, fid in enumerate(frames)}
    by_frame = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)

    verdicts = []
    for d in detections:
        i = pos[d.frame_id]
        count = 0
        lo = max(0, i - window)
        hi = min(len(frames), i + window + 1)
        for j in range(lo, hi):
            for other in by_frame.get(frames[j], ()):
                if other.label != d.label:
                    continue
                if j == i and other is d:
                    count += 1
                    break
                if iou(d.box, other.box) >= iou_threshold:
                    count += 1
                    break
        if count >= min_appearances:
            verdicts.append(Verdict(d, RejectReason.KEPT, "temporal"))
        else:
            verdicts.append(Verdict(d, RejectReason.TEMPORALLY_INCONSISTENT, "temporal"))
    return tuple(verdicts)


class AgreementReport(object):
    """Result of comparing two detectors on the same frames."""

    def __init__(self, agreed, only_a, only_b, class_conflicts):
        self.agreed = tuple(agreed)
        self.only_a = tuple(only_a)
        self.only_b = tuple(only_b)
        self.class_conflicts = tuple(class_conflicts)

    @property
    def precision_proxy(self):
        """Fraction of model A's detections corroborated by model B.

        A proxy, not a measurement: it cannot distinguish a shared blind spot
        from genuine agreement.
        """
        total = len(self.agreed) + len(self.only_a)
        if total == 0:
            return None
        return len(self.agreed) / float(total)

    def __repr__(self):
        return ("AgreementReport(agreed={}, only_a={}, only_b={}, conflicts={})"
                .format(len(self.agreed), len(self.only_a), len(self.only_b),
                        len(self.class_conflicts)))


def cross_model_agreement(dets_a, dets_b, iou_threshold=0.5):
    """Require both class match and overlap.

    A high-overlap disagreement about the *class* is reported as a conflict
    rather than silently resolved to the higher score. Those are exactly the
    confusable-sibling cases, and quietly picking a winner would hide the one
    error mode this project most needs to see.
    """
    by_frame_b = {}
    for d in dets_b:
        by_frame_b.setdefault(d.frame_id, []).append(d)

    agreed, only_a, conflicts = [], [], []
    matched_b = set()
    for a in dets_a:
        best = None
        best_iou = 0.0
        conflict = None
        for b in by_frame_b.get(a.frame_id, ()):
            ov = iou(a.box, b.box)
            if ov < iou_threshold:
                continue
            if b.label == a.label:
                if ov > best_iou:
                    best, best_iou = b, ov
            elif conflict is None or ov > conflict[1]:
                conflict = (b, ov)
        if best is not None:
            agreed.append((a, best, best_iou))
            matched_b.add(id(best))
        else:
            only_a.append(a)
            if conflict is not None:
                conflicts.append((a, conflict[0], conflict[1]))

    only_b = [b for b in dets_b if id(b) not in matched_b]
    return AgreementReport(agreed, only_a, only_b, conflicts)


def track_stability(track):
    """Label-free quality proxy: how steady a track's box is.

    Returns the mean overlap between consecutive boxes, in zero to one.
    Higher is steadier. Useful on the unlabelled pool where no ground truth
    exists, but note that no published evaluation establishes that this
    correlates with true accuracy, so treat it as a monitoring signal rather
    than a metric.
    """
    pts = track.points
    if len(pts) < 2:
        return None
    overlaps = [iou(pts[i].box, pts[i + 1].box) for i in range(len(pts) - 1)]
    return sum(overlaps) / float(len(overlaps))
