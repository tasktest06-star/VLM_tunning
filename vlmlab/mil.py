"""The clip label as a constraint: presence gate and accept/reject cascade.

This is where weak supervision does real work. A clip labelled with two
instruments guarantees both are present somewhere, and is strong evidence
every other queried class is absent. That turns one label per clip into a
per-detection accept or reject rule with no extra annotation.

Two design choices are deliberate.

**The gate is a maximum over frames.** That is the standard
multiple-instance assumption and it matches the task exactly: a positive clip
need only contain the object in one frame, while a negative clip must contain
it in none.

**Every rejection carries a reason code.** Without it the cascade cannot be
tested stage by stage, and a silent drop is indistinguishable from a bug.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.geometry import area_ratio, min_side, nms_per_class
from vlmlab.types import Detection, RejectReason, Verdict

__all__ = [
    "presence_score", "clip_passes_presence", "CascadeResult",
    "run_cascade", "background_boxes", "cascade_summary",
]


def presence_score(detections, label):
    """Maximum combined score across frames for one label.

    Uses the decoupled presence head where the backend provides one, since it
    answers the presence question directly rather than inferring it from a
    localisation score.
    """
    best = 0.0
    for d in detections:
        if d.label != label:
            continue
        s = d.combined_score()
        if s > best:
            best = s
    return best


def clip_passes_presence(detections, label, threshold):
    """The multiple-instance test for one label on one clip."""
    return presence_score(detections, label) >= threshold


class CascadeResult(object):
    def __init__(self, verdicts, missing_labels):
        self.verdicts = tuple(verdicts)
        self.missing_labels = tuple(missing_labels)

    @property
    def accepted(self):
        return tuple(v.detection for v in self.verdicts
                     if v.reason == RejectReason.KEPT)

    @property
    def background(self):
        """Guaranteed false positives, kept deliberately as background boxes."""
        return tuple(v.detection for v in self.verdicts
                     if v.reason == RejectReason.KEPT_AS_BACKGROUND)

    @property
    def rejected(self):
        return tuple(v for v in self.verdicts if not v.accepted)

    def reasons(self):
        counts = {}
        for v in self.verdicts:
            counts[v.reason.value] = counts.get(v.reason.value, 0) + 1
        return counts

    def __repr__(self):
        return ("CascadeResult(accepted={}, background={}, rejected={}, missing={})"
                .format(len(self.accepted), len(self.background),
                        len(self.rejected), list(self.missing_labels)))


def run_cascade(detections, clip, federated_labels, cfg, image_size=None):
    """Filter detections against the clip's label set and the quality gates.

    Order matters. The clip-label test comes first and is absolute: a
    detection of a class the clip is not labelled with cannot be a true
    positive no matter how confident the detector is. Score and geometry
    gates follow, then suppression.

    ``cfg`` is a ``CascadeConfig``. ``image_size`` is ``(width, height)`` and
    may be omitted to skip the area gates.
    """
    verdicts = []
    survivors = []

    for d in detections:
        status = federated_labels.status(d.label)

        if status == "negative":
            # Queried, absent from the clip's labels, so this is guaranteed
            # wrong. Keep it as an explicit background box rather than
            # discarding it: it is free, correctly labelled negative data.
            verdicts.append(Verdict(d, RejectReason.KEPT_AS_BACKGROUND, "clip_labels"))
            continue
        if status == "unknown":
            verdicts.append(Verdict(d, RejectReason.NOT_IN_CLIP_LABELS, "clip_labels"))
            continue

        if d.score < cfg.score_threshold:
            verdicts.append(Verdict(d, RejectReason.BELOW_SCORE, "score"))
            continue

        if d.presence is not None and d.presence < cfg.presence_threshold:
            verdicts.append(Verdict(d, RejectReason.BELOW_PRESENCE, "presence"))
            continue

        if min_side(d.box) < cfg.min_side_px:
            verdicts.append(Verdict(d, RejectReason.SIDE_TOO_SMALL, "geometry"))
            continue

        if image_size is not None:
            ratio = area_ratio(d.box, image_size[0], image_size[1])
            if ratio < cfg.min_area_ratio or ratio > cfg.max_area_ratio:
                verdicts.append(Verdict(d, RejectReason.AREA_OUT_OF_RANGE, "geometry"))
                continue

        survivors.append(d)

    # Suppression, per class, among survivors on the same frame.
    by_frame = {}
    for d in survivors:
        by_frame.setdefault(d.frame_id, []).append(d)
    kept_ids = set()
    for frame_id, group in by_frame.items():
        keep = nms_per_class([g.box for g in group], [g.score for g in group],
                             [g.label for g in group], cfg.nms_iou)
        for i, g in enumerate(group):
            if i in keep:
                kept_ids.add(id(g))
    for d in survivors:
        if id(d) in kept_ids:
            verdicts.append(Verdict(d, RejectReason.KEPT, "nms"))
        else:
            verdicts.append(Verdict(d, RejectReason.SUPPRESSED_NMS, "nms"))

    # Which labelled classes produced nothing at all? Those clips go to the
    # human one-box queue, which is the insurance policy for when text
    # prompts collapse on a class no detector has seen as a box.
    found = set(v.detection.label for v in verdicts if v.reason == RejectReason.KEPT)
    missing = sorted(set(clip.label_set()) - found)

    return CascadeResult(verdicts, missing)


def background_boxes(result):
    """Guaranteed false positives, ready to export as negative training data."""
    return result.background


def cascade_summary(results):
    """Aggregate reason counts across clips, for the monitoring dashboard."""
    total = {}
    missing = {}
    for r in results:
        for reason, n in r.reasons().items():
            total[reason] = total.get(reason, 0) + n
        for lab in r.missing_labels:
            missing[lab] = missing.get(lab, 0) + 1
    return {"reasons": total, "labels_never_found": missing}
