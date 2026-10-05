"""Name generic detections with a crop classifier, under the clip-label constraint.

This closes the gap between the two halves of the recommended architecture.
The detector localises with generic prompts, because fine-grained instrument
names fail as detection prompts: a surgical-instrument study using the same
model family found the instrument name unusable against a large domain gap and
fell back to a generic prompt plus a classifier on the resulting crops. Five of
this project's classes have never been seen as a box by any released detector,
so asking the detector to name them is asking it to guess.

Without this step, generic detections come back labelled with the prompt that
found them, fail the clip-label test, and are discarded. So the generic
localisation pass contributes nothing at all, which is how the pipeline
behaved before this module existed.

Two constraints make it cheap and safe.

**The classifier's vocabulary is restricted to the clip's own labels plus its
hard negatives.** That is the multiple-instance constraint applied to naming
rather than to detection: a crop in this clip can only be something the clip
is labelled with, or something known to be absent. It also bounds cost.

**A crop that cannot be named confidently is rejected, not guessed.** Both an
absolute floor and a margin over the runner-up are required, because the
confusable benchtop siblings are exactly where a forced choice would be wrong
and would look confident.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.types import BoxSource, Detection, RejectReason, Verdict

__all__ = ["NamingResult", "is_generic", "name_detections"]


class NamingResult(object):
    def __init__(self, named, rejected, passthrough, decisions):
        #: Generic detections successfully given a fine-grained label.
        self.named = tuple(named)
        #: Generic detections the classifier could not name confidently.
        self.rejected = tuple(rejected)
        #: Detections that already carried a canonical label.
        self.passthrough = tuple(passthrough)
        #: One record per classified crop, for auditing.
        self.decisions = tuple(decisions)

    @property
    def all_detections(self):
        return self.passthrough + self.named

    def summary(self):
        return {"n_named": len(self.named), "n_rejected": len(self.rejected),
                "n_passthrough": len(self.passthrough)}

    def __repr__(self):
        return "NamingResult({})".format(self.summary())


def is_generic(detection, registry):
    """True when this detection's label is not a canonical class.

    Generic detections are the output of the localisation prompts and are the
    ones needing a name.
    """
    label = str(detection.label)
    if label in registry.names:
        return False
    return registry.resolve(label.replace("_", " ")) is None


def name_detections(detections, clip, registry, crop_classifier, frame_dir=None,
                    min_confidence=0.35, min_margin=0.10, hard_negatives=(),
                    frame_path_fn=None):
    """Assign fine-grained labels to generic detections.

    ``crop_classifier`` is a ``CropClassifier``. Returns a ``NamingResult``.

    Crops are grouped per frame so the classifier runs once per frame rather
    than once per box, which matters because the backbone forward pass is the
    expensive part and the text side is cached.
    """
    vocabulary = sorted(set(clip.label_set()) | set(hard_negatives))
    if not vocabulary:
        raise ValueError("clip {!r} has no labels to name against"
                         .format(clip.clip_id))

    passthrough = []
    generic_by_frame = {}
    for det in detections:
        if is_generic(det, registry):
            generic_by_frame.setdefault(det.frame_id, []).append(det)
        else:
            passthrough.append(det)

    named = []
    rejected = []
    decisions = []
    present = set(clip.label_set())

    for frame_id, group in sorted(generic_by_frame.items()):
        path = frame_path_fn(frame_id) if frame_path_fn else frame_id
        rows = crop_classifier.classify(path, tuple(d.box for d in group),
                                        tuple(vocabulary))
        for det, scores in zip(group, rows):
            ranked = sorted(zip(scores, vocabulary), key=lambda t: (-t[0], t[1]))
            top_score, top_label = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = top_score - runner_up
            decision = {
                "frame_id": frame_id, "prompt": det.prompt,
                "chosen": top_label, "confidence": float(top_score),
                "margin": float(margin),
                "runner_up": ranked[1][1] if len(ranked) > 1 else None,
                "in_clip_labels": top_label in present,
            }
            if top_score < min_confidence or margin < min_margin:
                # Refuse to guess. The confusable siblings are exactly where a
                # forced choice would be wrong and would look confident.
                decision["outcome"] = "rejected"
                decisions.append(decision)
                rejected.append(Verdict(det, RejectReason.AMBIGUOUS_LABEL,
                                        "crop_naming"))
                continue
            decision["outcome"] = "named"
            decisions.append(decision)
            named.append(Detection(
                frame_id=det.frame_id, label=top_label, box=det.box,
                # Combine the detector's localisation confidence with the
                # classifier's naming confidence: both must hold.
                score=float(det.score * top_score),
                box_source=det.box_source,
                # Presence is deliberately dropped rather than inherited. The
                # detector's presence head scored the GENERIC concept that
                # found this box, so after renaming that number refers to a
                # different thing and the presence gate would be judging this
                # label against evidence about another one. There is no
                # presence score for the assigned label, so we say so. The
                # naming confidence is already folded into the score above,
                # and the clip-label constraint still applies.
                presence=None,
                prompt="{} -> {}".format(det.prompt, top_label),
                model_id="{}+{}".format(det.model_id or "detector",
                                        crop_classifier.model_id)))

    return NamingResult(named, rejected, passthrough, decisions)
