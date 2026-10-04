"""Federated label bookkeeping, shared by the exporter and the metrics.

This module exists so that the dataset export and the evaluation cannot
disagree about which classes were negative on a clip. If only the exporter
owned this logic, the reported average precision and the tracking metrics
would silently diverge.

The clip's multi-label set is a genuine three-way signal:

* classes **in** the label set are present somewhere in the clip;
* classes that were **queried but absent** from the label set are negative,
  and every detection of them is a guaranteed false positive, so they become
  free background boxes;
* classes **never queried** are unknown, and must be ignored rather than
  counted as a miss.

Conflating the third with the second is what inflates federated precision.
"""
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = ["FederatedLabels", "derive", "FederatedError"]


class FederatedError(ValueError):
    pass


class FederatedLabels(object):
    def __init__(self, clip_id, positives, negatives, not_exhaustive):
        self.clip_id = clip_id
        self.positives = frozenset(positives)
        self.negatives = frozenset(negatives)
        self.not_exhaustive = frozenset(not_exhaustive)
        overlap = self.positives & self.negatives
        if overlap:
            raise FederatedError(
                "clip {!r}: {} are both positive and negative"
                .format(clip_id, sorted(overlap)))
        if self.positives & self.not_exhaustive:
            raise FederatedError(
                "clip {!r}: a labelled class cannot be unknown".format(clip_id))

    @property
    def queried(self):
        return self.positives | self.negatives

    def status(self, label):
        if label in self.positives:
            return "positive"
        if label in self.negatives:
            return "negative"
        return "unknown"

    def is_guaranteed_false_positive(self, label):
        """True when a detection of this label must be wrong on this clip."""
        return label in self.negatives

    def to_tao_fields(self):
        """The fields a federated tracking evaluator consumes.

        Emitting these is what makes an unqueried class ignored rather than
        counted against you.
        """
        return {
            "neg_category_ids": sorted(self.negatives),
            "not_exhaustive_category_ids": sorted(self.not_exhaustive),
        }

    def __repr__(self):
        return ("FederatedLabels(clip={!r}, pos={}, neg={}, unknown={})"
                .format(self.clip_id, len(self.positives), len(self.negatives),
                        len(self.not_exhaustive)))


def derive(clip, queried_labels, vocabulary):
    """Split a vocabulary into positive, negative and unknown for one clip.

    ``queried_labels`` is what the detector was actually prompted with. Only
    those can be asserted negative, because absence of a prompt is not
    evidence of absence of an object.
    """
    vocab = set(vocabulary)
    present = set(clip.label_set())
    unknown_present = present - vocab
    if unknown_present:
        raise FederatedError(
            "clip {!r} is labelled with {} which are outside the vocabulary"
            .format(clip.clip_id, sorted(unknown_present)))
    queried = set(queried_labels) & vocab
    negatives = queried - present
    not_exhaustive = vocab - queried - present
    return FederatedLabels(clip.clip_id, present, negatives, not_exhaustive)
