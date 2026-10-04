"""Prompt policy: generic localisation, per-clip restriction, hard negatives.

Three rules, each from a verified finding.

**Never prompt a fine-grained instrument name for localisation.** A
surgical-instrument study using the same detector family found the instrument
name unusable against a large domain gap and fell back to a generic prompt
plus a classifier on the resulting crops. Five of this project's classes have
never been seen as a human-drawn box by any released detector, so asking the
detector to name them is asking it to guess.

**Restrict each clip's prompt set to its own labels.** Cost is linear in
concepts queried because the per-concept decoder pass does not amortise. On a
three-minute clip this is worth roughly 2.5 times, moving the unlabelled pool
from about two working days to one overnight.

**Keep the template pool small and curated.** A large uniform pool measured
worse than a small curated one, and machine-generated attribute descriptors
measured net negative against a fair ensemble.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.registry import GENERIC_LOCALISATION_PROMPTS, Registry

__all__ = [
    "PromptPlan", "build_localisation_prompts", "build_clip_prompt_plan",
    "FineGrainedPromptError",
]


class FineGrainedPromptError(ValueError):
    """A localisation prompt contained a fine-grained class name."""


class PromptPlan(object):
    """What to prompt for one clip, and why."""

    def __init__(self, clip_id, localisation, positives, hard_negatives,
                 positive_classes=(), negative_classes=(), note=""):
        self.clip_id = clip_id
        # Display forms, what the detector is actually prompted with.
        self.localisation = tuple(localisation)
        self.positives = tuple(positives)
        self.hard_negatives = tuple(hard_negatives)
        # Canonical class names for the same sets. Kept separately because
        # the federated bookkeeping and the registry key on canonical names,
        # and silently passing a display form makes the negative set empty,
        # which would quietly disable the guaranteed-false-positive rule.
        self.positive_classes = tuple(positive_classes)
        self.negative_classes = tuple(negative_classes)
        self.note = note

    def queried_classes(self):
        """Canonical names the detector was prompted for, for federated derivation."""
        return self.positive_classes + self.negative_classes

    @property
    def n_concepts(self):
        """Concepts billed by the detector for this clip."""
        return len(self.localisation) + len(self.positives) + len(self.hard_negatives)

    def all_prompts(self):
        return self.localisation + self.positives + self.hard_negatives

    def estimated_seconds(self, keyframes, per_concept_s=0.31, set_image_s=0.03):
        """Cost model: image encoding amortises, the per-concept pass does not."""
        return keyframes * (set_image_s + self.n_concepts * per_concept_s)

    def __repr__(self):
        return ("PromptPlan(clip={!r}, loc={}, pos={}, neg={}, concepts={})"
                .format(self.clip_id, len(self.localisation), len(self.positives),
                        len(self.hard_negatives), self.n_concepts))


def build_localisation_prompts(registry, extra=()):
    """Generic, class-agnostic prompts, checked to contain no class name."""
    prompts = tuple(GENERIC_LOCALISATION_PROMPTS) + tuple(extra)
    canonical = set()
    for name in registry.names:
        canonical.add(name.replace("_", " ").lower())
    for p in prompts:
        low = p.lower()
        for name in sorted(canonical):
            # A generic prompt must not name a specific instrument. Guard
            # against the whole word only, so "glassware" is allowed even
            # though "glass" appears inside other words.
            if name in low.split(" ") or low == name:
                raise FineGrainedPromptError(
                    "localisation prompt {!r} names the class {!r}; localisation "
                    "must stay class-agnostic".format(p, name))
    return prompts


def build_clip_prompt_plan(clip, registry, max_concepts=5, n_hard_negatives=4,
                           use_generic=True, paraphrases_per_class=0):
    """Build the prompt plan for one clip from its own label set.

    ``max_concepts`` is a hard cap because it is a cost control, not a
    preference. When the cap binds, generic prompts are trimmed first, then
    hard negatives, and the clip's own labels are kept last, since dropping a
    labelled class would throw away the supervision entirely.
    """
    labels = sorted(clip.label_set())
    for lab in labels:
        registry.get(lab)  # raises UnknownLabel on a typo in the manifest

    positives = []
    positive_classes = []
    for lab in labels:
        positives.append(lab.replace("_", " "))
        positive_classes.append(lab)
        if paraphrases_per_class:
            entry = registry.get(lab)
            extra = entry.paraphrases[:paraphrases_per_class]
            positives.extend(extra)
            positive_classes.extend([lab] * len(extra))

    negative_classes = list(registry.hard_negatives_for(labels, n=n_hard_negatives))
    negatives = [n.replace("_", " ") for n in negative_classes]
    generic = list(build_localisation_prompts(registry)) if use_generic else []

    note = ""
    budget = max_concepts
    if len(positives) > budget:
        note = ("clip has {} labelled classes, above the {}-concept cap; cap raised "
                "to keep all supervision".format(len(positives), budget))
        budget = len(positives)
    remaining = budget - len(positives)
    n_gen = min(len(generic), max(0, remaining // 2)) if remaining > 0 else 0
    generic = generic[:n_gen]
    remaining -= n_gen
    keep_neg = max(0, remaining)
    negatives = negatives[:keep_neg]
    negative_classes = negative_classes[:keep_neg]

    return PromptPlan(clip.clip_id, generic, positives, negatives,
                      positive_classes=positive_classes,
                      negative_classes=negative_classes, note=note)
