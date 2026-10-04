"""Class registry: names, paraphrases, hard negatives, and the homonym table.

The homonym table is the most load-bearing data in the project. Verified by
grepping the actual dataset category files:

* ``fume_hood`` in LVIS is category 565, a **frequent** class, defined as a
  covering that exhausts fumes, synonym ``exhaust_hood``. It is a kitchen
  range hood. That is the worst case, because a detector is confidently
  trained on the wrong object under exactly the name you would prompt with.
* ``shaker`` in LVIS is a container something is shaken from, so a condiment
  shaker, not an orbital shaker.
* ``microscope`` exists but is rare, under ten images, so effectively
  untrained.

Confirmed absent from Open Images, LVIS and Objects365 entirely: centrifuge,
autoclave and spectrophotometer.
"""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ClassEntry", "Registry", "AmbiguousMatch", "UnknownLabel",
    "levenshtein", "DEFAULT_REGISTRY", "GENERIC_LOCALISATION_PROMPTS",
]


class AmbiguousMatch(Exception):
    """Two classes are equally good matches, so resolution would be arbitrary."""


class UnknownLabel(Exception):
    pass


def levenshtein(a, b):
    """Edit distance with a rolling row. Linear in memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


# Generic prompts for class-agnostic localisation.
#
# This is the single most important architectural finding. Fine-grained
# instrument names fail as detection prompts: a surgical-instrument study
# found the instrument name unusable against a large domain gap and fell back
# to a generic prompt plus a crop classifier. So the detector is asked only to
# find objects, never to name them.
GENERIC_LOCALISATION_PROMPTS = (
    "laboratory instrument",
    "machine on a laboratory bench",
    "glassware",
    "handheld laboratory tool",
    "scientific equipment",
)


@dataclass(frozen=True)
class ClassEntry:
    name: str
    paraphrases: Tuple[str, ...] = ()
    confusable_with: Tuple[str, ...] = ()
    forbidden_substrings: Tuple[str, ...] = ()
    has_public_boxes: bool = True
    note: str = ""


# Placeholder vocabulary: glassware and benchtop instruments mixed, which is
# the realistic case and also where the confusable-sibling problem bites
# hardest. Edit this, or load your own from JSON. Nothing in the code
# hardcodes these names.
DEFAULT_REGISTRY = (
    # --- glassware: good public box coverage, about 22k CC BY 4.0 instances ---
    ClassEntry("beaker", ("glass beaker", "laboratory beaker"),
               confusable_with=("measuring_cylinder", "conical_flask"),
               forbidden_substrings=("coffee", "mug"),
               note="present in Open Images with real boxes"),
    ClassEntry("conical_flask", ("erlenmeyer flask", "conical laboratory flask"),
               confusable_with=("beaker", "round_bottom_flask", "volumetric_flask"),
               forbidden_substrings=("thermos", "vacuum flask", "hip flask"),
               note="Objects365 'Flask' is often a thermos"),
    ClassEntry("round_bottom_flask", ("round bottomed flask", "boiling flask"),
               confusable_with=("conical_flask", "volumetric_flask")),
    ClassEntry("volumetric_flask", ("graduated flask",),
               confusable_with=("conical_flask", "round_bottom_flask")),
    ClassEntry("measuring_cylinder", ("graduated cylinder", "measuring cylinder"),
               confusable_with=("beaker", "test_tube")),
    ClassEntry("test_tube", ("laboratory test tube",),
               confusable_with=("test_tube_holder", "measuring_cylinder"),
               forbidden_substrings=("holder", "rack", "stand"),
               note="the substring trap: 'test tube holder' must not resolve here"),
    ClassEntry("test_tube_holder", ("test tube rack", "test tube stand"),
               confusable_with=("test_tube",)),
    ClassEntry("pipette", ("volumetric pipette", "laboratory pipette"),
               confusable_with=("burette",),
               has_public_boxes=False,
               note="boxes exist only inside the chemistry dataset"),
    # --- benchtop instruments: five have no public boxes anywhere ---
    ClassEntry("microscope", ("optical microscope", "light microscope"),
               confusable_with=("spectrophotometer",),
               note="in LVIS but rare, under ten images, so effectively untrained"),
    ClassEntry("centrifuge", ("benchtop centrifuge", "laboratory centrifuge"),
               confusable_with=("orbital_shaker", "incubator"),
               has_public_boxes=False,
               note="confirmed absent from Open Images, LVIS and Objects365"),
    ClassEntry("autoclave", ("laboratory autoclave", "steam steriliser"),
               confusable_with=("incubator", "oven"),
               has_public_boxes=False,
               note="confirmed absent from all three major box datasets"),
    ClassEntry("spectrophotometer", ("UV-Vis spectrophotometer",
                                     "benchtop instrument with a sample drawer"),
               confusable_with=("analytical_balance", "microscope"),
               has_public_boxes=False,
               note="confirmed absent; ImageNet-21k has 628 images, research use only"),
    ClassEntry("orbital_shaker", ("laboratory shaker", "platform shaker"),
               confusable_with=("centrifuge", "incubator"),
               forbidden_substrings=("condiment", "salt", "pepper", "cocktail"),
               has_public_boxes=False,
               note="LVIS 'shaker' is a condiment shaker"),
    ClassEntry("fume_hood", ("laboratory fume cupboard", "fume extraction hood"),
               confusable_with=("autoclave",),
               forbidden_substrings=("range", "kitchen", "exhaust", "cooker", "stove"),
               has_public_boxes=False,
               note="LVIS category 565 fume_hood is a KITCHEN RANGE HOOD, and it is "
                    "a frequent class, so the detector is confidently wrong"),
    ClassEntry("analytical_balance", ("laboratory balance", "precision scale"),
               confusable_with=("spectrophotometer",),
               forbidden_substrings=("bathroom", "kitchen", "weighing scale"),
               has_public_boxes=False,
               note="LVIS 'scale' is dominated by bathroom and kitchen scales"),
)


class Registry:
    """Resolves free-text labels onto the canonical vocabulary."""

    def __init__(self, entries=None, max_edit_distance=2, min_substring_len=4):
        self.entries = tuple(entries) if entries is not None else DEFAULT_REGISTRY
        self.max_edit_distance = max_edit_distance
        self.min_substring_len = min_substring_len
        self._by_name = {e.name: e for e in self.entries}
        if len(self._by_name) != len(self.entries):
            raise ValueError("duplicate class name in registry")
        self._alias = {}
        for e in self.entries:
            for alias in (e.name,) + tuple(e.paraphrases):
                key = self._norm(alias)
                if key in self._alias and self._alias[key] != e.name:
                    raise ValueError("alias {!r} maps to two classes".format(alias))
                self._alias[key] = e.name

    # ---- basics -------------------------------------------------------
    @staticmethod
    def _norm(s):
        return " ".join(str(s).lower().replace("_", " ").split())

    @property
    def names(self):
        return tuple(e.name for e in self.entries)

    def get(self, name):
        if name not in self._by_name:
            raise UnknownLabel(name)
        return self._by_name[name]

    def classes_without_public_boxes(self):
        return tuple(e.name for e in self.entries if not e.has_public_boxes)

    # ---- resolution ---------------------------------------------------
    def resolve(self, raw):
        """Three stages: exact alias, then nearest edit distance, then guarded substring.

        Differs from the obvious implementation in two ways that matter.
        Stage two takes the **nearest** candidate rather than the first one
        within threshold, so resolution does not depend on registry order.
        Stage three refuses a known false friend, which is where the homonym
        trap lives.

        Returns the canonical name, or None when nothing matches.
        """
        key = self._norm(raw)
        if not key:
            return None
        if key in self._alias:
            return self._alias[key]

        # Stage 2: nearest alias by edit distance, argmin not first-match.
        scored = []
        for alias, canon in self._alias.items():
            d = levenshtein(key, alias)
            if d <= self.max_edit_distance:
                scored.append((d, canon))
        if scored:
            best = min(d for d, _ in scored)
            winners = sorted({canon for d, canon in scored if d == best})
            if len(winners) > 1:
                raise AmbiguousMatch(
                    "{!r} is equidistant ({}) from {}".format(raw, best, winners))
            return winners[0]

        # Stage 3: substring, but only where it is not a known false friend.
        candidates = []
        for e in self.entries:
            if any(bad in key for bad in (self._norm(f) for f in e.forbidden_substrings)):
                continue
            for alias in (e.name,) + tuple(e.paraphrases):
                na = self._norm(alias)
                # The length floor applies to the *class* side, so a short
                # incoming fragment cannot pull in a long class name.
                if len(na) < self.min_substring_len:
                    continue
                if na in key:
                    candidates.append((len(na), e.name))
                    break
        if candidates:
            longest = max(n for n, _ in candidates)
            winners = sorted({name for n, name in candidates if n == longest})
            if len(winners) > 1:
                raise AmbiguousMatch("{!r} matches {} equally".format(raw, winners))
            return winners[0]
        return None

    # ---- hard negatives ------------------------------------------------
    def hard_negatives_for(self, present_labels, n=4):
        """Confusable classes that are *absent* from this clip's label set.

        Every detection one of these produces is a guaranteed false positive
        and therefore a free background box.

        Deterministic: ordered by how many present labels name each as
        confusable, then alphabetically.
        """
        present = set(present_labels)
        for lab in present:
            if lab not in self._by_name:
                raise UnknownLabel(lab)
        votes = {}
        for lab in present:
            for cand in self._by_name[lab].confusable_with:
                if cand in present or cand not in self._by_name:
                    continue
                votes[cand] = votes.get(cand, 0) + 1
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        out = [name for name, _ in ranked][:n]
        if len(out) < n:
            # Top up with any absent class, alphabetically, for determinism.
            for e in self.entries:
                if len(out) >= n:
                    break
                if e.name not in present and e.name not in out:
                    out.append(e.name)
        return tuple(out[:n])

    # ---- serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, data, **kw):
        entries = []
        for item in data.get("classes", []):
            entries.append(ClassEntry(
                name=item["name"],
                paraphrases=tuple(item.get("paraphrases", ())),
                confusable_with=tuple(item.get("confusable_with", ())),
                forbidden_substrings=tuple(item.get("forbidden_substrings", ())),
                has_public_boxes=bool(item.get("has_public_boxes", True)),
                note=item.get("note", ""),
            ))
        return cls(entries or None, **kw)

    def to_dict(self):
        return {"classes": [
            {"name": e.name, "paraphrases": list(e.paraphrases),
             "confusable_with": list(e.confusable_with),
             "forbidden_substrings": list(e.forbidden_substrings),
             "has_public_boxes": e.has_public_boxes, "note": e.note}
            for e in self.entries]}
