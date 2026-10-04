import unittest

from vlmlab.prompts import (FineGrainedPromptError, build_clip_prompt_plan,
                            build_localisation_prompts)
from vlmlab.registry import Registry, UnknownLabel
from vlmlab.types import ClipRecord


def _clip(labels=("centrifuge", "beaker")):
    return ClipRecord("c1", "s1", labels, duration_s=180.0,
                      container_frame_count=5400, fps=30.0)


class TestLocalisationIsClassAgnostic(unittest.TestCase):
    """The central architectural rule: never prompt an instrument name."""

    def test_default_prompts_name_no_class(self):
        reg = Registry()
        prompts = build_localisation_prompts(reg)
        canonical = {n.replace("_", " ") for n in reg.names}
        for p in prompts:
            self.assertNotIn(p.lower(), canonical)

    def test_a_class_name_as_a_localisation_prompt_is_refused(self):
        reg = Registry()
        with self.assertRaises(FineGrainedPromptError):
            build_localisation_prompts(reg, extra=("centrifuge",))

    def test_generic_wording_is_allowed(self):
        reg = Registry()
        prompts = build_localisation_prompts(reg, extra=("apparatus on a bench",))
        self.assertIn("apparatus on a bench", prompts)


class TestClipRestriction(unittest.TestCase):
    def test_positives_are_exactly_the_clip_labels(self):
        reg = Registry()
        plan = build_clip_prompt_plan(_clip(), reg)
        self.assertEqual({"beaker", "centrifuge"}, set(plan.positive_classes))

    def test_concept_count_respects_the_cap(self):
        reg = Registry()
        for cap in (2, 3, 5, 8):
            plan = build_clip_prompt_plan(_clip(), reg, max_concepts=cap)
            self.assertLessEqual(plan.n_concepts, max(cap, len(plan.positives)))

    def test_labelled_classes_are_never_dropped_for_the_cap(self):
        """Dropping a labelled class would discard the supervision entirely."""
        reg = Registry()
        labels = ("beaker", "centrifuge", "microscope", "test_tube",
                  "autoclave", "fume_hood")
        clip = ClipRecord("c", "s", labels, duration_s=180.0,
                          container_frame_count=5400)
        plan = build_clip_prompt_plan(clip, reg, max_concepts=2)
        self.assertEqual(set(labels), set(plan.positive_classes))
        self.assertTrue(plan.note)

    def test_hard_negatives_exclude_the_clip_labels(self):
        reg = Registry()
        plan = build_clip_prompt_plan(_clip(), reg, max_concepts=8,
                                      n_hard_negatives=4)
        self.assertEqual(set(), set(plan.negative_classes) &
                         set(plan.positive_classes))

    def test_canonical_and_display_forms_stay_aligned(self):
        """The display form has spaces; the federated layer keys on canonical."""
        reg = Registry()
        plan = build_clip_prompt_plan(_clip(), reg, max_concepts=8)
        self.assertEqual(len(plan.hard_negatives), len(plan.negative_classes))
        for canon in plan.negative_classes:
            self.assertIn(canon, reg.names)

    def test_queried_classes_combines_both_sides(self):
        reg = Registry()
        plan = build_clip_prompt_plan(_clip(), reg, max_concepts=8)
        self.assertEqual(set(plan.positive_classes) | set(plan.negative_classes),
                         set(plan.queried_classes()))

    def test_unknown_label_in_the_manifest_raises(self):
        reg = Registry()
        clip = ClipRecord("c", "s", ("not_a_class",), duration_s=10.0,
                          container_frame_count=100)
        with self.assertRaises(UnknownLabel):
            build_clip_prompt_plan(clip, reg)

    def test_deterministic(self):
        reg = Registry()
        a = build_clip_prompt_plan(_clip(), reg)
        b = build_clip_prompt_plan(_clip(), reg)
        self.assertEqual(a.all_prompts(), b.all_prompts())


class TestCostModel(unittest.TestCase):
    def test_cost_is_linear_in_concepts(self):
        reg = Registry()
        plan = build_clip_prompt_plan(_clip(), reg, max_concepts=8)
        one = plan.estimated_seconds(45)
        self.assertGreater(one, 0)
        # Doubling keyframes doubles the cost.
        self.assertAlmostEqual(2 * one, plan.estimated_seconds(90), places=6)

    def test_restricting_the_vocabulary_reduces_cost(self):
        reg = Registry()
        narrow = build_clip_prompt_plan(_clip(), reg, max_concepts=3,
                                        n_hard_negatives=0, use_generic=False)
        wide = build_clip_prompt_plan(_clip(), reg, max_concepts=12,
                                      n_hard_negatives=6)
        self.assertLess(narrow.estimated_seconds(45), wide.estimated_seconds(45))


if __name__ == "__main__":
    unittest.main()
