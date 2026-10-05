"""The step that makes generic localisation contribute anything at all.

Before this existed, generic detections carried the prompt that found them,
failed the clip-label test, and were discarded. The architecture was half
wired.
"""
import unittest

from vlmlab.backends.fake import FakeCropClassifier
from vlmlab.naming import is_generic, name_detections
from vlmlab.registry import Registry
from vlmlab.types import Box, BoxSource, ClipRecord, Detection, RejectReason


def _clip(labels=("beaker", "centrifuge")):
    return ClipRecord("c1", "s1", labels, duration_s=180.0,
                      container_frame_count=5400)


def _det(label, frame="f1", score=0.9, presence=0.8, box=None):
    return Detection(frame, label, box or Box(10, 10, 60, 80), score,
                     BoxSource.DETECTION_HEAD, presence=presence, prompt=label)


class TestIsGeneric(unittest.TestCase):
    def setUp(self):
        self.reg = Registry()

    def test_canonical_labels_are_not_generic(self):
        for name in ("beaker", "centrifuge", "fume_hood"):
            self.assertFalse(is_generic(_det(name), self.reg))

    def test_generic_prompts_are_generic(self):
        for name in ("laboratory_instrument", "machine_on_a_laboratory_bench",
                     "scientific_equipment"):
            self.assertTrue(is_generic(_det(name), self.reg))

    def test_a_resolvable_paraphrase_is_not_generic(self):
        self.assertFalse(is_generic(_det("erlenmeyer_flask"), self.reg))


class TestNaming(unittest.TestCase):
    def setUp(self):
        self.reg = Registry()
        self.clip = _clip()

    def test_generic_detections_get_a_fine_grained_label(self):
        cc = FakeCropClassifier(favour={"beaker": 5.0})
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc)
        self.assertEqual(1, len(res.named))
        self.assertEqual("beaker", res.named[0].label)

    def test_canonical_detections_pass_through_untouched(self):
        cc = FakeCropClassifier()
        res = name_detections([_det("beaker")], self.clip, self.reg, cc)
        self.assertEqual(1, len(res.passthrough))
        self.assertEqual(0, len(res.named))

    def test_vocabulary_is_restricted_to_the_clip(self):
        """A crop here can only be a clip label or a known negative."""
        cc = FakeCropClassifier()
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc, hard_negatives=("microscope",))
        allowed = set(self.clip.labels) | {"microscope"}
        for d in res.named:
            self.assertIn(d.label, allowed)
        for dec in res.decisions:
            self.assertIn(dec["chosen"], allowed)

    def test_low_confidence_is_rejected_not_guessed(self):
        cc = FakeCropClassifier()
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc, min_confidence=0.99)
        self.assertEqual(0, len(res.named))
        self.assertEqual(1, len(res.rejected))
        self.assertEqual(RejectReason.AMBIGUOUS_LABEL, res.rejected[0].reason)

    def test_small_margin_is_rejected(self):
        """The confusable siblings are where a forced choice goes wrong."""
        cc = FakeCropClassifier()
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc, min_confidence=0.0, min_margin=0.99)
        self.assertEqual(0, len(res.named))
        self.assertEqual(1, len(res.rejected))

    def test_presence_is_dropped_not_inherited(self):
        """The presence head scored the GENERIC concept, not the new label."""
        cc = FakeCropClassifier(favour={"beaker": 5.0})
        res = name_detections([_det("laboratory_instrument", presence=0.9)],
                              self.clip, self.reg, cc)
        self.assertIsNone(res.named[0].presence)

    def test_score_combines_localisation_and_naming(self):
        cc = FakeCropClassifier(favour={"beaker": 50.0})
        res = name_detections([_det("laboratory_instrument", score=0.8)],
                              self.clip, self.reg, cc)
        named = res.named[0]
        self.assertLessEqual(named.score, 0.8)
        self.assertGreater(named.score, 0.0)

    def test_decisions_record_the_runner_up(self):
        cc = FakeCropClassifier()
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc, min_confidence=0.0, min_margin=0.0)
        dec = res.decisions[0]
        self.assertIn("runner_up", dec)
        self.assertIn("margin", dec)
        self.assertIn("confidence", dec)

    def test_model_id_records_both_stages(self):
        cc = FakeCropClassifier(favour={"beaker": 5.0})
        res = name_detections([_det("laboratory_instrument")], self.clip,
                              self.reg, cc)
        self.assertIn(cc.model_id, res.named[0].model_id)

    def test_classifier_runs_once_per_frame_not_per_box(self):
        cc = FakeCropClassifier(favour={"beaker": 5.0})
        dets = [_det("laboratory_instrument", frame="f1",
                     box=Box(0, 0, 10, 10)),
                _det("laboratory_instrument", frame="f1",
                     box=Box(20, 20, 30, 30)),
                _det("laboratory_instrument", frame="f2",
                     box=Box(0, 0, 10, 10))]
        name_detections(dets, self.clip, self.reg, cc)
        self.assertEqual(2, len(cc.call_log))

    def test_a_clip_with_no_labels_raises(self):
        clip = ClipRecord("c", "s", (), duration_s=10.0,
                          container_frame_count=100)
        with self.assertRaises(ValueError):
            name_detections([_det("laboratory_instrument")], clip, self.reg,
                            FakeCropClassifier())

    def test_deterministic(self):
        cc = FakeCropClassifier()
        a = name_detections([_det("laboratory_instrument")], self.clip,
                            self.reg, cc, min_margin=0.0, min_confidence=0.0)
        b = name_detections([_det("laboratory_instrument")], self.clip,
                            self.reg, cc, min_margin=0.0, min_confidence=0.0)
        self.assertEqual([d.label for d in a.named], [d.label for d in b.named])


if __name__ == "__main__":
    unittest.main()
