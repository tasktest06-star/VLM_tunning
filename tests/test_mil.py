"""The clip-label constraint must be absolute, whatever the detector's confidence."""
import unittest

from vlmlab import federated
from vlmlab.config import CascadeConfig
from vlmlab.mil import (cascade_summary, clip_passes_presence, presence_score,
                        run_cascade)
from vlmlab.prompts import build_clip_prompt_plan
from vlmlab.registry import Registry
from vlmlab.types import Box, BoxSource, ClipRecord, Detection, RejectReason


def _setup(labels=("centrifuge", "beaker")):
    reg = Registry()
    clip = ClipRecord("c1", "s1", labels, duration_s=180.0,
                      container_frame_count=5400, fps=30.0)
    plan = build_clip_prompt_plan(clip, reg, max_concepts=6, n_hard_negatives=4)
    fed = federated.derive(clip, plan.queried_classes(), reg.names)
    return reg, clip, plan, fed


def _det(frame, label, box=None, score=0.9, presence=0.9):
    return Detection(frame, label, box or Box(20, 20, 120, 140), score,
                     BoxSource.DETECTION_HEAD, presence=presence)


class TestClipLabelConstraint(unittest.TestCase):
    def test_never_accepts_a_class_outside_the_clip_labels(self):
        _reg, clip, _plan, fed = _setup()
        cfg = CascadeConfig()
        dets = [_det("f1", "microscope", score=0.99, presence=0.99)]
        res = run_cascade(dets, clip, fed, cfg)
        self.assertEqual(0, len(res.accepted))
        self.assertEqual(RejectReason.NOT_IN_CLIP_LABELS, res.verdicts[0].reason)

    def test_confidence_cannot_override_the_constraint(self):
        _reg, clip, _plan, fed = _setup()
        cfg = CascadeConfig()
        for score in (0.5, 0.9, 0.999, 1.0):
            res = run_cascade([_det("f1", "microscope", score=score,
                                    presence=score)], clip, fed, cfg)
            self.assertEqual(0, len(res.accepted))

    def test_queried_but_absent_class_becomes_a_background_box(self):
        """A guaranteed false positive is free, correctly labelled negative data."""
        _reg, clip, plan, fed = _setup()
        cfg = CascadeConfig()
        neg = plan.negative_classes[0]
        res = run_cascade([_det("f1", neg)], clip, fed, cfg)
        self.assertEqual(0, len(res.accepted))
        self.assertEqual(1, len(res.background))
        self.assertEqual(RejectReason.KEPT_AS_BACKGROUND, res.verdicts[0].reason)

    def test_a_labelled_class_is_accepted(self):
        _reg, clip, _plan, fed = _setup()
        cfg = CascadeConfig()
        res = run_cascade([_det("f1", "beaker")], clip, fed, cfg)
        self.assertEqual(1, len(res.accepted))
        self.assertEqual("beaker", res.accepted[0].label)


class TestGates(unittest.TestCase):
    def setUp(self):
        self.reg, self.clip, self.plan, self.fed = _setup()
        self.cfg = CascadeConfig()

    def test_below_score_rejected(self):
        res = run_cascade([_det("f1", "beaker", score=0.05)], self.clip,
                          self.fed, self.cfg)
        self.assertEqual(RejectReason.BELOW_SCORE, res.verdicts[0].reason)

    def test_below_presence_rejected(self):
        res = run_cascade([_det("f1", "beaker", presence=0.01)], self.clip,
                          self.fed, self.cfg)
        self.assertEqual(RejectReason.BELOW_PRESENCE, res.verdicts[0].reason)

    def test_tiny_side_rejected(self):
        res = run_cascade([_det("f1", "beaker", box=Box(0, 0, 5, 5))],
                          self.clip, self.fed, self.cfg)
        self.assertEqual(RejectReason.SIDE_TOO_SMALL, res.verdicts[0].reason)

    def test_area_out_of_range_rejected(self):
        res = run_cascade([_det("f1", "beaker", box=Box(0, 0, 639, 479))],
                          self.clip, self.fed, self.cfg, image_size=(640, 480))
        self.assertEqual(RejectReason.AREA_OUT_OF_RANGE, res.verdicts[0].reason)

    def test_suppression_within_a_frame(self):
        dets = [_det("f1", "beaker", box=Box(20, 20, 120, 140), score=0.9),
                _det("f1", "beaker", box=Box(22, 22, 122, 142), score=0.6)]
        res = run_cascade(dets, self.clip, self.fed, self.cfg)
        self.assertEqual(1, len(res.accepted))
        self.assertIn(RejectReason.SUPPRESSED_NMS,
                      [v.reason for v in res.verdicts])

    def test_every_detection_gets_exactly_one_verdict(self):
        dets = [_det("f1", "beaker"), _det("f1", "microscope"),
                _det("f2", "beaker", box=Box(0, 0, 4, 4))]
        res = run_cascade(dets, self.clip, self.fed, self.cfg)
        self.assertEqual(len(dets), len(res.verdicts))


class TestPresenceGate(unittest.TestCase):
    def test_maximum_over_frames(self):
        dets = [_det("f1", "beaker", score=0.4, presence=0.3),
                _det("f2", "beaker", score=0.9, presence=0.9),
                _det("f3", "beaker", score=0.2, presence=0.2)]
        self.assertAlmostEqual(0.81, presence_score(dets, "beaker"), places=6)

    def test_one_strong_frame_passes_a_positive_clip(self):
        dets = [_det("f{}".format(i), "beaker", score=0.1, presence=0.1)
                for i in range(10)]
        dets.append(_det("f99", "beaker", score=0.95, presence=0.95))
        self.assertTrue(clip_passes_presence(dets, "beaker", 0.5))

    def test_all_weak_frames_fail(self):
        dets = [_det("f{}".format(i), "beaker", score=0.3, presence=0.3)
                for i in range(20)]
        self.assertFalse(clip_passes_presence(dets, "beaker", 0.5))

    def test_absent_label_scores_zero(self):
        self.assertEqual(0.0, presence_score([_det("f1", "beaker")], "autoclave"))


class TestMissingLabels(unittest.TestCase):
    def test_a_label_that_produced_nothing_is_reported(self):
        """These clips go to the human one-box queue."""
        _reg, clip, _plan, fed = _setup()
        cfg = CascadeConfig()
        res = run_cascade([_det("f1", "beaker")], clip, fed, cfg)
        self.assertIn("centrifuge", res.missing_labels)
        self.assertNotIn("beaker", res.missing_labels)

    def test_summary_aggregates(self):
        _reg, clip, _plan, fed = _setup()
        cfg = CascadeConfig()
        results = [run_cascade([_det("f1", "beaker")], clip, fed, cfg)
                   for _ in range(3)]
        summary = cascade_summary(results)
        self.assertEqual(3, summary["labels_never_found"]["centrifuge"])


if __name__ == "__main__":
    unittest.main()
