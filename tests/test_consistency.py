import unittest

from vlmlab.consistency import (cross_model_agreement, temporal_consistency,
                                track_stability)
from vlmlab.types import (Box, BoxSource, Detection, RejectReason, Track,
                          TrackPoint)


def _d(frame, label, box, score=0.9, model="a"):
    return Detection(frame, label, box, score, BoxSource.DETECTION_HEAD,
                     presence=0.9, model_id=model)


class TestTemporalConsistency(unittest.TestCase):
    def test_a_single_appearance_is_dropped(self):
        dets = [_d("f001", "beaker", Box(0, 0, 10, 10))]
        v = temporal_consistency(dets, min_appearances=2)
        self.assertEqual(RejectReason.TEMPORALLY_INCONSISTENT, v[0].reason)

    def test_a_persistent_object_survives(self):
        dets = [_d("f00{}".format(i), "beaker", Box(0, 0, 10, 10))
                for i in range(1, 6)]
        v = temporal_consistency(dets, min_appearances=2)
        self.assertTrue(all(x.reason == RejectReason.KEPT for x in v))

    def test_min_appearances_of_one_keeps_everything(self):
        dets = [_d("f001", "beaker", Box(0, 0, 10, 10))]
        v = temporal_consistency(dets, min_appearances=1)
        self.assertEqual(RejectReason.KEPT, v[0].reason)

    def test_a_moved_box_is_not_corroboration(self):
        dets = [_d("f001", "beaker", Box(0, 0, 10, 10)),
                _d("f002", "beaker", Box(500, 500, 510, 510))]
        v = temporal_consistency(dets, min_appearances=2, iou_threshold=0.5)
        self.assertTrue(all(x.reason == RejectReason.TEMPORALLY_INCONSISTENT
                            for x in v))

    def test_different_labels_do_not_corroborate(self):
        dets = [_d("f001", "beaker", Box(0, 0, 10, 10)),
                _d("f002", "centrifuge", Box(0, 0, 10, 10))]
        v = temporal_consistency(dets, min_appearances=2)
        self.assertTrue(all(x.reason == RejectReason.TEMPORALLY_INCONSISTENT
                            for x in v))

    def test_bad_threshold_raises(self):
        with self.assertRaises(ValueError):
            temporal_consistency([], min_appearances=0)


class TestCrossModelAgreement(unittest.TestCase):
    def test_agreement_needs_both_overlap_and_class(self):
        a = [_d("f1", "beaker", Box(0, 0, 10, 10))]
        b = [_d("f1", "beaker", Box(1, 1, 11, 11), model="b")]
        rep = cross_model_agreement(a, b, 0.5)
        self.assertEqual(1, len(rep.agreed))
        self.assertEqual(0, len(rep.only_a))

    def test_class_disagreement_is_flagged_not_resolved(self):
        """These are the confusable-sibling cases; silently picking a winner
        would hide the one error mode this project most needs to see."""
        a = [_d("f1", "centrifuge", Box(0, 0, 10, 10), score=0.6)]
        b = [_d("f1", "orbital_shaker", Box(0, 0, 10, 10), score=0.95, model="b")]
        rep = cross_model_agreement(a, b, 0.5)
        self.assertEqual(0, len(rep.agreed))
        self.assertEqual(1, len(rep.class_conflicts))
        self.assertEqual(1, len(rep.only_a))

    def test_no_overlap_means_no_agreement(self):
        a = [_d("f1", "beaker", Box(0, 0, 10, 10))]
        b = [_d("f1", "beaker", Box(500, 500, 510, 510), model="b")]
        rep = cross_model_agreement(a, b, 0.5)
        self.assertEqual(0, len(rep.agreed))
        self.assertEqual(1, len(rep.only_a))
        self.assertEqual(1, len(rep.only_b))

    def test_different_frames_never_agree(self):
        a = [_d("f1", "beaker", Box(0, 0, 10, 10))]
        b = [_d("f2", "beaker", Box(0, 0, 10, 10), model="b")]
        rep = cross_model_agreement(a, b, 0.5)
        self.assertEqual(0, len(rep.agreed))

    def test_precision_proxy(self):
        a = [_d("f1", "beaker", Box(0, 0, 10, 10)),
             _d("f1", "beaker", Box(100, 100, 110, 110))]
        b = [_d("f1", "beaker", Box(0, 0, 10, 10), model="b")]
        rep = cross_model_agreement(a, b, 0.5)
        self.assertAlmostEqual(0.5, rep.precision_proxy)

    def test_proxy_is_none_with_no_detections(self):
        self.assertIsNone(cross_model_agreement([], []).precision_proxy)


class TestTrackStability(unittest.TestCase):
    def test_a_still_object_is_perfectly_stable(self):
        pts = tuple(TrackPoint("f{}".format(i), Box(0, 0, 10, 10), 0.9)
                    for i in range(5))
        tr = Track(1, "beaker", pts, BoxSource.PROPAGATED)
        self.assertAlmostEqual(1.0, track_stability(tr))

    def test_a_jumping_box_is_unstable(self):
        pts = (TrackPoint("f0", Box(0, 0, 10, 10), 0.9),
               TrackPoint("f1", Box(500, 500, 510, 510), 0.9))
        tr = Track(1, "beaker", pts, BoxSource.PROPAGATED)
        self.assertEqual(0.0, track_stability(tr))

    def test_single_point_has_no_stability(self):
        tr = Track(1, "beaker", (TrackPoint("f0", Box(0, 0, 10, 10), 0.9),),
                   BoxSource.PROPAGATED)
        self.assertIsNone(track_stability(tr))


if __name__ == "__main__":
    unittest.main()
