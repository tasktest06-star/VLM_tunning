"""Eleven hand-computed fixtures, with the literal constants in the test body.

F4 and F8 alone catch most implementation errors: the first proves the
101-point interpolation is applied rather than a naive mean, and the second
proves the overlap threshold is inclusive.
"""
import random
import unittest

from vlmlab.eval.metrics import (GroundTruth, Prediction, ap_from_matches,
                                 average_precision, match_detections,
                                 mean_average_precision, per_class_recall)
from vlmlab.types import Box


def P(img, lab, box, score):
    return Prediction(img, lab, box, score)


def G(img, lab, box, ignore=False):
    return GroundTruth(img, lab, box, ignore)


class TestFixtures(unittest.TestCase):
    def test_f1_perfect_hit(self):
        r = average_precision([P("i", "a", Box(0, 0, 10, 10), 0.9)],
                              [G("i", "a", Box(0, 0, 10, 10))])
        self.assertEqual(1.0, r.ap)

    def test_f2_positives_exist_but_nothing_detected(self):
        """Zero, not None: there was something to find and it was missed."""
        r = average_precision([], [G("i", "a", Box(0, 0, 10, 10))])
        self.assertEqual(0.0, r.ap)
        self.assertEqual(1, r.n_pos)

    def test_f3_no_positives_gives_none_never_zero(self):
        r = average_precision([P("i", "a", Box(0, 0, 10, 10), 0.9)], [])
        self.assertIsNone(r.ap)
        self.assertEqual(0, r.n_pos)

    def test_f4_half_recall_is_51_over_101(self):
        r = average_precision(
            [P("i", "a", Box(0, 0, 10, 10), 0.9)],
            [G("i", "a", Box(0, 0, 10, 10)), G("i", "a", Box(50, 50, 60, 60))])
        self.assertAlmostEqual(0.504950495049505, r.ap, places=15)

    def test_f5_higher_scoring_false_positive(self):
        r = average_precision(
            [P("i", "a", Box(100, 100, 110, 110), 0.99),
             P("i", "a", Box(0, 0, 10, 10), 0.5)],
            [G("i", "a", Box(0, 0, 10, 10)), G("i", "a", Box(50, 50, 60, 60))])
        self.assertAlmostEqual(0.25247524752475248, r.ap, places=15)

    def test_f6_one_ground_truth_cannot_be_matched_twice(self):
        t = match_detections([P("i", "a", Box(0, 0, 10, 10), 0.9),
                              P("i", "a", Box(1, 1, 11, 11), 0.8)],
                             [G("i", "a", Box(0, 0, 10, 10))])
        self.assertEqual(1, sum(1 for x in t.is_tp if x))
        self.assertEqual(2, t.n_dets)

    def test_f7_consumes_the_higher_overlap_ground_truth(self):
        t = match_detections([P("i", "a", Box(0, 0, 10, 10), 0.9)],
                             [G("i", "a", Box(0, 0, 6, 10)),
                              G("i", "a", Box(0, 0, 9, 10))])
        self.assertEqual((True,), t.is_tp)
        self.assertEqual(2, t.n_pos)

    def test_f8_overlap_of_exactly_one_half_is_a_true_positive(self):
        """The threshold is inclusive."""
        r = average_precision([P("i", "a", Box(0, 0, 4, 2), 0.9)],
                              [G("i", "a", Box(0, 0, 4, 4))],
                              iou_threshold=0.5)
        self.assertEqual(1.0, r.ap)

    def test_f9_ignored_ground_truth_behaves_like_no_ground_truth(self):
        r = average_precision([P("i", "a", Box(0, 0, 10, 10), 0.9)],
                              [G("i", "a", Box(0, 0, 10, 10), ignore=True)])
        self.assertIsNone(r.ap)
        self.assertEqual(1, r.n_ignored)

    def test_f10_truncation_happens_before_matching(self):
        """The default detection cap is exactly this truncation."""
        t = match_detections([P("i", "a", Box(100, 100, 110, 110), 0.9),
                              P("i", "a", Box(0, 0, 10, 10), 0.1)],
                             [G("i", "a", Box(0, 0, 10, 10))], max_dets=1)
        self.assertEqual((False,), t.is_tp)
        t2 = match_detections([P("i", "a", Box(100, 100, 110, 110), 0.9),
                               P("i", "a", Box(0, 0, 10, 10), 0.1)],
                              [G("i", "a", Box(0, 0, 10, 10))], max_dets=0)
        self.assertEqual(1, sum(1 for x in t2.is_tp if x))

    def test_f11_stable_across_input_shuffles(self):
        base = None
        for seed in range(100):
            preds = [P("i", "a", Box(0, 0, 10, 10), 0.5),
                     P("i", "a", Box(100, 100, 110, 110), 0.5)]
            random.Random(seed).shuffle(preds)
            r = average_precision(preds, [G("i", "a", Box(0, 0, 10, 10))])
            if base is None:
                base = r.ap
            self.assertEqual(base, r.ap)


class TestSplitMatchesAndCurve(unittest.TestCase):
    def test_two_stage_matches_one_shot(self):
        """The permutation test relies on re-running only the curve half."""
        preds = [P("i", "a", Box(0, 0, 10, 10), 0.9),
                 P("i", "a", Box(50, 50, 60, 60), 0.4)]
        gts = [G("i", "a", Box(0, 0, 10, 10)), G("i", "a", Box(50, 50, 60, 60))]
        one = average_precision(preds, gts)
        two = ap_from_matches(match_detections(preds, gts))
        self.assertEqual(one.ap, two.ap)


class TestMacroAverage(unittest.TestCase):
    def test_excludes_classes_without_positives(self):
        preds = [P("i", "a", Box(0, 0, 10, 10), 0.9),
                 P("i", "b", Box(0, 0, 10, 10), 0.9)]
        gts = [G("i", "a", Box(0, 0, 10, 10))]
        res = mean_average_precision(preds, gts, labels=["a", "b"])
        self.assertEqual(1, res["n_classes_averaged"])
        self.assertEqual(2, res["n_classes_total"])
        self.assertEqual(1.0, res["mAP"])

    def test_all_classes_absent_gives_none(self):
        res = mean_average_precision([], [], labels=["a", "b"])
        self.assertIsNone(res["mAP"])
        self.assertEqual(0, res["n_classes_averaged"])

    def test_labels_are_not_mixed(self):
        preds = [P("i", "a", Box(0, 0, 10, 10), 0.9)]
        gts = [G("i", "b", Box(0, 0, 10, 10))]
        res = mean_average_precision(preds, gts, labels=["a", "b"])
        self.assertEqual(0.0, res["per_class"]["b"].ap)
        self.assertIsNone(res["per_class"]["a"].ap)


class TestPerClassRecall(unittest.TestCase):
    def test_recall_with_exact_interval(self):
        preds = [P("i", "a", Box(0, 0, 10, 10), 0.9)]
        gts = [G("i", "a", Box(0, 0, 10, 10)), G("i", "a", Box(50, 50, 60, 60))]
        res = per_class_recall(preds, gts, "a", score_threshold=0.5)
        self.assertEqual(0.5, res["recall"])
        self.assertEqual(2, res["n_pos"])
        lo, hi = res["ci95"]
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_threshold_filters_predictions(self):
        preds = [P("i", "a", Box(0, 0, 10, 10), 0.2)]
        gts = [G("i", "a", Box(0, 0, 10, 10))]
        self.assertEqual(0.0, per_class_recall(preds, gts, "a", 0.5)["recall"])


if __name__ == "__main__":
    unittest.main()
