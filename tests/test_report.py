"""The scoring path, with everything the research insists on applied."""
import unittest

from vlmlab.config import Config
from vlmlab.eval.metrics import Prediction
from vlmlab.eval.report import compare, score
from vlmlab.gold import GoldBox, GoldFrame, GoldSet
from vlmlab.types import Box


def _gold(n_sessions=12, frames_per_session=3, labels=("beaker", "centrifuge")):
    frames = []
    for s in range(n_sessions):
        for f in range(frames_per_session):
            fid = "s{}_f{}.jpg".format(s, f)
            boxes = tuple(GoldBox(lab, Box(10 + 50 * i, 10, 60 + 50 * i, 80))
                          for i, lab in enumerate(labels))
            frames.append(GoldFrame("c{}_{}".format(s, f), "session{}".format(s),
                                    f, fid, 640, 480, boxes, labels))
    return GoldSet(frames, labels)


def _preds(gold, hit_fraction=1.0, jitter=0.0):
    out = []
    n = 0
    total = sum(len(f.boxes) for f in gold.annotated())
    for f in gold.annotated():
        for b in f.boxes:
            n += 1
            if n > total * hit_fraction:
                continue
            box = Box(b.box.x1 + jitter, b.box.y1, b.box.x2 + jitter, b.box.y2)
            out.append(Prediction(f.frame_id, b.label, box, 0.9))
    return out


def _cfg():
    cfg = Config()
    cfg.detector.frame_size = 448
    cfg.eval.n_resamples = 120
    return cfg


class TestScore(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        gold = _gold()
        rep = score(gold, _preds(gold), _cfg())
        self.assertAlmostEqual(1.0, rep["mAP"], places=6)

    def test_no_predictions_score_zero_not_none(self):
        gold = _gold()
        rep = score(gold, [], _cfg())
        self.assertEqual(0.0, rep["mAP"])

    def test_interval_is_reported(self):
        gold = _gold()
        rep = score(gold, _preds(gold, hit_fraction=0.6), _cfg())
        lo, hi = rep["ci95"]
        self.assertIsNotNone(lo)
        self.assertLessEqual(lo, rep["mAP"])
        self.assertLessEqual(rep["mAP"], hi)

    def test_session_is_the_resampling_unit(self):
        gold = _gold(n_sessions=12)
        rep = score(gold, _preds(gold), _cfg())
        self.assertEqual(12, rep["n_sessions"])

    def test_test_choice_follows_group_count(self):
        few = score(_gold(9), [], _cfg())
        many = score(_gold(20), [], _cfg())
        self.assertEqual("permutation", few["statistical_test"])
        self.assertEqual("bootstrap", many["statistical_test"])

    def test_both_design_effects_are_reported_separately(self):
        rep = score(_gold(), _preds(_gold()), _cfg())
        self.assertIn("kish_design_effect", rep.to_dict())
        self.assertIn("icc_design_effect", rep.to_dict())

    def test_per_class_recall_replaces_per_class_precision(self):
        gold = _gold()
        rep = score(gold, _preds(gold), _cfg())
        self.assertIn("beaker", rep["per_class_recall"])
        self.assertIn("ci95", rep["per_class_recall"]["beaker"])
        self.assertIn("not reportable", rep["note"])

    def test_predictions_on_unannotated_frames_are_dropped(self):
        gold = _gold()
        preds = list(_preds(gold))
        preds.append(Prediction("nowhere.jpg", "beaker", Box(0, 0, 10, 10), 0.9))
        rep = score(gold, preds, _cfg())
        self.assertEqual(1, rep["predictions_filtered"]
                         ["dropped_unannotated_frame"])
        self.assertTrue(any("dropped rather than counted" in w
                            for w in rep["warnings"]))

    def test_unannotated_gold_warns(self):
        gold = _gold()
        blank = GoldFrame("cx", "sessionX", 0, "blank.jpg", 640, 480, (), ())
        gold = GoldSet(gold.frames + (blank,), gold.vocabulary)
        rep = score(gold, [], _cfg())
        self.assertTrue(any("not yet annotated" in w for w in rep["warnings"]))


class TestCompare(unittest.TestCase):
    def test_identical_pipelines_are_not_resolvable(self):
        gold = _gold()
        p = _preds(gold)
        rep = compare(gold, p, list(p), _cfg())
        self.assertAlmostEqual(0.0, rep["difference"], places=6)
        self.assertFalse(rep["resolvable"])
        self.assertTrue(any("not resolvable" in w for w in rep["warnings"]))

    def test_a_large_difference_is_resolvable(self):
        gold = _gold()
        rep = compare(gold, _preds(gold), [], _cfg())
        self.assertGreater(rep["difference"], 0.5)
        self.assertTrue(rep["resolvable"])

    def test_the_p_value_floor_is_surfaced(self):
        gold = _gold(9)
        rep = compare(gold, _preds(gold), [], _cfg())
        self.assertGreater(rep["p_value_floor"], 0.0)
        self.assertTrue(any("cannot report a p-value below" in w
                            for w in rep["warnings"]))

    def test_minimum_detectable_difference_is_reported(self):
        gold = _gold()
        rep = compare(gold, _preds(gold), _preds(gold, hit_fraction=0.5), _cfg())
        self.assertIsNotNone(rep["minimum_detectable_difference"])

    def test_pre_registered_threshold_is_honoured(self):
        """A threshold above the metric's own range makes nothing resolvable."""
        cfg = _cfg()
        cfg.eval.pre_registered_mde = 150.0  # 1.5 in metric units
        gold = _gold()
        rep = compare(gold, _preds(gold), [], cfg)
        self.assertFalse(rep["resolvable"])
        self.assertTrue(any("below the pre-registered minimum" in w
                            for w in rep["warnings"]))


if __name__ == "__main__":
    unittest.main()
