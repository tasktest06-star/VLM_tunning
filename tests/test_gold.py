import os
import tempfile
import unittest

from vlmlab.gold import (GoldBox, GoldError, GoldFrame, GoldSet,
                         annotation_budget, select_gold_frames)
from vlmlab.types import Box, ClipRecord


def _clips(n=150, n_sessions=15):
    return [ClipRecord("clip_{:04d}".format(i), "session{}".format(i % n_sessions),
                       ("beaker", "centrifuge"), duration_s=180.0,
                       container_frame_count=5400, fps=30.0)
            for i in range(n)]


class TestSelection(unittest.TestCase):
    def test_takes_frames_from_every_clip(self):
        """Spread beats depth: never dense on a few clips."""
        clips = _clips(150)
        gs = select_gold_frames(clips, frames_per_clip=3, sample_fps=2.0)
        self.assertEqual(450, len(gs.frames))
        self.assertEqual(150, len(gs.clips))

    def test_frames_are_spaced_well_apart_on_a_long_clip(self):
        gs = select_gold_frames(_clips(1), frames_per_clip=3, sample_fps=2.0)
        idx = [f.frame_index for f in gs.frames]
        gaps_s = [(idx[i + 1] - idx[i]) / 2.0 for i in range(len(idx) - 1)]
        for g in gaps_s:
            self.assertGreater(g, 30.0)

    def test_interior_quantiles_avoid_the_clip_edges(self):
        gs = select_gold_frames(_clips(1), frames_per_clip=3, sample_fps=2.0)
        idx = [f.frame_index for f in gs.frames]
        self.assertGreater(idx[0], 0)
        self.assertLess(idx[-1], 359)

    def test_short_clip_cannot_exceed_its_spacing_budget(self):
        clip = ClipRecord("c", "s", ("beaker",), duration_s=4.0,
                          container_frame_count=120, fps=30.0)
        gs = select_gold_frames([clip], frames_per_clip=8, min_spacing_s=2.0,
                                sample_fps=2.0)
        self.assertLessEqual(len(gs.frames), 3)

    def test_missing_duration_raises(self):
        clip = ClipRecord("c", "s", ("beaker",), container_frame_count=100)
        with self.assertRaises(GoldError):
            select_gold_frames([clip])

    def test_missing_frame_count_raises(self):
        clip = ClipRecord("c", "s", ("beaker",), duration_s=10.0)
        with self.assertRaises(GoldError):
            select_gold_frames([clip])

    def test_sessions_are_carried_through(self):
        gs = select_gold_frames(_clips(30, 6), frames_per_clip=2, sample_fps=2.0)
        self.assertEqual(6, len(gs.sessions))


class TestWarnings(unittest.TestCase):
    def test_unannotated_frames_warn(self):
        gs = select_gold_frames(_clips(10), frames_per_clip=2, sample_fps=2.0,
                                vocabulary=("beaker",))
        self.assertTrue(any("not yet annotated" in w for w in gs.warnings()))

    def test_too_few_sessions_warn(self):
        gs = select_gold_frames(_clips(10, 3), frames_per_clip=2, sample_fps=2.0)
        self.assertTrue(any("recording sessions" in w for w in gs.warnings()))

    def test_sparse_class_warns_about_unfalsifiability(self):
        frame = GoldFrame("c", "s", 0, "f0.jpg", 640, 480,
                          boxes=(GoldBox("beaker", Box(0, 0, 10, 10)),),
                          exhaustive_for=("beaker", "centrifuge"))
        gs = GoldSet([frame], ("beaker", "centrifuge"))
        ws = gs.warnings()
        self.assertTrue(any("unfalsifiable" in w for w in ws))
        self.assertTrue(any("no gold instances" in w for w in ws))


class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.frame = GoldFrame("c1", "s1", 0, "f0.jpg", 640, 480,
                               boxes=(GoldBox("beaker", Box(0, 0, 10, 10)),),
                               exhaustive_for=("beaker",))
        self.gs = GoldSet([self.frame], ("beaker", "centrifuge"))

    def test_predictions_on_unannotated_frames_are_dropped(self):
        """Counting them wrong would penalise the model because nobody looked."""
        from vlmlab.eval.metrics import Prediction
        preds = [Prediction("not-a-gold-frame.jpg", "beaker",
                            Box(0, 0, 10, 10), 0.9)]
        kept, rep = self.gs.filter_predictions(preds)
        self.assertEqual(0, len(kept))
        self.assertEqual(1, rep["dropped_unannotated_frame"])

    def test_predictions_of_unchecked_classes_are_dropped(self):
        from vlmlab.eval.metrics import Prediction
        preds = [Prediction("f0.jpg", "centrifuge", Box(0, 0, 10, 10), 0.9)]
        kept, rep = self.gs.filter_predictions(preds)
        self.assertEqual(0, len(kept))
        self.assertEqual(1, rep["dropped_class_not_checked"])

    def test_checked_classes_survive(self):
        from vlmlab.eval.metrics import Prediction
        preds = [Prediction("f0.jpg", "beaker", Box(0, 0, 10, 10), 0.9)]
        kept, _ = self.gs.filter_predictions(preds)
        self.assertEqual(1, len(kept))


class TestSessionUnits(unittest.TestCase):
    def test_grouped_by_session_not_frame(self):
        frames = [
            GoldFrame("c1", "sA", 0, "a0.jpg", 640, 480,
                      boxes=(GoldBox("beaker", Box(0, 0, 10, 10)),),
                      exhaustive_for=("beaker",)),
            GoldFrame("c2", "sA", 0, "a1.jpg", 640, 480,
                      boxes=(GoldBox("beaker", Box(0, 0, 10, 10)),),
                      exhaustive_for=("beaker",)),
            GoldFrame("c3", "sB", 0, "b0.jpg", 640, 480,
                      boxes=(GoldBox("beaker", Box(0, 0, 10, 10)),),
                      exhaustive_for=("beaker",)),
        ]
        gs = GoldSet(frames, ("beaker",))
        units = gs.units_by_session([])
        self.assertEqual(2, len(units))
        by_id = {u["session_id"]: u for u in units}
        self.assertEqual(2, len(by_id["sA"]["gts"]))
        self.assertEqual(1, len(by_id["sB"]["gts"]))


class TestRoundTrip(unittest.TestCase):
    def test_save_and_load(self):
        frame = GoldFrame("c1", "s1", 7, "f7.jpg", 640, 480,
                          boxes=(GoldBox("beaker", Box(1, 2, 3, 4),
                                         track_id=5),),
                          exhaustive_for=("beaker", "centrifuge"))
        gs = GoldSet([frame], ("beaker", "centrifuge"))
        path = os.path.join(tempfile.mkdtemp(), "gold.json")
        gs.save(path)
        back = GoldSet.load(path)
        self.assertEqual(gs.vocabulary, back.vocabulary)
        self.assertEqual(1, len(back.frames))
        self.assertEqual((1.0, 2.0, 3.0, 4.0), back.frames[0].boxes[0].box.as_tuple())
        self.assertEqual(5, back.frames[0].boxes[0].track_id)


class TestBudget(unittest.TestCase):
    def test_uses_the_corrected_per_box_time(self):
        b = annotation_budget(150, 3)
        self.assertEqual(450, b["n_frames"])
        self.assertEqual(1350, b["n_boxes"])
        self.assertGreater(b["hours_from_scratch"], 5.0)
        self.assertLess(b["hours_verify_and_correct"], b["hours_from_scratch"])

    def test_verify_speedup_is_the_measured_one_not_the_claimed_one(self):
        b = annotation_budget(150, 3, verify_speedup=1.5)
        ratio = b["hours_from_scratch"] / b["hours_verify_and_correct"]
        self.assertAlmostEqual(1.5, ratio, delta=0.05)


if __name__ == "__main__":
    unittest.main()
