"""Tracking metrics must separate detection failure from association failure.

That separation is the whole reason for choosing this metric family here: a
three-minute clip exceeds the backend's documented limit six times over, so
cross-chunk identity stitching is the dominant risk and a detection-weighted
metric would hide it.
"""
import unittest

from vlmlab.eval.tracking import ALPHAS, hota, track_summary
from vlmlab.types import Box, BoxSource, Track, TrackPoint


def _d(frame, track_id, label="beaker", box=None, score=0.9):
    return {"frame_id": frame, "label": label,
            "box": box or Box(0, 0, 10, 10), "track_id": track_id,
            "score": score}


class TestHota(unittest.TestCase):
    def test_perfect_tracking_scores_one(self):
        g = [_d("f{}".format(i), 1) for i in range(5)]
        p = [_d("f{}".format(i), 7) for i in range(5)]
        r = hota(g, p)
        self.assertAlmostEqual(1.0, r["hota"], places=6)
        self.assertAlmostEqual(1.0, r["det_a"], places=6)
        self.assertAlmostEqual(1.0, r["ass_a"], places=6)

    def test_identity_switch_hurts_association_not_detection(self):
        """The property that makes this metric worth computing."""
        g = [_d("f{}".format(i), 1) for i in range(6)]
        p = [_d("f{}".format(i), 7 if i < 3 else 8) for i in range(6)]
        r = hota(g, p)
        self.assertAlmostEqual(1.0, r["det_a"], places=6)
        self.assertLess(r["ass_a"], 0.75)
        self.assertLess(r["hota"], r["det_a"])

    def test_missing_everything_scores_zero(self):
        g = [_d("f{}".format(i), 1) for i in range(5)]
        r = hota(g, [])
        self.assertEqual(0.0, r["hota"])
        self.assertEqual(0.0, r["det_a"])

    def test_no_gold_returns_none_not_zero(self):
        r = hota([], [_d("f0", 1)])
        self.assertIsNone(r["hota"])
        self.assertEqual(0, r["n_gold"])

    def test_false_positives_reduce_detection_accuracy(self):
        g = [_d("f0", 1)]
        p = [_d("f0", 7), _d("f0", 8, box=Box(100, 100, 110, 110))]
        r = hota(g, p)
        self.assertLess(r["det_a"], 1.0)

    def test_label_mismatch_is_not_a_match(self):
        g = [_d("f0", 1, label="beaker")]
        p = [_d("f0", 7, label="centrifuge")]
        r = hota(g, p)
        self.assertEqual(0.0, r["det_a"])

    def test_alpha_sweep_is_the_published_range(self):
        self.assertEqual(19, len(ALPHAS))
        self.assertAlmostEqual(0.05, ALPHAS[0])
        self.assertAlmostEqual(0.95, ALPHAS[-1])

    def test_per_alpha_rows_are_reported(self):
        g = [_d("f0", 1)]
        r = hota(g, [_d("f0", 7)])
        self.assertEqual(len(ALPHAS), len(r["per_alpha"]))
        for row in r["per_alpha"]:
            self.assertIn("tp", row)
            self.assertIn("fn", row)
            self.assertIn("fp", row)

    def test_loose_overlap_fails_at_strict_thresholds(self):
        g = [_d("f0", 1, box=Box(0, 0, 10, 10))]
        p = [_d("f0", 7, box=Box(0, 0, 10, 5))]
        r = hota(g, p)
        strict = [row for row in r["per_alpha"] if row["alpha"] >= 0.9]
        self.assertTrue(all(row["tp"] == 0 for row in strict))
        loose = [row for row in r["per_alpha"] if row["alpha"] <= 0.5]
        self.assertTrue(all(row["tp"] == 1 for row in loose))

    def test_the_approximation_is_declared(self):
        r = hota([_d("f0", 1)], [_d("f0", 7)])
        self.assertIn("greedy", r["note"])
        self.assertIn("not comparable", r["note"].lower())


class TestTrackSummary(unittest.TestCase):
    def _track(self, tid, n, chunks=(0,), frag=None):
        pts = tuple(TrackPoint("f{}".format(i), Box(0, 0, 10, 10), 0.9)
                    for i in range(n))
        return Track(tid, "beaker", pts, BoxSource.PROPAGATED, frag, chunks)

    def test_empty(self):
        self.assertEqual({"n_tracks": 0}, track_summary([]))

    def test_counts_stitched_tracks(self):
        s = track_summary([self._track(1, 5, (0,)), self._track(2, 5, (0, 1))])
        self.assertEqual(2, s["n_tracks"])
        self.assertEqual(1, s["n_stitched"])
        self.assertAlmostEqual(0.5, s["fraction_stitched"])

    def test_length_statistics(self):
        s = track_summary([self._track(1, 2), self._track(2, 8)])
        self.assertEqual(2, s["min_length"])
        self.assertEqual(8, s["max_length"])
        self.assertAlmostEqual(5.0, s["mean_length"])

    def test_fragmentation_averaged_when_present(self):
        s = track_summary([self._track(1, 3, frag=0.2),
                           self._track(2, 3, frag=0.4)])
        self.assertAlmostEqual(0.3, s["mean_mask_fragmentation"])

    def test_fragmentation_none_when_absent(self):
        s = track_summary([self._track(1, 3)])
        self.assertIsNone(s["mean_mask_fragmentation"])

    def test_explains_why_stitching_is_expected(self):
        s = track_summary([self._track(1, 3)])
        self.assertIn("30-second", s["note"])


if __name__ == "__main__":
    unittest.main()
