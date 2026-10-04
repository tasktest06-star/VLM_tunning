import unittest

from vlmlab.config import Config
from vlmlab.eval.protocol import (LeakageError, ThresholdChoice, build_protocol,
                                  choose_global_threshold, f1_objective)
from vlmlab.types import ClipRecord


def _clips(n_sessions, per_session=3):
    out = []
    for s in range(n_sessions):
        for c in range(per_session):
            out.append(ClipRecord("s{}c{}".format(s, c), "s{}".format(s),
                                  ("beaker",) if (s + c) % 2 == 0 else ("centrifuge",),
                                  duration_s=180.0, container_frame_count=5400))
    return out


class TestOneGlobalThreshold(unittest.TestCase):
    """There is deliberately no per-class threshold interface at all."""

    def test_no_per_class_interface_exists(self):
        tc = ThresholdChoice(0.5, "f1", 10, [])
        self.assertFalse(hasattr(tc, "per_class_thresholds"))
        self.assertFalse(hasattr(tc, "thresholds"))
        self.assertIsInstance(tc.threshold, float)

    def test_config_forbids_per_class(self):
        self.assertFalse(Config().eval.per_class_thresholds)

    def test_picks_the_best_threshold(self):
        records = [(0.9, True, 0), (0.8, True, 1), (0.7, True, 2),
                   (0.3, False, 0), (0.2, False, 1), (0.1, False, 2)]
        tc = choose_global_threshold(records, f1_objective)
        self.assertGreater(tc.threshold, 0.3)
        self.assertLessEqual(tc.threshold, 0.7)

    def test_curve_is_recorded_for_audit(self):
        records = [(0.9, True, 0), (0.1, False, 1)]
        tc = choose_global_threshold(records, f1_objective)
        self.assertTrue(tc.curve)
        for t, v in tc.curve:
            self.assertIsInstance(t, float)


class TestLeakageGuard(unittest.TestCase):
    def test_missing_fold_identifier_refused(self):
        with self.assertRaises(LeakageError):
            choose_global_threshold([(0.9, True, None)], f1_objective)
        with self.assertRaises(LeakageError):
            choose_global_threshold([(0.9, True)], f1_objective)

    def test_fitting_on_the_held_out_fold_refused(self):
        records = [(0.9, True, 0), (0.8, True, 1)]
        with self.assertRaises(LeakageError):
            choose_global_threshold(records, f1_objective, fold_of=0)

    def test_other_folds_are_fine(self):
        records = [(0.9, True, 1), (0.2, False, 2)]
        tc = choose_global_threshold(records, f1_objective, fold_of=0)
        self.assertIsInstance(tc.threshold, float)

    def test_guard_can_be_disabled_explicitly(self):
        tc = choose_global_threshold([(0.9, True, None)], f1_objective,
                                      in_fold_guard=False)
        self.assertIsInstance(tc.threshold, float)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            choose_global_threshold([], f1_objective)


class TestBuildProtocol(unittest.TestCase):
    def test_few_sessions_gives_leave_one_out_and_permutation(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        report = build_protocol(_clips(9), cfg)
        self.assertEqual("leave-one-session-out", report.scheme)
        self.assertEqual("permutation", report.test_name)
        self.assertEqual(9, report.n_groups)

    def test_many_sessions_gives_grouped_folds_and_bootstrap(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        report = build_protocol(_clips(20), cfg)
        self.assertEqual("grouped-5-fold", report.scheme)
        self.assertEqual("bootstrap", report.test_name)

    def test_summary_is_serialisable(self):
        import json
        cfg = Config()
        cfg.detector.frame_size = 448
        summary = build_protocol(_clips(14), cfg).summary()
        json.dumps(summary)
        self.assertIn("statistical_test", summary)

    def test_folds_never_leak_a_session(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        report = build_protocol(_clips(20), cfg)
        for train, val in report.folds:
            self.assertEqual(set(), {c.session_id for c in train} &
                             {c.session_id for c in val})


if __name__ == "__main__":
    unittest.main()
