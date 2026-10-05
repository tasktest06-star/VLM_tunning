import unittest

from vlmlab.export.rtdetr_yaml import render_dataset_yaml
from vlmlab.registry import Registry
from vlmlab.train.retention import (RetentionError, choose_held_out_classes,
                                    format_retention, retention_report)
from vlmlab.train.sweep import (DEFAULT_ALPHAS, SweepError,
                                choose_operating_point, format_sweep, run_sweep)


class TestHeldOutSelection(unittest.TestCase):
    def test_prefers_classes_that_have_public_boxes(self):
        """Withholding a class with no public data would waste scarce labels."""
        reg = Registry()
        held = choose_held_out_classes(reg, n=6)
        with_boxes = sum(1 for h in held if reg.get(h).has_public_boxes)
        self.assertGreaterEqual(with_boxes, 5)

    def test_respects_always_keep(self):
        reg = Registry()
        held = choose_held_out_classes(reg, n=4, always_keep=("beaker",))
        self.assertNotIn("beaker", held)

    def test_refuses_to_hold_out_everything(self):
        reg = Registry()
        with self.assertRaises(RetentionError):
            choose_held_out_classes(reg, n=len(reg.names))

    def test_deterministic(self):
        reg = Registry()
        self.assertEqual(choose_held_out_classes(reg, n=5),
                         choose_held_out_classes(reg, n=5))


class TestRetentionReport(unittest.TestCase):
    def test_the_median_catches_what_the_mean_hides(self):
        """Four classes collapse, two improve. The mean barely moves."""
        held = ["a", "b", "c", "d", "e", "f"]
        before = {k: 0.5 for k in held}
        after = {"a": 0.05, "b": 0.05, "c": 0.05, "d": 0.05,
                 "e": 1.3, "f": 1.3}
        rep = retention_report(before, after, held)
        self.assertLess(rep["median_relative"], -0.5)
        self.assertGreater(rep["mean_relative"], -0.2)
        self.assertTrue(any("mean-based retention metric hides" in w
                            for w in rep["warnings"]))

    def test_headline_is_the_median(self):
        rep = retention_report({"a": 1.0}, {"a": 0.9}, ["a"])
        self.assertEqual("median", rep["headline"])

    def test_large_median_drop_warns(self):
        rep = retention_report({"a": 1.0, "b": 1.0}, {"a": 0.2, "b": 0.2},
                               ["a", "b"])
        self.assertTrue(any("Reduce the step budget" in w
                            for w in rep["warnings"]))

    def test_missing_classes_are_reported_not_skipped(self):
        rep = retention_report({"a": 1.0, "b": 1.0}, {"a": 0.9}, ["a", "b"])
        self.assertTrue(any("missing from a side" in w for w in rep["warnings"]))

    def test_no_shared_classes_raises(self):
        with self.assertRaises(RetentionError):
            retention_report({"a": 1.0}, {"b": 1.0}, ["a"])

    def test_empty_held_out_raises(self):
        with self.assertRaises(RetentionError):
            retention_report({}, {}, [])

    def test_formats_without_error(self):
        rep = retention_report({"a": 1.0, "b": 0.8}, {"a": 0.9, "b": 0.7},
                               ["a", "b"])
        text = format_retention(rep)
        self.assertIn("MEDIAN", text)
        self.assertIn("diagnostic only", text)


class TestSweep(unittest.TestCase):
    def _curve(self, mapping):
        return run_sweep(mapping.keys(),
                         lambda a: {"in_domain": mapping[a][0],
                                    "retention": mapping[a][1]})

    def test_both_endpoints_are_required(self):
        with self.assertRaises(SweepError):
            run_sweep((0.2, 0.4), lambda a: {"in_domain": 1, "retention": 1})

    def test_out_of_range_coefficient_raises(self):
        with self.assertRaises(SweepError):
            run_sweep((0.0, 1.5), lambda a: {"in_domain": 1, "retention": 1})

    def test_missing_keys_raise(self):
        with self.assertRaises(SweepError):
            run_sweep((0.0, 1.0), lambda a: {"in_domain": 1})

    def test_picks_an_interior_point_over_the_fine_tuned_end(self):
        rows = self._curve({0.0: (0.60, 0.60), 0.4: (0.72, 0.62),
                            1.0: (0.78, 0.30)})
        dec = choose_operating_point(rows)
        self.assertEqual(0.4, dec["chosen"])
        self.assertGreater(dec["gain_over_frozen"], 0.1)

    def test_detects_a_point_dominating_both_endpoints(self):
        rows = self._curve({0.0: (0.60, 0.60), 0.4: (0.80, 0.65),
                            1.0: (0.78, 0.30)})
        dec = choose_operating_point(rows)
        self.assertIn(0.4, dec["dominates_both_endpoints"])

    def test_falls_back_to_the_frozen_model_when_nothing_retains(self):
        """A real outcome, not an undecided one: fine-tuning bought nothing."""
        rows = self._curve({0.0: (0.60, 0.60), 0.4: (0.70, 0.10),
                            1.0: (0.78, 0.05)})
        dec = choose_operating_point(rows)
        self.assertEqual(0.0, dec["chosen"])
        self.assertTrue(dec["fell_back_to_frozen"])
        self.assertIn("Reduce the step budget", dec["reason"])
        self.assertIn("FELL BACK", format_sweep(dec))

    def test_an_impossible_floor_yields_no_choice(self):
        rows = self._curve({0.0: (0.60, 0.60), 1.0: (0.78, 0.30)})
        dec = choose_operating_point(rows, min_retention_fraction=1.5)
        self.assertIsNone(dec["chosen"])
        self.assertIn("exceeds even the frozen", dec["reason"])

    def test_default_alphas_span_both_endpoints(self):
        self.assertIn(0.0, DEFAULT_ALPHAS)
        self.assertIn(1.0, DEFAULT_ALPHAS)

    def test_formats_without_error(self):
        rows = self._curve({0.0: (0.6, 0.6), 0.4: (0.72, 0.62),
                            1.0: (0.78, 0.3)})
        text = format_sweep(choose_operating_point(rows))
        self.assertIn("chosen", text)


class TestDatasetDescriptor(unittest.TestCase):
    def test_zero_indexed_names_match_export_order(self):
        text = render_dataset_yaml("/data", ["beaker", "centrifuge"])
        self.assertIn("  0: beaker", text)
        self.assertIn("  1: centrifuge", text)
        self.assertIn("nc: 2", text)

    def test_warns_in_the_file_about_the_index_offset(self):
        text = render_dataset_yaml("/data", ["beaker"])
        self.assertIn("zero-indexed", text)

    def test_empty_class_list_raises(self):
        with self.assertRaises(ValueError):
            render_dataset_yaml("/data", [])

    def test_a_name_that_would_break_the_format_raises(self):
        with self.assertRaises(ValueError):
            render_dataset_yaml("/data", ["bad: name"])

    def test_optional_test_split(self):
        text = render_dataset_yaml("/data", ["a"], test="images/test")
        self.assertIn("test: images/test", text)


class TestCrosscheck(unittest.TestCase):
    def test_declares_itself_unavailable_here(self):
        from vlmlab.eval.crosscheck import available
        self.assertFalse(available())

    def test_raises_a_clear_error_rather_than_failing_obscurely(self):
        from vlmlab.eval.crosscheck import (CrosscheckUnavailable,
                                            compare_average_precision)
        with self.assertRaises(CrosscheckUnavailable) as ctx:
            compare_average_precision([], [], ["a"])
        self.assertIn("tested reference", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
