import unittest

from vlmlab.config import Config
from vlmlab.preflight import (check_concept_budget, check_config_frozen_fields,
                              check_frame_grid, check_no_flash_attention,
                              check_provenance, check_single_visible_device,
                              format_report, run_pure_checks)
from vlmlab.frames import plan_frame_grid
from vlmlab.prompts import build_clip_prompt_plan
from vlmlab.registry import Registry
from vlmlab.types import ClipRecord


def _cfg():
    cfg = Config()
    cfg.detector.frame_size = 448
    return cfg


class TestVisibleDevices(unittest.TestCase):
    def test_unset_fails(self):
        self.assertFalse(check_single_visible_device({}).ok)

    def test_empty_fails(self):
        self.assertFalse(check_single_visible_device(
            {"CUDA_VISIBLE_DEVICES": ""}).ok)

    def test_one_passes(self):
        self.assertTrue(check_single_visible_device(
            {"CUDA_VISIBLE_DEVICES": "0"}).ok)

    def test_two_fails(self):
        f = check_single_visible_device({"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertFalse(f.ok)
        self.assertIn("shard clips", f.detail)


class TestFlashAttention(unittest.TestCase):
    def test_imported_fails(self):
        self.assertFalse(check_no_flash_attention({"flash_attn"}).ok)

    def test_absent_passes(self):
        self.assertTrue(check_no_flash_attention(set()).ok)

    def test_requested_in_config_fails(self):
        cfg = _cfg()
        cfg.detector.allow_flash_attention = True
        self.assertFalse(check_no_flash_attention(set(), cfg).ok)


class TestFrozenFields(unittest.TestCase):
    def test_default_config_passes(self):
        self.assertTrue(check_config_frozen_fields(_cfg()).ok)

    def test_tampering_is_detected(self):
        cfg = _cfg()
        cfg.detector.attn_implementation = "eager"
        f = check_config_frozen_fields(cfg)
        self.assertFalse(f.ok)
        self.assertIn("attn_implementation", f.detail)

    def test_quantisation_tampering_detected(self):
        cfg = _cfg()
        cfg.detector.quant_type = "fp4"
        self.assertFalse(check_config_frozen_fields(cfg).ok)


class TestConceptBudget(unittest.TestCase):
    def _plans(self, n_labels, max_concepts):
        reg = Registry()
        labels = tuple(reg.names[:n_labels])
        clip = ClipRecord("c", "s", labels, duration_s=180.0,
                          container_frame_count=5400)
        return [build_clip_prompt_plan(clip, reg, max_concepts=max_concepts)]

    def test_within_budget_passes(self):
        f = check_concept_budget(self._plans(2, 5), 5)
        self.assertTrue(f.ok)
        self.assertIn("hours", f.detail)

    def test_over_cap_fails(self):
        plans = self._plans(10, 2)
        f = check_concept_budget(plans, 2)
        self.assertFalse(f.ok)

    def test_no_plans_is_not_an_error(self):
        self.assertTrue(check_concept_budget([], 5).ok)


class TestFrameGridCheck(unittest.TestCase):
    def test_missing_grid_fails(self):
        self.assertFalse(check_frame_grid(None).ok)

    def test_under_delivery_fails_when_required(self):
        grid = plan_frame_grid("c", 6.0, 180, 1.0, 16)
        self.assertFalse(check_frame_grid(grid, require_cap_bound=True).ok)
        self.assertTrue(check_frame_grid(grid, require_cap_bound=False).ok)


class TestProvenance(unittest.TestCase):
    def test_unmeasured_gate_is_surfaced_as_a_warning(self):
        f = check_provenance(_cfg())
        self.assertEqual("warning", f.severity)


class TestReport(unittest.TestCase):
    def test_pinned_environment_has_no_errors(self):
        findings = run_pure_checks(_cfg(), {"CUDA_VISIBLE_DEVICES": "0"}, set())
        errors = [f for f in findings if not f.ok and f.severity == "error"]
        self.assertEqual([], errors)

    def test_unpinned_environment_has_an_error(self):
        findings = run_pure_checks(_cfg(), {}, set())
        errors = [f for f in findings if not f.ok and f.severity == "error"]
        self.assertEqual(1, len(errors))

    def test_failures_are_listed_first(self):
        findings = run_pure_checks(_cfg(), {}, set())
        self.assertFalse(findings[0].ok)

    def test_report_is_formatted(self):
        text = format_report(run_pure_checks(_cfg(), {}, set()))
        self.assertIn("FAIL", text)
        self.assertIn("checks", text)


if __name__ == "__main__":
    unittest.main()
