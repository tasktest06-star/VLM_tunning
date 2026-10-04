import json
import os
import tempfile
import unittest

from vlmlab.config import (Config, ConfigError, FROZEN_PATHS, FrozenFieldError,
                           Provenance)


class TestFrozenFields(unittest.TestCase):
    """Each frozen path is a library default that is wrong and silent."""

    def test_every_frozen_path_is_refused(self):
        bogus = {
            "detector.attn_implementation": "eager",
            "detector.quant_type": "fp4",
            "detector.quant_double": False,
            "detector.compute_dtype": "float32",
            "detector.allow_flash_attention": True,
            "detector.require_detection_head_boxes": False,
            "eval.batch_size": 8,
            "eval.per_class_thresholds": True,
        }
        for path in sorted(FROZEN_PATHS):
            section, field = path.split(".", 1)
            with self.assertRaises(FrozenFieldError,
                                   msg="{} was not refused".format(path)):
                Config.from_dict({section: {field: bogus[path]}})

    def test_the_refusal_explains_why(self):
        try:
            Config.from_dict({"eval": {"batch_size": 8}})
        except FrozenFieldError as exc:
            self.assertIn("out-of-memory", str(exc))

    def test_unsafe_escape_exists_only_for_tests(self):
        cfg = Config.from_dict({"eval": {"batch_size": 8}}, allow_unsafe=True)
        self.assertEqual(8, cfg.eval.batch_size)

    def test_non_frozen_fields_are_settable(self):
        cfg = Config.from_dict({"detector": {"keyframe_stride": 16}})
        self.assertEqual(16, cfg.detector.keyframe_stride)


class TestValidation(unittest.TestCase):
    def test_frame_size_has_no_default_and_must_be_set(self):
        cfg = Config()
        self.assertIsNone(cfg.detector.frame_size)
        with self.assertRaises(ConfigError) as ctx:
            cfg.validate()
        self.assertIn("frame_size", str(ctx.exception))

    def test_validates_once_frame_size_is_given(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        self.assertIs(cfg, cfg.validate())

    def test_tampered_frozen_value_is_caught_at_validation(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        cfg.detector.attn_implementation = "eager"
        with self.assertRaises(ConfigError):
            cfg.validate()

    def test_unknown_setting_refused(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"detector": {"nonexistent": 1}})
        with self.assertRaises(ConfigError):
            Config.from_dict({"nonexistent_section": {}})


class TestThreeMinuteClips(unittest.TestCase):
    def test_six_chunks_minimum(self):
        cfg = Config()
        cfg.detector.frame_size = 448
        self.assertEqual(180.0, cfg.clip_duration_s)
        self.assertGreaterEqual(cfg.min_chunks_per_clip(), 6)

    def test_keyframe_estimate(self):
        cfg = Config()
        self.assertEqual(45, cfg.estimated_keyframes_per_clip())

    def test_shorter_clips_need_fewer_chunks(self):
        cfg = Config()
        cfg.clip_duration_s = 20.0
        self.assertEqual(1, cfg.min_chunks_per_clip())


class TestProvenance(unittest.TestCase):
    def test_every_tier_is_populated(self):
        report = Config().provenance_report()
        for tier in (Provenance.VERIFIED, Provenance.ESTIMATED,
                     Provenance.UNMEASURED):
            self.assertIn(tier.value, report)
        self.assertGreater(len(report[Provenance.VERIFIED.value]), 10)

    def test_unmeasured_values_are_declared_not_hidden(self):
        report = Config().provenance_report()
        unmeasured = {p[0] for p in report[Provenance.UNMEASURED.value]}
        self.assertIn("detector.peak_vram_gb", unmeasured,
                      "peak memory on a 12 GB card is unmeasured anywhere and "
                      "must be declared as such")

    def test_sources_are_recorded(self):
        report = Config().provenance_report()
        for tier in report.values():
            for path, _value, source, _note in tier:
                self.assertTrue(source, "{} has no source".format(path))


class TestLoraDerivation(unittest.TestCase):
    def test_alpha_is_twice_the_rank(self):
        cfg = Config()
        self.assertEqual(2 * cfg.lora.rank, cfg.lora.resolved_alpha())

    def test_explicit_alpha_wins(self):
        cfg = Config()
        cfg.lora.alpha = 99
        self.assertEqual(99, cfg.lora.resolved_alpha())


class TestRoundTrip(unittest.TestCase):
    def test_from_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"detector": {"frame_size": 448, "keyframe_stride": 4}}, fh)
            path = fh.name
        try:
            cfg = Config.from_json(path)
            self.assertEqual(448, cfg.detector.frame_size)
            self.assertEqual(4, cfg.detector.keyframe_stride)
            cfg.validate()
        finally:
            os.unlink(path)

    def test_underscore_keys_are_treated_as_comments(self):
        cfg = Config.from_dict({
            "_comment": ["JSON has no comment syntax"],
            "detector": {"_note": "explain a choice", "frame_size": 448}})
        self.assertEqual(448, cfg.detector.frame_size)

    def test_real_typos_are_still_refused(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"detector": {"frame_sizee": 448}})

    def test_lists_become_tuples(self):
        cfg = Config.from_dict({"lora": {"target_modules": ["q_proj", "v_proj"]}})
        self.assertEqual(("q_proj", "v_proj"), cfg.lora.target_modules)


if __name__ == "__main__":
    unittest.main()
