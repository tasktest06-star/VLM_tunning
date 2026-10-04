import unittest

from vlmlab.config import LoraConfig
from vlmlab.train.schedule import (ScheduleError, effective_batch_size,
                                   plan_schedule, rank_for_label_budget,
                                   resolved_alpha, steps_per_epoch,
                                   total_steps, warmup_steps)


class TestAlpha(unittest.TestCase):
    def test_twice_the_rank(self):
        for r in (4, 8, 16, 32):
            self.assertEqual(2 * r, resolved_alpha(r))

    def test_explicit_override(self):
        self.assertEqual(99, resolved_alpha(8, 99))

    def test_bad_rank_raises(self):
        with self.assertRaises(ScheduleError):
            resolved_alpha(0)


class TestRankByBudget(unittest.TestCase):
    def test_small_budget_gets_attention_only(self):
        rank, modules, _note = rank_for_label_budget(150)
        self.assertEqual(8, rank)
        self.assertEqual(4, len(modules))
        self.assertTrue(all("proj" in m for m in modules))

    def test_feed_forward_appears_at_larger_budgets(self):
        _rank, modules, _note = rank_for_label_budget(800)
        self.assertIn("gate_proj", modules)

    def test_rank_increases_monotonically(self):
        ranks = [rank_for_label_budget(n)[0] for n in (40, 150, 800, 3000, 9000)]
        self.assertEqual(ranks, sorted(ranks))

    def test_very_small_budget_advises_against_fine_tuning(self):
        _rank, _modules, note = rank_for_label_budget(30)
        self.assertIn("frozen-feature", note)


class TestBatchArithmetic(unittest.TestCase):
    def test_effective_batch(self):
        self.assertEqual(16, effective_batch_size(1, 16, 1))
        self.assertEqual(32, effective_batch_size(1, 16, 2))

    def test_two_devices_double_the_effective_batch(self):
        """The trap: a launcher does this silently."""
        self.assertEqual(2 * effective_batch_size(1, 16, 1),
                         effective_batch_size(1, 16, 2))

    def test_steps_per_epoch(self):
        self.assertEqual(7, steps_per_epoch(120, 1, 16, 1))
        # Two devices halve the optimiser steps.
        self.assertEqual(3, steps_per_epoch(120, 1, 16, 2))

    def test_at_least_one_step(self):
        self.assertEqual(1, steps_per_epoch(4, 1, 16, 1))

    def test_total_and_warmup(self):
        total = total_steps(120, 1, 16, 2, 1)
        self.assertEqual(14, total)
        self.assertEqual(1, warmup_steps(total, 0.03))

    def test_bad_warmup_ratio_raises(self):
        with self.assertRaises(ScheduleError):
            warmup_steps(100, 1.0)

    def test_bad_batch_raises(self):
        with self.assertRaises(ScheduleError):
            effective_batch_size(0, 1, 1)


class TestPlanSchedule(unittest.TestCase):
    def test_refuses_more_than_one_visible_device(self):
        with self.assertRaises(ScheduleError) as ctx:
            plan_schedule(120, LoraConfig(), n_visible_devices=2)
        self.assertIn("Pin exactly one", str(ctx.exception))

    def test_warns_about_too_few_steps(self):
        plan = plan_schedule(120, LoraConfig())
        self.assertTrue(any("optimiser steps per epoch" in w
                            for w in plan["warnings"]))

    def test_warns_about_too_many_epochs(self):
        cfg = LoraConfig()
        cfg.epochs = 10
        plan = plan_schedule(150, cfg)
        self.assertTrue(any("Overfitting" in w for w in plan["warnings"]))

    def test_warns_about_excessive_rank(self):
        cfg = LoraConfig()
        cfg.rank = 64
        plan = plan_schedule(150, cfg)
        self.assertTrue(any("exceeds the recommended" in w
                            for w in plan["warnings"]))

    def test_warns_when_alpha_is_not_twice_the_rank(self):
        cfg = LoraConfig()
        cfg.alpha = 8
        cfg.rank = 8
        plan = plan_schedule(150, cfg)
        self.assertTrue(any("not twice the rank" in w for w in plan["warnings"]))

    def test_warns_when_the_vision_tower_is_trainable(self):
        cfg = LoraConfig()
        cfg.freeze_vision_tower = False
        plan = plan_schedule(150, cfg)
        self.assertTrue(any("vision tower" in w for w in plan["warnings"]))

    def test_warns_when_fused_cross_entropy_is_not_required(self):
        cfg = LoraConfig()
        cfg.require_fused_cross_entropy = False
        plan = plan_schedule(150, cfg)
        self.assertTrue(any("fused cross entropy" in w for w in plan["warnings"]))

    def test_a_sane_configuration_still_reports_the_step_warning(self):
        """At this label budget, few steps is unavoidable and worth saying."""
        cfg = LoraConfig()
        cfg.grad_accum_steps = 4
        plan = plan_schedule(120, cfg)
        self.assertEqual(30, plan["steps_per_epoch"])
        self.assertFalse(any("optimiser steps per epoch" in w
                             for w in plan["warnings"]))


if __name__ == "__main__":
    unittest.main()
