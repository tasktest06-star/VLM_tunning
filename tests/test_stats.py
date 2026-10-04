"""Statistics, including a genuine coverage test of the bootstrap.

The coverage test is the only one that catches an off-by-one in the percentile
index, or the much worse mistake of resampling records instead of clusters.
"""
import os
import random
import statistics
import unittest

from vlmlab.eval.stats import (bootstrap_ci, choose_test, design_effect_from_icc,
                               effective_sample_size, icc_one_way,
                               kish_design_effect, mde_paired,
                               paired_group_permutation_test, post_hoc_power,
                               spread_vs_depth)

SLOW = os.environ.get("VLMLAB_SLOW") == "1"


def _mean(records):
    return (sum(records) / float(len(records))) if records else None


class TestKish(unittest.TestCase):
    def test_equal_sizes_give_exactly_one(self):
        self.assertEqual(1.0, kish_design_effect([1] * 10))
        self.assertEqual(1.0, kish_design_effect([7] * 5))

    def test_skewed_sizes_inflate(self):
        self.assertGreater(kish_design_effect([10, 1, 1]), 1.5)

    def test_effective_size_never_exceeds_total(self):
        for sizes in ([1] * 10, [10, 1, 1], [5, 5, 1]):
            self.assertLessEqual(effective_sample_size(sizes), sum(sizes) + 1e-9)

    def test_empty(self):
        self.assertEqual(1.0, kish_design_effect([]))


class TestIcc(unittest.TestCase):
    def test_no_within_variance_gives_high_correlation(self):
        groups = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]]
        self.assertGreater(icc_one_way(groups), 0.9)

    def test_no_between_variance_gives_zero(self):
        rng = random.Random(0)
        groups = [[rng.gauss(0, 1) for _ in range(30)] for _ in range(6)]
        icc = icc_one_way(groups)
        self.assertLess(icc, 0.2)

    def test_single_group_returns_none(self):
        self.assertIsNone(icc_one_way([[1.0, 2.0]]))

    def test_design_effect_from_correlation(self):
        self.assertAlmostEqual(1.0, design_effect_from_icc(1, 0.6))
        self.assertAlmostEqual(2.8, design_effect_from_icc(4, 0.6), places=6)


class TestBootstrap(unittest.TestCase):
    def test_point_estimate_uses_pooled_records(self):
        units = [[1.0, 2.0], [3.0, 4.0]]
        res = bootstrap_ci(units, _mean, n_resamples=50, seed=0)
        self.assertAlmostEqual(2.5, res.point)

    def test_interval_brackets_the_point(self):
        rng = random.Random(1)
        units = [[rng.gauss(5, 1) for _ in range(6)] for _ in range(20)]
        res = bootstrap_ci(units, _mean, n_resamples=500, seed=0)
        self.assertLessEqual(res.lo, res.point)
        self.assertLessEqual(res.point, res.hi)

    def test_deterministic_given_a_seed(self):
        units = [[float(i)] * 3 for i in range(10)]
        a = bootstrap_ci(units, _mean, n_resamples=200, seed=5)
        b = bootstrap_ci(units, _mean, n_resamples=200, seed=5)
        self.assertEqual((a.lo, a.hi), (b.lo, b.hi))

    def test_degenerate_replicates_are_counted(self):
        def sometimes_none(records):
            return None if len(records) < 3 else _mean(records)
        units = [[1.0], [2.0]]
        res = bootstrap_ci(units, sometimes_none, n_resamples=100, seed=0)
        self.assertGreater(res.n_degenerate, 0)
        self.assertFalse(res.trustworthy)

    def test_empty_units(self):
        res = bootstrap_ci([], _mean, n_resamples=10)
        self.assertIsNone(res.point)

    def test_empirical_coverage_is_near_nominal(self):
        """The test that catches a percentile off-by-one, or resampling records.

        Clustered normal data with a known mean. A correct 95 percent interval
        should contain the truth about 95 percent of the time.
        """
        n_sim = 400 if SLOW else 60
        n_clusters, per_cluster = 15, 5
        true_mean = 10.0
        covered = 0
        for sim in range(n_sim):
            rng = random.Random(1000 + sim)
            units = []
            for _ in range(n_clusters):
                offset = rng.gauss(0, 1.5)       # cluster effect
                units.append([true_mean + offset + rng.gauss(0, 0.5)
                              for _ in range(per_cluster)])
            res = bootstrap_ci(units, _mean, n_resamples=300, seed=sim)
            if res.lo is not None and res.lo <= true_mean <= res.hi:
                covered += 1
        rate = covered / float(n_sim)
        self.assertGreater(rate, 0.84, "coverage {:.3f} is too low".format(rate))
        self.assertLess(rate, 1.0001)


class TestPermutation(unittest.TestCase):
    def test_identical_systems_give_p_of_one(self):
        units = [[float(i)] * 4 for i in range(9)]
        res = paired_group_permutation_test(units, [list(u) for u in units], _mean)
        self.assertEqual(0.0, res.diff)
        self.assertAlmostEqual(1.0, res.p_value, places=6)

    def test_exact_below_the_limit(self):
        units = [[float(i)] for i in range(9)]
        res = paired_group_permutation_test(units, [[x - 1] for x in
                                                    [u[0] for u in units]], _mean)
        self.assertTrue(res.exact)
        self.assertEqual(512, res.n_perm)

    def test_consistent_dominance_hits_the_floor(self):
        units_a = [[10.0] for _ in range(9)]
        units_b = [[1.0] for _ in range(9)]
        res = paired_group_permutation_test(units_a, units_b, _mean)
        self.assertAlmostEqual(res.p_floor * 3, res.p_value, delta=res.p_floor * 3)
        self.assertLess(res.p_value, 0.01)

    def test_p_floor_is_honest_about_what_it_can_report(self):
        units = [[1.0] for _ in range(9)]
        res = paired_group_permutation_test(units, [[2.0] for _ in range(9)], _mean)
        self.assertGreater(res.p_floor, 0.001,
                           "nine groups cannot report p below about 0.004")

    def test_mismatched_group_counts_raise(self):
        with self.assertRaises(ValueError):
            paired_group_permutation_test([[1.0]], [[1.0], [2.0]], _mean)

    def test_sampling_above_the_exact_limit(self):
        units = [[float(i)] for i in range(16)]
        res = paired_group_permutation_test(units, [[x[0] + 1] for x in units],
                                            _mean, n_perm=200, seed=0)
        self.assertFalse(res.exact)
        self.assertLessEqual(res.n_perm, 200)


class TestMde(unittest.TestCase):
    def test_bonferroni_widens_the_detectable_difference(self):
        one = mde_paired(5.0, 15, n_comparisons=1).mde
        ten = mde_paired(5.0, 15, n_comparisons=10).mde
        self.assertGreater(ten, one)

    def test_more_units_narrow_it(self):
        self.assertGreater(mde_paired(5.0, 10).mde, mde_paired(5.0, 40).mde)

    def test_zero_variance_is_perfectly_detectable(self):
        self.assertEqual(0.0, mde_paired(0.0, 10).mde)

    def test_too_few_units_raises(self):
        with self.assertRaises(ValueError):
            mde_paired(1.0, 1)

    def test_power_rises_with_effect_size(self):
        small = post_hoc_power(1.0, 5.0, 15)
        large = post_hoc_power(10.0, 5.0, 15)
        self.assertLess(small, large)
        self.assertGreater(large, 0.8)


class TestTestChoice(unittest.TestCase):
    def test_few_groups_use_permutation(self):
        self.assertEqual("permutation", choose_test(9)[0])

    def test_many_groups_use_bootstrap(self):
        self.assertEqual("bootstrap", choose_test(20)[0])

    def test_boundary(self):
        self.assertEqual("permutation", choose_test(14)[0])
        self.assertEqual("bootstrap", choose_test(15)[0])


class TestSpreadVsDepth(unittest.TestCase):
    def test_spreading_across_clips_beats_depth_at_equal_cost(self):
        """The opposite of the natural instinct, so worth a regression test."""
        wide = spread_vs_depth(300, 1, 0.6)
        narrow = spread_vs_depth(30, 10, 0.6)
        self.assertEqual(wide["total_frames"], narrow["total_frames"])
        self.assertGreater(wide["n_eff"], narrow["n_eff"] * 5)

    def test_zero_correlation_means_no_penalty(self):
        s = spread_vs_depth(50, 4, 0.0)
        self.assertAlmostEqual(float(s["total_frames"]), s["n_eff"])


if __name__ == "__main__":
    unittest.main()
