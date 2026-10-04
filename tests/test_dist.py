"""Distribution functions, checked against published tables.

These exist so the statistics layer needs no third-party library. The inverse
Student-t is the only thing in the evaluation design that genuinely wants one,
and approximating it would be the wrong trade when the resulting number
decides whether a result is reportable.
"""
import unittest

from vlmlab.eval.dist import (betainc, clopper_pearson, normal_ppf, t_cdf,
                              t_ppf, wilson)


class TestIncompleteBeta(unittest.TestCase):
    def test_uniform_case(self):
        for x in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(x, betainc(1, 1, x), places=10)

    def test_endpoints(self):
        self.assertEqual(0.0, betainc(2, 3, 0.0))
        self.assertEqual(1.0, betainc(2, 3, 1.0))

    def test_monotone(self):
        prev = -1.0
        for i in range(21):
            v = betainc(2, 5, i / 20.0)
            self.assertGreaterEqual(v, prev)
            prev = v

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            betainc(1, 1, 1.5)


class TestStudentT(unittest.TestCase):
    TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 8: 2.306,
             10: 2.228, 15: 2.131, 19: 2.093, 20: 2.086, 30: 2.042,
             60: 2.000, 100: 1.984}

    def test_matches_published_two_sided_values(self):
        for df, expect in self.TABLE.items():
            got = t_ppf(0.975, df)
            self.assertAlmostEqual(expect, got, places=3,
                                   msg="df={} got {}".format(df, got))

    def test_cdf_at_zero_is_one_half(self):
        for df in (1, 5, 30, 500):
            self.assertAlmostEqual(0.5, t_cdf(0.0, df), places=10)

    def test_cdf_is_monotone(self):
        prev = -1.0
        for t in (-5.0, -1.0, 0.0, 1.0, 5.0):
            v = t_cdf(t, 10)
            self.assertGreater(v, prev)
            prev = v

    def test_symmetry(self):
        self.assertAlmostEqual(1.0 - t_cdf(1.5, 7), t_cdf(-1.5, 7), places=10)

    def test_large_degrees_of_freedom_approach_the_normal(self):
        self.assertAlmostEqual(normal_ppf(0.975), t_ppf(0.975, 5000), places=6)

    def test_inverse_round_trip(self):
        for df in (3, 12, 40):
            for p in (0.05, 0.5, 0.9, 0.975):
                self.assertAlmostEqual(p, t_cdf(t_ppf(p, df), df), places=6)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            t_ppf(0.0, 5)
        with self.assertRaises(ValueError):
            t_ppf(0.5, 0)


class TestClopperPearson(unittest.TestCase):
    def test_zero_successes(self):
        lo, hi = clopper_pearson(0, 10)
        self.assertEqual(0.0, lo)
        self.assertAlmostEqual(0.3085, hi, places=3)

    def test_all_successes(self):
        lo, hi = clopper_pearson(10, 10)
        self.assertAlmostEqual(0.6915, lo, places=3)
        self.assertEqual(1.0, hi)

    def test_middle(self):
        lo, hi = clopper_pearson(3, 10)
        self.assertAlmostEqual(0.0667, lo, places=3)
        self.assertAlmostEqual(0.6525, hi, places=3)

    def test_brackets_the_estimate(self):
        for k in range(0, 11):
            lo, hi = clopper_pearson(k, 10)
            self.assertLessEqual(lo, k / 10.0)
            self.assertGreaterEqual(hi, k / 10.0)

    def test_monotone_in_successes(self):
        prev_lo = -1.0
        for k in range(0, 11):
            lo, _hi = clopper_pearson(k, 10)
            self.assertGreaterEqual(lo, prev_lo)
            prev_lo = lo

    def test_narrows_with_more_trials(self):
        wide = clopper_pearson(3, 10)
        narrow = clopper_pearson(30, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_three_positives_is_effectively_unfalsifiable(self):
        """The research's point about per-class figures, made concrete."""
        lo, hi = clopper_pearson(2, 3)
        self.assertGreater(hi - lo, 0.7)

    def test_zero_trials(self):
        self.assertEqual((0.0, 1.0), clopper_pearson(0, 0))

    def test_invalid_counts_raise(self):
        with self.assertRaises(ValueError):
            clopper_pearson(11, 10)


class TestWilson(unittest.TestCase):
    def test_brackets_and_stays_in_range(self):
        for k in range(0, 11):
            lo, hi = wilson(k, 10)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_narrower_than_exact(self):
        w = wilson(3, 10)
        c = clopper_pearson(3, 10)
        self.assertLessEqual(w[1] - w[0], c[1] - c[0] + 1e-9)


if __name__ == "__main__":
    unittest.main()
