"""The tests that matter most: session leakage, and a control proving they work."""
import unittest

from vlmlab.splits import (TooFewGroupsError, balance_report, choose_fold_scheme,
                           group_kfold, grouped_vs_ungrouped_gap,
                           leave_one_group_out, leaky_clip_kfold)


def _items(n_sessions, clips_per_session):
    out = []
    for s in range(n_sessions):
        for c in range(clips_per_session):
            out.append({"clip": "s{}_c{}".format(s, c), "session": "s{}".format(s),
                        "labels": ["beaker"] if (s + c) % 2 == 0 else ["centrifuge"]})
    return out


class TestGroupedSplitNeverLeaks(unittest.TestCase):
    """The single most important test in the suite."""

    def test_no_session_in_two_folds_across_many_seeds(self):
        items = _items(20, 8)
        for seed in range(25):
            folds = group_kfold(items, lambda i: i["session"], 5, seed=seed)
            for fold_i, (train, val) in enumerate(folds):
                tr = {i["session"] for i in train}
                va = {i["session"] for i in val}
                self.assertEqual(set(), tr & va,
                                 "seed {} fold {} leaks sessions {}"
                                 .format(seed, fold_i, tr & va))

    def test_each_session_validated_exactly_once(self):
        items = _items(20, 8)
        folds = group_kfold(items, lambda i: i["session"], 5, seed=0)
        seen = {}
        for fold_i, (_train, val) in enumerate(folds):
            for s in {i["session"] for i in val}:
                seen.setdefault(s, []).append(fold_i)
        self.assertEqual(20, len(seen))
        for s, folds_in in seen.items():
            self.assertEqual(1, len(folds_in), "{} validated in {}".format(s, folds_in))

    def test_every_item_appears(self):
        items = _items(12, 5)
        folds = group_kfold(items, lambda i: i["session"], 4, seed=3)
        for train, val in folds:
            self.assertEqual(len(items), len(train) + len(val))

    def test_negative_control_the_leaky_splitter_does_leak(self):
        """Proves the leakage assertion above has power.

        If this test did not fail for the leaky splitter, the test above would
        not catch a real mistake either.
        """
        items = _items(6, 4)
        folds = leaky_clip_kfold(items, lambda i: i["clip"], 4, seed=0)
        leaked = False
        for train, val in folds:
            tr = {i["session"] for i in train}
            va = {i["session"] for i in val}
            if tr & va:
                leaked = True
        self.assertTrue(leaked,
                        "a clip-grouped split should leak sessions; if it does not, "
                        "the leakage detector has no power")

    def test_seed_is_deterministic(self):
        items = _items(10, 3)
        a = group_kfold(items, lambda i: i["session"], 5, seed=7)
        b = group_kfold(items, lambda i: i["session"], 5, seed=7)
        self.assertEqual([[i["clip"] for i in v] for _t, v in a],
                         [[i["clip"] for i in v] for _t, v in b])

    def test_different_seeds_differ(self):
        items = _items(12, 3)
        a = group_kfold(items, lambda i: i["session"], 4, seed=1)
        b = group_kfold(items, lambda i: i["session"], 4, seed=2)
        self.assertNotEqual([[i["clip"] for i in v] for _t, v in a],
                            [[i["clip"] for i in v] for _t, v in b])


class TestGuards(unittest.TestCase):
    def test_too_few_groups_raises(self):
        items = _items(3, 10)
        with self.assertRaises(TooFewGroupsError):
            group_kfold(items, lambda i: i["session"], 2)

    def test_more_folds_than_groups_raises(self):
        items = _items(5, 4)
        with self.assertRaises(TooFewGroupsError):
            group_kfold(items, lambda i: i["session"], 8, min_groups=2)

    def test_n_splits_below_two_raises(self):
        with self.assertRaises(ValueError):
            group_kfold(_items(5, 2), lambda i: i["session"], 1, min_groups=2)


class TestFoldScheme(unittest.TestCase):
    def test_few_sessions_uses_leave_one_out(self):
        items = _items(8, 5)
        folds, scheme = choose_fold_scheme(items, lambda i: i["session"],
                                           loso_below=12, n_folds=5)
        self.assertEqual("leave-one-session-out", scheme)
        self.assertEqual(8, len(folds))

    def test_many_sessions_uses_grouped_folds(self):
        items = _items(18, 5)
        folds, scheme = choose_fold_scheme(items, lambda i: i["session"],
                                           loso_below=12, n_folds=5)
        self.assertEqual("grouped-5-fold", scheme)
        self.assertEqual(5, len(folds))

    def test_leave_one_out_holds_exactly_one_session(self):
        items = _items(6, 4)
        for train, val in leave_one_group_out(items, lambda i: i["session"]):
            self.assertEqual(1, len({i["session"] for i in val}))


class TestBalanceReport(unittest.TestCase):
    def test_warns_on_fold_without_positives(self):
        """A class confined to one session cannot appear in every fold."""
        items = [{"clip": "c{}".format(i), "session": "s{}".format(i),
                  "labels": ["beaker", "rare"] if i == 0 else ["beaker"]}
                 for i in range(8)]
        folds = group_kfold(items, lambda i: i["session"], 4, seed=0)
        rep = balance_report(folds, lambda i: i["labels"])
        self.assertTrue(any("no validation positives" in w
                            for w in rep["warnings"]),
                        "expected a warning; got {}".format(rep["warnings"]))
        self.assertEqual(1, rep["totals"]["rare"])

    def test_warns_when_class_is_unfalsifiable(self):
        items = [{"clip": "c{}".format(i), "session": "s{}".format(i),
                  "labels": ["rare"] if i == 0 else ["common"]}
                 for i in range(8)]
        folds = group_kfold(items, lambda i: i["session"], 4, seed=0)
        rep = balance_report(folds, lambda i: i["labels"])
        self.assertTrue(any("unfalsifiable" in w for w in rep["warnings"]))


class TestLeakageGap(unittest.TestCase):
    def test_planted_session_effect_inflates_the_ungrouped_score(self):
        """Turns the research's central warning into a regression test.

        Each session gets its own signature value. A model that can memorise a
        session scores well when sessions straddle the split and badly when
        they do not. Grouping must therefore produce the lower score.
        """
        items = _items(8, 6)
        for it in items:
            it["signature"] = it["session"]

        def score(train, val):
            # Stands in for a model that matches a session signature, which is
            # exactly what a frozen-feature model does with a shared background.
            known = {i["signature"] for i in train}
            if not val:
                return 0.0
            return sum(1 for i in val if i["signature"] in known) / float(len(val))

        gap = grouped_vs_ungrouped_gap(items, lambda i: i["session"],
                                       lambda i: i["clip"], score,
                                       n_splits=4, seed=0)
        self.assertAlmostEqual(0.0, gap["grouped_mean"], places=6)
        self.assertGreater(gap["ungrouped_mean"], 0.9)
        self.assertGreater(gap["gap"], 0.9)


if __name__ == "__main__":
    unittest.main()
