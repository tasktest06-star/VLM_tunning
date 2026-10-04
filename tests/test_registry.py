"""The homonym guards are the real content here.

Verified from the actual dataset category files: the exhaust-hood sense is a
*frequent* class under the name a practitioner would prompt with, so the
detector is confidently trained on the wrong object. A naive substring matcher
would resolve it.
"""
import unittest

from vlmlab.registry import (AmbiguousMatch, ClassEntry, Registry, UnknownLabel,
                             levenshtein)


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(0, levenshtein("beaker", "beaker"))

    def test_empty(self):
        self.assertEqual(6, levenshtein("", "beaker"))
        self.assertEqual(0, levenshtein("", ""))

    def test_symmetric(self):
        self.assertEqual(levenshtein("flask", "flasks"),
                         levenshtein("flasks", "flask"))

    def test_known_distances(self):
        self.assertEqual(1, levenshtein("microscope", "microscop"))
        self.assertEqual(3, levenshtein("kitten", "sitting"))


class TestHomonymGuards(unittest.TestCase):
    """Each of these resolving to a laboratory class would poison the pipeline."""

    def setUp(self):
        self.r = Registry()

    def test_kitchen_range_hood_is_not_a_fume_hood(self):
        for bad in ("kitchen range hood", "exhaust hood", "range hood",
                    "cooker hood", "stove hood"):
            self.assertIsNone(self.r.resolve(bad),
                              "{!r} resolved to a laboratory class".format(bad))

    def test_condiment_shaker_is_not_an_orbital_shaker(self):
        for bad in ("condiment shaker", "salt shaker", "cocktail shaker",
                    "pepper shaker"):
            self.assertIsNone(self.r.resolve(bad))

    def test_household_scale_is_not_an_analytical_balance(self):
        for bad in ("bathroom scale", "kitchen scale", "weighing scale"):
            self.assertIsNone(self.r.resolve(bad))

    def test_thermos_is_not_a_conical_flask(self):
        for bad in ("vacuum flask", "thermos", "hip flask"):
            self.assertIsNone(self.r.resolve(bad))

    def test_coffee_mug_is_not_a_beaker(self):
        self.assertIsNone(self.r.resolve("coffee mug"))


class TestSubstringTrap(unittest.TestCase):
    def test_test_tube_holder_does_not_resolve_to_test_tube(self):
        r = Registry()
        self.assertEqual("test_tube_holder", r.resolve("test tube holder"))
        self.assertEqual("test_tube_holder", r.resolve("test tube rack"))
        self.assertEqual("test_tube", r.resolve("test tube"))

    def test_longest_match_wins_not_first(self):
        r = Registry()
        self.assertEqual("test_tube_holder",
                         r.resolve("a laboratory test tube holder on a bench"))


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.r = Registry()

    def test_exact_and_case_insensitive(self):
        self.assertEqual("beaker", self.r.resolve("Beaker"))
        self.assertEqual("beaker", self.r.resolve("  BEAKER  "))

    def test_underscores_and_spaces_interchangeable(self):
        self.assertEqual("fume_hood", self.r.resolve("fume hood"))
        self.assertEqual("fume_hood", self.r.resolve("fume_hood"))

    def test_paraphrase(self):
        self.assertEqual("conical_flask", self.r.resolve("erlenmeyer flask"))
        self.assertEqual("spectrophotometer",
                         self.r.resolve("UV-Vis spectrophotometer"))

    def test_typo_within_distance(self):
        self.assertEqual("microscope", self.r.resolve("microscop"))

    def test_unrelated_returns_none(self):
        self.assertIsNone(self.r.resolve("bicycle"))
        self.assertIsNone(self.r.resolve(""))

    def test_ambiguous_tie_raises_rather_than_guessing(self):
        r = Registry(entries=[ClassEntry("abcd"), ClassEntry("abce")])
        with self.assertRaises(AmbiguousMatch):
            r.resolve("abcf")

    def test_order_independence(self):
        """Resolution must not depend on the order classes were declared."""
        fwd = Registry()
        rev = Registry(entries=tuple(reversed(Registry().entries)))
        for probe in ("test tube holder", "erlenmeyer flask", "fume hood",
                      "graduated cylinder"):
            self.assertEqual(fwd.resolve(probe), rev.resolve(probe),
                             "{!r} resolved differently".format(probe))


class TestHardNegatives(unittest.TestCase):
    def setUp(self):
        self.r = Registry()

    def test_never_includes_a_present_label(self):
        present = {"centrifuge", "beaker"}
        negs = self.r.hard_negatives_for(present, n=5)
        self.assertEqual(set(), set(negs) & present)

    def test_deterministic(self):
        a = self.r.hard_negatives_for({"centrifuge"}, n=4)
        b = self.r.hard_negatives_for({"centrifuge"}, n=4)
        self.assertEqual(a, b)

    def test_returns_the_requested_count(self):
        for n in (1, 3, 4, 5):
            self.assertEqual(n, len(self.r.hard_negatives_for({"beaker"}, n=n)))

    def test_prefers_confusable_classes(self):
        negs = self.r.hard_negatives_for({"centrifuge"}, n=1)
        self.assertIn(negs[0], self.r.get("centrifuge").confusable_with)

    def test_unknown_label_raises(self):
        with self.assertRaises(UnknownLabel):
            self.r.hard_negatives_for({"not_a_class"})


class TestRegistryData(unittest.TestCase):
    def test_classes_without_public_boxes_are_recorded(self):
        r = Registry()
        missing = set(r.classes_without_public_boxes())
        for name in ("centrifuge", "autoclave", "orbital_shaker",
                     "spectrophotometer", "fume_hood"):
            self.assertIn(name, missing,
                          "{} has no public boxes and should be flagged".format(name))

    def test_round_trip_through_dict(self):
        r = Registry()
        r2 = Registry.from_dict(r.to_dict())
        self.assertEqual(r.names, r2.names)

    def test_duplicate_names_refused(self):
        with self.assertRaises(ValueError):
            Registry(entries=[ClassEntry("a"), ClassEntry("a")])

    def test_conflicting_alias_refused(self):
        with self.assertRaises(ValueError):
            Registry(entries=[ClassEntry("a", ("shared",)),
                              ClassEntry("b", ("shared",))])


if __name__ == "__main__":
    unittest.main()
