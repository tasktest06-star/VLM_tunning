import unittest

from vlmlab import federated
from vlmlab.registry import Registry
from vlmlab.types import ClipRecord


def _clip(labels=("centrifuge", "beaker")):
    return ClipRecord("c1", "s1", labels, duration_s=180.0,
                      container_frame_count=5400)


class TestDerivation(unittest.TestCase):
    def setUp(self):
        self.reg = Registry()
        self.clip = _clip()

    def test_three_way_split_is_disjoint_and_complete(self):
        queried = ["centrifuge", "beaker", "microscope", "autoclave"]
        fed = federated.derive(self.clip, queried, self.reg.names)
        self.assertEqual(set(), fed.positives & fed.negatives)
        self.assertEqual(set(), fed.positives & fed.not_exhaustive)
        self.assertEqual(set(), fed.negatives & fed.not_exhaustive)
        union = fed.positives | fed.negatives | fed.not_exhaustive
        self.assertEqual(set(self.reg.names), union)

    def test_only_queried_classes_can_be_negative(self):
        """Absence of a prompt is not evidence of absence of an object."""
        fed = federated.derive(self.clip, ["centrifuge", "beaker"],
                               self.reg.names)
        self.assertEqual(set(), fed.negatives)
        self.assertIn("microscope", fed.not_exhaustive)

    def test_a_queried_absent_class_is_negative(self):
        fed = federated.derive(self.clip, ["centrifuge", "beaker", "microscope"],
                               self.reg.names)
        self.assertIn("microscope", fed.negatives)
        self.assertTrue(fed.is_guaranteed_false_positive("microscope"))

    def test_a_clip_label_is_never_negative(self):
        fed = federated.derive(self.clip, list(self.reg.names), self.reg.names)
        for lab in self.clip.labels:
            self.assertNotIn(lab, fed.negatives)
            self.assertFalse(fed.is_guaranteed_false_positive(lab))

    def test_unknown_status_for_never_queried(self):
        fed = federated.derive(self.clip, ["centrifuge", "beaker"],
                               self.reg.names)
        self.assertEqual("unknown", fed.status("autoclave"))
        self.assertFalse(fed.is_guaranteed_false_positive("autoclave"))

    def test_label_outside_the_vocabulary_raises(self):
        clip = ClipRecord("c", "s", ("not_a_class",), duration_s=10.0,
                          container_frame_count=100)
        with self.assertRaises(federated.FederatedError):
            federated.derive(clip, [], self.reg.names)

    def test_contradictory_construction_raises(self):
        with self.assertRaises(federated.FederatedError):
            federated.FederatedLabels("c", {"a"}, {"a"}, set())
        with self.assertRaises(federated.FederatedError):
            federated.FederatedLabels("c", {"a"}, set(), {"a"})


class TestTaoFields(unittest.TestCase):
    def test_emits_both_lists(self):
        reg = Registry()
        fed = federated.derive(_clip(), ["centrifuge", "beaker", "microscope"],
                               reg.names)
        fields = fed.to_tao_fields()
        self.assertIn("neg_category_ids", fields)
        self.assertIn("not_exhaustive_category_ids", fields)
        self.assertIn("microscope", fields["neg_category_ids"])

    def test_sorted_for_stable_output(self):
        reg = Registry()
        fed = federated.derive(_clip(), list(reg.names), reg.names)
        fields = fed.to_tao_fields()
        self.assertEqual(sorted(fields["neg_category_ids"]),
                         fields["neg_category_ids"])


if __name__ == "__main__":
    unittest.main()
