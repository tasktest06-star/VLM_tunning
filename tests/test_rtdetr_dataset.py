import unittest

from vlmlab.train.rtdetr_dataset import CocoIndex


DOC = {
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "b.jpg", "width": 640, "height": 480},
        {"id": 3, "file_name": "c.jpg", "width": 100, "height": 100},
    ],
    "categories": [{"id": 1, "name": "beaker"}, {"id": 2, "name": "centrifuge"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 2,
         "bbox": [64, 48, 128, 96], "iscrowd": 0},
        {"id": 2, "image_id": 3, "category_id": 1,
         "bbox": [0, 0, 50, 50], "iscrowd": 1},
    ],
}


class TestIndexOffset(unittest.TestCase):
    def test_one_indexed_category_becomes_zero_indexed_label(self):
        """Easy to get wrong, and silently shifts every class by one."""
        idx = CocoIndex.from_dict(DOC)
        self.assertEqual((1,), idx.to_targets(1).labels)

    def test_normalised_centre_form(self):
        idx = CocoIndex.from_dict(DOC)
        boxes = idx.to_targets(1).boxes
        self.assertEqual(1, len(boxes))
        cx, cy, w, h = boxes[0]
        self.assertAlmostEqual(0.2, cx)
        self.assertAlmostEqual(0.2, cy)
        self.assertAlmostEqual(0.2, w)
        self.assertAlmostEqual(0.2, h)


class TestEmptyImages(unittest.TestCase):
    def test_unannotated_image_still_yields_a_target(self):
        """Skipping it would discard a usable negative and can invalidate the loss."""
        idx = CocoIndex.from_dict(DOC)
        t = idx.to_targets(2)
        self.assertTrue(t.is_empty)
        self.assertEqual((), t.boxes)
        self.assertEqual((), t.labels)

    def test_coverage_counts_empties(self):
        idx = CocoIndex.from_dict(DOC)
        cov = idx.coverage()
        self.assertEqual(3, cov["n_images"])
        self.assertEqual(2, cov["n_with_annotations"])
        self.assertEqual(1, cov["n_empty"])


class TestFiltering(unittest.TestCase):
    def test_crowd_annotations_are_excluded(self):
        idx = CocoIndex.from_dict(DOC)
        self.assertTrue(idx.to_targets(3).is_empty)

    def test_zero_area_boxes_dropped(self):
        doc = dict(DOC)
        doc = {**DOC, "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10, 10, 0, 50], "iscrowd": 0}]}
        idx = CocoIndex.from_dict(doc)
        self.assertTrue(idx.to_targets(1).is_empty)


class TestGuards(unittest.TestCase):
    def test_unknown_image_raises(self):
        idx = CocoIndex.from_dict(DOC)
        with self.assertRaises(KeyError):
            idx.to_targets(99)

    def test_zero_dimension_raises(self):
        doc = {**DOC, "images": [{"id": 1, "file_name": "a.jpg",
                                  "width": 0, "height": 480}]}
        idx = CocoIndex.from_dict(doc)
        with self.assertRaises(ValueError):
            idx.to_targets(1)

    def test_image_ids_sorted(self):
        idx = CocoIndex.from_dict(DOC)
        self.assertEqual([1, 2, 3], idx.image_ids())

    def test_class_count(self):
        self.assertEqual(2, CocoIndex.from_dict(DOC).n_classes)


if __name__ == "__main__":
    unittest.main()
