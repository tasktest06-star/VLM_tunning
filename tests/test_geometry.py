import random
import unittest

from vlmlab.geometry import (area_ratio, clip_to_image, cxcywh_to_xyxy, iou,
                             min_side, nms, nms_per_class, normalise,
                             xywh_to_xyxy, xyxy_to_cxcywh, xyxy_to_xywh)
from vlmlab.types import Box


class TestIou(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(1.0, iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)))

    def test_disjoint(self):
        self.assertEqual(0.0, iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)))

    def test_touching_edges_is_zero_not_negative(self):
        self.assertEqual(0.0, iou(Box(0, 0, 2, 2), Box(2, 0, 4, 2)))
        self.assertEqual(0.0, iou(Box(0, 0, 2, 2), Box(0, 2, 2, 4)))

    def test_zero_area_does_not_divide_by_zero(self):
        self.assertEqual(0.0, iou(Box(1, 1, 1, 1), Box(0, 0, 2, 2)))
        self.assertEqual(0.0, iou(Box(1, 1, 1, 1), Box(1, 1, 1, 1)))

    def test_exact_half_overlap(self):
        """The fixture the average-precision threshold test depends on."""
        self.assertEqual(0.5, iou(Box(0, 0, 4, 4), Box(0, 0, 4, 2)))

    def test_symmetric(self):
        a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
        self.assertEqual(iou(a, b), iou(b, a))

    def test_contained(self):
        self.assertAlmostEqual(0.25, iou(Box(0, 0, 10, 10), Box(0, 0, 5, 5)))


class TestConversions(unittest.TestCase):
    def test_xywh_round_trip(self):
        b = Box(3, 5, 13, 25)
        self.assertEqual(b.as_tuple(), xywh_to_xyxy(*xyxy_to_xywh(b)).as_tuple())

    def test_cxcywh_round_trip(self):
        b = Box(4, 6, 14, 26)
        self.assertEqual(b.as_tuple(), cxcywh_to_xyxy(*xyxy_to_cxcywh(b)).as_tuple())

    def test_cxcywh_centre_is_correct(self):
        cx, cy, w, h = xyxy_to_cxcywh(Box(0, 0, 4, 2))
        self.assertEqual((2.0, 1.0, 4, 2), (cx, cy, w, h))

    def test_normalise(self):
        n = normalise(Box(0, 0, 320, 240), 640, 480)
        self.assertEqual((0.0, 0.0, 0.5, 0.5), n.as_tuple())

    def test_normalise_rejects_zero_dimension(self):
        with self.assertRaises(ValueError):
            normalise(Box(0, 0, 1, 1), 0, 10)

    def test_clip_to_image(self):
        c = clip_to_image(Box(-5, -5, 700, 500), 640, 480)
        self.assertEqual((0.0, 0.0, 640.0, 480.0), c.as_tuple())

    def test_area_ratio(self):
        self.assertAlmostEqual(0.25, area_ratio(Box(0, 0, 320, 240), 640, 480))

    def test_min_side(self):
        self.assertEqual(2, min_side(Box(0, 0, 4, 2)))


class TestNms(unittest.TestCase):
    def test_empty(self):
        self.assertEqual([], nms([], [], 0.5))

    def test_keeps_highest_of_overlapping_pair(self):
        boxes = [Box(0, 0, 10, 10), Box(1, 1, 11, 11)]
        self.assertEqual([0], nms(boxes, [0.9, 0.5], 0.5))

    def test_keeps_both_when_disjoint(self):
        boxes = [Box(0, 0, 10, 10), Box(50, 50, 60, 60)]
        self.assertEqual([0, 1], nms(boxes, [0.9, 0.5], 0.5))

    def test_deterministic_across_input_order(self):
        """Equal scores must not make the result depend on input order."""
        base = None
        for seed in range(100):
            boxes = [Box(0, 0, 10, 10), Box(50, 50, 60, 60), Box(1, 1, 11, 11)]
            scores = [0.5, 0.5, 0.5]
            pairs = list(zip(boxes, scores))
            random.Random(seed).shuffle(pairs)
            b = [p[0] for p in pairs]
            s = [p[1] for p in pairs]
            kept = sorted(b[i].as_tuple() for i in nms(b, s, 0.5))
            if base is None:
                base = kept
            self.assertEqual(base, kept)

    def test_per_class_does_not_suppress_across_labels(self):
        boxes = [Box(0, 0, 10, 10), Box(1, 1, 11, 11)]
        keep = nms_per_class(boxes, [0.9, 0.8], ["a", "b"], 0.5)
        self.assertEqual([0, 1], keep)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            nms([Box(0, 0, 1, 1)], [0.5, 0.5], 0.5)


class TestBox(unittest.TestCase):
    def test_degenerate_box_refused(self):
        with self.assertRaises(ValueError):
            Box(10, 0, 0, 10)

    def test_frozen(self):
        b = Box(0, 0, 1, 1)
        with self.assertRaises(Exception):
            b.x1 = 5


if __name__ == "__main__":
    unittest.main()
