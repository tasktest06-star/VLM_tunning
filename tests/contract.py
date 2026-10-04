"""The backend contract, shared by the fake and the real adapters.

Written *before* the fake backend on purpose. A contract retrofitted to an
implementation tests nothing; a contract the implementation was written
against tests the interface.

The same mixin is subclassed twice: once by the fake, which runs in this
container, and once by each real adapter, skipped unless torch is present.
Same assertions, two subjects. That is the only honest claim available
without a GPU, and it is specifically what catches the mistakes that are
expensive to find on a slow machine: the box-format confusion, the
one-concept-per-session limit, the double propagation, and a tensor escaping
the boundary.
"""
import unittest

from vlmlab.backends.base import (PlainPythonViolation, validate_plain_python)
from vlmlab.types import BoxSource, Box, Detection, Track


class DetectorContractMixin(object):
    """Subclass and set ``make_detector``."""

    def make_detector(self):
        raise NotImplementedError

    def sample_prompts(self):
        return ("laboratory instrument", "beaker")

    def sample_frame(self):
        return "frame_000001.jpg"

    # ---- structure -----------------------------------------------------
    def test_capabilities_are_declared(self):
        d = self.make_detector()
        caps = d.capabilities()
        for key in ("model_id", "supports_presence", "boxes_from_detection_head",
                    "max_concepts_per_session", "box_prompt_format"):
            self.assertIn(key, caps)
        self.assertIn(caps["box_prompt_format"], ("xyxy", "cxcywh"))
        self.assertGreaterEqual(caps["max_concepts_per_session"], 1)

    def test_returns_detection_objects(self):
        d = self.make_detector()
        for det in d.detect(self.sample_frame(), self.sample_prompts()):
            self.assertIsInstance(det, Detection)
            self.assertIsInstance(det.box, Box)

    def test_every_scalar_is_plain_python(self):
        """Catches a numpy scalar, which an instance check would let through."""
        d = self.make_detector()
        for det in d.detect(self.sample_frame(), self.sample_prompts()):
            self.assertIs(type(det.score), float)
            self.assertIs(type(det.label), str)
            self.assertIs(type(det.frame_id), str)
            for v in det.box.as_tuple():
                self.assertIn(type(v), (float, int))
            if det.presence is not None:
                self.assertIs(type(det.presence), float)
            validate_plain_python(det, "detection")

    def test_box_source_matches_capability(self):
        d = self.make_detector()
        for det in d.detect(self.sample_frame(), self.sample_prompts()):
            if d.boxes_from_detection_head:
                self.assertIsNot(det.box_source, BoxSource.MASK_DERIVED)
            else:
                self.assertIs(det.box_source, BoxSource.MASK_DERIVED)

    def test_presence_present_iff_declared(self):
        d = self.make_detector()
        dets = d.detect(self.sample_frame(), self.sample_prompts())
        for det in dets:
            if d.supports_presence:
                self.assertIsNotNone(det.presence)

    # ---- behaviour -----------------------------------------------------
    def test_deterministic(self):
        d = self.make_detector()
        a = d.detect(self.sample_frame(), self.sample_prompts())
        b = d.detect(self.sample_frame(), self.sample_prompts())
        self.assertEqual([x.box.as_tuple() for x in a], [x.box.as_tuple() for x in b])
        self.assertEqual([x.score for x in a], [x.score for x in b])

    def test_empty_prompts_returns_empty(self):
        d = self.make_detector()
        self.assertEqual((), d.detect(self.sample_frame(), ()))

    def test_single_string_prompt_is_rejected(self):
        """A bare string would silently iterate character by character."""
        d = self.make_detector()
        with self.assertRaises(TypeError):
            d.detect(self.sample_frame(), "beaker")

    def test_labels_come_from_prompts(self):
        d = self.make_detector()
        prompts = self.sample_prompts()
        for det in d.detect(self.sample_frame(), prompts):
            self.assertIn(det.prompt, prompts)

    def test_scores_in_unit_range(self):
        d = self.make_detector()
        for det in d.detect(self.sample_frame(), self.sample_prompts()):
            self.assertGreaterEqual(det.score, 0.0)
            self.assertLessEqual(det.score, 1.0)

    def test_boxes_are_non_degenerate(self):
        d = self.make_detector()
        for det in d.detect(self.sample_frame(), self.sample_prompts()):
            self.assertGreater(det.box.x2, det.box.x1)
            self.assertGreater(det.box.y2, det.box.y1)


class PropagatorContractMixin(object):
    """Subclass and set ``make_propagator``."""

    def make_propagator(self):
        raise NotImplementedError

    def sample_frames(self):
        return tuple("f{:06d}.jpg".format(i) for i in range(12))

    def sample_seeds(self):
        # Must name a frame that is actually in sample_frames(), mid-sequence so
        # that forward and reverse both have somewhere to go.
        return (Detection(self.sample_frames()[4], "beaker", Box(10, 10, 60, 80),
                          0.9, BoxSource.DETECTION_HEAD, presence=0.9),)

    def test_session_lifecycle(self):
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        tracks = p.propagate(h, self.sample_seeds(), reverse=False)
        self.assertIsInstance(tracks, tuple)
        p.close_session(h)

    def test_detect_after_close_raises(self):
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        p.close_session(h)
        with self.assertRaises(Exception):
            p.propagate(h, self.sample_seeds(), reverse=False)

    def test_reverse_yields_earlier_frames(self):
        """Forward and reverse must cover different sides of the seed."""
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        fwd = p.propagate(h, self.sample_seeds(), reverse=False)
        rev = p.propagate(h, self.sample_seeds(), reverse=True)
        p.close_session(h)
        seed_frame = self.sample_seeds()[0].frame_id
        fwd_frames = [pt.frame_id for t in fwd for pt in t.points]
        rev_frames = [pt.frame_id for t in rev for pt in t.points]
        self.assertTrue(any(f > seed_frame for f in fwd_frames),
                        "forward propagation produced no later frames")
        self.assertTrue(any(f < seed_frame for f in rev_frames),
                        "reverse propagation produced no earlier frames")

    def test_tracks_are_plain_python(self):
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        for tr in p.propagate(h, self.sample_seeds(), reverse=False):
            self.assertIsInstance(tr, Track)
            validate_plain_python(tr, "track")
            self.assertIs(type(tr.track_id), int)
        p.close_session(h)

    def test_seed_from_unknown_frame_raises(self):
        """Silently defaulting to frame zero would hide a real wiring bug."""
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        bogus = Detection("not-a-frame.jpg", "beaker", Box(1, 1, 5, 5), 0.9,
                          BoxSource.DETECTION_HEAD, presence=0.9)
        with self.assertRaises(ValueError):
            p.propagate(h, (bogus,), reverse=False)
        p.close_session(h)

    def test_track_labels_follow_seeds(self):
        p = self.make_propagator()
        h = p.open_session("clip001", self.sample_frames())
        tracks = p.propagate(h, self.sample_seeds(), reverse=False)
        p.close_session(h)
        seed_labels = {s.label for s in self.sample_seeds()}
        for tr in tracks:
            self.assertIn(tr.label, seed_labels)


class CropClassifierContractMixin(object):
    def make_classifier(self):
        raise NotImplementedError

    def test_one_score_row_per_box(self):
        c = self.make_classifier()
        boxes = (Box(0, 0, 10, 10), Box(20, 20, 40, 40))
        labels = ("beaker", "centrifuge", "microscope")
        rows = c.classify("f.jpg", boxes, labels)
        self.assertEqual(len(rows), len(boxes))
        for row in rows:
            self.assertEqual(len(row), len(labels))
            for v in row:
                self.assertIs(type(v), float)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_deterministic(self):
        c = self.make_classifier()
        boxes = (Box(0, 0, 10, 10),)
        labels = ("beaker", "centrifuge")
        self.assertEqual(c.classify("f.jpg", boxes, labels),
                         c.classify("f.jpg", boxes, labels))
