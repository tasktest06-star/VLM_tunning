"""The fake backends must satisfy the same contract the real adapters will.

This is the file that makes the dependency-free design defensible: the
assertions here are literally the ones a GPU machine will run against SAM 3
and SAM 2, via the same mixins.
"""
import unittest

from tests.contract import (CropClassifierContractMixin, DetectorContractMixin,
                            PropagatorContractMixin)
from vlmlab.backends.base import PlainPythonViolation
from vlmlab.backends.fake import (FakeCropClassifier, FakeDetector,
                                  FakeMaskDetector, FakePropagator)


class TestFakeDetectorContract(DetectorContractMixin, unittest.TestCase):
    def make_detector(self):
        return FakeDetector(hit_rate=1.0)


class TestFakeMaskDetectorContract(DetectorContractMixin, unittest.TestCase):
    def make_detector(self):
        return FakeMaskDetector(hit_rate=1.0)


class TestFakePropagatorContract(PropagatorContractMixin, unittest.TestCase):
    def make_propagator(self):
        return FakePropagator()


class TestFakeCropClassifierContract(CropClassifierContractMixin, unittest.TestCase):
    def make_classifier(self):
        return FakeCropClassifier()


class TestFakeSpecifics(unittest.TestCase):
    def test_call_log_records_prompts(self):
        """Lets a test verify the per-clip vocabulary restriction was honoured."""
        d = FakeDetector(hit_rate=1.0)
        d.detect("f1.jpg", ("beaker", "centrifuge"))
        self.assertEqual(1, len(d.call_log))
        self.assertEqual(["beaker", "centrifuge"], d.call_log[0]["prompts"])

    def test_hit_rate_zero_finds_nothing(self):
        d = FakeDetector(hit_rate=0.0)
        self.assertEqual((), d.detect("f1.jpg", ("beaker",)))

    def test_mask_detector_is_refused_by_the_head_guard(self):
        """A backend that lies about its box source must fail loudly."""
        class Liar(FakeMaskDetector):
            boxes_from_detection_head = True  # claims a head, returns mask boxes
        with self.assertRaises(PlainPythonViolation):
            Liar(hit_rate=1.0).detect("f1.jpg", ("beaker",))

    def test_sessions_are_released(self):
        p = FakePropagator()
        h = p.open_session("c1", ("f0.jpg", "f1.jpg"))
        self.assertEqual(1, p.open_handles())
        p.close_session(h)
        self.assertEqual(0, p.open_handles())


if __name__ == "__main__":
    unittest.main()
