"""Backend interfaces. No torch at module level, so this always imports.

**The rule that makes the whole design testable: every value crossing this
boundary is plain Python.** A float, int, str, bool, list, tuple or None.
Adapters convert before returning. That is what keeps the cascade, the
filters and the metrics runnable and testable on a machine with no GPU.

Two guards are structural rather than advisory.

``detect`` is final and wraps the subclass implementation in inference mode,
so a subclass **cannot** forget it. Omitting it was measured at eight
gigabytes per frame, which is an immediate out-of-memory failure on a 12 GB
card.

The interfaces accept an already-extracted frame path and never a video path,
so decoding cannot end up inside a loop. Decoding is a separate one-time step.

Capabilities are declared rather than inferred from a model name, because the
orchestration genuinely needs to know three things: whether a presence score
is available, whether boxes come from a real detection head or are derived
from masks, and how many concepts one session accepts. The reference detector
accepts exactly one, because adding a prompt resets the session state.
"""
import abc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.types import BoxSource, Detection, Track

__all__ = [
    "PlainPythonViolation", "SessionClosedError", "validate_plain_python",
    "KeyframeDetector", "Propagator", "CropClassifier", "inference_mode",
]

_PLAIN = (float, int, str, bool)


class PlainPythonViolation(TypeError):
    """A tensor or array escaped across the backend boundary."""


class SessionClosedError(RuntimeError):
    pass


def validate_plain_python(value, path="value"):
    """Reject anything that is not a plain Python primitive.

    Uses exact type identity, not an instance check, because a numpy scalar
    passes ``isinstance(x, float)`` and would slip through.
    """
    if value is None:
        return value
    t = type(value)
    if t in _PLAIN:
        return value
    if t in (list, tuple):
        for i, item in enumerate(value):
            validate_plain_python(item, "{}[{}]".format(path, i))
        return value
    if t is dict:
        for k, v in value.items():
            validate_plain_python(k, "{} key".format(path))
            validate_plain_python(v, "{}[{!r}]".format(path, k))
        return value
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            validate_plain_python(getattr(value, name), "{}.{}".format(path, name))
        return value
    if isinstance(value, BoxSource):
        return value
    raise PlainPythonViolation(
        "{} is {}, not a plain Python primitive. Convert with float()/int()/"
        ".tolist() inside the adapter before returning."
        .format(path, t.__module__ + "." + t.__name__))


def inference_mode():
    """Context manager disabling autograd when torch is present, else a no-op.

    Lazy on purpose: this module must import with no dependencies installed.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        import contextlib
        return contextlib.nullcontext()
    return torch.inference_mode()


class KeyframeDetector(abc.ABC):
    """Detects boxes on single frames from text prompts."""

    #: True when the backend exposes a decoupled image-level presence score.
    supports_presence = False
    #: False when boxes are derived from masks, so quality is capped by mask quality.
    boxes_from_detection_head = True
    #: Concepts accepted per session. The reference detector accepts one,
    #: because adding a prompt resets the session.
    max_concepts_per_session = 1
    #: "xyxy" or "cxcywh". Getting this wrong is silent and ruinous.
    box_prompt_format = "xyxy"
    model_id = "abstract"

    def detect(self, frame_path, prompts):
        """Final. Do not override; implement ``_detect_impl`` instead."""
        if isinstance(prompts, str):
            raise TypeError("prompts must be a sequence, not a single string")
        prompts = tuple(prompts)
        if not prompts:
            return ()
        with inference_mode():
            out = self._detect_impl(str(frame_path), prompts)
        result = tuple(out)
        for i, det in enumerate(result):
            if not isinstance(det, Detection):
                raise TypeError("detection {} is {}, expected Detection"
                                .format(i, type(det).__name__))
            validate_plain_python(det, "detections[{}]".format(i))
            if self.boxes_from_detection_head and det.box_source is BoxSource.MASK_DERIVED:
                raise PlainPythonViolation(
                    "backend claims detection-head boxes but returned a mask-derived box")
            if self.supports_presence and det.presence is None:
                raise ValueError(
                    "backend declares supports_presence but returned presence=None")
        return result

    @abc.abstractmethod
    def _detect_impl(self, frame_path, prompts):
        """Return a sequence of Detection. Plain Python values only."""

    def capabilities(self):
        return {
            "model_id": self.model_id,
            "supports_presence": self.supports_presence,
            "boxes_from_detection_head": self.boxes_from_detection_head,
            "max_concepts_per_session": self.max_concepts_per_session,
            "box_prompt_format": self.box_prompt_format,
        }


class Propagator(abc.ABC):
    """Propagates seed boxes through a clip, maintaining identity."""

    model_id = "abstract"
    produces_box_source = BoxSource.PROPAGATED

    def open_session(self, clip_id, frame_paths):
        handle = self._open_impl(str(clip_id), tuple(str(p) for p in frame_paths))
        return handle

    def propagate(self, handle, seeds, reverse=False):
        """Propagate seeds. Call twice, forward then reverse.

        ``reverse`` is a parameter because the real interface takes a reverse
        flag and has no single call that does both directions. The research
        found a document claiming otherwise, and the keyword it named does not
        exist in the source.
        """
        with inference_mode():
            out = self._propagate_impl(handle, tuple(seeds), bool(reverse))
        result = tuple(out)
        for i, tr in enumerate(result):
            if not isinstance(tr, Track):
                raise TypeError("track {} is {}, expected Track"
                                .format(i, type(tr).__name__))
            validate_plain_python(tr, "tracks[{}]".format(i))
        return result

    def close_session(self, handle):
        """Release state. Required between clips: inference states accumulate."""
        self._close_impl(handle)

    @abc.abstractmethod
    def _open_impl(self, clip_id, frame_paths):
        ...

    @abc.abstractmethod
    def _propagate_impl(self, handle, seeds, reverse):
        ...

    @abc.abstractmethod
    def _close_impl(self, handle):
        ...


class CropClassifier(abc.ABC):
    """Classifies a cropped region into the fine-grained vocabulary.

    Separate from detection on purpose. Fine-grained instrument names fail as
    detection prompts, so the detector localises generically and this names
    the result.
    """

    model_id = "abstract"

    def classify(self, frame_path, boxes, labels):
        with inference_mode():
            out = self._classify_impl(str(frame_path), tuple(boxes), tuple(labels))
        result = tuple(out)
        for i, scores in enumerate(result):
            validate_plain_python(scores, "scores[{}]".format(i))
            if len(scores) != len(labels):
                raise ValueError(
                    "classifier returned {} scores for {} labels"
                    .format(len(scores), len(labels)))
        return result

    @abc.abstractmethod
    def _classify_impl(self, frame_path, boxes, labels):
        """Return one score tuple per box, aligned to ``labels``."""
