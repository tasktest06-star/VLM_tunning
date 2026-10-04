"""Deterministic, dependency-free backends.

Their purpose is not to simulate a model. It is to let the entire pipeline run
end to end on a machine with no GPU, so that orchestration, chunking, the
cascade, the exporters and the evaluation are genuinely exercised rather than
assumed to work. Output is a deterministic function of the frame identifier
and prompt, so tests can assert exact values.

Each records a ``call_log``, which lets a test verify things no return value
would reveal, such as whether the pipeline honoured the per-clip vocabulary
restriction or issued one session per concept.
"""
import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.backends.base import (CropClassifier, KeyframeDetector, Propagator,
                                  SessionClosedError)
from vlmlab.types import Box, BoxSource, Detection, Track, TrackPoint

__all__ = ["FakeDetector", "FakePropagator", "FakeCropClassifier", "FakeMaskDetector"]


def _unit(*parts):
    """Deterministic pseudo-random float in [0, 1) from the given parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


class FakeDetector(KeyframeDetector):
    """A detector whose output is a hash of its inputs.

    ``hit_rate`` controls how often a prompt produces a detection, so a test
    can create both found and missing classes deliberately.
    """

    supports_presence = True
    boxes_from_detection_head = True
    max_concepts_per_session = 8
    box_prompt_format = "xyxy"
    model_id = "fake/detector-v1"

    def __init__(self, image_size=(640, 480), hit_rate=0.7, seed="a",
                 max_per_prompt=2, coherent=True, jitter=0.004):
        self.image_size = tuple(image_size)
        self.hit_rate = float(hit_rate)
        self.seed = str(seed)
        self.max_per_prompt = int(max_per_prompt)
        # Laboratory equipment is static on a bench, so a realistic fake puts
        # the same object in the same place across frames with small jitter.
        # A spatially random fake would be rejected wholesale by the temporal
        # consistency filter and would leave every downstream stage untested.
        self.coherent = bool(coherent)
        self.jitter = float(jitter)
        self.call_log = []

    @staticmethod
    def _scene_key(frame_path):
        """Everything before the frame index, so one clip is one scene."""
        stem = str(frame_path).rsplit("/", 1)[-1]
        if "_" in stem:
            return stem.rsplit("_", 1)[0]
        return stem

    def _detect_impl(self, frame_path, prompts):
        self.call_log.append({"op": "detect", "frame": frame_path,
                              "prompts": list(prompts)})
        w, h = self.image_size
        out = []
        # Scene-stable identity when coherent, per-frame otherwise.
        anchor = self._scene_key(frame_path) if self.coherent else frame_path
        for prompt in prompts:
            if _unit(self.seed, anchor, prompt, "hit") >= self.hit_rate:
                continue
            n = 1 + int(_unit(self.seed, anchor, prompt, "n") * self.max_per_prompt)
            for k in range(n):
                jx = (_unit(self.seed, frame_path, prompt, k, "jx") - 0.5) * 2 * self.jitter
                jy = (_unit(self.seed, frame_path, prompt, k, "jy") - 0.5) * 2 * self.jitter
                cx = 0.1 + 0.8 * _unit(self.seed, anchor, prompt, k, "cx") + jx
                cy = 0.1 + 0.8 * _unit(self.seed, anchor, prompt, k, "cy") + jy
                bw = 0.10 + 0.18 * _unit(self.seed, anchor, prompt, k, "w")
                bh = 0.10 + 0.18 * _unit(self.seed, anchor, prompt, k, "h")
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                    continue
                score = 0.35 + 0.6 * _unit(self.seed, anchor, prompt, k, "s")
                presence = 0.3 + 0.69 * _unit(self.seed, anchor, prompt, "p")
                out.append(Detection(
                    frame_id=str(frame_path), label=str(prompt).replace(" ", "_"),
                    box=Box(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score), box_source=BoxSource.DETECTION_HEAD,
                    presence=float(presence), prompt=str(prompt),
                    model_id=self.model_id))
        return tuple(out)


class FakeMaskDetector(FakeDetector):
    """Same, but declares mask-derived boxes.

    Exists so a test can prove the pipeline refuses it when boxes are the
    deliverable, which is the guard protecting against silently inflated boxes
    from a fragmented mask.
    """

    boxes_from_detection_head = False
    model_id = "fake/mask-detector-v1"

    def _detect_impl(self, frame_path, prompts):
        dets = super(FakeMaskDetector, self)._detect_impl(frame_path, prompts)
        return tuple(
            Detection(d.frame_id, d.label, d.box, d.score, BoxSource.MASK_DERIVED,
                      d.presence, d.prompt, self.model_id)
            for d in dets)


class _Session(object):
    def __init__(self, clip_id, frame_paths):
        self.clip_id = clip_id
        self.frames = tuple(frame_paths)
        self.open = True
        self.next_track_id = 1


class FakePropagator(Propagator):
    """Propagates a seed box with a small deterministic drift."""

    model_id = "fake/propagator-v1"

    def __init__(self, drift=1.5, decay=0.01, seed="b"):
        self.drift = float(drift)
        self.decay = float(decay)
        self.seed = str(seed)
        self.call_log = []
        self._sessions = {}
        self._counter = 0

    def _open_impl(self, clip_id, frame_paths):
        self._counter += 1
        handle = "{}#{}".format(clip_id, self._counter)
        self._sessions[handle] = _Session(clip_id, frame_paths)
        self.call_log.append({"op": "open", "clip": clip_id,
                              "n_frames": len(frame_paths), "handle": handle})
        return handle

    def _propagate_impl(self, handle, seeds, reverse):
        sess = self._sessions.get(handle)
        if sess is None or not sess.open:
            raise SessionClosedError("session {!r} is closed".format(handle))
        self.call_log.append({"op": "propagate", "handle": handle,
                              "n_seeds": len(seeds), "reverse": bool(reverse)})
        tracks = []
        for seed in seeds:
            try:
                start = sess.frames.index(seed.frame_id)
            except ValueError:
                raise ValueError(
                    "seed frame {!r} is not in session {!r}. Defaulting to the "
                    "first frame would silently propagate from the wrong place."
                    .format(seed.frame_id, handle))
            order = range(start - 1, -1, -1) if reverse else range(start, len(sess.frames))
            points = []
            for step, i in enumerate(order):
                fid = sess.frames[i]
                dx = self.drift * step * (1 if not reverse else -1)
                b = seed.box
                points.append(TrackPoint(
                    frame_id=str(fid),
                    box=Box(float(b.x1 + dx), float(b.y1), float(b.x2 + dx), float(b.y2)),
                    score=float(max(0.05, seed.score - self.decay * step))))
            if not points:
                continue
            tracks.append(Track(
                track_id=int(sess.next_track_id), label=str(seed.label),
                points=tuple(points), box_source=BoxSource.PROPAGATED,
                mask_fragmentation=float(_unit(self.seed, handle, seed.label, "frag") * 0.4),
                chunk_ids=(0,)))
            sess.next_track_id += 1
        return tuple(tracks)

    def _close_impl(self, handle):
        sess = self._sessions.get(handle)
        if sess is not None:
            sess.open = False
        self.call_log.append({"op": "close", "handle": handle})

    def open_handles(self):
        """How many sessions are still open. Should be zero between clips."""
        return sum(1 for s in self._sessions.values() if s.open)


class FakeCropClassifier(CropClassifier):
    """Scores a crop against the fine-grained vocabulary, deterministically."""

    model_id = "fake/crop-classifier-v1"

    def __init__(self, seed="c", favour=None):
        self.seed = str(seed)
        self.favour = dict(favour or {})
        self.call_log = []

    def _classify_impl(self, frame_path, boxes, labels):
        self.call_log.append({"op": "classify", "frame": frame_path,
                              "n_boxes": len(boxes), "labels": list(labels)})
        rows = []
        for i, box in enumerate(boxes):
            raw = []
            for lab in labels:
                v = _unit(self.seed, frame_path, i, lab)
                v += self.favour.get(lab, 0.0)
                raw.append(max(0.0, v))
            total = sum(raw)
            if total <= 0:
                rows.append(tuple(0.0 for _ in labels))
            else:
                rows.append(tuple(float(v / total) for v in raw))
        return tuple(rows)
