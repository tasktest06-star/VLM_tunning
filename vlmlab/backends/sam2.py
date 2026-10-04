"""SAM 2.1 propagator.

NOT EXECUTED. Written against the contract, which already passes against the
fake backend.

Three corrections from verification.

**There is no single call covering both directions.** The real signature takes
a reverse flag, and the keyword a research document named does not exist in
the source. So propagation runs twice, which the orchestration does.

**Device and dtype must be set explicitly on the session.** The session device
defaults to the processor and will not raise; it merely runs fifty to a
hundred times slower. The dtype defaults to single precision.

**Sessions must be closed between clips.** Inference state accumulates and is
not released otherwise, which has been reported as a memory leak.

Boxes produced here are mask-derived, which is declared honestly so the
pipeline will refuse them where a real detection head is required.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vlmlab.backends.base import Propagator, SessionClosedError
from vlmlab.types import Box, BoxSource, Track, TrackPoint

__all__ = ["Sam2Propagator"]


class Sam2Propagator(Propagator):
    model_id = "facebook/sam2.1-hiera-small"
    produces_box_source = BoxSource.MASK_DERIVED

    def __init__(self, cfg, device="cuda", verify_guards=True):
        self.cfg = cfg
        self.device = device
        self.verify_guards = bool(verify_guards)
        self._model = None
        self._processor = None
        self._sessions = {}
        self._counter = 0
        if cfg.propagator.model_id:
            self.model_id = cfg.propagator.model_id

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        try:
            from transformers import Sam2VideoModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Sam2VideoModel is unavailable in this version of the modelling "
                "library.") from exc

        dtype = getattr(torch, self.cfg.detector.compute_dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Sam2VideoModel.from_pretrained(
            self.model_id, dtype=dtype,
            attn_implementation=self.cfg.detector.attn_implementation,
        ).to(self.device)
        self._model.eval()
        if self.verify_guards:
            from vlmlab.backends._torch_guards import (assert_dtype,
                                                       assert_on_cuda)
            assert_on_cuda(self._model)
            assert_dtype(self._model, self.cfg.detector.compute_dtype)

    def _open_impl(self, clip_id, frame_paths):
        self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        dtype = getattr(torch, self.cfg.detector.compute_dtype)
        session = self._processor.init_video_session(
            video=frames,
            # All three default to the processor unless given, so they must be
            # passed. Offloading state and storage is the VRAM mitigation that
            # makes a 12 GB card viable.
            inference_device=self.device,
            inference_state_device="cpu" if self.cfg.propagator.offload_state_to_cpu else self.device,
            video_storage_device="cpu" if self.cfg.propagator.offload_video_to_cpu else self.device,
            processing_device="cpu",
            dtype=dtype,
        )
        self._counter += 1
        handle = "{}#{}".format(clip_id, self._counter)
        self._sessions[handle] = {"session": session, "frames": tuple(frame_paths),
                                  "open": True, "next_id": 1}
        return handle

    def _propagate_impl(self, handle, seeds, reverse):
        rec = self._sessions.get(handle)
        if rec is None or not rec["open"]:
            raise SessionClosedError("session {!r} is closed".format(handle))
        import torch  # noqa: PLC0415

        frames = rec["frames"]
        session = rec["session"]
        tracks = []
        for seed in seeds:
            try:
                start = frames.index(seed.frame_id)
            except ValueError:
                raise ValueError(
                    "seed frame {!r} is not in session {!r}".format(
                        seed.frame_id, handle))
            # Box prompts are centre-width-height, not corner form.
            from vlmlab.geometry import xyxy_to_cxcywh
            cx, cy, bw, bh = xyxy_to_cxcywh(seed.box)
            self._processor.add_prompt(
                session, frame_idx=start,
                boxes_xywh=[[cx, cy, bw, bh]], box_labels=[1])

            points = []
            # The real signature exposes a reverse flag; there is no call that
            # covers both directions at once.
            for out in self._model.propagate_in_video_iterator(
                    session, start_frame_idx=start, reverse=bool(reverse)):
                idx = int(getattr(out, "frame_idx", 0))
                masks = self._processor.post_process_masks(out)
                boxes = _masks_to_boxes(masks)
                if not boxes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in boxes[0])
                if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                    continue
                points.append(TrackPoint(
                    frame_id=str(frames[idx]), box=Box(x1, y1, x2, y2),
                    score=float(seed.score)))
            if not points:
                continue
            tracks.append(Track(
                track_id=int(rec["next_id"]), label=str(seed.label),
                points=tuple(points),
                # Honest: these came from masks, not a detection head.
                box_source=BoxSource.MASK_DERIVED,
                mask_fragmentation=_fragmentation(masks),
                chunk_ids=(0,)))
            rec["next_id"] += 1
        return tuple(tracks)

    def _close_impl(self, handle):
        rec = self._sessions.get(handle)
        if rec is None:
            return
        rec["open"] = False
        session = rec.get("session")
        reset = getattr(session, "reset_inference_session", None)
        if callable(reset):
            reset()
        rec["session"] = None
        import gc
        gc.collect()

    def open_handles(self):
        return sum(1 for r in self._sessions.values() if r["open"])


def _masks_to_boxes(masks):
    """Tight boxes around each mask. Quality is capped by mask quality."""
    try:
        from torchvision.ops import masks_to_boxes  # noqa: PLC0415
    except ImportError:
        return []
    if masks is None or len(masks) == 0:
        return []
    boxes = masks_to_boxes(masks)
    return [[float(v) for v in row] for row in boxes.tolist()]


def _fragmentation(masks):
    """Crude fragmentation proxy, for the audit the research requires.

    No criterion is given anywhere, so this is a placeholder with an
    unmeasured threshold in configuration. Replace it once you have gold
    masks to calibrate against.
    """
    if masks is None or len(masks) == 0:
        return None
    try:
        total = float(masks[0].sum().item())
        area = float(masks[0].numel())
    except Exception:  # noqa: BLE001
        return None
    if area <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - total / area))
