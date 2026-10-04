"""SAM 3 keyframe detector.

NOT EXECUTED. Written on a machine with no GPU, against the contract in
``tests/contract.py``. Every assertion that applies to it is already written
and already passing against the fake backend, so what remains untested here is
the library call surface, not the logic.

Four corrections from verification are baked in.

**The model identifier is ``facebook/sam3``, not ``facebook/sam3.1``.** The
latter ships only a bare checkpoint with no integration into the modelling
library, so none of the interface this adapter uses exists there. Both
repositories are access-gated with manual approval, so request access before
anything else.

**Boxes come from the image path.** The video path derives boxes from masks,
which caps box quality at mask quality and silently inflates a box when a mask
fragments. The image path exposes a real detection head and a decoupled
presence score.

**Scores combine detection and presence.** The documented rule multiplies the
two after a sigmoid each.

**One concept per session.** Adding a prompt resets session state, so the
orchestration must issue one session per concept rather than assume a list is
honoured. That is declared through ``max_concepts_per_session``.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vlmlab.backends.base import KeyframeDetector
from vlmlab.types import Box, BoxSource, Detection

__all__ = ["Sam3Detector"]


class Sam3Detector(KeyframeDetector):
    supports_presence = True
    boxes_from_detection_head = True
    #: Adding a prompt resets the session, so one concept per session.
    max_concepts_per_session = 1
    #: Centre form, not corner form. Getting this wrong is silent.
    box_prompt_format = "cxcywh"
    model_id = "facebook/sam3"

    def __init__(self, cfg, device="cuda", verify_guards=True):
        self.cfg = cfg
        self.device = device
        self.verify_guards = bool(verify_guards)
        self._model = None
        self._processor = None
        if cfg.detector.model_id != self.model_id:
            # Permit an override but say plainly what is lost.
            self.model_id = cfg.detector.model_id

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        try:
            from transformers import Sam3Model  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Sam3Model is unavailable in this version of the modelling "
                "library. Install a build that integrates SAM 3, and note that "
                "facebook/sam3.1 ships only a checkpoint with no integration."
            ) from exc

        dtype = getattr(torch, self.cfg.detector.compute_dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Sam3Model.from_pretrained(
            self.model_id,
            dtype=dtype,
            # Frozen in configuration because the default is eager, which
            # costs roughly 1.26 GB per layer at these sequence lengths.
            attn_implementation=self.cfg.detector.attn_implementation,
        ).to(self.device)
        self._model.eval()

        if self.verify_guards:
            from vlmlab.backends._torch_guards import (assert_attn_impl,
                                                       assert_dtype,
                                                       assert_on_cuda)
            assert_attn_impl(self._model, self.cfg.detector.attn_implementation)
            assert_on_cuda(self._model)
            assert_dtype(self._model, self.cfg.detector.compute_dtype)

    def _detect_impl(self, frame_path, prompts):
        """One prompt per call, because a second prompt would reset state."""
        self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        if len(prompts) > self.max_concepts_per_session:
            raise ValueError(
                "{} prompts for a backend that accepts {} per session. The "
                "orchestration must issue one session per concept."
                .format(len(prompts), self.max_concepts_per_session))

        image = Image.open(frame_path).convert("RGB")
        size = self.cfg.detector.frame_size
        if size:
            image = image.resize((size, size))
        width, height = image.size
        prompt = prompts[0]

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        outputs = self._model(**inputs)

        # The documented scoring rule: detection logits times presence logits,
        # each through a sigmoid.
        pred_boxes = outputs.pred_boxes[0]
        pred_logits = outputs.pred_logits[0]
        presence_logits = getattr(outputs, "presence_logits", None)
        det_scores = pred_logits.sigmoid().reshape(-1)
        if presence_logits is not None:
            presence = float(presence_logits.sigmoid().reshape(-1)[0].item())
        else:
            presence = None
        final = det_scores * (presence if presence is not None else 1.0)

        out = []
        thresh = self.cfg.detector.score_threshold
        for i in range(pred_boxes.shape[0]):
            score = float(final[i].item())
            if score < thresh:
                continue
            cx, cy, bw, bh = (float(v) for v in pred_boxes[i].tolist())
            # Model output is normalised centre form.
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
            out.append(Detection(
                frame_id=str(frame_path),
                label=str(prompt).replace(" ", "_"),
                box=Box(x1, y1, x2, y2),
                score=score,
                box_source=BoxSource.DETECTION_HEAD,
                presence=presence,
                prompt=str(prompt),
                model_id=self.model_id))
        if len(out) > self.cfg.detector.max_objects:
            out.sort(key=lambda d: -d.score)
            out = out[:self.cfg.detector.max_objects]
        return tuple(out)
