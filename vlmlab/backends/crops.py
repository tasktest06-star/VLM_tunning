"""Frozen-backbone crop classifier.

NOT EXECUTED. Written against the contract.

This is the half of the architecture that makes the project tractable.
Fine-grained instrument names fail as detection prompts, so the detector
localises generically and this names the crop. Five of the target classes have
never been seen as a box by any released detector, but all of them have
hundreds to over a thousand images in a public classification corpus that is
usable for research, which is exactly what trains this head.

The backbone stays frozen. Only a small head is trained, so this costs
minutes and a couple of gigabytes rather than a fine-tuning run.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vlmlab.backends.base import CropClassifier

__all__ = ["FrozenCropClassifier"]


class FrozenCropClassifier(CropClassifier):
    model_id = "google/siglip2-base-patch16-224"

    def __init__(self, model_id=None, device="cuda", prompt_template="a photo of a {}",
                 cache=None):
        if model_id:
            self.model_id = model_id
        self.device = device
        self.prompt_template = prompt_template
        self._model = None
        self._processor = None
        self._text_cache = {}
        self._cache = cache if cache is not None else {}

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(
            self.model_id, attn_implementation="sdpa").to(self.device)
        self._model.eval()

    def _text_features(self, labels):
        """Cached, because text features are constant across every frame."""
        import torch  # noqa: PLC0415
        missing = [l for l in labels if l not in self._text_cache]
        if missing:
            prompts = [self.prompt_template.format(l.replace("_", " "))
                       for l in missing]
            inputs = self._processor(text=prompts, return_tensors="pt",
                                     padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for lab, vec in zip(missing, feats):
                self._text_cache[lab] = vec
        return [self._text_cache[l] for l in labels]

    def _classify_impl(self, frame_path, boxes, labels):
        self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.open(frame_path).convert("RGB")
        crops = []
        for b in boxes:
            crops.append(image.crop((int(b.x1), int(b.y1),
                                     max(int(b.x2), int(b.x1) + 1),
                                     max(int(b.y2), int(b.y1) + 1))))
        if not crops:
            return ()

        inputs = self._processor(images=crops, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        img_feats = self._model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        txt = torch.stack(self._text_features(list(labels)))
        logits = img_feats @ txt.t()
        probs = logits.softmax(dim=-1)
        # Plain Python on the way out, as the boundary requires.
        return tuple(tuple(float(v) for v in row) for row in probs.tolist())
