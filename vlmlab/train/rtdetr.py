"""Student detector, phase four. Permissively licensed throughout.

NOT EXECUTED. Written from the modelling library's own documentation rather
than adapted from a sibling project, because neither sibling repository
carries a licence file, so their Apache claim is a docstring assertion rather
than a grant.

The student exists because the teacher is far too slow to deploy: cost is
linear in concepts queried, and a three-minute clip takes minutes rather than
seconds. Distilling into a small detector is what makes inference practical.

Note the licence asymmetry that decides the choice here. Every widely used
YOLO implementation is copyleft, and the published laboratory-equipment
baselines all use one, so their numbers are citable but their weights are not
usable. This model family is permissive for both code and weights.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["RtdetrTrainer"]


class RtdetrTrainer(object):
    def __init__(self, cfg, labels, device="cuda"):
        self.cfg = cfg
        self.labels = tuple(labels)
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import (AutoImageProcessor,  # noqa: PLC0415
                                  RTDetrForObjectDetection)
        scfg = self.cfg.student
        self._processor = AutoImageProcessor.from_pretrained(scfg.model_id)
        self._model = RTDetrForObjectDetection.from_pretrained(
            scfg.model_id,
            num_labels=len(self.labels),
            # Required when replacing the head for a different class count.
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def train(self, index, frame_dir, output_dir, epochs=None):
        """Train on exported annotations.

        Deliberately thin. The parts that can be wrong without a GPU, meaning
        the index, the one-to-zero index offset and the empty-target handling,
        live in ``rtdetr_dataset.py`` and are tested.
        """
        self._load()
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        scfg = self.cfg.student
        epochs = epochs or scfg.epochs
        processor = self._processor
        ids = index.image_ids()

        class _Wrapped(Dataset):
            def __len__(self):
                return len(ids)

            def __getitem__(self, i):
                target = index.to_targets(ids[i])
                path = "{}/{}".format(frame_dir.rstrip("/"), target.file_name)
                image = Image.open(path).convert("RGB")
                enc = processor(images=image, return_tensors="pt")
                boxes = torch.tensor(target.boxes, dtype=torch.float32) \
                    if target.boxes else torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.tensor(target.labels, dtype=torch.long) \
                    if target.labels else torch.zeros((0,), dtype=torch.long)
                return {"pixel_values": enc["pixel_values"][0],
                        "labels": {"class_labels": labels, "boxes": boxes}}

        def collate(batch):
            return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                    "labels": [b["labels"] for b in batch]}

        loader = DataLoader(_Wrapped(), batch_size=scfg.batch_size, shuffle=True,
                            collate_fn=collate)
        optim = torch.optim.AdamW(self._model.parameters(), lr=scfg.learning_rate,
                                  weight_decay=scfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=scfg.learning_rate, total_steps=max(1, len(loader) * epochs),
            pct_start=0.1)

        self._model.train()
        history = []
        for epoch in range(epochs):
            running = 0.0
            for batch in loader:
                optim.zero_grad()
                out = self._model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    labels=[{k: v.to(self.device) for k, v in t.items()}
                            for t in batch["labels"]])
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
                optim.step()
                sched.step()
                running += float(out.loss.item())
            history.append(running / max(1, len(loader)))
        self._model.save_pretrained(output_dir)
        return {"output_dir": output_dir, "loss_history": history}

    def predict(self, frame_path, threshold=None):
        """Returns the same record shape the detectors do, so it composes."""
        self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from vlmlab.types import Box, BoxSource, Detection

        # A low threshold keeps the precision-recall curve complete; raising
        # it here silently truncates the metric.
        threshold = self.cfg.student.inference_threshold if threshold is None else threshold
        image = Image.open(frame_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        self._model.eval()
        with torch.inference_mode():
            out = self._model(**inputs)
        results = self._processor.post_process_object_detection(
            out, target_sizes=torch.tensor([image.size[::-1]]),
            threshold=threshold)[0]
        dets = []
        for score, label, box in zip(results["scores"], results["labels"],
                                     results["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            dets.append(Detection(
                frame_id=str(frame_path),
                label=self.labels[int(label.item())],
                box=Box(x1, y1, x2, y2), score=float(score.item()),
                box_source=BoxSource.DETECTION_HEAD,
                model_id=self.cfg.student.model_id))
        return tuple(dets)
