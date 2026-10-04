"""Dataset plumbing for the student detector, with the pure part separated.

The index and target conversion are pure Python and therefore tested in this
repository. Only the thin tensor shim needs torch.

Two details are easy to get wrong and are handled explicitly: the annotation
format is one-indexed while the model is zero-indexed, and an image with no
annotations must still yield a well-formed empty target or the matching loss
becomes invalid.
"""
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["CocoIndex", "Target"]


class Target(object):
    """A single image's targets in normalised centre form, plain Python."""

    __slots__ = ("image_id", "file_name", "width", "height", "boxes", "labels")

    def __init__(self, image_id, file_name, width, height, boxes, labels):
        self.image_id = image_id
        self.file_name = file_name
        self.width = int(width)
        self.height = int(height)
        self.boxes = tuple(boxes)
        self.labels = tuple(labels)

    @property
    def is_empty(self):
        return len(self.boxes) == 0

    def __repr__(self):
        return ("Target(image_id={}, n={}, {}x{})"
                .format(self.image_id, len(self.boxes), self.width, self.height))


class CocoIndex(object):
    """Reads an annotation document once and serves per-image targets."""

    def __init__(self, doc):
        self.images = {img["id"]: img for img in doc.get("images", [])}
        self.categories = {c["id"]: c["name"] for c in doc.get("categories", [])}
        self.n_classes = len(self.categories)
        self._by_image = {}
        for ann in doc.get("annotations", []):
            self._by_image.setdefault(ann["image_id"], []).append(ann)

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def from_dict(cls, doc):
        return cls(doc)

    def image_ids(self):
        return sorted(self.images)

    def to_targets(self, image_id):
        """Normalised centre-form boxes and zero-indexed labels.

        Returns an empty target rather than skipping, so unannotated images
        still contribute as negatives.
        """
        img = self.images.get(image_id)
        if img is None:
            raise KeyError("image {!r} is not in the index".format(image_id))
        w = float(img["width"])
        h = float(img["height"])
        if w <= 0 or h <= 0:
            raise ValueError("image {!r} has non-positive dimensions".format(image_id))
        boxes = []
        labels = []
        for ann in self._by_image.get(image_id, ()):
            if ann.get("iscrowd"):
                continue
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            if bw <= 0 or bh <= 0:
                continue
            boxes.append(((x + bw / 2.0) / w, (y + bh / 2.0) / h, bw / w, bh / h))
            # One-indexed annotation category to zero-indexed model label.
            labels.append(int(ann["category_id"]) - 1)
        return Target(image_id, img["file_name"], int(w), int(h), boxes, labels)

    def coverage(self):
        """How many indexed images actually carry annotations."""
        with_ann = sum(1 for i in self.images if self._by_image.get(i))
        return {"n_images": len(self.images), "n_with_annotations": with_ann,
                "n_empty": len(self.images) - with_ann,
                "n_classes": self.n_classes}
