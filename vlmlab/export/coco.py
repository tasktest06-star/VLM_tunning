"""COCO writers. Ground truth and predictions are written by separate
functions so they cannot be confused.

The sibling project writes a ``score`` field into ground-truth annotations,
which makes a file that is neither valid ground truth nor valid predictions.
Keeping the two apart removes that whole class of mistake.

Categories are one-indexed, as the format requires. The model side is
zero-indexed, so the offset is applied on export and the detail is recorded
here rather than rediscovered.
"""
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.geometry import xyxy_to_xywh

__all__ = ["build_categories", "write_ground_truth", "write_predictions",
           "to_model_index", "from_model_index"]


def build_categories(labels):
    """One-indexed categories, in a stable order."""
    return [{"id": i + 1, "name": name, "supercategory": "lab_equipment"}
            for i, name in enumerate(labels)]


def to_model_index(category_id):
    """COCO one-indexed category to zero-indexed model label."""
    return category_id - 1


def from_model_index(model_index):
    return model_index + 1


def write_ground_truth(path, images, annotations, labels, info=None):
    """``images`` are dicts with id, file_name, width, height.

    ``annotations`` are tuples of ``(image_id, label, box, ignore)``.
    """
    name_to_id = {c["name"]: c["id"] for c in build_categories(labels)}
    out_anns = []
    for i, (image_id, label, box, ignore) in enumerate(annotations, start=1):
        if label not in name_to_id:
            raise KeyError("label {!r} is not in the vocabulary".format(label))
        x, y, w, h = xyxy_to_xywh(box)
        out_anns.append({
            "id": i, "image_id": image_id,
            "category_id": name_to_id[label],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h), "iscrowd": 1 if ignore else 0,
        })
    doc = {
        "info": info or {"description": "vlmlab gold set"},
        "images": list(images),
        "categories": build_categories(labels),
        "annotations": out_anns,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return doc


def write_predictions(path, predictions, labels):
    """Detection results, the flat list the format expects.

    ``predictions`` are tuples of ``(image_id, label, box, score)``.
    """
    name_to_id = {c["name"]: c["id"] for c in build_categories(labels)}
    rows = []
    for image_id, label, box, score in predictions:
        if label not in name_to_id:
            raise KeyError("label {!r} is not in the vocabulary".format(label))
        x, y, w, h = xyxy_to_xywh(box)
        rows.append({"image_id": image_id, "category_id": name_to_id[label],
                     "bbox": [float(x), float(y), float(w), float(h)],
                     "score": float(score)})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    return rows
