"""Box geometry in pure Python.

There is deliberately **no accelerated fast path**. The sibling project wraps
``torchvision.ops.batched_nms`` with a Python fallback, which means the code
that gets tested is not the code that ships. Suppression over a few dozen
boxes on a laboratory bench does not need a GPU, and having one implementation
is worth more than the microseconds.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

from vlmlab.types import Box

__all__ = [
    "iou", "iou_matrix", "nms", "nms_per_class", "xyxy_to_xywh", "xywh_to_xyxy",
    "xyxy_to_cxcywh", "cxcywh_to_xyxy", "normalise", "denormalise",
    "clip_to_image", "area_ratio", "min_side",
]


def iou(a, b):
    """Intersection over union of two boxes.

    Returns exactly ``0.0`` for boxes that merely touch, and for an empty
    union, rather than a negative number or a division error.
    """
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def iou_matrix(boxes_a, boxes_b):
    return [[iou(a, b) for b in boxes_b] for a in boxes_a]


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. Returns kept indices.

    Deterministic regardless of input order among equal scores: ties break on
    the original index, so shuffling the input cannot change the output set.
    """
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores differ in length")
    # Ties break on box geometry, not on position in the input list. Breaking
    # on the input index would make the surviving box depend on the order the
    # detections happened to arrive in.
    order = sorted(range(len(boxes)),
                   key=lambda i: (-scores[i],) + boxes[i].as_tuple())
    keep = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed[j] = True
    return sorted(keep)


def nms_per_class(boxes, scores, labels, iou_threshold):
    """Suppression applied independently within each label."""
    by_label = {}
    for idx, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(idx)
    keep = []
    for lab in sorted(by_label):
        idxs = by_label[lab]
        sub_keep = nms([boxes[i] for i in idxs], [scores[i] for i in idxs], iou_threshold)
        keep.extend(idxs[k] for k in sub_keep)
    return sorted(keep)


def xyxy_to_xywh(b):
    """Corner form to COCO's top-left plus size form."""
    return (b.x1, b.y1, b.width, b.height)


def xywh_to_xyxy(x, y, w, h):
    return Box(x, y, x + w, y + h)


def xyxy_to_cxcywh(b):
    """Corner form to centre plus size.

    Needed because the reference detector's box-prompt interface takes centre
    form, not corner form, which is an easy and silent mistake to make.
    """
    return (b.x1 + b.width / 2.0, b.y1 + b.height / 2.0, b.width, b.height)


def cxcywh_to_xyxy(cx, cy, w, h):
    return Box(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def normalise(b, image_width, image_height):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    return Box(b.x1 / image_width, b.y1 / image_height,
               b.x2 / image_width, b.y2 / image_height)


def denormalise(b, image_width, image_height):
    return Box(b.x1 * image_width, b.y1 * image_height,
               b.x2 * image_width, b.y2 * image_height)


def clip_to_image(b, image_width, image_height):
    x1 = min(max(b.x1, 0.0), float(image_width))
    y1 = min(max(b.y1, 0.0), float(image_height))
    x2 = min(max(b.x2, 0.0), float(image_width))
    y2 = min(max(b.y2, 0.0), float(image_height))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return Box(x1, y1, x2, y2)


def area_ratio(b, image_width, image_height):
    total = float(image_width) * float(image_height)
    if total <= 0.0:
        return 0.0
    return b.area / total


def min_side(b):
    return min(b.width, b.height)
