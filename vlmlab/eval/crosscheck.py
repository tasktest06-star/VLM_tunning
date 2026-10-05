"""Optional cross-check of the pure-Python average precision.

The pure implementation is the tested reference, because the metric has to be
auditable at this sample size and has to run where no package manager exists.
This module exists so a reviewer who wants the compiled implementation can get
agreement on demand, and it is skipped when that library is absent.

Two settings must be replicated deliberately or the comparison disagrees for
boring reasons rather than real ones: the per-image detection cap, and the
area ranges. Both are passed through explicitly.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["available", "CrosscheckUnavailable", "compare_average_precision"]


class CrosscheckUnavailable(RuntimeError):
    pass


def available():
    try:
        import pycocotools.coco  # noqa: F401,PLC0415
        import pycocotools.cocoeval  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def compare_average_precision(preds, gts, labels, iou_threshold=0.5,
                              max_dets=100, tolerance=1e-6):
    """Compare the pure implementation against the compiled one.

    Returns a dict with both values and the absolute difference. Raises
    ``CrosscheckUnavailable`` when the compiled library is not installed, which
    is the normal case in this repository.
    """
    if not available():
        raise CrosscheckUnavailable(
            "pycocotools is not installed. The pure-Python implementation is "
            "the tested reference and does not need it; install it only if you "
            "want this comparison.")

    from pycocotools.coco import COCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval  # noqa: PLC0415

    from vlmlab.eval.metrics import mean_average_precision

    labels = list(labels)
    name_to_id = {name: i + 1 for i, name in enumerate(labels)}
    image_ids = sorted({g.image_id for g in gts} | {p.image_id for p in preds})
    img_to_int = {name: i + 1 for i, name in enumerate(image_ids)}

    gt_doc = {
        "info": {},
        "licenses": [],
        "images": [{"id": img_to_int[n], "file_name": str(n),
                    "width": 10000, "height": 10000} for n in image_ids],
        "categories": [{"id": cid, "name": n, "supercategory": "o"}
                       for n, cid in sorted(name_to_id.items(),
                                            key=lambda kv: kv[1])],
        "annotations": [],
    }
    for i, g in enumerate(gts, start=1):
        w = g.box.x2 - g.box.x1
        h = g.box.y2 - g.box.y1
        gt_doc["annotations"].append({
            "id": i, "image_id": img_to_int[g.image_id],
            "category_id": name_to_id[g.label],
            "bbox": [g.box.x1, g.box.y1, w, h], "area": w * h,
            "iscrowd": 1 if g.ignore else 0})

    coco_gt = COCO()
    coco_gt.dataset = gt_doc
    coco_gt.createIndex()

    results = []
    for p in preds:
        w = p.box.x2 - p.box.x1
        h = p.box.y2 - p.box.y1
        results.append({"image_id": img_to_int[p.image_id],
                        "category_id": name_to_id[p.label],
                        "bbox": [p.box.x1, p.box.y1, w, h],
                        "score": p.score})
    if not results:
        raise ValueError("no predictions to compare")

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = [iou_threshold]
    ev.params.maxDets = [max_dets]
    ev.params.areaRng = [[0.0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    compiled = float(ev.stats[0])

    ours = mean_average_precision(preds, gts, labels=labels,
                                  iou_threshold=iou_threshold,
                                  max_dets=max_dets)
    pure = ours["mAP"]
    diff = None if pure is None else abs(pure - compiled)
    return {"pure": pure, "compiled": compiled, "abs_diff": diff,
            "agrees": (diff is not None and diff <= tolerance),
            "tolerance": tolerance,
            "n_classes_averaged": ours["n_classes_averaged"]}
