"""TAO-format writer carrying negative categories, for federated tracking evaluation.

This format exists nowhere in the sibling projects and is the reason the
federated bookkeeping is a shared module: emitting the negative and
not-exhaustive category lists is what makes an unqueried class *ignored*
rather than counted against the model.

Two operational notes the research established:

* use the maintained fork of the evaluation library, pinned at 1.3.0. The
  upstream package does not run on any modern array library, because its
  TAO reader still uses aliases removed years ago.
* set the maximum detections per image to unlimited. The default cap of 300
  is exactly the truncation that moves rare-class precision by several points.
"""
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.export.coco import build_categories
from vlmlab.geometry import xyxy_to_xywh

__all__ = ["write_tao", "EVAL_LIBRARY_PIN", "EVAL_MAX_DETECTIONS"]

#: The maintained fork. Upstream is unmaintained and does not import.
EVAL_LIBRARY_PIN = "trackeval==1.3.0"
#: Unlimited. The library default of 300 silently truncates.
EVAL_MAX_DETECTIONS = 0


def write_tao(path, clips, images, annotations, tracks, labels,
              federated_by_clip):
    """Write a TAO-style document with per-video federated category lists.

    ``annotations`` are tuples of ``(image_id, label, box, track_id, ignore)``.
    ``federated_by_clip`` maps a clip identifier to a ``FederatedLabels``.
    """
    cats = build_categories(labels)
    name_to_id = {c["name"]: c["id"] for c in cats}

    videos = []
    for clip in clips:
        fed = federated_by_clip.get(clip.clip_id)
        neg = sorted(name_to_id[n] for n in (fed.negatives if fed else ())
                     if n in name_to_id)
        nonex = sorted(name_to_id[n] for n in (fed.not_exhaustive if fed else ())
                       if n in name_to_id)
        videos.append({
            "id": clip.clip_id,
            "name": clip.clip_id,
            "metadata": {"session_id": clip.session_id,
                         "duration_s": clip.duration_s},
            # The two fields that make the evaluation federated.
            "neg_category_ids": neg,
            "not_exhaustive_category_ids": nonex,
        })

    track_rows = []
    for tr in tracks:
        if tr.label not in name_to_id:
            continue
        track_rows.append({
            "id": tr.track_id,
            "category_id": name_to_id[tr.label],
            "n_frames": tr.n_frames,
            "spans_chunks": bool(tr.spans_chunks),
            "chunk_ids": list(tr.chunk_ids),
        })

    ann_rows = []
    for i, (image_id, label, box, track_id, ignore) in enumerate(annotations, start=1):
        if label not in name_to_id:
            raise KeyError("label {!r} is not in the vocabulary".format(label))
        x, y, w, h = xyxy_to_xywh(box)
        ann_rows.append({
            "id": i, "image_id": image_id, "track_id": track_id,
            "category_id": name_to_id[label],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h), "iscrowd": 1 if ignore else 0,
        })

    doc = {
        "info": {
            "description": "vlmlab TAO-format export",
            "evaluation_library": EVAL_LIBRARY_PIN,
            "max_detections_per_image": EVAL_MAX_DETECTIONS,
            "note": ("Upstream TrackEval does not import on a modern array "
                     "library; use the pinned maintained fork. Set maximum "
                     "detections to unlimited or rare-class precision is "
                     "silently truncated."),
        },
        "videos": videos,
        "images": list(images),
        "categories": cats,
        "tracks": track_rows,
        "annotations": ann_rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return doc
