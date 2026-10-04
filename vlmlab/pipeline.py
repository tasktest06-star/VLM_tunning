"""Backend-agnostic orchestration.

Three things live here rather than in an adapter, because each is pure logic
that can be got wrong without a GPU and is expensive to debug on one.

**Chunking.** A three-minute clip exceeds the segmentation task's documented
thirty-second definition by six times, so every clip is split and the
identities are stitched across boundaries. This is the common path, not an
edge case, and it is the dominant accuracy risk in the design.

**Double propagation.** The real interface takes a reverse flag and has no
single call covering both directions, so propagation runs twice per seed. A
research document claimed a keyword that does not exist in the source; this
loop is the correction.

**One session per concept.** When a backend accepts only one concept per
session, because adding a prompt resets state, the orchestration issues a
session per concept rather than silently getting one concept's results.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab import federated
from vlmlab.consistency import cross_model_agreement, temporal_consistency
from vlmlab.frames import chunk_indices, keyframe_indices, plan_frame_grid
from vlmlab.mil import run_cascade
from vlmlab.prompts import build_clip_prompt_plan
from vlmlab.types import (Box, BoxSource, Detection, RejectReason, Track,
                          TrackPoint)

__all__ = ["ClipResult", "PipelineError", "process_clip", "stitch_tracks",
           "run_pipeline"]


class PipelineError(RuntimeError):
    pass


class ClipResult(object):
    def __init__(self, clip, grid, chunks, plan, detections, tracks, cascade,
                 federated_labels):
        self.clip = clip
        self.grid = grid
        self.chunks = tuple(chunks)
        self.plan = plan
        self.detections = tuple(detections)
        self.tracks = tuple(tracks)
        self.cascade = cascade
        self.federated = federated_labels

    def summary(self):
        return {
            "clip_id": self.clip.clip_id,
            "session_id": self.clip.session_id,
            "n_frames": self.grid.n_frames,
            "n_chunks": len(self.chunks),
            "n_concepts": self.plan.n_concepts,
            "n_detections": len(self.detections),
            "n_accepted": len(self.cascade.accepted),
            "n_background": len(self.cascade.background),
            "n_tracks": len(self.tracks),
            "n_stitched_tracks": sum(1 for t in self.tracks if t.spans_chunks),
            "labels_never_found": list(self.cascade.missing_labels),
            "reasons": self.cascade.reasons(),
        }


def _frame_path(clip_id, index, frame_dir=None, ext=".jpg"):
    name = "{}_{:06d}{}".format(clip_id, index, ext)
    return "{}/{}".format(frame_dir.rstrip("/"), name) if frame_dir else name


def stitch_tracks(per_chunk_tracks, iou_threshold=0.5):
    """Join identities across chunk boundaries by overlap on shared frames.

    Accuracy is lost here and it is unavoidable: the alternative is exceeding
    the documented duration limit for the backend.
    """
    from vlmlab.geometry import iou

    merged = []
    for chunk_id, tracks in sorted(per_chunk_tracks.items()):
        for tr in tracks:
            attached = None
            for m in merged:
                if m["label"] != tr.label:
                    continue
                shared = set(p.frame_id for p in m["points"]) & \
                    set(p.frame_id for p in tr.points)
                if not shared:
                    continue
                by_fid_m = {p.frame_id: p for p in m["points"]}
                best = 0.0
                for fid in shared:
                    other = next(p for p in tr.points if p.frame_id == fid)
                    best = max(best, iou(by_fid_m[fid].box, other.box))
                if best >= iou_threshold:
                    attached = m
                    break
            if attached is None:
                merged.append({"label": tr.label, "points": list(tr.points),
                               "chunks": [chunk_id],
                               "box_source": tr.box_source,
                               "frag": [tr.mask_fragmentation]})
            else:
                have = set(p.frame_id for p in attached["points"])
                attached["points"].extend(p for p in tr.points if p.frame_id not in have)
                attached["chunks"].append(chunk_id)
                attached["frag"].append(tr.mask_fragmentation)

    out = []
    for i, m in enumerate(merged, start=1):
        frags = [f for f in m["frag"] if f is not None]
        out.append(Track(
            track_id=i, label=m["label"],
            points=tuple(sorted(m["points"], key=lambda p: p.frame_id)),
            box_source=m["box_source"],
            mask_fragmentation=(sum(frags) / float(len(frags))) if frags else None,
            chunk_ids=tuple(sorted(set(m["chunks"])))))
    return tuple(out)


def process_clip(clip, detector, propagator, registry, cfg, frame_dir=None,
                 second_detector=None, logger=None):
    """Run one clip end to end. Backend-agnostic."""
    if cfg.detector.require_detection_head_boxes and \
            not detector.boxes_from_detection_head:
        raise PipelineError(
            "detector {!r} derives boxes from masks, so box quality is capped by "
            "mask quality. Boxes are this project's deliverable, so use the "
            "detection-head path.".format(detector.model_id))

    grid = plan_frame_grid(
        clip.clip_id, clip.duration_s, clip.container_frame_count,
        cfg.detector.sample_fps, cap=max(1, int((clip.duration_s or 0)
                                                * cfg.detector.sample_fps)) or 1)
    chunks = chunk_indices(grid, cfg.propagator.chunk_frames,
                           overlap=cfg.propagator.chunk_overlap,
                           clip_duration_s=clip.duration_s,
                           pcs_max_seconds=cfg.propagator.pcs_max_seconds)

    plan = build_clip_prompt_plan(
        clip, registry, max_concepts=cfg.detector.max_concepts_per_clip,
        n_hard_negatives=cfg.cascade.n_hard_negatives)
    fed = federated.derive(clip, plan.queried_classes(), registry.names)

    # --- detection on keyframes -------------------------------------------
    kf = keyframe_indices(grid, cfg.detector.keyframe_stride)
    prompts = plan.all_prompts()
    batches = ([ (p,) for p in prompts ]
               if detector.max_concepts_per_session < len(prompts)
               else [tuple(prompts)])

    detections = []
    for idx in kf:
        path = _frame_path(clip.clip_id, idx, frame_dir)
        for batch in batches:
            detections.extend(detector.detect(path, batch))

    if second_detector is not None:
        other = []
        for idx in kf:
            path = _frame_path(clip.clip_id, idx, frame_dir)
            other.extend(second_detector.detect(path, prompts))
        report = cross_model_agreement(detections, other,
                                       cfg.cascade.cross_model_iou)
        if logger:
            logger.info("clip %s cross-model %s precision_proxy=%s",
                        clip.clip_id, report, report.precision_proxy)
        detections = [a for a, _b, _ov in report.agreed]

    # --- temporal consistency ---------------------------------------------
    tc = temporal_consistency(detections, cfg.cascade.temporal_min_appearances,
                              iou_threshold=0.5)
    detections = [v.detection for v in tc if v.reason == RejectReason.KEPT]

    # --- propagation, per chunk, both directions --------------------------
    per_chunk = {}
    for ch in chunks:
        paths = tuple(_frame_path(clip.clip_id, i, frame_dir) for i in ch.indices)
        seeds = [d for d in detections if d.frame_id in set(paths)]
        if not seeds:
            per_chunk[ch.chunk_id] = ()
            continue
        handle = propagator.open_session(clip.clip_id, paths)
        try:
            fwd = propagator.propagate(handle, seeds, reverse=False)
            rev = propagator.propagate(handle, seeds, reverse=True)
        finally:
            # Must close between chunks: inference state accumulates and is
            # not released otherwise.
            propagator.close_session(handle)
        tagged = []
        for tr in list(fwd) + list(rev):
            tagged.append(Track(tr.track_id, tr.label, tr.points, tr.box_source,
                                tr.mask_fragmentation, (ch.chunk_id,)))
        per_chunk[ch.chunk_id] = tuple(tagged)

    tracks = stitch_tracks(per_chunk, cfg.propagator.stitch_iou)
    if cfg.cascade.max_mask_fragmentation is not None:
        tracks = tuple(
            t for t in tracks
            if t.mask_fragmentation is None
            or t.mask_fragmentation <= cfg.cascade.max_mask_fragmentation)

    cascade = run_cascade(detections, clip, fed, cfg.cascade,
                          image_size=(cfg.detector.frame_size,
                                      cfg.detector.frame_size)
                          if cfg.detector.frame_size else None)

    return ClipResult(clip, grid, chunks, plan, detections, tracks, cascade, fed)


def run_pipeline(clips, detector, propagator, registry, cfg, frame_dir=None,
                 second_detector=None, logger=None, checkpoint=None):
    """Process every clip, returning results and an aggregate report."""
    results = []
    for clip in clips:
        if checkpoint is not None and checkpoint.is_done("clip:" + clip.clip_id):
            continue
        res = process_clip(clip, detector, propagator, registry, cfg,
                           frame_dir=frame_dir, second_detector=second_detector,
                           logger=logger)
        results.append(res)
        if logger:
            logger.info("clip %s %s", clip.clip_id, res.summary())
    report = {
        "n_clips": len(results),
        "n_sessions": len({r.clip.session_id for r in results}),
        "total_detections": sum(len(r.detections) for r in results),
        "total_accepted": sum(len(r.cascade.accepted) for r in results),
        "total_background": sum(len(r.cascade.background) for r in results),
        "total_tracks": sum(len(r.tracks) for r in results),
        "stitched_tracks": sum(sum(1 for t in r.tracks if t.spans_chunks)
                               for r in results),
        "clips_with_missing_labels": sum(1 for r in results
                                         if r.cascade.missing_labels),
    }
    return tuple(results), report
