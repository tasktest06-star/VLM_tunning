"""Gold-set selection and scoring: the prerequisite for trusting any number.

The research is emphatic that the existing evaluation in comparable pipelines
measures agreement with the teacher rather than correctness, because the
validation split is itself pseudo-labelled. A small hand-annotated set, kept
entirely outside the pipeline, is the only fix.

Three findings shape the selection rule.

**Spread beats depth.** At a fixed annotation budget, frames spread across many
clips give far more effective sample size than dense annotation of a few. At an
intra-clip correlation of 0.6, three hundred frames over three hundred clips is
worth more than six times the same budget over thirty clips. Dense annotation of
a few clips is the single worst use of evaluation budget.

**The binding constraint is sessions, not frames.** Beyond three or four frames
per clip the interval barely moves, because it is floored by how many
independent recording sessions exist.

**Spacing is free here.** Three or four frames across a three-minute clip land
forty-five to sixty seconds apart, well beyond the two-second floor the research
assumed, so the realised correlation should be lower than planned for.
"""
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vlmlab.types import Box

__all__ = [
    "GoldFrame", "GoldBox", "GoldSet", "select_gold_frames",
    "annotation_budget", "GoldError",
]


class GoldError(ValueError):
    pass


class GoldBox(object):
    __slots__ = ("label", "box", "ignore", "track_id")

    def __init__(self, label, box, ignore=False, track_id=None):
        self.label = str(label)
        self.box = box
        self.ignore = bool(ignore)
        self.track_id = track_id

    def to_dict(self):
        return {"label": self.label, "box": list(self.box.as_tuple()),
                "ignore": self.ignore, "track_id": self.track_id}

    @classmethod
    def from_dict(cls, d):
        x1, y1, x2, y2 = d["box"]
        return cls(d["label"], Box(float(x1), float(y1), float(x2), float(y2)),
                   bool(d.get("ignore", False)), d.get("track_id"))


class GoldFrame(object):
    """One hand-annotated frame.

    ``exhaustive_for`` records which classes were checked on this frame. A
    class outside that set is unknown rather than absent, and scoring it as a
    miss is what inflates federated precision.
    """

    __slots__ = ("clip_id", "session_id", "frame_index", "frame_id", "width",
                 "height", "boxes", "exhaustive_for")

    def __init__(self, clip_id, session_id, frame_index, frame_id, width, height,
                 boxes=(), exhaustive_for=()):
        self.clip_id = str(clip_id)
        self.session_id = str(session_id)
        self.frame_index = int(frame_index)
        self.frame_id = str(frame_id)
        self.width = int(width)
        self.height = int(height)
        self.boxes = tuple(boxes)
        self.exhaustive_for = tuple(sorted(set(exhaustive_for)))

    def to_dict(self):
        return {"clip_id": self.clip_id, "session_id": self.session_id,
                "frame_index": self.frame_index, "frame_id": self.frame_id,
                "width": self.width, "height": self.height,
                "boxes": [b.to_dict() for b in self.boxes],
                "exhaustive_for": list(self.exhaustive_for)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["clip_id"], d["session_id"], d["frame_index"], d["frame_id"],
                   d["width"], d["height"],
                   [GoldBox.from_dict(b) for b in d.get("boxes", ())],
                   d.get("exhaustive_for", ()))

    @property
    def is_annotated(self):
        """False while the frame is still a blank to be filled in."""
        return bool(self.exhaustive_for)


class GoldSet(object):
    def __init__(self, frames=(), vocabulary=()):
        self.frames = tuple(frames)
        self.vocabulary = tuple(sorted(set(vocabulary)))

    # ---- persistence ---------------------------------------------------
    def to_dict(self):
        return {"vocabulary": list(self.vocabulary),
                "frames": [f.to_dict() for f in self.frames],
                "_note": ("exhaustive_for records which classes were CHECKED on "
                          "a frame. A class outside it is unknown, not absent.")}

    def save(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls([GoldFrame.from_dict(f) for f in d.get("frames", ())],
                   d.get("vocabulary", ()))

    # ---- shape ---------------------------------------------------------
    @property
    def sessions(self):
        return tuple(sorted({f.session_id for f in self.frames}))

    @property
    def clips(self):
        return tuple(sorted({f.clip_id for f in self.frames}))

    def annotated(self):
        return tuple(f for f in self.frames if f.is_annotated)

    def progress(self):
        done = len(self.annotated())
        return {"n_frames": len(self.frames), "n_annotated": done,
                "n_remaining": len(self.frames) - done,
                "n_clips": len(self.clips), "n_sessions": len(self.sessions),
                "n_boxes": sum(len(f.boxes) for f in self.annotated())}

    def per_class_counts(self):
        counts = {c: 0 for c in self.vocabulary}
        for f in self.annotated():
            for b in f.boxes:
                if b.label in counts:
                    counts[b.label] += 1
        return counts

    def warnings(self):
        """Everything that would make a reported number untrustworthy."""
        out = []
        prog = self.progress()
        if prog["n_remaining"]:
            out.append("{} of {} frames are not yet annotated; scoring now would "
                       "silently treat them as empty"
                       .format(prog["n_remaining"], prog["n_frames"]))
        if prog["n_sessions"] < 9:
            out.append("only {} recording sessions. The interval on the headline "
                       "metric is floored by session count, not frame count, and "
                       "below about nine it is too wide to rank methods"
                       .format(prog["n_sessions"]))
        for label, n in sorted(self.per_class_counts().items()):
            if 0 < n < 8:
                out.append("class {!r} has {} gold instances; per-class average "
                           "precision is effectively unfalsifiable below about 8"
                           .format(label, n))
            elif n == 0:
                out.append("class {!r} has no gold instances at all".format(label))
        return out

    # ---- scoring -------------------------------------------------------
    def to_ground_truth(self):
        """Convert to the records the metric consumes."""
        from vlmlab.eval.metrics import GroundTruth
        out = []
        for f in self.annotated():
            for b in f.boxes:
                out.append(GroundTruth(f.frame_id, b.label, b.box, b.ignore))
        return tuple(out)

    def filter_predictions(self, predictions):
        """Drop predictions the gold set cannot judge.

        Two filters, both necessary. A prediction on an unannotated frame has
        no ground truth and would count as a false positive purely because
        nobody looked. A prediction of a class that frame was not checked for
        is unknown rather than wrong.
        """
        by_frame = {f.frame_id: f for f in self.annotated()}
        kept = []
        dropped_frame = 0
        dropped_class = 0
        for p in predictions:
            frame = by_frame.get(p.image_id)
            if frame is None:
                dropped_frame += 1
                continue
            if p.label not in frame.exhaustive_for:
                dropped_class += 1
                continue
            kept.append(p)
        return tuple(kept), {"dropped_unannotated_frame": dropped_frame,
                             "dropped_class_not_checked": dropped_class}

    def units_by_session(self, predictions, iou_threshold=0.5):
        """Per-session match records, for the cluster bootstrap.

        The session is the resampling unit, because frames within a session
        are correlated and resampling frames would understate the interval.
        """
        from vlmlab.eval.metrics import match_detections
        gold_by_session = {}
        for f in self.annotated():
            gold_by_session.setdefault(f.session_id, []).append(f)
        pred_by_frame = {}
        for p in predictions:
            pred_by_frame.setdefault(p.image_id, []).append(p)

        units = []
        for session in sorted(gold_by_session):
            gts = []
            preds = []
            for f in gold_by_session[session]:
                for b in f.boxes:
                    from vlmlab.eval.metrics import GroundTruth
                    gts.append(GroundTruth(f.frame_id, b.label, b.box, b.ignore))
                preds.extend(pred_by_frame.get(f.frame_id, ()))
            units.append({"session_id": session, "gts": tuple(gts),
                          "preds": tuple(preds)})
        return tuple(units)


def select_gold_frames(clips, frames_per_clip=3, min_spacing_s=2.0,
                       sample_fps=2.0, vocabulary=(), frame_id_fn=None,
                       width=None, height=None):
    """Choose which frames to hand-annotate.

    Deliberately takes a few frames from **every** clip rather than many frames
    from a few, because spreading beats depth at equal cost by a wide margin.
    Frames are placed at interior quantiles so the first and last moments of a
    clip, which are often atypical, are not over-represented.
    """
    if frames_per_clip < 1:
        raise GoldError("frames_per_clip must be at least 1")
    out = []
    for clip in clips:
        if clip.duration_s is None or clip.duration_s <= 0:
            raise GoldError("clip {!r} has no duration; probe it first"
                            .format(clip.clip_id))
        if clip.container_frame_count is None or clip.container_frame_count <= 0:
            raise GoldError(
                "clip {!r} has no frame count. Refusing to guess, because the "
                "usual fallback samples only the opening of the clip."
                .format(clip.clip_id))

        achievable = int(clip.duration_s // min_spacing_s) + 1
        n = max(1, min(frames_per_clip, achievable))
        n_sampled = max(1, int(clip.duration_s * sample_fps))

        picks = []
        for i in range(n):
            # Interior quantiles: (i + 1) / (n + 1).
            frac = (i + 1) / float(n + 1)
            picks.append(min(n_sampled - 1, int(round(frac * (n_sampled - 1)))))
        picks = sorted(set(picks))

        for idx in picks:
            fid = (frame_id_fn(clip.clip_id, idx) if frame_id_fn
                   else "{}_{:06d}.jpg".format(clip.clip_id, idx))
            out.append(GoldFrame(
                clip_id=clip.clip_id, session_id=clip.session_id,
                frame_index=idx, frame_id=fid,
                width=width or 0, height=height or 0,
                boxes=(), exhaustive_for=()))
    return GoldSet(out, vocabulary)


def annotation_budget(n_clips, frames_per_clip, seconds_per_box=26.5,
                      boxes_per_frame=3.0, verify_speedup=1.5):
    """Estimate human time, using the corrected per-box figure.

    The widely quoted four-second box is mouse-gesture time only, measured on
    familiar classes. End to end, including finding the object and deciding
    what it is, the figure is about twenty-six seconds, and class decision time
    dominates for a domain expert on unfamiliar fine-grained equipment.

    Verify-and-correct of machine proposals is credited at about 1.5 times,
    which is what was actually measured in deployment, not the ten to twenty
    times sometimes claimed.
    """
    frames = n_clips * frames_per_clip
    boxes = frames * boxes_per_frame
    from_scratch_h = boxes * seconds_per_box / 3600.0
    verified_h = from_scratch_h / float(verify_speedup)
    return {"n_frames": frames, "n_boxes": int(round(boxes)),
            "hours_from_scratch": round(from_scratch_h, 1),
            "hours_verify_and_correct": round(verified_h, 1),
            "note": ("the 4-second box is mouse-gesture time on familiar "
                     "classes; 26.5 seconds is the end-to-end figure and class "
                     "decision time dominates here")}
