"""Record types shared by every module.

Deliberately has **zero intra-package imports**, so anything may depend on it
without creating a cycle, and so the eval and logic layers never have to reach
into ``backends/`` for a record definition.

Every field is a plain Python scalar or a tuple of them. Nothing here may hold
a tensor or an array. ``backends.base`` enforces that at the interface
boundary; this module makes it expressible.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BoxSource", "RejectReason", "Box", "Detection", "TrackPoint", "Track",
    "ClipRecord", "FrameGrid", "Chunk", "MatchTable", "APResult", "CIResult",
    "PermResult", "MDEResult", "Verdict",
]


class BoxSource(str, Enum):
    """Where a box came from. Load-bearing, not decorative.

    The research found that the video path of the reference detector derives
    boxes from masks, so box quality is capped by mask quality and a
    fragmented mask silently yields an inflated box. The image path exposes a
    real detection head. Carrying the provenance lets preflight refuse a
    mask-derived box when boxes are the deliverable.
    """

    DETECTION_HEAD = "detection_head"
    MASK_DERIVED = "mask_derived"
    PROPAGATED = "propagated"
    HUMAN = "human"


class RejectReason(str, Enum):
    """Why the cascade dropped a detection.

    Part of the return type on purpose: without it the cascade cannot be
    tested stage by stage.
    """

    KEPT = "kept"
    KEPT_AS_BACKGROUND = "kept_as_background"
    NOT_IN_CLIP_LABELS = "not_in_clip_labels"
    BELOW_SCORE = "below_score"
    BELOW_PRESENCE = "below_presence"
    AREA_OUT_OF_RANGE = "area_out_of_range"
    SIDE_TOO_SMALL = "side_too_small"
    SUPPRESSED_NMS = "suppressed_nms"
    TEMPORALLY_INCONSISTENT = "temporally_inconsistent"
    NO_CROSS_MODEL_AGREEMENT = "no_cross_model_agreement"
    MASK_FRAGMENTED = "mask_fragmented"
    AMBIGUOUS_LABEL = "ambiguous_label"


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in absolute pixel corner form."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                "degenerate box: ({}, {}, {}, {})".format(self.x1, self.y1, self.x2, self.y2))

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """One detection on one frame.

    ``presence`` is the decoupled image-level presence score where the backend
    provides one. It is the natural multiple-instance gate because it answers
    "is this concept in the frame at all" independently of localisation.
    """

    frame_id: str
    label: str
    box: Box
    score: float
    box_source: BoxSource
    presence: Optional[float] = None
    prompt: Optional[str] = None
    model_id: Optional[str] = None

    def combined_score(self):
        """The documented score-combination rule, when presence is available."""
        if self.presence is None:
            return self.score
        return self.score * self.presence


@dataclass(frozen=True)
class TrackPoint:
    frame_id: str
    box: Box
    score: float


@dataclass(frozen=True)
class Track:
    """A persistent identity across frames within one clip."""

    track_id: int
    label: str
    points: Tuple[TrackPoint, ...]
    box_source: BoxSource
    mask_fragmentation: Optional[float] = None
    chunk_ids: Tuple[int, ...] = ()

    @property
    def n_frames(self):
        return len(self.points)

    @property
    def spans_chunks(self):
        """True when this identity was stitched across a chunk boundary.

        Worth tracking separately: with three-minute clips the segmentation
        task's documented 30-second limit forces at least six chunks, so
        cross-chunk stitching is the common path and the dominant accuracy
        risk rather than an edge case.
        """
        return len(set(self.chunk_ids)) > 1


@dataclass(frozen=True)
class ClipRecord:
    """One labelled clip. ``session_id`` is the grouping unit for every split.

    Grouping by clip is not sufficient. Several clips typically show the same
    physical instrument on the same bench under the same lighting, so a
    clip-level split still leaks instrument identity and background.
    """

    clip_id: str
    session_id: str
    labels: Tuple[str, ...]
    duration_s: Optional[float] = None
    container_frame_count: Optional[int] = None
    fps: Optional[float] = None
    room_id: Optional[str] = None

    def label_set(self):
        return frozenset(self.labels)


@dataclass(frozen=True)
class FrameGrid:
    """The realised frame sampling plan for one clip."""

    clip_id: str
    indices: Tuple[int, ...]
    requested_fps: float
    cap: int
    cap_bound: bool
    source_frame_count: int

    @property
    def n_frames(self):
        return len(self.indices)


@dataclass(frozen=True)
class Chunk:
    """A contiguous window of a clip processed in one backend session."""

    chunk_id: int
    clip_id: str
    indices: Tuple[int, ...]
    overlap_with_previous: int = 0


@dataclass(frozen=True)
class MatchTable:
    """Output of detection-to-ground-truth matching, before the curve.

    Split out from the average-precision curve on purpose: the permutation
    test recomputes the metric hundreds of times, and only the curve half
    needs to re-run.
    """

    scores: Tuple[float, ...]
    is_tp: Tuple[bool, ...]
    n_pos: int
    n_ignored: int = 0

    @property
    def n_dets(self):
        return len(self.scores)


@dataclass(frozen=True)
class APResult:
    """``ap is None`` when there were no positives, never 0.0.

    Reporting 0.0 for an absent class silently drags a macro average down and
    hides how many classes were actually averaged.
    """

    ap: Optional[float]
    n_pos: int
    n_dets: int
    n_ignored: int = 0
    recall_points: int = 101


@dataclass(frozen=True)
class CIResult:
    point: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    n_units: int
    n_resamples: int
    n_degenerate: int = 0
    method: str = "percentile"

    @property
    def width(self):
        if self.lo is None or self.hi is None:
            return None
        return self.hi - self.lo

    @property
    def trustworthy(self):
        """False when too many resamples were degenerate to believe the interval."""
        if self.n_resamples == 0:
            return False
        return (self.n_degenerate / float(self.n_resamples)) <= 0.05


@dataclass(frozen=True)
class PermResult:
    diff: Optional[float]
    p_value: float
    n_perm: int
    exact: bool
    n_groups: int = 0

    @property
    def p_floor(self):
        """The smallest p this test could possibly report.

        At nine groups an exact sign-flip test cannot go below about 0.004, so
        a demand for p < 0.001 is unsatisfiable by construction.
        """
        return 1.0 / (self.n_perm + 1.0)


@dataclass(frozen=True)
class MDEResult:
    mde: float
    n_units: int
    sd_diff: float
    power: float
    alpha: float
    n_comparisons: int


@dataclass(frozen=True)
class Verdict:
    """Per-detection cascade outcome."""

    detection: Detection
    reason: RejectReason
    stage: str

    @property
    def accepted(self):
        return self.reason in (RejectReason.KEPT, RejectReason.KEPT_AS_BACKGROUND)


def to_jsonable(obj):
    """Recursively convert a record to JSON-safe primitives.

    Used by the checkpoint writer. Replaces the sibling project's pickle-based
    checkpoint, which would happily persist a tensor and so leak a GPU object
    into an artefact.
    """
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError("not JSON-safe: {!r} of type {}".format(obj, type(obj).__name__))
