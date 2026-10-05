"""Configuration, with provenance as a field rather than a comment.

Two ideas carry their weight here.

**Provenance.** Every value records whether it was verified against a primary
source, estimated, or never measured by anyone. The research produced a long
list of numbers that look equally authoritative but are not, so the
distinction is kept in the data and asserted by tests: no frozen field may be
unverified, and no unmeasured value may feed a gate threshold.

**Frozen fields.** Several of the engineering traps are library defaults that
are wrong and fail silently. Those settings are not overridable from a config
file. ``allow_unsafe=True`` exists only so a test can prove the guard fires.
"""
import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Provenance", "Param", "FrozenFieldError", "MissingRequiredError",
    "ConfigError", "DetectorConfig", "PropagatorConfig", "CascadeConfig",
    "EvalConfig", "LoraConfig", "StudentConfig", "Config", "FROZEN_PATHS",
]


class Provenance(str, Enum):
    VERIFIED = "verified"
    ESTIMATED = "estimated"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class Param:
    """A value that knows where it came from."""

    value: Any
    provenance: Provenance
    source: str
    note: str = ""


class ConfigError(Exception):
    pass


class FrozenFieldError(ConfigError):
    pass


class MissingRequiredError(ConfigError):
    pass


# Dotted paths that a config file may not override. Each corresponds to a
# verified engineering trap where the library default is wrong and silent.
FROZEN_PATHS = frozenset([
    "detector.attn_implementation",      # defaults to eager, ~1.26 GB per layer
    "detector.quant_type",               # defaults to fp4, not nf4
    "detector.quant_double",             # defaults to off
    "detector.compute_dtype",            # resolves to float32, halving throughput
    "detector.allow_flash_attention",    # unbuildable from the package index
    "detector.require_detection_head_boxes",
    "eval.batch_size",                   # defaults to 8, OOMs at end of epoch one
    "eval.per_class_thresholds",         # forbidden at this positive count
])

_PROVENANCE = {}


def _p(path, value, provenance, source, note=""):
    _PROVENANCE[path] = Param(value, provenance, source, note)
    return value


@dataclass
class DetectorConfig:
    """Keyframe detector settings.

    ``frame_size`` has no default on purpose. The research found resolution
    defaults wrong in both directions: one toolkit silently trains on 256-pixel
    thumbnails, while a raw processor sends 1080p frames through to roughly
    twelve thousand vision tokens and tens of gigabytes. Forcing the caller to
    state it removes both failure modes.
    """

    model_id: str = _p("detector.model_id", "facebook/sam3", Provenance.VERIFIED,
                       "docs/06 section 2",
                       "facebook/sam3.1 ships no transformers integration")
    frame_size: Optional[int] = None
    keyframe_stride: int = _p("detector.keyframe_stride", 8, Provenance.ESTIMATED,
                              "docs/03 section 6.1")
    sample_fps: float = _p("detector.sample_fps", 2.0, Provenance.ESTIMATED,
                           "docs/03 section 6.3")
    score_threshold: float = _p("detector.score_threshold", 0.60, Provenance.ESTIMATED,
                                "docs/03 section 3.8", "default is 0.5")
    new_detection_threshold: float = _p("detector.new_detection_threshold", 0.85,
                                        Provenance.ESTIMATED, "docs/03 section 3.8",
                                        "default is 0.7")
    nms_iou: float = _p("detector.nms_iou", 0.10, Provenance.ESTIMATED, "docs/03")
    max_objects: int = _p("detector.max_objects", 32, Provenance.ESTIMATED,
                          "docs/03", "library default 10000 effectively disables the cap")
    max_concepts_per_clip: int = _p("detector.max_concepts_per_clip", 5,
                                    Provenance.ESTIMATED, "docs/07 section 2",
                                    "cost is linear in concepts; restriction is mandatory")
    peak_vram_gb: Optional[float] = _p("detector.peak_vram_gb", None,
                                       Provenance.UNMEASURED, "docs/03 section 7C",
                                       "unmeasured anywhere; run scripts/benchmark_vram.py")
    # --- frozen ---
    attn_implementation: str = "sdpa"
    quant_type: str = "nf4"
    quant_double: bool = True
    compute_dtype: str = "bfloat16"
    allow_flash_attention: bool = False
    require_detection_head_boxes: bool = True


@dataclass
class PropagatorConfig:
    model_id: str = _p("propagator.model_id", "facebook/sam2.1-hiera-small",
                       Provenance.VERIFIED, "docs/08 phase one step 7",
                       "Apache 2.0 for code and weights")
    chunk_frames: int = _p("propagator.chunk_frames", 300, Provenance.VERIFIED,
                           "docs/07 section 3")
    chunk_overlap: int = _p("propagator.chunk_overlap", 8, Provenance.ESTIMATED,
                            "needed for cross-chunk identity matching")
    pcs_max_seconds: float = _p("propagator.pcs_max_seconds", 30.0, Provenance.VERIFIED,
                                "docs/06 section 4",
                                "the task is defined only up to 30s; 3-minute clips need >= 6 chunks")
    stitch_iou: float = _p("propagator.stitch_iou", 0.5, Provenance.ESTIMATED,
                           "cross-chunk identity matching")
    offload_state_to_cpu: bool = True
    offload_video_to_cpu: bool = True


@dataclass
class CascadeConfig:
    """The accept and reject cascade over pseudo-labels."""

    score_threshold: float = _p("cascade.score_threshold", 0.30,
                                Provenance.ESTIMATED, "docs/02",
                                "a second gate after the detector's own threshold")
    nms_iou: float = _p("cascade.nms_iou", 0.50, Provenance.ESTIMATED, "docs/02")
    presence_threshold: float = _p("cascade.presence_threshold", 0.50,
                                   Provenance.ESTIMATED, "docs/04 stage B")
    min_track_frames: int = _p("cascade.min_track_frames", 8, Provenance.ESTIMATED,
                               "docs/03 section 3")
    min_area_ratio: float = _p("cascade.min_area_ratio", 0.001, Provenance.ESTIMATED, "docs/02")
    max_area_ratio: float = _p("cascade.max_area_ratio", 0.90, Provenance.ESTIMATED, "docs/02")
    min_side_px: int = _p("cascade.min_side_px", 20, Provenance.ESTIMATED, "docs/02")
    temporal_min_appearances: int = _p("cascade.temporal_min_appearances", 2,
                                       Provenance.ESTIMATED, "docs/04")
    cross_model_iou: float = _p("cascade.cross_model_iou", 0.5, Provenance.ESTIMATED,
                                "docs/04 section 2.2",
                                "pseudo-box precision for two-detector agreement is "
                                "unmeasured in the literature; half-day experiment")
    n_hard_negatives: int = _p("cascade.n_hard_negatives", 4, Provenance.ESTIMATED,
                               "docs/07 section 2", "documents say 3 to 5")
    crop_min_confidence: float = _p("cascade.crop_min_confidence", 0.35,
                                    Provenance.ESTIMATED, "docs/07 section 2",
                                    "refuse to name a crop below this")
    crop_min_margin: float = _p("cascade.crop_min_margin", 0.10,
                                Provenance.ESTIMATED, "docs/07 section 2",
                                "margin over the runner-up; the confusable "
                                "siblings are where a forced choice goes wrong")
    max_mask_fragmentation: float = _p("cascade.max_mask_fragmentation", 0.35,
                                       Provenance.UNMEASURED, "docs/06 section 4",
                                       "audit required but no criterion given")


@dataclass
class EvalConfig:
    """Evaluation protocol.

    The fold rule follows the research: leave one session out when sessions
    are few, otherwise five grouped folds. The statistical test switches on
    group count rather than preference, because below roughly fifteen groups a
    bootstrap is resampling too few exchangeable units to be informative.
    """

    frames_per_clip: int = _p("eval.frames_per_clip", 3, Provenance.VERIFIED,
                              "docs/05 section 3", "450 to 600 gold frames total")
    min_frame_spacing_s: float = _p("eval.min_frame_spacing_s", 2.0, Provenance.VERIFIED,
                                    "docs/05 section 3",
                                    "3-minute clips give 45-60s naturally, so correlation "
                                    "is lower than the 0.6 assumed")
    loso_below_sessions: int = _p("eval.loso_below_sessions", 12, Provenance.ESTIMATED,
                                  "docs/08 section 2", "fold count was left unfixed")
    n_folds: int = _p("eval.n_folds", 5, Provenance.VERIFIED, "docs/05 section 3")
    permutation_below_groups: int = _p("eval.permutation_below_groups", 15,
                                       Provenance.VERIFIED, "docs/07 section 1")
    n_resamples: int = _p("eval.n_resamples", 2000, Provenance.VERIFIED,
                          "docs/07 section 1", "10000 buys nothing over 2000")
    iou_threshold: float = _p("eval.iou_threshold", 0.5, Provenance.VERIFIED, "docs/06")
    max_dets_per_image: int = _p("eval.max_dets_per_image", 0, Provenance.VERIFIED,
                                 "docs/06 section 9",
                                 "0 means unlimited; the default 300 cap moves rare-class "
                                 "precision by about 7 points")
    pre_registered_mde: float = _p("eval.pre_registered_mde", 10.0, Provenance.VERIFIED,
                                   "docs/08 phase two",
                                   "refuse to report a difference below this")
    seed: int = 0
    # --- frozen ---
    batch_size: int = 1
    per_class_thresholds: bool = False


@dataclass
class LoraConfig:
    """Low-rank adaptation, phase three.

    Rank scales with label count. At about 150 clips the research says use a
    small rank on attention projections only, cap the schedule at one or two
    epochs, and prefer a smaller backbone over a crippled larger one.
    """

    base_model: str = _p("lora.base_model", "Qwen/Qwen2.5-VL-3B-Instruct",
                         Provenance.ESTIMATED, "docs/08 section 3",
                         "documents say prefer 2-4B but name no model")
    rank: int = _p("lora.rank", 8, Provenance.VERIFIED, "docs/08 section 3",
                   "4 to 8 at 150 clips, attention only")
    alpha: Optional[int] = None  # derived as 2 * rank
    dropout: float = _p("lora.dropout", 0.05, Provenance.UNMEASURED, "docs/08",
                        "never mentioned in any document")
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = _p("lora.learning_rate", 2e-4, Provenance.VERIFIED,
                              "docs/08 section 3")
    epochs: int = _p("lora.epochs", 2, Provenance.VERIFIED, "docs/08 section 3",
                     "overfitting sets in after epoch two below 1000 clips")
    warmup_ratio: float = _p("lora.warmup_ratio", 0.03, Provenance.VERIFIED, "docs/08")
    weight_decay: float = _p("lora.weight_decay", 0.0, Provenance.VERIFIED, "docs/08")
    grad_clip: float = _p("lora.grad_clip", 1.0, Provenance.VERIFIED, "docs/08")
    per_device_batch_size: int = _p("lora.per_device_batch_size", 1, Provenance.ESTIMATED,
                                    "docs/08", "unspecified for the VLM run")
    grad_accum_steps: int = _p("lora.grad_accum_steps", 16, Provenance.ESTIMATED, "docs/08")
    frames_per_clip: int = _p("lora.frames_per_clip", 16, Provenance.VERIFIED, "docs/08")
    freeze_vision_tower: bool = True
    freeze_connector: bool = True
    require_fused_cross_entropy: bool = True
    held_out_classes: Tuple[str, ...] = ()
    weight_average_alphas: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def resolved_alpha(self):
        """Alpha is twice the rank.

        The common implementation's default scaling penalises higher ranks and
        one study calls the correction essential.
        """
        return self.alpha if self.alpha is not None else 2 * self.rank


@dataclass
class StudentConfig:
    """Distilled deployable detector, phase four."""

    model_id: str = _p("student.model_id", "PekingU/rtdetr_r50vd", Provenance.VERIFIED,
                       "docs/02 licence table", "Apache 2.0")
    epochs: int = _p("student.epochs", 50, Provenance.ESTIMATED, "docs/02")
    batch_size: int = _p("student.batch_size", 8, Provenance.ESTIMATED, "docs/02")
    learning_rate: float = _p("student.learning_rate", 1e-4, Provenance.ESTIMATED, "docs/02")
    weight_decay: float = _p("student.weight_decay", 1e-4, Provenance.ESTIMATED, "docs/02")
    image_size: int = _p("student.image_size", 640, Provenance.ESTIMATED, "docs/02")
    inference_threshold: float = _p("student.inference_threshold", 0.01, Provenance.VERIFIED,
                                    "sibling project discipline",
                                    "low threshold so the precision-recall curve is complete")


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    clip_duration_s: Optional[float] = _p("clip_duration_s", 180.0, Provenance.ESTIMATED,
                                          "user confirmed about 3 minutes")
    intra_clip_icc: Optional[float] = _p("intra_clip_icc", None, Provenance.UNMEASURED,
                                         "docs/05 section 3",
                                         "measure on the first 20 clips")

    @classmethod
    def from_dict(cls, data, allow_unsafe=False):
        """Build from a plain mapping, refusing to override a frozen field."""
        if not isinstance(data, Mapping):
            raise ConfigError("config must be a mapping")
        cfg = cls()
        sections = {
            "detector": cfg.detector, "propagator": cfg.propagator,
            "cascade": cfg.cascade, "eval": cfg.eval,
            "lora": cfg.lora, "student": cfg.student,
        }
        for key, value in data.items():
            # JSON has no comments, so an underscore prefix is the convention
            # for documenting a config file. Real typos are still refused.
            if key.startswith("_"):
                continue
            if key in sections:
                if not isinstance(value, Mapping):
                    raise ConfigError("section {} must be a mapping".format(key))
                target = sections[key]
                valid = {f.name for f in fields(target)}
                for sub, subval in value.items():
                    if sub.startswith("_"):
                        continue
                    path = "{}.{}".format(key, sub)
                    if sub not in valid:
                        raise ConfigError("unknown setting {}".format(path))
                    if path in FROZEN_PATHS and not allow_unsafe:
                        raise FrozenFieldError(
                            "{} is frozen: {}".format(path, _frozen_reason(path)))
                    if isinstance(subval, list):
                        subval = tuple(subval)
                    setattr(target, sub, subval)
            elif key in {f.name for f in fields(cls)}:
                setattr(cfg, key, value)
            else:
                raise ConfigError("unknown setting {}".format(key))
        return cfg

    @classmethod
    def from_json(cls, path, allow_unsafe=False):
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), allow_unsafe=allow_unsafe)

    def validate(self):
        """Raise on anything that would silently produce a wrong result."""
        errs = []
        if self.detector.frame_size is None:
            errs.append(
                "detector.frame_size has no default and must be set explicitly. "
                "Resolution defaults are wrong in both directions: too small "
                "trains a fine-grained classifier on thumbnails, too large "
                "exceeds the card.")
        if self.eval.batch_size != 1:
            errs.append("eval.batch_size must be 1")
        if self.eval.per_class_thresholds:
            errs.append("per-class thresholds are forbidden at this positive count")
        if self.detector.attn_implementation != "sdpa":
            errs.append("detector.attn_implementation must be sdpa")
        if self.detector.quant_type != "nf4" or not self.detector.quant_double:
            errs.append("quantisation must be nf4 with double quantisation")
        if self.detector.compute_dtype != "bfloat16":
            errs.append("compute dtype must be bfloat16")
        if self.detector.allow_flash_attention:
            errs.append("flash attention is not installable from the package index")
        if not (1 <= self.cascade.n_hard_negatives <= 8):
            errs.append("cascade.n_hard_negatives should be between 1 and 8")
        if self.eval.frames_per_clip < 1:
            errs.append("eval.frames_per_clip must be at least 1")
        if self.detector.max_concepts_per_clip < 1:
            errs.append("detector.max_concepts_per_clip must be at least 1")
        if errs:
            raise ConfigError("; ".join(errs))
        return self

    def min_chunks_per_clip(self):
        """How many backend sessions one clip needs.

        Bounded below by the task's documented 30-second definition, not only
        by the frame cap. At three minutes that is six, so cross-chunk
        identity stitching is the common path.
        """
        if self.clip_duration_s is None:
            return 1
        import math
        by_time = int(math.ceil(self.clip_duration_s / self.propagator.pcs_max_seconds))
        n_frames = int(self.clip_duration_s * self.detector.sample_fps)
        by_frames = int(math.ceil(n_frames / float(self.propagator.chunk_frames)))
        return max(1, by_time, by_frames)

    def estimated_keyframes_per_clip(self):
        if self.clip_duration_s is None:
            return 0
        return int(self.clip_duration_s * self.detector.sample_fps) // self.detector.keyframe_stride

    def provenance_report(self):
        """Group every recorded parameter by how well it is grounded."""
        out = {p.value: [] for p in Provenance}
        for path, param in sorted(_PROVENANCE.items()):
            out[param.provenance.value].append((path, param.value, param.source, param.note))
        return out


def _frozen_reason(path):
    reasons = {
        "detector.attn_implementation":
            "eager attention costs roughly 1.26 GB per layer at these sequence lengths",
        "detector.quant_type": "the library default is fp4, not nf4",
        "detector.quant_double": "double quantisation defaults to off",
        "detector.compute_dtype":
            "compute dtype resolves to float32, halving throughput on this card",
        "detector.allow_flash_attention":
            "no wheels are published; a source build takes hours or is killed",
        "detector.require_detection_head_boxes":
            "the video path derives boxes from masks, capping box quality",
        "eval.batch_size":
            "the library default of 8 causes an out-of-memory failure at the end of epoch one",
        "eval.per_class_thresholds":
            "fitting per-class thresholds on 2 to 6 positives has a standard error "
            "wider than the interval being searched",
    }
    return reasons.get(path, "verified engineering trap")
