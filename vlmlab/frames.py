"""Frame sampling and chunking. Pure arithmetic, and exactly what the traps corrupt.

Two verified failure modes are handled here rather than documented.

**The silent collapse.** One toolkit computes the sample count as
``min(total, cap, floor(duration * fps))``, so asking for one frame per second
with a sixteen-frame cap yields six frames on a six-second clip, and every
downstream memory and timing figure then describes a configuration that was
never run. Worse, when the container reports zero frames, which is routine for
WebM, fragmented MP4, phone recordings and remuxes, it returns indices zero to
fifteen, meaning **the first half second of every clip**, with no warning.
``plan_frame_grid`` raises instead.

**The chunk limit.** The segmentation task is formally defined only for video
up to thirty seconds. At three minutes that forces at least six sessions per
clip regardless of frame rate, so cross-chunk identity stitching is the common
path rather than an edge case.
"""
import math
from typing import List, Optional, Sequence, Tuple

from vlmlab.types import Chunk, FrameGrid

__all__ = [
    "ZeroFrameCountError", "UnderDeliveredError", "plan_frame_grid",
    "chunk_indices", "keyframe_indices", "estimate_vision_tokens",
    "validate_resolution", "ResolutionError",
]


class ZeroFrameCountError(ValueError):
    """The container reported no frames, so a uniform sample is impossible."""


class UnderDeliveredError(ValueError):
    """The requested rate cannot fill the cap, and the caller demanded it."""


class ResolutionError(ValueError):
    pass


def plan_frame_grid(clip_id, duration_s, container_frame_count, fps, cap,
                    require_cap_bound=False):
    """Plan a uniform frame sample, refusing to guess.

    Raises ``ZeroFrameCountError`` when the frame count is missing or
    non-positive. That is the whole point: the alternative implementation
    returns the first ``cap`` indices, which silently samples only the opening
    moment of every clip.
    """
    if container_frame_count is None or container_frame_count <= 0:
        raise ZeroFrameCountError(
            "clip {!r} reports container_frame_count={!r}. Refusing to fall back to "
            "the first {} indices, which would sample only the start of the clip. "
            "Re-probe the container, or pass a real frame count."
            .format(clip_id, container_frame_count, cap))
    if duration_s is None or duration_s <= 0:
        raise ValueError("clip {!r} has non-positive duration".format(clip_id))
    if fps <= 0:
        raise ValueError("fps must be positive")
    if cap < 1:
        raise ValueError("cap must be at least 1")

    wanted = int(math.floor(duration_s * fps))
    n = max(1, min(container_frame_count, cap, wanted))
    cap_bound = (n == cap)
    if not cap_bound and require_cap_bound:
        raise UnderDeliveredError(
            "clip {!r}: requested {} fps over {}s yields {} frames, below the cap of {}. "
            "Every downstream memory and timing figure would describe a configuration "
            "you did not run.".format(clip_id, fps, duration_s, n, cap))

    if n == 1:
        indices = (0,)
    else:
        step = (container_frame_count - 1) / float(n - 1)
        indices = tuple(int(round(i * step)) for i in range(n))
        # Deduplicate while preserving order, in case rounding collides.
        seen = []
        for i in indices:
            if not seen or i != seen[-1]:
                seen.append(i)
        indices = tuple(seen)

    return FrameGrid(clip_id=clip_id, indices=indices, requested_fps=float(fps),
                     cap=int(cap), cap_bound=bool(cap_bound),
                     source_frame_count=int(container_frame_count))


def chunk_indices(grid, chunk_frames, overlap=0, clip_duration_s=None,
                  pcs_max_seconds=None):
    """Split a frame grid into overlapping backend sessions.

    The number of chunks is bounded below by the task's documented duration
    limit as well as by the frame cap, because exceeding the documented limit
    degrades quality even when memory would allow it.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    if overlap < 0 or overlap >= chunk_frames:
        raise ValueError("overlap must be in [0, chunk_frames)")

    n = grid.n_frames
    effective = chunk_frames
    if clip_duration_s and pcs_max_seconds and pcs_max_seconds > 0:
        min_chunks_by_time = int(math.ceil(clip_duration_s / float(pcs_max_seconds)))
        if min_chunks_by_time > 1:
            by_time = int(math.ceil(n / float(min_chunks_by_time)))
            effective = min(effective, max(1, by_time))

    chunks = []
    start = 0
    cid = 0
    while start < n:
        end = min(start + effective, n)
        idxs = grid.indices[start:end]
        chunks.append(Chunk(chunk_id=cid, clip_id=grid.clip_id, indices=idxs,
                            overlap_with_previous=(overlap if cid > 0 else 0)))
        if end >= n:
            break
        start = end - overlap if overlap else end
        cid += 1
    return tuple(chunks)


def keyframe_indices(grid, stride):
    """Every ``stride``-th sampled frame, always including the first."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return tuple(grid.indices[i] for i in range(0, grid.n_frames, stride))


def estimate_vision_tokens(frame_size, n_frames, patch=14, merge=2):
    """Approximate vision-token count for a frame batch.

    Used to refuse a configuration that would obviously exceed the card,
    which is the other half of the resolution trap.
    """
    if frame_size <= 0 or n_frames <= 0:
        raise ValueError("frame_size and n_frames must be positive")
    per_side = frame_size // (patch * merge)
    return per_side * per_side * n_frames


def validate_resolution(frame_size, n_frames, max_tokens=4096, patch=14, merge=2):
    """Raise when the resolution and frame count together blow the budget."""
    tokens = estimate_vision_tokens(frame_size, n_frames, patch=patch, merge=merge)
    if frame_size < 224:
        raise ResolutionError(
            "frame_size={} trains a fine-grained classifier on thumbnails; "
            "confusable instruments will be indistinguishable".format(frame_size))
    if tokens > max_tokens:
        raise ResolutionError(
            "frame_size={} x {} frames is about {} vision tokens, above the {} budget. "
            "Reduce resolution or frames.".format(frame_size, n_frames, tokens, max_tokens))
    return tokens
