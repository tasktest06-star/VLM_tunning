"""Run-scoped logging, plus the one line that is the audit trail for the traps."""
import logging
import os
import sys
from typing import Any, Mapping, Optional

__all__ = ["get_logger", "log_realised_config"]


def get_logger(name="vlmlab", log_dir=None, run_id=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(stream)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fname = "{}.log".format(run_id or "run")
        fh = logging.FileHandler(os.path.join(log_dir, fname))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s"))
        logger.addHandler(fh)
    return logger


def log_realised_config(logger, cfg, frame_grid=None, n_visible_devices=None):
    """Dump what was actually resolved, not what was requested.

    This single call is the audit trail for most of the engineering traps:
    attention implementation, compute dtype, quantisation, the realised frame
    grid and the visible device count. Several traps are invisible precisely
    because the requested value and the effective value differ.
    """
    d = cfg.detector
    logger.info("detector      model=%s frame_size=%s attn=%s dtype=%s quant=%s/double=%s",
                d.model_id, d.frame_size, d.attn_implementation,
                d.compute_dtype, d.quant_type, d.quant_double)
    logger.info("detector      stride=%s sample_fps=%s max_concepts=%s max_objects=%s",
                d.keyframe_stride, d.sample_fps, d.max_concepts_per_clip, d.max_objects)
    p = cfg.propagator
    logger.info("propagator    model=%s chunk=%s overlap=%s pcs_limit=%ss offload=%s/%s",
                p.model_id, p.chunk_frames, p.chunk_overlap, p.pcs_max_seconds,
                p.offload_state_to_cpu, p.offload_video_to_cpu)
    logger.info("eval          batch=%s frames/clip=%s folds=%s resamples=%s mde=%s",
                cfg.eval.batch_size, cfg.eval.frames_per_clip, cfg.eval.n_folds,
                cfg.eval.n_resamples, cfg.eval.pre_registered_mde)
    if frame_grid is not None:
        logger.info("frame grid    n=%s cap_bound=%s source_frames=%s",
                    frame_grid.n_frames, frame_grid.cap_bound,
                    frame_grid.source_frame_count)
    if n_visible_devices is not None:
        logger.info("devices       visible=%s (must be 1; shard clips across processes)",
                    n_visible_devices)
    if d.peak_vram_gb is None:
        logger.warning("detector peak VRAM is UNMEASURED. No published figure exists "
                       "for a 12 GB card. Run scripts/benchmark_vram.py first.")
