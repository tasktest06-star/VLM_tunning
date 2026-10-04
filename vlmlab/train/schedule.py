"""Training-schedule arithmetic. Pure stdlib, and exactly what the traps corrupt.

Separated from the adapter because every one of these numbers can be wrong
without a GPU, and getting them wrong is silent:

* a launcher that quietly uses both cards doubles the effective batch and
  halves the optimiser steps, which changes warmup and the schedule far more
  than the throughput gain is worth;
* the common low-rank implementation's default scaling penalises higher ranks,
  so alpha must be set to twice the rank;
* below a thousand clips, overfitting sets in after the second epoch, so the
  step budget has to be capped rather than left to converge.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "resolved_alpha", "rank_for_label_budget", "steps_per_epoch",
    "total_steps", "warmup_steps", "ScheduleError", "plan_schedule",
    "effective_batch_size",
]


class ScheduleError(ValueError):
    pass


def resolved_alpha(rank, alpha=None):
    """Alpha is twice the rank unless explicitly overridden."""
    if rank < 1:
        raise ScheduleError("rank must be at least 1")
    return 2 * rank if alpha is None else alpha


def rank_for_label_budget(n_clips):
    """Rank and target modules by label count, per the research.

    Returns ``(rank, target_modules, note)``.
    """
    if n_clips < 50:
        return 4, ("q_proj", "k_proj", "v_proj", "o_proj"), (
            "below 50 clips, prefer a frozen-feature probe over any fine-tune")
    if n_clips <= 300:
        return 8, ("q_proj", "k_proj", "v_proj", "o_proj"), (
            "attention projections only; the feed-forward layers overfit at this size")
    if n_clips <= 1000:
        return 16, ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"), (
            "feed-forward layers become safe to include")
    if n_clips <= 5000:
        return 32, ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"), ""
    return 64, ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"), (
        "at this rank enable rank-stabilised scaling")


def effective_batch_size(per_device, grad_accum, n_visible_devices):
    """What the optimiser actually sees.

    The trap is that a launcher switches to distributed training whenever more
    than one device is visible, so the effective batch silently doubles.
    """
    if per_device < 1 or grad_accum < 1 or n_visible_devices < 1:
        raise ScheduleError("batch, accumulation and device count must be positive")
    return per_device * grad_accum * n_visible_devices


def steps_per_epoch(n_train, per_device, grad_accum, n_visible_devices=1):
    eff = effective_batch_size(per_device, grad_accum, n_visible_devices)
    return max(1, n_train // eff)


def total_steps(n_train, per_device, grad_accum, epochs, n_visible_devices=1):
    return steps_per_epoch(n_train, per_device, grad_accum, n_visible_devices) * epochs


def warmup_steps(total, ratio):
    if not (0.0 <= ratio < 1.0):
        raise ScheduleError("warmup ratio must be in [0, 1)")
    return max(1, int(round(total * ratio))) if total > 0 else 0


def plan_schedule(n_train, cfg, n_visible_devices=1):
    """Resolve a full schedule and flag anything degenerate.

    Deliberately warns rather than silently proceeding when the step count is
    so small that warmup and the schedule stop being meaningful, which is the
    normal situation at this label budget.
    """
    if n_visible_devices != 1:
        raise ScheduleError(
            "{} visible devices. Pin exactly one: a launcher that switches to "
            "distributed training doubles the effective batch and halves the "
            "optimiser steps, which changes the schedule more than the throughput "
            "gain is worth. Shard clips across two processes instead."
            .format(n_visible_devices))

    spe = steps_per_epoch(n_train, cfg.per_device_batch_size, cfg.grad_accum_steps, 1)
    total = spe * cfg.epochs
    warm = warmup_steps(total, cfg.warmup_ratio)
    alpha = resolved_alpha(cfg.rank, cfg.alpha)
    rec_rank, rec_modules, rank_note = rank_for_label_budget(n_train)

    warnings = []
    if spe < 10:
        warnings.append(
            "only {} optimiser steps per epoch at {} training clips with an effective "
            "batch of {}. Warmup and the cosine schedule are close to meaningless; "
            "consider lowering gradient accumulation."
            .format(spe, n_train, effective_batch_size(
                cfg.per_device_batch_size, cfg.grad_accum_steps, 1)))
    if cfg.epochs > 2 and n_train <= 1000:
        warnings.append(
            "{} epochs at {} clips. Overfitting sets in after epoch two below a "
            "thousand clips.".format(cfg.epochs, n_train))
    if cfg.rank > rec_rank:
        warnings.append(
            "rank {} exceeds the recommended {} for {} clips; higher ranks memorise "
            "the answer distribution at this size. {}"
            .format(cfg.rank, rec_rank, n_train, rank_note))
    if alpha != 2 * cfg.rank:
        warnings.append(
            "alpha {} is not twice the rank ({}). The common implementation's default "
            "scaling penalises higher ranks and the correction is considered essential."
            .format(alpha, 2 * cfg.rank))
    if not cfg.freeze_vision_tower:
        warnings.append(
            "the vision tower is trainable. Tuning the visual backbone measurably "
            "degrades performance, at p = 0.004.")
    if not cfg.require_fused_cross_entropy:
        warnings.append(
            "fused cross entropy is not required. Without it the logits tensor over a "
            "152,000-token vocabulary is materialised three times, which decides "
            "whether this fits in 12 GB.")

    return {
        "steps_per_epoch": spe,
        "total_steps": total,
        "warmup_steps": warm,
        "effective_batch": effective_batch_size(
            cfg.per_device_batch_size, cfg.grad_accum_steps, 1),
        "rank": cfg.rank,
        "alpha": alpha,
        "recommended_rank": rec_rank,
        "recommended_modules": rec_modules,
        "warnings": warnings,
    }
