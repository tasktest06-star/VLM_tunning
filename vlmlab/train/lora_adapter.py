"""Low-rank fine-tune of a vision-language model, phase three.

NOT EXECUTED. The schedule arithmetic it depends on lives in
``schedule.py`` and is fully unit-tested, because that is the part the traps
corrupt and the part that can be wrong without a GPU.

Fine-tuning is in the plan, which reverses an earlier position. The collapse
figure that prohibition rested on came from a thermal-infrared fine-tune
evaluated on colour images, so it measured a modality shift rather than a
vocabulary effect, and this project is colour throughout. The real evidence
runs the other way: a full fine-tune of an open-vocabulary detector cost under
three points in the wild while gaining nearly ten points on classes it never
saw, and weight averaging finished above the frozen model.

Four rules follow and are enforced here.

* Fine-tune a **copy**, with the step budget capped. At this label count the
  instinct to train to convergence is exactly wrong.
* Freeze the vision tower and the connector. Tuning the visual backbone
  measurably degrades performance.
* Never replace the region-text scoring head with a closed-set classifier.
  That is the thing that genuinely destroys extensibility.
* Sweep the weight-average coefficient afterwards and pick a point on the
  frontier, rather than shipping the fine-tuned weights directly.

Retention is measured by the **median** across held-out classes, never the
mean: in the reference experiment a nine percent mean drop concealed a
sixty-one percent median collapse.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["LoraTrainer", "weight_average", "retention_median"]


class LoraTrainer(object):
    def __init__(self, cfg, registry, device="cuda", verify_guards=True):
        self.cfg = cfg
        self.registry = registry
        self.device = device
        self.verify_guards = bool(verify_guards)
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: PLC0415

        lcfg = self.cfg.lora
        dtype = getattr(torch, self.cfg.detector.compute_dtype)
        quant = None
        if self.cfg.detector.quant_type:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            # All three of these defaults are wrong in the library: the type
            # defaults to a float form, double quantisation to off, and the
            # compute dtype resolves to single precision.
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.detector.quant_type,
                bnb_4bit_use_double_quant=self.cfg.detector.quant_double,
                bnb_4bit_compute_dtype=dtype)

        self._processor = AutoProcessor.from_pretrained(lcfg.base_model)
        self._model = AutoModelForVision2Seq.from_pretrained(
            lcfg.base_model, dtype=dtype, quantization_config=quant,
            attn_implementation=self.cfg.detector.attn_implementation)

        if self.verify_guards:
            from vlmlab.backends._torch_guards import (assert_attn_impl,
                                                       assert_quantisation)
            assert_attn_impl(self._model, self.cfg.detector.attn_implementation)
            assert_quantisation(self._model, self.cfg.detector.quant_type,
                                self.cfg.detector.quant_double,
                                self.cfg.detector.compute_dtype)

        from peft import LoraConfig as PeftLoraConfig, get_peft_model  # noqa: PLC0415
        peft_cfg = PeftLoraConfig(
            r=lcfg.rank,
            # Twice the rank. The library's default scaling penalises higher
            # ranks and the correction is considered essential.
            lora_alpha=lcfg.resolved_alpha(),
            lora_dropout=lcfg.dropout,
            target_modules=list(lcfg.target_modules),
            bias="none", task_type="CAUSAL_LM")
        self._model = get_peft_model(self._model, peft_cfg)

        if lcfg.freeze_vision_tower:
            self._freeze("vision")
        if lcfg.freeze_connector:
            self._freeze("projector")
        self._model.to(self.device)

    def _freeze(self, needle):
        for name, param in self._model.named_parameters():
            if needle in name.lower():
                param.requires_grad = False

    def plan(self, n_train, n_visible_devices=1):
        """Resolve the schedule and surface its warnings before training."""
        from vlmlab.train.schedule import plan_schedule
        return plan_schedule(n_train, self.cfg.lora, n_visible_devices)

    def assert_fused_cross_entropy(self):
        """Behavioural, because the flag reports success while doing nothing."""
        import torch  # noqa: PLC0415
        from vlmlab.backends._torch_guards import assert_fused_cross_entropy

        def forward(model):
            ids = torch.ones((1, 8), dtype=torch.long, device=self.device)
            return model(input_ids=ids, labels=ids)

        return assert_fused_cross_entropy(self._model, forward)


def weight_average(base_state, tuned_state, alpha):
    """Interpolate between the frozen and fine-tuned weights.

    Free, one average per coefficient, and in the reference experiment the
    interpolated model beat both endpoints. This is mitigation number one, not
    a fallback.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    out = {}
    for key in base_state:
        if key not in tuned_state:
            out[key] = base_state[key]
            continue
        out[key] = base_state[key] * (1.0 - alpha) + tuned_state[key] * alpha
    return out


def retention_median(per_class_before, per_class_after):
    """Median retention across held-out classes, plus the mean for contrast.

    Report the median. In the reference experiment the mean fell nine percent
    while the median fell sixty-one, so a mean-based retention metric hides
    exactly the failure it is meant to detect.
    """
    import statistics
    shared = sorted(set(per_class_before) & set(per_class_after))
    if not shared:
        return None
    before = [per_class_before[c] for c in shared]
    after = [per_class_after[c] for c in shared]
    return {
        "n_classes": len(shared),
        "median_before": statistics.median(before),
        "median_after": statistics.median(after),
        "median_delta": statistics.median(after) - statistics.median(before),
        "mean_before": statistics.mean(before),
        "mean_after": statistics.mean(after),
        "mean_delta": statistics.mean(after) - statistics.mean(before),
        "note": ("report the median; a mean-based retention metric concealed a "
                 "61 percent median collapse behind a 9 percent mean drop"),
    }
