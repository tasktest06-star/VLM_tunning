"""Guards over a loaded model. Needs torch, so every import is lazy.

Each guard reads the *effective* value back from the loaded object rather than
trusting the argument that was passed. That distinction is the whole point:
several of the traps are cases where a library silently downgrades or ignores
what it was given and reports success.

NOT EXECUTED IN THIS REPOSITORY'S TEST RUN. No GPU is available where this was
written. The assertion logic is exercised against stubs in the test suite; the
real checks run the first time you load a model.
"""
from typing import Any, Optional

__all__ = [
    "GuardFailure", "assert_attn_impl", "assert_on_cuda", "assert_dtype",
    "assert_quantisation", "assert_fused_cross_entropy", "assert_inference_mode",
]


class GuardFailure(RuntimeError):
    pass


def assert_attn_impl(model, expected="sdpa"):
    """Read the attention implementation back after loading.

    Checking the keyword you passed is insufficient: the library downgrades
    silently, and eager attention costs roughly 1.26 GB per layer at these
    sequence lengths, which alone exceeds a 12 GB card.
    """
    cfg = getattr(model, "config", None)
    actual = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    if actual is None:
        raise GuardFailure(
            "could not read the effective attention implementation; do not assume "
            "the keyword you passed was honoured")
    if actual != expected:
        raise GuardFailure(
            "attention implementation is {!r}, expected {!r}. Eager attention costs "
            "about 1.26 GB per layer here.".format(actual, expected))
    return actual


def assert_on_cuda(obj):
    """Parameters must live on the accelerator.

    The reference video session defaults its device to the processor and will
    not raise. It simply runs fifty to a hundred times slower, and the usual
    conclusion is that the model is unusable on this card.
    """
    import torch  # noqa: PLC0415
    params = getattr(obj, "parameters", None)
    if callable(params):
        try:
            dev = next(params()).device
        except StopIteration:
            raise GuardFailure("model has no parameters to inspect")
        if dev.type != "cuda":
            raise GuardFailure(
                "parameters are on {!r}. This will not error, it will just run "
                "50 to 100 times slower.".format(str(dev)))
        return str(dev)
    dev = getattr(obj, "inference_device", None) or getattr(obj, "device", None)
    if dev is None:
        raise GuardFailure("cannot determine the device for {!r}".format(type(obj)))
    if "cuda" not in str(dev):
        raise GuardFailure("device is {!r}, expected cuda".format(str(dev)))
    return str(dev)


def assert_dtype(obj, expected="bfloat16"):
    """Must be set on both the model and the session; it defaults to float32."""
    import torch  # noqa: PLC0415
    want = getattr(torch, expected)
    params = getattr(obj, "parameters", None)
    if callable(params):
        try:
            actual = next(params()).dtype
        except StopIteration:
            raise GuardFailure("no parameters to inspect")
    else:
        actual = getattr(obj, "dtype", None)
    if actual is None:
        raise GuardFailure("cannot determine dtype")
    if actual != want:
        raise GuardFailure("dtype is {}, expected {}".format(actual, want))
    return str(actual)


def assert_quantisation(model, quant_type="nf4", double=True,
                        compute_dtype="bfloat16"):
    """Four-bit defaults are wrong three ways at once.

    The quantisation type defaults to a float form rather than normal float,
    double quantisation defaults to off, and the compute dtype resolves to
    single precision. On this card that pushes the dequantised matrix
    multiplies onto the slower path, costing two to four times for no reason,
    and the slowdown is usually blamed on the hardware.
    """
    cfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if cfg is None:
        return None  # not a quantised load
    actual_type = getattr(cfg, "bnb_4bit_quant_type", None)
    actual_double = getattr(cfg, "bnb_4bit_use_double_quant", None)
    actual_compute = getattr(cfg, "bnb_4bit_compute_dtype", None)
    problems = []
    if actual_type != quant_type:
        problems.append("quant_type={!r} expected {!r}".format(actual_type, quant_type))
    if bool(actual_double) != bool(double):
        problems.append("double_quant={!r} expected {!r}".format(actual_double, double))
    if actual_compute is not None and compute_dtype not in str(actual_compute):
        problems.append("compute_dtype={!r} expected {!r}"
                        .format(str(actual_compute), compute_dtype))
    if problems:
        raise GuardFailure("; ".join(problems))
    return True


def assert_fused_cross_entropy(model, forward_fn):
    """Behavioural check, because the trap is a flag that reports success.

    One toolkit's own flag for this has no branch for the relevant model type:
    it logs that the model is unsupported and returns, so reading the flag
    proves nothing. The only reliable test is to run a labelled forward pass
    and confirm no logits tensor was materialised.

    Without the fusion, logits over a 152,000-token vocabulary are built three
    times over, which decides whether training fits in 12 GB at all.
    """
    out = forward_fn(model)
    logits = getattr(out, "logits", "missing")
    loss = getattr(out, "loss", None)
    if logits is not None and logits != "missing":
        raise GuardFailure(
            "a labelled forward pass returned a logits tensor, so the fused "
            "cross entropy did not apply. The flag reporting success is not "
            "evidence; this is.")
    if loss is None:
        raise GuardFailure("labelled forward produced no loss")
    return True


def assert_inference_mode():
    import torch  # noqa: PLC0415
    if not torch.is_inference_mode_enabled():
        raise GuardFailure(
            "inference mode is not active. Omitting it was measured at eight "
            "gigabytes per frame.")
    return True
