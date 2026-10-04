# Implementation

What is here, what is tested, and what is not. Read
[08-resolved-constraints-and-plan.md](08-resolved-constraints-and-plan.md) for
why the pipeline has this shape, and
[07-engineering-traps-and-measurability.md](07-engineering-traps-and-measurability.md)
for the traps the code enforces.

## Run it now, with no dependencies

```bash
python3 -m unittest discover tests          # 377 tests, standard library only
python3 -m vlmlab.cli run --backend fake \
    --manifest examples/manifest.json --config examples/config.json
```

Both work on a machine with no GPU, no CUDA and no package manager. That is
the point of the design, not a coincidence.

## The rule that makes this testable

**Every value crossing a backend interface is plain Python**: float, int, str,
bool, list, tuple or None. Adapters convert before returning. So the cascade,
the filters, the statistics and the exporters are all exercised here, even
though the real backends need a GPU.

It is enforced, not hoped for. `backends/base.py` validates with exact type
identity rather than an instance check, because a numeric scalar from an array
library passes the latter. `tests/test_py39_compat.py` additionally rejects any
import of a tensor library outside the two adapter directories, and rejects a
module-level import even there.

## Layout

| Path | Tested here | Notes |
|---|---|---|
| `vlmlab/types.py` | yes | Frozen records, zero intra-package imports |
| `vlmlab/config.py` | yes | Provenance per value, frozen trap defaults |
| `vlmlab/registry.py` | yes | Vocabulary, paraphrases, homonym guards |
| `vlmlab/prompts.py` | yes | Generic localisation, per-clip restriction, hard negatives |
| `vlmlab/frames.py` | yes | Frame planning and chunking |
| `vlmlab/splits.py` | yes | Session-grouped folds, leakage gap |
| `vlmlab/geometry.py` | yes | Overlap, suppression, conversions |
| `vlmlab/mil.py` | yes | The clip-label constraint cascade |
| `vlmlab/consistency.py` | yes | Temporal and cross-model filters |
| `vlmlab/federated.py` | yes | Positive, negative, unknown bookkeeping |
| `vlmlab/pipeline.py` | yes | Orchestration, chunking, double propagation |
| `vlmlab/preflight.py` | yes | Pure runtime checks |
| `vlmlab/checkpoint.py` | yes | Resume, JSON not pickle |
| `vlmlab/eval/dist.py` | yes | Incomplete beta, inverse Student-t, exact intervals |
| `vlmlab/eval/metrics.py` | yes | Average precision, eleven fixtures |
| `vlmlab/eval/stats.py` | yes | Cluster bootstrap, permutation, design effects |
| `vlmlab/eval/protocol.py` | yes | Folds, one global threshold |
| `vlmlab/export/coco.py`, `tao.py` | yes | Writers, federated categories |
| `vlmlab/train/schedule.py` | yes | Rank, alpha, steps, warmup |
| `vlmlab/train/rtdetr_dataset.py` | yes | Index, offset, empty targets |
| `vlmlab/backends/base.py`, `fake.py` | yes | Interfaces and the stub |
| `vlmlab/backends/sam3.py` | **no** | Needs a GPU |
| `vlmlab/backends/sam2.py` | **no** | Needs a GPU |
| `vlmlab/backends/crops.py` | **no** | Needs a GPU |
| `vlmlab/backends/_torch_guards.py` | **no** | Logic exercised against stubs |
| `vlmlab/train/lora_adapter.py` | **no** | Needs a GPU |
| `vlmlab/train/rtdetr.py` | **no** | Needs a GPU |

## How the untested files are still constrained

`tests/contract.py` holds the backend contract. The fake backend subclasses it
and runs here; each real adapter subclasses **the same mixin**, skipped unless
a tensor library is present. Same assertions, two subjects. That is the only
honest claim available without a GPU, and it covers exactly the mistakes that
are expensive to find on slow hardware: the centre-versus-corner box format,
the one-concept-per-session limit, running propagation twice, and a tensor
escaping the boundary.

The contract was written **before** the fake, and it immediately caught an
inconsistency in its own fixture plus a case where the fake silently defaulted
to the first frame instead of complaining.

## The traps, and where each is enforced

| Trap | Enforced by |
|---|---|
| Attention defaults to eager, about 1.26 GB per layer | Frozen config, plus a guard reading the effective value back after loading |
| Detector session defaults to the processor, 50 to 100 times slower | Adapter guard on parameters and session device |
| Omitting inference mode costs 8 GB per frame | **Structural.** `detect` is final and wraps the subclass, so it cannot be forgotten |
| Four-bit defaults wrong three ways | Frozen config, plus a guard re-reading the quantisation config |
| Fused cross-entropy flag is a silent no-op | **Behavioural.** A labelled forward pass must return no logits tensor |
| Evaluation batch size defaults to 8 | Frozen config, checked at validation |
| Frame sampling returns the first half second | `plan_frame_grid` raises rather than guessing |
| Resolution wrong in both directions | No default for frame size, plus a token-budget ceiling |
| Launcher silently uses both GPUs | Pure environment check, and the schedule planner refuses |
| Flash attention unbuildable | Frozen off, plus a positive check that it is absent |
| Video decode is the real wall clock | **Structural.** Interfaces accept a frame path, never a video path |

## The tests that carry the most weight

- **No session appears in two folds**, over 25 seeds, with a deliberately leaky splitter as a negative control proving the assertion has power.
- **The cascade never accepts a class outside the clip's label set**, at any confidence up to 1.0.
- **Average precision matches eleven hand-computed fixtures** with the literal constants in the test body. Two of them, half recall at 51/101 and an overlap of exactly one half, catch most implementation errors.
- **Bootstrap coverage** against clustered data with a known mean, which is the only test that catches a percentile off-by-one or resampling records instead of clusters.
- **Planted leakage**: the same data scored with grouped and ungrouped splits, asserting the ungrouped score is inflated. The research's central warning as a regression test.
- **Spreading frames across clips beats depth** at equal annotation cost, by more than fivefold in effective sample size.

## Bugs this process found

Written down because each was silent and each would have produced a plausible wrong number.

1. **Equal-score ties broke on input position** in both average precision and suppression, so shuffling the same detections changed the result. Both now break ties on content.
2. **The split seed did nothing.** A deterministic sort after the shuffle meant equally sized sessions always landed in the same folds, so fold-assignment variance could not be assessed.
3. **The design effect divided by record count instead of cluster count**, so equally sized clusters wrongly reported a penalty of seven rather than one.
4. **The prompt plan handed display strings to the federated deriver**, which keys on canonical names, so the negative set came out empty and the guaranteed-false-positive rule was silently disabled.
5. **The fake propagator defaulted to the first frame** when a seed named an unknown frame, which would have hidden a real wiring error.

## What is not verified

No file importing a tensor library was executed. **No memory figure, no
throughput number and no accuracy number in this repository was produced by
running a model.** Those files say so in their docstrings.

The first task on the GPU machine is `scripts/benchmark_vram.py`, because peak
detector memory on a 12 GB card is unmeasured anywhere in the literature and
every cost estimate depends on it. Run `python3 -m vlmlab.cli provenance` to
see the other three values nobody has measured.

## Order of work

```bash
# 0. Request access to the gated weights. A human approves it, so do this first.
# 1. Extract frames once. Decoding must never sit inside a loop.
python3 scripts/extract_frames.py --manifest raw.json --out-dir frames \
    --fps 2 --write-manifest examples/manifest.json
# 2. Check the environment. Exits non-zero if a trap is live.
CUDA_VISIBLE_DEVICES=0 python3 scripts/preflight.py --config examples/config.json
# 3. Measure memory before planning anything around it.
python3 scripts/benchmark_vram.py --frame frames/clip_0001_000001.jpg
# 4. Cost the run before spending it.
python3 -m vlmlab.cli plan --manifest examples/manifest.json --config examples/config.json
# 5. Two independent single-GPU workers. Never one distributed job.
CUDA_VISIBLE_DEVICES=0 python3 -m vlmlab.cli run --backend real --index 0 --of 2 &
CUDA_VISIBLE_DEVICES=1 python3 -m vlmlab.cli run --backend real --index 1 --of 2 &
```
