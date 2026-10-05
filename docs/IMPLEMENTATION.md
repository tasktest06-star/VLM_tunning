# Implementation

What is here, what is tested, and what is not. Read
[08-resolved-constraints-and-plan.md](08-resolved-constraints-and-plan.md) for
why the pipeline has this shape, and
[07-engineering-traps-and-measurability.md](07-engineering-traps-and-measurability.md)
for the traps the code enforces.

## Run it now, with no dependencies

```bash
python3 -m unittest discover tests          # 500 tests, standard library only
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
| `vlmlab/naming.py` | yes | Names generic detections under the clip-label constraint |
| `vlmlab/gold.py` | yes | Gold-frame selection, scoring records, annotation budget |
| `vlmlab/eval/report.py` | yes | End-to-end scoring and paired comparison |
| `vlmlab/eval/tracking.py` | yes | Detection and association accuracy over tracks |
| `vlmlab/eval/crosscheck.py` | n/a | Optional comparison, skipped without the compiled library |
| `vlmlab/data/public.py` | yes | Public box datasets, mapping and session re-splitting |
| `vlmlab/data/imagenet21k.py` | yes | Classification crops for the fine-grained head |
| `vlmlab/data/rf100vl.py` | yes | The free dry run before spending annotation budget |
| `vlmlab/train/retention.py` | yes | Held-out-class retention, reported by median |
| `vlmlab/train/sweep.py` | yes | Weight-average sweep and operating-point choice |
| `vlmlab/export/rtdetr_yaml.py` | yes | Student dataset descriptor, write-only |
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
- **An identity switch halves association accuracy while leaving detection accuracy untouched**, which is the property that makes the tracking metric worth computing when every clip is stitched.
- **The median catches what the mean hides**: a case where four of six withheld classes collapse while the mean moves seven percent is flagged, which is the exact pattern that concealed a sixty-one percent median collapse in the reference experiment.
- **Generic localisation now contributes accepted boxes** rather than zero.

## The architecture is now actually wired

Earlier the two halves existed separately and the join was missing, so generic
localisation contributed **zero** accepted boxes: detections came back labelled
with the prompt that found them, failed the clip-label test, and were
discarded. `naming.py` closes that. Measured on the fake backend, accepted boxes
went from 12 to 18 on the same clip, and the leftover generic labels went to
none.

Two constraints keep it honest. The classifier's vocabulary is restricted to
the clip's own labels plus its hard negatives, which is the multiple-instance
constraint applied to naming rather than detection. And a crop that cannot be
named confidently is rejected rather than guessed, requiring both an absolute
floor and a margin over the runner-up, because the confusable siblings are
exactly where a forced choice would be wrong and would look confident.

## What the data modules establish

Running `python3 -m vlmlab.cli datasets` prints the current position:

- Nine vocabulary classes have downloadable public boxes, roughly 15,700 instances across four permissively licensed sets.
- Six have none anywhere.
- **Classification crops rescue four of those six**, about 3,600 images, because research-only use unblocked that corpus. Only two classes remain genuinely unsupplied and need your own footage.

The balanced cap for the crop head comes out at 65 images per class, set by the
sparsest target. Equal allocation is enforced in the plan because the macro
average weights classes equally and proportional allocation inflates its
variance by about a third.

## Bugs this process found

Written down because each was silent and each would have produced a plausible wrong number.

1. **Equal-score ties broke on input position** in both average precision and suppression, so shuffling the same detections changed the result. Both now break ties on content.
2. **The split seed did nothing.** A deterministic sort after the shuffle meant equally sized sessions always landed in the same folds, so fold-assignment variance could not be assessed.
3. **The design effect divided by record count instead of cluster count**, so equally sized clusters wrongly reported a penalty of seven rather than one.
4. **The prompt plan handed display strings to the federated deriver**, which keys on canonical names, so the negative set came out empty and the guaranteed-false-positive rule was silently disabled.
5. **The fake propagator defaulted to the first frame** when a seed named an unknown frame, which would have hidden a real wiring error.
6. **Renamed crops inherited the presence score of the generic concept** that found them, so the presence gate was judging a label against evidence about a different thing. Presence is now dropped on rename and the reason recorded.
7. **The sweep's refusal branch was unreachable**, because the frozen model always satisfies its own retention floor. Choosing the frozen model is now reported as a real outcome, meaning fine-tuning bought nothing acceptable, rather than as no decision.
8. **Three dangling references**: a cross-check module and a dataset descriptor were named in documentation but did not exist, and the two-GPU command the implementation guide printed used flags the command line did not accept.

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
