# Runbook

Execution, phase by phase, with what to expect and what to do when it goes
wrong. Written for the machine with the two GPUs, not for this repository's
test environment.

Read [GETTING_STARTED.md](GETTING_STARTED.md) first if you have not run the
fake backend yet.

## Day zero, before any GPU work

**Request access to the gated weights.** A human approves each request, so this
blocks everything and nothing you do shortens the wait.

```bash
# Then confirm the token works and the environment is sane.
CUDA_VISIBLE_DEVICES=0 python3 -m vlmlab.cli preflight --config configs/mine.json
```

Exits non-zero if any check fails. Fix every failure before continuing, because
each corresponds to a library default that is wrong and fails silently rather
than loudly.

**Set up three separate environments.** The constraints are mutually
unsatisfiable in one: the detector, the evaluation library and the student
trainer disagree about versions. Keep them apart rather than fighting it.

## Phase one, the annotation assistant

Acceptance here is qualitative, by inspection. You are not claiming a number
yet, and you should not pretend to.

### 1. Extract frames once

```bash
python3 scripts/extract_frames.py --manifest raw.json --out-dir frames \
    --fps 2 --write-manifest manifest.json
```

Decoding must never sit inside a training loop. In a comparable pipeline that
was the actual wall clock: it seeks per frame, tests membership against a list,
and converts to an image, per clip and per epoch, with no worker processes. A
three-minute clip at thirty frames per second means over five thousand full
decodes to retrieve a handful of frames.

This also probes each clip and writes back the real duration and frame count,
which the pipeline needs and refuses to guess.

Expect roughly 8 GB of frames for 150 three-minute clips at two frames per
second, and about 80 GB including a pool ten times larger.

### 2. Measure memory before planning around it

```bash
python3 scripts/benchmark_vram.py --frame frames/clip_0001_000001.jpg --out vram.json
```

**This is experiment one and the single biggest unknown in the whole plan.**
Peak detector memory on a 12GB card is unmeasured anywhere in the literature.
The only nearby data points are under 25 GB on a datacentre card for long
multi-class video, and an out-of-memory report on a 24 GB card. Every cost
estimate downstream depends on what this prints.

The schedule runs cheapest configuration first and stops at the first
out-of-memory failure, so you get a usable answer even when it fails.

### 3. Cost the run before spending it

```bash
python3 -m vlmlab.cli plan --manifest manifest.json --config configs/mine.json
```

At five concepts per clip, 150 three-minute clips is roughly two hours and a
pool ten times larger about twenty. At eight concepts those become four to five
and forty-three to fifty-one. That difference is why prompt restriction is a
mandatory stage.

### 4. Run it, two workers, never one distributed job

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vlmlab.cli run --backend real \
    --manifest manifest.json --config configs/mine.json \
    --frame-dir frames --index 0 --of 2 --out shard0.json &
CUDA_VISIBLE_DEVICES=1 python3 -m vlmlab.cli run --backend real \
    --manifest manifest.json --config configs/mine.json \
    --frame-dir frames --index 1 --of 2 --out shard1.json &
wait
```

There is no peer-to-peer link on this hardware, so sharding a model across the
bus would be communication-bound. Two independent workers have zero inter-card
traffic and scale almost linearly. The shard split is deterministic, disjoint
and exhaustive.

To see which clips each worker will take before launching anything:

```bash
python3 -m vlmlab.cli shard --manifest manifest.json --index 0 --of 2
python3 -m vlmlab.cli shard --manifest manifest.json --index 1 --of 2
```

Useful when one worker dies and you want to rerun only its share, and for
confirming the two lists are disjoint and together cover every clip.

### 5. Read the output critically

| Symptom in the summary | What it means |
|---|---|
| `n_chunks` is 1 | Clip duration is wrong in the manifest. A three-minute clip needs six or more |
| `n_detections` is 0 | Prompts are wrong, or thresholds too high. Check `reasons` |
| `naming.n_named` is 0 | The crop classifier is not wired in, so generic localisation is contributing nothing |
| `naming.n_rejected` dominates | Confidence or margin floor is too high, or the crop classifier is untrained on these classes |
| `labels_never_found` is long | Those classes need hand-drawn seed boxes. Expected for the five with no public boxes |
| `n_background` is 0 | Hard negatives are not being prompted, so you are discarding free negative data |
| `reasons` contains only `kept` | Nothing is being filtered. Suspicious rather than good |
| `n_stitched_tracks` is 0 | Stitching is not happening, which contradicts the chunk count |

## Phase two, measurement

Only once phase one produces usable proposals.

### 1. Choose the frames to annotate

```bash
python3 -m vlmlab.cli gold-select --manifest manifest.json \
    --config configs/mine.json --frames-per-clip 3 --out gold.json
```

Three frames per clip across every clip. On three-minute clips they land about
45 seconds apart, well beyond the two-second floor. Expect 450 frames and
roughly 6 to 10 hours of human time, less if you verify machine proposals
rather than drawing from scratch.

Do **not** densely annotate a few clips. At equal cost that gives less than a
fifth of the effective sample size.

### 2. Annotate, then check progress

```bash
python3 -m vlmlab.cli gold-status --gold gold.json
```

Fill in `boxes` and `exhaustive_for` on each frame. `exhaustive_for` is the list
of classes you actually checked on that frame; a class outside it is treated as
unknown rather than absent, which is what stops honest ignorance counting as a
correct rejection.

### 3. Resolve the protocol

```bash
python3 -m vlmlab.cli protocol --manifest manifest.json --config configs/mine.json
```

Tells you the fold scheme and which statistical test your session count
permits, and warns about every class too sparse to be falsifiable.

### 4. Score

```python
from vlmlab.config import Config
from vlmlab.eval.report import score, compare
from vlmlab.gold import GoldSet

cfg = Config.from_json("configs/mine.json")
gold = GoldSet.load("gold.json")
report = score(gold, predictions, cfg)
print(report.to_dict()["mAP"], report.to_dict()["ci95"])
for w in report.to_dict()["warnings"]:
    print("WARNING:", w)
```

Read the warnings before the number. If the interval is wider than the
difference you care about, you do not have a result.

To compare two pipelines, use `compare`, which pairs by session and refuses to
call a difference below your pre-registered threshold a result.

## Phase three, fine-tuning

```python
from vlmlab.train.schedule import plan_schedule
plan = plan_schedule(n_train=120, cfg=cfg.lora, n_visible_devices=1)
for w in plan["warnings"]:
    print("WARNING:", w)
```

Expect a warning about having only about seven optimiser steps per epoch. That
is real at this label budget, not a misconfiguration, and it is why the step
count is capped rather than trained to convergence.

Then hold out five to eight classes entirely, fine-tune a copy, and sweep the
weight-average coefficient:

```python
from vlmlab.train.retention import choose_held_out_classes, retention_report
from vlmlab.train.sweep import run_sweep, choose_operating_point, format_sweep
```

**Report the median across withheld classes, never the mean.** In the reference
experiment a nine percent mean drop concealed a sixty-one percent median
collapse.

If the sweep falls back to the frozen model, that is a real answer: fine-tuning
bought nothing that retained acceptably. Reduce the step budget and sweep again.

## Phase four, the student

Distil into a small permissively licensed detector, because the teacher is far
too slow to deploy: cost is linear in concepts queried and a three-minute clip
takes minutes rather than seconds.

## Troubleshooting

Symptom, cause, fix. Every row is a verified failure mode, not a guess.

| Symptom | Cause | Fix |
|---|---|---|
| Out of memory immediately, far below the expected footprint | Attention fell back to the eager implementation, costing about 1.26 GB **per layer** | Pass the scaled dot-product implementation and assert it took effect after loading. Checking the argument you passed is not enough; the library downgrades silently |
| Out of memory at eight gigabytes per frame | Inference mode not active | Cannot happen through the provided interface, which wraps it. If you call a model directly, wrap it yourself |
| Out of memory only at the end of epoch one | Evaluation batch size defaulted to eight | It is frozen to one in configuration. If you bypassed that, set it back |
| Detector is 50 to 100 times slower than expected but does not error | The session device defaulted to the processor | Pass the accelerator explicitly. The published example passes an undefined variable |
| Two to four times slower than expected, and quality is worse | Four-bit defaults are wrong three ways: the wrong quantisation type, double quantisation off, and single-precision compute | All three are frozen in configuration. Assert them back after loading |
| Training fits in theory but not in practice | Fused cross-entropy silently did not apply. One toolkit's flag has no branch for the relevant model and logs that it is unsupported | Verify behaviourally: run a labelled forward pass and confirm no logits tensor came back |
| Every clip seems to contain only its opening moment | The container reported no frame count and something fell back to the first N indices | The provided planner raises instead. Re-probe the container |
| Memory and timing figures do not match any run you did | Requested frame rate could not fill the cap, so you got fewer frames than planned | The planner reports whether the cap bound. Make it fatal if you depend on it |
| Fine-grained classes are indistinguishable | Resolution defaulted to thumbnails | Frame size has no default and must be set. There is also a token ceiling for the opposite mistake |
| Effective batch doubled and optimiser steps halved without you changing anything | The launcher saw two cards and went distributed | Pin one device. The schedule planner refuses more than one |
| A source build runs for hours or is killed | You tried to install flash attention, which publishes no wheels | You do not need it. It is frozen off |
| Evaluation library will not import | Upstream is unmaintained and uses numerics aliases removed years ago | Use the maintained fork pinned at 1.3.0 |
| Rare-class precision moves several points for no reason | Default cap of 300 detections per image truncated them | Set it to unlimited. The exporter records this inside the file |
| Validation score is excellent, real footage is poor | The split leaked. Frames or clips from one session landed on both sides | Group by session, and publish the grouped-versus-ungrouped gap as your leakage estimate |
| A reported improvement will not reproduce | It was smaller than the interval | Check the detectable difference before running the comparison, not after |

## The two things that are still unmeasured

`python3 -m vlmlab.cli provenance` lists four values nobody has measured. Two
matter operationally:

- **Peak detector memory on a 12GB card.** No published figure exists. Benchmark it first.
- **A claimed fivefold speedup from graph capture at 512 pixels**, which rests on a single unreproduced report. Plan with the slower figure and treat the speedup as a bonus if it materialises.
