# Getting started

Ten minutes, no installation, no GPU. By the end you will have run the whole
pipeline and understood what it printed.

## 1. Check it works

```bash
cd VLM_tunning
python3 -m unittest discover tests
```

Expected, on any machine with Python 3.9 or newer and nothing installed:

```
Ran 509 tests in 15s

OK
```

If that passes, the standard-library core is sound. There is nothing to
install, because the core deliberately depends on nothing. That is what lets
you run and test it before touching a GPU.

## 2. Run the pipeline on the fake backend

```bash
python3 -m vlmlab.cli run --backend fake \
    --manifest examples/manifest.json --config examples/config.json
```

The fake backend is not a simulation of a model. It is a deterministic stub
that implements exactly the same interface the real detector will, so running
it exercises the real orchestration: frame planning, chunking, prompt
restriction, detection, propagation in both directions, cross-chunk stitching,
the accept-and-reject cascade, and the exporters.

## 3. Read what it printed

One clip's summary looks like this:

```json
{
  "clip_id": "clip_0001",
  "session_id": "lab_a_day1",
  "n_frames": 180,
  "n_chunks": 8,
  "n_concepts": 5,
  "n_detections": 40,
  "n_accepted": 16,
  "n_background": 18,
  "n_tracks": 61,
  "n_stitched_tracks": 12,
  "labels_never_found": [],
  "reasons": { "kept_as_background": 18, "below_score": 6, "kept": 16 },
  "naming": { "n_named": 10, "n_rejected": 2, "n_passthrough": 30 }
}
```

Field by field, because every one of these is telling you something:

| Field | Meaning | What to watch for |
|---|---|---|
| `n_frames` | Frames sampled from the clip | At 2 frames per second a three-minute clip gives 360 |
| `n_chunks` | Backend sessions needed | **Never 1 for a three-minute clip.** The segmentation task is defined only to 30 seconds, so six is the floor |
| `n_concepts` | Prompts billed for this clip | Cost is linear in this. Above about eight, the unlabelled pool becomes days rather than nights |
| `n_detections` | Boxes surviving to the cascade | Zero means prompts are wrong or thresholds too high |
| `n_accepted` | Boxes that became training labels | The useful output |
| `n_background` | Guaranteed false positives kept deliberately | Free, correctly labelled negative data. See below |
| `n_tracks` | Identities after stitching | |
| `n_stitched_tracks` | Identities joined across a chunk boundary | A high fraction is **expected**, and it is the main accuracy risk |
| `labels_never_found` | Labelled classes the detector never produced | These clips go to the human one-box queue |
| `reasons` | Why each detection was kept or dropped | The audit trail. A reason you did not expect is a bug |
| `naming` | Generic detections given a fine-grained label | `n_named` of zero means the crop classifier is not wired in |

**Why background boxes are not waste.** Each clip is prompted with three to
five confusable classes it is *not* labelled with. Every detection of those is
guaranteed wrong, so it becomes an explicitly labelled negative at no cost.
Eighteen free negatives from one clip is the system working as designed.

**Why `below_score` appearing is healthy.** It means the gates are doing
something. A `reasons` block containing only `kept` would mean nothing is being
filtered, which at this label budget would be suspicious rather than good.

## 4. See what it will cost before spending it

```bash
python3 -m vlmlab.cli plan --manifest examples/manifest.json --config examples/config.json
```

```
clips=4 sessions=2 keyframes/clip=45 min_chunks/clip=6
estimated detector time: 0.1 h for 4 clips (0.8 h for a 10x pool)
```

Scaled to 150 clips at five concepts each, that is roughly two hours, and about
twenty for a pool ten times larger. Those numbers move linearly with the
concept count, which is why restricting each clip's prompts to its own labels
is a mandatory stage rather than an optimisation.

## 5. Check the environment before using a GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vlmlab.cli preflight --config examples/config.json
```

```
  [WARN] provenance: unmeasured values feed gates: cascade.max_mask_fragmentation
  [PASS] single_visible_device: one device visible (0)
  [PASS] no_flash_attention: absent and not requested
  [PASS] frozen_fields: 8 frozen fields hold
  [PASS] concept_budget: no plans to check
  5 checks, 0 failed
```

Exits non-zero if any check fails. Try it without pinning a device and watch it
refuse: a launcher that sees two cards silently runs distributed, which doubles
the effective batch and halves the optimiser steps.

## 6. See which numbers nobody has measured

```bash
python3 -m vlmlab.cli provenance
```

Every configuration value is tagged verified, estimated, or unmeasured, with
its source. Four are unmeasured, and the most important is peak detector memory
on a 12GB card: no published figure exists anywhere, and every cost estimate
depends on it. That is why the first real task on a GPU is the benchmark, not
training.

## 7. See what data already exists

```bash
python3 -m vlmlab.cli datasets
```

Prints what is downloadable, what it covers, and what it does not. The useful
summary: nine vocabulary classes have public boxes, six have none, and
classification crops rescue four of those six. Only two classes genuinely need
your own footage.

## Where to go next

- [ARCHITECTURE.md](ARCHITECTURE.md) explains why the pipeline has these stages.
- [RUNBOOK.md](RUNBOOK.md) is the execution guide for the real GPU work, with troubleshooting.
- [MODULES.md](MODULES.md) is the module-by-module reference.
- [IMPLEMENTATION.md](IMPLEMENTATION.md) says what is tested and what is not.
- [00-executive-summary.md](00-executive-summary.md) is the research this all came from.
