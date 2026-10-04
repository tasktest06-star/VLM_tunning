# Fine-tuning a vision-language model from limited video-level labels

Research into recognising objects in video when the only supervision is a small number of
clip-level labels. The concrete target is laboratory equipment.

The question that prompted this was how to fine-tune a vision-language model under that constraint.
**The answer is: yes, but last, on a copy, with a capped schedule and weight averaging, and only
after you have proved you can measure the difference.** An earlier version of this research said not
to fine-tune at all. Verification overturned that. See [docs/06](docs/06-post-verification-corrections.md).

## The constraints this is scoped to

| Constraint | Value |
|---|---|
| Labelled clips | Under 200, assume about 150 |
| Label granularity | Video level only. No boxes, no masks, no timestamps |
| Labels per clip | Several. Multi-label |
| Class list | Open-ended and growing |
| Required output | Boxes per frame plus tracking over time |
| Unlabelled footage | Large pool available |
| Hardware | Two RTX 3060, 12GB each. PCIe, no NVLink, sm_86, bf16 yes, FP8 no |
| Budget | Low cost is a hard requirement |

Two of these pull against each other, and resolving that tension is most of the answer. An
open-ended class list requires zero-shot generalisation. Fine-tuning on a closed label set is known
to destroy exactly that.

## The code

An implementation of the three-phase plan lives in `vlmlab/`. It runs with
**no dependencies at all**:

```bash
python3 -m unittest discover tests          # 377 tests, standard library only
python3 -m vlmlab.cli run --backend fake \
    --manifest examples/manifest.json --config examples/config.json
```

That is deliberate. The core is standard library only, so the clip-label
cascade, the statistics, the exporters and the whole orchestration are
genuinely tested rather than assumed. The three GPU adapters are written
against the same contract the fake backend passes, and are marked untested in
their own docstrings. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for
what is and is not verified, and for the five silent bugs the test suite
caught.

**No memory figure, throughput number or accuracy number in this repository
came from running a model.** The first task on the GPU machine is
`scripts/benchmark_vram.py`, because peak detector memory on a 12GB card is
unmeasured anywhere and every cost estimate depends on it.

## Read in this order

| Document | Read it for |
|---|---|
| [docs/00-executive-summary.md](docs/00-executive-summary.md) | The answer. Verified facts, recommended pipeline, honest ceiling, decision guide. Stands alone. |
| [docs/05-annotation-budget-verification.md](docs/05-annotation-budget-verification.md) | **Read before spending any of your own hours annotating.** Refutes the headline claim from the earlier studies and resolves how much to annotate. |
| [docs/02-datasets-and-licences.md](docs/02-datasets-and-licences.md) | Reference tables. Public data you can download today, which detector has seen which class, the homonym trap, licence landmines. |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | What the code does, what is tested, and the eleven traps it enforces mechanically. |
| [docs/08-resolved-constraints-and-plan.md](docs/08-resolved-constraints-and-plan.md) | **Start here. Current position.** The three confirmed project constraints, what they unblock, the three-phase plan, and the fine-tuning recipe for this label budget. |
| [docs/06-post-verification-corrections.md](docs/06-post-verification-corrections.md) | **Authoritative. Wins over 00 to 05.** Reverses the no-fine-tuning verdict, fixes the model identifier, and corrects the cost model by two to four times. |
| [docs/07-engineering-traps-and-measurability.md](docs/07-engineering-traps-and-measurability.md) | **Read before writing code.** Eleven verified traps that each silently invalidate a run, plus the finding that measurement is floored by your session count. |
| [docs/01-gap-analysis.md](docs/01-gap-analysis.md) | Twelve defects found in a sibling auto-labelling pipeline, each with a file and line reference. Applies to `customobjectdetection`, not to this repository. |
| [docs/03-tracking-and-open-vocabulary.md](docs/03-tracking-and-open-vocabulary.md) | Long primary study. Trackers, adding a class without retraining, evaluating with no box ground truth. |
| [docs/04-unlabelled-pool-and-annotation.md](docs/04-unlabelled-pool-and-annotation.md) | Long primary study. Semi-supervised detection from zero boxes, pseudo-label constraints. |

Documents 03 and 04 are long and meant to be searched rather than read front to back. Their
annotation guidance is **superseded** by document 05 and they carry pointers saying so.

## The five findings that change what you would otherwise do

**Fine-tuning is safe if done correctly, and the earlier prohibition here was wrong.** The collapse
figure it rested on, 51.90 to 0.10, came from a thermal-infrared fine-tune evaluated on colour
images. That is a modality shift, not a vocabulary effect, and this project is colour throughout. The
real evidence runs the other way: a full fine-tune of an open-vocabulary detector cost 2.9 points
in the wild while **gaining 9.7 points on classes it never saw**, and weight averaging ended up 3.3
points above the frozen model. Fine-tune a copy, without mask loss, on a capped schedule, then sweep
the weight-average coefficient. Measure retention by the **median** across held-out classes, never
the mean, because a 9% mean drop concealed a 61% median collapse.

**You are not starting from zero box labels.** Roughly 22,000 CC BY 4.0 box-annotated
laboratory-apparatus instances are downloadable across four datasets, two of them video-derived. The
gap is specific: no public boxes exist anywhere for centrifuge, autoclave, orbital shaker,
spectrophotometer, or a real fume hood. That is where labelling effort belongs.

**Bare class names retrieve the wrong objects.** In LVIS, `fume_hood` is category 565, a *frequent*
class defined as a covering that exhausts fumes, with synonym `exhaust_hood`. It is a kitchen range
hood. `shaker` means a condiment shaker. This is the worst case, because a detector is confidently
trained on the wrong object under exactly the name you would prompt with.

**The widely repeated one-box-per-clip result is an arithmetic artefact.** It held that one box per
clip moves a detector from 58% to 88% of fully supervised. Verification traced it to a 2016 result
where human verification with zero boxes drawn reaches 58% against 66% for full supervision, and 58
divided by 66 is 87.9%. The two figures are the same result stated twice. Verify-and-correct is
still the best use of annotation time, but for a different reason.

**Semi-supervised object detection does not apply here.** No published method bootstraps from
image-level or video-level labels alone, and a 2026 result shows the state of the art collapsing to
1.00 mean average precision at one shot per class.

**Split localisation from classification.** Fine-grained instrument names fail as detection prompts.
A surgical-instrument study found the instrument name unusable and fell back to a generic prompt plus
a crop classifier. So localise with generic prompts such as "laboratory instrument" or "glassware",
then classify the crops with a frozen image-text model under the clip label as a multiple-instance
constraint. This is what makes centrifuge and spectrophotometer tractable at all.

**Measurement is floored by your recording sessions, not your clips.** With four sessions the honest
interval on the headline metric is about 15 points wide, and labelling more clips in the same rooms
does not narrow it. Most method comparisons this project would want are therefore unresolvable.

## The recommended pipeline, in brief

Frozen foundation models as annotators, the video-level label as a constraint rather than a training
signal, and distillation into a small permissively licensed detector.

1. Warm start on the free public boxes.
2. Two frozen teachers, SAM 3 and MM-Grounding-DINO, whose training mixes genuinely differ.
3. Prompt each clip with **only its own label set**. Raises precision and bounds cost.
4. Add grounded hard negatives, because of the homonym problem.
5. Propagate with SAM 2.1-small, Apache 2.0 for code and weights.
6. Reconcile against the clip's presence, absence and count constraints.
7. Distil into RF-DETR, Apache 2.0.

The step that makes 150 labels leverage thousands of clips: train a clip-level multi-label
classifier on the labelled clips, then use its predictions to supply the per-clip vocabulary
constraint across all unlabelled footage.

## Honest ceiling

Two independent benchmarks put video-level supervision at roughly 58% of fully-supervised
performance at overlap 0.5, and roughly 35% at strict thresholds. Expect weak strict-threshold
localisation throughout. That is inherent to the supervision available, not a flaw in the method.

Revised figures, superseding the executive summary: detection at overlap 0.5 of **30 to 50 on common
classes and 5 to 20 on rare or confusable ones**, averaged over overlaps 12 to 22, tracking 35 to 55
on static equipment. Clip-level multi-label, which is what the supervision actually fits, reaches
0.65 to 0.85 on common classes.

Confusable benchtop siblings sit at 0.40 to 0.70 and **moving them needs 50 to 100 clips per class
from different rooms, cameras and instrument brands, not more method work.** That is a data-diversity
problem, not a modelling one.

Plainly: the realistic outcome is a useful annotation-assist and retrieval tool, not a production
detector.

## Status and how to read the confidence markers

Numbers are tagged as measured in a source, vendor-claimed, or estimated. Treat them as what the
source reports, not as what this project will reproduce. Several cited benchmarks are close to
saturated and were collected under far easier conditions than handheld laboratory footage.

Verification is ongoing. Two questions remain open and are marked in the documents:

- **How many distinct recording sessions, rooms and physical instrument units are behind the 150 clips?** This blocks the measurement plan entirely. See document 07, section 8.
- An unreproduced throughput figure that drove the best-case cost model. Plan without it.
- A claimed fivefold speedup from graph capture, still resting on one unreproduced report.

## Confirmed constraints

| Constraint | Value | Consequence |
|---|---|---|
| Recording sessions behind the clips | 9 to 20 | Measurement viable. Interval 8 to 11 points, so differences above about 10 points are resolvable |
| Use | Research and publication only | Unblocks ImageNet-21k laboratory crops and the Objects365-derived detectors, which were the best resources available |
| Deliverable | Annotation assistant first, measurement second | See the three-phase plan in document 08 |

Research-only use matters more than it sounds. ImageNet-21k's 17 laboratory synsets, about 14,400
images, are the only public source with real coverage of the classes that have no boxes anywhere:
autoclave, microscope, analytical balance, spectrophotometer and centrifuge. They are now usable, and
they are the obvious training set for the crop classifier.

## Branches

- `main` holds the settled deliverable.
- `research/weak-label-vlm-finetuning` carries corrections while remaining verification completes, so `main` stays stable until those corrections are themselves checked.

Per-approach branches should be cut when implementation starts, one per candidate pipeline, rather
than now while there is no code to diverge.

## Method

Nine general technique families and six setup-specific routes were surveyed, each followed by an
adversarial verification pass instructed to refute rather than confirm. Two gap studies covered
tracking and the unlabelled pool, then a six-cluster attack and a dedicated annotation-economics
check were run against those studies. That last pass overturned the single most actionable claim,
which is why the verification trail is kept rather than summarised away.

Dataset category claims were verified by downloading and searching the actual category files.
Licences were read from the repositories rather than from summaries. SAM 3's existence, model
identifiers, gating and licence terms were confirmed directly against arXiv and Hugging Face.
