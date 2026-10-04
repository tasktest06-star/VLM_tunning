# Fine-tuning a vision-language model from limited video-level labels

Research into recognising objects in video when the only supervision is a small number of
clip-level labels. The concrete target is laboratory equipment.

The question that prompted this was how to fine-tune a vision-language model under that constraint.
**The answer is that you should not.** These documents set out why, and what to do instead.

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

## Read in this order

| Document | Read it for |
|---|---|
| [docs/00-executive-summary.md](docs/00-executive-summary.md) | The answer. Verified facts, recommended pipeline, honest ceiling, decision guide. Stands alone. |
| [docs/05-annotation-budget-verification.md](docs/05-annotation-budget-verification.md) | **Read before spending any of your own hours annotating.** Refutes the headline claim from the earlier studies and resolves how much to annotate. |
| [docs/02-datasets-and-licences.md](docs/02-datasets-and-licences.md) | Reference tables. Public data you can download today, which detector has seen which class, the homonym trap, licence landmines. |
| [docs/01-gap-analysis.md](docs/01-gap-analysis.md) | Twelve defects found in a sibling auto-labelling pipeline, each with a file and line reference. Applies to `customobjectdetection`, not to this repository. |
| [docs/03-tracking-and-open-vocabulary.md](docs/03-tracking-and-open-vocabulary.md) | Long primary study. Trackers, adding a class without retraining, evaluating with no box ground truth. |
| [docs/04-unlabelled-pool-and-annotation.md](docs/04-unlabelled-pool-and-annotation.md) | Long primary study. Semi-supervised detection from zero boxes, pseudo-label constraints. |

Documents 03 and 04 are long and meant to be searched rather than read front to back. Their
annotation guidance is **superseded** by document 05 and they carry pointers saying so.

## The five findings that change what you would otherwise do

**Do not fine-tune the open-vocabulary model.** Reported collapse is from 51.90 to 0.10 average
precision at 0.5 after a single-class fine-tune, against 51.90 held exactly by a frozen detector
with visual prompting. The source experiment may be degenerate and that is still under verification,
but the direction is not in question and it is fatal to a growing vocabulary.

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

Treat the per-class estimates in the executive summary as **under downward revision**. Zero-shot
median average precision across a 35-domain suite is 11.9 to 18.4, with specialised domains as low
as 0.25, and laboratory equipment is fine-grained and largely out of vocabulary.

## Status and how to read the confidence markers

Numbers are tagged as measured in a source, vendor-claimed, or estimated. Treat them as what the
source reports, not as what this project will reproduce. Several cited benchmarks are close to
saturated and were collected under far easier conditions than handheld laboratory footage.

Verification is ongoing. Two questions remain open and are marked in the documents:

- The catastrophic-forgetting magnitude, whose source experiment may be degenerate.
- An unreproduced throughput figure that drives the best-case cost model. Plan with the slower number.

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
