# Fine-tuning a vision-language model from limited video-level labels

Executive summary of research conducted 2026-10-04 for laboratory-equipment recognition in video.

## The problem as scoped

| Constraint | Value | Consequence |
|---|---|---|
| Labelled clips | Under 200, assume about 150 | Deep few-shot territory |
| Label granularity | Video level only | No boxes, no masks, no timestamps |
| Labels per clip | Several, multi-label | Per-class sigmoid outputs and threshold calibration |
| Class list | Open-ended and growing | Adding a class must not require retraining |
| Required output | Boxes per frame plus tracking over time | Spatio-temporal, from clip-level labels only |
| Unlabelled footage | Large pool available | Semi-supervised use is central |
| Hardware | Two RTX 3060, 12GB each | 12GB per card binds; PCIe, no NVLink, sm_86, bf16 yes, FP8 no |

Two of these constraints pull against each other, and resolving that tension is most of the answer.
An open-ended class list requires zero-shot generalisation. Fine-tuning on a closed label set is
known to destroy exactly that. So the plan cannot be "fine-tune a model on 150 clips", however the
request was originally phrased.

## The headline answer

**Do not fine-tune a large video-language model, and do not build a semi-supervised detector.** Both
were examined in depth and both lose at this label budget. Use frozen foundation models as
annotators, let the video-level labels act as a constraint rather than as a training signal, and
distil the result into a small deployable detector.

### Why fine-tuning a video-language model loses

The reported evidence on catastrophic forgetting is severe. A YOLO-World model fine-tuned on a
single class fell from 51.90 to 0.10 average precision at 0.5 on COCO. Head-only fine-tuning
reached only 0.66. A frozen detector with visual prompting held at 51.90 exactly.

That is an extreme configuration and it is under adversarial verification, including whether it
generalises to a 10-to-30-class fine-tune or is an artefact of a degenerate single-class setup. The
direction, however, is not in doubt, and it is fatal to a growing vocabulary.

Separately, nothing of the sort fits the hardware comfortably. Full fine-tuning of the relevant
models needs roughly 14.4GB before activations, which exceeds 12GB.

### Why semi-supervised object detection loses

No published method bootstraps from image-level or video-level labels alone. The closest,
Omni-DETR, handles mixed weak annotation types including image-level tags, reporting 34.7 against
28.0 mean average precision, but still requires a box-labelled burn-in phase.

More decisively, a January 2026 result shows Semi-DETR collapsing to 1.00 mean average precision on
PASCAL VOC and 0.80 on a domain set at one shot per class, despite being state of the art at 1% of
COCO. These methods are tuned for a regime an order of magnitude above this one.

## Verified foundation

Checked directly against primary sources rather than taken from a report.

SAM 3 exists and it produces the exact output this project needs. The paper is "SAM 3: Segment
Anything with Concepts", arXiv 2511.16719, submitted 20 November 2025 and revised 28 March 2026,
led by Christoph Feichtenhofer. It introduces promptable concept segmentation, takes a noun phrase
or an image exemplar, and returns masks plus unique identities for every matching instance. That is
boxes plus tracking from a text prompt, with no retraining, which is precisely what an open-ended
class list requires.

| Item | Verified value |
|---|---|
| Base weights | `facebook/sam3`, 0.9B parameters, published 20 Nov 2025 |
| Newer variant | `facebook/sam3.1`, updated 27 Mar 2026 |
| What 3.1 adds | Object Multiplex, joint multi-object tracking, roughly 7x faster at 128 objects on one H100, better video object segmentation on 6 of 7 benchmarks |
| Access | Gated on both, requires sharing contact details with Meta |
| Licence | SAM License. Commercial use permitted, royalty-free |
| Duties | Acknowledge in publications, redistribute under the same terms |
| Limits | No military, nuclear, weapons or espionage use. No monthly-active-user threshold |

The licence does not block this project. The gating does matter operationally, because access must
be requested by hand and will break unattended automation.

## Recommended pipeline

Each stage uses the video-level label for what it is genuinely good at, which is constraining and
filtering rather than training.

1. **Warm start on free public boxes.** Roughly 22,000 CC BY 4.0 laboratory-apparatus instances are downloadable across four datasets, two of them video-derived. This covers the glassware and apparatus half of the vocabulary at no annotation cost.
2. **Dual frozen foundation teacher.** Run SAM 3 and MM-Grounding-DINO, both frozen. Their training mixes differ, so agreement between them is real evidence rather than a shared blind spot.
3. **Prompt each clip with only its own label set.** The key move. It raises precision, and because SAM 3 costs roughly 300 milliseconds per concept per frame, it is a cost and memory necessity rather than an accuracy nicety.
4. **Add grounded hard negatives.** Mandatory, because of the homonym problem described below.
5. **Propagate with SAM 2.1-small.** Apache 2.0 for code and weights, and it interpolates between detections.
6. **Reconcile against presence, absence and counts.** A clip labelled with two instruments guarantees both are present and is strong evidence every other class is absent. This is where the weak label does real work.
7. **Distil into a small deployable model.** RF-DETR, Apache 2.0, paired with a permissively licensed tracker.

The step that makes 150 labels leverage thousands of clips: **train a clip-level multi-label
classifier on the labelled clips, then use its predictions to supply the per-clip vocabulary
constraint across all unlabelled footage.** The expensive supervision is reused as a cheap filter at
scale.

## Findings that change what you would otherwise do

**You are not starting from zero box labels.** See section 1 of the dataset reference. The gap is
specific: no public boxes exist anywhere for centrifuge, autoclave, orbital shaker,
spectrophotometer, or a real fume hood. That is where labelling effort belongs.

**Bare class names retrieve the wrong objects.** In LVIS, `fume_hood` resolves to WordNet
`hood.n.06`, an exhaust hood, meaning a kitchen range hood. The class `microscope` has 3 instances
across 2 images. Prompting with equipment names alone returns domestic objects confidently.

**Most of your vocabulary has never been seen as a box by any detector.** Centrifuge, pipette,
autoclave, spectrophotometer and fume hood fall in that category. Only MM-Grounding-DINO's `v3det`
or `pretrain_all` checkpoints have seen microscope, test tube, funnel, thermometer or stirrer.

**Tracking propagation is weaker than assumed.** The one clean ablation available found
tracker-mined boxes contributed 0.95 of an 11.90 average-precision gain, and in a second experiment
tracker-only pseudo-labels were net negative, moving 15.66 down to 11.73. Propagation belongs in
the pipeline for interpolation, not as a primary label source.

**Active learning is not worth building, but not for the reason first reported.** Verification found
the negative figures were read off the *largest* budget column of their tables, and one was
sign-inverted: the method reported at minus 2.4 is actually plus 7.9 at a small budget. One source
paper explicitly concludes it finds no support for the claim that uncertainty sampling fails at low
budgets. The conclusion survives on a better argument. At roughly 150 clips you sit below the
smallest budget at which any active-learning method for detection has been shown to beat random
selection, and the one strong detection positive used a ten-to-one selection ratio over a two
million image pool. If you intend to annotate nearly your whole pool, active learning cannot help by
construction. Select clips randomly. See document 05.

**Synthetic rendering is the most under-exploited option.** Rendering 47 Objaverse microscope meshes
across a thousand poses yields roughly 47,000 synthetic microscope boxes, against 21 real ones
across LVIS, Visual Genome and V3Det combined.

**Train instance-level, not only category-level.** If the clips come from one or two labs, you are
recognising specific instruments rather than abstract categories, which is materially easier.
EgoObjects, MIT licensed, is the closest precedent.

## Honest ceiling

Two independent benchmarks put video-level or image-level supervision at roughly 58% of
fully-supervised performance at intersection-over-union 0.5, and roughly 35% at strict thresholds.

The right proxy for a novel narrow domain is RF100-VL, where SAM 3 scores 15.2 average precision
zero-shot rising to 36.5 at ten shots per class, rather than its headline LVIS figure.

| Outcome | Predicted average precision at 0.5 |
|---|---|
| Common, visually distinct classes | 40 to 55 |
| Visually confusable classes | 10 to 25 |
| Small objects such as pipettes | 5 to 15 |

Roughly three to four weeks of wall-clock on two 3060 cards. Strict-threshold localisation will be
weak throughout. That is inherent to the available supervision, not a flaw in the method.

**These figures are under downward revision.** Later verification argues the right reference is not
COCO but the Object Detection in the Wild suite, where zero-shot median average precision is 11.9 to
18.4 and specialised domains fall to 0.25. See the final open question below.

## Decision guide

- **Need presence tags per clip only.** Frozen backbone plus a small trained head. Cheapest by far, and at 150 clips it is competitive with anything more elaborate.
- **Need boxes, closed and stable class list.** Warm start on public boxes, pseudo-label, distil into RF-DETR.
- **Need boxes, open-ended class list.** The recommended pipeline above. Do not fine-tune the open-vocabulary model itself.
- **Need boxes plus tracking.** As above, with SAM 3 supplying identities natively and SAM 2.1 interpolating.
- **Tempted to fine-tune a video-language model.** Measure a frozen baseline first. The research found no configuration at this label budget where fine-tuning wins, and several where it destroys the generalisation the class list depends on.

## Open questions

Stated rather than hidden. Two that appeared here originally have now been resolved by a dedicated
verification pass, recorded in document 05.

**Resolved: the annotation budget is 450 frames.** Three frames per clip across all 150 clips, at
least 2 seconds apart, scored by 5-fold cross-validation over clips rather than a fixed holdout.
Documents 03 and 04 disagreed two-fold and were both arguing about a saturating axis. Doubling from
300 to 600 frames buys 0.66 mean average precision points of confidence interval, because the floor
is set by clip count rather than frame count. Three further consequences: spreading frames across
many clips beats densely boxing a few, which is the worst use of evaluation budget; cross-validation
is a free 1.4-fold power gain; and per-class precision is unfalsifiable below about 8 clips per
class, so allocate frames equally across classes rather than in proportion to frequency.

**Resolved, and refuted: the one-box-per-clip claim was an arithmetic artefact.** It held that one
box per clip moves a detector from 58% to 88% of fully supervised. Verification traced it to a 2016
result where human verification with zero boxes drawn reaches 58% against 66% for full supervision,
and 58 divided by 66 is 87.9%. The two figures are the same result stated twice, not a before and an
after. The unit was also wrong: every underlying result is per image, never per clip.

What survives is still the top recommendation, on different grounds. That 2016 result does transfer
in kind, because its input supervision is image-level labels with no boxes, which is nearly this
project's situation. Reaching 88% of fully supervised by verifying machine proposals, with no boxes
drawn, is real and should be a floor with modern proposals. So **verify-and-correct is the best use
of annotation time**, and one box per clip stays in the plan as a propagation prompt rather than as a
route to 88%.

**Still open: the forgetting magnitude needs its setting checked.** The collapse from 51.90 to 0.10
came from a single-class fine-tune, which may be degenerate. Whether it generalises to a realistic
multi-class fine-tune is still under verification.

**Still open: one throughput figure is unreproduced.** A claimed fivefold speedup from CUDA graphs at
512 pixels drives the best-case cost model and traces to a single unresolved issue thread. Plan with
the slower number.

**Newly open: the performance prior may be too optimistic.** Verification flags that COCO figures are
the wrong reference. On the 35-dataset Object Detection in the Wild suite, zero-shot Grounding DINO
has a median average precision of 11.9 to 18.4, with specialised domains as low as 0.25. Those are
averaged over overlap thresholds so they are not directly comparable to the at-0.5 table above, and
the pipeline adds a warm start and distillation above zero-shot, but the correction points downward.

**Three numbers to stop quoting**, all traced to misreadings: 35 seconds per box, a fivefold extreme
clicking speedup, and a tenfold to twentyfold gain from verification. Real verification gains measured
in deployment were about 1.5-fold.
