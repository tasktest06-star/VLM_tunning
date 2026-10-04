# Post-verification corrections

> **This document is authoritative. Where it conflicts with documents 00 to 05, it wins.**
>
> Six adversarial verifiers attacked the load-bearing claims in the earlier research against primary
> sources, then a reconciler merged the verdicts. The pipeline shape survived. Three load-bearing
> pieces did not, and one of them was this project's headline recommendation.

## 1. The headline reversal: you *should* fine-tune

Documents 00 and the repository README said do not fine-tune the open-vocabulary detector, on the
strength of a reported collapse from 51.90 to 0.10 average precision at 0.5 after a single-class
fine-tune. **That prohibition was wrong and is withdrawn.**

**The collapse figure was a modality shift, not a vocabulary effect.** It came from a YOLO-World-S
fine-tune on 12,025 thermal **infrared** images, evaluated on RGB COCO. The model did not forget its
vocabulary. It was asked to work in a different imaging modality. This project is RGB throughout, so
the figure does not apply.

**The actual evidence points the other way.** OWLv2, section 4.6, fine-tunes an open-vocabulary
detector on LVIS base classes for about 100 epochs:

| Measure | Before | After full fine-tune | With weight averaging at 0.4 |
|---|---|---|---|
| Object Detection in the Wild, 13 datasets | 53.0 | 50.1 | **56.3** |
| LVIS rare, never annotated during fine-tuning | 34.9 | **44.6** | — |

So a full fine-tune cost 2.9 points of in-the-wild performance while **gaining 9.7 points on classes
it never saw**. The paper observes verbatim that performance on rare classes "behaves similarly to
LVIS frequent and improves during fine-tuning, even though no LVIS rare classes are seen". And
weight-space ensembling at a fine-tuned weight of 0.4 ends up **3.3 points above the frozen model**.

SAM 3's own few-shot results confirm it. The jump from 15.2 to 36.5 on RF100-VL, and 61.0 to 71.8 on
Object Detection in the Wild, is plain low-learning-rate full fine-tuning. The paper states "We
fine-tune SAM 3 without mask loss".

### What to do instead of the prohibition

1. Pseudo-label the unlabelled pool with the frozen teacher first. This also moves you out of the worst few-shot retention regime before any weights change.
2. Fine-tune a **copy**, with **no mask loss** and an explicitly **capped** step budget. With 150 clips the instinct to train to convergence is exactly wrong.
3. Sweep the weight-average coefficient over 0, 0.2, 0.4, 0.6, 0.8 and 1.0, and pick your point on the frontier. **Weight-space ensembling is now mitigation number one.** It is free, one weight average per coefficient.
4. Never replace the region-text scoring head with a closed-set softmax. That is the thing that genuinely destroys extensibility.
5. Do **not** co-train a general-vocabulary replay set. OWLv2 tried it and found it worse than weight ensembling, and it costs compute you do not have.

### Measure retention with the median, never the mean

This is the trap inside the good news. After full fine-tuning, OWLv2's **mean** across 35 in-the-wild
datasets falls only 24.4 to 22.3, a 9% drop. The **median** across the same 35 falls 16.2 to 6.3, a
**61% collapse**. Weight ensembling recovers the median only to 10.6.

A mean-based retention metric hides a 61% median collapse. So hold out 5 to 8 laboratory classes
entirely from fine-tuning and report the **per-class median**, never the mean.

## 2. The model identifier is wrong

Use **`facebook/sam3`**, not `facebook/sam3.1`.

The 3.1 repository has `library_name: checkpoint` and ships only `sam3.1_multiplex.pt`, with no
`model.safetensors`. Its model card states verbatim: "This repository hosts only the SAM 3.1 model
checkpoints — there is no Hugging Face Transformers integration."

Every interface the plan depends on exists only for `facebook/sam3`: `get_text_features` at
`modeling_sam3.py:2230`, `get_vision_features` at `:2269`, `add_text_prompt(self, inference_session,
text: str | list[str])` at `processing_sam3_video.py:101`, and `prompt_to_obj_ids` at `:273`.

So the choice is explicit. Take `facebook/sam3` with Transformers and get the multi-prompt list and
the cached-embedding path, but not Object Multiplex and not 3.1's accuracy. Or take `facebook/sam3.1`
with Meta's own package and get Object Multiplex, but lose the multi-prompt list and the documented
embedding reuse. **Pin `facebook/sam3`.** Do not use the community safetensors conversions of 3.1,
which are not published by Meta.

## 3. The cost model was two to four times optimistic

The earlier claim was that caching vision embeddings makes cost independent of vocabulary size. It
does not. The paper cited for it says the saving applies to **image encoding only**. The per-concept
decoder pass is the larger term and does not amortise.

Measured on an RTX 5070: `set_image` amortises from 0.423 seconds to about 0.030, while
`process_prompt` stays at 0.285 to 0.339 seconds **per concept**. End-to-end that is 2.5 frames per
second at one class against roughly 0.7 at forty.

**Model cost as `set_image` plus the number of classes queried times the per-concept cost.** Add a
term for tracked objects, since the paper states cost scales linearly in them and a cluttered bench
holds 15 to 30.

| Stage | Earlier estimate | Corrected |
|---|---|---|
| 150 labelled clips | 40 to 90 minutes | **2 to 4 hours** |
| Ten times that unlabelled | 7 to 15 hours, one overnight | **25 to 40 hours, two to three nights** |
| Human annotation | 5 to 6 hours | **12 to 16 hours** |
| Full rounds on two 3060s | 3 to 4 weeks | 3 to 4 weeks, about 230 kWh |

**This strengthens rather than weakens the recommended design.** Restricting each clip's prompt set
to its own labels, which the clip-level multi-labels give you free, is now a **mandatory stage**
rather than an optimisation. And the keyframe-plus-small-propagator split is more necessary than was
argued, because full video inference is far more expensive than assumed. The design was right for a
reason the research got wrong.

Never persist vision embeddings. At keyframe stride 8 that would be 1.79 terabytes.

## 4. Take boxes from the image path, not the video path

Boxes are your deliverable, and in the Transformers **video** path they are derived from masks via
`masks_to_boxes` at `processing_sam3_video.py:339`. Box quality is therefore capped by mask quality,
and a fragmented or leaking mask silently produces an inflated box.

The **image** path exposes a real detection head: `pred_boxes` at `modeling_sam3.py:195`, computed at
`:2454`, alongside `presence_logits` at `:197` with the documented scoring rule
`final_scores = pred_logits.sigmoid() * presence_logits.sigmoid()` at `:173`.

So take keyframe boxes from the image path and use `presence_logits` for the reject logic. Use the
video path only for inter-keyframe boxes and identities, and audit mask fragmentation there.

Also note that promptable concept segmentation is formally defined only for "an image or short video
(<=30 secs)". Longer clips need chunking with overlap-based identity matching across chunks, which
costs accuracy.

## 5. Two interface errors in the earlier research

**`propagation_direction` does not exist.** The real signature is `propagate_in_video(inference_state,
start_frame_idx=None, max_frame_num_to_track=None, reverse=False, force_tracker_propagation=False)`.
Call it twice, with `reverse=False` then `reverse=True`. A grep across all four relevant source files
returned zero hits for the keyword the earlier document claimed, despite tagging it as verified
against the interface. Treat every similar tag in documents 03 and 04 as unchecked.

**SAM 3 has no cross-image exemplar path.** Meta's signature is `add_prompt(self, inference_state,
frame_idx, text_str=None, boxes_xywh=None, box_labels=None)`, so the arguments are `boxes_xywh` and
`box_labels`, in centre-width-height form rather than corner form. Worse, `add_prompt` calls
`reset_state` with the comment "since it's a semantic prompt, we start over", and `text_str` takes a
single string. That means one concept per session, multiplying per-clip cost by vocabulary size.

**Use OWLv2 for "add a class from photos".** Its `image_guided_detection(..., query_pixel_values=...)`
at `modeling_owlv2.py:1233` is a genuine cross-image interface, Apache 2.0 and ungated. The DINOv3
prototype bank the earlier research recommended is itself gated with a bespoke licence, so it
de-risks nothing.

## 6. Hard-negative prompting loses its evidence

The earlier research ranked inference-time hard-negative prompting second, at high confidence, worth
0.44 to 0.68 on the image-level presence metric. Those numbers are exact but they come from a
**training-data composition** ablation, comparing 0 against 30 hard-negative noun phrases per image
during training. The released checkpoint already has that benefit baked in. No inference-time
ablation exists anywhere in the paper.

Demote it to an untested hypothesis with no effect size. It remains a cheap and sensible engineering
idea given the homonym problem, so implement it and measure it, but do not credit it with a gain in
any budget.

## 7. The benchmark proxy was wrong

RF100-VL does **not** contain laboratory equipment. The phrase "laboratory imaging" appears only in a
GitHub README as a label for the cluster the paper calls Medical, whose 13 datasets are X-ray, MRI,
dental, mammography, urine sediment and stomata micrographs. There are no photographs of benchtop
instruments anywhere in it, and Medical is its hardest cluster at about 2.1 average precision
zero-shot.

The closest relevant cluster is **Industrial**, covering circuit-board defects, screws, surface-mount
components, water meters and tubes, at **10.3 zero-shot rising to 37.5 at ten shots per class**. Use
that and the Other cluster at 18.1 and 32.6 as the bracket, not the single 15.2 figure.

**A free dry run.** RF100-VL is Apache 2.0, so run the entire few-shot pipeline against its Industrial
cluster before spending any annotation budget on your own footage.

## 8. Licence corrections, several in the opposite direction

The earlier framing of this as restricted SAM against clean Apache was backwards.

**Objects365 is academic-use only.** Its download page states verbatim: "The Objects365 dataset is
available for the academic purpose only." Annotations are CC BY 4.0, images sit under Flickr terms
with an explicit no-redistribution clause. **This transitively taints the weights of every detector
pretrained on it: Grounding DINO, every MM-Grounding-DINO checkpoint, and GLIP.** The mmdetection
code is genuinely Apache 2.0, but Apache-licensed code wrapping academic-only-derived weights is not
a commercial fallback. So document 02's listing of MM-Grounding-DINO as safe is **withdrawn**.

For a commercial product, SAM 3 is arguably the cleaner side, because its licence is at least an
explicit royalty-free grant from the rights holder. The genuinely clean open-vocabulary option is
**OWLv2**, Apache 2.0, ungated, derived from WebLI and CLIP.

**Further blockers the earlier research missed.**

| Dependency | Problem | Substitute |
|---|---|---|
| WeakSAM, called "the method to copy" | **No licence at all.** Default is all rights reserved | Reimplement from the paper, or skip it: its proposal generator needs a retrained classifier per class, which conflicts with the open-vocabulary requirement |
| TrCAM and TCAM | GPL-3.0. The only two open implementations of the video-level localisation route | Drop that route. Use an open-vocabulary teacher feeding a cheap student instead |
| HeinSight4.0 **weights** | Derived from Ultralytics, so AGPL-3.0 regardless of the CC BY 4.0 deposit. A depositor cannot extinguish an upstream claim | Use its **data**, which is genuinely CC BY 4.0. Never its weights |
| ImageNet-21k laboratory synsets | Non-commercial research only | V3Det and Open Images for vocabulary, the four CC BY 4.0 sets for real boxes |
| ModPrompt, ranked mitigation two | No licence, source largely stubs, paper unrefereed | Weight-space ensembling of your own checkpoints, which is free and actually measured |

**Two obligations nobody stated.** The SAM licence includes a pass-through requirement, so any
derivative must ship the agreement, and a clause under which Meta may modify the terms with immediate
effect. And the four CC BY 4.0 laboratory datasets attach attribution to derivative **models**, not
only to redistributed data, so warm-starting on them puts an attribution notice on the release
checklist.

Drop "Acceptable Use Policy" from any SAM triage table. No such policy is incorporated by reference.

## 9. Evaluation tooling is broken upstream

Upstream TrackEval does not run on any modern stack. Its `datasets/tao.py` uses the removed aliases
`np.bool` at lines 305 and 310 and `np.int` at lines 331, 355 and 362, all five confirmed by grep.
NumPy removed them in version 1.24, released December 2022, and upstream has been unmaintained since
November 2022.

Use **`trackeval==1.3.0`**, the maintained MIT fork, which keeps the identical federated evaluation
block with the deprecated aliases fixed. Budget about an hour for the exporter and dependency
pinning, not zero. **Set maximum detections to unlimited**, because the default of 300 per image is
exactly the truncation that moves rare-class precision by 7 points.

SAM 3's own vendored evaluation toolkit is not a substitute. It is hard-coded class-agnostic, so its
federated negative-category branch is dead code, and it carries the SAM licence rather than MIT.

## 10. Annotation budget, revised upward

The reconciler lands on **600 frames**, arriving there by taking the smaller study's independent
sample figure of 245 and applying the realised clustering design effect of about 2.5, giving roughly
613. Document 05's independent analysis landed on 450.

These are one step apart on a saturating curve rather than a genuine conflict. Document 05's own table
puts 450 frames at a detectable difference of 2.08 and 600 at 1.90, so the extra 150 frames buy 0.18
points.

**Annotate 3 to 4 frames per clip across all 150 clips, so 450 to 600 frames, and treat 600 as the
target if the time exists.** Everything else in document 05 stands: spread across clips rather than
depth within them, cross-validation over clips rather than a fixed holdout, and equal allocation
across classes rather than proportional to frequency.

Two harder numbers follow. **The minimum detectable difference is 3.5 to 4 points paired at 600
frames, and 4.5 to 5 after correcting for about ten comparisons. So the whole project can afford
roughly five gold-arbitrated pipeline variants in total.** Choose them deliberately. And document 04's
2.0-point halt trigger sits below the noise floor, so it would stop healthy runs. Use 3.5 points or
three standard deviations of your measured round-zero variance.

Per-class average precision is not reportable at any affordable budget, with a half-width of about
0.14 at 120 instances per class. Commit up front to per-class recall at a frozen threshold with exact
binomial intervals, plus pooled paired mean average precision.

## 11. Honest ceiling, revised down

**Plainly: the realistic outcome is a useful annotation-assist and retrieval tool, not a production
detector.**

| Task | Realistic outcome |
|---|---|
| Clip-level multi-label, which is what your supervision actually fits | Mean average precision 0.65 to 0.85 common, 0.35 to 0.60 rare |
| Detection at overlap 0.5, the actual deliverable | **30 to 50 common, 5 to 20 rare or confusable** |
| Detection averaged over overlaps | 12 to 22, with 5 to 12 at the strict threshold |
| Tracking | 35 to 55 on static equipment, 20 to 35 once handled or occluded |

Document 00's estimate of 40 to 55 is the optimistic end. The reasoning for revising down: the
video-level ceiling is 50 to 58% of fully supervised at overlap 0.5, and you should assume the lower
half, because both anchor benchmarks average about one instance of interest per video while a
laboratory bench holds about three instances and two to three classes per frame. That makes the
per-instance supervision ratio materially worse.

Rare classes are effectively **unmeasurable** at about 15 clips per class, where the interval is
roughly plus or minus 0.25. And tracking metrics as designed carry a 5 to 10 point interval, so they
can arbitrate only large tracker changes. They cannot choose between two good trackers. Decide that
on qualitative failure inspection instead of pretending the metric decided.

## 12. The tension that does not dissolve

**The growing-vocabulary requirement and the accuracy requirement are in direct conflict, and no
architecture here resolves it.**

SAM 3's own limitations section says it "struggles to generalize to fine-grained out-of-domain
concepts... in a zero-shot manner". On one open-vocabulary split it scored 76.03 on base classes
against **1.67 on novel ones**.

A brand-new class added by text alone will land near that floor. Adding 5 to 20 exemplar photos lifts
it substantially, but that is a human in the loop, not free extensibility. And the deployed student is
closed-set, so in production "add a class" means serve it from the frozen teacher now and distil it
into the student at the next retrain.

State that plainly rather than letting "little or no retraining" stand unqualified.

## 13. What remains unresolved

One throughput figure, a claimed 32 frames per second on a datacentre card, could not be traced to any
primary source, and it underpinned the earlier throughput budgeting. Plan without it.

The claimed fivefold speedup from graph capture at 512 pixels still rests on a single unreproduced
report. It is experiment one for a reason.

Two workflows surveying general technique families were still completing when this was written.
Corrections from them will follow.
