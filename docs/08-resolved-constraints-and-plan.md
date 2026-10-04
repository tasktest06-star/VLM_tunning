# Resolved constraints, and the plan that follows

> Final consolidation. Three project constraints were confirmed after the research completed, and
> they relax several conclusions in documents 02, 06 and 07. This document and document 07 together
> are the current position.

## 1. The three answers that changed the guidance

| Question | Answer | What it changes |
|---|---|---|
| Distinct recording sessions, rooms or instrument units | **9 to 20** | Measurement is viable. Interval 8 to 11 points, so differences above about 10 points are resolvable |
| Commercial or research use | **Research and publication only** | Most licence blockers in documents 02 and 06 no longer bind |
| Deliverable | **Both, in sequence.** Assistant first, then measurement | Sequences the work and removes the need to choose |

### Measurement is better placed than feared

Document 07 warned that the confidence interval on the headline metric is floored by session count,
and at four sessions it is about 15 points wide. **At 9 to 20 sessions it is 8 to 11 points.**

So method ranking is possible, for differences above roughly 10 points. That is still a coarse
instrument. It means you can tell a frozen baseline from a well-tuned pipeline, and you cannot tell
two good trackers apart. Plan about five gold-arbitrated comparisons across the whole project and
choose them deliberately.

Grouped splitting by session remains mandatory. With 9 to 20 sessions you can afford a
session-disjoint held-out set **and** grouped cross-validation, which the four-session case could not.
Publish the grouped against ungrouped gap as your leakage estimate.

### Research-only use unblocks the best resources

This reverses most of the licence analysis. The following become usable, with attribution:

- **ImageNet-21k's 17 laboratory synsets**, roughly 14,400 images. This is the largest laboratory-equipment image resource in existence and the only public source with meaningful coverage of the classes that have no boxes anywhere: autoclave 1,277 images, microscope 1,218, analytical balance 1,047, spectrophotometer 628, centrifuge 514. It was previously blocked as non-commercial. **Use it.** Crops from it are the obvious training set for the fine-grained crop classifier in section 3.
- **Objects365-derived detectors**, meaning Grounding DINO, every MM-Grounding-DINO checkpoint, and GLIP. The `v3det` and `pretrain_all` checkpoints are the best-grounded available for this vocabulary, being the only ones to have seen microscope, test tube, funnel, thermometer and stirrer as human-drawn boxes.
- **Ultralytics YOLO**, for baselines and for reproducing the published laboratory-dataset numbers.
- **The GPL-licensed video localisation implementations**, if that route is wanted.

Still genuinely blocked, for different reasons: **LabPics** carries three mutually inconsistent
licence statements, so the grant is ambiguous rather than restrictive, and **WeakSAM** has no licence
at all. Avoid both regardless of use.

Obligations that survive research use: the four CC BY 4.0 laboratory datasets attach attribution to
derivative **models**, so name them in any release or paper. SAM 3's licence carries a publication
acknowledgement duty and a pass-through requirement.

**The practical consequence is that OWLv2 is no longer forced.** Run MM-Grounding-DINO alongside
SAM 3 as originally recommended, since their training mixes genuinely differ and that is what makes
cross-model agreement meaningful.

## 2. The sequenced plan

### Phase one, the annotation assistant

Goal is a tool that proposes boxes and tracks well enough to make human labelling fast. Acceptance is
qualitative, by inspection, because at this stage you are not claiming a number.

1. **Request access to `facebook/sam3` today.** It is gated with manual human approval and will block everything else. Build the ungated path in parallel.
2. **Fix the split before anything else.** Group by recording session, not merely by clip. In the sibling pipeline this is `src/annotation/coco_builder.py` lines 97 to 99.
3. **Warm start** on the roughly 22,000 CC BY 4.0 laboratory boxes, re-split by capture session, since both figshare sets ship random frame splits whose own validation numbers are inflated.
4. **Localise class-agnostically.** Generic prompts only: "laboratory instrument", "machine on a laboratory bench", "glassware", "handheld lab tool". Do not prompt instrument names.
5. **Classify the crops finely**, with a frozen image-text backbone plus a small head, trained on ImageNet-21k laboratory crops and your own clips under the clip label as a multiple-instance constraint. This is the step that makes centrifuge and spectrophotometer tractable, and it is now well supplied with data.
6. **Restrict each clip's prompt set to its own labels**, and add 3 to 5 confusable absent classes as hard negatives whose every detection is a guaranteed false positive and therefore a free background box.
7. **Propagate** with SAM 2.1-small for inter-keyframe boxes and identities. Take delivered boxes from the image path, which has a real detection head, not the video path, which derives them from masks.
8. **Hand-draw one box per clip** where text prompts collapse. Roughly 15 to 30 seconds per clip for a single object and 40 to 70 for several, so 1 to 3 hours for 150 clips. This is the insurance policy and the highest-yield human spend.

### Phase two, measurement

Only once phase one has produced cheap labels across more sessions.

1. **Annotate 3 to 4 frames per clip across all 150 clips**, so 450 to 600 frames, at least 2 seconds apart.
2. **Score by grouped cross-validation over sessions.** Use a group permutation test rather than a bootstrap if the session count lands near the low end.
3. **Pre-register the minimum detectable difference** at 10 points and refuse to report below it.
4. **One global threshold** on pooled out-of-fold scores. Never fit per-class thresholds on 2 to 6 positives.
5. **Evaluation tooling:** `trackeval==1.3.0`, with the 300-detection cap disabled.

### Phase three, fine-tuning

Now justified, since document 06 withdrew the prohibition. Judge it against the frozen baseline on the
**same grouped folds**. A win smaller than 10 points is not a win.

## 3. The fine-tuning recipe for this label budget

From the final research sweep. The headline result is that low-rank adaptation is not a compromise
here: across six low-resource video tasks at 1.2k to 9.2k training clips it **beat full fine-tuning on
every classification task**, at 12.1M against 87.1M trainable parameters.

| Task | Low-rank adaptation | Full fine-tune |
|---|---|---|
| Emotion recognition | **62.1** | 59.1 |
| Industrial step recognition | **68.7** | 66.7 |
| Animal behaviour | **72.6** | 71.8 |

### At 150 clips on a 12GB card

**Prefer a 2 to 4 billion parameter backbone over a crippled 7B one.** Then:

- Rank 4 to 8, **attention projections only**, not the feed-forward layers. Rank scales with labels: 300 to 1,000 clips moves you to rank 8 to 16 with feed-forward included, and 1,000 to 5,000 to rank 16 to 32.
- **Set alpha to twice the rank.** The default scaling in the common implementation penalises higher ranks, and one study calls the correction essential.
- **Freeze the vision tower and the connector.** Tuning the visual backbone significantly degrades performance, at p = 0.004.
- Learning rate 2e-4 at these ranks. Low-rank adaptation needs 10 to 20 times the full fine-tuning rate.
- 8-bit AdamW, cosine schedule, 3% warmup, no weight decay, gradient clipping at 1.0.
- **One to two epochs only.** Overfitting sets in after epoch two below 1,000 clips.
- 16 frames, with 4-bit NF4 quantisation of the base weights.
- Compute loss on the answer tokens only.

### The memory lever nobody mentions

**A fused linear cross-entropy kernel is the single largest memory saving**, and it is the difference
between fitting in 12GB and not. At 8,192 tokens against a 152,000-token vocabulary the logits tensor
is 2.5 GB in bf16, and roughly 7 to 8 GB once the float32 upcast inside the loss and its gradient are
counted. That is frequently larger than the entire adapted model's activation footprint, and almost
all of it is logits for visual positions that never enter the loss.

Enable it, then **assert that it applied**. Document 07 records that one toolkit's own flag for this
is a verified no-op on the relevant model type and fails silently.

Also set the attention implementation explicitly. Document 07 records that the default materialises
roughly 1.26 GB per layer at these sequence lengths, which alone exceeds the card.

### The caveat on reported gains

Low-rank solutions contain dimensions absent from full fine-tuning, and they **degrade more outside
the fine-tuning distribution even at matched in-distribution accuracy**. So your held-out-domain
numbers may flatter the model. Combined with document 06's finding that retention must be measured by
the median across held-out classes rather than the mean, this means: hold out 5 to 8 laboratory
classes entirely, report the per-class median, and sweep the weight-average coefficient rather than
shipping the fine-tuned weights directly.

## 4. Revised expectations

With 9 to 20 sessions and research-only licensing, the ceiling in documents 06 and 07 improves
modestly, because ImageNet-21k crops directly address the worst-supplied classes.

| Outcome | Expectation |
|---|---|
| Clip-level multi-label, what the supervision fits | 0.65 to 0.85 common, 0.35 to 0.60 rare |
| Detection at overlap 0.5 | 30 to 50 common, 5 to 20 confusable |
| Confusable benchtop siblings | 0.40 to 0.70, improving with ImageNet-21k crops |
| Tracking | 35 to 55 static, 20 to 35 once handled |
| Measurement interval | **8 to 11 points** |

The remaining error on confusable siblings is a **data-diversity problem, not a modelling one**.
Closing it needs 50 to 100 clips per class across different rooms, cameras and instrument brands. At
9 to 20 sessions you are part of the way there, which is why phase one exists: use the assistant to
label cheaply, then record more.
