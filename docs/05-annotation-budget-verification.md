# Adversarial verification of the annotation-budget claims

> **Status: this document supersedes the annotation guidance in documents 03 and 04.** It was
> produced by a verifier instructed to refute rather than confirm, working only from primary
> sources. It overturns the single most actionable claim in the earlier research and resolves the
> two-fold disagreement between documents 03 and 04 on gold-set size.
>
> Read section 1 and the single recommendation in section 3 if nothing else.

## The headline correction

Earlier research reported that **annotating one bounding box per clip takes a detector from 58% to
88% of fully-supervised performance.** That claim is wrong, and the way it is wrong is instructive.

The verifier traced it to three separate origins, none of which says it. In Papadopoulos and
colleagues, CVPR 2016, human *verification* with **zero boxes drawn** reaches 58% mean average
precision against 66% for full supervision on VOC 2007. And 58 divided by 66 is 87.9%.

**The 58% and the 88% are the same result stated two ways, not a before and an after.** The apparent
gain was an artefact of comparing a number against its own ratio. Splicing figures from two
unrelated papers on two different datasets reproduces the same pair coincidentally.

The unit is also wrong. Every underlying result is **per image**, never per clip. A search of
full texts for a detector trained from literally one box per video returned nothing.

**What survives, and it matters.** Papadopoulos does transfer in kind, because its input supervision
is image-level class labels with no boxes, which is almost exactly this project's video-level
multi-label situation. Reaching 88% of fully-supervised with zero boxes drawn, by verifying machine
proposals, is real. With 2026-era proposals rather than the 2014 EdgeBoxes it used, 88% should be a
floor rather than a ceiling. So **verify-and-correct remains the top-ranked use of annotation time,
just for a different reason than was originally claimed.**

One box per clip also stays in the plan, but on different grounds: it is **SAM 2's optimal prompt**,
not a route to 88%.

## 1. Verdict summary

| Claim | Verdict | Correction |
|---|---|---|
| One box per clip gives 58% to 88% | **Wrong** | Same result stated twice; per image, not per clip; cluttered-scene regime caps at 58 to 69% |
| 68-point swing from label prototypicality | **Overstated** | CIFAR-10 classification at 1 label per class, oracle selection. The 10% floor is chance, a non-converged run. The paper's own appendix gives ~37.5%, so the honest range is ~43 points. Realistically capturable against random: **+13.7**. Not detection |
| Learning-loss selection at minus 5.07 | **Confirmed but misframed** | Read from the **largest** budget column, roughly 12,000 boxes. 1.30 of the gap pre-exists at cycle zero. Selection-only deficit is about minus 3.8 |
| Core-set at plus 0.07, "zero" | **Confirmed but cherry-picked** | Smallest of ten per-cycle deltas. Mean is plus 0.21, and core-set is at or above random at every cycle |
| ProbCover at minus 2.4 | **Sign-inverted framing** | That is at roughly 1,414 labels. At 202 labels ProbCover is **plus 7.9** over random. It is also linear-probe top-1 accuracy, not mean average precision |
| Uncertainty sampling at minus 6.5 to minus 8.3 | **Confirmed, not a general law** | Two cells of one row. In those same rows **8 of 11 other methods beat random**. The source paper concludes it finds "no support for the notion that uncertainty sampling is ineffective in low-budget" active learning |
| MaxHerding is the only verified positive | **Numbers right, "only" wrong** | True range plus 0.8 to plus 1.7. Others also beat random, including one at plus 9.3 |
| "35 seconds per box" is a crowd-work figure | **Partially confirmed** | No source prints 35. It is a rounding of 34.5 that drops a 7.8-second stage. The source's own totals are **42.4 seconds median, 88.0 mean** |
| Professionals draw a box in 3.7 to 5.9 seconds | **Partially confirmed** | 3.7 seconds is **mouse-gesture time only**, excluding object search, review and quality assurance. The same study's end-to-end figure is 81.7 seconds per image, about **26.5 seconds per box** |
| Verify-and-correct buys 2.3 to 3.7x | **Confirmed, and conservative** | Google's like-for-like measurement found verification at 8.5 seconds per box against drawing at 7.4, so verification was about **15% slower**, which is why they abandoned it for 13.1 million boxes. Deployed gains elsewhere: **1.49x and 1.54x** |
| Extreme clicking is 44% slower for professionals | **Confirmed verbatim, with a caveat** | The 44% is **not statistically significant**. The 87% increase in annotation creation time **is**, at p below 0.001. Quote the 87% |
| 10 consecutive frames inflate the confidence interval 2.8x, about 9x variance | **Confirmed for the interval, variance wrong** | 2.8 squared is **7.84**, not 9. It also does not hold for frames 2 seconds apart, where the inflation is only 1.8 to 2.3x |

## 2. Why most of these numbers do not transfer

The verifier's central point is that the active-learning negatives were read off the **largest**
budget column of their tables, then presented as small-budget results. Three of four were.

The conclusion for this project nevertheless survives, on a better argument:

> At roughly 150 clips you sit **below the smallest budget at which any active-learning method for
> detection has been shown to beat random selection.** The one strong positive for detection used
> 100,000 to 847,000 images drawn from a 2 million image pool, a selection ratio of about 10 to 1.
> If you intend to annotate nearly your whole pool, active learning cannot help by construction.

So do not build active learning, but understand that the reason is budget scale and selection ratio,
not that the methods are broken.

Two further transfer failures worth noting. Classification-to-detection transfer fails outright: one
method is plus 39.4 on CIFAR-10 classification and minus 2.2 on detection in the same table. And the
3.7-second box is a familiar, easy class drawn by a full-time annotator. A domain expert labelling
unfamiliar fine-grained equipment is dominated by **class decision time**, which that measurement
excluded, so the realistic figure is nearer 26.5 seconds per box.

## 3. The resolved annotation budget

Documents 03 and 04 disagreed, recommending roughly 600 and roughly 250 to 300 frames. The verifier
resolved it with an explicit power analysis, and found both were arguing about a saturating axis
while ignoring the one that matters.

**Doubling from 300 to 600 frames buys 0.66 mean average precision points of confidence interval.**
The confidence interval floors regardless of frames added, because the floor is set by **clip count**,
not frame count.

### Recommendation: 450 gold frames, as 3 frames per clip across all 150 clips, at least 2 seconds apart, scored by 5-fold cross-validation over clips rather than a fixed holdout

Assumptions behind the table: 3 ground-truth instances per frame, a per-instance hit rate of 0.65,
intra-clip correlation of 0.6, 95% two-sided intervals, and 80% power. The cluster column resamples
clips rather than boxes. All values are the verifier's derivation.

| Clips | Frames per clip | Frames | Boxes | Cluster interval | Effective n | Detectable difference |
|---|---|---|---|---|---|---|
| 30 | 4 | 120 | 360 | 8.01 | 136 | 4.26 |
| 50 | 4 | 200 | 600 | 6.06 | 238 | 3.22 |
| 150 | 1 | 150 | 450 | 5.43 | 297 | 3.83 |
| 150 | 2 | 300 | 900 | 4.35 | 462 | 2.59 |
| **150** | **3** | **450** | **1350** | **4.02** | **541** | **2.08** |
| 150 | 4 | 600 | 1800 | 3.69 | 642 | 1.90 |
| 150 | 8 | 1200 | 3600 | 3.31 | 800 | 1.33 |

Four findings follow, and each changes what you would otherwise do.

**Spread beats depth at equal cost.** Holding 300 frames fixed: 300 clips at 1 frame gives an
interval of 3.79, while 30 clips at 10 frames gives 6.85. **Densely boxing a few clips is the single
worst use of evaluation budget**, which contradicts the obvious instinct.

**Cross-validation is a free 1.4-fold power gain.** The same 300 frames give 4.40 under 5-fold
cross-validation across all 150 clips, against 6.11 for a fixed 50-clip holdout. Every frame also
stays available for training in four folds out of five.

**Rare-class average precision is unfalsifiable below about 8 to 12 clips per class.** With 2 clips
the interval halfwidth is 19.6 points. With a single clip the cluster bootstrap degenerates to zero
width, meaning the number cannot be wrong because it cannot be tested.

**Allocate frames equally across classes, not in proportion to frequency.** Because mean average
precision is an unweighted class mean, equal allocation gives 1.00 times the standard deviation
while proportional-to-frequency gives 1.38 and a skewed allocation 1.74. So annotate every clip
containing a rare class, and subsample clips of common ones.

**Why 450 is sufficient.** The effects actually in contention are 5 to 8 points: zero-shot to
one-shot at plus 7.8, one-shot to ten-shot at plus 5.0, adding tags at plus 6.7, and semi-supervised
video detection at plus 6.3. A 450-frame gold set resolves those with a 2.4 to 3.7-fold margin. Only
pseudo-label refinement gains of about 2 points remain marginal.

One caveat on the clustering correction. The correlation is driven by scene and instance identity
rather than temporal proximity, so the same physical instrument under the same lighting stays
correlated even 10 seconds apart. Spacing frames 2 seconds apart decorrelates less than assumed.
Measure the actual correlation on your first 20 clips.

## 4. Recommended allocation of 4 to 8 hours

| Task | Volume | Time | Why |
|---|---|---|---|
| Vocabulary and prompt design, including a false-friend audit | — | 0.5 h | The cheapest high-impact work available. See the homonym findings in document 02 |
| Verify and correct detector proposals, one seed box per object per clip | ~450 boxes | 1.5 h | Best-evidenced option. 88% of fully supervised with zero boxes drawn, from image-level labels |
| Scrub and refine propagation output per clip | 150 clips | 1.5 h | Turns the seed boxes into dense per-frame boxes and track identities |
| Verify to gold, 3 frames per clip, as the evaluation set | 450 frames | 1.0 h | Gives a detectable difference of 2.08 |
| Top up the five classes with no public boxes to 8 or more clips each | ~60 boxes | 0.2 h | Below 8 clips, per-class precision is unfalsifiable |
| Track-identity spot check on about 15 clips | 15 clips | 0.25 h | Measurement only. ByteTrack association needs no track labels |
| **Core total** | | **~5 h** | |

If 8 hours are available, add a second verification round, and video-level-only labels on about 100
new clips from the unlabelled pool, since tags recover 34.7 of the 36.8 mean average precision that
full boxes provide.

**Ranked by evidence:** verify-and-correct machine proposals first; one box per clip second, as a
propagation prompt rather than for the refuted reason; boxes for the five missing classes third;
more video-level-only labels fourth; densely boxing a few clips fifth and clearly worst; track
identities last and near-zero value.

## 5. Buy rather than build, for the unambiguous classes

Published per-box prices: Google Vertex AI at 63 dollars per thousand boxes, Roboflow at 10 cents
per box, and one vendor at 2 cents per object on a 6 dollar hourly rate.

So 1,800 boxes costs between 36 and 180 dollars. Break-even against 5 hours of your own time is an
hourly rate of only 8 to 39 dollars.

**Outsource bulk drawing for the unambiguous classes such as beaker, flask and sink. Keep your own
hours for the two things money cannot buy: deciding the class boundaries, and adjudicating
centrifuge against shaker against spectrophotometer.**

A telling detail: Google's price per box at the measured crowd wage implies about 29.5 seconds of
paid labour per box. The market still prices a box as if it takes 30 seconds, not 4.

## 6. A more pessimistic performance prior

The verifier flags that COCO numbers are the wrong reference for this domain. On the 35-dataset
Object Detection in the Wild suite, zero-shot Grounding DINO averages 22.3 to 26.1 average precision
but has a **median of 11.9 to 18.4**, with specialised domains as low as 0.25 to 9.1.

Laboratory equipment is fine-grained and largely out of vocabulary, so expect the low end. Note that
those are COCO-style figures averaged over overlap thresholds, so they are not directly comparable
to the average-precision-at-0.5 estimates in the executive summary, and the recommended pipeline
adds a warm start, pseudo-labels and distillation on top of zero-shot. The direction of the
correction is nonetheless downward.

## 7. Three numbers to stop quoting

- **"35 seconds per box."** A rounded two-stage subtotal that drops a third stage, not a measurement.
- **"5x faster extreme clicking."** A cross-population comparison. For professionals it was slower.
- **"10 to 20x from verification."** One yes-or-no question divided by a three-stage crowd pipeline. Even the original authors claimed only 6 to 9x, and deployed gains were about 1.5x.

## 8. What could not be verified

No published empirical validation of confidence intervals for detection mean average precision could
be found, so the power analysis rests on standard design-effect theory plus simulation rather than on
a published variance study. Professional throughput in boxes per hour is unpublished by every vendor.
Per-box pricing for most commercial annotation services is quote-only. The intra-clip correlation for
this specific footage is unmeasurable without the data, though conclusions hold across a correlation
range of 0.4 to 0.9.

The verifier also found several misattributed arXiv identifiers in the material it was given, and one
cited paper title that does not exist. Citations in documents 03 and 04 should be spot-checked before
being quoted onward.

---

Full working notes, including every extracted table, follow.

# Adversarial verification: annotation-budget claims for lab-equipment video detection

Status: in progress. Tags: `[SRC]` = measured in a primary source, `[DER]` = my derivation,
`[EST]` = my estimate.

## Confirmed by me directly (primary sources fetched)

### *** CLAIM 1 RESOLVED: the "58% to 88%" is a MISREADING of one table ***
arXiv **1602.08405**, Papadopoulos, Uijlings, Keller, Ferrari, CVPR 2016,
"We don't need no bounding-boxes: Training object class detectors using only human verification".

Table 1, PASCAL VOC 2007 trainval (5011 images), VOC-style mAP at IoU >= 0.5 `[SRC]`:
- VGG16 row verbatim: "VGG16 | 55% | 61% | **58%** | **66%**" where 58% = their Yes/No
  verification method and 66% = the FULLY SUPERVISED reference.
- AlexNet, reduced trainval (3550 images): "The object detectors learned by our scheme achieve
  **45% mAP**, almost as good as the fully supervised ones (**51% mAP**)."

**58/66 = 87.9% and 45/51 = 88.2%.** `[DER]`

=> The claim "takes a detector from 58% to 88% of fully-supervised" conflates the **absolute mAP
(58%)** with the **ratio to fully-supervised (88%)**. They are the same result stated two ways, not a
before/after. There is no 58% starting point and no 30-point gain.

What the setting actually is `[SRC]`:
- Supervision input: **image-level class labels** — "Given a set of training images with image-level
  labels, our scheme iteratively alternates between updating object detectors, re-localizing objects
  in the training images, and querying humans for verification."
- **No human-drawn boxes at all** — trains "without ever drawing any bounding-box".
- Unit of human work = a **verification click**: "Yes/No verification...took on average 1.6 seconds
  per verification"; YPCMM 2.4 s. "96% CorLoc after checking each image only 2.5 times on average"
  (~8,875 verifications over 3,550 images). `[SRC]` + `[DER]` for the count.
- Proposals from **EdgeBoxes** (not box-supervised); Fast R-CNN AlexNet/VGG16, ImageNet-pretrained.
- Annotators: "five annotators from our university" — in-house, not crowdworkers.
- Time: "Our method instead requires only 5.8 hours, a reduction in human effort of a factor of
  6x-9x", against "33 hours of annotation time (when assuming an optimistic 26 s per image), or even
  53 hours (when assuming a more realistic 42 s per image)" — both **per image**, cited to Su et al.
- Dataset scale: 3,550-5,011 **images**, PASCAL VOC. Average ~1.5 object instances/image (VOC is
  much less cluttered than COCO's 7.3). `[EST]` for the VOC density figure.

Transfer to this project: the UNIT is wrong in the claim (verification clicks, not one box per clip)
and the scale is wrong (5k images, not 150 clips), BUT the supervision assumption — class labels
known at the image level, no boxes — is **almost exactly the user's situation** (video-level
multi-label). This makes it the single most relevant paper in the whole set, and it argues for
verify-and-correct, not for drawing boxes. `[DER]`
Modernisation note `[EST]`: EdgeBoxes (2014) is a far weaker proposal source than OWLv2 or
Grounding DINO; the 88% figure should be a floor, not a ceiling, with 2026 proposals.

### SIOD — the likely origin of the "one box" claim
- arXiv 2203.15353, "SIOD: Single Instance Annotated Per Category Per Image for Object Detection",
  Li, Pan, Yan, Tang, Zheng (CVPR 2022). ar5iv full text Table 1, MS COCO.
- Dataset built by "randomly preserving one annotation for each existing category in each image"
  from COCO2017 train = 117,316 images. `[SRC]`
- Annotation density retained: **2.93 instances/image** vs 7.28 full = **40% of instances**. `[SRC]`
  => roughly 344k boxes, NOT one box per image, and NOT one box per video.
- Table 1 `[SRC]`:
  - CenterNet-R18: FSOD AP 17.3 / SIOD-base 13.9 / DMiner 16.8
  - CenterNet-R101: FSOD AP 22.6 / SIOD-base 15.1 / DMiner 19.7
  - Faster-RCNN R50-C4: FSOD 32.8 / base 27.0 / DMiner 29.2
  - FCOS R50-FPN: FSOD 27.1 / base 22.0 / DMiner 23.6
- Ratios `[DER]`: R101 base = 15.1/22.6 = **66.8%**; DMiner = 19.7/22.6 = **87.2%**.
  87.2% is close to the claimed "88%" but the baseline is 66.8%, **not 58%**.
- Metric is COCO **AP (mAP50-95)**, not mAP50. `[SRC]`

### Omni-DETR — supervision-type ladder
- arXiv 2203.16089, Table 2, COCO-standard-10% `[SRC]`:
  10%-supervised baseline 28.0 mAP; +90% Tags 34.7; +90% Tags w/counts 35.2;
  +90% Points 34.1; +90% Points w/tags 35.7; +90% extreme-clicking boxes 36.4;
  +90% full boxes 36.8.
- Table 5 (tags) at 1/5/10/20%: 20.1 / 31.7 / 35.9 / 38.1 vs supervised 11.0 / 23.7 / 29.2 / 33.6. `[SRC]`
- Table 6 (points) at 5/10/20/30%: 32.5 / 37.1 / 39.0 / 40.1 vs Point DETR 26.2 / 30.4 / 33.3 / 34.8. `[SRC]`
- Reading: **image-level tags recover most of the gap that boxes would** (34.7 vs 36.8 when added to a
  10% box-supervised base). Tags are cheap. This is directly relevant.

### ByteTrack — tracking needs NO track-identity labels for training
- arXiv 2110.06864. The BYTE association step is **training-free**: Kalman filter motion prediction
  plus IoU matching. `[SRC]`
- ReID is **optional**; in their ablation motion-only scored **76.6 MOTA** vs **76.3 MOTA** with ReID,
  i.e. ReID did not help. `[SRC]`
- MOT17 test: 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA, detector YOLOX-X. `[SRC]`
- Consequence `[DER]`: annotating track identities has **no training value** for a
  tracking-by-detection pipeline. Its only value is measuring IDF1/HOTA. Budget accordingly.

## Power analysis (my derivation, pure-Python cluster bootstrap)

Generative model: C clips x m annotated frames x 3 ground-truth instances/frame; per-instance hit
probability logit = mu + clip effect + frame effect + shared instance difficulty + model noise;
base hit rate 0.65; paired A/B via common random numbers; 95% CI from bootstrap resampling CLIPS.

### Design effect — the "2.8x CI inflation" claim
Analytic: DEFF = 1 + (m-1)rho, CI inflation = sqrt(DEFF), ceiling sqrt(m). `[DER]`

| m | rho needed | DEFF | CI inflation |
|---|---|---|---|
| 10 | 0.90 | 9.10 | 3.02x |
| 10 | 0.76 | 7.84 | **2.80x** |
| 10 | 0.72 | 7.48 | 2.73x |
| 10 | 0.50 | 5.50 | 2.35x |
| 10 | 0.30 | 3.70 | 1.92x |

Simulation agreed (2.74x at latent ICC 0.9 == observed rho ~0.72). `[DER]`
Verdict: **2.8x CI inflation is sound for 10 consecutive frames**, but it requires observed
intra-clip correlation rho ~= 0.76. The companion "~9x variance" is arithmetically wrong:
2.8^2 = **7.84x**, and 9x variance would need rho = 0.889.

### MDE vs gold-set size (averaged over 14 eval-set draws, ICC 0.6)

| clips | fr/clip | frames | SE paired | MDE80 paired | 95% CI on absolute mAP |
|---|---|---|---|---|---|
| 20 | 4 | 80 | 1.81 | 5.07 | +-9.61 |
| 30 | 4 | 120 | 1.48 | 4.14 | +-8.12 |
| 50 | 4 | 200 | 1.13 | 3.17 | +-6.29 |
| 50 | 6 | 300 | 0.95 | 2.65 | +-5.95 |
| 75 | 4 | 300 | 0.90 | 2.53 | +-5.31 |
| 100 | 3 | 300 | 0.93 | 2.61 | +-4.77 |
| 150 | 1 | 150 | 1.40 | 3.93 | +-5.29 |
| 150 | 2 | 300 | 0.93 | 2.60 | +-4.33 |
| 150 | 4 | 600 | 0.67 | 1.87 | +-3.71 |
| 150 | 8 | 1200 | 0.49 | 1.36 | +-3.30 |

### Saturation — frames per clip stop paying after ~3
50 eval clips, marginal gain in CI halfwidth from the m-th frame: `[DER]`
1->2 frames +2.03 pts, 2->3 +0.70, 3->4 +0.58, 4->6 +0.14, 6->8 +0.11, 8->24 +0.66 total.
CI floors at ~+-5.2 (ICC .6) / ~+-6.4 (ICC .9) no matter how many frames are added.
**The floor is set by the clip count.**

### Spread beats depth at equal cost (300 frames fixed)

| clips x frames | 95% CI (ICC .6) | 95% CI (ICC .9) |
|---|---|---|
| 300 x 1 | +-3.79 | +-3.82 |
| 150 x 2 | +-4.38 | +-4.57 |
| 75 x 4 | +-5.23 | +-5.75 |
| 30 x 10 | +-6.85 | +-8.00 |
| 20 x 15 | +-8.00 | +-9.68 |

### Rare-class AP is uninformative
95% CI halfwidth on a SINGLE class's AP vs number of clips containing it: `[DER]`
2 clips +-19.6, 3 clips +-19.0, 5 clips +-18.6, 8 clips +-14.9, 12 clips +-12.5, 20 clips +-9.7.
1 clip = cluster bootstrap degenerates to zero width, i.e. **unfalsifiable**.

### Cross-validation over clips strictly dominates a fixed holdout

| design | eval clips | eval frames | 95% CI abs | MDE80 |
|---|---|---|---|---|
| fixed holdout 50 clips x 3 fr | 50 | 150 | +-6.94 | 3.76 |
| fixed holdout 50 clips x 6 fr | 50 | 300 | +-6.11 | 2.55 |
| 5-fold CV, 150 clips x 2 fr | 150 | 300 | **+-4.40** | 2.60 |
| 5-fold CV, 150 clips x 3 fr | 150 | 450 | +-3.97 | 2.22 |
| 5-fold CV, 150 clips x 4 fr | 150 | 600 | +-3.79 | 1.84 |

Same 300 annotated frames: CV over all 150 clips gives +-4.40 vs +-6.11 for a 50-clip holdout,
AND every frame is also available for training in 4/5 folds. Free 1.4x power gain. `[DER]`

### GLIP ODinW — THE most transferable benchmark for this project
arXiv 2112.03857, Appendix E.3 Table 11. ODinW = 13 real-world domain-specific detection datasets. `[SRC]`

| Model | 0-shot | 1-shot | 3-shot | 5-shot | 10-shot | All data |
|---|---|---|---|---|---|---|
| DyHead-T (COCO) | - | 31.9+-4.1 | 44.2+-0.4 | 44.7+-2.1 | 50.1+-2.0 | 63.2 |
| DyHead-T (O365) | - | 33.8+-4.3 | 43.6+-1.2 | 46.4+-1.4 | 50.8+-1.6 | 60.8 |
| GLIP-T (A) | 28.7 | 43.5+-1.5 | 48.8+-0.4 | 50.4+-0.7 | 54.1+-0.5 | 63.6 |
| GLIP-T (B) | 33.2 | 48.0+-0.8 | 52.0+-0.4 | 53.2+-0.9 | 54.9+-0.7 | 62.7 |
| GLIP-T (C) | 44.4 | 49.6+-0.3 | 53.8+-0.2 | 54.8+-1.0 | 57.2+-1.1 | 63.9 |
| GLIP-T | 46.5 | 51.1+-0.1 | 54.9+-0.3 | 56.4+-0.5 | 58.4+-0.2 | 64.9 |
| GLIP-L | 52.1 | 59.9+-1.7 | 62.1+-0.8 | 64.2+-0.4 | 64.9+-0.9 | 68.9 |

- "X-shot" = "we randomly sample the dataset such that there are at least XX examples per category" —
  examples are IMAGES, per CATEGORY. 3 seeds per setting. `[SRC]`
- As % of All-data `[DER]`: GLIP-L 0-shot **75.6%**, 1-shot **86.9%**, 3-shot 90.1%, 5-shot 93.2%, 10-shot 94.2%.
  GLIP-T(A) 0-shot 45.1% -> 1-shot 68.4%. GLIP-T(B) 53.0% -> 76.6%. GLIP-T(C) 69.5% -> 77.6%.
- **1-shot 86.9% is the real "88%".** The matching "58%" does not appear; the 0-shot figure is 75.6%
  for GLIP-L and 45-53% for the weaker variants.
- "a zero-shot GLIP-L outperforms a 10-shot supervised baseline (Dynamic Head) pre-trained on
  Objects365 while a 1-shot GLIP-L rivals with a fully supervised Dynamic Head" `[SRC]`
- ODinW train-set sizes (Table 9) `[SRC]`: Packages 19, Mushrooms 41, AerialDrone 52, Thermal 142,
  Raccoon 150, Shellfish 406, Aquarium 448, Pothole 465, Vehicles 878, Rabbits 1980, Pistols 2377,
  EgoHands 3840, PascalVOC 13690. Median ~450; **several are 19-150 images, i.e. exactly this project's scale.**
- Prompt tuning: "by only tuning the prompt embeddings, GLIP-T and GLIP-L can achieve performance
  close to full-model tuning" `[SRC]` -> fits 2x3060 12GB.

### Zero-shot open-vocabulary ceilings
- OWLv2 (2306.09683) `[SRC]`: LVIS APr **44.6** / AP 49.4 (L/14); APr 47.2 (G/14); OWL-ViT L/14 APr 31.2.
  "Neither the annotator nor our models have seen any human-generated box annotations for LVIS_rare
  classes." Self-trained on WebLI pseudo-boxes (10B images), then fine-tuned on LVIS_base.
- Grounding DINO (2303.05499) `[SRC]`: zero-shot COCO 46.7-48.4 AP (T, 172M), 52.5 (L, 341M);
  zero-shot LVIS 25.6-27.4 (T), 33.9 (L); **ODinW zero-shot average only 20.0-26.1 AP**.
- YOLO-World (2401.17270) `[SRC]`: zero-shot LVIS AP/APr = S 26.2/19.1, M 31.0/23.8, L 35.4/27.6;
  zero-shot COCO AP 37.6/42.8/44.4-45.1; 13M/29M/48M params; 52-74 FPS on V100 re-parameterised;
  "prompt-then-detect" offline vocabulary = add classes without retraining.
- **Key adversarial point `[DER]`: COCO/LVIS zero-shot numbers flatter the in-the-wild case.
  Grounding DINO's ODinW average of 20-26 AP is the honest prior for lab equipment.**

### TFA — few-shot detection, and the real size of selection noise
arXiv 2003.06957 `[SRC]`:
- VOC novel AP50, TFA w/cos R101: 1-shot 39.8, 2-shot 36.1 (non-monotonic!), 3-shot 44.7,
  5-shot 55.7, 10-shot 56.0. COCO novel AP: 10-shot 10.0, 30-shot 13.7 (no 1-shot in Table 3).
- Verbatim: "the average AP50 across 40 runs is around 15 points lower than the AP50 on the first
  run in the 1-shot case"; "the accuracy measurements have high variance, making published
  comparisons unreliable"; "the sample variance is large due to the few samples that are used for training".
- "shot" = K annotated objects/instances per class. `[SRC]`
- **This is the honest magnitude of sample-selection sensitivity in few-shot DETECTION: ~15 points
  at 1-shot, and +-4.1 std across seeds for DyHead 1-shot on ODinW. Not 68.** `[DER]`

## Public box availability for the five "no public data" classes (my checks, official lists)

| class | Open Images V7 boxable (600) | LVIS v1 (1203) | Objects365 (365) | verdict |
|---|---|---|---|---|
| centrifuge | absent | absent | absent | **no public boxes - confirmed** |
| autoclave | absent | absent | absent | **no public boxes - confirmed** |
| orbital shaker | "Cocktail shaker", "Salt and pepper shakers" | `shaker` (freq c) = "a container in which something can be shaken" | absent | **false friend; no public boxes** |
| spectrophotometer | absent | absent | absent | **no public boxes - confirmed** |
| fume hood | absent | `fume_hood` id 565, **frequency 'f'**, synonyms [fume_hood, exhaust_hood], def "metal covering leading to a vent that exhausts smoke or fumes" | `Extractor` | **claim PARTLY WRONG: token exists and is frequent, but the referent is a KITCHEN RANGE HOOD** |

Adjacent classes that DO have public boxes: `Beaker` (OI /m/0d20w4), `Flask` (O365), `microscope`
(LVIS freq 'r' = rare, <10 imgs; also O365), `Syringe` (OI + LVIS 'r'), `Scale`, `Glove`, `Sink`, `Tap`. `[SRC]`
LVIS v1 frequency split overall: 405 frequent / 461 common / 337 rare. `[SRC]`

## Cross-class allocation (my derivation)
mAP is an unweighted mean over classes, so Var(mAP) = (1/K^2) sum_c Var(AP_c), and
Var(AP_c) ~ sigma^2 / n_c (n_c = clips annotated for class c). Minimising sum_c 1/n_c subject to
sum n_c = N gives **n_c equal across classes** (not proportional to frequency). `[DER]`

| allocation of 60 clip-slots over 12 classes | sum(1/n)/K^2 | relative SD of mAP |
|---|---|---|
| equal (5 each) | 0.01667 | 1.00x |
| proportional to frequency | 0.03185 | 1.38x |
| heavily skewed | 0.05035 | 1.74x |

Simulation agreed in ordering: rare-first +-5.69, equal +-5.82, proportional +-5.90,
common-only-ignore-tail +-6.09 (95% CI on mAP). `[DER]`

### SAM 2 — one box per clip propagates to the whole clip, with identity
arXiv 2408.00714 `[SRC]`:
- Zero-shot across 17 video datasets, prompt on first frame only (Table 4):
  1-click **64.7** J&F, 3-click 75.3, 5-click 77.6, **bounding box 74.4**, GT mask 79.3.
- Semi-supervised VOS (mask prompt, Hiera-L): DAVIS17 **90.7**, YouTube-VOS19 89.3,
  **MOSE val 77.9**, SA-V val 77.9. MOSE is the crowded/occluded benchmark -> expect ~77, not ~90,
  in cluttered lab footage. `[SRC]` + `[DER]` for the read-across.
- Class-agnostic: "can be prompted by clicks (positive or negative), boxes, or masks" and outputs
  only masks, no semantic labels. `[SRC]` -> the user's VIDEO-LEVEL class labels supply the name.
- Identity maintained across frames via a memory bank of past predictions. `[SRC]`
- Data engine (MASK annotation, not box): Phase 1 SAM-assisted per-frame **37.8 s/frame** ->
  Phase 2 SAM+SAM2-Mask propagation **7.4 s/frame (~5.1x)** -> Phase 3 full SAM 2 with
  "occasional refinement clicks" **4.5 s/frame (~8.4x)**. The 8.4x baseline is Phase 1
  (already SAM-assisted), NOT raw manual. `[SRC]`
- Speed: 43.8 FPS Hiera-B+, 30.2 FPS Hiera-L on A100. `[SRC]`
- **Consequence `[DER]`: "one box per clip" is exactly SAM 2's optimal prompt. One box per object per
  clip yields dense per-frame boxes AND track identities for the whole clip. This is the mechanism
  that makes the headline claim's UNIT (per clip) the right one, even though its cited numbers are wrong.**

### Counter-evidence on active learning: it works, but only at huge scale
arXiv 2004.04699 "Scalable Active Learning for Object Detection" `[SRC]`:
- AL 73.2% wMAP vs random 69.2% wMAP = **+4.0 points**, at iteration 3 with ~700k images.
- Initial labelled pool 100k images; 200k images selected per round; final labelled set 847k;
  unlabelled pool 2,000,000 images.
- Best acquisition: Det-Ent (confidence + boxes). Labelling cost parity "within 5%".
- **Decisive read-across `[DER]`: the only strong AL-for-detection positive is at a ~10:1
  pool-to-budget ratio. With 150 labelled clips that you intend to annotate almost entirely, the
  selection ratio is ~1:1 and AL cannot help by construction — there is nothing to select.**

### Annotation-time anchors I obtained firsthand
- Verification click, in-house annotators: **1.6 s** (Yes/No), **2.4 s** (YPCMM).
  arXiv 1602.08405 `[SRC]`
- AMT box drawing, cited from Su et al. 2012: **26 s per image** without quality control,
  **42 s per image** with quality control. arXiv 1602.08405 quoting its ref 50 `[SRC]`
- ILSVRC (arXiv 1409.0575) confirms the AMT three-step pipeline — "Drawing, Quality verification,
  Coverage verification" — and points to Su et al. 2012 for "evaluation of the overall cost";
  it reports **no seconds figures itself**. `[SRC]` So the ~26-42 s family is confirmed
  crowdsourced/AMT in lineage.
- SAM 2 mask annotation with propagation + refinement clicks: **4.5 s/frame**, vs **37.8 s/frame**
  for SAM-assisted per-frame work. arXiv 2408.00714 `[SRC]` (MASKS, not boxes.)

### Recommended allocation (my derivation, ~4.6 h core, extensible to 8 h)
Architecture: open-vocab detector (YOLO-World-L / Grounding DINO-T on 12GB) names objects ->
SAM 2 propagates each seed through the clip with identity -> ByteTrack or SAM2 memory for tracks.
Nothing in this pipeline needs track-identity training labels.

| # | task | volume | unit time | total | basis |
|---|---|---|---|---|---|
| 1 | vocabulary + prompt design, false-friend audit (fume hood vs kitchen hood, shaker, flask) | - | - | 30 min | `[EST]` |
| 2 | seed ONE box per object instance per clip | 150 clips x ~3 inst = 450 boxes | ~10 s | 75 min | `[EST]` |
| 3 | scrub + refine SAM 2 propagation, per clip | 150 clips | ~40 s | 100 min | `[EST]`, bracketed by 4.5 s/frame `[SRC]` |
| 4 | verify-to-gold 2 frames/clip = the eval set | 300 frames x 3 = 900 decisions | ~3 s | 45 min | `[DER]` from 1.6 s `[SRC]` |
| 5 | top up the 5 zero-public-box classes to >=8-12 clips each | ~60 boxes | ~10 s | 10 min | `[DER]` from power analysis E |
| 6 | track-ID spot check for IDF1/HOTA sanity | 15 clips | ~60 s | 15 min | `[EST]` |
|  | **core total** | | | **~4.6 h** | |
| 7 | (if 8 h) third gold frame per clip + second verification round | 150 frames + re-verify | | +2.0 h | `[DER]` |
| 8 | (if 8 h) label more clips at video level only from the unlabelled pool | ~100 clips | ~30 s | +0.8 h | `[EST]` |

Evaluation protocol: **5-fold cross-validation over CLIPS**, never a fixed holdout. Same 300
annotated frames give +-4.40 mAP (CV over 150 clips) vs +-6.11 (50-clip holdout), and all frames
remain available for training in 4/5 folds. `[DER]`

### Ranking of the six options in the brief (evidence-weighted)
1. **Verify-and-correct machine proposals** — best-supported. 1602.08405 reaches 88% of fully
   supervised with zero drawn boxes, starting from image-level labels, 6-9x time saving. `[SRC]`
2. **One box per clip** — right UNIT, wrong citation. Justified not by 58->88 but because it is
   SAM 2's optimal prompt (74.4 J&F zero-shot from a box) and yields dense boxes + identity. `[SRC]`
3. **Boxes only for the 5 classes with no public data** — justified: confirmed absent from
   OI V7 / LVIS v1 / O365, and rare-class AP is unfalsifiable below ~8 clips. `[SRC]`+`[DER]`
4. **Label more clips at video level only** — cheap and effective: Omni-DETR shows tags recover
   34.7 of the 36.8 mAP that full boxes give on top of a 10% base. `[SRC]`
5. **Densely box a few clips** — worst use of eval budget: at equal cost, 20 clips x 15 frames gives
   +-8.00 vs +-3.79 for 300 clips x 1 frame. `[DER]`
6. **Annotate track identities** — near-zero value. ByteTrack association is training-free and its
   motion-only variant (76.6 MOTA) matched/beat its ReID variant (76.3). `[SRC]`

## CLAIM 1 — second and third origins found (verification thread)

The 58/88 pair is ALSO reproducible by splicing **two different papers on two different datasets**:
- **58%** = Co-mining (arXiv **2012.01950**, AAAI 2021) Table 1, MS COCO-2017, RetinaNet R50-FPN,
  COCO AP. "Extreme" split = *"the extreme set only preserves one annotation"* (one box per IMAGE).
  **Extreme baseline 20.9 AP vs Full 36.0 AP** `[SRC]` -> **58.1%** `[DER]`. Co-mining itself only
  lifts it to 23.0 = 63.9%. FCOS: 22.1 -> 23.1 vs 38.4 full.
- **88%** = BRL (arXiv **2002.05274**, ICASSP 2020) Table 6, VOC 2007 test, mAP50.
  **RetinaNet+BRL extreme 0.662 vs its own normal 0.753** `[SRC]` -> **87.9%** `[DER]`.
  BRL's VOC "extreme" drops only **64.95%** of boxes => VOC ~2.85 instances/image `[DER]`.
- So **58% is the CLUTTERED (COCO, 7.28 inst/img) regime and 88% is the NEAR-SINGLE-OBJECT
  (VOC, ~2.85 inst/img) regime.** Splicing them yields a claim false in both. This is the single
  most important adversarial finding: the clutter assumption is exactly inverted for lab footage.

Most likely single origin remains **arXiv 1602.08405** Table 1 + its prose, which contains all three
elements in one paragraph `[SRC]`: *"Training Fast R-CNN from one bounding-box per class per image,
results in 55% mAP, while our Yes/No human verification scheme gets to 50% mAP. ... Training with
full supervision leads to 66% mAP, while our verification scheme delivers 58% mAP. Hence, on both
CNN architectures our verification-based training scheme produces high quality detectors, achieving
90% of the mAP of their fully supervised counterparts."*
Note the **roles are inverted**: there, one box per class per image IS a reference condition (55%),
and the method uses **zero** boxes.

Other near-misses `[SRC]`+`[DER]`: Calibrated Teacher (2303.07582) T2 COCO 70%-removed Split-1:
Co-mining 24.9 -> Ours 36.7 vs Full 41.4 = 60.1% -> 88.6% (but the "low" end is a rival method's
score and 70%-removed is ~2.2 boxes/image). SparseDet (2201.04620) T2 VOC AP50 extreme:
BRL 66.20 / Co-mining 69.60 / Ours 75.59 / Oracle 83.09.

### The real per-VIDEO analogue — SSVOD
arXiv **2309.01391** (WACV 2024), *one fully-annotated KEY FRAME per video* (not one box) `[SRC]`:
- ImageNet-VID val, mAP@0.5, 1 frame/video at 25/50/75/100% of videos:
  Supervised image baseline 30.1 / 37.6 / 42.5 / **47.8**;
  Supervised video baseline (SELSA) 39.6 / 43.1 / 48.9 / **55.0**;
  **SSVOD 48.6 / 53.4 / 60.5 / 63.8**.
- Verbatim: *"In case of extremely sparse annotations of one labeled frame per video, an average of
  **22% supervised performance reduction** is observed compared to the dense supervision of 15
  labeled frames per video."*
- **EPIC-KITCHENS (egocentric clutter — closest analogue to lab footage), 1 frame/video:
  Supervised 30.4 +-0.18 -> SSVOD 36.7 +-0.22.** `[SRC]`
- Dense SELSA reference 80.25-84.3 mAP `[SRC, arXiv 1907.06390]` => 1 frame/video sits at
  ~68% -> ~80% of dense `[DER, cross-paper, approximate]`.
- **No paper trains a detector from literally one box per video.** Full-text/abstract searches for
  "one box per clip", "box per video", "one annotated frame per video" returned zero. `[SRC: null result]`

### WSOD — what video-level tags alone actually buy
VOC07 test mAP50 `[SRC]`: WSDDN 34.8, OICR 41.2, PCL 43.5, C-MIL 50.5, C-MIL+FRCNN 53.1.
Fully supervised Fast R-CNN VGG16 on VOC07 = **66.9** `[SRC, arXiv 1504.08083]`.
=> 52.0% / 61.6% / 65.0% / 75.5% / 79.4% of fully supervised `[DER]`; absolute gap 16-32 mAP50 points.
**On cluttered COCO, image-level-labels-only MIL WSOL = 8.9 mAP vs 24.0 fully supervised = 37%**
`[SRC, arXiv 1704.06189]` `[DER]`. Clutter roughly halves the weak-supervision payoff.

### Click supervision — the cheap-annotation curve
arXiv **1704.06189** (CVPR 2017), VOC07, AlexNet Fast R-CNN `[SRC]`:
MIL WSOL 29.6 mAP -> one centre-click **45.9** (+3.8 h) -> two clicks **49.1** (7.6 h total);
full supervision **55.5** (71 h). VGG16: 32.4 -> ... -> two-click **57.5** vs full **65.9**.
=> 53.3% -> 82.7% -> **88.5%** of full `[DER]`. COCO: 8.9 -> 18.3 -> 19.3 vs full 24.0
(clicks **simulated**, not collected) `[SRC]`.
Caveat verbatim: the fully-supervised comparator uses *"one per class per image, for fair comparison"*.

### CORRECTION to my earlier OWLv2 note
The headline **44.6 AP_rare is the OWL-ST+FT row, which consumes LVIS_base human boxes**
(none for the rare classes themselves). The row with no LVIS fine-tuning at all is
**OWL-ST = 33.5 AP_val_all / 34.9 AP_val_rare**. `[SRC]`

### Domain-shift reality check — the number that should set expectations
Grounding DINO Table 4, ODinW-**35** zero-shot `[SRC]`:
**AP average 22.3 (Swin-T) / 26.1 (Swin-L); AP MEDIAN 11.9 (T) / 18.4 (L).**
Per-dataset Swin-T lows: MaskWearing **0.25**, AmericanSignLanguageLetters **0.78**,
HardHatWorkers **4.05**, BCCD 11.96, ChessPiece 15.62, Aquarium 18.64.
=> For a fine-grained, out-of-vocabulary domain like lab equipment, **expect the low end
(single digits to ~20 AP), not the 52.5 COCO headline.** `[DER]`

## CLAIM 2 RESOLVED — the 68 points is CIFAR-10 CLASSIFICATION, and it is an oracle range

Primary source: **FixMatch**, Sohn et al., NeurIPS 2020, arXiv **2001.07685**, **Section 4.4
"Barely Supervised Learning"** (PDF-only; no ar5iv). Verbatim `[SRC]`:
> "Using the same hyperparameters, the model trained only on the most prototypical examples reaches
> a median of 78% accuracy (with a maximum of 84% accuracy); training on the middle of the
> distribution reaches 65% accuracy; and training on only the outliers fails to converge completely,
> with 10% accuracy."

**78 - 10 = 68.** The number "68" is never printed. `[DER]`

- Setting: **CIFAR-10, 10 labels TOTAL (1 per class), top-1 accuracy, ~50k unlabeled images,
  semi-supervised. IMAGE CLASSIFICATION, not detection. No mAP anywhere.** `[SRC]`
- **10% is chance level** for 10-class CIFAR-10 — the low end is a *non-converged run*, not a weak model.
- "Prototypicality" is an **ORACLE** score, not a usable heuristic. Verbatim `[SRC]`:
  > "This example ordering was determined after training many CIFAR-10 models with all labeled data.
  > We thus do not envision this as a practical method for choosing examples for use in SSL"
  The ordering comes from Carlini et al., arXiv **1910.13427** (density/outlier metrics derived from
  ensembles of fully-supervised models). FixMatch does not say which of the five metrics.
- **The paper's own Appendix B.7 / Figure 7 contradicts the 10% floor**: best bucket ~80.5%,
  outlier bucket ~**37.5%** => honest oracle range **~43 points, not 68**. (Figure values recovered
  from the PDF vector coordinates by my delegate — treat as `[SRC, reconstructed]`, with the
  "over 80% accuracy when training on the best examples" sentence as the calibration check.)
  Parsimonious reading: the 10% rests on a **single collapsed run**, so 68 is largely a
  training-instability artifact. `[DER]`
- Confounds in the same paper `[SRC]`: random selection alone spans **48.58%-85.32%** (median 64.28)
  across four 1-per-class draws; and with labels held FIXED, seed alone moves CIFAR-10 error across
  **5.46 / 6.17 / 9.37 / 10.85 / 13.32** (Table 8).
- **Realistically capturable**: oracle-prototypical median 78% vs random median 64.28% =
  **+13.7 points** `[DER]`, and even that end needs the oracle.
- Best *practical, label-free* heuristic — **TypiClust** (arXiv **2202.02794**, ICML 2022):
  CIFAR-10 / 10 labels / FlexMatch: Random 53.8, Balanced 55.9, Coreset 56.6, TPC_RP 80.5,
  **TPC_DC 93.2 = +39.4 over random** `[SRC]`. But it decays fast:
  **CIFAR-100 / 300 labels +20.3; TinyImageNet / 1000 labels +6.4** `[SRC]`. Without SSL the
  prototypicality axis spans only ~10-19 points. `[SRC, read off Fig 8c]`
- **For DETECTION the swing is arithmetically impossible**: TFA 1-shot VOC novel
  **nAP50 = 25.3 +- 2.2 (95% CI over 30 random shot draws)** `[SRC]` — the absolute mean is 25,
  so a 68-point swing cannot exist on that metric.
- Ruled out as origins by e-print grep (no "68" as a gap) `[SRC, null results]`: TypiClust 2202.02794,
  Sorscher 2206.14486, Carlini 1910.13427 (its own CIFAR-10 oracle range is ~14 points),
  TFA 2003.06957, Dhillon 1909.02729, ProbCover 2205.11320, "Making Your First Choice" 2210.02442.

## Open items
- Active learning negative results (delegated, in progress)
- Annotation time economics (delegated, in progress)
- 68-point prototypicality swing (delegated, in progress)
- Active learning negative results (delegated, in progress)
- Annotation time economics (delegated, in progress)

## FINAL POWER TABLE (my derivation)

Assumptions, stated: 3 ground-truth instances per annotated frame; detector per-instance hit rate
p = 0.65; latent intra-clip ICC = 0.6 (frames >= 2 s apart); 95% two-sided; 80% power so
MDE = 2.80 x SE. "binomial (iid)" = 1.96*sqrt(p(1-p)/N) treating every box as independent — an
OPTIMISTIC LOWER BOUND. "cluster" = bootstrap resampling CLIPS — realistic.
DEFF = (cluster/binomial)^2; n_eff = N/DEFF.

| clips | fr/clip | frames | boxes | binom +- | cluster +- | DEFF | n_eff | MDE80 |
|---|---|---|---|---|---|---|---|---|
| 30 | 2 | 60 | 180 | 6.97 | 9.59 | 1.90 | 95 | 5.73 |
| 30 | 4 | 120 | 360 | 4.93 | 8.01 | 2.64 | 136 | 4.26 |
| 50 | 2 | 100 | 300 | 5.40 | 7.26 | 1.81 | 166 | 4.52 |
| 50 | 4 | 200 | 600 | 3.82 | 6.06 | 2.52 | 238 | 3.22 |
| 75 | 4 | 300 | 900 | 3.12 | 4.95 | 2.52 | 357 | 2.65 |
| 100 | 3 | 300 | 900 | 3.12 | 4.84 | 2.41 | 373 | 2.70 |
| **150** | **1** | **150** | 450 | 4.41 | 5.43 | 1.52 | 297 | 3.83 |
| **150** | **2** | **300** | 900 | 3.12 | 4.35 | 1.95 | 462 | 2.59 |
| **150** | **3** | **450** | 1350 | 2.54 | 4.02 | 2.49 | 541 | **2.08** |
| **150** | **4** | **600** | 1800 | 2.20 | 3.69 | 2.80 | 642 | 1.90 |
| 150 | 8 | 1200 | 3600 | 1.56 | 3.31 | 4.50 | 800 | 1.33 |

Pure binomial reference (no clustering): N=150 -> +-7.63; 300 -> +-5.40; 450 -> +-4.41;
600 -> +-3.82; 900 -> +-3.12; 1800 -> +-2.20 points.

**n_eff saturates.** 900 boxes over 150 clips -> n_eff 462. Quadrupling to 3600 boxes ->
n_eff 800, only 1.7x. DEFF climbs 1.52 -> 1.95 -> 2.49 -> 2.80 -> 4.50 as frames/clip goes
1 -> 2 -> 3 -> 4 -> 8, exactly as 1+(m-1)rho predicts.

### Resolution of the 600-vs-250/300 contradiction
- 300 frames (2/clip x 150 clips): abs CI +-4.35, MDE80 **2.59**
- 450 frames (3/clip x 150 clips): abs CI +-4.02, MDE80 **2.08**  <- the knee
- 600 frames (4/clip x 150 clips): abs CI +-3.69, MDE80 **1.90**
Doubling 300 -> 600 frames buys **0.66 points of CI and 0.69 points of MDE**. `[DER]`
The two prior reports were arguing along a **saturating axis** while ignoring the axis that
matters (clip count), and neither separated EVALUATION frames from TRAINING frames.

### Why 450 is enough: effect sizes actually in play `[SRC]`
| change | measured effect | detectable at MDE 2.08? |
|---|---|---|
| GLIP-L zero-shot -> 1-shot (ODinW) | 52.1 -> 59.9 = **+7.8** | yes, 3.7x margin |
| GLIP-L 1-shot -> 10-shot (ODinW) | 59.9 -> 64.9 = **+5.0** | yes |
| Omni-DETR 10% base -> +90% tags (COCO) | 28.0 -> 34.7 = **+6.7** | yes |
| SSVOD on EPIC-KITCHENS, 1 frame/video | 30.4 -> 36.7 = **+6.3** | yes |
| Co-mining pseudo-label mining, COCO extreme | 20.9 -> 23.0 = **+2.1** | marginal |
Every decision worth making in this project is a 5-8 point effect. A 450-frame gold set resolves
those with a 2.4-3.7x margin. 600+ frames is over-engineering the ruler. `[DER]`

**SINGLE RECOMMENDATION: 450 gold frames = 3 frames per clip, >= 2 s apart, across ALL 150 clips,
scored by 5-fold CROSS-VALIDATION OVER CLIPS (never a fixed holdout).**
Those 450 frames are a *verification* target (check SAM 2 output), not a from-scratch drawing target.

## CLAIM 3 — active learning, my own direct verification

### MaxHerding CONFIRMED exactly (and it is an APPENDIX detection experiment)
arXiv **2407.12212**, "Generalized Coverage for More Robust Low-Budget Active Learning" (ECCV 2024).
Primary task is **image classification** (CIFAR10/100, TinyImageNet, ImageNet). Detection appears
only in **Appendix 0.D**: *"we use YOLOv5 on PASCAL VOC dataset. For the active learning methods, we
use features from the penultimate layer."* Metric **mAP50-95**, 5 runs. `[SRC]`

| Method | 200 labels | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| Random | 4.4 +-0.3 | 11.9 +-0.7 | 20.7 +-1.0 | 27.2 +-0.7 | 33.4 +-1.0 |
| Coreset | 4.4 +-0.3 | 11.2 +-0.8 | 19.3 +-0.9 | 26.4 +-0.8 | 33.2 +-0.9 |
| TypiClust | 4.6 +-0.2 | 10.9 +-0.9 | 18.5 +-0.6 | 27.2 +-0.5 | 32.5 +-0.7 |
| **MaxHerding** | **5.6 +-0.1** | **13.4 +-0.7** | **21.5 +-0.8** | **28.9 +-0.5** | **34.9 +-0.6** |

- MaxHerding minus Random: **+1.2, +1.5, +0.8, +1.7, +1.5** `[DER]`. The claimed "+1.2 to +1.5 mAP"
  is exactly the 200- and 400-label cells. Baseline **is** Random. CONFIRMED.
- **But**: at 200 labels Random = 4.4 and MaxHerding = 5.6 mAP50-95 — both detectors are broken.
  The gain is real but on a useless model. At 600 labels the CIs overlap (21.5+-0.8 vs 20.7+-1.0),
  so the gain is not significant there. `[DER]`
- **Bonus finding the prior report missed**: in this same table **Coreset is -0.0/-0.7/-1.4/-0.8/-0.2
  and TypiClust is +0.2/-1.0/-2.2/0.0/-0.9 versus Random** `[DER]`. TypiClust — the star of
  low-budget *classification* AL (+39.4 on CIFAR-10/10-labels) — **LOSES to random on detection.**
  The classification-to-detection transfer fails. This is the strongest single piece of evidence
  for the user's thesis, and it is in the same table as the one positive result.

### Food101 / ProbCover -2.4: NOT FOUND
Checked at dataset-list level `[SRC, null results]`:
- ProbCover's own paper (2205.11320): CIFAR-10/100, TinyImageNet, ImageNet, ImageNet-50/100.
  **No Food101.** No case of ProbCover below random reported in its own paper.
- MaxHerding (2407.12212): CIFAR10/100, TinyImageNet, ImageNet, VOC. **Food101 not mentioned anywhere.**
- Uncertainty Herding (2412.20644): CIFAR-10/100, TinyImageNet, DomainNet, ImageNet. **No Food101.**
- A Cross-Domain Benchmark for AL (2408.00426): Splice, DNA, USPS, FashionMNIST, CIFAR-10,
  News Category, TopV2, Honeypot, Diverging Sine. **No Food101. ProbCover not benchmarked.**
- arXiv abstract search `abs:"Food101" AND abs:"active learning"` -> 2 hits, neither a low-budget
  AL benchmark containing ProbCover.
=> **"ProbCover -2.4 on Food101" is UNVERIFIABLE from my searches.**

### Cold-start AL — the balancing literature the brief asked for
**COLosSAL** (MICCAI 2023, arXiv **2307.12004**) `[SRC]`:
- Definition verbatim: *"When the entire data pool is unlabeled, which samples should one select as
  the initial set? This problem is known as cold-start active learning, a low-budget paradigm of AL
  that permits only one chance to request annotations from experts without access to any previously
  annotated data."*
- Headline verbatim: **"No strategy evaluated in our benchmark consistently outperforms the random
  selection average performance."**
- Qualified exception verbatim: *"TypiClust...achieves comparable or superior performance compared to
  random selection across all tasks."*
- Budgets: **m=5 volumes** (3 for Heart), and m=10 (5 for Heart). Genuinely tiny — the user's regime.
- Also: cold-start AL *"becomes more effective when more budget is allowed."*
- Results are heatmaps of Dice differences; no precise per-cell numbers in the main text.

**A Cross-Domain Benchmark for Active Learning** (arXiv **2408.00426**) `[SRC]`:
- Conclusion verbatim: *"there exists no clear SOTA method for AL. The superiority of methods is
  strongly dataset- and domain-dependent."*
- Documented below-random cases: LeastConfident on semi-encoded USPS q=100 **0.230+-0.034 vs Random
  0.468+-0.024**; BALD **0.285+-0.046 vs Random 0.468+-0.024**. (Classification, tabular/image mix.)
- LL4AL explicitly excluded: *"Learning Loss for AL introduces an updated training of the
  classification model with an auxiliary loss and therefore cannot be compared fairly."*

## CLAIM 3 — the five AL numbers traced (verification thread). ALL REAL, THREE MIS-FRAMED.

Provenance: the five claims merge **three unrelated papers**, none named in the brief.

### Claims 1+2 — Box-Level Active Detection, CVPR 2023, arXiv **2303.13089**, supp. table `tab:voc`
Setting `[SRC]`: VOC0712 trainval (16,551 images / 40K boxes), eval VOC07 test, **Faster R-CNN R50**
(mmdetection), **mAP50**, **BOX-level budget**: 3K boxes init + 1K/cycle, 3 runs, Random baseline present.
- Cycle 9: `LearningLoss* ... 74.73+-0.91` vs `Random ... 79.80+-0.10` -> **-5.07** `[DER; not printed]`
- Cycle 9: `CoreSet ... 79.87+-0.31` vs `Random ... 79.80+-0.10` -> **+0.07** `[DER; not printed]`
- **Cycle 9 is the LARGEST budget point in the table (~12K boxes ~= 30% of all boxes), not a small one.**
- LL4AL per-cycle deltas: -1.30, -3.00, -3.26, -3.80, -4.60, -4.60, -4.80, -5.03, -4.66, -5.07;
  mean **-4.01** `[DER]`. **1.30 of the gap exists at cycle 0 before any AL selection** (66.70 vs 68.00)
  because the row is asterisked: *"Methods marked with * have specialized detector architectures."*
  Selection-only deficit ~= **-3.8** `[DER]`.
- Core-set per-cycle: 0.00, +0.10, +0.20, +0.77, +0.10, +0.10, +0.20, +0.33, +0.24, +0.07;
  mean **+0.21** `[DER]`. **Core-set is >= random at EVERY cycle.** Honest reading: statistically
  indistinguishable (error bars +-0.31 / +-0.10), not "zero at small budgets".

### Claims 3+4 — Gupte et al., "Revisiting Active Learning in the Era of Vision Foundation Models", TMLR, arXiv **2401.14555**
Setting `[SRC]`: **linear probe on frozen DINOv2 ViT-g/14**, **top-1 accuracy** (NOT mAP),
budget = 1 sample/class/iteration, 5 seeds, Random baseline present.
- **Food101**, t=14 (~1,414 labels): ProbCover `84.5+-0.7` vs Random `86.9` -> **-2.4** `[DER]`;
  t=16 (~1,616): `85.1+-0.4` vs `87.5+-0.2` -> -2.4. **Encoded in the source only as a red cell colour.**
- **BUT at t=2 (202 labels) ProbCover `72.3` vs Random `64.4` = +7.9** `[DER]`; t=4 +0.9; crossover ~t=6;
  mean over the 11 reported iterations **~= +0.9** `[DER]`. **The claim inverts the budget dependence.**
- **CIFAR100**, Entropy: t=2 (200 labels) `58.0+-2.3` vs Random `64.5+-2.7` = **-6.5**;
  t=4 (400 labels) `70.4+-3.2` vs `78.7+-1.6` = **-8.3** `[DER]`. That IS the claimed range —
  **two cells of one row of one dataset.**
- **In those same two rows, 8 of 11 other AL methods BEAT random** `[DER]`: at t=2 Margins +5.4,
  BALD +2.9, pBALD +6.7, BADGE +8.0, ProbCover +11.7, Alfamix +13.5, TypiClust +16.3, DropQuery +19.0.
- ***** THE SHARPEST FINDING IN THIS WHOLE REPORT ***** — the paper that is the SOURCE of claims 3
  and 4 states the OPPOSITE of the thesis they are used to support. Verbatim `[SRC]`:
  **"we see no evidence of a phase transition and find no support for the notion that uncertainty
  sampling is ineffective in low-budget AL"**

### Claim 5 — MaxHerding confirmed, but "the only verified positive" is FALSE
Deltas +1.2/+1.5/+0.8/+1.7/+1.5 confirmed (see my own read above). True range **+0.8 to +1.7**,
mean +1.34; the quoted "+1.2 to +1.5" cherry-picks |L|=200/400/1000. `[DER]`
Other verified detection positives vs Random `[SRC]`: sCOMPAS **+2.33 to +4.00** mAP50 VOC
(2303.13089); CALD **+2.9 mAP average** VOC07 (2103.10374); PPAL **+9.3** at 1,241 images
(2211.11612, `Random 51.5+-1.5` vs `DCUS+CCMS 60.8+-0.5`, RetinaNet); MeanEntropy +1.2 to +2.1 every cycle.

### Strongest evidence on the other side `[SRC]`
- **Original LL4AL (arXiv 1905.03677) contradicts claim 1**: *"In the last active learning cycle, our
  method achieves 0.7338 mAP which is 2.21% higher than 0.7117 of the random baseline. The entropy
  and core-set methods, showing 0.7222 and 0.7171 respectively, also perform better than the random
  baseline."* SSD/VGG16, VOC07+12, 1K init +1K/cycle. **LL4AL = +2.21, never evaluated below 1,000 images.**
- Choi et al. (2103.16130, ICCV 2021): VOC07 SSD mAP at 2k/3k/4k — LL4AL +0.04/+0.66/+0.43,
  Core-set +0.00/+0.21/+0.10 `[DER]`.
- **Lüth et al. (arXiv 2301.10625, NeurIPS 2023)** — best-controlled benchmark, verbatim:
  *"The comprehensive study of various starting budgets on all datasets reveals that AL methods are
  more robust with regard to small starting budgets than previously reported. With the exception of
  Entropy we did not observe cold start problems..."* On imbalanced CIFAR-10-LT every AL method beat
  random (`Random 0.3615` vs `Core-Set 0.3939`); but `Entropy 0.4307` vs `Random 0.4720` on
  CIFAR-10 low-label (-4.13) and `BALD 0.4494` vs `Random 0.5298` on MIO-TCD (-8.04).
- DCoM (2407.01804) CIFAR-10 clean crossover: ProbCover vs Random **+5.86** at |L|=10,
  **+7.30** at 200, **-0.77** at 2,600 `[DER]`.
- CALD (2103.10374): *"500 labeled images as random initialization and 500 as budget per cycle"*;
  *"CALD is 8.4% and 7.0% higher than random selection... in the first cycle on VOC 2012 and 2007"*;
  *"in the first cycle, the improvements are the largest and gradually decrease"*.
- Cold start, properly sourced: canonical vision paper is **Chen et al. arXiv 2210.02442 (ICLR 2023)**,
  not "Jin et al.". Farquhar/Gal/Rainforth (2101.11665) cut the other way: *"we show that this bias
  can be actively helpful when training overparameterized models..."*. Chandra et al. (2011.14696)
  *"could not conclusively prove"* a benefit from smart initial pools.
- Where the claim genuinely holds: FreeSel (2309.17342) `Entropy (DeiT-S) 56.33` vs `random 64.21` at
  VOC 3k (-7.88), Core-set `60.59` vs `64.21` (-3.62), called *"the failure case of Core-Set"*.
  Mittal et al. (1912.05361): *"State-of-the-art active learning approaches often fail to outperform
  simple random sampling, especially when the labeling budget is small."* **Caveat: its PASCAL VOC
  work is semantic SEGMENTATION in mIoU with click budgets, not detection mAP.**

### Corrections to the brief's own citations `[SRC, verified by the thread]`
- **arXiv 2009.13687 is NOT Elezi et al.** — it is a physics.optics paper. Elezi et al. = **2106.11921**.
- **arXiv 2104.00231 is not MI-AOD** — MI-AOD = **2104.02324**.
- Do not cite **2203.13450** (Zhan et al. survey) here: no PASCAL VOC, no mAP, no Food101, and it
  excludes Core-set by design.
- "Revisiting Active Learning for Object Detection" does not exist under that title; the intended
  paper is likely **2401.14555** or **2403.14800** ("Deep Active Learning: A Reality Check").

### My synthesis for DETECTION specifically `[DER]`
Density/coverage AL works well for **classification** at 10-400 labels (TypiClust +39.4 at 10 labels).
For **detection** the picture is far thinner: in MaxHerding's own VOC table, **TypiClust (-2.2 at 600)
and Core-set (-1.4 at 600) LOSE to random**, and MaxHerding's +1.2 at **200 images** is the only
verified sub-500-image detection win. CALD starts at 500-1,000 images; PPAL's +9.3 is at 1,241.
**At ~150 clips the user sits BELOW the smallest budget at which any AL method for detection has been
shown to beat random.** Recommendation: random/stratified selection, spend the saved effort on coverage.

## CLAIM 4 — annotation economics (verification thread + my own reads)

### "35 s per box" — population right, number is a re-aggregation
Su, Deng, Fei-Fei, "Crowdsourcing Annotations for Visual Object Detection", AAAI HCOMP workshop 2012,
**Table 1** (author-hosted PDF; AAAI returned 403) `[SRC]`. **AMT crowdworkers**, ImageNet,
10 categories, 200-image subset. Median / mean **per box**:
`Drawing 25.5 / 50.8` · `Quality Verification 9.0 / 21.9` · `Coverage Verification 7.8 / 15.3`
· **`Total 42.4 / 88.0`**. Text: *"our system costs an average of 88.0 seconds of worker time for
each bounding box."* Timing *includes* image download (1-5 s).
- **No primary source prints "35 s."** It is Papadopoulos's rounding of 25.5+9.0 = **34.5 s**, which
  **drops Su's 7.8 s coverage stage**. Extreme-clicking §2 verbatim: *"we use 25.5s+9.0s=34.5s as the
  reference time. This is a conservative estimate: ... the median time increases to 55s. If we use
  average times instead of medians, the cost raises further to 117s."* `[SRC]`

### Professional 3.7-5.9 s — printed, but it is only the MOUSE GESTURE
**Williams et al., "Snapper: Accelerating Bounding Box Annotation...", ACM IUI 2024** `[SRC]`.
**18 full-time annotators** at *"a large technology corporation in which annotators are hired as
full-time employees"*; PASCAL VOC 2012, 90 images, 3-5 objects/image; inside AWS SageMaker Ground Truth.
- §5.3.2: *"Average annotation creation time varied across the **BL (mu=3.7s; sigma=2.5)**,
  EC (mu=6.9s; sigma=16.8), SN-GT (mu=2.9s; sigma=1.9), and SN-DM (mu=2.2s; sigma=1.3) conditions."*
- §5.3.3 edit times: BL **2.2 s**, EC 2.4 s, SN-GT 2.1 s, SN-DM 1.5 s.
- §5.3.1 task times: **BL mu=81.7 s**, EC mu=117.4 s, SN-GT 63.4 s, SN-DM 47.1 s.
- **The 5.9 s is 3.7+2.2 and is NEVER PRINTED — it is a derivation.** `[DER]`
- **Definition is narrow**: creation = mouse-down -> mouse-up. Excludes object search, review, QA.
  The same study's end-to-end cost is **~26.5 s/box of actual professional labour** `[DER]`
  (81.7 s/image at ~3.08 boxes/image).
- Only other measured professional box time: **Open Images V4 (arXiv 1811.00982) §2.4.2:
  *"On average over the complete dataset, it took 7.4 seconds to draw a single bounding box"***
  (Google-internal annotators, extreme clicking; qualification gate *"20 seconds or less"*). `[SRC]`
- **LVIS (1908.03195), Haussmann (2004.04699) and COCO (1405.0312) report NO per-box times** —
  checked directly. COCO gives only *"over 22 worker hours per 1,000 segmentations"* (~79.2 s/instance `[DER]`).

### Verify-and-correct 2.3-3.7x — CONFIRMED as arithmetic, and conservative
Inputs `[SRC]`: 1.6 s Yes/No verification (five university annotators, n=5); 2.4 s YPCMM;
EC Table 2 whole-pipeline amortized **4.5 s/instance at 81% IoU>0.5** vs extreme clicks
**7.0 s at 97%**; Open Images §2.4.3 *"verifying a single box took 3.5 seconds on average ...
we measured an average time of 8.5 seconds per box produced."*

| baseline | / 1.6 s (one question) | / 4.5 s (whole pipeline) |
|---|---|---|
| 88.0 s (Su mean total) | 55.0x | 19.6x |
| 42.4 s (Su median total) | 26.5x | 9.4x |
| 34.5 s ("35 s") | **21.6x** | **7.7x** |
| 26.5 s (pro end-to-end, derived) | 16.6x | 5.9x |
| 7.4 s (pro extreme click, measured) | 4.6x | 1.6x |
| 5.9 s (3.7+2.2 pro gesture) | **3.7x** | 1.31x |
| 3.7 s (pro draw gesture) | **2.31x** | **0.82x = 22% SLOWER** |
All ratios `[DER]`. The claim's 2.3-3.7x is exactly 3.7/1.6 and 5.9/1.6.
- **"10-20x" is only reachable by dividing ONE yes/no question by the AMT 3-stage pipeline.**
  Even the CVPR'16 authors claimed only **6x-9x**.
- **Google's own like-for-like professional comparison kills it**: verification **8.5 s/box produced**
  vs extreme clicking **7.4 s/box** = 0.87x, i.e. verification was ~15% SLOWER. Verbatim:
  *"Since extreme clicking takes about the same annotation time, but it is easier to deploy and
  delivers more accurate boxes ... we used it to annotate all remaining boxes (i.e. 13.1 million
  in the training set)."* `[SRC]`
- Deployed, measured assisted-annotation gains on real annotators: **1.49x** (Snapper, 33% task-time
  reduction) and **1.54x** (VLM pre-annotation on video, 35%, arXiv 2510.21798 — but 18 **volunteers**,
  not professionals, and no per-frame seconds). `[SRC]`

### Extreme clicking 44% SLOWER — CONFIRMED VERBATIM, with two caveats
Snapper §5.3.1 verbatim `[SRC]`: *"Aggregating SN-GT and SN-DM to represent the Snapper annotation
mode, we find that using Snapper led to a 33% reduction in time per task as compared to the BL,
while using **EC led to an increase of 44% in time per task** as compared to the BL."*
(BL 81.7 s vs EC 117.4 s.) §5.3.2: *"compared to the BL mode, **EC led to an increase in annotation
creation time by 87%**"* (3.7 -> 6.9 s), with footnote *"This replicates the finding for average
creation time in Papadopoulos et al."*
- **CAVEAT 1: the 44% task-time difference is NOT statistically significant** — *"significant
  differences in total job time exist between the BL and SN conditions ... and between the EC and SN
  conditions ... but not between the BL and EC conditions"* (EC sigma = 321.4 s). `[SRC]`
- **CAVEAT 2: the 87% creation-time difference IS significant** (p<0.001, r=1.0). `[SRC]`
  => **Quote the 87% creation-time reversal, not the 44%.**
- My own independent read of the EC paper (arXiv 1708.02750) `[SRC]`: extreme clicking **7.0 s**
  (VOC07) / 7.2 s (VOC12); baseline **34.5 s = 25.5 + 9.0, CITED from Su et al., NOT measured on
  their own annotators**; population = **AMT crowdworkers** paid *"$0.15 to annotate a batch of 10
  images ... about $7.7 per hour"*; claimed **"5x faster"**; quality 88% mIoU / 94% IoU>0.7 /
  **66 mAP with Fast R-CNN VGG16, identical to GT boxes**; **no mention anywhere of being slower
  for any population.** So the reversal is a genuine third-party refutation, not a paper self-contradiction.

### Commercial pricing, checked 2026-10-04 `[SRC]`
- **Google Cloud Vertex AI** (live, per 1,000 units per human labeler; tier1 = first 50k/mo):
  Image **bounding box $63 / $49**; rotated box $86/$60; image classification $35/$25;
  **segmentation $870/$850**; polygon/polyline $257/$180; **video object tracking bounding box
  $86/$60**; active learning data item $80/$56. Unit rule: *"if an image with 2 bounding boxes and
  3 human labelers counts for 2 * 3 = 6 units."* => **$0.063/box** at one labeler `[DER]`.
- **Roboflow** (live): *"Starting at $0.10/bounding box"*, $0.20/polygon, $0.05/classification.
- **Label Your Data** (live): *"our bounding box annotation starts at $0.02 per object"*;
  *"typically $6 per annotator hour"* => ~300 boxes/hr implied (12 s/box) `[DER]`.
- **AWS SageMaker Ground Truth**: live pricing page now redirects to "Request a demo" with no prices.
  Archived (2024-11-22) worked example used $0.08/article, $0.04/post, $0.036/post x 3 labelers;
  **Ground Truth Plus is quote-only and has never published a $/box.**
- **No published $/box found (all quote-only, checked 2026-10-04)**: Scale AI, Labelbox
  (pricing page 404; pivoted to agents), SuperAnnotate, Encord, V7, CloudFactory, Appen,
  Hive (now model-API pricing only).
- Throughput `[DER]`: 7.4 s/box -> **486 boxes/hr**; 26.5 s/box pro end-to-end -> **136 boxes/hr**;
  42.4 s -> 85/hr; 88.0 s -> 41/hr. Open Images qualification gate <=20 s/box -> >=180 boxes/hr `[SRC]`.
  **No vendor publishes boxes/hour; no primary source states professional boxes/hour directly.**
- Telling cross-check `[DER]`: Google's $0.063/box at the EC paper's measured $7.7/hr crowd wage
  implies **~29.5 s of paid labour per box** — today's official cloud price is still set as if a box
  takes ~30 s, not 4 s.
