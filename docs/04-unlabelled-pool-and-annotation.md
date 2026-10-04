# Exploiting the unlabelled pool, and where to spend annotation effort

> **Provenance and status.** Primary research output, 2026-10-04. Investigates semi-supervised
> object detection that bootstraps from zero box labels, using the clip's multi-label set as a
> constraint on pseudo-boxes, self-training schedules and their failure modes, whether active
> learning beats random selection at a tiny budget, and what public laboratory data actually exists.
>
> **This document has not yet completed adversarial verification.** The author reports 304
> source-backed claims, 70 flagged estimates, and 15 explicitly unverified items. Section 7.3 lists
> what could not be found, and section 7.4 lists internal inconsistencies in the sources themselves.
> Where this document and `00-executive-summary.md` disagree, the summary reflects later checking.
>
> Two findings here were significant enough to reshape the plan and are carried into the summary:
> roughly 22,000 public CC BY 4.0 box-annotated laboratory instances exist, and the LVIS and
> Objects365 category audits revealed that superficially matching class names are domestic homonyms.
>
> **SUPERSEDED IN PART.** [`05-annotation-budget-verification.md`](05-annotation-budget-verification.md)
> overturns this document's annotation guidance. In particular the one-box-per-clip claim was traced
> to an arithmetic artefact, the recommended gold-set size is now 450 frames across all 150 clips,
> and several annotation-time and active-learning figures were misframed. Read document 05 before
> acting on anything in this file about how to spend annotation effort.
>
> Known conflict: this document recommends roughly 250 to 300 annotated frames while
> `03-tracking-and-open-vocabulary.md` recommends roughly 600. A dedicated power analysis is
> resolving it. Do not act on either number yet.

---

# Squeezing value out of the unlabelled pool — lab-equipment detection from 150 video-level labels

Research date: 2026-10-04. Every number is tagged:
- `[SRC]` measured in a cited source
- `[VENDOR]` vendor/author claim not independently reproduced
- `[EST]` my own estimate/derivation
- `[UNVERIFIED]` I could not confirm it — treat as low confidence

---

## 1. Headline recommendation

**Do not build an SSOD system. Build a supervised warm-start on free public boxes, then a constrained auto-labelling loop around a frozen foundation teacher, and distil into a small student.**

Two framing corrections first, both evidence-driven:

1. **You are not actually at zero boxes.** I found **~22,000 CC BY 4.0 box-annotated lab-apparatus instances across four public datasets** (§5.6) covering beaker, flask, test tube, pipette, measuring cylinder, funnel, burette, balance and more — two of them *video-derived*. Use them. The "zero boxes" premise holds only for `centrifuge`, `autoclave`, `orbital shaker`, `spectrophotometer` and real laboratory `fume hood`, which have **zero public boxes anywhere** `[SRC]`.
2. **Every SSOD method in the canon (STAC → Sparse Semi-DETR) assumes a box-labelled seed set, and the transformer ones collapse below ~10 boxes/class** — Semi-DETR scores **1.00 mAP on VOC and 0.80 on a domain-specific set at 1 shot/class** `[SRC]` despite being SOTA at 1% of COCO. So "pick the best SSOD method" is actively the wrong move here (§5.1b).

**The loop I would build:**

**(a) Warm-start, supervised and free.** Fine-tune RF-DETR-small (Apache-2.0, 30.5M params) on the ~22k public CC-BY lab boxes, plus synthetic renders from Objaverse-LVIS (**47 microscope meshes vs 21 real microscope boxes in LVIS+VG+V3Det combined** `[SRC]`) for the five classes with no public boxes. **(b) Dual frozen teacher:** SAM 3 in promptable-concept-segmentation mode *and* MM-Grounding-DINO-B `pretrain_all` — the latter specifically because its V3Det/OpenImages training mix is the only released one containing `microscope`, `beaker`, `test tube`, `syringe`, `funnel`, `thermometer` as named *box* categories `[SRC]`. Prompt both **per clip with only that clip's label set plus Objects365-grounded hard negatives**. **(c)** Propagate every accepted detection through the clip with **SAM 2.1-small** (Apache-2.0, 46M, SA-V J&F 76.6 `[SRC]`) to mint boxes on every frame, not just the seed frame. **(d)** Filter by presence/absence/count reconciliation against the video-level multi-label plus a temporal-persistence gate (§2). **(e)** Distil into RF-DETR + ByteTrack for deployment. **(f)** Train a clip-level multi-label classifier on the 150 clips and use *its predictions* to supply the per-clip vocabulary constraint across the whole unlabelled pool — this is how 150 video-level labels get leveraged over 5,000 clips instead of 150. Three rounds, monitored by a label-free signal the video-level labels hand you for free: **the pseudo-label class histogram must match the clip-label class histogram.**

**The free lunch is not "SSOD on unlabelled video".** It is three things, in descending order of value:
1. **Per-clip vocabulary restriction** turns open-vocabulary detection's hardest problem — naming — into a 3-to-8-way closed-set problem. It is also a **2–3× compute saving**, because SAM 3's cost is linear in the number of prompted concepts `[SRC]`.
2. **Verified absence**, which no amount of unlabelled video can give you, and which is worth more than any scoring trick (SAM 3's own ablation: hard negatives move image-level MCC **0.44 → 0.68** `[SRC]`).
3. **Tracking propagation** turns one good detection into many training boxes — but see the correction immediately below, because the brief over-rates this and the evidence is clear.

### 1b. 🔴 Two corrections to the brief's premises, both evidence-backed

**Correction 1 — tracking-based label propagation is a garnish, not the main course.** The brief calls it "high leverage". The one paper that ablates it cleanly says otherwise. RoyChowdhury et al., *Automatic adaptation of object detectors to new domains using self-training*, CVPR 2019, BDD pedestrian AP `[SRC]`:

| Variant | AP | Δ vs baseline | Δ vs detector-only |
|---|---:|---:|---:|
| Source-only baseline | 15.21 | — | — |
| **Detector pseudo-labels only** | **26.16** | **+10.95** | — |
| Tracker pseudo-labels only | 26.28 | +11.07 | +0.12 |
| **Detector + tracker-mined hard positives** | 27.11 | +11.90 | **+0.95** |
| + consistency constraint | 28.43 | +13.22 | +2.27 |
| **Soft/smoothed labels (λ=0.3)** | **28.59** | **+13.38** | **+2.43** |

And on WIDER→CS6 faces, **tracker-only pseudo-labels were net negative: 15.66 → 11.73 AP, worse than doing nothing** `[SRC]`. The authors' explanation, verbatim: *"Using only samples from the tracker and ignoring all pseudo-labels from the detector… brings down the performance… we may miss a lot of actual faces in an image if we choose to train only on faces picked up by tracking alone."*

So: **tracking-mined boxes are worth ~+1 AP of a ~+12 AP total; soft labels on detector boxes are worth more; and tracking used alone can actively hurt.** Keep propagation — but demote it, never let it be the sole source of a training box, and always pair it with the hard-negative channel (see the P-N learning criterion in §3).

There is a formal theory for exactly when propagation blows up, and it is 2010: **Kalal, Mikolajczyk & Matas, *Tracking-Learning-Detection*, IEEE TPAMI** `[SRC]`. With `P⁺,R⁺` the precision/recall of the tracker-as-positive-expert and `P⁻,R⁻` of the negative expert, the false-positive/false-negative counts evolve as a linear system whose transition matrix must have **both eigenvalues < 1** for errors to cancel. Their measured sequences make the consequence concrete: the `Panda` sequence had P-expert precision **0.31 → λ₂ = 0.99 → final F-score only 0.25**, whereas `Car` had P⁺ = 1.00 → λ = 0.48/0.54 → F-score 0.90. **This is both a design rule (you must have an N-expert) and a monitoring signal (§3).**

**Correction 2 — the brief's "no boxes, and we're not labelling more" framing costs you more than every technique in this report combined.** Chéron et al., *A flexible model for training action localization with varying levels of supervision*, NeurIPS 2018, is a clean single-model supervision ladder `[SRC]`:

| Supervision | UCF101-24 mAP@0.2 | mAP@0.5 | % of fully supervised @0.2 |
|---|---:|---:|---:|
| **Video-level labels only** | **43.9** | **17.7** | **58%** |
| Temporal point | 45.5 | 18.7 | 60% |
| Temporal bounds | 47.3 | 20.1 | 62% |
| **1 bounding box** | **66.8** | 36.9 | **88%** |
| Temporal + 1 box | 70.6 | 38.6 | 93% |
| **Temporal + 3 boxes** | **74.5** | 43.2 | **98%** |
| Fully supervised | 76.0 | 50.1 | 100% |

Authors verbatim: *"using only one bounding box already enables good performance (66.8%)"*; temporal+3 keys *"almost matches accuracy of the fully-supervised setup (−1.5%)"*.

**One box per clip moves you from 58% to 88% of fully supervised. For 150 clips that is 150 boxes — a couple of hours of work.** No technique in this report comes close to that return. §4 makes this the headline recommendation on labelling.

---

## 2. The constrained pseudo-labelling design (precise spec)

### 2.0 Notation
- Clip `c`, with human-given positive label set `P(c)` (e.g. `{microscope, pipette}`).
- Global class list `K` (open-ended, growing).
- Implied negative set `N(c) = K \ P(c)` — "strong evidence of absence", *not* a guarantee.
- `H` = global hand-written hard-negative noun-phrase list (confusables that are **not** target classes): `monitor, desktop computer, printer, 3D printer, coffee machine, water cooler, dishwasher, oven, microwave, toaster, cardboard box, filing cabinet, office chair, fire extinguisher, first aid box, paper towel dispenser, sink, kettle`.

### 2.1 Why a per-clip vocabulary is the highest-value filter

Three independent `[SRC]` results say the same thing — telling a detector which classes are present, and which confusables are not, is worth more than any scoring trick:

| Evidence | Measured effect | Source |
|---|---|---|
| SAM 3 **hard negatives in the prompt set**: 0 → 30 hard negatives per image | image-level classification MCC (IL_MCC) **0.44 → 0.68** | SAM 3 paper Table 8(b) `[SRC]` |
| SAM 3 **presence head** (decouples "is this concept here at all" from "where") | **+1.5 cgF1** (50.7 → 52.2); IL_MCC 0.77 → 0.82 | SAM 3 paper Table 8(a) `[SRC]` |
| DECOLA **conditioning the detector on the set of present classes** before pseudo-labelling | **+13.4 to +17.2 c-AP@20** (R50); +14.5 to +15.9 (Swin-B) vs unconditioned | DECOLA Fig. 6 `[SRC]` |
| Omni-DETR using **image-level tags** as the weak annotation on 90% of COCO | **34.7 vs 28.0 mAP** supervised-only (+6.7) at 10% boxes | Omni-DETR Table 2 `[SRC]` |

DECOLA had to *train* a language-conditioned detector to get class conditioning. This project gets it for free at inference time by simply not querying classes it knows are absent. That is the single most important design decision in the pipeline.

### 2.2 The accept/reject cascade

Run per clip. Each stage is ordered cheapest-and-most-decisive first.

**Stage A — vocabulary gate (hard).**
Query the teacher with noun phrases for `P(c) ∪ H` **only**. Never query `N(c)`.
- Any detection whose assigned class ∉ `P(c)` → **reject as a pseudo-box**, but **retain as a hard-negative region** for the student (see Stage D-absence).
- Effect `[EST]`: removes the dominant open-vocab failure mode (confident box, wrong name). On an 8-class problem with ~20 plausible confusables this should cut false-positive *class* assignments by roughly an order of magnitude; I have no direct measurement for lab equipment, so treat the magnitude as `[EST]` and the direction as `[SRC]` (the three rows above).

**Stage B — score gate (adaptive, per class, per round).**
Do **not** use one global threshold. Two signals, both gated:
1. **Presence score** (SAM 3's presence token) ≥ `τ_pres`. Start `τ_pres = 0.5`. This answers "is class k in this frame at all".
2. **Per-query localisation score** ≥ `τ_k`, where `τ_k` is set by fitting a **two-component Gaussian mixture to the per-class score histogram over the whole clip set and keeping the high-mean component** — this is exactly Consistent-Teacher's GMM adaptive-threshold module, which is part of the configuration that reaches 25.50 / 36.60 / 40.20 mAP at 1/5/10% COCO versus fixed-threshold predecessors around 20–22 / 30–33 / 34–37 `[SRC]`.
- Fixed-threshold fallback if the GMM is unstable (it will be for rare classes with <50 detections): `τ_k = 0.35` for SAM 3 PCS scores `[EST]`, then tune on the gold set.

**🔴 Two measured findings that must override your instinct to filter hard. Read these before setting any threshold.**

**(i) Do not tune the threshold against pseudo-label precision. Tune it against downstream mAP. They are not monotonically related.** VL-PLM (ECCV 2022) measured both `[SRC]`:

| τ | pseudo-label AP (AP@PL) | boxes/img | Downstream Novel AP | Downstream Overall AP |
|---:|---:|---:|---:|---:|
| **0.05** | **20.6 (highest)** | 85.15 | **19.3 (worst)** | 27.0 |
| **0.95** | 18.0 | 2.93 | **31.3 (best)** | **49.1** |
| 0.99 | 11.1 | 1.62 | 27.2 | 49.0 |

The configuration with the **highest** pseudo-label AP produced the **worst** downstream novel-class AP. This is the single most important caveat for this entire plan: **a gold set measuring pseudo-label quality is not a substitute for a gold set measuring the student's detection mAP.** Budget for the latter.

**(ii) When the unlabelled pool is large, moderate thresholds beat strict ones — recall matters more than precision.** OWLv2 / OWL-ST (NeurIPS 2023) swept the pseudo-annotation threshold over 10B WebLI images and found **0.3 beat both 0.5 and 0.7** `[SRC]`, with retained image counts 0.1≈5B, **0.3≈2B**, 0.5≈782M, 0.7≈224M. Their conclusion verbatim: *"Moderate thresholds (0.3) outperformed strict filtering, contradicting prior work favoring high-confidence-only approaches."* Their data scaling was roughly **logarithmic**: 100M ≈ 38 AP_rare → 500M ≈ 41 → 2B = 44.6.

**How to reconcile (i), (ii) and the precision-first cascade above.** They are not actually in conflict; they apply at different pool sizes:
- **On the 150 labelled clips (R1), be precision-first.** The pool is small, every bad box is a large fraction of your training set, and the clip-level constraints are *exact*. Use the GMM thresholds and the full cascade.
- **On the large unlabelled pool (R2/R3), loosen.** With ~5,000 clips the OWL-ST regime applies: drop `τ_k` by roughly 0.1–0.15 `[EST]` and let volume do the work, because the student can average out noise it sees once but not a systematic gap it never sees at all.
- **In both cases, sweep the threshold on the student's gold-set mAP at least once.** Three values is enough to find the shape. This is the one hyperparameter where guessing is measurably dangerous.

Also note the field-wide reproduction noise: at 1% COCO the SSOD literature sits in a **±3 AP band** (Dense Teacher is 19.64 in its own paper and 22.38 in every paper citing it; the supervised 1% baseline is variously 9.05, 10.02 and 11.24) `[SRC]`. **Do not over-read deltas smaller than ~3 AP in anything you read, or in your own low-label experiments.**

**Stage B2 — per-class size-prior gate (cheapest high-value filter in the whole cascade).**
Reject any box whose area or aspect ratio falls outside a hand-written per-class range. This is three lines of code and the weakly-supervised literature says it is worth more than any attention mechanism: C-WSL's oracle probe showed that **removing proposals smaller than the smallest true object lifted person AP from 20.3 → 40.0 and overall VOC07 mAP from 45.7 → 52.7** `[SRC]`, i.e. **a size prior alone recovers roughly 35% of the entire weak-to-full supervision gap** `[calc]`, whereas published part-domination fixes recover only +2.6 to +3.8 mAP `[SRC]`.

Lab equipment has unusually strong, knowable size priors. Starting values, all `[EST]`, to be calibrated on the gold set:

| Class | min box height (frac. of frame) | max | aspect ratio w/h |
|---|---|---|---|
| fume hood | 0.25 | 1.0 | 0.8–3.0 |
| autoclave | 0.20 | 0.9 | 0.5–1.5 |
| centrifuge | 0.10 | 0.6 | 0.7–1.6 |
| microscope | 0.10 | 0.8 | 0.4–1.2 |
| spectrophotometer / plate reader | 0.08 | 0.5 | 0.8–2.2 |
| orbital shaker | 0.08 | 0.5 | 0.8–2.5 |
| analytical balance | 0.06 | 0.4 | 0.8–2.0 |
| glassware (beaker/flask) | 0.03 | 0.4 | 0.4–1.5 |
| pipette | 0.03 | 0.35 | 0.1–0.8 |

This single gate directly attacks the documented WSOD failure where the detector latches onto a discriminative *part* — a screen, a knob, an eyepiece — instead of the whole instrument. In this domain that failure is near-certain without it: every one of these machines is a box with a panel on the front.

**Stage C — temporal persistence gate (the lab-specific win).**
Track every Stage-B survivor forward and backward with the teacher's tracker (SAM 3 PVS mode; SAM 2.1 is the fallback). Accept a tracklet iff all hold:
- persists ≥ `L` frames, with **class-dependent `L`**:
  - **static equipment** (microscope, centrifuge, fume hood, autoclave, balance, shaker, spectrophotometer): `L = 0.5 s` (≈12–15 frames at 25–30 fps) `[EST]`, and additionally **box-motion-after-camera-compensation below 2% of frame width per frame** `[EST]`. A fume hood that moves is not a fume hood.
  - **handled items** (pipette, glassware): `L = 0.25 s` (≈6–8 frames) `[EST]`, no motion constraint.
- mean IoU between the tracker's propagated box and the detector's independent re-detection ≥ **0.5** on frames where the detector fires `[EST]`
- the detector independently re-fires on ≥ **40%** of tracklet frames `[EST]`

Then **mint boxes on every frame of the accepted tracklet**, including frames where the detector found nothing. This is the volume multiplier: 150 clips × ~300 frames ≈ **45k labelled frames** from a few hundred seed detections `[EST]`.

Evidence this works: PseudoProp (CVPR-WAD 2022) propagates pseudo-labels bidirectionally using motion continuity and reports **+7.4% mAP75** over the then-SOTA SSOD baseline on Cityscapes `[SRC]`. The "baked goods" paper (arXiv 2602.09979, Feb 2026) — nearly this project's exact setting, image-level labels only, OWLv2 + Grounding DINO localisation, **SAM 2 as the pseudo-label propagation model**, YOLOv11 student — reports **+19.3% improvement under non-ideal deployment conditions** `[SRC]`. POPCat (arXiv 2406.17183) turns **149 ground-truth annotations into labels over 40,947 frames in 115 videos (≈1:274 leverage)** via point-tracker propagation + SAM, reporting mAP50 gains of +9.6% (GMOT-40), +43.1% (AnimalTrack), +9.4% (VisDrone) over prior best `[SRC]`. And the closest published analogue to this whole plan — **Autonomous Temporal Pseudo-Labeling for Fish Detection** (*Applied Sciences* 12(12):5910, 2022) — bootstraps YOLOv4 on a *foreign, noisy* source dataset, runs it over 21 unlabelled videos (695 min), associates boxes **forwards and backwards** to fill missing annotations, builds 35,510 images / 113,886 boxes with **zero target-domain human boxes**, and reports **93.11% mAP on in-domain and 73.61% on unseen data, +24.65 and +25.53 over the noisy baseline** `[SRC]`.

**🏆 But the strongest measured temporal result in this project's exact protocol is temporal consistency used as a *filter*, not as a propagator.** SSVOD (WACV 2024, arXiv 2309.01391) uses **flow-warped predictions from neighbouring frames plus cross-IoU and cross-divergence selection** to decide which pseudo-boxes to keep. ImageNet-VID mAP@0.5, where "1 kf / x% vids" means **one labelled key frame per video in x% of training videos** (well under 1% of frames labelled in every column) `[SRC]`:

| | 1 kf, 25% vids | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| Supervised image (Faster R-CNN) | 30.1 | 37.6 | 42.5 | 47.8 |
| STAC | 37.2 | 44.2 | 46.6 | 50.9 |
| Unbiased Teacher | 39.1 | 44.6 | 48.3 | 53.4 |
| Soft Teacher | 40.2 | 46.7 | 49.7 | 54.7 |
| Supervised video (SELSA) | 39.6 | 43.1 | 48.9 | 55.0 |
| Watch & Learn (Misra et al.) | 42.9 | 48.5 | 53.1 | 57.7 |
| **SSVOD** | **48.6** | **53.4** | **60.5** | **63.8** |

That is **+9.0 mAP over the SELSA video baseline and +8.4 over Soft Teacher at one labelled key frame per video** `[SRC]`, and it is detector-agnostic (FGFA +8.5, SELSA +9.2, TROI +8.7). Cross-dataset: EPIC-KITCHENS 30.4 → 36.7; YouTube-VIS 45.3 → 53.6. **Swapping FlowNet → FlowFormer alone was worth another +4.6 mAP (63.8 → 68.4)** — i.e. flow quality matters materially. ⚠️ **The code is unavailable: the repo listed on Scholar (`github.com/enyacgroup/SSVOD`) returns HTTP 404 as of 2026-10-04** `[SRC]`. You will have to reimplement the cross-IoU/cross-divergence selection, which is a modest amount of work and, on this evidence, the single best-value temporal component available.

**Mine hard negatives from temporal isolation too — not just positives.** Jin, RoyChowdhury et al. (ECCV 2018) threshold detections at 0.8, build a tracklet by NCC template matching within ±5 frames (similarity 0.5), and flag a detection as **"isolated"** — hence a likely hard negative — if `max IoU(tracklet prediction, detections in adjacent frames) < 0.2`. Measured purity on manual audit: **hard positives 94.46% (faces) / 83.13% (pedestrians); hard negatives 88.65% / 74.48% precision** `[SRC]`. Gains: WIDER-easy AP .907 → .921, **Caltech Pedestrian log-average miss rate 23.83 → 18.72**, COCO dog 26.9 → 28.1, COCO train 33.9 → 35.4 `[SRC]`. **The hard-negative channel was as valuable as the positive one, and almost everyone implements only the positive half.** It is also exactly the N-expert that P-N learning theory (§1b) says you need to keep the error eigenvalues below 1.

**Stage D — clip-level presence/absence/count reconciliation (posterior regularisation).**

*Presence (hard constraint, projection not rejection).* Every class in `P(c)` **must** receive ≥1 accepted tracklet. If a class in `P(c)` has zero survivors, do **not** discard the label — **lower `τ_k` for that clip alone until exactly one tracklet survives** ("forced positive"), and mark it:
- down-weight its classification loss ×0.3 `[EST]`
- **exclude it from the box-regression loss entirely** `[EST]` (its geometry is untrustworthy; see §2.3)
- log it: the **forced-positive rate is a primary degradation monitor** (§3).

*Absence (the asymmetric payoff).* For every `k ∈ N(c)`, the clip is a **verified negative** for `k`. Any Stage-A-rejected detection of class `k` in clip `c` becomes a **hard-negative region** added to the student's background with loss weight ×2 `[EST]`. This is information you can never obtain from pseudo-labels alone, and it is exactly the signal SAM 3's own ablation shows is worth IL_MCC 0.44 → 0.68 `[SRC]`.
- **Caveat, stated plainly:** "not labelled" ≠ "absent". A clip annotator who wrote `{microscope, pipette}` may simply not have mentioned the balance on the bench. This is the single-positive / missing-label problem (Cole et al., *Multi-Label Learning from Single Positive Labels*, CVPR 2021 `[SRC]` — exists and is the right reference; I did not extract its exact mAP deltas, so the size of the penalty is `[UNVERIFIED]`). Mitigation: treat absence as a **soft** negative with weight ×2 only for classes the clip-level classifier *also* scores low; for classes the classifier scores high but the human did not list, do nothing (neither positive nor negative) — an ignore region.

*Count (if you can get it).* Counts are extremely cheap supervision: C-WSL (ECCV 2018) measured per-class counts as **>2× cheaper than centre-clicking and >38× cheaper than bounding boxes** `[SRC]`, and found counts give "large improvements in WSL". If you collect counts, cap accepted tracklets per class at the stated count, keeping highest-scoring. Even without human counts, SAM 3's own counting is good enough to use as a prior: **CountBench MAE 0.12 / 93.8% accuracy; PixMo-Count MAE 0.21 / 86.2%** `[SRC]`.

*The machinery.* Presence/absence/count are inequality constraints on the output distribution. The clean way to impose them differentiably is the **log-barrier extension of constrained-CNN losses** (Kervadec et al., *Constrained-CNN losses for weakly supervised segmentation*, Medical Image Analysis 2019, 293 citations `[SRC]`; and *Bounding boxes for weakly supervised segmentation: global constraints get close to full supervision*, MIDL 2020 `[SRC]`), which is SGD-compatible and avoids Lagrangian dual steps. Older lineage: CCNN (*Constrained Convolutional Neural Networks for Weakly Supervised Segmentation*, arXiv 1506.03648) and *Weakly Supervised Object Detection with Posterior Regularization* (BMVC 2014, 114 citations) `[SRC]`. **Opinion:** implement presence/absence as the simple projection + reweighting above first. Only reach for log-barrier constrained losses if the simple version plateaus — the papers are from segmentation and medical imaging, and the engineering cost is real.

**Stage E — cross-model agreement, as a soft weight not a hard gate.**
Second teacher, architecturally independent of SAM 3: **MM-Grounding-DINO-T** (Apache-2.0, zero-shot COCO 50.4, LVIS minival 41.4 `[SRC]`) or **OWLv2-B/16** (155M params, Apache-2.0 `[SRC]`). Decision rule:
- both fire, same class, IoU ≥ 0.5 → accept at **`τ_k − 0.15`** (relaxed) `[EST]`
- one fires only → require **`τ_k + 0.10`** (strict) `[EST]`
- **weight the resulting pseudo-box by a consensus factor rather than hard-gating it** (see below).

**Honesty flag, and it is a precise one.** Nobody has published pseudo-box precision/recall for *one versus two open-vocabulary detectors required to agree at IoU ≥ t*. That exact measurement does not exist as of 2026-10-04 `[UNVERIFIED]`. **This is roughly a half-day experiment on your own data and it would be a genuinely novel number — do it in R0 and let the result decide whether Stage E earns its compute.**

What *is* measured, and all of it points the same way `[SRC]`:

| Evidence | Comparison | Measured delta |
|---|---|---|
| **DetMatch** (ECCV'22) — the cleanest isolation | confidence thresholding **vs** adding cross-model Hungarian agreement | KITTI 1%: 45.9 → 54.4 (thresholding) → **57.9 (+3.5 on top of thresholding)** → 59.4; 1% Pedestrian 30.4 → 42.7 → **57.3**. Its Fig. 3: **matching cost correlates with GT IoU far better than classification confidence or predicted IoU** |
| **Cross Pseudo Supervision** (CVPR'21) | single-net pseudo-supervision **vs** two differently-initialised nets | VOC **1/16 labels: 59.54 → 68.21 (+8.67)**; 1/8 +4.15; 1/4 +1.69; **1/2 +0.74** |
| **Instant-Teaching★ co-rectify** (CVPR'21) | one model **vs** two rectifying each other | COCO **1% +2.05**, 2% +1.75, 5% +1.25, 10% +0.95, 100% +0.60 mAP |
| **CrossRectify** | self-labeling **vs** two same-arch different-init detectors | VOC AP50 +0.40 → **+1.83** → +3.24 with WBF. Diagnostic: at τ=0.8 **~9% of pseudo-boxes are still misclassified, and discarding just those is worth +2.30 vs +0.40 for plain self-labeling** |
| **Robust & Label-Efficient Deep Waste Detection** (arXiv 2508.18799) — closest to the question | 1 detector **vs** 4-detector consensus → downstream student mAP | hard labels Swin-T 47.7 → **48.6 (+0.9)**; Swin-B 52.5 → **53.8 (+1.3)**; **consensus-weighted *soft* labels 49.3 / 54.3 (+1.6 / +1.8)**. Consensus factor `cf = exp(−α·spread)·[1 + β(|C_models|−2)]`, **α=5.0, β=0.1**. ⚠️ members were fine-tuned, not zero-shot; no ensemble-size ablation |
| **CMPL** (CVPR'22) — video | same-size second model **vs** a deliberately 4× smaller, structurally different one | K400 1%: self-labeling 6.78 → same-size 9.97 → **4× smaller & different 12.90 (+2.93)** |
| **Deep Co-Training** (ECCV'18) | 2 / 4 / 8 views | CIFAR-10@4k error 9.03 → 8.54 → 8.35. **~85–90% of the gain arrives with the second view** — a third model is not worth it |
| **JoCoR vs Co-teaching+** (CVPR'20) | agreement-maximising **vs** disagreement-first | CIFAR-10 Sym-50%: 79.41 vs 57.05; Clothing1M 70.30 vs **59.32 (collapses)**. **Agreement-maximisation wins; disagreement-first is unstable** |
| **Weighted Boxes Fusion** (*IVC* 2021, MIT, `pip install ensemble-boxes`) | NMS **vs** coordinate fusion | COCO 2 models: 0.521 → NMS 0.5269 → **WBF 0.5344**. **Architecturally mixed** OpenImages ensemble: 0.5164 → 0.5642 → **0.5982**. The WBF-over-NMS margin is **4.5× larger for diverse ensembles** |

**Three design consequences, all evidence-driven:**
1. **The agreement gain is largest exactly where this project sits** — CPS +8.67 at 1/16 labels versus +0.74 at 1/2; Instant-Teaching +2.05 at 1% versus +0.60 at 100%. Stage E is *more* worth it here than in any paper.
2. **Use a soft consensus weight, not a hard AND.** The waste-detection paper measured consensus-weighted soft labels at **+1.6/+1.8 versus +0.9/+1.3 for hard consensus labels** `[SRC]`. Adopt their factor: `cf = exp(−5.0 · box_spread) · [1 + 0.1(n_models − 2)]`, and multiply the pseudo-box's classification loss by it.
3. **Pick models that are architecturally and pre-training-wise *different*, and prefer two over three.** SAM 3 (Perception Encoder + DETR) and MM-GDINO (Swin + BERT grounding) or OWLv2 (ViT + CLIP/SigLIP) are maximally independent. Diversity is the measured active ingredient (WBF's 4.5× margin; CMPL's smaller-and-different model beating a same-size one by +2.93). Deep Co-Training needed adversarial perturbation specifically to *stop* two same-architecture nets collapsing into each other.

**🔑 And the finding that should shape what the second model is asked to do.** MarvelOVD (ECCV 2024) measured that **CLIP's misclassification rate on valid objects is only 3.3%, but it fails to identify noisy/non-object boxes 76.6% of the time** `[SRC]`. A VLM is a good *classifier* and a terrible *objectness verifier*. Corollary: **two detectors agreeing on a BOX is informative in a way two VLMs agreeing on a LABEL is not.** Ask the second model for localisation/objectness evidence, not just class confirmation.

Corroborating this, the **Mcity Data Engine** (IEEE ITSC 2025, arXiv 2504.21614, MIT) is the only public project that builds exactly this IoU-agreement filter (3-of-5 majority over OWLv2-B, OWLv2-L-E, OmDet-Turbo-T, Grounding-DINO-T, OWL-ViT-L). It publishes no ensemble precision/recall row — but its per-checkpoint spread is the argument for consensus all by itself `[SRC]`: Grounding-DINO-B precision **0.6774** / recall **0.0166**; OWLv2L-E precision **0.1804** / recall **0.4388**; OmDet-Turbo-T 0.0082 / 0.3142. **A ~22× precision range and ~26× recall range across checkpoints of the same two families. No single checkpoint is simultaneously precise and complete.**

### 2.3 Classification vs box-regression asymmetry — resolved, with the citable number

The brief asks which SSOD papers restrict pseudo-labels to classification, and why. The answer is precise: **only Unbiased Teacher v1 zeroes box regression.** Verbatim `[SRC]`:

> *"we do not apply unsupervised losses for the bounding box regression since the naive confidence thresholding is not able to filter the pseudo-labels that are potentially incorrect for bounding box regression (because the confidence of predicted bounding boxes only indicate the confidence of predicted object categories instead of the quality of bounding box locations)"*

Its code zeroes **both** RPN localisation and ROI box regression:
```python
if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
    loss_dict[key] = record_dict[key] * 0        # pseudo bbox regression <- 0
elif key[-6:] == "pseudo":
    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
```

**The decisive measurement is UT v2's Table 6** — AP at increasing IoU strictness versus a no-unsupervised-regression baseline `[SRC]`:

| | AP55 | AP65 | AP75 | AP85 | AP95 |
|---|---:|---:|---:|---:|---:|
| No unsupervised regression | 29.71 | 24.64 | 17.55 | 8.33 | 0.35 |
| **Confidence-thresholded regression** | 30.60 (+0.89) | 25.07 (+0.43) | 17.96 (+0.41) | **8.22 (−0.11)** | **0.32 (−0.03)** |
| **"Listen2Student"** (relative uncertainty) | 30.78 (+1.07) | 26.19 (+1.56) | 19.64 (+2.09) | **10.47 (+2.14)** | 0.58 (+0.23) |

Paper verbatim: *"although the confidence thresholding can improve the easier evaluation metrics (e.g., AP55), it cannot improve or even degrades the results on stricter evaluation metrics (e.g., AP95)."* Corroborating: **SAS-Det (CVPR 2024) measured −3.6 AP50 from letting noisy pseudo-boxes into the box-regression loss** `[SRC]`. Note also Dense Teacher's contrary data point on a *dense* head: at 10% COCO, cls-only 33.34 → **cls+reg 35.11 (+1.8)** `[SRC]` — so on one-stage dense detectors regression pseudo-labels do pay, provided the selection is dense rather than thresholded.

**Everyone after UT v1 brought regression back by replacing "classification score" with a localisation-specific gate.** Ranked by implementation cost here:

1. **🏆 Soft Teacher's box-jitter variance — cheapest, measured +0.6 mAP** (33.6 → 34.2 at 10% COCO) `[SRC]`. Pre-filter foreground score > 0.5 (*"reduces … from hundreds to around 17 per image"*). Jitter each survivor **N=10** times (paper says uniform ±6% of h/w; **the code uses Gaussian** `torch.randn(...) * (box_scale * 0.06)` — a real paper/code discrepancy), re-refine, compute per-coordinate std normalised by half-perimeter, average over 4 coords, **keep for regression only if < 0.02**. Maintains two disjoint sets: `G_cls` (score > 0.9) and `G_reg` (variance < 0.02). Motivation verbatim: *"the localization accuracy and the foreground score of the box candidates do not show a strong positive correlation."* Cost is ~10 extra ROI-head forwards per retained box — material on 12 GB, so apply it only to boxes that already survived Stages A–D.
2. **Consistent-Teacher's FAM-3D** — re-align the heads so the classification score *becomes* a valid localisation proxy, then use the full loss with no regression gate. Better, bigger change.
3. **UT v2's Listen2Student** — best numbers, needs an extra uncertainty head trained with NPLL.

**Recommendation:** pseudo-labels freely for classification; gate box regression on **Soft Teacher's jitter variance < 0.02**, and additionally exclude (a) any forced-positive from Stage D and (b) any box whose only support is tracker propagation. Three independent lines of evidence converge on that last exclusion: the regression asymmetry above, the P-N learning stability requirement (§1b), and RoyChowdhury's tracker-only result.

**And the rule that reconciles all the soft-vs-hard evidence: hard pseudo-*boxes*, soft pseudo-*classes*.** RoyChowdhury's best variant was label smoothing, not harder propagation — `ỹᵢ = λ·sᵢ + (1−λ)·yᵢ`, with **λ=0.3 (pedestrians) / 0.7 (faces)**, `θ = 0.5/0.8` for tracker-only boxes `[SRC]`. Meanwhile Noisy Student used soft labels at threshold 0.3, but the detection literature consistently found hard better (Caine et al.: hard ~0.5 beat soft; Zoph: hard 0.5; CPS: hard argmax beats soft consistency by ~2 mIoU) `[SRC]`. Soft on the class, hard on the geometry fits every one of those results.

---

## 3. Self-training schedule

Pool size assumption used throughout: **~5,000 unlabelled clips, ~30 s each at 25 fps, decoded at 2 fps → ~750k candidate frames.** Stated explicitly because every cost number below scales with it.

### Rounds

**R0 — measure, don't train (1–2 days).**
- Write the class list as **disambiguating** noun phrases, not bare class names — see the prompt table in §5.5, which is derived from the fact that every superficially-matching LVIS/Objects365 category is a domestic homonym. Build the hard-negative list `H` from real Objects365 class strings (§5.5b).
- **Label a gold evaluation set. This is non-negotiable and is the only thing standing between you and silent failure.** ~40 frames × ~8 boxes ≈ 300–400 boxes, from held-out clips, stratified so every class appears ≥10 times.
- Run SAM 3 **and** MM-Grounding-DINO-B zero-shot, each with and without per-clip vocabulary restriction. Four numbers. The vocabulary-restriction delta is your evidence that the free lunch is real; the two-model delta tells you whether Stage E agreement is worth its cost.
- **Audit the clip labels' recall** (§8.3 item 3): exhaustively re-annotate *presence* on 20 clips. If annotator recall < ~0.85 `[EST]`, turn absence-mining off.

**R0.5 — supervised warm-start on free public data (NEW, and it should come first).**
- Fine-tune RF-DETR-small on the **~22k CC BY 4.0 lab-apparatus boxes** of §5.6 (chemistry-25 + physics-27 + Sasaki + HeinSight4.0), mapping their label spaces onto yours. Attribute properly — CC BY requires it.
- For the five classes with **zero public boxes** (`centrifuge`, `autoclave`, `orbital shaker`, `spectrophotometer`, real `fume hood`): render synthetic data from the **CC-BY/CC0 subset of Objaverse** (§5.6) and/or use ImageNet-21k crops (`centrifuge` 514, `spectrophotometer` 628, `autoclave` 1,277, `analytical balance` 1,047 images `[SRC]`) as *classification* supervision for a cropped-region classifier that re-scores the teacher's boxes. **Note the ImageNet licence is non-commercial research/education only** — fine for ablations, not for a shipped model.
- Expect the public-data warm-start to transfer poorly on its own (those datasets are near-saturated and low-diversity — RF-DETR gets **mAP@50 0.992** on chemistry-25 `[SRC]`, which is a diversity warning, not a quality signal). Its value is as a **better-than-random initialisation and a better-than-nothing second opinion in Stage E**, not as a product.

**R1 — constrained auto-label the 150 labelled clips → student v1.**
- Full cascade §2.2 on 150 clips. Grows: boxes (0 → ~45k frames `[EST]`).
- Train RF-DETR-nano/small. Freeze nothing; batch 4 + grad-accum 4 + `gradient_checkpointing`.

**R1.5 — 🚨 filter the unlabelled pool for out-of-distribution content. Do not skip this.**
A "large pool of unlabelled video" from a real site will contain corridors, offices, car parks, meeting rooms and empty benches. The measured consequence of feeding that in unfiltered is that **the pool makes your detector worse than not using it at all**: Unbiased Teacher scored **37.2 AP on a clean pool, 34.5 AP on an OOD-contaminated pool — below its own 35.6 AP labelled-only baseline** — and a kNN-to-class-prototype filter recovered it to **42.7 AP at only 26% slower iterations** `[SRC]`. Corroborated on Semi-Aves: **53.8 → 45.7** with open-set unlabelled data `[SRC]`.
- Build a per-class feature bank from the 150 clips (frozen DINOv2 or the student's own backbone), compute kNN cosine distance from each pool frame to the class prototypes, and threshold adaptively at **mean + β·std** `[EST]`.
- **~1 day of work, and it is the highest-risk item in the whole plan that you can cheaply de-risk.** The brief does not mention it; it should.
- Order the surviving pool by difficulty (clip-classifier confidence) so that R2 consumes easy footage and R3 consumes hard footage — gradual self-training measured **87.9% vs 33.0%** for direct `[SRC]`.

**R2 — clip-level classifier → constrain the pool → student v2.**
- Train a **clip-level multi-label classifier** on the 150 clips (this is the correct use of video-level labels: clip-level multi-label recognition is a far easier problem than detection and 150 clips is enough for a frozen-backbone linear/attention head).
- Run it on all 5,000 pool clips. For clips where the predicted label set is confident, use the **predicted** `P̂(c)` as the Stage-A vocabulary. Only admit clips above a classifier-confidence threshold; start by admitting the top 40% `[EST]`.
- Grows: clips (150 → ~2,000), boxes (~45k → ~400k frames `[EST]`).

**R3 — ensemble teacher on the full pool → student v3.**
- Teacher = SAM 3 + student v2 (student v2 is now in-domain and will catch things SAM 3 misses; SAM 3 catches what the student misses). Keep Stage C and D.
- Grows: pool coverage to ~80%; add the hard-negative mining pass.

**R4+ — only if R3 improved the gold-set AP by >1 point.** Expect diminishing returns by R3. Zoph et al. (*Rethinking Pre-training and Self-training*, NeurIPS 2020) measured self-training gains of **+1.3 to +3.4 AP across all COCO dataset sizes** `[SRC]` — real but bounded, and that was with a much stronger starting point than this project has.

### Stopping criteria (concrete)

Stop when **any** of these fires:

0. **🏆 The P-N learning eigenvalue test — the only principled, label-free stopping criterion I found, and it is 2010.** Kalal et al.'s TLD gives the error dynamics of exactly this loop `[SRC]`. With `P⁺, R⁺` the precision/recall of your positive expert (the accepted-pseudo-box channel) and `P⁻, R⁻` of your negative expert (the Stage-D absence mining + temporal-isolation hard negatives), the false-positive/false-negative counts evolve as

   ```
   α(k+1) = (1 − R⁻)·α(k) + ((1 − P⁺)/P⁺)·R⁺·β(k)
   β(k+1) = ((1 − P⁻)/P⁻)·R⁻·α(k) + (1 − R⁺)·β(k)
   ```

   and *"converges to zero if both eigenvalues λ₁, λ₂ of the transition matrix M are smaller than one."* For the symmetric case `P⁺=R⁺=P⁻=R⁻=1−ε` this reduces to **stable iff ε < 0.5**. Their measured sequences make the stakes concrete: `Car` had P⁺ = 1.00 → λ = 0.48/0.54 → final F-score **0.90**; `Panda` had P⁺ = **0.31** → λ₂ = **0.99** → final F-score **0.25** `[SRC]`. Their own reading: *"The larger these eigenvalues are, the less the P-N learning improves the performance."*

   **Operationally:** every round, estimate `P⁺` on your gold set and `P⁻` by auditing ~50 sampled hard negatives. Compute the eigenvalues. **If max λ > 0.9, stop and fix the experts — do not run another round.** This single criterion would have caught every failure mode in §3b before it cost you a training run. It also tells you *which* lever to pull: a high λ driven by low `P⁺` means tighten the cascade; driven by low `R⁻` means you are not mining enough negatives.

1. **Gold-set AP50 improvement < 1.0 point** between rounds. Caveat: on ~350 gold boxes the AP standard error is large — I estimate ±2–3 AP `[EST]` — and the SSOD literature's own reproduction band at 1% COCO is **±3 AP** `[SRC]`. So a 1-point "improvement" is noise. Treat two consecutive sub-1-point rounds as a stop, and report confidence intervals, not point estimates. **And measure student mAP, not pseudo-label precision** — §2.2(i) shows they can move in opposite directions.
2. **Forced-positive rate rises** (Stage D). If the fraction of `P(c)` classes needing a threshold drop increases round over round, the model is losing classes — classic class collapse.
3. **Pseudo-box count per frame drifts >20%** in either direction `[EST]`. Up means false positives are compounding; down means collapse.
4. **Mean accepted-tracklet length falls.** Precision degradation shows up as flicker before it shows up in AP.
5. **Teacher–student disagreement rate rises.**

### 🏆 The label-free monitor you get for free — and it comes with measured calibration anchors

The 150 video-level labels give an **empirical class-prevalence distribution**. Pseudo-labels on the pool must produce a class distribution consistent with it. So:

> Monitor **KL(class prior from the 150 clip labels ‖ pseudo-label class histogram over the pool)** every round.

This needs no new ground truth, is computed over the whole pool, and fires early on exactly the two failure modes you fear most. **And Unbiased Teacher has already calibrated the scale for you** — its own ablation reports the pseudo-label-distribution KL alongside the resulting mAP `[SRC]`:

| Configuration | PL-distribution KL | 1% COCO mAP |
|---|---:|---:|
| Cross-entropy, no EMA (**broken**) | **1.7915** | 13.42 |
| Focal loss, no EMA | 0.2001 | 17.85 |
| **Focal loss + EMA (healthy)** | **0.0851** | **21.19** |

**So 0.085 is "healthy", 1.79 is "broken", and the metric moves ~21× between them.** That gives real thresholds: **ALERT at KL > 0.3, HALT at KL > 1.0** `[EST]`, interpolated between two measured anchors. Note also what fixed it: **focal loss on the unsupervised branch plus teacher EMA, both one-line changes, together worth +7.8 mAP** `[SRC]`. These are not details; they are the anti-collapse mechanism.

When KL rises, the correct response is **per-class thresholds and rebalanced sampling, not a lower global threshold** (§3b).

### The full monitoring dashboard — 8 scalars per round

Design premises: your only free *real* ground truth is the 150 clip-level multi-labels; you need the frozen 250–300-frame box probe of §4.5; you need a **second frozen unlabelled "stop set" of ~2,000 frames** used only for agreement statistics; and you should **train two differently-seeded students per round, one per GPU** — which is exactly what two 3060s are good for, and it unlocks the only label-free signals with theorems behind them.

| # | Scalar | Trigger | Backing |
|---|---|---|---|
| **1** | **Clip-level macro-F1 on the 150 labelled clips** (aggregate boxes → clip multi-label by max-over-frames), with per-class Wilson CIs | **HALT the round** if macro-F1 drops more than the paired MDD (**at n=150, ≈5–7 pts**). Ignore smaller moves | CI math `[derived]`; **primary metric, zero marginal cost per round** |
| **2** | **Box-probe mAP50 on the frozen 250–300-frame set**, 3-seed median, same seeds every round | **HALT** if it drops **>2.0 points** vs the best previous round. Measure your own seed σ in round 0 and set the trigger at 3σ | `[EST]`, grounded in the 1.27–1.82-pt measured classification seed range `[SRC]` and Detectron2's "large variance" warning |
| **3** | **Prediction-agreement κ between consecutive rounds** on the 2,000-frame stop set (reduce detections to per-frame-per-class presence; Cohen's κ; 3-round window) | **STOP at mean κ > 0.99. Do NOT use 0.95** — the Stabilizing-Predictions bound `4(1−T)/T` leaves **21 F-points of slack at 0.95 versus 4.0 at 0.99** | Paper-backed for binary classification; **the boxes→presence reduction is my adaptation and the theorem does not transfer as stated** `[EST]`. Stop set must be a large *fixed* random set, not the shrinking pool `[SRC]` |
| **4** | **Pseudo-label count, total and per class** | **ALERT** if the total moves **>±30%** in one round, or any class moves **>2×**. Count explosion with flat #1 = confirmation-bias runaway; collapse = threshold now above the model's score distribution | Plotting PL counts per round is established practice (UT Fig. 4c, UPS Fig. 1c) `[SRC]`; the ±30%/2× cut-points are `[EST]` |
| **5** | **KL(clip-label prior ‖ pseudo-label class histogram)** | **ALERT > 0.3, HALT > 1.0** | anchors **1.7915 / 0.0851** measured `[SRC]`; cut-points `[EST]` |
| **6** | **Cross-seed disagreement of the two students** on the stop set: per-class presence disagreement, mean Hungarian-matched IoU, and the **R² of ID-agreement vs OOD-agreement across your checkpoint history** | Use the **trend**. **DISABLE entirely if that R² < 0.90** (the Agreement-on-the-Line authors report ≥0.95 strong, ≤0.75 weak, method fails in the weak regime). ALERT if disagreement rises >5 pts while #1 is flat | `[SRC]` — ⚠️ with an important caveat: the measured calibration error is **worst precisely for differently-initialised pairs**, which is your configuration, so treat the absolute value as biased and use only the trend |
| **7** | **Box stability (BoS) on a large unlabelled slice** → regression-predicted mAP → residual against #2 | Early-warning trend only. **DISABLE if probe mAP50 < 15, if you use a transformer backbone, or if the slice has <50 images** — all documented breakdown conditions. ALERT if the residual exceeds 2× the method's own RMSE (~5 pts) | `[SRC]` (arXiv 2403.13803, including its own four stated limitations). Re-fit the BoS→mAP map every round; its successor explicitly notes a single linear fit is inadequate across mAP regimes |
| **8** | **ATC / difference-of-confidence on the clip-level head**, plus the ATC-vs-truth residual on the 150 clips | **ALERT if the residual exceeds 7 points** (ATC's typical MAE is 1–5; worst novel-subpopulation case 18.26). **Critically: a rising ATC estimate with flat or falling #1 is evidence a NEW CLASS has entered the pool**, because ATC over-estimates target performance when class-conditional support shifts | `[SRC]`; per-class ATC on a multi-label head is my extension `[EST]`, 7-pt trigger `[EST]` |

**Composite stop rule** `[EST]`. Stop when **all** of: (a) κ > 0.99 for 3 consecutive rounds; **and** (b) clip-macro-F1 has not improved beyond its paired MDD for 3 rounds; **and** (c) probe mAP50 has not improved by >2 pts for 3 rounds.

**Roll back immediately, without waiting for the stop rule, when any of:** #1 drops beyond MDD · #5 KL > 1.0 · #4 total count swings >±30% with #1 flat · #8 residual > 7 pts · the §3 P-N eigenvalue test gives max λ > 0.9.

**🚨 And one rule that overrides all of the above: never select a checkpoint by a label-free score alone.** SKADA-Bench measured unsupervised model-selection scorers at Spearman ρ ≤ 0.06 (entropy, DEV, SND) up to 0.71 (circular validation), versus **0.98 for a supervised scorer — with up to 10% accuracy lost by selecting on the unsupervised score** `[SRC]`. Keep the best checkpoint by #1, then #2. The label-free signals are alarms, not objectives.

**One idea I would build that I cannot back with a paper:** **temporal track-consistency as a free label-free detector metric** — ID-switch rate, track fragmentation, detection↔track IoU agreement, and per-track class-label flip rate, computed on the unlabelled pool. Video gives you a consistency oracle that still images do not. I searched for published label-free *video/tracking* evaluation metrics and found none on-topic `[UNVERIFIED — no paper found]`. The closest published cousins are SSVOD's cross-IoU/cross-divergence filters. It is cheap, it is specific to this project's advantage, and it is worth prototyping — but label it as unvalidated.

---

## 3b. Avoiding the classic self-training failure

The three failure modes, in the order they will actually bite this project:

**(1) Confirmation bias / error amplification.** The student learns the teacher's false positives, then becomes the teacher. In a *closed* loop this compounds without bound. **The structural defence here is that the loop is not closed:** SAM 3 is frozen and never learns from its own output, so the teacher cannot drift. Only the student can, and the student is re-filtered through the frozen teacher and the clip-level constraints every round. **Opinion: keep the teacher frozen for all rounds.** The classic mean-teacher/EMA design — where the teacher *is* an EMA of the student — is exactly the architecture that enables drift, and the only reason SSOD papers tolerate it is that they have real box labels anchoring the supervised loss. This project does not, so do not copy that pattern. If you later add an EMA teacher, keep the frozen foundation teacher in the ensemble as an anchor.

**(2) Class collapse / rich-get-richer.** Common classes (glassware, pipettes) accumulate pseudo-labels; rare classes (autoclave, spectrophotometer) lose them round over round until they vanish. **Defences, in order of value:**
   - The **presence constraint** (§2.2 Stage D) makes collapse structurally impossible on the 150 labelled clips: a class in `P(c)` always gets at least one tracklet, by construction. This is a genuinely strong guarantee and it is the main reason to spend the video-level labels this way.
   - **Cap** per-class pseudo-label contribution to the training set at, say, 4× the median class count `[EST]`, and **oversample** tail classes to parity. Unbiased Teacher's entire contribution is "a class-balance loss to downweight overly confident pseudo-labels" `[SRC]` — same idea, applied at the loss rather than the sampler.
   - Monitor the **forced-positive rate** per class; a rising rate is collapse in progress.

**(3) Rich-get-richer on *easy views*, which is the under-discussed one.** Tracking propagation mints boxes on *every frame of accepted tracklets*, so the training set over-represents the viewpoints where the teacher already succeeds, and under-represents the occluded/blurred/oblique views where it fails. Left alone, the student gets very good at exactly what it was already good at. **Defences:** (a) cap boxes minted per tracklet (e.g. keep ≤30 frames per tracklet, sampled to maximise appearance diversity rather than uniformly in time) `[EST]`; (b) deliberately mine frames where **the tracker says an object is present but the detector does not fire** — these are the hard positives, and they are free. This is the one place where I would spend real engineering effort; it is the highest-value idea in the self-training part of this design.

**Calibration pitfall — and the fix is two hours of work with a measured 20× improvement.** Detector confidence is not a probability, and self-training makes it worse: each round the student is trained to be confident on exactly the pseudo-labels the teacher was confident about, so the score distribution sharpens without the underlying accuracy improving. The magnitudes are measured `[SRC]`:

| Finding | Measured |
|---|---|
| Faster R-CNN detection-ECE, **post-hoc histogram binning** (15 bins) | **19.235 → 0.890** at IoU 0.6; **32.458 → 0.910** at IoU 0.75. **Histogram binning beat logistic and beta calibration in all cases for detection** |
| Temperature scaling, classification | ECE 16.53 → 1.26 |
| Calibration under few-label SSL | **5.3× worse**: FixMatch ECE **35.88** vs supervised 6.77 |
| Calibrating pseudo-labels before selection | worth **+2.82 points** on its own (UPS) |

**So: fit a 15-bin histogram-binning calibrator on your gold probe set, re-fit it every round, and threshold on calibrated scores.** Free at inference, ~2 hours to implement, and the detection-specific evidence says histogram binning specifically. Never carry a raw threshold across rounds; and always report the *number* of accepted boxes alongside the threshold — if the threshold holds steady while the count doubles, the model got more confident, not more correct.

**🚨 Class-imbalance fix: over-sample rare pseudo-labels, do not filter them harder.** The measured asymmetry is striking: CReST found **minority-class recall of 8.4% at precision 97.7%** `[SRC]` — rare-class pseudo-labels are *scarce but accurate*, so the correct response is to amplify them, not distrust them. CReST's rebalanced sampling (`μ_l = (N_{L+1−l}/N_1)^α`, **α = 1/3**) gave CIFAR10-LT γ=100 **66.3 → 78.1** and ImageNet127 **65.8 → 73.7** over 3 generations `[SRC]`. ⚠️ **α = 1 reverses the bias and drops majority recall by up to 7%** — use α ≈ 1/3. Cheaper still, an **auxiliary balanced head (ABC)** — one extra linear layer with Bernoulli masks inverse to class frequency on a shared backbone — gave CIFAR-10-LT γ=100: FixMatch 72.3/53.8 → **81.1/72.0, i.e. +18.2 on minority classes** `[SRC]`. That is the cheapest long-tail fix available and it is ~20 lines.

Relatedly, **distribution alignment on the pseudo-label marginal using your 150 clips' class distribution as p(y)** is worth more than most architecture choices: ReMixMatch's ablation puts it at **1.34 points, larger than every component except augmentation** `[SRC]`, and the mechanism is confirmed by DARP (measured pseudo-label imbalance ratio **1046** versus a true 150) `[SRC]`. ~20 lines, and it is the best measured-effect-per-line item in this report.

**🔴 Cap the loop at 2–3 rounds, and there is theory saying so.** The error bound for iterative self-training is `err_T ≤ β^(T+1)(α₀ + O(1/√n))`, it is **tight**, and error grows as `min(0.5, ½·2^T·α₀)` **even with infinite unlabelled data** `[SRC]`. Empirically: Noisy Student got **+0.5 and +0.3 from rounds 2 and 3 — with full ImageNet labels and 130M images** `[SRC]`. You will not do better. **Three rounds with a hard regression gate; stop.**

**And if you must iterate, make it gradual.** On rotating-MNIST, direct self-training to the target domain reached **33.0%** while *gradual* self-training through intermediate domains reached **87.9%** `[SRC]`. For this project that means: **run the easy footage first** (clean, well-lit, unoccluded benchtop clips), then admit the messy footage in later rounds. Order the pool by a difficulty proxy — clip-classifier confidence, or mean teacher score — and ramp.

**Finally: do not assume collapse will be obvious.** FixMatch's weak-for-strong variant *"peaked at 45% … and progressively collapsed to 12%"*; the outlier-label run sat at **10% = chance**; Unbiased Teacher without EMA reaches a state where *"instances of most object categories in pseudo-labels disappear"* `[SRC]`. **Log the per-class pseudo-label histogram and its KL against the clip-label marginal every N iterations and alarm on it** — that metric moves 21× between a collapsing and a healthy run (see below).

**A useful perspective check on where to spend your calendar time** `[SRC]`: FixMatch's *entire* threshold sweep from 0.25 to 0.99 spans **1.56 points**, and Semi-DETR's 0.2→0.6 sweep spans **0.9 mAP**. Meanwhile the backbone is worth **35–48 points** and OOD pool filtering **8.2 AP** (§4.2). **Threshold tuning is a second-order knob that consumes first-order calendar time.** Set it once with a 3-point sweep against student mAP and move on.

## 4. Active learning verdict

### 4.1 The straight answer

**No. At a 20–50 clip budget, do not implement an active-learning acquisition function. Use stratified random selection, and spend the saved engineering time on the seed set's *content* and on verify-and-correct throughput.** One exception is worth ~100 lines: **MaxHerding over frozen DINOv2/SigLIP embeddings**, which is the only method with a verified positive result at detection budgets this small.

The evidence is not ambiguous, and several widely-cited methods are **measurably worse than random** `[SRC]`:

| Method | Measured vs random at small budgets | Verdict |
|---|---|---|
| **LL4AL / learning-loss** | **−5.07 mAP50 on VOC, −2.74 on COCO** under fair protocols; exactly **0.00** at 2k on VOC07+12; −0.30 on COCO at 7k; *"never manages to outperform the random baseline"* | 🚫 **actively harmful**, and it requires surgery on the detector. Worst value-per-hour in the field |
| **Uncertainty / entropy / margin / BALD at round 0** | **−6.5 to −8.3** at 200–600 CIFAR-100 labels (DINOv2 features); **entropy 23.0 vs random 43.5** on Yahoo at 32 labels; a 2026 remote-sensing detection study finds it **below random at 50–300 images** | 🚫 documented cold-start failure across ~10 independent groups. Useful *later*, poisonous at round 0 |
| **Core-set / k-center-greedy** (Sener & Savarese) | range **−6.7pp to +1.5pp**; **+0.07 (i.e. zero)** in the only fair box-budget detection benchmark; **−3.83pp** at CIFAR-100 20%. BADGE's own paper: *"often performs worse than random"*; Cluster-Margin's: *"does not go beyond the performance of Random Sampling on all datasets"* | 🚫 not worth it — use MaxHerding instead |
| **ProbCover** | **+0.0** vs random by 1,000 CIFAR-100 labels; **−2.4 on Food101**. Its own successor paper says the δ heuristic *"frequently fail[s]"* and accuracy *"may even decrease with more labels"* **on imbalanced data** — and this project's class list *is* imbalanced | 🚫 and the δ hyperparameter is the repo's top unanswered issue |
| **TypiClust** | **+16.5pp @100 labels, decaying to +0.1 @1,600 and −0.2 on Food101**; **loses to random at 4 of 5 VOC detection budgets**; its RP/DC variants differ by up to 12.6pp with the winner flipping by dataset | ⚠️ round 0 only, then stop. And see §4.2 — its gain is largely an implicit class-balancer |
| **MI-AOD** | headline low-budget gain is **structurally zero**; independent reproduction: *"consistent improvement cannot be assured"*, reaching no more than 72% mAP50 vs random's **79.80 ± 0.10**; **−0.57** on COCO | 🚫 as an AL method. Its *training* trick (instance-uncertainty re-weighting) is the valuable part — get that from Soft Teacher instead |
| **ALMDN** | +0.48 ≈ **1.1σ** at its own smallest budget; **−1.1 to −3.5 mAP50** under a fair box budget | 🚫 |
| **CALD** | workshop paper with **no standard deviations anywhere**; VAAL hits exactly **+0.0%** in its own cycle-2 table | 🚫 unfalsifiable as published |
| **BatchBALD** | MNIST-family classification only; needs MC-dropout + a Bayesian head; **no bounding-box formulation exists** | 🚫 not applicable |
| **BADGE** | genuinely the most reliable classical method, not harmful — but **no clean multi-label/multi-instance detection analogue exists** | ⚠️ nothing to port |
| **MeanEntropy** | the **only** classical method beating random at *every* cycle in the one unified, strongly-regularised box-budget benchmark: **+1.2** | ✅ the honest classical baseline |
| **🏆 MaxHerding** (frozen embeddings, one-shot, no model in the loop) | **+1.2 to +1.5 mAP at 200–1,000 VOC detection images (~3σ)** — the only verified positive at this budget | ✅ **implement this one.** ~100 lines, lengthscale = 1, no tuning, no model needed |

One detection-specific nuance worth knowing, because it **contradicts** the classification literature: **PPAL** reports that *"uncertainty-based sampling is more critical in the early AL stages while diversity-based sampling is more essential in later rounds"* `[SRC]` — the exact opposite of TypiClust's claim for classification. The detection-specific finding should win on a detection task, but the two bodies of evidence genuinely disagree and I would not bet engineering time on either.

### 4.2 Why random wins here, and what actually *is* the lever

The low-budget AL literature's wins are largely **class coverage in disguise**. TypiClust's own figure shows its selections have total-variation distance **0.09** from the class prior versus random's **0.33** `[SRC]` — i.e. much of its gain is implicit class balancing, which you can get directly and for free. CSVAL puts it plainly: *"most active querying strategies contain selection bias to specific classes"* `[SRC]`. And at n=150 clips with K=40–60 classes, **7–19 classes receive zero examples under plain random sampling** `[SRC]`.

**So: stratify over the video-level multi-labels you already have.** Round-robin so every class and every rare co-occurrence appears, and guarantee ≥k clips per class. Hours of work, free, and it captures most of what the clever methods were buying.

**🏆 But the dominant lever is not the acquisition function at all — it is *which specific instances* you label.** The single most striking measurement in this entire report: FixMatch at **one label per class**, same method, same hyperparameters, varying only *which* example was chosen `[SRC]`:

| Label choice | Accuracy |
|---|---|
| **Most-prototypical** | **median 78%, max 84%** |
| Middle | 65% |
| **Outliers** | **10% (= chance; fails to converge)** |

**A 68-point swing from label prototypicality alone.** Compare that to **+1.2 to +1.5 mAP** for the best published acquisition function at this budget. **Hand-pick your clips: canonical, unambiguous, well-lit, centre-framed, one clear instance per class.** Do not sample them. This is the highest effect-to-effort item in the report and it costs about a day.

Two further levers that dwarf the acquisition function `[SRC]`:
- **Backbone: 35–48 points.** FineSSL at 1 label/class: FixMatch-from-scratch **48.45 ± 2.99** → fine-tuned foundation model **96.15 ± 0.13**, at **1/6 the training time**. Oliver et al.: plain **ImageNet transfer (12.09% error) beats the best SSL method (13.13%)**, and still beats it (12.91%) **with zero class overlap**.
- **🚨 OOD filtering of the unlabelled pool: 8.2 AP, and getting it wrong makes the pool actively harmful.** Unbiased Teacher measured **37.2 AP on a clean pool → 34.5 AP on an OOD-contaminated pool, which is *below* the 35.6 AP labelled-only baseline**; a kNN-to-class-prototype filter recovers it to **42.7** at only 26% slower iterations `[SRC]`. Corroborated on Semi-Aves: **53.8 → 45.7** with open-set unlabelled data `[SRC]`. **For this project that means: a "large pool of unlabelled video" containing corridors, offices, car parks and meeting rooms will make your detector worse unless you filter it.** Budget a day for a per-class feature-bank + kNN cosine filter (threshold at mean + β·std). This is the highest-risk item you can cheaply de-risk, and the brief does not mention it.

### 4.3 What to label, in what unit, how many

**Recommendation, in priority order:**

1. **🏆 One box on one frame of every clip you already have (150 boxes).** This is not active learning, it is the single highest-return annotation action available. Chéron et al.: video-level labels **43.9 mAP@0.2 → one box 66.8 (88% of fully supervised) → temporal + 3 boxes 74.5 (98%)** `[SRC]`. See §1b.
2. **Then 20–50 *new* clips, chosen by stratified random over the label set + MaxHerding on frozen DINOv2 embeddings**, prioritising the five classes with zero public boxes (`centrifuge`, `autoclave`, `orbital shaker`, `spectrophotometer`, real `fume hood`) and the fine-grained confusables.
3. **Unit: verify-and-correct machine-proposed boxes on frames sampled at 1 Hz** — not from-scratch boxing, and not whole clips densely.
   - **1 Hz is where YouTube-BB, TAO and AVA all independently converged** `[SRC]`.
   - **Uniform keyframe sampling beats human-chosen keyframes** `[SRC]` — so do not let annotators pick frames.
   - Kuznetsova et al.: keyframes + visual interpolation gives **>50% total human-time reduction and 60% fewer manual boxes** `[SRC]`.
4. **Also collect per-class counts per clip** — see §2.2 Stage D; counts are measured as **>2× cheaper than a centre click and >38× cheaper than a bounding box** `[SRC]` and they tighten the presence constraint into a count constraint.
5. **Separately and non-negotiably: a frozen box-labelled probe set of 250–300 densely-boxed frames**, labelled once, never trained on. See §4.5 for why 250 and not 50.

### 4.4 🔴 Verify-and-correct: the real numbers, and a correction to the folklore

The brief asks for evidence that verify-and-correct is much cheaper than from-scratch boxing. It is — **but the commonly-cited speedup factors are inflated, and one widely-recommended technique is actually slower.**

| Measurement | Value | Source status |
|---|---|---|
| Papadopoulos et al., CVPR 2016, **human verification instead of drawing** | **6×–9× annotation-time reduction**, detectors *"almost as good as those trained in a fully supervised setting"* | `[SRC]` |
| **Verify a proposed box** (professional annotators) | **1.6 s** | `[SRC]` |
| **Draw a box from scratch** (professional, in-house) | **3.7–5.9 s** | `[SRC]` |
| ⇒ **realistic verify-vs-draw speedup** | **~2.3–3.7×** | `[calc]` |
| Papadopoulos et al., ICCV 2017, **extreme clicking** | 7 s/box vs a **34.5 s AMT crowdworker baseline** = "5× faster" | `[SRC]` — but see below |
| 🚨 **Extreme clicking measured against *professional* annotators** | **+44% task time and +87% cumulative creation time**, 18 professional annotators, versus plain drag-drawing | `[SRC]` |
| **SAM-2-in-the-loop annotation** | **37.8 → 4.5 s per frame (8.4×)** | `[SRC]` |
| Vendor throughput claims (Roboflow "95%", V7 "10×") | no baseline stated | `[VENDOR]` — unfalsifiable, ignore |

**Three corrections that matter:**
1. **The famous "35 s per box" is an Amazon-Mechanical-Turk figure, not a professional-annotator figure.** Against a trained in-house annotator the true baseline is **3.7–5.9 s**, so verify-and-correct buys ~**2.3–3.7×**, not ~20×. Plan with the smaller number.
2. **Do not adopt extreme clicking.** It is **+44% slower** for professional annotators. Its published speedup exists only relative to an inflated crowdworker baseline. I had this wrong before checking.
3. **The real 8× comes from putting a segmentation model in the loop** (37.8 → 4.5 s/frame), not from a clicking protocol. That is already in this plan via SAM 2/SAM 3.

**Budget estimate for this project** `[EST]`, using 1.6 s/verify, 5 s/correction, ~25% correction rate, 8 boxes/frame:
- 150 existing clips × 1 frame × 8 boxes ≈ 1,200 boxes → **~45 min**
- 50 new clips × 5 frames (1 Hz over 5 s) × 8 boxes = 2,000 boxes → **~75 min**
- 300-frame frozen probe set, dense, higher care ≈ 2,400 boxes at ~8 s → **~5.5 h**
- 150 clip-level count annotations → **~1 h**
- **Total ≈ 8–9 hours of human time**, of which the frozen probe set is the majority — and it is the part you must not skimp on.

### 4.5 Why the eval set must be ~250–300 frames, not 50

This is the part of the plan people cut, and the statistics say it is the part that makes everything else meaningless if you cut it. Binomial/Wilson intervals for a proportion `[derived]`:

| Target 95% half-width | n needed at p=0.80 | at p=0.50 (worst case) |
|---|---:|---:|
| ±10 pts | 60 | 93 |
| **±5 pts** | **245** | **381** |
| ±3 pts | 682 | 1,064 |
| ±1 pt | 6,145 | 9,601 |

And the one that should genuinely change your plan — **per-class recall confidence versus the number of ground-truth instances of that class** (at recall ≈ 0.7) `[derived]`:

| GT instances of the class | 95% Wilson half-width |
|---|---|
| 5 | **±31.4 pts** |
| 10 | **±24.8 pts** |
| 20 | ±18.7 pts |
| 50 | ±12.3 pts |
| 100 | ±8.8 pts |

**With 10 instances of "spectrophotometer" in your eval set, your per-class recall estimate is ±25 points. You cannot distinguish 50% from 75% recall.** Stratify the probe set so every class has **≥30 instances** `[EST]`, and report intervals rather than point estimates.

**Two cheap statistical wins:**
- **🏆 Always compare checkpoints *paired*, on a frozen item list** (McNemar / paired bootstrap). The minimum detectable difference then depends only on the *discordant* items, making paired comparison **3–5× cheaper** `[derived]`: at n=100 with 10% disagreement, paired gives **±8.9 pts** versus **±15.8 pts** unpaired.
- **Keep the probe set frozen and random.** Active Testing (ICML 2021) can choose test points for ~2–4× label efficiency `[SRC]`, but it introduces a selection bias requiring the `R̂_LURE` estimator and it breaks paired comparison. Use active selection for *training* labels only.

**Honest caveats on evaluation, which you should state out loud to stakeholders:**
- **Oliver et al.: getting ±1% at 95% confidence needs ~20,000 examples**, and with a realistically-sized validation set *"differentiating between the performance of the models is not feasible"* `[SRC]`.
- **Test-set label noise bounds everything.** Northcutt et al. (NeurIPS 2021 D&B): **≥3.3% label errors on average across 10 benchmark test sets, ≥6% on ImageNet validation**; and rankings genuinely flip — on corrected ImageNet labels **ResNet-18 beats ResNet-50** once the prevalence of originally-mislabelled test examples rises by just 6% `[SRC]`.
- **There is no paper on mAP confidence intervals or required test-set size for object detection.** I looked hard; ~8 distinct searches returned nothing on-topic. If you need a citation here, there isn't one — say so. `[UNVERIFIED]` by absence.
- **The widely-repeated "~0.2–0.4 AP detection seed variance" is folklore I could not verify** — it is not in Detectron2's MODEL_ZOO (which says only, qualitatively, *"the final results of these configs have large variance across different runs"*) and no paper states it `[UNVERIFIED]` — **do not cite it.** The best-measured proxy is classification: CIFAR-10 seed std **±0.20%** over 500 seeds with a **1.27-point range**; a 10,000-seed study found a **1.82-point range** `[SRC]`. My working number for a small-data detector fine-tune: **±0.5–1.5 AP** `[EST]`. **Measure your own with ≥3 seeds in round 0 and treat anything below ~2 mAP as noise.**

---

## 5. Tables

### 5.1 SSOD methods: COCO partial-label mAP (the standard 1/5/10% protocol)

All numbers `[SRC]`. Primary source for the comparison block: Sparse Semi-DETR (CVPR 2024, arXiv 2404.01819) Table 1, which tabulates the field consistently. Repo-confirmed numbers noted separately.

| Method | Venue | Detector | 1% | 5% | 10% | Code |
|---|---|---|---|---|---|---|
| **Supervised only** | — | FCOS | 8.43 | 17.01 | 20.98 | — |
| **Supervised only** | — | Faster R-CNN | 9.05 | 18.47 | 23.86 | — |
| **Supervised only** | — | DINO | 18.00 | 29.50 | 35.00 | — |
| Humble Teacher | CVPR 21 | two-stage | 16.96 | 27.70 | 31.61 | yes |
| Instant-Teaching | CVPR 21 | two-stage | 18.05 | 26.75 | 30.40 | — |
| Unbiased Teacher | ICLR 21 | Faster R-CNN | 20.16 | 27.84 | 31.39 | MIT |
| Soft Teacher | ICCV 21 | two-stage | 20.46 | 30.74 | 34.04 | yes |
| DSL | CVPR 22 | one-stage | 22.03 | 30.87 | 36.22 | yes |
| Dense Teacher | ECCV 22 | one-stage | 22.38 | 33.01 | 37.13 | yes |
| PseCo | ECCV 22 | two-stage | 22.43 | 32.50 | 36.06 | yes |
| Unbiased Teacher v2 | CVPR 22 | one-stage | 22.71 | 30.08 | 32.61 | yes |
| Consistent-Teacher | CVPR 23 (highlight) | one-stage | 25.50 | 36.60 | 40.20 | Apache-2.0 |
| **Omni-DETR** | CVPR 22 | DINO | 27.60 | 37.70 | 41.30 | Apache (amazon-research) |
| Semi-DETR | CVPR 23 | DINO | 30.50 | 40.10 | 43.50 | PaddleDetection |
| **Sparse Semi-DETR** | CVPR 24 | DINO | **30.90** | **40.80** | **44.30** | — |

Repo-confirmed extras:
- Unbiased Teacher official repo also reports **2% → 24.16** `[SRC]`; VOC07→VOC12 **AP50 80.51 / AP 54.48**; licence MIT.
- Consistent-Teacher official repo: **100% COCO + unlabelled → 48.20** `[SRC]`; Apache-2.0.
- STAC: **5% labelled → 24.38 mAP**, beating a supervised baseline that reaches 23.86 at 10% ("2× data efficiency"); VOC07 **AP50 76.30 → 79.08** `[SRC]`.
- Soft Teacher: full COCO + 123k unlabelled → **40.9 → 44.5 (+3.6)**; Swin-L test-dev **58.9 → 60.4**, with Objects365 pre-training **61.3** `[SRC]`.
- MixPL (arXiv 2312.07006): DINO reaches **60.2 mAP** on COCO val2017; DINO Swin-L **+2.5 mAP** `[SRC]`.
- Sparse Semi-DETR VOC: AP50 **86.30** / AP50:95 **65.51** vs supervised DINO 81.20 / 59.60 `[SRC]`.
- Sparse Semi-DETR small-object AP at 1%: **14.8** (vs Semi-DETR 13.6) — note how brutal small-object AP is in the low-label regime `[SRC]`.

**Read this table the right way.** The gains at 1% are enormous in relative terms — Faster R-CNN 9.05 → Soft Teacher 20.46 is **+11.4 AP, a 2.26× multiplier** `[SRC]`; DINO 18.0 → Semi-DETR 30.5 is **+12.5 AP, 1.69×** `[SRC]`. At 10% the multiplier shrinks (23.86 → 40.20 for Consistent-Teacher is +16.3 but 1.68×; DINO 35.0 → 44.3 is only 1.27×). **Semi-supervision pays most exactly where this project sits.** But every row assumes box labels exist.

### 5.1b The number that invalidates the whole SSOD route at *this* budget

The 1/5/10%-of-COCO protocol is **not** a tiny budget. 1% of COCO ≈ 1,180 images ≈ ~15 box-labelled instances per class. This project is far below that. A January 2026 paper measured SSOD in the genuinely tiny regime — *Practical Insights into Semi-Supervised Object Detection Approaches* (Wang, Balasubramaniyam, Sangem, Guevara, Caragea; arXiv 2601.13380), comparing MixPL, Semi-DETR and Consistent-Teacher at **1 / 5 / 50 / 150 shots per class** on COCO, VOC and a domain-specific "Beetle" dataset. All numbers `[SRC]`:

| Shots/class | Dataset | MixPL | Semi-DETR | Consistent-Teacher |
|---|---|---|---|---|
| **1** | MS-COCO | **8.80** | 6.00 | 6.13 |
| 5 | MS-COCO | **23.30** | 20.20 | 16.27 |
| 50 | MS-COCO | 35.80 | **37.10** | 29.14 |
| 150 | MS-COCO | 41.60 | **42.20** | 33.42 |
| **1** | Pascal VOC | **9.00** | **1.00** | 1.40 |
| 5 | Pascal VOC | **45.30** | **6.20** | 31.65 |
| 50 | Pascal VOC | **58.10** | 50.40 | 45.20 |
| 150 | Pascal VOC | **63.10** | 55.90 | 52.00 |
| **1** | Beetle (domain-specific) | 6.00 | **0.80** | **13.08** |
| 5 | Beetle | 10.60 | 6.20 | **31.65** |
| 50 | Beetle | **69.30** | 61.60 | 58.66 |
| 150 | Beetle | **71.10** | 65.10 | 68.70 |

Hardware: 4× A100 80 GB `[SRC]`. Model size / latency: MixPL 920 MB / 37–40 ms; Semi-DETR 885 MB / 40–43 ms; **Consistent-Teacher 372 MB / 9–15 ms** `[SRC]`. Their recommendation: "For high-accuracy demands with sufficient computational resources, MixPL or Semi-DETR are optimal choices; for latency-sensitive and resource-constrained environments, Consistent-Teacher provides a highly favorable trade-off" `[SRC]`.

**Read the bolded cells.** Semi-DETR — the best method in the world at 1% of COCO (30.5 mAP) — **collapses to 1.00 mAP on VOC and 0.80 mAP on Beetle at 1 shot per class**. The transformer SSOD methods need enough labels to bootstrap matching and they simply do not work below a few tens of boxes per class. This is the single strongest piece of evidence against "pick the SOTA SSOD method and run it", and it is directly on point: with zero boxes, even after round 1 mints a handful, you are in the regime where the published SOTA is worse than useless. **Rank order at this budget: MixPL > Consistent-Teacher >> Semi-DETR.** Consistent-Teacher is additionally the only one that is both 2.5× smaller and 3× faster and reaches 13.08 at 1 shot on the domain-specific dataset — if you ever do run an SSOD framework here, run that one.

### 5.2 Training cost of the published SSOD methods (why none of these configs run as-is)

| Method | Hardware in the paper/repo | Schedule | Source |
|---|---|---|---|
| Sparse Semi-DETR | **8× RTX A6000 (48 GB)** | ~2 days, 120k iters | paper `[SRC]` |
| Consistent-Teacher | 8× V100, 5 img/GPU; or **2×8 + fp16 → <1 day COCO** | 180k iters (1/5/10%) | repo `[SRC]` |
| Unbiased Teacher | 8 GPUs, 16 labelled + 16 unlabelled | — | repo `[SRC]` |
| Omni-DETR | R50, batch 16, **100–1000 epochs** depending on label fraction | — | paper `[SRC]` |
| DECOLA | 8× V100 32 GB, ~50 h (R50); 2 nodes × 8 for Swin-L | — | paper `[SRC]` |

A 2× RTX 3060 box is roughly **1/16 to 1/30** of these setups in aggregate memory and throughput `[EST]`. Reproducing any of these recipes end-to-end is out of scope. This is a central reason the recommendation is "frozen foundation teacher + small student", not "train an SSOD framework".

### 5.3 SSOD that starts from zero boxes — the honest answer

| Approach | Needs boxes? | What it actually gives | Source |
|---|---|---|---|
| STAC / UT / Soft / Dense / Consistent-Teacher / Semi-DETR / Sparse Semi-DETR | **Yes** — 1% of COCO ≈ 1,180 box-labelled images | nothing at 0 boxes | all `[SRC]` |
| **Omni-DETR** with `TagsU`/`TagsK` | **Yes, still** — two-stage: "burn-in training of the student network alone on the labeled data" then student–teacher. Experiments use 1–30% fully labelled | image-level tags on the *rest*: 10% boxes + 90% tags → **34.7 vs 28.0** (+6.7) | paper `[SRC]` |
| **DECOLA** | Yes for phase 1 | then self-trains on **ImageNet-21k image-level labels** (14M images); LVIS OV APnovel R50 **27.6**, Swin-L **46.9** | paper `[SRC]` |
| **SAM 3 / open-vocab teacher → student** | **No** | the only route that works at literally zero boxes | §6 |

**Conclusion for §1 of the brief: there is no published SSOD method that bootstraps from image-level or video-level labels alone.** Omni-DETR is the closest and is explicitly advertised for mixed weak annotation including image-level tags — but it still requires a burn-in on fully box-labelled data. The zero-box route must be a foundation detector's output used as the initial "labelled" set, which is what the recommendation does.

### 5.4 Foundation-model ceiling on *novel, narrow* domains — the most important numbers in this report

| Benchmark | Model | Zero-shot AP | 10-shot AP | Source |
|---|---|---|---|---|
| **RF100-VL** (100 novel-domain detection datasets) | SAM 3 | **15.2** | **36.5** | SAM 3 Table 2 `[SRC]` |
| RF100-VL | Grounding DINO-T | 15.7 | 33.7 | SAM 3 Table 2 `[SRC]` |
| ODinW13 | SAM 3 | 61.0 | 71.8 | SAM 3 Table 2 `[SRC]` |
| ODinW13 | gDINO 1.5-Pro | 58.7 | 67.9 | SAM 3 Table 2 `[SRC]` |
| LVIS (zero-shot, boxes) | SAM 3 | 48.5 | — | SAM 3 `[SRC]` |
| LVIS (zero-shot, boxes) | OWLv2* | 43.4 | — | SAM 3 `[SRC]` |
| LVIS (zero-shot, boxes) | DINO-X | 38.5 | — | SAM 3 `[SRC]` |
| COCO (zero-shot, boxes) | SAM 3 | 56.4 | — | SAM 3 `[SRC]` |
| COCO (zero-shot, boxes) | OWLv2* | 46.1 | — | SAM 3 `[SRC]` |
| SA-Co/Gold | SAM 3 / **human** | 54.1 / **72.8** cgF1 | — | SAM 3 `[SRC]` |

RF100-VL is a NeurIPS 2025 Datasets & Benchmarks paper (*Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models*, Robicheaux et al.) built specifically from "diverse concepts not commonly found in VLM pre-training". Its headline finding: GroundingDINO and Qwen2.5-VL "achieve **less than 2% zero-shot accuracy** on challenging medical imaging datasets" `[SRC]`.

**Interpretation for lab equipment `[EST]`:** lab equipment sits between ODinW13 (common-ish objects, SAM 3 ZS 61.0) and RF100-VL (genuinely novel, SAM 3 ZS 15.2). Microscope, beaker, flask, test tube, syringe and pipette are web-common nouns; centrifuge and autoclave are moderately common; "spectrophotometer", "orbital shaker", "plate reader" are rare and visually near-identical grey boxes. My estimate: **zero-shot AP50 roughly 30–50 for the common half of the vocabulary and 5–15 for the rare half** `[EST]`. The 15.2 → 36.5 jump from 10 shots per class is the strongest single argument in this report for spending a small labelling budget.

### 5.4b WSOD / WSOL from image-level labels only — the honest ceiling, and the one method you should actually copy

#### 5.4a The classic MIL line plateaued, then foundation models broke the plateau

VOC07 test mAP@0.5, trained on VOC07 trainval (5,011 images), **image-level labels only**. Consolidated from WeakSAM Table 1 and cross-verified against each original paper. All `[SRC]`.

| Method | Year | Proposals | Backbone | VOC07 AP50 | COCO AP | COCO AP50 | **% of FRCNN-VGG16 (69.9)** |
|---|---|---|---|---|---|---|---|
| **Faster R-CNN (FULL box sup.)** | 2015 | RPN | VGG16 | **69.9** | **21.2** | **41.5** | 100% |
| **Faster R-CNN (FULL)** | — | R50-FPN | — | **76.1** | **37.9** | **58.8** | — |
| WSDDN | 2016 | EdgeBoxes | VGG16 | 34.8 | 9.5 | 19.2 | 49.8% |
| OICR | 2017 | SelSearch | VGG16 | 41.2 | 8.0 | 18.9 | 58.9% |
| PCL | 2018 | SelSearch | VGG16 | 43.5 | 8.5 | 19.4 | 62.2% |
| C-MIL | 2019 | SelSearch | VGG16 | 50.5 | — | — | 72.2% |
| **MIST / Wetectron** | 2020 | SS, MCG | VGG16 | 54.9 | 11.4 | 24.3 | 78.5% |
| **CASD** | 2020 | SelSearch | VGG16 | 56.8 | 12.8 | 26.4 | 81.3% |
| CBL (WSOD-CBL) | 2023 | SelSearch | VGG16 | 57.4 | 13.6 | 27.6 | 82.1% |
| **SCEC** (pure-MIL SOTA) | 2025 | SelSearch | VGG16 | **58.2** | — | — | **83.3%** |
| SoS-WSOD (+self-train) | 2022 | RPN | +FRCNN R50 | 64.4 | 16.6 | 32.8 | 92.1% |
| W2N (CASD) | 2022 | RPN | +FRCNN R50-FPN | 65.4 | 15.9 | 33.3 | 93.6% |
| WSOVOD‡ | 2024 | LO-WSRPN + **SAM** | RN50-WS-MRRP | 63.4 | 20.5 | 29.1 | 90.7% |
| **WeakSAM (MIST)** | 2024 | **SAM** | VGG16 | 67.4 | 22.9 | 35.2 | 96.4% |
| **WeakSAM (MIST) + FRCNN R50** | 2024 | SAM→RPN | +FRCNN R50 | **71.8** | 23.8 | 38.5 | **102.7%** |
| **WeakSAM (MIST) + DINO** | 2024 | SAM→RPN | +DINO (DETR) R50 | **73.4** | **26.6** | 39.3 | **105.0%** |

Papers' own verbatim summaries `[SRC]`: MIST — *"ours is 78.1% of Faster R-CNN on average"* (VOC) and *"ours is 56.9% of Faster R-CNN on average"* (COCO). WeakSAM — *"WeakSAM (MIST) with DINO retraining even has comparable performance with fully-supervised Faster R-CNN."*

**The honest ceiling, stated four ways:**
1. **VOC07 mAP@0.5: classic MIL-WSOD retains 78–83% of a matched fully-supervised detector.** Foundation-model proposals take that to 96–105% `[SRC]`/`[calc]`.
2. **COCO is brutally different. MIST's 11.4 AP is 30.1% of detectron2's R50-FPN (37.9 AP)** `[calc]`. The widely-quoted "56.9%" is against a 2015 VGG16 baseline, not a modern detector. **Anyone quoting 56.9% as "the WSOD gap on COCO" is quoting a decade-old comparator.**
3. **mAP@0.5 flatters WSOD; the boxes are systematically loose.** AP/AP50 ratio is **0.48–0.51 for classic WSOD vs 0.645 for fully-supervised R50-FPN** `[calc]`. At IoU 0.75 SoS-WSOD retains only **37.0%** of fully-supervised; at IoU 0.5, 55.8% `[calc]`.
4. **Small objects collapse.** On matched R50-C4 backbones WSOD retains **39.9% of AP, 42.3% of AP_large, but only 28.0% of AP_small** `[calc]`. Absolute AP_small: MIST **3.6**, SoS-WSOD WSOD-stage **2.3**, vs fully-supervised R50-FPN **22.4** `[SRC]`. The within-model large/small penalty is **4.5–8.8× for WSOD vs 2.2–3.6× fully supervised** `[calc]`.

#### 5.4b WeakSAM is the method to copy, and it fits a 3060

**WeakSAM** (*Segment Anything Meets Weakly-supervised Instance-level Recognition*, **ACM MM 2024**, arXiv 2402.14812, repo [github.com/hustvl/WeakSAM](https://github.com/hustvl/WeakSAM)). Image-level labels only. **4 GPUs, 5,667 MiB per card, 9 h WSOD training** `[SRC]` — i.e. **it fits comfortably in 12 GB.**

Its ablation is the most decision-relevant table in the whole weakly-supervised literature, because it shows *the bottleneck was never the MIL objective* `[SRC]`:

| Proposals | #/img | Recall@0.50 | Recall@0.75 | **Recall@0.90** | AP50 | GPU mem/card | Proposal gen | WSOD train |
|---|---|---|---|---|---|---|---|---|
| Selective Search | 2001 | 92.6 | 57.7 | **19.2** | 54.9 | **17,810 MiB** | 11.6 h | 16 h |
| SAM + dense grid only | 129 | 79.6 | 50.7 | 24.3 | 45.2 | — | — | — |
| + fine CAM peaks | 151 | 88.9 | 67.0 | 37.2 | 63.3 (+18.1) | — | — | — |
| **+ cross-attention peaks** | **213** | **95.6** | **75.0** | **42.1** | **67.4 (+22.2)** | **5,667 MiB** (−68%) | **4 h** (−66%) | **9 h** (−44%) |

**Seventeen years of MIL research bought roughly +20 mAP on VOC07. Swapping Selective Search for SAM-prompted proposals bought +22.2 mAP in one step, while cutting VRAM by 68% and training time by 44%** `[calc]`. The binding constraint was proposal tightness: Selective Search has **19.2% recall at IoU 0.90**.

This is a direct, independent, measured validation of this report's architecture: **a frozen foundation model that produces tight class-agnostic proposals, plus weak labels to name them, plus a student retrained on the result.** WeakSAM's retrain ablation also matters: Faster R-CNN 68.4 (top-1 pseudo-GT) → 70.7 (adaptive pseudo-GT) → **71.8** (+RoI drop); DINO 71.1 → 72.8 → **73.4** `[SRC]`.

#### 5.4c Where WSOD breaks, with numbers — and every failure mode applies to this project

**Part domination** — per-class VOC07 AP as % of fully-supervised `[SRC]`/`[calc]`: **person 31.1%** (CASD 23.7 vs FRCNN 76.3), **chair 48.5%**, **bottle 59.1%**, plant 66.2%, table 73.4% … versus **aeroplane 100.7%**, tv 98.4%, motorbike 96.9%, bus 95.3%. MIST's own failure taxonomy, verbatim `[SRC]`: *"(1) relevant parts are predicted as instances of objects (hands and legs, bike wheels); (2) **in extreme examples, part domination remains (model converges to a face detector)**; (3) object co-occurrence confuses the detector when it predicts the sea as a surfboard or the baseball court as a bat."*

**Instance ambiguity is the single biggest failure mode, not part domination.** OIM (AAAI 2020) `[SRC]`: VOC07 trainval has 7,913 image-level labels but 15,662 annotated instances ⇒ **49.5% of all ground-truth instances are structurally unreachable as positives under the one-instance-per-(class,image) assumption, and are actively trained as negatives** `[calc]`. MIST's ablation apportions the blame `[SRC]`: baseline 42.5 → +PCL 43.5 → **+MIST top-15% selection 51.4 (+8.9)** → +Concrete DropBlock 54.9 (+3.5). **Instance ambiguity = 72% of the gain; part domination = 28%** `[calc]`. MIST, verbatim: *"This aligns with our observation that instance ambiguity is the biggest bottleneck for WSOD."* Its hyperparameters: select top **p = 0.15** of ranked proposals, then NMS at **IoU 0.2** `[SRC]`.

**🏆 The most actionable finding for this project — an object-size prior recovers ~35% of the entire WSOD→FSOD gap.** C-WSL's oracle probe, verbatim `[SRC]`: *"Without constraints provided by tight bounding boxes, rigid parts are easier to learn and mostly sufficient to differentiate the object from others. So, WSL detectors focus on local parts instead of the whole object"* … *"We preprocess the region candidates by removing all boxes whose size is smaller than the smallest object… **The AP on 'person' improves to 40.0% and the mAP over all the classes improves to 52.7%**"* (from 20.3 and 45.7). Published part-domination fixes recover only +2.6 to +3.8 mAP `[SRC]` — roughly half as much. **Lab equipment has exceptionally strong, knowable size priors** (a fume hood occupies ≥15% of frame height; a microscope ≥8%; an autoclave ≥20%; a pipette ≤10%). **Add a per-class minimum/maximum box-area gate to the §2.2 cascade. It is three lines of code and the literature says it is worth more than any attention mechanism.**

**Co-occurrence** — a genuine literature gap: **no WSOD paper publishes a co-occurrence confusion matrix or per-class-pair error breakdown** `[SRC]`. The quantified evidence lives in weakly-supervised segmentation. W-OoD (CVPR 2022), verbatim `[SRC]`: *"**if one always sees train on rail, how can one learn that rail is not part of the train?**"* — and its per-class deltas show the effect is a **pure precision failure** (VOC12 seed precision +4.7, recall +0.3) `[calc]`, with `train` gaining +10.4 IoU, 4.5× the average. For this project the analogues are pipette↔bench, centrifuge↔bench, microscope↔stool, glassware↔fume-hood-interior. **The absence/known-negative constraint in §2.2 Stage D is the direct counter-measure**, and it is why that stage is worth the engineering.

**Evaluation honesty — CorLoc is not detection.** Verbatim from WSDDN `[SRC]`: CorLoc is *"the percentage of images… for which **the most confident detected bounding box** overlaps by at least 50% with one of these instances… **CorLoc is evaluated on the union of the training and validation subset.**"* It is blind to missed instances, to all false positives below the top-1 box, and to held-out generalisation. **Discount every CorLoc number you read, including the video ones below, by a large and unknown factor.**

#### 5.4d WSOL from image-level labels is formally ill-posed — and few-shot boxes beat it

**Choe et al., *Evaluating Weakly Supervised Object Localization Methods Right*, CVPR 2020** (arXiv 2001.07437). What it actually concluded, verbatim from the abstract `[SRC]`:

> *"**we argue that WSOL task is ill-posed with only image-level labels**, and propose a new evaluation protocol where full supervision is limited to only a small held-out set… We observe that, under our protocol, **the five most recent WSOL methods have not made a major improvement over the CAM baseline**. Moreover, we report that **existing WSOL methods have not reached the few-shot learning baseline**, where the full-supervision at validation time is used for model training instead."*

And: *"**The best overall improvements over CAM (63.8% total mean) is a mere +0.2pp boost by HaS.**"* `[SRC]`

| Method | ImageNet | CUB | OpenImages | **Total mean** |
|---|---|---|---|---|
| **CAM (2016)** | 63.5 | 68.8 | 59.1 | **63.8** |
| HaS | −0.1 | +1.9 | −1.3 | **+0.2** |
| ACoL | −1.4 | −0.6 | −0.7 | **−0.9** |
| SPG | +0.0 | −2.8 | −0.5 | **−1.1** |
| ADL | −1.4 | +2.0 | −0.9 | **−0.1** |
| **Few-shot baseline (5–10 boxes/class)** | 66.3 | **92.0** | 68.7 | **🏆 75.7** |
| Centre-Gaussian baseline | 52.5 | 59.7 | 45.8 | 52.3 |

All `[SRC]`. Also: they document hidden test-set tuning in prior work (*"HaS has found the operating threshold τ via 'observing a few qualitative results'… SPG has performed a 'grid search'"*), and their reproduction drops ADL CUB Top-1 Loc from a reported 52.4 to **39.2** `[SRC]`. Their formal statement: *"if background cues are more strongly associated with the target labels than some foreground cues, the localization task cannot be solved, even when we know the exact posterior"* `[SRC]` — i.e. the co-occurrence problem is not a bug to be engineered away.

**Did it hold up?** Partially. The narrow claim (erasing-style methods ≈ CAM) is robust — CALM (ICCV'21), the one protocol-faithful follow-up, gets +2.4 total `[SRC]`. The broad claim is **refuted on CUB** by BGC (CVPR'22, CUB 63.7→80.1) and BAS (CUB 61.1→85.2, ImageNet 62.4→68.6) `[SRC]`, and by SAT (ICCV'23, CUB MaxBoxAccV2 92.62) `[SRC]`. **But no post-2022 paper demonstrably re-ran Choe's held-out hyperparameter discipline** — they reuse his metric and his published CAM numbers `[EST]` — so criticism #2 may still apply to all of them. No 2024–2026 paper re-runs the protocol at all; the thread has gone quiet.

**Transformer-attention localisation** (CUB / ImageNet GT-known) `[SRC]`: CAM-VGG16 56.0 / 59.0 → TS-CAM 87.7 / 67.6 → LCTR 92.4 / 68.7 → SCM 96.6 / 68.8 → SAT 98.45 / 73.13 → GenPromp (Stable Diffusion) 98.0 / 74.9. Two warnings `[SRC]`: on ImageNet, **TS-CAM's GT-known (67.6) is *below* I2C's 68.5** — its headline gain is classification-driven; and naively applying CAM to a ViT **collapses** (TS-CAM's own "TransCAM" baseline: CUB 18.3 GT-known). Note also that **"attention rollout" (arXiv 2005.00928) is an NLP paper with no vision localisation numbers** — the vision numbers people cite are Chefer et al. CVPR 2021 (rollout ImageNet-Seg mIoU 55.42 vs Chefer 61.95) `[SRC]`.

**Maintained CAM code**, if you want a cheap CAM baseline: `jacobgil/pytorch-grad-cam` (**12,991★, last push 2026-08-13**, supports CNNs *and* ViTs, detection, segmentation, and CLIP) and `frgfm/torch-cam` (2,305★, pushed 2026-10-04) `[SRC]`. The original author repos are dormant.

#### 5.4e CLIP-based localisation: the blunt answer is that it does not produce evaluated boxes

**Across 14 training-free CLIP localisation papers checked programmatically (MaskCLIP, SCLIP, ClearCLIP, NACLIP, ProxyCLIP, CLIP-DINOiser, GEM, CorrCLIP, CASS, PEARL, OV-Stitcher, Trident, Talk2DINO, CLIP-ES), *zero* report a box metric (AP50 or box mAP)** `[SRC]`. CLIP-ES and Trident compute boxes internally (CAM connected components → enclosing rectangle, used as a SAM prompt) but never score them. **Do not let mIoU masquerade as detection quality.**

Training-free open-vocab **segmentation**, 8-benchmark average mIoU, CLIP ViT-B/16 `[SRC]`: naive dense CLIP 13.6 → MaskCLIP 30.3 → ClearCLIP 38.1 → SCLIP 38.2 → NACLIP 39.4 → CLIP-DINOiser 40.3 → ProxyCLIP 42.3 → Talk2DINO 43.8 → CASS 44.4 → Trident 45.8 → OV-Stitcher 50.7 → **CorrCLIP 51.0**. (±2 mIoU cross-paper reproduction variance; protocol-sensitive.)

Notable mechanics worth knowing `[SRC]`: **ClearCLIP** shows CLIP's **residual branch alone scores 0.01 mIoU** on COCO-Stuff vs 11.6 for the attention branch, and removing residual+FFN from the last block moves avg mIoU 27.3 → 37.5. **CLIP Surgery** quantifies CLIP's *inverted* attention (mean Score Contrast −27.9% to −14.0%, i.e. background scores *higher* than foreground, → +41.6 to +44.6% after surgery). **CLIP-ES** generates all VOC trainaug pseudo-masks in **0.6 h / 2 GB** vs AdvCAM's 77.2 h / 18 GB, reaching VOC val 71.1 mIoU.

**The only verified "frozen CLIP → boxes" numbers** are phrase localisation (arXiv 2204.03647) `[SRC]`: Flickr30k Acc@IoU **43.80 (frozen CLIP ViT-L/14) vs 63.39 (fully box-supervised ZSGNet) = 69%** `[calc]`. But that is one box per given phrase, single instance, no background rejection, no multi-instance NMS. **It is not detection.**

**CLIPSelf (ICLR'24) is the one that does report real box AP** `[SRC]`: OV-COCO novel AP50 17.5 → **37.6** (B/16), 24.7 → **44.3** (L/14); OV-LVIS mAP_r 11.5 → 25.3, 24.2 → 34.9. DeCLIP (2025) pushes novel AP50 to **43.3 / 50.2**. 🔴 **But the F-ViT detector they feed is trained with ground-truth boxes for the 48 OV-COCO base classes.** They remove the need for *novel-class* boxes, not for boxes.

#### 5.4f DINOv3 beats CLIP decisively for localisation — but only when frozen

This is a clean, important, and under-appreciated result. All `[SRC]` from the DINOv3 paper (arXiv **2508.10104**, 2025-08-13) and Talk2DINO (CVPR'25).

Matched ViT-L, matched sequence length, text-aligned, **zero-shot dense**:

| Model | IN1k | Img→Txt retrieval | **ADE20k mIoU** | **Cityscapes mIoU** |
|---|---|---|---|---|
| CLIP | 76.6 | 57.9 | **6.0** | **11.5** |
| SigLIP 2 | 83.1 | 71.4 | 10.8 | 16.3 |
| Perception Encoder | 83.5 | **75.9** | 17.6 | 21.4 |
| dino.txt (DINOv2) | 81.6 | 62.5 | 19.2 | 27.4 |
| **DINOv3 dino.txt** | 82.3 | 63.7 | **24.7** | **36.9** |

**DINOv3+text = 4.1× CLIP's zero-shot ADE20k mIoU while being *worse* at retrieval** `[calc]`. Global image-text alignment and dense localisation are nearly anti-correlated. Box-level object discovery (TokenCut on frozen features, CorLoc) `[SRC]`: **PEcore G/14 14.2, SigLIPv2 g/16 20.5 — vs DINO S/16 61.1, DINOv2 g/14 55.6, DINOv3 7B/16 66.1.** **CLIP-family backbones are 3–4.5× worse at box-level object discovery** `[calc]`.

Three caveats that change how you'd use this:
- **The DINO advantage is a *frozen* advantage.** With a mask-supervised decoder on top, the gap shrinks from +18.7 mIoU zero-shot to **+1.6** (`dinov3.seg`: CLIP ViT-L 48.83 vs dinov3.txt 50.44) `[SRC]`/`[calc]`. **If you are going to train a decoder anyway — and you are — backbone choice matters far less than the zero-shot tables suggest.**
- **CLIP's patch features are fine; its *alignment* is broken.** Talk2DINO's patch-level linear probe on VOC: **CLIP 0.89**, DINOv2 0.96–0.97, MAE 0.56 `[SRC]`. The information is there — which is exactly why attention surgery works with no retraining.
- **DINOv2 *regresses* vs DINOv1 on unsupervised object discovery** (VOC07 CorLoc 55.6 vs 61.1) despite much better linear probing, blamed on dense-feature artefacts `[SRC]`. **If you use DINO features for discovery, test v1 and v3, not just v2.** Note RF-DETR uses a DINOv2 backbone `[SRC]` — fine, because it is trained, not frozen.

#### 5.4g WSOL from *video*-level labels — directly this project's supervision

YouTube-Objects **CorLoc**, video-level tags only `[SRC]`:

| Method | Supervision | YTO v1.0 CorLoc | YTO v2.2 CorLoc |
|---|---|---|---|
| Grad-CAM++ | video-level tags | 63.2 | 61.2 |
| LayerCAM | video-level tags | 65.6 | 66.0 |
| **TCAM** (arXiv 2208.14542) | video-level tags | 73.0 | 72.2 |
| **TrCAM-V, CLIP pseudo-labels** (arXiv 2407.06018) | video-level tags | **84.8** | 76.7 |
| **TrCAM-V, CLIPSeg pseudo-labels** | video-level tags | 83.8 | **80.0** |

TrCAM-V = DeiT with **DINO pre-training**, localisation head trained on **frozen-CLIP Grad-CAM pseudo-labels**, plus a CRF loss; real-time per-frame inference. Repos: [TrCAM](https://github.com/shakeebmurtaza/TrCAM/), [TCAM](https://github.com/sbelharbi/tcam-wsol-video). **TrCAM-V has no stated venue — arXiv preprint, 8 Jul 2024** `[SRC]`.

🔴 **Do not over-read 84.8%.** YouTube-Objects is **10 classes with one dominant object per video**, and CorLoc scores only the single most-confident box, on trainval (§5.4c). This project is multi-class, multi-instance, cluttered, with small objects. `[EST]` Real mAP in this project's setting will be *far* below what 84.8 CorLoc suggests. The useful takeaway is the *architecture* — frozen-CLIP/CLIPSeg pseudo-labels + DINO-pretrained backbone + CRF — not the number.

#### 5.4h Open-vocabulary pseudo-label self-training — what the canonical examples actually did

| Method | Venue | Pseudo-label mechanism | Headline | Needs boxes? |
|---|---|---|---|---|
| **Detic** | ECCV'22 | **max-size proposal assignment**: assign all image labels to the *largest-area* proposal, not the highest-scoring one | 🏆 **24.6 vs 25.5 = 96.5% of fully-supervised novel-class LVIS mask mAP** `[calc]`; LVIS all-class 41.7 (Swin-B); cross-dataset "reaches 70–80% of dataset-specific oracles without using 1.8M annotations" | yes — 866 LVIS **base** classes with boxes |
| **OWLv2 / OWL-ST** | NeurIPS'23 | frozen OWL-ViT L/14 annotates **~10B WebLI** images with alt-text N-grams (≤10); 7 prompt templates ensembled; **keep boxes ≥0.1, keep images with ≥1 box >0.3** → ~2B examples | **LVIS APr 31.2 → 44.6 (+43% relative)** | 🔴 yes — the *annotator* was trained on Objects365 + Visual Genome human boxes; `+FT` adds LVIS_base |
| **GLIP** | CVPR'22 | teacher GLIP grounds parsed noun phrases on 24M web pairs → **78.1M phrase-box pseudo-annotations, 58.4M unique noun phrases** | zero-shot COCO 49.8, LVIS 26.9 | yes |
| **RegionCLIP** | CVPR'22 | LVIS-base RPN regions × 4,764-concept pool from captions, teacher-CLIP cosine match | OV-LVIS APr 22.0; OV-COCO novel AP50 43.3 | yes |
| **PB-OVD** | ECCV'22 | **ALBEF Grad-CAM** over cross-attention → max-activation proposal | OV-COCO novel AP **30.8** (vs OVR-CNN 22.8) | image-*caption* pairs only |
| **VL-PLM** | ECCV'22 | class-agnostic RPN + **CLIP crop scoring**, RoI head applied ~10× to refine; score = (S_RPN + max CLIP prob)/2; **τ=0.8** | OV-COCO novel AP **34.4**; **SSOD 1% COCO 15.35 vs 11.18** | base-class boxes |
| **DST-Det** | 2023 | CLIP zero-shot classifies would-be-*background* proposals as novel, **online, no offline relabelling** | COCO novel AP 46.7 with CLIPSelf | base-class boxes |
| **ALWOD** | **ICCV'23** | AL + WSOD: image-level labels on all + a tiny fully-boxed seed | 🏆 **VOC07 71.7 AP50 with 5% boxed**; COCO14 42.5 AP50 / 27.2 AP with **1% boxed** | 1–5% boxed |

**Detic's mechanism is worth stealing verbatim.** Instead of assigning image labels to the highest-scoring proposal (which needs good detections, which need labels — the chicken-and-egg), assign all image labels to the **largest-area** proposal: `L_max-size = BCE(W·f_j, c), j = argmax_j size(b_j)`. Measured justification `[SRC]`: on a hand-annotated subset of ImageNet-L, **max-size boxes cover 92.8% of target objects vs 69.0% for prediction-based assignment.** Also note Detic's own negative control: **naive self-training at threshold 0.5 *hurt* novel-class mAP (16.3 → 15.6)** `[SRC]` — a measured instance of the confirmation-bias failure in §3b.

**Engineering tool**: [Autodistill](https://github.com/autodistill/autodistill) — **2,784★, last push 2026-09-29, actively maintained** `[SRC]`. "Big slow foundation models train small fast supervised models." Base Model (GroundedSAM/Grounding DINO/CLIP) + Ontology → Dataset → Target Model (YOLOv8/DETR). No published benchmarks; it is plumbing, and this project's plumbing is most of the work.

**Also worth knowing: CutLER** (CVPR'23) — zero-*label* class-agnostic detector trained only on ImageNet MaskCut pseudo-masks: COCO val2017 **AP50 21.9 / AP 12.3**, VOC AP50 36.9, 11-benchmark average AP50 **24.3 vs FreeSOLO 9.0 (2.7×)** `[SRC]`. More useful as an *initialisation*: **+7.3 AP_box over MoCo-v2 at 5% COCO labels** `[SRC]`.

#### 5.4i 🏆 The annotation-budget number that should decide your strategy

**"Retrieve and Segment"** (arXiv 2602.23339, Feb 2026), same evaluation across supervision budgets `[SRC]`:

| Method | #annotated images | 6-dataset avg mIoU |
|---|---|---|
| LPOSS+ (training-free) | **0** | 33.2 |
| DINOv3.txt + SAM (training-free) | **0** | 27.9 |
| CAT-Seg (trained on COCO, 118k masks) | 118,000 | 47.8 |
| **+ RnS, B=1** | **66** | **🏆 49.9** |
| + RnS, B=20 | 964 | 61.9 |
| Fully supervised | 20,000 | **73.4** |

`[calc]`: **training-free open-vocabulary localisation sits at ~45% of fully-supervised quality, and 66 in-domain annotated images beat a model trained on 118,000 masks.** Put that number in front of whoever controls the labelling budget.

### 5.5 LVIS lab-equipment audit — I grepped the actual category file

I did not take this on trust. I fetched LVIS v1 train category metadata (1,203 categories) from `https://raw.githubusercontent.com/facebookresearch/Detic/main/datasets/metadata/lvis_v1_train_cat_info.json` and grepped it. All numbers below are **exact values from that file** `[SRC]`.

**What exists, and how little of it there is:**

| LVIS category | id | freq | train images | train instances | LVIS's own definition |
|---|---|---|---|---|---|
| `microscope` | 686 | **rare** | **2** | **3** | "magnifier of the image of small objects" |
| `fume_hood` (syn. `exhaust_hood`) | 565 | frequent | 193 | 208 | "metal covering leading to a vent that exhausts smoke or fumes" |
| `scale_(measuring_instrument)` | 919 | frequent | 158 | 178 | "a measuring instrument for weighing" |
| `syringe` | 1047 | rare | 5 | 14 | "a medical instrument used to inject or withdraw fluids" |
| `thermometer` | 1081 | common | 29 | 33 | — |
| `funnel` | 479 | rare | 9 | 9 | "cone-shaped utensil… into a container with a small mouth" |
| `dropper` (syn. `eye_dropper`) | 398 | **rare** | **1** | **1** | "pipet consisting of a small tube with a vacuum bulb" |
| `stirrer` | 1016 | rare | 8 | 18 | "an implement used for stirring" |
| `shaker` | 934 | common | 13 | 24 | "a container in which something can be shaken" |
| `cylinder` | 352 | **rare** | **1** | **3** | "a cylindrical container" |
| `lab_coat` | 620 | rare | 7 | 42 | — |
| `goggles` | 502 | frequent | 1530 | 3202 | — |
| `measuring_cup` | 680 | common | 71 | 139 | — |
| `tripod` | 1121 | frequent | 101 | 132 | — |
| `jar` / `bottle` / `dish` | 591/133/369 | frequent | 423 / 1901 / 106 | 2002 / 7969 / 532 | generic containers / dishware |

**What does not exist in LVIS at all** (confirmed by substring search over all 1,203 names) `[SRC]`:
`centrifuge`, `pipette`, `pipettor`, `beaker`, `flask`, `test_tube`, `petri` (dish), `bunsen` (burner), `autoclave`, `spectrophotometer`, `incubator`, `burette`, `vial`, `balance`, `colorimeter`, `thermocycler`, `desiccator`, `crucible`.

**Two conclusions, and the second one is the actionable one.**

**(a) LVIS gives you essentially nothing.** `microscope` has 2 images. `dropper` and `cylinder` have 1 image each. Six of this project's nine target classes are absent outright. Any claim that "LVIS covers lab equipment" is false. The same is almost certainly true of COCO (80 classes, none of them lab equipment) — and it means **every published open-vocabulary zero-shot number (LVIS AP 48.5 for SAM 3, 41.4 for MM-GDINO-T) tells you nothing about this project's classes**, because those classes are not in the benchmark. RF100-VL's 15.2 AP is the honest proxy, not LVIS's 48.5.

**(b) The LVIS categories that *superficially* match are domestic homonyms, and this is a concrete trap.** Read the definitions column:
- `fume_hood` is `hood.n.06`, "metal covering leading to a vent that exhausts smoke or fumes", synonym `exhaust_hood` — those 193 images are overwhelmingly **kitchen extractor hoods**, not laboratory fume hoods.
- `scale_(measuring_instrument)` is bathroom and kitchen scales, not analytical balances.
- `shaker` is "a container in which something can be shaken" — salt and cocktail shakers, not **orbital** shakers.
- `dropper` is an **eye** dropper, not a micropipette.
- `dish` is "dishware normally used as a container for holding or serving food", not a **petri** dish.

So the pre-training priors of every open-vocabulary detector point the bare words at the wrong objects. **This directly determines the prompt engineering:**

| Do not prompt | Prompt instead | Put in the hard-negative list `H` |
|---|---|---|
| "fume hood" | "laboratory fume hood", "chemical fume hood sash" | kitchen extractor hood, range hood, cooker hood |
| "balance", "scale" | "analytical balance", "precision laboratory balance" | bathroom scale, kitchen scale, postal scale |
| "shaker" | "orbital shaker", "laboratory shaker platform" | salt shaker, cocktail shaker |
| "dish" | "petri dish", "culture dish" | dinner plate, bowl, saucer |
| "flask" | "Erlenmeyer flask", "conical laboratory flask", "volumetric flask" | hip flask, thermos, vacuum flask |
| "pipette" | "micropipette", "laboratory pipettor" | eye dropper, syringe, pen |
| "cylinder" | "graduated cylinder", "measuring cylinder" | gas cylinder, tin can |
| "microscope" | "optical microscope", "benchtop laboratory microscope" | telescope, camera on tripod |
| "centrifuge" | "benchtop laboratory centrifuge" | washing machine, rice cooker, bread maker |
| "autoclave" | "laboratory autoclave steriliser" | pressure cooker, oven, microwave |
| "spectrophotometer" | "UV-Vis spectrophotometer", "benchtop spectrophotometer" | printer, desktop computer, plate reader, thermocycler |

This single table is probably worth more to day-one accuracy than any of the self-training machinery, and it is free. It also explains why the hard-negative evidence (SAM 3 IL_MCC **0.44 → 0.68** with hard negatives `[SRC]`) matters so much in *this* domain specifically: the confusables are not obscure, they are the dominant meaning of the word.

### 5.5b Objects365 audit — also grepped directly

I grepped the Objects365 v1 and v2 class lists from `https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/datasets/objects365.py` (376 and 375 captured class strings respectively, consistent with 365 classes each). All `[SRC]`.

**This matters more than LVIS**, because Objects365 is in the pre-training mix of Grounding DINO, MM-Grounding-DINO (O365 + GoldG + V3Det `[SRC]`), GLIP and the DINO baselines used throughout the SSOD tables. Whatever is a *boxed* Objects365 class, these detectors genuinely know how to localise.

| Target class | Objects365 v1 | Objects365 v2 |
|---|---|---|
| microscope | **`microscope`** ✓ | — absent |
| flask | — absent | **`Flask`** ✓ |
| balance/scale | `scale` ✓ | `Scale` ✓ |
| fume hood | — | **`Extractor`** ✓ (i.e. kitchen extractor again) |
| measuring cup | `measuring cup` ✓ | — |
| centrifuge, pipette, beaker, test tube, syringe, thermometer, funnel, autoclave, spectrophotometer, shaker, petri dish | **absent from both** | **absent from both** |

Note the v1/v2 asymmetry: **`microscope` is in v1 but not v2; `Flask` is in v2 but not v1.** Which one your chosen detector saw depends on which O365 release its authors used — worth checking per-model rather than assuming `[EST]`.

**The silver lining: Objects365 hands you a ready-made, box-level-grounded hard-negative list.** These are all real O365 classes the detectors localise well, and they are precisely this project's confusables `[SRC]`:
`printer`, `coffee machine`, `kettle`, `washing machine / drying machine`, `oven`, `microwave`, `refrigerator`, `gas stove`, `computer box`, `laptop`, `projector`, `sink`, `faucet`, `fire extinguisher`, `trophy`, `globe`, `tripod`, `binoculars`, `bottle`, `glove`.

Use exactly these strings in `H`. Hard negatives drawn from classes the model has strong box priors for are far more effective than invented phrases, and this is what SAM 3's IL_MCC 0.44 → 0.68 result is measuring `[SRC]`.

### 5.6 Other public datasets and licences

**The single most important dataset finding, and it changes the plan: ~22,000 public, CC BY 4.0, box-annotated lab-apparatus instances exist.** The project is *not* at zero boxes for the glassware/apparatus half of its vocabulary. These were verified from figshare/Zenodo/Mendeley/DataCite APIs and paper full-text.

| Dataset | What | Size | Relevant classes | Licence | URL / DOI |
|---|---|---|---|---|---|
| **Chemistry Lab Image Dataset, 25 Apparatus Categories** | real lab photos, YOLO boxes | **4,599 imgs / 6,960 inst** | `Beaker` 395, `Conical_Flask` 406, `Glass_Rod` 398, `Round_Bottom_Flask_1/2/3_Neck` 362/267/263, `Calorimeter` 338, `Separating_Funnel` 315, `Burette_Stands` 302, `Volumetric_Pipet` 280, `Pipette` 259, `Reagent_Bottle` 259, `TestTube_Holder` 256, `Test_Tube` 255, `Buchner_Funnel` 247, `Mechanical_Balance_Scale` 245, `Precision_Weight_Scale` 244, `Volumetric_Flask` 223, `Measuring_Cylinder` 210, `Wash_Bottle` 207, + 6 more | **CC BY 4.0** | [10.6084/m9.figshare.29110433](https://doi.org/10.6084/m9.figshare.29110433) · paper [10.1038/s41597-025-05952-3](https://doi.org/10.1038/s41597-025-05952-3) |
| **Physics Lab Equipment Image Dataset** | real lab photos, boxes + polygons | **5,121 imgs / 8,700 inst / 27 classes** | class names `[UNVERIFIED]` (189 MB `.rar`, not extractable in the research environment) | **CC BY 4.0** | [10.6084/m9.figshare.30984658](https://doi.org/10.6084/m9.figshare.30984658) |
| **Annotated Chemical Apparatus Image Dataset** (Sasaki/Fujinami/Nakai) | **frames from smartphone VIDEOS of chemical experiments**, YOLO boxes | **5,078 imgs**, 4.82 GB | 6 apparatus classes **+ experimenter's hands**; exact names `[UNVERIFIED]` | **CC BY 4.0** | [data.mendeley.com/datasets/8p2hvgdvpn/1](https://data.mendeley.com/datasets/8p2hvgdvpn/1) · [10.1039/D4DD00015C](https://doi.org/10.1039/D4DD00015C) |
| **HeinSight4.0** | **video-derived** frames, boxes + YOLOv8 weights | **6,031 phase + 6,523 vessel imgs** | vessel detector "to enhance detection across a variety of laboratory equipment and setups"; phase classes `Empty/Residue/Homogeneous Layer/Heterogeneous Layer/Solid` | data+weights **CC BY 4.0**, code **MIT** | [zenodo.org/records/15605098](https://zenodo.org/records/15605098) · [gitlab.com/heingroup/heinsight4.0](https://gitlab.com/heingroup/heinsight4.0) |
| **LabPics / Vector-LabPics** | instance + semantic **segmentation** of vessels and material phases | V1 2,187; V2 ~7,900 total | `Beaker, RoundFlask, Erlenmeyer, Cylinder, Burete, SeparatoryFunnel, Funnel, Pipette, Syringe, Tube, Vial, Dish, ChromatographyColumn, Condenser, HeatingVessel, Jar, Bottle, …` | ⚠️ **CONFLICT: Zenodo says MIT, project page says CC BY-NC-ND 4.0**; images partly scraped from Twitter/YouTube/C&EN | [zenodo.org/record/4736111](https://zenodo.org/record/4736111) · [cs.toronto.edu/chemselfies](https://www.cs.toronto.edu/chemselfies/) |

`[SRC]` for all sizes/licences above. Caveats that matter:
- **These are saturated, low-diversity benchmarks.** The chemistry-25 paper reports **mAP@50 = 0.992 with RF-DETR**, 0.987 YOLOv11, 0.92 YOLOv12 `[SRC]`. Near-saturation on a 25-way task means single institution, four phone cameras, mostly isolated objects on clean benches. Expect a large drop on cluttered, motion-blurred, occluded video `[EST]`.
- All three main sets are **Roboflow exports** (640×640 letterboxed, `.rf.<hash>` filenames, 70/20/10 random split). Check for train/test leakage across near-duplicate frames before trusting any number you reproduce.
- The **two video-derived sets (Sasaki, HeinSight4.0) are the most valuable** — they are the only lab data with real video statistics (motion blur, hands, occlusion). **Neither has track IDs.**
- **LabPics is unusable commercially until the licence conflict is resolved in writing.** ND would arguably forbid training on it.

#### Big-vocabulary datasets: the independently-verified counts

| Dataset | Lab coverage | Licence |
|---|---|---|
| **LVIS v1.0** | see §5.5 — `microscope` 3 inst; `fume_hood` is a **kitchen range hood** (`hood.n.06`); six target classes absent | annotations **CC BY 4.0**, images under COCO terms |
| **Objects365** | see §5.5b — `microscope` (v1), `Flask`/`Scale`/`Extractor` (v2) | 🚨 **NO EXPLICIT LICENCE FOUND** on objects365.org `[SRC]` — and it is in *every* open-vocab detector's pre-training mix, which transitively clouds those weights' provenance |
| **Open Images V7** (9.01M train imgs, 601 boxable classes) | **boxes:** `Beaker` **168 boxes / 86 imgs + 158 masks**, `Syringe` 127, `Scale` 139, `Measuring cup` 74, `Medical equipment` 2,060, `Tripod` 1,446, `Goggles` 9,636. **image-level only:** `Laboratory equipment` 323, `Microscope` 234, `Laboratory flask` 154, `Petri dish` 143, `Test tube` 117, `Autoclave` 69, `Graduated cylinder` 53, `Pipette` 52, `Bunsen burner` 30, `Centrifuge` 26 | annotations **CC BY 4.0**, images **CC BY 2.0** with explicit no-warranty on per-image status |
| **ImageNet-21k (Winter21)** | 🏆 **17 lab synsets, ~14,400 images** — the single biggest lab-equipment image resource that exists: `lab coat` n03630383 1,365; `autoclave` n02758960 1,277; **`microscope` n03760671 1,218**; `incubator` n03567066 1,188; `analytical balance` n02708555 1,047; `beaker` n02815834 982; `Erlenmeyer flask` n03295246 958; `Petri dish` n03920288 951; `hot plate` n03543254 901; `bunsen burner` n02921029 794; `graduated cylinder` n03451711 766; `crucible` n03140126 668; **`spectrophotometer` n04272928 628**; **`centrifuge` n02995998 514**; `burette` n02921592 128; `pipet/pipette` n03947111 81 | 🚨 **ImageNet terms: non-commercial research/education only** |
| **ImageNet-1k** | `beaker`, `lab coat`, `measuring cup`, `Petri dish`, `syringe` are 1k classes — so **every standard backbone already has discriminative features for these five** | ImageNet terms |
| **Visual Genome** (108,077 imgs, 2.52M objects) | `microscope` 4, `flask` 16, `glassware` 32, `beaker` 10, `funnel` 9, `thermometer` 30, `syringe` 2, `petri dish` 1, `fume hood` 1; **zero** centrifuge/pipette/autoclave/test tube | **CC BY 4.0** |
| **V3Det** (ICCV'23, 13,204 categories, 1.36M train boxes) | best lab *vocabulary* of any big dataset: `beaker` 74, `syringe` 92, `stirrer` 67, `test tube` 61, `funnel` 27, `thermometer` 37, **`microscope` 14**. But **median 22 boxes/category and 5,868 of 13,204 categories have <20 boxes** — it is a vocabulary, not a training set | HF card `myownskyW7/V3Det` says **cc-by-4.0**; mirror declares none; web-crawled images with **no per-image provenance** |

**Zero public boxes exist, anywhere, for: `centrifuge`, `autoclave`, `orbital shaker`, `spectrophotometer`, and real laboratory `fume hood`** `[SRC]`. `pipette` and `analytical balance` exist only inside the chemistry-25 dataset. **These five classes must come from your own data, ImageNet-21k crops, or synthetic.** That is a plain finding and it should drive where the labelling budget goes.

#### Which released weights have actually seen these classes as boxes

This is the most actionable model-selection result in the whole report `[SRC]`:

| Class | Seen as human-drawn boxes by a released detector? |
|---|---|
| `beaker`, `syringe`, `scale`, `measuring cup` | **yes** — via Open Images → Grounding DINO-**B**, MM-GDINO-L |
| `microscope`, `test tube`, `funnel`, `thermometer`, `stirrer` | **only via V3Det** ⇒ **only the MM-Grounding-DINO `*_v3det` / `pretrain_all` checkpoints** |
| `centrifuge`, `pipette`, `petri dish`, `autoclave`, `orbital shaker`, `spectrophotometer`, `analytical balance`, real `fume hood` | **never, by any released detector.** Zero-shot hits come purely from CLIP/caption-text generalisation — prompt-engineer and *measure*, never assume |

⇒ **`grounding_dino_swin-b_pretrain_all` / `swin-t_pretrain_obj365_goldg_v3det` (MM-Grounding-DINO, Apache-2.0) is the best-grounded zero-shot starting point for this vocabulary**, zero-shot COCO 50.4–53.0 `[SRC]`. Use it alongside SAM 3, not instead of it — they have genuinely different training mixes, which is exactly what makes Stage E (cross-model agreement) meaningful here.

#### Model/weight licences — one of these will bite you
| Model | Licence | Verdict |
|---|---|---|
| Grounding DINO, **MM-Grounding-DINO**, OWLv2 (`google/owlv2-base-patch16-ensemble`), **RF-DETR**, Detectron2 | **Apache-2.0** `[SRC]` | ✅ use freely |
| GLIP | MIT `[SRC]` | ✅ |
| **SAM 2 / 2.1** | **Apache-2.0, code AND weights** `[SRC]` | ✅ |
| **SAM 3 / 3.1** | **SAM License** — commercial OK, derivatives under same terms, ITAR/weapons/nuclear/espionage excluded `[SRC]` | ⚠️ workable, needs legal read |
| **YOLO-World** | 🚨 **GPL-3.0** `[SRC]` | ❌ copyleft by default |
| **Ultralytics YOLOv5/v8/v11/v12** | 🚨 **AGPL-3.0** `[SRC]` | ❌ — note *every* YOLO baseline in the lab-dataset papers is AGPL-encumbered, so reuse their **numbers** but not their **weights** |

#### Egocentric / procedural datasets, ranked by licence (the binding constraint)
| Tier | Dataset | Licence | Boxes? | Notes |
|---|---|---|---|---|
| 🟢 commercial-clean | **IndustReal** | **Apache-2.0** (code *and* data) | ✅ COCO-format assembly-state detection | 84 videos, 27 participants, HoloLens 2 |
| 🟢 | **HoloAssist** | **CDLA-Permissive-2.0** | ❌ | 169 h, 350 instructor-performer pairs |
| 🟢 | **EgoObjects** | **MIT** | ✅ LVIS-format | **114K frames, 9K+ videos, 14.4K unique instances, 368 categories**, each instance in ~44.8 images |
| 🟡 bespoke agreement | **Ego4D** (3,600 h) / **Ego-Exo4D** (1,286 h) / **PACO-EGO4D** | signed EGO4D agreement; **terms `[UNVERIFIED]`** | ✅ via VQ2D / PACO | Ego-Exo4D **Health** domain = 397 takes / 114.5 h, the only large procedural clinical corpus |
| 🔴 non-commercial | **EPIC-KITCHENS-100** (100 h, 90k segments, 97 verbs/300 nouns) + **VISOR** (271K masks); **Assembly101** (4,321 videos, 18M hand poses) | **CC BY-NC 4.0** | VISOR ✅; Assembly101 ❌ | method is reusable, data is not |
| ⚫ unknown | **MECCANO** (**64,349 active-object boxes / 20 classes, 89,628 hand boxes**, 12 fps ego), **EgoProceL**, ATTACH, HA4M | **none stated** `[SRC]` | MECCANO ✅ | shame — MECCANO is the closest annotation schema |

**None of these contain lab equipment.** Their value is methodological:
- **EgoObjects is the closest *task shape*** — instance-level detection from wearable video, the same physical object under ~44.8 views, MIT. If your 150 clips are mostly one or two labs, **train instance-level, not just category-level**; that is the single best idea to steal.
- **MECCANO is the closest *annotation shape*** — active-object + next-active-object + hand boxes on 12 fps egocentric tool manipulation.
- **EPIC-KITCHENS is the reference for narration-derived labels** (20,000 narrations → 97 verbs / 300 nouns → 90,000 segments). If your clips have any spoken or written protocol text, copy this pattern — it is a second free supervision channel this brief did not consider.

#### Synthetic — the most under-exploited option
**Objaverse-LVIS** asset inventory, counted from the official `lvis-annotations.json.gz` `[SRC]`: **47 `microscope`, 67 `cylinder`, 63 `tripod`, 59 `glove`, 56 `goggles`, 27 `lab_coat`, 22 `fume_hood`, 19 `syringe`, 14 `thermometer`, 12 `funnel`, 11 `scale_(measuring_instrument)`, 11 `measuring_cup`**.

47 microscope meshes × ~1,000 random poses/lighting/backgrounds = **47,000 synthetic microscope boxes, versus the 21 real microscope boxes in LVIS + Visual Genome + V3Det combined** `[SRC]`. That ratio is the headline. Licensing is unusually clean: whole dataset **ODC-By v1.0**, per-object licences enumerated (**721K CC-BY 4.0 + 3.5K CC0** commercially usable; 77K NC/NC-SA to filter out) with a per-object `license` field to filter on `[SRC]`.
- Limitation: Objaverse-LVIS inherits the LVIS taxonomy, so there is **no bucket** for beaker/flask/test tube/petri dish/centrifuge/pipette. Assets probably exist in the un-annotated ~750K remainder but that search was not done — **`[UNVERIFIED]` for those six classes.**
- Objaverse-**XL** is murkier: `odc-by` on the card but the Polycam subset is academic-non-commercial-by-request.
- `[UNVERIFIED]`: lab assets in NVIDIA Omniverse / Isaac Sim, BlenderKit, Sketchfab. The only sim-adjacent lab evidence found is **AutoBio** (18 MIT-licensed HF datasets, MuJoCo+Blender, task names referencing an Eppendorf Centrifuge 5430, pipette, thermal cycler, thermal mixer on a UR5e) — but **no paper, project page or asset repo could be found** across arXiv/OpenAlex/Semantic Scholar/GitHub, so treat its provenance as unverified.

#### Plainly-stated gaps
1. **No public lab-equipment video dataset with boxes *and* track IDs exists.** Nothing examined has tracking annotations at all. Your 150 clips may genuinely be first-of-kind. ⇒ Budget to annotate tracks yourself, and **prefer a detect-then-track pipeline so you only ever need per-frame boxes.** This independently confirms the §6.4 deployment choice.
2. **Roboflow Universe could not be crawled** — `universe.roboflow.com` returns HTTP 403 behind Cloudflare, `api.roboflow.com` needs a key. So the common assumption "there are lots of CC BY 4.0 lab-equipment projects on Roboflow Universe" is **`[UNVERIFIED]`**. All three main lab datasets were annotated and exported from Roboflow, so near-duplicate mirrors almost certainly exist. **A 20-minute sweep with an API key is worth doing and is the cheapest open action in this report.**
3. Four prior lab benchmarks are cited in the literature but **could not be found downloadable**: Cheng et al. 2023 (4,072 imgs / 7 classes), Ali et al. Sensors 2022 (481 imgs / 4 lab-PPE classes), Ding et al. 2022, Zou et al. MTAP 2024.
4. `[UNVERIFIED]`: physics-27 class names, Sasaki's 6 class names, EgoObjects' 368-category list, Ego4D licence text, Objects365 licence (appears not to exist), MECCANO/EgoProceL licences.
5. Correction to a common mis-citation: the ImageNet microscope synset is **n03760671**; `n03666591` is `lighter, light, igniter`.

---

## 6. Costed plan on 2× RTX 3060 12 GB

Pool assumption restated: **5,000 unlabelled clips × 30 s @ 25 fps, decoded at 2 fps ≈ 750k frames.** Lab equipment is static or slow, so 2 fps is plenty for mining; tracking runs at native fps only inside accepted segments.

### 6.1 The role split that makes this fit in 12 GB

Do **not** use one model for everything. Three models, each chosen for its licence and its footprint:

| Model | Params | Role | Licence | VRAM on a 3060 | Verdict |
|---|---|---|---|---|---|
| **SAM 3 / 3.1** | **848M–0.9B** `[SRC]` | teacher: *naming + detection* at low frame rate | **SAM License** (commercial OK, restricted) `[SRC]` | ~2 GB bf16 weights + 4–6 GB activations @1008² ⇒ **~6–8 GB** `[EST]` | **inference only** |
| SAM 3 **full fine-tune** | 0.9B | — | — | fp32 master 3.6 + AdamW states 7.2 + grads 3.6 = **~14.4 GB before activations** `[EST]` | **does not fit** |
| **SAM 2.1 hiera-small** | **46M** `[SRC]` | *propagation / tracking* through the clip | **Apache-2.0 (code AND weights)** `[SRC]` | ~1–2 GB `[EST]` | **yes — use this, not SAM 3, for propagation** |
| SAM 2.1 hiera-tiny / b+ / large | 38.9M / 80.8M / 224.4M `[SRC]` | " | Apache-2.0 | 1–5 GB `[EST]` | tiny if tight, large if you have headroom |
| **MM-Grounding-DINO Swin-T** | ~170M `[EST]` | 2nd teacher / Apache-only teacher | mmdet Apache-2.0 (weights licence `[UNVERIFIED]`) | ~3–4 GB `[EST]` | yes |
| **RF-DETR nano…large** | **30.5M–33.9M** `[SRC]` | **student (deployed)** | **Apache-2.0** `[SRC]` | ~2 GB inf; train at `batch_size=2–4, grad_accum=8–4, gradient_checkpointing=True` `[EST]` | **yes** |
| RF-DETR XL / 2XL | ~126M `[SRC]` | — | **PML 1.0** `[SRC]` | — | avoid (licence + cost) |
| ByteTrack / OC-SORT | negligible | deployed tracker | MIT `[UNVERIFIED]` | <0.5 GB `[EST]` | yes |

SAM 2.1 propagation quality is high enough to trust: **SA-V test J&F 76.5 / 76.6 / 78.2 / 79.5** for tiny/small/base+/large, at 91.2 / 84.8 / 64.1 / 39.5 FPS on an A100 `[SRC]`. A 3060 is roughly 1/8–1/12 of an A100 on this workload `[EST]` ⇒ **hiera-small ≈ 8–11 FPS on a 3060** `[EST]`. Param counts confirmed from the HuggingFace API: **38,963,010 / 46,060,866 / 80,850,690 / 224,447,154** `[SRC]`.

**What a box prompt costs you versus a perfect mask** (SAM 2 Table 4, zero-shot J&F across 17 video datasets) `[SRC]`: 1-click 64.7 · 3-click 75.3 · 5-click 77.6 · **bounding box 74.4** · ground-truth mask 79.3. So **a detector box is worth ~4.9 J&F less than a perfect mask** — an acceptable price, and the reason to prompt with boxes rather than trying to synthesise masks first.

**🔴 SAM 2 VRAM, derived from the source because no paper states it.** `image_size = 1024` and `num_maskmem = 7` (confirmed in `configs/sam2.1/sam2.1_hiera_s.yaml`). `load_video_frames` allocates `torch.zeros(num_frames, 3, 1024, 1024, dtype=torch.float32)` and, with `offload_video_to_cpu=False`, puts it **on the GPU** ⇒ **12.58 MB per frame of pure frame buffer** `[derived from code]`. That means **300 frames ≈ 3.8 GB and 1,000 frames ≈ 12.6 GB — OOM on a 3060 before the model even runs.** Per-frame state is retained for all processed frames; `maskmem_features` are cast to bf16 and moved to `storage_device`, which is CPU only if `offload_state_to_cpu=True`. Community reports (GitHub issues, **not peer-reviewed**): a 2:30 video @24 fps consumed **~60 GB, dropping to ~21 GB with `offload_video_to_cpu=True`** (issue #196); superlinear slowdown with length — an 8 s clip ≈1 s/frame versus a 40 s clip ≈1 min/frame (issue #180); *"inference_state grows endlessly with video length"* (issue #545) `[SRC]`.

**Mandatory 3060 recipe:** `offload_video_to_cpu=True`, `offload_state_to_cpu=True`, `torch.autocast("cuda", dtype=torch.bfloat16)`, and **chunk clips to ≤200–400 frames**. Or use the HuggingFace `transformers` **streaming** mode (`model(inference_session=..., frame=...)`) and never preload the clip at all. Forward *and* backward propagation are first-class: `propagate_in_video(..., reverse=False)` then `(..., reverse=True)` from the seed frame `[SRC]`.

**Cheaper propagators, all Apache-2.0, worth benchmarking against SAM 2.1-small** `[SRC]`:

| System | Params | Measured | Licence |
|---|---|---|---|
| **EfficientTAM-S** | 34M | **SA-V J&F 74.5 @ 85.0 FPS** vs SAM 2's 74.7 @ 43.8 FPS — **2× the throughput for −0.2 J&F**; DAVIS-17 89.2 (SAM 2 88.9) | Apache-2.0 |
| **EdgeTAM** | — | **150.9 FPS on A100, "22× faster than SAM 2"**; SA-V val/test 72.3/71.7, DAVIS-17 87.7 | Apache-2.0 |
| **Cutie** | — | MOSE val **68.5** (XMem 59.8) at similar runtime; DAVIS-17 val 90.8 | MIT |
| **XMem** | — | **1.4 GB GPU cap** with 10k long-term elements, 22.6 FPS; degradation 1×→3× video length only **−0.2 J&F** (STCN −2.7) | MIT |
| **SAMURAI** | — | SAM 2 + Kalman motion, zero-shot, **+7.1% AUC on LaSOT-ext** for single-object box tracking | Apache-2.0 |

⚠️ **Avoid for licence reasons: DEVA (CC-BY-NC-SA 4.0), SAM2Long (CC-BY-NC 4.0, though it is worth +3.0 J&F average and up to +5.3 on SA-V/LVOS), SAM-Track (AGPL-3.0), CoTracker/CoTracker3 (CC-BY-NC 4.0 — non-commercial, despite being the best point tracker)** `[SRC]`. If you need point tracking commercially, use **LocoTrack** (8.2M/11.5M, Apache-2.0, **7,244 points/sec on an RTX 3090**, DAVIS AJ 63.0) or **TAPIR/BootsTAPIR** (Apache-2.0 including weights) or **Track-On** (MIT, **16.8 FPS at ~400 points, 0.73 GB peak** — the only published point-tracker VRAM figure) `[SRC]`.

**If you add optical flow (SSVOD's filter needs it): use `neuflow2`.** PTLFlow's single-harness benchmark on an **RTX 3090** (same Ampere generation as your 3060, so memory transfers directly), 500×1000 input, batch 1 `[SRC]`: `neuflow2` **9.03M params, 12.1 ms fp16, 0.78 GB, KITTI-15 EPE 4.065** — versus `raft` 5.26M / 85.0 ms / 0.87 GB / EPE 5.018, and `flowformer` 16.17M / 182 ms / **3.48 GB** / EPE 4.534. **neuflow2 beats RAFT's accuracy at ~7× the speed.** Note `ccmr_p` needs **12.90 GB in fp32 and would not fit your card at all**. Licences: RAFT/SEA-RAFT **BSD-3-Clause**; GMFlow/FlowFormer/NeuFlow/RAPIDFlow **Apache-2.0**; UniMatch **MIT** — but ⚠️ **PTLFlow's own *retrained* weights are CC-BY-NC-SA 4.0**, so load the original authors' checkpoints `[SRC]`.

**Deployed tracker choice, measured on an identical frozen detector** (BoxMOT, MOT17-ablation, same YOLOX-x detections + same ReID for every row) `[SRC]`: BoT-SORT **69.68 HOTA** > BoostTrack 69.25 > StrongSORT 68.05 > **ByteTrack 67.68** > OC-SORT 66.44 > SFSort 62.65. So **BoT-SORT buys +2.00 HOTA over ByteTrack at the cost of a ReID pass, and ByteTrack beats OC-SORT by +1.24 HOTA at zero extra compute** — making ByteTrack the better motion-only default, contrary to common assumption. A cheap ReID net is available if you want BoT-SORT: `osnet_x0_25` is **0.59M params / 0.08 GFLOPs** at Market1501 R1 **91.2** `[SRC]`, ~60× cheaper in FLOPs than the benchmark default.

🚨 **The biggest licensing landmine in this report: `BoxMOT` is AGPL-3.0** (network copyleft — it reaches hosted services) `[SRC]`. `SORT` and `DeepSORT` are **GPL-3.0**. **ByteTrack, OC-SORT, BoT-SORT, Deep OC-SORT, Hybrid-SORT and BoostTrack are individually MIT** — so **vendor the individual MIT trackers from their own repos rather than depending on BoxMOT.** Use BoxMOT only as a local benchmarking harness you never ship.

**The decisive VRAM finding.** SAM 3's own 100-image fine-tuning recipe (`sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`) already runs at `train_batch_size: 1`, `resolution: 1008`, `gradient_accumulation_steps: 1`, `gpus_per_node: 2`, `max_epochs: 20`, bf16 AMP, AdamW, `lr_transformer: 8e-5`, `lr_vision_backbone: 2.5e-5`, `lr_language_backbone: 5e-6`, `wd: 0.1`, grad-clip 0.1, vision-backbone layer decay 0.9 `[SRC]`. Batch 1 at 1008² on 2 GPUs is the *minimum* configuration Meta ships, and they were not on 12 GB cards. **Full fine-tuning of SAM 3 on a 3060 is not viable** `[EST]`. A real user fine-tuning SAM 3 on a 5-class custom COCO-format dataset reports "**No object detected in inference after fine-tuning**" (sam3 issue #324) `[SRC]` — further reason to leave it frozen.

RF-DETR training guidance is explicit: "adjust `batch_size` and `grad_accum_steps` to maintain a total batch size of 16"; A100 → `batch_size=16, grad_accum=1`; **T4 (16 GB) → `batch_size=4, grad_accum=4`**; `gradient_checkpointing` available `[SRC]`.

### 6.2 Three hard operational constraints from real 3060-class reports

These come from the SAM 3 issue tracker and they reshape the plan. All `[SRC]` (user reports, so treat magnitudes as indicative):

1. **Per-concept cost dominates, not per-image cost.** The paper's "30 ms for a single image with 100+ detected objects" is on an **H200** `[SRC]`. A user on an **RTX 5070** measures **~0.36 s/image**, broken down as `set_image` 424 ms (first) → ~30 ms (cached) plus **~285–338 ms of prompt processing *per concept*** (sam3 issue #358) `[SRC]`. So an 8-concept query costs ~8× the prompt stage. A 3060 is ≈0.5–0.6× a 5070 `[EST]` ⇒ budget **~0.5–0.6 s per concept per frame** on a 3060, with `set_image` amortised across concepts.
   - **Consequence: the per-clip vocabulary restriction is a 2–3× *cost* win as well as an accuracy win.** Querying 3 classes instead of 8+20 negatives is the difference between a tractable and an intractable mining pass.
   - Corollary: run the hard-negative list `H` **sparsely** — on a 10% sample of frames, not every frame. Negatives are for calibration, not for every box.
2. **Video propagation memory grows with clip length and can be enormous.** A user reports "for a 200M video, the memory usage reaches **50 GB during propagation**, and a **3000-frame video causes OOM on A100 80 GB**" (sam3 issue #481) `[SRC]`; the video-predictor notebook OOMs on videos longer than the bundled example (issue #169) `[SRC]`. **You must chunk propagation** into short windows — I would use **50–150 frames (2–5 s) with 10-frame overlap** `[EST]` — and stitch tracklets across chunk boundaries by IoU. Be aware this loses identity: "some objects that have been detected in batch 1 are lost at the transition with batch 2, the tracking context is lost" (issue #354, explicitly from a consumer-GPU user) `[SRC]`. This is the main reason to propagate with **SAM 2.1-small instead of SAM 3** — smaller memory bank, Apache-2.0, and purpose-built for exactly this.
3. **Training/inference memory scales with the number of simultaneous classes**, because SAM 3 uses one query set per class: "if you scale to 100 classes in a single forward pass, the number of decoder queries becomes [excessive]" — mitigated by `category_chunk_size`, which the shipped config sets to **2** with the comment "You can increase this based on the memory of your GPU" (issue #476 and the config) `[SRC]`. On a 3060, keep `category_chunk_size` at 2 and keep the per-clip vocabulary small. **The class list being "open-ended and growing" is therefore a direct VRAM and latency liability** — one more reason the per-clip restriction is structural, not cosmetic.

Two further practical notes: there are **no smaller official SAM 3 variants** (issue #219, open request) `[SRC]`; and **SAM 3.1 has loading problems on public `main`** — `sam3.1_multiplex.pt` reportedly fails every builder entry point as of 2026-04-20 while the SAM 3.0 checkpoint loads and runs on the same install (issue #526), with a maintainer-adjacent "For anyone struggling with v3.1" thread pointing to the community `MuggledSAM` repo for lower-memory/CPU operation (issue #554) `[SRC]`. **Start on SAM 3.0, not 3.1.**

### 6.2b 🔴 The small-batch tax, and the only published 12 GB SSOD recipe

**Every SSOD paper in §5.1 trained on 8 GPUs; Unbiased Teacher's headline used 32.** The penalty for shrinking the batch is measured, not hypothetical:
- Unbiased Teacher README: 1% COCO, `32+32` on 4 nodes → **20.75 AP**; `16+16` on one node → **20.16 AP** (−0.6), with the explicit warning *"using the lower batch size leads to lower accuracy"* `[SRC]`.
- **UT v2 Table 5, same method: `8+8` → 21.84 vs `32+32` → 25.40 at 1% COCO — a −3.6 AP penalty for a 4× smaller batch** `[SRC]`.

Budget for this honestly: on 2× 3060 you will be at an effective batch of ~4, roughly 8× smaller than the papers' setups, so **expect to give up something like 3–5 AP purely to batch size** `[EST]`. Gradient accumulation recovers the statistics but not the BatchNorm behaviour, and it costs wall-clock.

**Consistent-Teacher is the only repo that explicitly ships a 12 GB configuration.** README verbatim: *"we support the `2x8` and `fp16` training setting to ensure everyone is able to run the code, even with only 12G graphic cards"* and *"With `8x2+fp16`, the total training time for MS-COCO is less than 1 day"* `[SRC]`. Its `2x8` deltas are exactly the co-adjustments to copy whenever you shrink a batch, verified verbatim from `consistent_teacher_r50_fpn_coco_180k_10p_2x8_fp16.py`:

```python
samples_per_gpu = 2 ; sample_ratio = [1, 1]      # vs 5 and [1, 4] in the 8-GPU config
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)   # LR HALVED
MeanTeacher(momentum=0.9998, interval=1, warm_up=0)                          # EMA RAISED
unsup_weight = 1.0                                                           # lambda_u HALVED
fp16 = dict(loss_scale=512.)
# measured result: 38.0 / 37.9 mAP at COCO 10%, vs 40.2 for the 8x5 config
```

**🔑 The rule: halve the LR, halve λ_u, raise the EMA momentum.** Cost of the shrink: **40.2 → 38.0 mAP, i.e. −2.2** `[SRC]`. Note `2x8` still means 8 GPUs × 2 = effective batch 16; on two cards you get 4, so expect a further hit.

**Reference EMA/mean-teacher hyperparameters actually used** (all read from the papers' own configs, `[SRC]`): EMA decay **0.9996** (Unbiased Teacher, Dense Teacher), **0.999** (Soft Teacher, PseCo, Semi-DETR), **0.9995** (Consistent-Teacher; 0.9998 in the 2×8 config). Burn-in: **2,000 iters** (UT), **5,000–20,000** (Dense Teacher), **0** (Soft Teacher, Semi-DETR; Consistent-Teacher instead forces λ_u = 0 until `warmup_step=10000`). λ_u: **4.0** (UT, Soft Teacher partial, Dense Teacher), **2.0** (STAC, Consistent-Teacher). Thresholds: **0.9** (STAC), **0.7** (UT), **0.5→0.9→0.9→0.02 variance** (Soft Teacher), **per-class GMM over a 100-score ring buffer** (Consistent-Teacher), **none — top k=1% of pixels** (Dense Teacher). Note two gotchas: UT/UTv2 set `STEPS=(179990,179995)` so **the LR never actually decays**, and UT v2 renormalises losses as `sup × 1/(λ+1)`, `unsup × λ/(λ+1)` so **λ_u=3 means a 0.25/0.75 split, not 1:3** `[SRC]`.

**Augmentation recipes, verbatim from the shipped code** `[SRC]`. Soft Teacher / Consistent-Teacher / PseCo share a byte-identical `rand_aug.py`:
```python
# WEAK (teacher):  RandResize([(1333,400),(1333,1200)], 'range') + RandFlip(0.5)
# STRONG (student): same resize+flip, then
ShuffledSequential([
  OneOf([Identity, AutoContrast, RandEqualize, RandSolarize, RandColor,
         RandContrast, RandBrightness, RandSharpness, RandPosterize]),   # 1 of 9
  OneOf([RandTranslate(x=(-0.1,0.1)), RandTranslate(y=(-0.1,0.1)),
         RandRotate(angle=(-30,30)), [RandShear(x=(-30,30)), RandShear(y=(-30,30))]]),
])
RandErase(n_iterations=(1,5), size=[0,0.2], squared=True)
```
Note **the teacher also gets large-scale jitter (short edge 400–1200)**, not just a flip — UT v2 adopted this and reported *"a significant improvement"*. Unbiased Teacher / Dense Teacher instead use a SimCLR-style photometric-only strong aug (ColorJitter 0.4/0.4/0.4/0.1 at p=0.8, RandomGrayscale p=0.2, GaussianBlur p=0.5, three stacked RandomErasing) with **no geometric ops at all**, applying the weak geometric transform first so **both views share identical geometry and no box re-projection matrix is needed** — a material implementation simplification over the Soft Teacher family. ⚠️ **Consistent-Teacher's shipped config appears to have the strong/weak branch tags swapped** relative to Soft Teacher/PseCo (open, unanswered [issue #36](https://github.com/Adamdad/ConsistentTeacher/issues/36)) `[SRC]` — replicate carefully.

**Architecture choice for 12 GB:** prefer the dense one-stage methods. Soft Teacher's box jittering costs 10 extra ROI-head forwards per retained box **plus** a second teacher ROI pass — material here. Consistent-Teacher is also **372 MB and 9–15 ms/image versus MixPL's 920 MB / 37–40 ms** `[SRC]`.

**🔑 And the calibration that should temper all of the above.** Zoph et al. measured that at **20% of COCO, ImageNet++ pre-training alone is worth +5.2 AP (30.7 → 35.9) while self-training adds only +1.3 on top** `[SRC]`. This project is *far* below 20% of COCO. **A strong pretrained/foundation backbone will out-earn every self-training trick in this report** — which is the deepest justification for the frozen-foundation-teacher architecture over an SSOD framework. Self-training's compensating virtue is that it is strongest exactly where pre-training fails: under heavy augmentation, Zoph measured **pre-training hurting by −1.0 AP while self-training gave +1.3** `[SRC]`. So use large-scale jitter + RandAugment *and* self-training.

Use **Zoph's loss normalisation** — it is what made their self-training stable `[SRC]`:
```
L̂ = 1/(1+α) · ( L_h + α · (L̄_h / L̄_p) · L_p )     # moving-average decay 0.9997
```

### 6.3 Stage-by-stage cost (revised with measured latencies)

Pool assumption restated: **5,000 unlabelled clips × 30 s @ 25 fps**. Mining frame rate is a free parameter and it is the single biggest cost lever. Lab equipment is static or slow, so **mine at 0.5 fps (≈15 frames/clip)** and propagate at native fps only inside accepted segments.

Unit cost used: **0.55 s per concept per mined frame on one 3060** `[EST]`, derived from the 5070 measurement above. Mean per-clip vocabulary **3 classes** `[EST]`.

| Stage | What runs | Volume | VRAM | Wall-clock, 2× 3060 |
|---|---|---|---|---|
| R0 prompt + gold set | human | 40 frames, ~350 boxes | — | **1–2 days human** |
| R0 decode + cache pool | NVDEC → JPEG | 5,000 clips | — | **1–2 days** (I/O bound) |
| R0 zero-shot baseline | SAM 3, 150 clips @0.5 fps, 3 concepts | 2.3k frames | 6–8 GB | **~35 min** |
| R1a auto-label 150 clips | SAM 3 PCS, per-clip vocab, @2 fps here (small set, afford it) | 9k frames × 3 | 6–8 GB | **~4 h** |
| R1b propagate | **SAM 2.1-small**, chunked, native fps in accepted segments | ~45k frames | 2 GB | **~1.5–3 h** |
| R1c train student v1 | RF-DETR-small, ~45k frames, 20 ep | — | ~10 GB | **8–14 h** |
| R2a clip multi-label classifier | frozen backbone + head, 150 clips | trivial | <4 GB | **<1 h** |
| R2b classify 5,000 pool clips | 8 frames/clip | 40k fwd | <4 GB | **~20 min** |
| R2c auto-label admitted pool (~2,000 clips) | SAM 3 PCS @0.5 fps | 30k frames × 3 | 6–8 GB | **~14 h** |
| R2d propagate | SAM 2.1-small | ~400k frames | 2 GB | **~1–1.5 days** |
| R2e train student v2 | RF-DETR-small, ~400k frames | — | ~10 GB | **2.5–4 days** |
| R3a ensemble pass, full pool | SAM 3 @0.5 fps + student v2 (student is ~100× cheaper) | 75k frames × 3 | 6–8 GB | **~1.5 days** |
| R3b propagate + mine hard negatives | SAM 2.1-small | ~750k frames | 2 GB | **2–3 days** |
| R3c train student v3 | — | ~10 GB | — | **3–5 days** |
| **Total R0→R3** | | | | **~3–4 weeks wall-clock**, both GPUs near-continuously `[EST]` |

What would blow this up, and the mitigation:
- Mining at 2 fps instead of 0.5 fps on the full pool: **4× the SAM 3 cost**, pushing R3a alone past a week. Don't. Static equipment does not need 2 fps.
- Querying 8 classes + 20 negatives every frame instead of 3: **~9× the SAM 3 cost**. Don't. Per-clip vocabulary; negatives on a 10% frame sample.
- **No NVLink** ⇒ run the two GPUs as *independent workers* (GPU0 mines, GPU1 trains) rather than DDP-ing anything large. DDP over PCIe is fine for a 30M-param student, pointless for SAM 3.
- Storage: ~750k JPEGs @ ~80 KB ≈ **60 GB** plus tracklet JSON ⇒ **budget 250 GB** `[EST]`.
- Power: 2× 3060 ≈ 340 W board; 4 weeks ≈ **~230 kWh** `[EST]`. That is the whole point of the hard cost constraint — a rounding error against any cloud equivalent.

### 6.4 What to deploy
**RF-DETR-nano or -small (Apache-2.0) + ByteTrack/OC-SORT.** Never deploy SAM 3: 0.9B params, measured ~0.36 s/image on a 5070 for *one* concept `[SRC]`, restricted licence, and the student is ~25× smaller. RF-DETR latency on a T4 with TensorRT FP16 is **2.3–17.2 ms** across the size range `[SRC]`; a 3060 is comparable or faster `[EST]`.

### 6.5 The all-Apache-2.0 fallback (make this decision early)
If the SAM License's "distribute derivatives only under the same terms" clause is unacceptable to your legal review, you can build the identical pipeline with no SAM 3 at all:
- teacher: **MM-Grounding-DINO-T** — zero-shot COCO **50.6 mAP**, LVIS MiniVal **41.4 mAP** `[SRC]`, trained on O365 + GoldG + V3Det
- propagation: **SAM 2.1-small**, Apache-2.0 `[SRC]`
- student: **RF-DETR**, Apache-2.0 `[SRC]`

You give up roughly **48.5 → 41.4 LVIS AP** of teacher quality `[SRC]`, SAM 3's presence head, and its image-exemplar prompting. My judgement: that is a real but survivable loss (**~10–20% relative pseudo-label quality** `[EST]`), and it is the right trade if the product ships commercially and you want zero licence ambiguity.

### 6.6 Licence notes that matter
- **SAM 3 / SAM 3.1: "SAM License"** — commercial use **is** permitted ("non-exclusive, worldwide, non-transferable and royalty-free limited license" to "use, reproduce, distribute, copy, create derivative works of, and make modifications"), with attribution for publications, no revenue/MAU threshold, and restrictions on reverse engineering and on ITAR/military/weapons/nuclear/espionage uses. Derivatives must be distributed under the same terms `[SRC]`. **For a lab-equipment product this is workable** — but note it is *not* Apache-2.0, and "derivative works under the same terms" means pseudo-labels-derived student weights deserve a legal read before shipping. If that risk is unacceptable, use MM-Grounding-DINO/OWLv2 as the teacher instead and accept lower quality (LVIS ZS 38.5–43.4 vs 48.5 `[SRC]`).
- **RF-DETR nano→large: Apache-2.0**; XL/2XL under PML 1.0 `[SRC]`. Stay in the Apache tier.
- SAM 3 requires **Python 3.12+, PyTorch 2.7+, CUDA 12.6+** `[SRC]`. sm_86 is supported by CUDA 12.6; this is fine on a 3060 but means a reasonably current driver.

---

## 7. Sources

### 7.1 Confirmed by a primary source (paper text, official repo, official config, or dataset API)

**Foundation models / teachers**
- SAM 3: *Segment Anything with Concepts*, Carion, Gustafson, Hu, Debnath et al. (Meta) — arXiv **2511.16719**, 2025-11-20 (v2 2026-03-28). https://arxiv.org/abs/2511.16719 · https://arxiv.org/html/2511.16719v2 · repo https://github.com/facebookresearch/sam3 · card https://huggingface.co/facebook/sam3 · licence https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE · SAM 3.1 notes https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md · 100-image fine-tune config https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
- SAM 3 real-world VRAM/latency reports: issues [#358](https://github.com/facebookresearch/sam3/issues/358) (RTX 5070 ~0.36 s/image, ~300 ms per concept), [#481](https://github.com/facebookresearch/sam3/issues/481) (~50 GB during propagation; 3000 frames OOM on A100 80 GB), [#354](https://github.com/facebookresearch/sam3/issues/354) (chunk-boundary identity loss on consumer GPUs), [#476](https://github.com/facebookresearch/sam3/issues/476) (`category_chunk_size`, memory scales with class count), [#324](https://github.com/facebookresearch/sam3/issues/324) (custom fine-tune produced no detections), [#219](https://github.com/facebookresearch/sam3/issues/219) (no smaller variants), [#526](https://github.com/facebookresearch/sam3/issues/526)/[#554](https://github.com/facebookresearch/sam3/issues/554) (SAM 3.1 loading problems; MuggledSAM workaround)
- SAM 2: *Segment Anything in Images and Videos*, Ravi et al. — arXiv **2408.00714**; repo + model table + Apache-2.0 https://github.com/facebookresearch/sam2
- Grounding DINO — arXiv **2303.05499**, Apache-2.0; https://huggingface.co/IDEA-Research/grounding-dino-tiny (172,277,902 params), `-base` (232,810,880)
- MM-Grounding-DINO — arXiv **2401.02361**; configs + zero-shot numbers https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino
- OWLv2 / OWL-ST: *Scaling Open-Vocabulary Object Detection*, Minderer, Gritsenko, Houlsby, NeurIPS 2023 — arXiv **2306.09683** · https://huggingface.co/google/owlv2-base-patch16-ensemble (154,966,791 params, apache-2.0)
- GLIP — arXiv **2112.03857** · YOLO-World — arXiv **2401.17270** (⚠️ GPL-3.0) · DINO-X, T-Rex2, Grounding DINO 1.5/1.6 Pro — API-only
- RF-DETR — https://github.com/roboflow/rf-detr (Apache-2.0 nano→large, 30.5–33.9M; PML-1.0 XL/2XL) · training guidance https://rfdetr.roboflow.com/learn/train/
- Perception Encoder backbone — Bolya et al. 2025 (cited by SAM 3)
- DINOv3 — arXiv **2508.10104**, 2025-08-13 · repo https://github.com/facebookresearch/dinov3
- EfficientTAM arXiv **2411.18933** (Apache-2.0) · EdgeTAM arXiv **2501.07256** (Apache-2.0) · SAMURAI arXiv **2411.11922** · Cutie arXiv **2310.12982** (MIT) · XMem arXiv **2207.07115** (MIT)
- ⚠️ Non-commercial: DEVA (CC-BY-NC-SA 4.0) · SAM2Long arXiv **2410.16268** (CC-BY-NC 4.0) · SAM-Track arXiv **2305.06558** (AGPL-3.0) · CoTracker / CoTracker3 arXiv **2410.11831** (CC-BY-NC 4.0)
- Commercial-safe point trackers: LocoTrack (Apache-2.0) · TAPIR / BootsTAPIR / TAPNext (Apache-2.0 incl. weights) · Track-On, ICLR 2025 (MIT, 16.8 FPS / ~400 points / 0.73 GB)

**SSOD**
- STAC arXiv **2005.04757** · Unbiased Teacher arXiv **2102.09480** + repo https://github.com/facebookresearch/unbiased-teacher (MIT; 1%/2%/5%/10% = 20.16/24.16/27.84/31.39) · Unbiased Teacher v2 (CVPR'22) · Soft Teacher arXiv **2106.09018** · Humble Teacher · Instant-Teaching (CVPR'21) · Dense Teacher arXiv **2207.02541** · PseCo (ECCV'22) · DSL (CVPR'22) · LabelMatch (CVPR'22) · ARSL (CVPR'23)
- Consistent-Teacher arXiv **2209.01589** + repo https://github.com/Adamdad/ConsistentTeacher (Apache-2.0; 1%/5%/10% = 25.50/36.60/40.20; full COCO 48.20; `2x8+fp16` 12 GB config and its 38.0 result; [issue #36](https://github.com/Adamdad/ConsistentTeacher/issues/36) strong/weak tag swap)
- Semi-DETR arXiv **2307.08095** · **Sparse Semi-DETR arXiv 2404.01819** (CVPR'24) — https://arxiv.org/html/2404.01819v1, the consolidated 1/5/10% comparison table and the 8× RTX A6000 / 2-day cost
- Omni-DETR arXiv **2203.16089** (CVPR'22) — https://ar5iv.labs.arxiv.org/html/2203.16089, the TagsU/TagsK ablation (34.7 vs 28.0) and the burn-in requirement · repo https://github.com/amazon-research/omni-detr
- MixPL arXiv **2312.07006**
- **Practical Insights into Semi-Supervised Object Detection Approaches**, Wang, Balasubramaniyam, Sangem, Guevara, Caragea — arXiv **2601.13380**, 2026-01-19 (the 1/5/50/150-shot tables; 4× A100)
- SSOD surveys: arXiv **2407.08460** (CNN→Transformer, 27 methods) · arXiv **2306.14106**
- DetMatch arXiv **2203.09510** · CrossRectify arXiv **2201.10734** · Instant-Teaching arXiv **2103.11402** · CPS arXiv **2106.01226** · SAS-Det (CVPR'24)

**WSOD / WSOL / weak supervision**
- WSDDN arXiv **1511.02853** · OICR arXiv **1704.00138** · PCL arXiv **1807.03342** · C-MIL arXiv **1904.05647** · MIST/Wetectron arXiv **2004.04725** (+ the ImageNet-VID weak-video benchmark and the instance-ambiguity ablation) · CASD arXiv **2010.12023** · SoS-WSOD arXiv **2106.04073** · W2N arXiv **2207.12104** · CBL arXiv **2308.05991** · SCEC arXiv **2505.16294** · OIM arXiv **2002.01087** · HUWSOD arXiv **2406.19394**
- **WeakSAM arXiv 2402.14812** (ACM MM 2024) — https://ar5iv.labs.arxiv.org/html/2402.14812 · repo https://github.com/hustvl/WeakSAM (the proposal-recall table and the 5,667 MiB / 9 h efficiency table)
- WSOVOD arXiv **2312.12437** (AAAI'24) · repo https://github.com/HunterJ-Lin/WSOVOD
- C-WSL arXiv **1711.05282** (ECCV'18) — counts as supervision, the oracle size-prior probe, and the 2×/38× annotation-cost figures
- **Choe et al., Evaluating Weakly Supervised Object Localization Methods Right**, CVPR 2020 — arXiv **2001.07437** · repo https://github.com/clovaai/wsolevaluation (MIT, unmaintained since 2020)
- CALM (ICCV'21) arXiv **2106.07861** · BGC (CVPR'22) arXiv **2204.00220** · BAS arXiv **2309.12943** · SAT arXiv **2303.10438** · TS-CAM arXiv **2103.14862** · GenPromp arXiv **2307.09756** · LayerCAM (IEEE TIP 2021) http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf · Chefer et al. arXiv **2012.09838**
- **Chéron, Alayrac, Laptev, Schmid, A flexible model for training action localization with varying levels of supervision**, NeurIPS 2018 — https://arxiv.org/pdf/1806.11328 (the supervision ladder: 43.9 / 66.8 / 74.5 / 76.0)
- Spot On arXiv **1604.07602** · STIL arXiv **1807.02800** · TCAM arXiv **2208.14542** · **TrCAM-V arXiv 2407.06018** (no stated venue) · CoLo-CAM arXiv **2303.09044** · VDST-Net arXiv **2407.15794**
- **Detic arXiv 2201.02605** (ECCV'22) — the max-size loss, the 92.8% vs 69.0% coverage measurement, and the naive-self-training negative control · repo https://github.com/facebookresearch/Detic
- DECOLA arXiv **2311.17902** — https://ar5iv.labs.arxiv.org/html/2311.17902 (class-conditioned pseudo-labelling, +13.4–17.2 c-AP@20) · repo https://github.com/janghyuncho/DECOLA
- RegionCLIP arXiv **2112.09106** · PB-OVD arXiv **2111.09452** · VL-PLM arXiv **2207.08954** (the threshold trap) · MarvelOVD arXiv **2407.21465** (3.3% vs 76.6%) · DST-Det arXiv **2310.01393** · CLIPSelf arXiv **2310.01403** · DeCLIP arXiv **2508.11256** · CutLER arXiv **2301.11320**
- ALWOD arXiv **2309.07914** (ICCV'23) · Point-Teaching arXiv **2206.00274** · Multi-Label Learning from Single Positive Labels arXiv **2106.09708** (CVPR'21)
- Constrained-CNN losses, Kervadec et al., *Medical Image Analysis* 2019 — https://arxiv.org/pdf/1805.04628 · *Bounding boxes for weakly supervised segmentation: global constraints get close to full supervision*, MIDL 2020 — arXiv **2004.06816** · CCNN arXiv **1506.03648** · *WSOD with Posterior Regularization*, BMVC 2014
- CLIP localisation: MaskCLIP arXiv **2112.01071** · CLIP-ES arXiv **2212.09506** · SCLIP arXiv **2312.01597** · ClearCLIP arXiv **2407.12442** · CLIP Surgery arXiv **2304.05653** · CorrCLIP, OV-Stitcher, Trident, CASS, PEARL, Talk2DINO (CVPR'25) · phrase localisation arXiv **2204.03647** · W-OoD arXiv **2203.03860** · CONTA arXiv **2009.12547**
- **Retrieve and Segment arXiv 2602.23339** (Feb 2026) — the 66-images-beat-118k-masks result

**Video self-training, propagation, and tracking**
- **Kalal, Mikolajczyk, Matas, Tracking-Learning-Detection**, IEEE TPAMI — http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf (P-N learning, the eigenvalue stability theory, Median-Flow)
- Misra, Shrivastava, Hebert, *Watch and Learn*, CVPR 2015 — arXiv **1505.05769**
- **RoyChowdhury et al., Automatic adaptation of object detectors to new domains using self-training**, CVPR 2019 — https://ar5iv.labs.arxiv.org/html/1904.07305 (the +0.95-of-+11.90 tracking ablation and the tracker-only regression)
- Jin, RoyChowdhury et al., *Unsupervised Hard Example Mining from Videos*, ECCV 2018 — https://ar5iv.labs.arxiv.org/html/1808.04285
- **SSVOD arXiv 2309.01391** (WACV 2024) — the strongest sparse-annotation video result; ⚠️ repo `github.com/enyacgroup/SSVOD` returns 404
- PseudoProp arXiv **2203.05983** (CVPR-WAD'22) · POPCat arXiv **2406.17183** · SA-VIS arXiv **2606.20140** · Label-Efficient Online Continual OD arXiv **2206.00309** · Stable Mean Teacher arXiv **2412.07072** (AAAI'25)
- *Autonomous Temporal Pseudo-Labeling for Fish Detection*, *Applied Sciences* 12(12):5910 (2022), doi **10.3390/app12125910**
- Offboard auto-labelling: 3DAL arXiv **2103.05073** · CTRL arXiv **2304.12315** · DetZero arXiv **2306.06023** · LabelFormer arXiv **2311.01444** · Auto4D arXiv **2101.06586** · Caine et al. arXiv **2103.02093**
- Seq-NMS arXiv **1602.08465** · Deep Feature Flow arXiv **1611.07715** · FGFA arXiv **1703.10025** · Detect to Track arXiv **1710.03958** · Weinzaepfel et al. arXiv **1506.01929**
- CMPL arXiv **2112.09690** · MvPL arXiv **2104.00682** · Deep Co-Training arXiv **1803.05984** · JoCoR arXiv **2003.02752** · WBF arXiv **1910.13302** (MIT, `pip install ensemble-boxes`)
- **Mcity Data Engine arXiv 2504.21614** (IEEE ITSC 2025, MIT) — the only public IoU-consensus-over-open-vocab-detectors implementation, and its 22×/26× per-checkpoint precision/recall spread
- *Robust & Label-Efficient Deep Waste Detection* arXiv **2508.18799** — consensus-weighted soft labels
- Noisy Student arXiv **1911.04252** · **Zoph et al., Rethinking Pre-training and Self-training**, NeurIPS 2020 — arXiv **2006.06882**
- FixMatch (NeurIPS 2020) · FlexMatch arXiv **2110.08263** · FreeMatch arXiv **2205.07246** · SoftMatch arXiv **2301.10921** · USB / `semilearn` https://github.com/microsoft/Semi-supervised-learning (MIT)
- PTLFlow benchmark https://ptlflow.readthedocs.io/en/latest/results/model_benchmark.html (RTX 3090, the only single-harness flow params/latency/VRAM table) · BoxMOT https://github.com/mikel-brostrom/boxmot (⚠️ AGPL-3.0) and its identical-detector tracker comparison
- Autodistill https://github.com/autodistill/autodistill (Apache-2.0, 2.8k★, actively maintained; **no published quality benchmarks**)

**Active learning, calibration, evaluation**
- MaxHerding / UHerding · TypiClust · ProbCover · Core-set (Sener & Savarese) · BADGE · Cluster-Margin · BatchBALD · LL4AL · MI-AOD · ALMDN · CALD · PPAL · CSVAL · Lüth et al. · Gupte et al. · Deuce · Mittal et al. *Parting with Illusions about Deep Active Learning* · Munjal et al. (CVPR'22) · **Oliver et al., Realistic Evaluation of Deep SSL Algorithms** (the ~20,000-examples-for-±1% result and the class-mismatch experiment)
- Active Testing arXiv **2103.05331** (ICML'21) · Reusable holdout, Dwork et al., *Science* 349(6248):636–638, doi **10.1126/science.aaa9375** · Recht et al. arXiv **1902.10811** · Mania et al. arXiv **1905.12580** · PPI arXiv **2301.09633** / PPI++ arXiv **2311.01453** · *Evaluating the Evaluators*, ICLR 2024 https://openreview.net/forum?id=kiwyQsZIGP
- Stabilizing Predictions arXiv **1409.5165** + bound/variance arXiv **1504.06329** + stop-set guidance arXiv **2201.05460** · AL stopping-criteria study arXiv **2110.03802** (*Machine Learning* 2024)
- ATC arXiv **2201.04234** · DoC arXiv **2107.03315** · Agreement-on-the-Line arXiv **2106.13799** + R² gating arXiv **2206.13089** · *Can you trust your model's uncertainty?* arXiv **1906.02530** · SKADA-Bench arXiv **2407.11676**
- **Box Stability (BoS) arXiv 2403.13803** (label-free detector mAP prediction, incl. its own four limitations) · PCR arXiv **2508.12082**
- Küppers et al. (detection calibration, COCO val2017 only — for Cityscapes/BDD D-ECE cite **Munir et al., CVPR 2023**) · Guo et al. (temperature scaling) · UPS arXiv **2101.06329** · CalibrateMix
- Arazo et al. *Pseudo-Labeling and Confirmation Bias* · DARP arXiv **2102.08473** · CReST arXiv **2102.09559** · Adsh · ABC · DebiasPL · ReMixMatch · DST (Debiased Self-Training) · Kumar et al. (gradual domain adaptation) · CascadeMatch · FineSSL
- **Northcutt, Athalye, Mueller, *Pervasive Label Errors in Test Sets*,** NeurIPS 2021 D&B — arXiv **2103.14749** · Picard, *torch.manual_seed(3407) is all you need* — arXiv **2109.08203** · Madaan et al. arXiv **2406.10229** · Maier-Hein et al. arXiv **1806.02051** · LVIS eval caps, Dave et al. arXiv **2102.01066**
- Annotation cost: Papadopoulos et al., *We don't need no bounding-boxes*, CVPR 2016 — arXiv **1602.08405** (6×–9×) · *Extreme clicking*, ICCV 2017 — arXiv **1708.02750** (7 s vs a 34.5 s AMT baseline) · Kuznetsova et al. (keyframes + interpolation, >50% time / 60% fewer boxes; uniform beats human-chosen keyframes)

**Datasets (counts verified by downloading and counting the authoritative file)**
- LVIS https://www.lvisdataset.org/ — categories grepped from https://raw.githubusercontent.com/facebookresearch/Detic/main/datasets/metadata/lvis_v1_train_cat_info.json (annotations CC BY 4.0)
- Objects365 https://www.objects365.org/overview.html — class lists grepped from https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/datasets/objects365.py (🚨 **no explicit licence found**)
- Open Images V7 https://storage.googleapis.com/openimages/web/factsfigures_v7.html (annotations CC BY 4.0, images CC BY 2.0)
- ImageNet-21k Winter21 https://image-net.org/download-images (🚨 non-commercial research/education only) · Visual Genome https://homes.cs.washington.edu/~ranjay/visualgenome/ (CC BY 4.0) · V3Det https://github.com/V3Det/V3Det
- **Roboflow100-VL arXiv 2505.20612** (NeurIPS 2025 D&B) · repo https://github.com/roboflow/rf100-vl (Apache-2.0 code)
- Chemistry-25 doi **10.6084/m9.figshare.29110433** + paper doi **10.1038/s41597-025-05952-3** (CC BY 4.0) · Physics-27 doi **10.6084/m9.figshare.30984658** (CC BY 4.0) · Sasaki et al. https://data.mendeley.com/datasets/8p2hvgdvpn/1 + doi **10.1039/D4DD00015C** (CC BY 4.0) · HeinSight4.0 https://zenodo.org/records/15605098 (CC BY 4.0 data, MIT code) · LabPics https://zenodo.org/record/4736111 + https://www.cs.toronto.edu/chemselfies/ (⚠️ **MIT vs CC BY-NC-ND conflict**)
- Egocentric: IndustReal https://github.com/TimSchoonbeek/IndustReal (**Apache-2.0**) · HoloAssist https://holoassist.github.io/ (**CDLA-Permissive-2.0**) · EgoObjects https://github.com/facebookresearch/EgoObjects (**MIT**) · PACO https://github.com/facebookresearch/paco · Ego4D https://ego4d-data.org/ and Ego-Exo4D https://ego-exo4d-data.org/ (bespoke agreement, terms unverified) · EPIC-KITCHENS-100 https://epic-kitchens.github.io/ + VISOR (CC BY-NC 4.0) · Assembly101 https://assembly-101.github.io/ (CC BY-NC 4.0) · MECCANO https://iplab.dmi.unict.it/MECCANO/ (no licence stated)
- Industrial anomaly (different task): MVTec AD / LOCO (CC BY-NC-SA 4.0) · VisA https://github.com/amazon-science/spot-diff (CC BY 4.0)
- Synthetic: Objaverse https://objaverse.allenai.org/ (ODC-By v1.0, per-object licences; counts from the official `lvis-annotations.json.gz`) · Objaverse-XL https://huggingface.co/datasets/allenai/objaverse-xl · AutoBio https://huggingface.co/datasets/autobio-bench/pipette-mujoco (MIT, **provenance unverified — no paper, project page or asset repo found**)

### 7.2 My own estimates and derivations (not from any source)
- All VRAM figures for a 3060 specifically, except where derived from code (SAM 2's 12.58 MB/frame frame buffer) or from a shipped config.
- The 0.55 s/concept/frame unit cost, extrapolated from the measured RTX 5070 figure.
- All wall-clock figures in §6.3; the 5,000-clip / 750k-frame pool assumption; storage and power.
- Every threshold, gate value and schedule constant marked `[EST]`: `τ_pres = 0.5`, `τ_k = 0.35` fallback, persistence lengths `L`, the 0.5 IoU and 40% re-fire gates, the per-class size-prior table in §2.2 Stage B2, the Stage-E score offsets, the ±30%/2× count triggers, the KL 0.3/1.0 cut-points, the κ and R² gates, the 7-pt ATC residual, the ≥30-instances-per-class probe requirement, the ±0.5–1.5 AP seed-variance working number.
- The whole of §8.2 (predicted mAP for this project) and the lab-equipment zero-shot interpolation between RF100-VL and ODinW13.
- The prompt-rewriting and hard-negative tables in §5.5 (derived from verified LVIS/Objects365 definitions, but the specific phrasings are mine).
- The architecture choices themselves: the three-model role split, the R0.5 public-data warm-start, R1.5 OOD filtering, the clip-classifier-supplies-the-vocabulary mechanism, and the forced-positive projection in Stage D.

### 7.3 🔴 Do NOT cite these — searched for and not found, or verified as folklore
- **A paper on mAP confidence intervals or required test-set size for object detection.** ~8 distinct searches, nothing on-topic. State the absence.
- **The "~0.2–0.4 AP detection seed variance" figure.** Not in Detectron2's MODEL_ZOO (which says only, qualitatively, *"large variance across different runs"*); no paper found.
- **"Mining Better Samples for Semi-supervised Video Object Detection"** — does not exist. The similar-sounding real paper is *Mining Better Samples for Contrastive Learning of Temporal Correspondence* (CVPR 2021), which is self-supervised correspondence, not detection.
- **"TCSSOD"** — not a real method name. **"DEFT"** detector-performance-estimation — zero hits. **"Is Self-Training Really Helping?"** — no such paper.
- **An established "ImageNet-VID with 10%/20% of frames labelled" benchmark** — does not exist.
- **Any quality benchmark for Autodistill / Roboflow auto-labelling.** Roboflow's "83% → 89% mAP with SAM 3" is an *augmentation* ablation, not a propagation gain.
- **Pseudo-box precision/recall for 1 vs 2 open-vocabulary detectors required to agree at IoU ≥ t.** Unmeasured anywhere. This is a half-day experiment and would be a novel contribution.
- **A direct port of FlexMatch CPL / FreeMatch SAT / SoftMatch soft weighting to 2D detection.** Four full-text searches, zero hits — appears genuinely novel as of 2026-10-04.
- **Any AL or data-selection work using open-vocabulary detector embeddings/scores**, and **any 2024–2026 low-budget AL paper running a class-balanced/stratified random control.** Both absent; the second is why the headline low-budget AL results should be treated as not independently verified.
- A **co-occurrence confusion matrix or per-class-pair error breakdown in any WSOD paper** — a real literature gap.
- **Per-variant VRAM for SAM 2** and **CoTracker VRAM as a function of (points, frames)** — only community issue reports exist.
- **3060-specific latency for anything in this report.** Every published timing is A100 / H200 / RTX 3090 / V100 / T4.

### 7.4 Internal inconsistencies and traps found in the sources themselves
- **HuggingFace cards for both `grounding-dino-tiny` and `grounding-dino-base` claim "52.5 AP COCO zero-shot".** That is the paper's **Swin-L** figure; the official repo gives **48.4** for Swin-T. Grounding DINO-B's 56.7 is **not zero-shot — COCO is in its training data.**
- **YOLO-World's LVIS numbers differ by ~8 AP between paper (35.0) and repo model zoo (26.8 @640 / 28.7 @1280)** under different protocols.
- **Dense Teacher is 19.64 in its own paper and 22.38 in every paper citing it.** The supervised 1% COCO baseline is variously quoted as 9.05, 10.02 and 11.24. **The field's reproduction band at 1% COCO is ±3 AP.**
- **Soft Teacher's box jitter: paper says uniform ±6%, code uses Gaussian σ=6%.** Its LR steps are 110k/160k in the paper, 120k/160k in the config.
- **UT v2's teacher-certainty floor is 0.5 in the paper and 0.8 in the code.** PseCo's λ_u is 4.0 in the paper and 2.0 in the config. Consistent-Teacher is 25.30 in the paper and 25.50 in the repo.
- **UT / UT v2 set `STEPS=(179990,179995)`, so the LR never actually decays.** UT v2 renormalises losses so λ_u = 3 means a 0.25/0.75 split, not 1:3.
- **Consistent-Teacher's shipped config appears to swap the strong/weak branch tags** relative to Soft Teacher and PseCo ([issue #36](https://github.com/Adamdad/ConsistentTeacher/issues/36), open and unanswered).
- **WSOVOD's COCO AP/AP50 ratio is 0.70**, higher than a fully-supervised R50-FPN (0.645) and far outside every other WSOD method's 0.48–0.51. Treat with suspicion.
- **ALWOD's claimed "fully supervised = 90.2 AP50 on VOC2007"** is anomalously high and its training split is unconfirmed.
- **MIST's per-class VOC07 row sums to 55.5 but is labelled 54.9.** Use 54.9.
- **MIST's "56.9% of Faster R-CNN" on COCO is against the 2015 VGG16 detector.** Against a modern R50-FPN it is **30.1%**.
- **The ImageNet microscope synset is n03760671**; the commonly-mis-cited `n03666591` is `lighter, light, igniter`.
- **`microscope` is in Objects365 v1 but not v2; `Flask` is in v2 but not v1.** Check which release your detector saw.
- **LVIS's `fume_hood` is `hood.n.06`, "metal covering leading to a vent that exhausts smoke or fumes", synonym `exhaust_hood`** — i.e. a kitchen range hood, not a laboratory fume hood. The same trap recurs in V3Det and Visual Genome.
- **Figure-only results (no numeric table exists in the source), so unverifiable:** TypiClust's headline percentages, ProbCover's entire results section, MaxHerding's classification results, Sener & Savarese's core-set numbers, BADGE's absolute accuracies, CALD's figures (no error bars at all), MI-AOD's SOTA figure, Unbiased Teacher Fig. 9 and Consistent-Teacher Fig. 8 threshold sweeps.
- **No standard deviations published at all:** CALD (3 trials), MI-AOD (5 runs), LL4AL per-budget, UHerding Table 1.
- **Snapper's abstract (39%, 95%) contradicts its own body §5.3.1–5.3.3 (33%, 32%, 59%).** Cite the body.
- **BoS reports its own vehicle RMSE as 2.25% while its successor PCR reports BoS at 6.94%** on a different meta-dataset.

---

## 8. Honest ceiling

### 8.1 Five measured anchors that bound what is achievable

**Anchor 1 — novel narrow domains are hard even for the 2026 SOTA.** On RF100-VL (100 detection datasets chosen for concepts *not* common in VLM pre-training), SAM 3 scores **15.2 AP zero-shot → 36.5 AP with 10 shots per class**; Grounding DINO-T **15.7 → 33.7** `[SRC]`. The RF100-VL paper's own headline is that GroundingDINO and Qwen2.5-VL "achieve **less than 2% zero-shot accuracy** on challenging medical imaging datasets" `[SRC]`. On ODinW13 (more ordinary objects) SAM 3 gets **61.0 → 71.8** `[SRC]`.

**Anchor 2 — video is substantially harder than images, and the *detector* is the bottleneck, not the tracker.** SAM 3 video PCS cgF1 vs human `[SRC]`:

| Split | SAM 3 | Human | SAM 3 as % of human |
|---|---|---|---|
| SA-V | 30.3 | 53.1 | 57% |
| YT-Temporal-1B | 50.8 | 71.2 | 71% |
| SmartGlasses (egocentric) | 36.4 | 58.5 | 62% |

And the ablations `[SRC]`:
- **SAM 3 Detector + plain tracking-by-detection: cgF1 25.7–47.6**, versus the full joint tracker's 30.3–50.8. Tracking-by-detection costs only ~3–4 cgF1. **This directly validates deploying a small student detector + ByteTrack rather than a joint tracker** — the architecture you can afford is close to the architecture you can't.
- **LLMDet + SAM 3's tracker: cgF1 2.3–8.0.** Swapping in a weaker detector destroys video performance even with the same tracker. Caveat: the weaker detector was plugged in without co-training, so this over-states the effect `[EST]` — but the direction is unambiguous. **Spend your quality budget on the detection/naming stage.**
- GLEE, one noun phrase at a time: cgF1 0.1–2.2 `[SRC]`.
- The SmartGlasses row is the closest published proxy for this project (egocentric, cluttered, object-rich indoor scenes): **36.4 cgF1, 62% of human** `[SRC]`.

**Anchor 3 — self-training gains are real but bounded, and they shrink as you improve.** Zoph et al. (NeurIPS 2020), RetinaNet + EfficientNet-B7, hard pseudo-boxes at threshold 0.5 `[SRC]`:

| Setup | 20% COCO | 50% | 100% |
|---|---:|---:|---:|
| Random init | 30.7 | 39.6 | 44.3 |
| + ImageNet self-training | **+3.4 → 34.1** | +1.8 → 41.4 | +1.3 → 45.6 |
| ImageNet init | 33.3 | 38.8 | 43.3 |
| + self-training | **+2.7 → 36.0** | +1.7 → 40.5 | +1.3 → 44.6 |
| ImageNet++ init | **35.9** | 39.9 | 43.8 |
| + self-training | +1.3 → **37.2** | +1.6 → 41.5 | +0.8 → 44.6 |

Self-training's gain grows as labels shrink (+1.3 → +1.8 → +3.4 going 100% → 50% → 20%) and is additive to pre-training — **but pre-training dominates in the low-data regime** (+5.2 AP from ImageNet++ init alone at 20%, versus +1.3 from self-training on top). Soft Teacher on full COCO + 123k unlabelled: 40.9 → 44.5 (+3.6) `[SRC]`. **Nobody gets a 2× from self-training once they are already in a decent place.**

**Anchor 4 — the video-level-labels-only ceiling is roughly 58% of fully supervised, measured twice, independently.**
- **Chéron et al. (NeurIPS 2018)**, UCF101-24 tubes: video-level labels **43.9 mAP@0.2 vs 76.0 fully supervised = 58%**; at the stricter IoU, **17.7 vs 50.1 = 35%** `[SRC]`.
- **MIST (CVPR 2020)**, ImageNet VID with 15 key frames per video and *frame-level category labels only* — the paper states it is *"the first to benchmark weakly supervised video object detection"*: **MIST + optical flow 46.9 AP (R-101) vs STMN fully supervised 80.5 = 58%**; VGG16 38.3 vs 61.7 = 62% `[SRC]`. Adding FlowNet2 feature warping to the weak model was worth **+1.2 to +1.7 AP** `[SRC]`.

Two independent benchmarks, two different tasks, the same ~58% at IoU 0.5-ish and a much worse ~35% at strict IoU. **That 58%/35% pair is the most defensible single statement of this project's ceiling without new boxes.** And the signature is consistent across all weak supervision: **AP50 survives, AP75 collapses** — MIST's COCO numbers are 12.4 AP / 25.8 AP50 / **10.5 AP75** `[SRC]`. You will get roughly-right boxes and badly-localised ones.

**Anchor 5 — CAM-based video localisation is not the route.** Weakly-supervised video object *localization* from video-level labels reaches YouTube-Objects CorLoc 82.1% (CoLo-CAM, *Pattern Recognition* 2025) and 84.8% (TrCAM-V) `[SRC]` — but that is single-dominant-object, top-1-box, trainval CorLoc (§5.4c). The honest comparison is VDST-Net on surgical video with **clip-level presence labels only: IoU 61.80% / Dice 67.80%**, versus TCAM's 28.22% IoU `[SRC]`. **Promptable detection seeded by your clip-level label list beats CAMs; do not build a CAM pipeline.**

### 8.2 My predicted numbers for this project

All of §8.2 is `[EST]`. Tagged by confidence.

| Quantity | Prediction | Confidence |
|---|---|---|
| **Clip-level multi-label accuracy** (what you can already do) — frozen image/video encoder + attention-pooled multi-label head on 150 clips | **mAP 0.70–0.85** for the common classes, 0.45–0.65 for rare ones | medium-high. Clip-level multi-label recognition is a *much* easier problem than detection and 150 clips is adequate for a linear/attention head on frozen features. I could not verify a directly comparable few-shot video multi-label number — `[UNVERIFIED]` on the exact figure. |
| **Detection AP50, common classes** (microscope, centrifuge, beaker/flask, pipette, balance) after R3, no new labels | **40–55 AP50** | medium |
| **Detection AP50, rare/confusable classes** (spectrophotometer, orbital shaker, plate reader, autoclave) after R3, no new labels | **10–25 AP50** | medium. These are grey boxes with screens. Text prompts cannot separate them; this is a genuine limit, not a tuning problem. |
| **Detection AP50:95, all classes, after R3** | **22–32** | medium-low |
| **Small-object AP** (pipette tips, test tubes in a rack, individual wells) | **5–15 AP** | medium. Sparse Semi-DETR's 1%-label small-object AP is **14.8** even on COCO with real labels `[SRC]`; small objects are where every low-label method dies. |
| **Tracking**, deployed student + ByteTrack, static equipment | **HOTA 45–60** | medium-low. Static objects track easily; the errors will be ID switches across occlusion by the operator's body. SAM 3's own BURST HOTA is **44.5** `[SRC]` as a sanity ceiling. |
| **Tracking**, handled items (pipettes, glassware in motion) | **HOTA 25–40** | low |
| **Pseudo-label precision after the full §2.2 cascade** | **0.85–0.93** at recall 0.4–0.6 | medium. The cascade is precision-oriented by design; you are deliberately trading recall away, and tracking propagation buys the volume back. |
| **With +50 verified-and-corrected clips** (§4) | **+5 to +12 AP50** on the classes you target | medium-high. Anchored on RF100-VL's 15.2 → 36.5 for 10 shots/class `[SRC]`, scaled down because you start from a better place than zero-shot. |
| 🏆 **With one box drawn on one frame of each of the 150 existing clips** | **+10 to +20 AP50** — the largest single jump available | **high**, and it is the one prediction I would defend hardest. Chéron et al. measured exactly this transition on video tubes: **43.9 → 66.8 mAP@0.2, i.e. 58% → 88% of fully supervised, from one box per video** `[SRC]`. |

**The defensible one-line summary of the ceiling, stated without new boxes:** two independent benchmarks measured video-level/image-level supervision at **~58% of fully-supervised performance at IoU 0.5 and ~35% at strict IoU** — Chéron et al. on UCF101-24 tubes (43.9 vs 76.0; 17.7 vs 50.1) and MIST on ImageNet VID with frame-level category labels only (46.9 vs 80.5) `[SRC]`. **Assume this project lands at 50–60% of what it would achieve with full box supervision, and that the shortfall is concentrated in localisation precision and small objects, not in "did it find the object at all".**

### 8.3 Where this pipeline will visibly fail
1. **Fine-grained near-duplicates.** Spectrophotometer vs plate reader vs thermocycler vs PCR machine: beige boxes with a lid and a small screen. Open-vocabulary text prompting has essentially no purchase here. **The fix is not more unlabelled video.** It is either (a) a few labelled exemplars per class — SAM 3's image-exemplar prompting gets **AP⁺ 76.8 on COCO from a single exemplar vs T-Rex2's 58.5** `[SRC]`, which is the strongest argument for exemplar-based few-shot over text; or (b) reading the instrument's label/badge with OCR, which for lab equipment is unusually effective and criminally underused.
2. **Small and occluded objects.** See the 5–15 AP row. No amount of self-training fixes small-object detection at this label budget.
3. **The absence assumption.** If clip annotators under-reported (listed only the salient equipment), the known-negative mining in Stage D will actively teach the student that real objects are background. **This is the most dangerous single failure mode in the design.** Mitigation is in §2.2 (ignore-regions for classes the clip classifier scores high but the human didn't list) — but you should also directly *audit* it: take 20 clips, exhaustively re-annotate presence, and measure the annotator's recall. If clip-label recall is below ~0.85 `[EST]`, **turn absence mining off** and rely on presence + temporal + agreement only.
4. **Chunk-boundary identity loss in propagation** (sam3 issue #354) `[SRC]`. Long tracks will fragment on a 12 GB card. Tracklet fragments are still useful training boxes; they are not useful for "how long was the centrifuge running" style downstream questions.
5. **The growing class list.** Every new class needs a new noun phrase, a new hard-negative set, and ideally 5–10 exemplars. The pipeline is designed to absorb this (nothing is retrained from scratch; add the class to the vocabulary, re-mine, fine-tune the student) — but the per-concept SAM 3 cost is **linear in vocabulary size** `[SRC]`, so a 50-class vocabulary makes full-pool mining ~17× more expensive than a 3-class one. Budget for per-clip vocabularies staying small.

### 8.4 What it would take to do materially better — ranked by measured value per pound

| Rank | Action | Measured basis | Cost |
|---|---|---|---|
| **1** | **Draw one box on one frame of each of the 150 clips.** | Chéron et al.: video-level **43.9 → one box 66.8 mAP@0.2, i.e. 58% → 88% of fully supervised** `[SRC]` | **~45 min** of verify-and-correct |
| **2** | **Use a foundation backbone and PEFT it; never train from scratch.** | FineSSL at 1 label/class: **48.45 → 96.15**, at 1/6 the training time. Zoph: ImageNet++ init alone is **+5.2 AP at 20% COCO** vs +1.3 from self-training `[SRC]` | **negative** — it is cheaper |
| **3** | **Hand-pick the clips for prototypicality rather than sampling them.** | FixMatch at 1 label/class: prototypical **78%** / middle 65% / **outliers 10% (chance)** — a **68-point swing** `[SRC]` | **~1 day** |
| **4** | **Filter the unlabelled pool for OOD content.** | UT: clean **37.2** → OOD-contaminated **34.5** (below the 35.6 labelled-only baseline) → filtered **42.7**, at 26% slower iterations `[SRC]` | **~1 day** |
| **5** | **~10 box-labelled exemplar *frames* per class** (not clips). | RF100-VL: SAM 3 **15.2 → 36.5 AP** from 10 shots/class `[SRC]` | **a few hours** |
| **6** | **Exemplar-based prompting instead of text for the confusables.** | SAM 3 single-exemplar **COCO AP⁺ 76.8 vs T-Rex2 58.5**; interactive 3 clicks **+21.6 cgF1 over text-only** `[SRC]` | hours |
| **7** | **Train on the ~22k public CC-BY lab-apparatus boxes** (§5.6). | ~14.8k images / ~22k boxes exist, free, permissive | **~2 days** of label-space mapping |
| **8** | **Distribution alignment + per-class rank thresholds + a balanced auxiliary head.** | DA worth **1.34 pts** (larger than every ReMixMatch component but augmentation); ABC worth **+18.2 on minority classes**; LabelMatch **+5.1 mAP at 1% COCO** over UT/Soft Teacher `[SRC]` | **~1 day total**, ~60 lines |
| **9** | **Post-hoc histogram-binning calibration of the teacher.** | Faster R-CNN D-ECE **19.235 → 0.890** `[SRC]` | **~2 hours** |
| **10** | **Counts on the 150 clips.** | **>2× cheaper than a centre click, >38× cheaper than a box** `[SRC]`; turns presence into a count constraint | **~1 hour** |
| **11** | **Synthetic renders from the CC-BY/CC0 Objaverse subset** for the five zero-public-box classes. | **47 microscope meshes vs 21 real microscope boxes in LVIS+VG+V3Det combined** `[SRC]` | ~1 week |
| **12** | **OCR on instrument front panels.** | none — `[EST]` | ~2 days |
| **13** | **Rent a big GPU for the teacher pass only.** | the teacher pass is the only genuinely compute-bound stage and is embarrassingly parallel | ~£100 |
| **—** | **More unlabelled video.** | OWL-ST's data scaling is **logarithmic** (100M→500M→2B gave 38→41→44.6 AP_rare) `[SRC]` | — |

**The pool is not the bottleneck.** Naming, fine-grained discrimination, and the absence of *any* box supervision are. Items 1–4 cost about three days and a day of annotation between them, and on the measured evidence they are worth more than items 5–13 combined.

### 8.5 Three non-obvious implications, stated once
1. **At this budget the acquisition function is a rounding error and the *identity of the seed examples* is the main lever.** Measured: 68 points from label prototypicality, 35–48 points from the backbone, 8.2 AP from OOD filtering, 30 points (58%→88%) from one box per clip — against **+1.2 to +1.5 mAP** for the best published acquisition function at detection budgets this small.
2. **Teacher EMA plus focal loss on the unsupervised branch is not a detail, it is the anti-collapse mechanism.** Unbiased Teacher: pseudo-label-distribution KL **1.7915 → 0.2001 → 0.0851** and 1% COCO mAP **13.42 → 17.85 → 21.19** across those two one-line changes `[SRC]`.
3. **Every newly-added class needs an explicit rare-class intervention on the round it is introduced, or it will be permanently starved.** Measured: CReST minority recall **8.4%** (at 97.7% precision); CascadeMatch — generic SSOD buys frequent classes **+2.5 APf** but rare classes only **+1.1 APr**; DST — **1.0%** pseudo-label accuracy on the bottom 20 classes `[SRC]`. Since this project's class list is *explicitly open-ended and growing*, build the intervention into the pipeline now: rank-based per-class thresholds, over-sampling at α ≈ 1/3, a balanced auxiliary head, **and a per-clip "is class c exhaustively labelled here?" flag** so a newly-added class does not silently poison your existing clips with false negatives. That last flag is the cheapest thing in this report and the easiest to regret omitting.
