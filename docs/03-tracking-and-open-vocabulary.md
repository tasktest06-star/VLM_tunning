# Tracking and open-vocabulary extensibility

> **Provenance and status.** Primary research output, 2026-10-04. Investigates multi-object tracking
> for static cluttered laboratory equipment, the mechanism for adding a class without retraining,
> the evidence that fine-tuning destroys open-vocabulary ability, and how to evaluate boxes and
> tracks when no box or track ground truth exists.
>
> **This document has not yet completed adversarial verification.** Numbers are tagged by the author
> as measured in a source, vendor-claimed, or estimated, and section 7 records an explicit
> correction trail plus the claims that could not be resolved. Where this document and
> `00-executive-summary.md` disagree, the summary reflects later checking and wins.
>
> Two load-bearing items are known to be uncertain and are called out in the summary: the CUDA
> graphs speedup rests on a single unreproduced report, and the recommended annotation budget
> conflicts two-fold with `04-unlabelled-pool-and-annotation.md`.
>
> **SUPERSEDED IN PART.** [`05-annotation-budget-verification.md`](05-annotation-budget-verification.md)
> overturns this document's annotation guidance. In particular the one-box-per-clip claim was traced
> to an arithmetic artefact, the recommended gold-set size is now 450 frames across all 150 clips,
> and several annotation-time and active-learning figures were misframed. Read document 05 before
> acting on anything in this file about how to spend annotation effort.
>
> Independently confirmed since writing: SAM 3 exists as described, `facebook/sam3` and
> `facebook/sam3.1` both exist and are gated, and the SAM License permits commercial use.

---

# Tracking + Open Vocabulary for Lab-Equipment Video (2x RTX 3060 12GB)

Research date: **2026-10-04**. Every number is tagged `[measured-in-source]`, `[vendor-claimed]`, or `[my estimate]`.

---

## 1. Headline recommendation

**Detector: SAM 3 image PCS (`facebook/sam3.1`) at 560px on keyframes (stride 8, error-triggered), with a cached per-frame vision embedding swept over a list of per-class text embeddings. Tracker: SAM 2.1-hiera-small or EfficientTAM-S at 512px with CUDA graphs, box-prompted from the detector, plus MASA for cross-keyframe association and a BoT-SORT-style IoU/GMC gate. bf16 throughout. Two processes, one per GPU. Full `Sam3VideoModel` end-to-end PCS reserved for the ~600-frame gold set.**

SAM 3 is the only open-weights system I found that natively emits exactly your required output — per-frame `boxes` (XYXY absolute) + `masks` + persistent `object_ids` + a `prompt_to_obj_ids` map — from an *open* vocabulary with **zero retraining**. Adding "spectrophotometer" is literally appending a string to a list; the HF API lets you compute that class's text embedding once (`get_text_features`) and cache it forever, and compute each frame's vision embedding once (`get_vision_features`) and sweep all N classes against it, so **cost scales with frames, not with vocabulary size**. It also dominates the exemplar-prompting alternative head-to-head: **COCO image-exemplar AP+ 76.8 vs T-Rex2's 58.5; LVIS 76.0 vs 65.8; ODinW13 82.2 vs 61.8** `[measured]` — and T-Rex2 is API-only anyway. Where SAM 3 is weak (**RF100-VL zero-shot 15.2 mAP vs 36.5 at 10-shot vs ~56 fully supervised** `[measured]`, on a benchmark that literally includes a *laboratory imaging* domain) nothing open-weights is better, so that ceiling is a property of your problem, not of the model choice.

**Cost is what forces the split architecture, and the reason is specific:** SAM 3's tracker *shares* the detector's Perception-Encoder-L backbone, so a propagated frame costs nearly as much as a detected frame. That is why end-to-end video PCS measures only **2.5 FPS @1 class and 0.7 FPS @40 classes on an H100, with <25 GB peak** `[measured, SAM3-ASH]` — and why a user hit CUDA-OOM in `propagate_in_video` on a **24 GB** 4090 `[measured]`. SAM 3.1 video is separately reported at **32 FPS on an H100**, which scales to **0.8-3.4 FPS on a 3060** — so end-to-end SAM 3 video over 150 clips is a **6-24 GPU-hour** job. Swapping the propagation stage to **SAM 2.1-small or EfficientTAM-S at 512px with CUDA graphs** (38-46M params vs 848M, Apache-2.0) and detecting only every 8th frame brings the same 150 clips to **~40-90 minutes** across the two cards (§6.2). The 512px + CUDA-graphs choice is deliberate and non-obvious: the one published sm_86 measurement shows **5.8x from kernel-launch elimination at 512px but only 1.4-1.9x at 1024px**, i.e. SAM 2 is launch-bound rather than compute-bound on this class of card. MASA earns its place because it is the only association model with published **novel-class** numbers (TAO **47.0 TETA base / 40.8 novel** vs OVTrack's 36.3 / 32.0 `[measured]`), it is Apache-2.0, it is trained only on unlabelled static images so it carries no pedestrian motion prior, and its head is "4 conv layers and 1 fully connected layer" `[measured]`.

Three hard "don'ts", each measured rather than stylistic:

1. **Do not fine-tune the detector on your 150 clips.** Closed-set fine-tuning on a narrow vocabulary is measured to take YOLO-World's COCO AP50 from **51.90 to 0.10**; head-only tuning gives **0.66**, i.e. equally catastrophic (§4).
2. **Do not rely on cross-image exemplars in SAM 3.** I checked the source — they are not supported. Plan on a DINOv3 prototype bank over SAM 3's class-agnostic proposals, or OWLv2's documented `image_guided_detection()`, for the "add a class from five photos" path (§3.5).
3. **Do not export a YOLO-World/YOLOE detector to ONNX/TensorRT.** The export **bakes the class vocabulary into the graph as constants, and YOLO-World does it silently** — you would ship something that looks open-vocabulary, passes tests on today's classes, and quietly ignores class #11 (§6.3).

---

## 2. Comparison tables

### 2a. Trackers / propagators

FPS is the *association/propagation* stage unless noted. "3060 VRAM" marked `[est]` is my scaling from the cited hardware.

| Model | Open weights | Licence | Needs per-class training | VRAM (12GB 3060) | FPS | Novel classes | Confidence |
|---|---|---|---|---|---|---|---|
| **SAM 3 / 3.1 video (PCS)** `facebook/sam3.1` | Yes (HF, **manually gated**) | **SAM License** (commercial OK, AUP + trade-control limits) | **No** | ~6-9 GB at 1008px, ~3-5 GB at 560px `[my est]`; <25 GB measured on H100 for long multi-class video `[measured]`; OOM reported on a 24 GB 4090 in `propagate_in_video` `[measured, issue #511]` | 2.5 FPS @1 class / 0.7 FPS @40 classes on H100, and **32 FPS on H100** for SAM 3.1 VOS `[measured]` → **~0.8-3.4 FPS on a 3060** `[my est]` | **Yes — text or same-image box exemplar** | High on capability, **medium on 3060 VRAM** |
| **SAM 2.1** (hiera-t → l) | Yes, ungated (`facebook/sam2.1-hiera-tiny` = 38.96M, verified) | **Apache-2.0** | No | ~2-3 GB (tiny) to ~5-7 GB (large) `[my est]` | 91.2 (t) / 84.8 (s) / 64.1 (b+) / 39.5 (l) FPS on A100 `[measured]` → **7-15 FPS (s) @1024px on a 3060; ~25-60 FPS @512px with CUDA graphs** `[my est]`. Cross-check: **17.7 FPS eager @512px for Hiera-S on an RTX A6000 (sm_86)** `[measured, single user report]` | Class-agnostic (needs an external detector/prompt) | High on licence/FPS-on-A100; **medium on the 3060 figure** |
| **EfficientTAM** (Ti / S) | Yes | **Apache-2.0** | No | ~1.5-2.5 GB `[my est]` | **Ti: 18M @ 96.2 FPS A100, SA-V 70.7 J&F. S: 34M @ 85.0 FPS, 74.5 J&F** `[measured]` | Class-agnostic | **High — cheaper than SAM 2.1-tiny at similar quality; the value pick for propagation** |
| **MASA** (R50 / Detic / G-DINO / SAM variants) | Yes (HF) | **Apache-2.0** | **No** — trained only on unlabelled SA-1B-500K static images | +~1-2 GB over the detector `[my est]` | Track head = 4 conv + 1 FC; no published FPS `[my est: >100 FPS association-only]` | **Yes — explicitly open-vocab** (TAO novel TETA 40.8) | High on claims, **low on FPS** |
| **ByteTrack** | Yes | **MIT** | No (motion only) | <0.3 GB `[my est]` | 29.6 FPS incl. YOLOX on V100 `[measured]`; association alone ~hundreds FPS `[my est]` | Yes (class-agnostic IoU) | High |
| **BoT-SORT / BoT-SORT-ReID** | Yes | **MIT** | No; ReID model is pedestrian-trained | <1 GB (ReID adds ~0.5 GB) `[my est]` | **6.6 / 4.5 FPS measured on an RTX 3060** (incl. YOLOX-X det.) `[measured-in-source]` | Yes for motion; **ReID embeddings are person-specific** | High (3060 is the paper's own GPU) |
| **OC-SORT** | Yes | **MIT** | No (pure motion) | <0.3 GB `[my est]` | ~28 FPS on 2080Ti incl. det.; **700 FPS association-only on CPU** `[measured]` | Yes | High |
| **Deep OC-SORT** | Yes | **MIT** | No; ReID is pedestrian-trained | <1 GB `[my est]` | Not published `[my est: ~BoT-SORT-ReID]` | Partly — ReID transfer is unvalidated for equipment | Medium |
| **StrongSORT / ++** | Yes | **GPL-3.0** ⚠ | No; BoT ReID pedestrian-trained | <1 GB `[my est]` | AFLink 591.9 Hz, GSI 140.9 Hz `[measured]` | Partly (same ReID caveat) | Medium |
| **Hybrid-SORT** | Yes | **MIT** | No; "training-free, plug-and-play" | <1 GB `[my est]` | 27.8 FPS on DanceTrack `[measured]` | Yes | Medium |
| **BoostTrack / BoostTrack++** | Yes (via BoxMOT) | AGPL-3.0 via BoxMOT ⚠ | No | <1 GB `[my est]` | Not published | Yes | Medium |
| **OccluBoost** (BoxMOT's top scorer, 71.10 HOTA) | Yes (BoxMOT) | **AGPL-3.0** ⚠ | No | <1 GB `[my est]` | Not published | Yes | **Low — no arXiv paper exists; unpublished** |
| **SAMURAI** | Yes | **Apache-2.0** | No ("zero-shot, we directly use SAM 2.1 weights") | ≈SAM 2.1 + Kalman `[my est]` | ≈SAM 2.1 `[my est]` | Yes | **Single-object VOT only** — needs N passes for N objects |
| **DAM4SAM** | Yes | **Apache-2.0** | No | ≈SAM 2.1 + distractor memory `[my est]` | Not published | Yes | **Single-object VOT only**; strong on distractors (DiDi robustness 0.944) |
| **Cutie** (small/base) | Yes | **MIT** | No (class-agnostic VOS) | ~2-4 GB `[my est]` | ~45.5 FPS on V100 (small) `[measured]` → ~20 FPS on 3060 `[my est]` | Class-agnostic | High |
| **XMem** | Yes | **MIT** | No | low — designed for "very long videos with limited GPU memory", 10,000+ frames `[vendor-claimed]` | ~20 FPS stated; 56.6 FPS on V100 in Cutie's table `[measured]` | Class-agnostic | High |
| **DEVA** | Yes | **CC-BY-NC-SA 4.0 — NON-COMMERCIAL + ShareAlike** ⚠⚠ | **No** — "class/task-agnostic bi-directional temporal propagation" | ~3-5 GB + image model `[my est]` | 7.8 FPS at defaults (M2F-R50) on V100 `[measured]` | **Yes** via Grounded-SAM; BURST OWTA 69.8, uncommon-class 53.3 `[measured]` | High (licence verified from LICENSE.md) |
| **OVTrack** | Yes (Google Drive) | **Apache-2.0** | No | ~4-6 GB `[my est]` | Not published | **Yes** — TAO 37.1 base / 28.8 novel TETA `[measured]` | Medium — superseded by MASA |
| **TRACT** (trajectory-aware OV-MOT) | **"Code will be released"** — not available | n/a | No | n/a | n/a | Yes (OV-TAO) | **Low — unreleased** |
| **SAM3-ASH** (arXiv 2610.01022, 1 Oct 2026) | **No release commitment** | n/a | No | <25 GB on H100 MIG `[measured]` | 2.5 / 0.7 / 0.017 FPS at 1 / 40 / 1196 classes `[measured]` | Yes | **Low availability, high informational value** |
| **CoTracker3** | Yes | **CC-BY-NC — non-commercial** ⚠ | No | not published | not published | Points only, **not boxes** | High on licence, low on fit |

**Verified benchmark anchors.** MOT17 test: ByteTrack 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA; BoT-SORT 80.5 / 80.2 / 65.0; Deep OC-SORT 79.4 / 80.6 / 64.9; Hybrid-SORT 79.3 / 78.4 / 63.6; OC-SORT 78.0 MOTA / 63.2 HOTA `[all measured-in-source]`. BoxMOT's own MOT17-ablation HOTA ranking: OccluBoost 71.10 > BoT-SORT 69.68 > BoostTrack 69.25 > StrongSORT 68.05 > DeepOCSORT 67.95 > ByteTrack 67.68 > HybridSORT 67.31 > OCSORT 66.44 > SFSort 62.65 `[measured-in-source]`. **The whole SORT family sits inside a 5-HOTA-point band. Pick on licence and camera-motion handling, not on HOTA.**

### 2b. Open-vocabulary detectors

| Model | Open weights | Licence | Needs per-class training | VRAM (12GB 3060) | FPS | Novel classes | Confidence |
|---|---|---|---|---|---|---|---|
| **SAM 3 image (PCS)** | Yes (gated) | SAM License | **No** | ~5-8 GB @1008px bf16; ~3 GB @560px `[my est]`; 848M params, fp16 weights ≈1.63 GB `[measured]` | 30 ms/img @100+ objects on H200 `[measured]` → **~0.5-1.0 s @1008px, ~0.2-0.3 s @560px on a 3060** `[my est, §6.0]`. Video: **32 FPS on H100** `[measured]` → **0.8-3.4 FPS on a 3060**. ⚠ one user measured **1.1 s on a 4090 via ONNX-Runtime TensorRT EP at FP32** `[measured — likely a broken export]` | **Yes — text AND same-image box exemplars** | High |
| **OWLv2** (`google/owlv2-base/large-patch14-ensemble`) | Yes, **ungated**, base = 155M (verified) | Apache-2.0 | **No** | ~2-5 GB `[my est]` | **~2-4 FPS** `[my est]`; latency/VRAM **not published anywhere** ⚠. Text-embedding pre-compute gains ≈0 at small vocabularies | **Yes — text + first-class `image_guided_detection()` one-shot API, genuinely cross-image** | High on capability, low on perf data |
| **OmDet-Turbo** (in HF transformers) | Yes | **Apache-2.0** | No | ~2-4 GB `[my est]` | **~14 FPS** `[my est]`, with a **built-in language cache** | Yes | **Medium — the most interesting *permissive* speed/accuracy middle ground, since YOLOE/YOLO-World are AGPL and OWLv2 is slow** |
| **MM-Grounding-DINO** (Swin-T/B/L) | Yes | Apache-2.0 | No | ~3-6 GB `[my est]` | not published `[my est: 5-10 FPS Swin-T]` | Yes — LVIS-minival 41.4 AP, COCO 50.6, ODinW13 0.533 `[measured]` | High |
| **Grounding DINO** (orig.) | Yes | Apache-2.0 | No | ~3-5 GB `[my est]` | not published | Yes — COCO ZS 48.4 AP `[measured]`; RF100-VL **15.7 ZS → 33.3 @10-shot** `[measured]` | High |
| **YOLO-World** S/M/L/X | Yes | **GPL-v3** ⚠ (repo says "supported for commercial usage") | No; offline vocab re-parameterization | ~1-3 GB `[my est]` | very fast `[my est: 30-80 FPS on 3060]` | Yes — LVIS AP 18.5 / 24.1 / 26.8 / 28.6 `[measured]` | High |
| **YOLOE-v8/11 S/M/L** | Yes | **AGPL-3.0** ⚠ | No | ~1-3 GB `[my est]` | **305.8 / 156.7 / 102.5 FPS on T4** `[measured]` → ~1.5-2x on 3060 `[my est]` | **Yes — text, VISUAL prompt (SAVPE), and prompt-free**; LVIS 27.9 / 32.6 / 35.9 AP `[measured]` | High |
| **YOLOE-26** (Ultralytics YOLO26 family, Jan 2026) | YOLO26 detection weights yes; **YOLOE-26 open-vocab checkpoints not confirmed** | **AGPL-3.0 + Enterprise (dual)** ⚠ | No | ~1-3 GB `[my est]` | YOLO26 det. **1.7-11.8 ms on T4 TensorRT** across n→x `[measured]` | **Yes — text, visual, and prompt-free**; **LVIS-minival 40.6 AP (text prompting, YOLOE-26x)** `[measured]` | **Medium — best speed/accuracy on paper, but AGPL and unconfirmed weights** |
| **OV-DEIM** (DEIMv2-based real-time OV-DETR, Mar 2026) | "Code and pretrained models are available" (github.com/wleilei/OV-DEIM) | **not stated** ⚠ | No | ~2-4 GB `[my est]` | not published | Yes — claims SOTA on OV benchmarks with "notable improvements on challenging rare categories"; **no numbers in the abstract** | **Low — unverified numbers and licence** |
| **Florence-2-large** | Yes | **MIT** | No | ~2-3 GB (0.77B) `[my est]` | not published | Partial — `<CAPTION_TO_PHRASE_GROUNDING>` only, no arbitrary class list; COCO ZS **37.5 mAP** `[measured]` | High |
| **Rex-Omni** (3B, Qwen2.5-VL-3B base) | Yes (`IDEA-Research/Rex-Omni`, +AWQ) | **IDEA License 1.0 + Qwen Research License** ⚠ (research-restricted) | No | ~7 GB bf16, ~4 GB AWQ `[my est]` | slow — autoregressive coordinate decoding `[my est: <2 FPS on 3060]` | Yes — text, box, point, visual reference | Medium |
| **T-Rex2** | **No — API only** | IDEA License 1.0 (code) | No | n/a | n/a | Yes, but SAM 3 beats it: image-exemplar AP+ COCO 58.5 vs 76.8, LVIS 65.8 vs 76.0, ODinW13 61.8 vs 82.2 `[measured]` | High — **rule out** |
| **DINO-X Pro** | **No — API only** | Apache-2.0 (client) | No | n/a | n/a | Yes — COCO ZS 56.0 AP, LVIS-minival 59.7 / rare 63.3 `[vendor-claimed]` | High — **rule out** (cost + no local) |

### 2b-bis. Licence triage (this turned out to be a bigger discriminator than accuracy)

| Tier | Components | Note |
|---|---|---|
| ✅ **Safe (permissive)** | SAM 2.1, **EfficientTAM**, MASA, OVTrack, SAMURAI, DAM4SAM, **MobileSAM** (**Apache-2.0**); ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, Hybrid-SORT, Cutie, XMem, **TrackEval**, **TransNetV2** (**MIT**); OWLv2, Grounding DINO, MM-Grounding-DINO, **OmDet-Turbo** (**Apache-2.0**); Florence-2 (**MIT**); **torchcodec, PyAV, PySceneDetect** (**BSD-3**); **PyNvVideoCodec** (**MIT**); **DALI** (Apache-2.0) | Build the whole stack from this tier if you can |
| ⚠ **Usable with care** | **SAM 3 / 3.1 (SAM License)** — commercial use explicitly permitted and royalty-free, no MAU cap, no restriction on using outputs to train other models, you own derivatives; but AUP + ITAR/trade-control clauses, publication-acknowledgement duty, and **HF weights are manually gated**. **DINOv3 (DINOv3 License, gated).** | Legal review needed; gating also breaks unattended CI |
| ⚠⚠ **Copyleft — contaminates your code** | YOLO-World (**GPL-v3**), StrongSORT (**GPL-3.0**), YOLOE + YOLO26/YOLOE-26 + BoxMOT (**AGPL-3.0**; Ultralytics offers a paid Enterprise licence), **SAM-Track (AGPL-3.0 + explicitly requires written permission for commercial use)**, **NeLux decode (AGPL-3.0)**, `video-keyframe-detector` (GPL-3.0), **EdgeSAM (NTU S-Lab License 1.0)** | Fine for internal research, a problem if you ship |
| ❌ **Non-commercial — rule out if there is any product path** | **DEVA (CC-BY-NC-SA 4.0 — verified from LICENSE.md; also wants Gurobi, a commercial solver)**, **CoTracker3 (CC-BY-NC)**, EOVSAM (contaminated by NVIDIA RADIO's "non-commercial research or evaluation" clause), **Panda-70M splitting code (Snap Inc., non-commercial research only)** | Note DEVA is NC while the same author's Cutie and XMem are MIT — easy trap |
| ❌ **No weights at all** | T-Rex2, DINO-X (API-only) | Also violates the low-cost constraint |

### 2c. Detector-plus-tracker vs promptable propagation

**(a) Per-frame open-vocab detector + motion/appearance tracker.**

- *Accuracy.* Bounded by per-frame detection. On specialist domains this ceiling is brutal: RF100-VL (100 datasets, 7 domains **including "laboratory imaging"**) zero-shot — Grounding DINO **15.7 mAP**, SAM 3 **15.2**, Gemini 2.5 Pro 13.3, Qwen2.5-VL-72B 5.4; at 10-shot — SAM 3 **36.5**, Grounding DINO 33.3; fully supervised YOLOv8m/v11m **56.4 / 56.5** `[all measured-in-source]`. Association adds little error for static objects.
- *Cost.* N classes x T frames. With a DETR-style detector this is the dominant cost. Mitigated by SAM 3's cached-embedding path (one vision forward, N cheap text heads) — the ASH paper measures **6.5x** throughput going from prompt-batch 1 to 40 `[measured]`.
- *Failure modes.* Flicker (independent per-frame decisions → boxes appear/disappear); ID fragmentation across occlusion; duplicate IDs when an object leaves and re-enters; and the detector's class decision can flip frame-to-frame on a fine-grained class (centrifuge vs shaker).
- *Verdict.* **This wins for static equipment in a mostly-static camera.** When objects do not move, IoU between consecutive detections is ~1.0, so even the dumbest association (ByteTrack, OC-SORT) is near-perfect. The remaining error is all detection, and you can fix it cheaply with temporal majority voting over the track.

**(b) Detect once on a keyframe, then propagate (SAM 2/3 VOS, MASA, Cutie/XMem/DEVA).**

- *Accuracy.* Propagation is excellent where it is well posed: SAM 2.1-large DAVIS-ish J&F 79.5 on SA-V test, Cutie-base DAVIS-2017 88.8, SAM 3 DAVIS 92.2 / YT-VOS19 89.7 `[measured]`. But it degrades hard under the exact conditions in a lab: **SAM 2 drops from 76.4 on MOSEv1 to 50.9 on MOSEv2** (-25.5), where MOSEv2 adds "frequent object disappearance/reappearance, severe occlusions and crowding, smaller objects" `[measured]`. SAM 3.1 recovers to **62.3** on MOSEv2 `[measured]` — still a ~26-point gap from DAVIS.
- *Cost.* One detector call per keyframe + cheap propagation. DEVA's recipe is the canonical one: run the image model **every 5 frames** with an n=3-frame in-clip consensus, 7.8 FPS at defaults on a V100 `[measured]`. SAM 3 video's own equivalent knob is `recondition_every_nth_frame=16` `[measured, HF config]`.
- *Failure modes.* Cannot find objects that appear after the keyframe without re-detection (Grounded-SAM-2's "continuous ID" script for exactly this is self-described as "still under develop" and "not that stable now" `[measured-in-source]`); memory-bank drift onto a visually similar neighbour (the problem DAM4SAM's distractor-aware memory targets, DiDi robustness 0.944 `[measured]`); unbounded VRAM growth on long clips.
- *Verdict.* **This wins for the moving handheld camera**, where position is not a reliable identity cue and mask-level propagation with a learned memory tracks through viewpoint change far better than a Kalman box.

**The benchmark mismatch, stated plainly.** ByteTrack/BoT-SORT/OC-SORT/StrongSORT/Hybrid-SORT are all tuned on MOT17/MOT20 — upright pedestrians, roughly linear motion, discriminative clothing, a camera that mostly pans. Lab equipment violates three of those four: it does not move at all, adjacent instances are often *identical* (a row of pipettes, two of the same centrifuge), and it is 60-80% occluded by hands, racks and other equipment. The nearest published analogue is **DanceTrack**, built precisely because "object appearance is not sufficiently discriminative", where the authors report "a significant performance drop... when compared against existing benchmarks" `[measured-in-source]`. On DanceTrack the motion-centric trackers pull ahead (Hybrid-SORT 62.2, Deep OC-SORT 61.3, OC-SORT 55.1 HOTA `[measured]`), which tells you **appearance ReID is not the answer for identical equipment.**

**Direct answer to "static bench + static camera vs moving handheld camera":**

| | Static equipment, mostly-static camera | Moving handheld camera |
|---|---|---|
| **What is hard** | *Detection only.* Boxes barely move, so consecutive-frame IoU ≈ 1.0 and association is almost free | *Both.* Position is no longer an identity cue; viewpoint, scale and motion blur all change |
| **What is easy** | Association. The correct motion prior is **zero velocity** — a Kalman filter with constant-velocity is already right, trivially | Nothing. This is the regime the MOT benchmarks actually cover |
| **Best architecture** | **(a) per-frame detector + trivial IoU association.** Or better still, the keyframe-cluster trick below | **(b) detect-then-propagate** with a learned mask memory (SAM 3 tracker, Cutie, XMem) |
| **Best tracker** | ByteTrack or OC-SORT (free, MIT, zero-velocity-correct) | **BoT-SORT** (GMC: measured **+0.94 HOTA / +1.62 IDF1** on MOT17-val) or SAM 3's own memory tracker or MASA |
| **Avoid** | Anything with a pedestrian ReID model — the appearance term is noise on identical equipment | ByteTrack and OC-SORT: **no camera-motion compensation at all** |
| **Dominant failure** | Detector flicker; two identical instances swapping IDs | Memory drift onto a visually similar neighbour (the DAM4SAM "distractor" problem) |
| **Detector stride you can afford** | Very long — 16-32 frames, or even 8 frames total per clip | Short — 5-8 frames (DEVA's default is every 5) |

**A cheap trick you should exploit, because your objects are static.** For a fixed-camera clip, a piece of equipment's box is *constant*. So: run the detector on K≈8 frames spread across the clip, cluster the resulting boxes spatially (greedy IoU ≥ 0.7 agglomeration), keep clusters supported by ≥⌈K/2⌉ frames, and emit each surviving cluster as a single track with a constant box and a majority-voted class. That is a complete, valid spatio-temporal output at **K detector calls per clip instead of 900**, it is robust to per-frame flicker by construction, and it beats any learned tracker on this sub-case because it encodes the correct prior (zero velocity). Detect camera motion first — mean optical-flow magnitude or the GMC homography's translation norm over a 1 s window — and route each clip to this path or to the full tracker. `[my own design; no source, but it follows directly from the geometry]` I would expect a meaningful fraction of lab footage (fixed tripod / wall-mounted / fixed overhead) to qualify.

**Which degrades least, ranked (my judgement, grounded in the above):**
1. **SAM 3 / SAM 3.1 video** — mask-level memory tracker, distractor-aware-ish via `suppress_overlapping_based_on_recent_occlusion_threshold=0.7` and detector reconditioning; best measured clutter score (MOSEv2 62.3).
2. **MASA** — class-agnostic appearance matching trained on *static-image* augmentations, so it was never tuned to pedestrian motion priors at all; the only tracker with published *novel-class* numbers.
3. **BoT-SORT** — because of GMC. Measured CMC contribution on MOT17-val: **+0.64 MOTA, +1.62 IDF1, +0.94 HOTA** `[measured-in-source]`. That gain will be substantially larger on handheld footage.
4. **DEVA / Cutie / XMem** — class-agnostic bi-directional propagation, cheap, long-video-safe; weaker at re-finding new objects.
5. **ByteTrack / OC-SORT** — perfect for static-camera static-equipment, no GMC so they fall apart on handheld.
6. **Deep OC-SORT / StrongSORT / Hybrid-SORT** — their ReID models are pedestrian-trained; on identical equipment the appearance term is noise at best, actively harmful at worst. (StrongSORT is also GPL-3.0.)
7. **SAMURAI / DAM4SAM** — excellent single-object trackers, wrong shape of problem (N objects → N passes).
8. **CoTracker3** — points not boxes, and **CC-BY-NC**.

---

## 3. Open-vocabulary extensibility mechanism — the concrete design

**Requirement: adding a class must need zero retraining.** Here is the design, in the order the data flows. Everything marked ✅ is confirmed from official docs/paper; ⚠ needs a one-hour experiment.

### 3.1 The class registry (this *is* the vocabulary)

A single JSON/YAML file. Adding `spectrophotometer` means appending one record — a name, a handful of prompt strings, a list of hard negatives, and (importantly) a few exemplar crops. Nothing is retrained, ever. **The full schema, revised in light of the measured evidence, is in §3.8** — read that version, not a simplified one, because two of the obvious fields (LLM attribute fragments; a global score threshold) turn out to be actively harmful.

### 3.2 Text-embedding channel (primary, truly zero-shot) ✅

SAM 3's HF API exposes exactly the two hooks you need:

- `text_embeds = model.get_text_features(**processor(text=prompt))` — compute **once per class**, cache to disk as a `.npy`. Adding a class = one forward pass of a CLIP text encoder (`Sam3VideoProcessor` wraps a `CLIPTokenizer` ✅). Zero gradient steps.
- `vision_embeds = model.get_vision_features(pixel_values=...)` — compute **once per frame**, then sweep all N cached class embeddings against it: `model(vision_embeds=vision_embeds, text_embeds=..., attention_mask=...)`. This decouples cost-per-frame from vocabulary size, which is the whole ballgame for a growing class list. (Same idea as ASH's "Generalized Presence Token", measured **6.5x** speedup from prompt-batch 1→40 `[measured-in-source]`.)
- For video, `processor.add_text_prompt(session, ["microscope","centrifuge","pipette", ...])` takes a **list** and the docs state the model "efficiently reuses vision features across all prompts" and "detects objects from ALL prompts in a single pass", returning `prompt_to_obj_ids` ✅.

### 3.3 Prompt engineering — do far less of it than you expect, and do the *right* part

I came into this expecting LLM-generated descriptors (CuPL/DCLIP-style) to be the highest-leverage cheap win. **The evidence says otherwise, and in detection it says the opposite.** Ranked by measured value:

**① Hand-disambiguate the class *names*. Highest ROI by a wide margin.** CLIP's own prompt-engineering notebook reports that hand-editing class names — `"nail"` → `"metal nail"` (CLIP read it as fingernail), `"kite"` → `"kite (bird of prey)"` — is worth **+1.5% top-1 on ViT-B/32**, with the author estimating *"another 0.5% to 1% top_1 could be gained"* `[measured-in-source]`. That is the same order as the entire 80-prompt ensemble, for an afternoon of work, and your vocabulary is exactly the kind that needs it. Write `"analytical balance (laboratory weighing scale)"`, `"orbital shaker (laboratory platform shaker)"`, `"fume hood (laboratory extraction cabinet)"`.

**② Check your string formatting. This is not a joke — it is a ~30-point effect.** The agrifood study reports verbatim: *"This step [replacing underscores with spaces] is **critical for beans: naive underscore names degrade accuracy by ~30 percentage points due to tokenization mismatch in the text encoder**"* `[measured-in-source]`. If your registry keys are `fume_hood`, render them as `fume hood` before tokenising.

**③ Ensemble templates — but a *small, curated* pool, not a big one.** More prompts is **not** monotonically better, confirmed three independent ways:
- ZPE (ICML 2023): a **247-prompt pool with uniform averaging scores 67.59 on ImageNet — 0.72 BELOW the 80 hand-crafted prompts (68.31)**, because of *"a long tail behaviour, where a small number of prompts have large scores, but most prompts are 'bad'"* `[measured]`.
- CLIP's own notebook: sequential forward selection terminated at **7 templates**, and *"This subset performs a bit better than the full 80 ensemble reported in the paper, especially for the smaller models"* `[quoted]`.
- The agrifood study: **domain-specific pools of 51-52 prompts consistently beat generic pools of 247-426** `[measured]`.

So: curate ~7-30 lab-specific templates ("a photo of a {} on a laboratory bench", "a close-up of a {} in a lab") and average the L2-normalised embeddings. Expect **+2 to +4 points** over a single template. ZPE's decomposition (ViT-B/16, ImageNet): bare class name → single template **+2.43**; single template → 80-ensemble **+1.94**; total **+4.37** `[measured]`.

**④ LLM descriptors: use full task-conditioned *sentences*, not attribute fragments — and only at the classification stage.** The fair-baseline comparison is damning for the attribute-fragment style. MPVR (ECCV 2024) is the only source benchmarking all of these under one protocol (ViT-B/32, 20 datasets):

| Method | 20-dataset avg | vs single template | **vs a fair dataset-specific ensemble** |
|---|---|---|---|
| CLIP `a photo of a <c>` | 57.83 | — | −1.91 |
| **CLIP dataset-specific ensemble** | **59.74** | +1.91 | — |
| **DCLIP** (LLM attribute fragments) | 59.23 | +1.40 | **−0.51** |
| **WaffleCLIP** (*random* descriptors) | 59.96 | +2.14 | **+0.22** |
| **MPVR** (LLM full sentences, GPT) | **62.85** | +5.03 | **+3.12** |

`[all measured-in-source]`. DCLIP's own LaTeX source admits the baseline problem verbatim: *"**We do not compare to the ensemble of 80 hand-tuned, class-agnostic prompts used in the original CLIP paper** as these require substantial manual labor."* Against that ensemble, CuPL Table 2 has DCLIP at **75.00 vs 75.54 — i.e. −0.54 on ImageNet.** ⚠ **Never compare DCLIP's, CuPL's and WaffleCLIP's published deltas to each other — their baselines are bare class name / 80-prompt ensemble / single template respectively, which inflates by 3-5x.**

WaffleCLIP's ablations are the actual disproof of the semantics story: using the **same random descriptor set for every class** still scores 58.16 vs DCLIP's 58.56, so class-*specific* semantics contributes only **0.40 points**; and switching mean→**max** pooling (pick the single best-matching descriptor) *drops* to 54.74, **below plain CLIP** — *"performance actually drops, showing that the VLM cannot leverage the additional semantics"* `[quoted]`. On the most fine-grained benchmark (Stanford Cars), **DCLIP is −1.46 below plain CLIP** `[measured]`. CuPL is honest about the mechanism too: it *"differs in its predictions from the standard method for 11.50% of predictions (CuPL correct for 4.48%, standard correct for 3.32%, neither correct for 3.70%)"* for a net **+1.15** — noise-shaping, not knowledge injection `[measured]`.

**The reconciliation:** short attribute fragments carry no usable semantics; full task-conditioned sentences do (+3.12). And MPVR's gain lands exactly where generic photo-templates are mismatched to the vocabulary — **EuroSAT +9.8, DTD +8.4, Flowers +7.2, but ImageNet only +1.7** `[measured]`. Lab equipment is the mismatched regime, so this is the one LLM-text mechanism worth building. Budget ~**+3 points**, not +10.

**⑤ ❌ Do NOT put LLM descriptors in the detection head.** This is the finding that overturns the obvious plan. DVDet's own ablation on OV-COCO:

| Hierarchical LLM descriptors | Learned region prompt | VLDet mAP_novel | RegionCLIP mAP_novel |
|---|---|---|---|
| — | — | 32.0 | 26.8 |
| **✓** | **—** | **29.7 (−2.3)** | **24.6 (−2.2)** |
| — | ✓ | 33.1 | 27.3 |
| ✓ | ✓ | 34.6 | 28.4 |

`[measured-in-source]`. The paper's own words: *"the contribution of fine-grained categorical descriptors is **compromised clearly at the absence of prompt learning**."* **Descriptors help only when bolted to trained region-prompt machinery — which your zero-retraining constraint forbids.** There is no established zero-shot, descriptor-only win in open-vocabulary *detection*. (Relatedly, DetPro shows CoOp-style prompt learning transfers poorly to detection: **15.3 vs 19.1 seg AP_r** — and much of DetPro's own gain is an ensembling effect, *"`Ensemble (0.5:1.0:0.1)` outperforms `IoU range = [0.5-1.0]` by +3.0 AP_r"*.)

### 3.3-bis The hard ceiling on the text channel — and why it pushes you to exemplars

Three independent measurements say the text channel will underperform badly on scientific-instrument names, and that **no amount of prompting fixes it**.

**(a) The closest published proxy to your problem is at-or-below chance.** The agrifood study's **Beans** task is 3-class plant pathology — specialist technical vocabulary, chance = 0.333 — scored with a uniform prompt ensemble:

| Model | Food-101 | Agri. crops | **Beans (chance 0.333)** |
|---|---|---|---|
| CLIP-B/32 | 0.830 | 0.616 | **0.287 ← below chance** |
| CLIP-B/16 | 0.881 | 0.709 | **0.307 ← below chance** |
| CLIP-L/14 | 0.932 | 0.780 | 0.398 |
| SigLIP-B | 0.912 | 0.869 | 0.404 |
| SigLIP-L | 0.943 | 0.910 | 0.635 |
| SigLIP-400M | 0.957 | 0.941 | 0.635 |

`[measured-in-source]`. ⚠ Preprint (arXiv:2608.18116), venue unconfirmed — medium confidence on provenance, numbers read from source. Corroborated in a second specialist domain: an IEEE ISBI 2025 paper on rare medical events reports *"we evaluated standard zero-shot methods like CLIP and CUPL, but **neither possessed knowledge of rare medical events**"*, with standard zero-shot CLIP tabulated as **"All ICs classified as Noise"** — total collapse `[quoted]`.

→ **Use SigLIP/SigLIP 2 rather than CLIP wherever you control the text encoder.** On specialist vocabulary the gap is 0.307 → 0.635, not a rounding error. (SAM 3's own processor wraps a `CLIPTokenizer` `[verified]`, so you do not control it there — another reason the exemplar path matters.)

**(b) Prompt engineering shifts the intercept, not the slope.** Udandarao et al. (NeurIPS 2024) establish a log-linear relation between pretraining concept frequency and zero-shot performance, then state verbatim: *"We observe that the **strong log-linear trend between concept frequency and zero-shot performance consistently holds across different prompting strategies**"* — including the 80-prompt ensemble `[quoted]`. And *"over **two-thirds of concepts** occur at almost negligible frequencies relative to the size of the datasets."* The long-tail penalty, measured **with** the ensemble:

| Pretraining | Model | ImageNet | Let-It-Wag | Drop |
|---|---|---|---|---|
| OpenAI-WIT | ViT-L/14 | 75.54 | 45.31 | **−30.23 (−40%)** |
| OpenAI-WIT | ViT-B/32 | 63.32 | 33.52 | −29.80 (−47%) |
| WebLI (SigLIP recipe) | SO400M | 83.44 | 67.32 | −16.12 (−19%) |
| DataComp-DFN | ViT-H/14 | 83.44 | 71.91 | −11.53 (−14%) |

`[measured; the "SigLIP recipe" label is an inference — the paper cites those rows to PaLI]`

**(c) Text-prompt confidence is not a usable score on unfamiliar names.** The agrifood study: *"On Beans, max-conf **collapses as a predictor of correctness (AUROC 0.27-0.72)**. The model's confidence no longer reflects visual evidence — **it reflects lexical biases in the embedding space**... since CLIP has no representation of plant diseases, no single prompt is reliably good. But the collective disagreement among all prompts faithfully reflects the model's errors."* Their prompt-disagreement signal reaches **AUROC 0.646 vs max-conf 0.275** on CLIP-B/16 — a **+0.37 gap** `[measured]`.

**Three concrete design consequences:**
1. **Gate on exemplar cosine similarity, not text logits**, for any class whose name is rare. The §3.5 prototype path sidesteps this failure mode entirely — which is a further argument for it.
2. **Use prompt-ensemble *variance* as a reject signal** (and as the active-learning acquisition function in §5.4). It is the better-calibrated quantity.
3. **Re-calibrate the per-class threshold `min_score` on the gold set for every new class.** A single global threshold will be badly wrong for rare names.

> **Honest gap, stated plainly: nobody has measured prompt ensembling, LLM descriptors, or exemplar prompting on scientific-instrument vocabulary.** Three independent full-text searches returned zero results. Everything in §3.3 and §3.3-bis is transfer from plant pathology, medical rare events, aerial imagery and fine-grained cars. Experiment 6 in §7D is how you find out.

There is also a **SAM 3 Agent** shipped in the repo (`sam3/agent/` with `agent_core.py`, `client_llm.py`, `client_sam3.py`, `system_prompts/` `[verified in repo tree]`) which wraps an LLM around SAM 3 for query decomposition and iterative refinement. This is the supported route for *compositional* prompts ("an analytical balance with the draft shield open", "the pipette being held") that a bare noun phrase cannot express — and it needs no retraining either. Cost: one LLM call per query. A 2026 robotics evaluation flags the relevant limitation — SAM 3 can "find shelves but not 'the top shelf'" (arXiv:2609.31760) `[measured-in-source]` — so expect to need the Agent, or hand-written spatial post-rules, for any relational class.

### 3.4 Hard negatives — do not skip this ✅

SAM 3's ablation: adding hard negatives moves image-level recognition **IL_MCC 0.44 → 0.68** `[measured-in-source]`. Concretely: when you prompt `"centrifuge"`, *also* prompt its confusable neighbours (`"microplate reader"`, `"autoclave"`) in the same batch and resolve by argmax over class scores per box, rather than thresholding each class independently. This is free and it is the main defence against the fine-grained confusions that will otherwise dominate your error budget.

### 3.5 Visual-exemplar channel (for the classes text cannot reach)

Two distinct cases, and they are **not** the same mechanism:

**(i) Same-image exemplar — natively supported, enormous gain ✅.** SAM 3's image API takes `input_boxes` + `input_boxes_labels` (1=positive, 0=negative), and the video predictor's `add_prompt(session_id, frame_idx, text=..., bounding_boxes=..., bounding_box_labels=...)` docstring is literally "Add text, box and/or point prompt on a specific video frame" ✅. Measured effect of **one** GT box drawn in the same image (SAM 3 Table 3, AP / AP+(T) / AP+(I) / AP+(T+I)):

| Benchmark | text-only AP | AP+ text | AP+ 1 image exemplar | AP+ text+exemplar | T-Rex2 AP+ image |
|---|---|---|---|---|---|
| COCO | 56.4 | 58.8 | **76.8** | **78.1** | 58.5 |
| LVIS | 52.4 | 54.7 | **76.0** | **78.4** | 65.8 |
| ODinW13 | 61.1 | 63.1 | **82.2** | 81.8 | 61.8 |

`[all measured-in-source]`. Read that again: **one box costs an annotator ~5 seconds and buys ~+18-22 AP.** This reframes your annotation budget — spend it on *one exemplar box per class per clip*, fed back as a prompt, not on exhaustive boxes.

**(ii) Cross-image / cross-clip exemplar — the real "add a class from 5 photos" ask. ❌ NOT supported by SAM 3's reference API.** I checked the source. `sam3/model/sam3_image_processor.py` exposes exactly `set_image`, `set_image_batch`, `set_text_prompt`, `add_geometric_prompt(box, label, state)`, `reset_all_prompts`, `set_confidence_threshold`. The `add_geometric_prompt` docstring reads *"Adds a box prompt and run the inference. **The image needs to be set**, but not necessarily the text prompt. The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range"* `[verified in source]`. There is **no** method to pass an exemplar bank, pre-computed exemplar embeddings, or a box referencing a different image. This matches the paper's architecture: each exemplar is "an embedding for the position, an embedding for the label, and **ROI-pooled visual features**" `[quoted]` — pooled from the current image's feature map.

*The hack that might work:* the exemplar encoder takes ROI-pooled features + position + label, and nothing in that formulation strictly requires the ROI to come from the same feature map. Pooling from a stored exemplar frame's features and injecting the token is plausibly ~50 lines. **Unvalidated — treat as a research spike, not a plan.** Design the fallback as the default:

1. **A DINOv3 prototype bank over SAM 3's class-agnostic proposals** (my recommended fallback). SAM 3 already gives you high-recall boxes; drop the text classification and re-classify: crop each box, embed with `facebook/dinov3-vits16-pretrain-lvd1689m` (384-d, patch 16, "high-quality dense features... without fine-tuning" ✅), and nearest-neighbour against a per-class bank of exemplar embeddings with a cosine threshold. Adding a class = appending k vectors to a `.npy`. Truly zero training, ~10 ms/crop, <1 GB VRAM for ViT-S `[my est]`. The DINOv3 Licence is Meta-specific — verify before commercial use.
2. **OWLv2 `image_guided_detection()`** — a *first-class, documented* cross-image one-shot API: the HF processor's `query_images` argument is documented as "Query images to use for image-guided object detection. When provided, these images serve as visual queries to find similar objects in the main `images`" ✅. Apache-2.0, open weights. This is the cleanest licensed path to "here are 5 photos of a spectrophotometer, go find it".
3. **YOLOE's SAVPE visual-prompt mode** — text / visual / prompt-free in one model, 102-306 FPS on a T4 `[measured]`. But **AGPL-3.0** ⚠.
4. T-Rex2's cross-image "generic visual prompt" is the conceptually ideal API but it is **API-only, no weights**, and SAM 3 beats it on every exemplar benchmark above. Rule it out.

### 3.6 Adding a class to the *tracker*

**You do not.** Decouple, which is what every good system here does:

- Track **class-agnostically** (SAM 3's tracker is a SAM 2-style mask tracker that knows nothing about the class; MASA is explicitly "class-agnostic"; DEVA is "class/task-agnostic temporal propagation"; ByteTrack/OC-SORT are pure geometry) `[all measured-in-source]`.
- Then **vote the class over the whole track**: for each track, aggregate per-frame class scores (mean of top-k frames, or score-weighted mean) and assign a single label. This both makes the vocabulary trivially extensible *and* fixes per-frame class flicker. Re-voting after adding a class requires replaying only the cached per-box embeddings, not the video.
- Keep the box→embedding cache keyed by `(clip, frame, box_hash)` so that adding class #21 costs you a matrix multiply over the cache, not a re-run of the pipeline. **This is the single most important engineering decision in the design.**

### 3.7 The mechanism ranking, with measured effect sizes

| Rank | Mechanism | Zero retraining? | Measured effect | Confidence |
|---|---|---|---|---|
| **1** | **Same-image exemplar box** (SAM 3 `input_boxes`) | ✅ Yes | **+18.0 AP on COCO (58.8→76.8), +21.3 on LVIS, +19.1 on ODinW13** from ONE box | **High** — SAM 3 Table 3 |
| **2** | **Hard negatives in the prompt batch** | ✅ Yes | **IL_MCC 0.44 → 0.68** | **High** — SAM 3 ablation |
| **3** | **Cross-image exemplar prototype bank** (DINOv3 over class-agnostic proposals, or OWLv2 `image_guided_detection`) | ✅ Yes | Not directly measured for this stack. Nearest anchors: OWLv2 OWL-ST lifts LVIS **rare** AP 31.2→44.6; SAM 3 10-shot RF100-VL 15.2→36.5 | **Medium** — the *design* is sound and the APIs are documented; the *number* is unmeasured |
| **4** | **Class-name disambiguation** (parentheticals, no underscores) | ✅ Yes | **+1.5% top-1** (CLIP notebook); **~+30 points** from fixing underscores on specialist names (agrifood) | **High** |
| **5** | **Curated small template ensemble** (7-30, lab-specific) | ✅ Yes | **+1.94 to +4.37** over a single template; but 247 uniform is **−0.72 vs 80 curated** | **High** |
| **6** | **Swap CLIP → SigLIP/SigLIP 2** as the text encoder where you control it | ✅ Yes | On specialist 3-class vocabulary: **0.307 → 0.635** | **Medium** (preprint) |
| **7** | **LLM full-sentence descriptors** (MPVR style), classification stage only | ✅ Yes | **+3.12** over a fair ensemble (20 datasets); concentrated in mismatched domains (EuroSAT +9.8) | **Medium-High** |
| **8** | **Per-track class voting** over a class-agnostic tracker | ✅ Yes | Not measured here; structurally required for extensibility (§3.6) | Medium |
| **9** | **LLM attribute-fragment descriptors** (DCLIP style) | ✅ Yes | **−0.51 to −0.54** vs a fair ensemble; **−1.46** on fine-grained Cars | **High — negative result** |
| **10** | **LLM descriptors in a detection head without trained prompts** | ✅ Yes | **−2.2 to −2.3 mAP_novel** (DVDet ablation) | **High — negative result. Do not do this.** |
| **11** | Few-shot fine-tuning of a separable adapter | ❌ No | SAM 3 RF100-VL 15.2→36.5, ODinW13 61.0→71.8 at 10-shot | High — but see §4 |

**The shape of the answer:** the exemplar/visual channel is worth an order of magnitude more than the text-engineering channel (+18 AP vs +3), and two of the text mechanisms are actively harmful. **Spend your effort on exemplars and hard negatives; spend an afternoon on class names and a curated template pool; skip LLM attribute descriptors entirely.**

### 3.8 Revised class registry schema

Reflecting §3.3's findings — note the explicit disambiguation, the absence of attribute-fragment descriptors, and the exemplar-first scoring:

```yaml
- id: spectrophotometer
  # ① disambiguated display name, spaces not underscores, parenthetical gloss
  canonical_name: "spectrophotometer (uv-vis absorbance reader)"
  # ③ curated, lab-specific templates — 7-30, NOT 200+
  templates:
    - "a photo of a {} on a laboratory bench"
    - "a close-up photo of a {} in a lab"
    - "a {} among other laboratory instruments"
  # ④ MPVR-style FULL SENTENCES, not attribute fragments. Generated once, offline.
  llm_sentences:
    - "A benchtop spectrophotometer sits on a lab bench, a grey rectangular instrument
       with a hinged sample compartment lid and a small digital display."
  # ② hard negatives — resolve by argmax across classes, not per-class thresholds
  hard_negatives: ["microplate reader", "pcr thermocycler", "desktop computer", "autoclave"]
  # ③ exemplar crops: the primary channel for this class
  exemplars:
    - {clip: 0041, frame: 300, box: [x1,y1,x2,y2]}
  # per-class threshold, recalibrated on the gold set whenever a class is added
  min_score: 0.45
  # which channel gates this class: "text" | "exemplar" | "both"
  gate: exemplar        # rare name -> do NOT trust text logits (§3.3-bis-c)
```

Cache, per class, one `text_embeds.npy` (mean of L2-normalised template + sentence embeddings) and one `prototypes.npy` (k DINOv3 exemplar vectors). **Adding a class writes two small files and touches no weights.**

---

## 4. The fine-tuning-destroys-open-vocab problem

**It is real, it is severe, and the magnitude is driven by how *narrow* your fine-tuning vocabulary is — not by how many parameters you touch.** With ~150 clips and ~10-20 classes you are deep in the danger zone.

### 4.1 The headline evidence

| Setting | What was tuned | Metric | Before | After | Δ | Source |
|---|---|---|---|---|---|---|
| **YOLO-World full FT on LLVIP-IR (1 class)** | all 76.81M | COCO AP50 | **51.90** | **0.10** | **−51.80 (−99.8%)** | ModPrompt, arXiv 2412.00622 Table 4 ✅ *independently verified* |
| **same, head-only FT** | 2.31M (head) | COCO AP50 | 51.90 | **0.66** | **−51.24** | same ✅ |
| same, ModPrompt (visual prompt, detector frozen) | 3.08M | COCO AP50 | 51.90 | **51.90 ± 0.00** | **0.00** | same ✅ |
| Grounding DINO + TFA closed-set FT, 1-shot ODinW-13 | detector | ZCOCO (zero-shot COCO mAP) | 47.37 | 18.84 | **−28.53** | ZiRa, arXiv 2403.01680 |
| same, full-shot | detector | ZCOCO | 47.37 | 30.97 | −16.40 | ZiRa |
| **YOLO-World fine-tuning the CLIP text encoder on O365 (365 classes)** | CLIP text enc. | LVIS zero-shot **APr** | **14.5** | **8.6** | **−5.9 (−41%)** | YOLO-World, arXiv 2401.17270 Table 5 |
| CLIP ViT-L/14@336 end-to-end FT on ImageNet | full image enc. | avg over 5 distribution shifts | 73.4 | 68.6 | −4.8 (ImageNet-A: 77.7→65.4, **−12.3**) | WiSE-FT, arXiv 2109.01903 |
| CLIP sequential FT over 8 tasks (MTIL) | full CLIP | Transfer (zero-shot on unseen tasks) | 69.4 | 44.6 | **−24.8** | ZSCL, arXiv 2303.06628 |
| **CoOp** 16-shot prompt tuning, 11 datasets (encoders frozen!) | 16 prompt vectors | New-class acc. | 74.22 | 63.22 | **−11.00** | CoCoOp 2203.05557 / PromptSRC 2307.06948 |

Note the last row: **even tuning 16 prompt tokens with both encoders frozen loses 11 points of unseen-class accuracy**, and CoOp's harmonic mean (71.66) ends up *below* plain CLIP's (71.70). Worst per-dataset CoOp cases: UCF101 −21.45, DTD −18.72, Flowers102 −18.13. There is a clean dose-response: the bigger the base-class gain, the bigger the new-class loss.

The mechanistic counter-example that proves what is being destroyed: in the *same* YOLO-World table, fine-tuning a **BERT** text encoder on O365 *improves* zero-shot LVIS (AP 14.6→18.3, APr 3.4→6.6). Only CLIP degrades. The paper's own words: *"Fine-tuning CLIP leads to a severe performance drop... fine-tuning on O365 may degrade the generalization ability of the pre-trained CLIP, which contains only 365 categories and lacks abundant textual information."* **You can only destroy pretrained open-vocabulary alignment if it was there to begin with.**

### 4.2 Scale of damage vs vocabulary breadth [my inference, from cross-paper comparison]

| FT vocabulary size | Observed damage to original vocabulary |
|---|---|
| 1 class | −51.8 AP50 → effectively total (ModPrompt/LLVIP) |
| ~13 small datasets (ODinW-13) | −16 to −28 ZCOCO (ZiRa) |
| 365 classes (O365) | −3.1 AP / −5.9 APr (YOLO-World) |
| 1000 classes (ImageNet) | −4.8 avg-shift (WiSE-FT) |

**Your situation (~10-20 classes, 150 clips) sits at the catastrophic end.** My estimate: full fine-tuning SAM 3 on your clips would take its ability to find a *newly named* 21st class from "works out of the box" to "near zero", while gaining maybe +15-20 AP on the 20 classes you trained on. That is precisely the trade you have been told not to make.

### 4.3 Mitigations, ranked by measured effect

| Rank | Mitigation | Measured effect | Evidence |
|---|---|---|---|
| **1** | **Don't fine-tune the detector at all.** Add classes via text + exemplars + a swappable prototype head (§3). | 0.00 loss by construction | — |
| **2** | **Frozen detector + external visual-prompt / side branch** (ModPrompt style; 3.08M trainable) | COCO AP50 **51.90 preserved exactly**; target-domain AP50 95.63 vs 97.43 for full FT → **−1.8 target for +51.8 retained** | arXiv 2412.00622 ✅ |
| **3** | **Reparameterizable zero-interference side branch (ZiRa)** | ZCOCO 46.06 (−1.31 only) *while* ODinW-13 Avg 59.73 — beats iDETR downstream (58.71) **and** retains 13.91 more zero-shot AP than CL-DETR | arXiv 2403.01680 |
| **4** | **Freeze the text encoder** (cheapest single switch) | +3.1 AP / **+5.9 APr** on zero-shot LVIS vs tuning it | arXiv 2401.17270 |
| **5** | **WiSE-FT weight interpolation, α=0.5**, if you must fully fine-tune | CLIP-L/14@336: avg-shift **76.9** vs 68.6 FT (**+8.3**) vs 73.4 zero-shot (**+3.5**) — *and* ImageNet 86.8 vs 86.2. **Strictly dominates both endpoints.** Free, no extra data. Limit: only 52.3 Transfer in the harder MTIL setting vs ZSCL's 68.1. | arXiv 2109.01903 |
| **6** | **Distil/replay toward the base model** (ZSCL / VR-LwF) | ZSCL recovers Transfer 44.6 → **68.1** (ceiling 69.4) = **95% recovery**; VR-LwF 44.17 → **62.03** (ceiling 62.62) = **96%** | 2303.06628, 2207.09248 |
| **7** | **LP-FT** (linear probe first, then FT) | **+10% OOD, +1% ID** over full FT across 10 shift datasets | arXiv 2202.10054 |
| **8** | **FLYP** (fine-tune with the contrastive pretraining loss) | +4.2% OOD over standard FT; >1% both ID and OOD over LP-FT | arXiv 2212.00638 |
| **9** | **Regularised prompt tuning** if you tune prompts at all: PromptSRC New 76.10, KgCoOp 73.6, CoCoOp 71.69 — vs CoOp's 63.22 | 2307.06948, 2303.13283, 2203.05557 |
| **10** | Adapter-only tuning | ZCOCO −5.07 (vs −16.40 full FT) but costs ~7.6 AP of downstream performance | 2403.01680 |

### 4.4 Two anti-patterns to avoid

- ⚠ **"Just fine-tune the head" does NOT protect you.** Head-only FT gave COCO AP50 **0.66** vs full FT's 0.10 — both are total wipeouts. On an open-vocab detector, the head *is* the region-text alignment; replacing it with a closed softmax is the destruction.
- ⚠ **"LoRA, so no forgetting" is wrong.** CLIP-LoRA is an *accuracy* result, not a retention result, and it tunes **both** encoders. In SAMCL's forgetting table LoRA ranks **worst** (FM 0.123) vs EWC 0.111, O-LoRA 0.091, ER 0.010. PEFT ≠ protection.

### 4.5 Honest gaps

- **No paper quantifies loss of generality after fine-tuning SAM / SAM 2 / SAM 3 on a narrow dataset.** Many papers *assert* the risk to motivate adapter designs (ProMISe, TopoLoRA-SAM, SAM2 adapters), but with no numbers. Treat "fine-tuning SAM 3 destroys its concept vocabulary" as **plausible-but-unquantified** — strongly suggested by the detector evidence, not directly measured. SAM 3's own README_TRAIN.md documents custom fine-tuning and contains **no warning** about this, and the paper's few-shot section ("We fine-tune SAM 3 without mask loss... we do not perform any prompt tuning") reports only downstream AP, never retention.
- SAM 3's **own** few-shot numbers show fine-tuning *is* worth a lot in-domain: ODinW13 61.0 → 71.8, RF100-VL 15.2 → 36.5 at 10-shot `[measured]`. So the right answer is not "never tune" but **"tune a separable, discardable adapter and keep the frozen zero-shot model alongside it, and measure retention on a held-out vocabulary."**
- The field systematically under-reports this: GLIP, Grounding DINO, MM-Grounding-DINO and YOLO-World's own fine-tuning tables all report downstream gains with **no** zero-shot retention metric. MM-Grounding-DINO reports Brain Tumor 0.4 → 47.5 AP after fine-tuning and never asks what happened to COCO. **Do not read their silence as safety.**

---

## 5. Evaluation protocol

### 5.1 The key structural insight: SAM 3's own metric decomposes exactly where your supervision stops

SAM 3 scores video/image PCS with **cgF1 = 100 × pmF1 × IL_MCC** `[measured-in-source, quoted]`, where:

- **IL_MCC** = *image-level* Matthews correlation coefficient — "binary prediction at the image level ('is the object present?') **without regard for mask quality**" `[quoted]`.
- **pmF1** = "positive micro F1", localization quality "on positive media-phrase pairs with at least one ground-truth mask" `[quoted]`.

**IL_MCC is computable, today, from your 150 clip-level multi-labels with zero new annotation.** A clip labelled `{microscope, pipette}` is a *positive* for those two classes and an *explicit hard negative* for every other class in your registry. MCC (unlike F1 or accuracy) needs both and is well behaved under the class imbalance you will have. And it is sensitive to exactly the failure you care about: SAM 3's own ablation moves IL_MCC **0.44 → 0.68** purely by adding hard negatives `[measured]`.

So: **split the evaluation along the same seam.**

| What you measure | Metric | Annotation needed | Cost |
|---|---|---|---|
| Does the system know *which* equipment is in the clip? | **IL_MCC** (+ clip-level multi-label mAP / per-class F1) | **You already have it** — 150 clip labels | **£0** |
| Does it put the box in the right place? | **mAP@[.5:.95]**, AP50, and `pmF1` | small exhaustively-boxed gold frame set | §5.2 |
| Does it keep identity over time? | **HOTA** (+ DetA/AssA split), **TETA** (LocA/AssocA/**ClsA**) | short ID-annotated segments | §5.2 |
| Nothing — pure sanity/regression signal | self-consistency proxies | **none** | §5.4 |

### 5.1-bis Which track metric, and why not MOTA

| Metric | GT required | Needs exhaustive GT? | Sparse/federated-safe |
|---|---|---|---|
| MOTA / CLEAR | boxes + IDs every frame | **Yes** (FP term) | **No** |
| IDF1 | full trajectories + IDs | **Yes** (IDFP term) | **No** |
| **HOTA** = √(DetA·AssA) | boxes + IDs on annotated frames | Yes for DetA, no for DetRe/AssA | Partly |
| **OWTA** = √(DetRe·AssA) | boxes + IDs, recall side only | **No** | **Yes** (designed for it) |
| **TETA** = (LocA+AssocA+ClsA)/3 | boxes + IDs; classes may be wrong/missing | **No** | **Yes** (designed for it) |
| TAO federated track mAP | tracks + per-video pos/neg/unknown class sets | per-class subset only | **Yes** |

**Do not use MOTA.** The HOTA paper measures it directly: *"Detection only score MODA explains 99.4% of MOTA variation,"* while *"IDSWs explain only 23.7%"*; the detection-to-association error ratio *"varies between 42.3 and 186.4"* across top trackers; IDSWs measure *"association errors only one time-step back"*; ID transfers get no penalty at all; and it is frame-rate dependent — *"Same tracker performance at 40fps yields 0.99 MOTA versus 0.90 at 4fps"* `[all measured-in-source]`. That last property alone disqualifies it: your annotation frame rate is a free parameter, and MOTA would move with it.

**Primary: TETA = (LocA + AssocA + ClsA)/3.** Built for exactly your two problems. (a) Fine-grained confusion — *"MOT implicitly assumes that the classification performance is near-perfect. However, this is far from the case in recent large-scale MOT datasets, which contain large numbers of classes with many rare or semantically similar categories."* Your classes *are* "grey box on a bench" — balance vs shaker vs spectrophotometer. (b) Incomplete annotation — TETA builds a **local cluster** around each GT box (predictions join if IoU > r), and *"predictions not assigned to any clusters during evaluation are ignored, avoiding false penalties for unannotated objects"* `[all quoted, arXiv 2207.12978]`. The arithmetic (not geometric) mean is deliberate: a classification failure cannot zero out the whole score. Code: https://github.com/SysCV/tet, Apache-2.0.

**Secondary: HOTA + its sub-metrics, and OWTA when GT is sparse.** HOTA = √(DetA·AssA), averaged over α ∈ {0.05,...,0.95}, decomposing into DetA/AssA/LocA/DetPr/DetRe/AssPr/AssRe. The user study found *"HOTA performs much better than both MOTA and IDF1"* at matching human judgement `[measured]`. For non-exhaustive GT use **OWTA = √(DetRe·AssA)** from *Opening up Open-World Tracking* (CVPR 2022), whose rationale is exactly yours: *"If we consider unlabeled regions as non-objects (FPs), we may be penalizing the tracking system for tracking regions that could still be considered to be valid objects"*, which works because *"the FPA term in AssA is not affected by FP tracks that are not matched to ground truth"* `[quoted]`. ⚠ OWTA alone is gameable by emitting many tracks — always pair it with the federated AP for precision control.

### 5.1-ter The practical shortcut: TrackEval already implements federated evaluation

This is the biggest time-saver in the whole protocol. **TrackEval** (https://github.com/JonathonLuiten/TrackEval, **MIT**) does not advertise federated support, but `trackeval/datasets/tao.py` implements it. Its docstring, verbatim:

> *"Unmatched tracker detections are removed if there is not ground truth data and the class does not belong to the categories marked as negative for this sequence. Additionally, unmatched tracker detections for classes which are marked as not exhaustively labeled are removed."*

```python
is_not_exhaustively_labeled = cls_id in raw_data['not_exhaustively_labeled_cls']
is_neg_category             = cls_id in raw_data['neg_cat_ids']
if gt_ids.shape[0] == 0 and not is_neg_category:
    to_remove_tracker = unmatched_indices
elif is_not_exhaustively_labeled:
    to_remove_tracker = unmatched_indices
else:
    to_remove_tracker = np.array([], dtype=np.int)
```

**So: emit your gold set as a TAO-format JSON with per-clip `neg_category_ids` and `not_exhaustive_category_ids`, and you get HOTA / DetA / AssA / LocA / IDF1 / TrackMAP under your own federated annotation, for free, with correct false-positive suppression.** Day-one task: write the exporter. BURST follows the same LVIS-style schema — *"we... provide two fields per video which convey (1) which classes are present but not exhaustively annotated, and (2) which classes are definitely not present"* `[quoted]`.

**And the SAM 3 repo separately ships** `sam3/eval/hota_eval_toolkit/`, `sam3/eval/teta_eval_toolkit/`, `cgf1_eval.py`, `coco_eval.py`, `ytvis_eval.py` `[verified in repo tree]`. Between TrackEval and SAM 3's eval dir you implement **zero** metrics. (SAM 3 also reports **pHOTA**, a phrase-level HOTA variant defined in its appendix F.5 — formula not retrieved, implementation present.)

### 5.2 How many frames to hand-annotate — the number and the justification

> **Answer: 600 frames = 4 frames from each of 150 clips, spread ≥2 s apart, never consecutive — plus 30 sub-clips annotated at 1 Hz with persistent IDs (~450 more frames). ~2,600 boxes, ≈5-6 hours of one annotator. If the budget is tight, 300 frames (2/clip) is the usable floor at ≈2.7 h.**

**Step 1 — the variance law.** A Monte-Carlo study over 101-point COCO-style AP (250-2,000 independent eval sets per cell) gives `hw·√n` near-constant, i.e. AP's sampling error behaves almost exactly like a binomial proportion **in the number of GT instances**:

> **95% half-width(AP) ≈ 2·√(AP·(1−AP)·D) / √(n_instances)**

| frames (1/clip) | ~instances | SD(AP) | 95% half-width |
|---|---|---|---|
| 10 | 30 | 0.075 | ±0.146 |
| 30 | 90 | 0.043 | ±0.084 |
| 50 | 150 | 0.032 | ±0.063 |
| 100 | 299 | 0.023 | ±0.045 |
| 200 | 602 | 0.017 | ±0.032 |
| 400 | 1202 | 0.012 | ±0.023 |

**Step 2 — and here is the finding that should actually change your plan. The design effect `D`:**

| Sampling scheme | measured `hw·√n` | implied **D** |
|---|---|---|
| i.i.d. frames | 0.89-1.05 | 1.0 |
| **1 frame per clip**, clip-level difficulty | 1.18-1.32 | **≈1.7** |
| **10 consecutive frames per clip** | 2.69-2.97 | **≈9** |

**Sampling 10 consecutive frames from one clip inflates your AP confidence interval ~2.8x (variance ~9x) versus the same instance count spread across clips.** 600 instances drawn from 20 clips is about as informative as **70** instances drawn from 70 clips. This is the single highest-leverage decision in the whole evaluation design, and it is free to get right: **spread the budget across all 150 clips; never annotate consecutive frames for the detection set.**

**Step 3 — the recommendation.** With AP≈0.5 (worst case) and D≈1.7:

| target per-class half-width | instances/class |
|---|---|
| ±0.20 | 41 |
| ±0.15 | 73 |
| **±0.10** | **163** |
| ±0.05 | 653 |

| Tier | Frames | Sampling | instances/class | Expected CI |
|---|---|---|---|---|
| T0 smoke test | 150 | 1/clip × 150 clips | 25-40 | per-class ±0.20-0.25, mAP ±0.09 |
| **T1 minimum usable** | **300** | 2/clip, far apart | **≥60** | per-class ±0.15-0.17, **mAP ±0.06** |
| **T2 recommended** | **600** | 4/clip, ≥2 s apart | **≥120** | per-class ±0.10-0.12, **mAP ±0.04** |
| T3 publication-grade | 1,500+ | 10/clip | ≥400 | per-class ±0.06, mAP ±0.025 |

**Step 4 — why not smaller.** Maier-Hein et al. (*Nature Communications* 9:5217, 2018) bootstrapped 4,000+ real biomedical-challenge rankings at a **median of 20 test cases** and found *"the first rank is stable (the winner stays the winner) for 21, 11, and 9% of the tasks"* depending on metric; *"a median of 15% and up to 100% of the other teams were ranked first in at least 1% of the bootstrap partitions"*; and *"leaving a single test case out led to 67% of the teams other than the winning team ranking first"* in one task `[measured-in-source]`. **At 20 items your leaderboard is noise.** That is the empirical reason a 20-frame eval set is not a shortcut.

**Step 5 — per-class floor.** Never report an AP for a class with **<30 instances**. LVIS's "rare" bin is 1-10 images and AP_r is pathologically unstable — Dave et al. show **AP_r moving 12.6 → 19.5 purely by raising the per-image detection cap from 300 to 5,000** `[measured-in-source]`. Below 30 instances, report **recall@IoU0.5 at a fixed score threshold with a Clopper-Pearson interval** instead; it is a clean binomial and far more honest. Also use **AP^fixed** (per-class 10,000-detection cap, no per-image cap) rather than stock COCO AP, which retains "98.5% of full AP" without the truncation artefact.

**Step 6 — track set.** Follow the TAO/BURST protocol and annotate **at 1 Hz** — TAO: *"label them with bounding boxes at 1 frame-per-second"*; BURST: train 6 fps, **val/test 1 fps**, because *"annotating at 6fps is thus a compromise... it reduces annotation cost while still ensuring smooth scene progression"* `[both measured-in-source]`. 30 clips × ~15 s at 1 Hz ≈ 450 frames with persistent IDs. Stratify to cover all classes and deliberately include the hard cases: an occlusion, a hand passing over, a camera pan, two identical instances side by side.

**Step 7 — federate, don't exhaust.** Adopt LVIS's construction verbatim: *"A federated dataset is... the union of smaller constituent datasets, each of which looks exactly like a traditional object detection dataset for a single category"*, with a positive set 𝒫_c (exhaustive for c) and negative set 𝒩_c (c definitely absent), and *"we do not count false positives for category c on images i that have e_i^c set to false. We do measure recall on these images"* `[measured-in-source]`. TAO applies the same rule to video: *"During evaluation of a particular category, we use only videos with exhaustive labels for computing precision and all videos for computing recall."* This is also what keeps your protocol valid when class #21 arrives — you annotate class 21 on the frames you need, not the whole gold set.

### 5.2-bis Measured annotation times (not estimates)

| Task | Measured time | Source |
|---|---|---|
| Conventional box draw + verify (ILSVRC/PASCAL) | **25.5 s draw + 9.0 s verify = 34.5 s/box**; +7.8 s/image "any others?" check | Papadopoulos et al. |
| **Extreme clicking (4 extreme points)** | **7 s/box**, **5x faster**; mean IoU **88%**, 98% of boxes IoU>0.5; detectors trained on them score "identical" mAP to GT boxes | Extreme Clicking, ICCV 2017 |
| Box annotation **inside video** (MOT) | **5.2 s/box** | PathTrack, ICCV 2017 |
| Trajectory/ID annotation by cursor-follow | **only ~30% slower than watching the video in real time**; ~2x faster than VATIC | PathTrack |
| Centre click | **1.87 s/click** | Papadopoulos et al., CVPR 2017 |
| Human verification (yes/no on a proposed box) | **1.6 s** | Papadopoulos et al., CVPR 2016 |
| SAM assisted-manual mask | **34 s → 14 s/mask** as the model improved; "14 seconds is **6.5x faster** than mask annotation for COCO" and "only **2x slower** than bounding-box labeling with extreme points" | SAM, 2023 |
| COCO instance mask | **79 s/instance** ("over 22 worker hours per 1,000 segmentations") | COCO |

**Budget, built from those numbers:**

| Line item | Calculation | Time |
|---|---|---|
| T2 boxes, extreme clicking | 1,500 × 7 s | **175 min** |
| Per-frame "any others?" coverage check | 600 × 7.8 s | **78 min** |
| Per-clip × per-class presence verification | 150 × ~10 × 1.6 s | **40 min** |
| Track set boxes, in-video rate | 1,125 × 5.2 s | **98 min** |
| Track ID linking (1.3x real time) | 30 × 15 s × 1.3 | **10 min** |
| | **Total** | **≈ 6.7 h** (T1 variant ≈ 4.3 h) |
| *Same with naive drag-and-draw @34.5 s/box* | | *≈ 18-20 h* |

**Two budget-critical corollaries.** (1) **The annotation interface matters more than the frame count** — extreme clicking vs drag-and-draw is a 3x swing on the same 600 frames. (2) **Do not annotate masks.** 79 s/instance vs 7 s/box is an 11x multiplier for information you do not need, since your required output is boxes + tracks. SAM-in-the-loop masks at 14 s/mask are still 2x slower than extreme-click boxes.

Finally, budget a second annotator on ~50 frames for agreement: SAM 3's own SA-Co/Gold uses **three** annotators with oracle best-of scoring specifically to absorb noun-phrase ambiguity `[measured]`, and "glassware" vs "beaker" vs "volumetric flask" will be genuinely contested in your data.

### 5.3 The clip-level proxy — use it, but know how it lies

Aggregate per-frame detections into a clip-level predicted class set (e.g. class c present if max-over-frames score > τ_c, or if ≥k frames exceed τ_c), then score against the clip multi-label with per-class F1 / mAP / MCC. This is a **multiple-instance-learning** consistency check and it is free.

**Its failure mode is severe and you must say so out loud:** a detector that puts a box on the *bench* every frame and labels it "centrifuge" will score a perfect clip-level F1 while having zero localization quality. The proxy constrains *recognition* only. Never report it alone; always pair it with the 300-frame localization read. Treat clip-level F1 as a **regression test** (did a prompt change break recognition?) and mAP/HOTA as the **quality metric**.

**Four more failure modes, ranked by how badly they will bite you:**

1. **Localization is entirely unconstrained on positives.** Max-pooling over space *and* time means any box anywhere in any frame with the right label satisfies the proxy. Not fixable by changing the pooling function. Always run the degenerate control: "emit one full-image box per class in the clip's label set" — it should score near-perfect clip mAP and ~0 AP. If your proxy can't tell that apart from a real detector, it is measuring nothing.
2. **Part/context shortcut** — the documented MIL failure (C-MIL, §5.5). Your classes are unusually exposed: "microscope" ← eyepiece, "fume hood" ← sash handle, "centrifuge" ← lid latch.
3. **Co-occurrence / scene gist.** Lab equipment co-occurs heavily (centrifuge + balance + glassware on one bench). With only 150 clips the co-occurrence structure is almost certainly memorisable, so a model can get the clip label set right from scene gist alone. Clip-level mAP will look great and transfer will not.
4. **Tiny-N instability.** 150 clips × ~10 classes, with rare classes in maybe 10-20 clips. A class present in 15 clips has a clip-level AP 95% CI of roughly **±0.25** `[my derivation from §5.2]` — and recall Maier-Hein et al.: at ~20 cases, removing a single case reflipped the winner in up to 67% of tasks `[measured]`.

**Where the clip labels genuinely earn their keep: the negative side.** Class *c* absent from a clip ⇒ every detection of *c* in that clip is a false positive, **localization-free**. Report this as a first-class metric:

```
FPR_abs(c) = (# detections of c above θ in clips where c is labelled absent) / (# frames in those clips)
```

Zero extra annotation, directly penalises hallucination, and it is exactly the `neg_category_ids` mechanism that TAO/BURST/LVIS use to make precision computable (§5.1-ter, §5.2 step 7). **This is the single highest-ROI thing in your existing labels.** Pair it with a **clip-level recall floor** (fraction of (clip, class) positives where the detector fires at least once) as an upper bound on achievable DetRe. Those two numbers plus the 600-frame federated AP are a defensible suite; clip-level mAP on its own is not.

⚠ And per Choe et al.: **do not tune θ on the clip-level proxy and then report localization.** That is precisely the leakage that produced a decade of illusory WSOL progress.

### 5.4 Label-free self-consistency proxies (zero annotation, run on the whole unlabelled pool)

These cost nothing and are how you tune thresholds on 1.35M frames you will never label. Rank them by how much I trust them:

1. **Forward/backward cycle consistency.** SAM 3's native video predictor exposes `propagate_in_video(propagation_direction="both")` `[verified, API]`. Propagate forward from a keyframe, then backward, and measure mask IoU between the two passes at each frame. Low IoU = the track is unreliable. This is the classic forward-backward error idea from TLD-era tracking, now essentially free. **Best single unsupervised signal — it correlates with real failure and needs nothing.**
2. **Detector-vs-tracker disagreement.** SAM 3 video already computes this internally (`assoc_iou_thresh=0.1`, `trk_assoc_iou_thresh=0.5`, `new_det_thresh=0.7`, `recondition_every_nth_frame=16` `[verified, config]`) and exposes both `obj_id_to_score` (detector) and `obj_id_to_tracker_score` per frame `[verified, API]`. Log the gap. A track whose detector score collapses while the tracker score stays high is drifting onto a distractor.
3. **Two-model agreement.** Run a second, architecturally unrelated open-vocab detector (MM-Grounding-DINO or OWLv2) on the keyframes and measure box agreement at IoU 0.5. Disagreement concentrates on exactly the hard/ambiguous instances, which is also your active-learning queue for the next annotation round.
4. **Temporal stability / jitter.** For a static object with a static camera the box should be nearly constant: report per-track std-dev of centre and of √area, normalised by box size, and a **flicker rate** (fraction of frames where a track's detection drops out and returns within k frames). Both have near-zero expected value in your static case, which makes them unusually diagnostic here — much more so than on pedestrian benchmarks.
5. **Track-length and count sanity.** Number of distinct tracks per clip should be small and stable; a sudden jump in track count across a parameter change means ID fragmentation.
6. **`removed_obj_ids` / `suppressed_obj_ids` volume.** SAM 3's hotstart heuristics (`hotstart_delay=15`, `hotstart_unmatch_thresh=8`, `hotstart_dup_thresh=8`) report what they pruned `[verified, API]`. A high prune rate is a direct signal that the prompt is generating junk.

**Published formulations you can lift directly:**

- **Forward-backward error** — Kalal, Mikolajczyk, Matas, *Forward-Backward Error: Automatic Detection of Tracking Failures*, ICPR 2010: *"the tracking is performed forward and backward in time and the discrepancies between these two trajectories are measured. We demonstrate that the proposed error enables reliable detection of tracking failures and selection of reliable trajectories"* `[quoted]`. The deployed TLD test is concrete and worth copying verbatim: *"A failure of the tracker is declared if median |d_i − d_m| > 10 pixels"* over a 10×10 grid of pyramidal Lucas-Kanade points `[quoted from the TLD TPAMI paper]`. **This is an actual published failure-detection method, not a repurposed training loss.**
- **Warp-based temporal consistency** — Liu et al., *Efficient Semantic Video Segmentation with Per-frame Inference*, ECCV 2020: `E_warp(Q_{t−1}, Q_t) = |Q_t ∩ Q̂_{t−1}| / |Q_t ∪ Q̂_{t−1}|`, where `Q̂_{t−1}` is the prediction from t−1 **flow-warped** into t. Requires no ground truth — it compares the model to its own warped prediction `[quoted]`. Box analogue: warp frame t−1's mask into t with RAFT, IoU against frame t's mask.
- **Flicker / flip probability** — the ImageNet-P temporal form from Hendrycks & Dietterich (ICLR 2019): `FP = 1/(m(n−1)) Σ_i Σ_j 1(f(x_j) ≠ f(x_{j−1}))`. **Needs no labels at all** — it only counts prediction changes between consecutive frames `[confirmed]`.
- **Stability Index** — Wang et al., *Towards Stable 3D Object Detection* (2024) and Zhang & Wang, *On The Stability of Video Detection and Tracking* (2016): per-track σ of centre/scale/aspect normalised by box size, plus detect↔miss flip rate. Both as published require GT associations; swap "GT track" for "predicted track" as the grouping key to get a label-free variant `[my adaptation]`.
- **Detic's pseudo-box diagnostics** — the best published precedent for validating pseudo-boxes *without* new labels. Detic (ECCV 2022) reports **cover rate** ("ratio of assigned proposal covering the ground truth box": **92.8% for max-size vs 69.0% for prediction-based** assignment) and **consistency** ("average IoU between assigned boxes across training iterations") `[measured-in-source]`. Consistency is fully label-free. Compute the same thing across your prompt variants.
- **Agreement-on-the-line** (NeurIPS 2022) is the theoretical licence for detector-ensemble agreement: *"the OOD agreement between the predictions of any two pairs of neural networks also observes a strong linear correlation with their ID agreement... without any labeled data, we can predict the OOD accuracy of classifiers"* `[quoted]`. Related: **ATC** (ICLR 2022) *"estimates target performance 2-4x more accurately than prior methods"* on classification.

**⚠ The honest limitation — state this to stakeholders.** Every proxy above measures **consistency**, which is necessary but *not sufficient*. A detector that confidently and stably puts a "centrifuge" box on the same wall clock in every frame scores **perfectly** on flicker, warp consistency, FB cycle error and SI, and has zero AP. Two measured results make this concrete: the Stability-Index paper finds *"no evident correlation between detection accuracy and model stability"*, and Zhang & Wang find accuracy *"has relatively low correlation"* with their three stability components `[both measured-in-source]`. There is also **no established method for label-free mAP estimation in object detection** — the label-free performance-prediction literature (ATC, AutoEval, agreement-on-the-line) is all classification-side `[confirmed by search]`.

**So use proxies for exactly three things:** (a) CI regression alarms across all 150 clips; (b) pre-ranking a large candidate pool down to ~5 configurations before spending labelled-set budget; (c) **choosing which frames to annotate** — high-disagreement frames are literally the active-testing acquisition function `q*(i) ∝ E[loss]` (Kossen et al., ICML 2021, which reports *"after acquiring labels for only 5 test points... the standard deviation of active testing is already as low as it is for i.i.d. acquisition at step 40"* on synthetic data, and 2-4x labelling efficiency on real classification) `[measured]`. **Never as the headline accuracy number.** And calibrate them once against the gold set so they become an estimator rather than a smell test.

### 5.5 What weakly supervised detection practice actually does

**Every paper in the WSOD line keeps a small fully-annotated split and reports real detection metrics on it.** That is the answer: there is no accepted way to validate pseudo-boxes without some boxes.

- **CorLoc**, the canonical localization metric, from WSDDN (CVPR 2016) verbatim: *"the percentage of images that contain at least one instance of the target object class for which the most confident detected bounding box overlaps by at least 50% with one of these instances"*, and *"Differently from AP, which is measured on the PASCAL test set, CorLoc is evaluated on the union of the training and validation subset"* `[quoted]`. The MIL assumption is stated equally plainly: *"an image is interpreted as a bag of regions. If the image is labeled as positive, then one of the regions is assumed to tightly contain the object of interest. If the image is labeled as negative, then no region contains the object."*
- ⚠ **But CorLoc is weak and you should not use it.** It considers only the single most-confident box per image at a single IoU of 0.5 — nearly free to satisfy. Use **MaxBoxAccV2** instead (averaged over δ ∈ {0.3, 0.5, 0.7}, matching **all** predicted boxes), plus **PxAP**, from Choe et al.
- Representative numbers: OICR (CVPR 2017) VOC2007 **41.2 mAP / 60.6 CorLoc**, rising to 47.0 / 64.3 with an ensemble `[measured]`. Click supervision: MIL baseline 43.4-44.5 CorLoc / 29.6 mAP → **1 click 73.3 / 45.9** → 2 clicks 78.5 / 49.1, against 55.5 mAP fully supervised `[measured]`. The part-localization failure mode is explicit in C-MIL: methods *"are prone to get stuck into local minima (falsely localize object parts) while missing full object extent during training"* `[quoted]` — directly relevant, since "microscope" is easily satisfied by an eyepiece and "fume hood" by a sash handle.

**The single most important paper here for you** is Choe et al., *Evaluating Weakly Supervised Object Localization Methods Right* (CVPR 2020). Its finding: years of apparent WSOL progress was **threshold-tuning leakage** — *"the mixed policies for selecting τ has contributed to the illusory improvement of WSOL performances"*, and *"recent WSOL methods have not led to major improvements compared to CAM, when validated in the same data splits and same evaluation metrics"* (best ImageNet gain +0.3 pp) `[quoted]`. Three consequences you must design around:

1. **Carve a third split.** Choe et al. use train-weaksup / **train-fullsup (for hyperparameters only)** / test, with a fully-annotated budget of **ImageNet 10, CUB ~5, OpenImages 25 samples per class** `[measured]`. For ~10 classes that is **50-250 frames** — cheap. Without it, every threshold you tune leaks.
2. **Freeze the test set before any tuning.** Non-negotiable.
3. ⚠ **Their most uncomfortable finding: *"Few-shot learning baselines... outperforms existing WSOL methods."*** Once you have 50-250 fully-annotated frames, *training* on them may beat anything you extract from 150 video-level labels. **Budget a few-shot fully-supervised control** — and note this is consistent with SAM 3's own RF100-VL 15.2 ZS → 36.5 @10-shot `[measured]`. The weakly-supervised route is not automatically the right one.

Spatio-temporal precedent for the asymmetric protocol: Mettes et al., *Spot On* (ECCV 2016) trains on **points** and tests on **boxes** — 16,411 points across 29,802 training frames vs 15,835 boxes across 31,295 test frames `[measured]`. **Cheap supervision for the pipeline, boxes only on the eval split.** That is exactly your situation.

> ⚠ Confidence note on §5.2: the variance law, the design-effect table and the tier recommendations come from a Monte-Carlo simulation, **not** from a published paper — there is no canonical paper on mAP-estimator variance vs eval-set size (an arXiv query for `"average precision" AND "confidence interval" AND "object detection"` returns **0 results**). The annotation times in §5.2-bis **are** measured and sourced. Validate the variance claim on your own data with a **clip-level** bootstrap (1,000 resamples; resample whole clips, never frames or detections — detections within a frame are not independent).

---

## 6. Costed pipeline

### 6.0 The cost derivation you need to understand first

SAM 3's vision backbone is a **Perception Encoder ViT-L-class model: hidden 1024, 32 layers, patch 14, image_size 1008, window_size 24** `[measured, HF config]` → **5,184 tokens per frame**. I estimate **~5 TFLOPs per frame** at 1008px `[my estimate, from the config]`.

**RTX 3060 12GB, verified specs:** GA106, 3,584 CUDA cores, 1,777 MHz boost, **FP32 12.74 TFLOPS boost (9.46 base)**, 12 GB GDDR6, **360 GB/s**.

> ⚠ **Plan with 25.5 TFLOPS for bf16/fp16, not 51.2 and certainly not 102.4.** This caught me out mid-research and it matters. Wikipedia's "tensor compute" cell reads 51.2 / [102.4], but on **GeForce** Ampere the tensor-core rate is **halved when accumulating in FP32** — which is exactly what `torch.autocast` does. The 25.5 figure was independently re-derived from NVIDIA's own stated formula and validated against three published SKUs (3080 → 119.07 vs published 119; 3090 → 142.33 vs 142; 3070 → 81.25 vs 81.3). **The marketing number is 4x optimistic.** (An earlier revision of this report used 51.2; that was wrong.)

Two independent routes to the latency:
- **Scaling the published number.** 30 ms/image on an H200 `[measured]`. H200 bf16-dense ≈ 989 TFLOPS vs 25.5 → **38.8x** compute gap; memory bandwidth 4.8 TB/s vs 360 GB/s → **13.3x**. Realistic batch-1 ViT penalty: **15-25x** → **450-750 ms/frame**.
- **FLOPs / utilisation.** The H200 figure implies ≈17% utilisation (5 TFLOPs / (989 TFLOPS × 0.030)). Applying the same 17% to 25.5 TFLOPS gives 4.3 TFLOPS effective → 5e12 / 4.3e12 ≈ **1.15 s/frame**.

**→ SAM 3 image PCS on a 3060: ~0.5-1.0 s/frame at 1008px, ~0.2-0.3 s/frame at 560px (tokens drop 5184→1600) `[my estimate]`.** Note this now *brackets* the community report of 1.1-1.4 s on a 4090 via ONNX-Runtime/TensorRT-EP at FP32 `[measured, issue #424]` — a 4090 should be ~3x faster than a 3060, so that number still looks like a badly-utilised export, but the gap is no longer absurd.

**Corroborating anchor for the SAM 3 *video* path:** SAM 3.1 video is reported at **32 FPS on an H100** `[measured-in-source]` → scaling by compute (12.2x) to bandwidth (5.7x) gives **0.8-3.4 FPS on a 3060**. That is consistent with SAM3-ASH's 2.5 FPS @1 class on an H100 and confirms end-to-end SAM 3 video is not viable at scale on this hardware.

**The crucial consequence:** SAM 3's *tracker shares that same backbone*, so a propagated frame costs almost as much as a detected frame. Unlike SAM 2 (Hiera-T, 38.9M), "detect sparsely + propagate with SAM 3" saves you very little. This is why end-to-end SAM 3 video PCS measures only **2.5 FPS @1 class / 0.7 FPS @40 classes on an H100 with <25 GB peak** `[measured, SAM3-ASH]` → my estimate **0.1-0.4 FPS on a 3060, and at real risk of OOM** (a user hit CUDA-OOM in `propagate_in_video` on a **24 GB** 4090 `[measured, issue #511]`).

### 6.1 Recommended two-track architecture

**Track A — "gold" pass (for the ~200-frame evaluation set and the 150 labelled clips at reduced frame rate).** Full `Sam3VideoModel` PCS, 560px, bf16, `inference_state_device="cpu"`, `video_storage_device="cpu"`, `max_vision_features_cache_size=1`, or the native repo's `offload_video_to_cpu=True, offload_state_to_cpu=True` `[measured, API]`. Use **pre-loaded (offline) mode, not streaming** — the docs warn streaming "disables hotstart heuristics that remove unmatched and duplicate objects... This may result in more false positive detections and duplicate object tracks" `[measured]`. Use `propagate_in_video(propagation_direction="both")` so you get forward *and* backward passes — free cycle-consistency signal (§5).

**Track B — "bulk" pass (the 150 clips at full rate, and the 10x unlabelled pool).** Decouple:

| Stage | Model | GPU | VRAM | Throughput on 3060 | Cost/frame |
|---|---|---|---|---|---|
| 0. Decode | NVDEC (1 engine/card, 5th gen; H.264/HEVC 8+10-bit, VP9-8, **AV1 8+10-bit**) `[measured, NVIDIA matrix]` | either | ~0.3 GB | **~700 fps 1080p H.264** (vendor claims 748 fps Ampere; 592-667 fps measured on a 3090) | **~1.4 ms** |
| 0b. Shot detection | PySceneDetect Adaptive @256px (BSD-3) | CPU | — | ~2,080 frames/s | ~0.5 ms |
| 1. Keyframe detect, stride 8 | SAM 3 image PCS @560px, bf16, cached `vision_embeds` swept over all N class `text_embeds` | **GPU 0** | ~3-4 GB `[my est]` | **~3-5 fps of keyframes** `[my est]` | 200-330 ms ÷ 8 = **25-41 ms** |
| 2. Mask propagation | **SAM 2.1-hiera-small or EfficientTAM-S @512px + CUDA graphs**, box-prompted from stage 1 | **GPU 1** | ~2-3 GB `[my est]` | **7-15 fps @1024px; ~25-60 fps @512px + CUDA graphs** `[my est, high uncertainty]` | **67-143 ms → ~17-40 ms** |
| 3. Association / ID stitching | **MASA-R50** (Apache-2.0, 4 conv + 1 FC head) + IoU/GMC gate (BoT-SORT-style) | GPU 1 | ~1-2 GB `[my est]` | **>50 fps** `[my est]` | <20 ms |
| 4. Per-track class voting | cached 256-d query embeddings and/or DINOv3 ViT-S/16 crop embeddings (384-d) | GPU 1 or CPU | <1 GB | thousands/s | <1 ms |
| 5. Cache write | boxes + COCO RLE | CPU | — | — | <0.1 ms |

**The propagation stage is the bottleneck, by one to two orders of magnitude.** Decode + shot detection + PCIe + caching together are under ~5 ms/frame. Do not optimise anything else first.

**The highest-leverage single optimisation: run SAM 2 at 512px with CUDA graphs rather than 1024px without.** The one published sm_86 measurement (RTX A6000, `sam2#768`) reports Hiera-S at **17.7 FPS eager @512px**, and CUDA graphs / `torch.compile(mode="reduce-overhead")` on padded static shapes gives **5.8x at 512px but only 1.4-1.9x at 1024px** `[measured-in-source, single user report — load-bearing, verify it]`. The implication is that SAM 2's A100 figure is heavily *launch-bound*, not compute-bound, so kernel-launch elimination beats raw FLOPs here. SAM 2's own Table 9(a) ablates 512²/768²/1024² as a documented speed-accuracy knob.

**Consider EfficientTAM instead of SAM 2.1-tiny** (Apache-2.0): EfficientTAM-Ti is **18M params @ 96.2 FPS on A100, SA-V test 70.7 J&F**; EfficientTAM-S is **34M @ 85.0 FPS, 74.5 J&F** `[measured]` — vs SAM 2.1-tiny's 38.9M @ 91.2 FPS / 76.5. For static lab equipment the J&F gap is likely irrelevant and the parameter saving is real.

**Multi-GPU: two separate processes, `CUDA_VISIBLE_DEVICES`-pinned, pinned-host handoff. Not one process with two devices.** The hard reason: **GPU-to-GPU P2P does not exist on GA106**, so cross-device CUDA IPC *cannot* work regardless of how you configure it, and `can_device_access_peer` is known to return misleading values. Ship **boxes and RLE** across the process boundary (~11 KB/frame, ≈0.0005 ms at 23 GB/s), **never embeddings or frames** (44 MB/frame). GPU 1 has its own NVDEC engine — **re-decode there rather than transferring pixels.** Before writing any code, run `nvbandwidth` and CUDA's `simpleP2P` to confirm your actual link: one documented case had PCIe bandwidth collapse to ⅓ with no root cause ever found.

### 6.2 Wall-clock

Assumption to make the arithmetic concrete: **30 s/clip at 30 fps = 900 frames/clip** `[my assumption — rescale linearly]`.

End-to-end configurations, ranked (all `[my estimate]` built on the measured anchors above):

| Configuration | Throughput | vs 30 fps real time |
|---|---|---|
| Detect-only, fast OV detector, TRT FP16, every frame, 1 GPU | **60-100 FPS** | 2-3x faster |
| Detect stride 8 + SAM 2.1-S @**1024px** bf16, 2 GPUs | **7-15 FPS** | **0.25-0.5x — tracker-bound** |
| Detect stride 8 + SAM 2.1-S/EfficientTAM-S @**512px + CUDA graphs**, 2 GPUs | **~25-60 FPS** `[high uncertainty — measure it]` | **~1-2x** |
| Detect stride 8 + SAM 2.1-**T** @512px + CUDA graphs | **~30-70 FPS** | cheapest viable real-time path |
| **SAM 3 video PCS end-to-end** | **0.8-3.4 FPS** | 0.03-0.1x |
| DEVA (Grounding DINO + XMem) | ~2-4 FPS | plus CC-BY-NC-SA **and** a Gurobi dependency |

Wall-clock, assuming **30 s/clip at 30 fps = 900 frames/clip** `[my assumption — rescale linearly]`:

| Workload | Frames | Track A (full SAM 3 video PCS, 0.8-3.4 FPS) | Track B (keyframe + propagate @512px + CUDA graphs, ~25-60 FPS) |
|---|---|---|---|
| 150 labelled clips | 135,000 (75 min) | 11-47 h on 1 GPU → **~6-24 h on 2 GPUs** | **~40-90 min** |
| 150 clips subsampled to 5 Hz | 22,500 | **~1-4 h on 2 GPUs** | ~7-15 min |
| ~10x unlabelled pool | 1,350,000 (12.5 h) | **not viable** (60-240 h) | **~7-15 h** (one overnight) |
| 600-frame gold eval set | 600 | **~3-12 min** — use Track A here, always | — |

⚠ If the @512px + CUDA-graphs figure does not materialise (it rests on a single unresolved user report), fall back to the **7-15 FPS** row: 150 clips becomes **~2.5-5 h** and the 10x pool **~25-54 h**. Both are still tractable; Track A is not.

**Ranked speed levers, biggest first:**
1. **512px + CUDA graphs on the propagator** — up to **5.8x** `[measured on sm_86, single report]`. Pad to static shapes; dynamic shapes defeat graph capture.
2. **Keyframe stride 8 instead of per-frame detection** — 8x on the detector stage.
3. **Drop SAM 3 to 560px** — ~3x `[my est]`, with an explicit accuracy warning from the docs.
4. **Cache `vision_embeds` and sweep all N classes against it** — makes cost independent of vocabulary size; ASH measures **6.5x** at 40 prompts `[measured]`.
5. **Propagate with SAM 2.1-tiny/EfficientTAM, not SAM 3** — ~10x on that stage `[my est]`.
6. **Pre-compute text embeddings.** Worth **3.7x measured for YOLO-World**; ≈0 for OWLv2 at small vocabularies; **architecturally impossible for Grounding DINO** `[measured]`.
7. `torch.compile` on the SAM 3 image encoder — maybe 1.2-1.4x on sm_86 `[my est]`.

> ⚠ **Do NOT TensorRT SAM 2's memory attention.** The only published sm_86 measurement is a **3x regression**. Compile/export the *image encoder* only; keep memory attention in PyTorch bf16.

### 6.3 Decode, sampling, and export

**Decode — use `torchcodec`.** PyTorch-native, **BSD-3-Clause**, actively maintained by Meta (v0.17, PyTorch 2.11+, Python 3.10-3.14), and *"On CUDA GPUs, TorchCodec supports decoding and encoding of videos and jpeg images"* — it returns PyTorch tensors directly, so there is no CPU round-trip `[verified]`. Requires an FFmpeg built with the NVIDIA codec support. The RTX 3060 has **1 NVDEC engine (5th gen)** per card covering H.264 8/10-bit, HEVC 8/10-bit, VP9 8-bit and **AV1 8/10-bit** `[measured, NVIDIA matrix]` — two cards give you two engines. **Decode will not be your bottleneck**: NVDEC does 1080p H.264 at hundreds of fps `[my est]` while your detector runs at ~5 fps, a ~100x margin. Do not spend engineering time here.

**Measured decode throughput:** NVIDIA claims **748 fps** for 1080p on Ampere NVDEC, and **592-667 fps was measured on an RTX 3090** `[measured-in-source]` → **~1.4 ms/frame, ~700 fps** on a 3060. CPU decode via torchcodec hit **1,589 fps on 24 threads**, scaling to ~320-450 fps on 8 threads `[measured / scaled]`. Either way, against a 7-60 fps pipeline, **decode has a 10-100x margin. Do not spend engineering time here.**

Alternatives, ranked: **PyNvVideoCodec** (MIT, lower-level NVDEC bindings, v2.2.x current); **NVIDIA DALI** (Apache-2.0, mature, heavier integration); **PyAV** (BSD-3, fine at these rates); **`decord`** — Apache-2.0 and it does support NVDEC via `-DUSE_CUDA=ON`, **but "only CPU versions are provided with PYPI now"** so GPU needs a source build, and I could not verify its recent commit activity `[maintenance status unverified — treat as a risk]`; **⚠ `NeLux`** posts the fastest published decode numbers but is **AGPL-3.0**. torchcodec supersedes decord for this use case. ⚠ I found **no measured NVDEC fps for a 3060 specifically**, and no `nvidia-smi` video-clock figure for it.

**Frame sampling.** What the implementations actually use, all verified:

| System | Detector / re-prompt interval |
|---|---|
| **DEVA** | **every 5 frames**, n=3 in-clip consensus; its own sweep says **5 is the knee and 7 falls off** |
| **SAM-Track** | `sam_gap = 10` (⚠ a secondary source claimed 100; `model_args.py` says **10**) |
| **SAM 3 video** | `recondition_every_nth_frame = 16` |
| **Grounded-SAM-2** | 20 |

**Both SAM 2 and SAM 3 prefer *error-triggered* over *periodic* re-prompting** `[measured-in-source]` — which is also the cheaper design. Recommended: **stride 8 as the ceiling, but trigger early on (a) a shot boundary, (b) mask-IoU between the propagated and the last detected mask dropping below ~0.6, (c) a tracker-score collapse.** Route on camera motion: **8 frames when moving, 16-32 when static** (mean optical-flow magnitude or the GMC homography translation norm over a 1 s window). Shot boundaries from **PySceneDetect** (BSD-3, Adaptive detector at 256px runs ~2,080 frames/s on CPU) or **TransNetV2** (MIT); avoid `video-keyframe-detector` (GPL-3.0).

⚠ **There is no published "detector stride k → J&F" curve for any VOS model** — the closest thing is DEVA's 3/5/7 merge-frequency row. So the stride is a hyperparameter you must sweep on your own gold set, not one you can look up. For the detection *eval* set the sampling rule is the opposite and non-negotiable: spread across clips, never consecutive (§5.2, design effect ≈9).

> ### ⚠⚠ The export trap that would silently kill your open vocabulary
>
> **Exporting YOLO-World or YOLOE to ONNX/TensorRT bakes the class vocabulary into the graph as constants — and YOLO-World does it *silently*** `[measured-in-source]`. You would ship an artefact that looks like an open-vocabulary detector, passes your tests on the current 10 classes, and then simply ignores class #11 with no error. This is the "prompt-then-detect / re-parameterization" feature working exactly as designed, turned against you.
>
> If you need a runtime-variable class list from an exported graph, your options are: **TAO Grounding DINO** (text token IDs are genuine graph inputs) or **`optimum-cli`-exported OWLv2**. Otherwise, keep the detector in PyTorch and accept the speed cost. Add a CI test that exercises a class absent at export time and asserts a non-empty result.

**Export — ⚠ do not plan on it for SAM 3.** The evidence is actively discouraging: a user measured **~1.1 s/image on an RTX 4090 via ONNX-Runtime's TensorRT EP at FP32**, versus 1.4 s on the CUDA EP — a mere 20% gain — and **no maintainer replied** `[measured, issue #424, opened 22 Jan 2026]`. There is no official TensorRT path, and the requester's explicit asks (native engine export, FP16/BF16/INT8 variants) are unanswered. Community ONNX ports exist (`wkentaro/sam3-onnx-models-v0.3.0`) but with **no model card** `[verified]`. By contrast, export is a solved problem for the *fallback* detectors: Ultralytics ships ONNX/TensorRT/OpenVINO for YOLO-World/YOLOE/YOLO26, and BoxMOT exports ONNX/OpenVINO/TFLite for its ReID models `[verified]`. **Decision: run SAM 3 in native PyTorch bf16 with `torch.compile`** (SAM 3.1's release notes cite "Enhanced torch.compile support" `[vendor-claimed]`); expect maybe **1.2-1.4x on sm_86** `[my est]`. If you need a 10x+ speedup, the answer is distillation to a YOLO student (§6.5), not export.

**Multi-GPU with no NVLink.** Run **two separate processes**, not one process with two devices: detector on GPU 0, propagation+association on GPU 1, communicating via a filesystem/queue of NPZ shards. Rationale: (a) PCIe round-trips for frames/embeddings would eat the benefit — a single 1008px SAM 3 vision embedding is **10.6 MB** (§6.6), so shipping it cross-device is ~1 ms of PCIe per frame plus synchronisation; (b) a process boundary isolates the OOM risk, which given issue #511 is a real concern; (c) it lets you restart one stage without re-running the other. Do **not** rely on CUDA IPC / shared CUDA tensors across devices without NVLink `[my recommendation, not sourced]`.

### 6.4 Practical gotchas (each one found in the repo's issue tracker)

| Gotcha | Evidence | Mitigation |
|---|---|---|
| **The video session defaults to `dtype=torch.float32`** — a 2x memory and ~2x speed waste on a card where you cannot afford either | `init_video_session(..., dtype: torch.dtype = torch.float32)` `[verified, HF API]` | Pass `dtype=torch.bfloat16` explicitly. Ampere sm_86 supports bf16 natively |
| **Use bf16, not fp16 — there is no speed reason to prefer fp16 on sm_86** | bf16 and fp16 run at **identical** tensor-core rates on sm_86; fp16 has a documented overflow failure class in **DETR-style** models (and SAM 3's detector is DETR-based); SAM 2/3's own code uses bf16 `[measured / verified]` | `torch.autocast(dtype=torch.bfloat16)` everywhere. Do not chase fp16 |
| **bf16 dtype-mismatch bugs are real and recent** | Issue #507 *"RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float during image inference"*; PR #570 *"bfloat16 dtype preservation"*; PR #516 *"CUDA autocast for SAM3 streaming nodes"* `[measured, issue tracker]` | Pin a commit **after** those PRs land; smoke-test bf16 end-to-end before building on it |
| **Weights are manually gated on HF** | `"gated": "manual"` `[verified via HF API]` | Request access *now* — it is a human-in-the-loop delay, and it will break unattended CI. Mirror the weights locally |
| **No small SAM 3 variant exists or is planned** | Issue #219 asks for Tiny/Small/distilled; **no maintainer response since Nov 2025** `[verified]` | Plan around 848M. Resolution reduction (§6.0) is your only official lever |
| **Default resolution is 1008 and lowering it is officially discouraged** | *"⚠ Custom resolutions may degrade accuracy. The model is meant to be used at 1008px resolution"* `[quoted, HF docs]`; issue #361 reduced it anyway to avoid OOM during fine-tuning | Measure the accuracy cost on your gold set at 1008 / 672 / 560 before committing (§7D exp. 1) |
| **Streaming mode is quietly worse** | *"Streaming inference disables hotstart heuristics that remove unmatched and duplicate objects... may result in more false positive detections and duplicate object tracks"* `[quoted]` | Use pre-loaded (offline) mode for all annotation work. Streaming only if you ever need live inference |
| **Community INT8/ONNX ports have no published accuracy numbers** | `ussoewwin/SAM3.1-ConvRot-INT8` claims "50% VRAM reduction" (1.63 GB → 0.84 GB) with **no accuracy or latency data** `[verified]` | Do not adopt without measuring on your gold set |

### 6.5 Optional Stage 5: distil to a cheap student for the *frozen* part of the vocabulary

There is now published precedent for exactly the cost fix you will be tempted by. **"SAM3-Assisted Training of Lightweight YOLO Models for Precision Pig Farming" (arXiv:2605.25860, May 2026)** uses SAM 3 as an "offline auto-annotator" generating **zero-shot pseudo-labels with no manual labelling**, trains a YOLOv8m on them, and reports **79.4% mAP** and AP50 >99% in low-occlusion scenes, at a **200x inference speedup relative to the SAM 3 teacher** `[measured-in-source; note: a narrow, near-single-class domain, so treat 79.4 as an upper bound for your multi-class case]`.

And this is **licence-clean**: I read the SAM License and it contains **no restriction on using model outputs to train other models**, no derivative-naming rule and no MAU cap `[verified]`.

**But it is precisely the move that destroys open-vocabulary ability** — a YOLO head is a closed softmax (§4). So the correct shape is a **two-model system, not a replacement**:

- **Student (fast path):** a YOLO-ish detector distilled from SAM 3 pseudo-labels on the *currently frozen* N classes. Runs the 10x unlabelled pool at hundreds of FPS. Re-trained from scratch whenever the vocabulary changes — which is cheap, because the labels are generated, not annotated.
- **Teacher (open path):** frozen SAM 3, never fine-tuned. The *only* thing that handles class N+1. Also the thing that generates the next student's labels.

This gives you the speed without the lock-in, and it means "add a class" = "add a string to the registry, run the teacher on a sample, retrain the student overnight" rather than "re-annotate". **Do this only after experiments 1/2/5 (§7D) have pinned down teacher quality** — distilling a 15 mAP teacher produces a 15 mAP student, and you will have baked in its errors.

### 6.6 Disk footprint (explicit math)

| Artefact | Per unit | 150 clips (135k frames) | 10x pool (1.35M frames) |
|---|---|---|---|
| Boxes, parquet (~20 B/box, 10 boxes/frame) | 200 B/frame | **27 MB** | 270 MB |
| Boxes, JSON (~100 B/box) | 1 KB/frame | 135 MB | 1.35 GB |
| Masks, COCO RLE (~500 B/mask `[my est]`) | 5 KB/frame | **675 MB** | 6.75 GB |
| Masks, 1-bit PNG cropped to box | ~1-3 KB/mask | ~1.4-4 GB | 14-40 GB |
| DINOv3 ViT-S/16 crop embeddings, fp16 (384 x 2 B) | 768 B/crop | **1.04 GB** | 10.4 GB |
| SAM 3 **vision** embeddings @1008px, fp16 (5184 x 1024 x 2 B) | **10.6 MB/frame** | **89 GB** (keyframes only) | **890 GB** ❌ |
| same @560px (1600 x 1024 x 2 B) | 3.3 MB/frame | 28 GB (keyframes only) | 280 GB ❌ |
| SAM 3 per-box decoder query embeddings, fp16 (256 x 2 B) | 512 B/box | **0.7 GB** | 7 GB |

**Rule: never persist vision embeddings.** The independently-derived figures are stark: caching **SAM 2 image embeddings costs ~906 GB per hour of 30 fps video**, and full ViT patch tokens ~**227 GB/hour** — whereas **boxes + RLE + a CLS vector per box is ~11 KB/frame ≈ 1.35 GB/hour** `[derived]`. **Re-running the encoder is cheaper than reading the cache.** Two further traps: never cache the pre-`conv_s0/s1` `backbone_fpn` tensors (**5.25x larger than necessary**), and never cache `vision_pos_enc` (it is free to recompute).

**So persist exactly this:** boxes (parquet) + COCO RLE masks + 256-d per-box query embeddings + 384-d DINOv3 crop embeddings ≈ **1.35 GB per hour of video**, i.e. ~1.7 GB for the 150 labelled clips and ~17 GB for the whole 10x pool. That is precisely the set you need in order to **re-vote the class of every box when class #21 arrives, without touching the video again** — which is the whole point of the §3.6 design.

⚠ Unverified in this area: no controlled JSON-vs-parquet benchmark on detection records, no measured 1920×1080 1-bit PNG mask size, and no published zstd/blosc ratio for fp16 embeddings specifically. The numbers above are byte-count arithmetic, not compression measurements.

---

## 7. Sources

### 7A. Confirmed by source (I fetched the page/table and read the number)

**SAM 3 / SAM 3.1**
- SAM 3: Segment Anything with Concepts — arXiv:2511.16719, 20 Nov 2025 — https://arxiv.org/abs/2511.16719 — abstract; 30 ms/image on H200 with 100+ objects; "near real-time for ~5 concurrent objects" in video; SA-Co/Gold 54.1 cgF1, box 55.7 cgF1; COCO 56.4 AP; LVIS zero-shot 48.8 mask AP; SA-V test 30.3 cgF1 / 58.0 pHOTA; BURST 57.4; MOSEv2 60.3; DAVIS 92.2; YT-VOS19 89.7; ODinW13 61.0 ZS / 71.8 10-shot; RF100-VL 15.2 ZS / 36.5 10-shot; gDino-T 49.7 ZS, gDino 1.5-Pro 58.7/67.9, Gemini 2.5-Pro 33.7 on ODinW13; presence head +1.5 cgF1 / +0.05 IL_MCC; hard negatives IL_MCC 0.44→0.68; Table 3 exemplar numbers (COCO 56.4 / 58.8 / 76.8 / 78.1; LVIS 52.4 / 54.7 / 76.0 / 78.4; ODinW13 61.1 / 63.1 / 82.2 / 81.8; T-Rex2 58.5 / 65.8 / 61.8); exemplar = "a single input box sampled at random from the ground truth"; "ROI-pooled visual features"; cgF1 = 100*pmF1*IL_MCC; aligned Perception Encoder backbone
- facebookresearch/sam3 — https://github.com/facebookresearch/sam3 — 848M params; detector + SAM-2-style tracker sharing a vision encoder; SAM License; Python 3.12+/PyTorch 2.7+/CUDA 12.6+; eval dir contains `hota_eval_toolkit/`, `teta_eval_toolkit/`, `cgf1_eval.py`, `coco_eval.py`, `ytvis_eval.py`; `README_TRAIN.md` documents custom fine-tuning (Roboflow, ODinW13) with no retention warning
- SAM License — https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE — "non-exclusive, worldwide, non-transferable and royalty-free limited license"; no reverse engineering; ITAR/Trade-Control and military/nuclear/weapons prohibitions; publication acknowledgement required; **no MAU cap, no derivative-naming rule, no restriction on using outputs to train other models**; you own your derivatives
- RELEASE_SAM3p1.md — https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md — Object Multiplex "~7x speedup at 128 objects on a single H100"; MOSEv2 60.3→62.3; YT-Temporal-1B 50.8→52.9 cgF1; SmartGlasses 63.6→64.4 pHOTA; improved VOS on 6/7 benchmarks; "Enhanced torch.compile support"; released 03/27/2026
- HF `facebook/sam3` — https://huggingface.co/api/models/facebook/sam3 — **manually gated**; licence "other"; 0.9B; last modified 2025-11-20; ~10.3 GB repo
- HF `facebook/sam3.1` — https://huggingface.co/facebook/sam3.1 — gated; Object Multiplex description
- HF docs SAM3 — https://huggingface.co/docs/transformers/en/model_doc/sam3 — `Sam3Model`/`Sam3Processor`; `get_vision_features()`, `get_text_features()`, `forward(vision_embeds=..., text_embeds=..., attention_mask=...)`; `input_boxes` + `input_boxes_labels` (1/0/−10); batched mixed text+box prompts; `config.image_size = 560` with "⚠ Custom resolutions may degrade accuracy. The model is meant to be used at 1008px resolution"; ViT config hidden 1024 / 32 layers / patch 14 / image 1008 / window 24 / pretrain 336
- HF docs SAM3 Video — https://huggingface.co/docs/transformers/en/model_doc/sam3_video — `add_text_prompt(session, ["person","bed","lamp"])`, "detects objects from ALL prompts in a single pass", "efficiently reuses vision features across all prompts", `prompt_to_obj_ids`; per-frame output `object_ids / scores / boxes (XYXY abs) / masks`; `init_video_session(inference_device, inference_state_device, processing_device, video_storage_device, max_vision_features_cache_size=1, dtype)`; streaming "disables hotstart heuristics... may result in more false positive detections and duplicate object tracks"; config defaults `score_threshold_detection=0.5, det_nms_thresh=0.1, assoc_iou_thresh=0.1, trk_assoc_iou_thresh=0.5, new_det_thresh=0.7, recondition_every_nth_frame=16, hotstart_delay=15, init_trk_keep_alive=30, max_num_objects=10000, low_res_mask_size=288, suppress_overlapping_based_on_recent_occlusion_threshold=0.7`
- HF docs SAM3 Tracker — https://huggingface.co/docs/transformers/en/model_doc/sam3_tracker — PVS (SAM1/2-style point/box/mask); `attention_similarity` / `target_embedding` PerSAM personalization hooks
- sam3 base predictor — https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/model/sam3_base_predictor.py — `add_prompt(session_id, frame_idx, text=, points=, point_labels=, bounding_boxes=, bounding_box_labels=, ...)` "Add text, box and/or point prompt on a specific video frame"; `start_session(..., offload_video_to_cpu=False, offload_state_to_cpu=False)`; `propagate_in_video(..., propagation_direction="both", ...)`
- SAM 3 issues — #511 CUDA OOM in `propagate_in_video` on a **24 GB RTX 4090**; #424 "~1.1s per image" on RTX 4090 via ONNX-Runtime TensorRT EP, FP32, 600x500, vs 1.4 s CUDA EP; #219 "~850M parameters", no maintainer response on smaller variants; #361 reduced hydra resolution from 1008 to avoid OOM
- HF `ussoewwin/SAM3.1-ConvRot-INT8` — SAM 3.1 fp16 baseline **1.63 GB**, INT8 ConvRot 0.84 GB, "50% VRAM reduction" claimed, **no accuracy numbers** → low confidence

**Trackers**
- ASH / SAM3-ASH — arXiv:2610.01022, 1 Oct 2026 — https://arxiv.org/abs/2610.01022 — single 40 GB H100 MIG, peak GPU memory **<25 GB**; **2.5 FPS @1 class, 0.7 FPS @40 categories, 0.017 FPS @1196 classes (LV-VIS)**; MOTS20 HOTA **68.1** zero-shot vs trained ReMOTSv2 65.4; YT-VIS AP 49.8/48.5/47.1; DanceTrack 66.6 HOTA, MOT17 45.2 HOTA; chunks Δ=50 overlap ω=10; prompt batch B=5, **6.5x speedup** from batch 1→40; "Generalized Presence Token" reduces image encoding from O(N) to O(1) in #prompts; **no code-release commitment**
- MASA — arXiv:2406.04221 (CVPR 2024 Highlight) + https://github.com/siyuanliii/masa — **Apache-2.0**; variants MASA-R50 / SAM-vitB / SAM-vitH / Detic / GroundingDINO; TAO open-vocab MOT Table 2: MASA-Detic **47.0 base / 40.8 novel TETA** vs OVTrack 36.3 / 32.0; README reports MASA-GroundingDINO 47.3 / 41.9; BDD100K IDF1 71.7 / AssocA 52.9 vs ByteTrack 70.4 / 51.5; trained on SA-1B-500K unlabelled static images, 12 epochs, batch 128; track head = "4 convolutional layers and 1 fully connected layer"; "~10x faster than SAM" for proposals; plug-and-play with GroundingDINO/Detic/YOLOX/CO-DETR; **no published FPS**
- ByteTrack — https://github.com/ifzhang/ByteTrack — **MIT**; MOT17 test 80.3 MOTA / 77.3 IDF1 / **63.1 HOTA** at **29.6 FPS on a single V100** (incl. YOLOX); no ReID
- BoT-SORT — arXiv:2206.14651 + https://github.com/NirAharon/BoT-SORT — **MIT**; MOT17 test 80.5 MOTA / 80.2 IDF1 / **65.0 HOTA**; MOT20 77.8/77.5/63.3; **CMC ablation on MOT17-val: ByteTrack 77.66/79.77/67.88 → +KF 77.67/79.89/68.12 → +CMC 78.31/81.51/69.06 → full 78.39/81.53/69.11**, i.e. CMC alone +0.64 MOTA / +1.62 IDF1 / +0.94 HOTA; **"ran on a desktop with ... NVIDIA GeForce RTX 3060 GPU"; 6.6 FPS (BoT-SORT) / 4.5 FPS (BoT-SORT-ReID) on MOT17 test**
- OC-SORT — https://github.com/noahcao/OC_SORT — **MIT**; "pure motion-model-based"; MOT17 63.2 HOTA / 78.0 MOTA; MOT20 62.4/75.9; DanceTrack 55.1/89.4; ~28 FPS on 2080Ti, **700 FPS association-only on i9 CPU**
- Deep OC-SORT — https://github.com/GerardMaggiolino/Deep-OC-SORT — **MIT**; MOT17 64.9/79.4/80.6, MOT20 63.9/75.6/79.2, DanceTrack 61.3/92.3/61.5; adds adaptive appearance + CMC + dynamic alpha; "+~6 HOTA over OC-SORT on DanceTrack"
- StrongSORT/++ — https://github.com/dyhBUPT/StrongSORT — **GPL-3.0** ⚠; AFLink 591.9 Hz, GSI 140.9 Hz; BoT ReID
- Hybrid-SORT — https://github.com/ymzis69/HybridSORT — **MIT**; AAAI 2024; MOT17 63.6/79.3/78.4, MOT20 62.5/76.4/76.2, DanceTrack 62.2/91.6/63.0 at **27.8 FPS**; weak cues = velocity direction, confidence state, height state; "training-free, plug-and-play"
- BoxMOT — https://github.com/mikel-brostrom/boxmot — **AGPL-3.0** ⚠; MOT17-ablation HOTA/MOTA/IDF1: occluboost 71.10/78.50/85.28, botsort 69.68/78.23/82.33, boosttrack 69.25/75.91/83.20, strongsort 68.05/76.19/80.76, deepocsort 67.95/75.83/80.54, bytetrack 67.68/78.04/79.16, hybridsort 67.31/74.09/78.87, ocsort 66.44/74.55/77.90, sfsort 62.65/76.87/69.18; exports ONNX/OpenVINO/TFLite
- BoostTrack++ — arXiv:2408.13003 — SOTA HOTA/IDF1 on MOT20
- **OccluBoost — no arXiv paper exists** (`all:"OccluBoost"` returns 0 results). Unpublished; BoxMOT-internal. **Low confidence.**
- SAMURAI — https://github.com/yangchris11/samurai — **Apache-2.0**; SOTA on LaSOT/GOT-10k/TrackingNet/OTB100/NFS/LaSOT-ext; "It is a zero-shot method, we directly use the weights from SAM 2.1"; Kalman motion-aware memory; **single-object VOT**; no FPS/VRAM published
- DAM4SAM / **SAM2.1++** — arXiv:2411.17576 + https://github.com/jovanavidenovic/DAM4SAM — **Apache-2.0**; distractor-aware memory + "introspection-based update strategy"; *"modern trackers still struggle in the presence of distractors"*; introduces the **distractor-distilled DiDi dataset** "to study the distractor problem better"; DiDi quality 0.694 / robustness 0.944; "outperforms SAM2.1 and related SAM memory extensions on seven benchmarks and sets a solid new state-of-the-art on six"; **single-object**; no FPS published. A "DAM4SAM3" is *referenced* by HyperDAM (arXiv:2609.34396) but I could not locate its origin → **low confidence**. **Relevance: the distractor problem *is* your "two identical centrifuges side by side" problem — this is the literature to watch.**
- SAM 2 — https://github.com/facebookresearch/sam2 — **Apache-2.0**; SAM2.1 tiny 38.9M @ **91.2 FPS**, small 46M @ 84.8, base+ 80.8M @ 64.1, large 224.4M @ **39.5 FPS**, "Speed measured on an A100 with torch 2.5.1, cuda 12.4"; SA-V test J&F 76.5-79.5; MOSE val 71.8-74.6
- DEVA — arXiv:2309.03903 (ICCV 2023) + https://github.com/hkchengrex/Tracking-Anything-with-DEVA — "task-specific image-level segmentation and class/task-agnostic bi-directional temporal propagation"; in-clip consensus n=3 (semi-online) / n=1 (online), image model applied **every 5 frames**, IoU>0.5, α=0.5, deletion after L=5 frames; **7.8 FPS at defaults (M2F-R50)**, 25.8 FPS propagation-only on DAVIS-2017, V100; BURST test **OWTA 69.8**, uncommon classes 53.3; VIPSeg VPQ 56.0; open-vocab via Grounded-SAM. **LICENCE: https://raw.githubusercontent.com/hkchengrex/Tracking-Anything-with-DEVA/main/LICENSE.md = CC-BY-NC-SA 4.0 — non-commercial, ShareAlike. Verified. Do not ship this commercially.** (Contrast Cutie and XMem by the same author, which are MIT.)
- Cutie — arXiv:2310.12982 (CVPR 2024 Highlight) + https://github.com/hkchengrex/Cutie — **MIT**; MOSE 62.2 (small) / 64.0 (base) / 68.3 (base + MOSE train); DAVIS-2017 87.2 / 88.8; YTVOS19 86.2 / 86.1; **~45.5 FPS (small) vs XMem 56.6 FPS, recorded on YouTubeVOS with a V100**; "+8.7 J&F over XMem with a similar running time"
- XMem — https://github.com/hkchengrex/XMem — **MIT**; "Expect ~20 FPS even with long videos"; "Handle very long videos with limited GPU memory usage"; "videos with more than 10,000 frames with ease"
- OVTrack — https://github.com/SysCV/ovtrack (CVPR 2023) — **Apache-2.0**; weights on Google Drive; TETA 37.1 base / 28.8 novel; TAO 21.2 Track AP50; "trained solely on static images"
- TRACT — arXiv:2503.08145 — OV-TAO; "Code will be released" → **not available**
- CoTracker / CoTracker3 — https://github.com/facebookresearch/co-tracker — **CC-BY-NC (non-commercial)** ⚠; CoTracker3 released 15 Oct 2024, latest; TAP-Vid Kinetics 67.8 / DAVIS 76.9 (offline); **points, not boxes**
- HOTA — arXiv:2009.07736 (IJCV 2021) — "Previous metrics overemphasize the importance of either detection or association"; "decomposes into a family of sub-metrics... evaluate each of five basic error types separately"
- TETA / TETer — arXiv:2207.12978 — "Track Every Thing Accuracy (TETA), breaking tracking measurement into three sub-factors: localization, association, and classification"; "MOT implicitly assumes that the classification performance is near-perfect. However, this is far from the case in recent large-scale MOT datasets, which contain large numbers of classes with many rare or semantically similar categories"
- DanceTrack — arXiv:2111.14690 — "methods for multi-object tracking should also work when object appearance is not sufficiently discriminative"; "a significant performance drop on DanceTrack when compared against existing benchmarks"
- MOSEv2 — arXiv:2508.05630 — **SAM2 drops 76.4 (MOSEv1) → 50.9 (MOSEv2)**; 5,024 videos / 701,976 masks / 10,074 objects / 200 categories; adds "frequent object disappearance/reappearance, severe occlusions and crowding, smaller objects, adverse weather, low-light, multi-shot, camouflaged objects"; 20 VOS + 9 VOT methods show "consistent performance drops"
- BURST — arXiv:2209.12118 — unified VOS/MOTS benchmark, six tasks, consistent metrics

**Detectors**
- RF100-VL — arXiv:2505.20612 + https://github.com/roboflow/rf100-vl — 100 datasets, 7 domains **including "laboratory imaging"**; Table 2: GroundingDINO **15.7 ZS → 33.3 @10-shot**; Qwen2.5-VL-72B 5.4 → 7.5; Gemini 2.5 Pro 13.3 → 9.2; YOLOv8m 56.4 / YOLOv11m 56.5 fully supervised; "VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets"
- OWLv2 — arXiv:2306.09683 + https://huggingface.co/docs/transformers/en/model_doc/owlv2 — Apache-2.0 HF checkpoints (`google/owlv2-base-patch16-ensemble`, L/14); OWL-ST raises LVIS **rare** AP from 31.2 → **44.6** (+43% rel.) with no human boxes for those classes; **`Owlv2ForObjectDetection.image_guided_detection()` + `Owlv2Processor(query_images=...)`: "Query images to use for image-guided object detection. When provided, these images serve as visual queries to find similar objects in the main `images`. The query images override any text prompts"**; objectness head for query-agnostic ranking
- MM-Grounding-DINO — arXiv:2401.02361 + https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino — Apache-2.0; T/B/L; COCO ZS 50.4-52.5 mAP; LVIS-minival ZS 40.5-41.4 AP; ODinW13 0.533 / ODinW35 0.284; "We release all our models"
- Grounding DINO — https://github.com/IDEA-Research/GroundingDINO — **Apache-2.0**; Swin-T COCO ZS 48.4 AP, FT 57.2
- YOLO-World — arXiv:2401.17270 + https://github.com/AILab-CVC/YOLO-World — **GPL-v3** ⚠ ("supported for commercial usage"); LVIS AP S 18.5 / M 24.1 / L 26.8 / X 28.6 @640; "prompt-then-detect" offline vocabulary re-parameterization; **Table 5: frozen CLIP text enc. 22.4 AP / 14.5 APr vs fine-tuned CLIP 19.3 / 8.6; BERT fine-tuned 18.3 / 6.6 vs frozen 14.6 / 3.4**
- YOLOE — https://github.com/THU-MIG/yoloe — **AGPL-3.0** ⚠; text (RepRTA) / visual (SAVPE) / prompt-free (LRPC); LVIS-minival text-prompt AP 27.9 (v8-S, 305.8 FPS T4) / 32.6 (M, 156.7) / 35.9 (L, 102.5); trained 12-23.5 h on 8x RTX 4090
- Florence-2-large — https://huggingface.co/microsoft/Florence-2-large — **MIT**; 0.77B; `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>`, `<REGION_PROPOSAL>`; COCO det ZS **37.5 mAP**, FT 43.4
- T-Rex2 — https://github.com/IDEA-Research/T-Rex (ECCV 2024) — **API-only, weights not released**; IDEA License 1.0 for code; interactive / generic cross-image / customized-embedding visual prompts
- Rex-Omni — https://github.com/IDEA-Research/Rex-Omni — 3B on Qwen2.5-VL-3B-Instruct; **IDEA License 1.0 + Qwen Research License** ⚠; `IDEA-Research/Rex-Omni` + `-AWQ`; text/box/point/visual-reference prompting
- DINO-X — https://github.com/IDEA-Research/DINO-X-API — **API-only**; COCO ZS 56.0 AP, LVIS-minival 59.7 / rare 63.3, LVIS-val 52.4 / rare 56.5, SGinW 51.7 mask AP `[vendor-claimed]`
- Grounded-SAM-2 — https://github.com/IDEA-Research/Grounded-SAM-2 — Apache-2.0 + BSD-3; GroundingDINO/Florence-2/DINO-X + SAM 2; continuous-ID detect-then-propagate is **"still under develop"** and **"not that stable now"**
- YOLO26 / YOLOE-26 — arXiv:2606.03748 (Jun 2026) + https://docs.ultralytics.com/models/yolo26/ — YOLO26 COCO **40.9 (n) → 57.5 (x) mAP at 1.7-11.8 ms on T4 TensorRT**; NMS-free dual head, DFL removed, MuSGD optimizer; **YOLOE-26x LVIS-minival 40.6 AP under text prompting**, supports "text-, visual-, and prompt-free inference"; **Ultralytics licence is AGPL-3.0 + Enterprise** ⚠ (the paper itself is CC-BY-4.0); docs do **not** confirm public YOLOE-26 open-vocab checkpoints
- OV-DEIM — arXiv:2603.07022 (Mar 2026) — "end-to-end DETR-style open-vocabulary detector built upon the recent DEIMv2 framework" + GridSynthetic augmentation; "Code and pretrained models are available" at github.com/wleilei/OV-DEIM; **no LVIS/COCO numbers, no FPS, no licence stated in the abstract** → low confidence
- DINOv3 — arXiv:2508.10104 + https://huggingface.co/docs/transformers/en/model_doc/dinov3 + https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m — `facebook/dinov3-*` collection; **ViT-S/16 = 21.6M params, 384-d embedding, patch 16, 4 register tokens**, up to ViT-7B/16 (6.7B, 4096-d), plus ConvNeXt-Tiny (29M); "produces high-quality dense features... **without fine-tuning**"; **licence = "DINOv3 License", model is gated** ⚠
- OWLv2 base — https://huggingface.co/api/models/google/owlv2-base-patch16-ensemble — **Apache-2.0, NOT gated, ~155M params**
- SigLIP 2 — arXiv:2502.14786 + https://huggingface.co/docs/transformers/en/model_doc/siglip2 — `google/siglip2-*`; NaFlex (native aspect ratio) + FixRes; adds decoder pretraining, self-distillation and masked prediction "to improve dense prediction tasks". **`google/siglip2-so400m-patch14-384`: Apache-2.0, NOT gated, ~1.14B params** (≈2.3 GB at bf16) — this is the variant that scored 0.635 on the specialist Beans task vs CLIP-B/16's 0.307
- SAM 3 Agent — `sam3/agent/` (`agent_core.py`, `client_llm.py`, `client_sam3.py`, `system_prompts/`, `examples/sam3_agent.ipynb`) `[verified in repo tree]` — LLM wrapper for query decomposition / iterative refinement
- "What Stops Recursive Self-Improvement in Robotics?" — arXiv:2609.31760 (Sep 2026) — evaluates SAM 3 limitations, reports it can "find shelves but not 'the top shelf'" (relational/spatial language failure)
- EOVSAM — arXiv:2608.02284 + https://github.com/hustvl/EOVSAM — "accelerates inference by up to 338x" over vanilla SAM 3 and "consistently improves segmentation accuracy"; ADE20K-847 16.6 / ADE-150 39.0 / PC-459 20.4 / PC-59 60.9 / PAS-20 97.0 / ADE20K PQ 30.9 vs SAM 3's 14.0 / 37.8 / 18.3 / 58.3 / 96.3. **BUT: semantic/panoptic segmentation only — no instances, boxes or tracks — and the licence is contaminated by NVIDIA RADIO's "non-commercial research or evaluation" clause.** The 338x is vs exhaustive per-class vocabulary traversal (847 classes), so it does not transfer to a 10-20-class list.
- ActiveSAM — arXiv:2606.16996 + https://github.com/VILA-Lab/ActiveSAM — CC BY 4.0; "training-free inference framework that turns SAM 3 into an active-vocabulary segmenter"; 7.3-12.2x over SegEarth-OV3 on large-vocabulary datasets; +2.1 mIoU avg over 8 benchmarks. Semantic segmentation. (Useful idea: use the presence head to prune absent classes *before* running the full decoder.)
- SAM3-Assisted Training of Lightweight YOLO Models for Precision Pig Farming — arXiv:2605.25860, May 2026 — SAM 3 as "offline auto-annotator" producing zero-shot pseudo-labels; YOLOv8m student reaches **79.4% mAP with no human annotation**, AP50 >99% in low-occlusion scenes, **200x faster than the SAM 3 teacher**; PigLife dataset; "foundation models can function as effective, zero-annotation-cost supervisors". Caveat: narrow near-single-class domain
- DAM4SAM3 — referenced as a basis by HyperDAM, arXiv:2609.34396 (Sep 2026). Origin/repo **not located** → low confidence
- SAM3Dual (3rd place, MOSEv2 track) — arXiv:2608.22193 — training-free temporal memory extension, **64.37 J&F** on MOSEv2; Competitive Memory Readout (2nd place) — arXiv:2608.22064. Both are evidence that SAM 3's memory can be improved training-free in cluttered/occluded video

**Fine-tuning / open-vocab retention** (see §4 table for the numbers)
- ModPrompt — arXiv:2412.00622 — Table 4 ✅ *verified twice*: YOLO-World on LLVIP-IR, COCO AP50 FT **0.10** (76.81M) / HFT **0.66** (2.31M) / ModPrompt **51.90±0.00** (3.08M); LLVIP AP50 97.43 / 93.57 / 95.63. Caption: "we... show the catastrophic forgetting in HFT and FT baselines"
- ZiRa — arXiv:2403.01680 — Grounding DINO Swin-T; ZCOCO 47.37 → TFA 1-shot 18.84, full-shot 30.97; OW-DETR 31.22, CL-DETR 32.15, iDETR 37.32, adapter-tuning 42.30, **ZiRa 46.06** with ODinW-13 Avg 59.73
- WiSE-FT — arXiv:2109.01903 — CLIP ViT-L/14@336: zero-shot avg-shift 73.4, FT 68.6, **α=0.5 interp. 76.9**; ImageNet 76.6 / 86.2 / 86.8; IN-A 77.7 / 65.4 / 79.9
- ZSCL / MTIL — arXiv:2303.06628 — Transfer 69.4 (ZS) / 44.6 (naive FT) / **68.1 (ZSCL)**; WiSE-FT only 52.3 in this setting
- VR-LwF — arXiv:2207.09248 — ZS-Acc 62.62 → 44.17 (naive FT) → **62.03 (VR-LwF)**
- CoOp / CoCoOp — arXiv:2203.05557 — 11-dataset avg Base/New/HM: CLIP 69.34/74.22/71.70, CoOp 82.69/**63.22**/71.66, CoCoOp 80.47/71.69/75.83
- KgCoOp — arXiv:2303.13283 — 80.73/73.6/77.0 (reports CoOp New as 67.99 — reproduction disagreement, so quote CoOp's penalty as a **−6 to −11 range**)
- PromptSRC — arXiv:2307.06948 — 84.26/**76.10**/79.97; MaPLe 82.28/75.14/78.55; ProDA 81.56/72.30/76.65
- LP-FT — arXiv:2202.10054 — full FT ~7-10% worse OOD than linear probing; LP-FT +10% OOD / +1% ID over FT
- FLYP — arXiv:2212.00638 — +4.2% OOD over standard FT
- F-VLM — arXiv:2209.15639 — frozen backbone LVIS mask APr 18.6 vs 18.1 unfrozen (overall AP 24.2 vs 26.2): "finetuning... slightly compromises the novel category"
- ViLD — arXiv:2104.13921 — base-only supervised detector APr 0.0; ViLD-text 10.1, ViLD 16.1, ViLD-ensemble 16.6, Supervised-RFS 12.3
- CLIP-LoRA — arXiv:2405.18541 — accuracy result, not a retention result; tunes both encoders; CLIP ZS 11-dataset avg 65.1
- SAMCL — arXiv:2412.05012 — forgetting-measure table: LoRA **worst** (FM 0.123), EWC 0.111, O-LoRA 0.091, MoDA 0.020, ER 0.010, SAMCL 0.0019. **No plain full-FT baseline row.**

**Prompt engineering, LLM descriptors and the text-channel ceiling**
- **MPVR** — *Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs*, ECCV 2024 — arXiv:2403.11755 — the only one-protocol head-to-head (ViT-B/32, 20 datasets): CLIP single 57.83 / **CLIP dataset-specific ensemble 59.74** / DCLIP 59.23 / WaffleCLIP 59.96 / Waffle+Concepts+GPT 60.39 / MPVR-Mixtral 62.41 / **MPVR-GPT 62.85**; per-dataset gains EuroSAT +9.8, DTD +8.4, Flowers +7.2, ImageNet only +1.7; *"the CLIP text encoder responds favorably to semantically rich text descriptions (prompts), instead of randomly generated descriptors as in Waffle"*
- **CuPL** — Pratt, Covert, Liu, Farhadi, ICCV 2023 — Table 2 has **DCLIP at 75.00 vs the 80-prompt ensemble's 75.54 (−0.54)** on ImageNet; CuPL(full) **+1.37** / CuPL(base) **+0.72** mean over 15 datasets, with base **−1.04 on Stanford Cars and −2.14 on RESISC45**; *"CuPL outperforms the baseline even with just one hand-written sentence"*; *"begins to outperform the baseline at just 25 image-prompts"*; non-LLM sources are worse than nothing — **WordNet definitions 73.44, Wikipedia first sentence 68.20** vs 75.54; churn: *"differs... for 11.50% of predictions (CuPL correct for 4.48%, standard correct for 3.32%, neither correct for 3.70%)"*
- **DCLIP** — Menon & Vondrick, *Visual Classification via Description from Large Language Models*, ICLR 2023 (notable top-5%) — +3.42 on ImageNet **against a bare-class-name baseline**; its source admits verbatim: *"We do not compare to the ensemble of 80 hand-tuned, class-agnostic prompts used in the original CLIP paper as these require substantial manual labor for each dataset"*
- **WaffleCLIP** — ICCV 2023 — random descriptors; **same random set for every class scores 58.16 vs DCLIP's 58.56** → class-specific semantics worth only **0.40**; **mean→max pooling drops to 54.74, below plain CLIP** — *"performance actually drops, showing that the VLM cannot leverage the additional semantics"*; Stanford Cars: CLIP 58.54 → **DCLIP 57.08 (−1.46)** → Waffle 58.91 → Waffle+Concepts 59.70; LLM descriptors *"have a structurally different impact... which we find to be complementary to randomization"* (58.57 → 60.21 combined)
- **ZPE** — *Zero-shot Prompt Ensembling / zero-label prompt weighting*, ICML 2023 — arXiv:2302.06235 — ViT-B/16 ImageNet: **247-prompt uniform average 67.59 vs 80 hand-crafted 68.31 (−0.72)**; *"we observe a long tail behaviour, where a small number of prompts have large scores, but most prompts are 'bad'"*; ZPE weighting 68.60; decomposition class-name→single-template **+2.43**, single→80-ensemble **+1.94**. ⚠ Intro and Table 1 disagree (64.18/66.92/68.57 vs 63.94/66.37/68.31) — cite the table
- **CLIP prompt-engineering notebook** (OpenAI) — forward selection terminated at **7 templates**, and *"This subset performs a bit better than the full 80 ensemble reported in the paper, especially for the smaller models"*; **hand-editing class names worth +1.5% top-1 on ViT-B/32** (`"nail"`→`"metal nail"`, `"kite"`→`"kite (bird of prey)"`), author estimates *"another 0.5% to 1%"* available. ⚠ Its 55.93/83.36 figures are **ImageNet-V2**, not ImageNet
- **DVDet** — ICLR 2024 poster — OV-COCO Table 4: LLM hierarchical descriptors **alone** give VLDet mAP_novel **32.0 → 29.7 (−2.3)** and RegionCLIP **26.8 → 24.6 (−2.2)**; with learned region prompt 34.6 / 28.4; *"the contribution of fine-grained categorical descriptors is compromised clearly at the absence of prompt learning"*
- **DetPro** — CVPR 2022 — CoOp-equivalent prompt training **15.3 vs DetPro's 19.1 seg AP_r**; *"`Ensemble (0.5:1.0:0.1)` outperforms `IoU range = [0.5-1.0]` by +3.0 AP_r"* (much of the gain is ensembling)
- **Udandarao et al., concept frequency → zero-shot performance**, NeurIPS 2024 — Let-It-Wag drops **with the 80-prompt ensemble**: OpenAI ViT-L/14 75.54→45.31 (−30.23); ViT-B/32 63.32→33.52; WebLI SO400M 83.44→67.32 (−16.12); DataComp-DFN ViT-H/14 83.44→71.91 (−11.53). Decisive quote: *"the strong log-linear trend between concept frequency and zero-shot performance consistently holds across different prompting strategies"*; *"over two-thirds of concepts occur at almost negligible frequencies"*
- **Agrifood prompt-quality study** — arXiv:2608.18116 (Jul 2026) ⚠ **preprint, venue unconfirmed, medium confidence** — uniform-ensemble accuracy on **Beans** (3-class plant pathology, chance 0.333): CLIP-B/32 **0.287**, CLIP-B/16 **0.307** (both below chance), CLIP-L/14 0.398, SigLIP-B 0.404, SigLIP-L 0.635, SigLIP-400M 0.635; *"critical for beans: naive underscore names degrade accuracy by ~30 percentage points due to tokenization mismatch in the text encoder"*; *"On Beans, max-conf collapses as a predictor of correctness (AUROC 0.27-0.72)... it reflects lexical biases in the embedding space"*; prompt-disagreement AUROC **0.646 vs max-conf 0.275** on CLIP-B/16; prompt-quality transfers across domains ρ=0.765
- Rare-event medical zero-shot — arXiv:2501.16481, IEEE ISBI 2025 — *"we evaluated standard zero-shot methods like CLIP and CUPL, but neither possessed knowledge of rare medical events"*; zero-shot CLIP tabulated as **"All ICs classified as Noise"**
- CoOp — IJCV 2022; CoCoOp — CVPR 2022 — *"the learned context is not generalizable to wider unseen classes... CoOp overfits base classes"*
- ⚠ **Not relevant, listed to prevent misuse:** *"Does CLIP Know My Face?"* (arXiv:2209.07341, JAIR 80, 2024) is a **privacy / membership-inference** paper about identifying *people*; it says nothing about rare technical nouns. BiomedCoOp (arXiv:2411.15232) and PETR (arXiv:2609.23600) — found but **unverified, no numbers extracted**; both require training

**Evaluation metrics, protocols and annotation cost**
- HOTA — Luiten, Ošep, Dendorfer, Torr, Geiger, Leal-Taixé, Leibe, IJCV 129 (2021) — arXiv:2009.07736 — HOTA = √(DetA·AssA) over α ∈ {0.05..0.95}; sub-metrics DetA/AssA/LocA/DetPr/DetRe/AssPr/AssRe; **MOTA pathologies: "Detection only score MODA explains 99.4% of MOTA variation", "IDSWs explain only 23.7%", det:assoc error ratio "varies between 42.3 and 186.4", "association errors only one time-step back", "Same tracker performance at 40fps yields 0.99 MOTA versus 0.90 at 4fps"**; IDF1 shows "unintuitive and non-monotonic behavior regarding detection" and "unmatched trajectories' association accuracy is completely ignored"; user study: "HOTA performs much better than both MOTA and IDF1"
- IDF1 — Ristani et al., ECCV 2016 workshop — arXiv:1609.01775 — IDF1 = 2·IDTP/(2·IDTP+IDFP+IDFN); single min-cost bipartite matching of **whole trajectories**
- TETA / TETer — Li, Danelljan, Ding, Huang, Yu, ECCV 2022 — arXiv:2207.12978 + https://github.com/SysCV/tet (Apache-2.0) — TETA = (LocA+AssocA+ClsA)/3, **arithmetic** mean; local clusters: "predictions not assigned to any clusters during evaluation are ignored, avoiding false penalties for unannotated objects"; "deals with the challenging incomplete annotation problem"
- OWTA — Liu, Zulfikar, Luiten, Dave, Ramanan, Leibe, Ošep, Leal-Taixé, *Opening up Open-World Tracking*, CVPR 2022 oral — arXiv:2104.11221 — OWTA = √(DetRe·AssA); "If we consider unlabeled regions as non-objects (FPs), we may be penalizing the tracking system for tracking regions that could still be considered to be valid objects"; "the FPA term in AssA is not affected by FP tracks that are not matched to ground truth"
- TAO — Dave, Khurana, Tokmakov, Schmid, Ramanan, ECCV 2020 — arXiv:2005.10356 — 2,907 videos / 833 categories; **"label them with bounding boxes at 1 frame-per-second"**; "we ask the annotators to label tracks for up to 10 objects in every video... then separately collect exhaustive labels for every category for a subset of videos"; **"During evaluation of a particular category, we use only videos with exhaustive labels for computing precision and all videos for computing recall"**; LVIS federated mAP with 3D spatio-temporal IoU at 0.5
- BURST — Athar et al., WACV 2023 — arXiv:2209.12118 + https://github.com/Ali2500/BURST-benchmark — **train 6 fps, val/test 1 fps**; "Annotating at 6fps is thus a compromise... it reduces annotation cost while still ensuring smooth scene progression"; "The annotations in this dataset are not exhaustive... we do provide two fields per video which convey (1) which classes are present but not exhaustively annotated, and (2) which classes are definitely not present"; HOTA across six tasks, DetA→DetRe for open-world; 2,914 videos / 16,089 tracks / ~600k masks / 482 classes
- **TrackEval** — https://github.com/JonathonLuiten/TrackEval (**MIT**) — HOTA family, CLEARMOT, Identity, VACE, TrackMAP, J&F; official code for MOTChallenge/KITTI/RobMOTS/Open-World-Tracking. **`trackeval/datasets/tao.py` implements federated FP suppression** — docstring: "Unmatched tracker detections are removed if there is not ground truth data and the class does not belong to the categories marked as negative for this sequence. Additionally, unmatched tracker detections for classes which are marked as not exhaustively labeled are removed." (code verified at https://raw.githubusercontent.com/JonathonLuiten/TrackEval/master/trackeval/datasets/tao.py)
- LVIS — Gupta, Dollár, Girshick, CVPR 2019 — arXiv:1908.03195 — "A federated dataset is a dataset that is formed by the union of smaller constituent datasets, each of which looks exactly like a traditional object detection dataset for a single category"; 𝒫_c exhaustive for c, 𝒩_c c-absent; "During evaluation, we do not count false positives for category c on images i that have e_i^c set to false. We do measure recall on these images"; negative-set target 1% of images (~820 at 82k); **rare = 1-10 images, common = 11-100, frequent = >100**; 11.2 instances and 3.4 categories per image
- Dave, Dollár, Ramanan, Kirillov, Girshick, *Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details* — arXiv:2102.01066 — per-image 300-detection cap truncates rare-class recall: **AP_r 12.6 → 19.5** at a 5,000 cap; **AP^fixed** (per-class 10,000 cap, no per-image cap) retains "98.5% of full AP"; also AP^pool
- Maier-Hein et al., *Why rankings of biomedical image analysis competitions should be interpreted with care*, Nature Communications 9:5217 (2018) — https://doi.org/10.1038/s41467-018-07619-7 — median **20** test cases per task; **"the first rank is stable (the winner stays the winner) for 21, 11, and 9% of the tasks"**; "a median of 15% and up to 100% of the other teams were ranked first in at least 1% of the bootstrap partitions"; "leaving a single test case out led to 67% of the teams other than the winning team ranking first"
- Metrics Reloaded — Maier-Hein, Reinke et al., Nature Methods 2024 — arXiv:2206.01653
- Active Testing — Kossen, Farquhar, Gal, Rainforth, ICML 2021 — arXiv:2103.05331 — LURE estimator with `q*(i) ∝ E[loss]`; "after acquiring labels for only 5 test points... the standard deviation of active testing is already as low as it is for i.i.d. acquisition at step 40"; 2-4x labelling efficiency on real classification. ⚠ Built for **means**, not for mAP — apply to decomposable components only
- Prediction-Powered Inference — Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, *Science* 382 (2023) — arXiv:2301.09633 + https://github.com/aangelopoulos/ppi_py. Active Statistical Inference — Zrnic & Candès — arXiv:2403.03208. Same caveat (means, not mAP)
- LRP / oLRP — Oksuz, Cam, Akbas, Kalkan, ECCV 2018 — arXiv:1807.01696 — AP's "(i) inability to distinguish very different RP curves, and (ii) lack of directly measuring bounding box localization accuracy"
- Beyond mAP — Jena et al., CVPR 2023 — arXiv:2207.01614 — AP "does not penalize duplicate predictions in the high-recall range"
- TIDE — Bolya, Foley, Hays, Hoffman, ECCV 2020 — arXiv:2008.08115 — dAP decomposition into classification/localization/both/duplicate/background/missed. **Requires full GT** — run it on the 600-frame set
- TAP-Vid — Doersch et al., NeurIPS 2022 D&B — arXiv:2211.03726 — deliberately sparse GT ("given a finite annotation budget, we prioritize diversity"); AJ / <δ_avg over {1,2,4,8,16} px / OA; **"roughly 3.3 annotator hours are required to track 30 points through every frame on a 10-second video"**
- Extreme clicking — Papadopoulos, Uijlings, Keller, Ferrari, ICCV 2017 — arXiv:1708.02750 — **7 s/box, 5x faster than the 34.5 s/box (25.5 draw + 9.0 verify) ILSVRC reference**; mean IoU 88%, 92% >0.7, 98% >0.5; Fast-RCNN mAP "identical" to GT-box training
- Click supervision — Papadopoulos et al., CVPR 2017 — arXiv:1704.06189 — centre click **1.87 s**; VOC07 1 click/class/image in 3.8 h; MIL 43.4-44.5 CorLoc / 29.6 mAP → 1-click **73.3 / 45.9** → 2-click 78.5 / 49.1 (full sup. 55.5 mAP)
- Human verification — Papadopoulos et al., CVPR 2016 — arXiv:1602.08405 — **1.6 s** per yes/no; 6-9x cheaper overall
- PathTrack — Manen, Gygli, Dai, Van Gool, ICCV 2017 — arXiv:1703.02437 — **"box annotations take 5.2 seconds on average"**; path annotation **"only 30% slower than watching the video in real time"**; 720 seqs / 16,287 trajectories
- COCO — Lin et al. — arXiv:1405.0312 — **"over 22 worker hours per 1,000 segmentations" = 79 s/instance**; ">70,000 worker hours" total
- SAM — Kirillov et al. — arXiv:2304.02643 — assisted-manual mask **34 s → 14 s**; "14 seconds is 6.5x faster than mask annotation for COCO" and "only 2x slower than bounding-box labeling with extreme points"
- Points instead of masks — Cheng, Parkhi, Kirillov, CVPR 2022 — arXiv:2104.06404 — 10 points "approximately 5 times faster than annotating full object masks"; Mask R-CNN at 94-98% of full supervision
- Forward-Backward Error — Kalal, Mikolajczyk, Matas, ICPR 2010, DOI 10.1109/ICPR.2010.675 — "the tracking is performed forward and backward in time and the discrepancies between these two trajectories are measured... enables reliable detection of tracking failures and selection of reliable trajectories"; TLD (TPAMI) failure test: **"A failure of the tracker is declared if median |d_i − d_m| > 10 pixels"**
- Cycle-consistency of time — Wang, Jabri, Efros, CVPR 2019 oral — arXiv:1903.07593. ⚠ Used as a **training** signal; the paper does **not** use cycle error as a test-time quality measure
- E_warp — Liu, Shen, Yu, Wang, ECCV 2020 — arXiv:2002.11433 — `E_warp = |Q_t ∩ Q̂_{t−1}| / |Q_t ∪ Q̂_{t−1}|`, flow-warped, **no GT required**
- ImageNet-P flip probability — Hendrycks & Dietterich, ICLR 2019 — arXiv:1903.12261 — temporal FP needs **no labels**
- Stability — Zhang & Wang, arXiv:1611.06467 (E_F / E_C / E_R; "accuracy metric has relatively low correlation with other three stability metrics"); Wang et al., *Towards Stable 3D Object Detection*, arXiv:2407.04305 (SI; **"There is no evident correlation between detection accuracy and model stability"**)
- ATC — Garg, Balakrishnan, Lipton, Neyshabur, Sedghi, ICLR 2022 — arXiv:2201.04234 — "estimates target performance 2-4x more accurately than prior methods". AutoEval — Deng & Zheng, CVPR 2021 — arXiv:2007.02915. Agreement-on-the-line — Baek, Jiang, Raghunathan, Kolter, NeurIPS 2022 — arXiv:2206.13089 — "without any labeled data, we can predict the OOD accuracy of classifiers". ⚠ All classification-side
- WSDDN — Bilen & Vedaldi, CVPR 2016 — arXiv:1511.02853 — **CorLoc** = "the percentage of images that contain at least one instance of the target object class for which the most confident detected bounding box overlaps by at least 50% with one of these instances"; "Differently from AP, which is measured on the PASCAL test set, CorLoc is evaluated on the union of the training and validation subset"; MIL bag assumption
- OICR — arXiv:1704.00138 — VOC2007 **41.2 mAP / 60.6 CorLoc**; +FRCNN ensemble 47.0 / 64.3. PCL — arXiv:1807.03342. C-MIL — arXiv:1904.05647 — methods "are prone to get stuck into local minima (falsely localize object parts) while missing full object extent during training"
- **Choe, Oh, Lee, Chun, Akata, Shim, *Evaluating Weakly Supervised Object Localization Methods Right*, CVPR 2020 — arXiv:2001.07437** — "the mixed policies for selecting τ has contributed to the illusory improvement of WSOL performances"; three-split protocol with **train-fullsup budget: ImageNet 10 / CUB ~5 / OpenImages 25 samples per class**; MaxBoxAcc, **MaxBoxAccV2** (δ ∈ {0.3,0.5,0.7}, all boxes), PxAP; "recent WSOL methods have not led to major improvements compared to CAM, when validated in the same data splits and same evaluation metrics"; **"Few-shot learning baselines... outperforms existing WSOL methods"**
- Detic — Zhou, Girshick, Dollár, Krähenbühl, Misra, ECCV 2022 — arXiv:2201.02605 — max-size loss; **label-free pseudo-box diagnostics: cover rate 92.8% (max-size) vs 69.0% (prediction-based); "consistency" = average IoU between assigned boxes across training iterations**
- Spot On — Mettes, van Gemert, Snoek, ECCV 2016 — arXiv:1604.07602 — trains on **points**, tests on **boxes**: 16,411 points / 29,802 train frames vs 15,835 boxes / 31,295 test frames; "a point is roughly 10-15 times faster to annotate than a box"
- OVR-CNN — Zareian et al., CVPR 2021 oral — arXiv:2011.10678. AutoLoc — arXiv:1807.08333. STPN — arXiv:1712.05080. Finding Action Tubes — arXiv:1411.6031 (frame-AP / video-AP origin; definitions not verified verbatim)

**Hardware**
- NVIDIA Video Encode and Decode GPU Support Matrix — https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new — RTX 3060 / Ampere: **1 NVENC, 1 NVDEC (5th gen)**; decode H.264 8+10-bit, HEVC 8+10-bit, VP9 8-bit (not 10/12), **AV1 8+10-bit = YES**. Two cards ⇒ **two** NVDEC engines total
- torchcodec — https://github.com/pytorch/torchcodec — **BSD-3-Clause**, Meta-maintained, v0.17, PyTorch 2.11+/Python 3.10-3.14; *"a PyTorch-native library for decoding and encoding media: videos, audio, and images, on CPU and CUDA GPU"*; *"On CUDA GPUs, TorchCodec supports decoding and encoding of videos and jpeg images"*; needs FFmpeg with NVIDIA codec support; **no published throughput comparison vs decord/PyAV**
- decord — https://github.com/dmlc/decord — **Apache-2.0**; NVDEC via `-DUSE_CUDA=ON` but *"only CPU versions are provided with PYPI now"*; 352 commits / 201 open issues; **last-commit date not verified (GitHub rate-limited) → maintenance status is a risk**
- **RTX 3060 bf16/fp16 tensor throughput = 25.5 TFLOPS** (FP32-accumulate rate; the GeForce halving) — re-derived from NVIDIA's own stated formula in the GA102 whitepaper and cross-validated against three published SKUs: RTX 3080 → 119.07 vs published 119; 3090 → 142.33 vs 142; 3070 → 81.25 vs 81.3. **Wikipedia's 51.2 / [102.4] cell is the FP16-accumulate / sparse figure and is 2x / 4x optimistic for autocast workloads.**
- **sm_86 SAM 2 measurements** (the only ones that exist, all on Ampere consumer/pro silicon):
  - https://github.com/facebookresearch/sam2/issues/768 — **RTX A6000, eager PyTorch @512px: Hiera-S 17.7 FPS, Hiera-L 10.9 FPS; CUDA graphs give 5.8x at 512px but only 1.4-1.9x at 1024px.** ⚠ **Single unresolved user report, and it is load-bearing for my throughput estimates — verify it first.**
  - https://github.com/flybroken/sam2-tensorrt — **A10 (sm_86), FP16, 6 TRT engines, single target: ~33 FPS / 30 ms**
  - https://github.com/tier4/sam2_trt_inference — **RTX 3070 Ti (sm_86), FP16, base+, 94 boxes: 414 ms**
  - ⚠ **TensorRT on SAM 2's memory attention measured a 3x REGRESSION on sm_86.** Export the image encoder only.
- **SAM 3.1 video: 32 FPS on an H100** `[measured-in-source]` → 0.8-3.4 FPS on a 3060 by compute/bandwidth scaling
- EfficientTAM — Apache-2.0 — **Ti 18M @ 96.2 FPS A100 / SA-V 70.7 J&F; S 34M @ 85.0 FPS / 74.5 J&F**
- NVDEC throughput — NVIDIA claims **748 fps 1080p on Ampere**; **592-667 fps measured on an RTX 3090**; torchcodec CPU decode **1,589 fps on 24 threads**
- PySceneDetect — BSD-3 — Adaptive detector @256px ≈ **2,080 frames/s** on CPU (derived). TransNetV2 / AutoShot / Katna — MIT. `video-keyframe-detector` — **GPL-3.0**
- Detector/re-prompt intervals, all verified from source: **DEVA every 5 frames (its own sweep: 5 is the knee, 7 falls off)**; **SAM-Track `sam_gap = 10`** (⚠ a secondary source said 100 — `model_args.py` says 10); **SAM 3 `recondition_every_nth_frame = 16`**; **Grounded-SAM-2 = 20**. Both SAM 2 and SAM 3 prefer **error-triggered** over periodic re-prompting
- **Export bakes in the vocabulary:** YOLO-World and YOLOE ONNX/TensorRT exports hard-code the class embeddings as graph constants, **and YOLO-World does so silently**. Runtime-variable alternatives: **TAO Grounding DINO** (token IDs are graph inputs) and `optimum-cli`-exported **OWLv2**
- **Text-embedding pre-compute payoff: 3.7x measured for YOLO-World; ≈0 for OWLv2 at small vocabularies; architecturally impossible for Grounding DINO**
- **No GPU-to-GPU P2P on GA106**, so cross-device CUDA IPC cannot work; `can_device_access_peer` is known to return misleading values. ⚠ The "Ampere GeForce has no P2P" conclusion is **inferential** — I found no NVIDIA statement naming Ampere GeForce specifically
- Caching cost derivations: SAM 2 image embeddings **~906 GB/hour** at 30 fps; full ViT patch tokens **~227 GB/hour**; boxes + RLE + CLS vector **~11 KB/frame ≈ 1.35 GB/hour**. Pre-`conv_s0/s1` `backbone_fpn` is **5.25x** larger than needed; `vision_pos_enc` is free to recompute
- bf16 vs fp16 on sm_86: **identical tensor-core rate**; fp16 has a documented overflow failure class in DETR-style models; SAM 2/3's own code uses bf16
- SAM-Track — **AGPL-3.0** + *"To utilize or further develop this project for commercial purposes through proprietary means, permission must be granted by us"*. MobileSAM — Apache-2.0. EdgeSAM — **NTU S-Lab License 1.0**. NeLux — **AGPL-3.0**. Panda-70M splitting code — **Snap Inc., non-commercial research only**. DEVA additionally depends on **Gurobi**
- OmDet-Turbo (HF transformers) — **Apache-2.0**, ~14 FPS estimated, built-in language cache — the most interesting permissive middle ground
- RTX 3060 12GB specs — https://en.wikipedia.org/wiki/GeForce_RTX_30_series — GA106, 3,584 CUDA cores, 1,777 MHz boost, **FP32 9.462 base / 12.74 boost TFLOPS**, 12 GB GDDR6, **360 GB/s**. ⚠ The same table's "FP16 tensor 51.2 / [102.4]" cell is the **FP16-accumulate / sparse** rate; for FP32 accumulate (what `torch.autocast` uses) the GeForce rate is halved to **25.5 TFLOPS** — see the derivation above. Plan with 25.5. (TechPowerUp returns HTTP 403 to automated fetches, so Wikipedia + NVIDIA's product page + the GA102 whitepaper were the sources here.)

### 7B. My own estimates (not measured anywhere)

- All "VRAM on 3060" figures marked `[my est]` in §2, and the 0.5-1.0 s/frame @1008px and 0.2-0.3 s/frame @560px SAM 3 figures in §6.0 (two derivations: latency scaling from the H200 number with a 15-25x batch-1 penalty, and a ~5 TFLOPs/frame FLOPs count at the utilisation implied by the H200 figure)
- The ~5 TFLOPs/frame FLOPs count itself (derived by hand from the verified ViT config: 1024 hidden, 32 layers, patch 14, 1008px → 5,184 tokens, window 24, intermediate 4,736)
- H200 ≈ 989 TFLOPS bf16 dense / 4.8 TB/s — used only as a scaling ratio
- The **~25-60 FPS @512px + CUDA graphs** propagation figure in §6.2. This extrapolates a **single unresolved user report** (sam2#768) and is the most fragile number in the report. Experiment: measure it on day one; the 7-15 FPS @1024px row is the safe fallback
- The ~14 FPS OmDet-Turbo and ~2-4 FPS OWLv2 figures
- ⚠ **Correction trail, recorded deliberately:** an earlier revision of this report used 25.5 TFLOPS, then "corrected" it to 51.2 on the basis of Wikipedia's tensor-compute cell, and has now reverted to **25.5**. The GeForce FP32-accumulate halving is real and 25.5 is the right planning number; the revised figure was independently validated by re-deriving three published SKUs from NVIDIA's own formula. **All latency and wall-clock figures in §6 now use 25.5.**
- Every wall-clock row in §6.2 and every disk-footprint row in §6.6
- The "300 frames / ±8 AP" power argument in §5.2 and the 3-6 s/box annotation-time figure
- The assumption of 30 s/clip at 30 fps, 6-10 instances/frame, 3-5 classes/frame
- The ranking of trackers by "degrades least under static+cluttered+occluded" in §2c
- MASA FPS (no published number)
- The inference that SAM 3 exemplars are **same-image only** (from "ROI-pooled visual features" + the paper never evaluating cross-image exemplars) — **this is the single most important thing to test**

### 7C. Known gaps and discrepancies I could not resolve

- **SAM 3 VRAM on a 12 GB card is unmeasured anywhere I could find.** The only hard data points are "<25 GB on H100 for long multi-class video" and "OOM on a 24 GB 4090". Benchmark it yourself first.
- **pHOTA's formula** (SAM 3 appendix F.5) — not retrieved.
- **No paper quantifies loss of generality after fine-tuning SAM/SAM2/SAM3 narrowly.** Many assert it; none measure it.
- MASA README (47.3/41.9) vs MASA paper Table 2 (47.0/40.8) — different detector backbone rows; both are "MASA", neither is wrong.
- DEVA's exact licence file contents — I confirmed a LICENSE.md exists and the Cutie/XMem lineage is MIT, but did not read DEVA's own file.
- DINOv3's licence terms — not read.
- **No numeric decoder-comparison table exists** (torchcodec's blog chart is a PNG), and I could not confirm `decord`'s last-commit date. No measured NVDEC fps for a 3060 specifically, and no `nvidia-smi` video-clock figure for it. The "decode is not the bottleneck" conclusion rests on a 10-100x margin and is robust to being wrong by an order of magnitude; the library *choice* is a recommendation.
- **No measured ONNX/TensorRT speedup for SAM 3, YOLO-World, YOLOE or OWLv2 on any GPU.** The only SAM 3 export data point is the discouraging 4090/FP32 one. The magnitude of SAM 2's `vos_optimized` speedup is also unpublished.
- **No published "detector stride k → J&F" curve for any VOS model.** DEVA's 3/5/7 merge-frequency row is the closest thing. Stride must be swept on your own gold set.
- **No `torch.compile` speedups measured for ViT/DETR/SAM on any sm_86 consumer GPU** — all official numbers are A100. No FlashAttention-2 benchmarks on sm_86. No measured effect of `allow_fp16_accumulation=True` on sm_86.
- **Latency and VRAM are unpublished for OWLv2, Grounding DINO, MM-Grounding-DINO, T-Rex2 and DINO-X.** There is also **no batch-size sweep published for any model** in this report.
- **No `can_device_access_peer` result on a stock-driver 30-series pair, and no NVIDIA statement naming Ampere GeForce specifically re P2P** — the "no P2P on GA106" conclusion is inferential. No `nvbandwidth` figure for a 3060. No systematic 2x3060 or 2x3090 pipeline-split benchmark exists.
- **Two load-bearing citations are single unresolved user reports** and are flagged in place: `sam2#768` (the A6000 CUDA-graph numbers, which drive the §6.2 best case) and a vLLM report of 4xRTX 3090 tensor-parallel corruption.
- One secondary-source error was caught: a summary claimed SAM-Track's `sam_gap` default is 100; `model_args.py` says **10**. That class of error propagates — treat any single-source number here as provisional.
- OccluBoost (BoxMOT's best-scoring tracker) has **no paper**; its 71.10 HOTA is a repo-reported number only.
- **No one has evaluated prompt ensembling, LLM descriptors, or exemplar prompting on scientific-instrument vocabulary.** Three independent full-text searches returned zero. Everything in §3.3 is transfer from plant pathology, rare medical events, aerial imagery and fine-grained cars.
- **No canonical paper on mAP-estimator variance vs eval-set size** (arXiv query returns 0 results). §5.2's variance law is a simulation.
- **PPI / active testing has not been applied to mAP or HOTA** — both frameworks estimate *means*; mAP is a non-linear functional of a ranking. Apply them only to decomposable components (per-class recall at fixed threshold, mean IoU of matched boxes, DetRe, AssA) and keep AP/HOTA bootstrap-only.
- **No established method for label-free mAP estimation in detection.** ATC / AutoEval / agreement-on-the-line are all classification-side.
- Cycle-consistency-of-time (Wang et al., CVPR 2019) uses cycle error as a **training** signal and never as a test-time quality measure. Using it as a metric is an extension — defensible via Kalal et al. (which *is* a failure-detection method), but not established practice for model selection.
- `frame-AP` / `video-AP` definitions from *Finding Action Tubes* not verified verbatim.
- DINOv3's and the Ultralytics YOLOE-26 checkpoints' availability/terms not fully pinned.

### 7C-bis Methodological traps to avoid when quoting these numbers

1. **Never compare DCLIP's, CuPL's and WaffleCLIP's published deltas to each other.** Their baselines are bare class name / 80-prompt ensemble / single template respectively — mixing them inflates by 3-5x. Only MPVR benchmarks them under one protocol.
2. **CLIP's prompt-engineering notebook reports ImageNet-V2 numbers** (55.93 / 83.36), not ImageNet.
3. **ZPE's intro and Table 1 disagree** (64.18/66.92/68.57 vs 63.94/66.37/68.31). Cite the table.
4. **WaffleCLIP and MPVR genuinely disagree and are both right** — about attribute fragments vs full sentences respectively. Report the tension, do not collapse it.
5. **MASA's README (47.3/41.9) and paper Table 2 (47.0/40.8)** are different detector rows; neither is wrong.
6. **SAM 3's +18-22 AP exemplar gain is a *same-image* GT box**, not few-shot transfer. Do not quote it as evidence for cross-clip exemplars.
7. **Resample clips, not frames or detections,** in every bootstrap. Detections within a frame are not independent, and frames within a clip have a design effect of up to ~9.

---

## 7D. The eight experiments that resolve the open questions (do these before building)

Every one of these is a few hours at most, and each one collapses a `[my estimate]` into a measurement.

| # | Experiment | Resolves | Decision it unblocks |
|---|---|---|---|
| 1 | Load `facebook/sam3.1` on one 3060, bf16, `image_size` ∈ {1008, 672, 560}, run `Sam3Model` on 20 lab frames with 15 class prompts using the cached-`vision_embeds` loop. Record peak `torch.cuda.max_memory_allocated()` and ms/frame. | **The single biggest unknown.** SAM 3 VRAM + latency on a 12 GB card. | Whether SAM 3 is the detector at all (vs OWLv2 / MM-GDINO) |
| 2 | Same, but `Sam3VideoModel.propagate_in_video_iterator` on one 30 s clip with 10 prompts, `inference_state_device="cpu"`, `video_storage_device="cpu"`. | Does end-to-end video PCS fit, and at what FPS? | Track A viability for the gold set |
| 3 | Hand-box 20 frames. Run SAM 3 text-only vs text + one same-image exemplar box. Measure AP50 on those 20 frames. | Does the **+18-22 AP** exemplar gain `[measured on COCO/LVIS]` transfer to lab equipment? | Whether the annotation budget goes to exemplar boxes or exhaustive boxes |
| 4 | Pool exemplar ROI features from frame A, inject into frame B's exemplar token (~50-line patch to the exemplar encoder). | Can SAM 3 do **cross-image** exemplars? (Reference API says no — §3.5-ii) | Whether the DINOv3 prototype bank is needed at all |
| 5 | SAM 3 keyframes @stride 8 + SAM 2.1-S/EfficientTAM-S propagation vs full SAM 3 video PCS, on 5 clips. Compare HOTA against hand-ID'd GT and wall-clock. | Is the ~20x cheaper Track B actually worse, and by how much? | The whole pipeline architecture |
| 6 | 3-way prompt bake-off on 50 frames: bare class name / curated template ensemble / ensemble + explicit hard negatives. Score IL_MCC and AP50. Also test `fume_hood` vs `fume hood`. | Does the measured IL_MCC 0.44→0.68 hard-negative gain reproduce? Does the ~30-point underscore effect bite you? | The class-registry schema (§3.8) |
| **7** | **SAM 2.1-S / EfficientTAM-S at 512px with `torch.compile(mode="reduce-overhead")` and padded static shapes, vs 1024px eager.** Measure FPS and J&F on 5 clips. | **The most fragile number in this report.** A single sm_86 user report claims **5.8x from CUDA graphs at 512px** vs 1.4-1.9x at 1024px. If true, Track B runs at real time; if not, it runs at ~0.3x. | Whether the 10x unlabelled pool takes 7 h or 50 h |
| **8** | Run CUDA's `simpleP2P` and `nvbandwidth` on the actual pair of cards. | Confirms there is no P2P on GA106 (expected) and measures the real PCIe link — one documented case saw bandwidth collapse to ⅓ with no root cause found. | The inter-process handoff design (§6.3) |

Order: **7 → 1 → 2 → 5** (architecture and cost), then **8** (30 minutes, do it while the others run), then 3 → 6 (prompt/exemplar design), then 4 (the upside spike).

---

## 8. What would change my recommendation

**1. If SAM 3 at 560px cannot be made to run end-to-end on a 12 GB 3060 without OOM.** This is my biggest open risk: a user reported CUDA-OOM in `propagate_in_video` on a **24 GB** 4090 `[measured, issue #511]`, SAM3-ASH measured **<25 GB peak** on an H100 for multi-class long video `[measured]`, and Meta has **no plans announced for smaller SAM 3 variants** (issue #219, no maintainer reply since Nov 2025). Spend your first hour on exactly this test. If SAM 3 video does not fit, my recommendation flips to **MM-Grounding-DINO-Swin-T (Apache-2.0, LVIS-minival 41.4 AP) or OWLv2-base + MASA + BoT-SORT**, with SAM 2.1-tiny for mask refinement — 3-6 GB, no gating, no restrictive licence, and OWLv2 brings a documented *cross-image* `image_guided_detection()` one-shot API that SAM 3 may not have.

**2. If cross-image exemplars can be made to work in SAM 3.** I have now *confirmed* they are not supported by the reference API (§3.5-ii), and the architecture pools exemplar features from the current image. But the exemplar encoder's inputs (position embedding + label embedding + ROI-pooled features) do not *logically* require same-image provenance. If a ~50-line spike shows you can pool from a stored exemplar frame and inject the token, the whole design collapses to one model — SAM 3 with a registry of text strings plus exemplar crops — and §3.5's DINOv3/OWLv2 fallback apparatus becomes dead code. Given that one same-image exemplar is worth **+18-22 AP** `[measured]`, that upside is large enough to justify the spike. As things stand, the **DINOv3 prototype bank over SAM 3's class-agnostic proposals is load-bearing** and deserves real engineering, with OWLv2's documented `image_guided_detection()` as the licensed, lower-risk alternative.

**3. If the SAM License is unacceptable to the project.** SAM 3 is under the custom **SAM License** (commercial use permitted, royalty-free; AUP + ITAR/trade-control restrictions; publication-acknowledgement requirement; no MAU cap; no restriction on using outputs to train other models; you own your derivatives) and the HF weights are **manually gated** `[all verified from the LICENSE file and HF API]`. If legal says no, or if gating blocks CI, drop to the **Apache-2.0 stack**: SAM 2.1 + MASA + MM-Grounding-DINO/OWLv2 + ByteTrack/BoT-SORT (all MIT/Apache-2.0). Explicitly avoid StrongSORT (GPL-3.0), YOLO-World (GPL-v3), YOLOE and BoxMOT (AGPL-3.0), CoTracker3 (CC-BY-NC) and anything API-only (T-Rex2, DINO-X) given the low-cost constraint.

**4. If the gold set shows text prompting is *useless* rather than merely weak.** My expectation has got worse during this research, not better. The anchors: RF100-VL zero-shot 15.2 mAP for SAM 3 / 15.7 for Grounding DINO (with a *laboratory imaging* domain in the mix); and on the closest published proxy for specialist technical vocabulary, CLIP-B/16 scores **0.307 on a 3-class task where chance is 0.333 — below chance** `[both measured]`. A rare-medical-event study reports zero-shot CLIP classifying everything as noise. If your measurement lands there, the response is **not** fine-tuning (§4) and **not** more prompt engineering — Udandarao et al. show *"the strong log-linear trend between concept frequency and zero-shot performance consistently holds across different prompting strategies"*, i.e. **prompts move the intercept, not the slope** `[quoted]`. The response is to make the pipeline **exemplar-primary**: text becomes a coarse proposal generator, and all class decisions are made by cosine similarity against a DINOv3 prototype bank. The design in §3 already supports this via the per-class `gate: exemplar` field — but if text collapses entirely you should also reconsider the detector, because OWLv2's `image_guided_detection()` is *natively* exemplar-driven whereas SAM 3's exemplars are same-image only.

**5. If CUDA graphs do not deliver ~5x on the propagator at 512px.** This is the most fragile load-bearing number in the report — it rests on a **single unresolved sm_86 user report** (sam2#768: Hiera-S 17.7 FPS eager @512px, 5.8x from CUDA graphs at 512px vs 1.4-1.9x at 1024px). If it holds, Track B runs at roughly real time and the 10x unlabelled pool is a single overnight job. If it does not, you are at **7-15 FPS** and the pool takes ~25-54 h — still tractable, but it changes the propagator choice (go to EfficientTAM-Ti, 18M params) and may push you toward distilling a YOLO student sooner (§6.5). **This is experiment 7 and it should be the first thing you run.**

**6. (Minor) If "two-stage" turns out unnecessary because YOLOE-26 weights appear.** YOLOE-26x reports **LVIS-minival 40.6 AP under text prompting** with YOLO-class latency (1.7-11.8 ms on T4 TensorRT for the detection models), and supports text / visual / prompt-free prompting `[measured]`. If the open-vocab checkpoints are actually published under terms you can accept (currently AGPL-3.0 + Enterprise, and the docs do not confirm the OV weights are released), the entire keyframe-stride + propagation apparatus could collapse into a single fast per-frame detector plus ByteTrack — which would be dramatically simpler. Check this before building.
