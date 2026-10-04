# Engineering traps, and whether anything here is measurable

> Output of the setup-specific research sweep and its red team. Every trap below was verified by
> reading library source, not inferred. Read this before writing code and before trusting any number
> you produce.

## 1. The finding that reframes the project

**The binding constraint on measurement is the number of distinct recording sessions, not the number
of clips.** A group-level bootstrap on macro mean average precision floors at:

| Recording sessions | 95% interval width on macro mAP |
|---|---|
| 4 | 14.9 points |
| 6 | 13.2 |
| 8 | 12.5 |
| 10 | 11.1 |
| 15 | 9.3 |
| 20 | 8.2 |

If the 150 clips came from 5 laboratory rooms over a few days, the honest interval is about 13 points
wide, **and labelling 150 more clips in those same rooms does not narrow it at all.**

It follows that most method comparisons this project would want to make are not resolvable. With 40
held-out clips, 10 classes, about 4 positives per class, and two methods sharing a backbone so their
errors correlate at 0.8, a paired bootstrap has:

- 12% power to detect a true 6-point gain
- 44% power at 12 points
- 80% power only at **19 points**

So staged go/no-go gates of the form "proceed if average precision at 0.5 exceeds 40" **will fire on
noise**. Per-class average precision with 3 positives has an 87-point range at fixed true
separability.

**Two consequences that no other document states.**

Merging confusable classes does **not** rescue power. Simulation gives 48% at ten classes with four
positives and 45% at five merged classes with eight. Dropping terms from a macro average raises its
variance about as fast as extra positives per class lower it.

The only thing that works is grouped repeated cross-validation over all 150 clips, which reaches 97%
power. But then you cannot also hold a clean untouched test set. **At 150 clips you must choose
between one trustworthy number and the ability to rank methods.** Choose deliberately, and say which
you chose.

Below about 15 groups, use a group permutation test. A bootstrap is resampling too few exchangeable
units to be informative, and 10,000 resamples buys nothing over 2,000.

## 2. The architectural refinement that matters most

Fine-grained instrument names do not work as detection prompts. A surgical-instrument study using
SAM 3 found the instrument name "could not be directly used due to a large domain gap" and fell back
to the generic prompt "tool" plus a classifier trained on the resulting crops.

**So split localisation from classification.**

- **Localise class-agnostically**, with generic prompts: "laboratory instrument", "machine on a laboratory bench", "glassware", "handheld lab tool". These are in-vocabulary and they work.
- **Classify the crops finely**, with a SigLIP 2 or DINOv3 crop classifier trained under the video-level label as a multiple-instance constraint.

This directly solves the problem that centrifuge, autoclave, orbital shaker, spectrophotometer and
fume hood have never been seen as boxes by any released detector. You stop asking the detector to
know what a spectrophotometer is.

Zero-shot reality on the relevant benchmark: the best open-vocabulary detector reaches 15.7 mean
average precision, with another at 13.6 and a frontier multimodal model at 11.6. SAM 3 at ten shots
per class reaches 35.7 on the same benchmark. Few-shot adaptation more than doubles zero-shot, which
is the strongest single argument for spending annotation effort rather than prompt effort.

**SAM 3's presence head is the right multiple-instance gate.** It scores whether a concept is in the
frame at all, separately from localising it, which makes it exactly a frame-level classifier. Take
the maximum over frames of the presence score for an exact test: a labelled-positive clip must have
some frame scoring high, and a labelled-negative clip must not.

**Hard negatives, concretely.** For each clip, prompt the 3 to 5 most confusable classes that are
*not* in its label set. Any track they produce is a guaranteed false positive, so export it as an
explicit background box for the student. This mirrors how the model's own training negatives were
built.

## 3. Verified engineering traps

Each was confirmed by reading source. Any one of these silently invalidates a run.

| Trap | What actually happens | Fix |
|---|---|---|
| **Attention defaults to eager** | The largest VRAM lever in the whole corpus, and it appears in no recipe. At 8 frames by 784 patches, eager attention materialises about 1.26 GB **per layer** across 24 layers, which OOMs a 12GB card instantly. SAM 3 at 5,184 tokens is 0.86 GB per layer. Every quoted VRAM figure silently assumes otherwise | Pass `attn_implementation='sdpa'` on every model and encoder call |
| **SAM 3 runs on CPU by default** | `init_video_session` defaults `inference_device='cpu'`, and the published documentation snippet passes an undefined variable. Forget to pass `'cuda'` and it will not error, it will just run 50 to 100 times slower, and you will conclude SAM 3 is unusable on a 3060 | Pass `inference_device='cuda'` explicitly, and set `dtype` on both `from_pretrained` and `init_video_session` since it defaults to float32 |
| **Omitting inference mode costs 8 GB per frame** | A user measured 8 GB per frame without it. With it, the image detector at batch 1 peaks near 6 GB and fits | Wrap everything in `torch.inference_mode()`. Also chunk clips to 300 frames and reset the session between clips, since inference states accumulate and are not released |
| **4-bit quantisation defaults are wrong three ways** | The quantisation type defaults to FP4 rather than NF4, double quantisation is off, and the compute dtype resolves to **float32**. On this GPU generation that pushes dequantised matrix multiplies onto the roughly 12.7 TFLOPS float32 path instead of the roughly 25.6 TFLOPS bf16 path. A silent two to four times slowdown plus measurably worse quality, which gets blamed on the card | Set the quantisation type to NF4, enable double quantisation, and set the compute dtype to bfloat16 |
| **Video decode is the wall clock, not the GPU** | One toolkit seeks to the start then iterates **every** frame with PyAV, testing frame membership against a Python list, then converting to an image. Per clip, per epoch, inside the collator, with dataloader workers defaulting to zero. A 30-second clip at 30 frames per second means 900 full decodes to retrieve 16 frames | Pre-extract frames to JPEG once and feed images |
| **Frame sampling collapses silently, two ways** | Requesting 1 frame per second with a 16-frame cap yields **6** frames on a 6-second clip, so every downstream figure describes a configuration you never ran. Far worse: when the container reports zero frames, which is routine for WebM, fragmented MP4, phone recordings and remuxes, the code returns indices 0 to 15, meaning **the first half-second of every clip**, with no warning | Set the rate so the cap binds, and log the realised frame grid on the first batch |
| **A kernel-optimisation flag is a verified no-op** | One toolkit's own fused-kernel flag has no branch for the relevant model type. It logs that the model is unsupported and returns. Without the fused loss, logits over a 152,000-token vocabulary are materialised three times over, about 2.31 GiB at 16 frames and a hard failure at 64 | Use the underlying library's own flag instead, then assert it applied by checking that a labelled forward pass returns no logits |
| **Evaluation batch size defaults to 8** | Combined with per-epoch evaluation, this adds 1.5 to 2.5 GiB at the end of **epoch one**, after you have already concluded the configuration fits. The likeliest cause of a late surprise failure | Set it to 1 |
| **Resolution is wrong in both directions** | One toolkit defaults video frames to 256 by 256, so you train a fine-grained classifier on thumbnails. The raw processor with the shipped config does the opposite, sending 16 frames of 1080p through to about 12,000 vision tokens and roughly 25 GiB | Set resolution explicitly, and never rely on either default |
| **The launcher silently uses both GPUs** | It switches to distributed training whenever more than one device is visible, so a two-card box becomes a two-way data-parallel job. That doubles effective batch and halves optimizer steps, dropping from about 15 to about 7 steps per epoch, which changes warmup and the schedule far more than the throughput gain is worth | Pin visible devices to one card, or halve gradient accumulation deliberately |
| **Flash attention cannot be installed from the package index** | Only a source tarball is published, no wheels. A from-source build takes two to six hours or gets killed for memory. It does support this GPU generation, but the attention implementation built into PyTorch is sufficient and free here. The newest generation is datacentre-only, and FP8 does not exist on this card at all | Use the built-in scaled dot-product attention |

## 4. Two corrections to the hardware story

**12GB is not your binding constraint.** Every frozen-feature recipe peaks at 1.5 to 3.5 GB, and a
low-rank fine-tune of a 2B-parameter vision-language model peaks around 6.0 to 6.5 GB on one card.
Your 150 labels are the constraint, and the measurability floor above is the real one.

**Never use multi-GPU parallelism here.** No data parallel, no sharding, no tensor or pipeline
parallelism, ever. Run two independent single-GPU processes sharding clips between them. Zero
inter-card traffic and near-linear throughput, which is the opposite of what sharding over a
no-peer-to-peer PCIe link would give.

## 5. Do not fit per-class thresholds

With 150 clips, about 12 classes, and 2 to 3 positives per clip, any held-out split holds 2 to 6
positives per class. A threshold fitted on that has a standard error wider than the interval being
searched. Use **one global threshold** tuned on pooled out-of-fold scores, headline the
threshold-free metric, and report per-class figures with bootstrap intervals without tuning twelve
free parameters on thirty clips.

## 6. The recommended sequence

**Do not start by fine-tuning. Start by proving you can measure anything.**

1. **Build a session-disjoint held-out set first**, with a group identifier on every clip. Publish the grouped against ungrouped gap as your leakage estimate. This is the single highest-value hour in the project.
2. **Establish the zero-shot floor** with a frozen image-text model and well-built prompts.
3. **Establish the frozen-feature floor**: cached features from two backbones, a multi-label multiple-instance head, and a learning curve. Minutes of compute, 2 GB of memory.
4. **Then** run the low-rank fine-tune, judged against that floor on the **same grouped folds**, with a pre-registered minimum detectable difference of 10 to 15 points.

At roughly 15 clips per class, the published spread between competing classifier heads is about 4
points. **A fine-tuning win smaller than your minimum detectable difference is not a win.**

Group and session leakage is the single largest risk, and it is near-certain with 150 clips from one
or two rooms. Several clips almost certainly show the same physical instrument on the same bench
under the same lighting. Nearest-neighbour and cache-based methods are worst affected, because they
can literally retrieve a near-duplicate. Grouping must be by **session**, not merely by clip, or you
will still be matching backgrounds.

## 7. Honest ceiling, by class difficulty

| Class group | Expected macro mean average precision |
|---|---|
| Visually distinct: fume hood, autoclave, microscope, glassware | 0.85 to 0.95 |
| Overall | 0.65 to 0.85 |
| Confusable benchtop siblings: centrifuge against shaker against incubator, balance against spectrophotometer, pipette variants | **0.40 to 0.70** |

Individual class figures carry 25 to 40 point intervals at 3 to 5 test positives.

**Moving the confusable siblings requires 50 to 100 clips per class from different rooms, cameras and
instrument brands. It does not require more method work.** That is the most useful thing in this
document: the remaining error is a data-diversity problem, not a modelling problem.

## 8. One open question that blocks the measurement plan

**How many distinct recording sessions, rooms, and physical instrument units are behind the 150
clips?** Everything in section 1 depends on it. If the answer is four or five, the honest interval is
about 13 points wide and the project should be scoped as an annotation-assist tool with qualitative
acceptance rather than a measured detector.
