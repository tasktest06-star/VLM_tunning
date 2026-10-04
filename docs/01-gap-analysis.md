# Gap analysis of the sibling auto-labelling pipeline

> **This document analyses a different repository.** The findings below are against
> `tasktest06-star/customobjectdetection`, commit `8d1dc90` on branch
> `feature/auto-label-pipeline`, not against any code in this repository. It is kept here because
> the defects it describes are the concrete, checkable consequences of the research findings, and
> because anyone building this pipeline will hit the same twelve problems.

Findings against commit `8d1dc90` on `feature/auto-label-pipeline`. Every item below was read
directly from the source, and line numbers are given so each can be checked. Items are ordered by
how much damage they cause.

The architecture is sound. Discovery, frame extraction, zero-shot pseudo-labelling, quality
filtering, dataset generation, training, and a self-training loop is close to what the research
recommends. The problems are all in the details, and several are silent.

## Summary table

| # | Defect | Where | Severity |
|---|---|---|---|
| 1 | Validation set is pseudo-labelled, so the metric measures teacher agreement | `scripts/evaluate.py`, `src/annotation/coco_builder.py:97` | Critical |
| 2 | Agreement filter falls back to keeping everything it disagrees with | `src/training/self_trainer.py:85-87, 96` | Critical |
| 3 | Train/val split leaks and is reshuffled unseeded every round | `src/annotation/coco_builder.py:97-99` | Critical |
| 4 | All classes prompted at once, truncating silently past the text limit | `src/pseudo_labeling/grounding_dino.py:38` | High |
| 5 | Label matching is a bidirectional substring test, conflating fine-grained classes | `src/annotation/coco_builder.py:35-41` | High |
| 6 | Teacher re-runs identically every round, wasting a full labelling pass | `src/training/self_trainer.py:136-139` | High |
| 7 | AGPL-3.0 dependency with no licence file in the repository | `requirements.txt`, `configs/pipeline_config.yaml` | High |
| 8 | `self_training.confidence_threshold` is documented and set but never read | nowhere in any `.py` | Medium |
| 9 | `epochs` hardcoded to 30, ignoring config, on the self-training path | `src/training/self_trainer.py:131` | Medium |
| 10 | Quality filter re-applies the threshold the labeller already applied | `src/pipeline.py:68` | Low |
| 11 | Per-frame seeking makes extraction far slower than sequential reads | `src/video_pipeline/frame_extractor.py:46` | Medium |
| 12 | Flat image copy can collide across classes sharing a video stem | `coco_builder.py:104-106` | Medium |

---

## 1. The reported metric cannot detect failure

`scripts/evaluate.py` delegates to the trainer's `evaluate` on `data.yaml`. That file's `val`
split is produced by `COCOBuilder.build_dataset`, which splits the *pseudo-labelled* detections.
So the ground truth in validation came from the same teacher that produced the training labels.

Mean average precision therefore measures how well the student reproduces the teacher, including
the teacher's mistakes. It rises as the student conforms, and it cannot fall when the teacher is
wrong about a class. On a vocabulary where several classes have never been seen as boxes by any
released detector, the teacher will be confidently wrong, and this metric will report success.

**Fix.** Hand-annotate a small gold set, keep it entirely outside the pipeline, and report on that.
Nothing else in the project can be trusted until this exists. See `04-unlabelled-pool-and-annotation.md`
for the budget, and note that the two research passes disagree two-fold on its size, so that number
is still being resolved.

## 2. The agreement filter is close to a no-op where it matters

`_ensemble` keeps teacher boxes the student also found. But when the student finds nothing, the
docstring says it "falls back to all teacher boxes when the student finds nothing, so we don't lose
coverage on hard examples", implemented at lines 85-87, and again per-image at line 96:

```python
idx = kept if kept else list(range(len(td["boxes_xyxy"])))
```

The consequence inverts the intent. On easy images, where the student agrees, boxes are kept and
little is filtered. On hard images, where the student disagrees, *everything* is kept unfiltered.
So the filter removes boxes only where it was least needed, and preserves unverified teacher output
exactly where verification was the point.

There is a deeper problem. The student is trained on the teacher's labels, so student-teacher
agreement is not independent evidence. It measures how well distillation worked. The research
identified this as confirmation-bias amplification, and here it appears in an unusually direct form.

**Fix.** Agreement must come from two *independently pretrained* detectors with different training
mixes, not from a teacher and its own student. Grounding DINO paired with OWLv2 or SAM 3 gives real
independence. Drop the keep-everything fallback and let low-agreement images contribute nothing.

## 3. The split leaks, and moves under you

```python
random.shuffle(detections)                                  # line 97
n_train = int(len(detections) * self.train_ratio)           # line 98
splits = {"train": detections[:n_train], "val": detections[n_train:]}
```

Two separate problems.

**Leakage.** The unit of shuffling is a frame. Frames extracted from one video at 0.5 frames per
second are near-duplicates of each other, so the same scene, lighting, and physical object appear
on both sides of the split. Reported performance will be inflated and will not survive contact with
a new recording.

**Non-comparability.** There is no seed, and `build_dataset` is called once per self-training round
at `self_trainer.py:127`. Every round therefore gets a different random split. Metrics across rounds
are computed on different data, so round-over-round changes cannot be attributed to the method.

**Fix.** Group the split by source video, and ideally by recording session or room. The filename
produced at `frame_extractor.py:51` already embeds the video stem, so the grouping key is available
without new metadata. Seed the shuffle, and compute the split once rather than per round.

## 4. Prompting every class at once truncates silently

```python
text_prompt = ". ".join(class_names) + "."                  # line 38
```

Grounding DINO's text encoder has a bounded input length. Past that bound the prompt is truncated,
and classes near the end of the list are simply never detected. Nothing in the code warns about
this, so a growing vocabulary degrades invisibly: adding classes quietly disables the ones that fall
off the end.

This is also the wrong economics. Open-vocabulary detectors cost roughly linearly in the number of
concepts prompted, and SAM 3 is reported at around 300 milliseconds per concept per frame. With an
open-ended class list that cost is unbounded.

**Fix.** Prompt each clip with only the classes its own video-level label set names, which is the
project's main free lunch. It bounds cost, removes the truncation risk, and raises precision by
making most false positives impossible. For unlabelled footage, use a clip-level multi-label
classifier trained on the labelled clips to predict the vocabulary to prompt with.

## 5. Bidirectional substring matching conflates similar classes

```python
for cls in self.class_names:
    if cls.lower() in label or label in cls.lower():
        return cls
```

Two failure directions, and the first match wins, so behaviour depends on class list order.

`cls in label` means the detector returning `test tube holder` matches the class `test tube` if that
class is listed first. `label in cls` means a returned fragment `tube` matches the class `test tube`.
For a fine-grained vocabulary where `pipette` and `volumetric pipette`, or `test tube` and
`test tube holder`, both exist, labels are silently cross-assigned.

This matters more than it looks, because `grounding_dino.py:55` returns the detector's *text spans*
rather than clean class names. With a period-separated prompt those spans can be fragments or merged
across class boundaries, and they feed straight into this matcher.

**Fix.** Match exactly against a known vocabulary, normalising case and whitespace. Resolve
ambiguity by longest match rather than first match. Count and report unmatched labels instead of
dropping them silently at line 60, because a high unmatched rate is the signal that prompts need
rewriting.

## 6. The teacher pass is recomputed identically every round

```python
teacher_dets = self.teacher.label_batch(image_paths, self.class_names)   # line 136
student_dets = self._student_predict(model_path, image_paths)            # line 139
```

The teacher is in `eval()` mode with fixed thresholds and the same prompt and images every round, so
`teacher_dets` is identical each time. Only the student changes. Rounds beyond the first therefore
add no new teacher information, they just re-filter one fixed teacher output against a progressively
more teacher-conformant student. That is exactly the "diminishing returns after 2-3 rounds" the
README reports, and it is structural rather than empirical.

It is also the most expensive operation in the pipeline, repeated for no benefit.

**Fix.** Compute teacher labels once and cache them. If rounds are to add information, something
must actually change between them: the prompt set, the thresholds, the frames, or a second
independent teacher.

## 7. Licence exposure

`requirements.txt` pins `ultralytics>=8.0.0` and the config trains `yolov8n.pt`. Ultralytics is
AGPL-3.0. The repository currently has no licence file at all.

This is not theoretical. Every YOLO baseline in the published lab-equipment papers carries the same
encumbrance, so their reported numbers are citable but their weights are not usable in a
non-copyleft project. The research found a clean substitute set: RF-DETR and RT-DETR for detection
and SAM 2.1 for propagation are Apache 2.0 for both code and weights. SAM 3 is usable commercially
under its own licence. The existing `apache2-rtdetr-pipeline` branch suggests this was already
noticed.

Full table in `02-datasets-and-licences.md`.

## 8 and 9. Config keys that do nothing

`self_training.confidence_threshold` is set in `configs/pipeline_config.yaml` and documented in the
README, but a search across every `.py` file finds no reader for it. The student inference threshold
is the hardcoded default in `_student_predict`. The two happen to coincide at 0.50 today, so there is
no visible symptom, and tuning the config silently has no effect.

Separately, `self_trainer.py:131` passes `epochs=30`, ignoring `training.epochs`, which
`pipeline.py:159` honours only on the non-self-training path. So enabling self-training silently
changes the training length. `agree_iou` is likewise reachable only as a default and is absent from
the config entirely.

**Fix.** Wire these through, or delete them from the config and README. A config key that is read
nowhere is worse than a missing one, because it invites tuning that cannot work.

## 10 to 12. Smaller items

**Redundant threshold.** `pipeline.py:68` sets the quality filter's `min_score` to the same
`box_threshold` the labeller already applied, so the first of the filter's three stages removes
nothing. Harmless, but it hides the fact that there is no independent quality gate.

**Extraction is seek-bound.** `frame_extractor.py:46` calls `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)`
once per kept frame. Random seeking is far slower than a sequential read with skipping, and on some
codecs it is inexact, so the frame you get is not the frame you asked for. This matters because
decode throughput is a real constraint when the host has few cores.

**Filename collisions across classes.** `frame_extractor.py:51` includes the video stem, so names
are unique within a class directory. But `coco_builder.py:104` copies into one flat directory using
only `src.name`, and line 105 skips the copy when the destination already exists. If the same video
is downloaded for two classes, or two classes draw videos with identical stems, the second frame's
annotations silently attach to the first frame's image.

## What is correct and should not be changed

Worth stating, because a gap analysis reads more negatively than the code deserves.

The non-maximum suppression in `quality_filter.py` is a correct greedy implementation, and the
index mapping back through `orig_idx[k]` at line 94 is right. Per-class suppression is the right
choice. The class index order is consistent between `coco_builder.py:119` writing `names` and
`self_trainer.py:62` reading it back. The error path in `_student_predict` appends nothing while the
teacher's appends a placeholder, which looks like an alignment bug but is safe, because `_ensemble`
keys by image path at line 81 rather than by position. The blur rejection and the area-ratio bounds
are sensible guards.
