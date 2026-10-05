# Module reference

Forty-one modules, grouped by what they are for. The public name list for each
comes from its own `__all__`, so this reflects the code rather than intent.

A module marked **GPU** imports a tensor library lazily and was never executed
in this repository. Everything else runs and is tested with the standard
library alone.

## Start here

| Module | What it is for |
|---|---|
| `vlmlab.types` | The record types everything else agrees on. Frozen dataclasses, zero intra-package imports, so nothing creates a cycle |
| `vlmlab.config` | Typed configuration where every value carries a provenance tag and eight fields cannot be overridden |
| `vlmlab.pipeline` | The orchestration. Read `process_clip` to see the whole flow in one function |
| `vlmlab.cli` | Nine subcommands. `python3 -m vlmlab.cli --help` |

`types`: `Box` `Detection` `Track` `TrackPoint` `ClipRecord` `FrameGrid` `Chunk`
`BoxSource` `RejectReason` `Verdict` `MatchTable` `APResult` `CIResult`
`PermResult` `MDEResult`

`config`: `Config` `DetectorConfig` `PropagatorConfig` `CascadeConfig`
`EvalConfig` `LoraConfig` `StudentConfig` `Param` `Provenance` `FROZEN_PATHS`
`ConfigError` `FrozenFieldError` `MissingRequiredError`

`pipeline`: `process_clip` `run_pipeline` `stitch_tracks` `ClipResult`
`PipelineError`

## Vocabulary and prompting

| Module | What it is for |
|---|---|
| `vlmlab.registry` | Class names, paraphrases, confusable siblings, and the homonym guards. The most load-bearing data in the project |
| `vlmlab.prompts` | Builds one clip's prompt set: generic localisation, the clip's own labels, and hard negatives |
| `vlmlab.naming` | Names generic detections with a crop classifier. Without this, generic localisation contributes nothing |

`registry`: `Registry` `ClassEntry` `DEFAULT_REGISTRY`
`GENERIC_LOCALISATION_PROMPTS` `levenshtein` `AmbiguousMatch` `UnknownLabel`

`prompts`: `build_clip_prompt_plan` `build_localisation_prompts` `PromptPlan`
`FineGrainedPromptError`

`naming`: `name_detections` `is_generic` `NamingResult`

**Why `registry` matters more than it looks.** In one major dataset the fume
hood entry is a kitchen range hood, and it is a *frequent* class, so a detector
is confidently trained on the wrong object under exactly the name you would
prompt with. The shaker entry is a condiment shaker. `resolve` refuses both,
takes the nearest edit-distance match rather than the first within threshold, so
resolution does not depend on declaration order, and gates substring matching on
a per-class list of known false friends.

## Frames, splits, geometry

| Module | What it is for |
|---|---|
| `vlmlab.frames` | Frame-grid planning and chunking. Raises rather than guessing |
| `vlmlab.splits` | Session-grouped folds, and the grouped-versus-ungrouped leakage gap |
| `vlmlab.geometry` | Overlap, suppression, box-format conversion. No accelerated path, deliberately |

`frames`: `plan_frame_grid` `chunk_indices` `keyframe_indices`
`validate_resolution` `estimate_vision_tokens` `ZeroFrameCountError`
`UnderDeliveredError` `ResolutionError`

`splits`: `group_kfold` `leave_one_group_out` `choose_fold_scheme`
`balance_report` `grouped_vs_ungrouped_gap` `leaky_clip_kfold`
`TooFewGroupsError`

`geometry`: `iou` `iou_matrix` `nms` `nms_per_class` `xyxy_to_xywh`
`xywh_to_xyxy` `xyxy_to_cxcywh` `cxcywh_to_xyxy` `normalise` `denormalise`
`clip_to_image` `area_ratio` `min_side`

**`leaky_clip_kfold` is not for use.** It exists as a negative control in the
test suite: if the leakage assertions cannot detect a deliberately leaky
splitter, they would not detect a real mistake either.

## Using the clip label

| Module | What it is for |
|---|---|
| `vlmlab.federated` | Splits the vocabulary into positive, negative and unknown for one clip. Shared by the exporter and the metrics so they cannot disagree |
| `vlmlab.mil` | The accept-and-reject cascade. Clip-label test first and absolute |
| `vlmlab.consistency` | Temporal corroboration and cross-model agreement |

`federated`: `derive` `FederatedLabels` `FederatedError`

`mil`: `run_cascade` `presence_score` `clip_passes_presence` `CascadeResult`
`background_boxes` `cascade_summary`

`consistency`: `temporal_consistency` `cross_model_agreement` `track_stability`
`AgreementReport`

**Every rejection carries a reason code.** Without that the cascade cannot be
tested stage by stage, and a silent drop is indistinguishable from a bug.

## The gold set

| Module | What it is for |
|---|---|
| `vlmlab.gold` | Which frames to hand-annotate, the annotation format, and the records scoring consumes |

`gold`: `select_gold_frames` `GoldSet` `GoldFrame` `GoldBox`
`annotation_budget` `GoldError`

Three frames per clip across **every** clip, at interior quantiles. Spread beats
depth by more than fivefold in effective sample size at equal cost, and dense
annotation of a few clips is the worst available use of the budget.

## Evaluation

| Module | What it is for |
|---|---|
| `vlmlab.eval.report` | Gold set plus predictions in, a defensible number out. Start here |
| `vlmlab.eval.metrics` | Average precision, split into matching and curve |
| `vlmlab.eval.stats` | Cluster bootstrap, group permutation test, both design effects, detectable difference |
| `vlmlab.eval.dist` | Incomplete beta, inverse Student-t, exact binomial intervals. This is what avoids needing a numerics library |
| `vlmlab.eval.protocol` | Fold scheme and the one global threshold |
| `vlmlab.eval.tracking` | Detection and association accuracy over tracks |
| `vlmlab.eval.crosscheck` | Optional comparison against the compiled implementation. Skipped when absent |

`report`: `score` `compare` `Report` `ComparisonReport`

`metrics`: `average_precision` `mean_average_precision` `match_detections`
`ap_from_matches` `per_class_recall` `GroundTruth` `Prediction`

`stats`: `bootstrap_ci` `paired_group_permutation_test` `kish_design_effect`
`icc_one_way` `design_effect_from_icc` `effective_sample_size` `mde_paired`
`post_hoc_power` `choose_test` `spread_vs_depth`

`dist`: `betainc` `t_cdf` `t_ppf` `normal_ppf` `clopper_pearson` `wilson`
`log_beta` `betacf`

`protocol`: `build_protocol` `choose_global_threshold` `ThresholdChoice`
`ProtocolReport` `LeakageError`

`tracking`: `hota` `track_summary` `ALPHAS` `TrackingError`

**Three things to know.** `match_detections` is separate from `ap_from_matches`
because the permutation test recomputes the metric hundreds of times and only
the cheap half should re-run. There is deliberately no per-class threshold
interface anywhere, because any split here holds two to six positives per class
and a threshold fitted on that has a standard error wider than the interval
being searched. And `choose_global_threshold` raises if you hand it in-fold
scores.

## Exporters

| Module | What it is for |
|---|---|
| `vlmlab.export.coco` | Annotation writers. Ground truth and predictions are separate functions so they cannot be confused |
| `vlmlab.export.tao` | Tracking file carrying negative categories, for federated evaluation |
| `vlmlab.export.rtdetr_yaml` | Student dataset descriptor. Write-only |

`coco`: `write_ground_truth` `write_predictions` `build_categories`
`to_model_index` `from_model_index`

`tao`: `write_tao` `EVAL_LIBRARY_PIN` `EVAL_MAX_DETECTIONS`

`rtdetr_yaml`: `render_dataset_yaml` `write_dataset_yaml`

**Two operational constants travel inside the tracking file.** Use the
maintained fork of the evaluation library pinned at 1.3.0, because upstream does
not import on a modern numerics library. And set maximum detections to
unlimited: the default cap of 300 is exactly the truncation that moves
rare-class precision by several points.

## Data

| Module | What it is for |
|---|---|
| `vlmlab.data.public` | The four permissively licensed box datasets, mapping onto this vocabulary, and session re-splitting |
| `vlmlab.data.imagenet21k` | Classification crops for the fine-grained head. Licence-gated |
| `vlmlab.data.rf100vl` | The free dry run before spending any annotation budget |

`public`: `PUBLIC_DATASETS` `AVOID` `DatasetSpec` `download_instructions`
`normalise_yolo_export` `resplit_by_session` `attribution_notice`
`coverage_against_registry`

`imagenet21k`: `SYNSETS` `MISCITED` `LICENCE` `synsets_for_registry`
`build_crop_plan` `coverage_report` `LicenceError`

`rf100vl`: `CLUSTERS` `LICENCE` `recommended_cluster` `expectation_bracket`
`dry_run_plan` `WITHDRAWN_PROXY`

**`resplit_by_session` is not optional.** Those archives ship a random split
over frames, so near-duplicates straddle it and their own validation figures are
inflated. Reusing their split reproduces the exact leakage this project exists
to avoid.

## Backends

| Module | What it is for |
|---|---|
| `vlmlab.backends.base` | The three interfaces, with capability flags. Imports with nothing installed |
| `vlmlab.backends.fake` | Deterministic stubs with a call log. What lets the pipeline run here |
| `vlmlab.backends.sam3` | **GPU.** Keyframe detector, image path |
| `vlmlab.backends.sam2` | **GPU.** Propagator, declares mask-derived boxes honestly |
| `vlmlab.backends.crops` | **GPU.** Frozen-backbone crop classifier |
| `vlmlab.backends._torch_guards` | **GPU.** Guards that read the effective value back after loading |

`base`: `KeyframeDetector` `Propagator` `CropClassifier` `validate_plain_python`
`inference_mode` `PlainPythonViolation` `SessionClosedError`

`fake`: `FakeDetector` `FakePropagator` `FakeCropClassifier` `FakeMaskDetector`

`_torch_guards`: `assert_attn_impl` `assert_on_cuda` `assert_dtype`
`assert_quantisation` `assert_fused_cross_entropy` `assert_inference_mode`
`GuardFailure`

**Capabilities are declared, not inferred from a model name.** The orchestration
needs three facts: whether a presence score exists, whether boxes come from a
real detection head, and how many concepts one session accepts. The reference
detector accepts exactly one, because adding a prompt resets session state.

## Training

| Module | What it is for |
|---|---|
| `vlmlab.train.schedule` | Rank, alpha, steps, warmup. Pure arithmetic, and exactly what the traps corrupt |
| `vlmlab.train.retention` | Held-out-class retention, reported by median |
| `vlmlab.train.sweep` | Weight-average sweep and operating-point choice |
| `vlmlab.train.rtdetr_dataset` | Annotation index, index offset, empty targets |
| `vlmlab.train.lora_adapter` | **GPU.** Low-rank fine-tune |
| `vlmlab.train.rtdetr` | **GPU.** Student detector |

`schedule`: `plan_schedule` `rank_for_label_budget` `resolved_alpha`
`steps_per_epoch` `total_steps` `warmup_steps` `effective_batch_size`
`ScheduleError`

`retention`: `retention_report` `choose_held_out_classes` `format_retention`
`RetentionError`

`sweep`: `run_sweep` `choose_operating_point` `format_sweep` `DEFAULT_ALPHAS`
`SweepError`

`rtdetr_dataset`: `CocoIndex` `Target`

**Report the median, never the mean.** In the reference experiment the mean
across withheld domains fell nine percent while the median fell sixty-one. A
mean-based retention metric hides exactly the failure it is meant to detect, and
`retention_report` warns when it sees that pattern.

## Operations

| Module | What it is for |
|---|---|
| `vlmlab.preflight` | Pure checks over an environment mapping or a configuration |
| `vlmlab.checkpoint` | Resume, serialised as JSON rather than pickled |
| `vlmlab.logging_setup` | Run-scoped logging, plus the one line that audits the traps |

`preflight`: `run_pure_checks` `format_report` `check_single_visible_device`
`check_no_flash_attention` `check_config_frozen_fields` `check_concept_budget`
`check_frame_grid` `check_provenance` `Finding` `PreflightFailure`

`checkpoint`: `Checkpoint` `config_hash`

`logging_setup`: `get_logger` `log_realised_config`

**The checkpoint uses JSON on purpose.** Pickling arbitrary objects would
happily persist a tensor into an artefact, which defeats the plain-Python
boundary the rest of the design depends on. It also stores a configuration hash
and invalidates prior steps when that changes, so resuming after an edit cannot
silently mix two configurations.
