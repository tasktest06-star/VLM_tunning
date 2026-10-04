# Datasets, detector coverage, and licences

Reference tables for a laboratory-equipment vocabulary. Sizes and licences were verified against
figshare, Zenodo, Mendeley, DataCite, and the repositories themselves. Category counts marked as
grepped were obtained by downloading and searching the actual category files rather than trusting
dataset documentation.

Some rows are still under adversarial verification. Those are marked. Treat an unmarked row as
checked against a primary source, and treat every number as what the source says rather than as
what this project will reproduce.

## 1. Public box-annotated laboratory data exists

This is the single most useful finding for a project that believed it had zero box labels. Roughly
22,000 box-annotated laboratory-apparatus instances are downloadable under CC BY 4.0.

| Dataset | Content | Size | Licence | Locator |
|---|---|---|---|---|
| Chemistry Lab Image Dataset, 25 apparatus categories | Real lab photos, YOLO boxes | 4,599 images, 6,960 instances | CC BY 4.0 | doi 10.6084/m9.figshare.29110433 |
| Physics Lab Equipment Image Dataset | Photos, boxes and polygons, 27 classes | 5,121 images, 8,700 instances | CC BY 4.0 | doi 10.6084/m9.figshare.30984658 |
| Annotated Chemical Apparatus Image Dataset | Frames from smartphone video of experiments, 6 apparatus classes plus hands | 5,078 images | CC BY 4.0 | data.mendeley.com/datasets/8p2hvgdvpn/1 |
| HeinSight4.0 | Video-derived frames, boxes, plus released weights | 6,031 phase and 6,523 vessel images | Data CC BY 4.0, code MIT | zenodo.org/records/15605098 |
| LabPics / Vector-LabPics | Instance and semantic segmentation of vessels and phases | Roughly 7,900 across two versions | **Conflicting.** Zenodo states MIT, project page states CC BY-NC-ND 4.0 | zenodo.org/record/4736111 |

Known class names in the chemistry set include beaker, conical flask, glass rod, round-bottom flask
in one, two and three neck variants, calorimeter, separating funnel, burette stand, volumetric
pipette, pipette, reagent bottle, test tube and test tube holder, Buchner funnel, mechanical balance,
precision scale, volumetric flask, measuring cylinder, and wash bottle.

**Three caveats that matter more than the sizes.**

These benchmarks are close to saturated. The chemistry paper reports mean average precision at 0.5
of 0.992 with RF-DETR. A 25-way task at 0.99 indicates one institution, few cameras, and mostly
isolated objects on clean benches. Expect a large drop on cluttered handheld footage, and do not
quote these numbers as achievable on your own data.

All three main sets are Roboflow exports, recognisable from letterboxed 640 by 640 images, hashed
filenames, and random splits. Check for near-duplicate leakage across the split before reproducing
any published figure. This is the same defect described as item 3 in the gap analysis.

Leave LabPics alone until the licence conflict is resolved in writing. A no-derivatives term would
arguably forbid training on it.

## 2. The gap that should direct your labelling budget

**No public boxes exist anywhere for centrifuge, autoclave, orbital shaker, spectrophotometer, or a
real laboratory fume hood.** Pipette and analytical balance exist only inside the chemistry dataset.

Those five classes must come from your own footage, from ImageNet-21k crops, or from synthetic
rendering. Spending annotation effort on beakers and flasks duplicates work that is already done and
freely available.

## 3. Which released detector has actually seen your classes

Zero-shot performance depends heavily on whether a class ever appeared as a human-drawn box in a
detector's training mix, rather than only as caption text.

| Your classes | Seen as boxes by a released detector |
|---|---|
| Beaker, syringe, scale, measuring cup | Yes, via Open Images, so Grounding DINO base and MM-Grounding-DINO large |
| Microscope, test tube, funnel, thermometer, stirrer | Only via V3Det, so only the MM-Grounding-DINO `v3det` or `pretrain_all` checkpoints |
| Centrifuge, pipette, petri dish, autoclave, orbital shaker, spectrophotometer, analytical balance, real fume hood | Never, by any released detector |

For the last row, any zero-shot detection comes purely from text generalisation. Measure it, never
assume it. This directly contradicts the current configuration, which uses
`IDEA-Research/grounding-dino-tiny`, the weakest available checkpoint.

The practical recommendation is to run MM-Grounding-DINO alongside SAM 3 rather than instead of it.
Their training mixes genuinely differ, which is what makes cross-model agreement meaningful rather
than two models sharing one blind spot. See item 2 of the gap analysis for why the current
teacher-and-its-own-student agreement does not provide this.

## 4. The homonym trap

Category names do not mean what they appear to mean. These were confirmed by grepping the category
files directly.

| Prompt you would write | What the dataset label actually denotes |
|---|---|
| `fume hood` | LVIS `fume_hood`, category id 565, is a **frequent** class defined as "metal covering leading to a vent that exhausts smoke or fumes", with synonym `exhaust_hood`. It is a kitchen range hood. Objects365 similarly has `Extractor` |
| `shaker` | LVIS `shaker` is "a container in which something can be shaken", meaning a condiment shaker, not an orbital shaker |
| `microscope` | Present in LVIS but **rare**, under 10 images, so effectively untrained. Also present in Objects365 |
| `scale` | Bathroom and kitchen scales dominate, not analytical balances |
| `balance` | Ambiguous between the instrument and the abstract sense |

Independently confirmed absent from all three major box datasets, meaning Open Images V7 boxable,
LVIS v1 and Objects365: **centrifuge, autoclave, spectrophotometer**. Confirmed present: `Beaker` in
Open Images, `Flask` in Objects365, `microscope` in both LVIS and Objects365 but rare in LVIS.

Note that `fume_hood` being a *frequent* LVIS class is the worst case. A detector has been trained on
many confident examples of the wrong object under the right name, so it will fire on kitchen
extractors with high confidence rather than simply failing.

Prompting a detector with a bare equipment name will therefore return domestic objects with high
confidence. Prompt rewriting and explicit hard negatives are mandatory, not polish, and the current
config has no field for either. A fume hood prompt needs kitchen range hood as an explicit negative.

Related correction found during the audit: the ImageNet microscope synset is `n03760671`. The
frequently mis-cited `n03666591` is a lighter.

## 5. Big-vocabulary datasets

| Dataset | Laboratory coverage | Licence |
|---|---|---|
| LVIS v1.0 | `microscope` 3 instances, `fume_hood` is a range hood, six target classes absent | Annotations CC BY 4.0, images under COCO terms |
| Objects365 | `microscope` in v1; flask, scale, extractor in v2 | **No explicit licence found.** It is in nearly every open-vocabulary detector's pretraining mix, which clouds those weights' provenance |
| Open Images V7 | Boxes for beaker 168, syringe 127, scale 139, measuring cup 74. Image-level only for laboratory equipment 323, microscope 234, autoclave 69, pipette 52, centrifuge 26 | Annotations CC BY 4.0, images CC BY 2.0 with no per-image warranty |
| ImageNet-21k Winter21 | 17 laboratory synsets, roughly 14,400 images. Autoclave 1,277, microscope 1,218, incubator 1,188, analytical balance 1,047, beaker 982, spectrophotometer 628, centrifuge 514 | **Non-commercial research and education only** |
| ImageNet-1k | Beaker, lab coat, measuring cup, petri dish and syringe are 1k classes, so every standard backbone already has discriminative features for those five | ImageNet terms |
| Visual Genome | Sparse. Microscope 4, flask 16, beaker 10. Zero centrifuge, pipette, autoclave, test tube | CC BY 4.0 |
| V3Det | Best vocabulary of any large set, but a median of 22 boxes per category and 5,868 of 13,204 categories below 20 boxes. It is a vocabulary, not a training set | Model card states CC BY 4.0, mirror declares none, images web-crawled with no per-image provenance |

ImageNet-21k is the largest laboratory-equipment image resource in existence and the only public
source with meaningful spectrophotometer and centrifuge coverage. It is also non-commercial, so it
can support research and prototyping but not a shipped product.

## 6. Model and weight licences

One of these will bite late, when it is expensive to unpick.

| Model | Licence | Verdict |
|---|---|---|
| Grounding DINO, MM-Grounding-DINO, OWLv2, RF-DETR, Detectron2 | Apache 2.0 | Safe |
| SAM 2 and 2.1, code and weights | Apache 2.0 | Safe |
| GLIP | MIT | Safe |
| SAM 3 and 3.1 | SAM License. Commercial use permitted, royalty-free, derivatives inherit the terms, attribution required in publications, no military, nuclear, weapons or espionage use, no monthly-active-user threshold | Workable, worth a legal read |
| YOLO-World | GPL-3.0 | Avoid |
| Ultralytics YOLOv5, v8, v11, v12 | AGPL-3.0 | Avoid. **This repository currently depends on it** |
| BoxMOT | AGPL-3.0 | Avoid |
| DEVA | CC BY-NC-SA | Avoid for commercial use. Note the same author's Cutie and XMem are MIT |
| CoTracker and CoTracker3 | CC BY-NC | Avoid for commercial use |
| SAM-Track | AGPL plus written commercial permission required | Avoid |

Every YOLO baseline in the published laboratory-equipment papers is copyleft-encumbered. Their
numbers are citable, their weights are not usable. An all-Apache pipeline exists: MM-Grounding-DINO
or OWLv2 for detection, SAM 2.1 for propagation, RF-DETR for the distilled student.

Both SAM 3 repositories are gated behind a contact-sharing agreement with Meta, so access must be
requested manually. That will break any unattended setup script or continuous integration job.
Request access before building anything that depends on it.

## 7. Synthetic rendering is under-exploited

Objaverse-LVIS asset counts, taken from the official annotation file: 47 microscope, 67 cylinder,
63 tripod, 59 glove, 56 goggles, 27 lab coat, 22 fume hood, 19 syringe, 14 thermometer, 12 funnel,
11 scale, 11 measuring cup.

The argument is the ratio. Rendering 47 microscope meshes across a thousand poses, lightings and
backgrounds yields roughly 47,000 synthetic microscope boxes, against the 21 real microscope boxes
that exist across LVIS, Visual Genome and V3Det combined.

Licensing is unusually clean. The collection is ODC-By 1.0 with a per-object `license` field, so
roughly 721,000 CC BY 4.0 and 3,500 CC0 objects can be filtered in and 77,000 non-commercial objects
filtered out.

The limitation is the taxonomy. Objaverse-LVIS inherits LVIS categories, so there is no bucket for
beaker, flask, test tube, petri dish, centrifuge or pipette. Assets for those may exist in the
unannotated remainder, which was not searched.

## 8. Egocentric and procedural datasets

None contain laboratory equipment. Their value is methodological.

| Dataset | Licence | Boxes | Why it matters |
|---|---|---|---|
| EgoObjects | MIT | Yes, LVIS format | 114,000 frames, 9,000 videos, 14,400 unique instances, each appearing in roughly 44.8 images |
| IndustReal | Apache 2.0, code and data | Yes, COCO format | 84 videos of assembly state detection |
| HoloAssist | CDLA-Permissive 2.0 | No | 169 hours of instructor and performer pairs |
| Ego4D and Ego-Exo4D | Signed agreement | Yes, via VQ2D and PACO | Ego-Exo4D's health domain is the only large procedural clinical corpus |
| EPIC-KITCHENS-100 and VISOR, Assembly101 | CC BY-NC | VISOR yes | Method reusable, data not |
| MECCANO | None stated | Yes | Closest annotation shape: active object, next active object, and hand boxes |

**The idea worth stealing is from EgoObjects: train instance-level, not only category-level.** It is
the closest match in task shape, being instance detection from wearable video where the same
physical object recurs across roughly 44.8 images. If your 150 clips come from one or two labs, you
are recognising specific instruments rather than abstract categories, and that is a materially
easier problem than the benchmarks suggest.

A second idea, from EPIC-KITCHENS: narration-derived labels. That project turned 20,000 narrations
into 97 verbs and 300 nouns across 90,000 segments. If your clips carry any spoken commentary or
written protocol text, that is a free second supervision channel.

## 9. Stated gaps

**No public laboratory-equipment video dataset with both boxes and track identities exists.** Nothing
examined has tracking annotations at all. Your clips may genuinely be first of their kind. The
practical consequence is to prefer a detect-then-track design, so that only per-frame boxes are ever
needed as supervision.

Roboflow Universe could not be crawled, because the site returns a 403 behind its content delivery
network and the interface needs a key. Since all three main laboratory datasets were exported from
Roboflow, near-duplicate mirrors very likely exist there. A short sweep with an interface key is the
cheapest open action available.

Four laboratory benchmarks cited in the literature could not be found downloadable: Cheng and
colleagues 2023, Ali and colleagues 2022, Ding and colleagues 2022, and Zou and colleagues 2024.

Unverified at time of writing: the physics set's 27 class names, the chemical apparatus set's 6 class
names, EgoObjects' full 368-category list, Ego4D's licence text, and whether an Objects365 licence
exists at all.
