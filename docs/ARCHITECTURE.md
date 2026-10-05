# Architecture

Why the pipeline has these stages, and what would go wrong without each one.
Every design choice here traces to a finding in the research, and the ones that
look odd are usually the ones that matter.

## The problem in one paragraph

You have about 150 video clips, three minutes each. Each clip carries a list of
the equipment it contains and nothing more: no boxes, no masks, no timestamps.
You need boxes in every frame plus identities over time, for a class list that
keeps growing. Two constraints pull against each other. An open-ended class list
needs zero-shot generalisation, and fine-tuning on a closed set is the classic
way to destroy it.

## The two insights the design rests on

**Do not ask the detector to name things.** Fine-grained instrument names fail
as detection prompts. A surgical-instrument study using the same model family
found the instrument name unusable against a large domain gap and fell back to a
generic prompt plus a classifier on the crops. Five of this project's classes
have never been seen as a human-drawn box by any released detector, so asking
the detector to name them is asking it to guess. **So: localise generically,
name separately.**

**The clip label is a constraint, not a training signal.** A clip labelled with
two instruments guarantees both are present somewhere and is strong evidence
every other prompted class is absent. That turns one label per clip into a
per-detection accept-or-reject rule, and it turns every detection of a
deliberately prompted absent class into free negative data.

## The data flow

```
  manifest.json            one row per clip: id, session, labels, duration, frame count
        |
        v
  [ frames.py ]            plan a uniform sample; RAISE if the frame count is missing
        |                  3-minute clip @ 2fps -> 360 frames
        v
  [ frames.py ]            chunk it. The segmentation task is defined only to 30s,
        |                  so a 3-minute clip needs >= 6 sessions, not 1
        v
  [ prompts.py ]           build this clip's prompt set:
        |                    generic localisation  ("laboratory instrument")
        |                  + the clip's own labels
        |                  + 3-5 confusable classes it is NOT labelled with
        v
  [ federated.py ]         split the vocabulary three ways for this clip:
        |                    positive  = in the clip's labels
        |                    negative  = prompted but absent  -> guaranteed wrong
        |                    unknown   = never prompted        -> must be ignored
        v
  [ backends/*.py ]        detect on keyframes. IMAGE path, not video path:
        |                  the video path derives boxes from masks, so box quality
        |                  is capped by mask quality
        v
  [ naming.py ]            name the generic detections with a crop classifier,
        |                  restricted to this clip's labels + its hard negatives.
        |                  Reject rather than guess when confidence or margin is low.
        v
  [ consistency.py ]       drop boxes no nearby frame corroborates
        |                  (optionally: require two independent detectors to agree)
        v
  [ backends/*.py ]        propagate each seed, FORWARD then REVERSE, per chunk.
        |                  There is no single call covering both directions.
        v
  [ pipeline.py ]          stitch identities across chunk boundaries
        |
        v
  [ mil.py ]               the cascade: clip-label test first and absolute,
        |                  then score, presence, geometry, suppression.
        |                  Every rejection carries a reason code.
        v
  [ export/ ]              annotations for training + a federated file for tracking
        |
        v
  [ train/ ]               distil into a small student; optionally fine-tune
```

## Why each stage exists

**Frame planning raises rather than guessing.** A comparable implementation
returns the first N indices when a container reports no frame count, which
silently samples only the opening moment of every clip and is routine for phone
recordings and remuxes. Refusing is the only safe behaviour.

**Chunking is the common path, not an edge case.** Six sessions per clip means
cross-chunk identity stitching happens constantly, and it is the single largest
accuracy risk in the design. It is also unavoidable: the alternative is
exceeding the backend's documented duration limit.

**Prompt restriction is a cost control.** The per-concept decoder pass does not
amortise; only image encoding does. On a three-minute clip, restricting the
vocabulary to the clip's own labels is worth about 2.5 times, which moves the
unlabelled pool from roughly two working days to one overnight.

**Hard negatives are free training data.** Prompting classes the clip is not
labelled with costs a little inference and yields explicitly labelled negatives,
because every one of those detections is guaranteed wrong.

**The three-way federated split prevents a specific inflation.** Treating a
never-prompted class as absent rather than unknown counts honest ignorance as a
correct rejection, which inflates precision.

**Naming is what makes generic localisation worth anything.** Without it,
generic detections carry the prompt that found them, fail the clip-label test,
and are discarded, so the generic pass contributes exactly zero. That was the
state of this codebase before `naming.py` existed.

**Double propagation is a correction.** One research document named a keyword
that does not exist in the source. The real interface takes a reverse flag and
has no call covering both directions, so the loop runs twice.

**The cascade puts the clip-label test first and makes it absolute.** A
detection of a class the clip is not labelled with cannot be a true positive,
whatever the detector's confidence. Tested up to a confidence of 1.0.

## Where the standard-library boundary sits, and why

```
  vlmlab/                       stdlib only, fully tested, no GPU
    types config registry prompts frames splits geometry
    mil consistency federated naming gold pipeline preflight checkpoint
    eval/   dist metrics stats protocol report tracking
    export/ coco tao rtdetr_yaml
    data/   public imagenet21k rf100vl
    train/  schedule

  vlmlab/backends/  vlmlab/train/     lazy torch imports, NOT executed here
    base fake                         <- these two are stdlib and tested
    sam3 sam2 crops _torch_guards
    lora_adapter rtdetr rtdetr_dataset
```

**The rule that makes it work: every value crossing a backend interface is
plain Python.** A float, int, str, bool, list, tuple or None. Adapters convert
before returning. That is why the cascade, the filters, the statistics and the
exporters are all genuinely tested despite the real backends needing a GPU.

It is enforced, not hoped for. The validator uses exact type identity rather
than an instance check, because a numeric scalar from an array library passes
the latter. A syntax-tree lint rejects any tensor import outside the two adapter
directories, and rejects a module-level import even there, so the fake backend
stays importable with nothing installed.

## Three structural guarantees

Some mistakes are too easy to make, so the design makes them impossible rather
than warning about them.

**Inference mode cannot be forgotten.** `detect` is final in the base class and
wraps the subclass implementation. Omitting it was measured at eight gigabytes
per frame, which is an instant failure on a 12GB card.

**Decoding cannot end up in a loop.** The interfaces accept an extracted frame
path and never a video path. In a comparable pipeline, decoding inside the data
loader was the actual wall clock: it seeks per frame, tests membership against a
list, and converts to an image, per clip and per epoch.

**A mask-only backend is refused.** Detections carry their box provenance, and
the configuration requires detection-head boxes, so a backend that derives boxes
from masks fails loudly instead of silently emitting inflated boxes when a mask
fragments.

## What this architecture cannot fix

**The growing-vocabulary requirement and the accuracy requirement genuinely
conflict.** The detector's own limitations note says it struggles with
fine-grained out-of-domain concepts zero-shot, and on one open-vocabulary split
it scored 76 on base classes against 1.67 on novel ones. A brand-new class added
by text alone lands near that floor. Five to twenty exemplar photos lift it
substantially, but that is a human in the loop, not free extensibility. The
deployed student is closed-set, so in production "add a class" means serve it
from the frozen teacher now and distil it at the next retrain.

**Measurement is floored by your recording sessions, not your clips.** At four
sessions the honest interval on the headline metric is about fifteen points wide
and no amount of extra labelling narrows it. At nine to twenty it is eight to
eleven. That is why `splits.py` groups by session and why `report.py` resamples
sessions rather than frames.
