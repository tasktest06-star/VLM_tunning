# Branches

## The three branches here are identical

`main`, `implementation/pipeline` and `research/weak-label-vlm-finetuning` all
point at the same commit and have byte-identical trees. That is deliberate, and
worth stating plainly so nobody goes looking for differences that do not exist.

| Branch | Intended role |
|---|---|
| `main` | The default. Everything, research and code |
| `implementation/pipeline` | Where code changes land, so review is separable from the research |
| `research/weak-label-vlm-finetuning` | Where research corrections land, for the same reason |

They were kept in sync on request. The split exists so that future work can
diverge without reorganising anything: a code change can go to
`implementation/pipeline` and a corrected number to the research branch, and
neither has to wait for the other.

**Per-branch documentation would therefore be the same file three times.** This
document is that explanation rather than three copies of it.

## When to let them diverge

Cut the sync when either of these becomes true:

- **Code review needs to be independent of research review.** Push code only to `implementation/pipeline`, open a pull request against `main`, and leave the research branch alone.
- **A GPU run contradicts a documented number.** Correct it on `research/weak-label-vlm-finetuning` first, because a correction should be reviewable without also reviewing whatever code landed that week.

Per-approach branches are worth cutting when there is genuinely code to
diverge, one per candidate pipeline. There was none when the repository was
created, which is why none exist.

## The sibling repository

`tasktest06-star/customobjectdetection` is a separate repository containing a
different, earlier pipeline written by someone else. This project's research
branch there adds documentation only and changes none of its code.

| Branch there | Whose | Contents |
|---|---|---|
| `research/weak-label-vlm-finetuning` | This project | Research documents plus a gap analysis of that pipeline |
| `feature/auto-label-pipeline` | Not this project | The original pipeline |
| `apache2-rtdetr-pipeline` | Not this project | A licence-motivated rewrite, on an unrelated history |
| `yolo/feature/auto-label-pipeline` | Not this project | Documentation only on top of the original |
| `yolo_limitations_enhancements` | Not this project | Quality improvements to the original |

The gap analysis found twelve defects in that pipeline, three of them critical,
and it is reproduced in this repository as
[01-gap-analysis.md](01-gap-analysis.md) with a header saying it targets a
different codebase. The two most serious are worth knowing even if you never
touch that repository, because they are easy to repeat: its reported metric
measures agreement with the teacher rather than correctness, because the
validation split is itself pseudo-labelled; and its agreement filter keeps all
teacher boxes whenever the student disagrees, so it filters only where filtering
was least needed.
