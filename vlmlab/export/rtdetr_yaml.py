"""Dataset descriptor for the student detector.

Written by string template rather than a serialisation library, because the
core has no third-party dependencies and this file is **write-only**: nothing
in this project ever parses it back. The canonical artefact is the annotation
JSON; this exists purely because the training entry point expects a descriptor.
"""
import os
from typing import Iterable, Optional, Sequence

__all__ = ["render_dataset_yaml", "write_dataset_yaml"]


def render_dataset_yaml(root, labels, train="images/train", val="images/val",
                        test=None):
    """Render the descriptor.

    Class order is the single thing that matters here: it must match the
    one-indexed category order used on export, because the model side is
    zero-indexed and a mismatch shifts every class silently.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("at least one class is required")
    for name in labels:
        if ":" in name or name.strip() != name:
            raise ValueError("class name {!r} would break the descriptor".format(name))

    lines = [
        "# Written by vlmlab. Write-only: nothing here parses it back.",
        "# Class order MUST match the one-indexed export order, because the",
        "# model is zero-indexed and a mismatch shifts every class silently.",
        "path: {}".format(root),
        "train: {}".format(train),
        "val: {}".format(val),
    ]
    if test:
        lines.append("test: {}".format(test))
    lines.append("nc: {}".format(len(labels)))
    lines.append("names:")
    for i, name in enumerate(labels):
        lines.append("  {}: {}".format(i, name))
    return "\n".join(lines) + "\n"


def write_dataset_yaml(path, root, labels, **kw):
    text = render_dataset_yaml(root, labels, **kw)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
