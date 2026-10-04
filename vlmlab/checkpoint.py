"""Resumable pipeline steps, serialised as JSON rather than pickled.

Reimplemented rather than lifted. The shape of the original is right, but it
pickles arbitrary objects, which would happily persist a tensor into an
artefact and silently defeat the plain-Python boundary the rest of the design
depends on. JSON over the record types forces that boundary to hold.

It also records a configuration hash, so resuming after an edit does not
quietly mix results produced under two different configurations.
"""
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from vlmlab.types import to_jsonable

__all__ = ["Checkpoint", "config_hash"]


def config_hash(cfg_dict):
    """Stable hash of a configuration mapping."""
    blob = json.dumps(to_jsonable(cfg_dict), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Checkpoint(object):
    def __init__(self, root, steps, cfg_hash=None):
        self.root = str(root)
        self.steps = tuple(steps)
        self.cfg_hash = cfg_hash
        self._meta_path = os.path.join(self.root, "meta.json")
        os.makedirs(self.root, exist_ok=True)
        self.meta = self._load_meta()
        if cfg_hash is not None:
            stored = self.meta.get("config_hash")
            if stored is not None and stored != cfg_hash:
                self.meta = {"config_hash": cfg_hash, "done": {},
                             "invalidated_from": stored}
                self._save_meta()
            else:
                self.meta["config_hash"] = cfg_hash
                self._save_meta()

    def _load_meta(self):
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"config_hash": self.cfg_hash, "done": {}}

    def _save_meta(self):
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._meta_path)

    def _payload_path(self, step):
        return os.path.join(self.root, "{}.json".format(step))

    def _check(self, step):
        if step not in self.steps:
            raise KeyError("unknown step {!r}; known steps are {}"
                           .format(step, list(self.steps)))

    def is_done(self, step):
        self._check(step)
        return bool(self.meta.get("done", {}).get(step))

    def save(self, step, payload):
        self._check(step)
        with open(self._payload_path(step), "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(payload), fh, indent=2, sort_keys=True)
        self.meta.setdefault("done", {})[step] = True
        self._save_meta()

    def load(self, step):
        self._check(step)
        path = self._payload_path(step)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def reset(self, step=None):
        if step is None:
            self.meta["done"] = {}
        else:
            self._check(step)
            self.meta.setdefault("done", {}).pop(step, None)
        self._save_meta()

    def status(self):
        return [(s, self.is_done(s)) for s in self.steps]
