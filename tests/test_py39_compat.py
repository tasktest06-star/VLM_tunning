"""Mechanical enforcement of the two project-wide invariants.

This test is deliberately the first file in the build order, so that every
module written after it is policed. It enforces:

1.  Python 3.9 compatibility.  The authoring container runs 3.9.25 and the
    whole point of the stdlib core is that it is testable there.  PEP 604
    unions (``int | str``) are *syntactically* valid on 3.9 but raise
    ``TypeError`` at runtime in any position that is actually evaluated:
    function signatures (at ``def`` time), module and class bodies, and
    dataclass fields.  They are silently discarded inside function bodies,
    which makes "it imported fine" worthless as evidence.  So we reject the
    form everywhere in an annotation, including the latent positions, because
    a later ``from __future__ import annotations`` or a ``get_type_hints``
    call would make them live.

2.  The stdlib / GPU boundary.  ``torch``, ``numpy`` and ``yaml`` may be
    imported only inside ``vlmlab/backends/`` and ``vlmlab/train/``, and only
    inside a function body there, never at module level.  That keeps
    ``backends/base.py`` and the fake backend importable with no dependencies,
    which is what lets the pipeline run end to end in this container.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_DIRS = ("backends", "train")
GPU_MODULES = {"torch", "numpy", "yaml", "transformers", "peft", "torchvision", "PIL", "cv2"}


def _python_files():
    for base in ("vlmlab", "tests", "scripts"):
        d = ROOT / base
        if d.is_dir():
            for p in sorted(d.rglob("*.py")):
                yield p


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class _AnnotationUnionVisitor(ast.NodeVisitor):
    """Collects every BinOp using ``|`` that sits in an annotation position."""

    def __init__(self):
        self.hits = []

    def _scan(self, node):
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                self.hits.append(getattr(sub, "lineno", -1))

    def visit_AnnAssign(self, node):
        self._scan(node.annotation)
        self.generic_visit(node)

    def _visit_func(self, node):
        a = node.args
        for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
            self._scan(arg.annotation)
        for arg in (a.vararg, a.kwarg):
            if arg is not None:
                self._scan(arg.annotation)
        self._scan(node.returns)
        self.generic_visit(node)

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func


class TestPython39Compatible(unittest.TestCase):
    def test_no_pep604_union_in_annotations(self):
        offenders = []
        for path in _python_files():
            v = _AnnotationUnionVisitor()
            v.visit(_parse(path))
            for line in v.hits:
                offenders.append("{}:{}".format(path.relative_to(ROOT), line))
        self.assertEqual(
            [], offenders,
            "PEP 604 unions raise TypeError on Python 3.9 in evaluated annotation "
            "positions. Use typing.Optional / typing.Union instead.\n  "
            + "\n  ".join(offenders),
        )

    def test_no_match_statements(self):
        offenders = []
        for path in _python_files():
            src = path.read_text(encoding="utf-8")
            if "match " not in src:
                continue
            for node in ast.walk(_parse(path)):
                if type(node).__name__ == "Match":
                    offenders.append(
                        "{}:{}".format(path.relative_to(ROOT), node.lineno))
        self.assertEqual([], offenders,
                         "match statements require Python 3.10:\n  " + "\n  ".join(offenders))

    def test_no_walrus_in_comprehension_defaults(self):
        # Walrus is 3.8+, so allowed. This test exists to document that we
        # checked, and to fail loudly if someone lowers the floor below 3.8.
        import sys
        self.assertGreaterEqual(sys.version_info[:2], (3, 8))


class TestDependencyBoundary(unittest.TestCase):
    def _imports(self, tree):
        """Yield (module_root, lineno, at_module_level)."""
        module_level_nodes = set(id(n) for n in tree.body)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name.split(".")[0], node.lineno, id(node) in module_level_nodes
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    yield node.module.split(".")[0], node.lineno, id(node) in module_level_nodes

    def test_core_has_no_gpu_imports(self):
        offenders = []
        for path in _python_files():
            rel = path.relative_to(ROOT)
            if rel.parts[0] != "vlmlab":
                continue
            in_gpu_dir = len(rel.parts) > 2 and rel.parts[1] in GPU_DIRS
            for mod, line, _ in self._imports(_parse(path)):
                if mod in GPU_MODULES and not in_gpu_dir:
                    offenders.append("{}:{} imports {}".format(rel, line, mod))
        self.assertEqual(
            [], offenders,
            "The stdlib core must not import a GPU dependency. Move this into "
            "vlmlab/backends/ or vlmlab/train/.\n  " + "\n  ".join(offenders),
        )

    def test_gpu_imports_are_lazy(self):
        """Even inside the adapter dirs, the import must be inside a function."""
        offenders = []
        for path in _python_files():
            rel = path.relative_to(ROOT)
            if rel.parts[0] != "vlmlab":
                continue
            if not (len(rel.parts) > 2 and rel.parts[1] in GPU_DIRS):
                continue
            for mod, line, at_module_level in self._imports(_parse(path)):
                if mod in GPU_MODULES and at_module_level:
                    offenders.append("{}:{} module-level import of {}".format(rel, line, mod))
        self.assertEqual(
            [], offenders,
            "GPU imports must be inside a function body so the module stays "
            "importable without torch installed.\n  " + "\n  ".join(offenders),
        )

    def test_types_module_has_no_intra_package_imports(self):
        path = ROOT / "vlmlab" / "types.py"
        if not path.exists():
            self.skipTest("types.py not written yet")
        offenders = []
        for mod, line, _ in self._imports(_parse(path)):
            if mod == "vlmlab":
                offenders.append("line {}".format(line))
        self.assertEqual([], offenders,
                         "types.py must have zero intra-package imports so every "
                         "other module can depend on it without a cycle.")

    def test_every_core_module_imports_cleanly(self):
        import importlib
        failures = []
        for path in sorted((ROOT / "vlmlab").rglob("*.py")):
            rel = path.relative_to(ROOT)
            if len(rel.parts) > 2 and rel.parts[1] in GPU_DIRS:
                if path.name not in ("__init__.py", "base.py", "fake.py", "schedule.py"):
                    continue
            name = ".".join(rel.with_suffix("").parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001
                failures.append("{}: {}: {}".format(name, type(exc).__name__, exc))
        self.assertEqual([], failures, "\n  ".join(failures))


if __name__ == "__main__":
    unittest.main()
