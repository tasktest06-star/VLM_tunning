"""Keep the documentation honest.

Documentation rots silently, and this repository has already shipped one guide
printing a two-GPU command whose flags the command line did not accept. These
tests extract claims from the documents and check them against the code, so
that particular failure cannot recur unnoticed.
"""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


#: The numbered research documents describe external work, including a sibling
#: project's pipeline. References to files that do not exist in THIS tree are
#: legitimate there, so existence checks skip them.
RESEARCH_DOCS = tuple("{:02d}".format(i) for i in range(0, 9))


def _is_research_doc(name):
    return name[:2] in RESEARCH_DOCS


def _doc_text(include_research=True):
    out = {}
    for p in sorted(DOCS.glob("*.md")):
        if not include_research and _is_research_doc(p.name):
            continue
        out[p.name] = p.read_text(encoding="utf-8")
    out["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
    return out


class TestDocumentedCliSubcommands(unittest.TestCase):
    def _real_subcommands(self):
        src = (ROOT / "vlmlab" / "cli.py").read_text(encoding="utf-8")
        return set(re.findall(r'sub\.add_parser\(\s*"([a-z-]+)"', src))

    def test_every_documented_subcommand_exists(self):
        real = self._real_subcommands()
        self.assertTrue(real, "could not parse any subcommands from cli.py")
        offenders = []
        for name, text in _doc_text().items():
            for cmd in re.findall(r"vlmlab\.cli\s+([a-z][a-z-]*)", text):
                if cmd not in real:
                    offenders.append("{} documents '{}'".format(name, cmd))
        self.assertEqual([], offenders,
                         "documented subcommands that do not exist: {}. Real: {}"
                         .format(offenders, sorted(real)))

    def test_every_subcommand_is_documented_somewhere(self):
        text = "\n".join(_doc_text().values())
        missing = [c for c in sorted(self._real_subcommands())
                   if "vlmlab.cli {}".format(c) not in text]
        self.assertEqual([], missing,
                         "subcommands nobody documented: {}".format(missing))


class TestDocumentedFlags(unittest.TestCase):
    @staticmethod
    def _flags_of(path):
        src = path.read_text(encoding="utf-8")
        return set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', src))

    def _cli_flags(self):
        return self._flags_of(ROOT / "vlmlab" / "cli.py")

    def test_documented_cli_flags_exist(self):
        """Catches the real failure this repository already shipped once:
        a guide printing a two-GPU command whose flags did not exist."""
        real = self._cli_flags()
        offenders = []
        for name, text in _doc_text(include_research=False).items():
            for line in text.splitlines():
                if "vlmlab.cli" not in line:
                    continue
                for flag in re.findall(r"(--[a-z][a-z-]+)", line):
                    if flag == "--help" or flag in real:
                        continue
                    offenders.append("{}: {}".format(name, flag))
        self.assertEqual([], offenders,
                         "documented flags the command line does not accept: {}"
                         .format(sorted(set(offenders))))

    def test_documented_script_flags_exist(self):
        """Each script is checked against its own argument parser."""
        offenders = []
        for name, text in _doc_text(include_research=False).items():
            for line in text.splitlines():
                m = re.search(r"scripts/([a-z_]+\.py)", line)
                if not m:
                    continue
                script = ROOT / "scripts" / m.group(1)
                if not script.exists():
                    continue
                real = self._flags_of(script) | self._cli_flags()
                for flag in re.findall(r"(--[a-z][a-z-]+)", line):
                    if flag == "--help" or flag in real:
                        continue
                    offenders.append("{}: {} not accepted by {}".format(
                        name, flag, m.group(1)))
        self.assertEqual([], offenders, "\n  ".join(sorted(set(offenders))))


class TestDocumentedModules(unittest.TestCase):
    def test_every_referenced_module_exists(self):
        offenders = []
        for name, text in _doc_text(include_research=False).items():
            for mod in set(re.findall(r"`?(vlmlab(?:\.[a-z_0-9]+)+)`?", text)):
                if mod.endswith(".cli"):
                    continue
                parts = mod.split(".")
                candidate = ROOT.joinpath(*parts).with_suffix(".py")
                pkg = ROOT.joinpath(*parts) / "__init__.py"
                if not candidate.exists() and not pkg.exists():
                    offenders.append("{} references {}".format(name, mod))
        self.assertEqual([], offenders, "\n  ".join(offenders))

    def test_every_referenced_script_exists(self):
        offenders = []
        for name, text in _doc_text(include_research=False).items():
            for script in set(re.findall(r"scripts/([a-z_]+\.py)", text)):
                if not (ROOT / "scripts" / script).exists():
                    offenders.append("{} references scripts/{}".format(name, script))
        self.assertEqual([], offenders, "\n  ".join(offenders))


class TestDocumentedPublicNames(unittest.TestCase):
    """Names listed in the module reference must really be exported."""

    def test_module_reference_names_are_real(self):
        path = DOCS / "MODULES.md"
        if not path.exists():
            self.skipTest("MODULES.md not present")
        import importlib
        text = path.read_text(encoding="utf-8")
        # Lines of the form: `modname`: `A` `B` `C`
        pattern = re.compile(r"^`([a-z_][a-z_0-9.]*)`:\s*(.+)$", re.MULTILINE)
        checked = 0
        offenders = []
        for mod_short, names_blob in pattern.findall(text):
            candidates = ["vlmlab." + mod_short, "vlmlab.eval." + mod_short,
                          "vlmlab.export." + mod_short, "vlmlab.data." + mod_short,
                          "vlmlab.train." + mod_short, "vlmlab.backends." + mod_short]
            mod = None
            for cand in candidates:
                try:
                    mod = importlib.import_module(cand)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if mod is None:
                offenders.append("could not import any module for '{}'".format(mod_short))
                continue
            exported = set(getattr(mod, "__all__", ()) or ())
            for name in re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", names_blob):
                checked += 1
                if name not in exported:
                    offenders.append("{}.{} is documented but not exported"
                                     .format(mod.__name__, name))
        self.assertGreater(checked, 100, "the reference parsed too few names")
        self.assertEqual([], offenders, "\n  ".join(offenders))


class TestDocCrossLinks(unittest.TestCase):
    def test_internal_links_resolve(self):
        offenders = []
        for name, text in _doc_text().items():
            base = ROOT if name == "README.md" else DOCS
            for target in re.findall(r"\]\((?!https?://)([^)#]+)\)", text):
                if not (base / target).exists():
                    offenders.append("{} links to missing {}".format(name, target))
        self.assertEqual([], offenders, "\n  ".join(offenders))


class TestTestCountClaim(unittest.TestCase):
    def test_the_advertised_test_count_is_not_wildly_wrong(self):
        """Catches a stale count after tests are added or removed."""
        text = "\n".join(_doc_text().values())
        claims = [int(n) for n in re.findall(r"(\d{3,4}) tests", text)]
        if not claims:
            self.skipTest("no test count advertised")
        actual = 0
        for p in sorted((ROOT / "tests").glob("test_*.py")):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    actual += 1
        for claim in set(claims):
            self.assertLess(abs(claim - actual) / float(actual), 0.25,
                            "documentation claims {} tests, roughly {} exist"
                            .format(claim, actual))


if __name__ == "__main__":
    unittest.main()
