"""Output units and final results of a stage: -only chooses the units, -output names choose the saved results.

-only UNIT[,UNIT...]: the minimal output units (one table or one pdf each) or group names a stage makes; without it the
stage makes its default units.  -output NAME: a name saves the result whose default file name or result key it
matches (KEY=FILE saves result KEY as FILE); naming one file saves only that file, the others go to the unsaved
scratch copy.  -output alone: every result under its default name.  -test without -output: results/test/test.csv,
test1.csv, test.pdf, ...
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import MODE_TEST
from .io_utils import SettingsError, ensure_writable, log

OUTPUT_KINDS = ("csv", "pdf", "json")
UNSAVED_DIR = "unsaved"
TEST_STEM = "test"
ALL_UNITS = "all"  # -only group of every unit of a stage
UNSAVED_RULE = "not saved (no --output): scratch copy"
UNNAMED_RULE = "not saved (not named in --output): scratch copy"
_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_PLANS: list["OutputPlan"] = []


# --------------------------------------------------------------------------
# -only: output units
# --------------------------------------------------------------------------
# argparse action of -only: comma / space separated units or group names, checked and expanded to units
class _OnlyAction(argparse.Action):

    # remember the units and groups of the stage
    def __init__(self, option_strings, dest, units=(), groups=None, **kwargs):
        self.units = tuple(units)
        self.groups = {ALL_UNITS: self.units, **(groups or {})}
        super().__init__(option_strings, dest, **kwargs)

    # add the named units (stage order) to the units chosen so far
    def __call__(self, parser, namespace, values, option_string=None):
        tokens = [t.strip() for v in (values or []) for t in str(v).split(",") if t.strip()]
        if not tokens:
            parser.error(f"-only: give units or groups ({units_text(self.units, self.groups)})")
        chosen = set()
        for token in tokens:
            name = token.lower()
            if name in self.units:
                chosen.add(name)
            elif name in self.groups:
                chosen.update(self.groups[name])
            else:
                parser.error(f"-only {token!r}: unknown unit; {units_text(self.units, self.groups)}")
        current = getattr(namespace, self.dest, None)
        if current is not None and current is not self.default:  # a second -only adds its units
            chosen.update(current)
        setattr(namespace, self.dest, tuple(u for u in self.units if u in chosen))


# 'units: a, b, c; groups: g (a, b), all' for help texts and errors
def units_text(units, groups=None) -> str:
    groups = {ALL_UNITS: tuple(units), **(groups or {})}
    return (f"units: {', '.join(units)}; groups: "
            + ", ".join(f"{g} ({', '.join(m)})" if g != ALL_UNITS else g for g, m in groups.items()))


# add -only UNIT[,UNIT...] to a stage parser: its units (stage order), group names and default units
def add_only_argument(p, units, groups: dict | None = None, default=None) -> None:
    """Without -only args.only = the default units (None: every unit); RunContext.wants(unit) asks for one."""
    units = tuple(str(u).lower() for u in units)
    groups = {str(g).lower(): tuple(str(u).lower() for u in m) for g, m in (groups or {}).items()}
    bad = sorted({u for m in groups.values() for u in m if u not in units} | ({ALL_UNITS} & set(units))
                 | (set(groups) & set(units)))
    if bad or len(set(units)) != len(units):
        raise ValueError(f"add_only_argument: bad units / groups {bad or units}")
    default_units = units if default is None else tuple(
        u for u in units if u in {x for d in default for x in groups.get(str(d).lower(), (str(d).lower(),))})
    p.add_argument("--only", "-only", dest="only", nargs="+", action=_OnlyAction, units=units, groups=groups,
                   default=default_units, metavar="UNIT",
                   help=f"make only these output units (one table or one pdf each; comma or space separated, "
                        f"groups allowed); {units_text(units, groups)}; default: {', '.join(default_units)}")


# the units a parsed command makes (None: the stage has no -only)
def selected_units(args) -> tuple | None:
    only = getattr(args, "only", None)
    return None if only is None else tuple(only)


# --------------------------------------------------------------------------
# -output: final results
# --------------------------------------------------------------------------
# one final result of a stage: key (its unit), kind (csv / pdf / json), default path, description
@dataclass(frozen=True)
class ResultSlot:
    key: str
    kind: str
    default: Path
    what: str = ""


# argparse type of one -output name: FILE or KEY=FILE, FILE ending in .csv / .pdf / .json, no folder
def output_name(text: str) -> str:
    raw = str(text).strip()
    key, _, name = raw.rpartition("=")
    kind = Path(name).suffix.lower().lstrip(".")
    if (not name or kind not in OUTPUT_KINDS or Path(name).name != name or raw.startswith("-")
            or (key and not _KEY.match(key)) or raw.startswith("=")):
        raise argparse.ArgumentTypeError(
            f"--output {text!r}: give result file names ending in .csv, .pdf or .json (no folder: every result goes "
            f"to the stage's own results folder), e.g. --output energy_response.csv, or KEY=FILE to save the result "
            f"KEY under another name; a dataset or option written right after --output is read as a name, so put "
            f"the names last or before another option")
    return raw


# add -output [NAME ...] to a stage parser
def add_output_argument(p) -> None:
    p.add_argument("--output", "-output", dest="outputs", nargs="*", action="extend", type=output_name,
                   default=None, metavar="NAME",
                   help="final results (results/fullset/<stage>/, -test: results/test/<stage>/): a name saves the "
                        "result whose default file name (or result key + extension) it is; KEY=FILE saves result KEY "
                        "as FILE; naming one file saves only that file; --output alone = every result with its "
                        "default name. Without --output a full-set run saves no final result (the code still runs; "
                        "intermediates are written as always) and a -test run writes results/test/test.csv, "
                        "test1.csv, test.pdf, ...")


# provenance path of a result under a user or test name: <name>.provenance.json
def named_provenance(path: Path) -> Path:
    return path.with_name(path.name + ".provenance.json")


# destinations of the final results of one stage run
class OutputPlan:
    """preview=True (-dry-run, check, run log): same destinations, nothing emptied or logged."""

    # build the plan (names checked, warnings logged)
    def __init__(self, ctx, stage_dir: str, slots, names=None, preview: bool = False):
        self.ctx, self.stage_dir, self.preview = ctx, stage_dir, bool(preview)
        self.slots = list(slots)
        self.names = getattr(ctx, "outputs", None) if names is None else names
        keys = [s.key for s in self.slots]
        if len(set(keys)) != len(keys):
            raise ValueError(f"OutputPlan({stage_dir}): duplicate result keys {keys}")
        bad = [s for s in self.slots if s.kind not in OUTPUT_KINDS]
        if bad:
            raise ValueError(f"OutputPlan({stage_dir}): unknown result kinds {[(s.key, s.kind) for s in bad]}")
        twice = sorted({n for n in (self.names or []) if list(self.names).count(n) > 1})
        if twice:  # two results would be written to one file
            raise ValueError(f"--output: the name(s) {', '.join(twice)} are given more than once")
        self.warnings: list[str] = []
        self._dest: dict[str, Path] = {}
        self._saved: dict[str, bool] = {}
        self._rule: dict[str, str] = {}
        self._assign()
        if not self.preview:
            for w in self.warnings:
                log(f"WARNING: {w}")
            _PLANS.append(self)

    # destination of every result: scratch copy, test names, -output names by result name or default names
    def _assign(self) -> None:
        test = bool(getattr(self.ctx, "test", False))
        if self.names is None and not test:
            scratch = self._scratch()
            if scratch.exists() and not self.preview:
                shutil.rmtree(ensure_writable(scratch), ignore_errors=True)
            for s in self.slots:
                self._set(s, scratch / Path(s.default).name, False, UNSAVED_RULE)
            return
        if self.names is None:  # test, no --output: test.csv, test1.csv, ... in results/test/
            counters: dict[str, int] = {}
            for s in self.slots:
                self._set(s, self._test_path(s.kind, counters), True, "test name")
            return
        if not self.names:  # --output alone: every result with its default name
            for s in self.slots:
                self._set(s, Path(s.default), True, "default name (--output)")
            return
        named, unmatched = self._match_names()
        stage_results = self.ctx.results_dir(self.stage_dir)
        for s in self.slots:
            if s.key in named:
                self._set(s, stage_results / named[s.key], True, "--output name")
            else:
                self._set(s, self._scratch() / Path(s.default).name, False, UNNAMED_RULE)
        if unmatched:  # a stage may make several plans: a name can belong to another plan of the same run
            self.warnings.append(f"{self.stage_dir}: --output names that name no result here, ignored: "
                                 f"{', '.join(unmatched)} (results: {self.order_text()})")

    # {result key: file name} of the -output names, and the names that match no result
    def _match_names(self) -> tuple[dict[str, str], list[str]]:
        named: dict[str, str] = {}
        unmatched = []
        for text in self.names:
            key, _, name = str(text).rpartition("=")
            kind = Path(name).suffix.lower().lstrip(".")
            if key:
                hits = [s for s in self.slots if s.key == key]
                if hits and hits[0].kind != kind:
                    raise SettingsError(f"--output {text}: the result {key} is a {hits[0].kind} file, not .{kind}")
            else:
                hits = [s for s in self.slots if Path(s.default).name == name] or [
                    s for s in self.slots if s.key == Path(name).stem and s.kind == kind]
            if not hits:
                unmatched.append(str(text))
                continue
            if len(hits) > 1:
                raise SettingsError(f"--output {text}: names several results ({', '.join(s.key for s in hits)}); "
                                    f"use KEY=FILE")
            if hits[0].key in named:
                raise SettingsError(f"--output: the result {hits[0].key} is named twice "
                                    f"({named[hits[0].key]} and {name})")
            named[hits[0].key] = name
        return named, unmatched

    # scratch folder of the unsaved results: cache/[test/]<stage>/unsaved/
    def _scratch(self) -> Path:
        return self.ctx.cache_dir(self.stage_dir, UNSAVED_DIR)

    # next test name of a kind: test.csv, test1.csv, ...
    def _test_path(self, kind: str, counters: dict) -> Path:
        i = counters.get(kind, 0)
        counters[kind] = i + 1
        return self.ctx.results_dir(mode=MODE_TEST) / f"{TEST_STEM}{'' if i == 0 else i}.{kind}"

    # record the destination of one result
    def _set(self, slot: ResultSlot, dest: Path, saved: bool, rule: str) -> None:
        self._dest[slot.key], self._saved[slot.key], self._rule[slot.key] = Path(dest), saved, rule

    # where to write a result
    def path(self, key: str) -> Path:
        return self._dest[key]

    # is a result saved? (False: the unsaved scratch copy)
    def saved(self, key: str) -> bool:
        return self._saved[key]

    # how the destination of a result was chosen
    def rule(self, key: str) -> str:
        return self._rule[key]

    # provenance path of a result: the stage's record under a default name, else <name>.provenance.json
    def provenance(self, key: str, default_record: Path | None = None) -> Path:
        dest = self._dest[key]
        slot = next(s for s in self.slots if s.key == key)
        if default_record is not None and dest.name == Path(slot.default).name:
            return dest.with_name(Path(default_record).name)
        return named_provenance(dest)

    # the results of this plan: 'energy_response (energy_response.csv), spectra (energy_spectra.pdf), ...'
    def order_text(self) -> str:
        return ", ".join(f"{s.key} ({Path(s.default).name})" for s in self.slots) or "none"

    # one line per result for the console and the run log
    def summary_lines(self) -> list[str]:
        return [f"{s.kind} {s.key}: {self._dest[s.key]} [{self._rule[s.key]}]" for s in self.slots]


# the output plans made since the last reset (the run log reads them)
def plans() -> list[OutputPlan]:
    return list(_PLANS)


# forget the plans of the previous stage
def reset() -> None:
    _PLANS.clear()
