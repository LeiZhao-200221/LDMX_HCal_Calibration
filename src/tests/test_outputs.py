"""Tests of common/outputs.py: -only units and groups; where the final results of a stage go (-output by result
name or KEY=FILE, test names, the unsaved scratch copy)."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import outputs  # noqa: E402
from ldmx_hcal.common.outputs import OutputPlan, ResultSlot  # noqa: E402


# minimal RunContext: results/[fullset|test]/<stage>/ and cache/[test/]<stage>/ under a temporary folder
class _Ctx:

    # root folder, mode and -output names
    def __init__(self, root: Path, test: bool = False, names=None):
        self.root, self.test, self.outputs = Path(root), test, names

    # fullset or test
    @property
    def mode(self) -> str:
        return "test" if self.test else "fullset"

    # results/<mode>/<parts>
    def results_dir(self, *parts, mode=None):
        return self.root.joinpath("results", mode or self.mode, *parts)

    # cache/[test/]<parts>
    def cache_dir(self, *parts, mode=None):
        return self.root.joinpath("cache", *(["test"] if (mode or self.mode) == "test" else []), *parts)


# the four results of a ci-like stage
def _slots(ctx):
    d = ctx.results_dir("charge_injection")
    return [ResultSlot("energy_response", "csv", d / "energy_response.csv"),
            ResultSlot("spectra", "pdf", d / "energy_spectra.pdf"),
            ResultSlot("linearity", "pdf", d / "energy_linearity.pdf"),
            ResultSlot("scan", "csv", d / "quality_threshold_scan.csv")]


# output plan of the four results and its log
def _plan(ctx):
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        plan = OutputPlan(ctx, "charge_injection", _slots(ctx))
    return plan, log.getvalue()


# full set without -output: every result to the emptied scratch copy
def test_fullset_without_output_saves_nothing_into_results():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp)
        scratch = ctx.cache_dir("charge_injection", "unsaved")
        scratch.mkdir(parents=True)
        (scratch / "old.pdf").write_text("x")  # the scratch folder is emptied at the start of a run
        plan, _ = _plan(ctx)
        assert not (scratch / "old.pdf").exists()
        for s in plan.slots:
            assert not plan.saved(s.key) and plan.path(s.key).parent == scratch
            assert plan.path(s.key).name == s.default.name
        assert "results" not in str(plan.path("spectra").relative_to(tmp))
        assert scratch == Path(tmp) / "cache" / "charge_injection" / "unsaved"      # no fullset folder in cache/


# -test without -output: test.csv, test.pdf, test1.pdf, test1.csv in stage order
def test_test_mode_numbers_the_results_per_type_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp, test=True)
        plan, _ = _plan(ctx)
        common = ctx.results_dir()
        keys = ("energy_response", "spectra", "linearity", "scan")
        assert [plan.path(k).relative_to(common).as_posix() for k in keys] \
            == ["test.csv", "test.pdf", "test1.pdf", "test1.csv"]
        assert all(plan.saved(s.key) for s in plan.slots)
        # another stage with one pdf overwrites test.pdf (the same name), its own run never overwrites itself
        other = OutputPlan(ctx, "pedestal_mpv", [ResultSlot("mpv_fits", "pdf", ctx.results_dir("x", "mpv.pdf"))])
        assert other.path("mpv_fits") == plan.path("spectra")
        assert other.provenance("mpv_fits") == common / "test.pdf.provenance.json"


# -output names are matched by result: default file name, key + extension, or KEY=FILE; only named results are saved
def test_names_are_matched_by_result_name():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp, names=["quality_threshold_scan.csv"])
        plan, log = _plan(ctx)
        d = ctx.results_dir("charge_injection")
        scratch = ctx.cache_dir("charge_injection", "unsaved")
        assert plan.path("scan") == d / "quality_threshold_scan.csv" and plan.saved("scan")   # not energy_response
        for key in ("energy_response", "spectra", "linearity"):
            assert not plan.saved(key) and plan.path(key).parent == scratch and plan.rule(key) == outputs.UNNAMED_RULE
        assert "WARNING" not in log
        ctx = _Ctx(tmp, names=["spectra.pdf", "energy_response=mine.csv", "linearity=lin.pdf", "x.json"])
        plan, log = _plan(ctx)
        assert plan.path("spectra") == d / "spectra.pdf" and plan.path("energy_response") == d / "mine.csv"
        assert plan.path("linearity") == d / "lin.pdf" and not plan.saved("scan")
        assert "name no result here, ignored: x.json" in log
        record = d / "energy_response.provenance.json"
        assert plan.provenance("energy_response", default_record=record) == d / "mine.csv.provenance.json"
        assert plan.provenance("spectra") == d / "spectra.pdf.provenance.json"
        for names, words in ((["energy_response=a.pdf"], "is a csv file"),
                             (["energy_spectra.pdf", "spectra=b.pdf"], "named twice")):
            try:
                _plan(_Ctx(tmp, names=names))
            except outputs.SettingsError as exc:
                assert words in str(exc), (names, exc)
            else:
                raise AssertionError(f"{names} must be refused")
        same = [ResultSlot("a", "pdf", d / "x" / "same.pdf"), ResultSlot("b", "pdf", d / "y" / "same.pdf")]
        try:
            OutputPlan(_Ctx(tmp, names=["same.pdf"]), "x", same, preview=True)
        except outputs.SettingsError as exc:
            assert "names several results (a, b); use KEY=FILE" in str(exc)
        else:
            raise AssertionError("an ambiguous name must be refused")


# -output alone keeps every default name; -test with names saves only the named results
def test_output_alone_keeps_every_default_name_without_warning():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp, names=[])
        plan, log = _plan(ctx)
        assert all(plan.path(s.key) == s.default and plan.saved(s.key) for s in plan.slots)
        assert "WARNING" not in log
        tctx = _Ctx(tmp, test=True, names=["energy_linearity.pdf"])
        tplan, tlog = _plan(tctx)
        assert tplan.path("linearity") == tctx.results_dir("charge_injection") / "energy_linearity.pdf"
        assert tplan.path("spectra") == tctx.cache_dir("charge_injection", "unsaved") / "energy_spectra.pdf"
        assert not tplan.saved("spectra") and "WARNING" not in tlog


# -output names: .csv / .pdf / .json, no folder, KEY=FILE with a plain key
def test_output_names_are_checked():
    for good in ("a.csv", "B.PDF", "r.json", "energy_response=a.csv", "Muon_4GeV.beam.layers=l.csv"):
        assert outputs.output_name(good) == good
    for bad in ("electron_0.3GeV", "a.png", "dir/a.csv", "-x.csv", "", "=a.csv", "k y=a.csv", "k=a.png"):
        try:
            outputs.output_name(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"{bad!r} accepted")
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp)
        d = ctx.results_dir("x")
        for slots in ([ResultSlot("a", "csv", d / "a.csv"), ResultSlot("a", "pdf", d / "a.pdf")],
                      [ResultSlot("a", "png", d / "a.png")]):
            try:
                OutputPlan(ctx, "x", slots)
            except ValueError:
                continue
            raise AssertionError("bad slots accepted")


# preview plans touch nothing; a name given twice is refused
def test_preview_deletes_nothing_registers_nothing_and_duplicates_are_refused():
    outputs.reset()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _Ctx(tmp)
        scratch = ctx.cache_dir("charge_injection", "unsaved")
        scratch.mkdir(parents=True)
        (scratch / "energy_spectra.pdf").write_text("made by the run")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            plan = OutputPlan(ctx, "charge_injection", _slots(ctx), preview=True)
        assert (scratch / "energy_spectra.pdf").exists() and outputs.plans() == [] and log.getvalue() == ""
        assert plan.path("spectra") == scratch / "energy_spectra.pdf" and not plan.saved("spectra")
        assert plan.rule("spectra").startswith("not saved")
        named = OutputPlan(_Ctx(tmp, names=["energy_spectra.pdf", "a.pdf"]), "charge_injection", _slots(ctx),
                           preview=True)
        assert named.warnings and named.rule("spectra") == "--output name"
        assert named.rule("linearity") == outputs.UNNAMED_RULE
        try:
            OutputPlan(_Ctx(tmp, names=["a.pdf", "a.pdf"]), "charge_injection", _slots(ctx), preview=True)
        except ValueError as exc:
            assert "more than once" in str(exc)
        else:
            raise AssertionError("duplicate name accepted")
    outputs.reset()


# the plan registry and its text
def test_registry_and_order_text():
    outputs.reset()
    with tempfile.TemporaryDirectory() as tmp:
        plan, _ = _plan(_Ctx(tmp, names=[]))
        assert outputs.plans() == [plan]
        assert plan.order_text() == ("energy_response (energy_response.csv), spectra (energy_spectra.pdf), "
                                     "linearity (energy_linearity.pdf), scan (quality_threshold_scan.csv)")
        assert len(plan.summary_lines()) == 4
    outputs.reset()
    assert outputs.plans() == []


# -only: units and groups, comma or space separated, a second -only adds, unknown names stop the parser
def test_only_units_and_groups():
    p = argparse.ArgumentParser(exit_on_error=False)
    outputs.add_only_argument(p, ("pedestal_csv", "mpv_csv", "pedestal_plot", "mpv_plot"),
                              groups={"csv": ("pedestal_csv", "mpv_csv"), "plots": ("pedestal_plot", "mpv_plot")},
                              default=("csv",))
    assert p.parse_args([]).only == ("pedestal_csv", "mpv_csv")
    assert p.parse_args(["-only", "mpv_plot,pedestal_csv"]).only == ("pedestal_csv", "mpv_plot")
    assert p.parse_args(["-only", "plots", "MPV_CSV"]).only == ("mpv_csv", "pedestal_plot", "mpv_plot")
    assert p.parse_args(["-only", "pedestal_csv", "-only", "mpv_plot"]).only == ("pedestal_csv", "mpv_plot")
    assert p.parse_args(["-only", "all"]).only == ("pedestal_csv", "mpv_csv", "pedestal_plot", "mpv_plot")
    assert outputs.selected_units(p.parse_args(["-only", "csv"])) == ("pedestal_csv", "mpv_csv")
    assert outputs.selected_units(argparse.Namespace()) is None
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            p.parse_args(["-only", "nope"])
    except (SystemExit, argparse.ArgumentError):
        pass
    else:
        raise AssertionError("an unknown unit must be refused")
    assert "unknown unit" in err.getvalue() and "groups: all, csv (pedestal_csv, mpv_csv)" in err.getvalue()
    for units, groups in ((("a", "a"), None), (("a",), {"g": ("b",)}), (("a", "all"), None), (("a",), {"a": ("a",)})):
        try:
            outputs.add_only_argument(argparse.ArgumentParser(), units, groups)
        except ValueError:
            continue
        raise AssertionError(f"bad units {units} / groups {groups} accepted")
