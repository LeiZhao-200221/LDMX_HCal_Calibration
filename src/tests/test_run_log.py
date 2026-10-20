"""Tests of the run logs (ldmx_hcal.common.run_log): the template sections, NA vs active features, the bug
location of a synthetic exception, SystemExit as a user error, a failing logger never breaks a run, protected
folders, the runner command log (all), failures before a stage starts (guarded) and the switches; the final
results of the --output rule (common/outputs.py, Lei 2026-10-05): [2] 'final results (--output)', [6] every
final result with its destination and saved / not saved, [4] / [5] read where THIS run's plan put the result
tables, [7] the WARNINGs of names that match no result, the dry-run plan display; -only units and -no-json.
Stub stages and temporary folders only, a few seconds.

Author: Lei
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import io_utils, outputs, run_log  # noqa: E402
from ldmx_hcal.common.config import RunContext  # noqa: E402

CLEAN_ENV = {run_log.ENV_SWITCH: "", run_log.ENV_LOGS_DIR: ""}


# RunContext with every tree under tmp
def _ctx(tmp: Path, test: bool = False, protected=(), outputs_=None) -> RunContext:
    cfg = {"paths": {"generated_data_dir": tmp / "data", "calibration_dir": tmp / "cal", "results_dir": tmp / "res",
                     "cache_dir": tmp / "cache"},
           "run": {"chunksize": 3}, "safety": {"protected_dirs": list(protected)}}
    return RunContext(cfg=cfg, test=test, outputs=outputs_)


# runner arguments without run flags
def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(run_names=[], datasets=None, from_saved=False, **kw)


# A stage module for the logger: io_plan lists nothing, run(ctx, args, datasets) is ``run``.
def _stub(name: str, run, **attrs):
    mod = types.ModuleType(f"_runlog_stub_{name}")
    mod.add_arguments = lambda p: None
    mod.io_plan = lambda ctx, args, datasets: []
    mod.run = run
    mod.__dict__.update(attrs)
    sys.modules[mod.__name__] = mod
    return mod


# Logs switched on for the block (as on the command line), environment switches cleared.
class _Enabled:

    # switch the logs on, clear the environment switches
    def __enter__(self):
        self.env = patch.dict(os.environ, CLEAN_ENV)
        self.env.start()
        for k in CLEAN_ENV:
            os.environ.pop(k, None)
        run_log.set_enabled(True)
        return self

    # switch the logs off and restore the environment
    def __exit__(self, *exc):
        run_log.set_enabled(False)
        run_log._STATE["argv"] = None
        self.env.stop()
        return False


# Run ``body(ctx)`` as stage ``name`` inside its log, like runner.run_stage; returns (result, stdout, stderr).
def _run_stub(name: str, ctx, body, args=None, direct: bool = True):
    mod = _stub(name, lambda c, a, d: body(c))
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(run_log.STAGE_MODULES, {name: mod.__name__}), redirect_stdout(out), redirect_stderr(err):
        with run_log.stage_log(name, args or _args(), ctx, direct):
            result = mod.run(ctx, args, [])
    return result, out.getvalue(), err.getvalue()


# log files of a stage folder
def _logs(root: Path, folder: str) -> list[Path]:
    return sorted((root / folder).glob("*.log"))


# require fn to raise exc_type; returns the exception
def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} not raised")


# ----------------------------------------------------------------------------------------------
# template, console transcript, index
# ----------------------------------------------------------------------------------------------
# template sections console and index
def test_template_sections_console_and_index():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp)

        # stage body that prints, warns and writes one table
        def body(c):
            print("hello from the stage")
            print("WARNING: something looks odd")
            out = io_utils.ensure_writable(c.results_dir("fake") / "table.csv")
            out.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            return 42

        result, out, _ = _run_stub("fake", ctx, body)
        assert result == 42 and "hello from the stage" in out          # the console still shows everything
        logs = _logs(tmp / "logs", "fake")
        assert len(logs) == 1, logs
        text = logs[0].read_text(encoding="utf-8")
        positions = [text.index(s) for s in run_log.SECTIONS]
        assert positions == sorted(positions), "sections out of order"
        assert "exit status         : OK" in text
        assert "table.csv  (" in text and "2 rows" in text            # [6] the output with its rows
        assert "WARNING: something looks odd" in text.split("[7] Warnings")[1]
        assert text.split("[8] Bugs / errors")[1].strip().startswith("none")
        console = logs[0].with_name(logs[0].stem + ".console.txt").read_text(encoding="utf-8")
        assert "hello from the stage" in console
        index = (tmp / "logs" / "fake" / "index.tsv").read_text(encoding="utf-8").splitlines()
        assert index[0].split("\t") == list(run_log.INDEX_COLUMNS)
        row = dict(zip(run_log.INDEX_COLUMNS, index[1].split("\t")))
        assert row["status"] == "ok" and row["stage"] == "fake" and row["log"] == logs[0].name
        assert sys.stdout is not None and not isinstance(sys.stdout, run_log._Tee)


# test mode in name and mode
def test_test_mode_in_name_and_mode():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        _run_stub("fake", _ctx(tmp, test=True), lambda c: None)
        (log,) = _logs(tmp / "logs", "fake")
        assert "_fake_test" in log.name
        assert "mode                : test (first 5000 events)" in log.read_text(encoding="utf-8")
        line = (tmp / "logs" / "fake" / "index.tsv").read_text(encoding="utf-8").splitlines()[1]
        assert "test (first 5000 events)" in line


# ----------------------------------------------------------------------------------------------
# [2] features: NA vs active, sources, fit-range options
# ----------------------------------------------------------------------------------------------
# features na and sources on a synthetic module
def test_features_na_and_sources_on_a_synthetic_module():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx(tmp)
        ctx.cfg["fakesec"] = {"window": [1, 5]}                         # yml differs from DEFAULTS
        mod = _stub("feat", lambda c, a, d: None, CONFIG_SECTION="fakesec",
                    DEFAULTS={"window": [0, 10], "threshold": 0.5, "switch": False})
        specs = (run_log.F("window", None, "window"),
                 run_log.F("threshold (-threshold)", None, "threshold", ("threshold",)),
                 run_log.F("switch (-switch)", None, "switch", ("switch",)),
                 run_log.F("reasoned", lambda r: run_log.na("because")))
        with patch.dict(run_log.STAGE_MODULES, {"feat": mod.__name__}):
            r = run_log._Run("feat", _args(threshold=0.25, switch=False), ctx, True)
        r.s = {"window": [1, 5], "threshold": 0.25, "switch": False}
        lines = dict(ln.split(": ", 1) for ln in (x.strip() for x in run_log._feature_lines(r, specs)))
        lines = {k.strip(): v for k, v in lines.items()}
        assert lines["window"] == "[1, 5]  [yml]"
        assert lines["threshold (-threshold)"] == "0.25  [command line]"
        assert lines["switch (-switch)"] == "NA"
        assert lines["reasoned"] == "NA (because)"


# fit range options and linearity lines
def test_fit_range_options_and_linearity_lines():
    # _Run of an empty stage with these arguments
    def run(**kw):
        r = run_log._Run("nothing", argparse.Namespace(**kw), None, True)
        return r

    na = "automatic single Gaussian"
    assert run_log.fit_range_feature(run(), "fit", na) == "NA (automatic single Gaussian)"
    assert run_log.fit_range_feature(run(fit_ranges=[(5.0, 10.0)]), "fit", na) == "[5, 10] single Gaussian"
    assert run_log.fit_range_feature(run(fit_ranges=[(5.0, 10.0), (60.0, 110.0)]), "fit", na) == \
        "[5, 10], [60, 110] double Gaussian"
    # --linfit (dest linearity_fits): one straight line per particle; adcpeak default no fit, ci default per scale
    assert run_log.linfit_feature(run(linearity_fits={"electron": [0.0, 1.0], "pion": None})) == \
        "electron [0, 1] GeV; pion every point"
    assert run_log.linfit_feature(run(linearity_fits={})) == "none (points only)"
    assert run_log.linfit_feature(run()) == "NA (no fit: points only)"
    from ldmx_hcal.energy import charge_injection as ci
    viaset = run()
    viaset.mod = ci
    assert run_log.linfit_feature(viaset, ["adc_only", "pureci", "s"]) == (
        "NA (default per scale: adc_only: none (points only), pureci: electron [0, 0.999] GeV, s: electron every point)")
    viaset.s = {"energy_response": {"linearity_fits": {"pion": [0.5, 4.0]}}}
    assert run_log.linfit_feature(viaset, ["s"]) == "pion [0.5, 4] GeV"
    # --resfit (dest resolution_fit_range), also read from the resolved energy_response settings
    assert run_log.resfit_feature(run(resolution_fit_range=[0.3, 4.0]), "x") == \
        "[0.3, 4] GeV (resolution fitted on these points only)"
    viaset = run()
    viaset.s = {"energy_response": {"resolution_fit_range": [0.5, 4.0]}}
    assert run_log.resfit_feature(viaset, "x").startswith("[0.5, 4] GeV")
    assert run_log.resfit_feature(run(), "every electron point") == "NA (every electron point)"
    # --bin (dest fit_bins): N or N1,N2; NA = the figure's bins
    assert run_log.fit_bins_feature(run(fit_ranges=[(1, 2), (3, 4)], fit_bins=[30, 40])) == "30,40"
    assert run_log.fit_bins_feature(run(fit_ranges=[(1, 2)], fit_bins=[30])) == "30"
    assert run_log.fit_bins_feature(run(fit_ranges=[(1, 2)])) == "NA (figure bins)"


# fit results of the response rows
def test_fit_results_of_the_response_rows():
    # [5] lines of the energy-response tables: range_gauss / range_double_gauss / pion_core_gauss spectra with
    # their --fit ranges and errors, one linearity line per particle (or points only), the resolution fit on the
    # electron points, the e/h points of the s scale.
    import pandas as pd

    nan = float("nan")
    ci = pd.DataFrame([
        {"row_type": "spectrum", "dataset": "electron_0.3GeV", "path": "s", "fit_kind": "two_pass", "fit_ok": True,
         "fit_status": "ok", "mu_mip": 7.57, "mu_error_mip": 0.019, "sigma_mip": 3.86, "sigma_error_mip": 0.019,
         "sigma_over_mu": 0.51, "chi2": 179.1, "ndf": 54, "fit_low_mip": 0.27, "fit_high_mip": 15.3, "fit_ranges": ""},
        {"row_type": "spectrum", "dataset": "electron_1GeV", "path": "s", "fit_kind": "range_gauss", "fit_ok": True,
         "fit_status": "ok", "mu_mip": 30.1, "mu_error_mip": 0.03, "sigma_mip": 6.1, "sigma_error_mip": 0.02,
         "sigma_over_mu": 0.2, "chi2": 40.0, "ndf": 20, "fit_low_mip": nan, "fit_high_mip": nan,
         "fit_ranges": "[20, 40]", "fit_bins": "30"},
        {"row_type": "spectrum", "dataset": "pion_4GeV", "path": "s", "fit_kind": "range_double_gauss",
         "fit_ok": True,
         "fit_status": "ok", "mu_mip": nan, "mu_low_mip": 23.0, "mu_low_error_mip": 0.1, "sigma_low_mip": 3.3,
         "sigma_low_error_mip": 0.05, "mu_high_mip": 82.9, "mu_high_error_mip": 0.4, "sigma_high_mip": 27.8,
         "sigma_high_error_mip": 0.3, "chi2": 264.3, "ndf": 82, "fit_ranges": "[5, 40], [60, 110]", "fit_bins": ""},
        {"row_type": "spectrum", "dataset": "pion_1GeV", "path": "pureci", "fit_kind": "pion_core_gauss",
         "fit_ok": False, "fit_status": "fit_failed (too few bins)", "mu_mip": nan, "chi2": nan, "ndf": 0,
         "fit_ranges": ""},
        {"row_type": "linearity", "path": "adc_only", "fit_kind": "points_only", "fit_ok": False,
         "fit_status": "no fit requested", "slope_mip_per_gev": nan},
        {"row_type": "linearity", "path": "pureci", "particle": "electron", "fit_kind": "linear_weighted",
         "fit_ok": True, "fit_status": "ok", "linearity_fit_range": "[0, 1]", "slope_mip_per_gev": 26.0,
         "slope_error_mip_per_gev": 0.1, "intercept_mip": 0.5, "intercept_error_mip": 0.05, "chi2": 3.0, "ndf": 2,
         "fitted_energies": "0.3;0.4;0.5;1"},
        {"row_type": "resolution", "path": "s", "particle": "electron", "fit_kind": "three_term", "fit_ok": True,
         "fit_status": "ok", "resolution_fit_range": "all points", "resolution_a_mev": 136.9,
         "resolution_a_error_mev": 1.1, "resolution_b_sqrt_mev": 4.1, "resolution_b_error_sqrt_mev": 0.09,
         "resolution_c": 0.08, "resolution_c_error": 0.001, "chi2": 52.2, "ndf": 2,
         "fitted_energies": "300;400;500;1000;4000 MeV"},
        {"row_type": "e_over_h", "path": "s", "dataset": "pion_4GeV", "particle": "pion", "e_over_h": 1.25,
         "e_over_h_error": 0.02, "electron_source": "electron run", "eh_filled": True, "eh_note": ""},
        {"row_type": "e_over_h", "path": "s", "dataset": "pion_3GeV", "particle": "pion", "e_over_h": 1.3,
         "e_over_h_error": 0.03, "electron_source": "electron line fit", "eh_filled": False, "eh_note": ""},
    ])
    # the rows name their run by key; the log shows the reference (run_names), a key that names no run as it is
    names = {"electron_0.3GeV": "electron 0.3", "electron_1GeV": "electron 1", "pion_4GeV": "pion 4",
             "pion_3GeV": "pion 3"}
    text = "\n".join(run_log._ci_response_lines(ci, names))
    assert "electron 0.3 [s]: two_pass, fit range [0.27, 15.3] MIP: mu = 7.57 ± 0.019" in text
    assert "electron 1 [s]: range_gauss, fit range --fit [20, 40] MIP, bins 30: mu = 30.1 ± 0.03" in text
    assert ("pion 4 [s]: range_double_gauss, fit range --fit [5, 40], [60, 110] MIP: low peak mu = 23 ± 0.1, "
            "sigma = 3.3 ± 0.05; high peak mu = 82.9 ± 0.4, sigma = 27.8 ± 0.3 [MIP]") in text
    assert "pion_1GeV [pureci]: pion_core_gauss" in text and "FAILED (fit_failed (too few bins))" in text
    assert "linearity [adc_only]: no fit (points only) - no fit requested" in text
    assert "linearity [pureci] electron (linear_weighted, [0, 1]): slope = 26 ± 0.1 MIP/GeV" in text
    assert "resolution [s] (three_term, electron points, all points): a = 136.9 ± 1.1 MeV" in text
    assert "e/h [s] pion 4: e/h = 1.25 ± 0.02 (filled: mu_electron from the electron run)" in text
    assert "e/h [s] pion 3: e/h = 1.3 ± 0.03 (hollow: mu_electron from the electron line fit)" in text
    adc = pd.DataFrame([
        {"record": "spectrum", "dataset": "electron_1GeV", "spectrum_fit": "range", "n_events": 900,
         "mu": 4994.0, "mu_error": 4.1, "sigma": 1094.0, "sigma_error": 3.9, "sigma_over_mu": 0.219,
         "sigma_over_mu_error": 0.001, "fit_low": 3000.0, "fit_high": 7000.0, "chi2": 50.0, "ndf": 30, "status": "ok",
         "fit_ranges": "[3000, 7000]", "bin_rule": "figure bins"},
        {"record": "spectrum", "dataset": "pion_3GeV", "spectrum_fit": "range", "n_events": 800,
         "mu": nan, "mu_low": 5000.0, "mu_low_error": 20.0, "sigma_low": 900.0, "sigma_low_error": 15.0,
         "mu_high": 14000.0, "mu_high_error": 60.0, "sigma_high": 3000.0, "sigma_high_error": 40.0, "chi2": 70.0,
         "ndf": 40, "status": "ok", "fit_ranges": "[4000, 7000], [8000, 20000]"},
        {"record": "linearity", "particle": "electron", "fit_range_gev": "[0.3, 1]", "weighted": True, "ok": True,
         "slope": 4900.0, "slope_error": 10.0, "intercept": 20.0, "intercept_error": 5.0, "chi2": 2.0, "ndf": 2,
         "fitted_energies": "0.3;0.4;0.5;1", "status": "ok"},
        {"record": "resolution", "particle": "electron", "model": "three_term",
         "fit_range_gev": "[0.3, 4]", "ok": True, "a": 7.8, "a_error": 2.2, "b": 0.04, "b_error": 0.09, "c": 0.05,
         "c_error": 0.01, "chi2": 0.015, "ndf": 3, "fitted_energies": "300;400;500;1000;4000 MeV", "status": "ok"},
    ])
    text = "\n".join(run_log._adcpeak_response_lines(adc, " a.u.", names))
    assert "electron 1: range, fit range --fit [3000, 7000] a.u., bins figure bins: mu = 4994 ± 4.1" in text
    assert "sigma/mu = 0.219 ± 0.001" in text
    assert ("pion 3: range double Gaussian, fit range --fit [4000, 7000], [8000, 20000] a.u.: low peak mu = 5000 ± 20"
            in text)
    assert "linearity electron ([0.3, 1], weighted True): slope = 4900 ± 10 a.u./GeV" in text
    assert ("resolution (three_term, electron points, [0.3, 4]): a = 7.8 ± 2.2, b = 0.04 ± 0.09, "
            "c = 0.05 ± 0.01, chi2/ndf") in text
    no_line = "\n".join(run_log._adcpeak_response_lines(adc[adc["record"] != "linearity"], " a.u.", names))
    assert "linearity: no fit (points only; -linfit <particle>[a,b] fits a line)" in no_line


# feature tables read every stage module
def test_feature_tables_read_every_stage_module():
    # Every FEATURES entry of every stage can be read with the default options and with typical flags (a renamed
    # setting key or option shows up here as '? (cannot be read ...)' / 'NOT RESOLVED').
    import unittest

    import runner
    from ldmx_hcal.common.config import load_config, redirect_products

    electron_03 = ["--electron", "0.3"]
    cases = {"make-csv": [["make-csv", *electron_03, "-test"],
                          ["make-csv", *electron_03, "-waveform", "-duplicate-digis", "drop", "-no-json"],
                          ["make-csv", "-input", "x.root", "-output-csv", "x.csv", "-only", "csv,qa", "-output",
                           "x_qa.pdf"],
                          ["make-csv", *electron_03, "-only", "qa"]],
             "calib": [["calib"], ["calib", "-only", "pedestal", "-mpv-nsigma", "4"],
                       ["calib", "-only", "plots", "-from-saved"], ["calib", "-output"],
                       ["calib", "--electron", "4", "-no-json"]],
             "clean": [["clean", *electron_03, "--muon", "4"],
                       ["clean", *electron_03, "-pedestalsigma", "-muon-standard", "-only", "tables"]],
             "adcpeak": [["adcpeak"], ["adcpeak", "-tot-mode", "none", "-only", "response"],
                         ["adcpeak", "-fit-dataset", "electron", "4", "-only", "line"],
                         ["adcpeak", "-linfit", "electron[0,1]", "pion", "-bar-csv"]],
             "ci": [["ci", "--electron", "0.3"], ["ci", "-no-quality-filter", "-scale-particle", "pion", "muon", "-s"],
                    ["ci", "-test", "-only", "response", "-output", "energy_response.csv"],
                    ["ci", "-only", "parameters"], ["ci", "-pureci", "-linfit", "electron[0,0.5]", "-from-saved"]],
             "plots": [["plots"], ["plots", "-energy", "pureci", "-picker", "random", "-only", "dist_bar,hitmap"],
                       ["plots", "-energy", "arbitrary", "-variant", "none"]]}
    want = {None: "NA (not saved: scratch copies in", (): "default names (--output)"}
    parser = runner.build_parser()
    problems, checked = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        cfg = redirect_products(load_config(), tmp)
        for stage, argvs in cases.items():
            if stage in runner.BROKEN_STAGES:
                continue
            for argv in argvs:
                args = parser.parse_args(argv)
                names = getattr(args, "outputs", None)
                ctx = RunContext(cfg=cfg, test="-test" in argv, outputs=names, write_json="-no-json" not in argv)
                r = run_log._Run(stage, args, ctx, True)
                r.resolve()
                lines = run_log.features_section(r)
                for line in lines:
                    if "? (cannot be read" in line or "NOT RESOLVED" in line or "partly resolved" in line:
                        problems.append(f"{' '.join(argv)}: {line.strip()}")
                final = next((ln.split(": ", 1)[1] for ln in lines
                              if ln.strip().startswith("final results (--output)")), None)
                expected = run_log.TEST_NAMES_TEXT if names is None and "-test" in argv else want.get(
                    tuple(names) if names is not None else None, "names: " + ", ".join(names or []))
                if final is None or not final.startswith(expected):
                    problems.append(f"{' '.join(argv)}: final results (--output) = {final!r}, expected {expected!r}")
                units = next((ln.split(": ", 1)[1] for ln in lines if ln.strip().startswith("output units (-only)")))
                if ("-only" in argv) != units.endswith("[command line]"):
                    problems.append(f"{' '.join(argv)}: output units (-only) = {units!r}")
                json_line = next(ln for ln in lines if ln.strip().startswith("json records"))
                if ("-no-json" in argv) != ("NA (-no-json" in json_line):
                    problems.append(f"{' '.join(argv)}: {json_line.strip()}")
                checked += 1
    assert not problems, "\n".join(problems)
    if runner.BROKEN_STAGES:
        raise unittest.SkipTest(f"{checked} cases ok; stage modules not importable now: {sorted(runner.BROKEN_STAGES)}")


# ----------------------------------------------------------------------------------------------
# [8] bugs and user errors
# ----------------------------------------------------------------------------------------------
# a bug: division by zero
def _buggy():
    return 1 / 0  # the BUG LOCATION of test_bug_location_of_a_synthetic_exception


# an error raised inside the json library
def _library_error():
    return json.loads("{")  # raised inside the json library, called from here


# bug location of a synthetic exception
def test_bug_location_of_a_synthetic_exception():
    exc = _raises(ZeroDivisionError, _buggy)
    loc, raised, worker = run_log.bug_location(exc)
    assert loc.startswith("src/tests/test_run_log.py:") and " in _buggy: return 1 / 0" in loc, loc
    assert raised is None and worker is None
    exc = _raises(ValueError, _library_error)
    loc, raised, _ = run_log.bug_location(exc)
    assert " in _library_error: return json.loads" in loc, loc
    assert raised is not None and "json" in raised and not raised.startswith("src/"), raised
    lines = "\n".join(run_log.error_section(exc))
    assert "BUG: JSONDecodeError" in lines and "BUG LOCATION: src/tests/test_run_log.py:" in lines
    assert "RAISED AT:" in lines and "full traceback:" in lines and "Traceback (most recent call last)" in lines


# failing stage logs the bug and reraises the same exception
def test_failing_stage_logs_the_bug_and_reraises_the_same_exception():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        holder = {}

        # stage body that fails with a bug
        def body(c):
            try:
                _buggy()
            except ZeroDivisionError as e:
                holder["exc"] = e
                raise

        exc = _raises(ZeroDivisionError, _run_stub, "fake", _ctx(tmp), body)
        assert exc is holder["exc"]                                   # unchanged: same exit code and traceback
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        assert "exit status         : FAILED (BUG: ZeroDivisionError)" in text
        assert "BUG LOCATION: src/tests/test_run_log.py:" in text and "in _buggy" in text
        assert "\tFAILED\t" in (tmp / "logs" / "fake" / "index.tsv").read_text(encoding="utf-8")


# systemexit is a user error not a bug
def test_systemexit_is_a_user_error_not_a_bug():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)

        # stage body that stops with a user error
        def refuse(c):
            raise SystemExit("ci: --quality-threshold 2 must lie in [0, 1]")

        exc = _raises(SystemExit, _run_stub, "fake", _ctx(tmp), refuse)
        assert exc.code == "ci: --quality-threshold 2 must lie in [0, 1]"
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        errors = text.split("[8] Bugs / errors")[1]
        assert "USER/INPUT ERROR" in errors and "--quality-threshold 2 must lie in [0, 1]" in errors
        assert "BUG LOCATION" not in errors and "raised in: src/tests/test_run_log.py:" in errors
        assert "FAILED (USER/INPUT ERROR, exit code 1)" in text
        # io_utils.SettingsError is a SystemExit too; SystemExit(0) is a success
        _raises(io_utils.SettingsError, _run_stub, "fake2", _ctx(tmp),
                lambda c: (_ for _ in ()).throw(io_utils.SettingsError("bad setting")))
        assert "USER/INPUT ERROR" in _logs(tmp / "logs", "fake2")[0].read_text(encoding="utf-8")
        _raises(SystemExit, _run_stub, "fake3", _ctx(tmp), lambda c: (_ for _ in ()).throw(SystemExit(0)))
        text = _logs(tmp / "logs", "fake3")[0].read_text(encoding="utf-8")
        assert "exit status         : OK" in text and text.split("[8] Bugs / errors")[1].strip().startswith("none")


# ----------------------------------------------------------------------------------------------
# the logger never breaks a run; protected folders; switches
# ----------------------------------------------------------------------------------------------
# a failing logger never breaks the run
def test_a_failing_logger_never_breaks_the_run():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        stdout_before = sys.stdout

        # _render that fails on the final render
        def broken_render(self, exc, final):
            if final:
                raise RuntimeError("logger exploded")
            return "header\n"

        with patch.object(run_log._Session, "_render", broken_render):
            result, out, err = _run_stub("fake", _ctx(tmp), lambda c: "stage result")
        assert result == "stage result"
        warnings_ = [ln for ln in err.splitlines() if ln.startswith("[run-log] WARNING")]
        assert len(warnings_) == 1 and "logger exploded" in warnings_[0], err
        assert sys.stdout is stdout_before
        # a logs root that cannot be created (a file is in the way): no log, one warning, the stage runs
        (tmp / "blocked").write_text("x")
        with patch.dict(os.environ, {run_log.ENV_LOGS_DIR: str(tmp / "blocked")}):
            result, _, err = _run_stub("fake", _ctx(tmp), lambda c: "still runs")
        assert result == "still runs" and err.count("[run-log] WARNING") == 1, err
        # a failing stage with a failing logger: the stage's own exception comes through
        with patch.object(run_log._Session, "_render", broken_render):
            _raises(ZeroDivisionError, _run_stub, "fake", _ctx(tmp), lambda c: _buggy())


# protected logs folder is refused
def test_protected_logs_folder_is_refused():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp, protected=[tmp / "logs"])
        try:
            result, _, err = _run_stub("fake", ctx, lambda c: "ran")
        finally:
            io_utils.set_protected_dirs([])
        assert result == "ran"
        assert not (tmp / "logs").exists(), "a log was written into a protected folder"
        assert "protected" in err and err.count("[run-log] WARNING") == 1, err


# logs only on the command line and switches
def test_logs_only_on_the_command_line_and_switches():
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CLEAN_ENV):
        tmp = Path(tmp)
        for k in CLEAN_ENV:
            os.environ.pop(k, None)
        run_log.set_enabled(False)
        _run_stub("fake", _ctx(tmp), lambda c: None)                # a library / test call: no log
        assert not (tmp / "logs").exists()
        os.environ[run_log.ENV_SWITCH] = "1"                         # switched on explicitly
        _run_stub("fake", _ctx(tmp), lambda c: None)
        assert len(_logs(tmp / "logs", "fake")) == 1
        with _Enabled():
            os.environ[run_log.ENV_SWITCH] = "0"                     # off wins over the command line
            _run_stub("fake", _ctx(tmp), lambda c: None)
        assert len(_logs(tmp / "logs", "fake")) == 1
        os.environ[run_log.ENV_LOGS_DIR] = str(tmp / "elsewhere")    # LDMX_LOGS_DIR moves the root
        os.environ[run_log.ENV_SWITCH] = "1"
        _run_stub("fake", _ctx(tmp), lambda c: None)
        assert len(_logs(tmp / "elsewhere", "fake")) == 1


# logs root follows the config
def test_logs_root_follows_the_config():
    with patch.dict(os.environ, CLEAN_ENV):
        os.environ.pop(run_log.ENV_LOGS_DIR, None)
        assert run_log.logs_root({"paths": {"results_dir": Path("/x/results")}}) == Path("/x/logs")
        other = Path(tempfile.gettempdir()) / "y" / "l"
        assert run_log.logs_root({"paths": {"results_dir": Path("/x/results"), "logs_dir": other}}) == other
        from ldmx_hcal.common.config import REPO_ROOT, load_config
        assert run_log.logs_root(load_config()) == REPO_ROOT / "logs"


# ----------------------------------------------------------------------------------------------
# runner commands: all (one log per stage + the runner log), failures before a stage starts
# ----------------------------------------------------------------------------------------------
# all command log lists every stage
def test_all_command_log_lists_every_stage():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp)
        args = argparse.Namespace(stages=None, with_make_csv=False, run_names=[], datasets=None, from_saved=False)
        good, bad = _stub("first", lambda c, a, d: print("first ok")), _stub("second", lambda c, a, d: _buggy())
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(run_log.STAGE_MODULES, {"first": good.__name__, "second": bad.__name__}), \
                redirect_stdout(out), redirect_stderr(err):
            # all-chain with a good and a failing stage
            def chain():
                with run_log.command_log("all", args, ctx):
                    for name, mod in (("first", good), ("second", bad)):
                        with run_log.stage_log(name, _args(), ctx, False):
                            mod.run(ctx, None, [])
            _raises(ZeroDivisionError, chain)
        assert len(_logs(tmp / "logs", "first")) == 1 and len(_logs(tmp / "logs", "second")) == 1
        (runner_log,) = _logs(tmp / "logs", "runner")
        text = runner_log.read_text(encoding="utf-8")
        stats = text.split("[4] Data statistics")[1].split("[5]")[0]
        assert "first" in stats and "OK" in stats and "second" in stats and "FAILED (BUG: ZeroDivisionError)" in stats
        errors = text.split("[8] Bugs / errors")[1]
        assert "stage second FAILED" in errors and "BUG LOCATION: src/tests/test_run_log.py:" in errors
        assert "inside 'all'" in _logs(tmp / "logs", "first")[0].read_text(encoding="utf-8")
        console = runner_log.with_name(runner_log.stem + ".console.txt").read_text(encoding="utf-8")
        assert "first ok" in console                                  # the whole command's transcript


# guarded logs an argument error before the stage
def test_guarded_logs_an_argument_error_before_the_stage():
    import runner

    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        os.environ[run_log.ENV_LOGS_DIR] = str(tmp)
        err = io.StringIO()
        with patch.object(sys, "argv", ["runner.py", "ci", "-no-such-option"]), redirect_stderr(err):
            exc = _raises(SystemExit, run_log.guarded, runner.main)
        assert exc.code == 2                                          # argparse's exit code, unchanged
        (log,) = _logs(tmp, "charge_injection")
        text = log.read_text(encoding="utf-8")
        assert "src/runner.py ci -no-such-option" in text and "stopped before the stage started" in text
        assert "USER/INPUT ERROR" in text and "unrecognized arguments: -no-such-option" in text
        # --help is not a failure: no log
        with patch.object(sys, "argv", ["runner.py", "--help"]), redirect_stdout(io.StringIO()):
            assert _raises(SystemExit, run_log.guarded, runner.main).code == 0
        assert len(list(tmp.rglob("*.log"))) == 1


# ----------------------------------------------------------------------------------------------
# final results (--output rule, common/outputs.py, Lei 2026-10-05)
# ----------------------------------------------------------------------------------------------
# text of one section of a log
def _section_text(text: str, title: str) -> str:
    return text.split(title, 1)[1].split("\n[", 1)[0]


# two result slots: a table and a figure
def _two_results(c) -> list:
    res = c.results_dir("fake")
    return [outputs.ResultSlot("table", "csv", res / "table.csv", "the result table"),
            outputs.ResultSlot("fig", "pdf", res / "fig.pdf", "a figure of the run")]


# three result slots: table, scan and figure
def _three_results(c) -> list:
    res = c.results_dir("fake")
    return _two_results(c)[:1] + [outputs.ResultSlot("scan", "csv", res / "scan.csv", "a second table")] + \
        _two_results(c)[1:]


# stub stage body: one output plan, every result written at its destination
def _result_stage(slots_of, record_key: str | None = None):
    # Body of a stub stage with final results: ONE output plan (slots ``slots_of(ctx)``), every result written at
    # its plan destination (csv: 2 rows, pdf: 2 pages), the provenance record of ``record_key`` where the plan puts
    # it, and one intermediate (data/[test/]fake/events.csv, written whatever --output says).
    # write every result of the plan, the record and one intermediate
    def body(c):
        plan = outputs.OutputPlan(c, "fake", slots_of(c))
        for s in plan.slots:
            text = {"csv": "a,b\n1,2\n3,4\n", "pdf": "%PDF-1.4\n/Type /Page\n/Type /Page\n"}.get(s.kind, "{}")
            io_utils.ensure_writable(plan.path(s.key)).write_text(text, encoding="utf-8")
        if record_key:
            io_utils.ensure_writable(plan.provenance(record_key)).write_text('{"stage": "fake"}', encoding="utf-8")
        io_utils.ensure_writable(c.data_dir("fake") / "events.csv").write_text("x\n1\n2\n3\n", encoding="utf-8")
        return plan
    return body


# no output fullset logs the unsaved scratch copies
def test_no_output_fullset_logs_the_unsaved_scratch_copies():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp)                                               # full set, no --output
        earlier = outputs.OutputPlan(ctx, "earlier", [outputs.ResultSlot("old", "csv", tmp / "old.csv")])
        assert earlier in outputs.plans()
        _run_stub("fake", ctx, _result_stage(_two_results))
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        scratch = ctx.cache_dir("fake", outputs.UNSAVED_DIR)
        features = _section_text(text, "[2] Features")
        assert run_log._kv("final results (--output)", f"NA (not saved: scratch copies in {scratch})") in features
        out = _section_text(text, "[6] Outputs")
        assert (f"final results: 2 of 2 NOT SAVED (no --output): scratch copies in {scratch}; use --output [names] "
                f"to save them") in out
        assert f"csv table -> {scratch / 'table.csv'}  (NOT SAVED (no --output): scratch copy; " in out
        assert f"pdf fig -> {scratch / 'fig.pdf'}  (NOT SAVED (no --output): scratch copy; " in out
        assert "2 rows)" in out and "2 pages)" in out and "figure: a figure of the run" in out
        assert "what: the result table" in out
        others = out.split("intermediates and other files written in this run:")[1]
        assert "events.csv  (" in others and "table.csv" not in others        # each file listed once
        assert "csv old ->" not in text                                 # the earlier stage's plan was forgotten
        assert earlier not in outputs.plans()
        assert not (tmp / "res").exists()                               # nothing in results/
        assert "--output names" not in _section_text(text, "[7] Warnings")


# output names matched by result name with warnings
def test_output_names_matched_by_result_name_with_warnings():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp, outputs_=["fig.pdf", "extra.pdf"])              # the table is not named; one name too many
        plan, _, _ = _run_stub("fake", ctx, _result_stage(_two_results))
        res = ctx.results_dir("fake")
        scratch = ctx.cache_dir("fake", outputs.UNSAVED_DIR)
        assert plan.path("table") == scratch / "table.csv" and plan.path("fig") == res / "fig.pdf"
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        assert run_log._kv("final results (--output)", "names: fig.pdf, extra.pdf") in text
        out = _section_text(text, "[6] Outputs")
        assert (f"final results: 1 of 2 NOT SAVED (not named in --output): scratch copies in {scratch}; use --output "
                f"[names] to save them") in out
        assert f"csv table -> {scratch / 'table.csv'}  (NOT SAVED (not named in --output): scratch copy; " in out
        assert f"pdf fig -> {res / 'fig.pdf'}  (saved: --output name; " in out
        warnings_ = _section_text(text, "[7] Warnings")
        assert "--output names that name no result here, ignored: extra.pdf" in warnings_
        assert "[output plan]" not in warnings_                         # the console already held them
        # a plan WARNING the console does not hold (made outside the captured stage) is still listed
        r = run_log._Run("fake", _args(), ctx, True)
        r.result_plans = [plan]
        lines = run_log.warnings_section("", "fake", run_log._plan_warnings(r))
        assert any("ignored: extra.pdf" in ln and "[output plan]" in ln for ln in lines)
        # KEY=FILE saves result KEY under another name
        ctx = _ctx(tmp, outputs_=["fig=only.pdf", "table.csv"])
        plan, _, _ = _run_stub("fake3", ctx, _result_stage(_two_results))
        text = _logs(tmp / "logs", "fake3")[0].read_text(encoding="utf-8")
        assert f"pdf fig -> {res / 'only.pdf'}  (saved: --output name; " in text
        assert "final results: 2 saved" in text
        # --output alone: every result under its default name, no warning
        ctx = _ctx(tmp, outputs_=[])
        _run_stub("fake2", ctx, _result_stage(_two_results))
        text = _logs(tmp / "logs", "fake2")[0].read_text(encoding="utf-8")
        assert run_log._kv("final results (--output)", "default names (--output)") in text
        assert f"pdf fig -> {res / 'fig.pdf'}  (saved: default name (--output); " in text
        assert _section_text(text, "[7] Warnings").strip() == "none"


# -no-json: [1] mode and [2] say so; a make-csv csv without its provenance record is still listed in [4].
def test_no_json_run_is_logged():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = RunContext(cfg=_ctx(tmp).cfg, write_json=False)
        try:
            _run_stub("fake", ctx, lambda c: None)
        finally:
            io_utils.set_json_records(True, [], [])
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        assert "mode                : fullset, no-json" in text
        assert "NA (-no-json: no record json in data/, calibration/, results/; the cache records stay)" in text
        from ldmx_hcal.conversion import make_analysis_files as mk
        csv = io_utils.ensure_writable(ctx.analysis_files_dir() / "Muon_4GeV" / "fixed_Muon_4GeV.csv")
        csv.write_bytes(b"pf_event\n1\n2\n")
        r = _fake_run(ctx, [], provenance_path=mk.provenance_path, qa_pdf_path=mk.qa_pdf_path)
        r.written, r.console = [csv], ""
        stats, _ = run_log.report_make_csv(r)
        assert "  Muon_4GeV/fixed_Muon_4GeV.csv: 13 B, 2 rows (no provenance record: -no-json)" in stats, stats


# test mode without output logs test names and their provenance
def test_test_mode_without_output_logs_test_names_and_their_provenance():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp, test=True)
        _run_stub("fake", ctx, _result_stage(_three_results, record_key="table"))
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        assert run_log._kv("final results (--output)", "test names (results/test/test*.csv/pdf)") in text
        test_dir = ctx.results_dir()
        out = _section_text(text, "[6] Outputs")
        assert f"csv table -> {test_dir / 'test.csv'}  (saved: test name; " in out
        assert f"csv scan -> {test_dir / 'test1.csv'}  (saved: test name; " in out     # numbered within one run
        assert f"pdf fig -> {test_dir / 'test.pdf'}  (saved: test name; " in out
        assert f"provenance: {test_dir / 'test.csv.provenance.json'}" in out          # next to the test name
        assert "test.csv.provenance.json" not in out.split("intermediates and other files")[1]
        # the next test run (of any stage) overwrites the same names; its log shows them written again
        _run_stub("fake2", ctx, _result_stage(_two_results))
        out = _section_text(_logs(tmp / "logs", "fake2")[0].read_text(encoding="utf-8"), "[6] Outputs")
        line = next(ln for ln in out.splitlines() if f"csv table -> {test_dir / 'test.csv'}  (saved: test name; " in ln)
        assert line.endswith("2 rows)") and "NOT WRITTEN" not in line, line
        assert "csv scan" not in out                                    # only the results of this run


# a stage that writes no final result and a failed one
def test_a_stage_that_writes_no_final_result_and_a_failed_one():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        _run_stub("fake", _ctx(tmp, outputs_=["a.pdf"]), _result_stage(lambda c: []))   # e.g. -no-plots
        text = _logs(tmp / "logs", "fake")[0].read_text(encoding="utf-8")
        assert "final results: none in this run (its output plan has no result, e.g. -no-plots)" in text
        assert "ignored: a.pdf (results: none)" in _section_text(text, "[7] Warnings")

        # stage body that makes its output plan and stops
        def fails_after_the_plan(c):
            outputs.OutputPlan(c, "fake", _two_results(c))
            raise SystemExit("stopped before writing")

        _raises(SystemExit, _run_stub, "fake2", _ctx(tmp), fails_after_the_plan)
        out = _section_text(_logs(tmp / "logs", "fake2")[0].read_text(encoding="utf-8"), "[6] Outputs")
        scratch = _ctx(tmp).cache_dir("fake", outputs.UNSAVED_DIR)
        assert f"csv table -> {scratch / 'table.csv'}  (NOT SAVED (no --output): scratch copy; NOT WRITTEN in this " \
               f"run (missing))" in out


# dry run and runner plan show the final result destinations
def test_dry_run_and_runner_plan_show_the_final_result_destinations():
    import runner

    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp)
        ctx.dry_run = True
        scratch = ctx.cache_dir("fake", outputs.UNSAVED_DIR)
        res = ctx.results_dir("fake")
        items = [{"role": "out", "path": tmp / "data" / "fake" / "events.csv"},
                 {"role": "out", "path": scratch / "table.csv", "result": "table", "saved": False,
                  "note": "(not saved: no --output)"},                                 # clean / ci style
                 {"role": "out", "path": res / "only.pdf", "result": "fig", "saved": True,
                  "note": f"final result pdf fig: {res / 'only.pdf'} [--output name]"},   # calib / adcpeak style
                 {"role": "out", "path": scratch / "electron_4GeV_dist_event_s.pdf",
                  "result": "electron_4GeV_dist_event_s",
                  "note": "not saved: no --output"}]                                   # plots / make-csv style
        mod = _stub("dry", lambda c, a, d: None, io_plan=lambda c, a, d: list(items))
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(run_log.STAGE_MODULES, {"dry": mod.__name__}), redirect_stdout(out), redirect_stderr(err):
            with run_log.stage_log("dry", _args(), ctx, True):
                runner.print_plan("dry", mod.io_plan(ctx, None, []), ctx)
        printed = out.getvalue()
        assert f"[out] new     {scratch / 'table.csv'}   (final result table; not saved: no --output)" in printed
        assert f"[out] new     {res / 'only.pdf'}   (final result fig [--output name])" in printed
        assert (f"{scratch / 'electron_4GeV_dist_event_s.pdf'}   (final result electron_4GeV_dist_event_s; "
                "not saved: "
                "no --output)") in printed
        assert f"{tmp / 'data' / 'fake' / 'events.csv'}\n" in printed      # an intermediate: no note
        text = _logs(tmp / "logs", "dry")[0].read_text(encoding="utf-8")
        assert "_dryrun" in _logs(tmp / "logs", "dry")[0].name
        planned = _section_text(text, "[6] Outputs")
        assert f"[new] {scratch / 'table.csv'}  (final result table; not saved: no --output)" in planned
        assert f"[new] {res / 'only.pdf'}  (final result fig [--output name])" in planned
        assert not scratch.exists() and not res.exists()
        # the note without a rule in the item: from the --output names of the context
        item = {"role": "out", "path": res / "x.pdf", "result": "x"}
        assert run_log.final_result_note(item, _ctx(tmp, outputs_=["x.pdf"])) == "final result x [--output name]"
        assert run_log.final_result_note(item, _ctx(tmp, outputs_=[])) == "final result x [default name (--output)]"
        assert run_log.final_result_note(item, _ctx(tmp, test=True)) == "final result x [test name]"
        assert run_log.final_result_note({"role": "out", "path": res / "y.csv"}, ctx) == ""


# all log counts the final results of each stage
def test_all_log_counts_the_final_results_of_each_stage():
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = _ctx(tmp)
        args = argparse.Namespace(stages=None, with_make_csv=False, run_names=[], datasets=None, from_saved=False)
        first = _stub("first", lambda c, a, d: _result_stage(_two_results)(c))
        second = _stub("second", lambda c, a, d: print("no output plan"))
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(run_log.STAGE_MODULES, {"first": first.__name__, "second": second.__name__}), \
                redirect_stdout(out), redirect_stderr(err):
            with run_log.command_log("all", args, ctx):
                for name, mod in (("first", first), ("second", second)):
                    with run_log.stage_log(name, _args(), ctx, False):
                        mod.run(ctx, None, [])
                    # real stages are seconds apart; Windows file times are coarser than time.time_ns(), so two
                    # stub stages within the same millisecond could share a timestamp (a flaky file count)
                    time.sleep(0.05)
        (log,) = _logs(tmp / "logs", "runner")
        text = log.read_text(encoding="utf-8")
        scratch = ctx.cache_dir("<stage>", outputs.UNSAVED_DIR)
        assert run_log._kv("final results (--output)", f"NA (not saved: scratch copies in {scratch})") in text
        out6 = _section_text(text, "[6] Outputs")
        assert "first: 3 files, 2 final results NOT SAVED (no --output: scratch copies) (see " in out6
        assert "second: 0 files (see " in out6
        # the second stage's log does not see the first stage's plan (reset when its log starts)
        second_log = _logs(tmp / "logs", "second")[0].read_text(encoding="utf-8")
        assert "csv table" not in second_log and "none (no file was written in this run)" in second_log


# A _Run of a synthetic stage module (``attrs``) whose run made the output ``plans``.
def _fake_run(ctx, plans, **attrs) -> run_log._Run:
    r = run_log._Run("fake", _args(fit_ranges=None), ctx, True)
    r.mod = types.SimpleNamespace(**attrs)
    r.result_plans, r.finals = list(plans), None
    return r


# A file of an earlier run at ``path`` (modified an hour ago).
def _old(path: Path, text: str) -> Path:
    io_utils.ensure_writable(path).write_text(text, encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


# adcpeak report reads the response where this run put it
def test_adcpeak_report_reads_the_response_where_this_run_put_it():
    import pandas as pd

    # provenance json next to a product
    def prov(p):
        return Path(p).with_name(Path(p).stem + ".provenance.json")

    row = {"record": "spectrum", "dataset": "electron_1GeV", "spectrum_fit": "two_pass", "n_events": 900,
           "mu": 4994.0, "mu_error": 4.1, "sigma": 1094.0, "sigma_error": 3.9, "sigma_over_mu": 0.219,
           "fit_low": 3000.0, "fit_high": 7000.0, "chi2": 50.0, "ndf": 30, "status": "ok"}
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        tmp = Path(tmp)
        for test in (False, True):
            ctx = _ctx(tmp, test=test)                                  # no --output: scratch copy / test.csv
            ctx.cfg["datasets"] = {"electron_1GeV": {"particle": "electron", "energy_gev": 1.0}}
            default = ctx.results_dir("tot_adc_peak") / "energy_response.csv"
            _old(default, "record,dataset\nspectrum,OLD_RUN\n")            # an earlier run's table at the default path
            outputs.reset()
            plan = outputs.OutputPlan(ctx, "tot_adc_peak", [outputs.ResultSlot("energy_response", "csv", default)])
            r = _fake_run(ctx, [plan], parameters_csv=lambda c: c.calib_dir("tot_adc_peak") / "params.csv",
                          provenance_path=prov,
                          energy_response_csv=lambda c, sfx: c.results_dir("tot_adc_peak") / "energy_response.csv",
                          result_record=lambda pl, key: pl.provenance(key, default_record=prov(default)))
            r.s = {"tot_mode": "linear", "slope_window": [0.6, 4.0]}
            r.extra = {"steps": {"response": True, "energy_response": True, "saved_events": True}, "suffix": ""}
            stats, fits = run_log.report_adcpeak(r)                     # nothing written yet: say so
            dest = plan.path("energy_response")
            assert f"  energy response not written in this run ({dest}: missing)" in fits
            pd.DataFrame([row]).to_csv(io_utils.ensure_writable(dest), index=False)
            record = plan.provenance("energy_response", default_record=prov(default))
            record.write_text(json.dumps({"events": {"from": "saved copy", "files": [
                {"path": "cache/tot_adc_peak/Electron_1GeV/fixed_electron_1GeV_events.csv",
                 "committed": "2026-10-20"}]}}), encoding="utf-8")
            stats, fits = run_log.report_adcpeak(r)
            text = "\n".join(stats + fits)
            state = "saved: test name" if test else "NOT SAVED (no --output): scratch copy"
            assert f"energy response table of this run: {dest} ({state})" in text
            assert dest.name == ("test.csv" if test else "energy_response.csv")
            assert "electron 1: two_pass, fit range [3000, 7000] a.u.: mu = 4994 ± 4.1" in text
            assert "OLD_RUN" not in text
            assert "the response read the saved copy (fixed_electron_1GeV_events.csv)" in text   # its own record
            assert "linearity: no fit (points only" in text
        outputs.reset()


# ci report reads the response and scan where this run put them
def test_ci_report_reads_the_response_and_scan_where_this_run_put_them():
    import pandas as pd

    from ldmx_hcal.energy import charge_injection as ci

    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        tmp = Path(tmp)
        ctx = _ctx(tmp, outputs_=["resp.csv"])                          # response=resp.csv; the scan is not named
        ctx.cfg["datasets"] = {"electron_1GeV": {"particle": "electron", "energy_gev": 1.0}}
        res = ctx.results_dir("charge_injection")
        _old(res / "resp.csv", "row_type,dataset\nspectrum,OLD_RUN\n")
        outputs.reset()
        plan = outputs.OutputPlan(ctx, "charge_injection", [
            outputs.ResultSlot("response", "csv", res / "energy_response.csv"),
            outputs.ResultSlot("quality_scan", "csv", res / "quality_threshold_scan.csv")])
        scratch = ctx.cache_dir("charge_injection", outputs.UNSAVED_DIR)
        assert plan.path("response") == scratch / "energy_response.csv"   # resp.csv names no result: not saved
        plan = outputs.OutputPlan(ctx, "charge_injection", [
            outputs.ResultSlot("response", "csv", res / "energy_response.csv"),
            outputs.ResultSlot("quality_scan", "csv", res / "quality_threshold_scan.csv")],
            names=["response=resp.csv"])
        r = _fake_run(ctx, [plan], EVENT_UNITS=ci.EVENT_UNITS, needs_ci=lambda s: False,
                      result_file=lambda c, key, t: (_ for _ in ()).throw(AssertionError("plan first")))
        r.s = {"variant_tag": "", "quality_filter": False, "selection_path_used": "s", "bad_bar_rule": "used_ends",
               "compute_paths": ["adc_only"], "from_saved": False, "units": ("response", "quality_scan")}
        r.console = ""
        stats, fits = run_log.report_ci(r)
        assert f"  quality-threshold scan not written in this run ({scratch / 'quality_threshold_scan.csv'}: " \
               f"missing)" in fits
        assert "  no event statistics in this run (no event was reconstructed or read)" in stats  # old resp.csv unread
        assert "  CI map / s: not used (adc_only reconstruction)" in fits
        pd.DataFrame([{"row_type": "spectrum", "dataset": "electron_1GeV", "particle": "electron",
                       "beam_energy_gev": 1.0, "n_csv_files": 2,
                       "n_events_in_file": 1000, "n_events_reconstructed": 990, "n_events_selected": 900,
                       "retention_fraction": 0.9, "n_bars_selected_events": 5000, "path": "adc_only",
                       "fit_kind": "two_pass", "fit_ok": True, "fit_status": "ok", "mu_mip": 30.0,
                       "mu_error_mip": 0.1, "sigma_mip": 6.0, "sigma_error_mip": 0.05, "sigma_over_mu": 0.2,
                       "chi2": 40.0, "ndf": 20, "fit_low_mip": 10.0, "fit_high_mip": 50.0}]).to_csv(res / "resp.csv",
                                                                                                  index=False)
        io_utils.ensure_writable(scratch / "quality_threshold_scan.csv").write_text("threshold,n\n0.1,5\n",
                                                                                    encoding="utf-8")
        stats, fits = run_log.report_ci(r)
        text = "\n".join(stats + fits)
        assert f"energy response table of this run: {res / 'resp.csv'} (saved: --output name)" in text
        assert "  electron 1: 2 csv, 1000 events in file, 990 reconstructed, 900 selected" in text
        assert "electron 1 [adc_only]: two_pass, fit range [10, 50] MIP: mu = 30 ± 0.1" in text
        assert "OLD_RUN" not in text
        assert f"1 rows -> {scratch / 'quality_threshold_scan.csv'}" in text
        assert "quality-threshold scan (check)" in text
        # no response table (default units): the run lines and the scale lines of the console summary
        r.result_plans, r.finals = [], None
        r.s = {**r.s, "units": ("events", "linearity")}
        r.mod.result_file = lambda c, key, t: res / f"never_{key}.csv"
        r.console = ("[  12.0s] ci summary (fitted CI parameters):\n"
                     "[  12.0s]   electron_1GeV: 900/990 events selected (0.9091) in 2 csv, 5,000 bars | s: ...\n"
                     "[  12.0s]   s: linearity electron slope 30.1 +/- 0.1 MIP/GeV (ok), sigma/mu c = 0.08 (ok)\n"
                     "[  12.1s] ci done in 12.1s\n")
        stats, fits = run_log.report_ci(r)
        assert "  electron 1: 900/990 events selected (0.9091) in 2 csv, 5,000 bars" in stats
        assert "  [s] linearity electron slope 30.1 +/- 0.1 MIP/GeV (ok), sigma/mu c = 0.08 (ok)" in fits
        outputs.reset()


# clean report reads the layers where this run put them
def test_clean_report_reads_the_layers_where_this_run_put_them():
    from ldmx_hcal.common.config import Dataset
    from ldmx_hcal.selection import clean_dataset as cd

    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        tmp = Path(tmp)
        ctx = _ctx(tmp, outputs_=["Muon_4GeV.beam.layers=lay.csv"])
        ctx.cfg["datasets"] = {"muon_4GeV": {"particle": "muon", "energy_gev": 4.0}}
        ds = Dataset("muon_4GeV", "muon", 4.0, "Muon 4 GeV")
        beam = io_utils.ensure_writable(ctx.run_dir(ds) / "beam.csv")
        beam.write_text("pf_event\n1\n", encoding="utf-8")
        res = ctx.results_dir("clean_dataset")
        _old(cd.layers_csv(ctx, ds, beam), "layer,n_clean_bars,energy_mev_bar_mean\n1,1,999\n")   # an earlier run
        outputs.reset()
        plan = outputs.OutputPlan(ctx, "clean_dataset", cd.result_slots(ctx, [types.SimpleNamespace(ds=ds)],
                                                                        ("cutflow", "layers")))
        r = _fake_run(ctx, [plan], clean_csv=cd.clean_csv, provenance_json=cd.provenance_json,
                      cache_json=cd.cache_json, result_key=cd.result_key, results_json=cd.results_json,
                      layers_csv=cd.layers_csv)
        r.s, r.extra = {"mip_fit_range": [0.7, 1.3]}, {"jobs": [types.SimpleNamespace(ds=ds)]}
        record = {"counts": {"input_events": 100, "output_events": 80, "input_rows": 1000, "output_rows": 300},
                  "selection": {"standard": "muon", "threshold_column": "t"}}
        io_utils.ensure_writable(cd.cache_json(ctx, ds, beam)).write_text(json.dumps(record), encoding="utf-8")
        stats, _ = run_log.report_clean(r)
        text = "\n".join(stats)
        assert "muon 4 (Muon_4GeV/beam.csv): muon standard, threshold t: 100 events / 1000 rows in -> 80 events kept " \
               "(80.00 %)" in text, text
        assert "[clean csv not written: no clean unit (-only)]" in text
        assert "    per-layer table: not written in this run" in stats and "999" not in text
        layers = "layer,n_clean_bars,energy_mev_bar_mean\n1,4,9.0\n2,3,5.0\n"
        key = cd.result_key(ds, beam, "layers")
        assert key == "Muon_4GeV.beam.layers" and plan.path(key) == res / "lay.csv"
        io_utils.ensure_writable(plan.path(key)).write_text(layers, encoding="utf-8")
        stats, fits = run_log.report_clean(r)
        assert "    layers with clean bars: 2/2; mean bar energy 5 MeV (layer 2) - 9 MeV (layer 1)" in stats
        assert "drawn on the diagnostics pdf only" in fits[0]
        outputs.reset()


# calib report lists the pdfs with their pages and destinations
def test_calib_report_lists_the_pdfs_with_their_pages_and_destinations():
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        tmp = Path(tmp)
        ctx = _ctx(tmp, test=True)
        res = ctx.results_dir("pedestal_mpv")
        outputs.reset()
        plan = outputs.OutputPlan(ctx, "pedestal_mpv", [
            outputs.ResultSlot("pedestal_plot", "pdf", res / "pedestal_fits.pdf"),
            outputs.ResultSlot("mpv_plot", "pdf", res / "mpv_fits.pdf")])
        all_units = {"pedestal_csv": True, "mpv_csv": True, "pedestal_plot": True, "mpv_plot": True}
        attrs = dict(STAGE_DIR="pedestal_mpv", chosen_units=lambda c: dict(all_units),
                     table_csv=lambda c, units: c.calib_dir("pedestal_mpv") / "table.csv",
                     provenance_json=lambda p: Path(p).with_name("table.provenance.json"))
        r = _fake_run(ctx, [plan], **attrs)
        r.s = {}
        io_utils.ensure_writable(plan.path("pedestal_plot")).write_text("/Type /Page /Type /Page /Type /Page",
                                                                         encoding="utf-8")
        stats, fits = run_log.report_calib(r)
        test_dir = ctx.results_dir()
        assert "  figures (final results):" in stats
        assert f"    pedestal_plot: 35 B, 3 pages -> {test_dir / 'test.pdf'} (saved: test name)" in stats
        assert f"    mpv_plot: NOT WRITTEN in this run (missing) -> {test_dir / 'test1.pdf'} (saved: test name)" \
            in stats
        assert any(ln.startswith("  calibration table not written in this run") for ln in stats)
        empty = outputs.OutputPlan(ctx, "pedestal_mpv", [])            # -no-plots: no pdf
        stats, _ = run_log.report_calib(_fake_run(ctx, [empty], **attrs))
        assert "  figures (final results): none in this run" in stats
        # -only plots: no table; the fit summary of the console
        attrs["table_csv"] = lambda c, units: None
        r = _fake_run(ctx, [empty], **attrs)
        r.console = "[  3.0s] summary: pedestal NaN 0, MPV status {'OK': 380}\n"
        stats, fits = run_log.report_calib(r)
        assert "  no table in this run (no csv unit chosen with -only)" in stats
        assert fits == ["  fits drawn only (no table written); pedestal NaN 0, MPV status {'OK': 380}"]
        outputs.reset()


# ----------------------------------------------------------------------------------------------
# the runs a stage was given (Lei 2026-10-06: any run is accepted, the log shows what was given)
# ----------------------------------------------------------------------------------------------
# A resolved _Run of the real stage module on configure/config.yml (products redirected to ``tmp``).
def _real_run(stage: str, argv: list, tmp: Path) -> run_log._Run:
    import runner
    from ldmx_hcal.common.config import load_config, redirect_products

    args = runner.build_parser().parse_args([stage, *argv])
    r = run_log._Run(stage, args, RunContext(cfg=redirect_products(load_config(), tmp)), True)
    r.resolve()
    return r


# run inputs are references noted in warnings when not the default
def test_run_inputs_are_references_noted_in_warnings_when_not_the_default():
    # [2] / [7]: the run-choosing inputs (runs, calib source run, adcpeak fit run) are written as references
    # with their source; one that differs from its default gets an 'input:' note in [7] (information only: the
    # stage ran on it); calib given several runs says that the stage stops on it; a run that does not exist is NOT
    # RESOLVED with the reason.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        r = _real_run("calib", [], tmp)
        (ri,) = run_log.run_inputs(r)
        assert (ri.role, [d.key for d in ri.runs], ri.source) == ("source run", ["muon_4GeV"], "default")
        assert run_log.input_notes(r) == [] and run_log.run_name(r, "muon_4GeV") == "muon 4"
        assert run_log.run_name(r, "muon_9GeV") == "muon_9GeV"                   # a key that names no run: as it is
        r = _real_run("calib", ["--electron", "4"], tmp)
        assert run_log.input_notes(r) == ["input: calib source run = electron 4  [command line] (default muon 4); "
                                          "the stage ran on what was given, check its results"]
        lines = run_log.input_lines(r)
        assert lines[0] == "  input runs of this stage (as given; compare them with the error above):"
        assert lines[1].strip().startswith("source run") and "electron 4 (" in lines[1], lines
        assert lines[1].endswith("[command line] (default muon 4)"), lines
        features = "\n".join(run_log.features_section(r))
        assert "source run (run flag, else source_dataset)" in features and "electron 4 (" in features
        r = _real_run("calib", ["--electron"], tmp)                               # 5 runs: calib stops on them
        (note,) = run_log.input_notes(r)
        assert note.startswith("input: calib source run = electron 0.3, electron 0.4, electron 0.5, electron 1, "
                               "electron 4  [command line] (default muon 4); the stage stops on it: [calib] the run "
                               "flags --electron chose 5 runs"), note
        r = _real_run("adcpeak", ["-fit-dataset", "electron", "4"], tmp)
        notes = run_log.input_notes(r)
        assert "input: adcpeak fit run = electron 4  [command line] (default pion 3); the stage ran on what was " \
               "given, check its results" in notes, notes
        r = _real_run("adcpeak", ["-fit-dataset", "muon", "9"], tmp)
        fit = next(ri for ri in run_log.run_inputs(r) if ri.role == "fit run")
        assert run_log.input_text(fit).startswith("NOT RESOLVED: 'muon 9'  [command line]: -fit-dataset: there is "
                                                  "no run 'muon 9'; runs of configure/datasets.yml"), fit
        assert run_log.input_notes(_real_run("clean", ["--muon", "4"], tmp)) == []  # analysed runs: a normal choice
        assert run_log.input_notes(_real_run("clean", [], tmp)) == []


# a stage log that stops with a user error
def _stop_in_stage_log(name, args, ctx):
    with run_log.stage_log(name, args, ctx, True):
        raise SystemExit("[calib] electron 4: synthetic stop of the stage")


# error section lists the input runs next to the error
def test_error_section_lists_the_input_runs_next_to_the_error():
    # [8]: the input runs of the stage follow the message of a USER/INPUT ERROR and come before the traceback of
    # a BUG, so an input that does not suit the analysis can be traced; a stage log of calib --electron 4 that stops
    # has its input note in [7] and its input runs in [8].
    import runner
    from ldmx_hcal.common.config import load_config, redirect_products

    inputs = ["  input runs of this stage (as given; compare them with the error above):",
              "    source run                  : electron 4 (x.csv)  [command line] (default muon 4)"]
    lines = run_log.error_section(SystemExit("[calib] electron 4: no MIP peak"), "", inputs)
    assert lines[0].startswith("  USER/INPUT ERROR") and lines[-2:] == inputs, lines
    lines = run_log.error_section(_raises(ZeroDivisionError, _buggy), "", inputs)
    i = lines.index("  full traceback:")
    assert lines[0].startswith("  BUG: ZeroDivisionError") and lines[i - 2:i] == inputs, lines
    assert run_log.error_section(None, "", inputs) == ["  none"]
    with tempfile.TemporaryDirectory() as tmp, _Enabled():
        tmp = Path(tmp)
        ctx = RunContext(cfg=redirect_products(load_config(), tmp))
        args = runner.build_parser().parse_args(["calib", "--electron", "4"])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _raises(SystemExit, _stop_in_stage_log, "calib", args, ctx)
        (log,) = _logs(tmp / "logs", "calibration_parameters")
        text = log.read_text(encoding="utf-8")
        warnings_ = " ".join(_section_text(text, "[7] Warnings").split())     # the log collapses repeated spaces
        assert ("input: calib source run = electron 4 [command line] (default muon 4); the stage ran on what was "
                "given, check its results") in warnings_, warnings_
        errors = _section_text(text, "[8] Bugs / errors")
        assert "USER/INPUT ERROR" in errors and "[calib] electron 4: synthetic stop of the stage" in errors
        assert "input runs of this stage (as given; compare them with the error above):" in errors
        assert "source run" in errors and "electron 4 (" in errors, errors


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["run_log"]))
