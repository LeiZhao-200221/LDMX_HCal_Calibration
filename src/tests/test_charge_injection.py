"""Tests of the ci stage (ldmx_hcal.energy.charge_injection): synthetic data, about a minute.

    python -X utf8 src/tests/run_all.py charge_injection          (from the data_analysis folder)

Synthetic tests:
  * names and units (Lei 2026-10-20): the scales adc_only, pureci (PureCI) and s; the output units parameters, events,
    linearity, resolution, eh and the checks response, quality_scan chosen with -only; no energy spectra pdf.
  * s samples: -scale-particle a b gives one path s_<sample> each; s = the sample with the fewest channels that
    borrow from the prior (tie: the first given); the products are tagged s_<samples> unless pion alone.
  * the parameter cache (cache/, sidecar-last product set) and its official copy (unit parameters only).
  * BLOCK 1-4 building blocks: monotone map, shrinkage, cross validation, channel keys, crossover estimator with
    the TOT cut in fC (Lei D2), rail-aware merge and bad-bar rules (Lei D4), compact event accumulation.
  * linearity per particle (electron fit, pion points; per-scale defaults; -linfit), resolution (electron fit),
    e/h on the s scale (filled / hollow points) and the response rows joined by run key.
  * complete synthetic stage runs: one event csv per input csv in a sub-folder per run, the per-event / per-bar /
    per-end cache arrays, -only units, -from-saved, -output by result name, rebuild commands, temp files.
  * Lei 2026-10-02 fit options: single Gaussians by default, -fit / -bin ranges, failed fits = points only.

Author: Lei
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import fitting, io_utils, outputs  # noqa: E402
from ldmx_hcal.common.config import Dataset, RunContext, resolve_datasets  # noqa: E402
from ldmx_hcal.common.io_utils import ChannelKeyError, ProductSetError, SettingsError  # noqa: E402
from ldmx_hcal.energy import charge_injection as ci  # noqa: E402
from ldmx_hcal.energy import energy_response as er  # noqa: E402

KEYS = ["layer", "strip", "end"]


# fn() must raise exc_type with every text in its message; returns the message
def _expect(exc_type, fn, *texts):
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            fn()
    except exc_type as exc:
        msg = str(exc)
        for text in texts:
            assert text in msg, (text, msg)
        return msg
    raise AssertionError(f"{getattr(fn, '__name__', fn)} must raise {exc_type.__name__}")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
RUNS = {"electron_0.5GeV": ("electron", 0.5), "electron_1GeV": ("electron", 1.0), "electron_4GeV": ("electron", 4.0),
        "muon_4GeV": ("muon", 4.0), "pion_3GeV": ("pion", 3.0), "pion_4GeV": ("pion", 4.0)}


# a RunContext under tmp; outputs () = --output alone, None = no --output, names = --output NAME ...
def _ctx(tmp: Path, test: bool = False, stage_cfg: dict | None = None, dry_run: bool = False,
         make_plots: bool = False, clean_input: bool = False, outputs: tuple | None = (),
         write_json: bool = True) -> RunContext:
    data = tmp / "data"
    cfg = {"paths": {"data_dir": data, "generated_data_dir": data, "calibration_dir": tmp / "calibration",
                     "results_dir": tmp / "results", "cache_dir": tmp / "cache",
                     "channel_map": tmp / "channel_map.csv", "ci_scan_dir": tmp / "scans",
                     "analysis_files_dir": data / "analysis_files", "root_dir": data / "root"},
           "datasets": {k: {"particle": p, "energy_gev": e} for k, (p, e) in RUNS.items()},
           "safety": {"protected_dirs": []}, "run": {"test_events": 5000, "chunksize": 2_000_000}}
    if stage_cfg:
        cfg["charge_injection"] = stage_cfg
    return RunContext(cfg=cfg, test=test, dry_run=dry_run, make_plots=make_plots, workers=1, clean_input=clean_input,
                      outputs=None if outputs is None else list(outputs), write_json=write_json)


# the ci stage parser plus the runner options a stage reads (the argv joined as runner.main does)
def _args(*argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--from-saved", "-from-saved", dest="from_saved", action="store_true")
    ci.add_arguments(p)
    return p.parse_args(fitting.normalize_range_args(list(argv)))


# the argparse error message of a refused ci command line ('' when it parses)
def _parse_error(*argv) -> str:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            _args(*argv)
    except SystemExit:
        return err.getvalue()
    return ""


# resolve_settings with -only applied as the runner does (ctx.units)
def _settings_of(ctx, *argv) -> dict:
    args = _args(*argv)
    ctx.units = outputs.selected_units(args)
    return ci.resolve_settings(ctx, args)


# ci.run with -only applied as the runner does; returns the log
def _run(ctx, datasets, *argv) -> str:
    args = _args(*argv)
    ctx.units = outputs.selected_units(args)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ci.run(ctx, args, datasets)
    return out.getvalue()


# ci.io_plan with -only applied as the runner does
def _plan(ctx, datasets, *argv) -> list:
    args = _args(*argv)
    ctx.units = outputs.selected_units(args)
    return ci.io_plan(ctx, args, datasets)


# a run (Dataset) without a folder on disk
def _ds(key, particle, energy) -> Dataset:
    return Dataset(key=key, particle=particle, energy_gev=float(energy), label=key, kind="data")


# a run summary as _respond builds it, with given fits
def _summary(ds: Dataset, mu: float, sigma: float, n: int = 100, paths=("s",)) -> dict:
    fit = {"kind": "two_pass", "ok": True, "status": "ok", "mu": mu, "sigma": sigma, "mu_error": sigma / 50,
           "sigma_error": sigma / 70, "sigma_over_mu": sigma / mu, "chi2": 1.0, "ndf": 3}
    scan = {"mu": mu, "sigma": sigma, "sigma_over_mu": sigma / mu, "n_detected_peaks": 1, "status": "ok"}
    return {"ds": ds, "n_csv": 1, "n_events_in_file": n, "n_reconstructed": n, "n_selected": n,
            "n_bars_selected": 10 * n, "fits": {p: {"fit": dict(fit), "scan": dict(scan)} for p in paths}}


# ci settings (DEFAULTS, selection by s) with the energy_response settings rs, checked as the flags are
def _fit_settings(**rs) -> dict:
    s = ci._settings({})
    s.update(selection_path_used="s", energy_response=er.check_settings(er.settings(None, rs)))
    return s


# spectrum rows by run key
def _rows_by_key(rows, path="s") -> dict:
    return {r["dataset"]: r for r in rows if r["row_type"] == "spectrum" and r["path"] == path}


# call quietly
def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------------------------------
# names and units
# ----------------------------------------------------------------------------------------------
# scales and units
def test_scales_and_units():
    assert ci.PATHS == ("adc_only", "pureci", "s") and ci.DEFAULTS["selection_path"] == "s"
    assert ci.UNITS == ("parameters", "events", "linearity", "resolution", "eh", "response", "quality_scan")
    assert ci.UNIT_GROUPS == {"figures": ("linearity", "resolution", "eh"), "checks": ("response", "quality_scan")}
    assert _args().only == ("parameters", "events", "linearity", "resolution", "eh")  # checks only on request
    assert _args("-only", "checks").only == ("response", "quality_scan")
    assert _args("-only", "events,eh", "-only", "parameters").only == ("parameters", "events", "eh")
    assert "unknown unit" in _parse_error("-only", "spectra")
    s = _settings_of(_ctx(Path(tempfile.gettempdir())), "-pureci")
    assert s["energy_paths"] == ["pureci"] and s["compute_paths"] == ["pureci", "s"]  # s selects the events
    assert ci.event_columns(["adc_only", "pureci", "s"]) == [
        "event_source", "pf_event", "energy_adc_only_mip", "energy_pureci_mip", "energy_s_mip", "n_bars",
        "bad_strip_fraction", "selected"]
    assert [n for n in ci.bar_record_dtype(["pureci", "s"]).names] == [
        "pf_event", "layer", "strip", "tot_end0", "tot_end1", "bar_energy_pureci_mip", "end0_energy_pureci_mip",
        "end1_energy_pureci_mip", "bar_energy_s_mip", "end0_energy_s_mip", "end1_energy_s_mip"]
    source = Path(ci.__file__).read_text(encoding="utf-8")
    assert "ci_tot_threshold_fc" in ci.CI_TABLE_COLUMNS and "ci_thr_status" in ci.CI_TABLE_COLUMNS
    assert ci.BAD_BAR_RULES == ("used_ends", "used_ends_strict")
    for column in (*ci.CI_TABLE_COLUMNS, *ci.NODE_COLUMNS):
        assert column in ci.__doc__, column
    assert all(len(line) <= 120 for line in source.splitlines())
    assert ci.DEFAULT_DATASETS == ["electron 0.3", "electron 0.4", "electron 0.5", "electron 1", "electron 4",
                                   "pion 0.5", "pion 1", "pion 3", "pion 4"]


# output routing
def test_output_routing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        e4 = _ds("electron_4GeV", "electron", 4)
        for test, mode in ((False, "fullset"), (True, "test")):
            ctx = _ctx(tmp, test=test)
            sub = "test" if test else ""
            cal = tmp / "calibration" / sub / "charge_injection"
            dat = tmp / "data" / sub / "charge_injection"
            res = tmp / "results" / mode / "charge_injection"
            cache = tmp / "cache" / sub / "charge_injection"
            assert ci.parameters_csv(ctx) == cal / "ci_channel_parameters.csv"
            assert ci.nodes_csv(ctx, "s_muon") == cal / "ci_tot_charge_nodes_s_muon.csv"
            assert ci.parameters_record(ctx) == cal / "ci_channel_parameters.provenance.json"
            assert ci.parameters_cache_files(ctx)[2] == cache / "parameters" / "parameters.json"
            csv = tmp / "data" / "analysis_files" / "Electron_4GeV" / "fixed_Electron_4GeV.csv"
            run_dat, run_cache = dat / "Electron_4GeV", cache / "events" / "Electron_4GeV"
            assert ci.event_energies_csv(ctx, e4, csv) == run_dat / "fixed_Electron_4GeV_event_energies.csv"
            assert ci.event_energies_csv(ctx, e4, csv, "q") == (dat / "Electron_4GeV"
                                                                 / "fixed_Electron_4GeV_event_energies_q.csv")
            assert ci.event_cache_files(ctx, e4, csv) == (run_cache / "fixed_Electron_4GeV.npz",
                                                          run_cache / "fixed_Electron_4GeV_bars.npy",
                                                          run_cache / "fixed_Electron_4GeV.json")
            assert ci.event_cache_files(ctx, e4, csv, "clean")[0].parent == cache / "events_clean" / "Electron_4GeV"
            for unit, name in (("response", "energy_response.csv"), ("quality_scan", "quality_threshold_scan.csv"),
                               ("linearity", "energy_linearity.pdf"), ("resolution", "energy_resolution.pdf"),
                               ("eh", "e_over_h.pdf")):
                assert ci.result_default(ctx, unit) == res / name
            assert ci.result_default(ctx, "eh", "q") == res / "e_over_h_q.pdf"
            assert ci.results_record(ctx) == cache / "final_results.json"
        _expect(KeyError, lambda: ci.result_file(_ctx(tmp), "energy_spectra"), "unknown final result")


# variant tags scale samples and output tag validation
def test_variant_tags_scale_samples_and_output_tag_validation():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        cases = [((), "", ["adc_only", "pureci", "s"]),
                 (("--scale-particle", "muon"), "s_muon", ["adc_only", "pureci", "s"]),
                 (("--scale-particle", "pion"), "", ["adc_only", "pureci", "s"]),
                 (("--scale-particle", "mix"), "s_mix", ["adc_only", "pureci", "s"]),
                 (("--scale-particle", "pion", "muon"), "s_pion_muon",
                  ["adc_only", "pureci", "s", "s_pion", "s_muon"]),
                 (("--scale-particle", "muon", "electron", "-s"), "s_muon_electron", ["s", "s_muon", "s_electron"]),
                 (("--scale-particle", "muon", "-pureci", "--selection-path", "pureci"), "s_muon", ["pureci"]),
                 (("--output-tag", "q0p1"), "q0p1", ["adc_only", "pureci", "s"]),
                 (("--scale-particle", "muon", "--output-tag", "X"), "s_muon_X", ["adc_only", "pureci", "s"]),
                 (("--shrinkage", "cv", "--bad-bar-rule", "used_ends_strict", "-linfit", "electron[0,100]",
                   "-normalerror"), "", ["adc_only", "pureci", "s"])]
        for argv, tag, paths in cases:
            s = _settings_of(ctx, *argv)
            assert (s["variant_tag"], s["output_paths"]) == (tag, paths), (argv, s["variant_tag"], s["output_paths"])
        s = _settings_of(ctx, "--scale-particle", "pion", "muon", "-adc-only")
        assert s["compute_paths"] == ["adc_only", "s"] and s["scale_particles"] == ["pion", "muon"]
        clean = _ctx(Path(tmp), clean_input=True)
        for argv, expected in (((), "clean"), (("--output-tag", "X"), "X_clean"),
                               (("--scale-particle", "muon"), "s_muon_clean")):
            assert _settings_of(clean, *argv)["variant_tag"] == expected, argv
        for bad in ("a/b", "x y", "..\\up", "x!y"):
            _expect(SystemExit, lambda: _settings_of(ctx, "--output-tag", bad))
        for reserved in ("s_pion", "S_PION", "s_electron_x", "s_muon", "s_mix", "s_pion_muon", "clean", "x_clean"):
            _expect(SystemExit, lambda: _settings_of(ctx, "--output-tag", reserved), "reserved")
        for fine in ("s_pion2", "x_s_pion", "spion", "cleaner", "unclean"):
            assert _settings_of(ctx, "--output-tag", fine)["variant_tag"] == fine
        assert "unrecognized" in _parse_error("--scale-particle") or "expected" in _parse_error("--scale-particle")
        assert "invalid choice" in _parse_error("--scale-particle", "kaon")
        ctx.cfg["charge_injection"] = {"scale_particle": ["pion", "pion"]}
        _expect(SettingsError, lambda: _settings_of(ctx), "scale_particle", "distinct")
        ctx.cfg["charge_injection"] = {"scale_particle": ["muon", "pion"]}
        assert _settings_of(ctx)["variant_tag"] == "s_muon_pion"
    assert ci.reserved_tag_clash("s_pion") == "s_pion" and ci.reserved_tag_clash("S_Pion_x") == "s_pion"
    assert "s_pion" not in ci.RESERVED_TAG_PARTS and set(ci.RESERVED_TAG_PARTS) == {"s_muon", "s_electron", "s_mix"}
    assert ci.path_scale("s_muon") == "s" and ci.path_scale("pureci") == "pureci" and ci.is_scale_path("s_mix")
    assert ci.path_label("s_muon") == "charge injection x s_muon (muon training)"
    assert ci.expand_paths(["s", "adc_only"], ["muon"]) == ["adc_only", "s"]


# settings are validated before any io
def test_settings_are_validated_before_any_io():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        for argv, text in ((("--shrinkage", "-1"), "shrinkage_lambda"), (("--shrinkage", "abc"), "shrinkage"),
                           (("--quality-threshold", "nan"), "quality_threshold"),
                           (("--min-link-hits", "-2"), "min_link_hits"), (("-min-bin-count", "-1"), "min_bin_count"),
                           (("-bin", "30"), "--bin needs --fit"),
                           (("-fit[5,10]", "-bin", "30,40"), "one value per fit range")):
            _expect(SettingsError, lambda: _settings_of(ctx, *argv), text)
        for yml, text in (({"chunk_rows": 0}, "chunk_rows"), ({"lambda_grid": []}, "lambda_grid"),
                          ({"fit_quantiles": [0.9, 0.1]}, "fit_quantiles"), ({"ratio_max": float("nan")}, "ratio_max"),
                          ({"known_bad_bars": [[12, 4.5]]}, "known_bad_bars"), ({"quality_filter": "yes"}, "quality"),
                          ({"bad_bar_rule": "nope"}, "bad_bar_rule"), ({"lambda_grids": [1.0]}, "unknown"),
                          ({"energy_paths": ["nope"]}, "energy_paths")):
            ctx.cfg["charge_injection"] = yml
            _expect(SettingsError, lambda: _settings_of(ctx), text)
        ctx.cfg["charge_injection"] = {"chunk_rows": 0, "ratio_max": -1.0}
        msg = _expect(SettingsError, lambda: _settings_of(ctx))
        assert "chunk_rows" in msg and "ratio_max" in msg  # every problem at once
        ctx.cfg["charge_injection"] = {"bad_bar_rule": "USED_ENDS_STRICT", "audit_failures": False}
        s = _settings_of(ctx)
        assert s["bad_bar_rule"] == "used_ends_strict" and s["audit_failures"] is False
        ctx.cfg["charge_injection"] = {}
        assert _settings_of(ctx, "--shrinkage", "CV")["shrinkage_lambda"] == "cv"
        assert _settings_of(ctx, "-no-audit-failures")["audit_failures"] is False
        ctx.cfg["energy_response"] = {"linearity_mod": "full"}
        _expect(SettingsError, lambda: _settings_of(ctx), "linearity_mod", "unknown")
        ctx.cfg["energy_response"] = {"fit_ranges": [[5, 10]]}
        _expect(SettingsError, lambda: _settings_of(ctx), "fit_ranges", "command-line only")
        ctx.cfg["energy_response"] = {"normal_error": True}
        assert _settings_of(ctx)["energy_response"]["normal_error"] is True
        # units: the figures need plots, -from-saved fits no parameters; nothing left is refused
        ctx.cfg["energy_response"] = {}
        assert _settings_of(ctx, "-only", "figures,parameters")["units"] == ("parameters",)  # make_plots False
        _expect(SettingsError, lambda: _settings_of(ctx, "-only", "eh"), "makes nothing", "-no-plots")
        _expect(SettingsError, lambda: _settings_of(ctx, "-only", "parameters", "--from-saved"), "-from-saved")
        assert _settings_of(ctx, "--from-saved")["units"] == ("events",)
        ctx.make_plots = True
        assert _settings_of(ctx, "--from-saved")["units"] == ("events", "linearity", "resolution", "eh")


# ----------------------------------------------------------------------------------------------
# BLOCK 1-2 building blocks
# ----------------------------------------------------------------------------------------------
# pava and monotone charge map
def test_pava_and_monotone_charge_map():
    assert np.allclose(ci.pava_increasing([1.0, 3.0, 2.0, 4.0]), [1.0, 2.5, 2.5, 4.0])
    assert np.allclose(ci.pava_increasing([3.0, 2.0, 1.0], [1.0, 1.0, 2.0]), [1.75, 1.75, 1.75])
    t = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    q = 2.0 * t + 50.0
    m = ci.MonotoneChargeMap(t, q)
    assert np.allclose(m(t), q) and math.isclose(m(250.0), 550.0, rel_tol=1e-12)
    assert math.isclose(m(700.0), q[-1] + 200.0, rel_tol=1e-12) and math.isclose(m(50.0), q[0] - 100.0)
    assert m(-10_000.0) == 0.0 and isinstance(m(250.0), float) and m(np.array([250.0])).shape == (1,)
    nodes = pd.DataFrame({"layer": 3, "strip": 4, "end": 1, "node_index": np.arange(6), "tot_counts": t, "q_fc": q})
    mappers = ci.build_tot_charge_mappers(nodes)
    out = ci.q_from_tot([250.0, 250.0, 250.0], [3, 3, 5], [4, 4, 4], [1, 2, 1], mappers)
    assert math.isclose(out[0], 550.0) and np.isnan(out[1]) and np.isnan(out[2])


# tot threshold calibration names and channel status
def test_tot_threshold_calibration_names_and_channel_status():
    # The TOT-threshold calibration (provided by Erik Wallin) keeps its 7 parameters under ci_tot_threshold_fc /
    # ci_thr_*; a channel without injection events has no threshold and is protected.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "map.csv").write_text("DPM,LINK,CHAN,PLANE,STRIP,END\n0,0,3,1,0,0\n")
        scan = tmp / "scan.csv"
        pd.DataFrame(columns=ci.SCAN_COLUMNS).to_csv(scan, index=False)
        table, nodes = _quiet(ci.calibrate_charge_injection, [scan], tmp / "map.csv", None)
        assert list(table.columns) == ci.CI_TABLE_COLUMNS and len(nodes) == 0
        row = table.iloc[0]
        assert row["ci_status"] == ci.STATUS_PROTECTED and row["ci_thr_status"] == "NO_INJECTION_EVENTS"
        assert "ci_tot_threshold_fc" in row["ci_reason"] and math.isnan(row["ci_tot_threshold_fc"])
        assert ci.threshold_parameter_table(table).empty
        assert list(ci.threshold_parameter_table(table.assign(ci_tot_threshold_fc=500.0)).columns) == [
            *KEYS, *ci.THRESHOLD_COLUMNS]


# link prior shrinkage and cv fallback
def test_link_prior_shrinkage_and_cv_fallback():
    direct = np.array([0.5, 0.8, np.nan, 0.2, 0.4])
    counts = np.array([10, 2, 0, 5, 0])
    dpm, link = np.array([0, 0, 0, 1, 1]), np.array([0, 0, 0, 1, 1])
    ok = np.array([True, True, True, True, False])
    out = ci.shrink_to_link_prior(direct, counts, dpm, link, ok, shrinkage=5.0)
    prior0 = math.exp(np.median(np.log([0.5, 0.8])))
    assert math.isclose(out["prior"][0], prior0) and math.isclose(out["prior"][3], 0.2)
    w = 10 / 15
    assert math.isclose(out["estimate"][0], math.exp(w * math.log(0.5) + (1 - w) * math.log(prior0)), rel_tol=1e-12)
    assert math.isclose(out["estimate"][2], prior0) and np.isnan(out["estimate"][4])
    guarded = ci.shrink_to_link_prior(direct, counts, dpm, link, ok, 5.0, min_link_hits=6)
    assert guarded["prior_level"][3] == "global" and guarded["prior_level"][0] == "link"
    kmax = 2100
    hits = {"uid": np.zeros(0, np.int32), "ratio": np.zeros(0), "parity": np.zeros(0, np.int8),
            "pair_uid0": np.zeros(0, np.int32), "pair_uid1": np.zeros(0, np.int32), "pair_ratio0": np.zeros(0),
            "pair_ratio1": np.zeros(0), "pair_parity": np.zeros(0, np.int8),
            "censored_counts": np.zeros(2 * kmax, np.int64), "bar_key_size": kmax, "stats": {}}
    table = pd.DataFrame({"layer": [1, 1], "strip": [0, 0], "end": [0, 1], "dpm": [0, 0], "link": [0, 0],
                          "ci_status": [ci.CI_OK, ci.CI_OK], "ci_tot_threshold_fc": [500.0, 500.0]})
    st = _quiet(ci.scale_table_from_hits, hits, ci.scale_fit_metadata(table, None), table, {"shrinkage_lambda": "cv"})
    assert st["shrinkage"].tolist() == [5.0, 5.0] and (st["s_source"] == ci.SOURCE_NO_MODEL).all()
    assert st.attrs["diagnostics"]["cv_fallback"] is True
    assert not [c for c in st.columns if "floor" in c]


# scale table of one sample: s_final values and their sources (for the default-sample choice)
def _scale(sample, values, sources) -> pd.DataFrame:
    return pd.DataFrame({"layer": [1, 1, 2], "strip": [0, 0, 0], "end": [0, 1, 0], "dpm": 0, "link": 0,
                         "ci_status": ci.CI_OK, "training_particle": sample, "shrinkage": 5.0, "s_direct": values,
                         "s_exact_n": 1, "s_censored_n": 0, "shrinkage_weight": 0.5, "s_link_prior": 0.6,
                         "s_prior_level": "link", "s_final": values, "s_source": sources, "global_prior": 0.6,
                         "cv_json": "{}"})


# default s is the sample that borrows least
def test_default_s_is_the_sample_that_borrows_least():
    # Lei 2026-10-20: among several samples the default s is the one with the fewest channels whose s comes from
    # the (DPM, LINK) / global prior, whatever the order (also against pion); a tie keeps the first given.
    D, B, N = ci.SOURCE_DIRECT, ci.SOURCE_BORROWED, ci.SOURCE_NO_MODEL
    pion = _scale("pion", [0.5, 0.6, 0.7], [D, B, B])
    muon = _scale("muon", [0.55, 0.65, 0.75], [D, D, B])
    electron = _scale("electron", [0.4, 0.4, np.nan], [D, B, N])
    assert ci.borrowed_channels(pion) == 2 and ci.borrowed_channels(electron) == 1
    assert ci.choose_default_sample({"pion": pion, "muon": muon}) == ("muon", {"pion": 2, "muon": 1})
    assert ci.choose_default_sample({"muon": muon, "pion": pion})[0] == "muon"
    assert ci.choose_default_sample({"electron": electron, "muon": muon})[0] == "electron"  # tie: first given
    assert ci.choose_default_sample({"pion": pion})[0] == "pion"
    cols = ci.scale_columns({"pion": pion, "muon": muon}, "muon")
    assert cols["s_final"].tolist() == [0.55, 0.65, 0.75] and set(cols["training_particle"]) == {"muon"}
    assert cols["s_pion"].tolist() == [0.5, 0.6, 0.7] and cols["s_muon"].tolist() == cols["s_final"].tolist()
    assert cols["s_pion_source"].tolist() == [D, B, B] and "cv_json" not in cols and "dpm" not in cols
    single = ci.scale_columns({"pion": pion}, "pion")
    assert "s_pion" not in single and single["s_final"].tolist() == [0.5, 0.6, 0.7]


# random exact-overlap hits of 24 channels
def _random_hits(seed=4, n=4000, n_channels=24):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"layer": np.repeat(np.arange(1, 7), 4), "strip": np.tile([0, 0, 1, 1], 6),
                          "end": np.tile([0, 1], 12), "dpm": np.repeat([0, 1], 12), "link": np.repeat([0, 1, 2, 3], 6),
                          "ci_status": ci.CI_OK, "ci_tot_threshold_fc": 500.0})
    meta = ci.scale_fit_metadata(table, None)
    uid = meta["uid"][rng.integers(0, n_channels, n)].astype(np.int32)
    pair_key = (table["layer"].to_numpy() * 100 + table["strip"].to_numpy())[rng.integers(0, n_channels, 600)]
    hits = {"uid": uid, "ratio": rng.lognormal(-0.3, 0.2, n).astype(np.float32).astype(float),
            "parity": rng.integers(0, 2, n).astype(np.int8), "pair_uid0": pair_key.astype(np.int32),
            "pair_uid1": (2100 + pair_key).astype(np.int32), "pair_ratio0": rng.lognormal(-0.3, 0.2, 600),
            "pair_ratio1": rng.lognormal(-0.3, 0.2, 600), "pair_parity": rng.integers(0, 2, 600).astype(np.int8),
            "censored_counts": np.zeros(4200, np.int64), "bar_key_size": 2100, "stats": {}}
    return hits, meta, table


# The even-event channel medians are computed once for the whole lambda grid (and reused by the diagnostics).
def test_cross_validation_reuses_the_training_medians():
    hits, meta, table = _random_hits()
    s = ci._settings({"shrinkage_lambda": "cv"})
    train = hits["parity"] == 0
    stats = ci.group_statistics(hits, train)
    for lam in s["lambda_grid"]:
        direct, counts = ci.channel_medians(hits["uid"], hits["ratio"], train, 4200)
        ref = ci.shrink_to_link_prior(direct, counts, meta["dpm"], meta["link"], meta["scale_ok"], float(lam),
                                      0)["estimate"]
        est = ci.robust_group_estimates(hits, train, meta, lam, s, stats=stats)["estimate"]
        assert np.array_equal(est, ref, equal_nan=True)
    calls = []
    real = ci.channel_medians

    # count the median calls
    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    with patch.object(ci, "channel_medians", side_effect=counting):
        st = _quiet(ci.scale_table_from_hits, hits, meta, table, s)
    assert len(calls) == 2  # training medians once + the full sample
    reference = _quiet(ci.scale_table_from_hits, hits, meta, table, s)
    assert np.array_equal(st["s_final"].to_numpy(), reference["s_final"].to_numpy(), equal_nan=True)


# A sample of several runs / csv files: the hits are concatenated and fitted once.
def test_pooled_training_hits():
    # synthetic hits of one file
    def hits(seed, n):
        r = np.random.default_rng(seed)
        return {"uid": r.integers(0, 4, n).astype(np.int32), "ratio": r.uniform(0.5, 1.5, n),
                "parity": r.integers(0, 2, n).astype(np.int8), "pair_uid0": np.zeros(0, np.int32),
                "pair_uid1": np.zeros(0, np.int32), "pair_ratio0": np.zeros(0), "pair_ratio1": np.zeros(0),
                "pair_parity": np.zeros(0, np.int8), "censored_counts": np.ones(8, np.int64), "bar_key_size": 4,
                "stats": {"training_file": f"run{seed}.csv", "rows_seen": n, "candidate_hits": n, "exact_hits": n,
                          "censored_hits": 8, "exact_pairs": 0, "exact_pairs_odd": 0, "stream_s": 0.1}}
    a, b, c = hits(1, 50), hits(2, 70), hits(3, 30)
    pooled = _quiet(ci.merge_exact_overlap_hits, [a, b, c])
    assert len(pooled["uid"]) == 150 and pooled["stats"]["exact_hits"] == 150
    assert np.array_equal(pooled["ratio"], np.concatenate([a["ratio"], b["ratio"], c["ratio"]]))
    assert pooled["censored_counts"].tolist() == [3] * 8
    assert pooled["stats"]["training_file"] == "run1.csv + run2.csv + run3.csv"
    assert _quiet(ci.merge_exact_overlap_hits, [a]) is a
    assert ci.DEFAULTS["scale_training_datasets"] == {"pion": "pion 4", "muon": "muon 4", "electron": "electron 4",
                                                      "mix": ["pion 4", "muon 4", "electron 4"]}


# every ci input table refuses invalid or duplicated keys
def test_every_ci_input_table_refuses_invalid_or_duplicated_keys():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good_map = "DPM,LINK,CHAN,PLANE,STRIP,END\n0,0,3,1,0,0\n0,0,4,1,0,1\n"
        path = tmp / "map.csv"
        path.write_text(good_map)
        assert len(ci.load_channel_map(path)) == 2
        for text, what in ((good_map + "0,0,5,1,0,0\n", "share"), (good_map + "0,0,3,1,1,0\n", "electronics"),
                           ("DPM,LINK,CHAN,PLANE,STRIP,END\n0,0,3,1,0.5,0\n", "non-integer"),
                           ("DPM,LINK,CHAN,PLANE,STRIP,END\n0,0,3,1,0,2\n", "outside the detector"),
                           ("DPM,LINK,CHAN,PLANE,STRIP,END\n0,-1,3,1,0,0\n", "DPM / LINK / CHAN")):
            path.write_text(text)
            _expect(ChannelKeyError, lambda: ci.load_channel_map(path), what)
        path.write_text(good_map)
        ci_rows = ci.load_channel_map(path)
        cal = pd.DataFrame({"layer": [1, 1, 1], "strip": [0, 0, 0], "end": [0, 0, 1], "pedestal": [100.0, 900.0, 100.0],
                            "pedestal_range_with_20percent": 120.0, "mpv": [50.0, 10.0, 50.0]})
        _expect(ChannelKeyError, lambda: ci._calibration_value_arrays(ci_rows, cal), "calibration table", "share")
        assert ci._calibration_value_arrays(ci_rows, cal.iloc[[0, 2]])["mpv"].tolist() == [50.0, 50.0]
        nodes = pd.DataFrame({"layer": 1, "strip": 0, "end": [0, 0, 0, 1], "node_index": [0, 1, 2, 0],
                              "tot_counts": [10.0, 20.0, 30.0, 10.0], "q_fc": [100.0, 200.0, 300.0, 100.0]})
        assert len(ci.check_node_table(nodes, "nodes")) == 4
        for change, what in (({"node_index": [0, 1, 1, 0]}, "share"),
                             ({"tot_counts": [10.0, 30.0, 20.0, 10.0]}, "increase"),
                             ({"tot_counts": [10.0, np.nan, 30.0, 10.0]}, "non-finite"),
                             ({"layer": [1, 1, 1, 20]}, "outside")):
            _expect(ChannelKeyError, lambda: ci.check_node_table(nodes.assign(**change), "nodes"), what)


# ----------------------------------------------------------------------------------------------
# BLOCK 3-4: estimators, merges, bad-bar flags (D4), selection, compact accumulation
# ----------------------------------------------------------------------------------------------
# one bar L2 S3: pedestal 100, hit threshold 120, mpv 50, q = 2 TOT [fC], k = 1.5, TOT threshold 300 fC, s 0.8 / 0.5
def _synthetic_parameters(bad_bar_rule="used_ends", paths=("adc_only", "pureci", "s")):
    calibration = pd.DataFrame({"layer": [2, 2], "strip": [3, 3], "end": [0, 1], "pedestal": [100.0, 100.0],
                                "pedestal_range_with_20percent": [120.0, 120.0], "mpv": [50.0, 50.0]})
    table = pd.DataFrame({"layer": [2, 2], "strip": [3, 3], "end": [0, 1], "ci_tot_threshold_fc": [300.0, 300.0],
                          "ci_k_adc_per_fc": [1.5, 1.5], "mono_q_min_fc": [0.0, 0.0], "mono_q_max_fc": [8000.0, 8000.0],
                          "s_final": [0.8, 0.5], "s_muon": [0.4, 0.4]})
    tot = np.array([0.0, 1000.0, 2000.0, 4000.0])
    nodes = pd.concat([pd.DataFrame({"layer": 2, "strip": 3, "end": e, "node_index": np.arange(4), "tot_counts": tot,
                                     "q_fc": 2.0 * tot}) for e in (0, 1)], ignore_index=True)
    return ci.build_reconstruction_parameters(calibration, table, nodes,
                                              ci._settings({"bad_bar_rule": bad_bar_rule,
                                                            "compute_paths": list(paths)}))


# end inputs of rows of the bar L2 S3
def _ends(params, tot0, adc0, max0, tot1, adc1, max1):
    n = len(tot0)
    layer, strip = np.full(n, 2), np.full(n, 3)
    key = layer * 100 + strip
    bad = np.zeros(n, bool)
    return [ci.end_common_inputs(e, np.asarray(t, float), np.asarray(a, float), np.asarray(m, float), layer, strip,
                                 key, bad, params) for e, t, a, m in ((0, tot0, adc0, max0), (1, tot1, adc1, max1))]


# crossover estimator and the tot cut in charge
def test_crossover_estimator_and_the_tot_cut_in_charge():
    # Lei D2: a TOT end counts only when q(TOT) > the TOT threshold (both fC): TOT 200 counts = 400 fC > 300 fC is
    # converted, TOT 100 = 200 fC is not.  E = max(ADC energy, k s q / mpv).
    params = _synthetic_parameters(paths=("adc_only", "pureci", "s", "s_muon"))
    assert set(params["scales"]) == {"s", "s_muon"}
    c0, _ = _ends(params, [0, 1000, 200, 100, 1000], [400, 600, 600, 600, 130], [500, 1023, 1023, 1023, 1023],
                  [0] * 5, [300] * 5, [500] * 5)
    assert c0["candidate"].tolist() == [False, True, True, False, True]
    raw = (np.array([400, 600, 600, 600, 130]) - 100) / 50
    adc = ci.adc_only(c0)
    assert np.allclose(adc["energy"], raw) and not adc["tot_conversion_used"].any()
    pure = ci.pureci(c0)
    tot_e = 1.5 * 2000.0 / 50.0   # k * 1 * q(TOT=1000) / mpv = 60 MIP
    assert np.isclose(pure["energy"][1], tot_e) and np.isclose(pure["energy"][2], 1.5 * 400 / 50)
    assert pure["tot_conversion_used"].tolist() == [False, True, True, False, True]
    s = ci.s_correction(c0, params)
    assert np.isclose(s["energy"][1], 0.8 * tot_e)                       # end 0 has s = 0.8
    assert np.isclose(s["energy"][4], max(raw[4], 0.8 * tot_e))          # max(ADC, TOT): TOT never lowers
    assert np.isclose(ci.s_correction(c0, params, "s_muon")["energy"][1], 0.4 * tot_e)
    for path, fn in (("adc_only", ci.adc_only), ("pureci", ci.pureci)):
        assert np.array_equal(ci.end_energies(path, c0, params)["energy"], fn(c0)["energy"])
    assert np.array_equal(ci.end_energies("s_muon", c0, params)["energy"],
                          ci.s_correction(c0, params, "s_muon")["energy"])
    assert ci.bar_merge("adc_only") is ci.merge_raw and ci.bar_merge("s_muon") is ci.merge_rail_aware


# rail aware merge reports used ends and d4 bad bars
def test_rail_aware_merge_reports_used_ends_and_d4_bad_bars():
    # Lei D4: a bar is bad when no used end is trusted and an end is clipped (saturated, TOT not convertible);
    # used_ends_strict also flags used saturated ends whose convertible TOT stayed below the clipped ADC value.
    params = _synthetic_parameters()
    # end0 at the rail, TOT 100 fired but below the TOT threshold (no conversion); end1 clean 900
    c0, c1 = _ends(params, [100], [1100], [1023], [0], [900], [900])
    e0, e1 = ci.pureci(c0), ci.pureci(c1)
    bar, used0, used1 = ci.merge_rail_aware(e0, e1)
    assert np.isclose(bar[0], (1100 - 100) / 50) and used0.tolist() == [True] and used1.tolist() == [False]
    for rule in ci.BAD_BAR_RULES:
        bad, sat = ci.bar_quality_flags([e0, e1], [c0, c1], np.isfinite(bar), (used0, used1), rule)
        assert bad[0] and sat[0], rule
    _expect(ValueError, lambda: ci.bar_quality_flags([e0, e1], [c0, c1], np.isfinite(bar), (used0, used1), "x"),
            "bad_bar_rule")
    # both ends at the rail with a convertible TOT (400 counts = 800 fC) whose s energy (19.2 / 12 MIP) stays below
    # the clipped ADC energy 20 MIP: both ends used, none trusted, none clipped -> bad only under used_ends_strict
    c0, c1 = _ends(params, [400], [1100], [1023], [400], [1100], [1023])
    e0, e1 = ci.s_correction(c0, params), ci.s_correction(c1, params)
    assert e0["tot_conversion_available"][0] and not e0["tot_conversion_used"][0] and np.isclose(e0["energy"][0], 20)
    bar, used0, used1 = ci.merge_rail_aware(e0, e1)
    flags = {rule: ci.bar_quality_flags([e0, e1], [c0, c1], np.isfinite(bar), (used0, used1), rule)[0][0]
             for rule in ci.BAD_BAR_RULES}
    assert flags == {"used_ends": False, "used_ends_strict": True}
    # railed end WITHOUT TOT -> the other end (clean, trusted): good under both rules
    c0, c1 = _ends(params, [0], [1100], [1023], [0], [900], [900])
    e0, e1 = ci.pureci(c0), ci.pureci(c1)
    bar, used0, used1 = ci.merge_rail_aware(e0, e1)
    assert np.isclose(bar[0], 16.0) and (used0[0], used1[0]) == (False, True)
    for rule in ci.BAD_BAR_RULES:
        assert not ci.bar_quality_flags([e0, e1], [c0, c1], np.isfinite(bar), (used0, used1), rule)[0][0]
    # ADC-only merge: mean / single / no end above threshold
    c0, c1 = _ends(params, [0, 0, 0], [400, 400, 110], [500, 500, 500], [0, 0, 0], [600, 110, 110], [500, 500, 500])
    bar, used0, used1 = ci.merge_raw(ci.adc_only(c0), ci.adc_only(c1))
    assert np.allclose(bar[:2], [0.5 * (6 + 10), 6.0]) and np.isnan(bar[2])
    assert used0.tolist() == [True, True, False] and used1.tolist() == [True, False, False]


# d4 event bad strip fraction by rule
def test_d4_event_bad_strip_fraction_by_rule():
    chunk = pd.DataFrame({"pf_event": [5, 5, 6, 7], "layer": [2, 2, 2, 2], "strip": [3, 3, 3, 3],
                          "tot_end0": [100.0, 0.0, 0.0, 400.0], "adc_sum_end0": [1100.0, 500.0, 500.0, 1100.0],
                          "adc_max_end0": [1023.0, 500.0, 500.0, 1023.0], "tot_end1": [0.0, 0.0, 0.0, 400.0],
                          "adc_sum_end1": [900.0, 400.0, 400.0, 1100.0], "adc_max_end1": [900.0, 400.0, 400.0, 1023.0]})
    out = {}
    for rule in ci.BAD_BAR_RULES:
        params = _synthetic_parameters(rule)
        r = ci.reconstruct_chunk(chunk, params, ["s"], "s", end_values=True)
        assert {"s_end0", "s_end1"} <= set(r)
        acc = ci.EventAccumulator(["s"])
        acc.add(r["pf"], r["included"], r, r["unrecoverable"], r["saturated_unconverted"])
        out[rule] = acc.finalize()
    assert out["used_ends"]["bad_strip_fraction"].tolist() == [0.5, 0.0, 0.0]
    assert out["used_ends_strict"]["bad_strip_fraction"].tolist() == [0.5, 0.0, 1.0]
    assert np.array_equal(out["used_ends"]["energy"]["s"], out["used_ends_strict"]["energy"]["s"])
    s = ci._settings({})
    assert ci.select_events(out["used_ends"], s).tolist() == [False, True, True]
    assert ci.select_events(out["used_ends_strict"], s).tolist() == [False, True, False]


# event selection and chunk border sums
def test_event_selection_and_chunk_border_sums():
    ev = {"pf_event": np.arange(4), "bad_strip_fraction": np.array([0.0, 0.125, 0.12500001, 1.0])}
    s = ci._settings({})
    assert ci.select_events(ev, s).tolist() == [True, True, False, False]
    assert ci.select_events(ev, {**s, "quality_filter": False}).all()
    rng = np.random.default_rng(3)
    pf = np.repeat(np.arange(50), 7)
    energy = rng.random(pf.size)
    included = rng.random(pf.size) > 0.2
    bad = included & (rng.random(pf.size) > 0.9)
    results = []
    for cuts in ((pf.size,), (100, 101, 150), tuple(range(1, pf.size, 1))):
        acc = ci.EventAccumulator(["s"])
        edges = [0, *[c for c in cuts if c < pf.size], pf.size]
        for a, b in zip(edges[:-1], edges[1:]):
            acc.add(pf[a:b], included[a:b], {"s": energy[a:b]}, bad[a:b], np.zeros(b - a, bool))
        results.append(acc.finalize())
    for r in results[1:]:
        assert np.array_equal(r["pf_event"], results[0]["pf_event"])
        assert np.array_equal(r["n_bars"], results[0]["n_bars"])
        assert np.allclose(r["energy"]["s"], results[0]["energy"]["s"], rtol=0, atol=1e-12)
    expected = np.bincount(pf[included], weights=energy[included], minlength=50)
    keep = np.bincount(pf[included], minlength=50) > 0
    assert np.allclose(results[0]["energy"]["s"], expected[keep])


# sparse event ids use compact storage
def test_sparse_event_ids_use_compact_storage():
    rng = np.random.default_rng(9)
    ids = np.array([7, 10**9, 3, 2**46 - 1, 42, 10**12])
    pf = np.repeat(ids, 5)[rng.permutation(30)]
    energy = rng.random(pf.size)
    included = rng.random(pf.size) > 0.1
    bad = included & (rng.random(pf.size) > 0.7)
    acc = ci.EventAccumulator(["s", "adc_only"])
    requested = []
    real_reserve = acc._reserve
    acc._reserve = lambda size: (requested.append(size), real_reserve(size))[1]
    for a, b in ((0, 7), (7, 8), (8, 21), (21, pf.size)):
        acc.add(pf[a:b], included[a:b], {"s": energy[a:b], "adc_only": 2 * energy[a:b]}, bad[a:b],
                np.zeros(b - a, bool))
    out = acc.finalize()
    assert max(requested) <= len(ids) and acc._cap < 2048
    assert out["pf_event"].tolist() == sorted(ids.tolist()) and out["n_events_in_file"] == len(ids)


# ----------------------------------------------------------------------------------------------
# the parameter cache (cache/) and its official copy (unit parameters)
# ----------------------------------------------------------------------------------------------
# mocked BLOCK 1 / BLOCK 2 of one channel: counts the calls, marks each table
class _FakeFits:
    # Mocked BLOCK 1 / BLOCK 2 (one channel): counts the calls; s_final = lambda + an offset per sample (a marker
    # of the fit that wrote a table); the muon sample borrows nothing, the pion sample one channel.

    # no call yet
    def __init__(self):
        self.ci, self.scale = 0, []

    # the CI map of one channel
    def calibrate(self, scan_files, channel_map_csv, settings, calibration_df=None):
        self.ci += 1
        table = pd.DataFrame({c: [np.nan] for c in ci.CI_TABLE_COLUMNS})
        table[["layer", "strip", "end", "dpm", "link", "chan"]] = [[1, 0, 0, 0, 0, 3]]
        table["ci_status"], table["ci_reason"], table["ci_tot_threshold_fc"] = ci.CI_OK, "", 555.0
        nodes = pd.DataFrame({"layer": [1, 1], "strip": [0, 0], "end": [0, 0], "node_index": [0, 1],
                              "tot_counts": [10.0, 20.0], "q_fc": [100.0, 200.0]})
        return table, nodes

    # the s table of one sample
    def scale_fit(self, table, nodes, calibration, train_csv, particle, s, chunksize=None):
        self.scale.append((particle, [Path(f).name for f in train_csv]))
        out = table[KEYS + ["dpm", "link", "ci_status"]].copy()
        out["training_particle"] = particle
        out["s_final"] = float(s["shrinkage_lambda"]) + {"pion": 0.0, "muon": 0.5}.get(particle, 0.25)
        out["s_source"] = ci.SOURCE_BORROWED if particle == "pion" else ci.SOURCE_DIRECT
        out.attrs["diagnostics"] = {"synthetic": True, "particle": particle}
        return out


# the calibration fitting functions replaced by the fake
@contextlib.contextmanager
def _patch_fits(fake: _FakeFits):
    with patch.object(ci, "calibrate_charge_injection", fake.calibrate), \
            patch.object(ci, "fit_scale_factors", fake.scale_fit):
        yield


# scan, channel map, the training runs' csv files and a calibration table under tmp
def _parameter_fixture(tmp: Path, stage: dict | None = None):
    (tmp / "scans").mkdir(exist_ok=True)
    (tmp / "scans" / "scan.csv").write_text("x\n1\n")
    (tmp / "channel_map.csv").write_text("DPM,LINK,CHAN,PLANE,STRIP,END\n0,0,3,1,0,0\n")
    runs = tmp / "data" / "analysis_files"
    for folder, names in (("Muon_4GeV", ["m.csv"]), ("Pion_4GeV", ["p1.csv", "p2.csv"])):
        (runs / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            (runs / folder / name).write_text("pf_event\n1\n")
    ctx = _ctx(tmp, stage_cfg=stage or {"scan_files": ["${ci_scan_dir}/scan.csv"]})
    calib = ctx.calibration_product()
    calib.parent.mkdir(parents=True, exist_ok=True)
    calib.write_text("layer,strip,end,pedestal,pedestal_range_with_20percent,mpv\n1,0,0,100,120,50\n")
    return ctx, pd.read_csv(calib), calib


# parameter cache samples and official copy
def test_parameter_cache_samples_and_official_copy():
    fake = _FakeFits()
    with tempfile.TemporaryDirectory() as tmp, _patch_fits(fake):
        tmp = Path(tmp)
        ctx, calibration, calib = _parameter_fixture(tmp)

        # obtain the parameters of a command line
        def obtain(*argv, stage_cfg=None):
            if stage_cfg is not None:
                ctx.cfg["charge_injection"] = stage_cfg
            return _quiet(ci.obtain_parameters, ctx, _settings_of(ctx, *argv), calibration, calib)

        t1, _, info1, written = obtain()
        assert info1["source"] == "fitted" and fake.ci == 1 and fake.scale == [("pion", ["p1.csv", "p2.csv"])]
        assert written == [ci.parameters_csv(ctx), ci.nodes_csv(ctx)]  # default units: parameters included
        for meta, members in ((ci.parameters_cache_files(ctx)[2], ci.parameters_cache_files(ctx)[:2]),
                              (ci.parameters_record(ctx), (ci.parameters_csv(ctx), ci.nodes_csv(ctx)))):
            assert io_utils.product_set_valid(meta, members=members)[0], meta
        assert ci.parameters_csv(ctx).read_bytes() == ci.parameters_cache_files(ctx)[0].read_bytes()
        record = io_utils.read_json(ci.parameters_record(ctx))
        assert record["scale_signature"]["training"]["pion"]["datasets"] == ["pion_4GeV"]
        assert record["s_choice"]["default"] == "pion" and "ci_channel_parameters" in record["columns"]
        _, _, info2, _ = obtain()
        assert info2["source"] == "cached" and fake.ci == 1 and len(fake.scale) == 1
        # an s setting refits only BLOCK 2; calib rewriting the same bytes keeps the cache (content identity)
        _, _, info3, _ = obtain(stage_cfg={"scan_files": ["${ci_scan_dir}/scan.csv"], "shrinkage_lambda": 7.0})
        assert info3["source"] == "cached CI map + new s fit" and fake.ci == 1 and len(fake.scale) == 2
        calib.write_bytes(calib.read_bytes())
        os.utime(calib, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
        assert obtain()[2]["source"] == "cached"
        # two samples: one fit each, s = the one that borrows least (muon), columns s_pion / s_muon
        table, _, info, _ = obtain("--scale-particle", "pion", "muon")
        assert fake.scale[-2:] == [("pion", ["p1.csv", "p2.csv"]), ("muon", ["m.csv"])]
        assert info["s_choice"]["default"] == "muon" and info["s_choice"]["n_borrowed"] == {"pion": 1, "muon": 0}
        assert table["s_final"].tolist() == [7.5] and table["s_muon"].tolist() == [7.5]
        assert table["s_pion"].tolist() == [7.0] and table["s_pion_source"].tolist() == [ci.SOURCE_BORROWED]
        assert ci.parameters_csv(ctx, "s_pion_muon").exists()
        # -only events with an s path: the cache is used, no official copy is written
        n_ci = fake.ci
        ci.parameters_csv(ctx).unlink()
        _, _, info, written = obtain("-only", "events")
        assert written == [] and not ci.parameters_csv(ctx).exists() and fake.ci == n_ci
        # a configured scan file that disappears: a clear error, never the stale cache
        (tmp / "scans" / "scan.csv").unlink()
        _expect(SystemExit, obtain, "Missing input file")


# interrupted parameter commit never mixes generations
def test_interrupted_parameter_commit_never_mixes_generations():
    fake = _FakeFits()
    with tempfile.TemporaryDirectory() as tmp, _patch_fits(fake):
        tmp = Path(tmp)
        ctx, calibration, calib = _parameter_fixture(tmp)

        # obtain the parameters of one lambda
        def obtain(lam):
            ctx.cfg["charge_injection"] = {"scan_files": ["${ci_scan_dir}/scan.csv"], "shrinkage_lambda": float(lam)}
            return _quiet(ci.obtain_parameters, ctx, _settings_of(ctx, "-only", "events"), calibration, calib)

        table, _, info, _ = obtain(5)
        assert info["source"] == "fitted" and float(table["s_final"][0]) == 5.0
        cache_csv, cache_nodes, cache_meta = ci.parameters_cache_files(ctx)
        real_write_json = io_utils.write_json

        # the cache record write fails
        def failing_record(obj, path):
            if Path(path) == cache_meta:
                raise OSError("injected manifest write failure")
            return real_write_json(obj, path)

        with patch.object(io_utils, "write_json", side_effect=failing_record):
            _expect(OSError, lambda: obtain(10), "injected")
        assert not cache_meta.exists()
        table, _, info, _ = obtain(5)
        assert float(table["s_final"][0]) == 5.0 and info["source"] == "fitted"  # refitted, never mixed
        text = cache_csv.read_text()
        cache_csv.write_text(text.replace("5.0", "9.0"))  # edited by hand: the member hash refuses it
        table, _, info, _ = obtain(5)
        assert info["source"] == "fitted" and float(table["s_final"][0]) == 5.0


# training runs any run is used as given
def test_training_runs_any_run_is_used_as_given():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        s = _settings_of(ctx)

        # training runs of one sample mapping
        def training(sample, mapping):
            ci._TRAINING_NOTES.clear()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                runs = ci._training_datasets(ctx, {**s, "scale_training_datasets": {
                    **s["scale_training_datasets"], sample: mapping}}, sample)
            return [d.key for d in runs], out.getvalue()

        assert training("muon", "muon 4") == (["muon_4GeV"], "")
        keys, text = training("mix", ["pion 4", "pion_4GeV", "muon 4"])
        assert keys == ["pion_4GeV", "muon_4GeV"] and "pion 4 named twice" in text, text
        keys, text = training("pion", "muon 4")
        assert keys == ["muon_4GeV"] and "the pion sample trains s on muon 4, not on pion data" in text, text
        _expect(SystemExit, lambda: training("pion", "pion 9"), "pion 9")
        ci._TRAINING_NOTES.clear()


# ----------------------------------------------------------------------------------------------
# linearity / resolution / e/h and the response rows
# ----------------------------------------------------------------------------------------------
# Each spectrum row carries its own point (sigma / sqrt(N), its particle's line), any run order.
def test_response_rows_join_the_points_by_run_key():
    s = _fit_settings(linearity_fits={"electron": None})
    runs = {k: (_ds(k, p, e), mu, sig) for k, p, e, mu, sig in (
        ("electron_0.3GeV", "electron", 0.3, 3.1, 0.31), ("electron_0.5GeV", "electron", 0.5, 5.2, 0.4),
        ("electron_1GeV", "electron", 1.0, 10.5, 0.6), ("electron_4GeV", "electron", 4.0, 38.0, 2.1),
        ("muon_4GeV", "muon", 4.0, 23.0, 3.4), ("pion_3GeV", "pion", 3.0, 30.0, 9.0))}
    columns = ("mu_stat_error_mip", "ratio_error", "linearity_fitted", "linearity_predicted_mip", "mu_mip")
    reference = None
    for order in (sorted(runs), sorted(runs, reverse=True), ["muon_4GeV", "electron_1GeV", "pion_3GeV",
                                                             "electron_0.3GeV", "electron_4GeV", "electron_0.5GeV"]):
        n_of = {k: 50 + 10 * i for i, k in enumerate(sorted(runs))}
        summaries = [_summary(runs[k][0], runs[k][1], runs[k][2], n=n_of[k]) for k in order]
        dep = {"s": ci.energy_dependence(summaries, "s", s)}
        rows = _rows_by_key(ci.energy_response_rows(summaries, dep, ["s"], s))
        table = {(k, c): rows[k].get(c) for k in rows for c in columns}
        reference = reference or table
        assert json_equal(table, reference), order
    for k in ("electron_0.3GeV", "electron_4GeV", "pion_3GeV"):
        assert math.isclose(rows[k]["mu_stat_error_mip"], runs[k][2] / math.sqrt(n_of[k]), rel_tol=1e-12)
    assert rows["electron_4GeV"]["linearity_fitted"] and not rows["pion_3GeV"]["linearity_fitted"]
    assert math.isnan(rows["pion_3GeV"]["linearity_predicted_mip"])  # no pion line: points only
    _expect(ValueError, lambda: ci.energy_dependence([summaries[0], summaries[0]], "s", s), "duplicate")


# json-equal (NaN == NaN)
def json_equal(a, b) -> bool:
    import json
    return json.dumps(ci._json_safe({str(k): v for k, v in a.items()}), sort_keys=True) == json.dumps(
        ci._json_safe({str(k): v for k, v in b.items()}), sort_keys=True)


# linearity per scale resolution and eh
def test_linearity_per_scale_resolution_and_eh():
    # Lei 2026-10-20: adc_only no fit, PureCI electron fit below 1 GeV, s every electron point; the pion points
    # are drawn, never fitted together with the electrons; -linfit replaces the defaults on every scale; the
    # resolution fits the electrons only; e/h only on the s scales, filled with an electron run at the pion energy,
    # hollow from the electron line otherwise.
    e = [_ds(k, "electron", E) for k, E in (("electron_0.3GeV", 0.3), ("electron_0.5GeV", 0.5), ("electron_1GeV", 1.0),
                                            ("electron_4GeV", 4.0))]
    pions = [_ds("pion_3GeV", "pion", 3.0), _ds("pion_4GeV", "pion", 4.0)]
    paths = ["adc_only", "pureci", "s", "s_muon"]
    summaries = ([_summary(d, m, 0.1 * m, n=400, paths=paths) for d, m in zip(e, (8.0, 14.3, 29.5, 102.0))]
                 + [_summary(d, m, 0.2 * m, n=400, paths=paths) for d, m in zip(pions, (60.0, 80.0))])
    s = _fit_settings()
    dep = {p: ci.energy_dependence(summaries, p, s) for p in paths}
    assert dep["adc_only"]["linearity"]["fits"] == {} and dep["adc_only"]["eh"] is None
    assert dep["pureci"]["linearity"]["fits"]["electron"]["fit_range"] == [0.0, 0.999] and dep["pureci"]["eh"] is None
    for p in ("s", "s_muon"):
        lin = dep[p]["linearity"]
        assert set(lin["fits"]) == {"electron"} and lin["fits"]["electron"]["fit_range"] is None
        assert lin["fitted"].tolist() == [True] * 4 + [False] * 2  # the pion points: drawn only
        eh = dep[p]["eh"]
        rows = {r["run"]: r for r in eh["rows"]}
        assert rows["pion_4GeV"]["filled"] and math.isclose(rows["pion_4GeV"]["e_over_h"], 102.0 / 80.0)
        assert not rows["pion_3GeV"]["filled"] and rows["pion_3GeV"]["electron_source"] == "electron line fit"
    res = dep["s"]["resolution"]
    assert res["fit_particle"] == "electron" and res["fitted"].tolist() == [True] * 4 + [False] * 2
    rows = ci.energy_response_rows(summaries, dep, paths, s)
    lin = [r for r in rows if r["row_type"] == "linearity"]
    assert [(r["path"], r.get("particle", ""), r["fit_kind"]) for r in lin] == [
        ("adc_only", "", "points_only"), ("pureci", "electron", "linear_weighted"),
        ("s", "electron", "linear_weighted"), ("s_muon", "electron", "linear_weighted")]
    assert next(r for r in lin if r["path"] == "pureci")["linearity_fit_range"] == "[0, 0.999]"
    assert next(r for r in lin if r["path"] == "s")["linearity_fit_range"] == "all points"
    eh_rows = [r for r in rows if r["row_type"] == "e_over_h"]
    assert {(r["path"], r["dataset"]) for r in eh_rows} == {(p, d.key) for p in ("s", "s_muon") for d in pions}
    assert all("error_scale" not in r for r in rows)
    # -linfit replaces the defaults on every scale: one line per particle, never electrons and pions together
    s = _fit_settings(linearity_fits={"electron": [0.3, 1.0], "pion": None}, normal_error=True)
    dep = {p: ci.energy_dependence(summaries, p, s) for p in paths}
    for p in paths:
        fits = dep[p]["linearity"]["fits"]
        assert set(fits) == {"electron", "pion"} and fits["electron"]["n_fitted"] == 3 and fits["pion"]["n_fitted"] == 2
    rows = ci.energy_response_rows(summaries, dep, paths, s)
    lin = [r for r in rows if r["row_type"] == "linearity"]
    assert len(lin) == 8 and all("slope_error_scaled_mip_per_gev" in r for r in lin)
    assert "resolution_a_error_scaled_mev" in next(r for r in rows if r["row_type"] == "resolution")
    spec = _rows_by_key(rows)
    assert spec["pion_3GeV"]["linearity_fitted"] and np.isfinite(spec["pion_3GeV"]["linearity_predicted_mip"])
    assert not spec["electron_4GeV"]["linearity_fitted"]  # outside the electron range [0.3, 1]


# ----------------------------------------------------------------------------------------------
# complete synthetic stage runs (CI parameters injected)
# ----------------------------------------------------------------------------------------------
# parameters of the synthetic detector: bars L2 S3 and L3 S3, both ends
def _stage_parameters():
    table = pd.DataFrame({"layer": [2, 2, 3, 3], "strip": [3] * 4, "end": [0, 1, 0, 1], "dpm": [0] * 4,
                          "link": [0] * 4, "chan": [1, 2, 3, 4], "ci_status": [ci.CI_OK] * 4,
                          "ci_tot_threshold_fc": [300.0] * 4, "ci_k_adc_per_fc": [1.5] * 4,
                          "mono_q_min_fc": [0.0] * 4, "mono_q_max_fc": [8000.0] * 4, "s_final": [0.8, 0.5, 0.7, 0.6],
                          "s_pion": [0.8, 0.5, 0.7, 0.6], "s_muon": [0.6, 0.6, 0.6, 0.6]})
    tot = np.array([0.0, 1000.0, 2000.0, 4000.0])
    nodes = pd.concat([pd.DataFrame({"layer": layer, "strip": 3, "end": e, "node_index": np.arange(4),
                                     "tot_counts": tot, "q_fc": 2.0 * tot}) for layer in (2, 3) for e in (0, 1)],
                      ignore_index=True)
    return table, nodes


# obtain_parameters / parameter_signatures replaced by the synthetic table
@contextlib.contextmanager
def _injected_parameters(version: dict):
    """obtain_parameters / parameter_signatures replaced by the synthetic table; version["v"] is its identity."""
    table, nodes = _stage_parameters()

    # synthetic signatures
    def signatures(ctx, s, calib, need_ci, need_scale):
        return {"kind": "chain", "inputs": {}, "ci_signature": None, "scale_signature": None,
                "identity": {"kind": "chain", "ci_signature_hash": f"ci{version['v']}",
                             "scale_signature_hash": f"s{version['v']}" if need_scale else None}}

    # synthetic parameters
    def obtain(ctx, s, calibration, path):
        ident = signatures(ctx, s, path, True, True)["identity"]
        return table, nodes, {"source": "synthetic", "identity": ident, **ident}, []

    with patch.object(ci, "obtain_parameters", obtain), patch.object(ci, "parameter_signatures", signatures):
        yield


BEAM_SCALES = {"electron_0.5GeV": 300, "electron_1GeV": 500, "electron_4GeV": 800, "muon_4GeV": 450,
               "pion_3GeV": 700, "pion_4GeV": 900}
BEAM_FILES = {"electron_4GeV": ("a.csv", "b.csv")}   # electron 4 is a run of two csv files


# synthetic beam csv files in their run folders: two bars per event
def _write_beams(ctx) -> None:
    rng = np.random.default_rng(7)
    for key, scale in BEAM_SCALES.items():
        ds = resolve_datasets(ctx.cfg, [key])[0]
        for name in BEAM_FILES.get(key, ("beam.csv",)):
            n = 1500
            pf = np.repeat(np.arange(1, n + 1), 2)
            adc = rng.normal(scale, 0.08 * scale, (2 * n, 2)).clip(130, 2000)
            rows = pd.DataFrame({"pf_event": pf, "layer": np.tile([2, 3], n), "strip": 3,
                                 "tot_end0": np.where(adc[:, 0] > 1023, 1000 + adc[:, 0], 0).round(),
                                 "adc_sum_end0": adc[:, 0].round(), "adc_max_end0": np.minimum(adc[:, 0], 1023).round(),
                                 "tot_end1": np.where(adc[:, 1] > 1023, 1000 + adc[:, 1], 0).round(),
                                 "adc_sum_end1": adc[:, 1].round(),
                                 "adc_max_end1": np.minimum(adc[:, 1], 1023).round()})
            path = ctx.run_dir(ds, mode="fullset") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(path, index=False)


# a stage context with beams and the calibration table it reads
def _stage_fixture(tmp: Path, make_plots=False, outputs: tuple | None = (), test: bool = False):
    ctx = _ctx(tmp, test=test, make_plots=make_plots, outputs=outputs)
    _write_beams(ctx)
    calib = ctx.calibration_product(mode="fullset")
    calib.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layer": [2, 2, 3, 3], "strip": [3, 3, 3, 3], "end": [0, 1, 0, 1], "pedestal": 100.0,
                  "pedestal_range_with_20percent": 120.0, "mpv": 50.0}).to_csv(calib, index=False)
    return ctx, calib


# temp / staged files left anywhere
def _leftover_temps(root: Path) -> list:
    return [p for p in root.rglob("*") if p.is_file() and (".tmp" in p.name or ".old-" in p.name)]


# synthetic stage one product per csv and the plot cache
def test_synthetic_stage_one_product_per_csv_and_the_plot_cache():
    # A complete run: one event csv per input csv in a sub-folder per run, the per-event / per-bar / per-end
    # arrays of every computed path in cache/, the figures in results/ (no spectra pdf), the checks only on
    # request; the bar records sum to the event energies.
    version = {"v": 1}
    with tempfile.TemporaryDirectory() as tmp, _injected_parameters(version):
        tmp = Path(tmp)
        ctx, calib = _stage_fixture(tmp, make_plots=True)
        ds = resolve_datasets(ctx.cfg, ["electron 4", "electron 1", "electron 0.5", "pion 3", "pion 4", "muon 4"])
        log = _run(ctx, ds, "--output-tag", "syn")
        dat, res = tmp / "data" / "charge_injection", tmp / "results" / "fullset" / "charge_injection"
        cache = tmp / "cache" / "charge_injection" / "events_syn"
        assert sorted(p.relative_to(dat).as_posix() for p in dat.rglob("*.csv")) == sorted(
            [f"Electron_4GeV/{n}_event_energies_syn.csv" for n in ("a", "b")]
            + [f"{f}/beam_event_energies_syn.csv" for f in ("Electron_1GeV", "Electron_0.5GeV", "Pion_3GeV",
                                                             "Pion_4GeV", "Muon_4GeV")])
        assert sorted(p.name for p in res.iterdir()) == ["e_over_h_syn.pdf", "energy_linearity_syn.pdf",
                                                         "energy_resolution_syn.pdf"]
        assert not list(tmp.rglob("*spectra*")) and not list(tmp.rglob("bar_energies*"))
        assert sorted(p.name for p in (cache / "Electron_4GeV").iterdir()) == [
            "a.json", "a.npz", "a_bars.npy", "b.json", "b.npz", "b_bars.npy"]
        a_csv = tmp / "data" / "analysis_files" / "Electron_4GeV" / "a.csv"
        e4 = ds[0]
        events = pd.read_csv(ci.event_energies_csv(ctx, e4, a_csv, "syn"))
        assert list(events.columns) == ci.event_columns(list(ci.PATHS))
        assert set(events["event_source"]) == {"Electron_4GeV/a.csv"} and events["pf_event"].iloc[0] == 1
        npz, npy, meta = ci.event_cache_files(ctx, e4, a_csv, "syn")
        arrays = ci.load_event_arrays(npz)
        bars = ci.load_bar_records(npy, mmap=False)  # a memory map would lock the file (Windows)
        assert set(arrays["energy"]) == set(ci.PATHS) and np.array_equal(arrays["pf_event"], events["pf_event"])
        frame = pd.DataFrame({"pf_event": bars["pf_event"], "s": bars["bar_energy_s_mip"].astype(float)})
        sums = frame.groupby("pf_event")["s"].sum().reindex(events["pf_event"]).to_numpy()
        assert np.allclose(sums, events["energy_s_mip"], rtol=1e-5)  # float32 bar records
        assert np.isfinite(bars["end0_energy_pureci_mip"]).any() and set(bars["layer"]) == {2, 3}
        record = io_utils.read_json(meta)
        assert record["event_source"] == "Electron_4GeV/a.csv" and record["n_bars"] == len(bars)
        assert io_utils.product_set_valid(meta, members=[npz, npy])[0]
        rec = ci.read_provenance(ci.event_energies_csv(ctx, e4, a_csv, "syn"))
        assert rec["event_source"] == "Electron_4GeV/a.csv" and rec["dataset"] == "electron_4GeV"
        assert rec["columns"] == list(events.columns) and rec["rows"] == len(events)
        assert rec["settings"]["bad_bar_rule"] == "used_ends" and rec["ci_parameters"]["source"] == "synthetic"
        assert "ci: units parameters, events, linearity, resolution, eh;" in log
        assert "in 2 csv" in log and not _leftover_temps(tmp)
        # the checks only on request: the response table and the quality scan (electron runs)
        _run(ctx, ds, "--output-tag", "syn", "-only", "checks")
        response = pd.read_csv(res / "energy_response_syn.csv")
        spec = response[(response["row_type"] == "spectrum") & (response["path"] == "s")].set_index("dataset")
        assert spec.loc["electron_4GeV", "n_csv_files"] == 2 and spec.loc["electron_4GeV", "n_events_in_file"] == 3000
        for key in ("electron_0.5GeV", "electron_1GeV", "electron_4GeV"):
            row = spec.loc[key]
            assert row["fit_ok"] and math.isclose(row["mu_stat_error_mip"],
                                                  row["sigma_mip"] / math.sqrt(row["n_events_selected"]), rel_tol=1e-9)
        assert set(response["row_type"]) == {"spectrum", "linearity", "resolution", "e_over_h"}
        eh = response[response["row_type"] == "e_over_h"].set_index("dataset")
        assert bool(eh.loc["pion_4GeV", "eh_filled"]) and not bool(eh.loc["pion_3GeV", "eh_filled"])
        scan = pd.read_csv(res / "quality_threshold_scan_syn.csv")
        assert set(scan["dataset"]) == {"electron_0.5GeV", "electron_1GeV", "electron_4GeV"}
        assert "rule_selected" not in scan and "guard_pass" not in scan
        # -only events: the event csv only, no figure; several samples: one s_<sample> column each
        ctx.make_plots = False
        pi4 = ds[4]
        _run(ctx, [pi4], "-only", "events", "--scale-particle", "pion", "muon")
        events = pd.read_csv(ci.event_energies_csv(ctx, pi4, ctx.run_csvs(pi4)[0], "s_pion_muon"))
        assert list(events.columns) == ci.event_columns(["adc_only", "pureci", "s", "s_pion", "s_muon"])
        assert np.allclose(events["energy_s_mip"], events["energy_s_pion_mip"])  # the injected s = s_pion
        assert not (events["energy_s_mip"] == events["energy_s_muon_mip"]).all()
        assert not list((tmp / "results" / "fullset" / "charge_injection").glob("*s_pion_muon*"))
        assert not _leftover_temps(tmp)


# from saved redoes the units from the stored event arrays
def test_from_saved_redoes_the_units_from_the_stored_event_arrays():
    # -from-saved reads no beam csv and no CI parameter; with unchanged settings it reproduces the event csv bit
    # for bit; it applies new response / selection settings; it refuses arrays of other inputs, parameters or
    # reconstruction settings with the rebuild command.
    version = {"v": 1}
    with tempfile.TemporaryDirectory() as tmp, _injected_parameters(version):
        tmp = Path(tmp)
        ctx, calib = _stage_fixture(tmp)
        ds = resolve_datasets(ctx.cfg, ["electron 0.5", "electron 1", "electron 4", "pion 4"])
        _run(ctx, ds, "-only", "events,checks")
        res = tmp / "results" / "fullset" / "charge_injection"
        a_csv = tmp / "data" / "analysis_files" / "Electron_4GeV" / "a.csv"
        event_csv = ci.event_energies_csv(ctx, ds[2], a_csv)
        normal_events = event_csv.read_bytes()
        normal_response = pd.read_csv(res / "energy_response.csv")

        # a beam csv must never be read
        def no_beam_read(*a, **k):
            raise AssertionError("-from-saved must not read a beam csv")

        with patch.object(ci, "iter_csv", side_effect=no_beam_read), \
                patch.object(ci, "load_calibration", side_effect=no_beam_read):
            _run(ctx, ds, "--from-saved", "-only", "events,checks")
            assert event_csv.read_bytes() == normal_events
            pd.testing.assert_frame_equal(pd.read_csv(res / "energy_response.csv"), normal_response)
            record = ci.read_provenance(event_csv)
            assert record["from_saved"] is True and record["reconstruction_run_id"]
            # another linearity fit / error mode
            _run(ctx, ds, "--from-saved", "-only", "response", "-linfit", "electron", "pion", "-normalerror")
            response = pd.read_csv(res / "energy_response.csv")
            lin = response[response["row_type"] == "linearity"]
            assert set(lin["particle"]) == {"electron", "pion"} and len(lin) == 6
            assert "slope_error_scaled_mip_per_gev" in response.columns
            # a subset of the stored paths, another quality cut
            _run(ctx, ds[:2], "--from-saved", "-s", "-only", "events")
            _run(ctx, ds, "--from-saved", "--quality-threshold", "0.0", "-only", "events")
            assert pd.read_csv(event_csv)["selected"].sum() <= pd.read_csv(io.BytesIO(normal_events))["selected"].sum()
            # refusals: other reconstruction settings, other parameters, other calibration content, other beam csv
            msg = _expect(ProductSetError, lambda: _run(ctx, ds, "--from-saved", "-bad-bar-rule", "used_ends_strict"),
                          "reconstruction_settings", "Rebuild it with")
            assert "-bad-bar-rule used_ends_strict" in msg.split("Rebuild it with: ", 1)[1]
            version["v"] = 2
            _expect(ProductSetError, lambda: _run(ctx, ds, "--from-saved"), "'parameters' differs")
            version["v"] = 1
            original = calib.read_text()
            calib.write_text(original.replace("50.0", "51.0"))
            _expect(ProductSetError, lambda: _run(ctx, ds, "--from-saved"), "'calibration' differs")
            calib.write_text(original)
            st = a_csv.stat()
            os.utime(a_csv, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
            _expect(ProductSetError, lambda: _run(ctx, ds, "--from-saved"), "'beam_csv' differs")
            os.utime(a_csv, ns=(st.st_atime_ns, st.st_mtime_ns))
            _run(ctx, ds, "--from-saved", "-only", "events")  # back to the original inputs: accepted again
            _expect(SettingsError, lambda: _run(ctx, ds, "--from-saved", "-only", "parameters"), "makes nothing")
        # arrays without the s path cannot serve a run that needs it
        _run(ctx, ds[:1], "--adc-only", "-pureci", "--selection-path", "adc_only", "--output-tag", "adc")
        _run(ctx, ds[:1], "--from-saved", "-pureci", "--selection-path", "adc_only", "--output-tag", "adc")
        _expect(ProductSetError, lambda: _run(ctx, ds[:1], "--from-saved", "-s", "--selection-path", "adc_only",
                                              "--output-tag", "adc"), "'parameters' differs")
        # an --output-tag without stored arrays of its own reads the untagged arrays
        log = _run(ctx, ds, "--from-saved", "--output-tag", "q", "-only", "events")
        assert "no stored event arrays of Electron_0.5GeV/beam.csv under --output-tag q" in log
        assert ci.event_energies_csv(ctx, ds[2], a_csv, "q").read_bytes() == normal_events
        plan = _plan(ctx, ds, "--from-saved", "--output-tag", "fresh", "-only", "events")
        items = [p for p in plan if p["role"] == "cache"]
        assert len(items) == 5 and all(io_utils.product_set_valid(p["path"], p["expected"], p["members"])[0]
                                       for p in items)
        assert not _leftover_temps(tmp)


# final results follow the units and output names
def test_final_results_follow_the_units_and_output_names():
    version = {"v": 1}
    with tempfile.TemporaryDirectory() as tmp, _injected_parameters(version):
        tmp = Path(tmp)
        ctx, _ = _stage_fixture(tmp, make_plots=True, outputs=None)
        res = tmp / "results" / "fullset" / "charge_injection"
        e1, mu4, pi4 = resolve_datasets(ctx.cfg, ["electron 1", "muon 4", "pion 4"])

        # slots of a command line and runs
        def slots(argv, datasets):
            s = _settings_of(ctx, *argv)
            return [(x.key, x.kind, x.default.name) for x in ci.result_slots(ctx, s, datasets, s["units"])]

        assert slots((), [e1, pi4]) == [("linearity", "pdf", "energy_linearity.pdf"),
                                        ("resolution", "pdf", "energy_resolution.pdf"),
                                        ("eh", "pdf", "e_over_h.pdf")]
        assert slots(("-only", "all", "--output-tag", "q"), [e1, pi4]) == [
            ("response", "csv", "energy_response_q.csv"), ("quality_scan", "csv", "quality_threshold_scan_q.csv"),
            ("linearity", "pdf", "energy_linearity_q.pdf"), ("resolution", "pdf", "energy_resolution_q.pdf"),
            ("eh", "pdf", "e_over_h_q.pdf")]
        assert [k for k, _, _ in slots(("-only", "all"), [mu4])] == ["response", "linearity", "resolution"]
        assert [k for k, _, _ in slots(("-pureci", "-only", "figures"), [e1, pi4])] == ["linearity", "resolution"]
        assert ci.unit_skip_reason("eh", _settings_of(ctx), [e1]) == "e/h needs a pion run: none selected"
        # full set without --output: nothing in results/, the scratch copies in cache/.../unsaved/
        log = _run(ctx, [e1, pi4], "-only", "response,eh")
        unsaved = tmp / "cache" / "charge_injection" / "unsaved"
        assert not res.exists() and sorted(p.name for p in unsaved.iterdir()) == ["e_over_h.pdf", "energy_response.csv"]
        assert "final results not saved (no --output)" in log
        # -output by result name: KEY.ext, the default file name, or KEY=FILE; the others stay unsaved
        ctx.outputs = ["eh.pdf", "response=my_table.csv"]
        _run(ctx, [e1, pi4], "-only", "checks,figures")
        assert sorted(p.name for p in res.iterdir()) == ["eh.pdf", "my_table.csv"]
        assert ci.result_file(ctx, "response") == res / "my_table.csv"
        ctx.outputs = ["quality_threshold_scan.csv"]
        _run(ctx, [e1, pi4], "-only", "checks")
        assert (res / "quality_threshold_scan.csv").exists() and not (res / "energy_response.csv").exists()
        # -test without --output: results/test/test.csv, test.pdf, ...
        ctx.outputs = None
        ctx.test = True
        _run(ctx, [e1, pi4], "-only", "response,linearity")
        assert sorted(p.name for p in (tmp / "results" / "test").iterdir() if p.is_file()) == ["test.csv", "test.pdf"]
        assert not _leftover_temps(tmp)


# dry run plan lists the per csv products
def test_dry_run_plan_lists_the_per_csv_products():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx, _ = _stage_fixture(tmp, make_plots=True)
        (tmp / "scans").mkdir()
        (tmp / "scans" / "scan.csv").write_text("x\n")
        ctx.cfg["charge_injection"] = {"scan_files": ["${ci_scan_dir}/scan.csv"]}
        ctx.dry_run = True
        ds = resolve_datasets(ctx.cfg, ["electron 4"])
        plan = _plan(ctx, ds, "--scale-particle", "pion", "muon")
        outs = [Path(x["path"]).relative_to(tmp).as_posix() for x in plan if x["role"] == "out"]
        assert "data/charge_injection/Electron_4GeV/a_event_energies_s_pion_muon.csv" in outs
        assert "cache/charge_injection/events_s_pion_muon/Electron_4GeV/b_bars.npy" in outs
        assert "calibration/charge_injection/ci_channel_parameters_s_pion_muon.csv" in outs
        assert "results/fullset/charge_injection/energy_linearity_s_pion_muon.pdf" in outs
        ins = [Path(x["path"]).relative_to(tmp).as_posix() for x in plan if x["role"] == "in"]
        assert ins.count("data/analysis_files/Pion_4GeV/beam.csv") == 1
        assert "data/analysis_files/Muon_4GeV/beam.csv" in ins
        assert "data/analysis_files/Electron_4GeV/a.csv" in ins and "scans/scan.csv" in ins
        plan = _plan(ctx, ds, "-only", "parameters")
        assert not [x for x in plan if "Electron_4GeV" in str(x["path"]) or Path(x["path"]).suffix == ".pdf"]
        ctx.write_json = False
        io_utils.set_json_records(False, [tmp / "data", tmp / "calibration", tmp / "results"], [tmp / "cache"])
        try:
            plan = _plan(ctx, ds, "-only", "events,parameters")
            assert not [x for x in plan if str(x["path"]).endswith(".provenance.json")]
            assert [x for x in plan if Path(x["path"]).name == "parameters.json"]  # the cache record stays
        finally:
            io_utils.set_json_records(True)


# rebuild command repeats the run flags
def test_rebuild_command_repeats_the_run_flags():
    assert ci._flag_value(7.0) == "7" and ci._flag_value([0.5, 1.0]) == "0.5 1" and ci._flag_value("a b") == '"a b"'
    assert ci.range_flag_text("-fit", [(5.0, 10.0), (60.0, 110.0)]) == "-fit[5,10],[60,110]"
    assert ci.range_flag_text("-bin", [30, 40]) == "-bin 30,40"
    assert ci.range_flag_text("-linfit", {"electron": [0.3, 1.0], "pion": None}) == "-linfit electron[0.3,1] pion"
    assert ci.range_flag_text("-linfit", {}) == "-linfit none"
    assert ci.range_flag_text("-resfit", [0.0, 100.0]) == "-resfit[0,100]"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx(tmp, test=True, clean_input=True)
        ctx.cfg["_config_file"] = tmp / "my.yml"
        ctx.n_test_events = 2000
        ds = resolve_datasets(ctx.cfg, ["electron 0.5", "electron 4"])
        s = _settings_of(ctx, "--from-saved", "-s", "-shrinkage", "7", "-scale-particle", "pion", "muon",
                         "-quality-threshold", "0.1", "-output-tag", "t7", "-fit[5,10],[60,110]", "-bin", "30,40",
                         "-linfit", "electron[0.3,1]", "-resfit[0.3,1]", "-normalerror")
        cmd = ci._rebuild_command(ctx, s, ds)
        assert cmd.startswith(".\\runner.ps1 ci --electron 0.5 --electron 4 "
                              f"-config {ci._flag_value(tmp / 'my.yml')} -test -test-events 2000 -clean-input -s "
                              "-scale-particle pion muon -shrinkage 7 -output-tag t7 -fit[5,10],[60,110] -bin 30,40 "
                              "-linfit electron[0.3,1] -resfit[0.3,1]   (then repeat"), cmd
        assert "-quality-threshold" not in cmd and "-normalerror" not in cmd
        flags = cmd.split("   (then")[0].split(" -output-tag t7 ")[1].split()
        again = _settings_of(_ctx(tmp), *flags)["energy_response"]
        assert {k: again[k] for k in er.RANGE_SETTING_KEYS} == {k: s["energy_response"][k]
                                                                 for k in er.RANGE_SETTING_KEYS}
        assert _settings_of(ctx, "--from-saved", "-linfit", "none")["_rebuild_range_flags"] == ["-linfit none"]
        plain = _ctx(tmp)
        plain.cfg["_config_file"] = ci.DEFAULT_CONFIG
        assert ci._mode_flags(plain) == ""
        assert ci._rebuild_command(plain, _settings_of(plain, "--from-saved"), ds).startswith(
            ".\\runner.ps1 ci --electron 0.5 --electron 4   (then repeat")


# the pid of a process that has ended
def _dead_pid() -> int:
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# bar record temps follow the unique temp pattern and orphans are swept
def test_bar_record_temps_follow_the_unique_temp_pattern_and_orphans_are_swept():
    version = {"v": 1}
    with tempfile.TemporaryDirectory() as tmp, _injected_parameters(version):
        tmp = Path(tmp)
        ctx, _ = _stage_fixture(tmp)
        ds = resolve_datasets(ctx.cfg, ["electron 4"])
        cache = tmp / "cache" / "charge_injection"
        cache.mkdir(parents=True)
        dead = _dead_pid()
        orphan, recent = (cache / f"bar_records.Electron_4GeV.a.tmp-{dead}-0123abcd.bin",
                          cache / f"bar_records.Electron_4GeV.b.tmp-{dead}-89abcdef.bin")
        for f in (orphan, recent):
            f.write_bytes(b"x" * 64)
        old = time.time() - 2 * io_utils.ORPHAN_MIN_AGE_S
        os.utime(orphan, (old, old))
        seen = []
        real = ci.write_npy_records

        # remember the raw record files
        def spy(dst, raw, dtype, n):
            seen.append(Path(raw))
            assert Path(raw).exists()
            return real(dst, raw, dtype, n)

        with patch.object(ci, "write_npy_records", side_effect=spy):
            _run(ctx, ds, "-only", "events")
        assert not orphan.exists() and recent.exists()
        assert [p.parent for p in seen] == [cache, cache]
        for p, stem in zip(seen, ("a", "b")):
            m = io_utils._UNIQUE_TEMP.search(p.name)
            assert m and int(m.group(1)) == os.getpid() and p.name.startswith(f"bar_records.Electron_4GeV.{stem}.tmp-")
            assert not p.exists()
        recent.unlink()
        assert not _leftover_temps(tmp)
        # a failing csv leaves no temp file and the old products intact
        e4_dir = tmp / "data" / "charge_injection" / "Electron_4GeV"
        before = {p.name: p.read_bytes() for p in e4_dir.iterdir()}
        bad = tmp / "data" / "analysis_files" / "Electron_4GeV" / "b.csv"
        good = bad.read_text()
        bad.write_text(good.replace("\n1500,3,3,", "\n1500,25,3,", 1))
        _expect(ChannelKeyError, lambda: _run(ctx, ds, "-only", "events"), "outside the detector")
        assert not _leftover_temps(tmp)
        assert {p.name: p.read_bytes() for p in e4_dir.iterdir() if p.name.startswith("b")} == {
            k: v for k, v in before.items() if k.startswith("b")}


# ----------------------------------------------------------------------------------------------
# Lei 2026-10-02: single Gaussians by default, --fit / --bin user ranges, failed fits = points only
# ----------------------------------------------------------------------------------------------
# a run summary with the spectrum fit of s (one path)
def _page_summary(ds: Dataset, values, s: dict, path: str = "s") -> dict:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    return {"ds": ds, "n_csv": 1, "n_events_in_file": n, "n_reconstructed": n, "n_selected": n,
            "n_bars_selected": 10 * n, "values": {path: values},
            "fits": {path: ci.spectrum_fit(values, ds.particle, s)}}


# default spectrum fits are single gaussians
def test_default_spectrum_fits_are_single_gaussians():
    s = _fit_settings()
    rng = np.random.default_rng(4)
    pion = np.r_[rng.normal(20.0, 3.0, 30000), rng.normal(60.0, 8.0, 20000)]  # two peaks: still ONE Gaussian
    fit = ci.spectrum_fit(pion, "pion", s)["fit"]
    assert fit["kind"] == "pion_core_gauss" and fit["ok"] and abs(fit["mu"] - 20.0) < 0.3, fit
    muon_like = ci.fit_muon_core(pion, s)
    pion_like = ci.fit_pion_core(pion, {**s, "pion_fit_bins": s["muon_fit_bins"]})
    assert {k: v for k, v in pion_like.items() if k != "kind"} == {k: v for k, v in muon_like.items() if k != "kind"}
    m = ci.spectrum_fit(rng.normal(23.0, 3.4, 20000), "muon", s)["fit"]
    assert m["kind"] == "muon_core_gauss" and m["ok"] and abs(m["mu"] - 23.0) < 0.3
    e = ci.spectrum_fit(rng.normal(38.0, 2.1, 20000), "electron", s)["fit"]
    assert e["kind"] == "two_pass" and e["ok"] and abs(e["mu"] - 38.0) < 0.2
    for values in (np.zeros(0), np.array([5.0]), np.full(1000, 7.0)):
        bad = ci.fit_pion_core(values, s)
        assert not bad["ok"] and bad["status"].startswith("fit_failed") and math.isnan(bad["mu"])
    with patch.object(ci, "curve_fit", side_effect=RuntimeError("simulated: no convergence")):
        broken = ci.fit_muon_core(pion, s)
    assert broken["status"] == "fit_failed: RuntimeError: simulated: no convergence" and broken["bin_width"] > 0
    # the spectrum page for the plots stage (ci itself draws no spectra)
    import matplotlib.pyplot as plt

    for values, particle, fit_ in ((pion, "pion", fit), (np.full(1000, 7.0), "muon", bad)):
        f, ax = plt.subplots()
        ci.draw_spectrum(ax, values, fit_, particle, "s", s, title="t", counts_text="1 of 1 events")
        labels = [ln.get_label() for ln in ax.lines if not ln.get_label().startswith("_")]
        assert labels == (["Gaussian fit"] if fit_["ok"] else []), labels
        assert any("single Gaussian core fit" in t.get_text() for t in ax.texts)
        plt.close(f)


# fit ranges single and double gaussian rows and failures
def test_fit_ranges_single_and_double_gaussian_rows_and_failures():
    rng = np.random.default_rng(11)
    values = rng.normal(20.0, 2.5, 40000)
    s = _fit_settings(fit_ranges=[[15, 25]])
    for particle in ("electron", "muon", "pion"):
        fit = ci.spectrum_fit(values, particle, s)["fit"]
        assert fit["kind"] == "range_gauss" and fit["ok"] and abs(fit["mu"] - 20.0) < 0.1, (particle, fit["status"])
        row = ci.energy_response_rows([_page_summary(_ds("x", particle, 1.0), values, s)], {}, ["s"], s)[0]
        assert row["fit_kind"] == "range_gauss" and row["fit_ranges"] == "[15, 25]" and row["fit_bins"] == ""
        assert (row["fit_low_mip"], row["fit_high_mip"]) == (15.0, 25.0) and math.isnan(row["mu_low_mip"])
    b = ci.spectrum_fit(values, "electron", _fit_settings(fit_ranges=[[15, 25]], fit_bins=[40]))["fit"]
    assert b["ok"] and ci._fit_bin_width(b) == 0.25 and ci._fit_brief(b).endswith("(--fit [15, 25])")
    pion = np.r_[rng.normal(22.0, 3.0, 30000), rng.normal(80.0, 10.0, 30000)]
    s2 = _fit_settings(fit_ranges=[[15, 32], [50, 120]], fit_bins=[30, 40])
    row = ci.energy_response_rows([_page_summary(_ds("pion_4GeV", "pion", 4.0), pion, s2)], {}, ["s"], s2)[0]
    assert row["fit_kind"] == "range_double_gauss" and row["fit_bins"] == "30,40"
    assert abs(row["mu_low_mip"] - 22.0) < 0.3 and abs(row["mu_high_mip"] - 80.0) < 1.0 and math.isnan(row["mu_mip"])
    assert abs(row["area_low_events"] - 30000) < 900 and abs(row["area_high_events"] - 30000) < 900
    assert ci._fit_settings_text(s2["energy_response"]).startswith(
        "spectrum fits --fit [15, 32], [50, 120] MIP (double Gaussian, --bin 30,40)")
    assert ci._fit_settings_text(_fit_settings()["energy_response"]) == (
        "automatic single-Gaussian spectrum fits, linearity fits default per scale (adc_only none, PureCI "
        "electron below 1 GeV, s electron)")
    nothing = _fit_settings(fit_ranges=[[200, 300]])
    summ = _page_summary(_ds("muon_4GeV", "muon", 4.0), values, nothing)
    row = ci.energy_response_rows([summ], {}, ["s"], nothing)[0]
    assert row["fit_ok"] is False and row["fit_status"].startswith("fit_failed") and math.isnan(row["mu_mip"])


# fit replaces linearity source peak scan for the points
def test_fit_replaces_linearity_source_peak_scan_for_the_points():
    rng = np.random.default_rng(21)
    e = [_ds(k, "electron", E) for k, E in (("electron_0.3GeV", 0.3), ("electron_0.5GeV", 0.5), ("electron_1GeV", 1.0))]
    spectra = {d.key: np.r_[rng.normal(20.0, 2.0, 4000), rng.normal(40.0, 3.0, 8000)] for d in e}

    # dependence and rows of one linearity source and fit settings
    def response(source: str, **rs):
        s = _fit_settings(**rs)
        s["linearity_source"] = source
        summaries = [_page_summary(d, spectra[d.key], s) for d in e]
        dep = ci.energy_dependence(summaries, "s", s)
        return s, summaries, dep, ci.energy_response_rows(summaries, {"s": dep}, ["s"], s)

    s, summaries, dep, rows = response("peak_scan")
    assert all(abs(m - 40.0) < 0.5 for m in dep["linearity"]["mu"])  # the taller peak
    assert {r["linearity_source"] for r in rows if r["row_type"] != "spectrum"} == {"peak_scan"}
    s, summaries, dep, rows = response("peak_scan", fit_ranges=[[17, 23]])
    assert all(abs(m - 20.0) < 0.2 for m in dep["linearity"]["mu"])
    assert {r["linearity_source"] for r in rows if r["row_type"] != "spectrum"} == {"range_fit"}
    assert ci.linearity_source_note(s) and "(replaces linearity_source peak_scan)" in ci.points_source_text(s)
    s, summaries, dep, rows = response("peak_scan", fit_ranges=[[15, 25], [30, 50]])
    assert dep["no_point_reasons"] == ["double Gaussian (two --fit ranges): no single mu"] * 3
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("linearity", "resolution"):
            ci.plot_energy_dependence(Path(tmp) / f"{kind}.pdf", kind, {"s": dep}, ["s"], s, "")
            assert (Path(tmp) / f"{kind}.pdf").stat().st_size > 0
    assert ci._no_point_reason({"kind": "two_pass", "ok": False, "status": "fit_failed: too few bins"}) == \
        "spectrum fit failed: too few bins"


# fit range flags reach the settings
def test_fit_range_flags_reach_the_settings():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        s = _settings_of(ctx, "-fit[15", "32]", "[50", "120]", "-bin", "30", "40", "-linfit", "electron[0.3", "1]",
                         "-resfit[0.3", "4]")
        rs = s["energy_response"]
        assert rs["fit_ranges"] == [[15.0, 32.0], [50.0, 120.0]] and rs["fit_bins"] == [30, 40]
        assert rs["linearity_fits"] == {"electron": [0.3, 1.0]} and rs["resolution_fit_range"] == [0.3, 4.0]
        assert s["variant_tag"] == ""
        for argv, words in ((("-fit[5,10],[60,110]", "-bin", "30"), "one value per fit range"),
                            (("-bin", "30"), "--bin needs --fit")):
            _expect(SettingsError, lambda: _settings_of(ctx, *argv), words)
    for argv, words in ((("-fit[5,10]", "-fit[20,30]"), "--fit given twice"), (("-fit",), "--fit needs a range"),
                        (("-fit[5,20],[10,30]",), "overlap"), (("-linfit", "[0,1]"), "particle")):
        message = _parse_error(*argv)
        assert words in message, (argv, message)
    p = argparse.ArgumentParser(allow_abbrev=False)
    ci.add_arguments(p)
    text = p.format_help()
    assert "--linfit" in text and "--only" in text and "--pureci" in text


# -no-json: the event csv without its provenance json; the cache records stay and -from-saved reads them
def test_no_json_keeps_only_the_cache_records():
    with tempfile.TemporaryDirectory() as tmp, _injected_parameters({"v": 1}):
        tmp = Path(tmp)
        ctx, _ = _stage_fixture(tmp)
        ctx.write_json = False
        io_utils.set_json_records(False, [tmp / "data", tmp / "calibration", tmp / "results"], [tmp / "cache"])
        try:
            e4 = resolve_datasets(ctx.cfg, ["electron 4"])[0]
            _run(ctx, [e4], "-only", "events")
            csvs = ctx.run_csvs(e4)
            made = [ci.event_energies_csv(ctx, e4, c) for c in csvs]
            assert len(made) == 2 and all(p.exists() for p in made)
            assert not list((tmp / "data").rglob("*.provenance.json"))
            assert all(ci.event_cache_files(ctx, e4, c)[2].exists() for c in csvs)   # cache records stay
            first = made[0].read_bytes()
            _run(ctx, [e4], "-only", "events", "-from-saved")
            assert made[0].read_bytes() == first
        finally:
            io_utils.set_json_records(True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_all  # noqa: E402

    sys.exit(run_all.main(["charge_injection"]))
