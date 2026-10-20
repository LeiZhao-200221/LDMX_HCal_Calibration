"""Tests of ldmx_hcal.energy.tot_adc_peak (stage "adcpeak"; synthetic data only, about a minute).

    python -X utf8 src/tests/run_all.py tot_adc_peak

Covered:
  routing    line table -> calibration/[test/]tot_adc_peak; fit pdf, response csv and pdfs -> results/<mode>/
             tot_adc_peak; fit bins and the per-csv events / bar values -> cache/[test/]tot_adc_peak/<run>/; nothing in
             data/ (Lei 2026-10-20: the event / bar values are cache only)
  -only      the five units (line_table, line_fits, energy_response, energy_spectra, linearity_resolution) and the
             groups line / response / csv / plots / all; -no-plots drops the pdfs ('nothing to make' when none is
             left); -bar-csv and the response range flags need a response unit
  settings   unknown yml keys, bad values and command-line-only keys stop before any IO; --output-tag is file-name
             safe and never an automatic variant tag (none / mean_fallback)
  line fit   the bins cover the whole fit_adc_window (hits with adc_max 1018-1021 are fitted); the bins of a run are
             summed over its csv files; the display sample is bounded, reproducible, chunk independent, follows the
             documented key formula and keeps its exact set with the key bound (also with hash ties); the fit does not
             depend on the sample
  table      the line table is checked at every entry (duplicates, bad keys, NaN on an ok line); read back exactly
  events     Arbitrary Energy per csv; streamed bar values == in-memory bars with and without the event cut (second
             pass only on selected rows); the event accumulator is bit-identical to the two-stage reduction; the
             rail-aware end merge (provided by Erik Wallin)
  response   spectrum per run over all its csv files (two_pass / peak_scan / -fit single and double Gaussian, failed
             -fit = points only); linearity: every particle drawn, no line by default, -linfit <particle>[a,b] one line
             per particle; resolution fitted to the electron points only; energy_response.csv columns
  -from-saved  bit-identical response; every dependency verified (reconstruction settings, calibration and line
             table content, input csv, a missing or modified saved copy) with the rebuild command; the fit pdf and
             line table from the saved bins without reading the csv; -output-tag reads the untagged copy
  --output   names matched by result (default file name, key + extension, KEY=FILE); full set without --output ->
             the scratch copy only; -test without --output -> results/test/test.pdf, test.csv, ...

Author: Lei
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import fitting  # noqa: E402
from ldmx_hcal.common import geometry as geo  # noqa: E402
from ldmx_hcal.common import io_utils  # noqa: E402
from ldmx_hcal.common import outputs as out_plans  # noqa: E402
from ldmx_hcal.common.config import Dataset, RunContext, resolve_datasets  # noqa: E402
from ldmx_hcal.energy import energy_response as er  # noqa: E402
from ldmx_hcal.energy import tot_adc_peak as ta  # noqa: E402

BEAM_COLUMNS = ["pf_event", "layer", "strip", "adc_sum_end0", "adc_max_end0", "tot_end0", "adc_sum_end1",
                "adc_max_end1", "tot_end1"]


# ----------------------------------------------------------------------------------------------
# synthetic inputs
# ----------------------------------------------------------------------------------------------
# RunContext whose trees live in tmp; runs: key -> (particle, energy, run folder)
def _ctx(tmp: Path, runs: dict, make_plots=False, test=False, chunksize=64, sections=None, outputs=()):
    """outputs = ctx.outputs: () = --output alone (default names), None = no --output, or a list of names."""
    cfg = {"paths": {"generated_data_dir": tmp / "data", "calibration_dir": tmp / "calibration",
                     "results_dir": tmp / "results", "cache_dir": tmp / "cache"},
           "run": {"chunksize": chunksize}, "safety": {"protected_dirs": []},
           "datasets": {k: {"particle": p, "energy_gev": e, "label": f"{p} {e:g} GeV", "kind": "data",
                            "folder": str(folder)} for k, (p, e, folder) in runs.items()}}
    cfg.update(sections or {})
    return RunContext(cfg=cfg, test=test, make_plots=make_plots, outputs=None if outputs is None else list(outputs))


# stage namespace as the runner builds it (+ ctx.units from -only when a ctx is given)
def _args(*argv, from_saved=False, ctx=None):
    p = argparse.ArgumentParser()
    ta.add_arguments(p)
    ns = p.parse_args(list(argv))
    ns.from_saved = from_saved
    if ctx is not None:
        ctx.units = out_plans.selected_units(ns)
    return ns


# ta.run with the -only units set on ctx as the runner does
def _run(ctx, datasets, *argv, from_saved=False):
    return ta.run(ctx, _args(*argv, from_saved=from_saved, ctx=ctx), datasets)


# ta.io_plan with the -only units set on ctx
def _plan(ctx, datasets, *argv, from_saved=False):
    return ta.io_plan(ctx, _args(*argv, from_saved=from_saved, ctx=ctx), datasets)


# default settings with overrides
def _settings(**over):
    s = ta.stage_settings(SimpleNamespace(stage_cfg=lambda name: {}), None)
    s.update(over)
    return s


# (result, stdout text) of fn()
def _logged(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


# the message of the exception fn must raise
def _raises(fn, exc_type=BaseException) -> str:
    try:
        _logged(fn)
    except exc_type as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} must raise {exc_type.__name__}")


# what the stage parser prints for a command line it refuses (exit code 2)
def _parser_error(*argv) -> str:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            _args(*argv)
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code)
            return err.getvalue()
    raise AssertionError(f"the parser must refuse {argv}")


# fit csv sorted by (layer, strip, pf_event): (1,0) both ends TOT = 2 adc - 600, (1,1) end1 no TOT, (10,3) end1 1.5 adc
def _fit_csv(path: Path, n_events=400, seed=1, first_event=1) -> Path:
    rng = np.random.default_rng(seed)
    parts = []
    ev = np.arange(first_event, first_event + n_events)
    for layer, strip, lines in ((1, 0, ((2.0, -600.0), (2.0, -600.0))), (1, 1, ((2.0, -600.0), None)),
                                (10, 3, (None, (1.5, -100.0)))):
        cols = {"pf_event": ev, "layer": layer, "strip": strip}
        for end, line in enumerate(lines):
            adc = rng.integers(880, 1024, n_events)
            if line is None:
                tot = np.zeros(n_events, dtype=int)
            else:
                tot = np.where(adc >= 949, np.round(line[0] * adc + line[1] + rng.normal(0, 4, n_events)), 0)
            cols[f"adc_max_end{end}"] = adc
            cols[f"tot_end{end}"] = tot.astype(int)
            cols[f"adc_sum_end{end}"] = 8 * 100 + 5 * adc
        parts.append(pd.DataFrame(cols))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True)[BEAM_COLUMNS].to_csv(path, index=False)
    return path


# event csv with every case of the reconstruction, sorted by (layer, strip, pf_event), a row per bar and event
def _event_csv(path: Path, n_events=60, seed=7, bars=None) -> Path:
    rng = np.random.default_rng(seed)
    bars = bars or [(1, 0), (1, 1), (2, 3), (9, 7), (10, 0), (12, 5), (19, 11)]
    parts = []
    ev = np.arange(1, n_events + 1)
    for layer, strip in bars:
        cols = {"pf_event": ev, "layer": layer, "strip": strip}
        for end in geo.ENDS:
            kind = rng.integers(0, 4, n_events)          # 0 no hit, 1 small hit, 2 large hit, 3 at the rail
            adc_max = np.select([kind == 0, kind == 1, kind == 2], [rng.integers(95, 110, n_events),
                                                                    rng.integers(160, 400, n_events),
                                                                    rng.integers(600, 1000, n_events)], 1023)
            adc_max = np.where((kind == 3) & (rng.random(n_events) < 0.3), 1022, adc_max)
            tot = np.where(kind == 3, rng.integers(1400, 3000, n_events), 0)
            tot = np.where((kind == 3) & (rng.random(n_events) < 0.2), 0, tot)   # railed without TOT
            cols[f"adc_max_end{end}"] = adc_max
            cols[f"tot_end{end}"] = tot
            cols[f"adc_sum_end{end}"] = np.where(kind == 0, rng.integers(780, 900, n_events), 3 * adc_max + 700)
        parts.append(pd.DataFrame(cols))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True)[BEAM_COLUMNS].to_csv(path, index=False)
    return path


# synthetic calibration table (every channel the same pedestal)
def _calibration_csv(path: Path, pedestal_per_sample=100.0) -> Path:
    df = geo.all_channels()
    df["pedestal"] = 8 * pedestal_per_sample
    df["pedestal_per_time_sample"] = pedestal_per_sample
    df["pedestal_sigma"] = 5.0
    df["pedestal_mu"] = 0.0
    df["pedestal_range_with_5sigma"] = 825.0
    df["pedestal_range_with_20percent"] = 960.0
    df["mpv"] = 500.0
    df["mpv_error"] = 1.0
    df["mpv_status"] = "OK"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# a line table: every usable_every-th channel has a usable line
def _line_table(usable_every=2) -> pd.DataFrame:
    df = geo.all_channels()
    good = (np.arange(len(df)) % usable_every) == 0
    df["tot_per_adc_slope"] = np.where(good, 2.0, 0.0)
    df["tot_offset"] = np.where(good, -600.0, 0.0)
    df["status"] = np.where(good, "ok", "no_data")
    return df


# independent per-channel fit with bins over the whole window (pd.cut, numpy polyfit)
def _reference_lines(csv_paths, s: dict) -> dict:
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    lo, hi = s["fit_adc_window"]
    tlo, thi = s["fit_tot_window"]
    w = s["fit_bin_width"]
    edges = lo + w * np.arange(int((hi - lo) // w) + 2)
    out = {}
    for (layer, strip), g in df.groupby(["layer", "strip"]):
        for end in geo.ENDS:
            a, t = g[f"adc_max_end{end}"].to_numpy(float), g[f"tot_end{end}"].to_numpy(float)
            m = (a >= lo) & (a <= hi) & (t > tlo) & (t < thi)
            if not m.any():
                continue
            b = pd.cut(a[m], edges, right=False)
            grouped = pd.DataFrame({"b": b, "a": a[m], "t": t[m]}).dropna().groupby("b", observed=True)
            stats = grouped["t"].agg(["mean", "size"])
            x = grouped["a"].mean().to_numpy()  # x = bin mean adc_max (adc_max is an integer, not the bin centre)
            y, n = stats["mean"].to_numpy(), stats["size"].to_numpy(float)
            keep = np.abs(y - y.mean()) <= 3 * y.std()
            if keep.sum() < 2:
                continue
            slope, offset = np.polyfit(x[keep], y[keep], 1, w=np.sqrt(n[keep]))
            out[(int(layer), int(strip), end)] = (slope, offset, int(n[keep].sum()))
    return out


# synthetic chain: pion 3 fit run (two csv files), electron 1 / electron 4 runs, calibration table -> (ctx, runs)
def _chain(tmp: Path, test=False, make_plots=False, chunksize=97, n_events=80, outputs=()):
    runs = tmp / "runs"
    _fit_csv(runs / "Pion_3GeV" / "a.csv", n_events=150, seed=1)
    _fit_csv(runs / "Pion_3GeV" / "b.csv", n_events=150, seed=2, first_event=1)
    _event_csv(runs / "Electron_1GeV" / "e1.csv", n_events=n_events, seed=11)
    _event_csv(runs / "Electron_4GeV" / "e4a.csv", n_events=n_events, seed=12)
    _event_csv(runs / "Electron_4GeV" / "e4b.csv", n_events=n_events // 2, seed=13)
    ctx = _ctx(tmp, {"pion_3GeV": ("pion", 3.0, runs / "Pion_3GeV"),
                     "electron_1GeV": ("electron", 1.0, runs / "Electron_1GeV"),
                     "electron_4GeV": ("electron", 4.0, runs / "Electron_4GeV")},
               make_plots=make_plots, test=test, chunksize=chunksize, outputs=outputs)
    _calibration_csv(ctx.calibration_product(mode="fullset"))
    return ctx, resolve_datasets(ctx.cfg, ["electron 1", "electron 4"])


# temp files of any writer
def _temp_leftovers(*folders) -> list[str]:
    out = []
    for folder in folders:
        if Path(folder).exists():
            out += [p.name for p in Path(folder).rglob("*") if ".tmp" in p.name or ".old-" in p.name]
    return sorted(out)


# sorted file names of a folder ([] if absent)
def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in Path(folder).iterdir()) if Path(folder).exists() else []


# ----------------------------------------------------------------------------------------------
# routing, variant tags, settings
# ----------------------------------------------------------------------------------------------
# product paths per tree and mode; path functions create nothing
def test_output_routing_by_category_and_mode():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for test, mode in ((False, "fullset"), (True, "test")):
            ctx = _ctx(tmp, {"electron_1GeV": ("electron", 1.0, tmp / "runs" / "Electron_1GeV")}, test=test)
            ds = resolve_datasets(ctx.cfg, ["electron 1"])[0]
            sub = "test" if test else ""
            assert ta.parameters_csv(ctx) == tmp / "calibration" / sub / "tot_adc_peak" / "tot_adc_peak_parameters.csv"
            res = tmp / "results" / mode / "tot_adc_peak"
            assert ta.fits_pdf(ctx) == res / "tot_adc_peak_fits.pdf"
            assert ta.energy_response_csv(ctx, "_none") == res / "energy_response_none.csv"
            assert ta.energy_spectra_pdf(ctx, "none") == res / "energy_spectra_none.pdf"
            assert ta.linearity_resolution_pdf(ctx) == res / "linearity_resolution.pdf"
            cache = tmp / "cache" / sub / "tot_adc_peak"
            assert ta.saved_events_csv(ctx, ds, "x/run1.csv") == cache / "Electron_1GeV" / "run1_events.csv"
            assert ta.bar_values_csv(ctx, ds, "run1.csv", "none") == cache / "Electron_1GeV" / "run1_bars_none.csv"
            assert ta.saved_fit_histograms(ctx, "pion_3GeV") == cache / "fit_histograms_pion_3GeV.npz"
        assert not any((tmp / d).exists() for d in ("data", "results", "cache", "calibration"))
    assert ta.SUPPORTS_FROM_SAVED is True


# variant tag = tot_mode (if not linear) + output tag; --output-tag is checked
def test_variant_tag_and_output_tag():
    for over, tag in (({}, ""), ({"tot_mode": "none"}, "none"), ({"tot_mode": "mean_fallback"}, "mean_fallback"),
                      ({"output_tag": "q0125"}, "q0125"), ({"tot_mode": "none", "output_tag": "x"}, "none_x")):
        s = _settings(**over)
        assert ta.variant_tag(s) == tag and ta.output_suffix(s) == (f"_{tag}" if tag else ""), over
    with tempfile.TemporaryDirectory() as td:
        ctx = _ctx(Path(td), {})
        for bad, words in (("../x", "may only contain"), ("a/b", "may only contain"), (".hidden", "must start"),
                           ("none", "reserved"), ("None_q", "reserved"), ("mean_fallback", "reserved")):
            assert words in _raises(lambda: ta.stage_settings(ctx, _args("--output-tag", bad)), SystemExit), bad
        for given, tag in (("q0125", "q0125"), ("_x", "x"), ("nonex", "nonex"), ("v1.2-b_c", "v1.2-b_c")):
            s, _ = _logged(lambda: ta.stage_settings(ctx, _args("--output-tag", given)))
            assert s["output_tag"] == tag
        msg = _raises(lambda: ta.stage_settings(_ctx(Path(td), {}, sections={"tot_adc_peak": {"output_tag": "none"}}),
                                                _args()), SystemExit)
        assert "reserved" in msg


# unknown keys, bad values and command-line-only keys stop before any IO
def test_settings_are_checked_before_any_io():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s = ta.stage_settings(_ctx(tmp, {}), _args())
        assert s["fit_bin_width"] == 3 and s["spectrum_fit"] == "two_pass" and s["merge_rail_adc"] == 1022
        bad = {"fit_bin_width": -3, "fit_adc_window": [1021, 949], "fit_min_bins": 1, "plot_max_points": 2.5,
               "quality_threshold": float("nan"), "saturation_adc": 0,
               "write_bar_csv": "yes", "spectrum_fit": "fast", "tot_mode": "quadratic"}
        msg = _raises(lambda: ta.stage_settings(_ctx(tmp, {}, sections={"tot_adc_peak": bad}), _args()),
                      io_utils.SettingsError)
        for key in bad:
            assert key in msg, (key, msg)
        c = _ctx(tmp, {}, sections={"tot_adc_peak": {"fit_adc_windw": [949, 1021]}})       # a typo is never ignored
        assert "unknown setting" in _raises(lambda: ta.stage_settings(c, _args()), io_utils.SettingsError)
        for key in ("fit_ranges", "fit_bins"):
            c = _ctx(tmp, {}, sections={"tot_adc_peak": {key: [1]}})
            assert "command-line only" in _raises(lambda: ta.stage_settings(c, _args()), io_utils.SettingsError)
        for argv, key in ((("--quality-threshold", "1.5"), "quality_threshold"),
                          (("--slope-window", "4", "0.6"), "slope_window"),
                          (("--saturation-adc", "-1"), "saturation_adc")):
            assert key in _raises(lambda: ta.stage_settings(_ctx(tmp, {}), _args(*argv)), io_utils.SettingsError)
        for bad_er in ({"linearity_fits": {"electron": [1.0, 0.3]}}, {"resolution_fit_range": [0.3]},
                       {"min_bin_count": -1}):
            c = _ctx(tmp, {}, sections={"energy_response": bad_er})
            assert next(iter(bad_er)) in _raises(lambda: ta.response_settings(c, _settings(), _args()),
                                                 io_utils.SettingsError)
        assert "needs a range" in _parser_error("-fit")


# -only units and groups, -no-plots, and the flags that need a response unit
def test_only_units_and_their_combinations():
    with tempfile.TemporaryDirectory() as td:
        ctx, ds = _chain(Path(td), make_plots=True)
        for argv, units in (((), ta.UNITS), (("-only", "line"), ("line_table", "line_fits")),
                            (("-only", "response"), ta.RESPONSE_UNITS), (("-only", "csv"),
                                                                         ("line_table", "energy_response")),
                            (("-only", "plots"), ta.PLOT_UNITS), (("-only", "energy_spectra,line_table"),
                                                                  ("line_table", "energy_spectra"))):
            steps = ta.resolve_steps(ctx, ta.stage_settings(ctx, _args(*argv, ctx=ctx)), ds)
            assert steps["units"] == units, (argv, steps["units"])
            assert steps["line"] == any(u.startswith("line_") for u in units)
            assert steps["response"] == any(u in ta.RESPONSE_UNITS for u in units)
        assert "unknown unit" in _parser_error("-only", "events")
        ctx.make_plots = False
        steps = ta.resolve_steps(ctx, ta.stage_settings(ctx, _args(ctx=ctx)), ds)
        assert steps["units"] == ("line_table", "energy_response")
        assert "nothing to make" in _raises(lambda: ta.resolve_steps(
            ctx, ta.stage_settings(ctx, _args("-only", "plots", ctx=ctx)), ds), io_utils.SettingsError)
        ctx.make_plots = True
        for argv, words in ((("-only", "line", "-bar-csv"), "--bar-csv"), (("-bar-csv",), "-from-saved"),
                            (("-only", "line", "-linfit", "electron"), "--linfit")):
            saved = argv == ("-bar-csv",)
            msg = _raises(lambda: ta.resolve_steps(
                ctx, ta.stage_settings(ctx, _args(*argv, from_saved=saved, ctx=ctx)), ds), io_utils.SettingsError)
            assert words in msg, (argv, msg)
        c = _ctx(Path(td), {}, sections={"tot_adc_peak": {"write_bar_csv": True}})
        _, text = _logged(lambda: ta.resolve_steps(c, ta.stage_settings(c, _args("-only", "line", ctx=c)), ds))
        assert "write_bar_csv: true (yml) ignored" in text


# ----------------------------------------------------------------------------------------------
# line fit
# ----------------------------------------------------------------------------------------------
# the bins cover the whole fit_adc_window: hits with adc_max 1018-1021 enter the fit
def test_fit_bins_cover_the_whole_adc_window():
    edges = ta.fit_bin_edges(_settings())
    assert edges[0] == 949 and edges[-1] == 1024 and len(edges) == 26 and edges[-2] <= 1021 < edges[-1]
    assert ta.fit_bin_edges(_settings(fit_adc_window=[949, 1020]))[-1] > 1020
    with tempfile.TemporaryDirectory() as td:
        csv = _fit_csv(Path(td) / "fit.csv", n_events=500, seed=3)
        df = pd.read_csv(csv)
        s = _settings()
        hist = ta.accumulate_fit_histograms(csv, s, 10_000, keep_points=False)
        slot = int(ta.channel_slot(1, 0, 0))
        sel = (df["layer"] == 1) & (df["strip"] == 0) & df["adc_max_end0"].between(949, 1021) & \
            (df["tot_end0"] > 500) & (df["tot_end0"] < 2000)
        assert int(hist.counts[slot].sum()) == int(hist.n_window[slot]) == int(sel.sum())    # no hit lost
        assert int(hist.counts[slot][-1]) == int((sel & (df["adc_max_end0"] >= 1021)).sum()) > 0


# fit csv files of a run summed; the fit equals an independent reference and does not depend on the sample
def test_fit_parameters_sum_the_run_csvs_and_do_not_depend_on_the_display_sample():
    with tempfile.TemporaryDirectory() as td:
        a = _fit_csv(Path(td) / "a.csv", n_events=300, seed=3)
        b = _fit_csv(Path(td) / "b.csv", n_events=200, seed=4)
        tables = []
        for keep, cap, chunksize in ((False, 5000, 10_000), (True, 5000, 10_000), (True, 7, 41), (True, 0, 999)):
            s = _settings(plot_max_points=cap)
            hist = ta.accumulate_fit_histograms([a, b], s, chunksize, keep_points=keep)
            tables.append(ta.fit_tot_lines(hist, s, "pion_3GeV")[0])
        for t in tables[1:]:
            pd.testing.assert_frame_equal(t, tables[0], check_exact=False, rtol=1e-12)
        table = tables[0]
        assert len(table) == 384
        ref = _reference_lines([a, b], _settings())
        ok = table[table["status"] == "ok"]
        assert {(r.layer, r.strip, r.end) for r in ok.itertuples()} == set(ref)
        for r in ok.itertuples():
            slope, offset, n_fit = ref[(r.layer, r.strip, r.end)]
            assert np.isclose(r.tot_per_adc_slope, slope, rtol=1e-9) and np.isclose(r.tot_offset, offset, rtol=1e-9)
            assert r.n_hits_fit == n_fit
        assert abs(ok.set_index(["layer", "strip", "end"]).loc[(10, 3, 1), "tot_per_adc_slope"] - 1.5) < 0.05
        status = table.set_index(["layer", "strip", "end"])["status"]
        assert status[(1, 1, 1)] == "no_hits_in_window" and status[(2, 0, 0)] == "no_data"
        one = ta.accumulate_fit_histograms(a, _settings(), 50, keep_points=False)
        two = ta.accumulate_fit_histograms([a, b], _settings(), 50, keep_points=False)
        assert two.n_rows == one.n_rows + len(pd.read_csv(b)) and (two.counts >= one.counts).all()


# the display sample is bounded, reproducible and chunk independent; the true fired count is kept
def test_fired_hit_sample_is_bounded_reproducible_and_chunk_independent():
    with tempfile.TemporaryDirectory() as td:
        csv = _fit_csv(Path(td) / "fit.csv", n_events=400)
        df = pd.read_csv(csv)
        s = _settings(plot_max_points=50)
        results = [ta.accumulate_fit_histograms(csv, s, chunksize, keep_points=True) for chunksize in (37, 400, 10_000)]
        slot = int(ta.channel_slot(1, 0, 0))
        fired = (df["layer"] == 1) & (df["strip"] == 0) & (df["adc_max_end0"] >= 900) & (df["tot_end0"] > 0)
        for hist in results:
            assert hist.sample_cap == 50 and all(len(x) <= 50 for x, _ in hist.points.values())
            assert int(hist.n_fired[slot]) == int(fired.sum()) and len(hist.points[slot][0]) == 50
            assert float(hist.tot_max[slot]) == float(df.loc[fired, "tot_end0"].max())
        ref = results[0].points
        for hist in results[1:]:
            assert sorted(hist.points) == sorted(ref)
            for k in ref:
                assert np.array_equal(hist.points[k][0], ref[k][0]) and np.array_equal(hist.points[k][1], ref[k][1])
        full = ta.accumulate_fit_histograms(csv, _settings(plot_max_points=0), 37, True)
        assert np.array_equal(full.points[slot][1], df.loc[fired, "tot_end0"].to_numpy())
        none_slot = int(ta.channel_slot(1, 1, 1))
        assert none_slot not in ref and np.isnan(results[0].tot_max[none_slot])


_M64 = (1 << 64) - 1


# plain-Python SplitMix64 output for state x
def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


# key = splitmix64((2 * row + end) XOR splitmix64(seed)), bottom-k per channel-end
def test_display_sample_follows_the_documented_key_formula():
    with tempfile.TemporaryDirectory() as td:
        csv = _fit_csv(Path(td) / "fit.csv", n_events=300, seed=5)
        df = pd.read_csv(csv)
        cap, seed = 20, 777
        hist = ta.accumulate_fit_histograms(csv, _settings(plot_max_points=cap, plot_sample_seed=seed), 53, True)
        seed_key = _splitmix64(seed)
        expected = {}
        for row, rec in enumerate(df.itertuples(index=False)):
            for end in geo.ENDS:
                adc, tot = getattr(rec, f"adc_max_end{end}"), getattr(rec, f"tot_end{end}")
                if adc >= 900 and tot > 0:
                    order = 2 * row + end
                    expected.setdefault(int(ta.channel_slot(rec.layer, rec.strip, end)), []).append(
                        (_splitmix64(order ^ seed_key), order, adc, tot))
        assert sorted(expected) == sorted(hist.points)
        for slot, hits in expected.items():
            kept = sorted(sorted(hits)[:cap], key=lambda h: h[1])
            a, t = hist.points[slot]
            assert a.tolist() == [h[2] for h in kept] and t.tolist() == [h[3] for h in kept]


# the per-channel key bound keeps the exact sample, also with forced hash ties
def test_display_sample_key_bound_keeps_the_exact_sample_also_with_hash_ties():
    rng = np.random.default_rng(3)
    n = 3000
    slot = rng.integers(0, 40, n) * 2
    order = np.arange(n, dtype=np.int64) * 2 + rng.integers(0, 2, n)
    adc, tot = rng.integers(900, 1023, n), rng.integers(1, 3000, n)
    real_mix = ta._mix64
    for coarse in (False, True):
        mix = (lambda x: real_mix(x) >> np.uint64(58)) if coarse else real_mix
        with mock.patch.object(ta, "_mix64", side_effect=mix):
            for cap in (1, 7, 60):
                ref = None
                for chunk in (1, 13, n):
                    new = ta.FiredHitSample(cap, 12345)
                    for i in range(0, n, chunk):
                        sl = slice(i, i + chunk)
                        new.add(slot[sl], order[sl], adc[sl], tot[sl])
                    got = new.points()
                    assert len(new.slot) <= cap * 40 and all(len(v[0]) <= cap for v in got.values())
                    if ref is None:
                        ref = got
                        continue
                    assert sorted(got) == sorted(ref), (coarse, cap, chunk)
                    for k in ref:
                        assert np.array_equal(got[k][0], ref[k][0]) and np.array_equal(got[k][1], ref[k][1])


# ----------------------------------------------------------------------------------------------
# the line table
# ----------------------------------------------------------------------------------------------
# a calibration table for build_event_inputs
def _calib_frame():
    return geo.all_channels().assign(pedestal=800., pedestal_per_time_sample=100., pedestal_range_with_20percent=960.)


# duplicates, bad keys, text values and NaN on an ok line are refused; a table without the line columns is no line table
def test_tot_line_table_is_checked_at_every_entry():
    s = _settings()
    rows = pd.DataFrame([dict(layer=1, strip=0, end=0, tot_per_adc_slope=slope, tot_offset=0., status="ok")
                         for slope in (1., 2.)])
    for frame in (rows, rows.iloc[::-1]):
        assert "share a (layer, strip, end) key" in _raises(lambda: ta.build_event_inputs(_calib_frame(), frame, s),
                                                            io_utils.ChannelKeyError)
    base = _line_table()
    for table in (base.assign(layer=np.where(np.arange(len(base)) == 3, 0, base["layer"])),
                  base.assign(end=np.where(np.arange(len(base)) == 7, 2, base["end"])),
                  base.assign(end=("end" + base["end"].astype(str)))):
        _raises(lambda: ta.build_event_inputs(_calib_frame(), table, s), io_utils.ChannelKeyError)
    ok_nan = base.copy()
    ok_nan.loc[0, "tot_offset"] = np.nan
    msg = _raises(lambda: ta.build_event_inputs(_calib_frame(), ok_nan, s), io_utils.InputError)
    assert "status 'ok'" in msg and "layer 1 strip 0 end 0" in msg
    other = base.rename(columns={"tot_per_adc_slope": "slope", "tot_offset": "intercept"}).drop(columns="status")
    assert "lacks columns" in _raises(lambda: ta.build_event_inputs(_calib_frame(), other, s), io_utils.InputError)
    lax = base.copy()
    lax.loc[1, ["tot_per_adc_slope", "tot_offset"]] = np.nan                        # status no_data: never used
    inp = ta.build_event_inputs(_calib_frame(), lax.drop(index=[10, 11]), s)
    assert inp.n_usable_lines == 191                                               # index 10 dropped
    none = ta.build_event_inputs(_calib_frame(), None, _settings(tot_mode="none"))  # tot_mode none: no table
    assert none.n_usable_lines == 0 and not none.conversion_allowed.any()


# the line table is parsed exactly: the events of a fit run equal those of a later run that reads the table
def test_line_table_reads_back_exactly():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rng = np.random.default_rng(3)
        table = _line_table(usable_every=1)
        table["tot_per_adc_slope"] = rng.uniform(0.5, 4.0, len(table))
        table["tot_offset"] = rng.normal(-600.0, 200.0, len(table))
        table.to_csv(tmp / "t.csv", index=False)
        back = ta.load_tot_parameters(tmp / "t.csv")
        for col in ("tot_per_adc_slope", "tot_offset"):
            assert np.array_equal(back[col].to_numpy().view(np.int64), table[col].to_numpy().view(np.int64))
        ctx, ds = _chain(tmp / "chain")
        _logged(lambda: _run(ctx, ds, "-only", "line_table,energy_response"))
        first = {p: p.read_bytes() for p in (ta.cache_dir(ctx)).rglob("*_events.csv")}
        assert len(first) == 3
        _logged(lambda: _run(ctx, ds, "-only", "energy_response"))
        assert {p: p.read_bytes() for p in first} == first


# ----------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------
# collects the frames of a bar writer
class _ListWriter:

    # empty collector
    def __init__(self):
        self.frames = []

    # keep one frame
    def write(self, df):
        if df is not None and len(df):
            self.frames.append(df)

    # every frame in one table
    def frame(self):
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame(columns=ta.BAR_COLUMNS)


# streamed bar values == in-memory bars, with and without the event cut; the second pass reads selected rows only
def test_streamed_bar_values_equal_in_memory_bars_with_and_without_event_cut():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        csv = _event_csv(tmp / "e1.csv", n_events=90, seed=21)
        calib = ta.load_calibration(_calibration_csv(tmp / "calib.csv"))
        n_rows = len(pd.read_csv(csv))
        for thr in (None, 0.2, 0.0):
            s = _settings(quality_threshold=thr)
            inp = ta.build_event_inputs(calib, _line_table(), s)
            ev_ref, bars_ref = ta.arbitrary_energies_for_csv(csv, "Electron_1GeV/e1.csv", inp, s, n_rows + 1,
                                                             keep_bars=True)
            assert list(ev_ref.columns) == list(ta.EVENT_COLUMNS) and list(bars_ref.columns) == ta.BAR_COLUMNS
            assert set(ev_ref["event_source"]) == {"Electron_1GeV/e1.csv"}
            if thr is not None:
                assert 0 < int(ev_ref["selected"].sum()) < len(ev_ref)
            counted, real = [], ta._reconstruct_chunk

            # count the rows handed to the reconstruction
            def counting(chunk, *a):
                counted.append(len(chunk))
                return real(chunk, *a)

            for chunksize in (5, 23, n_rows + 1):
                writer = _ListWriter()
                counted.clear()
                with mock.patch.object(ta, "_reconstruct_chunk", side_effect=counting):
                    ev, none = ta.arbitrary_energies_for_csv(csv, "Electron_1GeV/e1.csv", inp, s, chunksize,
                                                             bar_writer=writer)
                assert none is None
                pd.testing.assert_frame_equal(ev, ev_ref)
                pd.testing.assert_frame_equal(writer.frame().reset_index(drop=True), bars_ref.reset_index(drop=True),
                                              check_dtype=False)
                if thr is not None:
                    assert sum(counted) - n_rows == int(ev_ref["selected"].sum()) * 7


# the compact event accumulator is bit-identical to the two-stage reduction
def test_event_accumulator_is_bit_identical_to_the_two_stage_reduction():
    rng = np.random.default_rng(8)
    n = 4000
    pf = np.concatenate([np.sort(rng.integers(0, 300, n // 2)), rng.integers(10**9, 10**9 + 50, n // 2)])
    w = {"a": rng.normal(0, 1e3, n) * 10.0 ** rng.integers(-8, 8, n), "b": rng.integers(0, 2, n).astype(float)}
    for chunk in (1, 17, 333, n):
        acc = ta.EventAccumulator(["a", "b"])
        parts = []
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            acc.add(pf[sl], {k: v[sl] for k, v in w.items()})
            ids, inv = np.unique(pf[sl], return_inverse=True)
            parts.append((ids, {k: np.bincount(inv, weights=v[sl], minlength=len(ids)) for k, v in w.items()}))
        all_ids, inv = np.unique(np.concatenate([p[0] for p in parts]), return_inverse=True)
        ref = {k: np.bincount(inv, weights=np.concatenate([p[1][k] for p in parts]), minlength=len(all_ids))
               for k in w}
        ids, sums = acc.finalize()
        assert np.array_equal(ids, all_ids)
        for k in w:
            assert np.array_equal(sums[k].view(np.int64), ref[k].view(np.int64)), (chunk, k)


# rail-aware end merge (provided by Erik Wallin): the railed end with TOT, else the other valid end, else 0
def test_rail_aware_merge_policy():
    # one end dict of n bars
    def end(value, valid, sat, tpos, used=False, keep=True):
        n = len(value)
        return {"value": np.asarray(value, float), "valid": np.asarray(valid), "sat": np.asarray(sat),
                "tpos": np.asarray(tpos), "used": np.full(n, used), "keep": np.full(n, keep)}

    e0 = end([500.0, 400.0, 300.0], [True, False, False], [True, True, True], [True, True, False])
    e1 = end([200.0, 150.0, 100.0], [True, True, False], [False, False, False], [False, False, False])
    assert ta.merge_ends_rail_aware(e0, e1).tolist() == [500.0, 150.0, 0.0]
    assert ta.merge_ends_raw(e0, e1).tolist() == [350.0, 150.0, 0.0]


# ----------------------------------------------------------------------------------------------
# response
# ----------------------------------------------------------------------------------------------
# a spectrum point of one run
def _point(key, particle, energy, mu, n=1000):
    ds = Dataset(key=key, particle=particle, energy_gev=energy, label=key, folder="unused")
    return {"dataset": ds, "method": "two_pass", "note": "", "values": np.array([mu]),
            "fit": {"mu": mu, "mu_error": 1.0, "sigma": 0.2 * mu * energy ** -0.5, "sigma_error": 0.01 * mu,
                    "n_values": n, "sigma_over_mu": 0.2 * energy ** -0.5, "status": "ok", "ok": True}}


# every particle drawn; no line by default; -linfit one line per particle; resolution fitted to electrons only
def test_linearity_and_resolution_per_particle():
    with tempfile.TemporaryDirectory() as td:
        ctx = _ctx(Path(td), {})
        s = _settings()
        spectra = [_point("electron_0.5GeV", "electron", 0.5, 5.0), _point("electron_1GeV", "electron", 1.0, 10.1),
                   _point("electron_4GeV", "electron", 4.0, 39.0), _point("pion_3GeV", "pion", 3.0, 25.0),
                   _point("pion_4GeV", "pion", 4.0, 33.0), _point("muon_4GeV", "muon", 4.0, 9.0)]
        rs = ta.response_settings(ctx, s, _args())
        resp = ta.response_fits(spectra, rs)
        assert resp["linearity"]["fits"] == {} and not resp["linearity"]["fitted"].any()
        res = resp["resolution"]
        assert res["fitted"].tolist() == [True, True, True, False, False, False] and res["fit_particle"] == "electron"
        table = ta.response_table(resp, s, rs)
        assert list(table["record"]) == ["spectrum"] * 6 + ["resolution"]
        assert list(table.columns) == [c for c in ta.RESPONSE_COLUMNS if c in table.columns]
        assert not [c for c in table.columns if c.endswith("_scaled")] and "linearity_predicted" not in table.columns
        rs = ta.response_settings(ctx, s, _args("-linfit", "electron[0,1]", "pion", "-normalerror"))
        resp = ta.response_fits(spectra, rs)
        lin = resp["linearity"]
        assert sorted(lin["fits"]) == ["electron", "pion"]
        assert lin["fits"]["electron"]["fitted"].tolist() == [True, True, False, False, False, False]
        assert lin["fits"]["pion"]["fitted"].tolist() == [False, False, False, True, True, False]
        table = ta.response_table(resp, s, rs)
        rows = table[table["record"] == "linearity"].set_index("particle")
        assert list(rows.index) == ["electron", "pion"] and rows.loc["electron", "fit_range_gev"] == "[0, 1]"
        assert "slope_error_scaled" in table.columns
        spec = table[table["record"] == "spectrum"].set_index("dataset")
        assert spec.loc["pion_3GeV", "in_linearity_fit"] and np.isnan(spec.loc["muon_4GeV", "linearity_predicted"])
        assert np.isclose(spec.loc["pion_3GeV", "linearity_predicted"], lin["fits"]["pion"]["predicted"][3])
        ta.plot_response(Path(td) / "s.pdf", Path(td) / "l.pdf", resp, rs, "test")
        assert (Path(td) / "l.pdf").stat().st_size > 10_000 and (Path(td) / "s.pdf").stat().st_size > 1000


# synthetic spectra of two runs: electron 1 N(5000, 1100), pion 3 two peaks
def _spectrum_events(seed=5):
    rng = np.random.default_rng(seed)
    values = {"electron_1GeV": rng.normal(5000.0, 1100.0, 20000),
              "pion_3GeV": np.concatenate([rng.normal(3000.0, 500.0, 6000), rng.normal(12000.0, 2500.0, 14000)])}
    events = pd.concat([pd.DataFrame({"dataset": k, "pf_event": np.arange(len(v)), "arbitrary_energy": v,
                                      "selected": True}) for k, v in values.items()], ignore_index=True)
    ds = {k: Dataset(key=k, particle=p, energy_gev=e, label=k, folder="unused")
          for k, p, e in (("electron_1GeV", "electron", 1.0), ("pion_3GeV", "pion", 3.0))}
    return events, ds, values


# entries of ref equal in got (floats bit for bit)
def _same_fit(got: dict, ref: dict) -> None:
    for key, value in ref.items():
        g = got[key]
        if isinstance(value, np.ndarray):
            assert np.array_equal(g, value, equal_nan=value.dtype.kind == "f"), key
        elif isinstance(value, float) and np.isnan(value):
            assert isinstance(g, float) and np.isnan(g), key
        else:
            assert g == value, (key, g, value)


# two_pass / peak_scan fits equal the energy_response calls; -fit single and double Gaussian
def test_spectrum_fits_and_fit_ranges():
    events, ds, values = _spectrum_events()
    with tempfile.TemporaryDirectory() as td:
        ctx = _ctx(Path(td), {})
        s = _settings()
        rs = ta.response_settings(ctx, s, _args())
        spectra = ta.fit_spectra(events, [ds["electron_1GeV"], ds["pion_3GeV"]], s, rs)
        for p in spectra:
            _same_fit(p["fit"], er.two_pass_gauss_fit(values[p["dataset"].key], bins=90, min_bin_count=10,
                                                      window_nsigma=2.0, min_fit_bins=8, upper_percentile=99.8,
                                                      upper_factor=1.08, upper_min=3.0))
        ps = _settings(spectrum_fit="peak_scan")
        (pk,) = ta.fit_spectra(events, [ds["electron_1GeV"]], ps, rs)
        _same_fit(pk["fit"], er.peak_scan_fit(values["electron_1GeV"], n_bins=300, upper_percentile=99.7,
                                              upper_factor=1.12, upper_min=3.0))
        rs1 = ta.response_settings(ctx, s, _args("--fit", "[3500,6500]"))
        (p,) = ta.fit_spectra(events, [ds["electron_1GeV"]], s, rs1)
        fit = p["fit"]
        assert p["method"] == "range" and fit["kind"] == "range_gauss" and fit["ok"]
        assert abs(fit["mu"] - 5000) < 5 * fit["mu_error"]
        assert p["note"].startswith("--fit [3500, 6500]: single Gaussian, replaces the automatic two_pass fit")
        assert p["note"].endswith("of the two-pass page")
        rs2 = ta.response_settings(ctx, s, _args("--fit", "[2000,4000],[7000,17000]", "--bin", "20,40"))
        (p2,) = ta.fit_spectra(events, [ds["pion_3GeV"]], s, rs2)
        f2 = p2["fit"]
        assert f2["kind"] == "range_double_gauss" and f2["ok"] and np.isnan(f2["mu"])
        for tag, mu in (("low", 3000), ("high", 12000)):
            assert abs(f2[f"mu_{tag}"] - mu) < 5 * f2[f"mu_{tag}_error"], tag
        table = ta.response_table(ta.response_fits([p, p2], rs2), s, rs2)
        spec = table[table["record"] == "spectrum"].set_index("dataset")
        assert spec.loc["pion_3GeV", "fit_ranges"] == "[2000, 4000], [7000, 17000]"
        assert spec.loc["pion_3GeV", "bin_rule"] == "20,40 equal bins per range" and np.isnan(spec.loc["pion_3GeV",
                                                                                                      "mu"])
        assert spec.loc["electron_1GeV", "status"] == f"ok ({p['note']})"
        assert ta.no_point_reason(p) is None and ta.no_point_reason(p2) == "double Gaussian --fit: no single mu"
        with mock.patch.object(fitting, "curve_fit", side_effect=RuntimeError("no convergence (test)")):
            (pf,) = ta.fit_spectra(events, [ds["electron_1GeV"]], s, rs1)
        assert not pf["fit"]["ok"] and "no convergence (test)" in pf["fit"]["status"] and np.isnan(pf["fit"]["mu"])
        assert ta.no_point_reason(pf).startswith("spectrum fit failed: ")


# ----------------------------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------------------------
# a -test run writes each product into its tree: per-csv cache in run sub-folders, nothing in data/
def test_end_to_end_test_run_writes_each_product_into_its_tree():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp, test=True, outputs=[])
        _logged(lambda: _run(ctx, ds, "-bar-csv"))
        cal, res, cache = (tmp / d / "test" / "tot_adc_peak" for d in ("calibration", "results", "cache"))
        assert _names(cal) == ["tot_adc_peak_parameters.csv", "tot_adc_peak_parameters.provenance.json"]
        assert _names(res) == ["energy_response.csv", "energy_response.provenance.json"]
        assert _names(cache) == ["Electron_1GeV", "Electron_4GeV", "fit_histograms_pion_3GeV.npz",
                                 "fit_histograms_pion_3GeV.provenance.json"]
        assert _names(cache / "Electron_4GeV") == ["e4a_bars.csv", "e4a_events.csv", "e4a_events.provenance.json",
                                                   "e4b_bars.csv", "e4b_events.csv", "e4b_events.provenance.json"]
        assert not (tmp / "data" / "test" / "tot_adc_peak").exists() and not (tmp / "data" / "tot_adc_peak").exists()
        assert not any((tmp / d / "tot_adc_peak").exists() for d in ("calibration", "cache"))
        ev = pd.read_csv(cache / "Electron_4GeV" / "e4b_events.csv")
        assert list(ev.columns) == list(ta.EVENT_COLUMNS) and set(ev["event_source"]) == {"Electron_4GeV/e4b.csv"}
        bars = pd.read_csv(cache / "Electron_4GeV" / "e4b_bars.csv")
        assert list(bars.columns) == ta.BAR_COLUMNS
        e4b = ctx.run_csvs(ds[1])[1]
        assert ta.bar_values_status(ctx, ds[1], e4b)["current"] is True
        table = pd.read_csv(res / "energy_response.csv")
        spec = table[table["record"] == "spectrum"].set_index("dataset")
        assert spec.loc["electron_4GeV", "n_events"] == 120                # both csv files of the run
        prov = io_utils.read_json(res / "energy_response.provenance.json")
        assert prov["events"]["from"] == "computed in this run" and len(prov["events"]["files"]) == 3
        lt = io_utils.read_json(cal / "tot_adc_peak_parameters.provenance.json")
        assert len(lt["inputs"]["fit_csvs"]) == 2 and lt["bin_edges"][-1] == 1024
        plan, _ = _logged(lambda: _plan(ctx, ds, "-bar-csv"))
        outs = {Path(p["path"]) for p in plan if p["role"] == "out"}
        assert ta.parameters_csv(ctx) in outs and ta.bar_values_csv(ctx, ds[1], e4b) in outs
        assert all(".\\runner.ps1" in p["produced_by"] for p in plan
                   if p.get("produced_by") and p.get("kind") != "auto")                # auto: the test subsets
        # a later run without -bar-csv removes the bar values that no longer belong to the events
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-slope-window", "1.2", "4"))
        assert not (cache / "Electron_4GeV" / "e4b_bars.csv").exists()
        assert ta.bar_values_status(ctx, ds[1], e4b)["current"] is None
        assert _temp_leftovers(tmp / "cache", tmp / "results", tmp / "calibration") == []


# -from-saved: bit-identical response, every dependency verified, fit bins from the saved copy
def test_from_saved_is_bit_identical_and_verifies_every_dependency():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp)
        _logged(lambda: _run(ctx, ds, "-only", "csv", "-quality-threshold", "0.3"))
        resp_csv = ta.energy_response_csv(ctx)
        full_run = resp_csv.read_bytes()
        table_bytes = ta.parameters_csv(ctx).read_bytes()
        resp_csv.unlink()
        with mock.patch.object(ta, "arbitrary_energies_for_csv", side_effect=AssertionError("csv read")), \
                mock.patch.object(ta, "accumulate_fit_histograms", side_effect=AssertionError("fit csv read")):
            _, text = _logged(lambda: _run(ctx, ds, "-only", "csv", "-quality-threshold", "0.3", from_saved=True))
        assert resp_csv.read_bytes() == full_run and ta.parameters_csv(ctx).read_bytes() == table_bytes
        assert "verified" in text
        assert io_utils.read_json(ta.provenance_path(resp_csv))["events"]["from"] == "saved copy"
        # response settings and the quality threshold may change (the cut is re-applied)
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-quality-threshold", "0.05", from_saved=True))
        reselected = resp_csv.read_bytes()
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-quality-threshold", "0.05"))
        assert resp_csv.read_bytes() == reselected != full_run
        # settings that change the events: refused with the reason and the rebuild command
        for argv, words in ((("--slope-window", "1.2", "4"), "slope_window: saved [0.6, 4.0], now [1.2, 4.0]"),
                            (("--saturation-adc", "1000"), "saturation_adc"),
                            (("--merge-rail-adc", "1023"), "merge_rail_adc")):
            msg = _raises(lambda: _run(ctx, ds, "-only", "energy_response", *argv, from_saved=True),
                          io_utils.ProductSetError)
            assert words in msg and "Rebuild them with" in msg and "-only energy_response" in msg, (argv, msg)
        # other calibration content is refused, the same content is accepted
        calib = ctx.calibration_product()
        _calibration_csv(calib, pedestal_per_sample=101.0)
        assert "calibration table differs" in _raises(
            lambda: _run(ctx, ds, "-only", "energy_response", from_saved=True), io_utils.ProductSetError)
        _calibration_csv(calib)
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", from_saved=True))
        # a refit line table with other content is refused, unless the events used no lines (tot_mode none)
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-tot-mode", "none"))
        _logged(lambda: _run(ctx, ds, "-only", "line_table", "-fit-adc-window", "970", "1021"))
        assert "TOT line table differs" in _raises(
            lambda: _run(ctx, ds, "-only", "energy_response", from_saved=True), io_utils.ProductSetError)
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-tot-mode", "none", from_saved=True))
        # the fit bins: other settings refused with the rebuild command
        msg = _raises(lambda: _run(ctx, ds, "-only", "line_table", from_saved=True), io_utils.ProductSetError)
        assert "fit histograms" in msg and "-fit-adc-window" not in msg and "-only line_table" in msg
        # other beam data, a run without saved events, a modified saved file
        e1 = ctx.run_csvs(ds[0])[0]
        _event_csv(e1, n_events=80, seed=99)
        assert "beam csv of electron 1 (Electron_1GeV/e1.csv) differs" in _raises(
            lambda: _run(ctx, ds, "-only", "energy_response", "-tot-mode", "none", from_saved=True),
            io_utils.ProductSetError)
        _event_csv(e1, n_events=80, seed=11)
        _event_csv(tmp / "runs" / "Electron_0.5GeV" / "e05.csv", seed=13)
        ctx.cfg["datasets"]["electron_0.5GeV"] = {"particle": "electron", "energy_gev": 0.5, "kind": "data",
                                                  "folder": str(tmp / "runs" / "Electron_0.5GeV")}
        more = resolve_datasets(ctx.cfg, ["electron 1", "electron 4", "electron 0.5"])
        assert "no saved events of electron 0.5" in _raises(
            lambda: _run(ctx, more, "-only", "energy_response", "-tot-mode", "none", from_saved=True),
            io_utils.ProductSetError)
        saved = ta.saved_events_csv(ctx, ds[0], e1, "none")
        with open(saved, "a", encoding="utf-8") as fh:
            fh.write("Electron_1GeV/e1.csv,999999,1.0,1,1,0,0,0,0.0,0,True\n")
        assert "changed since it was committed" in _raises(
            lambda: _run(ctx, ds, "-only", "energy_response", "-tot-mode", "none", from_saved=True),
            io_utils.ProductSetError)


# -output-tag without its own saved copy reads the untagged copy; the rebuild command repeats this run's flags
def test_output_tag_from_saved_and_rebuild_commands():
    assert ta._flag_value(1022.0) == "1022" and ta._flag_value([0.6, 4.0]) == "0.6 4"
    assert ta._flag_value("a b") == '"a b"' and ta._flag_value("\\\\?\\C:\\cfg\\my.yml") == "C:\\cfg\\my.yml"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp)
        _logged(lambda: _run(ctx, ds, "-only", "csv"))
        uncut = ta.energy_response_csv(ctx).read_bytes()
        _, text = _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-quality-threshold", "0.05",
                                       "-output-tag", "q005", from_saved=True))
        assert ta.energy_response_csv(ctx).read_bytes() == uncut and "reading e1_events.csv" in text
        cut = ta.energy_response_csv(ctx, "q005").read_bytes()
        _logged(lambda: _run(ctx, ds, "-only", "energy_response", "-quality-threshold", "0.05", "-output-tag",
                             "q005"))
        assert ta.energy_response_csv(ctx, "q005").read_bytes() == cut
        msg = _raises(lambda: _run(ctx, ds, "-only", "energy_response", "-output-tag", "q005", "-slope-window",
                                   "1.2", "4", from_saved=True), io_utils.ProductSetError)
        cmd = msg.split("Rebuild them with: ", 1)[1]
        assert cmd.startswith(".\\runner.ps1 adcpeak --electron 1 --electron 4 -slope-window 1.2 4 -output-tag "
                              "q005 -only energy_response"), cmd
        # a switch to the clean datasets is named, the command repeats -clean-input
        clean = {d.key: _event_csv(tmp / f"{d.key}_clean.csv", n_events=40, seed=21 + i) for i, d in enumerate(ds)}
        ctx.clean_input = True
        ctx.input_csv = lambda d, csv: clean[d.key]
        msg = _raises(lambda: _run(ctx, ds, "-only", "energy_response", from_saved=True), io_utils.ProductSetError)
        assert "the saved events were made from the beam csv, this run reads the clean datasets" in msg
        assert " -clean-input -only energy_response" in msg
        ctx.clean_input = False
        del ctx.input_csv
        ctx.cfg["_config_file"] = tmp / "my.yml"
        ctx.test, ctx.n_test_events = True, 2000
        assert ta._mode_flags(ctx) == f" -config {tmp / 'my.yml'} -test -test-events 2000"


# ----------------------------------------------------------------------------------------------
# --output (common/outputs.py)
# ----------------------------------------------------------------------------------------------
# plot functions replaced by stubs that write a marker and record (key, path)
@contextlib.contextmanager
def _stub_pdfs():
    drawn = []

    # stub of the fit pages
    def fits(path, table, details, hist, s, title):
        drawn.append(("line_fits", Path(path)))
        io_utils.ensure_writable(Path(path)).write_bytes(b"%PDF-fits")

    # stub of the response pages
    def pages(spectra_path, linres_path, resp, rs, title):
        for key, path in (("energy_spectra", spectra_path), ("linearity_resolution", linres_path)):
            if path is not None:
                drawn.append((key, Path(path)))
                io_utils.ensure_writable(Path(path)).write_bytes(f"%PDF-{key}".encode())

    with mock.patch.object(ta, "plot_fit_pages", side_effect=fits), \
            mock.patch.object(ta, "plot_response", side_effect=pages):
        yield drawn


# (key, path, saved) of the final-result items of an io_plan
def _final_items(plan) -> list[tuple]:
    return [(p["result"], Path(p["path"]), p["saved"]) for p in plan if p.get("result")]


# full set without --output: final results only in the scratch folder; the intermediates as always
def test_fullset_without_output_saves_no_final_result():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp, make_plots=True, outputs=None)
        scratch = tmp / "cache" / "tot_adc_peak" / "unsaved"
        with _stub_pdfs() as drawn:
            _, text = _logged(lambda: _run(ctx, ds))
        assert not (tmp / "results").exists()
        assert _names(scratch) == ["energy_response.csv", "energy_response.provenance.json", "energy_spectra.pdf",
                                   "linearity_resolution.pdf", "tot_adc_peak_fits.pdf"]
        assert [k for k, _ in drawn] == ["line_fits", "energy_spectra", "linearity_resolution"]
        assert text.count("final results not saved (no --output)") == 1
        assert ta.parameters_csv(ctx).exists() and ta.saved_fit_histograms(ctx, "pion_3GeV").exists()
        last = ta.last_run_results()
        assert list(last) == ["line_fits", "energy_response", "energy_spectra", "linearity_resolution"]
        assert last["energy_response"]["provenance"] == scratch / "energy_response.provenance.json"
        before, n_plans = _names(scratch), len(out_plans.plans())
        plan, _ = _logged(lambda: _plan(ctx, ds))
        assert _names(scratch) == before and len(out_plans.plans()) == n_plans
        assert [k for k, _, saved in _final_items(plan) if not saved] == list(last)
        with _stub_pdfs() as drawn:
            _logged(lambda: _run(ctx, ds, "-only", "line_fits", from_saved=True))
        assert drawn == [("line_fits", scratch / "tot_adc_peak_fits.pdf")] and _names(scratch) == [
            "tot_adc_peak_fits.pdf"]


# -output names matched by result name (default file name, key + extension, KEY=FILE); -test numbered names
def test_output_names_by_result_and_test_names():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp, make_plots=True, outputs=[])
        res = tmp / "results" / "fullset" / "tot_adc_peak"
        with _stub_pdfs() as drawn:
            _, text = _logged(lambda: _run(ctx, ds))
        assert _names(res) == ["energy_response.csv", "energy_response.provenance.json", "energy_spectra.pdf",
                               "linearity_resolution.pdf", "tot_adc_peak_fits.pdf"] and "WARNING" not in text
        ctx.outputs = ["energy_spectra=spec.pdf", "energy_response.csv"]
        with _stub_pdfs() as drawn:
            _, text = _logged(lambda: _run(ctx, ds, "-only", "response", from_saved=True))
        scratch = tmp / "cache" / "tot_adc_peak" / "unsaved"
        assert drawn == [("energy_spectra", res / "spec.pdf"),
                         ("linearity_resolution", scratch / "linearity_resolution.pdf")]
        ctx.outputs = ["linearity_resolution.pdf", "nothing.csv"]
        plan, text = _logged(lambda: _plan(ctx, ds, "-only", "linearity_resolution", from_saved=True))
        assert _final_items(plan) == [("linearity_resolution", res / "linearity_resolution.pdf", True)]
        assert "ignored: nothing.csv" in text
        ctx.outputs = None
        ctx.test = True
        top = tmp / "results" / "test"
        with _stub_pdfs() as drawn:
            _logged(lambda: _run(ctx, ds))
        assert drawn == [("line_fits", top / "test.pdf"), ("energy_spectra", top / "test1.pdf"),
                         ("linearity_resolution", top / "test2.pdf")]
        assert _names(top) == ["test.csv", "test.csv.provenance.json", "test.pdf", "test1.pdf", "test2.pdf"]


# -no-json: no record next to the line table and the response table; the cache records stay, -from-saved works
def test_no_json_keeps_only_the_cache_records():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ctx, ds = _chain(tmp, test=True, outputs=[])
        ctx.write_json = False
        io_utils.set_json_records(False, [tmp / "data", tmp / "calibration", tmp / "results"], [tmp / "cache"])
        try:
            _logged(lambda: _run(ctx, ds, "-only", "csv"))
            cal, res, cache = (tmp / d / "test" / "tot_adc_peak" for d in ("calibration", "results", "cache"))
            assert _names(cal) == ["tot_adc_peak_parameters.csv"] and _names(res) == ["energy_response.csv"]
            assert "e4a_events.provenance.json" in _names(cache / "Electron_4GeV")
            first = (res / "energy_response.csv").read_bytes()
            _logged(lambda: _run(ctx, ds, "-only", "energy_response", from_saved=True))
            assert (res / "energy_response.csv").read_bytes() == first
        finally:
            io_utils.set_json_records(True)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_") and callable(f):
            t = time.time()
            f()
            print("PASS", name, f"({time.time() - t:.1f}s)")
