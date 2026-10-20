"""Tests of the clean-dataset stage (ldmx_hcal.selection.clean_dataset), Lei's decision D8 of 2026-10-01.

Synthetic beam csv files and calibration tables in a temporary folder (about 1 minute):
  * definitions: end hit (adc_sum >= threshold column, finite pedestal / threshold, mpv > 0), E_MIP / E_MeV,
    both threshold columns read from the calibration table (never recomputed, no fallback);
  * muon standard (both ends hit; at most one kept bar per layer, layers may be missing; layer-1 bar with
    EACH end in [1.84, 7.83] MeV inclusive) and otherparticle standard (either end hit, no event cut);
  * one product per csv of a run, in a sub-folder per run (nothing merged);
  * output = byte-exact subset of the beam csv (header and lines copied, file order, LF / CRLF, last line
    without terminator, extra columns, any read-block size of pass 2), independent of chunk size and row
    order, compared with an independent pandas implementation; test mode == full run restricted to the
    first N events; a beam csv rewritten between the two passes is refused;
  * currency: product sets (sidecar last), current_clean_csv / RunContext.input_csv with -clean-input,
    stale after a beam csv change (also a same-size edit of a file hashed only at its edges), a
    calibration-value change or a yml change of the selection (flag choices stay accepted); current after
    a rewrite with identical values, and for a test subset after its rebuild or deletion (never rebuilt
    by the check);
  * Codex round 2: CLEAN2-01 (a failed commit never pairs a new csv with an old record), CLEAN2-02 (the
    calibration keys are validated by io_utils.validate_channel_keys before any beam csv is read), numeric
    settings checked before IO, temp files of a failed run removed;
  * -from-saved (the result units from the cache without reading the beam csv; refused with the reason
    and both the rebuild and the stored-selection command when anything it depends on changed), -only units,
    -no-json, --no-plots, dry-run plan (writes nothing), --output by result name, rebuild commands, runner flags;
  * --fit / --bin on the diagnostic per-end E_MIP spectrum fits: option parsing next to the
    selection and particle flags (PowerShell spelling via runner), refusals before any IO, default fits unchanged,
    one range == the same fits restricted (== the default with that mip_fit_range), --bin points on the page scale,
    two ranges = double Gaussian + Landau on the first range, failed fits = points only (also a range without
    data) + the legend line '<model> fit failed on [a, b]: <reason>' (also next to a fit that succeeded, also for
    a double Gaussian that range_gauss_fit fails), and the clean dataset / its record / cache / currency untouched
    while -from-saved redraws the pdf.
A test on the real 5000-event muon 4 subset (skipped when it is not on this machine) compares the stage with
the pandas reference on real data.

    python -X utf8 src/tests/run_all.py clean

Author: Lei
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402
from ldmx_hcal.common import geometry as geo  # noqa: E402
from ldmx_hcal.common import io_utils  # noqa: E402
from ldmx_hcal.common import outputs as outputs_mod  # noqa: E402
from ldmx_hcal.common.config import Dataset, RunContext, load_config, redirect_products, resolve_datasets  # noqa: E402
from ldmx_hcal.common.io_utils import read_json  # noqa: E402
from ldmx_hcal.selection import clean_dataset as cd  # noqa: E402

BEAM_HEADER = ["pf_event", "tot_end0", "toa_end0", "adc_sum_end0", "adc_mean_end0", "adc_max_end0", "layer", "strip",
               "pf_spill", "pf_ticks", "tot_end1", "toa_end1", "adc_sum_end1", "adc_mean_end1", "adc_max_end1"]
PED, MPV, SIGMA = 100.0, 100.0, 2.0     # synthetic calibration: E_MIP = (adc - 100)/100
THR_SCALE, THR_SIGMA = 1.2 * PED, PED + 5 * SIGMA      # 120 and 110 ADC
K = cd.DEFAULTS["mip_energy_mev"]


# --------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------
# temp folder of one test
def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="ldmx_clean_test_")


# All 384 channels: pedestal 100, sigma 2, mpv 100, thresholds 120 (1.2 x) and 110 (+5 sigma).
def _calib_frame(overrides: dict | None = None) -> pd.DataFrame:
    df = geo.all_channels()
    df["pedestal"] = PED
    df["pedestal_sigma"] = SIGMA
    df["pedestal_range_with_5sigma"] = THR_SIGMA
    df["pedestal_range_with_20percent"] = THR_SCALE
    df["mpv"] = MPV
    df["mpv_status"] = "OK"
    df["mpv_fit_status"] = "CONVERGED"
    for (layer, strip, end), cols in (overrides or {}).items():
        sel = (df["layer"] == layer) & (df["strip"] == strip) & (df["end"] == end)
        for col, value in cols.items():
            df.loc[sel, col] = value
    return df


# ChannelCalibration read from a written calibration table
def _calib(frame: pd.DataFrame | None = None, column: str = "pedestal_range_with_20percent") -> cd.ChannelCalibration:
    with _tmpdir() as tmp:
        path = Path(tmp) / "calibration_parameters.csv"
        (_calib_frame() if frame is None else frame).to_csv(path, index=False)
        return cd.read_calibration(path, column)


# one beam csv row of a bar (make-csv schema)
def _row(event, layer, strip, adc0=100, adc1=100, tot0=0, tot1=0, toa0=0.0, toa1=0.0) -> dict:
    return dict(pf_event=event, tot_end0=tot0, toa_end0=toa0, adc_sum_end0=adc0, adc_mean_end0=adc0 / 8,
                adc_max_end0=adc0 // 8, layer=layer, strip=strip, pf_spill=0, pf_ticks=0, tot_end1=tot1,
                toa_end1=toa1, adc_sum_end1=adc1, adc_mean_end1=adc1 / 8, adc_max_end1=adc1 // 8)


# adc_sum of an end with about ``mev`` MeV (E_MeV = (adc - 100)/100 * 4.83143).
def _adc(mev: float) -> int:
    return int(round(PED + mev / K * MPV))


# Beam csv in the make-csv schema; bar-major like the real files (rows of one event spread out).
def _write_beam(path: Path, rows: list[dict], bar_major: bool = True) -> Path:
    df = pd.DataFrame(rows, columns=BEAM_HEADER)
    keys = ["layer", "strip", "pf_event"] if bar_major else ["pf_event", "layer", "strip"]
    df.sort_values(keys, kind="stable").to_csv(path, index=False)
    return path


# One through-going muon: one bar per layer, both ends at about ``mev``.
def _muon_track(event, mev=4.8, layers=range(1, 20), strip=3) -> list[dict]:
    a = _adc(mev)
    return [_row(event, layer, strip, a, a, toa0=10.0, toa1=10.0) for layer in layers]


# Pedestal-only rows for the other bars (no zero suppression in the real files).
def _noise(event, layers=range(1, 20), skip_strip=3) -> list[dict]:
    return [_row(event, layer, s) for layer in layers for s in range(geo.n_strips(layer)) if s != skip_strip]


# Clean muons, double bars, one-ended hits, missing layers, showers, layer-1 window failures.
def _synthetic_run(n_events: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for ev in range(1, n_events + 1):
        kind = ev % 6
        if kind == 0:                                   # clean muon
            rows += _muon_track(ev, mev=float(rng.uniform(3.5, 6.0)))
        elif kind == 1:                                 # second kept bar in layer 15 -> muon veto
            rows += _muon_track(ev)
            rows.append(_row(ev, 15, 7, _adc(5.0), _adc(4.0)))
        elif kind == 2:                                 # shower: many bars, random signals
            for layer in range(1, 20):
                for s in range(geo.n_strips(layer)):
                    rows.append(_row(ev, layer, s, int(rng.integers(90, 900)), int(rng.integers(90, 900))))
        elif kind == 3:                                 # one-ended hits + a track in layers 1-9 only
            rows += _muon_track(ev, layers=range(1, 10))
            rows.append(_row(ev, 12, 5, _adc(5.0), 100))
            rows.append(_row(ev, 13, 5, 100, _adc(5.0)))
        elif kind == 4:                                 # low signal: layer-1 bar below the window
            rows += _muon_track(ev, mev=1.2)
        else:                                           # layer 1 missing
            rows += _muon_track(ev, layers=range(2, 20))
        rows += _noise(ev) if kind != 2 else []
    return rows


# --------------------------------------------------------------------------
# a run context in a temporary product tree
# --------------------------------------------------------------------------
# RunContext with every product tree (data, calibration, results, cache) inside tmp
def _ctx(tmp: Path, test: bool = False, make_plots: bool = False, n_test_events: int = 5000, chunksize: int = 1000,
         section: dict | None = None, outputs: tuple | list | None = (), write_json: bool = True) -> RunContext:
    # Every product tree (data, calibration, results AND cache) inside ``tmp`` (config.redirect_products).
    #
    # ``outputs`` = ctx.outputs (the --output names): () = --output alone (the default here, so the tests find the
    # final results under their default names), None = no --output.
    cfg = redirect_products(load_config(), tmp)
    cfg["run"] = {**cfg.get("run", {}), "chunksize": chunksize}
    cfg.pop(cd.CONFIG_SECTION, None)
    if section is not None:
        cfg[cd.CONFIG_SECTION] = section
    return RunContext(cfg=cfg, test=test, make_plots=make_plots, n_test_events=n_test_events,
                      outputs=None if outputs is None else list(outputs), write_json=write_json)


# Write the calibration table where the stage reads it (the code's own table of this mode).
def _standard_calib(ctx: RunContext, frame: pd.DataFrame | None = None) -> Path:
    path = ctx.calibration_product()
    path.parent.mkdir(parents=True, exist_ok=True)
    (_calib_frame() if frame is None else frame).to_csv(path, index=False)
    return path


# A beam csv in the run folder tmp/runs/<folder>/ (one run = the folder of its csv files).
def _run_beam(tmp: Path, rows: list[dict], folder: str = "Muon_4GeV", name: str = "beam.csv",
              bar_major: bool = True) -> Path:
    path = tmp / "runs" / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_beam(path, rows, bar_major=bar_major)


# The run whose folder holds ``csv`` (an absolute run folder).
def _dataset(csv: Path, key: str = "muon_4GeV", particle: str = "muon") -> Dataset:
    return Dataset(key=key, particle=particle, energy_gev=4.0, label=f"synthetic {key}", kind="data",
                   folder=str(Path(csv).parent))


# The (first) csv of a run folder.
def _csv(ds: Dataset) -> Path:
    return sorted(Path(ds.folder).glob("*.csv"))[0]


# clean arguments parsed with the common runner options
def _args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    runner.add_common(p)
    cd.add_arguments(p)
    return p.parse_args(argv)


# call fn silently; returns (result, console text)
def _quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


# Run fn and require ``exc`` (whose text contains ``contains``); return the message.
def _refused(fn, *a, contains: str = "", exc=SystemExit, **k) -> str:
    try:
        _quiet(fn, *a, **k)
    except exc as e:
        text = str(e)
        assert contains in text, f"expected {contains!r} in: {text}"
        return text
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# cd.run as the runner calls it: -only sets ctx.units.
def _run(ctx, argv, datasets):
    args = _args(argv)
    ctx.units = outputs_mod.selected_units(args)
    with patch.object(RunContext, "_made_test_csv", return_value=None):
        return _quiet(cd.run, ctx, args, datasets)


# lines of a file, with their line endings
def _lines(path: Path) -> list[bytes]:
    return Path(path).read_bytes().splitlines(keepends=True)


# Independent pandas implementation of D8: the data-row indices (0 = first line after the header).
def _reference_rows(beam_csv: Path, calib: pd.DataFrame, standard: str, column: str,
                    window=(1.84, 7.83), k: float = K) -> list[int]:
    beam = pd.read_csv(beam_csv, usecols=cd.BEAM_COLUMNS)
    cal = calib.set_index(["layer", "strip", "end"])
    hits, e_mev = [], []
    for end in (0, 1):
        c = cal.xs(end, level="end")[["pedestal", "mpv", column]]
        m = beam[["layer", "strip"]].merge(c, left_on=["layer", "strip"], right_index=True, how="left")
        adc = beam[f"adc_sum_end{end}"].astype(float)
        ok = m["pedestal"].notna() & m[column].notna() & (m["mpv"] > 0)
        hits.append(ok & (adc >= m[column]))
        e_mev.append((adc - m["pedestal"]) / m["mpv"] * k)
    kept = (hits[0] & hits[1]) if standard == "muon" else (hits[0] | hits[1])
    kb = beam[kept]
    if standard == "muon":
        n = kb.groupby(["pf_event", "layer"]).size()
        vetoed = set(n[n >= 2].index.get_level_values(0))
        e0, e1 = e_mev[0][kept], e_mev[1][kept]
        l1 = kb[(kb["layer"] == 1) & e0.between(*window) & e1.between(*window)]
        good = set(l1["pf_event"]) - vetoed
        kb = kb[kb["pf_event"].isin(good)]
    return sorted(int(i) for i in kb.index)


# header plus the given data rows of the beam csv, byte for byte
def _expected_lines(beam_csv: Path, rows: list[int]) -> list[bytes]:
    lines = _lines(beam_csv)
    return [lines[0]] + [lines[1 + r] for r in rows]


# --------------------------------------------------------------------------
# definitions
# --------------------------------------------------------------------------
# product paths follow the layout
def test_product_paths_follow_the_layout():
    # One product per input csv, in a sub-folder per run: data/[test/]clean_dataset/<run>/<stem>_clean.csv,
    # results/<mode>/clean_dataset/<run>/..., cache/[test/]clean_dataset/<run>/...; one result slot per unit and csv.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        a = _run_beam(tmp, [], name="a.csv")
        b = _run_beam(tmp, [], name="b.csv")
        ds = _dataset(a)
        for test, mode in ((False, "fullset"), (True, "test")):
            ctx = _ctx(tmp, test=test)
            assert ctx.run_csvs(ds) == [a, b] and cd.run_csvs(ctx, ds) == [a, b]
            sub = "test" if test else ""      # data / cache: no fullset folder; results keep the mode folder
            want = tmp / "data" / sub / "clean_dataset" / "Muon_4GeV" / "a_clean.csv"
            assert cd.clean_csv(ctx, ds, a) == ctx.clean_product(ds, a) == want
            assert cd.clean_csv(ctx, ds, b).name == "b_clean.csv"
            assert cd.provenance_json(ctx, ds, a).name == "a_clean.provenance.json"
            assert cd.provenance_json(ctx, ds, a).parent == cd.clean_csv(ctx, ds, a).parent
            for path in (cd.cutflow_csv(ctx, ds, a), cd.layers_csv(ctx, ds, a), cd.diagnostics_pdf(ctx, ds, a),
                         cd.results_json(ctx, ds, a)):
                assert path.parent == tmp / "results" / mode / "clean_dataset" / "Muon_4GeV", path
            assert [p.name for p in (cd.cutflow_csv(ctx, ds, a), cd.layers_csv(ctx, ds, a),
                                     cd.diagnostics_pdf(ctx, ds, a), cd.results_json(ctx, ds, a))] == [
                "a_clean_cutflow.csv", "a_clean_layers.csv", "a_clean_diagnostics.pdf", "a_clean_results.json"]
            for path in (cd.cache_npz(ctx, ds, a), cd.cache_json(ctx, ds, a)):
                assert path.parent == tmp / "cache" / sub / "clean_dataset" / "Muon_4GeV", path
            job = cd.Job(ds, "muon", "", "pedestalscale", "", {})
            slots = cd.result_slots(ctx, [job], cd.UNITS)
            assert [(s.key, s.kind, s.default) for s in slots[:4]] == [
                ("Muon_4GeV.a.cutflow", "csv", cd.cutflow_csv(ctx, ds, a)),
                ("Muon_4GeV.a.layers", "csv", cd.layers_csv(ctx, ds, a)),
                ("Muon_4GeV.a.diagnostics", "pdf", cd.diagnostics_pdf(ctx, ds, a)),
                ("Muon_4GeV.a.results", "json", cd.results_json(ctx, ds, a))]
            assert [s.key for s in slots[4:]] == [f"Muon_4GeV.b.{u}" for u in ("cutflow", "layers", "diagnostics",
                                                                               "results")]
            assert [s.key for s in cd.result_slots(ctx, [job], ("layers",))] == ["Muon_4GeV.a.layers",
                                                                               "Muon_4GeV.a.results",
                                                                               "Muon_4GeV.b.layers",
                                                                               "Muon_4GeV.b.results"]
            assert cd.result_slots(ctx, [job], ("clean",)) == []
        assert cd.DEFAULT_DATASETS == ["all"] and not hasattr(cd, "PARTICLES")
        assert cd.SUPPORTS_FROM_SAVED is True and cd.SUPPORTS_CLEAN_INPUT is False
        assert cd.UNITS == ("clean", "cutflow", "layers", "diagnostics")


# hit = adc_sum >= threshold (inclusive) with finite pedestal / threshold and mpv > 0; E_MIP = (adc-ped)/mpv.
def test_end_hit_and_energy_definition():
    adc = np.array([120, 119, 120, 120, 120, 120, 300])
    ped = np.array([100, 100, np.nan, 100, 100, 100, 100.0])
    mpv = np.array([100, 100, 100, np.nan, 0.0, -999, 50.0])
    thr = np.array([120, 120, 120, 120, 120, 120, np.nan])
    hit, e_mip = cd.end_quantities(adc, ped, mpv, thr)
    assert hit.tolist() == [True, False, False, False, False, False, False]
    np.testing.assert_array_equal(e_mip[[0, 1, 6]], [0.2, 0.19, 4.0])
    assert np.isnan(e_mip[2:6]).all()                          # no usable pedestal / mpv -> no energy
    assert cd.bar_kept(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool), "muon").tolist() == \
        [True, False, False, False]
    assert cd.bar_kept(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool), "otherparticle").tolist() == \
        [True, True, True, False]


# Both thresholds are columns of the table (here deliberately NOT 1.2 x ped / ped + 5 sigma), no fallback.
def test_threshold_modes_read_the_calibration_columns():
    frame = _calib_frame()
    frame["pedestal_range_with_20percent"] = 150.0             # not 1.2 * 100
    frame["pedestal_range_with_5sigma"] = 130.0                # not 100 + 5 * 2
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        path = tmp / "cal.csv"
        frame.to_csv(path, index=False)
        scale = cd.read_calibration(path, "pedestal_range_with_20percent")
        sigma = cd.read_calibration(path, "pedestal_range_with_5sigma")
        assert np.nanmax(scale.threshold) == 150.0 and np.nanmax(sigma.threshold) == 130.0
        assert scale.values_sha256 != sigma.values_sha256
        beam = _write_beam(tmp / "beam.csv", [_row(1, 1, 0, 140, 140), _row(1, 2, 0, 150, 150),
                                              _row(1, 3, 0, 129, 135)])
        for calib, expected in ((scale, [(2, 0)]), (sigma, [(1, 0), (2, 0)])):
            bars, counts = cd.scan_beam_csv(beam, calib, "muon", 100)
            assert sorted(zip(bars.layer.tolist(), bars.strip.tolist())) == expected
        frame.drop(columns=["pedestal_range_with_5sigma"]).to_csv(path, index=False)
        _refused(cd.read_calibration, path, "pedestal_range_with_5sigma", contains="pedestal_range_with_5sigma")
    assert cd.THRESHOLD_COLUMNS == {"pedestalscale": "pedestal_range_with_20percent",
                                    "pedestalsigma": "pedestal_range_with_5sigma"}


# HitBars from (event, layer, strip, e_mip0, e_mip1) tuples with both ends hit.
def _bars(rows: list[tuple]) -> cd.HitBars:
    a = np.array(rows, dtype=float).reshape(-1, 5)
    n = len(a)
    return cd.HitBars(row=np.arange(n, dtype=np.int64), pf_event=a[:, 0].astype(np.int64),
                      layer=a[:, 1].astype(np.int8), strip=a[:, 2].astype(np.int8),
                      hit_end0=np.ones(n, bool), hit_end1=np.ones(n, bool), e_mip_end0=a[:, 3],
                      e_mip_end1=a[:, 4], in_clean=np.zeros(n, bool))


# muon event rules
def test_muon_event_rules():
    # At most one kept bar per layer (missing layers allowed); layer-1 bar with EACH end in the window
    # (inclusive edges).
    lo_mip, hi_mip = 0.5, 1.5
    settings = {**cd.DEFAULTS, "layer1_energy_window_mev": [lo_mip * K, hi_mip * K]}   # exact float edges
    sel = cd.selection_definition("muon", "pedestalscale", settings)
    rows = []
    rows += [(1, layer, 3, 1.0, 1.0) for layer in range(1, 20)]                 # good, every layer
    rows += [(2, layer, 3, 1.0, 1.0) for layer in (1, 4, 9)]                    # good, layers missing
    rows += [(3, layer, 3, 1.0, 1.0) for layer in range(1, 20)] + [(3, 15, 7, 1.0, 1.0)]   # two bars in layer 15
    rows += [(4, 1, 3, 1.0, 1.0), (4, 1, 4, 1.0, 1.0)]                          # two bars in layer 1
    rows += [(5, layer, 3, 1.0, 1.0) for layer in range(2, 20)]                 # no layer-1 bar
    rows += [(6, 1, 3, lo_mip, hi_mip), (6, 2, 3, 1.0, 1.0)]                    # both edges: inclusive
    rows += [(7, 1, 3, lo_mip - 1e-12, 1.0)]                                    # end 0 just below
    rows += [(8, 1, 3, 1.0, hi_mip + 1e-12)]                                    # end 1 just above
    rows += [(9, 1, 3, 1.0, 3.0)]                                               # only one end inside
    rows += [(10, 1, 3, np.nan, 1.0)]                                           # no energy on one end
    bars = _bars(rows)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(bars))                                           # the file order does not matter
    for name in cd._BAR_DTYPES:
        setattr(bars, name, getattr(bars, name)[perm])
    steps = cd.select_events(bars, "muon", sel)
    kept = sorted(set(bars.pf_event[bars.in_clean].tolist()))
    assert kept == [1, 2, 6], kept
    assert [s["step"] for s in steps] == ["bars_both_ends_hit", "at_most_one_bar_per_layer",
                                          "layer1_bar_in_energy_window"]
    assert steps[0]["events"] == 10 and steps[1]["events"] == 8 and steps[2]["events"] == 3
    assert steps[2]["rows"] == 19 + 3 + 2 == int(bars.in_clean.sum())
    empty = _bars([])
    assert [s["rows"] for s in cd.select_events(empty, "muon", sel)] == [0, 0, 0]


# otherparticle: a bar is kept when EITHER end is a hit, no event cut; muon needs both ends.
def test_otherparticle_rules_and_row_flow():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        rows = [_row(1, 1, 0, 200, 100), _row(1, 2, 0, 100, 200), _row(1, 3, 0, 100, 100),     # one-ended hits
                _row(2, 1, 0, 100, 100),                                                       # nothing
                _row(3, 5, 1, 300, 300), _row(3, 5, 2, 300, 300), _row(3, 1, 0, _adc(4), _adc(4))]
        beam = _write_beam(tmp / "beam.csv", rows)
        calib = _calib()
        bars, counts = cd.scan_beam_csv(beam, calib, "otherparticle", 2)
        steps = cd.select_events(bars, "otherparticle", cd.selection_definition("otherparticle", "pedestalscale",
                                                                                cd.DEFAULTS))
        assert counts["input_rows"] == 7 and counts["input_events"] == 3
        assert counts["hit_end0"] == 4 and counts["hit_end1"] == 4
        assert len(bars) == 5 and bars.in_clean.all() and steps[-1]["events"] == 2
        bars, _ = cd.scan_beam_csv(beam, calib, "muon", 2)
        steps = cd.select_events(bars, "muon", cd.selection_definition("muon", "pedestalscale", cd.DEFAULTS))
        assert len(bars) == 3 and steps[1]["events"] == 0 and not bars.in_clean.any()   # layer 5 holds two bars
    assert cd.selection_definition("otherparticle", "pedestalsigma", cd.DEFAULTS) == {
        "standard": "otherparticle", "threshold_mode": "pedestalsigma",
        "threshold_column": "pedestal_range_with_5sigma"}
    assert cd.selection_definition("muon", "pedestalscale", cd.DEFAULTS)["layer1_energy_window_mev"] == [1.84, 7.83]


# --------------------------------------------------------------------------
# the output file
# --------------------------------------------------------------------------
# output is a byte exact subset and matches the reference
def test_output_is_a_byte_exact_subset_and_matches_the_reference():
    # Header + selected lines copied byte for byte in file order; the rows equal an independent pandas
    # implementation for both standards and both thresholds, for any chunk size and row order.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        frame = _calib_frame({(1, 5, 0): {"mpv": np.nan}, (5, 2, 1): {"pedestal_range_with_20percent": np.nan},
                              (7, 6, 0): {"pedestal_range_with_5sigma": np.nan}})
        calib_path = tmp / "cal.csv"
        frame.to_csv(calib_path, index=False)
        rows = _synthetic_run(36)
        for bar_major in (True, False):
            beam = _write_beam(tmp / f"beam_{bar_major}.csv", rows, bar_major=bar_major)
            for standard in ("muon", "otherparticle"):
                for mode, column in cd.THRESHOLD_COLUMNS.items():
                    calib = cd.read_calibration(calib_path, column)
                    sel = cd.selection_definition(standard, mode, cd.DEFAULTS)
                    expected = _reference_rows(beam, frame, standard, column)
                    outputs = []
                    for chunksize in (97, 100_000):                  # 97: events spread over many chunks
                        bars, counts = cd.scan_beam_csv(beam, calib, standard, chunksize)
                        cd.select_events(bars, standard, sel)
                        assert bars.row[bars.in_clean].tolist() == expected, (bar_major, standard, mode, chunksize)
                        out = tmp / f"out_{chunksize}.csv"
                        keep = bars.in_clean
                        info = cd.copy_selected_rows(beam, out, bars.row[keep],
                                                     (bars.pf_event[keep], bars.layer[keep], bars.strip[keep]),
                                                     counts["input_rows"])
                        assert info["rows"] == len(expected)
                        outputs.append(out.read_bytes())
                    assert outputs[0] == outputs[1]
                    assert _lines(tmp / "out_97.csv") == _expected_lines(beam, expected)
                    assert len(expected) > 0
        # the muon selection keeps the clean muons and the track in layers 1-9; it drops double bars,
        # low layer-1 energies, a missing layer 1 and every noise row
        beam = _write_beam(tmp / "beam.csv", rows)
        kept = pd.read_csv(beam).iloc[_reference_rows(beam, _calib_frame(), "muon", "pedestal_range_with_20percent")]
        assert set(kept["pf_event"] % 6) == {0, 3} and (kept["strip"] == 3).all()


# copy keeps the text and line endings
def test_copy_keeps_the_text_and_line_endings():
    # Extra columns, unusual number spellings, CRLF and a last line without terminator survive byte for byte;
    # a file that changed between the two passes is refused.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        header = "pf_event,layer,strip,adc_sum_end0,adc_sum_end1,toa_end0,raw_tot_sum_end0,note_x"
        body = ["1,1,3,600,600,1.50000,0,a", "1,2,3,100,100,0.0,0,b", "2,1,3,600,600,7,12,c",
                "2,1,4,600,600,1e1,0,d", "3,1,3,600,600,0,0,e"]
        calib = _calib()
        for eol, final in (("\n", "\n"), ("\r\n", "\r\n"), ("\n", ""), ("\r\n", "")):
            src = tmp / "beam.csv"
            src.write_bytes((eol.join([header] + body) + final).encode())
            bars, counts = cd.scan_beam_csv(src, calib, "otherparticle", 2)
            cd.select_events(bars, "otherparticle", {})
            keep = bars.in_clean
            out = tmp / "out.csv"
            cd.copy_selected_rows(src, out, bars.row[keep], (bars.pf_event[keep], bars.layer[keep], bars.strip[keep]),
                                  counts["input_rows"])
            want = [header] + [body[i] for i in (0, 2, 3, 4)]
            assert out.read_bytes() == (eol.join(want) + eol).encode(), (eol, final)
        # blank lines are skipped by both passes
        src.write_bytes(("\n".join([header, body[0], "", body[2]]) + "\n\n").encode())
        bars, counts = cd.scan_beam_csv(src, calib, "otherparticle", 2)
        keep = np.ones(len(bars), bool)
        cd.copy_selected_rows(src, out, bars.row, (bars.pf_event, bars.layer, bars.strip), counts["input_rows"])
        assert out.read_bytes() == ("\n".join([header, body[0], body[2]]) + "\n").encode()
        # the file changed between the passes: another row count, or another line at a selected index
        _refused(cd.copy_selected_rows, src, out, bars.row[keep], (bars.pf_event, bars.layer, bars.strip),
                 counts["input_rows"] + 1, exc=RuntimeError, contains="changed while it was read")
        _refused(cd.copy_selected_rows, src, out, bars.row, (bars.pf_event + 1, bars.layer, bars.strip),
                 counts["input_rows"], exc=RuntimeError, contains="changed while it was read")


# A few hundred bytes per event: kept and vetoed muons, a missing layer 1, noise (pass-2 block tests).
def _small_beam_rows() -> list[dict]:
    return (_muon_track(1) + _muon_track(2, mev=1.2) + _muon_track(3, layers=range(2, 20)) + _muon_track(4)
            + [_row(4, 15, 7, _adc(5.0), _adc(4.0))] + _noise(5, layers=(1, 2)) + _muon_track(6, layers=(1, 7, 19))
            + [_row(7, 3, 2, _adc(5.0), 100)])


# copy with any read block size
def test_copy_with_any_read_block_size():
    # Pass 2 reads in blocks (COPY_BLOCK_BYTES = 16 MiB); lines that span blocks are carried over.  With blocks
    # of 1, 2, 3, 7 and 61 bytes the output equals the one-block output and the reference, for LF / CRLF, blank
    # lines and a last line without terminator.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        frame = _calib_frame()
        calib = _calib(frame)
        text = _write_beam(tmp / "base.csv", _small_beam_rows(), bar_major=False).read_bytes().replace(b"\r\n", b"\n")
        header, *body = text.split(b"\n")[:-1]
        rng = np.random.default_rng(5)
        n_checked = 0
        for eol in (b"\n", b"\r\n"):
            for final in (True, False):
                for blanks in (False, True):
                    lines = list(body)
                    if blanks:
                        for i in sorted(rng.choice(np.arange(1, len(lines)), 6, replace=False), reverse=True):
                            lines.insert(int(i), b"")
                    data = eol.join([header] + lines) + (eol + (eol if blanks else b"") if final else b"")
                    src = tmp / "beam.csv"
                    src.write_bytes(data)
                    nonblank = [ln for ln in lines if ln]
                    for standard in ("muon", "otherparticle"):
                        bars, counts = cd.scan_beam_csv(src, calib, standard, 50)
                        cd.select_events(bars, standard, cd.selection_definition(standard, "pedestalscale",
                                                                                 cd.DEFAULTS))
                        keep = bars.in_clean
                        rows = _reference_rows(src, frame, standard, "pedestal_range_with_20percent")
                        assert bars.row[keep].tolist() == rows and rows
                        want = header + eol + b"".join(nonblank[r] + eol for r in rows)
                        for block in (1, 2, 3, 7, 61, cd.COPY_BLOCK_BYTES):
                            out = tmp / "out.csv"
                            with patch.object(cd, "COPY_BLOCK_BYTES", block):
                                cd.copy_selected_rows(src, out, bars.row[keep], (bars.pf_event[keep],
                                                      bars.layer[keep], bars.strip[keep]), counts["input_rows"])
                            assert out.read_bytes() == want, (eol, final, blanks, standard, block)
                            n_checked += 1
        assert n_checked == 2 * 2 * 2 * 2 * 6


# beam csv rewritten between the passes is refused
def test_beam_csv_rewritten_between_the_passes_is_refused():
    # A beam csv rewritten after pass 1 with the same keys and row count but other adc values passes the
    # line checks of pass 2; the size / modification-time check after pass 2 refuses it, and nothing is
    # committed.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(12))
        ds = _dataset(src)
        ctx = _ctx(tmp)
        _standard_calib(ctx)
        real = cd.select_events

        # real pass 1, then the beam csv is rewritten with other adc values
        def rewrite_after_pass_1(bars, standard, selection):
            steps = real(bars, standard, selection)
            st = src.stat()
            df = pd.read_csv(src)
            df["adc_sum_end1"] += 1                               # same keys, same rows, other energies
            df.to_csv(src, index=False)
            os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
            return steps

        with patch.object(cd, "select_events", side_effect=rewrite_after_pass_1), io_utils.temp_registry():
            _refused(cd.run, ctx, _args([]), [ds], exc=io_utils.InputError,
                     contains="was rewritten while this stage read it")
        folder = cd.clean_csv(ctx, ds, src).parent
        assert not cd.clean_csv(ctx, ds, src).exists() and not cd.provenance_json(ctx, ds, src).exists()
        names = [p.name for p in folder.iterdir()] if folder.exists() else []
        assert not [n for n in names if ".tmp" in n or ".old-" in n], names


# test subset equals full output restricted
def test_test_subset_equals_full_output_restricted():
    # Every decision is event-local: the clean dataset of the first N events (test mode) is the full clean
    # dataset restricted to those events, line for line.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        full_csv = _run_beam(tmp, _synthetic_run(36))
        ds = _dataset(full_csv)
        full = _ctx(tmp, test=False)
        _standard_calib(full)
        _run(full, [], [ds])
        test = _ctx(tmp, test=True, n_test_events=13)   # no test table: the full-set table
        _run(test, [], [ds])
        full_lines, test_lines = _lines(cd.clean_csv(full, ds, full_csv)), _lines(cd.clean_csv(test, ds, full_csv))
        first = sorted(pd.read_csv(full_csv)["pf_event"].unique())[:13]
        restricted = [full_lines[0]] + [ln for ln in full_lines[1:] if int(ln.split(b",")[0]) in set(first)]
        assert test_lines == restricted and len(test_lines) > 1
        assert read_json(cd.provenance_json(test, ds, full_csv))["input"]["path"] == \
            io_utils.canonical_path(test.test_subset_csv(ds, full_csv))


# --------------------------------------------------------------------------
# run, product sets and currency
# --------------------------------------------------------------------------
# run writes product sets and the clean input closes
def test_run_writes_product_sets_and_the_clean_input_closes():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        beam = _run_beam(tmp, _synthetic_run(30))
        ds = _dataset(beam)
        ctx = _ctx(tmp, make_plots=True)
        calib_path = _standard_calib(ctx)
        _, out = _run(ctx, [], [ds])
        assert "standard muon [default for muon runs" in out and "pedestalscale" in out
        record = read_json(cd.provenance_json(ctx, ds, beam))
        assert record["protocol"] == io_utils.PRODUCT_SET_PROTOCOL
        assert list(record["members"]) == ["beam_clean.csv"] and record["csv"] == "Muon_4GeV/beam.csv"
        assert record["selection"] == {"standard": "muon", "threshold_mode": "pedestalscale",
                                       "threshold_column": "pedestal_range_with_20percent",
                                       "mip_energy_mev": K, "layer1_energy_window_mev": [1.84, 7.83]}
        assert record["input"] == record["input_file"] == cd.file_identity(beam) == cd.beam_identity(ctx, ds, beam)
        assert record["input"]["hash"] == "full" and "mtime_ns" not in record["input"]     # small file: full sha256
        assert record["selection_flags"] == {"standard": None, "threshold_mode": None}
        assert record["calibration"]["values_digest_of"] == ["pedestal", "mpv", "pedestal_range_with_20percent"]
        assert record["counts"]["output_rows"] == len(_lines(cd.clean_csv(ctx, ds, beam))) - 1
        assert [c["step"] for c in record["cutflow"]] == ["input", "bars_both_ends_hit", "at_most_one_bar_per_layer",
                                                          "layer1_bar_in_energy_window"]
        flow = pd.read_csv(cd.cutflow_csv(ctx, ds, beam))
        assert flow["rows"].tolist() == [c["rows"] for c in record["cutflow"]]
        results = read_json(cd.results_json(ctx, ds, beam))
        assert results["clean_csv"]["sha256"] == record["members"]["beam_clean.csv"]["sha256"]
        assert sorted(results["members"]) == ["beam_clean_cutflow.csv", "beam_clean_diagnostics.pdf",
                                              "beam_clean_layers.csv"]
        assert results["results"] == {"cutflow": "beam_clean_cutflow.csv", "layers": "beam_clean_layers.csv",
                                      "diagnostics": "beam_clean_diagnostics.pdf"} and results["saved"] is True
        assert io_utils.product_set_valid(cd.cache_json(ctx, ds, beam))[0]
        cache = read_json(cd.cache_json(ctx, ds, beam))
        assert cache["clean_csv"]["sha256"] == results["clean_csv"]["sha256"] and cache["cutflow"] == record["cutflow"]
        cd._REPORTED.clear()
        path, out = _quiet(cd.current_clean_csv, ctx, ds, beam)
        assert path == cd.clean_csv(ctx, ds, beam)
        assert "clean input Muon_4GeV/beam.csv: beam_clean.csv (standard muon" in out
        ctx.clean_input = True                                    # -clean-input: RunContext.input_csv reads it
        assert _quiet(ctx.input_csv, ds, beam)[0] == cd.clean_csv(ctx, ds, beam)
        ctx.clean_input = False
        # a rewrite of the calibration with the same values as parsed (other row order, an extra column, another
        # spelling of exactly representable numbers) keeps it current
        frame = _calib_frame().sample(frac=1.0, random_state=1)
        frame["comment"] = "rewritten"
        frame.to_csv(calib_path, index=False, float_format="%.4f")
        assert cd.current_clean_csv(ctx, ds, beam) is not None
        # other values -> stale, with the reason
        _calib_frame({(1, 3, 0): {"mpv": 90.0}}).to_csv(calib_path, index=False)
        ok, reason, _ = cd.clean_status(ctx, ds, beam)
        assert not ok and "other pedestal / mpv / pedestal_range_with_20percent values" in reason
        assert cd.current_clean_csv(ctx, ds, beam) is None
        _refused(lambda: setattr(ctx, "clean_input", True) or ctx.input_csv(ds, beam), contains="clean --muon 4")
        ctx.clean_input = False
        _standard_calib(ctx)
        assert cd.current_clean_csv(ctx, ds, beam) is not None
        # a change of the threshold column alone (the one this dataset used) makes it stale as well
        _standard_calib(ctx, _calib_frame({(19, 11, 1): {"pedestal_range_with_20percent": 121.0}}))
        assert cd.current_clean_csv(ctx, ds, beam) is None
        _standard_calib(ctx, _calib_frame({(19, 11, 1): {"pedestal_range_with_5sigma": 111.0}}))  # not used
        assert cd.current_clean_csv(ctx, ds, beam) is not None
        # another beam csv -> stale; the clean csv edited by hand -> stale
        _write_beam(beam, _synthetic_run(31))
        assert "beam csv differs" in cd.clean_status(ctx, ds, beam)[1]
        _write_beam(beam, _synthetic_run(30))
        assert cd.current_clean_csv(ctx, ds, beam) is not None          # same content again: current
        with open(cd.clean_csv(ctx, ds, beam), "ab") as fh:
            fh.write(b"\n")
        assert "changed since it was committed" in cd.clean_status(ctx, ds, beam)[1]
        # the plan of a consumer shows it as a stale product of 'clean'
        ctx.clean_input, ctx.dry_run = True, True
        item = ctx.input_plan_item(ds, beam)
        assert item["kind"] == "product" and item.get("stale") is True and "clean --muon 4" in item["produced_by"]


# one product per csv of a run
def test_one_product_per_csv_of_a_run():
    # A run of two csv files gives two clean csv files (each a byte-exact subset of its own beam csv), two caches
    # and two sets of results, one to one; nothing is merged.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        a = _run_beam(tmp, _synthetic_run(18, seed=1), name="a.csv")
        b = _run_beam(tmp, _synthetic_run(12, seed=2), name="b.csv")
        ds = _dataset(a)
        ctx = _ctx(tmp)
        frame = _calib_frame()
        _standard_calib(ctx, frame)
        _, out = _run(ctx, [], [ds])
        assert "2 csv file(s)" in out
        for src in (a, b):
            rows = _reference_rows(src, frame, "muon", "pedestal_range_with_20percent")
            assert _lines(cd.clean_csv(ctx, ds, src)) == _expected_lines(src, rows)
            assert read_json(cd.provenance_json(ctx, ds, src))["csv"] == f"Muon_4GeV/{src.name}"
            assert io_utils.product_set_valid(cd.cache_json(ctx, ds, src), members=[cd.cache_npz(ctx, ds, src)])[0]
            assert cd.cutflow_csv(ctx, ds, src).exists() and cd.layers_csv(ctx, ds, src).exists()
            assert read_json(cd.results_json(ctx, ds, src))["csv"] == f"Muon_4GeV/{src.name}"
            assert cd.current_clean_csv(ctx, ds, src) == cd.clean_csv(ctx, ds, src)
        folder = cd.clean_csv(ctx, ds, a).parent
        assert sorted(p.name for p in folder.iterdir()) == ["a_clean.csv", "a_clean.provenance.json",
                                                            "b_clean.csv", "b_clean.provenance.json"]
        # a run folder without any csv stops before anything is written
        empty = Dataset(key="pion_4GeV", particle="pion", energy_gev=4.0, label="empty", folder=str(tmp / "none"))
        _refused(cd.run, ctx, _args([]), [empty], exc=io_utils.InputError, contains="no csv file in")


# test subset identity is what it is copied from
def test_test_subset_identity_is_what_it_is_copied_from():
    # A test subset (cache/test/test_subsets) is identified by its full csv (path, size, modification time),
    # N and the builder version: the clean dataset stays current when the subset is rebuilt or deleted (cache/
    # deleted), in a plan and in a run alike, and the check never rebuilds the subset; another N or a changed
    # full csv makes it stale.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(30))
        ds = _dataset(src)
        ctx = _ctx(tmp, test=True, n_test_events=12)
        _standard_calib(ctx)
        _run(ctx, [], [ds])
        subset = ctx.test_subset_csv(ds, src)
        record = read_json(cd.provenance_json(ctx, ds, src))
        assert record["input"] == {"kind": "test-subset", "path": io_utils.canonical_path(subset),
                                   "builder": io_utils.TEST_SUBSET_BUILDER,
                                   "source": io_utils.file_signature(src), "n_events_requested": 12}
        assert record["input_file"] == cd.file_identity(subset)
        with patch.object(RunContext, "_made_test_csv", return_value=None), \
                patch.object(io_utils, "build_test_subset", side_effect=AssertionError("subset rebuilt")):
            assert cd.current_clean_csv(ctx, ds, src) is not None
            os.remove(subset)
            subset.with_name(subset.stem + ".provenance.json").unlink()
            for dry_run in (True, False):
                ctx.dry_run = dry_run
                assert cd.clean_status(ctx, ds, src)[:2] == (True, "ok"), dry_run
            ctx.dry_run, ctx.clean_input = False, True
            assert _quiet(ctx.input_csv, ds, src)[0] == cd.clean_csv(ctx, ds, src) and not subset.exists()
            ctx.clean_input = False
            other = _ctx(tmp, test=True, n_test_events=11)
            assert _quiet(cd.current_clean_csv, other, ds, src)[0] is None
            st = src.stat()                                       # the full csv touched: the subset would differ
            os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
            ok, reason, _ = cd.clean_status(ctx, ds, src)
            assert not ok and "beam csv differs" in reason
        _run(ctx, [], [ds])                                       # a run rebuilds the subset and is current again
        assert subset.exists() and cd.current_clean_csv(ctx, ds, src) is not None


# large beam csv identity includes the modification time
def test_large_beam_csv_identity_includes_the_modification_time():
    # A file of 50 MiB or more is hashed at its first + last 4 MiB only, so its identity also holds the
    # modification time: a same-size edit in the middle (one adc value of a selected layer-1 bar) makes the
    # clean dataset stale although the edge hash is unchanged.  (Limits scaled down to 2 kB / 256 B.)
    with patch.object(io_utils, "FULL_HASH_LIMIT", 2048), patch.object(io_utils, "EDGE_HASH_BYTES", 256), \
            _tmpdir() as tmp:
        tmp = Path(tmp)
        beam = _run_beam(tmp, _synthetic_run(30), bar_major=False)
        ds = _dataset(beam)
        ctx = _ctx(tmp)
        _standard_calib(ctx)
        _run(ctx, [], [ds])
        record = read_json(cd.provenance_json(ctx, ds, beam))
        assert record["input"]["hash"] == "edges-4MiB" and record["input"]["mtime_ns"] == beam.stat().st_mtime_ns
        data = beam.read_bytes()
        lines = data.split(b"\n")
        selected = set(cd.clean_csv(ctx, ds, beam).read_bytes().split(b"\n")[1:])
        i = next(i for i in range(len(lines) // 2, len(lines)) if lines[i] in selected
                 and lines[i].split(b",")[6] == b"1" and len(lines[i].split(b",")[3]) == 3)
        fields = lines[i].split(b",")
        fields[3] = b"999"                                        # E_MIP 8.99: the event leaves the selection
        lines[i] = b",".join(fields)
        edited = b"\n".join(lines)
        assert len(edited) == len(data) and edited[:256] == data[:256] and edited[-256:] == data[-256:]
        st = beam.stat()
        beam.write_bytes(edited)
        os.utime(beam, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        assert io_utils.content_hash(beam)[0] == record["input"]["sha256"]        # the edges did not change
        ok, reason, _ = cd.clean_status(ctx, ds, beam)
        assert not ok and "beam csv differs" in reason
        _run(ctx, [], [ds])
        assert lines[i] not in set(cd.clean_csv(ctx, ds, beam).read_bytes().split(b"\n"))
        assert cd.current_clean_csv(ctx, ds, beam) is not None


# yml selection change makes the clean dataset stale
def test_yml_selection_change_makes_the_clean_dataset_stale():
    # A clean dataset made with the yml selection is stale once the yml defines another one
    # (threshold_mode, standard_by_particle, and for the muon standard layer1_energy_window_mev / mip_energy_mev).
    # Display settings do not matter, and a choice made by a flag stays accepted.  The consumer log names the
    # differing keys and the clean command that makes the dataset again (with the stored flags).
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        mu_csv = _run_beam(tmp, _synthetic_run(30))
        el_csv = _run_beam(tmp, _synthetic_run(30), folder="Electron_4GeV")
        mu, el = _dataset(mu_csv), _dataset(el_csv, key="electron_4GeV", particle="electron")
        ctx = _ctx(tmp)
        _standard_calib(ctx)
        _run(ctx, [], [mu, el])
        cases = (({"threshold_mode": "pedestalsigma"}, False, False,
                  'threshold_mode: made with "pedestalscale", configuration now "pedestalsigma"'),
                 ({"layer1_energy_window_mev": [4.0, 5.0]}, False, True, "layer1_energy_window_mev: made with [1.84"),
                 ({"mip_energy_mev": 1.0}, False, True, "mip_energy_mev: made with 4.83143, configuration now 1.0"),
                 ({"standard_by_particle": {"muon": "otherparticle"}}, False, True, "standard: made with"),
                 ({"standard_by_particle": {"electron": "muon"}}, True, False, "standard: made with"),
                 ({"mip_spectrum_bins": [0.0, 4.0, 40], "event_energy_bin_mev": 2.0}, True, True, ""),
                 ({"threshold_mode": "pedestalscale", "mip_energy_mev": 4.83143,
                   "layer1_energy_window_mev": [1.84, 7.83]}, True, True, ""))
        for section, mu_ok, el_ok, words in cases:
            ctx.cfg[cd.CONFIG_SECTION] = section
            for ds, want in ((mu, mu_ok), (el, el_ok)):
                ok, reason, _ = cd.clean_status(ctx, ds, _csv(ds))
                assert ok == want, (section, ds.key, reason)
                if not ok:
                    assert "clean_dataset settings of the configuration now give another selection" in reason
                    assert words in reason, (section, reason)
        ctx.cfg[cd.CONFIG_SECTION] = {"threshold_mode": "pedestalsigma"}
        cd._REPORTED.clear()
        path, out = _quiet(cd.current_clean_csv, ctx, mu, mu_csv)
        assert path is None and "threshold_mode" in out and "Make it again with: .\\runner.ps1 clean --muon 4\n" in out
        ctx.clean_input = True
        _refused(ctx.input_csv, mu, mu_csv, contains="Make it with: .\\runner.ps1 clean --muon 4")
        ctx.clean_input = False
        ctx.cfg[cd.CONFIG_SECTION] = {"mip_energy_mev": -1.0}     # an unusable section: stale with the reason
        assert "settings of the configuration cannot be used" in cd.clean_status(ctx, mu, mu_csv)[1]
        # flags: -pedestalsigma is kept whatever threshold_mode says; the window still counts
        ctx.cfg.pop(cd.CONFIG_SECTION)
        _run(ctx, ["-pedestalsigma"], [mu])
        assert read_json(cd.provenance_json(ctx, mu, mu_csv))["selection_flags"] == {"standard": None,
                                                                             "threshold_mode": "pedestalsigma"}
        for section in ({}, {"threshold_mode": "pedestalscale"}, {"threshold_mode": "pedestalsigma"}):
            ctx.cfg[cd.CONFIG_SECTION] = section
            assert cd.clean_status(ctx, mu, mu_csv)[0], section
        ctx.cfg[cd.CONFIG_SECTION] = {"layer1_energy_window_mev": [2.0, 7.0]}
        cd._REPORTED.clear()
        path, out = _quiet(cd.current_clean_csv, ctx, mu, mu_csv)
        assert path is None and "Make it again with: .\\runner.ps1 clean --muon 4 -pedestalsigma\n" in out
        # -muon-standard on an electron run: standard_by_particle does not matter, the muon window does
        ctx.cfg.pop(cd.CONFIG_SECTION)
        _run(ctx, ["-muon-standard"], [el])
        ctx.cfg[cd.CONFIG_SECTION] = {"standard_by_particle": {"electron": "otherparticle", "muon": "otherparticle"}}
        assert cd.clean_status(ctx, el, el_csv)[0]
        ctx.cfg[cd.CONFIG_SECTION] = {"mip_energy_mev": 5.0}
        assert "mip_energy_mev" in cd.clean_status(ctx, el, el_csv)[1]
        # a fresh run with the new yml makes it current again
        _run(ctx, ["-muon-standard"], [el])
        assert cd.clean_status(ctx, el, el_csv)[0]


# commit failures never mix generations
def test_commit_failures_never_mix_generations():
    # CLEAN2-01: the csv rename fails (file locked) -> the previous csv AND record stay byte-identical and
    # valid; the record write fails -> no record (not current), never a new csv with an old record.  No temp
    # file is left behind; the next run succeeds.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(30)))
        ctx = _ctx(tmp)
        std = _standard_calib(ctx)
        _run(ctx, [], [ds])
        csv, meta = cd.clean_csv(ctx, ds, _csv(ds)), cd.provenance_json(ctx, ds, _csv(ds))
        old_csv, old_meta = csv.read_bytes(), meta.read_bytes()
        _standard_calib(ctx, _calib_frame({(1, 3, 0): {"mpv": 300.0}, (1, 3, 1): {"mpv": 300.0}}))  # other rows
        original = os.replace

        # os.replace that fails on the clean csv (a locked file)
        def locked(src, dst):
            if Path(dst) == csv:
                raise PermissionError("simulated: clean csv opened in Excel")
            return original(src, dst)

        with patch.object(io_utils.os, "replace", side_effect=locked), io_utils.temp_registry():
            _refused(cd.run, ctx, _args([]), [ds], exc=PermissionError)
        assert csv.read_bytes() == old_csv and meta.read_bytes() == old_meta
        assert io_utils.product_set_valid(meta, members=[csv])[0]
        assert cd.current_clean_csv(ctx, ds, _csv(ds)) is None    # valid pair, but made with the old calibration
        leftovers = [p.name for p in csv.parent.iterdir() if ".tmp" in p.name or ".old-" in p.name]
        assert not leftovers, leftovers
        real_write_json = io_utils.write_json

        # write_json that fails on the record (disk full)
        def failing(obj, path):
            if Path(path) == meta:
                raise OSError("simulated: disk full while writing the record")
            return real_write_json(obj, path)

        with patch.object(io_utils, "write_json", side_effect=failing), io_utils.temp_registry():
            _refused(cd.run, ctx, _args([]), [ds], exc=OSError)
        assert not meta.exists() and csv.read_bytes() != old_csv   # new csv, but no record: never trusted
        assert cd.clean_status(ctx, ds, _csv(ds))[1].startswith("no metadata")
        assert not [p.name for p in csv.parent.iterdir() if ".tmp" in p.name or ".old-" in p.name]
        _run(ctx, [], [ds])
        assert cd.current_clean_csv(ctx, ds, _csv(ds)) == csv and std.exists()


# calibration keys are validated before the beam csv is read
def test_calibration_keys_are_validated_before_the_beam_csv_is_read():
    # CLEAN2-02 (shared key rule): duplicated, fractional or non-physical calibration keys are refused before
    # any beam csv is read, whatever the row order.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(6)))
        ctx = _ctx(tmp)
        good = _calib_frame()
        dup = pd.concat([good, good.iloc[[0]].assign(mpv=900.0)], ignore_index=True)
        frac = good.copy()
        frac["strip"] = frac["strip"].astype(float)
        frac.loc[0, "strip"] = 0.9
        nonphysical = pd.concat([good, pd.DataFrame([{**good.iloc[0].to_dict(), "strip": 8}])], ignore_index=True)
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
            for bad, words in ((dup, "share a"), (dup.iloc[::-1], "share a"), (frac, "non-integer"),
                               (nonphysical, "outside the detector")):
                _standard_calib(ctx, bad)
                _refused(cd.run, ctx, _args([]), [ds], exc=io_utils.ChannelKeyError, contains=words)
        assert not cd.clean_csv(ctx, ds, _csv(ds)).exists()
    assert "joe_mpv_file" not in vars(_args([])) and "preset" not in vars(_args([]))


# settings and flags are checked before any io
def test_settings_and_flags_are_checked_before_any_io():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(tmp / "runs" / "Muon_4GeV" / "absent.csv")
        bad = {"layer1_energy_window_mev": [7.83, 1.84], "mip_energy_mev": float("nan"),
               "threshold_mode": "pedestal", "standard_by_particle": {"muon": "mu"},
               "mip_spectrum_bins": [0, 8, 0], "diagnostics_min_bin_count": -1, "event_energy_bin_mev": 0}
        ctx = _ctx(tmp, section=bad)
        text = _refused(cd.run, ctx, _args([]), [ds], exc=io_utils.SettingsError)
        for key in bad:
            assert key in text, key
        assert sorted(p.name for p in tmp.iterdir()) == []                 # refused before any folder or file
        ctx = _ctx(tmp, section={"preset_by_particle": {"muon": "m1"}, "threshold_mode": "pedestalsigma"})
        settings, unknown = cd.load_settings(ctx)
        assert unknown == ["preset_by_particle"] and settings["threshold_mode"] == "pedestalsigma"
        jobs = cd.make_jobs(settings, _args([]), [ds, _dataset(tmp / "e" / "absent.csv", "electron_4GeV", "electron")])
        assert [(j.standard, j.threshold_mode) for j in jobs] == [("muon", "pedestalsigma"),
                                                                 ("otherparticle", "pedestalsigma")]
        jobs = cd.make_jobs(settings, _args(["-otherparticle", "-pedestalscale"]), [ds])
        assert (jobs[0].standard, jobs[0].threshold_column) == ("otherparticle", "pedestal_range_with_20percent")
        assert cd.rebuild_command(ctx, jobs[0]) == ".\\runner.ps1 clean --muon 4 -otherparticle -pedestalscale"
        assert jobs[0].flags == {"standard": "otherparticle", "threshold_mode": "pedestalscale"}
        test = _ctx(tmp, test=True, n_test_events=12)
        assert cd.rebuild_command(test, jobs[0], from_saved=True) == (
            ".\\runner.ps1 clean --muon 4 -otherparticle -pedestalscale -test -test-events 12 -from-saved")
        # no input gate: a run of a particle without its own standard is not refused, it gets the
        # general rule otherparticle, named in the standard source (log, provenance); a flag applies to any run
        kaon = cd.make_jobs(settings, _args([]), [_dataset(tmp / "k" / "absent.csv", "kaon_4GeV", "kaon")])[0]
        assert (kaon.standard, kaon.standard_source) == (
            "otherparticle", "general rule for kaon 4: clean_dataset.standard_by_particle has no kaon entry")
        electron = _dataset(tmp / "e" / "absent.csv", "electron_4GeV", "electron")
        assert cd.make_jobs(settings, _args(["-muon-standard"]), [electron])[0].standard == "muon"
        _refused(cd.load_settings, _ctx(tmp, section={"standard_by_particle": {"mu": "muon"}}),
                 exc=io_utils.SettingsError, contains="keyed by full particle names")
        _refused(cd.run, ctx, _args([]), [ds], contains="no csv file in")      # the run folder holds no csv
        _run_beam(tmp, _synthetic_run(6))
        _refused(cd.run, ctx, _args([]), [ds], contains="Missing input file")   # no calibration table yet
    for argv in (["-muon-standard", "-otherparticle"], ["-pedestalscale", "-pedestalsigma"]):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            _refused(_args, argv)                                # argparse: mutually exclusive


# unknown config keys are reported not fatal
def test_unknown_config_keys_are_reported_not_fatal():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(6)))
        ctx = _ctx(tmp, section={"fix_bugs": "all", "mip_energy_mev": 4.83143})
        _standard_calib(ctx)
        _, out = _run(ctx, [], [ds])
        assert "keys this stage does not use" in out and "fix_bugs" in out
        assert cd.clean_csv(ctx, ds, _csv(ds)).exists()


# failed run leaves no temp files and keeps the old products
def test_failed_run_leaves_no_temp_files_and_keeps_the_old_products():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(18)))
        ctx = _ctx(tmp, make_plots=True)
        _standard_calib(ctx)
        _run(ctx, [], [ds])
        src = _csv(ds)
        results = [cd.cutflow_csv(ctx, ds, src), cd.layers_csv(ctx, ds, src), cd.diagnostics_pdf(ctx, ds, src),
                   cd.results_json(ctx, ds, src)]
        before = {p: p.read_bytes() for p in results + [cd.clean_csv(ctx, ds, _csv(ds))]}
        with patch.object(cd, "draw_diagnostics", side_effect=RuntimeError("plot failed")), io_utils.temp_registry():
            _refused(cd.run, ctx, _args([]), [ds], exc=RuntimeError)
        for folder in (cd.clean_csv(ctx, ds, src).parent, results[0].parent, cd.cache_npz(ctx, ds, src).parent):
            names = [p.name for p in folder.iterdir()]
            assert not [n for n in names if ".tmp" in n or ".old-" in n], names
        # the clean dataset (same content) and the cache were committed again; the results set was not
        # committed, so its previous files and record are untouched and still valid
        for p, data in before.items():
            assert p.read_bytes() == data, p
        assert io_utils.product_set_valid(cd.results_json(ctx, ds, _csv(ds)))[0]
        assert io_utils.product_set_valid(cd.cache_json(ctx, ds, _csv(ds)))[0]
        assert cd.current_clean_csv(ctx, ds, _csv(ds)) is not None


# --------------------------------------------------------------------------
# results, -from-saved, plan
# --------------------------------------------------------------------------
# The per-layer csv and the pdf come from one summary; the numbers are exact.
def test_summary_values_and_one_computation():
    bars = _bars([(1, 1, 3, 1.0, 1.2), (1, 2, 3, 0.8, 1.0), (2, 1, 3, 1.1, 0.9), (2, 2, 4, 1.0, 1.0),
                  (3, 1, 3, 9.0, 9.0)])
    bars.in_clean = np.array([1, 1, 1, 1, 0], bool)
    s = cd.summarize(bars, "muon", K)
    t = s.layers.set_index("layer")
    assert t.loc[1, "n_clean_bars"] == 2 and t.loc[2, "n_events"] == 2 and t.loc[3, "n_clean_bars"] == 0
    np.testing.assert_allclose(t.loc[1, "energy_mev_end0_mean"], 1.05 * K)
    np.testing.assert_allclose(t.loc[1, "energy_mev_bar_mean"], (1.0 + 1.2 + 1.1 + 0.9) / 4 * K)
    assert np.isnan(t.loc[3, "energy_mev_end0_mean"])
    assert s.event_bars.tolist() == [2, 2] and s.event_layers.tolist() == [2, 2]
    np.testing.assert_allclose(s.event_energy, [(1.1 + 0.9) * K, 2.0 * K])
    assert sorted(s.multiplicity.tolist()) == [1] * 5
    assert s.bar_count[0, 3 + 2] == 2 and s.bar_count[1, 4 + 2] == 1          # 8-bar layers at positions 2..9
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(18)))
        ctx = _ctx(tmp, make_plots=True)
        _standard_calib(ctx)
        calls = []
        real = cd.summarize
        with patch.object(cd, "summarize", side_effect=lambda *a: calls.append(1) or real(*a)):
            _run(ctx, [], [ds])
        assert len(calls) == 1


# from saved redraws without reading the beam csv
def test_from_saved_redraws_without_reading_the_beam_csv():
    # -from-saved makes the result units from the kept-bar cache (the beam csv is only hashed for its identity);
    # with -no-plots the two tables without the pdf.  Refused with the reason and the rebuild command when the cache
    # has another selection, another beam csv or other calibration values, or is incomplete.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(24))
        ds = _dataset(src)
        ctx = _ctx(tmp, make_plots=True)
        calib_path = _standard_calib(ctx)
        _run(ctx, [], [ds])
        flow, layers = cd.cutflow_csv(ctx, ds, src).read_bytes(), cd.layers_csv(ctx, ds, src).read_bytes()
        cd.diagnostics_pdf(ctx, ds, src).unlink()
        clean_before = cd.clean_csv(ctx, ds, src).read_bytes()
        ctx.cfg[cd.CONFIG_SECTION] = {"mip_spectrum_bins": [0.0, 4.0, 40]}        # display settings may change
        no_read = (patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")),
                   patch.object(cd, "copy_selected_rows", side_effect=AssertionError("pass 2")))
        with no_read[0], no_read[1]:
            _, out = _run(ctx, ["-from-saved"], [ds])
        assert cd.diagnostics_pdf(ctx, ds, src).exists() and "the result units from beam_hit_bars.npz" in out
        assert cd.cutflow_csv(ctx, ds, src).read_bytes() == flow and cd.layers_csv(ctx, ds, src).read_bytes() == layers
        assert cd.clean_csv(ctx, ds, src).read_bytes() == clean_before
        record = read_json(cd.results_json(ctx, ds, src))
        assert record["drawn"].startswith("redrawn with -from-saved") and \
            record["diagnostics"]["mip_spectrum_bins"] == [0.0, 4.0, 40] and \
            record["diagnostics"]["mip_energy_mev"] == K
        assert sorted(record["results"]) == ["cutflow", "diagnostics", "layers"]
        # -no-plots: the two tables from the cache, no pdf; the record lists only them
        ctx.make_plots = False
        cd.cutflow_csv(ctx, ds, src).unlink()
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
            _run(ctx, ["-from-saved"], [ds])
        assert cd.cutflow_csv(ctx, ds, src).read_bytes() == flow
        assert read_json(cd.results_json(ctx, ds, src))["results"] == {"cutflow": "beam_clean_cutflow.csv",
                                                                     "layers": "beam_clean_layers.csv"}
        ctx.make_plots = True
        # another selection than the stored one
        text = _refused(cd.run, ctx, _args(["-from-saved", "-pedestalsigma"]), [ds], exc=io_utils.ProductSetError,
                        contains='threshold_mode: made with "pedestalscale", this call asks for "pedestalsigma"')
        assert "Rebuild it with: .\\runner.ps1 clean --muon 4 -pedestalsigma" in text and "standard:" not in text
        assert "or use the stored selection with: .\\runner.ps1 clean --muon 4 -from-saved" in text
        # stored with -pedestalsigma, asked without it: the message gives the call that uses the stored one
        _run(ctx, ["-pedestalsigma"], [ds])
        text = _refused(cd.run, ctx, _args(["-from-saved"]), [ds], exc=io_utils.ProductSetError,
                        contains="this call asks for")
        assert "or use the stored selection with: .\\runner.ps1 clean --muon 4 -pedestalsigma -from-saved" in text
        _, out = _run(ctx, ["-pedestalsigma", "-from-saved"], [ds])
        assert "the result units from" in out
        _run(ctx, [], [ds])
        # another beam csv than the one the cache was made from; the same content again is accepted
        _write_beam(src, _synthetic_run(25))
        _refused(cd.run, ctx, _args(["-from-saved"]), [ds], exc=io_utils.ProductSetError,
                 contains="is not current: the beam csv differs")
        _write_beam(src, _synthetic_run(24))
        _run(ctx, ["-from-saved"], [ds])
        # the calibration values changed since the selection run
        _standard_calib(ctx, _calib_frame({(2, 3, 0): {"pedestal": 101.0}}))
        _refused(cd.run, ctx, _args(["-from-saved"]), [ds], exc=io_utils.ProductSetError,
                 contains="is not current: calibration_parameters.csv now holds other")
        calib_path.unlink()
        _refused(cd.run, ctx, _args(["-from-saved"]), [ds], exc=io_utils.ProductSetError, contains="is missing")
        # an incomplete cache
        _standard_calib(ctx)
        cd.cache_npz(ctx, ds, src).unlink()
        text = _refused(cd.run, ctx, _args(["-from-saved"]), [ds], exc=io_utils.ProductSetError,
                        contains="clean kept-bar cache of Muon_4GeV/beam.csv")
        assert "Rebuild it with: .\\runner.ps1 clean --muon 4" in text


# only units make each output on its own
def test_only_units_make_each_output_on_its_own():
    # -only: each unit alone (clean csv, cut flow, layers, pdf), groups; the results json lists only what the call
    # wrote; every selection run writes the kept-bar cache; -no-plots drops the pdf and leaves an earlier one alone; a
    # call that makes nothing is refused before any file is read.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(18))
        ds = _dataset(src)
        ctx = _ctx(tmp, make_plots=True)
        _standard_calib(ctx)
        paths = {"clean": cd.clean_csv(ctx, ds, src), "cutflow": cd.cutflow_csv(ctx, ds, src),
                 "layers": cd.layers_csv(ctx, ds, src), "diagnostics": cd.diagnostics_pdf(ctx, ds, src)}
        record = cd.results_json(ctx, ds, src)
        for unit in cd.UNITS:
            _, out = _run(ctx, ["-only", unit], [ds])
            assert {u for u, p in paths.items() if p.exists()} == {unit}, unit
            assert f"output units: {unit}" in out and cd.cache_npz(ctx, ds, src).exists()
            if unit == "clean":
                assert not record.exists() and cd.provenance_json(ctx, ds, src).exists()
            else:
                rec = read_json(record)
                assert list(rec["results"]) == [unit] and list(rec["members"]) == [paths[unit].name]
                assert rec["clean_csv"]["sha256"] is None                # no clean csv in this call
                record.unlink()
            for p in (*paths.values(), cd.provenance_json(ctx, ds, src), cd.cache_npz(ctx, ds, src)):
                p.unlink(missing_ok=True)
        _run(ctx, ["-only", "tables"], [ds])
        assert {u for u, p in paths.items() if p.exists()} == {"cutflow", "layers"}
        _run(ctx, ["-only", "clean,diagnostics", "-only", "layers"], [ds])
        assert all(p.exists() for p in paths.values())
        assert read_json(record)["clean_csv"]["sha256"] == \
            read_json(cd.provenance_json(ctx, ds, src))["members"]["beam_clean.csv"]["sha256"]
        # -no-plots: no pdf; the earlier pdf stays, the record lists the two tables only
        pdf = paths["diagnostics"].read_bytes()
        ctx.make_plots = False
        _, out = _run(ctx, [], [ds])
        assert paths["diagnostics"].read_bytes() == pdf and "removed" not in out
        assert sorted(read_json(record)["members"]) == ["beam_clean_cutflow.csv", "beam_clean_layers.csv"]
        # nothing to make: refused before any file is read
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
            _refused(_run, ctx, ["-only", "diagnostics"], [ds], exc=io_utils.SettingsError, contains="needs plots")
            ctx.make_plots = True
            _refused(_run, ctx, ["-from-saved", "-only", "clean"], [ds], exc=io_utils.SettingsError,
                     contains="-from-saved cannot make the clean csv")
            # -from-saved -only layers: the layers table alone, from the cache
            paths["layers"].unlink()
            _run(ctx, ["-from-saved", "-only", "layers"], [ds])
        assert paths["layers"].exists() and list(read_json(record)["results"]) == ["layers"]
    with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
        _refused(_args, ["-only", "bogus"])                      # argparse: unknown unit
    assert _args(["-only", "results"]).only == ("cutflow", "layers", "diagnostics") and _args([]).only == cd.UNITS


# no json writes no record next to the products
def test_no_json_writes_no_record_next_to_the_products():
    # -no-json: no provenance json next to the clean csv and no results json in results/; the cache records stay
    # (-from-saved reads them); a clean csv without its record is used, not verified; the plan lists no such json.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        try:
            src = _run_beam(tmp, _synthetic_run(18))
            ds = _dataset(src)
            ctx = _ctx(tmp, write_json=False)
            _standard_calib(ctx)
            _run(ctx, [], [ds])
            assert cd.clean_csv(ctx, ds, src).exists() and not cd.provenance_json(ctx, ds, src).exists()
            assert cd.cutflow_csv(ctx, ds, src).exists() and not cd.results_json(ctx, ds, src).exists()
            assert io_utils.product_set_valid(cd.cache_json(ctx, ds, src), members=[cd.cache_npz(ctx, ds, src)])[0]
            ok, reason, record = cd.clean_status(ctx, ds, src)
            assert ok and record is None and reason.startswith("not verified (-no-json"), reason
            cd._REPORTED.clear()
            path, out = _quiet(cd.current_clean_csv, ctx, ds, src)
            assert path == cd.clean_csv(ctx, ds, src) and "not verified" in out
            with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
                _run(ctx, ["-from-saved"], [ds])
            ctx.dry_run = True
            names = {Path(i["path"]).name for i in cd.io_plan(ctx, _args([]), [ds]) if i["role"] == "out"}
            assert names == {"beam_clean.csv", "beam_hit_bars.npz", "beam_hit_bars.json", "beam_clean_cutflow.csv",
                             "beam_clean_layers.csv"}
        finally:
            io_utils.set_json_records(True)


# otherparticle run and empty selection
def test_otherparticle_run_and_empty_selection():
    # An electron dataset gets the otherparticle standard; a selection without any clean event still writes
    # a valid (header-only) clean csv and the results.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(12), folder="Electron_4GeV")
        ds = _dataset(src, key="electron_4GeV", particle="electron")
        ctx = _ctx(tmp, make_plots=True)
        frame = _calib_frame()
        _standard_calib(ctx, frame)
        _run(ctx, [], [ds])
        record = read_json(cd.provenance_json(ctx, ds, src))
        assert record["selection"]["standard"] == "otherparticle"
        assert "layer1_energy_window_mev" not in record["selection"]
        assert _lines(cd.clean_csv(ctx, ds, src)) == _expected_lines(src, _reference_rows(
            src, frame, "otherparticle", "pedestal_range_with_20percent"))
        noise_csv = _run_beam(tmp, _noise(1) + _noise(2), name="noise.csv")
        noise = _dataset(noise_csv, key="muon_4GeV")
        _run(ctx, [], [noise])
        assert _lines(cd.clean_csv(ctx, noise, noise_csv)) == _lines(noise_csv)[:1]
        assert read_json(cd.provenance_json(ctx, noise, noise_csv))["counts"]["output_events"] == 0
        assert cd.diagnostics_pdf(ctx, noise, noise_csv).exists()
        assert cd.current_clean_csv(ctx, noise, noise_csv) is not None


# dry run plan writes nothing
def test_dry_run_plan_writes_nothing():
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(6))
        ds = _dataset(src)
        ctx = _ctx(tmp, test=True, make_plots=True)
        ctx.dry_run = True
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            plan = cd.io_plan(ctx, _args([]), [ds])
            (n_ext, n_pend), out = _quiet(runner.print_plan, "clean", plan, ctx)
        assert plan[0]["kind"] == "auto" and plan[0]["path"] == ctx.test_subset_csv(ds, src)
        assert plan[1]["kind"] == "product" and plan[1]["produced_by"] == ".\\runner.ps1 calib -test"
        outs = {Path(i["path"]).name for i in plan if i["role"] == "out"}
        assert outs == {"beam_clean.csv", "beam_clean.provenance.json", "beam_hit_bars.npz", "beam_hit_bars.json",
                        "beam_clean_cutflow.csv", "beam_clean_layers.csv", "beam_clean_diagnostics.pdf",
                        "beam_clean_results.json"}
        assert (n_ext, n_pend) == (0, 1) and "[in ] AUTO" in out
        ctx.make_plots = False
        assert "beam_clean_diagnostics.pdf" not in {Path(i["path"]).name for i in cd.io_plan(ctx, _args([]), [ds])}
        ctx.make_plots = True
        plan = cd.io_plan(ctx, _args(["-from-saved"]), [ds])
        assert [i["role"] for i in plan][:3] == ["in", "cache", "in"] and plan[0]["path"] == src
        out = _quiet(runner.print_plan, "clean", plan, ctx)[1]
        assert "[cache] rebuild" in out
        assert sorted(p.name for p in tmp.iterdir()) == ["runs"]                  # nothing created
        # after a run: the from-saved plan shows the cache as reusable
        ctx.dry_run = False
        _standard_calib(ctx)
        _run(ctx, [], [ds])
        ctx.dry_run = True
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            out = _quiet(runner.print_plan, "clean", cd.io_plan(ctx, _args(["-from-saved"]), [ds]), ctx)[1]
        assert "[cache] reuse" in out and "[in ] ok " in out


# --------------------------------------------------------------------------
# --output: which final results are saved, and where (common/outputs.py)
# --------------------------------------------------------------------------
_PARTS = ("cutflow", "layers", "diagnostics", "results")
_DEFAULT_NAMES = ("clean_cutflow.csv", "clean_layers.csv", "clean_diagnostics.pdf", "clean_results.json")


# no output full set saves no final result but the chain runs
def test_no_output_full_set_saves_no_final_result_but_the_chain_runs():
    # Full set without --output: no final result in results/ (the code still makes them, into the scratch folder
    # cache/clean_dataset/unsaved/, emptied at the start of each run); the intermediates are written as always, so the
    # clean dataset is current for a later stage (-clean-input) and -from-saved works.  One 'not saved' line per run.
    # Results saved earlier with --output are left alone by later runs without it (also with -no-plots).
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        mu_csv = _run_beam(tmp, _synthetic_run(24), name="mu.csv")
        el_csv = _run_beam(tmp, _synthetic_run(24), folder="Electron_4GeV", name="el.csv")
        mu, el = _dataset(mu_csv), _dataset(el_csv, key="electron_4GeV", particle="electron")
        ctx = _ctx(tmp, make_plots=True, outputs=None)
        _standard_calib(ctx)
        scratch = ctx.cache_dir(cd.STAGE_DIR, outputs_mod.UNSAVED_DIR)
        scratch.mkdir(parents=True)
        (scratch / "stale.csv").write_text("from an earlier run")
        _, out = _run(ctx, [], [mu, el])
        assert not (tmp / "results").exists()
        assert sorted(p.name for p in scratch.iterdir()) == sorted(f"{s}_{n}" for s in ("mu", "el")
                                                                   for n in _DEFAULT_NAMES)
        record = read_json(scratch / "mu_clean_results.json")
        assert record["saved"] is False and io_utils.product_set_valid(scratch / "mu_clean_results.json")[0]
        assert sorted(record["members"]) == ["mu_clean_cutflow.csv", "mu_clean_diagnostics.pdf",
                                             "mu_clean_layers.csv"]
        assert out.count("final results not saved (no --output)") == 1
        assert (f"final results not saved (no --output): scratch copies in {scratch}; use --output [names] to save "
                f"them") in out
        assert "final results:" not in out
        for ds, src in ((mu, mu_csv), (el, el_csv)):           # intermediates: written as always
            assert io_utils.product_set_valid(cd.provenance_json(ctx, ds, src), members=[cd.clean_csv(ctx, ds, src)])[0]
            assert io_utils.product_set_valid(cd.cache_json(ctx, ds, src), members=[cd.cache_npz(ctx, ds, src)])[0]
            cd._REPORTED.clear()
            assert _quiet(cd.current_clean_csv, ctx, ds, src)[0] == cd.clean_csv(ctx, ds, src)
        ctx.clean_input = True                                # a later stage reads the clean dataset
        assert _quiet(ctx.input_csv, mu, mu_csv)[0] == cd.clean_csv(ctx, mu, mu_csv)
        ctx.clean_input = False
        # -from-saved needs no saved final result; its results go to the (emptied) scratch folder as well
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
            _, out = _run(ctx, ["-from-saved"], [mu])
        assert "the result units from" in out and "final results not saved (no --output)" in out
        assert not (tmp / "results").exists()
        assert sorted(p.name for p in scratch.iterdir()) == sorted(f"mu_{n}" for n in _DEFAULT_NAMES)
        assert read_json(scratch / "mu_clean_results.json")["drawn"].startswith("redrawn with -from-saved")
        # saved with --output alone; later runs without --output never touch them
        ctx.outputs = []
        _run(ctx, [], [mu])
        saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (
            cd.cutflow_csv(ctx, mu, mu_csv), cd.layers_csv(ctx, mu, mu_csv), cd.diagnostics_pdf(ctx, mu, mu_csv),
            cd.results_json(ctx, mu, mu_csv))}
        ctx.outputs = None
        _run(ctx, [], [mu])
        ctx.make_plots = False
        _run(ctx, [], [mu])
        for p, (data, stamp) in saved.items():
            assert p.read_bytes() == data and p.stat().st_mtime_ns == stamp, p
        assert io_utils.product_set_valid(cd.results_json(ctx, mu, mu_csv))[0]
        assert sorted(p.name for p in scratch.iterdir()) == ["mu_clean_cutflow.csv", "mu_clean_layers.csv",
                                                             "mu_clean_results.json"]


# output names are matched by result name
def test_output_names_are_matched_by_result_name():
    # --output names are matched by result name: the default file name, the result key + extension, or KEY=FILE
    # (another file name).  Naming one file saves only that file; the others go to the scratch copy.  A name of no
    # result is ignored with a WARNING; --output alone saves every result under its default name in the run folder.
    # The record json lists the files under their names.  Two runs with csv files of the same name are refused when
    # their results would meet in one file.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(18))
        ds = _dataset(src)
        ctx = _ctx(tmp, make_plots=True, outputs=["beam_clean_cutflow.csv"])
        _standard_calib(ctx)
        scratch = ctx.cache_dir(cd.STAGE_DIR, outputs_mod.UNSAVED_DIR)
        _, out = _run(ctx, [], [ds])
        plan = outputs_mod.plans()[-1]
        key = "Muon_4GeV.beam.cutflow"
        assert [s.key for s in plan.slots if plan.saved(s.key)] == [key] and plan.path(key).exists()
        assert plan.path(key).name == "beam_clean_cutflow.csv" and f"csv {key}: {plan.path(key)} [--output name]" in out
        assert sorted(p.name for p in scratch.iterdir()) == ["beam_clean_diagnostics.pdf", "beam_clean_layers.csv",
                                                             "beam_clean_results.json"]
        # KEY=FILE and key + extension
        ctx.outputs = ["Muon_4GeV.beam.layers=layers.csv", "Muon_4GeV.beam.diagnostics.pdf",
                       "Muon_4GeV.beam.results=record.json"]
        _, out = _run(ctx, [], [ds])
        plan = outputs_mod.plans()[-1]
        assert plan.path("Muon_4GeV.beam.layers").name == "layers.csv"
        assert plan.path("Muon_4GeV.beam.diagnostics").name == "Muon_4GeV.beam.diagnostics.pdf"
        assert not plan.saved("Muon_4GeV.beam.cutflow") and "WARNING" not in out
        record = read_json(plan.path("Muon_4GeV.beam.results"))
        assert record["results"] == {"cutflow": "beam_clean_cutflow.csv", "layers": "layers.csv",
                                     "diagnostics": "Muon_4GeV.beam.diagnostics.pdf"} and record["saved"] is True
        assert io_utils.product_set_valid(plan.path("Muon_4GeV.beam.results"))[0]
        assert pd.read_csv(plan.path("Muon_4GeV.beam.layers"))["layer"].tolist() == list(range(1, 20))
        # a name of no result: ignored with a WARNING
        ctx.outputs = ["nothing.csv"]
        _, out = _run(ctx, [], [ds])
        assert "--output names that name no result here, ignored: nothing.csv" in out
        # --output alone: the default names in the run sub-folder, no warning
        ctx.outputs = []
        _, out = _run(ctx, [], [ds])
        assert "WARNING" not in out
        for part, name in zip(_PARTS, _DEFAULT_NAMES):
            assert f"Muon_4GeV.beam.{part}: {cd.results_dir(ctx) / 'Muon_4GeV' / ('beam_' + name)} [default name" in out
        # two runs whose csv files share a name: fine under the default names, refused for the scratch copy
        el_csv = _run_beam(tmp, _synthetic_run(6), folder="Electron_4GeV")
        el = _dataset(el_csv, key="electron_4GeV", particle="electron")
        _run(ctx, [], [ds, el])
        assert cd.cutflow_csv(ctx, el, el_csv).exists()
        assert cd.cutflow_csv(ctx, el, el_csv) != cd.cutflow_csv(ctx, ds, src)
        ctx.outputs = None
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")):
            _refused(_run, ctx, [], [ds, el], exc=io_utils.SettingsError, contains="would be written to one file")


# test mode without output writes numbered test names
def test_test_mode_without_output_writes_numbered_test_names():
    # -test without --output: results/test/test.csv, test1.csv, ..., test.pdf, test.json, numbered per type in the
    # order of the run (one run never overwrites its own results); the record test.json lists test.csv, test1.csv and
    # test.pdf, test1.json the files of the 2nd run; the next test run overwrites the same names.  Nothing goes to
    # results/test/clean_dataset/; the intermediates are written.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        mu_csv = _run_beam(tmp, _synthetic_run(18), name="mu.csv")
        el_csv = _run_beam(tmp, _synthetic_run(18), folder="Electron_4GeV", name="el.csv")
        mu, el = _dataset(mu_csv), _dataset(el_csv, key="electron_4GeV", particle="electron")
        ctx = _ctx(tmp, test=True, make_plots=True, n_test_events=12, outputs=None)
        _standard_calib(ctx)
        _, out = _run(ctx, [], [mu, el])
        folder = tmp / "results" / "test"
        assert sorted(p.name for p in folder.iterdir()) == ["test.csv", "test.json", "test.pdf", "test1.csv",
                                                            "test1.json", "test1.pdf", "test2.csv", "test3.csv"]
        first, second = read_json(folder / "test.json"), read_json(folder / "test1.json")
        assert first["dataset"] == "muon_4GeV" and first["csv"] == "Muon_4GeV/mu.csv"
        assert sorted(first["members"]) == ["test.csv", "test.pdf", "test1.csv"]
        assert first["results"] == {"cutflow": "test.csv", "layers": "test1.csv", "diagnostics": "test.pdf"}
        assert second["dataset"] == "electron_4GeV"
        assert sorted(second["members"]) == ["test1.pdf", "test2.csv", "test3.csv"]
        assert second["results"] == {"cutflow": "test2.csv", "layers": "test3.csv", "diagnostics": "test1.pdf"}
        assert io_utils.product_set_valid(folder / "test.json")[0]
        assert io_utils.product_set_valid(folder / "test1.json")[0]
        flow = read_json(cd.cache_json(ctx, mu, mu_csv))["cutflow"]
        assert pd.read_csv(folder / "test.csv")["rows"].tolist() == [c["rows"] for c in flow]
        assert "not saved" not in out and f"csv Muon_4GeV.mu.cutflow: {folder / 'test.csv'} [test name]" in out
        assert f"json Electron_4GeV.el.results: {folder / 'test1.json'} [test name]" in out
        for ds, src, run in ((mu, mu_csv, "Muon_4GeV"), (el, el_csv, "Electron_4GeV")):
            assert cd.clean_csv(ctx, ds, src).exists() and cd.cache_json(ctx, ds, src).exists()
            assert cd.clean_csv(ctx, ds, src).parent == tmp / "data" / "test" / "clean_dataset" / run
        # the next test run (electron 4 only, -no-plots) overwrites test.csv, test1.csv and test.json
        ctx.make_plots = False
        _run(ctx, [], [el])
        record = read_json(folder / "test.json")
        assert record["dataset"] == "electron_4GeV" and sorted(record["members"]) == ["test.csv", "test1.csv"]
        assert io_utils.product_set_valid(folder / "test.json")[0]
        # a name in -test saves only that result; the others go to the scratch copy
        ctx.outputs = ["Muon_4GeV.mu.cutflow=flow.csv"]
        _run(ctx, [], [mu])
        plan = outputs_mod.plans()[-1]
        assert plan.path("Muon_4GeV.mu.cutflow").name == "flow.csv" and plan.path("Muon_4GeV.mu.cutflow").exists()
        assert not plan.saved("Muon_4GeV.mu.layers") and not plan.saved("Muon_4GeV.mu.results")


# plan shows where the final results go and deletes nothing
def test_plan_shows_where_the_final_results_go_and_deletes_nothing():
    # io_plan (-dry-run, check, and the run log during real runs) lists the intermediates and the final results at
    # the destinations of the run: '(not saved: no --output)' in the scratch folder, --output names (with the
    # warning of the run), test names, only the -only units.  It never empties the scratch folder and registers no
    # output plan.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        src = _run_beam(tmp, _synthetic_run(6))
        ds = _dataset(src)
        ctx = _ctx(tmp, make_plots=True, outputs=None)
        ctx.dry_run = True
        scratch = ctx.cache_dir(cd.STAGE_DIR, outputs_mod.UNSAVED_DIR)
        scratch.mkdir(parents=True)
        (scratch / "beam_clean_cutflow.csv").write_text("scratch copy of the last run")
        n_plans = len(outputs_mod.plans())
        plan = cd.io_plan(ctx, _args([]), [ds])
        finals = [i for i in plan if i.get("result")]
        assert [i["result"] for i in finals] == [f"Muon_4GeV.beam.{p}" for p in _PARTS]
        assert [Path(i["path"]) for i in finals] == [scratch / f"beam_{n}" for n in _DEFAULT_NAMES]
        assert all(i["note"] == "(not saved: no --output)" and i["saved"] is False for i in finals)
        assert {Path(i["path"]).name for i in plan if i["role"] == "out" and not i.get("result")} == {
            "beam_clean.csv", "beam_clean.provenance.json", "beam_hit_bars.npz", "beam_hit_bars.json"}
        out = _quiet(runner.print_plan, "clean", plan, ctx)[1]
        assert str(scratch / "beam_clean_diagnostics.pdf") in out
        assert (scratch / "beam_clean_cutflow.csv").read_text() == "scratch copy of the last run"
        assert len(outputs_mod.plans()) == n_plans
        # names: their destinations, and the warning the run would give
        ctx.outputs = ["beam_clean_cutflow.csv", "y.pdf"]
        plan, out = _quiet(cd.io_plan, ctx, _args([]), [ds])
        finals = {i["result"]: i for i in plan if i.get("result")}
        assert finals["Muon_4GeV.beam.cutflow"]["saved"] is True
        assert Path(finals["Muon_4GeV.beam.cutflow"]["path"]).name == "beam_clean_cutflow.csv"
        assert not any(finals[f"Muon_4GeV.beam.{p}"]["saved"] for p in ("layers", "diagnostics", "results"))
        assert "ignored: y.pdf" in out
        # -only layers: the layers table, its record and the cache
        args = _args(["-only", "layers"])
        ctx.units = outputs_mod.selected_units(args)
        plan = _quiet(cd.io_plan, ctx, args, [ds])[0]
        assert [i["result"] for i in plan if i.get("result")] == ["Muon_4GeV.beam.layers", "Muon_4GeV.beam.results"]
        assert {Path(i["path"]).name for i in plan if i["role"] == "out" and not i.get("result")} == {
            "beam_hit_bars.npz", "beam_hit_bars.json"}
        ctx.units = None
        # -from-saved: no intermediate, the result units; with -no-plots the two tables
        plan = _quiet(cd.io_plan, ctx, _args(["-from-saved"]), [ds])[0]
        assert [i["result"] for i in plan if i.get("result")] == [f"Muon_4GeV.beam.{p}" for p in _PARTS]
        assert not [i for i in plan if i["role"] == "out" and not i.get("result")]
        ctx.make_plots = False
        plan = _quiet(cd.io_plan, ctx, _args(["-from-saved"]), [ds])[0]
        assert [i["result"] for i in plan if i.get("result")] == ["Muon_4GeV.beam.cutflow", "Muon_4GeV.beam.layers",
                                                                  "Muon_4GeV.beam.results"]
        # -test without --output: the test names
        test = _ctx(tmp, test=True, make_plots=True, outputs=None)
        test.dry_run = True
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            plan = cd.io_plan(test, _args([]), [ds])
        assert [Path(i["path"]).name for i in plan if i.get("result")] == ["test.csv", "test1.csv", "test.pdf",
                                                                           "test.json"]
        assert not (tmp / "results").exists() and len(outputs_mod.plans()) == n_plans
        assert sorted(p.name for p in scratch.iterdir()) == ["beam_clean_cutflow.csv"]


# runner flags
def test_runner_flags():
    parser = runner.build_parser()
    args = parser.parse_args(["clean", "--muon", "4", "-muon-standard", "-pedestalsigma", "-test", "-from-saved"])
    assert (args.standard, args.threshold_mode, args.from_saved) == ("muon", "pedestalsigma", True)
    assert args.outputs is None
    args = parser.parse_args(["clean", "--muon", "4", "-test", "-output", "flow.csv", "diag.pdf"])
    assert args.outputs == ["flow.csv", "diag.pdf"] and runner.make_context(args).outputs == ["flow.csv", "diag.pdf"]
    assert parser.parse_args(["clean", "-output"]).outputs == []
    args = parser.parse_args(["clean"])
    assert args.standard is None and args.threshold_mode is None and args.only == cd.UNITS
    assert parser.parse_args(["clean", "--muon", "4", "-only", "cutflow,layers"]).only == ("cutflow", "layers")
    for window, text in (([1.84, 7.83], "[1.84, 7.83] MeV"), ([2.0, 7.0], "[2, 7] MeV")):   # help from DEFAULTS
        with patch.dict(cd.DEFAULTS, {"layer1_energy_window_mev": window}):
            p = argparse.ArgumentParser()
            cd.add_arguments(p)
            assert text in " ".join(p.format_help().split()) and "4.83143 MeV per MIP" in p.format_help()
    assert "clean" not in runner.BROKEN_STAGES
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.yml"
        cfg.write_text((Path(runner.__file__).resolve().parents[1] / "configure" / "config.yml")
                       .read_text(encoding="utf-8"), encoding="utf-8")
        _refused(runner.main, ["clean", "-clean-input", "--config", str(cfg)],
                 contains="'clean' does not accept --clean-input")


# --------------------------------------------------------------------------
# --fit / --bin: the diagnostic per-end E_MIP spectrum fits (Lei 2026-10-02)
# --------------------------------------------------------------------------
# E_MIP of hit ends: a Landau-like MIP peak at about 1.0 (30000) and a second population at 2.0 (3000).
def _mip_values(seed: int = 4) -> np.ndarray:
    from scipy.stats import moyal

    rng = np.random.default_rng(seed)
    return np.r_[moyal.rvs(loc=1.0, scale=0.08, size=30000, random_state=rng) + rng.normal(0, 0.06, 30000),
                 rng.normal(2.0, 0.15, 3000)]


# Lines with data that _draw_end_fits puts on an axis (legend-only entries have none).
def _drawn_curves(f: dict) -> list:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        cd._draw_end_fits(ax, f)
        return [ln for ln in ax.get_lines() if len(ln.get_xdata())]
    finally:
        plt.close(fig)


# fit options parse and coexist with the selection and particle flags
def test_fit_options_parse_and_coexist_with_the_selection_and_particle_flags():
    # --fit / --bin next to -muon-standard / -otherparticle, -pedestalscale / -pedestalsigma and the common particle
    # flags --muon / --electron (nargs *): no clash, also in the PowerShell spelling joined by runner.main; a --bin
    # that does not match --fit is refused before any file or folder is made (run and plan).
    from ldmx_hcal.common import fitting

    parser = runner.build_parser()
    ns = parser.parse_args(fitting.normalize_range_args(["clean", "--muon", "-muon-standard", "-fit[0.6", "1.6]",
                                                         "-bin", "15", "--electron", "4", "-pedestalsigma", "-test"]))
    assert (ns.standard, ns.threshold_mode, ns.fit_ranges, ns.fit_bins) == ("muon", "pedestalsigma", [(0.6, 1.6)], [15])
    assert ns.select_muon == [[]] and ns.select_electron == [["4"]] and ns.run_names == [] and ns.test
    ns = parser.parse_args(fitting.normalize_range_args(["clean", "--electron", "-otherparticle", "-fit[0.5", "1.1]",
                                                         "[1.1", "2.5]", "-bin", "15", "20", "-from-saved"]))
    assert ns.standard == "otherparticle" and ns.fit_ranges == [(0.5, 1.1), (1.1, 2.5)] and ns.fit_bins == [15, 20]
    assert ns.select_electron == [[]] and ns.from_saved
    args = _args(["-fit", "[0.6,1.6]"])
    assert args.fit_ranges == [(0.6, 1.6)] and args.fit_bins is None and _args([]).fit_ranges is None
    assert cd.diagnostic_fit_settings(args) == {"fit_ranges": [[0.6, 1.6]], "fit_bins": None}
    assert cd.diagnostic_fit_settings(_args([])) == {"fit_ranges": None, "fit_bins": None}
    with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
        _refused(_args, ["--fit", "[0.1,0.2],[0.3,0.4],[0.5,0.6]"])        # argparse: at most two ranges
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(tmp / "absent.csv")
        ctx = _ctx(tmp)
        for argv, words in ((["--bin", "15"], "clean: --bin needs --fit"),
                            (["--fit", "[0.5,1.1],[1.1,2.5]", "--bin", "15"], "one value per fit range")):
            for fn in (cd.run, cd.io_plan):
                _refused(fn, ctx, _args(argv), [ds], exc=io_utils.SettingsError, contains=words)
        assert sorted(p.name for p in tmp.iterdir()) == []                      # nothing made


# spectrum fits with ranges bins and failures
def test_spectrum_fits_with_ranges_bins_and_failures():
    # end_spectrum_fits on a synthetic E_MIP spectrum.  Default: the automatic fits.  One range: the same Gaussian
    # and Landau fits restricted to it (on the page bins it equals the default with that mip_fit_range); --bin N: N
    # equal bins of the range, drawn as points scaled to the page bins.  Two ranges: double Gaussian (range_gauss_fit,
    # one component per range), Landau on the first range only.  Failed fits: no value, no curve, points only.
    mip = _mip_values()
    s = dict(cd.DEFAULTS)
    lo, hi, n = s["mip_spectrum_bins"]
    edges = np.linspace(lo, hi, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.histogram(mip, bins=edges)[0]
    f = cd.end_spectrum_fits(mip, s)
    x, y, g, lan = cd._spectrum_fits(centers, counts, s["mip_fit_range"], s["diagnostics_min_bin_count"])
    assert np.array_equal(f["x"], x) and np.array_equal(f["y"], y) and f["scale"] == 1.0 and f["double"] is None
    assert np.array_equal(f["gauss"].params, g.params) and np.array_equal(f["landau"].params, lan.params)
    assert f["gauss_range"] == (0.7, 1.3) and len(_drawn_curves(f)) == 2
    # one range on the page bins == the default fit with that mip_fit_range (edges between bin centres)
    one = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.68, 1.32]]})
    ref = cd.end_spectrum_fits(mip, {**s, "mip_fit_range": [0.68, 1.32]})
    assert np.array_equal(one["gauss"].params, ref["gauss"].params)
    assert np.array_equal(one["landau"].params, ref["landau"].params) and np.array_equal(one["y"], ref["y"])
    assert one["gauss_range"] == (0.68, 1.32) and one["fit_points"] == [] and one["scale"] == 1.0
    # --bin 12 on [0.8, 1.4]: 0.05-MIP bins, points on the page scale (x 0.04 / 0.05)
    binned = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.8, 1.4]], "fit_bins": [12]})
    own = np.histogram(mip, bins=np.linspace(0.8, 1.4, 13))[0]
    px, py = binned["fit_points"][0]
    assert np.allclose(px, np.linspace(0.825, 1.375, 12)) and np.allclose(py, own * 0.04 / 0.05)
    assert np.isclose(binned["scale"], 0.8) and 0.95 < binned["gauss"].params[1] < 1.15
    assert [ln.get_label().split(" ")[0] for ln in _drawn_curves(binned)] == ["fit", "Gauss", "Landau"]
    # two ranges: double Gaussian, Landau on the first range only
    two = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.7, 1.5], [1.6, 2.5]]})
    dbl = two["double"]
    assert dbl["ok"] and dbl["kind"] == "range_double_gauss" and two["gauss"] is None
    assert 1.0 < dbl["mu_low"] < 1.15 and 1.9 < dbl["mu_high"] < 2.05 and two["landau_range"] == (0.7, 1.5)
    labels = [ln.get_label() for ln in _drawn_curves(two)]
    assert any("first range only" in t for t in labels) and any(t.startswith("double Gauss (sum)") for t in labels)
    two_binned = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.7, 1.5], [1.6, 2.5]], "fit_bins": [15, 20]})
    assert [len(p[0]) for p in two_binned["fit_points"]] == [15, 20] and two_binned["double"]["ok"]
    assert [len(seg["counts"]) for seg in two_binned["double"]["segments"]] == [15, 20]
    # failures: no MIP peak in the range -> no fit; a range without data -> the double Gaussian failed (R3)
    empty = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[5.0, 6.0]]})
    assert empty["gauss"] is None and empty["landau"] is None and _drawn_curves(empty) == []
    bad = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.7, 1.5], [5.0, 6.0]], "fit_bins": [10, 10]})
    assert not bad["double"]["ok"] and bad["double"]["status"].startswith("fit_failed: [5, 6] has 0 bin(s)")
    assert np.isnan(bad["double"]["mu_high"]) and np.isnan(bad["double"]["mu_low"])
    assert bad["double"]["components"] == []
    drawn = [ln.get_label() for ln in _drawn_curves(bad)]   # [5, 6] holds no point; the Landau of [0.7, 1.5] stays
    assert [t.split(" ")[0] for t in drawn] == ["fit", "Landau"] and drawn[1].endswith("(first range only)"), drawn


# Every label _draw_end_fits puts on an axis: curves and legend-only lines (a failed fit's reason).
def _legend_labels(f: dict) -> list:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        cd._draw_end_fits(ax, f)
        return [ln.get_label() for ln in ax.get_lines()]
    finally:
        plt.close(fig)


# failed spectrum fit is named on the page also next to a good fit
def test_failed_spectrum_fit_is_named_on_the_page_also_next_to_a_good_fit():
    # R3 (Lei 2026-10-02): a Gaussian, Landau or double-Gaussian diagnostic fit that fails draws no curve and no
    # value but the legend line '<model> fit failed on [a, b]: <reason>' in the place of its curve -- default fits, one
    # range, two ranges -- also when the other fit succeeded.  Reasons: the optimizer's message, a peak outside the
    # range, too few points, the range_gauss_fit status (e.g. a component stuck on its range bound: a failure of the
    # shared fit is shown as points + this line).  A page whose fits all succeed has no such line (default unchanged).
    from ldmx_hcal.common import fitting

    mip = _mip_values()
    s = dict(cd.DEFAULTS)
    real_fit, real_range = fitting.fit_curve, fitting.range_gauss_fit

    # fit_curve stand-in factory that fails one model in the way how
    def failing(model_name, how):
        # real fit, then the chosen model is turned into a failure
        def fit_curve(model, x, y, p0, *args, **kwargs):
            res = real_fit(model, x, y, p0, *args, **kwargs)
            if model is not getattr(fitting, model_name):
                return res
            if how == "message":
                nan = np.full(len(p0), np.nan)
                return fitting.FitResult(False, nan, nan, n_points=len(x),
                                         message="Optimal parameters not found: synthetic")
            params = res.params.copy()
            params[1 if model_name == "gauss" else 0] = 5.0          # converged, the peak outside the range
            return fitting.FitResult(True, params, res.errors, res.chi2, res.ndf, res.n_points)
        return patch.object(fitting, "fit_curve", fit_curve)

    # model names of the curves drawn on the page
    def curves(f):
        return [ln.get_label().split(" ")[0] for ln in _drawn_curves(f)]

    good = cd.end_spectrum_fits(mip, s)
    assert good["why"] == {"gauss": None, "landau": None} and not any("failed" in t for t in _legend_labels(good))
    assert curves(good) == ["Gauss", "Landau"] and len(_legend_labels(good)) == 2
    # default fits: the Gaussian fails (optimizer), the Landau stays -- and the reverse (peak outside the range)
    with failing("gauss", "message"):
        f = cd.end_spectrum_fits(mip, s)
    assert f["gauss"] is None and f["landau"] is not None and curves(f) == ["Landau"]
    assert f["why"]["gauss"] == "Optimal parameters not found: synthetic" and f["why"]["landau"] is None
    labels = _legend_labels(f)
    assert labels[0] == "Gauss fit failed on [0.7, 1.3]: Optimal parameters not found: synthetic", labels
    assert labels[1].startswith("Landau MPV=") and len(labels) == 2
    with failing("landau", "outside"):
        f = cd.end_spectrum_fits(mip, s)
    assert f["landau"] is None and curves(f) == ["Gauss"]
    assert _legend_labels(f)[1] == "Landau fit failed on [0.7, 1.3]: peak 5 outside [0.7, 1.3]"
    # one range: the same, with the user's range
    with failing("gauss", "outside"):
        f = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.7, 1.3]]})
    assert curves(f) == ["Landau"]
    assert _legend_labels(f)[0] == "Gauss fit failed on [0.7, 1.3]: peak 5 outside [0.7, 1.3]"
    assert _legend_labels(f)[1].startswith("Landau MPV=") and _legend_labels(f)[1].endswith("on [0.7, 1.3]")
    # nothing to fit: the old note first, then each model with its reason
    empty = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[5.0, 6.0]]})
    assert _legend_labels(empty) == ["no MIP peak fitted in [5, 6]",
                                     "Gauss fit failed on [5, 6]: 0 point(s) in the range (needs >= 5)",
                                     "Landau fit failed on [5, 6]: 0 point(s) in the range (needs >= 5)"]
    # two ranges: a double Gaussian that range_gauss_fit fails (here: a component stuck on its range bound) -> points
    # + its reason, next to the Landau of the first range that succeeded; no single-Gaussian line (not asked for)
    two = [[0.7, 1.5], [1.6, 2.5]]

    # range fit stand-in: a failed fit with component 2 on its range bound
    def stuck(*args, **kwargs):
        res = real_range(*args, **kwargs)
        return {**res, **{k: np.nan for k in res if k.startswith(("mu", "sigma", "area", "amplitude", "chi2"))},
                "ok": False, "components": [], "ndf": 0, "status": "fit_failed: component 2 mu on its range bound 2.5"}

    with patch.object(fitting, "range_gauss_fit", stuck):
        f = cd.end_spectrum_fits(mip, {**s, "fit_ranges": two, "fit_bins": [16, 18]})
    labels = _legend_labels(f)
    assert curves(f) == ["fit", "fit", "Landau"] and not any(t.startswith("Gauss") for t in labels), labels
    assert "double Gauss fit failed on [0.7, 1.5], [1.6, 2.5]: component 2 mu on its range bound 2.5" in labels
    # ... and a double Gaussian that converged next to a Landau that failed
    with failing("landau", "message"):
        f = cd.end_spectrum_fits(mip, {**s, "fit_ranges": two})
    assert f["double"]["ok"] and curves(f) == ["Gauss", "Gauss", "double"]
    assert _legend_labels(f)[-1] == ("Landau fit failed on [0.7, 1.5] (first range only): "
                                     "Optimal parameters not found: synthetic")
    bad = cd.end_spectrum_fits(mip, {**s, "fit_ranges": [[0.7, 1.5], [5.0, 6.0]], "fit_bins": [10, 10]})
    assert any(t.startswith("double Gauss fit failed on [0.7, 1.5], [5, 6]: [5, 6] has 0 bin(s)")
               for t in _legend_labels(bad)), _legend_labels(bad)


# fit does not touch the clean dataset and from saved redraws
def test_fit_does_not_touch_the_clean_dataset_and_from_saved_redraws():
    # --fit / --bin act on the diagnostics pdf only: the clean csv, its record (selection, selection hash, beam and
    # calibration identities), the kept-bar cache and clean_status are the same as without them; the results record
    # names them; -from-saved accepts them (no beam csv read) and redraws the pdf with the new fits.
    with _tmpdir() as tmp:
        tmp = Path(tmp)
        ds = _dataset(_run_beam(tmp, _synthetic_run(30)))
        ctx = _ctx(tmp, make_plots=True)
        _standard_calib(ctx)
        _run(ctx, [], [ds])
        csv_bytes = cd.clean_csv(ctx, ds, _csv(ds)).read_bytes()
        keys = ("input", "calibration_values", "selection", "selection_hash", "selection_flags", "counts", "cutflow")
        record = {k: read_json(cd.provenance_json(ctx, ds, _csv(ds)))[k] for k in keys}
        identity = cd._cache_identity(read_json(cd.provenance_json(ctx, ds, _csv(ds))))
        cache = {k: read_json(cd.cache_json(ctx, ds, _csv(ds)))[k] for k in identity}
        assert read_json(cd.results_json(ctx, ds, _csv(ds)))["diagnostics"]["fit_ranges"] is None
        seen = []
        real = cd.draw_diagnostics

        # spy of draw_diagnostics: records the fit ranges and bins it receives
        def spy(pdf, ds_, bars, summary, cutflow, settings, job, title_lines):
            seen.append((settings.get("fit_ranges"), settings.get("fit_bins")))
            return real(pdf, ds_, bars, summary, cutflow, settings, job, title_lines)

        with patch.object(cd, "draw_diagnostics", side_effect=spy):
            _, out = _run(ctx, ["-fit", "[0.6,1.6]", "-bin", "20"], [ds])
        assert "diagnostic per-end spectrum fits: --fit [0.6, 1.6]" in out and "does not depend on them" in out
        assert cd.clean_csv(ctx, ds, _csv(ds)).read_bytes() == csv_bytes
        assert {k: read_json(cd.provenance_json(ctx, ds, _csv(ds)))[k] for k in keys} == record
        assert {k: read_json(cd.cache_json(ctx, ds, _csv(ds)))[k] for k in cache} == cache
        assert cd.clean_status(ctx, ds, _csv(ds))[:2] == (True, "ok")
        assert cd.current_clean_csv(ctx, ds, _csv(ds)) == cd.clean_csv(ctx, ds, _csv(ds))
        diag = read_json(cd.results_json(ctx, ds, _csv(ds)))["diagnostics"]
        assert (diag["fit_ranges"], diag["fit_bins"]) == ([[0.6, 1.6]], [20]) and seen == [([[0.6, 1.6]], [20])]
        # -from-saved with two ranges: accepted, nothing of the beam csv read, the pdf redrawn with them
        seen.clear()
        with patch.object(cd, "iter_csv", side_effect=AssertionError("beam csv read")), \
                patch.object(cd, "copy_selected_rows", side_effect=AssertionError("pass 2")), \
                patch.object(cd, "draw_diagnostics", side_effect=spy):
            _, out = _run(ctx, ["-from-saved", "-fit", "[0.5,1.1],[1.1,2.5]", "-bin", "15,20"], [ds])
        assert "the result units from" in out and seen == [([[0.5, 1.1], [1.1, 2.5]], [15, 20])]
        results = read_json(cd.results_json(ctx, ds, _csv(ds)))
        assert results["drawn"].startswith("redrawn with -from-saved")
        assert results["diagnostics"]["fit_bins"] == [15, 20]
        assert cd.clean_csv(ctx, ds, _csv(ds)).read_bytes() == csv_bytes and cd.clean_status(ctx, ds, _csv(ds))[0]
        ctx.clean_input = True                                    # the consumers still read it
        assert _quiet(ctx.input_csv, ds, _csv(ds))[0] == cd.clean_csv(ctx, ds, _csv(ds))
        ctx.clean_input = False
        # -no-plots: nothing is drawn, and the log says the flags have no effect
        ctx.make_plots = False
        _, out = _run(ctx, ["-fit", "[0.6,1.6]"], [ds])
        assert "nothing is drawn, so they have no effect" in out
        assert cd.clean_csv(ctx, ds, _csv(ds)).read_bytes() == csv_bytes


# --------------------------------------------------------------------------
# real data (skipped when the 5000-event muon 4 subset is not on this machine)
# --------------------------------------------------------------------------
# real test subset matches the reference
def test_real_test_subset_matches_the_reference():
    # muon 4 / electron 4 test subsets of this machine with the test calibration: the stage selects exactly the rows of
    # the pandas reference, for both thresholds (read-only use of the real files; products in a temp folder).
    cfg = load_config()
    ctx_real = RunContext(cfg=cfg, test=True)
    calib = ctx_real.calibration_file()
    pairs = []
    for key, standard in (("muon 4", "muon"), ("electron 4", "otherparticle")):
        try:
            ds = resolve_datasets(cfg, [key])[0]
        except (SystemExit, IndexError):
            continue
        for csv in ctx_real.run_csvs(ds):
            subset = ctx_real.test_subset_csv(ds, csv)
            if subset.exists():
                pairs.append((subset, standard))
    if not pairs or not Path(calib).exists():
        raise unittest.SkipTest("no real test subset / test calibration on this machine")
    frame = pd.read_csv(calib)
    with _tmpdir() as tmp:
        out = Path(tmp) / "out.csv"
        for subset, standard in pairs:
            for mode, column in cd.THRESHOLD_COLUMNS.items():
                cal = cd.read_calibration(calib, column)
                bars, counts = cd.scan_beam_csv(subset, cal, standard, 2_000_000)
                cd.select_events(bars, standard, cd.selection_definition(standard, mode, cd.DEFAULTS))
                rows = bars.row[bars.in_clean]
                assert rows.tolist() == _reference_rows(subset, frame, standard, column), (subset.name, mode)
                keep = bars.in_clean
                cd.copy_selected_rows(subset, out, rows, (bars.pf_event[keep], bars.layer[keep], bars.strip[keep]),
                                      counts["input_rows"])
                lines = out.read_bytes().splitlines(keepends=True)
                with open(subset, "rb") as fh:
                    assert lines[0] == fh.readline()
                assert len(lines) == len(rows) + 1 and len(rows) > 0


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["clean_dataset"]))
