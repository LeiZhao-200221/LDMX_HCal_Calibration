"""Tests of the plots stage (ldmx_hcal.visualization.multiscale_plots): synthetic data, about a minute.

    python -X utf8 src/tests/run_all.py multiscale          (from the data_analysis folder)

Synthetic tests:
  * units and flags (Lei 2026-10-20): dist_event, dist_bar, dist_channel, event_display, bar_maps, layer, hitmap
    chosen with -only (groups distributions, energy); the scales s, pureci, adc_only, s_<sample> and arbitrary.
  * the values come from the producer cache: ci per-csv event arrays + bar records, adcpeak events + bar values;
    a run of two csv files is reduced into one figure set with event key = csv index << 32 | pf_event; the numbers
    equal a direct computation from the cache arrays (selected events only, per bar and per end).
  * the producer check: stale members, another scale, mode, input csv or calibration stop the run with the
    producer's rebuild command; adcpeak without -bar-csv values is refused.
  * the hit map from the beam csv files of the run; the pickers across csv files; the ci spectrum fit on dist_event.
  * -from-saved re-draws identical payloads and refuses what the cache does not hold; output routing (-output by
    name, no -output, -test names); the dry-run plan; settings checked before any input is read; complete pdfs.

Author: Lei
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
import runner  # noqa: E402
from ldmx_hcal.common import geometry as geo  # noqa: E402
from ldmx_hcal.common import io_utils, outputs  # noqa: E402
from ldmx_hcal.common.config import RunContext, dataset_registry  # noqa: E402
from ldmx_hcal.energy import charge_injection as ci  # noqa: E402
from ldmx_hcal.energy import tot_adc_peak as ta  # noqa: E402
from ldmx_hcal.visualization import multiscale_plots as mp  # noqa: E402

RUNS = {"electron_4GeV": ("electron", 4.0), "pion_4GeV": ("pion", 4.0)}
CI_PATHS = ("adc_only", "pureci", "s")


# ----------------------------------------------------------------------------------------------
# synthetic inputs and helpers
# ----------------------------------------------------------------------------------------------
# a RunContext under tmp; outputs () = --output alone, None = no --output, names = --output NAME ...
def _ctx(tmp: Path, test: bool = False, plots: dict | None = None, ci_cfg: dict | None = None, outputs_=(),
         chunksize: int = 7, clean: bool = False, dry_run: bool = False) -> RunContext:
    data = tmp / "data"
    cfg = {"paths": {"generated_data_dir": data, "calibration_dir": tmp / "calibration", "results_dir": tmp / "results",
                     "cache_dir": tmp / "cache", "analysis_files_dir": data / "analysis_files"},
           "datasets": {k: {"particle": p, "energy_gev": e} for k, (p, e) in RUNS.items()},
           "run": {"chunksize": chunksize, "test_events": 5000}, "plots": dict(plots or {}),
           "safety": {"protected_dirs": []}}
    if ci_cfg:
        cfg["charge_injection"] = ci_cfg
    return RunContext(cfg=cfg, test=test, workers=1, clean_input=clean, dry_run=dry_run,
                      outputs=None if outputs_ is None else list(outputs_))


# the Dataset of a run key
def _ds(ctx, key: str):
    return dataset_registry(ctx.cfg)[key]


# the runner options of every stage plus the plots options
def _args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    runner.add_common(p)
    mp.add_arguments(p)
    return p.parse_args(list(argv))


# resolve_settings with -only applied as the runner does (ctx.units)
def _settings(ctx, argv=()) -> dict:
    args = _args(argv)
    ctx.units = outputs.selected_units(args)
    return mp.resolve_settings(ctx, args)


# io_plan of the stage (ctx.units from -only as the runner sets it)
def _plan(ctx, argv, datasets) -> list[dict]:
    args = _args(argv)
    ctx.units = outputs.selected_units(args)
    return mp.io_plan(ctx, args, datasets)


# run the stage (ctx.units from -only as the runner sets it)
def _run(ctx, argv, datasets) -> None:
    args = _args(argv)
    ctx.units = outputs.selected_units(args)
    with contextlib.redirect_stdout(io.StringIO()):
        mp.run(ctx, args, datasets)


# run the stage with the pdf writer replaced by a recorder: {(run key, unit): payload}, '_paths': pdf paths
def _capture(ctx, argv, datasets) -> dict:
    captured = {"_paths": []}

    # record the payload instead of drawing
    def fake_write(unit, payload, path):
        name = Path(path).name
        key = next((d.key for d in datasets if name.startswith(f"{d.key}_")), name)
        captured[(key, unit)] = payload
        captured["_paths"].append(Path(path))
        return len(payload.get("events", [])) if unit == "event_display" else 0

    with patch.object(mp, "write_part", fake_write):
        _run(ctx, argv, datasets)
    return captured


# fn() must stop (SystemExit) with every text in the message; returns the message
def _refused(fn, *texts) -> str:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except SystemExit as exc:
        msg = str(exc)
        for t in texts:
            assert t in msg, (t, msg)
        return msg
    raise AssertionError(f"not refused (expected {texts})")


# pages of a pdf
def _pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", Path(path).read_bytes()))


# payloads equal: arrays bit for bit, text lines except the 're-drawn from' note of -from-saved
def _assert_same(a, b, where="payload") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (where, sorted(set(a) ^ set(b)))
        for k in a:
            _assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list) and a and all(isinstance(x, str) for x in a):
        assert [x for x in a if not x.startswith("re-drawn from")] == \
               [x for x in b if not x.startswith("re-drawn from")], where
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        b = np.asarray(b)
        assert a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), where
    elif dataclasses.is_dataclass(a):
        _assert_same(dataclasses.asdict(a), dataclasses.asdict(b), where)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), where
    else:
        assert a == b, (where, a, b)


# calibration table of this mode: 384 channels, threshold 120 ADC
def _calibration(ctx) -> Path:
    path = ctx.calibration_product()
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = geo.all_channels()
    cal["pedestal"] = 100.0
    cal["pedestal_sigma"] = 5.0
    cal["pedestal_range_with_20percent"] = 120.0
    cal["mpv"] = 50.0
    cal["mpv_status"] = "OK"
    cal.to_csv(path, index=False)
    return path


# synthetic bars of n events: (pf, layer, strip) with 4..20 distinct bars per event
def _bars(rng, n_events: int, first: int = 1) -> pd.DataFrame:
    allbars = geo.all_bars()[["layer", "strip"]].to_numpy()
    rows = []
    for ev in range(first, first + n_events):
        for layer, strip in allbars[rng.choice(len(allbars), size=int(rng.integers(4, 20)), replace=False)]:
            rows.append((ev, layer, strip))
    return pd.DataFrame(rows, columns=["pf_event", "layer", "strip"])


# beam csv of a run folder (one csv file of the run) from bars
def _beam_csv(ctx, ds, name: str, bars: pd.DataFrame, rng) -> Path:
    path = ctx.run_dir(ds) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = bars.copy()
    frame["adc_sum_end0"] = rng.integers(85, 700, len(frame))
    frame["adc_sum_end1"] = rng.integers(85, 700, len(frame))
    frame.to_csv(path, index=False)
    return path


# ci cache of one csv: npz (per event), bars npy, json record; returns the true per-bar table
def _ci_cache(ctx, ds, csv: Path, bars: pd.DataFrame, rng, tag: str = "", paths=CI_PATHS,
              record: dict | None = None) -> pd.DataFrame:
    npz, npy, meta = ci.event_cache_files(ctx, ds, csv, tag)
    rec = np.zeros(len(bars), dtype=ci.bar_record_dtype(paths))
    rec["pf_event"], rec["layer"], rec["strip"] = bars["pf_event"], bars["layer"], bars["strip"]
    truth = bars.copy()
    for p in paths:
        e0 = rng.gamma(2.0, 1.5, len(bars)).astype(np.float32)
        e1 = rng.gamma(2.0, 1.5, len(bars)).astype(np.float32)
        e1[::7] = np.nan
        bar = np.where(np.isfinite(e1), 0.5 * (e0 + e1), e0).astype(np.float32)
        bar[::11] = np.nan                                   # not counted on this path
        rec[f"bar_energy_{p}_mip"], rec[f"end0_energy_{p}_mip"], rec[f"end1_energy_{p}_mip"] = bar, e0, e1
        truth[f"bar_{p}"], truth[f"end0_{p}"], truth[f"end1_{p}"] = bar, e0, e1
    pf = np.unique(bars["pf_event"].to_numpy())
    arrays = {"pf_event": pf, "n_bars": bars.groupby("pf_event").size().reindex(pf).to_numpy(np.int64),
              "n_unrecoverable_bars": np.zeros(len(pf), np.int64),
              "bad_strip_fraction": np.where(np.arange(len(pf)) % 5 == 4, 0.5, 0.0),
              "saturated_unconverted_fraction": np.zeros(len(pf)), "n_events_in_file": np.int64(len(pf) + 3)}
    for p in paths:
        sums = truth.groupby("pf_event")[f"bar_{p}"].apply(lambda v: float(np.nansum(v.astype(np.float64))))
        arrays[f"energy_{p}"] = sums.reindex(pf).to_numpy(np.float64)
    with io_utils.ProductSet(meta) as ps:
        np.savez(ps.stage(npz), **arrays)
        np.save(ps.stage(npy), rec)
        base = {"schema": ci.EVENT_CACHE_SCHEMA, "mode": ctx.mode,
                "n_test_events": int(ctx.n_test_events) if ctx.test else None,
                "beam_csv": io_utils.file_signature(ctx.input_plan_item(ds, csv)["path"]),
                "calibration": {"sha256": io_utils.content_hash(ctx.calibration_file())[0]},
                "compute_paths": list(paths), "stage": "ci"}
        ps.commit({**base, **(record or {})})
    truth["selected"] = truth["pf_event"].map(dict(zip(pf, arrays["bad_strip_fraction"] <= 0.125)))
    return truth


# adcpeak cache of one csv: events csv + bar values csv + provenance; returns the true per-bar table
def _adcpeak_cache(ctx, ds, csv: Path, bars: pd.DataFrame, rng, tag: str = "", with_bars: bool = True,
                   record: dict | None = None) -> pd.DataFrame:
    events_csv, bars_csv = ta.saved_events_csv(ctx, ds, csv, tag), ta.bar_values_csv(ctx, ds, csv, tag)
    truth = bars.copy()
    truth["bar"] = rng.gamma(2.0, 30.0, len(bars))
    truth["end0"], truth["end1"] = rng.gamma(2.0, 15.0, len(bars)), rng.gamma(2.0, 15.0, len(bars))
    pf = np.unique(bars["pf_event"].to_numpy())
    selected = np.arange(len(pf)) % 4 != 3
    events = pd.DataFrame({"event_source": ctx.event_source(ds, csv), "pf_event": pf,
                           "arbitrary_energy": truth.groupby("pf_event")["bar"].sum().reindex(pf).to_numpy(),
                           "n_bars": truth.groupby("pf_event").size().reindex(pf).to_numpy()})
    for c in ta.EVENT_COLUMNS:
        if c not in events:
            events[c] = 0
    events["selected"] = selected
    truth["selected"] = truth["pf_event"].map(dict(zip(pf, selected)))
    kept = truth[truth["selected"]]
    with io_utils.ProductSet(ta.provenance_path(events_csv)) as ps:
        events[list(ta.EVENT_COLUMNS)].to_csv(ps.stage(events_csv), index=False)
        if with_bars:
            pd.DataFrame({"event_source": ctx.event_source(ds, csv), "pf_event": kept["pf_event"],
                          "layer": kept["layer"], "strip": kept["strip"], "bar_arbitrary_energy": kept["bar"],
                          "end0_arbitrary_energy": kept["end0"], "end1_arbitrary_energy": kept["end1"]}
                         ).to_csv(ps.stage(bars_csv), index=False)
        base = {"stage": "adcpeak", "mode": ctx.mode, "n_test_events": ctx.n_test_events if ctx.test else None,
                "input_path": str(ctx.input_plan_item(ds, csv)["path"]),
                "calibration_content": {"sha256": io_utils.content_hash(ctx.calibration_file())[0]}}
        ps.commit({**base, **(record or {})})
    return truth


# one run of two csv files (a.csv, b.csv) with beam csv, calibration and the ci cache: (ctx, ds, {csv: truth})
def _ci_run(tmp: Path, key: str = "pion_4GeV", seed: int = 1, n_events=(30, 25), **ctx_kw):
    ctx = _ctx(tmp, **ctx_kw)
    rng = np.random.default_rng(seed)
    _calibration(ctx)
    ds = _ds(ctx, key)
    truth = {}
    for name, n in zip(("a.csv", "b.csv"), n_events):
        bars = _bars(rng, n)
        csv = _beam_csv(ctx, ds, name, bars, rng)
        truth[name] = _ci_cache(ctx, ds, csv, bars, rng)
    return ctx, ds, truth


# expected reduction from the true bar tables: selected events, counted bars of one path
def _expected(truth: dict, path: str) -> dict:
    keys, energy, cells, values, layer_e = [], [], [], [], {}
    end_values, end_channels = [], []
    for i, name in enumerate(sorted(truth)):
        t = truth[name][truth[name]["selected"].astype(bool)]
        bar = t[f"bar_{path}"].to_numpy(np.float64)
        ok = np.isfinite(bar)
        c = t[ok]
        k = (np.int64(i) << 32) | c["pf_event"].to_numpy(np.int64)
        cell = mp.cell_index(c["layer"].to_numpy(), c["strip"].to_numpy())
        cells.append(cell)
        values.append(bar[ok])
        keys.append(np.unique((np.int64(i) << 32) | t["pf_event"].to_numpy(np.int64)))
        energy.append(t.groupby("pf_event")[f"bar_{path}"].apply(lambda v: float(np.nansum(v.astype(np.float64))))
                      .to_numpy())
        for kk, layer, e in zip(k, c["layer"].to_numpy(), bar[ok]):
            layer_e[(kk, layer)] = layer_e.get((kk, layer), 0.0) + e
        for end in (0, 1):
            v = c[f"end{end}_{path}"].to_numpy(np.float64)
            f = np.isfinite(v)
            end_values.append(v[f])
            end_channels.append(cell[f] * 2 + end)
    cells, values = np.concatenate(cells), np.concatenate(values)
    return {"keys": np.concatenate(keys), "energy": np.concatenate(energy),
            "cell_hits": np.bincount(cells, minlength=mp.N_CELLS),
            "cell_total": np.bincount(cells, weights=values, minlength=mp.N_CELLS), "values": np.sort(values),
            "layer_e": layer_e, "end_values": np.sort(np.concatenate(end_values)),
            "end_channels": np.bincount(np.concatenate(end_channels), minlength=mp.N_CHANNELS)}


# ----------------------------------------------------------------------------------------------
# units, flags, settings
# ----------------------------------------------------------------------------------------------
# geometry: 8-bar layers sit at positions 2..9; 12-bar layers fill the row
def test_geometry_layout():
    assert mp.PHYSICAL_MASK.shape == (19, 12)
    assert mp.PHYSICAL_MASK[0].tolist() == [False, False] + [True] * 8 + [False, False]
    assert mp.PHYSICAL_MASK[18].all()
    assert int(mp.PHYSICAL_MASK.sum()) == 192
    assert mp.cell_index(np.array([1]), np.array([0]))[0] == 2
    assert mp.cell_index(np.array([19]), np.array([11]))[0] == 18 * 12 + 11


# units, groups and scales; unknown units and scales are refused
def test_units_flags_and_names():
    assert mp.UNITS == ("dist_event", "dist_bar", "dist_channel", "event_display", "bar_maps", "layer", "hitmap")
    assert set(mp.ENERGY_SOURCES) >= {"s", "pureci", "adc_only", "arbitrary", "s_muon"}
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        assert _settings(ctx)["units"] == list(mp.UNITS)
        assert _settings(ctx, ["-only", "distributions"])["units"] == ["dist_event", "dist_bar", "dist_channel"]
        s = _settings(ctx, ["-only", "hitmap,event_display"])
        assert s["units"] == ["event_display", "hitmap"] and s["energy_units"] == ["event_display"]
        assert _settings(ctx, ["-only", "energy"])["units"] == list(mp.ENERGY_UNITS)
    for bad in (["-energy", "nope"], ["-only", "nope"], ["-only", "dist_event", "-only", "bars"]):
        _refused(lambda g=bad: _args(g))


# every setting is checked before any input is read; unknown yml keys stop the run
def test_settings_are_checked_before_any_input_is_read():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for bad in ({"n_event_displays": -1}, {"spectrum_quantile": 1.5}, {"event_colormap": "nope"},
                    {"variant": "a/b"}, {"robust_percentile": 120}):
            _refused(lambda b=bad: _settings(_ctx(tmp, plots=b)))
        _refused(lambda: _settings(_ctx(tmp, plots={"unit": ["dist_event"]})), "unknown setting", "unit")
        _refused(lambda: _settings(_ctx(tmp), ["-energy-window", "5", "1"]))
        ctx = _ctx(tmp)
        _refused(lambda: _run(ctx, ["-n-events", "-2"], [_ds(ctx, "pion_4GeV")]))
        assert not (tmp / "cache").exists() and not (tmp / "results").exists()


# producer tag: the tag of a default ci / adcpeak run, -variant, clean part with -clean-input
def test_producer_tag():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert mp.producer_tag(_ctx(tmp), _settings(_ctx(tmp))) == ""
        ctx = _ctx(tmp, ci_cfg={"scale_particle": ["pion", "muon"]})
        assert mp.producer_tag(ctx, _settings(ctx)) == "s_pion_muon"
        ctx = _ctx(tmp, clean=True)
        assert mp.producer_tag(ctx, _settings(ctx)) == "clean"
        assert mp.producer_tag(ctx, _settings(ctx, ["-variant", "s_muon"])) == "s_muon_clean"
        assert mp.producer_tag(ctx, _settings(ctx, ["-variant", "s_muon_clean"])) == "s_muon_clean"
        ctx = _ctx(tmp)
        assert mp.producer_tag(ctx, _settings(ctx, ["-energy", "arbitrary"])) == ""
        assert mp.producer_tag(ctx, _settings(ctx, ["-energy", "arbitrary", "-variant", "none"])) == "none"


# ----------------------------------------------------------------------------------------------
# reduction from the producer caches
# ----------------------------------------------------------------------------------------------
# a run of two csv files: the reduced numbers equal a direct computation from the ci cache, on every path
def test_ci_cache_reduction_equals_direct_computation():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, truth = _ci_run(Path(tmp))
        for path in CI_PATHS:
            cap = _capture(ctx, ["-energy", path, "-only", "energy", "-n-events", "0"], [ds])
            exp = _expected(truth, path)
            p_event, p_bar = cap[(ds.key, "dist_event")], cap[(ds.key, "dist_bar")]
            assert np.array_equal(p_event["values"], exp["energy"])
            assert p_event["overview"]["n"] == len(exp["keys"])
            assert np.array_equal(p_bar["n"], exp["cell_hits"])
            assert np.allclose(p_bar["total"], exp["cell_total"], rtol=1e-12)
            assert int(p_bar["counts"].sum() + p_bar["overflow"].sum()) == len(exp["values"])
            p_chan = cap[(ds.key, "dist_channel")]
            assert np.array_equal(p_chan["n"], exp["end_channels"])
            p_layer = cap[(ds.key, "layer")]
            total_layer = sum(exp["layer_e"].values())
            assert np.isclose(p_layer["e_mean"].sum() * len(exp["keys"]), total_layer, rtol=1e-9)
            maps = cap[(ds.key, "bar_maps")]["maps"]
            assert np.isclose(np.nansum(maps["occupancy"]) * len(exp["keys"]), exp["cell_hits"].sum())
            assert "a.csv, b.csv" in " ".join(p_event["lines"])
            assert p_event["fit"] is not None and p_event["particle"] == "pion"


# adcpeak: events with selected = true, bar values per csv; Arbitrary Energy labels
def test_adcpeak_cache_reduction():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx(tmp)
        rng = np.random.default_rng(3)
        _calibration(ctx)
        ds = _ds(ctx, "electron_4GeV")
        truth = {}
        for name in ("a.csv", "b.csv"):
            bars = _bars(rng, 20)
            csv = _beam_csv(ctx, ds, name, bars, rng)
            truth[name] = _adcpeak_cache(ctx, ds, csv, bars, rng)
        cap = _capture(ctx, ["-energy", "arbitrary", "-only", "dist_event,dist_bar,dist_channel"], [ds])
        sel = [t[t["selected"]] for _, t in sorted(truth.items())]
        energy = np.concatenate([t.groupby("pf_event")["bar"].sum().to_numpy() for t in sel])
        p = cap[(ds.key, "dist_event")]
        assert np.allclose(p["values"], energy, rtol=1e-12) and p["fit"] is None
        assert p["axis"] == ta.ARBITRARY_ENERGY and p["unit"] == "a.u."
        assert int(cap[(ds.key, "dist_bar")]["n"].sum()) == sum(len(t) for t in sel)
        assert int(cap[(ds.key, "dist_channel")]["n"].sum()) == 2 * sum(len(t) for t in sel)
        assert cap["_paths"][0].name == "electron_4GeV_dist_event_arbitrary.pdf"


# the producer check: missing / tampered / other scale / mode / input / calibration stop with the rebuild command
def test_producer_check_refuses_with_the_rebuild_command():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx, ds, _ = _ci_run(tmp / "a")
        _refused(lambda: _run(ctx, ["-energy", "s_muon", "-only", "dist_event"], [ds]), "not s_muon",
                 ".\\runner.ps1 ci --pion 4 -only events")
        npz = ci.event_cache_files(ctx, ds, ctx.run_dir(ds) / "a.csv")[0]
        with open(npz, "ab") as fh:
            fh.write(b"x")
        _refused(lambda: _run(ctx, ["-only", "dist_event"], [ds]), "a.csv", "changed since it was committed")
        ctx, ds, _ = _ci_run(tmp / "b")
        _calibration(ctx).write_text("layer,strip,end\n", encoding="utf-8")
        _refused(lambda: _run(ctx, ["-only", "dist_bar"], [ds]), "another calibration table")
        ctx, ds, _ = _ci_run(tmp / "c")
        test_ctx = _ctx(tmp / "c", test=True)
        _refused(lambda: _run(test_ctx, ["-only", "dist_bar"], [ds]), "no metadata")
        ctx = _ctx(tmp / "d")
        rng = np.random.default_rng(5)
        _calibration(ctx)
        ds = _ds(ctx, "pion_4GeV")
        bars = _bars(rng, 10)
        csv = _beam_csv(ctx, ds, "a.csv", bars, rng)
        _ci_cache(ctx, ds, csv, bars, rng, record={"beam_csv": {"path": str(tmp / "other.csv")}})
        _refused(lambda: _run(ctx, ["-only", "layer"], [ds]), "this run reads")
        _adcpeak_cache(ctx, ds, csv, bars, rng, with_bars=False)
        _refused(lambda: _run(ctx, ["-energy", "arbitrary", "-only", "dist_bar"], [ds]), "-bar-csv",
                 ".\\runner.ps1 adcpeak --pion 4 -only energy_response -bar-csv")
        _refused(lambda: _run(ctx, ["-energy", "arbitrary", "-variant", "none", "-only", "dist_bar"], [ds]),
                 "no metadata", "variant 'none'")
        empty = _ds(ctx, "electron_4GeV")
        _refused(lambda: _run(ctx, ["-only", "dist_event"], [empty]), "no csv file", "make-csv --electron 4")


# the hit map: both / per-end hit fractions over the beam csv files of the run; first events across csv files
def test_hitmap_from_the_beam_csv_files():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, _ = _ci_run(Path(tmp), n_events=(8, 6))
        cap = _capture(ctx, ["-only", "hitmap", "-hitmap-events", "10"], [ds])
        p = cap[(ds.key, "hitmap")]
        frames = [pd.read_csv(ctx.run_dir(ds) / n) for n in ("a.csv", "b.csv")]
        assert p["n_events"] == 14
        both = np.zeros(mp.N_CELLS)
        for f in frames:
            hit = (f["adc_sum_end0"] > 120) & (f["adc_sum_end1"] > 120)
            np.add.at(both, mp.cell_index(f["layer"][hit].to_numpy(), f["strip"][hit].to_numpy()), 1)
        grid = both.reshape(19, 12) / 14
        assert np.allclose(np.nan_to_num(p["grids"]["both"]), np.where(mp.PHYSICAL_MASK, grid, 0))
        assert len(p["first_events"]) == 10 and p["first_names"][8] == "Event 1 (b.csv)"
        assert cap["_paths"][0].name == "pion_4GeV_hitmap.pdf"


# pickers across csv files: first, highest, random, explicit ids (every csv), energy window
def test_pickers_across_csv_files():
    keys = np.array([1, 2, 3, (1 << 32) | 1, (1 << 32) | 5], dtype=np.int64)
    energy = np.array([5.0, 1.0, 9.0, 7.0, 3.0])
    s = {"explicit_events": None, "n_event_displays": 2, "event_picker": "first", "energy_window": None,
         "random_seed": 1}
    assert mp.pick_events(keys, energy, "MIP", s)[0].tolist() == [1, 2]
    assert mp.pick_events(keys, energy, "MIP", {**s, "event_picker": "highest"})[0].tolist() == [3, (1 << 32) | 1]
    assert len(mp.pick_events(keys, energy, "MIP", {**s, "event_picker": "random"})[0]) == 2
    picked = mp.pick_events(keys, energy, "MIP", {**s, "explicit_events": [1, 99]})[0]
    assert picked.tolist() == [1, (1 << 32) | 1, 99]
    assert mp.pick_events(keys, energy, "MIP", {**s, "energy_window": [4.0, 8.0]})[0].tolist() == [1, (1 << 32) | 1]
    assert mp.event_name((1 << 32) | 5, ["a.csv", "b.csv"]) == "Event 5 (b.csv)"
    assert mp.event_name(5, ["a.csv"]) == "Event 5"
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, truth = _ci_run(Path(tmp))
        cap = _capture(ctx, ["-only", "event_display", "-picker", "highest", "-n-events", "3", "-per-end"], [ds])
        p = cap[(ds.key, "event_display")]
        exp = _expected(truth, "s")
        top = exp["keys"][np.lexsort((exp["keys"], -exp["energy"]))[:3]]
        assert [e["id"] for e in p["events"]] == top.tolist()
        for ev in p["events"]:
            assert np.isclose(np.nansum(ev["grid"]), ev["total"], rtol=1e-6) and ev["ends"] is not None


# ----------------------------------------------------------------------------------------------
# -from-saved, outputs, plans, pdfs
# ----------------------------------------------------------------------------------------------
# -from-saved re-draws identical payloads without reading the producer cache; refuses what it does not hold
def test_from_saved_redraws_identical_payloads():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, _ = _ci_run(Path(tmp))
        argv = ["-n-events", "4", "-per-end"]
        first = _capture(ctx, argv, [ds])
        with patch.object(mp, "csv_bars", side_effect=AssertionError("read")), \
                patch.object(mp, "csv_events", side_effect=AssertionError("read")), \
                patch.object(mp, "iter_beam_csv", side_effect=AssertionError("read")):
            again = _capture(ctx, argv + ["-from-saved"], [ds])
        assert first.keys() == again.keys()
        for k in first:
            _assert_same(first[k], again[k], str(k))
        _refused(lambda: _run(ctx, ["-n-events", "9", "-picker", "highest", "-from-saved", "-only", "event_display"],
                              [ds]), "does not hold")
        _capture(ctx, ["-only", "dist_event"], [ds])
        _refused(lambda: _run(ctx, ["-from-saved", "-only", "dist_bar"], [ds]), "did not draw dist_bar")
        csv = ctx.run_dir(ds) / "a.csv"
        _ci_cache(ctx, ds, csv, _bars(np.random.default_rng(9), 5), np.random.default_rng(9))  # ci ran again
        _refused(lambda: _run(ctx, ["-from-saved", "-only", "dist_event"], [ds]), "'inputs' differs")


# output routing: no -output (full set: scratch copy), -output by name, -test names; the dry-run plan
def test_output_routing_and_plan():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, _ = _ci_run(Path(tmp), outputs_=None)
        cap = _capture(ctx, ["-only", "dist_event,hitmap"], [ds])
        assert all("unsaved" in p.parts for p in cap["_paths"]), cap["_paths"]
        ctx.outputs = ["pion_4GeV_dist_event_s.pdf"]
        cap = _capture(ctx, ["-only", "dist_event,hitmap"], [ds])
        assert cap["_paths"][0] == ctx.results_dir("multiscale_plots") / "pion_4GeV_dist_event_s.pdf"
        assert "unsaved" in cap["_paths"][1].parts
        ctx.outputs = []
        plan = _plan(ctx, ["-only", "dist_bar,hitmap"], [ds])
        ins = [str(i["path"]) for i in plan if i["role"] == "in"]
        assert any(p.endswith("a.npz") for p in ins) and any(p.endswith("b_bars.npy") for p in ins)
        assert any(p.endswith("calibration_parameters.csv") for p in ins) and any(p.endswith("b.csv") for p in ins)
        outs = [Path(i["path"]).name for i in plan if i["role"] == "out"]
        assert "pion_4GeV_dist_bar_s.pdf" in outs and "pion_4GeV_hitmap.pdf" in outs
        assert "pion_4GeV_energy_s.npz" in outs
        assert not any(i.get("stale") for i in plan)
        saved = _plan(ctx, ["-only", "dist_bar", "-from-saved"], [ds])
        assert [Path(i["path"]).name for i in saved if i["role"] == "cache"] == ["pion_4GeV_energy_s.json"]
        assert not any(i["role"] == "in" for i in saved)


# -no-plots draws nothing and reads nothing
def test_no_plots_draws_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.make_plots = False
        _run(ctx, [], [_ds(ctx, "pion_4GeV")])
        assert not (Path(tmp) / "results").exists() and not (Path(tmp) / "cache").exists()


# complete pdfs of every unit (small data): file names and page counts
def test_end_to_end_pdfs():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, ds, _ = _ci_run(Path(tmp), n_events=(12, 10))
        _run(ctx, ["-energy", "pureci", "-n-events", "2", "-per-end", "-hitmap-events", "2"], [ds])
        folder = ctx.results_dir("multiscale_plots")
        pages = {p.name: _pdf_pages(p) for p in sorted(folder.glob("*.pdf"))}
        assert pages == {"pion_4GeV_bar_maps_pureci.pdf": 4, "pion_4GeV_dist_bar_pureci.pdf": 20,
                         "pion_4GeV_dist_channel_pureci.pdf": 39, "pion_4GeV_dist_event_pureci.pdf": 3,
                         "pion_4GeV_event_display_pureci.pdf": 7, "pion_4GeV_hitmap.pdf": 5,
                         "pion_4GeV_layer_pureci.pdf": 4}, pages
        assert not list(folder.glob("*.tmp*"))


# -no-json: plots reads the producer caches (their records stay in cache/) and draws as before
def test_no_json_reads_the_producer_cache_records():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx, ds, _ = _ci_run(tmp)
        ctx.write_json = False
        io_utils.set_json_records(False, [tmp / "data", tmp / "calibration", tmp / "results"], [tmp / "cache"])
        try:
            cap = _capture(ctx, ["-only", "dist_event"], [ds])
            assert (ds.key, "dist_event") in cap
            assert list((tmp / "cache" / "multiscale_plots").glob("*.json"))
            assert not list((tmp / "results").rglob("*.json"))
        finally:
            io_utils.set_json_records(True)
