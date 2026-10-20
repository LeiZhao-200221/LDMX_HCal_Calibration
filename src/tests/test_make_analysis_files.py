"""Tests of the make-csv stage (conversion/make_analysis_files.py): ROOT file -> beam csv, one csv per ROOT file.

Synthetic ROOT ntuples are written with uproot into temporary folders; the expected values come from a
transcription of Joe Muse's conversion rules (apply_calibrations.tot_calib / toa_calib, the makeAnalysisFiles
groupby + inner join).  Covers
  columns      the raw TOT / TOA sums are always written, after the 15 per-bar columns (+ waveform)
  runs         a run of datasets.yml is the folder data/analysis_files/<Particle>_<E>GeV/ with one csv per ROOT file,
               named after it (nothing merged); -test: data/test/analysis_files/<run>/
  -input       one csv per file directly in data/analysis_files/ (-test: data/test/analysis_files/); -output-csv
               names the csv of ONE -input file (a file name only)
  -only        units csv and qa; -only qa draws the QA pdf from the existing csv
  -no-json     no provenance json; the test csv is then read without verification
  reuse rule   current_test_csv: the -test csv is reused only with a matching provenance
  readers      LDMX_Events + --tot-mode tot_comp, one chunk schema, early refusal of tot_ne_adc
  raw sums     taken before the single-hit / 2048 rules, inside the sample window, independent of tot_mode
  settings     every make_csv setting / flag is validated before any file is opened; unknown yml keys refused
  product set  csv + provenance written sidecar-last (failure injection)
  TOA table    keys follow the shared channel-key rule; bx_shift / mean_shift by name or at positions 4 / 5
  QA pdfs      the final results of make-csv (common/outputs.py): scratch copy without --output, -output by result
               name (KEY=FILE), test names with -test
plus streaming across chunk borders and repeated digis (error by default, drop).

Author: Lei
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import io_utils  # noqa: E402
from ldmx_hcal.common import outputs as outputs_mod  # noqa: E402
from ldmx_hcal.common.config import (REPO_ROOT, RunContext, dataset_registry, load_config,  # noqa: E402
                                     resolve_datasets)
from ldmx_hcal.conversion import make_analysis_files as m  # noqa: E402

try:
    import uproot  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"uproot not installed: {exc}")

BAR_HEADER = ("pf_event,tot_end0,toa_end0,adc_sum_end0,adc_mean_end0,adc_max_end0,layer,strip,pf_spill,pf_ticks,"
              "tot_end1,toa_end1,adc_sum_end1,adc_mean_end1,adc_max_end1")
RAW_SUMS = ["raw_tot_sum_end0", "raw_tot_sum_end1", "raw_toa_sum_end0", "raw_toa_sum_end1"]
DEFAULT_HEADER = BAR_HEADER + "," + ",".join(RAW_SUMS)
BARS = [(1, 0), (1, 3), (1, 7), (2, 1), (2, 4), (3, 2), (10, 0), (10, 11), (19, 5)]
NO_TOA_ROW_WITH_HITS = (2, 1)     # no TOA-table row and TOA hits -> whole bar dropped
NO_TOA_ROW_NO_HITS = (3, 2)       # no TOA-table row, never a TOA hit -> kept, toa = 0
RUN = "Muon_4GeV"                 # run folder of the synthetic run syn (muon 4)
QUIET = contextlib.redirect_stdout


# --------------------------------------------------------------------------
# reference rules (transcribed from Joe Muse's src/apply_calibrations.py and src/makeAnalysisFiles.py)
# --------------------------------------------------------------------------
# reference tot_calib (Joe Muse): 2048 -> 0; a TOT only when exactly one sample is non-zero
def ref_tot_calib(column_data):
    result = []
    for this_val in column_data:
        if this_val != 0:
            if this_val == 2048:
                this_val = 0
            result.append(this_val)
    return 0 if len(result) != 1 else sum(result)


# toa_calib; ``first`` = position of column_data[0] in the digi (k counts from sample 0).
def ref_toa_calib(column_data, calibration_data, end, first=0):
    result = []
    for k, this_val in enumerate(column_data, start=first):
        if this_val != 0:
            this_bx = 25 * k
            if this_val < calibration_data[0][4]:
                this_bx = 25 * (k + 1)
            result.append(((this_val / 1023) * 25) + this_bx)
    if len(result) != 1:
        return 0
    return sum(result) - calibration_data[0][5] if end == 1 else sum(result)


# reference conversion: per bar and end, groupby(pf_event) -> tot/toa/adc rules -> join
def reference_convert(long: pd.DataFrame, toa_table: pd.DataFrame | None, window=(0, None)) -> pd.DataFrame:
    # Per bar: per end groupby(pf_event) in file order -> tot/toa/adc rules -> inner join on pf_event.
    # A bar without a TOA row raises IndexError on its first TOA hit; the bare except drops it.
    frames = []
    for (layer, strip), g in long.groupby(["layer", "strip"], sort=True):
        rows = ([[0, 0, 0, 0, 0.0, 0.0]] if toa_table is None else
                toa_table[(toa_table["layer"] == layer - 1) & (toa_table["bar"] == strip)].values.tolist())
        try:
            ends = []
            for end in (0, 1):
                recs = []
                for ev, ge in g[g["end"] == end].groupby("pf_event", sort=True):
                    lo, hi = window
                    ge = ge.iloc[lo:hi]
                    adc = ge["adc"].tolist()
                    recs.append({"pf_event": ev, "layer": layer, "strip": strip,
                                 "pf_spill": int(ge["pf_spill"].iloc[0]), "pf_ticks": int(ge["pf_ticks"].iloc[0]),
                                 "tot": ref_tot_calib(ge["tot"].tolist()),
                                 "toa": ref_toa_calib(ge["toa"].tolist(), rows, end, first=lo),
                                 "adc_sum": sum(adc), "adc_mean": sum(adc) / len(adc), "adc_max": max(adc),
                                 "raw_tot_sum": int(ge["tot"].sum()), "raw_toa_sum": int(ge["toa"].sum())})
                ends.append(pd.DataFrame(recs))
        except IndexError:
            continue
        if not len(ends[0]) or not len(ends[1]):
            continue
        df = pd.merge(ends[0], ends[1], on="pf_event", suffixes=("_end0", "_end1"))
        df = df.drop(columns=["layer_end0", "strip_end0", "pf_spill_end0", "pf_ticks_end0"])
        df = df.rename(columns={"layer_end1": "layer", "strip_end1": "strip", "pf_spill_end1": "pf_spill",
                                "pf_ticks_end1": "pf_ticks"})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[m.beam_csv_columns()]


# --------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------
# event-major long table (one row per sample) with every TOT / TOA case
def synthetic_long(n_events=14, seed=7, first_event=1, n_samples=8) -> pd.DataFrame:
    # Event-major long table (one row per sample); random bar subsets (not dense), random channel order,
    # some digis without the other end, TOT cases: none / single / several / 2048, TOA: none / one / two.
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("pf_event", "layer", "strip", "end", "i_sample", "adc", "tot", "toa", "pf_spill",
                            "pf_ticks", "tot_comp")}
    for ev in range(first_event, first_event + n_events):
        chosen = [b for b in BARS if rng.random() < 0.85]
        digis = [(b, e) for b in chosen for e in (0, 1) if rng.random() > 0.06]
        for i in rng.permutation(len(digis)):
            (layer, strip), end = digis[i]
            adc = rng.integers(0, 1024, n_samples)
            tot = np.zeros(n_samples, int)
            n_tot = rng.choice([0, 0, 1, 1, 2])
            tot[rng.choice(n_samples, n_tot, replace=False)] = rng.choice([7, 900, 2048, 4095], n_tot)
            toa = np.zeros(n_samples, int)
            if (layer, strip) != NO_TOA_ROW_NO_HITS:
                n_toa = rng.choice([0, 1, 1, 2])
                toa[rng.choice(n_samples, n_toa, replace=False)] = rng.integers(1, 1024, n_toa)
            tot_comp = (tot != 0) & (rng.random(n_samples) < 0.75)
            tot_comp |= rng.random(n_samples) < 0.03
            for k in range(n_samples):
                for name, v in (("pf_event", ev), ("layer", layer), ("strip", strip), ("end", end), ("i_sample", k),
                                ("adc", adc[k]), ("tot", tot[k]), ("toa", toa[k]), ("pf_spill", ev // 4),
                                ("pf_ticks", 100 * ev + end), ("tot_comp", tot_comp[k])):
                    cols[name].append(v)
    df = pd.DataFrame(cols)
    for c in df.columns:
        df[c] = df[c].astype(bool if c == "tot_comp" else np.int32)
    return df


# write the long digi table as a ROOT ntuple
def write_ntuple(path: Path, long: pd.DataFrame, with_tot_comp=True) -> Path:
    import uproot

    path.parent.mkdir(parents=True, exist_ok=True)
    branches = {c: long[c].to_numpy() for c in long.columns if with_tot_comp or c != "tot_comp"}
    with uproot.recreate(path) as f:
        f.mkdir("ntuplizehgcroc")
        f["ntuplizehgcroc"]["hgcroc"] = branches
    return path


# write a TOA calibration table (two bars without a row)
def write_toa_table(path: Path, offset=0.0) -> pd.DataFrame:
    rows = []
    for layer, strip in BARS:
        if (layer, strip) in (NO_TOA_ROW_WITH_HITS, NO_TOA_ROW_NO_HITS):
            continue
        rows.append({"layer": layer - 1, "bar": strip, "Det_id_0": 1, "Det_id_1": 0,
                     "bx_shift": [465, 720, 978][strip % 3], "mean_shift": round(-3.7779 + strip + offset, 6)})
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


# config.yml of the synthetic setup under tmp
def write_config(tmp: Path, extra: str = "") -> Path:
    t = tmp.as_posix()
    text = f"""
paths:
  data_dir: {t}/data
  root_dir: {t}/root
  analysis_files_dir: {t}/data/analysis_files
  ci_scan_dir: {t}/ci
  channel_map: {(REPO_ROOT / 'configure' / 'channel_map.csv').as_posix()}
  toa_calibration: {t}/toa.csv
  generated_data_dir: {t}/data
  calibration_dir: {t}/cal
  results_dir: {t}/res
  cache_dir: {t}/cache
safety:
  protected_dirs: [{t}/protected]
run:
  test_events: 5000
  chunksize: 1000
  workers: 1
make_csv:
  chunk_entries: 50
{extra}
datasets:
  syn:
    label: "synthetic"
    particle: muon
    energy_gev: 4
    kind: data
    root_files: ["${{root_dir}}/syn.root"]
"""
    path = tmp / "configure" / "config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# run the runner silently; returns its console text
def run_runner(argv: list[str]) -> str:
    import runner

    out = io.StringIO()
    with QUIET(out):
        code = runner.main(argv)
    assert code == 0, out.getvalue()
    return out.getvalue()


# synthetic ROOT file, TOA table and config in a temp folder
@contextlib.contextmanager
def synthetic_setup(n_events=14, with_tot_comp=True, extra=""):
    with tempfile.TemporaryDirectory(prefix="ldmx_makecsv_") as td:
        tmp = Path(td)
        long = synthetic_long(n_events)
        write_ntuple(tmp / "root" / "syn.root", long, with_tot_comp)
        table = write_toa_table(tmp / "toa.csv")
        cfg = write_config(tmp, extra)
        try:
            yield tmp, cfg, long, table
        finally:
            io_utils.set_protected_dirs([])
            io_utils.set_json_records(True)


# read a csv with exact float values
def read_csv_exact(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


# the given columns are equal row by row
def assert_frames_equal(got: pd.DataFrame, want: pd.DataFrame, columns):
    assert len(got) == len(want), (len(got), len(want))
    for c in columns:
        a, b = got[c].to_numpy(np.float64), want[c].to_numpy(np.float64)
        assert np.array_equal(a, b), (c, np.flatnonzero(a != b)[:5], a[a != b][:5], b[a != b][:5])


# rows of the first n events in file order
def first_events(long: pd.DataFrame, n: int) -> pd.DataFrame:
    ev = long["pf_event"].to_numpy()
    keep = pd.unique(ev)[:n]
    return long[np.isin(ev, keep)]


# temp files left in folder
def temp_leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in Path(folder).rglob("*") if ".tmp" in p.name or ".old-" in p.name)


# A bare conversion job (no run): one ROOT file -> one csv.
def job(root: Path, out: Path, limit=None, mc=False) -> m.Job:
    return m.Job("j", "j", Path(root), Path(out), mc, limit)


# The full-set csv of the (one-file) run: data/analysis_files/<run>/<ROOT name>.csv.
def run_csv(ctx, ds) -> Path:
    return ctx.run_csvs(ds)[0]


# -test run folder: data/test/analysis_files/<run>
def tmp_test_dir(tmp: Path, run: str = RUN) -> Path:
    return tmp / "data" / "test" / "analysis_files" / run


# full-set run folder: data/analysis_files/<run>
def full_dir(tmp: Path, run: str = RUN) -> Path:
    return tmp / "data" / "analysis_files" / run


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
# columns
def test_columns():
    assert ",".join(m.beam_csv_columns()) == DEFAULT_HEADER
    wf = m.beam_csv_columns(waveform=True)
    assert len(wf) == 35 and wf[6] == "adc_0_end0" and wf[30] == "adc_7_end1" and wf[31:] == RAW_SUMS
    assert m.RAW_SUM_COLUMNS == RAW_SUMS and m.DEFAULTS["waveform"] is False
    assert m.DEFAULTS["duplicate_digis"] == "error" and m.DEFAULTS["tot_mode"] == "tot_branch"
    assert m.UNITS == ("csv", "qa") and m.DEFAULT_UNITS == ("csv",)


# tot / toa / adc aggregation of the fast and the generic path == the reference Python loops.
def test_vectorised_rules_equal_reference_functions():
    rng = np.random.default_rng(11)
    G, S = 4000, 8
    layer = rng.integers(1, 20, G)
    strip = np.array([rng.integers(0, 8 if lay <= 9 else 12) for lay in layer])
    end = rng.integers(0, 2, G)
    tot = np.zeros((G, S), np.int32)
    for j, n in enumerate(rng.choice([0, 1, 2, 3], G, p=[0.5, 0.35, 0.1, 0.05])):
        tot[j, rng.choice(S, n, replace=False)] = rng.choice([2048, 5, 900, 4095], n)
    toa = np.zeros((G, S), np.int32)
    for j, n in enumerate(rng.choice([0, 1, 2], G, p=[0.4, 0.5, 0.1])):
        toa[j, rng.choice(S, n, replace=False)] = rng.integers(1, 1024, n)
    adc = rng.integers(0, 1024, (G, S)).astype(np.int32)
    c = {"pf_event": np.repeat(np.arange(G), S), "layer": np.repeat(layer, S).astype(np.int32),
         "strip": np.repeat(strip, S).astype(np.int32), "end": np.repeat(end, S).astype(np.int32),
         "i_sample": np.tile(np.arange(S), G).astype(np.int32), "adc": adc.ravel(), "tot": tot.ravel(),
         "toa": toa.ravel(), "pf_spill": np.zeros(G * S, np.int32), "pf_ticks": np.zeros(G * S, np.int32)}
    settings = {**m.DEFAULTS, "window": (0, None)}
    stats = m._new_stats()
    fast = m._digis_regular(c, S, m._pack_key(c), settings, True, stats)
    perm = rng.permutation(G)
    cs = {k: v.reshape(G, S)[perm].ravel() for k, v in c.items()}
    gen = m._digis_generic(cs, m._pack_key(cs), settings, True, stats)  # key-sorted = original digi order
    assert np.array_equal(gen["event"], np.arange(G))
    table = pd.DataFrame({"layer": np.arange(19).repeat(12), "bar": np.tile(np.arange(12), 19), "d0": 0, "d1": 0,
                          "bx_shift": rng.choice([465, 720, 978], 228), "mean_shift": rng.normal(0, 10, 228)})
    cal = m.load_toa_calibration(None)
    cal.toa_bx_wrap_code[table["layer"] + 1, table["bar"]] = table["bx_shift"]
    cal.toa_end1_offset_ns[table["layer"] + 1, table["bar"]] = table["mean_shift"]
    for d in (fast, gen):
        ref_tot = [ref_tot_calib(tot[i].tolist()) for i in range(G)]
        assert np.array_equal(d["tot"], ref_tot)
        assert np.array_equal(d["adc_sum"], adc.sum(1)) and np.array_equal(d["adc_max"], adc.max(1))
        assert np.array_equal(d["raw_tot_sum"], tot.sum(1)) and np.array_equal(d["raw_toa_sum"], toa.sum(1))
        assert np.array_equal(d["waveform"], adc)
        for i in range(0, G, 7):
            row = table[(table["layer"] == layer[i] - 1) & (table["bar"] == strip[i])].values.tolist()
            want = ref_toa_calib(toa[i].tolist(), row, int(end[i]))
            bx, off, _ = cal.constants(int(layer[i]), int(strip[i]))
            got = m.toa_to_ns(d["toa_raw"][i], d["toa_k"][i], bx, off if end[i] == 1 else None)
            assert float(got) == float(want), (i, got, want)


# sample window keeps the toa time origin
def test_sample_window_keeps_the_toa_time_origin():
    # --sample-window only selects the samples used: k still counts from sample 0 (toa_calib on the whole
    # digi), so a single TOA hit inside the window gives the same toa [ns] as without the window (both paths).
    G, S, lo, hi = 6, 8, 2, 6
    toa = np.zeros((G, S), np.int32)
    toa[np.arange(G), [2, 3, 4, 5, 2, 5]] = [100, 500, 900, 7, 1000, 600]  # one hit per digi, inside 2..5
    c = {"pf_event": np.repeat(np.arange(G), S), "layer": np.full(G * S, 1, np.int32),
         "strip": np.full(G * S, 0, np.int32), "end": np.repeat(np.arange(G) % 2, S).astype(np.int32),
         "i_sample": np.tile(np.arange(S), G).astype(np.int32), "adc": np.full(G * S, 100, np.int32),
         "tot": np.zeros(G * S, np.int32), "toa": toa.ravel(), "pf_spill": np.zeros(G * S, np.int32),
         "pf_ticks": np.zeros(G * S, np.int32)}
    key = m._pack_key(c)
    full = {**m.DEFAULTS, "window": (0, None)}
    win = {**m.DEFAULTS, "window": (lo, hi)}
    want = m._digis_regular(c, S, key, full, False, m._new_stats())["toa_k"]
    assert want.tolist() == [2, 3, 4, 5, 2, 5]
    for digis in (m._digis_regular(c, S, key, win, False, m._new_stats()),
                  m._digis_generic(c, key, win, False, m._new_stats())):
        assert digis["toa_k"].tolist() == want.tolist()
        got = m.toa_to_ns(digis["toa_raw"], digis["toa_k"], 465.0)
        assert np.array_equal(got, m.toa_to_ns(toa.max(1), want, 465.0))


# synthetic run end to end matches reference
def test_synthetic_run_end_to_end_matches_reference():
    # runner make-csv --test / full set on a synthetic run: reference values + raw sums, bar drops, the run folder,
    # provenance (sidecar-last product set).
    with synthetic_setup() as (tmp, cfg, long, table):
        run_runner(["make-csv", "--muon", "4", "--test", "--test-events", "9", "--config", str(cfg), "--workers", "1"])
        out = tmp_test_dir(tmp) / "syn.csv"
        assert out.exists() and not full_dir(tmp).exists()
        text = out.read_bytes().decode()
        assert text.split("\r\n")[0] == DEFAULT_HEADER and "\r\n" in text
        got = read_csv_exact(out)
        assert_frames_equal(got, reference_convert(first_events(long, 9), table), m.beam_csv_columns())
        bars = set(zip(got["layer"], got["strip"]))
        assert NO_TOA_ROW_WITH_HITS not in bars and NO_TOA_ROW_NO_HITS in bars
        assert (got.loc[got["strip"].eq(2) & got["layer"].eq(3), ["toa_end0", "toa_end1"]] == 0).all().all()
        record = json.loads(m.provenance_path(out).read_text(encoding="utf-8"))
        assert record["format_version"] == m.CSV_FORMAT_VERSION and record["dataset"] == "syn"
        assert record["run_folder"] == RUN and record["event_limit"] == 9 and record["events"] == 9
        assert record["n_samples"] == 8 and record["n_samples_used"] == 8
        assert record["waveform"] is False
        assert record["columns"] == m.beam_csv_columns()
        assert record["input"]["size"] == (tmp / "root" / "syn.root").stat().st_size
        assert record["settings_hash"] == io_utils.settings_hash(record["csv_settings"])
        assert record["protocol"] == io_utils.PRODUCT_SET_PROTOCOL and list(record["members"]) == ["syn.csv"]
        assert record["members"]["syn.csv"]["sha256"] == io_utils.content_hash(out)[0]
        assert record["output"] == io_utils.file_signature(out)
        assert io_utils.product_set_valid(m.provenance_path(out), members=[out]) == (True, "ok")
        assert temp_leftovers(tmp) == []
        # full set: into the run folder data/analysis_files/Muon_4GeV/, the csv named after the ROOT file
        run_runner(["make-csv", "--muon", "4", "--config", str(cfg), "--workers", "1"])
        full = full_dir(tmp) / "syn.csv"
        assert_frames_equal(read_csv_exact(full), reference_convert(long, table), m.beam_csv_columns())
        ctx = RunContext(cfg=load_config(cfg))
        ds = resolve_datasets(ctx.cfg, ["muon 4"])[0]
        assert ctx.run_csvs(ds) == [full]
        # an existing csv is skipped without --overwrite, replaced with it; -max-events without -test makes the
        # run's csv itself, with a WARNING
        stamp = full.stat().st_mtime_ns
        log = run_runner(["make-csv", "--muon", "4", "--config", str(cfg), "--workers", "1", "--max-events", "4"])
        assert "not overwritten" in log and full.stat().st_mtime_ns == stamp and "formal csv" not in log
        log = run_runner(["make-csv", "--muon", "4", "--config", str(cfg), "--workers", "1", "--overwrite",
                          "--max-events", "4"])
        assert "WARNING: [muon 4] the formal csv syn.csv is made from the first 4 events only" in log, log
        assert_frames_equal(read_csv_exact(full), reference_convert(first_events(long, 4), table),
                            m.beam_csv_columns())
        assert json.loads(m.provenance_path(full).read_text(encoding="utf-8"))["event_limit"] == 4
        assert sorted(p.name for p in full_dir(tmp).iterdir()) == ["syn.csv", "syn.provenance.json"]


# a run with two root files gives two csv
def test_a_run_with_two_root_files_gives_two_csv():
    # Nothing is merged: every ROOT file of a run becomes its own csv in the run folder (both may start at
    # event 1); each has its own test csv, reused by the later test stages per csv.
    with synthetic_setup(n_events=5) as (tmp, cfg, long, table):
        second = synthetic_long(4, seed=3)
        write_ntuple(tmp / "root" / "syn_b.root", second)
        text = cfg.read_text(encoding="utf-8")
        cfg.write_text(text.replace('root_files: ["${root_dir}/syn.root"]',
                                    'root_files: ["${root_dir}/syn.root", "${root_dir}/syn_b.root"]'), encoding="utf-8")
        base = ["--config", str(cfg), "--workers", "1"]
        run_runner(["make-csv", "--muon", "4"] + base)
        a, b = full_dir(tmp) / "syn.csv", full_dir(tmp) / "syn_b.csv"
        assert_frames_equal(read_csv_exact(a), reference_convert(long, table), m.beam_csv_columns())
        assert_frames_equal(read_csv_exact(b), reference_convert(second, table), m.beam_csv_columns())
        assert read_csv_exact(a)["pf_event"].min() == read_csv_exact(b)["pf_event"].min() == 1
        run_runner(["make-csv", "--muon", "4", "--test", "--test-events", "3"] + base)
        ctx = RunContext(cfg=load_config(cfg), test=True, n_test_events=3)
        ds = resolve_datasets(ctx.cfg, ["muon 4"])[0]
        assert ctx.run_csvs(ds) == [a, b]
        made = [tmp_test_dir(tmp) / "syn.csv", tmp_test_dir(tmp) / "syn_b.csv"]
        with QUIET(io.StringIO()):
            assert [m.current_test_csv(ctx, ds, c) for c in (a, b)] == made
            assert [ctx.beam_csv(ds, c) for c in (a, b)] == made
        assert json.loads(m.provenance_path(tmp_test_dir(tmp) / "syn_b.csv").read_text())["events"] == 3
        # the same ROOT file listed twice would write one csv twice: refused before any conversion
        cfg.write_text(text.replace('root_files: ["${root_dir}/syn.root"]',
                                    'root_files: ["${root_dir}/syn.root", "${root_dir}/syn.root"]'), encoding="utf-8")
        _refused(["make-csv", "--muon", "4", "-overwrite"] + base, "would both write")


# Events crossing chunk borders stay complete; exactly the first N events in file order.
def test_streaming_independent_of_chunk_size():
    with synthetic_setup(n_events=20) as (tmp, cfg, long, table):
        cal = m.load_toa_calibration(tmp / "toa.csv")
        texts = set()
        for chunk_entries in (1, 7, 64, 333, 10 ** 6):
            settings = {**m.DEFAULTS, "window": (0, None), "chunk_entries": chunk_entries,
                        "toa_calibration": tmp / "toa.csv"}
            with QUIET(io.StringIO()):
                store, stats = m.convert(job(tmp / "root" / "syn.root", tmp / "x.csv", 11), settings)
                m.write_beam_csv(store, tmp / f"chunk{chunk_entries}.csv", cal, settings, stats)
            assert stats["events"] == 11
            texts.add((tmp / f"chunk{chunk_entries}.csv").read_bytes())
        assert len(texts) == 1
        got = read_csv_exact(tmp / "chunk1.csv")
        assert_frames_equal(got, reference_convert(first_events(long, 11), table), m.beam_csv_columns())


# raw sums before the single hit rules
def test_raw_sums_before_the_single_hit_rules():
    # The raw sums are taken before the single-hit / 2048 rules, inside the sample window; the generic path
    # (shuffled digis) gives the same csv.
    S = 8
    cases = [  # (tot samples, toa samples) of end 0; end 1 always quiet
        ([0] * 8, [0] * 8),                        # true pedestal: kept by the strict selection
        ([20, 30, 0, 0, 0, 0, 0, 0], [0] * 8),     # two TOT hits: tot_end0 = 0 but raw sum 50
        ([2048, 0, 0, 0, 0, 0, 0, 0], [0] * 8),    # 2048 sentinel: tot_end0 = 0 but raw sum 2048
        ([0] * 8, [0, 300, 0, 400, 0, 0, 0, 0]),   # two TOA hits: toa_end0 = 0 but raw sum 700
        ([0, 7, 0, 0, 0, 0, 0, 0], [0, 0, 99, 0, 0, 0, 0, 0]),  # single hits
    ]
    rows = []
    for ev, (tot, toa) in enumerate(cases, start=1):
        for end in (0, 1):
            for k in range(S):
                rows.append({"pf_event": ev, "layer": 1, "strip": 3, "end": end, "i_sample": k, "adc": 100 + k,
                             "tot": tot[k] if end == 0 else 0, "toa": toa[k] if end == 0 else 0,
                             "pf_spill": 0, "pf_ticks": 0, "tot_comp": False})
    long = pd.DataFrame(rows).astype({c: np.int32 for c in rows[0] if c != "tot_comp"})
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = write_ntuple(tmp / "ped.root", long)
        cal = m.load_toa_calibration(None)
        outs = {}
        for window in ((0, None), (2, 8)):
            settings = {**m.DEFAULTS, "window": window, "toa_calibration": "none"}
            with QUIET(io.StringIO()):
                store, stats = m.convert(job(src, tmp / "x.csv"), settings)
                path = tmp / f"out_{window[0]}.csv"
                m.write_beam_csv(store, path, cal, settings, stats)
            outs[window[0]] = path
        assert outs[0].read_bytes().decode().split("\r\n")[0] == DEFAULT_HEADER
        df = read_csv_exact(outs[0])
        assert df["tot_end0"].tolist() == [0, 0, 0, 0, 7]
        assert df["raw_tot_sum_end0"].tolist() == [0, 50, 2048, 0, 7]
        assert df["raw_toa_sum_end0"].tolist() == [0, 0, 0, 700, 99]
        assert (df[["raw_tot_sum_end1", "raw_toa_sum_end1"]] == 0).all().all()
        strict = (df[m.RAW_SUM_COLUMNS] == 0).all(axis=1)
        approx = (df[["tot_end0", "tot_end1", "toa_end0", "toa_end1"]] == 0).all(axis=1)
        assert strict.tolist() == [True, False, False, False, False]  # pedestal selection on the raw words
        assert approx.tolist() == [True, True, True, True, False]    # what the tot / toa columns alone allow
        win = read_csv_exact(outs[2])  # samples 2..7 only
        assert win["raw_tot_sum_end0"].tolist() == [0, 0, 0, 0, 0]
        assert win["raw_toa_sum_end0"].tolist() == [0, 0, 0, 400, 99]
        shuffled = pd.concat([g for _, g in long.groupby(["pf_event", "end"])][::-1], ignore_index=True)
        src2 = write_ntuple(tmp / "ped2.root", shuffled)
        settings = {**m.DEFAULTS, "window": (0, None), "toa_calibration": "none"}
        with QUIET(io.StringIO()):
            store, stats = m.convert(job(src2, tmp / "x.csv"), settings)
            m.write_beam_csv(store, tmp / "gen.csv", cal, settings, stats)
        assert (tmp / "gen.csv").read_bytes() == outs[0].read_bytes()


# sample window synthetic matches reference
def test_sample_window_synthetic_matches_reference():
    with synthetic_setup() as (tmp, cfg, long, table):
        run_runner(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1",
                    "-sample-window", "1:7"])
        out = tmp_test_dir(tmp) / "syn.csv"
        assert_frames_equal(read_csv_exact(out), reference_convert(long, table, window=(1, 7)), m.beam_csv_columns())
        record = json.loads(m.provenance_path(out).read_text(encoding="utf-8"))
        assert record["columns"][-4:] == m.RAW_SUM_COLUMNS
        assert record["n_samples"] == 8 and record["n_samples_used"] == 6


# In-memory LDMX_Events with HcalDigis_sim (packed HGCROC words, 8 samples, version 0).
class FakeEventTree:

    num_entries = 3
    PREFIX = "HcalDigis_sim"

    # events with quiet, single-hit and TOT-complete digis
    def __init__(self):
        # packed HGCROC word: tot_complete << 30 | adc << 10 | toa
        def word(adc, toa=0, tot_complete=False):
            return (int(tot_complete) << 30) | (adc << 10) | toa
        quiet = [word(100)] * 8
        single = list(quiet)
        single[3] = word(150, tot_complete=True)          # one TOT-complete sample: TOT = 150
        single[2] = word(120, toa=500)
        double = list(quiet)
        double[2] = word(300, tot_complete=True)          # two TOT-complete samples: tot = 0 (single-TOT rule)
        double[4] = word(310, tot_complete=True)
        bar_id = lambda layer, strip, end: (end << 19) | (layer << 8) | strip  # noqa: E731
        self.events = [  # (event number, [(channel id, 8 words), ...])
            (1, [(bar_id(1, 0, 0), single), (bar_id(1, 0, 1), quiet)]),
            (2, [(bar_id(1, 0, 0), quiet), (bar_id(1, 0, 1), quiet), (bar_id(2, 5, 0), double),
                 (bar_id(2, 5, 1), single)]),
            (3, [(bar_id(2, 5, 0), quiet)]),  # one end only: dropped by the inner join
        ]

    # branch names of the tree
    def keys(self, recursive=True):
        return ["EventHeader", self.PREFIX]

    # branches of the events in [entry_start, entry_stop) as awkward arrays
    def arrays(self, names, entry_start=0, entry_stop=None, library="ak"):
        import awkward as ak

        evs = self.events[entry_start:entry_stop]
        return ak.Array({
            names[0]: [[cid for cid, _ in digis] for _, digis in evs],
            names[1]: [[w for _, words in digis for w in words] for _, digis in evs],
            names[2]: [8] * len(evs), names[3]: [0] * len(evs), names[4]: [n for n, _ in evs],
        })


# _open_input stand-in that returns the in-memory tree
def _fake_open(tree):
    return lambda path, settings, quiet=False: (contextlib.nullcontext(), "events", tree)


# LDMX_Events + --tot-mode tot_comp works; both readers deliver the same chunk schema.
def test_event_file_tot_comp_and_schema():
    tree = FakeEventTree()
    settings = {**m.DEFAULTS, "window": (0, None), "tot_mode": "tot_comp"}
    stats = m._new_stats()
    with QUIET(io.StringIO()):
        chunk = next(m._iter_event_file(tree, settings, None, stats))
    assert set(m.CHUNK_FIELDS) <= set(chunk) and "tot_comp" in chunk and "i_sample" in chunk
    m._check_chunk_schema(chunk, settings, "fake")
    digis = m._digis_regular(chunk, 8, m._pack_key(chunk), settings, False, stats)
    assert digis["tot"].tolist() == [150, 0, 0, 0, 0, 150, 0]
    results = {}
    for mode in ("tot_branch", "tot_comp"):
        s = {**m.DEFAULTS, "window": (0, None), "tot_mode": mode, "toa_calibration": "none"}
        j = job(Path("fake.root"), Path("unused.csv"))
        with patch.object(m, "_open_input", _fake_open(tree)), QUIET(io.StringIO()):
            assert m.validate_inputs(j, s) == "events"
            store, st = m.convert(j, s)
        assert st["mc"] is True and st["samples_per_digi"] == 8 and st["unpaired_digis"] == 1
        rows = {(store.slots.layer[k], store.slots.strip[k]): store.bar_rows(k) for k in store.slots.ordered()
                if k < len(store.rows) and store.rows[k]}
        results[mode] = rows
        assert rows[(1, 0)]["event"].tolist() == [1, 2] and rows[(1, 0)]["tot0"].tolist() == [150, 0]
        assert rows[(2, 5)]["tot0"].tolist() == [0] and rows[(2, 5)]["tot1"].tolist() == [150]
        assert rows[(1, 0)]["adc_sum0"].tolist() == [100 * 6 + 150 + 120, 800]
    for key in results["tot_branch"]:  # identical for event files
        for col, v in results["tot_branch"][key].items():
            assert np.array_equal(v, results["tot_comp"][key][col])
    # tot_ne_adc would erase every TOT of an event file: refused before reading data
    s = {**m.DEFAULTS, "window": (0, None), "tot_mode": "tot_ne_adc"}
    for call in (m.validate_inputs, m.convert):
        with patch.object(m, "_open_input", _fake_open(tree)):
            try:
                call(job(Path("fake.root"), Path("unused.csv")), s)
            except SystemExit as exc:
                assert "tot_ne_adc" in str(exc)
            else:
                raise AssertionError("tot_ne_adc on an event file must be refused")


# ntuple tot modes
def test_ntuple_tot_modes():
    with synthetic_setup(with_tot_comp=False) as (tmp, cfg, long, table):
        j = job(tmp / "root" / "syn.root", tmp / "x.csv")
        try:
            m.validate_inputs(j, {**m.DEFAULTS, "tot_mode": "tot_comp"})
        except SystemExit as exc:
            assert "tot_comp" in str(exc)
        else:
            raise AssertionError("tot_comp without the branch must be refused before reading")
        write_ntuple(tmp / "root" / "syn.root", long, with_tot_comp=True)
        for mode, source in (("tot_comp", np.where(long["tot_comp"], long["tot"], 0)),
                             ("tot_ne_adc", np.where(long["tot"] != long["adc"], long["tot"], 0))):
            settings = {**m.DEFAULTS, "window": (0, None), "tot_mode": mode, "toa_calibration": tmp / "toa.csv"}
            with QUIET(io.StringIO()):
                assert m.validate_inputs(j, settings) == "ntuple"
                store, stats = m.convert(j, settings)
                m.write_beam_csv(store, tmp / f"{mode}.csv", m.load_toa_calibration(tmp / "toa.csv"), settings, stats)
            got = read_csv_exact(tmp / f"{mode}.csv")
            assert_frames_equal(got, reference_convert(long.assign(tot=source), table), m.beam_csv_columns()[:15])
            # the raw sums come from the raw tot branch, whatever --tot-mode selects for tot_end*
            assert_frames_equal(got, reference_convert(long, table), RAW_SUMS)


# The --test csv is reused only while every recorded fact still holds; with -no-json it is read unverified.
def test_current_test_csv_reuse_rule():
    with synthetic_setup() as (tmp, cfg, long, table):
        run_runner(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1", "--test-events", "6"])
        made = tmp_test_dir(tmp) / "syn.csv"
        prov = m.provenance_path(made)

        # -test RunContext and run syn for n events (extra: extra yml lines)
        def ctx_for(n=6, extra=""):
            c = write_config(tmp, extra) if extra else cfg
            ctx = RunContext(cfg=load_config(c), test=True, n_test_events=n)
            return ctx, resolve_datasets(ctx.cfg, ["syn"])[0]

        # the reusable -test csv for n events, or None
        def current(n=6, extra=""):
            ctx, ds = ctx_for(n, extra)
            with QUIET(io.StringIO()):
                return m.current_test_csv(ctx, ds, run_csv(ctx, ds))

        assert current() == made
        ctx, ds = ctx_for()
        assert run_csv(ctx, ds) == full_dir(tmp) / "syn.csv"  # no full-set csv yet: the expected name
        assert ctx.input_csv(ds, run_csv(ctx, ds)) == made  # RunContext uses it
        assert m.current_test_csv(ctx, ds, full_dir(tmp) / "other.csv") is None  # another csv of the run
        assert current(n=5) is None  # other event count
        for extra in ("  waveform: true", "  sample_window: '0:4'", "  tot_mode: tot_comp", "  toa_calibration: none",
                      "  duplicate_digis: drop"):
            assert current(extra=extra) is None, extra
            write_config(tmp)
        # chunking does not change the csv; 'os' = crlf on Windows
        same = ("  chunk_entries: 7",) + (("  line_terminator: os",) if os.linesep == "\r\n" else ())
        for extra in same:
            assert current(extra=extra) == made, extra
            write_config(tmp)
        # TOA table content changed
        write_toa_table(tmp / "toa.csv", offset=0.5)
        assert current() is None
        write_toa_table(tmp / "toa.csv")
        os.utime(tmp / "toa.csv", ns=(json.loads(prov.read_text())["csv_settings"]["toa_calibration"]["mtime_ns"],) * 2)
        assert current() == made
        # ROOT input touched / replaced / pointed elsewhere
        root = tmp / "root" / "syn.root"
        st = root.stat()
        os.utime(root, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        assert current() is None
        os.utime(root, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert current() == made
        other = tmp / "root2" / "syn.root"
        other.parent.mkdir()
        other.write_bytes(root.read_bytes())
        text = cfg.read_text(encoding="utf-8")
        cfg.write_text(text.replace("${root_dir}/syn.root", other.as_posix()), encoding="utf-8")
        assert current() is None
        cfg.write_text(text, encoding="utf-8")
        assert current() == made
        # provenance: corrupt / other format / other dataset / other converter code or pandas / missing
        good = prov.read_text(encoding="utf-8")
        rec = json.loads(good)
        assert rec["converter"] == m.converter_fingerprint()
        for bad in ("{", json.dumps({**rec, "format_version": 0}), json.dumps({**rec, "dataset": "other"}),
                    json.dumps({**rec, "input": None}),
                    json.dumps({**rec, "converter": {**rec["converter"], "code": "0"}}),
                    json.dumps({**rec, "converter": {**rec["converter"], "pandas": "0.0"}}),
                    json.dumps({k: v for k, v in rec.items() if k != "converter"}),
                    json.dumps({k: v for k, v in rec.items() if k not in ("protocol", "members")})):
            prov.write_text(bad, encoding="utf-8")
            assert current() is None
        prov.write_text(json.dumps({**rec, "converter": {**rec["converter"], "code": "0"}}), encoding="utf-8")
        ctx, ds = ctx_for()
        reason = m.provenance_mismatch(made, "syn", ds.root_files[0], 6,
                                       m.csv_settings(m.configured_settings(ctx), False))
        assert reason.startswith("made by another converter (code") and "pandas" not in reason, reason
        prov.unlink()
        assert current() is None
        prov.write_text(good, encoding="utf-8")
        assert current() == made
        # the csv content changed with the same size and modification time: caught by the content hash
        data = bytearray(made.read_bytes())
        st = made.stat()
        j = len(data) // 2
        while not chr(data[j]).isdigit():
            j += 1
        data[j] = ord("1") if data[j] != ord("1") else ord("2")  # one digit in the middle of the file
        made.write_bytes(bytes(data))
        os.utime(made, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert made.stat().st_size == st.st_size and made.stat().st_mtime_ns == st.st_mtime_ns
        assert current() is None
        assert m.provenance_mismatch(made, "syn", ds.root_files[0], 6,
                                     m.csv_settings(m.configured_settings(ctx), False)
                                     ) == "the csv was changed after make-csv wrote it"
        # a stale csv is not rewritten without --overwrite, but the run says why it does not match
        log = run_runner(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1",
                          "--test-events", "6"])
        assert "not overwritten" in log and "does not match" in log
        run_runner(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1", "--test-events", "6",
                    "--overwrite"])
        assert current() == made
        # -no-json: no provenance json is written; the test csv is then read without verification
        run_runner(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1", "--test-events", "6",
                    "--overwrite", "-no-json"])
        assert made.exists() and not prov.exists()
        ctx = RunContext(cfg=load_config(cfg), test=True, n_test_events=6, write_json=False)
        with QUIET(io.StringIO()):
            assert m.current_test_csv(ctx, ds, run_csv(ctx, ds)) == made
        ctx = RunContext(cfg=load_config(cfg), test=True, n_test_events=6)  # json on: no record, not reused
        with QUIET(io.StringIO()):
            assert m.current_test_csv(ctx, ds, run_csv(ctx, ds)) is None


# routing with this machine config
def test_routing_with_this_machine_config():
    # With the real configure/: every planned csv of a run goes into its run folder of paths.analysis_files_dir
    # (-test: data/test/analysis_files/<run>/), the QA pdfs to their result destinations; nothing is planned inside a
    # protected folder (plan only, no data read).
    import runner

    cfg = load_config()
    protected = [Path(p).resolve() for p in cfg["safety"]["protected_dirs"]]
    parser = runner.build_parser()
    for argv in (["make-csv"], ["make-csv", "--test"], ["make-csv", "--muon", "4", "-waveform", "-overwrite"],
                 ["make-csv", "--muon", "4", "-max-events", "10"],
                 ["make-csv", "--muon", "4", "-test", "-only", "csv,qa", "-overwrite"],
                 ["make-csv", "--muon", "4", "-only", "csv", "qa"], ["make-csv", "--muon", "4", "-only", "all",
                                                                     "-output"]):
        args = parser.parse_args(argv)
        try:
            ctx = runner.make_context(args)
            ctx.units = outputs_mod.selected_units(args)
            with QUIET(io.StringIO()):
                plan = m.io_plan(ctx, args, runner.datasets_for(ctx, m, args))
        finally:
            io_utils.set_protected_dirs([])
        outs = [Path(p["path"]).resolve() for p in plan if p["role"] == "out"]
        if "--muon" not in argv and not any(d.root_files and all(Path(r).exists() for r in d.root_files)
                                             for d in dataset_registry(cfg).values()):
            assert not outs, argv   # no ROOT file on this machine (e.g. a fresh clone): bare make-csv converts nothing
            continue
        assert outs, argv
        for p in outs:
            assert not any(d == p or d in p.parents for d in protected), (argv, p)
            if p.suffix == ".pdf":  # the QA pdf, the final result
                if ctx.test:  # -test without --output: results/test/test.pdf
                    assert p == ctx.results_dir().resolve() / "test.pdf", p
                elif ctx.outputs is None:  # full set without --output: the scratch copy, not saved
                    assert p.parent == ctx.cache_dir("make_csv_qa", "unsaved").resolve(), p
                else:
                    assert p.parent.parent == ctx.results_dir("make_csv_qa").resolve(), p
            else:
                assert p.parent.parent == ctx.analysis_files_dir().resolve(), p
        csvs = [p for p in outs if p.suffix == ".csv"]
        assert csvs and all(p.with_name(p.stem + ".provenance.json") in outs for p in csvs)


# output csv names one input file
def test_output_csv_names_one_input_file():
    # -output-csv takes a csv FILE NAME only (argparse refuses a folder part, a drive, '..', a leading '-' and a
    # name without .csv); it names the csv of ONE -input file; a run's csv is always named after its ROOT file; a
    # protected output folder is refused before any data is read.
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        base = ["--config", str(cfg), "--workers", "1"]
        src = str(tmp / "root" / "syn.root")
        for name in (str(tmp / "data" / "new.csv"), "sub/x.csv", "sub\\x.csv", "..", "C:x.csv", "x.txt", "x", ""):
            _argparse_refused(["make-csv", "-input", src, "--output-csv", name, "--overwrite"] + base,
                              "argument --output-csv")
        msg = _argparse_refused(["make-csv", "-input", src, "--output-csv=-x.csv"] + base, "FILE NAME only")
        assert "data/analysis_files/" in msg and "data/test/analysis_files/ with -test" in msg, msg
        for name in ("x.csv", "a b.csv", "muon_12321_v2.CSV"):
            assert m._csv_file_name(name) == name
        _refused(["make-csv", "--muon", "4", "--output-csv", "x.csv"] + base, "--output-csv names the csv of one "
                 "-input file", "named after its ROOT file")
        other = _copy(tmp / "root" / "syn.root", tmp / "root" / "other.root")
        _refused(["make-csv", "-input", src, str(other), "--output-csv", "y.csv"] + base,
                 "--output-csv names one csv, so it takes one -input file; this run converts 2")
        log = run_runner(["make-csv", "-input", src, "--output-csv", "x.csv", "--max-events", "3"] + base)
        made = tmp / "data" / "analysis_files"
        assert (made / "x.csv").exists() and (made / "x.provenance.json").exists() and "formal csv" not in log
        run_runner(["make-csv", "-input", src, "--test", "--output-csv", "x.csv", "--max-events", "3"] + base)
        assert (tmp / "data" / "test" / "analysis_files" / "x.provenance.json").exists()
        # a protected output folder: refused by ensure_writable before any data is read
        text = cfg.read_text(encoding="utf-8").replace(f"protected_dirs: [{tmp.as_posix()}/protected]",
                                                       f"protected_dirs: [{tmp.as_posix()}/data/analysis_files]")
        cfg.write_text(text, encoding="utf-8")
        _refused(["make-csv", "-input", src, "--output-csv", "y.csv"] + base, "protected")
        _refused(["make-csv", "--muon", "4"] + base, "protected")
        assert not (made / "y.csv").exists() and not full_dir(tmp).exists()


# Run the runner and expect a refusal (SystemExit naming ``words``) before any conversion.
def _refused(argv: list[str], *words: str) -> str:
    with patch.object(m, "convert", side_effect=AssertionError("must not convert")):
        try:
            run_runner(argv)
        except SystemExit as exc:
            for w in words:
                assert w in str(exc), (argv, w, str(exc))
            return str(exc)
    raise AssertionError(f"{argv} must be refused")


# Run the runner and expect argparse to stop it (exit code 2, the reason on stderr naming ``words``).
def _argparse_refused(argv: list[str], *words: str) -> str:
    err = io.StringIO()
    with patch.object(m, "convert", side_effect=AssertionError("must not convert")), contextlib.redirect_stderr(err):
        try:
            run_runner(argv)
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code, err.getvalue())
        else:
            raise AssertionError(f"{argv} must be refused by argparse")
    for w in words:
        assert w in err.getvalue(), (argv, w, err.getvalue())
    return err.getvalue()


# copy a file, creating the target folder
def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


# input files one csv each
def test_input_files_one_csv_each():
    # -input FILES: one csv per file, named after the file, directly in data/analysis_files/ (-test:
    # data/test/analysis_files/), never in a run folder; two files with one name are refused; the QA result key
    # replaces characters outside A-Z a-z 0-9 _ . - by _.
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        base = ["--config", str(cfg), "--workers", "1"]
        a = _copy(tmp / "root" / "syn.root", tmp / "in" / "a.root")
        b = _copy(tmp / "root" / "syn.root", tmp / "in" / "b.root.root")
        run_runner(["make-csv", "-input", str(a), str(b), "--max-events", "2"] + base)
        made = tmp / "data" / "analysis_files"
        assert sorted(p.name for p in made.iterdir()) == ["a.csv", "a.provenance.json", "b.csv", "b.provenance.json"]
        assert read_csv_exact(made / "a.csv").equals(read_csv_exact(made / "b.csv"))
        assert json.loads((made / "a.provenance.json").read_text())["run_folder"] is None
        run_runner(["make-csv", "-input", str(a), "--test", "--max-events", "2"] + base)
        assert (tmp / "data" / "test" / "analysis_files" / "a.csv").is_file()
        twin = _copy(a, tmp / "in2" / "a.root")
        _refused(["make-csv", "-input", str(a), str(twin), "-overwrite"] + base, "would both write")
        spaced = _copy(a, tmp / "in" / "my run.root")
        dry = run_runner(["make-csv", "--test", "-input", str(spaced), "-only", "csv,qa", "--dry-run"] + base)
        assert "my run.csv" in dry, dry
        assert m.qa_key(Path("my run.csv")) == "my_run_qa"


# formal csv is made with the given settings and a warning
def test_formal_csv_is_made_with_the_given_settings_and_a_warning():
    # A run's csv (data/analysis_files/<run>/<ROOT name>.csv, read by every full-set stage) is made with what was
    # given; its provenance records how, and a WARNING names the settings that differ from configure/config.yml and
    # the event limit (also in the dry-run plan).  No WARNING in test mode, for -input files, for options without csv
    # effect, for a flag that repeats the configured value, or when the existing csv is skipped (no -overwrite).
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        base = ["--config", str(cfg), "--workers", "1"]
        made = full_dir(tmp) / "syn.csv"
        warning = "WARNING: [muon 4] the formal csv syn.csv is made"
        differ = "with settings that differ from configure/config.yml ({})"
        for flags, key in ((["-tot-mode", "tot_comp"], "tot_mode"), (["-waveform"], "waveform"),
                           (["-sample-window", "1:7"], "window"), (["-toa-calibration", "none"], "toa_calibration"),
                           (["-mc"], "mc"), (["-duplicate-digis", "drop"], "duplicate_digis")):
            log = run_runner(["make-csv", "--muon", "4"] + flags + base + ["--dry-run"])
            assert f"{warning} {differ.format(key)}\n" in log, (flags, log)
        log = run_runner(["make-csv", "--muon", "4", "--max-events", "2", "--dry-run"] + base)
        assert f"{warning} from the first 2 events only" in log, log
        log = run_runner(["make-csv", "--muon", "4", "-max-events", "2", "-waveform", "-tot-mode", "tot_comp",
                          "--dry-run"] + base)
        assert f"{warning} {differ.format('tot_mode, waveform')} and from the first 2 events only" in log, log
        assert not made.exists()                                                  # dry runs write nothing
        src = str(tmp / "root" / "syn.root")
        for argv in (["make-csv", "--muon", "4", "--test", "-tot-mode", "tot_comp", "--max-events", "2"],
                     ["make-csv", "-input", src, "-tot-mode", "tot_comp"],
                     ["make-csv", "--muon", "4", "-chunk-entries", "9"],
                     ["make-csv", "--muon", "4", "-tot-mode", "tot_branch"]):
            assert "formal csv" not in run_runner(argv + base + ["--dry-run"]), argv
        # the option in the yml: the whole chain uses it (no WARNING); the command-line default then differs
        text = cfg.read_text(encoding="utf-8")
        cfg.write_text(text.replace("make_csv:\n", "make_csv:\n  tot_mode: tot_comp\n"), encoding="utf-8")
        assert "formal csv" not in run_runner(["make-csv", "--muon", "4", "--dry-run"] + base)
        assert "formal csv" not in run_runner(["make-csv", "--muon", "4", "-tot-mode", "tot_comp", "--dry-run"] + base)
        log = run_runner(["make-csv", "--muon", "4", "-tot-mode", "tot_branch", "--dry-run"] + base)
        assert f"{warning} {differ.format('tot_mode')}" in log, log
        cfg.write_text(text, encoding="utf-8")
        # a real run: the run's csv is written with -waveform from the first 2 events, the WARNING logged once
        log = run_runner(["make-csv", "--muon", "4", "-waveform", "--max-events", "2"] + base)
        assert log.count(f"{warning} {differ.format('waveform')} and from the first 2 events only") == 1, log
        assert made.read_bytes().decode().split("\r\n")[0] == ",".join(m.beam_csv_columns(True))
        assert_frames_equal(read_csv_exact(made), reference_convert(first_events(long, 2), table),
                            m.beam_csv_columns())
        record = json.loads(m.provenance_path(made).read_text(encoding="utf-8"))
        assert record["waveform"] is True and record["csv_settings"]["waveform"] is True
        assert record["event_limit"] == 2 and record["events"] == 2
        # an existing csv is skipped without -overwrite (the mismatch note, no WARNING); -overwrite replaces it
        stamp = made.stat().st_mtime_ns
        log = run_runner(["make-csv", "--muon", "4"] + base)
        assert "not overwritten" in log and "does not match" in log and "formal csv" not in log, log
        assert made.stat().st_mtime_ns == stamp
        log = run_runner(["make-csv", "--muon", "4", "-overwrite"] + base)
        assert "formal csv" not in log and made.stat().st_mtime_ns != stamp
        record = json.loads(m.provenance_path(made).read_text(encoding="utf-8"))
        assert record["waveform"] is False and record["event_limit"] is None and record["events"] == 4


# sample counts checked before any conversion
def test_sample_counts_checked_before_any_conversion():
    # --sample-window / --waveform / --mc-samples are checked on the first digis of every file before the
    # first job is converted (a refused file leaves no product); options without effect are named in the log.
    # The MC file is converted with -mc -input (simulation runs cannot be chosen on the command line).
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        mc_root = write_ntuple(tmp / "root" / "mc.root", synthetic_long(4, n_samples=10))
        mc = ["-mc", "-input", str(mc_root)]
        base = ["--config", str(cfg), "--workers", "1", "--test"]
        _refused(["make-csv", "--muon", "4", "-sample-window", "8:10"] + base, "selects no sample", "syn.root")
        _refused(["make-csv", "--muon", "4", "-waveform", "-sample-window", "1:8"] + base, "--waveform needs",
                 "7 of 8")
        _refused(["make-csv", *mc, "-mc-samples", "8"] + base, "--mc-samples 8", "mc.root has 10")
        _refused(["make-csv", "--muon", "4", "-mc-samples", "x"] + base, "positive integer")
        assert not [p for p in tmp.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")
                    and "analysis_files" in p.parts]
        log = run_runner(["make-csv", "--muon", "4", "-mc-samples", "8", "-tot-mode", "tot_ne_adc",
                          "--max-events", "2"] + base)
        assert log.count("ignored") == 1 and "tot_ne_adc is meant for flat MC ntuples" in log, log
        log = run_runner(["make-csv", *mc, "-mc-samples", "10", "--max-events", "2"] + base)
        assert "ignored" not in log
        record = json.loads(m.provenance_path(tmp / "data" / "test" / "analysis_files" / "mc.csv").read_text())
        assert record["n_samples"] == 10 and record["mc"] is True


# converter fingerprint
def test_converter_fingerprint():
    # A test csv made by other converter code is not reused (no manual version bump needed); docstrings,
    # comments and functions outside CSV_CODE (QA plots, options) do not count.
    src = Path(m.__file__).read_text(encoding="utf-8")
    fp = m.converter_fingerprint(src)
    assert fp == m.converter_fingerprint() and set(fp) == {"code", "numpy", "pandas", "uproot"}

    # code part of the converter fingerprint of a source text
    def code(text):
        return m.converter_fingerprint(text)["code"]

    same = [src.replace('"""Fixed operation order: another order', '"""Changed docstring. Another order'),
            src.replace("TOA_FULL_SCALE = 1023     #", "TOA_FULL_SCALE = 1023  # another comment;"),
            src.replace("    pages = 0\n", "    pages = 0  # QA only\n    pages += 0\n")]
    changed = [src.replace("TOA_FULL_SCALE = 1023", "TOA_FULL_SCALE = 1024"),
               src.replace("lineterminator=term)", 'lineterminator=term).replace("e+", "E+")'),
               src.replace("TOT_BAD_CODE = 2048", "TOT_BAD_CODE = 2049")]
    for s in same:
        assert s != src and code(s) == fp["code"]
    for s in changed:
        assert s != src and code(s) != fp["code"]
    try:
        m.converter_fingerprint(src.replace("def _csv_text(", "def _csv_text_renamed("))
    except RuntimeError as exc:
        assert "_csv_text" in str(exc)
    else:
        raise AssertionError("a renamed csv function must be noticed")


# A repeated (event, layer, strip, end) digi: error (default) or drop; never merged.
def test_duplicate_digi_policies():
    S = 10
    rows = []
    for end, reps in ((0, 2), (1, 1)):
        for r in range(reps):
            for k in range(S):
                rows.append({"pf_event": 1, "layer": 4, "strip": 2, "end": end, "i_sample": k,
                             "adc": 10 + k + 100 * r, "tot": 0, "toa": 0, "pf_spill": 0, "pf_ticks": 0})
    c = {k: np.array([r[k] for r in rows], np.int32) for k in rows[0]}
    base = {**m.DEFAULTS, "window": (0, None)}
    for policy in ("error", "drop"):
        store, stats = m.BarStore(), m._new_stats()
        try:
            m._process_chunk(dict(c), S, {**base, "duplicate_digis": policy}, False, store, stats)
        except SystemExit as exc:
            assert policy == "error" and "never merged" in str(exc) and "--duplicate-digis drop" in str(exc), exc
            continue
        assert policy == "drop"
        assert stats["duplicate_digis"] == 1 and stats["unpaired_digis"] == 1 and not store.rows.sum()
    assert m.SETTINGS_SPEC["duplicate_digis"] == ("choice", ["error", "drop"])


# settings validated before any io
def test_settings_validated_before_any_io():
    # Every make_csv setting (yml or flag) is checked before a file is opened; the refusal names the setting.
    # Valid spellings (7.0 for 7, open sample windows) are accepted; unknown units and choices are argparse errors.
    with synthetic_setup(n_events=3) as (tmp, cfg, long, table):
        base = ["--config", str(cfg), "--workers", "1", "--test"]
        text = cfg.read_text(encoding="utf-8")

        # replace the make_csv line chunk_entries: 50 of the synthetic config
        def yml(extra):  # replaces the synthetic config's own make_csv line (chunk_entries: 50)
            assert "  chunk_entries: 50\n" in text
            cfg.write_text(text.replace("  chunk_entries: 50\n", f"{extra}\n"), encoding="utf-8")

        # the runner refuses argv before any ROOT file or TOA table is read
        def guarded(argv, *words):
            with patch.object(m, "_open_input", side_effect=AssertionError("must not open a ROOT file")), \
                    patch.object(m, "load_toa_calibration", side_effect=AssertionError("must not read the TOA table")):
                msg = _refused(argv, *words)
            assert "invalid setting" in msg, msg

        for extra, word in (("  chunk_entries: 0", "chunk_entries"), ("  chunk_entries: -5", "chunk_entries"),
                            ("  chunk_entries: 2.5", "chunk_entries"), ("  events_per_chunk: 0", "events_per_chunk"),
                            ("  max_events: 0", "max_events"), ("  qa_bins: 0", "qa_bins"),
                            ("  qa_cmin: .nan", "qa_cmin"), ("  waveform: 'yes'", "waveform"),
                            ("  tot_mode: tot", "tot_mode"), ("  duplicate_digis: keep", "duplicate_digis"),
                            ("  duplicate_digis: auto", "duplicate_digis"),
                            ("  missing_toa_calibration: skip", "missing_toa_calibration"),
                            ("  line_terminator: cr", "line_terminator"), ("  sample_window: '4'", "sample_window"),
                            ("  sample_window: '5:2'", "sample_window"), ("  sample_window: 'a:b'", "sample_window"),
                            ("  mc_samples_per_digi: 0", "mc_samples_per_digi"),
                            ("  mc_samples_per_digi: 2.5", "mc_samples_per_digi"),
                            ("  mc_samples_per_digi: true", "mc_samples_per_digi"),
                            ("  parallel_write_min_rows: -1", "parallel_write_min_rows"),
                            ("  toa_calibration: ''", "toa_calibration")):
            yml(extra)
            guarded(["make-csv", "--muon", "4"] + base, word, "make_csv")
            guarded(["make-csv", "--muon", "4", "--dry-run"] + base, word)  # the plan is refused too
        cfg.write_text(text, encoding="utf-8")
        for flags, word in ((["-chunk-entries", "0"], "chunk_entries"), (["-max-events", "0"], "max_events"),
                            (["-max-events", "-3"], "max_events"), (["-sample-window", "3:3"], "sample_window"),
                            (["-sample-window", "x"], "sample_window"), (["-mc-samples", "0"], "positive integer"),
                            (["-mc-samples", "x"], "positive integer")):
            guarded(["make-csv", "--muon", "4"] + flags + base, word)
        for flags in (["-only", "pdf"], ["-duplicate-digis", "merge"]):
            _argparse_refused(["make-csv", "--muon", "4"] + flags + base, flags[0].lstrip("-"))
        # a key make-csv does not know (e.g. a typo) is refused, never ignored
        for extra, words in (("  max_event: 3", ("'max_event'", "did you mean 'max_events'")),
                             ("  wavefrom: true", ("'wavefrom'", "did you mean 'waveform'")),
                             ("  overwrites: true", ("'overwrites'", "did you mean 'overwrite'"))):
            yml("  chunk_entries: 50\n" + extra)
            for argv in (["make-csv", "--muon", "4"] + base, ["make-csv", "--muon", "4", "--dry-run"] + base):
                with patch.object(m, "_open_input", side_effect=AssertionError("must not open a ROOT file")):
                    msg = _refused(argv, "unknown setting", "make_csv", *words)
        assert "valid keys" in msg
        # accepted: 7.0 is the integer 7 (also for mc_samples_per_digi); 'A:' = from A to the last sample,
        # ':B' = the first B samples
        assert m._parse_window("1:") == (1, None) and m._parse_window(":7") == (0, 7)
        assert m._parse_window("all") == m._parse_window(None) == m._parse_window("") == (0, None)
        yml("  chunk_entries: 7.0\n  mc_samples_per_digi: 10.0")
        ctx = RunContext(cfg=load_config(cfg), test=True)
        s = m.configured_settings(ctx)
        assert s["chunk_entries"] == 7 and isinstance(s["chunk_entries"], int)
        assert s["mc_samples_per_digi"] == 10 and isinstance(s["mc_samples_per_digi"], int)
        for spelling, want in (("auto", "auto"), (" AUTO ", "auto"), ("10", 10), (10, 10), (np.int64(8), 8)):
            assert m._mc_samples(spelling) == want, spelling
        for bad in ("10.0", "-1", 0, -3, 2.5, float("nan"), True, None, [10]):
            assert m._mc_samples(bad) is None and m._mc_samples_rule(bad), bad
        for window in ("1:", ":7"):
            run_runner(["make-csv", "--muon", "4", "-sample-window", window, "--dry-run"] + base)


# toa table keys follow the channel key rule
def test_toa_table_keys_follow_the_channel_key_rule():
    # The TOA table keys are finite integers inside the prototype, each bar once; a corrupt table stops the run
    # before any ROOT file is opened.  Valid tables give the same lookup (first matching row, columns by name or
    # positions 4 / 5) whatever their row order.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        good = write_toa_table(tmp / "good.csv")
        cal = m.load_toa_calibration(tmp / "good.csv")
        for layer in range(1, 20):
            for strip in range(12):
                rows = good[(good["layer"] == layer - 1) & (good["bar"] == strip)].values.tolist()
                bx, off, has = cal.constants(layer, strip)
                assert has == bool(rows), (layer, strip)
                if rows:
                    assert (bx, off) == (float(rows[0][4]), float(rows[0][5]))
        good.sample(frac=1, random_state=3).to_csv(tmp / "shuffled.csv", index=False)
        good.rename(columns={"bx_shift": "c4", "mean_shift": "c5"}).to_csv(tmp / "unnamed.csv", index=False)
        # named value columns need no position: four columns, in any order
        good[["layer", "bar", "bx_shift", "mean_shift"]].to_csv(tmp / "four.csv", index=False)
        good[["mean_shift", "bar", "bx_shift", "layer"]].to_csv(tmp / "reordered.csv", index=False)
        for other in ("shuffled.csv", "unnamed.csv", "four.csv", "reordered.csv"):
            o = m.load_toa_calibration(tmp / other)
            for a, b in ((cal.toa_bx_wrap_code, o.toa_bx_wrap_code), (cal.toa_end1_offset_ns, o.toa_end1_offset_ns),
                         (cal.has_row, o.has_row)):
                assert np.array_equal(a, b), other
        real_path = REPO_ROOT / "calibration" / "reference" / "toa_calibration_phase3.csv"
        if real_path.exists():  # an input: not in a fresh clone
            real = m.load_toa_calibration(real_path)
            bars = m.geo.all_bars()
            assert int(real.has_row.sum()) == 192 and real.has_row[bars["layer"].to_numpy(), bars["strip"].to_numpy()].all()

        # the good TOA table with values changed in its first row
        def table_with(**change) -> pd.DataFrame:
            t = good.copy()
            for col, value in change.items():
                t[col] = t[col].astype(object)
                t.loc[0, col] = value
            return t

        cases = [
            (pd.concat([good, good.iloc[[0]].assign(mean_shift=99.0)], ignore_index=True), io_utils.ChannelKeyError,
             "share"),
            (table_with(layer=0.5), io_utils.ChannelKeyError, "non-integer"),
            (table_with(layer=float("nan")), io_utils.ChannelKeyError, "missing or non-numeric"),
            (table_with(layer="x"), io_utils.ChannelKeyError, "missing or non-numeric"),
            (table_with(layer=19), io_utils.ChannelKeyError, "outside the detector"),   # 0-based 19 = layer 20
            (table_with(layer=0, bar=8), io_utils.ChannelKeyError, "outside the detector"),  # layer 1 has 8 bars
            (table_with(bar=-1), io_utils.ChannelKeyError, "outside the detector"),
            (table_with(bx_shift=float("nan")), io_utils.SettingsError, "bx_shift"),
            (good.drop(columns=["bar"]), io_utils.SettingsError, "layer, bar"),
            (good.assign(layer=good["layer"] > 0), io_utils.ChannelKeyError, "true/false"),  # never read as 0 / 1
            (good.assign(bar=good["bar"] > 0), io_utils.ChannelKeyError, "true/false"),
            # mean_shift neither named nor at position 5
            (good[["layer", "bar", "Det_id_0", "Det_id_1", "bx_shift"]], io_utils.SettingsError, "positions 4 / 5"),
            # position 5 is the named bx_shift itself: never used twice
            (good.rename(columns={"mean_shift": "m"})[["layer", "bar", "Det_id_0", "Det_id_1", "m", "bx_shift"]],
             io_utils.SettingsError, "positions 4 / 5"),
        ]
        for i, (table, error, word) in enumerate(cases):
            path = tmp / f"bad{i}.csv"
            table.to_csv(path, index=False)
            try:
                m.load_toa_calibration(path)
            except error as exc:
                assert word in str(exc) and str(path) in str(exc), (i, str(exc))
            else:
                raise AssertionError(f"case {i} ({word}) must be refused")
    # in a run: refused before any ROOT file is opened
    with synthetic_setup(n_events=3) as (tmp, cfg, long, table):
        pd.concat([table, table.iloc[[0]].assign(bx_shift=1)], ignore_index=True).to_csv(tmp / "toa.csv", index=False)
        with patch.object(m, "_open_input", side_effect=AssertionError("must not open a ROOT file")):
            _refused(["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1"],
                     "TOA table", "share")
        assert not (tmp / "data").exists()


# csv and provenance commit sidecar last
def test_csv_and_provenance_commit_sidecar_last():
    # The csv and its provenance json are one product set.  A failure while writing the csv or renaming it leaves
    # the previous pair untouched (and reusable); a failure of the json write after the rename leaves no json, so
    # the new csv is never reused and the next run says why.  No temp file is left behind in any case.
    with synthetic_setup(n_events=5) as (tmp, cfg, long, table):
        argv = ["make-csv", "--muon", "4", "--test", "--config", str(cfg), "--workers", "1"]
        variant = ["--overwrite", "-tot-mode", "tot_comp"]  # another csv content (test mode allows the flag)
        run_runner(argv)
        out = tmp_test_dir(tmp) / "syn.csv"
        prov = m.provenance_path(out)
        ctx = RunContext(cfg=load_config(cfg), test=True)
        ds = resolve_datasets(ctx.cfg, ["syn"])[0]

        # the reusable -test csv, or None
        def current():
            with QUIET(io.StringIO()):
                return m.current_test_csv(ctx, ds, run_csv(ctx, ds))

        old_csv, old_prov = out.read_bytes(), prov.read_bytes()
        assert current() == out

        # csv, provenance and reuse unchanged, no temp file left
        def unchanged():
            assert out.read_bytes() == old_csv and prov.read_bytes() == old_prov and current() == out
            assert temp_leftovers(tmp) == [], temp_leftovers(tmp)

        # (1) the csv write fails half way
        real_text, calls = m._csv_text, {"n": 0}

        # _csv_text that fails on its third call
        def failing_text(*a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("disk full (injected)")
            return real_text(*a, **k)

        with patch.object(m, "_csv_text", failing_text):
            try:
                run_runner(argv + variant)
            except OSError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("the injected write failure must surface")
        unchanged()
        # (2) the csv cannot be renamed into place (e.g. opened in Excel): the old json is restored
        real_replace = os.replace

        # os.replace that fails on the csv
        def failing_replace(src, dst):
            if io_utils.canonical_path(dst) == io_utils.canonical_path(out):
                raise PermissionError("csv opened elsewhere (injected)")
            return real_replace(src, dst)

        with patch.object(io_utils.os, "replace", failing_replace):
            try:
                run_runner(argv + variant)
            except PermissionError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("the injected rename failure must surface")
        unchanged()
        # (3) the json cannot be written after the csv was renamed: no json describes the new csv
        with patch.object(io_utils, "write_json", side_effect=OSError("json write failed (injected)")):
            try:
                run_runner(argv + variant)
            except OSError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("the injected json failure must surface")
        assert out.read_bytes() != old_csv and not prov.exists() and current() is None
        assert temp_leftovers(tmp) == []
        log = run_runner(argv)
        assert "not overwritten" in log and "provenance json missing" in log, log
        run_runner(argv + ["--overwrite"])
        assert out.read_bytes() == old_csv and current() == out and temp_leftovers(tmp) == []


# io plan inputs are external
def test_io_plan_inputs_are_external():
    # The ROOT files and the TOA table are external inputs (kind "external"): absent, they are MISSING for
    # 'runner check' (exit 2), never PENDING products of the chain; -only qa reads the csv as a product.
    import runner

    with synthetic_setup(n_events=2) as (tmp, cfg, long, table):
        parser = runner.build_parser()
        args = parser.parse_args(["make-csv", "--muon", "4", "--test", "--config", str(cfg)])
        ctx = runner.make_context(args)
        ctx.units = outputs_mod.selected_units(args)
        plan = m.io_plan(ctx, args, runner.datasets_for(ctx, m, args))
        ins = [p for p in plan if p["role"] == "in"]
        assert all(p["kind"] == "external" for p in ins) and {Path(p["path"]).name for p in ins} == {"toa.csv",
                                                                                                      "syn.root"}
        outs = [Path(p["path"]) for p in plan if p["role"] == "out"]
        assert outs == [tmp_test_dir(tmp) / "syn.csv", tmp_test_dir(tmp) / "syn.provenance.json"]
        with QUIET(io.StringIO()):
            assert runner.print_plan("make-csv", plan, ctx) == (0, 0)
        (tmp / "root" / "syn.root").rename(tmp / "root" / "gone.root")
        with QUIET(io.StringIO()) as out:
            assert runner.print_plan("make-csv", plan, ctx) == (1, 0)
        assert "MISSING" in out.getvalue()
        # -only qa: the existing csv is the input, the QA pdf the only output; -no-json: no provenance planned
        args = parser.parse_args(["make-csv", "--muon", "4", "--test", "-only", "qa", "--config", str(cfg)])
        ctx = runner.make_context(args)
        ctx.units = outputs_mod.selected_units(args)
        plan = m.io_plan(ctx, args, runner.datasets_for(ctx, m, args))
        assert [(p["role"], Path(p["path"]).name, p.get("kind")) for p in plan] == [
            ("in", "syn.csv", "product"), ("out", "test.pdf", None)], plan
        args = parser.parse_args(["make-csv", "--muon", "4", "--test", "-no-json", "--config", str(cfg)])
        ctx = runner.make_context(args)
        ctx.units = outputs_mod.selected_units(args)
        plan = m.io_plan(ctx, args, runner.datasets_for(ctx, m, args))
        assert [Path(p["path"]).name for p in plan if p["role"] == "out"] == ["syn.csv"]


# subset of a converted csv carries the sample counts
def test_subset_of_a_converted_csv_carries_the_sample_counts():
    # The provenance 'output' block describes the renamed csv, so io_utils.build_test_subset hands n_samples /
    # n_samples_used / waveform of a make-csv csv on to its test subset.
    with synthetic_setup(n_events=6) as (tmp, cfg, long, table):
        run_runner(["make-csv", "--muon", "4", "-sample-window", "1:7", "--config", str(cfg), "--workers", "1"])
        out = full_dir(tmp) / "syn.csv"
        sub = tmp / "cache" / "sub.csv"
        with QUIET(io.StringIO()):
            io_utils.build_test_subset(out, sub, 3)
        record = io_utils.read_json(sub.with_name("sub.provenance.json"))
        assert record["source_provenance"]["matches_source"] is True
        assert (record["n_samples"], record["n_samples_used"], record["waveform"]) == (8, 6, False)


# qa data from the store and from the existing csv
def test_qa_data_from_the_store_and_from_the_existing_csv():
    # -only csv,qa: after a conversion the QA pages are drawn from the in-memory store; when the csv is skipped
    # (exists, no -overwrite) or with -only qa from the csv.  All give the same bars and bit-identical values (the
    # csv floats are parsed round-trip; the renderer is replaced by a recorder).
    with synthetic_setup(n_events=6) as (tmp, cfg, long, table):
        seen = []

        # make_qa_pdf stand-in: records the bars it receives
        def recorder(bar_iter, pdf_path, title, info, settings):
            seen.append({(layer, strip): d for layer, strip, d in bar_iter})
            return 0

        argv = ["make-csv", "--muon", "4", "--test", "-only", "csv,qa", "--config", str(cfg), "--workers", "1"]
        with patch.object(m, "make_qa_pdf", recorder):
            log1 = run_runner(argv)
            log2 = run_runner(argv)
            log3 = run_runner(["make-csv", "--muon", "4", "--test", "-only", "qa", "--config", str(cfg)])
        assert "QA plots: 0 pages" in log1 and "QA plots from the existing csv" in log2
        assert "QA plots from the existing csv" in log3 and "not overwritten" not in log3
        store_data, csv_data, qa_only = seen
        written = read_csv_exact(tmp_test_dir(tmp) / "syn.csv")
        bars = set(zip(written["layer"].tolist(), written["strip"].tolist()))
        assert set(store_data) == set(csv_data) == set(qa_only) == bars and NO_TOA_ROW_WITH_HITS not in bars
        for key in store_data:
            for c in m.QA_COLUMNS:
                assert np.array_equal(store_data[key][c], csv_data[key][c]), (key, c)
        assert any(np.any(d["toa_end1"] % 1 != 0) for d in csv_data.values())  # real (non-integer) TOA values
        # -only qa without the csv: refused, with the command that makes it
        _refused(["make-csv", "--muon", "4", "-only", "qa", "--config", str(cfg)], "-only qa draws the QA pdf",
                 "make-csv --muon 4")
        # the keys of the csv follow the shared channel-key rule
        bad = tmp / "free" / "bad.csv"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text(written.assign(layer=written["layer"].where(written.index != 3, 20)).to_csv(index=False))
        try:
            list(m._qa_bars_from_csv(bad, 4))
        except io_utils.ChannelKeyError as exc:
            assert "outside the detector" in str(exc)
        else:
            raise AssertionError("a layer 20 must be refused")


# --------------------------------------------------------------------------
# final results (common/outputs.py): the QA pdfs; the csv is the csv unit
# --------------------------------------------------------------------------
# config with a second run syn2 (muon 1, ROOT file syn2.root)
def _two_dataset_config(tmp: Path, cfg: Path) -> Path:
    # A second run syn2 (muon 1, label 'synthetic 2', ROOT file syn2.root): 'make-csv --muon 4 1 -only csv,qa'
    # draws two QA pdfs, syn's first.
    _copy(tmp / "root" / "syn.root", tmp / "root" / "syn2.root")
    text = cfg.read_text(encoding="utf-8")
    cfg.write_text(text.replace("datasets:\n", "datasets:\n  syn2:\n    label: \"synthetic 2\"\n    particle: muon\n"
                                               "    energy_gev: 1\n    kind: data\n"
                                               "    root_files: [\"${root_dir}/syn2.root\"]\n"), encoding="utf-8")
    return cfg


# make_qa_pdf replaced by a writer of a one-line file (the job label): where each QA pdf went, in order.
@contextlib.contextmanager
def _qa_recorder():
    seen = []

    # make_qa_pdf stand-in: writes the title, records the path
    def fake(bar_iter, pdf_path, title, info, settings):
        n = sum(1 for _ in bar_iter)
        path = io_utils.ensure_writable(Path(pdf_path))
        path.write_text(f"{title}\n", encoding="utf-8")
        seen.append(path)
        return n

    with patch.object(m, "make_qa_pdf", fake):
        yield seen


# qa pdfs not saved without output in a fullset run
def test_qa_pdfs_not_saved_without_output_in_a_fullset_run():
    # A full-set run without --output draws the QA pdfs into cache/make_csv_qa/unsaved/ (emptied at the start of
    # each run), writes nothing to results/ and says so in ONE line; the csv + provenance are written as always and
    # the next run reads that csv.  io_plan lists the scratch destination with its note, never empties the scratch
    # and leaves no plan in outputs.plans().
    import runner

    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        _two_dataset_config(tmp, cfg)
        base = ["--config", str(cfg), "--workers", "1"]
        scratch = tmp / "cache" / "make_csv_qa" / "unsaved"
        scratch.mkdir(parents=True)
        (scratch / "old_qa.pdf").write_text("previous run", encoding="utf-8")
        parser = runner.build_parser()
        outputs_mod.reset()
        for extra in ([], ["--output", "syn_qa=n.pdf"]):
            args = parser.parse_args(["make-csv", "--muon", "4", "-only", "csv,qa"] + extra + base)
            try:
                ctx = runner.make_context(args)
                ctx.units = outputs_mod.selected_units(args)
                with QUIET(io.StringIO()):
                    plan = m.io_plan(ctx, args, runner.datasets_for(ctx, m, args))
            finally:
                io_utils.set_protected_dirs([])
            qa = [p for p in plan if p.get("result") == "syn_qa"]
            if extra:
                assert qa == [{"role": "out", "path": tmp / "res" / "fullset" / "make_csv_qa" / "n.pdf",
                               "result": "syn_qa"}], qa
            else:
                assert qa == [{"role": "out", "path": scratch / "syn_qa.pdf", "result": "syn_qa",
                               "note": "not saved: no --output"}], qa
            assert (scratch / "old_qa.pdf").exists() and outputs_mod.plans() == []
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "1", "-only", "csv,qa"] + base)
        assert seen == [scratch / "syn2_qa.pdf", scratch / "syn_qa.pdf"] or \
               seen == [scratch / "syn_qa.pdf", scratch / "syn2_qa.pdf"], seen
        assert sorted(p.name for p in scratch.iterdir()) == ["syn2_qa.pdf", "syn_qa.pdf"]  # old_qa.pdf removed
        assert not (tmp / "res").exists()
        assert log.count("final results not saved (no --output): scratch copies in") == 1 and str(scratch) in log
        assert log.count("(scratch copy, not saved)") == 2 and "use --output [names] to save them" in log
        for run, key in ((RUN, "syn"), ("Muon_1GeV", "syn2")):  # the csv unit
            assert (full_dir(tmp, run) / f"{key}.csv").is_file()
            assert (full_dir(tmp, run) / f"{key}.provenance.json").is_file()
        # the next run reads the existing csv (skipped, QA from the existing csv); still nothing saved
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "-only", "csv,qa"] + base)
        assert "QA plots from the existing csv" in log and seen == [scratch / "syn_qa.pdf"]
        assert sorted(p.name for p in scratch.iterdir()) == ["syn_qa.pdf"] and not (tmp / "res").exists()
        # without the qa unit there is no final result: no 'not saved' line
        assert "not saved" not in run_runner(["make-csv", "--muon", "4"] + base)


# qa pdfs saved with output names
def test_qa_pdfs_saved_with_output_names():
    # --output names the QA pdfs by result: KEY=FILE, or the default file name; naming one saves only that one (the
    # other goes to the unsaved scratch copy); a name of no result is ignored with a WARNING; --output alone saves every
    # QA pdf under its default name results/fullset/make_csv_qa/<run>/<csv stem>_qa.pdf; no record json next to it.
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        _two_dataset_config(tmp, cfg)
        base = ["--config", str(cfg), "--workers", "1"]
        qa_dir = tmp / "res" / "fullset" / "make_csv_qa"
        scratch = tmp / "cache" / "make_csv_qa" / "unsaved"
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "1", "-only", "csv,qa", "--output", "syn_qa=b.pdf",
                              "syn2_qa=a.pdf"] + base)
        assert sorted(seen) == sorted([qa_dir / "b.pdf", qa_dir / "a.pdf"]), seen
        assert (qa_dir / "b.pdf").read_text(encoding="utf-8") == "synthetic: syn.csv\n"
        assert (qa_dir / "a.pdf").read_text(encoding="utf-8") == "synthetic 2: syn2.csv\n"
        assert f"-> {qa_dir / 'b.pdf'}" in log and f"-> {qa_dir / 'a.pdf'}" in log and "not saved" not in log
        # the default file name of one result: only that one is saved
        with _qa_recorder() as seen:
            run_runner(["make-csv", "--muon", "4", "1", "-only", "csv,qa", "--output", "syn_qa.pdf"] + base)
        assert sorted(seen) == sorted([qa_dir / "syn_qa.pdf", scratch / "syn2_qa.pdf"]), seen
        # a name of no result: ignored (WARNING), the QA pdf goes to the scratch copy
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "-only", "csv,qa", "--output", "z.pdf"] + base)
        assert "name no result here, ignored: z.pdf" in log and seen == [scratch / "syn_qa.pdf"], log
        # --output alone: the default names in the run sub-folders
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "1", "-only", "csv,qa", "--output"] + base)
        assert sorted(seen) == sorted([qa_dir / RUN / "syn_qa.pdf", qa_dir / "Muon_1GeV" / "syn2_qa.pdf"]), seen
        assert not list(qa_dir.rglob("*.json")) and (full_dir(tmp) / "syn.provenance.json").is_file()


# qa pdfs of a test run without output are test names
def test_qa_pdfs_of_a_test_run_without_output_are_test_names():
    # -test without --output: the QA pdfs are results/test/test.pdf, test1.pdf, ...; the test csv is written as
    # always and later test-mode stages read it (current_test_csv); with a name the QA pdf goes to
    # results/test/make_csv_qa/, without a record json next to it.
    with synthetic_setup(n_events=4) as (tmp, cfg, long, table):
        _two_dataset_config(tmp, cfg)
        base = ["--config", str(cfg), "--workers", "1", "--test"]
        res = tmp / "res" / "test"
        with _qa_recorder() as seen:
            log = run_runner(["make-csv", "--muon", "4", "1", "-only", "csv,qa"] + base)
        assert seen == [res / "test.pdf", res / "test1.pdf"], seen
        assert {(res / n).read_text(encoding="utf-8") for n in ("test.pdf", "test1.pdf")} == {
            "synthetic: syn.csv\n", "synthetic 2: syn2.csv\n"}
        assert f"-> {res / 'test1.pdf'}" in log and "not saved" not in log and not (res / "make_csv_qa").exists()
        ctx = RunContext(cfg=load_config(cfg), test=True)
        for key, run in (("syn", RUN), ("syn2", "Muon_1GeV")):
            ds = resolve_datasets(ctx.cfg, [key])[0]
            assert m.current_test_csv(ctx, ds, run_csv(ctx, ds)) == tmp_test_dir(tmp, run) / f"{key}.csv"
        with _qa_recorder() as seen:
            run_runner(["make-csv", "--muon", "4", "-only", "csv,qa", "--output", "syn_qa=mine.pdf"] + base)
        assert seen == [res / "make_csv_qa" / "mine.pdf"] and not list(res.rglob("*.json"))
        io_utils.set_protected_dirs([])


# the whole chain on a new input: two ROOT files of a run -> two csv -> two clean products, nothing merged
def test_two_root_files_give_two_csv_and_two_clean_products():
    from ldmx_hcal.common import geometry as geo
    from ldmx_hcal.selection import clean_dataset as cd

    with synthetic_setup(n_events=6) as (tmp, cfg, long, table):
        write_ntuple(tmp / "root" / "syn_b.root", synthetic_long(5, seed=3))
        text = cfg.read_text(encoding="utf-8")
        cfg.write_text(text.replace('root_files: ["${root_dir}/syn.root"]',
                                    'root_files: ["${root_dir}/syn.root", "${root_dir}/syn_b.root"]'), encoding="utf-8")
        base = ["--config", str(cfg), "--workers", "1"]
        run_runner(["make-csv", "--muon", "4"] + base)
        ctx = RunContext(cfg=load_config(cfg))
        ds = resolve_datasets(ctx.cfg, ["muon 4"])[0]
        a, b = full_dir(tmp) / "syn.csv", full_dir(tmp) / "syn_b.csv"
        assert ctx.run_csvs(ds) == [a, b]
        calib = geo.all_channels()
        calib["pedestal"], calib["pedestal_sigma"], calib["mpv"] = 100.0, 2.0, 100.0
        calib["pedestal_range_with_5sigma"], calib["pedestal_range_with_20percent"] = 110.0, 120.0
        calib["mpv_status"], calib["mpv_fit_status"] = "OK", "CONVERGED"
        ctx.calibration_product().parent.mkdir(parents=True, exist_ok=True)
        calib.to_csv(ctx.calibration_product(), index=False)
        run_runner(["clean", "--muon", "4", "-otherparticle", "-only", "clean,cutflow", "-no-plots"] + base)
        folder = tmp / "data" / "clean_dataset" / RUN
        assert sorted(p.name for p in folder.glob("*.csv")) == ["syn_b_clean.csv", "syn_clean.csv"]
        for src in (a, b):
            assert cd.clean_csv(ctx, ds, src) == folder / f"{src.stem}_clean.csv"
            record = json.loads(cd.provenance_json(ctx, ds, src).read_text(encoding="utf-8"))
            assert record["csv"] == f"{RUN}/{src.name}" and list(record["members"]) == [f"{src.stem}_clean.csv"]
            lines = cd.clean_csv(ctx, ds, src).read_text(encoding="utf-8").splitlines()
            source = set(src.read_text(encoding="utf-8").splitlines())
            assert lines[0] == DEFAULT_HEADER and set(lines[1:]) <= source   # rows of its own csv only


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["make_analysis"]))
