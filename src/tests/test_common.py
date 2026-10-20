"""Tests of the shared infrastructure: config / layout, cache folder, dataset names, test subsets,
product-set commits (sidecar last), temp-file cleanup, channel-key and settings validation, write
protection, --clean-input, structured io plans and the check exit codes.  Synthetic files
only, a few seconds.

Covers the Codex review items INF-03 (test subset independent of chunk size and row order), INF-05
(check exit code), INF-01 (writes into protected folders are refused) and, from round 2, INF2-01
(subset + record committed as one product set; failure injection), INF2-02 (channel keys checked
before the compact dtype: 257 never wraps to 1), the check --test classification of auto-built
subsets, CI-R2-04 (temp registry), B6 (numeric settings) and CAL2-02 (subset keeps the converter's
n_samples / n_samples_used).  Round-3 verification: check with empty product trees (subsets AUTO, clean
datasets PENDING, same exit code), kinds inferred for plain plan items, stale subsets shown as AUTO with
the reason, every stage inside a temp registry, -from-saved / -clean-input gates (calib opt-in, Lei D8),
orphan temp sweep, node-index and boolean keys, open windows, redirect_products for fixtures.

Author: Lei
"""

from __future__ import annotations

import io
import json
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
from ldmx_hcal.common import io_utils  # noqa: E402
from ldmx_hcal.common.config import (DEFAULT_CONFIG, REPO_ROOT, Dataset, RunContext, load_config,  # noqa: E402
                                     resolve_datasets)

HEADER = ("pf_event,tot_end0,toa_end0,adc_sum_end0,adc_mean_end0,adc_max_end0,layer,strip,pf_spill,pf_ticks,"
          "tot_end1,toa_end1,adc_sum_end1,adc_mean_end1,adc_max_end1")


# one beam csv line; toa with many digits (the subset must keep the text exactly)
def _row(event: int, layer: int, strip: int) -> str:
    return (f"{event},0,{event * 0.1 + 0.123456789012345678},{800 + event},100.125,{110 + strip},{layer},{strip},"
            f"1,{event * 7},0,0.0,{801 + event},100.0,111")


# write a beam csv with the header and the given lines
def _write(path: Path, rows: list[str]) -> None:
    path.write_bytes(("\n".join([HEADER] + rows) + "\n").encode())


# call fn silently; returns (result, console text)
def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


# require fn to raise exc_type (whose text contains contains); returns the exception
def _raises(exc_type, fn, *args, contains: str | None = None, **kwargs):
    try:
        with redirect_stdout(io.StringIO()):
            fn(*args, **kwargs)
    except exc_type as exc:
        if contains is not None:
            assert contains in str(exc), (contains, str(exc))
        return exc
    raise AssertionError(f"{fn} did not raise {exc_type.__name__}")


# provenance record next to a test subset
def _record(dst: Path) -> dict:
    return json.loads(dst.with_name(f"{dst.stem}.provenance.json").read_text())


# RunContext with every tree under tmp (with_run: the run muon 4 in Muon_4GeV/)
def _ctx_on(tmp: Path, test: bool = True, with_run: bool = False, **kwargs) -> RunContext:
    cfg = {"paths": {"generated_data_dir": tmp / "data", "calibration_dir": tmp / "cal", "results_dir": tmp / "res",
                     "cache_dir": tmp / "cache", "analysis_files_dir": tmp / "data" / "analysis_files"},
           "run": {"chunksize": 3}, "safety": {"protected_dirs": []}}
    if with_run:  # the run muon 4: its folder data/analysis_files/Muon_4GeV/
        cfg["datasets"] = {"muon_4GeV": {"particle": "muon", "energy_gev": 4.0, "folder": "Muon_4GeV"}}
    return RunContext(cfg=cfg, test=test, **kwargs)


# the synthetic run muon 4
def _ds(key: str = "muon_4GeV") -> Dataset:
    return Dataset(key=key, particle="muon", energy_gev=4.0, label="synthetic", kind="data")


# A csv of the run muon 4 in its folder of _ctx_on (not written).
def _run_csv(tmp: Path, name: str = "beam.csv") -> Path:
    return tmp / "data" / "analysis_files" / "Muon_4GeV" / name


# ----------------------------------------------------------------------------------------------
# layout, config, cache folder
# ----------------------------------------------------------------------------------------------
# repository layout
def test_repository_layout():
    assert (REPO_ROOT / "configure" / "config.yml").exists()
    assert DEFAULT_CONFIG == REPO_ROOT / "configure" / "config.yml"
    for d in ("src", "data", "calibration", "results", "configure", "env"):
        assert (REPO_ROOT / d).is_dir(), d
    assert (REPO_ROOT / "calibration" / "reference").is_dir()  # the TOA table is an input: not in a fresh clone
    assert (REPO_ROOT / "configure" / "channel_map.csv").exists()
    for name in ("requirements.txt", "requirements-lock.txt", "Dockerfile", "docker.yml", "data_inputs_sha256.txt"):
        assert (REPO_ROOT / "env" / name).exists(), name
    assert "COPY env/requirements.txt" in (REPO_ROOT / "env" / "Dockerfile").read_text(encoding="utf-8")
    assert (REPO_ROOT / "data" / "README.md").exists()
    for d in ("root", "analysis_files", "charge_injection_scans"):  # input folders kept in git by a .gitkeep
        assert (REPO_ROOT / "data" / d / ".gitkeep").exists(), d
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/cache/*" in gitignore and "env/" not in gitignore  # cache content ignored, the folder kept
    assert "cache/" in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


# config paths and modes
def test_config_paths_and_modes():
    cfg = load_config()
    for key in ("data_dir", "root_dir", "analysis_files_dir", "ci_scan_dir", "channel_map", "toa_calibration",
                "generated_data_dir", "calibration_dir", "results_dir", "cache_dir"):
        assert key in cfg["paths"], key
    assert "make_csv_output_dir" not in cfg["paths"]   # make-csv writes the run folders of analysis_files_dir
    assert cfg["paths"]["cache_dir"] == REPO_ROOT / "cache"
    full = RunContext(cfg=cfg)
    test = RunContext(cfg=cfg, test=True)
    assert full.mode == "fullset" and test.mode == "test"
    # a normal run writes straight into data / calibration / cache (no fullset folder), a test run into their
    # test/ folder; the results tree keeps a folder per mode (results/fullset/<stage>/, results/test/<stage>/)
    assert full.results_dir("x") == cfg["paths"]["results_dir"] / "fullset" / "x"
    assert test.results_dir("x") == cfg["paths"]["results_dir"] / "test" / "x"
    assert test.results_dir("x", mode="fullset") == full.results_dir("x")
    assert full.data_dir("clean_dataset") == cfg["paths"]["generated_data_dir"] / "clean_dataset"
    assert full.calib_dir("pedestal_mpv") == cfg["paths"]["calibration_dir"] / "pedestal_mpv"
    assert test.calib_dir("pedestal_mpv") == cfg["paths"]["calibration_dir"] / "test" / "pedestal_mpv"
    assert test.data_dir("clean_dataset") == cfg["paths"]["generated_data_dir"] / "test" / "clean_dataset"
    assert full.cache_dir("ci") == REPO_ROOT / "cache" / "ci"
    assert test.cache_dir("ci", mode="fullset") == REPO_ROOT / "cache" / "ci"
    assert full.calibration_product().name == "calibration_parameters.csv"
    assert full.calibration_product().parent.name == "pedestal_mpv"
    assert full.calibration_product().parent == cfg["paths"]["calibration_dir"] / "pedestal_mpv"
    assert not hasattr(full, "output_root") and not hasattr(full, "out_dir")
    ds = resolve_datasets(cfg, ["muon 4"])[0]
    # a run is the folder of its csv files (one per ROOT file); products: one per csv, a sub-folder per run
    assert ds.folder == "Muon_4GeV"
    assert full.run_dir(ds) == cfg["paths"]["analysis_files_dir"] / "Muon_4GeV"
    assert test.run_dir(ds) == cfg["paths"]["generated_data_dir"] / "test" / "analysis_files" / "Muon_4GeV"
    csv = full.run_dir(ds) / "fixed_Muon_4GeV.csv"
    assert full.root_csv(ds, "x/fixed_Muon_4GeV.root") == csv and test.test_csv(ds, csv) == test.run_dir(ds) / csv.name
    assert full.test_subset_csv(ds, csv) == (REPO_ROOT / "cache" / "test" / "test_subsets" / "Muon_4GeV"
                                             / "fixed_Muon_4GeV_first5000events.csv")
    assert test.clean_product(ds, csv) == (cfg["paths"]["generated_data_dir"] / "test" / "clean_dataset"
                                           / "Muon_4GeV" / "fixed_Muon_4GeV_clean.csv")
    assert full.clean_product(ds, csv, mode="test") == test.clean_product(ds, csv)
    assert full.event_source(ds, csv) == "Muon_4GeV/fixed_Muon_4GeV.csv"


# Fixtures that copy load_config() must move ALL product trees (the cache is configured separately).
def test_redirect_products_moves_every_product_tree():
    from ldmx_hcal.common.config import PRODUCT_TREES, redirect_products

    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        moved = redirect_products(cfg, tmp)
        ctx = RunContext(cfg=moved, test=True)
        ds = resolve_datasets(moved, ["muon 4"])[0]
        for p in (ctx.data_dir("x"), ctx.calib_dir("x"), ctx.results_dir("x"), ctx.cache_dir("x"),
                  ctx.test_subset_csv(ds, "a.csv"), ctx.clean_product(ds, "a.csv"), ctx.calibration_product()):
            assert Path(tmp) in p.parents, p
        assert set(PRODUCT_TREES) == {"generated_data_dir", "calibration_dir", "results_dir", "cache_dir"}
        assert moved["paths"]["analysis_files_dir"] == cfg["paths"]["analysis_files_dir"]   # inputs stay
        assert cfg["paths"]["cache_dir"] == REPO_ROOT / "cache"                             # original untouched


# cache dir without yml entry follows the data tree
def test_cache_dir_without_yml_entry_follows_the_data_tree():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"paths": {"generated_data_dir": Path(tmp) / "data", "calibration_dir": Path(tmp) / "c",
                         "results_dir": Path(tmp) / "r"}}
        ctx = RunContext(cfg=cfg, test=True)
        assert ctx.cache_dir("x") == Path(tmp) / "cache" / "test" / "x"
        assert ctx.test_subset_csv(_ds(), "b.csv").parent == Path(tmp) / "cache" / "test" / "test_subsets" / "Muon_4GeV"


# the final layout (Lei 2026-10-06): products and inputs in one tree, one file each (relative to the temp folder)
LAYOUT_PRODUCTS = {"fullset": ["data/clean_dataset/c.csv", "calibration/pedestal_mpv/p.csv",
                               "results/fullset/pedestal_mpv/fits.pdf", "cache/pedestal_mpv/h.npz",
                               "cache/pedestal_mpv/unsaved/u.pdf"],
                   "test": ["data/test/clean_dataset/c.csv", "calibration/test/pedestal_mpv/p.csv",
                            "results/test/test.pdf", "results/test/pedestal_mpv/fits.pdf",
                            "cache/test/test_subsets/s.csv"]}
LAYOUT_OTHERS = ["data/root/run.root", "data/analysis_files/fixed_Muon_4GeV.csv", "data/scans/scan.csv",
                 "calibration/reference/toa.csv", "data/README.md", "results/flat_stage/old.pdf"]
LAYOUT_NEVER = ["data/test", "data/analysis_files", "data/root", "data/scans", "calibration/reference",
                "calibration/test", "results/test", "cache/test", "results/flat_stage"]   # for a normal run


# RunContext of the final layout under tmp, with every product and input file written
def layout_context(tmp: Path, test: bool = False) -> RunContext:
    # A RunContext of the final layout under ``tmp`` with every file of LAYOUT_PRODUCTS / LAYOUT_OTHERS written:
    # the inputs (data/root/, data/analysis_files/ = where make-csv writes, data/scans/ = an input folder of another
    # name, calibration/reference/) sit in the roots of the product trees.
    data, cal = tmp / "data", tmp / "calibration"
    cfg = {"paths": {"data_dir": data, "root_dir": data / "root", "analysis_files_dir": data / "analysis_files",
                     "ci_scan_dir": data / "scans",
                     "toa_calibration": cal / "reference" / "toa.csv", "generated_data_dir": data,
                     "calibration_dir": cal, "results_dir": tmp / "results", "cache_dir": tmp / "cache"},
           "safety": {"protected_dirs": []},
           "datasets": {"muon_4GeV": {"particle": "muon", "energy_gev": 4.0, "folder": "Muon_4GeV"}}}
    for rel in LAYOUT_PRODUCTS["fullset"] + LAYOUT_PRODUCTS["test"] + LAYOUT_OTHERS:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text("x\n", encoding="utf-8")
    return RunContext(cfg=cfg, test=test)


# mode tree dirs never enter the inputs or the other mode
def test_mode_tree_dirs_never_enter_the_inputs_or_the_other_mode():
    # Final layout (Lei 2026-10-06): a normal run's products sit directly in data/, calibration/ and cache/
    # (data/<stage>/, no fullset folder) and in results/fullset/<stage>/; the roots of those trees also hold the
    # inputs (data/root/, data/analysis_files/, calibration/reference/, any folder holding a configured input) and
    # the test trees.  mode_tree_dirs / in_products of a normal run never name them; a test run has <tree>/test/.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        full, test = layout_context(tmp), layout_context(tmp, test=True)
        rel = lambda paths: [p.relative_to(tmp).as_posix() for p in paths]  # noqa: E731
        assert rel(full.mode_tree_dirs()) == ["data/clean_dataset", "calibration/pedestal_mpv", "results/fullset",
                                              "cache/pedestal_mpv"]
        assert rel(test.mode_tree_dirs()) == ["data/test", "calibration/test", "results/test", "cache/test"]
        assert full.mode_tree_dirs(mode="test") == test.mode_tree_dirs()
        walked = full.mode_tree_dirs()
        for never in LAYOUT_NEVER:
            assert not any(d == tmp / never or d in (tmp / never).parents or tmp / never in d.parents
                           for d in walked), never
        for mode, ctx in (("fullset", full), ("test", test)):
            other = "test" if mode == "fullset" else "fullset"
            for f in LAYOUT_PRODUCTS[mode]:
                assert ctx.in_products(tmp / f) and ctx.in_products(tmp / f, mode="any"), (mode, f)
                assert not ctx.in_products(tmp / f, mode=other), (mode, f)
        for f in LAYOUT_OTHERS:   # inputs, a file at a tree root, a results folder without a mode: never products
            assert not full.in_products(tmp / f, mode="any"), f
        for d in LAYOUT_NEVER:
            assert not full.in_products(tmp / d), d


# written file scan of a normal run never enters the inputs or the test tree
def test_written_file_scan_of_a_normal_run_never_enters_the_inputs_or_the_test_tree():
    # The run log lists the files a run wrote ([6], run_log._scan_written) by walking RunContext.mode_tree_dirs
    # only: a normal run never walks data/test, data/analysis_files, data/root or calibration/reference (13 GB of
    # inputs on this machine), a test run only the test trees; make-csv also lists the files directly in
    # paths.analysis_files_dir (-input csv files) and in its run folders, never deeper.
    import argparse

    from ldmx_hcal.common import run_log

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        real_walk, walked = os.walk, []
        never = [tmp / d for d in LAYOUT_NEVER]

        # os.walk that records every folder it enters
        def recording_walk(top, *args, **kwargs):
            for item in real_walk(top, *args, **kwargs):
                walked.append(Path(item[0]))
                yield item

        # files the run log would list as written by stage
        def scan(stage: str, ctx: RunContext) -> list[str]:
            r = run_log._Run(stage, argparse.Namespace(run_names=[], datasets=None, from_saved=False), ctx, True)
            r.start_ns = 0          # every file counts as written in this run: only the folders decide
            walked.clear()
            with patch.object(run_log.os, "walk", recording_walk):
                return sorted(p.relative_to(tmp).as_posix() for p in run_log._scan_written(r))

        # walked folders inside a folder the scan must never enter
        def entered_never() -> list[Path]:
            return [d for d in walked if any(d == n or n in d.parents for n in never)]

        full = layout_context(tmp)
        (tmp / "data" / "analysis_files" / "Muon_4GeV" / "deeper").mkdir(parents=True)
        (tmp / "data" / "analysis_files" / "Muon_4GeV" / "a.csv").write_text("x\n", encoding="utf-8")
        (tmp / "data" / "analysis_files" / "Muon_4GeV" / "deeper" / "deep.csv").write_text("x\n", encoding="utf-8")
        assert scan("fake", full) == sorted(LAYOUT_PRODUCTS["fullset"])
        assert walked and not entered_never(), walked
        made = ["data/analysis_files/fixed_Muon_4GeV.csv", "data/analysis_files/Muon_4GeV/a.csv"]  # not deeper
        assert scan("make-csv", full) == sorted(LAYOUT_PRODUCTS["fullset"] + made)
        assert walked and not entered_never(), walked
        assert scan("fake", layout_context(tmp, test=True)) == sorted(LAYOUT_PRODUCTS["test"])
        assert walked and all(d.relative_to(tmp).parts[1] == "test" for d in walked), walked


ELECTRON_KEYS = {"electron_0.3GeV", "electron_0.4GeV", "electron_0.5GeV", "electron_1GeV", "electron_4GeV"}


# run references and keys
def test_run_references_and_keys():
    # A run is named by its reference ("muon 4": full particle name, energy in GeV, case-insensitive, 4GeV /
    # 300MeV also work) or by its derived key (muon_4GeV).  Any run of datasets.yml can be named; abbreviated
    # particle names are not run names and stop with the reason and every existing run in reference form.
    from ldmx_hcal.common.config import known_runs, parse_run_ref, resolve_runs, run_flags, run_key, run_ref

    cfg = load_config()
    names = lambda toks: [d.key for d in resolve_datasets(cfg, toks)]  # noqa: E731
    assert parse_run_ref("muon 4") == ("muon", 4.0) and parse_run_ref("Electron 300MeV") == ("electron", 0.3)
    assert parse_run_ref("pion_0.5GeV") == ("pion", 0.5) and parse_run_ref("MUON 4 GeV") == ("muon", 4.0)
    assert run_key("muon", 4) == "muon_4GeV" and run_key("electron", 0.3, run=304) == "electron_0.3GeV_run304"
    assert run_key("muon", 4.0, kind="mc") == "mc_muon_4GeV"
    assert names(["electron 0.3"]) == ["electron_0.3GeV"]
    assert names(["electron_0.3GeV", "Muon 4", "pion 3GeV"]) == ["electron_0.3GeV", "muon_4GeV", "pion_3GeV"]
    assert names(["electron 0.5", "electron 500MeV"]) == ["electron_0.5GeV"]      # one run, named twice
    assert set(names(["electron"])) == ELECTRON_KEYS
    assert names(["muon 4", "electron"])[0] == "muon_4GeV"
    assert set(names(["all"])) == set(cfg["datasets"]) and resolve_datasets(cfg, [], default=None) == []
    assert names([]) == [] and names(None) == []
    muon = resolve_runs(cfg, "muon 4")[0]
    assert (muon.key, run_ref(muon), run_flags(muon)) == ("muon_4GeV", "muon 4", "--muon 4")
    assert resolve_runs(cfg, [muon]) == [muon]                                   # Dataset objects pass through
    runs = known_runs(cfg)
    assert runs.startswith("electron 0.3, electron 0.4") and "muon 4" in runs and "pion 0.5" in runs
    # old abbreviated names: refused with the reason, the setting named and the runs listed
    for old in ("mu4", "e0.3", "pi3"):
        exc = _raises(SystemExit, resolve_runs, cfg, old, "-fit-dataset")
        text = str(exc)
        assert text.startswith(f"-fit-dataset: '{old}': unknown particle") and "full particle name" in text, text
        assert "runs of configure/datasets.yml: electron 0.3" in text, text
    for text in ("0.3GeVe", "nonsense"):
        _raises(SystemExit, resolve_runs, cfg, text, "x", contains="is not a run reference")
    _raises(ValueError, parse_run_ref, "mu4", contains="write the full particle name")
    _raises(SystemExit, resolve_runs, cfg, "muon 5", "calibration.source_dataset",
            contains="calibration.source_dataset: there is no run 'muon 5'; runs of configure/datasets.yml")


# run keys are derived from datasets yml
def test_run_keys_are_derived_from_datasets_yml():
    # datasets.yml has no run names and no csv paths: the key is <particle>_<E>GeV, + _run<N> for several runs at
    # one energy, mc_ in front for simulation; the run folder is <Particle>_<E>GeV[_run<N>] (MC_ for simulation).
    # Any other field (csv, key, aliases) is an unknown field; a 'datasets:' override section is keyed by reference
    # or key.
    from ldmx_hcal.common.config import (_dataset_overrides, datasets_from_tree, resolve_runs, run_ref,
                                         run_setting)

    tree = {"electron": {"0.3GeV": [{"runs": [304]}, {"root": "b.root", "runs": 305}], "4": None},
            "muon": {"4GeV": {"root": "m.root", "runs": [287]}},
            "mc": {"muon": {"4GeV": {"root": "sim.root"}}}}
    base = datasets_from_tree(tree)
    assert set(base) == {"electron_0.3GeV_run304", "electron_0.3GeV_run305", "electron_4GeV", "muon_4GeV",
                         "mc_muon_4GeV"}
    assert {k: d["folder"] for k, d in base.items()} == {
        "electron_0.3GeV_run304": "Electron_0.3GeV_run304", "electron_0.3GeV_run305": "Electron_0.3GeV_run305",
        "electron_4GeV": "Electron_4GeV", "muon_4GeV": "Muon_4GeV", "mc_muon_4GeV": "MC_Muon_4GeV"}
    cfg = {"datasets": base}
    keys = lambda refs: [d.key for d in resolve_runs(cfg, refs)]  # noqa: E731
    assert keys("electron 0.3") == ["electron_0.3GeV_run304", "electron_0.3GeV_run305"]   # every run there
    assert keys("electron 0.3 run 305") == ["electron_0.3GeV_run305"]
    assert keys("mc muon 4") == ["mc_muon_4GeV"] and keys("muon 4") == ["muon_4GeV"]
    assert "mc_muon_4GeV" not in keys("all") and keys("electron_0.3GeV_RUN304") == ["electron_0.3GeV_run304"]
    by_key = {d.key: d for d in resolve_runs(cfg, "all") + resolve_runs(cfg, "mc muon 4")}
    assert run_ref(by_key["electron_0.3GeV_run304"]) == "electron 0.3 run 304"
    assert run_ref(by_key["mc_muon_4GeV"]) == "mc muon 4" and run_ref(by_key["electron_4GeV"]) == "electron 4"
    for field in ("csv", "colour"):
        _raises(SystemExit, datasets_from_tree, {"muon": {"4GeV": {"root": "m.root", field: "x"}}},
                contains=f"unknown field(s) ['{field}']")
    # 'datasets:' overrides of config.yml / local.yml: by reference or key; unknown runs and ambiguity stop
    assert set(_dataset_overrides({"Muon 4": {"label": "x"}}, base, "d.yml")) == {"muon_4GeV"}
    assert set(_dataset_overrides({"electron_4GeV": {"root": "r.root"}}, base, "d.yml")) == {"electron_4GeV"}
    _raises(SystemExit, _dataset_overrides, {"electron 0.3": {"label": "x"}}, base, "d.yml",
            contains="names several runs (electron 0.3 run 304, electron 0.3 run 305)")
    _raises(SystemExit, _dataset_overrides, {"muon 5": {"label": "x"}}, base, "d.yml", contains="is not a run of d.yml")
    new = _dataset_overrides({"extra": {"particle": "pion", "energy_gev": 2}}, base, "d.yml")
    assert set(new) == {"pion_2GeV"} and new["pion_2GeV"]["folder"] == "Pion_2GeV"   # a new run: derived key
    _raises(SystemExit, _dataset_overrides, {"muon 4": {"csv": "x"}}, base, "d.yml", contains="unknown field(s)")
    # per-run yml mappings: keyed by reference; the most specific key wins; an unreadable key stops
    one_of_two = by_key["electron_0.3GeV_run304"]
    mapping = {"electron": 0, "electron 0.3": 1, "electron 0.3 run 304": 2}
    assert run_setting(mapping, one_of_two) == 2 and run_setting(mapping, by_key["electron_0.3GeV_run305"]) == 1
    assert run_setting(mapping, by_key["electron_4GeV"]) == 0 and run_setting(mapping, by_key["muon_4GeV"]) is None
    assert run_setting({**mapping, "electron_0.3GeV_run304": 3}, one_of_two) == 3
    assert run_setting({"muon 9": 1}, by_key["muon_4GeV"], default="d") == "d"   # names no run: applies to none
    assert run_setting(None, one_of_two, default=5) == 5
    _raises(SystemExit, run_setting, {"pion": 1, "e0.3": 2}, one_of_two, what="clean_dataset.selection",
            contains="[config] clean_dataset.selection: key 'e0.3'")


# ----------------------------------------------------------------------------------------------
# test subsets
# ----------------------------------------------------------------------------------------------
# INF-03: 6 events x 2 bars, 5 requested, chunks of 4 rows -> events 1..5 complete.
def test_subset_small_chunks_bar_major():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "beam.csv", Path(tmp) / "sub.csv"
        rows = [_row(e, 1, s) for s in (0, 1) for e in range(1, 7)]  # bar-major
        _write(src, rows)
        _quiet(io_utils.build_test_subset, src, dst, 5, chunksize=4)
        got = pd.read_csv(dst)
        assert sorted(got.pf_event.unique()) == [1, 2, 3, 4, 5]
        assert len(got) == 10
        kept = dst.read_bytes().decode().splitlines()
        assert kept[0] == HEADER
        assert kept[1:] == [r for r in rows if int(r.split(",")[0]) <= 5]  # byte-exact lines, original order
        rec = _record(dst)
        assert rec["protocol"] == io_utils.PRODUCT_SET_PROTOCOL and list(rec["members"]) == ["sub.csv"]
        assert rec["members"]["sub.csv"]["size"] == dst.stat().st_size and rec["n_rows"] == 10
        assert not [p for p in Path(tmp).iterdir() if ".tmp" in p.name or ".old-" in p.name]


# subset event major sparse nonconsecutive
def test_subset_event_major_sparse_nonconsecutive():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "beam.csv", Path(tmp) / "sub.csv"
        events = [40, 7, 3, 99, 12, 5]
        rows = []
        for e in events:  # event-major, not sorted, bars missing for some events
            rows.append(_row(e, 1, 0))
            if e % 2:
                rows.append(_row(e, 2, 3))
        _write(src, rows)
        _quiet(io_utils.build_test_subset, src, dst, 4, chunksize=3)
        assert sorted(pd.read_csv(dst).pf_event.unique()) == [3, 5, 7, 12]
        record = _record(dst)
        assert record["n_events"] == 4 and record["first_event"] == 3 and record["last_event"] == 12


# subset fewer events than requested and cache invalidation
def test_subset_fewer_events_than_requested_and_cache_invalidation():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "beam.csv", Path(tmp) / "sub.csv"
        _write(src, [_row(e, 1, 0) for e in (1, 2)])
        _, out = _quiet(io_utils.build_test_subset, src, dst, 5, chunksize=1)
        assert "only 2 events" in out
        assert _record(dst)["n_events"] == 2
        # same source -> reused (no rebuild message, no scan of the source)
        with patch.object(io_utils, "_smallest_event_ids", side_effect=AssertionError("rescanned")):
            _, out = _quiet(io_utils.build_test_subset, src, dst, 5, chunksize=1)
        assert "extracting" not in out
        # changed source -> rebuilt
        _write(src, [_row(e, 1, 0) for e in (1, 2, 3)])
        _, out = _quiet(io_utils.build_test_subset, src, dst, 5, chunksize=1)
        assert "extracting" in out
        assert sorted(pd.read_csv(dst).pf_event) == [1, 2, 3]
        # missing provenance -> never trusted
        dst.with_suffix(".provenance.json").unlink()
        _, out = _quiet(io_utils.build_test_subset, src, dst, 5, chunksize=1)
        assert "extracting" in out


# A record without member signatures is never trusted.
def test_subset_record_without_member_signatures_is_rebuilt():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "beam.csv", Path(tmp) / "sub.csv"
        _write(src, [_row(e, 1, 0) for e in (1, 2)])
        _quiet(io_utils.build_test_subset, src, dst, 1)
        rec = _record(dst)
        old = {k: v for k, v in rec.items() if k not in ("protocol", "members", "committed")}
        dst.with_suffix(".provenance.json").write_text(json.dumps(old))
        _, out = _quiet(io_utils.build_test_subset, src, dst, 1)
        assert "no member signatures" in out and "extracting" in out


# subset changed or truncated output is rebuilt
def test_subset_changed_or_truncated_output_is_rebuilt():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "beam.csv", Path(tmp) / "sub.csv"
        _write(src, [_row(e, 1, 0) for e in (1, 2, 3)])
        _quiet(io_utils.build_test_subset, src, dst, 2)
        good = dst.read_bytes()
        # same size and modification time, different content (one digit) -> the content hash catches it
        st = dst.stat()
        dst.write_bytes(good.replace(b"\n1,", b"\n9,", 1))
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        ok, reason = io_utils.product_set_valid(dst.with_suffix(".provenance.json"))
        assert not ok and "content" in reason
        _, out = _quiet(io_utils.build_test_subset, src, dst, 2)
        assert "extracting" in out and dst.read_bytes() == good
        dst.write_bytes(good[:-10])  # truncated
        _, out = _quiet(io_utils.build_test_subset, src, dst, 2)
        assert "extracting" in out and dst.read_bytes() == good


# subset sidecar failure never reuses other generation
def test_subset_sidecar_failure_never_reuses_other_generation():
    # INF2-01 (Codex probe, both forms): the subset csv is replaced, then the record write fails.
    # The next request must rebuild instead of reusing the csv of the failed run.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # (a) same source, N changes 1 -> 2 (failed) -> 1
        src, dst = tmp / "source.csv", tmp / "subset.csv"
        src.write_text("pf_event,layer,strip\n1,1,0\n2,1,0\n", encoding="utf-8")
        _quiet(io_utils.build_test_subset, src, dst, 1, chunksize=1)
        with patch.object(io_utils, "write_json", side_effect=OSError("simulated sidecar failure")):
            _raises(OSError, io_utils.build_test_subset, src, dst, 2, chunksize=1)
        assert not dst.with_suffix(".provenance.json").exists()       # no record pairs with the new csv
        _quiet(io_utils.build_test_subset, src, dst, 1, chunksize=1)
        assert pd.read_csv(dst).pf_event.tolist() == [1] and _record(dst)["n_events_requested"] == 1
        # (b) source switch A -> B (failed) -> A at constant N and key
        src_a, src_b, dst = tmp / "a.csv", tmp / "b.csv", tmp / "muon_4GeV_first1events.csv"
        src_a.write_text("pf_event,layer,strip\n1,1,0\n", encoding="utf-8")
        src_b.write_text("pf_event,layer,strip\n99,1,0\n", encoding="utf-8")
        _quiet(io_utils.build_test_subset, src_a, dst, 1, chunksize=1)
        with patch.object(io_utils, "write_json", side_effect=OSError("simulated provenance write failure")):
            _raises(OSError, io_utils.build_test_subset, src_b, dst, 1, chunksize=1)
        _quiet(io_utils.build_test_subset, src_a, dst, 1, chunksize=1)
        assert pd.read_csv(dst).pf_event.tolist() == [1]
        assert _record(dst)["source"] == io_utils.file_signature(src_a)
        assert not [p for p in tmp.iterdir() if ".tmp" in p.name or ".old-" in p.name]


# The csv cannot be replaced (e.g. opened in Excel): old csv AND old record stay byte-identical and valid.
def test_subset_rename_failure_keeps_the_old_pair():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, dst = tmp / "source.csv", tmp / "subset.csv"
        src.write_text("pf_event,layer,strip\n1,1,0\n2,1,0\n", encoding="utf-8")
        _quiet(io_utils.build_test_subset, src, dst, 1)
        meta = dst.with_suffix(".provenance.json")
        old_csv, old_meta = dst.read_bytes(), meta.read_bytes()
        real_replace = os.replace

        # os.replace that fails on dst (a file in use)
        def locked(a, b):
            if Path(b) == dst:
                raise PermissionError("simulated: file in use")
            return real_replace(a, b)

        with patch.object(io_utils.os, "replace", side_effect=locked):
            _raises(PermissionError, io_utils.build_test_subset, src, dst, 2)
        assert dst.read_bytes() == old_csv and meta.read_bytes() == old_meta
        assert not [p for p in tmp.iterdir() if ".tmp" in p.name or ".old-" in p.name]
        with patch.object(io_utils, "_smallest_event_ids", side_effect=AssertionError("rescanned")):
            _quiet(io_utils.build_test_subset, src, dst, 1)  # the old pair is still valid: reused


# CAL2-02: the subset record carries n_samples / n_samples_used of the make-csv provenance of its source.
def test_subset_carries_converter_sample_counts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, dst = tmp / "fixed_x.csv", tmp / "sub.csv"
        _write(src, [_row(e, 1, 0) for e in (1, 2)])
        st = src.stat()
        prov = {"n_samples": 10, "n_samples_used": 8, "waveform": False,
                "output": {"size": st.st_size, "mtime_ns": st.st_mtime_ns}}
        (tmp / "fixed_x.provenance.json").write_text(json.dumps(prov))
        _quiet(io_utils.build_test_subset, src, dst, 1)
        rec = _record(dst)
        assert (rec["n_samples"], rec["n_samples_used"], rec["waveform"]) == (10, 8, False)
        assert rec["source_provenance"]["matches_source"] is True
        # a provenance record that describes another version of the source is not taken over
        prov["output"]["size"] += 1
        (tmp / "fixed_x.provenance.json").write_text(json.dumps(prov))
        dst.unlink()
        _quiet(io_utils.build_test_subset, src, dst, 1)
        rec = _record(dst)
        assert "n_samples" not in rec and rec["source_provenance"]["matches_source"] is False
        # a record without the output signature of the file (hand-made / foreign) is not trusted either
        (tmp / "fixed_x.provenance.json").write_text(json.dumps({"n_samples": 10, "n_samples_used": 4}))
        dst.unlink()
        _quiet(io_utils.build_test_subset, src, dst, 1)
        rec = _record(dst)
        assert "n_samples_used" not in rec and rec["source_provenance"]["matches_source"] is False
        # no provenance at all: nothing claimed
        (tmp / "fixed_x.provenance.json").unlink()
        dst.unlink()
        _quiet(io_utils.build_test_subset, src, dst, 1)
        assert "source_provenance" not in _record(dst)


# input csv builds the subset in the cache
def test_input_csv_builds_the_subset_in_the_cache():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _run_csv(tmp)
        src.parent.mkdir(parents=True)
        _write(src, [_row(e, 1, 0) for e in (5, 1, 3)])
        ctx = _ctx_on(tmp, n_test_events=2)
        subset = tmp / "cache" / "test" / "test_subsets" / "Muon_4GeV" / "beam_first2events.csv"
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            ctx.dry_run = True
            assert ctx.input_csv(_ds(), src) == subset
            assert not (tmp / "cache").exists()  # a dry run builds nothing
            ctx.dry_run = False
            path, _ = _quiet(ctx.input_csv, _ds(), src)
        assert path == subset
        assert sorted(pd.read_csv(path).pf_event) == [1, 3]
        assert RunContext(cfg=ctx.cfg).input_csv(_ds(), src) == src  # full set: the csv itself


# a run is the folder of its csv files
def test_a_run_is_the_folder_of_its_csv_files():
    # make-csv: no merging (Lei 2026-10-20): a run is its folder data/analysis_files/<Particle>_<E>GeV/, one csv per
    # ROOT file; every stage reads every csv of it (test mode: the first-N-events subset of each file) and writes one
    # product per csv; an event is (event source, pf_event).
    from ldmx_hcal.common.config import EVENT_ID_COLUMNS

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds = Dataset("muon_4GeV", "muon", 4.0, "Muon 4", root_files=[tmp / "root" / "b.root", tmp / "root" / "a.root"])
        ctx = _ctx_on(tmp, n_test_events=1)
        run = tmp / "data" / "analysis_files" / "Muon_4GeV"
        assert ctx.run_csvs(ds) == [run / "b.csv", run / "a.csv"]          # not made yet: one per ROOT file
        items = ctx.input_plan_items(ds)
        assert [i["kind"] for i in items] == ["auto", "auto"] and items[0]["source"] == run / "b.csv"
        run.mkdir(parents=True)
        for name in ("b.csv", "a.csv"):  # two files that both start at pf_event 1
            _write(run / name, [_row(e, 1, 0) for e in (1, 2)])
        assert ctx.run_csvs(ds) == [run / "a.csv", run / "b.csv"]          # every csv of the folder, by name
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            pairs, _ = _quiet(ctx.run_inputs, ds)
        subsets = tmp / "cache" / "test" / "test_subsets" / "Muon_4GeV"
        assert pairs == [(run / "a.csv", subsets / "a_first1events.csv"),
                         (run / "b.csv", subsets / "b_first1events.csv")]
        assert [sorted(pd.read_csv(p).pf_event.unique()) for _, p in pairs] == [[1], [1]]
        assert RunContext(cfg=ctx.cfg).input_csvs(ds) == [run / "a.csv", run / "b.csv"]
        assert ctx.csv_product(ctx.data_dir("x"), ds, run / "a.csv", "_y.csv") == (
            tmp / "data" / "test" / "x" / "Muon_4GeV" / "a_y.csv")
        assert EVENT_ID_COLUMNS[1] == "pf_event"
        assert {ctx.event_source(ds, c) for c, _ in pairs} == {"Muon_4GeV/a.csv", "Muon_4GeV/b.csv"}
        empty = Dataset("pion_4GeV", "pion", 4.0, "Pion 4")
        assert ctx.input_plan_items(empty) == [{"role": "in", "kind": "external",
                                                "path": tmp / "data" / "analysis_files" / "Pion_4GeV" / "*.csv",
                                                "produced_by": ".\\runner.ps1 make-csv --pion 4"}]
        _raises(SystemExit, ctx.input_csvs, empty, contains="no csv file in")


# only units and output names by result
def test_only_units_and_output_names_by_result():
    # -only UNIT[,UNIT]: units or groups of a stage, unknown -> argparse error listing the units, default units
    # without it; -output names match results by name (the ci quality-scan overwrite bug): naming one file saves
    # only that file.
    import argparse

    from ldmx_hcal.common import outputs

    p = argparse.ArgumentParser()
    outputs.add_only_argument(p, ["table", "fits", "scan"], groups={"pdf": ["fits"]}, default=["table", "pdf"])
    assert p.parse_args([]).only == ("table", "fits")
    assert p.parse_args(["-only", "scan,pdf"]).only == ("fits", "scan")
    assert p.parse_args(["-only", "scan", "table"]).only == ("table", "scan")
    assert p.parse_args(["-only", "all"]).only == ("table", "fits", "scan")
    err = io.StringIO()
    with patch("sys.stderr", err):
        assert _raises(SystemExit, p.parse_args, ["-only", "nope"]).code == 2
    assert "unknown unit; units: table, fits, scan; groups: all, pdf (fits)" in err.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx_on(tmp, test=False)
        assert ctx.wants("anything")                                   # no -only: every unit
        ctx.units = p.parse_args(["-only", "scan"]).only
        assert ctx.wants("scan") and not ctx.wants("table") and ctx.wants("table", "scan")
        res = ctx.results_dir("ci")
        slots = [outputs.ResultSlot("energy_response", "csv", res / "energy_response.csv"),
                 outputs.ResultSlot("quality_threshold_scan", "csv", res / "quality_threshold_scan.csv"),
                 outputs.ResultSlot("spectra", "pdf", res / "energy_spectra.pdf")]

        # saved file name of every result for these -output names, and the plan
        def plan(names):
            pl = outputs.OutputPlan(ctx, "ci", slots, names=names, preview=True)
            return {s.key: (pl.path(s.key).name if pl.saved(s.key) else None) for s in slots}, pl

        saved, _ = plan(["quality_threshold_scan.csv"])
        assert saved == {"energy_response": None, "quality_threshold_scan": "quality_threshold_scan.csv",
                         "spectra": None}
        saved, _ = plan(["spectra=mine.pdf", "energy_response.csv"])
        assert saved == {"energy_response": "energy_response.csv", "quality_threshold_scan": None,
                         "spectra": "mine.pdf"}
        assert plan([])[0] == {s.key: Path(s.default).name for s in slots}
        assert plan(None)[0] == {s.key: None for s in slots}
        saved, pl = plan(["other.csv"])
        assert saved == {s.key: None for s in slots} and "other.csv" in pl.warnings[0]
        _raises(io_utils.SettingsError, plan, ["spectra=x.csv"], contains="is a pdf file")
        _raises(argparse.ArgumentTypeError, outputs.output_name, "dir/x.csv")


# no json writes no record next to products
def test_no_json_writes_no_record_next_to_products():
    # -no-json: no record json next to the products in data/, calibration/, results/ (an old one is removed);
    # the cache records stay; a check that needs the missing record does not verify (no crash).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx_on(tmp, test=False, write_json=False)
        try:
            prod = ctx.data_dir("clean_dataset", "x.csv")
            rec = prod.with_name("x.provenance.json")
            rec.parent.mkdir(parents=True)
            rec.write_text("{}")
            io_utils.commit_product_set({prod: lambda t: t.write_text("a\n1\n")}, {"stage": "t"}, rec)
            assert prod.exists() and not rec.exists()
            assert io_utils.product_set_valid(rec, {"stage": "t"}, [prod]) == (
                True, "not verified (-no-json: no record x.provenance.json)")
            assert not io_utils.product_set_valid(rec, {"stage": "t"})[0]
            io_utils.write_json({"a": 1}, ctx.results_dir("s", "r.json"))
            assert not ctx.results_dir("s", "r.json").exists()
            cache = ctx.cache_dir("s", "c.json")
            io_utils.write_json({"a": 1}, cache)
            assert cache.exists()
        finally:
            _ctx_on(tmp, test=False)                                    # records on again
        assert not io_utils.product_set_valid(rec, {"stage": "t"}, [prod])[0]


# ----------------------------------------------------------------------------------------------
# product sets
# ----------------------------------------------------------------------------------------------
# product set commit and validation
def test_product_set_commit_and_validation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        table, hist, meta = tmp / "s" / "table.csv", tmp / "s" / "hist.npz", tmp / "s" / "table.provenance.json"
        with io_utils.ProductSet(meta) as ps:
            pd.DataFrame({"a": [1, 2]}).to_csv(ps.stage(table), index=False)
            staged = ps.stage(hist)
            assert staged.suffix == ".npz" and staged.parent == hist.parent
            np.savez(staged, h=np.arange(5))  # np.savez would append .npz to a name without it
            rec = ps.commit({"input": {"size": 3}, "settings": {"window": (1, 2)}, "algorithm": "v1"})
        assert set(rec["members"]) == {"table.csv", "hist.npz"} and rec["protocol"] == io_utils.PRODUCT_SET_PROTOCOL
        assert sorted(p.name for p in meta.parent.iterdir()) == ["hist.npz", "table.csv", "table.provenance.json"]
        want = {"input": {"size": 3}, "settings": {"window": [1, 2]}, "algorithm": "v1"}
        assert io_utils.product_set_valid(meta, want, members=[table, hist]) == (True, "ok")
        ok, reason = io_utils.product_set_valid(meta, {**want, "algorithm": "v2"})
        assert not ok and "'algorithm' differs" in reason
        ok, reason = io_utils.product_set_valid(meta, {**want, "schema": None})   # absent never equals None
        assert not ok and "'schema' is not recorded" in reason
        ok, reason = io_utils.product_set_valid(meta, want, members=[tmp / "s" / "other.csv"])
        assert not ok and "not part of the stored set" in reason
        hist.unlink()
        ok, reason = io_utils.product_set_valid(meta, want)
        assert not ok and "missing" in reason
        exc = _raises(io_utils.ProductSetError, io_utils.require_product_set, meta, want,
                      ".\\runner.ps1 calib -test", what="histogram cache", contains="Rebuild it with")
        assert isinstance(exc, ValueError) and isinstance(exc, SystemExit) and "missing" in str(exc)
        ok, reason = io_utils.product_set_valid(tmp / "nothing.json")
        assert not ok and "no metadata" in reason


# product set failures never mix generations
def test_product_set_failures_never_mix_generations():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a, b, meta = tmp / "a.csv", tmp / "b.csv", tmp / "set.json"

        # member writer that writes text
        def writer(text):
            return lambda p: p.write_text(text)

        io_utils.commit_product_set({a: writer("a1"), b: writer("b1")}, {"gen": 1}, meta)
        old = (a.read_bytes(), b.read_bytes(), meta.read_bytes())

        # member writer that writes part of a file and fails
        def boom(p):
            p.write_text("partial")
            raise RuntimeError("writer failed")

        # a writer fails: nothing replaced, temp files removed, the old set stays valid
        _raises(RuntimeError, io_utils.commit_product_set, {a: writer("a2"), b: boom}, {"gen": 2}, meta)
        assert (a.read_bytes(), b.read_bytes(), meta.read_bytes()) == old
        assert io_utils.product_set_valid(meta, {"gen": 1})[0]
        assert sorted(p.name for p in tmp.iterdir()) == ["a.csv", "b.csv", "set.json"]
        # the second rename fails: a is new, b is old -> no metadata at all (rebuilt next time)
        real_replace = os.replace

        # os.replace that fails on the second member
        def second_fails(x, y):
            if Path(y) == b:
                raise PermissionError("simulated lock")
            return real_replace(x, y)

        with patch.object(io_utils.os, "replace", side_effect=second_fails):
            _raises(PermissionError, io_utils.commit_product_set, {a: writer("a3"), b: writer("b3")}, {"gen": 3}, meta)
        assert not meta.exists() and not io_utils.product_set_valid(meta, {"gen": 1})[0]
        assert sorted(p.name for p in tmp.iterdir()) == ["a.csv", "b.csv"]
        # the metadata write fails: members are new, no metadata -> invalid
        io_utils.commit_product_set({a: writer("a4"), b: writer("b4")}, {"gen": 4}, meta)
        with patch.object(io_utils, "write_json", side_effect=OSError("disk full")):
            _raises(OSError, io_utils.commit_product_set, {a: writer("a5"), b: writer("b5")}, {"gen": 5}, meta)
        assert not meta.exists() and a.read_text() == "a5"
        assert sorted(p.name for p in tmp.iterdir()) == ["a.csv", "b.csv"]
        _raises(ValueError, io_utils.commit_product_set, {a: writer("x")}, {"members": 1}, meta)  # reserved key


# product set large file hash and protection
def test_product_set_large_file_hash_and_protection():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        big, meta = tmp / "big.bin", tmp / "big.json"
        with patch.object(io_utils, "FULL_HASH_LIMIT", 100), patch.object(io_utils, "EDGE_HASH_BYTES", 16):
            io_utils.commit_product_set({big: lambda p: p.write_bytes(bytes(range(256)) * 4)}, {}, meta)
            assert json.loads(meta.read_text())["members"]["big.bin"]["hash"] == "edges-4MiB"
            assert io_utils.product_set_valid(meta)[0]
            st = big.stat()
            data = bytearray(big.read_bytes())
            data[-1] ^= 0xFF  # tail changed, same size and mtime
            big.write_bytes(bytes(data))
            os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns))
            assert not io_utils.product_set_valid(meta)[0]
        protected = tmp / "old"
        protected.mkdir()
        io_utils.set_protected_dirs([protected])
        try:
            _raises(SystemExit, io_utils.commit_product_set, {protected / "x.csv": lambda p: p.write_text("x")}, {},
                    tmp / "x.json", contains="protected")
            _raises(SystemExit, io_utils.commit_product_set, {tmp / "y.csv": lambda p: p.write_text("y")}, {},
                    protected / "y.json", contains="protected")
            assert not any(protected.iterdir()) and not (tmp / "y.csv").exists()
        finally:
            io_utils.set_protected_dirs([])


# The pid of a process that has ended.
def _dead_pid() -> int:
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    pid = proc.pid
    del proc
    return pid


# Temp files of a killed run are swept only when their process is gone and they are old.
def test_orphan_temps_of_killed_runs_are_swept():
    # Windows soon gives the pid of an ended process to a new one (other programs start processes all the time):
    # the sweep runs with a fixed pid table (our own pid alive, the dead pid gone), so it cannot flake on a reused pid
    assert io_utils._pid_alive(os.getpid())
    # a just-ended process is reported dead (a few tries: its pid may already belong to a new process)
    assert any(not io_utils._pid_alive(_dead_pid()) for _ in range(3))
    dead = 4 * 999_983                                                  # never alive in the patched table
    alive = patch.object(io_utils, "_pid_alive", new=lambda pid: int(pid) == os.getpid())
    with tempfile.TemporaryDirectory() as tmp, alive:
        tmp = Path(tmp) / "sweep"
        tmp.mkdir()
        old = io_utils.time.time() - 2 * io_utils.ORPHAN_MIN_AGE_S
        names = {f"sub.tmp-{dead}-0123abcd.csv": True,                 # killed subset build: removed
                 f"sub.provenance.json.old-{dead}-0123abcd": True,      # its metadata backup: removed
                 f"live.tmp-{os.getpid()}-0123abcd.csv": False,         # a running process: kept
                 "data.csv": False, "table.csv.tmp": False,             # not a unique temp name: kept
                 f"x.tmp-{dead}-0123abcz.csv": False}                   # not the exact pattern: kept
        for name in names:
            (tmp / name).write_text("x")
            os.utime(tmp / name, (old, old))
        recent = tmp / f"recent.tmp-{dead}-0123abcd.csv"                # dead pid but touched just now: kept
        recent.write_text("x")
        meta = tmp / "set.json"
        _, out = _quiet(io_utils.commit_product_set, {tmp / "a.csv": lambda p: p.write_text("a")}, {}, meta)
        left = sorted(p.name for p in tmp.iterdir())
        assert left == sorted([n for n, gone in names.items() if not gone] + [recent.name, "a.csv", "set.json"]), left
        assert "removed 2 temp file(s)" in out
        assert io_utils.product_set_valid(meta)[0]
        # a protected folder is never swept
        io_utils.set_protected_dirs([tmp])
        try:
            assert io_utils.sweep_orphan_temps(tmp, min_age_s=0) == []
        finally:
            io_utils.set_protected_dirs([])
        assert io_utils.sweep_orphan_temps(tmp, min_age_s=0) == [recent]


# ----------------------------------------------------------------------------------------------
# temp files of a run (CI-R2-04)
# ----------------------------------------------------------------------------------------------
# temp registry removes every registered file
def test_temp_registry_removes_every_registered_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        closed = []

        # stand-in of a worker handle that records its closing
        class Handle:
            # record that the handle was closed
            def close(self):
                closed.append(True)

        # a run with a worker temp, a handle and a csv writer; fail: it raises
        def run(fail: bool):
            with io_utils.temp_registry() as reg:
                worker_tmp = reg.register(tmp / "dataset.bin.tmp")   # registered before the worker starts
                worker_tmp.write_bytes(b"partial")                    # the worker wrote it, its result never came
                reg.register_closer(Handle())
                writer = io_utils.ChunkedCsvWriter(tmp / "events.csv")
                writer.write(pd.DataFrame({"a": [1]}))
                assert writer.tmp in reg.paths
                io_utils.atomic_write_csv(pd.DataFrame({"b": [2]}), tmp / "done.csv")
                if fail:
                    raise RuntimeError("worker failed")
                writer.close()

        _raises(RuntimeError, run, True)
        assert sorted(p.name for p in tmp.iterdir()) == ["done.csv"] and closed == [True]
        run(False)  # success: the committed csv stays, the orphan worker file is still removed
        assert sorted(p.name for p in tmp.iterdir()) == ["done.csv", "events.csv"] and len(closed) == 2
        assert io_utils.active_registry() is None


# writers clean up when the rename fails
def test_writers_clean_up_when_the_rename_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "old.csv").write_text("old\n")
        real_replace = os.replace

        # os.replace that fails on old.csv and x.pdf
        def fail(a, b):
            if Path(b).name in ("old.csv", "x.pdf"):
                raise PermissionError("simulated lock")
            return real_replace(a, b)

        with patch.object(io_utils.os, "replace", side_effect=fail):
            _raises(PermissionError, io_utils.atomic_write_csv, pd.DataFrame({"a": [1]}), tmp / "old.csv")
            w = io_utils.ChunkedCsvWriter(tmp / "old.csv")
            w.write(pd.DataFrame({"a": [1]}))
            _raises(PermissionError, w.close)
            _raises(PermissionError, io_utils.write_json, {"a": 1}, tmp / "old.csv")

            # write a one-page pdf with atomic_pdf
            def draw():
                import matplotlib.pyplot as plt

                with io_utils.atomic_pdf(tmp / "x.pdf") as pdf:
                    fig = plt.figure()
                    pdf.savefig(fig)
                    plt.close(fig)

            _raises(PermissionError, draw)
        assert sorted(p.name for p in tmp.iterdir()) == ["old.csv"] and (tmp / "old.csv").read_text() == "old\n"
        w = io_utils.ChunkedCsvWriter(tmp / "f.csv", float_format="%.3g")  # to_csv options pass through
        w.write(pd.DataFrame({"x": [1.0 / 3]}))
        w.close()
        assert (tmp / "f.csv").read_text().splitlines() == ["x", "0.333"]


# ----------------------------------------------------------------------------------------------
# channel keys (INF2-02 and the shared rule of CLEAN2-02 / CI-R2-03 / R2-TP-01)
# ----------------------------------------------------------------------------------------------
# validate channel keys rule
def test_validate_channel_keys_rule():
    ok = pd.DataFrame({"layer": [1.0, 9, 10, 19], "strip": [0, 7, 11, 0], "end": [0, 1, 0, 1], "mpv": [1.0, 2, 3, 4]})
    out = io_utils.validate_channel_keys(ok, source="mpv table")
    assert out["layer"].dtype == np.int64 and out["layer"].tolist() == [1, 9, 10, 19]
    assert out["mpv"].tolist() == [1.0, 2, 3, 4] and ok["layer"].dtype == np.float64  # input untouched
    bad_rows = {
        "257 / 258 (never wrapped to a channel)": (257, 258, 0),
        "-255": (-255, 0, 0),
        "strip 8 in an 8-bar layer": (1, 8, 0),
        "strip 12 in a 12-bar layer": (10, 12, 0),
        "end 2": (1, 0, 2),
        "layer 0": (0, 0, 0),
        "fraction": (1, 0.9, 0),
        "NaN": (1, np.nan, 0),
        "text": (1, "x", 0),
    }
    for what, (layer, strip, end) in bad_rows.items():
        df = pd.DataFrame({"layer": [2, layer], "strip": [1, strip], "end": [0, end]})
        exc = _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, df, source="T")
        assert isinstance(exc, ValueError) and str(exc).startswith("T: invalid channel keys"), what
        assert "row 1" in str(exc), (what, str(exc))
    # duplicates: rejected whatever the row order (a conflicting duplicate can never pick a value)
    dup = pd.DataFrame({"layer": [1, 1], "strip": [0, 0], "end": [0, 0], "mpv": [100.0, 900.0]})
    for frame in (dup, dup.iloc[::-1]):
        _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, frame, contains="share a")
    assert len(io_utils.validate_channel_keys(dup, unique=False)) == 2  # event tables: ranges only
    # node tables: unique per channel + node index, several nodes per channel are fine
    nodes = pd.DataFrame({"layer": [1, 1, 1], "strip": [0, 0, 0], "end": [0, 0, 1], "node_index": [0, 1, 0]})
    io_utils.validate_channel_keys(nodes, node_key="node_index")
    _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, nodes.assign(node_index=[0, 0, 0]),
            node_key="node_index", contains="node_index")
    _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, nodes.assign(node_index=[0, np.nan, 0]),
            node_key="node_index", contains="empty")
    for bad_nodes in ([0.5, 1.5, 0], [0, -1, 0], [0, "x", 0]):   # node indices are integers >= 0 as well
        _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, nodes.assign(node_index=bad_nodes),
                node_key="node_index", contains="node_index")
    out = io_utils.validate_channel_keys(nodes.assign(node_index=[0.0, 1.0, 0.0]), node_key="node_index")
    assert out["node_index"].dtype == np.int64 and out["node_index"].tolist() == [0, 1, 0]
    # true / false are never read as channel 1 / 0
    _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys,
            pd.DataFrame({"layer": [True, True], "strip": [0, 1], "end": [0, 0]}), contains="true/false")
    _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys,
            pd.DataFrame({"layer": [1, 2], "strip": [0, 1], "end": pd.Series([False, 1], dtype=object)}),
            contains="true/false")
    _raises(io_utils.ChannelKeyError, io_utils.validate_channel_keys, nodes.drop(columns="end"),
            contains="missing key column")
    # bar tables: (layer, strip) only
    io_utils.validate_channel_keys(pd.DataFrame({"layer": [3, 12], "strip": [7, 11]}), keys=("layer", "strip"))
    _raises(ValueError, io_utils.validate_channel_keys, ok, keys=("layer", "channel"))


# iter csv checks keys before narrowing
def test_iter_csv_checks_keys_before_narrowing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wrap = tmp / "wrap.csv"
        wrap.write_text("pf_event,layer,strip\n1,257,258\n", encoding="utf-8")   # Codex INF2-02 probe
        _raises(io_utils.ChannelKeyError, lambda: next(io_utils.iter_csv(wrap, chunksize=1)), contains="csv line 2")
        for body in ("1,-255,0", "1,1,", "1,1.5,0", "1,1,8", "1,20,0"):
            p = tmp / "bad.csv"
            p.write_text(f"pf_event,layer,strip\n1,1,0\n2,2,1\n{body}\n", encoding="utf-8")
            exc = _raises(io_utils.ChannelKeyError, lambda: list(io_utils.iter_csv(p, chunksize=2)))
            assert "csv line 4" in str(exc), (body, str(exc))   # found in the second chunk, right line
        # valid input: exactly the frame a direct read with the compact dtypes gives
        good = tmp / "good.csv"
        _write(good, [_row(e, layer, s) for e in (1, 2) for layer, s in ((1, 0), (9, 7), (19, 11))])
        chunks = list(io_utils.iter_csv(good, chunksize=4))
        got = pd.concat(chunks)
        direct = pd.read_csv(good, dtype=io_utils._dtypes_for(io_utils.csv_columns(good)))
        pd.testing.assert_frame_equal(got, direct)
        assert got["layer"].dtype == np.int8 and got["strip"].dtype == np.int8
        cols = ["pf_event", "layer", "strip"]
        wide = pd.concat(io_utils.iter_csv(good, cols, dtypes={"layer": "int16", "strip": "int16"}))
        assert wide["layer"].dtype == np.int16 and wide["layer"].tolist() == direct["layer"].tolist()
        no_keys = next(io_utils.iter_csv(good, ["pf_event", "adc_sum_end0"]))   # nothing to check
        assert list(no_keys.columns) == ["pf_event", "adc_sum_end0"]
        _raises(io_utils.SettingsError, lambda: next(io_utils.iter_csv(good, chunksize=0)))
        assert len(io_utils.read_csv_columns(good, cols, row_filter=lambda c: c["layer"] > 1)) == 4


# ----------------------------------------------------------------------------------------------
# settings validation (Codex B6)
# ----------------------------------------------------------------------------------------------
# validate settings
def test_validate_settings():
    spec = {"chunksize": "int>0", "lam": "float>=0", "thr": "float", "frac": "fraction", "win": "window",
            "mode": ("choice", ["full", "low", "line"]), "steps": "even_int>0", "cut": "float>0?",
            "grid": ("list", "float>0"), "flag": "bool", "k": ("range", 1, 10)}
    good = {"chunksize": 5000, "lam": 0.0, "thr": -3.5, "frac": 0.125, "win": [1.2, 4], "mode": "line",
            "steps": 4000, "cut": None, "grid": [1, 2.5], "flag": False, "k": 5, "unchecked": float("nan")}
    assert io_utils.validate_settings(good, spec) is good
    assert io_utils.validate_settings({"chunksize": 5000.0}, spec)                 # integral float is an int
    bad = {"chunksize": 0, "lam": -1, "thr": float("nan"), "frac": 1.5, "win": [4, 1.2], "mode": "auto",
           "steps": 3, "cut": float("inf"), "grid": [1, -2], "flag": "yes", "k": 11}
    exc = _raises(io_utils.SettingsError, io_utils.validate_settings, bad, spec, source="ci")
    assert isinstance(exc, ValueError) and isinstance(exc, SystemExit)
    text = str(exc)
    assert text.startswith("ci: invalid setting(s)")
    for key in bad:
        assert f"{key} = " in text, key
    for value in (True, "5", 2.5, None):
        _raises(io_utils.SettingsError, io_utils.validate_settings, {"chunksize": value}, spec)
    _raises(io_utils.SettingsError, io_utils.validate_settings, {"win": [1, 1]}, spec)       # not ordered
    _raises(io_utils.SettingsError, io_utils.validate_settings, {"win": [1, 2, 3]}, spec)
    _raises(io_utils.SettingsError, io_utils.validate_settings, {"win": [0, None]}, spec)    # closed window
    # open windows (e.g. the make-csv sample window [0, null]): None = open end, order checked when both set
    for value in ([0, None], [None, 8], [None, None], [2, 8]):
        io_utils.validate_settings({"w": value}, {"w": "window_open"})
    for value in ([8, 2], [0, float("nan")], [0], "0-8", [True, 8]):
        _raises(io_utils.SettingsError, io_utils.validate_settings, {"w": value}, {"w": "window_open"})
    _raises(ValueError, io_utils.validate_settings, {"x": 1}, {"x": "positive"})              # unknown rule
    assert io_utils.validate_settings({"x": 3}, {"x": lambda v: None if v % 2 else "odd"})


# ----------------------------------------------------------------------------------------------
# write protection
# ----------------------------------------------------------------------------------------------
# protected folders refuse writes
def test_protected_folders_refuse_writes():
    with tempfile.TemporaryDirectory() as tmp:
        protected = Path(tmp) / "old_analysis"
        protected.mkdir()
        io_utils.set_protected_dirs([protected])
        try:
            for write in (lambda p: io_utils.atomic_write_csv(pd.DataFrame({"a": [1]}), p),
                          lambda p: io_utils.write_json({"a": 1}, p),
                          lambda p: io_utils.ChunkedCsvWriter(p),
                          lambda p: io_utils.ProductSet(Path(tmp) / "m.json").stage(p)):
                try:
                    write(protected / "sub" / "x.csv")
                except SystemExit as exc:
                    assert "protected" in str(exc)
                else:
                    raise AssertionError("write into a protected folder was not refused")
            io_utils.atomic_write_csv(pd.DataFrame({"a": [1]}), Path(tmp) / "ok" / "x.csv")  # elsewhere: fine
            assert not any(protected.rglob("*"))
        finally:
            io_utils.set_protected_dirs([])


# A target beyond MAX_PATH (Windows returns it with a \\?\ prefix) is still refused.
def test_protected_folders_long_paths():
    with tempfile.TemporaryDirectory() as tmp:
        protected = Path(tmp) / "old"
        protected.mkdir()
        io_utils.set_protected_dirs([protected])
        try:
            deep = protected.joinpath(*(["d" * 60] * 5)) / "x.csv"
            assert len(str(deep)) > 300
            try:
                io_utils.ensure_writable(deep)
            except SystemExit:
                pass
            else:
                raise AssertionError("long path inside a protected folder was not refused")
            assert io_utils.canonical_path(Path(tmp) / "a" / ".." / "old") == io_utils.canonical_path(protected)
        finally:
            io_utils.set_protected_dirs([])


_CHILD = """
import sys
sys.path.insert(0, {src!r})
import pandas as pd
from ldmx_hcal.common import io_utils
try:
    io_utils.atomic_write_csv(pd.DataFrame({{"a": [1]}}), {target!r})
except SystemExit:
    print("refused")
else:
    print("written")
"""


# Child processes (worker pools are spawned, not forked, on Windows) inherit the protection.
def test_protected_folders_in_worker_processes():
    import subprocess

    src = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as tmp:
        protected = Path(tmp) / "old"
        protected.mkdir()
        io_utils.set_protected_dirs([protected])
        try:
            for target, expected in ((protected / "x.csv", "refused"), (Path(tmp) / "ok" / "x.csv", "written")):
                out = subprocess.run([sys.executable, "-c", _CHILD.format(src=src, target=str(target))],
                                     capture_output=True, text=True, timeout=120)
                assert out.stdout.strip() == expected, (target, out.stdout, out.stderr[-500:])
        finally:
            io_utils.set_protected_dirs([])


# local config protects original folders
def test_local_config_protects_original_folders():
    # Input folders outside the repository are protected; make-csv's run folders (analysis_files_dir) and the
    # product trees are written, so they are never inside a protected folder.
    cfg = load_config()
    protected = [Path(p) for p in cfg["safety"]["protected_dirs"]]
    for key in ("ci_scan_dir", "root_dir"):
        p = Path(cfg["paths"][key]).resolve()
        if REPO_ROOT in p.parents:
            continue  # data inside the repository: nothing to protect
        assert any(d == p or d in p.parents for d in protected), f"{key}={p} is outside the repo but not protected"
    for key in ("analysis_files_dir", "cache_dir", "generated_data_dir", "calibration_dir", "results_dir"):
        out = Path(cfg["paths"][key]).resolve()
        assert not any(d == out or d in out.parents for d in protected), f"{key} points into a protected folder"


# ----------------------------------------------------------------------------------------------
# --clean-input and structured io plans
# ----------------------------------------------------------------------------------------------
# clean input reads the clean dataset
def test_clean_input_reads_the_clean_dataset():
    from ldmx_hcal.selection import clean_dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds, src = _ds(), _run_csv(tmp)
        ctx = _ctx_on(tmp, test=True, clean_input=True)
        product = ctx.clean_product(ds, src)
        assert product == tmp / "data" / "test" / "clean_dataset" / "Muon_4GeV" / "beam_clean.csv"
        current = {"path": None}
        with patch.object(clean_dataset, "current_clean_csv", create=True,
                          new=lambda c, d, f: current["path"]):
            exc = _raises(SystemExit, ctx.input_csv, ds, src, contains=".\\runner.ps1 clean --muon 4 -test")
            assert "missing or was made from other inputs" in str(exc)
            item = ctx.input_plan_item(ds, src)
            assert item["kind"] == "product" and item["path"] == product and "stale" not in item
            product.parent.mkdir(parents=True)
            product.write_text("pf_event\n")
            assert ctx.input_plan_item(ds, src)["stale"] is True          # exists, but not current
            ctx.dry_run = True
            assert ctx.input_csv(ds, src) == product                      # the plan shows where it would read
            # the beam csv stays reachable (calib: pedestal rows, which the clean selection removes)
            with patch.object(RunContext, "_made_test_csv", return_value=None):
                assert ctx.beam_csv(ds, src) == ctx.test_subset_csv(ds, src)
                assert ctx.beam_plan_item(ds, src)["kind"] == "auto"
            ctx.dry_run = False
            current["path"] = product
            assert ctx.input_csv(ds, src) == product
            item = ctx.input_plan_item(ds, src)
            assert item == {"role": "in", "kind": "product", "path": product,
                            "produced_by": ".\\runner.ps1 clean --muon 4 -test"}
            full = _ctx_on(tmp, test=False, clean_input=True)
            assert full.beam_csv(ds, src) == src
            assert full.input_plan_item(ds, src)["produced_by"] == ".\\runner.ps1 clean --muon 4"


# check --test: an auto-built subset whose full csv exists is neither external-missing nor pending.
def test_plan_items_and_check_classification():
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _run_csv(tmp)
        src.parent.mkdir(parents=True)
        ds = _ds()
        ctx = _ctx_on(tmp, test=True, dry_run=True)
        with patch.object(RunContext, "_made_test_csv", return_value=None):
            item = ctx.input_plan_item(ds, src, ".\\runner.ps1 make-csv --muon 4")
            assert item["kind"] == "auto" and item["source"] == src and item["path"] == ctx.test_subset_csv(ds, src)
            counts = lambda plan: _quiet(runner.print_plan, "x", plan)[0]  # noqa: E731
            assert counts([item]) == (1, 0)          # neither the subset nor its source exists: external missing
            _write(src, [_row(1, 1, 0)])
            assert runner.classify_input(item) == "AUTO" and counts([item]) == (0, 0)   # Codex INF2-OPT
            ctx.dry_run = False
            _quiet(ctx.input_csv, ds, src)
            assert runner.classify_input(item) == "ok" and item["n_events"] == 5000
            made = tmp / "made.csv"
            made.write_text("x")
            with patch.object(RunContext, "_made_test_csv", return_value=made):
                item = ctx.input_plan_item(ds, src)
            assert item == {"role": "in", "kind": "product", "path": made,
                            "produced_by": ".\\runner.ps1 make-csv --muon 4 -test"}
        full = RunContext(cfg=ctx.cfg).input_plan_item(ds, tmp / "absent.csv")
        assert full["kind"] == "external" and counts([full]) == (1, 0)
        assert counts([{"role": "in", "kind": "product", "path": tmp / "no.csv", "produced_by": "x"}]) == (0, 1)
        assert counts([{"role": "in", "kind": "product", "path": src, "stale": True}]) == (0, 1)
        # items without kind keep the producer heuristic
        assert counts([{"role": "in", "path": tmp / "no.csv", "produced_by": ".\\runner.ps1 calib"}]) == (0, 1)
        assert counts([{"role": "in", "path": tmp / "no.csv", "produced_by": "a given table"}]) == (1, 0)
        assert counts([{"role": "out", "path": tmp / "no.csv"}]) == (0, 0)
        # stored intermediates: reuse, or rebuild with the reason (never counted as missing)
        meta = tmp / "cache" / "test" / "stage" / "events.provenance.json"
        io_utils.commit_product_set({meta.with_name("events.csv"): lambda p: p.write_text("e\n1\n")},
                                    {"settings": {"fit": "line"}}, meta)
        (n_ext, n_pend), out = _quiet(runner.print_plan, "x", [
            {"role": "cache", "path": meta, "expected": {"settings": {"fit": "line"}}},
            {"role": "cache", "path": meta, "expected": {"settings": {"fit": "low"}}}])
        assert (n_ext, n_pend) == (0, 0)
        assert "[cache] reuse" in out and "[cache] rebuild" in out and "'settings' differs" in out


# check infers the kind of plain plan items
def test_check_infers_the_kind_of_plain_plan_items():
    # Verifier round 3: module plans that still list the beam csv as {'path': ctx.input_csv(ds),
    # 'produced_by': 'make-csv ...'} (no kind) must not report an auto subset or a clean dataset as a missing
    # external input; anything inside the product trees or the cache is a chain product.  A stored subset that
    # the next run would rebuild (source changed) is AUTO with the reason, and MISSING once its source is gone.
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _run_csv(tmp)
        src.parent.mkdir(parents=True)
        _write(src, [_row(e, 1, 0) for e in (1, 2, 3)])
        ctx = _ctx_on(tmp, test=True, with_run=True, n_test_events=2, dry_run=True)
        ds = resolve_datasets(ctx.cfg, ["muon 4"])[0]

        # print the plan; returns (external missing, pending, console text)
        def show(plan):
            (n_ext, n_pend), out = _quiet(runner.print_plan, "x", plan, ctx)
            return n_ext, n_pend, out

        with patch.object(RunContext, "_made_test_csv", return_value=None), \
                patch.object(RunContext, "_current_clean_csv", return_value=None):
            beam = {"role": "in", "path": ctx.input_csv(ds, src), "produced_by": ".\\runner.ps1 make-csv --muon 4"}
            assert beam["path"] == ctx.test_subset_csv(ds, src)
            n_ext, n_pend, out = show([beam])
            assert (n_ext, n_pend) == (0, 0) and "AUTO" in out and "by the first -test run" in out
            assert _quiet(runner.print_plan, "x", [beam])[0] == (1, 0)          # without ctx: old heuristic
            plain = [{"role": "in", "path": ctx.calibration_file(), "produced_by": "a table"},     # product tree
                     {"role": "in", "path": tmp / "data" / "test" / "analysis_files" / "x.csv",
                      "produced_by": ".\\runner.ps1 make-csv --muon 4 -test"},                   # product tree
                     {"role": "in", "path": tmp / "cache" / "test" / "calib" / "h.npz"},          # cache
                     {"role": "in", "path": tmp / "given.csv", "produced_by": "a given table"}]   # external
            n_ext, n_pend, out = show(plain)
            assert (n_ext, n_pend) == (1, 3), out
            # --clean-input: the clean dataset is a product of 'clean'; the beam csv (calib) stays auto
            ctx.clean_input = True
            clean = {"role": "in", "path": ctx.input_csv(ds, src), "produced_by": ".\\runner.ps1 make-csv --muon 4"}
            n_ext, n_pend, out = show([clean, {"role": "in", "path": ctx.beam_csv(ds, src)}])
            assert (n_ext, n_pend) == (0, 1) and "<- .\\runner.ps1 clean --muon 4 -test" in out and "AUTO" in out
            ctx.clean_input = False
            # a stored subset: ok while valid, AUTO with the reason once the next run would rebuild it
            ctx.dry_run = False
            _quiet(ctx.input_csv, ds, src)
            ctx.dry_run = True
            assert show([beam])[2].count("[in ] ok ") == 1
            _write(src, [_row(e, 1, 0) for e in (1, 2, 3, 4)])
            n_ext, n_pend, out = show([beam])
            assert (n_ext, n_pend) == (0, 0) and "AUTO" in out and "'source' differs" in out
            src.unlink()                          # the run folder holds no csv any more: the folder is missing
            n_ext, n_pend, out = show(ctx.input_plan_items(ds))
            assert (n_ext, n_pend) == (1, 0) and "MISSING" in out and "Muon_4GeV" in out


# copy of config.yml with every product tree under tmp
def _tmp_config(tmp: Path) -> Path:
    cfg_file = tmp / "config.yml"
    text = (REPO_ROOT / "configure" / "config.yml").read_text(encoding="utf-8")
    for key, sub in (("data_dir: data ", "data"), ("calibration_dir: calibration ", "cal"),
                     ("results_dir: results ", "res"), ("cache_dir: cache ", "cache")):
        assert key in text, key
        text = text.replace(key, f"{key.split(':')[0]}: {(tmp / sub).as_posix()} ")
    cfg_file.write_text(text, encoding="utf-8")
    return cfg_file


# runner clean input refusals and all
def test_runner_clean_input_refusals_and_all():
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        cfg = str(_tmp_config(Path(tmp)))
        import importlib

        calib_mod = importlib.import_module(runner.STAGES["calib"])
        refused = ["clean", "make-csv"] + ([] if getattr(calib_mod, "SUPPORTS_CLEAN_INPUT", False) else ["calib"])
        for stage in refused:
            _raises(SystemExit, runner.main, [stage, "-clean-input", "--config", cfg],
                    contains=f"'{stage}' does not accept --clean-input")
        _raises(SystemExit, runner.main, ["make-csv", "-from-saved", "--config", cfg],
                contains="'make-csv' does not accept --from-saved")
        ctx = RunContext(cfg=load_config(cfg), clean_input=True)
        with runner.clean_input_for("calib", ctx, direct=False):
            assert ctx.clean_input is False          # inside "all": calib runs on the beam csv
        assert ctx.clean_input is True
        with runner.clean_input_for("ci", ctx, direct=True):
            assert ctx.clean_input is True
        parser = runner.build_parser()
        args = parser.parse_args(["all", "-clean-input", "-test", "-from-saved"])
        choices = next(a for a in parser._actions if a.dest == "stage").choices
        ns = runner.stage_defaults(choices["ci"], args)
        assert ns.clean_input is True and ns.test is True and ns.from_saved is True
        assert runner.make_context(args).clean_input is True
        assert parser.parse_args(["ci", "--from-saved"]).from_saved is True


# stage module for runner tests: records (clean_input, from_saved); fail_in leaves temps
def _stub_stage(name: str, seen: dict, fail_in: Path | None = None, **flags):
    # A stage module for runner tests: records (clean_input, from_saved); with ``fail_in`` it leaves temp files
    # of the io_utils writers there and raises.
    import types

    mod = types.ModuleType(f"_stub_stage_{name.replace('-', '_')}")
    mod.__dict__.update(flags)
    mod.add_arguments = lambda p: None
    mod.io_plan = lambda ctx, args, datasets: []

    # run of the stub stage
    def run(ctx, args, datasets):
        seen[name] = (ctx.clean_input, getattr(args, "from_saved", False))
        if fail_in is not None:
            writer = io_utils.ChunkedCsvWriter(fail_in / "events.csv")      # abandoned, never closed
            writer.write(pd.DataFrame({"a": [1]}))
            ps = io_utils.ProductSet(fail_in / "set.json")                    # staged, never committed
            ps.stage(fail_in / "table.csv").write_text("partial")
            raise RuntimeError("stage failed")

    mod.run = run
    sys.modules[mod.__name__] = mod
    return mod


# runner stage wrappers
def test_runner_stage_wrappers():
    # Every stage runs inside a temp registry (verifier round 3); -from-saved only for stages that declare
    # SUPPORTS_FROM_SAVED (never make-csv); -clean-input for calib only when it declares SUPPORTS_CLEAN_INPUT
    # (Lei D8: calib reads the clean subset for the MPV, the pedestal from the beam csv).
    import argparse

    import runner

    # runner arguments without run flags
    def args(**kw):
        return argparse.Namespace(run_names=[], datasets=None, from_saved=False, **kw)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx_on(tmp, test=False)
        seen: dict = {}
        out_dir = tmp / "out"
        stubs = {"fake": _stub_stage("fake", seen, fail_in=out_dir),
                 "plain": _stub_stage("plain", seen),
                 "saver": _stub_stage("saver", seen, SUPPORTS_FROM_SAVED=True),
                 "make-csv": _stub_stage("make-csv", seen, SUPPORTS_FROM_SAVED=True),
                 "calib": _stub_stage("calib", seen),
                 "ci": _stub_stage("ci", seen)}
        with patch.dict(runner.STAGES, {k: m.__name__ for k, m in stubs.items()}):
            # a failing stage leaves no temp file of the io_utils writers behind
            _raises(RuntimeError, runner.run_stage, "fake", args(), ctx)
            assert out_dir.exists() and list(out_dir.iterdir()) == [] and io_utils.active_registry() is None
            # --from-saved
            a = args()
            a.from_saved = True
            _raises(SystemExit, runner.run_stage, "plain", a, ctx, contains="'plain' does not accept --from-saved")
            _raises(SystemExit, runner.run_stage, "make-csv", a, ctx, contains="converts ROOT files")
            _quiet(runner.run_stage, "saver", a, ctx)
            assert seen["saver"] == (False, True)
            _, out = _quiet(runner.run_stage, "plain", a, ctx, False)          # inside "all": runs normally
            assert seen["plain"] == (False, False) and "runs from its inputs" in out
            # --clean-input
            ctx.clean_input = True
            _raises(SystemExit, runner.run_stage, "calib", args(), ctx, contains="pedestal needs the no-signal rows")
            _quiet(runner.run_stage, "ci", args(), ctx)
            assert seen["ci"] == (True, False)
            stubs["calib"].SUPPORTS_CLEAN_INPUT = True                        # calib fits the MPV on the clean set
            _quiet(runner.run_stage, "calib", args(), ctx)
            assert seen["calib"] == (True, False)
            _quiet(runner.run_stage, "calib", args(), ctx, False)             # inside "all": before clean exists
            assert seen["calib"] == (False, False) and ctx.clean_input is True
            stubs["ci"].SUPPORTS_CLEAN_INPUT = False                          # a stage may opt out
            _raises(SystemExit, runner.run_stage, "ci", args(), ctx, contains="'ci' does not accept --clean-input")
            _quiet(runner.run_stage, "ci", args(), ctx, False)
            assert seen["ci"] == (False, False)
        for m in stubs.values():
            sys.modules.pop(m.__name__, None)


# config.yml (+ local.yml) with every product tree under tmp; None if local.yml moves one
def _config_with_products_in(tmp: Path) -> Path | None:
    # configure/config.yml (+ this machine's local.yml) with every PRODUCT tree redirected to ``tmp``; the
    # inputs stay where this machine has them.  None if local.yml relocates a product tree itself.
    import shutil

    text = (REPO_ROOT / "configure" / "config.yml").read_text(encoding="utf-8")
    for key, sub in (("generated_data_dir: ${data_dir} ", "data"), ("calibration_dir: calibration ", "cal"),
                     ("results_dir: results ", "res"), ("cache_dir: cache ", "cache")):
        assert key in text, key
        text = text.replace(key, f"{key.split(':')[0]}: {(tmp / sub).as_posix()} ")
    cfg_file = tmp / "config.yml"
    cfg_file.write_text(text, encoding="utf-8")
    local = REPO_ROOT / "configure" / "local.yml"
    if local.exists():
        shutil.copy(local, tmp / "local.yml")
    paths = load_config(cfg_file)["paths"]
    if any(tmp not in Path(paths[k]).parents for k in ("generated_data_dir", "calibration_dir", "results_dir",
                                                         "cache_dir")):
        return None
    return cfg_file


# check with empty product trees
def test_check_with_empty_product_trees():
    # Verifier round 3 (Codex INF2-OPT end to end): with empty data / calibration / results / cache trees (e.g.
    # after deleting cache/), check -test and check -clean-input give the same exit code as check: test subsets
    # are AUTO, clean datasets PENDING (made by 'clean'), never missing external inputs.  check writes nothing.
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_file = _config_with_products_in(tmp)
        if cfg_file is None:
            raise unittest.SkipTest("configure/local.yml relocates a product tree")
        codes, outs = {}, {}
        for opts in ((), ("-test",), ("-clean-input",), ("-test", "-clean-input")):
            with redirect_stdout(io.StringIO()) as out:
                codes[opts] = runner.main(["check", *opts, "--config", str(cfg_file)])
            outs[opts] = out.getvalue()
        assert len(set(codes.values())) == 1, codes
        for opts, text in outs.items():   # a product is never missing; a test subset only when its run csv is absent
            bad = [line for line in text.splitlines() if "MISSING" in line and str(tmp) in line
                   and not (", which is absent" in line
                            and not Path(line.split("<- built from ", 1)[-1].rsplit(", which", 1)[0]).exists())]
            assert not bad, (opts, bad[:3])
        cfg = load_config(cfg_file)
        sources = [c for ds in resolve_datasets(cfg, ["all"]) for c in RunContext(cfg=cfg).run_csvs(ds)]
        if any(p.exists() for p in sources):
            assert "[in ] AUTO" in outs[("-test",)]
        assert "<- .\\runner.ps1 clean " in outs[("-clean-input",)]
        assert "-test" in outs[("-test", "-clean-input")].split("<- .\\runner.ps1 clean ", 1)[1].splitlines()[0]
        assert sorted(p.name for p in tmp.iterdir()) == sorted(["config.yml"] + (["local.yml"] if (
            REPO_ROOT / "configure" / "local.yml").exists() else []))   # check creates nothing


# runner rejects bad run options
def test_runner_rejects_bad_run_options():
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = _tmp_config(Path(tmp))
        for argv in (["--test-events", "0"], ["--workers", "-2"]):
            _raises(io_utils.SettingsError, runner.main, ["check", "--config", str(cfg_file), *argv],
                    contains="command line")
        cfg_file.write_text(cfg_file.read_text(encoding="utf-8").replace("chunksize: 2000000", "chunksize: -5"),
                            encoding="utf-8")
        _raises(io_utils.SettingsError, runner.main, ["check", "--config", str(cfg_file)], contains="chunksize")


# runner refuses run names and selects runs with particle flags
def test_runner_refuses_run_names_and_selects_runs_with_particle_flags():
    # Runs are chosen only with --electron E / --muon E / --pion E: a run name (muon_4GeV, "electron 0.3",
    # all, electron) stops with the hint to the flags; the flags pick the runs
    # of configure/datasets.yml (the flag alone = every run of that particle, an energy belongs to its flag).  A
    # stage without flags runs on its DEFAULT_DATASETS, written as run references.
    import types

    import runner

    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = str(_tmp_config(Path(tmp)))
        for argv in (["ci", "muon_4GeV"], ["make-csv", "electron 0.3", "-test"], ["all", "electron"],
                     ["check", "all"], ["clean", "pion_3GeV", "--muon"]):
            err = io.StringIO()
            with patch("sys.stderr", err):
                exc = _raises(SystemExit, runner.main, [*argv, "--config", cfg_file])
            assert exc.code == 2, argv
            assert "are not accepted: choose runs with --electron E, --muon E, --pion E" in err.getvalue(), argv
        parser = runner.build_parser()
        ctx = runner.make_context(parser.parse_args(["ci", "--config", cfg_file]))
        stage = types.SimpleNamespace(DEFAULT_DATASETS=["all"], __name__="stage")

        # run keys the runner selects for these ci flags
        def keys(*flags, mod=stage):
            return [d.key for d in runner.datasets_for(ctx, mod, parser.parse_args(["ci", *flags]))]

        assert keys("--muon", "4") == ["muon_4GeV"] and keys("-pion", "3") == ["pion_3GeV"]
        assert keys("--electron", "1", "0.3", "--pion", "4") == ["electron_1GeV", "electron_0.3GeV", "pion_4GeV"]
        assert keys("--electron", "4", "--muon", "4") == ["electron_4GeV", "muon_4GeV"]
        assert set(keys("--electron")) == ELECTRON_KEYS
        assert keys() == [d.key for d in resolve_datasets(ctx.cfg, ["all"])]
        _raises(SystemExit, keys, "--muon", "3",
                contains="--muon 3: there is no run 'muon 3'; runs of configure/datasets.yml: electron 0.3")
        # stage defaults are run references; any run is accepted (no particle restriction of a stage)
        refs = types.SimpleNamespace(DEFAULT_DATASETS=["muon 4", "Electron 0.3", "pion 500MeV"], __name__="x")
        assert keys(mod=refs) == ["muon_4GeV", "electron_0.3GeV", "pion_0.5GeV"]
        assert keys("--pion", "3", mod=types.SimpleNamespace(DEFAULT_DATASETS=["electron 4"])) == ["pion_3GeV"]
        bad = types.SimpleNamespace(DEFAULT_DATASETS=["muon 9"], __name__="some_stage")
        _raises(SystemExit, keys, mod=bad, contains="DEFAULT_DATASETS of some_stage: there is no run 'muon 9'")


# the READMEs, each in the folder it describes (Lei 2026-10-20); the runner docstring names them
README_FILES = ("README.md", "src/README.md", "configure/README.md", "logs/README.md", "data/README.md",
                "calibration/README.md", "results/README.md")


# runner help names the readmes
def test_runner_help_names_the_readmes():
    # 'runner -h' is the short runner docstring: the stages in chain order and the READMEs (data_analysis/README.md,
    # src/README.md, ...), which hold the long help (commands, options, outputs).
    import runner

    parser = runner.build_parser()
    assert parser.description == runner.__doc__
    text = " ".join(runner.__doc__.split())
    for name in ("README.md", "src/README.md", "configure/README.md", "logs/README.md"):
        assert name in text, name
    assert ", ".join(runner.STAGES) in text, text   # the stages in chain order
    for name in README_FILES:   # every README the runner names exists
        assert (REPO_ROOT / name).exists(), name


# README command tokens -> argv: the [ ] of the optional parts dropped, the range brackets (-fit[5,10]) kept
def _readme_argv(tokens: list[str]) -> list[str]:
    argv = []
    for token in tokens:
        while token.startswith("["):
            token = token[1:]
        while token.endswith("]") and token.count("]") > token.count("["):
            token = token[:-1]
        argv.append(token)
    return argv


# readme commands parse through the runner
def test_readme_commands_parse_through_the_runner():
    # The full command of every stage and of setup / check / all in src/README.md (every optional part in [ ]
    # given) parses through the runner as PowerShell passes it: every stage, no stray token left as a run name.
    import runner
    from ldmx_hcal.common.fitting import normalize_range_args

    commands = {}
    for name in ("src/README.md",):
        for line in (REPO_ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.startswith(".\\runner.ps1 ") and "<" not in line:
                argv = _readme_argv(line.split()[1:])
                commands.setdefault(argv[0], []).append(argv)
    assert set(commands) == {*runner.STAGES, "setup", "check", "all"}, sorted(commands)
    parser = runner.build_parser()
    assert not runner.BROKEN_STAGES, runner.BROKEN_STAGES
    for stage, found in commands.items():
        for argv in found:
            err = io.StringIO()
            with patch("sys.stderr", err):
                try:
                    args = parser.parse_args(normalize_range_args(argv))
                except SystemExit:
                    raise AssertionError(f"{stage}: the README command does not parse: {argv}\n"
                                         f"{err.getvalue()}") from None
            assert args.stage == stage and not getattr(args, "run_names", None), (argv, args)


# INF-05: missing external inputs -> exit 2; the report is printed either way.
def test_check_exit_codes():
    import runner

    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = _tmp_config(Path(tmp))
        with redirect_stdout(io.StringIO()) as out:
            code = runner.main(["check", "--config", str(cfg_file)])
        assert code == runner.EXIT_EXTERNAL_MISSING, (code, out.getvalue()[-2000:])
        assert "MISSING" in out.getvalue()
        assert not (Path(tmp) / "cache").exists() and not (Path(tmp) / "data").exists()  # check writes nothing


# ----------------------------------------------------------------------------------------------
# the test runner itself
# ----------------------------------------------------------------------------------------------
_FAKE_TESTS = """
import os, unittest
# passes when run_all set the Agg backend
def test_a():
    assert os.environ["MPLBACKEND"] == "Agg"
# fails
def test_b():
    raise AssertionError("expected failure")
# skipped
def test_c():
    raise unittest.SkipTest("no data")
"""


# run all isolates modules
def test_run_all_isolates_modules():
    import run_all

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "test_fake.py"
        fake.write_text(_FAKE_TESTS, encoding="utf-8")
        counts, out = _quiet(run_all.run_isolated, fake, None)
        assert counts == (1, 1, 1), out
        assert "PASS  test_fake.test_a" in out and "expected failure" in out
        crash = Path(tmp) / "test_crash.py"
        crash.write_text("import os\ndef test_x():\n    os._exit(3)\n", encoding="utf-8")
        counts, out = _quiet(run_all.run_isolated, crash, None)
        assert counts == (0, 1, 0) and "without a result" in out


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["common"]))
