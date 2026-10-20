"""Stage clean: keep the clean bars of every beam csv of a run, one clean csv per beam csv (read with -clean-input).

E_MIP: energy of one bar end in MIPs, (adc_sum - pedestal) / mpv; E_MeV = E_MIP x mip_energy_mev.
hit: a bar end with adc_sum >= the threshold of the calibration table (1.2 x pedestal, or pedestal + 5 sigma).
Output units (-only): clean (data/[test/]clean_dataset/<run>/<csv stem>_clean.csv), cutflow, layers, diagnostics
(results/<mode>/clean_dataset/<run>/<csv stem>_clean_cutflow.csv, _clean_layers.csv, _clean_diagnostics.pdf).

Variables of <csv stem>_clean.csv (one row per clean bar): the beam csv columns, same order (rows copied as they are)
Variables of <csv stem>_clean_cutflow.csv (one row per selection step, column order): step, rows, events, definition
Variables of <csv stem>_clean_layers.csv (one row per layer, column order):
  layer, n_clean_bars, n_events, n_hit_end0, n_hit_end1, energy_mev_end0_mean, energy_mev_end1_mean, energy_mev_bar_mean
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..common import fitting
from ..common import geometry as geo
from ..common import outputs
from ..common.calibration_io import calibration_lookup, load_calibration
from ..common.config import MODE_FULLSET, REPO_ROOT, RUN_PARTICLES, mode_parts, run_flags, run_ref
from ..common.io_utils import (TEST_SUBSET_BUILDER, InputError, ProductSet, ProductSetError, SettingsError,
                               canonical_path, content_hash, ensure_writable, file_signature, iter_csv,
                               json_records_on, log, product_set_valid, python_info, read_json, require,
                               require_product_set, settings_hash, subset_status, timed, validate_channel_keys,
                               validate_settings)

HELP = ("clean dataset: one csv of clean bars per beam csv (-muon-standard / -otherparticle, -pedestalscale / "
        "-pedestalsigma) that later stages read with -clean-input")
DEFAULT_DATASETS = ["all"]             # every data run of configure/datasets.yml, whatever its particle
CONFIG_SECTION = "clean_dataset"
STAGE_DIR = "clean_dataset"            # data/[test/]clean_dataset, results/<mode>/clean_dataset, cache/[test/]...
SUPPORTS_FROM_SAVED = True             # --from-saved: the result units from the kept-bar cache in cache/[test/]...
SUPPORTS_CLEAN_INPUT = False           # it makes the clean dataset from the beam csv (the runner refuses it too)
ALGORITHM = "clean-d8-v1"              # bump when the selection or the output definition changes
PROVENANCE_VERSION = 5                 # bump when the records change

# output units (-only): one file each; the results json is the record of the result units written
UNITS = ("clean", "cutflow", "layers", "diagnostics")
UNIT_GROUPS = {"tables": ("cutflow", "layers"), "results": ("cutflow", "layers", "diagnostics")}
RESULT_UNITS = ("cutflow", "layers", "diagnostics")   # results/<mode>/clean_dataset/<run>/ (output plan)

DEFAULTS = {
    # ---- selection (part of the identity of the clean dataset) ----
    "standard_by_particle": {"muon": "muon", "electron": "otherparticle", "pion": "otherparticle"},
    "threshold_mode": "pedestalscale",         # pedestalscale (1.2 x pedestal) | pedestalsigma (pedestal + 5 sigma)
    "mip_energy_mev": 4.83143,                 # energy of one MIP in a bar [MeV] (open question: 4.83143 vs 4.83148)
    "layer1_energy_window_mev": [1.84, 7.83],  # muon: both ends of the layer-1 bar inside, inclusive [MeV]
    # ---- diagnostics only (--from-saved redraws with new values) ----
    "mip_spectrum_bins": [0.0, 8.0, 200],      # per-end E_MIP histogram (low, high, number of bins)
    "mip_fit_range": [0.7, 1.3],               # Gauss / Landau fit range on the per-end E_MIP spectra [MIP]
    "diagnostics_min_bin_count": 1,            # spectrum bins with fewer entries are not drawn / fitted
    "event_energy_bin_mev": 5.0,               # bin width of the per-event energy histogram [MeV]
}

STANDARDS = {
    "muon": "bar kept when both ends are hits; event kept when every layer has at most one kept bar and the "
            "layer-1 kept bar has both end energies in layer1_energy_window_mev (inclusive)",
    "otherparticle": "bar kept when either end is a hit; event kept when it has at least one kept bar",
}
THRESHOLD_COLUMNS = {"pedestalscale": "pedestal_range_with_20percent",   # 1.2 x pedestal
                     "pedestalsigma": "pedestal_range_with_5sigma"}      # pedestal + 5 sigma
THRESHOLD_TEXT = {"pedestalscale": "1.2 x pedestal", "pedestalsigma": "pedestal + 5 sigma"}
HIT_DEFINITION = "adc_sum >= threshold with a finite pedestal, a finite threshold and mpv > 0 (ci end hit)"
ENERGY_DEFINITION = "E_MIP = (adc_sum - pedestal) / mpv; E_MeV = E_MIP * mip_energy_mev"

BEAM_COLUMNS = ["pf_event", "layer", "strip", "adc_sum_end0", "adc_sum_end1"]
CACHE_ARRAYS = ("pf_event", "layer", "strip", "hit_end0", "hit_end1", "e_mip_end0", "e_mip_end1", "in_clean")
COPY_BLOCK_BYTES = 16 * 1024 ** 2       # pass 2 reads the beam csv in blocks of this size


# settings rule of mip_spectrum_bins: [low, high, number of bins]
def _bins_rule(value):
    ok = (isinstance(value, (list, tuple)) and len(value) == 3
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in value)
          and float(value[0]) < float(value[1]) and float(value[2]).is_integer() and value[2] > 0)
    return None if ok else "[low, high, number of bins] with low < high and a positive integer number of bins"


# settings rule of standard_by_particle: full particle name -> muon / otherparticle
def _standard_map_rule(value):
    if not isinstance(value, dict) or not value:
        return f"a mapping particle -> one of {list(STANDARDS)}"
    keys = [k for k in value if k not in RUN_PARTICLES]   # never ignored silently (e.g. an abbreviated name)
    if keys:
        return f"a mapping particle -> standard keyed by full particle names {list(RUN_PARTICLES)} (got {keys})"
    bad = {k: v for k, v in value.items() if v not in STANDARDS}
    return f"a mapping particle -> one of {list(STANDARDS)} (got {bad})" if bad else None


SETTINGS_RULES = {
    "standard_by_particle": _standard_map_rule,
    "threshold_mode": ("choice", list(THRESHOLD_COLUMNS)),
    "mip_energy_mev": "float>0",
    "layer1_energy_window_mev": "window",
    "mip_spectrum_bins": _bins_rule,
    "mip_fit_range": "window",
    "diagnostics_min_bin_count": "int>=0",
    "event_energy_bin_mev": "float>0",
}


# --------------------------------------------------------------------------
# settings and jobs
# --------------------------------------------------------------------------
# clean settings: DEFAULTS < yml section clean_dataset, checked; also returns the unknown keys
def load_settings(ctx) -> tuple[dict, list[str]]:
    settings = copy.deepcopy(DEFAULTS)
    unknown = []
    for key, value in (ctx.stage_cfg(CONFIG_SECTION) or {}).items():
        if key not in DEFAULTS:
            unknown.append(key)
        elif isinstance(value, dict) and isinstance(settings[key], dict):
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    validate_settings(settings, SETTINGS_RULES, source=f"configure/config.yml {CONFIG_SECTION}")
    return settings, unknown


# numpy scalar or number -> float
def _number(value) -> float:
    return float(value.item() if isinstance(value, np.generic) else value)


STANDARD_FLAGS = {"muon": "muon-standard", "otherparticle": "otherparticle"}  # standard -> its flag


# what decides the rows of the clean dataset (window and MeV constant: muon standard only)
def selection_definition(standard: str, threshold_mode: str, settings: dict) -> dict:
    out = {"standard": standard, "threshold_mode": threshold_mode,
           "threshold_column": THRESHOLD_COLUMNS[threshold_mode]}
    if standard == "muon":
        out["mip_energy_mev"] = _number(settings["mip_energy_mev"])
        out["layer1_energy_window_mev"] = [_number(v) for v in settings["layer1_energy_window_mev"]]
    return out


# one run of a clean call: its standard, threshold mode, their sources and the selection
@dataclass
class Job:
    ds: object
    standard: str
    standard_source: str
    threshold_mode: str
    threshold_source: str
    selection: dict
    standard_flag: bool = False       # the standard was given with -muon-standard / -otherparticle
    threshold_flag: bool = False      # the threshold mode was given with -pedestalscale / -pedestalsigma

    # calibration column of the hit threshold
    @property
    def threshold_column(self) -> str:
        return THRESHOLD_COLUMNS[self.threshold_mode]

    # selection flags of the call (None = not given), stored in the provenance
    @property
    def flags(self) -> dict:
        """The currentness check rebuilds the selection from these flags and the configuration of now."""
        return {"standard": self.standard if self.standard_flag else None,
                "threshold_mode": self.threshold_mode if self.threshold_flag else None}


# one Job per run: a flag wins, else standard_by_particle, else otherparticle (no run refused)
def make_jobs(settings: dict, args, datasets) -> list[Job]:
    flag_mode = getattr(args, "threshold_mode", None)
    mode = flag_mode or settings["threshold_mode"]
    mode_source = f"-{flag_mode} flag" if flag_mode else "config clean_dataset.threshold_mode"
    flag_standard = getattr(args, "standard", None)
    by_particle = settings["standard_by_particle"] or {}
    jobs = []
    for ds in datasets:
        if flag_standard:
            standard, source = flag_standard, f"-{STANDARD_FLAGS[flag_standard]} flag"
        elif ds.particle in by_particle:
            standard = by_particle[ds.particle]
            source = f"default for {ds.particle} runs, clean_dataset.standard_by_particle"
        else:   # no rule of its own: the general rule (bar kept when either end is a hit)
            standard = "otherparticle"
            source = (f"general rule for {run_ref(ds)}: clean_dataset.standard_by_particle has no {ds.particle} "
                      f"entry")
        jobs.append(Job(ds, standard, source, mode, mode_source, selection_definition(standard, mode, settings),
                        standard_flag=bool(flag_standard), threshold_flag=bool(flag_mode)))
    return jobs


# the Job that clean with these selection flags defines under the configuration of now
def configured_job(ctx, ds, flags: dict | None = None) -> Job:
    settings, _ = load_settings(ctx)
    flags = flags or {}
    args = SimpleNamespace(standard=flags.get("standard"), threshold_mode=flags.get("threshold_mode"))
    return make_jobs(settings, args, [ds])[0]


# clean command that remakes (from_saved: redraws) the products of a job
def rebuild_command(ctx, job: Job, from_saved: bool = False) -> str:
    cmd = [f".\\runner.ps1 clean {run_flags(job.ds)}"]
    if job.standard_flag:
        cmd.append(f"-{STANDARD_FLAGS[job.standard]}")
    if job.threshold_flag:
        cmd.append(f"-{job.threshold_mode}")
    if ctx.test:
        cmd.append("-test")
        if int(ctx.n_test_events) != int((ctx.cfg.get("run") or {}).get("test_events", 5000)):
            cmd.append(f"-test-events {int(ctx.n_test_events)}")
    if from_saved:
        cmd.append("-from-saved")
    return " ".join(cmd)


# 'key: <have> value, <want> value' for every selection key that differs
def _selection_diff(stored: dict, wanted: dict, have: str, want: str) -> str:
    return "; ".join(f"{k}: {have} {_short(stored.get(k), 60)}, {want} {_short(wanted.get(k), 60)}"
                     for k in sorted(set(stored) | set(wanted)) if stored.get(k) != wanted.get(k))


# --------------------------------------------------------------------------
# product paths (one set per input csv, a sub-folder per run)
# --------------------------------------------------------------------------
# csv files of a run (full set, file-name order), or a stop with the make-csv command
def run_csvs(ctx, ds) -> list[Path]:
    csvs = ctx.run_csvs(ds)
    if not csvs:
        raise InputError(f"run {run_ref(ds)}: no csv file in {ctx.run_dir(ds, mode=MODE_FULLSET)}; make them with "
                         f".\\runner.ps1 make-csv {run_flags(ds)}")
    return csvs


# path of the clean dataset of one csv: data/[test/]clean_dataset/<run>/<csv stem>_clean.csv
def clean_csv(ctx, ds, csv, mode: str | None = None) -> Path:
    return Path(ctx.clean_product(ds, csv, mode=mode))


# provenance path next to the clean dataset: <csv stem>_clean.provenance.json
def provenance_json(ctx, ds, csv, mode: str | None = None) -> Path:
    path = clean_csv(ctx, ds, csv, mode)
    return path.with_name(f"{path.stem}.provenance.json")


# folder of the final results: results/<mode>/clean_dataset/
def results_dir(ctx, mode: str | None = None) -> Path:
    return ctx.results_dir(STAGE_DIR, mode=mode)


# default path of the cut-flow csv: results/<mode>/clean_dataset/<run>/<csv stem>_clean_cutflow.csv
def cutflow_csv(ctx, ds, csv) -> Path:
    return ctx.csv_product(results_dir(ctx), ds, csv, "_clean_cutflow.csv")


# default path of the per-layer csv
def layers_csv(ctx, ds, csv) -> Path:
    return ctx.csv_product(results_dir(ctx), ds, csv, "_clean_layers.csv")


# default path of the diagnostics pdf
def diagnostics_pdf(ctx, ds, csv) -> Path:
    return ctx.csv_product(results_dir(ctx), ds, csv, "_clean_diagnostics.pdf")


# default path of the results record (the result units written by the last call)
def results_json(ctx, ds, csv) -> Path:
    return ctx.csv_product(results_dir(ctx), ds, csv, "_clean_results.json")


# kept-bar cache (input of -from-saved): cache/[test/]clean_dataset/<run>/<csv stem>_hit_bars.npz
def cache_npz(ctx, ds, csv) -> Path:
    return ctx.csv_product(ctx.cache_dir(STAGE_DIR), ds, csv, "_hit_bars.npz")


# record of the kept-bar cache
def cache_json(ctx, ds, csv) -> Path:
    return ctx.csv_product(ctx.cache_dir(STAGE_DIR), ds, csv, "_hit_bars.json")


# files a selection run of one csv writes besides its final results (the clean csv only with unit clean)
def _intermediates(ctx, ds, csv, units) -> list[Path]:
    out = [clean_csv(ctx, ds, csv), provenance_json(ctx, ds, csv)] if "clean" in units else []
    return out + [cache_npz(ctx, ds, csv), cache_json(ctx, ds, csv)]


# --------------------------------------------------------------------------
# output units and the output plan of the final results (common/outputs.py)
# --------------------------------------------------------------------------
# (unit, kind, default path, description) of the final results of one csv, in the order a run writes them
RESULT_PARTS = (
    ("cutflow", "csv", cutflow_csv, "clean cut flow: rows / events after every step"),
    ("layers", "csv", layers_csv, "per layer: clean bars, events, hit ends, mean energies"),
    ("diagnostics", "pdf", diagnostics_pdf, "clean diagnostics: cut flow, per-end E_MIP spectra with fits, maps"),
    ("results", "json", results_json, "record of the result units above written by this call"),
)


# output units of this call: -only, without the pdf (-no-plots) and the clean csv (-from-saved); refused when empty
def call_units(ctx, args) -> tuple:
    chosen = [u for u in UNITS if ctx.wants(u)]
    from_saved = getattr(args, "from_saved", False)
    units = tuple(u for u in chosen if not (u == "diagnostics" and not ctx.make_plots)
                  and not (u == "clean" and from_saved))
    if not units:
        why = []
        if "diagnostics" in chosen:
            why.append("the diagnostics pdf needs plots (no -no-plots)")
        if "clean" in chosen:
            why.append("-from-saved cannot make the clean csv (pass 2 copies lines of the beam csv)")
        raise SettingsError(f"clean: -only {', '.join(chosen)} makes nothing in this call: {'; '.join(why)}")
    return units


# output-plan key of one final result: <run>.<csv stem>.<unit>
def result_key(ds, csv, unit: str) -> str:
    return f"{Path(ds.folder).name}.{Path(csv).stem}.{unit}"


# final results of the call in the -output order: per csv its chosen result units, then their record json
def result_slots(ctx, jobs, units) -> list:
    slots = []
    for job in jobs:
        for csv in ctx.run_csvs(job.ds):                 # a run without csv: no slot (the run stops on it)
            parts = [p for p in RESULT_PARTS if p[0] in units]
            if parts:
                parts.append(RESULT_PARTS[-1])
            for unit, kind, default, what in parts:
                slots.append(outputs.ResultSlot(result_key(job.ds, csv, unit), kind, default(ctx, job.ds, csv),
                                                f"{ctx.event_source(job.ds, csv)}: {what}"))
    return slots


# output plan of a call: -output names, test names or the unsaved scratch copy
def output_plan(ctx, args, jobs, units, preview: bool = False) -> outputs.OutputPlan:
    """preview=True: same destinations, the scratch folder is not emptied and the plan not registered."""
    try:
        plan = outputs.OutputPlan(ctx, STAGE_DIR, result_slots(ctx, jobs, units), preview=preview)
    except ValueError as exc:   # two results would be written to one file
        raise SettingsError(f"clean: {exc}") from None
    dest = [canonical_path(plan.path(s.key)) for s in plan.slots]
    twice = sorted({Path(d).name for d in dest if dest.count(d) > 1})
    if twice:   # the scratch copy keeps only the file name: two runs with csv files of the same name meet there
        raise SettingsError(f"clean: two results of this call would be written to one file ({', '.join(twice)}); "
                            f"runs of this call hold csv files of the same name: save them with --output (the "
                            f"default names keep the run folder) or run them one at a time")
    return plan


# does the plan hold this result?
def _has_slot(plan, key: str) -> bool:
    return any(s.key == key for s in plan.slots)


# log where the final results went (or that they were not saved)
def _log_final_results(plan) -> None:
    unsaved = [s for s in plan.slots if not plan.saved(s.key)]
    if unsaved:
        log(f"final results not saved (no --output): scratch copies in {plan.path(unsaved[0].key).parent}; use "
            f"--output [names] to save them")
    lines = [line for s, line in zip(plan.slots, plan.summary_lines()) if plan.saved(s.key)]
    if lines:
        log("final results:")
        for line in lines:
            log(f"  {line}")


# --------------------------------------------------------------------------
# identities
# --------------------------------------------------------------------------
# identity of a file: path, size, content hash (+ mtime when only the edges are hashed)
def file_identity(path: Path) -> dict:
    """Files >= 50 MiB hash only the first + last 4 MiB, so the mtime is part of their identity."""
    path = Path(path)
    st = path.stat()
    digest, method = content_hash(path)
    ident = {"kind": "file", "path": canonical_path(path), "size": int(st.st_size), "sha256": digest,
             "hash": method}
    if method != "full":
        ident["mtime_ns"] = int(st.st_mtime_ns)
    return ident


# identity of the beam csv this mode reads for one csv of a run, nothing built
def beam_identity(ctx, ds, csv) -> dict:
    """A test subset is identified by what it is copied from, so a rebuilt subset keeps it."""
    item = ctx.beam_plan_item(ds, csv)
    path = Path(item["path"])
    if item.get("kind") == "auto":
        source = Path(item["source"])
        if not source.exists():
            raise InputError(f"the full csv {source} of the test subset {path.name} ({run_ref(ds)}) is missing")
        return {"kind": "test-subset", "path": canonical_path(path), "builder": TEST_SUBSET_BUILDER,
                "source": file_signature(source), "n_events_requested": int(item["n_events"])}
    if not path.exists():
        raise InputError(f"the beam csv {ctx.event_source(ds, csv)} of {run_ref(ds)} ({path}) is missing")
    return file_identity(path)


# sha256 of a whole file
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# value as it reads back from json
def _as_json(value):
    return json.loads(json.dumps(value, sort_keys=True, default=lambda v: v.item() if isinstance(v, np.generic)
                                 else str(v)))


# value as short json text, for messages
def _short(value, limit: int = 140) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
# pedestal, mpv and threshold of one table as (20, 12, 2) arrays [layer, strip, end]
@dataclass
class ChannelCalibration:
    path: Path
    threshold_column: str
    pedestal: np.ndarray
    mpv: np.ndarray
    threshold: np.ndarray
    sha256: str
    values_sha256: str
    mpv_status_counts: dict


CALIBRATION_VALUES_DIGEST = "clean-calibration-values-1"   # bump if the digest below changes


# sha256 of the pedestal, mpv and threshold values as read (not of the file text)
def calibration_values_digest(pedestal: np.ndarray, mpv: np.ndarray, threshold: np.ndarray,
                              threshold_column: str) -> str:
    """Another row order gives the same digest; another number format may not (stale)."""
    h = hashlib.sha256(f"{CALIBRATION_VALUES_DIGEST};{threshold_column}".encode("utf-8"))
    for values in (pedestal, mpv, threshold):
        a = np.asarray(values, dtype="<f8") + 0.0             # -0.0 -> 0.0
        a = np.where(np.isnan(a), np.nan, a)                  # one NaN bit pattern
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# read pedestal, mpv and the hit-threshold column of a calibration table
def read_calibration(path, threshold_column: str, check: bool = False) -> ChannelCalibration:
    """The threshold column must be in the table (never recomputed); check=True runs load_calibration."""
    path = Path(path)
    if threshold_column not in THRESHOLD_COLUMNS.values():
        raise SettingsError(f"unknown hit-threshold column '{threshold_column}' (known: "
                            f"{list(THRESHOLD_COLUMNS.values())})")
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise InputError(f"calibration table {path} cannot be read: {exc}") from exc
    df = validate_channel_keys(df, source=f"calibration table {path}")
    missing = [c for c in ("pedestal", "mpv", threshold_column) if c not in df.columns]
    if missing:
        raise InputError(f"calibration table {path} has no column(s) {missing}; the clean selection reads "
                         f"pedestal, mpv and {threshold_column}. Regenerate the table with .\\runner.ps1 calib")
    if check:
        load_calibration(path)
    pedestal, mpv = calibration_lookup(df, "pedestal"), calibration_lookup(df, "mpv")
    threshold = calibration_lookup(df, threshold_column)
    status = {}
    if "mpv_status" in df.columns:
        finite = np.isfinite(pd.to_numeric(df["mpv"], errors="coerce").to_numpy(float))
        status = {str(k): int(v) for k, v in df.loc[finite, "mpv_status"].fillna("").astype(str)
                  .value_counts().sort_index().items()}
    return ChannelCalibration(path, threshold_column, pedestal, mpv, threshold, _sha256(path),
                              calibration_values_digest(pedestal, mpv, threshold, threshold_column), status)


# warn about channels with a finite mpv but an unusual mpv_status
def _warn_calibration_status(calib: ChannelCalibration) -> None:
    unusual = {k: v for k, v in calib.mpv_status_counts.items() if k not in ("OK", "")}
    if unusual:
        log(f"WARNING: the calibration table has channels with a finite mpv but status {unusual}; "
            f"they are used as they are (as in the ci stage)")


# where the calibration table comes from (for the log and the provenance)
def calibration_source(ctx, path) -> str:
    if canonical_path(path) == canonical_path(ctx.calibration_product()):
        return "/".join(("calibration", *mode_parts(ctx.mode), "pedestal_mpv product"))
    return "calibration/pedestal_mpv product (test-mode fallback: no test table)"


# --------------------------------------------------------------------------
# per-end quantities and the selection
# --------------------------------------------------------------------------
# end hit + energy: hit = adc_sum >= threshold, E_MIP = (adc_sum - pedestal) / mpv (based on Joe Muse, revised by Lei)
def end_quantities(adc_sum, pedestal, mpv, threshold) -> tuple[np.ndarray, np.ndarray]:
    """An end without a finite pedestal / threshold or with mpv <= 0 is no hit."""
    adc = np.asarray(adc_sum, dtype=np.float64)
    pedestal, mpv, threshold = (np.asarray(a, dtype=np.float64) for a in (pedestal, mpv, threshold))
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = np.isfinite(pedestal) & np.isfinite(mpv) & (mpv > 0)
        hit = usable & np.isfinite(threshold) & (adc >= threshold)   # hit = adc_sum >= threshold
        e_mip = np.where(usable, (adc - pedestal) / np.where(usable, mpv, 1.0), np.nan)   # (adc - pedestal) / mpv
    return hit, e_mip


# the kept bars of one beam csv (bar rule, before the event cuts), in file order
@dataclass
class HitBars:
    row: np.ndarray          # int64: data-row index in the beam csv (0 = first line after the header)
    pf_event: np.ndarray     # int64
    layer: np.ndarray        # int8
    strip: np.ndarray        # int8
    hit_end0: np.ndarray     # bool
    hit_end1: np.ndarray     # bool
    e_mip_end0: np.ndarray   # float64
    e_mip_end1: np.ndarray   # float64
    in_clean: np.ndarray     # bool: the bar is a row of the clean dataset (set by select_events)

    # number of kept bars
    def __len__(self) -> int:
        return len(self.pf_event)


_BAR_DTYPES = {"row": np.int64, "pf_event": np.int64, "layer": np.int8, "strip": np.int8, "hit_end0": bool,
               "hit_end1": bool, "e_mip_end0": np.float64, "e_mip_end1": np.float64, "in_clean": bool}


# bar rule: muon = hit_end0 & hit_end1, otherparticle = hit_end0 | hit_end1
def bar_kept(hit_end0: np.ndarray, hit_end1: np.ndarray, standard: str) -> np.ndarray:
    return (hit_end0 & hit_end1) if standard == "muon" else (hit_end0 | hit_end1)


# pass 1: stream the beam csv (5 columns) and keep the bars that pass the bar rule
def scan_beam_csv(src: Path, calib: ChannelCalibration, standard: str, chunksize: int) -> tuple[HitBars, dict]:
    """Returns (kept bars with in_clean = False, counts)."""
    parts = {name: [] for name in _BAR_DTYPES if name != "in_clean"}
    n_rows, n_hits = 0, [0, 0]
    events = np.empty(0, dtype=np.int64)
    for chunk in iter_csv(src, BEAM_COLUMNS, chunksize):
        ev = chunk["pf_event"].to_numpy(np.int64)
        layer, strip = chunk["layer"].to_numpy(), chunk["strip"].to_numpy()
        li, si = layer.astype(np.intp), strip.astype(np.intp)
        hits, e_mips = [], []
        for end in (0, 1):
            hit, e_mip = end_quantities(chunk[f"adc_sum_end{end}"].to_numpy(), calib.pedestal[li, si, end],
                                        calib.mpv[li, si, end], calib.threshold[li, si, end])
            hits.append(hit)
            e_mips.append(e_mip)
            n_hits[end] += int(hit.sum())
        idx = np.flatnonzero(bar_kept(hits[0], hits[1], standard))
        parts["row"].append(n_rows + idx)
        parts["pf_event"].append(ev[idx])
        parts["layer"].append(layer[idx].astype(np.int8))
        parts["strip"].append(strip[idx].astype(np.int8))
        for end in (0, 1):
            parts[f"hit_end{end}"].append(hits[end][idx])
            parts[f"e_mip_end{end}"].append(e_mips[end][idx])
        events = np.union1d(events, ev)
        n_rows += len(chunk)
    arrays = {name: (np.concatenate(p).astype(_BAR_DTYPES[name], copy=False) if p
                     else np.empty(0, dtype=_BAR_DTYPES[name])) for name, p in parts.items()}
    bars = HitBars(**arrays, in_clean=np.zeros(len(arrays["row"]), dtype=bool))
    counts = {"input_rows": int(n_rows), "input_events": int(len(events)), "hit_end0": n_hits[0],
              "hit_end1": n_hits[1], "kept_bars": int(len(bars))}
    return bars, counts


# kept bars sorted by (event, layer), with the segment ids computed once
@dataclass
class EventLayerSegments:
    order: np.ndarray        # sorting permutation of the bars
    event_id: np.ndarray     # per sorted bar: index of its event (0..n_events-1)
    n_events: int
    bars_in_layer: np.ndarray  # per sorted bar: kept bars of its (event, layer)


# sort the bars by (event, layer); kept bars per (event, layer)
def event_layer_segments(pf_event: np.ndarray, layer: np.ndarray) -> EventLayerSegments:
    n = len(pf_event)
    if not n:
        z = np.zeros(0, dtype=np.int64)
        return EventLayerSegments(z, z, 0, z)
    order = np.lexsort((layer, pf_event))
    ev, ly = pf_event[order], layer[order]
    new_event = np.ones(n, dtype=bool)
    new_event[1:] = ev[1:] != ev[:-1]
    new_layer = new_event.copy()
    new_layer[1:] |= ly[1:] != ly[:-1]
    event_id = np.cumsum(new_event) - 1
    layer_id = np.cumsum(new_layer) - 1
    return EventLayerSegments(order, event_id, int(event_id[-1]) + 1, np.bincount(layer_id)[layer_id])


# event cuts: muon = <= 1 kept bar per layer, layer-1 bar with both E_MeV in window (based on Joe Muse, revised by Lei)
def select_events(bars: HitBars, standard: str, selection: dict) -> list[dict]:
    """Sets bars.in_clean; returns the cut-flow steps after the input."""
    n = len(bars)
    seg = event_layer_segments(bars.pf_event, bars.layer)
    rule = "both ends hit" if standard == "muon" else "either end hit"
    steps = [dict(step=f"bars_{rule.replace(' ', '_')}", rows=n, events=seg.n_events,
                  definition=f"bars with {rule} ({HIT_DEFINITION})")]
    if standard != "muon":
        bars.in_clean = np.ones(n, dtype=bool)
        return steps
    if not n:
        bars.in_clean = np.zeros(0, dtype=bool)
        return steps + [dict(step="at_most_one_bar_per_layer", rows=0, events=0, definition=""),
                        dict(step="layer1_bar_in_energy_window", rows=0, events=0, definition="")]
    lo, hi = selection["layer1_energy_window_mev"]
    k = float(selection["mip_energy_mev"])
    o = seg.order
    multi = np.bincount(seg.event_id, weights=(seg.bars_in_layer >= 2).astype(np.float64),
                        minlength=seg.n_events) > 0
    e0, e1 = bars.e_mip_end0[o] * k, bars.e_mip_end1[o] * k
    with np.errstate(invalid="ignore"):
        layer1_ok = (bars.layer[o] == 1) & (e0 >= lo) & (e0 <= hi) & (e1 >= lo) & (e1 <= hi)   # both E_MeV in window
    has_layer1 = np.bincount(seg.event_id, weights=layer1_ok.astype(np.float64), minlength=seg.n_events) > 0
    one_per_layer = ~multi
    keep_event = one_per_layer & has_layer1   # muon event cut
    in_clean = np.empty(n, dtype=bool)
    in_clean[o] = keep_event[seg.event_id]
    bars.in_clean = in_clean
    return steps + [
        dict(step="at_most_one_bar_per_layer", rows=int(one_per_layer[seg.event_id].sum()),
             events=int(one_per_layer.sum()),
             definition="events whose layers hold at most one kept bar each (layers without a kept bar allowed)"),
        dict(step="layer1_bar_in_energy_window", rows=int(in_clean.sum()), events=int(keep_event.sum()),
             definition=f"... and whose layer-1 kept bar has E_MeV of each end in [{lo:g}, {hi:g}] MeV "
                        f"(inclusive; {k:g} MeV per MIP)")]


# --------------------------------------------------------------------------
# pass 2: byte-exact copy of the selected rows
# --------------------------------------------------------------------------
# csv field -> int (also '3.0')
def _field_int(value: bytes) -> int:
    try:
        return int(value)
    except ValueError:
        return int(float(value))


# pass 2: copy the header and the selected lines of the beam csv byte for byte
def copy_selected_rows(src: Path, dst: Path, rows: np.ndarray, keys: tuple, n_rows_expected: int) -> dict:
    """Each copied line must hold the (pf_event, layer, strip) of pass 1, else the csv changed."""
    rows = np.asarray(rows, dtype=np.int64)
    exp_event, exp_layer, exp_strip = (np.asarray(k, dtype=np.int64) for k in keys)
    if len(rows) > 1 and np.any(np.diff(rows) <= 0):
        raise ValueError("copy_selected_rows: rows must be strictly increasing")
    written = row = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        header = fin.readline()
        fout.write(header)
        terminator = header[len(header.rstrip(b"\r\n")):] or b"\n"
        names = header.rstrip(b"\r\n").lstrip(b"\xef\xbb\xbf").split(b",")     # a UTF-8 BOM is not a name
        try:
            cols = [names.index(c) for c in (b"pf_event", b"layer", b"strip")]
        except ValueError as exc:
            raise InputError(f"{src}: the header has no pf_event / layer / strip column") from exc
        last = max(cols)

        # write the selected lines of one block, each checked against pass 1
        def emit(lines: list) -> None:
            nonlocal written, row
            if b"" in lines or b"\r" in lines:
                lines = [ln for ln in lines if ln and ln != b"\r"]
            lo, hi = np.searchsorted(rows, [row, row + len(lines)])
            if hi > lo:
                take = []
                for j in range(lo, hi):
                    line = lines[int(rows[j]) - row]
                    fields = line.rstrip(b"\r").split(b",", last + 1)
                    try:
                        found = tuple(_field_int(fields[c]) for c in cols)
                    except (IndexError, ValueError):
                        found = None
                    if found != (exp_event[j], exp_layer[j], exp_strip[j]):
                        raise RuntimeError(f"{src}: data line {int(rows[j]) + 1} is {found}, but the first pass "
                                           f"read {(int(exp_event[j]), int(exp_layer[j]), int(exp_strip[j]))} "
                                           f"there: the file changed while it was read; run again")
                    take.append(line)
                fout.write(b"\n".join(take) + b"\n")
                written += len(take)
            row += len(lines)

        carry = b""
        while True:
            block = fin.read(COPY_BLOCK_BYTES)
            if not block:
                break
            block = carry + block
            cut = block.rfind(b"\n")
            if cut < 0:
                carry = block
                continue
            carry = block[cut + 1:]
            emit(block[:cut].split(b"\n"))
        if carry and carry != b"\r":                 # a last line without a line terminator
            before = written
            emit([carry])
            if written > before:                     # emit() appended "\n": use the header's terminator
                fout.seek(-1, 1)
                fout.truncate()
                fout.write(terminator)
    if row != n_rows_expected:
        raise RuntimeError(f"{src}: {row} data lines now, {n_rows_expected} in the first pass: the file changed "
                           f"while it was read; run again")
    if written != len(rows):
        raise RuntimeError(f"{src}: copied {written} of {len(rows)} selected lines")
    return {"rows": int(written), "line_terminator": "CRLF" if terminator == b"\r\n" else "LF"}


# --------------------------------------------------------------------------
# diagnostics (one summary for the csv and the pdf)
# --------------------------------------------------------------------------
# every quantity of the per-layer csv and the pdf
@dataclass
class Summary:
    layers: pd.DataFrame        # per layer (the layers csv)
    bar_count: np.ndarray       # (19, 12) grid: clean bars per bar
    bar_energy: np.ndarray      # (19, 12) grid: mean bar energy [MeV]
    event_bars: np.ndarray      # per clean event: clean bars
    event_layers: np.ndarray    # per clean event: layers with a clean bar
    event_energy: np.ndarray    # per clean event: sum of the bar energies [MeV]
    multiplicity: np.ndarray    # per (event, layer) with a kept bar, before the event cuts: kept bars
    layer1_e0: np.ndarray       # muon: layer-1 kept bars of events with <= 1 kept bar per layer: E_MeV end 0
    layer1_e1: np.ndarray


# bar energy: mean E_MeV of the hit ends of the bar
def bar_energy_mev(bars: HitBars, k: float) -> np.ndarray:
    e0 = np.where(bars.hit_end0, bars.e_mip_end0 * k, 0.0)
    e1 = np.where(bars.hit_end1, bars.e_mip_end1 * k, 0.0)
    n = bars.hit_end0.astype(np.int64) + bars.hit_end1.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (e0 + e1) / np.maximum(n, 1), np.nan)   # (E0 + E1) / hit ends


# per-layer table, maps and per-event quantities of the clean bars, computed once
def summarize(bars: HitBars, standard: str, k: float) -> Summary:
    c = bars.in_clean
    layer = bars.layer[c].astype(np.int64)
    strip = bars.strip[c].astype(np.int64)
    h0, h1 = bars.hit_end0[c], bars.hit_end1[c]
    e0, e1 = bars.e_mip_end0[c] * k, bars.e_mip_end1[c] * k
    both = h0 & h1
    nl = geo.N_LAYERS + 1

    # sum per layer 1..19
    def per_layer(weights=None):
        return np.bincount(layer, weights=weights, minlength=nl)[1:]

    # total / count, NaN where the count is 0
    def mean(total, count):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    n_bars = per_layer()
    n_h0, n_h1, n_both = per_layer(h0.astype(float)), per_layer(h1.astype(float)), per_layer(both.astype(float))
    seg = event_layer_segments(bars.pf_event[c], bars.layer[c])
    first_in_layer = np.zeros(len(layer), dtype=bool)
    if len(layer):
        ev_s, ly_s = bars.pf_event[c][seg.order], layer[seg.order]
        first = np.ones(len(ly_s), dtype=bool)
        first[1:] = (ev_s[1:] != ev_s[:-1]) | (ly_s[1:] != ly_s[:-1])
        first_in_layer[seg.order] = first
    layers = pd.DataFrame({
        "layer": np.arange(1, nl),
        "n_clean_bars": n_bars.astype(np.int64),
        "n_events": per_layer(first_in_layer.astype(float)).astype(np.int64),
        "n_hit_end0": n_h0.astype(np.int64),
        "n_hit_end1": n_h1.astype(np.int64),
        "energy_mev_end0_mean": mean(per_layer(np.where(h0, e0, 0.0)), n_h0),
        "energy_mev_end1_mean": mean(per_layer(np.where(h1, e1, 0.0)), n_h1),
        "energy_mev_bar_mean": mean(per_layer(np.where(both, 0.5 * (e0 + e1), 0.0)), n_both),
    })
    bar_e = bar_energy_mev(bars, k)[c]
    flat = layer * geo.N_STRIPS_WIDE + strip
    size = nl * geo.N_STRIPS_WIDE
    count = np.bincount(flat, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        energy = np.bincount(flat, weights=bar_e, minlength=size) / np.where(count > 0, count, np.nan)
    allbars = geo.all_bars()
    idx = allbars["layer"].to_numpy() * geo.N_STRIPS_WIDE + allbars["strip"].to_numpy()
    grid_n = geo.fill_layer_grid(allbars["layer"], allbars["strip"], count[idx])
    grid_e = geo.fill_layer_grid(allbars["layer"], allbars["strip"], energy[idx])
    ev_bars = np.bincount(seg.event_id, minlength=seg.n_events) if len(layer) else np.zeros(0, np.int64)
    ev_layers = np.bincount(seg.event_id, weights=first_in_layer[seg.order].astype(float),
                            minlength=seg.n_events).astype(np.int64) if len(layer) else np.zeros(0, np.int64)
    ev_energy = np.bincount(seg.event_id, weights=bar_e[seg.order], minlength=seg.n_events) if len(layer) \
        else np.zeros(0)
    # before the event cuts (all kept bars)
    all_seg = event_layer_segments(bars.pf_event, bars.layer)
    if len(bars):
        ev_s, ly_s = bars.pf_event[all_seg.order], bars.layer[all_seg.order]
        first = np.ones(len(ly_s), dtype=bool)
        first[1:] = (ev_s[1:] != ev_s[:-1]) | (ly_s[1:] != ly_s[:-1])
        multiplicity = all_seg.bars_in_layer[first]
    else:
        multiplicity = np.zeros(0, dtype=np.int64)
    l1_e0 = l1_e1 = np.zeros(0)
    if standard == "muon" and len(bars):
        o = all_seg.order
        multi = np.bincount(all_seg.event_id, weights=(all_seg.bars_in_layer >= 2).astype(float),
                            minlength=all_seg.n_events) > 0
        sel = (bars.layer[o] == 1) & ~multi[all_seg.event_id]
        l1_e0, l1_e1 = bars.e_mip_end0[o][sel] * k, bars.e_mip_end1[o][sel] * k
    return Summary(layers, grid_n, grid_e, ev_bars, ev_layers, ev_energy, multiplicity, l1_e0, l1_e1)


# repo-relative ASCII path for the pdf
def _short_path(path) -> str:
    p = Path(str(path))
    try:
        text = str(p.resolve().relative_to(REPO_ROOT)) if p.is_absolute() else str(path)
    except (ValueError, OSError):
        text = str(Path("...", *p.parts[-2:]))
    return text.encode("ascii", "replace").decode("ascii")


# a fit's failure reason as one short line
def _fit_reason(text, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    if text.startswith("fit_failed"):          # range_gauss_fit status: 'fit_failed: <reason>'
        text = text[len("fit_failed"):].lstrip(": ") or "no reason given"
    return text if len(text) <= limit else text[:limit - 3] + "..."


# Gaussian and Landau fits of the spectrum points; a failed fit gives None and its reason
def _peak_fits(xf, yf, fit_range, gauss_fit: bool = True, landau_fit: bool = True, why: dict | None = None):
    """A fit is kept only if it converged with finite errors and its peak lies inside fit_range."""
    from ..common.fitting import fit_curve, gauss, landau

    why = {} if why is None else why
    why.update(gauss=None, landau=None)
    if len(xf) < 5:
        text = f"{len(xf)} point(s) in the range (needs >= 5)"
        why.update(gauss=text if gauss_fit else None, landau=text if landau_fit else None)
        return None, None
    peak = float(xf[np.argmax(yf)])
    g = fit_curve(gauss, xf, yf, [float(yf.max()), peak, max((xf.max() - xf.min()) / 10, 0.05)]) if gauss_fit else None
    eta0 = 0.1
    lan = fit_curve(landau, xf, yf, [peak, eta0, float(yf.max()) * eta0 / 0.18]) if landau_fit else None

    # why a fit is not kept (None: kept)
    def failure(fit, i_peak):
        if not fit.ok:
            return _fit_reason(fit.message or "no finite parameters")
        if not np.all(np.isfinite(fit.errors)):
            return "non-finite errors (no covariance)"
        if not fit_range[0] < fit.params[i_peak] < fit_range[1]:
            return f"peak {fit.params[i_peak]:.4g} outside [{fit_range[0]:g}, {fit_range[1]:g}]"
        return None
    for name, fit, i_peak in (("gauss", g, 1), ("landau", lan, 0)):
        if fit is not None:
            why[name] = failure(fit, i_peak)
    return (g if g is not None and why["gauss"] is None else None), \
        (lan if lan is not None and why["landau"] is None else None)


# default diagnostic fits: Gaussian and Landau on mip_fit_range
def _spectrum_fits(centers, counts, fit_range, min_count, why: dict | None = None):
    keep = counts >= min_count
    x, y = centers[keep], counts[keep].astype(float)
    sel = (x >= fit_range[0]) & (x <= fit_range[1])
    g, lan = _peak_fits(x[sel], y[sel], fit_range, why=why)
    return x, y, g, lan


# points and fits of one per-end E_MIP spectrum (default, -fit[a,b] or -fit[a,b],[c,d])
def end_spectrum_fits(mip: np.ndarray, settings: dict) -> dict:
    """A failed fit has no curve and no value; the page names it with its reason."""
    lo, hi, nbins = settings["mip_spectrum_bins"]
    edges = np.linspace(float(lo), float(hi), int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    page_width = (float(hi) - float(lo)) / int(nbins)
    counts, _ = np.histogram(mip, bins=edges)
    min_count = settings["diagnostics_min_bin_count"]
    ranges = [tuple(map(float, r)) for r in (settings.get("fit_ranges") or [])]
    bins = settings.get("fit_bins") or None
    why = {"gauss": None, "landau": None}
    out = {"page_width": page_width, "ranges": ranges, "bins": bins, "fit_points": [], "gauss": None,
           "landau": None, "double": None, "scale": 1.0, "gauss_range": None, "landau_range": None, "why": why}
    if not ranges:
        fit_range = settings["mip_fit_range"]
        x, y, g, lan = _spectrum_fits(centers, counts, fit_range, min_count, why=why)
        out.update(x=x, y=y, gauss=g, landau=lan, gauss_range=tuple(fit_range), landau_range=tuple(fit_range))
        return out
    keep = counts >= min_count
    out["x"], out["y"] = centers[keep], counts[keep].astype(float)
    segments = []
    for i, rng in enumerate(ranges):
        if bins:
            e = np.linspace(rng[0], rng[1], int(bins[i]) + 1)
            c, cx, w = np.histogram(mip, bins=e)[0], 0.5 * (e[:-1] + e[1:]), (rng[1] - rng[0]) / int(bins[i])
        else:
            idx = np.flatnonzero(fitting.in_range(centers, rng))
            c, cx, w = counts[idx], centers[idx], page_width
        sel = c >= min_count
        segments.append((cx[sel], c[sel].astype(float), w))
        if bins:
            out["fit_points"].append((cx[sel], c[sel] * (page_width / w)))
    first_x, first_y, first_w = segments[0]
    out["scale"] = page_width / first_w if bins else 1.0
    if len(ranges) == 1:
        g, lan = _peak_fits(first_x, first_y, ranges[0], why=why)
        out.update(gauss=g, landau=lan, gauss_range=ranges[0], landau_range=ranges[0])
        return out
    edges_list = fitting.range_bin_edges(ranges, bins, edges)
    dbl = fitting.range_gauss_fit(mip, ranges, edges_list, min_bin_count=max(int(min_count), 1))
    # a component needs bins of its own: range_gauss_fit only counts the bins of both ranges together, and a range
    # without data leaves its component at the start values with zero errors, reported ok
    thin = [(r, int(np.sum(seg["fitted"]))) for r, seg in zip(ranges, dbl["segments"]) if np.sum(seg["fitted"]) < 3]
    if dbl["ok"] and thin:   # R3: NaN parameters, points only
        dbl = {**dbl, **{k: np.nan for k in dbl if k.startswith(("mu", "sigma", "area", "amplitude", "chi2"))},
               "ok": False, "components": [], "ndf": 0,
               "status": "fit_failed: " + "; ".join(f"[{a:g}, {b:g}] has {n} bin(s) with >= {max(int(min_count), 1)} "
                                                    f"entries (a component needs >= 3)" for (a, b), n in thin)}
    out["double"] = dbl
    _, lan = _peak_fits(first_x, first_y, ranges[0], gauss_fit=False, why=why)
    out.update(landau=lan, landau_range=ranges[0])
    return out


# draw the fitted curves of one spectrum; a failed fit gets a legend line with its reason
def _draw_end_fits(ax, f: dict) -> None:
    from ..common.fitting import format_ranges, gauss, landau, range_fit_curve

    ranges, scale = f["ranges"], f["scale"]
    on = (lambda r: f" on [{r[0]:g}, {r[1]:g}]") if ranges else (lambda r: "")
    positive = f["y"][f["y"] > 0]
    floor = 0.5 * float(positive.min()) if ranges and positive.size else -np.inf
    why = f.get("why") or {}

    # -fit: hide curve parts far below the points
    def shown(values):   # --fit: curve parts far below the points would stretch the log axis
        return values if not ranges else np.where(values >= floor, values, np.nan)

    # legend-only line of a failed fit
    def failed(model, rng, reason, suffix=""):   # legend-only: no curve, no value
        ax.plot([], [], " ", label=f"{model} fit failed on [{rng[0]:g}, {rng[1]:g}]{suffix}: "
                                   f"{_fit_reason(reason or 'no fit')}")

    for i, (px, py) in enumerate(f["fit_points"]):
        a, b = ranges[i]
        ax.plot(px, py, "o", ms=3, mfc="none", color="k",
                label=f"fit bins [{a:g}, {b:g}]: {f['bins'][i]} (scaled to the {f['page_width']:g}-MIP page bins)")
    g, lan, dbl = f["gauss"], f["landau"], f["double"]
    if g is None and lan is None and dbl is None:
        ax.plot([], [], " ", label=f"no MIP peak fitted in {list(f['gauss_range'])}" if not ranges else
                f"no MIP peak fitted in {format_ranges(ranges)}")
    if g is not None:
        xx = np.linspace(*f["gauss_range"], 300)
        ax.plot(xx, shown(gauss(xx, *g.params) * scale), "r-",
                label=f"Gauss mu={g.params[1]:.3f}+-{g.errors[1]:.3f}, sigma={abs(g.params[2]):.3f}"
                      + on(f["gauss_range"]))
    elif len(ranges) < 2:   # two ranges: the double Gaussian replaces the single one (not asked for)
        failed("Gauss", f["gauss_range"], why.get("gauss"))
    if dbl is not None:
        if dbl["ok"]:
            xx = np.linspace(dbl["fit_low"], dbl["fit_high"], 400)
            for c, comp in enumerate(dbl["components"]):
                ax.plot(xx, shown(range_fit_curve(dbl, xx, f["page_width"], component=c)), "r--", lw=0.9,
                        label=f"Gauss {c + 1} mu={comp['mu']:.3f}+-{comp['mu_error']:.3f}, "
                              f"sigma={comp['sigma']:.3f}" + on(comp["range"]))
            ax.plot(xx, shown(range_fit_curve(dbl, xx, f["page_width"])), "r-",
                    label=f"double Gauss (sum), chi2/ndf={dbl['chi2_over_ndf']:.2f}")
        else:
            ax.plot([], [], " ", label=f"double Gauss fit failed on {format_ranges(ranges)}: "
                                       f"{_fit_reason(dbl.get('status') or 'no fit')}")
    if lan is not None:
        xx = np.linspace(*f["landau_range"], 300)
        ax.plot(xx, shown(landau(xx, *lan.params) * scale), "g--",
                label=f"Landau MPV={lan.params[0]:.3f}+-{lan.errors[0]:.3f}" + on(f["landau_range"])
                      + (" (first range only)" if len(ranges) == 2 else ""))
    else:
        failed("Landau", f["landau_range"], why.get("landau"), " (first range only)" if len(ranges) == 2 else "")


# text of the -fit / -bin choice ('' = default fits)
def spectrum_fit_text(settings: dict) -> str:
    ranges = settings.get("fit_ranges")
    if not ranges:
        return ""
    bins = settings.get("fit_bins")
    what = ("Gaussian and Landau on the range" if len(ranges) == 1 else
            "double Gaussian (one component per range), Landau on the first range only")
    binning = (f"--bin {','.join(map(str, bins))}: equal bins per range" if bins else
               "the mip_spectrum_bins bins with centres in the range")
    return f"--fit {fitting.format_ranges(ranges)} (x = E_MIP [MIP]): {what}; {binning}"


# -fit / -bin of a clean call: diagnostics pdf only, never part of the selection
def diagnostic_fit_settings(args) -> dict:
    ranges, bins = getattr(args, "fit_ranges", None), getattr(args, "fit_bins", None)
    why = fitting.range_options_error(ranges or [], bins)
    if why:
        raise SettingsError(f"clean: {why}")
    return {"fit_ranges": [[float(a), float(b)] for a, b in ranges] if ranges else None,
            "fit_bins": [int(n) for n in bins] if bins else None}


# diagnostics pdf: cut flow page, per-end spectra, muon cuts, per-layer energies, maps, events
def draw_diagnostics(pdf, ds, bars: HitBars, summary: Summary, cutflow: list, settings: dict, job: Job,
                     title_lines: list[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    from ..common import plotting

    k = float(settings["mip_energy_mev"])
    lines = list(title_lines) + ["", f"{'step':<34}{'rows':>12}{'events':>10}"]
    lines += [f"{c['step']:<34}{c['rows']:>12}{c['events']:>10}" for c in cutflow]
    lines += ["", "definitions:"]
    for text in (ENERGY_DEFINITION, f"hit end: {HIT_DEFINITION}", f"{job.standard}: {STANDARDS[job.standard]}"):
        lines += textwrap.wrap(text, width=100, initial_indent="  ", subsequent_indent="    ")
    if spectrum_fit_text(settings):   # --fit / --bin (diagnostics only)
        lines += textwrap.wrap(f"per-end spectrum fits: {spectrum_fit_text(settings)}", width=100,
                               initial_indent="  ", subsequent_indent="    ")
    plotting.text_page(pdf, f"Clean dataset - {ds.label}", lines)
    c = bars.in_clean
    if not c.any():
        return

    lo, hi, nbins = settings["mip_spectrum_bins"]
    ranges = settings.get("fit_ranges") or []
    fig, axes = plotting.new_figure(1, 2, figsize=(14, 5.5))
    for end, ax in zip((0, 1), axes):
        hit = getattr(bars, f"hit_end{end}")[c]
        mip = getattr(bars, f"e_mip_end{end}")[c][hit]
        f = end_spectrum_fits(mip, settings)
        x, y = f["x"], f["y"]
        ax.plot(x, y, ".", ms=4, label=f"end {end}: hit ends of clean bars ({len(mip)})")
        _draw_end_fits(ax, f)
        if len(y) and y.max() > 0:
            ax.set_yscale("log")
        ax.set_xlim(min([float(lo)] + [float(r[0]) for r in ranges]), max([float(hi)] + [float(r[1]) for r in ranges]))
        ax.set_xlabel("E_MIP = (adc_sum - pedestal) / mpv [MIP]")
        ax.set_ylabel("ends per bin")
        sec = ax.secondary_xaxis("top", functions=(lambda v: v * k, lambda v: v / k))
        sec.set_xlabel(f"E_MeV [MeV] ({k:g} MeV per MIP)")
        ax.set_title(f"per-end energy, end {end}", pad=28)
        ax.legend(fontsize=7)
    fig.tight_layout()
    plotting.save(fig, pdf)

    if job.standard == "muon":
        fig, axes = plotting.new_figure(1, 2, figsize=(14, 5.5))
        m = summary.multiplicity
        top = int(m.max()) if len(m) else 1
        axes[0].hist(m, bins=np.arange(0.5, top + 1.5), histtype="step")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("kept bars in one (event, layer), before the event cuts")
        axes[0].set_ylabel("(event, layer) pairs")
        axes[0].set_title(f"{int((m >= 2).sum())} of {len(m)} pairs hold >= 2 kept bars (event vetoed)")
        wlo, whi = job.selection["layer1_energy_window_mev"]
        e0, e1 = summary.layer1_e0, summary.layer1_e1
        finite = np.isfinite(e0) & np.isfinite(e1)
        if finite.any():
            both = np.r_[e0[finite], e1[finite]]
            upper = max(float(np.percentile(both, 99.5)) * 1.1, whi * 1.5)
            edges2 = np.linspace(min(0.0, float(both.min())), upper, 101)
            _, _, _, im = axes[1].hist2d(e0[finite], e1[finite], bins=[edges2, edges2], cmin=1, norm=LogNorm(),
                                         cmap="viridis")
            plt.colorbar(im, ax=axes[1], label="layer-1 bars")
        axes[1].add_patch(Rectangle((wlo, wlo), whi - wlo, whi - wlo, fill=False, color="r", lw=1.2,
                                    label=f"window [{wlo:g}, {whi:g}] MeV"))
        with np.errstate(invalid="ignore"):
            inside = (e0 >= wlo) & (e0 <= whi) & (e1 >= wlo) & (e1 <= whi)
        axes[1].set_xlabel("layer-1 kept bar: E_MeV end 0 [MeV]")
        axes[1].set_ylabel("E_MeV end 1 [MeV]")
        axes[1].set_title(f"events with <= 1 kept bar per layer: {int(inside.sum())} of {len(e0)} "
                          f"layer-1 bars inside")
        axes[1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        plotting.save(fig, pdf)

    t = summary.layers
    fig, axes = plotting.new_figure(1, 2, figsize=(14, 5.5))
    axes[0].plot(t["layer"], t["energy_mev_end0_mean"], "o-", label="mean E_MeV, hit ends 0")
    axes[0].plot(t["layer"], t["energy_mev_end1_mean"], "s-", label="mean E_MeV, hit ends 1")
    axes[0].plot(t["layer"], t["energy_mev_bar_mean"], "^--", label="mean (E0 + E1)/2, bars with both ends hit")
    axes[0].axhline(k, color="k", lw=0.8, ls=":", label=f"1 MIP = {k:g} MeV")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("mean energy [MeV]")
    axes[0].set_title("clean bars: mean energy per layer")
    axes[0].legend(fontsize=7)
    axes[1].bar(t["layer"], t["n_clean_bars"], color="tab:blue", alpha=0.7)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("clean bars")
    axes[1].set_title("clean bars per layer")
    for ax in axes:
        ax.set_xticks(range(1, geo.N_LAYERS + 1))
    fig.tight_layout()
    plotting.save(fig, pdf)

    fig, axes = plotting.new_figure(1, 2, figsize=(15, 6.5))
    plotting.layer_map(axes[0], summary.bar_count, title="clean bars per bar", cbar_label="bars", annotate=True)
    plotting.layer_map(axes[1], summary.bar_energy, title="mean bar energy (hit ends) per bar",
                       cbar_label="MeV", annotate=True, fmt="{:.1f}")
    fig.tight_layout()
    plotting.save(fig, pdf)

    fig, axes = plotting.new_figure(1, 3, figsize=(18, 5.5))
    nmax = int(summary.event_bars.max())
    axes[0].hist(summary.event_bars, bins=np.arange(0.5, nmax + 1.5), histtype="step")
    axes[0].set_xlabel("clean bars per event")
    axes[0].set_ylabel("events")
    axes[1].hist(summary.event_layers, bins=np.arange(0.5, geo.N_LAYERS + 1.5), histtype="step")
    axes[1].set_xlabel("layers with a clean bar per event")
    energy = summary.event_energy
    width = float(settings["event_energy_bin_mev"])
    bins = np.arange(np.floor(energy.min()), energy.max() + width, width) if np.isfinite(energy).all() else 50
    axes[2].hist(energy, bins=bins, histtype="step")
    axes[2].set_xlabel("event energy: sum of the bar energies [MeV]")
    axes[2].set_title(f"mean {energy.mean():.2f} MeV, rms {energy.std():.2f} MeV, {len(energy)} events")
    fig.tight_layout()
    plotting.save(fig, pdf)


# title page lines: run, csv, beam csv, calibration, standard, threshold, clean csv
def _title_lines(ctx, ds, job: Job, info: dict, drawn: str) -> list[str]:
    cal = info.get("calibration") or {}
    inp = info.get("input_file") or {}
    counts = info.get("counts") or {}
    clean = info.get("clean_csv") or {}
    lines = [f"run          : {ds.key}  ({ds.label}){'  [TEST MODE]' if ctx.test else ''}",
             f"csv          : {info.get('csv', '?')}",
             f"beam csv     : {_short_path(inp.get('path', '?'))}  ({inp.get('hash', '?')} sha256 "
             f"{str(inp.get('sha256', '?'))[:12]})"]
    source = (info.get("input") or {}).get("source")
    if source:
        lines.append(f"               test subset of {_short_path(source.get('path', '?'))}")
    lines += [f"calibration  : {_short_path(cal.get('path', '?'))}  sha256 {str(cal.get('sha256', '?'))[:12]}",
              f"               values sha256 {str(cal.get('values_sha256', '?'))[:12]} (pedestal, mpv, threshold)",
              f"standard     : {job.standard}  [{info.get('standard_source', job.standard_source)}]",
              f"threshold    : {job.threshold_mode} = {job.threshold_column} ({THRESHOLD_TEXT[job.threshold_mode]})  "
              f"[{info.get('threshold_source', job.threshold_source)}]"]
    if job.standard == "muon":
        lo, hi = job.selection["layer1_energy_window_mev"]
        lines.append(f"layer 1      : both ends in [{lo:g}, {hi:g}] MeV, {job.selection['mip_energy_mev']:g} "
                     f"MeV per MIP")
    clean_text = (f"{Path(clean['path']).name}  sha256 {str(clean['sha256'])[:12]}" if clean.get("sha256")
                  else "not written by the selection run (-only)")
    lines += [f"clean csv    : {clean_text}",
              f"selected     : {counts.get('output_rows', '?')} rows, {counts.get('output_events', '?')} events",
              f"made         : {info.get('created', '?')}; {drawn}"]
    return lines


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------
# member signature of the clean csv in its record
def _clean_member(record: dict | None) -> dict | None:
    for key, sig in ((record or {}).get("members") or {}).items():
        if key.endswith("_clean.csv"):
            return sig
    return None


# peak resident memory of this process so far [MB] and the method
def _peak_memory_mb() -> tuple[float | None, str]:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            # PROCESS_MEMORY_COUNTERS of GetProcessMemoryInfo
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 2 ** 20, "GetProcessMemoryInfo PeakWorkingSetSize"
        except (OSError, AttributeError, ValueError):
            pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10), "getrusage ru_maxrss"
    except (ImportError, OSError, ValueError):
        return None, "unavailable"


# stop when a file was rewritten while it was read
def _check_unchanged(path: Path, before: dict, what: str) -> None:
    now = file_signature(path)
    if (now["size"], now["mtime_ns"]) != (before["size"], before["mtime_ns"]):
        raise InputError(f"{path} ({what}) was rewritten while this stage read it; run again")


# write the kept-bar cache
def _save_bars(path: Path, bars: HitBars) -> None:
    with open(path, "wb") as fh:
        np.savez(fh, **{name: getattr(bars, name) for name in CACHE_ARRAYS})


# read the kept-bar cache
def load_bars(path: Path) -> HitBars:
    """Row indices are not stored: -1."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].astype(_BAR_DTYPES[name], copy=False) for name in CACHE_ARRAYS}
    return HitBars(row=np.full(len(arrays["pf_event"]), -1, dtype=np.int64), **arrays)


# what the cached kept bars depend on: beam csv identity, calibration values, selection
def _cache_identity(info: dict) -> dict:
    return {"provenance_version": PROVENANCE_VERSION, "algorithm": ALGORITHM, "input": info["input"],
            "calibration_values": info["calibration_values"], "selection": info["selection"]}


# record of the kept-bar cache: its identity, what the result units need, the clean csv of the same run (or None)
def _cache_info(record: dict, clean: dict) -> dict:
    keys = ("dataset", "csv", "mode", "input_file", "calibration", "selection_hash", "selection_flags",
            "standard_source", "threshold_source", "counts", "cutflow", "created")
    return _as_json({**_cache_identity(record), **{k: record.get(k) for k in keys}, "clean_csv": clean,
                     "arrays": list(CACHE_ARRAYS)})


# settings the results depend on beyond the selection (display, -fit / -bin, MeV)
def _diagnostics_settings(settings: dict) -> dict:
    out = {k: settings[k] for k in ("mip_energy_mev", "mip_spectrum_bins", "mip_fit_range",
                                    "diagnostics_min_bin_count", "event_energy_bin_mev")}
    out.update({k: settings.get(k) for k in ("fit_ranges", "fit_bins")})   # --fit / --bin
    return out


# write the result units of one csv that the plan holds, then their record json (which lists only them)
def write_results(ctx, settings: dict, job: Job, csv, bars: HitBars, cutflow: list, info: dict, drawn: str,
                  plan) -> list:
    """Returns the files written."""
    ds = job.ds
    keys = {unit: result_key(ds, csv, unit) for unit, *_ in RESULT_PARTS}
    units = [u for u in RESULT_UNITS if _has_slot(plan, keys[u])]
    if not units:
        return []
    summary = (summarize(bars, job.standard, float(settings["mip_energy_mev"]))
               if {"layers", "diagnostics"} & set(units) else None)
    record_path = plan.path(keys["results"])
    written, names = [], {}
    meta = {"provenance_version": PROVENANCE_VERSION, "stage": "clean-results", "dataset": ds.key,
            "csv": info.get("csv"), "mode": ctx.mode, "clean_csv": info.get("clean_csv"),
            "selection": info.get("selection"), "selection_hash": info.get("selection_hash"),
            "diagnostics": _diagnostics_settings(settings), "drawn": drawn,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with ProductSet(record_path) as ps:
        for unit in units:
            path = plan.path(keys[unit])
            if unit == "cutflow":
                pd.DataFrame([{k: c.get(k, "") for k in ("step", "rows", "events", "definition")} for c in cutflow]
                             ).to_csv(ps.stage(path), index=False)
            elif unit == "layers":
                summary.layers.to_csv(ps.stage(path), index=False)
            else:
                from matplotlib.backends.backend_pdf import PdfPages

                with PdfPages(str(ps.stage(path))) as pdf:
                    draw_diagnostics(pdf, ds, bars, summary, cutflow, settings, job,
                                     _title_lines(ctx, ds, job, info, drawn))
            written.append(path)
            names[unit] = path.name
        ps.commit({**meta, "results": names, "saved": plan.saved(keys["results"])})
    if json_records_on(record_path):
        written.append(record_path)
    return written


# provenance record of the clean dataset of one csv (pass-2 output added by the caller)
def _official_record(ctx, ds, csv, job: Job, src: Path, input_id: dict, input_file: dict,
                     calib: ChannelCalibration, counts: dict, cutflow: list) -> dict:
    return _as_json({
        "provenance_version": PROVENANCE_VERSION,
        "algorithm": ALGORITHM,
        "stage": "clean",
        "module": __name__,
        "dataset": ds.key,
        "csv": ctx.event_source(ds, csv),        # '<run>/<csv name>' of the full set
        "particle": ds.particle,
        "mode": ctx.mode,
        "n_test_events": ctx.n_test_events if ctx.test else None,
        "input": input_id,                       # compared by clean_status (beam_identity)
        "input_file": input_file,                # the file that was read (information)
        "input_provenance": read_json(Path(src).with_name(f"{Path(src).stem}.provenance.json")),
        "calibration": {"path": canonical_path(calib.path), "sha256": calib.sha256,
                        "source": calibration_source(ctx, calib.path), "threshold_column": calib.threshold_column,
                        "values_sha256": calib.values_sha256,
                        "values_digest_of": ["pedestal", "mpv", calib.threshold_column],
                        "mpv_status_counts": calib.mpv_status_counts},
        "calibration_values": {"threshold_column": calib.threshold_column, "values_sha256": calib.values_sha256},
        "selection": job.selection,
        "selection_hash": settings_hash(_as_json(job.selection)),
        "selection_flags": job.flags,            # with the configuration of now they must give "selection"
        "standard_source": job.standard_source,
        "threshold_source": job.threshold_source,
        "definitions": {"energy": ENERGY_DEFINITION, "hit_end": HIT_DEFINITION,
                        "standard": STANDARDS[job.standard],
                        "threshold": f"{job.threshold_column} ({THRESHOLD_TEXT[job.threshold_mode]})",
                        "output": "header and the selected data lines of the beam csv, byte for byte, file order"},
        "counts": counts,
        "cutflow": cutflow,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": python_info(),
    })


# selection run of one csv: pass 1, event cuts, pass 2 (unit clean), kept-bar cache, result units
def select_csv(ctx, settings: dict, job: Job, csv, calib: ChannelCalibration, chunksize: int, plan,
               units) -> dict:
    """Every product set is committed record last: a failure never pairs a new csv with an old record."""
    ds = job.ds
    source = ctx.event_source(ds, csv)
    src = Path(require(ctx.beam_csv(ds, csv), produced_by=f".\\runner.ps1 make-csv {run_flags(ds)}"))
    before = file_signature(src)
    input_id = beam_identity(ctx, ds, csv)
    if input_id["path"] != canonical_path(src):
        raise InputError(f"[clean] {source}: the beam csv of this mode changed from {src} to {input_id['path']} "
                         f"while the run started; run again")
    if input_id["kind"] == "test-subset":                # the subset read must be the copy the identity names
        ok, why = subset_status(csv, src, ctx.n_test_events)
        if not ok:
            raise InputError(f"[clean] {source}: the test subset {src} does not match {csv} ({why}); run again")
    input_file = file_identity(src)
    with timed(f"{source}: pass 1 over {src.name} ({job.standard}, {job.threshold_mode})"):
        bars, counts = scan_beam_csv(src, calib, job.standard, chunksize)
    cutflow = [dict(step="input", rows=counts["input_rows"], events=counts["input_events"],
                    definition="rows / events of the beam csv")] + select_events(bars, job.standard, job.selection)
    counts["output_rows"], counts["output_events"] = cutflow[-1]["rows"], cutflow[-1]["events"]
    record = _official_record(ctx, ds, csv, job, src, input_id, input_file, calib, counts, cutflow)
    clean = {"path": canonical_path(clean_csv(ctx, ds, csv)), "sha256": None}
    written = []
    if "clean" in units:
        keep = bars.in_clean
        with timed(f"{source}: pass 2, copy {counts['output_rows']} rows -> {clean_csv(ctx, ds, csv).name}"):
            with ProductSet(provenance_json(ctx, ds, csv)) as ps:
                output = copy_selected_rows(src, ps.stage(clean_csv(ctx, ds, csv)), bars.row[keep],
                                            (bars.pf_event[keep], bars.layer[keep], bars.strip[keep]),
                                            counts["input_rows"])
                _check_unchanged(src, before, "beam csv")
                output.update(events=counts["output_events"], columns=list(pd.read_csv(src, nrows=0).columns))
                peak, method = _peak_memory_mb()
                record.update(output=_as_json(output), peak_memory_mb=None if peak is None else round(peak, 1),
                              peak_memory_method=method)
                official = ps.commit(record)
        clean["sha256"] = (_clean_member(official) or {}).get("sha256")
        written.append(clean_csv(ctx, ds, csv))
        if json_records_on(provenance_json(ctx, ds, csv)):
            written.append(provenance_json(ctx, ds, csv))
    else:
        _check_unchanged(src, before, "beam csv")
    info = _cache_info(record, clean)
    with ProductSet(cache_json(ctx, ds, csv)) as ps:
        _save_bars(ps.stage(cache_npz(ctx, ds, csv)), bars)
        ps.commit(info)
    written.append(cache_npz(ctx, ds, csv))
    written += write_results(ctx, settings, job, csv, bars, cutflow, info, "selection run", plan)
    for c in cutflow:
        log(f"  {source} {c['step']:<34} rows {c['rows']:>10}  events {c['events']:>8}")
    peak, _ = _peak_memory_mb()
    if peak is not None:
        log(f"  {source}: peak memory of this process so far {peak:.0f} MB")
    return {"source": source, "standard": job.standard, "rows": counts["output_rows"],
            "events": counts["output_events"], "written": written}


# why the calibration table no longer holds the values of a selection (None: the same values)
def _calibration_change(ctx, stored: dict) -> str | None:
    calib_path = Path(ctx.calibration_file())
    if not calib_path.exists():
        return f"the calibration table {calib_path} is missing"
    try:
        calib = read_calibration(calib_path, stored.get("threshold_column"))
    except (SystemExit, ValueError) as exc:
        return f"the calibration table {calib_path} cannot be used ({exc})"
    if calib.values_sha256 != stored.get("values_sha256"):
        return (f"{calib_path.name} now holds other pedestal / mpv / {stored.get('threshold_column')} values than "
                f"at the clean run")
    return None


# -from-saved: the result units of one csv from its kept-bar cache (beam csv only hashed, never read)
def redraw_from_saved(ctx, settings: dict, job: Job, csv, plan) -> dict:
    """Refused with the rebuild command when the cache has another selection, beam csv or calibration."""
    ds = job.ds
    source = ctx.event_source(ds, csv)
    hint = rebuild_command(ctx, job)
    info = require_product_set(cache_json(ctx, ds, csv),
                               {"provenance_version": PROVENANCE_VERSION, "algorithm": ALGORITHM}, rebuild=hint,
                               what=f"clean kept-bar cache of {source}", members=[cache_npz(ctx, ds, csv)])
    stored, wanted = info.get("selection") or {}, _as_json(job.selection)
    if stored != wanted:
        diff = _selection_diff(stored, wanted, "made with", "this call asks for")
        redraw = rebuild_command(ctx, configured_job(ctx, ds, info.get("selection_flags")), from_saved=True)
        raise ProductSetError(f"-from-saved: the kept-bar cache of {source} was made with another selection than "
                              f"this call asks for ({diff}).\n  Rebuild it with: {hint}\n"
                              f"  or use the stored selection with: {redraw}")
    try:
        now = beam_identity(ctx, ds, csv)
    except (SystemExit, OSError) as exc:
        raise ProductSetError(f"-from-saved: the beam csv of {source} cannot be checked ({exc}).\n"
                              f"  Rebuild with: {hint}") from None
    why = None if now == info.get("input") else (f"the beam csv differs from the one it was made from (stored "
                                                 f"{_short(info.get('input'))}, now {_short(now)})")
    why = why or _calibration_change(ctx, info.get("calibration_values") or {})
    if why:
        raise ProductSetError(f"-from-saved: the kept-bar cache of {source} ({ctx.mode}) is not current: {why}.\n"
                              f"  Rebuild it with: {hint}")
    bars = load_bars(cache_npz(ctx, ds, csv))
    counts = info.get("counts") or {}
    if int(bars.in_clean.sum()) != int(counts.get("output_rows", -1)):
        raise ProductSetError(f"-from-saved: the cache of {source} holds {int(bars.in_clean.sum())} clean bars, its "
                              f"record {counts.get('output_rows')}.\n  Rebuild with: {hint}")
    with timed(f"{source}: the result units from {cache_npz(ctx, ds, csv).name}"):
        written = write_results(ctx, settings, job, csv, bars, info.get("cutflow") or [], info,
                                f"redrawn with -from-saved from {cache_npz(ctx, ds, csv).name}", plan)
    return {"source": source, "standard": job.standard, "rows": counts.get("output_rows"),
            "events": counts.get("output_events"), "written": written}


# --------------------------------------------------------------------------
# currency: is the clean dataset of a csv current?  (RunContext.input_csv with -clean-input)
# --------------------------------------------------------------------------
_REPORTED: set = set()


# is the clean dataset of one csv current? (current, reason, record; record None with -no-json)
def clean_status(ctx, ds, csv) -> tuple[bool, str, dict | None]:
    """Current = intact, same beam csv identity, same selection now, same calibration values."""
    path, meta = clean_csv(ctx, ds, csv), provenance_json(ctx, ds, csv)
    ok, reason = product_set_valid(meta, {"provenance_version": PROVENANCE_VERSION, "algorithm": ALGORITHM},
                                   members=[path])
    if not ok:
        return False, reason, None
    record = read_json(meta)
    if not isinstance(record, dict):                     # -no-json: no record, nothing to verify
        return True, reason, None
    try:
        now = beam_identity(ctx, ds, csv)
    except (SystemExit, OSError) as exc:                 # InputError, or the file vanished while checked
        return False, f"its beam csv cannot be used ({exc})", record
    if now != record.get("input"):
        return False, (f"the beam csv differs from the one it was made from (stored {_short(record.get('input'))}, "
                       f"now {_short(now)})"), record
    try:
        wanted = _as_json(configured_job(ctx, ds, record.get("selection_flags")).selection)
    except SystemExit as exc:                            # SettingsError
        return False, f"the clean_dataset settings of the configuration cannot be used ({exc})", record
    stored_sel = record.get("selection") or {}
    if stored_sel != wanted:
        return False, (f"the clean_dataset settings of the configuration now give another selection "
                       f"({_selection_diff(stored_sel, wanted, 'made with', 'configuration now')})"), record
    why = _calibration_change(ctx, record.get("calibration_values") or {})
    return (False, why, record) if why else (True, "ok", record)


# clean command that makes the dataset again with its recorded flags
def _remake_command(ctx, ds, record: dict | None) -> str:
    try:
        return rebuild_command(ctx, configured_job(ctx, ds, (record or {}).get("selection_flags")))
    except SystemExit:
        return f".\\runner.ps1 clean {run_flags(ds)}" + (" -test" if ctx.test else "")


# -clean-input: the clean dataset of one csv when it is current, else None (reason logged)
def current_clean_csv(ctx, ds, csv) -> Path | None:
    ok, reason, record = clean_status(ctx, ds, csv)
    path = clean_csv(ctx, ds, csv)
    if not ctx.dry_run:
        sel = (record or {}).get("selection") or {}
        key = (canonical_path(path), ok, reason, str((_clean_member(record) or {}).get("sha256")))
        if key not in _REPORTED:
            _REPORTED.add(key)
            source = ctx.event_source(ds, csv)
            if ok and record is None:
                log(f"clean input {source}: {path.name} ({reason})")
            elif ok:
                counts = record.get("counts") or {}
                log(f"clean input {source}: {path.name} (standard {sel.get('standard')} "
                    f"[{record.get('standard_source')}], threshold {sel.get('threshold_mode')} = "
                    f"{sel.get('threshold_column')} [{record.get('threshold_source')}]; "
                    f"{counts.get('output_rows')} rows, {counts.get('output_events')} events; "
                    f"made {record.get('created')})")
            elif record is not None or path.exists():
                log(f"clean dataset {path.name} of {source} ({ctx.mode}) is not current: {reason}.  Make it again "
                    f"with: {_remake_command(ctx, ds, record)}")
    return path if ok else None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
# clean command-line options
def add_arguments(p: argparse.ArgumentParser) -> None:
    lo, hi = DEFAULTS["layer1_energy_window_mev"]
    by_particle = DEFAULTS["standard_by_particle"]
    std = p.add_mutually_exclusive_group()
    std.add_argument("--muon-standard", "-muon-standard", dest="standard", action="store_const", const="muon",
                     default=None,
                     help=f"muon standard: bar kept when both ends are hits; events with at most one kept bar per "
                          f"layer and a layer-1 kept bar with both end energies in layer1_energy_window_mev "
                          f"(default [{lo:g}, {hi:g}] MeV at mip_energy_mev = {DEFAULTS['mip_energy_mev']:g} MeV per "
                          f"MIP) (default for "
                          f"{', '.join(k for k, v in by_particle.items() if v == 'muon')} datasets)")
    std.add_argument("--otherparticle", "-otherparticle", "--otherparticle-standard", "-otherparticle-standard",
                     dest="standard", action="store_const",
                     const="otherparticle",
                     help=f"other-particle standard: bar kept when either end is a hit, no event cut (default for "
                          f"{', '.join(k for k, v in by_particle.items() if v == 'otherparticle')} datasets)")
    thr = p.add_mutually_exclusive_group()
    thr.add_argument("--pedestalscale", "-pedestalscale", dest="threshold_mode", action="store_const",
                     const="pedestalscale", default=None,
                     help=f"hit threshold {THRESHOLD_COLUMNS['pedestalscale']} = 1.2 x pedestal of the calibration "
                          f"table ({'default, ' if DEFAULTS['threshold_mode'] == 'pedestalscale' else ''}config "
                          f"threshold_mode; the 2022 analysis, in sync with that year's MC standard)")
    thr.add_argument("--pedestalsigma", "-pedestalsigma", dest="threshold_mode", action="store_const",
                     const="pedestalsigma",
                     help=f"hit threshold {THRESHOLD_COLUMNS['pedestalsigma']} = pedestal + 5 sigma of the "
                          f"calibration table ({'default, ' if DEFAULTS['threshold_mode'] == 'pedestalsigma' else ''}"
                          f"recommended by Lei for future years)")
    # every fit gets a range option; here the diagnostic fits only (the selection never fits)
    p.add_argument("--fit", "-fit", dest="fit_ranges", type=fitting.parse_ranges, default=None, metavar="[a,b]",
                   help="diagnostics only: the per-end E_MIP spectrum fits of <csv stem>_clean_diagnostics.pdf "
                        "(x = E_MIP [MIP]): --fit[a,b] fits the Gaussian and the Landau on [a, b]; --fit[a,b],[c,d] "
                        "a double Gaussian (one component per range) and the Landau on [a, b] only; the clean "
                        "dataset is not affected; with -from-saved the pdf is redrawn (default: mip_fit_range "
                        f"{DEFAULTS['mip_fit_range']}, single Gaussian)")
    p.add_argument("--bin", "-bin", dest="fit_bins", type=fitting.parse_bins, default=None, metavar="N[,N2]",
                   help="with --fit: N equal bins in each fit range, one value per range (default: the "
                        "mip_spectrum_bins bins whose centres lie in the range)")
    outputs.add_only_argument(p, UNITS, groups=UNIT_GROUPS)


# plan item of the calibration table (this mode's table; -test without one: the full-set table)
def _calibration_plan_item(ctx) -> dict:
    path = Path(ctx.calibration_file())
    test_table = ctx.test and canonical_path(path) == canonical_path(ctx.calibration_product())
    return {"role": "in", "kind": "product", "path": path,
            "produced_by": ".\\runner.ps1 calib" + (" -test" if test_table else "")}


# plan item of the file the beam identity reads: the beam csv, or for a test subset its full csv
def _identity_plan_item(ctx, ds, csv) -> dict:
    item = ctx.beam_plan_item(ds, csv)
    if item.get("kind") != "auto":
        return item
    return {"role": "in", "kind": "external", "path": Path(item["source"]),
            "produced_by": f".\\runner.ps1 make-csv {run_flags(ds)}"}


# plan items of the final results per (run key, csv), at the destinations of the output plan
def _final_plan_items(ctx, args, jobs, units) -> dict:
    preview = output_plan(ctx, args, jobs, units, preview=True)
    for w in preview.warnings:
        log(f"WARNING: {w}")
    items = {}
    for job in jobs:
        for csv in ctx.run_csvs(job.ds):
            for unit, kind, *_ in RESULT_PARTS:
                key = result_key(job.ds, csv, unit)
                if not _has_slot(preview, key) or (kind == "json" and not json_records_on(preview.path(key))):
                    continue
                item = {"role": "out", "path": preview.path(key), "result": key, "saved": preview.saved(key)}
                if not preview.saved(key):
                    item["note"] = "(not saved: no --output)"
                items.setdefault((job.ds.key, str(csv)), []).append(item)
    return items


# inputs and outputs of the call (for -dry-run, check and the run log)
def io_plan(ctx, args, datasets) -> list[dict]:
    """Reads only small records; writes and deletes nothing."""
    settings, _ = load_settings(ctx)
    jobs = make_jobs(settings, args, datasets)
    diagnostic_fit_settings(args)                       # --fit / --bin: refused here as in the run
    units = call_units(ctx, args)
    finals = _final_plan_items(ctx, args, jobs, units)
    from_saved = getattr(args, "from_saved", False)
    plan, outs = [], []
    for job in jobs:
        ds = job.ds
        csvs = ctx.run_csvs(ds)
        if not csvs:
            plan += ctx.beam_plan_items(ds)              # the run folder as a missing input
        for csv in csvs:
            if from_saved:
                plan += [_identity_plan_item(ctx, ds, csv),
                         {"role": "cache", "path": cache_json(ctx, ds, csv), "members": [cache_npz(ctx, ds, csv)],
                          "expected": {"provenance_version": PROVENANCE_VERSION, "algorithm": ALGORITHM,
                                       "selection": _as_json(job.selection)}}]
            else:
                plan.append(ctx.beam_plan_item(ds, csv))
                outs += [{"role": "out", "path": p} for p in _intermediates(ctx, ds, csv, units)
                         if p.suffix != ".json" or json_records_on(p)]
            outs += finals.get((ds.key, str(csv)), [])
    return plan + [_calibration_plan_item(ctx)] + outs


# stage entry: settings, jobs, units, output plan, then per csv a selection run or the units from the cache
def run(ctx, args, datasets) -> None:
    settings, unknown = load_settings(ctx)
    if unknown:
        log(f"WARNING: config {CONFIG_SECTION} keys this stage does not use: {', '.join(sorted(unknown))}")
    jobs = make_jobs(settings, args, datasets)          # a bad flag or setting fails before any file is read
    settings = {**settings, **diagnostic_fit_settings(args)}     # --fit / --bin: the pdf only
    units = call_units(ctx, args)
    if spectrum_fit_text(settings):
        log(f"diagnostic per-end spectrum fits: {spectrum_fit_text(settings)}; the clean dataset does not depend on "
            "them" + ("" if "diagnostics" in units else " (no diagnostics pdf: nothing is drawn, so they have no "
                                                        "effect)"))
    if getattr(args, "select_muon", None) and not getattr(args, "standard", None):
        log("note: --muon selects the muon runs (as in every stage); the muon selection standard on another run "
            "is -muon-standard")
    if not jobs:
        log("clean: no dataset selected")
        return
    for job in jobs:                                     # a run without csv stops before anything is written
        csvs = run_csvs(ctx, job.ds)
        log(f"  {job.ds.key}: standard {job.standard} [{job.standard_source}]; threshold {job.threshold_mode} = "
            f"{job.threshold_column} [{job.threshold_source}]; {len(csvs)} csv file(s) in {job.ds.folder}")
    log(f"output units: {', '.join(units)}")
    # ONE output plan per call (it empties the scratch folder of a full-set run without --output)
    plan = output_plan(ctx, args, jobs, units)
    pairs = [(job, csv) for job in jobs for csv in run_csvs(ctx, job.ds)]
    if getattr(args, "from_saved", False):
        summary = [redraw_from_saved(ctx, settings, job, csv, plan) for job, csv in pairs]
    else:
        for job, csv in pairs:                           # a protected output folder fails before any read
            for path in _intermediates(ctx, job.ds, csv, units):
                ensure_writable(path)
        for s in plan.slots:
            ensure_writable(plan.path(s.key))
        calib_path = Path(require(ctx.calibration_file(), produced_by=_calibration_plan_item(ctx)["produced_by"]))
        before = file_signature(calib_path)
        calib = read_calibration(calib_path, jobs[0].threshold_column, check=True)   # one threshold mode per call
        _check_unchanged(calib_path, before, "calibration table")
        log(f"calibration table: {calib_path} ({calibration_source(ctx, calib_path)}; threshold column "
            f"{calib.threshold_column})")
        _warn_calibration_status(calib)
        chunksize = int(ctx.cfg.get("run", {}).get("chunksize", 2_000_000))
        summary = [select_csv(ctx, settings, job, csv, calib, chunksize, plan, units) for job, csv in pairs]
    log("clean summary:")
    for s in summary:
        log(f"  {s['source']}: {s['standard']}: {s['rows']} rows in {s['events']} events -> "
            + (", ".join(Path(p).name for p in s["written"]) or "nothing written"))
    _log_final_results(plan)
