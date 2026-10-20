"""ROOT file -> beam csv (stage make-csv): one csv per ROOT file, nothing merged.

Every time sample of a digi (one bar end in one event) is summed into per-end variables; the two ends of a bar are
paired per event (a bar end without the other end is dropped). A run of configure/datasets.yml is the folder
data/analysis_files/<Particle>_<E>GeV[_run<N>]/ (-test: data/test/analysis_files/<run>/) with one csv per ROOT file
of the run, named after it (fixed_Muon_4GeV.root -> Muon_4GeV/fixed_Muon_4GeV.csv). -input FILES: one csv per file
directly in data/analysis_files/ (-test: data/test/analysis_files/). Output units (-only): csv (the beam csv + its
provenance json <csv stem>.provenance.json) and qa (results/<mode>/make_csv_qa/[<run>/]<csv stem>_qa.pdf).

Variables of the beam csv (column order), one row per (pf_event, layer, strip):
  pf_event, tot_end0, toa_end0, adc_sum_end0, adc_mean_end0, adc_max_end0, layer, strip, pf_spill, pf_ticks,
  tot_end1, toa_end1, adc_sum_end1, adc_mean_end1, adc_max_end1,
  raw_tot_sum_end0, raw_tot_sum_end1, raw_toa_sum_end0, raw_toa_sum_end1
  adc_0_end0 .. adc_7_end0 and adc_0_end1 .. adc_7_end1 (with -waveform, after adc_max_end0 / adc_max_end1)
  adc_sum = sum of the ADC samples a[k] (pedestal included), adc_mean = adc_sum / n, adc_max = max a[k];
  tot = the single non-zero TOT sample (2048 -> 0), 0 if none or several;
  toa [ns] = (q/1023)*25 + 25*(k + [q < bx_shift]) of the single non-zero TOA sample q at position k, minus
  mean_shift on end 1, 0 if none or several (TOA table calibration/reference/toa_calibration_phase3.csv);
  raw_tot_sum / raw_toa_sum = sums of the raw TOT / TOA words (before the single-hit and 2048 rules).
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..common import geometry as geo
from ..common import outputs
from ..common.config import SELECTABLE_PARTICLES, resolve_path, root_csv_name, run_flags, run_ref
from ..common.io_utils import (ProductSet, SettingsError, canonical_path, ensure_writable, file_signature,
                               json_records_on, log, product_set_valid, read_json, require, settings_hash,
                               validate_channel_keys, validate_settings)

HELP = "ROOT file -> per-bar beam csv (one csv per ROOT file, raw TOT/TOA sums included), data and MC"
DEFAULT_DATASETS = ["all"]  # no run chosen: every data run of configure/datasets.yml whose ROOT files are here
CONFIG_SECTION = "make_csv"
UNITS = ("csv", "qa")       # -only output units: the beam csv (+ provenance), the QA pdf
DEFAULT_UNITS = ("csv",)    # without -only: the csv only

DEFAULTS = {
    "tree": "ntuplizehgcroc/hgcroc",
    "events_tree": "LDMX_Events",
    "digi_branch": "auto",
    "chunk_entries": 3_072_000,
    "events_per_chunk": 20_000,
    "waveform": False,
    "max_events": None,
    "mc": False,
    "mc_samples_per_digi": "auto",
    "tot_mode": "tot_branch",
    "duplicate_digis": "error",
    "sample_window": "all",
    "toa_calibration": None,
    "missing_toa_calibration": "drop",
    "line_terminator": "crlf",
    "parallel_write_min_rows": 2_000_000,
    "overwrite": False,
    "qa_bins": 100,
    "qa_cmin": 0.01,
    "qa_cmap": "jet",
}

# fixed HGCROC constants (based on Joe Muse's apply_calibrations.py, revised by Lei)
BX_NS = 25                # ns per bunch crossing (time sample)
TOA_FULL_SCALE = 1023     # divisor of the 10-bit TOA code (1023, not 1024, as in Joe Muse's code)
TOT_BAD_CODE = 2048       # TOT value mapped to 0 ("not yet understood") but still counted as non-zero
N_WAVEFORM_SAMPLES = geo.N_TIME_SAMPLES  # adc_0..adc_7
DATA_SAMPLES_PER_DIGI = geo.N_TIME_SAMPLES
ID_RANGE = 256            # layer / strip ids must be < 256 (HcalDigiID has 8 bits for each)

NTUPLE_BRANCHES = ["pf_event", "layer", "strip", "end", "i_sample", "adc", "tot", "toa", "pf_spill", "pf_ticks"]
# chunk schema shared by both readers (one row per digi sample); "i_sample" (sample index, else file
# order) and "tot_comp" (TOT-complete flag; always from event files, from ntuples with --tot-mode tot_comp)
# are optional
CHUNK_FIELDS = ("pf_event", "layer", "strip", "end", "adc", "tot", "toa", "pf_spill", "pf_ticks")
PRODUCED_BY_ROOT = ("LDMX ntuplizer (external); point the run's 'root' at it in configure/datasets.yml (or in a "
                    "'datasets:' entry of configure/local.yml, keyed by the run, e.g. \"muon 4\":)")
PRODUCED_BY_TOA = "Joe Muse's TOA calibration (calibration/reference/toa_calibration_phase3.csv)"
_LINE_TERMINATORS = {"crlf": "\r\n", "lf": "\n", "os": os.linesep}

# product locations (contract: make-csv [analysis_files / make_csv_qa])
STAGE_DIR = "analysis_files"   # data/analysis_files/<run>/ (paths.analysis_files_dir), data/test/analysis_files/<run>/
QA_DIR = "make_csv_qa"         # results/fullset/make_csv_qa/<run>/, results/test/make_csv_qa/<run>/ (-test)
CSV_FORMAT_VERSION = 1         # bump when a column definition or the text format changes (documents the schema;
#                                the reuse rule also compares the code fingerprint, see converter_fingerprint)
RAW_SUM_COLUMNS = ["raw_tot_sum_end0", "raw_tot_sum_end1", "raw_toa_sum_end0", "raw_toa_sum_end1"]
# settings that change the csv content: hashed into the provenance (+ mc and the TOA table signature)
CSV_SETTING_KEYS = ("tree", "events_tree", "digi_branch", "waveform", "tot_mode", "duplicate_digis", "window",
                    "line_terminator")
# the code that decides the csv content (module-level functions, classes and constants of this file);
# converter_fingerprint() hashes their syntax trees, so a test csv written by other code is not reused
CSV_CODE = (
    "BX_NS", "TOA_FULL_SCALE", "TOT_BAD_CODE", "N_WAVEFORM_SAMPLES", "ID_RANGE", "NTUPLE_BRANCHES", "CHUNK_FIELDS",
    "_LINE_TERMINATORS", "RAW_SUM_COLUMNS", "beam_csv_columns", "load_toa_calibration", "toa_to_ns",
    "select_single_tot", "_parse_window", "_concat", "_take", "_event_chunks", "_dense_entry_stop", "_open_input",
    "_iter_ntuple", "_digi_prefix", "_event_file_names", "_iter_event_file", "_pack_key", "_tot_source",
    "_digis_regular", "_digis_generic", "_subset", "BarSlots", "_compact", "_Chunk", "BarStore", "_pair_ends",
    "_samples_per_digi", "_check_first_chunk", "_generic_digis", "_process_chunk", "convert", "_bar_columns",
    "_written_bars", "_csv_text", "write_beam_csv")
_FINGERPRINT: dict | None = None
TOT_NE_ADC_EVENTS = ("--tot-mode tot_ne_adc is only for flat MC ntuples whose tot branch repeats adc; in an "
                     "ldmx-sw event file the TOT of a TOT-complete sample IS the adc field, so tot_ne_adc would "
                     "erase every TOT. Use --tot-mode tot_branch or tot_comp (identical for event files).")


# --------------------------------------------------------------------------
# csv schema and canonical paths
# --------------------------------------------------------------------------
# beam-csv column order: 15 per-bar columns (+ waveform samples), then the raw sums
def beam_csv_columns(waveform: bool = False) -> list[str]:
    # columns of one end
    def end_block(e):
        cols = [f"tot_end{e}", f"toa_end{e}", f"adc_sum_end{e}", f"adc_mean_end{e}", f"adc_max_end{e}"]
        return cols + ([f"adc_{k}_end{e}" for k in range(N_WAVEFORM_SAMPLES)] if waveform else [])
    cols = ["pf_event"] + end_block(0) + ["layer", "strip", "pf_spill", "pf_ticks"] + end_block(1)
    return cols + list(RAW_SUM_COLUMNS)


# provenance json path: <csv stem>.provenance.json
def provenance_path(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}.provenance.json")


# default QA pdf path: results/<mode>/make_csv_qa/[<run>/]<csv stem>_qa.pdf
def qa_pdf_path(ctx, csv: Path, ds=None) -> Path:
    folder = ctx.results_dir(QA_DIR) if ds is None else ctx.results_dir(QA_DIR, Path(ds.folder).name)
    return folder / f"{Path(csv).stem}_qa.pdf"


# result key of a QA pdf: <csv stem>_qa (characters outside A-Z a-z 0-9 _ . - become _)
def qa_key(csv: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(csv).stem) + "_qa"


# final results of the run (for -output): one QA pdf per job of the qa unit
def qa_slots(jobs) -> list:
    """The csv and its provenance are the csv unit, not a final result: never in the plan."""
    return [outputs.ResultSlot(qa_key(job.out_csv), "pdf", Path(job.qa_pdf), f"QA histograms of {job.label}")
            for job in jobs if job.qa_pdf is not None]


# send every QA pdf to its -output destination; returns the output plan
def route_qa(ctx, jobs, dry: bool = False) -> outputs.OutputPlan:
    """dry=True (dry-run, check, run log): a preview plan that never empties the scratch folder."""
    try:
        plan = outputs.OutputPlan(ctx, QA_DIR, qa_slots(jobs), preview=dry)
    except ValueError as exc:  # two results would be written to one file
        raise SettingsError(f"make-csv: {exc}") from None
    if dry:
        for w in plan.warnings:
            log(f"WARNING: {w}")
    for job in jobs:
        if job.qa_pdf is not None:
            job.qa_pdf, job.qa_saved = plan.path(qa_key(job.out_csv)), plan.saved(qa_key(job.out_csv))
    return plan


# make_csv settings of the yml alone (no command-line flags)
def configured_settings(ctx) -> dict:
    return _settings(ctx, argparse.Namespace())


# warning when a run's csv is made from part of the events or with settings other than the yml
def formal_csv_warning(ctx, ds, out_csv: Path, settings: dict, mc: bool, event_limit: int | None) -> str | None:
    """The run's csv is made with what was given (its provenance records how); None = nothing to say."""
    if ctx.test or ds is None:
        return None
    base = configured_settings(ctx)
    expected = csv_settings(base, bool(base["mc"]) or ds.kind == "mc")
    got = csv_settings(settings, mc)
    diff = sorted(k for k in set(expected) | set(got) if expected.get(k) != got.get(k))
    parts = [f"with settings that differ from configure/config.yml ({', '.join(diff)})"] if diff else []
    if event_limit:
        parts.append(f"from the first {event_limit} events only")
    if not parts:
        return None
    return f"[{run_ref(ds)}] the formal csv {out_csv.name} is made {' and '.join(parts)}"


# --------------------------------------------------------------------------
# TOA calibration table (Joe's toa_calibration_phase3.csv)
# --------------------------------------------------------------------------
# per-bar TOA constants (bx_shift, mean_shift) as [layer, strip] lookups
@dataclass
class ToaCalibration:

    source: str
    toa_bx_wrap_code: np.ndarray   # TOA code below which a hit belongs to the next BX (bx_shift)
    toa_end1_offset_ns: np.ndarray  # end1 - end0 mean offset subtracted from end 1 (mean_shift)
    has_row: np.ndarray            # bar present in the table

    # (bx_shift, mean_shift, has_row) of one bar
    def constants(self, layer: int, strip: int) -> tuple[float, float, bool]:
        return (float(self.toa_bx_wrap_code[layer, strip]), float(self.toa_end1_offset_ns[layer, strip]),
                bool(self.has_row[layer, strip]))


# read the TOA table (layer, bar, bx_shift, mean_shift); none = no TOA correction
def load_toa_calibration(path) -> ToaCalibration:
    """Keys are checked (each prototype bar at most once); a corrupt table stops the run."""
    shape = (ID_RANGE, ID_RANGE)
    if path is None or str(path).strip().lower() == "none":
        return ToaCalibration("none", np.zeros(shape), np.zeros(shape), np.ones(shape, bool))
    path = require(Path(path), produced_by=PRODUCED_BY_TOA)
    table = pd.read_csv(path)
    cols = list(table.columns)

    # column by name, else at its position in Joe Muse's table
    def value_column(name: str, position: int):
        return name if name in cols else (cols[position] if len(cols) > position else None)

    bx_col, offset_col = value_column("bx_shift", 4), value_column("mean_shift", 5)
    if ("layer" not in cols or "bar" not in cols or bx_col is None or offset_col is None
            or bx_col == offset_col or {bx_col, offset_col} & {"layer", "bar"}):
        raise SettingsError(f"TOA table {path}: expected the columns layer, bar, bx_shift, mean_shift (by name, or "
                            f"bx_shift / mean_shift at the positions 4 / 5; got {cols})")

    # key column as float, shifted (0-based layer -> 1-based)
    def key(col: str, shift: int) -> pd.Series:  # true / false stay as they are: the shared rule refuses them
        c = table[col]
        return c if pd.api.types.is_bool_dtype(c) else pd.to_numeric(c, errors="coerce").astype(float) + shift

    keys = validate_channel_keys(pd.DataFrame({"layer": key("layer", 1), "strip": key("bar", 0)}),
                                 keys=("layer", "strip"), source=f"TOA table {path} (its 0-based layer is shown + 1)")
    values = {}
    for col in (bx_col, offset_col):
        v = pd.to_numeric(table[col], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(v).all():
            i = int(np.argmax(~np.isfinite(v)))
            raise SettingsError(f"TOA table {path}: column {col} must hold finite numbers (row {i}: "
                                f"{table[col].iloc[i]!r})")
        values[col] = v
    layer, strip = keys["layer"].to_numpy(), keys["strip"].to_numpy()
    bx = np.zeros(shape)
    offset = np.zeros(shape)
    has = np.zeros(shape, bool)
    bx[layer, strip] = values[bx_col]
    offset[layer, strip] = values[offset_col]
    has[layer, strip] = True
    return ToaCalibration(str(path), bx, offset, has)


# toa calculation, toa [ns] = (q/1023)*25 + 25*(k + [q < bx_shift]), minus mean_shift for end 1; 0 if q = 0
def toa_to_ns(toa_code, sample_position, bx_wrap_code: float, end1_offset_ns: float | None = None) -> np.ndarray:
    """Fixed operation order: another order changes the last bit of the csv values."""
    code = np.asarray(toa_code, dtype=np.float64)
    k = np.asarray(sample_position, dtype=np.int64)
    t = (code / TOA_FULL_SCALE) * BX_NS + BX_NS * (k + (code < bx_wrap_code))
    if end1_offset_ns is not None:
        t = t - end1_offset_ns
    return np.where(code != 0, t, 0.0)


# tot calculation, the single non-zero TOT sample (2048 -> 0); 0 if none or several
def select_single_tot(tot_value, n_nonzero) -> np.ndarray:
    tot_value = np.asarray(tot_value)
    return np.where(np.asarray(n_nonzero) == 1, np.where(tot_value == TOT_BAD_CODE, 0, tot_value), 0)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# -sample-window: all -> (0, None), A:B -> (A, B) = samples A..B-1
def _parse_window(value) -> tuple[int, int | None]:
    if value is None or str(value).strip().lower() in ("", "all", "none"):
        return 0, None
    text = str(value).strip()
    lo, sep, hi = text.partition(":")
    try:
        if not sep:
            raise ValueError
        lo = int(lo) if lo.strip() else 0
        hi = int(hi) if hi.strip() else None
    except ValueError:
        lo, hi = -1, None
    if lo < 0 or (hi is not None and hi <= lo):
        raise SettingsError(f"--sample-window / make_csv.sample_window {value!r}: expected all or A:B with "
                            f"integers 0 <= A < B (samples A..B-1)")
    return lo, hi


# settings check of sample_window
def _window_rule(value) -> str | None:
    try:
        _parse_window(value)
    except SettingsError:
        return "all or A:B with integers 0 <= A < B (samples A..B-1)"
    return None


# -mc-samples: auto or a positive integer (None if invalid)
def _mc_samples(value) -> str | int | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        return int(text) if text.isdigit() and int(text) > 0 else None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return int(value) if np.isfinite(value) and float(value).is_integer() and value > 0 else None


# settings check of mc_samples_per_digi
def _mc_samples_rule(value) -> str | None:
    return None if _mc_samples(value) is not None else "auto or a positive integer"


# settings check of toa_calibration
def _toa_table_rule(value) -> str | None:
    if value is None or (isinstance(value, (str, Path)) and str(value).strip()):
        return None
    return "a csv path, none (no TOA correction) or null (= paths.toa_calibration)"


# yml section make_csv + command-line flags, checked before any file is opened
SETTINGS_SPEC = {
    "tree": "str", "events_tree": "str", "digi_branch": "str",
    "chunk_entries": "int>0", "events_per_chunk": "int>0",
    "waveform": "bool", "mc": "bool", "overwrite": "bool",
    "max_events": "int>0?",
    "mc_samples_per_digi": _mc_samples_rule,
    "tot_mode": ("choice", ["tot_branch", "tot_comp", "tot_ne_adc"]),
    "duplicate_digis": ("choice", ["error", "drop"]),
    "sample_window": _window_rule,
    "toa_calibration": _toa_table_rule,
    "missing_toa_calibration": ("choice", ["drop", "error"]),
    "line_terminator": lambda v: None if str(v).lower() in _LINE_TERMINATORS else "crlf, lf or os",
    "parallel_write_min_rows": "int>=0",
    "qa_bins": "int>0", "qa_cmin": "float>=0", "qa_cmap": "str",
}


# make-csv settings: defaults < yml section make_csv < command line, checked
def _settings(ctx, args) -> dict:
    """An unknown yml key (e.g. a typo) is refused."""
    yml = ctx.stage_cfg(CONFIG_SECTION)
    unknown = sorted(set(yml) - set(DEFAULTS), key=str)
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), list(DEFAULTS), n=1)
            hints.append(f"{k!r} (did you mean {close[0]!r}?)" if close else repr(k))
        raise SettingsError(f"make-csv: unknown setting(s) {', '.join(hints)} in the yml section {CONFIG_SECTION} "
                            f"(configure/config.yml or local.yml); the valid keys are those of "
                            f"make_analysis_files.DEFAULTS")
    s = {**DEFAULTS, **yml}
    for flag in ("waveform", "mc", "overwrite"):
        if getattr(args, flag, False):
            s[flag] = True
    for opt in ("tot_mode", "duplicate_digis", "sample_window", "chunk_entries"):
        v = getattr(args, opt, None)
        if v is not None:
            s[opt] = v
    if getattr(args, "mc_samples", None) is not None:
        s["mc_samples_per_digi"] = args.mc_samples
    cli_toa = getattr(args, "toa_calibration", None)
    if cli_toa is not None:
        s["toa_calibration"] = cli_toa
    validate_settings(s, SETTINGS_SPEC, source="make-csv settings (configure/config.yml section make_csv and "
                                               "the command-line flags)")
    if cli_toa is not None:
        s["toa_calibration"] = "none" if str(cli_toa).strip().lower() == "none" else _cli_path(cli_toa)
    elif s.get("toa_calibration") is None:
        s["toa_calibration"] = ctx.path("toa_calibration")
    elif str(s["toa_calibration"]).strip().lower() != "none":
        s["toa_calibration"] = resolve_path(s["toa_calibration"])
    s["window"] = _parse_window(s["sample_window"])
    for key in ("chunk_entries", "events_per_chunk", "parallel_write_min_rows", "qa_bins"):
        s[key] = int(s[key])  # 5.0 in the yml is the integer 5
    if s["max_events"] is not None:
        s["max_events"] = int(s["max_events"])
    s["mc_samples_per_digi"] = _mc_samples(s["mc_samples_per_digi"])  # "auto" or an int (10.0 -> 10)
    return s


# the settings that change the csv content (stored in the provenance)
def csv_settings(settings: dict, mc: bool) -> dict:
    out = {k: _json_safe(settings[k]) for k in CSV_SETTING_KEYS}
    term = _LINE_TERMINATORS[str(settings["line_terminator"]).lower()]
    out["line_terminator"] = {"\r\n": "crlf", "\n": "lf"}.get(term, repr(term))
    out["mc"] = bool(mc)
    toa = settings["toa_calibration"]
    if toa is None or str(toa).strip().lower() == "none":
        out["toa_calibration"] = "none"
    else:
        try:
            out["toa_calibration"] = file_signature(Path(toa))
        except OSError:
            out["toa_calibration"] = {"path": str(toa), "missing": True}
    return json.loads(json.dumps(out, default=str))  # the json form (tuples -> lists) is what is stored


# copy of a syntax tree without docstrings (for the code fingerprint)
def _strip_docstrings(node: ast.AST) -> ast.AST:
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                n.body = n.body[1:] or [ast.Pass()]
    return node


# installed version of a package ('missing' if absent)
def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


# converter fingerprint: hash of the csv code + numpy / pandas / uproot versions
def converter_fingerprint(source: str | None = None) -> dict:
    """A test csv written by other converter code is not reused."""
    global _FINGERPRINT
    if source is None and _FINGERPRINT is not None:
        return dict(_FINGERPRINT)
    tree = ast.parse(source if source is not None else Path(__file__).read_text(encoding="utf-8"))
    parts, found = [], set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in CSV_CODE:
            found.add(name)
            parts.append(ast.dump(_strip_docstrings(node), include_attributes=False))
    missing = sorted(set(CSV_CODE) - found)
    if missing:  # keeps CSV_CODE honest: a renamed csv function must be renamed there too
        raise RuntimeError(f"CSV_CODE names not found in make_analysis_files.py: {missing}")
    out = {"code": hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16],
           "numpy": np.__version__, "pandas": pd.__version__, "uproot": _package_version("uproot")}
    if source is None:
        _FINGERPRINT = dict(out)
    return out


# --------------------------------------------------------------------------
# reading: event-contiguous chunks of the long (one row per sample) table
# --------------------------------------------------------------------------
# concatenate two chunks
def _concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in b}


# slice every array of a chunk
def _take(c: dict, sl) -> dict:
    return {k: v[sl] for k, v in c.items()}


# cut the long table into chunks that never split an event; stop after event_limit events
def _event_chunks(read, num_entries: int, step: int, event_limit: int | None, planned_stop: int | None,
                  stats: dict):
    stop = min(planned_stop or num_entries, num_entries)
    start, carry, n_events = 0, None, 0
    while start < stop:
        end = min(start + step, stop)
        c = read(start, end)
        stats["entries_read"] += end - start
        start = end
        if carry is not None:
            c = _concat(carry, c)
            carry = None
        ev = c["pf_event"]
        if not len(ev):
            continue
        run_starts = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]])
        last = start >= stop
        if last and event_limit and n_events + len(run_starts) < event_limit and stop < num_entries:
            stop = num_entries  # planned stop too early: keep streaming
            last = False
        complete = run_starts if last else run_starts[:-1]
        cut = len(ev) if last else int(run_starts[-1])
        if event_limit and n_events + len(complete) >= event_limit:
            k = event_limit - n_events
            cut = int(run_starts[k]) if k < len(run_starts) else len(ev)
            stats["events"] += k
            yield _take(c, slice(0, cut))
            return
        n_events += len(complete)
        stats["events"] += len(complete)
        if cut:
            yield _take(c, slice(0, cut))
        if not last:
            carry = _take(c, slice(cut, None))


# entry where the first n events end (dense files), else None
def _dense_entry_stop(tree, n_events: int, probe: int = 1_000_000) -> int | None:
    num = tree.num_entries
    head = tree["pf_event"].array(entry_stop=min(num, probe), library="np")
    change = np.flatnonzero(head[1:] != head[:-1])
    if not len(change):
        return None
    rows = int(change[0]) + 1
    stop = n_events * rows
    if num % rows or stop >= num:
        return None
    window = tree["pf_event"].array(entry_start=stop - rows, entry_stop=stop + 1, library="np")
    if np.all(window[:-1] == window[0]) and window[-1] != window[-2]:
        return stop
    return None


# open a ROOT file: HGCROC ntuple or ldmx-sw event file (highest cycle)
def _open_input(path: Path, settings: dict, quiet: bool = False):
    import uproot

    fh = uproot.open(path)
    keys = {k.split(";")[0] for k in fh.keys()}
    if settings["tree"] in keys:
        cycles = [k for k in fh.keys() if k.split(";")[0] == settings["tree"]]
        if len(cycles) > 1 and not quiet:
            log(f"  {path.name}: {len(cycles)} cycles of {settings['tree']} ({', '.join(cycles)}); reading the highest")
        return fh, "ntuple", fh[settings["tree"]]
    if settings["events_tree"] in keys:
        return fh, "events", fh[settings["events_tree"]]
    fh.close()
    raise SystemExit(f"{path}: neither '{settings['tree']}' nor '{settings['events_tree']}' found "
                     f"(keys: {sorted(keys)})")


# read an HGCROC ntuple chunk by chunk
def _iter_ntuple(tree, settings: dict, event_limit: int | None, stats: dict):
    available = set(tree.keys())
    missing = [b for b in NTUPLE_BRANCHES if b not in available and b != "i_sample"]
    if missing:
        raise SystemExit(f"tree {settings['tree']} lacks branches {missing}")
    branches = [b for b in NTUPLE_BRANCHES if b in available]
    if settings["tot_mode"] == "tot_comp":
        if "tot_comp" not in available:
            raise SystemExit("--tot-mode tot_comp needs the tot_comp branch")
        branches.append("tot_comp")
    planned = None
    if event_limit:
        planned = _dense_entry_stop(tree, event_limit)
        stats["dense_entry_stop"] = planned

    # read entries a..b
    def read(a, b):
        return tree.arrays(branches, entry_start=a, entry_stop=b, library="np")

    yield from _event_chunks(read, tree.num_entries, settings["chunk_entries"], event_limit, planned, stats)


# HcalDigis_<pass> branch of an event file
def _digi_prefix(tree, settings: dict) -> str:
    prefix = settings["digi_branch"]
    top = list(tree.keys(recursive=False))
    if prefix == "auto":
        cands = [k for k in top if k.startswith("HcalDigis_")]
        if not cands:
            raise SystemExit(f"no HcalDigis_<pass> branch in {settings['events_tree']}")
        return cands[0]
    if prefix not in top:
        raise SystemExit(f"digi_branch {prefix!r} not found in {settings['events_tree']} (branches: {top})")
    return prefix


# branches read from an event file
def _event_file_names(prefix: str) -> list[str]:
    return [f"{prefix}/channelIDs_", f"{prefix}/samples_", f"{prefix}/numSamplesPerDigi_", f"{prefix}/version_",
            "EventHeader/eventNumber_"]


# read an ldmx-sw event file (MC): unpack the 32-bit HGCROC sample words
def _iter_event_file(tree, settings: dict, event_limit: int | None, stats: dict):
    """Word: bit 31 TOT in progress, bit 30 TOT complete, bits 29-20 ADC(t-1), 19-10 ADC(t), 9-0 TOA."""
    import awkward as ak

    prefix = _digi_prefix(tree, settings)
    names = _event_file_names(prefix)
    stop = min(tree.num_entries, event_limit) if event_limit else tree.num_entries
    step = max(1, int(settings["events_per_chunk"]))
    stats["digi_branch"] = prefix
    log(f"  ldmx-sw event file: unpacking {prefix} (MC mode, TOT = ADC field of TOT-complete samples)")
    for a in range(0, stop, step):
        b = min(a + step, stop)
        arr = tree.arrays(names, entry_start=a, entry_stop=b, library="ak")
        stats["entries_read"] += b - a
        stats["events"] += b - a
        n_s = np.unique(ak.to_numpy(arr[names[2]]))
        if len(n_s) != 1:
            raise SystemExit(f"numSamplesPerDigi_ is not constant: {n_s}")
        n_s = int(n_s[0])
        version = np.unique(ak.to_numpy(arr[names[3]]))
        if np.any(version != 0):
            stats.setdefault("warnings", []).append(
                f"HGCROC digi version {version.tolist()} (layout checked for 0 only)")
        ids = ak.to_numpy(ak.flatten(arr[names[0]])).astype(np.uint32)
        words = ak.to_numpy(ak.flatten(arr[names[1]])).astype(np.uint32)
        if len(words) != n_s * len(ids):
            raise SystemExit(f"{prefix}: samples_ length != numSamplesPerDigi_ x channelIDs_")
        per_event = ak.to_numpy(ak.num(arr[names[0]]))
        event = np.repeat(ak.to_numpy(arr[names[4]]).astype(np.int64), per_event)
        adc = ((words >> 10) & 0x3FF).astype(np.int32)
        tot_complete = ((words >> 30) & 1).astype(bool)
        zeros = np.zeros(len(words), np.int32)
        yield {
            "pf_event": np.repeat(event, n_s),
            "layer": np.repeat(((ids >> 8) & 0xFF).astype(np.int32), n_s),
            "strip": np.repeat((ids & 0xFF).astype(np.int32), n_s),
            "end": np.repeat(((ids >> 19) & 1).astype(np.int32), n_s),
            "i_sample": np.tile(np.arange(n_s, dtype=np.int32), len(ids)),
            "adc": adc,
            "tot": np.where(tot_complete, adc, 0).astype(np.int32),  # data-like convention (unverified for MC)
            "toa": (words & 0x3FF).astype(np.int32),
            "pf_spill": zeros,
            "pf_ticks": zeros,
            "tot_comp": tot_complete,  # same schema as an ntuple chunk read with --tot-mode tot_comp
        }


# --------------------------------------------------------------------------
# aggregation of the time samples of every digi
# --------------------------------------------------------------------------
# (event, layer, strip, end) -> one int64 key
def _pack_key(c: dict) -> np.ndarray:
    ev = c["pf_event"].astype(np.int64)
    layer = c["layer"]
    strip = c["strip"]
    if len(ev) and (ev.min() < 0 or ev.max() >= 1 << 46 or layer.min() < 0 or layer.max() >= ID_RANGE
                    or strip.min() < 0 or strip.max() >= ID_RANGE):
        raise SystemExit("pf_event / layer / strip out of the supported range (event >= 0, layer and strip < 256)")
    return (ev << 17) | (layer.astype(np.int64) << 9) | (strip.astype(np.int64) << 1) | c["end"].astype(np.int64)


# both readers deliver the same fields
def _check_chunk_schema(c: dict, settings: dict, source: str) -> None:
    missing = [k for k in CHUNK_FIELDS if k not in c]
    if settings["tot_mode"] == "tot_comp" and "tot_comp" not in c:
        missing.append("tot_comp")
    if missing:
        raise SystemExit(f"{source}: the reader delivered no {missing} (needed with --tot-mode {settings['tot_mode']})")


# TOT samples used, by -tot-mode (tot_branch | tot_comp | tot_ne_adc)
def _tot_source(c: dict, tot_mode: str) -> np.ndarray:
    if tot_mode == "tot_comp":
        if "tot_comp" not in c:
            raise SystemExit("--tot-mode tot_comp: this input carries no TOT-complete flag (tot_comp)")
        return np.where(c["tot_comp"], c["tot"], 0)
    if tot_mode == "tot_ne_adc":
        return np.where(c["tot"] != c["adc"], c["tot"], 0)
    return c["tot"]


# per-digi variables, fast path (every n consecutive rows are one digi): adc_sum, adc_max, tot, toa, raw sums
def _digis_regular(c: dict, n_samples: int, key: np.ndarray, settings: dict, waveform: bool, stats: dict):
    """None if the chunk is not regular (then _digis_generic)."""
    n_rows = len(key)
    if n_samples <= 0 or n_rows % n_samples:
        return None
    blocks = key.reshape(-1, n_samples)
    if not np.all(blocks == blocks[:, :1]):
        return None
    if "i_sample" in c:
        mismatch = int(np.count_nonzero((c["i_sample"].reshape(-1, n_samples) != np.arange(n_samples)).any(1)))
        stats["i_sample_mismatch_digis"] += mismatch
    lo, hi = settings["window"]
    cols = slice(lo, hi)
    adc = c["adc"].reshape(-1, n_samples)[:, cols]
    tot = _tot_source(c, settings["tot_mode"]).reshape(-1, n_samples)[:, cols]
    toa = c["toa"].reshape(-1, n_samples)[:, cols]
    n_digi, n_used = adc.shape
    rows = np.arange(n_digi)
    tot_nonzero = tot != 0
    tot_count = tot_nonzero.sum(1)
    tot_value = tot[rows, tot_nonzero.argmax(1)]
    toa_nonzero = toa != 0
    toa_count = toa_nonzero.sum(1)
    toa_k = toa_nonzero.argmax(1)
    digis = {
        "key": blocks[:, 0],
        "event": c["pf_event"][::n_samples].astype(np.int64),
        "layer": c["layer"][::n_samples],
        "strip": c["strip"][::n_samples],
        "end": c["end"][::n_samples],
        "spill": c["pf_spill"][::n_samples],
        "ticks": c["pf_ticks"][::n_samples],
        "n_samples": np.full(n_digi, n_used, np.int64),
        "adc_sum": adc.sum(1, dtype=np.int64),  # adc_sum = Σ a[k] (including pedestal)
        "adc_max": adc.max(1) if n_used else np.zeros(n_digi, np.int64),  # adc_max = max a[k]
        "tot": select_single_tot(tot_value, tot_count),
        "toa_count": toa_count,
        "toa_raw": np.where(toa_count == 1, toa[rows, toa_k], 0),
        "toa_k": toa_k + lo,  # position in the digi (from sample 0), also with --sample-window
    }
    # raw words of the used samples, before any TOT/TOA rule
    digis["raw_tot_sum"] = c["tot"].reshape(-1, n_samples)[:, cols].sum(1, dtype=np.int64)  # Σ t[k], raw
    digis["raw_toa_sum"] = toa.sum(1, dtype=np.int64)  # Σ q[k], raw
    if waveform:
        wf = np.zeros((n_digi, N_WAVEFORM_SAMPLES), adc.dtype)
        m = min(n_used, N_WAVEFORM_SAMPLES)
        wf[:, :m] = adc[:, :m]
        digis["waveform"] = wf
    return digis


# per-digi variables, generic path (any row order, grouped by key): as _digis_regular
def _digis_generic(c: dict, key: np.ndarray, settings: dict, waveform: bool, stats: dict) -> dict:
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    n = len(key_sorted)
    # a new digi (readout) starts where the key changes or i_sample does not increase (file order)
    new_readout = np.r_[True, key[1:] != key[:-1]]
    if "i_sample" in c:
        new_readout[1:] |= c["i_sample"][1:] <= c["i_sample"][:-1]
    readout_id = np.cumsum(new_readout)[order]
    new_group = np.r_[True, key_sorted[1:] != key_sorted[:-1]]
    starts = np.flatnonzero(new_group)
    group = np.cumsum(new_group) - 1
    pos = np.arange(n) - starts[group]
    new_in_group = np.r_[False, readout_id[1:] != readout_id[:-1]] & ~new_group
    n_readouts = 1 + np.add.reduceat(new_in_group.astype(np.int64), starts)  # > 1: repeated digi
    lo, hi = settings["window"]
    if lo or hi is not None:
        keep = (pos >= lo) & ((pos < hi) if hi is not None else True)
        order, key_sorted, group, pos = order[keep], key_sorted[keep], group[keep], pos[keep] - lo
        if not len(key_sorted):
            return None
        starts = np.flatnonzero(np.r_[True, key_sorted[1:] != key_sorted[:-1]])
        n_readouts = n_readouts[group[starts]]
    n = len(key_sorted)
    if not n:
        return None
    n_used = np.diff(np.r_[starts, n])
    adc = c["adc"][order]
    tot = _tot_source(c, settings["tot_mode"])[order]
    toa = c["toa"][order]
    big = np.iinfo(np.int64).max
    tot_nonzero = tot != 0
    tot_count = np.add.reduceat(tot_nonzero.astype(np.int64), starts)
    tot_first = np.minimum.reduceat(np.where(tot_nonzero, pos, big), starts)
    tot_value = np.zeros(len(starts), tot.dtype)
    has = tot_count > 0
    tot_value[has] = tot[starts[has] + tot_first[has]]
    toa_nonzero = toa != 0
    toa_count = np.add.reduceat(toa_nonzero.astype(np.int64), starts)
    toa_first = np.minimum.reduceat(np.where(toa_nonzero, pos, big), starts)
    toa_k = np.where(toa_count > 0, toa_first, 0)
    single = toa_count == 1
    toa_raw = np.zeros(len(starts), toa.dtype)
    toa_raw[single] = toa[starts[single] + toa_k[single]]
    first = order[starts]
    if "i_sample" in c:  # repeated digis restart i_sample by construction: not counted here
        mismatch = np.add.reduceat((c["i_sample"][order] != pos + lo).astype(np.int64), starts) > 0
        stats["i_sample_mismatch_digis"] += int(np.count_nonzero(mismatch & (n_readouts == 1)))
    digis = {
        "key": key_sorted[starts],
        "event": c["pf_event"][first].astype(np.int64),
        "layer": c["layer"][first],
        "strip": c["strip"][first],
        "end": c["end"][first],
        "spill": c["pf_spill"][first],
        "ticks": c["pf_ticks"][first],
        "n_samples": n_used.astype(np.int64),
        "adc_sum": np.add.reduceat(adc.astype(np.int64), starts),  # adc_sum = Σ a[k] (including pedestal)
        "adc_max": np.maximum.reduceat(adc, starts),  # adc_max = max a[k]
        "tot": select_single_tot(tot_value, tot_count),
        "toa_count": toa_count,
        "toa_raw": toa_raw,
        "toa_k": toa_k + lo,  # position in the digi (from sample 0), also with --sample-window
        "n_readouts": n_readouts,
    }
    # raw words of the used samples, before any TOT/TOA rule
    digis["raw_tot_sum"] = np.add.reduceat(c["tot"][order].astype(np.int64), starts)  # Σ t[k], raw
    digis["raw_toa_sum"] = np.add.reduceat(toa.astype(np.int64), starts)  # Σ q[k], raw
    if waveform:
        wf = np.zeros((len(starts), N_WAVEFORM_SAMPLES), adc.dtype)
        for p in range(N_WAVEFORM_SAMPLES):
            ok = n_used > p
            wf[ok, p] = adc[starts[ok] + p]
        digis["waveform"] = wf
    return digis


# select rows of every array
def _subset(d: dict, mask) -> dict:
    return {k: v[mask] for k, v in d.items()}


# --------------------------------------------------------------------------
# pairing of the two ends and the per-bar result store
# --------------------------------------------------------------------------
# (layer, strip) -> bar index; the 192 bars first, unknown bars appended
class BarSlots:

    # index the 192 prototype bars
    def __init__(self):
        self.lut = np.full((ID_RANGE, ID_RANGE), -1, np.int64)
        bars = geo.all_bars()
        self.layer = bars["layer"].tolist()
        self.strip = bars["strip"].tolist()
        self.lut[self.layer, self.strip] = np.arange(len(self.layer))

    # number of bars
    @property
    def n(self) -> int:
        return len(self.layer)

    # bar index of every (layer, strip), adding unknown bars
    def of(self, layer, strip) -> np.ndarray:
        slot = self.lut[layer, strip]
        if np.any(slot < 0):
            new = np.unique(np.stack([layer[slot < 0], strip[slot < 0]], 1), axis=0)
            for lay, st in new:
                self.lut[lay, st] = self.n
                self.layer.append(int(lay))
                self.strip.append(int(st))
            slot = self.lut[layer, strip]
        return slot

    # bar indices in output order: layer, then strip
    def ordered(self) -> list[int]:
        return sorted(range(self.n), key=lambda s: (self.layer[s], self.strip[s]))


# smallest integer dtype that holds the values (saves memory)
def _compact(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind not in "iu" or not a.size:
        return a
    lo, hi = int(a.min()), int(a.max())
    for dt in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return a.astype(dt, copy=False)
    return a


# one chunk of paired rows, grouped by bar
@dataclass
class _Chunk:
    offsets: np.ndarray  # rows of slot s are [offsets[s], offsets[s+1])
    data: dict


# all paired (event, bar) rows of one ROOT file, kept in memory until the csv is written
@dataclass
class BarStore:

    slots: BarSlots = field(default_factory=BarSlots)
    chunks: list = field(default_factory=list)
    monotonic: bool = True
    last_event: int | None = None
    toa_any: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))       # any TOA sample, either end
    toa_single: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), bool))  # a digi with exactly one
    short: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))         # a digi with < 8 samples
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    # extend the per-bar flags to new bars
    def _grow(self):
        n = self.slots.n
        if len(self.toa_any) < n:
            pad = n - len(self.toa_any)
            self.toa_any = np.r_[self.toa_any, np.zeros(pad, bool)]
            self.toa_single = np.r_[self.toa_single, np.zeros((pad, 2), bool)]
            self.short = np.r_[self.short, np.zeros(pad, bool)]
            self.rows = np.r_[self.rows, np.zeros(pad, np.int64)]

    # per-bar flags: any TOA hit, single TOA per end, digi with < 8 samples
    def add_flags(self, slot, digis):
        self._grow()
        self.toa_any[slot[digis["toa_count"] > 0]] = True
        single = digis["toa_count"] == 1
        self.toa_single[slot[single], digis["end"][single]] = True
        self.short[slot[digis["n_samples"] < N_WAVEFORM_SAMPLES]] = True

    # add the paired rows of one chunk
    def add(self, counts: np.ndarray, data: dict):
        if not len(data["event"]):
            return
        lo, hi = int(data["event"].min()), int(data["event"].max())
        if self.last_event is not None and lo <= self.last_event:
            self.monotonic = False
        self.last_event = hi if self.last_event is None else max(self.last_event, hi)
        self._grow()
        self.rows[:len(counts)] += counts
        self.chunks.append(_Chunk(np.r_[0, np.cumsum(counts)], {k: _compact(v) for k, v in data.items()}))

    # all rows of one bar in event order (None if it has none)
    def bar_rows(self, slot: int) -> dict | None:
        pieces = [(ch.data, slice(ch.offsets[slot], ch.offsets[slot + 1])) for ch in self.chunks
                  if slot + 1 < len(ch.offsets) and ch.offsets[slot + 1] > ch.offsets[slot]]
        if not pieces:
            return None
        out = {k: np.concatenate([data[k][sl] for data, sl in pieces]) for k in pieces[0][0]}
        if not self.monotonic:
            order = np.argsort(out["event"], kind="stable")
            out = {k: v[order] for k, v in out.items()}
        return out


# pair end 0 with end 1 per (event, bar), inner join (a missing end drops the row); pf_spill, pf_ticks of end 1
def _pair_ends(digis: dict, slots: BarSlots) -> tuple[np.ndarray, dict, np.ndarray, int, int]:
    """Returns (rows per bar, row arrays, bar of every digi, unpaired, duplicates)."""
    ev = digis["event"]
    if len(ev) and np.all(ev[1:] >= ev[:-1]):
        change = np.r_[True, ev[1:] != ev[:-1]]
        ev_idx = np.cumsum(change) - 1
        ev_values = ev[change]
    else:
        ev_values, ev_idx = np.unique(ev, return_inverse=True)
    slot = slots.of(digis["layer"], digis["strip"])
    nb = slots.n
    cell = (ev_idx * nb + slot) * 2 + digis["end"]
    n_cells = len(ev_values) * nb * 2
    counts = np.bincount(cell, minlength=n_cells)
    n_dup = int(np.count_nonzero(counts[cell] > 1))
    digi_at = np.full(n_cells, -1, np.int64)
    digi_at[cell] = np.arange(len(ev))
    digi_at = digi_at.reshape(len(ev_values), nb, 2)
    d0 = digi_at[:, :, 0].T
    d1 = digi_at[:, :, 1].T
    paired = (d0 >= 0) & (d1 >= 0)
    n_unpaired = int(np.count_nonzero((d0 >= 0) ^ (d1 >= 0)))
    s_idx, e_idx = np.nonzero(paired)
    i0 = d0[s_idx, e_idx]
    i1 = d1[s_idx, e_idx]
    # pf_spill, pf_ticks of end 1
    rows = {"event": ev_values[e_idx], "spill": digis["spill"][i1], "ticks": digis["ticks"][i1]}
    for e, idx in ((0, i0), (1, i1)):
        for name in ("tot", "toa_raw", "toa_k", "adc_sum", "adc_max", "n_samples", "raw_tot_sum", "raw_toa_sum"):
            rows[f"{name}{e}"] = digis[name][idx]
        if "waveform" in digis:
            rows[f"waveform{e}"] = digis["waveform"][idx]
    return paired.sum(1), rows, slot, n_unpaired, n_dup


# --------------------------------------------------------------------------
# conversion driver
# --------------------------------------------------------------------------
# one conversion: one ROOT file -> one csv (+ its QA pdf)
@dataclass
class Job:
    key: str                           # provenance 'dataset': the run key, or the csv stem of an -input file
    label: str                         # QA title
    root: Path                         # the ROOT file
    out_csv: Path
    mc: bool
    event_limit: int | None
    ds: object = None                  # the run (config.Dataset); None for an -input file
    qa_pdf: Path | None = None         # qa unit: the default name, then its destination (route_qa)
    qa_saved: bool = True              # False: the QA pdf is the unsaved scratch copy (full set, no --output)
    warning: str | None = None         # run csv from part of the events / other settings (formal_csv_warning)

    # the job in messages: '<run folder>/<csv name>', or the csv name of an -input file
    @property
    def tag(self) -> str:
        return self.out_csv.name if self.ds is None else f"{Path(self.ds.folder).name}/{self.out_csv.name}"

    # the job in sentences: 'muon 4 (fixed_Muon_4GeV.root)', or the ROOT file name
    @property
    def name(self) -> str:
        return Path(self.root).name if self.ds is None else f"{run_ref(self.ds)} ({Path(self.root).name})"


# empty conversion statistics
def _new_stats() -> dict:
    return {"entries_read": 0, "events": 0, "digis": 0, "unpaired_digis": 0, "duplicate_digis": 0,
            "i_sample_mismatch_digis": 0, "irregular_digis": 0, "other_end_rows": 0, "generic_chunks": 0,
            "samples_per_digi": None, "tot_equals_adc_fraction": None, "warnings": []}


# samples per digi of a chunk
def _samples_per_digi(c: dict) -> int:
    if "i_sample" in c and len(c["i_sample"]):
        return int(c["i_sample"].max()) + 1
    key = _pack_key(c)
    change = np.flatnonzero(key[1:] != key[:-1])
    return int(change[0]) + 1 if len(change) else len(key)


# check -mc-samples, -sample-window, -waveform against the samples per digi
def _check_sample_counts(n_samples: int, name: str, mc: bool, settings: dict) -> int:
    """Returns the number of samples used."""
    want = str(settings["mc_samples_per_digi"]).strip().lower()
    if mc and want != "auto" and int(want) != n_samples:
        raise SystemExit(f"--mc-samples {want}: {name} has {n_samples} samples per digi")
    lo, hi = settings["window"]
    n_window = len(range(n_samples)[lo:hi])
    if not n_window:
        raise SystemExit(f"--sample-window {settings['sample_window']}: selects no sample of the {n_samples}-sample "
                         f"digis of {name}")
    if settings["waveform"] and n_window < N_WAVEFORM_SAMPLES:
        raise SystemExit(f"--waveform needs at least 8 samples per digi inside the sample window ({name}: "
                         f"{n_window} of {n_samples})")
    return n_window


# checks on the first chunk of a file (samples per digi, MC tot == adc warning)
def _check_first_chunk(c: dict, path: Path, kind: str, mc: bool, settings: dict, stats: dict) -> int:
    _check_chunk_schema(c, settings, path.name)
    n_samples = _samples_per_digi(c)
    stats["samples_per_digi"] = n_samples
    _check_sample_counts(n_samples, path.name, mc, settings)
    if not mc and n_samples != DATA_SAMPLES_PER_DIGI:
        stats["warnings"].append(f"{path.name}: {n_samples} samples per digi (data have 8): looks like MC, "
                                 "consider --mc")
    if mc and kind == "ntuple":
        nz = c["adc"] != 0
        frac = float(np.mean(c["tot"][nz] == c["adc"][nz])) if nz.any() else 0.0
        stats["tot_equals_adc_fraction"] = frac
        if settings["tot_mode"] == "tot_branch" and frac > 0.5:
            stats["warnings"].append(
                f"MC tot branch equals adc in {100 * frac:.1f}% of non-zero samples: the single-TOT rule "
                "gives tot_end* = 0 for every multi-sample signal (see --tot-mode tot_comp / tot_ne_adc)")
    return n_samples


# generic path + the repeated-digi rule (-duplicate-digis error | drop)
def _generic_digis(c: dict, key: np.ndarray, n_samples: int, settings: dict, waveform: bool,
                   stats: dict) -> dict | None:
    stats["generic_chunks"] += 1
    digis = _digis_generic(c, key, settings, waveform, stats)
    if digis is None:
        return None
    repeated = digis.pop("n_readouts") > 1
    if repeated.any():
        n_rep = int(np.count_nonzero(repeated))
        stats["duplicate_digis"] += n_rep
        if settings["duplicate_digis"] == "error":
            i = int(np.flatnonzero(repeated)[0])
            raise SystemExit(
                f"{n_rep} (event, layer, strip, end) keys carry more than one digi (first: event "
                f"{int(digis['event'][i])} L{int(digis['layer'][i])} S{int(digis['strip'][i])} "
                f"end {int(digis['end'][i])}; e.g. the MC digi-ID bug). Repeated digis are never merged: "
                "use --duplicate-digis drop to leave them out.")
        digis = _subset(digis, ~repeated)
    expected = len(range(n_samples)[settings["window"][0]:settings["window"][1]])
    stats["irregular_digis"] += int(np.count_nonzero(digis["n_samples"] != expected))
    return digis


# one chunk: per-digi variables -> pair the ends -> store
def _process_chunk(c: dict, n_samples: int, settings: dict, waveform: bool, store: BarStore, stats: dict):
    ends_ok = (c["end"] == 0) | (c["end"] == 1)
    if not ends_ok.all():  # a bar has end 0 and end 1 only
        stats["other_end_rows"] += int(np.count_nonzero(~ends_ok))
        c = _take(c, ends_ok)
    if not len(c["end"]):
        return
    key = _pack_key(c)
    digis = _digis_regular(c, n_samples, key, settings, waveform, stats)
    generic = digis is None
    if generic:
        digis = _generic_digis(c, key, n_samples, settings, waveform, stats)
        if digis is None:
            return
    counts, rows, slot, n_unpaired, n_dup = _pair_ends(digis, store.slots)
    if n_dup and not generic:  # repeated digis in regular blocks: the generic path applies -duplicate-digis
        digis = _generic_digis(c, key, n_samples, settings, waveform, stats)
        if digis is None:
            return
        counts, rows, slot, n_unpaired, n_dup = _pair_ends(digis, store.slots)
    if n_dup:
        raise SystemExit(f"{n_dup} digis share an (event, bar, end) after aggregation (internal error)")
    stats["digis"] += len(digis["event"])
    stats["unpaired_digis"] += n_unpaired
    store.add_flags(slot, digis)
    store.add(counts, rows)


PEEK_ENTRIES = 65_536  # entries read per ntuple by validate_inputs (> 21 dense 8-sample data events)


# samples per digi from the first digis of a file
def _peek_samples_per_digi(tree, kind: str, settings: dict) -> int | None:
    n = min(int(tree.num_entries), PEEK_ENTRIES if kind == "ntuple" else 1)
    if n <= 0:
        return None
    if kind == "events":
        import awkward as ak

        name = _event_file_names(_digi_prefix(tree, settings))
        arr = tree.arrays(name, entry_start=0, entry_stop=n, library="ak")
        values = np.unique(ak.to_numpy(arr[name[2]]))
        return int(values[0]) if len(values) == 1 else None  # not constant: the reader stops with a message
    available = set(tree.keys())
    branches = ["i_sample"] if "i_sample" in available else ["pf_event", "layer", "strip", "end"]
    c = tree.arrays(branches, entry_start=0, entry_stop=n, library="np")
    return _samples_per_digi(c)


# open the ROOT file of a job; check reader, options and sample counts before converting
def validate_inputs(job: Job, settings: dict) -> str:
    """Returns the reader kind (ntuple or events)."""
    f = Path(job.root)
    fh, kind, tree = _open_input(f, settings, quiet=True)
    with fh:
        if kind == "ntuple":
            available = set(tree.keys())
            missing = [b for b in NTUPLE_BRANCHES if b not in available and b != "i_sample"]
            if missing:
                raise SystemExit(f"{f.name}: tree {settings['tree']} lacks branches {missing}")
            if settings["tot_mode"] == "tot_comp" and "tot_comp" not in available:
                raise SystemExit(f"{f.name}: --tot-mode tot_comp needs the tot_comp branch")
        else:
            if settings["tot_mode"] == "tot_ne_adc":
                raise SystemExit(f"{f.name}: {TOT_NE_ADC_EVENTS}")
            _digi_prefix(tree, settings)
        n_samples = _peek_samples_per_digi(tree, kind, settings)
        if n_samples is not None:
            _check_sample_counts(n_samples, f.name, job.mc or kind == "events", settings)
    return kind


# warnings for options without effect on this job
def option_warnings(settings: dict, job: Job, kind: str) -> list[str]:
    mc = job.mc or kind == "events"
    out = []
    if settings["tot_mode"] == "tot_ne_adc" and not mc:
        out.append("--tot-mode tot_ne_adc is meant for flat MC ntuples whose tot branch repeats adc; on this data "
                   "ntuple every TOT sample equal to its ADC sample is set to 0 before the single-TOT rule")
    if str(settings["mc_samples_per_digi"]).strip().lower() != "auto" and not mc:
        out.append(f"--mc-samples {settings['mc_samples_per_digi']} is checked only in MC mode (-mc, kind: mc or "
                   f"an ldmx-sw event file): ignored for this job")
    return out


# read the ROOT file of a job into a BarStore (nothing written)
def convert(job: Job, settings: dict) -> tuple[BarStore, dict]:
    store = BarStore()
    stats = _new_stats()
    waveform = bool(settings["waveform"])
    f = Path(job.root)
    fh, kind, tree = _open_input(f, settings)
    mc = job.mc or kind == "events"
    stats["mc"] = mc
    with fh:
        if kind == "events" and settings["tot_mode"] == "tot_ne_adc":
            raise SystemExit(f"{f.name}: {TOT_NE_ADC_EVENTS}")
        stats["input_entries"] = int(tree.num_entries)
        if kind == "events":
            chunks = _iter_event_file(tree, settings, job.event_limit, stats)
        else:
            chunks = _iter_ntuple(tree, settings, job.event_limit, stats)
        n_samples = None
        for c in chunks:
            if n_samples is None:
                n_samples = _check_first_chunk(c, f, kind, mc, settings, stats)
            _process_chunk(c, n_samples, settings, waveform, store, stats)
    return store, stats


# beam-csv columns of one bar: adc_mean = adc_sum / n, toa [ns] via toa_to_ns
def _bar_columns(rows: dict, layer: int, strip: int, cal: ToaCalibration, toa_float: tuple[bool, bool],
                 waveform: bool) -> dict:
    n = len(rows["event"])
    bx, offset, _ = cal.constants(layer, strip)
    out = {"pf_event": rows["event"].astype(np.int64)}
    for e in (0, 1):
        toa = toa_to_ns(rows[f"toa_raw{e}"], rows[f"toa_k{e}"], bx, offset if e == 1 else None)
        adc_sum = rows[f"adc_sum{e}"].astype(np.int64)
        out[f"tot_end{e}"] = rows[f"tot{e}"].astype(np.int64)
        out[f"toa_end{e}"] = toa if toa_float[e] else toa.astype(np.int64)
        out[f"adc_sum_end{e}"] = adc_sum
        out[f"adc_mean_end{e}"] = adc_sum.astype(np.float64) / rows[f"n_samples{e}"].astype(np.float64)  # adc_sum / n
        out[f"adc_max_end{e}"] = rows[f"adc_max{e}"].astype(np.int64)
        if waveform:
            wf = rows[f"waveform{e}"]
            for k in range(N_WAVEFORM_SAMPLES):
                out[f"adc_{k}_end{e}"] = wf[:, k].astype(np.int64)
        if e == 0:
            out["layer"] = np.full(n, layer, np.int64)
            out["strip"] = np.full(n, strip, np.int64)
            out["pf_spill"] = rows["spill"].astype(np.int64)
            out["pf_ticks"] = rows["ticks"].astype(np.int64)
    for name in ("raw_tot_sum", "raw_toa_sum"):
        for e in (0, 1):
            out[f"{name}_end{e}"] = rows[f"{name}{e}"].astype(np.int64)
    return out


# bars written to the csv (layer -> strip); a bar with TOA hits but no TOA-table row is dropped (as Joe Muse)
def _written_bars(store: BarStore, cal: ToaCalibration, settings: dict, waveform: bool, stats: dict) -> list[int]:
    keep, dropped = [], []
    for s in store.slots.ordered():
        if s >= len(store.rows) or store.rows[s] == 0:
            continue
        layer, strip = store.slots.layer[s], store.slots.strip[s]
        if not cal.has_row[layer, strip] and store.toa_any[s]:
            if settings["missing_toa_calibration"] == "error":
                raise SystemExit(f"L{layer} S{strip} has TOA hits but no row in {cal.source}")
            dropped.append(f"L{layer}S{strip} (no TOA-table row)")
            continue
        if waveform and store.short[s]:
            dropped.append(f"L{layer}S{strip} (digi with < 8 samples, waveform mode)")
            continue
        keep.append(s)
    if dropped:
        stats["warnings"].append(f"dropped {len(dropped)} whole bars: {', '.join(dropped)}")
    stats["dropped_bars"] = dropped
    return keep


# csv text of one bar (pandas formatting, floats as numpy str)
def _csv_text(cols: dict, columns: list[str], term: str) -> str:
    frame = {}
    for c in columns:
        v = cols[c]
        if v.dtype.kind == "f" and len(v):
            uniq, inverse = np.unique(v, return_inverse=True)
            v = uniq.astype(str)[inverse]
        frame[c] = v
    return pd.DataFrame(frame, columns=columns).to_csv(None, header=False, index=False, lineterminator=term)


# write the csv in row order (layer, strip, pf_event); returns the rows
def write_beam_csv(store: BarStore, path: Path, cal: ToaCalibration, settings: dict, stats: dict,
                   workers: int = 1) -> int:
    waveform = bool(settings["waveform"])
    bars = _written_bars(store, cal, settings, waveform, stats)
    toa_float = tuple(bool(store.toa_single[bars, e].any()) if bars else False for e in (0, 1))
    columns = beam_csv_columns(waveform)
    term = _LINE_TERMINATORS[str(settings["line_terminator"]).lower()]
    path = ensure_writable(Path(path))  # refuses safety.protected_dirs, creates the folder
    n_rows = 0

    # (rows, columns) of every written bar
    def payloads():
        for s in bars:
            rows = store.bar_rows(s)
            if rows is not None:
                yield len(rows["event"]), _bar_columns(rows, store.slots.layer[s], store.slots.strip[s], cal,
                                                       toa_float, waveform)

    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(",".join(columns) + term)
            if workers > 1:
                from collections import deque
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    pending = deque()
                    for n, cols in payloads():
                        pending.append(pool.submit(_csv_text, cols, columns, term))
                        n_rows += n
                        if len(pending) >= 2 * workers:  # bounded look-ahead keeps memory flat
                            fh.write(pending.popleft().result())
                    while pending:
                        fh.write(pending.popleft().result())
            else:
                for n, cols in payloads():
                    fh.write(_csv_text(cols, columns, term))
                    n_rows += n
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return n_rows


# write csv + provenance json as one set: csv to a temp file, renamed, json written last
def commit_beam_csv(store: BarStore, out_csv: Path, cal: ToaCalibration, settings: dict, stats: dict,
                    record: Callable[[int, dict], dict], workers: int = 1) -> tuple[int, dict]:
    """A failed run never leaves a json that describes another csv."""
    out_csv = Path(out_csv)
    with ProductSet(provenance_path(out_csv)) as ps:
        tmp = ps.stage(out_csv)
        n_rows = write_beam_csv(store, tmp, cal, settings, stats, workers=workers)
        st = tmp.stat()
        output = {"path": canonical_path(out_csv), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        written = ps.commit(record(n_rows, output))
    if file_signature(out_csv) != output:  # a rename keeps size and mtime on NTFS / POSIX; say so if not
        log(f"WARNING: {out_csv.name} changed its signature when renamed into place; its provenance "
            f"'output' block does not describe it (the test-subset builder will not trust it)")
    return n_rows, written


# --------------------------------------------------------------------------
# QA plots: one multipage pdf per csv (based on Joe Muse's makeAnalysisFiles.make_plots, revised by Lei)
# --------------------------------------------------------------------------
QA_LAYER_HISTS = [("adc_sum_end0", "ADC sum end 0"), ("toa_end0", "TOA end 0 [ns]"), ("tot_end0", "TOT end 0")]
QA_BAR_PAIRS = [
    ("adc_sum_end0", "tot_end0", "ADC sum end 0", "TOT end 0"),
    ("adc_sum_end0", "toa_end0", "ADC sum end 0", "TOA end 0 [ns]"),
    ("tot_end0", "toa_end0", "TOT end 0", "TOA end 0 [ns]"),
    ("adc_sum_end0", "adc_sum_end1", "ADC sum end 0", "ADC sum end 1"),
    ("toa_end0", "toa_end1", "TOA end 0 [ns]", "TOA end 1 [ns]"),
    ("tot_end0", "tot_end1", "TOT end 0", "TOT end 1"),
]
QA_COLUMNS = ["adc_sum_end0", "adc_sum_end1", "toa_end0", "toa_end1", "tot_end0", "tot_end1"]


# QA data per bar from the in-memory store
def _qa_bars_from_store(store: BarStore, cal: ToaCalibration, settings: dict):
    bars = _written_bars(store, cal, settings, bool(settings["waveform"]), {"warnings": []})
    for s in bars:
        rows = store.bar_rows(s)
        layer, strip = store.slots.layer[s], store.slots.strip[s]
        cols = _bar_columns(rows, layer, strip, cal, (True, True), False)
        yield layer, strip, {c: np.asarray(cols[c], float) for c in QA_COLUMNS}


# QA data per bar from an existing csv
def _qa_bars_from_csv(path: Path, chunksize: int):
    parts: dict = {}
    dtypes = {"layer": "float64", "strip": "float64", **{c: "float64" for c in QA_COLUMNS}}
    for chunk in pd.read_csv(path, usecols=["layer", "strip"] + QA_COLUMNS, dtype=dtypes, chunksize=int(chunksize),
                             float_precision="round_trip"):
        chunk = validate_channel_keys(chunk, keys=("layer", "strip"), unique=False, source=f"beam csv {path}")
        for (layer, strip), g in chunk.groupby(["layer", "strip"], sort=False):
            parts.setdefault((int(layer), int(strip)), []).append({c: g[c].to_numpy(float) for c in QA_COLUMNS})
    for (layer, strip) in sorted(parts):
        p = parts.pop((layer, strip))
        yield layer, strip, {c: np.concatenate([x[c] for x in p]) for c in QA_COLUMNS}


# QA pdf: per layer ADC / TOA / TOT histograms, per bar 2-d correlations (TOA fired / not)
def make_qa_pdf(bar_iter, pdf_path: Path, title: str, info: list[str], settings: dict) -> int:
    """Returns the page count."""
    from matplotlib.colors import LogNorm

    from ..common import plotting as pl
    from ..common.io_utils import atomic_pdf

    bins, cmin, cmap = int(settings["qa_bins"]), float(settings["qa_cmin"]), settings["qa_cmap"]
    selections = (("TOA fired", lambda d: (d["toa_end0"] > 0) & (d["toa_end1"] > 0)),
                  ("TOA not fired", lambda d: (d["toa_end0"] == 0) & (d["toa_end1"] == 0)))
    pages = 0

    # one page per layer: end-0 histograms of every bar
    def layer_page(pdf, layer, bars):
        fig, axes = pl.new_figure(2, 3, figsize=(15, 9))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.91, wspace=0.25, hspace=0.3)
        for r, (sel_name, sel) in enumerate(selections):
            for col, (var, label) in enumerate(QA_LAYER_HISTS):
                ax = axes[r, col]
                for strip, d in bars:
                    v = d[var][sel(d)]
                    if len(v):
                        counts, edges = np.histogram(v, bins=bins)
                        ax.stairs(counts, edges, label=f"Bar {strip}")
                ax.set_yscale("log")
                ax.set_xlabel(label)
                ax.set_ylabel("Events")
                ax.set_title(f"{sel_name}, layer {layer}")
                if ax.has_data():
                    ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{title}: layer {layer}")
        pl.save(fig, pdf)

    # one page per bar: six 2-d correlations
    def bar_page(pdf, layer, strip, d):
        fig, axes = pl.new_figure(4, 3, figsize=(12, 15))
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.04, top=0.94, wspace=0.35, hspace=0.35)
        for r, (sel_name, sel) in enumerate(selections):
            m = sel(d)
            for i, (xv, yv, xl, yl) in enumerate(QA_BAR_PAIRS):
                ax = axes[2 * r + i // 3, i % 3]
                x, y = d[xv][m], d[yv][m]
                h = np.histogram2d(x, y, bins=bins)[0] if len(x) else np.zeros((1, 1))
                if (h >= cmin).any():  # hist2d look (bins=100, cmin=0.01, cmap='jet', LogNorm)
                    xe = (x.min(), x.max()) if x.max() > x.min() else (x.min() - 0.5, x.min() + 0.5)
                    ye = (y.min(), y.max()) if y.max() > y.min() else (y.min() - 0.5, y.min() + 0.5)
                    ax.imshow(np.where(h >= cmin, h, np.nan).T, origin="lower", aspect="auto", extent=(*xe, *ye),
                              cmap=cmap, norm=LogNorm(vmin=max(cmin, 1.0), vmax=h.max()), interpolation="nearest")
                else:
                    ax.text(0.5, 0.5, "no entries", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_title(f"{sel_name} ({int(m.sum())})", fontsize=9)
        fig.suptitle(f"{title}: layer {layer} bar {strip}")
        pl.save(fig, pdf)

    with atomic_pdf(pdf_path) as pdf:
        pl.text_page(pdf, f"make-csv QA: {title}", info)
        pages += 1
        current, bars = None, []
        for layer, strip, d in bar_iter:
            if current is not None and layer != current:
                layer_page(pdf, current, bars)
                pages += 1
                for s, dd in bars:
                    bar_page(pdf, current, s, dd)
                    pages += 1
                bars = []
            current = layer
            bars.append((strip, d))
        if bars:
            layer_page(pdf, current, bars)
            pages += 1
            for s, dd in bars:
                bar_page(pdf, current, s, dd)
                pages += 1
    return pages


# --------------------------------------------------------------------------
# stage API
# --------------------------------------------------------------------------
# make-csv command-line options
def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("make-csv options")
    g.add_argument("--waveform", "-waveform", action="store_true",
                   help="add adc_0..adc_7 per end (per-sample ADC) [off]")
    g.add_argument("--max-events", "-max-events", type=int, default=None, metavar="N",
                   help="first N events of every ROOT file in file order, N >= 1; without -test they make the run's "
                        "csv (data/analysis_files/<run>/<ROOT name>.csv, a WARNING says so) "
                        "[all; --test: run.test_events]")
    g.add_argument("--mc", "-mc", action="store_true", help="MC mode (10-sample digis) [off]")
    g.add_argument("--mc-samples", "-mc-samples", default=None, metavar="N|auto",
                   help="time samples per MC digi, checked before conversion (MC mode only) [auto]")
    g.add_argument("--tot-mode", "-tot-mode", choices=["tot_branch", "tot_comp", "tot_ne_adc"], default=None,
                   help="TOT sample source [tot_branch]; tot_comp = TOT-complete samples only; tot_ne_adc only for "
                        "flat MC ntuples")
    g.add_argument("--duplicate-digis", "-duplicate-digis", choices=["error", "drop"], default=None,
                   help="repeated (event, layer, strip, end) digis: stop, or leave them out [error]")
    g.add_argument("--sample-window", "-sample-window", default=None, metavar="all|A:B",
                   help="aggregate only samples A..B-1 of each digi [all]")
    g.add_argument("--toa-calibration", "-toa-calibration", default=None, metavar="CSV|none",
                   help="TOA table [paths.toa_calibration]; none = no BX wrap / end-1 offset")
    g.add_argument("--overwrite", "-overwrite", action="store_true",
                   help="replace an existing output csv [off: refuse and skip]")
    g.add_argument("--input", "-input", nargs="+", default=None, metavar="ROOT",
                   help="convert these ROOT files instead of the runs: one csv per file, named after it, directly in "
                        "data/analysis_files/ (-test: data/test/analysis_files/); the later stages read only the "
                        "runs of configure/datasets.yml [none]")
    g.add_argument("--output-csv", "-output-csv", dest="output", type=_csv_file_name, default=None,
                   metavar="NAME.csv",
                   help="file name of the csv of ONE -input file, no folder: it goes to data/analysis_files/ "
                        "(-test: data/test/analysis_files/) [<input file name>.csv]; the csv of a run is always "
                        "named after its ROOT file")
    g.add_argument("--chunk-entries", "-chunk-entries", type=int, default=None, metavar="N",
                   help="ROOT entries per streamed chunk [3072000]")
    outputs.add_only_argument(p, UNITS, default=DEFAULT_UNITS)


# events to convert per ROOT file: -max-events, else run.test_events with -test, else all
def _event_limit(ctx, args, settings) -> int | None:
    if getattr(args, "max_events", None) is not None:
        validate_settings({"max_events": args.max_events}, {"max_events": "int>0"},
                          source="make-csv --max-events (leave it out for all events)")
        return int(args.max_events)
    if ctx.test:
        return int(ctx.n_test_events)
    return int(settings["max_events"]) if settings.get("max_events") else None


# -output-csv: a csv file name only (the folder is fixed by the mode)
def _csv_file_name(value: str) -> str:
    text = str(value).strip()
    if (not text or text.startswith("-") or text in (".", "..") or Path(text).name != text
            or any(c in text for c in "/\\:")):
        raise argparse.ArgumentTypeError(
            f"{value!r}: give a csv FILE NAME only (e.g. muon_12321_v2.csv), no folder and not starting with '-': "
            f"the csv goes to data/analysis_files/, or to data/test/analysis_files/ with -test")
    if not text.lower().endswith(".csv"):
        raise argparse.ArgumentTypeError(f"{value!r}: the beam csv name must end in .csv")
    return text


# command-line path -> absolute path
def _cli_path(value) -> Path:
    """abspath, not resolve(): resolve() turns paths over 260 characters into a form uproot cannot open."""
    return Path(os.path.abspath(Path(str(value)).expanduser()))


# make-csv command of a run or an -input file (for messages and plans)
def _make_command(ctx, job) -> str:
    what = run_flags(job.ds) if job.ds is not None else f"-input {job.root}"
    return f".\\runner.ps1 make-csv {what}" + (" -test" if ctx.test else "")


# refuse two jobs that write one csv (the same ROOT file twice, two -input files with one name)
def _check_targets(jobs) -> None:
    seen: dict = {}
    for job in jobs:
        key = canonical_path(job.out_csv)
        if key in seen:
            raise SystemExit(f"make-csv: {seen[key].name} and {job.name} would both write {job.out_csv}; convert "
                             f"them in separate commands (-input with -output-csv <name>.csv)")
        seen[key] = job


# jobs of a run: one per ROOT file; without the csv unit (-only qa) one per csv already in the run folder
def _run_jobs(ctx, ds, settings, limit, qa: bool) -> list[Job]:
    mc = bool(settings["mc"]) or ds.kind == "mc"
    if ds.root_files:
        pairs = [(Path(root), ctx.root_csv(ds, root)) for root in ds.root_files]
    elif ctx.wants("csv"):
        return []
    else:  # the qa unit alone: the csv files of the run folder (also of a run without ROOT file)
        pairs = [(None, ctx.test_csv(ds, csv) if ctx.test else Path(csv)) for csv in ctx.run_csvs(ds)
                 if Path(csv).exists()]
    return [Job(ds.key, f"{ds.label}: {csv.name}", root, csv, mc, limit, ds=ds,
                qa_pdf=qa_pdf_path(ctx, csv, ds) if qa else None) for root, csv in pairs]


# the jobs (one per ROOT file of every run, or per -input file), target checks, and the QA output plan
def _jobs(ctx, args, datasets, settings, dry: bool = False) -> tuple[list[Job], outputs.OutputPlan, list]:
    """Returns (jobs, QA output plan, chosen runs without a ROOT file); a real run stops on such a run."""
    limit = _event_limit(ctx, args, settings)
    make = ctx.wants("csv")
    qa = ctx.wants("qa") and ctx.make_plots
    name = getattr(args, "output", None)  # --output-csv (dest "output"); --output = ctx.outputs (QA pdfs)
    jobs, rootless = [], []
    if getattr(args, "input", None):
        files = [_cli_path(p) for p in args.input]
        if name and len(files) != 1:
            raise SystemExit(f"--output-csv names one csv, so it takes one -input file; this run converts "
                             f"{len(files)} ({', '.join(f.name for f in files)}): leave --output-csv out (each csv "
                             f"is named after its file) or convert them one by one")
        for f in files:
            csv = ctx.analysis_files_dir() / (name or root_csv_name(f))
            jobs.append(Job(csv.stem, csv.name, f, csv, bool(settings["mc"]), limit,
                            qa_pdf=qa_pdf_path(ctx, csv) if qa else None))
    else:
        if name:
            raise SystemExit("--output-csv names the csv of one -input file; the csv of a run is named after its "
                             "ROOT file (data/analysis_files/<run>/<ROOT file name>.csv)")
        explicit = getattr(args, "datasets", None) or any(getattr(args, f"select_{p}", None)
                                                          for p in SELECTABLE_PARTICLES)
        if not explicit and make:  # default (DEFAULT_DATASETS "all"): every data run whose ROOT files are here
            absent = [d for d in datasets if not d.root_files or not all(Path(p).exists() for p in d.root_files)]
            if absent and not ctx.dry_run:
                log(f"make-csv: no run chosen, so only the runs with their ROOT files on this machine are converted; "
                    f"not converted (configure/datasets.yml 'root' missing here): "
                    f"{', '.join(run_ref(d) for d in absent)}; their csv files in data/analysis_files/<run>/ are used "
                    f"as they are.  Choose a run with its flag (e.g. {run_flags(absent[0])}) to convert it anyway.")
            datasets = [d for d in datasets if d not in absent]
        for ds in datasets:
            run_jobs = _run_jobs(ctx, ds, settings, limit, qa)
            if make and not ds.root_files:
                rootless.append(ds)
            jobs += run_jobs
    if rootless and not dry:
        ds = rootless[0]
        raise SystemExit(f"make-csv: no ROOT file configured for {', '.join(run_ref(d) for d in rootless)}: set "
                         f"its 'root' in configure/datasets.yml or in configure/local.yml (datasets: "
                         f"\"{run_ref(ds)}\": root: <file>); its csv files in {ctx.run_dir(ds)} are read as they are")
    _check_targets(jobs)
    plan = route_qa(ctx, jobs, dry=dry)  # job.qa_pdf -> its destination
    for job in jobs:
        if make and not (job.out_csv.exists() and not settings["overwrite"]):  # the csv will be written
            job.warning = formal_csv_warning(ctx, job.ds, job.out_csv, settings, job.mc, job.event_limit)
            if dry and job.warning:
                log(f"WARNING: {job.warning}")
    return jobs, plan, rootless


# inputs and outputs of the run (for -dry-run, check and the run log); deletes nothing
def io_plan(ctx, args, datasets) -> list[dict]:
    settings = _settings(ctx, args)
    make = ctx.wants("csv")
    plan = []
    if make and str(settings["toa_calibration"]).lower() != "none":
        plan.append({"role": "in", "kind": "external", "path": Path(settings["toa_calibration"]),
                     "produced_by": PRODUCED_BY_TOA})
    jobs, _, rootless = _jobs(ctx, args, datasets, settings, dry=True)
    for ds in rootless:
        plan.append({"role": "in", "kind": "external", "path": Path(f"<no ROOT file set for {run_ref(ds)}>"),
                     "produced_by": PRODUCED_BY_ROOT})
    for job in jobs:
        if make:
            plan.append({"role": "in", "kind": "external", "path": Path(job.root), "produced_by": PRODUCED_BY_ROOT})
            plan.append({"role": "out", "path": job.out_csv})
            if json_records_on(provenance_path(job.out_csv)):
                plan.append({"role": "out", "path": provenance_path(job.out_csv)})
        elif job.qa_pdf:  # the qa unit alone reads the existing csv
            plan.append({"role": "in", "kind": "product", "path": job.out_csv,
                         "produced_by": _make_command(ctx, job)})
        if job.qa_pdf:
            item = {"role": "out", "path": job.qa_pdf, "result": qa_key(job.out_csv)}
            if not job.qa_saved:
                item["note"] = "not saved: no --output"
            plan.append(item)
    return plan


# value -> json-compatible value
def _json_safe(v):
    if isinstance(v, (Path,)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


# --------------------------------------------------------------------------
# provenance and the test-mode reuse rule
# --------------------------------------------------------------------------
# provenance of a csv: ROOT file, settings, code fingerprint, columns, statistics, csv signature
def _provenance_record(ctx, job: Job, settings: dict, stats: dict, n_rows: int, seconds: float,
                       output: dict) -> dict:
    import uproot

    cs = csv_settings(settings, job.mc)
    n_samples = stats.get("samples_per_digi")
    lo, hi = settings["window"]
    return {
        "format_version": CSV_FORMAT_VERSION,
        "stage": "make-csv", "dataset": job.key,
        "run_folder": None if job.ds is None else Path(job.ds.folder).name, "mode": ctx.mode, "test_mode": ctx.test,
        "output_csv": str(job.out_csv), "output": output,
        "columns": beam_csv_columns(bool(settings["waveform"])),
        "waveform": bool(settings["waveform"]),
        "input": file_signature(job.root), "event_limit": job.event_limit, "events": stats["events"],
        "rows": n_rows, "n_samples": n_samples, "n_samples_used": None if n_samples is None else len(
            range(n_samples)[lo:hi]), "mc": bool(stats.get("mc", job.mc)),
        "csv_settings": cs, "settings_hash": settings_hash(cs), "converter": converter_fingerprint(),
        "seconds": round(seconds, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {k: _json_safe(v) for k, v in stats.items()},
        "settings": {k: _json_safe(v) for k, v in settings.items()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__},
    }


# why an existing csv does not match this run (None = it matches; with -no-json and no record: not verified)
def provenance_mismatch(csv_path: Path, key: str, root, event_limit, expected_settings: dict) -> str | None:
    """Checks dataset, ROOT file, event limit, format, converter, csv settings and csv hash."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return "not made yet"
    prov = provenance_path(csv_path)
    if not prov.exists() and not json_records_on(prov):
        return None
    record = read_json(prov)
    if not isinstance(record, dict):
        return "provenance json missing or unreadable"
    if record.get("format_version") != CSV_FORMAT_VERSION:
        return f"csv format version {record.get('format_version')} != {CSV_FORMAT_VERSION}"
    if record.get("dataset") != key:
        return f"made for dataset {record.get('dataset')!r}, not {key!r}"
    made_by, now_by = record.get("converter"), converter_fingerprint()
    if made_by != now_by:
        if not isinstance(made_by, dict):
            return "made by another converter (no converter fingerprint in its provenance)"
        changed = sorted(k for k in now_by if made_by.get(k) != now_by[k])
        return f"made by another converter ({', '.join(changed)} changed since)"
    if record.get("event_limit") != event_limit:
        return f"event limit {record.get('event_limit')} != {event_limit}"
    if root is None:
        return "no ROOT file configured (cannot verify the source)"
    try:
        signature = file_signature(Path(root))
    except OSError:
        return "ROOT file not found (cannot verify the source)"
    if record.get("input") != signature:
        return "ROOT file differs (path, size or modification time)"
    if record.get("settings_hash") != settings_hash(expected_settings):
        stored = record.get("csv_settings") or {}
        diff = sorted(k for k in expected_settings if stored.get(k) != expected_settings[k])
        return f"csv settings differ ({', '.join(diff) or 'hash'})"
    ok, reason = product_set_valid(prov, members=[csv_path])
    if not ok:
        if "changed since it was committed" in reason:
            return "the csv was changed after make-csv wrote it"
        return reason
    return None


_REPORTED: set = set()


# ROOT file of a run that gives this csv name, else None
def _root_of(ds, csv: Path):
    name = os.path.normcase(Path(csv).name)
    return next((Path(r) for r in ds.root_files if os.path.normcase(root_csv_name(r)) == name), None)


# the -test csv of one csv of a run if later test stages may read it, else None (they use the subset)
def current_test_csv(ctx, ds, csv) -> Path | None:
    """csv: the full-set csv of the run (data/analysis_files/<run>/<name>.csv)."""
    csv = Path(csv)
    path = ctx.test_csv(ds, csv)
    if not path.exists():
        return None
    try:
        settings = configured_settings(ctx)
    except SystemExit as exc:
        reason = f"make_csv settings invalid: {exc}"
    else:
        expected = csv_settings(settings, bool(settings["mc"]) or ds.kind == "mc")
        reason = provenance_mismatch(path, ds.key, _root_of(ds, csv), int(ctx.n_test_events), expected)
    if reason is None:
        return path
    if (str(path), reason) not in _REPORTED:
        _REPORTED.add((str(path), reason))
        if reason.startswith("csv settings differ"):
            hint = "; to use it, set those options in the make_csv section of configure/config.yml"
        elif "cannot verify" in reason or reason.startswith("make_csv settings invalid"):
            hint = ""
        else:
            hint = f"; remake it with 'make-csv {run_flags(ds)} -test -overwrite'"
        log(f"[{run_ref(ds)}] test mode: not reading {path.name} from make-csv --test ({reason}{hint}); "
            f"using the exact first-{ctx.n_test_events}-events subset of {csv.name}")
    return None


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
# cheap checks of a job before any conversion; returns the option warnings
def _prepare(job: Job, settings: dict) -> list[str]:
    require(Path(job.root), produced_by=PRODUCED_BY_ROOT)
    kind = validate_inputs(job, settings)
    for p in (job.out_csv, provenance_path(job.out_csv), job.qa_pdf):
        if p is not None:
            ensure_writable(p)
    return option_warnings(settings, job, kind)


# log line of a QA pdf
def _qa_log(job: Job, pages: int, what: str = "QA plots") -> str:
    where = f"-> {job.qa_pdf}" if job.qa_saved else "(scratch copy, not saved)"
    return f"[{job.tag}] {what}: {pages} pages {where}"


# QA pdf of a job drawn from its existing csv
def _qa_from_csv(ctx, job: Job, settings: dict) -> None:
    pages = make_qa_pdf(_qa_bars_from_csv(job.out_csv, int(ctx.cfg.get("run", {}).get("chunksize", 2_000_000))),
                        job.qa_pdf, job.label, [f"csv: {job.out_csv.name}"], settings)
    log(_qa_log(job, pages, "QA plots from the existing csv"))


# stage entry: settings, jobs, checks, convert, write csv + provenance, QA pdfs
def run(ctx, args, datasets) -> None:
    settings = _settings(ctx, args)
    make = ctx.wants("csv")
    jobs, _, _ = _jobs(ctx, args, datasets, settings)  # builds the output plan of the QA pdfs (route_qa)
    if not jobs or not (make or any(job.qa_pdf for job in jobs)):
        log("make-csv: nothing to do (no ROOT file or csv selected, or -only qa with -no-plots)")
        return
    if not make:
        missing = [job for job in jobs if not job.out_csv.exists()]
        if missing:
            raise SystemExit(f"make-csv -only qa draws the QA pdf from the existing csv; missing: "
                             f"{', '.join(str(j.out_csv) for j in missing)}. Make it first: "
                             f"{_make_command(ctx, missing[0])}")
    plan = [(job, not make or (job.out_csv.exists() and not settings["overwrite"])) for job in jobs]
    # every check of every job before the first conversion: the TOA table, then inputs and targets
    cal = load_toa_calibration(settings["toa_calibration"]) if not all(skip for _, skip in plan) else None
    for job, skip in plan:
        if not skip:
            for w in _prepare(job, settings):
                log(f"[{job.tag}] WARNING: {w}")
            if job.warning:
                log(f"WARNING: {job.warning}")
    summary = []
    for job, skip in plan:
        if skip:
            if make:
                log(f"[{job.tag}] output exists, not overwritten: {job.out_csv}\n"
                    f"          re-run with --overwrite (or -overwrite) to replace it")
                reason = provenance_mismatch(job.out_csv, job.key, job.root, job.event_limit,
                                             csv_settings(settings, job.mc))
                if reason is not None:
                    log(f"[{job.tag}] WARNING: the existing csv does not match this run ({reason})")
            if job.qa_pdf:
                _qa_from_csv(ctx, job, settings)
            summary.append(f"{job.tag}: {'skipped (exists)' if make else 'QA from the existing csv'}")
            continue
        t0 = time.time()
        limit_txt = f"first {job.event_limit} events" if job.event_limit else "all events"
        log(f"[{job.tag}] {Path(job.root).name} ({limit_txt}{', MC' if job.mc else ''}"
            f"{', waveform' if settings['waveform'] else ''})")
        store, stats = convert(job, settings)
        t_read = time.time() - t0
        big = int(store.rows.sum()) >= int(settings["parallel_write_min_rows"])
        n_rows, _ = commit_beam_csv(
            store, job.out_csv, cal, settings, stats, workers=ctx.workers if big else 1,
            record=lambda rows, output, job=job, stats=stats, t0=t0: _provenance_record(
                ctx, job, settings, stats, rows, time.time() - t0, output))
        seconds = time.time() - t0
        for w in stats["warnings"]:
            log(f"[{job.tag}] WARNING: {w}")
        n_events = stats["events"]
        log(f"[{job.tag}] {stats['entries_read']:,} entries, {n_events:,} events, {stats['digis']:,} digis "
            f"-> {n_rows:,} rows in {seconds:.1f}s (read+aggregate {t_read:.1f}s) -> {job.out_csv}")
        if stats["unpaired_digis"]:
            log(f"[{job.tag}] {stats['unpaired_digis']:,} digis without the other end were dropped (inner join)")
        if stats["duplicate_digis"]:
            log(f"[{job.tag}] {stats['duplicate_digis']:,} repeated digis left out (--duplicate-digis drop)")
        if stats["i_sample_mismatch_digis"]:
            log(f"[{job.tag}] {stats['i_sample_mismatch_digis']:,} digis whose i_sample differs from the file order")
        if job.qa_pdf:
            pages = make_qa_pdf(_qa_bars_from_store(store, cal, settings), job.qa_pdf, job.label,
                                [f"input: {Path(job.root).name}", f"csv: {job.out_csv.name}",
                                 f"events: {n_events}", f"rows: {n_rows}"], settings)
            log(_qa_log(job, pages))
        summary.append(f"{job.tag}: {n_rows:,} rows, {n_events:,} events -> {job.out_csv}")
        del store
    log("make-csv summary:\n  " + "\n  ".join(summary))
    unsaved = [job.qa_pdf for job in jobs if job.qa_pdf is not None and not job.qa_saved]
    if unsaved:
        log(f"make-csv: final results not saved (no --output): scratch copies in {unsaved[0].parent}; "
            f"use --output [names] to save them")
