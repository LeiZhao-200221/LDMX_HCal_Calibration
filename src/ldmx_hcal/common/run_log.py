"""Run logs: a summary log, the console transcript and an index line of every command-line run, in logs/<module>/."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import importlib
import io
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter, namedtuple
from datetime import datetime
from pathlib import Path

from .config import REPO_ROOT
from .outputs import UNNAMED_RULE, UNSAVED_RULE

SRC_ROOT = REPO_ROOT / "src"
ENV_SWITCH = "LDMX_RUN_LOG"      # 0 = no logs at all, 1 = logs also outside the command line
ENV_LOGS_DIR = "LDMX_LOGS_DIR"   # overrides the logs root (paths.logs_dir)
RUNNER_DIR = "runner"
STAGE_MODULES = {                # stage -> module (the runner's STAGES, used when the runner is not loaded)
    "make-csv": "ldmx_hcal.conversion.make_analysis_files",
    "calib": "ldmx_hcal.calibration.calibration_parameters",
    "clean": "ldmx_hcal.selection.clean_dataset",
    "adcpeak": "ldmx_hcal.energy.tot_adc_peak",
    "ci": "ldmx_hcal.energy.charge_injection",
    "plots": "ldmx_hcal.visualization.multiscale_plots",
}
MODULE_DIRS = {stage: mod.rsplit(".", 1)[1] for stage, mod in STAGE_MODULES.items()}
COMMANDS = ("all", "check", "setup")
BEAM_CSV_IN_CHAIN = ("make-csv", "calib", "clean")   # runner.BEAM_CSV_IN_CHAIN (fallback copy)
CALIBRATION_READERS = ("calib", "clean", "adcpeak", "ci", "plots")
INDEX_COLUMNS = ("start", "stage", "command", "mode", "status", "duration_s", "log")
SECTIONS = ("[1] Run", "[2] Features", "[3] Inputs", "[4] Data statistics", "[5] Fit results", "[6] Outputs",
            "[7] Warnings", "[8] Bugs / errors")
LABEL_WIDTH = 56
MAX_LINES = 80                    # longest list printed in one section (inputs, outputs, warnings)
MAX_CHANNELS = 96                 # failed channels listed per status before "... and N more"
SMALL_CSV_BYTES = 20 * 1024 ** 2  # csv files up to this size are counted line by line (others: provenance)
PDF_PAGE_BYTES = 200 * 1024 ** 2  # pdf files up to this size get a page count
WARNING_RE = re.compile(r"warning|\bnote:|refus|\bstale\b|\bignored\b", re.IGNORECASE)
EXTRA_WARNING_RE = {"make_analysis_files": re.compile(r"not overwritten|skipping", re.IGNORECASE)}
CHECK_EXIT = {0: "all external inputs present", 1: "a stage cannot be loaded", 2: "an external input is missing",
              3: "products of earlier stages are missing or stale (--strict)"}

# logged: {id: exception} of the failures already logged (the reference keeps the id from being reused)
_STATE = {"enabled": False, "argv": None, "opened": 0, "logged": {}, "stack": []}


# ===================================================================================================
# switches and locations
# ===================================================================================================
# are runs logged? on under guarded() (the command line); LDMX_RUN_LOG = 0 / 1 overrides
def enabled() -> bool:
    flag = os.environ.get(ENV_SWITCH, "").strip().lower()
    if flag in ("0", "off", "false", "no"):
        return False
    if flag in ("1", "on", "true", "yes"):
        return True
    return bool(_STATE["enabled"])


# switch the logs on / off for this process
def set_enabled(on: bool) -> None:
    _STATE["enabled"] = bool(on)


# logs root: LDMX_LOGS_DIR, else paths.logs_dir, else <parent of paths.results_dir>/logs
def logs_root(cfg: dict | None) -> Path:
    if os.environ.get(ENV_LOGS_DIR):
        return Path(os.path.abspath(os.path.expanduser(os.environ[ENV_LOGS_DIR])))
    paths = (cfg or {}).get("paths") or {}
    if paths.get("logs_dir"):
        p = Path(paths["logs_dir"])
        return p if p.is_absolute() else REPO_ROOT / p
    if paths.get("results_dir"):
        return Path(paths["results_dir"]).parent / "logs"
    return REPO_ROOT / "logs"


# the log cannot be written (e.g. the logs folder is protected)
class _LogRefused(Exception):
    pass


# refuse a logs folder inside safety.protected_dirs
def _check_writable(folder: Path, cfg: dict | None) -> None:
    from . import io_utils

    dirs = list(io_utils.protected_dirs()) + [Path(d) for d in ((cfg or {}).get("safety") or {}).get(
        "protected_dirs") or [] if d]
    target = io_utils.canonical_path(folder)
    for d in dirs:
        root = io_utils.canonical_path(d)
        if target == root or target.startswith(root.rstrip("\\/") + os.sep):
            raise _LogRefused(f"the logs folder {folder} is inside the protected folder {d} (safety.protected_dirs); "
                              f"set paths.logs_dir or {ENV_LOGS_DIR} elsewhere")


# one '[run-log] WARNING' line on stderr (never raises)
def _warn(msg: str) -> None:
    try:
        print(f"[run-log] WARNING: {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a warning must never raise
        pass


# ===================================================================================================
# formatting helpers
# ===================================================================================================
# a 'not active' value with its reason, printed as it is (no source tag)
class _NA(str):
    pass


# 'NA' or 'NA (reason)'
def na(reason: str = "") -> _NA:
    return _NA(f"NA ({reason})" if reason else "NA")


# number as short text: integers plain, else 4 significant digits; None -> NA, bool -> yes / no
def _num(v, digits: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x.is_integer() and abs(x) < 1e12:
        return str(int(x))
    return f"{x:.{digits}g}"


# is the value a finite number?
def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# 'value ± error' (the error only when it is finite)
def _pm(v, err, digits: int = 4) -> str:
    return f"{_num(v, digits)} ± {_num(err, 2)}" if _finite(err) else _num(v, digits)


# a / b as a percentage
def _pct(a, b) -> str:
    try:
        return f"{100.0 * float(a) / float(b):.2f} %"
    except (TypeError, ValueError, ZeroDivisionError):
        return "NA"


# file size as text (B, KB, MB, GB)
def _size(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# one-line text cut to a length
def _short(obj, limit: int = 160) -> str:
    text = " ".join(str(obj).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


# one value as text (lists, dicts, paths, numbers)
def _atom(x) -> str:
    if x is None:
        return "null"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_atom(y) for y in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_atom(y)}" for k, y in x.items()) + "}"
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (bool, int, float)) or type(x).__module__ == "numpy":
        return _num(x, 6)
    return str(x)


# printed value of a feature
def _value_text(v) -> str | None:
    """None = not active (printed as NA)."""
    if isinstance(v, _NA):
        return str(v)
    if v is None or v is False:
        return None
    if v is True:
        return "on"
    if isinstance(v, str):
        return v if v.strip() else None
    if isinstance(v, (list, tuple, set)):
        return _atom(list(v)) if len(v) else None
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_atom(x)}" for k, x in v.items()) or None
    return _atom(v)


# '  label: value' line with an aligned label
def _kv(label: str, value, width: int = LABEL_WIDTH) -> str:
    return f"  {label:<{width}}: {value}"


# equal values? (tuples as lists, int == float)
def _same(a, b) -> bool:
    if isinstance(a, tuple):
        a = list(a)
    if isinstance(b, tuple):
        b = list(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


# path relative to the repository, else as it is
def _repo_rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError, RuntimeError):
        return str(path)


# text usable in a file name
def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "run"


# comma-separated items wrapped into indented lines
def _wrap_items(items: list[str], indent: str = "      ", width: int = 110) -> list[str]:
    lines, cur = [], indent
    for it in items:
        piece = it if cur == indent else ", " + it
        if len(cur) + len(piece) > width and cur != indent:
            lines.append(cur + ",")
            cur = indent + it
        else:
            cur += piece
    if cur.strip():
        lines.append(cur)
    return lines


# silence prints and warnings of the stage functions the logger calls
@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


# a json record as a dict, else None
def _read_json(path):
    from .io_utils import read_json

    rec = read_json(path) if path is not None else None
    return rec if isinstance(rec, dict) else None


# a small product table as a DataFrame, else None
def _csv(path, max_bytes: int = 50 * 1024 ** 2):
    import pandas as pd

    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size > max_bytes:
            return None
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001 - an unreadable table is reported as missing
        return None


# a row value, default when missing or NaN
def _val(row, col, default=None):
    try:
        v = row.get(col, default)
    except AttributeError:
        return default
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


# 'status count' pairs, most frequent first
def _counts_text(counter) -> str:
    pairs = sorted(dict(counter).items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join(f"{k} {v}" for k, v in pairs) or "none"


# the failed channels (layer/strip/end) grouped by status and reason, never all 384
def _channel_lines(df, status_col: str, ok_values, reason_col: str | None = None, label: str = "") -> list[str]:
    if df is None or status_col not in df.columns:
        return []
    bad = df[~df[status_col].astype(str).isin([str(v) for v in ok_values])]
    if bad.empty:
        return [f"    {label}failed channels: none"]
    out = [f"    {label}failed channels (layer/strip/end): {len(bad)}"]
    for status, grp in bad.groupby(bad[status_col].astype(str), sort=True):
        out.append(f"      {status} ({len(grp)}):")
        reasons = grp[reason_col].fillna("").astype(str).map(lambda x: _short(x, 110)) if reason_col else None
        groups = grp.groupby(reasons, sort=False) if reasons is not None else [("", grp)]
        for reason, sub in groups:
            if reason:
                out.append(f"        reason: {reason}")
            names = [f"L{int(row['layer'])}/S{int(row['strip'])}/E{int(row['end'])}"
                     for _, row in sub.head(MAX_CHANNELS).iterrows()]
            out += _wrap_items(names, indent="          " if reason else "        ")
            if len(sub) > MAX_CHANNELS:
                out.append(f"          ... and {len(sub) - MAX_CHANNELS} more")
    return out


# 'median x (range a - b, n values)' of the finite values
def _range_text(values, unit: str = "", digits: int = 4) -> str:
    import numpy as np

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "none finite"
    return (f"median {_num(np.median(v), digits)}{unit} (range {_num(v.min(), digits)} - {_num(v.max(), digits)}"
            f"{unit}, {v.size} values)")


# ===================================================================================================
# the run as the FEATURES / REPORTS functions see it (read-only use of the stage modules)
# ===================================================================================================
# an attribute of the loaded runner module, else None
def _runner_attr(name: str):
    for modname in ("__main__", "runner"):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, name) and hasattr(mod, "STAGES"):
            return getattr(mod, name)
    return None


# what a FEATURES / REPORTS function sees: module, ctx, args, settings, runs, console, final results
class _Run:

    # set up the run of one stage
    def __init__(self, name: str, args, ctx, direct: bool):
        self.name, self.args, self.ctx, self.direct = name, args, ctx, direct
        self.folder = MODULE_DIRS.get(name, _safe_name(name))
        stages = _runner_attr("STAGES") or STAGE_MODULES
        self.module_name = stages.get(name) or STAGE_MODULES.get(name)
        self.mod = None
        if self.module_name:
            try:
                self.mod = sys.modules.get(self.module_name) or importlib.import_module(self.module_name)
            except Exception:  # noqa: BLE001 - a broken stage is logged by the run itself
                self.mod = None
        self.s, self.s_error, self.extra = None, None, {}
        self.datasets, self.datasets_error = None, None
        self.inputs = None         # the run-choosing inputs (run_inputs), computed after resolve()
        self.names = None          # {run key: run reference} of the configuration (run_names)
        self.start_ns = time.time_ns()
        self.console = ""
        self.exc = None
        self.plan, self.plan_error = None, None
        self.written: list[Path] = []
        self.result_plans = None   # the output plans (common/outputs.py) this run made, read when it ends
        self.finals = None         # its final results (FinalResult), see final_results()
        if ctx is not None and getattr(ctx, "clean_input", False):
            chain = _runner_attr("BEAM_CSV_IN_CHAIN") or BEAM_CSV_IN_CHAIN
            self.clean = bool(direct or name not in chain)  # inside all/check these stages read the beam csv
        else:
            self.clean = False

    # -- settings / datasets --
    # ctx as the stage saw it (clean input inside 'all', its -only units), optionally as a dry run; restored after
    @contextlib.contextmanager
    def ctx_state(self, dry_run: bool | None = None):
        saved = (self.ctx.dry_run, self.ctx.clean_input)
        self.ctx.clean_input = self.clean
        if dry_run is not None:
            self.ctx.dry_run = dry_run
        try:
            with self.units_state():
                yield
        finally:
            self.ctx.dry_run, self.ctx.clean_input = saved

    # ctx with the stage's -only units (ctx.wants), restored after
    @contextlib.contextmanager
    def units_state(self):
        saved = getattr(self.ctx, "units", None)
        self.ctx.units = _chosen_units(self.args)
        try:
            yield
        finally:
            self.ctx.units = saved

    # runs and resolved settings of the stage (its own functions, silenced)
    def resolve(self) -> None:
        if self.ctx is None or self.mod is None:
            return
        self.s, self.s_error, self.extra = None, None, {}
        self.inputs = None
        with _quiet(), self.ctx_state():
            try:
                fn = _runner_attr("datasets_for")
                self.datasets = list(fn(self.ctx, self.mod, self.args) if fn else _datasets_for(self.ctx, self.mod,
                                                                                                self.args))
            except BaseException as exc:  # noqa: BLE001 - SystemExit of a bad dataset name included
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self.datasets_error = _short(f"{type(exc).__name__}: {_exc_message(exc)}")
            resolver = RESOLVERS.get(self.folder)
            if resolver is not None:
                try:
                    resolver(self)
                except BaseException as exc:  # noqa: BLE001
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    self.s_error = _short(f"{type(exc).__name__}: {_exc_message(exc)}", 300)

    # -- helpers --
    # was one of these options given on the command line? ('name?': also when False)
    def cli(self, *names) -> bool:
        for n in names:
            allow_false = n.endswith("?")
            v = getattr(self.args, n.rstrip("?"), None)
            if v is None or (v is False and not allow_false) or (isinstance(v, (list, tuple)) and not v):
                continue
            return True
        return False

    # was the file written in this run? (modified after the start)
    def fresh(self, path) -> bool:
        try:
            return path is not None and Path(path).stat().st_mtime_ns >= self.start_ns
        except OSError:
            return False

    # 'written in this run' / 'not rewritten in this run, last written <time>' / 'missing'
    def stamp(self, path) -> str:
        try:
            st = Path(path).stat()
        except (OSError, TypeError):
            return "missing"
        if st.st_mtime_ns >= self.start_ns:
            return "written in this run"
        return f"not rewritten in this run, last written {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}"

    # run keys of the stage's runs
    @property
    def keys(self) -> list[str]:
        return [d.key for d in (self.datasets or [])]


# output units of a parsed command (-only, else the stage's default units; None: the stage has no -only)
def _chosen_units(args) -> tuple | None:
    only = getattr(args, "only", None)
    return None if only is None else tuple(only)


# default output units of a stage module (DEFAULT_UNITS with its groups expanded, else every unit)
def _default_units(mod) -> tuple | None:
    units = getattr(mod, "UNITS", None)
    if units is None:
        return None
    groups = getattr(mod, "UNIT_GROUPS", None) or {}
    default = getattr(mod, "DEFAULT_UNITS", None) or units
    names = {u for d in default for u in groups.get(d, (d,))}
    return tuple(u for u in units if u in names)


# runs of a stage (copy of runner.datasets_for, used when the runner is not loaded)
def _datasets_for(ctx, mod, args):
    from .config import SELECTABLE_PARTICLES, resolve_datasets, select_by_particle

    tokens = list(getattr(args, "datasets", None) or [])  # internal: runs added by 'all'
    selections = {p: getattr(args, f"select_{p}", None) for p in SELECTABLE_PARTICLES}
    tokens += [k for k in select_by_particle(ctx.cfg, selections) if k not in tokens]
    return resolve_datasets(ctx.cfg, tokens, default=getattr(mod, "DEFAULT_DATASETS", None),
                            what=None if tokens else f"DEFAULT_DATASETS of {getattr(mod, '__name__', 'the stage')}")


# message of an exception (the code text of a SystemExit)
def _exc_message(exc) -> str:
    if isinstance(exc, SystemExit):
        return exc.code if isinstance(exc.code, str) else str(exc.code)
    return str(exc)


# ---- resolvers: the stage's own settings functions (read-only) ------------------------------------
# make-csv settings and event limit
def _resolve_make_csv(r: _Run) -> None:
    r.s = r.mod._settings(r.ctx, r.args)
    r.extra["event_limit"] = r.mod._event_limit(r.ctx, r.args, r.s)


# calib settings
def _resolve_calib(r: _Run) -> None:
    r.s = r.mod.settings_for(r.ctx, r.args)


# clean settings, unknown yml keys and jobs
def _resolve_clean(r: _Run) -> None:
    r.s, r.extra["unknown"] = r.mod.load_settings(r.ctx)
    r.extra["jobs"] = r.mod.make_jobs(r.s, r.args, r.datasets or [])


# adcpeak settings, variant, suffix, response settings and steps
def _resolve_adcpeak(r: _Run) -> None:
    r.s = r.mod.stage_settings(r.ctx, r.args)
    r.extra["variant"] = r.mod.variant_tag(r.s)
    r.extra["suffix"] = r.mod.output_suffix(r.s)
    r.extra["rs"] = r.mod.response_settings(r.ctx, r.s, r.args)
    r.extra["steps"] = r.mod.resolve_steps(r.ctx, r.s, r.datasets or [])


# ci settings, variant, response settings and s training runs
def _resolve_ci(r: _Run) -> None:
    r.s = r.mod.resolve_settings(r.ctx, r.args)
    r.extra["variant"] = r.s.get("variant_tag") or ""
    r.extra["rs"] = r.s.get("energy_response") or {}
    r.extra["training"], r.extra["training_error"] = _ci_training_runs(r)


# ({sample: runs}, error) of the ci s training (s parameter, Lei); the stage's _TRAINING_NOTES are restored after
def _ci_training_runs(r: _Run) -> tuple:
    notes = getattr(r.mod, "_TRAINING_NOTES", None)
    saved = set(notes) if isinstance(notes, set) else None
    try:
        return {x: list(r.mod._training_datasets(r.ctx, r.s, x)) for x in r.s["scale_particles"]}, None
    except BaseException as exc:  # noqa: BLE001 - SystemExit of a reference that names no run included
        if isinstance(exc, KeyboardInterrupt):
            raise
        return None, _exc_message(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
    finally:
        if saved is not None:
            notes.clear()
            notes.update(saved)


# plots settings and render processes
def _resolve_plots(r: _Run) -> None:
    r.s = r.mod.resolve_settings(r.ctx, r.args)
    r.extra["workers"] = r.mod.render_workers(r.args, r.s)


RESOLVERS = {"make_analysis_files": _resolve_make_csv, "calibration_parameters": _resolve_calib,
             "clean_dataset": _resolve_clean, "tot_adc_peak": _resolve_adcpeak, "charge_injection": _resolve_ci,
             "multiscale_plots": _resolve_plots}


# file-name variant of a stage run (the product suffix where the stage has one; calib: its -only units)
def _variant(r: _Run) -> str:
    s, f = r.s or {}, r.folder
    if f in ("tot_adc_peak", "charge_injection"):
        return str(r.extra.get("variant") or "")
    if f == "multiscale_plots" and s:
        return "_".join(x for x in (s.get("energy_source"), _plots_tag(r)) if x)
    if f == "calibration_parameters":
        units = _chosen_units(r.args)
        return "-".join(units) if units is not None and units != _default_units(r.mod) else ""
    if f == "clean_dataset":
        return "_".join(x for x in (getattr(r.args, "standard", None), getattr(r.args, "threshold_mode", None)) if x)
    return ""


# ===================================================================================================
# the runs a stage was given (Lei 2026-10-06: any run of configure/datasets.yml is accepted wherever a run is
# taken, "even if it is wrong"; the log is where a wrong choice must show).  A run is written as its reference
# ("muon 4"); the run key (muon_4GeV) names files only.  The run-choosing inputs of a stage (its runs, calib's
# source run, adcpeak's fit run, ci's s training runs) are listed in [2] with their source, noted in [7] when
# they differ from the shipped default, and printed next to the error in [8].
# ===================================================================================================
RunInput = namedtuple("RunInput", "role runs source default default_runs error given note")
INPUT_NOTE_TAIL = "the stage ran on what was given, check its results"


# {run key: run reference} of the configuration ('muon_4GeV' -> 'muon 4')
def run_names(r: _Run) -> dict:
    if r.names is None:
        try:
            from .config import dataset_registry, run_ref
            r.names = {k: run_ref(d) for k, d in dataset_registry(r.ctx.cfg).items()}
        except Exception:  # noqa: BLE001 - a hand-made context without runs: keys are shown as they are
            r.names = {}
    return r.names


# reference of a run key; a key that names no run is shown as it is
def run_name(r: _Run, key) -> str:
    return run_names(r).get(str(key), str(key))


# 'electron 0.3, electron 4' (files: each with its run folder, e.g. 'electron 4 (Electron_4GeV/)')
def _refs(runs, files: bool = False) -> str:
    from .config import run_ref

    if not runs:
        return "none"
    return ", ".join(run_ref(d) + (f" ({Path(d.folder).name}/)" if files else "") for d in runs)


# run flags as typed: '--electron 0.3 1 --pion'
def _flag_text(args) -> str:
    from .config import SELECTABLE_PARTICLES

    parts = []
    for p in SELECTABLE_PARTICLES:
        for group in getattr(args, f"select_{p}", None) or []:
            parts.append(f"--{p}" + "".join(f" {e}" for e in (group or [])))
    return " ".join(parts)


# (runs, error) of a run setting (config.resolve_runs); never raises
def _resolve_quiet(cfg, value, what: str) -> tuple:
    from .config import resolve_runs

    try:
        return list(resolve_runs(cfg, value, what)), None
    except SystemExit as exc:
        return None, _exc_message(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# a run setting as written: 'muon 4'; a list as 'pion 4, muon 4'
def _value_words(value) -> str:
    items = [value] if isinstance(value, str) or value is None else list(value)
    return ", ".join(str(v) for v in items if v is not None) or "none"


# the runs a stage works on: the run flags (+ the runs 'all' adds), else DEFAULT_DATASETS
def _stage_runs_input(r: _Run):
    if r.folder == "make_analysis_files" and getattr(r.args, "input", None):
        return None   # -input FILE: the ROOT files given replace the runs
    flags = _flag_text(r.args)
    added = [str(k) for k in getattr(r.args, "datasets", None) or []]   # internal: runs added by 'all'
    default = getattr(r.mod, "DEFAULT_DATASETS", None)
    default_runs, _ = _resolve_quiet(r.ctx.cfg, default, f"DEFAULT_DATASETS of {r.module_name}")
    words = [default] if isinstance(default, str) else list(default or [])
    default_text = ("all runs of configure/datasets.yml" if any(str(w).strip().lower() == "all" for w in words)
                    else (_refs(default_runs) if default_runs is not None else _value_words(default)))
    note = ""
    if added:
        from .config import SELECTABLE_PARTICLES, select_by_particle
        try:
            flagged = select_by_particle(r.ctx.cfg, {p: getattr(r.args, f"select_{p}", None)
                                                     for p in SELECTABLE_PARTICLES})
        except SystemExit:
            flagged = []
        extra = [run_name(r, k) for k in added if k not in flagged]
        if extra:
            note = f"{', '.join(extra)} added by 'all' (read by later stages)"
    source = "command line" if (flags or added) else "default"
    runs = None if r.datasets is None else list(r.datasets)
    return RunInput("runs", runs, source, default_text, default_runs, r.datasets_error if runs is None else None,
                    flags or default_text, note)


# calib's source run: the run flag, else calibration.source_dataset (several runs: an error)
def _calib_inputs(r: _Run) -> list:
    a, mod, cfg = r.args, r.mod, r.ctx.cfg
    default =(getattr(mod, "DEFAULTS", None) or {}).get("source_dataset")
    default_runs, _ = _resolve_quiet(cfg, default, "calibration.source_dataset (module default)")
    flags = _flag_text(a)
    if flags or r.datasets:
        source, given = "command line", flags
        runs, error = (None, r.datasets_error) if r.datasets is None else (list(r.datasets), None)
        origin = f"the run flag{'s' if len(r.datasets or []) > 1 else ''} {flags}"
    else:
        value = (r.s or {}).get("source_dataset", default)
        source, given = _source(r, "source_dataset", ()), repr(value)
        runs, error = _resolve_quiet(cfg, value, "calibration.source_dataset")
        origin = f"calibration.source_dataset = {value!r}"
    if runs is not None and len(runs) != 1:
        one = getattr(mod, "_one_run_problem", None)
        error = one(runs, origin, source == "command line") if callable(one) else \
            f"calib calibrates on one run; {origin} gives {len(runs)}"
    return [RunInput("source run", runs, source, _value_words(default), default_runs, error, given, "")]


# adcpeak's runs and fit run (-fit-dataset, else tot_adc_peak.fit_dataset) when the line part runs
def _adcpeak_inputs(r: _Run) -> list:
    out = [_stage_runs_input(r)]
    steps, s, mod = _steps(r), r.s, r.mod
    if not s or not steps.get("line"):
        return out
    default = (getattr(mod, "DEFAULTS", None) or {}).get("fit_dataset")
    default_runs, _ = _resolve_quiet(r.ctx.cfg, default, "tot_adc_peak.fit_dataset (module default)")
    cli = "fit_dataset" in (s.get("_cli") or [])
    value, source = s["fit_dataset"], ("command line" if cli else _source(r, "fit_dataset", ()))
    try:
        runs, error = [mod._fit_dataset(r.ctx, s, value)], None
    except SystemExit as exc:
        runs, error = None, _exc_message(exc)
    except Exception as exc:  # noqa: BLE001
        runs, error = None, f"{type(exc).__name__}: {exc}"
    out.append(RunInput("fit run", runs, source, _value_words(default), default_runs, error, repr(value), ""))
    return out


# ci's runs and the s training runs of every sample (s parameter, Lei): scale_training_datasets[<sample>]
def _ci_inputs(r: _Run) -> list:
    out = [_stage_runs_input(r)]
    s, mod = r.s, r.mod
    if not s or not _ci_fits_s(r):
        return out
    training, error = r.extra.get("training"), r.extra.get("training_error")
    if training is None and error is None:
        training, error = _ci_training_runs(r)
    yml = (r.ctx.stage_cfg(getattr(mod, "CONFIG_SECTION", "charge_injection")) or {}).get("scale_training_datasets")
    for sample in s["scale_particles"]:
        value = (s.get("scale_training_datasets") or {}).get(sample)
        default = ((getattr(mod, "DEFAULTS", None) or {}).get("scale_training_datasets") or {}).get(sample)
        default_runs, _ = _resolve_quiet(r.ctx.cfg, default, f"charge_injection.scale_training_datasets.{sample}")
        source = "yml" if isinstance(yml, dict) and sample in yml and not _same(yml[sample], default) else "default"
        runs = None if training is None else training.get(sample)
        out.append(RunInput(f"training runs ({sample} sample)", runs, source, _value_words(default), default_runs,
                            error, repr(value), ""))
    return out


# does this ci call fit s (an s scale reconstructed, or the parameters unit)? never with -from-saved
def _ci_fits_s(r: _Run) -> bool:
    s = r.s or {}
    return not s.get("from_saved") and bool(r.mod.needs_scale(s))


RUN_INPUTS = {"make_analysis_files": lambda r: [_stage_runs_input(r)], "calibration_parameters": _calib_inputs,
              "clean_dataset": lambda r: [_stage_runs_input(r)], "tot_adc_peak": _adcpeak_inputs,
              "charge_injection": _ci_inputs, "multiscale_plots": lambda r: [_stage_runs_input(r)]}


# run-choosing inputs of the stage (RunInput), once per resolve(); never raises
def run_inputs(r: _Run) -> list:
    if r.inputs is not None:
        return r.inputs
    fn = RUN_INPUTS.get(r.folder)
    out = []
    if fn is not None and r.ctx is not None and r.mod is not None:
        try:
            with _quiet():
                out = [ri for ri in fn(r) if ri is not None]
        except Exception as exc:  # noqa: BLE001 - one unreadable input never hides the rest of the log
            out = [RunInput("runs", None, "?", "?", None, f"cannot be read: {type(exc).__name__}: {_short(exc, 160)}",
                            "?", "")]
    r.inputs = out
    return out


# does the input name other runs than its default?
def input_differs(ri) -> bool:
    if ri.runs is None or ri.default_runs is None:
        return False
    return sorted(d.key for d in ri.runs) != sorted(d.key for d in ri.default_runs)


# 'electron 4 (fixed_Electron_4GeV.csv)  [command line] (default muon 4)' or 'NOT RESOLVED: ...'
def input_text(ri, with_error: bool = True) -> str:
    tag = f"  [{ri.source}]"
    if ri.runs is None:
        return f"NOT RESOLVED: {ri.given}{tag}: {_short(ri.error or '?', 400)}"
    text = _refs(ri.runs, files=ri.role != "runs" and len(ri.runs) <= 3) + tag
    if ri.note:
        text += f" + {ri.note}"
    if input_differs(ri):
        text += f" (default {ri.default})"
    if ri.error and with_error:
        text += f"; the stage stops: {_short(ri.error, 400)}"
    return text


# [7] 'input: ...' notes: a role input (not 'runs') that differs from its default
def input_notes(r: _Run) -> list[str]:
    out = []
    for ri in run_inputs(r):
        if ri.role != "runs" and input_differs(ri):
            tail = f"the stage stops on it: {_short(ri.error, 300)}" if ri.error else INPUT_NOTE_TAIL
            out.append(f"input: {r.name} {ri.role} = {_refs(ri.runs)}  [{ri.source}] (default {ri.default}); {tail}")
    return out


# [8] the input runs of the stage with their roles, printed next to its error
def input_lines(r: _Run) -> list[str]:
    inputs = run_inputs(r) if r is not None else []
    if not inputs:
        return []
    lines = ["  input runs of this stage (as given; compare them with the error above):"]
    lines += [f"    {ri.role:<28}: {input_text(ri, with_error=False)}" for ri in inputs]
    return lines


# ===================================================================================================
# final results (Lei 2026-10-05, common/outputs.py): where THIS run put them.  A real run: every slot of the
# output plans it made (outputs.plans(), reset by stage_log when the stage starts); a dry run makes no plan:
# the final-result items ("result" key) of the stage's io_plan.
# ===================================================================================================
FinalResult = namedtuple("FinalResult", "key kind path saved rule default what plan")
TEST_NAMES_TEXT = "test names (results/test/test*.csv/pdf)"


# forget the output plans of an earlier stage (outputs.reset()); never raises
def reset_output_plans() -> None:
    from . import outputs

    try:
        outputs.reset()
    except Exception:  # noqa: BLE001 - the logger never breaks a run
        pass


# output plans made since the last reset (the plans of this stage)
def _run_plans() -> list:
    from . import outputs

    try:
        return list(outputs.plans())
    except Exception:  # noqa: BLE001
        return []


# rule of an io_plan note ending in [rule]
def _rule_of(line: str) -> str:
    m = re.search(r"\[([^\[\]]+)\]\s*$", str(line or ""))
    return m.group(1) if m else ""


# would an io_plan final-result item be saved?
def _item_saved(item: dict) -> bool:
    if "saved" in item:
        return bool(item["saved"])
    return "not saved" not in str(item.get("note") or "")


# rule of a saved io_plan final-result item: its note, else the --output names of ctx
def _item_rule(item: dict, ctx=None) -> str:
    rule = _rule_of(item.get("note"))
    if rule or ctx is None:
        return rule
    names = getattr(ctx, "outputs", None)
    if names is None:
        return "test name" if getattr(ctx, "test", False) else ""
    return "--output name" if names else "default name (--output)"


# rule of an unsaved result: no --output at all, or not named in --output
def _unsaved_rule(ctx) -> str:
    return UNNAMED_RULE if getattr(ctx, "outputs", None) else UNSAVED_RULE


# 'no --output' or 'not named in --output': why a result was not saved
def _unsaved_why(ctx) -> str:
    return "not named in --output" if getattr(ctx, "outputs", None) else "no --output"


# plan-display note of a final-result item: 'final result <key> [<rule>]' or 'not saved'
def final_result_note(item: dict, ctx=None) -> str:
    """'' for an item that is not a final result."""
    key = item.get("result") if isinstance(item, dict) else None
    if not key:
        return ""
    if not _item_saved(item):
        return f"final result {key}; not saved: {_unsaved_why(ctx)}"
    rule = _item_rule(item, ctx)
    return f"final result {key}" + (f" [{rule}]" if rule else "")


# final results of the run (FinalResult), in the order of its plans, computed once
def final_results(r: "_Run") -> list:
    if r.finals is not None:
        return r.finals
    out = []
    if r.ctx is not None and getattr(r.ctx, "dry_run", False):
        for it in r.plan or []:
            if it.get("role") == "out" and it.get("result"):
                p = Path(it["path"])
                saved = _item_saved(it)
                out.append(FinalResult(str(it["result"]), p.suffix.lower().lstrip("."), p, saved,
                                       _item_rule(it, r.ctx) if saved else _unsaved_rule(r.ctx), None, "",
                                       None))
    else:
        for plan in r.result_plans or []:
            try:
                for slot in plan.slots:
                    out.append(FinalResult(slot.key, slot.kind, Path(plan.path(slot.key)), bool(plan.saved(slot.key)),
                                           plan.rule(slot.key), Path(slot.default), slot.what, plan))
            except Exception:  # noqa: BLE001 - a plan that cannot be read is left out
                continue
    r.finals = out
    return out


# the FinalResult with this key, or None
def final_result(r: "_Run", key: str):
    return next((fr for fr in final_results(r) if fr.key == key), None)


# where THIS run put a final result, else fallback() (the default path)
def result_path(r: "_Run", key: str, fallback=None):
    fr = final_result(r, key)
    if fr is not None:
        return fr.path
    return fallback() if callable(fallback) else fallback


# 'saved: <rule>' or 'NOT SAVED (no --output): scratch copy' / 'NOT SAVED (not named in --output): ...'
def saved_text(fr) -> str:
    if fr.saved:
        return f"saved: {fr.rule}" if fr.rule else "saved"
    rule = fr.rule or UNSAVED_RULE
    return "NOT SAVED" + rule[len("not saved"):] if rule.startswith("not saved") else f"NOT SAVED: {rule}"


# folder name of the stage's final results
def _results_stage_dir(r: "_Run") -> str:
    for plan in r.result_plans or []:
        return plan.stage_dir
    if r.folder == "make_analysis_files":
        return str(getattr(r.mod, "QA_DIR", None) or "make_csv_qa")
    return str(getattr(r.mod, "STAGE_DIR", None) or r.folder)


# cache/<stage_dir>/unsaved/: the scratch copies of a full-set run without --output
def scratch_dir(ctx, stage_dir: str) -> Path:
    from . import outputs

    return Path(ctx.cache_dir(stage_dir, outputs.UNSAVED_DIR))


# value of 'final results (--output)': NA (not saved) | default names | names | test names
def outputs_text(ctx, scratch) -> str:
    names = getattr(ctx, "outputs", None)
    if names is None:
        return TEST_NAMES_TEXT if getattr(ctx, "test", False) else na(f"not saved: scratch copies in {scratch}")
    if not names:
        return "default names (--output)"
    return "names: " + ", ".join(str(n) for n in names)


# ===================================================================================================
# [2] FEATURES: one table per module.  F(label, get, key, args):
#   get(r) -> value; None / False / "" / [] = not active (NA); na("why") = NA with a reason; True = "on".
#   key  = settings key (source "yml" when the yml differs from the module DEFAULTS; "er:<key>" = the shared
#          energy_response section), args = argparse dests (source "command line" when given; "dest?" = also
#          when given as False).  Without get: the resolved setting r.s[key].
# ===================================================================================================
F = namedtuple("F", "label get key args", defaults=(None, None, ()))

FIT_OPTION_NAMES = {   # argparse dest = settings key of --fit / --bin / --resfit (Lei 2026-10-02)
    "fit": "fit_ranges",
    "bin": "fit_bins",
    "resfit": "resolution_fit_range",
}


# every number in a value (text, list or number)
def _numbers(v) -> list[float]:
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, str):
        return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", v)]
    if isinstance(v, dict):
        return []
    try:
        out = []
        for x in v:
            out += _numbers(x)
        return out
    except TypeError:
        return []


# the resolved energy-response settings of the run (adcpeak: extra rs, ci: s['energy_response'])
def _response_settings(r: _Run) -> dict:
    rs = r.extra.get("rs")
    if isinstance(rs, dict) and rs:
        return rs
    return (r.s or {}).get("energy_response") or {}


# value of --fit / --bin / --resfit from args or settings (None when absent)
def fit_option(r: _Run, which: str):
    name = FIT_OPTION_NAMES[which]
    v = getattr(r.args, name, None)
    if v is not None and not isinstance(v, bool):
        return v
    for d in (r.s or {}, _response_settings(r)):
        v = d.get(name) if isinstance(d, dict) else None
        if v is not None and not isinstance(v, bool) and _numbers(v):
            return v
    return None


# printed fit range: NA (<automatic>) | [a, b] <one> | [a, b], [c, d] <two>
def fit_range_feature(r: _Run, which: str, na_text: str, one: str = "single Gaussian",
                      two: str | None = "double Gaussian"):
    v = fit_option(r, which)
    if v is None:
        return na(na_text)
    x = _numbers(v)
    if len(x) == 2:
        return f"[{_num(x[0])}, {_num(x[1])}] {one}"
    if len(x) == 4 and two:
        return f"[{_num(x[0])}, {_num(x[1])}], [{_num(x[2])}, {_num(x[3])}] {two}"
    return f"{_atom(v)} (as given)"


# printed --bin: N or N1,N2; NA = the bins of the figure
def fit_bins_feature(r: _Run):
    v = fit_option(r, "bin")
    if v is None:
        return na("figure bins")
    return ",".join(_num(n) for n in _numbers(v))


# lines of a -linfit value: 'electron [0, 1] GeV; pion every point', 'none' for {}
def linfit_text(fits: dict) -> str:
    if not fits:
        return "none (points only)"
    parts = []
    for particle, rng in fits.items():
        x = _numbers(rng)
        parts.append(f"{particle} [{_num(x[0])}, {_num(x[1])}] GeV" if len(x) == 2 else f"{particle} every point")
    return "; ".join(parts)


# printed --linfit: the straight lines of the run (one per particle), else the default of the stage / scales
def linfit_feature(r: _Run, scales=None):
    """scales: the ci scales of the run (their default lines); None: adcpeak (no default line)."""
    v = getattr(r.args, "linearity_fits", None)
    if v is None:
        v = _response_settings(r).get("linearity_fits")
    if isinstance(v, dict):
        return linfit_text(v)
    if v is not None:
        return f"{_atom(v)} (as given)"
    if not scales:
        return na("no fit: points only")
    from ..energy import energy_response as er

    return na("default per scale: " + ", ".join(
        f"{p}: {linfit_text(er.scale_linearity_fits({}, r.mod.path_scale(p)))}" for p in scales))


# printed --resfit: '[a, b] GeV', else NA
def resfit_feature(r: _Run, na_text: str):
    v = fit_option(r, "resfit")
    x = _numbers(v) if v is not None else []
    if len(x) == 2:
        return f"[{_num(x[0])}, {_num(x[1])}] GeV (resolution fitted on these points only)"
    return na(na_text) if v is None else f"{_atom(v)} (as given)"


# ---- common options of every stage (runner add_common) ----
# runs of the stage by reference, with their source
def _f_datasets(r: _Run):
    if r.folder == "calibration_parameters":
        return na("the run flags choose calib's source run, see 'source run'")
    if r.folder == "make_analysis_files" and getattr(r.args, "input", None):
        return na("-input: the ROOT files given replace the runs")
    if r.datasets_error:
        return f"not resolved ({r.datasets_error})"
    if not r.datasets:
        return na("no run")
    ri = next((x for x in run_inputs(r) if x.role == "runs"), None)
    if ri is None:   # a stage the log does not know
        given = r.cli("datasets") or bool(_flag_text(r.args))
        return ", ".join(run_name(r, k) for k in r.keys) + ("  [command line]" if given else "  [default]")
    if r.folder == "make_analysis_files" and ri.source == "default" and r.ctx.wants("csv"):  # make-csv's own default
        runs = [d for d in r.datasets if d.root_files and all(Path(p).exists() for p in d.root_files)]
        if not runs:
            return na("no run of configure/datasets.yml has its ROOT file on this machine")
        return _refs(runs) + "  [default: the runs whose ROOT file is on this machine]"
    return input_text(ri)


# output units of the run (-only), with their source; NA for a stage without -only
def _f_units(r: _Run):
    units = _chosen_units(r.args)
    if units is None:
        return na("this stage has no -only")
    default = _default_units(r.mod)
    tag = "default" if default is not None and units == default else "command line"
    return f"{', '.join(units) or 'none'}  [{tag}]"


# clean-input feature
def _f_clean(r: _Run):
    if r.clean:
        test = "test/" if getattr(r.ctx, "test", False) else ""
        return f"on (reads data/{test}clean_dataset/<run>/<csv stem>_clean.csv of every csv)"
    if getattr(r.ctx, "clean_input", False):
        return na("-clean-input given; this stage runs on the beam csv inside 'all'")
    return None


# from-saved feature
def _f_from_saved(r: _Run):
    if getattr(r.args, "from_saved", False):
        test = "test/" if getattr(r.ctx, "test", False) else ""
        return f"on (restart from cache/{test}{getattr(r.mod, 'STAGE_DIR', r.folder)}/)"
    return None


# json records next to the products (-no-json switches them off)
def _f_json(r: _Run):
    if getattr(r.ctx, "write_json", True):
        return "on"
    return na("-no-json: no record json in data/, calibration/, results/; the cache records stay")


# where the final results go (from ctx.outputs and the mode)
def _f_outputs(r: _Run):
    return outputs_text(r.ctx, scratch_dir(r.ctx, _results_stage_dir(r)))


COMMON_FEATURES = (
    F("runs (--electron / --muon / --pion)", _f_datasets),
    F("output units (-only)", _f_units),
    F("test mode (-test, -test-events)", lambda r: f"first {r.ctx.n_test_events} events of every input"
      if r.ctx.test else None),
    F("dry run (-dry-run)", lambda r: "on (plan only, nothing executed)" if r.ctx.dry_run else None),
    F("clean input (-clean-input)", _f_clean),
    F("from saved (-from-saved)", _f_from_saved),
    F("figures (-no-plots switches them off)", lambda r: "on" if r.ctx.make_plots else na("-no-plots")),
    F("json records (-no-json switches them off)", _f_json),
    F("final results (--output)", _f_outputs),
    F("workers (-workers)", lambda r: r.ctx.workers, None, ("workers",)),
)


# yml settings (config.yml + local.yml) that differ from the module DEFAULTS
def _yml_differences(r: _Run) -> list[str]:
    out = []
    sections = [(getattr(r.mod, "CONFIG_SECTION", None), getattr(r.mod, "DEFAULTS", None))]
    if r.folder in ("tot_adc_peak", "charge_injection"):
        from ..energy import energy_response as er

        sections.append((er.CONFIG_SECTION, er.DEFAULTS))
    for section, defaults in sections:
        if not section or not isinstance(defaults, dict):
            continue
        for k, v in (r.ctx.stage_cfg(section) or {}).items():
            if k not in defaults or _same(v, defaults[k]):
                continue
            if "${" in str(defaults[k]):   # a path default: compare it expanded like the yml value
                try:
                    from .config import _expand_tree
                    variables = {n: str(x) for n, x in (r.ctx.cfg.get("paths") or {}).items()}
                    expanded = _expand_tree(defaults[k], variables)
                    if _same(v, expanded) or "${" in str(expanded):
                        continue
                except Exception:  # noqa: BLE001
                    continue
            out.append(f"{section}.{k} = {_atom(v)} (default {_atom(defaults[k])})")
    return out


# ---- make-csv (conversion/make_analysis_files.py) ----
# make-csv output csv: -output-csv name (one -input file), else the naming rule
def _make_csv_output(r: _Run):
    if not r.ctx.wants("csv"):
        return na("-only without csv: the existing csv files are read for the QA pdf")
    where = r.ctx.analysis_files_dir()
    if getattr(r.args, "output", None):   # -output-csv: the csv of one -input file
        return str(where / r.args.output)
    if getattr(r.args, "input", None):
        return na(f"named after each input file: {where}{os.sep}<input name>.csv")
    return na(f"per ROOT file of a run: {where}{os.sep}<run>{os.sep}<ROOT file name>.csv")


# first paths + '(+N more)'
def _paths_text(paths, limit: int = 2) -> str:
    paths = [str(p) for p in paths]
    more = f" (+{len(paths) - limit} more, see [6])" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


# -only qa: where this run's QA pdfs went, else the default name
def _make_csv_qa(r: _Run):
    if not r.ctx.wants("qa"):
        return None
    if not r.ctx.make_plots:
        return na("-no-plots")
    finals = final_results(r)
    if finals:
        state = "" if all(fr.saved for fr in finals) else " (not saved: scratch copies)"
        return f"on -> {_paths_text([fr.path for fr in finals])}{state}"
    return f"on (default name {r.mod.qa_pdf_path(r.ctx, '<csv stem>')})"


FEATURES_MAKE_CSV = (
    F("input ROOT files (-input)", lambda r: [Path(p).name for p in r.args.input] if getattr(r.args, "input", None)
      else na("the root files of the selected runs in configure/datasets.yml")),
    F("output csv (-output-csv)", _make_csv_output),
    F("QA pdf (-only qa)", _make_csv_qa),
    F("event limit per ROOT file (-max-events / -test)", lambda r: f"first {r.extra['event_limit']} events"
      if r.extra.get("event_limit") else na("all events"), "max_events", ("max_events", "test")),
    F("waveform columns adc_0..7 (-waveform)", None, "waveform", ("waveform",)),
    F("MC mode (-mc)", None, "mc", ("mc",)),
    F("MC samples per digi (-mc-samples)", lambda r: r.s["mc_samples_per_digi"]
      if r.s["mc"] and r.s["mc_samples_per_digi"] != "auto" else na("auto: no check"), "mc_samples_per_digi",
      ("mc_samples",)),
    F("TOT sample source (-tot-mode)", None, "tot_mode", ("tot_mode",)),
    F("repeated digis (-duplicate-digis)", None, "duplicate_digis", ("duplicate_digis",)),
    F("sample window (-sample-window)", lambda r: r.s["sample_window"]
      if str(r.s["sample_window"]).lower() != "all" else na("all samples"), "sample_window", ("sample_window",)),
    F("TOA calibration (-toa-calibration)", lambda r: "none (no BX wrap, no end-1 offset)"
      if str(r.s["toa_calibration"]).lower() == "none" else str(r.s["toa_calibration"]), "toa_calibration",
      ("toa_calibration",)),
    F("bar without TOA constants", None, "missing_toa_calibration"),
    F("overwrite an existing csv (-overwrite)", None, "overwrite", ("overwrite",)),
    F("ntuple entries per chunk (-chunk-entries)", None, "chunk_entries", ("chunk_entries",)),
    F("csv line terminator", None, "line_terminator"),
)


# ---- calib (calibration/calibration_parameters.py) ----
# calib units of this run {unit: on} (-only, -no-plots)
def _calib_units(r: _Run) -> dict:
    with r.ctx_state():
        return r.mod.chosen_units(r.ctx)


# what calib does in this run: the fits and where the numbers come from
def _calib_action(r: _Run) -> str:
    units = _calib_units(r)
    fits = "pedestal + MPV fit" if r.mod.needs_mpv(units) else "pedestal fit (no MPV unit)"
    source = ("the saved histograms (-from-saved)" if getattr(r.args, "from_saved", False)
              else "the csv files of the source run")
    return f"{fits} from {source}"


# table this calib run writes, NA without a csv unit
def _calib_table(r: _Run):
    with r.ctx_state():
        path = r.mod.table_csv(r.ctx, _calib_units(r))
    return str(path) if path is not None else na("no csv unit chosen (-only)")


# pdfs this calib run draws, NA without a plot unit
def _calib_pdfs(r: _Run):
    parts = r.mod.drawn_parts(_calib_units(r))
    return ", ".join(f"{p}_fits.pdf" for p in parts) or na("no plot unit (-only / -no-plots)")


# [2] value of a run-choosing input, NA when the stage does not use one
def _input_feature(r: _Run, role: str, inactive: str):
    ri = next((x for x in run_inputs(r) if x.role == role or x.role.startswith(role + " (")), None)
    return na(inactive) if ri is None else input_text(ri)


FEATURES_CALIB = (
    F("action", _calib_action),
    F("table written", _calib_table),
    F("pdfs drawn", _calib_pdfs),
    F("source run (run flag, else source_dataset)", lambda r: _input_feature(r, "source run", "no source run"),
      "source_dataset"),
    F("pedestal selection (-pedestal-selection)", None, "pedestal_selection", ("pedestal_selection",)),
    F("pedestal anchor", None, "pedestal_anchor"),
    F("pedestal window half-width [ADC] (-pedestal-window)", None, "pedestal_window_halfwidth", ("pedestal_window",)),
    F("threshold columns written", lambda r: f"pedestal_range_with_5sigma = pedestal + "
      f"{_num(r.s['pedestal_range_nsigma'])} sigma; pedestal_range_with_20percent = "
      f"{_num(1 + float(r.s['pedestal_range_fraction']))} x pedestal"),
    F("MPV cut (-mpv-nsigma)", lambda r: f"adc_sum - pedestal > {_num(r.s['mpv_nsigma'])} x pedestal_sigma",
      "mpv_nsigma", ("mpv_nsigma",)),
    F("MPV histogram range [ADC]", None, "mpv_hist_range"),
    F("MPV final bin width [ADC] (-mpv-bin-width)", None, "mpv_final_bin_width", ("mpv_bin_width",)),
    F("MPV fit window (-mpv-window-widths)", lambda r: f"rough peak ± {_num(r.s['mpv_window_widths'])} x FWHM/2.3548",
      "mpv_window_widths", ("mpv_window_widths",)),
    F("MPV rough fit", lambda r: f"{r.s['mpv_rough_likelihood']} likelihood, start at the {r.s['mpv_rough_start']}"),
    F("fit range (--fit)", lambda r: fit_range_feature(r, "fit", "automatic window: rough peak ± K w",
                                                       "ADC above the pedestal: Landau x Gauss final-fit window, "
                                                       "every channel", None), "fit_ranges", ("fit_ranges",)),
    F("bins (--bin)", fit_bins_feature, "fit_bins", ("fit_bins",)),
    F("min entries / bins of an MPV fit", lambda r: f"{r.s['mpv_min_entries']} entries, {r.s['mpv_min_bins']} bins"),
    F("histogram cache (input of -from-saved)", lambda r: (r.s["write_histogram_cache"] and not getattr(
        r.args, "from_saved", False)) and str(r.mod.histogram_cache(r.ctx)), "write_histogram_cache"),
)


# ---- clean (selection/clean_dataset.py) ----
# hit threshold mode of the clean run: -pedestalscale / -pedestalsigma, else the yml
def _clean_threshold_mode(r: _Run) -> str:
    return getattr(r.args, "threshold_mode", None) or r.s["threshold_mode"]


FEATURES_CLEAN = (
    F("standard (-muon-standard / -otherparticle)", lambda r: getattr(r.args, "standard", None) or na(
        "per particle: " + ", ".join(f"{k} -> {v}" for k, v in (r.s["standard_by_particle"] or {}).items())),
      "standard_by_particle", ("standard",)),
    F("standard per run", lambda r: ", ".join(
        f"{run_name(r, j.ds.key)}: {j.standard}" + (f" (general rule: no {j.ds.particle} entry)" if str(
            getattr(j, "standard_source", "")).startswith("general rule") else "")
        for j in r.extra.get("jobs") or []) or None),
    F("hit threshold (-pedestalscale / -pedestalsigma)", lambda r: f"{_clean_threshold_mode(r)} = "
      f"{r.mod.THRESHOLD_COLUMNS[_clean_threshold_mode(r)]}", "threshold_mode", ("threshold_mode",)),
    F("MIP energy [MeV]", None, "mip_energy_mev"),
    F("muon layer-1 window [MeV]", lambda r: r.s["layer1_energy_window_mev"]
      if any(j.standard == "muon" for j in r.extra.get("jobs") or []) else na("no run uses the muon standard"),
      "layer1_energy_window_mev"),
    F("diagnostics: E_MIP spectrum bins [lo, hi, n]", None, "mip_spectrum_bins"),
    F("diagnostics: MIP fit range [MIP]", None, "mip_fit_range"),
    F("fit range (--fit), diagnostics only", lambda r: fit_range_feature(
        r, "fit", f"mip_fit_range {_atom(r.s.get('mip_fit_range'))}: single Gaussian + Landau",
        "single Gaussian + Landau", "double Gaussian (+ Landau on the first range)"), "fit_ranges", ("fit_ranges",)),
    F("bins (--bin), diagnostics only", fit_bins_feature, "fit_bins", ("fit_bins",)),
    F("diagnostics: event energy bin [MeV]", None, "event_energy_bin_mev"),
    F("yml keys this stage does not use", lambda r: r.extra.get("unknown") or None),
)


# ---- adcpeak (energy/tot_adc_peak.py) ----
# adcpeak steps of this run (resolve_steps: units, line, response, events, saved_events, from_saved)
def _steps(r: _Run) -> dict:
    return r.extra.get("steps") or {}


# line part of this run: line table and / or fit pdf, from the csv or the saved bins
def _adcpeak_line(r: _Run):
    steps = _steps(r)
    made = [u for u in ("line_table", "line_fits") if steps.get(u)]
    if not made:
        return na("no line unit chosen (-only)")
    return ", ".join(made) + (" from the saved bins (-from-saved)" if steps.get("from_saved") else " fitted")


# response part of this run: its units and where the events come from
def _adcpeak_response(r: _Run):
    steps = _steps(r)
    if not steps.get("response"):
        return na("no response unit chosen (-only), or no run")
    made = [u for u in ("energy_response", "energy_spectra", "linearity_resolution") if steps.get(u)]
    events = ("the saved events in the cache (-from-saved)" if steps.get("saved_events")
              else "events reconstructed in this run")
    return f"{', '.join(made)} from {events}"


# line table the event reconstruction reads: this mode's own table, NA with tot_mode none
def _adcpeak_table(r: _Run):
    if r.s["tot_mode"] == "none":
        return na("tot_mode none: no TOT line")
    if not _steps(r).get("events"):
        return na("no event reconstruction in this run")
    return str(r.mod.parameters_csv(r.ctx))


# adc_max bins of the line fit: width, count and the edges (they cover the whole fit window)
def _adcpeak_bins(r: _Run):
    edges = r.mod.fit_bin_edges(r.s)
    return (f"width {_num(r.s['fit_bin_width'])} ADC, {len(edges) - 1} bins {_num(edges[0])} - {_num(edges[-1])} "
            f"(the whole fit ADC window)")


FEATURES_ADCPEAK = (
    F("line part (line_table / line_fits)", _adcpeak_line),
    F("response part (energy_response / energy_spectra / linearity_resolution)", _adcpeak_response),
    F("fit run (-fit-dataset PARTICLE E)", lambda r: _input_feature(r, "fit run", "no line unit in this run"),
      "fit_dataset", ("fit_dataset",)),
    F("fit ADC window, inclusive (-fit-adc-window)", None, "fit_adc_window", ("fit_adc_window",)),
    F("fit TOT window, exclusive (-fit-tot-window)", None, "fit_tot_window", ("fit_tot_window",)),
    F("fit ADC bins", _adcpeak_bins, "fit_bin_width"),
    F("line table read by the events", _adcpeak_table),
    F("TOT mode (-tot-mode)", None, "tot_mode", ("tot_mode",)),
    F("usable slope window [low, high) (-slope-window)", None, "slope_window", ("slope_window",)),
    F("saturation ADC (-saturation-adc)", None, "saturation_adc", ("saturation_adc",)),
    F("merge rail ADC (-merge-rail-adc)", None, "merge_rail_adc", ("merge_rail_adc",)),
    F("end hit threshold column", None, "hit_threshold_column"),
    F("quality threshold (-quality-threshold)", lambda r: r.s.get("quality_threshold") if r.s.get("quality_threshold")
      is not None else na("no cut"), "quality_threshold", ("quality_threshold",)),
    F("spectrum fit (-spectrum-fit)", lambda r: "range (--fit)" if fit_option(r, "fit") is not None
      else r.s["spectrum_fit"], "spectrum_fit", ("spectrum_fit",)),
    F("fit range (--fit)", lambda r: fit_range_feature(r, "fit", "automatic single Gaussian"),
      "er:fit_ranges", ("fit_ranges",)),
    F("bins (--bin)", fit_bins_feature, "er:fit_bins", ("fit_bins",)),
    F("linearity lines (--linfit)", lambda r: linfit_feature(r), "er:linearity_fits", ("linearity_fits",)),
    F("resolution fit range (--resfit)", lambda r: resfit_feature(r, "every electron point; pion points drawn only"),
      "er:resolution_fit_range", ("resolution_fit_range",)),
    F("point errors (-error-prescription)", lambda r: _response_settings(r).get("error_prescription"),
      "er:error_prescription", ("error_prescription",)),
    F("scaled errors (-normalerror)", lambda r: _response_settings(r).get("normal_error"), "er:normal_error",
      ("normal_error",)),
    F("bar values in the cache (-bar-csv)", lambda r: (r.s["write_bar_csv"] and _steps(r).get("events"))
      and f"on (cache/{'test/' if r.ctx.test else ''}tot_adc_peak/<run>/<csv stem>_bars.csv)", "write_bar_csv",
      ("write_bar_csv",)),
    F("output tag (-output-tag)", lambda r: r.s.get("output_tag"), "output_tag", ("output_tag",)),
    F("variant tag (product file suffix)", lambda r: r.extra.get("variant") or na("default products, no suffix")),
)


# ---- ci (energy/charge_injection.py) ----
# spectrum fits that give the linearity points: range_fit with --fit, else the yml
def _ci_linearity_source(r: _Run):
    return r.mod.linearity_source_used(r.s)


# s training samples (s parameter, Lei), or with runs their runs; NA when no s is fitted
def _ci_training(r: _Run, runs: bool = False):
    s = r.s
    if s.get("from_saved"):
        return na("-from-saved: no s fit")
    if not _ci_fits_s(r):
        return na("no s fit in this call (no s scale, no parameters unit)")
    if runs:
        inputs = [ri for ri in run_inputs(r) if ri.role.startswith("training runs")]
        return "; ".join(f"{ri.role[len('training runs ('):-len(' sample)')]}: {input_text(ri)}" for ri in inputs)
    samples = list(s["scale_particles"])
    if len(samples) == 1:
        return samples[0]
    return (f"{', '.join(samples)} (one path s_<sample> each; s = the sample with the fewest channels that borrow "
            f"s from the prior)")


# ci checks of this call (response table, quality scan), NA when none is chosen
def _ci_checks(r: _Run):
    checks = [u for u in ("response", "quality_scan") if u in r.s["units"]]
    return ", ".join(checks) or na("only on request (-only response / quality_scan)")


FEATURES_CI = (
    F("energy scales (-adc-only / -pureci / -s)", None, "energy_paths", ("adc_only", "pureci", "s_path")),
    F("scales computed", None, "compute_paths"),
    F("selection scale (-selection-path)", lambda r: f"{r.s['selection_path_used']}" + (
        " (auto)" if r.s["selection_path"] == "auto" else ""), "selection_path", ("selection_path",)),
    F("quality filter (-quality-threshold / -no-quality-filter)", lambda r: f"bad_strip_fraction <= "
      f"{_num(r.s['quality_threshold'])}" if r.s["quality_filter"] else None, "quality_threshold",
      ("quality_threshold", "no_quality_filter")),
    F("bad-bar rule (-bad-bar-rule)", None, "bad_bar_rule", ("bad_bar_rule",)),
    F("TOT sample mode (-tot-sample-mode)", None, "tot_sample_mode", ("tot_sample_mode",)),
    F("s training samples (-scale-particle)", _ci_training, "scale_particle", ("scale_particle",)),
    F("s training runs (scale_training_datasets)", lambda r: _ci_training(r, runs=True), "scale_training_datasets"),
    F("shrinkage lambda (-shrinkage)", lambda r: r.s["shrinkage_lambda"] if _ci_fits_s(r) else na("no s fit"),
      "shrinkage_lambda", ("shrinkage",)),
    F("min exact hits of a link prior (-min-link-hits)", None, "min_link_hits", ("min_link_hits",)),
    F("audit failures (-no-audit-failures)", None, "audit_failures", ("no_audit_failures",)),
    F("refit CI map + s (-refit)", None, "refit", ("refit",)),
    F("event csv (-only events)", lambda r: "events" in r.s["units"] and (
        f"on (data/{'test/' if r.ctx.test else ''}charge_injection/<run>/<csv stem>_event_energies"
        f"{r.mod.tag_suffix(r.s.get('variant_tag'))}.csv)")),
    F("checks (-only response / quality_scan)", _ci_checks),
    F("linearity source (electron points)", _ci_linearity_source, "linearity_source"),
    F("fit range (--fit)", lambda r: fit_range_feature(r, "fit", "automatic single Gaussian"),
      "er:fit_ranges", ("fit_ranges",)),
    F("bins (--bin)", fit_bins_feature, "er:fit_bins", ("fit_bins",)),
    F("linearity lines (--linfit)", lambda r: linfit_feature(r, r.s.get("output_paths")), "er:linearity_fits",
      ("linearity_fits",)),
    F("resolution fit range (--resfit)", lambda r: resfit_feature(r, "every electron point; pion points drawn only"),
      "er:resolution_fit_range", ("resolution_fit_range",)),
    F("point errors (-error-prescription)", lambda r: _response_settings(r).get("error_prescription"),
      "er:error_prescription", ("error_prescription",)),
    F("scaled errors (-normalerror)", lambda r: _response_settings(r).get("normal_error"), "er:normal_error",
      ("normal_error",)),
    F("output tag (-output-tag)", None, "output_tag", ("output_tag",)),
    F("variant tag (product file suffix)", lambda r: r.s.get("variant_tag") or na("default products, no suffix")),
)


# ---- plots (visualization/multiscale_plots.py) ----
# producer variant tag the plots run reads (multiscale_plots.producer_tag)
def _plots_tag(r: _Run) -> str:
    with r.ctx_state():
        return str(r.mod.producer_tag(r.ctx, r.s) or "")


# where the energy values of the plots run come from (the ci / adcpeak cache)
def _plots_values(r: _Run):
    if not r.s["energy_units"]:
        return na("hitmap only: no energy values")
    test = "test/" if r.ctx.test else ""
    tag = _plots_tag(r)
    if r.s["energy_source"] == r.mod.ARBITRARY:
        suffix = f"_{tag}" if tag else ""
        return f"adcpeak cache cache/{test}tot_adc_peak/<run>/<csv stem>_events{suffix}.csv / _bars{suffix}.csv"
    return f"ci cache cache/{test}charge_injection/events{'_' + tag if tag else ''}/<run>/<csv stem>.npz / _bars.npy"


FEATURES_PLOTS = (
    F("energy scale (-energy)", None, "energy_source", ("energy",)),
    F("producer variant (-variant)", lambda r: _plots_tag(r) or na("the untagged producer files"), "variant",
      ("variant",)),
    F("energy values from", _plots_values),
    F("event picker (-picker)", lambda r: na("explicit -events") if r.s.get("explicit_events") else r.s["event_picker"],
      "event_picker", ("picker",)),
    F("explicit events (-events)", lambda r: r.s.get("explicit_events"), None, ("events",)),
    F("energy window (-energy-window)", lambda r: r.s.get("energy_window"), None, ("energy_window",)),
    F("event displays per run (-n-events)", lambda r: r.s["n_event_displays"] if "event_display" in r.s["units"]
      else na("no event_display unit"), "n_event_displays", ("n_events",)),
    F("random seed (-seed)", lambda r: r.s["random_seed"] if r.s["event_picker"] == "random"
      else na("picker is not random"), "random_seed", ("seed",)),
    F("colour scale (-color-scale, -vmax)", lambda r: r.s["color_scale"] + (
        f", vmax {_num(r.s['fixed_vmax_override'] or r.s['color_vmax_mip'])}" if r.s["color_scale"] == "fixed" else ""),
      "color_scale", ("color_scale", "vmax")),
    F("per-end displays (-per-end)", None, "per_end_displays", ("per_end",)),
    F("hit-map event displays (-hitmap-events)", lambda r: r.s["hitmap_events"] if "hitmap" in r.s["units"]
      else na("no hitmap unit"), "hitmap_events", ("hitmap_events",)),
    F("hit threshold column (hitmap)", lambda r: r.s["hit_threshold_column"] if "hitmap" in r.s["units"]
      else na("no hitmap unit"), "hit_threshold_column"),
    F("overview min bin count (-min-bin-count)", None, "min_bin_count", ("min_bin_count",)),
    F("store reduced arrays (cache)", lambda r: r.s["save_cache"] and not getattr(r.args, "from_saved", False),
      "save_cache"),
    F("render processes", lambda r: r.extra.get("workers")),
)

FEATURES = {
    "make_analysis_files": FEATURES_MAKE_CSV,
    "calibration_parameters": FEATURES_CALIB,
    "clean_dataset": FEATURES_CLEAN,
    "tot_adc_peak": FEATURES_ADCPEAK,
    "charge_injection": FEATURES_CI,
    "multiscale_plots": FEATURES_PLOTS,
}


# source of a value: command line, yml (differs from DEFAULTS) or default
def _source(r: _Run, key: str | None, args) -> str:
    if args and r.cli(*args):
        return "command line"
    if key:
        section, defaults, k = getattr(r.mod, "CONFIG_SECTION", None), getattr(r.mod, "DEFAULTS", {}), key
        if key.startswith("er:"):
            from ..energy import energy_response as er
            section, defaults, k = er.CONFIG_SECTION, er.DEFAULTS, key[3:]
        yml = r.ctx.stage_cfg(section) or {} if section else {}
        if k in yml and not _same(yml[k], defaults.get(k)):
            return "yml"
    return "default"


# one '<feature>: <value>  [source]' line per spec
def _feature_lines(r: _Run, specs) -> list[str]:
    out = []
    for f in specs:
        try:
            value = f.get(r) if f.get is not None else (r.s or {}).get(f.key)
            text = _value_text(value)
            if text is None:
                text = "NA"
            elif not isinstance(value, _NA) and (f.key or f.args) and "  [" not in text:
                text += f"  [{_source(r, f.key, f.args)}]"
        except Exception as exc:  # noqa: BLE001 - one feature that cannot be read never hides the others
            text = f"? (cannot be read: {type(exc).__name__}: {_short(exc, 80)})"
        out.append(_kv(f.label, text))
    return out


# [2] Features: common options, the module table, yml differences
def features_section(r: _Run) -> list[str]:
    with r.units_state() if r.ctx is not None else contextlib.nullcontext():
        return _features_body(r)


# the lines of [2] (inside the stage's -only units)
def _features_body(r: _Run) -> list[str]:
    lines = ["  (NA = not active; [default] / [yml] / [command line] = where the value comes from)"]
    lines += _feature_lines(r, COMMON_FEATURES)
    specs = FEATURES.get(r.folder)
    if specs is None:
        lines.append(_kv("module options", "unknown stage: no feature table"))
        return lines
    if r.s is None:
        lines.append(_kv("module options", f"NOT RESOLVED ({r.s_error or 'no settings'})"))
        return lines
    if r.s_error:
        lines.append(_kv("note", f"settings partly resolved ({r.s_error})"))
    lines += _feature_lines(r, specs)
    diffs = _yml_differences(r)
    lines.append(_kv("yml settings != module defaults", "NA (none)" if not diffs else diffs[0]))
    lines += [_kv("", d) for d in diffs[1:]]
    return lines


# ===================================================================================================
# [4] Data statistics and [5] Fit results: one REPORT function per module -> (stats lines, fit lines).
# Read only the products of THIS run (r.fresh) and their provenance records.
# ===================================================================================================
# [5] line of one spectrum fit: range, mu, sigma, sigma/mu, chi2/ndf, status
def _spectrum_line(label: str, kind, lo, hi, mu, mu_e, sig, sig_e, som, chi2, ndf, status, ok, unit: str,
                   range_text: str | None = None, som_e=None) -> str:
    rng = range_text or (f"[{_num(lo)}, {_num(hi)}]{unit}" if _finite(lo) and _finite(hi) else "range NA")
    red = f" ({_num(float(chi2) / float(ndf), 3)})" if _finite(chi2) and _finite(ndf) and float(ndf) > 0 else ""
    st = str(status or "")
    st = ("ok" if st in ("ok", "") else f"ok ({_short(st, 90)})") if ok else f"FAILED ({_short(st, 90)})"
    return (f"  {label}: {kind}, fit range {rng}: mu = {_pm(mu, mu_e)}, sigma = {_pm(sig, sig_e)}, "
            f"sigma/mu = {_pm(som, som_e)}, chi2/ndf = {_num(chi2)}/{_num(ndf)}{red}, {st}")


# 'a = x ± e, b = ...' of the finite model parameters
def _params_text(row, names) -> str:
    parts = [f"{label} = {_pm(row.get(col), row.get(err))}{unit}" for label, col, err, unit in names
             if _finite(row.get(col))]
    return ", ".join(parts) or "no parameters"


# csv / json boolean -> True / False
def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        return bool(v) and not (isinstance(v, float) and math.isnan(v))
    except (TypeError, ValueError):
        return False


# ---- make-csv ----
# [4] / [5] of make-csv: events, rows, samples, anomalies per csv written; the QA pdfs
def report_make_csv(r: _Run):
    stats, fits = [], ["  no fits in this stage (ROOT ntuple -> per-bar beam csv conversion)"]
    root = _path_key(r.ctx.analysis_files_dir())
    n = 0
    for p in r.written:
        if p.suffix.lower() != ".csv" or not _path_key(p).startswith(root):
            continue
        n += 1
        rec = _read_json(r.mod.provenance_path(p))
        if not rec or rec.get("stage") != "make-csv":
            stats.append(f"  {p.parent.name}/{p.name}: {_file_facts(p)} (no provenance record: -no-json)")
            continue
        st = rec.get("stats") or {}
        where = f"{rec['run_folder']}/{p.name}" if rec.get("run_folder") else p.name
        source = Path(str((rec.get("input") or {}).get("path", "?"))).name
        stats.append(f"  {run_name(r, rec.get('dataset'))} ({where}): {_num(rec.get('events'))} events, "
                     f"{_num(rec.get('rows'))} csv rows (bars), {_num(rec.get('n_samples'))} samples per digi "
                     f"({_num(rec.get('n_samples_used'))} used), {_num(st.get('entries_read'))} ntuple entries, "
                     f"{_num(st.get('digis'))} digis, {_num(rec.get('seconds'))} s, from {source}")
        anomalies = {k: st.get(k) for k in ("unpaired_digis", "duplicate_digis", "i_sample_mismatch_digis",
                                            "irregular_digis", "other_end_rows", "dropped_bars") if st.get(k)}
        stats.append("    anomalies: " + (", ".join(f"{k} {_atom(v)}" for k, v in anomalies.items()) or "none"))
        for w in st.get("warnings") or []:
            stats.append(f"    warning: {_short(w, 200)}")
    for line in r.console.splitlines():
        if "output exists, not overwritten" in line:
            stats.append("  " + _short(line.split("]", 1)[-1].strip(), 200)
                         + " (no conversion; -overwrite replaces it)")
    if not n:
        stats.insert(0, "  no csv converted in this run" + ("" if r.ctx.wants("csv") else
                                                            " (-only without csv: the QA pdf reads the existing csv)"))
    return stats + _figure_lines(r, label="QA pdfs (final results)"), fits


# ---- final results read where THIS run's output plan put them ----
# facts of a final result written in this run, or why it is not there
def _written_text(r: _Run, path) -> str:
    p = Path(path)
    if r.fresh(p):
        return _file_facts(p)
    if not p.exists():
        return "NOT WRITTEN in this run (missing)"
    return f"NOT WRITTEN in this run (a file of an earlier run: {r.stamp(p)})"


# [4] lines of this run's final results of a kind: pages -> destination (saved or not)
def _figure_lines(r: _Run, kind: str = "pdf", label: str = "figures (final results)") -> list[str]:
    finals = [fr for fr in final_results(r) if fr.kind == kind]
    if not finals:
        return [f"  {label}: none in this run"] if r.result_plans else []
    lines = [f"  {label}:"]
    for fr in finals:
        lines.append(f"    {fr.key}: {_written_text(r, fr.path)} -> {fr.path} ({saved_text(fr)})")
    return lines


# ---- calib ----
# [4] / [5] of calib: the table of this run and its provenance, then the pdfs
def report_calib(r: _Run):
    stats, fits = _report_calib_table(r)
    stats = list(stats) + _figure_lines(r)
    return stats, fits


# calib table statistics and fit summary: pedestal, MPV status, failed channels
def _report_calib_table(r: _Run):
    stats, fits = [], []
    ctx, mod = r.ctx, r.mod
    with r.ctx_state():
        table = mod.table_csv(ctx, mod.chosen_units(ctx))
    if table is None:
        stats.append("  no table in this run (no csv unit chosen with -only)")
        summary = next((ln.split("summary: ", 1)[1] for ln in r.console.splitlines() if "summary: " in ln), None)
        fits.append("  fits drawn only (no table written)" + (f"; {summary}" if summary else ""))
        return stats, fits
    if not r.fresh(table):
        stats.append(f"  calibration table not written in this run: {table} ({r.stamp(table)})")
        fits.append("  no fit results (no table written)")
        return stats, fits
    rec = _read_json(mod.provenance_json(table)) or {}
    df = _csv(table)
    stats.append(_kv("table", f"{table.name} ({rec.get('table', '?')} table)", 34))
    if rec:
        stats.append(_kv("source run", f"{run_name(r, rec.get('source_dataset'))} = {rec.get('source_label')}", 34))
        for role, sigs in (rec.get("inputs") or {}).items():
            names = ", ".join(Path(str((s or {}).get("path", "?"))).name for s in sigs or []) or "none"
            stats.append(_kv(f"{role} input", names, 34))
        how = "from the saved histograms (-from-saved)" if rec.get("from_saved") else "read in this run"
        stats.append(_kv("csv rows", f"{_num(rec.get('n_rows'))} ({how})", 34))
        stats.append(_kv("pedestal selection / anchor",
                         f"{rec.get('pedestal_selection')} / {rec.get('pedestal_anchor_method')}", 34))
    else:
        stats.append(_kv("provenance", "no record (-no-json)", 34))
    if df is None:
        fits.append(f"  table unreadable: {table}")
        return stats, fits
    stats.append(_kv("channels in the table", len(df), 34))
    for col, what in (("n_pedestal_events", "pedestal events per channel"),
                      ("n_mpv_entries", "MPV entries per channel")):
        if col in df.columns:
            stats.append(_kv(what, _range_text(df[col]), 34))
    if "pedestal" in df.columns:
        import numpy as np
        ok = np.isfinite(df["pedestal"].to_numpy(float))
        fits.append(f"  pedestal (Gaussian MLE in the open window): {int(ok.sum())}/{len(df)} channels fitted")
        fits.append(f"    pedestal [ADC]: {_range_text(df['pedestal'])}")
        if "pedestal_sigma" in df.columns:
            fits.append(f"    pedestal sigma [ADC]: {_range_text(df['pedestal_sigma'])}")
        if not ok.all():
            fits += _channel_lines(df.assign(_ped=np.where(ok, "OK", "NO_PEDESTAL")), "_ped", ("OK",),
                                   label="pedestal ")
    if "mpv_status" in df.columns:
        counts = Counter(df["mpv_status"].astype(str))
        fits.append(f"  MPV ({rec.get('mpv_method') or 'two-stage Landau x Gauss'}): status {_counts_text(counts)}")
        good = df[df["mpv_status"].astype(str) == "OK"]
        if not good.empty and "mpv" in df.columns:
            fits.append(f"    MPV [ADC] (OK channels): {_range_text(good['mpv'])}")
            if "mpv_error" in df.columns:
                rel = (good["mpv_error"] / good["mpv"]).to_numpy(float) * 100
                fits.append(f"    MPV relative error [%]: {_range_text(rel, digits=3)}")
            if "mpv_chi2_ndf" in df.columns:
                fits.append(f"    chi2/ndf: {_range_text(good['mpv_chi2_ndf'], digits=3)}")
            if {"mpv_fit_low", "mpv_fit_high"} <= set(df.columns):
                fits.append(f"    fit window [ADC]: low {_range_text(good['mpv_fit_low'])}; high "
                            f"{_range_text(good['mpv_fit_high'])}")
        if rec.get("mpv_fit_window"):
            fits.append(f"    final-fit window (provenance mpv_fit_window): {rec['mpv_fit_window']}")
        if "mpv_fit_status" in df.columns:
            fits.append(f"    final-fit convergence: {_counts_text(Counter(df['mpv_fit_status'].astype(str)))}")
        if "mpv_covariance_status" in df.columns and not good.empty:
            cov = Counter(good["mpv_covariance_status"].astype(str))
            fits.append(f"    covariance of the OK fits: {_counts_text(cov)}")
        fits += _channel_lines(df, "mpv_status", ("OK",), "mpv_fit_message" if "mpv_fit_message" in df.columns
                               else None, label="MPV ")
    return stats, fits


# ---- clean ----
# [4] / [5] of clean: per csv of every run the events in / kept, cut flow and per-layer table
def report_clean(r: _Run):
    stats, fits = [], []
    ctx, mod = r.ctx, r.mod
    jobs = r.extra.get("jobs") or []
    if not jobs:
        return ["  no dataset selected"], ["  no fits"]
    from_saved = getattr(r.args, "from_saved", False)
    for job in jobs:
        ds = job.ds
        for csv in ctx.run_csvs(ds):
            _clean_csv_lines(r, ds, csv, from_saved, stats)
    rng = fit_option(r, "fit") or (r.s or {}).get("mip_fit_range")
    drawn = ("drawn on the diagnostics pdf only (<run>/<csv stem>_clean_diagnostics.pdf by default; where this run "
             "put it: [6])") if ctx.make_plots else "not drawn in this run (-no-plots)"
    fits.append(f"  clean stores no fit values in its products: the diagnostic fits of the per-end E_MIP spectra "
                f"(range {_atom(rng)} MIP) are {drawn}; the selection itself fits nothing")
    return stats, fits


# [4] lines of one csv of a clean run: counts and cut flow (provenance, else the kept-bar cache record), layers
def _clean_csv_lines(r: _Run, ds, csv, from_saved: bool, stats: list) -> None:
    ctx, mod = r.ctx, r.mod
    clean, prov, cache = mod.clean_csv(ctx, ds, csv), mod.provenance_json(ctx, ds, csv), mod.cache_json(ctx, ds, csv)
    results = _clean_result(r, ds, csv, "results", "results_json")
    rec = _read_json(prov) if r.fresh(prov) else None
    if rec is None and (r.fresh(cache) or (from_saved and results is not None)):
        rec = _read_json(cache)
    name = f"{run_name(r, ds.key)} ({ctx.event_source(ds, csv)})"
    if rec is None:
        stats.append(f"  {name}: not selected in this run ({clean}: {r.stamp(clean)})")
        return
    c = rec.get("counts") or {}
    sel = rec.get("selection") or {}
    if r.fresh(clean):
        tail = ""
    elif from_saved:
        tail = "  [from saved: clean csv not rewritten]"
    else:
        tail = "  [clean csv not written: no clean unit (-only)]"
    stats.append(f"  {name}: {sel.get('standard', '?')} standard, threshold {sel.get('threshold_column', '?')}: "
                 f"{_num(c.get('input_events'))} events / {_num(c.get('input_rows'))} rows in -> "
                 f"{_num(c.get('output_events'))} events kept ({_pct(c.get('output_events'), c.get('input_events'))}), "
                 f"{_num(c.get('output_rows'))} clean bars; hit ends: end0 {_num(c.get('hit_end0'))}, end1 "
                 f"{_num(c.get('hit_end1'))}" + tail)
    flow = rec.get("cutflow") or []
    if len(flow) > 1:
        steps = [f"{f.get('step')} {_num(f.get('events'))} ev / {_num(f.get('rows'))} rows" for f in flow]
        stats.append("    cut flow: " + " -> ".join(steps))
    layers_path = _clean_result(r, ds, csv, "layers", "layers_csv")
    layers = _csv(layers_path) if layers_path is not None else None
    if layers_path is None:
        stats.append("    per-layer table: not written in this run")
    if layers is not None and "energy_mev_bar_mean" in layers.columns and len(layers):
        lo = layers.loc[layers["energy_mev_bar_mean"].idxmin()]
        hi = layers.loc[layers["energy_mev_bar_mean"].idxmax()]
        stats.append(f"    layers with clean bars: {int((layers['n_clean_bars'] > 0).sum())}/{len(layers)}; mean bar "
                     f"energy {_num(lo['energy_mev_bar_mean'])} MeV (layer {int(lo['layer'])}) - "
                     f"{_num(hi['energy_mev_bar_mean'])} MeV (layer {int(hi['layer'])})")


# clean result of one csv written in THIS run: its plan destination, else its default path (or None)
def _clean_result(r: _Run, ds, csv, unit: str, default_fn: str):
    candidates = [result_path(r, r.mod.result_key(ds, csv, unit)), getattr(r.mod, default_fn)(r.ctx, ds, csv)]
    return next((Path(p) for p in candidates if p is not None and r.fresh(p)), None)


# ---- response tables of adcpeak and ci ----
# the two components of a double-Gaussian row
def _double_text(row, suffix: str, unit: str) -> str:
    # mu and sigma of one component
    def comp(tag):
        mu = _pm(row.get(f"mu_{tag}{suffix}"), row.get(f"mu_{tag}_error{suffix}"))
        sig = _pm(row.get(f"sigma_{tag}{suffix}"), row.get(f"sigma_{tag}_error{suffix}"))
        return f"{tag} peak mu = {mu}, sigma = {sig}"
    return f"{comp('low')}; {comp('high')}{unit and ' [' + unit.strip() + ']'}"


# fit range of a spectrum row: the --fit text when given, else [fit_low, fit_high]
def _range_label(row, lo_col: str, hi_col: str, unit: str) -> str:
    text = str(_val(row, "fit_ranges", "") or "").strip()
    if text:
        raw = _val(row, "fit_bins", "") or _val(row, "bin_rule", "") or ""
        bins = (_num(raw) if isinstance(raw, (int, float)) else str(raw)).strip()   # pandas reads 30 as 30.0
        return f"--fit {text}{unit}" + (f", bins {bins}" if bins else "")
    lo, hi = row.get(lo_col), row.get(hi_col)
    return f"[{_num(lo)}, {_num(hi)}]{unit}" if _finite(lo) and _finite(hi) else "range NA"


# the rows of one record type of a response table (none when the column is missing)
def _rows_of(df, column: str, kind: str):
    return df[df[column] == kind] if column in df.columns else df.iloc[0:0]


# 'ok' or 'FAILED (<status>)'
def _ok_text(ok, status, limit: int = 80) -> str:
    return "ok" if _truthy(ok) else f"FAILED ({_short(status or '', limit)})"


# [5] lines of an adcpeak energy-response table: spectra, linearity per particle, resolution
def _adcpeak_response_lines(df, unit: str, names: dict | None = None) -> list[str]:
    out = []
    names = names or {}
    for _, row in _rows_of(df, "record", "spectrum").iterrows():
        key = str(row.get("dataset"))
        label = names.get(key, key)
        status = str(_val(row, "status", ""))
        ok = status.startswith("ok")
        method = _val(row, "spectrum_fit")
        if not _finite(row.get("mu")) and _finite(row.get("mu_low")):   # --fit[a,b],[c,d]: double Gaussian
            st = "ok" if ok else f"FAILED ({_short(status, 90)})"
            out.append(f"  {label}: {method} double Gaussian, fit range "
                       f"{_range_label(row, 'fit_low', 'fit_high', unit)}: {_double_text(row, '', '')}, "
                       f"chi2/ndf = {_num(row.get('chi2'))}/{_num(row.get('ndf'))}, {st}, "
                       f"{_num(_val(row, 'n_events'))} events")
            continue
        line = _spectrum_line(label, method, _val(row, "fit_low"), _val(row, "fit_high"), _val(row, "mu"),
                              _val(row, "mu_error"), _val(row, "sigma"), _val(row, "sigma_error"),
                              _val(row, "sigma_over_mu"), _val(row, "chi2"), _val(row, "ndf"), status, ok, unit,
                              range_text=_range_label(row, "fit_low", "fit_high", unit),
                              som_e=row.get("sigma_over_mu_error"))
        out.append(line + f", {_num(_val(row, 'n_events'))} events")
    lines = _rows_of(df, "record", "linearity")
    if lines.empty:
        out.append("  linearity: no fit (points only; -linfit <particle>[a,b] fits a line)")
    for _, row in lines.iterrows():
        out.append(f"  linearity {_val(row, 'particle')} ({_val(row, 'fit_range_gev')}, weighted "
                   f"{_val(row, 'weighted')}): slope = {_pm(row.get('slope'), row.get('slope_error'))}{unit}/GeV, "
                   f"intercept = {_pm(row.get('intercept'), row.get('intercept_error'))}{unit}, chi2/ndf = "
                   f"{_num(row.get('chi2'))}/{_num(row.get('ndf'))}, points {_val(row, 'fitted_energies')}, "
                   + _ok_text(row.get("ok"), _val(row, "status", "")))
    for _, row in _rows_of(df, "record", "resolution").iterrows():
        status = str(_val(row, "status", ""))
        if not _truthy(row.get("ok")) and not _finite(row.get("a")):
            out.append(f"  resolution ({_val(row, 'model')}): no fit, FAILED ({_short(status, 100)})")
            continue
        params = _params_text(row, (("a", "a", "a_error", ""), ("b", "b", "b_error", ""), ("c", "c", "c_error", "")))
        out.append(f"  resolution ({_val(row, 'model')}, {_val(row, 'particle')} points, "
                   f"{_val(row, 'fit_range_gev')}): {params}, chi2/ndf = {_num(row.get('chi2'))}/"
                   f"{_num(row.get('ndf'))}, points {_val(row, 'fitted_energies')}, " + _ok_text(row.get("ok"), status))
    return out


# [5] lines of a ci energy-response table: spectra, linearity per particle, resolution and e/h per scale
def _ci_response_lines(df, names: dict | None = None) -> list[str]:
    out = []
    names = names or {}
    for _, row in _rows_of(df, "row_type", "spectrum").iterrows():
        key = str(row.get("dataset"))
        label = f"{names.get(key, key)} [{row.get('path')}]"
        kind = _val(row, "fit_kind")
        ok = _truthy(row.get("fit_ok"))
        if kind == "range_double_gauss" or (not _finite(row.get("mu_mip")) and _finite(row.get("mu_low_mip"))):
            out.append(f"  {label}: {kind}, fit range {_range_label(row, 'fit_low_mip', 'fit_high_mip', ' MIP')}: "
                       f"{_double_text(row, '_mip', ' MIP')}; chi2/ndf = {_num(row.get('chi2'))}/"
                       f"{_num(row.get('ndf'))}, "
                       + _ok_text(ok, _val(row, "fit_status", "")))
            continue
        out.append(_spectrum_line(label, kind, row.get("fit_low_mip"), row.get("fit_high_mip"), row.get("mu_mip"),
                                  row.get("mu_error_mip"), row.get("sigma_mip"), row.get("sigma_error_mip"),
                                  row.get("sigma_over_mu"), row.get("chi2"), row.get("ndf"),
                                  _val(row, "fit_status", ""), ok, " MIP",
                                  range_text=_range_label(row, "fit_low_mip", "fit_high_mip", " MIP")))
    for _, row in _rows_of(df, "row_type", "linearity").iterrows():
        if _val(row, "fit_kind") == "points_only" or not _finite(row.get("slope_mip_per_gev")):
            out.append(f"  linearity [{row.get('path')}]: no fit (points only) - "
                       f"{_short(_val(row, 'fit_status', ''), 100)}")
            continue
        out.append(f"  linearity [{row.get('path')}] {_val(row, 'particle')} ({_val(row, 'fit_kind')}, "
                   f"{_val(row, 'linearity_fit_range')}): slope = "
                   f"{_pm(row.get('slope_mip_per_gev'), row.get('slope_error_mip_per_gev'))} MIP/GeV, intercept = "
                   f"{_pm(row.get('intercept_mip'), row.get('intercept_error_mip'))} MIP, chi2/ndf = "
                   f"{_num(row.get('chi2'))}/{_num(row.get('ndf'))}, points {_val(row, 'fitted_energies')}, "
                   + _ok_text(row.get("fit_ok"), _val(row, "fit_status", "")))
    for _, row in _rows_of(df, "row_type", "resolution").iterrows():
        if not _truthy(row.get("fit_ok")) and not _finite(row.get("resolution_a_mev")):
            out.append(f"  resolution [{row.get('path')}] ({_val(row, 'fit_kind')}): no fit, FAILED "
                       f"({_short(_val(row, 'fit_status', ''), 100)})")
            continue
        params = _params_text(row, (("a", "resolution_a_mev", "resolution_a_error_mev", " MeV"),
                                    ("b", "resolution_b_sqrt_mev", "resolution_b_error_sqrt_mev", " sqrt(MeV)"),
                                    ("c", "resolution_c", "resolution_c_error", "")))
        out.append(f"  resolution [{row.get('path')}] ({_val(row, 'fit_kind')}, {_val(row, 'particle')} points, "
                   f"{_val(row, 'resolution_fit_range')}): {params}, chi2/ndf = {_num(row.get('chi2'))}/"
                   f"{_num(row.get('ndf'))}, points {_val(row, 'fitted_energies')}, "
                   + _ok_text(row.get("fit_ok"), _val(row, "fit_status", "")))
    for _, row in _rows_of(df, "row_type", "e_over_h").iterrows():
        key = str(row.get("dataset"))
        point = "filled" if _truthy(row.get("eh_filled")) else "hollow"
        value = (f"e/h = {_pm(row.get('e_over_h'), row.get('e_over_h_error'))} ({point}: mu_electron from the "
                 f"{_val(row, 'electron_source', '?')})" if _finite(row.get("e_over_h")) else "no point")
        note = str(_val(row, "eh_note", "") or "").strip()
        out.append(f"  e/h [{row.get('path')}] {names.get(key, key)}: {value}"
                   + (f" - {_short(note, 100)}" if note else ""))
    return out


# ---- adcpeak ----
# [4] / [5] of adcpeak: line fit, Arbitrary Energy per csv, response table, TOT-ADC lines
def report_adcpeak(r: _Run):
    stats, fits = [], []
    s, ctx, mod = r.s, r.ctx, r.mod
    if s is None:
        return [f"  settings not resolved ({r.s_error})"], []
    steps, suffix = _steps(r), r.extra.get("suffix", "")
    params = mod.parameters_csv(ctx)
    prec = _read_json(mod.provenance_path(params)) if r.fresh(params) else None
    if r.fresh(params):
        prec = prec or {}
        stats.append(f"  line fit on {run_name(r, prec.get('fit_dataset', '?'))}, {_num(prec.get('rows_read'))} "
                     f"beam-csv rows read ({prec.get('fit_bins_from', 'no provenance record: -no-json')}); relation "
                     f"{prec.get('relation', 'TOT = slope * adc_max + offset')}")
    elif s["tot_mode"] == "none":
        stats.append("  no line table used (tot_mode none)")
    else:
        stats.append(f"  line table used: {params} ({r.stamp(params)})")
    if steps.get("events"):
        for ds in r.datasets or []:
            for csv in ctx.run_csvs(ds):
                stats.append(_adcpeak_events_line(r, ds, csv, suffix))
    # the response table where THIS run's output plan put it (saved, user / test name, or the scratch copy)
    resp_fr = final_result(r, "energy_response")
    resp_path = resp_fr.path if resp_fr is not None else mod.energy_response_csv(ctx, suffix)
    resp = _csv(resp_path) if r.fresh(resp_path) else None
    if resp is not None:
        stats.append(f"  energy response table of this run: {resp_path}"
                     + (f" ({saved_text(resp_fr)})" if resp_fr is not None else ""))
        if steps.get("saved_events"):   # the response read saved events: its record names them
            src = (_read_json(_adcpeak_response_record(r, resp_fr, resp_path)) or {}).get("events") or {}
            files = [Path(str(f.get("path", "?"))).name for f in src.get("files") or []]
            stats.append(f"  events not recomputed in this run: the response read the {src.get('from', 'saved copy')} "
                         f"({', '.join(files) or 'files not recorded'})")
        if "record" in resp.columns:
            for _, row in resp[resp["record"] == "spectrum"].iterrows():
                stats.append(f"    {run_name(r, row.get('dataset'))}: {_num(_val(row, 'n_events'))} events in the "
                             f"spectrum")
    # [5] TOT-ADC lines (the table this run wrote or read)
    table = _csv(params) if (steps.get("line_table") or steps.get("events")) and s["tot_mode"] != "none" else None
    if table is not None and "status" in table.columns:
        counts = Counter(table["status"].astype(str))
        n_ok = counts.get("ok", 0)
        lo, hi = (s.get("slope_window") or [0.6, 4.0])[:2]
        slope = table["tot_per_adc_slope"].astype(float) if "tot_per_adc_slope" in table.columns else None
        ok = table["status"].astype(str) == "ok"
        usable = int((ok & (slope >= lo) & (slope < hi)).sum()) if slope is not None else 0
        how = "fitted in this run" if r.fresh(params) else "table used, " + r.stamp(params)
        fits.append(f"  TOT-ADC lines ({how}): "
                    f"ok {n_ok} / failed {len(table) - n_ok} of {len(table)} channel-ends; usable for the events "
                    f"(ok and {_num(lo)} <= slope < {_num(hi)}): {usable}")
        fits.append(f"    status: {_counts_text(counts)}")
        if slope is not None and n_ok:
            fits.append(f"    slope [TOT/ADC] of ok lines: {_range_text(slope[table['status'].astype(str) == 'ok'])}")
        fits += _channel_lines(table, "status", ("ok",), label="line ")
    elif steps.get("line_table"):
        fits.append(f"  TOT-ADC lines: table not readable ({params})")
    if resp is not None:
        fits += _adcpeak_response_lines(resp, " a.u.", run_names(r))
    elif steps.get("energy_response"):
        fits.append(f"  energy response not written in this run ({resp_path}: {r.stamp(resp_path)})")
    else:
        fits.append("  energy response: not a unit of this run (-only)")
    return stats, fits


# [4] line of the Arbitrary Energy of one input csv (its saved copy in the cache and its record)
def _adcpeak_events_line(r: _Run, ds, csv, suffix: str) -> str:
    path = r.mod.saved_events_csv(r.ctx, ds, csv, suffix)
    source = r.ctx.event_source(ds, csv)
    if not r.fresh(path):
        return f"  {source}: events not computed in this run ({path}: {r.stamp(path)})"
    rec = _read_json(r.mod.provenance_path(path)) or {}
    bars = rec.get("bar_values") or {}
    return (f"  {source}: {_num(rec.get('rows'))} events (Arbitrary Energy, quality threshold "
            f"{_num(rec.get('quality_threshold'))}) -> cache {path.name}"
            + (f", bar values {_num(bars.get('rows'))} rows" if bars else ""))


# provenance record of the adcpeak response csv of this run
def _adcpeak_response_record(r: _Run, fr, path: Path) -> Path:
    if fr is not None and fr.plan is not None:
        return Path(r.mod.result_record(fr.plan, fr.key))
    return Path(r.mod.provenance_path(path))


# ---- ci ----
# [4] / [5] of ci: events per run and csv, CI map (provided by Erik Wallin), s fit (s parameter, Lei), response fits
def report_ci(r: _Run):
    stats, fits = [], []
    s, ctx, mod = r.s, r.ctx, r.mod
    if s is None:
        return [f"  settings not resolved ({r.s_error})"], []
    tag = s.get("variant_tag") or ""
    thr = f"bad_strip_fraction <= {_num(s['quality_threshold'])}" if s.get("quality_filter") else "no quality filter"
    stats.append(f"  units {', '.join(s.get('units') or [])}; event selection: {s.get('selection_path_used')} "
                 f"bad-strip flags, {thr}, bad-bar rule {s.get('bad_bar_rule')}")
    # the response table where THIS run's output plan put it (saved, user / test name, or the scratch copy)
    resp_fr = final_result(r, "response")
    resp_path = _ci_result_path(r, "response", tag)
    resp = _csv(resp_path) if r.fresh(resp_path) else None
    summary = _ci_console_summary(r.console)
    if resp is not None:
        stats.append(f"  energy response table of this run: {resp_path}"
                     + (f" ({saved_text(resp_fr)})" if resp_fr is not None else ""))
    if resp is not None and "row_type" in resp.columns:
        spec = resp[resp["row_type"] == "spectrum"]
        for key, grp in spec.groupby("dataset", sort=False):
            row = grp.iloc[0]
            stats.append(f"  {run_name(r, key)}: {_num(row.get('n_csv_files'))} csv, "
                         f"{_num(row.get('n_events_in_file'))} events in file, "
                         f"{_num(row.get('n_events_reconstructed'))} reconstructed, "
                         f"{_num(row.get('n_events_selected'))} selected (retention "
                         f"{_num(row.get('retention_fraction'))}), {_num(row.get('n_bars_selected_events'))} bars in "
                         f"the selected events")
    elif summary["runs"]:
        stats += [f"  {run_name(r, key)}: {text}" for key, text in summary["runs"]]
    elif any(u in s.get("units", ()) for u in mod.EVENT_UNITS):
        stats.append("  no event statistics in this run (no event was reconstructed or read)")
    if "events" in s.get("units", ()):
        for ds in r.datasets or []:
            for csv in ctx.run_csvs(ds):
                stats.append(_ci_event_csv_line(r, ds, csv, tag))
    if s.get("from_saved"):
        stats.append("  -from-saved: the stored event arrays were read (no reconstruction, no parameter fit)")
    # [5] parameters: CI map + s fit
    fits += _ci_parameter_lines(r, tag)
    if resp is not None:
        fits += _ci_response_lines(resp, run_names(r))
    elif summary["scales"]:
        fits += [f"  {line}" for line in summary["scales"]]
        fits.append("  (the fit numbers of every run and scale: -only response writes energy_response.csv)")
    qs = _ci_result_path(r, "quality_scan", tag)
    if r.fresh(qs):
        fits.append(f"  quality-threshold scan (check): {_file_facts(qs)} -> {qs}")
    elif "quality_scan" in s.get("units", ()):
        fits.append(f"  quality-threshold scan not written in this run ({qs}: {r.stamp(qs)})")
    return stats, fits


# the run lines and the per-scale lines of the 'ci summary' block of the console
def _ci_console_summary(console: str) -> dict:
    runs, scales, inside = [], [], False
    for line in console.splitlines():
        text = re.sub(r"^\[\s*[\d.]+s\]\s?", "", line)
        if text.startswith("ci summary"):
            inside = True
            continue
        if inside:
            m = re.match(r"\s+(\S+): (.*events selected.*)$", text)
            if m:
                runs.append((m.group(1), m.group(2).split(" | ")[0]))
                continue
            m = re.match(r"\s+(\S+): (linearity .*)$", text)
            if m:
                scales.append(f"[{m.group(1)}] {m.group(2)}")
                continue
            inside = False
    return {"runs": runs, "scales": scales}


# [4] line of the event csv of one input csv (unit events)
def _ci_event_csv_line(r: _Run, ds, csv, tag: str) -> str:
    path = r.mod.event_energies_csv(r.ctx, ds, csv, tag)
    if not r.fresh(path):
        return f"  event csv {r.ctx.event_source(ds, csv)}: not written in this run ({path}: {r.stamp(path)})"
    return f"  event csv {r.ctx.event_source(ds, csv)}: {_file_facts(path)} -> {path}"


# [5] lines of the CI map (provided by Erik Wallin) and the s fit (s parameter, Lei) of this call
def _ci_parameter_lines(r: _Run, tag: str) -> list[str]:
    s, ctx, mod = r.s, r.ctx, r.mod
    if s.get("from_saved"):
        return ["  CI map / s: not fitted (-from-saved reads the stored event arrays)"]
    if not mod.needs_ci(s):
        return ["  CI map / s: not used (adc_only reconstruction)"]
    official = mod.parameters_csv(ctx, tag)
    cache_csv, _, cache_meta = mod.parameters_cache_files(ctx, tag)
    params = official if r.fresh(official) else cache_csv
    rec = _read_json(cache_meta) or {}
    source = next((m.group(1) for m in (re.search(r"CI parameters \(([^)]*)\):", ln) for ln in r.console.splitlines())
                   if m), "?")
    out = [f"  CI parameters: {params} ({r.stamp(params)}; {source})"
           + ("" if params == official else "; the official copy is written with -only parameters")]
    table = _csv(params)
    if table is not None and "ci_status" in table.columns:
        counts = Counter(table["ci_status"].astype(str))
        out.append(f"    CI map: {counts.get('CI_OK', 0)} / {len(table)} channels CI_OK; status {_counts_text(counts)}")
        out += _channel_lines(table, "ci_status", ("CI_OK",), "ci_reason" if "ci_reason" in table.columns else None,
                              label="CI ")
    if not _ci_fits_s(r) or not rec:
        return out
    choice = rec.get("s_choice") or {}
    diags = rec.get("scale_diagnostics") or {}
    training = ((rec.get("scale_signature") or {}).get("training")) or {}
    for sample in s["scale_particles"]:
        diag = diags.get(sample) or {}
        runs = ", ".join(run_name(r, k) for k in (training.get(sample) or {}).get("datasets") or []) or "?"
        out.append(f"    s training, {sample} sample: runs {runs}; exact-overlap hits {_num(diag.get('exact_hits'))} "
                   f"(censored {_num(diag.get('censored_hits'))}); lambda = {_num(diag.get('selected_shrinkage'))} "
                   f"({diag.get('shrinkage_mode', '?')}; cv would pick {_num(diag.get('cv_selected_shrinkage'))}); "
                   f"global prior {_num(diag.get('global_prior'))}")
    if len(choice.get("samples") or []) > 1:
        borrowed = ", ".join(f"{x} {n}" for x, n in (choice.get("n_borrowed") or {}).items())
        out.append(f"    s = s_{choice.get('default')}: {choice.get('rule', '?')}; borrowing channels {borrowed}")
    if table is not None and "s_source" in table.columns:
        src = Counter(table["s_source"].astype(str))
        out.append(f"    s per channel: direct {src.get('DIRECT_SHRUNK_TO_LINK', 0)}, borrowed "
                   f"{src.get('BORROWED_LINK_OR_GLOBAL', 0)}, no model {src.get('RAW_NO_MODEL', 0)}, no CI "
                   f"{src.get('RAW_NO_CI', 0)}" + (f" (all: {_counts_text(src)})" if len(src) > 4 else ""))
        if "s_final" in table.columns:
            out.append(f"    s_final: {_range_text(table['s_final'])}")
    return out


# where THIS ci run put a final result: plan, else result_file (the record of the last run, else the default)
def _ci_result_path(r: _Run, key: str, tag: str) -> Path:
    fr = final_result(r, key)
    if fr is not None:
        return fr.path
    return Path(r.mod.result_file(r.ctx, key, tag))


# ---- plots ----
# [4] / [5] of plots: events drawn per run (from the console and the cache json)
def report_plots(r: _Run):
    stats = []
    pat = re.compile(r"plots: (\S+): (\d.*(?:events|hit map only).*)$")
    for line in r.console.splitlines():
        m = pat.search(line)
        if m:
            stats.append(f"  {run_name(r, m.group(1))}: {m.group(2)}")
    head = next((ln.split("plots: ", 1)[1] for ln in r.console.splitlines() if "plots: units " in ln), None)
    if head:
        stats.insert(0, f"  {head}")
    for p in r.written:
        if p.suffix == ".json" and "multiscale_plots" in p.parts:
            rec = _read_json(p) or {}
            if rec.get("kind") == "energy":
                stats.append(f"    cache {p.name}: {_num(rec.get('n_events'))} events, displays picked by "
                             f"{rec.get('rows_picked_by')}")
    if not stats:
        stats.append("  no figure drawn in this run")
    return stats, ["  no fits in this stage (figures only; dist_event draws the ci spectrum fit)"]


REPORTS = {
    "make_analysis_files": report_make_csv,
    "calibration_parameters": report_calib,
    "clean_dataset": report_clean,
    "tot_adc_peak": report_adcpeak,
    "charge_injection": report_ci,
    "multiscale_plots": report_plots,
}

# What the figures of each module show ([6] Outputs)
FIGURES = {
    "make_analysis_files": (("*_qa.pdf", "QA: per layer end-0 ADC / TOA / TOT histograms, per bar the 2-d "
                                         "correlations (fired / not fired)"),),
    "calibration_parameters": (
        ("pedestal_fits.pdf", "summary maps + one panel per channel: adc_sum - anchor of the pedestal events with the "
                              "Gaussian MLE"),
        ("mpv_fits.pdf", "summary maps + one panel per channel: MIP spectrum above the cut, fit window, Landau x Gauss "
                         "fit")),
    "clean_dataset": (("*_clean_diagnostics.pdf", "selection summary, per-end E_MIP spectra with Gauss / Landau fits, "
                                                  "muon vetoes, per-layer mean energy and clean bars, event energy"),),
    "tot_adc_peak": (
        ("tot_adc_peak_fits*.pdf", "slope maps + one page per channel-end: fired hits, bin means, fitted TOT-ADC line"),
        ("energy_spectra*.pdf", "Arbitrary Energy spectrum of every run with its peak fit"),
        ("linearity_resolution*.pdf", "linearity (mu vs E, every particle; -linfit lines) and resolution (sigma/mu "
                                      "vs E, electron fit)")),
    "charge_injection": (
        ("energy_linearity*.pdf", "per scale: mu vs beam energy, electron line (adc_only none, PureCI < 1 GeV, s all "
                                  "points) + pion points, summary table"),
        ("energy_resolution*.pdf", "per scale: sigma/mu vs beam energy, electron fit + pion points, summary table"),
        ("e_over_h*.pdf", "e/h on the s scale: filled (electron run) / hollow (electron line) points, no fit + table")),
    "multiscale_plots": (
        ("*_dist_event_*.pdf", "summary, event-energy distribution with the ci spectrum fit, event / n-bar overview"),
        ("*_dist_bar_*.pdf", "bar energy distribution of every bar"),
        ("*_dist_channel_*.pdf", "channel (bar end) energy distributions, one page per end and layer"),
        ("*_event_display_*.pdf", "event displays of the picked events (+ per-end and end0-end1 pages)"),
        ("*_bar_maps_*.pdf", "occupancy, mean energy per hit and per event"),
        ("*_layer_*.pdf", "longitudinal profile + per-layer energy and multiplicity"),
        ("*_hitmap.pdf", "raw hit fractions (no energy) + per-event hit displays")),
}


# ===================================================================================================
# [3] Inputs and [6] Outputs
# ===================================================================================================
# the stage's io_plan as a dry run (once; the error is kept)
def _plan(r: _Run):
    if r.plan is not None or r.plan_error is not None:
        return r.plan
    if r.mod is None or not hasattr(r.mod, "io_plan") or r.ctx is None:
        r.plan_error = "this stage has no io_plan"
        return None
    if r.datasets is None:
        r.plan_error = f"datasets not resolved ({r.datasets_error})"
        return None
    try:
        with _quiet(), r.ctx_state(dry_run=True):
            r.plan = list(r.mod.io_plan(r.ctx, r.args, r.datasets))
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        r.plan_error = _short(f"{type(exc).__name__}: {_exc_message(exc)}", 200)
    return r.plan


# provenance record of a file: '<name>.provenance.json', else '<stem>.provenance.json'
def _provenance_of(path: Path) -> dict | None:
    path = Path(path)
    for rec in (path.with_name(path.name + ".provenance.json"), path.with_name(path.stem + ".provenance.json")):
        if rec.exists():
            return _read_json(rec)
    return None


# rows / events of a csv from its provenance record; small csv counted; large ones never read
def _csv_facts(path: Path) -> str:
    rec = _provenance_of(path) or {}
    members = rec.get("members") if isinstance(rec.get("members"), dict) else {}
    events = rows = None
    if path.name in members:
        counts = rec.get("counts") or {}
        if rec.get("stage") == "test-subset":
            events, rows = rec.get("n_events"), rec.get("n_rows")
        elif counts.get("output_rows") is not None:                  # clean dataset
            events, rows = counts.get("output_events"), counts.get("output_rows")
        elif rec.get("rows") is not None:                            # make-csv csv, event csv
            events, rows = rec.get("events"), rec.get("rows")
    parts = []
    if isinstance(events, (int, float)):
        parts.append(f"{_num(events)} events")
    if isinstance(rows, (int, float)):
        parts.append(f"{_num(rows)} rows")
    if parts:
        return ", ".join(parts)
    try:
        if path.stat().st_size <= SMALL_CSV_BYTES:
            with open(path, "rb") as fh:
                n = sum(block.count(b"\n") for block in iter(lambda: fh.read(1 << 20), b""))
            return f"{max(n - 1, 0)} rows"
    except OSError:
        pass
    return ""


# page count of a pdf (up to PDF_PAGE_BYTES)
def _pdf_pages(path: Path) -> str:
    try:
        if path.stat().st_size > PDF_PAGE_BYTES:
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    n = len(re.findall(rb"/Type\s*/Page(?![s\w])", data))
    return f"{n} pages" if n else ""


# size + rows / pages / record kind of a file
def _file_facts(path: Path) -> str:
    try:
        size = _size(path.stat().st_size)
    except OSError:
        return "missing"
    extra = ""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        extra = _csv_facts(path)
    elif suffix == ".pdf":
        extra = _pdf_pages(path)
    elif suffix == ".json":
        extra = "provenance record" if path.name.endswith(".provenance.json") else "metadata record"
    elif suffix == ".npz":
        extra = "stored arrays (input of -from-saved)"
    return size + (f", {extra}" if extra else "")


# [3] Inputs: status, size and rows of every input and cache of the plan
def inputs_section(r: _Run) -> list[str]:
    plan = _plan(r)
    if plan is None:
        return [f"  input plan not available ({r.plan_error})"]
    lines, seen = [], set()
    from .io_utils import canonical_path

    for item in plan:
        role = item.get("role", "in")
        if role not in ("in", "cache"):
            continue
        p = Path(item["path"])
        key = (role, canonical_path(p))
        if key in seen:
            continue
        seen.add(key)
        if role == "cache":
            lines.append(f"  [cache  ] {p}  ({'exists' if p.exists() else 'missing'}; stored intermediate)")
            continue
        status = "ok" if p.exists() else ("MISSING" if item.get("kind") == "external" else "PENDING")
        if item.get("stale"):
            status = "STALE"
        facts = _file_facts(p) if p.exists() else "missing" + (f"; made by: {item['produced_by']}"
                                                               if item.get("produced_by") else "")
        kind = item.get("kind")
        lines.append(f"  [{status:<7}] {p}  ({facts}" + (f"; {kind}" if kind else "") + ")")
    if not lines:
        lines.append("  none listed by the stage plan")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + [f"  ... and {len(lines) - MAX_LINES} more"]
    return lines


# files written during the run: the trees of the run's mode, make-csv's run folders, plan outputs
def _scan_written(r: _Run) -> list[Path]:
    """Only RunContext.mode_tree_dirs are walked: never the inputs or the other mode's trees."""
    ctx = r.ctx
    flat = []
    try:
        roots = [Path(d) for d in ctx.mode_tree_dirs()]
    except Exception:  # noqa: BLE001 - hand-made contexts may lack a tree
        roots = []
    if r.folder == "make_analysis_files" and not getattr(ctx, "test", False):
        # make-csv's normal-mode csv folder (data/analysis_files/: -input csv files) and its run folders; a -test
        # run writes into data/test/ (walked)
        try:
            folder = Path(ctx.analysis_files_dir())
            flat += [folder] + ([Path(e.path) for e in os.scandir(folder) if e.is_dir()] if folder.is_dir() else [])
        except (OSError, KeyError, TypeError):
            pass
    found: dict[str, Path] = {}

    # keep a file modified after the start (no temporary files)
    def consider(path: Path) -> None:
        name = path.name
        if ".tmp" in name or ".old-" in name:
            return
        try:
            if path.stat().st_mtime_ns >= r.start_ns:
                found.setdefault(os.path.normcase(str(path)), path)
        except OSError:
            pass

    for root in roots:
        if root.is_dir():
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    consider(Path(dirpath) / f)
    for folder in flat:
        if folder.is_dir():
            for entry in os.scandir(folder):
                if entry.is_file():
                    consider(Path(entry.path))
    for item in (r.plan or []):
        if item.get("role") == "out":
            p = Path(item["path"])
            if p.is_file():
                consider(p)
    for fr in final_results(r):   # the final results where the run's output plans put them
        if fr.path.is_file():
            consider(fr.path)
    return sorted(found.values(), key=lambda p: (str(p.parent).lower(), p.name.lower()))


# what a figure shows (FIGURES pattern match)
def _figure_text(folder: str, path: Path) -> str | None:
    for pattern, text in FIGURES.get(folder, ()):
        if fnmatch.fnmatch(path.name, pattern):
            return text
    return None


# normalised absolute path for comparisons
def _path_key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


# provenance record written next to a final result in this run, or None
def _result_record_path(r: _Run, fr):
    for rec in (fr.path.with_name(fr.path.name + ".provenance.json"),
                fr.path.with_name(fr.path.stem + ".provenance.json")):
        if r.fresh(rec):
            return rec
    return None


# [6] the final results: kind, key, destination, saved or not, facts, figure, record
def final_results_lines(r: _Run) -> list[str]:
    finals = final_results(r)
    if not finals:
        if r.result_plans:
            return ["  final results: none in this run (its output plan has no result, e.g. -no-plots)"]
        return ["  final results: none (no output plan was made in this run)"]
    unsaved = [fr for fr in finals if not fr.saved]
    if unsaved:
        lines = [f"  final results: {len(unsaved)} of {len(finals)} NOT SAVED ({_unsaved_why(r.ctx)}): scratch "
                 f"copies in {unsaved[0].path.parent}; use --output [names] to save them"]
    else:
        lines = [f"  final results: {len(finals)} saved"]
    for fr in finals[:MAX_LINES]:
        lines.append(f"    {fr.kind} {fr.key} -> {fr.path}  ({saved_text(fr)}; {_written_text(r, fr.path)})")
        fig = (_figure_text(r.folder, fr.default) if fr.default is not None else None) or (
            fr.what if fr.kind == "pdf" else None)
        if fig:
            lines.append(f"      figure: {fig}")
        elif fr.what:
            lines.append(f"      what: {fr.what}")
        rec = _result_record_path(r, fr)
        if rec is not None:
            lines.append(f"      provenance: {rec}  ({_file_facts(rec)})")
    if len(finals) > MAX_LINES:
        lines.append(f"    ... and {len(finals) - MAX_LINES} more final results")
    return lines


# [6] Outputs: final results, then every other file written (dry run: the planned outputs)
def outputs_section(r: _Run) -> list[str]:
    if r.ctx is not None and r.ctx.dry_run:
        plan = _plan(r) or []
        outs = [it for it in plan if it.get("role") == "out"]
        lines = ["  dry run: nothing was written; planned outputs (ok = exists and would be replaced, new; final "
                 "results at their output-plan destinations):"]
        for it in outs[:MAX_LINES]:
            p = Path(it["path"])
            note = final_result_note(it, r.ctx)
            lines.append(f"    [{'ok ' if p.exists() else 'new'}] {p}" + (f"  ({note})" if note else ""))
        return lines
    finals = final_results(r)
    lines = final_results_lines(r)
    shown = {_path_key(fr.path) for fr in finals}
    shown |= {_path_key(rec) for rec in (_result_record_path(r, fr) for fr in finals) if rec is not None}
    files = [p for p in r.written if _path_key(p) not in shown]
    if not files:
        if not finals and not r.result_plans:
            return ["  none (no file was written in this run)"]
        lines.append("  no other file was written in this run")
        return lines
    lines.append("  intermediates and other files written in this run:")
    parent = None
    for p in files[:MAX_LINES * 2]:
        if p.parent != parent:
            parent = p.parent
            lines.append(f"  {parent}{os.sep}")
        lines.append(f"    {p.name}  ({_file_facts(p)})")
        fig = _figure_text(r.folder, p)
        if fig:
            lines.append(f"      figure: {fig}")
    if len(files) > MAX_LINES * 2:
        lines.append(f"  ... and {len(files) - MAX_LINES * 2} more files")
    plan = r.plan or []
    missing = [Path(it["path"]) for it in plan if it.get("role") == "out" and not it.get("result")
               and _path_key(it["path"]) not in shown and not r.fresh(Path(it["path"]))]
    if missing:
        lines.append("  planned by the stage but not (re)written in this run:")
        lines += [f"    {p}  ({r.stamp(p)})" for p in missing[:MAX_LINES]]
    return lines


# WARNINGs of this run's output plans (--output names not aligned / ignored)
def _plan_warnings(r: _Run) -> list[str]:
    out = []
    for plan in r.result_plans or []:
        out += [str(w) for w in getattr(plan, "warnings", None) or []]
    return out


# [7] Warnings: input notes, console warning lines, output-plan WARNINGs
def warnings_section(console: str, folder: str = "", plan_warnings=(), notes=()) -> list[str]:
    extra = EXTRA_WARNING_RE.get(folder)
    out, last = ["  " + _short(n, 1000) for n in notes], None
    for line in console.splitlines():
        if line.startswith("# console transcript") or line.startswith("# started") or line.startswith("[run-log]"):
            continue
        if WARNING_RE.search(line) or (extra is not None and extra.search(line)):
            text = line.rstrip()
            if text != last:   # an --output WARNING keeps its tail (which names got default names / were ignored)
                out.append("  " + _short(text, 1000 if "--output names" in text else 300))
            last = text
    for w in plan_warnings:   # output-plan WARNINGs the console does not hold
        if w not in console:
            out.append("  " + _short(f"WARNING: {w}  [output plan]", 1000))
    if not out:
        return ["  none"]
    if len(out) > MAX_LINES:
        out = out[:MAX_LINES] + [f"  ... and {len(out) - MAX_LINES} more (see the console transcript)"]
    return out


# ===================================================================================================
# [8] Bugs / errors
# ===================================================================================================
# is the file inside src/?
def _in_src(filename: str) -> bool:
    try:
        target = os.path.normcase(os.path.realpath(filename))
        root = os.path.normcase(os.path.realpath(str(SRC_ROOT)))
        return target.startswith(root + os.sep)
    except (TypeError, ValueError, OSError):
        return False


# 'src/<file>:<line> in <function>: <code>'
def _frame_text(filename: str, lineno, name: str, line: str | None) -> str:
    return f"{_repo_rel(filename)}:{lineno} in {name}: {(line or '').strip()}"


# (BUG LOCATION, RAISED AT, worker location): innermost src/ frame, library frame, worker frame
def bug_location(exc: BaseException) -> tuple[str | None, str | None, str | None]:
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    repo = [f for f in frames if _in_src(f.filename)]   # BUG LOCATION = repo[-1], the innermost frame in src/
    loc = _frame_text(repo[-1].filename, repo[-1].lineno, repo[-1].name, repo[-1].line) if repo else None
    raised = None
    if frames and not _in_src(frames[-1].filename):   # RAISED AT = the innermost frame, when it is in a library
        f = frames[-1]
        raised = _frame_text(f.filename, f.lineno, f.name, f.line)
    worker = None
    remote = getattr(exc.__cause__, "tb", None)
    if isinstance(remote, str):
        hits = re.findall(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', remote)
        hits = [h for h in hits if _in_src(h[0])]
        if hits:
            worker = _frame_text(*hits[-1])
    return loc, raised, worker


# is the exception a failure? (SystemExit 0 / None is not)
def is_failure(exc: BaseException | None) -> bool:
    return exc is not None and not (isinstance(exc, SystemExit) and exc.code in (None, 0))


# (status of [1], index column): OK | FAILED (...) | INTERRUPTED
def run_status(exc: BaseException | None, code=None, meaning: dict | None = None) -> tuple[str, str]:
    if exc is None or not is_failure(exc):
        if code not in (None, 0):
            why = (meaning or {}).get(code, "")
            return f"FAILED (exit code {code}{': ' + why if why else ''})", "FAILED"
        return "OK", "ok"
    if isinstance(exc, KeyboardInterrupt):
        return "INTERRUPTED (Ctrl+C)", "INTERRUPTED"
    if isinstance(exc, SystemExit):
        return f"FAILED (USER/INPUT ERROR, exit code {1 if isinstance(exc.code, str) else exc.code})", "FAILED"
    return f"FAILED (BUG: {type(exc).__name__})", "FAILED"


# [8] Bugs / errors: none | USER/INPUT ERROR | BUG with location and traceback, + input runs
def error_section(exc: BaseException | None, console: str = "", inputs=()) -> list[str]:
    if not is_failure(exc):
        return ["  none"]
    inputs = list(inputs or [])
    loc, raised, worker = bug_location(exc)
    msg = _exc_message(exc).rstrip()
    if isinstance(exc, SystemExit) and not isinstance(exc.code, str) and console:
        said = [ln.strip() for ln in console.splitlines() if "error:" in ln.lower() or "usage:" in ln.lower()]
        if said:   # argparse prints its message, then exits with code 2
            msg = f"exit code {exc.code}; the command printed:\n" + "\n".join(said[-4:])
    if isinstance(exc, KeyboardInterrupt):
        return ["  INTERRUPTED by the user (KeyboardInterrupt)"] + ([f"  at: {loc}"] if loc else [])
    if isinstance(exc, SystemExit):
        lines = ["  USER/INPUT ERROR (not a bug: the run stopped on purpose with this message)",
                 f"  type     : {type(exc).__name__}",
                 "  message  :"]
        lines += ["    " + ln for ln in (msg or "(no message)").splitlines()]
        if loc:
            lines.append(f"  raised in: {loc}")
        return lines + inputs
    lines = [f"  BUG: {type(exc).__name__}: {_short(msg, 400)}",
             f"  BUG LOCATION: {loc}" if loc else
             "  BUG LOCATION: no frame inside src/ (raised outside this repository)"]
    if raised:
        lines.append(f"  RAISED AT: {raised}")
    if worker:
        lines.append(f"  BUG LOCATION (worker process): {worker}")
    lines += inputs
    lines.append("  full traceback:")
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    lines += ["    " + ln for ln in text.splitlines()]
    return lines


# ===================================================================================================
# [1] Run
# ===================================================================================================
# command as typed: 'src/runner.py <args>'
def command_text(argv) -> str:
    argv = [str(a) for a in (argv or [])]
    joined = subprocess.list2cmdline(argv) if os.name == "nt" else " ".join(
        a if re.fullmatch(r"[\w@%+=:,./-]+", a) else "'" + a.replace("'", "'\"'\"'") + "'" for a in argv)
    return f"src/runner.py {joined}".rstrip()


# python and package versions
def _versions() -> str:
    out = [f"python {platform.python_version()}"]
    for name, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                       ("yaml", "PyYAML"), ("uproot", "uproot")):
        mod = sys.modules.get(name)
        v = getattr(mod, "__version__", None) if mod is not None else None
        if v is None:
            try:
                from importlib.metadata import version
                v = version(dist)
            except Exception:  # noqa: BLE001
                v = "not installed"
        out.append(f"{name} {v}")
    return ", ".join(out)


# first 12 hex of the sha256 of a file
def _sha12(path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unreadable"


# sha256 (12 hex) of the stage module, energy_response.py where shared, common/ and runner.py
def code_fingerprint(module_name: str | None) -> str:
    parts = []
    mod = sys.modules.get(module_name) if module_name else None
    if mod is not None and getattr(mod, "__file__", None):
        parts.append(f"{Path(mod.__file__).name} {_sha12(mod.__file__)}")
        if module_name.endswith(("tot_adc_peak", "charge_injection")):
            er = Path(mod.__file__).with_name("energy_response.py")
            parts.append(f"energy_response.py {_sha12(er)}")
    common = Path(__file__).parent
    h = hashlib.sha256()
    for f in sorted(common.glob("*.py")):
        try:
            h.update(f.name.encode() + f.read_bytes())
        except OSError:
            pass
    parts.append(f"common/*.py {h.hexdigest()[:12]}")
    parts.append(f"runner.py {_sha12(SRC_ROOT / 'runner.py')}")
    return "; ".join(parts)


# config files of the run (config.yml, local.yml, datasets.yml, LDMX_DATA_DIR)
def _config_files(cfg: dict | None) -> str:
    cfg = cfg or {}
    out = []
    f = cfg.get("_config_file")
    if f:
        out.append(_repo_rel(f))
        local = Path(f).with_name("local.yml")
        if local.exists():
            out.append(_repo_rel(local))
    ds = (cfg.get("paths") or {}).get("datasets_file")
    if ds:
        out.append(_repo_rel(ds))
    if os.environ.get("LDMX_DATA_DIR"):
        out.append(f"LDMX_DATA_DIR={os.environ['LDMX_DATA_DIR']}")
    return "; ".join(out) or "NA (configuration given in code)"


# mode text: fullset / test, dry-run, from-saved, clean-input, no-plots, inside 'all'
def _mode_text(ctx, args, clean: bool, direct: bool) -> str:
    if ctx is None:
        return "test" if getattr(args, "test", False) else "fullset"
    parts = [f"test (first {ctx.n_test_events} events)" if ctx.test else "fullset"]
    if ctx.dry_run:
        parts.append("dry-run")
    if getattr(args, "from_saved", False):
        parts.append("from-saved")
    if clean:
        parts.append("clean-input")
    if not ctx.make_plots:
        parts.append("no-plots")
    if not getattr(ctx, "write_json", True):
        parts.append("no-json")
    if not direct:
        parts.append("inside 'all'")
    return ", ".join(parts)


# ===================================================================================================
# sessions: console tee, log file, index
# ===================================================================================================
# sys.stdout / sys.stderr replacement: writes to the stream and to the console transcript
class _Tee:

    # keep the stream and the transcript
    def __init__(self, stream, sink):
        self._stream, self._sink = stream, sink

    # write to both; a broken transcript is dropped
    def write(self, s):
        n = self._stream.write(s)
        sink = self._sink
        if sink is not None and isinstance(s, str):
            try:
                sink.write(s)
            except Exception:  # noqa: BLE001 - a broken transcript never breaks the run
                self._sink = None
        return n

    # write every line
    def writelines(self, lines):
        for line in lines:
            self.write(line)

    # flush both
    def flush(self):
        self._stream.flush()
        sink = self._sink
        if sink is not None:
            try:
                sink.flush()
            except Exception:  # noqa: BLE001
                self._sink = None

    # stop copying to the transcript
    def detach_sink(self):
        self._sink = None

    # everything else from the original stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


# one log: a stage run, a runner command (all / check / setup) or an early failure
class _Session:

    # set up the session (folder, command, run)
    def __init__(self, kind: str, name: str, args, ctx, direct: bool = True, cfg: dict | None = None):
        self.kind, self.name, self.args, self.ctx, self.direct = kind, name, args, ctx, direct
        self.cfg = cfg if cfg is not None else (getattr(ctx, "cfg", None) or {})
        self.folder = MODULE_DIRS.get(name, _safe_name(name)) if kind == "stage" else RUNNER_DIR
        self.argv = list(_STATE["argv"] if _STATE["argv"] is not None else sys.argv[1:])
        self.command = command_text(self.argv)
        self.problems: list[str] = []
        self.children: list[dict] = []
        self.code = None
        self.parent = None
        self.r = _Run(name, args, ctx, direct) if kind == "stage" else None
        self.fh = None
        self.tee_out = self.tee_err = None

    # ---- naming ----
    # file stem: <time>_<stage>[_test][_dryrun][_<variant>][-N]
    def _stem(self) -> str:
        ts = self.start.strftime("%Y%m%d-%H%M%S")
        bits = [ts, _safe_name(self.name)]
        test = bool(getattr(self.ctx if self.ctx is not None else self.args, "test", False))
        if test:
            bits.append("test")
        if self.ctx is not None and getattr(self.ctx, "dry_run", False) and self.name != "check":
            bits.append("dryrun")
        if self.r is not None:
            try:
                with _quiet():
                    self.r.resolve()
                v = _variant(self.r)
            except Exception:  # noqa: BLE001
                v = ""
            if v:
                bits.append(_safe_name(v))
        base = "_".join(bits)
        stem, i = base, 1
        while (self.dir / f"{stem}.log").exists() or (self.dir / f"{stem}.console.txt").exists():
            i += 1
            stem = f"{base}-{i}"
        return stem

    # ---- open / close ----
    # open the log: write [1] as RUNNING, start the console transcript
    def open(self) -> None:
        self.start = datetime.now()
        self.start_ns = time.time_ns()
        self.t0 = time.perf_counter()
        if self.r is not None:
            self.r.start_ns = self.start_ns
        self.dir = logs_root(self.cfg) / self.folder
        _check_writable(self.dir, self.cfg)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = self._stem()
        self.log_path = self.dir / f"{self.stem}.log"
        self.console_path = self.dir / f"{self.stem}.console.txt"
        self.fh = open(self.console_path, "w", encoding="utf-8", errors="replace", newline="")
        self.fh.write(f"# console transcript: {self.command}\n# started {self.start:%Y-%m-%d %H:%M:%S}\n")
        self._write(self._render(None, final=False))
        self.out, self.err = sys.stdout, sys.stderr
        self.tee_out, self.tee_err = _Tee(self.out, self.fh), _Tee(self.err, self.fh)
        sys.stdout, sys.stderr = self.tee_out, self.tee_err
        stack = _STATE["stack"]
        self.parent = stack[-1] if stack else None
        stack.append(self)
        _STATE["opened"] += 1

    # restore stdout / stderr and close the transcript
    def _detach(self) -> None:
        for attr, tee, old in (("stdout", self.tee_out, getattr(self, "out", None)),
                               ("stderr", self.tee_err, getattr(self, "err", None))):
            if tee is None:
                continue
            try:
                tee.flush()
            except Exception:  # noqa: BLE001
                pass
            tee.detach_sink()
            if getattr(sys, attr) is tee:
                setattr(sys, attr, old)
        if self.fh is not None:
            try:
                self.fh.close()
            except Exception:  # noqa: BLE001
                pass
            self.fh = None
        try:
            _STATE["stack"].remove(self)
        except ValueError:
            pass

    # close the log: status, full render, index line, summary for the parent 'all'
    def close(self, exc: BaseException | None) -> None:
        self.end = datetime.now()
        self.duration = time.perf_counter() - self.t0
        self._detach()
        if exc is not None:
            _STATE["logged"][id(exc)] = exc   # the reference keeps the id from being reused
        t = time.perf_counter()
        self.status, self.index_status = run_status(exc, self.code, CHECK_EXIT if self.name == "check" else None)
        text = self._render(exc, final=True)
        text += f"\n(log written in {time.perf_counter() - t:.2f} s by ldmx_hcal.common.run_log)\n"
        self._write(text)
        self._index()
        if self.parent is not None and self.parent.kind == "command":
            self.parent.children.append(self.summary(exc))
        if self.problems:
            _warn(f"{self.log_path}: incomplete ({_short('; '.join(self.problems), 240)})")

    # summary of a stage for the log of 'all'
    def summary(self, exc) -> dict:
        n_out = len(self.r.written) if self.r is not None else 0
        head = ""
        fits = getattr(self, "_fit_lines", None) or []
        if fits:
            head = next((ln.strip() for ln in fits if ln.strip()), "")
            n_failed = sum(bool(re.search(r"(?<![A-Z_])FAILED \(", ln)) for ln in fits)   # not RAW_CI_FAILED
            if n_failed:
                head += f"  [{n_failed} fit line(s) FAILED]"
        loc = bug_location(exc)[0] if is_failure(exc) and not isinstance(exc, SystemExit) else None
        inputs = [f"{ri.role} {input_text(ri, with_error=False)}"
                  for ri in (run_inputs(self.r) if self.r is not None else [])]
        return {"name": self.name, "status": self.status, "duration": self.duration, "log": self.log_path,
                "outputs": n_out, "headline": head, "final": self._final_text(),
                "error": None if not is_failure(exc) else f"{type(exc).__name__}: {_short(_exc_message(exc), 160)}",
                "location": loc, "inputs": inputs}

    # 'N final results saved' / 'N final results NOT SAVED' / 'no final result'
    def _final_text(self) -> str:
        finals = (self.r.finals or []) if self.r is not None else []
        if not finals:
            return "no final result" if self.r is not None and self.r.result_plans else ""
        unsaved = sum(not fr.saved for fr in finals)
        if unsaved:
            count = f"{len(finals)}" if unsaved == len(finals) else f"{unsaved} of {len(finals)}"
            return f"{count} final results NOT SAVED ({_unsaved_why(self.ctx)}: scratch copies)"
        return f"{len(finals)} final results saved"

    # write the log file atomically (tmp + replace)
    def _write(self, text: str) -> None:
        tmp = self.log_path.with_name(self.log_path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.log_path)

    # append one line to logs/<module>/index.tsv
    def _index(self) -> None:
        path = self.dir / "index.tsv"
        new = not path.exists()
        fields = [f"{self.start:%Y-%m-%d %H:%M:%S}", self.name, self.command, self._mode(), self.index_status,
                  f"{self.duration:.1f}", self.log_path.name]
        with open(path, "a", encoding="utf-8", newline="") as fh:
            if new:
                fh.write("\t".join(INDEX_COLUMNS) + "\n")
            fh.write("\t".join(re.sub(r"[\t\r\n]+", " ", str(f)) for f in fields) + "\n")

    # mode text of the session (check: no dry-run)
    def _mode(self) -> str:
        clean = self.r.clean if self.r is not None else bool(getattr(self.ctx, "clean_input", False))
        text = _mode_text(self.ctx, self.args, clean, self.direct)
        return text.replace(", dry-run", "") if self.name == "check" else text   # check is always a plan

    # ---- rendering ----
    # one section; a broken section is reported and never hides the others
    def _section(self, title: str, fn) -> list[str]:
        try:
            body = fn()
        except Exception as exc:  # noqa: BLE001 - one broken section never hides the others
            self.problems.append(f"{title}: {type(exc).__name__}: {_short(exc, 120)}")
            body = [f"  (not available: {type(exc).__name__}: {_short(exc, 200)})"]
        return [title] + list(body) + [""]

    # [1] Run lines
    def _run_lines(self, exc, final: bool) -> list[str]:
        if self.kind == "early":
            what = f"{self.name} (stopped before the stage started)"
        elif self.r is not None:
            what = (f"{self.name} (module {self.r.module_name or '?'}), "
                    + ("run directly" if self.direct else "inside 'all'"))
        else:
            what = f"{self.name} (runner command)"
        lines = [_kv("command (as typed)", self.command, 20),
                 _kv("stage", what, 20),
                 _kv("start", f"{self.start:%Y-%m-%d %H:%M:%S}", 20)]
        if final:
            lines += [_kv("end", f"{self.end:%Y-%m-%d %H:%M:%S}", 20),
                      _kv("duration", f"{self.duration:.1f} s", 20),
                      _kv("exit status", self.status, 20)]
        else:
            lines += [_kv("exit status", "RUNNING (if this stays: the run is still going, or was killed)", 20)]
        lines += [_kv("mode", self._mode(), 20),
                  _kv("config files", _config_files(self.cfg), 20),
                  _kv("data_dir", str(((self.cfg or {}).get("paths") or {}).get("data_dir", "?")), 20),
                  _kv("python / packages", _versions(), 20),
                  _kv("python executable", sys.executable, 20),
                  _kv("code fingerprint", code_fingerprint(self.r.module_name if self.r is not None
                                                           else STAGE_MODULES.get(self.name)), 20),
                  _kv("working directory", os.getcwd(), 20),
                  _kv("console transcript", str(self.console_path), 20)]
        return lines

    # whole log text (at the start: [1] only, RUNNING)
    def _render(self, exc, final: bool) -> str:
        title = (f"LDMX HCal 2022 analysis - run log of stage '{self.name}' ({self.folder})" if self.kind == "stage"
                 else f"LDMX HCal 2022 analysis - run log of '{self.name}'")
        head = [title, "=" * len(title), ""]
        if not final:
            body = self._section(SECTIONS[0], lambda: self._run_lines(None, False))
            body += [f"{t}\n  (written when the run ends)\n" for t in SECTIONS[1:]]
            return "\n".join(head + body) + "\n"
        if self.kind == "stage":
            body = self._render_stage(exc)
        else:
            body = self._render_command(exc)
        return "\n".join(head + body)

    # the 8 sections of a stage log
    def _render_stage(self, exc) -> list[str]:
        r = self.r
        r.exc = exc
        try:
            r.console = self.console_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            r.console = ""
        r.result_plans, r.finals = _run_plans(), None   # before io_plan below (it must not add a plan of its own)
        r.resolve()
        _plan(r)
        try:
            final_results(r)
        except Exception as e:  # noqa: BLE001
            self.problems.append(f"final results: {type(e).__name__}: {_short(e, 120)}")
            r.finals = []
        try:
            r.written = _scan_written(r) if r.ctx is not None and not r.ctx.dry_run else []
        except Exception as e:  # noqa: BLE001
            self.problems.append(f"outputs: {type(e).__name__}: {_short(e, 120)}")
            r.written = []
        stats, fits = ["  (not available)"], ["  (not available)"]
        if r.ctx is not None and r.ctx.dry_run:
            stats = ["  dry run: nothing executed (the plan is in [3] and [6])"]
            fits = ["  dry run: no fit"]
        elif REPORTS.get(r.folder) is None:
            stats = ["  no module-specific statistics (stage unknown to the run log)"]
            fits = ["  none"]
        else:
            try:
                with _quiet(), r.ctx_state():
                    stats, fits = REPORTS[r.folder](r)
                if is_failure(exc):
                    note = "  (the run FAILED: only products written before the failure are listed)"
                    stats, fits = [note] + list(stats), list(fits)
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"report: {type(e).__name__}: {_short(e, 120)}")
                stats = fits = [f"  (not available: {type(e).__name__}: {_short(e, 200)})"]
        self._fit_lines = fits
        out = []
        out += self._section(SECTIONS[0], lambda: self._run_lines(exc, True))
        out += self._section(SECTIONS[1], lambda: features_section(r))
        out += self._section(SECTIONS[2], lambda: inputs_section(r))
        out += self._section(SECTIONS[3], lambda: stats or ["  none"])
        out += self._section(SECTIONS[4], lambda: fits or ["  none"])
        out += self._section(SECTIONS[5], lambda: outputs_section(r))
        out += self._section(SECTIONS[6], lambda: warnings_section(r.console, r.folder, _plan_warnings(r),
                                                                   input_notes(r)))
        out += self._section(SECTIONS[7], lambda: error_section(exc, r.console, input_lines(r)))
        return out

    # ---- runner commands (all / check / setup / early failure) ----
    # the 8 sections of a runner command log (all / check / setup / early failure)
    def _render_command(self, exc) -> list[str]:
        try:
            console = self.console_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            console = ""
        name = self.name
        out = []
        out += self._section(SECTIONS[0], lambda: self._run_lines(exc, True))
        if self.kind == "early":
            out += self._section(SECTIONS[1], lambda: ["  NA (the command stopped before its stage started)"])
            out += self._section(SECTIONS[2], lambda: ["  NA"])
            out += self._section(SECTIONS[3], lambda: ["  NA"])
            out += self._section(SECTIONS[4], lambda: ["  NA"])
            out += self._section(SECTIONS[5], lambda: ["  none"])
            out += self._section(SECTIONS[6], lambda: warnings_section(console))
            out += self._section(SECTIONS[7], lambda: error_section(exc, console))
            return out
        out += self._section(SECTIONS[1], lambda: self._command_features())
        if name == "all":
            out += self._section(SECTIONS[2], lambda: ["  see the stage logs listed in [4]"])
            out += self._section(SECTIONS[3], self._all_stats)
            out += self._section(SECTIONS[4], self._all_fits)
            out += self._section(SECTIONS[5], lambda: [f"  {c['name']}: {c['outputs']} files"
                                                       + (f", {c['final']}" if c.get("final") else "")
                                                       + f" (see {c['log'].name})" for c in self.children]
                                         or ["  none"])
        elif name == "check":
            stats = _check_stats(console)
            out += self._section(SECTIONS[2], lambda: ["  every stage's plan, counted in [4] (full list in the console "
                                                       "transcript)"])
            out += self._section(SECTIONS[3], lambda: stats + [_kv("exit code", f"{self.code} = "
                                                                   f"{CHECK_EXIT.get(self.code, '?')}", 20)])
            out += self._section(SECTIONS[4], lambda: ["  none (check only lists paths)"])
            out += self._section(SECTIONS[5], lambda: ["  none (check writes nothing)"])
        elif name == "setup":
            folder = getattr(self.args, "data_dir", None) or "NA (show the current setting)"
            out += self._section(SECTIONS[2], lambda: [f"  data folder: {folder}"])
            out += self._section(SECTIONS[3], lambda: ["  " + ln for ln in console.splitlines()[2:]
                                                       if ln.strip()][:MAX_LINES])
            out += self._section(SECTIONS[4], lambda: ["  none"])
            local = REPO_ROOT / "configure" / "local.yml"
            wrote = local.exists() and local.stat().st_mtime_ns >= self.start_ns
            out += self._section(SECTIONS[5], lambda: [f"  {local}  (updated)" if wrote else "  none"])
        else:
            out += self._section(SECTIONS[2], lambda: ["  NA"])
            out += self._section(SECTIONS[3], lambda: ["  NA"])
            out += self._section(SECTIONS[4], lambda: ["  NA"])
            out += self._section(SECTIONS[5], lambda: ["  none"])
        out += self._section(SECTIONS[6], lambda: warnings_section(console))
        out += self._section(SECTIONS[7], lambda: self._command_errors(exc, console))
        return out

    # [2] of a runner command
    def _command_features(self) -> list[str]:
        a, ctx = self.args, self.ctx
        if self.name == "setup":
            return [_kv("data folder (DATA_DIR)", getattr(a, "data_dir", None) or "NA (show the current setting)"),
                    _kv("protected folders (-protect)", _value_text(getattr(a, "protect", None)) or "NA")]
        if a is None:
            return ["  NA (the command line could not be parsed)"]
        lines = ["  (NA = not active)"]
        if self.name == "all":
            stages = getattr(a, "stages", None)
            lines.append(_kv("stages (-stages)", _value_text(stages) or "NA (default chain: "
                             + ("make-csv, " if getattr(a, "with_make_csv", False) else "")
                             + "calib, clean, adcpeak, ci, plots)"))
            lines.append(_kv("start from ROOT (-with-make-csv)", "on" if getattr(a, "with_make_csv", False) else "NA"))
        if self.name == "check":
            lines.append(_kv("strict (-strict)", "on" if getattr(a, "strict", False) else "NA"))
        lines.append(_kv("runs (--electron / --muon / --pion)", self._runs_text(a, ctx)))
        if ctx is not None:
            lines.append(_kv("test mode (-test)", f"first {ctx.n_test_events} events" if ctx.test else "NA"))
            lines.append(_kv("dry run (-dry-run)", "on" if ctx.dry_run and self.name != "check" else "NA"))
            lines.append(_kv("clean input (-clean-input)", "on (make-csv, calib, clean on the beam csv)"
                             if ctx.clean_input else "NA"))
            lines.append(_kv("from saved (-from-saved)", "on" if getattr(a, "from_saved", False) else "NA"))
            lines.append(_kv("figures", "on" if ctx.make_plots else "NA (-no-plots)"))
            lines.append(_kv("json records (-no-json)", "on" if getattr(ctx, "write_json", True) else "NA (-no-json)"))
            lines.append(_kv("final results (--output)", outputs_text(ctx, scratch_dir(ctx, "<stage>"))))
            lines.append(_kv("workers (-workers)", ctx.workers))
        return lines

    # runs chosen with the run flags, by reference; NA without a flag
    @staticmethod
    def _runs_text(a, ctx) -> str:
        try:
            flags = _flag_text(a)
        except Exception:  # noqa: BLE001
            flags = ""
        if not flags:
            return "NA (each stage's default)"
        try:
            from .config import SELECTABLE_PARTICLES, dataset_registry, run_ref, select_by_particle
            with _quiet():
                keys = select_by_particle(ctx.cfg, {p: getattr(a, f"select_{p}", None) for p in SELECTABLE_PARTICLES})
                reg = dataset_registry(ctx.cfg)
            return f"{', '.join(run_ref(reg[k]) for k in keys)} ({flags})  [command line]"
        except BaseException as exc:  # noqa: BLE001 - SystemExit of a run that does not exist included
            if isinstance(exc, KeyboardInterrupt):
                raise
            return f"NOT RESOLVED: {flags}  [command line]: {_short(_exc_message(exc), 300)}"

    # [4] of 'all': one line per stage (status, duration, outputs, log)
    def _all_stats(self) -> list[str]:
        if not self.children:
            return ["  no stage ran"]
        lines = [f"  {'stage':<9} {'status':<34} {'duration':>9}  {'outputs':>7}  log"]
        for c in self.children:
            lines.append(f"  {c['name']:<9} {_short(c['status'], 34):<34} {c['duration']:>7.1f} s  "
                         f"{c['outputs']:>7}  {c['log']}")
        return lines

    # [5] of 'all': the first fit line of every stage
    def _all_fits(self) -> list[str]:
        lines = []
        for c in self.children:
            lines.append(f"  {c['name']}: {c['headline'] or 'see its log'}")
        return lines or ["  none"]

    # [8] of a runner command: failed stages, then the exception or check's exit code
    def _command_errors(self, exc, console: str = "") -> list[str]:
        lines = []
        failed = [c for c in self.children if c.get("error")]
        for c in failed:
            lines.append(f"  stage {c['name']} FAILED: {c['error']}" + (f"\n    BUG LOCATION: {c['location']}"
                                                                         if c.get("location") else ""))
            if c.get("inputs"):
                lines.append("    input runs: " + "; ".join(c["inputs"]))
            lines.append(f"    details: {c['log']}")
        body = error_section(exc, console)
        if self.name == "check" and self.code not in (None, 0):
            body = [f"  check FAILED: exit code {self.code} = {CHECK_EXIT.get(self.code, '?')}"] + [
                "  " + ln.strip() for ln in console.splitlines() if "UNAVAILABLE" in ln or "MISSING" in ln][:MAX_LINES]
        if failed and is_failure(exc):
            lines.append("  the exception that ended the command:")
        return lines + body if (is_failure(exc) or self.code not in (None, 0) or not lines) else lines


# [4] of check: item counts per stage and every item that is not ok
def _check_stats(console: str) -> list[str]:
    stage, table, problems = None, {}, []
    for line in console.splitlines():
        m = re.match(r"\s*=== (\S+) ===\s*$", line)
        if m:
            stage = m.group(1)
            table[stage] = Counter()
            continue
        m = re.match(r"\s+\[(\w+)\s*\]\s+(\S+)\s+(.*)$", line)
        if m and stage:
            role, tag = m.group(1), m.group(2)
            table[stage][f"{role}:{tag}"] += 1
            if tag in ("MISSING", "PENDING", "STALE", "rebuild"):
                problems.append(f"    {stage} [{role}] {tag} {_short(m.group(3), 200)}")
        elif stage and "UNAVAILABLE" in line:
            table[stage]["UNAVAILABLE"] += 1
            problems.append(f"    {stage}: {line.strip()}")
    lines = [f"  {s:<9} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for s, c in table.items()]
    if problems:
        lines.append("  items that are not ok:")
        lines += problems[:MAX_LINES]
    return lines or ["  no stage listed"]


# stands in when logging is off
class _NullSession:

    # return the code
    @staticmethod
    def result(code):
        return code


# open a session, or None (logs off, or the log cannot be written: one warning)
def _open(kind, name, args, ctx, direct=True, cfg=None):
    if not enabled():
        return None
    try:
        s = _Session(kind, name, args, ctx, direct, cfg)
        s.open()
        return s
    except BaseException as exc:  # noqa: BLE001 - SystemExit of a protected folder included
        if isinstance(exc, KeyboardInterrupt):
            raise
        _warn(f"no log for '{name}': {type(exc).__name__}: {_short(_exc_message(exc), 240)}")
        try:
            if "s" in locals() and s is not None:
                s._detach()
        except Exception:  # noqa: BLE001
            pass
        return None


# close a session; a failure gives one warning
def _close(session, exc) -> None:
    if session is None:
        return
    try:
        session.close(exc)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        try:
            session._detach()
        except Exception:  # noqa: BLE001
            pass
        _warn(f"log of '{session.name}' not written: {type(e).__name__}: {_short(_exc_message(e), 240)}")


# ===================================================================================================
# public entry points (used by src/runner.py)
# ===================================================================================================
# log one stage run in logs/<module>/; the stage's exception is re-raised unchanged
@contextlib.contextmanager
def stage_log(name: str, args, ctx, direct: bool = True):
    reset_output_plans()
    session = _open("stage", name, args, ctx, direct)
    try:
        yield session
    except BaseException as exc:
        _close(session, exc)
        raise
    else:
        _close(session, None)


# handle of a runner command log
class _CommandHandle:
    # keep the session
    def __init__(self, session):
        self.session = session

    # record the exit code of the command and return it
    def result(self, code):
        if self.session is not None:
            self.session.code = code
        return code


# log a runner command (all / check / setup) in logs/runner/
@contextlib.contextmanager
def command_log(name: str, args, ctx=None):
    cfg = getattr(ctx, "cfg", None)
    if cfg is None and enabled():
        try:
            from .config import load_config
            with _quiet():
                cfg = load_config(getattr(args, "config", None))
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            cfg = {}
    session = _open("command", name, args, ctx, True, cfg)
    handle = _CommandHandle(session)
    try:
        yield handle
    except BaseException as exc:
        _close(session, exc)
        raise
    else:
        _close(session, None)


# keeps what the command prints until the first log opens
class _PreCapture:

    # empty buffer
    def __init__(self):
        self.buf = io.StringIO()
        self.out = self.err = None

    # tee stdout / stderr into the buffer
    def start(self):
        self.out, self.err = sys.stdout, sys.stderr
        self.tee_out, self.tee_err = _Tee(self.out, self), _Tee(self.err, self)
        sys.stdout, sys.stderr = self.tee_out, self.tee_err

    # buffer the text while no log is open
    def write(self, s):
        if _STATE["opened"] == 0:
            self.buf.write(s)

    # nothing to flush
    def flush(self):
        pass

    # restore stdout / stderr
    def stop(self):
        for attr, tee, old in (("stdout", self.tee_out, self.out), ("stderr", self.tee_err, self.err)):
            tee.detach_sink()
            if getattr(sys, attr) is tee:
                setattr(sys, attr, old)


# log a command that failed before any log was open (e.g. a mistyped option)
def _early_failure(argv, exc, text: str, start: datetime, t0: float) -> None:
    stage = next((a for a in argv if not str(a).startswith("-")), None)
    name = stage if stage in MODULE_DIRS or stage in COMMANDS else (stage or "no-stage")
    cfg = {}
    try:
        from .config import load_config
        conf = None
        for i, a in enumerate(argv):
            if a in ("--config", "-config") and i + 1 < len(argv):
                conf = argv[i + 1]
            elif str(a).startswith(("--config=", "-config=")):
                conf = str(a).split("=", 1)[1]
        with _quiet():
            cfg = load_config(conf)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
    args = type("Args", (), {"test": any(a in ("-test", "--test") for a in argv)})()
    s = _Session("early", name, args, None, True, cfg)
    s.folder = MODULE_DIRS.get(name, RUNNER_DIR)
    s.start, s.start_ns, s.t0 = start, time.time_ns(), t0
    s.dir = logs_root(cfg) / s.folder
    _check_writable(s.dir, cfg)
    s.dir.mkdir(parents=True, exist_ok=True)
    s.stem = s._stem()
    s.log_path, s.console_path = s.dir / f"{s.stem}.log", s.dir / f"{s.stem}.console.txt"
    s.console_path.write_text(f"# console transcript: {s.command}\n# started {start:%Y-%m-%d %H:%M:%S}\n" + text,
                              encoding="utf-8")
    s.close(exc)


# command-line entry point: sys.exit(run_log.guarded(main)); logs on, early failures logged
def guarded(main):
    """The exception is re-raised unchanged, so the exit code stays the same."""
    _STATE.update(enabled=True, argv=list(sys.argv[1:]), opened=0)
    start, t0 = datetime.now(), time.perf_counter()
    pre = None
    if enabled():
        try:
            pre = _PreCapture()
            pre.start()
        except Exception:  # noqa: BLE001
            pre = None
    try:
        return main()
    except BaseException as exc:
        if enabled() and is_failure(exc) and id(exc) not in _STATE["logged"] and not _STATE["stack"]:
            try:
                if pre is not None:
                    pre.stop()
                _early_failure(_STATE["argv"], exc, pre.buf.getvalue() if pre is not None else "", start, t0)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                _warn(f"the failure was not logged: {type(e).__name__}: {_short(_exc_message(e), 240)}")
        raise
    finally:
        if pre is not None:
            try:
                pre.stop()
            except Exception:  # noqa: BLE001
                pass
