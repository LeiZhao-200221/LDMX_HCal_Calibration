"""Shared IO: chunked csv, write protection, json records (-no-json), key checks, product sets, test subsets, log."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from . import geometry as geo

# compact dtypes for the beam csv schema (only applied to columns that are read)
BEAM_DTYPES = {
    "pf_event": "int64",
    "pf_spill": "int32",
    "pf_ticks": "int64",
    "layer": "int8",
    "strip": "int8",
    "tot_end0": "int32", "tot_end1": "int32",
    "toa_end0": "float32", "toa_end1": "float32",
    "adc_sum_end0": "int32", "adc_sum_end1": "int32",
    "adc_mean_end0": "float32", "adc_mean_end1": "float32",
    "adc_max_end0": "int32", "adc_max_end1": "int32",
}
BEAM_DTYPES.update({f"adc_{i}_end{e}": "int16" for i in range(8) for e in (0, 1)})
# raw no-signal sums written by make-csv (strict pedestal selection, default since round 3)
BEAM_DTYPES.update({f"raw_{w}_sum_end{e}": "int32" for w in ("tot", "toa") for e in (0, 1)})

# channel-key columns iter_csv validates before narrowing them to their compact dtype
KEY_COLUMNS_CHECKED_ON_READ = ("layer", "strip")

DEFAULT_CHUNKSIZE = 2_000_000
_T0 = time.time()
_PROTECTED_DIRS: list[Path] = []
TEST_SUBSET_BUILDER = "exact-lines-v2"  # bump when the subset definition changes (invalidates caches)

PRODUCT_SET_PROTOCOL = "sidecar-last-v1"  # metadata written last, with a signature of every member
FULL_HASH_LIMIT = 50 * 1024 ** 2         # files below this size: sha256 of the whole content
EDGE_HASH_BYTES = 4 * 1024 ** 2          # larger files: sha256 of the first + last 4 MiB and the size
_RESERVED_METADATA_KEYS = ("protocol", "members", "committed")
# unique temp names of product sets: <stem>.tmp-<pid>-<8 hex><suffix> and <metadata>.old-<pid>-<8 hex>
_UNIQUE_TEMP = re.compile(r"\.(?:tmp|old)-(\d+)-[0-9a-f]{8}(?:\.|$)")
ORPHAN_MIN_AGE_S = 3600  # an orphan temp file is swept only when untouched this long (and its process is gone)


# --------------------------------------------------------------------------
# user-facing errors
# --------------------------------------------------------------------------
# input or setting problem: ends the run with its message, catchable as ValueError
class InputError(SystemExit, ValueError):
    pass


# a stage setting (yml or command line) that cannot be used
class SettingsError(InputError):
    pass


# channel keys missing, non-integer, outside the detector or duplicated
class ChannelKeyError(InputError):
    pass


# a stored product that cannot be reused
class ProductSetError(InputError):
    pass


# --------------------------------------------------------------------------
# write protection
# --------------------------------------------------------------------------
# comparable absolute path: links resolved, long-path prefix removed, case-folded on Windows
def canonical_path(path) -> str:
    text = os.path.realpath(os.path.abspath(str(path)))
    for prefix in ("\\\\?\\UNC\\", "\\\\?\\"):
        if text.startswith(prefix):
            text = ("\\\\" + text[len(prefix):]) if prefix.endswith("UNC\\") else text[len(prefix):]
            break
    return os.path.normcase(text).rstrip("\\/") or text


_PROTECTED_ENV = "LDMX_PROTECTED_DIRS"  # carries the list into worker processes (spawned, not forked, on Windows)


# set the folders nothing may be written into (exported to worker processes too)
def set_protected_dirs(dirs: Iterable) -> None:
    _PROTECTED_DIRS[:] = [Path(d) for d in dirs if d]
    if _PROTECTED_DIRS:
        os.environ[_PROTECTED_ENV] = os.pathsep.join(str(d) for d in _PROTECTED_DIRS)
    else:
        os.environ.pop(_PROTECTED_ENV, None)


# the protected folders (copy)
def protected_dirs() -> list[Path]:
    return list(_PROTECTED_DIRS)


if os.environ.get(_PROTECTED_ENV):  # inherited from the parent process
    _PROTECTED_DIRS[:] = [Path(d) for d in os.environ[_PROTECTED_ENV].split(os.pathsep) if d]


# the protected folder that holds a path, else None
def _protecting_dir(path: Path) -> Path | None:
    target = canonical_path(path)
    for d in _PROTECTED_DIRS:
        root = canonical_path(d)
        if target == root or target.startswith(root + os.sep):
            return d
    return None


# --------------------------------------------------------------------------
# json records (-no-json): none next to the products of data/, calibration/, results/; cache/ keeps its own
# --------------------------------------------------------------------------
_JSON_OFF_ENV = "LDMX_NO_JSON_DIRS"     # carries -no-json into worker processes: the folders without records
_JSON_KEEP_ENV = "LDMX_JSON_KEEP_DIRS"  # and the folders that keep them (the cache tree)
_JSON_OFF_DIRS: list[Path] = []
_JSON_KEEP_DIRS: list[Path] = []


# switch the record json next to the products on or off (off: none inside record_dirs, except inside keep_dirs)
def set_json_records(enabled: bool, record_dirs: Iterable = (), keep_dirs: Iterable = ()) -> None:
    _JSON_OFF_DIRS[:] = [] if enabled else [Path(d) for d in record_dirs if d]
    _JSON_KEEP_DIRS[:] = [] if enabled else [Path(d) for d in keep_dirs if d]
    for env, dirs in ((_JSON_OFF_ENV, _JSON_OFF_DIRS), (_JSON_KEEP_ENV, _JSON_KEEP_DIRS)):
        if dirs:
            os.environ[env] = os.pathsep.join(str(d) for d in dirs)
        else:
            os.environ.pop(env, None)


for _env, _dirs in ((_JSON_OFF_ENV, _JSON_OFF_DIRS), (_JSON_KEEP_ENV, _JSON_KEEP_DIRS)):
    if os.environ.get(_env):  # inherited from the parent process
        _dirs[:] = [Path(d) for d in os.environ[_env].split(os.pathsep) if d]


# is a path inside one of the folders?
def _inside_any(path, dirs: list[Path]) -> bool:
    target = canonical_path(path)
    for d in dirs:
        root = canonical_path(d)
        if target == root or target.startswith(root + os.sep):
            return True
    return False


# is a record json written at this path? (False only with -no-json, next to a product outside cache/)
def json_records_on(path) -> bool:
    if not _JSON_OFF_DIRS:
        return True
    return _inside_any(path, _JSON_KEEP_DIRS) or not _inside_any(path, _JSON_OFF_DIRS)


# refuse a path inside a protected folder, else create its parent folder
def ensure_writable(path: Path) -> Path:
    path = Path(path)
    d = _protecting_dir(path)
    if d is not None:
        raise SystemExit(f"Refusing to write {path}: it is inside the protected folder {d} "
                         f"(safety.protected_dirs in configure/local.yml). Write the product into this "
                         f"repository's data/, calibration/, results/ or cache/ folders instead.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# identities, json, logging
# --------------------------------------------------------------------------
# cheap file identity: path, size, mtime_ns
def file_signature(path: Path) -> dict:
    path = Path(path)
    st = path.stat()
    return {"path": canonical_path(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


# content hash: sha256 of the whole file (< 50 MiB), else of its first + last 4 MiB + size
def content_hash(path: Path) -> tuple[str, str]:
    """Returns (sha256 hex, 'full' or 'edges-4MiB')."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if size < FULL_HASH_LIMIT:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
            return h.hexdigest(), "full"
        h.update(fh.read(EDGE_HASH_BYTES))
        fh.seek(size - EDGE_HASH_BYTES)
        h.update(fh.read(EDGE_HASH_BYTES))
    h.update(str(size).encode("ascii"))
    return h.hexdigest(), "edges-4MiB"


# signature of a product-set member: size, mtime_ns, content hash
def member_signature(path: Path) -> dict:
    st = Path(path).stat()
    digest, method = content_hash(path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "sha256": digest, "hash": method}


# short stable hash of a settings dict
def settings_hash(settings: dict) -> str:
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# an object as it reads back from json (tuples -> lists, Path -> str)
def _as_json(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


# atomic json write (temp file + rename); -no-json next to a product: nothing written, an old record removed
def write_json(obj, path: Path) -> Path:
    if not json_records_on(path):
        _drop_record(Path(path))
        return Path(path)
    path = ensure_writable(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        _unlink_quietly(tmp)
        raise
    return path


# remove a record json that no longer describes its product (-no-json run); never inside a protected folder
def _drop_record(path: Path) -> None:
    if _protecting_dir(path) is None:
        _unlink_quietly(path)


# read a json file (None when missing or unreadable)
def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# print a message with the seconds since start
def log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


# log the start and the duration of a block
@contextmanager
def timed(what: str):
    t = time.time()
    log(f"{what} ...")
    yield
    log(f"{what} done in {time.time() - t:.1f}s")


# stop with a message (and the producing command) when an input is missing
def require(path: Path, produced_by: str | None = None) -> Path:
    path = Path(path)
    if not path.exists():
        hint = f"\n  -> it is produced by: {produced_by}" if produced_by else ""
        raise SystemExit(f"Missing input file: {path}{hint}")
    return path


# delete a file; False when it was not there or could not be deleted
def _unlink_quietly(path) -> bool:
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# --------------------------------------------------------------------------
# settings validation
# --------------------------------------------------------------------------
# integer value? (bool never, 5.0 yes)
def _is_int(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, np.integer)):
        return True
    return isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer()


# a number (not bool)?
def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


# a finite number?
def _finite(v) -> bool:
    return _is_number(v) and math.isfinite(float(v))


# [low, high] with finite ends or None (open), ordered when both are set?
def _open_window(v) -> bool:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        return False
    if not all(x is None or _finite(x) for x in v):
        return False
    return v[0] is None or v[1] is None or float(v[0]) < float(v[1])


_RULES: dict[str, tuple[Callable, str]] = {
    "int": (_is_int, "an integer"),
    "int>0": (lambda v: _is_int(v) and v > 0, "a positive integer"),
    "int>=0": (lambda v: _is_int(v) and v >= 0, "a non-negative integer"),
    "even_int>0": (lambda v: _is_int(v) and v > 0 and int(v) % 2 == 0, "a positive even integer"),
    "float": (_finite, "a finite number"),
    "float>0": (lambda v: _finite(v) and v > 0, "a finite number > 0"),
    "float>=0": (lambda v: _finite(v) and v >= 0, "a finite number >= 0"),
    "fraction": (lambda v: _finite(v) and 0 <= v <= 1, "a number in [0, 1]"),
    "bool": (lambda v: isinstance(v, (bool, np.bool_)), "true or false"),
    "str": (lambda v: isinstance(v, str) and v != "", "a non-empty text"),
    "window": (lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and all(_finite(x) for x in v)
               and float(v[0]) < float(v[1]), "an ordered pair [low, high] of finite numbers with low < high"),
    "window_open": (_open_window, "a pair [low, high] of finite numbers or null (open end), low < high when "
                                  "both are set"),
}


# None when a value meets a rule, else what it must be
def _check_rule(value, rule) -> str | None:
    if isinstance(rule, str):
        optional = rule.endswith("?")
        name = rule[:-1] if optional else rule
        if optional and value is None:
            return None
        if name not in _RULES:
            raise ValueError(f"validate_settings: unknown rule '{rule}' (known: {sorted(_RULES)}, "
                             f"('choice', [...]), ('range', lo, hi), ('list', rule), callable)")
        test, text = _RULES[name]
        return None if test(value) else text + (" (or empty)" if optional else "")
    if callable(rule):
        return rule(value)
    kind = rule[0]
    if kind == "choice":
        options = list(rule[1])
        return None if value in options else f"one of {options}"
    if kind == "range":
        lo, hi = rule[1], rule[2]
        return None if _finite(value) and lo <= value <= hi else f"a number in [{lo}, {hi}]"
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            return f"a list (each element {_describe(rule[1])})"
        for i, item in enumerate(value):
            problem = _check_rule(item, rule[1])
            if problem:
                return f"a list whose elements are each {problem} (element {i} = {item!r})"
        return None
    raise ValueError(f"validate_settings: unknown rule {rule!r}")


# text of a rule, for messages
def _describe(rule) -> str:
    if isinstance(rule, str):
        return _RULES.get(rule.rstrip("?"), (None, rule))[1]
    return repr(rule)


# check settings against rules before any IO; SettingsError lists every problem
def validate_settings(settings: dict, spec: dict, source: str = "settings") -> dict:
    """Rules: _RULES names (+ '?' = may be None), ('choice', opts), ('range', lo, hi), ('list', rule), callable."""
    problems = []
    for key, rule in spec.items():
        if key not in settings:
            continue
        value = settings[key]
        problem = _check_rule(value, rule)
        if problem:
            problems.append(f"{key} = {value!r}: must be {problem}")
    if problems:
        raise SettingsError(f"{source}: invalid setting(s):\n  " + "\n  ".join(problems))
    return settings


# --------------------------------------------------------------------------
# channel keys
# --------------------------------------------------------------------------
_KEY_NAMES = ("layer", "strip", "end")


# problems of float key arrays: non-finite, non-integer, outside the detector
def _key_problems(values: dict[str, np.ndarray], describe_row: Callable[[int], str]) -> list[str]:
    n = len(next(iter(values.values()))) if values else 0
    if not n:
        return []
    problems = []
    with np.errstate(invalid="ignore"):
        bad = np.zeros(n, bool)
        for v in values.values():
            bad |= ~np.isfinite(v)
        if bad.any():
            i = int(np.argmax(bad))
            problems.append(f"{int(bad.sum())} row(s) with a missing or non-numeric key, first at {describe_row(i)}")
        frac = np.zeros(n, bool)
        for v in values.values():
            frac |= np.isfinite(v) & (v != np.floor(v))
        if frac.any():
            i = int(np.argmax(frac))
            problems.append(f"{int(frac.sum())} row(s) with a non-integer key, first at {describe_row(i)}")
        ok = ~(bad | frac)
        outside = np.zeros(n, bool)
        layer = values.get("layer")
        if layer is not None:
            outside |= ok & ((layer < 1) | (layer > geo.N_LAYERS))
        strip = values.get("strip")
        if strip is not None:
            if layer is not None:
                lay = np.clip(np.where(ok, layer, 1), 1, geo.N_LAYERS)
                n_strips = np.where(lay <= geo.LAST_NARROW_LAYER, geo.N_STRIPS_NARROW, geo.N_STRIPS_WIDE)
            else:
                n_strips = geo.N_STRIPS_WIDE
            outside |= ok & ((strip < 0) | (strip >= n_strips))
        end = values.get("end")
        if end is not None:
            outside |= ok & ~np.isin(end, geo.ENDS)
    if outside.any():
        i = int(np.argmax(outside))
        problems.append(f"{int(outside.sum())} key(s) outside the detector (layer 1-19; strip 0-7 in layers 1-9, "
                        f"0-11 in layers 10-19; end 0/1), first at {describe_row(i)}")
    return problems


# a column holding True / False?
def _has_booleans(col: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(col):
        return True
    if col.dtype == object:
        return any(isinstance(v, (bool, np.bool_)) for v in col.to_numpy())
    return False


# 'row i (layer=.., strip=..)' for messages
def _row_text(df: pd.DataFrame, columns, i: int) -> str:
    vals = ", ".join(f"{c}={df[c].iloc[i]!r}" for c in columns)
    return f"row {i} ({vals})"


# the one channel-key rule of every per-channel table: finite integers inside the detector, each channel once
def validate_channel_keys(df: pd.DataFrame, *, keys: Iterable[str] = ("layer", "strip", "end"),
                          node_key: str | Iterable[str] | None = None, unique: bool = True,
                          source: str = "table") -> pd.DataFrame:
    """Returns a copy with int64 key columns; ChannelKeyError names the first bad row."""
    keys = tuple(keys)
    unknown = [k for k in keys if k not in _KEY_NAMES]
    if unknown or not keys:
        raise ValueError(f"validate_channel_keys: keys must be a subset of {_KEY_NAMES}, got {keys}")
    nodes = (node_key,) if isinstance(node_key, str) else tuple(node_key or ())
    missing = [c for c in keys + nodes if c not in df.columns]
    if missing:
        raise ChannelKeyError(f"{source}: missing key column(s) {missing} (columns: {list(df.columns)})")
    booleans = [c for c in keys + nodes if _has_booleans(df[c])]
    if booleans:
        raise ChannelKeyError(f"{source}: invalid channel keys: column(s) {booleans} hold true/false values, "
                              f"not integers")
    values = {k: pd.to_numeric(df[k], errors="coerce").to_numpy(np.float64) for k in keys}
    problems = _key_problems(values, lambda i: _row_text(df, keys, i))
    node_values = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in nodes}
    for c, v in node_values.items():
        with np.errstate(invalid="ignore"):
            empty = ~np.isfinite(v)
            bad = ~empty & ((v != np.floor(v)) | (v < 0))
        if empty.any():
            problems.append(f"{int(empty.sum())} row(s) with an empty or non-numeric {c}, "
                            f"first at {_row_text(df, keys + nodes, int(np.argmax(empty)))}")
        if bad.any():
            problems.append(f"{int(bad.sum())} row(s) whose {c} is not an integer >= 0, "
                            f"first at {_row_text(df, keys + nodes, int(np.argmax(bad)))}")
    if problems:
        raise ChannelKeyError(f"{source}: invalid channel keys: " + "; ".join(problems))
    out = df.copy()
    for k in keys:
        out[k] = values[k].astype(np.int64)
    for c, v in node_values.items():
        out[c] = v.astype(np.int64)
    if unique and len(out):
        dup = out.duplicated(subset=list(keys + nodes), keep=False).to_numpy()
        if dup.any():
            i = int(np.argmax(dup))
            per = "(" + ", ".join(keys + nodes) + ")"
            raise ChannelKeyError(f"{source}: invalid channel keys: {int(dup.sum())} rows share a {per} key, "
                                  f"first at {_row_text(out, keys + nodes, i)}; every key must appear once")
    return out


# check the key columns of one csv chunk, then narrow them to their compact dtype
def _checked_key_columns(chunk: pd.DataFrame, columns: list[str], target: dict, path: Path) -> pd.DataFrame:
    values = {c: chunk[c].to_numpy(np.float64) for c in columns}
    start = int(chunk.index[0]) if len(chunk) else 0

    # 'csv line N (layer=.., strip=..)' for messages
    def describe(i: int) -> str:
        vals = ", ".join(f"{c}={values[c][i]:g}" for c in columns)
        return f"csv line {start + i + 2} ({vals})"  # +1 header, +1 one-based

    problems = _key_problems(values, describe)
    if problems:
        raise ChannelKeyError(f"{path}: invalid channel keys: " + "; ".join(problems))
    for c in columns:
        want = target.get(c) or "int64"
        ints = values[c].astype(np.int64)
        try:
            kind = np.dtype(want).kind
        except TypeError:  # a pandas extension dtype such as "Int16"
            kind = None
        if kind is None:
            chunk[c] = pd.array(ints).astype(want)
        elif kind in "iuf":
            chunk[c] = ints.astype(want)
        else:  # text requested: the integer spelling, as a direct read would give
            chunk[c] = ints.astype(str).astype(object)
    return chunk


# --------------------------------------------------------------------------
# reading csv
# --------------------------------------------------------------------------
# header of a csv
def csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


# compact dtypes of the given beam-csv columns
def _dtypes_for(columns: Iterable[str]) -> dict:
    return {c: BEAM_DTYPES[c] for c in columns if c in BEAM_DTYPES}


# read a large csv chunk by chunk: needed columns only, compact dtypes, channel keys checked
def iter_csv(path: Path, usecols: list[str] | None = None, chunksize: int = DEFAULT_CHUNKSIZE,
             dtypes: dict | None = None) -> Iterator[pd.DataFrame]:
    path = Path(path)
    if not _is_int(chunksize) or chunksize <= 0:
        raise SettingsError(f"iter_csv: chunksize must be a positive integer, got {chunksize!r}")
    cols = usecols or csv_columns(path)
    dt = _dtypes_for(cols)
    if dtypes:
        dt.update(dtypes)
    keys = [c for c in KEY_COLUMNS_CHECKED_ON_READ if c in cols]
    target = {c: dt.pop(c, None) for c in keys}
    dt.update({c: "float64" for c in keys})
    for chunk in pd.read_csv(path, usecols=cols, dtype=dt, chunksize=int(chunksize), engine="c"):
        if keys:
            chunk = _checked_key_columns(chunk, keys, target, path)
        yield chunk


# read a csv chunk by chunk into one frame (optional row_filter)
def read_csv_columns(path: Path, usecols: list[str] | None = None, chunksize: int = DEFAULT_CHUNKSIZE,
                     row_filter=None) -> pd.DataFrame:
    parts = []
    for chunk in iter_csv(path, usecols, chunksize):
        if row_filter is not None:
            chunk = chunk[np.asarray(row_filter(chunk), dtype=bool)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        cols = usecols or csv_columns(path)
        return pd.DataFrame({c: pd.Series(dtype=BEAM_DTYPES.get(c, "float64")) for c in cols})
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# temp files of a run
# --------------------------------------------------------------------------
# temp paths (and open writers) of one run, removed by cleanup
class TempRegistry:

    # empty registry
    def __init__(self):
        self._paths: list[Path] = []
        self._closers: list = []

    # register a temp path (before the file is created)
    def register(self, path) -> Path:
        path = Path(path)
        if path not in self._paths:
            self._paths.append(path)
        return path

    # register an object to close before its temp file is removed
    def register_closer(self, obj):
        self._closers.append(obj)
        return obj

    # a path is no longer temporary
    def release(self, path) -> None:
        path = Path(path)
        if path in self._paths:
            self._paths.remove(path)

    # registered paths (copy)
    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    # close every writer, then delete every registered path
    def cleanup(self) -> list[Path]:
        for obj in reversed(self._closers):
            try:
                obj.close()
            except Exception:  # noqa: BLE001 - cleanup must reach every file
                pass
        self._closers.clear()
        removed = [p for p in self._paths if _unlink_quietly(p)]
        self._paths.clear()
        return removed


_ACTIVE_REGISTRIES: list[TempRegistry] = []


# the innermost temp_registry of this process, or None
def active_registry() -> TempRegistry | None:
    return _ACTIVE_REGISTRIES[-1] if _ACTIVE_REGISTRIES else None


# with temp_registry(): every temp path registered inside is deleted when the block ends
@contextmanager
def temp_registry():
    reg = TempRegistry()
    _ACTIVE_REGISTRIES.append(reg)
    try:
        yield reg
    finally:
        _ACTIVE_REGISTRIES.remove(reg)
        reg.cleanup()


# register a temp path with the given or the active registry
def _register(path: Path, registry: TempRegistry | None = None) -> Path:
    reg = registry or active_registry()
    if reg is not None:
        reg.register(path)
    return path


# release a temp path from the given or the active registry
def _release(path: Path, registry: TempRegistry | None = None) -> None:
    reg = registry or active_registry()
    if reg is not None:
        reg.release(path)


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
# df.to_csv through a temp file renamed into place
def atomic_write_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    path = ensure_writable(path)
    tmp = _register(path.with_suffix(path.suffix + ".tmp"))
    try:
        df.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        _unlink_quietly(tmp)
        raise
    finally:
        _release(tmp)
    return path


# append frames to a csv through a temp file, renamed on close
class ChunkedCsvWriter:

    # open the writer (temp file registered)
    def __init__(self, path: Path, registry: TempRegistry | None = None, **to_csv_kwargs):
        self.path = ensure_writable(path)
        self.tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._registry = registry or active_registry()
        _register(self.tmp, self._registry)
        self._header = True
        self._csv_kwargs = to_csv_kwargs
        self.rows = 0

    # append one frame (header with the first one)
    def write(self, df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        df.to_csv(self.tmp, mode="w" if self._header else "a", header=self._header, index=False,
                  **self._csv_kwargs)
        self._header = False
        self.rows += len(df)

    # rename the temp file into place (an empty csv when nothing was written)
    def close(self, columns: list[str] | None = None) -> Path:
        try:
            if self._header:  # nothing written: still leave a valid (empty) csv
                pd.DataFrame(columns=columns or []).to_csv(self.tmp, index=False)
            os.replace(self.tmp, self.path)
        except BaseException:
            self.abort()
            raise
        _release(self.tmp, self._registry)
        return self.path

    # remove the temp file; the old file stays
    def abort(self) -> None:
        _unlink_quietly(self.tmp)
        _release(self.tmp, self._registry)

    # context manager: the writer
    def __enter__(self):
        return self

    # close on success, abort on error
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()


# with atomic_pdf(p) as pdf: the pdf reaches p only on success
@contextmanager
def atomic_pdf(path: Path):
    from matplotlib.backends.backend_pdf import PdfPages

    path = ensure_writable(path)
    tmp = _register(path.with_name(path.stem + ".tmp.pdf"))
    pdf = PdfPages(tmp)
    try:
        yield pdf
        pdf.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            pdf.close()
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        _unlink_quietly(tmp)
        raise
    finally:
        _release(tmp)


# --------------------------------------------------------------------------
# orphan temp files of killed runs
# --------------------------------------------------------------------------
# is a process running on this machine? (unknown -> True)
def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, AttributeError, ValueError):
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, OverflowError):
        return True
    return True


# remove the temp files of killed runs in a folder (process gone, untouched for 1 h)
def sweep_orphan_temps(folder: Path, min_age_s: float = ORPHAN_MIN_AGE_S) -> list[Path]:
    folder = Path(folder)
    if _protecting_dir(folder) is not None:
        return []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return []
    now = time.time()
    removed = []
    for entry in entries:
        m = _UNIQUE_TEMP.search(entry.name)
        if m is None or not entry.is_file(follow_symlinks=False):
            continue
        try:
            if now - entry.stat().st_mtime < min_age_s or _pid_alive(int(m.group(1))):
                continue
        except OSError:
            continue
        if _unlink_quietly(entry.path):
            removed.append(Path(entry.path))
    if removed:
        log(f"removed {len(removed)} temp file(s) left by an interrupted run in {folder}: "
            f"{', '.join(p.name for p in removed)}")
    return removed


_SWEPT: set[str] = set()


# sweep a folder at most once per process
def _sweep_once(folder: Path) -> None:
    key = canonical_path(folder)
    if key not in _SWEPT:
        _SWEPT.add(key)
        sweep_orphan_temps(folder)


# --------------------------------------------------------------------------
# product sets: data files + metadata json, committed sidecar-last
# --------------------------------------------------------------------------
# member name in the metadata: path relative to the metadata folder
def _member_key(final: Path, metadata_path: Path) -> str:
    final, base = Path(os.path.abspath(final)), Path(os.path.abspath(metadata_path)).parent
    try:
        return final.relative_to(base).as_posix()
    except ValueError:
        return str(final)


# member path from its name in the metadata
def _member_path(key: str, metadata_path: Path) -> Path:
    p = Path(key)
    return p if p.is_absolute() else Path(metadata_path).parent / p


# write several files + their metadata json, metadata last: a reader never sees a mixed set
class ProductSet:

    # new set for one metadata json
    def __init__(self, metadata_path: Path, registry: TempRegistry | None = None):
        self.metadata_path = Path(metadata_path)
        self._registry = registry or active_registry()
        self._token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._staged: dict[Path, Path] = {}
        self.record: dict | None = None

    # temp path to write a member to (same folder, same suffix)
    def stage(self, final: Path) -> Path:
        final = ensure_writable(Path(final))
        if final == self.metadata_path:
            raise ValueError("the metadata json is written by commit(), not staged")
        _sweep_once(final.parent)
        tmp = final.with_name(f"{final.stem}.tmp-{self._token}{final.suffix}")
        self._staged[final] = _register(tmp, self._registry)
        return tmp

    # stage a member and call its writer on the temp path
    def add(self, final: Path, writer: Callable[[Path], dict | None]) -> dict:
        out = writer(self.stage(final))
        return out if isinstance(out, dict) else {}

    # remove every staged temp file
    def abort(self) -> None:
        for tmp in self._staged.values():
            _unlink_quietly(tmp)
            _release(tmp, self._registry)
        self._staged.clear()

    # rename every temp file into place, then write the metadata with every member's signature (last)
    def commit(self, metadata: dict) -> dict:
        clash = [k for k in _RESERVED_METADATA_KEYS if k in metadata]
        if clash:
            raise ValueError(f"metadata keys {clash} are reserved for the product-set protocol")
        missing = [str(t) for t in self._staged.values() if not t.exists()]
        if missing:
            raise RuntimeError(f"product set {self.metadata_path.name}: staged file(s) never written: {missing}")
        meta_path = ensure_writable(self.metadata_path)
        _sweep_once(meta_path.parent)
        backup = None
        if meta_path.exists():
            backup = _register(meta_path.with_name(f"{meta_path.name}.old-{self._token}"), self._registry)
            os.replace(meta_path, backup)  # from here on the old set is no longer advertised
        renamed = 0
        try:
            for final, tmp in self._staged.items():
                os.replace(tmp, final)
                _release(tmp, self._registry)
                renamed += 1
        except BaseException:
            restored = False
            if renamed == 0 and backup is not None:
                try:  # nothing replaced yet: the old pair is intact, advertise it again
                    os.replace(backup, meta_path)
                    restored = True
                except OSError:
                    pass
            if backup is not None:
                if not restored:
                    _unlink_quietly(backup)
                _release(backup, self._registry)
            self.abort()
            raise
        if backup is not None:  # the old set is gone from here on: its metadata is useless
            _unlink_quietly(backup)
            _release(backup, self._registry)
        finals = list(self._staged)
        self._staged.clear()
        members = {_member_key(final, meta_path): member_signature(final) for final in finals}
        record = {**metadata, "protocol": PRODUCT_SET_PROTOCOL, "members": members,
                  "committed": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(record, meta_path)  # LAST: the set becomes valid only now (-no-json: no record, not verified)
        self.record = record
        return record

    # context manager: the set
    def __enter__(self):
        return self

    # abort on error
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()


# write {final path: writer} as one product set with its metadata
def commit_product_set(members: dict, metadata: dict, metadata_path: Path,
                       registry: TempRegistry | None = None) -> dict:
    extra: dict = {}
    with ProductSet(metadata_path, registry) as ps:
        for final, writer in members.items():
            extra.update(ps.add(Path(final), writer))
        return ps.commit({**metadata, **extra})


# short json text of a value, for messages
def _short(value, limit: int = 160) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# (True, 'ok') when a stored set may be reused (metadata there, identity equal, members unchanged), else (False, reason)
def product_set_valid(metadata_path: Path, expected: dict | None = None,
                      members: Iterable[Path] = ()) -> tuple[bool, str]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        members = [Path(p) for p in members]
        if not json_records_on(metadata_path) and members and all(p.exists() for p in members):
            return True, f"not verified (-no-json: no record {metadata_path.name})"
        return False, f"no metadata {metadata_path.name}"
    record = read_json(metadata_path)
    if not isinstance(record, dict):
        return False, f"metadata {metadata_path.name} is unreadable"
    if record.get("protocol") != PRODUCT_SET_PROTOCOL or not isinstance(record.get("members"), dict):
        return False, f"metadata {metadata_path.name} has no member signatures"
    for key, want in _as_json(expected or {}).items():
        if key not in record:
            return False, f"'{key}' is not recorded (requested {_short(want)})"
        have = record[key]
        if json.dumps(have, sort_keys=True) != json.dumps(want, sort_keys=True):
            return False, f"'{key}' differs: stored {_short(have)}, requested {_short(want)}"
    stored = record["members"]
    for path in members:
        if _member_key(Path(path), metadata_path) not in stored:
            return False, f"{Path(path).name} is not part of the stored set"
    for key, sig in stored.items():
        path = _member_path(key, metadata_path)
        try:
            st = path.stat()
        except OSError:
            return False, f"member {key} is missing"
        if int(st.st_size) != sig.get("size") or int(st.st_mtime_ns) != sig.get("mtime_ns"):
            return False, f"member {key} changed since it was committed (size / modification time)"
        if content_hash(path)[0] != sig.get("sha256"):
            return False, f"member {key} changed since it was committed (content)"
    return True, "ok"


# metadata of a reusable set, or ProductSetError with the reason and the rebuild command (-from-saved)
def require_product_set(metadata_path: Path, expected: dict | None, rebuild: str, what: str = "stored product",
                        members: Iterable[Path] = ()) -> dict:
    ok, reason = product_set_valid(metadata_path, expected, members)
    if not ok:
        raise ProductSetError(f"cannot reuse the {what} ({Path(metadata_path).parent}): {reason}.\n"
                              f"  Rebuild it with: {rebuild}")
    return read_json(metadata_path) or {}


# --------------------------------------------------------------------------
# test subsets
# --------------------------------------------------------------------------
# the n smallest distinct pf_event ids of a csv
def _smallest_event_ids(src: Path, n_events: int, chunksize: int) -> np.ndarray:
    best = np.empty(0, dtype=np.int64)
    for chunk in pd.read_csv(src, usecols=["pf_event"], dtype={"pf_event": "int64"}, chunksize=chunksize):
        best = np.union1d(best, np.unique(chunk["pf_event"].to_numpy()))[:n_events]
    return best


SOURCE_PROVENANCE_FIELDS = ("n_samples", "n_samples_used", "waveform")


# converter facts of a beam csv from its make-csv provenance (only when it names this file)
def source_provenance(src: Path) -> dict | None:
    src = Path(src)
    rec_path = src.with_name(f"{src.stem}.provenance.json")
    record = read_json(rec_path)
    if not isinstance(record, dict):
        return None
    out = record.get("output") if isinstance(record.get("output"), dict) else None
    st = src.stat()
    matches = out is not None and out.get("size") == int(st.st_size) and out.get("mtime_ns") == int(st.st_mtime_ns)
    info = {"path": str(rec_path), "matches_source": bool(matches)}
    if matches:
        info.update({k: record[k] for k in SOURCE_PROVENANCE_FIELDS if k in record})
    return info


# provenance path of a test subset
def _subset_record_path(dst: Path) -> Path:
    dst = Path(dst)
    return dst.with_name(f"{dst.stem}.provenance.json")


# what a test subset depends on: builder version, source file, N
def _subset_identity(src: Path, n_events: int) -> dict:
    return {"builder": TEST_SUBSET_BUILDER, "source": file_signature(src), "n_events_requested": int(n_events)}


# (True, 'ok') when a stored test subset may be reused, else (False, reason)
def subset_status(src: Path, dst: Path, n_events: int) -> tuple[bool, str]:
    if not Path(src).exists():
        return False, f"its source {Path(src).name} is missing"
    return product_set_valid(_subset_record_path(dst), _subset_identity(src, n_events), members=[Path(dst)])


# test subset: every row of the N smallest pf_event ids, copied byte for byte, cached as a product set
def build_test_subset(src: Path, dst: Path, n_events: int, chunksize: int = DEFAULT_CHUNKSIZE) -> Path:
    src, dst = Path(src), Path(dst)
    require(src)
    if not _is_int(n_events) or n_events <= 0:
        raise SettingsError(f"test subset: the number of events must be a positive integer, got {n_events!r}")
    record_path = _subset_record_path(dst)
    wanted = _subset_identity(src, n_events)
    ok, reason = product_set_valid(record_path, wanted, members=[dst])
    if ok:
        return dst
    if dst.exists() or record_path.exists():
        log(f"test subset {dst.name} is rebuilt: {reason}")
    log(f"test mode: extracting the first {n_events} events of {src.name} (exact copy, cached)")
    ids = _smallest_event_ids(src, int(n_events), min(int(chunksize), 5_000_000))
    keep = set(int(i) for i in ids)

    # copy the rows of the kept events
    def write_subset(tmp: Path) -> dict:
        n_rows = 0
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            header = fin.readline()
            fout.write(header)
            k = header.rstrip(b"\r\n").split(b",").index(b"pf_event")
            for line in fin:
                field_ = line.split(b",", k + 1)[k]
                try:
                    event = int(field_)
                except ValueError:
                    event = int(float(field_))
                if event in keep:
                    fout.write(line)
                    n_rows += 1
        return {"n_rows": n_rows}

    if len(ids) < n_events:
        log(f"WARNING: {src.name} holds only {len(ids)} events (< {n_events} requested); "
            f"the test subset has all of them")
    meta = {**wanted, "stage": "test-subset", "n_events": int(len(ids)),
            "first_event": int(ids[0]) if len(ids) else None,
            "last_event": int(ids[-1]) if len(ids) else None}
    conv = source_provenance(src)
    if conv is not None:
        meta["source_provenance"] = conv
        meta.update({k: conv[k] for k in SOURCE_PROVENANCE_FIELDS if k in conv})
    record = commit_product_set({dst: write_subset}, meta, record_path)
    log(f"test subset: {len(ids)} events, {record['n_rows']} rows -> {dst}")
    return dst


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
# one BLAS thread in worker processes
def limit_blas_threads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


# python / numpy / pandas versions
def python_info() -> str:
    return f"python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}"
