"""Stage plots: per-run pdfs on one energy scale (event / bar / channel distributions, event displays, maps, layers).

Energy scale (-energy): s, pureci, adc_only (ci paths, [MIP]; s_<sample> with several ci training samples) or
arbitrary (adcpeak Arbitrary Energy [a.u.]). The values come from the ci / adcpeak cache of every csv of the run
(per event, per bar, per end); only the producer's selected events are drawn; nothing is reconstructed here.
event energy: the producer's event value (sum of its bar energies); bar energy: one bar in one event; channel energy:
one bar end in one event. hit (hitmap, beam csv): an end has adc_sum > the calibration column hit_threshold_column.
event key = csv index * 2^32 + pf_event (csv index: place of the csv in its run folder, sorted by name).
Output units (-only): dist_event, dist_bar, dist_channel, event_display, bar_maps, layer, hitmap (one pdf per run).

No csv output: the results are results/<mode>/multiscale_plots/<run key>_<unit>_<scale>[_<tag>].pdf
(hitmap: <run key>_hitmap.pdf).
Cache cache/[test/]multiscale_plots/ (-from-saved):
  <run key>_energy_<scale>[_<tag>].npz: csv_names, event_keys, event_energy, event_n_bars, n_counted, cell_hits,
  cell_total, [layer_energy, layer_hits], [values, value_cells], [end_values, end_channels], [rows_key, rows_layer,
  rows_strip, rows_energy, rows_end0, rows_end1, row_events]; channel = cell * 2 + end, cell = (layer - 1) * 12 +
  strip position;
  <run key>_hitmap.npz: csv_names, n_events, both, end0, end1, first_events, first_key, first_cell.
"""

from __future__ import annotations

import argparse
import copy
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import geometry as geo
from ..common import io_utils
from ..common import outputs
from ..common import plotting as plot
from ..common.calibration_io import calibration_lookup, load_calibration
from ..common.config import REPO_ROOT, run_flags, run_ref
from ..common.io_utils import (ProductSet, SettingsError, atomic_pdf, canonical_path, content_hash, ensure_writable,
                               file_signature, iter_csv, log, product_set_valid, read_json, require,
                               require_product_set, timed, validate_settings)
from ..energy import charge_injection as ci
from ..energy import energy_response as er
from ..energy import tot_adc_peak as ta

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (after plotting: Agg backend)
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HELP = "per-run figures on one energy scale: event / bar / channel distributions, event displays, maps, hit maps"
DEFAULT_DATASETS = ["electron 4"]  # run references
CONFIG_SECTION = "plots"
STAGE_DIR = "multiscale_plots"  # results/<mode>/multiscale_plots/, cache/[test/]multiscale_plots/
SUPPORTS_FROM_SAVED = True      # -from-saved: re-draw from the reduced arrays in cache/[test/]multiscale_plots/
SUPPORTS_CLEAN_INPUT = True     # -clean-input: the hit map reads the clean dataset, the scales the -clean-input cache
CACHE_VERSION = "plots-reduced-v2"  # bump when the stored arrays or their meaning change (invalidates the cache)

ARBITRARY = "arbitrary"                                    # adcpeak Arbitrary Energy [a.u.]
CI_SOURCES = ("s", "pureci", "adc_only", *(ci.sample_path(p) for p in ci.SCALE_PARTICLES))  # ci paths [MIP]
ENERGY_SOURCES = (*CI_SOURCES, ARBITRARY)
UNITS = ("dist_event", "dist_bar", "dist_channel", "event_display", "bar_maps", "layer", "hitmap")
ENERGY_UNITS = UNITS[:-1]
UNIT_GROUPS = {"distributions": ("dist_event", "dist_bar", "dist_channel"), "energy": ENERGY_UNITS}
UNIT_TEXT = {"dist_event": "event energy distribution with the ci spectrum fit + event / n-bar overview",
             "dist_bar": "bar energy distribution per bar", "dist_channel": "channel (bar end) energy distributions",
             "event_display": "event displays of the picked events", "bar_maps": "occupancy and mean energy maps",
             "layer": "longitudinal profile + per-layer energy and multiplicity",
             "hitmap": "raw hit fractions + hit displays (beam csv, no energy)"}
# reduced arrays each energy unit needs: bars = the bar values were read; layers / values / ends / rows on top
UNIT_NEEDS = {"dist_event": (), "dist_bar": ("bars", "values"), "dist_channel": ("bars", "ends"),
              "event_display": ("bars", "rows"), "bar_maps": ("bars",), "layer": ("bars", "layers")}
PICKERS = ("first", "highest", "random")
COLOR_SCALES = ("fixed", "robust")
KEY_SHIFT = 32                                             # event key = csv index << 32 | pf_event
PF_MASK = (1 << KEY_SHIFT) - 1
VARIANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

DEFAULTS = {
    "energy_source": "s",                  # s | pureci | adc_only | s_<sample> (ci) | arbitrary (adcpeak)
    "variant": None,                       # producer variant tag (null: the tag of a default ci / adcpeak run)
    "hit_threshold_column": "pedestal_range_with_20percent",  # calibration column: end hit threshold (hitmap)
    "event_picker": "first",               # first | highest | random
    "n_event_displays": 10,                # event displays per run
    "random_seed": 12345,                  # seed of the random picker
    "per_end_displays": False,             # add the per-end 2x2 and difference pages per event
    "color_scale": "fixed",                # fixed | robust (per-event colour scale)
    "color_vmax_mip": 10.0,                # fixed colour-scale maximum in MIP
    "robust_percentile": 98.0,             # robust scale: percentile of the event's bar energies
    "robust_max_fraction": 0.8,            # robust scale: fraction of the event's maximum
    "robust_vmax_min": 1.0,                # robust scale: lower clamp
    "robust_vmax_max_mip": 50.0,           # robust scale: upper clamp in MIP (no clamp for arbitrary units)
    "difference_percentile": 98.0,         # end0-end1 map: percentile of |diff| over all bars
    "difference_vmax_min": 0.5,            # end0-end1 map: lower clamp of the symmetric scale
    "difference_vmax_max_mip": 50.0,       # end0-end1 map: upper clamp in MIP
    "event_colormap": "plasma",            # colour map of the event displays
    "map_colormap": "viridis",             # colour map of the aggregated bar maps
    "overview_bins": 300,                  # event-energy overview: bins on [0, max]
    "min_bin_count": 10,                   # event-energy overview: draw only bins with >= this many events
    "spectrum_bins": 100,                  # bins of the per-bar / per-channel / per-layer spectra
    "spectrum_quantile": 0.999,            # spectra range: [0, 1.05 * this quantile of the values]
    "annotate_maps": True,                 # write the cell values into the bar maps
    "hitmap_events": 20,                   # per-event hit displays in the hit-map pdf
    "workers": 1,                          # pdf rendering processes when -workers is not given
    "save_cache": True,                    # store the reduced arrays in cache/[test/]multiscale_plots/ (-from-saved)
}


# settings check: a matplotlib colour-map name
def _colormap_rule(value):
    return None if isinstance(value, str) and value in matplotlib.colormaps else "a matplotlib colour-map name"


# settings check: a number in (0, 1]
def _quantile_rule(value):
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 < float(value) <= 1.0
    return None if ok else "a number in (0, 1]"


# settings check of the variant: null or a file-name safe tag
def _variant_rule(value):
    ok = value is None or (isinstance(value, str) and (not value or VARIANT_PATTERN.match(value)))
    return None if ok else "null or a tag of letters, digits, '_', '.' and '-'"


# settings check of -events: non-negative integer pf_event ids
def _events_rule(value):
    if value is None:
        return None
    ok = isinstance(value, (list, tuple)) and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                                   and v >= 0 for v in value)
    return None if ok else "a list of non-negative integer pf_event ids"


# every setting is checked (io_utils.validate_settings) before any input is read
SETTINGS_SPEC = {
    "energy_source": ("choice", ENERGY_SOURCES),
    "variant": _variant_rule,
    "hit_threshold_column": "str",
    "event_picker": ("choice", PICKERS),
    "n_event_displays": "int>=0",
    "random_seed": "int>=0",
    "per_end_displays": "bool",
    "color_scale": ("choice", COLOR_SCALES),
    "color_vmax_mip": "float>0",
    "robust_percentile": ("range", 0, 100),
    "robust_max_fraction": "fraction",
    "robust_vmax_min": "float>0",
    "robust_vmax_max_mip": "float>0",
    "difference_percentile": ("range", 0, 100),
    "difference_vmax_min": "float>0",
    "difference_vmax_max_mip": "float>0",
    "event_colormap": _colormap_rule,
    "map_colormap": _colormap_rule,
    "overview_bins": "int>0",
    "min_bin_count": "int>=0",
    "spectrum_bins": "int>0",
    "spectrum_quantile": _quantile_rule,
    "annotate_maps": "bool",
    "hitmap_events": "int>=0",
    "workers": "int>0?",                   # null = 1
    "save_cache": "bool",
    "fixed_vmax_override": "float>0?",     # -vmax
    "energy_window": "window?",            # -energy-window LO HI (LO < HI)
    "explicit_events": _events_rule,       # -events
}

BEAM_COLUMNS = ["pf_event", "layer", "strip", "adc_sum_end0", "adc_sum_end1"]
ADCPEAK_EVENT_COLUMNS = ["pf_event", "arbitrary_energy", "n_bars", "selected"]
ADCPEAK_BAR_COLUMNS = ["pf_event", "layer", "strip", "bar_arbitrary_energy", "end0_arbitrary_energy",
                       "end1_arbitrary_energy"]

VIEWS = ("odd", "even")
VIEW_LAYERS = {"odd": geo.LAYERS[geo.LAYERS % 2 == 1], "even": geo.LAYERS[geo.LAYERS % 2 == 0]}
N_CELLS = geo.N_LAYERS * geo.N_POSITIONS
N_CHANNELS = N_CELLS * 2


# (19, 12) mask of the cells that hold a bar (8-bar layers 1-9: positions 2..9)
def _physical_mask() -> np.ndarray:
    bars = geo.all_bars()
    return np.isfinite(geo.fill_layer_grid(bars["layer"], bars["strip"], 1.0))


PHYSICAL_MASK = _physical_mask()


# flat cell index: (layer - 1) * 12 + strip position
def cell_index(layer, strip) -> np.ndarray:
    layer = np.asarray(layer, dtype=np.intp)
    return (layer - 1) * geo.N_POSITIONS + np.asarray(geo.physical_position(layer, strip), dtype=np.intp)


# event key: csv index << 32 | pf_event
def event_key(index: int, pf) -> np.ndarray:
    return (np.int64(index) << KEY_SHIFT) | np.asarray(pf, dtype=np.int64)


# mask: is each value in the sorted unique id array?
def in_sorted(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not len(sorted_ids):
        return np.zeros(values.shape, dtype=bool)
    pos = np.searchsorted(sorted_ids, values)
    pos[pos == len(sorted_ids)] = 0
    return sorted_ids[pos] == values


# event name for titles: 'Event <pf_event>' (+ ' (<csv name>)' when the run has several csv files)
def event_name(key, csv_names) -> str:
    key = int(key)
    pf, index = key & PF_MASK, key >> KEY_SHIFT
    if len(csv_names) > 1 and index < len(csv_names):
        return f"Event {pf} ({csv_names[index]})"
    return f"Event {pf}"


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------
# folder of the pdfs: results/<mode>/multiscale_plots/
def output_dir(ctx) -> Path:
    return ctx.results_dir(STAGE_DIR)


# file suffix of a variant tag: '' or '_<tag>'
def tag_suffix(tag) -> str:
    v = str(tag or "").strip().lstrip("_")
    return f"_{v}" if v else ""


# default pdf: results/<mode>/multiscale_plots/<run key>_<unit>_<scale>[_<tag>].pdf (hitmap: <run key>_hitmap.pdf)
def output_pdf(ctx, ds, unit: str, source: str = "s", tag: str = "") -> Path:
    if unit == "hitmap":
        return output_dir(ctx) / f"{ds.key}_hitmap.pdf"
    return output_dir(ctx) / f"{ds.key}_{unit}_{source}{tag_suffix(tag)}.pdf"


# result key of a pdf: its default file name without .pdf
def result_key(ctx, ds, unit: str, s: dict, tag: str = "") -> str:
    return output_pdf(ctx, ds, unit, s["energy_source"], tag).stem


# final results of the run (for -output): every pdf, runs in order, then the units
def result_slots(ctx, datasets, s: dict, tag: str = "") -> list:
    """Empty with -no-plots; the reduced arrays in cache/ are intermediates."""
    if not ctx.make_plots:
        return []
    return [outputs.ResultSlot(result_key(ctx, ds, unit, s, tag), "pdf", output_pdf(ctx, ds, unit,
                                                                                  s["energy_source"], tag),
                               f"{run_ref(ds)} {unit}: {UNIT_TEXT[unit]}")
            for ds in datasets for unit in s["units"]]


# output plan of this stage (an -output name given twice stops the run)
def _output_plan(ctx, slots, preview: bool = False) -> "outputs.OutputPlan":
    try:
        return outputs.OutputPlan(ctx, STAGE_DIR, slots, preview=preview)
    except ValueError as exc:  # two results would be written to one file
        raise SettingsError(f"plots: {exc}") from None


# ({key: (path, saved)}, plan) of the pdfs from the output plan
def pdf_destinations(ctx, slots, dry: bool = False) -> tuple[dict, "outputs.OutputPlan"]:
    """dry=True (dry-run, check, run log): a preview plan that never empties the scratch folder."""
    plan = _output_plan(ctx, slots, preview=dry)
    if dry:
        for w in plan.warnings:
            log(f"WARNING: {w}")
    return {s.key: (plan.path(s.key), plan.saved(s.key)) for s in slots}, plan


# --------------------------------------------------------------------------
# the producers: ci / adcpeak cache of every csv of a run
# --------------------------------------------------------------------------
# training samples of the ci yml: ['pion'] by default
def _ci_yml_samples(ctx) -> list[str]:
    return ci.scale_samples(ctx.stage_cfg(ci.CONFIG_SECTION).get("scale_particle", ci.DEFAULT_SCALE_PARTICLE))


# producer variant tag read: -variant, else the tag of a default producer run; ci with -clean-input: + clean
def producer_tag(ctx, s: dict) -> str:
    variant, source = s.get("variant"), s["energy_source"]
    if source == ARBITRARY:
        if variant is not None:
            return str(variant)
        yml = ctx.stage_cfg(ta.CONFIG_SECTION)
        return ta.variant_tag({"tot_mode": yml.get("tot_mode", ta.DEFAULTS["tot_mode"]),
                               "output_tag": yml.get("output_tag")})
    if variant is None:
        yml = ctx.stage_cfg(ci.CONFIG_SECTION)
        return ci.variant_tag({"scale_particles": _ci_yml_samples(ctx), "output_tag": yml.get("output_tag"),
                               "clean_input": bool(ctx.clean_input)})
    part, low = ci.CLEAN_TAG_PART, str(variant).lower()
    if ctx.clean_input and not (low == part or low.endswith("_" + part)):
        return f"{variant}_{part}" if variant else part
    return str(variant)


# ci settings applied to the ci cache: ci.DEFAULTS < yml charge_injection (selection, spectrum fit)
def ci_settings(ctx) -> dict:
    s = copy.deepcopy(ci.DEFAULTS)
    s.update({k: copy.deepcopy(v) for k, v in ctx.stage_cfg(ci.CONFIG_SECTION).items() if k in ci.DEFAULTS})
    validate_settings(s, ci.SETTINGS_RULES, source="plots: ci settings (yml section charge_injection)")
    s["scale_particles"] = ci.scale_samples(s["scale_particle"])
    selection = s["selection_path"]
    if selection == "auto":
        selection = next((p for p in ci.SELECTION_PREFERENCE if p in s["energy_paths"]), "s")
    s["selection_path_used"] = selection
    s["energy_response"] = er.check_settings(er.settings(ctx, {}))
    return s


# cache files of one csv: ci (npz, bars npy, json) | adcpeak (events csv, bars csv, provenance json)
def producer_files(ctx, ds, csv, source: str, tag: str) -> tuple[Path, Path, Path]:
    if source == ARBITRARY:
        events = ta.saved_events_csv(ctx, ds, csv, tag)
        return events, ta.bar_values_csv(ctx, ds, csv, tag), ta.provenance_path(events)
    return ci.event_cache_files(ctx, ds, csv, tag)


# runner options a rebuild command repeats: -test [-test-events N], -clean-input
def _mode_flags(ctx) -> str:
    flags = ""
    if ctx.test:
        flags += " -test"
        if int(ctx.n_test_events) != int((ctx.cfg.get("run") or {}).get("test_events", 5000)):
            flags += f" -test-events {int(ctx.n_test_events)}"
    if ctx.clean_input:
        flags += " -clean-input"
    return flags


# command that writes the producer cache of these runs for this scale and tag
def producer_command(ctx, datasets, source: str, tag: str) -> str:
    stage, only = ("adcpeak", " -only energy_response -bar-csv") if source == ARBITRARY else ("ci", " -only events")
    runs = " ".join(run_flags(d) for d in datasets)
    cmd = f".\\runner.ps1 {stage} {runs}".rstrip() + _mode_flags(ctx) + only
    base = str(tag or "")
    if source != ARBITRARY and ctx.clean_input:
        base = base[: -len(ci.CLEAN_TAG_PART)].rstrip("_") if base.lower().endswith(ci.CLEAN_TAG_PART) else base
    return cmd + (f"   (+ the flags of the {stage} variant '{base}')" if base else "")


# csv files of a run (sorted by name), or a stop naming the run folder
def run_csvs(ctx, ds) -> list[Path]:
    csvs = ctx.run_csvs(ds)
    if not csvs and not ctx.dry_run:
        raise SystemExit(f"plots: run {run_ref(ds)} has no csv file in {ctx.run_dir(ds, mode='fullset')}; make them "
                         f"with .\\runner.ps1 make-csv {run_flags(ds)}")
    return csvs


# command that makes the calibration table this run reads (-test: the test table, or the full-set one it falls back to)
def _calib_command(ctx) -> str:
    test_table = ctx.test and Path(ctx.calibration_file()) == Path(ctx.calibration_product())
    return ".\\runner.ps1 calib" + (" -test" if test_table else "")


# content hash of this run's calibration table, or None if there is none
def calibration_hash(ctx) -> str | None:
    path = ctx.calibration_file()
    return content_hash(path)[0] if Path(path).exists() else None


# problems of the producer cache of one csv for this run (members, scale, mode, input csv, calibration); [] = current
def producer_problems(ctx, ds, csv, source: str, files, calib_hash: str | None) -> list[str]:
    data, bars, meta = files
    if source == ARBITRARY and data.exists() and not bars.exists():
        return [f"{bars.name} is missing: adcpeak writes the bar values only with -bar-csv"]
    ok, reason = product_set_valid(meta, None, members=[data, bars])
    if not ok:
        return [reason]
    record = read_json(meta) or {}
    problems = []
    if source != ARBITRARY and source not in (record.get("compute_paths") or []):
        problems.append(f"it holds the paths {record.get('compute_paths')}, not {source} (several training "
                        "samples: give their ci variant tag with -variant, e.g. -variant s_pion_muon)")
    want_n = int(ctx.n_test_events) if ctx.test else None
    if record.get("mode") != ctx.mode or record.get("n_test_events") != want_n:
        problems.append(f"made in mode {record.get('mode')} with test events {record.get('n_test_events')}, this "
                        f"run: {ctx.mode} with {want_n}")
    made_from = (record.get("beam_csv") or {}).get("path") if source != ARBITRARY else record.get("input_path")
    expected = ctx.input_plan_item(ds, csv)["path"]
    if not made_from or canonical_path(made_from) != canonical_path(expected):
        problems.append(f"made from {made_from}, this run reads {expected}")
    calib = (record.get("calibration") if source != ARBITRARY else record.get("calibration_content")) or {}
    if calib_hash is not None and calib.get("sha256") != calib_hash:
        problems.append("made with another calibration table than the current one")
    return problems


# producer files of every csv of the runs after the check, else a stop with the reasons and the rebuild command
def check_producers(ctx, datasets, source: str, tag: str) -> dict:
    """Returns {run key: [(csv, files), ...]}."""
    calib_hash = calibration_hash(ctx)
    found, problems, stale = {}, [], []
    for ds in datasets:
        found[ds.key] = []
        for csv in run_csvs(ctx, ds):
            files = producer_files(ctx, ds, csv, source, tag)
            for p in producer_problems(ctx, ds, csv, source, files, calib_hash):
                problems.append(f"{ctx.event_source(ds, csv)} ({files[2].name}): {p}")
                if ds not in stale:
                    stale.append(ds)
            found[ds.key].append((csv, files))
    if problems:
        text = "\n  - ".join(problems)
        raise SystemExit(f"plots: the {source} values of the {'ci' if source != ARBITRARY else 'adcpeak'} cache "
                         f"cannot be drawn:\n  - {text}\n  Make them with: "
                         f"{producer_command(ctx, stale, source, tag)}")
    return found


# selected events of one csv: (pf_event, event energy, reconstructed bars)
def csv_events(source: str, files, s_ci: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source == ARBITRARY:
        ev = pd.read_csv(files[0], usecols=ADCPEAK_EVENT_COLUMNS, float_precision="round_trip")
        sel = ev["selected"].astype(bool).to_numpy() & (ev["n_bars"].to_numpy() > 0)
        return (ev["pf_event"].to_numpy(np.int64)[sel], ev["arbitrary_energy"].to_numpy(np.float64)[sel],
                ev["n_bars"].to_numpy(np.int64)[sel])
    ev = ci.load_event_arrays(files[0])
    sel = ci.select_events(ev, s_ci)
    return (ev["pf_event"].astype(np.int64)[sel], ev["energy"][source].astype(np.float64)[sel],
            ev["n_bars"].astype(np.int64)[sel])


# counted bars of one chunk: bars with a finite bar energy
def _counted(index: int, pf, layer, strip, bar, end0, end1) -> "Counted":
    bar = np.asarray(bar, dtype=np.float64)
    sel = np.isfinite(bar)
    return Counted(event_key(index, np.asarray(pf)[sel]), np.asarray(layer, dtype=np.intp)[sel],
                   np.asarray(strip, dtype=np.intp)[sel], bar[sel], np.asarray(end0, dtype=np.float64)[sel],
                   np.asarray(end1, dtype=np.float64)[sel])


# bars of one csv chunk by chunk (ci: memory-mapped records, closed after the pass)
def csv_bars(source: str, files, index: int, chunksize: int):
    if source == ARBITRARY:
        for chunk in iter_csv(files[1], ADCPEAK_BAR_COLUMNS, chunksize):
            yield _counted(index, *(chunk[c].to_numpy() for c in ADCPEAK_BAR_COLUMNS))
        return
    records = ci.load_bar_records(files[1], mmap=True)
    names = ("pf_event", "layer", "strip", f"bar_energy_{source}_mip", f"end0_energy_{source}_mip",
             f"end1_energy_{source}_mip")
    try:
        for start in range(0, len(records), chunksize):
            part = records[start:start + chunksize]
            yield _counted(index, *(np.array(part[n]) for n in names))
            del part
    finally:
        mm = getattr(records, "_mmap", None)
        del records
        if mm is not None:
            mm.close()  # a held map blocks ci from replacing the file on Windows


# --------------------------------------------------------------------------
# CLI and settings
# --------------------------------------------------------------------------
# plots command-line options
def add_arguments(p: argparse.ArgumentParser) -> None:
    outputs.add_only_argument(p, UNITS, UNIT_GROUPS)
    g = p.add_argument_group("plots stage")
    g.add_argument("--energy", "-energy", dest="energy", choices=ENERGY_SOURCES, default=None,
                   help="energy scale: s, pureci, adc_only (ci, MIP; s_<sample> with several training samples) or "
                        f"arbitrary (adcpeak Arbitrary Energy, a.u.) (default: {DEFAULTS['energy_source']})")
    g.add_argument("--variant", "-variant", dest="variant", default=None, metavar="TAG",
                   help="variant tag of the producer cache (ci: e.g. s_muon, s_pion_muon or an -output-tag; adcpeak: "
                        "none, mean_fallback or an -output-tag); with -clean-input ci's clean part is added "
                        "(default: the tag of a ci / adcpeak run with the yml settings)")
    g.add_argument("--picker", "-picker", dest="picker", choices=PICKERS, default=None,
                   help=f"event picker (default: {DEFAULTS['event_picker']})")
    g.add_argument("--n-events", "-n-events", dest="n_events", type=int, default=None, metavar="N",
                   help=f"number of event displays (default: {DEFAULTS['n_event_displays']})")
    g.add_argument("--seed", "-seed", dest="seed", type=int, default=None,
                   help=f"seed of the random picker (default: {DEFAULTS['random_seed']})")
    g.add_argument("--events", "-events", dest="events", type=int, nargs="+", default=None, metavar="ID",
                   help="pf_event ids to display (overrides -picker; in every csv of the run; default: none)")
    g.add_argument("--energy-window", "-energy-window", dest="energy_window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="pick only events with LO <= E_event < HI (default: none)")
    g.add_argument("--color-scale", "-color-scale", dest="color_scale", choices=COLOR_SCALES, default=None,
                   help=f"event-display colour scale (default: {DEFAULTS['color_scale']})")
    g.add_argument("--vmax", "-vmax", dest="vmax", type=float, default=None,
                   help=f"maximum of the fixed colour scale (default: {DEFAULTS['color_vmax_mip']} MIP)")
    g.add_argument("--per-end", "-per-end", dest="per_end", action="store_true",
                   help="event displays: add the per-end 2x2 and end0-end1 difference pages per event (default: off)")
    g.add_argument("--hitmap-events", "-hitmap-events", dest="hitmap_events", type=int, default=None, metavar="N",
                   help=f"per-event hit displays in the hit-map pdf (default: {DEFAULTS['hitmap_events']})")
    g.add_argument("--min-bin-count", "-min-bin-count", dest="min_bin_count", type=int, default=None, metavar="N",
                   help=f"event overview: only bins with >= N events (default: {DEFAULTS['min_bin_count']})")


# plots settings: DEFAULTS < yml section plots < command line, checked; units from -only
def resolve_settings(ctx, args) -> dict:
    yml = ctx.stage_cfg(CONFIG_SECTION)
    unknown = sorted(set(yml) - set(DEFAULTS))
    if unknown:  # a typo or an unknown key is never silently ignored
        raise SettingsError(f"plots: unknown setting(s) {unknown} in the yml section {CONFIG_SECTION} (configure/"
                            f"config.yml or local.yml); the valid keys are those of multiscale_plots.DEFAULTS")
    s = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(yml)}
    cli = {
        "energy_source": getattr(args, "energy", None),
        "variant": getattr(args, "variant", None),
        "event_picker": getattr(args, "picker", None),
        "n_event_displays": getattr(args, "n_events", None),
        "random_seed": getattr(args, "seed", None),
        "color_scale": getattr(args, "color_scale", None),
        "hitmap_events": getattr(args, "hitmap_events", None),
        "min_bin_count": getattr(args, "min_bin_count", None),
    }
    s.update({k: v for k, v in cli.items() if v is not None})
    if getattr(args, "per_end", False):
        s["per_end_displays"] = True
    s["fixed_vmax_override"] = getattr(args, "vmax", None)
    s["explicit_events"] = getattr(args, "events", None)
    s["energy_window"] = getattr(args, "energy_window", None)
    if isinstance(s["variant"], str):
        s["variant"] = s["variant"].strip().lstrip("_")
    validate_settings(s, SETTINGS_SPEC, source="plots settings (configure/config.yml 'plots' + command line)")
    if s["energy_window"] is not None:
        s["energy_window"] = [float(v) for v in s["energy_window"]]
    s["units"] = [u for u in UNITS if ctx.wants(u)]
    s["energy_units"] = [u for u in s["units"] if u in ENERGY_UNITS]
    return s


# pdf rendering processes: -workers, else plots.workers (default 1)
def render_workers(args, s: dict) -> int:
    explicit = getattr(args, "workers", None)
    return max(1, int(explicit if explicit else (s.get("workers") or 1)))


# reduced arrays the energy units of a run need (bars, layers, values, ends, rows)
def energy_needs(s: dict) -> set:
    return {n for u in s["energy_units"] for n in UNIT_NEEDS[u]}


# stored product sets of a run: energy and / or hitmap
def _cache_kinds(s: dict) -> list[str]:
    return (["energy"] if s["energy_units"] else []) + (["hitmap"] if "hitmap" in s["units"] else [])


# inputs and outputs of the run (for -dry-run, check and the run log); deletes nothing
def io_plan(ctx, args, datasets) -> list[dict]:
    s = resolve_settings(ctx, args)
    from_saved = bool(getattr(args, "from_saved", False))
    source = s["energy_source"]
    tag = producer_tag(ctx, s) if s["energy_units"] else ""
    plan = []
    calib = ctx.calibration_file() if "hitmap" in s["units"] else None
    if calib is not None and not from_saved:
        plan.append({"role": "in", "path": calib, "produced_by": _calib_command(ctx)})
    dest, _ = pdf_destinations(ctx, result_slots(ctx, datasets, s, tag), dry=True)
    calib_hash = calibration_hash(ctx) if s["energy_units"] and not from_saved else None
    for ds in datasets:
        csvs = ctx.run_csvs(ds)
        if not from_saved and (s["energy_units"] or "hitmap" in s["units"]) and not csvs:
            plan += ctx.input_plan_items(ds)
        if s["energy_units"] and not from_saved:
            make = producer_command(ctx, [ds], source, tag)
            for csv in csvs:
                files = producer_files(ctx, ds, csv, source, tag)
                stale = files[0].exists() and bool(producer_problems(ctx, ds, csv, source, files, calib_hash))
                for f in files[:2]:
                    plan.append({"role": "in", "kind": "product", "path": f, "produced_by": make,
                                 **({"stale": True} if stale else {})})
        if "hitmap" in s["units"] and not from_saved:
            plan += [ctx.input_plan_item(ds, csv, f".\\runner.ps1 make-csv {run_flags(ds)}") for csv in csvs]
        for unit in s["units"] if ctx.make_plots else ():
            key = result_key(ctx, ds, unit, s, tag)
            path, saved = dest[key]
            item = {"role": "out", "path": path, "result": key}
            if not saved:
                item["note"] = "not saved: no --output"
            plan.append(item)
        for kind in _cache_kinds(s) if ctx.make_plots else ():
            npz, meta = cache_paths(ctx, ds, kind, source, tag)
            if from_saved:
                plan.append({"role": "cache", "path": meta, "members": [npz],
                             "expected": _identity(ctx, ds, kind, s, tag, _planned_sources(ctx, ds, kind, s, tag),
                                                   calib)})
            elif s["save_cache"]:
                plan += [{"role": "out", "path": npz}, {"role": "out", "path": meta}]
    return plan


# --------------------------------------------------------------------------
# streamed input: counted bars and raw hits
# --------------------------------------------------------------------------
# counted bars of one chunk: one row per event x bar with a finite bar energy
@dataclass
class Counted:
    key: np.ndarray
    layer: np.ndarray
    strip: np.ndarray
    energy: np.ndarray
    end0: np.ndarray
    end1: np.ndarray

    # rows selected by a mask
    def take(self, mask) -> "Counted":
        return Counted(self.key[mask], self.layer[mask], self.strip[mask], self.energy[mask], self.end0[mask],
                       self.end1[mask])

    # join chunks into one table
    @staticmethod
    def concat(parts: list) -> "Counted":
        # one column of every chunk, concatenated
        def cat(name, dtype):
            arrays = [getattr(p, name) for p in parts]
            return np.concatenate(arrays).astype(dtype, copy=False) if arrays else np.empty(0, dtype=dtype)

        return Counted(cat("key", np.int64), cat("layer", np.intp), cat("strip", np.intp), cat("energy", np.float64),
                       cat("end0", np.float64), cat("end1", np.float64))


# raw hit flags of one beam-csv chunk (end hit = adc_sum > threshold)
@dataclass
class HitRows:
    key: np.ndarray
    cell: np.ndarray
    hit0: np.ndarray
    hit1: np.ndarray


# calibration lookup arr[layer, strip, end] of the hit threshold
@dataclass
class BeamTables:
    threshold: np.ndarray
    threshold_column: str


# hit-threshold lookup of the beam csv (stop if the threshold column is missing)
def beam_tables(calib: pd.DataFrame, s: dict) -> BeamTables:
    threshold_column = s["hit_threshold_column"]
    if threshold_column not in calib.columns:
        raise SystemExit(f"plots: calibration table has no column '{threshold_column}' (hit_threshold_column)")
    return BeamTables(calibration_lookup(calib, threshold_column), threshold_column)


# read a beam csv chunk by chunk: (event keys, raw hits)
def iter_beam_csv(path: Path, tables: BeamTables, chunksize: int, index: int = 0):
    for chunk in iter_csv(path, BEAM_COLUMNS, chunksize):
        key = event_key(index, chunk["pf_event"].to_numpy(np.int64))
        if not len(key):
            continue
        layer = chunk["layer"].to_numpy(np.intp)
        strip = chunk["strip"].to_numpy(np.intp)
        adc = [chunk[f"adc_sum_end{e}"].to_numpy(np.float64) for e in geo.ENDS]
        thr = [tables.threshold[layer, strip, e] for e in geo.ENDS]
        hit = [np.isfinite(thr[e]) & (adc[e] > thr[e]) for e in geo.ENDS]  # strict '>'
        yield np.unique(key), HitRows(key, cell_index(layer, strip), hit[0], hit[1])


# --------------------------------------------------------------------------
# reduction of one run
# --------------------------------------------------------------------------
# counted-bar rows of the picked events (event displays)
class EventRows:

    # set up the collector for the picked event keys
    def __init__(self, keys):
        self.keys = np.unique(np.asarray(keys, dtype=np.int64))
        self.parts: list[Counted] = []

    # keep the rows of picked events
    def add(self, counted: Counted) -> None:
        mask = in_sorted(counted.key, self.keys)
        if mask.any():
            self.parts.append(counted.take(mask))

    # the held rows as one table
    def table(self) -> Counted:
        return Counted.concat(self.parts)


# reduced energies of one run
@dataclass
class EnergySummary:
    csv_names: list             # csv files of the run (event key: their index)
    event_keys: np.ndarray      # selected events with >= 1 reconstructed bar, sorted
    event_energy: np.ndarray    # producer event energy per event
    event_n_bars: np.ndarray    # reconstructed bars per event
    n_counted: int              # counted bars in total (bars read)
    cell_hits: np.ndarray       # (N_CELLS,) counted bars per cell
    cell_total: np.ndarray      # (N_CELLS,) energy sum per cell
    layer_energy: np.ndarray | None  # (n_events, 19) energy per event x layer
    layer_hits: np.ndarray | None    # (n_events, 19) counted bars per event x layer
    values: np.ndarray | None        # every counted bar energy (dist_bar)
    value_cells: np.ndarray | None   # their cells (uint8)
    end_values: np.ndarray | None    # every finite end energy of a counted bar (dist_channel)
    end_channels: np.ndarray | None  # their channels: cell * 2 + end (uint16)
    holds: dict                 # which reduced arrays exist: bars, layers, values, ends
    units: str                  # axis label of a bar / end value
    unit: str                   # short unit for titles
    axis: str                   # axis label of the event value
    source: str
    label: str                  # scale + variant, shown in the titles
    definition: str
    input_dir: Path             # producer cache folder of the run

    # selected events
    @property
    def n_events(self) -> int:
        return len(self.event_keys)

    # (event energy, reconstructed bars) of one event (NaN, 0 if none)
    def event_total(self, key) -> tuple[float, int]:
        i = np.searchsorted(self.event_keys, key)
        if i < len(self.event_keys) and self.event_keys[i] == key:
            return float(self.event_energy[i]), int(self.event_n_bars[i])
        return float("nan"), 0


# streamed sums of the counted bars of one run: per cell, per event x layer, values and end values
class EnergyAccumulator:

    # empty sums for the known event keys
    def __init__(self, keys: np.ndarray, needs: set, rows: EventRows | None = None):
        n = len(keys)
        self.keys = keys
        self.cell_hits = np.zeros(N_CELLS, dtype=np.int64)
        self.cell_total = np.zeros(N_CELLS, dtype=np.float64)
        self.layer_energy = np.zeros((n, geo.N_LAYERS), dtype=np.float64) if "layers" in needs else None
        self.layer_hits = np.zeros((n, geo.N_LAYERS), dtype=np.int32) if "layers" in needs else None
        self.values = [] if "values" in needs else None
        self.value_cells = [] if "values" in needs else None
        self.end_values = [] if "ends" in needs else None
        self.end_channels = [] if "ends" in needs else None
        self.n_counted = 0
        self.rows = rows

    # add one chunk of counted bars (bars of other events are dropped)
    def add(self, counted: Counted) -> None:
        known = in_sorted(counted.key, self.keys)
        if not known.all():
            counted = counted.take(known)
        if not len(counted.key):
            return
        if self.rows is not None:
            self.rows.add(counted)
        cell = cell_index(counted.layer, counted.strip)
        self.n_counted += len(cell)
        self.cell_hits += np.bincount(cell, minlength=N_CELLS)
        self.cell_total += np.bincount(cell, weights=counted.energy, minlength=N_CELLS)  # energy sum per cell
        if self.layer_energy is not None:
            row = np.searchsorted(self.keys, counted.key)
            layer = counted.layer - 1
            np.add.at(self.layer_energy, (row, layer), counted.energy)  # layer energy = Σ its bar energies
            np.add.at(self.layer_hits, (row, layer), 1)
        if self.values is not None:
            self.values.append(counted.energy.astype(np.float32))
            self.value_cells.append(cell.astype(np.uint8))
        if self.end_values is not None:
            for e, values in enumerate((counted.end0, counted.end1)):
                ok = np.isfinite(values)
                self.end_values.append(values[ok].astype(np.float32))
                self.end_channels.append((cell[ok] * 2 + e).astype(np.uint16))

    # the EnergySummary of the run
    def finish(self, events: tuple, read_bars: bool, **meta) -> EnergySummary:
        # one list of chunks as an array
        def joined(parts, dtype):
            if parts is None:
                return None
            return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)

        holds = {"bars": bool(read_bars), "layers": self.layer_energy is not None and read_bars,
                 "values": self.values is not None and read_bars, "ends": self.end_values is not None and read_bars}
        return EnergySummary(event_keys=events[0], event_energy=events[1], event_n_bars=events[2],
                             n_counted=self.n_counted, cell_hits=self.cell_hits, cell_total=self.cell_total,
                             layer_energy=self.layer_energy, layer_hits=self.layer_hits,
                             values=joined(self.values, np.float32), value_cells=joined(self.value_cells, np.uint8),
                             end_values=joined(self.end_values, np.float32),
                             end_channels=joined(self.end_channels, np.uint16), holds=holds, **meta)


# raw hit counts per cell and the hit cells of the first events
@dataclass
class HitCounts:
    n_events: int
    both: np.ndarray
    end0: np.ndarray
    end1: np.ndarray
    first_events: np.ndarray
    first_key: np.ndarray
    first_cell: np.ndarray
    threshold_column: str
    csv_names: list = field(default_factory=list)


# streamed hit counts (both ends, end0, end1) + the hit cells of the first N events
class HitAccumulator:

    # empty counts
    def __init__(self, n_first: int, threshold_column: str):
        self.counts = np.zeros((3, N_CELLS), dtype=np.int64)
        self.all_events = np.empty(0, dtype=np.int64)
        self.n_first = int(n_first)
        self.first = np.empty(0, dtype=np.int64)
        self.key, self.cell = [], []
        self.threshold_column = threshold_column

    # add one chunk of hit flags
    def add(self, events: np.ndarray, hits: HitRows) -> None:
        both = hits.hit0 & hits.hit1
        self.counts[0] += np.bincount(hits.cell[both], minlength=N_CELLS)
        self.counts[1] += np.bincount(hits.cell[hits.hit0], minlength=N_CELLS)
        self.counts[2] += np.bincount(hits.cell[hits.hit1], minlength=N_CELLS)
        new = events[~in_sorted(events, self.all_events)]
        if new.size:
            self.all_events = np.union1d(self.all_events, new)
        if not self.n_first:
            return
        self.first = np.union1d(self.first, events)[:self.n_first]
        cutoff = self.first[-1] if len(self.first) == self.n_first else np.iinfo(np.int64).max
        keep = [k <= cutoff for k in self.key]  # drop the rows of events that fell out of the first N
        self.key = [k[m] for k, m in zip(self.key, keep)]
        self.cell = [c[m] for c, m in zip(self.cell, keep)]
        sel = both & (hits.key <= cutoff)
        self.key.append(hits.key[sel])
        self.cell.append(hits.cell[sel])

    # the HitCounts of the run
    def finish(self, csv_names) -> HitCounts:
        key = np.concatenate(self.key).astype(np.int64, copy=False) if self.key else np.empty(0, dtype=np.int64)
        cell = np.concatenate(self.cell).astype(np.intp, copy=False) if self.cell else np.empty(0, dtype=np.intp)
        sel = np.isin(key, self.first)
        return HitCounts(len(self.all_events), self.counts[0], self.counts[1], self.counts[2], self.first, key[sel],
                         cell[sel], self.threshold_column, list(csv_names))


# --------------------------------------------------------------------------
# event pickers and colour scales
# --------------------------------------------------------------------------
# event keys to display: -events, first N, N highest-energy or N random (seeded)
def pick_events(keys: np.ndarray, energy: np.ndarray, unit: str, s: dict) -> tuple[np.ndarray, str]:
    """Returns (keys, description); an -events id found in no csv gets the key of the first csv."""
    if s.get("explicit_events"):
        ids = np.unique(np.asarray(s["explicit_events"], dtype=np.int64))
        pf = keys & PF_MASK
        found = keys[np.isin(pf, ids)]
        missing = ids[~np.isin(ids, pf)]
        return np.concatenate([found, missing]).astype(np.int64), "explicit -events list"
    n = int(s["n_event_displays"])
    picker = s["event_picker"]
    window = s.get("energy_window")
    pool_text = ""
    if window is not None:
        lo, hi = window
        inside = (energy >= lo) & (energy < hi)
        keys, energy = keys[inside], energy[inside]
        pool_text = f" with {lo:g} <= E < {hi:g} {unit}"
    if n == 0:
        return np.empty(0, dtype=np.int64), "no event displays"
    if picker == "first":
        return keys[:n], f"first {n} events{pool_text} (ascending csv, pf_event)"
    if picker == "highest":
        order = np.lexsort((keys, -energy))[:n]
        return keys[order], f"{n} highest-energy events{pool_text}"
    rng = np.random.default_rng(int(s["random_seed"]))
    chosen = rng.choice(keys, size=min(n, len(keys)), replace=False) if len(keys) else keys
    return np.sort(chosen), f"{n} random events{pool_text} (seed {int(s['random_seed'])})"


# colour scale of the event displays: fixed or robust
@dataclass
class ColorScale:
    mode: str
    vmax: float
    percentile: float
    max_fraction: float
    vmax_min: float
    vmax_max: float

    # colour maximum: fixed vmax, or robust = max(vmin, min(max(p98, 0.8 x max), vmax_max))
    def limit(self, values) -> float:
        if self.mode == "fixed":
            return self.vmax
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return self.vmax_min
        return max(self.vmax_min, min(max(np.percentile(v, self.percentile), self.max_fraction * v.max()),
                                      self.vmax_max))

    # text of the scale for the summary page
    def describe(self, unit: str) -> str:
        if self.mode == "fixed":
            return f"fixed 0..{self.vmax:g} {unit}"
        return (f"robust per event: max({self.vmax_min:g}, min(max(p{self.percentile:g}, "
                f"{self.max_fraction:g} x max), {self.vmax_max:g})) {unit}")


# colour scale of a run (-vmax forces fixed; arbitrary units: robust, no upper clamp)
def color_scale(s: dict, bars) -> ColorScale:
    arbitrary = bars.unit != "MIP"
    mode = s["color_scale"]
    vmax = s.get("fixed_vmax_override")
    if vmax is not None:
        mode = "fixed"
    elif mode == "fixed" and arbitrary:
        mode = "robust"  # no fixed scale is defined for arbitrary units
    return ColorScale(mode=mode, vmax=float(vmax if vmax is not None else s["color_vmax_mip"]),
                      percentile=float(s["robust_percentile"]), max_fraction=float(s["robust_max_fraction"]),
                      vmax_min=float(s["robust_vmax_min"]),
                      vmax_max=np.inf if arbitrary else float(s["robust_vmax_max_mip"]))


# --------------------------------------------------------------------------
# reduced figure data (all fixed size except the picked-event grids)
# --------------------------------------------------------------------------
# text with non-ASCII characters replaced
def _ascii(text) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


# path relative to data_analysis, else .../<folder>/<file>
def _short_path(path) -> str:
    p = Path(path)
    try:
        return _ascii(p.resolve().relative_to(REPO_ROOT).as_posix())
    except (ValueError, OSError):
        return _ascii(".../" + "/".join(p.parts[-2:]))


# spectrum bin edges on [min(0, min), 1.05 x quantile 0.999]
def _spectrum_edges(values: np.ndarray, s: dict) -> np.ndarray:
    v = values[np.isfinite(values)] if values is not None else np.empty(0)
    hi = float(np.quantile(v, float(s["spectrum_quantile"]))) * 1.05 if v.size else 1.0
    lo = min(0.0, float(v.min())) if v.size else 0.0
    if not hi > lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(s["spectrum_bins"]) + 1)


# histogram of the values of every group at once: (counts, overflow)
def _grouped_histogram(group: np.ndarray, values: np.ndarray, n_groups: int, edges: np.ndarray,
                       block: int = 1 << 20):
    n_bins = len(edges) - 1
    counts = np.zeros(n_groups * n_bins, dtype=np.int64)
    overflow = np.zeros(n_groups, dtype=np.int64)
    for start in range(0, len(values), block):
        v = np.asarray(values[start:start + block], dtype=np.float64)
        g = np.asarray(group[start:start + block], dtype=np.intp)
        idx = np.searchsorted(edges, v, side="right") - 1
        idx[v == edges[-1]] = n_bins - 1  # np.histogram: last bin closed
        inside = (idx >= 0) & (idx < n_bins)
        counts += np.bincount(g[inside] * n_bins + idx[inside], minlength=n_groups * n_bins)
        overflow += np.bincount(g[~inside & (v > edges[-1])], minlength=n_groups)
    return counts.reshape(n_groups, n_bins), overflow


# event-energy overview: 300-bin histogram on [0, max], n-bar distribution, 2D histogram
def overview_data(energy: np.ndarray, n_bars: np.ndarray, s: dict) -> dict:
    if not len(energy):
        return {"empty": True}
    emax = float(np.max(energy))
    emax = emax if emax > 0 else 1.0
    edges = np.linspace(0.0, emax, int(s["overview_bins"]) + 1)
    hist, _ = np.histogram(energy, bins=edges)
    nb_edges = np.arange(-0.5, n_bars.max() + 1.5)
    nb_counts, _ = np.histogram(n_bars, bins=nb_edges)
    h2, xe, ye = np.histogram2d(n_bars, energy, bins=[nb_edges, np.linspace(0.0, emax, 101)])
    return {"empty": False, "emax": emax, "edges": edges, "hist": hist, "min_count": int(s["min_bin_count"]),
            "n": len(energy), "mean": energy.mean(), "median": np.median(energy), "nb_edges": nb_edges,
            "nb_counts": nb_counts, "nb_mean": n_bars.mean(), "nb_median": np.median(n_bars),
            "h2": h2, "xe": xe, "ye": ye}


# bar maps: occupancy = n_hits / n_events, mean_hit = total / n_hits, mean_event = total / n_events
def cell_statistics(bars: EnergySummary) -> dict:
    n_events = max(bars.n_events, 1)
    n_hits = bars.cell_hits.reshape(geo.N_LAYERS, geo.N_POSITIONS)
    total = bars.cell_total.reshape(geo.N_LAYERS, geo.N_POSITIONS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_hit = np.where(n_hits > 0, total / n_hits, np.nan)
    out = {"n_hits": n_hits, "occupancy": n_hits / n_events, "mean_hit": mean_hit, "mean_event": total / n_events}
    return {k: np.where(PHYSICAL_MASK, v, np.nan) for k, v in out.items()}


# per-layer mean and standard error: sem = std(ddof=1) / sqrt(n)
def _mean_sem(m: np.ndarray):
    n = m.shape[0]
    if n == 0:
        return np.full(m.shape[1], np.nan), np.full(m.shape[1], np.nan)
    sem = m.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(m.shape[1], np.nan)
    return m.mean(axis=0), sem


# title head: '<run label> [<scale label>]'
def _title_head(ds_label: str, tag: str) -> str:
    return f"{ds_label} [{tag}]"


# [TEST MODE] / [CLEAN INPUT] marks
def _input_marks(ctx) -> str:
    return ("   [TEST MODE]" if ctx.test else "") + ("   [CLEAN INPUT]" if getattr(ctx, "clean_input", False) else "")


# summary-page lines shared by the energy pdfs
def _common_lines(ctx, ds, bars: EnergySummary, info: dict) -> list[str]:
    names = bars.csv_names
    lines = [
        f"run           : {run_ref(ds)} ({ds.label})",
        f"energy scale  : {bars.label}  [{bars.units}]",
        f"definition    : {bars.definition}",
        f"input         : {_short_path(bars.input_dir)} ({len(names)} csv file{'' if len(names) == 1 else 's'}: "
        f"{', '.join(names)})" + _input_marks(ctx),
    ]
    if info.get("from_saved"):
        lines.append(f"re-drawn from : {info['from_saved']}")
    if ctx.test:
        lines.append(f"test mode     : the producer's -test cache ({int(ctx.n_test_events)} events per csv)")
    lines += [
        f"selected events (>= 1 reconstructed bar) : {bars.n_events}",
        "counted bars                             : "
        + (f"{bars.n_counted}" if bars.holds.get("bars") else "not read (no bar unit)"),
        f"mean event energy                        : {bars.event_energy.mean():.4g} {bars.unit}" if bars.n_events
        else "mean event energy                        : -",
        "layout        : x = strip position 0..11 (layers 1-9: strips 0-7 at positions 2-9), y = layer;",
        "                odd and even layers in separate panels",
    ]
    return lines


# (19, 12) grids of one event: bar energy and, if asked, each end
def _event_grids(rows: Counted, key, with_ends: bool):
    sel = np.flatnonzero(rows.key == key)
    layer, strip = rows.layer[sel], rows.strip[sel]
    grid = geo.fill_layer_grid(layer, strip, rows.energy[sel])
    ends = None
    if with_ends:
        ends = [geo.fill_layer_grid(layer, strip, rows.end0[sel]), geo.fill_layer_grid(layer, strip, rows.end1[sel])]
    return grid, ends


# base of every energy payload: labels and the summary lines
def _base_payload(ctx, ds, bars: EnergySummary, info: dict) -> dict:
    return {"label": ds.label, "tag": bars.label, "source": bars.source, "unit": bars.unit, "units": bars.units,
            "axis": bars.axis, "lines": _common_lines(ctx, ds, bars, info), "n_events": bars.n_events,
            "csv_names": list(bars.csv_names)}


# figure data of dist_event: event energies, the ci spectrum fit, the event / n-bar overview
def dist_event_payload(ctx, ds, bars: EnergySummary, s: dict, info: dict, s_ci: dict | None) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    values, fit = bars.event_energy, None
    if bars.source == ARBITRARY:
        text = "none (the adcpeak spectrum fits are drawn by adcpeak, energy_spectra.pdf)"
    elif len(values) < 2:
        text = "none (fewer than 2 events)"
    else:
        fit = ci.spectrum_fit(values, ds.particle, s_ci)["fit"]
        text = f"ci spectrum fit of a {ds.particle} run ({fit.get('kind')}, {fit.get('status', '')})"
    p["lines"] += ["", f"spectrum fit  : {text}"]
    p.update(values=values, fit=fit, particle=ds.particle, ci=s_ci if fit is not None else None,
             overview=overview_data(values, bars.event_n_bars, s))
    return p


# figure data of event_display: the picked events
def event_display_payload(ctx, ds, bars: EnergySummary, s: dict, picked: np.ndarray, how: str, rows: Counted,
                          info: dict) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    scale = color_scale(s, bars)
    per_end = bool(s["per_end_displays"])
    in_input = in_sorted(picked, bars.event_keys) if len(picked) else np.zeros(0, dtype=bool)
    p["lines"] += [
        "",
        f"picked events: {how}",
        f"colour scale : {scale.describe(bars.unit)}",
        f"per-end pages: {'yes' if per_end else 'no'}",
        "",
        f"{'event':>34} {'E [' + bars.unit + ']':>14} {'bars':>6}",
    ] + [f"{event_name(k, bars.csv_names):>34} {bars.event_total(k)[0]:>14.3f} {bars.event_total(k)[1]:>6d}"
         for k in picked[:60]]
    if len(picked) > 60:
        p["lines"].append(f"... {len(picked) - 60} more")
    events = []
    for key, known in zip(picked, in_input):
        grid, ends = _event_grids(rows, key, per_end)
        total, n_bars = bars.event_total(key)
        events.append({"id": int(key), "name": event_name(key, bars.csv_names), "grid": grid, "ends": ends,
                       "total": total, "n_bars": n_bars, "in_input": bool(known)})
    p.update(scale=scale, cmap=s["event_colormap"], per_end=per_end, events=events,
             difference={k: s[k] for k in ("difference_percentile", "difference_vmax_min", "difference_vmax_max_mip")})
    return p


# figure data of dist_bar: bar energy spectra per bar
def dist_bar_payload(ctx, ds, bars: EnergySummary, s: dict, info: dict) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    edges = _spectrum_edges(bars.values, s)
    counts, overflow = _grouped_histogram(bars.value_cells, bars.values, N_CELLS, edges)
    p.update(edges=edges, counts=counts, overflow=overflow, n=bars.cell_hits, total=bars.cell_total)
    return p


# figure data of dist_channel: end energy spectra per channel (cell * 2 + end)
def dist_channel_payload(ctx, ds, bars: EnergySummary, s: dict, info: dict) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    edges = _spectrum_edges(bars.end_values, s)
    counts, overflow = _grouped_histogram(bars.end_channels, bars.end_values, N_CHANNELS, edges)
    channel = bars.end_channels.astype(np.intp)
    p["lines"] += ["", "channel       : one bar end; its value in an event = the producer's end energy (finite ends "
                       "of the counted bars)"]
    p.update(edges=edges, counts=counts, overflow=overflow, n=np.bincount(channel, minlength=N_CHANNELS),
             total=np.bincount(channel, weights=bars.end_values.astype(np.float64), minlength=N_CHANNELS))
    return p


# figure data of bar_maps: occupancy, mean energy per hit, mean energy per event
def bar_maps_payload(ctx, ds, bars: EnergySummary, s: dict, info: dict) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    p.update(maps=cell_statistics(bars), cmap=s["map_colormap"], annotate=bool(s["annotate_maps"]))
    return p


# figure data of layer: mean, sem and distributions per layer
def layer_payload(ctx, ds, bars: EnergySummary, s: dict, info: dict) -> dict:
    p = _base_payload(ctx, ds, bars, info)
    energy, hits = bars.layer_energy, bars.layer_hits
    e_mean, e_sem = _mean_sem(energy)
    h_mean, h_sem = _mean_sem(hits.astype(float))
    hit_layer = hits > 0
    e_edges = _spectrum_edges(energy[hit_layer], s)
    e_counts = np.zeros((geo.N_LAYERS, len(e_edges) - 1), dtype=np.int64)
    e_n = np.zeros(geo.N_LAYERS, dtype=np.int64)
    e_avg = np.full(geo.N_LAYERS, np.nan)
    m_edges = np.arange(-0.5, geo.N_STRIPS_WIDE + 1.5)
    m_counts = np.zeros((geo.N_LAYERS, len(m_edges) - 1), dtype=np.int64)
    m_avg = np.full(geo.N_LAYERS, np.nan)
    for i in range(geo.N_LAYERS):
        values = energy[hit_layer[:, i], i]
        e_counts[i], _ = np.histogram(values, bins=e_edges)
        e_n[i] = len(values)
        if len(values):
            e_avg[i] = values.mean()
        m_counts[i], _ = np.histogram(hits[:, i], bins=m_edges)
        if len(hits):
            m_avg[i] = hits[:, i].mean()
    p.update(e_mean=e_mean, e_sem=e_sem, h_mean=h_mean, h_sem=h_sem, e_edges=e_edges, e_counts=e_counts, e_n=e_n,
             e_avg=e_avg, m_edges=m_edges, m_counts=m_counts, m_avg=m_avg)
    return p


# figure data of the hit-map pdf: hit fraction = hits / events per cell
def hitmap_payload(ctx, ds, hits: HitCounts, beam_csvs: list, calib_path: Path, s: dict,
                   from_saved: str = "") -> dict:
    n = max(hits.n_events, 1)
    grids = {k: np.where(PHYSICAL_MASK, v.reshape(geo.N_LAYERS, geo.N_POSITIONS) / n, np.nan)
             for k, v in (("both", hits.both), ("end0", hits.end0), ("end1", hits.end1))}
    rule = f"adc_sum > {hits.threshold_column}"
    lines = [f"run          : {run_ref(ds)} ({ds.label})"]
    lines += [f"beam csv     : {_short_path(p)}" + _input_marks(ctx) for p in beam_csvs]
    lines.append(f"calibration  : {_short_path(calib_path)}")
    if from_saved:
        lines.append(f"re-drawn from: {from_saved}")
    if ctx.test:
        lines.append(f"test mode    : the -test csv of each csv ({int(ctx.n_test_events)} events)")
    lines += [
        f"events       : {hits.n_events}",
        f"hit rule     : bar hit if both ends {rule}",
        f"per-end maps : end hit if {rule}",
        f"hit displays : first {len(hits.first_events)} events (ascending csv, pf_event)",
    ]
    names = [event_name(k, hits.csv_names) for k in hits.first_events]
    return {"label": ds.label, "lines": lines, "grids": grids, "rule": rule, "n_events": hits.n_events,
            "first_events": hits.first_events, "first_names": names, "first_key": hits.first_key,
            "first_cell": hits.first_cell, "cmap": s["map_colormap"], "annotate": bool(s["annotate_maps"])}


# --------------------------------------------------------------------------
# drawing primitives (physical layout, odd / even views)
# --------------------------------------------------------------------------
# detector axes: cells of size 1, ticks at the centres, layer 1 at the bottom
def layout_axes(ax, xlabel: bool = True) -> None:
    ax.set_xlim(0, geo.N_POSITIONS)
    ax.set_ylim(0, geo.N_LAYERS)
    ax.set_xticks(np.arange(geo.N_POSITIONS) + 0.5)
    ax.set_xticklabels(range(geo.N_POSITIONS), fontsize=7)
    ax.set_yticks(np.arange(geo.N_LAYERS) + 0.5)
    ax.set_yticklabels(range(1, geo.N_LAYERS + 1), fontsize=7)
    if xlabel:
        ax.set_xlabel("strip position")


# draw the cells of the odd or even layers as coloured rectangles
def draw_view(ax, grid: np.ndarray, view: str, norm, cmap, outline: bool = True):
    layers = VIEW_LAYERS[view]
    sub = grid[layers - 1]
    rows, positions = np.nonzero(np.isfinite(sub))  # layer ascending, then position ascending
    pc = None
    if rows.size:
        rects = [Rectangle((p, layers[r] - 1), 1, 1) for r, p in zip(rows, positions)]
        pc = PatchCollection(rects, cmap=cmap, norm=norm, zorder=2)
        pc.set_array(sub[rows, positions])
        ax.add_collection(pc)
    if outline:
        lr, lp = np.nonzero(PHYSICAL_MASK[layers - 1])
        ax.add_collection(PatchCollection([Rectangle((p, layers[r] - 1), 1, 1) for r, p in zip(lr, lp)],
                                          facecolor="none", edgecolor="0.7", linewidth=0.4, zorder=3))
    layout_axes(ax)
    return pc


# figure with nrows x (odd, even) panels and colour-bar axes ('shared', 'rows' or None)
def view_figure(nrows: int = 1, figsize=(8.0, 6.0), colorbar: str | None = "shared", top: float = 0.91):
    fig = plt.figure(figsize=figsize)
    widths = [1.0, 1.0, 0.05] if colorbar else [1.0, 1.0]
    gs = fig.add_gridspec(nrows, len(widths), width_ratios=widths, left=0.08, right=0.9 if colorbar else 0.97,
                          bottom=0.09, top=top, wspace=0.18, hspace=0.28)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(nrows)])
    if colorbar == "shared":
        caxes = [fig.add_subplot(gs[:, 2])]
    elif colorbar == "rows":
        caxes = [fig.add_subplot(gs[r, 2]) for r in range(nrows)]
    else:
        caxes = []
    return fig, (axs[0] if nrows == 1 else axs), caxes


# write the cell values into a map
def _annotate(ax, grid: np.ndarray, view: str, norm, fmt: str) -> None:
    layers = VIEW_LAYERS[view]
    for layer in layers:
        for pos in np.flatnonzero(np.isfinite(grid[layer - 1])):
            v = grid[layer - 1, pos]
            dark = norm(v) < 0.6 if np.isfinite(norm(v)) else True
            ax.text(pos + 0.5, layer - 0.5, fmt.format(v), ha="center", va="center", fontsize=5,
                    color="w" if dark else "k", zorder=4)


# two-view page of a per-bar quantity, linear scale from 0
def map_page(pdf, grid: np.ndarray, title: str, label: str, cmap: str, fmt: str, annotate: bool) -> None:
    finite = grid[np.isfinite(grid)]
    top = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=top)
    fig, axs, caxes = view_figure(1, figsize=(11, 6.5), top=0.9)
    for ax, view in zip(axs, VIEWS):
        draw_view(ax, grid, view, norm, cmap)
        if annotate:
            _annotate(ax, grid, view, norm, fmt)
        ax.set_title(f"{view} layers")
    axs[0].set_ylabel("Layer")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=caxes[0], label=label)
    fig.suptitle(_ascii(title), fontsize=11)
    plot.save(fig, pdf)


# text page, long lines wrapped
def summary_page(pdf, title: str, lines: list[str], width: int = 100) -> None:
    out = []
    for line in lines:
        line = _ascii(line)
        indent = " " * (line.index(": ") + 2 if ": " in line[:44] else 4)
        out += textwrap.wrap(line, width, subsequent_indent=indent, break_on_hyphens=False) or [""]
    plot.text_page(pdf, _ascii(title), out)


# log y axis of a histogram (empty histogram: plain axis)
def _log_y(ax, counts) -> None:
    top = np.max(counts, initial=0)
    if top > 0:
        ax.set_yscale("log", nonpositive="clip")
        ax.set_ylim(0.5, top * 2.0)
        ax.yaxis.set_minor_locator(NullLocator())  # minor log ticks triple the rendering time
    else:  # empty histogram (e.g. a bar never counted): no meaningless -0.04..0.04 count axis
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])


# --------------------------------------------------------------------------
# dist_event and event_display pages
# --------------------------------------------------------------------------
# event-energy overview page: 300-bin plot, histogram, reconstructed bars per event
def event_overview_page(pdf, p: dict) -> None:
    d, unit, axis = p["overview"], p["unit"], p["axis"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.91, wspace=0.25, hspace=0.3)
    if d["empty"]:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "no selected event", ha="center", va="center", transform=ax.transAxes)
        plot.save(fig, pdf)
        return
    emax, edges, hist, min_count = d["emax"], d["edges"], d["hist"], d["min_count"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    shown = hist >= min_count
    ax = axs[0, 0]
    ax.scatter(centers[shown], hist[shown], s=10)
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(min_count, 1))
    ax.set_xlim(0, emax)
    ax.set_xlabel(axis)
    ax.set_ylabel("Events per bin")
    ax.set_title(f"Event distribution ({len(edges) - 1} bins, bins with >= {min_count} events)")
    if not shown.any():
        ax.text(0.5, 0.5, f"no bin with >= {min_count} events", ha="center", va="center", transform=ax.transAxes)
    ax = axs[0, 1]
    ax.stairs(hist, edges, color="C0")
    ax.set_xlabel(axis)
    ax.set_ylabel("Events per bin")
    ax.set_title(f"N = {d['n']}, mean = {d['mean']:.4g}, median = {d['median']:.4g} {unit}")
    ax = axs[1, 0]
    ax.stairs(d["nb_counts"], d["nb_edges"], color="C1")
    ax.set_xlabel("reconstructed bars per event")
    ax.set_ylabel("Events")
    ax.set_title(f"mean = {d['nb_mean']:.2f}, median = {d['nb_median']:g}")
    ax = axs[1, 1]
    h2, xe, ye = d["h2"], d["xe"], d["ye"]
    if h2.max() > 0:
        im = ax.pcolormesh(xe, ye, np.ma.masked_equal(h2.T, 0), norm=LogNorm(vmin=1, vmax=h2.max()), cmap="viridis")
        fig.colorbar(im, ax=ax, label="Events")
    ax.set_xlabel("reconstructed bars per event")
    ax.set_ylabel(axis)
    fig.suptitle(_ascii(f"{_title_head(p['label'], p['tag'])} | event overview"), fontsize=12)
    plot.save(fig, pdf)


# event-energy page with the ci spectrum fit (charge_injection.draw_spectrum)
def event_fit_page(pdf, p: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.84)
    ci.draw_spectrum(ax, p["values"], p["fit"], p["particle"], p["source"], p["ci"],
                     title=_ascii(f"{_title_head(p['label'], p['tag'])} | event energy"),
                     counts_text=f"{len(p['values']):,} selected events")
    plot.save(fig, pdf)


# write the dist_event pdf: summary, spectrum with the ci fit, overview
def write_dist_event(path: Path, p: dict) -> int:
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: event energy", p["lines"])
        if p["fit"] is not None:
            event_fit_page(pdf, p)
        event_overview_page(pdf, p)
    return 0


# event display: odd | even panels, one colour scale; returns the vmax
def event_display_page(pdf, p: dict, ev: dict) -> float:
    grid, scale, cmap = ev["grid"], p["scale"], p["cmap"]
    vmax = scale.limit(grid)
    norm = Normalize(vmin=0, vmax=vmax)
    fig, axs, caxes = view_figure(1, figsize=(8, 6))
    for ax, view in zip(axs, VIEWS):
        draw_view(ax, grid, view, norm, cmap)
    axs[0].set_ylabel("Layer")
    axs[0].set_title(f"{ev['name']} (odd layers)")
    axs[1].set_title("even layers")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=caxes[0], label=p["units"])
    note = "" if ev["in_input"] else "  (not a selected event)"
    energy_text = f"E = {ev['total']:.2f} {p['unit']} in {ev['n_bars']} bars" if ev["n_bars"] else "no counted bar"
    fig.suptitle(_ascii(f"{_title_head(p['label'], p['tag'])} | {ev['name']} | {energy_text}{note}"), fontsize=11)
    plot.save(fig, pdf)
    return vmax


# per-end event display: 2 x 2 panels, end by end
def event_per_end_page(pdf, p: dict, ev: dict) -> None:
    ends, scale, cmap = ev["ends"], p["scale"], p["cmap"]
    limits = [scale.limit(g) for g in ends]
    if scale.mode == "fixed":
        norms = [Normalize(vmin=0, vmax=limits[0])] * 2
    else:
        norms = [Normalize(vmin=0, vmax=v) for v in limits]
    fig, axs, caxes = view_figure(2, figsize=(9, 8), colorbar="shared" if scale.mode == "fixed" else "rows",
                                  top=0.92)
    for i, end in enumerate(geo.ENDS):
        for j, view in enumerate(VIEWS):
            ax = axs[i, j]
            draw_view(ax, ends[i], view, norms[i], cmap)
            ax.set_title(f"end{end} | {view}")
            if i == 0:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel("Layer")
    unit = p["unit"]
    if scale.mode == "fixed":
        fig.colorbar(ScalarMappable(norm=norms[0], cmap=cmap), cax=caxes[0], label=f"{unit} (per end)")
    else:
        for i, end in enumerate(geo.ENDS):
            fig.colorbar(ScalarMappable(norm=norms[i], cmap=cmap), cax=caxes[i], label=f"{unit} (end{end})")
    fig.suptitle(_ascii(f"{_title_head(p['label'], p['tag'])} | {ev['name']} | end by end (no merge)"), fontsize=11)
    plot.save(fig, pdf)


# end0 - end1 map: missing end -> 0, symmetric scale around 0
def event_difference_page(pdf, p: dict, ev: dict) -> None:
    ends, d = ev["ends"], p["difference"]
    diff = np.where(PHYSICAL_MASK, np.nan_to_num(ends[0], nan=0.0) - np.nan_to_num(ends[1], nan=0.0), np.nan)
    absdiff = np.abs(diff[PHYSICAL_MASK])
    top = float(d["difference_vmax_max_mip"]) if p["unit"] == "MIP" else np.inf
    vmax = max(float(d["difference_vmax_min"]), min(np.percentile(absdiff, float(d["difference_percentile"])), top))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axs, caxes = view_figure(1, figsize=(10, 6))
    for ax, view in zip(axs, VIEWS):
        draw_view(ax, diff, view, norm, "seismic")
        ax.set_title(f"{view} layers")
    axs[0].set_ylabel("Layer")
    fig.colorbar(ScalarMappable(norm=norm, cmap="seismic"), cax=caxes[0], label=f"(end0 - end1) {p['unit']}")
    fig.suptitle(_ascii(f"{_title_head(p['label'], p['tag'])} | {ev['name']} | end0 - end1 "
                        "(missing end -> 0; red: end0 > end1)"), fontsize=10)
    plot.save(fig, pdf)


# write the event_display pdf; returns the number of event displays
def write_event_display(path: Path, p: dict) -> int:
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: event displays", p["lines"])
        for ev in p["events"]:
            event_display_page(pdf, p, ev)
            if p["per_end"]:
                event_per_end_page(pdf, p, ev)
                event_difference_page(pdf, p, ev)
    return len(p["events"])


# --------------------------------------------------------------------------
# dist_bar, dist_channel and bar_maps pages
# --------------------------------------------------------------------------
# spectra pages: one page per layer, one panel per strip position; group = cell, or cell * 2 + end per channel
def spectra_pages(pdf, p: dict, end: int | None = None) -> None:
    edges, counts, overflow, n, total = p["edges"], p["counts"], p["overflow"], p["n"], p["total"]
    head = _title_head(p["label"], p["tag"])
    what = "bar energy spectra" if end is None else f"end{end} energy spectra (one channel per panel)"
    for layer in geo.LAYERS:
        fig, axs = plt.subplots(3, 4, figsize=(11, 8.5))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.91, wspace=0.25, hspace=0.35)
        for pos, ax in enumerate(axs.flat):
            ax.tick_params(labelsize=7)
            if not PHYSICAL_MASK[layer - 1, pos]:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"position {pos}: no bar", ha="center", va="center", color="0.6",
                        fontsize=8, transform=ax.transAxes)
                continue
            c = (layer - 1) * geo.N_POSITIONS + pos
            g = c if end is None else c * 2 + end
            strip = pos - (geo.NARROW_OFFSET if layer <= geo.LAST_NARROW_LAYER else 0)
            ax.stairs(counts[g], edges, color="C0" if end is None else ("C2" if end == 0 else "C3"))
            ax.set_xlim(edges[0], edges[-1])
            _log_y(ax, counts[g])
            text = f"N = {n[g]}"
            if n[g]:
                text += f"\nmean = {total[g] / n[g]:.3g}"
            if overflow[g]:
                text += f"\noverflow = {overflow[g]}"
            ax.text(0.97, 0.95, text, ha="right", va="top", fontsize=7, transform=ax.transAxes)
            ax.set_title(f"strip {strip} (position {pos})" + ("" if end is None else f" end{end}"), fontsize=8)
            if pos >= 8:
                ax.set_xlabel(p["units"], fontsize=8)
        fig.suptitle(_ascii(f"{head} | layer {layer} ({geo.n_strips(layer)} bars) | {what}"), fontsize=11)
        plot.save(fig, pdf)


# write the dist_bar pdf: summary, bar energy spectra per layer
def write_dist_bar(path: Path, p: dict) -> int:
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: bar energy distributions", p["lines"])
        spectra_pages(pdf, p)
    return 0


# write the dist_channel pdf: summary, end energy spectra per end and layer
def write_dist_channel(path: Path, p: dict) -> int:
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: channel energy distributions", p["lines"])
        for end in geo.ENDS:
            spectra_pages(pdf, p, end=end)
    return 0


# write the bar_maps pdf: occupancy and mean energy maps
def write_bar_maps(path: Path, p: dict) -> int:
    st, cmap, annotate, unit = p["maps"], p["cmap"], p["annotate"], p["unit"]
    head = _title_head(p["label"], p["tag"])
    per_events = f"over {p['n_events']} selected events"
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: bar maps", p["lines"])
        map_page(pdf, st["occupancy"], f"{head} | occupancy: fraction of events with the bar counted ({per_events})",
                 "fraction of events", cmap, "{:.2f}", annotate)
        map_page(pdf, st["mean_hit"], f"{head} | mean bar energy over the counted hits",
                 f"mean bar energy [{unit}]", cmap, "{:.1f}", annotate)
        map_page(pdf, st["mean_event"], f"{head} | mean bar energy per event ({per_events})",
                 f"energy per event [{unit}]", cmap, "{:.2f}", annotate)
    return 0


# --------------------------------------------------------------------------
# layer pages
# --------------------------------------------------------------------------
# 4 x 5 panels, one per layer
def _layer_grid_figure():
    fig, axs = plt.subplots(4, 5, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.91, wspace=0.28, hspace=0.42)
    return fig, axs


# write the layer pdf: longitudinal profile, layer energy and multiplicity per layer
def write_layer(path: Path, p: dict) -> int:
    layers, unit = geo.LAYERS, p["unit"]
    head = _title_head(p["label"], p["tag"])
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{p['label']}: layers", p["lines"])
        fig, axs = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.92, hspace=0.12)
        for view, marker, color in (("odd", "o", "C0"), ("even", "s", "C3")):
            sel = VIEW_LAYERS[view] - 1
            axs[0].errorbar(layers[sel], p["e_mean"][sel], yerr=p["e_sem"][sel], fmt=marker, color=color, capsize=3,
                            label=f"{view} layers")
            axs[1].errorbar(layers[sel], p["h_mean"][sel], yerr=p["h_sem"][sel], fmt=marker, color=color, capsize=3,
                            label=f"{view} layers")
        axs[0].plot(layers, p["e_mean"], color="0.6", lw=0.8, zorder=0)
        axs[1].plot(layers, p["h_mean"], color="0.6", lw=0.8, zorder=0)
        axs[0].set_ylabel(f"mean layer energy per event [{unit}]")
        axs[1].set_ylabel("mean counted bars per event")
        axs[1].set_xlabel("layer")
        axs[1].set_xticks(layers)
        for ax in axs:
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(_ascii(f"{head} | longitudinal profile ({p['n_events']} events, error = standard error)"),
                     fontsize=11)
        plot.save(fig, pdf)

        edges = p["e_edges"]
        fig, axs = _layer_grid_figure()
        for i, ax in enumerate(axs.flat):
            ax.tick_params(labelsize=7)
            if i >= geo.N_LAYERS:
                ax.set_axis_off()
                continue
            counts = p["e_counts"][i]
            ax.stairs(counts, edges, color="C0")
            ax.set_xlim(edges[0], edges[-1])
            _log_y(ax, counts)
            ax.set_title(f"layer {i + 1} ({'odd' if (i + 1) % 2 else 'even'})", fontsize=8)
            n_values = int(p["e_n"][i])
            ax.text(0.97, 0.95, f"N = {n_values}" + (f"\nmean = {p['e_avg'][i]:.3g}" if n_values else ""),
                    ha="right", va="top", fontsize=7, transform=ax.transAxes)
            if i >= 15:
                ax.set_xlabel(f"layer energy [{unit}]", fontsize=8)
        fig.suptitle(_ascii(f"{head} | layer energy per event (events with >= 1 counted bar in the layer)"),
                     fontsize=11)
        plot.save(fig, pdf)

        edges = p["m_edges"]
        fig, axs = _layer_grid_figure()
        for i, ax in enumerate(axs.flat):
            ax.tick_params(labelsize=7)
            if i >= geo.N_LAYERS:
                ax.set_axis_off()
                continue
            counts = p["m_counts"][i]
            ax.stairs(counts, edges, color="C1")
            ax.set_xlim(edges[0], edges[-1])
            _log_y(ax, counts)
            ax.set_title(f"layer {i + 1} ({geo.n_strips(i + 1)} bars)", fontsize=8)
            ax.text(0.97, 0.95, f"mean = {p['m_avg'][i]:.2f}" if p["n_events"] else "", ha="right", va="top",
                    fontsize=7, transform=ax.transAxes)
            if i >= 15:
                ax.set_xlabel("counted bars", fontsize=8)
        fig.suptitle(_ascii(f"{head} | hit multiplicity per layer ({p['n_events']} events)"), fontsize=11)
        plot.save(fig, pdf)
    return 0


# --------------------------------------------------------------------------
# hitmap pages
# --------------------------------------------------------------------------
# write the hit-map pdf: hit fractions, per-end fractions, first-event hit displays
def write_hitmap(path: Path, p: dict) -> int:
    grids, rule, n_events, label = p["grids"], p["rule"], p["n_events"], p["label"]
    cmap, annotate = p["cmap"], p["annotate"]
    with atomic_pdf(path) as pdf:
        summary_page(pdf, f"{label}: hit map", p["lines"])
        map_page(pdf, grids["both"], f"{label} | fraction of events with both ends {rule} ({n_events} events)",
                 "fraction of events", cmap, "{:.2f}", annotate)
        top = np.nanmax([np.nanmax(grids["end0"]), np.nanmax(grids["end1"]), 1e-12])
        norm = Normalize(vmin=0.0, vmax=top)
        fig, axs, caxes = view_figure(2, figsize=(10, 9.5), top=0.93)
        for i, end in enumerate(geo.ENDS):
            for j, view in enumerate(VIEWS):
                ax = axs[i, j]
                draw_view(ax, grids[f"end{end}"], view, norm, cmap)
                if annotate:
                    _annotate(ax, grids[f"end{end}"], view, norm, "{:.2f}")
                ax.set_title(f"end{end} | {view}")
                if i == 0:
                    ax.set_xlabel("")
                if j == 0:
                    ax.set_ylabel("Layer")
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=caxes[0], label="fraction of events")
        fig.suptitle(_ascii(f"{label} | per-end hit fraction ({rule}, {n_events} events)"), fontsize=11)
        plot.save(fig, pdf)
        binary = Normalize(vmin=0, vmax=1)
        for key, name in zip(p["first_events"], p["first_names"]):
            grid = np.full(N_CELLS, np.nan)
            grid[p["first_cell"][p["first_key"] == key]] = 1.0
            grid = grid.reshape(geo.N_LAYERS, geo.N_POSITIONS)
            fig, axs, _ = view_figure(1, figsize=(8, 6), colorbar=None)
            for ax, view in zip(axs, VIEWS):
                draw_view(ax, grid, view, binary, "gray_r")
            axs[0].set_ylabel("Layer")
            axs[0].set_title(f"{name} (odd layers)")
            axs[1].set_title("even layers")
            fig.suptitle(_ascii(f"{label} | {name} | {int(np.isfinite(grid).sum())} bars with both ends {rule}"),
                         fontsize=10)
            plot.save(fig, pdf)
    return 0


WRITERS = {"dist_event": write_dist_event, "dist_bar": write_dist_bar, "dist_channel": write_dist_channel,
           "event_display": write_event_display, "bar_maps": write_bar_maps, "layer": write_layer,
           "hitmap": write_hitmap}


# write one pdf from its payload (also in a worker process)
def write_part(unit: str, payload: dict, path: Path) -> int:
    return WRITERS[unit](Path(path), payload)


# worker initializer: install the parent's protected-folder list
def _init_render_worker(protected: list) -> None:
    io_utils.set_protected_dirs(protected)


# process pool of the pdf rendering
def render_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_render_worker,
                               initargs=(io_utils.protected_dirs(),))


# temp pdf that atomic_pdf writes for a path
def _worker_temp(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".tmp.pdf")


# write the pdfs of a run: serially, or in one process pool for the whole run
class Renderer:

    # set up (the pool is created on first use)
    def __init__(self, workers: int, max_jobs: int):
        self.workers = max(1, min(int(workers), int(max_jobs) or 1))
        self.pool: ProcessPoolExecutor | None = None
        self.pools_created = 0

    # enter the with block
    def __enter__(self):
        return self

    # shut the pool down (cancel on an error)
    def __exit__(self, exc_type, exc, tb):
        self.close(cancel=exc_type is not None)

    # shut the pool down
    def close(self, cancel: bool = False) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=cancel)
            self.pool = None

    # write the pdfs of one run; returns {unit: event displays or 0}
    def render(self, jobs: list[tuple[str, Path]], payloads: dict) -> dict:
        for _, path in jobs:
            ensure_writable(path)
        if self.workers < 2 or len(jobs) < 2:
            return {unit: write_part(unit, payloads[unit], path) for unit, path in jobs}
        if self.pool is None:
            self.pool = render_pool(self.workers)
            self.pools_created += 1
        registry = io_utils.active_registry()
        temps = {unit: _worker_temp(path) for unit, path in jobs}
        if registry is not None:
            for tmp in temps.values():
                registry.register(tmp)
        futures = {unit: self.pool.submit(write_part, unit, payloads[unit], path) for unit, path in jobs}
        results = {}
        for unit, future in futures.items():
            results[unit] = future.result()
            if registry is not None:  # written and renamed by the worker: no longer a temp file
                registry.release(temps[unit])
        return results


# --------------------------------------------------------------------------
# saved intermediates (cache/[test/]multiscale_plots/, -from-saved)
# --------------------------------------------------------------------------
EVENT_ARRAYS = ("event_keys", "event_energy", "event_n_bars", "cell_hits", "cell_total")
HOLD_ARRAYS = {"layers": ("layer_energy", "layer_hits"), "values": ("values", "value_cells"),
               "ends": ("end_values", "end_channels")}


# (npz, json) of a stored set: cache/[test/]multiscale_plots/<run key>_hitmap | <run key>_energy_<scale>[_<tag>]
def cache_paths(ctx, ds, kind: str, source: str = "s", tag: str = "") -> tuple[Path, Path]:
    stem = f"{ds.key}_hitmap" if kind == "hitmap" else f"{ds.key}_energy_{source}{tag_suffix(tag)}"
    folder = ctx.cache_dir(STAGE_DIR)
    return folder / f"{stem}.npz", folder / f"{stem}.json"


# file signature, or None
def _signature_or_none(path) -> dict | None:
    try:
        return file_signature(Path(path)) if path is not None else None
    except OSError:
        return None


# files a stored set depends on (nothing read): the producer records, or the csv files the hit map reads
def _planned_sources(ctx, ds, kind: str, s: dict, tag: str) -> list[Path]:
    csvs = ctx.run_csvs(ds)
    if kind == "hitmap":
        return [Path(ctx.input_plan_item(ds, csv)["path"]) for csv in csvs]
    return [producer_files(ctx, ds, csv, s["energy_source"], tag)[2] for csv in csvs]


# event selection of the ci cache values (quality cut); adcpeak: its own selected column
def selection_of(s_ci: dict | None) -> dict | None:
    if s_ci is None:
        return None
    return {k: s_ci[k] for k in ("quality_filter", "quality_threshold", "quality_tolerance")}


# what a stored set was made from (compared before reuse)
def _identity(ctx, ds, kind: str, s: dict, tag: str, sources, calib_path, selection: dict | None = None) -> dict:
    ident = {"cache": CACHE_VERSION, "kind": kind, "dataset": ds.key, "mode": ctx.mode,
             "test_events": int(ctx.n_test_events) if ctx.test else None,
             "inputs": [_signature_or_none(p) for p in sources]}
    if kind == "hitmap":
        ident.update(calibration=_signature_or_none(calib_path), threshold_column=s["hit_threshold_column"])
    else:
        ident.update(source=s["energy_source"], producer_tag=tag, selection=selection)
    return ident


# write one stored set (npz, json record last); a failure is only logged
def _commit_cache(npz: Path, meta: Path, arrays: dict, record: dict) -> None:
    try:
        with ProductSet(meta) as ps:
            np.savez(ps.stage(npz), **arrays)
            ps.commit(record)
    except OSError as exc:
        log(f"plots: WARNING could not store {npz.name} in the cache ({type(exc).__name__}: {exc}); "
            "-from-saved will not find it")


# store the reduced arrays of a run in cache/[test/]multiscale_plots/
def save_cache(ctx, ds, inputs: "StageInputs", data: "DatasetData") -> list[Path]:
    s, written = inputs.s, []
    sm = data.summary
    if sm is not None:
        npz, meta = cache_paths(ctx, ds, "energy", s["energy_source"], inputs.tag)
        sources = [files[2] for _, files in inputs.producers[ds.key]]
        ident = _identity(ctx, ds, "energy", s, inputs.tag, sources, None, inputs.selection)
        arrays = {k: getattr(sm, k) for k in EVENT_ARRAYS}
        arrays.update(n_counted=np.array(sm.n_counted, dtype=np.int64), csv_names=np.array(sm.csv_names, dtype=str))
        for hold, keys in HOLD_ARRAYS.items():
            if sm.holds.get(hold):
                arrays.update({k: getattr(sm, k) for k in keys})
        holds = {**sm.holds, "rows": data.rows is not None}
        if data.rows is not None:
            r = data.rows
            arrays.update(rows_key=r.key, rows_layer=r.layer, rows_strip=r.strip, rows_energy=r.energy,
                          rows_end0=r.end0, rows_end1=r.end1, row_events=np.asarray(data.row_events, np.int64))
        record = {**ident, "holds": holds, "n_events": int(sm.n_events),
                  "n_row_events": int(len(data.row_events)) if data.row_events is not None else 0,
                  "rows_picked_by": data.how,
                  "content": "reduced energies of one run (multiscale_plots EnergySummary) + display rows"}
        _commit_cache(npz, meta, arrays, record)
        written.append(npz)
    if data.hits is not None:
        npz, meta = cache_paths(ctx, ds, "hitmap")
        h = data.hits
        ident = _identity(ctx, ds, "hitmap", s, "", data.beam_csvs, inputs.calib_path)
        record = {**ident, "n_first": int(s["hitmap_events"]), "n_events": int(h.n_events),
                  "content": "raw hit counts of one run (multiscale_plots HitCounts)"}
        _commit_cache(npz, meta, {"both": h.both, "end0": h.end0, "end1": h.end1, "first_events": h.first_events,
                                  "first_key": h.first_key, "first_cell": h.first_cell,
                                  "n_events": np.array(h.n_events, dtype=np.int64),
                                  "csv_names": np.array(h.csv_names, dtype=str)}, record)
        written.append(npz)
    return written


# -from-saved refusal with the reason
def _refuse_saved(ds, what: str, reason: str) -> SystemExit:
    return SystemExit(f"plots -from-saved: {run_ref(ds)}: cannot draw {what} from the cache: {reason}.\n"
                      "  Run the same plots command without -from-saved (it reads the inputs and stores new "
                      "intermediates).")


# (record, open npz, npz path) of a stored set that matches this run, else stop
def _open_record(ctx, ds, kind: str, inputs: "StageInputs", sources):
    npz, meta = cache_paths(ctx, ds, kind, inputs.s["energy_source"], inputs.tag)
    expected = _identity(ctx, ds, kind, inputs.s, inputs.tag if kind == "energy" else "", sources, inputs.calib_path,
                         inputs.selection)
    record = require_product_set(meta, expected, rebuild="the same plots command without -from-saved",
                                 what=f"stored {kind} arrays of {run_ref(ds)} ({meta.name})", members=[npz])
    return record, np.load(npz, allow_pickle=False), npz


# -from-saved: the reduced data of a run from the cache, checked against this run
def load_saved(ctx, ds, inputs: "StageInputs", check_only: bool = False) -> "DatasetData | None":
    """check_only: every check without loading the large arrays."""
    s = inputs.s
    data = DatasetData()
    if s["energy_units"]:
        sources = _planned_sources(ctx, ds, "energy", s, inputs.tag)
        record, z, npz_path = _open_record(ctx, ds, "energy", inputs, sources)
        with z:
            holds = record.get("holds") or {}
            for unit in s["energy_units"]:
                if not all(holds.get(n) for n in UNIT_NEEDS[unit]):
                    raise _refuse_saved(ds, unit, f"the run that stored {npz_path.name} did not draw {unit}")
            meta = _summary_meta(ctx, ds, inputs)
            names = [str(x) for x in z["csv_names"]]
            full = {k: z[k] for k in EVENT_ARRAYS[:3]}
            if not check_only:
                full.update({k: z[k] for k in EVENT_ARRAYS[3:]})
                for hold, keys in HOLD_ARRAYS.items():
                    full.update({k: (z[k] if holds.get(hold) else None) for k in keys})
            summary = EnergySummary(csv_names=names, n_counted=int(z["n_counted"]), holds=holds,
                                    **{k: full.get(k) for k in (*EVENT_ARRAYS, "layer_energy", "layer_hits", "values",
                                                                "value_cells", "end_values", "end_channels")},
                                    **meta)
            data.summary = summary
            if "event_display" in s["units"]:
                picked, how = pick_events(summary.event_keys, summary.event_energy, summary.unit, s)
                row_events = z["row_events"]
                needed = picked[in_sorted(picked, summary.event_keys)]
                missing = np.unique(needed[~in_sorted(needed, np.unique(row_events))])
                if missing.size:
                    raise _refuse_saved(ds, "the event displays",
                                        f"they need the rows of {missing.size} event(s) the cache does not hold "
                                        f"(e.g. {', '.join(event_name(m, names) for m in missing[:5])}); it holds "
                                        f"the rows of {len(row_events)} events ({record.get('rows_picked_by', '?')})")
                data.picked, data.how, data.row_events = picked, how, row_events
                if not check_only:
                    data.rows = Counted(z["rows_key"], z["rows_layer"], z["rows_strip"], z["rows_energy"],
                                        z["rows_end0"], z["rows_end1"])
        data.from_saved = f"{_short_path(npz_path)} (stored {record.get('committed', '?')})"
    if "hitmap" in s["units"]:
        sources = _planned_sources(ctx, ds, "hitmap", s, "")
        record, z, npz_path = _open_record(ctx, ds, "hitmap", inputs, sources)
        with z:
            k, first = int(s["hitmap_events"]), z["first_events"]
            truncated = len(first) == int(record.get("n_first", 0)) and len(first) < int(record["n_events"])
            if k > len(first) and truncated:
                raise _refuse_saved(ds, "the hit map",
                                    f"{k} hit displays requested, the cache holds the first {len(first)} events")
            if not check_only:
                first = first[:k]
                sel = np.isin(z["first_key"], first)
                data.hits = HitCounts(int(record["n_events"]), z["both"], z["end0"], z["end1"], first,
                                      z["first_key"][sel], z["first_cell"][sel], s["hit_threshold_column"],
                                      [str(x) for x in z["csv_names"]])
        data.beam_csvs = sources
        data.hit_note = f"{_short_path(npz_path)} (stored {record.get('committed', '?')})"
    return None if check_only else data


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
# inputs shared by every run of one plots call
@dataclass
class StageInputs:
    s: dict
    chunksize: int
    tag: str = ""                     # producer variant tag read (energy units)
    label: str = ""                   # scale + variant: head of every title
    s_ci: dict | None = None          # ci settings (ci scales): event selection and spectrum fit
    selection: dict | None = None     # the ci quality cut applied (identity of the stored energies)
    calib_path: Path | None = None    # calibration table of the hit map
    tables: BeamTables | None = None
    producers: dict = field(default_factory=dict)  # {run key: [(csv, producer files), ...]} after the check


# reduced data of one run (from the producer cache, the beam csv or the plots cache)
@dataclass
class DatasetData:
    summary: EnergySummary | None = None
    rows: Counted | None = None           # counted-bar rows of the displayed events (event_display)
    row_events: np.ndarray | None = None  # events whose rows ``rows`` holds completely
    picked: np.ndarray | None = None
    how: str = ""
    hits: HitCounts | None = None
    beam_csvs: list = field(default_factory=list)
    from_saved: str = ""                  # energy pages: the cache file they were re-drawn from
    hit_note: str = ""                    # hit map: the cache file it was re-drawn from


# title label of a scale: '<scale>: <what it is>[, variant <tag>]'
def source_label(source: str, tag: str = "") -> str:
    what = "adcpeak Arbitrary Energy" if source == ARBITRARY else ci.path_label(source)
    return f"{source}: {what}" + (f", variant {tag}" if tag else "")


# units, labels, definition and producer folder of the energy summary
def _summary_meta(ctx, ds, inputs: StageInputs) -> dict:
    source = inputs.s["energy_source"]
    if source == ARBITRARY:
        return dict(units=ta.ARBITRARY_ENERGY, unit="a.u.", axis=ta.ARBITRARY_ENERGY, source=source,
                    label=inputs.label, input_dir=ctx.csv_product(ta.cache_dir(ctx), ds, "x.csv", "").parent,
                    definition="adcpeak Arbitrary Energy: bar value = its end ADC peaks above the pedestal, merged; "
                               "event value = sum of its bar values; events with selected = true")
    return dict(units=plot.MIP_UNITS, unit="MIP", axis="Event energy [MIP]", source=source, label=inputs.label,
                input_dir=ctx.csv_product(ci.event_cache_dir(ctx, inputs.tag), ds, "x.csv", "").parent,
                definition=f"ci path {source}: bar energy from its end energies, event energy = sum of its bar "
                           f"energies; selected events: {ci.selection_text(inputs.s_ci)}")


# read and reduce the producer cache and the beam csv of one run
def reduce_run(ctx, ds, inputs: StageInputs) -> DatasetData:
    s = inputs.s
    source = s["energy_source"]
    data = DatasetData()
    if s["energy_units"]:
        items = inputs.producers[ds.key]
        keys, energy, n_bars = [], [], []
        for i, (_, files) in enumerate(items):
            pf, e, nb = csv_events(source, files, inputs.s_ci)
            keys.append(event_key(i, pf))
            energy.append(e)
            n_bars.append(nb)
        keys = np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)
        order = np.argsort(keys, kind="stable")
        events = (keys[order], (np.concatenate(energy) if energy else np.empty(0))[order],
                  (np.concatenate(n_bars) if n_bars else np.empty(0, dtype=np.int64))[order])
        needs = energy_needs(s)
        rows = None
        if "event_display" in s["units"]:
            data.picked, data.how = pick_events(events[0], events[1], "a.u." if source == ARBITRARY else "MIP", s)
            rows = EventRows(data.picked)
        acc = EnergyAccumulator(events[0], needs, rows)
        if "bars" in needs:
            for i, (_, files) in enumerate(items):
                for counted in csv_bars(source, files, i, inputs.chunksize):
                    acc.add(counted)
        data.summary = acc.finish(events, "bars" in needs, csv_names=[csv.name for csv, _ in items],
                                  **_summary_meta(ctx, ds, inputs))
        if rows is not None:
            data.rows, data.row_events = rows.table(), rows.keys
    if "hitmap" in s["units"]:
        hit_acc = HitAccumulator(int(s["hitmap_events"]), inputs.tables.threshold_column)
        names = []
        for i, (csv, path) in enumerate(ctx.run_inputs(ds)):
            path = require(path, produced_by=f".\\runner.ps1 make-csv {run_flags(ds)}")
            names.append(csv.name)
            data.beam_csvs.append(path)
            for events, hit_rows in iter_beam_csv(path, inputs.tables, inputs.chunksize, i):
                hit_acc.add(events, hit_rows)
        data.hits = hit_acc.finish(names)
    return data


# figure data per unit of one run; counts for the log
def build_payloads(ctx, ds, data: DatasetData, inputs: StageInputs) -> tuple[dict, dict]:
    s = inputs.s
    info = {"from_saved": data.from_saved}
    sm = data.summary
    builders = {"dist_event": lambda: dist_event_payload(ctx, ds, sm, s, info, inputs.s_ci),
                "dist_bar": lambda: dist_bar_payload(ctx, ds, sm, s, info),
                "dist_channel": lambda: dist_channel_payload(ctx, ds, sm, s, info),
                "event_display": lambda: event_display_payload(ctx, ds, sm, s, data.picked, data.how, data.rows,
                                                               info),
                "bar_maps": lambda: bar_maps_payload(ctx, ds, sm, s, info),
                "layer": lambda: layer_payload(ctx, ds, sm, s, info),
                "hitmap": lambda: hitmap_payload(ctx, ds, data.hits, data.beam_csvs, inputs.calib_path, s,
                                                 data.hit_note)}
    payloads = {unit: builders[unit]() for unit in s["units"]}
    stats = {}
    if sm is not None:
        stats.update(events=sm.n_events, bars=sm.n_counted if sm.holds.get("bars") else None)
    if data.hits is not None:
        stats.setdefault("events", data.hits.n_events)
    return payloads, stats


# stage entry: settings, producer check, output plan, reduce or load, render the pdfs
def run(ctx, args, datasets) -> None:
    if not ctx.make_plots:
        log("plots: -no-plots given, nothing to draw (every unit is a pdf)")
        _output_plan(ctx, [])  # the plan of a run without results: -output names are ignored
        return
    s = resolve_settings(ctx, args)
    _output_plan(ctx, [], preview=True)  # an -output name given twice stops the run here, before any input is read
    from_saved = bool(getattr(args, "from_saved", False))
    source = s["energy_source"]
    workers = render_workers(args, s)
    inputs = StageInputs(s=s, chunksize=int(ctx.cfg.get("run", {}).get("chunksize", 2_000_000)))
    if s["energy_units"]:
        inputs.tag = producer_tag(ctx, s)
        inputs.label = source_label(source, inputs.tag)
        if source != ARBITRARY:
            inputs.s_ci = ci_settings(ctx)
            inputs.selection = selection_of(inputs.s_ci)
    if "hitmap" in s["units"]:
        inputs.calib_path = require(ctx.calibration_file(), produced_by=_calib_command(ctx))
        if not from_saved:  # -from-saved compares only the calibration file's signature
            inputs.tables = beam_tables(load_calibration(inputs.calib_path), s)
    log(f"plots: units {', '.join(s['units'])}" + (f"; energy scale {inputs.label}" if s["energy_units"] else "")
        + ("; from the saved intermediates" if from_saved else "")
        + f"; {workers} rendering process{'es' if workers > 1 else ''}")
    # the output plan of the final results (the pdfs), before any input is read (common/outputs.py)
    dest, _ = pdf_destinations(ctx, result_slots(ctx, datasets, s, inputs.tag))
    pdf_jobs = {ds.key: [(unit, dest[result_key(ctx, ds, unit, s, inputs.tag)][0]) for unit in s["units"]]
                for ds in datasets}
    saved_of = {path: saved for path, saved in dest.values()}
    for jobs in pdf_jobs.values():  # refuse a protected output folder before reading any input
        for _, path in jobs:
            ensure_writable(path)
    store = s["save_cache"] and not from_saved
    if store:
        for ds in datasets:
            for kind in _cache_kinds(s):
                for path in cache_paths(ctx, ds, kind, source, inputs.tag):
                    ensure_writable(path)
    if from_saved:  # every run is checked before anything is drawn
        for ds in datasets:
            load_saved(ctx, ds, inputs, check_only=True)
    elif s["energy_units"]:
        inputs.producers = check_producers(ctx, datasets, source, inputs.tag)
    written, summary = [], []
    with Renderer(workers, len(s["units"])) as renderer:
        for ds in datasets:
            if from_saved:
                data = load_saved(ctx, ds, inputs)
            else:
                with timed(f"plots: reading {ds.key}"):
                    data = reduce_run(ctx, ds, inputs)
                if store:  # before drawing: a failed rendering can be redone with -from-saved
                    save_cache(ctx, ds, inputs, data)
            with timed(f"plots {ds.key}"):
                payloads, stats = build_payloads(ctx, ds, data, inputs)
                del data
                jobs = pdf_jobs[ds.key]
                results = renderer.render(jobs, payloads)  # with -workers N: one pdf per process
                del payloads
            written += [path for _, path in jobs]
            if s["energy_units"]:
                bars = "bars not read" if stats["bars"] is None else f"{stats['bars']} counted bars"
                summary.append(f"{ds.key}: {stats['events']} selected events, {bars}, "
                               f"{results.get('event_display', 0)} event displays")
            else:
                summary.append(f"{ds.key}: {stats['events']} events (hit map only)")
    for line in summary:
        log(f"plots: {line}")
    for path in written:
        if saved_of[path]:
            log(f"plots: wrote {path}")
    unsaved = [path for path in written if not saved_of[path]]
    if unsaved:
        log(f"plots: final results not saved (no --output): scratch copies in {unsaved[0].parent}; use --output "
            "[names] to save them")
    if store:
        log(f"plots: reduced arrays stored in {ctx.cache_dir(STAGE_DIR)} (re-draw with -from-saved)")
