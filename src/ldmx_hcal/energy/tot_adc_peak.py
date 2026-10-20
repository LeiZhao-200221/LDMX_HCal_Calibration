"""Stage adcpeak: TOT -> ADC-peak line per channel-end, Arbitrary Energy per event, linearity and resolution.

tot_per_adc_slope, tot_offset: the line TOT = slope * adc_max + offset of a channel-end, fitted below the ADC rail on
the whole fit_adc_window; it turns the TOT of a saturated end into its ADC peak, (TOT - offset) / slope.
Arbitrary Energy [a.u.]: the sum of the bar values of an event (bar value = its end ADC peaks above the pedestal,
merged). It is no energy: only the resolution sigma/mu (unit-free) and the relative linearity mean something.
Output units (-only): line_table, line_fits, energy_response, energy_spectra, linearity_resolution.

Variables of tot_adc_peak_parameters.csv (line_table; one row per channel-end, column order):
  layer, strip, end, tot_per_adc_slope, tot_offset, status, n_hits_window, n_hits_fit, n_bins_filled, n_bins_removed,
  n_bins_fit, fit_dataset
Variables of energy_response.csv (record spectrum: one row per run; linearity: one per fitted particle; resolution:
one per model):
  record, tot_mode, units, dataset, particle, energy_gev, spectrum_fit, n_events, mu, mu_error, sigma, sigma_error,
  sigma_over_mu, fit_low, fit_high, n_fit_bins, chi2, ndf, status, [fit_ranges, bin_rule, sigma_over_mu_error: with
  -fit; mu_low, mu_low_error, sigma_low, sigma_low_error, area_low, mu_high, mu_high_error, sigma_high,
  sigma_high_error, area_high: with -fit[a,b],[c,d]], point_mu_error, point_ratio_error, point_error_n,
  in_resolution_fit, resolution_predicted, in_linearity_fit, linearity_predicted, linearity_residual_fraction,
  fit_range_gev, ok, weighted, slope, slope_error, intercept, intercept_error, chi2_over_ndf, n_fitted,
  fitted_energies, model, a, a_error, b, b_error, c, c_error, error_prescription, [error_scale, slope_error_scaled,
  intercept_error_scaled, a_error_scaled, b_error_scaled, c_error_scaled: with -normalerror]
Variables of the cache cache/[test/]tot_adc_peak/<run>/ (one file per input csv; -from-saved and plots read them):
  <csv stem>_events.csv (one row per event): event_source, pf_event, arbitrary_energy, n_bars, n_ends_hit,
  n_ends_converted, n_ends_saturated_unconverted, n_unrecoverable_bars, unrecoverable_bar_fraction,
  n_saturated_unconverted_bars, selected
  <csv stem>_bars.csv (-bar-csv; one row per counted bar of a selected event): event_source, pf_event, layer, strip,
  bar_arbitrary_energy, end0_arbitrary_energy, end1_arbitrary_energy
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import energy_response as er
from ..common import geometry as geo
from ..common import outputs
from ..common.calibration_io import calibration_lookup, load_calibration
from ..common.config import DEFAULT_CONFIG, resolve_runs, run_flags, run_ref
from ..common.fitting import format_ranges
from ..common.io_utils import (InputError, ProductSet, ProductSetError, SettingsError, atomic_pdf, content_hash,
                               iter_csv, log, product_set_valid, read_json, require, settings_hash, timed,
                               validate_channel_keys, validate_settings)

HELP = "TOT -> ADC-peak line fit + Arbitrary Energy per event, linearity, resolution (arbitrary units)"
STAGE_DIR = "tot_adc_peak"     # calibration|results|cache/[test/]tot_adc_peak
RUNNER = ".\\runner.ps1"        # command shown in "produced by" hints
DEFAULT_DATASETS = ["electron 0.3", "electron 0.4", "electron 0.5", "electron 1", "electron 4"]  # run references
CONFIG_SECTION = "tot_adc_peak"
SUPPORTS_FROM_SAVED = True     # runner -from-saved: line fit bins and events from cache/[test/]tot_adc_peak/

TOT_MODES = ("linear", "none", "mean_fallback")
SPECTRUM_FITS = ("two_pass", "peak_scan")
ARBITRARY_ENERGY = "Arbitrary Energy [a.u.]"   # axis label of the per-event sum (no energy scale)

# output units (-only): one file each; the line table is a calibration constant, the others final results
UNITS = ("line_table", "line_fits", "energy_response", "energy_spectra", "linearity_resolution")
UNIT_GROUPS = {"line": ("line_table", "line_fits"),
               "response": ("energy_response", "energy_spectra", "linearity_resolution"),
               "csv": ("line_table", "energy_response"),
               "plots": ("line_fits", "energy_spectra", "linearity_resolution")}
PLOT_UNITS = ("line_fits", "energy_spectra", "linearity_resolution")
RESPONSE_UNITS = ("energy_response", "energy_spectra", "linearity_resolution")

# identity of the saved intermediates (cache/[test/]tot_adc_peak/): a change of the columns (schema) or of the
# reconstruction / bin accumulation (algorithm) makes -from-saved refuse every older copy
EVENT_SCHEMA = "adcpeak-events-1"                 # the columns of EVENT_COLUMNS, in this order
EVENT_ALGORITHM = "adcpeak-events-1"
FIT_HIST_SCHEMA = "adcpeak-fit-histograms-2"
FIT_HIST_ALGORITHM = "adcpeak-fit-histograms-2"   # 2: x = bin mean adc_max
TOT_TABLE_UNUSED = "unused (tot_mode none)"      # dependency token of events made without TOT lines

DEFAULTS = {
    # line fit TOT = slope * adc_max + offset
    "fit_dataset": "pion 3",                   # run of the line fit (every csv of its folder); any run
    "fit_adc_window": [949, 1021],             # adc_max window of the fit, inclusive
    "fit_tot_window": [500, 2000],             # TOT window of the fit, exclusive (2024 version: [1, 500])
    "fit_bin_width": 3,                        # ADC bin width; the bins cover the whole fit_adc_window
    "fit_outlier_nsigma": 3.0,                 # bin means farther than this many sigma (ddof 0) are removed
    "fit_min_bins": 2,                         # bins needed after the outlier cut
    "plot_adc_window": [900, 1050],            # fit pages: x range; scatter of fired hits with adc_max >= low
    "plot_max_points": 5000,                   # scatter points kept per channel-end while reading (<= 0: all)
    "plot_sample_seed": 12345,                 # seed of that per-channel random sample (bottom-k of hashed keys)
    # event reconstruction
    "tot_mode": "linear",                      # linear | none | mean_fallback
    "slope_window": [0.6, 4.0],                # usable line: low <= slope < high [TOT counts per ADC count]
    "require_fit_ok": True,                    # the line must have status ok
    "saturation_adc": 1022,                    # adc_max >= this: saturated / near the rail (spec 06)
    "merge_rail_adc": None,                    # end merge rail test adc_max >= this (null: saturation_adc)
    "hit_threshold_column": "pedestal_range_with_20percent",  # end hit: adc_sum >= this calibration column
    "quality_threshold": None,                 # keep events with unrecoverable_bar_fraction <= this (0..1)
    # response
    "spectrum_fit": "two_pass",                # two_pass | peak_scan
    # output
    "write_bar_csv": False,                    # also write the bar values into the cache (input of plots)
    "output_tag": None,                        # extra suffix of the cache and response files (checked:
                                               # [A-Za-z0-9][A-Za-z0-9._-]*, not none / mean_fallback[_...])
}

READ_COLUMNS_FIT = ["layer", "strip", "adc_max_end0", "tot_end0", "adc_max_end1", "tot_end1"]
READ_COLUMNS_EVENTS = ["pf_event", "layer", "strip", "adc_sum_end0", "adc_max_end0", "tot_end0", "adc_sum_end1",
                       "adc_max_end1", "tot_end1"]

N_BAR_SLOTS = (geo.N_LAYERS + 1) * geo.N_STRIPS_WIDE          # bar slot = layer * 12 + strip
N_CHANNEL_SLOTS = N_BAR_SLOTS * 2                               # channel slot = bar slot * 2 + end

PARAMETER_COLUMNS = {
    "layer": "layer 1..19",
    "strip": "bar index in the layer (0-based)",
    "end": "0 / 1",
    "tot_per_adc_slope": "slope of TOT = slope * adc_max + offset [TOT counts per ADC count] (0 = not fittable)",
    "tot_offset": "offset of the same line [TOT counts] (0 = not fittable)",
    "status": "ok | no_data | no_hits_in_window | no_filled_bins | too_few_bins_after_outlier_cut",
    "n_hits_window": "hits passing the ADC and TOT windows",
    "n_hits_fit": "hits in the bins used by the fit",
    "n_bins_filled": "non-empty ADC bins",
    "n_bins_removed": "bins removed by the outlier cut",
    "n_bins_fit": "bins used by the fit",
    "fit_dataset": "run key of the fit run (e.g. pion_3GeV)",
}
EVENT_COLUMNS = {
    "event_source": "'<run>/<csv name>' of the event (with pf_event the event identity)",
    "pf_event": "event id in its csv",
    "arbitrary_energy": "Arbitrary Energy = sum of the counted bar values [a.u.]",
    "n_bars": "reconstructed bars",
    "n_ends_hit": "ends above the hit threshold",
    "n_ends_converted": "hit ends whose value comes from TOT",
    "n_ends_saturated_unconverted": "hit ends at/above saturation_adc without a TOT value",
    "n_unrecoverable_bars": "bars with a railed end without usable TOT and no reliable other end",
    "unrecoverable_bar_fraction": "n_unrecoverable_bars / max(n_bars, 1)",
    "n_saturated_unconverted_bars": "bars with an end at the rail and no converted end",
    "selected": "passes --quality-threshold (always true when the cut is off)",
}
BAR_COLUMNS = ["event_source", "pf_event", "layer", "strip", "bar_arbitrary_energy", "end0_arbitrary_energy",
               "end1_arbitrary_energy"]


# --------------------------------------------------------------------------
# output paths (single source of truth; downstream stages import these)
#   calibration/[test/]tot_adc_peak/  per-channel line table (a calibration constant)
#   results/<mode>/tot_adc_peak/      fit pdf, response table and pdfs (<mode> = fullset | test)
#   cache/[test/]tot_adc_peak/        fit bins, per-csv events / bar values for -from-saved and plots
# --------------------------------------------------------------------------
# folder of the line table: calibration/[test/]tot_adc_peak
def calibration_dir(ctx, mode: str | None = None) -> Path:
    return ctx.calib_dir(STAGE_DIR, mode=mode)


# folder of the final results: results/<mode>/tot_adc_peak
def results_dir(ctx, mode: str | None = None) -> Path:
    return ctx.results_dir(STAGE_DIR, mode=mode)


# folder of the saved intermediates: cache/[test/]tot_adc_peak
def cache_dir(ctx, mode: str | None = None) -> Path:
    return ctx.cache_dir(STAGE_DIR, mode=mode)


# file-name suffix of a variant tag: '' -> '', 'none' or '_none' -> '_none'
def tag_suffix(tag) -> str:
    tag = str(tag or "").strip().lstrip("_")
    return f"_{tag}" if tag else ""


# --output-tag: file-name safe (letters, digits, '.', '_', '-'; first character a letter or digit), so a tag can
# never leave the stage folders (no path separators, no leading '.').
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
# automatic variant-tag parts (see variant_tag) -> the flag that sets them.  An --output-tag equal to one of them,
# or starting with one of them + "_", is refused (case-insensitive: Windows file names), so every file name maps
# back to exactly one combination of variant flags.
RESERVED_TAG_PARTS = {"none": "--tot-mode none", "mean_fallback": "--tot-mode mean_fallback"}


# the automatic variant tag an -output-tag collides with, else None
def reserved_tag_clash(tag) -> str | None:
    low = str(tag or "").strip().lstrip("_").lower()
    return next((r for r in RESERVED_TAG_PARTS if low == r or low.startswith(r + "_")), None)


# checked -output-tag: file-name safe, not an automatic variant tag; None if empty
def normalize_output_tag(value) -> str | None:
    if value is None:
        return None
    tag = str(value).strip().lstrip("_")
    if not tag:
        return None
    if not _TAG_PATTERN.fullmatch(tag):
        raise SystemExit(f"adcpeak: --output-tag {value!r} may only contain letters, digits, '.', '_' and '-' and "
                         "must start with a letter or digit (it becomes part of the file names)")
    clash = reserved_tag_clash(tag)
    if clash:
        raise SystemExit(f"adcpeak: --output-tag {value!r} is reserved: '{clash}' is the automatic variant tag of "
                         f"{RESERVED_TAG_PARTS[clash]}; use that flag, or choose another tag")
    if tag != str(value):
        log(f"adcpeak: --output-tag {value!r} used as {tag!r}")
    return tag


# variant tag: the TOT mode if not linear and the -output-tag, joined by '_'
def variant_tag(s: dict) -> str:
    parts = [] if s["tot_mode"] == "linear" else [s["tot_mode"]]
    tag = str(s.get("output_tag") or "").strip().lstrip("_")
    if tag:
        parts.append(tag)
    return "_".join(parts)


# file-name suffix of the run's variant: '' for the default run, else '_<tag>'
def output_suffix(s: dict) -> str:
    return tag_suffix(variant_tag(s))


# path of the line table: calibration/[test/]tot_adc_peak/tot_adc_peak_parameters.csv
def parameters_csv(ctx, mode: str | None = None) -> Path:
    return calibration_dir(ctx, mode) / "tot_adc_peak_parameters.csv"


# default path of the fit pdf: results/<mode>/tot_adc_peak/tot_adc_peak_fits.pdf
def fits_pdf(ctx) -> Path:
    return results_dir(ctx) / "tot_adc_peak_fits.pdf"


# default path of the response table: results/<mode>/tot_adc_peak/energy_response<_tag>.csv
def energy_response_csv(ctx, tag: str = "") -> Path:
    return results_dir(ctx) / f"energy_response{tag_suffix(tag)}.csv"


# default path of the spectra pdf: results/<mode>/tot_adc_peak/energy_spectra<_tag>.pdf
def energy_spectra_pdf(ctx, tag: str = "") -> Path:
    return results_dir(ctx) / f"energy_spectra{tag_suffix(tag)}.pdf"


# default path of the linearity / resolution pdf: linearity_resolution<_tag>.pdf
def linearity_resolution_pdf(ctx, tag: str = "") -> Path:
    return results_dir(ctx) / f"linearity_resolution{tag_suffix(tag)}.pdf"


# saved events of one input csv: cache/[test/]tot_adc_peak/<run>/<csv stem>_events<_tag>.csv
def saved_events_csv(ctx, ds, csv, tag: str = "", mode: str | None = None) -> Path:
    return ctx.csv_product(cache_dir(ctx, mode), ds, csv, f"_events{tag_suffix(tag)}.csv")


# bar values of one input csv (-bar-csv): cache/[test/]tot_adc_peak/<run>/<csv stem>_bars<_tag>.csv
def bar_values_csv(ctx, ds, csv, tag: str = "", mode: str | None = None) -> Path:
    return ctx.csv_product(cache_dir(ctx, mode), ds, csv, f"_bars{tag_suffix(tag)}.csv")


# saved fit bins + display sample of the fit run: cache/[test/]tot_adc_peak/fit_histograms_<run key>.npz
def saved_fit_histograms(ctx, fit_dataset: str, mode: str | None = None) -> Path:
    return cache_dir(ctx, mode) / f"fit_histograms_{fit_dataset}.npz"


# provenance path next to a product: x.csv -> x.provenance.json
def provenance_path(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + ".provenance.json")


# content identity of an input file: size + sha256 (path and mtime not part of it)
def content_identity(path) -> dict:
    path = Path(path)
    digest, method = content_hash(path)
    return {"size": int(path.stat().st_size), "sha256": digest, "hash": method}


# content identity, or {'missing': path}
def _identity_or_missing(path) -> dict:
    try:
        return content_identity(path)
    except OSError:
        return {"missing": str(path)}


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# finite number (not bool)?
def _number(v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            and math.isfinite(float(v)))


# settings rule of fit_min_bins: an integer >= 2
def _rule_min_bins(v) -> str | None:
    ok = isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 2
    return None if ok else "an integer >= 2 (a line needs two bins)"


# numeric / choice settings of the stage, checked before any IO (io_utils.validate_settings rules)
SETTINGS_RULES = {
    "fit_dataset": "str", "fit_adc_window": "window", "fit_tot_window": "window", "fit_bin_width": "float>0",
    "fit_outlier_nsigma": "float>0", "fit_min_bins": _rule_min_bins, "plot_adc_window": "window",
    "plot_max_points": "int", "plot_sample_seed": "int", "tot_mode": ("choice", list(TOT_MODES)),
    "slope_window": "window", "require_fit_ok": "bool", "saturation_adc": "float>0", "merge_rail_adc": "float>0",
    "hit_threshold_column": "str", "quality_threshold": "fraction?", "spectrum_fit": ("choice", list(SPECTRUM_FITS)),
    "write_bar_csv": "bool",
}
SETTINGS_SOURCE = "adcpeak settings (configure/config.yml section tot_adc_peak, command line)"
# the shared range flags of the response (energy_response.RANGE_SETTING_KEYS): settings key -> flag
RESPONSE_RANGE_FLAGS = {"fit_ranges": "--fit", "fit_bins": "--bin", "linearity_fits": "--linfit",
                        "resolution_fit_range": "--resfit"}
# the spectrum range keys that only the command line sets (energy_response.CLI_ONLY_SETTINGS): the yml section
# tot_adc_peak refuses them with this hint
CLI_ONLY_HINTS = {
    "fit_ranges": "-fit[a,b] (single Gaussian) or -fit[a,b],[c,d] (double Gaussian), in a.u., on the adcpeak command "
                  "line, e.g. '.\\runner.ps1 adcpeak --electron 1 -from-saved -only response -fit[3500,6500] "
                  "-output-tag electron_1_fit'",
    "fit_bins": "-bin N (or -bin N1,N2, one value per range) together with -fit on the adcpeak command line",
}


# error naming fit_ranges / fit_bins set in the yml (command line only), else None
def cli_only_settings_error(section: dict, source: str) -> SettingsError | None:
    keys = [k for k in er.CLI_ONLY_SETTINGS if k in (section or {})]
    if not keys:
        return None
    lines = [f"{k}: command-line only, remove it from the yml and give "
             f"{CLI_ONLY_HINTS.get(k, er.CLI_ONLY_SETTINGS[k])}" for k in keys]
    return SettingsError(f"{source}:\n  " + "\n  ".join(lines))


# adcpeak settings: DEFAULTS < yml section tot_adc_peak < command line, checked before any IO
def stage_settings(ctx, args=None) -> dict:
    """s['_cli']: keys set on the command line; s['_from_saved']: -from-saved."""
    section = ctx.stage_cfg(CONFIG_SECTION) or {}
    problem = cli_only_settings_error(section, SETTINGS_SOURCE)
    if problem:
        raise problem
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise SettingsError(f"{SETTINGS_SOURCE}: unknown setting(s) {unknown}; the valid keys are {sorted(DEFAULTS)}")
    s = copy.deepcopy({**DEFAULTS, **section})
    cli: list[str] = []

    # command-line value, None if not given
    def arg(name):
        return getattr(args, name, None) if args is not None else None

    for key in ("tot_mode", "saturation_adc", "merge_rail_adc", "quality_threshold", "spectrum_fit", "fit_dataset",
                "output_tag"):
        if arg(key) is not None:
            value = arg(key)
            if key == "fit_dataset" and isinstance(value, (list, tuple)):  # -fit-dataset pion 3: the words of the run
                value = " ".join(str(v) for v in value)
            s[key] = value
            cli.append(key)
    for key in ("slope_window", "fit_adc_window", "fit_tot_window"):
        if arg(key) is not None:
            s[key] = [float(v) for v in arg(key)]
            cli.append(key)
    if arg("bar_csv"):
        s["write_bar_csv"] = True
        cli.append("write_bar_csv")
    for key in RESPONSE_RANGE_FLAGS:  # --fit / --bin / --linfit / --resfit: response settings, recorded for
        if arg(key) is not None:      # resolve_steps (they do nothing without a response unit)
            cli.append(key)
    if s["merge_rail_adc"] is None:
        s["merge_rail_adc"] = s["saturation_adc"]
    validate_settings(s, SETTINGS_RULES, source=SETTINGS_SOURCE)
    s["slope_window"] = [float(v) for v in s["slope_window"]]
    s["output_tag"] = normalize_output_tag(s.get("output_tag"))
    s["_cli"] = cli
    s["_from_saved"] = bool(arg("from_saved"))
    return s


# the output units of this run: -only (else every unit) without the pdfs under -no-plots
def chosen_units(ctx) -> tuple:
    return tuple(u for u in UNITS if ctx.wants(u) and (u not in PLOT_UNITS or bool(ctx.make_plots)))


# what the run makes: the chosen units, the line part, the events (computed or saved); stop on an empty mix
def resolve_steps(ctx, s: dict, datasets) -> dict:
    units = chosen_units(ctx)
    cli = set(s.get("_cli", []))
    from_saved = bool(s.get("_from_saved"))
    if not units:
        raise SettingsError("adcpeak: nothing to make: the units chosen with -only are pdfs and -no-plots is given "
                            "(drop one of them)")
    steps = {u: u in units for u in UNITS}
    steps["units"] = units
    steps["line"] = steps["line_table"] or steps["line_fits"]
    steps["response"] = any(steps[u] for u in RESPONSE_UNITS) and bool(datasets)
    steps["events"] = steps["response"] and not from_saved
    steps["saved_events"] = steps["response"] and from_saved
    steps["from_saved"] = from_saved
    if s["write_bar_csv"] and not steps["events"]:
        why = ("-from-saved reads the saved events and reconstructs none" if steps["response"] else
               "no response unit is chosen, so no event is reconstructed")
        if "write_bar_csv" in cli:
            raise SettingsError(f"adcpeak: --bar-csv writes the bar values of the event reconstruction, and {why}; "
                                "drop one of them (e.g. '.\\runner.ps1 adcpeak -only energy_response -bar-csv')")
        log(f"adcpeak: write_bar_csv: true (yml) ignored: {why}")
    ranges = [flag for key, flag in RESPONSE_RANGE_FLAGS.items() if key in cli]
    if ranges and not steps["response"]:
        raise SettingsError(f"adcpeak: {' / '.join(ranges)} set the response fits (spectrum / linearity / "
                            "resolution), and no response unit is chosen in this run (-only); drop one of them")
    return steps


# response settings: energy_response DEFAULTS < yml energy_response < stage keys < flags
def response_settings(ctx, s: dict, args=None) -> dict:
    flags = {k: getattr(args, k, None) for k in er.FLAG_KEYS} if args is not None else {}
    out = er.settings(ctx, flags)
    return er.check_settings(out, source="adcpeak energy-response settings (yml energy_response, command line)")


# --------------------------------------------------------------------------
# line fit: per channel-end TOT vs ADC-peak line
# --------------------------------------------------------------------------
# adc_max bin edges of the line fit: low + width * i, the last edge above high, so the whole window is fitted
def fit_bin_edges(s: dict) -> np.ndarray:
    lo, hi = (float(v) for v in s["fit_adc_window"])
    width = float(s["fit_bin_width"])
    n_bins = int(math.floor((hi - lo) / width)) + 1
    return lo + width * np.arange(n_bins + 1)


# channel slot = (layer * 12 + strip) * 2 + end
def channel_slot(layer: np.ndarray, strip: np.ndarray, end) -> np.ndarray:
    return (np.asarray(layer, dtype=np.intp) * geo.N_STRIPS_WIDE + np.asarray(strip, dtype=np.intp)) * 2 + end


# per channel-end bins of the line fit + display sample of the fired hits
@dataclass
class FitHistograms:
    edges: np.ndarray
    counts: np.ndarray          # (N_CHANNEL_SLOTS, n_bins) hits per ADC bin
    tot_sums: np.ndarray        # (N_CHANNEL_SLOTS, n_bins) sum of TOT per ADC bin
    adc_sums: np.ndarray        # (N_CHANNEL_SLOTS, n_bins) sum of adc_max per ADC bin (x of the fit = bin mean)
    n_window: np.ndarray        # (N_CHANNEL_SLOTS,) hits passing the ADC and TOT windows
    bar_rows: np.ndarray        # (N_BAR_SLOTS,) csv rows per bar
    points: dict                # slot -> (adc, tot) display sample of the fired hits (file order)
    n_rows: int
    n_fired: np.ndarray | None = None   # (N_CHANNEL_SLOTS,) true number of fired hits (adc_max >= plot low, TOT > 0)
    tot_max: np.ndarray | None = None   # (N_CHANNEL_SLOTS,) largest TOT of those hits (NaN: none)
    sample_cap: int = 0                 # display points kept per channel-end (0 = all)


_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)


# splitmix64 hash of integers: a fixed pseudo-random 64-bit key
def _mix64(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64).reshape(-1) + _SPLITMIX_GAMMA
    x = (x ^ (x >> np.uint64(30))) * _SPLITMIX_M1
    x = (x ^ (x >> np.uint64(27))) * _SPLITMIX_M2
    return x ^ (x >> np.uint64(31))


# per channel-end random sample of the fired hits, bounded while reading (bottom-k)
class FiredHitSample:
    """Key of a hit = mix64((2 * row + end) XOR mix64(seed)); a channel keeps the cap smallest keys."""

    _NO_BOUND = np.iinfo(np.uint64).max

    # empty sample of at most cap points per channel-end
    def __init__(self, cap, seed):
        self.cap = int(cap) if cap is not None and int(cap) > 0 else 0
        self.seed_key = _mix64(np.array([int(seed) % (1 << 64)], dtype=np.uint64))[0]
        self.slot = np.empty(0, dtype=np.int16)
        self.order = np.empty(0, dtype=np.int64)
        self.key = np.empty(0, dtype=np.uint64)
        self.adc = np.empty(0, dtype=np.int16)
        self.tot = np.empty(0, dtype=np.int16)
        self.n_fired = np.zeros(N_CHANNEL_SLOTS, dtype=np.int64)
        self.tot_max = np.full(N_CHANNEL_SLOTS, -np.inf)
        self.bound = np.full(N_CHANNEL_SLOTS, self._NO_BOUND, dtype=np.uint64)  # largest kept key of a full channel

    # add the fired hits of one chunk (every hit counted, a key above the bound dropped)
    def add(self, slot: np.ndarray, order: np.ndarray, adc: np.ndarray, tot: np.ndarray) -> None:
        if not len(slot):
            return
        slot = np.asarray(slot, dtype=np.intp)
        self.n_fired += np.bincount(slot, minlength=N_CHANNEL_SLOTS)
        np.maximum.at(self.tot_max, slot, np.asarray(tot, dtype=np.float64))
        order = np.asarray(order, dtype=np.int64)
        key = _mix64(order.astype(np.uint64) ^ self.seed_key)
        adc, tot = np.asarray(adc), np.asarray(tot)
        if self.cap:
            maybe = key <= self.bound[slot]
            if not maybe.all():
                slot, order, key, adc, tot = slot[maybe], order[maybe], key[maybe], adc[maybe], tot[maybe]
            if not len(slot):
                return
        self.slot = np.concatenate([self.slot, slot.astype(np.int16)])
        self.order = np.concatenate([self.order, order])
        self.key = np.concatenate([self.key, key])
        self.adc = np.concatenate([self.adc, adc.astype(np.int16)])
        self.tot = np.concatenate([self.tot, tot.astype(np.int16)])
        if self.cap and (np.bincount(self.slot.astype(np.intp), minlength=N_CHANNEL_SLOTS) > self.cap).any():
            self._prune()

    # keep the cap smallest keys of every channel and update the bounds
    def _prune(self) -> None:
        idx = np.lexsort((self.order, self.key, self.slot))      # by slot, then key (order breaks ties)
        s_sorted = self.slot[idx]
        starts = np.zeros(len(idx), dtype=np.int64)
        boundary = np.flatnonzero(np.diff(s_sorted)) + 1
        starts[boundary] = boundary
        rank = np.arange(len(idx)) - np.maximum.accumulate(starts)
        last = idx[rank == self.cap - 1]                         # the cap-th smallest key of every full channel
        self.bound[:] = self._NO_BOUND
        self.bound[self.slot[last].astype(np.intp)] = self.key[last]
        keep = idx[rank < self.cap]
        self.slot, self.order, self.key, self.adc, self.tot = (self.slot[keep], self.order[keep], self.key[keep],
                                                               self.adc[keep], self.tot[keep])

    # slot -> (adc, tot) of the kept hits in csv order
    def points(self) -> dict:
        out = {}
        if not len(self.slot):
            return out
        idx = np.lexsort((self.order, self.slot))
        slots = self.slot[idx]
        adc_all, tot_all = self.adc[idx], self.tot[idx]
        uniq, start = np.unique(slots, return_index=True)
        stop = np.append(start[1:], len(slots))
        for u, a, b in zip(uniq, start, stop):
            out[int(u)] = (adc_all[a:b], tot_all[a:b])
        return out


# read the fit csv files chunk by chunk: hits and TOT sums per adc_max bin per channel-end (+ display sample)
def accumulate_fit_histograms(csv_paths, s: dict, chunksize: int, keep_points: bool) -> FitHistograms:
    """csv_paths: one csv or the csv files of the fit run, summed (the row numbers of the sample keys run on)."""
    paths = [csv_paths] if isinstance(csv_paths, (str, Path)) else list(csv_paths)
    edges = fit_bin_edges(s)
    n_bins = len(edges) - 1
    adc_lo, adc_hi = (float(v) for v in s["fit_adc_window"])
    tot_lo, tot_hi = (float(v) for v in s["fit_tot_window"])
    plot_lo = float(s["plot_adc_window"][0])
    counts = np.zeros(N_CHANNEL_SLOTS * n_bins, dtype=np.int64)
    sums = np.zeros(N_CHANNEL_SLOTS * n_bins, dtype=np.float64)
    adc_sums = np.zeros(N_CHANNEL_SLOTS * n_bins, dtype=np.float64)
    n_window = np.zeros(N_CHANNEL_SLOTS, dtype=np.int64)
    bar_rows = np.zeros(N_BAR_SLOTS, dtype=np.int64)
    sample = FiredHitSample(s.get("plot_max_points"), s.get("plot_sample_seed", 12345)) if keep_points else None
    n_rows = 0
    for path in paths:
        for chunk in iter_csv(path, READ_COLUMNS_FIT, chunksize):
            layer = chunk["layer"].to_numpy(np.intp)
            strip = chunk["strip"].to_numpy(np.intp)
            row0 = n_rows
            n_rows += len(chunk)
            bar_rows += np.bincount(layer * geo.N_STRIPS_WIDE + strip, minlength=N_BAR_SLOTS)
            for end in geo.ENDS:
                adc = chunk[f"adc_max_end{end}"].to_numpy(np.float64)
                tot = chunk[f"tot_end{end}"].to_numpy(np.float64)
                sel = (adc >= adc_lo) & (adc <= adc_hi) & (tot > tot_lo) & (tot < tot_hi)
                slot = channel_slot(layer[sel], strip[sel], end)
                n_window += np.bincount(slot, minlength=N_CHANNEL_SLOTS)
                ibin = np.searchsorted(edges, adc[sel], side="right") - 1          # edges[i] <= a < edges[i+1]
                inside = (ibin >= 0) & (ibin < n_bins)
                flat = slot[inside] * n_bins + ibin[inside]
                counts += np.bincount(flat, minlength=counts.size)
                sums += np.bincount(flat, weights=tot[sel][inside], minlength=sums.size)
                adc_sums += np.bincount(flat, weights=adc[sel][inside], minlength=adc_sums.size)
                if sample is not None:
                    fired = np.flatnonzero((adc >= plot_lo) & (tot > 0))
                    sample.add(channel_slot(layer[fired], strip[fired], end), (row0 + fired) * 2 + end,
                               adc[fired], tot[fired])
    extra = {}
    if sample is not None:
        tot_max = np.where(sample.n_fired > 0, sample.tot_max, np.nan)
        extra = {"n_fired": sample.n_fired, "tot_max": tot_max, "sample_cap": sample.cap}
    return FitHistograms(edges, counts.reshape(N_CHANNEL_SLOTS, n_bins), sums.reshape(N_CHANNEL_SLOTS, n_bins),
                         adc_sums.reshape(N_CHANNEL_SLOTS, n_bins), n_window, bar_rows, sample.points() if sample is not None else {}, n_rows, **extra)


# settings that decide the content of FitHistograms (bins, display sample) = the identity of a saved copy
FIT_HIST_BIN_KEYS = ("fit_adc_window", "fit_tot_window", "fit_bin_width")


# dependency tokens of saved fit bins: fit csv contents, bin and sample settings, version
def fit_histogram_tokens(s: dict, fit_dataset: str, input_identities: list) -> dict:
    # numbers as floats (949 == 949.0)
    def num(v):
        return [float(x) for x in v] if isinstance(v, (list, tuple)) else (float(v) if _number(v) else v)

    return {"schema": FIT_HIST_SCHEMA, "algorithm": FIT_HIST_ALGORITHM, "fit_dataset": fit_dataset,
            "input_csvs": list(input_identities), "bins": {k: num(s[k]) for k in FIT_HIST_BIN_KEYS},
            "sample": {"plot_adc_low": float(s["plot_adc_window"][0]),
                       "plot_max_points": int(s["plot_max_points"]) if int(s["plot_max_points"]) > 0 else 0,
                       "plot_sample_seed": int(s["plot_sample_seed"])}}


# save the fit bins + display sample to the cache (npz, provenance last)
def save_fit_histograms(path: Path, hist: FitHistograms, tokens: dict) -> dict:
    slots = np.array(sorted(hist.points), dtype=np.int64)
    sizes = np.array([len(hist.points[k][0]) for k in slots], dtype=np.int64)
    adc = np.concatenate([hist.points[k][0] for k in slots]) if len(slots) else np.zeros(0, np.int16)
    tot = np.concatenate([hist.points[k][1] for k in slots]) if len(slots) else np.zeros(0, np.int16)
    arrays = {"edges": hist.edges, "counts": hist.counts, "tot_sums": hist.tot_sums, "adc_sums": hist.adc_sums,
              "n_window": hist.n_window,
              "bar_rows": hist.bar_rows, "n_rows": np.array(hist.n_rows), "n_fired": hist.n_fired,
              "tot_max": hist.tot_max, "sample_cap": np.array(hist.sample_cap), "point_slots": slots,
              "point_sizes": sizes, "point_adc": adc, "point_tot": tot}
    with ProductSet(provenance_path(path)) as ps:
        tmp = ps.stage(path)
        with open(tmp, "wb") as fh:  # a file object: np.savez must not append another ".npz"
            np.savez(fh, **arrays)
        return ps.commit({"stage": "adcpeak", "product": "fit histograms (saved copy for -from-saved)",
                          **tokens, "n_rows": int(hist.n_rows)})


# read saved fit bins + display sample
def load_fit_histograms(path: Path) -> FitHistograms:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    points, start = {}, 0
    for slot, size in zip(a["point_slots"].tolist(), a["point_sizes"].tolist()):
        points[int(slot)] = (a["point_adc"][start:start + size], a["point_tot"][start:start + size])
        start += size
    return FitHistograms(a["edges"], a["counts"], a["tot_sums"], a["adc_sums"], a["n_window"], a["bar_rows"], points,
                         int(a["n_rows"]), n_fired=a["n_fired"], tot_max=a["tot_max"],
                         sample_cap=int(a["sample_cap"]))


# weighted least squares: slope = Σ w dx dy / Σ w dx^2, offset = y_mean - slope * x_mean
def weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    dx = x - x_mean
    slope = float(np.sum(w * dx * (y - y_mean)) / np.sum(w * dx * dx))  # slope = Σ w dx dy / Σ w dx^2
    return slope, float(y_mean - slope * x_mean)


# line fit of one channel-end: x = bin mean adc_max, y = bin mean TOT, bins > 3 sigma off dropped, weighted line
# TOT = slope * adc_max + offset
def fit_channel(counts: np.ndarray, tot_sums: np.ndarray, adc_sums: np.ndarray, n_window: int, has_rows: bool,
                s: dict) -> dict:
    out = {"tot_per_adc_slope": 0.0, "tot_offset": 0.0, "status": "ok", "n_hits_window": int(n_window),
           "n_hits_fit": 0, "n_bins_filled": 0, "n_bins_removed": 0, "n_bins_fit": 0,
           "x": np.empty(0), "mean": np.empty(0), "weight": np.empty(0), "kept": np.empty(0, dtype=bool)}
    if not has_rows:
        out["status"] = "no_data"
        return out
    if n_window == 0:
        out["status"] = "no_hits_in_window"
        return out
    filled = counts > 0
    if not filled.any():
        out["status"] = "no_filled_bins"
        return out
    weight = counts[filled].astype(np.float64)
    x = adc_sums[filled] / weight  # bin mean adc_max (adc_max is an integer: not the bin centre)
    mean = tot_sums[filled] / weight  # bin mean TOT, weight = hits
    kept = np.abs(mean - mean.mean()) <= float(s["fit_outlier_nsigma"]) * mean.std()  # 3-sigma cut
    out.update(x=x, mean=mean, weight=weight, kept=kept, n_bins_filled=int(filled.sum()),
               n_bins_removed=int((~kept).sum()))
    if kept.sum() < int(s["fit_min_bins"]):
        out["status"] = "too_few_bins_after_outlier_cut"
        return out
    slope, offset = weighted_line(x[kept], mean[kept], weight[kept])
    out.update(tot_per_adc_slope=slope, tot_offset=offset, n_bins_fit=int(kept.sum()),
               n_hits_fit=int(weight[kept].sum()))
    return out


# fit every channel-end of the 384 -> (line table, per-slot bins for the pages)
def fit_tot_lines(hist: FitHistograms, s: dict, fit_dataset: str) -> tuple[pd.DataFrame, dict]:
    rows, details = [], {}
    for layer, strip, end in geo.all_channels().itertuples(index=False):
        slot = int(channel_slot(layer, strip, end))
        res = fit_channel(hist.counts[slot], hist.tot_sums[slot], hist.adc_sums[slot], int(hist.n_window[slot]),
                          bool(hist.bar_rows[layer * geo.N_STRIPS_WIDE + strip] > 0), s)
        details[slot] = res
        rows.append({"layer": int(layer), "strip": int(strip), "end": int(end),
                     **{k: res[k] for k in PARAMETER_COLUMNS if k in res}, "fit_dataset": fit_dataset})
    return pd.DataFrame(rows, columns=list(PARAMETER_COLUMNS)), details


# usable line: low <= slope < high
def slope_usable(slope, lo: float, hi: float) -> np.ndarray:
    slope = np.asarray(slope, dtype=float)
    with np.errstate(invalid="ignore"):
        return (slope >= lo) & (slope < hi)


# counts of the line table for the log and the summary page
def parameter_summary(table: pd.DataFrame, s: dict) -> dict:
    m = table["tot_per_adc_slope"].to_numpy(float)
    b = table["tot_offset"].to_numpy(float)
    lo, hi = s["slope_window"]
    return {"rows": len(table), "zero_rows": int(((m == 0) & (b == 0)).sum()), "negative": int((m < 0).sum()),
            "ge4": int((m >= 4).sum()), f"in_[{lo:g},{hi:g})": int(slope_usable(m, lo, hi).sum()),
            "status": table["status"].value_counts().to_dict()}


# fit pdf: slope maps + one page per channel-end (fired hits, bin means, line)
def plot_fit_pages(path: Path, table: pd.DataFrame, details: dict, hist: FitHistograms, s: dict, title: str) -> None:
    from ..common import plotting as plot

    lo, hi = s["slope_window"]
    adc_lo, adc_hi = (float(v) for v in s["fit_adc_window"])
    tot_lo, tot_hi = (float(v) for v in s["fit_tot_window"])
    x_lo, x_hi = (float(v) for v in s["plot_adc_window"])
    summary = parameter_summary(table, s)
    with atomic_pdf(path) as pdf:
        fig, axes = plot.new_figure(1, 2, figsize=(14, 8.5))
        for end, ax in zip(geo.ENDS, axes):
            sub = table[table["end"] == end]
            vals = np.where(sub["status"].eq("ok"), sub["tot_per_adc_slope"], np.nan)
            plot.layer_map(ax, geo.fill_layer_grid(sub["layer"], sub["strip"], vals),
                           title=f"end {end}: fitted slope [TOT/ADC] (blank = not fittable)",
                           cbar_label="slope", cmap="coolwarm", vmin=-1, vmax=5, annotate=True, fmt="{:.2f}")
        fig.suptitle(f"{title}\nstatus: {summary['status']};  usable {lo:g} <= slope < {hi:g}: "
                     f"{summary[f'in_[{lo:g},{hi:g})']} / {summary['rows']};  negative {summary['negative']}, "
                     f">= 4: {summary['ge4']}, zero rows {summary['zero_rows']}", fontsize=10)
        fig.tight_layout()
        plot.save(fig, pdf)

        # one axes for all channel pages: static artists stay, only the per-page artists are swapped
        # (ax.cla() + automatic title placement cost ~0.2 s per page)
        fig, ax = plot.new_figure(figsize=(8, 6))
        ax.axvspan(adc_lo, adc_hi, color="#F3E6C8", alpha=0.35, zorder=0,
                   label=f"fit window {adc_lo:g} <= adc_max <= {adc_hi:g}, {tot_lo:g} < TOT < {tot_hi:g}")
        for y in (tot_lo, tot_hi):
            ax.axhline(y, color="#B0B7BF", linestyle=":", linewidth=1.0, zorder=1)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("ADC peak (adc_max) [ADC]")
        ax.set_ylabel("TOT [counts]")
        note = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8, color="#5A626A")
        page_artists: list = []
        for row in table.itertuples(index=False):
            for artist in page_artists:
                artist.remove()
            page_artists = []
            slot = int(channel_slot(row.layer, row.strip, row.end))
            det = details[slot]
            pts = hist.points.get(slot)
            n_fired, y_top = 0, 1.2 * tot_hi
            if pts is not None and len(pts[0]):
                a, t = pts
                n_fired = int(hist.n_fired[slot]) if hist.n_fired is not None else len(a)   # true count
                t_max = float(hist.tot_max[slot]) if hist.tot_max is not None else float(t.max())
                y_top = 1.05 * t_max
                drawn = f" ({len(a):,} drawn)" if n_fired > len(a) else ""
                page_artists.append(ax.scatter(a, t, s=4, color="#9DB9D5", alpha=0.5, edgecolors="none",
                                               rasterized=True,
                                               label=f"fired hits (adc_max >= {x_lo:g}, TOT > 0): {n_fired:,}"
                                                     f"{drawn}"))
            x, mean, weight, kept = det["x"], det["mean"], det["weight"], det["kept"]
            if len(x):
                size = np.clip(weight * 10, 8, 400)
                if (~kept).any():
                    page_artists.append(ax.scatter(x[~kept], mean[~kept], s=size[~kept], color="gray",
                                                   label="removed bins (3 sigma)", zorder=3))
                page_artists.append(ax.scatter(x[kept], mean[kept], s=size[kept], color="#1F5F99",
                                               label="bin mean TOT (size ~ hits)", zorder=4))
            ok = row.status == "ok"
            if ok:
                xx = np.linspace(adc_lo, adc_hi, 100)
                page_artists += ax.plot(xx, row.tot_per_adc_slope * xx + row.tot_offset, "--", color="#B5231E",
                                        linewidth=2.5, zorder=5,
                                        label=f"TOT = {row.tot_per_adc_slope:.3f}*ADC + {row.tot_offset:.3f}")
                if kept.any():
                    y_top = 1.2 * float(mean[kept].max())  # page range: 1.2 x the largest fitted bin mean
            ax.set_ylim(0, max(y_top, 1.0))
            usable = ok and lo <= row.tot_per_adc_slope < hi
            ax.set_title(f"Layer {row.layer} Strip {row.strip} end{row.end}: {row.status}"
                         f"{'' if not ok else ('  (usable)' if usable else f'  (slope outside [{lo:g}, {hi:g}))')}",
                         y=1.01)
            note.set_text(f"hits in window {row.n_hits_window:,}, used {row.n_hits_fit:,}; bins filled "
                          f"{row.n_bins_filled}, removed {row.n_bins_removed}")
            # data sit high (y range 1.2 x max bin mean): legend bottom right, above the hit-count note
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=7)
            pdf.savefig(fig)
        plot.plt.close(fig)


LINE_TABLE_COLUMNS = ("layer", "strip", "end", "tot_per_adc_slope", "tot_offset", "status")


# check a TOT line table: valid unique channel keys, finite slope / offset on every 'ok' line
def check_tot_table(df: pd.DataFrame, source: str = "TOT line table") -> pd.DataFrame:
    missing = [c for c in LINE_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise InputError(f"{source}: TOT line table lacks columns {missing} (columns: {list(df.columns)})")
    out = validate_channel_keys(df.copy(), keys=("layer", "strip", "end"), source=source)
    for col in ("tot_per_adc_slope", "tot_offset"):
        values = pd.to_numeric(out[col], errors="coerce")
        text = out[col].notna() & values.isna()
        if text.any():
            i = int(np.flatnonzero(text.to_numpy())[0])
            raise InputError(f"{source}: {col} is not a number in {int(text.sum())} row(s), first at row {i} "
                             f"({col}={out[col].iloc[i]!r})")
        out[col] = values.astype(np.float64)
    out["status"] = out["status"].astype(str).str.strip()
    ok = out["status"].eq("ok").to_numpy()
    finite = np.isfinite(out["tot_per_adc_slope"].to_numpy()) & np.isfinite(out["tot_offset"].to_numpy())
    bad = ok & ~finite
    if bad.any():
        i = int(np.argmax(bad))
        r = out.iloc[i]
        raise InputError(f"{source}: {int(bad.sum())} line(s) with status 'ok' but a missing / non-finite slope or "
                         f"offset, first at layer {r['layer']} strip {r['strip']} end {r['end']} "
                         f"(slope={r['tot_per_adc_slope']!r}, offset={r['tot_offset']!r})")
    return out


# read this stage's line table, floats parsed exactly, then checked
def load_tot_parameters(path: Path) -> pd.DataFrame:
    return check_tot_table(pd.read_csv(path, float_precision="round_trip"), source=str(path))


# settings that decide the bins of a line fit (recorded in the table's provenance json)
FIT_SETTING_KEYS = ("fit_adc_window", "fit_tot_window", "fit_bin_width", "fit_outlier_nsigma", "fit_min_bins")


# --------------------------------------------------------------------------
# event reconstruction: Arbitrary Energy = Σ bar values of an event
# --------------------------------------------------------------------------
# per (bar slot, end) lookups of the event reconstruction
@dataclass
class EventInputs:
    ped_sample: np.ndarray        # pedestal_per_time_sample [ADC]
    hit_threshold: np.ndarray     # adc_sum threshold of an end hit [ADC]
    slope: np.ndarray             # line used for the TOT conversion (own or mean), NaN if none
    offset: np.ndarray
    conversion_allowed: np.ndarray
    n_usable_lines: int
    mean_slope: float
    mean_offset: float


# (layer, strip, end) array -> (bar slot, end)
def _slot_table(arr3: np.ndarray) -> np.ndarray:
    return arr3.reshape(N_BAR_SLOTS, 2)


# event lookups: pedestal per sample, hit threshold, usable line (per -tot-mode; tot_mode none needs no table)
def build_event_inputs(calib: pd.DataFrame, params: pd.DataFrame | None, s: dict) -> EventInputs:
    ped = _slot_table(calibration_lookup(calib, "pedestal_per_time_sample"))
    column = s["hit_threshold_column"]
    if column in calib.columns:
        thr = _slot_table(calibration_lookup(calib, column))
    else:
        log(f"adcpeak: calibration table has no '{column}'; using 1.2 * pedestal as the hit threshold")
        thr = 1.2 * _slot_table(calibration_lookup(calib, "pedestal"))
    if s["tot_mode"] == "none":
        nan = np.full((N_BAR_SLOTS, 2), np.nan)
        return EventInputs(ped, thr, nan, nan.copy(), np.zeros(nan.shape, dtype=bool), 0, np.nan, np.nan)
    table = check_tot_table(params)
    m = _slot_table(calibration_lookup(table, "tot_per_adc_slope"))
    b = _slot_table(calibration_lookup(table, "tot_offset"))
    table["_ok"] = (table["status"] == "ok").astype(float)
    fit_ok = _slot_table(calibration_lookup(table, "_ok", fill=0.0)) > 0
    lo, hi = s["slope_window"]
    usable = slope_usable(m, lo, hi) & np.isfinite(b)   # a line needs a finite offset too
    if s["require_fit_ok"]:
        usable &= fit_ok
    real = np.zeros((N_BAR_SLOTS, 2), dtype=bool)
    ch = geo.all_channels()
    real[ch["layer"].to_numpy() * geo.N_STRIPS_WIDE + ch["strip"].to_numpy(), ch["end"].to_numpy()] = True
    order = np.flatnonzero((usable & real).ravel())  # (layer, strip, end) order
    mean_slope = float(m.ravel()[order].mean()) if order.size else np.nan
    mean_offset = float(b.ravel()[order].mean()) if order.size else np.nan
    if s["tot_mode"] == "linear":
        slope, offset, allowed = np.where(usable, m, np.nan), np.where(usable, b, np.nan), usable
    else:  # mean_fallback: the mean usable line for the channels without their own
        slope, offset = np.where(usable, m, mean_slope), np.where(usable, b, mean_offset)
        allowed = real & np.isfinite(slope)
    return EventInputs(ped, thr, slope, offset, allowed, int((usable & real).sum()), mean_slope, mean_offset)


# end value = adc_max - ped; at the rail max(that, (TOT - offset) / slope - ped)
def _end_value(adc_sum, adc, tot, slot, end, inp: EventInputs, s: dict) -> dict:
    sat = float(s["saturation_adc"])
    ped = inp.ped_sample[slot, end]
    thr = inp.hit_threshold[slot, end]
    with np.errstate(invalid="ignore"):
        keep = np.isfinite(ped) & (adc_sum >= thr)
        raw = adc - ped  # raw peak = adc_max - pedestal_per_time_sample
        near_rail = adc >= sat
        tpos = tot > 0
        if s["tot_mode"] == "none":
            branch = np.zeros(adc.shape, dtype=bool)
            tot_value = np.full(adc.shape, np.nan)
        else:
            candidate = keep & near_rail & tpos & inp.conversion_allowed[slot, end]
            with np.errstate(divide="ignore"):
                tot_value = (tot - inp.offset[slot, end]) / inp.slope[slot, end] - ped  # TOT peak - ped
            branch = candidate & np.isfinite(tot_value) & (tot_value > 0)
        used = branch & (tot_value > raw)
        value = np.where(branch, np.maximum(raw, tot_value), raw)  # max(raw, TOT peak) at the rail
        value = np.where(keep, value, np.nan)
        valid = keep & np.isfinite(value) & (value >= 0)
    return {"keep": keep, "sat": adc >= float(s["merge_rail_adc"]), "tpos": tpos, "near_rail": keep & near_rail,
            "branch": branch, "value": value, "valid": valid, "used": used, "below_rail": adc < sat}


# rail-aware end merge of the charge injection (provided by Erik Wallin); NaN = no hit
def merge_ends_rail_aware(e0: dict, e1: dict) -> np.ndarray:
    n = len(e0["value"])
    bar = np.full(n, np.nan)
    hit = e0["keep"] | e1["keep"]
    no_valid = np.zeros(n, dtype=bool)
    sat0_only = e0["sat"] & ~e1["sat"]
    sat1_only = e1["sat"] & ~e0["sat"]
    for sat_only, es, eo in ((sat0_only, e0, e1), (sat1_only, e1, e0)):
        a = sat_only & es["tpos"]
        m1 = a & es["valid"]
        m2 = a & ~es["valid"] & eo["valid"]
        no_valid |= a & ~es["valid"] & ~eo["valid"]
        b = sat_only & ~es["tpos"]
        m3 = b & eo["valid"]
        m4 = b & ~eo["valid"] & es["valid"]
        no_valid |= b & ~eo["valid"] & ~es["valid"]
        bar[m1] = es["value"][m1]
        bar[m2] = eo["value"][m2]
        bar[m3] = eo["value"][m3]
        bar[m4] = es["value"][m4]
    general = ~sat0_only & ~sat1_only & hit
    c0 = general & e0["used"] & e0["valid"]
    c1 = general & e1["used"] & e1["valid"]
    both = c0 & c1
    bar[both] = 0.5 * (e0["value"][both] + e1["value"][both])
    only0, only1 = c0 & ~c1, c1 & ~c0
    bar[only0] = e0["value"][only0]
    bar[only1] = e1["value"][only1]
    none = general & ~c0 & ~c1
    both_raw = none & e0["valid"] & e1["valid"]
    raw0 = none & e0["valid"] & ~e1["valid"]
    raw1 = none & e1["valid"] & ~e0["valid"]
    bar[both_raw] = 0.5 * (e0["value"][both_raw] + e1["value"][both_raw])
    bar[raw0] = e0["value"][raw0]
    bar[raw1] = e1["value"][raw1]
    no_valid |= none & ~e0["valid"] & ~e1["valid"]
    bar[no_valid & hit] = 0.0
    return bar


# raw end merge (charge injection): mean of two valid ends, else the valid end, else 0
def merge_ends_raw(e0: dict, e1: dict) -> np.ndarray:
    v0, v1 = e0["valid"], e1["valid"]
    bar = np.full(len(v0), np.nan)
    both = v0 & v1
    bar[both] = 0.5 * (e0["value"][both] + e1["value"][both])
    bar[v0 & ~v1] = e0["value"][v0 & ~v1]
    bar[v1 & ~v0] = e1["value"][v1 & ~v0]
    bar[(e0["keep"] | e1["keep"]) & ~v0 & ~v1] = 0.0
    return bar


# bar values + bad-bar flags (unrecoverable: railed end without TOT, no reliable other end)
def reconstruct_bars(adc_sum, adc, tot, slot, inp: EventInputs, s: dict) -> dict:
    e0, e1 = (_end_value(adc_sum[e], adc[e], tot[e], slot, e, inp, s) for e in geo.ENDS)
    bar = merge_ends_raw(e0, e1) if s["tot_mode"] == "none" else merge_ends_rail_aware(e0, e1)
    included = np.isfinite(bar)
    missing = [e["near_rail"] & ~e["branch"] for e in (e0, e1)]            # railed without usable TOT
    reliable = [e["used"] | (e["valid"] & e["below_rail"]) for e in (e0, e1)]
    return {
        "present": included,
        "bar": bar,
        "end_values": [np.where(e["valid"], e["value"], np.nan) for e in (e0, e1)],
        "n_ends_hit": e0["keep"].astype(np.int8) + e1["keep"].astype(np.int8),
        "n_ends_converted": e0["used"].astype(np.int8) + e1["used"].astype(np.int8),
        "n_ends_saturated_unconverted": sum((e["near_rail"] & ~e["used"]).astype(np.int8) for e in (e0, e1)),
        "unrecoverable": included & ((missing[0] & ~reliable[1]) | (missing[1] & ~reliable[0])),
        "saturated_unconverted": included & (e0["near_rail"] | e1["near_rail"]) & ~e0["used"] & ~e1["used"],
    }


# per-event sums over streamed chunks (one running table, sums in file order)
class EventAccumulator:

    # empty running table of the given columns
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.ids = np.empty(0, dtype=np.int64)
        self.sums = {k: np.empty(0) for k in columns}

    # add the per-event sums of one chunk
    def add(self, pf: np.ndarray, values: dict) -> None:
        if pf.size == 0:
            return
        ids, inverse = np.unique(pf, return_inverse=True)
        part = {k: np.bincount(inverse, weights=values[k], minlength=len(ids)) for k in self.columns}
        if not self.ids.size:
            self.ids, self.sums = ids, part
            return
        if ids.size == self.ids.size and np.array_equal(ids, self.ids):    # the usual case: the same events
            for k in self.columns:
                self.sums[k] += part[k]
            return
        merged = np.union1d(self.ids, ids)
        if merged.size == self.ids.size:                                    # no new event
            pos = np.searchsorted(self.ids, ids)
            for k in self.columns:
                self.sums[k][pos] += part[k]
            return
        old_pos, new_pos = np.searchsorted(merged, self.ids), np.searchsorted(merged, ids)
        for k in self.columns:
            total = np.zeros(merged.size)
            total[old_pos] = self.sums[k]
            total[new_pos] += part[k]
            self.sums[k] = total
        self.ids = merged

    # (event ids, sums per column)
    def finalize(self) -> tuple[np.ndarray, dict]:
        return self.ids, self.sums


# bar reconstruction of one chunk of beam rows -> (pf_event per row, bar values)
def _reconstruct_chunk(chunk: pd.DataFrame, inp: EventInputs, s: dict) -> tuple[np.ndarray, dict]:
    pf = chunk["pf_event"].to_numpy(np.int64)
    slot = chunk["layer"].to_numpy(np.intp) * geo.N_STRIPS_WIDE + chunk["strip"].to_numpy(np.intp)
    adc = [chunk[f"adc_max_end{e}"].to_numpy(np.float64) for e in geo.ENDS]
    tot = [chunk[f"tot_end{e}"].to_numpy(np.float64) for e in geo.ENDS]
    adc_sum = [chunk[f"adc_sum_end{e}"].to_numpy(np.float64) for e in geo.ENDS]
    return pf, reconstruct_bars(adc_sum, adc, tot, slot, inp, s)


# bars of one chunk as bar-value rows
def _bar_frame(source: str, chunk: pd.DataFrame, pf: np.ndarray, rec: dict, rows: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "pf_event": pf[rows], "layer": chunk["layer"].to_numpy()[rows],
        "strip": chunk["strip"].to_numpy()[rows], "bar_arbitrary_energy": rec["bar"][rows],
        "end0_arbitrary_energy": rec["end_values"][0][rows],
        "end1_arbitrary_energy": rec["end_values"][1][rows]})
    frame.insert(0, "event_source", source)
    return frame


# do the bar values need complete events (active quality cut)?
def event_selection_needed(s: dict) -> bool:
    return s["quality_threshold"] is not None


# Arbitrary Energy of the events of one csv: arbitrary_energy = Σ bar values of the event, read chunk by chunk
def arbitrary_energies_for_csv(csv_path: Path, source: str, inp: EventInputs, s: dict, chunksize: int,
                               keep_bars: bool = False, bar_writer=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """source: '<run>/<csv name>'. With a quality cut the bar values are written in a second pass."""
    acc_cols = ["arbitrary_energy", "n_bars", "n_ends_hit", "n_ends_converted", "n_ends_saturated_unconverted",
                "n_unrecoverable_bars", "n_saturated_unconverted_bars"]
    acc = EventAccumulator(acc_cols)
    stream_now = bar_writer is not None and not event_selection_needed(s)
    bar_parts = []
    for chunk in iter_csv(csv_path, READ_COLUMNS_EVENTS, chunksize):
        pf, rec = _reconstruct_chunk(chunk, inp, s)
        present = rec["present"]
        counted = np.isfinite(rec["bar"])
        values = {"arbitrary_energy": np.where(counted, rec["bar"], 0.0)[present],  # Σ bar values per event
                  "n_bars": counted[present].astype(np.float64),
                  "n_ends_hit": rec["n_ends_hit"][present].astype(np.float64),
                  "n_ends_converted": rec["n_ends_converted"][present].astype(np.float64),
                  "n_ends_saturated_unconverted": rec["n_ends_saturated_unconverted"][present].astype(np.float64),
                  "n_unrecoverable_bars": rec["unrecoverable"][present].astype(np.float64),
                  "n_saturated_unconverted_bars": rec["saturated_unconverted"][present].astype(np.float64)}
        acc.add(pf[present], values)
        if counted.any() and (keep_bars or stream_now):
            frame = _bar_frame(source, chunk, pf, rec, counted)
            if keep_bars:
                bar_parts.append(frame)
            if stream_now:
                bar_writer.write(frame)
    ids, sums = acc.finalize()
    events = pd.DataFrame({"event_source": source, "pf_event": ids})
    for key in acc_cols:
        events[key] = sums[key] if key == "arbitrary_energy" else sums[key].round().astype(np.int64)
    events["unrecoverable_bar_fraction"] = events["n_unrecoverable_bars"] / np.maximum(events["n_bars"], 1)
    thr = s["quality_threshold"]
    events["selected"] = True if thr is None else events["unrecoverable_bar_fraction"] <= float(thr) + 1e-12
    events = events[list(EVENT_COLUMNS)]
    selected_ids = np.sort(events.loc[events["selected"].astype(bool), "pf_event"].to_numpy())
    bars = None
    if keep_bars:
        bars = pd.concat(bar_parts, ignore_index=True) if bar_parts else pd.DataFrame(columns=BAR_COLUMNS)
        bars = bars[np.isin(bars["pf_event"].to_numpy(), selected_ids)]
    if bar_writer is not None and not stream_now:  # second pass: bars of the selected (complete) events only
        for chunk in iter_csv(csv_path, READ_COLUMNS_EVENTS, chunksize):
            wanted = np.isin(chunk["pf_event"].to_numpy(np.int64), selected_ids)
            if not wanted.any():
                continue
            if not wanted.all():
                chunk = chunk[wanted]
            pf, rec = _reconstruct_chunk(chunk, inp, s)
            rows = np.isfinite(rec["bar"])
            if rows.any():
                bar_writer.write(_bar_frame(source, chunk, pf, rec, rows))
    return events, bars


# --------------------------------------------------------------------------
# saved events: dependency tokens and verification
# --------------------------------------------------------------------------
# settings that decide the events (not the quality threshold, no response setting)
def reconstruction_settings(s: dict) -> dict:
    out = {"tot_mode": s["tot_mode"], "saturation_adc": float(s["saturation_adc"]),
           "hit_threshold_column": str(s["hit_threshold_column"])}
    if s["tot_mode"] != "none":
        out.update(slope_window=[float(v) for v in s["slope_window"]], require_fit_ok=bool(s["require_fit_ok"]),
                   merge_rail_adc=float(s["merge_rail_adc"]))
    return out


# dependency tokens of the events of one csv: version, variant, settings, content of the inputs
def event_tokens(s: dict, calibration: dict, tot_parameters: dict | str, input_identity: dict) -> dict:
    return {"schema": EVENT_SCHEMA, "algorithm": EVENT_ALGORITHM, "variant_tag": variant_tag(s),
            "reconstruction": reconstruction_settings(s), "calibration_content": calibration,
            "tot_parameters_content": tot_parameters, "input_csv": input_identity}


# content identities shared by every csv of a run: (calibration table, line table or 'unused')
def shared_identities(ctx, s: dict) -> tuple[dict, dict | str]:
    tot = TOT_TABLE_UNUSED if s["tot_mode"] == "none" else _identity_or_missing(parameters_csv(ctx))
    return _identity_or_missing(ctx.calibration_file()), tot


# tokens that the saved events of one csv must carry to be reused now (path: the csv read for it, if known)
def current_event_tokens(ctx, s: dict, ds, csv, shared=None, plan_only: bool = False, path=None) -> dict:
    calib, tot = shared if shared is not None else shared_identities(ctx, s)
    if path is None:
        path = ctx.input_plan_item(ds, csv)["path"] if plan_only else ctx.input_csv(ds, csv)
    return event_tokens(s, calib, tot, _identity_or_missing(path))


# short text of a content identity
def _short_hash(value) -> str:
    if isinstance(value, dict) and "sha256" in value:
        return f"sha256 {value['sha256'][:12]}, {value['size']:,} bytes"
    return json.dumps(value, default=str)


# name of the input kind (clean dataset or beam csv)
def _input_kind(clean: bool) -> str:
    return "the clean datasets (-clean-input)" if clean else "the beam csv"


# readable reasons why saved events do not match the tokens (empty: they match)
def token_differences(record: dict, expected: dict, run: dict | None = None, label: str = "") -> list[str]:
    problems = []
    for key in ("schema", "algorithm", "variant_tag"):
        if record.get(key) != expected[key]:
            problems.append(f"{key}: saved {record.get(key)!r}, now {expected[key]!r}")
    saved_rec, now_rec = record.get("reconstruction") or {}, expected["reconstruction"]
    for key in sorted(set(saved_rec) | set(now_rec)):
        if json.dumps(saved_rec.get(key)) != json.dumps(now_rec.get(key)):
            problems.append(f"{key}: saved {saved_rec.get(key, '(not used)')!r}, "
                            f"now {now_rec.get(key, '(not used)')!r}")
    for key, what in (("calibration_content", "calibration table"), ("tot_parameters_content", "TOT line table")):
        if json.dumps(record.get(key), sort_keys=True) != json.dumps(expected[key], sort_keys=True):
            problems.append(f"the {what} differs (saved: {_short_hash(record.get(key))}; now: "
                            f"{_short_hash(expected[key])})")
    if json.dumps(record.get("input_csv"), sort_keys=True) != json.dumps(expected["input_csv"], sort_keys=True):
        now_clean = bool((run or {}).get("clean_input"))
        saved_clean = bool(record.get("clean_input"))
        if run is not None and saved_clean != now_clean:  # the content decides; these lines only say why
            problems.append(f"input: the saved events were made from {_input_kind(saved_clean)}, this run reads "
                            f"{_input_kind(now_clean)}")
        if run is not None and record.get("test_mode") and run.get("test_mode") and \
                record.get("n_test_events") != run.get("n_test_events"):
            problems.append(f"test events: saved {record.get('n_test_events')}, now {run.get('n_test_events')}")
        problems.append(f"the {'clean dataset' if now_clean else 'beam csv'} of {label} differs (saved: "
                        f"{_short_hash(record.get('input_csv'))}; now: {_short_hash(expected['input_csv'])})")
    return problems


# (saved copy, its settings) of one csv read by a response from the cache
def saved_events_source(ctx, s: dict, ds, csv) -> tuple[Path, dict]:
    """An -output-tag without a saved copy of its own reads the copy of the same variant without the tag."""
    path = saved_events_csv(ctx, ds, csv, output_suffix(s))
    if s.get("output_tag") and not path.exists() and not provenance_path(path).exists():
        base = {**s, "output_tag": None}
        return saved_events_csv(ctx, ds, csv, output_suffix(base)), base
    return path, s


# (True, 'ok') when the saved events of one csv may be reused, else (False, reason)
def saved_events_status(ctx, s: dict, ds, csv, shared=None, plan_only: bool = False,
                        path_read=None) -> tuple[bool, str]:
    path, s_copy = saved_events_source(ctx, s, ds, csv)
    label = f"{run_ref(ds)} ({ctx.event_source(ds, csv)})"
    if not path.exists() and not provenance_path(path).exists():
        return False, f"no saved events of {label}"
    ok, reason = product_set_valid(provenance_path(path), None, members=[path])
    if not ok:
        return False, reason
    problems = token_differences(read_json(provenance_path(path)) or {},
                                 current_event_tokens(ctx, s_copy, ds, csv, shared, plan_only, path_read),
                                 run=_run_record(ctx), label=label)
    return (not problems), ("; ".join(problems) if problems else "ok")


# a setting as command-line text
def _flag_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_flag_value(v) for v in value)
    if isinstance(value, (float, np.floating)):
        text = repr(float(value))
        return text[:-2] if text.endswith(".0") else text
    text = str(value).removeprefix("\\\\?\\")  # Windows long-path prefix of a resolved path: not for a command line
    return f'"{text}"' if (not text or any(c.isspace() for c in text)) else text


# runner options a rebuild command repeats: -config, -test [-test-events], -clean-input
def _mode_flags(ctx) -> str:
    flags = ""
    cfg_file = ctx.cfg.get("_config_file")
    if cfg_file and Path(cfg_file).resolve() != DEFAULT_CONFIG.resolve():
        flags += f" -config {_flag_value(cfg_file)}"
    if ctx.test:
        flags += " -test"
        if int(ctx.n_test_events) != int((ctx.cfg.get("run") or {}).get("test_events", 5000)):
            flags += f" -test-events {ctx.n_test_events}"
    if ctx.clean_input:
        flags += " -clean-input"
    return flags


# stage flags that a rebuild command repeats when this run set them on the command line (the rebuild reads the
# same yml again, so yml values need no flag)
STAGE_FLAGS = {"tot_mode": "-tot-mode", "slope_window": "-slope-window", "saturation_adc": "-saturation-adc",
               "merge_rail_adc": "-merge-rail-adc", "output_tag": "-output-tag", "fit_dataset": "-fit-dataset",
               "fit_adc_window": "-fit-adc-window", "fit_tot_window": "-fit-tot-window"}
EVENT_REBUILD_KEYS = ("tot_mode", "slope_window", "saturation_adc", "merge_rail_adc", "output_tag")
FIT_HIST_REBUILD_KEYS = ("fit_dataset", "fit_adc_window", "fit_tot_window")


# stage flags of keys given on this run's command line, with their values
def _stage_flags(s: dict, keys) -> str:
    cli, out = set(s.get("_cli", [])), ""
    for key in keys:
        value = s.get(key)
        if key not in cli or value is None or value is False:
            continue
        if key == "fit_dataset":  # a run reference takes its words unquoted: -fit-dataset pion 3
            out += f" {STAGE_FLAGS[key]} {value}"
            continue
        out += f" {STAGE_FLAGS[key]}" + ("" if value is True else f" {_flag_value(value)}")
    return out


# command that rebuilds the saved events (and the response table)
def events_command(ctx, s: dict, datasets) -> str:
    return (f"{RUNNER} adcpeak {' '.join(run_flags(d) for d in datasets)}{_mode_flags(ctx)}"
            f"{_stage_flags(s, EVENT_REBUILD_KEYS)} -only energy_response")


# command that rebuilds the saved fit bins (and the line table)
def fit_histograms_command(ctx, s: dict) -> str:
    return f"{RUNNER} adcpeak{_mode_flags(ctx)}{_stage_flags(s, FIT_HIST_REBUILD_KEYS)} -only line_table"


# saved events of every csv of the runs after the token check (+ run key column), else stop with the rebuild command
def load_saved_events(ctx, s: dict, datasets) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (events, one source entry per csv: path, committed)."""
    shared = shared_identities(ctx, s)
    frames, sources = [], []
    for ds in datasets:
        for csv, read in ctx.run_inputs(ds):
            path, s_copy = saved_events_source(ctx, s, ds, csv)
            if s_copy is not s:
                log(f"adcpeak: no saved copy {saved_events_csv(ctx, ds, csv, output_suffix(s)).name} of "
                    f"--output-tag {s['output_tag']}: reading {path.name} (same variant without the tag)")
            ok, reason = saved_events_status(ctx, s, ds, csv, shared, path_read=read)
            if not ok:
                raise ProductSetError(f"adcpeak: cannot reuse the saved events {path}: {reason}.\n"
                                      f"  Rebuild them with: {events_command(ctx, s_copy, datasets)}")
            events = pd.read_csv(path, dtype={"event_source": str}, float_precision="round_trip")
            if list(events.columns) != list(EVENT_COLUMNS):
                raise ProductSetError(f"adcpeak: {path} has the columns {list(events.columns)}, not "
                                      f"{list(EVENT_COLUMNS)}; rebuild it with: "
                                      f"{events_command(ctx, s_copy, datasets)}")
            events.insert(0, "dataset", ds.key)
            frames.append(events)
            sources.append({"path": path, "committed": (read_json(provenance_path(path)) or {}).get("committed")})
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset", *EVENT_COLUMNS])
    return events, sources


# do the bar values of one csv belong to its current saved events? (read by the plots stage)
def bar_values_status(ctx, ds, csv, tag: str = "", mode: str | None = None) -> dict:
    bars, events = bar_values_csv(ctx, ds, csv, tag, mode), saved_events_csv(ctx, ds, csv, tag, mode)
    out = {"bar_csv": bars, "exists": bars.exists(), "current": None, "reason": ""}
    if not bars.exists():
        out["reason"] = "no bar values (make them with -bar-csv)"
        return out
    ok, reason = product_set_valid(provenance_path(events), None, members=[events, bars])
    out["current"] = bool(ok)
    out["reason"] = "ok" if ok else reason
    return out


# --------------------------------------------------------------------------
# energy response
# --------------------------------------------------------------------------
# (fit method, bin edges, page name) of the run's spectrum page without -fit
def default_page_edges(values, s: dict, rs: dict) -> tuple[str, np.ndarray, str]:
    if s["spectrum_fit"] == "peak_scan":
        return "peak_scan", er.default_spectrum_edges(values, bins=int(rs["peak_scan_bins"]),
                                                      upper_percentile=float(rs["peak_scan_upper_percentile"]),
                                                      upper_factor=float(rs["peak_scan_upper_factor"]),
                                                      upper_min=float(rs["peak_scan_upper_min"])), "peak-scan histogram"
    return "two_pass", er.default_spectrum_edges(values, bins=int(rs["spectrum_bins"]),
                                                 upper_percentile=float(rs["hist_upper_percentile"]),
                                                 upper_factor=float(rs["hist_upper_factor"]),
                                                 upper_min=float(rs["hist_upper_min"])), "two-pass page"


# status / footer note of a -fit spectrum fit: ranges, model, replaced fit, bins
def range_fit_note(fit: dict, replaced: str, page: str) -> str:
    model = ("double Gaussian (no single mu: no linearity / resolution point)"
             if fit.get("kind") == "range_double_gauss" else "single Gaussian")
    bins = fit.get("bin_rule", "") + ("" if fit.get("fit_bins") else f" of the {page}")
    return f"--fit {fit.get('fit_ranges_text', '')}: {model}, replaces the automatic {replaced} fit; {bins}"


# spectrum fit of every run (all its csv files): two_pass | peak_scan, or the -fit fit
def fit_spectra(events: pd.DataFrame, datasets, s: dict, rs: dict) -> list[dict]:
    out = []
    for ds in datasets:
        sel = (events["dataset"] == ds.key) & events["selected"].astype(bool)
        values = events.loc[sel, "arbitrary_energy"].to_numpy(np.float64)
        if rs.get("fit_ranges"):  # --fit[a,b] (single) / --fit[a,b],[c,d] (double): the user range(s) only
            replaced, edges, page = default_page_edges(values, s, rs)
            fit = er.range_spectrum_from_settings(values, rs, edges)
            out.append({"dataset": ds, "method": "range", "note": range_fit_note(fit, replaced, page), "fit": fit,
                        "values": values})
            continue
        if s["spectrum_fit"] == "peak_scan":
            fit = er.peak_scan_fit(values, n_bins=int(rs["peak_scan_bins"]),
                                   upper_percentile=float(rs["peak_scan_upper_percentile"]),
                                   upper_factor=float(rs["peak_scan_upper_factor"]),
                                   upper_min=float(rs["peak_scan_upper_min"]))
        else:
            fit = er.two_pass_gauss_fit(values, bins=int(rs["spectrum_bins"]), min_bin_count=int(rs["min_bin_count"]),
                                        window_nsigma=float(rs["window_nsigma"]), min_fit_bins=int(rs["min_fit_bins"]),
                                        upper_percentile=float(rs["hist_upper_percentile"]),
                                        upper_factor=float(rs["hist_upper_factor"]),
                                        upper_min=float(rs["hist_upper_min"]))
        out.append({"dataset": ds, "method": s["spectrum_fit"], "note": "", "fit": fit, "values": values})
    return out


# linearity and resolution from the spectrum points (every particle drawn; lines per -linfit, resolution: electrons)
def response_fits(spectra: list[dict], rs: dict) -> dict:
    energies = np.array([p["dataset"].energy_gev for p in spectra], dtype=float)
    particles = [str(p["dataset"].particle).lower() for p in spectra]
    mu = np.array([p["fit"].get("mu", np.nan) for p in spectra], dtype=float)
    sigma = np.array([p["fit"].get("sigma", np.nan) for p in spectra], dtype=float)
    sigma_err = np.array([p["fit"].get("sigma_error", np.nan) for p in spectra], dtype=float)
    n = np.array([p["fit"].get("n_values", np.nan) for p in spectra], dtype=float)
    errs = er.point_errors(mu, sigma, n=n, sigma_error=sigma_err, prescription=rs["error_prescription"])
    lin = er.linearity_from_settings(energies, mu, errs["mu_error"], particles, rs)  # adcpeak: no default line
    resolution = er.resolution_from_settings(energies, errs["ratio"], errs["ratio_error"], rs, particles=particles)
    return {"points": spectra, "energies": energies, "particles": particles, "errors": errs, "n": n,
            "linearity": lin, "resolution": resolution}


# table columns of the two components of a double-Gaussian --fit (fitting.range_gauss_fit keys, in range order)
RANGE_COMPONENT_KEYS = tuple(f"{name}_{tag}{suffix}" for tag in ("low", "high") for name, suffix in
                             (("mu", ""), ("mu", "_error"), ("sigma", ""), ("sigma", "_error"), ("area", "")))
# column order of energy_response.csv (the columns a table has, in this order)
RESPONSE_COLUMNS = (
    "record", "tot_mode", "units", "dataset", "particle", "energy_gev", "spectrum_fit", "n_events", "mu", "mu_error",
    "sigma", "sigma_error", "sigma_over_mu", "fit_low", "fit_high", "n_fit_bins", "chi2", "ndf", "status",
    "fit_ranges", "bin_rule", "sigma_over_mu_error", *RANGE_COMPONENT_KEYS, "point_mu_error", "point_ratio_error",
    "point_error_n", "in_resolution_fit", "resolution_predicted", "in_linearity_fit", "linearity_predicted",
    "linearity_residual_fraction", "fit_range_gev", "ok", "weighted", "slope", "slope_error", "intercept",
    "intercept_error", "chi2_over_ndf", "n_fitted", "fitted_energies", "model", "a", "a_error", "b", "b_error", "c",
    "c_error", "error_prescription", "error_scale", "slope_error_scaled", "intercept_error_scaled", "a_error_scaled",
    "b_error_scaled", "c_error_scaled")
SCALED_COLUMNS = ("error_scale", "slope_error_scaled", "intercept_error_scaled", "a_error_scaled", "b_error_scaled",
                  "c_error_scaled")


# energy_response.csv: spectrum rows, then linearity and resolution rows
def response_table(resp: dict, s: dict, rs: dict) -> pd.DataFrame:
    common = {"tot_mode": s["tot_mode"], "units": "arbitrary units"}
    lin = resp["linearity"]
    res = resp["resolution"]
    rows = []
    for i, p in enumerate(resp["points"]):
        fit, ds = p["fit"], p["dataset"]
        row = {"record": "spectrum", **common, "dataset": ds.key, "particle": ds.particle, "energy_gev": ds.energy_gev,
               "spectrum_fit": p["method"], "n_events": fit.get("n_values"), "mu": fit.get("mu"),
               "mu_error": fit.get("mu_error"), "sigma": fit.get("sigma"), "sigma_error": fit.get("sigma_error"),
               "sigma_over_mu": fit.get("sigma_over_mu"), "fit_low": fit.get("fit_low"),
               "fit_high": fit.get("fit_high"), "n_fit_bins": fit.get("n_fit_bins", fit.get("core_bin_count")),
               "chi2": fit.get("chi2"), "ndf": fit.get("ndf"),
               "status": (fit.get("status") + (f" ({p['note']})" if p["note"] else ""))}
        if p["method"] == "range":  # --fit: the ranges, the bins and, for two ranges, both components
            row.update(fit_ranges=fit.get("fit_ranges_text"), bin_rule=fit.get("bin_rule"),
                       sigma_over_mu_error=fit.get("sigma_over_mu_error"))
            if fit.get("kind") == "range_double_gauss":
                row.update({k: fit.get(k) for k in RANGE_COMPONENT_KEYS})
        row.update(point_mu_error=resp["errors"]["mu_error"][i], point_ratio_error=resp["errors"]["ratio_error"][i],
                   point_error_n=resp["n"][i], in_resolution_fit=bool(res["fitted"][i]),
                   resolution_predicted=res["predicted"][i], in_linearity_fit=bool(lin["fitted"][i]))
        line = lin["fits"].get(resp["particles"][i])
        if line is not None:  # the line of the point's own particle
            row.update(linearity_predicted=line["predicted"][i],
                       linearity_residual_fraction=line["residual_fraction"][i])
        rows.append(row)
    for lrow in er.linearity_rows(lin):
        rows.append({"record": "linearity", **common, **lrow})
    rng = res.get("fit_range")
    rows.append({"record": "resolution", **common, "particle": res["fit_particle"],
                 "fit_range_gev": "all points" if rng is None else format_ranges([rng]), "ok": bool(res["ok"]),
                 "model": res["model"], "a": res["a"], "a_error": res["a_error"], "b": res["b"],
                 "b_error": res["b_error"], "c": res["c"], "c_error": res["c_error"], "chi2": res["chi2"],
                 "ndf": res["ndf"], "chi2_over_ndf": res["chi2_over_ndf"], "n_fitted": res["n_fitted"],
                 "fitted_energies": res["fitted_energies"], "status": res["status"],
                 "error_prescription": rs["error_prescription"], "error_scale": res["error_scale"],
                 "a_error_scaled": res["a_error_scaled"], "b_error_scaled": res["b_error_scaled"],
                 "c_error_scaled": res["c_error_scaled"]})
    table = pd.DataFrame(rows)
    keep = [c for c in RESPONSE_COLUMNS if c in table.columns and (rs.get("normal_error") or c not in SCALED_COLUMNS)]
    table = table[keep]
    for col in ("n_events", "n_fit_bins", "ndf", "n_fitted", "point_error_n"):
        if col in table.columns:  # integers with blanks on the other record types
            table[col] = pd.to_numeric(table[col], errors="coerce").round().astype("Int64")
    return table


# why a run has no linearity / resolution point (None if it has one)
def no_point_reason(p: dict) -> str | None:
    fit = p["fit"]
    if fit.get("ok"):
        return "double Gaussian --fit: no single mu" if fit.get("kind") == "range_double_gauss" else None
    status = str(fit.get("status") or "no status")
    status = status[len("fit_failed: "):] if status.startswith("fit_failed: ") else status
    return "spectrum fit failed: " + (status if len(status) <= 160 else status[:157] + "...")


# one row of the spectrum summary page
def _spectrum_summary_row(p: dict) -> dict:
    fit = p["fit"]
    row = {"dataset": p["dataset"].key, "fit": p["method"], "events": fit.get("n_values"), "mu": fit.get("mu"),
           "sigma": fit.get("sigma"), "sigma/mu": fit.get("sigma_over_mu"), "status": fit.get("status")}
    if p["method"] == "range":
        row["fit range(s)"] = fit.get("fit_ranges_text")
        if fit.get("kind") == "range_double_gauss":
            row["mu low / high"] = f"{fit.get('mu_low', np.nan):.4g} / {fit.get('mu_high', np.nan):.4g}"
            row["sigma low / high"] = f"{fit.get('sigma_low', np.nan):.4g} / {fit.get('sigma_high', np.nan):.4g}"
    return row


# energy_spectra pdf (one page per run + summary) and / or linearity_resolution pdf (None: not drawn)
def plot_response(spectra_path: Path | None, linres_path: Path | None, resp: dict, rs: dict, title: str) -> None:
    from ..common import plotting as plot

    spectra = resp["points"]
    if spectra_path is not None:
        with atomic_pdf(spectra_path) as pdf:
            for p in spectra:
                fig, ax = er.new_page()
                er.plot_spectrum_with_fit(ax, p["values"], p["fit"], ARBITRARY_ENERGY,
                                          title=f"{p['dataset'].label}: Arbitrary Energy ({title})")
                if p["note"]:
                    er.add_footer(ax, p["note"])
                plot.save(fig, pdf)
            er.summary_table_page(pdf, f"Spectrum fits ({title})", [_spectrum_summary_row(p) for p in spectra])
    if linres_path is None:
        return
    with atomic_pdf(linres_path) as pdf:
        error_note = f"point errors: {rs['error_prescription']}, N = events"
        if rs.get("normal_error"):
            error_note += "; parameter errors x sqrt(chi2/ndf) when chi2/ndf > 1 (--normalerror)"
        reasons = [no_point_reason(p) for p in spectra]  # used only for runs without a point
        fig, ax = er.new_page()
        er.plot_linearity(ax, resp["linearity"], ARBITRARY_ENERGY, title=f"Linearity of the Arbitrary Energy ({title})",
                          counts=resp["n"], no_point_reasons=reasons)
        er.add_footer(ax, error_note)
        plot.save(fig, pdf)
        fig, ax = er.new_page()
        er.plot_resolution(ax, resp["resolution"], ARBITRARY_ENERGY,
                           title=f"Resolution of the Arbitrary Energy ({title})", no_point_reasons=reasons)
        er.add_footer(ax, error_note)
        plot.save(fig, pdf)
        rows = [{"dataset": p["dataset"].key, "E [GeV]": p["dataset"].energy_gev, "mu": p["fit"].get("mu"),
                 "mu error": resp["errors"]["mu_error"][i], "sigma/mu": resp["errors"]["ratio"][i],
                 "sigma/mu error": resp["errors"]["ratio_error"][i]} for i, p in enumerate(spectra)]
        er.summary_table_page(pdf, f"Linearity / resolution points ({title})", rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
# adcpeak command-line options (+ -only and the shared energy-response flags)
def add_arguments(p: argparse.ArgumentParser) -> None:
    outputs.add_only_argument(p, UNITS, UNIT_GROUPS)
    g = p.add_argument_group("adcpeak options")
    g.add_argument("--fit-dataset", "-fit-dataset", nargs="+", default=None, metavar=("PARTICLE", "E"),
                   help=f"run of the line fit, any run of configure/datasets.yml: -fit-dataset pion 3 (also one "
                        f"quoted word \"pion 3\") (default {DEFAULTS['fit_dataset']})")
    g.add_argument("--fit-adc-window", "-fit-adc-window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="adc_max window of the fit, inclusive; the bins cover all of it (default 949 1021)")
    g.add_argument("--fit-tot-window", "-fit-tot-window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="TOT window of the fit, exclusive (default 500 2000; 2024 version 1 500)")
    g.add_argument("--slope-window", "-slope-window", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                   help="usable line: LOW <= slope < HIGH (default 0.6 4)")
    g.add_argument("--tot-mode", "-tot-mode", choices=TOT_MODES, default=None,
                   help="linear | none (ADC only) | mean_fallback (mean line for unusable channels) (default linear)")
    g.add_argument("--saturation-adc", "-saturation-adc", type=float, default=None, metavar="ADC",
                   help=f"adc_max >= ADC is saturated / near the rail (default {DEFAULTS['saturation_adc']})")
    g.add_argument("--merge-rail-adc", "-merge-rail-adc", type=float, default=None, metavar="ADC",
                   help="end merge: end at the rail if adc_max >= ADC (default: --saturation-adc; the CI system "
                        "uses 1023)")
    g.add_argument("--quality-threshold", "-quality-threshold", type=float, default=None, metavar="X",
                   help="keep events with unrecoverable_bar_fraction <= X (default: off; CI uses 0.125)")
    g.add_argument("--spectrum-fit", "-spectrum-fit", choices=SPECTRUM_FITS, default=None,
                   help="two_pass | peak_scan (default two_pass)")
    g.add_argument("--bar-csv", "-bar-csv", action="store_true",
                   help="also write the bar values of every input csv into cache/[test/]tot_adc_peak/<run>/ for "
                        "'.\\runner.ps1 plots -energy arbitrary' (needs a response unit, no -from-saved; default off)")
    g.add_argument("--output-tag", "-output-tag", default=None, metavar="TAG",
                   help="extra suffix of the cache and response files: letters, digits, '.', '_', '-', starting "
                        "with a letter or digit; not 'none', 'mean_fallback' or one of them + '_...' (those are the "
                        "automatic variant tags) (default none)")
    er.add_fit_arguments(p)


# the one run of the line fit (stop when the reference names none or several)
def _fit_dataset(ctx, s: dict, ref: str | None = None, what: str | None = None):
    ref = s["fit_dataset"] if ref is None else ref
    if what is None:
        what = "-fit-dataset" if "fit_dataset" in s.get("_cli", []) else "tot_adc_peak.fit_dataset"
    found = resolve_runs(ctx.cfg, ref, what)
    if len(found) != 1:
        hint = f"; name one of them, e.g. -fit-dataset {run_ref(found[0])}" if found else ""
        raise SystemExit(f"{what}: the line fit reads one run, and {str(ref)!r} names {len(found)} "
                         f"({', '.join(run_ref(d) for d in found) or 'none'}){hint}")
    return found[0]


# runner command text
def _producer(stage_args: str) -> str:
    return f"{RUNNER} {stage_args}"


# command that writes this mode's line table
def _table_producer(ctx) -> str:
    return _producer(f"adcpeak{' -test' if ctx.test else ''} -only line_table")


# --------------------------------------------------------------------------
# final results (common/outputs.py): the fit pdf, the response csv and the two response pdfs of
# results/<mode>/tot_adc_peak/ go where the output plan of the run says; the line table and the caches are
# intermediates (the chain and -from-saved read only them)
# --------------------------------------------------------------------------
RESULT_WHAT = {
    "line_fits": "TOT vs ADC-peak line fit pages (summary + one page per channel-end)",
    "energy_response": "spectrum / linearity / resolution table",
    "energy_spectra": "Arbitrary Energy spectra with the peak fits",
    "linearity_resolution": "linearity and resolution pages",
}


# final-result slots of the chosen units, in write order
def result_slots(ctx, s: dict, steps: dict) -> list[outputs.ResultSlot]:
    suffix = output_suffix(s)
    defaults = {"line_fits": ("pdf", fits_pdf(ctx)), "energy_response": ("csv", energy_response_csv(ctx, suffix)),
                "energy_spectra": ("pdf", energy_spectra_pdf(ctx, suffix)),
                "linearity_resolution": ("pdf", linearity_resolution_pdf(ctx, suffix))}
    slots = []
    for key, (kind, default) in defaults.items():
        if steps.get(key) and (key == "line_fits" or steps["response"]):
            slots.append(outputs.ResultSlot(key, kind, Path(default), RESULT_WHAT[key]))
    return slots


# output plan of this stage (an -output name given twice stops the run)
def _output_plan(ctx, slots, preview: bool = False) -> outputs.OutputPlan:
    try:
        return outputs.OutputPlan(ctx, STAGE_DIR, slots, preview=preview)
    except ValueError as exc:   # two results would be written to one file
        raise SettingsError(f"adcpeak: {exc}") from None


# output plan of the run: -output names, test names or the unsaved scratch copy
def result_plan(ctx, s: dict, steps: dict, preview: bool = False) -> outputs.OutputPlan:
    """preview (dry-run, check, run log): same destinations, the scratch folder is not emptied."""
    return _output_plan(ctx, result_slots(ctx, s, steps), preview)


# provenance path of a final result: the default record, or <name>.provenance.json
def result_record(plan: outputs.OutputPlan, key: str) -> Path:
    slot = next(sl for sl in plan.slots if sl.key == key)
    return plan.provenance(key, default_record=provenance_path(slot.default))


# log where the final results went (or that they were not saved)
def report_results(plan: outputs.OutputPlan) -> None:
    unsaved = [sl for sl in plan.slots if not plan.saved(sl.key)]
    if unsaved:
        log(f"final results not saved (no --output): scratch copies in {plan.path(unsaved[0].key).parent}; use "
            "--output [names] to save them")
    for slot, line in zip(plan.slots, plan.summary_lines()):
        if plan.saved(slot.key):
            log(f"final result {line}")


# dry-run / check items of the final results at their planned destinations
def _result_items(plan: outputs.OutputPlan) -> dict[str, dict]:
    return {sl.key: {"role": "out", "path": plan.path(sl.key), "result": sl.key, "saved": plan.saved(sl.key),
                     "note": f"final result {line}" if plan.saved(sl.key) else "not saved: no --output (scratch copy)"}
            for sl, line in zip(plan.slots, plan.summary_lines())}


# final results of the last adcpeak run in this process (for the run log)
def last_run_results() -> dict[str, dict]:
    for plan in reversed(outputs.plans()):
        if plan.stage_dir == STAGE_DIR:
            return {sl.key: {"path": plan.path(sl.key), "saved": plan.saved(sl.key),
                             "provenance": result_record(plan, sl.key) if sl.key == "energy_response" else None}
                    for sl in plan.slots}
    return {}


# inputs and outputs of the run (for -dry-run, check and the run log); settings checked as in run
def io_plan(ctx, args, datasets) -> list[dict]:
    s = stage_settings(ctx, args)
    steps = resolve_steps(ctx, s, datasets)
    if steps["response"]:
        response_settings(ctx, s, args)  # the same refusal as run(), before any plan line is printed
    preview = result_plan(ctx, s, steps, preview=True)
    for w in preview.warnings:
        log(f"WARNING: {w}")
    results = _result_items(preview)
    suffix = output_suffix(s)
    test = " -test" if ctx.test else ""
    plan = []
    if steps["line"]:
        ds = _fit_dataset(ctx, s)
        hist = saved_fit_histograms(ctx, ds.key)
        if steps["from_saved"]:
            plan.append(dict(role="in", kind="product", path=hist, produced_by=fit_histograms_command(ctx, s)))
            sources = [item["path"] for item in ctx.input_plan_items(ds)]
            plan.append(dict(role="cache", path=provenance_path(hist), members=[hist],
                             expected=fit_histogram_tokens(s, ds.key, [_identity_or_missing(p) for p in sources])))
        else:
            plan += ctx.input_plan_items(ds, produced_by=_producer(f"make-csv {run_flags(ds)}"))
            plan.append(dict(role="out", path=hist))
        if steps["line_table"]:
            plan.append(dict(role="out", path=parameters_csv(ctx)))
        if steps["line_fits"]:
            plan.append(results["line_fits"])
    if steps["response"]:
        if steps["events"]:
            for ds in datasets:
                plan += ctx.input_plan_items(ds, produced_by=_producer(f"make-csv {run_flags(ds)}"))
            plan.append(dict(role="in", path=ctx.calibration_file(), produced_by=_producer(f"calib{test}")))
            if s["tot_mode"] != "none" and not steps["line_table"]:
                plan.append(dict(role="in", path=parameters_csv(ctx), produced_by=_table_producer(ctx)))
            for ds in datasets:
                for csv in ctx.run_csvs(ds):
                    plan.append(dict(role="out", path=saved_events_csv(ctx, ds, csv, suffix)))
                    if s["write_bar_csv"]:
                        plan.append(dict(role="out", path=bar_values_csv(ctx, ds, csv, suffix)))
        else:
            shared = shared_identities(ctx, s)
            for ds in datasets:
                for csv in ctx.run_csvs(ds):
                    saved, s_copy = saved_events_source(ctx, s, ds, csv)
                    plan.append(dict(role="in", kind="product", path=saved,
                                     produced_by=events_command(ctx, s_copy, datasets)))
                    plan.append(dict(role="cache", path=provenance_path(saved), members=[saved],
                                     expected=current_event_tokens(ctx, s_copy, ds, csv, shared, plan_only=True)))
        plan += [results[k] for k in RESPONSE_UNITS if k in results]
    return plan


# csv rows per chunk (run.chunksize)
def _chunksize(ctx) -> int:
    return int(ctx.cfg.get("run", {}).get("chunksize", 2_000_000))


# append DataFrames to one csv (header once, full precision)
class CsvStream:

    # new stream into path
    def __init__(self, path: Path, float_format: str | None = None):
        self.path = Path(path)
        self.float_format = float_format
        self._header = True
        self.rows = 0

    # append one frame
    def write(self, df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        df.to_csv(self.path, mode="w" if self._header else "a", header=self._header, index=False,
                  float_format=self.float_format)
        self._header = False
        self.rows += len(df)

    # header-only csv when nothing was written; returns the path
    def finish(self, columns: list[str]) -> Path:
        if self._header:
            pd.DataFrame(columns=columns).to_csv(self.path, index=False)
            self._header = False
        return self.path


# write the event table of one csv (full precision; read back with float_precision='round_trip')
def _write_events_csv(events: pd.DataFrame, path: Path) -> None:
    events.to_csv(path, index=False)


# reconstruction text of the page titles
def _logic_title(s: dict) -> str:
    lo, hi = s["slope_window"]
    text = f"TOT {s['tot_mode']}"
    if s["tot_mode"] != "none":
        text += f" {lo:g} <= slope < {hi:g}"
    if s["quality_threshold"] is not None:
        text += f", bad-bar fraction <= {s['quality_threshold']:g}"
    return text


# run part of a provenance record
def _run_record(ctx) -> dict:
    return {"stage": "adcpeak", "module": __name__, "mode": ctx.mode, "test_mode": ctx.test,
            "n_test_events": ctx.n_test_events if ctx.test else None, "clean_input": bool(ctx.clean_input)}


# saved fit bins after the token check, else stop with the rebuild command
def load_verified_fit_histograms(ctx, s: dict, ds) -> tuple[FitHistograms, list]:
    """Returns (bins, content identities of the fit csv files)."""
    hist_path = saved_fit_histograms(ctx, ds.key)
    sources = [require(path, produced_by=_producer(f"make-csv {run_flags(ds)}")) for _, path in ctx.run_inputs(ds)]
    identities = [content_identity(p) for p in sources]
    ok, reason = product_set_valid(provenance_path(hist_path), fit_histogram_tokens(s, ds.key, identities),
                                   members=[hist_path])
    if not ok:
        raise ProductSetError(f"adcpeak: cannot reuse the saved fit histograms {hist_path}: {reason}.\n"
                              f"  Rebuild them with: {fit_histograms_command(ctx, s)} (reads the fit run's csv once)")
    log(f"  bins and display sample from the saved copy {hist_path.name} (fit csv and settings verified)")
    return load_fit_histograms(hist_path), identities


# line part: fit bins of the fit run (its csv files, or the saved copy) -> line table and / or fit pdf
def run_line(ctx, s: dict, steps: dict, plan: outputs.OutputPlan | None = None) -> list[Path]:
    ds = _fit_dataset(ctx, s)
    source = "-fit-dataset" if "fit_dataset" in s.get("_cli", []) else "tot_adc_peak.fit_dataset"
    hist_path = saved_fit_histograms(ctx, ds.key)
    written = []
    with timed(f"TOT vs ADC-peak line fit on {ds.key} (fit run {run_ref(ds)} from {source})"):
        if steps["from_saved"]:
            hist, identities = load_verified_fit_histograms(ctx, s, ds)
            bins_from = f"saved copy {hist_path}"
        else:
            paths = [require(path, produced_by=_producer(f"make-csv {run_flags(ds)}"))
                     for _, path in ctx.run_inputs(ds)]
            hist = accumulate_fit_histograms(paths, s, _chunksize(ctx), keep_points=True)
            identities = [content_identity(p) for p in paths]
            save_fit_histograms(hist_path, hist, fit_histogram_tokens(s, ds.key, identities))
            written.append(hist_path)
            bins_from = "csv files: " + ", ".join(str(p) for p in paths)
        table, details = fit_tot_lines(hist, s, ds.key)
    log(f"line fit {parameter_summary(table, s)}")
    if steps["line_table"]:
        out = parameters_csv(ctx)
        fit_settings = {k: s[k] for k in FIT_SETTING_KEYS}
        with ProductSet(provenance_path(out)) as ps:
            table.to_csv(ps.stage(out), index=False)
            ps.commit({**_run_record(ctx), "product": "TOT line table", "fit_dataset": ds.key,
                       "fit_bins_from": bins_from, "inputs": {"fit_csvs": identities}, "rows_read": hist.n_rows,
                       "bin_edges": hist.edges.tolist(), **fit_settings,
                       "fit_settings_hash": settings_hash(fit_settings), "columns": list(PARAMETER_COLUMNS),
                       "relation": "TOT = tot_per_adc_slope * adc_max + tot_offset"})
        log(f"line table -> {out}")
        written.append(out)
    if steps["line_fits"]:
        pdf_path = plan.path("line_fits") if plan is not None else fits_pdf(ctx)
        with timed(f"fit pages -> {pdf_path}"):
            plot_fit_pages(pdf_path, table, details, hist, s,
                           f"TOT vs ADC-peak line fit on {ds.label}{' [TEST MODE]' if ctx.test else ''}")
        written.append(pdf_path)
    return written


# events of one csv -> its saved copy (+ bar values with -bar-csv), one product set
def run_csv_events(ctx, s: dict, ds, csv, path: Path, inp: EventInputs, shared: tuple) -> pd.DataFrame:
    suffix = output_suffix(s)
    source = ctx.event_source(ds, csv)
    out, bars = saved_events_csv(ctx, ds, csv, suffix), bar_values_csv(ctx, ds, csv, suffix)
    tokens = event_tokens(s, shared[0], shared[1], content_identity(path))
    with ProductSet(provenance_path(out)) as ps:
        writer = CsvStream(ps.stage(bars)) if s["write_bar_csv"] else None
        with timed(f"Arbitrary Energy of {source} ({path.name})"):
            events, _ = arbitrary_energies_for_csv(path, source, inp, s, _chunksize(ctx), bar_writer=writer)
        if writer is not None:
            writer.finish(BAR_COLUMNS)
        _write_events_csv(events, ps.stage(out))
        ps.commit({**_run_record(ctx), "product": "Arbitrary Energy per event (saved copy for -from-saved)",
                   **tokens, "event_source": source, "input_path": str(path), "rows": int(len(events)),
                   "quality_threshold": s["quality_threshold"], "columns": list(EVENT_COLUMNS),
                   "bar_values": {"path": str(bars), "rows": writer.rows, "columns": BAR_COLUMNS,
                                  "content": "counted bars of the events with selected == true"}
                   if writer is not None else None})
    if writer is None and bars.exists():
        bars.unlink()  # the bar values of earlier events no longer belong to this saved copy
    log(f"  {source}: {len(events):,} events, {int(events['selected'].sum()):,} selected, mean "
        f"{events['arbitrary_energy'].mean():.1f} a.u., converted ends {int(events['n_ends_converted'].sum()):,}")
    return events


# event part: Arbitrary Energy of every csv of every run -> the saved copies in the cache
def run_events(ctx, s: dict, datasets) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (events with the run key column, one source entry per csv)."""
    test = " -test" if ctx.test else ""
    calib_path = require(ctx.calibration_file(), produced_by=_producer(f"calib{test}"))
    calib = load_calibration(calib_path)
    params = None
    if s["tot_mode"] != "none":
        params = load_tot_parameters(require(parameters_csv(ctx), produced_by=_table_producer(ctx)))
    inp = build_event_inputs(calib, params, s)
    shared = shared_identities(ctx, s)
    lo, hi = s["slope_window"]
    lines = ("no TOT lines (tot_mode none)" if s["tot_mode"] == "none" else
             f"lines {parameters_csv(ctx).name}: {inp.n_usable_lines}/384 usable ({lo:g} <= slope < {hi:g})")
    log(f"event reconstruction {_logic_title(s)}; calibration {calib_path.name}; {lines}"
        + (f", mean line slope {inp.mean_slope:.5f} offset {inp.mean_offset:.3f}" if s["tot_mode"] == "mean_fallback"
           else ""))
    frames, sources = [], []
    for ds in datasets:
        for csv, path in ctx.run_inputs(ds):
            path = require(path, produced_by=_producer(f"make-csv {run_flags(ds)}"))
            events = run_csv_events(ctx, s, ds, csv, path, inp, shared)
            events.insert(0, "dataset", ds.key)
            frames.append(events)
            out = saved_events_csv(ctx, ds, csv, output_suffix(s))
            sources.append({"path": out, "committed": (read_json(provenance_path(out)) or {}).get("committed")})
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset", *EVENT_COLUMNS])
    return events, sources


# the response units of a run (energy_response; the pdfs unless -no-plots)
def _response_units(ctx) -> tuple:
    return tuple(u for u in chosen_units(ctx) if u in RESPONSE_UNITS)


# response: spectrum fits, linearity, resolution -> energy_response csv and pdfs of the chosen units
def run_response(ctx, s: dict, rs: dict, datasets, events: pd.DataFrame | None = None, sources=None,
                 plan: outputs.OutputPlan | None = None) -> pd.DataFrame:
    """events None: the verified saved copies (-from-saved); the quality cut of this run is re-applied."""
    steps = {u: u in _response_units(ctx) for u in UNITS}
    steps["response"] = True
    plan = plan if plan is not None else _output_plan(ctx, result_slots(ctx, s, steps))
    origin = "computed in this run"
    if events is None:
        events, sources = load_saved_events(ctx, s, datasets)
        origin = "saved copy"
        log("events from the saved copies in the cache (reconstruction settings, calibration, line table and input "
            "csv verified)")
        thr = s["quality_threshold"]  # the event cut follows this run's --quality-threshold, not the saved column
        selected = (np.ones(len(events), dtype=bool) if thr is None
                    else events["unrecoverable_bar_fraction"].to_numpy(float) <= float(thr) + 1e-12)
        if int(selected.sum()) != int(events["selected"].astype(bool).sum()):
            log(f"  selection re-applied (quality threshold {thr}): {int(selected.sum()):,} of {len(events):,} "
                f"events selected (saved column: {int(events['selected'].astype(bool).sum()):,})")
        events["selected"] = selected
    events = events.copy()
    events["dataset"] = events["dataset"].astype(str)
    if rs.get("fit_ranges"):  # --fit: one user range fit per run instead of the automatic one
        n_ranges = len(rs["fit_ranges"])
        log(f"adcpeak: --fit {format_ranges(rs['fit_ranges'])} a.u. ({'double' if n_ranges == 2 else 'single'} "
            f"Gaussian) replaces the automatic {s['spectrum_fit']} fit of every run of this command")
        if len(datasets) > 1:
            log(f"adcpeak: NOTE the same --fit range is used for {', '.join(run_ref(d) for d in datasets)}; their "
                f"peaks lie at different values, so normally choose one run per --fit command (e.g. "
                f"'{RUNNER} adcpeak --electron 1 -from-saved -only response -fit[a,b] -output-tag TAG')")
    spectra = fit_spectra(events, datasets, s, rs)
    resp = response_fits(spectra, rs)
    table = response_table(resp, s, rs)
    for p in spectra:
        f = p["fit"]
        if f.get("kind") == "range_double_gauss":  # no single mu / sigma: both components
            log(f"  {p['dataset'].key}: double Gaussian, low mu = {f.get('mu_low', np.nan):.2f} sigma = "
                f"{f.get('sigma_low', np.nan):.2f}, high mu = {f.get('mu_high', np.nan):.2f} sigma = "
                f"{f.get('sigma_high', np.nan):.2f} ({p['method']} {f.get('fit_ranges_text')}; {f.get('status')})")
            continue
        log(f"  {p['dataset'].key}: mu = {f.get('mu', np.nan):.2f}, sigma = {f.get('sigma', np.nan):.2f}, "
            f"sigma/mu = {f.get('sigma_over_mu', np.nan):.4f}, sigma_err = {f.get('sigma_error', np.nan):.2f} "
            f"({p['method']}; {f.get('status')})")
    log(f"  linearity: {resp['linearity']['status']}")
    res = resp["resolution"]
    log(f"  resolution ({res['fit_particle']} points): a {res['a']:.4g}, b {res['b']:.4g}, c {res['c']:.4g} "
        f"({res['status']})")
    if steps["energy_response"]:
        out = plan.path("energy_response")
        with ProductSet(result_record(plan, "energy_response")) as ps:
            table.to_csv(ps.stage(out), index=False)
            ps.commit({**_run_record(ctx), "product": "energy response", "variant_tag": variant_tag(s),
                       "events": {"from": origin, "files": sources or []},
                       "reconstruction": reconstruction_settings(s), "quality_threshold": s["quality_threshold"],
                       "spectrum_fit": "range" if rs.get("fit_ranges") else s["spectrum_fit"],
                       "datasets": [d.key for d in datasets], "columns": list(table.columns),
                       "response_settings": {k: rs[k] for k in sorted(rs)}})
        log(f"energy response -> {out}")
    if steps["energy_spectra"] or steps["linearity_resolution"]:
        title = _logic_title(s) + (" [TEST MODE]" if ctx.test else "")
        with timed("response pdfs"):
            plot_response(plan.path("energy_spectra") if steps["energy_spectra"] else None,
                          plan.path("linearity_resolution") if steps["linearity_resolution"] else None,
                          resp, rs, title)
    return table


# stage entry: settings, units, output plan, then the line part and the response
def run(ctx, args, datasets) -> None:
    s = stage_settings(ctx, args)
    steps = resolve_steps(ctx, s, datasets)
    rs = response_settings(ctx, s, args) if steps["response"] else None  # checked before any file is read
    plan = result_plan(ctx, s, steps)
    written = []
    if steps["from_saved"]:
        log(f"adcpeak -from-saved: fit bins and events from {cache_dir(ctx)}")
    if steps["line"]:
        written += run_line(ctx, s, steps, plan)
    if steps["response"]:
        events = sources = None
        if steps["events"]:
            events, sources = run_events(ctx, s, datasets)
            written += [Path(src["path"]) for src in sources]
        run_response(ctx, s, rs, datasets, events, sources, plan=plan)
        written += [plan.path(k) for k in RESPONSE_UNITS if steps[k]]
    log("adcpeak wrote:\n  " + "\n  ".join(str(p) for p in written))
    report_results(plan)
