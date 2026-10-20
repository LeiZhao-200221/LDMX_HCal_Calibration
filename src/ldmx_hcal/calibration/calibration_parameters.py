"""Stage calib: per-channel pedestal and MIP MPV of one source run -> calibration_parameters.csv.

pedestal: the most probable reading of a channel without signal (pedestal events: raw TOT and TOA sums of both ends
are 0); pedestal = anchor + mean of adc_sum - anchor inside |adc_sum - anchor| < 200 ADC.
mpv: the most probable reading of a MIP in a channel after the pedestal is subtracted: the peak of a Landau (x) Gauss
fit of x = adc_sum - pedestal above 5 pedestal sigma.

Output units (-only): pedestal_csv, mpv_csv, pedestal_plot (pedestal_fits.pdf), mpv_plot (mpv_fits.pdf); groups
pedestal, mpv, csv, plots, all.  Both csv units give the full table calibration_parameters.csv (the table the later
stages read); one csv unit alone gives its part table pedestal_parameters.csv or mpv_parameters.csv.  Every run
computes what it writes from the csv files of the source run (the MPV needs the pedestal, so an MPV-only run computes
it too); nothing is taken from an existing table.

Variables of calibration_parameters.csv (one row per channel, column order):
  layer, strip, end,
  pedestal, pedestal_per_time_sample, pedestal_sigma, pedestal_mu, pedestal_range_with_5sigma,
  pedestal_range_with_20percent, n_pedestal_events,
  mpv, mpv_error, landau_width, gauss_sigma, mpv_fit_low, mpv_fit_high, n_mpv_entries, mpv_chi2_ndf, mpv_status,
  pedestal_anchor, pedestal_anchor_method, pedestal_selection, n_selected_events,
  mpv_method, mpv_pedestal, mpv_pedestal_sigma, mpv_threshold, mpv_rough_floor, mpv_rough_status, mpv_rough_peak,
  mpv_rough_width, mpv_fit_status, mpv_fit_message, mpv_covariance_status, mpv_active_bounds, landau_mpv_param,
  langaus_area, n_mpv_fit_entries, mpv_deviance, mpv_ndf
Variables of pedestal_parameters.csv: layer, strip, end + the pedestal columns in the same order (pedestal ...
  n_pedestal_events, pedestal_anchor, pedestal_anchor_method, pedestal_selection, n_selected_events).
Variables of mpv_parameters.csv: layer, strip, end + the MPV columns in the same order (mpv ... mpv_status,
  mpv_method ... mpv_ndf); mpv_pedestal and mpv_pedestal_sigma are the pedestal and sigma the fit used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..common import fitting
from ..common import geometry as geo
from ..common import io_utils
from ..common import outputs
from ..common.calibration_io import (CALIBRATION_COLUMNS, INTEGER_COLUMNS, KEY_COLUMNS, STRING_COLUMNS,
                                     empty_calibration)
from ..common.config import MODE_FULLSET, resolve_runs, run_flags, run_ref
from ..common.io_utils import log

HELP = ("pedestal + MIP MPV per channel (one pass over the csv files of the source run: the run flag, default muon 4) "
        "-> calibration_parameters.csv")
DEFAULT_DATASETS = None      # no run flag: the source run is calibration.source_dataset ("muon 4"), see source_runs
CONFIG_SECTION = "calibration"
STAGE_DIR = "pedestal_mpv"   # calibration/[test/]pedestal_mpv, results/<mode>/pedestal_mpv, cache/[test/]pedestal_mpv
SUPPORTS_FROM_SAVED = True   # refit from the raw histograms in cache/[test/]pedestal_mpv/
SUPPORTS_CLEAN_INPUT = True  # MPV from the clean dataset of each csv (RunContext.input_csv), pedestal from the beam csv
DEFAULTS = {
    "output_csv": "calibration_parameters.csv",
    "histogram_cache": "calibration_histograms.npz",
    "write_histogram_cache": True,
    "source_dataset": "muon 4",   # a run reference; a run flag (calib --electron 4) replaces it
    # pedestal
    "pedestal_selection": "raw_samples",
    "pedestal_anchor": "auto",
    "pedestal_window_halfwidth": 200.0,
    "pedestal_range_nsigma": 5.0,
    "pedestal_range_fraction": 0.2,
    # MIP MPV
    "mpv_nsigma": 5.0,
    "mpv_hist_range": [0.0, 2000.0],
    "mpv_rough_bin_width": 40.0,
    "mpv_final_bin_width": 20.0,
    "mpv_window_widths": 2.0,
    "mpv_rough_likelihood": "neyman",
    "mpv_rough_start": "valley",
    "mpv_window_above_valley": False,
    "mpv_require_rough_convergence": False,
    "mpv_min_entries": 100,
    "mpv_min_bins": 6,
    "mpv_eta_range": [5.0, 1000.0],
    "mpv_sigma_range": [1.0, 1000.0],
    "langaus_n_steps": 100,
    "langaus_n_sigma": 8.0,
    "langaus_max_step_per_eta": 0.5,
}
PEDESTAL_CSV, MPV_CSV = "pedestal_parameters.csv", "mpv_parameters.csv"   # part tables (one csv unit alone)

N_ADC_SUM = geo.N_TIME_SAMPLES * geo.ADC_SAMPLE_RAIL + 1   # adc_sum values 0..8184
N_ADC_SAMPLE = geo.ADC_SAMPLE_RAIL + 1                     # single-sample values 0..1023
N_CHANNELS = 384
SAMPLE_COLUMNS = [f"adc_{i}_end{e}" for e in geo.ENDS for i in range(geo.N_TIME_SAMPLES)]
RAW_FLAG_COLUMNS = ["raw_tot_sum_end0", "raw_tot_sum_end1", "raw_toa_sum_end0", "raw_toa_sum_end1"]
CSV_FLAG_COLUMNS = ["tot_end0", "tot_end1", "toa_end0", "toa_end1"]
SELECTION_RAW, SELECTION_CSV = "raw_samples", "csv_tot_toa_zero"
ANCHOR_MODE, ANCHOR_MEDIAN = "sample_mode", "median_sum"
MPV_METHOD = "two_stage_langaus"
PARAM_NAMES = ("landau_mpv_param", "landau_width", "langaus_area", "gauss_sigma")
# what the final fit returns: NaN in the table unless mpv_status is OK (R3, Lei 2026-10-02; ``_clear_failed_fit``)
FIT_VALUE_COLUMNS = PARAM_NAMES + ("mpv_chi2_ndf", "mpv_deviance", "mpv_ndf")
PEDESTAL_COLUMNS = ["pedestal", "pedestal_per_time_sample", "pedestal_sigma", "pedestal_mu",
                    "pedestal_range_with_5sigma", "pedestal_range_with_20percent", "n_pedestal_events",
                    "pedestal_anchor", "pedestal_anchor_method", "pedestal_selection", "n_selected_events"]
MPV_COLUMNS = ["mpv", "mpv_error", "landau_width", "gauss_sigma", "mpv_fit_low", "mpv_fit_high",
               "n_mpv_entries", "mpv_chi2_ndf", "mpv_status",
               "mpv_method", "mpv_pedestal", "mpv_pedestal_sigma", "mpv_threshold", "mpv_rough_floor",
               "mpv_rough_status", "mpv_rough_peak", "mpv_rough_width", "mpv_fit_status", "mpv_fit_message",
               "mpv_covariance_status", "mpv_active_bounds", "landau_mpv_param", "langaus_area",
               "n_mpv_fit_entries", "mpv_deviance", "mpv_ndf"]
PEDESTAL_PLOT_RANGE = (-200, 400)   # pedestal plot range around the anchor [ADC]
PEDESTAL_PLOT_BIN = 6               # 100 bins over the plot range

# raw-histogram cache (dependency tokens of -from-saved; bump a version when its definition changes)
CACHE_FORMAT = "calib-histograms-v3"            # v3: histograms summed over the csv files of the run
ACCUMULATOR_VERSION = "adc-sum-int-bincount-v1"  # what accumulate_histograms counts and how
HISTOGRAM_SCHEMA = {"channels": N_CHANNELS, "channel_order": "geometry.all_channels (layer, strip, end)",
                    "adc_sum_values": N_ADC_SUM, "adc_sample_values": N_ADC_SAMPLE}
CACHE_IDENTITY = {"format": CACHE_FORMAT, "accumulator": ACCUMULATOR_VERSION, "schema": HISTOGRAM_SCHEMA}


# --------------------------------------------------------------------------
# output units (-only, Lei 2026-10-20): one table or one pdf each
# --------------------------------------------------------------------------
UNITS = ("pedestal_csv", "mpv_csv", "pedestal_plot", "mpv_plot")
UNIT_GROUPS = {"pedestal": ("pedestal_csv", "pedestal_plot"), "mpv": ("mpv_csv", "mpv_plot"),
               "csv": ("pedestal_csv", "mpv_csv"), "plots": ("pedestal_plot", "mpv_plot")}


# units this run makes {unit: on}: -only (default all); -no-plots drops the plot units
def chosen_units(ctx) -> dict[str, bool]:
    return {u: ctx.wants(u) and (u.endswith("_csv") or bool(ctx.make_plots)) for u in UNITS}


# the chosen units, or stop when nothing is left to make
def units_or_stop(ctx) -> dict[str, bool]:
    units = chosen_units(ctx)
    if not any(units.values()):
        raise SystemExit("calib: nothing to make: -only chose only plot units (pedestal_plot, mpv_plot) and "
                         "-no-plots switches the plots off")
    return units


# does the run fit the MPV? (mpv_csv or mpv_plot chosen)
def needs_mpv(units: dict) -> bool:
    return bool(units["mpv_csv"] or units["mpv_plot"])


# --------------------------------------------------------------------------
# canonical products
# --------------------------------------------------------------------------
# path of the full table: calibration/[test/]pedestal_mpv/calibration_parameters.csv
def output_csv(ctx) -> Path:
    return ctx.calibration_product()


# path of a part table: calibration/[test/]pedestal_mpv/pedestal_parameters.csv or mpv_parameters.csv
def part_csv(ctx, name: str) -> Path:
    return ctx.calib_dir(STAGE_DIR) / (PEDESTAL_CSV if name == "pedestal" else MPV_CSV)


# table of the chosen csv units: both -> the full table, one -> its part table, none -> None
def table_csv(ctx, units: dict | None = None) -> Path | None:
    units = units or chosen_units(ctx)
    if units["pedestal_csv"] and units["mpv_csv"]:
        return output_csv(ctx)
    if units["pedestal_csv"] or units["mpv_csv"]:
        return part_csv(ctx, "pedestal" if units["pedestal_csv"] else "mpv")
    return None


# columns of the table of the chosen csv units (keys first, CALIBRATION_COLUMNS order)
def table_columns(units: dict) -> list[str]:
    if units["pedestal_csv"] and units["mpv_csv"]:
        return KEY_COLUMNS + list(CALIBRATION_COLUMNS)
    part = PEDESTAL_COLUMNS if units["pedestal_csv"] else MPV_COLUMNS
    return KEY_COLUMNS + [c for c in CALIBRATION_COLUMNS if c in part]


# provenance path next to a product: x.csv -> x.provenance.json
def provenance_json(path: Path) -> Path:
    return Path(path).with_suffix(".provenance.json")


# default path of pedestal_fits.pdf: results/<mode>/pedestal_mpv/
def pedestal_pdf(ctx) -> Path:
    return ctx.results_dir(STAGE_DIR) / "pedestal_fits.pdf"


# default path of mpv_fits.pdf: results/<mode>/pedestal_mpv/
def mpv_pdf(ctx) -> Path:
    return ctx.results_dir(STAGE_DIR) / "mpv_fits.pdf"


# path of the raw-histogram cache (input of -from-saved)
def histogram_cache(ctx) -> Path:
    return ctx.cache_dir(STAGE_DIR) / settings_for(ctx)["histogram_cache"]


# --------------------------------------------------------------------------
# final results (common/outputs.py): the two pdfs; the tables and the cache are intermediates
# --------------------------------------------------------------------------
RESULT_PARTS = ("pedestal", "mpv")      # the order in which every calib run writes its pdfs
RESULT_KEY = {"pedestal": "pedestal_plot", "mpv": "mpv_plot"}   # result key = -only unit
RESULT_WHAT = {"pedestal": "summary maps + the pedestal window of every channel",
               "mpv": "summary maps + the final Landau (x) Gauss MPV fit of every channel"}


# parts whose pdf the chosen units draw, in the order the run writes them
def drawn_parts(units: dict) -> list[str]:
    return [n for n in RESULT_PARTS if units[RESULT_KEY[n]]]


# final results of a run (for -output): pedestal_fits.pdf, then mpv_fits.pdf
def result_slots(ctx, parts) -> list[outputs.ResultSlot]:
    defaults = {"pedestal": pedestal_pdf(ctx), "mpv": mpv_pdf(ctx)}
    return [outputs.ResultSlot(RESULT_KEY[n], "pdf", defaults[n], RESULT_WHAT[n]) for n in RESULT_PARTS if n in parts]


# output plan of the run: -output names, test names or the unsaved scratch copy
def result_plan(ctx, parts, preview: bool = False) -> outputs.OutputPlan:
    try:
        return outputs.OutputPlan(ctx, STAGE_DIR, result_slots(ctx, parts), preview=preview)
    except ValueError as exc:   # an --output name given twice: two results would be written to one file
        raise io_utils.SettingsError(f"calib: {exc}") from None


# log where the final results went (or that they were not saved)
def report_results(plan: outputs.OutputPlan) -> None:
    unsaved = [s for s in plan.slots if not plan.saved(s.key)]
    if unsaved:
        why = "not named in --output" if plan.names else "no --output"
        log(f"final results not saved ({why}): scratch copies in {plan.path(unsaved[0].key).parent}; use "
            "--output [names] to save them")
    for slot, line in zip(plan.slots, plan.summary_lines()):
        if plan.saved(slot.key):
            log(f"final result {line}")


# dry-run / check items of the final results at their planned destinations
def _result_items(ctx, parts) -> list[dict]:
    slots = result_slots(ctx, parts)
    if not slots:
        return []
    plan = result_plan(ctx, parts, preview=True)
    for w in plan.warnings:
        log(f"WARNING: {w}")
    return [{"role": "out", "path": plan.path(s.key), "result": s.key, "saved": plan.saved(s.key),
             "note": f"final result {line}" if plan.saved(s.key) else "not saved: no --output (scratch copy)"}
            for s, line in zip(slots, plan.summary_lines())]


# --------------------------------------------------------------------------
# CLI and settings
# --------------------------------------------------------------------------
# calib command-line options
def add_arguments(p) -> None:
    outputs.add_only_argument(p, UNITS, UNIT_GROUPS)
    p.add_argument("--pedestal-selection", "-pedestal-selection", default=None,
                   choices=(SELECTION_RAW, SELECTION_CSV),
                   help="pedestal events: raw_samples (default: raw TOT/TOA sums of both ends == 0 = the selection "
                        "on the ROOT digis; needs the raw_*_sum columns make-csv writes by default, else the run "
                        "stops) or csv_tot_toa_zero (explicit approximation from the csv tot/toa columns)")
    p.add_argument("--mpv-nsigma", "-mpv-nsigma", type=float, default=None, metavar="N",
                   help="MPV fits use x = adc_sum - pedestal > N * pedestal_sigma (default 5)")
    p.add_argument("--mpv-window-widths", "-mpv-window-widths", type=float, default=None, metavar="K",
                   help="final fit on bin centres in peak +- K * FWHM/2.3548 of the rough fit (default 2)")
    p.add_argument("--mpv-bin-width", "-mpv-bin-width", type=float, default=None, metavar="ADC",
                   help="bin width of the final MPV fit (default 20; must divide the histogram range into "
                        "whole bins)")
    p.add_argument("--pedestal-window", "-pedestal-window", type=float, default=None, metavar="ADC",
                   help="half width of the open pedestal window around the anchor (default 200)")
    # Lei 2026-10-02: every fit gets a range option; the MPV model stays Landau (x) Gauss (one range only)
    p.add_argument("--fit", "-fit", dest="fit_ranges", type=mpv_fit_range_arg, default=None, metavar="[a,b]",
                   help="final MPV fit window [a, b] in x = adc_sum - pedestal [ADC] (the x axis of mpv_fits.pdf), "
                        "the same for EVERY channel, instead of the automatic window peak +- K w of the rough fit; the "
                        "rough fit still gives the start values (the histogram inside [a, b] when it fails); the model "
                        "stays Landau (x) Gauss, so one range only; [a, b] inside mpv_hist_range [0, 2000]; with "
                        "-from-saved it refits from the cached histograms (default: automatic window)")
    p.add_argument("--bin", "-bin", dest="fit_bins", type=fitting.parse_bins, default=None, metavar="N",
                   help="with --fit: N equal bins on [a, b] for the final fit (>= mpv_min_bins, at least 1 ADC wide: "
                        "adc_sum is an integer) (default: the mpv_final_bin_width bins of mpv_fits.pdf whose centres "
                        "lie in [a, b])")


CHOICES = {"mpv_rough_likelihood": ("neyman", "poisson"), "mpv_rough_start": ("valley", "threshold"),
           "pedestal_selection": (SELECTION_RAW, SELECTION_CSV),
           "pedestal_anchor": ("auto", ANCHOR_MODE, ANCHOR_MEDIAN)}
BOOLEANS = ("mpv_window_above_valley", "mpv_require_rough_convergence", "write_histogram_cache")
_TRUE, _FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


# yml / command-line boolean -> True / False (None if empty)
def _as_bool(value) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE + _FALSE:
        return value.strip().lower() in _TRUE
    return None


# --fit[a,b] / --bin N (Lei 2026-10-02): the final MPV fit window chosen by the user.  Settings keys fit_ranges
# ([[a, b]] or None) and fit_bins ([N] or None); command line (or the calibration section of the yml), not in
# DEFAULTS: None = the automatic window.  They change the MPV values, so they are fit settings (fit_settings: the
# workers, the histogram cache, the table provenance and its settings_hash).
FIT_WINDOW_KEYS = ("fit_ranges", "fit_bins")


# argparse type of -fit: exactly one range [a, b]
def mpv_fit_range_arg(text) -> list[tuple[float, float]]:
    ranges = fitting.parse_ranges(text, max_ranges=99, flag="--fit")
    if len(ranges) != 1:
        raise argparse.ArgumentTypeError(
            f"--fit {text!r}: the MPV fit is a Landau(x)Gauss fit: one range, --fit[a,b] with a, b in "
            f"x = adc_sum - pedestal [ADC] (got {len(ranges)} ranges; two ranges = double Gaussian, which belongs to "
            "the Gaussian fits of other stages)")
    return ranges


# range / bin setting as text for fitting.parse_ranges / parse_bins
def _setting_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=float)


# setting not given?
def _unset(value) -> bool:
    return value is None or (isinstance(value, (str, list, tuple)) and len(value) == 0)


# -fit / -bin rules; normalises fit_ranges / fit_bins
def _check_fit_window(s: dict, bad, hist_range, final_bw, min_bins) -> None:
    raw_ranges, raw_bins = s.get("fit_ranges"), s.get("fit_bins")
    ranges = bins = None
    if not _unset(raw_ranges):
        try:
            ranges = [tuple(map(float, r)) for r in mpv_fit_range_arg(_setting_text(raw_ranges))]
        except argparse.ArgumentTypeError as exc:
            bad("fit_ranges", raw_ranges, str(exc))
    if not _unset(raw_bins):
        try:
            bins = fitting.parse_bins(_setting_text(raw_bins))
        except argparse.ArgumentTypeError as exc:
            bad("fit_bins", raw_bins, str(exc))
    if bins and (ranges or _unset(raw_ranges)):
        # Landau(x)Gauss: mpv_min_bins is checked below
        why = fitting.range_options_error(ranges or [], bins, gauss_min_bins=False)
        if why:
            bad("fit_bins", raw_bins, why)
            bins = None
    if ranges:
        a, b = ranges[0]
        if hist_range is not None and not (hist_range[0] <= a and b <= hist_range[1]):
            bad("fit_ranges", raw_ranges, f"--fit [{a:g}, {b:g}] must lie inside mpv_hist_range [{hist_range[0]:g}, "
                                          f"{hist_range[1]:g}] ADC: the MPV histograms (x = adc_sum - pedestal) "
                                          "end there")
        if bins:
            n, width = bins[0], (b - a) / bins[0]
            if min_bins is not None and n < min_bins:
                bad("fit_bins", raw_bins, f"--bin {n}: the Landau(x)Gauss fit (4 parameters) needs at least "
                                          f"mpv_min_bins = {min_bins} bins")
            if width < 1.0:
                bad("fit_bins", raw_bins, f"--bin {n} on [{a:g}, {b:g}] gives {width:.3g}-ADC bins: adc_sum is an "
                                          f"integer, so bins narrower than 1 ADC are partly empty (at most "
                                          f"{int(np.floor(b - a))} bins on this range)")
        elif hist_range is not None and final_bw is not None and min_bins is not None:
            edges = np.arange(hist_range[0], hist_range[1] + 0.5 * final_bw, final_bw)
            inside = int(fitting.in_range(0.5 * (edges[:-1] + edges[1:]), (a, b)).sum())
            if inside < min_bins:
                bad("fit_ranges", raw_ranges, f"--fit [{a:g}, {b:g}] holds {inside} bin(s) of mpv_final_bin_width = "
                                              f"{final_bw:g} ADC (centres inside); the fit needs at least mpv_min_bins "
                                              f"= {min_bins}: widen the range or give --bin N")
    s["fit_ranges"] = [list(r) for r in ranges] if ranges else None
    s["fit_bins"] = [int(n) for n in bins] if bins else None


# refuse -fit / -bin when the run fits no MPV (before any IO)
def check_fit_options(ctx, args) -> None:
    given = [flag for flag, key in (("--fit", "fit_ranges"), ("--bin", "fit_bins")) if getattr(args, key, None)]
    if given and not needs_mpv(chosen_units(ctx)):
        raise io_utils.SettingsError(f"calib: {' / '.join(given)} set the final MPV fit window, but this run fits no "
                                     "MPV: -only chose no MPV unit (mpv_csv, mpv_plot; -no-plots drops mpv_plot)")


# text of the MPV fit window ('' = automatic)
def fit_window_text(settings: dict | None) -> str:
    ranges = (settings or {}).get("fit_ranges")
    if not ranges:
        return ""
    (a, b), = ranges
    bins = settings.get("fit_bins")
    if bins:
        n = int(bins[0])
        return f"--fit [{a:g}, {b:g}] ADC --bin {n}: {n} equal bins of {(b - a) / n:g} ADC"
    return f"--fit [{a:g}, {b:g}] ADC: the {float(settings['mpv_final_bin_width']):g}-ADC bins with centres inside"


# check every setting before any IO
def validate_settings(s: dict) -> dict:
    problems: list[str] = []

    # collect one problem
    def bad(key, value, rule):
        problems.append(f"calibration.{key} = {value!r}: {rule}")

    # positive number check
    def number(key, ok, rule):
        try:
            v = float(s[key])
        except (TypeError, ValueError, OverflowError):
            bad(key, s[key], "expected a number")
            return None
        if not (np.isfinite(v) and ok(v)):
            bad(key, s[key], rule)
            return None
        return v

    # positive integer check
    def integer(key, minimum):
        v = s[key]
        try:   # null, text, .inf and huge values all end in the clean SettingsError below
            f = float(v)
            valid = not isinstance(v, (bool, np.bool_)) and bool(np.isfinite(f)) and f == int(f) and int(f) >= minimum
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            bad(key, v, f"expected an integer >= {minimum}")
            return None
        return int(f)

    # ordered pair check
    def pair(key, positive=False):
        v = s[key]
        try:
            lo, hi = (float(t) for t in v)
        except (TypeError, ValueError, OverflowError):
            bad(key, v, "expected [low, high]")
            return None
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi and (lo > 0 or not positive)):
            bad(key, v, "expected finite [low, high] with low < high" + (" and low > 0" if positive else ""))
            return None
        return lo, hi

    for key, allowed in CHOICES.items():
        value = str(s[key]).lower()
        if value in allowed:
            s[key] = value
        else:
            bad(key, s[key], f"expected one of {allowed}")
    for key in BOOLEANS:
        value = _as_bool(s[key])
        if value is None:
            bad(key, s[key], "expected true or false")
        else:
            s[key] = value
    number("pedestal_window_halfwidth", lambda v: v > 0, "must be > 0 ADC")
    number("pedestal_range_nsigma", lambda v: v >= 0, "must be >= 0")
    number("pedestal_range_fraction", lambda v: v > -1, "must be > -1")
    number("mpv_nsigma", lambda v: v >= 0, "must be >= 0")
    hist_range = pair("mpv_hist_range")
    widths = {}
    for key in ("mpv_rough_bin_width", "mpv_final_bin_width"):
        if hist_range is None:
            number(key, lambda v: v > 0, "must be > 0")
            continue
        lo, hi = hist_range
        width = hi - lo
        bw = number(key, lambda v: 0 < v <= width, f"must be > 0 and <= the histogram range {width:g} ADC")
        if bw is not None:
            n_bins = width / bw
            if abs(n_bins - round(n_bins)) > 1e-9 * max(1.0, n_bins):
                bad(key, s[key], f"must divide mpv_hist_range [{lo:g}, {hi:g}] into whole bins ({width:g} / "
                                 f"{bw:g} = {n_bins:.4g}): the last bin would end outside the histogram range")
            else:
                widths[key] = bw
    number("mpv_window_widths", lambda v: v > 0, "must be > 0")
    integer("mpv_min_entries", 0)
    min_bins = integer("mpv_min_bins", 5)       # 4 fit parameters: a fit needs more bins than parameters
    _check_fit_window(s, bad, hist_range, widths.get("mpv_final_bin_width"), min_bins)   # --fit / --bin
    pair("mpv_eta_range", positive=True)
    pair("mpv_sigma_range", positive=True)
    n = integer("langaus_n_steps", 2)
    if n is not None and n % 2:
        bad("langaus_n_steps", s["langaus_n_steps"], "expected a positive EVEN integer (symmetric midpoint grid)")
    number("langaus_n_sigma", lambda v: v > 0, "must be > 0")
    number("langaus_max_step_per_eta", lambda v: v >= 0, "must be >= 0 (0 = fixed number of steps)")
    if problems:
        raise io_utils.SettingsError("invalid calibration setting(s) (configure/config.yml section calibration, "
                                     "configure/local.yml or the command line):\n  " + "\n  ".join(problems))
    return s


# calib settings: defaults < yml section calibration < command line, checked
def settings_for(ctx, args=None) -> dict:
    s = {**DEFAULTS, **ctx.stage_cfg(CONFIG_SECTION)}
    if args is not None:
        cli = {"mpv_nsigma": getattr(args, "mpv_nsigma", None),
               "mpv_window_widths": getattr(args, "mpv_window_widths", None),
               "mpv_final_bin_width": getattr(args, "mpv_bin_width", None),
               "pedestal_window_halfwidth": getattr(args, "pedestal_window", None),
               "pedestal_selection": getattr(args, "pedestal_selection", None),
               "fit_ranges": getattr(args, "fit_ranges", None),        # --fit[a,b] (Lei 2026-10-02)
               "fit_bins": getattr(args, "fit_bins", None)}            # --bin N
        s.update({k: v for k, v in cli.items() if v is not None})
    return validate_settings(s)


# the settings an MPV fit depends on (sent to the workers, stored with the results)
def fit_settings(settings: dict) -> dict:
    return {k: v for k, v in settings.items() if k.startswith(("mpv_", "langaus_")) or k in FIT_WINDOW_KEYS}


# source run(s): the run flags, else calibration.source_dataset (any run is accepted)
def source_runs(ctx, datasets) -> tuple[list, str]:
    if datasets:
        runs = list(datasets)
        return runs, f"the run flag{'s' if len(runs) > 1 else ''} {' '.join(run_flags(d) for d in runs)}"
    value = settings_for(ctx).get("source_dataset")
    return resolve_runs(ctx.cfg, value, "calibration.source_dataset"), f"calibration.source_dataset = {value!r}"


# why the given runs cannot be the source (calib needs exactly one), else None
def _one_run_problem(runs, origin: str, from_flags: bool) -> str | None:
    if len(runs) == 1:
        return None
    if not runs:
        return (f"[calib] {origin} names no run: calib needs one source run; write a run reference there (e.g. "
                f"\"muon 4\") or choose the run with a run flag (.\\runner.ps1 calib --muon 4)")
    names = ", ".join(run_ref(d) for d in runs)
    if not from_flags:
        hint = f"write one of them there, e.g. \"{run_ref(runs[0])}\""
    elif len({run_flags(d) for d in runs}) == len(runs):
        hint = f"give one run flag, e.g. .\\runner.ps1 calib {run_flags(runs[0])}"
    else:   # several runs at one energy: a flag cannot tell them apart, a reference with its run number can
        hint = (f"a run flag cannot tell runs at one energy apart: give no run flag and write one of them in "
                f"calibration.source_dataset, e.g. \"{run_ref(runs[0])}\"")
    return (f"[calib] {origin} {'chose' if from_flags else 'names'} {len(runs)} runs ({names}); calib calibrates "
            f"on one run: {hint}")


# the one source run, or stop with the runs named
def source_dataset(ctx, datasets, report: bool = False):
    runs, origin = source_runs(ctx, datasets)
    problem = _one_run_problem(runs, origin, bool(datasets))
    if problem:
        raise SystemExit(problem)
    if report:
        log(f"source run of the pedestals and MIPs: {run_ref(runs[0])} (from {origin})")
    return runs[0]


# make-csv command that rewrites the csv files of the source run with the raw-sum columns
def _make_csv_hint(ctx, ds) -> str:
    return f".\\runner.ps1 make-csv {run_flags(ds)}{' -test' if ctx.test else ''} -overwrite"


# calib command that rereads the csv files and refreshes the histogram cache
def _rebuild_command(ctx, datasets=()) -> str:
    flags = "".join(flag for flag, on in ((" -test", ctx.test), (" -clean-input", ctx.clean_input)) if on)
    if ctx.units is not None and set(ctx.units) != set(UNITS):
        flags += f" -only {','.join(ctx.units)}"
    runs = "".join(f" {run_flags(d)}" for d in datasets or ())
    return f".\\runner.ps1 calib{runs}{flags}   (the same options without -from-saved)"


# a csv with the header of the beam csv of one plan item (nothing is built)
def _beam_header_csv(item: dict) -> tuple[Path | None, str]:
    path = Path(item["path"])
    if item.get("kind") == "auto":
        source = Path(item["source"])
        if source.exists():
            return source, f"{source.name} (source of the test subset {path.name})"
    return (path, path.name) if path.exists() else (None, "")


# plan item of one beam csv (STALE when the pedestal selection cannot use it)
def _pedestal_plan_item(ctx, ds, item: dict, settings: dict) -> dict:
    header, name = _beam_header_csv(item)
    if header is None:
        return item
    try:
        resolve_pedestal_methods(io_utils.csv_columns(header), settings["pedestal_selection"],
                                 settings["pedestal_anchor"], name)
    except (io_utils.InputError, OSError, ValueError) as exc:
        why = str(exc).splitlines()[0].rstrip(".")
        return {**item, "kind": "external" if item.get("kind") == "external" else "product", "stale": True,
                "produced_by": f"{_make_csv_hint(ctx, ds)}   (calib would stop: {why}; or -pedestal-selection "
                               f"{SELECTION_CSV})"}
    return item


# inputs and outputs of the run (for -dry-run, check and the run log)
def io_plan(ctx, args, datasets) -> list[dict]:
    check_fit_options(ctx, args)
    units = units_or_stop(ctx)
    runs, origin = source_runs(ctx, datasets)
    problem = _one_run_problem(runs, origin, bool(datasets))
    if problem and runs:   # the run stops on this (source_dataset); the plan shows why instead of stopping check
        items = [item for d in runs for item in ctx.beam_plan_items(d)]
        return [{**item, "kind": "external" if item.get("kind") == "external" else "product", "stale": True,
                 "produced_by": f"calib would stop: {problem}"} for item in items]
    if problem:   # no run at all: a configuration error
        raise SystemExit(problem)
    ds = runs[0]
    settings = settings_for(ctx, args)
    # the pedestal always comes from the beam csv files; a csv the strict selection cannot use is STALE
    plan = [_pedestal_plan_item(ctx, ds, item, settings) for item in ctx.beam_plan_items(ds)]
    inputs = {"pedestal": [Path(item["path"]) for item in plan]}
    if needs_mpv(units):   # the MPV: the beam csv files, or their clean datasets with --clean-input
        if ctx.clean_input:
            clean_items = ctx.input_plan_items(ds)
            plan += clean_items
            inputs["mpv"] = [Path(item["path"]) for item in clean_items]
        else:
            inputs["mpv"] = inputs["pedestal"]
    if getattr(args, "from_saved", False):   # the cached histograms replace the csv pass (shown as reuse / rebuild)
        plan.append({"role": "cache", "path": provenance_json(histogram_cache(ctx)),
                     "expected": _planned_cache_identity(inputs, settings), "members": [histogram_cache(ctx)]})
    elif settings["write_histogram_cache"]:
        plan += [{"role": "out", "path": histogram_cache(ctx)},
                 {"role": "out", "path": provenance_json(histogram_cache(ctx))}]
    table = table_csv(ctx, units)
    if table is not None:
        plan.append({"role": "out", "path": table})
        if ctx.write_json:
            plan.append({"role": "out", "path": provenance_json(table)})
    return plan + _result_items(ctx, drawn_parts(units))   # pedestal_fits.pdf before mpv_fits.pdf


# --------------------------------------------------------------------------
# inputs: which csv files, their identity, the converter's sample counts
# --------------------------------------------------------------------------
# csv files of the source run, or stop with the run folder and the make-csv command
def _run_csvs(ctx, ds) -> list[Path]:
    csvs = ctx.run_csvs(ds)
    if not csvs:
        raise SystemExit(f"[calib] run {run_ref(ds)}: no csv file in {ctx.run_dir(ds, mode=MODE_FULLSET)}; make "
                         f"them with .\\runner.ps1 make-csv {run_flags(ds)}")
    return csvs


# beam csv of one csv of the source run (test mode: the test csv or the first-N-events subset)
def _beam_input(ctx, ds, csv: Path, from_saved: bool) -> Path:
    produced_by = f".\\runner.ps1 make-csv {run_flags(ds)}"
    if not from_saved:
        return io_utils.require(ctx.beam_csv(ds, csv), produced_by=produced_by)
    item = ctx.beam_plan_item(ds, csv)
    path = Path(item["path"])
    if item.get("kind") == "auto":
        ok, reason = io_utils.subset_status(item["source"], path, item["n_events"])
        if not ok:
            raise io_utils.ProductSetError(
                f"-from-saved: the test subset {path} must be built or rebuilt first ({reason}); that reads the "
                f"full csv, so run once without -from-saved: .\\runner.ps1 calib -test")
    return io_utils.require(path, produced_by=produced_by)


# MPV input of one csv: its beam csv, or its clean dataset with -clean-input (-from-saved: the clean product,
# checked against the cache record)
def _mpv_input(ctx, ds, csv: Path, beam: Path, from_saved: bool) -> Path:
    if not ctx.clean_input:
        return beam
    if from_saved:
        return io_utils.require(ctx.clean_product(ds, csv),
                                produced_by=f".\\runner.ps1 clean {run_flags(ds)}" + (" -test" if ctx.test else ""))
    return ctx.input_csv(ds, csv)


# input csv files of each part (pedestal: beam csv; mpv: beam csv or clean dataset) + clean dataset -> its beam csv
def stage_inputs(ctx, ds, need_mpv: bool, from_saved: bool = False) -> tuple[dict[str, list[Path]], dict]:
    inputs: dict[str, list[Path]] = {"pedestal": []}
    if need_mpv:
        inputs["mpv"] = []
    upstream: dict[Path, Path] = {}
    for csv in _run_csvs(ctx, ds):
        beam = _beam_input(ctx, ds, csv, from_saved)
        inputs["pedestal"].append(beam)
        if need_mpv:
            path = _mpv_input(ctx, ds, csv, beam, from_saved)
            inputs["mpv"].append(path)
            if path != beam:
                upstream[path] = beam
    return inputs, upstream


# columns every csv of a list has (the pedestal methods must work on all of them)
def common_columns(paths) -> set[str]:
    sets = [set(io_utils.csv_columns(p)) for p in paths]
    return set.intersection(*sets) if sets else set()


# names of csv files for messages
def _names(paths) -> str:
    return ", ".join(Path(p).name for p in paths)


# identity of an input csv: path, size, mtime, content hash
def input_identity(path: Path) -> dict:
    return {"path": io_utils.canonical_path(path), **io_utils.member_signature(path)}


# what each part's histograms depend on (input csv files, selection, anchor)
def histogram_identities(inputs: dict[str, list[Path]], selection: str = "", anchor: str = "") -> dict[str, dict]:
    out = {}
    if "pedestal" in inputs:
        out["pedestal"] = {"inputs": [input_identity(p) for p in inputs["pedestal"]], "selection": selection,
                           "anchor": anchor}
    if "mpv" in inputs:
        out["mpv"] = {"inputs": [input_identity(p) for p in inputs["mpv"]]}
    return out


# the identities as the top-level keys of the cache record
def flat_identity(identities: dict[str, dict]) -> dict:
    return {f"{name}_histograms_{field}": value for name, ident in identities.items() for field, value in ident.items()}


# expected cache identity for the dry-run plan
def _planned_cache_identity(inputs: dict[str, list[Path]], settings: dict) -> dict:
    expected = dict(CACHE_IDENTITY)
    for name, paths in inputs.items():
        missing = [p for p in paths if not p.exists()]
        if missing or not paths:
            expected[f"{name}_histograms_inputs"] = f"not yet made: {missing[0] if missing else 'no csv'}"
            continue
        selection = anchor = ""
        if name == "pedestal":
            try:
                selection, anchor = resolve_pedestal_methods(common_columns(paths), settings["pedestal_selection"],
                                                             settings["pedestal_anchor"], csv_name=_names(paths))
            except SystemExit as exc:
                selection = anchor = f"not usable: {str(exc).splitlines()[0]}"
        expected.update(flat_identity(histogram_identities({name: paths}, selection, anchor)))
    return expected


SAMPLE_FIELDS = ("n_samples", "n_samples_used", "waveform")


# converter facts recorded for the csv a product was made from
def _recorded_input_facts(record: dict, upstream: Path | None = None) -> tuple[dict | None, str]:
    src = record.get("input")
    if not isinstance(src, dict) or not src.get("path"):
        return None, ""
    path = Path(upstream) if upstream is not None else Path(src["path"])
    try:
        if not path.exists():
            return None, ""
        if src.get("kind") == "test-subset":   # the subset's own record must name the same source
            if (io_utils.read_json(provenance_json(path)) or {}).get("source") != src.get("source"):
                return None, ""
        elif path.stat().st_size != src.get("size") or io_utils.content_hash(path)[0] != src.get("sha256"):
            return None, ""
    except OSError:
        return None, ""
    return _converter_facts(path, follow=False)


# converter facts (samples summed into adc_sum) that describe a csv
def _converter_facts(csv_path: Path, follow: bool = True, upstream: Path | None = None) -> tuple[dict | None, str]:
    csv_path = Path(csv_path)
    try:
        conv = io_utils.source_provenance(csv_path)
    except OSError:
        conv = None
    if conv is not None and conv.get("matches_source"):
        return {k: conv[k] for k in SAMPLE_FIELDS if k in conv}, "make-csv provenance"
    record_path = provenance_json(csv_path)
    record = io_utils.read_json(record_path)
    if isinstance(record, dict) and record.get("protocol") == io_utils.PRODUCT_SET_PROTOCOL:
        ok, _ = io_utils.product_set_valid(record_path, None, members=[csv_path])
        if ok:
            facts = {k: record[k] for k in SAMPLE_FIELDS if k in record}
            source = record.get("source_provenance")
            if not facts and isinstance(source, dict) and source.get("matches_source"):
                facts = {k: source[k] for k in SAMPLE_FIELDS if k in source}
            origin = f"{record.get('stage') or 'product'} record {record_path.name}"
            if not facts and follow:
                facts_up, origin_up = _recorded_input_facts(record, upstream)
                if facts_up is not None:
                    return facts_up, f"{origin} -> {origin_up}"
            return facts, origin
    if conv is not None:
        return None, f"{record_path.name} describes another version of the file"
    return None, "no converter provenance"


# distinct sample counts of a provenance field
def _distinct_counts(value) -> list[int] | None:
    values = list(value.values()) if isinstance(value, dict) else list(value) if isinstance(value, (list, tuple)) \
        else [value]
    out = set()
    for v in values:
        if v is None or isinstance(v, (bool, np.bool_)):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f) or f != int(f):
            return None
        out.add(int(f))
    return sorted(out) or None


# check that adc_sum is a sum of 8 time samples (warning otherwise)
def input_samples_note(csv_path: Path, upstream: Path | None = None) -> dict:
    csv_path = Path(csv_path)
    facts, origin = _converter_facts(csv_path, upstream=upstream)
    facts = facts or {}
    used = _distinct_counts(facts["n_samples_used"]) if "n_samples_used" in facts else None
    want = geo.N_TIME_SAMPLES
    note = {"csv": str(csv_path), "record": origin, **{k: facts.get(k) for k in SAMPLE_FIELDS}}
    assumes = (f"the pedestal anchor (8 x mode), pedestal_per_time_sample = pedestal / 8, the adc_sum range and the "
               f"MPV definitions assume {want} (data)")
    if used is None:
        note["status"] = "unknown"
        log(f"{csv_path.name}: number of time samples summed into adc_sum unknown ({origin}); {assumes}")
    elif len(used) > 1:
        note["status"] = "mixed"
        log(f"WARNING: {csv_path.name} sums different numbers of time samples per input file "
            f"(n_samples_used {facts['n_samples_used']}); {assumes}.  Check the result.")
    elif used[0] != want:
        note["status"] = "mismatch"
        log(f"WARNING: {csv_path.name} sums {used[0]} time samples per digi into adc_sum (make-csv n_samples_used; "
            f"raw samples per digi: {facts.get('n_samples')}); {assumes}.  Check the result.")
    else:
        note["status"] = "ok"
    return note


# --------------------------------------------------------------------------
# single pass: integer histograms per channel
# --------------------------------------------------------------------------
# lookup [layer, strip, end] -> channel index 0..383
def channel_index_table() -> np.ndarray:
    ch = geo.all_channels()
    table = np.full((geo.N_LAYERS + 1, geo.N_STRIPS_WIDE, 2), -1, dtype=np.int64)
    table[ch["layer"], ch["strip"], ch["end"]] = np.arange(len(ch))
    return table


# no-signal selection and anchor method for this csv
def resolve_pedestal_methods(columns, selection: str = SELECTION_RAW, anchor: str = "auto",
                             csv_name: str = "the beam csv", make_csv_hint: str | None = None) -> tuple[str, str]:
    """raw_samples needs the raw-sum columns, else stop; auto anchor = sample_mode (waveform csv) or median_sum."""
    columns = set(columns)
    if selection not in (SELECTION_RAW, SELECTION_CSV):
        raise io_utils.SettingsError(f"pedestal_selection = {selection!r}: expected {SELECTION_RAW} or {SELECTION_CSV}")
    if selection == SELECTION_RAW and not all(c in columns for c in RAW_FLAG_COLUMNS):
        hint = make_csv_hint or ".\\runner.ps1 make-csv --<particle> E"
        raise io_utils.InputError(
            f"{csv_name} has no raw-sum columns {RAW_FLAG_COLUMNS}: the default pedestal selection "
            f"{SELECTION_RAW} (raw TOT and TOA sums of both ends == 0, the selection on the ROOT digis) needs them, "
            "and calib does not fall back to the approximation on its own.\n"
            f"  Regenerate the csv with make-csv (it always writes these columns): {hint}\n"
            f"  or ask for the approximation explicitly: -pedestal-selection {SELECTION_CSV}")
    has_samples = all(c in columns for c in SAMPLE_COLUMNS)
    if anchor == "auto":
        anchor = ANCHOR_MODE if has_samples else ANCHOR_MEDIAN
    if anchor == ANCHOR_MODE and not has_samples:
        raise io_utils.InputError("pedestal_anchor = sample_mode needs the per-sample columns adc_0_end0 .. "
                                  "adc_7_end1 (make the csv with make-csv -waveform)")
    return selection, anchor


# read the csv once: per-channel adc_sum histograms of no-signal events and of all events
def accumulate_histograms(csv_path: Path, need_pedestal: bool, need_all: bool,
                          chunksize: int = io_utils.DEFAULT_CHUNKSIZE, selection: str = SELECTION_RAW,
                          anchor: str = "auto", make_csv_hint: str | None = None) -> dict:
    """No signal = raw TOT and TOA sums of both ends are 0 (raw_samples)."""
    columns = io_utils.csv_columns(csv_path)
    selection, anchor = (resolve_pedestal_methods(columns, selection, anchor, Path(csv_path).name, make_csv_hint)
                         if need_pedestal else ("", ""))
    has_samples = need_pedestal and anchor == ANCHOR_MODE
    flag_columns = RAW_FLAG_COLUMNS if selection == SELECTION_RAW else CSV_FLAG_COLUMNS
    usecols = ["layer", "strip", "adc_sum_end0", "adc_sum_end1"]
    if need_pedestal:
        usecols += flag_columns + (SAMPLE_COLUMNS if has_samples else [])
    table = channel_index_table()
    size = N_CHANNELS * N_ADC_SUM
    h_sel = np.zeros(size, np.int64) if need_pedestal else None
    h_all = np.zeros(size, np.int64) if need_all else None
    h_sample = np.zeros(N_CHANNELS * N_ADC_SAMPLE, np.int64) if has_samples else None
    n_rows = n_bad = 0
    try:
        for chunk in io_utils.iter_csv(csv_path, usecols, chunksize):
            layer = chunk["layer"].to_numpy(np.int64)       # validated by iter_csv: always a real channel
            strip = chunk["strip"].to_numpy(np.int64)
            if need_pedestal:
                no_signal = np.ones(len(chunk), bool)
                for c in flag_columns:
                    no_signal &= chunk[c].to_numpy() == 0
            for end in geo.ENDS:
                adc_sum = chunk[f"adc_sum_end{end}"].to_numpy(np.int64)
                chan = table[layer, strip, end]
                good = (adc_sum >= 0) & (adc_sum < N_ADC_SUM)
                n_bad += int((~good).sum())
                flat = chan * N_ADC_SUM + adc_sum
                if need_all:
                    h_all += np.bincount(flat[good], minlength=size)
                if need_pedestal:
                    keep = good & no_signal
                    h_sel += np.bincount(flat[keep], minlength=size)
                    if has_samples:
                        for i in range(geo.N_TIME_SAMPLES):
                            adc = chunk[f"adc_{i}_end{end}"].to_numpy(np.int64)
                            ok = keep & (adc >= 0) & (adc < N_ADC_SAMPLE)
                            h_sample += np.bincount(chan[ok] * N_ADC_SAMPLE + adc[ok], minlength=h_sample.size)
            n_rows += len(chunk)
    except io_utils.InputError:
        raise
    except ValueError as exc:   # pandas: a non-integer / empty value in an integer column
        raise io_utils.InputError(f"{csv_path}: the columns {usecols} must hold integers (the histograms count "
                                  f"integer ADC values; nothing is truncated): {exc}") from exc
    if n_bad:
        log(f"warning: {n_bad} (row, end) entries with adc_sum outside 0..{N_ADC_SUM - 1} were ignored")
    return {
        "h_selected": None if h_sel is None else h_sel.reshape(N_CHANNELS, N_ADC_SUM),
        "h_all": None if h_all is None else h_all.reshape(N_CHANNELS, N_ADC_SUM),
        "h_sample": None if h_sample is None else h_sample.reshape(N_CHANNELS, N_ADC_SAMPLE),
        "n_rows": n_rows,
        "pedestal_selection": selection,
        "pedestal_anchor_method": anchor,
    }


# add the histograms of one csv to the sums of the run
def _add_histograms(total: dict, h: dict, keys) -> None:
    for k in keys:
        if h.get(k) is not None:
            total[k] = h[k] if total.get(k) is None else total[k] + h[k]


# raw histograms of this run, summed over its csv files (one pass per csv when pedestal and MPV read it)
def read_histograms(inputs: dict[str, list[Path]], chunksize: int, selection: str, anchor: str,
                    make_csv_hint: str | None = None) -> dict:
    peds, mpvs = inputs["pedestal"], inputs.get("mpv")
    shared = mpvs is not None and list(mpvs) == list(peds)
    out = {"h_selected": None, "h_all": None, "h_sample": None, "pedestal_selection": selection,
           "pedestal_anchor_method": anchor, "n_rows_pedestal": 0}
    for path in peds:
        with io_utils.timed(f"reading {path.name} ({'one pass' if shared else 'pedestal histograms'})"):
            h = accumulate_histograms(path, True, shared, chunksize, selection, anchor, make_csv_hint)
        _add_histograms(out, h, ("h_selected", "h_sample", "h_all"))
        out["n_rows_pedestal"] += h["n_rows"]
    if shared:
        out["n_rows_mpv"] = out["n_rows_pedestal"]
    elif mpvs is not None:
        out["n_rows_mpv"] = 0
        for path in mpvs:
            with io_utils.timed(f"reading {path.name} (MPV histograms)"):
                h = accumulate_histograms(path, False, True, chunksize)
            _add_histograms(out, h, ("h_all",))
            out["n_rows_mpv"] += h["n_rows"]
    out["n_rows"] = out["n_rows_pedestal"]
    return out


# --------------------------------------------------------------------------
# pedestal (vectorized over the 384 channels)
# --------------------------------------------------------------------------
# exact median of the values behind each histogram row
def histogram_median(hist: np.ndarray) -> np.ndarray:
    n = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    lo = (cum > ((n - 1) // 2)[:, None]).argmax(axis=1)   # 0-based sorted positions (n-1)//2 and n//2
    hi = (cum > (n // 2)[:, None]).argmax(axis=1)
    return np.where(n > 0, 0.5 * (lo + hi), np.nan)


# pedestal calculation (based on Joe Muse, revised by Lei): pedestal = anchor + mean, pedestal_sigma = std of adc_sum
# in |adc_sum - anchor| < 200
def pedestals_from_histograms(h_selected: np.ndarray, h_sample: np.ndarray | None, settings: dict) -> pd.DataFrame:
    """Gaussian MLE in closed form (no curve fit)."""
    halfwidth = float(settings["pedestal_window_halfwidth"])
    n_selected = h_selected.sum(axis=1)
    if h_sample is not None:
        sample_mode = h_sample.argmax(axis=1).astype(float)        # first maximum = scipy.stats.mode tie-break
        sample_mode[h_sample.sum(axis=1) == 0] = np.nan
        anchor = geo.N_TIME_SAMPLES * sample_mode  # anchor = 8 x single-sample mode
    else:
        # anchor = 8 x round(median(adc_sum) / 8)
        anchor = geo.N_TIME_SAMPLES * np.round(histogram_median(h_selected) / geo.N_TIME_SAMPLES)
    values = np.arange(N_ADC_SUM, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = values[None, :] - anchor[:, None]
        weight = np.where(np.abs(x) < halfwidth, h_selected, 0).astype(float)   # |adc_sum - anchor| < 200
        n_window = weight.sum(axis=1)
        mu = np.nansum(weight * x, axis=1) / n_window  # mean
        sigma = np.sqrt(np.nansum(weight * (x - mu[:, None]) ** 2, axis=1) / n_window)  # std
    mu[n_window == 0] = np.nan
    sigma[n_window == 0] = np.nan
    pedestal = anchor + mu  # pedestal = anchor + mean
    df = geo.all_channels()
    df["pedestal"] = pedestal
    df["pedestal_per_time_sample"] = sample_mode if h_sample is not None else pedestal / geo.N_TIME_SAMPLES
    df["pedestal_sigma"] = sigma
    df["pedestal_mu"] = mu
    df["pedestal_range_with_5sigma"] = pedestal + float(settings["pedestal_range_nsigma"]) * sigma  # pedestal + 5 sigma
    df["pedestal_range_with_20percent"] = (1.0 + float(settings["pedestal_range_fraction"])) * pedestal  # 1.2 pedestal
    df["n_pedestal_events"] = n_window.astype(np.int64)
    df["pedestal_anchor"] = anchor
    df["pedestal_anchor_method"] = ANCHOR_MODE if h_sample is not None else ANCHOR_MEDIAN
    df["n_selected_events"] = n_selected
    return df


# --------------------------------------------------------------------------
# MIP MPV: two-stage Landau (x) Gauss fit of one channel
# --------------------------------------------------------------------------
LANGAUS_MAX_STEPS = 4000   # cap of the adaptive convolution grid (reached only for eta near its lower bound)


# Landau (x) Gauss convolution settings
def langaus_config(settings: dict) -> tuple[int, float, float]:
    return (int(settings["langaus_n_steps"]), float(settings["langaus_n_sigma"]),
            float(settings["langaus_max_step_per_eta"]))


# convolution steps so that the step is at most eta / 2
def langaus_steps(eta: float, sigma: float, cfg: tuple[int, float, float]) -> int:
    n_min, n_sigma, step_per_eta = cfg
    n_min = int(n_min) + int(n_min) % 2
    if step_per_eta <= 0:
        return n_min
    n = int(np.ceil(2.0 * n_sigma * abs(sigma) / (step_per_eta * max(abs(eta), 1e-9))))
    n = min(max(n, n_min), LANGAUS_MAX_STEPS)
    return int(n + n % 2)


# Landau (x) Gauss curve (ROOT langaus.C parameterisation)
def langaus_curve(x, mpv_landau, eta, area, sigma, cfg):
    return fitting.langaus(x, mpv_landau, eta, area, sigma, n_steps=langaus_steps(eta, sigma, cfg), n_sigma=cfg[1])


# expected counts per bin of the Landau (x) Gauss model
def binned_langaus(bin_width: float, cfg: tuple[int, float, float]):
    # model(x, *params)
    def model(centers, mpv_landau, eta, area, sigma):
        return bin_width * langaus_curve(centers, mpv_landau, eta, area, sigma, cfg)
    return model


# peak position of the Landau (x) Gauss curve (grid search + Brent)
def langaus_peak(mpv_landau: float, eta: float, sigma: float, cfg: tuple[int, float, float],
                 near: float | None = None) -> float:
    # curve at one point
    def curve(t):
        return langaus_curve(t, mpv_landau, eta, 1.0, sigma, cfg)

    span = abs(eta) + abs(sigma)
    if near is None:
        grid = np.linspace(mpv_landau - 2 * span, mpv_landau + 5 * span, 351)
    else:
        grid = np.linspace(near - 0.2 * span, near + 0.2 * span, 41)
    i = int(np.clip(np.argmax(curve(grid)), 1, len(grid) - 2))
    res = minimize_scalar(lambda t: -curve(t)[0], bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


# Poisson deviance of a model on binned counts
def poisson_deviance(model, centers, counts, params) -> float:
    mu = np.clip(model(centers, *params), 1e-12, None)
    term = np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(mu - counts + term))


# parameter covariance from the deviance Hessian (NaN unless positive definite)
def deviance_covariance_with_status(model, centers, counts, params, free=None) -> tuple[np.ndarray, str]:
    p = np.asarray(params, dtype=float)
    k = len(p)
    free = np.ones(k, bool) if free is None else np.asarray(free, bool)
    idx = np.where(free)[0]
    cov = np.full((k, k), np.nan)
    if len(idx) == 0:
        return cov, "NO_FREE_PARAMETERS"
    h = 1e-4 * np.maximum(np.abs(p), 1e-3)

    # deviance at a parameter point
    def f(shift):
        return poisson_deviance(model, centers, counts, p + shift)

    f0 = f(np.zeros(k))
    hess = np.zeros((len(idx), len(idx)))
    for a, i in enumerate(idx):
        ei = np.zeros(k)
        ei[i] = h[i]
        hess[a, a] = (f(ei) - 2 * f0 + f(-ei)) / h[i] ** 2
        for b in range(a + 1, len(idx)):
            j = idx[b]
            ej = np.zeros(k)
            ej[j] = h[j]
            hess[a, b] = hess[b, a] = (f(ei + ej) - f(ei - ej) - f(ej - ei) + f(-ei - ej)) / (4 * h[i] * h[j])
    half = 0.5 * hess
    status = fitting.positive_definite_status(half)
    if status != "OK":
        return cov, status
    try:
        sub = np.linalg.inv(half)
    except np.linalg.LinAlgError:
        return cov, "SINGULAR"
    if not np.all(np.isfinite(sub)) or np.any(np.diag(sub) <= 0):
        return cov, "NOT_POSITIVE_DEFINITE"
    cov[:] = 0.0
    cov[np.ix_(idx, idx)] = sub
    return cov, "OK"


# covariance only
def deviance_covariance(model, centers, counts, params, free=None) -> np.ndarray:
    return deviance_covariance_with_status(model, centers, counts, params, free)[0]


# binned Poisson likelihood fit on scaled parameters (restarted once if not converged)
def fit_poisson_scaled(model, centers, counts, p0, lower, upper) -> fitting.FitResult:
    scale = np.where(np.asarray(p0) != 0, np.abs(p0), 1.0)

    # model in scaled parameters
    def scaled(c, *q):
        return model(c, *(np.asarray(q) * scale))

    q0 = np.asarray(p0, float) / scale
    bounds = (np.asarray(lower, float) / scale, np.asarray(upper, float) / scale)
    # the parameter errors of fit_binned_poisson are not used here (the MPV error comes from
    # deviance_covariance): skipping their Hessian leaves the fitted parameters identical
    res = fitting.fit_binned_poisson(scaled, centers, counts, q0, bounds, compute_errors=False)
    if not res.ok and np.all(np.isfinite(res.params)):
        again = fitting.fit_binned_poisson(scaled, centers, counts, res.params, bounds, compute_errors=False)
        if np.all(np.isfinite(again.params)) and again.chi2 <= res.chi2 + 1e-9:
            again.message = f"{again.message} (after a restart)"
            res = again
    res.params = res.params * scale
    res.errors = res.errors * scale
    return res


# chi2 fit with sqrt(n) errors on non-empty bins (the rough fit)
def fit_neyman(model, centers, counts, p0, lower, upper) -> fitting.FitResult:
    nonzero = counts > 0
    return fitting.fit_curve(model, centers[nonzero], counts[nonzero], p0, sigma=np.sqrt(counts[nonzero]),
                             bounds=(lower, upper))


# CONVERGED / NOT_CONVERGED / FAILED of a fit
def fit_status(res: fitting.FitResult) -> str:
    if not np.all(np.isfinite(res.params)):
        return "FAILED"
    return "CONVERGED" if res.ok else "NOT_CONVERGED"


# 3-bin running mean
def smoothed(counts: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(counts, float), 1, mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


# bin of the MIP peak: highest interior local maximum of the smoothed spectrum
def mip_peak_bin(counts: np.ndarray) -> int:
    s = smoothed(counts)
    interior = np.where((s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] > 0))[0] + 1
    return int(interior[np.argmax(s[interior])]) if len(interior) else int(np.argmax(s))


# lowest bin between the noise shoulder and the MIP peak
def valley_bin(counts: np.ndarray, i_peak: int) -> int:
    s = smoothed(counts)
    return int(np.argmin(s[:i_peak + 1]))


# start values: peak bin, eta ~ FWHM / 8, sigma ~ FWHM / 4, area ~ entries
def start_values(centers, counts, bin_width, lower, upper) -> np.ndarray:
    s = smoothed(counts)
    i_peak = mip_peak_bin(counts)
    half = s[i_peak] / 2.0
    left = np.where(s[:i_peak] < half)[0]
    right = np.where(s[i_peak:] < half)[0]
    x_left = centers[left[-1]] if len(left) else centers[0]
    x_right = centers[i_peak + right[0]] if len(right) else centers[-1]
    fwhm = max(x_right - x_left, 2.0 * bin_width)
    p0 = np.array([centers[i_peak], fwhm / 8.0, max(float(counts.sum()), 1.0), fwhm / 4.0])
    return np.clip(p0, lower, upper)


# MPV input: x = adc_sum - pedestal, keep x > mpv_nsigma * pedestal_sigma
def mpv_spectrum(task: dict, settings: dict) -> tuple[float, np.ndarray, np.ndarray]:
    threshold = float(settings["mpv_nsigma"]) * task["pedestal_sigma"]
    x = np.arange(task["v0"], task["v0"] + len(task["counts"]), dtype=float) - task["pedestal"]  # adc_sum - pedestal
    above = x > threshold                                           # strict cut
    return threshold, x[above], task["counts"][above].astype(float)


# histogram of x on [0, 2000] ADC in bins of a given width
def binned_spectrum(x: np.ndarray, weights: np.ndarray, settings: dict, bin_width: float):
    lo_hist, hi_hist = map(float, settings["mpv_hist_range"])
    edges = np.arange(lo_hist, hi_hist + 0.5 * bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges, weights=weights)
    return edges, counts


# empty fit record of one channel
def _new_fit_record(task: dict) -> dict:
    return {"channel": task["channel"], "mpv_status": "OK", "mpv": np.nan, "mpv_error": np.nan,
            "landau_width": np.nan, "gauss_sigma": np.nan, "mpv_fit_low": np.nan, "mpv_fit_high": np.nan,
            "n_mpv_entries": 0, "mpv_chi2_ndf": np.nan, "mpv_method": MPV_METHOD,
            "mpv_pedestal": task["pedestal"], "mpv_pedestal_sigma": task["pedestal_sigma"],
            "mpv_threshold": np.nan, "mpv_rough_floor": np.nan, "mpv_rough_status": "NOT_RUN",
            "mpv_rough_peak": np.nan, "mpv_rough_width": np.nan, "mpv_fit_status": "NOT_RUN",
            "mpv_fit_message": "", "mpv_covariance_status": "NOT_COMPUTED", "mpv_active_bounds": "",
            "landau_mpv_param": np.nan, "langaus_area": np.nan, "n_mpv_fit_entries": 0,
            "mpv_deviance": np.nan, "mpv_ndf": np.nan,
            # drawing only: final_counts = the page bins of mpv_fits.pdf; window = first / last fitted bin (indices
            # into fit_edges with --bin, else into the page bins); fit_edges / fit_counts = the --bin bins
            "params": None, "final_counts": None, "window": None, "fit_edges": None, "fit_counts": None,
            "final_stage": False, "seconds": 0.0}


# two-stage MPV fit of one channel (worker; failures give NaN + mpv_status, never raise)
def fit_channel_mpv(task: dict) -> dict:
    s = task["settings"]
    out = _new_fit_record(task)
    pedestal, sigma_ped = task["pedestal"], task["pedestal_sigma"]
    if not (np.isfinite(pedestal) and np.isfinite(sigma_ped)):
        out["mpv_status"] = "NO_PEDESTAL"
        return out
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _fit_channel(task, s, out)
    except Exception as exc:  # a single channel must never stop the stage
        # the stage that raised is marked FAILED, so the QA columns agree with mpv_status: before the rough
        # fit had a status -> rough stage (ROUGH_FIT_FAILED); afterwards -> final stage (FIT_FAILED, also when
        # the final optimizer had converged and the peak / Jacobian / covariance step raised).  An earlier
        # optimizer message is kept in front of the exception text.
        text = f"exception {type(exc).__name__}: {exc}"
        if out["mpv_rough_status"] == "NOT_RUN" and not out["final_stage"]:   # --fit may skip the rough fit
            out.update(mpv_rough_status="FAILED", mpv_status="ROUGH_FIT_FAILED")
        else:
            out.update(mpv_fit_status="FAILED", mpv_status="FIT_FAILED")
        previous = str(out["mpv_fit_message"] or "")
        out.update(mpv=np.nan, mpv_error=np.nan, mpv_fit_message=f"{previous}; {text}" if previous else text)
    _clear_failed_fit(out)
    out["seconds"] = time.perf_counter() - t0
    return out


# a failed fit keeps no fitted values (NaN), only its status and window
def _clear_failed_fit(out: dict) -> None:
    if out["mpv_status"] == "OK":
        return
    for name in FIT_VALUE_COLUMNS:
        out[name] = np.nan
    out["params"] = None


# MPV of one channel: rough fit -> window peak +- 2w -> final fit (or the -fit window)
def _fit_channel(task: dict, s: dict, out: dict) -> None:
    lo_hist, hi_hist = map(float, s["mpv_hist_range"])
    cfg = langaus_config(s)
    threshold, x, weights = mpv_spectrum(task, s)
    out["mpv_threshold"] = threshold

    bw_final = float(s["mpv_final_bin_width"])
    f_edges, f_counts = binned_spectrum(x, weights, s, bw_final)
    out["final_counts"] = f_counts
    out["n_mpv_entries"] = int(f_counts.sum())
    user_window = window_bins(x, weights, s, f_edges, f_counts) if s.get("fit_ranges") else None   # --fit[a,b]
    if user_window is not None and s.get("fit_bins"):     # --bin N: drawn as points, also when the fit fails
        out["fit_edges"], out["fit_counts"] = user_window[0], user_window[1]
    if out["n_mpv_entries"] < int(s["mpv_min_entries"]):
        out["mpv_status"] = "LOW_STATS"
        return

    lower = np.array([lo_hist, s["mpv_eta_range"][0], 1.0, s["mpv_sigma_range"][0]], float)
    upper = np.array([hi_hist, s["mpv_eta_range"][1], 100.0 * out["n_mpv_entries"], s["mpv_sigma_range"][1]], float)
    if user_window is not None:   # Lei 2026-10-02: the user's window for every channel, same model
        _fit_user_window(x, weights, s, out, threshold, user_window, lower, upper, cfg)
        return

    # rough fit: every bin fully above the cut
    state, message, rough_params, fit_floor, peak, width = _rough_fit(x, weights, s, threshold, lower, upper, cfg, out)
    if state != "OK":
        out["mpv_status"] = state
        if message:
            out["mpv_fit_message"] = message
        return

    # final fit: bin centres in peak +- K w, bins fully above the cut
    k = float(s["mpv_window_widths"])
    f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
    floor = fit_floor if bool(s["mpv_window_above_valley"]) else threshold
    window = (f_centers >= peak - k * width) & (f_centers <= peak + k * width) & (f_edges[:-1] >= floor)
    idx = np.where(window)[0]
    if len(idx) < int(s["mpv_min_bins"]):
        out["mpv_status"] = "LOW_STATS"
        return
    out["window"] = (int(idx[0]), int(idx[-1]))
    out["mpv_fit_low"], out["mpv_fit_high"] = float(f_edges[idx[0]]), float(f_edges[idx[-1] + 1])
    _final_fit(out, f_centers[idx], f_counts[idx], bw_final, np.clip(rough_params, lower, upper), lower, upper, cfg)


# rough fit: 40-ADC bins from the valley, chi2; gives the peak and w = FWHM / 2.3548
def _rough_fit(x, weights, s: dict, threshold: float, lower, upper, cfg, out: dict) -> tuple:
    bw_rough = float(s["mpv_rough_bin_width"])
    r_edges, r_counts = binned_spectrum(x, weights, s, bw_rough)
    full = r_edges[:-1] >= threshold
    r_low_edges = r_edges[:-1][full]
    r_centers, r_counts = 0.5 * (r_edges[:-1] + r_edges[1:])[full], r_counts[full]
    fit_floor = threshold
    if len(r_centers) >= int(s["mpv_min_bins"]) and s["mpv_rough_start"] == "valley":
        i0 = valley_bin(r_counts, mip_peak_bin(r_counts))
        fit_floor = max(threshold, float(r_low_edges[i0]))
        r_centers, r_counts = r_centers[i0:], r_counts[i0:]
    out["mpv_rough_floor"] = fit_floor
    nothing = (None, fit_floor, np.nan, np.nan)
    if len(r_centers) < int(s["mpv_min_bins"]):
        return ("LOW_STATS", "") + nothing
    rough_model = binned_langaus(bw_rough, cfg)
    p0 = start_values(r_centers, r_counts, bw_rough, lower, upper)
    if s["mpv_rough_likelihood"] == "neyman":
        rough = fit_neyman(rough_model, r_centers, r_counts, p0, lower, upper)
    else:
        rough = fit_poisson_scaled(rough_model, r_centers, r_counts, p0, lower, upper)
    out["mpv_rough_status"] = fit_status(rough)
    if out["mpv_rough_status"] == "FAILED" or (out["mpv_rough_status"] == "NOT_CONVERGED"
                                               and bool(s.get("mpv_require_rough_convergence", False))):
        return ("ROUGH_FIT_FAILED", f"rough fit: {rough.message}") + nothing
    # a NOT_CONVERGED rough fit (finite parameters) only seeds the window and the start values of the final
    # fit; the result is judged by the final fit alone (converged + peak inside the window)
    m_l, eta, _, sig = rough.params
    span = eta + sig
    peak, fwhm = fitting.curve_peak_and_fwhm(lambda t: langaus_curve(t, m_l, eta, 1.0, sig, cfg),
                                             m_l - 4 * span, m_l + 12 * span, n=2001)
    if not (np.isfinite(peak) and np.isfinite(fwhm) and fwhm > 0):
        return ("ROUGH_FIT_FAILED", "rough fit: curve without a finite peak / FWHM") + nothing
    width = fwhm / fitting.FWHM_PER_SIGMA
    out["mpv_rough_peak"], out["mpv_rough_width"] = peak, width
    return "OK", "", rough.params, fit_floor, peak, width


# bins of the user's -fit window (-bin N: N equal bins)
def window_bins(x, weights, s: dict, f_edges: np.ndarray, f_counts: np.ndarray) -> tuple:
    (a, b), = s["fit_ranges"]
    if s.get("fit_bins"):
        edges = np.linspace(float(a), float(b), int(s["fit_bins"][0]) + 1)
        counts, _ = np.histogram(x, bins=edges, weights=weights)
        return edges, counts, np.ones(len(counts), bool)
    return f_edges, f_counts, fitting.in_range(0.5 * (f_edges[:-1] + f_edges[1:]), (a, b))


# final fit on the user's -fit window, the same for every channel
def _fit_user_window(x, weights, s: dict, out: dict, threshold: float, user_window: tuple, lower, upper,
                     cfg) -> None:
    edges, counts, inside = user_window
    (a, b), = s["fit_ranges"]
    tag = f"--fit [{a:g}, {b:g}]"
    bin_width = (b - a) / int(s["fit_bins"][0]) if s.get("fit_bins") else float(s["mpv_final_bin_width"])
    idx = np.where(inside & (edges[:-1] >= threshold))[0]
    if len(idx) < int(s["mpv_min_bins"]):
        out["mpv_status"] = "LOW_STATS"
        out["mpv_fit_message"] = (f"{tag}: {len(idx)} bin(s) fully above the cut x > {threshold:.4g} ADC; the fit "
                                  f"needs mpv_min_bins = {int(s['mpv_min_bins'])}")
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    c_fit, n_fit = centers[idx], counts[idx]
    try:
        state, message, rough_params = _rough_fit(x, weights, s, threshold, lower, upper, cfg, out)[:3]
    except Exception as exc:  # the rough fit only seeds this fit: its failure is not the channel's
        out["mpv_rough_status"] = "FAILED"
        state, message, rough_params = "ROUGH_FIT_FAILED", f"rough fit: exception {type(exc).__name__}: {exc}", None
    if state == "OK":
        p_start, note = np.clip(rough_params, lower, upper), ""
    else:
        p_start = start_values(c_fit, n_fit, bin_width, lower, upper)
        note = f"start values from the histogram in {tag} ({message or 'rough fit: too few bins'})"
    out["window"] = (int(idx[0]), int(idx[-1]))
    out["mpv_fit_low"], out["mpv_fit_high"] = float(edges[idx[0]]), float(edges[idx[-1] + 1])
    out["final_stage"] = True
    _final_fit(out, c_fit, n_fit, bin_width, p_start, lower, upper, cfg, note)


# mpv calculation: binned Poisson fit in the window; mpv = peak of the curve, mpv_error = sqrt(J C J^T)
def _final_fit(out: dict, c_fit, n_fit, bin_width: float, p_start, lower, upper, cfg, note: str = "") -> None:
    """The peak must lie inside the window; a non-converged fit gives FIT_FAILED."""
    final_model = binned_langaus(bin_width, cfg)
    out["n_mpv_fit_entries"] = int(n_fit.sum())
    final = fit_poisson_scaled(final_model, c_fit, n_fit, p_start, lower, upper)
    out["mpv_fit_status"] = fit_status(final)
    out["mpv_fit_message"] = str(final.message) + (f"; {note}" if note else "")
    if out["mpv_fit_status"] == "FAILED":
        out["mpv_status"] = "FIT_FAILED"
        return
    m_l, eta, area, sig = final.params
    out["params"] = final.params
    out["landau_mpv_param"], out["langaus_area"] = m_l, area
    out["landau_width"], out["gauss_sigma"] = eta, sig
    out["mpv_deviance"], out["mpv_ndf"] = float(final.chi2), int(final.ndf)
    out["mpv_chi2_ndf"] = final.chi2 / final.ndf if final.ndf > 0 else np.nan
    at_bound_lo = np.abs(final.params - lower) <= 1e-9 * np.maximum(np.abs(lower), 1.0)
    at_bound_hi = np.abs(final.params - upper) <= 1e-9 * np.maximum(np.abs(upper), 1.0)
    out["mpv_active_bounds"] = ";".join([f"{n}@lower" for n, b in zip(PARAM_NAMES, at_bound_lo) if b]
                                        + [f"{n}@upper" for n, b in zip(PARAM_NAMES, at_bound_hi) if b])
    if out["mpv_fit_status"] != "CONVERGED":
        # CAL-01: the optimizer stopped without convergence (iteration limit, line-search failure, ...): no MPV /
        # error enters the calibration; mpv_fit_message and mpv_active_bounds say why, and fit_channel_mpv then
        # sets the parameter columns NaN (R3, Lei 2026-10-02: ``_clear_failed_fit``), so the page shows the points
        out["mpv_status"] = "FIT_FAILED"
        return
    # derivatives around the optimum use the convolution grid of the optimum (the adaptive number of steps
    # would add tiny jumps that ruin finite differences)
    cfg_local = (langaus_steps(eta, sig, cfg), cfg[1], 0.0)
    mpv = langaus_peak(m_l, eta, sig, cfg_local)
    if not (out["mpv_fit_low"] <= mpv <= out["mpv_fit_high"]):
        out["mpv_status"] = "PEAK_OUTSIDE_WINDOW"
        return
    # error: J C J^T, peak = mpv_landau + delta(eta, sigma)  ->  J = (1, d/d eta, 0, d/d sigma)
    cov, out["mpv_covariance_status"] = deviance_covariance_with_status(
        binned_langaus(bin_width, cfg_local), c_fit, n_fit, final.params, free=~(at_bound_lo | at_bound_hi))
    jac = np.array([1.0, 0.0, 0.0, 0.0])
    for j, value in ((1, eta), (3, sig)):
        h = 1e-2 * max(abs(value), 1.0)
        plus, minus = final.params.copy(), final.params.copy()
        plus[j] += h
        minus[j] -= h
        jac[j] = (langaus_peak(plus[0], plus[1], plus[3], cfg_local, near=mpv)
                  - langaus_peak(minus[0], minus[1], minus[3], cfg_local, near=mpv)) / (2 * h)
    variance = float(jac @ cov @ jac)
    out["mpv_error"] = np.sqrt(variance) if np.isfinite(variance) and variance > 0 else np.nan
    out["mpv"] = mpv


# one fit task per channel (only the histogram part above the cut)
def mpv_tasks(h_all: np.ndarray, pedestal: np.ndarray, pedestal_sigma: np.ndarray, settings: dict) -> list[dict]:
    fit_s = fit_settings(settings)
    hi_hist = float(settings["mpv_hist_range"][1])
    tasks = []
    for c in range(N_CHANNELS):
        ped, sig = float(pedestal[c]), float(pedestal_sigma[c])
        if np.isfinite(ped) and np.isfinite(sig):
            v0 = int(np.clip(np.floor(ped), 0, N_ADC_SUM))
            v1 = int(np.clip(np.ceil(ped + hi_hist) + 1, v0, N_ADC_SUM))
        else:
            v0 = v1 = 0
        tasks.append({"channel": c, "pedestal": ped, "pedestal_sigma": sig, "v0": v0,
                      "counts": h_all[c, v0:v1].copy(), "settings": fit_s})
    return tasks


# run every channel fit, in worker processes when workers > 1
def run_mpv_fits(tasks: list[dict], workers: int, side_job=None) -> list[dict]:
    if workers <= 1:
        fits = [fit_channel_mpv(t) for t in tasks]
        if side_job is not None:
            side_job[0](*side_job[1])
        return fits
    io_utils.limit_blas_threads()  # inherited by the worker processes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        side = pool.submit(side_job[0], *side_job[1]) if side_job is not None else None
        fits = list(pool.map(fit_channel_mpv, tasks, chunksize=max(1, len(tasks) // (8 * workers))))
        if side is not None:
            side.result()
    return fits


# --------------------------------------------------------------------------
# raw-histogram cache (intermediate of -from-saved)
# --------------------------------------------------------------------------
# write the integer histograms + their record (record last)
def write_histogram_cache(path: Path, hists: dict, meta: dict) -> Path:
    arrays = {}
    for key in ("h_selected", "h_all", "h_sample"):
        h = hists.get(key)
        if h is not None:
            arrays[key] = h.astype(np.int32) if h.max(initial=0) < np.iinfo(np.int32).max else h
    meta = {**meta, **CACHE_IDENTITY, "arrays": sorted(arrays)}
    text = json.dumps(meta, sort_keys=True, default=str)

    # npz writer
    def write(tmp: Path) -> None:
        np.savez_compressed(tmp, meta=np.array(text), **arrays)

    io_utils.commit_product_set({path: write}, meta, provenance_json(path))
    return path


# read the histogram cache
def read_histogram_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        if meta.get("format") != CACHE_FORMAT:
            raise io_utils.ProductSetError(f"{path}: histogram cache format {meta.get('format')!r} (this version "
                                           f"reads {CACHE_FORMAT}); rerun .\\runner.ps1 calib")
        out = {"meta": meta}
        for key in ("h_selected", "h_all", "h_sample"):
            out[key] = npz[key].astype(np.int64) if key in npz.files else None
    return out


# is the histogram cache complete and unchanged?
def cache_status(path: Path) -> tuple[bool, str]:
    return io_utils.product_set_valid(provenance_json(path), CACHE_IDENTITY, members=[path])


# 'path (size, mtime_ns)' of input identities, for messages
def _files_text(idents) -> str:
    return "; ".join(f"{i.get('path')} (size {i.get('size')}, mtime_ns {i.get('mtime_ns')})"
                     for i in idents if isinstance(i, dict))


# how the input csv files differ from those the cache was made from
def _input_change(record_path: Path, identities: dict[str, dict]) -> str:
    record = io_utils.read_json(record_path) or {}
    text = ""
    for name, ident in identities.items():
        stored, now = record.get(f"{name}_histograms_inputs"), ident["inputs"]
        if isinstance(stored, list) and json.dumps(stored, sort_keys=True) != json.dumps(now, sort_keys=True):
            text += (f"\n  the cached {name} histograms were made from {_files_text(stored)}; this run reads "
                     f"{_files_text(now)}")
    return text


# -from-saved: the cached histograms of this run's inputs, or stop with the reason
def saved_histograms(ctx, identities: dict[str, dict], rebuild: str) -> dict:
    path = histogram_cache(ctx)
    record_path = provenance_json(path)
    ok, reason = io_utils.product_set_valid(record_path, {**CACHE_IDENTITY, **flat_identity(identities)},
                                            members=[path])
    if not ok:
        raise io_utils.ProductSetError(f"-from-saved: the raw histograms in {path.parent} cannot be reused: "
                                       f"{reason}{_input_change(record_path, identities)}\n  Rebuild them with: "
                                       f"{rebuild}")
    cache = read_histogram_cache(path)
    meta = cache["meta"]
    ident = identities["pedestal"]
    need = ["h_selected"] + (["h_sample"] if ident["anchor"] == ANCHOR_MODE else [])
    missing = [k for k in need if cache[k] is None]
    if missing:
        raise io_utils.ProductSetError(f"-from-saved: {path} lacks the pedestal histograms {missing}\n"
                                       f"  Rebuild them with: {rebuild}")
    out = {"h_selected": cache["h_selected"], "h_all": None,
           "h_sample": cache["h_sample"] if ident["anchor"] == ANCHOR_MODE else None,
           "pedestal_selection": ident["selection"], "pedestal_anchor_method": ident["anchor"],
           "n_rows_pedestal": meta.get("n_rows_pedestal")}
    if "mpv" in identities:
        if cache["h_all"] is None:
            raise io_utils.ProductSetError(f"-from-saved: {path} lacks the MPV histograms (h_all)\n"
                                           f"  Rebuild them with: {rebuild}")
        out.update(h_all=cache["h_all"], n_rows_mpv=meta.get("n_rows_mpv"))
    out["n_rows"] = out["n_rows_pedestal"]
    log(f"-from-saved: {' and '.join(identities)} histograms reused from {path} (they match the input csv files and "
        "the pedestal selection); no csv is read")
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
# one page of four detector maps
def _summary_maps(pdf, table: pd.DataFrame, columns: list[tuple[str, int, str]], title: str) -> None:
    from ..common import plotting

    fig, axes = plotting.new_figure(2, 2, figsize=(11, 8.5))
    for ax, (column, end, label) in zip(axes.flat, columns):
        sub = table[table["end"] == end]
        grid = geo.fill_layer_grid(sub["layer"], sub["strip"], pd.to_numeric(sub[column], errors="coerce"))
        plotting.layer_map(ax, grid, title=f"{label}, end {end}", cbar_label=label, annotate=True)
    fig.suptitle(title)
    fig.tight_layout()
    plotting.save(fig, pdf)


# 2 x 2 panels per page on one reused figure (fast pdfs)
class _PanelPages:

    # set up the figure
    def __init__(self, setup_axis):
        from ..common import plotting

        self.fig, axes = plotting.new_figure(2, 2, figsize=(11, 8.5))
        self.panels = []
        for ax in axes.flat:
            panel = setup_axis(ax)
            panel["ax"] = ax
            panel["box"] = ax.text(0.97, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                                   family="monospace", bbox=dict(boxstyle="round", fc="w", ec="0.7"))
            panel["message"] = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center")
            self.panels.append(panel)
        # fixed margins: page content changes, and no layout engine means one draw per savefig
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.22, hspace=0.32)

    # draw every item, 4 per page
    def write(self, pdf, n_items: int, fill) -> None:
        for first in range(0, n_items, len(self.panels)):
            for k, panel in enumerate(self.panels):
                index = first + k
                visible = index < n_items and bool(fill(panel, index))
                if index >= n_items:
                    panel["ax"].set_title("")
                for artist in panel["data"]:
                    artist.set_visible(visible)
                panel["box"].set_visible(visible)
                panel["message"].set_visible(index < n_items and not visible)
            pdf.savefig(self.fig)

    # close the figure
    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


# pedestal page counts: adc_sum - anchor in 6-ADC bins over [-200, 400)
def pedestal_plot_counts(h_selected: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    lo, hi = PEDESTAL_PLOT_RANGE
    counts = np.zeros((N_CHANNELS, hi - lo))
    for c in np.where(np.isfinite(anchor))[0]:
        v_lo = int(anchor[c]) + lo
        src_lo, src_hi = max(v_lo, 0), min(v_lo + hi - lo, N_ADC_SUM)
        if src_hi > src_lo:
            counts[c, src_lo - v_lo:src_hi - v_lo] = h_selected[c, src_lo:src_hi]
    return counts.reshape(N_CHANNELS, -1, PEDESTAL_PLOT_BIN).sum(axis=2)


# pedestal_fits.pdf: summary maps + one panel per channel
def plot_pedestals(path: Path, table: pd.DataFrame, plot_counts: np.ndarray, halfwidth: float,
                   source_label: str) -> Path:
    from matplotlib.ticker import FuncFormatter, NullLocator

    lo, hi = PEDESTAL_PLOT_RANGE
    edges = np.arange(lo, hi + PEDESTAL_PLOT_BIN, PEDESTAL_PLOT_BIN, dtype=float)
    xs = np.linspace(lo, hi, 301)

    # panel artists
    def setup(ax):
        steps = ax.stairs(np.ones(len(edges) - 1), edges, color="C0")
        gauss_line, = ax.plot([], [], "k-", lw=1.2)
        for edge in (-halfwidth, halfwidth):
            ax.axvline(edge, color="gray", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # plain labels: no mathtext
        ax.set_xlim(lo, hi)
        ax.set_xlabel("adc_sum - anchor [ADC]  (selected: no TOT/TOA on both ends)")
        ax.set_ylabel(f"events / {PEDESTAL_PLOT_BIN} ADC")
        return {"data": [steps, gauss_line], "steps": steps, "gauss": gauss_line}

    # one channel's panel
    def fill(panel, c):
        row = table.iloc[c]
        name = geo.channel_name(row["layer"], row["strip"], row["end"])
        panel["ax"].set_title(name)
        anchor = row["pedestal_anchor"]
        if not np.isfinite(anchor):
            panel["message"].set_text("no pedestal events")
            return False
        counts = plot_counts[c]
        panel["steps"].set_data(counts, edges)
        mu, sd, n = row["pedestal_mu"], row["pedestal_sigma"], row["n_pedestal_events"]
        if np.isfinite(sd) and sd > 0:
            panel["gauss"].set_data(xs, fitting.gauss(xs, n * PEDESTAL_PLOT_BIN * fitting.INV_SQRT_2PI / sd, mu, sd))
        else:
            panel["gauss"].set_data([], [])
        panel["ax"].set_ylim(0.5, max(counts.max(), 1.0) * 3)
        panel["box"].set_text(f"anchor   {anchor:8.0f}\npedestal {row['pedestal']:8.2f}\nsigma    {sd:8.2f}\n"
                              f"mu       {mu:8.2f}\nN        {int(n):8d}")
        return True

    with io_utils.atomic_pdf(path) as pdf:
        _summary_maps(pdf, table, [("pedestal", 0, "pedestal [ADC]"), ("pedestal", 1, "pedestal [ADC]"),
                                   ("pedestal_sigma", 0, "pedestal sigma [ADC]"),
                                   ("pedestal_sigma", 1, "pedestal sigma [ADC]")],
                      f"Pedestals from {source_label}")
        pages = _PanelPages(setup)
        pages.write(pdf, len(table), fill)
        pages.close()
    return path


# mpv_status -> number of channels
def _status_counts(table: pd.DataFrame) -> dict:
    counts = table["mpv_status"].astype(str).value_counts()
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# fitted curve of one panel (none when the fit failed)
def mpv_panel_curve(fit: dict, row, model, xs: np.ndarray):
    if fit.get("params") is None or not np.isfinite(row["mpv"]):
        return None
    ys = model(xs, *fit["params"])
    inside = (xs >= row["mpv_fit_low"]) & (xs <= row["mpv_fit_high"])
    return np.where(inside, ys, np.nan), np.where(inside, np.nan, ys)


# -bin bins of one panel as points on the page scale
def window_points(fit: dict, page_width: float):
    fe, fc = fit.get("fit_edges"), fit.get("fit_counts")
    if fe is None or fc is None:
        return None
    fe, fc = np.asarray(fe, float), np.asarray(fc, float)
    scale = page_width / np.diff(fe)
    return 0.5 * (fe[:-1] + fe[1:]), fc * scale, np.sqrt(fc) * scale


# mpv_fits.pdf: summary maps + the final fit of every channel
def plot_mpv(path: Path, table: pd.DataFrame, fits: list[dict], settings: dict, source_label: str) -> Path:
    from matplotlib.patches import Rectangle

    lo_hist, hi_hist = map(float, settings["mpv_hist_range"])
    bw = float(settings["mpv_final_bin_width"])
    edges = np.arange(lo_hist, hi_hist + 0.5 * bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xs = np.linspace(lo_hist, hi_hist, 501)
    model = binned_langaus(bw, langaus_config(settings))
    status = _status_counts(table)
    window_text = fit_window_text(settings)
    user_bins = bool(window_text) and bool(settings.get("fit_bins"))

    # panel artists
    def setup(ax):
        window = Rectangle((0, 0), 0, 1, transform=ax.get_xaxis_transform(), color="C1", alpha=0.15, lw=0)
        ax.add_patch(window)
        steps = ax.stairs(np.ones(len(centers)), edges, color="C0")
        bars = ax.vlines(centers, 0, 0, color="C0", lw=0.7)
        inside, = ax.plot([], [], "r-", lw=1.5)
        outside, = ax.plot([], [], "r--", lw=0.8)
        peak = ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlim(lo_hist, hi_hist)
        ax.set_xlabel(f"adc_sum - pedestal [ADC]  (> {float(settings['mpv_nsigma']):g} pedestal sigma)")
        ax.set_ylabel(f"entries / {bw:g} ADC")
        panel = {"data": [window, steps, bars, inside, outside, peak], "window": window, "steps": steps,
                 "bars": bars, "inside": inside, "outside": outside, "peak": peak}
        if user_bins:   # --bin: the fitted bins as points (only then: the default pages are unchanged)
            points, = ax.plot([], [], "o", ms=2.5, color="k")
            point_bars = ax.vlines([0.0], 0, 0, color="k", lw=0.7)
            panel.update(points=points, point_bars=point_bars)
            panel["data"] += [points, point_bars]
        return panel

    # one channel's panel
    def fill(panel, c):
        fit, row = fits[c], table.iloc[c]
        panel["ax"].set_title(f"{geo.channel_name(row['layer'], row['strip'], row['end'])}   {row['mpv_status']}")
        counts = fit.get("final_counts")
        if counts is None:
            panel["message"].set_text(str(row["mpv_status"]))
            return False
        panel["steps"].set_data(counts, edges)
        err = np.sqrt(counts)
        panel["bars"].set_segments([((x, y - e), (x, y + e)) for x, y, e in zip(centers, counts, err)])
        if fit.get("window") is not None:
            i0, i1 = fit["window"]
            w_edges = fit["fit_edges"] if fit.get("fit_edges") is not None else edges   # --bin: its own bins
            panel["window"].set_x(w_edges[i0])
            panel["window"].set_width(w_edges[i1 + 1] - w_edges[i0])
        else:
            panel["window"].set_width(0)
        curve = mpv_panel_curve(fit, row, model, xs)
        if curve is not None:
            panel["inside"].set_data(xs, curve[0])
            panel["outside"].set_data(xs, curve[1])
        else:   # a failed fit: the histogram (and the --bin points) only
            panel["inside"].set_data([], [])
            panel["outside"].set_data([], [])
        top = counts.max()
        if "points" in panel:
            pts = window_points(fit, bw)
            if pts is not None:
                panel["points"].set_data(pts[0], pts[1])
                panel["point_bars"].set_segments([((x, y - e), (x, y + e)) for x, y, e in zip(*pts)])
                top = max(top, float(np.max(pts[1], initial=0.0)))
            else:
                panel["points"].set_data([], [])
                panel["point_bars"].set_segments([])
        mpv = row["mpv"] if np.isfinite(row["mpv"]) else -1e9
        panel["peak"].set_xdata([mpv, mpv])
        panel["ax"].set_ylim(0, max(top, 1.0) * 1.3)
        entries = f"entries  {int(row['n_mpv_entries']):7d}"
        if np.isfinite(row["mpv"]):
            panel["box"].set_text(f"MPV      {row['mpv']:7.1f} +- {row['mpv_error']:.1f}\n"
                                  f"eta      {row['landau_width']:7.1f}\nsigma    {row['gauss_sigma']:7.1f}\n"
                                  f"chi2/ndf {row['mpv_chi2_ndf']:7.2f}\n{entries}")
        else:
            panel["box"].set_text(f"{row['mpv_status']}\n{entries}")
        return True

    with io_utils.atomic_pdf(path) as pdf:
        _summary_maps(pdf, table, [("mpv", 0, "MPV [ADC]"), ("mpv", 1, "MPV [ADC]"),
                                   ("mpv_error", 0, "MPV error [ADC]"), ("mpv_error", 1, "MPV error [ADC]")],
                      f"MIP MPV from {source_label}\nstatus: {status}"
                      + (f"\nfinal fit window of every channel: {window_text}"
                         + ("  (black points, scaled to the page bins)" if user_bins else "")
                         if window_text else ""))
        pages = _PanelPages(setup)
        pages.write(pdf, len(table), fill)
        pages.close()
    return path


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------
# the table as written: the given columns (default: keys + every column), typed, sorted by channel
def table_frame(table: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = table[columns or KEY_COLUMNS + list(CALIBRATION_COLUMNS)].copy()
    for c in INTEGER_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    for c in STRING_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out.sort_values(KEY_COLUMNS).reset_index(drop=True)


# write a calibration table (+ provenance, record last)
def write_table(table: pd.DataFrame, path: Path, provenance: dict | None = None,
                columns: list[str] | None = None) -> Path:
    frame = table_frame(table, columns)
    if provenance is None:
        return io_utils.atomic_write_csv(frame, path)
    io_utils.commit_product_set({Path(path): lambda tmp: frame.to_csv(tmp, index=False)},
                                {**provenance, "columns": list(frame.columns)}, provenance_json(path))
    return Path(path)


# log the user's MPV fit window (-fit / -bin)
def _log_fit_window(settings: dict, args) -> None:
    text = fit_window_text(settings)
    if not text:
        return
    log(f"MPV final fit window of every channel: {text} (x = adc_sum - pedestal; bins fully above the cut; start "
        "values from the rough fit, from the histogram in the window when it fails; Lei 2026-10-02)")
    if getattr(args, "mpv_window_widths", None) is not None:
        log("note: --mpv-window-widths is not used with --fit (the window is [a, b])")
    if settings.get("fit_bins"):
        (a, b), = settings["fit_ranges"]
        width = (b - a) / int(settings["fit_bins"][0])
        if abs(width - round(width)) > 1e-9 * max(1.0, width):
            log(f"note: --bin {settings['fit_bins'][0]} on [{a:g}, {b:g}] gives {width:g}-ADC bins; adc_sum is an "
                f"integer, so these bins hold alternately {int(np.floor(width))} and {int(np.ceil(width))} adc_sum "
                f"values (a sawtooth of up to {1.0 / width:.0%} in the counts); an integer width avoids it")


# provenance text of the MPV fit window
def mpv_window_record(settings: dict) -> str:
    return fit_window_text(settings) or "automatic: rough-fit peak +- mpv_window_widths x FWHM/2.3548"


# stage entry: units, source run, histograms, pedestal, MPV fits, table, pdfs
def run(ctx, args, datasets) -> None:
    t_start = time.time()
    check_fit_options(ctx, args)
    units = units_or_stop(ctx)
    settings = settings_for(ctx, args)
    from_saved = bool(getattr(args, "from_saved", False))
    need_mpv, parts_drawn = needs_mpv(units), drawn_parts(units)
    result_plan(ctx, parts_drawn, preview=True)   # an --output problem stops the run here, before any IO

    ds = source_dataset(ctx, datasets, report=True)
    settings["source_dataset"] = run_ref(ds)   # the run used (also when a run flag chose it): provenance settings_all
    if ds.particle != "muon" and need_mpv:
        log(f"WARNING: calib on {run_ref(ds)}: the MIP MPVs come from a {ds.particle} run, but the MIP calibration is "
            f"defined on muons; the run is used as chosen: check the MPVs before the later stages read them")
    if need_mpv:
        _log_fit_window(settings, args)
    hint = _make_csv_hint(ctx, ds)
    for item in ctx.beam_plan_items(ds):   # Lei D5: refuse a csv without the raw sums before anything is built
        header, name = _beam_header_csv(item)
        if header is not None:
            resolve_pedestal_methods(io_utils.csv_columns(header), settings["pedestal_selection"],
                                     settings["pedestal_anchor"], name, hint)
    inputs, upstream = stage_inputs(ctx, ds, need_mpv, from_saved)
    if ctx.clean_input:
        log("-clean-input: " + (f"MPV fitted on the clean datasets {_names(inputs['mpv'])}; " if need_mpv else
                                "no MPV unit: nothing is read from the clean datasets; ")
            + f"pedestal from the beam csv {_names(inputs['pedestal'])} (the clean selection removes the no-signal "
              "rows)")
    # Lei D5: stops here, before any reading, when the strict selection cannot be applied
    selection, anchor = resolve_pedestal_methods(common_columns(inputs["pedestal"]), settings["pedestal_selection"],
                                                 settings["pedestal_anchor"], _names(inputs["pedestal"]), hint)
    nsigma, fraction = float(settings["pedestal_range_nsigma"]), float(settings["pedestal_range_fraction"])
    if (nsigma, fraction) != (5.0, 0.2):
        log(f"note: pedestal_range_with_5sigma = pedestal + {nsigma:g} sigma and pedestal_range_with_20percent = "
            f"{1.0 + fraction:g} x pedestal (non-default calibration.pedestal_range_nsigma / "
            "pedestal_range_fraction; the column names stay)")
    identities = histogram_identities(inputs, selection, anchor)
    notes: dict[Path, dict] = {}
    for path in dict.fromkeys(p for paths in inputs.values() for p in paths):   # converter sample counts, once per csv
        notes[path] = input_samples_note(path, upstream.get(path))
    samples = {role: [notes[p] for p in paths] for role, paths in inputs.items()}
    labels = {role: f"{ds.label} [{_names(paths)}]" + (" TEST" if ctx.test else "") for role, paths in inputs.items()}
    if from_saved:
        hists = saved_histograms(ctx, identities, _rebuild_command(ctx, datasets))
    else:
        hists = read_histograms(inputs, int(ctx.cfg.get("run", {}).get("chunksize", io_utils.DEFAULT_CHUNKSIZE)),
                                selection, anchor, hint)
    used = "used" if hists["h_sample"] is not None else "absent / not used"
    log(f"{hists['n_rows']} rows; per-sample columns {used}")
    run_token = io_utils.settings_hash({"time_ns": time.time_ns(), "inputs": [str(p) for p in inputs["pedestal"]],
                                        "pid": os.getpid()})
    created = time.strftime("%Y-%m-%d %H:%M:%S")

    # the pedestal: always computed (the MPV fit subtracts it)
    table = empty_calibration()
    ped = pedestals_from_histograms(hists["h_selected"], hists["h_sample"], settings)
    ped["pedestal_selection"] = hists["pedestal_selection"]
    for col in PEDESTAL_COLUMNS:   # both frames are in geometry.all_channels order
        table[col] = ped[col].to_numpy()
    log(f"pedestals: {int(np.isfinite(table['pedestal']).sum())}/{N_CHANNELS} channels, selection = "
        f"{hists['pedestal_selection']}, anchor = {hists['pedestal_anchor_method']}"
        + ("  (csv approximations: see the module help)"
           if SELECTION_CSV == hists["pedestal_selection"] or ANCHOR_MEDIAN == hists["pedestal_anchor_method"]
           else ""))

    # the final results of this run (Lei 2026-10-05): the chosen pdfs; the tables and the cache are intermediates
    plan = result_plan(ctx, parts_drawn)
    pedestal_pdf_job = None
    if units["pedestal_plot"]:
        counts = pedestal_plot_counts(hists["h_selected"], table["pedestal_anchor"].to_numpy())
        pedestal_pdf_job = (plot_pedestals, (plan.path(RESULT_KEY["pedestal"]), table.copy(), counts,
                                             float(settings["pedestal_window_halfwidth"]), labels["pedestal"]))
    fits = None
    if need_mpv:
        tasks = mpv_tasks(hists["h_all"], table["pedestal"].to_numpy(float),
                          table["pedestal_sigma"].to_numpy(float), settings)
        what = f"MPV fits of {N_CHANNELS} channels on {ctx.workers} worker(s)"
        registry, worker_tmp = io_utils.active_registry(), None
        if pedestal_pdf_job is not None and ctx.workers > 1 and registry is not None:
            pdf = plan.path(RESULT_KEY["pedestal"])   # drawn in a worker process (no registry there): its temp file
            worker_tmp = registry.register(pdf.with_name(pdf.stem + ".tmp.pdf"))
        with io_utils.timed(what + (" (+ pedestal_fits.pdf)" if pedestal_pdf_job else "")):
            fits = run_mpv_fits(tasks, ctx.workers, side_job=pedestal_pdf_job)
        if worker_tmp is not None:
            registry.release(worker_tmp)
        for col in MPV_COLUMNS:
            table[col] = [f[col] for f in fits]
    elif pedestal_pdf_job is not None:
        with io_utils.timed("pedestal_fits.pdf"):
            pedestal_pdf_job[0](*pedestal_pdf_job[1])

    meta = {"stage": "calib", "run_token": run_token, "units": [u for u in UNITS if units[u]], "mode": ctx.mode,
            "source_dataset": ds.key, "source_label": labels["pedestal"],
            "inputs": {role: [io_utils.file_signature(p) for p in paths] for role, paths in inputs.items()},
            "clean_input": bool(ctx.clean_input), "from_saved": from_saved, "n_rows": hists["n_rows"],
            "pedestal_selection": hists["pedestal_selection"],
            "pedestal_anchor_method": hists["pedestal_anchor_method"],
            "settings": {**fit_settings(settings), "pedestal_window_halfwidth": settings["pedestal_window_halfwidth"]},
            "created": created}
    cache_path = None
    if settings["write_histogram_cache"] and not from_saved:   # -from-saved read it: it stays as it is
        arrays = {k: hists[k] for k in ("h_selected", "h_all", "h_sample") if hists.get(k) is not None}
        cache_path = write_histogram_cache(histogram_cache(ctx), arrays, {
            **meta, **flat_identity(identities), "n_rows_pedestal": hists.get("n_rows_pedestal"),
            "n_rows_mpv": hists.get("n_rows_mpv")})

    status_counts = _status_counts(table) if need_mpv else {}
    path = table_csv(ctx, units)
    if path is not None:
        has_mpv = bool(units["mpv_csv"])
        write_table(table, path, {
            **meta, "table": "full" if units["pedestal_csv"] and has_mpv else "pedestal" if units["pedestal_csv"]
            else "mpv",
            "test_mode": bool(ctx.test), "n_test_events": int(ctx.n_test_events) if ctx.test else None,
            "input_samples": samples, "histograms": identities,
            "mpv_method": MPV_METHOD if has_mpv else None,
            "mpv_fit_window": mpv_window_record(settings) if has_mpv else None,   # --fit / --bin (Lei 2026-10-02)
            "settings_all": settings, "settings_hash": io_utils.settings_hash(settings),
            "histogram_cache": str(cache_path) if cache_path else None, "status_counts": status_counts,
            "versions": io_utils.python_info(),
        }, table_columns(units))
        log(f"calibration table written: {path} ({len(table)} rows)")
        if has_mpv and fit_window_text(settings):   # never mistaken for the default calibration (Lei 2026-10-02)
            log(f"note: the MPV columns of {path.name} come from the user window {fit_window_text(settings)}; the "
                "later stages read them as the calibration (provenance: mpv_fit_window, settings fit_ranges / "
                "fit_bins); calib without --fit (e.g. calib -from-saved) restores the automatic window")
    drawn = [plan.path(RESULT_KEY["pedestal"])] if pedestal_pdf_job is not None else []
    if units["mpv_plot"]:
        with io_utils.timed("mpv_fits.pdf"):
            drawn.append(plot_mpv(plan.path(RESULT_KEY["mpv"]), table, fits, settings, labels["mpv"]))

    text = f"summary: pedestal NaN {int(table['pedestal'].isna().sum())}"
    if need_mpv:
        mpv = table.loc[table["mpv_status"] == "OK", "mpv"]
        covariance = table.loc[table["mpv_status"] == "OK", "mpv_covariance_status"].astype(str).value_counts()
        text += (f", MPV status {status_counts}"
                 + (f", MPV median {mpv.median():.1f} ADC (range {mpv.min():.1f}-{mpv.max():.1f})" if len(mpv) else "")
                 + (f", covariance of OK fits {covariance.to_dict()}" if len(covariance) else ""))
    log(text)
    log(f"calib done in {time.time() - t_start:.1f}s: " + ", ".join(str(p) for p in [path, *drawn, cache_path] if p))
    report_results(plan)
