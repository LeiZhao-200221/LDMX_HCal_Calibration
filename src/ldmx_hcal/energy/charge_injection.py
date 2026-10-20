"""Stage ci: charge-injection map and TOT scale s per channel -> event energies [MIP] on three scales.

q(TOT): the injected charge [fC] that gives a TOT reading, from the charge-injection scans (provided by Erik Wallin).
s (column s_final, path s; s parameter, Lei): per-channel TOT scale = beam ADC / charge-injection prediction k q(TOT),
trained on pion 4 by default; several training samples (-scale-particle pion muon) give one path s_<sample> each, and
s = the sample with the fewest channels that borrow their s from the (DPM, LINK) / global prior.
energy [MIP]: per end (adc_sum - pedestal) / mpv, or max(that, k s q(TOT) / mpv) for an end near the ADC rail;
scales adc_only (no TOT), pureci (PureCI, s = 1) and s; event energy = sum of its bar energies.
Output units (-only): parameters, events, linearity, resolution, eh; on request the checks response, quality_scan.

Variables of ci_channel_parameters.csv (one row per channel, column order):
  layer, strip, end, dpm, link, chan, ci_status, ci_reason, ci_k_adc_per_fc, ci_P_injection, ci_q_threshold_fc,
  ci_pole_asym_slope_counts_per_fc, ci_pole_asym_intercept_counts, ci_pole_strength, ci_pole_position_fc,
  n_ci_measured, n_ci_reliable, n_ci_interp_nodes, n_ci_pava_merged, n_ci_pole_audit_outliers, n_ci_rail_excluded,
  mono_tot_min, mono_tot_max, mono_q_min_fc, mono_q_max_fc, mono_slope_low_fc_per_tot, mono_slope_high_fc_per_tot,
  ci_tot_threshold_fc, ci_thr_asym_slope_counts_per_fc, ci_thr_asym_intercept_counts, ci_thr_pole_strength,
  ci_thr_pole_position_fc, ci_thr_adc_gain_per_fc, ci_thr_adc_intercept_adc, ci_thr_status,
  n_scan_events, n_low_groups_kept, n_high_groups_kept, n_groups_dropped_repeats, n_groups_dropped_consistency,
  training_particle, shrinkage, s_direct, s_exact_n, s_censored_n, shrinkage_weight, s_link_prior, s_prior_level,
  s_final, s_source, global_prior, [s_<sample>, s_<sample>_source: one pair per sample with several samples]
Variables of ci_tot_charge_nodes.csv (one row per node of a channel's q(TOT) map):
  layer, strip, end, node_index, tot_counts, q_fc
Variables of <run>/<csv stem>_event_energies.csv (one row per reconstructed event of one input csv):
  event_source, pf_event, energy_adc_only_mip, energy_pureci_mip, energy_s_mip, [energy_s_<sample>_mip],
  n_bars, bad_strip_fraction, selected
Cache for the plots stage (cache/[test/]charge_injection/events<v>/<run>/, every reconstructed path):
  <csv stem>.npz (per event): pf_event, n_bars, n_unrecoverable_bars, bad_strip_fraction,
  saturated_unconverted_fraction, n_events_in_file, energy_<path>;
  <csv stem>_bars.npy (per bar of the reconstructed events, float32 values): pf_event, layer, strip, tot_end0,
  tot_end1, bar_energy_<path>_mip, end0_energy_<path>_mip, end1_energy_<path>_mip
Variables of energy_response.csv (-only response; spectrum rows per run and path, linearity rows per path and fitted
particle, one resolution row per path, e/h rows per s path and pion run):
  row_type, dataset, particle, beam_energy_gev, path, selection_path, quality_threshold, n_csv_files,
  n_events_in_file, n_events_reconstructed, n_events_selected, retention_fraction, n_bars_selected_events, fit_kind,
  fit_status, fit_ok, fit_ranges, fit_bins, mu_mip, mu_error_mip, sigma_mip, sigma_error_mip, sigma_over_mu, chi2,
  ndf, fit_low_mip, fit_high_mip, bin_width_mip, hist_upper_mip, pass1_mu_mip, pass1_sigma_mip, pass1_chi2,
  pass1_ndf, core_bin_count, mu_low_mip, mu_low_error_mip, sigma_low_mip, sigma_low_error_mip, mu_high_mip,
  mu_high_error_mip, sigma_high_mip, sigma_high_error_mip, [area_low_events, area_low_error_events,
  area_high_events, area_high_error_events: double -fit only], peak_scan_mu_mip, peak_scan_sigma_mip,
  peak_scan_sigma_over_mu, peak_scan_n_peaks, peak_scan_status, mu_stat_error_mip, ratio_error, linearity_fitted,
  linearity_predicted_mip, linearity_residual_fraction, fitted_energies, linearity_fit_range, slope_mip_per_gev,
  slope_error_mip_per_gev, intercept_mip, intercept_error_mip, chi2_over_ndf, max_abs_rel_residual,
  linearity_source, error_prescription, [error_scale, slope_error_scaled_mip_per_gev, intercept_error_scaled_mip:
  -normalerror only], resolution_fit_range, resolution_a_mev, resolution_a_error_mev, resolution_b_sqrt_mev,
  resolution_b_error_sqrt_mev, resolution_c, resolution_c_error, [resolution_a_error_scaled_mev,
  resolution_b_error_scaled_sqrt_mev, resolution_c_error_scaled: -normalerror only], e_over_h, e_over_h_error,
  mu_electron_mip, mu_electron_error_mip, mu_hadron_mip, mu_hadron_error_mip, electron_source, eh_filled, eh_note
Variables of quality_threshold_scan.csv (-only quality_scan; one row per threshold and electron run):
  threshold, dataset, beam_energy_gev, n_events_before, n_events_after, retention_fraction, peak_scan_mu_mip,
  peak_scan_sigma_mip, sigma_over_mu, n_detected_peaks, detected_peak_positions_mip,
  peak_shift_vs_unfiltered_fraction, resolution_change_vs_unfiltered_fraction, mean_bad_strip_fraction_all,
  shoulder_count, core_count, shoulder_retention_fraction, core_retention_fraction, configured_threshold
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import textwrap
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import peak_widths

from . import energy_response as er
from ..common import outputs
from ..common.calibration_io import KEY_COLUMNS, load_calibration
from ..common.config import DEFAULT_CONFIG, RUN_PARTICLES, resolve_path, resolve_runs, run_flags, run_ref
from ..common.fitting import format_ranges
from ..common.io_utils import (ChannelKeyError, ProductSet, SettingsError, active_registry, atomic_pdf,
                               atomic_write_csv, content_hash, ensure_writable, file_signature, iter_csv,
                               json_records_on, limit_blas_threads, log, product_set_valid, python_info, read_json,
                               require, require_product_set, settings_hash, sweep_orphan_temps, temp_registry,
                               validate_channel_keys, validate_settings, write_json)

# ==========================================================================
# BLOCK 0: constants and configuration
# ==========================================================================
HELP = "charge injection: CI map + s fit, event energies [MIP] on the scales adc_only, PureCI and s"
DEFAULT_DATASETS = ["electron 0.3", "electron 0.4", "electron 0.5", "electron 1", "electron 4",
                    "pion 0.5", "pion 1", "pion 3", "pion 4"]  # run references: electron linearity + pion points
CONFIG_SECTION = "charge_injection"
STAGE_DIR = "charge_injection"   # sub-folder of data/[test/], calibration/[test/], results/<mode>/, cache/[test/]

PATHS = ("adc_only", "pureci", "s")                     # the three scales, output-column order
SELECTION_PATHS = ("auto", "s", "pureci", "adc_only")
SELECTION_PREFERENCE = ("s", "pureci", "adc_only")      # "auto": the first reconstructed one
SCALE_PARTICLES = ("pion", "muon", "electron", "mix")   # training samples of the s fit; "mix" pools the hits of
                                                        # the muon, electron and pion runs and fits once
DEFAULT_SCALE_PARTICLE = "pion"                         # default training: the 4 GeV pion run (untagged products)
TOT_SAMPLE_MODES = ("sum8", "sum8_mask2048")
BAD_BAR_RULES = ("used_ends", "used_ends_strict")
UNITS = ("parameters", "events", "linearity", "resolution", "eh", "response", "quality_scan")
UNIT_GROUPS = {"figures": ("linearity", "resolution", "eh"), "checks": ("response", "quality_scan")}
DEFAULT_UNITS = ("parameters", "events", "figures")     # the checks only on request
FIGURE_UNITS = UNIT_GROUPS["figures"]
EVENT_UNITS = tuple(u for u in UNITS if u != "parameters")  # units that need the reconstructed events
SUPPORTS_FROM_SAVED = True    # -from-saved: event csv, figures and checks from the event arrays in
                              # cache/[test/]charge_injection/
SUPPORTS_CLEAN_INPUT = True   # -clean-input: every beam csv (also the s training runs) is the clean dataset

# versions of the algorithms behind the cached products (part of their identity: a new version never reuses
# an old cache).  Bump the matching one when a change alters what the product holds.
CI_MAP_ALGORITHM = "ci-map-v2"           # BLOCK 1 (CI scan -> map)
SCALE_ALGORITHM = "s-v4"                 # BLOCK 2 (no floor, Lei D3; training samples = lists of runs)
EVENT_ALGORITHM = "ci-events-v4"         # BLOCK 3-4 (bad bars by the ends used + clipped end, Lei D4)
PARAMETER_SCHEMA = "ci-parameters-v3"    # cache/[test/]charge_injection/parameters<v>/ (csv + nodes + json)
EVENT_CACHE_SCHEMA = "ci-event-arrays-v2"  # cache/[test/]charge_injection/events<v>/<run>/<csv stem>.*

# Every tunable constant with its default (merged with cfg["charge_injection"], then the CLI).
DEFAULTS = {
    # ---- reconstruction, event selection, outputs (BLOCK 3-5) ----
    "energy_paths": list(PATHS),           # scales reconstructed when no path flag is given
    "selection_path": "s",                 # bad-strip flags of this path select events: s (computed even if not
                                           # requested) | pureci | adc_only | auto
    "bad_bar_rule": "used_ends",           # used_ends (Lei D4): no end the bar energy uses is trusted and an end is
                                           # saturated with an unconvertible TOT | used_ends_strict (no used end
                                           # trusted); see bar_quality_flags
    "quality_filter": True,                # False = keep every reconstructed event (--no-quality-filter)
    "quality_threshold": 0.125,            # selected = bad_strip_fraction <= threshold + tolerance
    "quality_tolerance": 1e-12,            # floating-point tolerance of the cut
    "adc_max_rail": 1023,                  # merge tree: an end is "at the rail" when adc_max == this [ADC]
    "hit_threshold_column": "pedestal_range_with_20percent",  # calibration column of the end hit threshold
                                           # (reconstruction, CI protected gate and s training alike)
    "chunk_rows": 400000,                  # beam-csv rows per chunk
    "output_tag": None,                    # extra variant suffix of every product file (--output-tag)
    "linearity_source": "two_pass",        # electron points of the linearity: two_pass | peak_scan; --fit replaces
                                           # either (the --fit fits, Lei 2026-10-02: linearity_source_used)
    "muon_fit_bins": 180,                  # muon core Gaussian: bins on [q0.001, q0.995]
    "pion_fit_bins": 160,                  # pion core Gaussian (single, Lei 2026-10-02): bins on [q0.001, q0.995]
    "fit_quantiles": [0.001, 0.995],       # muon / pion fit histogram range (quantiles of the energies)
    "display_upper_quantile": 0.999,       # muon / pion page histogram: [0, this quantile]
    "display_min_bins": 60,                # muon / pion page histogram: at least this many bins
    "scan_threshold_grid": [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3,
                            0.35, 0.4, 0.5, 0.6, 0.75, 1.0],   # thresholds of the quality scan (-only quality_scan)
    "scan_shoulder_range_mip": [70.0, 100.0],      # quality scan, 4 GeV evaluation: low-side shoulder [lo, hi)
    "scan_core_range_mip": [105.0, 125.0],         # quality scan, 4 GeV evaluation: peak core [lo, hi)
    # ---- parameters (BLOCK 6) ----
    "scale_particle": DEFAULT_SCALE_PARTICLE,  # training sample(s) of the s fit: pion (default) | muon | electron |
                                           # mix, or a list of them (one s_<sample> each, s = the least borrowing)
    "scale_training_datasets": {"pion": "pion 4", "muon": "muon 4", "electron": "electron 4",
                                "mix": ["pion 4", "muon 4", "electron 4"]},  # training sample -> run reference(s);
                                           # any run, a list (or an entry naming several runs) is pooled
    "refit": False,                        # True = never reuse the cached CI / s parameters
    "scan_files": ["${ci_scan_dir}/scan_DPM0_CALIBRUN_coff14_20220424_220633.csv",
                   "${ci_scan_dir}/scan_DPM1_CALIBRUN_coff14_20220424_220543.csv"],  # CI scans (DPM0, DPM1)
    "channel_map_csv": None,               # null = paths.channel_map
    # ---- channel conventions shared by BLOCK 2 (s training) and BLOCK 3 (reconstruction) ----
    "bar_key_size": 2100,                  # bar key = layer*100 + strip < this; channel uid = end*size + key
    "adc_max_rail_guard": 1020.0,          # adc_max < this = ADC not railed (exact overlap); >= this = near
                                           # rail (reconstruction candidate / bad-strip flags) [ADC]
    "known_bad_bars": [[12, 4], [12, 5], [12, 6], [12, 7]],  # (layer, strip) never TOT-converted
    # ---- BLOCK 1: CI scan -> per-channel CI map (charge injection provided by Erik Wallin) ----
    "tot_sample_mode": "sum8",             # sum8 (sum of the 8 samples) | sum8_mask2048 (2048 samples -> 0 first)
    "tot_spurious_sample": 2048,           # TOT sample value treated as spurious by sum8_mask2048
    "adc_consist": 40.0,                   # max adc_sum8 range of a no-TOT DAC group [ADC]
    "tot_consist_low": 40.0,               # max tot_sum8 range of a TOT group, low cap [TOT counts]
    "tot_consist_high": 20.0,              # max tot_sum8 range of a TOT group, high cap [TOT counts]
    "min_repeats": 3,                      # min injections per DAC group
    "dac_fullscale": 2048,                 # DAC full scale
    "lowcap_fc": 500.0,                    # low-cap full-scale charge [fC]
    "highcap_fc": 8000.0,                  # high-cap full-scale charge [fC]
    "lead_dac_max": 100,                   # high-cap lead fit uses DAC < this
    "min_tot_region_points": 6,            # min TOT-region points for the pole fit
    "pole_asym_points": 5,                 # asymptote fitted on the N highest-q points
    "pole_a_bound": 1.0e5,                 # |A| bound of the pole fit used for the outlier mask
    "pole_b_margin_fc": 10.0,              # B in [0, turn-on - margin] [fC]
    "outlier_resid": 500.0,                # pole outlier: |TOT - pass-1 model| >= this [TOT counts]
    "audit_failures": True,                # run the 1e4-bound audit + TOT=1500 inversion (their failures fail
                                           # the channel, L12 S4-7; --no-audit-failures)
    "pole_a_bound_audit": 1.0e4,           # |A| bound of the audit fit
    "audit_probe_tot": 1500.0,             # audit inversion TOT [counts]
    "audit_grid_n": 4001,                  # audit inversion grid points
    "q_grid_n": 20001,                     # pole inversion grid points (pole_charge_from_tot, diagnostics only)
    "q_max_fc": 16376.0,                   # pole inversion grid end = 8 x 2047 [fC]
    "tot_rail_lo": 4094.0,                 # TOT >= this is railed: never a node [TOT counts]
    "min_reliable_points": 6,              # min reliable TOT-region points
    "min_monotone_nodes": 4,               # min unique monotone nodes
    "pava_tie_tol": 1e-9,                  # PAVA levels closer than this collapse into one node
    "endpoint_slope_points": 5,            # nodes used for the extrapolation slopes
    "endpoint_slope_fallback": 1e-6,       # slope if fit and secant are not positive [fC/count]
    "thr_min_repeats": 3,                  # TOT-threshold calibration: group size cut
    "thr_adc_consist": 40.0,               # TOT-threshold calibration: ADC range cut (no-TOT groups)
    "thr_tot_consist_low": 40.0,           # TOT-threshold calibration: TOT range cut, low cap
    "thr_tot_consist_high": 20.0,          # TOT-threshold calibration: TOT range cut, high cap
    "thr_asym_points": 5,                  # TOT-threshold calibration: asymptote on the last N groups
    "thr_a_bound": 1.0e4,                  # TOT-threshold calibration: |A| bound
    "thr_b_margin_fc": 10.0,               # TOT-threshold calibration: B <= threshold - margin [fC]
    # ---- BLOCK 2: exact-overlap training + hierarchical s fit (s parameter, Lei) ----
    "ratio_max": 50.0,                     # exact-overlap ratio window (0, ratio_max)
    "shrinkage_lambda": 5.0,               # lambda of w = n/(n+lambda), or "cv" (--shrinkage)
    "lambda_grid": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],  # cv grid
    "pointwise_loss_guard": 1.05,          # cv: eligible if pointwise error <= guard x minimum
    "cv_fallback_lambda": 5.0,             # cv: used when no lambda is eligible (low statistics)
    "cv_diagnostics": True,                # compute the cv table also with a fixed lambda (json only)
    "min_link_hits": 0,                    # link prior needs >= this many exact hits, else global (0 = any)
    "training_chunksize": 300000,          # training csv rows per chunk (performance only)
}

# settings that define the cached products (BLOCK 6 cache signatures): the CI map (BLOCK 1) and the s fit
CI_MAP_SETTING_KEYS = (
    "adc_consist", "audit_failures", "audit_grid_n", "audit_probe_tot", "dac_fullscale", "endpoint_slope_fallback",
    "endpoint_slope_points", "highcap_fc", "hit_threshold_column", "lead_dac_max", "lowcap_fc",
    "min_monotone_nodes", "min_reliable_points", "min_repeats", "min_tot_region_points", "outlier_resid",
    "pava_tie_tol", "pole_a_bound", "pole_a_bound_audit", "pole_asym_points", "pole_b_margin_fc", "q_grid_n",
    "q_max_fc", "thr_a_bound", "thr_adc_consist", "thr_asym_points", "thr_b_margin_fc", "thr_min_repeats",
    "thr_tot_consist_high", "thr_tot_consist_low", "tot_consist_high", "tot_consist_low", "tot_rail_lo",
    "tot_sample_mode", "tot_spurious_sample")
SCALE_SETTING_KEYS = ("bar_key_size", "adc_max_rail_guard", "ratio_max", "known_bad_bars", "shrinkage_lambda",
                      "lambda_grid", "pointwise_loss_guard", "cv_fallback_lambda", "cv_diagnostics",
                      "min_link_hits", "training_chunksize")
# settings that define the event arrays of one csv (event cache identity, -from-saved).  The quality cut
# (quality_filter / quality_threshold / quality_tolerance) is NOT among them: the arrays hold the bad-strip
# fraction of every reconstructed event, so -from-saved applies the current cut.  The computed paths are
# checked separately (the cache may hold more paths than a later run asks for).
RECONSTRUCTION_SETTING_KEYS = ("selection_path_used", "bad_bar_rule", "adc_max_rail", "adc_max_rail_guard",
                               "hit_threshold_column", "known_bad_bars", "bar_key_size", "chunk_rows",
                               "scale_particles")
SELECTION_SETTING_KEYS = ("quality_filter", "quality_threshold", "quality_tolerance", "selection_path_used")
assert set(CI_MAP_SETTING_KEYS) | set(SCALE_SETTING_KEYS) <= set(DEFAULTS), "charge_injection: unknown cache key"
assert set(RECONSTRUCTION_SETTING_KEYS) - {"selection_path_used", "scale_particles"} <= set(DEFAULTS), \
    "charge_injection: unknown key"

# beam csv columns read by BLOCK 2 (training) and BLOCK 4 (reconstruction)
READ_COLUMNS = ["pf_event", "layer", "strip", "tot_end0", "adc_sum_end0", "adc_max_end0",
                "tot_end1", "adc_sum_end1", "adc_max_end1"]
READ_DTYPES = {c: "float32" for c in READ_COLUMNS[3:]}  # parse dtype (exact for the integer counts)
READ_DTYPES.update({"pf_event": "int64", "layer": "int16", "strip": "int16"})

# BLOCK 1: injection scan columns, channel status, CI table layout
CI_OK = "CI_OK"
STATUS_PROTECTED = "RAW_MISSING_PROTECTED_CALIBRATION"
STATUS_NO_EVENTS = "RAW_NO_INJECTION_EVENTS"
STATUS_FAILED = "RAW_CI_FAILED"
LOW_CAP, HIGH_CAP = 0, 1
SCAN_ADC_COLUMNS = [f"ADC{i}" for i in range(8)]
SCAN_TOT_COLUMNS = [f"TOT{i}" for i in range(8)]
SCAN_COLUMNS = ["CALIB_DAC", "DPM", "ILINK", "CHAN", *SCAN_ADC_COLUMNS, *SCAN_TOT_COLUMNS, "CAPACITOR_TYPE"]
NODE_COLUMNS = ["layer", "strip", "end", "node_index", "tot_counts", "q_fc"]

# TOT-threshold calibration (provided by Erik Wallin): parameter -> ci_table column
THRESHOLD_COLUMNS = {
    "TOT_threshold": "ci_tot_threshold_fc",
    "tot_asym_slope": "ci_thr_asym_slope_counts_per_fc",
    "tot_asym_intercept": "ci_thr_asym_intercept_counts",
    "tot_param_a": "ci_thr_pole_strength",
    "tot_param_b": "ci_thr_pole_position_fc",
    "adc_fit_slope": "ci_thr_adc_gain_per_fc",
    "adc_fit_intercept": "ci_thr_adc_intercept_adc",
}

CI_TABLE_COLUMNS = [
    "layer", "strip", "end", "dpm", "link", "chan", "ci_status", "ci_reason",
    "ci_k_adc_per_fc", "ci_P_injection", "ci_q_threshold_fc",
    "ci_pole_asym_slope_counts_per_fc", "ci_pole_asym_intercept_counts", "ci_pole_strength", "ci_pole_position_fc",
    "n_ci_measured", "n_ci_reliable", "n_ci_interp_nodes", "n_ci_pava_merged", "n_ci_pole_audit_outliers",
    "n_ci_rail_excluded",
    "mono_tot_min", "mono_tot_max", "mono_q_min_fc", "mono_q_max_fc",
    "mono_slope_low_fc_per_tot", "mono_slope_high_fc_per_tot",
    *THRESHOLD_COLUMNS.values(), "ci_thr_status",
    "n_scan_events", "n_low_groups_kept", "n_high_groups_kept", "n_groups_dropped_repeats",
    "n_groups_dropped_consistency",
]
_INT_COLUMNS = ["n_ci_measured", "n_ci_reliable", "n_ci_interp_nodes", "n_ci_pava_merged",
                "n_ci_pole_audit_outliers", "n_ci_rail_excluded", "n_scan_events", "n_low_groups_kept",
                "n_high_groups_kept", "n_groups_dropped_repeats", "n_groups_dropped_consistency"]

# BLOCK 2: s_source labels of the scale table
SOURCE_DIRECT = "DIRECT_SHRUNK_TO_LINK"
SOURCE_BORROWED = "BORROWED_LINK_OR_GLOBAL"
SOURCE_NO_MODEL = "RAW_NO_MODEL"
SOURCE_NO_CI = "RAW_NO_CI"
SCALE_TABLE_DROP = ("dpm", "link", "ci_status", "cv_json")  # already in the CI table / kept in the json record

# meaning of the main ci_channel_parameters.csv columns (README / diagnostics)
PARAMETER_COLUMN_DOC = {
    "ci_k_adc_per_fc": "low-cap injection gain k of adc_sum8 = k q + P [ADC/fC]",
    "ci_P_injection": "low-cap injection intercept P (8-sample pedestal seen by CI) [ADC]",
    "ci_q_threshold_fc": "TOT turn-on charge of the CI map: min aligned high-cap charge with TOT > 0 [fC]",
    "ci_tot_threshold_fc": "TOT threshold of the channel [fC]: a TOT end needs q(TOT) above it",
    "mono_q_min_fc": "charge of the first monotone node (qnorm origin) [fC]",
    "mono_q_max_fc": "charge of the last monotone node [fC]",
    "mono_slope_high_fc_per_tot": "linear extrapolation slope above the last node [fC/TOT count]",
    "s_final": "TOT scale s: beam ADC over CI-predicted ADC (k q), direct median shrunk to its (DPM, LINK) prior",
}

# BLOCK 5: figure style
PATH_LABELS = {
    "adc_only": "ADC only (no TOT conversion)",
    "pureci": "PureCI (charge injection, s = 1)",
    "s": "charge injection x s",
}
PATH_COLORS = {"s": "#28679A", "pureci": "#D17A22", "adc_only": "#7B838C"}
FIT_COLOR = "#9E2F2A"
INK = "#20252B"
SUBTLE = "#59616A"
GRID = "#D8DDE3"
FWHM_PER_SIGMA_MUON_PION = 2.354820045  # FWHM / sigma of a Gaussian (muon / pion seed width)

_NAN = float("nan")
_VAR = re.compile(r"\$\{([^}]+)\}")


# settings: DEFAULTS overridden by the given ones (partial settings allowed)
def _settings(settings: dict | None) -> dict:
    return {**DEFAULTS, **(settings or {})}


# silence the warnings of the fits (a failure is reported through a status)
@contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


# bad-bar rule of the 1/8 event cut: used_ends or used_ends_strict
def bad_bar_rule(s: dict) -> str:
    rule = str(s.get("bad_bar_rule", "used_ends")).lower()
    if rule not in BAD_BAR_RULES:
        raise ValueError(f"bad_bar_rule must be one of {BAD_BAR_RULES}, not {rule!r}")
    return rule


# path of one training sample of several: s_<sample>
def sample_path(sample: str) -> str:
    return f"s_{sample}"


# is a path an s scale (s or s_<sample>)?
def is_scale_path(path: str) -> bool:
    return path == "s" or str(path).startswith("s_")


# scale of a path for the default linearity fits: adc_only | pureci | s
def path_scale(path: str) -> str:
    return "s" if is_scale_path(path) else path


# figure label of a path
def path_label(path: str) -> str:
    if path in PATH_LABELS:
        return PATH_LABELS[path]
    return f"charge injection x {path} ({path[2:]} training)"


# figure colour of a path
def path_color(path: str) -> str:
    return PATH_COLORS.get(path, PATH_COLORS["s"])


# check the channel keys of a per-channel table (invalid / duplicated keys stop the run)
def checked_channel_table(df: pd.DataFrame, source: str, node_key: str | None = None) -> pd.DataFrame:
    return validate_channel_keys(df, node_key=node_key, source=source)


# check a q(TOT) node table: unique keys, finite values, TOT increasing with node_index
def check_node_table(nodes: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in NODE_COLUMNS if c not in nodes.columns]
    if missing:
        raise ChannelKeyError(f"{source}: missing node column(s) {missing}")
    out = checked_channel_table(nodes[NODE_COLUMNS], source, node_key="node_index")
    tot = pd.to_numeric(out["tot_counts"], errors="coerce").to_numpy(float)
    q = pd.to_numeric(out["q_fc"], errors="coerce").to_numpy(float)
    bad = ~(np.isfinite(tot) & np.isfinite(q))
    if bad.any():
        i = int(np.argmax(bad))
        raise ChannelKeyError(f"{source}: {int(bad.sum())} node(s) with a missing / non-finite tot_counts or q_fc, "
                              f"first at row {i} (L{out['layer'].iat[i]} S{out['strip'].iat[i]} E{out['end'].iat[i]})")
    out["tot_counts"], out["q_fc"] = tot, q
    ordered = out.sort_values(KEY_COLUMNS + ["node_index"], kind="stable")
    same = (ordered[KEY_COLUMNS].to_numpy()[1:] == ordered[KEY_COLUMNS].to_numpy()[:-1]).all(axis=1)
    step = np.diff(ordered["tot_counts"].to_numpy(float))
    wrong = same & ~(step > 0)
    if wrong.any():
        row = ordered.iloc[int(np.argmax(wrong)) + 1]
        raise ChannelKeyError(f"{source}: tot_counts must increase with node_index in every channel; "
                              f"L{row['layer']} S{row['strip']} E{row['end']} node {row['node_index']} breaks it")
    return out


# bar keys layer*100 + strip of the known-bad bars (never TOT converted)
def known_bad_keys(s: dict) -> np.ndarray:
    return np.array([int(l) * 100 + int(st) for l, st in (s["known_bad_bars"] or [])], dtype=np.int64)


# per-channel values as a dense [end, layer*100 + strip] lookup (NaN where absent)
def channel_lut(df: pd.DataFrame | None, values, bar_key_size: int) -> np.ndarray:
    arr = np.full((2, bar_key_size), np.nan)
    if df is None or not len(df):
        return arr
    key = df["layer"].to_numpy(np.int64) * 100 + df["strip"].to_numpy(np.int64)
    end = df["end"].to_numpy(np.int64)
    ok = (key >= 0) & (key < bar_key_size) & ((end == 0) | (end == 1))
    arr[end[ok], key[ok]] = np.asarray(values, dtype=float)[ok]
    return arr


# ==========================================================================
# BLOCK 1: charge-injection calibration (injection scans -> per-channel CI map); charge injection provided by
# Erik Wallin
# ==========================================================================
# Per electronics channel:
#   1. scan parsing: per injection adc_sum8 = sum of the 8 ADC samples, tot_sum8 = sum of the 8 TOT samples
#      (tot_sample_mode=sum8_mask2048 zeroes the spurious 2048 samples of the DPM0 scan first);
#   2. repeatability filter per (capacitor, DAC) group: >= 3 injections and (no TOT and ADC range < 40)
#      or (TOT and TOT range < 40 low cap / 20 high cap);
#   3. DAC -> fC (2048 DAC = 500 fC low cap, 8000 fC high cap);
#   4. low-cap linear fit adc_sum8 = k q + P (k = ci_k_adc_per_fc, P = ci_P_injection);
#   5. high-cap charge aligned on the low-cap scale with the DAC < 100 lead fit;
#   6. TOT turn-on charge ci_q_threshold_fc = min aligned charge with TOT > 0;
#   7. TOT-region points (q >= turn-on, sorted by q);
#   8. two-pass pole fit TOT = m q + c0 - A/(q - B) (asymptote from the 5 highest-q points, |A| <= 1e5,
#      0 <= B <= turn-on - 10 fC, outliers |resid| >= 500 counts); the 1e4-bound audit fit and its TOT=1500
#      inversion fail a channel when they fail (L12 S4-7; audit_failures);
#   9. monotone measured-point map q(TOT): reliable points (0 < TOT < 4094, not a pole outlier) -> PAVA ->
#      collapse of equal levels -> PCHIP, linear extrapolation with positive endpoint slopes (first/last
#      5 nodes), q >= 0.
# The TOT-threshold calibration (threshold_injection_parameters, the 7 ci_tot_threshold_fc / ci_thr_* columns)
# gives every channel its TOT threshold [fC]: a TOT end counts only when q(TOT) is above it.
# Channel status: RAW_MISSING_PROTECTED_CALIBRATION (no TOT threshold, or no pedestal / hit threshold / mpv
# when a calibration table is given), RAW_NO_INJECTION_EVENTS, RAW_CI_FAILED ("<Exception>: <message>"), CI_OK.
# Entry points: calibrate_charge_injection(scan_files, channel_map_csv, settings, calibration_df=None)
# -> (ci_table, nodes); build_tot_charge_mappers(nodes) -> mappers; q_from_tot(tot, layer, strip, end,
# mappers) -> charge [fC] (NaN without a map); threshold_parameter_table(ci_table); pole_charge_from_tot.

# read the CI scans (provided by Erik Wallin): adc_sum8, tot_sum8 of every injection per channel
def parse_scan_files(scan_files, settings: dict | None = None) -> dict:
    """A channel present in several files keeps the events of the first file."""
    s = _settings(settings)
    mode = str(s["tot_sample_mode"]).lower()
    if mode not in TOT_SAMPLE_MODES:
        raise ValueError(f"tot_sample_mode must be sum8 or sum8_mask2048, not {mode!r}")
    channels: dict[tuple[int, int, int], dict] = {}
    for path in scan_files:
        df = pd.read_csv(path, usecols=SCAN_COLUMNS)
        if not all(pd.api.types.is_integer_dtype(df[c]) for c in SCAN_COLUMNS):
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
        adc = df[SCAN_ADC_COLUMNS].to_numpy(np.int64).sum(axis=1).astype(float)
        samples = df[SCAN_TOT_COLUMNS].to_numpy(np.int64)
        tot_raw = samples.sum(axis=1).astype(float)
        tot = tot_raw
        if mode == "sum8_mask2048":
            tot = np.where(samples == int(s["tot_spurious_sample"]), 0, samples).sum(axis=1).astype(float)
        dac = df["CALIB_DAC"].to_numpy(np.int64)
        cap = df["CAPACITOR_TYPE"].to_numpy(np.int64)
        keys = df[["DPM", "ILINK", "CHAN"]].to_numpy(np.int64)
        code = (keys[:, 0] * 1000 + keys[:, 1]) * 1000 + keys[:, 2]
        order = np.argsort(code, kind="stable")  # stable: file order inside a channel
        starts = np.flatnonzero(np.r_[True, code[order][1:] != code[order][:-1]]) if len(order) else []
        bounds = np.r_[starts, len(order)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows = order[a:b]
            key = tuple(int(v) for v in keys[rows[0]])
            if key in channels:
                log(f"ci scan: channel {key} also in {Path(path).name}; keeping the first file's events")
                continue
            channels[key] = {"calib_dac": dac[rows], "cap": cap[rows], "adc_sum8": adc[rows], "tot_sum8": tot[rows],
                             "tot_sum8_raw": tot_raw[rows]}
    return channels


# group the injections of one capacitor by DAC (mean and range of adc_sum8, tot_sum8)
def _dac_groups(dac: np.ndarray, adc: np.ndarray, tot: np.ndarray) -> dict:
    if dac.size == 0:
        z = np.zeros(0)
        return {"calib_dac": np.zeros(0, np.int64), "n": np.zeros(0, np.int64), "adc_mean": z,
                "adc_range": z, "tot_mean": z, "tot_range": z, "tot_max": z}
    order = np.argsort(dac, kind="stable")
    sd = dac[order]
    starts = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1]])
    n = np.diff(np.r_[starts, sd.size])
    a, t = adc[order], tot[order]
    appearance = np.argsort(order[starts], kind="stable")
    # integer-valued sums are exact, so sum/n is the exact group mean
    groups = {
        "calib_dac": sd[starts],
        "n": n,
        "adc_mean": np.add.reduceat(a, starts) / n,
        "adc_range": np.maximum.reduceat(a, starts) - np.minimum.reduceat(a, starts),
        "tot_mean": np.add.reduceat(t, starts) / n,
        "tot_range": np.maximum.reduceat(t, starts) - np.minimum.reduceat(t, starts),
        "tot_max": np.maximum.reduceat(t, starts),
    }
    return {k: v[appearance] for k, v in groups.items()}


# keep DAC groups with >= 3 injections and ADC range < 40 (no TOT) or TOT range < 40 / 20
def _repeatability_filter(groups: dict, tot_consist: float, s: dict):
    enough = groups["n"] >= int(s["min_repeats"])
    passed = enough & (((groups["tot_max"] == 0) & (groups["adc_range"] < float(s["adc_consist"])))
                       | ((groups["tot_max"] > 0) & (groups["tot_range"] < float(tot_consist))))
    return passed, int(np.count_nonzero(~enough)), int(np.count_nonzero(enough & ~passed))


# pole model TOT = m q + c0 - A / (q - B)
def _pole_model(q, A, B, m, c0):
    return m * q + c0 - A / (q - B)


# two-pass pole fit, asymptote fixed: pass 1 flags |TOT - model| >= 500 as outliers, pass 2 refits
def _two_pass_pole(q_sorted: np.ndarray, tot_sorted: np.ndarray, q_threshold: float, a_bound: float,
                   s: dict) -> dict:
    n_asym = int(s["pole_asym_points"])
    b_max = q_threshold - float(s["pole_b_margin_fc"])

    # asymptote TOT = m q + c0 on the highest-q points
    def fit_asym(qs, ts):
        m_, c0_ = np.polyfit(qs[-n_asym:], ts[-n_asym:], 1)
        return float(m_), float(c0_)  # Python floats: the audit divides by m in Python arithmetic

    m, c0 = fit_asym(q_sorted, tot_sorted)
    m1, c01 = m, c0
    (A1, B1), _ = curve_fit(lambda q, A, B: _pole_model(q, A, B, m, c0), q_sorted, tot_sorted,
                            bounds=([-a_bound, 0], [a_bound, b_max]))
    resid = tot_sorted - _pole_model(q_sorted, A1, B1, m, c0)
    outlier = np.abs(resid) >= float(s["outlier_resid"])
    inlier = ~outlier
    q_in, t_in = q_sorted[inlier], tot_sorted[inlier]
    if outlier[-n_asym:].any():
        m, c0 = fit_asym(q_in, t_in)
    (A2, B2), _ = curve_fit(lambda q, A, B: _pole_model(q, A, B, m, c0), q_in, t_in,
                            bounds=([-a_bound, 0], [a_bound, b_max]))
    return {"m": m, "c0": c0, "A": float(A2), "B": float(B2), "m1": m1, "c01": c01,
            "A1": float(A1), "B1": float(B1), "outlier_mask": outlier,
            "railed": bool(abs(abs(A2) - a_bound) < 1e-3 * a_bound)}


# invert the pole fit at one TOT (audit)
def _charge_at_tot(fit: dict, q_threshold: float, tot_value: float, s: dict) -> float:
    """Raises ZeroDivisionError when m == 0 above the grid."""
    qg = np.linspace(q_threshold, float(s["q_max_fc"]), int(s["audit_grid_n"]))
    tg = _pole_model(qg, fit["A"], fit["B"], fit["m"], fit["c0"])
    nonmono = np.where(np.diff(tg) <= 0)[0]
    if nonmono.size:
        cut = int(nonmono.max()) + 1
        qg, tg = qg[cut:], tg[cut:]
    if tot_value > tg[-1]:
        return float((tot_value - fit["c0"]) / fit["m"])
    return float(np.interp(tot_value, tg, qg))


# CI calibration of one channel: low cap adc_sum8 = k q + P, turn-on = min q with TOT > 0, pole fit
def injection_calibration(events: dict, settings: dict | None = None) -> dict:
    """Raises on a failing channel (the caller marks it RAW_CI_FAILED)."""
    s = _settings(settings)
    cap = events["cap"]
    groups, kept = {}, {}
    drops_rep = drops_con = 0
    for name, cap_value, consist in (("low", LOW_CAP, s["tot_consist_low"]), ("high", HIGH_CAP, s["tot_consist_high"])):
        sel = cap == cap_value
        g = _dac_groups(events["calib_dac"][sel], events["adc_sum8"][sel], events["tot_sum8"][sel])
        passed, n_rep, n_con = _repeatability_filter(g, consist, s)
        groups[name] = g
        kept[name] = passed
        drops_rep += n_rep
        drops_con += n_con
    low, high = groups["low"], groups["high"]
    kl, kh = kept["low"], kept["high"]
    if np.count_nonzero(kl) < 2 or np.count_nonzero(kh) < 2:
        raise RuntimeError("too few surviving injection groups to calibrate")

    fullscale = int(s["dac_fullscale"])
    q_low = low["calib_dac"][kl] / fullscale * float(s["lowcap_fc"])
    adc_low = low["adc_mean"][kl]
    slope, intercept = np.polyfit(q_low, adc_low, 1)  # adc_sum8 = k q + P (low cap)
    k, P = float(slope), float(intercept)

    dac_high = high["calib_dac"][kh]
    q_nominal = dac_high / fullscale * float(s["highcap_fc"])
    adc_high = high["adc_mean"][kh]
    tot_high = high["tot_mean"][kh]
    lead = dac_high < int(s["lead_dac_max"])
    if np.count_nonzero(lead) < 2:
        raise RuntimeError("too few high-cap leading points for charge alignment")
    lead_a, lead_b = np.polyfit(q_nominal[lead], adc_high[lead], 1)
    q_high = lead_a / k * q_nominal + (lead_b - P) / k

    fired = tot_high > 0
    if not fired.any():
        raise ValueError("min() iterable argument is empty")
    q_threshold = float(q_high[fired].min())  # turn-on = min q with TOT > 0

    region = q_high >= q_threshold
    order = np.argsort(q_high[region], kind="stable")
    q_reg = q_high[region][order]
    t_reg = tot_high[region][order]
    if q_reg.size < int(s["min_tot_region_points"]):
        raise RuntimeError("too few TOT-region points for pole fit")

    fit = _two_pass_pole(q_reg, t_reg, q_threshold, float(s["pole_a_bound"]), s)
    audit = {}
    if s["audit_failures"]:
        fit_audit = _two_pass_pole(q_reg, t_reg, q_threshold, float(s["pole_a_bound_audit"]), s)
        probe = float(s["audit_probe_tot"])
        audit = {"A_audit": fit_audit["A"], "B_audit": fit_audit["B"], "railed_audit": fit_audit["railed"],
                 "q_probe": _charge_at_tot(fit, q_threshold, probe, s),
                 "q_probe_audit": _charge_at_tot(fit_audit, q_threshold, probe, s)}
    return {
        "k": k, "P": P, "q_threshold": q_threshold,
        "m": fit["m"], "c0": fit["c0"], "A": fit["A"], "B": fit["B"],
        "m1": fit["m1"], "c01": fit["c01"], "A1": fit["A1"], "B1": fit["B1"], "railed": fit["railed"],
        "q_tot_region": q_reg, "tot_tot_region": t_reg, "outlier_mask": fit["outlier_mask"],
        "q_low": q_low, "adc_low": adc_low, "q_high": q_high, "adc_high": adc_high, "tot_high": tot_high,
        "lead_a": float(lead_a), "lead_b": float(lead_b),
        "n_low_groups_kept": int(np.count_nonzero(kl)), "n_high_groups_kept": int(np.count_nonzero(kh)),
        "n_groups_dropped_repeats": drops_rep, "n_groups_dropped_consistency": drops_con,
        "audit": audit,
    }


# weighted pool-adjacent-violators: non-decreasing fit of a sequence
def pava_increasing(values, weights=None) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    levels: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, (value, mass) in enumerate(zip(y, w)):
        levels.append(float(value))
        masses.append(float(mass))
        starts.append(i)
        ends.append(i + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_mass = masses[-2] + masses[-1]
            merged_level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / merged_mass
            levels[-2:] = [merged_level]
            masses[-2:] = [merged_mass]
            ends[-2:] = [ends[-1]]
            starts.pop()
    out = np.empty_like(y)
    for level, start, end in zip(levels, starts, ends):
        out[start:end] = level
    return out


# q(TOT) nodes: reliable points (0 < TOT < 4094, no pole outlier) -> PAVA -> equal levels merged
def monotone_nodes(cal: dict, settings: dict | None = None) -> dict:
    s = _settings(settings)
    q_all = np.asarray(cal["q_tot_region"], dtype=float)
    t_all = np.asarray(cal["tot_tot_region"], dtype=float)
    outlier = np.asarray(cal["outlier_mask"], dtype=bool)
    rail = float(s["tot_rail_lo"])
    reliable = np.isfinite(q_all) & np.isfinite(t_all) & (t_all > 0) & (t_all < rail) & ~outlier
    q, t = q_all[reliable], t_all[reliable]
    order = np.argsort(q, kind="stable")
    q, t = q[order], t[order]
    if len(q) < int(s["min_reliable_points"]):
        raise RuntimeError(f"too few reliable measured TOT nodes ({len(q)})")
    tol = float(s["pava_tie_tol"])
    t_iso = pava_increasing(t)
    t_nodes, q_nodes = [], []
    start = 0
    for i in range(1, len(t_iso) + 1):
        if i == len(t_iso) or abs(t_iso[i] - t_iso[start]) > tol:
            t_nodes.append(float(np.mean(t_iso[start:i])))
            q_nodes.append(float(np.mean(q[start:i])))
            start = i
    tn = np.asarray(t_nodes, dtype=float)
    qn = np.asarray(q_nodes, dtype=float)
    strict = np.r_[True, np.diff(tn) > tol]
    tn, qn = tn[strict], qn[strict]
    if len(tn) < int(s["min_monotone_nodes"]):
        raise RuntimeError(f"too few unique monotone nodes ({len(tn)})")
    return {"tot_nodes": tn, "q_nodes": qn, "n_measured": int(len(q_all)),
            "n_reliable": int(np.count_nonzero(reliable)), "n_merged": int(len(t_iso) - len(tn)),
            "n_outliers": int(np.count_nonzero(outlier)), "n_rail": int(np.count_nonzero(t_all >= rail))}


# extrapolation slope dq/dTOT of the first / last nodes (fit, else secant, else fallback)
def positive_endpoint_slope(t: np.ndarray, q: np.ndarray, side: str, n_points: int = 5,
                            fallback: float = 1e-6) -> float:
    n = min(int(n_points), len(t))
    sl = slice(0, n) if side == "low" else slice(len(t) - n, len(t))
    slope = float(np.polyfit(t[sl], q[sl], 1)[0])
    if not np.isfinite(slope) or slope <= 0:
        slope = float((q[sl][-1] - q[sl][0]) / max(t[sl][-1] - t[sl][0], 1e-9))
    if not np.isfinite(slope) or slope <= 0:
        slope = float(fallback)
    return slope


# q(TOT) map of one channel: PCHIP through the nodes, linear beyond them, q >= 0
class MonotoneChargeMap:

    __slots__ = ("pchip", "t_lo", "t_hi", "q_lo", "q_hi", "slope_low", "slope_high")

    # build the PCHIP and the two end slopes
    def __init__(self, tot_nodes, q_nodes, settings: dict | None = None):
        s = _settings(settings)
        t = np.asarray(tot_nodes, dtype=float)
        q = np.asarray(q_nodes, dtype=float)
        self.pchip = PchipInterpolator(t, q, extrapolate=False)
        n_pts, fb = int(s["endpoint_slope_points"]), float(s["endpoint_slope_fallback"])
        self.slope_low = positive_endpoint_slope(t, q, "low", n_pts, fb)
        self.slope_high = positive_endpoint_slope(t, q, "high", n_pts, fb)
        self.t_lo, self.t_hi = float(t[0]), float(t[-1])
        self.q_lo, self.q_hi = float(q[0]), float(q[-1])

    # charge [fC] of TOT values
    def __call__(self, tot):
        scalar = np.isscalar(tot)
        t = np.atleast_1d(np.asarray(tot, dtype=float))
        q = np.asarray(self.pchip(np.clip(t, self.t_lo, self.t_hi)), dtype=float)
        low = t < self.t_lo
        high = t > self.t_hi
        q[low] = self.q_lo + self.slope_low * (t[low] - self.t_lo)
        q[high] = self.q_hi + self.slope_high * (t[high] - self.t_hi)
        q = np.maximum(q, 0.0)
        return float(q[0]) if scalar else q


# q(TOT) from the pole fit (diagnostics only; the map is MonotoneChargeMap)
def pole_charge_from_tot(tot, m: float, c0: float, A: float, B: float, q_threshold: float,
                         settings: dict | None = None) -> np.ndarray:
    s = _settings(settings)
    q_grid = np.linspace(q_threshold, float(s["q_max_fc"]), int(s["q_grid_n"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        tot_grid = _pole_model(q_grid, A, B, m, c0)
    nonmono = np.where(np.diff(tot_grid) <= 0)[0]
    if nonmono.size:
        cut = int(nonmono.max()) + 1
        q_grid, tot_grid = q_grid[cut:], tot_grid[cut:]
    t = np.atleast_1d(np.asarray(tot, dtype=float))
    q = np.interp(t, tot_grid, q_grid)
    high = t > tot_grid[-1]
    if np.any(high):
        with np.errstate(divide="ignore", invalid="ignore"):
            q[high] = (t[high] - c0) / m
    return q


# TOT-threshold calibration of one channel (7 parameters, provided by Erik Wallin): TOT_threshold = min q, TOT > 0
def threshold_injection_parameters(events: dict, settings: dict | None = None) -> dict:
    """As provided: DAC groups in Python set order, no outlier rejection."""
    s = _settings(settings)
    min_rep = int(s["thr_min_repeats"])
    adc_consist = float(s["thr_adc_consist"])

    # group filter: mean adc_sum8, tot_sum8 of the kept DAC groups
    def group_filter(sel, tot_consist):
        dac, adc = events["calib_dac"][sel], events["adc_sum8"][sel]
        tot = events.get("tot_sum8_raw", events["tot_sum8"])[sel]
        out_dac, out_adc, out_tot = [], [], []
        for cd in set(dac.tolist()):  # Python set order of the DAC values, as provided
            member = dac == cd
            n = int(np.count_nonzero(member))
            if n < min_rep:
                continue
            sadcs, tots = adc[member], tot[member]
            if (sadcs.max() - sadcs.min() < adc_consist and tots.max() == 0) or \
                    (tots.max() - tots.min() < tot_consist):
                out_dac.append(cd)
                out_adc.append(sadcs.sum() / n)
                out_tot.append(tots.sum() / n)
        return np.asarray(out_dac, dtype=np.int64), np.asarray(out_adc, float), np.asarray(out_tot, float)

    fullscale = int(s["dac_fullscale"])
    dac_l, adc_l, _ = group_filter(events["cap"] == LOW_CAP, float(s["thr_tot_consist_low"]))
    dac_h, adc_h, tot_h = group_filter(events["cap"] == HIGH_CAP, float(s["thr_tot_consist_high"]))
    q_l = dac_l / fullscale * float(s["lowcap_fc"])
    q_h = dac_h / fullscale * float(s["highcap_fc"])
    c, d = np.polyfit(q_l, adc_l, 1)
    lead = dac_h < int(s["lead_dac_max"])
    a, b = np.polyfit(q_h[lead], adc_h[lead], 1)
    q_h = a / c * q_h + (b - d) / c
    fired = tot_h > 0
    if not fired.any():
        raise ValueError("min() iterable argument is empty")
    threshold = float(q_h[fired].min())
    region = q_h >= threshold
    xval, yval = q_h[region], tot_h[region]
    n_asym = int(s["thr_asym_points"])
    a_bound = float(s["thr_a_bound"])

    # straight line
    def lin(x, p0, p1):
        return p0 * x + p1

    popt, _ = curve_fit(lin, xval[-n_asym:], yval[-n_asym:])
    ppole, _ = curve_fit(lambda x, A, B: _pole_model(x, A, B, *popt), xval, yval,
                         bounds=([-a_bound, 0], [a_bound, threshold - float(s["thr_b_margin_fc"])]))
    return {"TOT_threshold": threshold, "tot_asym_slope": float(popt[0]), "tot_asym_intercept": float(popt[1]),
            "tot_param_a": float(ppole[0]), "tot_param_b": float(ppole[1]),
            "adc_fit_slope": float(c), "adc_fit_intercept": float(d)}


# read the channel map: (DPM, LINK, CHAN) -> layer, strip, end (both addresses unique)
def load_channel_map(channel_map_csv) -> pd.DataFrame:
    m = pd.read_csv(channel_map_csv).rename(columns={"PLANE": "layer", "STRIP": "strip", "END": "end",
                                                     "DPM": "dpm", "LINK": "link", "CHAN": "chan"})
    cols = ["layer", "strip", "end", "dpm", "link", "chan"]
    missing = [c for c in cols if c not in m.columns]
    if missing:
        raise ChannelKeyError(f"{channel_map_csv}: missing column(s) {missing} (DPM,LINK,CHAN,PLANE,STRIP,END)")
    m = checked_channel_table(m[cols], f"channel map {channel_map_csv}")
    elec = m[["dpm", "link", "chan"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not (np.isfinite(elec).all() and (elec == np.floor(elec)).all() and (elec >= 0).all()):
        raise ChannelKeyError(f"channel map {channel_map_csv}: DPM / LINK / CHAN must be integers >= 0")
    m[["dpm", "link", "chan"]] = elec.astype(np.int64)
    dup = m.duplicated(["dpm", "link", "chan"], keep=False).to_numpy()
    if dup.any():
        row = m.iloc[int(np.argmax(dup))]
        raise ChannelKeyError(f"channel map {channel_map_csv}: {int(dup.sum())} rows share an electronics address "
                              f"(DPM, LINK, CHAN), first ({row['dpm']}, {row['link']}, {row['chan']})")
    return m.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


# pedestal, hit threshold and mpv of the calibration table, aligned with the CI rows
def _calibration_value_arrays(ci: pd.DataFrame, calibration_df: pd.DataFrame | None,
                              hit_column: str = "pedestal_range_with_20percent") -> dict:
    out = {name: np.full(len(ci), np.nan) for name in ("pedestal", "hit_threshold", "mpv")}
    if calibration_df is None:
        return out
    pairs = [(src, dst) for src, dst in (("pedestal", "pedestal"), (hit_column, "hit_threshold"), ("mpv", "mpv"))
             if src in calibration_df.columns]
    cal = checked_channel_table(calibration_df[KEY_COLUMNS + list(dict.fromkeys(src for src, _ in pairs))],
                                "calibration table")
    merged = ci[KEY_COLUMNS].merge(cal, on=KEY_COLUMNS, how="left")
    for src, dst in pairs:
        out[dst] = pd.to_numeric(merged[src], errors="coerce").to_numpy(float)
    return out


# one channel of the CI table: TOT threshold, protected gate, then CI calibration and q(TOT) nodes
def _calibrate_channel(channel: dict, events: dict | None, beam: dict | None,
                       settings: dict) -> tuple[dict, pd.DataFrame | None]:
    s = settings
    rec = dict(channel)
    rec["n_scan_events"] = 0 if events is None else int(len(events["calib_dac"]))
    with _quiet():
        # TOT-threshold calibration from the scan
        if events is None:
            rec.update({col: np.nan for col in THRESHOLD_COLUMNS.values()}, ci_thr_status="NO_INJECTION_EVENTS")
        else:
            try:
                thr = threshold_injection_parameters(events, s)
                rec.update({THRESHOLD_COLUMNS[k]: v for k, v in thr.items()}, ci_thr_status="OK")
            except Exception as exc:  # noqa: BLE001 - any failure = no TOT threshold for the channel
                rec.update({col: np.nan for col in THRESHOLD_COLUMNS.values()},
                           ci_thr_status=f"{type(exc).__name__}: {exc}")

        missing = []
        if beam is not None:  # "protected calibration" gate, checked before the CI
            if not np.isfinite(beam["pedestal"]):
                missing.append("pedestal")
            if not np.isfinite(beam["hit_threshold"]):
                missing.append("hit_threshold")
            if not (np.isfinite(beam["mpv"]) and beam["mpv"] > 0):
                missing.append("mpv")
        if not np.isfinite(rec["ci_tot_threshold_fc"]):
            missing.append("ci_tot_threshold_fc")
        if missing:
            rec.update(ci_status=STATUS_PROTECTED,
                       ci_reason="missing pedestal/hit threshold/mpv/TOT threshold: " + ",".join(missing))
            return rec, None
        if events is None:
            rec.update(ci_status=STATUS_NO_EVENTS, ci_reason="electronics channel absent from injection scan")
            return rec, None
        try:
            cal = injection_calibration(events, s)
            mono = monotone_nodes(cal, s)
            mapper = MonotoneChargeMap(mono["tot_nodes"], mono["q_nodes"], s)
        except Exception as exc:  # noqa: BLE001 - any exception fails the channel
            rec.update(ci_status=STATUS_FAILED, ci_reason=f"{type(exc).__name__}: {exc}")
            return rec, None
    tn, qn = mono["tot_nodes"], mono["q_nodes"]
    rec.update(
        ci_status=CI_OK, ci_reason="",
        ci_k_adc_per_fc=cal["k"], ci_P_injection=cal["P"], ci_q_threshold_fc=cal["q_threshold"],
        ci_pole_asym_slope_counts_per_fc=cal["m"], ci_pole_asym_intercept_counts=cal["c0"],
        ci_pole_strength=cal["A"], ci_pole_position_fc=cal["B"],
        n_ci_measured=mono["n_measured"], n_ci_reliable=mono["n_reliable"],
        n_ci_interp_nodes=int(len(tn)), n_ci_pava_merged=mono["n_merged"],
        n_ci_pole_audit_outliers=mono["n_outliers"], n_ci_rail_excluded=mono["n_rail"],
        mono_tot_min=float(tn[0]), mono_tot_max=float(tn[-1]),
        mono_q_min_fc=float(qn[0]), mono_q_max_fc=float(qn[-1]),
        mono_slope_low_fc_per_tot=mapper.slope_low, mono_slope_high_fc_per_tot=mapper.slope_high,
        n_low_groups_kept=cal["n_low_groups_kept"], n_high_groups_kept=cal["n_high_groups_kept"],
        n_groups_dropped_repeats=cal["n_groups_dropped_repeats"],
        n_groups_dropped_consistency=cal["n_groups_dropped_consistency"])
    nodes = pd.DataFrame({"layer": rec["layer"], "strip": rec["strip"], "end": rec["end"],
                          "node_index": np.arange(len(tn)), "tot_counts": tn, "q_fc": qn})
    return rec, nodes


# CI calibration of every channel of the map -> (CI table, q(TOT) nodes)
def calibrate_charge_injection(scan_files, channel_map_csv, settings: dict | None,
                               calibration_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A channel without pedestal / hit threshold / mpv is RAW_MISSING_PROTECTED_CALIBRATION."""
    s = _settings(settings)
    t0 = time.time()
    scan = parse_scan_files(scan_files, s)
    ci = load_channel_map(channel_map_csv)
    beam = _calibration_value_arrays(ci, calibration_df, s["hit_threshold_column"])
    results = []
    for i, ch in enumerate(ci.to_dict("records")):
        beam_i = None if calibration_df is None else {k: float(v[i]) for k, v in beam.items()}
        results.append(_calibrate_channel(ch, scan.get((ch["dpm"], ch["link"], ch["chan"])), beam_i, s))

    table = pd.DataFrame([rec for rec, _ in results]).reindex(columns=CI_TABLE_COLUMNS)
    for c in _INT_COLUMNS:
        table[c] = table[c].fillna(0).astype(np.int64)
    table["ci_reason"] = table["ci_reason"].fillna("")
    node_parts = [n for _, n in results if n is not None]
    nodes = (pd.concat(node_parts, ignore_index=True) if node_parts
             else pd.DataFrame({c: pd.Series(dtype=float) for c in NODE_COLUMNS}))
    counts = table["ci_status"].value_counts().to_dict()
    log(f"ci calibration: {len(table)} channels {counts}, {len(nodes)} nodes, "
        f"{np.count_nonzero(np.isfinite(table['ci_tot_threshold_fc']))} TOT thresholds "
        f"[{time.time() - t0:.1f}s]")
    return table, nodes[NODE_COLUMNS]


# the TOT-threshold calibration in its 7-parameter layout (provided by Erik Wallin)
def threshold_parameter_table(ci_table: pd.DataFrame) -> pd.DataFrame:
    back = {v: k for k, v in THRESHOLD_COLUMNS.items()}
    out = ci_table[KEY_COLUMNS + list(back)].rename(columns=back)
    return out[np.isfinite(out["TOT_threshold"].to_numpy(float))].reset_index(drop=True)


# --------------------------------------------------------------------------
# BLOCK 1 (cont.): TOT -> charge maps (used by BLOCK 2 and BLOCK 3)
# --------------------------------------------------------------------------
# q(TOT) map of every channel from the node table
def build_tot_charge_mappers(nodes: pd.DataFrame, settings: dict | None = None) -> dict:
    s = _settings(settings)
    kmax = int(s["bar_key_size"])
    maps: dict[int, MonotoneChargeMap] = {}
    if nodes is not None and len(nodes):
        for (layer, strip, end), g in nodes.groupby(KEY_COLUMNS, sort=False):
            g = g.sort_values("node_index")
            key = int(layer) * 100 + int(strip)
            if not 0 <= key < kmax:
                continue
            maps[int(end) * kmax + key] = MonotoneChargeMap(g["tot_counts"].to_numpy(float),
                                                            g["q_fc"].to_numpy(float), s)
    has_map = np.zeros(2 * kmax, dtype=bool)
    if maps:
        has_map[np.fromiter(maps.keys(), dtype=np.int64)] = True
    return {"maps": maps, "has_map": has_map, "bar_key_size": kmax}


# channel uid = end * size + layer*100 + strip
def channel_uid(layer, strip, end, bar_key_size: int = 2100) -> np.ndarray:
    key = np.clip(np.asarray(layer, dtype=np.int64) * 100 + np.asarray(strip, dtype=np.int64), 0, bar_key_size - 1)
    return np.asarray(end, dtype=np.int64) * bar_key_size + key


# charge [fC] of TOT values through the map of each channel uid (NaN without a map)
def _q_from_uid(tot: np.ndarray, uid: np.ndarray, mappers: dict) -> np.ndarray:
    out = np.full(tot.shape, np.nan)
    has_map = mappers["has_map"]
    valid = (uid >= 0) & (uid < has_map.size)
    idx = np.flatnonzero(valid)
    idx = idx[has_map[uid[idx]]]
    if not idx.size:
        return out
    u = uid[idx]
    order = np.argsort(u, kind="stable")
    su = u[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    bounds = np.r_[starts, su.size]
    maps = mappers["maps"]
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = idx[order[a:b]]
        out[rows] = maps[int(su[a])](tot[rows])
    return out


# charge [fC] of TOT values: q = q(TOT) of the channel's map (NaN without a map)
def q_from_tot(tot, layer, strip, end, mappers: dict) -> np.ndarray:
    tot = np.atleast_1d(np.asarray(tot, dtype=float))
    layer_b, strip_b, end_b = (np.broadcast_to(np.asarray(v), tot.shape) for v in (layer, strip, end))
    uid = channel_uid(layer_b, strip_b, end_b, mappers["bar_key_size"])
    bad_end = (end_b != 0) & (end_b != 1)
    uid = np.where(bad_end, -1, uid)
    return _q_from_uid(tot, uid, mappers)


# ==========================================================================
# BLOCK 2: s scale fit (exact-overlap hits of the training runs -> per-channel TOT scale; s parameter, Lei)
# ==========================================================================
# Hierarchical exact-overlap estimator.  Exact-overlap hit = kept end (adc_sum >= hit threshold, finite pedestal,
# mpv > 0) whose CI-map charge q(TOT) > ci_tot_threshold_fc (both fC, Lei D2), CI map present, not L12 S4-7,
# adc_max < 1020 (ADC not railed); ratio = (adc_sum - pedestal) / (k q(TOT)) in (0, 50), rounded to float32.
# s_direct = median ratio of the channel; prior = geometric median of s_direct over its (DPM, LINK) (global
# geometric median when the link has none); s = exp(w ln s_direct + (1 - w) ln prior), w = n / (n + lambda);
# channels without exact hits borrow the prior.  lambda = 5 (fixed, default) or 'cv' (even/odd-event cross
# validation over {0,2,5,10,20,40,80}: smallest odd-event two-end inconsistency among lambdas with pointwise error
# <= 1.05 x minimum; the medians of the even events are computed once for the whole grid).  No lower bound on s
# (Lei D3).  Several training samples give one s each; the default s is the sample with the fewest channels that
# borrow from the prior (choose_default_sample).
# Entry points: fit_scale_factors(ci_table, nodes, calibration_df, training_csv, particle, settings) ->
# scale table (s_final = s); collect_exact_overlap_hits + scale_table_from_hits (one stream, several lambdas).

# lookups of the s fit (s parameter, Lei): pedestal, hit threshold, mpv, TOT threshold, k, q range
def build_channel_luts(ci_table: pd.DataFrame, calibration_df: pd.DataFrame | None,
                       settings: dict | None = None) -> dict:
    s = _settings(settings)
    kmax = int(s["bar_key_size"])
    beam = _calibration_value_arrays(ci_table, calibration_df, s["hit_threshold_column"])
    src = {
        "pedestal": beam["pedestal"], "hit_threshold": beam["hit_threshold"], "mpv": beam["mpv"],
        "tot_cut": ci_table["ci_tot_threshold_fc"].to_numpy(float),
        "k": ci_table["ci_k_adc_per_fc"].to_numpy(float),
        "q_min": ci_table["mono_q_min_fc"].to_numpy(float),
        "q_max": ci_table["mono_q_max_fc"].to_numpy(float),
    }
    luts = {name: channel_lut(ci_table, values, kmax) for name, values in src.items()}
    luts["bar_key_size"] = kmax
    return luts


# DPM, LINK and fit eligibility of each channel uid (s parameter, Lei)
def scale_fit_metadata(ci_table: pd.DataFrame, calibration_df: pd.DataFrame | None,
                       settings: dict | None = None) -> dict:
    s = _settings(settings)
    kmax = int(s["bar_key_size"])
    n_uid = 2 * kmax
    uid = channel_uid(ci_table["layer"], ci_table["strip"], ci_table["end"], kmax)
    beam = _calibration_value_arrays(ci_table, calibration_df, s["hit_threshold_column"])
    ci_ok = ci_table["ci_status"].astype(str).eq(CI_OK).to_numpy()
    if calibration_df is not None:  # same protected gate as the CI map, whatever the ci_table was built with
        ci_ok &= (np.isfinite(beam["pedestal"]) & np.isfinite(beam["hit_threshold"])
                  & np.isfinite(beam["mpv"]) & (beam["mpv"] > 0))
    ci_ok &= np.isfinite(ci_table["ci_tot_threshold_fc"].to_numpy(float))
    meta = {"dpm": np.full(n_uid, -1, dtype=np.int16), "link": np.full(n_uid, -1, dtype=np.int16),
            "scale_ok": np.zeros(n_uid, dtype=bool), "uid": uid}
    meta["dpm"][uid] = ci_table["dpm"].to_numpy(np.int16)
    meta["link"][uid] = ci_table["link"].to_numpy(np.int16)
    meta["scale_ok"][uid] = ci_ok
    return meta


# exact-overlap hits (s parameter, Lei): ratio = (adc_sum - pedestal) / (k q(TOT)), adc_max < 1020
def collect_exact_overlap_hits(training_csv, luts: dict, mappers: dict, settings: dict | None = None,
                               chunksize: int | None = None) -> dict:
    """Hit: adc_sum >= hit threshold, q(TOT) > TOT threshold, CI map, not L12 S4-7, ratio in (0, 50)."""
    s = _settings(settings)
    kmax = luts["bar_key_size"]
    guard = float(s["adc_max_rail_guard"])
    ratio_max = float(s["ratio_max"])
    bad_keys = known_bad_keys(s)
    parts = {k: [] for k in ("uid", "ratio", "parity", "pair_key", "pair_ratio0", "pair_ratio1", "pair_parity")}
    censored_counts = np.zeros(2 * kmax, dtype=np.int64)
    n_rows = n_candidates = 0
    t0 = time.time()
    # same columns and parse dtypes as the reconstruction (value columns float32, exact for counts)
    for chunk in iter_csv(training_csv, READ_COLUMNS, int(chunksize or s["training_chunksize"]), dtypes=READ_DTYPES):
        n = len(chunk)
        n_rows += n
        layer = chunk["layer"].to_numpy(np.int64)
        strip = chunk["strip"].to_numpy(np.int64)
        parity = (chunk["pf_event"].to_numpy(np.int64) & 1).astype(np.int8)
        raw_key = layer * 100 + strip
        key = np.clip(raw_key, 0, kmax - 1)
        bad_bar = np.isin(raw_key, bad_keys)
        pair_ratio = [np.full(n, np.nan), np.full(n, np.nan)]
        for e in (0, 1):
            tot = chunk[f"tot_end{e}"].to_numpy(np.float64)
            adc = chunk[f"adc_sum_end{e}"].to_numpy(np.float64)
            adc_max = chunk[f"adc_max_end{e}"].to_numpy(np.float64)
            ped, mpv = luts["pedestal"][e][key], luts["mpv"][e][key]
            hit_thr, cut = luts["hit_threshold"][e][key], luts["tot_cut"][e][key]
            k, qmin, qmax = luts["k"][e][key], luts["q_min"][e][key], luts["q_max"][e][key]
            keep = (np.isfinite(adc) & np.isfinite(hit_thr) & (adc >= hit_thr)
                    & np.isfinite(ped) & np.isfinite(mpv) & (mpv > 0))
            cand = (keep & np.isfinite(tot) & np.isfinite(cut) & (tot > 0) & np.isfinite(k) & (k > 0)
                    & np.isfinite(qmin) & np.isfinite(qmax) & (qmax > qmin) & ~bad_bar)
            idx = np.flatnonzero(cand)
            if not idx.size:
                continue
            uid = e * kmax + key[idx]
            q = _q_from_uid(tot[idx], uid, mappers)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (adc[idx] - ped[idx]) / (k[idx] * q)  # ratio = (adc_sum - pedestal) / (k q)
                qnorm = (q - qmin[idx]) / (qmax[idx] - qmin[idx])
            good = (np.isfinite(q) & (q > 0) & np.isfinite(ratio) & (ratio > 0) & (ratio < ratio_max)
                    & np.isfinite(qnorm) & (q > cut[idx]))
            n_candidates += int(np.count_nonzero(good))
            exact = good & (adc_max[idx] < guard)
            censored = good & ~exact
            if censored.any():
                censored_counts += np.bincount(uid[censored], minlength=2 * kmax)
            if not exact.any():
                continue
            parts["uid"].append(uid[exact].astype(np.int32))
            parts["ratio"].append(ratio[exact].astype(np.float32))
            parts["parity"].append(parity[idx[exact]])
            pair_ratio[e][idx[exact]] = ratio[exact]
        both = np.isfinite(pair_ratio[0]) & np.isfinite(pair_ratio[1])
        if both.any():
            parts["pair_key"].append(key[both].astype(np.int32))
            parts["pair_ratio0"].append(pair_ratio[0][both].astype(np.float32))
            parts["pair_ratio1"].append(pair_ratio[1][both].astype(np.float32))
            parts["pair_parity"].append(parity[both])

    # join the collected parts of one field
    def cat(name, dtype):
        return np.concatenate(parts[name]).astype(dtype) if parts[name] else np.zeros(0, dtype=dtype)

    hits = {
        "uid": cat("uid", np.int32), "ratio": cat("ratio", np.float64), "parity": cat("parity", np.int8),
        "pair_uid0": cat("pair_key", np.int32), "pair_ratio0": cat("pair_ratio0", np.float64),
        "pair_ratio1": cat("pair_ratio1", np.float64), "pair_parity": cat("pair_parity", np.int8),
        "censored_counts": censored_counts, "bar_key_size": kmax,
    }
    hits["pair_uid1"] = (kmax + hits["pair_uid0"]).astype(np.int32)
    hits["stats"] = {"training_file": Path(training_csv).name, "rows_seen": int(n_rows),
                     "candidate_hits": int(n_candidates), "exact_hits": int(len(hits["uid"])),
                     "censored_hits": int(censored_counts.sum()), "exact_pairs": int(len(hits["pair_uid0"])),
                     "exact_pairs_odd": int(np.count_nonzero(hits["pair_parity"] == 1)),
                     "stream_s": round(time.time() - t0, 1)}
    log(f"exact overlap {hits['stats']['training_file']}: rows={n_rows} exact={hits['stats']['exact_hits']} "
        f"censored={hits['stats']['censored_hits']} pairs={hits['stats']['exact_pairs']} "
        f"[{time.time() - t0:.1f}s]")
    return hits


# pool the hits of several training csv files into one s fit (s parameter, Lei)
def merge_exact_overlap_hits(parts: list) -> dict:
    if len(parts) == 1:
        return parts[0]
    kmax = parts[0]["bar_key_size"]
    if any(h["bar_key_size"] != kmax for h in parts):
        raise ValueError("merge_exact_overlap_hits: the files were collected with different bar_key_size")
    hits = {name: np.concatenate([h[name] for h in parts]).astype(parts[0][name].dtype)
            for name in ("uid", "ratio", "parity", "pair_uid0", "pair_uid1", "pair_ratio0", "pair_ratio1",
                         "pair_parity")}
    hits["censored_counts"] = np.sum([h["censored_counts"] for h in parts], axis=0)
    hits["bar_key_size"] = kmax
    per_file = [h["stats"] for h in parts]
    hits["stats"] = {"training_file": " + ".join(st["training_file"] for st in per_file),
                     **{k: int(sum(st[k] for st in per_file)) for k in ("rows_seen", "candidate_hits", "exact_hits",
                                                                          "censored_hits", "exact_pairs",
                                                                          "exact_pairs_odd")},
                     "stream_s": round(sum(st["stream_s"] for st in per_file), 1), "per_file": per_file}
    log(f"exact overlap pooled ({len(parts)} files): exact={hits['stats']['exact_hits']} "
        + ", ".join(f"{st['training_file']} {st['exact_hits']}" for st in per_file))
    return hits


# s_direct calculation (s parameter, Lei): s_direct = median ratio of the channel, n = its hit count
def channel_medians(uid: np.ndarray, ratio: np.ndarray, mask: np.ndarray, n_uid: int):
    direct = np.full(n_uid, np.nan)
    counts = np.zeros(n_uid, dtype=np.int64)
    sel = mask & np.isfinite(ratio) & (ratio > 0)
    u, r = uid[sel], ratio[sel]
    if not u.size:
        return direct, counts
    order = np.argsort(u, kind="stable")
    u, r = u[order], r[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    bounds = np.r_[starts, u.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        direct[u[a]] = float(np.median(r[a:b]))  # s_direct = median ratio
        counts[u[a]] = b - a
    return direct, counts


# shrinkage (s parameter, Lei): s = exp(w ln s_direct + (1 - w) ln prior), w = n / (n + lambda)
def shrink_to_link_prior(direct: np.ndarray, counts: np.ndarray, dpm: np.ndarray, link: np.ndarray,
                         scale_ok: np.ndarray, shrinkage: float, min_link_hits: int = 0) -> dict:
    """prior = geometric median of s_direct over the channel's (DPM, LINK), else global."""
    finite_direct = np.isfinite(direct) & (direct > 0) & scale_ok
    logs = np.log(direct[finite_direct])
    global_prior = float(np.exp(np.median(logs))) if len(logs) else np.nan
    prior = np.full(len(direct), global_prior)
    level = np.full(len(direct), "global", dtype=object)
    labels = dpm.astype(np.int32) * 100 + link.astype(np.int32)
    for group in np.unique(labels[finite_direct]):
        members = finite_direct & (labels == group)
        if min_link_hits and counts[members].sum() < min_link_hits:
            continue
        values = np.log(direct[members])
        if len(values):
            prior[labels == group] = float(np.exp(np.median(values)))  # geometric median over (DPM, LINK)
            level[labels == group] = "link"
    estimate = prior.copy()
    both = finite_direct & np.isfinite(prior) & (prior > 0)
    weight = np.full(len(direct), np.nan)
    weight[both] = counts[both] / (counts[both] + shrinkage)  # w = n / (n + lambda)
    estimate[both] = np.exp(weight[both] * np.log(direct[both]) + (1.0 - weight[both]) * np.log(prior[both]))
    estimate[~scale_ok] = np.nan
    level[~np.isfinite(prior)] = ""
    return {"estimate": estimate, "prior": prior, "prior_level": level, "global_prior": global_prior,
            "weight": weight}


# s_direct and hit count per channel of the selected hits (s parameter, Lei)
def group_statistics(hits: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return channel_medians(hits["uid"], hits["ratio"], mask, 2 * hits["bar_key_size"])


# s of every channel for one lambda (s parameter, Lei)
def robust_group_estimates(hits: dict, mask: np.ndarray, meta: dict, shrinkage: float, settings: dict,
                           stats: tuple | None = None) -> dict:
    direct, counts = stats if stats is not None else group_statistics(hits, mask)
    fit = shrink_to_link_prior(direct, counts, meta["dpm"], meta["link"], meta["scale_ok"], float(shrinkage),
                               int(settings.get("min_link_hits") or 0))
    fit.update(direct=direct, counts=counts)
    return fit


# median |ln(prediction / target)|
def median_abs_log_error(prediction: np.ndarray, target: np.ndarray) -> float:
    good = np.isfinite(prediction) & (prediction > 0) & np.isfinite(target) & (target > 0)
    if not np.any(good):
        return np.nan
    return float(np.median(np.abs(np.log(prediction[good] / target[good]))))


# lambda by cross validation (s parameter, Lei): train on even events, validate on odd ones
def cross_validate_shrinkage(hits: dict, meta: dict, settings: dict, train_stats: tuple | None = None) -> dict:
    """Chosen: smallest odd-event two-end inconsistency among lambdas with error <= guard x minimum."""
    train = hits["parity"] == 0
    validation = hits["parity"] == 1
    stats = train_stats if train_stats is not None else group_statistics(hits, train)
    uid_val, ratio_val = hits["uid"][validation], hits["ratio"][validation]
    pv = hits["pair_parity"] == 1
    pu0, pu1 = hits["pair_uid0"][pv], hits["pair_uid1"][pv]
    pr0, pr1 = hits["pair_ratio0"][pv], hits["pair_ratio1"][pv]
    rows = []
    for lam in settings["lambda_grid"]:
        est = robust_group_estimates(hits, train, meta, float(lam), settings, stats=stats)["estimate"]
        loss = median_abs_log_error(est[uid_val], ratio_val)
        good = (np.isfinite(est[pu0]) & (est[pu0] > 0) & np.isfinite(est[pu1]) & (est[pu1] > 0)
                & np.isfinite(pr0) & (pr0 > 0) & np.isfinite(pr1) & (pr1 > 0))
        diff = np.abs(np.log(est[pu0[good]] / pr0[good]) - np.log(est[pu1[good]] / pr1[good]))
        rows.append({"shrinkage": float(lam), "validation_male": loss, "pair_validation_n": int(len(diff)),
                     "pair_validation_median_abs_logdiff": float(np.median(diff)) if len(diff) else np.nan})
    finite = [r["validation_male"] for r in rows if np.isfinite(r["validation_male"])]
    guard = float(settings["pointwise_loss_guard"])
    minimum = min(finite) if finite else np.nan
    eligible = [r for r in rows if np.isfinite(r["validation_male"]) and r["validation_male"] <= guard * minimum
                and np.isfinite(r["pair_validation_median_abs_logdiff"])]
    chosen = (min(eligible, key=lambda r: (r["pair_validation_median_abs_logdiff"], r["validation_male"]))
              if eligible else None)
    return {"cv": rows, "minimum_pointwise_validation_male": minimum,
            "selected_shrinkage": None if chosen is None else chosen["shrinkage"],
            "selected_pair_validation_median_abs_logdiff": np.nan if chosen is None
            else chosen["pair_validation_median_abs_logdiff"],
            "selection_rule": ("minimize odd-event double-end median abs log difference among models with "
                               f"pointwise MALE <= {guard:g}x minimum")}


# numpy values -> json values (NaN -> None)
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# s table of every channel from the collected hits (s parameter, Lei)
def scale_table_from_hits(hits: dict, meta: dict, ci_table: pd.DataFrame, settings: dict | None = None,
                          particle: str = DEFAULT_SCALE_PARTICLE, shrinkage=None) -> pd.DataFrame:
    s = _settings(settings)
    lam_setting = s["shrinkage_lambda"] if shrinkage is None else shrinkage
    use_cv = isinstance(lam_setting, str) and lam_setting.strip().lower() == "cv"
    diag: dict = {"training_particle": particle, **hits["stats"], "shrinkage_mode": "cv" if use_cv else "fixed"}
    exact_all = np.ones(len(hits["uid"]), dtype=bool)

    cv = train_stats = None
    if use_cv or s["cv_diagnostics"]:
        train_stats = group_statistics(hits, hits["parity"] == 0)  # shared by the grid and the diagnostics below
        cv = cross_validate_shrinkage(hits, meta, s, train_stats=train_stats)
        diag.update(cv)
        diag["cv_selected_shrinkage"] = cv["selected_shrinkage"]
    if use_cv:
        if cv["selected_shrinkage"] is None:
            selected = float(s["cv_fallback_lambda"])
            log(f"scale fit: cross validation undecided (too few exact hits / pairs); lambda = {selected:g}")
            diag["cv_fallback"] = True
        else:
            selected = float(cv["selected_shrinkage"])
    else:
        selected = float(lam_setting)
    diag["selected_shrinkage"] = selected

    fit = robust_group_estimates(hits, exact_all, meta, selected, s)
    final = fit["estimate"]
    diag["global_prior"] = fit["global_prior"]

    if cv is not None:  # validation diagnostics (train-only refit at the selected lambda)
        validation = hits["parity"] == 1
        train_est = robust_group_estimates(hits, hits["parity"] == 0, meta, selected, s, stats=train_stats)["estimate"]
        val_uid, val_ratio = hits["uid"][validation], hits["ratio"][validation]
        diag["validation_exact_n"] = int(len(val_ratio))
        diag["validation_new_median_abs_log_error"] = median_abs_log_error(train_est[val_uid], val_ratio)
        diag["validation_new_underprediction_fraction"] = (float(np.mean(train_est[val_uid] < val_ratio))
                                                           if len(val_ratio) else np.nan)

    uid = meta["uid"]
    counts = fit["counts"][uid]
    est = final[uid]
    ok = meta["scale_ok"][uid]
    prior = fit["prior"][uid]
    source = np.full(len(uid), SOURCE_NO_CI, dtype=object)
    no_model = ~ok | ~np.isfinite(est)
    source[no_model] = SOURCE_NO_MODEL
    source[~no_model & (counts > 0)] = SOURCE_DIRECT
    source[~no_model & (counts == 0) & np.isfinite(prior)] = SOURCE_BORROWED
    out = pd.DataFrame({
        "layer": ci_table["layer"].to_numpy(), "strip": ci_table["strip"].to_numpy(),
        "end": ci_table["end"].to_numpy(), "dpm": ci_table["dpm"].to_numpy(), "link": ci_table["link"].to_numpy(),
        "ci_status": ci_table["ci_status"].to_numpy(), "training_particle": particle,
        "shrinkage": selected,
        "s_direct": fit["direct"][uid], "s_exact_n": counts, "s_censored_n": hits["censored_counts"][uid],
        "shrinkage_weight": fit["weight"][uid], "s_link_prior": prior, "s_prior_level": fit["prior_level"][uid],
        "s_final": est, "s_source": source, "global_prior": fit["global_prior"],
    })
    out["cv_json"] = json.dumps(_json_safe(diag), sort_keys=True)
    out.attrs["diagnostics"] = diag
    log(f"scale fit ({particle}): lambda={selected:g} ({diag['shrinkage_mode']}), global prior "
        f"{fit['global_prior']:.5g}, {int(np.count_nonzero(source == SOURCE_DIRECT))} direct, "
        f"{int(np.count_nonzero(source == SOURCE_BORROWED))} borrowed, "
        f"{int(np.count_nonzero(source == SOURCE_NO_MODEL))} no model")
    return out


# hierarchical s fit (s parameter, Lei): training csv(s) -> s_final per channel
def fit_scale_factors(ci_table: pd.DataFrame, nodes: pd.DataFrame, calibration_df: pd.DataFrame | None,
                      training_csv, particle: str = DEFAULT_SCALE_PARTICLE, settings: dict | None = None,
                      chunksize: int | None = None) -> pd.DataFrame:
    s = _settings(settings)
    t0 = time.time()
    luts = build_channel_luts(ci_table, calibration_df, s)
    mappers = build_tot_charge_mappers(nodes, s)
    meta = scale_fit_metadata(ci_table, calibration_df, s)
    files = [training_csv] if isinstance(training_csv, (str, Path)) else list(training_csv)
    hits = merge_exact_overlap_hits([collect_exact_overlap_hits(f, luts, mappers, s, chunksize) for f in files])
    out = scale_table_from_hits(hits, meta, ci_table, s, particle)
    out.attrs["diagnostics"]["elapsed_s"] = round(time.time() - t0, 1)
    return out


# channels whose s is borrowed from the (DPM, LINK) / global prior (no exact hit of their own)
def borrowed_channels(scale_table: pd.DataFrame) -> int:
    return int(np.count_nonzero(scale_table["s_source"].astype(str).to_numpy() == SOURCE_BORROWED))


# default s of several samples (s parameter, Lei): the sample with the fewest borrowing channels (tie: first)
def choose_default_sample(scales: dict) -> tuple[str, dict]:
    counts = {sample: borrowed_channels(table) for sample, table in scales.items()}
    order = list(counts)
    best = min(order, key=lambda x: (counts[x], order.index(x)))
    return best, counts


# scale columns of the parameter table: the default sample's fit + s_<sample>, s_<sample>_source of each sample
def scale_columns(scales: dict, default: str) -> pd.DataFrame:
    out = scales[default].drop(columns=[c for c in SCALE_TABLE_DROP if c in scales[default].columns])
    if len(scales) > 1:
        for sample, table in scales.items():
            extra = table[KEY_COLUMNS + ["s_final", "s_source"]].rename(
                columns={"s_final": sample_path(sample), "s_source": f"{sample_path(sample)}_source"})
            out = out.merge(extra, on=KEY_COLUMNS, how="left")
    return out


# ==========================================================================
# BLOCK 3: per-end estimators (one crossover estimator, three scales)
# ==========================================================================
# lookups and q(TOT) maps the end estimators read (pedestal, mpv, TOT threshold, k, s of every s path)
def build_reconstruction_parameters(calibration: pd.DataFrame, table: pd.DataFrame | None,
                                    nodes: pd.DataFrame | None, s: dict) -> dict:
    kmax = int(s["bar_key_size"])

    # one table column as a lookup
    def lut(df, column):
        if df is None or column not in df.columns:
            return channel_lut(None, None, kmax)
        return channel_lut(df, pd.to_numeric(df[column], errors="coerce").to_numpy(float), kmax)

    luts = {"pedestal": lut(calibration, "pedestal"), "hit_threshold": lut(calibration, s["hit_threshold_column"]),
            "mpv": lut(calibration, "mpv"), "tot_cut": lut(table, "ci_tot_threshold_fc"),
            "k": lut(table, "ci_k_adc_per_fc"), "q_min": lut(table, "mono_q_min_fc"),
            "q_max": lut(table, "mono_q_max_fc")}
    paths = s.get("compute_paths") or list(PATHS)
    scales = {p: lut(table, "s_final" if p == "s" else p) for p in paths if is_scale_path(p)}
    return {"luts": luts, "scales": scales, "mappers": build_tot_charge_mappers(nodes, s),
            "bar_key_size": kmax, "bad_keys": known_bad_keys(s),
            "adc_max_near_rail": float(s["adc_max_rail_guard"]), "adc_max_rail": float(s["adc_max_rail"]),
            "bad_bar_rule": bad_bar_rule(s)}


# path-independent end values: hit = adc_sum >= hit threshold, raw = (adc_sum - pedestal) / mpv, q
def end_common_inputs(end: int, tot, adc_sum, adc_max, layer, strip, key, bad_bar, params: dict) -> dict:
    """candidate: hit, adc_max >= 1020, q(TOT) > TOT threshold, k > 0, CI map, not L12 S4-7."""
    lut = params["luts"]
    ped, mpv = lut["pedestal"][end][key], lut["mpv"][end][key]
    hit_thr, cut = lut["hit_threshold"][end][key], lut["tot_cut"][end][key]
    k, q_min, q_max = lut["k"][end][key], lut["q_min"][end][key], lut["q_max"][end][key]
    near = params["adc_max_near_rail"]
    with np.errstate(invalid="ignore"):
        norm_ok = np.isfinite(ped) & np.isfinite(mpv) & (mpv > 0)
        end_hit = np.isfinite(adc_sum) & np.isfinite(hit_thr) & (adc_sum >= hit_thr) & norm_ok
        adc_max_finite = np.isfinite(adc_max)
        near_rail_adc = adc_max_finite & (adc_max >= near)
        tot_finite = np.isfinite(tot)
        candidate = (end_hit & near_rail_adc & tot_finite & np.isfinite(cut) & (tot > 0)
                     & np.isfinite(k) & (k > 0) & np.isfinite(q_min) & np.isfinite(q_max) & (q_max > q_min)
                     & ~bad_bar)
    q = np.full(tot.shape, np.nan)
    idx = np.flatnonzero(candidate)
    if idx.size:
        q[idx] = q_from_tot(tot[idx], layer[idx], strip[idx], end, params["mappers"])
        with np.errstate(invalid="ignore"):
            candidate &= q > cut  # q(TOT) > TOT threshold, both fC (Lei D2)
        q[~candidate] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (adc_sum - ped) / mpv  # E_ADC = (adc_sum - pedestal) / mpv
        adc_clean = adc_max_finite & (adc_max < near)
    return {"end": end, "key": key, "tot": tot, "end_hit": end_hit, "candidate": candidate, "q": q, "raw": raw,
            "k": k, "mpv": mpv, "near_rail": end_hit & near_rail_adc, "adc_clean": adc_clean,
            "adc_at_rail": adc_max == params["adc_max_rail"], "tot_fired": tot_finite & (tot > 0)}


# end energy calculation: E = max((adc_sum - pedestal) / mpv, k s q(TOT) / mpv) for a TOT candidate
def crossover_end(common: dict, scale=None) -> dict:
    raw = common["raw"]
    if scale is None:
        available = np.zeros(raw.shape, dtype=bool)
        tot_energy = np.full(raw.shape, np.nan)
    else:
        q = common["q"]
        with np.errstate(invalid="ignore"):
            available = common["candidate"] & np.isfinite(q) & (q > 0) & np.isfinite(scale) & (scale > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tot_energy = common["k"] * scale * q / common["mpv"]  # E_TOT = k s q(TOT) / mpv
    with np.errstate(invalid="ignore"):
        used = available & np.isfinite(tot_energy) & (tot_energy > raw)
        energy = np.where(available, np.maximum(raw, tot_energy), raw)  # E = max(E_ADC, E_TOT)
    energy[~common["end_hit"]] = np.nan
    with np.errstate(invalid="ignore"):
        valid = common["end_hit"] & np.isfinite(energy) & (energy >= 0)
    return {"end_hit": common["end_hit"], "adc_at_rail": common["adc_at_rail"], "tot_fired": common["tot_fired"],
            "tot_conversion_available": available, "energy": energy, "valid": valid, "tot_conversion_used": used}


# adc_only scale: E = (adc_sum - pedestal) / mpv
def adc_only(common: dict, params: dict | None = None) -> dict:
    return crossover_end(common, None)


# PureCI scale: the crossover with s = 1
def pureci(common: dict, params: dict | None = None) -> dict:
    return crossover_end(common, 1.0)


# s scale: the crossover with the channel's TOT scale s of the path (s parameter, Lei)
def s_correction(common: dict, params: dict, path: str = "s") -> dict:
    return crossover_end(common, params["scales"][path][common["end"]][common["key"]])


# end energies of one path: adc_only, pureci or an s path
def end_energies(path: str, common: dict, params: dict) -> dict:
    if path == "adc_only":
        return adc_only(common)
    if path == "pureci":
        return pureci(common)
    return s_correction(common, params, path)


# ==========================================================================
# BLOCK 4: bar merge, bar quality flags, event accumulation, event selection
# ==========================================================================
# bar energy of the CI scales: one end at the rail -> that end if its TOT fired, else mean of the ends
def merge_rail_aware(e0: dict, e1: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bar, used0, used1); one saturated end next to an unsaturated one gives the saturated end."""
    hit0, hit1 = e0["end_hit"], e1["end_hit"]
    rail0, rail1 = e0["adc_at_rail"], e1["adc_at_rail"]
    fired0, fired1 = e0["tot_fired"], e1["tot_fired"]
    E0, E1 = e0["energy"], e1["energy"]
    V0, V1 = e0["valid"], e1["valid"]
    C0, C1 = e0["tot_conversion_used"], e1["tot_conversion_used"]
    n = len(E0)
    bar = np.full(n, np.nan)
    used0, used1 = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    hit = hit0 | hit1
    no_valid_end = np.zeros(n, dtype=bool)
    rail0_only = rail0 & ~rail1
    rail1_only = rail1 & ~rail0
    for rail_only, fired, v_rail, v_other, e_rail, e_other, u_rail, u_other in (
            (rail0_only, fired0, V0, V1, E0, E1, used0, used1), (rail1_only, fired1, V1, V0, E1, E0, used1, used0)):
        a = rail_only & fired                       # railed end with TOT: prefer it
        m1 = a & v_rail
        m2 = a & ~v_rail & v_other
        no_valid_end |= a & ~v_rail & ~v_other
        b = rail_only & ~fired                      # railed end without TOT: prefer the other end
        m3 = b & v_other
        m4 = b & ~v_other & v_rail
        no_valid_end |= b & ~v_other & ~v_rail
        bar[m1] = e_rail[m1]
        bar[m2] = e_other[m2]
        bar[m3] = e_other[m3]
        bar[m4] = e_rail[m4]
        u_rail[m1 | m4] = True
        u_other[m2 | m3] = True
    general = ~rail0_only & ~rail1_only & hit       # neither or both ends at the rail
    c0 = general & C0 & V0
    c1 = general & C1 & V1
    both = c0 & c1
    bar[both] = 0.5 * (E0[both] + E1[both])
    only0 = c0 & ~c1
    only1 = c1 & ~c0
    bar[only0] = E0[only0]
    bar[only1] = E1[only1]
    none = general & ~c0 & ~c1
    both_raw = none & V0 & V1
    raw0 = none & V0 & ~V1
    raw1 = none & V1 & ~V0
    bar[both_raw] = 0.5 * (E0[both_raw] + E1[both_raw])
    bar[raw0] = E0[raw0]
    bar[raw1] = E1[raw1]
    used0 |= both | only0 | both_raw | raw0
    used1 |= both | only1 | both_raw | raw1
    no_valid_end |= none & ~V0 & ~V1
    no_valid_end &= hit
    bar[no_valid_end] = 0.0
    return bar, used0, used1


# bar energy of adc_only: mean of the valid ends, the valid end, or 0
def merge_raw(e0: dict, e1: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V0, V1 = e0["valid"], e1["valid"]
    E0, E1 = e0["energy"], e1["energy"]
    bar = np.full(len(E0), np.nan)
    both = V0 & V1
    only0 = V0 & ~V1
    only1 = V1 & ~V0
    bar[both] = 0.5 * (E0[both] + E1[both])
    bar[only0] = E0[only0]
    bar[only1] = E1[only1]
    bar[(e0["end_hit"] | e1["end_hit"]) & ~V0 & ~V1] = 0.0
    return bar, V0.copy(), V1.copy()


# bar merge of a path: merge_raw for adc_only, merge_rail_aware for the CI scales
def bar_merge(path: str):
    return merge_raw if path == "adc_only" else merge_rail_aware


# bad bar (used_ends): no end the bar energy uses is trusted and one end is clipped
def bar_quality_flags(ends: list, commons: list, included: np.ndarray, used,
                      rule: str = "used_ends") -> tuple[np.ndarray, np.ndarray]:
    """trusted = TOT conversion used or unsaturated ADC; clipped = adc_max >= 1020, no usable TOT."""
    if rule not in BAD_BAR_RULES:
        raise ValueError(f"bad_bar_rule must be one of {BAD_BAR_RULES}, not {rule!r}")
    reliable = [e["tot_conversion_used"] | (e["valid"] & c["adc_clean"]) for c, e in zip(commons, ends)]
    clipped = [c["near_rail"] & ~e["tot_conversion_available"] for c, e in zip(commons, ends)]
    used0, used1 = used
    unrecoverable = included & (~used0 | ~reliable[0]) & (~used1 | ~reliable[1])
    if rule == "used_ends":
        unrecoverable &= clipped[0] | clipped[1]
    saturated_unconverted = (included & (commons[0]["near_rail"] | commons[1]["near_rail"])
                             & ~ends[0]["tot_conversion_used"] & ~ends[1]["tot_conversion_used"])
    return unrecoverable, saturated_unconverted


# bar energies of every path and the bad-bar flags of one chunk of beam rows
def reconstruct_chunk(chunk: pd.DataFrame, params: dict, compute_paths, selection_path: str,
                      end_values: bool = False) -> dict:
    pf = chunk["pf_event"].to_numpy(np.int64)
    layer = chunk["layer"].to_numpy(np.int64)
    strip = chunk["strip"].to_numpy(np.int64)
    raw_key = layer * 100 + strip
    key = np.clip(raw_key, 0, params["bar_key_size"] - 1)
    bad_bar = np.isin(raw_key, params["bad_keys"])
    commons = [end_common_inputs(e, chunk[f"tot_end{e}"].to_numpy(np.float64),
                                 chunk[f"adc_sum_end{e}"].to_numpy(np.float64),
                                 chunk[f"adc_max_end{e}"].to_numpy(np.float64),
                                 layer, strip, key, bad_bar, params) for e in (0, 1)]
    out = {"pf": pf, "layer": layer, "strip": strip, "tot0": commons[0]["tot"], "tot1": commons[1]["tot"]}
    selection_ends = selection_used = None
    for path in compute_paths:
        ends = [end_energies(path, c, params) for c in commons]
        bar, used0, used1 = bar_merge(path)(ends[0], ends[1])
        out[path] = bar
        if end_values:
            for e in (0, 1):
                out[f"{path}_end{e}"] = np.where(ends[e]["valid"], ends[e]["energy"], np.nan)
        if path == selection_path:
            selection_ends, selection_used = ends, (used0, used1)
    included = np.isfinite(out[selection_path])
    out["included"] = included
    out["unrecoverable"], out["saturated_unconverted"] = bar_quality_flags(
        selection_ends, commons, included, selection_used, params.get("bad_bar_rule", "used_ends"))
    return out


# per-event sums: event energy = sum of bar energies, bad_strip_fraction = n_bad / n_bars
class EventAccumulator:

    # empty accumulator
    def __init__(self, paths):
        self.paths = list(paths)
        self.n = 0                                   # distinct events seen
        self._ids = np.zeros(0, dtype=np.int64)      # distinct pf_event, ascending
        self._slot_of = np.zeros(0, dtype=np.int64)  # slot of each entry of _ids
        self._cap = 0
        self.n_bars = np.zeros(0, dtype=np.int64)
        self.n_bad = np.zeros(0, dtype=np.int64)
        self.n_saturated_unconverted = np.zeros(0, dtype=np.int64)
        self.energy = {p: np.zeros(0) for p in self.paths}

    # grow the arrays
    def _reserve(self, size: int) -> None:
        if size <= self._cap:
            return
        new = max(size, int(self._cap * 1.5) + 1024)

        # copy an array into a larger one
        def grown(a):
            b = np.zeros(new, dtype=a.dtype)
            b[:self.n] = a[:self.n]
            return b

        self.n_bars, self.n_bad = grown(self.n_bars), grown(self.n_bad)
        self.n_saturated_unconverted = grown(self.n_saturated_unconverted)
        self.energy = {p: grown(a) for p, a in self.energy.items()}
        self._cap = new

    # slots of the events of a chunk (new events get new slots)
    def _slots(self, pf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uniq, inverse = np.unique(pf, return_inverse=True)
        pos = np.searchsorted(self._ids, uniq)
        known = np.zeros(len(uniq), dtype=bool)
        inside = pos < len(self._ids)
        known[inside] = self._ids[pos[inside]] == uniq[inside]
        slot = np.empty(len(uniq), dtype=np.int64)
        slot[known] = self._slot_of[pos[known]]
        new = ~known
        k = int(np.count_nonzero(new))
        if k:
            fresh = np.arange(self.n, self.n + k, dtype=np.int64)
            slot[new] = fresh
            self._reserve(self.n + k)
            self.n += k
            self._ids = np.insert(self._ids, pos[new], uniq[new])
            self._slot_of = np.insert(self._slot_of, pos[new], fresh)
        return slot, inverse.ravel()

    # add the bars of one chunk
    def add(self, pf: np.ndarray, included: np.ndarray, bars: dict, bad: np.ndarray, sat: np.ndarray) -> None:
        if not len(pf):
            return
        slot, inv = self._slots(np.asarray(pf, dtype=np.int64))
        m = len(slot)
        self.n_bars[slot] += np.bincount(inv[included], minlength=m)
        self.n_bad[slot] += np.bincount(inv[bad], minlength=m)
        self.n_saturated_unconverted[slot] += np.bincount(inv[sat], minlength=m)
        for p in self.paths:
            mask = included & np.isfinite(bars[p])
            self.energy[p][slot] += np.bincount(inv[mask], weights=bars[p][mask], minlength=m)

    # events with >= 1 reconstructed bar, sorted by pf_event
    def finalize(self) -> dict:
        keep = self.n_bars[self._slot_of] > 0
        slots = self._slot_of[keep]
        n_bars = self.n_bars[slots]
        n_bad = self.n_bad[slots]
        denominator = np.maximum(n_bars, 1).astype(float)
        return {"pf_event": self._ids[keep].astype(np.int64), "n_bars": n_bars, "n_unrecoverable_bars": n_bad,
                "bad_strip_fraction": n_bad / denominator,  # n_bad / n_bars
                "saturated_unconverted_fraction": self.n_saturated_unconverted[slots] / denominator,
                "energy": {p: self.energy[p][slots] for p in self.paths},
                "n_events_in_file": int(self.n)}


# layout of the per-bar cache record: event, bar, TOT, then bar / end0 / end1 energy of every path (float32)
def bar_record_dtype(paths) -> np.dtype:
    fields = [("pf_event", "i8"), ("layer", "i2"), ("strip", "i2"), ("tot_end0", "f4"), ("tot_end1", "f4")]
    for p in paths:
        fields += [(f"bar_energy_{p}_mip", "f4"), (f"end0_energy_{p}_mip", "f4"), (f"end1_energy_{p}_mip", "f4")]
    return np.dtype(fields)


# per-bar records of the included bars of one chunk
def _bar_records(r: dict, paths, dtype: np.dtype) -> np.ndarray:
    m = r["included"]
    rec = np.empty(int(np.count_nonzero(m)), dtype=dtype)
    rec["pf_event"], rec["layer"], rec["strip"] = r["pf"][m], r["layer"][m], r["strip"][m]
    rec["tot_end0"], rec["tot_end1"] = r["tot0"][m], r["tot1"][m]
    for p in paths:
        rec[f"bar_energy_{p}_mip"] = r[p][m]
        rec[f"end0_energy_{p}_mip"] = r[f"{p}_end0"][m]
        rec[f"end1_energy_{p}_mip"] = r[f"{p}_end1"][m]
    return rec


# a .npy file of n records from their raw bytes (header, then the bytes)
def write_npy_records(dst, raw, dtype: np.dtype, n: int) -> None:
    with open(dst, "wb") as out:
        np.lib.format.write_array_header_1_0(out, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                                   "fortran_order": False, "shape": (int(n),)})
        with open(raw, "rb") as src:
            shutil.copyfileobj(src, out, 16 << 20)


# one streamed pass over a beam csv: event sums (+ the per-bar cache records)
def reconstruct_dataset(job: dict) -> dict:
    t0 = time.time()
    params, compute = job["params"], job["compute_paths"]
    acc = EventAccumulator(compute)
    raw = job.get("bar_raw")
    dtype = bar_record_dtype(compute)
    n_rows = n_records = 0
    fh = open(raw, "wb") if raw else None
    try:
        for chunk in iter_csv(job["csv"], READ_COLUMNS, int(job["chunk_rows"]), dtypes=READ_DTYPES):
            n_rows += len(chunk)
            r = reconstruct_chunk(chunk, params, compute, job["selection_path"], end_values=fh is not None)
            acc.add(r["pf"], r["included"], r, r["unrecoverable"], r["saturated_unconverted"])
            if fh is not None:
                rec = _bar_records(r, compute, dtype)
                rec.tofile(fh)
                n_records += len(rec)
    finally:
        if fh is not None:
            fh.close()
    if raw:
        write_npy_records(job["bar_npy"], raw, dtype, n_records)
        Path(raw).unlink()
    events = acc.finalize()
    return {"key": job["key"], "events": events, "n_rows": n_rows, "bar_rows": n_records,
            "seconds": time.time() - t0}


# worker process: reconstruct one csv
def _reconstruct_worker(job: dict) -> dict:
    limit_blas_threads()
    return reconstruct_dataset(job)


# log line of one reconstructed csv
def _log_reconstruction(res: dict) -> None:
    ev = res["events"]
    log(f"  {res['key']}: {res['n_rows']:,} rows, {ev['n_events_in_file']:,} events, "
        f"{len(ev['pf_event']):,} reconstructed, {int(ev['n_bars'].sum()):,} bars [{res['seconds']:.1f}s]")


# reconstruct the csv files (in parallel with workers > 1), results in job order
def iter_reconstruction(jobs: list[dict], workers: int):
    if workers > 1 and len(jobs) > 1:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)))
        try:
            for res in pool.map(_reconstruct_worker, jobs):
                _log_reconstruction(res)
                yield res
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        for job in jobs:
            res = reconstruct_dataset(job)
            _log_reconstruction(res)
            yield res


# reconstruct all csv files at once (library use)
def run_reconstruction(jobs: list[dict], workers: int) -> list[dict]:
    with closing(iter_reconstruction(jobs, workers)) as results:
        return list(results)


# event selection: selected = bad_strip_fraction <= quality_threshold
def select_events(events: dict, s: dict) -> np.ndarray:
    if not s["quality_filter"]:
        return np.ones(len(events["pf_event"]), dtype=bool)
    return events["bad_strip_fraction"] <= float(s["quality_threshold"]) + float(s["quality_tolerance"])


# number of values in [low, high)
def _count_in(values: np.ndarray, bounds) -> int:
    return int(np.count_nonzero((values >= bounds[0]) & (values < bounds[1])))


# quality scan (check): peak, resolution and retention of the electron runs for every threshold of the grid
def scan_quality_threshold(electron_tables: list, s: dict) -> pd.DataFrame:
    ers = s["energy_response"]

    # automatic peak-scan fit of a spectrum
    def metric(values):
        return er.peak_scan_fit(values, n_bins=ers["peak_scan_bins"],
                                upper_percentile=ers["peak_scan_upper_percentile"],
                                upper_factor=ers["peak_scan_upper_factor"], upper_min=ers["peak_scan_upper_min"])

    shoulder, core = s["scan_shoulder_range_mip"], s["scan_core_range_mip"]
    rows = []
    for ds, energy, fraction in electron_tables:
        base = metric(energy)
        base_shoulder, base_core = _count_in(energy, shoulder), _count_in(energy, core)
        for thr in s["scan_threshold_grid"]:
            sel = fraction <= float(thr) + float(s["quality_tolerance"])
            values = energy[sel]
            m = metric(values)
            with np.errstate(all="ignore"):
                shift = (m["mu"] - base["mu"]) / base["mu"] if base["mu"] else np.nan
                change = ((m["sigma_over_mu"] - base["sigma_over_mu"]) / base["sigma_over_mu"]
                          if base["sigma_over_mu"] else np.nan)
            n_sh, n_co = _count_in(values, shoulder), _count_in(values, core)
            rows.append({
                "threshold": float(thr), "dataset": ds.key, "beam_energy_gev": ds.energy_gev,
                "n_events_before": int(len(energy)), "n_events_after": int(len(values)),
                "retention_fraction": len(values) / max(len(energy), 1), "peak_scan_mu_mip": m["mu"],
                "peak_scan_sigma_mip": m["sigma"], "sigma_over_mu": m["sigma_over_mu"],
                "n_detected_peaks": m["n_detected_peaks"],
                "detected_peak_positions_mip": ";".join(f"{x:.7g}" for x in m["peak_positions"]),
                "peak_shift_vs_unfiltered_fraction": shift, "resolution_change_vs_unfiltered_fraction": change,
                "mean_bad_strip_fraction_all": float(np.mean(fraction)) if len(fraction) else np.nan,
                "shoulder_count": n_sh, "core_count": n_co,
                "shoulder_retention_fraction": n_sh / max(base_shoulder, 1),
                "core_retention_fraction": n_co / max(base_core, 1),
                "configured_threshold": float(s["quality_threshold"]),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# BLOCK 5: spectrum fits, linearity / resolution / e/h, response table and figures
# ==========================================================================
# Gaussian
def _gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


# chi2 with variance max(n, 1), and its ndf
def _neyman_chi2(observed, predicted, n_par):
    return float(np.sum((observed - predicted) ** 2 / np.maximum(observed, 1.0))), int(len(observed) - n_par)


# fit histogram on the [q0.001, q0.995] quantile range
def _fit_histogram(values, bins, s):
    low, high = np.quantile(values, s["fit_quantiles"])
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(float), float(edges[1] - edges[0]), float(low), float(high)


# Default spectrum fits (Lei 2026-10-02, R2): every Gaussian fit is a SINGLE Gaussian.  Muon and pion use the muon
# core logic, each with its own histogram bins; a double Gaussian is only fitted on request, --fit[a,b],[c,d]
# (energy_response.range_spectrum_fit, R4).
CORE_FIT_KINDS = {"muon": "muon_core_gauss", "pion": "pion_core_gauss"}
CORE_FIT_BINS = {"muon": "muon_fit_bins", "pion": "pion_fit_bins"}   # setting of the core-fit histogram bins


# muon / pion core Gaussian: pass 1 on mu0 +- 2.5 sigma0, pass 2 on mu1 +- 2 sigma1
def core_gauss_fit(values, s: dict, bins: int, kind: str) -> dict:
    """Never raises: a failed fit has NaN values, ok False and status 'fit_failed: ...'."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    min_count = int(s["energy_response"]["min_bin_count"])
    out = {"kind": kind, "ok": False, "status": f"fit_failed: no data (fewer than 2 finite values: {v.size})",
           "n_values": int(v.size), "amplitude": _NAN,
           "mu": _NAN, "sigma": _NAN, "mu_error": _NAN, "sigma_error": _NAN, "sigma_over_mu": _NAN,
           "fit_low": _NAN, "fit_high": _NAN, "chi2": _NAN, "ndf": 0, "bin_width": _NAN, "initial_peak_mip": _NAN}
    if v.size < 2:
        return out
    try:
        with _quiet():
            centers, counts, width, low, high = _fit_histogram(v, int(bins), s)
            out["bin_width"] = width
            smoothed = gaussian_filter1d(counts, sigma=2.0)
            peak = int(np.argmax(smoothed))
            sigma0 = max(float(peak_widths(smoothed, [peak], rel_height=0.5)[0][0] * width / FWHM_PER_SIGMA_MUON_PION),
                         width)
            initial = [float(smoothed[peak]), float(centers[peak]), sigma0]
            bounds = ([0.0, low, width / 2.0], [np.inf, high, (high - low) / 2.0])
            first_mask = ((centers >= initial[1] - 2.5 * initial[2]) & (centers <= initial[1] + 2.5 * initial[2])
                          & (counts >= min_count))
            first, _ = curve_fit(_gaussian, centers[first_mask], counts[first_mask], p0=initial,
                                 sigma=np.sqrt(counts[first_mask]), absolute_sigma=True, bounds=bounds, maxfev=100_000)
            fit_low, fit_high = float(first[1] - 2.0 * first[2]), float(first[1] + 2.0 * first[2])
            mask = (centers >= fit_low) & (centers <= fit_high) & (counts >= min_count)
            final, cov = curve_fit(_gaussian, centers[mask], counts[mask], p0=first, sigma=np.sqrt(counts[mask]),
                                   absolute_sigma=True, bounds=bounds, maxfev=100_000)
            errors = np.sqrt(np.diag(cov))
            chi2, ndf = _neyman_chi2(counts[mask], _gaussian(centers[mask], *final), 3)
    except Exception as exc:  # low statistics: NaN + status, never an exception
        out["status"] = f"fit_failed: {type(exc).__name__}: {exc}"[:200]
        return out
    if not np.all(np.isfinite(final)):  # R3: no partial value
        out["status"] = "fit_failed: non-finite parameters"
        return out
    out.update(amplitude=float(final[0]), mu=float(final[1]), sigma=float(final[2]), mu_error=float(errors[1]),
               sigma_error=float(errors[2]), sigma_over_mu=float(final[2] / final[1]), fit_low=fit_low,
               fit_high=fit_high, chi2=chi2, ndf=ndf, initial_peak_mip=float(centers[peak]), status="ok", ok=True)
    return out


# default muon fit: core Gaussian on 180 bins
def fit_muon_core(values, s: dict) -> dict:
    return core_gauss_fit(values, s, int(s["muon_fit_bins"]), CORE_FIT_KINDS["muon"])


# default pion fit: core Gaussian on 160 bins (single Gaussian)
def fit_pion_core(values, s: dict) -> dict:
    return core_gauss_fit(values, s, int(s["pion_fit_bins"]), CORE_FIT_KINDS["pion"])


# bin width of the core-fit histogram of a muon / pion run
def core_fit_bin_width(values, particle: str, s: dict) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return _NAN
    with _quiet():
        return _fit_histogram(v, int(s[CORE_FIT_BINS[particle]]), s)[2]


# bin edges of a muon / pion spectrum page: [0, q0.999], at least 60 bins
def display_edges(values, bin_width: float, s: dict) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.zeros(0)
    high = float(np.quantile(v, s["display_upper_quantile"]))
    high = high if high > 0 else 1.0
    n_min = int(s["display_min_bins"])
    bins = max(n_min, int(np.ceil(high / bin_width))) if (np.isfinite(bin_width) and bin_width > 0) else n_min
    return np.linspace(0.0, high, bins + 1)


# muon / pion page histogram: (centers, counts, bin width, upper edge)
def display_histogram(values, bin_width: float, s: dict):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.zeros(0), np.zeros(0), _NAN, 1.0
    counts, edges = np.histogram(v, bins=display_edges(v, bin_width, s))
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(float), float(edges[1] - edges[0]), float(edges[-1])


# bin edges of the default spectrum page (the bins -fit fits without -bin)
def spectrum_page_edges(values, particle: str, s: dict) -> np.ndarray:
    if particle in CORE_FIT_BINS:
        return display_edges(values, core_fit_bin_width(values, particle, s), s)
    ers = s["energy_response"]
    return er.default_spectrum_edges(values, bins=int(ers["spectrum_bins"]),
                                     upper_percentile=float(ers["hist_upper_percentile"]),
                                     upper_factor=float(ers["hist_upper_factor"]),
                                     upper_min=float(ers["hist_upper_min"]))


# peak scan + spectrum fit of one run and path (default single Gaussian, or the -fit range fit)
def spectrum_fit(values: np.ndarray, particle: str, s: dict) -> dict:
    ers = s["energy_response"]
    scan = er.peak_scan_fit(values, n_bins=ers["peak_scan_bins"], upper_percentile=ers["peak_scan_upper_percentile"],
                            upper_factor=ers["peak_scan_upper_factor"], upper_min=ers["peak_scan_upper_min"])
    if ers.get("fit_ranges"):
        fit = er.range_spectrum_from_settings(values, ers, spectrum_page_edges(values, particle, s))
    elif particle == "muon":
        fit = fit_muon_core(values, s)
    elif particle == "pion":
        fit = fit_pion_core(values, s)
    else:
        fit = er.two_pass_gauss_fit(values, bins=int(ers["spectrum_bins"]), min_bin_count=int(ers["min_bin_count"]),
                                    window_nsigma=float(ers["window_nsigma"]), initial_mu=scan["seed_mu"],
                                    initial_sigma=scan["seed_sigma"], min_fit_bins=int(ers["min_fit_bins"]),
                                    upper_percentile=float(ers["hist_upper_percentile"]),
                                    upper_factor=float(ers["hist_upper_factor"]),
                                    upper_min=float(ers["hist_upper_min"]))
    return {"scan": scan, "fit": fit}


LINEARITY_SOURCE_FIT = "range_fit"   # linearity_source of the points when --fit gave them (Lei 2026-10-02)


# spectrum fits of the linearity points: range_fit with -fit, else linearity_source
def linearity_source_used(s: dict) -> str:
    return LINEARITY_SOURCE_FIT if s["energy_response"].get("fit_ranges") else s["linearity_source"]


# log note when -fit replaces linearity_source peak_scan
def linearity_source_note(s: dict) -> str | None:
    if linearity_source_used(s) == LINEARITY_SOURCE_FIT and s["linearity_source"] == "peak_scan":
        return ("ci: NOTE --fit replaces linearity_source peak_scan (yml) for the linearity / resolution points: they "
                "come from the --fit fits (a failed or two-range fit gives no point); the peak_scan_* columns keep "
                "the automatic peak-scan values")
    return None


# footer text naming the spectrum fits of the linearity / resolution points
def points_source_text(s: dict) -> str:
    if linearity_source_used(s) != LINEARITY_SOURCE_FIT:
        return f"{s['linearity_source']} peaks"
    text = f"--fit {format_ranges(s['energy_response']['fit_ranges'])} MIP range-fit peaks"
    return text + (" (replaces linearity_source peak_scan)" if s["linearity_source"] == "peak_scan" else "")


# why a spectrum fit gives no linearity point ('' when it gives one)
def _no_point_reason(src: dict) -> str:
    if src.get("kind") == "range_double_gauss" and src.get("ok"):
        return "double Gaussian (two --fit ranges): no single mu"
    status = str(src.get("status") or "")
    if src.get("ok") or not status:
        return ""
    what = "peak-scan fit" if src.get("kind") == "peak_scan" else "spectrum fit"
    return textwrap.shorten(f"{what} failed: {status.removeprefix('fit_failed: ')}", 110, placeholder=" ...")


# linearity (mu vs E), resolution (sigma / mu vs E) and, on an s path, e/h of one path over every run
def energy_dependence(summaries: list, path: str, s: dict) -> dict:
    """The point arrays follow the order of summaries; dataset_keys names that order."""
    keys = [summ["ds"].key for summ in summaries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"ci: duplicate dataset keys in the energy dependence: {keys}")
    ers = s["energy_response"]
    use_scan = linearity_source_used(s) == "peak_scan"
    E, mu, sigma, sigma_err, n, why, particles = [], [], [], [], [], [], []
    for summ in summaries:
        f = summ["fits"][path]
        src = f["scan"] if use_scan else f["fit"]
        E.append(summ["ds"].energy_gev)
        particles.append(summ["ds"].particle)
        mu.append(src.get("mu", _NAN))
        sigma.append(src.get("sigma", _NAN))
        sigma_err.append(src.get("sigma_error", _NAN))
        n.append(summ["n_selected"])
        why.append(_no_point_reason(src))
    pe = er.point_errors(mu, sigma, n=n, sigma_error=sigma_err, prescription=ers["error_prescription"])
    lin = er.linearity_from_settings(E, mu, pe["mu_error"], particles, ers, scale=path_scale(path))
    res = er.resolution_from_settings(E, pe["ratio"], pe["ratio_error"], ers, particles=particles)
    eh = None
    if is_scale_path(path):  # e/h only on an s scale (Lei 2026-10-20)
        eh = er.e_over_h(E, mu, pe["mu_error"], particles, electron_line=lin["fits"].get(er.ELECTRON), keys=keys)
    return {"linearity": lin, "resolution": res, "eh": eh, "errors": pe, "n": np.asarray(n, dtype=float),
            "dataset_keys": keys, "particles": particles, "no_point_reasons": why}


# '[a, b]' of a range ('' without one)
def _range_text(fit_range) -> str:
    return format_ranges([fit_range]) if fit_range is not None else ""


# '30,40' of a -bin setting ('' without one)
def _bins_text(bins) -> str:
    return ",".join(str(int(n)) for n in bins) if bins else ""


# bin_width_mip of a spectrum row
def _fit_bin_width(fit: dict) -> float:
    if fit.get("kind") not in er.RANGE_FIT_KINDS:
        return fit.get("bin_width", _NAN)
    if not fit.get("fit_bins"):
        return fit.get("page_width", _NAN)
    widths = [(float(b) - float(a)) / int(n) for (a, b), n in zip(fit["ranges"], fit["fit_bins"])]
    return widths[0] if all(math.isclose(w, widths[0], rel_tol=1e-9) for w in widths) else _NAN


# areas of the two components of a double-Gaussian -fit, in range order
def component_areas(fit: dict) -> dict:
    comps = sorted(fit.get("components") or [], key=lambda c: float(c["range"][0]))
    out = {}
    for tag, k in (("low", 0), ("high", 1)):
        comp = comps[k] if len(comps) == 2 else {}
        out[f"area_{tag}_events"] = float(comp.get("area", _NAN))
        out[f"area_{tag}_error_events"] = float(comp.get("area_error", _NAN))
    return out


# max |residual / prediction| of a line over its fitted points
def _max_rel_residual(line: dict) -> float:
    r = np.asarray(line["residual_fraction"], dtype=float)[np.asarray(line["fitted"], dtype=bool)]
    r = r[np.isfinite(r)]
    return float(np.max(np.abs(r))) if r.size else _NAN


# spectrum row of one run and path (+ its linearity point columns)
def _spectrum_row(summ: dict, p: str, dependence: dict, point_index: dict, s: dict) -> dict:
    ds = summ["ds"]
    f, scan = summ["fits"][p]["fit"], summ["fits"][p]["scan"]
    thr = float(s["quality_threshold"]) if s["quality_filter"] else _NAN
    row = {"row_type": "spectrum", "dataset": ds.key, "particle": ds.particle, "beam_energy_gev": ds.energy_gev,
           "path": p, "selection_path": s["selection_path_used"], "quality_threshold": thr,
           "n_csv_files": summ["n_csv"], "n_events_in_file": summ["n_events_in_file"],
           "n_events_reconstructed": summ["n_reconstructed"], "n_events_selected": summ["n_selected"],
           "retention_fraction": summ["n_selected"] / max(summ["n_reconstructed"], 1),
           "n_bars_selected_events": summ["n_bars_selected"],
           "fit_kind": f.get("kind"), "fit_status": f.get("status"), "fit_ok": bool(f.get("ok")),
           "fit_ranges": f.get("fit_ranges_text", ""), "fit_bins": _bins_text(f.get("fit_bins")),
           "mu_mip": f.get("mu", _NAN), "mu_error_mip": f.get("mu_error", _NAN),
           "sigma_mip": f.get("sigma", _NAN), "sigma_error_mip": f.get("sigma_error", _NAN),
           "sigma_over_mu": f.get("sigma_over_mu", _NAN), "chi2": f.get("chi2", _NAN), "ndf": f.get("ndf", 0),
           "fit_low_mip": f.get("fit_low", _NAN), "fit_high_mip": f.get("fit_high", _NAN),
           "bin_width_mip": _fit_bin_width(f), "hist_upper_mip": f.get("hist_upper", _NAN),
           "pass1_mu_mip": f.get("pass1_mu", _NAN), "pass1_sigma_mip": f.get("pass1_sigma", _NAN),
           "pass1_chi2": f.get("pass1_chi2", _NAN), "pass1_ndf": f.get("pass1_ndf", _NAN),
           "core_bin_count": f.get("core_bin_count", _NAN),
           "mu_low_mip": f.get("mu_low", _NAN), "mu_low_error_mip": f.get("mu_low_error", _NAN),
           "sigma_low_mip": f.get("sigma_low", _NAN), "sigma_low_error_mip": f.get("sigma_low_error", _NAN),
           "mu_high_mip": f.get("mu_high", _NAN), "mu_high_error_mip": f.get("mu_high_error", _NAN),
           "sigma_high_mip": f.get("sigma_high", _NAN), "sigma_high_error_mip": f.get("sigma_high_error", _NAN)}
    if f.get("kind") == "range_double_gauss":  # R4: the component weights (default rows: no such columns)
        row.update(component_areas(f))
    row.update(peak_scan_mu_mip=scan["mu"], peak_scan_sigma_mip=scan["sigma"],
               peak_scan_sigma_over_mu=scan["sigma_over_mu"], peak_scan_n_peaks=scan["n_detected_peaks"],
               peak_scan_status=scan["status"])
    if p in dependence and ds.key in point_index[p]:
        i = point_index[p][ds.key]
        dep = dependence[p]
        lin = dep["linearity"]
        line = lin["fits"].get(ds.particle)
        row.update(mu_stat_error_mip=float(dep["errors"]["mu_error"][i]),
                   ratio_error=float(dep["errors"]["ratio_error"][i]),
                   linearity_fitted=bool(lin["fitted"][i]),
                   linearity_predicted_mip=float(line["predicted"][i]) if line else _NAN,
                   linearity_residual_fraction=float(line["residual_fraction"][i]) if line else _NAN)
    return row


# linearity rows of one path: one per fitted particle, or one 'points only' row
def _linearity_rows(p: str, lin: dict, s: dict) -> list[dict]:
    common = {"row_type": "linearity", "path": p, "linearity_source": linearity_source_used(s),
              "error_prescription": s["energy_response"]["error_prescription"]}
    if not lin["fits"]:
        return [{**common, "fit_kind": "points_only", "fit_status": lin["status"], "fit_ok": False}]
    normal = bool(s["energy_response"].get("normal_error"))
    rows = []
    for r in er.linearity_rows(lin):
        line = lin["fits"][r["particle"]]
        row = {**common, "particle": r["particle"],
               "fit_kind": f"linear_{'weighted' if r['weighted'] else 'unweighted'}", "fit_status": r["status"],
               "fit_ok": r["ok"], "fitted_energies": r["fitted_energies"], "linearity_fit_range": r["fit_range_gev"],
               "slope_mip_per_gev": r["slope"], "slope_error_mip_per_gev": r["slope_error"],
               "intercept_mip": r["intercept"], "intercept_error_mip": r["intercept_error"], "chi2": r["chi2"],
               "ndf": r["ndf"], "chi2_over_ndf": r["chi2_over_ndf"], "max_abs_rel_residual": _max_rel_residual(line)}
        if normal:  # Lei D6: statistical AND scaled errors
            row.update(error_scale=r["error_scale"], slope_error_scaled_mip_per_gev=r["slope_error_scaled"],
                       intercept_error_scaled_mip=r["intercept_error_scaled"])
        rows.append(row)
    return rows


# resolution row of one path
def _resolution_row(p: str, res: dict, s: dict) -> dict:
    row = {"row_type": "resolution", "path": p, "particle": res["fit_particle"], "fit_kind": res["model"],
           "fit_status": res["status"], "fit_ok": bool(res["ok"]), "fitted_energies": res["fitted_energies"],
           "resolution_fit_range": _range_text(res.get("fit_range")),
           "resolution_a_mev": res["a"], "resolution_a_error_mev": res["a_error"],
           "resolution_b_sqrt_mev": res["b"], "resolution_b_error_sqrt_mev": res["b_error"],
           "resolution_c": res["c"], "resolution_c_error": res["c_error"], "chi2": res["chi2"],
           "ndf": res["ndf"], "chi2_over_ndf": res["chi2_over_ndf"], "linearity_source": linearity_source_used(s),
           "error_prescription": s["energy_response"]["error_prescription"]}
    if s["energy_response"].get("normal_error"):  # Lei D6: statistical AND scaled errors
        row.update(error_scale=res["error_scale"], resolution_a_error_scaled_mev=res["a_error_scaled"],
                   resolution_b_error_scaled_sqrt_mev=res["b_error_scaled"],
                   resolution_c_error_scaled=res["c_error_scaled"])
    return row


# e/h rows of one s path (one per pion run)
def _eh_rows(p: str, eh: dict) -> list[dict]:
    return [{"row_type": "e_over_h", "path": p, "dataset": r["run"], "particle": r["hadron"],
             "beam_energy_gev": r["energy_gev"], "e_over_h": r["e_over_h"], "e_over_h_error": r["e_over_h_error"],
             "mu_electron_mip": r["mu_electron"], "mu_electron_error_mip": r["mu_electron_error"],
             "mu_hadron_mip": r["mu_hadron"], "mu_hadron_error_mip": r["mu_hadron_error"],
             "electron_source": r["electron_source"], "eh_filled": bool(r["filled"]), "eh_note": r["note"]}
            for r in er.eh_rows(eh)]


# rows of energy_response.csv: spectrum (run x path), linearity, resolution and e/h (per path)
def energy_response_rows(summaries: list, dependence: dict, output_paths, s: dict) -> list[dict]:
    keys = [summ["ds"].key for summ in summaries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"ci: duplicate dataset keys in the spectrum summaries: {keys}")
    point_index = {p: {k: i for i, k in enumerate(dep["dataset_keys"])} for p, dep in dependence.items()}
    rows = [_spectrum_row(summ, p, dependence, point_index, s) for summ in summaries for p in output_paths]
    for p in output_paths:
        if p not in dependence:
            continue
        dep = dependence[p]
        rows += _linearity_rows(p, dep["linearity"], s)
        rows.append(_resolution_row(p, dep["resolution"], s))
        if dep["eh"] is not None:
            rows += _eh_rows(p, dep["eh"])
    return rows


# spectrum page of a muon / pion run: points and the core Gaussian (library for the plots stage)
def _plot_particle_panel(ax, values, fit: dict, path: str, s: dict) -> None:
    min_count = int(s["energy_response"]["min_bin_count"])
    centers, counts, _, upper = display_histogram(values, fit.get("bin_width", _NAN), s)
    visible = counts >= min_count
    if visible.any():
        ax.scatter(centers[visible], counts[visible], s=15, color=path_color(path), edgecolors="none", alpha=0.95,
                   label=f"event energies (count >= {min_count})", zorder=3)
    curve_max = 0.0
    if fit.get("ok") and np.isfinite(fit.get("fit_low", np.nan)):
        cx = np.linspace(fit["fit_low"], fit["fit_high"], 500)
        cy = _gaussian(cx, fit["amplitude"], fit["mu"], fit["sigma"])
        text = (f"mu={fit['mu']:.3f} MIP\nsigma={fit['sigma']:.3f} MIP\nsigma/mu={fit['sigma_over_mu']:.4f}\n"
                f"events={fit['n_values']:,}")
        ax.plot(cx, cy, color=FIT_COLOR, linewidth=1.8, label="Gaussian fit", zorder=4)
        for x in (fit["fit_low"], fit["fit_high"]):
            ax.axvline(x, color=FIT_COLOR, linestyle="--", linewidth=0.9, alpha=0.65)
        finite = cy[np.isfinite(cy)]
        curve_max = float(finite.max()) if finite.size else 0.0
    else:  # points only
        text = "fit failed:\n" + textwrap.fill(str(fit.get("status", "")), 48)
    ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=FIT_COLOR,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": FIT_COLOR, "linewidth": 0.8,
                  "alpha": 0.94})
    top = 1.35 * max(float(counts[visible].max()) if visible.any() else 0.0, curve_max, float(min_count))
    ax.set_yscale("log")
    er.set_log_count_limits(ax, counts[visible], top=top)  # bottom min(8, 0.8 x smallest drawn count), never inverted
    ax.set_xlim(0.0, upper)
    ax.set_xlabel("Event energy [MIP]", fontsize=10.5)
    ax.set_ylabel("Events per bin", fontsize=10.5)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.0, colors=INK)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.0, loc="upper left")


# subtitle of a -fit page: quality cut, fit range and model, bins
def _range_page_subtitle(fit: dict, thr_text: str) -> str:
    double = fit.get("kind") == "range_double_gauss"
    model = "double Gaussian, one component per range" if double else "single Gaussian"
    text = (f"{thr_text}; --fit {fit.get('fit_ranges_text', '')} MIP ({model})\n"
            f"{fit.get('bin_rule', '')}; bins with >= {fit.get('min_bin_count', '')} entries fitted")
    width = fit.get("page_width", _NAN)
    if fit.get("fit_bins") and np.isfinite(width):
        text += f"; points scaled to the page bin width {width:.3g} MIP"
    return text


# quality-cut text of a spectrum page
def selection_text(s: dict) -> str:
    return (f"bad-strip fraction <= {s['quality_threshold']:g} ({s['selection_path_used']} flags)"
            if s["quality_filter"] else "no quality filter")


# event-energy spectrum of one run and path with its ci fit (library for the plots stage; ci draws no spectra)
def draw_spectrum(ax, values, fit: dict, particle: str, path: str, s: dict, title: str = "",
                  counts_text: str = "") -> None:
    from ..common import plotting as plot

    thr_text = selection_text(s)
    if fit.get("kind") in er.RANGE_FIT_KINDS:  # --fit[a,b] (single) | --fit[a,b],[c,d] (double)
        er.plot_spectrum_with_fit(ax, values, fit, plot.MIP_UNITS, title=title,
                                  subtitle=_range_page_subtitle(fit, thr_text))
    elif fit.get("kind") == "two_pass":
        subtitle = None
        if fit.get("centers") is not None:
            subtitle = (f"{thr_text}; {fit['bins']} equal-width bins; bin width = {fit['bin_width']:.3g} MIP;"
                        f" second range = mu1 +/- {fit['window_nsigma']:g} sigma1 "
                        f"({100 * fit['window_gaussian_coverage']:.2f}% Gaussian core)")
        er.plot_spectrum_with_fit(ax, values, fit, plot.MIP_UNITS, title=title, subtitle=subtitle)
    else:  # muon / pion: single Gaussian core fit
        _plot_particle_panel(ax, values, fit, path, s)
        ax.set_title(title, fontsize=15, color=INK, pad=24)
        lo, hi = s["fit_quantiles"]
        ax.text(0.5, 1.01, f"{thr_text}; single Gaussian core fit ({s.get(CORE_FIT_BINS.get(particle, ''), '')} "
                           f"bins on [q{lo:g}, q{hi:g}])" + (f", {counts_text}" if counts_text else ""),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.4, color=SUBTLE)


# footer of the linearity / resolution / e/h pages
def _dependence_footer(s: dict) -> str:
    normal = bool(s["energy_response"].get("normal_error"))
    return (f"charge_injection | {points_source_text(s)} of the selected events; "
            f"errors: {s['energy_response']['error_prescription']}"
            + ("; --normalerror: errors x sqrt(chi2/ndf) when > 1" if normal else ""))


# summary-table rows of the linearity pages
def _linearity_table_rows(p: str, lin: dict, normal: bool) -> list[dict]:
    rows = []
    for r in er.linearity_rows(lin) or [None]:
        if r is None:
            rows.append({"path": p, "particle": "", "fit range [GeV]": "no fit"})
            continue
        row = {"path": p, "particle": r["particle"], "fit range [GeV]": r["fit_range_gev"],
               "slope [MIP/GeV]": r["slope"], "slope error": r["slope_error"], "intercept [MIP]": r["intercept"],
               "chi2/ndf": r["chi2_over_ndf"], "fitted points": r["fitted_energies"]}
        if normal:
            row["slope error x sqrt(chi2/ndf)"] = r["slope_error_scaled"]
        rows.append(row)
    return rows


# energy_linearity.pdf / energy_resolution.pdf: one page per path + summary table
def plot_energy_dependence(path: Path, kind: str, dependence: dict, output_paths, s: dict, tag: str) -> None:
    import matplotlib.pyplot as plt

    from ..common import plotting as plot

    normal = bool(s["energy_response"].get("normal_error"))
    rows = []
    with atomic_pdf(path) as pdf:
        for p in output_paths:
            dep = dependence[p]
            fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
            if kind == "linearity":
                er.plot_linearity(ax, dep["linearity"], plot.MIP_UNITS, counts=dep["n"],
                                  title=f"Energy linearity - {path_label(p)}{tag}",
                                  no_point_reasons=dep.get("no_point_reasons"))
                rows += _linearity_table_rows(p, dep["linearity"], normal)
            else:
                res = dep["resolution"]
                er.plot_resolution(ax, res, plot.MIP_UNITS, title=f"Energy resolution - {path_label(p)}{tag}",
                                   no_point_reasons=dep.get("no_point_reasons"))
                row = {"path": p, "fit range [GeV]": _range_text(res.get("fit_range")) or "all points",
                       "a [MeV]": res["a"], "b [sqrt(MeV)]": res["b"], "c": res["c"],
                       "chi2/ndf": res["chi2_over_ndf"], "fit ok": bool(res["ok"])}
                if normal:
                    row["error scale"] = res["error_scale"]
                rows.append(row)
            er.add_footer(ax, _dependence_footer(s))
            plot.save(fig, pdf)
        er.summary_table_page(pdf, f"Energy {kind} per scale{tag}", rows)


# e_over_h.pdf: one page per s path (filled / hollow points, no fit) + a table of the points
def plot_e_over_h(path: Path, dependence: dict, output_paths, s: dict, tag: str) -> None:
    import matplotlib.pyplot as plt

    from ..common import plotting as plot

    rows = []
    with atomic_pdf(path) as pdf:
        for p in output_paths:
            eh = dependence[p]["eh"]
            if eh is None:
                continue
            fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
            er.plot_eh(ax, eh, p, title=f"e/h on the {p} scale - {path_label(p)}{tag}")
            er.add_footer(ax, _dependence_footer(s))
            plot.save(fig, pdf)
            rows += [{"path": p, "run": r["run"], "E [GeV]": r["energy_gev"], "e/h": r["e_over_h"],
                      "error": r["e_over_h_error"], "mu_electron from": r["electron_source"] or r["note"]}
                     for r in er.eh_rows(eh)]
        er.summary_table_page(pdf, f"e/h per s scale{tag}", rows)


# ==========================================================================
# BLOCK 6: parameters (fit / cache), CLI, io plan, run, product paths
# ==========================================================================
# Product locations (contract: per-channel constants -> calibration/, event csv (one per input csv, a sub-folder
# per run) -> data/, figures and check tables -> results/ (the final results: default paths here, the destination
# of a run is its output plan, see result_slots), intermediates and the per-event / per-bar arrays for the plots
# stage -> cache/; [test/] = the test/ folder of each tree, only in test mode).  Every function takes the variant
# ``tag`` of the run that wrote the file ("" = default run; see variant_tag) and an optional ``mode`` (default:
# the mode of ctx).  They only build paths; the io_utils writers create folders.
RESULT_ORDER = ("response", "quality_scan", "linearity", "resolution", "eh")   # the order the run writes them
RESULT_FILES = {"response": ("energy_response", "csv"), "quality_scan": ("quality_threshold_scan", "csv"),
                "linearity": ("energy_linearity", "pdf"), "resolution": ("energy_resolution", "pdf"),
                "eh": ("e_over_h", "pdf")}
RESULT_WHAT = {"response": "spectrum, linearity, resolution and e/h fit numbers per run and path (check)",
               "quality_scan": "peak, resolution and retention of the electron runs per bad-strip threshold (check)",
               "linearity": "linearity: one page per scale (electron fit, other particles' points) + table",
               "resolution": "resolution: one page per scale (electron fit, pion points) + table",
               "eh": "e/h on the s scale: filled / hollow points, no fit + table"}
RESULTS_RECORD = "final_results"               # cache/[test/]charge_injection/final_results<v>.json
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")  # --output-tag: file-name safe, no separators
# automatic variant-tag parts (variant_tag) -> the flag that sets them.  An --output-tag equal to one of
# them, or starting with one of them + "_", is refused (case-insensitive: Windows file names); the clean
# part is appended LAST, so a tag equal to "clean" or ending with "_clean" is refused as well.  Every file
# name therefore maps back to exactly one combination of variant flags.
RESERVED_TAG_PARTS = {f"s_{p}": f"--scale-particle {p}" for p in SCALE_PARTICLES if p != DEFAULT_SCALE_PARTICLE}
# refused as an --output-tag too: the default (pion) training alone has no tag, with other samples it leads the tag
DEFAULT_TAG_PARTS = {f"s_{DEFAULT_SCALE_PARTICLE}": f"--scale-particle {DEFAULT_SCALE_PARTICLE} with other samples "
                                                    f"(alone: untagged files)"}
CLEAN_TAG_PART = "clean"                        # -clean-input (appended last)
TAG_PART_FLAGS = {**RESERVED_TAG_PARTS, **DEFAULT_TAG_PARTS, CLEAN_TAG_PART: "--clean-input"}


# file-name suffix of a variant tag: 's_muon' -> '_s_muon'
def tag_suffix(tag) -> str:
    tag = str(tag or "").strip().lstrip("_")
    return f"_{tag}" if tag else ""


# training samples of the s fit as a list: 'pion' -> ['pion'], ['pion', 'muon'] as given
def scale_samples(value) -> list[str]:
    return [value] if isinstance(value, str) else [str(x) for x in value]


# variant tag: s_<samples> (not for pion alone), -output-tag, clean, joined by '_'
def variant_tag(s: dict) -> str:
    parts = []
    samples = s.get("scale_particles") or scale_samples(s.get("scale_particle", DEFAULT_SCALE_PARTICLE))
    if list(samples) != [DEFAULT_SCALE_PARTICLE]:
        parts.append("s_" + "_".join(samples))
    tag = str(s.get("output_tag") or "").strip().lstrip("_")
    if tag:
        parts.append(tag)
    if s.get("clean_input"):
        parts.append(CLEAN_TAG_PART)
    return "_".join(parts)


# the automatic tag part an -output-tag collides with, else None
def reserved_tag_clash(tag) -> str | None:
    low = str(tag or "").strip().lstrip("_").lower()
    hit = next((r for r in (*RESERVED_TAG_PARTS, *DEFAULT_TAG_PARTS) if low == r or low.startswith(r + "_")), None)
    if hit is None and (low == CLEAN_TAG_PART or low.endswith("_" + CLEAN_TAG_PART)):
        hit = CLEAN_TAG_PART
    return hit


# provenance path next to a product: x.csv -> x.provenance.json
def provenance_path(product: Path) -> Path:
    product = Path(product)
    return product.with_name(product.stem + ".provenance.json")


# provenance record of a product, or None
def read_provenance(product: Path) -> dict | None:
    record = read_json(provenance_path(product))
    return record if isinstance(record, dict) else None


# folder calibration/[test/]charge_injection/
def calib_stage_dir(ctx, mode: str | None = None) -> Path:
    return ctx.calib_dir(STAGE_DIR, mode=mode)


# folder data/[test/]charge_injection/
def data_stage_dir(ctx, mode: str | None = None) -> Path:
    return ctx.data_dir(STAGE_DIR, mode=mode)


# folder results/<fullset|test>/charge_injection/
def results_stage_dir(ctx, mode: str | None = None) -> Path:
    return ctx.results_dir(STAGE_DIR, mode=mode)


# folder cache/[test/]charge_injection/<parts>
def cache_stage_dir(ctx, *parts: str, mode: str | None = None) -> Path:
    return ctx.cache_dir(STAGE_DIR, *parts, mode=mode)


# event csv of one input csv: data/[test/]charge_injection/<run>/<csv stem>_event_energies<v>.csv
def event_energies_csv(ctx, ds, csv, tag: str = "", mode: str | None = None) -> Path:
    return ctx.csv_product(data_stage_dir(ctx, mode), ds, csv, f"_event_energies{tag_suffix(tag)}.csv")


# path of ci_channel_parameters<v>.csv in calibration/[test/]charge_injection/
def parameters_csv(ctx, tag: str = "", mode: str | None = None) -> Path:
    return calib_stage_dir(ctx, mode) / f"ci_channel_parameters{tag_suffix(tag)}.csv"


# path of ci_tot_charge_nodes<v>.csv in calibration/[test/]charge_injection/
def nodes_csv(ctx, tag: str = "", mode: str | None = None) -> Path:
    return calib_stage_dir(ctx, mode) / f"ci_tot_charge_nodes{tag_suffix(tag)}.csv"


# provenance path of the official parameter + node csv
def parameters_record(ctx, tag: str = "", mode: str | None = None) -> Path:
    return calib_stage_dir(ctx, mode) / f"ci_channel_parameters{tag_suffix(tag)}.provenance.json"


# folder cache/[test/]charge_injection/parameters<v>/ (reusable CI map + s fit)
def parameters_cache_dir(ctx, tag: str = "", mode: str | None = None) -> Path:
    return cache_stage_dir(ctx, f"parameters{tag_suffix(tag)}", mode=mode)


# files of the parameter cache: parameter csv, node csv, parameters.json
def parameters_cache_files(ctx, tag: str = "", mode: str | None = None) -> tuple[Path, Path, Path]:
    d = parameters_cache_dir(ctx, tag, mode)
    return d / "ci_channel_parameters.csv", d / "ci_tot_charge_nodes.csv", d / "parameters.json"


# folder cache/[test/]charge_injection/events<v>/ (event and bar arrays: -from-saved and the plots stage)
def event_cache_dir(ctx, tag: str = "", mode: str | None = None) -> Path:
    return cache_stage_dir(ctx, f"events{tag_suffix(tag)}", mode=mode)


# cached arrays of one input csv: <run>/<csv stem>.npz (events), <csv stem>_bars.npy (bars), <csv stem>.json
def event_cache_files(ctx, ds, csv, tag: str = "", mode: str | None = None) -> tuple[Path, Path, Path]:
    d = event_cache_dir(ctx, tag, mode)
    return (ctx.csv_product(d, ds, csv, ".npz"), ctx.csv_product(d, ds, csv, "_bars.npy"),
            ctx.csv_product(d, ds, csv, ".json"))


# per-event arrays of one cached csv (plots stage, -from-saved)
def load_event_arrays(npz) -> dict:
    with np.load(npz, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if not k.startswith("energy_")}
        out["n_events_in_file"] = int(out["n_events_in_file"])
        out["energy"] = {k[len("energy_"):]: z[k] for k in z.files if k.startswith("energy_")}
    return out


# per-bar records of one cached csv, memory-mapped (plots stage)
def load_bar_records(npy, mmap: bool = True) -> np.ndarray:
    return np.load(npy, mmap_mode="r" if mmap else None, allow_pickle=False)


# default path of a final result: results/<mode>/charge_injection/<file><v>.<kind>
def result_default(ctx, unit: str, tag: str = "", mode: str | None = None) -> Path:
    stem, kind = RESULT_FILES[unit]
    return results_stage_dir(ctx, mode) / f"{stem}{tag_suffix(tag)}.{kind}"


# why a chosen result unit is not made for these runs and paths ('' when it is)
def unit_skip_reason(unit: str, s: dict, datasets) -> str:
    if not datasets:
        return "no run selected"
    if unit == "quality_scan" and not any(ds.particle == "electron" for ds in datasets):
        return "the quality scan uses the electron runs: none selected"
    if unit == "eh":
        if not any(is_scale_path(p) for p in s["output_paths"]):
            return "e/h is drawn on the s scale: the s path is not reconstructed (-s)"
        if not any(ds.particle == er.HADRON for ds in datasets):
            return f"e/h needs a {er.HADRON} run: none selected"
    return ""


# final results of a run (for -output): the chosen check tables and figures, in RESULT_ORDER
def result_slots(ctx, s: dict, datasets, units) -> list:
    vtag = s["variant_tag"]
    return [outputs.ResultSlot(u, RESULT_FILES[u][1], result_default(ctx, u, vtag), RESULT_WHAT[u])
            for u in RESULT_ORDER if u in units and not unit_skip_reason(u, s, datasets)]


# output plan of the stage (an -output name given twice stops the run)
def _output_plan(ctx, slots, preview: bool = False) -> "outputs.OutputPlan":
    try:
        return outputs.OutputPlan(ctx, STAGE_DIR, slots, preview=preview)
    except ValueError as exc:  # two results would be written to one file
        raise SettingsError(f"ci: {exc}") from None


# output plan of the run: -output names, test names or the unsaved scratch copy
def result_plan(ctx, s: dict, datasets, units, preview: bool = False) -> "outputs.OutputPlan":
    return _output_plan(ctx, result_slots(ctx, s, datasets, units), preview)


# dry-run / check items of the final results at their planned destinations
def result_items(ctx, s: dict, datasets, units) -> list[dict]:
    plan = result_plan(ctx, s, datasets, units, preview=True)
    for w in plan.warnings:
        log(f"WARNING: {w}")
    return [{"role": "out", "path": plan.path(x.key), "result": x.key, "saved": plan.saved(x.key),
             "note": f"final result {line}" if plan.saved(x.key) else "(not saved: no --output)"}
            for x, line in zip(plan.slots, plan.summary_lines())]


# path of final_results<v>.json: where the last run put its final results
def results_record(ctx, tag: str = "", mode: str | None = None) -> Path:
    return cache_stage_dir(ctx, mode=mode) / f"{RESULTS_RECORD}{tag_suffix(tag)}.json"


# write final_results<v>.json (destinations, run_id, fit flags)
def _write_results_record(ctx, s: dict, datasets, plan, run_id: str | None, event_csvs: list) -> Path:
    vtag = s["variant_tag"]
    results = {}
    for slot in plan.slots:
        path = plan.path(slot.key)
        results[slot.key] = {"kind": slot.kind, "path": str(path), "default": str(slot.default),
                             "saved": bool(plan.saved(slot.key)), "what": slot.what,
                             "signature": _file_signature(path) if path.exists() else None}
    return write_json({"stage": "ci", "mode": ctx.mode, "variant_tag": vtag,
                       "written": time.strftime("%Y-%m-%d %H:%M:%S"), "units": list(s["units"]),
                       "output_names": getattr(ctx, "outputs", None), "run_id": run_id,
                       "from_saved": bool(s.get("from_saved")), "datasets": [d.key for d in datasets],
                       "event_energies_csv": [str(p) for p in event_csvs],
                       "energy_response": {k: s["energy_response"].get(k) for k in er.FLAG_KEYS},
                       "order": plan.order_text(), "results": results}, results_record(ctx, vtag))


# where the last run of a variant wrote a final result (else its default path)
def result_file(ctx, key: str, tag: str = "", mode: str | None = None) -> Path:
    record = read_json(results_record(ctx, tag, mode))
    entry = (record.get("results") or {}).get(key) if isinstance(record, dict) else None
    if isinstance(entry, dict) and entry.get("path"):
        return Path(entry["path"])
    if key not in RESULT_FILES:
        raise KeyError(f"ci: unknown final result {key!r} (one of {sorted(RESULT_FILES)})")
    return result_default(ctx, key, tag, mode)


# log where the final results went (or that they were not saved)
def _log_final_results(plan) -> None:
    unsaved = [x for x in plan.slots if not plan.saved(x.key)]
    if unsaved:
        log(f"final results not saved (no --output): scratch copies in {plan.path(unsaved[0].key).parent}; use "
            f"--output [names] to save them")
    for slot, line in zip(plan.slots, plan.summary_lines()):
        if plan.saved(slot.key):
            log(f"  final result {line}")


# ci command-line options
def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("ci stage (defaults: config section 'charge_injection')")
    outputs.add_only_argument(p, UNITS, groups=UNIT_GROUPS, default=DEFAULT_UNITS)
    g.add_argument("--adc-only", "-adc-only", dest="adc_only", action="store_true",
                   help="reconstruct the adc_only scale (no path flag: all three)")
    g.add_argument("--pureci", "-pureci", dest="pureci", action="store_true",
                   help="reconstruct the PureCI scale (charge injection with s = 1)")
    g.add_argument("--s", "-s", dest="s_path", action="store_true",
                   help="reconstruct the s scale (charge injection x the TOT scale s; s parameter, Lei)")
    g.add_argument("--selection-path", "-selection-path", choices=SELECTION_PATHS, default=None,
                   help="scale whose bad-strip flags select events (default s, computed even if not requested; "
                        "auto = the most complete requested scale)")
    g.add_argument("--quality-threshold", "-quality-threshold", type=float, default=None, metavar="X",
                   help=f"keep events with bad_strip_fraction <= X (default {DEFAULTS['quality_threshold']})")
    g.add_argument("--no-quality-filter", "-no-quality-filter", dest="no_quality_filter", action="store_true",
                   help="keep every reconstructed event (default: filter on)")
    g.add_argument("--bad-bar-rule", "-bad-bar-rule", choices=BAD_BAR_RULES, default=None,
                   help="bad bar of the 1/8 event rule: used_ends = no end the bar energy uses is trusted and an "
                        "end is saturated with an unconvertible TOT (Lei D4); used_ends_strict = no used end "
                        "trusted (default used_ends)")
    g.add_argument("--scale-particle", "-scale-particle", nargs="+", choices=SCALE_PARTICLES, default=None,
                   metavar="SAMPLE",
                   help="training sample(s) of the s fit: pion = pion 4 (default), muon = muon 4, electron = "
                        "electron 4, mix = the hits of pion 4 + muon 4 + electron 4 pooled into one fit (the runs "
                        "of each sample: yml charge_injection.scale_training_datasets, any run). Several samples "
                        "(e.g. -scale-particle pion muon) give one path s_<sample> each; s is the sample with the "
                        "fewest channels that borrow s from the (DPM, LINK) / global prior. Anything but pion alone "
                        "writes *_s_<samples> files")
    g.add_argument("--shrinkage", "-shrinkage", default=None, metavar="L",
                   help="s shrinkage lambda (w = n/(n+L)), a number >= 0 or 'cv' (default 5)")
    g.add_argument("--tot-sample-mode", "-tot-sample-mode", choices=TOT_SAMPLE_MODES, default=None,
                   help="CI scan TOT of an injection: sum8 = sum of the 8 samples; sum8_mask2048 = spurious "
                        "2048 samples set to 0 first (default sum8)")
    g.add_argument("--no-audit-failures", "-no-audit-failures", dest="no_audit_failures", action="store_true",
                   help="skip the 1e4-bound pole audit whose failures fail L12 S4-7 (default off)")
    g.add_argument("--min-link-hits", "-min-link-hits", type=int, default=None, metavar="N",
                   help="a (DPM, LINK) prior of s needs >= N exact hits, else the global prior (default 0 = any)")
    g.add_argument("--refit", "-refit", action="store_true",
                   help="recompute the CI map and the s fit even if the cached parameters in "
                        "cache/[test/]charge_injection/ match (default off)")
    g.add_argument("--output-tag", "-output-tag", default=None, metavar="TAG",
                   help="extra variant suffix of every product file, e.g. q0p1 -> *_event_energies_q0p1.csv "
                        "(default none; letters, digits, '.', '_', '-'; the automatic parts s_pion, s_muon, "
                        "s_electron, s_mix (+ '_...') and clean ('clean', '..._clean') are reserved); with "
                        "-from-saved and no event arrays stored under the tag, the untagged arrays are read and only "
                        "the products of this run carry the tag")
    er.add_fit_arguments(p)


# -shrinkage value: a number or 'cv'
def _parse_shrinkage(value):
    if isinstance(value, str) and value.strip().lower() == "cv":
        return "cv"
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"ci: --shrinkage must be a number >= 0 or 'cv', not {value!r}")


# finite number (not a bool)?
def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


# settings check of shrinkage_lambda
def _shrinkage_rule(v):
    if v == "cv" or (_number(v) and float(v) >= 0):
        return None
    return "a number >= 0 or 'cv'"


# settings check of a quantile pair
def _quantile_window_rule(v):
    ok = isinstance(v, (list, tuple)) and len(v) == 2 and all(_number(x) for x in v) and 0 <= v[0] < v[1] <= 1
    return None if ok else "an ordered pair [low, high] with 0 <= low < high <= 1"


# settings check of a number in (0, 1]
def _open_unit_rule(v):
    return None if _number(v) and 0 < float(v) <= 1 else "a number in (0, 1]"


# settings check of a [layer, strip] list
def _bar_list_rule(v):
    ok = isinstance(v, (list, tuple)) and all(
        isinstance(x, (list, tuple)) and len(x) == 2 and all(_number(y) and float(y).is_integer() for y in x)
        for x in v)
    return None if ok else "a list of [layer, strip] integer pairs"


# settings check of scale_particle: one sample or a list of distinct samples (s parameter, Lei)
def _scale_particle_rule(v):
    values = [v] if isinstance(v, str) else v
    ok = (isinstance(values, (list, tuple)) and values and all(isinstance(x, str) and x in SCALE_PARTICLES
                                                               for x in values)
          and len(set(values)) == len(values))
    return None if ok else f"one of {list(SCALE_PARTICLES)} or a list of distinct ones, e.g. [pion, muon]"


# settings check of scale_training_datasets (s parameter, Lei)
def _training_sets_rule(v):
    if not isinstance(v, dict):
        return f"a mapping {{sample: run reference(s), e.g. 'pion 4'}} for {list(SCALE_PARTICLES)}"
    missing = [k for k in SCALE_PARTICLES if k not in v]
    bad = [k for k, x in v.items() if not (isinstance(x, str) and x) and not (
        isinstance(x, (list, tuple)) and x and all(isinstance(y, str) and y for y in x))]
    if missing or bad or set(v) - set(SCALE_PARTICLES):
        return (f"a mapping {{sample: run reference or list of references, e.g. 'pion 4' or ['pion 4', 'muon 4']}} "
                f"with exactly the samples {list(SCALE_PARTICLES)} (missing {missing}, bad {bad})")
    return None


# settings check of a non-empty number list
def _nonempty(rule):
    # check one value
    def check(v):
        if not isinstance(v, (list, tuple)) or not v:
            return "a non-empty list"
        bad = [x for x in v if not (_number(x) and (rule != "float>=0" or float(x) >= 0))]
        return f"a non-empty list of numbers >= 0 (bad: {bad})" if bad else None
    return check


# numeric / choice settings, checked before any file is read (io_utils.validate_settings rules)
SETTINGS_RULES = {
    "quality_filter": "bool", "quality_threshold": "float>=0", "quality_tolerance": "float>=0",
    "adc_max_rail": "float", "adc_max_rail_guard": "float", "chunk_rows": "int>0",
    "linearity_source": ("choice", ["two_pass", "peak_scan"]), "muon_fit_bins": "int>0", "pion_fit_bins": "int>0",
    "fit_quantiles": _quantile_window_rule, "display_upper_quantile": _open_unit_rule, "display_min_bins": "int>0",
    "scan_threshold_grid": _nonempty("float>=0"), "scan_shoulder_range_mip": "window",
    "scan_core_range_mip": "window", "selection_path": ("choice", list(SELECTION_PATHS)),
    "scale_particle": _scale_particle_rule, "scale_training_datasets": _training_sets_rule, "refit": "bool",
    "bar_key_size": "int>0", "known_bad_bars": _bar_list_rule,
    "bad_bar_rule": ("choice", list(BAD_BAR_RULES)), "tot_sample_mode": ("choice", list(TOT_SAMPLE_MODES)),
    "tot_spurious_sample": "int>=0", "adc_consist": "float>0", "tot_consist_low": "float>0",
    "tot_consist_high": "float>0", "min_repeats": "int>0", "dac_fullscale": "int>0", "lowcap_fc": "float>0",
    "highcap_fc": "float>0", "lead_dac_max": "int>0", "min_tot_region_points": "int>0",
    "pole_asym_points": "int>0", "pole_a_bound": "float>0", "pole_b_margin_fc": "float>=0",
    "outlier_resid": "float>0", "audit_failures": "bool", "pole_a_bound_audit": "float>0",
    "audit_probe_tot": "float>0", "audit_grid_n": "int>0", "q_grid_n": "int>0", "q_max_fc": "float>0",
    "tot_rail_lo": "float>0", "min_reliable_points": "int>0", "min_monotone_nodes": "int>0",
    "pava_tie_tol": "float>=0", "endpoint_slope_points": "int>0", "endpoint_slope_fallback": "float>0",
    "thr_min_repeats": "int>0", "thr_adc_consist": "float>0", "thr_tot_consist_low": "float>0",
    "thr_tot_consist_high": "float>0", "thr_asym_points": "int>0", "thr_a_bound": "float>0",
    "thr_b_margin_fc": "float>=0", "ratio_max": "float>0", "shrinkage_lambda": _shrinkage_rule,
    "lambda_grid": _nonempty("float>=0"), "pointwise_loss_guard": "float>0", "cv_fallback_lambda": "float>=0",
    "cv_diagnostics": "bool", "min_link_hits": "int>=0", "training_chunksize": "int>0",
}

# ci flags that decide the stored event arrays (or which copy -from-saved reads).  The rebuild command of a
# refused -from-saved run repeats the ones given on its command line (the rebuild reads the same yml, so yml
# values need no flag): (argparse dest, flag).
REBUILD_FLAGS = (("adc_only", "-adc-only"), ("pureci", "-pureci"), ("s_path", "-s"),
                 ("selection_path", "-selection-path"), ("bad_bar_rule", "-bad-bar-rule"),
                 ("tot_sample_mode", "-tot-sample-mode"), ("no_audit_failures", "-no-audit-failures"),
                 ("scale_particle", "-scale-particle"), ("shrinkage", "-shrinkage"),
                 ("min_link_hits", "-min-link-hits"), ("output_tag", "-output-tag"))
# the user fit-range flags (Lei 2026-10-02; argparse dest, flag): they change only the fits, never the event arrays,
# but the rebuild command repeats the ones of the command line, in the user syntax (range_flag_text), so that the
# rebuild run draws the same fits as the -from-saved run it prepares
RANGE_REBUILD_FLAGS = (("fit_ranges", "-fit"), ("fit_bins", "-bin"), ("linearity_fits", "-linfit"),
                       ("resolution_fit_range", "-resfit"))


# a fit-range flag as the user types it: -fit[5,10], -bin 30,40, -linfit electron[0,1] pion, -resfit[0.3,1]
def range_flag_text(flag: str, value) -> str:
    name = flag.lstrip("-")
    if name == "bin":
        return f"{flag} {_bins_text(value)}"
    if name == "linfit":
        if not value:
            return f"{flag} none"
        return flag + " " + " ".join(
            p + ("" if r is None else f"[{_flag_value(float(r[0]))},{_flag_value(float(r[1]))}]")
            for p, r in value.items())
    pairs = value if name == "fit" else [value]
    return flag + ",".join(f"[{_flag_value(float(a))},{_flag_value(float(b))}]" for a, b in pairs)


# reconstructed paths of the outputs: the chosen scales, s followed by s_<sample> with several samples
def expand_paths(paths, samples) -> list[str]:
    out = []
    for p in PATHS:
        if p in paths:
            out.append(p)
            if p == "s" and len(samples) > 1:
                out += [sample_path(x) for x in samples]
    return out


# ci settings: defaults < yml section charge_injection < command line, checked
def resolve_settings(ctx, args) -> dict:
    s = copy.deepcopy(DEFAULTS)
    yml = copy.deepcopy(ctx.stage_cfg(CONFIG_SECTION))
    unknown = sorted(set(yml) - set(DEFAULTS))
    if unknown:  # a typo is never silently ignored
        raise SettingsError(f"ci: unknown setting(s) {unknown} in the yml section {CONFIG_SECTION} (configure/"
                            f"config.yml or local.yml); the valid keys are those of charge_injection.DEFAULTS")
    s.update(yml)
    chosen = [p for p, flag in (("adc_only", "adc_only"), ("pureci", "pureci"), ("s", "s_path"))
              if getattr(args, flag, False)]
    if chosen:
        s["energy_paths"] = chosen
    cli = {"selection_path": getattr(args, "selection_path", None),
           "quality_threshold": getattr(args, "quality_threshold", None),
           "bad_bar_rule": getattr(args, "bad_bar_rule", None),
           "scale_particle": getattr(args, "scale_particle", None),
           "shrinkage_lambda": getattr(args, "shrinkage", None),
           "tot_sample_mode": getattr(args, "tot_sample_mode", None),
           "min_link_hits": getattr(args, "min_link_hits", None),
           "output_tag": getattr(args, "output_tag", None)}
    s.update({k: v for k, v in cli.items() if v is not None})
    for key, attr, value in (("quality_filter", "no_quality_filter", False), ("refit", "refit", True),
                             ("audit_failures", "no_audit_failures", False)):
        if getattr(args, attr, False):
            s[key] = value
    s["shrinkage_lambda"] = _parse_shrinkage(s["shrinkage_lambda"])
    for key in ("tot_sample_mode", "bad_bar_rule"):
        s[key] = str(s[key]).lower()
    validate_settings(s, SETTINGS_RULES, source="ci (charge_injection settings)")
    bad = [p for p in s["energy_paths"] if p not in PATHS]
    if bad or not s["energy_paths"]:
        raise SettingsError(f"ci: energy_paths {s['energy_paths']} must be a non-empty subset of {list(PATHS)}")
    s["scale_particles"] = scale_samples(s["scale_particle"])
    s["energy_paths"] = [p for p in PATHS if p in s["energy_paths"]]
    selection = s["selection_path"]
    if selection == "auto":
        selection = next(p for p in SELECTION_PREFERENCE if p in s["energy_paths"])
    s["selection_path_used"] = selection
    s["output_paths"] = expand_paths(s["energy_paths"], s["scale_particles"])
    s["compute_paths"] = s["output_paths"] + ([selection] if selection not in s["output_paths"] else [])
    tag = str(s["output_tag"] or "").strip().lstrip("_")
    if tag and not _TAG_PATTERN.fullmatch(tag):
        raise SettingsError(f"ci: --output-tag {s['output_tag']!r} must use only letters, digits, '.', '_' and '-'")
    clash = reserved_tag_clash(tag)
    if clash:
        raise SettingsError(f"ci: --output-tag {s['output_tag']!r} is reserved: '{clash}' is the automatic variant "
                            f"tag of {TAG_PART_FLAGS[clash]}; use that flag, or choose another tag")
    s["output_tag"] = tag or None
    s["clean_input"] = bool(getattr(ctx, "clean_input", False))
    s["from_saved"] = bool(getattr(args, "from_saved", False))
    s["variant_tag"] = variant_tag(s)
    given = {dest: (s["output_tag"] if dest == "output_tag" else getattr(args, dest, None))
             for dest, _ in REBUILD_FLAGS}
    s["_rebuild_flags"] = [(flag, given[dest]) for dest, flag in REBUILD_FLAGS if given[dest] not in (None, False, "")]
    s["energy_response"] = er.settings_from_args(ctx, args)  # --fit / --bin / --linfit / --resfit checked here
    s["_rebuild_range_flags"] = [range_flag_text(flag, getattr(args, dest)) for dest, flag in RANGE_REBUILD_FLAGS
                                 if getattr(args, dest, None) is not None]
    s["units"] = call_units(ctx, s)
    return s


# output units of this call: -only, without the figures (-no-plots) and the parameters (-from-saved)
def call_units(ctx, s: dict) -> tuple:
    chosen = [u for u in UNITS if ctx.wants(u)]
    units = tuple(u for u in chosen if not (u in FIGURE_UNITS and not ctx.make_plots)
                  and not (u == "parameters" and s["from_saved"]))
    if not units:
        why = []
        if any(u in FIGURE_UNITS for u in chosen):
            why.append("the figures need plots (no -no-plots)")
        if "parameters" in chosen:
            why.append("-from-saved reads the stored event arrays and fits no parameters")
        raise SettingsError(f"ci: -only {', '.join(chosen)} makes nothing in this call: {'; '.join(why)}")
    return units


# does the call reconstruct events (or read them with -from-saved)?
def needs_events(s: dict) -> bool:
    return any(u in EVENT_UNITS for u in s["units"])


# does the call need the CI map (a CI scale reconstructed, or the parameters unit)?
def needs_ci(s: dict) -> bool:
    return "parameters" in s["units"] or (needs_events(s) and any(p != "adc_only" for p in s["compute_paths"]))


# does the call need the s fit (an s scale reconstructed, or the parameters unit)?
def needs_scale(s: dict) -> bool:
    return "parameters" in s["units"] or (needs_events(s) and any(is_scale_path(p) for p in s["compute_paths"]))


# expand ${name} from paths: and resolve relative to data_analysis
def _expand_path(value, ctx) -> Path | None:
    if value in (None, ""):
        return None
    paths = dict(ctx.cfg.get("paths", {}))
    text = _VAR.sub(lambda m: str(paths.get(m.group(1), m.group(0))), str(value))
    return Path(os.path.normpath(str(resolve_path(text))))


_TRAINING_NOTES: set = set()   # notes on the training runs already logged by this process (said once per run)


# log a training-run note once
def _training_note(text: str) -> None:
    if text not in _TRAINING_NOTES:
        _TRAINING_NOTES.add(text)
        log(text)


# training runs of one sample of the s fit (s parameter, Lei): scale_training_datasets[sample]
def _training_datasets(ctx, s, sample: str) -> list:
    """Any run is accepted; a list (or a reference naming several runs) is pooled."""
    refs = s["scale_training_datasets"][sample]
    refs = [refs] if isinstance(refs, str) else list(refs)
    where = f"charge_injection.scale_training_datasets.{sample}"
    out = []
    for ref in refs:
        found = resolve_runs(ctx.cfg, ref, f"ci: {where}")
        if len(found) > 1:
            _training_note(f"ci: {where}: {ref!r} names {len(found)} runs ({', '.join(run_ref(d) for d in found)}); "
                           f"the exact-overlap hits of all of them are pooled into the one s fit")
        seen = {d.key for d in out}
        twice = [d for d in found if d.key in seen]
        if twice:
            _training_note(f"ci: WARNING: {where}: {', '.join(run_ref(d) for d in twice)} named twice ({refs}); "
                           f"the hits of a run are pooled once")
        new = [d for d in found if d.key not in seen]
        other = [d for d in new if sample in RUN_PARTICLES and (d.particle != sample or d.kind != "data")]
        if other:
            _training_note(f"ci: WARNING: {where}: the {sample} sample trains s on "
                           f"{', '.join(run_ref(d) for d in other)}, not on {sample} data; used as given")
        out += new
    return out


# path of a CI scan csv (a bare name is looked up in paths.ci_scan_dir)
def _scan_file_path(value, ctx) -> Path | None:
    path = _expand_path(value, ctx)
    scan_dir = ctx.cfg.get("paths", {}).get("ci_scan_dir")
    if path is not None and not path.exists() and scan_dir and "${" not in str(value) \
            and not Path(str(value)).is_absolute():
        return _expand_path(str(Path(str(scan_dir)) / str(value)), ctx)
    return path


# input files of the CI map and the s fit (training csv files only when asked; nothing is read)
def _parameter_inputs(ctx, s, need_scale: bool, training_csv: bool = True) -> dict:
    inputs = {"scan_files": [_scan_file_path(f, ctx) for f in s["scan_files"]],
              "channel_map": (_expand_path(s["channel_map_csv"], ctx) if s["channel_map_csv"]
                              else ctx.path("channel_map")),
              "training": {}}
    if need_scale:
        for sample in s["scale_particles"]:
            trains = _training_datasets(ctx, s, sample)
            files = [f for t in trains for f in ctx.input_csvs(t)] if training_csv else []
            inputs["training"][sample] = (trains, files)
    return inputs


# file signature (path, size, mtime), None for an unused input
def _file_signature(path) -> dict | None:
    return None if path is None else file_signature(Path(path))


# size + content hash of a file
def _content_identity(path) -> dict:
    digest, method = content_hash(Path(path))
    return {"size": int(Path(path).stat().st_size), "sha256": digest, "hash": method}


# json fallback for numpy values
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# canonical json text
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default)


# an object as it reads back from json
def _as_json(obj):
    return json.loads(_canonical(obj))


# cache signatures of the CI map and the s fit (inputs, settings, algorithm version)
def parameter_signatures(ctx, s: dict, calibration_path: Path, need_ci: bool, need_scale: bool) -> dict:
    if not (need_ci or need_scale):
        return {"kind": "none", "inputs": {}, "ci_signature": None, "scale_signature": None,
                "identity": {"kind": "none"}}
    inputs = _parameter_inputs(ctx, s, need_scale)
    for f in inputs["scan_files"]:
        require(f, "the charge-injection scan (charge_injection.scan_files in configure/config.yml)")
    require(inputs["channel_map"], "configure/config.yml (paths.channel_map)")
    ci_sig = _as_json({"algorithm": CI_MAP_ALGORITHM, "settings": {k: s[k] for k in sorted(CI_MAP_SETTING_KEYS)},
                       "scan_files": [_file_signature(f) for f in inputs["scan_files"]],
                       "channel_map": _file_signature(inputs["channel_map"]),
                       "calibration": _content_identity(calibration_path)})   # calib rewrites it every run
    scale_sig = None
    if need_scale:
        training = {}
        for sample, (trains, files) in inputs["training"].items():
            files = [require(f, f".\\runner.ps1 make-csv {' '.join(run_flags(t) for t in trains)}") for f in files]
            training[sample] = {"datasets": [d.key for d in trains], "csv": [_file_signature(f) for f in files]}
        scale_sig = _as_json({"ci": ci_sig, "algorithm": SCALE_ALGORITHM,
                              "settings": {k: s[k] for k in SCALE_SETTING_KEYS},
                              "samples": list(s["scale_particles"]), "training": training})
    ident = {"kind": "chain", "ci_signature_hash": settings_hash(ci_sig),
             "scale_signature_hash": settings_hash(scale_sig) if scale_sig is not None else None}
    return {"kind": "chain", "inputs": inputs, "ci_signature": ci_sig, "scale_signature": scale_sig,
            "identity": ident}


# read a stored parameter table (round-trip floats, checked keys)
def _read_parameter_csv(path: Path, source: str) -> pd.DataFrame:
    return checked_channel_table(pd.read_csv(path, float_precision="round_trip"), source)


# copy the cached parameter + node csv to calibration/[test/]charge_injection/ (unit parameters)
def _write_official_parameters(ctx, tag: str, cache_files: tuple, record: dict, source: str) -> list[Path]:
    cache_csv, cache_nodes, cache_meta = cache_files
    expected = {"schema": PARAMETER_SCHEMA, "ci_signature_hash": settings_hash(record["ci_signature"]),
                "scale_signature_hash": (settings_hash(record["scale_signature"])
                                         if record.get("scale_signature") else None)}
    official = parameters_record(ctx, tag)
    ok, _ = product_set_valid(official, expected, members=[parameters_csv(ctx, tag), nodes_csv(ctx, tag)])
    if not ok:
        with ProductSet(official) as ps:
            shutil.copyfile(cache_csv, ps.stage(parameters_csv(ctx, tag)))
            shutil.copyfile(cache_nodes, ps.stage(nodes_csv(ctx, tag)))
            ps.commit({**expected, "stage": "ci", "variant_tag": tag, "parameter_source": source,
                       "cache_record": str(cache_meta), "scale_diagnostics": record.get("scale_diagnostics"),
                       "s_choice": record.get("s_choice"), "scale_signature": record.get("scale_signature"),
                       "content": "per-channel CI map + s (ci_channel_parameters) and the monotone q(TOT) nodes",
                       "columns": {"ci_channel_parameters": list(pd.read_csv(cache_csv, nrows=0).columns),
                                   "ci_tot_charge_nodes": NODE_COLUMNS},
                       "column_doc": PARAMETER_COLUMN_DOC, "software": python_info()})
        log(f"CI parameters: {parameters_csv(ctx, tag).name}, {nodes_csv(ctx, tag).name}")
    else:
        log(f"CI parameters: {parameters_csv(ctx, tag).name}, {nodes_csv(ctx, tag).name} already current")
    return [parameters_csv(ctx, tag), nodes_csv(ctx, tag)]


# s fit of every training sample and the default s (s parameter, Lei) -> (scale columns, diagnostics, choice)
def fit_scale_samples(ctx, s: dict, table: pd.DataFrame, nodes: pd.DataFrame, calibration: pd.DataFrame,
                      training: dict) -> tuple[pd.DataFrame, dict, dict]:
    scales = {}
    for sample in s["scale_particles"]:
        trains, files = training[sample]
        log(f"s fit ({sample} sample, charge_injection.scale_training_datasets): training run(s) "
            + ", ".join(run_ref(d) for d in trains) + f" ({len(files)} csv file(s))")
        scales[sample] = fit_scale_factors(table, nodes, calibration, files, sample, s)
    default, borrowed = choose_default_sample(scales)
    rule = ("the sample with the fewest channels whose s is borrowed from the (DPM, LINK) / global prior "
            "(tie: the first given)")
    if len(scales) > 1:
        log(f"s = {sample_path(default)}: {rule}; borrowing channels "
            + ", ".join(f"{x} {n}" for x, n in borrowed.items()))
    choice = {"default": default, "samples": list(scales), "n_borrowed": borrowed, "rule": rule}
    diag = {x: _as_json(t.attrs.get("diagnostics", {})) for x, t in scales.items()}
    return scale_columns(scales, default), diag, choice


# CI map + s (s parameter, Lei): the cached fit, or a new fit; official copy with the unit parameters
def obtain_parameters(ctx, s: dict, calibration: pd.DataFrame, calibration_path: Path) -> tuple:
    """A changed s setting reruns only the s fit; a changed CI input refits both."""
    need_scale = needs_scale(s)
    sig = parameter_signatures(ctx, s, calibration_path, need_ci=True, need_scale=need_scale)
    inputs, tag = sig["inputs"], s["variant_tag"]
    ci_sig, scale_sig = sig["ci_signature"], sig["scale_signature"]
    cache_files = parameters_cache_files(ctx, tag)
    cache_csv, cache_nodes, cache_meta = cache_files
    table = nodes = None
    record: dict = {}
    if not s["refit"]:
        ok, why = product_set_valid(cache_meta, {"schema": PARAMETER_SCHEMA, "ci_signature": ci_sig},
                                    members=[cache_csv, cache_nodes])
        if ok:
            table = _read_parameter_csv(cache_csv, f"cached CI parameters {cache_csv}")
            nodes = check_node_table(pd.read_csv(cache_nodes, float_precision="round_trip"),
                                     f"cached CI nodes {cache_nodes}")
            record = read_json(cache_meta)
            log(f"CI map: reusing {cache_meta.parent} (inputs and settings unchanged)")
        elif cache_meta.exists() or cache_csv.exists():
            log(f"CI map: the cached parameters are not reused: {why}")
    ci_fitted = scale_fitted = False
    if table is None:
        table, nodes = calibrate_charge_injection(inputs["scan_files"], inputs["channel_map"], s, calibration)
        ci_fitted = True
    diag, choice = record.get("scale_diagnostics"), record.get("s_choice")
    if need_scale and (ci_fitted or "s_final" not in table.columns
                       or _canonical(record.get("scale_signature")) != _canonical(scale_sig)):
        table = table[CI_TABLE_COLUMNS]
        columns, diag, choice = fit_scale_samples(ctx, s, table, nodes, calibration, inputs["training"])
        table = table.merge(columns, on=KEY_COLUMNS, how="left")
        scale_fitted = True
    if ci_fitted or scale_fitted:
        keep = not ci_fitted  # a cached s fit stays valid with its CI map
        record = {"schema": PARAMETER_SCHEMA, "stage": "ci", "variant_tag": tag, "ci_signature": ci_sig,
                  "scale_signature": scale_sig if need_scale else (record.get("scale_signature") if keep else None),
                  "scale_diagnostics": diag if (need_scale or keep) else None,
                  "s_choice": choice if (need_scale or keep) else None,
                  "software": python_info(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        with ProductSet(cache_meta) as ps:
            table.to_csv(ps.stage(cache_csv), index=False)
            nodes.to_csv(ps.stage(cache_nodes), index=False)
            ps.commit(record)
        log(f"CI parameters cached: {cache_meta.parent}")
    source = ("fitted" if ci_fitted else "cached CI map + new s fit" if scale_fitted else "cached")
    written = []
    if "parameters" in s["units"]:
        written = _write_official_parameters(ctx, tag, cache_files, record, source)
    status = table["ci_status"].astype(str).value_counts().to_dict() if "ci_status" in table.columns else {}
    n_s = int(np.isfinite(pd.to_numeric(table.get("s_final", pd.Series(dtype=float)), errors="coerce")).sum())
    log(f"CI parameters ({source}): {status}, {len(nodes)} nodes, {n_s} channels with s"
        + (f" (s = {sample_path(choice['default'])})" if choice and len(choice.get("samples", [])) > 1 else ""))
    pinfo = {"source": source, "identity": sig["identity"], "files": [str(p) for p in written],
             "ci_signature_hash": sig["identity"]["ci_signature_hash"],
             "scale_signature_hash": sig["identity"]["scale_signature_hash"],
             "scale_signature": scale_sig, "s_choice": choice}
    return table, nodes, pinfo, written


# one-line fit summary for the log
def _fit_brief(fit: dict) -> str:
    if fit.get("kind") == "range_double_gauss":
        text = f"mu_low={fit.get('mu_low', _NAN):.4g} mu_high={fit.get('mu_high', _NAN):.4g}"
    else:
        text = f"mu={fit.get('mu', _NAN):.4g} sigma/mu={fit.get('sigma_over_mu', _NAN):.4g}"
    if fit.get("kind") in er.RANGE_FIT_KINDS:
        text += f" (--fit {fit.get('fit_ranges_text', '')})"
    return text if fit.get("status") == "ok" else f"{text} [{str(fit.get('status'))[:48]}]"


# -linfit choice as text: default per scale, none, or 'electron[0,1] pion'
def _linfit_text(fits) -> str:
    if fits is None:
        return "default per scale (adc_only none, PureCI electron below 1 GeV, s electron)"
    if not fits:
        return "none"
    return " ".join(p + ("" if r is None else f"[{r[0]:g},{r[1]:g}]") for p, r in fits.items())


# fit choices of the run for the log
def _fit_settings_text(rs: dict) -> str:
    ranges = rs.get("fit_ranges")
    if ranges:
        model = "double Gaussian" if len(ranges) == 2 else "single Gaussian"
        bins = f"--bin {_bins_text(rs['fit_bins'])}" if rs.get("fit_bins") else "default page bins"
        spectra = f"spectrum fits --fit {format_ranges(ranges)} MIP ({model}, {bins})"
    else:
        spectra = "automatic single-Gaussian spectrum fits"
    res = rs.get("resolution_fit_range")
    return (f"{spectra}, linearity fits {_linfit_text(rs.get('linearity_fits'))}"
            + ("" if res is None else f", resolution fit {_range_text(res)} GeV"))


# title tag of a -test run
def _test_tag(ctx) -> str:
    return f" [TEST: first {ctx.n_test_events} events]" if ctx.test else ""


# a setting as command-line text
def _flag_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_flag_value(v) for v in value)
    if isinstance(value, (float, np.floating)):
        text = repr(float(value))
        return text[:-2] if text.endswith(".0") else text
    text = str(value).removeprefix("\\\\?\\")
    return f'"{text}"' if (not text or any(c.isspace() for c in text)) else text


# runner options a rebuild command must repeat (-config, -test, -test-events, -clean-input)
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


# ci command that rewrites the event arrays a refused -from-saved run expects
def _rebuild_command(ctx, s: dict, datasets) -> str:
    flags = "".join(f" {flag}" + ("" if value is True else f" {_flag_value(value)}")
                    for flag, value in s.get("_rebuild_flags", []))
    flags += "".join(f" {text}" for text in s.get("_rebuild_range_flags", []))
    return (f".\\runner.ps1 ci {' '.join(run_flags(d) for d in datasets)}{_mode_flags(ctx)}{flags}"
            "   (then repeat this -from-saved command)")


# parameter identity of the event arrays: the CI map with a CI scale, the s fit with an s scale
def reconstruction_parameter_identity(s: dict, identity: dict) -> dict:
    compute = s["compute_paths"]
    if all(p == "adc_only" for p in compute):
        return {"kind": "none"}
    out = {"kind": "chain", "ci_signature_hash": identity.get("ci_signature_hash")}
    if any(is_scale_path(p) for p in compute):
        out["scale_signature_hash"] = identity.get("scale_signature_hash")
    return out


# what the event arrays depend on: algorithm, settings, calibration, CI / s identity
def reconstruction_identity(s: dict, calibration_path: Path, parameter_identity: dict) -> dict:
    return _as_json({"algorithm": EVENT_ALGORITHM,
                     "reconstruction_settings": {k: s[k] for k in RECONSTRUCTION_SETTING_KEYS},
                     "calibration": _content_identity(calibration_path),
                     "parameters": reconstruction_parameter_identity(s, parameter_identity)})


# identity of the cached event arrays of one input csv
def event_cache_token(ctx, ds, csv, beam_csv: Path, ident: dict) -> dict:
    return {"schema": EVENT_CACHE_SCHEMA, "mode": ctx.mode,
            "n_test_events": int(ctx.n_test_events) if ctx.test else None,
            "dataset": {"key": ds.key, "particle": ds.particle, "energy_gev": float(ds.energy_gev)},
            "event_source": ctx.event_source(ds, csv), "beam_csv": _file_signature(beam_csv), **ident}


EVENT_ARRAY_KEYS = ("pf_event", "n_bars", "n_unrecoverable_bars", "bad_strip_fraction",
                    "saturated_unconverted_fraction")


# store the event arrays of one input csv next to its bar records (staged in ps) and commit the set
def save_event_cache(ps: ProductSet, ctx, s: dict, ds, csv, beam_csv: Path, events: dict, n_bars: int,
                     ident: dict, run_id: str) -> Path:
    npz, npy, meta = event_cache_files(ctx, ds, csv, s["variant_tag"])
    arrays = {k: np.asarray(events[k]) for k in EVENT_ARRAY_KEYS}
    arrays["n_events_in_file"] = np.asarray(int(events["n_events_in_file"]), dtype=np.int64)
    for p, values in events["energy"].items():
        arrays[f"energy_{p}"] = np.asarray(values, dtype=np.float64)
    with open(ps.stage(npz), "wb") as fh:
        np.savez(fh, **arrays)
    ps.commit({**event_cache_token(ctx, ds, csv, beam_csv, ident), "stage": "ci",
               "compute_paths": list(events["energy"]), "run_id": run_id,
               "n_events_reconstructed": int(len(events["pf_event"])), "n_bars": int(n_bars),
               "event_arrays": list(arrays), "bar_fields": list(bar_record_dtype(events["energy"]).names),
               "content": "per-event arrays (npz) and per-bar records (npy) of one input csv, every computed path"})
    return meta


# tag of the event arrays -from-saved reads (the untagged ones when -output-tag has none)
def saved_events_tag(ctx, s: dict, ds, csv) -> str:
    tag = s["variant_tag"]
    if s.get("output_tag"):
        npz, _, meta = event_cache_files(ctx, ds, csv, tag)
        if not npz.exists() and not meta.exists():
            return variant_tag({**s, "output_tag": None})
    return tag


# read the stored event arrays of one input csv (beam_csv: the csv the arrays were made from), or stop
def load_event_cache(ctx, s: dict, ds, csv, beam_csv, ident: dict, datasets) -> tuple[dict, dict]:
    tag = saved_events_tag(ctx, s, ds, csv)
    npz, npy, meta = event_cache_files(ctx, ds, csv, tag)
    if tag != s["variant_tag"]:
        log(f"ci -from-saved: no stored event arrays of {ctx.event_source(ds, csv)} under --output-tag "
            f"{s['output_tag']}: reading {meta} (same variant without the tag; the tag names the files this run "
            f"writes)")
    rebuild = _rebuild_command(ctx, s, datasets)
    beam_csv = require(beam_csv, f".\\runner.ps1 make-csv {run_flags(ds)}")
    record = require_product_set(meta, event_cache_token(ctx, ds, csv, beam_csv, ident), rebuild,
                                 what=f"ci event arrays of {ctx.event_source(ds, csv)}", members=[npz, npy])
    missing = [p for p in s["compute_paths"] if p not in record.get("compute_paths", [])]
    if missing:
        raise SettingsError(f"ci -from-saved: the stored event arrays of {ctx.event_source(ds, csv)} ({meta}) hold "
                            f"the paths {record.get('compute_paths')}, not {missing}.\n  Rebuild them with: {rebuild}")
    events = load_event_arrays(npz)
    events["energy"] = {p: events["energy"][p] for p in s["compute_paths"]}
    return events, {**record, "beam_csv_path": str(beam_csv), "event_cache_json": str(meta), "event_cache_tag": tag}


# command that makes the calibration table this run reads (-test: the test table, or the full-set one it falls back to)
def _calib_command(ctx) -> str:
    test_table = ctx.test and Path(ctx.calibration_file()) == Path(ctx.calibration_product())
    return ".\\runner.ps1 calib" + (" -test" if test_table else "")


# inputs and outputs of the run (for -dry-run, check and the run log)
def io_plan(ctx, args, datasets) -> list[dict]:
    s = resolve_settings(ctx, args)
    tag, units = s["variant_tag"], s["units"]
    plan = [{"role": "in", "path": ctx.calibration_file(), "produced_by": _calib_command(ctx)}]
    events = needs_events(s) and bool(datasets)
    if needs_ci(s) and not s["from_saved"]:
        inputs = _parameter_inputs(ctx, s, needs_scale(s), training_csv=False)
        for f in inputs["scan_files"]:
            plan.append({"role": "in", "kind": "external", "path": f,
                         "produced_by": "the charge-injection scan (external)"})
        plan.append({"role": "in", "kind": "external", "path": inputs["channel_map"],
                     "produced_by": "configure/config.yml (paths.channel_map)"})
        listed = {d.key for d in datasets} if events else set()
        for trains, _ in inputs["training"].values():
            for train in trains:
                if train.key not in listed:  # listed once
                    listed.add(train.key)
                    plan += ctx.input_plan_items(train, f".\\runner.ps1 make-csv {run_flags(train)}")
        plan += [{"role": "out", "path": f} for f in parameters_cache_files(ctx, tag)]
        if "parameters" in units:
            plan += [{"role": "out", "path": parameters_csv(ctx, tag)}, {"role": "out", "path": nodes_csv(ctx, tag)}]
            if json_records_on(parameters_record(ctx, tag)):
                plan.append({"role": "out", "path": parameters_record(ctx, tag)})
    if not events:
        return plan
    for ds in datasets:
        plan += ctx.input_plan_items(ds, f".\\runner.ps1 make-csv {run_flags(ds)}")
    ident = None
    if s["from_saved"]:
        try:
            calib = ctx.calibration_file()
            sig = parameter_signatures(ctx, s, calib, need_ci=any(p != "adc_only" for p in s["compute_paths"]),
                                       need_scale=any(is_scale_path(p) for p in s["compute_paths"]))
            ident = reconstruction_identity(s, calib, sig["identity"])
        except (OSError, SystemExit):  # an input is missing: the items are listed without their identity
            pass
    for ds in datasets:
        for csv in ctx.run_csvs(ds):
            if s["from_saved"]:  # the copy the run reads (with --output-tag possibly the untagged one)
                npz, npy, meta = event_cache_files(ctx, ds, csv, saved_events_tag(ctx, s, ds, csv))
                item = {"role": "cache", "path": meta, "members": [npz, npy]}
                if ident is not None:
                    try:
                        item["expected"] = event_cache_token(ctx, ds, csv, ctx.input_csv(ds, csv), ident)
                    except (OSError, SystemExit):
                        pass
                plan.append(item)
            else:
                plan += [{"role": "out", "path": f} for f in event_cache_files(ctx, ds, csv, tag)]
            if "events" in units:
                path = event_energies_csv(ctx, ds, csv, tag)
                plan.append({"role": "out", "path": path})
                if json_records_on(provenance_path(path)):
                    plan.append({"role": "out", "path": provenance_path(path)})
    plan += result_items(ctx, s, datasets, units)
    plan.append({"role": "out", "path": results_record(ctx, tag)})
    return plan


# chunked csv writer (pyarrow when available)
class FastCsvWriter:

    # open the file
    def __init__(self, path: Path):
        self.path = ensure_writable(Path(path))  # refuses protected folders, creates the parent folder
        self.fh = open(self.path, "wb")
        reg = active_registry()
        if reg is not None:
            reg.register_closer(self.fh)
        self.header = True
        self.rows = 0

    # append a chunk
    def write(self, df: pd.DataFrame) -> None:
        if df is None or (not len(df) and not self.header):
            return
        try:
            import pyarrow as pa
            import pyarrow.csv as pacsv
        except ImportError:  # pragma: no cover - pyarrow is in env/requirements.txt
            self.fh.write(df.to_csv(index=False, header=self.header, lineterminator="\n").encode("utf-8"))
        else:
            if self.header:  # plain header line (arrow would quote the names)
                self.fh.write((",".join(map(str, df.columns)) + "\n").encode("utf-8"))
            table = pa.Table.from_pandas(df, preserve_index=False)
            try:
                body = pa.BufferOutputStream()
                pacsv.write_csv(table, body, write_options=pacsv.WriteOptions(include_header=False,
                                                                              quoting_style="none"))
            except pa.ArrowInvalid:  # a text value with a comma / quote: quote the text fields
                body = pa.BufferOutputStream()
                pacsv.write_csv(table, body, write_options=pacsv.WriteOptions(include_header=False,
                                                                              quoting_style="needed"))
            self.fh.write(body.getvalue().to_pybytes())
        self.header = False
        self.rows += len(df)

    # finish the file (a header line when nothing was written)
    def close(self, columns: list[str] | None = None) -> Path:
        try:
            if self.header:
                self.fh.write((",".join(columns or []) + "\n").encode("utf-8"))
        finally:
            self.fh.close()
        return self.path

    # close and delete the file
    def abort(self) -> None:
        self.fh.close()
        if self.path.exists():
            self.path.unlink()


# columns of an event csv
def event_columns(output_paths) -> list[str]:
    return ["event_source", "pf_event", *[f"energy_{p}_mip" for p in output_paths], "n_bars", "bad_strip_fraction",
            "selected"]


# event csv rows of one input csv
def _event_frame(source: str, events: dict, selected: np.ndarray, output_paths) -> pd.DataFrame:
    cols = {"event_source": source, "pf_event": events["pf_event"]}
    for p in output_paths:
        cols[f"energy_{p}_mip"] = events["energy"][p]
    cols.update(n_bars=events["n_bars"], bad_strip_fraction=events["bad_strip_fraction"], selected=selected)
    return pd.DataFrame(cols)[event_columns(output_paths)]


# settings that decide the content of an event csv (provenance records)
PROVENANCE_SETTING_KEYS = ("energy_paths", "output_paths", "selection_path", "selection_path_used", "compute_paths",
                           "quality_filter", "quality_threshold", "quality_tolerance", "bad_bar_rule", "adc_max_rail",
                           "adc_max_rail_guard", "hit_threshold_column", "known_bad_bars", "bar_key_size",
                           "chunk_rows", "scale_particles", "output_tag", "clean_input")


# events_id: identity of one event csv (input, reconstruction, selection)
def events_identity(s: dict, source: str, beam_sig: dict | None, ident: dict) -> str:
    return settings_hash(_as_json({"event_source": source, "beam_csv": beam_sig, "reconstruction": ident,
                                   "output_paths": s["output_paths"],
                                   "selection": {k: s[k] for k in SELECTION_SETTING_KEYS}}))


# provenance record of an event csv
def _provenance_record(ctx, s: dict, ds, csv, calib_path: Path, pinfo: dict, beam_sig: dict | None,
                       run_id: str, events_id: str) -> dict:
    settings = {k: s.get(k) for k in PROVENANCE_SETTING_KEYS}
    params = {k: pinfo.get(k) for k in ("source", "files", "identity", "ci_signature_hash", "scale_signature_hash",
                                        "scale_signature", "s_choice")}
    record = {"stage": "ci", "module": __name__, "mode": ctx.mode, "test_mode": bool(ctx.test),
              "n_test_events": ctx.n_test_events if ctx.test else None, "run_id": run_id, "events_id": events_id,
              "written": time.strftime("%Y-%m-%d %H:%M:%S"), "variant_tag": s.get("variant_tag", ""),
              "from_saved": bool(s.get("from_saved")), "clean_input": bool(s.get("clean_input")),
              "dataset": ds.key, "event_source": ctx.event_source(ds, csv), "settings": settings,
              "ci_parameters": params, "calibration": str(calib_path),
              "inputs": {"beam_csv": beam_sig, "calibration": _file_signature(calib_path)}}
    record["settings_hash"] = settings_hash(_as_json({"settings": settings, "dataset": ds.key,
                                                      "parameters": params["identity"],
                                                      "calibration": record["inputs"]["calibration"],
                                                      "beam_csv": beam_sig, "test_mode": record["test_mode"],
                                                      "n_test_events": record["n_test_events"]}))
    return record


# write the event csv of one input csv (unit events) with its provenance
def write_event_csv(ctx, s: dict, ds, csv, events: dict, selected: np.ndarray, record: dict) -> Path:
    path = event_energies_csv(ctx, ds, csv, s["variant_tag"])
    columns = event_columns(s["output_paths"])
    frame = _event_frame(ctx.event_source(ds, csv), events, selected, s["output_paths"])
    with ProductSet(provenance_path(path)) as ps:
        writer = FastCsvWriter(ps.stage(path))
        try:
            writer.write(frame)
        finally:
            writer.close(columns=columns)
        ps.commit({**record, "rows": int(len(frame)), "columns": columns,
                   "content": "one row per reconstructed event of one input csv (selected = quality cut)"})
    return path


# empty per-run summary of the fits (selected energies per path, counts)
def _new_summary(ds, paths) -> dict:
    return {"ds": ds, "n_csv": 0, "n_events_in_file": 0, "n_reconstructed": 0, "n_selected": 0, "n_bars_all": 0,
            "n_bars_selected": 0, "values": {p: [] for p in paths}, "scan": ([], [])}


# add one input csv to its run summary (+ the quality-scan input of an electron run)
def _add_to_summary(summ: dict, events: dict, selected: np.ndarray, paths, s: dict) -> None:
    summ["n_csv"] += 1
    summ["n_events_in_file"] += int(events["n_events_in_file"])
    summ["n_reconstructed"] += int(len(events["pf_event"]))
    summ["n_selected"] += int(selected.sum())
    summ["n_bars_all"] += int(events["n_bars"].sum())
    summ["n_bars_selected"] += int(events["n_bars"][selected].sum())
    for p in paths:
        summ["values"][p].append(events["energy"][p][selected])
    if "quality_scan" in s["units"] and summ["ds"].particle == "electron":
        summ["scan"][0].append(events["energy"][s["selection_path_used"]])
        summ["scan"][1].append(events["bad_strip_fraction"])


# finished run summaries: the values of the csv files of a run joined (statistics of the fits, nothing written)
def _finish_summaries(summaries: dict) -> list[dict]:
    out = []
    for summ in summaries.values():
        if not summ["n_csv"]:
            continue
        summ["values"] = {p: np.concatenate(v) if v else np.zeros(0) for p, v in summ["values"].items()}
        energy, fraction = summ.pop("scan")
        summ["scan"] = (np.concatenate(energy), np.concatenate(fraction)) if energy else None
        out.append(summ)
    return out


# temporary raw per-bar records of one input csv (cache/), turned into the npy member by the worker
def bar_raw_temp(ctx, vtag: str, ds, csv, token: str) -> Path:
    stem = Path(ctx.event_source(ds, csv)).with_suffix("").as_posix().replace("/", ".")
    return cache_stage_dir(ctx) / f"bar_records{tag_suffix(vtag)}.{stem}.tmp-{os.getpid()}-{token}.bin"


# reconstruct every input csv: event arrays + bar records (cache), event csv (unit events), run summaries
def _reconstruct(ctx, s: dict, datasets, calib_path: Path, calibration: pd.DataFrame, table, nodes,
                 pinfo: dict) -> tuple[list, list, str]:
    vtag, compute = s["variant_tag"], s["compute_paths"]
    params = build_reconstruction_parameters(calibration, table, nodes, s)
    ident = reconstruction_identity(s, calib_path, pinfo["identity"])
    token = uuid.uuid4().hex[:12]
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{token}"
    summaries = {ds.key: _new_summary(ds, s["output_paths"]) for ds in datasets}
    written = []
    sweep_orphan_temps(cache_stage_dir(ctx))  # bar records of a run that was killed hard
    with temp_registry() as reg:
        jobs, members = [], []
        for ds in datasets:
            for csv, path in ctx.run_inputs(ds):
                path = require(path, f".\\runner.ps1 make-csv {run_flags(ds)}")
                ps = ProductSet(event_cache_files(ctx, ds, csv, vtag)[2], reg)
                raw = str(reg.register(ensure_writable(bar_raw_temp(ctx, vtag, ds, csv, token[:8]))))
                jobs.append({"key": ctx.event_source(ds, csv), "csv": str(path), "params": params,
                             "compute_paths": compute, "selection_path": s["selection_path_used"],
                             "chunk_rows": int(s["chunk_rows"]), "bar_raw": raw,
                             "bar_npy": str(ps.stage(event_cache_files(ctx, ds, csv, vtag)[1]))})
                members.append((ds, csv, Path(path), ps, raw))
        log(f"ci: reconstructing {len(jobs)} csv file(s) of {len(datasets)} run(s) on "
            f"{min(max(ctx.workers, 1), max(len(jobs), 1))} process(es)")
        with closing(iter_reconstruction(jobs, ctx.workers)) as results:
            for (ds, csv, path, ps, raw), res in zip(members, results):
                reg.release(raw)
                ev = res["events"]
                selected = select_events(ev, s)
                beam_sig = _file_signature(path)
                written.append(save_event_cache(ps, ctx, s, ds, csv, path, ev, res["bar_rows"], ident, run_id))
                if "events" in s["units"]:
                    source = ctx.event_source(ds, csv)
                    record = _provenance_record(ctx, s, ds, csv, calib_path, pinfo, beam_sig, run_id,
                                                events_identity(s, source, beam_sig, ident))
                    written.append(write_event_csv(ctx, s, ds, csv, ev, selected, {
                        **record, "event_cache": str(event_cache_files(ctx, ds, csv, vtag)[2])}))
                _add_to_summary(summaries[ds.key], ev, selected, s["output_paths"], s)
                del ev, res, selected
    return _finish_summaries(summaries), written, run_id


# -from-saved: the stored event arrays replace the reconstruction; event csv (unit events) and run summaries
def _from_saved(ctx, s: dict, datasets, calib_path: Path) -> tuple[list, list, str]:
    compute = s["compute_paths"]
    sig = parameter_signatures(ctx, s, calib_path, need_ci=any(p != "adc_only" for p in compute),
                               need_scale=any(is_scale_path(p) for p in compute))
    pinfo = {"source": "from-saved (stored event arrays)", "identity": sig["identity"],
             "ci_signature_hash": sig["identity"].get("ci_signature_hash"),
             "scale_signature_hash": sig["identity"].get("scale_signature_hash"),
             "scale_signature": sig.get("scale_signature")}
    ident = reconstruction_identity(s, calib_path, sig["identity"])
    loaded = [(ds, csv, *load_event_cache(ctx, s, ds, csv, path, ident, datasets))
              for ds in datasets for csv, path in ctx.run_inputs(ds)]
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"
    summaries = {ds.key: _new_summary(ds, s["output_paths"]) for ds in datasets}
    written = []
    for ds, csv, ev, rec in loaded:
        selected = select_events(ev, s)
        log(f"  {ctx.event_source(ds, csv)}: {len(ev['pf_event']):,} reconstructed events from "
            f"{rec['event_cache_json']} (reconstruction run {rec.get('run_id')})")
        if "events" in s["units"]:
            source = ctx.event_source(ds, csv)
            beam_sig = _file_signature(rec["beam_csv_path"])
            record = _provenance_record(ctx, s, ds, csv, calib_path, pinfo, beam_sig, run_id,
                                        events_identity(s, source, beam_sig, ident))
            written.append(write_event_csv(ctx, s, ds, csv, ev, selected, {
                **record, "event_cache": rec["event_cache_json"], "event_cache_tag": rec["event_cache_tag"],
                "reconstruction_run_id": rec.get("run_id")}))
        _add_to_summary(summaries[ds.key], ev, selected, s["output_paths"], s)
    return _finish_summaries(summaries), written, run_id


# summaries in figure order: electrons first, then the other particles, each by beam energy
def _ordered(summaries: list) -> list:
    return sorted(summaries, key=lambda x: (x["ds"].particle != er.ELECTRON, x["ds"].particle, x["ds"].energy_gev))


# spectrum fits, linearity / resolution / e/h, check tables and figures -> the final results of the units
def _respond(ctx, s: dict, summaries: list, plan, tag: str) -> dict:
    paths, units = s["output_paths"], s["units"]
    planned = {slot.key for slot in plan.slots}
    summaries = _ordered(summaries)
    dependence = {}
    if planned & {"response", "linearity", "resolution", "eh"} and summaries:
        for summ in summaries:
            summ["fits"] = {p: spectrum_fit(summ["values"][p], summ["ds"].particle, s) for p in paths}
        dependence = {p: energy_dependence(summaries, p, s) for p in paths}
    if "response" in planned:
        atomic_write_csv(pd.DataFrame(energy_response_rows(summaries, dependence, paths, s)), plan.path("response"))
    if "quality_scan" in planned:
        tables = [(summ["ds"], *summ["scan"]) for summ in summaries if summ["scan"] is not None]
        atomic_write_csv(scan_quality_threshold(tables, s), plan.path("quality_scan"))
        log(f"quality scan (check): {len(tables)} electron run(s), {len(s['scan_threshold_grid'])} thresholds; the "
            f"cut uses {s['quality_threshold']:g}" + (" (test statistics)" if ctx.test else ""))
    for unit in ("linearity", "resolution"):
        if unit in planned:
            plot_energy_dependence(plan.path(unit), unit, dependence, paths, s, tag)
    if "eh" in planned:
        plot_e_over_h(plan.path("eh"), dependence, paths, s, tag)
    for unit in units:
        why = unit_skip_reason(unit, s, [summ["ds"] for summ in summaries]) if unit in RESULT_FILES else ""
        if why:
            log(f"ci: {unit} not made: {why}")
    return dependence


# log the run summary: selected events and fits per run, the linearity lines per path
def _log_summary(summaries: list, dependence: dict, paths, source: str, tag: str) -> None:
    log(f"ci summary ({source} CI parameters){tag}:")
    for summ in _ordered(summaries):
        fits = "; ".join(f"{p}: {_fit_brief(summ['fits'][p]['fit'])}" for p in paths) if "fits" in summ else ""
        log(f"  {summ['ds'].key}: {summ['n_selected']:,}/{summ['n_reconstructed']:,} events selected "
            f"({summ['n_selected'] / max(summ['n_reconstructed'], 1):.4f}) in {summ['n_csv']} csv, "
            f"{summ['n_bars_selected']:,} bars" + (f" | {fits}" if fits else ""))
    for p, dep in dependence.items():
        lines = [f"{part} slope {line['slope']:.4g} +/- {line['slope_error']:.2g} MIP/GeV ({line['status'][:40]})"
                 for part, line in dep["linearity"]["fits"].items()] or ["no fit (points only)"]
        res = dep["resolution"]
        log(f"  {p}: linearity {'; '.join(lines)}, sigma/mu c = {res['c']:.4g} ({res['status'][:40]})")


# run the stage
def run(ctx, args, datasets) -> None:
    s = resolve_settings(ctx, args)
    units = s["units"]
    t0 = time.time()
    vtag = s["variant_tag"]  # file-name variant (tag_suffix(vtag) on every product)
    tag = _test_tag(ctx) + (f" [variant {vtag}]" if vtag else "")  # figure-title / log tag
    events = needs_events(s) and bool(datasets)
    if needs_events(s) and not datasets:
        log("ci: no dataset selected: no event unit is made")
    plan = result_plan(ctx, s, datasets if events else [], units)  # final results: --output / test / scratch copy
    calib_path = require(ctx.calibration_file(), _calib_command(ctx))
    log(f"ci: units {', '.join(units)}; paths {s['output_paths']}, selection by {s['selection_path_used']} bad-strip "
        f"flags ({'<= ' + format(s['quality_threshold'], 'g') if s['quality_filter'] else 'no quality filter'}, "
        f"bad bars: {s['bad_bar_rule']}), {_fit_settings_text(s['energy_response'])}, calibration {calib_path.name}"
        + (f", variant '{vtag}'" if vtag else "") + (" [from saved event arrays]" if s["from_saved"] else ""))
    note = linearity_source_note(s)
    if note:  # --fit replaces the yml choice linearity_source peak_scan (Lei 2026-10-02): said once, at the start
        log(note)
    summaries, written, run_id = [], [], None
    pinfo = {"source": "none (ADC only)", "identity": {"kind": "none"}}
    if s["from_saved"]:
        pinfo = {"source": "from-saved"}
        if events:
            summaries, written, run_id = _from_saved(ctx, s, datasets, calib_path)
    else:
        calibration = None
        if needs_ci(s) or events:
            calibration = checked_channel_table(load_calibration(calib_path), f"calibration table {calib_path}")
        table = nodes = None
        if needs_ci(s):
            table, nodes, pinfo, official = obtain_parameters(ctx, s, calibration, calib_path)
            written += official
        if events:
            summaries, csv_written, run_id = _reconstruct(ctx, s, datasets, calib_path, calibration, table, nodes,
                                                          pinfo)
            written += csv_written
    dependence = _respond(ctx, s, summaries, plan, tag) if events else {}
    if events:
        event_csvs = [p for p in written if str(p).endswith(".csv") and "_event_energies" in Path(p).name]
        written.append(_write_results_record(ctx, s, datasets, plan, run_id, event_csvs))
        _log_summary(summaries, dependence, s["output_paths"], pinfo["source"], tag)
    for f in written:  # parameters, event csv, cache records, the record of the final results
        log(f"  wrote {f}")
    _log_final_results(plan)
    log(f"ci done in {time.time() - t0:.1f}s")
