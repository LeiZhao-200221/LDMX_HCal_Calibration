"""Schema and IO of the per-channel calibration tables written by calib: the full table (pedestal + MIP MPV)
and the part tables (pedestal only, MPV only); key columns layer, strip, end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo
from .io_utils import ChannelKeyError, log, validate_channel_keys

KEY_COLUMNS = ["layer", "strip", "end"]

# column -> meaning (the README table is generated from this dict).  Order = column order of the csv.
CALIBRATION_COLUMNS = {
    # ---- pedestal and MPV values ----
    "pedestal": "Gaussian-MLE mean of the 8-sample adc_sum of the pedestal events inside the open window "
                "|adc_sum - pedestal_anchor| < 200 [ADC] (= pedestal_anchor + pedestal_mu)",
    "pedestal_per_time_sample": "per-sample pedestal: the mode of the single-sample ADC of the pedestal events "
                                "(pedestal_anchor_method sample_mode) or pedestal/8 (median_sum) [ADC]",
    "pedestal_sigma": "Gaussian-MLE sigma (ddof 0) of the same window, tails included [ADC]",
    "pedestal_mu": "Gaussian-MLE mean of adc_sum - pedestal_anchor in the window; "
                   "pedestal = pedestal_anchor + pedestal_mu [ADC]",
    "pedestal_range_with_5sigma": "pedestal + 5*pedestal_sigma: the dynamic hit threshold (clean -pedestalsigma) [ADC]",
    "pedestal_range_with_20percent": "1.2*pedestal: the default hit threshold on adc_sum "
                                     "(clean -pedestalscale, adcpeak, ci, plots) [ADC]",
    "n_pedestal_events": "pedestal events inside the window (the events of the Gaussian MLE)",
    "mpv": "MIP most probable value of adc_sum - pedestal: numerical peak of the final Landau(x)Gauss fit; "
           "NaN unless mpv_status is OK [ADC]",
    "mpv_error": "statistical uncertainty of mpv, sqrt(J C J^T) with a positive-definite covariance; NaN when "
                 "mpv is NaN or mpv_covariance_status is not OK [ADC]",
    "landau_width": "Landau width (scale) eta of the final fit; NaN unless mpv_status is OK [ADC]",
    "gauss_sigma": "Gaussian smearing sigma of the final fit; NaN unless mpv_status is OK [ADC]",
    "mpv_fit_low": "lower edge of the final fit window [ADC]",
    "mpv_fit_high": "upper edge of the final fit window [ADC]",
    "n_mpv_entries": "entries of x = adc_sum - pedestal with x > mpv_threshold inside the histogram range "
                     "(all bins, not only the fit window)",
    "mpv_chi2_ndf": "Poisson deviance / ndf of the final fit; NaN unless mpv_status is OK",
    "mpv_status": "OK, or why mpv is missing: LOW_STATS, NO_PEDESTAL, ROUGH_FIT_FAILED, FIT_FAILED, "
                  "PEAK_OUTSIDE_WINDOW",
    # ---- method and fit-quality columns ----
    "pedestal_anchor": "window centre: 8 x per-sample ADC mode (sample_mode) or 8 x round(median(adc_sum)/8) "
                       "(median_sum) [ADC]",
    "pedestal_anchor_method": "sample_mode (csv made with make-csv -waveform) or median_sum (standard csv: an "
                              "approximation); chosen automatically",
    "pedestal_selection": "raw_samples (default: raw TOT/TOA word sums of both ends == 0, the selection on the ROOT "
                          "digis; the raw_*_sum columns make-csv writes by default) or csv_tot_toa_zero (only on "
                          "request: calibrated csv tot/toa of both ends == 0, an approximation and a superset of "
                          "raw_samples)",
    "n_selected_events": "events of the bar passing the pedestal selection (before the window)",
    "mpv_method": "two_stage_langaus: a rough Landau (x) Gauss fit sets the window of the final fit",
    "mpv_pedestal": "pedestal subtracted by the MPV fit (adc_sum - mpv_pedestal) [ADC]",
    "mpv_pedestal_sigma": "pedestal sigma that set the MPV cut (mpv_threshold = mpv_nsigma * it) [ADC]",
    "mpv_threshold": "x cut of the MPV fits: x = adc_sum - mpv_pedestal > mpv_threshold [ADC]",
    "mpv_rough_floor": "lower edge of the rough fit (valley between noise shoulder and MIP peak, or the cut) [ADC]",
    "mpv_rough_status": "rough fit: CONVERGED, NOT_CONVERGED (finite parameters, optimizer did not converge; "
                        "still seeds the final fit unless mpv_require_rough_convergence), FAILED (no finite "
                        "parameters, or an exception in the rough stage -> ROUGH_FIT_FAILED), NOT_RUN",
    "mpv_rough_peak": "peak of the rough fit curve (centre of the final window) [ADC]",
    "mpv_rough_width": "FWHM/2.3548 of the rough fit curve (final window = peak +- K * width) [ADC]",
    "mpv_fit_status": "final fit: CONVERGED, NOT_CONVERGED (-> mpv_status FIT_FAILED), FAILED (no finite "
                      "parameters, or an exception in the final stage, also after convergence -> FIT_FAILED), "
                      "NOT_RUN",
    "mpv_fit_message": "optimizer message of the final fit (or of the rough fit when it stopped there); the text of "
                       "an exception is appended after '; '",
    "mpv_covariance_status": "OK, NOT_POSITIVE_DEFINITE, SINGULAR, NOT_FINITE, NO_FREE_PARAMETERS (every parameter "
                             "on a bound) or NOT_COMPUTED (free-parameter Hessian of the deviance checked by "
                             "Cholesky + scaled eigenvalues)",
    "mpv_active_bounds": "final-fit parameters sitting on a bound, e.g. landau_width@lower (also for a rejected fit; "
                         "';'-separated; their covariance rows are zero, errors conditional on them)",
    "landau_mpv_param": "Landau MPV parameter of the final fit (langaus.C parameterisation, not the curve peak); "
                        "NaN unless mpv_status is OK [ADC]",
    "langaus_area": "area parameter (entries of the whole distribution) of the final fit; NaN unless mpv_status is OK",
    "n_mpv_fit_entries": "entries in the final fit window",
    "mpv_deviance": "Poisson deviance of the final fit; NaN unless mpv_status is OK",
    "mpv_ndf": "bins in the final fit window - 4; NaN unless mpv_status is OK",
}
STRING_COLUMNS = ("mpv_status", "pedestal_anchor_method", "pedestal_selection", "mpv_method", "mpv_rough_status",
                  "mpv_fit_status", "mpv_fit_message", "mpv_covariance_status", "mpv_active_bounds")
INTEGER_COLUMNS = ("n_pedestal_events", "n_mpv_entries", "n_selected_events", "n_mpv_fit_entries", "mpv_ndf")


# empty table: every channel, every column empty
def empty_calibration() -> pd.DataFrame:
    df = geo.all_channels()
    for col in CALIBRATION_COLUMNS:
        df[col] = "" if col in STRING_COLUMNS else np.nan
    return df


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
# channel-key problems of a table (empty list = valid)
def key_problems(df: pd.DataFrame) -> list[str]:
    try:
        validate_channel_keys(df, keys=KEY_COLUMNS, source="calibration table")
    except ChannelKeyError as exc:
        return [str(exc)]
    return []


# implausible values -> their channels (reported, never changed)
def value_problems(df: pd.DataFrame) -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = {}

    # record the channels of one problem
    def flag(name, mask):
        mask = np.asarray(mask, bool)
        if mask.any():
            out[name] = [tuple(int(v) for v in r) for r in df.loc[mask, KEY_COLUMNS].to_numpy()]

    num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in
           ("pedestal_sigma", "mpv", "mpv_error") if c in df.columns}
    with np.errstate(invalid="ignore"):
        if "pedestal_sigma" in num:
            flag("NONPOSITIVE_PEDESTAL_SIGMA", num["pedestal_sigma"] <= 0)
        if "mpv" in num:
            flag("NONPOSITIVE_MPV", num["mpv"] <= 0)
            if "mpv_status" in df.columns:
                flag("STATUS_OK_WITHOUT_MPV", (df["mpv_status"].astype(str).to_numpy() == "OK")
                     & ~np.isfinite(num["mpv"]))
        if "mpv_error" in num:
            flag("NEGATIVE_MPV_ERROR", num["mpv_error"] < 0)
    return out


# read a calibration table (full or part table): keys checked, sorted; implausible values logged
def load_calibration(path: str | Path, strict: bool = False, float_precision: str | None = None) -> pd.DataFrame:
    """strict=True: implausible values raise ValueError; float_precision='round_trip' reads floats exactly."""
    df = pd.read_csv(path, float_precision=float_precision)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ChannelKeyError(f"{path}: calibration table lacks key columns {missing}")
    df = validate_channel_keys(df, keys=KEY_COLUMNS, source=f"{path} (calibration table)")
    for c in STRING_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    values = value_problems(df)
    df.attrs["value_problems"] = values
    if values:
        text = "; ".join(f"{name}: {len(ch)} channel(s), e.g. L{ch[0][0]}S{ch[0][1]}E{ch[0][2]}"
                         for name, ch in values.items())
        if strict:
            raise ValueError(f"{path}: implausible calibration values: {text}")
        log(f"warning: {Path(path).name}: implausible calibration values (kept unchanged): {text}")
    return df


# --------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------
# dense array arr[layer, strip, end] of one column (absent channels: fill)
def calibration_lookup(df: pd.DataFrame, column: str, fill=np.nan) -> np.ndarray:
    arr = np.full((geo.N_LAYERS + 1, geo.N_STRIPS_WIDE, 2), fill, dtype=float)
    if column not in df.columns:
        return arr
    sub = validate_channel_keys(df[KEY_COLUMNS + [column]], keys=KEY_COLUMNS,
                                source=f"calibration_lookup('{column}')")
    arr[sub["layer"].to_numpy(), sub["strip"].to_numpy(), sub["end"].to_numpy()] = \
        pd.to_numeric(sub[column], errors="coerce").to_numpy(float)
    return arr


# table[layer, strip, end] for beam rows
def lookup_values(table: np.ndarray, layer, strip, end) -> np.ndarray:
    layer = np.asarray(layer, dtype=np.intp)
    strip = np.asarray(strip, dtype=np.intp)
    return table[layer, strip, end]
