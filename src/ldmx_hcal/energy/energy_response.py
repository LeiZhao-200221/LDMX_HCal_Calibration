"""Shared energy-response fits and figures of ci and adcpeak: spectrum peaks, linearity, resolution, e/h.

What it does (no file of its own; the stages write the rows and figures):
- spectrum peak of one run: two-pass Gaussian core fit (default), peak-scan fit, or the -fit[a,b]
  single / -fit[a,b],[c,d] double Gaussian; mu = peak position, sigma = width, sigma/mu = resolution point.
- linearity mu(E): the points of every particle drawn on one figure; a straight line mu = slope * E + intercept
  is fitted per particle on request (-linfit <particle>[a,b], one line each; electrons and pions are never fitted
  together). The default fits of the ci scales: adc_only no fit, PureCI electron points below 1 GeV, s every electron point.
- resolution sigma/mu(E) = sqrt(a^2/E^2 + b^2/E + c^2) (E in MeV), fitted to the electron points; the hadron points
  are drawn only.
- e/h(E) = mu_electron(E) / mu_pion(E) on one scale (s), points only: filled where an electron run has the pion
  energy, hollow where mu_electron(E) comes from the electron line fit.

Rows (the stages put them into their response tables):
- linearity_rows: particle, fit_range_gev, ok, status, weighted, slope, slope_error, intercept, intercept_error,
  chi2, ndf, chi2_over_ndf, error_scale, slope_error_scaled, intercept_error_scaled, n_fitted, fitted_energies.
- eh_rows: run, hadron, energy_gev, e_over_h, e_over_h_error, mu_electron, mu_electron_error, mu_hadron,
  mu_hadron_error, electron_source (electron run | electron line fit), filled, note.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
import warnings
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from ..common.config import RUN_PARTICLES
from ..common.fitting import (MAX_FIT_RANGES, format_ranges, in_range, parse_bins, parse_ranges, range_bin_edges,
                              range_fit_curve, range_gauss_fit, range_options_error, ranges_overlap_error)
from ..common.io_utils import SettingsError, validate_settings

CONFIG_SECTION = "energy_response"
DEFAULTS = {
    "spectrum_bins": 90,                      # two-pass core fit: equal-width bins over the display range
    "min_bin_count": 10,                      # bins with fewer entries are neither drawn nor fitted
    "window_nsigma": 2.0,                     # pass-2 window = mu1 +/- window_nsigma * sigma1
    "min_fit_bins": 8,                        # required visible bins (pass 1) and core bins (pass 2)
    "hist_upper_percentile": 99.8,            # display range upper edge = max(hist_upper_min, factor * percentile)
    "hist_upper_factor": 1.08,
    "hist_upper_min": 3.0,
    "peak_scan_bins": 300,                    # peak-scan ("unbinned") fit binning
    "peak_scan_upper_percentile": 99.7,
    "peak_scan_upper_factor": 1.12,
    "peak_scan_upper_min": 3.0,
    # user fit ranges (None = the default behaviour).  The spectrum ranges -fit / -bin are COMMAND-LINE ONLY
    # (CLI_ONLY_SETTINGS): ci and adcpeak spectra have different x axes (MIP / a.u.).  GeV in both stages:
    "linearity_fits": None,                   # -linfit: {particle: [a, b] or null = every point}; None = the scale's
                                              # default (SCALE_LINEARITY_FITS), {} = no fit
    "resolution_fit_range": None,             # -resfit[a,b]: fit a <= E <= b GeV only; None = every electron point
    "linearity_weighted": True,               # weighted (1/mu_err^2) line fit; False = unweighted polyfit
    "normal_error": False,                    # -normalerror: also errors x sqrt(chi2/ndf) when chi2/ndf > 1
    "error_prescription": "sigma_over_sqrt_n",  # point errors: sigma_over_sqrt_n | sigma_fit_error | gaussian
    "resolution_p0": [50.0, 8.0, 0.05],       # start values a [MeV], b [sqrt(MeV)], c
    "resolution_bounded": True,               # a, b, c >= 0 (trf); False = unbounded Levenberg-Marquardt
    "resolution_maxfev": 100000,              # curve_fit function-evaluation limit
}

ERROR_PRESCRIPTIONS = ("sigma_over_sqrt_n", "sigma_fit_error", "gaussian")
ELECTRON = "electron"                         # the particle of the resolution fit and of e/h's mu_electron
HADRON = "pion"                               # the hadron of e/h
# default linearity fits per ci scale (Lei 2026-10-20): {particle: [a, b] GeV or None = every point}; {} = no fit
SCALE_LINEARITY_FITS = {
    "adc_only": {},
    "pureci": {ELECTRON: [0.0, 0.999]},   # only the points below 1 GeV (Lei 2026-10-20)
    "s": {ELECTRON: None},
}
# the settings of the user fit ranges: JSON-friendly after check_settings
RANGE_SETTING_KEYS = ("fit_ranges", "fit_bins", "linearity_fits", "resolution_fit_range")
# settings that only the command line sets: every settings dict has them (None = the default fits), the yml
# section energy_response refuses them with this hint
_CLI_ONLY_WHY = ("is command-line only: the spectrum fit range is in the x units of each stage's page (ci: MIP, "
                 "adcpeak: a.u.), so one value in the shared yml section energy_response would be wrong for the "
                 "other stage; remove it from the yml and give")
CLI_ONLY_SETTINGS = {
    "fit_ranges": f"{_CLI_ONLY_WHY} -fit[a,b] (or -fit[a,b],[c,d]) on the stage's command line",
    "fit_bins": f"{_CLI_ONLY_WHY} -bin N (or -bin N1,N2) with -fit on the stage's command line",
}

_NAN = float("nan")
PEAK_SCAN_FWHM_PER_SIGMA = 2.3548  # FWHM / sigma of a Gaussian: the peak-scan seed width



# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
# silence fit warnings (poor fits are reported through status)
@contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


# finite values as a flat array
def _finite(values) -> np.ndarray:
    v = np.asarray(values if values is not None else [], dtype=float).ravel()
    return v[np.isfinite(v)]


# num / den (NaN for den 0)
def _ratio(num, den) -> float:
    num, den = float(num), float(den)
    return num / den if den != 0 and math.isfinite(den) else _NAN


# chi2 / ndf (NaN for ndf 0)
def _chi2_ndf(chi2, ndf) -> float:
    return chi2 / ndf if ndf > 0 else _NAN


# Gaussian + constant (charge-injection peak scan, provided by Erik Wallin)
def _gauss_with_constant(x, amplitude, mu, sigma, constant):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + constant


# Gaussian (charge-injection two-pass fit, provided by Erik Wallin)
def _gauss_core(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * np.square((x - mu) / sigma))


# resolution model: sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2), E in MeV
def _resolution_three_term(energy_mev, a, b, c):
    return np.sqrt(np.square(a) / np.square(energy_mev) + np.square(b) / energy_mev + np.square(c))


# --------------------------------------------------------------------------
# spectrum fits
# --------------------------------------------------------------------------
# peak-scan fit (charge injection, provided by Erik Wallin): Gaussian + constant in mu0 +- 2.5 sigma0
def peak_scan_fit(values, n_bins: int = 300, upper_percentile: float = 99.7, upper_factor: float = 1.12,
                  upper_min: float = 3.0) -> dict:
    """No fallback: a failed fit has NaN mu / sigma; seed_mu / seed_sigma only seed two_pass."""
    v = _finite(values)
    out = {"kind": "peak_scan", "ok": False, "status": "no_data", "n_values": int(v.size),
           "mu": _NAN, "sigma": _NAN, "sigma_over_mu": _NAN, "n_detected_peaks": 0, "peak_positions": [],
           "amplitude": _NAN, "constant": _NAN, "fit_low": _NAN, "fit_high": _NAN,
           "hist_upper": _NAN, "bin_width": _NAN, "seed_mu": _NAN, "seed_sigma": _NAN}
    if v.size == 0:
        return out
    emax = max(float(upper_min), float(np.percentile(v, upper_percentile)) * float(upper_factor))
    edges = np.linspace(0.0, emax, int(n_bins) + 1)
    counts, _ = np.histogram(v, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    out.update(hist_upper=emax, bin_width=bin_width, edges=edges, centers=centers, counts=counts)
    if counts.size < 5 or counts.max(initial=0) <= 0:
        out["status"] = "empty_histogram"
        return out

    smooth = gaussian_filter1d(counts.astype(float), sigma=2.0, mode="nearest")
    prominence = max(0.08 * float(smooth.max()), 10.0)
    peaks, properties = find_peaks(smooth, prominence=prominence, distance=5)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(smooth))])
        prominences = np.array([float(smooth[peaks[0]])])
    else:
        prominences = properties["prominences"]
    primary = int(peaks[int(np.argmax(prominences))])
    widths = peak_widths(smooth, [primary], rel_height=0.5)[0]
    mu0 = float(centers[primary])
    sig0 = max(float(widths[0]) * bin_width / PEAK_SCAN_FWHM_PER_SIGMA, bin_width)
    lo = max(0.0, mu0 - 2.5 * sig0)
    hi = mu0 + 2.5 * sig0
    selected = (centers >= lo) & (centers <= hi)
    x, y = centers[selected], counts[selected].astype(float)
    mu, sigma = _NAN, _NAN  # a failed fit gives no value (no smoothed-peak fallback, Lei 2026-10-02)
    status = f"fit_failed: {len(x)} bins in the fit window (< 7)"
    if len(x) >= 7:
        try:
            p0 = [max(float(y.max() - np.median(y)), 1.0), mu0, sig0, float(np.median(y))]
            lower = [0.0, lo, bin_width * 0.5, -np.inf]
            upper = [np.inf, hi, max(hi - lo, 2 * bin_width), np.inf]
            with _quiet():
                popt, _ = curve_fit(_gauss_with_constant, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
            mu, sigma = float(popt[1]), abs(float(popt[2]))
            out.update(amplitude=float(popt[0]), constant=float(popt[3]))
            status = "ok"
        except Exception as exc:  # a failed fit keeps no value
            status = f"fit_failed: {exc}"
    order = np.argsort(prominences)[::-1]
    fitted = math.isfinite(mu) and math.isfinite(sigma)
    out.update(mu=mu, sigma=sigma, sigma_over_mu=sigma / mu if mu > 0 else _NAN,
               seed_mu=mu if fitted else mu0, seed_sigma=sigma if fitted else sig0,  # start values only
               n_detected_peaks=int(len(peaks)),
               peak_positions=[float(centers[int(peaks[i])]) for i in order],
               fit_low=lo, fit_high=hi, status=status, ok=math.isfinite(mu) and math.isfinite(sigma))
    return out


# two-pass Gaussian fit (charge injection, provided by Erik Wallin): pass 1 on bins >= 10, pass 2 in mu1 +- 2 sigma1
def two_pass_gauss_fit(values, bins: int = 90, min_bin_count: int = 10, display_range=None,
                       window_nsigma: float = 2.0, initial_mu=None, initial_sigma=None, min_fit_bins: int = 8,
                       upper_percentile: float = 99.8, upper_factor: float = 1.08, upper_min: float = 3.0) -> dict:
    """Only pass 2 is the result (errors from the unscaled covariance)."""
    v = _finite(values)
    k = float(window_nsigma)
    out = {"kind": "two_pass", "ok": False, "status": "no_data", "n_values": int(v.size), "bins": int(bins),
           "min_bin_count": int(min_bin_count), "window_nsigma": k,
           "window_gaussian_coverage": math.erf(k / math.sqrt(2.0)),
           "hist_lower": _NAN, "hist_upper": _NAN, "bin_width": _NAN, "initial_mu": _NAN, "initial_sigma": _NAN,
           "pass1_bin_count": 0, "core_bin_count": 0, "fit_low": _NAN, "fit_high": _NAN,
           "edges": None, "centers": None, "counts": None, "visible_mask": None}
    for prefix in ("pass1_", ""):
        for name in ("amplitude", "amplitude_error", "mu", "mu_error", "sigma", "sigma_error", "chi2",
                     "chi2_over_ndf"):
            out[prefix + name] = _NAN
        out[prefix + "ndf"] = 0
    out["sigma_over_mu"] = _NAN

    if display_range is None:
        if v.size == 0:
            return out
        lower = 0.0
        upper = max(float(upper_min), float(np.percentile(v, upper_percentile)) * float(upper_factor))
    else:
        lower, upper = float(display_range[0]), float(display_range[1])
    edges = np.linspace(lower, upper, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(v, bins=edges)
    bin_width = float(edges[1] - edges[0])
    visible = counts >= min_bin_count
    out.update(hist_lower=lower, hist_upper=upper, bin_width=bin_width, edges=edges, centers=centers,
               counts=counts, visible_mask=visible, pass1_bin_count=int(np.count_nonzero(visible)))
    if np.count_nonzero(visible) < min_fit_bins:
        out["status"] = f"too_few_visible_bins: {np.count_nonzero(visible)} < {min_fit_bins}"
        return out

    first_x = centers[visible]
    first_y = counts[visible].astype(float)
    first_error = np.sqrt(first_y)
    if initial_mu is None or initial_sigma is None or not (np.isfinite(initial_mu) and np.isfinite(initial_sigma)):
        scan = peak_scan_fit(v)
        initial_mu, initial_sigma = scan["seed_mu"], scan["seed_sigma"]
        if not (np.isfinite(initial_mu) and np.isfinite(initial_sigma)):
            initial_mu = float(first_x[int(np.argmax(first_y))])
            initial_sigma = max(float(np.std(v)), bin_width)
    out.update(initial_mu=float(initial_mu), initial_sigma=float(initial_sigma))

    # pass 1: every visible bin; only locates the peak and defines the core window
    lb1 = np.array([0.0, lower, max(bin_width * 0.5, 1e-6)])
    ub1 = np.array([np.inf, upper, upper])
    # start values clipped into the bounds (curve_fit refuses an infeasible x0)
    p0 = np.clip(np.array([float(np.max(first_y)), initial_mu, initial_sigma]), lb1, ub1)
    try:
        with _quiet():
            first_parameters, first_covariance = curve_fit(
                _gauss_core, first_x, first_y, p0=p0, sigma=first_error, absolute_sigma=True,
                bounds=(lb1, ub1), maxfev=100_000)
    except Exception as exc:
        out["status"] = f"pass1_failed: {exc}"
        return out
    first_amplitude, first_mu, first_sigma = (float(p) for p in first_parameters)
    first_chi2 = float(np.sum(np.square((first_y - _gauss_core(first_x, *first_parameters)) / first_error)))
    first_ndf = int(len(first_x) - 3)
    first_errors = np.sqrt(np.maximum(np.diag(first_covariance), 0.0))
    out.update(pass1_amplitude=first_amplitude, pass1_amplitude_error=float(first_errors[0]),
               pass1_mu=first_mu, pass1_mu_error=float(first_errors[1]), pass1_sigma=first_sigma,
               pass1_sigma_error=float(first_errors[2]), pass1_chi2=first_chi2, pass1_ndf=first_ndf,
               pass1_chi2_over_ndf=_chi2_ndf(first_chi2, first_ndf))

    # pass 2: the Gaussian core mu1 +/- k sigma1 (bin centres, inclusive), visible bins only
    fit_low = max(lower, first_mu - k * first_sigma)
    fit_high = min(upper, first_mu + k * first_sigma)
    second_mask = visible & (centers >= fit_low) & (centers <= fit_high)
    second_x = centers[second_mask]
    second_y = counts[second_mask].astype(float)
    second_error = np.sqrt(second_y)
    out.update(fit_low=fit_low, fit_high=fit_high, core_bin_count=int(len(second_x)))
    if len(second_x) < min_fit_bins:
        out["status"] = f"too_few_core_bins: {len(second_x)} < {min_fit_bins}"
        return out
    lb2 = np.array([0.0, fit_low, max(bin_width * 0.5, first_sigma * 0.2)])
    ub2 = np.array([np.inf, fit_high, first_sigma * 2.5])
    try:
        with _quiet():
            second_parameters, second_covariance = curve_fit(
                _gauss_core, second_x, second_y, p0=first_parameters, sigma=second_error, absolute_sigma=True,
                bounds=(lb2, ub2), maxfev=100_000)
    except Exception as exc:
        out["status"] = f"pass2_failed: {exc}"
        return out
    amplitude, mu, sigma = (float(p) for p in second_parameters)
    chi2 = float(np.sum(np.square((second_y - _gauss_core(second_x, *second_parameters)) / second_error)))
    ndf = int(len(second_x) - 3)
    errors = np.sqrt(np.maximum(np.diag(second_covariance), 0.0))
    out.update(amplitude=amplitude, amplitude_error=float(errors[0]), mu=mu, mu_error=float(errors[1]),
               sigma=sigma, sigma_error=float(errors[2]), sigma_over_mu=_ratio(sigma, mu), chi2=chi2, ndf=ndf,
               chi2_over_ndf=_chi2_ndf(chi2, ndf), status="ok", ok=math.isfinite(mu) and math.isfinite(sigma))
    return out


# --------------------------------------------------------------------------
# user fit ranges of the spectrum pages: --fit / --bin (Lei 2026-10-02)
# --------------------------------------------------------------------------
RANGE_FIT_KINDS = ("range_gauss", "range_double_gauss")  # fitting.range_gauss_fit kinds (1 / 2 ranges)


# default page edges: equal bins on [0, max(upper_min, factor x percentile)]
def default_spectrum_edges(values, bins: int = 90, upper_percentile: float = 99.8, upper_factor: float = 1.08,
                           upper_min: float = 3.0) -> np.ndarray:
    v = _finite(values)
    upper = float(upper_min)
    if v.size:
        upper = max(float(upper_min), float(np.percentile(v, upper_percentile)) * float(upper_factor))
    return np.linspace(0.0, upper, int(bins) + 1)


# checked -fit ranges: 1-2 pairs a < b, not overlapping (ValueError otherwise)
def _user_ranges(ranges, max_ranges: int = MAX_FIT_RANGES["fit"]) -> list[tuple[float, float]]:
    example = "[(5, 10)] or [(15, 32), (50, 120)]"
    out = []
    try:
        for r in ranges:
            if isinstance(r, str) or len(r) != 2:
                raise TypeError
            out.append((float(r[0]), float(r[1])))
    except (TypeError, ValueError):
        raise ValueError(f"fit ranges {ranges!r}: give a list of pairs (a, b), e.g. {example}") from None
    if not 1 <= len(out) <= max_ranges or not all(math.isfinite(a) and math.isfinite(b) and a < b for a, b in out):
        raise ValueError(f"fit ranges {ranges!r}: 1 to {max_ranges} pairs (a, b) of finite numbers with a < b, "
                         f"e.g. {example}")
    overlap = ranges_overlap_error(out)
    if overlap:
        raise ValueError(f"fit ranges {ranges!r}: {overlap}")
    return out


# error of sigma/mu = (sigma/mu) sqrt((ds/sigma)^2 + (dmu/mu)^2 - 2 cov/(mu sigma))
def sigma_over_mu_error(mu, mu_error, sigma, sigma_error, mu_sigma_cov=0.0) -> float:
    m, dm, s, ds, cov = (float(v) for v in (mu, mu_error, sigma, sigma_error, mu_sigma_cov))
    if not all(math.isfinite(v) for v in (m, dm, s, ds, cov)) or m == 0 or s == 0:
        return _NAN
    rel2 = (ds / s) ** 2 + (dm / m) ** 2 - 2.0 * cov / (m * s)  # relative error^2 with the covariance term
    return abs(s / m) * math.sqrt(max(rel2, 0.0))


# checked -bin values: one positive integer per range, or None
def _user_bins(bins, ranges) -> list[int] | None:
    if bins is None or (isinstance(bins, (list, tuple)) and not len(bins)):
        return None
    try:
        values = [int(b) for b in bins]
        exact = all(float(b) == int(b) for b in bins)
    except (TypeError, ValueError):
        raise ValueError(f"fit bins {bins!r}: one positive integer per fit range, e.g. [30] or [30, 40]") from None
    problem = range_options_error(ranges, values)
    if problem or not exact or min(values) < 1:
        raise ValueError(problem or f"fit bins {bins!r}: one positive integer per fit range, e.g. [30] or [30, 40]")
    return values


# -fit fit: single Gaussian on [a, b] or double on [a, b], [c, d] (Gaussian integrated per bin)
def range_spectrum_fit(values, ranges, bins=None, default_edges=None, min_bin_count: int = 10) -> dict:
    """A failed fit (also a parameter on its bound) has NaN values, status 'fit_failed'."""
    ranges = _user_ranges(ranges)
    bins = _user_bins(bins, ranges)
    v = _finite(values)
    page_edges = np.asarray(default_spectrum_edges(v) if default_edges is None else default_edges, dtype=float)
    fit = range_gauss_fit(v, ranges, range_bin_edges(ranges, bins, page_edges), min_bin_count=int(min_bin_count))
    if page_edges.size >= 2:
        page_counts = np.histogram(v, bins=page_edges)[0]
        page_width = float(page_edges[1] - page_edges[0])
        lower, upper = float(page_edges[0]), float(page_edges[-1])
    else:
        page_counts, page_width, lower, upper = np.zeros(0, dtype=np.int64), _NAN, _NAN, _NAN
    page_centers = 0.5 * (page_edges[:-1] + page_edges[1:])
    ratio_error = _NAN
    if fit["ok"] and len(ranges) == 1:  # with the mu-sigma covariance of the fit (one-sided ranges correlate them)
        ratio_error = sigma_over_mu_error(fit["mu"], fit["mu_error"], fit["sigma"], fit["sigma_error"],
                                          fit["mu_sigma_cov"])
    rule =(f"{','.join(str(b) for b in bins)} equal bins per range" if bins
            else f"default page bins (width {page_width:.4g})")
    fit.update(sigma_over_mu_error=float(ratio_error), page_edges=page_edges, page_centers=page_centers,
               page_counts=page_counts, page_width=page_width, hist_lower=lower, hist_upper=upper,
               edges=page_edges, centers=page_centers, counts=page_counts, fit_bins=bins,
               fit_ranges_text=format_ranges(ranges), bin_rule=rule)
    return fit


# range_spectrum_fit with fit_ranges, fit_bins, min_bin_count of the settings
def range_spectrum_from_settings(values, rs: dict, default_edges=None) -> dict:
    if not rs.get("fit_ranges"):
        raise ValueError("range_spectrum_from_settings: no fit_ranges (--fit[a,b]) in the settings")
    return range_spectrum_fit(values, rs["fit_ranges"], bins=rs.get("fit_bins"), default_edges=default_edges,
                              min_bin_count=int(rs["min_bin_count"]))


# --------------------------------------------------------------------------
# linearity, resolution, e/h
# --------------------------------------------------------------------------
# point errors: mu_error = sigma / sqrt(N), ratio = sigma/mu, ratio_error per prescription
def point_errors(mu, sigma, n=None, sigma_error=None, prescription: str = "sigma_over_sqrt_n") -> dict:
    if prescription not in ERROR_PRESCRIPTIONS:
        raise ValueError(f"unknown error prescription {prescription!r}; choose from {ERROR_PRESCRIPTIONS}")
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_arr = np.full(mu.shape, np.nan) if n is None else np.asarray(n, dtype=float)
    with np.errstate(all="ignore"):
        ratio = sigma / mu
        mu_error = sigma / np.sqrt(n_arr)  # mu_error = sigma / sqrt(N)
        if prescription == "sigma_over_sqrt_n":
            ratio_error = mu_error / mu
        elif prescription == "sigma_fit_error":
            s_err = np.full(mu.shape, np.nan) if sigma_error is None else np.asarray(sigma_error, dtype=float)
            ratio_error = s_err / mu
        else:
            ratio_error = np.abs(ratio) * np.sqrt(1.0 / (2.0 * n_arr) + np.square(ratio) / n_arr)
    return {"mu_error": mu_error, "ratio": ratio, "ratio_error": ratio_error, "prescription": prescription}


# indices of the selected points sorted by energy
def _sorted_indices(sel: np.ndarray, energies: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(sel)
    return idx[np.argsort(energies[idx], kind="stable")]


# (a, b) in GeV of a range value, None for None
def gev_range(value, what: str = "fit_range") -> tuple[float, float] | None:
    if value is None:
        return None
    r = value
    if isinstance(r, (list, tuple, np.ndarray)) and len(r) == 1 and isinstance(r[0], (list, tuple, np.ndarray)):
        r = r[0]
    a = b = _NAN
    if isinstance(r, (list, tuple, np.ndarray)) and len(r) == 2:
        try:
            a, b = float(r[0]), float(r[1])
        except (TypeError, ValueError):
            pass
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError(f"{what}={value!r}: give None or (a, b) in GeV with a < b")
    return a, b


# particle names of the points as a lower-case string array (None: every point is an electron)
def particle_array(particles, n: int) -> np.ndarray:
    if particles is None:
        return np.full(n, ELECTRON, dtype=object)
    out = np.array([str(p).strip().lower() for p in particles], dtype=object)
    if out.size != n:
        raise ValueError(f"{out.size} particle names for {n} points")
    return out


# particles in drawing order: electron first, then in order of appearance
def _particle_order(particles) -> list[str]:
    seen = list(dict.fromkeys(str(p) for p in particles))
    return sorted(seen, key=lambda p: (p != ELECTRON, seen.index(p)))


# PDG error scale: sqrt(chi2/ndf) when chi2/ndf > 1, else 1
def error_scale(chi2: float, ndf: int, applicable: bool = True) -> float:
    if not applicable or ndf is None or ndf <= 0 or not np.isfinite(chi2):
        return 1.0
    ratio = float(chi2) / float(ndf)
    return math.sqrt(ratio) if ratio > 1.0 else 1.0


# straight line through the selected points: weighted least squares (1/mu_err^2) or unweighted polyfit
def _line_fit(E, y, err, sel, weighted: bool) -> dict:
    idx = _sorted_indices(sel, E)
    res = {"ok": False, "message": "", "slope": _NAN, "intercept": _NAN, "slope_error": _NAN,
           "intercept_error": _NAN, "covariance": np.full((2, 2), np.nan), "chi2": _NAN, "ndf": int(len(idx) - 2),
           "n_fitted": int(len(idx)), "idx": idx}
    if len(idx) < 2:
        res["message"] = f"fewer than 2 points ({len(idx)})"
        return res
    x, yy = E[idx], y[idx]
    design = np.column_stack([x, np.ones_like(x)])
    try:
        with _quiet():
            if weighted:
                yerr = err[idx]
                weights = 1.0 / np.square(yerr)
                covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
                slope, intercept = covariance @ (design.T @ (weights * yy))
                residual = yy - design @ np.array([slope, intercept])
                chi2 = float(np.sum(np.square(residual / yerr)))
            else:
                slope, intercept = np.polyfit(x, yy, 1)
                residual = yy - (slope * x + intercept)
                chi2 = float(np.sum(np.square(residual)))
                ndf = len(idx) - 2
                covariance = np.linalg.inv(design.T @ design) * (chi2 / ndf if ndf > 0 else np.nan)
    except (np.linalg.LinAlgError, ValueError) as exc:
        res["message"] = f"singular fit: {exc}"
        return res
    res.update(slope=float(slope), intercept=float(intercept), covariance=covariance, chi2=chi2,
               slope_error=float(np.sqrt(covariance[0, 0])) if covariance[0, 0] >= 0 else _NAN,
               intercept_error=float(np.sqrt(covariance[1, 1])) if covariance[1, 1] >= 0 else _NAN)
    finite_cov = bool(np.all(np.isfinite(covariance))) or not weighted  # unweighted 2-point fit has no errors
    if not (np.isfinite(slope) and np.isfinite(intercept) and finite_cov):
        res["message"] = "non-finite parameters or covariance"
        return res
    res["ok"] = True
    return res


# line prediction slope * E + intercept and its error from the full covariance
def line_prediction(result: dict, energies_gev, scaled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    design = np.column_stack([x, np.ones_like(x)])
    prediction = result["slope"] * x + result["intercept"]
    cov = result.get("covariance_scaled") if scaled else result["covariance"]
    with np.errstate(all="ignore"):
        variance = np.einsum("ij,jk,ik->i", design, np.asarray(cov, dtype=float), design)
    return prediction, np.sqrt(np.maximum(variance, 0.0))


# energies as 'e1;e2;...'
def _energy_list(energies, fmt="{:g}") -> str:
    return ";".join(fmt.format(e) for e in energies)


# one linearity line mu = slope * E + intercept through the selected points with a <= E <= b GeV (None: all)
def linearity_line(energies_gev, mu, mu_err=None, fit_range=None, weighted: bool = True,
                   normal_error: bool = False, select=None, particle: str = "") -> dict:
    """select: the candidate points (e.g. one particle); the masks and predictions cover every point."""
    E = np.asarray(energies_gev, dtype=float).ravel()
    y = np.asarray(mu, dtype=float).ravel()
    weighted = bool(weighted) and mu_err is not None
    err = np.full(E.shape, np.nan) if mu_err is None else np.asarray(mu_err, dtype=float).ravel()
    usable = np.isfinite(E) & np.isfinite(y)
    if weighted:
        usable &= np.isfinite(err) & (err > 0)
    if select is not None:
        usable &= np.asarray(select, dtype=bool).ravel()
    rng = gev_range(fit_range, "linearity fit_range")
    sel = usable if rng is None else usable & in_range(E, rng)
    chosen = _line_fit(E, y, err, sel, weighted)
    who = f"{particle} points" if particle else "points"
    where = "" if rng is None else f" with {rng[0]:g} <= E <= {rng[1]:g} GeV"
    out = {"kind": "linearity_line", "particle": particle, "weighted": weighted, "normal_error": bool(normal_error),
           "fit_range": None if rng is None else [rng[0], rng[1]], "ok": bool(chosen["ok"]),
           "status": "ok" if chosen["ok"] else f"failed: {who}{where}: {chosen['message']}"}
    fitted = np.zeros(E.shape, dtype=bool)
    if chosen["ok"]:
        fitted[chosen["idx"]] = True
        out.update({k: chosen[k] for k in ("slope", "slope_error", "intercept", "intercept_error", "covariance",
                                           "chi2", "ndf", "n_fitted")})
    else:
        out.update(slope=_NAN, slope_error=_NAN, intercept=_NAN, intercept_error=_NAN, chi2=_NAN, ndf=0,
                   covariance=np.full((2, 2), np.nan), n_fitted=0)
    out["chi2_over_ndf"] = _chi2_ndf(out["chi2"], out["ndf"]) if chosen["ok"] else _NAN
    scale = error_scale(out["chi2"], out["ndf"], weighted) if chosen["ok"] else _NAN
    out.update(error_scale=scale, slope_error_scaled=out["slope_error"] * scale,
               intercept_error_scaled=out["intercept_error"] * scale,
               covariance_scaled=np.asarray(out["covariance"], dtype=float) * scale ** 2)
    out["fitted_energies"] = (_energy_list(np.sort(E[fitted])) + " GeV") if fitted.any() else ""
    predicted, predicted_error = line_prediction(out, E)
    with np.errstate(all="ignore"):
        residual = y - predicted
        residual_fraction = residual / predicted
    out.update(fitted=fitted, predicted=predicted, predicted_error=predicted_error,
               predicted_error_scaled=line_prediction(out, E, scaled=True)[1], residual=residual,
               residual_fraction=residual_fraction)
    return out


# linearity of one scale: every particle's points, one line per particle of fits ({particle: [a, b] | None})
def linearity(energies_gev, mu, mu_err=None, particles=None, fits=None, weighted: bool = True,
              normal_error: bool = False) -> dict:
    """fits {} or None: points only. Electrons and pions are never fitted together."""
    E = np.asarray(energies_gev, dtype=float).ravel()
    y = np.asarray(mu, dtype=float).ravel()
    err = np.full(E.shape, np.nan) if mu_err is None else np.asarray(mu_err, dtype=float).ravel()
    parts = particle_array(particles, E.size)
    specs = {str(p).lower(): (None if r is None else list(gev_range(r, f"linearity fit range of {p}")))
             for p, r in (fits or {}).items()}
    lines = {p: linearity_line(E, y, mu_err, fit_range=r, weighted=weighted, normal_error=normal_error,
                               select=parts == p, particle=p) for p, r in specs.items()}
    fitted = np.zeros(E.shape, dtype=bool)
    for line in lines.values():
        fitted |= line["fitted"]
    status = "; ".join(f"{p}: {line['status']}" for p, line in lines.items()) or "no fit: points only"
    return {"kind": "linearity", "energies_gev": E, "mu": y, "mu_error": err, "particles": parts,
            "usable": np.isfinite(E) & np.isfinite(y), "weighted": bool(weighted) and mu_err is not None,
            "normal_error": bool(normal_error), "fit_specs": specs, "fits": lines, "fitted": fitted,
            "ok": bool(lines) and all(line["ok"] for line in lines.values()), "status": status}


# rows of the linearity fits of one scale (one per fitted particle)
def linearity_rows(result: dict) -> list[dict]:
    rows = []
    for p, line in result["fits"].items():
        rng = line["fit_range"]
        rows.append({"particle": p, "fit_range_gev": "all points" if rng is None else format_ranges([rng]),
                     "ok": bool(line["ok"]), "status": line["status"], "weighted": bool(line["weighted"]),
                     **{k: float(line[k]) for k in ("slope", "slope_error", "intercept", "intercept_error", "chi2")},
                     "ndf": int(line["ndf"]), "chi2_over_ndf": float(line["chi2_over_ndf"]),
                     "error_scale": float(line["error_scale"]),
                     "slope_error_scaled": float(line["slope_error_scaled"]),
                     "intercept_error_scaled": float(line["intercept_error_scaled"]),
                     "n_fitted": int(line["n_fitted"]), "fitted_energies": line["fitted_energies"]})
    return rows


# resolution fit: sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2) (E in MeV) on the fit particle's points
def resolution_fit(energies_gev, sigma_over_mu, err=None, particles=None,
                   fit_particle: str = ELECTRON, p0=(50.0, 8.0, 0.05), bounded: bool = True, maxfev=100_000,
                   normal_error: bool = False, fit_range=None) -> dict:
    """particles None: every point belongs to fit_particle; the other particles' points are drawn only."""
    E = np.asarray(energies_gev, dtype=float).ravel()
    E_mev = 1000.0 * E
    r = np.asarray(sigma_over_mu, dtype=float).ravel()
    re_ = np.full(E.shape, np.nan) if err is None else np.asarray(err, dtype=float).ravel()
    fit_particle = str(fit_particle).lower()
    parts = particle_array(particles, E.size) if particles is not None else np.full(E.size, fit_particle, object)
    usable = np.isfinite(E) & (E > 0) & np.isfinite(r) & np.isfinite(re_) & (re_ > 0)  # weighted fit
    sel = usable & (parts == fit_particle)
    rng = gev_range(fit_range, "resolution fit_range")
    if rng is not None:
        sel = sel & in_range(E, rng)
    idx = _sorted_indices(sel, E)
    n_par = 3
    out = {"kind": "resolution", "model": "three_term", "fit_particle": fit_particle, "particles": parts,
           "fit_range": None if rng is None else [rng[0], rng[1]],
           "ok": False, "status": "ok", "a": _NAN, "a_error": _NAN,
           "b": _NAN, "b_error": _NAN, "c": _NAN, "c_error": _NAN, "covariance": np.full((n_par, n_par), np.nan),
           "chi2": _NAN, "ndf": int(len(idx) - n_par), "chi2_over_ndf": _NAN, "n_fitted": 0,
           "fitted_energies": "", "p0": list(np.asarray(p0, dtype=float)),
           "bounded": bool(bounded), "normal_error": bool(normal_error),
           "error_scale": _NAN, "a_error_scaled": _NAN, "b_error_scaled": _NAN, "c_error_scaled": _NAN,
           "covariance_scaled": np.full((n_par, n_par), np.nan),
           "energies_gev": E, "energies_mev": E_mev, "ratio": r, "ratio_error": re_, "usable": usable,
           "fitted": np.zeros(E.shape, dtype=bool), "predicted": np.full(E.shape, np.nan)}
    if len(idx) < n_par:
        out["status"] = f"too_few_points: {len(idx)} {fit_particle} points < {n_par}" + (
            f" with {rng[0]:g} <= E <= {rng[1]:g} GeV" if rng is not None else "")
        return out
    x, y = E_mev[idx], r[idx]
    try:
        with _quiet():
            kwargs = {"p0": np.asarray(p0, dtype=float), "sigma": re_[idx], "absolute_sigma": True}
            if bounded:
                kwargs["bounds"] = (np.zeros(3), np.full(3, np.inf))
            if maxfev:
                kwargs["maxfev"] = int(maxfev)
            params, covariance = curve_fit(_resolution_three_term, x, y, **kwargs)
            fitted_y = _resolution_three_term(x, *params)
            chi2 = float(np.sum(np.square((y - fitted_y) / re_[idx])))
            params = np.abs(params)  # the model only depends on a^2, b^2, c^2
    except Exception as exc:
        out["status"] = f"fit_failed: {exc}"
        return out
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    scale = error_scale(chi2, out["ndf"])
    names = ("a", "b", "c")
    for i, name in enumerate(names):
        out[name] = float(params[i])
        out[name + "_error"] = float(errors[i])
        out[name + "_error_scaled"] = float(errors[i]) * scale
    if not np.all(np.isfinite(params)):  # failed: no parameters, no fitted point, no curve
        for name in names:
            out[name] = out[name + "_error"] = out[name + "_error_scaled"] = _NAN
        out["status"] = "fit_failed: non-finite parameters"
        return out
    out["fitted"][idx] = True
    out.update(covariance=covariance, chi2=chi2, chi2_over_ndf=_chi2_ndf(chi2, out["ndf"]), n_fitted=int(len(idx)),
               fitted_energies=_energy_list(x) + " MeV", ok=True,
               error_scale=scale, covariance_scaled=np.asarray(covariance, dtype=float) * scale ** 2,
               predicted=resolution_curve(out, E_mev))
    return out


# fitted sigma/mu at the given energies [MeV]
def resolution_curve(result: dict, energies_mev) -> np.ndarray:
    e = np.asarray(energies_mev, dtype=float)
    with np.errstate(all="ignore"):
        return _resolution_three_term(e, result["a"], result["b"], result["c"])


# linearity fits of a scale: the -linfit / yml linearity_fits when set, else the scale's default (ci paths)
def scale_linearity_fits(rs: dict, scale: str | None = None) -> dict:
    """scale: the ci path (adc_only | PureCI | s, any case); None or another name (adcpeak): no default fit."""
    chosen = rs.get("linearity_fits")
    if chosen is None:
        chosen = SCALE_LINEARITY_FITS.get(str(scale).lower(), {}) if scale is not None else {}
    return {str(p).lower(): (None if r is None else [float(r[0]), float(r[1])]) for p, r in chosen.items()}


# linearity with the options of the settings and the fits of the scale
def linearity_from_settings(energies_gev, mu, mu_err, particles, rs: dict, scale: str | None = None) -> dict:
    return linearity(energies_gev, mu, mu_err, particles=particles, fits=scale_linearity_fits(rs, scale),
                     weighted=bool(rs["linearity_weighted"]), normal_error=bool(rs["normal_error"]))


# resolution_fit with the options of the settings (fitted: the electron points)
def resolution_from_settings(energies_gev, sigma_over_mu, err, rs: dict, particles=None,
                             fit_particle: str = ELECTRON) -> dict:
    return resolution_fit(energies_gev, sigma_over_mu, err, particles=particles, fit_particle=fit_particle, p0=rs["resolution_p0"],
                          bounded=bool(rs["resolution_bounded"]), maxfev=rs["resolution_maxfev"],
                          normal_error=bool(rs["normal_error"]), fit_range=rs.get("resolution_fit_range"))


# e/h(E) = mu_electron(E) / mu_hadron(E) of one scale: electron run at E (filled) or electron line at E (hollow)
def e_over_h(energies_gev, mu, mu_err, particles, electron_line: dict | None = None, hadron: str = HADRON,
             keys=None) -> dict:
    """One point per hadron run; error = e/h sqrt((d mu_e / mu_e)^2 + (d mu_h / mu_h)^2). No fit."""
    E = np.asarray(energies_gev, dtype=float).ravel()
    y = np.asarray(mu, dtype=float).ravel()
    err = np.full(E.shape, np.nan) if mu_err is None else np.asarray(mu_err, dtype=float).ravel()
    parts = particle_array(particles, E.size)
    hadron = str(hadron).lower()
    keys = [""] * E.size if keys is None else [str(k) for k in keys]
    electron = (parts == ELECTRON) & np.isfinite(E) & np.isfinite(y) & (y > 0)
    line_ok = bool(electron_line and electron_line.get("ok"))
    scaled = bool(electron_line and electron_line.get("normal_error"))
    rows = []
    for i in np.flatnonzero(parts == hadron):
        e, mu_h, dmu_h = float(E[i]), float(y[i]), float(err[i])
        same = electron & np.isclose(E, e, rtol=1e-9, atol=1e-12)
        mu_e = dmu_e = _NAN
        source, note = "", ""
        if same.any():  # several electron runs at E: mean, error sqrt(sum err^2) / n
            mu_e = float(np.mean(y[same]))
            dmu_e = float(np.sqrt(np.sum(np.square(err[same])))) / int(same.sum())
            source = "electron run"
        elif line_ok:
            p, pe = line_prediction(electron_line, [e], scaled=scaled)
            mu_e, dmu_e, source = float(p[0]), float(pe[0]), "electron line fit"
        else:
            note = f"no electron run at {e:g} GeV and no electron line fit"
        if not note and not (math.isfinite(mu_e) and mu_e > 0):
            note = f"the electron line gives mu_electron <= 0 at {e:g} GeV"
        if not (math.isfinite(mu_h) and mu_h > 0):
            note = f"no fitted {hadron} peak"
        ratio = ratio_err = _NAN
        if not note:
            ratio = mu_e / mu_h
            ratio_err = abs(ratio) * math.sqrt((dmu_e / mu_e) ** 2 + (dmu_h / mu_h) ** 2)
        rows.append({"run": keys[i], "hadron": hadron, "energy_gev": e, "e_over_h": ratio,
                     "e_over_h_error": ratio_err, "mu_electron": mu_e, "mu_electron_error": dmu_e,
                     "mu_hadron": mu_h, "mu_hadron_error": dmu_h, "electron_source": source,
                     "filled": source == "electron run" and not note, "note": note})
    rows.sort(key=lambda r: r["energy_gev"])
    line_text = ""
    if line_ok:
        rng = electron_line.get("fit_range")
        line_text = "electron line fit to " + ("every electron point" if rng is None else
                                              f"{rng[0]:g} <= E <= {rng[1]:g} GeV")
    return {"kind": "e_over_h", "hadron": hadron, "rows": rows, "electron_line_ok": line_ok,
            "electron_line_text": line_text,
            "energies_gev": np.array([r["energy_gev"] for r in rows], dtype=float),
            "e_over_h": np.array([r["e_over_h"] for r in rows], dtype=float),
            "e_over_h_error": np.array([r["e_over_h_error"] for r in rows], dtype=float),
            "filled": np.array([r["filled"] for r in rows], dtype=bool)}


# rows of e/h (one per hadron run)
def eh_rows(result: dict) -> list[dict]:
    return [dict(r) for r in result["rows"]]


# --------------------------------------------------------------------------
# tables and settings
# --------------------------------------------------------------------------
# scalar entries of a result dict (one csv row)
def scalar_fields(result: dict, prefix: str = "", rename: dict | None = None) -> dict:
    row = {}
    for key, value in result.items():
        if isinstance(value, (np.generic,)):
            value = value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[prefix + (rename or {}).get(key, key)] = value
    return row


# yml section energy_response; unknown and command-line-only keys stop the run
def yml_section(ctx) -> dict:
    section = dict(ctx.stage_cfg(CONFIG_SECTION) or {}) if ctx is not None else {}
    cli_only = [f"{k}: {why}" for k, why in CLI_ONLY_SETTINGS.items() if k in section]
    if cli_only:
        raise SettingsError(f"energy_response: yml section {CONFIG_SECTION} (configure/config.yml or local.yml):\n  "
                            + "\n  ".join(cli_only))
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise SettingsError(f"energy_response: unknown setting(s) {unknown} in the yml section {CONFIG_SECTION} "
                            f"(configure/config.yml or local.yml); the valid keys are those of "
                            f"energy_response.DEFAULTS: {sorted(DEFAULTS)}")
    return section


# energy-response settings: DEFAULTS < yml energy_response < flags
def settings(ctx=None, overrides: dict | None = None) -> dict:
    out = {**DEFAULTS, **dict.fromkeys(CLI_ONLY_SETTINGS)}
    if ctx is not None:
        out.update(yml_section(ctx))
    out.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return out


# settings rule of resolution_p0: 3 finite numbers
def _p0_rule(value) -> str | None:
    ok = (isinstance(value, (list, tuple)) and len(value) == 3
          and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                  and math.isfinite(float(v)) for v in value))
    return None if ok else "a list of 3 finite numbers [a, b, c]"


# number (not bool)?
def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


# [a, b] of finite numbers with a < b?
def _pair_ok(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_number(x) and math.isfinite(float(x)) for x in v)
            and float(v[0]) < float(v[1]))


# settings rule of fit_ranges: 1-2 pairs
def _fit_ranges_rule(value) -> str | None:
    ok = value is None or (isinstance(value, (list, tuple)) and 1 <= len(value) <= MAX_FIT_RANGES["fit"]
                           and all(_pair_ok(r) for r in value))
    return None if ok else ("null or a list of 1-2 pairs [a, b] with a < b (one range: single Gaussian, two: "
                            "double Gaussian), e.g. [[5, 10]] or [[15, 32], [50, 120]] (-fit[a,b] / "
                            "-fit[a,b],[c,d])")


# settings rule of fit_bins: 1-2 positive integers
def _fit_bins_rule(value) -> str | None:
    ok = value is None or (isinstance(value, (list, tuple)) and 1 <= len(value) <= MAX_FIT_RANGES["fit"]
                           and all(_is_number(n) and math.isfinite(float(n)) and float(n) == int(n) and int(n) >= 1
                                   for n in value))
    return None if ok else ("null or a list of 1-2 positive integers, one per fit range, e.g. [30] or [30, 40] "
                            "(-bin N / -bin N1,N2)")


# settings rule of resolution_fit_range: null or one pair [a, b] in GeV
def _resfit_rule(value) -> str | None:
    return None if value is None or _pair_ok(value) else \
        "null or one pair [a, b] in GeV with a < b, e.g. [0.3, 1.0] (-resfit[a,b])"


# settings rule of linearity_fits: null or {particle: null | [a, b]}
def _linearity_fits_rule(value) -> str | None:
    ok = value is None or (isinstance(value, dict) and all(
        isinstance(p, str) and p.strip().lower() in RUN_PARTICLES and (r is None or _pair_ok(r))
        for p, r in value.items()))
    return None if ok else (f"null or a mapping {{particle: null (every point) or [a, b] in GeV}}, particles "
                            f"{', '.join(RUN_PARTICLES)}, e.g. {{electron: [0, 1], pion: null}} "
                            f"(-linfit electron[0,1] pion; {{}} = no fit)")


SETTINGS_RULES = {
    "spectrum_bins": "int>0", "min_bin_count": "int>=0", "window_nsigma": "float>0", "min_fit_bins": "int>0",
    "hist_upper_percentile": ("range", 0, 100), "hist_upper_factor": "float>0", "hist_upper_min": "float",
    "peak_scan_bins": "int>0", "peak_scan_upper_percentile": ("range", 0, 100), "peak_scan_upper_factor": "float>0",
    "peak_scan_upper_min": "float", "fit_ranges": _fit_ranges_rule, "fit_bins": _fit_bins_rule,
    "linearity_fits": _linearity_fits_rule, "resolution_fit_range": _resfit_rule,
    "linearity_weighted": "bool", "normal_error": "bool", "error_prescription": ("choice", list(ERROR_PRESCRIPTIONS)),
    "resolution_p0": _p0_rule, "resolution_bounded": "bool",
    "resolution_maxfev": "int>0?",
}


# check the energy-response settings before any IO (rules, -fit / -bin together); JSON-friendly ranges
def check_settings(s: dict, source: str = "energy_response") -> dict:
    validate_settings(s, SETTINGS_RULES, source=source)
    problem = range_options_error(s.get("fit_ranges"), s.get("fit_bins"))
    if problem:
        raise SettingsError(f"{source}: {problem}")
    if s.get("fit_ranges") is not None:
        s["fit_ranges"] = [[float(a), float(b)] for a, b in s["fit_ranges"]]
    if s.get("fit_bins") is not None:
        s["fit_bins"] = [int(n) for n in s["fit_bins"]]
    if s.get("resolution_fit_range") is not None:
        s["resolution_fit_range"] = [float(s["resolution_fit_range"][0]), float(s["resolution_fit_range"][1])]
    if s.get("linearity_fits") is not None:
        s["linearity_fits"] = {p.strip().lower(): (None if r is None else [float(r[0]), float(r[1])])
                               for p, r in s["linearity_fits"].items()}
    return s


# names of the settings the shared flags set (a stage reads them from its argparse namespace)
FLAG_KEYS = ("fit_ranges", "fit_bins", "linearity_fits", "resolution_fit_range", "normal_error",
             "spectrum_bins", "min_bin_count", "window_nsigma", "error_prescription")
_BARE_FLAG = object()  # argparse const of a -fit given without a value (nargs '?'): refused by _RangeFlag


# text of a parsed range-flag value (for messages)
def _flag_value_text(value) -> str:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return format_ranges(value)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, float) for v in value):
        return format_ranges([value])
    return ",".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)


# argparse action of a range flag, stored once (a second one stops the parser)
class _RangeFlag(argparse.Action):

    # action with an example and a repeat hint
    def __init__(self, option_strings, dest, example: str = "", repeat_hint: str = "", **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.example, self.repeat_hint = example, repeat_hint

    # store the value; a bare -fit or a repeated flag stops the parser
    def __call__(self, parser, namespace, values, option_string=None):
        if values is _BARE_FLAG:
            parser.error(f"{option_string} needs a range: {self.example} (single Gaussian) or "
                         f"{self.example.split('[')[0]}[a,b],[c,d] (double Gaussian), in the x units of the "
                         f"spectrum page")
        previous = getattr(namespace, self.dest, None)
        if previous is not None:
            hint = self.repeat_hint
            if self.dest == "fit_ranges" and len(previous) == 1 and len(values) == 1:
                (a, b), (c, d) = previous[0], values[0]
                hint = f"for a double Gaussian give both ranges in ONE flag: -fit[{a:g},{b:g}],[{c:g},{d:g}]"
            elif self.dest == "fit_bins" and len(previous) == 1 and len(values) == 1:
                hint = f"give one value per fit range in ONE flag: -bin {previous[0]},{values[0]}"
            parser.error(f"{option_string} given twice ({_flag_value_text(previous)} and {_flag_value_text(values)}): "
                         f"a later one would replace the earlier one; {hint}")
        setattr(namespace, self.dest, values)


# argparse type of -fit: '[a,b]' or '[a,b],[c,d]'
def fit_ranges_arg(text) -> list[tuple[float, float]]:
    return parse_ranges(text, max_ranges=MAX_FIT_RANGES["fit"], flag="--fit")


# argparse type of -resfit: '[a,b]' in GeV
def resfit_range_arg(text) -> list[float]:
    (a, b), = parse_ranges(text, max_ranges=MAX_FIT_RANGES["resfit"], flag="--resfit")
    return [a, b]


LINFIT_EXAMPLE = "-linfit electron[0,1] (several particles: -linfit electron[0,1] pion[0,4]; every point of a " \
                 "particle: -linfit electron; no fit: -linfit none)"
_LINFIT_ITEM = re.compile(r"[\s,;]*(?P<name>[A-Za-z]+)\s*,?\s*(?:\[(?P<range>[^\[\]]*)\])?[\s,;]*")


# '<particle>[a,b] <particle> ...' or 'none' -> {particle: [a, b] | None} (one range per particle)
def parse_linfit(text) -> dict:
    """PowerShell pieces ('electron[0', '1]') are joined with commas before parsing."""
    s = str(text).strip()
    if s.lower() == "none":
        return {}
    out, pos = {}, 0
    while pos < len(s):
        m = _LINFIT_ITEM.match(s, pos)
        if not m or m.end() == pos:
            raise argparse.ArgumentTypeError(f"-linfit {s!r}: name the particle in front of its range, e.g. "
                                             f"{LINFIT_EXAMPLE}")
        name = m.group("name").lower()
        if name not in RUN_PARTICLES and name.endswith("s") and name[:-1] in RUN_PARTICLES:
            name = name[:-1]
        if name not in RUN_PARTICLES:
            raise argparse.ArgumentTypeError(f"-linfit {s!r}: unknown particle {m.group('name')!r}; the particles "
                                             f"are {', '.join(RUN_PARTICLES)}, e.g. {LINFIT_EXAMPLE}")
        if name in out:
            raise argparse.ArgumentTypeError(f"-linfit {s!r}: {name} given twice (one line, one range per "
                                             f"particle)")
        rng = None
        if m.group("range") is not None:
            try:
                (a, b), = parse_ranges(f"[{m.group('range')}]", max_ranges=1, flag="--linfit")
            except argparse.ArgumentTypeError:
                raise argparse.ArgumentTypeError(f"-linfit {s!r}: {name}[{m.group('range')}] needs one range "
                                                 f"[a,b] in GeV with a < b, e.g. {LINFIT_EXAMPLE}") from None
            rng = [a, b]
        out[name] = rng
        pos = m.end()
    if not out:
        raise argparse.ArgumentTypeError(f"-linfit needs a particle, e.g. {LINFIT_EXAMPLE}")
    return out


# argparse action of -linfit: joins the tokens, parses them; a second -linfit adds its particles
class _LinfitFlag(argparse.Action):

    # store {particle: [a, b] | None}; 'none' = no fit
    def __call__(self, parser, namespace, values, option_string=None):
        tokens = [values] if isinstance(values, str) else list(values)
        try:
            fits = parse_linfit(",".join(t.strip() for t in tokens if t.strip()))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        previous = getattr(namespace, self.dest, None)
        if previous is not None:
            if not previous or not fits:
                parser.error(f"{option_string}: 'none' (no fit) cannot be combined with particle fits")
            twice = sorted(set(previous) & set(fits))
            if twice:
                parser.error(f"{option_string}: {', '.join(twice)} given twice (one line, one range per particle)")
            fits = {**previous, **fits}
        setattr(namespace, self.dest, fits)


# shared energy-response flags of ci and adcpeak (-fit, -bin, -linfit, -resfit, ...)
def add_fit_arguments(p) -> None:
    g = p.add_argument_group("energy-response fits (shared, ldmx_hcal.energy.energy_response; yml section "
                             "energy_response)")
    quote = " (bash/zsh: quote the brackets, '-fit[14,26]', or write -fit 14,26)"
    g.add_argument("--fit", "-fit", dest="fit_ranges", action=_RangeFlag, nargs="?", const=_BARE_FLAG,
                   type=fit_ranges_arg, default=None, metavar="[a,b][,[c,d]]", example="-fit[a,b]",
                   repeat_hint="for a double Gaussian give both ranges in ONE flag, e.g. -fit[15,32],[50,120]",
                   help="spectrum peaks: SINGLE Gaussian fitted directly on [a, b] (x units of the spectrum page), "
                        "e.g. -fit[14,26]; two ranges -fit[a,b],[c,d] fit a DOUBLE Gaussian, one component per "
                        "range (the ranges must not overlap; they may touch). A failed fit, also a mu or sigma on "
                        "its bound (e.g. mu on the range edge), leaves the points only (no curve, no value). "
                        "Command line only, once" + quote + ". Default: the automatic single-Gaussian fit of each "
                        "page")
    g.add_argument("--bin", "-bin", dest="fit_bins", action=_RangeFlag, type=parse_bins, default=None,
                   metavar="N[,N2]", example="-bin N",
                   repeat_hint="give one value per fit range in ONE flag, e.g. -bin 30,40",
                   help="with -fit: N equal bins in each fit range, one value per range (-bin 30, or -bin 30,40 "
                        "with two ranges; a single Gaussian needs N >= 4, a double >= 3 per range and >= 7 in all). "
                        "Default: the bins of the default page whose centres lie in the range; those with >= "
                        "-min-bin-count entries are fitted (= the drawn points of the two-pass pages; the peak-scan "
                        "pages also draw sparser bins: -min-bin-count 1 fits every non-empty bin)")
    g.add_argument("--linfit", "-linfit", dest="linearity_fits", action=_LinfitFlag, nargs="+", default=None,
                   metavar="PARTICLE[a,b]",
                   help="linearity: one straight line per named particle through its points with a <= E <= b GeV "
                        "(both inclusive); a particle without a range uses every point of it; electrons and pions "
                        "are never fitted together, the other particles' points are drawn only. E.g. "
                        "-linfit electron[0,1] or -linfit electron[0,1] pion (bash/zsh: 'electron[0,1]'); "
                        "-linfit none: no fit. Default: per ci scale adc_only no fit, PureCI electron below 1 GeV, "
                        "s electron (every point); adcpeak no fit")
    g.add_argument("--resfit", "-resfit", dest="resolution_fit_range", action=_RangeFlag, type=resfit_range_arg,
                   default=None, metavar="[a,b]", example="-resfit[0.3,1]", repeat_hint="one range only",
                   help="resolution: fit the sigma/mu model only to the electron points with a <= E <= b GeV (both "
                        "inclusive), e.g. -resfit[0.3,1] (bash/zsh: '-resfit[0.3,1]' or -resfit 0.3,1). Default: "
                        "every electron point; the hadron points are drawn, never fitted")
    g.add_argument("--normalerror", "-normalerror", "--normal-error", "-normal-error", dest="normal_error",
                   action="store_const", const=True, default=None,
                   help="error analysis mode: when chi2/ndf > 1 also give the linearity and resolution parameter "
                        "errors multiplied by sqrt(chi2/ndf) (tables: both columns; figures: the scaled errors) "
                        "(default off: statistical errors only)")
    g.add_argument("--spectrum-bins", "-spectrum-bins", type=int, default=None, metavar="N",
                   help=f"bins of the two-pass Gaussian core fit (default {DEFAULTS['spectrum_bins']})")
    g.add_argument("--min-bin-count", "-min-bin-count", type=int, default=None, metavar="N",
                   help=f"bins with fewer entries are not drawn (two-pass pages) and not fitted (also -fit range "
                        f"fits) (default {DEFAULTS['min_bin_count']})")
    g.add_argument("--window-nsigma", "-window-nsigma", type=float, default=None, metavar="K",
                   help=f"second pass fits mu1 +/- K sigma1 (default {DEFAULTS['window_nsigma']})")
    g.add_argument("--error-prescription", "-error-prescription", choices=ERROR_PRESCRIPTIONS, default=None,
                   help=f"linearity/resolution point errors (default {DEFAULTS['error_prescription']})")


# settings with the shared flags, checked
def settings_from_args(ctx, args) -> dict:
    return check_settings(settings(ctx, {k: getattr(args, k, None) for k in FLAG_KEYS}))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
_BLUE = "#28679A"
_POINT_BLUE = "#347DB7"
_ORANGE = "#D8832F"
_RED = "#B5231E"
_LIGHT_BLUE = "#DCEAF4"
_INK = "#20252B"
_SUBTLE = "#5A626A"
_GRID = "#D8DDE3"
_FOOTER = "#69727C"


# unit text of an axis label ('arbitrary units' -> 'a.u.')
def _unit(units_label: str) -> str:
    m = re.search(r"\[([^\]]+)\]", units_label or "")
    unit = m.group(1) if m else (units_label or "")
    return "a.u." if unit.strip().lower() == "arbitrary units" else unit


# font scale for the axes width
def _font_scale(ax) -> float:
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    return float(min(1.0, max(0.55, width_in / 9.5)))


# grid, spines, ticks
def _style_axis(ax, fs: float) -> None:
    ax.grid(color=_GRID, linewidth=0.75, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10.5 * fs, colors=_INK)


# title + subtitle of a page
def _title(ax, title, subtitle, fs, pad=18):
    if title:
        ax.set_title(title, fontsize=17 * fs, color=_INK, pad=pad * fs if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.4 * fs,
                color=_SUBTLE)


# result box in the top right corner
def _box(ax, text, fs, color=_RED):
    ax.text(0.975, 0.965, text, transform=ax.transAxes, ha="right", va="top", fontsize=10.0 * fs, color=color,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "linewidth": 1.0,
                  "alpha": 0.95})


# one 11 x 8.5 in page: (fig, ax)
def new_page(figsize=(11.0, 8.5)):
    from ..common import plotting  # noqa: F401  (Agg backend + rcParams)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize, constrained_layout=True)


LOG_COUNT_FLOOR = 8.0  # lower edge of the log count axis (bins with >= 10 entries drawn)


# lower edge of a log count axis: min(8, 0.8 x smallest drawn count)
def log_count_bottom(drawn_counts) -> float:
    c = np.asarray(drawn_counts, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    return min(LOG_COUNT_FLOOR, 0.8 * float(c.min())) if c.size else LOG_COUNT_FLOOR


# log y limits for the drawn counts (never inverted)
def set_log_count_limits(ax, drawn_counts, top: float | None = None) -> None:
    bottom = log_count_bottom(drawn_counts)
    if top is None:
        ax.set_ylim(bottom=bottom)
        top = ax.get_ylim()[1]
    c = np.asarray(drawn_counts, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if not top > bottom:
        top = max(10.0 * bottom, 1.5 * float(c.max())) if c.size else 10.0 * bottom
    ax.set_ylim(bottom, top)


# footer text under the axes
def add_footer(ax, text: str) -> None:
    ax.text(0.995, -0.115, text, transform=ax.transAxes, ha="right", va="top", fontsize=8.2 * _font_scale(ax),
            color=_FOOTER)


_GREY_POINT = "#A7AEB5"
_COMPONENT_GREEN = "#2E8B57"
RANGE_PAGE_HEADROOM = 0.2  # --fit page: the top 20 % of the log y axis stays free for the legend and result box


# 'value +/- error' text
def _pm(value, error, fmt: str) -> str:
    text = format(value, fmt)
    return text + (f" +/- {format(error, fmt)}" if np.isfinite(error) else "")


# -fit page: grey page bins outside the range(s), blue fitted bins, curve(s) if converged
def _plot_range_spectrum(ax, values, fit: dict, units_label: str, title: str | None = None,
                         subtitle: str | None = None) -> None:
    fs = _font_scale(ax)
    unit = _unit(units_label)
    n = fit.get("n_values")
    if n is None:
        n = int(_finite(values).size)
    ok = bool(fit.get("ok"))
    min_count = int(fit.get("min_bin_count", DEFAULTS["min_bin_count"]))
    ranges = [tuple(float(x) for x in r) for r in (fit.get("ranges") or [])]
    double = len(ranges) == 2
    custom = bool(fit.get("fit_bins"))
    w = float(fit.get("page_width", _NAN))
    per = w if np.isfinite(w) and w > 0 else None  # the y unit: events per page bin
    centers = np.asarray(fit.get("page_centers") if fit.get("page_centers") is not None else [], dtype=float)
    counts = np.asarray(fit.get("page_counts") if fit.get("page_counts") is not None else [], dtype=float)
    inside = np.zeros(centers.shape, dtype=bool)
    for r in ranges:
        inside |= in_range(centers, r)
    drawn = []
    grey = ~inside & (counts >= min_count)
    if grey.any():
        ax.scatter(centers[grey], counts[grey], s=20 * fs, color=_GREY_POINT, edgecolors="none", alpha=0.95,
                   label=f"default page bins outside the fit range(s) (count >= {min_count})", zorder=3)
        drawn.append(counts[grey])
    xs, ys = [], []
    for seg in fit.get("segments") or []:
        keep = np.asarray(seg["fitted"], dtype=bool)
        if keep.any():
            seg_counts = np.asarray(seg["counts"], dtype=float)[keep]
            xs.append(np.asarray(seg["centers"], dtype=float)[keep])
            # default bins are the page bins (factor 1); --bin bins are scaled to the page bin width
            ys.append(seg_counts * per / np.asarray(seg["widths"], dtype=float)[keep] if (custom and per)
                      else seg_counts)
    if xs:
        label = f"fitted bins in the range(s) (count >= {min_count})"
        if custom and per:
            label = f"fitted --bin bins (count >= {min_count}), scaled to the page bin width"
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=26 * fs, color=_POINT_BLUE, edgecolors="none",
                   alpha=0.95, label=label, zorder=4)
        drawn.append(np.concatenate(ys))
    for a, b in ranges:
        for position in (a, b):
            ax.axvline(position, color=_RED, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    curve_max = 0.0
    if ok and per:
        if double:  # the sum over the span of both ranges; the components dashed on top of it
            cx = np.linspace(fit["fit_low"], fit["fit_high"], 800)
            total = range_fit_curve(fit, cx, per)
            ax.plot(cx, total, color=_RED, linewidth=2.0, label="double Gaussian fit (sum)", zorder=5)
            for c, comp in enumerate(fit["components"]):
                a, b = comp["range"]
                ax.plot(cx, range_fit_curve(fit, cx, per, component=c), color=(_ORANGE, _COMPONENT_GREEN)[c % 2],
                        linestyle="--", linewidth=1.4, label=f"component in [{a:g}, {b:g}]", zorder=6)
        else:
            a, b = ranges[0]
            cx = np.linspace(a, b, 500)
            total = range_fit_curve(fit, cx, per)
            ax.plot(cx, total, color=_RED, linewidth=2.0, label="Gaussian fit (drawn only in the fit range)", zorder=5)
        finite = total[np.isfinite(total)]
        curve_max = float(finite.max()) if finite.size else 0.0
        for comp in fit["components"]:
            ax.axvline(comp["mu"], color=_RED, linewidth=1.5, zorder=2)
    if ok:
        chi2 = f"chi2/ndf = {fit['chi2']:.1f}/{fit['ndf']}"
        if double:
            lines = ["Double Gaussian fit"]
            for comp in fit["components"]:
                a, b = comp["range"]
                lines.append(f"[{a:g}, {b:g}]: mu = {_pm(comp['mu'], comp['mu_error'], '.2f')}, "
                             f"sigma = {_pm(comp['sigma'], comp['sigma_error'], '.2f')} {unit}")
            text = "\n".join(lines + [chi2])
        else:
            text = (f"Gaussian fit in {fit.get('fit_ranges_text', '')} {unit}\n"
                    f"mu = {_pm(fit['mu'], fit['mu_error'], '.2f')} {unit}\n"
                    f"sigma = {_pm(fit['sigma'], fit['sigma_error'], '.2f')} {unit}\n"
                    f"sigma/mu = {_pm(fit['sigma_over_mu'], fit.get('sigma_over_mu_error', _NAN), '.3f')}\n{chi2}")
        _box(ax, text, fs)
    else:  # points only (Lei 2026-10-02): no curve, no substitute value
        _box(ax, "fit failed:\n" + textwrap.fill(str(fit.get("status", "")), 56), fs)
    every = np.concatenate(drawn) if drawn else np.zeros(0)
    if np.any(every > 0):
        # explicit top (the Gaussian tails, down to ~1e-300, would stretch the log autoscale margin by decades):
        # the highest point or curve sits RANGE_PAGE_HEADROOM of the axis height below the top
        ax.set_yscale("log")
        peak = max(float(every.max()), curve_max)
        ratio = peak / log_count_bottom(every)
        top = peak * max(ratio ** (RANGE_PAGE_HEADROOM / (1.0 - RANGE_PAGE_HEADROOM)), 1.5) if ratio > 1 else None
        set_log_count_limits(ax, every, top=top if top is not None else 10.0 * peak)
    lows = [v for v in (fit.get("hist_lower"), fit.get("fit_low")) if v is not None and np.isfinite(v)]
    highs = [v for v in (fit.get("hist_upper"), fit.get("fit_high")) if v is not None and np.isfinite(v)]
    if lows and highs and min(lows) < max(highs):
        ax.set_xlim(min(lows), max(highs))
    ax.set_xlabel(units_label, fontsize=13 * fs)
    ax.set_ylabel(f"Events per {w:.3g} {unit}" if per else "Events per bin", fontsize=13 * fs)
    if subtitle is None:
        model = "double Gaussian, one component per range" if double else "single Gaussian"
        subtitle = (f"fit range(s) {fit.get('fit_ranges_text', '')} {unit} ({model})\n"
                    f"{fit.get('bin_rule', '')}; bins with >= {min_count} entries fitted")
        if custom and per:
            subtitle += f"; points scaled to the page bin width {w:.3g} {unit}"
    ax.grid(axis="y", which="major", color=_GRID, linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10.5 * fs, colors=_INK)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9.4 * fs, loc="upper left")
    _title(ax, title or f"N = {n:,}", subtitle, fs, pad=18 if "\n" not in subtitle else 34)


# spectrum page of any spectrum fit (two-pass, peak-scan, -fit range)
def plot_spectrum_with_fit(ax, values, fit: dict, units_label: str, title: str | None = None,
                           subtitle: str | None = None) -> None:
    if fit.get("kind") in RANGE_FIT_KINDS:
        _plot_range_spectrum(ax, values, fit, units_label, title=title, subtitle=subtitle)
        return
    fs = _font_scale(ax)
    unit = _unit(units_label)
    kind = fit.get("kind")
    n = fit.get("n_values")
    if n is None:
        n = int(_finite(values).size)
    ok = bool(fit.get("ok"))
    ax.set_xlabel(units_label, fontsize=13 * fs)
    if fit.get("centers") is None:
        ax.text(0.5, 0.5, f"no histogram: {fit.get('status', 'no data')}", transform=ax.transAxes, ha="center",
                va="center", color=_RED, fontsize=11 * fs)
        _title(ax, title or f"N = {n:,}", subtitle, fs)
        return
    centers, counts = fit["centers"], fit["counts"]
    if kind == "two_pass":
        visible = fit["visible_mask"]
        ax.scatter(centers[visible], counts[visible], s=24 * fs, color=_POINT_BLUE, edgecolors="none", alpha=0.95,
                   label=f"{fit['bins']}-bin event counts (count >= {fit['min_bin_count']})", zorder=3)
        ylabel = "Events per bin"
        if subtitle is None:
            subtitle = (f"{fit['bins']} equal-width bins; bin width = {fit['bin_width']:.3g} {unit}; "
                        f"second range = mu1 +/- {fit['window_nsigma']:g} sigma1 "
                        f"({100 * fit['window_gaussian_coverage']:.2f}% Gaussian core)")
        curve_label = "second-pass fit (drawn only in fit range)"
        box_title = "Second-pass Gaussian fit"
        if visible.any():
            ax.set_yscale("log")
            set_log_count_limits(ax, counts[visible])
        ax.set_xlim(fit["hist_lower"], fit["hist_upper"])

        # fitted curve of the two-pass fit: Gaussian
        def curve(x):
            return _gauss_core(x, fit["amplitude"], fit["mu"], fit["sigma"])
    else:  # peak_scan
        ax.stairs(counts, fit["edges"], color=_POINT_BLUE, label=f"{len(counts)}-bin event counts", zorder=3)
        ylabel = "Events per bin"
        if subtitle is None:
            subtitle = (f"peak scan: Gaussian + constant fit around the most prominent peak "
                        f"({fit['n_detected_peaks']} found)")
        curve_label = "Gaussian + constant fit"
        box_title = "Peak-scan fit"
        ax.set_xlim(0.0, fit["hist_upper"])

        # fitted curve of the peak scan: Gaussian + constant
        def curve(x):
            return _gauss_with_constant(x, fit["amplitude"], fit["mu"], fit["sigma"], fit["constant"])

    if ok and np.isfinite(fit.get("fit_low", np.nan)) and np.isfinite(fit.get("amplitude", np.nan)):
        cx = np.linspace(fit["fit_low"], fit["fit_high"], 500)
        ax.plot(cx, curve(cx), color=_RED, linewidth=2.0, label=curve_label, zorder=4)
    if ok:
        ax.axvline(fit["mu"], color=_RED, linewidth=1.7, zorder=2)
        for position in (fit.get("fit_low"), fit.get("fit_high")):
            if position is not None and np.isfinite(position):
                ax.axvline(position, color=_RED, linestyle="--", linewidth=1.2, alpha=0.75, zorder=2)
        mu_err = fit.get("mu_error", np.nan)
        sig_err = fit.get("sigma_error", np.nan)
        err_mu = f" +/- {mu_err:.2f}" if np.isfinite(mu_err) else ""
        err_sig = f" +/- {sig_err:.2f}" if np.isfinite(sig_err) else ""
        note = "" if fit.get("status") == "ok" else f"\n({str(fit.get('status'))[:48]})"
        _box(ax, f"{box_title}\nmu = {fit['mu']:.2f}{err_mu} {unit}\nsigma = {fit['sigma']:.2f}{err_sig} {unit}\n"
                 f"sigma/mu = {fit['sigma_over_mu']:.3f}{note}", fs)
    else:
        _box(ax, f"fit failed:\n{fit.get('status', '')}", fs)
    ax.set_ylabel(ylabel, fontsize=13 * fs)
    ax.grid(axis="y", which="major", color=_GRID, linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10.5 * fs, colors=_INK)
    ax.legend(frameon=False, fontsize=9.4 * fs, loc="upper left")
    _title(ax, title or f"N = {n:,}", subtitle, fs)


# staggered label offsets for close points
def _label_offsets(x: np.ndarray, span: float) -> list:
    order = np.argsort(x, kind="stable")
    clusters, current = [], []
    for i in order:
        if current and (x[i] - x[current[-1]]) >= 0.06 * span:
            clusters.append(current)
            current = []
        current.append(i)
    if current:
        clusters.append(current)
    stagger = [(-18, 10), (0, 27), (18, 44)]
    offsets = [(0, 10)] * len(x)
    for cl in clusters:
        if len(cl) > 1:
            for j, i in enumerate(cl):
                offsets[i] = stagger[j % 3]
    return offsets


# '; errors x sqrt(chi2/ndf) = x1.23' of a -normalerror fit
def _scaled_note(result: dict) -> str:
    if not result.get("normal_error") or not np.isfinite(result.get("error_scale", np.nan)):
        return ""
    return f"; errors x sqrt(chi2/ndf) = x{result['error_scale']:.3g}"


NO_POINT_REASON = "no fitted spectrum value: the spectrum fit failed, or a two-range (double Gaussian) fit"


# 'no point: 0.3 GeV (reason)' lines for datasets without a point
def _no_point_note(energies, reasons=None, scale: float = 1.0, unit: str = "GeV", suffix: str = "") -> str:
    groups: dict[str, list[float]] = {}
    for i, e in enumerate(energies):
        why = (reasons[i] if reasons is not None and i < len(reasons) else "") or NO_POINT_REASON
        groups.setdefault(str(why), []).append(float(e) * scale)
    lines = [f"no point: {', '.join(f'{e:g}' for e in sorted(set(es)))} {unit} ({why})" for why, es in groups.items()]
    return "\n".join(textwrap.fill(line, 70, break_on_hyphens=False) for line in lines) + suffix


# red x on the energy axis + the note for datasets without a point
def _draw_no_points(ax, x_missing, note: str, fs: float, corner: str) -> None:
    ax.plot(x_missing, np.zeros(len(x_missing)), "x", color=_RED, markersize=9 * fs, markeredgewidth=2.0,
            clip_on=False, zorder=6, label="no point (see the note)")
    x = 0.985 if corner == "right" else 0.015
    ax.text(x, 0.03, note, transform=ax.transAxes, ha=corner, va="bottom", fontsize=9.0 * fs, color=_RED,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": _RED, "linewidth": 0.8,
                  "alpha": 0.95})


_PARTICLE_STYLE = {ELECTRON: (_BLUE, "o"), "pion": (_ORANGE, "s"), "muon": (_COMPONENT_GREEN, "^")}


# (colour, marker) of a particle's points and line
def _particle_style(particle: str) -> tuple[str, str]:
    return _PARTICLE_STYLE.get(str(particle), (_SUBTLE, "D"))


# x upper edge of an energy axis [GeV] for the drawn and missing points
def _energy_axis_top(E, shown) -> tuple[float, float]:
    emax = float(np.nanmax(E[shown])) if shown.any() else 4.0
    return emax, (1.0625 * emax if emax > 0 else 1.0)


# 'electron: mu = (a +/- e) E + (b +/- e) unit; chi2/ndf' of one line
def _line_text(particle: str, line: dict, unit: str) -> str:
    rng = line.get("fit_range")
    where = "every point" if rng is None else f"{rng[0]:g} <= E <= {rng[1]:g} GeV"
    if not line["ok"]:
        return f"{particle} line ({where}): {line['status']}"
    scaled = bool(line.get("normal_error"))
    se, ie = ((line["slope_error_scaled"], line["intercept_error_scaled"]) if scaled
              else (line["slope_error"], line["intercept_error"]))
    return (f"{particle} ({where}): mu = ({line['slope']:.4f} +/- {se:.4f}) E + ({line['intercept']:.3f} +/- "
            f"{ie:.3f}) {unit}; chi2/ndf = {line['chi2']:.1f}/{line['ndf']}{_scaled_note(line)}")


# linearity page: every particle's points, one fitted line (+ band) per fitted particle
def plot_linearity(ax, result: dict, units_label: str, title: str | None = None, counts=None,
                   no_point_reasons=None) -> None:
    fs = _font_scale(ax)
    unit = _unit(units_label)
    E, y, err, parts = result["energies_gev"], result["mu"], result["mu_error"], result["particles"]
    drawn = np.isfinite(E) & np.isfinite(y)
    missing = np.isfinite(E) & ~np.isfinite(y)
    emax, x_hi = _energy_axis_top(E, drawn | missing)
    yerr = np.where(np.isfinite(err), err, 0.0)
    fx = np.linspace(0.0, 1.05 * emax, 500)
    for p, line in result["fits"].items():
        if not line["ok"]:
            continue
        color = _particle_style(p)[0]
        scaled = bool(line.get("normal_error"))
        pred, pred_err = line_prediction(line, fx, scaled=scaled)
        ax.fill_between(fx, pred - pred_err, pred + pred_err, color=color, alpha=0.15, linewidth=0)
        rng = line.get("fit_range")
        span = "every point" if rng is None else f"{rng[0]:g} <= E <= {rng[1]:g} GeV"
        ax.plot(fx, pred, color=color, linewidth=2.2,
                label=f"{p} line ({'weighted' if line['weighted'] else 'unweighted'}) fit to {span}, 1-sigma band")
    for p in _particle_order(parts[drawn]):
        color, marker = _particle_style(p)
        mine = drawn & (parts == p)
        fitted = mine & result["fitted"]
        rest = mine & ~fitted
        line = result["fits"].get(p)
        if fitted.any():
            ax.errorbar(E[fitted], y[fitted], yerr=yerr[fitted], fmt=marker, markersize=7.5 * fs, color=color,
                        markerfacecolor=color, capsize=3.5, linewidth=1.2, label=f"{p} points in the fit")
        if rest.any():
            what = "outside the fit range" if line is not None else "drawn only, not fitted"
            ax.errorbar(E[rest], y[rest], yerr=yerr[rest], fmt=marker, markersize=8.0 * fs, color=color,
                        markerfacecolor="white", markeredgewidth=1.8, capsize=3.5, linewidth=1.2,
                        label=f"{p} points ({what})")
        if line is not None and line["ok"]:  # residual of the particle's own points outside its fit range
            for j in np.flatnonzero(rest):
                ax.plot([E[j], E[j]], [line["predicted"][j], y[j]], color=color, linestyle=":", linewidth=1.4)
                if np.isfinite(line["residual_fraction"][j]):
                    ax.annotate(f"{100 * line['residual_fraction'][j]:+.1f}%", xy=(E[j], y[j]), xytext=(12, -4),
                                textcoords="offset points", ha="left", va="top", fontsize=8.5 * fs, color=color)
    if counts is not None and drawn.any():
        n = np.asarray(counts, dtype=float)
        for i, off in zip(np.flatnonzero(drawn), _label_offsets(E[drawn], x_hi)):
            label = f"{y[i]:.2f}" + (f"\nN={int(n[i]):,}" if np.isfinite(n[i]) else "")
            ax.annotate(label, xy=(E[i], y[i]), xytext=off, textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5 * fs, color=_SUBTLE)
    if missing.any():  # a run without a point is named, not dropped silently
        idx = np.flatnonzero(missing)
        reasons = None if no_point_reasons is None else [no_point_reasons[i] for i in idx]
        _draw_no_points(ax, E[idx], _no_point_note(E[idx], reasons), fs, "right")
    lines = [_line_text(p, line, unit) for p, line in result["fits"].items()]
    subtitle = "\n".join(lines) if lines else "no fit: points only (-linfit <particle>[a,b] fits one line per particle)"
    _title(ax, title or "Energy linearity", subtitle, fs, pad=18 + 16 * subtitle.count("\n"))
    ax.set_xlabel("Beam energy E [GeV]", fontsize=13 * fs)
    ax.set_ylabel(f"Fitted reconstructed peak mu [{unit}]", fontsize=13 * fs)
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(bottom=0.0)
    _style_axis(ax, fs)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9.5 * fs, loc="upper left")


# resolution page: the curve fitted to the electron points, the other particles' points drawn only
def plot_resolution(ax, result: dict, units_label: str, title: str | None = None, no_point_reasons=None) -> None:
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fs = _font_scale(ax)
    unit = _unit(units_label)
    E, r, rerr = result["energies_mev"], result["ratio"], result["ratio_error"]
    parts = result.get("particles")
    if parts is None:
        parts = np.full(E.size, result.get("fit_particle", ELECTRON), dtype=object)
    fit_particle = result.get("fit_particle", ELECTRON)
    drawn = np.isfinite(E) & (E > 0) & np.isfinite(r)
    missing = np.isfinite(E) & (E > 0) & ~np.isfinite(r)
    fitted = result["fitted"]
    span = drawn | missing
    lo = float(np.min(E[span])) / 1.2 if span.any() else 250.0
    hi = 1.125 * float(np.max(E[span])) if span.any() else 4500.0
    if hi <= lo:
        lo, hi = 0.5 * lo, 2.0 * hi
    yerr = np.where(np.isfinite(rerr), rerr, 0.0)
    if result["ok"]:
        fx = np.geomspace(lo, hi, 600)
        ax.plot(fx, resolution_curve(result, fx), color=_particle_style(fit_particle)[0], linewidth=2.3,
                label=f"three-term resolution fit to the {fit_particle} points")
    for p in _particle_order(parts[drawn]):
        color, marker = _particle_style(p)
        mine = drawn & (parts == p)
        used = mine & fitted
        rest = mine & ~fitted
        if used.any():
            ax.errorbar(E[used], r[used], yerr=yerr[used], fmt=marker, markersize=7.5 * fs, color=color,
                        markerfacecolor=color, capsize=3.5, linewidth=1.2,
                        label=f"{p} sigma/mu ({unit} spectra), fitted")
        if rest.any():
            what = "outside the fit range" if p == fit_particle else "drawn only, not fitted"
            ax.errorbar(E[rest], r[rest], yerr=yerr[rest], fmt=marker, markersize=7.5 * fs, color=color,
                        markerfacecolor="white", markeredgewidth=1.8, capsize=3.5, linewidth=1.2,
                        label=f"{p} sigma/mu ({what})")
    for x, y in zip(E[drawn], r[drawn]):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.8 * fs, color=_SUBTLE)
    if missing.any():  # a run without a point is named, not dropped silently
        idx = np.flatnonzero(missing)
        reasons = None if no_point_reasons is None else [no_point_reasons[i] for i in idx]
        _draw_no_points(ax, E[idx], _no_point_note(E[idx], reasons, unit="MeV"), fs, "left")
    scaled = bool(result.get("normal_error"))
    ae, be, ce = ((result["a_error_scaled"], result["b_error_scaled"], result["c_error_scaled"]) if scaled
                  else (result["a_error"], result["b_error"], result["c_error"]))
    line1 = f"sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2), E in MeV, fitted to the {fit_particle} points"
    line2 = (f"a = {result['a']:.3f} +/- {ae:.3f} MeV, b = {result['b']:.4f} +/- "
             f"{be:.4f} sqrt(MeV), c = {result['c']:.5f} +/- {ce:.5f}; "
             f"chi2/ndf = {result['chi2']:.1f}/{result['ndf']}{_scaled_note(result)}")
    if result.get("fit_range") is not None:  # -resfit[a,b] (GeV): the fitted points, in the MeV of the axis too
        a, b = result["fit_range"]
        line1 += f"; only {1000 * a:g} <= E <= {1000 * b:g} MeV (-resfit[{a:g},{b:g}] GeV)"
    subtitle = line1 + "\n" + (line2 if result["ok"] else f"resolution fit {result['status']}")
    _title(ax, title or "Reconstructed energy resolution", subtitle, fs, pad=48)
    ax.set_xlabel("Beam energy E [MeV] (log scale)", fontsize=13 * fs)
    ax.set_ylabel("Energy resolution sigma/mu", fontsize=13 * fs)
    ax.set_xscale("log")
    candidates = [100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000]
    ticks = [t for t in candidates if lo <= t <= hi]
    if ticks:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0.0)
    _style_axis(ax, fs)
    ax.grid(which="minor", color=_GRID, linewidth=0.45, alpha=0.45)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9.5 * fs, loc="upper right")


# e/h page: filled points (electron run at the energy), hollow points (electron line fit), no fit
def plot_eh(ax, result: dict, scale_label: str = "s", title: str | None = None) -> None:
    fs = _font_scale(ax)
    E, eh, err = result["energies_gev"], result["e_over_h"], result["e_over_h_error"]
    filled = result["filled"]
    drawn = np.isfinite(E) & np.isfinite(eh)
    missing = np.isfinite(E) & ~np.isfinite(eh)
    _, x_hi = _energy_axis_top(E, drawn | missing)
    yerr = np.where(np.isfinite(err), err, 0.0)
    color, marker = _particle_style(result["hadron"])
    ax.axhline(1.0, color=_SUBTLE, linestyle="--", linewidth=1.0, zorder=1)
    full, hollow = drawn & filled, drawn & ~filled
    if full.any():
        ax.errorbar(E[full], eh[full], yerr=yerr[full], fmt=marker, markersize=8.0 * fs, color=color,
                    markerfacecolor=color, capsize=3.5, linewidth=1.2,
                    label=f"electron and {result['hadron']} run at this energy")
    if hollow.any():
        ax.errorbar(E[hollow], eh[hollow], yerr=yerr[hollow], fmt=marker, markersize=8.0 * fs, color=color,
                    markerfacecolor="white", markeredgewidth=1.8, capsize=3.5, linewidth=1.2,
                    label=f"mu_electron(E) from the {result['electron_line_text'] or 'electron line fit'}")
    for x, v in zip(E[drawn], eh[drawn]):
        ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.8 * fs, color=_SUBTLE)
    if missing.any():
        idx = np.flatnonzero(missing)
        reasons = [result["rows"][i]["note"] for i in idx]
        _draw_no_points(ax, E[idx], _no_point_note(E[idx], reasons), fs, "right")
    subtitle = (f"e/h(E) = mu_electron(E) / mu_{result['hadron']}(E) on the {scale_label} scale (points only, no fit)"
                f"\nfilled: an electron run at the {result['hadron']} energy; hollow: mu_electron(E) from the "
                f"electron line fit")
    _title(ax, title or f"e/h on the {scale_label} scale", subtitle, fs, pad=34)
    ax.set_xlabel("Beam energy E [GeV]", fontsize=13 * fs)
    ax.set_ylabel(f"e/h = mu_electron / mu_{result['hadron']}", fontsize=13 * fs)
    ax.set_xlim(0.0, x_hi)
    values = eh[drawn] + yerr[drawn]
    top = max(1.2, 1.15 * float(np.max(values))) if values.size else 2.0
    ax.set_ylim(0.0, top)
    _style_axis(ax, fs)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9.5 * fs, loc="upper left")


# table cell text
def _format_cell(value) -> str:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan" if math.isnan(value) else str(value)
        return f"{value:.5g}"
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


# table page(s) of per-dataset results
def summary_table_page(pdf, title: str, rows, columns=None, footer: str | None = None,
                       rows_per_page: int = 28) -> None:
    from ..common import plotting
    import matplotlib.pyplot as plt

    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    headers = dict(columns) if isinstance(columns, dict) else {c: c for c in columns}
    keys = list(headers)
    n_pages = max(1, math.ceil(len(rows) / rows_per_page))
    for page in range(n_pages):
        chunk = rows[page * rows_per_page:(page + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.axis("off")
        page_title = title if n_pages == 1 else f"{title} ({page + 1}/{n_pages})"
        ax.set_title(page_title, fontsize=15, color=_INK, loc="left")
        if chunk and keys:
            cells = [[_format_cell(r.get(k)) for k in keys] for r in chunk]
            labels = [headers[k].replace("_", " ") for k in keys]
            table = ax.table(cellText=cells, colLabels=labels, loc="upper center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(max(5.0, min(9.0, 120.0 / max(len(keys), 1))))
            table.scale(1.0, 1.35)
            for (row_i, _), cell in table.get_celld().items():
                cell.set_edgecolor(_GRID)
                if row_i == 0:
                    cell.set_facecolor("#EEF2F6")
                    cell.set_text_props(weight="bold", color=_INK)
        else:
            ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes, color=_SUBTLE)
        if footer:
            fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=8.2, color=_FOOTER)
        plotting.save(fig, pdf)
