"""Shared fits: Gaussian, Landau, Landau (x) Gauss, line, resolution; the -fit / -bin / -resfit options."""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit, minimize
from scipy.special import ndtr

INV_SQRT_2PI = 0.3989422804014327
LANDAU_MPV_SHIFT = -0.22278298  # peak of the standard Landau density (ROOT convention)
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548...


# --------------------------------------------------------------------------
# model functions
# --------------------------------------------------------------------------
# Gaussian with peak height amplitude
def gauss(x, amplitude, mu, sigma):
    x = np.asarray(x, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


_P1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
_Q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
_P2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
_Q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
_P3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
_Q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
_P4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
_Q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
_P5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
_Q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
_P6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
_Q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
_A1 = (0.04166666667, -0.01996527778, 0.02709538966)
_A2 = (-1.845568670, -4.284640743)


# 4th-order polynomial (Horner)
def _poly(c, t):
    return c[0] + (c[1] + (c[2] + (c[3] + c[4] * t) * t) * t) * t


# standard Landau density (CERNLIB DENLAN, = ROOT TMath::Landau)
def landau_standard(v):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        m = v < -5.5
        if m.any():
            u = np.exp(v[m] + 1.0)
            val = np.where(u < 1e-10, 0.0,
                           0.3989422803 * (np.exp(-1.0 / u) / np.sqrt(u))
                           * (1 + (_A1[0] + (_A1[1] + _A1[2] * u) * u) * u))
            out[m] = val
        m = (v >= -5.5) & (v < -1.0)
        if m.any():
            w = v[m]
            u = np.exp(-w - 1.0)
            out[m] = np.exp(-u) * np.sqrt(u) * _poly(_P1, w) / _poly(_Q1, w)
        m = (v >= -1.0) & (v < 1.0)
        if m.any():
            w = v[m]
            out[m] = _poly(_P2, w) / _poly(_Q2, w)
        m = (v >= 1.0) & (v < 5.0)
        if m.any():
            w = v[m]
            out[m] = _poly(_P3, w) / _poly(_Q3, w)
        for lo, hi, p, q in ((5.0, 12.0, _P4, _Q4), (12.0, 50.0, _P5, _Q5), (50.0, 300.0, _P6, _Q6)):
            m = (v >= lo) & (v < hi)
            if m.any():
                u = 1.0 / v[m]
                out[m] = u * u * _poly(p, u) / _poly(q, u)
        m = v >= 300.0
        if m.any():
            w = v[m]
            u = 1.0 / (w - w * np.log(w) / (w + 1.0))
            out[m] = u * u * (1 + (_A2[0] + _A2[1] * u) * u)
    return out


# Landau with its peak at mpv, width eta, area amplitude
def landau(x, mpv, eta, amplitude=1.0):
    loc = mpv - LANDAU_MPV_SHIFT * eta
    return amplitude * landau_standard((np.asarray(x, dtype=float) - loc) / eta) / eta


LANGAUS_BLOCK_ELEMENTS = 1 << 18   # x points x convolution steps evaluated at once (2 MiB per float64 array)


# Landau (x) Gauss as ROOT langaus.C: numerical convolution on a symmetric grid of n_steps points
def langaus(x, mpv, eta, area, sigma, n_steps=100, n_sigma=5.0):
    """mpv is the Landau parameter, not the curve peak; n_steps must be a positive even integer."""
    if int(n_steps) != n_steps or n_steps < 2 or int(n_steps) % 2:
        raise ValueError(f"langaus: n_steps must be a positive even integer, got {n_steps!r}")
    n_steps = int(n_steps)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sigma = max(float(sigma), 1e-9)
    eta = max(float(eta), 1e-9)
    loc = mpv - LANDAU_MPV_SHIFT * eta
    half = n_steps // 2
    k = np.arange(1, half + 1) - 0.5
    step = 2.0 * n_sigma * sigma / n_steps
    offsets = np.concatenate([-n_sigma * sigma + k * step, n_sigma * sigma - k * step])  # symmetric grid
    gaus = np.exp(-0.5 * (offsets / sigma) ** 2)
    rows = max(1, LANGAUS_BLOCK_ELEMENTS // n_steps)
    sums = np.empty(len(x))
    for i in range(0, len(x), rows):
        xx = x[i:i + rows, None] + offsets[None, :]
        fland = landau_standard((xx - loc) / eta) / eta
        sums[i:i + rows] = (fland * gaus[None, :]).sum(axis=1)
    return area * step * sums * INV_SQRT_2PI / sigma


# straight line: slope * x + intercept
def linear(x, slope, intercept):
    return slope * np.asarray(x, dtype=float) + intercept


# resolution model: sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2)
def resolution_model(energy, a, b, c):
    e = np.asarray(energy, dtype=float)
    return np.sqrt(a * a / (e * e) + b * b / e + c * c)


# --------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------
# result of a fit: ok, params, errors, chi2, ndf
@dataclass
class FitResult:
    ok: bool
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    errors: np.ndarray = field(default_factory=lambda: np.array([]))
    chi2: float = np.nan
    ndf: int = 0
    n_points: int = 0
    message: str = ""

    # chi2 / ndf (NaN when ndf = 0)
    @property
    def chi2_ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else np.nan

    # parameter i (NaN when the fit failed)
    def __getitem__(self, i):
        return self.params[i] if self.ok else np.nan


# scipy curve_fit that never raises (ok False instead)
def fit_curve(model, x, y, p0, sigma=None, bounds=(-np.inf, np.inf), maxfev=20000) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    npar = len(p0)
    if len(x) <= npar:
        return FitResult(False, np.full(npar, np.nan), np.full(npar, np.nan), n_points=len(x),
                         message="too few points")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, pcov = curve_fit(model, x, y, p0=p0, sigma=sigma,
                                   absolute_sigma=sigma is not None, bounds=bounds, maxfev=maxfev)
    except (RuntimeError, ValueError, TypeError) as exc:
        return FitResult(False, np.full(npar, np.nan), np.full(npar, np.nan), n_points=len(x), message=str(exc))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf)) if np.all(np.isfinite(pcov)) else np.full(npar, np.nan)
    resid = y - model(x, *popt)
    chi2 = float(np.sum((resid / sigma) ** 2)) if sigma is not None else float(np.sum(resid ** 2))
    ok = bool(np.all(np.isfinite(popt)))
    return FitResult(ok, popt, perr, chi2, len(x) - npar, len(x))


# bin errors sqrt(N), at least 1
def poisson_errors(counts) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(counts, dtype=float), 1.0))


# binned Poisson fit: minimise the deviance 2 Σ(mu - n + n ln(n/mu)), errors from its Hessian
def fit_binned_poisson(model, centers, counts, p0, bounds, compute_errors=True) -> FitResult:
    """ok is the optimizer's convergence flag, not just finite parameters."""
    c = np.asarray(centers, dtype=float)
    n = np.asarray(counts, dtype=float)
    npar = len(p0)
    if len(c) <= npar:
        return FitResult(False, np.full(npar, np.nan), np.full(npar, np.nan), n_points=len(c), message="too few bins")
    lo = np.asarray(bounds[0], dtype=float) if np.ndim(bounds[0]) else np.full(npar, bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float) if np.ndim(bounds[1]) else np.full(npar, bounds[1], dtype=float)

    # Poisson deviance of the parameters p
    def deviance(p):
        mu = np.clip(model(c, *p), 1e-12, None)
        term = n * np.log(np.where(n > 0, n, 1.0) / mu) * (n > 0)
        return 2.0 * np.sum(mu - n + term)

    p_start = np.clip(np.asarray(p0, dtype=float), lo, hi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(deviance, p_start, method="L-BFGS-B", bounds=list(zip(lo, hi)))
    if not np.all(np.isfinite(res.x)):
        return FitResult(False, np.full(npar, np.nan), np.full(npar, np.nan), n_points=len(c), message=str(res.message))
    errs = _hessian_errors(deviance, res.x, scale=0.5) if compute_errors else np.full(npar, np.nan)
    return FitResult(bool(res.success), res.x, errs, float(res.fun), len(c) - npar, len(c), str(res.message))


# curvature matrix check: Cholesky + smallest scaled eigenvalue > rtol x largest
def positive_definite_status(matrix, rtol: float = 1e-10) -> str:
    """Returns OK, NOT_FINITE, NOT_POSITIVE_DEFINITE, SINGULAR or EMPTY."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return "EMPTY"
    if not np.all(np.isfinite(m)):
        return "NOT_FINITE"
    m = 0.5 * (m + m.T)
    d = np.diag(m)
    if np.any(d <= 0):
        return "NOT_POSITIVE_DEFINITE"
    scaled = m / np.sqrt(np.outer(d, d))
    try:
        np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError:
        return "NOT_POSITIVE_DEFINITE"
    w = np.linalg.eigvalsh(scaled)
    if not w[0] > rtol * w[-1]:
        return "SINGULAR"
    return "OK"


# errors sqrt(diag(H^-1)) of a central-difference Hessian; NaN unless positive definite
def _hessian_errors(fun, x, scale=1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1e-3)
    H = np.zeros((k, k))
    f0 = fun(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            if i == j:
                val = (fun(x + ei) - 2 * f0 + fun(x - ei)) / h[i] ** 2
            else:
                val = (fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = val * scale
    if positive_definite_status(H) != "OK":
        return np.full(k, np.nan)
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


# numerical peak position and FWHM of a curve on [lo, hi]
def curve_peak_and_fwhm(func, lo, hi, n=4001):
    x = np.linspace(lo, hi, n)
    y = func(x)
    if not np.any(np.isfinite(y)) or np.nanmax(y) <= 0:
        return np.nan, np.nan
    i = int(np.nanargmax(y))
    half = y[i] / 2.0
    left = np.where(y[:i] < half)[0]
    right = np.where(y[i:] < half)[0]
    if len(left) == 0 or len(right) == 0:
        return x[i], np.nan
    il, ir = left[-1], i + right[0]
    xl = np.interp(half, [y[il], y[il + 1]], [x[il], x[il + 1]])
    xr = np.interp(half, [y[ir], y[ir - 1]], [x[ir], x[ir - 1]])
    return x[i], xr - xl


# analytic weighted least squares y = slope x + intercept (no errors: scaled by sqrt(chi2/ndf))
def weighted_linear_fit(x, y, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    w = np.ones_like(x) if yerr is None else 1.0 / np.asarray(yerr, dtype=float) ** 2
    ok &= np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    n = len(x)
    out = dict(slope=np.nan, slope_error=np.nan, intercept=np.nan, intercept_error=np.nan,
               chi2=np.nan, ndf=n - 2, n_points=n, cov=np.full((2, 2), np.nan))
    if n < 2:
        return out
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx * Sx
    if D == 0:
        return out
    slope = (S * Sxy - Sx * Sy) / D
    intercept = (Sxx * Sy - Sx * Sxy) / D
    cov = np.array([[S / D, -Sx / D], [-Sx / D, Sxx / D]])
    chi2 = float((w * (y - slope * x - intercept) ** 2).sum())
    if yerr is None and n > 2:
        cov = cov * chi2 / (n - 2)
    out.update(slope=slope, intercept=intercept, slope_error=float(np.sqrt(cov[0, 0])),
               intercept_error=float(np.sqrt(cov[1, 1])), chi2=chi2, cov=cov)
    return out


# np.histogram + bin centres
def histogram(values, bins, range_=None):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=range_)
    return counts, edges, 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------
# user fit ranges and bins: --fit / --bin / --resfit (Lei 2026-10-02)
# --------------------------------------------------------------------------
# --fit[a,b]          one range: single Gaussian fitted on the bins in [a, b] (x axis of that figure)
# --fit[a,b],[c,d]    two ranges: double Gaussian, component 1 in [a, b], component 2 in [c, d]; the ranges must not
#                     overlap (they may touch, b == c: every event and every page bin is then fitted once)
# --bin N | N1,N2     N equal bins in each fit range (one value per range); default: the bins of the figure
# --resfit[a,b]       resolution: the usual model fitted on the points with a <= E <= b [GeV]
# Each flag once per command line (a second --fit would silently replace the first: refused, Lei 2026-10-02).
RANGE_FLAGS = ("fit", "resfit")
BIN_FLAG = "bin"
MAX_FIT_RANGES = {"fit": 2, "resfit": 1}
GAUSS_PARAMETERS = 3        # area, mu, sigma of one Gaussian component
MIN_BINS_PER_RANGE = 3      # a component needs >= 3 fitted bins of its own range (else it keeps its start values)
# a parameter within BOUND_RTOL x (b - a) of its bound counts as "at its bound" (failed fit, R3): the bounded TRF
# fit stops ~1e-13 x (b - a) from an active bound, healthy fits stay >= ~1e-2 x (b - a) away (measured 2026-10-02,
# 30 seeds of N(20, 3) x 20000 with one-sided and contained ranges: pinned <= 1.2e-13, healthy >= 0.067)
BOUND_RTOL = 1e-5
AREA_FLOOR = 1e-6           # a component area <= AREA_FLOOR x the fitted entries is "at its bound 0" (vanished)
_EXAMPLES = {"fit": "[5,10]", "resfit": "[0.3,1]"}
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER = re.compile(_NUM)
_FLAG = re.compile(r"(--?)(fit|resfit|bin)((?:[\[=]).*)?", re.S)
_NUMERIC_PIECE = re.compile(rf"[\[\],;\s]*{_NUM}(?:[\[\],;\s]+{_NUM})*[\[\],;\s]*")


# a range flag given twice: argparse-like error, exit code 2
class RangeFlagError(SystemExit):

    # keep the message
    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


# '--fit[5,10]' / '--bin 30' of a joined value, for messages
def _shown(name: str, text: str) -> str:
    if not text:
        return f"--{name}"
    return f"--{name}{text}" if text.startswith("[") else f"--{name} {text}"


# message for a range flag given twice, with the one-flag spelling
def _repeated_flag_message(name: str, first: str, second: str) -> str:
    said = f"--{name} given twice ({_shown(name, first)} and {_shown(name, second)})"
    if name == BIN_FLAG:
        nums = _NUMBER.findall(first) + _NUMBER.findall(second)
        example = ",".join(nums) if len(nums) == 2 else "30,40"
        return (f"{said}: a second --bin would replace the first; give one value per fit range in ONE flag, "
                f"e.g. --bin {example}")
    if name == "fit":
        nums = _NUMBER.findall(first) + _NUMBER.findall(second)
        pairs = [(nums[0], nums[1]), (nums[2], nums[3])] if len(nums) == 4 else [("5", "10"), ("60", "110")]
        both = ",".join(f"[{a},{b}]" for a, b in pairs)
        spaced = " ".join(f"{a},{b}" for a, b in pairs)
        return (f"{said}: a second --fit would replace the first; for a double Gaussian give both ranges in ONE "
                f"flag, --fit{both} (bash/zsh: '--fit{both}' or --fit {spaced})")
    return f"{said}: one range only, a second --{name} would replace the first (e.g. --{name}{_EXAMPLES[name]})"


# join every -fit / -bin / -resfit split by PowerShell into one token '--fit=a,b' (-linfit joins its own pieces)
def normalize_range_args(argv) -> list[str]:
    """In bash quote '--fit[5,10],[60,110]'; a one-dash ``-fit=5,10`` is a PowerShell parse error."""
    out, i, argv = [], 0, [str(a) for a in argv]
    seen = {}
    while i < len(argv):
        m = _FLAG.fullmatch(argv[i])
        if not m:
            out.append(argv[i])
            i += 1
            continue
        name, rest = m.group(2), m.group(3) or ""
        parts = [rest[1:] if rest.startswith("=") else rest] if rest else []
        i += 1
        while i < len(argv):
            text = ",".join(p for p in parts if p)
            depth = text.count("[") - text.count("]")
            nxt = argv[i]
            bracketed = "[" in text
            if depth > 0 or (nxt.startswith("[") and (bracketed or not text)) or \
                    (not bracketed and _NUMERIC_PIECE.fullmatch(nxt) is not None):
                parts.append(nxt)
                i += 1
            else:
                break
        text = ",".join(p for p in parts if p)
        if name in seen:
            message = _repeated_flag_message(name, seen[name], text)
            print(f"runner: error: {message}", file=sys.stderr)
            raise RangeFlagError(message)
        seen[name] = text
        out.append(f"--{name}={text}" if text else f"--{name}")
    return out


# why two fit ranges overlap (touching is allowed), else None
def ranges_overlap_error(ranges, flag: str = "--fit") -> str | None:
    if not ranges or len(ranges) < 2:
        return None
    rs = sorted((float(a), float(b)) for a, b in ranges)
    for (a1, b1), (a2, b2) in zip(rs, rs[1:]):
        if b1 > a2:
            example = (f"{flag}[{a1:g},{a2:g}],[{a2:g},{max(b1, b2):g}]" if a1 < a2
                       else f"{flag}[15,32],[32,120]")
            return (f"the ranges [{a1:g}, {b1:g}] and [{a2:g}, {b2:g}] overlap on [{a2:g}, {min(b1, b2):g}]: every "
                    f"event may enter the fit once (one Gaussian component per range); give ranges that do not "
                    f"overlap (they may touch), e.g. {example}")
    return None


# '[5,10],[60,110]' -> [(5, 10), (60, 110)]: a < b, at most max_ranges, no overlap
def parse_ranges(text, max_ranges: int = 2, flag: str = "--fit",
                 disjoint: bool | None = None) -> list[tuple[float, float]]:
    s = str(text).strip()
    numbers = [float(x) for x in _NUMBER.findall(s)]
    junk = set(_NUMBER.sub("", s)) - set("[](),; ")
    name = flag.lstrip("-")
    example = f"{flag}{_EXAMPLES.get(name, '[5,10]')}" + (f" or {flag}[5,10],[60,110]" if max_ranges > 1 else "")
    if junk or not numbers or len(numbers) % 2:
        hint = ""
        if not junk and len(numbers) >= 3:  # an odd count of 3+: most likely PowerShell's split -flag=0.3,1
            hint = (f" (PowerShell splits a one-dash -{name}=a.b,c at the decimal point: write -{name}[a,b], "
                    f"-{name} a,b or --{name}=a,b)")
        raise argparse.ArgumentTypeError(f"{flag} {text!r}: give pairs of numbers, e.g. {example}{hint}")
    ranges = [(numbers[k], numbers[k + 1]) for k in range(0, len(numbers), 2)]
    if len(ranges) > max_ranges:
        raise argparse.ArgumentTypeError(f"{flag} {text!r}: at most {max_ranges} range(s), got {len(ranges)}")
    bad = [r for r in ranges if not (math.isfinite(r[0]) and math.isfinite(r[1]) and r[0] < r[1])]
    if bad:
        raise argparse.ArgumentTypeError(f"{flag} {text!r}: every range needs a < b (got {bad})")
    if disjoint if disjoint is not None else max_ranges <= MAX_FIT_RANGES["fit"]:
        why = ranges_overlap_error(ranges, flag)
        if why:
            raise argparse.ArgumentTypeError(f"{flag} {text!r}: {why}")
    return ranges


# '30' or '30,40' -> [30] / [30, 40]
def parse_bins(text) -> list[int]:
    s = str(text).strip()
    numbers = _NUMBER.findall(s)
    junk = set(_NUMBER.sub("", s)) - set("[](),; ")
    if junk or not numbers or len(numbers) > 2 or any(not re.fullmatch(r"\+?\d+", x) or int(x) < 1 for x in numbers):
        raise argparse.ArgumentTypeError(f"--bin {text!r}: one positive integer per fit range, e.g. --bin 30 "
                                         f"(one range) or --bin 30,40 (two ranges)")
    return [int(x) for x in numbers]


# why -bin can never give a Gaussian fit (single: >= 4 bins; double: >= 3 per range, >= 7 in all), else None
def gauss_bins_error(ranges, bins) -> str | None:
    if not bins or not ranges or len(bins) != len(ranges):
        return None
    n = len(ranges)
    need_total = GAUSS_PARAMETERS * n + 1
    if min(int(b) for b in bins) >= MIN_BINS_PER_RANGE and sum(int(b) for b in bins) >= need_total:
        return None
    given = ",".join(str(int(b)) for b in bins)
    if n == 1:
        return (f"--bin {given}: a single Gaussian ({GAUSS_PARAMETERS} parameters) needs at least {need_total} bins on "
                f"its range (more bins than parameters), e.g. --bin 30")
    return (f"--bin {given}: a double Gaussian ({GAUSS_PARAMETERS * n} parameters) needs at least "
            f"{MIN_BINS_PER_RANGE} bins per range and {need_total} in all, e.g. --bin 30,40")


# why -fit / -bin cannot be used together, else None (checked before any IO)
def range_options_error(ranges, bins, gauss_min_bins: bool = True) -> str | None:
    if ranges:
        why = ranges_overlap_error(ranges)
        if why:
            return f"--fit {format_ranges(ranges)}: {why}"
    if bins and not ranges:
        return "--bin needs --fit: the bins are those of the fit range(s), e.g. --fit[5,10] --bin 30"
    if bins and len(bins) != len(ranges):
        return (f"--bin needs one value per fit range: --fit has {len(ranges)} range(s) "
                f"({format_ranges(ranges)}), --bin has {len(bins)} value(s) ({','.join(map(str, bins))})")
    if bins and gauss_min_bins:
        return gauss_bins_error(ranges, bins)
    return None


# [(5, 10)] -> '[5, 10]'
def format_ranges(ranges) -> str:
    return ", ".join(f"[{a:g}, {b:g}]" for a, b in (ranges or []))


# a <= x <= b, both ends inclusive
def in_range(x, fit_range, rtol: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a, b = (float(v) for v in fit_range)
    with np.errstate(invalid="ignore"):
        inside = (x >= a) & (x <= b)
    return inside | np.isclose(x, a, rtol=rtol, atol=0.0) | np.isclose(x, b, rtol=rtol, atol=0.0)


# bin edges of every fit range: -bin N equal bins, else the page bins with centres inside
def range_bin_edges(ranges, bins=None, default_edges=None) -> list[np.ndarray]:
    """Two touching ranges share no bin: the bin on the shared end goes to the higher range."""
    out = []
    starts = [float(r[0]) for r in ranges]
    for i, (a, b) in enumerate(ranges):
        if bins:
            out.append(np.linspace(float(a), float(b), int(bins[i]) + 1))
            continue
        e = np.asarray(default_edges, dtype=float)
        c = 0.5 * (e[:-1] + e[1:])
        sel = in_range(c, (a, b))
        if any(j != i and np.isclose(float(b), starts[j], rtol=1e-9, atol=0.0) for j in range(len(ranges))):
            sel &= ~np.isclose(c, float(b), rtol=1e-9, atol=0.0)  # the shared-end bin belongs to the higher range
        idx = np.flatnonzero(sel)
        out.append(e[idx[0]:idx[-1] + 2] if idx.size else np.zeros(0))
    return out


# Gaussian density with integral area
def gauss_area(x, area, mu, sigma):
    x = np.asarray(x, dtype=float)
    return area * INV_SQRT_2PI / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# entries of a Gaussian in a bin: area x (Phi(z_hi) - Phi(z_lo)), survival side above mu
def gauss_bin_integral(lo, hi, area, mu, sigma):
    zl = (np.asarray(lo, dtype=float) - mu) / sigma
    zh = (np.asarray(hi, dtype=float) - mu) / sigma
    return area * np.where(zl > 0, ndtr(-zl) - ndtr(-zh), ndtr(zh) - ndtr(zl))


GAUSS_FIT_KINDS = {1: "range_gauss", 2: "range_double_gauss"}


# fitted parameters on their bound (a failed fit): mu at a range edge, sigma at a bound, area 0
def _bound_problems(popt, ranges, lower, upper, entries: float) -> tuple[list[str], list[str]]:
    problems, names = [], []
    for c, (a, b) in enumerate(ranges):
        a, b = float(a), float(b)
        area, mu, sigma = (float(p) for p in popt[3 * c:3 * c + 3])
        s_lo, s_hi = float(lower[3 * c + 2]), float(upper[3 * c + 2])
        tol = BOUND_RTOL * (b - a)
        where = format_ranges([(a, b)])
        if mu - a <= tol or b - mu <= tol:
            names.append(f"mu {where}")
            problems.append(f"mu of {where} at its bound {a if mu - a <= b - mu else b:g} (the range edge: no peak "
                            f"inside the range; move or widen the range)")
        if sigma - s_lo <= tol:
            names.append(f"sigma {where}")
            problems.append(f"sigma of {where} at its bound {s_lo:.4g} (half the smallest bin width: the peak is "
                            f"narrower than the bins; use a finer --bin)")
        elif s_hi - sigma <= tol:
            names.append(f"sigma {where}")
            problems.append(f"sigma of {where} at its bound {s_hi:.4g} (the range width: no Gaussian peak inside "
                            f"the range)")
        if area <= AREA_FLOOR * max(entries, 1.0):
            names.append(f"area {where}")
            problems.append(f"area of {where} at its bound 0 (no entries left for this component)")
    return problems, names


# single / double Gaussian on the bins of the user ranges: Neyman chi2 of the bin integrals, one component per range
def range_gauss_fit(values, ranges, edges_list, min_bin_count: int = 10, maxfev: int = 200_000) -> dict:
    """Never raises: a failure gives NaN values, ok False and status 'fit_failed: <reason>'."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_comp = len(ranges)
    nan = float("nan")
    out = {"kind": GAUSS_FIT_KINDS.get(n_comp, "range_gauss"), "ok": False, "status": "fit_failed: no data",
           "ranges": [tuple(map(float, r)) for r in ranges], "n_values": int(v.size), "segments": [],
           "n_fit_bins": 0, "components": [], "min_bin_count": int(min_bin_count),
           "fit_low": float(min(r[0] for r in ranges)) if ranges else nan,
           "fit_high": float(max(r[1] for r in ranges)) if ranges else nan,
           "mu": nan, "mu_error": nan, "sigma": nan, "sigma_error": nan, "sigma_over_mu": nan, "mu_sigma_cov": nan,
           "amplitude": nan, "chi2": nan, "ndf": 0, "chi2_over_ndf": nan, "bound_parameters": []}
    for tag in ("low", "high"):
        out.update({f"mu_{tag}": nan, f"mu_{tag}_error": nan, f"sigma_{tag}": nan, f"sigma_{tag}_error": nan,
                    f"area_{tag}": nan})
    if n_comp not in (1, 2):
        out["status"] = f"fit_failed: {n_comp} ranges (1 or 2 allowed)"
        return out
    overlap = ranges_overlap_error(ranges)
    if overlap:
        out["status"] = f"fit_failed: {overlap}"
        return out
    edges_list = [np.asarray(e, dtype=float) for e in edges_list]
    spans = sorted((float(e[0]), float(e[-1]), k) for k, e in enumerate(edges_list) if e.size >= 2)
    for (lo1, hi1, k1), (lo2, hi2, k2) in zip(spans, spans[1:]):
        if hi1 > lo2:  # bins given directly (not from range_bin_edges) that overlap: an event would count twice
            out["status"] = (f"fit_failed: the bins of {format_ranges([ranges[k1]])} ([{lo1:g}, {hi1:g}]) and of "
                             f"{format_ranges([ranges[k2]])} ([{lo2:g}, {hi2:g}]) overlap: each event may enter the "
                             f"fit once")
            return out
    firsts = [float(e[0]) for e in edges_list if e.size >= 2]
    xs, ys, ws, los, his = [], [], [], [], []
    for rng, edges in zip(ranges, edges_list):
        if edges.size >= 2:
            counts = np.histogram(v, bins=edges)[0].astype(float)
            if any(float(edges[-1]) == f for f in firsts):  # touching segments: an entry on the shared edge once,
                counts[-1] -= np.count_nonzero(v == edges[-1])  # in the higher one (np.histogram closes its last bin)
            centers, widths = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
            lo_e, hi_e = edges[:-1], edges[1:]
        else:
            counts, centers, widths, lo_e, hi_e = (np.zeros(0) for _ in range(5))
        keep = counts >= max(int(min_bin_count), 1)
        out["segments"].append({"range": tuple(map(float, rng)), "edges": edges, "centers": centers,
                                "counts": counts, "widths": widths, "fitted": keep})
        xs.append(centers[keep])
        ys.append(counts[keep])
        ws.append(widths[keep])
        los.append(lo_e[keep])
        his.append(hi_e[keep])
    x, y, w, lo, hi = (np.concatenate(a) if a else np.zeros(0) for a in (xs, ys, ws, los, his))
    out["n_fit_bins"] = int(x.size)
    n_par = GAUSS_PARAMETERS * n_comp
    if x.size <= n_par:
        out["status"] = (f"fit_failed: {x.size} bins with >= {min_bin_count} entries in "
                         f"{format_ranges(ranges)} (need > {n_par})")
        return out
    thin = [(r, int(seg.size)) for r, seg in zip(ranges, xs) if seg.size < MIN_BINS_PER_RANGE]
    if thin:  # a component without its own bins would stay at its start values (no error, no meaning)
        out["status"] = ("fit_failed: " + "; ".join(f"{format_ranges([r])} has {n} bin(s) with >= {min_bin_count} "
                                                    f"entries (need >= {MIN_BINS_PER_RANGE} per range)"
                                                    for r, n in thin))
        return out
    p0, lower, upper = [], [], []
    w_min = float(w.min())
    for i, (a, b) in enumerate(ranges):
        seg_x, seg_y, seg_w = xs[i], ys[i], ws[i]
        if seg_x.size:
            j = int(np.argmax(seg_y / seg_w))
            mu0, area0 = float(seg_x[j]), float(seg_y.sum())
        else:
            mu0, area0 = 0.5 * (a + b), 1.0
        s_lo, s_hi = 0.5 * w_min, max(float(b - a), w_min)
        p0 += [max(area0, 1.0), min(max(mu0, a), b), min(max(0.25 * (b - a), s_lo * 1.0001), s_hi * 0.9999)]
        lower += [0.0, float(a), s_lo]
        upper += [np.inf, float(b), s_hi]

    # expected entries of the fitted bins
    def model(idx, *p):
        idx = np.asarray(idx, dtype=int)
        total = np.zeros(idx.size)
        for c in range(n_comp):
            total += gauss_bin_integral(lo[idx], hi[idx], *p[3 * c:3 * c + 3])
        return total

    err = np.sqrt(np.maximum(y, 1.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, pcov = curve_fit(model, np.arange(x.size), y, p0=p0, sigma=err, absolute_sigma=True,
                                   bounds=(lower, upper), maxfev=int(maxfev))
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:  # no substitute fit: the figure shows the points
        out["status"] = f"fit_failed: {type(exc).__name__}: {exc}"[:240]
        return out
    if np.all(np.isfinite(popt)):  # a parameter on its bound is the bound, not a measurement (R3)
        problems, names = _bound_problems(popt, ranges, lower, upper, float(y.sum()))
        if problems:
            out.update(status="fit_failed: " + "; ".join(problems), bound_parameters=names)
            return out
    if not (np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)) and np.all(perr > 0)):
        out["status"] = "fit_failed: non-finite parameters or errors (or a zero error: an unconstrained parameter)"
        return out
    chi2 = float(np.sum(((y - model(np.arange(x.size), *popt)) / err) ** 2))
    ndf = int(x.size - n_par)
    comps = [{"range": tuple(map(float, ranges[c])), "area": float(popt[3 * c]), "area_error": float(perr[3 * c]),
              "mu": float(popt[3 * c + 1]), "mu_error": float(perr[3 * c + 1]),
              "sigma": float(popt[3 * c + 2]), "sigma_error": float(perr[3 * c + 2]),
              "mu_sigma_cov": float(pcov[3 * c + 1, 3 * c + 2]),
              "covariance": np.asarray(pcov[3 * c:3 * c + 3, 3 * c:3 * c + 3], dtype=float).tolist()}
             for c in range(n_comp)]
    out.update(components=comps, chi2=chi2, ndf=ndf, chi2_over_ndf=chi2 / ndf if ndf > 0 else nan,
               status="ok", ok=True)
    if n_comp == 1:
        c = comps[0]
        out.update(mu=c["mu"], mu_error=c["mu_error"], sigma=c["sigma"], sigma_error=c["sigma_error"],
                   sigma_over_mu=c["sigma"] / c["mu"] if c["mu"] else nan, amplitude=c["area"],
                   mu_sigma_cov=c["mu_sigma_cov"])
    else:
        order = sorted(range(2), key=lambda c: ranges[c][0])
        for tag, c in zip(("low", "high"), order):
            comp = comps[c]
            out.update({f"mu_{tag}": comp["mu"], f"mu_{tag}_error": comp["mu_error"], f"sigma_{tag}": comp["sigma"],
                        f"sigma_{tag}_error": comp["sigma_error"], f"area_{tag}": comp["area"]})
    return out


# curve of a range fit on the page scale: density x page bin width
def range_fit_curve(fit: dict, x, per_width: float, component: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = np.zeros(x.shape)
    for c, comp in enumerate(fit.get("components") or []):
        if component is None or component == c:
            total += gauss_area(x, comp["area"], comp["mu"], comp["sigma"])
    return total * float(per_width)
