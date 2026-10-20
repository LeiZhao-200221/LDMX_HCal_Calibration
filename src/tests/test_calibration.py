"""Tests of the calib stage (ldmx_hcal.calibration.calibration_parameters), common/calibration_io.py and
common/fitting.py.  Synthetic data only (about a minute); one optional slow test.

    python -X utf8 src/tests/run_all.py calibration
    $env:LDMX_SLOW_TESTS=1     # + real pdf rendering of both pdfs (csv run == -from-saved run)

Covered:
  -only    the four units (pedestal_csv, mpv_csv, pedestal_plot, mpv_plot) and the groups pedestal / mpv / csv /
           plots / all (Lei 2026-10-20): one csv unit -> its part table (pedestal_parameters.csv /
           mpv_parameters.csv) with only its columns, both -> the full calibration_parameters.csv, plot units -> only
           their pdf; every run computes what it writes from the csv (an MPV-only run fits on its own pedestal),
           nothing is carried over from an existing table, no other file is rewritten; -no-plots drops the plot
           units ('nothing to make' when none is left)
  --output the two pdfs are the final results (pedestal_plot = pedestal_fits.pdf, mpv_plot = mpv_fits.pdf): full
           set without --output they go to cache/pedestal_mpv/unsaved/ only, the tables / provenance / histogram cache
           are written as always; names matched by result (default file name, key + extension or KEY=FILE), naming
           one saves only that one, a name of no result is ignored with a WARNING; --output alone = the usual names;
           -test without --output = results/test/test.pdf, test1.pdf; the dry-run plan lists the destinations and
           never empties the scratch folder or registers a plan; the worker temp file follows the plan
  --fit    --fit[a,b] / --bin N (Lei 2026-10-02): argparse types (one range: Landau (x) Gauss), PowerShell spelling via
           the runner, every rule refused with its reason before any IO, refused when -only chose no MPV unit; the
           default never enters the window code and a window equal to the automatic one is bit for bit the default;
           page bins or exact --bin bins, only bins above the cut, LOW_STATS with the reason, the rough fit only seeds;
           a failed fit keeps no parameters and draws no curve; -from-saved == csv run, the window in every settings
           record and in settings_hash, -from-saved -only mpv_plot draws the fitted bins
  D5       strict pedestal selection by default: raw_samples; a csv without the raw-sum columns stops the run (also an
           MPV-only run: it computes its pedestal), with the make-csv command, nothing built or written, also in test
           mode before the subset; the dry-run plan marks that csv STALE; csv_tot_toa_zero only explicitly
  D1b/G2   -from-saved: refit from the cached raw histograms with other fit settings and the csv reader disabled;
           identical to a csv run with the same settings; refused (reason + rebuild command) when the input csv, the
           pedestal selection, the accumulator version or the npz changed, the cache lacks a part, or it is gone;
           -from-saved never rewrites the cache; test mode never builds the subset under -from-saved
  D8       -clean-input on a real clean run: MPV from the clean dataset, pedestal from the beam csv; a clean dataset
           that is not current stops a csv run with the clean command, -from-saved refits on the cached histograms of
           the same file; a pedestal-only run never reads it; identities of both inputs in the provenance
  CAL2-02  sample-count check on n_samples_used (8->4 warns, 10->8 does not, 10->10 warns, mixed per file warns,
           missing = unknown), also through the automatic test subset and the clean dataset (its record -> the beam
           csv's make-csv provenance); one note per input csv and part in the provenance
  keys     load_calibration / calibration_lookup use io_utils.validate_channel_keys: duplicated, fractional,
           non-physical or empty keys refused (the row order can no longer decide a value)
  commit   table + provenance and histogram cache + record are product sets (record last): a failed rename keeps
           the old pair byte-identical and valid; the pedestal pdf drawn by a worker process has its temp file
           registered
  B6       settings validated before any IO, every problem listed; bin widths must divide the histogram range
  L5       langaus in x blocks: bit-identical to one block
  CAL-01   a final fit that does not converge never gives mpv_status OK or a finite mpv (mock, restart still failing,
           non-finite rough fit); a non-converged rough fit only seeds the final fit and is recorded
           (mpv_rough_status), or stops it with mpv_require_rough_convergence; an exception marks the stage that raised
  CAL-02   indefinite Hessian -> NaN covariance + status; positive-definite quadratic -> analytic covariance;
           bound-fixed parameters -> zero rows / columns
  CAL-03   raw_*_sum flags select pedestal events like the ROOT selection (all-zero, single non-zero, multiple
           non-zero, TOT 2048 sentinel); method columns; pedestal from histograms == direct np.mean / np.std of the
           window (both anchor methods); routing (calibration/, cache/[test/]pedestal_mpv, results/<mode>/)
  (the pdfs have no provenance record, so there is none next to a user or test name)

Author: Lei
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.calibration import calibration_parameters as cp  # noqa: E402
from ldmx_hcal.common import calibration_io as cio  # noqa: E402
from ldmx_hcal.common import fitting  # noqa: E402
from ldmx_hcal.common import geometry as geo  # noqa: E402
from ldmx_hcal.common import io_utils  # noqa: E402
from ldmx_hcal.common import outputs as outputs_mod  # noqa: E402
from ldmx_hcal.common.config import RunContext, dataset_registry  # noqa: E402

SLOW = os.environ.get("LDMX_SLOW_TESTS", "") not in ("", "0")
VALUE_COLUMNS = ["pedestal", "pedestal_per_time_sample", "pedestal_sigma", "pedestal_mu", "pedestal_range_with_5sigma",
                 "pedestal_range_with_20percent", "n_pedestal_events", "mpv", "mpv_error", "landau_width",
                 "gauss_sigma", "mpv_fit_low", "mpv_fit_high", "n_mpv_entries", "mpv_chi2_ndf", "mpv_status"]


# ----------------------------------------------------------------------------------------------
# synthetic inputs
# ----------------------------------------------------------------------------------------------
# calib settings: DEFAULTS with overrides, validated
def _settings(**over) -> dict:
    return cp.validate_settings({**cp.DEFAULTS, **over})


# One channel: pedestal 800 +- 10, a falling noise shoulder and a Landau (x) Gauss MIP peak (1-ADC bins).
def _mip_task(seed=3, n_mip=20000, mpv_landau=400.0, eta=40.0, sigma=60.0, settings=None) -> dict:
    s = settings or _settings()
    rng = np.random.default_rng(seed)
    x = np.arange(0, 2001, dtype=float)
    cfg = cp.langaus_config(s)
    expected = n_mip * fitting.langaus(x, mpv_landau, eta, 1.0, sigma, n_steps=cp.langaus_steps(eta, sigma, cfg),
                                       n_sigma=cfg[1])
    expected += 3000.0 * np.exp(-x / 25.0)
    counts = rng.poisson(expected).astype(np.int64)
    return {"channel": 0, "pedestal": 800.0, "pedestal_sigma": 10.0, "v0": 800, "counts": counts,
            "settings": cp.fit_settings(s)}


# true Landau (x) Gauss peak position of the default langaus settings
def _true_peak(mpv_landau=400.0, eta=40.0, sigma=60.0) -> float:
    s = _settings()
    cfg = cp.langaus_config(s)
    return cp.langaus_peak(mpv_landau, eta, sigma, (cp.langaus_steps(eta, sigma, cfg), cfg[1], 0.0))


MIP_BARS = ((5, 3), (12, 7))


# beam csv rows: Gaussian pedestals on all 192 bars, MIP hits in mip_bars (70 % of events)
def _beam_frame(n_events=1500, shift=0, raw_flags=True, multi_hit_fraction=0.0, seed=1,
                mip_bars=MIP_BARS) -> pd.DataFrame:
    # Beam csv rows (all 192 bars per event): Gaussian pedestals everywhere, MIPs in ``mip_bars`` (70 % of events).
    #
    # ``multi_hit_fraction`` of the non-MIP rows get a signal of +60 ADC with the calibrated csv tot/toa = 0 but
    # a non-zero raw TOA sum (what tot_calib / toa_calib do to several non-zero samples).
    rng = np.random.default_rng(seed)
    bars = geo.all_bars()
    nb = len(bars)
    layer = np.tile(bars["layer"].to_numpy(), n_events)
    strip = np.tile(bars["strip"].to_numpy(), n_events)
    event = np.repeat(np.arange(1, n_events + 1), nb)
    n = len(event)
    ped0 = 700.0 + 3 * layer + strip + shift
    ped1 = ped0 + 40.0
    adc0 = rng.normal(ped0, 10.0)
    adc1 = rng.normal(ped1, 12.0)
    tot0, tot1 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    toa0, toa1 = np.zeros(n), np.zeros(n)
    raw_tot0, raw_tot1 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    raw_toa0, raw_toa1 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    is_mip_bar = np.zeros(n, bool)
    for lay, st in mip_bars:
        is_mip_bar |= (layer == lay) & (strip == st)
    signal = is_mip_bar & (rng.random(n) < 0.7)
    from scipy.stats import moyal

    n_signal = int(signal.sum())
    e = moyal.rvs(loc=400.0, scale=40.0, size=n_signal, random_state=rng) + rng.normal(0, 50.0, n_signal)
    adc0[signal] += e
    adc1[signal] += 1.1 * e
    toa0[signal], toa1[signal] = 12.5, 13.0
    raw_toa0[signal], raw_toa1[signal] = 412, 420
    multi = ~is_mip_bar & (rng.random(n) < multi_hit_fraction)
    adc0[multi] += 60.0
    adc1[multi] += 60.0
    raw_toa0[multi], raw_toa1[multi] = 300, 280       # csv toa stays 0: several non-zero samples
    df = pd.DataFrame({"pf_event": event, "tot_end0": tot0, "toa_end0": toa0,
                       "adc_sum_end0": np.clip(np.round(adc0), 0, 8184).astype(np.int64),
                       "layer": layer, "strip": strip, "tot_end1": tot1, "toa_end1": toa1,
                       "adc_sum_end1": np.clip(np.round(adc1), 0, 8184).astype(np.int64)})
    if raw_flags:
        df["raw_tot_sum_end0"], df["raw_tot_sum_end1"] = raw_tot0, raw_tot1
        df["raw_toa_sum_end0"], df["raw_toa_sum_end1"] = raw_toa0, raw_toa1
    return df


# Write the rows to ``path``; its folder is the run folder of _ctx (every csv in it is a file of the run).
def _beam_csv(path: Path, n_events=1500, shift=0, raw_flags=True, multi_hit_fraction=0.0, seed=1,
              mip_bars=MIP_BARS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _beam_frame(n_events, shift, raw_flags, multi_hit_fraction, seed, mip_bars).to_csv(path, index=False)
    return path


# Run muon 4 = the folder of ``csv``; ``outputs``: the --output names (None = not given; [] = --output alone).
def _ctx(tmp: Path, csv: Path, test=False, make_plots=False, section=None, workers=1, outputs=None) -> RunContext:
    cfg = {"paths": {"generated_data_dir": tmp / "data", "calibration_dir": tmp / "calibration",
                     "results_dir": tmp / "results", "cache_dir": tmp / "cache"},
           "run": {"chunksize": 50_000}, "safety": {"protected_dirs": []},
           "datasets": {"muon_4GeV": {"particle": "muon", "energy_gev": 4, "label": "synthetic muons", "kind": "data",
                                      "folder": str(Path(csv).parent)}},
           "calibration": dict(section or {})}
    return RunContext(cfg=cfg, test=test, make_plots=make_plots, workers=workers, outputs=outputs)


# calib arguments parsed as the runner would (plus the common -from-saved)
def _args(*argv, from_saved=False):
    p = argparse.ArgumentParser()
    cp.add_arguments(p)
    args = p.parse_args(list(argv))
    args.from_saved = from_saved          # common runner option (runner.add_common)
    return args


# What the runner does before io_plan / run: ctx.units = the -only units of the command.
def _units(ctx, args):
    ctx.units = outputs_mod.selected_units(args)
    return args


# cp.run; returns the table the run writes (None for plot units only).
def _run(ctx, *argv, from_saved=False):
    cp.run(ctx, _units(ctx, _args(*argv, from_saved=from_saved)), [])
    return cp.table_csv(ctx)


# dry-run plan of calib for these arguments
def _plan(ctx, *argv, datasets=(), from_saved=False):
    return cp.io_plan(ctx, _units(ctx, _args(*argv, from_saved=from_saved)), list(datasets))


# skip the current test with a reason
def _skip(reason):
    raise unittest.SkipTest(reason)


# require fn to raise exc_type whose text contains every word of contains
def _raises(exc_type, fn, *a, contains=(), **k):
    try:
        fn(*a, **k)
    except exc_type as exc:
        for text in ([contains] if isinstance(contains, str) else contains):
            assert text in str(exc), (text, str(exc))
        return exc
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc_type.__name__}")


# stand-in for the csv reader: fails if the beam csv is read
def _no_csv_reading(*a, **k):
    raise AssertionError("the beam csv must not be read")


# fit stand-in that returns a non-converged result with the given parameters
def _failed(params=(500.0, 50.0, 1000.0, 40.0), message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"):
    # the non-converged fit result
    def fit(*args, **kwargs):
        return fitting.FitResult(False, np.array(params, float), np.ones(4), chi2=10.0, ndf=10, n_points=14,
                                 message=message)
    return fit


# ----------------------------------------------------------------------------------------------
# CAL-01: convergence of the MPV fits
# ----------------------------------------------------------------------------------------------
# Codex reproduction: both fitters return ok=False with finite parameters -> never OK / finite MPV.
def test_cal01_codex_mock_final_fit_not_converged_is_fit_failed():
    s = {**cp.DEFAULTS, "mpv_rough_start": "threshold"}
    task = dict(channel=0, pedestal=800.0, pedestal_sigma=10.0, v0=800, counts=np.ones(2001, dtype=np.int64),
                settings=s)
    # the covariance helpers are patched to "valid" too, so the test pins CAL-01 even if the covariance were
    # computed for a non-converged fit (the stage calls deviance_covariance_with_status)
    with patch.object(cp, "fit_neyman", _failed()), patch.object(cp, "fit_poisson_scaled", _failed()), \
            patch.object(cp, "deviance_covariance", return_value=np.eye(4)), \
            patch.object(cp, "deviance_covariance_with_status", return_value=(np.eye(4), "OK")) as cov:
        r = cp.fit_channel_mpv(task)
    assert cov.call_count == 0                                        # stopped before the covariance
    assert r["mpv_status"] == "FIT_FAILED", r["mpv_status"]
    assert np.isnan(r["mpv"]) and np.isnan(r["mpv_error"])
    assert r["mpv_rough_status"] == "NOT_CONVERGED" and r["mpv_fit_status"] == "NOT_CONVERGED"
    assert "ITERATIONS REACHED LIMIT" in r["mpv_fit_message"]
    _assert_no_fit_values(r)        # R3 (Lei 2026-10-02): the parameters of a failed fit are NaN, not diagnostics


# a channel without an MPV keeps no fitted values (cp._clear_failed_fit)
def _assert_no_fit_values(r: dict) -> None:
    # R3: a channel without an MPV keeps no fitted values (``cp._clear_failed_fit``): NaN parameter / chi2
    # columns, params None (no curve), while the status and message columns stay.
    assert r["mpv_status"] != "OK" and np.isnan(r["mpv"]) and np.isnan(r["mpv_error"]), r["mpv_status"]
    assert all(np.isnan(r[name]) for name in cp.FIT_VALUE_COLUMNS), {n: r[n] for n in cp.FIT_VALUE_COLUMNS}
    assert r["params"] is None and r["mpv_fit_status"] and r["mpv_status"]


# The final fit is restarted once; if the restart does not converge either -> FIT_FAILED, NaN.
def test_cal01_restart_still_not_converged_is_fit_failed():
    task = _mip_task()
    real = fitting.fit_binned_poisson
    calls = []

    # real fit, then marked as not converged
    def never_converges(*args, **kwargs):
        res = real(*args, **kwargs)
        calls.append(1)
        res.ok, res.message = False, "ABNORMAL_TERMINATION_IN_LNSRCH"
        return res

    with patch.object(cp.fitting, "fit_binned_poisson", never_converges):
        r = cp.fit_channel_mpv(task)
    assert len(calls) == 2, calls                                    # first attempt + one restart
    assert r["mpv_status"] == "FIT_FAILED" and np.isnan(r["mpv"]) and np.isnan(r["mpv_error"])
    assert r["mpv_fit_status"] == "NOT_CONVERGED" and "ABNORMAL" in r["mpv_fit_message"]
    assert "restart" in r["mpv_fit_message"]
    assert r["mpv_covariance_status"] == "NOT_COMPUTED"
    _assert_no_fit_values(r)


# cal01 rough not converged seeds final unless required
def test_cal01_rough_not_converged_seeds_final_unless_required():
    # Decision: a non-converged rough fit with finite parameters seeds the final fit (recorded);
    # mpv_require_rough_convergence: true turns it into ROUGH_FIT_FAILED.
    real = cp.fit_neyman

    # real rough fit, then marked as not converged
    def rough_not_converged(*args, **kwargs):
        res = real(*args, **kwargs)
        res.ok, res.message = False, "maxfev reached"
        return res

    with patch.object(cp, "fit_neyman", rough_not_converged):
        seeded = cp.fit_channel_mpv(_mip_task())
        strict = cp.fit_channel_mpv(_mip_task(settings=_settings(mpv_require_rough_convergence=True)))
    plain = cp.fit_channel_mpv(_mip_task())
    assert seeded["mpv_rough_status"] == "NOT_CONVERGED" and seeded["mpv_status"] == "OK"
    assert seeded["mpv"] == plain["mpv"]                                 # same seed values -> same final fit
    assert strict["mpv_status"] == "ROUGH_FIT_FAILED" and np.isnan(strict["mpv"]) and np.isnan(strict["mpv_error"])
    assert "maxfev" in strict["mpv_fit_message"]


# cal01 rough fit failed nonfinite
def test_cal01_rough_fit_failed_nonfinite():
    nan = _failed(params=(np.nan,) * 4, message="Optimal parameters not found")
    with patch.object(cp, "fit_neyman", nan):
        r = cp.fit_channel_mpv(_mip_task())
    assert r["mpv_status"] == "ROUGH_FIT_FAILED" and r["mpv_rough_status"] == "FAILED"
    assert np.isnan(r["mpv"]) and np.isnan(r["mpv_error"]) and r["mpv_fit_status"] == "NOT_RUN"


# An exception never leaves contradicting QA columns: the stage that raised is FAILED, mpv / error NaN.
def test_cal01_exception_marks_the_stage_that_raised():
    # fit stand-in that always raises
    def boom(*a, **k):
        raise ValueError("synthetic")

    with patch.object(cp, "fit_neyman", boom):                       # rough stage
        r = cp.fit_channel_mpv(_mip_task())
    assert r["mpv_status"] == "ROUGH_FIT_FAILED" and r["mpv_rough_status"] == "FAILED"
    assert r["mpv_fit_status"] == "NOT_RUN" and r["mpv_fit_message"] == "exception ValueError: synthetic"
    assert np.isnan(r["mpv"]) and np.isnan(r["mpv_error"])

    # verifier case: the final optimizer converged, then the peak search raised -> FIT_FAILED + FAILED, and the
    # optimizer message is kept in front of the exception text
    with patch.object(cp, "langaus_peak", boom):
        r = cp.fit_channel_mpv(_mip_task())
    assert r["mpv_status"] == "FIT_FAILED" and r["mpv_fit_status"] == "FAILED", r
    assert r["mpv_rough_status"] == "CONVERGED" and r["mpv_covariance_status"] == "NOT_COMPUTED"
    message = r["mpv_fit_message"]
    tail = "; exception ValueError: synthetic"
    assert message.endswith(tail) and len(message) > len(tail)
    assert np.isnan(r["mpv"]) and np.isnan(r["mpv_error"])
    _assert_no_fit_values(r)        # the converged parameters were written before the raise: cleared (R3)

    # an exception after an OK covariance (in the Jacobian of the peak) -> FIT_FAILED as well
    real, calls = cp.langaus_peak, []

    # real fit on the first call, raises on every later call
    def peak_then_boom(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("jacobian")
        return real(*a, **k)

    with patch.object(cp, "langaus_peak", peak_then_boom):
        r = cp.fit_channel_mpv(_mip_task())
    assert r["mpv_status"] == "FIT_FAILED" and r["mpv_fit_status"] == "FAILED" and np.isnan(r["mpv"])
    assert r["mpv_covariance_status"] == "OK" and "ValueError: jacobian" in r["mpv_fit_message"]
    _assert_no_fit_values(r)


# mpv fit success and qa columns
def test_mpv_fit_success_and_qa_columns():
    task = _mip_task()
    r = cp.fit_channel_mpv(task)
    assert r["mpv_status"] == "OK", (r["mpv_status"], r["mpv_fit_message"])
    assert r["mpv_rough_status"] == "CONVERGED" and r["mpv_fit_status"] == "CONVERGED"
    assert r["mpv_covariance_status"] == "OK" and np.isfinite(r["mpv_error"]) and r["mpv_error"] > 0
    assert abs(r["mpv"] - _true_peak()) < 5 * r["mpv_error"] + 2.0, (r["mpv"], _true_peak(), r["mpv_error"])
    assert r["mpv_fit_low"] <= r["mpv"] <= r["mpv_fit_high"]
    n_bins = int(round((r["mpv_fit_high"] - r["mpv_fit_low"]) / 20.0))
    assert r["mpv_ndf"] == n_bins - 4 and np.isclose(r["mpv_chi2_ndf"], r["mpv_deviance"] / r["mpv_ndf"])
    assert 0 < r["n_mpv_fit_entries"] <= r["n_mpv_entries"]
    assert r["mpv_threshold"] == 50.0 and r["mpv_rough_floor"] >= 50.0
    assert r["mpv_pedestal"] == 800.0 and r["mpv_pedestal_sigma"] == 10.0 and r["mpv_method"] == "two_stage_langaus"
    assert np.allclose(r["params"], [r["landau_mpv_param"], r["landau_width"], r["langaus_area"], r["gauss_sigma"]])


# PEAK_OUTSIDE_WINDOW: mpv and mpv_error both NaN.
def test_peak_outside_window_has_no_mpv_error():
    with patch.object(cp, "langaus_peak", return_value=-1.0e6):
        r = cp.fit_channel_mpv(_mip_task())
    assert r["mpv_status"] == "PEAK_OUTSIDE_WINDOW" and np.isnan(r["mpv"]) and np.isnan(r["mpv_error"])
    _assert_no_fit_values(r)
    assert r["mpv_fit_status"] == "CONVERGED" and np.isfinite(r["mpv_fit_low"]) and r["window"] is not None


# low stats and no pedestal
def test_low_stats_and_no_pedestal():
    t = _mip_task(n_mip=10)
    t["counts"] = np.zeros_like(t["counts"])
    t["counts"][100:150] = 1
    assert cp.fit_channel_mpv(t)["mpv_status"] == "LOW_STATS"
    t2 = dict(t, pedestal=np.nan)
    r = cp.fit_channel_mpv(t2)
    assert r["mpv_status"] == "NO_PEDESTAL" and r["final_counts"] is None


# ----------------------------------------------------------------------------------------------
# CAL-02: covariance validity
# ----------------------------------------------------------------------------------------------
# deviance stand-in factory with a chosen Hessian h
def _quadratic(h):
    # quadratic deviance p^T h p with a chosen Hessian h
    def deviance(model, centers, counts, p):
        return float(p @ h @ p)
    return deviance


# cal02 codex indefinite covariance is nan
def test_cal02_codex_indefinite_covariance_is_nan():
    cov_target = np.array([[1., 2., 0., 0.], [2., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    with patch.object(cp, "poisson_deviance", _quadratic(np.linalg.inv(cov_target))):
        cov, status = cp.deviance_covariance_with_status(None, None, None, np.zeros(4))
        cov_only = cp.deviance_covariance(None, None, None, np.zeros(4))
    assert status == "NOT_POSITIVE_DEFINITE", status
    assert np.all(np.isnan(cov)) and np.all(np.isnan(cov_only))


# Positive diagonal, negative eigenvalue: only the Cholesky / eigenvalue test catches it.
def test_cal02_positive_diagonal_but_indefinite_hessian():
    h = np.array([[1., 2., 0., 0.], [2., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    with patch.object(cp, "poisson_deviance", _quadratic(h)):
        cov, status = cp.deviance_covariance_with_status(None, None, None, np.zeros(4))
    assert status == "NOT_POSITIVE_DEFINITE" and np.all(np.isnan(cov))


# cal02 positive definite quadratic recovers covariance
def test_cal02_positive_definite_quadratic_recovers_covariance():
    cov_target = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.3, 0.0], [0.0, 0.3, 1.0, 0.5], [0.0, 0.0, 0.5, 3.0]])
    p = np.array([1.0, 2.0, 3.0, 4.0])

    # quadratic deviance around p: d^T inv(cov_target) d
    def deviance(model, centers, counts, q):
        d = q - p
        return float(d @ np.linalg.inv(cov_target) @ d)     # 0.5 * Hessian = inv(cov_target)

    with patch.object(cp, "poisson_deviance", deviance):
        cov, status = cp.deviance_covariance_with_status(None, None, None, p)
        fixed, status_fixed = cp.deviance_covariance_with_status(None, None, None, p,
                                                                 free=np.array([True, True, False, True]))
    assert status == "OK" and np.allclose(cov, cov_target, rtol=1e-5, atol=1e-7), cov
    assert status_fixed == "OK"
    assert np.all(fixed[2, :] == 0) and np.all(fixed[:, 2] == 0)
    idx = [0, 1, 3]
    expected = np.linalg.inv(np.linalg.inv(cov_target)[np.ix_(idx, idx)])   # conditional on the fixed parameter
    assert np.allclose(fixed[np.ix_(idx, idx)], expected, rtol=1e-5, atol=1e-7)


# positive definite status cases
def test_positive_definite_status_cases():
    assert fitting.positive_definite_status(np.diag([1.0, 1e6, 1e-6])) == "OK"          # scale invariant
    assert fitting.positive_definite_status([[1.0, 2.0], [2.0, 1.0]]) == "NOT_POSITIVE_DEFINITE"
    assert fitting.positive_definite_status([[1.0, 1.0], [1.0, 1.0]]) in ("SINGULAR", "NOT_POSITIVE_DEFINITE")
    assert fitting.positive_definite_status([[np.nan, 0.0], [0.0, 1.0]]) == "NOT_FINITE"
    assert fitting.positive_definite_status([[-1.0, 0.0], [0.0, 1.0]]) == "NOT_POSITIVE_DEFINITE"
    assert fitting.positive_definite_status(np.zeros((0, 0))) == "EMPTY"


# hessian errors nan for indefinite
def test_hessian_errors_nan_for_indefinite():
    h = np.array([[1.0, 2.0], [2.0, 1.0]])
    errs = fitting._hessian_errors(lambda q: float(q @ h @ q), np.zeros(2))
    assert np.all(np.isnan(errs))
    good = fitting._hessian_errors(lambda q: float(q @ np.diag([4.0, 1.0]) @ q), np.zeros(2), scale=0.5)
    assert np.allclose(good, [0.5, 1.0], rtol=1e-6)


# fit binned poisson compute errors flag gives identical parameters
def test_fit_binned_poisson_compute_errors_flag_gives_identical_parameters():
    rng = np.random.default_rng(5)
    centers = np.arange(-50, 51, 5.0)
    counts = rng.poisson(fitting.gauss(centers, 200.0, 3.0, 15.0)).astype(float)

    # Gaussian model of the binned Poisson fit
    def model(c, a, m, s):
        return fitting.gauss(c, a, m, s)

    p0, bounds = [150.0, 0.0, 10.0], ([1.0, -40.0, 1.0], [1e4, 40.0, 100.0])
    a = fitting.fit_binned_poisson(model, centers, counts, p0, bounds)
    b = fitting.fit_binned_poisson(model, centers, counts, p0, bounds, compute_errors=False)
    assert a.ok == b.ok and np.array_equal(a.params, b.params) and a.chi2 == b.chi2 and a.message == b.message
    assert np.all(np.isfinite(a.errors)) and np.all(np.isnan(b.errors))


# ----------------------------------------------------------------------------------------------
# n_steps, L5 blocks, settings validation (B6)
# ----------------------------------------------------------------------------------------------
# langaus steps always even and langaus rejects odd
def test_langaus_steps_always_even_and_langaus_rejects_odd():
    cfg = (100, 8.0, 0.5)
    for eta, sigma in ((5.0, 1.0), (5.0, 300.0), (40.0, 60.0), (1e-3, 1000.0), (100.0, 0.5)):
        n = cp.langaus_steps(eta, sigma, cfg)
        assert n % 2 == 0 and 100 <= n <= cp.LANGAUS_MAX_STEPS, (eta, sigma, n)
    assert cp.langaus_steps(5.0, 300.0, (101, 8.0, 0.0)) % 2 == 0
    for bad in (0, 1, 99, -4, 2.5):
        try:
            fitting.langaus(np.array([1.0]), 10.0, 2.0, 1.0, 3.0, n_steps=bad)
        except ValueError:
            continue
        raise AssertionError(f"langaus accepted n_steps={bad}")
    assert np.isfinite(fitting.langaus(np.array([1.0]), 10.0, 2.0, 1.0, 3.0, n_steps=100)).all()


# langaus blocks are bit identical
def test_langaus_blocks_are_bit_identical():
    # Codex L5: the x points are convolved in blocks; every block uses all nodes in the same order, so the curve
    # is bit-identical to a single block (also at block borders, for 1 point and for an empty x).
    rng = np.random.default_rng(7)
    for n_x in (0, 1, 7, 99, 351, 2001):
        for n_steps in (100, 384, 4000):
            x = np.sort(rng.uniform(-200.0, 2500.0, n_x))
            p = (rng.uniform(100, 900), rng.uniform(5, 200), rng.uniform(1, 1e5), rng.uniform(1, 600))
            with patch.object(fitting, "LANGAUS_BLOCK_ELEMENTS", 10 ** 12):        # one block
                whole = fitting.langaus(x, *p, n_steps=n_steps, n_sigma=8.0)
            for block in (1, 1000, 4096, fitting.LANGAUS_BLOCK_ELEMENTS):
                with patch.object(fitting, "LANGAUS_BLOCK_ELEMENTS", block):
                    part = fitting.langaus(x, *p, n_steps=n_steps, n_sigma=8.0)
                    assert np.array_equal(part, whole), (n_x, n_steps, block)


# settings validation rejects bad values
def test_settings_validation_rejects_bad_values():
    bad = [{"langaus_n_steps": 101}, {"langaus_n_steps": 0}, {"mpv_final_bin_width": 0},
           {"mpv_rough_bin_width": -40}, {"mpv_hist_range": [2000, 0]}, {"mpv_hist_range": [0]},
           {"mpv_window_widths": 0}, {"pedestal_window_halfwidth": -1}, {"mpv_eta_range": [0, 10]},
           {"mpv_sigma_range": [5, 1]}, {"mpv_min_bins": 4}, {"mpv_nsigma": float("nan")},
           {"mpv_final_bin_width": 5000}, {"mpv_rough_likelihood": "chi2"}, {"pedestal_selection": "raw"},
           {"mpv_require_rough_convergence": "maybe"},
           # yml values that would otherwise escape as a traceback (ValueError / TypeError / OverflowError)
           {"mpv_min_entries": "abc"}, {"langaus_n_steps": None}, {"langaus_n_steps": float("inf")},
           {"mpv_min_bins": float("nan")}, {"mpv_min_entries": 10 ** 400}, {"mpv_min_bins": True},
           {"mpv_min_entries": 2.5}, {"pedestal_window_halfwidth": 10 ** 400}, {"mpv_eta_range": [1, 10 ** 400]},
           {"mpv_nsigma": None},
           # no silent fall-back selection; bins must tile the histogram range
           {"pedestal_selection": "auto"}, {"mpv_final_bin_width": 30}, {"mpv_rough_bin_width": 60},
           {"pedestal_anchor": "mode"}]
    for over in bad:
        exc = _raises(SystemExit, _settings, **over)
        assert isinstance(exc, io_utils.SettingsError) and isinstance(exc, ValueError), type(exc)
        assert list(over)[0] in str(exc), (over, exc)
    assert "expected one of" in str(_raises(SystemExit, _settings, pedestal_selection="auto"))
    assert "whole bins" in str(_raises(SystemExit, _settings, mpv_final_bin_width=30))
    every = str(_raises(SystemExit, _settings, mpv_nsigma=-1, langaus_n_steps=3, mpv_window_widths="x"))
    assert all(k in every for k in ("mpv_nsigma", "langaus_n_steps", "mpv_window_widths"))   # all listed at once
    assert _settings(mpv_min_entries="100", langaus_n_steps=200.0)["langaus_n_steps"] == 200.0   # numeric text ok
    s = _settings(mpv_window_above_valley="false", write_histogram_cache="yes", pedestal_selection="CSV_TOT_TOA_ZERO")
    assert s["mpv_window_above_valley"] is False and s["write_histogram_cache"] is True
    assert s["pedestal_selection"] == "csv_tot_toa_zero"
    assert _settings(mpv_final_bin_width=25, mpv_hist_range=[0, 2000])["mpv_final_bin_width"] == 25   # 80 bins
    assert _settings()["pedestal_selection"] == "raw_samples"   # defaults are valid; strict selection by default


# ----------------------------------------------------------------------------------------------
# pedestal: D5 strict selection, CAL-03 selection, anchor methods, direct equivalence
# ----------------------------------------------------------------------------------------------
# direct pedestal: mean and std of the adc_sum within anchor +- halfwidth
def _direct_pedestal(adc_sum: np.ndarray, anchor: float, halfwidth=200.0):
    x = adc_sum - anchor
    w = x[np.abs(x) < halfwidth]
    return anchor + w.mean(), w.std(ddof=0), len(w)


# cal03 raw flags select like the root code
def test_cal03_raw_flags_select_like_the_root_code():
    # Four event classes of one bar: raw all-zero, one non-zero TOA sample (csv toa > 0), several non-zero
    # TOA samples (csv toa = 0, raw > 0), TOT = 2048 sentinel (csv tot = 0, raw = 2048).
    rng = np.random.default_rng(11)
    n = {"clean": 3000, "single": 400, "multi": 500, "sentinel": 300}
    rows = []
    for cls, k in n.items():
        base = rng.normal(760.0, 9.0, k) + {"clean": 0, "single": 300, "multi": 120, "sentinel": 150}[cls]
        csv_toa = 20.0 if cls == "single" else 0.0
        raw_toa = {"clean": 0, "single": 37, "multi": 81, "sentinel": 0}[cls]
        raw_tot = 2048 if cls == "sentinel" else 0
        rows.append(pd.DataFrame({"pf_event": np.arange(k), "layer": 3, "strip": 2, "tot_end0": 0, "tot_end1": 0,
                                  "toa_end0": csv_toa, "toa_end1": 0.0,
                                  "adc_sum_end0": np.round(base).astype(int),
                                  "adc_sum_end1": np.round(base).astype(int),
                                  "raw_tot_sum_end0": raw_tot, "raw_tot_sum_end1": 0,
                                  "raw_toa_sum_end0": raw_toa, "raw_toa_sum_end1": raw_toa, "cls": cls}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "beam.csv"
        df.drop(columns="cls").to_csv(path, index=False)
        c = int(cp.channel_index_table()[3, 2, 0])
        results = {}
        for selection in (None, "raw_samples", "csv_tot_toa_zero"):
            kw = {} if selection is None else {"selection": selection}
            h = cp.accumulate_histograms(path, True, False, chunksize=777, **kw)
            ped = cp.pedestals_from_histograms(h["h_selected"], h["h_sample"], _settings())
            results[selection] = (h["pedestal_selection"], ped.iloc[c])
        csv_only = Path(tmp) / "beam_no_flags.csv"
        df.drop(columns=["cls"] + cp.RAW_FLAG_COLUMNS).to_csv(csv_only, index=False)
        h_plain = cp.accumulate_histograms(csv_only, True, False, selection="csv_tot_toa_zero")
        # D5: the default (strict) selection on a csv without the raw sums stops, it never falls back
        _raises(SystemExit, cp.accumulate_histograms, csv_only, True, False,
                contains=("raw_tot_sum_end0", "make-csv", "-pedestal-selection csv_tot_toa_zero"))
        _raises(SystemExit, cp.accumulate_histograms, csv_only, True, False, selection="auto", contains="auto")
    assert results[None][0] == "raw_samples" and h_plain["pedestal_selection"] == "csv_tot_toa_zero"
    for selection, classes in (("raw_samples", ["clean"]), ("csv_tot_toa_zero", ["clean", "multi", "sentinel"])):
        label, row = results[selection]
        assert label == selection
        sel = df.loc[df["cls"].isin(classes), "adc_sum_end0"].to_numpy(float)
        anchor = 8 * np.round(np.median(sel) / 8)
        p, s, k = _direct_pedestal(sel, anchor)
        assert row["n_selected_events"] == len(sel), (selection, row["n_selected_events"], len(sel))
        assert row["pedestal_anchor"] == anchor and row["n_pedestal_events"] == k
        assert abs(row["pedestal"] - p) < 1e-9 and abs(row["pedestal_sigma"] - s) < 1e-9
        assert row["pedestal_anchor_method"] == "median_sum"
    assert results[None][1]["pedestal"] == results["raw_samples"][1]["pedestal"]
    assert results["csv_tot_toa_zero"][1]["pedestal"] > results["raw_samples"][1]["pedestal"] + 1.0

# d5 strict selection stops the stage without raw columns
def test_d5_strict_selection_stops_the_stage_without_raw_columns():
    # Lei D5: default pedestal_selection raw_samples; a beam csv without the raw sums stops the stage before
    # anything is read, built or written, with the make-csv command (full set and test mode: the test subset is
    # not built first, its source header decides), also an MPV-only run (it computes its own pedestal); the dry-run
    # plan marks that csv STALE with the same command; the approximation runs only when asked for.
    assert cp.DEFAULTS["pedestal_selection"] == "raw_samples"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "old" / "old.csv", n_events=200, raw_flags=False)
        for test, hint in ((False, ".\\runner.ps1 make-csv --muon 4 -overwrite"),
                           (True, ".\\runner.ps1 make-csv --muon 4 -test -overwrite")):
            ctx = _ctx(tmp / f"strict_{test}", csv, test=test)
            for from_saved in (False, True):
                for argv in ((), ("-only", "mpv_csv")):
                    with patch.object(io_utils, "iter_csv", _no_csv_reading), \
                            patch.object(io_utils, "build_test_subset", _no_csv_reading):
                        _raises(SystemExit, _run, ctx, *argv, from_saved=from_saved,
                                contains=(hint, "-pedestal-selection csv_tot_toa_zero", "raw_toa_sum_end1"))
            assert not (tmp / f"strict_{test}").exists()                 # nothing built or written
            for argv in ((), ("-only", "mpv_csv")):
                beam = [i for i in _plan(ctx, *argv) if i["role"] == "in"][0]
                assert beam["stale"] is True and hint in beam["produced_by"], beam
                assert "raw_tot_sum_end0" in beam["produced_by"] and beam["kind"] == ("product" if test else "external")
            ok = [i for i in _plan(ctx, "--pedestal-selection", "csv_tot_toa_zero") if i["role"] == "in"]
            assert not ok[0].get("stale") and ok[0]["kind"] == ("auto" if test else "external")
            assert not (tmp / f"strict_{test}").exists()                 # plans never build anything
        ctx = _ctx(tmp / "approx", csv)
        table = cio.load_calibration(_run(ctx, "--pedestal-selection", "csv_tot_toa_zero"))
        assert (table["pedestal_selection"] == "csv_tot_toa_zero").all()
        # a config that says 'auto' is refused with the allowed values
        _raises(SystemExit, _run, _ctx(tmp / "auto", csv, section={"pedestal_selection": "auto"}),
                contains="expected one of")


# Per-sample columns -> anchor 8 x mode (smallest value on ties, scipy.stats.mode), pedestal_mu, fold.
def test_pedestal_sample_mode_anchor_and_direct_equivalence():
    rng = np.random.default_rng(2)
    k = 2500
    samples = rng.integers(93, 99, size=(k, 8))
    samples[:, 0] = np.where(rng.random(k) < 0.5, 95, 97)            # 95 and 97 tie-break material
    df = pd.DataFrame({"pf_event": np.arange(k), "layer": 10, "strip": 11, "tot_end0": 0, "tot_end1": 0,
                       "toa_end0": 0.0, "toa_end1": 0.0})
    for c in cp.RAW_FLAG_COLUMNS:
        df[c] = 0
    for e in (0, 1):
        for i in range(8):
            df[f"adc_{i}_end{e}"] = samples[:, i] + 8 * e
        df[f"adc_sum_end{e}"] = df[[f"adc_{i}_end{e}" for i in range(8)]].sum(axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wave.csv"
        df.to_csv(path, index=False)
        h = cp.accumulate_histograms(path, True, False, chunksize=999)
        ped = cp.pedestals_from_histograms(h["h_selected"], h["h_sample"], _settings())
        h_med = cp.accumulate_histograms(path, True, False, anchor="median_sum")
        ped_med = cp.pedestals_from_histograms(h_med["h_selected"], h_med["h_sample"], _settings())
    assert (h["pedestal_selection"], h["pedestal_anchor_method"]) == ("raw_samples", "sample_mode")   # automatic anchor
    for e in (0, 1):
        c = int(cp.channel_index_table()[10, 11, e])
        values = samples.ravel() + 8 * e
        mode = int(np.bincount(values).argmax())
        row = ped.iloc[c]
        assert row["pedestal_anchor_method"] == "sample_mode" and row["pedestal_per_time_sample"] == mode
        p, s, n = _direct_pedestal(df[f"adc_sum_end{e}"].to_numpy(float), 8.0 * mode)
        assert row["pedestal_anchor"] == 8 * mode and abs(row["pedestal"] - p) < 1e-9
        assert abs(row["pedestal_sigma"] - s) < 1e-9 and row["n_pedestal_events"] == n
        assert abs(row["pedestal"] - (row["pedestal_anchor"] + row["pedestal_mu"])) < 1e-9
        rm = ped_med.iloc[c]
        assert rm["pedestal_anchor_method"] == "median_sum"
        assert abs(rm["pedestal_per_time_sample"] - rm["pedestal"] / 8) < 1e-12
        assert rm["pedestal_range_with_20percent"] == 1.2 * rm["pedestal"]
        assert rm["pedestal_range_with_5sigma"] == rm["pedestal"] + 5.0 * rm["pedestal_sigma"]


# The histograms count integer ADC values: a non-integer adc_sum stops the run (never truncated).
def test_accumulate_refuses_non_integer_adc_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "float.csv"
        df = _beam_frame(n_events=3)
        df["adc_sum_end1"] = df["adc_sum_end1"].astype(float)
        df.loc[5, "adc_sum_end1"] = 760.5
        df.to_csv(path, index=False)
        _raises(io_utils.InputError, cp.accumulate_histograms, path, True, True, contains=("adc_sum_end1", "integer"))
        df = _beam_frame(n_events=3)
        df.loc[5, "layer"] = 257                                       # keys: the shared iter_csv rule
        df.to_csv(path, index=False)
        _raises(io_utils.ChannelKeyError, cp.accumulate_histograms, path, True, True)

# ----------------------------------------------------------------------------------------------
# calibration_io: schema, keys (shared rule), values, lookups
# ----------------------------------------------------------------------------------------------
# schema column order parts and descriptions
def test_schema_column_order_parts_and_descriptions():
    cols = list(cio.CALIBRATION_COLUMNS)
    assert cols[:len(VALUE_COLUMNS)] == VALUE_COLUMNS
    desc = cio.CALIBRATION_COLUMNS["pedestal_mu"]
    assert "pedestal_anchor" in desc and "8*anchor" not in desc     # consistent definition
    for c in ("pedestal_selection", "pedestal_anchor_method", "mpv_rough_status", "mpv_covariance_status",
              "mpv_active_bounds", "n_mpv_fit_entries", "mpv_deviance", "mpv_ndf", "mpv_pedestal"):
        assert c in cols
    assert "pedestalsigma" in cio.CALIBRATION_COLUMNS["pedestal_range_with_5sigma"]
    assert "pedestalscale" in cio.CALIBRATION_COLUMNS["pedestal_range_with_20percent"]
    # the two part tables split the columns: every column in exactly one part
    assert set(cp.PEDESTAL_COLUMNS) | set(cp.MPV_COLUMNS) == set(cols)
    assert not set(cp.PEDESTAL_COLUMNS) & set(cp.MPV_COLUMNS)


# write a calibration table to path
def _write_table(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


# a valid calibration table: pedestal 800 +- 10, mpv 600, status OK
def _good_table() -> pd.DataFrame:
    good = cio.empty_calibration()
    good["pedestal"], good["pedestal_sigma"], good["mpv"], good["mpv_status"] = 800.0, 10.0, 600.0, "OK"
    good["mpv_fit_status"], good["mpv_method"] = "CONVERGED", "two_stage_langaus"
    return good


# load calibration rejects bad keys and reports values
def test_load_calibration_rejects_bad_keys_and_reports_values():
    good = _good_table()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df = cio.load_calibration(_write_table(tmp / "ok.csv", good.iloc[::-1]))
        assert list(df["layer"])[:2] == [1, 1] and not df.attrs["value_problems"] and len(df) == 384
        assert df["layer"].dtype == np.int64
        cases = {"dup": pd.concat([good, good.iloc[[5]]]),
                 "strip8_in_layer1": good.assign(strip=np.where(good.index == 0, 8, good["strip"])),
                 "negative_layer": good.assign(layer=np.where(good.index == 0, -1, good["layer"])),
                 "end2": good.assign(end=np.where(good.index == 0, 2, good["end"])),
                 "fraction": good.assign(strip=np.where(good.index == 0, 0.5, good["strip"])),
                 "missing_key": good.assign(layer=np.where(good.index == 0, np.nan, good["layer"])),
                 "layer257": good.assign(layer=np.where(good.index == 0, 257, good["layer"])),
                 "text_key": good.assign(end=np.where(good.index == 0, "x", good["end"].astype(str)))}
        for name, bad in cases.items():
            exc = _raises(ValueError, cio.load_calibration, _write_table(tmp / f"{name}.csv", bad), contains="key")
            assert isinstance(exc, io_utils.ChannelKeyError), (name, type(exc))
        assert cio.key_problems(good) == [] and "share" in cio.key_problems(cases["dup"])[0]
        values = good.copy()
        values.loc[0, "pedestal_sigma"] = -3.0
        values.loc[1, "mpv"] = -999.0
        values.loc[2, "mpv"] = np.nan
        p = _write_table(tmp / "values.csv", values)
        df = cio.load_calibration(p)                                    # reported, values unchanged
        assert set(df.attrs["value_problems"]) == {"NONPOSITIVE_PEDESTAL_SIGMA", "NONPOSITIVE_MPV",
                                                   "STATUS_OK_WITHOUT_MPV"}
        assert df.loc[1, "mpv"] == -999.0                              # a sentinel value reaches the consumer
        _raises(ValueError, cio.load_calibration, p, strict=True, contains="NONPOSITIVE_MPV")
        partial = good.iloc[10:]                                       # missing channels: read as they are
        assert len(cio.load_calibration(_write_table(tmp / "partial.csv", partial))) == 374
        # part tables: only the pedestal or only the MPV columns
        ped = cio.load_calibration(_write_table(tmp / "ped.csv", good[cio.KEY_COLUMNS + cp.PEDESTAL_COLUMNS]))
        assert "mpv" not in ped.columns and not ped.attrs["value_problems"]
        mpv = cio.load_calibration(_write_table(tmp / "mpv.csv", good[cio.KEY_COLUMNS + cp.MPV_COLUMNS]))
        assert "pedestal" not in mpv.columns and (mpv["mpv"] == 600.0).all()


# channel keys shared rule in lookups
def test_channel_keys_shared_rule_in_lookups():
    # Duplicated keys can no longer make the row order decide a value (900 vs 100), a fractional key is never
    # truncated onto a real channel, a non-physical key inside the lookup array (layer 1 strip 8) and an empty key are
    # refused; valid partial tables work.
    rows = pd.DataFrame([dict(layer=1, strip=0, end=0, mpv=100.), dict(layer=1, strip=0, end=0, mpv=900.)])
    for frame in (rows, rows.iloc[::-1]):
        _raises(io_utils.ChannelKeyError, cio.calibration_lookup, frame, "mpv", contains="share")
    for bad in (dict(layer=1, strip=0.9, end=0, mpv=1.), dict(layer=1, strip=8, end=0, mpv=1.),
                dict(layer=np.nan, strip=0, end=0, mpv=1.), dict(layer=2, strip=-1, end=1, mpv=1.),
                dict(layer=20, strip=0, end=0, mpv=1.), dict(layer=1, strip=0, end=2, mpv=1.)):
        _raises(ValueError, cio.calibration_lookup, pd.DataFrame([bad]), "mpv", contains="key")
    arr = cio.calibration_lookup(pd.DataFrame([dict(layer=1, strip=0, end=0, x=1.0)]), "x")
    assert arr[1, 0, 0] == 1.0 and np.isnan(arr[2, 11, 1]) and np.isnan(arr[0, 0, 0])
    assert np.isnan(cio.calibration_lookup(pd.DataFrame([dict(layer=1, strip=0, end=0)]), "absent")).all()
    slope = pd.DataFrame([dict(layer=1, strip=0, end=0, tot_per_adc_slope=s) for s in (1., 2.)])
    _raises(io_utils.ChannelKeyError, cio.calibration_lookup, slope, "tot_per_adc_slope")   # TOT line table probe


# ----------------------------------------------------------------------------------------------
# CAL2-02: converter sample counts (n_samples_used), also through the test subset
# ----------------------------------------------------------------------------------------------
# write a make-csv provenance record next to csv (describing its current size and time)
def _make_csv_record(csv: Path, **fields) -> None:
    st = csv.stat()
    io_utils.write_json({"output": {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}, **fields},
                        csv.with_name(f"{csv.stem}.provenance.json"))


# cal2 02 sample count uses n samples used
def test_cal2_02_sample_count_uses_n_samples_used():
    # Codex CAL2-02 probe as a test: 8 raw / 4 used warns, 10 raw / 8 used does not, 10 / 10 warns, different
    # counts per input file warn, and a csv without trustworthy converter provenance is 'unknown', never 'ok'.
    cases = [({"n_samples": 8, "n_samples_used": 4}, "mismatch", True),
             ({"n_samples": 10, "n_samples_used": 8}, "ok", False),
             ({"n_samples": 10, "n_samples_used": 10}, "mismatch", True),
             ({"n_samples": 8, "n_samples_used": {"a.root": 8, "b.root": 4}}, "mixed", True),
             ({"n_samples": 8, "n_samples_used": {"a.root": 8, "b.root": 8}}, "ok", False),
             ({"n_samples": 8}, "unknown", False),                      # old record: used count not recorded
             ({"n_samples": 8, "n_samples_used": "eight"}, "unknown", False)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (fields, status, warns) in enumerate(cases):
            csv = _beam_csv(tmp / f"m{i}.csv", n_events=2)
            _make_csv_record(csv, **fields)
            messages = []
            with patch.object(cp, "log", side_effect=messages.append):
                note = cp.input_samples_note(csv)
            assert note["status"] == status, (fields, note)
            assert any("WARNING" in m for m in messages) == warns, (fields, messages)
        bare = _beam_csv(tmp / "bare.csv", n_events=2)                   # no provenance at all
        assert cp.input_samples_note(bare)["status"] == "unknown"
        other = _beam_csv(tmp / "other.csv", n_events=2)
        io_utils.write_json({"output": {"size": 1, "mtime_ns": 1}, "n_samples": 8, "n_samples_used": 8},
                            tmp / "other.provenance.json")                # record of another file version
        note = cp.input_samples_note(other)
        assert note["status"] == "unknown" and "another version" in note["record"]

        # the automatic test subset keeps the converter counts of its source (and the check sees them)
        src = _beam_csv(tmp / "windowed" / "windowed.csv", n_events=30)
        _make_csv_record(src, n_samples=8, n_samples_used=4, waveform=False)
        subset = io_utils.build_test_subset(src, tmp / "cache" / "windowed_first10events.csv", 10, chunksize=500)
        messages = []
        with patch.object(cp, "log", side_effect=messages.append):
            note = cp.input_samples_note(subset)
        assert note["status"] == "mismatch" and note["n_samples_used"] == 4 and "test-subset" in note["record"]
        assert any("WARNING" in m and "4 time samples" in m for m in messages), messages
        # the stage records the note of every input csv per part; a part table records the parts it computed
        ctx = _ctx(tmp / "stage", src)
        prov = io_utils.read_json(cp.provenance_json(_run(ctx)))
        assert [n["status"] for n in prov["input_samples"]["pedestal"]] == ["mismatch"]
        assert [n["status"] for n in prov["input_samples"]["mpv"]] == ["mismatch"]
        prov = io_utils.read_json(cp.provenance_json(_run(ctx, "-only", "pedestal_csv")))
        assert set(prov["input_samples"]) == {"pedestal"}
        assert prov["input_samples"]["pedestal"][0]["status"] == "mismatch"


# ----------------------------------------------------------------------------------------------
# routing, help texts and -only
# ----------------------------------------------------------------------------------------------
# output routing and help texts
def test_output_routing_and_help_texts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for test, mode in ((False, "fullset"), (True, "test")):
            ctx = _ctx(tmp, tmp / "x" / "x.csv", test=test)
            sub = "test" if test else ""      # calibration / cache: no fullset folder; results keep the mode folder
            cal = tmp / "calibration" / sub / "pedestal_mpv"
            assert cp.output_csv(ctx) == cal / "calibration_parameters.csv"
            assert cp.part_csv(ctx, "pedestal") == cal / "pedestal_parameters.csv"
            assert cp.part_csv(ctx, "mpv") == cal / "mpv_parameters.csv"
            for units, name in ((None, "calibration_parameters.csv"), (("pedestal_csv",), "pedestal_parameters.csv"),
                                (("mpv_csv", "mpv_plot"), "mpv_parameters.csv"),
                                (("pedestal_csv", "mpv_csv"), "calibration_parameters.csv"),
                                (("pedestal_plot", "mpv_plot"), None)):
                ctx.units = units
                got = cp.table_csv(ctx)
                assert (got.name if got else None) == name, units
            ctx.units = None
            assert cp.pedestal_pdf(ctx) == tmp / "results" / mode / "pedestal_mpv" / "pedestal_fits.pdf"
            assert cp.mpv_pdf(ctx) == tmp / "results" / mode / "pedestal_mpv" / "mpv_fits.pdf"
            assert cp.histogram_cache(ctx) == tmp / "cache" / sub / "pedestal_mpv" / "calibration_histograms.npz"
            assert cp.provenance_json(cp.output_csv(ctx)).name == "calibration_parameters.provenance.json"
    for path in (Path(cp.__file__), Path(cio.__file__)):
        text = path.read_text(encoding="utf-8")
        assert "output_root" not in text and "out_dir(" not in text, path
    p = argparse.ArgumentParser()
    cp.add_arguments(p)
    help_text = (cp.__doc__ or "") + p.format_help() + (cio.__doc__ or "")
    for old in ("outputs/", "outputs\\", "outputs_test", "config/", "config\\", "python runner.py",
                "data/<mode>/pedestal_mpv", "data/test/test_subsets", "removed"):
        assert old not in help_text, old
    for column in (*cio.KEY_COLUMNS, *cio.CALIBRATION_COLUMNS):   # the module docstring lists every table variable
        assert column in cp.__doc__, column
    assert cp.PEDESTAL_CSV in cp.__doc__ and cp.MPV_CSV in cp.__doc__
    assert cp.SUPPORTS_FROM_SAVED is True and cp.SUPPORTS_CLEAN_INPUT is True
    for option in ("--from-saved", "-from-saved", "--clean-input", "-clean-input"):   # common runner options
        assert option not in p._option_string_actions, option
    assert "-only" in p._option_string_actions


# only units and groups
def test_only_units_and_groups():
    # -only: the four units in stage order, the groups pedestal / mpv / csv / plots / all, a second -only adds;
    # -no-plots drops the plot units.
    assert cp.UNITS == ("pedestal_csv", "mpv_csv", "pedestal_plot", "mpv_plot")
    assert _args().only == cp.UNITS and _args("-only", "all").only == cp.UNITS
    assert _args("-only", "mpv").only == ("mpv_csv", "mpv_plot")
    assert _args("-only", "pedestal").only == ("pedestal_csv", "pedestal_plot")
    assert _args("-only", "pedestal_plot,mpv_csv").only == ("mpv_csv", "pedestal_plot")
    assert _args("-only", "csv").only == ("pedestal_csv", "mpv_csv")
    assert _args("-only", "plots", "-only", "pedestal_csv").only == ("pedestal_csv", "pedestal_plot", "mpv_plot")
    assert "unknown unit" in _parse_error("-only", "table")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = _ctx(tmp, tmp / "x" / "x.csv", make_plots=False)
        _units(ctx, _args())
        assert cp.chosen_units(ctx) == {"pedestal_csv": True, "mpv_csv": True, "pedestal_plot": False,
                                        "mpv_plot": False}
        _units(ctx, _args("-only", "plots"))
        _raises(SystemExit, cp.units_or_stop, ctx, contains="nothing to make")
        _raises(SystemExit, cp.run, ctx, _args("-only", "plots"), [], contains="nothing to make")
        assert not (tmp / "calibration").exists() and not (tmp / "cache").exists()


# ----------------------------------------------------------------------------------------------
# the source run (Lei 2026-10-06: no input gates; the run given is the run used)
# ----------------------------------------------------------------------------------------------
# _ctx plus two electron runs on the same run folder (the stage cannot tell the particle from the data).
def _ctx_with_runs(tmp: Path, csv: Path, section=None) -> RunContext:
    ctx = _ctx(tmp, csv, section=section)
    for energy in (4, 1):
        ctx.cfg["datasets"][f"electron_{energy}GeV"] = {"particle": "electron", "energy_gev": energy, "kind": "data",
                                                       "folder": str(Path(csv).parent)}
    return ctx


# source run is the run flag of any particle
def test_source_run_is_the_run_flag_of_any_particle():
    # calib --electron 4 calibrates on electron 4; no flag = calibration.source_dataset (default "muon 4").  A
    # non-muon run is used as given, with a WARNING in the log; the run used is recorded in the provenance.
    assert cp.DEFAULTS["source_dataset"] == "muon 4" and not hasattr(cp, "PARTICLES")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "beam" / "beam.csv", n_events=300)
        ctx = _ctx_with_runs(tmp, csv)
        reg = dataset_registry(ctx.cfg)
        assert cp.source_dataset(ctx, []).key == "muon_4GeV"
        assert cp.source_dataset(ctx, [reg["electron_4GeV"]]).key == "electron_4GeV"
        assert cp.source_dataset(_ctx_with_runs(tmp, csv, {"source_dataset": "Electron 1GeV"}), []).key == \
            "electron_1GeV"
        with redirect_stdout(io.StringIO()) as text:
            cp.run(ctx, _args(), [reg["electron_4GeV"]])
        log_text = text.getvalue()
        assert ("source run of the pedestals and MIPs: electron 4 (from the run flag --electron 4)" in log_text), \
            log_text[-3000:]
        assert "WARNING: calib on electron 4: the MIP MPVs come from a electron run" in log_text
        prov = io_utils.read_json(cp.provenance_json(cp.output_csv(ctx)))
        assert prov["source_dataset"] == "electron_4GeV" and prov["inputs"]["pedestal"][0]["size"] == csv.stat().st_size
        # the rebuild command of a -from-saved refusal repeats the run flags
        assert cp._rebuild_command(ctx, [reg["electron_4GeV"]]).startswith(".\\runner.ps1 calib --electron 4 ")
        with redirect_stdout(io.StringIO()) as text:
            cp.run(ctx, _args(), [])
        assert "from calibration.source_dataset = 'muon 4'" in text.getvalue()
        assert "WARNING: calib on" not in text.getvalue()
        assert io_utils.read_json(cp.provenance_json(cp.output_csv(ctx)))["source_dataset"] == "muon_4GeV"
        # a pedestal-only run on a non-muon run fits no MIP: no warning
        with redirect_stdout(io.StringIO()) as text:
            _units(ctx, _args("-only", "pedestal_csv"))
            cp.run(ctx, _args("-only", "pedestal_csv"), [reg["electron_4GeV"]])
        assert "WARNING: calib on" not in text.getvalue()


# source run problems stop at the point of use
def test_source_run_problems_stop_at_the_point_of_use():
    # calib calibrates on one run: several runs from the flags (or a yml source naming several) stop before
    # anything is read, with the runs named by reference and how to choose one; check shows the same reason
    # (STALE) instead of stopping.  A source that names no run stops with the setting and the existing runs; a run
    # folder without any csv stops with the folder and the make-csv command.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "beam" / "beam.csv", n_events=100)
        ctx = _ctx_with_runs(tmp, csv)
        reg = dataset_registry(ctx.cfg)
        both = [reg["electron_4GeV"], reg["electron_1GeV"]]
        with patch.object(io_utils, "iter_csv", _no_csv_reading):   # stops before any reading
            exc = _raises(SystemExit, cp.run, ctx, _args(), both)
        assert str(exc).startswith("[calib] the run flags --electron 4 --electron 1 chose 2 runs (electron 4, "
                                   "electron 1); calib calibrates on one run: give one run flag, e.g. "
                                   ".\\runner.ps1 calib --electron 4"), str(exc)
        plan = cp.io_plan(ctx, _args(), both)
        assert len(plan) == 2 and all(i["stale"] and "calib would stop: [calib] the run flags" in i["produced_by"]
                                      for i in plan)
        for value, text in (("electron", "calibration.source_dataset = 'electron' names 2 runs (electron 4, "
                                         "electron 1); calib calibrates on one run: write one of them there"),
                            ("muon 9", "calibration.source_dataset: there is no run 'muon 9'; runs of "),
                            ("pion 3", "calibration.source_dataset: there is no run 'pion 3'"),
                            ([], "calibration.source_dataset = [] names no run")):
            _raises(SystemExit, cp.source_dataset, _ctx_with_runs(tmp, csv, {"source_dataset": value}), [],
                    contains=text)
        empty = tmp / "empty" / "none.csv"
        empty.parent.mkdir()
        _raises(SystemExit, cp.run, _ctx(tmp, empty), _args(), [], contains=("no csv file in", "make-csv --muon 4"))
        assert not (tmp / "calibration").exists() and not (tmp / "cache").exists()


# a run of several csv files sums their histograms
def test_a_run_of_several_csv_files_sums_their_histograms():
    # A run is the folder of its csv files (one per ROOT file): the histograms are summed over every csv of the
    # folder (an event is identified by its csv file, so two files may both start at event 1), and the provenance and
    # the cache record name every input csv.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frame = _beam_frame(n_events=600)
        for folder in ("one", "two"):
            (tmp / folder).mkdir()
        frame.to_csv(tmp / "one" / "a.csv", index=False)
        half = 300 * len(geo.all_bars())                                    # rows of the first 300 events
        frame.iloc[:half].to_csv(tmp / "two" / "a.csv", index=False)
        rest = frame.iloc[half:]
        rest.assign(pf_event=rest["pf_event"] - 300).to_csv(tmp / "two" / "b.csv", index=False)   # also from event 1
        whole = cio.load_calibration(_run(_ctx(tmp / "p1", tmp / "one" / "a.csv")), float_precision="round_trip")
        ctx = _ctx(tmp / "p2", tmp / "two" / "a.csv")
        split = cio.load_calibration(_run(ctx), float_precision="round_trip")
        for c in ("pedestal", "pedestal_sigma", "n_pedestal_events", "mpv", "mpv_error", "mpv_status"):
            assert split[c].equals(whole[c]), c
        prov = io_utils.read_json(cp.provenance_json(cp.output_csv(ctx)))
        assert [Path(i["path"]).name for i in prov["inputs"]["pedestal"]] == ["a.csv", "b.csv"]
        assert "[a.csv, b.csv]" in prov["source_label"]
        record = io_utils.read_json(cp.provenance_json(cp.histogram_cache(ctx)))
        assert len(record["pedestal_histograms_inputs"]) == 2 and len(record["mpv_histograms_inputs"]) == 2
        assert record["n_rows_pedestal"] == len(frame)
        plan = [Path(i["path"]).name for i in _plan(ctx) if i["role"] == "in"]
        assert plan == ["a.csv", "b.csv"]


# ----------------------------------------------------------------------------------------------
# end to end on a synthetic beam csv: the full run and every unit / group of -only
# ----------------------------------------------------------------------------------------------
# stage end to end units
def test_stage_end_to_end_units():
    # A full run (every unit) writes the full table + provenance + cache; each -only unit and group makes exactly
    # its files: one csv unit -> its part table with only its columns, both csv units -> the full table, plot units ->
    # only their pdf.  Every run computes what it writes from the csv: an MPV-only run fits on its own pedestal (also
    # when the full table holds another one), a pedestal-only run copies nothing from an existing table, and no run
    # rewrites a file it was not asked for.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", multi_hit_fraction=0.03)
        ctx = _ctx(tmp, csv)
        out = _run(ctx)
        assert out == cp.output_csv(ctx)
        table = cio.load_calibration(out)
        raw_table = pd.read_csv(out)
        prov = io_utils.read_json(cp.provenance_json(out))
        cache = cp.read_histogram_cache(cp.histogram_cache(ctx))
        assert list(raw_table.columns) == cio.KEY_COLUMNS + list(cio.CALIBRATION_COLUMNS)
        assert len(table) == 384 and not (tmp / "results").exists()          # --no-plots equivalent
        assert not (tmp / "data").exists()                                   # the cache is in cache/, nothing in data/
        mip = np.zeros(384, bool)
        for lay, st in MIP_BARS:
            mip |= ((table["layer"] == lay) & (table["strip"] == st)).to_numpy()
        assert (table.loc[mip, "mpv_status"] == "OK").all(), table.loc[mip, ["mpv_status", "mpv_fit_message"]]
        assert (table.loc[~mip, "mpv_status"] == "LOW_STATS").all()
        assert (table["pedestal_selection"] == "raw_samples").all()
        assert (table["pedestal_anchor_method"] == "median_sum").all()
        assert (table.loc[mip, "mpv_covariance_status"] == "OK").all()
        assert (table["mpv_method"] == "two_stage_langaus").all()
        assert np.array_equal(table["mpv_pedestal"], table["pedestal"])
        assert prov["run_token"] == cache["meta"]["run_token"] and prov["pedestal_selection"] == "raw_samples"
        assert prov["table"] == "full" and prov["units"] == ["pedestal_csv", "mpv_csv"]   # -no-plots: no plot unit
        assert prov["inputs"]["pedestal"][0]["size"] == csv.stat().st_size
        assert prov["columns"] == list(raw_table.columns)
        assert prov["from_saved"] is False and prov["clean_input"] is False
        assert prov["input_samples"]["pedestal"][0]["status"] == "unknown"   # synthetic csv: no converter record
        assert prov["settings"] == {**cp.fit_settings(cp.settings_for(ctx)), "pedestal_window_halfwidth": 200.0}
        assert prov["histograms"]["pedestal"]["selection"] == "raw_samples"
        for record, member in ((cp.provenance_json(out), out),
                               (cp.provenance_json(cp.histogram_cache(ctx)), cp.histogram_cache(ctx))):
            assert io_utils.product_set_valid(record, None, members=[member]) == (True, "ok")

        # dry-run plans: the table of the chosen csv units and the chosen pdfs
        plot_ctx = _ctx(tmp, csv, make_plots=True)
        plan = {(i["role"], Path(i["path"]).name) for i in _plan(plot_ctx)}
        assert {("out", "calibration_parameters.csv"), ("out", "calibration_histograms.provenance.json"),
                ("out", "pedestal_fits.pdf"), ("out", "mpv_fits.pdf")} <= plan, plan
        for argv, table_name, pdfs in ((("-only", "pedestal_csv"), "pedestal_parameters.csv", set()),
                                       (("-only", "mpv"), "mpv_parameters.csv", {"mpv_fits.pdf"}),
                                       (("-only", "plots"), None, {"pedestal_fits.pdf", "mpv_fits.pdf"})):
            names = {Path(i["path"]).name for i in _plan(plot_ctx, *argv) if i["role"] == "out"}
            assert {n for n in names if n.endswith("parameters.csv")} == ({table_name} if table_name else set()), names
            assert {n for n in names if n.endswith(".pdf")} == pdfs, (argv, names)

        # the csv approximation sees the multi-hit rows as pedestal events -> higher pedestal
        approx_ctx = _ctx(tmp / "approx", csv, section={"pedestal_selection": "csv_tot_toa_zero"})
        h = cp.accumulate_histograms(csv, True, False, selection="csv_tot_toa_zero")
        approx = cp.pedestals_from_histograms(h["h_selected"], None, cp.settings_for(approx_ctx))
        assert (approx.loc[~mip, "pedestal"].to_numpy() > table.loc[~mip, "pedestal"].to_numpy()).mean() > 0.9

        s = cp.settings_for(ctx)
        hists = cp.accumulate_histograms(csv, True, True)
        ped = cp.pedestals_from_histograms(hists["h_selected"], None, s)
        tasks = cp.mpv_tasks(hists["h_all"], ped["pedestal"].to_numpy(), ped["pedestal_sigma"].to_numpy(), s)
        fits = cp.run_mpv_fits(tasks, 1)
        assert np.array_equal(cache["h_all"], hists["h_all"])
        assert np.array_equal(cache["h_selected"], hists["h_selected"])
        ped_counts0 = cp.pedestal_plot_counts(hists["h_selected"], table["pedestal_anchor"].to_numpy(float))
        mpv_counts0 = [f["final_counts"] for f in fits]
        drawn = []   # what the stage draws (pdf rendering itself is the slow test's job)

        # stand-in of plot_pedestals: records what would be drawn
        def fake_pedestals(path, tab, counts, halfwidth, label):
            drawn.append(("pedestal", counts.copy(), halfwidth, label, None))
            return Path(path)

        # stand-in of plot_mpv: records what would be drawn
        def fake_mpv(path, tab, shown, settings, label):
            drawn.append(("mpv", [f["final_counts"] for f in shown], settings["mpv_final_bin_width"], label,
                          tab["mpv_status"].tolist()))
            return Path(path)

        tables = {"full": out, "pedestal": cp.part_csv(ctx, "pedestal"), "mpv": cp.part_csv(ctx, "mpv")}

        # run calib with the stand-ins; returns the tables it rewrote
        def run_drawn(*argv, csv_path=csv):
            drawn.clear()
            before = {k: (p.stat().st_mtime_ns if p.exists() else None) for k, p in tables.items()}
            with patch.object(cp, "plot_pedestals", fake_pedestals), patch.object(cp, "plot_mpv", fake_mpv):
                _run(_ctx(tmp, csv_path, make_plots=True), *argv)
            written = {k for k, p in tables.items() if p.exists() and p.stat().st_mtime_ns != before[k]}
            return written, {d[0]: d for d in drawn}

        # the same histogram counts (None matches None)
        def same_counts(a, b):
            return len(a) == len(b) and all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(a, b))

        full = pd.read_csv(out)
        for argv, files, pdfs in ((("-only", "pedestal_csv"), {"pedestal"}, set()),
                                  (("-only", "mpv_csv"), {"mpv"}, set()),
                                  (("-only", "pedestal_plot"), set(), {"pedestal"}),
                                  (("-only", "mpv_plot"), set(), {"mpv"}),
                                  (("-only", "pedestal"), {"pedestal"}, {"pedestal"}),
                                  (("-only", "mpv"), {"mpv"}, {"mpv"}),
                                  (("-only", "plots"), set(), {"pedestal", "mpv"}),
                                  (("-only", "csv"), {"full"}, set()),
                                  (("-only", "pedestal_csv,mpv_plot"), {"pedestal"}, {"mpv"}),
                                  ((), {"full"}, {"pedestal", "mpv"})):
            written, shown = run_drawn(*argv)
            assert written == files and set(shown) == pdfs, (argv, written, set(shown))
            if "pedestal" in shown:
                assert np.array_equal(shown["pedestal"][1], ped_counts0)
                assert shown["pedestal"][3] == prov["source_label"]
            if "mpv" in shown:
                assert same_counts(shown["mpv"][1], mpv_counts0) and shown["mpv"][4] == table["mpv_status"].tolist()
        # the part tables: only their columns, the same values as the full table of the same data
        for name, part in (("pedestal", cp.PEDESTAL_COLUMNS), ("mpv", cp.MPV_COLUMNS)):
            got = pd.read_csv(tables[name])
            assert list(got.columns) == cio.KEY_COLUMNS + [c for c in cio.CALIBRATION_COLUMNS if c in part]
            for c in part:
                assert got[c].equals(full[c]), (name, c)
            record = io_utils.read_json(cp.provenance_json(tables[name]))
            assert record["table"] == name and record["columns"] == list(got.columns)
        assert "mpv_pedestal" in pd.read_csv(tables["mpv"]).columns            # the pedestal the MPV fit used

        # shifted pedestals: each run computes its own pedestal; nothing is copied from (or into) the full table
        shifted = _beam_csv(tmp / "shifted" / "mu.csv", shift=4, multi_hit_fraction=0.03)
        stamp = out.read_bytes()
        assert run_drawn("-only", "pedestal_csv", csv_path=shifted)[0] == {"pedestal"}
        new_ped = pd.read_csv(tables["pedestal"])
        assert (new_ped.loc[mip, "pedestal"] - full.loc[mip, "pedestal"]).abs().min() > 2.0
        assert "mpv" not in new_ped.columns
        assert run_drawn("-only", "mpv_csv", csv_path=shifted)[0] == {"mpv"}
        new_mpv = pd.read_csv(tables["mpv"])
        assert new_mpv["mpv_pedestal"].equals(new_ped["pedestal"])           # its own pedestal, not the table's
        assert new_mpv["mpv_pedestal_sigma"].equals(new_ped["pedestal_sigma"])
        assert (new_mpv.loc[mip, "mpv_status"] == "OK").all()
        assert out.read_bytes() == stamp                                    # the full table was never touched


# ----------------------------------------------------------------------------------------------
# D1b / G2: -from-saved
# ----------------------------------------------------------------------------------------------
# from saved refits from the raw histograms without the csv
def test_from_saved_refits_from_the_raw_histograms_without_the_csv():
    # With the csv reader disabled, -from-saved refits pedestal and MPV with other fit settings from the cached raw
    # histograms; the table is byte-identical to a csv run with the same settings; part runs too; -from-saved never
    # rewrites the cache; any change of the input csv, the pedestal selection, the accumulator version or the npz, a
    # cache without the needed part, or a missing cache, refuses with the reason and the rebuild command.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", multi_hit_fraction=0.03)
        ctx = _ctx(tmp / "saved", csv)
        out = _run(ctx)
        hists = cp.accumulate_histograms(csv, True, True)
        cache = cp.read_histogram_cache(cp.histogram_cache(ctx))
        for key in ("h_selected", "h_all"):
            assert np.array_equal(cache[key], hists[key]), key       # element by element
        assert cache["h_sample"] is None
        token0 = io_utils.read_json(cp.provenance_json(out))["run_token"]
        npz_bytes = cp.histogram_cache(ctx).read_bytes()

        other = ("--mpv-nsigma", "6", "--mpv-window-widths", "1.5", "--pedestal-window", "150", "--mpv-bin-width", "25")
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            saved = _run(ctx, *other, from_saved=True)
        ref_ctx = _ctx(tmp / "reference", csv)
        reference = _run(ref_ctx, *other)                              # a csv run with the same settings
        assert saved.read_bytes() == reference.read_bytes()
        assert _run(_ctx(tmp / "default", csv)).read_bytes() != saved.read_bytes()      # the settings mattered
        prov = io_utils.read_json(cp.provenance_json(saved))
        assert prov["from_saved"] is True and prov["run_token"] != token0 and prov["histogram_cache"] is None
        assert cp.histogram_cache(ctx).read_bytes() == npz_bytes        # the cache stays as it was

        # part runs from the cache equal the same part runs from the csv
        for argv in (("-only", "mpv_csv", "--mpv-nsigma", "4"), ("-only", "pedestal_csv", "--pedestal-window", "180")):
            with patch.object(io_utils, "iter_csv", _no_csv_reading):
                from_cache = _run(ctx, *argv, from_saved=True)
            assert from_cache.read_bytes() == _run(ref_ctx, *argv).read_bytes(), argv

        rebuild = "Rebuild them with: .\\runner.ps1 calib"
        # the pedestal selection is part of the identity of the pedestal histograms, which every run needs
        for argv in ((), ("-only", "mpv_csv")):
            _raises(io_utils.ProductSetError, _run, ctx, *argv, "--pedestal-selection", "csv_tot_toa_zero",
                    from_saved=True, contains=("pedestal_histograms_selection", rebuild))
        # an accumulator / format change invalidates every cache
        with patch.dict(cp.CACHE_IDENTITY, accumulator="adc-sum-int-bincount-v2"):
            _raises(io_utils.ProductSetError, _run, ctx, from_saved=True, contains=("'accumulator' differs", rebuild))
        # a damaged npz: same size and modification time, one byte changed -> the content hash refuses it
        npz = cp.histogram_cache(ctx)
        blob, st = npz.read_bytes(), npz.stat()
        middle = len(blob) // 2
        npz.write_bytes(blob[:middle] + bytes([blob[middle] ^ 0xFF]) + blob[middle + 1:])
        os.utime(npz, ns=(st.st_atime_ns, st.st_mtime_ns))
        _raises(io_utils.ProductSetError, _run, ctx, from_saved=True,
                contains=("changed since it was committed (content)", rebuild))
        npz.write_bytes(blob)
        os.utime(npz, ns=(st.st_atime_ns, st.st_mtime_ns))
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            _run(ctx, "-only", "mpv_csv", from_saved=True)             # the committed bytes again: valid
        # a cache of a pedestal-only run holds no MPV histograms: an MPV run refuses it, a pedestal run uses it
        pctx = _ctx(tmp / "ped", csv)
        _run(pctx, "-only", "pedestal_csv")
        assert cp.read_histogram_cache(cp.histogram_cache(pctx))["h_all"] is None
        _raises(io_utils.ProductSetError, _run, pctx, "-only", "mpv_csv", from_saved=True,
                contains=("mpv_histograms_inputs", rebuild))
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            _run(pctx, "-only", "pedestal_csv", from_saved=True)
        # a new version of the input csv (same path)
        _beam_csv(csv, seed=9, multi_hit_fraction=0.03)
        exc = _raises(io_utils.ProductSetError, _run, ctx, from_saved=True,
                      contains=("_histograms_inputs' differs", "the cached pedestal histograms were made from",
                                "this run reads", rebuild))
        assert "-from-saved" not in str(exc).split("Rebuild them with:")[1].split("(")[0]
        # a missing cache
        npz.unlink()
        cp.provenance_json(npz).unlink()
        _raises(io_utils.ProductSetError, _run, ctx, from_saved=True, contains=("no metadata", rebuild))


# from saved in test mode never builds the subset
def test_from_saved_in_test_mode_never_builds_the_subset():
    # -from-saved must not read the full csv: a missing test subset stops it with the command that builds it;
    # after a normal -test run the subset and the cache exist and -from-saved works.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=300)
        ctx = _ctx(tmp, csv, test=True)
        with patch.object(io_utils, "build_test_subset", _no_csv_reading):
            _raises(io_utils.ProductSetError, _run, ctx, from_saved=True, contains=("test subset", "calib -test"))
        _run(ctx)                                                      # builds the subset and the cache
        assert ctx.test_subset_csv(dataset_registry(ctx.cfg)["muon_4GeV"], csv).exists()
        with patch.object(io_utils, "iter_csv", _no_csv_reading), \
                patch.object(io_utils, "build_test_subset", _no_csv_reading):
            out = _run(ctx, "--mpv-nsigma", "4", from_saved=True)
        assert io_utils.read_json(cp.provenance_json(out))["settings"]["mpv_nsigma"] == 4.0


# from saved plan shows reuse or rebuild
def test_from_saved_plan_shows_reuse_or_rebuild():
    # The dry-run plan of -from-saved lists the cache with the identity it needs (runner check: reuse /
    # rebuild with the reason); a normal plan never creates anything.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=200)
        ctx = _ctx(tmp, csv)
        item = [i for i in _plan(ctx, from_saved=True) if i["role"] == "cache"][0]
        assert not io_utils.product_set_valid(item["path"], item["expected"], item["members"])[0]   # nothing yet
        assert not (tmp / "cache").exists() and not (tmp / "calibration").exists()
        _run(ctx)
        item = [i for i in _plan(ctx, from_saved=True) if i["role"] == "cache"][0]
        assert io_utils.product_set_valid(item["path"], item["expected"], item["members"]) == (True, "ok")
        item = [i for i in _plan(ctx, "--pedestal-selection", "csv_tot_toa_zero", from_saved=True)
                if i["role"] == "cache"][0]
        ok, reason = io_utils.product_set_valid(item["path"], item["expected"], item["members"])
        assert not ok and "pedestal_histograms_selection" in reason
        assert not [i for i in _plan(ctx) if i["role"] == "cache"]
        beam = [i for i in _plan(ctx) if i["role"] == "in"][0]
        assert beam["kind"] == "external" and Path(beam["path"]) == csv


# ----------------------------------------------------------------------------------------------
# D8: -clean-input (MPV from the clean dataset, pedestal from the beam csv)
# ----------------------------------------------------------------------------------------------
# clean arguments parsed for -clean-input
def _clean_args(*argv):
    from ldmx_hcal.selection import clean_dataset as cd

    p = argparse.ArgumentParser()
    cd.add_arguments(p)
    args = p.parse_args(list(argv))
    args.from_saved = False
    return args


# clean input mpv from the clean dataset
def test_clean_input_mpv_from_the_clean_dataset():
    # Lei D8 with a REAL clean run (no mock of the currency check): MPV from the clean dataset of the csv, pedestal from
    # the beam csv.  The clean dataset must be current (RunContext.input_csv): after calib wrote new MPVs it is not, so
    # a csv run stops with the clean command, while -from-saved refits on the cached histograms of the same clean file
    # (the cache record checks it).  A pedestal-only run never reads the clean dataset.  The sample-count note follows
    # the clean record to the converter provenance of its beam csv.
    from ldmx_hcal.selection import clean_dataset as cd

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        beam = _beam_csv(tmp / "mu" / "mu.csv", n_events=800, multi_hit_fraction=0.03, mip_bars=((1, 3),) + MIP_BARS)
        _make_csv_record(beam, n_samples=8, n_samples_used=8, waveform=False)
        ctx = _ctx(tmp / "p", beam)
        ds = dataset_registry(ctx.cfg)["muon_4GeV"]
        reference = cio.load_calibration(_run(ctx), float_precision="round_trip")    # calib on the beam csv
        ctx.units = None
        cd.run(ctx, _clean_args("--muon-standard"), [ds])
        clean = ctx.clean_product(ds, beam)
        assert clean.exists() and len(pd.read_csv(clean)) > 300
        ctx.clean_input = True
        plan = [i for i in _plan(ctx) if i["role"] == "in"]
        assert [(i["kind"], Path(i["path"]), bool(i.get("stale"))) for i in plan] == [("external", beam, False),
                                                                                      ("product", clean, False)]

        # MPV on the clean dataset, pedestal from the beam csv
        out = _run(ctx)
        table = cio.load_calibration(out, float_precision="round_trip")
        assert table["pedestal"].equals(reference["pedestal"])
        assert table["pedestal_sigma"].equals(reference["pedestal_sigma"])
        h_clean = cp.accumulate_histograms(clean, False, True)["h_all"]
        s = cp.settings_for(ctx)
        fits = cp.run_mpv_fits(cp.mpv_tasks(h_clean, table["pedestal"].to_numpy(float),
                                            table["pedestal_sigma"].to_numpy(float), s), 1)
        assert np.array_equal(table["mpv"].to_numpy(float), np.array([f["mpv"] for f in fits], float), equal_nan=True)
        assert (table["mpv_status"] == "OK").sum() >= 4
        assert not np.array_equal(table["mpv"].to_numpy(float), reference["mpv"].to_numpy(float), equal_nan=True)
        prov = io_utils.read_json(cp.provenance_json(out))
        assert prov["clean_input"] is True and prov["inputs"]["mpv"][0]["path"] == io_utils.canonical_path(clean)
        assert Path(prov["inputs"]["pedestal"][0]["path"]).name == "mu.csv"
        note = prov["input_samples"]["mpv"][0]                           # clean record -> make-csv provenance
        assert note["status"] == "ok" and note["n_samples_used"] == 8 and "->" in note["record"], note
        record = io_utils.read_json(cp.provenance_json(cp.histogram_cache(ctx)))
        assert record["mpv_histograms_inputs"][0]["path"] == io_utils.canonical_path(clean)
        assert record["pedestal_histograms_inputs"][0]["path"] == io_utils.canonical_path(beam)
        assert np.array_equal(cp.read_histogram_cache(cp.histogram_cache(ctx))["h_all"], h_clean)

        # the table changed: a csv run stops with the clean command; -from-saved refits on the same clean file
        _raises(SystemExit, _run, ctx, "-only", "mpv_csv", contains="clean --muon 4")
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            saved = _run(ctx, "-only", "mpv_csv", "--mpv-nsigma", "6", from_saved=True)
        prov = io_utils.read_json(cp.provenance_json(saved))
        assert prov["from_saved"] is True and prov["inputs"]["mpv"][0]["path"] == io_utils.canonical_path(clean)
        # without -clean-input the cached MPV histograms (made from the clean dataset) do not match the beam csv
        ctx.clean_input = False
        _raises(io_utils.ProductSetError, _run, ctx, "-only", "mpv_csv", from_saved=True,
                contains="mpv_histograms_inputs")
        # a pedestal-only run never reads the clean dataset
        ctx.clean_input = True
        assert [Path(i["path"]).name for i in _plan(ctx, "-only", "pedestal_csv") if i["role"] == "in"] == ["mu.csv"]
        ped_only = cio.load_calibration(_run(ctx, "-only", "pedestal_csv"), float_precision="round_trip")
        assert ped_only["pedestal"].equals(reference["pedestal"]) and "mpv" not in ped_only.columns


# ----------------------------------------------------------------------------------------------
# commits: product sets survive failures, worker temp files are registered
# ----------------------------------------------------------------------------------------------
# failed commits keep the previous pair
def test_failed_commits_keep_the_previous_pair():
    # A rename that fails (e.g. the table is open in Excel) keeps the previous table + provenance (and cache +
    # record) byte-identical and valid: never a new file with an old record or the reverse.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=300)
        ctx = _ctx(tmp, csv)
        out = _run(ctx)
        npz = cp.histogram_cache(ctx)
        before = {p: p.read_bytes() for p in (out, cp.provenance_json(out), npz, cp.provenance_json(npz))}
        original = os.replace

        # os.replace stand-in that fails on target (a locked file)
        def locked(target):
            # os.replace that raises PermissionError for the locked target
            def replace(src, dst):
                if Path(dst) == target:
                    raise PermissionError(f"simulated lock on {Path(dst).name}")
                return original(src, dst)
            return replace

        for target in (npz, out):   # the cache is committed first, then the table
            with patch.object(io_utils.os, "replace", side_effect=locked(target)):
                _raises(PermissionError, _run, ctx, "--mpv-nsigma", "6")
            # the pair whose rename failed is untouched; a cache committed before a failed table is a complete,
            # valid set of its own
            pair = (target, cp.provenance_json(target))
            assert {p: p.read_bytes() for p in pair} == {p: before[p] for p in pair}, target.name
            if target == npz:
                assert {p: p.read_bytes() for p in before} == before
            assert io_utils.product_set_valid(cp.provenance_json(out), None, members=[out])[0]
            assert io_utils.product_set_valid(cp.provenance_json(npz), None, members=[npz])[0]
            leftovers = [p.name for p in (tmp / "calibration").rglob("*") if ".tmp" in p.name or ".old" in p.name]
            leftovers += [p.name for p in (tmp / "cache").rglob("*") if ".tmp" in p.name or ".old" in p.name]
            assert not leftovers, leftovers


# pedestal pdf temp of a worker is registered
def test_pedestal_pdf_temp_of_a_worker_is_registered():
    # With workers > 1 the pedestal pdf is drawn in a worker process (no temp registry there): the stage
    # registers its temp file, so a run that fails while the worker writes leaves nothing behind.  The temp file is
    # the one of the plan's destination: the usual name (--output), a user name, or the unsaved scratch copy.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=100)
        scratch = tmp / "cache" / "pedestal_mpv" / "unsaved"
        for names, temp in (([], tmp / "results" / "fullset" / "pedestal_mpv" / "pedestal_fits.tmp.pdf"),
                            (["pedestal_plot=ped.pdf"], tmp / "results" / "fullset" / "pedestal_mpv" / "ped.tmp.pdf"),
                            (None, scratch / "pedestal_fits.tmp.pdf")):
            ctx = _ctx(tmp, csv, make_plots=True, workers=2, outputs=names)

            # worker pool stand-in: leaves a half-written pdf temp and dies
            def dying_pool(tasks, workers, side_job=None, temp=temp):
                temp.parent.mkdir(parents=True, exist_ok=True)
                temp.write_bytes(b"%PDF half written by a worker")
                raise RuntimeError("worker died")

            with patch.object(cp, "run_mpv_fits", dying_pool):
                with io_utils.temp_registry(), redirect_stdout(io.StringIO()):
                    _raises(RuntimeError, _run, ctx)
            assert not temp.exists() and not cp.output_csv(ctx).exists(), names


# ----------------------------------------------------------------------------------------------
# --fit[a,b] / --bin N: the final MPV fit window of every channel (Lei 2026-10-02)
# ----------------------------------------------------------------------------------------------
# The argparse refusal of calib flags (argparse prints it on stderr and exits).
def _parse_error(*argv) -> str:
    err = io.StringIO()
    with patch("sys.stderr", err):
        _raises(SystemExit, _args, *argv)
    return err.getvalue()


# A small calibration table (first len(fits) channels) with the MPV columns of ``fits``.
def _mpv_row_table(fits: list[dict]) -> pd.DataFrame:
    rows = pd.DataFrame([{col: f[col] for col in cp.MPV_COLUMNS} for f in fits])
    return pd.concat([geo.all_channels().iloc[:len(fits)].reset_index(drop=True), rows], axis=1)

# fit options parse validate and refuse
def test_fit_options_parse_validate_and_refuse():
    # --fit[a,b] (one range: Landau (x) Gauss) and --bin N: argparse types, the PowerShell spelling through the
    # runner, JSON-friendly settings in fit_settings (default None), every rule with its reason (before any IO), and
    # a run that fits no MPV (-only without an MPV unit) refuses them in the run and in the plan.
    import runner

    args = _args("--fit", "[150,700]", "--bin", "22")
    assert args.fit_ranges == [(150.0, 700.0)] and args.fit_bins == [22]
    assert _args("-fit", "150,700").fit_ranges == [(150.0, 700.0)] and _args().fit_ranges is None
    text = _parse_error("--fit", "[100,300],[400,700]")
    assert "the MPV fit is a Landau(x)Gauss fit: one range" in text, text
    assert "one positive integer" in _parse_error("--fit", "[150,700]", "--bin", "2.5")
    # PowerShell splits -fit[150,700] at the comma; runner.main joins it again (fitting.normalize_range_args)
    ns = runner.build_parser().parse_args(fitting.normalize_range_args(["calib", "-fit[150", "700]", "-bin", "22",
                                                                        "-test", "-only", "mpv"]))
    assert (ns.fit_ranges, ns.fit_bins, ns.test, ns.only) == ([(150.0, 700.0)], [22], True, ("mpv_csv", "mpv_plot"))
    assert "calib" not in runner.BROKEN_STAGES
    # settings: [[a, b]] / [N] lists, part of fit_settings (workers, cache, provenance); default None
    s = _settings(fit_ranges=args.fit_ranges, fit_bins=args.fit_bins)
    assert s["fit_ranges"] == [[150.0, 700.0]] and s["fit_bins"] == [22]
    fs = cp.fit_settings(s)
    assert fs["fit_ranges"] == [[150.0, 700.0]] and json.loads(json.dumps(fs)) == fs
    assert _settings()["fit_ranges"] is None and _settings()["fit_bins"] is None
    assert cp.fit_settings(_settings())["fit_ranges"] is None
    assert _settings(fit_ranges="[150, 700]", fit_bins="22")["fit_ranges"] == [[150.0, 700.0]]   # yml spellings
    for over, words in (({"fit_bins": [22]}, "--bin needs --fit"),
                        ({"fit_ranges": [[150, 700]], "fit_bins": [10, 12]}, "one value per fit range"),
                        ({"fit_ranges": [[100, 300], [400, 700]]}, "one range"),
                        ({"fit_ranges": [[-20, 700]]}, "inside mpv_hist_range [0, 2000]"),
                        ({"fit_ranges": [[150, 2500]]}, "inside mpv_hist_range [0, 2000]"),
                        ({"fit_ranges": [[700, 150]]}, "a < b"),
                        ({"fit_ranges": [[150, 700]], "fit_bins": [4]}, "mpv_min_bins = 6"),
                        ({"fit_ranges": [[150, 160]], "fit_bins": [20]}, "narrower than 1 ADC"),
                        ({"fit_ranges": [[150, 200]]}, "widen the range or give --bin N")):
        exc = _raises(io_utils.SettingsError, _settings, **over)
        assert words in str(exc) and "calibration.fit_" in str(exc), (over, str(exc))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=50)
        ctx = _ctx(tmp, csv, make_plots=True)
        assert cp.settings_for(ctx, _args("--fit", "[150,700]"))["fit_ranges"] == [[150.0, 700.0]]
        for c, argv, words in ((ctx, ("-only", "pedestal_csv", "--fit", "[150,700]"), "-only chose no MPV unit"),
                               (ctx, ("-only", "pedestal", "--bin", "22"), "calib: --bin set the final MPV fit window"),
                               (_ctx(tmp, csv), ("-only", "pedestal_csv,mpv_plot", "--fit", "[150,700]"),
                                "-no-plots drops mpv_plot")):
            for fn in (cp.run, cp.io_plan):
                exc = _raises(io_utils.SettingsError, fn, c, _units(c, _args(*argv)), [])
                assert words in str(exc), (argv, fn.__name__, str(exc))
        assert sorted(p.name for p in tmp.iterdir()) == ["mu"]              # refused before any IO


# fit window equal to the automatic one and the default never enters it
def test_fit_window_equal_to_the_automatic_one_and_the_default_never_enters_it():
    # --fit[a,b] with a channel's automatic window [mpv_fit_low, mpv_fit_high] fits the same bins from the same start
    # values: every table column, the parameters and the drawn window are identical, bit for bit.  The default (no
    # --fit; also task settings without the keys) never calls the window code.
    base = cp.fit_channel_mpv(_mip_task())
    assert base["mpv_status"] == "OK" and base["fit_edges"] is None and base["fit_counts"] is None
    same = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[base["mpv_fit_low"], base["mpv_fit_high"]]])))
    for col in cp.MPV_COLUMNS:
        a, b = base[col], same[col]
        assert a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b)), (col, a, b)
    assert np.array_equal(base["params"], same["params"]) and base["window"] == same["window"]
    assert np.array_equal(base["final_counts"], same["final_counts"]) and same["fit_edges"] is None
    boom = AssertionError("the default fit must not use the --fit window code")
    with patch.object(cp, "window_bins", side_effect=boom), patch.object(cp, "_fit_user_window", side_effect=boom):
        plain = cp.fit_channel_mpv(_mip_task())
        task = _mip_task()
        task["settings"] = {k: v for k, v in task["settings"].items() if k not in cp.FIT_WINDOW_KEYS}
        bare = cp.fit_channel_mpv(task)
    for r in (plain, bare):
        assert r["mpv_status"] == "OK" and r["mpv"] == base["mpv"] and r["mpv_error"] == base["mpv_error"]
        assert np.array_equal(r["params"], base["params"]) and r["window"] == base["window"]


# fit range and bins on a synthetic mpv histogram
def test_fit_range_and_bins_on_a_synthetic_mpv_histogram():
    # The window [a, b] for the channel: page bins with centres inside, or --bin N equal bins counted exactly from the
    # 1-ADC histogram; only bins fully above the cut; too few bins -> LOW_STATS with the reason; the rough fit only
    # seeds (failed / raising / too few rough bins -> start values from the histogram, the same optimum); an exception
    # in the final stage is the final stage's; a failed fit keeps its parameters but has no curve (points only).
    task = _mip_task()
    r = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[200, 800]])))
    assert r["mpv_status"] == "OK" and (r["mpv_fit_low"], r["mpv_fit_high"]) == (200.0, 800.0)
    assert r["window"] == (10, 39) and r["mpv_ndf"] == 30 - 4 and r["fit_edges"] is None    # 20-ADC page bins
    assert abs(r["mpv"] - _true_peak()) < 5 * r["mpv_error"] + 2.0
    # --bin 24: 24 bins of 25 ADC on [200, 800], counted from the exact spectrum (x > cut = 5 x 10 ADC)
    rb = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[200, 800]], fit_bins=[24])))
    assert rb["mpv_status"] == "OK" and np.array_equal(rb["fit_edges"], np.linspace(200.0, 800.0, 25))
    x = np.arange(task["v0"], task["v0"] + len(task["counts"]), dtype=float) - task["pedestal"]
    above = x > 50.0
    direct = np.histogram(x[above], bins=rb["fit_edges"], weights=task["counts"][above].astype(float))[0]
    assert np.array_equal(rb["fit_counts"], direct) and rb["n_mpv_fit_entries"] == int(direct.sum())
    assert rb["mpv_ndf"] == 24 - 4 and rb["window"] == (0, 23)
    assert rb["mpv"] != r["mpv"] and abs(rb["mpv"] - r["mpv"]) < 3 * r["mpv_error"]   # same peak, other bins
    # a range that starts below the cut: only the bins fully above it enter
    rc = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[20, 700]], fit_bins=[68])))
    assert rc["window"][0] == 3 and (rc["mpv_fit_low"], rc["mpv_fit_high"]) == (50.0, 700.0)
    rl = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[0, 60]], fit_bins=[6])))
    assert rl["mpv_status"] == "LOW_STATS" and np.isnan(rl["mpv"]) and rl["mpv_rough_status"] == "NOT_RUN"
    assert "--fit [0, 60]: 1 bin(s) fully above the cut" in rl["mpv_fit_message"] and rl["fit_edges"] is not None
    # the rough fit only seeds: when it fails or raises, the histogram in the window gives the start values
    window = [[200, 800]]

    # fit stand-in that always raises
    def boom(*a, **k):
        raise ValueError("synthetic")

    for fake, words in ((_failed(params=(np.nan,) * 4, message="Optimal parameters not found"),
                         "rough fit: Optimal parameters not found"),
                        (boom, "rough fit: exception ValueError: synthetic")):
        with patch.object(cp, "fit_neyman", fake):
            rf = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=window)))
        assert rf["mpv_status"] == "OK" and rf["mpv_rough_status"] == "FAILED", rf["mpv_fit_message"]
        assert f"start values from the histogram in --fit [200, 800] ({words})" in rf["mpv_fit_message"]
        assert abs(rf["mpv"] - r["mpv"]) < 0.01 * r["mpv_error"]              # same bins: the same optimum
    coarse = {"mpv_rough_bin_width": 400.0}                                   # 4 rough bins above the cut
    assert cp.fit_channel_mpv(_mip_task(settings=_settings(**coarse)))["mpv_status"] == "LOW_STATS"   # automatic
    rr = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=window, **coarse)))
    assert rr["mpv_status"] == "OK" and rr["mpv_rough_status"] == "NOT_RUN"
    assert "(rough fit: too few bins)" in rr["mpv_fit_message"] and abs(rr["mpv"] - r["mpv"]) < 0.01 * r["mpv_error"]
    with patch.object(cp, "langaus_peak", boom):                             # final stage raises after no rough fit
        rx = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=window, **coarse)))
    assert rx["mpv_status"] == "FIT_FAILED" and rx["mpv_fit_status"] == "FAILED" and rx["mpv_rough_status"] == "NOT_RUN"
    assert np.isnan(rx["mpv"]) and rx["mpv_fit_message"].endswith("; exception ValueError: synthetic")
    # a window away from the peak: the fit converges, its peak lies outside -> no MPV, no curve, points only
    far = cp.fit_channel_mpv(_mip_task(settings=_settings(fit_ranges=[[900, 1500]])))
    assert far["mpv_status"] == "PEAK_OUTSIDE_WINDOW" and np.isnan(far["mpv"])
    _assert_no_fit_values(far)                                               # R3: NaN parameters, params None
    model, xs = cp.binned_langaus(20.0, cp.langaus_config(_settings())), np.linspace(0.0, 2000.0, 501)
    assert cp.mpv_panel_curve(far, far, model, xs) is None
    # a failed row that still holds its parameters: no curve either (finite-mpv check)
    old = {**far, "params": np.array([1000.0, 40.0, 20000.0, 60.0])}
    assert cp.mpv_panel_curve(old, old, model, xs) is None
    inside, outside = cp.mpv_panel_curve(r, r, model, xs)
    assert np.isfinite(inside).any() and np.isnan(inside[xs < 200.0]).all() and np.isfinite(outside[xs < 200.0]).all()


# mpv pdf with bins and a failed fit draws points only
def test_mpv_pdf_with_bins_and_a_failed_fit_draws_points_only():
    # mpv_fits.pdf (2 channels, real rendering): with --bin the fitted bins are points scaled to the page bins
    # (entries x 20 / 25), the title names the window; a failed fit (optimizer not converged) shows the points only, no
    # curve.  A default page has no such points and no window line.
    s = _settings(fit_ranges=[[200, 800]], fit_bins=[24])
    fits = [cp.fit_channel_mpv(_mip_task(settings=s))]
    with patch.object(cp, "fit_poisson_scaled", _failed(params=(420.0, 40.0, 20000.0, 60.0))):
        fits.append(cp.fit_channel_mpv(_mip_task(settings=s)))
    assert [f["mpv_status"] for f in fits] == ["OK", "FIT_FAILED"]
    _assert_no_fit_values(fits[1])                                          # R3: no parameters of the failed fit
    seen, titles = {}, []
    real_write, real_maps = cp._PanelPages.write, cp._summary_maps

    # spy of _PanelPages.write: records the points of every filled panel
    def spy_write(self, pdf, n_items, fill):
        # fill one panel, then record its inside points and scaled points
        def spy(panel, index):
            ok = fill(panel, index)
            if index < n_items:
                points = panel.get("points")
                seen[index] = (len(panel["inside"].get_xdata()), None if points is None else
                               np.asarray(points.get_ydata(), float))
            return ok
        return real_write(self, pdf, n_items, spy)

    # spy of _summary_maps: records the page titles
    def spy_maps(pdf, table, columns, title):
        titles.append(title)
        return real_maps(pdf, table, columns, title)

    with tempfile.TemporaryDirectory() as tmp, patch.object(cp._PanelPages, "write", spy_write), \
            patch.object(cp, "_summary_maps", spy_maps):
        path = cp.plot_mpv(Path(tmp) / "mpv_fits.pdf", _mpv_row_table(fits), fits, s, "synthetic")
        assert path.exists() and path.stat().st_size > 1000
        assert seen[0][0] > 0 and seen[1][0] == 0                            # curve only for the fit with an MPV
        for i in (0, 1):
            assert np.allclose(seen[i][1], fits[i]["fit_counts"] * 20.0 / 25.0)
        assert "final fit window of every channel: --fit [200, 800] ADC --bin 24: 24 equal bins of 25 ADC" in titles[0]
        default = [cp.fit_channel_mpv(_mip_task())]
        seen.clear()
        cp.plot_mpv(Path(tmp) / "default.pdf", _mpv_row_table(default), default, _settings(), "synthetic")
        assert seen[0][0] > 0 and seen[0][1] is None and "final fit window" not in titles[-1]

# failed fits keep no parameters in the table
def test_failed_fits_keep_no_parameters_in_the_table():
    # R3 (Lei 2026-10-02): a failed MPV fit (FIT_FAILED: optimizer not converged; PEAK_OUTSIDE_WINDOW) puts NaN
    # parameter / chi2 columns into the table, the status and message columns say why.
    task = _mip_task()
    ok = cp.fit_channel_mpv(task)
    with patch.object(cp, "fit_poisson_scaled", _failed(params=(420.0, 40.0, 20000.0, 60.0))):
        failed = cp.fit_channel_mpv(_mip_task())
    with patch.object(cp, "langaus_peak", return_value=-1.0e6):
        outside = cp.fit_channel_mpv(_mip_task())
    fits = [ok, failed, outside]
    assert [f["mpv_status"] for f in fits] == ["OK", "FIT_FAILED", "PEAK_OUTSIDE_WINDOW"]
    assert failed["mpv_fit_status"] == "NOT_CONVERGED" and "ITERATIONS REACHED LIMIT" in failed["mpv_fit_message"]
    assert outside["mpv_fit_status"] == "CONVERGED" and outside["mpv_covariance_status"] == "NOT_COMPUTED"
    rows = pd.DataFrame([{col: f[col] for col in cp.MPV_COLUMNS} for f in fits])
    table = pd.concat([geo.all_channels().reset_index(drop=True), rows], axis=1)    # other channels: no MPV part
    values = list(cp.FIT_VALUE_COLUMNS)
    assert table.loc[0, values].notna().all() and table.loc[[1, 2], values].isna().all().all()
    assert list(table.loc[:2, "mpv_status"]) == ["OK", "FIT_FAILED", "PEAK_OUTSIDE_WINDOW"]


# fit window end to end provenance and from saved
def test_fit_window_end_to_end_provenance_and_from_saved():
    # calib --fit[200,800] --bin 24 on a synthetic beam csv: -from-saved (no csv read) == the csv run byte for byte;
    # the pedestal (not a fit) is unchanged, the MPVs move; the window is recorded wherever the fit settings are
    # (settings, settings_all, the cache record) and changes settings_hash; -from-saved -only mpv_plot draws exactly
    # the fitted bins of the window.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv")
        ctx = _ctx(tmp / "saved", csv)
        default = cio.load_calibration(_run(ctx), float_precision="round_trip")
        prov0 = io_utils.read_json(cp.provenance_json(cp.output_csv(ctx)))
        assert prov0["settings"]["fit_ranges"] is None and prov0["mpv_fit_window"].startswith("automatic")
        flags = ("--fit", "[200,800]", "--bin", "24")
        with patch.object(io_utils, "iter_csv", _no_csv_reading), redirect_stdout(io.StringIO()) as log_text:
            saved = _run(ctx, *flags, from_saved=True)
        said = log_text.getvalue()                                            # the run says which window it used
        assert "MPV final fit window of every channel: --fit [200, 800] ADC --bin 24: 24 equal bins of 25 ADC" in said
        assert "come from the user window" in said and "restores the automatic window" in said
        ref_ctx = _ctx(tmp / "reference", csv)
        assert saved.read_bytes() == _run(ref_ctx, *flags).read_bytes()
        table = cio.load_calibration(saved, float_precision="round_trip")
        ok = (table["mpv_status"] == "OK").to_numpy()
        assert ok.sum() == 2 * len(MIP_BARS) and (default.loc[ok, "mpv_status"] == "OK").all()
        assert table["pedestal"].equals(default["pedestal"])
        assert table["pedestal_sigma"].equals(default["pedestal_sigma"])
        assert (table.loc[ok, "mpv"] != default.loc[ok, "mpv"]).all()
        assert (table.loc[ok, "mpv_fit_low"] == 200.0).all() and (table.loc[ok, "mpv_fit_high"] == 800.0).all()
        assert (table.loc[ok, "mpv_ndf"] == 20).all()
        prov = io_utils.read_json(cp.provenance_json(saved))
        record = io_utils.read_json(cp.provenance_json(cp.histogram_cache(ref_ctx)))
        for where in (prov["settings"], prov["settings_all"], record["settings"]):
            assert where["fit_ranges"] == [[200.0, 800.0]] and where["fit_bins"] == [24], where
        assert prov["settings_hash"] != prov0["settings_hash"]
        assert prov["mpv_fit_window"] == "--fit [200, 800] ADC --bin 24: 24 equal bins of 25 ADC"
        # -from-saved -only mpv_plot: the fitted bins of the window, nothing written but the pdf
        shown = {}

        # stand-in of plot_mpv: records the fits, settings and statuses shown
        def fake_mpv(path, tab, fits, settings, label):
            shown.update(fits=fits, settings=settings, status=tab["mpv_status"].tolist())
            return Path(path)

        stamp = saved.read_bytes()
        with patch.object(cp, "plot_mpv", fake_mpv), patch.object(io_utils, "iter_csv", _no_csv_reading):
            assert _run(_ctx(tmp / "saved", csv, make_plots=True), "-only", "mpv_plot", *flags, from_saved=True) is None
        assert saved.read_bytes() == stamp
        assert shown["settings"]["fit_ranges"] == [[200.0, 800.0]] and shown["settings"]["fit_bins"] == [24]
        assert shown["status"] == table["mpv_status"].tolist()
        cache = cp.read_histogram_cache(cp.histogram_cache(ctx))
        ordered = geo.all_channels().merge(table, on=cio.KEY_COLUMNS, how="left")
        s = cp.settings_for(ctx, _args(*flags))
        fits = cp.run_mpv_fits(cp.mpv_tasks(cache["h_all"], ordered["pedestal"].to_numpy(float),
                                            ordered["pedestal_sigma"].to_numpy(float), s), 1)
        for f, g in zip(fits, shown["fits"]):
            assert f["window"] == g["window"]
            for key in ("final_counts", "fit_edges", "fit_counts", "params"):
                assert (f[key] is None and g[key] is None) or np.array_equal(f[key], g[key]), key
        assert sum(g["fit_edges"] is not None for g in shown["fits"]) == 384      # every channel with a pedestal


# ----------------------------------------------------------------------------------------------
# --output: the two pdfs are the final results; tables, provenance and cache are intermediates
# ----------------------------------------------------------------------------------------------
NOT_SAVED = "final results not saved (no --output): scratch copies in {}; use --output [names] to save them"


# cp.run with pdf stand-ins; returns (pdf paths in write order, console text)
def _run_pdfs(ctx, *argv, from_saved=False) -> tuple[list[Path], str]:
    # cp.run with stand-ins of plot_pedestals / plot_mpv that write a small file where the stage tells them to (the
    # real rendering is the slow test's job): (the pdf paths in the order they were written, the console text).
    written = []

    # pdf stand-in factory: writes content where the stage asks
    def writer(content: bytes):
        # write content to path and record the path
        def write(path, *a):
            io_utils.ensure_writable(Path(path)).write_bytes(content)
            written.append(Path(path))
            return Path(path)
        return write

    with patch.object(cp, "plot_pedestals", writer(b"%PDF pedestal")), \
            patch.object(cp, "plot_mpv", writer(b"%PDF mpv")), redirect_stdout(io.StringIO()) as text:
        cp.run(ctx, _units(ctx, _args(*argv, from_saved=from_saved)), [])
    return written, text.getvalue()


# every file under root, relative and sorted
def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) if root.exists() else []


# output none fullset saves no pdf but every intermediate
def test_output_none_fullset_saves_no_pdf_but_every_intermediate():
    # Full set without --output: the pdfs are drawn into cache/pedestal_mpv/unsaved/ only (emptied at the start of
    # every run), nothing goes to results/, ONE line says so; the table + provenance and the histogram cache are written
    # as always, so -from-saved and the later stages (ctx.calibration_file) still work.  --output alone saves the usual
    # names.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=300)
        ctx = _ctx(tmp, csv, make_plots=True)
        scratch = tmp / "cache" / "pedestal_mpv" / "unsaved"
        scratch.mkdir(parents=True)
        (scratch / "earlier_run.pdf").write_bytes(b"x")
        written, text = _run_pdfs(ctx)
        assert written == [scratch / "pedestal_fits.pdf", scratch / "mpv_fits.pdf"]
        assert _files(scratch) == ["mpv_fits.pdf", "pedestal_fits.pdf"]          # the earlier copy is gone
        assert (scratch / "pedestal_fits.pdf").read_bytes() == b"%PDF pedestal"
        assert not (tmp / "results").exists()
        assert text.count("final results not saved") == 1 and NOT_SAVED.format(scratch) in text, text
        assert "final result pdf" not in text and "WARNING" not in text
        out, npz = cp.output_csv(ctx), cp.histogram_cache(ctx)
        for record, member in ((cp.provenance_json(out), out), (cp.provenance_json(npz), npz)):
            assert io_utils.product_set_valid(record, None, members=[member]) == (True, "ok")
        assert ctx.calibration_file() == out
        assert (cio.load_calibration(ctx.calibration_file())["mpv_status"] == "OK").sum() == 2 * len(MIP_BARS)
        # -from-saved -only plots (no csv read): both pdfs again into the scratch folder, no table rewritten
        stamp = out.stat().st_mtime_ns
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            written, text = _run_pdfs(ctx, "-only", "plots", from_saved=True)
        assert written == [scratch / "pedestal_fits.pdf", scratch / "mpv_fits.pdf"]
        assert NOT_SAVED.format(scratch) in text
        assert out.stat().st_mtime_ns == stamp and not (tmp / "results").exists()
        # --output alone: the usual names in results/fullset/pedestal_mpv/, each destination logged, no warning
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            written, text = _run_pdfs(_ctx(tmp, csv, make_plots=True, outputs=[]), "-only", "plots", from_saved=True)
        assert written == [cp.pedestal_pdf(ctx), cp.mpv_pdf(ctx)] and "not saved" not in text and "WARNING" not in text
        for key, path in zip(("pedestal_plot", "mpv_plot"), written):
            assert f"final result pdf {key}: {path} [default name (--output)]" in text, text
        assert _files(tmp / "results") == ["fullset/pedestal_mpv/mpv_fits.pdf",
                                           "fullset/pedestal_mpv/pedestal_fits.pdf"]
        # -no-plots: no final result, the scratch folder is emptied, no 'not saved' line; the table is written
        _, text = _run_pdfs(_ctx(tmp, csv))
        assert _files(scratch) == [] and "final results not saved" not in text and out.stat().st_mtime_ns != stamp


# output names match results by name
def test_output_names_match_results_by_name():
    # --output names are matched by result: the default file name, the result key + .pdf, or KEY=FILE; naming one
    # pdf saves only that one (the other goes to the scratch copy); a name of no result of this run is ignored with a
    # WARNING; KEY=FILE of the wrong type stops before any IO; -no-plots has no final result.  No provenance record next
    # to a pdf (they have none).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=300)
        d = tmp / "results" / "fullset" / "pedestal_mpv"
        scratch = tmp / "cache" / "pedestal_mpv" / "unsaved"
        names = ["pedestal_plot=ped.pdf", "mpv_plot=mpv.pdf"]
        written, text = _run_pdfs(_ctx(tmp, csv, make_plots=True, outputs=names))
        assert written == [d / "ped.pdf", d / "mpv.pdf"] and "WARNING" not in text and "not saved" not in text
        assert (d / "ped.pdf").read_bytes() == b"%PDF pedestal" and (d / "mpv.pdf").read_bytes() == b"%PDF mpv"
        assert f"final result pdf pedestal_plot: {d / 'ped.pdf'} [--output name]" in text
        assert f"final result pdf mpv_plot: {d / 'mpv.pdf'} [--output name]" in text
        written, text = _run_pdfs(_ctx(tmp, csv, make_plots=True, outputs=["mpv_fits.pdf"]))
        assert written == [scratch / "pedestal_fits.pdf", d / "mpv_fits.pdf"]
        assert "final results not saved (not named in --output)" in text, text
        written, _ = _run_pdfs(_ctx(tmp, csv, make_plots=True, outputs=["pedestal_plot.pdf"]), "-only", "pedestal")
        assert written == [d / "pedestal_plot.pdf"]
        written, text = _run_pdfs(_ctx(tmp, csv, make_plots=True, outputs=["mpv_fits.pdf"]), "-only", "pedestal_plot")
        assert written == [scratch / "pedestal_fits.pdf"]
        assert "WARNING: pedestal_mpv: --output names that name no result here, ignored: mpv_fits.pdf" in text, text
        before = _files(tmp / "calibration")
        _raises(io_utils.SettingsError, _run_pdfs, _ctx(tmp, csv, make_plots=True, outputs=["mpv_plot=m.csv"]),
                contains="is a pdf file")
        assert _files(tmp / "calibration") == before
        written, text = _run_pdfs(_ctx(tmp, csv, make_plots=False, outputs=["x.pdf"]))
        assert written == [] and "ignored: x.pdf (results: none)" in text, text
        assert not [p for p in (tmp / "results").rglob("*") if p.name.endswith(".provenance.json")]


# output test mode without names writes test pdf and test1 pdf
def test_output_test_mode_without_names_writes_test_pdf_and_test1_pdf():
    # -test without --output: results/test/test.pdf (the 1st pdf of the run) and test1.pdf (the 2nd): a run never
    # overwrites its own pdf, the next test run overwrites the same names (a run with one pdf: test.pdf only); nothing
    # in results/test/pedestal_mpv/; -test --output: the usual names there.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=300)
        r = tmp / "results" / "test"
        ctx = _ctx(tmp, csv, test=True, make_plots=True)
        written, text = _run_pdfs(ctx)
        assert written == [r / "test.pdf", r / "test1.pdf"] and _files(r) == ["test.pdf", "test1.pdf"]
        assert (r / "test.pdf").read_bytes() == b"%PDF pedestal" and (r / "test1.pdf").read_bytes() == b"%PDF mpv"
        assert f"final result pdf pedestal_plot: {r / 'test.pdf'} [test name]" in text and "not saved" not in text
        assert f"final result pdf mpv_plot: {r / 'test1.pdf'} [test name]" in text
        assert cp.output_csv(ctx).exists() and cp.histogram_cache(ctx).exists()         # the intermediates as always
        assert not (tmp / "cache" / "test" / "pedestal_mpv" / "unsaved").exists()
        # the next run overwrites test.pdf: a run with one pdf (-only mpv) writes test.pdf only
        (r / "test.pdf").write_bytes(b"old")
        written, _ = _run_pdfs(ctx, "-only", "mpv")
        assert written == [r / "test.pdf"] and (r / "test.pdf").read_bytes() == b"%PDF mpv"
        assert (r / "test1.pdf").read_bytes() == b"%PDF mpv"                              # not part of that run
        written, _ = _run_pdfs(_ctx(tmp, csv, test=True, make_plots=True, outputs=[]))
        assert written == [r / "pedestal_mpv" / "pedestal_fits.pdf", r / "pedestal_mpv" / "mpv_fits.pdf"]
        assert not [p for p in r.rglob("*") if p.name.endswith(".provenance.json")]


# output plan lists the destinations without touching the scratch folder
def test_output_plan_lists_the_destinations_without_touching_the_scratch_folder():
    # The dry-run plan (also what the run log reads after a run: it calls io_plan) lists the pdfs at the destinations
    # of the run's output plan, in the run's order: the scratch path (not saved) without --output, the named ones with
    # --output, test.pdf / test1.pdf in -test, only the chosen units with -only.  It never empties the scratch folder,
    # registers no plan and creates nothing.
    # the result destinations of the dry-run plan and its console text
    def results(ctx, *argv):
        with redirect_stdout(io.StringIO()) as text:
            plan = _plan(ctx, *argv)
        return [(i["result"], i["path"], i["saved"]) for i in plan if "result" in i], text.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=200)
        scratch = tmp / "cache" / "pedestal_mpv" / "unsaved"
        scratch.mkdir(parents=True)
        (scratch / "pedestal_fits.pdf").write_bytes(b"scratch copy of the last run")
        d, r = tmp / "results" / "fullset" / "pedestal_mpv", tmp / "results" / "test"
        outputs_mod.reset()
        got, _ = results(_ctx(tmp, csv, make_plots=True))
        assert got == [("pedestal_plot", scratch / "pedestal_fits.pdf", False),
                       ("mpv_plot", scratch / "mpv_fits.pdf", False)]
        got, _ = results(_ctx(tmp, csv, make_plots=True, outputs=["pedestal_plot=p.pdf"]))
        assert got == [("pedestal_plot", d / "p.pdf", True), ("mpv_plot", scratch / "mpv_fits.pdf", False)]
        got, _ = results(_ctx(tmp, csv, make_plots=True, test=True))
        assert got == [("pedestal_plot", r / "test.pdf", True), ("mpv_plot", r / "test1.pdf", True)]
        assert results(_ctx(tmp, csv, make_plots=True), "-only", "mpv")[0] == [("mpv_plot", scratch / "mpv_fits.pdf",
                                                                                 False)]
        assert results(_ctx(tmp, csv))[0] == []
        assert (scratch / "pedestal_fits.pdf").read_bytes() == b"scratch copy of the last run"
        assert outputs_mod.plans() == [] and not (tmp / "results").exists() and not (tmp / "calibration").exists()


# ----------------------------------------------------------------------------------------------
# slow
# ----------------------------------------------------------------------------------------------
# pdf bytes without the creation date
def _pdf_body(path: Path) -> bytes:
    import re

    return re.sub(rb"/CreationDate \([^)]*\)", b"", Path(path).read_bytes())


# slow pdfs from saved equal the csv run
def test_slow_pdfs_from_saved_equal_the_csv_run():
    # Real rendering: a csv run draws both pdfs; -from-saved -only plots (no csv read) gives byte-identical pdfs
    # (except the creation date).
    if not SLOW:
        _skip("set LDMX_SLOW_TESTS=1 (renders 4 x 384 panels, about 2 minutes)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=400)
        ctx = _ctx(tmp, csv, make_plots=True, outputs=[])               # --output: the usual names
        _run(ctx)
        first = {p.name: _pdf_body(p) for p in (cp.pedestal_pdf(ctx), cp.mpv_pdf(ctx))}
        assert all(len(body) > 10_000 for body in first.values())
        with patch.object(io_utils, "iter_csv", _no_csv_reading):
            _run(ctx, "-only", "plots", from_saved=True)
        again = {p.name: _pdf_body(p) for p in (cp.pedestal_pdf(ctx), cp.mpv_pdf(ctx))}
    assert first == again


# -no-json: no record next to the tables in calibration/; the histogram cache keeps its record, -from-saved works
def test_no_json_writes_no_record_next_to_the_tables():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = _beam_csv(tmp / "mu" / "mu.csv", n_events=600)
        ctx = _ctx(tmp, csv)
        ctx.write_json = False
        io_utils.set_json_records(False, [tmp / "data", tmp / "calibration", tmp / "results"], [tmp / "cache"])
        try:
            part = _run(ctx, "-only", "pedestal_csv")
            assert part == cp.part_csv(ctx, "pedestal") and part.exists() and not cp.provenance_json(part).exists()
            cache = cp.histogram_cache(ctx)
            assert cache.exists() and cp.provenance_json(cache).exists()      # cache records are always written
            first = part.read_bytes()
            assert _run(ctx, "-only", "pedestal_csv", from_saved=True) == part and part.read_bytes() == first
            assert not list((tmp / "calibration").rglob("*.json"))
        finally:
            io_utils.set_json_records(True)


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["calibration"]))
