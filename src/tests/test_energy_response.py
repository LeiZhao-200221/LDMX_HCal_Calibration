"""Tests of ldmx_hcal.energy.energy_response and the fitting.py range helpers (synthetic data, a few seconds).

    python -X utf8 src/tests/run_all.py energy_response      (from the data_analysis folder)

Covered: the spectrum fits (peak scan, two-pass, -fit[a,b] single / double Gaussian with -bin), point
errors, linearity (one line per particle, -linfit <particle>[a,b], the per-scale defaults adc_only / PureCI / s),
resolution (electrons fitted, pions drawn), e/h (filled / hollow points), -normalerror, the settings rules and the
figures.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common import fitting  # noqa: E402
from ldmx_hcal.common.io_utils import SettingsError  # noqa: E402
from ldmx_hcal.energy import energy_response as er  # noqa: E402

E5 = np.array([0.3, 0.4, 0.5, 1.0, 4.0])
EE = np.array([0.3, 0.4, 0.5, 1.0, 4.0])          # electron runs
EP = np.array([0.5, 1.0, 3.0, 4.0])               # pion runs (3 GeV has no electron run)
E_ALL = np.r_[EE, EP]
PARTS = ["electron"] * 5 + ["pion"] * 4
MU_ALL = np.r_[30.0 * EE + 1.0, 24.0 * EP + 0.5]  # pion response 0.8 of the electron's
ERR_ALL = np.full(E_ALL.size, 0.2)


# Gaussian peak plus a low-energy exponential tail (all >= 0)
def _electron_like(rng, n=60000, mu=20.0, sigma=4.0, tail=0.15):
    core = rng.normal(mu, sigma, int(n * (1 - tail)))
    low = rng.exponential(mu / 2, int(n * tail))
    v = np.concatenate([core, low])
    return v[v >= 0]


# raises AssertionError unless call() raises one of the exceptions
def _refused(call, exceptions=(ValueError,), words: str = ""):
    try:
        call()
    except exceptions as exc:
        assert words in str(exc), (words, str(exc))
        return str(exc)
    raise AssertionError(f"must be refused ({words})")


# parser with the shared energy-response flags
def _parser():
    p = argparse.ArgumentParser(allow_abbrev=False)
    er.add_fit_arguments(p)
    return p


# namespace of a command line (pieces joined as runner.main does)
def _flags(*argv):
    return _parser().parse_args(fitting.normalize_range_args(list(argv)))


# the argparse error message of a refused command line ('' when it parses)
def _parse_error(*argv) -> str:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            _flags(*argv)
    except SystemExit:
        return err.getvalue()
    return ""


# just enough of a RunContext for er.settings: the yml section energy_response
class _YmlCtx:

    # the yml section
    def __init__(self, section):
        self.section = section

    # yml section by name
    def stage_cfg(self, name):
        return self.section if name == er.CONFIG_SECTION else {}


# ----------------------------------------------------------------------------------------------
# spectrum fits
# ----------------------------------------------------------------------------------------------
# peak scan on a Gaussian: mu, sigma, one peak
def test_peak_scan_recovers_gaussian():
    fit = er.peak_scan_fit(np.random.default_rng(1).normal(20.0, 4.0, 50000))
    assert fit["status"] == "ok" and fit["ok"]
    assert abs(fit["mu"] - 20.0) < 0.2 and abs(fit["sigma"] - 4.0) < 0.3
    assert fit["n_detected_peaks"] == 1
    assert math.isclose(fit["sigma_over_mu"], fit["sigma"] / fit["mu"])


# empty / negative / tiny inputs give a status, never an exception
def test_peak_scan_degenerate_inputs_never_raise():
    empty = er.peak_scan_fit([])
    assert empty["status"] == "no_data" and not empty["ok"] and math.isnan(empty["mu"])
    negative = er.peak_scan_fit([-5.0, -3.0])
    assert negative["status"] == "empty_histogram" and math.isnan(negative["mu"])
    for values in ([1.0], [1.0, 1.0, 2.0], [np.nan, np.inf, 4.0]):
        fit = er.peak_scan_fit(values)
        assert fit["n_values"] == len([v for v in values if np.isfinite(v)]) and "status" in fit


# two-pass core fit: mu, sigma and the mu1 +/- 2 sigma1 window rule
def test_two_pass_recovers_gaussian_and_window_rule():
    values = _electron_like(np.random.default_rng(3), n=80000, mu=50.0, sigma=6.0)
    fit = er.two_pass_gauss_fit(values)
    assert fit["status"] == "ok" and fit["ok"]
    assert abs(fit["mu"] - 50.0) < 0.3 and abs(fit["sigma"] - 6.0) < 0.3
    upper = max(3.0, float(np.percentile(values, 99.8)) * 1.08)
    assert fit["hist_upper"] == upper and math.isclose(fit["bin_width"], upper / 90)
    assert fit["pass1_bin_count"] == int(np.count_nonzero(fit["counts"] >= 10))
    assert fit["fit_low"] == max(0.0, fit["pass1_mu"] - 2.0 * fit["pass1_sigma"])
    assert fit["fit_high"] == min(upper, fit["pass1_mu"] + 2.0 * fit["pass1_sigma"])
    core = (fit["counts"] >= 10) & (fit["centers"] >= fit["fit_low"]) & (fit["centers"] <= fit["fit_high"])
    assert fit["core_bin_count"] == int(np.count_nonzero(core)) and fit["ndf"] == fit["core_bin_count"] - 3
    assert abs(fit["window_gaussian_coverage"] - 0.9544997361) < 1e-9
    scan = er.peak_scan_fit(values)
    assert er.two_pass_gauss_fit(values)["initial_mu"] == scan["mu"]  # start values from the peak scan


# two-pass with few entries, a display range and another window
def test_two_pass_low_statistics_and_options():
    rng = np.random.default_rng(5)
    few = er.two_pass_gauss_fit(rng.normal(10, 2, 30))
    assert not few["ok"] and few["status"].startswith("too_few_visible_bins") and math.isnan(few["mu"])
    assert few["counts"] is not None
    none = er.two_pass_gauss_fit([])
    assert none["status"] == "no_data" and none["counts"] is None
    ranged = er.two_pass_gauss_fit(rng.normal(50, 5, 20000), display_range=(20.0, 80.0), bins=60)
    assert ranged["ok"] and ranged["edges"][0] == 20.0 and ranged["edges"][-1] == 80.0 and len(ranged["counts"]) == 60
    assert ranged["fit_low"] >= 20.0
    wide = er.two_pass_gauss_fit(rng.normal(50, 5, 20000), window_nsigma=1.0)
    assert wide["ok"] and math.isclose(wide["window_gaussian_coverage"], 0.682689492, rel_tol=1e-8)


# a failed peak scan / resolution fit gives NaN, never a fallback value
def test_a_failed_fit_gives_no_value():
    from unittest import mock

    values = np.random.default_rng(7).normal(10.0, 2.0, 20000)
    with mock.patch.object(er, "curve_fit", side_effect=RuntimeError("no convergence")):
        scan = er.peak_scan_fit(values)
    assert not scan["ok"] and scan["status"].startswith("fit_failed")
    assert math.isnan(scan["mu"]) and math.isnan(scan["sigma"]) and abs(scan["seed_mu"] - 10.0) < 0.5
    r = 0.2 / np.sqrt(E5) + 0.02
    with mock.patch.object(er, "curve_fit", lambda *a, **k: (np.array([np.inf, 1.0, 0.01]), np.eye(3))):
        res = er.resolution_fit(E5, r, r * 0.002)
    assert not res["ok"] and res["status"] == "fit_failed: non-finite parameters"
    assert not res["fitted"].any() and np.isnan(res["predicted"]).all() and math.isnan(res["a"])


# ----------------------------------------------------------------------------------------------
# point errors, linearity, resolution, e/h
# ----------------------------------------------------------------------------------------------
# sigma/sqrt(N), sigma_fit_error and gaussian prescriptions
def test_point_errors_prescriptions():
    mu, sigma, n = np.array([10.0, 20.0]), np.array([4.0, 5.0]), np.array([400, 2500])
    ci = er.point_errors(mu, sigma, n=n)
    assert np.allclose(ci["mu_error"], [0.2, 0.1]) and np.allclose(ci["ratio_error"], [0.02, 0.005])
    assert np.allclose(ci["ratio"], [0.4, 0.25])
    adc = er.point_errors(mu, sigma, n=n, sigma_error=[0.5, 0.2], prescription="sigma_fit_error")
    assert np.allclose(adc["ratio_error"], [0.05, 0.01])
    g = er.point_errors(mu, sigma, n=n, prescription="gaussian")
    want = [0.4 * np.sqrt(1 / 800 + 0.16 / 400), 0.25 * np.sqrt(1 / 5000 + 0.0625 / 2500)]
    assert np.allclose(g["ratio_error"], want)
    _refused(lambda: er.point_errors(mu, sigma, n=n, prescription="nope"))


# one line: inclusive range, every point inside, predictions over every point
def test_linearity_line_range_is_inclusive():
    mu = 30.0 * E5 - 1.0
    mu[-1] *= 0.88
    err = np.full(5, 0.05)
    low = er.linearity_line(E5, mu, err, fit_range=(0, 1.0))
    assert low["ok"] and low["fit_range"] == [0.0, 1.0]
    assert low["fitted"].tolist() == [True, True, True, True, False]  # 1 GeV included
    assert low["fitted_energies"] == "0.3;0.4;0.5;1 GeV" and math.isclose(low["slope"], 30.0)
    assert math.isclose(low["residual_fraction"][-1], -0.12, rel_tol=1e-9)
    every = er.linearity_line(E5, mu, err)
    assert every["ok"] and every["fitted"].all() and every["fit_range"] is None and every["ndf"] == 3
    mid = er.linearity_line(E5, mu, err, fit_range=(0.4, 1.0))
    assert mid["fitted"].tolist() == [False, True, True, True, False]
    near = er.linearity_line([0.3, 0.5, 1.0 + 1e-12], [9.0, 15.0, 30.0], [0.1] * 3, fit_range=(0, 1))
    assert near["fitted"].all()
    lonely = er.linearity_line([0.3, 4.0], [9.0, 100.0], [0.1, 0.1], fit_range=(0, 1))
    assert not lonely["ok"] and "fewer than 2 points (1)" in lonely["status"] and np.isnan(lonely["predicted"]).all()
    for bad in ("full", (1.0, 0.5), (0.3,), (0, np.inf)):
        _refused(lambda: er.linearity_line(E5, mu, err, fit_range=bad), words="GeV")


# a line through 0, 1, 2 points, the same energy, a NaN point
def test_linearity_line_any_number_of_points():
    assert not er.linearity_line([], [], [])["ok"]
    one = er.linearity_line([4.0], [100.0], [1.0])
    assert not one["ok"] and math.isnan(one["slope"]) and np.isnan(one["predicted"]).all()
    two = er.linearity_line([0.3, 0.5], [9.0, 15.0], [0.1, 0.1])
    assert two["ok"] and two["ndf"] == 0 and math.isclose(two["slope"], 30.0) and two["error_scale"] == 1.0
    assert not er.linearity_line([1.0, 1.0, 1.0], [30.0, 31.0, 29.0], [0.1] * 3)["ok"]  # singular
    nan_point = er.linearity_line(E5, 30 * E5, [0.1, np.nan, 0.1, 0.1, 0.1])
    assert nan_point["ok"] and not nan_point["fitted"][1]


# electron and pion points on one figure: one line per particle, never fitted together
def test_linearity_one_line_per_particle():
    both = er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS, fits={"electron": None, "pion": [0, 4]})
    assert math.isclose(both["fits"]["electron"]["slope"], 30.0) and math.isclose(both["fits"]["pion"]["slope"], 24.0)
    assert both["fits"]["electron"]["fitted"].tolist() == [True] * 5 + [False] * 4
    assert both["fits"]["pion"]["fitted"].tolist() == [False] * 5 + [True] * 4
    assert both["ok"] and both["fitted"].all() and both["status"] == "electron: ok; pion: ok"
    rows = er.linearity_rows(both)
    assert [r["particle"] for r in rows] == ["electron", "pion"]
    assert rows[0]["fit_range_gev"] == "all points" and rows[1]["fit_range_gev"] == "[0, 4]"
    assert json.loads(json.dumps(rows)) == rows
    electron_only = er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS, fits={"electron": [0, 1]})
    assert electron_only["fitted"].tolist() == [True] * 4 + [False] * 5  # pion 0.5 / 1 GeV are not fitted
    none = er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS, fits={})
    assert not none["fits"] and not none["ok"] and none["status"] == "no fit: points only"
    assert er.linearity_rows(none) == []
    missing = er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS, fits={"muon": None})
    assert not missing["ok"] and "muon" in missing["status"]
    _refused(lambda: er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS[:3]), words="particle names")


# default linearity fits per ci scale; -linfit replaces them; adcpeak (scale None) has none
def test_scale_default_linearity_fits():
    rs = er.settings()
    assert er.scale_linearity_fits(rs, "adc_only") == {}
    assert er.scale_linearity_fits(rs, "PureCI") == {"electron": [0.0, 0.999]}
    assert er.scale_linearity_fits(rs, "pureci") == {"electron": [0.0, 0.999]}
    assert er.scale_linearity_fits(rs, "s") == {"electron": None}
    assert er.scale_linearity_fits(rs, None) == {}
    assert er.scale_linearity_fits({**rs, "linearity_fits": {"pion": None}}, "s") == {"pion": None}
    assert er.scale_linearity_fits({**rs, "linearity_fits": {}}, "s") == {}
    pure = er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale="PureCI")
    assert pure["fits"]["electron"]["n_fitted"] == 3 and not pure["fitted"][3:].any()  # below 1 GeV
    s = er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale="s")
    assert s["fits"]["electron"]["n_fitted"] == 5 and list(s["fits"]) == ["electron"]
    adc = er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale="adc_only")
    assert adc["fits"] == {} and not adc["fitted"].any()


# -normalerror: errors x sqrt(chi2/ndf) when chi2/ndf > 1, the fit unchanged
def test_normalerror_scales_linearity_and_resolution_errors():
    mu = 30.0 * E5 - 1.0 + np.array([0.3, -0.2, 0.25, -0.3, 0.1])
    err = np.full(5, 0.05)
    plain = er.linearity_line(E5, mu, err)
    scaled = er.linearity_line(E5, mu, err, normal_error=True)
    for key in ("slope", "intercept", "slope_error", "intercept_error", "chi2", "ndf"):
        assert plain[key] == scaled[key], key
    ratio = scaled["chi2"] / scaled["ndf"]
    assert ratio > 1 and math.isclose(scaled["error_scale"], math.sqrt(ratio), rel_tol=1e-15)
    assert math.isclose(scaled["slope_error_scaled"], scaled["slope_error"] * math.sqrt(ratio), rel_tol=1e-12)
    assert np.allclose(scaled["covariance_scaled"], scaled["covariance"] * ratio, rtol=1e-12)
    good = er.linearity_line(E5, 30.0 * E5 - 1.0, err, normal_error=True)
    assert good["error_scale"] == 1.0
    assert er.linearity_line(E5, mu, None, weighted=False, normal_error=True)["error_scale"] == 1.0
    a, b, c = 100.0, 5.0, 0.06
    r = np.sqrt(a ** 2 / (E5 * 1000) ** 2 + b ** 2 / (E5 * 1000) + c ** 2) * (1 + np.array([.02, -.03, .01, .02, -.01]))
    res = er.resolution_fit(E5, r, r * 0.002, normal_error=True)
    assert res["ok"] and res["ndf"] == 2 and res["chi2_over_ndf"] > 1
    k = math.sqrt(res["chi2_over_ndf"])
    for name in ("a", "b", "c"):
        assert math.isclose(res[f"{name}_error_scaled"], res[f"{name}_error"] * k, rel_tol=1e-12), name
    rs = {**er.settings(), "linearity_fits": {"electron": [0, 1]}, "normal_error": True}
    via = er.linearity_from_settings(E5, mu, err, None, rs)
    assert via["normal_error"] and via["fits"]["electron"]["n_fitted"] == 4
    assert er.resolution_from_settings(E5, r, r * 0.002, rs)["normal_error"]


# three-term resolution fit: recovers a, b, c; too few points fail
def test_resolution_fit_recovers_and_degrades():
    a, b, c = 100.0, 5.0, 0.06
    e_mev = E5 * 1000
    ratio = np.sqrt(a ** 2 / e_mev ** 2 + b ** 2 / e_mev + c ** 2)
    fit = er.resolution_fit(E5, ratio, ratio * 0.01)
    assert fit["ok"] and fit["status"] == "ok" and fit["ndf"] == 2
    for name, want in (("a", a), ("b", b), ("c", c)):
        assert math.isclose(fit[name], want, rel_tol=1e-5), name
    assert np.allclose(fit["predicted"], ratio) and fit["fitted_energies"] == "300;400;500;1000;4000 MeV"
    two = er.resolution_fit([0.3, 1.0], [0.4, 0.2], [0.01, 0.01])
    assert not two["ok"] and two["status"].startswith("too_few_points") and math.isnan(two["a"])
    low = er.resolution_fit(E5, ratio, ratio * 0.01, fit_range=(0.3, 1.0))
    assert low["ok"] and low["fitted"].tolist() == [True, True, True, True, False]
    few = er.resolution_fit(E5, ratio, ratio * 0.01, fit_range=(0.9, 1.1))
    assert not few["ok"] and few["status"] == "too_few_points: 1 electron points < 3 with 0.9 <= E <= 1.1 GeV"


# resolution: the electron points are fitted, the pion points only drawn
def test_resolution_fits_electrons_only():
    re_ = np.sqrt((0.2 / np.sqrt(EE)) ** 2 + 0.05 ** 2)
    ratio = np.r_[re_, 0.4 / np.sqrt(EP)]
    err = 0.02 * ratio
    rs = er.settings()
    res = er.resolution_from_settings(E_ALL, ratio, err, rs, particles=PARTS)
    assert res["ok"] and res["n_fitted"] == 5 and res["fit_particle"] == "electron"
    assert res["fitted"].tolist() == [True] * 5 + [False] * 4
    alone = er.resolution_from_settings(EE, re_, 0.02 * re_, rs)
    assert (res["a"], res["b"], res["c"]) == (alone["a"], alone["b"], alone["c"])
    pions = er.resolution_from_settings(E_ALL, ratio, err, rs, particles=PARTS, fit_particle="pion")
    assert pions["n_fitted"] == 4 and pions["fitted"].tolist() == [False] * 5 + [True] * 4


# e/h on one scale: filled where an electron run has the pion energy, hollow from the electron line
def test_e_over_h_filled_and_hollow_points():
    rs = er.settings()
    line = er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale="s")["fits"]["electron"]
    keys = [f"run{i}" for i in range(E_ALL.size)]
    eh = er.e_over_h(E_ALL, MU_ALL, ERR_ALL, PARTS, electron_line=line, keys=keys)
    assert eh["energies_gev"].tolist() == [0.5, 1.0, 3.0, 4.0] and eh["hadron"] == "pion"
    assert eh["filled"].tolist() == [True, True, False, True]
    for e, v in zip(eh["energies_gev"], eh["e_over_h"]):
        assert math.isclose(v, (30 * e + 1) / (24 * e + 0.5), rel_tol=1e-6), e
    rows = er.eh_rows(eh)
    assert [r["electron_source"] for r in rows] == ["electron run", "electron run", "electron line fit",
                                                    "electron run"]
    assert rows[2]["run"] == "run7" and rows[2]["note"] == "" and np.isfinite(rows[2]["e_over_h_error"])
    want = abs(eh["e_over_h"][0]) * math.hypot(0.2 / 16.0, 0.2 / 12.5)
    assert math.isclose(rows[0]["e_over_h_error"], want, rel_tol=1e-9)
    assert "every electron point" in eh["electron_line_text"]
    noline = er.e_over_h(E_ALL, MU_ALL, ERR_ALL, PARTS, electron_line=None)
    assert np.isnan(noline["e_over_h"][2]) and "no electron run" in noline["rows"][2]["note"]
    assert noline["filled"].tolist() == [True, True, False, True]
    two = er.e_over_h([1.0, 1.0, 1.0], [30.0, 32.0, 24.0], [0.3, 0.4, 0.2], ["electron", "electron", "pion"])
    assert math.isclose(two["rows"][0]["mu_electron"], 31.0) and math.isclose(two["rows"][0]["mu_electron_error"], 0.25)
    dead = er.e_over_h([1.0, 1.0], [30.0, np.nan], [0.3, 0.2], ["electron", "pion"])
    assert dead["rows"][0]["note"] == "no fitted pion peak" and not dead["filled"].any()


# ----------------------------------------------------------------------------------------------
# flags and settings
# ----------------------------------------------------------------------------------------------
# -linfit <particle>[a,b]: spellings, PowerShell pieces, several particles, repeats, errors
def test_linfit_particle_syntax():
    assert _flags().linearity_fits is None
    for argv, want in ((["-linfit", "electron[0,1]"], {"electron": [0.0, 1.0]}),
                       (["-linfit", "electron[0", "1]"], {"electron": [0.0, 1.0]}),
                       (["-linfit", "electron[0", "1]", "pion[0", "4]"], {"electron": [0, 1], "pion": [0, 4]}),
                       (["-linfit", "electron[0,1]", "pion"], {"electron": [0, 1], "pion": None}),
                       (["-linfit", "electron", "[0.3,1]"], {"electron": [0.3, 1]}),
                       (["--linfit=electron[0,1]"], {"electron": [0, 1]}),
                       (["-linfit", "Electrons"], {"electron": None}),
                       (["-linfit", "muon[1,5]"], {"muon": [1.0, 5.0]}),
                       (["-linfit", "electron[0,1]", "-linfit", "pion"], {"electron": [0, 1], "pion": None}),
                       (["-linfit", "none"], {})):
        assert _flags(*argv).linearity_fits == want, argv
    assert fitting.normalize_range_args(["ci", "-linfit", "electron[0", "1]"]) == ["ci", "-linfit", "electron[0",
                                                                                   "1]"]
    for argv, words in ((["-linfit", "[0,1]"], "name the particle"), (["-linfit", "kaon[0,1]"], "unknown particle"),
                        (["-linfit", "electron[1,0]"], "a < b"),
                        (["-linfit", "electron[0,1]", "electron"], "electron given twice"),
                        (["-linfit", "electron[0,1]", "-linfit", "electron"], "electron given twice"),
                        (["-linfit", "none", "-linfit", "pion"], "'none'")):
        assert words in _parse_error(*argv), (argv, _parse_error(*argv))
    assert er.parse_linfit("electron[0,1];pion") == {"electron": [0.0, 1.0], "pion": None}


# default settings, flags, yml values and the flag-wins rule
def test_settings_and_flags():
    s = er.settings_from_args(None, _flags())
    every = {**er.DEFAULTS, **dict.fromkeys(er.CLI_ONLY_SETTINGS)}
    assert s == every and s["normal_error"] is False and er.settings(None) == every
    assert set(er.CLI_ONLY_SETTINGS) == {"fit_ranges", "fit_bins"} and not set(er.CLI_ONLY_SETTINGS) & set(er.DEFAULTS)
    for key in er.RANGE_SETTING_KEYS:
        assert s[key] is None, key
    for spelling in ("--normalerror", "-normalerror", "--normal-error", "-normal-error"):
        assert er.settings_from_args(None, _flags(spelling))["normal_error"] is True
    s = er.settings_from_args(None, _flags("-window-nsigma", "1.5", "-min-bin-count", "3"))
    assert s["window_nsigma"] == 1.5 and s["min_bin_count"] == 3 and s["spectrum_bins"] == 90
    assert set(er.FLAG_KEYS) <= set(er.DEFAULTS) | set(er.CLI_ONLY_SETTINGS)
    assert er.FLAG_KEYS == ("fit_ranges", "fit_bins", "linearity_fits", "resolution_fit_range", "normal_error",
                            "spectrum_bins", "min_bin_count", "window_nsigma", "error_prescription")
    yml = _YmlCtx({"linearity_fits": {"Electron": [0, 1]}})
    assert er.settings_from_args(yml, _flags())["linearity_fits"] == {"electron": [0.0, 1.0]}
    assert er.settings_from_args(yml, _flags("-linfit", "pion"))["linearity_fits"] == {"pion": None}  # flag wins
    s = er.settings_from_args(None, _flags("-linfit", "electron[0", "1]", "pion", "-resfit[0.3,4]"))
    assert s["linearity_fits"] == {"electron": [0.0, 1.0], "pion": None} and s["resolution_fit_range"] == [0.3, 4.0]
    assert json.loads(json.dumps({k: s[k] for k in er.RANGE_SETTING_KEYS})) == {k: s[k] for k in er.RANGE_SETTING_KEYS}


# -fit / -bin / -resfit: JSON-friendly lists, PowerShell pieces joined
def test_range_flags_give_json_friendly_settings():
    s = er.settings_from_args(None, _flags("--fit", "[5,10]"))
    assert s["fit_ranges"] == [[5.0, 10.0]] and s["fit_bins"] is None
    s = er.settings_from_args(None, _flags("--fit", "[5,10]", "--bin", "24"))
    assert s["fit_ranges"] == [[5.0, 10.0]] and s["fit_bins"] == [24]
    ns = _flags("-fit[15", "32]", "[50", "120]", "-bin", "30,40", "-linfit", "electron[0.3", "1]", "-resfit[0.3", "4]",
                "-normalerror")
    assert ns.fit_ranges == [(15.0, 32.0), (50.0, 120.0)] and ns.linearity_fits == {"electron": [0.3, 1.0]}
    s = er.settings_from_args(None, ns)
    assert s["fit_ranges"] == [[15.0, 32.0], [50.0, 120.0]] and s["fit_bins"] == [30, 40]
    assert s["resolution_fit_range"] == [0.3, 4.0] and s["normal_error"]
    assert json.loads(json.dumps(s)) == s
    for argv, words in ((["--fit", "[5]"], "pairs"), (["--fit", "[10,5]"], "a < b"),
                        (["--fit", "[1,2],[3,4],[5,6]"], "at most 2"), (["--bin", "0"], "positive integer"),
                        (["--bin", "1.5"], "positive integer"), (["--bin", "1,2,3"], "positive integer"),
                        (["--resfit", "[0,1],[2,3]"], "at most 1"), (["--resfit", "abc"], "pairs")):
        assert words in _parse_error(*argv), (argv, _parse_error(*argv))


# negative values in every spelling; a repeated -fit / -bin / -resfit is refused; a bare -fit names the syntax
def test_range_flag_spellings_repeats_and_bare_fit():
    for argv, want in ((["--fit=-5,10"], [(-5.0, 10.0)]), (["--fit", "-5", "10"], [(-5.0, 10.0)]),
                       (["-fit", "-5,10"], [(-5.0, 10.0)]), (["-fit[-5", "10]"], [(-5.0, 10.0)]),
                       (["--fit", "[-5,10]"], [(-5.0, 10.0)]),
                       (["--fit", "-20", "-5", "0", "8"], [(-20.0, -5.0), (0.0, 8.0)]),
                       (["-fit", "5,10", "60,110"], [(5.0, 10.0), (60.0, 110.0)]),
                       (["-fit[5", "10],[10", "20]"], [(5.0, 10.0), (10.0, 20.0)])):
        joined = fitting.normalize_range_args(argv)
        assert len(joined) == 1 and joined[0].startswith("--fit="), (argv, joined)
        assert _flags(*argv).fit_ranges == want, argv
    ns = _flags("--resfit=-0.5,4", "-bin", "30", "-fit[1,9]")
    assert ns.resolution_fit_range == [-0.5, 4.0] and ns.fit_bins == [30]
    for argv, words in ((["-fit[5,10]", "-fit[20,30]"], "--fit[5,10],[20,30]"),
                        (["-bin", "30", "--bin", "40"], "--bin 30,40"),
                        (["-resfit[0.3,1]", "--resfit=0,4"], "one range only")):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                fitting.normalize_range_args(["ci"] + argv)
        except fitting.RangeFlagError as exc:
            assert exc.code == 2 and words in exc.message, (argv, exc.message)
        else:
            raise AssertionError(f"{argv}: a repeated range flag must be refused")
    for argv in (["--fit"], ["-fit"], ["-fit", "--normalerror"]):
        message = _parse_error(*argv)
        assert "needs a range: -fit[a,b]" in message and "expected one argument" not in message, (argv, message)
    for argv, words in ((["--fit", "[5,10],[5,10]"], "overlap on [5, 10]"),
                        (["--fit", "[12,28],[22,45]"], "overlap on [22, 28]")):
        assert words in _parse_error(*argv), (argv, _parse_error(*argv))
    text = " ".join(_parser().format_help().split())
    assert "bash/zsh: quote the brackets, '-fit[14,26]', or write -fit 14,26" in text and "-fit=" not in text


# -bin needs -fit and one value per range; -fit / -bin are command line only
def test_bin_without_fit_or_with_the_wrong_count_is_refused():
    cases = (({}, ["--bin", "30"], "--bin needs --fit"),
             ({}, ["--fit", "[5,10]", "--bin", "30,40"], "one value per fit range"),
             ({}, ["--fit", "[5,10]", "--bin", "3"], "needs at least 4 bins"),
             ({}, ["--fit", "[5,10],[20,30]", "--bin", "3,3"], "at least 3 bins per range and 7 in all"),
             ({"fit_bins": [30]}, [], "command-line only"),
             ({"fit_ranges": [[5, 10], [20, 30]]}, ["--bin", "30"], "command-line only"))
    for section, argv, words in cases:
        _refused(lambda: er.settings_from_args(_YmlCtx(section), _flags(*argv)), (SettingsError,), words)
    for bad, words in (({"fit_bins": [30]}, "--bin needs --fit"),
                       ({"fit_ranges": [[5, 10], [8, 20]]}, "overlap on [8, 10]"),
                       ({"fit_ranges": [[5, 10]], "fit_bins": [3]}, "needs at least 4 bins")):
        _refused(lambda: er.check_settings({**er.DEFAULTS, **bad}), (SettingsError,), words)
    for ok in (["--fit", "[5,10]", "--bin", "4"], ["--fit", "[5,10],[10,20]", "--bin", "3,4"]):
        assert er.settings_from_args(None, _flags(*ok))["fit_bins"] is not None
    _refused(lambda: er.range_spectrum_fit(np.arange(100.0), [(5, 10)], bins=[3]))
    _refused(lambda: er.range_spectrum_fit(np.arange(100.0), [(5, 10), (8, 20)]))
    assert fitting.range_options_error([(5, 10)], [3], gauss_min_bins=False) is None


# bad numbers / choices stop the stage before any fit
def test_settings_are_validated_before_any_fit():
    for bad in ({"spectrum_bins": 0}, {"min_bin_count": -1}, {"window_nsigma": float("nan")},
                {"fit_ranges": [5, 10]}, {"fit_ranges": [[10, 5]]}, {"fit_ranges": []}, {"fit_bins": [0]},
                {"fit_bins": [True]}, {"linearity_fits": [0, 1]}, {"linearity_fits": {"kaon": None}},
                {"linearity_fits": {"electron": [1, 0]}}, {"linearity_fits": "electron"},
                {"resolution_fit_range": [0.3]}, {"resolution_fit_range": [0, float("inf")]},
                {"normal_error": "yes"}, {"error_prescription": "poisson"}, {"resolution_p0": [1.0, 2.0]},
                {"resolution_maxfev": 0}, {"hist_upper_percentile": 120}):
        exc = _refused(lambda: er.check_settings({**er.DEFAULTS, **bad}), (SettingsError,), list(bad)[0])
        assert exc
    good = er.check_settings({**er.DEFAULTS, "fit_ranges": ((5, 10),), "fit_bins": (30.0,),
                              "linearity_fits": {"Electron": (0, 1), "pion": None}, "resolution_fit_range": (0.3, 4)})
    assert good["fit_ranges"] == [[5.0, 10.0]] and good["fit_bins"] == [30] and isinstance(good["fit_bins"][0], int)
    assert good["linearity_fits"] == {"electron": [0.0, 1.0], "pion": None}
    assert good["resolution_fit_range"] == [0.3, 4.0]
    assert er.check_settings({**er.DEFAULTS, "resolution_maxfev": None})["resolution_maxfev"] is None
    _refused(lambda: er.settings_from_args(None, _flags("-spectrum-bins", "0")), (SettingsError,))


# unknown keys of the yml section stop the run; -fit / -bin keys are command line only
def test_yml_section_refuses_unknown_keys():
    for typo in ({"normalerror": True}, {"linearity_fit": [0, 1]}, {"normal_error": True, "resfit": [0, 1]}):
        bad = next(k for k in typo if k not in er.DEFAULTS)
        for call in (lambda: er.settings(_YmlCtx(typo)), lambda: er.settings_from_args(_YmlCtx(typo), _flags())):
            exc = _refused(call, (SettingsError,), bad)
            assert "unknown" in exc
    s = er.settings_from_args(_YmlCtx({"normal_error": True, "linearity_fits": {"electron": None},
                                       "resolution_fit_range": [0.3, 1]}), _flags())
    assert s["normal_error"] is True and s["linearity_fits"] == {"electron": None}
    assert s["resolution_fit_range"] == [0.3, 1.0] and s["fit_ranges"] is None
    assert er.yml_section(None) == {} and er.yml_section(_YmlCtx(None)) == {}
    for key, value in (("fit_ranges", [[15, 32], [50, 120]]), ("fit_bins", [30, 40])):
        exc = _refused(lambda: er.settings(_YmlCtx({key: value})), (SettingsError,), key)
        assert "command-line only" in exc and "MIP" in exc and "a.u." in exc and "unknown" not in exc


# ----------------------------------------------------------------------------------------------
# -fit[a,b] spectrum fits (fitting.py range helpers)
# ----------------------------------------------------------------------------------------------
# default page edges are the two-pass binning
def test_default_spectrum_edges_are_the_two_pass_binning():
    rng = np.random.default_rng(20)
    for values in (_electron_like(rng), _electron_like(rng, n=3000, mu=110, sigma=12), rng.normal(1.0, 0.3, 500)):
        assert np.array_equal(er.default_spectrum_edges(values), er.two_pass_gauss_fit(values)["edges"])
        options = {"bins": 60, "upper_percentile": 99.0, "upper_factor": 1.2, "upper_min": 5.0}
        assert np.array_equal(er.default_spectrum_edges(values, **options),
                              er.two_pass_gauss_fit(values, **options)["edges"])
    empty = er.default_spectrum_edges([np.nan])
    assert empty[0] == 0.0 and empty[-1] == 3.0 and empty.size == 91


# single and double Gaussian range fits recover mu and sigma
def test_range_spectrum_fit_single_and_double_recover_mu_and_sigma():
    rng = np.random.default_rng(21)
    values = rng.normal(20.0, 3.0, 60000)
    edges = er.default_spectrum_edges(values)
    one = er.range_spectrum_fit(values, [(14, 26)], default_edges=edges)
    assert one["ok"] and one["kind"] == "range_gauss" and one["ndf"] == one["n_fit_bins"] - 3
    assert abs(one["mu"] - 20.0) < 0.1 and abs(one["sigma"] - 3.0) < 0.1 and 0 < one["mu_error"] < 0.05
    assert math.isnan(one["mu_low"]) and math.isnan(one["mu_high"])
    seg = one["segments"][0]
    inside = fitting.in_range(one["page_centers"], (14, 26))
    assert np.array_equal(seg["centers"], one["page_centers"][inside])
    assert one["fit_bins"] is None and one["fit_ranges_text"] == "[14, 26]"
    b24 = er.range_spectrum_fit(values, [(14, 26)], bins=[24], default_edges=edges)
    assert np.allclose(b24["segments"][0]["edges"], np.linspace(14, 26, 25)) and b24["fit_bins"] == [24]
    assert b24["bin_rule"] == "24 equal bins per range" and abs(b24["mu"] - 20.0) < 0.1
    pion = np.concatenate([rng.normal(25.0, 4.0, 30000), rng.normal(80.0, 10.0, 20000)])
    two = er.range_spectrum_fit(pion, [(15, 35), (60, 100)], bins=[30, 40])
    assert two["ok"] and two["kind"] == "range_double_gauss" and two["fit_ranges_text"] == "[15, 35], [60, 100]"
    assert abs(two["mu_low"] - 25.0) < 0.2 and abs(two["mu_high"] - 80.0) < 0.5 and math.isnan(two["mu"])
    rs = {**er.DEFAULTS, "fit_ranges": [[15, 35], [60, 100]], "fit_bins": [30, 40]}
    via = er.range_spectrum_from_settings(pion, rs, er.default_spectrum_edges(pion))
    assert via["mu_low"] == two["mu_low"] and via["mu_high"] == two["mu_high"]
    for call in (lambda: er.range_spectrum_fit(values, [(26, 14)]), lambda: er.range_spectrum_fit(values, []),
                 lambda: er.range_spectrum_fit(values, [(14, 26), (30, 40)], bins=[30]),
                 lambda: er.range_spectrum_from_settings(values, er.DEFAULTS, edges)):
        _refused(call)


# drawing of a range page: (figure, axis, legend, texts, curves)
def _range_page(values, fit):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    er.plot_spectrum_with_fit(ax, values, fit, "Energy [MIP]")
    handles, labels = ax.get_legend_handles_labels()
    texts = "\n".join([ax.get_title()] + [t.get_text() for t in ax.texts])
    curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
    return fig, ax, dict(zip(labels, handles)), texts, curves


# a failed range fit: NaN parameters, the page draws the points only
def test_failed_range_fit_shows_the_points_only():
    import matplotlib.pyplot as plt
    from unittest import mock

    values = np.random.default_rng(22).normal(20.0, 3.0, 20000)
    empty = er.range_spectrum_fit(values, [(200, 300)])
    with mock.patch.object(fitting, "curve_fit", side_effect=RuntimeError("no convergence")):
        broken = er.range_spectrum_fit(values, [(14, 26)])
    for fit in (empty, broken):
        assert not fit["ok"] and fit["status"].startswith("fit_failed: ") and fit["components"] == []
        for key in ("mu", "sigma", "mu_error", "sigma_over_mu", "chi2"):
            assert math.isnan(fit[key]), key
        fig, ax, legend, texts, curves = _range_page(values, fit)
        try:
            assert not curves and "fit failed:" in texts
            assert any("outside the fit range" in label for label in legend)
        finally:
            plt.close(fig)


# custom -bin points are scaled to the page bin width; a double Gaussian draws three curves
def test_range_page_scales_custom_bins_to_the_page_bin_width():
    import matplotlib.pyplot as plt

    values = np.random.default_rng(23).normal(20.0, 3.0, 60000)
    edges = er.default_spectrum_edges(values)
    w = edges[1] - edges[0]
    fit = er.range_spectrum_fit(values, [(14, 26)], bins=[48], default_edges=edges)
    fig, ax, legend, texts, curves = _range_page(values, fit)
    try:
        seg = fit["segments"][0]
        blue = next(h for label, h in legend.items() if "scaled to the page bin width" in label)
        assert np.allclose(blue.get_offsets()[:, 1], seg["counts"][seg["fitted"]] * w / 0.25, rtol=1e-12)
        assert ax.get_ylabel() == f"Events per {w:.3g} MIP" and ax.get_yscale() == "log" and len(curves) == 1
        peak = float(np.max(curves[0].get_ydata()))
        assert math.isclose(peak, 60000 * w / (3.0 * math.sqrt(2 * math.pi)), rel_tol=0.03)
    finally:
        plt.close(fig)
    pion = np.concatenate([np.random.default_rng(24).normal(25.0, 4.0, 30000),
                           np.random.default_rng(25).normal(80.0, 10.0, 20000)])
    double = er.range_spectrum_fit(pion, [(15, 35), (60, 100)], bins=[30, 40])
    fig, ax, legend, texts, curves = _range_page(pion, double)
    try:
        assert double["ok"] and len(curves) == 3 and "Double Gaussian fit" in texts
    finally:
        plt.close(fig)


# a mu / sigma / area on its bound is a failed fit; peaks inside the range are never flagged
def test_parameter_on_its_bound_is_a_failed_fit():
    for seed in range(4):
        values = np.random.default_rng(300 + seed).normal(20.0, 3.0, 20000)
        edges = er.default_spectrum_edges(values)
        for rng, edge in (((21, 30), 21), ((10, 19), 19)):
            fit = er.range_spectrum_fit(values, [rng], default_edges=edges)
            assert not fit["ok"] and fit["bound_parameters"] == [f"mu [{rng[0]}, {rng[1]}]"]
            assert fit["status"].startswith(f"fit_failed: mu of [{rng[0]}, {rng[1]}] at its bound {edge} ")
        for rng in ((14, 26), (17, 30), (12, 23)):
            fit = er.range_spectrum_fit(values, [rng], default_edges=edges)
            assert fit["ok"] and fit["bound_parameters"] == [], (seed, rng, fit["status"])
    problems, names = fitting._bound_problems([1e-9, 25.0, 4.0, 5000.0, 80.0, 10.0], [(15, 35), (60, 100)],
                                              [0, 15, 0.5, 0, 60, 0.5], [np.inf, 35, 20, np.inf, 100, 40], 30000.0)
    assert names == ["area [15, 35]"] and "area of [15, 35] at its bound 0" in problems[0]


# the bin-integral model has no Sheppard bias on coarse bins
def test_bin_integral_model_has_no_sheppard_bias():
    from scipy.stats import norm

    sigmas = [er.range_spectrum_fit(np.random.default_rng(400 + s).normal(20.0, 3.0, 20000), [(11, 29)],
                                    bins=[4])["sigma"] for s in range(8)]
    assert abs(np.mean(sigmas) - 3.0) < 0.05, np.mean(sigmas)
    lo, hi = np.array([-1.0, 1.0, 8.0, 30.0]), np.array([1.0, 2.0, 9.0, 31.0])
    got = fitting.gauss_bin_integral(lo, hi, 1000.0, 0.0, 1.0)
    assert np.allclose(got, 1000.0 * (norm.sf(lo) - norm.sf(hi)), rtol=1e-12, atol=0) and got[-1] > 0


# sigma/mu error with the mu-sigma covariance
def test_sigma_over_mu_error_uses_the_mu_sigma_covariance():
    assert math.isclose(er.sigma_over_mu_error(40.0, 0.1, 5.0, 0.1, 0.0), 0.125 * math.hypot(0.02, 0.0025))
    assert math.isnan(er.sigma_over_mu_error(0.0, 0.1, 5.0, 0.1, 0.0))
    edges = np.arange(0.0, 80.5, 0.5)
    fit = er.range_spectrum_fit(np.random.default_rng(500).normal(40.0, 5.0, 6000), [(28, 42)], default_edges=edges)
    c = fit["components"][0]
    assert fit["ok"] and fit["mu_sigma_cov"] == c["mu_sigma_cov"] == c["covariance"][1][2]
    assert math.isclose(fit["sigma_over_mu_error"], er.sigma_over_mu_error(
        fit["mu"], fit["mu_error"], fit["sigma"], fit["sigma_error"], fit["mu_sigma_cov"]))


# overlapping ranges are refused; touching ranges fit every entry once
def test_overlapping_ranges_refused_touching_ranges_fit_every_event_once():
    for text in ("[5,10],[5,10]", "[12,28],[22,45]"):
        _refused(lambda: fitting.parse_ranges(text), (argparse.ArgumentTypeError,), "overlap")
    assert fitting.parse_ranges("[5,10],[10,20]") == [(5.0, 10.0), (10.0, 20.0)]
    assert "overlap on [8, 10]" in fitting.range_options_error([(5, 10), (8, 20)], None)
    page = np.linspace(0.0, 8.0, 201)
    lower, upper = fitting.range_bin_edges([(0.5, 1.1), (1.1, 2.5)], None, page)
    assert lower[-1] == upper[0] and math.isclose(lower[-1], 1.08)
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(1.0, 0.15, 20000), rng.normal(1.8, 0.3, 10000), np.full(40, lower[-1])])
    fit = fitting.range_gauss_fit(values, [(0.5, 1.1), (1.1, 2.5)], (lower, upper), min_bin_count=1)
    entries = sum(float(seg["counts"].sum()) for seg in fit["segments"])
    assert entries == np.count_nonzero((values >= lower[0]) & (values <= upper[-1]))


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
# log count axis follows -min-bin-count and is never inverted
def test_spectrum_y_axis_follows_min_bin_count():
    import matplotlib.pyplot as plt

    counts = np.arange(1.0, 8.0)
    fit = {"kind": "two_pass", "ok": False, "status": "small synthetic sample", "centers": counts,
           "counts": counts, "visible_mask": np.ones(7, dtype=bool), "bins": 7, "min_bin_count": 1,
           "hist_lower": 0.0, "hist_upper": 8.0, "n_values": 28, "bin_width": 1.0, "window_nsigma": 2.0,
           "window_gaussian_coverage": 0.95}
    fig, ax = plt.subplots()
    try:
        er.plot_spectrum_with_fit(ax, [], fit, "Energy [MIP]", subtitle="probe")
        bottom, top = ax.get_ylim()
        assert top > bottom and 0 < bottom <= 0.8 and top >= counts.max()
    finally:
        plt.close(fig)
    assert er.log_count_bottom([10, 50]) == 8.0 and er.log_count_bottom([1, 50]) == 0.8
    assert er.log_count_bottom([]) == 8.0


# linearity page: particle styles, filled fitted points, hollow others, one subtitle line per fit
def test_linearity_page_per_particle():
    import matplotlib.pyplot as plt

    rs = er.settings()
    for scale, n_lines in (("adc_only", 0), ("PureCI", 1), ("s", 1)):
        lin = er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale=scale)
        fig, ax = plt.subplots()
        try:
            er.plot_linearity(ax, lin, "Energy [MIP]", title=f"linearity {scale}", counts=np.full(E_ALL.size, 1e3))
            labels = ax.get_legend_handles_labels()[1]
            assert sum("line (" in label for label in labels) == n_lines, (scale, labels)
            assert any(label.startswith("pion points") for label in labels)
            texts = " ".join(t.get_text() for t in ax.texts)
            if not n_lines:
                assert "no fit: points only" in texts, texts
            else:
                assert "electron (" in texts and "mu = (" in texts, texts
        finally:
            plt.close(fig)
    both = er.linearity(E_ALL, MU_ALL, ERR_ALL, PARTS, fits={"electron": None, "pion": None})
    fig, ax = plt.subplots()
    try:
        er.plot_linearity(ax, both, "Energy [MIP]")
        labels = ax.get_legend_handles_labels()[1]
        assert "electron points in the fit" in labels and "pion points in the fit" in labels
    finally:
        plt.close(fig)


# runs without a point are named on the linearity / resolution / e/h pages
def test_pages_name_the_missing_points():
    import matplotlib.pyplot as plt

    mu = 30.0 * E5 - 1.0
    mu[1] = mu[4] = np.nan
    lin = er.linearity(E5, mu, np.full(5, 0.3), fits={"electron": [0, 1]})
    fig, ax = plt.subplots()
    try:
        er.plot_linearity(ax, lin, "Energy [MIP]", no_point_reasons=["", "fit_failed: x", "", "", ""])
        texts = " ".join(" ".join(t.get_text() for t in ax.texts).split())
        assert "no point: 0.4 GeV (fit_failed: x)" in texts and f"no point: 4 GeV ({er.NO_POINT_REASON})" in texts
        assert ax.get_xlim()[1] >= 4.0 and "no point (see the note)" in ax.get_legend_handles_labels()[1]
    finally:
        plt.close(fig)
    ratio = np.sqrt(100 ** 2 / (E5 * 1000) ** 2 + 25 / (E5 * 1000) + 0.0036)
    ratio[0] = np.nan
    res = er.resolution_fit(E5, ratio, np.nan_to_num(ratio) * 0.01 + 1e-6)
    fig, ax = plt.subplots()
    try:
        er.plot_resolution(ax, res, "Energy [arbitrary units]")
        assert "no point: 300 MeV" in " ".join(" ".join(t.get_text() for t in ax.texts).split())
    finally:
        plt.close(fig)
    eh = er.e_over_h(E_ALL, MU_ALL, ERR_ALL, PARTS, electron_line=None)
    fig, ax = plt.subplots()
    try:
        er.plot_eh(ax, eh, "s")
        texts = " ".join(" ".join(t.get_text() for t in ax.texts).split())
        assert "no point: 3 GeV (no electron run at 3 GeV and no electron line fit)" in texts, texts
        assert "electron and pion run at this energy" in ax.get_legend_handles_labels()[1]
    finally:
        plt.close(fig)


# every figure kind renders into one pdf
def test_figures_all_kinds_and_failures():
    from matplotlib.backends.backend_pdf import PdfPages

    from ldmx_hcal.common import plotting

    rng = np.random.default_rng(10)
    values = _electron_like(rng, n=20000)
    fits = [er.two_pass_gauss_fit(values), er.two_pass_gauss_fit(values[:25]), er.two_pass_gauss_fit([]),
            er.peak_scan_fit(values), er.range_spectrum_fit(values, [(14, 26)]),
            er.range_spectrum_fit(values, [(2, 8), (14, 26)], bins=[12, 30]), er.range_spectrum_fit([], [(14, 26)])]
    rs = er.settings()
    lins = [er.linearity_from_settings(E_ALL, MU_ALL, ERR_ALL, PARTS, rs, scale=s) for s in ("adc_only", "PureCI", "s")]
    lins.append(er.linearity([4.0], [100.0], [1.0], fits={"electron": None}))
    ratio = np.r_[np.sqrt((0.2 / np.sqrt(EE)) ** 2 + 0.05 ** 2), 0.4 / np.sqrt(EP)]
    ress = [er.resolution_from_settings(E_ALL, ratio, 0.02 * ratio, rs, particles=PARTS),
            er.resolution_fit([1.0], [0.2], [0.01])]
    line = lins[2]["fits"]["electron"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "energy_response_smoke.pdf"
        with PdfPages(path) as pdf:
            fig, axes = plotting.new_figure(2, 4, figsize=(22, 9))
            for ax, fit in zip(axes.ravel(), fits):
                er.plot_spectrum_with_fit(ax, values, fit, plotting.MIP_UNITS, title="smoke")
                er.add_footer(ax, "Page n | smoke")
            plotting.save(fig, pdf)
            fig, axes = plotting.new_figure(2, 4, figsize=(22, 9))
            for ax, lin in zip(axes[0], lins):
                er.plot_linearity(ax, lin, plotting.MIP_UNITS)
            for ax, res in zip(axes[1][:3], ress):
                er.plot_resolution(ax, res, plotting.ARBITRARY_UNITS)
            er.plot_eh(axes[1][3], er.e_over_h(E_ALL, MU_ALL, ERR_ALL, PARTS, electron_line=line), "s")
            plotting.save(fig, pdf)
            er.summary_table_page(pdf, "summary", [er.scalar_fields(f) for f in fits[:2]],
                                  columns={"mu": "mu [MIP]", "sigma": "sigma", "status": "status"}, footer="smoke")
            er.summary_table_page(pdf, "empty", [])
        assert path.stat().st_size > 10_000


# scalar_fields keeps only the scalar entries
def test_scalar_fields():
    fit = er.two_pass_gauss_fit(np.random.default_rng(9).normal(20, 4, 20000))
    row = er.scalar_fields(fit)
    assert "edges" not in row and "counts" not in row and row["mu"] == fit["mu"]
    line = er.linearity_line(E5, 30 * E5, np.full(5, 0.1))
    assert "covariance" not in er.scalar_fields(line, "lin_") and "lin_slope" in er.scalar_fields(line, "lin_")


if __name__ == "__main__":
    failed = 0
    for name, func in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)):
        try:
            func()
            print(f"PASS  {name}")
        except unittest.SkipTest as exc:
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
