# calibration/

Variables of the output files in `calibration/` (normal run, the most complete version of each file). In this
README a variable is described once, at its first appearance; a later file lists only its new variables, or a name it
uses with another meaning. A variable described in [data/README.md](../data/README.md) or
[results/README.md](../results/README.md) is not repeated: the line "Described in ..." under a file names it.

A normal run writes the stage folders directly here (no `fullset/` folder); `test/` holds the tables of `-test` runs
(`test/<stage>/`). `reference/` holds a fixed input, the TOA table `toa_calibration_phase3.csv` (the per-bar TOA
constants by Joe Muse: bunch-crossing shift, end-1 offset) read by make-csv; it is committed to git, the other tables
are not. Each table has a `.provenance.json` next to it (unless `-no-json`) that says where it comes from and lists its
columns.

## 1. calibration/pedestal_mpv/calibration_parameters.csv (calib)

One row per channel (bar end). The full table, written when both csv units of calib are chosen (the default); it is
the table every later stage reads. One csv unit alone writes a part table with the same columns in the same order:
`pedestal_parameters.csv` (`-only pedestal_csv`: layer, strip, end and the pedestal columns pedestal ...
n_pedestal_events, pedestal_anchor, pedestal_anchor_method, pedestal_selection, n_selected_events) or
`mpv_parameters.csv` (`-only mpv_csv`: layer, strip, end and the MPV columns mpv ... mpv_status, mpv_method ...
mpv_ndf; mpv_pedestal and mpv_pedestal_sigma are the pedestal and sigma the MPV run computed and used).

| variable | meaning |
|---|---|
| end | end of the bar (0 or 1) |
| pedestal | most probable adc_sum without signal (mean of the no-signal events) |
| pedestal_per_time_sample | pedestal of one time sample |
| pedestal_sigma | spread (standard deviation) of the no-signal adc_sum |
| pedestal_mu | pedestal - pedestal_anchor |
| pedestal_range_with_5sigma | pedestal + 5 pedestal_sigma (hit threshold) |
| pedestal_range_with_20percent | 1.2 pedestal (default hit threshold) |
| n_pedestal_events | no-signal events used for the pedestal |
| mpv | most probable adc_sum - pedestal of a MIP (peak of the fitted curve) |
| mpv_error | statistical error of mpv |
| landau_width | Landau width of the MPV fit |
| gauss_sigma | Gaussian width of the MPV fit |
| mpv_fit_low, mpv_fit_high | edges of the MPV fit window |
| n_mpv_entries | entries above the MPV cut |
| mpv_chi2_ndf | fit quality (deviance / degrees of freedom) |
| mpv_status | OK, or why mpv is missing: LOW_STATS, NO_PEDESTAL, ROUGH_FIT_FAILED, FIT_FAILED or PEAK_OUTSIDE_WINDOW |
| pedestal_anchor | centre of the pedestal window |
| pedestal_anchor_method | how the centre was found (sample_mode or median_sum) |
| pedestal_selection | how the no-signal events were chosen (raw_samples or csv_tot_toa_zero) |
| n_selected_events | no-signal events before the window |
| mpv_method | MPV method (two_stage_langaus) |
| mpv_pedestal, mpv_pedestal_sigma | pedestal and pedestal_sigma used by the MPV fit |
| mpv_threshold | MPV cut: adc_sum - pedestal above this (5 pedestal_sigma) |
| mpv_rough_floor | start of the rough fit |
| mpv_rough_status | status of the rough fit: CONVERGED, NOT_CONVERGED, FAILED or NOT_RUN |
| mpv_rough_peak, mpv_rough_width | peak and width of the rough fit |
| mpv_fit_status | status of the final fit: CONVERGED, NOT_CONVERGED, FAILED or NOT_RUN |
| mpv_fit_message | fitter message (why a fit failed) |
| mpv_covariance_status | whether mpv_error could be computed: OK, NOT_POSITIVE_DEFINITE, SINGULAR, NOT_FINITE, NO_FREE_PARAMETERS or NOT_COMPUTED |
| mpv_active_bounds | fit parameters stuck at a limit |
| landau_mpv_param | Landau location parameter of the fit (not the peak) |
| langaus_area | fitted area |
| n_mpv_fit_entries | entries inside the fit window |
| mpv_deviance, mpv_ndf | fit deviance and degrees of freedom |

Described in [../data/README.md](../data/README.md): layer, strip

## 2. calibration/tot_adc_peak/tot_adc_peak_parameters.csv (adcpeak)

One row per channel (bar end), 384 rows. The line is fitted on the run `-fit-dataset` (default pion 3) below the ADC
rail, with adc_max bins covering the whole `fit_adc_window` [949, 1021]; it turns the TOT of a saturated end into its
ADC peak, (TOT - offset) / slope.

| variable | meaning |
|---|---|
| tot_per_adc_slope | slope of the line TOT = slope x adc_max + offset [TOT counts per ADC count] (0 = not fittable) |
| tot_offset | offset of that line [TOT counts] (0 = not fittable) |
| status | fit status: ok, no_data, no_hits_in_window, no_filled_bins or too_few_bins_after_outlier_cut |
| n_hits_window | hits inside the adc_max and TOT windows of the fit |
| n_hits_fit | hits in the bins used by the fit |
| n_bins_filled | non-empty adc_max bins |
| n_bins_removed | bins removed by the 3-sigma cut on the bin mean TOT |
| n_bins_fit | bins used by the fit |
| fit_dataset | run key of the fit run (e.g. pion_3GeV) |

Described in [../data/README.md](../data/README.md): layer, strip

## 3. calibration/charge_injection/ci_channel_parameters.csv (ci)

One row per channel (bar end) of the channel map, 384 rows: the charge-injection map (provided by Erik Wallin) and the
TOT scale s (s parameter, Lei). Written with the ci unit `parameters`; a run with other training samples or an
`-output-tag` adds its variant tag to the file name (e.g. `ci_channel_parameters_s_muon.csv`).

| variable | meaning |
|---|---|
| dpm, link, chan | electronics address of the channel: DPM, link, channel |
| ci_status | charge-injection result: CI_OK, RAW_MISSING_PROTECTED_CALIBRATION, RAW_NO_INJECTION_EVENTS or RAW_CI_FAILED |
| ci_reason | why the charge-injection calibration failed (empty for CI_OK) |
| ci_k_adc_per_fc | low-cap injection gain k of adc_sum8 = k q + P [ADC/fC] |
| ci_P_injection | low-cap injection intercept P (8-sample pedestal seen by the injection) [ADC] |
| ci_q_threshold_fc | TOT turn-on charge: smallest injected charge with TOT > 0 [fC] |
| ci_pole_asym_slope_counts_per_fc | slope m of the pole fit TOT = m q + c0 - A / (q - B) [TOT counts/fC] |
| ci_pole_asym_intercept_counts | intercept c0 of the pole fit [TOT counts] |
| ci_pole_strength | pole strength A of the pole fit |
| ci_pole_position_fc | pole position B of the pole fit [fC] |
| n_ci_measured | injection points in the TOT region (charge >= turn-on) |
| n_ci_reliable | reliable points used for the q(TOT) map (0 < TOT < 4094, no pole outlier) |
| n_ci_interp_nodes | number of nodes of the q(TOT) map |
| n_ci_pava_merged | points merged into a neighbour to make the map monotone (PAVA, equal levels) |
| n_ci_pole_audit_outliers | pole-fit outliers (\|TOT - model\| >= 500 counts) |
| n_ci_rail_excluded | points with a railed TOT (>= 4094) left out of the map |
| mono_tot_min, mono_tot_max | TOT of the first / last node of the q(TOT) map [TOT counts] |
| mono_q_min_fc, mono_q_max_fc | charge of the first / last node of the q(TOT) map [fC] |
| mono_slope_low_fc_per_tot | slope of the linear extrapolation of q(TOT) below the first node [fC per TOT count] |
| mono_slope_high_fc_per_tot | slope of the linear extrapolation of q(TOT) above the last node [fC per TOT count] |
| ci_tot_threshold_fc | TOT threshold of the channel [fC]: a TOT end needs q(TOT) above it (TOT-threshold calibration provided by Erik Wallin) |
| ci_thr_asym_slope_counts_per_fc | TOT-threshold calibration: asymptote slope of its pole fit [TOT counts/fC] |
| ci_thr_asym_intercept_counts | TOT-threshold calibration: asymptote intercept of its pole fit [TOT counts] |
| ci_thr_pole_strength | TOT-threshold calibration: pole strength A |
| ci_thr_pole_position_fc | TOT-threshold calibration: pole position B [fC] |
| ci_thr_adc_gain_per_fc | TOT-threshold calibration: ADC gain [ADC/fC] |
| ci_thr_adc_intercept_adc | TOT-threshold calibration: ADC intercept [ADC] |
| ci_thr_status | status of the TOT-threshold calibration: OK, NO_INJECTION_EVENTS or the error of its fit |
| n_scan_events | injections of the channel in the scans |
| n_low_groups_kept | low-cap DAC groups kept by the repeatability filter |
| n_high_groups_kept | high-cap DAC groups kept by the repeatability filter |
| n_groups_dropped_repeats | DAC groups dropped for fewer than 3 injections |
| n_groups_dropped_consistency | DAC groups dropped for a too large ADC / TOT range |
| training_particle | training sample of the s fit: pion, muon, electron or mix (with several samples: the sample chosen as s) |
| shrinkage | shrinkage lambda used in w = n / (n + lambda) |
| s_direct | median of ratio = (adc_sum - pedestal) / (k q(TOT)) over the channel's exact-overlap hits |
| s_exact_n | number n of exact-overlap hits of the channel (adc_max < 1020) |
| s_censored_n | hits with adc_max >= 1020 (counted, not used) |
| shrinkage_weight | w = n / (n + lambda) |
| s_link_prior | prior of s: geometric median of s_direct over the channel's (DPM, LINK), else the global prior |
| s_prior_level | level of the prior: link or global |
| s_final | TOT scale s of the channel = exp(w ln s_direct + (1 - w) ln prior), used by the s scale (s parameter, Lei) |
| s_source | where s comes from: DIRECT_SHRUNK_TO_LINK, BORROWED_LINK_OR_GLOBAL, RAW_NO_MODEL or RAW_NO_CI |
| global_prior | geometric median of s_direct over all channels |
| s_\<sample\> | with several training samples (`-scale-particle pion muon`): s_final of that sample (s_pion, s_muon, s_electron, s_mix) |
| s_\<sample\>_source | s_source of that sample; s is the sample with the fewest BORROWED_LINK_OR_GLOBAL channels |

Described in [../data/README.md](../data/README.md): layer, strip

## 4. calibration/charge_injection/ci_tot_charge_nodes.csv (ci)

One row per node of the monotone q(TOT) map of a channel.

| variable | meaning |
|---|---|
| node_index | index of the node in the channel's map (0, 1, ...) |
| tot_counts | TOT of the node [TOT counts] |
| q_fc | injected charge at that TOT [fC] |

Described in [../data/README.md](../data/README.md): layer, strip
