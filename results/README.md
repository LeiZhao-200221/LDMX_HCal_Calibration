# results/

Variables of the output files in `results/` (normal run, the most complete version of each file). In this README a
variable is described once, at its first appearance; a later file lists only its new variables, or a name it uses
with another meaning. A variable described in [data/README.md](../data/README.md) or
[calibration/README.md](../calibration/README.md) is not repeated: the line "Described in ..." under a file names it.
`_end0` / `_end1`: the two ends of a bar.

- `results/fullset/<stage>/`: the final results (figures and result tables) of normal runs, saved only with
  `-output`; without it a normal run draws them into the scratch folder `cache/<stage>/unsaved/` and saves nothing
  here.
- `results/test/`: `-test` runs: `test.csv`, `test1.csv`, ..., `test.pdf`, ... without `-output` (the next test run
  overwrites them), `results/test/<stage>/` with `-output`.

## Files

`<run>` = the run folder (e.g. `Muon_4GeV`), `<csv stem>` = the name of one beam csv of the run, `<v>` = the variant
tag of adcpeak / ci (empty for a default run).

| folder | files | output unit (`-only`) |
|---|---|---|
| `make_csv_qa/` | `<run>/<csv stem>_qa.pdf`: distributions of the beam csv columns | make-csv `qa` |
| `pedestal_mpv/` | `pedestal_fits.pdf`, `mpv_fits.pdf`: the pedestal and MPV fit of every channel | calib `pedestal_plot`, `mpv_plot` |
| `clean_dataset/` | `<run>/<csv stem>_clean_cutflow.csv` (section 1), `_clean_layers.csv` (section 2), `_clean_diagnostics.pdf` (hit and energy distributions, E_MIP fits), `_clean_results.json` (record of the files written) | clean `cutflow`, `layers`, `diagnostics` |
| `tot_adc_peak/` | `tot_adc_peak_fits.pdf` (line fit of every channel end), `energy_response<v>.csv` (section 3), `energy_spectra<v>.pdf` (Arbitrary Energy spectrum and fit of every run), `linearity_resolution<v>.pdf` | adcpeak `line_fits`, `energy_response`, `energy_spectra`, `linearity_resolution` |
| `charge_injection/` | `energy_linearity<v>.pdf`, `energy_resolution<v>.pdf`, `e_over_h<v>.pdf` (one page per scale + a table); on request `energy_response<v>.csv` (section 4) and `quality_threshold_scan<v>.csv` (section 5) | ci `linearity`, `resolution`, `eh`; `response`, `quality_scan` |
| `multiscale_plots/` | `<run key>_<unit>_<scale>[_<tag>].pdf` (units dist_event, dist_bar, dist_channel, event_display, bar_maps, layer), `<run key>_hitmap.pdf` | plots, one unit each |

## 1. results/fullset/clean_dataset/\<run\>/\<csv stem\>_clean_cutflow.csv (clean)

One row per selection step (input, the bar rule, then for the muon standard the two event cuts).

| variable | meaning |
|---|---|
| step | name of the step: input, bars_both_ends_hit or bars_either_end_hit, at_most_one_bar_per_layer, layer1_bar_in_energy_window |
| rows | bars (csv rows) left after the step |
| events | events left after the step |
| definition | what the step keeps, in words |

## 2. results/fullset/clean_dataset/\<run\>/\<csv stem\>_clean_layers.csv (clean)

One row per layer (1-19).

| variable | meaning |
|---|---|
| n_clean_bars | clean bars in the layer |
| n_events | clean events with a clean bar in the layer |
| n_hit_end0, n_hit_end1 | clean bars of the layer whose end is a hit (adc_sum >= threshold) |
| energy_mev_end0_mean, energy_mev_end1_mean | mean energy of the hit ends, (adc_sum - pedestal) / mpv x 4.83143 MeV [MeV] |
| energy_mev_bar_mean | mean (E0 + E1) / 2 of the clean bars with both ends hit [MeV] |

Described in [../data/README.md](../data/README.md): layer

## 3. results/fullset/tot_adc_peak/energy_response.csv (adcpeak)

One spectrum row per run (its Arbitrary Energy spectrum, all csv files of the run), then one linearity row per
particle fitted with `-linfit` (none by default) and one resolution row per model (column record). The values are in
Arbitrary Energy [a.u.]: the sum of the bar ADC peaks above the pedestal of an event, no energy scale.

| variable | meaning |
|---|---|
| record | row type: spectrum, linearity or resolution |
| tot_mode | TOT mode: linear, none or mean_fallback |
| units | unit of the values (arbitrary units) |
| dataset | run key (e.g. electron_0.3GeV) |
| particle | beam particle of the run; linearity row: the fitted particle; resolution row: the fit particle (electron) |
| energy_gev | beam energy [GeV] |
| spectrum_fit | spectrum fit method: two_pass, peak_scan or range (-fit) |
| n_events | selected events in the spectrum (another meaning than in section 2) |
| mu | mean of the Gaussian fitted to the peak [a.u.] |
| mu_error | fit error of mu [a.u.] |
| sigma | width of the Gaussian [a.u.] |
| sigma_error | fit error of sigma [a.u.] |
| sigma_over_mu | sigma / mu (resolution point of the run) |
| fit_low, fit_high | edges of the spectrum fit window [a.u.] |
| n_fit_bins | bins used by the spectrum fit |
| chi2 | chi2 of the fit of the row |
| ndf | degrees of freedom of the fit of the row |
| status | fit status of the row: ok, or why it failed / a note (another meaning than in [calibration/README.md](../calibration/README.md), section 2) |
| fit_ranges | -fit range(s) of the spectrum fit (with -fit) |
| bin_rule | bins of the -fit fit: default page bins or N equal bins per range |
| sigma_over_mu_error | error of sigma / mu with the mu-sigma covariance (single -fit) |
| mu_low, mu_low_error, sigma_low, sigma_low_error | double -fit: mean and width of the component in the first range, with their errors [a.u.] |
| area_low | double -fit: events of the component in the first range |
| mu_high, mu_high_error, sigma_high, sigma_high_error | double -fit: mean and width of the component in the second range, with their errors [a.u.] |
| area_high | double -fit: events of the component in the second range |
| point_mu_error | error of the linearity point, sigma / sqrt(N) [a.u.] |
| point_ratio_error | error of the resolution point sigma / mu |
| point_error_n | N used in the point errors (the selected events) |
| in_resolution_fit | the point was used in the resolution fit (electron points only) |
| resolution_predicted | fitted sigma / mu at this energy |
| in_linearity_fit | the point was used in the line of its particle |
| linearity_predicted | prediction of mu by the line of the run's particle at this energy [a.u.] |
| linearity_residual_fraction | (mu - prediction) / prediction |
| fit_range_gev | -linfit range of the particle (linearity row) or -resfit range (resolution row), '[a, b]' or 'all points' |
| ok | the linearity / resolution fit succeeded |
| weighted | the linearity line was a weighted fit (1 / mu_error^2) |
| slope | slope of mu = slope x E + intercept [a.u. per GeV] |
| slope_error | error of the slope |
| intercept | intercept of the line [a.u.] |
| intercept_error | error of the intercept |
| chi2_over_ndf | chi2 / ndf of the linearity or resolution fit |
| n_fitted | points used in the linearity / resolution fit |
| fitted_energies | beam energies of the fitted points [GeV] |
| model | resolution model: three_term, sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2) (E in MeV) |
| a | resolution noise term a [MeV] |
| a_error | error of a |
| b | resolution stochastic term b [sqrt(MeV)] |
| b_error | error of b |
| c | resolution constant term c |
| c_error | error of c |
| error_prescription | how the point errors were computed: sigma_over_sqrt_n, sigma_fit_error or gaussian |
| error_scale | -normalerror: sqrt(chi2/ndf) when chi2/ndf > 1, else 1 |
| slope_error_scaled, intercept_error_scaled | -normalerror: slope_error, intercept_error x error_scale |
| a_error_scaled, b_error_scaled, c_error_scaled | -normalerror: a_error, b_error, c_error x error_scale |

## 4. results/fullset/charge_injection/energy_response.csv (ci, -only response)

One spectrum row per run and scale, then per scale the linearity rows (one per fitted particle, or one points_only
row), one resolution row and, on the s scales, one e_over_h row per pion run (column row_type). The scales: adc_only,
pureci (PureCI, charge injection with s = 1), s (charge injection x the TOT scale s; s parameter, Lei) and, with
several training samples, s_\<sample\>.

| variable | meaning |
|---|---|
| row_type | spectrum, linearity, resolution or e_over_h |
| beam_energy_gev | beam energy of the run [GeV] |
| path | energy scale: adc_only, pureci, s or s_\<sample\> |
| selection_path | scale whose bad-bar flags selected the events |
| quality_threshold | cut on bad_strip_fraction (empty without the quality filter) |
| n_csv_files | csv files of the run summed in its spectrum |
| n_events_in_file | events in the csv files of the run |
| n_events_reconstructed | events with at least one reconstructed bar |
| n_events_selected | events passing the quality cut |
| retention_fraction | n_events_selected / n_events_reconstructed |
| n_bars_selected_events | reconstructed bars of the selected events |
| fit_kind | fit model: two_pass, muon_core_gauss, pion_core_gauss, range_gauss, range_double_gauss; linearity: linear_weighted, linear_unweighted or points_only; resolution: the model |
| fit_status | ok, or why the fit failed |
| fit_ok | the fit converged |
| fit_bins | the -bin value(s) of the row (empty without -bin) |
| mu_mip | fitted peak position mu [MIP] |
| mu_error_mip | fit error of mu [MIP] |
| sigma_mip | fitted Gaussian width sigma [MIP] |
| sigma_error_mip | fit error of sigma [MIP] |
| fit_low_mip, fit_high_mip | edges of the fit window [MIP] |
| bin_width_mip | bin width of the fitted histogram [MIP] |
| hist_upper_mip | upper edge of the two-pass histogram [MIP] |
| pass1_mu_mip | mu of the first pass of the two-pass fit [MIP] |
| pass1_sigma_mip | sigma of the first pass [MIP] |
| pass1_chi2 | chi2 of the first pass |
| pass1_ndf | degrees of freedom of the first pass |
| core_bin_count | bins in the second-pass core window |
| mu_low_mip, mu_low_error_mip, sigma_low_mip, sigma_low_error_mip | double -fit: mu and sigma of the component of the lower range, with their errors [MIP] |
| mu_high_mip, mu_high_error_mip, sigma_high_mip, sigma_high_error_mip | double -fit: mu and sigma of the component of the higher range, with their errors [MIP] |
| area_low_events, area_low_error_events | double -fit: area of the lower component and its error [events] |
| area_high_events, area_high_error_events | double -fit: area of the higher component and its error [events] |
| peak_scan_mu_mip | mu of the automatic peak-scan fit [MIP] |
| peak_scan_sigma_mip | sigma of the peak-scan fit [MIP] |
| peak_scan_sigma_over_mu | sigma / mu of the peak-scan fit |
| peak_scan_n_peaks | peaks found by the peak scan |
| peak_scan_status | status of the peak-scan fit |
| mu_stat_error_mip | statistical error of the linearity point mu [MIP] |
| ratio_error | error of the resolution point sigma / mu |
| linearity_fitted | the point was used in the line of its particle |
| linearity_predicted_mip | prediction of mu by the line of the run's particle at its beam energy [MIP] |
| linearity_fit_range | the -linfit range of the particle '[a, b]', or 'all points' [GeV] |
| slope_mip_per_gev | slope of the linearity line [MIP/GeV] |
| slope_error_mip_per_gev | error of the slope [MIP/GeV] |
| intercept_mip | intercept of the linearity line [MIP] |
| intercept_error_mip | error of the intercept [MIP] |
| max_abs_rel_residual | largest \|relative residual\| of the fitted points of the line |
| linearity_source | spectrum fits that gave the points: two_pass, peak_scan or range_fit (-fit) |
| slope_error_scaled_mip_per_gev | -normalerror: slope error x error_scale [MIP/GeV] |
| intercept_error_scaled_mip | -normalerror: intercept error x error_scale [MIP] |
| resolution_fit_range | the -resfit range '[a, b]' (empty without one) [GeV] |
| resolution_a_mev | noise term a of the sigma/mu model [MeV] |
| resolution_a_error_mev | error of a [MeV] |
| resolution_b_sqrt_mev | stochastic term b of the sigma/mu model [sqrt(MeV)] |
| resolution_b_error_sqrt_mev | error of b [sqrt(MeV)] |
| resolution_c | constant term c of the sigma/mu model |
| resolution_c_error | error of c |
| resolution_a_error_scaled_mev, resolution_b_error_scaled_sqrt_mev, resolution_c_error_scaled | -normalerror: the errors of a, b, c x error_scale |
| e_over_h | e/h = mu_electron / mu_hadron at the pion energy (s scale) |
| e_over_h_error | error of e/h |
| mu_electron_mip, mu_electron_error_mip | mu of the electrons at that energy and its error [MIP] |
| mu_hadron_mip, mu_hadron_error_mip | mu of the pion run and its error [MIP] |
| electron_source | where mu_electron comes from: electron run (filled point) or electron line fit (hollow point) |
| eh_filled | the point is filled (an electron run at that energy) |
| eh_note | why a pion run has no e/h point (empty when it has one) |

Described above: dataset, particle, sigma_over_mu, chi2, ndf, fit_ranges, linearity_residual_fraction, chi2_over_ndf,
fitted_energies, error_prescription, error_scale (section 3)

## 5. results/fullset/charge_injection/quality_threshold_scan.csv (ci, -only quality_scan)

One row per threshold of the scan grid and electron run: what the bad-strip cut would keep at each threshold (a check;
the cut itself does not change).

| variable | meaning |
|---|---|
| threshold | bad_strip_fraction threshold tried |
| n_events_before | reconstructed events before the cut |
| n_events_after | events kept by this threshold |
| sigma_over_mu | peak-scan sigma / mu of the kept events (another meaning than in section 3: from the peak scan) |
| n_detected_peaks | peaks found in the kept spectrum |
| detected_peak_positions_mip | positions of those peaks, separated by ';' [MIP] |
| peak_shift_vs_unfiltered_fraction | (mu - unfiltered mu) / unfiltered mu |
| resolution_change_vs_unfiltered_fraction | (sigma/mu - unfiltered sigma/mu) / unfiltered sigma/mu |
| mean_bad_strip_fraction_all | mean bad_strip_fraction of all reconstructed events |
| shoulder_count | kept events in the low-side shoulder [70, 100) MIP |
| core_count | kept events in the peak core [105, 125) MIP |
| shoulder_retention_fraction | shoulder_count / unfiltered shoulder count |
| core_retention_fraction | core_count / unfiltered core count |
| configured_threshold | the quality_threshold the run actually uses |

Described above: dataset (section 3), beam_energy_gev, retention_fraction, peak_scan_mu_mip, peak_scan_sigma_mip
(section 4)
