# Code

The configuration files are described in [configure/README.md](../configure/README.md), the run logs in
[logs/README.md](../logs/README.md), and the variables of the output files in
[data/README.md](../data/README.md), [calibration/README.md](../calibration/README.md) and
[results/README.md](../results/README.md). Installation, a new machine, the tests and the container: the last
sections of this file.

## Running

Every command runs from the `data_analysis` folder:

```
$env:LDMX_PYTHON = "..\venv312\Scripts\python.exe"
.\runner.ps1 <stage> [options]
```

Stage order: make-csv -> calib -> clean -> adcpeak / ci -> plots.

**Runs are folders.** A run of `configure/datasets.yml` is the folder of its csv files,
`data/analysis_files/<run>/`, named from its place in the yml: `muon: 4GeV:` -> `Muon_4GeV/`, several runs at one
energy -> `Muon_4GeV_run<N>/`. make-csv writes one csv per ROOT file of the run into it. Every later stage reads every
csv of the folder and writes one product per csv, in a sub-folder per run (`data/clean_dataset/Muon_4GeV/<csv
stem>_clean.csv`). Nothing is merged. An event is identified by its csv file and `pf_event` (`event_source`,
`pf_event`), because two csv files may both start at event 1. A fit that needs the statistics of a whole run (calib,
the spectrum of a run) sums the histograms of its csv files.

**Output units.** Every stage makes its outputs as units, one table or one pdf each, chosen with
`-only unit,unit` (single units or group names, comma or space separated). Without `-only` a stage makes its default
units. The units of each stage are listed in its section.

**Note on the commands.** `[ ]` = optional: change the value or delete the whole bracket, and the command still runs
(the values shown are defaults or examples); a bracket inside another one (`-bin` inside `-fit`) needs the outer one.
Every other command of a stage is its command with brackets deleted or changed. The common options (`-test`,
`-only`, `-output`, `-no-json`, `-from-saved`, `-clean-input`, ...) are described once, in Runner, Common options.

## 1. make-csv: ROOT ntuple -> beam csv

Code: `src/ldmx_hcal/conversion/make_analysis_files.py` (based on Joe Muse, revised by Lei)

```
.\runner.ps1 make-csv [--muon 4] [-test] [-max-events 5000] [-overwrite] [-waveform] [-sample-window all] [-tot-mode tot_branch] [-duplicate-digis error] [-toa-calibration calibration/reference/toa_calibration_phase3.csv] [-only csv,qa] [-output fixed_Muon_4GeV_qa.pdf] [-chunk-entries 3072000] [-workers 4] [-no-json] [-dry-run]
```

Every ROOT file of a run becomes one csv in the run folder, named after the ROOT file
(`fixed_Muon_4GeV.root` -> `data/analysis_files/Muon_4GeV/fixed_Muon_4GeV.csv`; `-test`:
`data/test/analysis_files/Muon_4GeV/`). `-input FILES` converts single ROOT files without `datasets.yml`: one csv per
file, directly in `data/analysis_files/` (`-test`: `data/test/analysis_files/`), named after the file
(`muon_12321.root` -> `muon_12321.csv`) or `-output-csv`. **The later stages read only the csv files of runs registered
in `configure/datasets.yml`**: to analyse an `-input` file, add its run to `datasets.yml` and convert it as a run. An
existing csv is skipped unless `-overwrite` is given. `-max-events N` without `-test` makes the run's csv from the first
N events of every ROOT file.

Units (`-only`): `csv` (the beam csv + its provenance json; default), `qa` (the QA pdf of each csv). `-only qa` alone
draws the QA pdf from the existing csv.

| option | what it does | default |
|---|---|---|
| `--electron E` / `--muon E` / `--pion E` | runs to convert (`configure/datasets.yml`) | every `kind: data` run whose ROOT file is on this machine |
| `-input ROOT ...` | convert these ROOT files instead of runs, one csv each | - |
| `-output-csv NAME.csv` | file name of the csv of ONE `-input` file, no folder (the csv of a run is always named after its ROOT file) | `<input file name>.csv` |
| `-max-events N` | first N events of every ROOT file (without `-test`: the run's csv) | all |
| `-overwrite` | replace an existing csv | off (skip it) |
| `-waveform` | add `adc_0..adc_7` per end | off |
| `-sample-window all\|A:B` | use only samples A..B-1 of each digi | all |
| `-tot-mode tot_branch\|tot_comp\|tot_ne_adc` | TOT sample source: the TOT branch, TOT-complete samples only, or (flat MC ntuples only) TOT != ADC | tot_branch |
| `-duplicate-digis error\|drop` | repeated (event, layer, strip, end) digis: stop, or leave them out (the real data have none) | error |
| `-toa-calibration CSV\|none` | TOA table; none = no bunch-crossing wrap, no end-1 offset | `calibration/reference/toa_calibration_phase3.csv` |
| `-mc` | MC mode (10-sample digis); MC files are converted with `-mc -input FILE` | off |
| `-mc-samples N\|auto` | time samples per MC digi, checked before the conversion | auto |
| `-chunk-entries N` | ROOT entries per streamed chunk | 3072000 |
| `-workers N` | processes formatting the text of large csv files | 4 |
| `-only csv,qa` | output units | csv |
| `-output [NAME.pdf ...]` | save the QA pdfs (name = default file name `<csv stem>_qa.pdf`, or `KEY=FILE`) | full set: not saved; `-test`: `results/test/test.pdf` |

Refused: `-from-saved`, `-clean-input`.

**Inputs**
- ROOT files: `data/root/` (`paths.root_dir`); the files of each run are listed in `configure/datasets.yml` (`root:`,
  a path or a list, e.g. `muon: 4GeV: root: ${root_dir}/fixed_Muon_4GeV.root`). `-input ROOT ...` reads the given files.
- TOA table: `calibration/reference/toa_calibration_phase3.csv` (`paths.toa_calibration`)
- `-only qa` without `csv`: the existing csv of each run
- settings: `configure/config.yml`, section `make_csv`

**Outputs**
- csv (unit `csv`): `data/analysis_files/<run>/<ROOT name>.csv` with `<csv stem>.provenance.json` next to it
  (`-test`: `data/test/analysis_files/<run>/`); `-input`: `data/analysis_files/<file name>.csv`
- QA pdf (unit `qa`): `results/fullset/make_csv_qa/<run>/<csv stem>_qa.pdf` (with `-output`); `-input`:
  `results/fullset/make_csv_qa/<csv stem>_qa.pdf`; `-test`: `results/test/test.pdf`, with `-output`
  `results/test/make_csv_qa/`
- log: `logs/make_analysis_files/`

## 2. calib: pedestal + MIP MPV per channel

Code: `src/ldmx_hcal/calibration/calibration_parameters.py` (based on Joe Muse, revised by Lei)

```
.\runner.ps1 calib [--muon 4] [-test] [-only pedestal_csv,mpv_csv,pedestal_plot,mpv_plot] [-pedestal-selection raw_samples] [-pedestal-window 200] [-mpv-nsigma 5] [-mpv-window-widths 2] [-mpv-bin-width 20] [-fit[150,700] [-bin 22]] [-workers 4] [-output pedestal_fits.pdf mpv_fits.pdf] [-no-json] [-dry-run]
```

calib reads the csv files of ONE source run (default muon 4) and sums their per-channel histograms. Every run computes
what it writes from these csv files: the MPV needs the pedestal, so an MPV-only run computes the pedestal too. Nothing
is taken from an existing table, and nothing is merged.

Units (`-only`): `pedestal_csv`, `mpv_csv`, `pedestal_plot` (`pedestal_fits.pdf`), `mpv_plot` (`mpv_fits.pdf`);
groups `pedestal` (pedestal_csv, pedestal_plot), `mpv` (mpv_csv, mpv_plot), `csv` (both tables), `plots` (both pdfs),
`all`. Default: every unit. Both csv units give the full table `calibration_parameters.csv`, the table every later
stage reads; one csv unit alone gives its part table `pedestal_parameters.csv` or `mpv_parameters.csv`. Examples:
`-only pedestal_plot` draws only `pedestal_fits.pdf`; `-only pedestal_csv` writes only the pedestal table;
`-only plots -from-saved` redraws both pdfs from the cached histograms without reading the csv.

| option | what it controls | default |
|---|---|---|
| `--electron E` / `--muon E` / `--pion E` | the source run: exactly one run, used as given (a non-muon run gets a WARNING in the log) | `calibration.source_dataset` = muon 4 |
| `-pedestal-selection raw_samples\|csv_tot_toa_zero` | no-signal events: the raw TOT / TOA sums of both ends are 0, or the csv `tot`, `toa` columns of both ends are 0 (an approximation) | raw_samples |
| `-pedestal-window ADC` | half width of the pedestal window around the anchor | 200 |
| `-mpv-nsigma N` | the MPV fit uses x = adc_sum - pedestal > N x pedestal_sigma | 5 |
| `-mpv-window-widths K` | final fit window: rough peak +- K x w (w = FWHM / 2.3548) | 2 |
| `-mpv-bin-width ADC` | bin width of the final fit (must divide 2000) | 20 |
| `-fit[a,b]` | final MPV fit window [a, b] in adc_sum - pedestal, the same for every channel | automatic window |
| `-bin N` | with `-fit`: N equal bins on [a, b] | the 20-ADC bins in [a, b] |
| `-workers N` | MPV fits in N processes | 4 |
| `-only UNIT,...` | output units | every unit |
| `-output [NAME.pdf ...]` | save the pdfs (`pedestal_fits.pdf`, `mpv_fits.pdf`, or `KEY=FILE` with the keys `pedestal_plot`, `mpv_plot`) | full set: not saved; `-test`: `results/test/test.pdf`, `test1.pdf` |
| `-from-saved` | refit from the cached histograms without reading the csv (the cache of an earlier calib run with the needed parts) | off |
| `-clean-input` | MPV from the clean datasets of the source run (the pedestal still from the beam csv) | off |

Refused: `-fit` / `-bin` without an MPV unit; `-only` units that leave nothing to make (e.g. `-only plots -no-plots`).

**Inputs**
- the csv files of the source run: `data/analysis_files/<run>/*.csv`, chosen with the run flag or
  `calibration.source_dataset`; `-test`: the `make-csv -test` csv, else the first-5000-events subsets in
  `cache/test/test_subsets/<run>/`. The csv needs the 4 raw TOT / TOA sum columns (make-csv writes them), unless
  `-pedestal-selection csv_tot_toa_zero`.
- `-clean-input`: the clean datasets `data/[test/]clean_dataset/<run>/<csv stem>_clean.csv`
- `-from-saved`: the histogram cache `cache/[test/]pedestal_mpv/calibration_histograms.npz`
- settings: `configure/config.yml`, section `calibration`

**Outputs**
- table: `calibration/pedestal_mpv/calibration_parameters.csv` (both csv units), `pedestal_parameters.csv` or
  `mpv_parameters.csv` (one csv unit), each with its `.provenance.json` (`-test`: `calibration/test/pedestal_mpv/`)
- pdfs: `results/fullset/pedestal_mpv/pedestal_fits.pdf`, `mpv_fits.pdf` (with `-output`); `-test`:
  `results/test/test.pdf`, `test1.pdf` (with `-output`: `results/test/pedestal_mpv/`)
- histogram cache: `cache/[test/]pedestal_mpv/calibration_histograms.npz` (the histograms of this run)
- log: `logs/calibration_parameters/`

## 3. clean: beam csv -> clean dataset

Code: `src/ldmx_hcal/selection/clean_dataset.py` (based on Joe Muse, revised by Lei)

```
.\runner.ps1 clean [--muon 4] [-test] [-muon-standard] [-pedestalscale] [-fit[0.7,1.3] [-bin 15]] [-only clean,tables] [-output fixed_Muon_4GeV_clean_cutflow.csv fixed_Muon_4GeV_clean_layers.csv fixed_Muon_4GeV_clean_diagnostics.pdf] [-no-json] [-dry-run]
```

One clean csv per beam csv: the rows of the clean bars, copied as they are. A hit is a bar end with adc_sum >= the
threshold column of the calibration table; the energy of an end is E_MeV = (adc_sum - pedestal) / mpv x
`mip_energy_mev` (4.83143 MeV per MIP). A later stage reads a clean dataset with `-clean-input` only while it is
current: same beam csv, same pedestal / mpv / threshold values, same selection; otherwise it stops and prints the clean
command that makes it again. `-fit` / `-bin` change only the diagnostics pdf, never the clean dataset.

Units (`-only`): `clean` (the clean csv + provenance), `cutflow`, `layers` (tables), `diagnostics` (pdf); groups
`tables` (cutflow, layers), `results` (cutflow, layers, diagnostics), `all`. Default: every unit. `-from-saved` makes
the result units from the kept-bar cache (it cannot make the clean csv).

| option | what it controls | default |
|---|---|---|
| `--electron E` / `--muon E` / `--pion E` | runs to clean (`configure/datasets.yml`) | every `kind: data` run |
| `-muon-standard` | muon standard: a bar is kept when both ends are hits; an event is kept when every layer holds at most one kept bar and its layer-1 kept bar has E_MeV of both ends in `layer1_energy_window_mev` ([1.84, 7.83] MeV) | for muon runs (`clean_dataset.standard_by_particle`) |
| `-otherparticle` | other-particle standard: a bar is kept when either end is a hit; an event is kept when it has a kept bar | for electron and pion runs |
| `-pedestalscale` | hit threshold `pedestal_range_with_20percent` = 1.2 x pedestal (the 2022 analysis, in sync with that year's MC) | on (`clean_dataset.threshold_mode`) |
| `-pedestalsigma` | hit threshold `pedestal_range_with_5sigma` = pedestal + 5 sigma (Lei's recommendation for future years) | off |
| `-fit[a,b]` / `-fit[a,b],[c,d]` | diagnostics pdf only: per-end E_MIP spectrum fits; one range: Gaussian + Landau on [a, b]; two ranges: double Gaussian + Landau on [a, b] | Gaussian + Landau on `mip_fit_range` [0.7, 1.3] MIP |
| `-bin N` / `-bin N1,N2` | with `-fit`: N equal bins in each fit range | the `mip_spectrum_bins` bins whose centres lie in the range |
| `-only UNIT,...` | output units | every unit |
| `-output [NAME ...]` | save the result units (default file names, or `KEY=FILE` with the key `<run>.<csv stem>.<unit>`, e.g. `Muon_4GeV.fixed_Muon_4GeV.layers=layers.csv`) | full set: not saved; `-test`: `results/test/test.csv`, `test1.csv`, `test.pdf` |
| `-from-saved` | the result units from the kept-bar cache, without reading the beam csv (only its identity is checked) | off |
| `-workers N` | not used (clean runs in one process) | - |

Refused: `-clean-input`; `-from-saved -only clean`; `-only diagnostics -no-plots`.

**Inputs**
- the csv files of each run: `data/analysis_files/<run>/*.csv`; `-test`: the `make-csv -test` csv, else the
  first-5000-events subsets in `cache/test/test_subsets/<run>/`. Columns read: `pf_event`, `layer`, `strip`,
  `adc_sum_end0`, `adc_sum_end1`.
- calibration table: `calibration/pedestal_mpv/calibration_parameters.csv` (`-test`: `calibration/test/pedestal_mpv/`
  when it exists, else the full-set table); columns `pedestal`, `mpv` and the threshold column
- `-from-saved`: the kept-bar cache `cache/[test/]clean_dataset/<run>/<csv stem>_hit_bars.npz` + `.json`
- settings: `configure/config.yml`, section `clean_dataset`; `run.chunksize`

**Outputs** (one set per beam csv)
- clean dataset (unit `clean`): `data/clean_dataset/<run>/<csv stem>_clean.csv` + `<csv stem>_clean.provenance.json`
  (`-test`: `data/test/clean_dataset/<run>/`)
- results (with `-output`): `results/fullset/clean_dataset/<run>/<csv stem>_clean_cutflow.csv`,
  `<csv stem>_clean_layers.csv`, `<csv stem>_clean_diagnostics.pdf`, and the record of the units written
  `<csv stem>_clean_results.json`; a result saved under a name given to `-output` goes to
  `results/fullset/clean_dataset/<name>`; `-test`: `results/test/test.csv`, `test1.csv`, `test.pdf` (with `-output`:
  `results/test/clean_dataset/`)
- kept-bar cache (always): `cache/[test/]clean_dataset/<run>/<csv stem>_hit_bars.npz` + `.json`
- log: `logs/clean_dataset/`

## 4. adcpeak: TOT -> ADC-peak line, Arbitrary Energy, linearity, resolution

Code: `src/ldmx_hcal/energy/tot_adc_peak.py` (shared response fits: `src/ldmx_hcal/energy/energy_response.py`)

```
.\runner.ps1 adcpeak [--electron 0.3 0.4 0.5 1 4] [-test] [-fit-dataset pion 3] [-tot-mode linear] [-slope-window 0.6 4] [-fit-adc-window 949 1021] [-fit-tot-window 500 2000] [-saturation-adc 1022] [-merge-rail-adc 1022] [-quality-threshold 0.125] [-spectrum-fit two_pass] [-fit[3500,6500] [-bin 40]] [-linfit electron[0,1]] [-resfit[0.3,1]] [-normalerror] [-spectrum-bins 90] [-min-bin-count 10] [-window-nsigma 2] [-error-prescription sigma_over_sqrt_n] [-bar-csv] [-output-tag q0125] [-only line,response] [-output tot_adc_peak_fits.pdf energy_response.csv energy_spectra.pdf linearity_resolution.pdf] [-no-json] [-dry-run]
```

The line TOT = slope x adc_max + offset of every channel end is fitted below the ADC rail on the run `-fit-dataset`
(default pion 3), with bins covering the whole `-fit-adc-window`. It turns the TOT of a saturated end into its ADC
peak. The per-event value is the **Arbitrary Energy** [a.u.]: the sum of the bar values of an event (bar value = its
end ADC peaks above the pedestal, merged). It is no energy: only the resolution sigma/mu (unit-free) and the relative
linearity mean something. The event and bar values are intermediates in `cache/`, one file per input csv. The spectrum
of a run is fitted over all its csv files.

Units (`-only`): `line_table`, `line_fits`, `energy_response`, `energy_spectra`, `linearity_resolution`; groups
`line` (line_table, line_fits), `response` (the three response files), `csv` (line_table, energy_response), `plots`
(the three pdfs), `all`. Default: every unit. A run without `line_table` reads this mode's own line table; if it is
missing, the run stops with `.\runner.ps1 adcpeak -only line_table`. With `-from-saved`, `line_table` / `line_fits`
come from the saved fit bins and the response units from the saved events.

A non-default `-tot-mode` or an `-output-tag` adds a variant tag `<v>` (`_none`, `_mean_fallback`, `_<tag>`, joined by
`_`) to the cache and response file names, so the default files are never overwritten. The `-output` names of the
command are the default names of a default run; with a variant they carry `<v>`, or name a result by its key
(`-output energy_response=table.csv`).

| option | what it controls | default |
|---|---|---|
| `--electron E` / `--muon E` / `--pion E` | runs of the response (any runs of `configure/datasets.yml`; the fit run is read in addition) | electron 0.3, 0.4, 0.5, 1, 4 |
| `-fit-dataset PARTICLE E` | run of the line fit (`-fit-dataset pion 3`, or one quoted word `"pion 3"`) | pion 3 |
| `-fit-adc-window LO HI` | adc_max window of the line fit, inclusive; the bins cover all of it | 949 1021 |
| `-fit-tot-window LO HI` | TOT window of the line fit, exclusive (2024 version: 1 500) | 500 2000 |
| `-slope-window LOW HIGH` | usable line: LOW <= slope < HIGH | 0.6 4 |
| `-tot-mode linear\|none\|mean_fallback` | saturated ends: the channel's own line / no TOT (ADC only) / the mean line for channels without a usable line | linear |
| `-saturation-adc ADC` | adc_max >= ADC is saturated (the TOT peak may replace it) | 1022 |
| `-merge-rail-adc ADC` | end merge: an end is at the rail if adc_max >= ADC | = `-saturation-adc` (ci uses 1023) |
| `-quality-threshold X` | keep events with unrecoverable_bar_fraction <= X (0..1) | off (ci uses 0.125) |
| `-spectrum-fit two_pass\|peak_scan` | automatic spectrum fit of each run | two_pass |
| `-fit[a,b]` / `-fit[a,b],[c,d]` | spectrum peak of every run: single Gaussian on [a, b] a.u. (replaces the automatic fit), or a double Gaussian (no linearity / resolution point) | the automatic fit |
| `-bin N` / `-bin N1,N2` | with `-fit`: N equal bins per fit range | the default page bins in the range |
| `-linfit PARTICLE[a,b] ...` | linearity: one straight line per named particle through its points with a <= E <= b GeV (a particle alone: every point of it); `-linfit none`: no line | no line (points only) |
| `-resfit[a,b]` | resolution: fit only the electron points with a <= E <= b GeV | every electron point |
| `-normalerror` | also the linearity / resolution errors x sqrt(chi2/ndf) when chi2/ndf > 1 | off |
| `-spectrum-bins N` | bins of the two-pass fit (also the default `-fit` page) | 90 |
| `-min-bin-count N` | bins with fewer entries are not drawn / fitted | 10 |
| `-window-nsigma K` | two-pass second fit window mu1 +- K sigma1 | 2 |
| `-error-prescription sigma_over_sqrt_n\|sigma_fit_error\|gaussian` | errors of the linearity / resolution points | sigma_over_sqrt_n |
| `-bar-csv` | also write the bar values of every input csv into the cache (input of `plots -energy arbitrary`) | off |
| `-output-tag TAG` | extra suffix of the cache and response files (letters, digits, `.`, `_`, `-`; not `none` or `mean_fallback`) | none |
| `-only UNIT,...` | output units | every unit |
| `-output [NAME ...]` | save the final results (default file names, or `KEY=FILE` with the keys `line_fits`, `energy_response`, `energy_spectra`, `linearity_resolution`) | full set: not saved; `-test`: `results/test/test.pdf`, `test.csv`, ... |
| `-from-saved` | the chosen units from the saved fit bins and events in `cache/[test/]tot_adc_peak/`, no event reconstruction | off |
| `-clean-input` | read the clean dataset of each csv instead of the beam csv | off |
| `-workers N` | not used (adcpeak runs in one process) | - |

Refused (the run, `-dry-run` and `runner check` all stop with the reason):
- `-only` units that leave nothing to make (only pdf units with `-no-plots`)
- `-bar-csv` without a response unit, or with `-from-saved`
- `-fit` / `-bin` / `-linfit` / `-resfit` without a response unit
- `-bin` without `-fit`, or without one value per range; two overlapping `-fit` ranges
- `fit_ranges` / `fit_bins` in the yml (command line only); an unknown key in the yml section `tot_adc_peak`
- an `-output-tag` that is not file-name safe or is reserved

**Inputs**
- the csv files of the fit run (default pion 3) and of the response runs: `data/analysis_files/<run>/*.csv`; `-test`:
  the `make-csv -test` csv, else the first-5000-events subsets in `cache/test/test_subsets/<run>/`
- `-clean-input`: `data/[test/]clean_dataset/<run>/<csv stem>_clean.csv`
- calibration table: `calibration/pedestal_mpv/calibration_parameters.csv` (columns `pedestal_per_time_sample`,
  `pedestal_range_with_20percent`); `-test`: `calibration/test/pedestal_mpv/`, else the full-set table
- line table (without the unit `line_table`): `calibration/[test/]tot_adc_peak/tot_adc_peak_parameters.csv`
- `-from-saved`: `cache/[test/]tot_adc_peak/fit_histograms_<fit run key>.npz` and
  `cache/[test/]tot_adc_peak/<run>/<csv stem>_events<v>.csv` (refused, with the rebuild command, when they do not
  match the current settings, calibration, line table or input csv)
- settings: `configure/config.yml`, sections `tot_adc_peak` and `energy_response`

**Outputs**
- line table (unit `line_table`): `calibration/tot_adc_peak/tot_adc_peak_parameters.csv` + `.provenance.json`
  (`-test`: `calibration/test/tot_adc_peak/`)
- final results (with `-output`), in `results/fullset/tot_adc_peak/`: `tot_adc_peak_fits.pdf` (`line_fits`),
  `energy_response<v>.csv` (`energy_response`), `energy_spectra<v>.pdf` (`energy_spectra`),
  `linearity_resolution<v>.pdf` (`linearity_resolution`); `-test`: `results/test/test.pdf`, `test.csv`, `test1.pdf`,
  `test2.pdf` (with `-output`: `results/test/tot_adc_peak/`)
- cache, in `cache/[test/]tot_adc_peak/`: `fit_histograms_<fit run key>.npz` (fit bins), and per input csv
  `<run>/<csv stem>_events<v>.csv` (Arbitrary Energy per event) and, with `-bar-csv`, `<run>/<csv stem>_bars<v>.csv`
  (bar and end values), each product set with its record
- log: `logs/tot_adc_peak/`

## 5. ci: charge-injection map, TOT scale s and event energies [MIP]

Code: `src/ldmx_hcal/energy/charge_injection.py` (charge injection provided by Erik Wallin, revised by Lei; s parameter,
Lei; shared response fits: `src/ldmx_hcal/energy/energy_response.py`)

```
.\runner.ps1 ci [--electron 0.3 0.4 0.5 1 4] [--muon 4] [--pion 4] [-test] [-adc-only] [-pureci] [-s] [-scale-particle pion muon] [-shrinkage 5] [-quality-threshold 0.125] [-bad-bar-rule used_ends] [-fit[5,10] [-bin 30]] [-linfit electron[0,1] pion] [-resfit[0.3,1]] [-normalerror] [-workers 4] [-output-tag q0p1] [-only figures,checks] [-output energy_response.csv quality_threshold_scan.csv energy_linearity.pdf energy_resolution.pdf e_over_h.pdf] [-no-json] [-dry-run]
```

Three energy scales [MIP], per end:
- **adc_only**: E = (adc_sum - pedestal) / mpv (no TOT).
- **PureCI**: E = max(ADC energy, k q(TOT) / mpv) for an end near the ADC rail, with the charge-injection map q(TOT)
  of the channel (s = 1). Column spelling `pureci`.
- **s**: E = max(ADC energy, k s q(TOT) / mpv), with the per-channel TOT scale s (s parameter, Lei).

The event energy is the sum of its bar energies. s is trained on pion 4 by default. `-scale-particle` picks other
samples (muon, electron, mix); several samples (`-scale-particle pion muon`) give one scale `s_<sample>` each, and `s`
is the sample with the fewest channels that borrow their s from the (DPM, LINK) / global prior. The log says which
sample became `s` and why. With no path flag all three scales are reconstructed. With no run flag the stage
reconstructs the electron runs 0.3, 0.4, 0.5, 1, 4 and the pion runs 0.5, 1, 3, 4. The spectra are fitted internally;
the energy distributions are drawn by plots.

Units (`-only`): `parameters` (`ci_channel_parameters.csv` + `ci_tot_charge_nodes.csv`), `events` (the event csv),
`linearity`, `resolution`, `eh` (pdfs); on request the checks `response` (`energy_response.csv`) and `quality_scan`
(`quality_threshold_scan.csv`). Groups: `figures` (linearity, resolution, eh), `checks` (response, quality_scan),
`all`. Default: parameters, events, figures. `-no-plots` drops the figures, `-from-saved` drops parameters. A unit with
nothing to make for the chosen runs (e.g. `eh` without a pion run) is skipped and logged.
- linearity: one page per scale; every particle's points drawn, lines per `-linfit`. Default lines: adc_only none,
  PureCI the electron points below 1 GeV, s every electron point. Electrons and pions are never fitted together.
- resolution: sigma/mu fitted to the electron points; the pion points are drawn.
- eh: e/h(E) = mu_electron(E) / mu_pion(E) on the s scale, points only: filled where an electron run has the pion
  energy, hollow where mu_electron(E) comes from the electron line of the s scale (e.g. pion 3).

The variant flags `-scale-particle` (anything but pion alone, e.g. `_s_muon`, `_s_pion_muon`), `-output-tag` and
`-clean-input` (`_clean`) add a variant suffix `<v>` to every product file, so they never overwrite the default files.
The `-output` names of the command are the default names of a default run; with a variant they carry `<v>`, or name
a result by its key (`-output response=table.csv`).

| option | what it controls | default |
|---|---|---|
| `--electron E...` / `--muon E...` / `--pion E...` | the runs reconstructed (any run of `configure/datasets.yml`) | electron 0.3, 0.4, 0.5, 1, 4 and pion 0.5, 1, 3, 4 |
| `-adc-only` | reconstruct the adc_only scale | (no path flag: all three) |
| `-pureci` | reconstruct the PureCI scale (charge injection with s = 1) | (no path flag: all three) |
| `-s` | reconstruct the s scale (charge injection x s; s parameter, Lei) | (no path flag: all three) |
| `-selection-path s\|pureci\|adc_only\|auto` | the scale whose bad-strip flags select the events (s is computed even when not requested; auto = the most complete requested scale) | s |
| `-quality-threshold X` | keep events with bad_strip_fraction <= X | 0.125 |
| `-no-quality-filter` | keep every reconstructed event | off |
| `-bad-bar-rule used_ends\|used_ends_strict` | bad bar of the event cut: no end the bar energy uses is trusted and one end is clipped (Lei) / no used end is trusted | used_ends |
| `-scale-particle SAMPLE ...` | training sample(s) of the s fit (s parameter, Lei): pion = pion 4, muon = muon 4, electron = electron 4, mix = the three pooled (runs: yml `charge_injection.scale_training_datasets`) | pion |
| `-shrinkage L` | lambda of the shrinkage weight w = n / (n + L), a number >= 0 or `cv` (s parameter, Lei) | 5 |
| `-min-link-hits N` | a (DPM, LINK) prior of s needs >= N exact-overlap hits, else the global prior | 0 |
| `-tot-sample-mode sum8\|sum8_mask2048` | CI scan TOT of an injection: the sum of the 8 samples, or that sum with the spurious 2048 samples set to 0 first | sum8 |
| `-no-audit-failures` | skip the 1e4-bound pole audit (its failures fail L12 S4-7) | off |
| `-refit` | recompute the CI map and the s fit even when the parameter cache matches | off |
| `-output-tag TAG` | extra suffix of every product file (letters, digits, `.`, `_`, `-`) | none |
| `-fit[a,b]` / `-fit[a,b],[c,d]` | spectrum peaks: single Gaussian on [a, b] MIP on every run (the points then come from these fits), or a double Gaussian (no point) | automatic single Gaussian (electron: two-pass fit; muon / pion: core fit) |
| `-bin N` / `-bin N1,N2` | with `-fit`: N equal bins in each fit range | the default page bins in the range |
| `-linfit PARTICLE[a,b] ...` | linearity lines, one per named particle, on every scale; `-linfit none`: no line | per scale (above) |
| `-resfit[a,b]` | resolution: fit only the electron points with a <= E <= b GeV | every electron point |
| `-normalerror` | also the linearity / resolution errors x sqrt(chi2/ndf) when chi2/ndf > 1 | off |
| `-spectrum-bins N` / `-min-bin-count N` / `-window-nsigma K` / `-error-prescription P` | spectrum and point-error settings (Common modules, energy_response.py) | 90 / 10 / 2 / sigma_over_sqrt_n |
| `-only UNIT,...` | output units | parameters, events, figures |
| `-output [NAME ...]` | save the final results (default file names, or `KEY=FILE` with the keys `response`, `quality_scan`, `linearity`, `resolution`, `eh`) | full set: not saved; `-test`: `results/test/test.csv`, `test.pdf`, ... |
| `-from-saved` | events, figures and checks from the event arrays of an earlier ci run with the same inputs and reconstruction settings (no CI fit, no reconstruction) | off |
| `-clean-input` | read the clean datasets instead of the beam csv (also for the s training runs) | off |
| `-workers N` | runs reconstructed in N parallel processes | 4 |

Refused:
- `-only` units that make nothing in this call (e.g. `-only figures -no-plots`, `-only parameters -from-saved`)
- an `-output-tag` equal to an automatic part: `s_pion`, `s_muon`, `s_electron`, `s_mix` (also with `_...` after
  them), or `clean` / `..._clean`
- a key in the yml sections `charge_injection` / `energy_response` that is not a setting; `-fit` / `-bin` in the yml

**Inputs**
- the csv files of every run: `data/analysis_files/<run>/*.csv`; `-test`: the `make-csv -test` csv, else the
  first-5000-events subsets in `cache/test/test_subsets/<run>/`; `-clean-input`:
  `data/[test/]clean_dataset/<run>/<csv stem>_clean.csv`
- calibration table: `calibration/pedestal_mpv/calibration_parameters.csv` (made by calib)
- charge-injection scans (provided by Erik Wallin):
  `data/charge_injection_scans/scan_DPM0_CALIBRUN_coff14_20220424_220633.csv` and
  `scan_DPM1_CALIBRUN_coff14_20220424_220543.csv` (`charge_injection.scan_files` in `paths.ci_scan_dir`)
- channel map: `configure/channel_map.csv` (`paths.channel_map`)
- training csv files of the s fit (s parameter, Lei): the pion 4 run by default
  (`charge_injection.scale_training_datasets`)
- `-from-saved`: the event arrays `cache/[test/]charge_injection/events<v>/<run>/<csv stem>.npz` + `.json`
- settings: `configure/config.yml`, sections `charge_injection` and `energy_response`

**Outputs** (`<v>` = the variant suffix: `_` + `s_<samples>`, then the `-output-tag`, then `clean`, joined by `_`; empty
for the default run)
- unit `parameters`: `calibration/charge_injection/ci_channel_parameters<v>.csv`, `ci_tot_charge_nodes<v>.csv` and
  `ci_channel_parameters<v>.provenance.json` (`-test`: `calibration/test/charge_injection/`)
- unit `events`: `data/charge_injection/<run>/<csv stem>_event_energies<v>.csv` + `.provenance.json`, one per input
  csv (`-test`: `data/test/charge_injection/<run>/`)
- final results (with `-output`), in `results/fullset/charge_injection/`: `energy_linearity<v>.pdf`,
  `energy_resolution<v>.pdf`, `e_over_h<v>.pdf`, and on request `energy_response<v>.csv`,
  `quality_threshold_scan<v>.csv`; `-test`: `results/test/test.csv`, `test.pdf`, ... (with `-output`:
  `results/test/charge_injection/`); where the last run put them: `cache/[test/]charge_injection/final_results<v>.json`
- caches, in `cache/[test/]charge_injection/`:
  - `parameters<v>/`: CI map + s fit, reused while the inputs and settings are unchanged (always written)
  - `events<v>/<run>/<csv stem>.npz` (per event), `<csv stem>_bars.npy` (per bar, every reconstructed scale) and
    `<csv stem>.json` (record): the input of `-from-saved` and of plots
- log: `logs/charge_injection/`

## 6. plots: figures on one energy scale

Code: `src/ldmx_hcal/visualization/multiscale_plots.py`

```
.\runner.ps1 plots [--pion 4] [-test] [-only dist_event,dist_channel,hitmap] [-energy s] [-variant s_pion_muon] [-picker first] [-n-events 10] [-seed 12345] [-energy-window 0 1000] [-color-scale fixed] [-vmax 10] [-per-end] [-hitmap-events 20] [-min-bin-count 10] [-workers 1] [-output pion_4GeV_dist_event_s.pdf pion_4GeV_dist_channel_s.pdf pion_4GeV_hitmap.pdf] [-no-json] [-dry-run]
```

Every energy figure of the analysis is drawn here, on one scale chosen with `-energy`: `s`, `pureci`, `adc_only`
(ci, [MIP]; `s_<sample>` with several ci training samples) or `arbitrary` (adcpeak Arbitrary Energy [a.u.]). The
values come from the producer cache of every csv of the run (per event, per bar, per end): nothing is reconstructed
here, and only the producer's selected events are drawn. Before anything is drawn, the producer files are checked:
valid record, the scale computed, same mode and test events, the same input csv and calibration table as now. On a
problem the run stops with the producer command, e.g. `.\runner.ps1 ci --pion 4 -only events` or
`.\runner.ps1 adcpeak --pion 4 -only energy_response -bar-csv`. The hit map does not depend on the scale: an end is
hit when adc_sum > `plots.hit_threshold_column` of the calibration table.

Units (`-only`, one pdf per run each): `dist_event` (event energy distribution with the ci spectrum fit drawn,
counted bars per event), `dist_bar` (bar energy distributions, one page per layer), `dist_channel` (channel = bar end
energy distributions, one page per end and layer, one panel per channel), `event_display` (the picked events; with
`-per-end` also the per-end and end0 - end1 pages), `bar_maps` (occupancy, mean energy per hit, mean energy per
event), `layer` (longitudinal profile, per-layer energy and multiplicity), `hitmap` (hit fractions and hit displays,
no energy). Groups: `distributions` (the three dist units), `energy` (every unit but hitmap), `all`. Default: every
unit.

| option | what it controls | default |
|---|---|---|
| `--electron E` / `--muon E` / `--pion E` | runs to draw, one set of pdfs per run (a run of several csv files gives one set) | electron 4 |
| `-energy s\|pureci\|adc_only\|s_pion\|s_muon\|s_electron\|s_mix\|arbitrary` | energy scale | s |
| `-variant TAG` | variant tag of the producer cache (ci: e.g. `s_muon`, `s_pion_muon` or an `-output-tag`; adcpeak: `none`, `mean_fallback` or an `-output-tag`); with `-clean-input` ci's `clean` part is added | the tag of a ci / adcpeak run with the yml settings |
| `-picker first\|highest\|random` | displayed events: the first N, the N highest-energy, N random | first |
| `-n-events N` | event displays per run | 10 |
| `-seed N` | seed of the random picker | 12345 |
| `-events ID [ID ...]` | display these pf_event ids, looked up in every csv of the run (replaces `-picker`) | none |
| `-energy-window LO HI` | pick only events with LO <= E_event < HI | none |
| `-color-scale fixed\|robust` | event-display colour scale: fixed 0..vmax, or per event max(1, min(max(p98, 0.8 x max), 50)) | fixed |
| `-vmax V` | maximum of the fixed colour scale | 10 MIP |
| `-per-end` | add the per-end 2 x 2 display and the end0 - end1 map of every displayed event | off |
| `-hitmap-events N` | per-event hit displays in the hit-map pdf (first N events) | 20 |
| `-min-bin-count N` | event-energy overview: draw only bins with >= N events | 10 |
| `-only UNIT,...` | output units | every unit |
| `-output [NAME.pdf ...]` | save the pdfs (default file names `<run key>_<unit>_<scale>[_<tag>].pdf`, `<run key>_hitmap.pdf`; or `KEY=FILE` with the key = the default file name without `.pdf`) | full set: not saved; `-test`: `results/test/test.pdf`, `test1.pdf`, ... |
| `-from-saved` | redraw from the reduced arrays in `cache/[test/]multiscale_plots/` (a unit or event the stored run did not keep is refused) | off |
| `-clean-input` | the hit map reads the clean datasets; the scales read the `-clean-input` producer cache | off |
| `-workers N` | pdfs rendered in N processes | 1 (`plots.workers`) |

**Inputs**
- ci scales: `cache/[test/]charge_injection/events<v>/<run>/<csv stem>.npz`, `_bars.npy`, `.json` (written by ci with
  an event unit)
- `arbitrary`: `cache/[test/]tot_adc_peak/<run>/<csv stem>_events<v>.csv` and `_bars<v>.csv` (written by
  `adcpeak -bar-csv` with a response unit)
- hit map: the csv files of each run (`data/analysis_files/<run>/*.csv`; `-test`: the test csv or subset) and the
  calibration table `calibration/[test/]pedestal_mpv/calibration_parameters.csv`
- `-from-saved`: `cache/[test/]multiscale_plots/<run key>_energy_<scale>[_<tag>].npz`, `<run key>_hitmap.npz`, each
  with its `.json`
- settings: `configure/config.yml`, section `plots` (and the `charge_injection` selection settings for the ci scales)

**Outputs**
- pdfs (with `-output`): `results/fullset/multiscale_plots/<run key>_<unit>_<scale>[_<tag>].pdf`,
  `<run key>_hitmap.pdf`; `-test`: `results/test/test.pdf`, `test1.pdf`, ... (with `-output`:
  `results/test/multiscale_plots/`)
- reduced arrays (for `-from-saved`; `plots.save_cache: false` stops them):
  `cache/[test/]multiscale_plots/<run key>_energy_<scale>[_<tag>].npz` + `.json`, `<run key>_hitmap.npz` + `.json`
- log: `logs/multiscale_plots/`

## Runner

Code: `src/runner.py`

| launcher | shell | note |
|---|---|---|
| `.\runner.ps1` | PowerShell | used in every example |
| `.\runner.cmd` | cmd.exe and PowerShell | same arguments; works when PowerShell refuses `.ps1` scripts |
| `./runner.sh` | bash (Linux, macOS, Docker) | uses `python3` unless `LDMX_PYTHON` is set |

All three run `python -X utf8 src/runner.py` with the same arguments and pass on its exit code. `LDMX_PYTHON` picks the
interpreter (default: `python` on PATH). Commands: the stages `make-csv`, `calib`, `clean`, `adcpeak`, `ci`, `plots`
(sections 1-6), plus `setup`, `check` and `all` below. Every option can be written as `-flag` or `--flag`
(`-test` = `--test`). Abbreviated options are refused. Help: `-h`, `--help` or `-help`, e.g. `.\runner.ps1 ci -help`.

### setup: point the analysis at the data folder

Code: `src/runner.py` (`run_setup`)

```
.\runner.ps1 setup [D:/LDMX/data] [-protect D:/old_analysis]
```

With no argument, `setup` only shows the current `data_dir` and where it comes from (the `LDMX_DATA_DIR` environment
variable, `configure/local.yml`, or `configure/config.yml`), what the folder holds and which runs of
`configure/datasets.yml` have csv files.

| option | what it controls | default |
|---|---|---|
| `DATA_DIR` | folder that holds `root/`, `analysis_files/` (one sub-folder per run) and `charge_injection_scans/`; written as `paths.data_dir` into `configure/local.yml` | not given: show the current setting |
| `-protect DIR ...` | folders no stage may ever write into, added to `safety.protected_dirs` of `local.yml`; refused when it would cover a folder the stages write (also `analysis_files_dir`, where make-csv writes) | - |

**Outputs**: `configure/local.yml`; log in `logs/runner/`.

### check: every input and output of every stage

Code: `src/runner.py` (`run_check`, `print_plan`)

```
.\runner.ps1 check [--muon 4] [-test] [-strict]
```

`check` prints the input / output plan of every stage with its default options. The common options change the plan
being checked exactly as they would change a run. Inside `check`, calib keeps its source run. A stage that cannot read
the clean dataset, or that has nothing stored to restart from, is checked as a normal run. No stage runs and no file
is written except the run log.

| option | what it controls | default |
|---|---|---|
| `-strict` | also fail (exit 3) when inputs made by earlier stages are missing or not current | off |

| status | meaning |
|---|---|
| `ok` | the input exists (an output: it exists and will be replaced) |
| `new` | an output that will be created |
| `MISSING` | an external input (data file, constant) is absent; for a test subset: its full csv is absent |
| `PENDING` | an input made by an earlier stage does not exist yet |
| `STALE` | an input made by an earlier stage exists but is not current |
| `AUTO` | a test subset that the next `-test` run builds (or rebuilds, with the reason) from its full csv |
| `reuse` / `rebuild (reason)` | a stored intermediate in `cache/` that can be reused, or why it cannot |

| exit code | meaning |
|---|---|
| 0 | every external input exists |
| 1 | a stage cannot be loaded (import error, or its options clash with the common ones) |
| 2 | an external input is missing: fix `configure/local.yml` |
| 3 | only with `-strict`: products of earlier stages are missing or stale |

### all: the chain

Code: `src/runner.py` (`main`)

```
.\runner.ps1 all [--electron 4] [-test] [-with-make-csv] [-stages calib clean adcpeak ci plots] [-workers 4] [-no-plots] [-output] [-no-json] [-dry-run]
```

`all` runs calib -> clean -> adcpeak -> ci -> plots, each stage with its own defaults, default units and default runs.
`-with-make-csv` puts make-csv first. `-stages` picks the stages explicitly (they still run in chain order), and then
`-with-make-csv` is ignored. The run flags and the common options go to every stage, with one exception: calib keeps
its source run `calibration.source_dataset` (muon 4), and the log says so. `all -clean-input` runs make-csv, calib and
clean on the beam csv, then adcpeak, ci and plots on the clean datasets; when runs are chosen, clean also cleans the
adcpeak fit run and the ci s training runs. `all -from-saved` needs an earlier normal run; a stage with nothing stored
runs normally.

| option | what it controls | default |
|---|---|---|
| `-with-make-csv` | run make-csv first (start from the ROOT files) | off |
| `-stages STAGE ...` | explicit list of stages, run in chain order | calib clean adcpeak ci plots |
| `-output` | save every stage's final results under default names (no names here) | off |

Refused: `-output NAME ...` (names belong to one stage) and stage options (`-only`, `-fit`, `-bar-csv`, ...): give them
to the stage itself after the chain.

**Outputs**: the products of every stage; one log per stage in `logs/<module>/` and a summary in `logs/runner/`.

### Common options

Every stage, `all` and `check` take these (`-only` is per stage):

| option | what it controls | default |
|---|---|---|
| `--electron E ...` / `--muon E ...` / `--pion E ...` | runs to process, by particle and beam energy in GeV (`configure/datasets.yml`); the flag alone = every run of that particle; an energy belongs to the particle flag in front of it (`--electron 0.3 1 --pion 4`). Also spelled `-electron`, `--Electron`, `--electrons`; energies as `0.3`, `0.3GeV`, `300MeV` | the stage's default runs |
| `-test` | test mode: the first N events of every input csv; products in the `test/` folder of every tree | off |
| `-test-events N` | N of `-test` (positive integer) | `run.test_events` = 5000 |
| `-only UNIT,...` | the output units of the stage (single units or groups, comma or space separated; a second `-only` adds to the first) | the stage's default units |
| `-no-plots` (`-noplots`) | no figures: drops the pdf units | off |
| `-workers N` | worker processes (what runs in parallel depends on the stage) | `run.workers` = 4; plots: `plots.workers` = 1 |
| `-dry-run` | print every input and output with its status (as `check`); run nothing, write nothing but the log | off |
| `-no-json` | write no record json (provenance, results record) next to the products in `data/`, `calibration/` and `results/`; the records in `cache/` stay. Checks that need a missing record do not verify | off (json on) |
| `-clean-input` | read the clean dataset of each csv of a run instead of the beam csv | off |
| `-from-saved` | restart from the intermediates the stage stored in `cache/[test/]<stage>/`, skipping its expensive part | off |
| `-output [NAME ...]` | which final results are saved, and under which name (rule below) | full set: not saved; `-test`: `results/test/test.csv`, `test.pdf`, ... |
| `-config YML` | another configuration file (the `local.yml` next to it overrides it) | `configure/config.yml` |

Refused: positional run names (`muon_4GeV`, `all`, `"muon 4"`): choose runs with the run flags. Invalid run options
(`-test-events`, `-workers`, and `run.test_events`, `run.workers`, `run.chunksize` in the yml) stop every command, with
the reason, before anything runs.

**Run names.** In yml settings and messages a run is written by its reference, `<particle> <energy in GeV>`, with the
full particle name: `muon 4`, `electron 0.3`, `pion 0.5`. With several runs at one energy the reference is
`electron 0.3 run 304`; a simulation run is `mc muon 4`. The run key `<particle>_<E>GeV` (`muon_4GeV`,
`electron_0.3GeV_run304`, `mc_muon_4GeV`) is derived from it and names log lines, provenance records and the plots
pdfs. The run folder `<Particle>_<E>GeV[_run<N>]` (`Muon_4GeV`, `MC_Muon_4GeV`) holds the run's csv files and names the
per-run sub-folders of the products. Any run of `datasets.yml` is accepted wherever runs are taken; whether it suits
the analysis is not checked, and the run log records it. The run flags choose data runs only; simulation files are
converted with `make-csv -mc -input FILE`.

**-test.** Each stage reads, per csv, the `make-csv -test` csv `data/test/analysis_files/<run>/<csv name>` while its
provenance still matches. Otherwise it reads the exact subset
`cache/test/test_subsets/<run>/<csv stem>_first<N>events.csv`: every row of the N smallest `pf_event` ids, copied as
it is and rebuilt when its source changes. With `-clean-input` it reads `data/test/clean_dataset/<run>/` instead. The
full-set products are never touched.

**-output.** Final results are the figures and result tables of `results/<mode>/<stage>/`. Intermediates
(calibration tables, clean datasets, event csv, caches) are always written, so the chain never breaks.

| command | final results |
|---|---|
| full set, no `-output` | not saved: drawn into the scratch folder `cache/<stage>/unsaved/` (emptied at every run), nothing in `results/` |
| `-output` alone | every final result under its default name in `results/fullset/<stage>/` |
| `-output NAME ...` | a name saves the result whose default file name (or result key + extension) it is; `KEY=FILE` saves result KEY as FILE; naming one file saves only that file, the others go to the scratch folder. Names only, no folder. A name that matches several results, a result named twice, or `KEY=FILE` with the wrong extension is refused |
| `-test`, no `-output` | `results/test/test.csv`, `test1.csv`, ... and `test.pdf`, `test1.pdf`, ..., numbered per type; the next test run overwrites them |
| `-test -output NAME ...` | the names in `results/test/<stage>/` |

A result saved under a user or test name gets its provenance as `<name>.provenance.json` (not with `-no-json`).

**-from-saved.** calib, clean, adcpeak, ci and plots store intermediates in `cache/[test/]<stage>/` during a normal
run. With `-from-saved` a stage restarts from them, provided they still match the inputs and settings they depend on;
otherwise it stops with the reason and the command that rebuilds them. make-csv refuses `-from-saved`; inside `all`
and `check` such a stage runs normally.

**-clean-input.** adcpeak, ci and plots read the clean dataset of each csv instead of the beam csv. calib fits the MPV
on the clean datasets and the pedestal on the beam csv. make-csv and clean refuse it. A clean dataset is used only
while it is current; otherwise the run stops with `Make it with: .\runner.ps1 clean <run flag> [-test]`.

### Folder layout

| | normal run | `-test` |
|---|---|---|
| beam csv (make-csv) | `data/analysis_files/<run>/` | `data/test/analysis_files/<run>/` |
| csv generated from the beam data | `data/<stage>/<run>/` | `data/test/<stage>/<run>/` |
| calibration tables | `calibration/<stage>/` | `calibration/test/<stage>/` |
| intermediates (cache, test subsets, scratch copies) | `cache/<stage>/` | `cache/test/<stage>/` |
| final results (figures, result tables) | `results/fullset/<stage>/` | `results/test/<stage>/`, `results/test/test*.csv / .pdf` |
| run logs | `logs/<module>/` | `logs/<module>/` |

Deleting `cache/` never removes an official output.

## Common modules

### src/ldmx_hcal/common/

- **`config.py`**: loads `config.yml` + `local.yml` + `LDMX_DATA_DIR` and expands `${name}`; reads the runs of
  `datasets.yml` (fields `root`, `runs`, `label`), applying a `datasets:` override per run. Run references
  (`parse_run_ref`, `run_ref`), run keys (`run_key`), run folders (`run_folder`, `root_csv_name`), the run flags
  (`select_by_particle`, `run_flags`) and run lookup (`resolve_runs`, `run_setting`). `RunContext` carries the common
  options (`units` and `wants(unit)` for `-only`, `write_json` for `-no-json`) and builds every path of both modes:
  `run_dir`, `run_csvs`, `root_csv`, `input_csv` / `beam_csv` (per csv: clean dataset or beam csv; with `-test` the
  make-csv test csv or the test subset), `csv_product` (one product per input csv, in a sub-folder per run),
  `clean_product`, `calibration_file`, and the plan items for `check`. `EVENT_ID_COLUMNS` = (`event_source`,
  `pf_event`) identifies an event across the csv files of a run.
- **`io_utils.py`**: chunked csv reading with compact dtypes and checked channel keys (`iter_csv`,
  `read_csv_columns`). Write protection for `safety.protected_dirs` (`ensure_writable`). One rule each for settings
  (`validate_settings`) and channel keys (`validate_channel_keys`: finite integers, layer 1-19, strip 0-7 / 0-11,
  end 0/1, each channel once). Atomic writers (`atomic_write_csv`, `ChunkedCsvWriter`, `atomic_pdf`, `write_json`)
  with a temp-file registry that the runner puts around every stage. Product sets (`ProductSet`,
  `product_set_valid`, `require_product_set`): the data files first, then the record json last, with each member's
  size, mtime and sha256; `-no-json` switches the records outside `cache/` off (`set_json_records`). The test subsets
  (`build_test_subset`, `subset_status`), and `log`.
- **`outputs.py`**: `add_only_argument` (the `-only` option of a stage: units, groups, defaults) and `selected_units`;
  the `-output` option and `OutputPlan`, which decides where each final result of a stage run goes: names matched by
  result, `KEY=FILE`, default names, test names, or the unsaved scratch copy in `cache/<stage>/unsaved/`.
- **`fitting.py`**: model functions: Gaussian, Landau (CERNLIB DENLAN = ROOT `TMath::Landau`), Landau (x) Gauss (as
  ROOT `langaus.C`), line, and the resolution model sqrt(a^2/E^2 + b^2/E + c^2). Fitters: `fit_curve`,
  `fit_binned_poisson` (deviance), `weighted_linear_fit`, and errors only from a positive-definite Hessian. The `-fit`
  / `-bin` / `-resfit` options: `normalize_range_args` re-joins ranges that PowerShell splits; `parse_ranges` and
  `parse_bins` read them; `range_gauss_fit` fits a single or double Gaussian on the user ranges, and a failed fit stays
  failed.
- **`calibration_io.py`**: schema of the calib tables (`CALIBRATION_COLUMNS` with the meaning of every column; the
  full table and the two part tables), `load_calibration` (keys checked; implausible values reported, never changed)
  and lookups `arr[layer, strip, end]`.
- **`geometry.py`**: prototype geometry: 19 layers, 8 bars in layers 1-9 and 12 in layers 10-19, 2 ends: 192 bars,
  384 channels. Drawing positions (the 8-bar layers sit on positions 2-9), channel lists and layer grids.
- **`plotting.py`**: matplotlib helpers (Agg backend, pdf pages, detector layer maps, step histograms, fit boxes).
- **`run_log.py`**: the run log of every command ([logs/README.md](../logs/README.md)).

### src/ldmx_hcal/energy/energy_response.py: shared energy-response fits (ci and adcpeak)

This module holds the spectrum, linearity, resolution and e/h fits and their figures, used by ci (MIP) and adcpeak
(Arbitrary Energy, a.u.). The inputs are arrays of event values and the beam energy and particle of each run. It writes
no file itself: the calling stage saves the tables and pdfs. A fit never raises on low statistics: a fit that cannot
be done gives NaN values, `ok` = false and a `status` text; there is no fallback value.

**Quantities**
- mu, sigma: mean and width of the Gaussian fitted to the peak of a run
- linearity: mu against the beam energy E, every particle drawn
- resolution: sigma/mu against E
- e/h(E) = mu_electron(E) / mu_pion(E) on one scale (ci: s)

**Spectrum fits**
- `two_pass_gauss_fit`: 90 bins on [0, max(3, 1.08 x p99.8)], fitted only on bins with >= 10 entries (sqrt(N)
  errors). Pass 1 uses all of these bins; pass 2 uses those in mu1 +- 2 sigma1. Pass 2 is the result. Provided by
  Erik Wallin. Default of ci and adcpeak.
- `peak_scan_fit`: 300 bins on [0, max(3, 1.12 x p99.7)], a smoothed peak search, then a Gaussian + constant fit in
  mu0 +- 2.5 sigma0. Provided by Erik Wallin.
- `range_spectrum_fit` (`-fit`): single Gaussian on [a, b], or a double Gaussian on [a, b], [c, d]. The Gaussian is
  integrated over each bin, with a Neyman chi2. A parameter that ends on its bound means the fit failed, and the page
  then shows only the points.

**Linearity, resolution, e/h**
- `linearity`: one weighted straight line mu = slope x E + intercept per particle named by `-linfit` (or the scale's
  default), through its points with a <= E <= b GeV (inclusive); electrons and pions are never fitted together. The
  points of the other particles are drawn only.
- `resolution_fit`: sigma/mu = sqrt(a^2/E^2 + b^2/E + c^2) with E in MeV, fitted to the electron points; `-resfit[a,b]` fits only a <= E <= b GeV. The hadron points are drawn only.
- `e_over_h`: one row per pion run. Filled point: an electron run at that energy (several: their mean). Hollow point:
  mu_electron from the electron line, its error from the line covariance. Otherwise no point, with a note.
- `point_errors`: mu_error = sigma / sqrt(N). The sigma/mu error follows `error_prescription`: `sigma_over_sqrt_n`,
  `sigma_fit_error` or `gaussian`.
- `-normalerror`: the parameter errors are also given multiplied by sqrt(chi2/ndf) when chi2/ndf > 1.

**Figures**
- `plot_spectrum_with_fit`, `plot_linearity` (every particle its own colour and marker: electron blue o, pion orange
  s, muon green ^; fitted points filled, the others hollow), `plot_resolution`, `plot_eh`, `summary_table_page`
- A run without a point is named on the page ("no point: 0.3 GeV (reason)") and marked with a red x.

**Shared flags** (`add_fit_arguments`; registered by ci and adcpeak):

| option | what it controls | default |
|---|---|---|
| `-fit[a,b]` / `-fit[a,b],[c,d]` | spectrum peak: single / double Gaussian on the range(s), in the x units of the stage's page (ci: MIP, adcpeak: a.u.) | the automatic fit of each page |
| `-bin N` / `-bin N1,N2` | with `-fit`: N equal bins per range (single N >= 4; double >= 3 per range and >= 7 in all) | the default page bins in the range |
| `-linfit PARTICLE[a,b] ...` | linearity: one line per named particle (electron, muon, pion); a particle alone uses every point of it; a second `-linfit` adds particles; `-linfit none`: no line | ci: adc_only none, PureCI electron[0,1], s electron; adcpeak: none |
| `-resfit[a,b]` | resolution fitted on the electron points a <= E <= b GeV only | every electron point |
| `-normalerror` | also errors x sqrt(chi2/ndf) when chi2/ndf > 1 | off |
| `-spectrum-bins N` | bins of the two-pass fit | 90 |
| `-min-bin-count N` | bins with fewer entries are not drawn / fitted | 10 |
| `-window-nsigma K` | two-pass second window mu1 +- K sigma1 | 2 |
| `-error-prescription P` | sigma_over_sqrt_n \| sigma_fit_error \| gaussian | sigma_over_sqrt_n |

Rules for the range flags:
- `-fit`, `-bin` and `-resfit` are given once; two ranges go in one flag: `-fit[5,10],[20,30]`.
- PowerShell splits `-fit[5,10]` and `electron[0,1]` at the comma, and the runner joins them back.
- In bash / zsh, quote the brackets (`'-fit[5,10]'`, `'electron[0,1]'`) or write `-fit 5,10`.

**Settings**: `configure/config.yml`, section `energy_response` (shared by ci and adcpeak):
- `spectrum_bins`, `min_bin_count`, `window_nsigma`, `min_fit_bins`, `hist_upper_*`, `peak_scan_*`
- `linearity_fits` (`{particle: [a, b] or null}`; null = the default of the scale; `{}` = no line),
  `resolution_fit_range` (GeV)
- `linearity_weighted`, `normal_error`, `error_prescription`
- `resolution_p0`, `resolution_bounded`, `resolution_maxfev`

Refused in that section: an unknown key (a typo such as `normalerror`), and `fit_ranges` / `fit_bins` (command line
only, because the ci and adcpeak spectra have different x axes).

## Installation

Python 3.10 or newer (developed with Python 3.12.2 on Windows 11). From the `data_analysis` folder:

```powershell
pip install -r env/requirements.txt -c env/requirements-lock.txt
```

`env/requirements-lock.txt` pins the exact versions the analysis was validated with; drop `-c ...` to take newer
releases.

**Windows PowerShell execution policy.** The Windows default blocks `.ps1` scripts, and `.\runner.ps1` then fails with
"running scripts is disabled on this system". Use `.\runner.cmd` with exactly the same arguments, or
`python -X utf8 src\runner.py ...`. Enabling scripts with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
changes a security setting of your Windows account, so that is your own choice; the analysis does not need it.

In PowerShell, write a value that contains `${...}` in single quotes (e.g. `'${root_dir}\file.root'`), so that
PowerShell does not expand it.

## A new machine, step by step

1. Install the packages (Installation).
2. Point at the data: `.\runner.ps1 setup D:/LDMX/data` writes `paths.data_dir` into `configure/local.yml`. The folder
   holds `root/`, `analysis_files/` (one sub-folder per run) and `charge_injection_scans/`. Protect original folders
   the paths point into with `-protect`. Optional: check the input files from the data folder with
   `sha256sum -c <data_analysis>/env/data_inputs_sha256.txt`.
3. Check the chain: `.\runner.ps1 check` and `.\runner.ps1 check -test` (exit 0 = every external input present).
4. Make the beam csv: `.\runner.ps1 make-csv` (every run whose ROOT file is on this machine).
5. Smoke test on the first 5000 events: `.\runner.ps1 all -test -workers 1`.
6. The full chain: `.\runner.ps1 all -output`, or the stages one by one (sections 1-6).

## Tests

`src/tests/run_all.py` runs every test module in its own Python process (memory is returned between modules and no
state leaks from one module into the next), with the Agg matplotlib backend and single-threaded BLAS; pytest also
works if installed. A test is a module-level `test_*()` function or a `unittest.TestCase`; a test whose data are not on
the machine is reported as `SKIP`. A module whose process dies without a result (crash, out of memory, `--timeout`)
counts as one failure. The exit code is 1 when a test fails. The tests use synthetic data and move every product tree
(data, calibration, results, cache) into a temporary folder, so no test touches the repository's products.

```powershell
python -X utf8 src\tests\run_all.py                              # all fast tests, one process per test file
python -X utf8 src\tests\run_all.py calibration clean            # only test files whose name contains one of the words
python -X utf8 src\tests\run_all.py --timeout 600                # a test file running longer than 600 s fails
python -X utf8 src\tests\run_all.py --in-process common          # everything in this process (debugging)
$env:LDMX_SLOW_TESTS=1; python -X utf8 src\tests\run_all.py      # also the slow tests
```

## Container

`env/Dockerfile` builds a reproducible image: `python:3.12-slim`, the packages of `env/requirements.txt` with the
pinned versions of `env/requirements-lock.txt`, single-threaded BLAS and the `Agg` matplotlib backend. At build time
`env/docker.yml` is copied to `configure/local.yml`, so inside the container `paths.data_dir` is `/data`. The entry
point is `python src/runner.py`, so the container takes the same arguments as the launchers (default: `--help`).
`.dockerignore` keeps `data/`, the fitted `calibration/` tables, `results/`, `cache/`, `logs/`,
`configure/local.yml`, pdf / png files and `.git` out of the image. Build from the `data_analysis` folder:

```bash
docker build -f env/Dockerfile -t ldmx-hcal-analysis .
docker run --rm -v /path/to/data:/data \
    -v "$PWD/calibration:/app/calibration" -v "$PWD/results:/app/results" -v "$PWD/cache:/app/cache" \
    ldmx-hcal-analysis all -test -workers 1
```

| host | container | content |
|---|---|---|
| the folder with `root/`, `analysis_files/`, `charge_injection_scans/` | `/data` | inputs; the generated csv go to `/data/<stage>/` and `/data/test/<stage>/` |
| `data_analysis/calibration` | `/app/calibration` | calibration tables (keep `reference/` inside: make-csv reads the TOA constants there) |
| `data_analysis/results` | `/app/results` | figures and result tables |
| `data_analysis/cache` | `/app/cache` | intermediates; without this mount every `-test` run rebuilds the subsets and `-from-saved` finds nothing |
