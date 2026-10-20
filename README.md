# data_analysis

HCal calibration: the full analysis of the LDMX 2022 test-beam HCal prototype data.

## How to run

From the `data_analysis` folder:

```
$env:LDMX_PYTHON = "..\venv312\Scripts\python.exe"
.\runner.ps1 <stage> [options]
```

Stage order: make-csv -> calib -> clean -> adcpeak / ci -> plots. Runs are chosen with `--electron E`, `--muon E` or
`--pion E` (beam energy E in GeV). The launchers `runner.ps1` (PowerShell), `runner.cmd` (cmd.exe) and `runner.sh`
(bash) take the same arguments. Every stage, its options and outputs, the installation, the tests and the container:
[src/README.md](src/README.md).

## src/

The code: the runner `runner.py` and the package `ldmx_hcal`. Details: [src/README.md](src/README.md).

### ldmx_hcal/conversion/

make-csv: ROOT ntuple -> beam csv.

### ldmx_hcal/calibration/

calib: pedestal and MIP MPV of every channel.

### ldmx_hcal/selection/

clean: the clean datasets that the later stages read with `-clean-input`.

### ldmx_hcal/energy/

adcpeak (Arbitrary Energy) and ci (charge injection, TOT scale s, event energies in MIP), with their shared spectrum,
linearity and resolution fits.

### ldmx_hcal/visualization/

plots: the per-run figures on one energy scale.

### ldmx_hcal/common/

The shared code: configuration, input / output, fits, figures, geometry and the run logs.

### tests/

The test suite (`run_all.py` and one test file per module).

## configure/

The configuration: the settings of every stage, the run list, the paths of this machine and the channel map.
Details: [configure/README.md](configure/README.md).

## env/

The environment: the Python package lists, the container recipe with its configuration, and the checksums of the
input files.

## data/

The inputs and the event-level csv the stages make (not in git). Variables: [data/README.md](data/README.md).

### root/

The decoded HGCROC ntuples (ROOT) of every run, read by make-csv.

### analysis_files/

The beam csv files: one sub-folder per run, one csv per ROOT file (make-csv).

### charge_injection_scans/

The charge-injection scans (provided by Erik Wallin), read by ci.

### clean_dataset/

The clean datasets: one csv per beam csv, in a sub-folder per run (clean).

### charge_injection/

The ci event energies on the three scales: one csv per beam csv, in a sub-folder per run (ci).

### test/

The same for `-test` runs, which read the first 5000 events.

## calibration/

The per-channel calibration tables. Variables: [calibration/README.md](calibration/README.md).

### reference/

Fixed constants kept in git: the TOA constants of every bar (Joe Muse), read by make-csv.

### pedestal_mpv/

The pedestal and MIP MPV of every channel (calib).

### tot_adc_peak/

The TOT -> ADC-peak line of every channel end (adcpeak).

### charge_injection/

The charge-injection map and the TOT scale s of every channel (ci).

### test/

The same for `-test` runs.

## results/

The final results: figures and result tables (not in git). Variables: [results/README.md](results/README.md).

### fullset/

The final results of normal runs, one folder per stage:
- `make_csv_qa/`: the per-run distribution pdfs (QA) of the beam csv
- `pedestal_mpv/`: the pedestal and MPV fit pdfs
- `clean_dataset/`: the cut flow, the per-layer table and the diagnostics pdf of every clean dataset
- `tot_adc_peak/`: the line-fit pdf, the response table, the spectra and the linearity / resolution figures
- `charge_injection/`: the linearity, resolution and e/h figures, and the check tables on request
- `multiscale_plots/`: the per-run distributions, event displays, maps and hit maps

### test/

The same for `-test` runs.

## cache/

The intermediates that `-from-saved` restarts from, and the test subsets (not in git; safe to delete).

### \<stage\>/

What a stage stores in a normal run, and `unsaved/`: the final results of a run without `-output`.

### test/

The same for `-test` runs, and `test_subsets/`: the first-N-event subsets of the beam csv.

## logs/

One log per run (not in git). Details: [logs/README.md](logs/README.md).

### \<module\>/

The logs of one stage module (`runner/`: `all`, `check` and `setup`), with an index of its runs.

## Authors

make-csv, calib and clean: created by Joe Muse (`makeAnalysisFiles.py` + `apply_calibrations.py`;
`calculatePedestals.py` + `calculateMIPs.py`; `selectCleanEvents.py`), revised by Lei. ci (charge injection): provided
by Erik Wallin, revised by Lei. **The s parameters (the per-channel TOT scale s, its training and fit): created by
Lei.** adcpeak, plots, the runner and the shared code (`src/ldmx_hcal/common/`): created by Lei.
