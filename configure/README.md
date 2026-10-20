# Configuration

## 1. config.yml

The one configuration file. Relative paths are relative to `data_analysis/`, and `${name}` expands to another entry of
`paths`. A command-line flag overrides a value for one run; `configure/local.yml` overrides any key on one machine. The
stage sections must equal the `DEFAULTS` of the modules (`src/tests/test_config_defaults.py`); an unknown key stops
make-csv, adcpeak, ci and plots (clean warns). Write small floats as `1.0e-12`, because PyYAML reads `1e-12` as a
string.

| section | what it holds |
|---|---|
| `paths` | inputs: `data_dir` (base of every input; `runner setup`, `LDMX_DATA_DIR`), `datasets_file` (`configure/datasets.yml`), `root_dir`, `analysis_files_dir` (the run folders; make-csv writes there), `ci_scan_dir`, `channel_map`, `toa_calibration`; products: `generated_data_dir` (data/), `calibration_dir`, `results_dir`, `cache_dir`; optional `logs_dir` |
| `safety` | `protected_dirs`: folders no stage may write into (checked by every writer, also in worker processes) |
| `run` | `test_events` (5000), `chunksize` (csv rows per chunk, 2000000), `workers` (4) |
| `make_csv` | make-csv settings (tree names, chunk sizes, waveform, TOT mode `tot_branch`, repeated digis `error` \| `drop`, sample window, TOA table and `missing_toa_calibration` `drop` \| `error`, csv format, QA pdf bins) |
| `calibration` | calib settings (table / cache names, source run `"muon 4"`, pedestal selection / anchor / window, MPV cut, bins, fit windows and bounds, langaus convolution) |
| `clean_dataset` | clean settings (standard per particle, threshold mode, MIP energy, diagnostics) |
| `tot_adc_peak` | adcpeak settings (line-fit run `"pion 3"`, its windows and bins, TOT mode, slope window, rail thresholds, quality cut, spectrum fit `two_pass` \| `peak_scan`, `write_bar_csv`, output tag) |
| `charge_injection` | ci settings (scales `energy_paths` [adc_only, pureci, s], `selection_path` s, quality cut, scan files, CI fit constants, Erik Wallin's TOT-threshold calibration `thr_*` and `audit_failures`, the s training sample(s) and runs `scale_particle` / `scale_training_datasets` (s parameter, Lei), shrinkage) |
| `energy_response` | spectrum / linearity / resolution fit settings shared by ci and adcpeak; `linearity_fits` ({particle: [a, b] GeV or null}; null = the default of the scale) ([src/README.md](../src/README.md), Common modules) |
| `plots` | plots settings (energy scale `energy_source` s, producer `variant` (null = the tag of a default ci / adcpeak run), hit-map threshold column, event displays, colour scales, bins, workers, cache) |
| `datasets` (optional) | changes single runs of `datasets.yml` (key: run reference or run key; fields `root`, `runs`, `label`, and `particle`, `energy_gev`, `kind` for a new run) |

The output units of a stage are chosen on the command line only (`-only`), not in the yml.

## 2. datasets.yml

The run registry, listed by particle and beam energy. Adding data means adding an entry; no code changes:

```yaml
<particle>:            # electron | muon | pion
  <energy>:            # e.g. 0.3GeV (also 0.3, 300MeV)
    root: ${root_dir}/fixed_electron_0.3GeV.root   # ntuple(s) for make-csv (optional, a path or a list)
    runs: [304]                                    # run number(s)
    label: Electron 0.3 GeV (run 304)              # display name (optional)
```

A run is the folder of its csv files, named from its place in the file: `data/analysis_files/<Particle>_<E>GeV/`
(`electron: 0.3GeV:` -> `Electron_0.3GeV/`). make-csv writes one csv per ROOT file of the run into it, named after the
ROOT file; a run without `root` is read from the csv files already in its folder (an energy with no field is allowed,
e.g. `4GeV:`). Several runs at one energy: a list of entries, each with its own `runs` (reference
`electron 0.3 run 304`, key `electron_0.3GeV_run304`, folder `Electron_0.3GeV_run304/`; `electron 0.3` alone names
all of them). Simulation: the same layout under `mc:` (reference `mc muon 4`, key `mc_muon_4GeV`, folder
`MC_Muon_4GeV/`). Run keys and folders are derived from the particle and energy, never written in the file. A key
given twice, an unknown field or a bad energy stops with the file and entry named. On the command line a run is chosen
as `--muon 4`; in yml settings it is written as `"muon 4"`.

## 3. local.yml

Machine-specific settings (git-ignored; template `configure/local.example.yml`). Any key here overrides `config.yml`.
Typical keys are `paths.data_dir` (written by `runner setup`), single input folders (`analysis_files_dir`,
`ci_scan_dir`, `root_dir`), `safety.protected_dirs`, a `datasets:` section that points single runs at other ROOT
files, and `charge_injection.scan_files`. The environment variable `LDMX_DATA_DIR` overrides `paths.data_dir` of both
files. `env/docker.yml` is the `local.yml` of the Docker image (`data_dir: /data`).

## 4. channel_map.csv

Electronics channel -> detector channel, 384 rows. Columns: `DPM`, `LINK`, `CHAN` (electronics address), `PLANE`
(layer), `STRIP`, `END`. Read by ci (`paths.channel_map`).
