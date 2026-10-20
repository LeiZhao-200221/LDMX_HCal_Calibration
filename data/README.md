# data/

Variables of the output files in `data/` (normal run, the most complete version of each file). In this README a
variable is described once, at its first appearance; a later file lists only its new variables, or a name it uses with
another meaning. A variable described in [calibration/README.md](../calibration/README.md) or
[results/README.md](../results/README.md) is not repeated: the line "Described in ..." under a file names it.
`_end0` / `_end1`: the two ends of a bar.

## Folders

Nothing here is committed to git except this README and the `.gitkeep` files of the input folders. The inputs may sit
in another folder: `paths.data_dir` in `configure/local.yml` (written by `.\runner.ps1 setup <folder>`;
`LDMX_DATA_DIR` overrides it). The runs and their ROOT files are listed in `configure/datasets.yml`. A run is the
folder `<run>` = `<Particle>_<E>GeV[_run<N>]` (e.g. `Muon_4GeV`), and every product is made per input csv, in a
sub-folder per run; nothing is merged.

- inputs
  - `root/`: decoded HGCROC ntuples (tree `ntuplizehgcroc/hgcroc`, e.g. `fixed_Muon_4GeV.root`) and MC files, read by
    make-csv
  - `analysis_files/<run>/`: the beam csv files of a run, one per ROOT file (section 1), written by make-csv and read
    by every later stage; `analysis_files/` itself holds the csv of `make-csv -input`, which the later stages do not
    read
  - `charge_injection_scans/`: the charge-injection scans of 2022-04-24 (provided by Erik Wallin), read by ci
- products (a normal run writes them here directly, no `fullset/` folder; each csv has a `.provenance.json` next to
  it, unless `-no-json`)
  - `clean_dataset/<run>/`: the clean datasets (clean, section 2)
  - `charge_injection/<run>/`: the event energies of ci (section 3)
  - `test/`: the same for `-test` runs (`test/<stage>/<run>/`), and `test/analysis_files/<run>/`: the beam csv of
    `make-csv -test`

Intermediates are kept in `cache/`, not here: among them the Arbitrary Energy of adcpeak per event and per bar
(`cache/tot_adc_peak/<run>/<csv stem>_events.csv`, `_bars.csv`) and the per-bar values of ci
(`cache/charge_injection/events/<run>/`). Their variables: `.\runner.ps1 adcpeak -h`, `.\runner.ps1 ci -h`.

Every provenance json says where its csv comes from (inputs, settings, code) and how it is built (its columns).

## 1. data/analysis_files/\<run\>/\<ROOT name\>.csv (make-csv, e.g. Muon_4GeV/fixed_Muon_4GeV.csv; with -waveform)

One row per event and bar (pf_event, layer, strip).

| variable | meaning |
|---|---|
| pf_event | event number in the csv (an event of a run is identified by its csv file and pf_event) |
| layer | layer of the bar (1-19) |
| strip | bar in the layer (from 0) |
| pf_spill | spill number |
| pf_ticks | time stamp of the event (clock ticks) |
| tot_end0, tot_end1 | final TOT used: the single non-zero TOT sample (2048 -> 0; 0 if none or several) |
| toa_end0, toa_end1 | final TOA used [ns]: from the single non-zero TOA sample, with the TOA table of the bar (0 if none or several) |
| adc_sum_end0, adc_sum_end1 | sum of the ADC of all time samples (pedestal included) |
| adc_mean_end0, adc_mean_end1 | adc_sum / number of time samples |
| adc_max_end0, adc_max_end1 | largest ADC of the time samples |
| adc_0_end0 .. adc_7_end0, adc_0_end1 .. adc_7_end1 | ADC of each time sample (with -waveform) |
| raw_tot_sum_end0, raw_tot_sum_end1 | sum of the raw TOT words of all time samples (before the single-hit and 2048 rules) |
| raw_toa_sum_end0, raw_toa_sum_end1 | sum of the raw TOA words of all time samples |

## 2. data/clean_dataset/\<run\>/\<csv stem\>_clean.csv (clean)

One row per clean bar (a kept bar of a kept event): the rows of one beam csv, copied as they are, in file order. The
columns are those of the beam csv (section 1), in the same order and with the same meaning; there is no new variable.

## 3. data/charge_injection/\<run\>/\<csv stem\>_event_energies.csv (ci)

One row per reconstructed event (at least one reconstructed bar) of one beam csv. The scales: adc_only (no TOT),
PureCI (charge injection with s = 1) and s (charge injection x the TOT scale s per channel; s parameter, Lei).

| variable | meaning |
|---|---|
| event_source | `<run>/<csv name>` of the event; with pf_event it identifies the event |
| energy_adc_only_mip | event energy on the adc_only scale = sum of its bar energies [MIP] |
| energy_pureci_mip | event energy on the PureCI scale [MIP] |
| energy_s_mip | event energy on the s scale [MIP] (with several training samples: the sample that borrows least) |
| energy_s_\<sample\>_mip | with several training samples (`-scale-particle pion muon`): event energy on the scale of that sample (s_pion, s_muon, s_electron, s_mix) [MIP] |
| n_bars | reconstructed bars of the event |
| bad_strip_fraction | bad bars / reconstructed bars, with the bad-bar flags of the selection scale |
| selected | the event passes the quality cut bad_strip_fraction <= quality_threshold (always true without the cut) |
