# Run logs

Code: `src/ldmx_hcal/common/run_log.py`

Every stage run and every runner command started from the command line writes a log. This covers `.\runner.ps1`,
`runner.cmd`, `runner.sh` and `python src/runner.py`. The log has three parts: a summary in 8 sections, the full console
transcript and one line in an index. The logger only reads: it calls each stage's own settings, path and `io_plan()`
functions, and it reads the products and provenance records written in this run. It never changes a run.

## 1. Files

```
logs/<module>/<YYYYmmdd-HHMMSS>_<stage>[_test][_dryrun][_<variant>].log   summary (8 sections, below)
logs/<module>/<same stem>.console.txt                                     everything the run printed (stdout + stderr, unchanged)
logs/<module>/index.tsv                                                   one line per run: start, stage, command, mode, status, duration_s, log
```

| `<module>` folder | written by |
|---|---|
| `make_analysis_files/` | make-csv |
| `calibration_parameters/` | calib |
| `clean_dataset/` | clean |
| `tot_adc_peak/` | adcpeak |
| `charge_injection/` | ci |
| `multiscale_plots/` | plots |
| `runner/` | `all`, `check`, `setup`, and a failed command whose stage cannot be identified |

- `_test`: a `-test` run, which reads the first 5000 events by default. It goes in the same folder, and the index `mode`
  column says `test (first 5000 events)`.
- `_dryrun`: a `-dry-run` run, which only prints the plan. `check` never gets this tag because it is always a plan.
- `_<variant>`: the product suffix of the run: ci / adcpeak: the variant tag, e.g. `s_muon` or `logdemo`; plots: the
  energy scale + the producer variant; calib: the `-only` units joined by `-` when they differ from the default
  (e.g. `pedestal_csv-pedestal_plot`); clean: the standard / threshold flags.
- A second run in the same second gets `-2` added to its name.
- `all`: each stage writes its own log in its module folder. `logs/runner/` also gets an `all` log that lists each
  stage's status, duration, number of output files, final results, log path and one line of its fit results.
- A command that fails before its stage starts (for example a mistyped option, `ci -bogus`) is logged in that stage's
  folder (`logs/charge_injection/`). If no stage can be identified, it goes to `logs/runner/`.
- The `.log` file is written when the run starts, with `[1]` showing `exit status: RUNNING`, and it is rewritten when
  the run ends. `index.tsv` gets its line only when the run ends. A run that was killed therefore stays `RUNNING` and
  has no index line; its `.console.txt` holds everything printed up to that moment.

## 2. Switches, location, protection

- Logs are written only from the command line: `runner.py`, run as a script, goes through `run_log.guarded`. Stage
  functions called from Python and the test suite write no logs.
- `LDMX_RUN_LOG=0` switches every log off. `LDMX_RUN_LOG=1` also logs outside the command line.
- Logs root, first match wins: (1) the environment variable `LDMX_LOGS_DIR`; (2) `paths.logs_dir` (optional; put it in
  `configure/local.yml`); (3) `<parent of paths.results_dir>/logs`, which is `data_analysis/logs` with the shipped
  `configure/config.yml`.
- No log is written if the logs folder is inside `safety.protected_dirs`.
- The logger never changes a run. Any failure inside it is caught and reported as one line on stderr:
  `[run-log] WARNING: ...`. The stage's own exception, including `SystemExit`, is re-raised unchanged after the log is
  written, so exit codes stay the same. Writing a log takes about 0.01 - 0.05 s, and the last line of the log gives the
  actual time.

## 3. The 8 sections (every module)

| section | content |
|---|---|
| `[1] Run` | <ul><li>command as typed, stage (module; run directly or inside `all`)</li><li>start / end / duration, exit status, mode</li><li>config files (`config.yml`, `local.yml`, `datasets.yml`, `LDMX_DATA_DIR`), `data_dir`</li><li>python and package versions, python executable</li><li>code fingerprint: the first 12 hex digits of the sha256 of the stage module, plus `energy_response.py` for ci / adcpeak, `common/*.py` and `runner.py`</li><li>working directory, console transcript path</li></ul> |
| `[2] Features` | <ul><li>Every option of the stage, one line each: `<feature>: <value>  [default\|yml\|command line]`.</li><li>An option that is not active shows `NA (what the program does instead)`.</li><li>The common options come first: runs, output units (`-only`), test, dry run, clean input, from saved, figures, json records (`-no-json`), final results (`--output`), workers.</li><li>Then the module's own table, and last `yml settings != module defaults`.</li></ul> |
| `[3] Inputs` | <ul><li>Every input and cache of the stage's `io_plan()` with its status: `ok` / `MISSING` / `PENDING` / `STALE`.</li><li>Size, plus rows / events when its provenance record gives them.</li><li>A run is listed with its run folder; every csv of the folder is one input.</li><li>Small csv files (20 MB or less) are counted; large csv files are never re-read.</li></ul> |
| `[4] Data statistics` | <ul><li>Per run (and per csv of the run where the stage writes one product per csv): events in the file, reconstructed and selected, retention, bars.</li><li>Per-channel stages: channels OK, and only the failed channels (`L<layer>/S<strip>/E<end>` + status + reason), never all 384. At most 96 are listed per status, then `... and N more`.</li></ul> |
| `[5] Fit results` | <ul><li>Per run and path: fit type, fit range, mu +- error, sigma +- error, sigma/mu, chi2/ndf, status (`FAILED (reason)` when the fit failed).</li><li>Linearity / resolution parameters, or `no fit (points only)`.</li><li>calib pedestal / MPV summary, adcpeak TOT-ADC lines, ci CI map and s training (with several samples: which sample became s and the borrowing counts), the e/h points.</li></ul> |
| `[6] Outputs` | <ul><li>First the final results of the run: kind, key, destination, `saved: <rule>`, or `NOT SAVED (no --output)` / `NOT SAVED (not named in --output)`: scratch copy, size, csv rows, pdf pages, what a figure shows, and its provenance record.</li><li>Then every other file written in this run (the intermediates).</li><li>Dry run: the planned outputs.</li></ul> |
| `[7] Warnings` | <ul><li>First the `input: ...` notes (section 5).</li><li>Then the console lines that contain `WARNING`, `note:`, `refused`, `stale` or `ignored` (make-csv also `not overwritten` and `skipping`).</li><li>Then the WARNINGs of the output plan (an `--output` name that matches no result) that the console does not hold.</li></ul> |
| `[8] Bugs / errors` | <ul><li>`none`, `USER/INPUT ERROR`, or `BUG` with its location (section 4).</li><li>On a failure, followed by the input runs of the stage.</li></ul> |

Source tags in `[2]`: `[command line]`: given in this command; `[yml]`: the `config.yml` / `local.yml` value differs
from the module DEFAULTS (path defaults are compared with `${...}` expanded); `[default]`: the module default.

`final results (--output)` says where the final results go: `NA (not saved: scratch copies in cache/<stage>/unsaved)`
(full set without `--output`), `default names (--output)`, `names: a.csv, b.pdf` (the results not named go to the
scratch copy), or `test names (results/test/test*.csv/pdf)` (`-test`). `output units (-only)` lists the units of the
run, `[default]` or `[command line]`. `json records (-no-json switches them off)` says whether the record json next to
the products were written; with `-no-json` the `[4]` / `[5]` statistics come from the files and the cache records, and
a product without a record is marked `(no provenance record: -no-json)`.

The statistics come only from the files written in this run, meaning files modified after the start, and from their
provenance records. A file that was not rewritten is reported as `not rewritten in this run, last written <time>`. Final
results are read where this run's output plan put them: the saved path in `results/fullset/<stage>/` (`-test`:
`results/test/<stage>/`), a user or test name, or the scratch copy in `cache/<stage>/unsaved/`. They are never read from
a default path, which may hold another run's file. `stage_log` calls `outputs.reset()` when a stage starts, so a log
reads only that stage's plans.

## 4. Exit status and errors

| `exit status` in `[1]` | meaning |
|---|---|
| `OK` | the run ended normally |
| `FAILED (BUG: <exception type>)` | a Python exception, i.e. a program error. `[8]` gives its location and the full traceback |
| `FAILED (USER/INPUT ERROR, exit code N)` | the run stopped on purpose with a message: a `SystemExit` from the code (a wrong option value, an input without the needed columns, a run reference that names no run), or an argparse error such as a mistyped option (exit code 2). This is not a bug |
| `INTERRUPTED (Ctrl+C)` | stopped by the user |
| `RUNNING (...)` | still running, or killed (power loss, closed window, `Stop-Process`). See `.console.txt` |
| `FAILED (exit code N: ...)` | `check` only: the check finished with a non-zero exit code (1 = a stage cannot be loaded, 2 = an external input is missing, 3 = products of earlier stages are missing or stale with `-strict`) |

`[8]` of a BUG:

```
  BUG: KeyError: 'linearity_mode'
  BUG LOCATION: src/ldmx_hcal/energy/charge_injection.py:3964 in run: <that line of code>
  RAISED AT: <library file>:<line> in <function>: <code>     (only when the innermost frame is in a library)
  BUG LOCATION (worker process): ...                          (an error inside a -workers process)
  input runs of this stage (as given; compare them with the error above):
    runs                        : electron 0.3, electron 0.4, electron 0.5, electron 1, electron 4  [default]
    training runs (pion sample) : pion 4 (Pion_4GeV/)  [default]
  full traceback:
    Traceback (most recent call last): ...
```

- `BUG LOCATION`: the innermost traceback frame inside `src/` (this repository): file, line, function and code. This is
  the place to fix.
- `RAISED AT`: the innermost frame overall. It is printed only when that frame is outside `src/`, for example inside
  pandas or scipy.
- `USER/INPUT ERROR` has no traceback. It gives `type`, `message` (all lines, as printed; for argparse, its `error:`
  line) and `raised in:` (the function that decided to stop).
- Both kinds are followed by `input runs of this stage`. These lines do not change the classification: a stop on purpose
  is still a USER/INPUT ERROR, and an exception is still a BUG. They show when an input does not suit the analysis, for
  example calib on `electron 4`.

Find failures and bugs (PowerShell, from `data_analysis`):

```powershell
Get-Content logs\charge_injection\index.tsv | Select-Object -Last 5      # last 5 ci runs
Select-String -Path logs\*\index.tsv -Pattern "FAILED"                     # every failed run
Select-String -Path logs\*\*.log -Pattern "BUG LOCATION"                   # every bug location
```

## 5. Runs in the logs and the input notes

- A run is written as its reference, e.g. `muon 4`, `electron 0.3`, `pion 0.5`, `electron 0.3 run 304`, `mc muon 4`. It
  is never abbreviated.
- The run key (`muon_4GeV`) appears only in file names, console progress lines and provenance; the run folder
  (`Muon_4GeV/`) names the per-run sub-folders. The log converts keys to references. A key that names no run of
  `configure/datasets.yml` is shown as it is.
- Any run of `configure/datasets.yml` is accepted wherever a stage takes a run. The log is where a wrong choice shows:
  - `[2]`: every run-choosing input gets one line with its runs, its source and `(default ...)` when it differs. The
    inputs are the stage's runs (run flags, else `DEFAULT_DATASETS`), calib's `source run` (the run flag, else
    `calibration.source_dataset`, default `muon 4`), adcpeak's `fit run` (`-fit-dataset`, else
    `tot_adc_peak.fit_dataset`, default `pion 3`) and ci's `s training runs`
    (`charge_injection.scale_training_datasets.<sample>`, one line per sample). A single source run, fit run or training
    run also names its run folder. An input that names no run shows `NOT RESOLVED: <as given>  [source]: <reason>`.
  - `[7]`: an `input: ...` note for every role input (source run, fit run, training runs) that differs from its default,
    e.g.
    `input: calib source run = electron 4  [command line] (default muon 4); the stage ran on what was given, check its results`.
    This is information, never a refusal. An input the stage cannot use (calib given several runs) ends with
    `the stage stops on it: <reason>` instead.
  - `[8]`: the input runs and their roles, printed next to the error. The `all` log adds `input runs: ...` under each
    failed stage.

## 6. Runner logs (logs/runner/)

| section | `all` | `check` | `setup` | failed before the stage started |
|---|---|---|---|---|
| [2] | `-stages`, `-with-make-csv`, runs (reference + flags), the common options | `-strict` + the common options | data folder, `-protect` | NA |
| [3] | see the stage logs | every stage's plan (full list in the console) | data folder | NA |
| [4] | one line per stage: status, duration, outputs, log path | item counts per stage, every item that is not ok, exit code and its meaning | what setup printed | NA |
| [5] | the first fit line of every stage (with the number of FAILED fits) | none | none | NA |
| [6] | files and final results per stage | none (check writes nothing) | `configure/local.yml` if it was changed | none |
| [7] | console warnings | console warnings | console warnings | console warnings |
| [8] | failed stage, BUG LOCATION, its input runs, its log path, then the exception | on a non-zero exit code: its meaning and the UNAVAILABLE / MISSING items | as for a stage | the error (argparse, configuration) |

## 7. Known limits

- What `-workers N` processes print themselves does not reach `.console.txt`. Everything the main process prints does.
- `[6]` finds files by modification time. A second command running at the same time can add its files.
- `command (as typed)` shows the arguments Python received (`src/runner.py ...`). It does not show which wrapper started
  the command (`runner.ps1` / `runner.cmd`).
- If a stage fails before it makes its output plan, `[6]` says
  `final results: none (no output plan was made in this run)`. If a planned result was not written, it says
  `NOT WRITTEN in this run (missing)`.

## 8. For developers

The runner uses these entry points: `run_log.guarded(main)` in `__main__`; `stage_log(name, args, ctx, direct)` around
every stage; `command_log(name, args, ctx)` around `all` / `check` / `setup`, used as
`with ... as rl: return rl.result(code)`; `final_result_note(item, ctx)` for the final-result note of `print_plan`.

What a module puts in its log is defined in one place: `FEATURES` (one table per module, for `[2]`), `REPORTS` (one
function per module, returning the lines of `[4]` and `[5]`) and `FIGURES` (what each pdf shows, for `[6]`). If a stage
changes its column names, its log shows `NA` or `? (cannot be read ...)`, and the run does not fail.
