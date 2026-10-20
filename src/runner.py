#!/usr/bin/env python
"""Runner of the LDMX 2022 HCal test-beam analysis: one command per stage, run from the data_analysis folder.

    .\\runner.ps1 <stage> [--electron E] [--muon E] [--pion E] [options]    (runner.cmd, runner.sh: same)

Stages, in chain order: make-csv, calib, clean, adcpeak, ci, plots; all (the chain), check (every input and
output), setup (point the analysis at the data folder).  Help of one stage: .\\runner.ps1 <stage> -h.
READMEs (in data_analysis/): README.md (overview), src/README.md (every stage, the runner, common code),
configure/README.md, logs/README.md; output variables: data/, calibration/, results/README.md.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # .../data_analysis/src
for _stream in (sys.stdout, sys.stderr):  # non-ASCII paths must never crash a cp1252 console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from ldmx_hcal.common.config import (DATA_DIR_ENV, DEFAULT_CONFIG, MODE_FULLSET, MODE_TEST,  # noqa: E402
                                     RESULTS_MODE_DIRS, SELECTABLE_PARTICLES, RunContext, dataset_registry,
                                     load_config, mode_parts, resolve_datasets, resolve_runs, run_flags, run_ref,
                                     select_by_particle)
from ldmx_hcal.common.io_utils import (canonical_path, log, product_set_valid, subset_status,  # noqa: E402
                                       temp_registry, validate_settings)
from ldmx_hcal.common.fitting import normalize_range_args  # noqa: E402
from ldmx_hcal.common.outputs import add_output_argument, selected_units  # noqa: E402
from ldmx_hcal.common import run_log  # noqa: E402  (logs/<module>/: summary + console transcript of every run)

# stage name -> module (order = pipeline order used by "all")
STAGES = {
    "make-csv": "ldmx_hcal.conversion.make_analysis_files",
    "calib": "ldmx_hcal.calibration.calibration_parameters",
    "clean": "ldmx_hcal.selection.clean_dataset",
    "adcpeak": "ldmx_hcal.energy.tot_adc_peak",
    "ci": "ldmx_hcal.energy.charge_injection",
    "plots": "ldmx_hcal.visualization.multiscale_plots",
}
ALL_DEFAULT_STAGES = ["calib", "clean", "adcpeak", "ci", "plots"]  # make-csv only with --with-make-csv
BROKEN_STAGES: dict[str, str] = {}  # stage -> import error (reported instead of crashing the whole CLI)

# --clean-input: stages that never read the clean dataset, and why
CLEAN_INPUT_REFUSED = {
    "make-csv": "it reads ROOT files, not the beam csv",
    "clean": "it makes the clean dataset from the beam csv",
}
# stages that read it only when their module declares SUPPORTS_CLEAN_INPUT = True (the others need no
# declaration: they read the beam csv through RunContext.input_csv), and why they refuse until then
CLEAN_INPUT_OPT_IN = {  # calib opts in (SUPPORTS_CLEAN_INPUT = True): pedestal from RunContext.beam_csv, MPV clean
    "calib": "the pedestal needs the no-signal rows that the clean selection removes, and this calib module "
             "does not declare SUPPORTS_CLEAN_INPUT (pedestal from the beam csv, MPV from the clean dataset)",
}
# stages that run on the beam csv inside "all" / "check" even with --clean-input (clean needs a calibration)
BEAM_CSV_IN_CHAIN = ("make-csv", "calib", "clean")
# --from-saved: only stages whose module declares SUPPORTS_FROM_SAVED = True
FROM_SAVED_REFUSED = {"make-csv": "it converts ROOT files and stores no intermediate to restart from"}

# exit codes of "check"
EXIT_OK, EXIT_UNAVAILABLE, EXIT_EXTERNAL_MISSING, EXIT_CHAIN_PENDING = 0, 1, 2, 3

COMMON_KEYS = ("datasets", "config", "test", "test_events", "no_plots", "workers", "dry_run",
               "no_json", "clean_input", "from_saved", "outputs", *(f"select_{p}" for p in SELECTABLE_PARTICLES))


# common options of every stage: run flags, -config, -test, -workers, -dry-run, -no-json, -output (-only: per stage)
def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_names", nargs="*", help=argparse.SUPPRESS)  # positional run names: refused in main
    p.set_defaults(datasets=None)  # internal: runs added by 'all -clean-input' (with_chain_runs)
    for particle in SELECTABLE_PARTICLES:  # --electron 0.3 1 / --muon / --pion 4 (runs of configure/datasets.yml)
        cap = particle.capitalize()
        p.add_argument(f"--{particle}", f"-{particle}", f"--{particle}s", f"-{particle}s", f"--{cap}", f"-{cap}",
                       dest=f"select_{particle}", nargs="*", action="append", default=None, metavar="E",
                       help=f"{particle} runs of configure/datasets.yml: --{particle} 0.3 4 = the 0.3 and 4 GeV "
                            f"runs; --{particle} alone = every {particle} run (an energy belongs to the particle "
                            f"flag in front of it)")
    p.add_argument("--config", "-config", default=None, help="yml config (default: configure/config.yml)")
    p.add_argument("--test", "-test", action="store_true",
                   help="test mode: only the first N events (default 5000); products go to data/test, "
                        "calibration/test, results/test and cache/test")
    p.add_argument("--test-events", "-test-events", type=int, default=None,
                   help="number of events in test mode (default from config run.test_events = 5000)")
    p.add_argument("--no-plots", "-no-plots", "--noplots", "-noplots", dest="no_plots", action="store_true",
                   help="skip every figure (csv outputs only)")
    p.add_argument("--workers", "-workers", type=int, default=None, help="parallel worker processes")
    p.add_argument("--dry-run", "-dry-run", action="store_true",
                   help="print the input/output plan and whether inputs exist; run nothing")
    p.add_argument("--no-json", "-no-json", dest="no_json", action="store_true",
                   help="write no record json (provenance, results record) next to the products in data/, "
                        "calibration/ and results/; the records in cache/ stay. Checks that need a missing record "
                        "do not verify")
    p.add_argument("--clean-input", "-clean-input", dest="clean_input", action="store_true",
                   help="read the clean dataset of each csv of a run (data/[test/]clean_dataset/<run>/"
                        "<csv stem>_clean.csv, written by 'clean' in the same mode) instead of the beam csv; refused "
                        "by make-csv and clean (calib: MPV from the clean dataset, pedestal from the beam csv)")
    add_output_argument(p)  # --output NAME ...: final results by result name (common/outputs.py)
    p.add_argument("--from-saved", "-from-saved", dest="from_saved", action="store_true",
                   help="restart from the intermediates this stage stored in cache/[test/]<stage>/ and skip its "
                        "expensive part; stops with the reason and the rebuild command when they do not match "
                        "the current inputs / settings")


# command-line parser: one sub-command per stage + all, check, setup
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for name, modname in STAGES.items():
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # a broken/missing stage must not take the other stages down
            BROKEN_STAGES[name] = f"{type(exc).__name__}: {exc}"
            sp = sub.add_parser(name, help=f"(unavailable: {BROKEN_STAGES[name]})", allow_abbrev=False)
            add_common(sp)
            continue
        sp = sub.add_parser(name, help=getattr(mod, "HELP", ""), allow_abbrev=False,
                            description=mod.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        add_common(sp)
        try:
            mod.add_arguments(sp)
        except argparse.ArgumentError as exc:  # e.g. the stage defines a common option (-from-saved) again
            BROKEN_STAGES[name] = f"its options clash with the common runner options: {exc}"
    sp = sub.add_parser("all", help="run calib -> clean -> adcpeak -> ci -> plots with default options",
                        allow_abbrev=False)
    add_common(sp)
    sp.add_argument("--with-make-csv", "-with-make-csv", action="store_true",
                    help="start from ROOT files (run make-csv first)")
    sp.add_argument("--stages", "-stages", nargs="+", choices=list(STAGES), default=None,
                    help="explicit subset of stages, in pipeline order")
    sp = sub.add_parser("check", allow_abbrev=False,
                        help="chain check: list every stage's inputs/outputs; exit code 0 = all external "
                             "inputs present, 2 = an external input is missing, 1 = a stage cannot be loaded, "
                             "3 (with --strict) = products of earlier stages missing or stale")
    add_common(sp)
    sp.add_argument("--strict", "-strict", action="store_true",
                    help="also fail (exit 3) when inputs produced by earlier stages are missing or stale")
    sp = sub.add_parser("setup", allow_abbrev=False,
                        help="point the analysis at your data folder: writes configure/local.yml (paths.data_dir) "
                             "and lists what the folder holds")
    sp.add_argument("data_dir", nargs="?", default=None, metavar="DATA_DIR",
                    help="folder with root/, analysis_files/ (one sub-folder per run) and charge_injection_scans/; "
                         "omitted: show the current setting")
    sp.add_argument("--protect", "-protect", nargs="+", default=None, metavar="DIR",
                    help="folders no stage may ever write into (e.g. the original data elsewhere), added to "
                         "safety.protected_dirs of local.yml; refused when it would cover a folder the stages write")
    return parser


SETUP_SUBFOLDERS = (("root", "decoded ntuples (*.root) for make-csv", "*.root*"),
                    ("analysis_files", "beam csv: one folder per run of configure/datasets.yml", "*/*.csv"),
                    ("charge_injection_scans", "charge-injection scans scan_DPM*.csv (stage ci)", "*.csv"))


# launcher name for messages: .\runner.ps1 (Windows) or ./runner.sh
def _runner_cmd() -> str:
    return ".\\runner.ps1" if os.name == "nt" else "./runner.sh"


# every folder a stage writes into: product trees, their test folders, the make-csv run folders (analysis_files_dir)
def _write_tree_paths(cfg: dict) -> list[Path]:
    paths = cfg.get("paths", {})
    trees = [Path(paths[k]) for k in ("generated_data_dir", "calibration_dir", "results_dir", "cache_dir")
             if paths.get(k)]
    out = trees + ([Path(paths["analysis_files_dir"])] if paths.get("analysis_files_dir") else [])
    if paths.get("results_dir"):
        out.append(Path(paths["results_dir"]) / RESULTS_MODE_DIRS[MODE_FULLSET])
    return out + [t.joinpath(*mode_parts(MODE_TEST)) for t in trees]


# setup: write paths.data_dir (+ -protect folders) into configure/local.yml, list the data folder
def run_setup(args) -> int:
    import yaml

    local = DEFAULT_CONFIG.with_name("local.yml")
    if args.data_dir or args.protect:
        current = {}
        if local.exists():
            with open(local, "r", encoding="utf-8") as fh:
                current = yaml.safe_load(fh) or {}
            if not isinstance(current, dict):
                raise SystemExit(f"setup: {local} is not a yml mapping; fix or remove it first")
        if args.data_dir:
            folder = Path(args.data_dir).expanduser().resolve()
            if not folder.is_dir():
                raise SystemExit(f"setup: {folder} is not a folder")
            current.setdefault("paths", {})["data_dir"] = folder.as_posix()
        if args.protect:
            protect = [Path(d).expanduser().resolve() for d in args.protect]
            saved_env = os.environ.get(DATA_DIR_ENV)
            if args.data_dir:  # where the stages would write with the new data_dir
                os.environ[DATA_DIR_ENV] = current["paths"]["data_dir"]
            try:
                trial = load_config()
            finally:
                if saved_env is None:
                    os.environ.pop(DATA_DIR_ENV, None)
                else:
                    os.environ[DATA_DIR_ENV] = saved_env
            clash = [(d, w) for d in protect for w in _write_tree_paths(trial) if w == d or d in w.parents]
            if clash:
                d, w = clash[0]
                raise SystemExit(f"setup: protecting {d} would forbid the stages to write {w}; protect the original "
                                 f"data elsewhere (e.g. an old analysis folder), not the folders this analysis writes")
            safety = current.setdefault("safety", {}) or {}
            current["safety"] = safety
            old = [str(x) for x in (safety.get("protected_dirs") or [])]
            safety["protected_dirs"] = old + [d.as_posix() for d in protect if d.as_posix() not in old]
        header = ("# Machine-specific settings (git-ignored).  Updated by 'runner setup' (it keeps every other key).\n"
                  "# Every input path follows paths.data_dir (see configure/config.yml paths:).\n")
        local.write_text(header + yaml.safe_dump(current, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"updated {local}")
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    local_cfg = {}
    if local.exists():
        with open(local, "r", encoding="utf-8") as fh:
            local_cfg = yaml.safe_load(fh) or {}
    source = (f"environment variable {DATA_DIR_ENV}" if os.environ.get(DATA_DIR_ENV)
              else ("configure/local.yml" if ((local_cfg.get("paths") or {}).get("data_dir")) else
                    "configure/config.yml (default)"))
    print(f"data_dir = {data_dir}   (from {source})")
    for sub, what, pattern in SETUP_SUBFOLDERS:
        folder = data_dir / sub
        n = len(list(folder.glob(pattern))) if folder.is_dir() else 0
        print(f"  {sub + '/':26s} {('%d file(s)' % n) if folder.is_dir() else 'MISSING':12s} {what}")
    print("runs of configure/datasets.yml (run folder: beam csv files):")
    ctx = RunContext(cfg=cfg)
    missing = 0
    for ds in dataset_registry(cfg).values():
        csvs = ctx.run_csvs(ds)
        n_found = sum(p.exists() for p in csvs)
        ok = bool(csvs) and n_found == len(csvs)
        missing += not ok
        print(f"  {'ok     ' if ok else 'MISSING'} {run_flags(ds)}  [{ds.key}]  {ctx.run_dir(ds)}  "
              f"({n_found} of {len(csvs)} csv)")
    print(f"{missing} run(s) without their csv files." if missing else "every run has its csv files.")
    print(f"next: {_runner_cmd()} check   (every input and output of every stage)")
    return 0


RUN_OPTION_RULES = {"test_events": "int>0", "workers": "int>0", "chunksize": "int>0"}


# run context of a command: config + common options (run options checked first)
def make_context(args) -> RunContext:
    cfg = load_config(args.config)
    run_cfg = cfg.get("run", {}) or {}
    validate_settings(run_cfg, RUN_OPTION_RULES, source="configure/config.yml run")
    validate_settings({k: v for k, v in (("test_events", args.test_events), ("workers", args.workers))
                       if v is not None}, RUN_OPTION_RULES, source="command line")
    return RunContext(
        cfg=cfg,
        test=bool(args.test),
        n_test_events=int(args.test_events or run_cfg.get("test_events", 5000)),
        make_plots=not args.no_plots,
        workers=int(args.workers or run_cfg.get("workers", 1)),
        dry_run=bool(getattr(args, "dry_run", False)),
        clean_input=bool(getattr(args, "clean_input", False)),
        outputs=getattr(args, "outputs", None),
        write_json=not getattr(args, "no_json", False),
    )


# runs of a stage: the run flags, else the stage's DEFAULT_DATASETS
def datasets_for(ctx: RunContext, mod, args):
    tokens = list(getattr(args, "datasets", None) or [])
    selections = {p: getattr(args, f"select_{p}", None) for p in SELECTABLE_PARTICLES}
    tokens += [k for k in select_by_particle(ctx.cfg, selections) if k not in tokens]
    return resolve_datasets(ctx.cfg, tokens, default=getattr(mod, "DEFAULT_DATASETS", None),
                            what=None if tokens else f"DEFAULT_DATASETS of {mod.__name__}")


# extra runs 'all' reads: the adcpeak fit run and the ci s training runs (s parameter, Lei)
def chain_runs(ctx: RunContext, stages) -> list[str]:
    runs = []
    if "adcpeak" in stages:
        fit = (ctx.cfg.get("tot_adc_peak") or {}).get("fit_dataset") or "pion 3"
        runs += resolve_runs(ctx.cfg, fit, "tot_adc_peak.fit_dataset")
    if "ci" in stages:
        ci_cfg = ctx.cfg.get("charge_injection") or {}
        samples = ci_cfg.get("scale_particle") or "pion"
        for sample in ([samples] if isinstance(samples, str) else list(samples)):  # one or several samples
            train = (ci_cfg.get("scale_training_datasets") or {}).get(sample) or []
            runs += resolve_runs(ctx.cfg, train, f"charge_injection.scale_training_datasets.{sample}")
    keys: list[str] = []
    for ds in runs:
        if ds.key not in keys:
            keys.append(ds.key)
    return keys


# all -clean-input with chosen runs: clean also the runs adcpeak / ci read besides them
def with_chain_runs(ctx: RunContext, sargs: argparse.Namespace, stages) -> argparse.Namespace:
    chosen = list(sargs.datasets or [])
    if not chosen and not any(getattr(sargs, f"select_{p}", None) for p in SELECTABLE_PARTICLES):
        return sargs
    extra = [k for k in chain_runs(ctx, stages) if k not in chosen]
    if extra:
        reg = dataset_registry(ctx.cfg)
        log(f"all -clean-input: clean also {', '.join(run_ref(reg[k]) for k in extra)} "
            f"(read by {' and '.join(s for s in ('adcpeak', 'ci') if s in stages)})")
        sargs.datasets = chosen + extra
    return sargs


# a stage's own defaults, overlaid by the common options of the command
def stage_defaults(parser_for_stage: argparse.ArgumentParser, args) -> argparse.Namespace:
    ns = parser_for_stage.parse_args([])
    for key in COMMON_KEYS:
        setattr(ns, key, getattr(args, key, getattr(ns, key, None)))
    return ns


# stage options inside all / check: calib keeps its source run (run flags = analysed runs)
def chain_args(name: str, parser_for_stage: argparse.ArgumentParser, args) -> argparse.Namespace:
    ns = stage_defaults(parser_for_stage, args)
    if name == "calib":
        for p in SELECTABLE_PARTICLES:
            setattr(ns, f"select_{p}", None)
    return ns


# module flag of a stage (SUPPORTS_FROM_SAVED, SUPPORTS_CLEAN_INPUT), default when not set
def _declared(mod, attr: str, default: bool) -> bool:
    value = getattr(mod, attr, None)
    return default if value is None else bool(value)


# why a stage cannot read the clean dataset (None = it can)
def clean_input_refusal(name: str, mod) -> str | None:
    if name in CLEAN_INPUT_REFUSED:
        return CLEAN_INPUT_REFUSED[name]
    if not _declared(mod, "SUPPORTS_CLEAN_INPUT", name not in CLEAN_INPUT_OPT_IN):
        return CLEAN_INPUT_OPT_IN.get(name, "this stage does not read the beam csv through RunContext.input_csv")
    return None


# apply -clean-input to one stage: refused directly if it cannot; beam csv inside all / check
@contextmanager
def clean_input_for(name: str, ctx: RunContext, direct: bool, mod=None):
    if not ctx.clean_input:
        yield
        return
    refusal = clean_input_refusal(name, mod)
    if direct and refusal:
        raise SystemExit(f"'{name}' does not accept --clean-input: {refusal}.")
    if direct or (name not in BEAM_CSV_IN_CHAIN and refusal is None):
        yield
        return
    ctx.clean_input = False
    try:
        yield
    finally:
        ctx.clean_input = True


# apply -from-saved to one stage: refused directly without SUPPORTS_FROM_SAVED, switched off inside all / check
def from_saved_for(name: str, mod, args, direct: bool) -> str | None:
    """Returns the reason when -from-saved was switched off, else None."""
    if not getattr(args, "from_saved", False):
        return None
    if name not in FROM_SAVED_REFUSED and _declared(mod, "SUPPORTS_FROM_SAVED", False):
        return None
    why = FROM_SAVED_REFUSED.get(name, "this stage has no stored intermediates to restart from (yet)")
    if direct:
        raise SystemExit(f"'{name}' does not accept --from-saved: {why}.")
    args.from_saved = False
    return why


# ---------------------------------------------------------------------------------------------------
# plans: classification of inputs
# ---------------------------------------------------------------------------------------------------
# plan item without a kind: a missing input made by another stage of the chain?
def _is_chain_product(item) -> bool:
    producer = str(item.get("produced_by") or "")
    return "runner" in producer and "make-csv" not in producer


# give plan items without a kind the kind the chain knows for their path
class PlanKinds:

    # every beam-csv path of every csv of a run (full csv, test csv, test subset, clean dataset) -> (run, csv)
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self._csvs = {}
        for ds in dataset_registry(ctx.cfg).values():
            for csv in ctx.run_csvs(ds):
                for p in (ctx.test_subset_csv(ds, csv), ctx.test_csv(ds, csv), ctx.clean_product(ds, csv), csv):
                    self._csvs.setdefault(canonical_path(p), (ds, csv))

    # the item with its kind filled in (beam csv, else product inside a product tree)
    def __call__(self, item: dict) -> dict:
        if item.get("kind") or item.get("role", "in") != "in":
            return item
        target = canonical_path(item["path"])
        ds, csv = self._csvs.get(target, (None, None))
        if ds is not None:
            for known in (self.ctx.input_plan_item(ds, csv), self.ctx.beam_plan_item(ds, csv)):
                if canonical_path(known["path"]) == target:
                    return {**item, **{k: v for k, v in known.items() if k not in ("role", "path")}}
        if self.ctx.in_products(target, mode="any"):
            return {**item, "kind": "product"}
        return item


# (status, note) of a plan input: ok / MISSING / PENDING / STALE / AUTO
def describe_input(item) -> tuple[str, str]:
    kind = item.get("kind")
    path = Path(item["path"])
    if kind == "auto":
        source = item.get("source")
        if source is not None and not Path(source).exists():
            return "MISSING", f"built from {source}, which is absent"
        if not path.exists():
            return "AUTO", f"built from {source} by the first -test run"
        if source is None or item.get("n_events") is None:
            return "ok", ""
        ok, reason = subset_status(source, path, item["n_events"])
        return ("ok", "") if ok else ("AUTO", f"rebuilt from {source} by the next -test run: {reason}")
    if item.get("stale"):
        return "STALE", ""
    if path.exists():
        return "ok", ""
    if kind == "external":
        return "MISSING", ""
    if kind == "product":
        return "PENDING", ""
    return ("PENDING" if _is_chain_product(item) else "MISSING"), ""


# status of a plan input (describe_input without the note)
def classify_input(item) -> str:
    return describe_input(item)[0]


# print a stage plan: [in ] input status, [out] ok / new (+ final-result rule), [cache] reuse / rebuild
def print_plan(stage: str, plan, ctx: RunContext | None = None) -> tuple[int, int]:
    """Returns (missing external inputs, missing or stale chain products)."""
    external = pending = 0
    kinds = PlanKinds(ctx) if ctx is not None else (lambda it: it)
    print(f"\n=== {stage} ===")
    for item in plan:
        role = item.get("role", "in")
        path = Path(item["path"])
        if role == "in":
            item = kinds(item)
            tag, note = describe_input(item)
            external += tag == "MISSING"
            pending += tag in ("PENDING", "STALE")
            if note:
                src = f"   <- {note}"
            else:
                src = f"   <- {item['produced_by']}" if item.get("produced_by") else ""
        elif role == "cache":
            ok, reason = product_set_valid(path, item.get("expected"), item.get("members", ()))
            tag, src = ("reuse", "") if ok else ("rebuild", f"   ({reason})")
        else:
            tag = "ok " if path.exists() else "new"
            note = run_log.final_result_note(item, ctx)   # final results: where the --output rule puts them
            src = f"   ({note})" if note else ""
        print(f"  [{role:3}] {tag:7} {path}{src}")
    return external, pending


# ---------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------
# run one stage inside its run log logs/<module>/ (the stage's exception is re-raised unchanged)
def run_stage(name: str, args, ctx: RunContext, direct: bool = True) -> None:
    with run_log.stage_log(name, args, ctx, direct):
        _run_stage(name, args, ctx, direct)


# run one stage: -from-saved / -clean-input rules, its runs, then its plan (-dry-run) or run()
def _run_stage(name: str, args, ctx: RunContext, direct: bool = True) -> None:
    if name in BROKEN_STAGES:
        raise SystemExit(f"stage '{name}' cannot be loaded: {BROKEN_STAGES[name]}")
    mod = importlib.import_module(STAGES[name])
    off = from_saved_for(name, mod, args, direct)
    if off and not ctx.dry_run:
        log(f"stage '{name}' runs from its inputs (no --from-saved: {off})")
    ctx.units = selected_units(args)  # -only (the stage's default units without it)
    with clean_input_for(name, ctx, direct, mod):
        datasets = datasets_for(ctx, mod, args)
        if ctx.dry_run:
            print_plan(name, mod.io_plan(ctx, args, datasets), ctx)
            return
        flags = ((" [TEST MODE]" if ctx.test else "") + (" [CLEAN INPUT]" if ctx.clean_input else "")
                 + (" [FROM SAVED]" if getattr(args, "from_saved", False) else ""))
        log(f"stage '{name}' on {[d.key for d in datasets] or '-'}{flags}")
        with temp_registry():  # temp files of the io_utils writers are removed if the stage fails
            mod.run(ctx, args, datasets)


# check: every stage's plan; exit 0 ok, 1 stage not loadable, 2 input missing, 3 (-strict) pending
def run_check(args, ctx: RunContext, choices) -> int:
    ctx.dry_run = True
    external = pending = unavailable = 0
    for name in STAGES:
        if name in BROKEN_STAGES:
            print(f"\n=== {name} ===\n  UNAVAILABLE: {BROKEN_STAGES[name]}")
            unavailable += 1
            continue
        mod = importlib.import_module(STAGES[name])
        sargs = chain_args(name, choices[name], args)
        from_saved_for(name, mod, sargs, direct=False)
        ctx.units = selected_units(sargs)
        with clean_input_for(name, ctx, False, mod):
            e, p = print_plan(name, mod.io_plan(ctx, sargs, datasets_for(ctx, mod, sargs)), ctx)
        external += e
        pending += p
    print()
    if unavailable:
        print(f"{unavailable} stage(s) cannot be loaded.")
    if external:
        print(f"{external} external input(s) missing (data files / constants): fix configure/local.yml.")
    if pending:
        print(f"{pending} input(s) are products of earlier stages that are not made yet or not current "
              f"(PENDING / STALE).")
    if not (unavailable or external or pending):
        print("All inputs present: chain is closed.")
    if unavailable:
        return EXIT_UNAVAILABLE
    if external:
        return EXIT_EXTERNAL_MISSING
    if pending and getattr(args, "strict", False):
        return EXIT_CHAIN_PENDING
    return EXIT_OK


# stop on an unknown token that starts with a short flag (-h, -s): argparse would read -hxyz as -h (help, exit 0,
# nothing run) and -sxyz as -s plus junk
def _refuse_short_prefix(parser: argparse.ArgumentParser, argv) -> None:
    known = {s for a in parser._actions for s in a.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                known.update(s for a in sub._actions for s in a.option_strings)
    shorts = {s for s in known if len(s) == 2 and s[0] == "-" and s[1] != "-"}
    for token in argv:
        if token == "--":
            break
        if len(token) > 2 and token[:2] in shorts and token.split("=", 1)[0] not in known:
            parser.error(f"unrecognized argument: {token} (see -h for the options)")


# parse the command line and run setup, check, all or one stage
def main(argv=None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["--help" if a == "-help" else a for a in argv]  # argparse would read -help as -h + "elp"
    # --fit[5,10],[60,110] / --bin 30,40 / --linfit[0.3,1] / --resfit[..]: PowerShell splits them at the commas
    argv = normalize_range_args(argv)
    _refuse_short_prefix(parser, argv)
    args = parser.parse_args(argv)
    if getattr(args, "run_names", None):
        parser.error(f"run names ({' '.join(args.run_names)}) are not accepted: choose runs with --electron E, "
                     f"--muon E, --pion E (e.g. --muon 4; the flag alone = every run of that particle)")
    if args.stage == "setup":
        with run_log.command_log("setup", args) as rl:  # logs/runner/
            return rl.result(run_setup(args))
    ctx = make_context(args)
    sub_parsers = {a.dest: a for a in parser._actions if isinstance(a, argparse._SubParsersAction)}
    choices = sub_parsers["stage"].choices

    if args.stage == "check":
        with run_log.command_log("check", args, ctx) as rl:  # logs/runner/
            return rl.result(run_check(args, ctx, choices))

    if args.stage == "all":
        if getattr(args, "outputs", None):  # names belong to one stage's results
            parser.error("all: --output takes no names here (each stage has its own results); use --output "
                         "alone to save every stage's results under their default names, or run the stages "
                         "one by one with names")
        with run_log.command_log("all", args, ctx):  # logs/runner/ (+ one log per stage in its module folder)
            stages = args.stages or ((["make-csv"] if args.with_make_csv else []) + ALL_DEFAULT_STAGES)
            for name in STAGES:
                if name in stages:
                    sargs = chain_args(name, choices[name], args)
                    if name == "clean" and ctx.clean_input:
                        sargs = with_chain_runs(ctx, sargs, stages)
                    if name == "calib" and any(getattr(args, f"select_{p}", None) for p in SELECTABLE_PARTICLES):
                        log("all: calib uses its source run calibration.source_dataset (the run flags choose the "
                            "analysed runs); to calibrate on another run, run calib alone, e.g. "
                            ".\\runner.ps1 calib --electron 4")
                    run_stage(name, sargs, ctx, direct=False)
        return 0

    run_stage(args.stage, args, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(run_log.guarded(main))  # switches the run logs on; also logs a failure before any stage starts
