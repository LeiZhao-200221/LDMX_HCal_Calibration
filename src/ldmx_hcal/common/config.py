"""Configuration: config.yml + local.yml + datasets.yml, run references, and the run context with every product path."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# src/ldmx_hcal/common/config.py -> parents[3] = data_analysis/
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configure" / "config.yml"

MODE_FULLSET = "fullset"   # the normal mode: its products sit directly in each tree (data/<stage>/)
MODE_TEST = "test"         # -test: its products sit in the test folder of each tree (data/test/<stage>/)
TEST_DIR = "test"          # that folder
# results/ keeps one folder per mode (Lei 2026-10-06): results/fullset/<stage>/ and results/test/
RESULTS_MODE_DIRS = {MODE_FULLSET: "fullset", MODE_TEST: TEST_DIR}
# input folders that sit directly in a product tree (data/root/, calibration/reference/, ...): never a product
INPUT_DIRS = ("root", "analysis_files", "charge_injection_scans", "reference")

_VAR = re.compile(r"\$\{([^}]+)\}")


# --------------------------------------------------------------------------
# yml loading
# --------------------------------------------------------------------------
# expand ${name} references in a string (nested ones in a few passes)
def _expand_string(value: str, variables: dict) -> str:
    for _ in range(10):  # nested references resolve in a few passes
        new = _VAR.sub(lambda m: str(variables.get(m.group(1), m.group(0))), value)
        if new == value:
            break
        value = new
    return value


# expand ${name} in every string of a yml tree
def _expand_tree(node, variables: dict):
    if isinstance(node, str):
        return _expand_string(node, variables)
    if isinstance(node, list):
        return [_expand_tree(v, variables) for v in node]
    if isinstance(node, dict):
        return {k: _expand_tree(v, variables) for k, v in node.items()}
    return node


# absolute path; a relative path is relative to data_analysis/
def resolve_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# merge two yml trees, the override wins key by key
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


DATA_DIR_ENV = "LDMX_DATA_DIR"   # environment variable: overrides paths.data_dir (yml files untouched)


# read config.yml + local.yml (+ LDMX_DATA_DIR), expand the paths, add the runs of datasets.yml
def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    local = cfg_path.with_name("local.yml")
    if local.exists():
        with open(local, "r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    if os.environ.get(DATA_DIR_ENV):  # a relative value is relative to the current folder, like a typed path
        raw.setdefault("paths", {})["data_dir"] = os.path.abspath(os.path.expanduser(os.environ[DATA_DIR_ENV]))
    cfg = copy.deepcopy(raw)
    variables = {k: str(v) for k, v in raw.get("paths", {}).items()}
    for _ in range(10):
        variables = {k: _expand_string(v, variables) for k, v in variables.items()}
    cfg["paths"] = {k: resolve_path(v) for k, v in variables.items()}
    str_vars = {k: str(v) for k, v in cfg["paths"].items()}
    for section, body in raw.items():
        if section != "paths":
            cfg[section] = _expand_tree(body, str_vars)
    datasets_file = cfg["paths"].get("datasets_file")
    if datasets_file is not None:
        if not Path(datasets_file).exists():
            raise SystemExit(f"[config] paths.datasets_file {datasets_file} does not exist")
        with open(datasets_file, "r", encoding="utf-8") as fh:
            tree = _expand_tree(yaml.load(fh, Loader=_UniqueKeyLoader) or {}, str_vars)
        base = datasets_from_tree(tree, str(datasets_file))
        cfg["datasets"] = _deep_merge(base, _dataset_overrides(cfg.get("datasets") or {}, base, str(datasets_file)))
    safety = cfg.get("safety") or {}
    cfg["safety"] = {**safety, "protected_dirs": [resolve_path(d) for d in (safety.get("protected_dirs") or []) if d]}
    cfg["_config_file"] = cfg_path.resolve()
    return cfg


# --------------------------------------------------------------------------
# configure/datasets.yml: <particle>: <energy>: {root, runs, label}; run key and run folder derived
# --------------------------------------------------------------------------
# yml loader that refuses a key given twice in one block
class _UniqueKeyLoader(yaml.SafeLoader):
    pass


# build a yml mapping, stop on a repeated key
def _unique_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise SystemExit(f"[config] {node.start_mark.name}: line {key_node.start_mark.line + 1}: the key {key!r} "
                             f"appears twice in the same block (the second would silently replace the first)")
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)

DATASET_OVERRIDE_RENAMES = {"root": "root_files", "runs": "run_numbers"}  # datasets.yml names -> internal names
DATASET_ENTRY_FIELDS = {"root", "runs", "label"}
# fields a datasets: entry of config.yml / local.yml may set (particle / energy_gev / kind only for a new run)
DATASET_OVERRIDE_FIELDS = DATASET_ENTRY_FIELDS | {"particle", "energy_gev", "kind"}


# stop on fields a run entry does not have
def _unknown_fields_error(where: str, entry: dict, valid: set) -> None:
    unknown = sorted(str(k) for k in set(entry) - valid)
    if unknown:
        raise SystemExit(f"[config] {where}: unknown field(s) {unknown}; valid: {sorted(valid)}")


# datasets: section of config.yml / local.yml (keys: run reference or run key), merged over datasets.yml
def _dataset_overrides(overrides: dict, base: dict, source: str) -> dict:
    out = {}
    for name, entry in (overrides or {}).items():
        where = f"datasets: {name}"
        if not isinstance(entry, dict):
            raise SystemExit(f"[config] {where}: expected a mapping (root, runs, label), got {entry!r}")
        _unknown_fields_error(where, entry, DATASET_OVERRIDE_FIELDS)
        entry = {DATASET_OVERRIDE_RENAMES.get(k, k): v for k, v in entry.items()}
        for k in ("root_files", "run_numbers"):
            if isinstance(entry.get(k), (str, int)):
                entry[k] = [entry[k]]
        hits = _base_runs_named(name, base)
        if len(hits) > 1:
            several = ", ".join(_ref_text(base[k].get("kind", "data"), base[k]["particle"], base[k]["energy_gev"],
                                          _key_run(k)) for k in hits)
            raise SystemExit(f"[config] {where}: {name!r} names several runs ({several}); name one of them")
        if hits:
            key = hits[0]
        elif all(k in entry for k in ("particle", "energy_gev")):
            try:
                particle = _particle_word(entry["particle"], str(entry["particle"]))
                energy = parse_energy(entry["energy_gev"])
            except ValueError as exc:
                raise SystemExit(f"[config] {where}: {exc}")
            kind = str(entry.get("kind", "data")).lower()
            run = _parse_ref_or_none(name, 3)
            key = run_key(particle, energy, kind, run)
            entry = {**entry, "particle": particle, "energy_gev": energy, "kind": kind,
                     "folder": run_folder(particle, energy, kind, run)}
        else:
            raise SystemExit(f"[config] {where}: {name!r} is not a run of {source} (runs: {_base_refs(base)}); "
                             f"add new runs to that file (a 'datasets:' entry in config.yml / local.yml only "
                             f"changes fields of existing runs)")
        if key in out:
            raise SystemExit(f"[config] datasets: two entries name the run {key}")
        out[key] = entry
    return out


# keys of the datasets.yml runs that a datasets: override key names
def _base_runs_named(name, base: dict) -> list[str]:
    text = str(name).strip()
    by_key = {k.lower(): k for k in base}
    if text.lower() in by_key:
        return [by_key[text.lower()]]
    try:
        parts = _parse_ref(text)
    except ValueError:
        return []
    return [k for k, d in base.items() if _ref_hits(parts, str(d.get("kind", "data")).lower(), d["particle"],
                                                   float(d["energy_gev"]), d.get("run_numbers") or [])]


# every datasets.yml run as a reference, for messages
def _base_refs(base: dict) -> str:
    return ", ".join(_ref_text(d.get("kind", "data"), d["particle"], d["energy_gev"], _key_run(k))
                     for k, d in base.items()) or "none"


# one field of a parsed run reference, None if the text is not one
def _parse_ref_or_none(text, index: int):
    try:
        return _parse_ref(text)[index]
    except ValueError:
        return None


# {run key: entry} from the particle / energy tree of datasets.yml (mc: -> mc_ keys)
def datasets_from_tree(tree: dict, source: str = "configure/datasets.yml") -> dict:
    out: dict = {}

    # add one run entry, checked: energy, fields, runs, no duplicate
    def add(particle, energy_text, entry, kind, n_entries):
        where = f"{source}: {'mc: ' if kind == 'mc' else ''}{particle} {energy_text}"
        if entry is None:  # "4GeV:" alone: a run whose csv files are already in its folder
            entry = {}
        if not isinstance(entry, dict):
            raise SystemExit(f"[config] {where}: expected a mapping with root/runs/label, got {entry!r}")
        try:
            energy = parse_energy(energy_text)
        except ValueError as exc:
            raise SystemExit(f"[config] {where}: {exc}")
        _unknown_fields_error(where, entry, DATASET_ENTRY_FIELDS)
        runs = entry.get("runs") or []
        runs = [runs] if isinstance(runs, (int, str)) else list(runs)
        try:
            runs = [int(r) for r in runs]
        except (TypeError, ValueError):
            raise SystemExit(f"[config] {where}: runs must be run numbers, got {entry.get('runs')!r}")
        if n_entries > 1 and not runs:
            raise SystemExit(f"[config] {where}: several runs at this energy need 'runs' each "
                             f"(the keys are {particle}_{energy:g}GeV_run<N>)")
        run = runs[0] if n_entries > 1 else None
        key = run_key(particle, energy, kind, run)
        if key in out:
            raise SystemExit(f"[config] {source}: the run {key} appears twice ({particle} {energy_text})")
        roots = entry.get("root") or []
        roots = [roots] if isinstance(roots, str) else list(roots)
        run_text = f" (run{'s' if len(runs) > 1 else ''} {', '.join(str(r) for r in runs)})" if runs else ""
        label = entry.get("label") or f"{'MC ' if kind == 'mc' else ''}{particle.capitalize()} {energy:g} GeV{run_text}"
        out[key] = {"label": str(label), "particle": particle, "energy_gev": energy, "kind": kind,
                    "run_numbers": runs, "root_files": roots, "folder": run_folder(particle, energy, kind, run)}

    # walk one section: particle -> energy -> entry or list of entries
    def walk(section: dict, kind: str):
        for pname, energies in (section or {}).items():
            if kind == "data" and str(pname).strip().lower() == "mc":
                continue
            try:
                particle = _particle_word(pname, str(pname))
            except ValueError:
                raise SystemExit(f"[config] {source}: unknown particle section {pname!r} "
                                 f"(use {', '.join(RUN_PARTICLES)}, or 'mc' for simulation)")
            if not isinstance(energies, dict):
                raise SystemExit(f"[config] {source}: {pname}: expected <energy>: entries, got {energies!r}")
            for etext, entries in energies.items():
                items = entries if isinstance(entries, list) else [entries]
                for entry in items:
                    add(particle, etext, entry, kind, len(items))

    if not isinstance(tree, dict):
        raise SystemExit(f"[config] {source}: expected particle sections (electron:, muon:, pion:, mc:)")
    walk(tree, "data")
    mc = next((v for k, v in tree.items() if str(k).strip().lower() == "mc"), None)
    if mc:
        walk(mc, "mc")
    return out


# --------------------------------------------------------------------------
# run references and run keys
#
#   reference  "muon 4", "electron 0.3", "pion 0.5"   what users, yml settings and code defaults write; case-
#              insensitive, energy in GeV (4GeV / 300MeV also work, see parse_energy), "mc muon 4" for a
#              simulation run, "electron 0.3 run 304" for one of several runs at one energy.
#   key        muon_4GeV, electron_0.3GeV_run304, mc_muon_4GeV   names the products (file names), log lines
#              and provenance; derived from the datasets.yml entry (run_key), never written by hand.
# Abbreviated particle names are not run names: always the full name ("muon 4").
# --------------------------------------------------------------------------
RUN_PARTICLES = ("electron", "muon", "pion")   # particle sections of datasets.yml (+ "mc:" for simulation)
_PARTICLE_NAMES = {**{p: p for p in RUN_PARTICLES}, **{f"{p}s": p for p in RUN_PARTICLES}}
_RUN_REF = re.compile(r"^\s*(?:(mc)[\s_]+)?([a-z]+)[\s_]*([0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?\s*(?:gev|mev)?)"
                      r"(?:[\s_]*run[\s_]*(\d+))?\s*$", re.IGNORECASE)
_ENERGY = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(gev|mev)?\s*$", re.IGNORECASE)


# beam energy in GeV from 0.3, '0.3GeV' or '300MeV'
def parse_energy(value) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        energy = float(value)
    else:
        m = _ENERGY.match(str(value))
        if not m:
            raise ValueError(f"not a beam energy: {value!r} (write e.g. 0.3, 0.3GeV or 300MeV)")
        energy = float(m.group(1)) / (1000.0 if (m.group(2) or "").lower() == "mev" else 1.0)
    if not energy > 0:
        raise ValueError(f"not a positive beam energy: {value!r}")
    return energy


# full particle name (plural accepted) -> particle; abbreviations refused
def _particle_word(word, text: str) -> str:
    particle = _PARTICLE_NAMES.get(str(word).strip().lower())
    if particle is None:
        raise ValueError(f"{text!r}: unknown particle {str(word)!r}; write the full particle name "
                         f"({', '.join(RUN_PARTICLES)}), e.g. 'muon 4' or 'electron 0.3'")
    return particle


# (kind, particle, energy_gev, run number or None) of a run reference or run key
def _parse_ref(text) -> tuple[str, str, float, int | None]:
    s = str(text).strip()
    m = _RUN_REF.match(s)
    if not m:
        raise ValueError(f"{s!r} is not a run reference: write '<particle> <energy in GeV>', e.g. 'muon 4', "
                         f"'electron 0.3', 'pion 0.5'")
    particle = _particle_word(m.group(2), s)
    energy = parse_energy(m.group(3))
    return ("mc" if m.group(1) else "data"), particle, energy, (int(m.group(4)) if m.group(4) else None)


# (particle, energy_gev) of a run reference: 'muon 4' -> ('muon', 4.0)
def parse_run_ref(text) -> tuple[str, float]:
    _, particle, energy, _ = _parse_ref(text)
    return particle, energy


# run key: <particle>_<E>GeV, + _run<N> for one of several runs at one energy, mc_ for simulation
def run_key(particle: str, energy_gev: float, kind: str = "data", run: int | None = None) -> str:
    key = f"{particle}_{float(energy_gev):g}GeV" + (f"_run{int(run)}" if run is not None else "")
    return key if str(kind).lower() == "data" else f"mc_{key}"


# run folder: <Particle>_<E>GeV, + _run<N> for one of several runs at one energy, MC_ for simulation
def run_folder(particle: str, energy_gev: float, kind: str = "data", run: int | None = None) -> str:
    name = f"{str(particle).capitalize()}_{float(energy_gev):g}GeV" + (f"_run{int(run)}" if run is not None else "")
    return name if str(kind).lower() == "data" else f"MC_{name}"


# csv name of a ROOT file: fixed_Muon_4GeV.root -> fixed_Muon_4GeV.csv (every .root suffix dropped)
def root_csv_name(root) -> str:
    name = Path(str(root)).name
    while name.lower().endswith(".root"):
        name = name[:-len(".root")]
    return f"{name}.csv"


# run number of a key that ends in _run<N>, else None
def _key_run(key: str) -> int | None:
    m = re.search(r"_run(\d+)$", str(key))
    return int(m.group(1)) if m else None


# reference text: 'muon 4', 'electron 0.3 run 304', 'mc muon 4'
def _ref_text(kind, particle, energy, run) -> str:
    return (f"{'mc ' if str(kind).lower() != 'data' else ''}{particle} {float(energy):g}"
            + (f" run {run}" if run is not None else ""))


# reference of a run as users and yml files write it: 'muon 4'
def run_ref(ds) -> str:
    return _ref_text(ds.kind, ds.particle, ds.energy_gev, _key_run(ds.key))


# two beam energies equal within rounding?
def _same_energy(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


# does a parsed reference name this run?
def _ref_hits(parts, kind: str, particle: str, energy: float, runs) -> bool:
    pkind, pparticle, penergy, prun = parts
    return (pkind == kind and pparticle == particle and _same_energy(energy, penergy)
            and (prun is None or prun in [int(r) for r in runs]))


PRODUCT_TREES = {"generated_data_dir": "data", "calibration_dir": "calibration", "results_dir": "results",
                 "cache_dir": "cache"}
# paths: entries that are neither product trees nor inputs (data_dir is the parent of the inputs AND of data/)
NON_INPUT_PATHS = (*PRODUCT_TREES, "data_dir", "logs_dir")


# folders of a mode inside data/, calibration/, cache/: () normal run, ('test',) with -test
def mode_parts(mode: str) -> tuple[str, ...]:
    if mode == MODE_FULLSET:
        return ()
    if mode == MODE_TEST:
        return (TEST_DIR,)
    raise ValueError(f"unknown mode {mode!r} (expected {MODE_FULLSET!r} or {MODE_TEST!r})")


# copy of a config whose product trees lie under root (tests, scratch runs)
def redirect_products(cfg: dict, root: str | Path) -> dict:
    out = copy.deepcopy(cfg)
    root = Path(root)
    out["paths"] = {**out.get("paths", {}), **{key: root / sub for key, sub in PRODUCT_TREES.items()}}
    return out


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
# one run of datasets.yml: key, particle, energy, label, kind, run folder, ROOT files, run numbers
@dataclass
class Dataset:
    key: str
    particle: str
    energy_gev: float
    label: str
    kind: str = "data"  # "data" or "mc"
    folder: str = ""   # run folder in analysis_files/ (default <Particle>_<E>GeV[_run<N>], from the key)
    root_files: list[Path] = field(default_factory=list)
    run_numbers: list[int] = field(default_factory=list)

    # default run folder from particle, energy, kind and the _run<N> of the key
    def __post_init__(self):
        if not self.folder:
            self.folder = run_folder(self.particle, self.energy_gev, self.kind, _key_run(self.key))

    # beam energy as text: '4 GeV'
    @property
    def energy_label(self) -> str:
        return f"{self.energy_gev:g} GeV"

    # file-name tag: electron_0p3GeV
    @property
    def tag(self) -> str:
        return f"{self.particle}_{self.energy_gev:g}GeV".replace(".", "p")


# {run key: Dataset} of a loaded config
def dataset_registry(cfg: dict) -> dict[str, Dataset]:
    reg = {}
    for key, d in (cfg.get("datasets") or {}).items():
        roots = d.get("root_files") or []
        if isinstance(roots, str):
            roots = [roots]
        reg[key] = Dataset(
            key=key,
            particle=str(d["particle"]).lower(),
            energy_gev=float(d["energy_gev"]),
            label=d.get("label", key),
            kind=str(d.get("kind", "data")).lower(),
            folder=str(d.get("folder") or ""),
            root_files=[resolve_path(r) for r in roots],
            run_numbers=[int(r) for r in (d.get("run_numbers") or [])],
        )
    return reg


# datasets.yml path for messages (relative to data_analysis/)
def _datasets_file(cfg: dict) -> str:
    path = (cfg.get("paths") or {}).get("datasets_file")
    if path is None:
        return "configure/datasets.yml"
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# every run as a reference, for messages: 'electron 0.3, ..., muon 4, pion 0.5'
def known_runs(cfg: dict) -> str:
    reg = dataset_registry(cfg)
    order = sorted(reg.values(), key=lambda d: (d.kind != "data", d.particle, d.energy_gev, d.key))
    return ", ".join(run_ref(d) for d in order) or "none"


# runs named by one item: run reference, run key, particle name or 'all'
def _runs_named(cfg: dict, reg: dict[str, Dataset], item, what: str) -> list[Dataset]:
    text = str(item).strip()
    low = text.lower()
    if low == "all":  # every data run (simulation runs are named one by one: "mc muon 4")
        return [d for d in reg.values() if d.kind == "data"]
    by_key = {k.lower(): d for k, d in reg.items()}
    if low in by_key:
        return [by_key[low]]
    hint = f"runs of {_datasets_file(cfg)}: {known_runs(cfg)}"
    if low in _PARTICLE_NAMES:  # "electron" alone: every data run of that particle (like --electron alone)
        hits = [d for d in reg.values() if d.kind == "data" and d.particle == _PARTICLE_NAMES[low]]
        if not hits:
            raise SystemExit(f"{what}: there is no {_PARTICLE_NAMES[low]} run; {hint}")
        return hits
    try:
        parts = _parse_ref(text)
    except ValueError as exc:
        raise SystemExit(f"{what}: {exc}; {hint}")
    hits = [d for d in reg.values() if _ref_hits(parts, d.kind, d.particle, d.energy_gev, d.run_numbers)]
    if not hits:
        raise SystemExit(f"{what}: there is no run {_ref_text(*parts)!r}; {hint}")
    return hits


# runs named by references / run keys / particle names / 'all', in order, no repeats
def resolve_runs(cfg: dict, refs, what: str = "run") -> list[Dataset]:
    """An unknown run stops with what (the setting or option) and every existing run."""
    reg = dataset_registry(cfg)
    items = [] if refs is None else ([refs] if isinstance(refs, (str, Dataset)) else list(refs))
    chosen: list[Dataset] = []
    for item in items:
        hits = [item] if isinstance(item, Dataset) else _runs_named(cfg, reg, item, what)
        chosen += [d for d in hits if all(d.key != c.key for c in chosen)]
    return chosen


# runs of the given tokens, else the stage's default runs
def resolve_datasets(cfg: dict, tokens, default=None, what: str | None = None) -> list[Dataset]:
    tokens = [tokens] if isinstance(tokens, str) else list(tokens or [])
    if tokens:
        return resolve_runs(cfg, tokens, what or "selected runs")
    return resolve_runs(cfg, default, what or "default runs")


# value of a per-run yml mapping {run: value} for one run (default when no key names it)
def run_setting(mapping, ds, default=None, what: str | None = None):
    """The most specific key wins: run key > reference with run number > reference > particle."""
    if mapping is None:
        return default
    name = what or "per-run setting"
    if not isinstance(mapping, dict):
        raise SystemExit(f"[config] {name}: expected a mapping {{run: value}} (e.g. 'electron 0.3: ...'), "
                         f"got {mapping!r}")
    best, rank = default, -1
    for key, value in mapping.items():
        r = _setting_rank(key, ds, name)
        if r > rank:
            best, rank = value, r
    return best


# how specifically a per-run key names the run (-1 = not at all)
def _setting_rank(key, ds, what: str) -> int:
    text = str(key).strip()
    low = text.lower()
    if low == str(ds.key).lower():
        return 3
    if low in _PARTICLE_NAMES:
        return 0 if (ds.kind == "data" and ds.particle == _PARTICLE_NAMES[low]) else -1
    try:
        parts = _parse_ref(text)
    except ValueError as exc:
        raise SystemExit(f"[config] {what}: key {text!r}: {exc}")
    if not _ref_hits(parts, ds.kind, ds.particle, ds.energy_gev, ds.run_numbers):
        return -1
    return 2 if parts[3] is not None else 1


SELECTABLE_PARTICLES = RUN_PARTICLES   # command-line flags --electron / --muon / --pion


# command-line flags of one run, for messages: --muon 4 (the run key for a simulation run)
def run_flags(ds) -> str:
    if ds.kind != "data" or ds.particle not in SELECTABLE_PARTICLES:
        return ds.key
    return f"--{ds.particle} {ds.energy_gev:g}"


# run keys chosen with --electron / --muon / --pion E ... (a flag alone = every run of that particle)
def select_by_particle(cfg: dict, selections: dict) -> list[str]:
    reg = dataset_registry(cfg)
    keys: list[str] = []
    for particle in SELECTABLE_PARTICLES:
        occurrences = selections.get(particle)
        if not occurrences:
            continue
        runs = sorted((d for d in reg.values() if d.particle == particle and d.kind == "data"),
                      key=lambda d: (d.energy_gev, d.key))
        if not runs:
            raise SystemExit(f"--{particle}: there is no {particle} run; runs of {_datasets_file(cfg)}: "
                             f"{known_runs(cfg)}")
        wanted = [e for occ in occurrences for e in (occ or [])]
        if any(not occ for occ in occurrences) or not wanted:
            chosen = runs
        else:
            chosen = []
            for token in wanted:
                try:
                    energy = parse_energy(token)
                except ValueError as exc:
                    raise SystemExit(f"--{particle} {token}: {exc} (expected a beam energy, e.g. --{particle} 4)")
                hit = [d for d in runs if _same_energy(d.energy_gev, energy)]
                if not hit:
                    raise SystemExit(f"--{particle} {token}: there is no run '{particle} {energy:g}'; "
                                     f"runs of {_datasets_file(cfg)}: {known_runs(cfg)}")
                chosen += hit
        keys += [d.key for d in chosen if d.key not in keys]
    return keys


# --------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------
# options shared by every stage + the product paths of both modes (paths only, nothing is made)
@dataclass
class RunContext:

    cfg: dict
    test: bool = False
    n_test_events: int = 5000
    make_plots: bool = True
    workers: int = 1
    dry_run: bool = False
    clean_input: bool = False  # --clean-input: read the clean dataset instead of the beam csv
    outputs: list | None = None  # --output names (None: not given; []: --output alone); common/outputs.py
    write_json: bool = True  # False with -no-json: no record json next to the products (cache records stay)
    units: tuple | None = None  # -only: output units of the running stage (None: every unit); common/outputs.py

    # switch on the write protection (safety.protected_dirs) and the json records (-no-json)
    def __post_init__(self):
        from .io_utils import set_json_records, set_protected_dirs  # local import avoids a cycle

        set_protected_dirs((self.cfg.get("safety") or {}).get("protected_dirs") or [])
        try:
            cache = [self.cache_root()]
        except (KeyError, TypeError):  # hand-made config without any tree
            cache = []
        paths = self.cfg.get("paths") or {}
        records = [Path(paths[k]) for k in ("generated_data_dir", "calibration_dir", "results_dir") if paths.get(k)]
        set_json_records(self.write_json, records, cache)

    # does the running stage make this output unit (any of these units)? -only, else the stage's default units
    def wants(self, *units: str) -> bool:
        return self.units is None or any(u in self.units for u in units)

    # 'fullset' or 'test'
    @property
    def mode(self) -> str:
        return MODE_TEST if self.test else MODE_FULLSET

    # a fixed path of paths: in the yml
    def path(self, key: str) -> Path:
        return self.cfg["paths"][key]

    # the yml section of a stage
    def stage_cfg(self, name: str) -> dict:
        return self.cfg.get(name) or {}

    # -- the product trees (pure path construction, nothing is created) --
    # <tree>/[test/]<parts>
    def _in_tree(self, root, parts, mode: str | None) -> Path:
        return Path(root).joinpath(*mode_parts(mode or self.mode), *parts)

    # data/<parts> (-test: data/test/<parts>)
    def data_dir(self, *parts: str, mode: str | None = None) -> Path:
        return self._in_tree(self.cfg["paths"]["generated_data_dir"], parts, mode)

    # calibration/<parts> (-test: calibration/test/<parts>)
    def calib_dir(self, *parts: str, mode: str | None = None) -> Path:
        return self._in_tree(self.cfg["paths"]["calibration_dir"], parts, mode)

    # results/fullset/<parts> (-test: results/test/<parts>)
    def results_dir(self, *parts: str, mode: str | None = None) -> Path:
        return self._results_root().joinpath(RESULTS_MODE_DIRS[mode or self.mode], *parts)

    # root of the results tree
    def _results_root(self) -> Path:
        return Path(self.cfg["paths"]["results_dir"])

    # is this tree root the results tree?
    def _is_results_root(self, root) -> bool:
        from .io_utils import canonical_path

        return bool(self.cfg.get("paths", {}).get("results_dir")) and canonical_path(root) == canonical_path(
            self._results_root())

    # the cache tree: paths.cache_dir, else cache/ next to the data tree
    def cache_root(self) -> Path:
        root = self.cfg["paths"].get("cache_dir")
        if root is None:
            root = Path(self.cfg["paths"]["generated_data_dir"]).parent / "cache"
        return Path(root)

    # cache/<parts> (-test: cache/test/<parts>): intermediates only
    def cache_dir(self, *parts: str, mode: str | None = None) -> Path:
        return self._in_tree(self.cache_root(), parts, mode)

    # -- whole modes: what code that walks, lists or classifies every product of a mode must use --
    # roots of the product trees of this config
    def tree_roots(self) -> list[Path]:
        paths = self.cfg.get("paths") or {}
        roots = [Path(paths[k]) for k in ("generated_data_dir", "calibration_dir", "results_dir") if paths.get(k)]
        try:
            roots.append(self.cache_root())
        except (KeyError, TypeError):  # hand-made config without any tree
            pass
        return roots

    # every input of the config: paths: inputs + the run folder and ROOT files of every run
    def input_locations(self) -> list[Path]:
        from .io_utils import canonical_path  # local import avoids a cycle

        out = [Path(v) for k, v in (self.cfg.get("paths") or {}).items() if v and k not in NON_INPUT_PATHS]
        for ds in dataset_registry(self.cfg).values():
            out += [p for p in [self.run_dir(ds, mode=MODE_FULLSET), *ds.root_files] if p is not None]
        seen, unique = set(), []
        for p in out:
            key = canonical_path(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # is a folder at a tree root a normal-mode product folder (not test/, not an input)?
    def _tree_entry_is_product(self, entry: Path, mode: str, inputs: list[str]) -> bool:
        from .io_utils import canonical_path

        if mode != MODE_FULLSET or entry.name == TEST_DIR or entry.name in INPUT_DIRS:
            return False
        key = canonical_path(entry)
        sep = key.rstrip("\\/") + os.sep
        return not any(i == key or i.startswith(sep) for i in inputs)

    # the folders that hold every product of a mode (never an input or the other mode)
    def mode_tree_dirs(self, mode: str | None = None) -> list[Path]:
        mode = mode or self.mode
        if mode == MODE_TEST:
            return [d for d in (r.joinpath(*mode_parts(mode)) for r in self.tree_roots()) if d.is_dir()]
        from .io_utils import canonical_path

        inputs = [canonical_path(p) for p in self.input_locations()]
        out = []
        for root in self.tree_roots():
            if self._is_results_root(root):  # results/fullset/ holds every normal-mode result
                d = root / RESULTS_MODE_DIRS[MODE_FULLSET]
                out += [d] if d.is_dir() else []
                continue
            try:
                entries = sorted(e for e in root.iterdir() if e.is_dir())
            except OSError:
                continue
            out += [e for e in entries if self._tree_entry_is_product(e, mode, inputs)]
        return out

    # does a path lie inside the products of a mode ('any' = both)?
    def in_products(self, path, mode: str | None = None) -> bool:
        from .io_utils import canonical_path

        modes = (MODE_FULLSET, MODE_TEST) if mode == "any" else (mode or self.mode,)
        target = canonical_path(path)
        inputs = None
        for root in self.tree_roots():
            base = canonical_path(root).rstrip("\\/") + os.sep
            if not target.startswith(base):
                continue
            rest = target[len(base):].split(os.sep)
            if self._is_results_root(root):  # results/fullset/... or results/test/...
                if (rest[0] == RESULTS_MODE_DIRS[MODE_FULLSET] and MODE_FULLSET in modes) or (
                        rest[0] == TEST_DIR and MODE_TEST in modes):
                    return True
                continue
            if rest[0] == TEST_DIR:  # the test tree or a path inside it
                if MODE_TEST in modes:
                    return True
                continue
            if len(rest) == 1 and (Path(target).is_file() or Path(rest[0]).suffix):
                continue  # a file at the root of the tree (README.md): never a product
            if MODE_FULLSET in modes:
                if inputs is None:
                    inputs = [canonical_path(p) for p in self.input_locations()]
                if self._tree_entry_is_product(Path(base) / rest[0], MODE_FULLSET, inputs):
                    return True
        return False

    # -- canonical locations of stage products (single source of truth) --
    # calib table: calibration/[test/]pedestal_mpv/calibration_parameters.csv
    def calibration_product(self, mode: str | None = None) -> Path:
        name = self.stage_cfg("calibration").get("output_csv", "calibration_parameters.csv")
        return self.calib_dir("pedestal_mpv", mode=mode) / name

    # table the later stages read: this mode's table (-test without one: the full-set table)
    def calibration_file(self) -> Path:
        product = self.calibration_product()
        if self.test and not product.exists():
            production = self.calibration_product(mode=MODE_FULLSET)
            if production.exists():
                return production
        return product

    # -- runs: a run is the folder of its csv files (one csv per ROOT file, nothing merged) --
    # analysis_files folder: paths.analysis_files_dir (-test: data/test/analysis_files, the make-csv -test csv)
    def analysis_files_dir(self, mode: str | None = None) -> Path:
        if (mode or self.mode) == MODE_TEST:
            return self.data_dir(ANALYSIS_FILES_DIR, mode=MODE_TEST)
        folder = (self.cfg.get("paths") or {}).get("analysis_files_dir")
        return Path(folder) if folder else self.data_dir(ANALYSIS_FILES_DIR, mode=MODE_FULLSET)

    # run folder: analysis_files/<Particle>_<E>GeV[_run<N>]/ (-test: data/test/analysis_files/<run>/)
    def run_dir(self, ds: Dataset, mode: str | None = None) -> Path:
        if (mode or self.mode) == MODE_TEST:
            return self.analysis_files_dir(MODE_TEST) / _run_name(ds)
        return self.analysis_files_dir(MODE_FULLSET) / ds.folder

    # csv files of a run (full set): every csv of its folder, else one per ROOT file (still to be made)
    def run_csvs(self, ds: Dataset) -> list[Path]:
        folder = self.run_dir(ds, mode=MODE_FULLSET)
        found = sorted((p for p in folder.glob("*.csv") if p.is_file()), key=lambda p: p.name) \
            if folder.is_dir() else []
        return found or [folder / root_csv_name(r) for r in ds.root_files]

    # csv of one ROOT file of a run: <run folder>/<ROOT file name>.csv (-test: in the test run folder)
    def root_csv(self, ds: Dataset, root, mode: str | None = None) -> Path:
        return self.run_dir(ds, mode) / root_csv_name(root)

    # make-csv -test csv of one csv of a run: data/test/analysis_files/<run>/<csv name>
    def test_csv(self, ds: Dataset, csv) -> Path:
        return self.run_dir(ds, mode=MODE_TEST) / Path(csv).name

    # test subset of one csv of a run: cache/test/test_subsets/<run>/<csv stem>_first<N>events.csv
    def test_subset_csv(self, ds: Dataset, csv) -> Path:
        return (self.cache_dir("test_subsets", _run_name(ds), mode=MODE_TEST)
                / f"{Path(csv).stem}_first{self.n_test_events}events.csv")

    # product of one input csv: <folder>/<run>/<csv stem><suffix> (one product per csv, a sub-folder per run)
    def csv_product(self, folder, ds: Dataset, csv, suffix: str) -> Path:
        return Path(folder) / _run_name(ds) / f"{Path(csv).stem}{suffix}"

    # clean dataset of one csv: data/[test/]clean_dataset/<run>/<csv stem>_clean.csv
    def clean_product(self, ds: Dataset, csv, mode: str | None = None) -> Path:
        return self.csv_product(self.data_dir("clean_dataset", mode=mode), ds, csv, "_clean.csv")

    # event source of one csv of a run: '<run>/<csv name>'; event identity = (event source, pf_event)
    def event_source(self, ds: Dataset, csv) -> str:
        return f"{_run_name(ds)}/{Path(csv).name}"

    # the make-csv -test csv of one csv of a run when its provenance is current, else None
    def _made_test_csv(self, ds: Dataset, csv) -> Path | None:
        if not self.test_csv(ds, csv).exists():
            return None
        try:
            from ..conversion.make_analysis_files import current_test_csv
        except ImportError:  # converter unavailable: subsets still work
            return None
        return current_test_csv(self, ds, Path(csv))

    # the clean dataset of one csv of a run when it is current, else None
    def _current_clean_csv(self, ds: Dataset, csv) -> Path | None:
        try:
            from ..selection.clean_dataset import current_clean_csv
        except ImportError as exc:
            raise SystemExit(f"--clean-input: the clean stage cannot be loaded ({type(exc).__name__}: {exc})")
        path = current_clean_csv(self, ds, Path(csv))
        return Path(path) if path is not None else None

    # clean command that makes a run's clean dataset (with_flags: the selection flags of csv's record)
    def _clean_hint(self, ds: Dataset, csv=None, with_flags: bool = False) -> str:
        if with_flags and csv is not None:
            try:
                from ..selection import clean_dataset as clean
                record = clean.clean_status(self, ds, Path(csv))[2]
                return clean._remake_command(self, ds, record if isinstance(record, dict) else None)
            except Exception:  # the hint must never hide the error it belongs to
                pass
        return f".\\runner.ps1 clean {run_flags(ds)}" + (" -test" if self.test else "")

    # the csv files of a run, or a stop with the folder and the make-csv command (dry run: maybe empty)
    def _run_csvs_or_stop(self, ds: Dataset) -> list[Path]:
        csvs = self.run_csvs(ds)
        if csvs or self.dry_run:
            return csvs
        raise SystemExit(f"run {run_ref(ds)}: no csv file in {self.run_dir(ds, mode=MODE_FULLSET)} and no ROOT "
                         f"file in {_datasets_file(self.cfg)}; put its csv files there or make them with "
                         f".\\runner.ps1 make-csv {run_flags(ds)}")

    # (run csv, csv to read) of every csv of a run, in run_csvs order: the pairs a stage loops over
    def run_inputs(self, ds: Dataset) -> list[tuple[Path, Path]]:
        return [(csv, self.input_csv(ds, csv)) for csv in self._run_csvs_or_stop(ds)]

    # every csv a stage reads for a run (input_csv of each csv of the run)
    def input_csvs(self, ds: Dataset) -> list[Path]:
        return [path for _, path in self.run_inputs(ds)]

    # every beam csv of a run (beam_csv of each csv of the run)
    def beam_csvs(self, ds: Dataset) -> list[Path]:
        return [self.beam_csv(ds, csv) for csv in self._run_csvs_or_stop(ds)]

    # csv a stage reads for one csv of a run: its clean dataset with -clean-input, else beam_csv
    def input_csv(self, ds: Dataset, csv) -> Path:
        """With dry_run nothing is built; the path the run would read is returned."""
        csv = Path(csv)
        if self.clean_input:
            if self.dry_run:  # a plan only shows where the run would read
                try:
                    return self._current_clean_csv(ds, csv) or self.clean_product(ds, csv)
                except SystemExit:
                    return self.clean_product(ds, csv)
            path = self._current_clean_csv(ds, csv)
            if path is not None:
                return path
            raise SystemExit(f"--clean-input: no current clean dataset for {run_ref(ds)} {csv.name} ({self.mode}): "
                             f"{self.clean_product(ds, csv)} is missing or was made from other inputs / settings.\n"
                             f"  Make it with: {self._clean_hint(ds, csv, with_flags=True)}")
        return self.beam_csv(ds, csv)

    # beam csv of one csv of a run: the csv; -test: its make-csv -test csv, else its first-N-events subset
    def beam_csv(self, ds: Dataset, csv) -> Path:
        """With dry_run nothing is built; the path the run would read is returned."""
        csv = Path(csv)
        if not self.test:
            return csv
        made = self._made_test_csv(ds, csv)
        if made is not None:
            return made
        cache = self.test_subset_csv(ds, csv)
        if self.dry_run:
            return cache
        from .io_utils import build_test_subset  # local import avoids a cycle

        return build_test_subset(csv, cache, self.n_test_events,
                                 chunksize=int(self.cfg.get("run", {}).get("chunksize", 2_000_000)))

    # plan items of every csv a stage reads for a run (the run folder as missing input when it holds none)
    def input_plan_items(self, ds: Dataset, produced_by: str | None = None) -> list[dict]:
        csvs = self.run_csvs(ds)
        if not csvs:
            return [self._missing_run_item(ds, produced_by)]
        return [self.input_plan_item(ds, csv, produced_by) for csv in csvs]

    # plan items of every beam csv of a run
    def beam_plan_items(self, ds: Dataset, produced_by: str | None = None) -> list[dict]:
        csvs = self.run_csvs(ds)
        if not csvs:
            return [self._missing_run_item(ds, produced_by)]
        return [self.beam_plan_item(ds, csv, produced_by) for csv in csvs]

    # plan item of a run folder without any csv: <run folder>/*.csv, missing
    def _missing_run_item(self, ds: Dataset, produced_by: str | None) -> dict:
        return {"role": "in", "kind": "external", "path": self.run_dir(ds, mode=MODE_FULLSET) / "*.csv",
                "produced_by": produced_by or f".\\runner.ps1 make-csv {run_flags(ds)}"}

    # plan item of input_csv with its kind: external / product / auto (+ stale)
    def input_plan_item(self, ds: Dataset, csv, produced_by: str | None = None) -> dict:
        csv = Path(csv)
        if self.clean_input:
            try:
                path = self._current_clean_csv(ds, csv)
            except SystemExit:
                path = None
            item = {"role": "in", "kind": "product", "path": path or self.clean_product(ds, csv),
                    "produced_by": self._clean_hint(ds)}
            if path is None and self.clean_product(ds, csv).exists():
                item["stale"] = True
            return item
        return self.beam_plan_item(ds, csv, produced_by)

    # plan item of beam_csv with its kind
    def beam_plan_item(self, ds: Dataset, csv, produced_by: str | None = None) -> dict:
        csv = Path(csv)
        make_cmd = produced_by or f".\\runner.ps1 make-csv {run_flags(ds)}"
        if not self.test:
            return {"role": "in", "kind": "external", "path": csv, "produced_by": make_cmd}
        made = self._made_test_csv(ds, csv)
        if made is not None:
            return {"role": "in", "kind": "product", "path": made, "produced_by": f"{make_cmd} -test"}
        return {"role": "in", "kind": "auto", "path": self.test_subset_csv(ds, csv), "source": csv,
                "n_events": int(self.n_test_events),
                "produced_by": f"test subset of {csv.name} (built automatically from the full csv)"}


ANALYSIS_FILES_DIR = "analysis_files"  # the run folders: data/analysis_files/<run>/, data/test/analysis_files/<run>/
EVENT_SOURCE_COLUMN = "event_source"   # '<run>/<csv name>' of an event (RunContext.event_source)
EVENT_ID_COLUMNS = (EVENT_SOURCE_COLUMN, "pf_event")  # an event: two csv files may both start at pf_event 1


# name of a run's folder (its last part, also for a hand-made absolute folder)
def _run_name(ds: Dataset) -> str:
    return Path(ds.folder).name
