"""configure/config.yml must list every stage default with exactly the value the code uses.

The yml is the documented place to read and change defaults, while each module keeps its own
``DEFAULTS`` dict (so the code also runs without the yml).  This test fails as soon as the two drift
apart: a key added to a module but not to the yml, a value changed on one side only, or a stray key.
The command-line-only settings (energy_response.CLI_ONLY_SETTINGS: the spectrum fit range -fit / -bin) are in
neither, and no yml of configure/ may set them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmx_hcal.common.config import DEFAULT_CONFIG, load_config  # noqa: E402

SECTIONS = {
    "make_csv": "ldmx_hcal.conversion.make_analysis_files",
    "calibration": "ldmx_hcal.calibration.calibration_parameters",
    "clean_dataset": "ldmx_hcal.selection.clean_dataset",
    "tot_adc_peak": "ldmx_hcal.energy.tot_adc_peak",
    "charge_injection": "ldmx_hcal.energy.charge_injection",
    "energy_response": "ldmx_hcal.energy.energy_response",
    "plots": "ldmx_hcal.visualization.multiscale_plots",
}


# equal yml and DEFAULTS values (tuples as lists, numbers as floats)
def _same(a, b) -> bool:
    if isinstance(b, tuple):
        b = list(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


# every stage section of config.yml equals the module DEFAULTS
def test_every_stage_section_equals_module_defaults():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))  # unexpanded, as written
    problems = []
    for section, modname in SECTIONS.items():
        mod = importlib.import_module(modname)
        assert getattr(mod, "CONFIG_SECTION", section) == section, (modname, mod.CONFIG_SECTION)
        defaults = mod.DEFAULTS
        yml = raw.get(section)
        if not isinstance(yml, dict):
            problems.append(f"{section}: missing in configure/config.yml")
            continue
        for k in sorted(set(defaults) - set(yml)):
            problems.append(f"{section}.{k}: in DEFAULTS, missing in the yml")
        for k in sorted(set(yml) - set(defaults)):
            problems.append(f"{section}.{k}: in the yml, unknown to the module")
        for k in sorted(set(yml) & set(defaults)):
            if not _same(yml[k], defaults[k]):
                problems.append(f"{section}.{k}: yml {yml[k]!r} != DEFAULTS {defaults[k]!r}")
    assert not problems, "\n".join(problems)


# the config loads with every stage section
def test_config_loads_with_local_overrides():
    cfg = load_config()
    for section in SECTIONS:
        assert section in cfg, section


# -fit / -bin are command line only (MIP vs a.u. spectra); linearity_fits / resolution_fit_range (GeV) are yml keys
def test_spectrum_fit_ranges_are_command_line_only():
    from ldmx_hcal.common.io_utils import SettingsError
    from ldmx_hcal.energy import energy_response as er

    cli_only = set(er.CLI_ONLY_SETTINGS)
    assert cli_only == {"fit_ranges", "fit_bins"} and not cli_only & set(er.DEFAULTS)
    for path in sorted(DEFAULT_CONFIG.parent.glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        section = raw.get("energy_response") if isinstance(raw, dict) else None
        assert not cli_only & set(section or {}), f"{path.name}: energy_response sets {cli_only & set(section)}"
    section = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["energy_response"]
    assert {"linearity_fits", "resolution_fit_range"} <= set(section) & set(er.DEFAULTS)

    # yml section energy_response plus extra keys
    class _Ctx:

        # extra keys
        def __init__(self, extra):
            self.extra = extra

        # the section with the extra keys
        def stage_cfg(self, name):
            return {**section, **self.extra} if name == er.CONFIG_SECTION else {}

    assert er.settings(_Ctx({}))["fit_ranges"] is None   # the shipped section loads; the key exists (None)
    for key, value in (("fit_ranges", [[14, 26]]), ("fit_bins", [24])):
        try:
            er.settings(_Ctx({key: value}))
        except SettingsError as exc:
            assert key in str(exc) and "command-line only" in str(exc), exc
        else:
            raise AssertionError(f"energy_response.{key} in the yml must stop the run")


if __name__ == "__main__":
    from run_all import main

    sys.exit(main(["config_defaults"]))
