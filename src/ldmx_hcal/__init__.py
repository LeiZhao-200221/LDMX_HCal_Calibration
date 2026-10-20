"""LDMX 2022 CERN test-beam HCal prototype analysis (data side).

Stages (run through ``runner.ps1`` / ``src/runner.py``), by category:
    conversion.make_analysis_files       ROOT (data + MC) -> per-bar beam csv
    calibration.calibration_parameters   pedestal + MIP MPV per channel
    selection.clean_dataset              clean muon (and electron) selections
    energy.tot_adc_peak                  TOT -> ADC-peak line table, Arbitrary Energy response [a.u.]
    energy.charge_injection              CI map + s fit; adc_only / PureCI / s reconstruction (MIP), e/h
    energy.energy_response               shared spectrum, linearity, resolution and e/h fits
    visualization.multiscale_plots       energy distributions (event / bar / channel), event displays, bar maps,
                                         layer profiles and hit maps
Shared code (``common``): config, geometry, io_utils, fitting, plotting, calibration_io.
"""

__version__ = "1.1.0"
