"""HCal prototype geometry: 19 layers of 8 (layers 1-9) or 12 bars, 2 ends: 192 bars, 384 channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_LAYERS = 19
LAYERS = np.arange(1, N_LAYERS + 1)
LAST_NARROW_LAYER = 9          # layers 1..9 have 8 bars
N_STRIPS_NARROW = 8
N_STRIPS_WIDE = 12
N_POSITIONS = 12               # physical strip positions 0..11
NARROW_OFFSET = (N_STRIPS_WIDE - N_STRIPS_NARROW) // 2  # = 2
ENDS = (0, 1)
N_TIME_SAMPLES = 8             # HGCROC samples per readout
ADC_SAMPLE_RAIL = 1023         # 10-bit ADC saturation per sample


# bars in a layer: 8 (layers 1-9) or 12
def n_strips(layer):
    layer = np.asarray(layer)
    out = np.where(layer <= LAST_NARROW_LAYER, N_STRIPS_NARROW, N_STRIPS_WIDE)
    return out.item() if out.ndim == 0 else out


# drawing position 0..11: strip + 2 in the 8-bar layers
def physical_position(layer, strip):
    layer = np.asarray(layer)
    strip = np.asarray(strip)
    out = np.where(layer <= LAST_NARROW_LAYER, strip + NARROW_OFFSET, strip)
    return out.item() if out.ndim == 0 else out


# all 192 (layer, strip) in detector order
def all_bars() -> pd.DataFrame:
    rows = [(l, s) for l in LAYERS for s in range(n_strips(l))]
    return pd.DataFrame(rows, columns=["layer", "strip"])


# all 384 (layer, strip, end) in detector order
def all_channels() -> pd.DataFrame:
    rows = [(l, s, e) for l in LAYERS for s in range(n_strips(l)) for e in ENDS]
    return pd.DataFrame(rows, columns=["layer", "strip", "end"])


# channel name: L1S0E0
def channel_name(layer, strip, end) -> str:
    return f"L{int(layer)}S{int(strip)}E{int(end)}"


# (layer, strip) inside the detector?
def is_valid_channel(layer, strip) -> np.ndarray:
    layer = np.asarray(layer)
    strip = np.asarray(strip)
    return (layer >= 1) & (layer <= N_LAYERS) & (strip >= 0) & (strip < n_strips(np.clip(layer, 1, N_LAYERS)))


# (19, 12) grid [layer - 1, drawing position] filled with fill
def empty_layer_grid(fill=np.nan) -> np.ndarray:
    return np.full((N_LAYERS, N_POSITIONS), fill, dtype=float)


# grid of per-bar values at their drawing positions
def fill_layer_grid(layer, strip, values, fill=np.nan) -> np.ndarray:
    grid = empty_layer_grid(fill)
    layer = np.asarray(layer, dtype=int)
    pos = np.asarray(physical_position(layer, strip), dtype=int)
    grid[layer - 1, pos] = np.asarray(values, dtype=float)
    return grid
