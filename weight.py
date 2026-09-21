"""
weight.py  --  Phase 3, module 2

Steel weight of a candidate pole shaft.

Standard library only. Depends on geometry.py.

--------------------------------------------------------------------------
CONVENTIONS  (validated against three real PLS-POLE exports)
--------------------------------------------------------------------------
Density
    490 lb/ft^3. This reproduces PLS-POLE's per-tube weights to about
    0.15% and its reported total to within 0.4 lb on all three reference
    files. PLS-POLE takes the density from the base plate properties
    table; poles carry a `weight_density_override` field which is 0 in
    all reference files, meaning "use the default".

Per-tube weight
    A tube's weight covers its FULL fabricated length, including the lap.
    The lap is therefore counted twice in the total -- once for the female
    tube and once for the male tube inside it -- which is correct, because
    both pieces of steel exist and both are paid for. Confirmed:
    sum(PLS-POLE tube weights) equals PLS-POLE's reported shaft weight
    exactly on the direct-buried pole (24,809.7 lb), and equals reported
    weight minus base plate weight on both base-plate poles.

Embedded steel
    Counted. The buried portion of a direct-buried pole is real material.
    Embedment depth is a FIXED input and is never optimized.

Base plate
    NOT included. Base plate optimization is out of scope: the user
    optimizes the shaft first, then designs the base plate to suit. For
    reference, `pls_reported_weight` in an export for a base-plate pole
    equals shaft weight plus base plate weight.
"""

from __future__ import annotations

from typing import Optional

from geometry import (PoleSpec, dodecagon_section_properties,
                      STEEL_DENSITY_PCF)


def segment_weight(d_top: float, d_bot: float, length_ft: float,
                   thickness: float,
                   density_pcf: float = STEEL_DENSITY_PCF,
                   n_steps: int = 200) -> float:
    """Weight of one tapered 12-sided tube, in pounds.

    The cross-sectional area varies linearly with diameter, so the exact
    integral is the area at mid-length times the length. The numerical
    integration below is kept because it stays correct if the area
    formula is ever changed to something non-linear in D.

    d_top, d_bot, thickness in inches; length_ft in feet.
    """
    if length_ft <= 0:
        return 0.0
    dz_ft = length_ft / n_steps
    volume_in3 = 0.0
    for k in range(n_steps):
        frac = (k + 0.5) / n_steps
        d = d_top + (d_bot - d_top) * frac
        area_in2 = dodecagon_section_properties(d, thickness)['Ag']
        volume_in3 += area_in2 * dz_ft * 12.0
    return volume_in3 / 1728.0 * density_pcf


def pole_weight(spec: PoleSpec,
                density_pcf: float = STEEL_DENSITY_PCF,
                n_steps: int = 200) -> dict:
    """Shaft weight of a candidate pole.

    Returns a dict with the total, a per-tube breakdown, and the weight of
    steel below the ground line (reported separately for information --
    it IS included in the total).
    """
    per_tube = []
    total = 0.0
    for tb in spec.layout():
        w = segment_weight(tb['d_top'], tb['d_bot'], tb['length'],
                           tb['thickness'], density_pcf, n_steps)
        per_tube.append({
            'tube_no': tb['tube_no'],
            'length': tb['length'],
            'thickness': tb['thickness'],
            'd_top': tb['d_top'],
            'd_bot': tb['d_bot'],
            'lap': tb['lap'],
            'weight': w,
        })
        total += w

    embedded_weight = 0.0
    if spec.embedment:
        gl = spec.groundline_rel
        for tb in spec.layout():
            lo, hi = max(tb['start'], gl), min(tb['end'], spec.total_length)
            if hi > lo:
                embedded_weight += segment_weight(
                    spec.diameter_at(lo), spec.diameter_at(hi),
                    hi - lo, tb['thickness'], density_pcf, n_steps)

    return {
        'total_weight': total,
        'per_tube': per_tube,
        'embedded_weight': embedded_weight,
        'above_ground_weight': total - embedded_weight,
        'lap_double_counted_weight': sum(
            segment_weight(spec.diameter_at(tb['end'] - tb['lap']),
                           spec.diameter_at(tb['end']),
                           tb['lap'], tb['thickness'], density_pcf, n_steps)
            for tb in spec.layout() if tb['lap']),
    }


def weight_per_ft(spec: PoleSpec, density_pcf: float = STEEL_DENSITY_PCF) -> float:
    """Shaft weight divided by total length -- a quick comparison metric."""
    return pole_weight(spec, density_pcf)['total_weight'] / spec.total_length
