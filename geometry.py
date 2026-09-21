"""
geometry.py  --  Phase 3, module 1

Generates the geometry of a candidate steel transmission pole from a
compact design spec, and validates it against fabrication/design limits.

Standard library only -- no third-party dependencies -- so it runs
anywhere (local, Cloud Functions, Cloud Run, notebooks).

--------------------------------------------------------------------------
CONVENTIONS  (all derived from and validated against real PLS-POLE exports:
001_T316-S-01_105FT_FDN, 004_T316-S-02_90FT_FDN, 005_T316-S-03_145FT_EMB)
--------------------------------------------------------------------------
rel_dist
    Distance measured DOWN FROM THE TIP, in feet. This matches PLS-POLE's
    own `rel_dist` field. rel_dist = 0 at the tip, = total_length at the
    very bottom of the pole (bottom of embedment for a direct-buried pole).

Diameter
    12-sided (dodecagonal) poles only. D is the FLAT-TO-FLAT outside
    diameter. Taper is single and continuous over the whole pole:
        D(rel_dist) = tip_diameter + taper * rel_dist

Tube (segment) layout
    Tubes are numbered 1 = topmost. With laps:
        start_1     = 0
        start_(i+1) = start_i + L_i - lap_i
    which gives the identity  sum(L) - sum(lap) = total_length.
    Verified on all three reference files.

Slip joints -- the UPPER tube is the FEMALE
    Because the pole is tapered, the upper tube's bottom end is wider than
    the lower tube's top end, so the upper tube slips OVER the lower one.
        female_ID_in = D_bot(upper tube) - 2 * t(upper tube)
        lap_ft       = ceil(1.1 * 1.5 * female_ID_ft / 0.25) * 0.25
    i.e. 1.1 x [1.5 x female ID], rounded UP to the next 0.25 ft.
    Rounding is CEILING, not nearest: verified against 5 real joints
    (a "nearest" rule would have produced a lap shorter than required on
    two of them).

Section properties (dodecagonal), evaluated at MID-WALL diameter D - t
    Ag  = 3.22  * D_mid * t
    I   = 0.411 * D_mid^3 * t
    Cx  = 0.518 * (D_mid + t) * cos(15 deg)
    Cy  = 0.518 * (D_mid + t) * sin(15 deg)
    C   = sqrt(Cx^2 + Cy^2)                    (to the polygon VERTEX)
    max Q/(I*t) = 0.631 / (D_mid * t)
    max C/J     = 0.622 * (D_mid + t) / (D_mid^3 * t)
    Using D_mid = D - t (rather than the outer D) matches PLS-POLE's
    reported area and inertia to ~0.14%; using outer D overstates them by
    ~1-2%. This is a calibration finding, not a statement from the source
    formula sheet.

Flat width and local buckling
    w    = 0.268 * (D - t - 2*BR),  BR = bend_radius_factor * t
    w/t  is then checked against the max_wt limit.
    bend_radius_factor defaults to 4.5, which reproduces PLS-POLE's own
    reported w/t exactly on all three reference files. ASCE/SEI 48-19
    Sec 5.2.3.2.1 caps the bend radius at 4t; set the factor to 4.0 to
    apply that cap (about 0.4 points more conservative on usage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Standard plate gauges, 3/16 in to 1 in in 1/16 in steps.
GAUGES: List[float] = [round(0.1875 + 0.0625 * i, 4) for i in range(14)]

STEEL_DENSITY_PCF = 490.0     #: lb/ft^3
E_STEEL_KSI = 29000.0         #: ksi, ASCE 48-19 Sec 5.2.3.2.1
N_SIDES = 12                  #: dodecagonal only
BEND_RADIUS_FACTOR = 4.5      #: BR = factor * t;  4.0 applies the ASCE cap
HALF_ANGLE_DEG = 180.0 / N_SIDES   # 15 deg for a 12-sided polygon


# --------------------------------------------------------------------------
# Design limits
# --------------------------------------------------------------------------

@dataclass
class DesignLimits:
    """Fabrication and design constraints. All user-adjustable."""

    taper_min: float = 0.15               # in/ft
    taper_max: float = 0.50               # in/ft
    max_wt: float = 35.0                  # max w/t ratio
    max_segments: int = 6
    #: Section lengths in order of preference; the last value is the hard max.
    preferred_lengths: tuple = (53.0, 57.0, 60.0)
    #: Minimum height of the BOTTOM of the lowest slip joint above ground
    #: line. The bottom of a lap is the lower end of the female (upper) tube.
    min_base_joint_above_gl_ft: float = 10.0
    max_usage_pct: float = 100.0          # feasibility cutoff
    target_usage_pct: Optional[float] = None   # e.g. 95.0 to design to 95%
    gauges: tuple = tuple(GAUGES)
    bend_radius_factor: float = BEND_RADIUS_FACTOR

    @property
    def max_length(self) -> float:
        return max(self.preferred_lengths)

    @property
    def usage_cutoff(self) -> float:
        """The usage a candidate must not exceed to count as feasible."""
        return self.target_usage_pct if self.target_usage_pct else self.max_usage_pct


# --------------------------------------------------------------------------
# Design spec
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One tube. `length` is the FULL fabricated length, including any lap."""
    length: float                     # ft
    thickness: float                  # in
    fy: float = 65.0                  # ksi
    #: 'slip' or 'flange'. The joint at this tube's BOTTOM end. The lowest
    #: tube has no joint below it, so its value is ignored.
    joint_type: str = 'slip'
    #: Only used when the joint ABOVE this tube is a flange. A flange
    #: connection does not force diameter continuity -- the tubes are
    #: separate pieces bolted together, and the lower tube's top diameter
    #: is chosen to suit the flange. Leave as None to assume continuity.
    #: (Reference file 001 has a flange with a +0.380 in step, which no
    #: derived rule explains, so it must be supplied.)
    top_diameter_override: Optional[float] = None


#: Diametral assembly clearance at a slip joint, inches. The male (lower)
#: tube's outside diameter equals the female (upper) tube's inside diameter
#: at the top of the lap, minus this clearance. Measured as 0.1227-0.1282 in
#: across four real slip joints in two poles, i.e. 1/8 in; the scatter is
#: PLS-POLE reporting diameters to two decimals.
SLIP_CLEARANCE_IN = 0.125


@dataclass
class PoleSpec:
    """A complete candidate pole geometry."""
    tip_diameter: float               # in, flat-to-flat outside
    taper: float                      # in/ft, single and continuous RATE
    segments: List[Segment]
    embedment: float = 0.0            # ft; 0 for a base-plate pole. FIXED input.
    label: str = ''
    slip_clearance: float = SLIP_CLEARANCE_IN

    # ---- derived geometry -------------------------------------------------

    def layout(self) -> List[dict]:
        """Per-tube start/end rel_dist, top/bottom diameter, lap length and
        joint elevations.

        The taper RATE is constant over the whole pole, but the diameter is
        NOT continuous: at each slip joint the male (lower) tube sits inside
        the female (upper) tube, so its outside diameter steps down by
        2*t_female + slip_clearance. Walks top-down in one pass -- the lap
        of a joint depends only on quantities already established above it,
        so no iteration is needed.
        """
        out: List[dict] = []
        start = 0.0
        d_top = self.tip_diameter

        for i, seg in enumerate(self.segments):
            end = start + seg.length
            d_bot = d_top + self.taper * seg.length
            is_last = (i == len(self.segments) - 1)

            if is_last:
                lap, joint = 0.0, 'none'
            elif seg.joint_type == 'slip':
                female_id_ft = (d_bot - 2.0 * seg.thickness) / 12.0
                lap, joint = slip_joint_lap(female_id_ft), 'slip'
            else:
                lap, joint = 0.0, seg.joint_type

            out.append({
                'tube_no': i + 1,
                'start': start,
                'end': end,
                'length': seg.length,
                'lap': lap,
                'thickness': seg.thickness,
                'fy': seg.fy,
                'joint_type': joint,
                'd_top': d_top,
                'd_bot': d_bot,
                # The lap spans [next tube's start, this tube's end]. Its
                # BOTTOM is this tube's end -- the lower end of the female
                # tube -- and that is the point the minimum-height-above-
                # ground-line rule measures to.
                'lap_top': end - lap if lap else None,
                'lap_bottom': end if lap else None,
            })

            if not is_last:
                nxt = self.segments[i + 1]
                next_start = end - lap
                if nxt.top_diameter_override is not None:
                    d_top = nxt.top_diameter_override
                elif joint == 'slip':
                    # Diameter of the female tube at the top of the lap,
                    # less its two wall thicknesses, less the clearance.
                    d_at_lap_top = d_top + self.taper * (next_start - start)
                    d_top = (d_at_lap_top - 2.0 * seg.thickness
                             - self.slip_clearance)
                else:
                    # Flange/butt with no override: assume continuity.
                    d_top = d_bot
                start = next_start

        return out

    def laps(self) -> List[float]:
        return [tb['lap'] for tb in self.layout()]

    @property
    def total_length(self) -> float:
        """sum(L) - sum(lap). Verified identity on all reference files."""
        return sum(s.length for s in self.segments) - sum(self.laps())

    @property
    def agl_height(self) -> float:
        """Above-ground height -- the basis for the deflection limit."""
        return self.total_length - self.embedment

    @property
    def groundline_rel(self) -> float:
        """rel_dist at the ground line."""
        return self.total_length - self.embedment

    @property
    def base_diameter(self) -> float:
        return self.layout()[-1]['d_bot']

    def diameter_at(self, rel_dist: float, tol: float = 1e-6) -> float:
        """Outside flat-to-flat diameter at a distance below the tip.

        Inside a lap two tubes are present; the female (upper) tube is on
        the outside, so it governs the outside diameter.
        """
        lay = self.layout()
        hits = [tb for tb in lay
                if tb['start'] - tol <= rel_dist <= tb['end'] + tol]
        if not hits:
            tb = min(lay, key=lambda t: min(abs(t['start'] - rel_dist),
                                            abs(t['end'] - rel_dist)))
        else:
            tb = min(hits, key=lambda t: t['tube_no'])
        return tb['d_top'] + self.taper * (rel_dist - tb['start'])

    def thickness_at(self, rel_dist: float, tol: float = 1e-6) -> float:
        """Governing wall thickness -- the female (upper) tube inside a lap,
        for consistency with diameter_at()."""
        lay = self.layout()
        hits = [tb for tb in lay
                if tb['start'] - tol <= rel_dist <= tb['end'] + tol]
        if not hits:
            return min(lay, key=lambda t: min(abs(t['start'] - rel_dist),
                                              abs(t['end'] - rel_dist)))['thickness']
        return min(hits, key=lambda t: t['tube_no'])['thickness']


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def slip_joint_lap(female_id_ft: float) -> float:
    """Design slip joint length = 1.1 x [1.5 x female ID], rounded UP to
    the next 0.25 ft. Validated against 5 real joints."""
    if female_id_ft <= 0:
        return 0.0
    return math.ceil(1.1 * 1.5 * female_id_ft / 0.25) * 0.25


def snap_to_gauge(t: float, gauges=tuple(GAUGES), direction: str = 'up') -> float:
    """Snap a wall thickness to an available plate gauge.
    direction='up' never reduces capacity; 'nearest' may."""
    if direction == 'nearest':
        return min(gauges, key=lambda g: abs(g - t))
    above = [g for g in gauges if g >= t - 1e-9]
    return min(above) if above else max(gauges)


def flat_width(D: float, t: float, bend_radius_factor: float = BEND_RADIUS_FACTOR) -> float:
    """w = 0.268 * (D - t - 2*BR), BR = factor * t."""
    return 0.268 * (D - t - 2.0 * bend_radius_factor * t)


def w_over_t(D: float, t: float, bend_radius_factor: float = BEND_RADIUS_FACTOR) -> float:
    return flat_width(D, t, bend_radius_factor) / t


def thickness_from_wt(D: float, wt: float,
                      bend_radius_factor: float = BEND_RADIUS_FACTOR) -> float:
    """Invert w/t to recover t. Useful for reading a PLS-POLE section table,
    where each row reports D and w/t but not thickness -- and where mapping
    thickness by elevation picks the wrong wall inside a lap."""
    return 0.268 * D / (wt + 0.268 * (1.0 + 2.0 * bend_radius_factor))


def dodecagon_section_properties(D: float, t: float) -> dict:
    """Section properties of a 12-sided tube, evaluated at mid-wall
    diameter D - t. See the module docstring for provenance."""
    d_mid = D - t
    a = math.radians(HALF_ANGLE_DEG)
    ag = 3.22 * d_mid * t
    i = 0.411 * d_mid ** 3 * t
    cx = 0.518 * (d_mid + t) * math.cos(a)
    cy = 0.518 * (d_mid + t) * math.sin(a)
    return {
        'D': D, 't': t, 'D_mid': d_mid,
        'Ag': ag,
        'I': i,
        'r': 0.358 * d_mid,
        'Cx': cx, 'Cy': cy,
        'C': math.sqrt(cx ** 2 + cy ** 2),
        'S': i / math.sqrt(cx ** 2 + cy ** 2),
        'Q_over_It_max': 0.631 / (d_mid * t),
        'C_over_J_max': 0.622 * (d_mid + t) / (d_mid ** 3 * t),
    }


# --------------------------------------------------------------------------
# Section table
# --------------------------------------------------------------------------

def build_sections(spec: PoleSpec,
                   spacing: float = 5.0,
                   rel_dists: Optional[List[float]] = None,
                   bend_radius_factor: float = BEND_RADIUS_FACTOR) -> List[dict]:
    """Build the section table a strength check runs over.

    spacing     nominal spacing in ft; tube ends, lap boundaries and the
                ground line are always included as section points.
    rel_dists   supply an explicit list to evaluate at given locations
                (used to compare directly against a PLS-POLE section table).
    """
    lay = spec.layout()
    total = spec.total_length

    if rel_dists is None:
        pts = set()
        n = max(1, int(math.ceil(total / spacing)))
        for k in range(n + 1):
            pts.add(round(min(k * spacing, total), 4))
        for tb in lay:
            pts.add(round(tb['start'], 4))
            pts.add(round(min(tb['end'], total), 4))
            if tb['lap_top'] is not None:
                pts.add(round(tb['lap_top'], 4))
                pts.add(round(tb['lap_bottom'], 4))
        if spec.embedment:
            pts.add(round(spec.groundline_rel, 4))
        rel_dists = sorted(pts)

    out = []
    for rd in rel_dists:
        D = spec.diameter_at(rd)
        t = spec.thickness_at(rd)
        props = dodecagon_section_properties(D, t)
        wt = w_over_t(D, t, bend_radius_factor)
        tube = next((tb['tube_no'] for tb in lay
                     if tb['start'] - 1e-6 <= rd <= tb['end'] + 1e-6), None)
        in_lap = sum(1 for tb in lay
                     if tb['start'] - 1e-6 <= rd <= tb['end'] + 1e-6) > 1
        out.append({
            'rel_dist': rd,
            'height_agl': spec.groundline_rel - rd,
            'below_groundline': rd > spec.groundline_rel + 1e-6,
            'tube_no': tube,
            'in_lap': in_lap,
            'D': D,
            't': t,
            'fy': next((tb['fy'] for tb in lay if tb['tube_no'] == tube), 65.0),
            'w_over_t': wt,
            **{k: props[k] for k in ('Ag', 'I', 'C', 'S',
                                     'Q_over_It_max', 'C_over_J_max')},
        })
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(spec: PoleSpec, limits: DesignLimits = DesignLimits(),
             target_length: Optional[float] = None) -> List[str]:
    """Return a list of constraint violations. Empty list means the geometry
    is buildable. This checks GEOMETRY only -- strength and deflection are
    the mechanics engine's job."""
    v: List[str] = []
    lay = spec.layout()

    if not (limits.taper_min - 1e-9 <= spec.taper <= limits.taper_max + 1e-9):
        v.append(f"taper {spec.taper:.4f} in/ft outside "
                 f"[{limits.taper_min}, {limits.taper_max}]")

    if len(spec.segments) > limits.max_segments:
        v.append(f"{len(spec.segments)} segments exceeds max {limits.max_segments}")

    for tb in lay:
        if tb['length'] > limits.max_length + 1e-9:
            v.append(f"tube {tb['tube_no']} length {tb['length']:.2f} ft "
                     f"exceeds max {limits.max_length} ft")

    for seg in spec.segments:
        if not any(abs(seg.thickness - g) < 1e-9 for g in limits.gauges):
            v.append(f"thickness {seg.thickness} in is not an available gauge")

    for s in build_sections(spec, bend_radius_factor=limits.bend_radius_factor):
        if s['w_over_t'] > limits.max_wt + 1e-9:
            v.append(f"w/t {s['w_over_t']:.2f} at rel_dist {s['rel_dist']:.2f} ft "
                     f"exceeds max {limits.max_wt}")
            break

    # Lowest slip joint clearance above ground line.
    slips = [tb for tb in lay if tb['lap_bottom'] is not None]
    if slips and spec.embedment:
        lowest = max(slips, key=lambda tb: tb['lap_bottom'])
        clear = spec.groundline_rel - lowest['lap_bottom']
        if clear < limits.min_base_joint_above_gl_ft - 1e-9:
            v.append(f"bottom of lowest slip joint is {clear:.2f} ft above ground "
                     f"line, below the {limits.min_base_joint_above_gl_ft} ft minimum "
                     f"(needs base tube >= embedment + minimum + lap = "
                     f"{spec.embedment + limits.min_base_joint_above_gl_ft + lowest['lap']:.2f} ft)")

    if target_length is not None and abs(spec.total_length - target_length) > 0.01:
        v.append(f"total length {spec.total_length:.2f} ft does not match "
                 f"target {target_length:.2f} ft")

    return v


def length_preference_rank(length: float, limits: DesignLimits = DesignLimits()) -> int:
    """0 = most preferred. Used to break ties between equal-weight candidates."""
    for i, pref in enumerate(limits.preferred_lengths):
        if abs(length - pref) < 0.01:
            return i
    return len(limits.preferred_lengths)


def describe(spec: PoleSpec) -> str:
    """Human-readable summary for logs and reports."""
    lines = [
        f"{spec.label or 'pole'}: {spec.total_length:.2f} ft total"
        + (f" ({spec.agl_height:.2f} ft AGL + {spec.embedment:.2f} ft embedded)"
           if spec.embedment else " (base plate)"),
        f"  {N_SIDES}-sided, D {spec.tip_diameter:.2f} -> {spec.base_diameter:.2f} in, "
        f"taper {spec.taper:.5f} in/ft",
    ]
    for tb in spec.layout():
        j = tb['joint_type']
        lap = f", lap {tb['lap']:.2f} ft" if tb['lap'] else ""
        lines.append(
            f"  tube {tb['tube_no']}: rel {tb['start']:7.2f}-{tb['end']:7.2f} ft  "
            f"L={tb['length']:6.2f}  t={tb['thickness']:.4f}  "
            f"D {tb['d_top']:6.2f}->{tb['d_bot']:6.2f}  joint={j}{lap}")
    return "\n".join(lines)
