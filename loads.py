"""
loads.py  --  Phase 3, module 1

Builds the applied load set for a CANDIDATE pole geometry, using the
geometry-independent loading taken from a baseline PLS-POLE XML export.

What changes with the candidate geometry (recomputed here)
    * Shaft wind      = q * Cd * (D_w + 2*t_ice) per unit length
    * Shaft self-wt   = sum of tube areas * density * DLF  (both tubes in a lap)
    * Shaft ice wt    = perimeter(D_w) * t_ice * ice density   (NO DLF)
    * Attachment lever arms: every arm / vang is measured from the pole
      FACE, so its horizontal offset from the pole centreline is D(s)/2 +
      the fixed component offsets.

What is carried over unchanged from the baseline XML
    * Load case definitions (DLF, wind pressures, ice)
    * Loads delivered to the structure at every insulator attachment
      (table 'loads_at_insulator_attachments_for_all_load_cases')
    * Attachment elevations (distance from tip), arm/vang geometry

PLS-POLE conventions reproduced (verified on 003, see test_loads_003.py)
    * Wind pressure is CONSTANT with height (no kz) for these models --
      the pressure in the load case table is used directly.
    * Drag coefficient = pole property 'default_drag_coef' (1.1), no
      Reynolds-number dependence even though Re is reported.
    * Projected width = flat-to-flat outside diameter. Inside a slip-joint
      lap PLS uses the MEAN of the female and male outside diameters.
    * Ice on the shaft: vertical = 12-sided perimeter x t_ice x density
      (no corner term, no dead load factor); wind width = D + 2 t_ice.
    * Self-weight at 490 pcf with DLF applied; both tubes counted in a lap.
    * Axes: x = longitudinal, y = transverse, z = vertical. Fz positive
      DOWNWARD. Forces in kips, lengths in ft unless noted.

Open items (flagged, not resolved)
    * Davit arm self-weight (~0.18 kips total in 003) and arm wind are not
      modelled. Symmetric left/right in 003; residual <= ~0.5 ft-k.
    * Post insulator T/B load split is held at the baseline values.
    * Arm flexibility is ignored (arms treated as rigid offsets).
    * Vangs attached to an arm joint are taken as hanging vertically by
      (vang length + arm tip depth/2), inferred from 003 positions.
    * Azimuth 0 = +y, 180 = -y confirmed. 90/270 assumed +x/-x (unverified).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from geometry import PoleSpec, dodecagon_section_properties, STEEL_DENSITY_PCF
from pls_pole_xml_parser import (parse_pls_pole_xml, get_field, get_single_table,
                                 get_load_case_instances)

PERIM_FACTOR = 12.0 * math.tan(math.radians(15.0))   # 3.2154, perimeter / D


# --------------------------------------------------------------------------
# Baseline data (geometry independent)
# --------------------------------------------------------------------------

@dataclass
class LoadCase:
    name: str
    dlf: float
    q_trans: float          # psf
    q_long: float           # psf
    ice_t: float            # in
    ice_density: float      # pcf
    wind_area_factor: float
    defl_check: str
    defl_limit: float


@dataclass
class AttachPoint:
    """Geometry of one structure attachment point, resolved to the pole."""
    label: str
    s_pole: float           # ft below tip where the chain meets the pole
    extra_horz: float       # ft beyond the pole face, along azimuth
    dz: float               # ft vertical offset (+ up)
    azimuth: float          # deg


@dataclass
class Baseline:
    source: str
    total_length: float
    embedment: float
    cd_pole: float
    load_cases: Dict[str, LoadCase]
    attach: Dict[str, AttachPoint]
    attach_loads: Dict[str, List[Tuple[str, float, float, float]]]

    @classmethod
    def from_xml(cls, path: str) -> 'Baseline':
        p = parse_pls_pole_xml(path)
        g = get_field

        prop = get_single_table(p, 'steel_pole_properties')[0]
        conn = get_single_table(p, 'steel_pole_connectivity')[0]
        cd = g(prop, 'default_drag_coef')
        length = g(prop, 'length')
        emb = 0.0
        if str(prop.get('base_plate', '')).lower() != 'yes':
            emb = g(conn, 'embed_override') or g(prop, 'default_embedded_length') or 0.0

        lcs = {}
        for r in get_single_table(p, 'vector_load_cases'):
            lcs[r['load_case_description']] = LoadCase(
                name=r['load_case_description'],
                dlf=g(r, 'dead_load_factor'),
                q_trans=g(r, 'trans_wind_pressure') or 0.0,
                q_long=g(r, 'longit_wind_pressure') or 0.0,
                ice_t=g(r, 'ice_thick') or 0.0,
                ice_density=g(r, 'ice_density') or 0.0,
                wind_area_factor=g(r, 'wind_area_factor') or 1.0,
                defl_check=r.get('pole_deflection_check') or 'No Limit',
                defl_limit=g(r, 'pole_deflection_limit_or') or 0.0,
            )

        # pole attach labels -> distance from tip
        pole_s = {}
        for inst in p['tables'].get('relative_attachment_labels_for', []):
            for r in inst['rows']:
                pole_s[r['joint_label']] = g(r, 'distance_from_origin_top_joint')
        pole_s.setdefault('P:t', 0.0)

        # davit property intermediate joints
        dprops = {}
        for inst in p['tables'].get('intermediate_joints', []):
            name = inst['titledetail'].split('"')[1] if '"' in inst['titledetail'] else ''
            dprops[name] = {r['joint_label']: (g(r, 'horz_offset'), g(r, 'vert_offset'))
                            for r in inst['rows']}
        dprop_tip_depth = {r['davit_property_label']: g(r, 'tip_diameter_or_depth')
                           for r in get_single_table(p, 'tubular_davit_properties')} \
            if 'tubular_davit_properties' in p['tables'] else {}

        davits = {r['davit_label']: r for r in
                  get_single_table(p, 'tubular_davit_arm_connectivity')} \
            if 'tubular_davit_arm_connectivity' in p['tables'] else {}

        def resolve(label: str) -> Tuple[float, float, float, float]:
            if label in pole_s:
                return pole_s[label], 0.0, 0.0, 0.0
            davit, _, jt = label.partition(':')
            if davit not in davits:
                warnings.warn(
                    f"resolve(): label '{label}' not found in pole joints "
                    f"or davit table — defaulting to tip (s=0). "
                    f"Check vang/attachment connectivity in the XML."
                )
                return 0.0, 0.0, 0.0, 0.0
            d = davits[davit]
            s, h, dz, az = resolve(d['attach_label'])
            if h == 0.0 and dz == 0.0:
                az = g(d, 'azimuth')
            if jt in ('O', ''):
                return s, h, dz, az
            dh, dv = dprops[d['davit_property_set']][jt]
            return s, h + dh, dz - dv, az

        attach = {}
        for r in (get_single_table(p, 'vang_connectivity')
                  if 'vang_connectivity' in p['tables'] else []):
            s, h, dz, az = resolve(r['attach_label'])
            L = g(r, 'length')
            if r['attach_label'] in pole_s:
                attach[r['vang_label']] = AttachPoint(r['vang_label'], s, h + L, dz,
                                                      g(r, 'azimuth'))
            else:
                prop_set = davits[r['attach_label'].split(':')[0]]['davit_property_set']
                depth = (dprop_tip_depth.get(prop_set) or 0.0) / 12.0
                attach[r['vang_label']] = AttachPoint(r['vang_label'], s, h,
                                                      dz - L - depth / 2.0, az)

        loads = {}
        for r in get_single_table(p, 'loads_at_insulator_attachments_for_all_load_cases'):
            loads.setdefault(r['load_case'], []).append((
                r['structure_attach_label'],
                g(r, 'structure_attach_load_x'),
                g(r, 'structure_attach_load_y'),
                g(r, 'structure_attach_load_z')))
        for lab in {a for v in loads.values() for a, *_ in v}:
            if lab not in attach:
                s, h, dz, az = resolve(lab)
                attach[lab] = AttachPoint(lab, s, h, dz, az)

        return cls(source=path, total_length=length, embedment=emb, cd_pole=cd,
                   load_cases=lcs, attach=attach, attach_loads=loads)


# --------------------------------------------------------------------------
# Candidate load model
# --------------------------------------------------------------------------

@dataclass
class Element:
    s_top: float
    s_bot: float
    D_wind: float
    D_out: float
    area: float
    tubes: List[Tuple[float, float]]
    fy: float
    fx: float
    fz: float
    above_ground: bool

    @property
    def s_mid(self) -> float:
        return 0.5 * (self.s_top + self.s_bot)


@dataclass
class PointLoad:
    label: str
    s: float
    dx: float
    dy: float
    dz: float
    Fx: float
    Fy: float
    Fz: float


@dataclass
class LoadModel:
    spec: PoleSpec
    case: LoadCase
    total_length: float
    elements: List[Element]
    points: List[PointLoad]

    def z_of(self, s: float) -> float:
        return self.total_length - self.spec.embedment - s


def tubes_at(spec: PoleSpec, s: float, tol: float = 1e-9) -> List[Tuple[float, float]]:
    out = []
    for tb in spec.layout():
        if tb['start'] - tol <= s <= tb['end'] + tol:
            out.append((tb['d_top'] + spec.taper * (s - tb['start']), tb['thickness']))
    return out


def build_load_model(spec: PoleSpec, base: Baseline, case_name: str,
                     ds: float = 0.25) -> LoadModel:
    lc = base.load_cases[case_name]
    H = spec.total_length
    gl = spec.groundline_rel
    n = max(1, int(round(H / ds)))
    ds = H / n
    elems = []
    for i in range(n):
        s0, s1 = i * ds, (i + 1) * ds
        sm = 0.5 * (s0 + s1)
        tb = tubes_at(spec, sm)
        D_w = sum(d for d, _ in tb) / len(tb)
        D_o = max(d for d, _ in tb)
        area = sum(dodecagon_section_properties(d, t)['Ag'] for d, t in tb)
        above = sm < gl
        wt = area / 144.0 * STEEL_DENSITY_PCF * ds * lc.dlf / 1000.0
        ice = (PERIM_FACTOR * D_w * lc.ice_t / 144.0 * lc.ice_density * ds / 1000.0
               if (lc.ice_t and above) else 0.0)
        width_ft = (D_w + 2.0 * lc.ice_t) / 12.0
        k = base.cd_pole * lc.wind_area_factor * width_ft * ds / 1000.0
        fy = lc.q_trans * k if above else 0.0
        fx = lc.q_long * k if above else 0.0
        elems.append(Element(s0, s1, D_w, D_o, area, tb, fy, fx, wt + ice, above))

    pts = []
    for lab, Fx, Fy, Fz in base.attach_loads.get(case_name, []):
        a = base.attach[lab]
        r = spec.diameter_at(a.s_pole) / 24.0 + a.extra_horz
        az = math.radians(a.azimuth)
        pts.append(PointLoad(lab, a.s_pole, r * math.sin(az), r * math.cos(az),
                             a.dz, Fx, Fy, Fz))
    return LoadModel(spec, lc, H, elems, pts)


# --------------------------------------------------------------------------
# First-order (undeformed) section forces
# --------------------------------------------------------------------------

def first_order_forces(model: LoadModel, s0: float) -> dict:
    """Resultants of everything ABOVE a section s0 ft below the tip, taken
    about that section in the UNDEFORMED geometry (no P-delta).
    Mx = transverse bending (from y-loads), My = longitudinal bending."""
    z0 = model.z_of(s0)
    P = Vx = Vy = Mx = My = T = 0.0
    for e in model.elements:
        if e.s_bot <= s0 + 1e-9:
            h = model.z_of(e.s_mid) - z0
            P += e.fz; Vx += e.fx; Vy += e.fy
            Mx += e.fy * h; My += e.fx * h
    for p in model.points:
        if p.s <= s0 + 1e-9:
            h = model.z_of(p.s) + p.dz - z0
            P += p.Fz; Vx += p.Fx; Vy += p.Fy
            Mx += p.Fy * h + p.Fz * p.dy
            My += p.Fx * h + p.Fz * p.dx
            T += p.Fx * p.dy - p.Fy * p.dx
    return dict(P=P, Vx=Vx, Vy=Vy, Mx=Mx, My=My, T=T)
