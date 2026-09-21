"""
strength.py  --  Phase 3, module 3

Full evaluation of one candidate pole against every load case:
second-order forces from deflection.py, ASCE 48-19 strength at section
points along the shaft (mechanics_engine.perimeter_stress_check), and the
deflection limit from the load case table.

Which tube is checked where (reproduces PLS-POLE, verified on 003)
    Inside a slip-joint lap PLS-POLE reports the FEMALE (upper) tube down to
    the top of the lap and the MALE (lower) tube from the bottom of the lap
    (the female's lower end). Sections strictly inside the lap are not
    checked. Each tube is therefore checked over
        [ max(start_i, end_(i-1)),  min(end_i, start_(i+1)) ].
Sections below the ground line are not checked (embedded poles).
At every attachment elevation both sides are checked (load excluded / load
included), matching PLS-POLE's 'End' and 'Origin' rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from geometry import PoleSpec, w_over_t, BEND_RADIUS_FACTOR
from loads import Baseline, build_load_model
from deflection import solve_deflection, deflection_check, DeflectionResult
from mechanics_engine import perimeter_stress_check


def strength_points(spec: PoleSpec, attach_s: List[float],
                    spacing: float = 1.0) -> List[dict]:
    """Section points: (s, tube dict, include_loads_at_s)."""
    lay = spec.layout()
    gl = spec.groundline_rel
    pts = []
    for i, tb in enumerate(lay):
        lo = max(tb['start'], lay[i - 1]['end'] if i > 0 else tb['start'])
        hi = min(tb['end'], lay[i + 1]['start'] if i + 1 < len(lay) else tb['end'])
        hi = min(hi, gl)
        if hi < lo - 1e-9:
            continue
        ss = set(np.round(np.arange(lo, hi + 1e-9, spacing), 4).tolist())
        ss.update([round(lo, 4), round(hi, 4)])
        att = [a for a in attach_s if lo - 1e-9 <= a <= hi + 1e-9]
        for s in sorted(ss):
            pts.append(dict(s=s, tube=tb, include=True))
        for a in att:      # both sides of every attachment
            pts.append(dict(s=a, tube=tb, include=False))
            pts.append(dict(s=a, tube=tb, include=True))
    return pts


def check_case_strength(spec: PoleSpec, res: DeflectionResult,
                        points: List[dict],
                        bend_radius_factor: float = BEND_RADIUS_FACTOR,
                        shear_mode: str = 'resultant') -> List[dict]:
    rows = []
    for p in points:
        s, tb = p['s'], p['tube']
        D = tb['d_top'] + spec.taper * (s - tb['start'])
        t = tb['thickness']
        wt = w_over_t(D, t, bend_radius_factor)
        f = res.section_forces(s, p['include'])
        c = perimeter_stress_check(f['axial_local'], f['Mx'], f['My'],
                                   f['Vy_local'], f['Vx_local'], f['T'],
                                   D, t, tb['fy'], wt, shear_mode)
        rows.append(dict(s=s, height_agl=spec.groundline_rel - s,
                         tube_no=tb['tube_no'], side='below' if p['include'] else 'above',
                         D=D, t=t, fy=tb['fy'], w_over_t=wt,
                         P=f['axial_local'], Mx=f['Mx'], My=f['My'],
                         V=float(np.hypot(f['Vy_local'], f['Vx_local'])), T=f['T'],
                         **{k: c[k] for k in ('p_a', 'm_s', 'v_q', 't_r', 'res',
                                              'Fa', 'Fa_equation', 'usage')}))
    return rows


@dataclass
class CaseResult:
    case: str
    defl: DeflectionResult
    rows: List[dict]
    max_strength: float
    gov_row: dict
    defl_check: Optional[dict]


@dataclass
class CandidateResult:
    spec: PoleSpec
    cases: Dict[str, CaseResult]
    max_strength: float
    gov_strength_case: str
    max_defl_usage: Optional[float]
    gov_defl_case: Optional[str]

    @property
    def governing_usage(self) -> float:
        return max(self.max_strength, self.max_defl_usage or 0.0)

    def passes(self, strength_limit: float = 100.0,
               defl_limit: float = 100.0) -> bool:
        return (self.max_strength <= strength_limit + 1e-9 and
                (self.max_defl_usage or 0.0) <= defl_limit + 1e-9)


def evaluate_candidate(spec: PoleSpec, base: Baseline,
                       cases: Optional[List[str]] = None,
                       bend_radius_factor: float = BEND_RADIUS_FACTOR,
                       shear_mode: str = 'resultant',
                       lap_stiffness: str = 'outer',
                       spacing: float = 1.0, ds: float = 0.25,
                       keep_rows: bool = True) -> CandidateResult:
    cases = cases or list(base.load_cases.keys())
    attach_s = sorted({a.s_pole for a in base.attach.values()})
    points = strength_points(spec, attach_s, spacing)
    out: Dict[str, CaseResult] = {}
    for c in cases:
        m = build_load_model(spec, base, c, ds=ds)
        r = solve_deflection(m, ds=ds, lap_stiffness=lap_stiffness)
        rows = check_case_strength(spec, r, points, bend_radius_factor, shear_mode)
        valid = [x for x in rows if x['usage'] is not None]
        gov = (max(valid, key=lambda x: x['usage']) if valid
               else dict(usage=float('inf')))
        # w/t outside the local-buckling equation range -> infeasible
        if len(valid) < len(rows):
            gov = dict(usage=float('inf'), note='w/t outside Eq 5.2-9 range')
        lc = base.load_cases[c]
        dc = deflection_check(r, lc.defl_check, lc.defl_limit)
        out[c] = CaseResult(c, r, rows if keep_rows else [], gov['usage'], gov, dc)

    sc = max(out.values(), key=lambda x: x.max_strength)
    dcs = [x for x in out.values() if x.defl_check and x.defl_check.get('usage') is not None]
    dg = max(dcs, key=lambda x: x.defl_check['usage']) if dcs else None
    return CandidateResult(spec, out, sc.max_strength, sc.case,
                           dg.defl_check['usage'] if dg else None,
                           dg.case if dg else None)
