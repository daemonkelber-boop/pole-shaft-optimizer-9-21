"""
baseplate.py -- Base plate + anchor bolt post-check and sizing for
base-plate poles. 12-sided shafts, round plates, single bolt ring.

PLATE CHECK -- reproduces PLS-POLE (validated, file 017: all 1008
bend-line rows, 84 load cases x 12 bend lines; bolt moment sum within
0.0015 ft-k, usage within 0.005 %, t_min within 0.0005 in):
  * ASCE 48-19 Appendix F wedge method. 12 bend lines on the outside
    faces (D_AF/2 from centre), length = D_AF*tan(15 deg). Face 1 normal
    = +y, faces counter-clockwise.
  * Bolts acting = inside the +/-15 deg wedge through the vertices; a
    bolt exactly on a vertex counts 1/2.
  * Bolt loads, Eq. F-4, rigid plate, leveling nuts, no grout:
        BL_i = P/n + Mx*y_i/sum(y^2) + My*x_i/sum(x^2)   (+ compression)
  * Bend-line moment = sum(w_i * |BL_i| * c_i)  -- tension and compression
    bolts both counted as positive (required to match PLS-POLE on
    mixed-sign bend lines).
  * fb = 6*M/(b*t^2) <= Fy(t);  usage = fb/Fy.
  * ASCE 48-19 6.4.2, as PLS-POLE applies it (confirmed from the row note
    in 017): when the resultant base moment |M| < 0.5*M_cap, (Mx, My) are
    scaled up to 0.5*M_cap in the SAME direction; P is not changed.
    M_cap = base-section moment capacity = Fa*I/(D_AF/2) (matches PLS
    t_moment_capacity, 33,334 vs 33,338 ft-k on 017).
  * P for the check = shaft axial force at the base, EXCLUDING the base
    plate weight (PLS vertical_load = support reaction - plate weight).

USER RULES (Jainesh, from his check workbook)
  * dBC  = D_AP + 2*t_pole + nut point-to-point + 1.0 in, rounded UP to
           0.5 in;  D_AP = D_AF / cos(15 deg); t_pole = wall at the plate.
  * OD   = dBC + 6.0 in.
  * Hole = floor to 0.5 in of: AF>=72: 0.65AF; >=42: 0.70AF; >=33: AF-10;
           >=18: AF-8; else AF-6.
  * Bolt chord spacing >= 2.67 * d_nom.
  * Plate Fy: 50 ksi t <= 4.0 in, 42 ksi t > 4.0 in; t >= 1.5 in, 1/4 in
    steps.  Bolt count: multiples of 4, pattern orientation as the XML.
  * No feasible layout -> the shaft is rejected (decision (a)).

ANCHOR BOLT CHECK -- derived from ASCE 48-19 (no workbook equivalent).
ASSUMPTIONS FLAGGED:
  * 9.3.1 -> 6.2.3 tension: ft = T/As <= Ft = 0.75 Fu, plus user cap
    T_max <= tension_cap (default 220 kips).
  * Compression bolts: ASCE gives no explicit compression rule; the C9-1
    convention (|T|, tension or compression) is used: |C|/As <= 0.75 Fu.
    The 220-kip cap applies to tension AND compression (user). Usage is
    compared to the target, so the effective cap = cap x target (user).
  * 9.3.2 -> 6.2.2 shear: fv = V_b/Ag <= Fv = 0.35 Fu (threads in the shear
    plane, per C9.3.3).  V_b = (V + T_z/r_bc)/n -- shear shared equally
    by all bolts, torsion as tangential force, added as scalars
    (conservative).
  * 9.3.3 -> 6.2.4 combined: ft <= Ft*sqrt(1 - (fv/Fv)^2).
  * C9-1 (bolt bending) NOT applied: leveling-nut gap < 2d (user), so
    9.3.3 / 6.2.4 governs.
  * The 50 % moment override is applied to the bolt loads too (6.4.2
    says "connection"); shear and torsion are actual values.
  * Fu = 100 ksi (ASTM A615 Gr 75 minimum tensile strength).
  * Development length, 9.3.4, Eq. 9.3-5 (No. 18 / 18J):
        Ld = 3.52*Fy/sqrt(f'c) * alpha*beta*gamma,  alpha = 1.2 (Gr 75),
        beta = 0.8 (spacing >= 6 in), gamma = As(req'd)/Ag,
        As(req'd) = max(T_max, C_max)/Ft (ASSUMPTION: compression
        included, conservative);  minimum 25 d (C9.3.4).
        f'c = 3 ksi, projection = 12 in (user defaults).
    Eq. 9.3-5 was read from extracted text (radical garbled) -- verify
    against the page image before production use.
  * Bolt weight = Ag * (Ld + projection) * 490 pcf; nuts/washers excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

N_SIDES = 12
HALF_WEDGE = 180.0 / N_SIDES
RHO_PCF = 490.0
COS15 = math.cos(math.radians(15.0))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass
class BasePlateRules:
    # anchor bolt: A615 Gr 75 #18J (values from the user's workbook)
    bolt_d: float = 2.25            # in, nominal
    bolt_tpi: float = 4.5           # threads per inch
    bolt_Fy: float = 75.0           # ksi
    bolt_Fu: float = 100.0          # ksi (ASTM A615 Gr 75)
    nut_pt_to_pt: float = 4.063     # in
    tension_cap: float = 220.0      # kips, user limit on max bolt tension AND compression
    spacing_factor: float = 2.67    # min chord spacing = factor * d
    n_step: int = 4                 # bolt counts: multiples of this
    start_deg: float = 0.0          # bolt 1 angle (from the XML)
    # plate
    t_min: float = 1.5
    t_step: float = 0.25
    edge_add: float = 6.0           # OD = dBC + edge_add
    bc_clear: float = 1.0
    bc_round: float = 0.5
    # usage target (%), same as the shaft target
    target: float = 100.0
    # development length / bolt weight -- REQUIRED for bolt weight
    fc_ksi: Optional[float] = 3.0        # 3000 psi (user)
    projection_in: Optional[float] = 12.0  # in (user)
    # 6.4.2
    apply_half_capacity: bool = True

    @property
    def Ag(self) -> float:
        return math.pi / 4.0 * self.bolt_d ** 2

    @property
    def As(self) -> float:          # ASCE 48-19 Eq. 6.2-3
        return math.pi / 4.0 * (self.bolt_d - 0.9743 / self.bolt_tpi) ** 2

    @property
    def min_spacing(self) -> float:
        return self.spacing_factor * self.bolt_d


def plate_fy(t: float) -> float:
    return 50.0 if t <= 4.0 + 1e-9 else 42.0


def _ceil_to(x, inc):
    return math.ceil(x / inc - 1e-9) * inc


def _floor_to(x, inc):
    return math.floor(x / inc + 1e-9) * inc


def bolt_circle(D_af: float, t_pole: float, R: BasePlateRules) -> float:
    return _ceil_to(D_af / COS15 + 2 * t_pole + R.nut_pt_to_pt + R.bc_clear, R.bc_round)


def hole_diameter(D_af: float) -> float:
    if D_af >= 72: v = 0.65 * D_af
    elif D_af >= 42: v = 0.70 * D_af
    elif D_af >= 33: v = D_af - 10
    elif D_af >= 18: v = D_af - 8
    else: v = D_af - 6
    return _floor_to(v, 0.5)


def base_moment_capacity(D_af: float, t: float, Fy: float,
                         bend_radius_factor: float = 4.5) -> float:
    """Fa * I / (D_AF/2), ft-k. Sharp-corner 12-gon I (matches PLS I)."""
    from geometry import w_over_t
    from mechanics_engine import local_buckling_Fa_dodecagonal
    T = math.tan(math.pi / N_SIDES)
    Ip = lambda a: N_SIDES * a ** 4 * T / 4.0 * (1 + T * T / 3.0)
    I = Ip(D_af / 2.0) - Ip(D_af / 2.0 - t)
    Fa, _ = local_buckling_Fa_dodecagonal(w_over_t(D_af, t, bend_radius_factor), Fy)
    if Fa is None:
        raise ValueError("base section w/t outside Eq. 5.2-9 range")
    return Fa * I / (D_af / 2.0) / 12.0


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------

def bolt_angles(n: int, R: BasePlateRules) -> np.ndarray:
    return R.start_deg + np.arange(n) * 360.0 / n


def bolt_loads(n: int, bc: float, P, Mx_ftk, My_ftk, R: BasePlateRules) -> np.ndarray:
    a = np.radians(bolt_angles(n, R)); r = bc / 2.0
    x, y = r * np.cos(a), r * np.sin(a)
    P, Mx, My = (np.atleast_1d(np.asarray(v, float)) for v in (P, Mx_ftk, My_ftk))
    return (P[:, None] / n + 12.0 * Mx[:, None] * y / np.sum(y ** 2)
            + 12.0 * My[:, None] * x / np.sum(x ** 2))


def wedge_matrix(n: int, bc: float, D_af: float, R: BasePlateRules) -> np.ndarray:
    ang = bolt_angles(n, R); r = bc / 2.0
    W = np.zeros((N_SIDES, n))
    for k in range(N_SIDES):
        da = (ang - (90.0 + 30.0 * k) + 180.0) % 360.0 - 180.0
        vert = np.isclose(np.abs(da), HALF_WEDGE, atol=1e-6)
        inside = (np.abs(da) < HALF_WEDGE - 1e-6) | vert
        c = r * np.cos(np.radians(da)) - D_af / 2.0
        if np.any(inside & (c <= 0)):
            raise ValueError("bolt inside the shaft face")
        W[k] = np.where(inside, np.where(vert, 0.5, 1.0) * c, 0.0)
    return W


def half_capacity_override(Mx, My, M_cap):
    """6.4.2 as PLS-POLE: raise |M| to 0.5*M_cap, same direction."""
    Mx, My = np.asarray(Mx, float).copy(), np.asarray(My, float).copy()
    M = np.hypot(Mx, My); half = 0.5 * M_cap
    low = M < half
    f = np.where(low & (M > 1e-9), half / np.where(M > 1e-9, M, 1.0), 1.0)
    Mx, My = Mx * f, My * f
    Mx = np.where(low & (M <= 1e-9), half, Mx)   # zero moment: put it on x (flag)
    return Mx, My, low


def plate_check(n, bc, D_af, t, P, Mx, My, R: BasePlateRules):
    BL = bolt_loads(n, bc, P, Mx, My, R)
    Msum = np.abs(BL) @ wedge_matrix(n, bc, D_af, R).T          # in-k
    b = D_af * math.tan(math.radians(HALF_WEDGE))
    usage = 6.0 * Msum / (b * t * t) / plate_fy(t) * 100.0
    return usage, Msum, BL, b


def anchor_bolt_check(BL, V, Tz_ftk, n, bc, R: BasePlateRules) -> dict:
    """BL (m,n) + compression. V kips, Tz ft-k per load case."""
    tens = np.maximum(-BL, 0.0).max(axis=1)        # max tension per case
    comp = np.maximum(BL, 0.0).max(axis=1)
    Ft, Fv = 0.75 * R.bolt_Fu, 0.35 * R.bolt_Fu
    Vb = (np.asarray(V) + np.abs(np.asarray(Tz_ftk)) * 12.0 / (bc / 2.0)) / n
    fv = Vb / R.Ag
    red = np.sqrt(np.clip(1.0 - (fv / Fv) ** 2, 0.0, None))
    u_shear = fv / Fv * 100
    u_tens = tens / R.As / np.where(red > 0, Ft * red, 1e-12) * 100
    u_comp = comp / R.As / np.where(red > 0, Ft * red, 1e-12) * 100
    u_cap = np.maximum(tens, comp) / R.tension_cap * 100   # cap on |axial|, scaled by target via usage
    u = np.maximum.reduce([u_shear, u_tens, u_comp, u_cap])
    i = int(np.argmax(u))
    return dict(usage=float(u[i]), case_idx=i,
                T_max=float(tens.max()), C_max=float(comp.max()),
                Vb_max=float(Vb.max()),
                u_tension=float(u_tens.max()), u_compression=float(u_comp.max()),
                u_shear=float(u_shear.max()), u_cap=float(u_cap.max()))


def bolt_length(T_max: float, R: BasePlateRules, spacing: float) -> Optional[float]:
    if R.fc_ksi is None or R.projection_in is None:
        return None
    ld = 3.52 * R.bolt_Fy / math.sqrt(R.fc_ksi)
    alpha = 1.2 if R.bolt_Fy >= 75 else 1.0
    beta = 0.8 if spacing >= 6.0 else 1.0
    gamma = min(T_max / (0.75 * R.bolt_Fu) / R.Ag, 1.0)
    Ld = max(ld * alpha * beta * gamma, 25.0 * R.bolt_d)
    return Ld + R.projection_in


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def size_baseplate(D_af: float, t_pole: float, Fy_pole: float, reactions: dict,
                   R: BasePlateRules, bend_radius_factor: float = 4.5) -> dict:
    """reactions: names, P (+comp, kips, plate weight excluded), Mx, My,
    V, T (ft-k) in PLS-POLE sign convention. Returns dict with 'best'
    (None -> shaft rejected) and the full feasible table."""
    bc = bolt_circle(D_af, t_pole, R)
    od = bc + R.edge_add
    hole = hole_diameter(D_af)
    Mcap = base_moment_capacity(D_af, t_pole, Fy_pole, bend_radius_factor)
    P = np.asarray(reactions['P'], float)
    Mx, My = np.asarray(reactions['Mx'], float), np.asarray(reactions['My'], float)
    n_over = 0
    if R.apply_half_capacity:
        Mx, My, low = half_capacity_override(Mx, My, Mcap)
        n_over = int(low.sum())
    n_max = int(math.floor(math.pi / math.asin(min(R.min_spacing / bc, 1.0)) + 1e-9))
    b = D_af * math.tan(math.radians(HALF_WEDGE))
    rows, rejected = [], []
    for n in range(R.n_step, n_max + 1, R.n_step):
        spacing = bc * math.sin(math.pi / n)
        BL = bolt_loads(n, bc, P, Mx, My, R)
        ab = anchor_bolt_check(BL, reactions['V'], reactions['T'], n, bc, R)
        if ab['usage'] > R.target + 1e-9:
            rejected.append((n, f"anchor bolts {ab['usage']:.1f}%")); continue
        Msum = (np.abs(BL) @ wedge_matrix(n, bc, D_af, R).T)
        Mg = float(Msum.max())
        t = max(R.t_min, _ceil_to(math.sqrt(6 * Mg / (b * 50.0 * R.target / 100)), R.t_step))
        if t > 4.0 + 1e-9:
            t = max(4.0 + R.t_step, _ceil_to(math.sqrt(6 * Mg / (b * 42.0 * R.target / 100)), R.t_step))
        u = 6 * Mg / (b * t * t) / plate_fy(t) * 100
        i, k = np.unravel_index(int(np.argmax(Msum)), Msum.shape)
        plate_wt = math.pi / 4 * (od ** 2 - hole ** 2 - n * R.bolt_d ** 2) * t * RHO_PCF / 1728
        L = bolt_length(max(ab['T_max'], ab['C_max']), R, spacing)
        bolt_wt = None if L is None else n * R.Ag * L * RHO_PCF / 1728
        rows.append(dict(n=n, t=t, Fy=plate_fy(t), plate_usage=u,
                         plate_case=reactions['names'][i], bend_line=int(k) + 1,
                         bolt_usage=ab['usage'], T_max=ab['T_max'], C_max=ab['C_max'],
                         u_tension=ab['u_tension'], u_compression=ab['u_compression'],
                         u_shear=ab['u_shear'], u_cap=ab['u_cap'],
                         bolt_case=reactions['names'][ab['case_idx']],
                         spacing=spacing, bolt_len=L, plate_wt=plate_wt, bolt_wt=bolt_wt,
                         total_wt=None if bolt_wt is None else plate_wt + bolt_wt))
    if rows and rows[0]['total_wt'] is not None:
        best = min(rows, key=lambda r: r['total_wt'])
        basis = 'plate + anchor bolt weight'
    elif rows:
        best = min(rows, key=lambda r: (r['n'], r['plate_wt']))
        basis = 'fewest bolts, then plate weight (bolt weight unavailable: f\'c / projection not set)'
    else:
        best, basis = None, 'no feasible layout -> shaft rejected'
    return dict(best=best, basis=basis, table=rows, rejected=rejected,
                bc=bc, od=od, hole=hole, n_max=n_max, M_cap=Mcap,
                n_overridden=n_over, D_af=D_af, t_pole=t_pole)


def check_existing(D_af, t_pole, Fy_pole, reactions, R: BasePlateRules,
                   n: int, t: float, bc: Optional[float] = None,
                   bend_radius_factor: float = 4.5) -> dict:
    """Check a given design (e.g. the PLS-POLE baseline)."""
    bc = bc or bolt_circle(D_af, t_pole, R)
    Mcap = base_moment_capacity(D_af, t_pole, Fy_pole, bend_radius_factor)
    Mx, My = np.asarray(reactions['Mx'], float), np.asarray(reactions['My'], float)
    if R.apply_half_capacity:
        Mx, My, _ = half_capacity_override(Mx, My, Mcap)
    usage, Msum, BL, b = plate_check(n, bc, D_af, t, reactions['P'], Mx, My, R)
    ab = anchor_bolt_check(BL, reactions['V'], reactions['T'], n, bc, R)
    i, k = np.unravel_index(int(np.argmax(usage)), usage.shape)
    return dict(plate_usage=float(usage[i, k]), plate_case=reactions['names'][i],
                bend_line=int(k) + 1, bolt=ab, bc=bc, M_cap=Mcap,
                usage_by_case=usage.max(axis=1))


def base_reactions(case_results: Dict) -> dict:
    """Base reactions from strength.CandidateResult.cases, converted to the
    PLS-POLE convention (solver My has the opposite sign -- verified on 017)."""
    names, P, Mx, My, V, T = [], [], [], [], [], []
    for name, cr in case_results.items():
        r = cr.defl
        f = r.section_forces(float(r.s[-1]))
        names.append(name); P.append(f['P']); Mx.append(f['Mx']); My.append(-f['My'])
        V.append(math.hypot(f['Vx'], f['Vy'])); T.append(f['T'])
    return dict(names=names, P=np.array(P), Mx=np.array(Mx), My=np.array(My),
                V=np.array(V), T=np.array(T))


def workbook_rules_check(D_af: float, t_pole: float, n: int, bc: float, od: float,
                         hole: float, R: BasePlateRules = BasePlateRules()) -> List[dict]:
    """Base plate geometry rules exactly as the user's check workbook
    (Check!I1:O21). Bolt count: one value (designed n) for both the spacing
    calc and the design, per workbook note 1. dBC / OD / hole are
    target-size rules: exact equality, larger OR smaller = NO GOOD."""
    bc_c = bolt_circle(D_af, t_pole, R)
    od_c = bc_c + R.edge_add
    hole_c = hole_diameter(D_af)
    sp = 2 * (bc / 2) * math.sin(math.pi / n)
    eq = lambda a, b: 'OKAY' if abs(a - b) < 1e-6 else 'NO GOOD'
    return [
        dict(check='Bolt spacing', calculated=round(sp, 3), required=round(R.min_spacing, 3),
             status='OKAY' if sp >= R.min_spacing - 1e-9 else 'NO GOOD'),
        dict(check='Bolt-circle diameter', calculated=bc_c, required=bc, status=eq(bc_c, bc)),
        dict(check='Plate diameter', calculated=od_c, required=od, status=eq(od_c, od)),
        dict(check='Hole diameter', calculated=hole_c, required=hole, status=eq(hole_c, hole)),
    ]


def option_detail(D_af: float, t_pole: float, Fy_pole: float, reactions: dict,
                  R: BasePlateRules, n: int, t: float,
                  bend_radius_factor: float = 4.5) -> List[dict]:
    """Per-load-case breakdown of one plate/bolt option (rule dBC).
    Moments shown are the values used in the check (after the 6.4.2
    override); P is the shaft axial at the base (plate weight excluded)."""
    bc = bolt_circle(D_af, t_pole, R)
    Mcap = base_moment_capacity(D_af, t_pole, Fy_pole, bend_radius_factor)
    Mx0, My0 = np.asarray(reactions['Mx'], float), np.asarray(reactions['My'], float)
    Mx, My, low = (half_capacity_override(Mx0, My0, Mcap) if R.apply_half_capacity
                   else (Mx0, My0, np.zeros(len(Mx0), bool)))
    usage, Msum, BL, b = plate_check(n, bc, D_af, t, reactions['P'], Mx, My, R)
    Ft, Fv = 0.75 * R.bolt_Fu, 0.35 * R.bolt_Fu
    out = []
    for i, name in enumerate(reactions['names']):
        k = int(np.argmax(usage[i]))
        T = max(-BL[i].min(), 0.0); C = max(BL[i].max(), 0.0)
        Vb = (reactions['V'][i] + abs(reactions['T'][i]) * 12.0 / (bc / 2.0)) / n
        fv = Vb / R.Ag
        red = math.sqrt(max(1.0 - (fv / Fv) ** 2, 0.0))
        u_ax = max(T, C) / R.As / (Ft * red) * 100 if red > 0 else float('inf')
        u_cap = max(T, C) / R.tension_cap * 100
        out.append({
            "load case": name,
            "P base (k)": round(float(reactions['P'][i]), 1),
            "|M| actual (ft-k)": round(float(math.hypot(Mx0[i], My0[i])), 0),
            "|M| used (ft-k)": round(float(math.hypot(Mx[i], My[i])), 0),
            "6.4.2 override": "yes" if low[i] else "",
            "plate usage %": round(float(usage[i, k]), 2),
            "bend line": k + 1,
            "max bolt T (k)": round(T, 1), "max bolt C (k)": round(C, 1),
            "bolt shear (k)": round(Vb, 2),
            "bolt stress usage %": round(u_ax, 2),
            "bolt cap usage %": round(u_cap, 2),
        })
    return out
