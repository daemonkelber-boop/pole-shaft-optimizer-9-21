"""
optimizer.py  --  Phase 3, module 4

Minimum-weight shaft search for a fixed total height, fixed embedment and
fixed loading (baseline PLS-POLE XML).

Design variables
    * tip and base diameter, each on a user grid (default 0.5 in)
    * number of segments and the fabricated length of every tube except the
      bottom one (the bottom tube takes the remainder)
    * wall thickness of every tube (gauge grid or user list)
Derived (never a free variable)
    * taper = (D_base - D_tip + sum of slip steps) / H,
      slip step = 2 t_female + slip clearance. Flange joints: no step.
      -> taper moves slightly whenever a thickness changes; D_tip and
         D_base stay exactly on the grid.
    * laps, from the lap rule, recomputed for every candidate
    * bottom tube length = H - sum(upper tube lengths) + sum(laps)

Search (all steps logged)
    1. Screening cases: the governing strength cases and the governing
       deflection cases of the baseline design.
    2. Enumerate (D_tip, D_base, layout) combinations on a COARSE grid,
       compute a cheap lower-bound weight for each (thickness from w/t and
       from the baseline moment envelope -- see _lower_bound), sort by it.
    3. In lower-bound order: size thicknesses (thicken the failing tube until
       it passes, then thin every tube as far as it will go), screening cases
       only, coarse analysis mesh. Stop when the lower bound of the remaining
       combinations exceeds the weight of the Kth-best design found.
    4. Refine around the best designs on the FINE grid (diameter increment,
       0.25 ft lengths).
    5. Verify the best designs on ALL load cases, full mesh; thicken on
       failure. Only fully verified designs are ranked.
    6. Section-length rule, then ranking with the tie band.

Pruning heuristic (flagged): the strength lower bound uses 85% of the
baseline second-order moment envelope. A candidate with a smaller diameter
sees less shaft wind, so its moments are lower than the baseline's; the 15%
margin is meant to keep the bound below the true requirement. It can only
cause a combination to be evaluated later, not to be wrongly accepted --
every reported design is fully analysed.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

from geometry import (PoleSpec, Segment, w_over_t, GAUGES, SLIP_CLEARANCE_IN)
from weight import pole_weight
from loads import Baseline
from strength import evaluate_candidate, CandidateResult


# --------------------------------------------------------------------------
# Constraints (every value user-editable)
# --------------------------------------------------------------------------

@dataclass
class OptConstraints:
    # diameters (in)
    tip_min: float = 10.0
    tip_max: float = 20.0
    tip_inc: float = 0.5
    base_min: float = 20.0
    base_max: float = 60.0
    base_inc: float = 0.5
    # thickness (in)
    t_min: float = 0.1875
    t_max: float = 1.0
    t_inc: float = 0.0625
    t_list: Optional[List[float]] = None      # overrides min/max/inc
    # taper / local buckling
    taper_min: float = 0.15
    taper_max: float = 0.50
    max_wt: float = 35.0
    bend_radius_factor: float = 4.5
    fy: float = 65.0
    # section lengths (ft)
    max_segments: int = 6
    len_preferred: float = 53.0
    len_normal_max: float = 57.0
    len_special_max: float = 60.0
    len_step: float = 0.25
    min_tube: float = 15.0
    #: Fixed bottom tube option (off by default). L is the FABRICATED length
    #: of the bottom tube, embedment included. The tube directly above it
    #: absorbs the remaining height on the 0.25 ft grid; if that tube cannot
    #: fit, the segment count is infeasible and the search moves to n + 1.
    fix_bottom: bool = False
    bottom_length: float = 40.0
    bottom_mode: str = 'exact'                # 'exact' or 'max'
    # joints
    joint_default: str = 'slip'               # 'slip' or 'flange'
    joint_overrides: Dict[int, str] = field(default_factory=dict)  # 1 = lowest joint
    lap_factor: float = 1.65                  # 1.1 x 1.5
    lap_round: float = 0.25
    slip_clearance: float = SLIP_CLEARANCE_IN
    min_slip_above_gl: float = 10.0
    # acceptance
    strength_target: float = 100.0
    defl_target: float = 100.0                # % of each case's XML limit
    # ranking / output
    tie_band_pct: float = 1.0
    #: acceptance tolerance, RELATIVE % of the target (0-1). Applies to
    #: strength and deflection: accept if usage <= target x (1 + tol/100).
    tolerance_pct: float = 0.0
    #: long-tube threshold X (%): a design with a tube longer than
    #: len_normal_max is accepted only if it is more than X% lighter than the
    #: best all-standard design with the same number of segments.
    long_tube_threshold_pct: float = 2.0
    n_alternates: int = 10
    # analysis options
    shear_mode: str = 'resultant'
    lap_stiffness: str = 'midpoint'
    # search effort
    coarse_diam_factor: int = 2               # coarse grid = factor x increment
    coarse_len_values: Tuple[float, ...] = (53.0, 55.0, 57.0, 58.5, 60.0)
    refine_top: int = 6
    max_evaluations: int = 8000
    time_limit_s: float = 900.0               # wall-clock cap for the search phases

    def strength_limit(self) -> float:
        return self.strength_target * (1.0 + self.tolerance_pct / 100.0)

    def defl_limit(self) -> float:
        return self.defl_target * (1.0 + self.tolerance_pct / 100.0)

    def gauges(self) -> List[float]:
        if self.t_list:
            return sorted(round(t, 4) for t in self.t_list)
        n = int(round((self.t_max - self.t_min) / self.t_inc))
        return [round(self.t_min + i * self.t_inc, 4) for i in range(n + 1)]

    def joint_type(self, pos_from_bottom: int) -> str:
        return self.joint_overrides.get(pos_from_bottom, self.joint_default)


# --------------------------------------------------------------------------
# Candidate construction
# --------------------------------------------------------------------------

@dataclass
class Design:
    tip: float
    base: float
    uppers: Tuple[float, ...]          # fabricated lengths, top -> down, excl. bottom
    ts: Tuple[int, ...]                # gauge indices, top -> bottom

    @property
    def n(self) -> int:
        return len(self.uppers) + 1


def make_spec(d: Design, H: float, emb: float, C: OptConstraints,
              G: List[float]) -> Tuple[Optional[PoleSpec], str]:
    """Build a PoleSpec from a Design, or return (None, reason)."""
    n = d.n
    ts = [G[i] for i in d.ts]
    jt = [C.joint_type(n - 1 - k) for k in range(n - 1)]      # joint k below tube k
    steps = sum(2.0 * ts[k] + C.slip_clearance for k in range(n - 1) if jt[k] == 'slip')
    taper = (d.base - d.tip + steps) / H
    if not (C.taper_min - 1e-9 <= taper <= C.taper_max + 1e-9):
        return None, f"taper {taper:.4f} outside [{C.taper_min}, {C.taper_max}]"
    uppers = list(d.uppers)
    flex = None
    if uppers and uppers[-1] is None:
        flex = len(uppers) - 1            # this tube absorbs the remainder
        uppers[flex] = 30.0               # starting guess, solved below
    segs = [Segment(L, ts[k], fy=C.fy, joint_type=jt[k]) for k, L in enumerate(uppers)]
    segs.append(Segment(30.0, ts[-1], fy=C.fy, joint_type='none'))
    spec = PoleSpec(tip_diameter=d.tip, taper=taper, segments=segs, embedment=emb,
                    slip_clearance=C.slip_clearance, lap_factor=C.lap_factor,
                    lap_round=C.lap_round)
    if flex is not None:
        # bottom fixed at L: solve the flexible tube for total height.
        # laps depend on the lengths, so iterate (laps sit on the 0.25 ft
        # grid, so this settles in a few passes).
        Lb = C.bottom_length
        prev = None
        for _ in range(8):
            laps = spec.laps()
            need = round(H + sum(laps) - Lb - sum(uppers[k] for k in range(len(uppers))
                                                  if k != flex), 4)
            if need == prev:
                break
            prev = need
            uppers[flex] = need
            segs[flex] = Segment(max(need, 0.25), ts[flex], fy=C.fy, joint_type=jt[flex])
            segs[-1] = Segment(Lb, ts[-1], fy=C.fy, joint_type='none')
            spec = PoleSpec(tip_diameter=d.tip, taper=taper, segments=segs, embedment=emb,
                            slip_clearance=C.slip_clearance, lap_factor=C.lap_factor,
                            lap_round=C.lap_round)
        if need < C.min_tube - 1e-9:
            return None, f"tube above the fixed bottom {need:.2f} ft < min {C.min_tube}"
        if need > C.len_special_max + 1e-9:
            return None, f"tube above the fixed bottom {need:.2f} ft > max {C.len_special_max}"
        uppers[flex] = need
    laps = spec.laps()
    bottom = round(H - sum(uppers) + sum(laps), 4)
    if C.fix_bottom:
        Lb = C.bottom_length
        if C.bottom_mode == 'exact' and abs(bottom - Lb) > 1e-6:
            return None, f"bottom tube {bottom:.2f} ft != fixed {Lb:g} ft"
        if bottom > Lb + 1e-6:
            return None, f"bottom tube {bottom:.2f} ft > fixed max {Lb:g} ft"
    if bottom < C.min_tube - 1e-9:
        return None, f"bottom tube {bottom:.2f} ft < min {C.min_tube}"
    if bottom > C.len_special_max + 1e-9:
        return None, f"bottom tube {bottom:.2f} ft > max {C.len_special_max}"
    segs[-1] = Segment(bottom, ts[-1], fy=C.fy, joint_type='none')
    for k in range(len(uppers)):
        segs[k] = Segment(uppers[k], ts[k], fy=C.fy, joint_type=jt[k])
    spec = PoleSpec(tip_diameter=d.tip, taper=taper, segments=segs, embedment=emb,
                    slip_clearance=C.slip_clearance, lap_factor=C.lap_factor,
                    lap_round=C.lap_round)
    lay = spec.layout()
    for tb in lay:
        wt = w_over_t(tb['d_bot'], tb['thickness'], C.bend_radius_factor)
        if wt > C.max_wt + 1e-9:
            return None, f"tube {tb['tube_no']} w/t {wt:.1f} > {C.max_wt}"
    slips = [tb for tb in lay if tb['joint_type'] == 'slip']
    if slips:
        low = max(slips, key=lambda tb: tb['lap_bottom'])
        clear = spec.groundline_rel - low['lap_bottom']
        if clear < C.min_slip_above_gl - 1e-9:
            return None, f"lowest slip joint {clear:.2f} ft above GL < {C.min_slip_above_gl}"
    if abs(spec.base_diameter - d.base) > 0.01:     # internal consistency
        return None, f"base D {spec.base_diameter:.3f} != {d.base}"
    return spec, ''


def tube_lengths(spec: PoleSpec) -> List[float]:
    return [s.length for s in spec.segments]


def length_class(spec: PoleSpec, C: OptConstraints) -> dict:
    Ls = tube_lengths(spec)
    special = [L for L in Ls if L > C.len_normal_max + 1e-9]
    dev = sum(abs(L - C.len_preferred) for L in Ls[:-1])
    return dict(n_special=len(special), deviation=dev,
                label=("uses >%.0f ft tube" % C.len_normal_max) if special else "standard")


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

@dataclass
class Evaluated:
    design: Design
    spec: PoleSpec
    weight: float
    strength: float
    defl: Optional[float]
    gov_case: str
    gov_check: str                  # 'strength' or 'deflection'
    verified: bool = False
    result: Optional[CandidateResult] = None


class Optimizer:
    def __init__(self, base: Baseline, baseline_spec: PoleSpec, C: OptConstraints,
                 progress: Optional[Callable[[float, str], None]] = None,
                 log_all: bool = False):
        self.base, self.C = base, C
        self.H = baseline_spec.total_length
        self.emb = baseline_spec.embedment
        self.G = C.gauges()
        self.progress = progress or (lambda f, m: None)
        self.log_all = log_all
        self.log: List[dict] = []
        self.cache: Dict[tuple, Optional[Evaluated]] = {}
        self.n_evals = 0
        self.found: Dict[tuple, Evaluated] = {}
        self.t0 = time.time()
        self.history: List[Tuple[int, float, float]] = []   # (evals, seconds, best screened lb)
        self.best_weight = float('inf')
        self.phase = 'Baseline analysis'

        # baseline run: screening cases + moment envelope for the bound
        self.progress(0.0, "Analysing baseline design (all load cases)")
        b = evaluate_candidate(baseline_spec, base, bend_radius_factor=C.bend_radius_factor,
                               shear_mode=C.shear_mode, lap_stiffness=C.lap_stiffness,
                               spacing=2.0, ds=0.5, keep_rows=False)
        self.baseline_result = b
        by_s = sorted(b.cases.values(), key=lambda x: -x.max_strength)
        scr = [x.case for x in by_s[:3]]
        dcs = sorted([x for x in b.cases.values() if x.defl_check and x.defl_check.get('usage')],
                     key=lambda x: -x.defl_check['usage'])
        scr += [x.case for x in dcs[:2] if x.case not in scr]
        self.screen_cases = scr
        env_s = b.cases[scr[0]].defl.s
        env = np.zeros_like(env_s)
        for x in b.cases.values():
            M = np.hypot(x.defl.Mx, x.defl.My)
            env = np.maximum(env, np.interp(env_s, x.defl.s, M))
        self.env_s, self.env_M = env_s, env
        # deflection-limited cases: moment diagrams + allowable tip deflection
        self.defl_cases = []
        for x in b.cases.values():
            dc = x.defl_check
            if dc and dc.get('allowable_ft'):
                allow = dc['allowable_ft'] * C.defl_target / 100.0
                self.defl_cases.append((x.defl.s.copy(), np.abs(x.defl.Mx).copy(),
                                        np.abs(x.defl.My).copy(), allow))
        self.truncated = False
        self.baseline_spec = baseline_spec

    # ---------------- helpers ----------------
    def _layouts(self, values) -> List[Tuple[float, ...]]:
        """Upper-tube length sets. Tubes longer than the normal max are only
        generated at a segment count where no all-standard layout is
        geometrically possible (the section-length rule), because they would
        be discarded at ranking anyway."""
        C, H = self.C, self.H
        out = []
        for n in range(1, C.max_segments + 1):
            nj = sum(1 for k in range(1, n) if C.joint_type(k) == 'slip')
            lap_hi = 12.0 * nj
            def possible(up, bottom_max):
                lo = H - sum(up)                      # bottom length before laps
                return lo + lap_hi >= C.min_tube and lo <= bottom_max
            if C.fix_bottom:
                # bottom is fixed; the tube directly above it is solved, so
                # only tubes 1..n-2 are enumerated (None marks the solved one)
                if n == 1:
                    out.append(())
                    continue
                for u in itertools.product(values, repeat=n - 2):
                    if sum(u) < H + 12.0 * nj:
                        out.append(tuple(u) + (None,))
                continue
            std = [u for u in itertools.product(
                       [v for v in values if v <= C.len_normal_max + 1e-9], repeat=n - 1)
                   if possible(u, C.len_normal_max)]
            if std:
                out += [tuple(u) for u in std]
            else:
                out += [tuple(u) for u in itertools.product(values, repeat=n - 1)
                        if possible(u, C.len_special_max)]
        return out

    def _lower_bound(self, d: Design) -> Tuple[Optional[float], Tuple[int, ...]]:
        """Relaxed (continuous-thickness) minimum weight and a starting gauge
        set, from three necessary conditions:
          * w/t <= max at each tube bottom
          * strength: M/S <= Fy using 85% of the baseline moment envelope
          * stiffness: first-order tip deflection <= allowable for every
            deflection-limited case, using 90% of the baseline moments,
            minimum-weight thickness distribution by Lagrange multiplier.
        Every condition is a relaxation, so the bound is optimistic -- it
        orders the search; it never accepts a design."""
        C, G = self.C, self.G
        spec, why = make_spec(Design(d.tip, d.base, d.uppers, tuple([len(G) - 1] * d.n)),
                              self.H, self.emb, C, G)
        if spec is None:
            spec, why = make_spec(Design(d.tip, d.base, d.uppers, tuple([0] * d.n)),
                                  self.H, self.emb, C, G)
            if spec is None:
                return None, ()
        lay = spec.layout()
        gl = spec.groundline_rel
        E = 29000.0
        t_str, c_w, samples = [], [], []
        for tb in lay:
            lo, hi = tb['start'], tb['end']
            ss = np.linspace(lo, hi, 12)
            D = tb['d_top'] + spec.taper * (ss - tb['start'])
            above = ss <= gl
            M = 0.85 * np.interp(ss, self.env_s, self.env_M)
            need = float(np.max(np.where(above, 6.0 * M / (0.411 * D * D * C.fy), 0.0)))
            need = max(need, 0.268 * tb['d_bot'] /
                       (C.max_wt + 0.268 * (1 + 2 * C.bend_radius_factor)), G[0])
            t_str.append(need)
            h = (hi - lo) / (len(ss) - 1)
            # lb of steel per inch of wall: 3.22 D * 12 in/ft / 1728 * 490 pcf
            c_w.append(float(_trapz(3.22 * D * 12.0 / 1728.0 * 490.0, dx=h)))
            samples.append((ss, D, above, h))
        # per-tube necessary conditions -> next gauge up is still a valid bound
        snap = []
        for need in t_str:
            k = next((i for i, g in enumerate(G) if g >= need - 1e-9), None)
            if k is None:
                return None, ()
            snap.append(G[k])
        t_str = snap
        t = np.array(t_str)
        for (s_, Mx_, My_, allow) in self.defl_cases:
            A = []
            for (ss, D, above, h) in samples:
                M = 0.9 * np.hypot(np.interp(ss, s_, Mx_), np.interp(ss, s_, My_))
                kap = np.where(above, M * ss * 144.0 / (E * 0.411 * D ** 3), 0.0)
                A.append(float(_trapz(kap, dx=h)))
            A = np.array(A)
            if np.sum(A / t) <= allow:
                continue
            cw = np.array(c_w)
            lo_l, hi_l = 1e-12, 1e12
            for _ in range(80):                       # bisection on multiplier
                lam = math.sqrt(lo_l * hi_l)
                tk = np.maximum(t, np.sqrt(A / (lam * cw)))
                if np.sum(A / tk) > allow:
                    hi_l = lam          # walls too thin -> smaller multiplier
                else:
                    lo_l = lam
            t = np.maximum(t, np.sqrt(A / (lo_l * cw)))   # feasible side
        if t.max() > G[-1] + 1e-9:
            return None, ()
        idx = tuple(next(i for i, g in enumerate(G) if g >= tk - 1e-9) for tk in t)
        return float(np.dot(c_w, t)), idx

    def _eval(self, d: Design, full: bool = False) -> Tuple[Optional[Evaluated], str]:
        key = (d.tip, d.base, d.uppers, d.ts, full)
        if key in self.cache:
            e = self.cache[key]
            return e, ('' if e else 'cached fail')
        spec, why = make_spec(d, self.H, self.emb, self.C, self.G)
        if spec is None:
            self.cache[key] = None
            self._log(d, None, why)
            return None, why
        C = self.C
        self.n_evals += 1
        r = evaluate_candidate(spec, self.base,
                               cases=None if full else self.screen_cases,
                               bend_radius_factor=C.bend_radius_factor,
                               shear_mode=C.shear_mode, lap_stiffness=C.lap_stiffness,
                               spacing=1.0 if full else 2.0, ds=0.25 if full else 0.5,
                               keep_rows=full)
        w = pole_weight(spec, n_steps=40)['total_weight']
        dmax = r.max_defl_usage
        gov_check = 'deflection' if (dmax or 0) / C.defl_target > r.max_strength / C.strength_target \
            else 'strength'
        e = Evaluated(d, spec, w, r.max_strength, dmax,
                      r.gov_defl_case if gov_check == 'deflection' else r.gov_strength_case,
                      gov_check, verified=full, result=r if full else None)
        ok = r.max_strength <= C.strength_limit() + 1e-9 and (dmax or 0) <= C.defl_limit() + 1e-9
        self.cache[key] = e if ok else None
        self._log(d, e, 'pass' if ok else 'fail', r)
        if not ok:
            return None, self._fail_reason(r)
        return e, ''

    def _fail_reason(self, r: CandidateResult) -> str:
        C = self.C
        if r.max_strength > C.strength_limit():
            g = r.cases[r.gov_strength_case].gov_row
            return f"strength:{g.get('tube_no', 1)}"
        return "deflection"

    def _log(self, d, e, status, r=None):
        if not self.log_all:
            return
        self.log.append(dict(tip=d.tip, base=d.base, tubes='/'.join(f"{x:g}" for x in d.uppers),
                             thick='/'.join(f"{self.G[i]:g}" for i in d.ts),
                             weight=round(e.weight, 1) if e else None,
                             strength=round(r.max_strength, 2) if r else None,
                             defl=round(r.max_defl_usage, 2) if (r and r.max_defl_usage) else None,
                             status=status))

    def _defl_tube(self, d: Design) -> int:
        """Tube whose stiffening reduces tip deflection most per lb added."""
        spec, _ = make_spec(d, self.H, self.emb, self.C, self.G)
        best, bi = -1.0, d.n - 1
        for k, tb in enumerate(spec.layout()):
            if d.ts[k] >= len(self.G) - 1:
                continue
            lo, hi = tb['start'], min(tb['end'], spec.groundline_rel)
            ss = np.linspace(lo, hi, 8)
            M = np.interp(ss, self.env_s, self.env_M)
            D = tb['d_top'] + spec.taper * (ss - tb['start'])
            # rotation contribution ~ int M*s/(E I); gain from +1 gauge ~ dI/I
            t0, t1 = self.G[d.ts[k]], self.G[d.ts[k] + 1]
            gain = float(np.sum(M * ss / (D ** 3 * t0))) * (t1 - t0) / t1
            cost = float(np.sum(D)) * (t1 - t0)
            if gain / max(cost, 1e-9) > best:
                best, bi = gain / cost, k
        return bi

    def size(self, d: Design, full: bool = False, max_up: int = 40,
             budget: bool = True) -> Optional[Evaluated]:
        """Thicken until passing, then thin each tube as far as possible."""
        G = self.G
        ts = list(d.ts)
        e, why = self._eval(Design(d.tip, d.base, d.uppers, tuple(ts)), full)
        ups = 0
        while e is None:
            if budget and self._over_budget():
                self.truncated = True
                return None
            if ups >= max_up:
                return None
            if why.startswith('strength'):
                k = int(why.split(':')[1]) - 1
            elif why == 'deflection':
                k = self._defl_tube(Design(d.tip, d.base, d.uppers, tuple(ts)))
            elif why.startswith('tube') and 'w/t' in why:
                k = int(why.split()[1]) - 1
            else:
                return None                     # geometry failure (taper, lengths)
            if ts[k] >= len(G) - 1:
                return None
            ts[k] += 1
            ups += 1
            e, why = self._eval(Design(d.tip, d.base, d.uppers, tuple(ts)), full)
        # thinning: per tube, binary search the thinnest passing gauge
        for _ in range(2):
            changed = False
            for k in range(len(ts)):
                lo, hi = 0, ts[k]
                while lo < hi:
                    mid = (lo + hi) // 2
                    trial = ts.copy(); trial[k] = mid
                    t_e, _ = self._eval(Design(d.tip, d.base, d.uppers, tuple(trial)), full)
                    if t_e is not None:
                        hi = mid
                    else:
                        lo = mid + 1
                if lo < ts[k]:
                    ts[k] = lo; changed = True
            if not changed:
                break
        e, _ = self._eval(Design(d.tip, d.base, d.uppers, tuple(ts)), full)
        return e

    def _over_budget(self) -> bool:
        return (self.n_evals >= self.C.max_evaluations or
                time.time() - self.t0 >= self.C.time_limit_s)

    def _record(self, e: Optional[Evaluated]):
        if e is None:
            return
        key = (e.design.tip, e.design.base, e.design.uppers, e.design.ts)
        self.found[key] = e
        if e.weight < self.best_weight - 1e-6:
            self.best_weight = e.weight
            self.history.append((self.n_evals, time.time() - self.t0, e.weight))

    def _kth_weight(self, k: int) -> float:
        ws = sorted(x.weight for x in self.found.values())
        return ws[k - 1] if len(ws) >= k else float('inf')

    def _search(self, combos: List[Tuple[float, Design]], label: str, f0: float, f1: float):
        K = self.C.n_alternates + 1
        n = len(combos)
        for i, (lb, d) in enumerate(combos):
            if self._over_budget():
                self.truncated = True
                break
            if lb > self._kth_weight(K) * (1 + 3 * self.C.tie_band_pct / 100):
                break
            self.phase = label
            self.progress(f0 + (f1 - f0) * i / max(n, 1),
                          f"{label}: {i + 1}/{n}  D {d.tip}/{d.base}  tubes "
                          f"{'/'.join('auto' if x is None else f'{x:g}' for x in d.uppers) or '-'}  "
                          f"evals {self.n_evals}  best "
                          f"{min((x.weight for x in self.found.values()), default=float('nan')):,.0f} lb")
            self._record(self.size(d))

    def _combos(self, tips, bases, layouts) -> List[Tuple[float, Design]]:
        C, H = self.C, self.H
        out = []
        for tip in tips:
            for base in bases:
                # quick taper pre-filter (steps between 0 and 2.2 in per joint)
                for up in layouts:
                    nj = len(up)
                    tmin = (base - tip) / H
                    tmax = (base - tip + nj * (2 * self.G[-1] + C.slip_clearance)) / H
                    if tmax < C.taper_min - 1e-9 or tmin > C.taper_max + 1e-9:
                        continue
                    d = Design(tip, base, up, ())
                    lb, idx = self._lower_bound(d)
                    if lb is None:
                        continue
                    out.append((lb, Design(tip, base, up, idx)))
        out.sort(key=lambda x: x[0])
        return out

    # ---------------- main ----------------
    def run(self) -> dict:
        C = self.C
        grid = lambda a, b, inc: [round(a + i * inc, 3) for i in range(int(round((b - a) / inc)) + 1)]
        cinc_t, cinc_b = C.tip_inc * C.coarse_diam_factor, C.base_inc * C.coarse_diam_factor
        tips = grid(C.tip_min, C.tip_max, cinc_t)
        bases = grid(C.base_min, C.base_max, cinc_b)
        lay_c = self._layouts([v for v in C.coarse_len_values
                               if C.len_preferred - 1e-9 <= v <= C.len_special_max + 1e-9])

        # Seed: the baseline geometry itself (if it fits the constraint set),
        # so the result can never be heavier than the baseline.
        self.baseline_seed = None
        self.seed_note = ''
        bs = self.baseline_spec
        try:
            ups = tuple(sg.length for sg in bs.segments[:-1])
            idx = tuple(next(i for i, g in enumerate(self.G) if abs(g - sg.thickness) < 1e-6)
                        for sg in bs.segments)
            snap = lambda v, inc: (round(round(v / inc) * inc, 3)
                                   if abs(v / inc - round(v / inc)) * inc < 0.01 else round(v, 3))
            seed = Design(snap(bs.tip_diameter, C.tip_inc), snap(bs.base_diameter, C.base_inc), ups, idx)
            self.phase = 'Sizing baseline (seed)'
            self.progress(0.01, "Sizing the baseline geometry (seed)")
            e = self.size(seed)
            self._record(e)
            self.baseline_seed = e
            if e is None:
                self.seed_note = "baseline geometry could not be made to pass within the constraint set"
            elif e.design.ts != idx:
                chg = ", ".join(f"tube {k + 1} {self.G[a]:g} -> {self.G[b]:g} in"
                                for k, (a, b) in enumerate(zip(idx, e.design.ts)) if a != b)
                self.seed_note = f"re-sized to the acceptance limits: {chg}"
            else:
                self.seed_note = "passes as-is (same thicknesses)"
        except StopIteration:
            self.seed_note = "baseline thickness not in the thickness list -- not seeded"

        self.progress(0.02, "Building coarse candidate list")
        combos = self._combos(tips, bases, lay_c)
        self._search(combos, "Coarse search", 0.05, 0.55)

        # refine around the best designs on the fine grid
        best = sorted(self.found.values(), key=lambda x: x.weight)[:C.refine_top]
        fine = []
        seen = set()
        for e in best:
            d = e.design
            ntips = [t for t in (d.tip - C.tip_inc, d.tip, d.tip + C.tip_inc)
                     if C.tip_min - 1e-9 <= t <= C.tip_max + 1e-9]
            nbases = [b for b in (d.base - C.base_inc, d.base, d.base + C.base_inc)
                      if C.base_min - 1e-9 <= b <= C.base_max + 1e-9]
            nlay = {d.uppers}
            for k in range(len(d.uppers)):
                if d.uppers[k] is None:
                    continue
                for dl in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                    L = round(d.uppers[k] + dl, 2)
                    if C.len_preferred - 1e-9 <= L <= C.len_special_max + 1e-9:
                        u = list(d.uppers); u[k] = L; nlay.add(tuple(u))
            for c in self._combos(ntips, nbases, sorted(nlay, key=lambda u: tuple(-1 if x is None else x for x in u))):
                key = (c[1].tip, c[1].base, c[1].uppers)
                if key not in seen:
                    seen.add(key); fine.append(c)
        fine.sort(key=lambda x: x[0])
        self._search(fine, "Refinement", 0.55, 0.8)

        # verify on all load cases, full mesh
        uniq = {}
        for x in sorted(self.found.values(), key=lambda x: x.weight):
            g = (x.design.tip, x.design.base, x.design.n, x.design.ts)
            uniq.setdefault(g, []).append(x)
        pool = []
        for g in list(uniq)[:C.n_alternates + 5]:
            # lightest variant plus the variants most preferred on length
            vs = uniq[g]
            pref = sorted(vs, key=lambda x: (length_class(x.spec, C)['n_special'],
                                              length_class(x.spec, C)['deviation'], x.weight))
            pick = [vs[0]] + [p for p in pref[:2] if p is not vs[0]]
            pool.extend(pick)
        # the long-tube rule compares the best standard and the best long-tube
        # design at each segment count -> both must always be verified
        ids = {id(x) for x in pool}
        by_n: Dict[int, Dict[bool, List[Evaluated]]] = {}
        for x in self.found.values():
            lng = length_class(x.spec, C)['n_special'] > 0
            by_n.setdefault(x.design.n, {}).setdefault(lng, []).append(x)
        for nn, d_ in by_n.items():
            for lng, xs in d_.items():
                for x in sorted(xs, key=lambda q: q.weight)[:2]:
                    if id(x) not in ids:
                        pool.append(x); ids.add(id(x))
        verified: Dict[tuple, Evaluated] = {}
        self.phase = 'Verifying on all load cases'
        for i, e in enumerate(pool):
            self.progress(0.8 + 0.18 * i / max(len(pool), 1),
                          f"Verifying on all load cases: {i + 1}/{len(pool)}")
            v = self.size(e.design, full=True, budget=False)
            if v is not None:
                k = (v.design.tip, v.design.base, v.design.uppers, v.design.ts)
                verified[k] = v
        self.progress(1.0, "Ranking")
        return self._rank(list(verified.values()))

    # ---------------- ranking ----------------
    def _rank(self, cands: List[Evaluated]) -> dict:
        C = self.C
        info = {}
        # Long-tube rule (threshold X): at each segment count a design with a
        # tube > len_normal_max is kept only if it is more than X% lighter
        # than the best all-standard design with the same segment count.
        by_n: Dict[int, List[Evaluated]] = {}
        for e in cands:
            by_n.setdefault(e.design.n, []).append(e)
        kept, dropped, long_table = [], [], []
        X = C.long_tube_threshold_pct
        for n, es in sorted(by_n.items()):
            std = [e for e in es if length_class(e.spec, C)['n_special'] == 0]
            lng = [e for e in es if length_class(e.spec, C)['n_special'] > 0]
            w_std = min((e.weight for e in std), default=None)
            w_lng = min((e.weight for e in lng), default=None)
            kept += std
            for e in lng:
                if w_std is None or e.weight <= w_std * (1 - X / 100.0) + 1e-9:
                    kept.append(e)
                else:
                    dropped.append(e)
            if w_std is not None and w_lng is not None:
                sav = (1 - w_lng / w_std) * 100
                dec = (f"long-tube designs used (saving {sav:.1f}% > X = {X:g}%)" if sav > X
                       else f"standard lengths kept (saving {sav:.1f}% <= X = {X:g}%)")
            elif w_lng is not None:
                sav, dec = None, "no standard-length design possible -> long tubes allowed"
            else:
                sav, dec = None, "standard lengths only"
            long_table.append({"segments": n,
                               "best standard (lb)": round(w_std) if w_std else None,
                               "best long-tube (lb)": round(w_lng) if w_lng else None,
                               "long-tube saving %": round(sav, 2) if sav is not None else None,
                               "decision": dec})
        common = dict(truncated=self.truncated, n_evals=self.n_evals,
                      elapsed=time.time() - self.t0, screen_cases=self.screen_cases,
                      history=list(self.history), long_table=long_table,
                      baseline=self.baseline_info(), dropped=dropped)
        if not kept:
            return dict(winner=None, alternates=[], kept=[], **common)
        wmin = min(e.weight for e in kept)
        band = wmin * (1 + C.tie_band_pct / 100.0)

        def key(e):
            lc = length_class(e.spec, C)
            in_band = e.weight <= band + 1e-9
            gu = max(e.strength / C.strength_target, (e.defl or 0) / C.defl_target)
            if in_band:
                return (0, e.design.n, lc['n_special'], lc['deviation'], e.weight, gu)
            return (1, e.weight, e.design.n, lc['n_special'], lc['deviation'], gu)

        ranked = sorted(kept, key=key)
        # Alternates must be DISTINCT designs: same diameters, thicknesses
        # and segment count with only tube lengths shifted are collapsed
        # into their best-ranked member (variant count reported).
        groups: Dict[tuple, List[Evaluated]] = {}
        order = []
        for e in ranked:
            g = (e.design.tip, e.design.base, e.design.n, e.design.ts)
            if g not in groups:
                groups[g] = []; order.append(g)
            groups[g].append(e)
        reps = [groups[g][0] for g in order]
        variants = {id(groups[g][0]): len(groups[g]) - 1 for g in order}
        win = reps[0]
        alts = reps[1:1 + C.n_alternates]
        return dict(winner=win, winner_variants=variants[id(win)],
                    alternates=[(a, self.reason(a, win, band), variants[id(a)]) for a in alts],
                    kept=kept, wmin=wmin, band=band, **common)

    def baseline_info(self) -> dict:
        b = self.baseline_result
        w = pole_weight(self.baseline_spec)['total_weight']
        s = self.baseline_seed
        return dict(weight=w, strength=b.max_strength, defl=b.max_defl_usage,
                    gov_strength_case=b.gov_strength_case, gov_defl_case=b.gov_defl_case,
                    seed_weight=s.weight if s else None, seed_note=self.seed_note)

    def reason(self, a: Evaluated, w: Evaluated, band: float) -> str:
        C = self.C
        r = []
        dw = a.weight - w.weight
        la, lw = length_class(a.spec, C), length_class(w.spec, C)
        if a.weight <= band + 1e-9:
            r.append("within tie band")
            if a.design.n > w.design.n:
                r.append(f"{a.design.n} segments vs {w.design.n}")
            elif la['n_special'] > lw['n_special']:
                r.append(la['label'])
            elif la['deviation'] > lw['deviation'] + 1e-9:
                r.append(f"tube lengths further from {C.len_preferred:g} ft")
            else:
                r.append(f"heavier by {dw:,.0f} lb")
        else:
            r.append(f"heavier by {dw:,.0f} lb (+{dw / w.weight * 100:.1f}%)")
            if a.design.n != w.design.n:
                r.append(f"{a.design.n} segments vs {w.design.n}")
            if la['n_special']:
                r.append(la['label'])
        r.append(f"governed by {a.gov_check} ({a.gov_case[:30]})")
        return "; ".join(r)


def summarize(e: Evaluated, C: OptConstraints) -> dict:
    s = e.spec
    lay = s.layout()
    return {
        "D tip (in)": s.tip_diameter, "D base (in)": round(s.base_diameter, 2),
        "taper (in/ft)": round(s.taper, 5), "segments": e.design.n,
        "tube lengths (ft)": " / ".join(f"{t['length']:g}" for t in lay),
        "thickness (in)": " / ".join(f"{t['thickness']:g}" for t in lay),
        "laps (ft)": " / ".join(f"{t['lap']:g}" for t in lay[:-1]) or "-",
        "joints": " / ".join(t['joint_type'] for t in lay[:-1]) or "-",
        "weight (lb)": round(e.weight, 0),
        "max strength %": round(e.strength, 2),
        "max defl %": round(e.defl, 2) if e.defl is not None else None,
        "governs": e.gov_check, "gov case": e.gov_case,
        "acceptance": ("within tolerance" if (e.strength > C.strength_target + 1e-9 or
                                               (e.defl or 0) > C.defl_target + 1e-9) else "at/below target"),
        "long tube": "yes" if length_class(e.spec, C)['n_special'] else "no",
    }
