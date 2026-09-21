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
    n_alternates: int = 10
    # analysis options
    shear_mode: str = 'resultant'
    lap_stiffness: str = 'outer'
    # search effort
    coarse_diam_factor: int = 2               # coarse grid = factor x increment
    coarse_len_values: Tuple[float, ...] = (53.0, 55.0, 57.0, 58.5, 60.0)
    refine_top: int = 6
    max_evaluations: int = 4000

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
    segs = [Segment(L, ts[k], fy=C.fy, joint_type=jt[k]) for k, L in enumerate(d.uppers)]
    segs.append(Segment(30.0, ts[-1], fy=C.fy, joint_type='none'))
    spec = PoleSpec(tip_diameter=d.tip, taper=taper, segments=segs, embedment=emb,
                    slip_clearance=C.slip_clearance, lap_factor=C.lap_factor,
                    lap_round=C.lap_round)
    laps = spec.laps()
    bottom = round(H - sum(d.uppers) + sum(laps), 4)
    if bottom < C.min_tube - 1e-9:
        return None, f"bottom tube {bottom:.2f} ft < min {C.min_tube}"
    if bottom > C.len_special_max + 1e-9:
        return None, f"bottom tube {bottom:.2f} ft > max {C.len_special_max}"
    segs[-1] = Segment(bottom, ts[-1], fy=C.fy, joint_type='none')
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

    # ---------------- helpers ----------------
    def _layouts(self, values) -> List[Tuple[float, ...]]:
        C, H = self.C, self.H
        out = []
        for n in range(1, C.max_segments + 1):
            for up in itertools.product(values, repeat=n - 1):
                # crude lap envelope: 0-12 ft per joint
                lo = H - sum(up)
                if lo + 12.0 * (n - 1) < C.min_tube or lo > C.len_special_max:
                    continue
                out.append(tuple(up))
        return out

    def _lower_bound(self, d: Design) -> Tuple[Optional[float], Tuple[int, ...]]:
        """Cheap lower-bound thickness per tube and the resulting weight."""
        C, G = self.C, self.G
        spec, why = make_spec(Design(d.tip, d.base, d.uppers, tuple([len(G) - 1] * d.n)),
                              self.H, self.emb, C, G)
        if spec is None and not why.startswith('tube'):
            # geometric failure independent of thickness -> try thinnest
            spec, why = make_spec(Design(d.tip, d.base, d.uppers, tuple([0] * d.n)),
                                  self.H, self.emb, C, G)
            if spec is None:
                return None, ()
        lay = spec.layout()
        idx = []
        for tb in lay:
            lo, hi = tb['start'], min(tb['end'], spec.groundline_rel)
            need = 0.0
            for s in np.linspace(lo, hi, 6):
                D = tb['d_top'] + spec.taper * (s - tb['start'])
                M = 0.85 * float(np.interp(s, self.env_s, self.env_M))
                need = max(need, 6.0 * M / (0.411 * D * D * C.fy))
            # w/t: w = 0.268 (D - t - 2 BR t) -> t >= 0.268 D / (max_wt + 0.268(1+2BR))
            need = max(need, 0.268 * tb['d_bot'] /
                       (C.max_wt + 0.268 * (1 + 2 * C.bend_radius_factor)))
            k = next((i for i, g in enumerate(G) if g >= need - 1e-9), None)
            if k is None:
                return None, ()
            idx.append(k)
        spec2, _ = make_spec(Design(d.tip, d.base, d.uppers, tuple(idx)),
                             self.H, self.emb, C, G)
        if spec2 is None:
            return None, tuple(idx)
        return pole_weight(spec2, n_steps=20)['total_weight'], tuple(idx)

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
        ok = r.max_strength <= C.strength_target + 1e-9 and (dmax or 0) <= C.defl_target + 1e-9
        self.cache[key] = e if ok else None
        self._log(d, e, 'pass' if ok else 'fail', r)
        if not ok:
            return None, self._fail_reason(r)
        return e, ''

    def _fail_reason(self, r: CandidateResult) -> str:
        C = self.C
        if r.max_strength > C.strength_target:
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

    def size(self, d: Design, full: bool = False, max_up: int = 40) -> Optional[Evaluated]:
        """Thicken until passing, then thin each tube as far as possible."""
        G = self.G
        ts = list(d.ts)
        e, why = self._eval(Design(d.tip, d.base, d.uppers, tuple(ts)), full)
        ups = 0
        while e is None:
            if ups >= max_up or self.n_evals >= self.C.max_evaluations:
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

    def _record(self, e: Optional[Evaluated]):
        if e is None:
            return
        key = (e.design.tip, e.design.base, e.design.uppers, e.design.ts)
        self.found[key] = e

    def _kth_weight(self, k: int) -> float:
        ws = sorted(x.weight for x in self.found.values())
        return ws[k - 1] if len(ws) >= k else float('inf')

    def _search(self, combos: List[Tuple[float, Design]], label: str, f0: float, f1: float):
        K = self.C.n_alternates + 1
        n = len(combos)
        for i, (lb, d) in enumerate(combos):
            if self.n_evals >= self.C.max_evaluations:
                break
            if lb > self._kth_weight(K) * (1 + 3 * self.C.tie_band_pct / 100):
                break
            self.progress(f0 + (f1 - f0) * i / max(n, 1),
                          f"{label}: {i + 1}/{n}  D {d.tip}/{d.base}  tubes "
                          f"{'/'.join(f'{x:g}' for x in d.uppers) or '-'}  "
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
                for dl in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                    L = round(d.uppers[k] + dl, 2)
                    if C.len_preferred - 1e-9 <= L <= C.len_special_max + 1e-9:
                        u = list(d.uppers); u[k] = L; nlay.add(tuple(u))
            for c in self._combos(ntips, nbases, sorted(nlay)):
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
        verified: Dict[tuple, Evaluated] = {}
        for i, e in enumerate(pool):
            self.progress(0.8 + 0.18 * i / max(len(pool), 1),
                          f"Verifying on all load cases: {i + 1}/{len(pool)}")
            v = self.size(e.design, full=True)
            if v is not None:
                k = (v.design.tip, v.design.base, v.design.uppers, v.design.ts)
                verified[k] = v
        self.progress(1.0, "Ranking")
        return self._rank(list(verified.values()))

    # ---------------- ranking ----------------
    def _rank(self, cands: List[Evaluated]) -> dict:
        C = self.C
        info = {}
        # section-length rule: >normal_max tube only if it saves a segment
        by_n: Dict[int, List[Evaluated]] = {}
        for e in cands:
            by_n.setdefault(e.design.n, []).append(e)
        kept, dropped = [], []
        for n, es in by_n.items():
            has_std = any(length_class(e.spec, C)['n_special'] == 0 for e in es)
            for e in es:
                if has_std and length_class(e.spec, C)['n_special'] > 0:
                    dropped.append(e)
                else:
                    kept.append(e)
        if not kept:
            return dict(winner=None, alternates=[], dropped=dropped, n_evals=self.n_evals,
                        elapsed=time.time() - self.t0, screen_cases=self.screen_cases)
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
                    dropped=dropped, wmin=wmin, band=band, n_evals=self.n_evals,
                    elapsed=time.time() - self.t0, screen_cases=self.screen_cases)

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
    }
