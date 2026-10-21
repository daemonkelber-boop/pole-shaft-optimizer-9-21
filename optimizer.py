"""
optimizer.py  --  Phase 3, module 4

Minimum-weight shaft search for a fixed total height, fixed embedment and
fixed loading (baseline PLS-POLE XML).

Design variables
    * tip and base diameter, each on a user grid (default 0.5 in)
    * number of segments and the fabricated length of every tube except the
      bottom one (the bottom tube takes the remainder). Non-bottom tubes may
      be anywhere from min_tube to len_special_max; the preferred / normal
      max lengths only decide ranking (see _rank)
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
    2. Enumerate (D_tip, D_base, layout) combinations on a COARSE grid. The
       layouts are the standard-length grid, the seed XML's own tube lengths
       and, per pair, the splice-height layouts (_splice_layouts). Then
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
    6. Ranking: lightest wins, subject to the fabrication rules -- extra
       segment or 57-60 ft tube only for a saving >= X% (fab_penalty_pct),
       short-tube layout over standard only for a saving >= NOISE_PCT.

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

from geometry import (PoleSpec, Segment, w_over_t, GAUGES, SLIP_CLEARANCE_IN,
                      slip_joint_lap)
from weight import pole_weight
from loads import Baseline, build_load_model, first_order_forces
from strength import evaluate_candidate, CandidateResult
from baseplate import BasePlateRules, size_baseplate, base_reactions


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
    max_wt: float = 38.0
    bend_radius_factor: float = 4.5
    fy: float = 65.0
    # section lengths (ft)
    max_segments: int = 6
    len_preferred: float = 53.0
    len_normal_max: float = 57.0
    len_special_max: float = 60.0
    len_step: float = 0.25
    min_tube: float = 15.0
    # thickness-dependent section length limits (optional)
    # mapping of {thickness_in: max_length_ft}, e.g. {0.75: 50.0}
    max_len_by_thick: Dict[float, float] = field(default_factory=dict)
    thick_len_mode: str = 'gte'                # 'gte' (>= thickness) or 'exact'
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
    min_slip_above_gl: float = 0.0
    # acceptance
    strength_target: float = 100.0
    defl_target: float = 100.0                # % of each case's XML limit
    # ranking / output
    #: search margin: a combination whose optimistic lower-bound weight is
    #: more than this % above the best design found is skipped.
    explore_margin_pct: float = 3.0
    #: acceptance tolerance, RELATIVE % of the target (0-1). Applies to
    #: strength and deflection: accept if usage <= target x (1 + tol/100).
    tolerance_pct: float = 0.0
    #: Minimum saving X (%) to accept a fabrication penalty. ONE setting for
    #: both penalties, test is ">= X%":
    #:   * extra segment: a design with more segments is used only if it is
    #:     at least X% lighter per extra segment than the best design with
    #:     fewer segments (also when a 57-60 ft tube is what removes the
    #:     segment -- fewer segments wins unless the extra segment saves X%);
    #:   * 57-60 ft tube: at the same segment count, a design with a tube
    #:     longer than len_normal_max is used only if it is at least X%
    #:     lighter than the best design WITHOUT a long tube (short-tube
    #:     layouts included).
    fab_penalty_pct: float = 2.0
    n_alternates: int = 10
    #: splice-height search: add, for every (tip, base) pair, the layout whose
    #: splice heights minimise the estimated weight at each segment count
    #: (non-bottom tubes anywhere from min_tube to len_normal_max).
    splice_search: bool = True
    # analysis options
    shear_mode: str = 'resultant'
    lap_stiffness: str = 'midpoint'
    # search effort
    coarse_diam_factor: int = 2               # coarse grid = factor x increment
    coarse_len_values: Tuple[float, ...] = (53.0, 55.0, 57.0, 58.5, 60.0)
    refine_top: int = 6
    max_evaluations: int = 200000          # effectively off; time_limit_s governs
    time_limit_s: float = 900.0               # wall-clock cap for the search phases
    # base plate post-check (base-plate poles only). None = off.
    #: Shaft-only ranking; a shaft with no feasible plate/anchor-bolt
    #: layout is rejected and the next-ranked shaft is promoted.
    bp_rules: Optional[BasePlateRules] = None

    @property
    def long_tube_threshold_pct(self) -> float:
        """Back-compat alias: the long-tube threshold is fab_penalty_pct."""
        return self.fab_penalty_pct

    def strength_limit(self) -> float:
        if self.strength_target <= 0:
            return float('inf')
        return self.strength_target * (1.0 + self.tolerance_pct / 100.0)

    def defl_limit(self) -> float:
        # target <= 0 means "no deflection limit" -> nothing can fail on it
        if self.defl_target <= 0:
            return float('inf')
        return self.defl_target * (1.0 + self.tolerance_pct / 100.0)

    def defl_disabled(self) -> bool:
        return self.defl_target <= 0

    def gauges(self) -> List[float]:
        if self.t_list:
            return sorted(round(t, 4) for t in self.t_list)
        n = int(round((self.t_max - self.t_min) / self.t_inc))
        return [round(self.t_min + i * self.t_inc, 4) for i in range(n + 1)]

    def joint_type(self, pos_from_bottom: int) -> str:
        return self.joint_overrides.get(pos_from_bottom, self.joint_default)

    def max_length_for_thickness(self, t: float) -> Optional[float]:
        """Return maximum allowable section length for thickness t, or None if unconstrained."""
        if not self.max_len_by_thick:
            return None
        limits = []
        for rule_t, max_l in self.max_len_by_thick.items():
            if self.thick_len_mode == 'gte':
                if t >= rule_t - 1e-6:
                    limits.append(max_l)
            else:
                if abs(t - rule_t) < 1e-4:
                    limits.append(max_l)
        return min(limits) if limits else None


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
              G: List[float], check_thick_len: bool = True) -> Tuple[Optional[PoleSpec], str]:
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
    if check_thick_len and C.max_len_by_thick:
        for tb in lay:
            cap = C.max_length_for_thickness(tb['thickness'])
            if cap is not None and tb['length'] > cap + 1e-6:
                return None, f"tube {tb['tube_no']} length {tb['length']:.2f} ft > max {cap:g} ft for thickness {tb['thickness']:g}\""
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


#: Hidden noise tolerance (%): within the same segment count, a layout with a
#: non-bottom tube shorter than the preferred length replaces the best
#: standard-length layout only if it is at least this much lighter. Smaller
#: differences are inside the engine's weight accuracy. Not a UI setting.
NOISE_PCT = 0.5


def length_class(spec: PoleSpec, C: OptConstraints) -> dict:
    """standard : every non-bottom tube within [len_preferred, len_normal_max]
    short    : a non-bottom tube shorter than len_preferred (no long tube)
    long     : any tube longer than len_normal_max
    The bottom tube takes the remainder, so it is never 'short'."""
    Ls = tube_lengths(spec)
    special = [L for L in Ls if L > C.len_normal_max + 1e-9]
    short = [L for L in Ls[:-1] if L < C.len_preferred - 1e-9]
    dev = sum(abs(L - C.len_preferred) for L in Ls[:-1])
    if special:
        cls, label = 'long', "uses >%.0f ft tube" % C.len_normal_max
    elif short:
        cls, label = 'short', "short tube (<%.0f ft)" % C.len_preferred
    else:
        cls, label = 'standard', "standard"
    return dict(n_special=len(special), n_short=len(short), deviation=dev,
                cls=cls, label=label)


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
    baseplate: Optional[dict] = None


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
        self._env_cache: Dict[tuple, tuple] = {}
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
        for x in ([] if C.defl_disabled() else b.cases.values()):
            dc = x.defl_check
            if dc and dc.get('allowable_ft'):
                allow = dc['allowable_ft'] * C.defl_target / 100.0
                self.defl_cases.append((x.defl.s.copy(), np.abs(x.defl.Mx).copy(),
                                        np.abs(x.defl.My).copy(), allow))
        self.truncated = False
        self.trunc_reason = ''
        self.lb_remaining = float('inf')
        self.baseline_spec = baseline_spec
        # Base plate feasibility as a SEARCH constraint (base-plate poles).
        # Demand = baseline base reactions (all load cases); feasibility is
        # cached per (base AF, bottom wall). The exact check with each
        # design's own reactions is repeated in _rank (post-check).
        self._splice_cache: Dict[tuple, List[Tuple[float, ...]]] = {}
        self._pd = None
        #: the seed XML's own upper-tube lengths, tried as a layout for every
        #: (tip, base) pair (set in run(); None = not usable)
        self.seed_uppers: Optional[Tuple[float, ...]] = None
        self.bp_on = C.bp_rules is not None and self.emb == 0
        self._bp_cache: Dict[tuple, bool] = {}
        if self.bp_on:
            import dataclasses
            self._bp_R = dataclasses.replace(C.bp_rules, target=C.strength_target)
            self._bp_reac = base_reactions(b.cases)

    # ---------------- helpers ----------------
    def bp_feasible(self, D_af: float, t: float) -> bool:
        """True if a plate / anchor bolt layout exists for this base AF and
        bottom wall (rule bolt circle, 2.67d spacing, anchor bolt checks,
        6.4.2), using the baseline base reactions."""
        if not self.bp_on:
            return True
        key = (round(D_af, 3), round(t, 4))
        if key not in self._bp_cache:
            try:
                out = size_baseplate(D_af, t, self.C.fy, self._bp_reac, self._bp_R,
                                     bend_radius_factor=self.C.bend_radius_factor)
                self._bp_cache[key] = out['best'] is not None
            except ValueError:
                self._bp_cache[key] = False
        return self._bp_cache[key]

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

    # ---------------- splice-height search ----------------
    def _pdelta_ratio(self, ss: np.ndarray) -> np.ndarray:
        """Baseline second-order / first-order moment ratio at rel_dist ss,
        clipped to [1.0, 1.3]. Scales a candidate's first-order envelope to
        an estimate of its P-delta moments for the splice search."""
        if getattr(self, '_pd', None) is None:
            first = self._cand_env(self.baseline_spec, self.env_s)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(first > 1e-6, self.env_M / first, 1.0)
            self._pd = np.clip(np.nan_to_num(r, nan=1.0), 1.0, 1.3)
        return np.interp(ss, self.env_s, self._pd)

    def _splice_layouts(self, tip: float, base: float) -> List[Tuple[float, ...]]:
        """Upper-tube lengths whose splice heights minimise estimated weight.

        For each segment count n (the fewest that fit with tubes <= normal
        max, and one more) a dynamic programme over splice positions on a
        0.5 ft grid picks the non-bottom tube lengths (min_tube .. normal
        max) that minimise the estimated steel weight, where each tube's wall
        is the largest gauge required anywhere along it and walls never get
        thinner going down. Required gauge per station = max(strength from
        the candidate's first-order moment envelope x the baseline P-delta
        ratio, w/t limit). It is an estimate that only proposes layouts: the
        normal sizing, screening and full verification decide every design.
        """
        C, G, H = self.C, self.G, self.H
        if not C.splice_search or C.fix_bottom:
            return []
        key = (tip, base)
        if key in self._splice_cache:
            return self._splice_cache[key]
        out: List[Tuple[float, ...]] = []
        self._splice_cache[key] = out
        step = 0.5
        N = int(math.floor(H / step + 1e-9))
        s = np.arange(N + 1) * step
        taper0 = (base - tip) / H
        D = tip + taper0 * s
        gl = self.H - self.emb
        ref = PoleSpec(tip_diameter=tip, taper=taper0,
                       segments=[Segment(H, G[0], fy=C.fy, joint_type='none')],
                       embedment=self.emb, slip_clearance=C.slip_clearance,
                       lap_factor=C.lap_factor, lap_round=C.lap_round)
        M = self._cand_env(ref, s) * self._pdelta_ratio(s)
        u = max(C.strength_target, 1.0) / 100.0
        # stress = 12 M / S, S ~ 0.77 D^2 t for a 12-sided tube
        t_str = np.where(s <= gl + 1e-9, 12.0 * M / (0.77 * D * D * C.fy * u), 0.0)
        t_wt = 0.268 * D / (C.max_wt + 0.268 * (1 + 2 * C.bend_radius_factor))
        need = np.maximum(t_str, t_wt)
        Garr = np.array(G)
        req = np.searchsorted(Garr, need - 1e-9).astype(int)     # gauge index
        BIG = len(G)                                              # infeasible
        req = np.minimum(req, BIG)
        cw = 3.22 * D * 12.0 / 1728.0 * 490.0                     # lb / ft / in
        CW = np.concatenate([[0.0], np.cumsum((cw[1:] + cw[:-1]) * 0.5 * step)])
        lmin = int(math.ceil(C.min_tube / step - 1e-9))
        lmax = int(math.floor(C.len_normal_max / step + 1e-9))
        if lmax < lmin:
            return out
        pad = np.concatenate([req, np.full(lmax + 1, BIG)])
        from numpy.lib.stride_tricks import sliding_window_view
        RM = np.maximum.accumulate(sliding_window_view(pad, lmax + 1)[:N + 1], axis=1)
        SM = np.maximum.accumulate(req[::-1])[::-1]               # max req[a:]
        lap_idx = np.zeros(N + 1, dtype=int)
        for b in range(N + 1):
            fid = (D[b] - 2.0 * 0.5) / 12.0
            lap_idx[b] = int(math.ceil(slip_joint_lap(fid, C.lap_factor, C.lap_round) / step - 1e-9))
        caps = np.array([C.max_length_for_thickness(g) or 1e9 for g in G] + [0.0])
        Gw = np.append(Garr, 1e9)                                 # weight per gauge idx
        gi = np.arange(len(G) + 1)
        INF = float('inf')
        # fewest segments with every tube <= normal max
        n_lo = None
        for n in range(1, C.max_segments + 1):
            laps_est = sum(1.65 * D[-1] / 12.0 for _ in range(n - 1))
            if n * C.len_normal_max - laps_est >= H - 1e-6:
                n_lo = n
                break
        if n_lo is None:
            return out
        Ga = len(G) + 1
        rows = np.arange(N + 1)

        def prefix_argmin(B):
            """index of the min of B[:, :j+1] for every j"""
            idx = np.zeros(B.shape, dtype=int)
            cur = B[:, 0].copy()
            for j in range(1, B.shape[1]):
                better = B[:, j] < cur
                idx[:, j] = np.where(better, j, idx[:, j - 1])
                cur = np.where(better, B[:, j], cur)
            return idx

        def place(B, r):
            """cost-free part of a transition: tube gauge g' >= r, taking the
            best predecessor gauge g (g <= r maps to r, g > r keeps g)."""
            pa = prefix_argmin(B)
            src = np.where(gi[None, :] > r[:, None], gi[None, :], -1)
            val = np.where(gi[None, :] > r[:, None], B, INF)
            k_ = np.arange(B.shape[0])
            src[k_, r] = pa[k_, r]
            val[k_, r] = B[k_, pa[k_, r]]
            return val, src

        for n in range(max(n_lo, 2), min(n_lo + 1, C.max_segments) + 1):
            best = np.full((N + 1, Ga), INF)
            best[0, 0] = 0.0
            ptrs = []                         # per upper tube: (a, l, g_src) arrays
            for k in range(n - 1):
                jt = C.joint_type(n - 1 - k)
                new = np.full_like(best, INF)
                pa_ = np.full(best.shape, -1, dtype=int)
                pl_ = np.full(best.shape, -1, dtype=int)
                pg_ = np.full(best.shape, -1, dtype=int)
                act = np.where(np.isfinite(best).any(axis=1))[0]
                for l in range(lmin, lmax + 1):
                    a_ = act[act + l <= N]
                    if a_.size == 0:
                        break
                    b_ = a_ + l
                    val, src = place(best[a_], RM[a_, l])
                    tot = val + (CW[b_] - CW[a_])[:, None] * Gw[None, :]
                    tot[:, BIG] = INF
                    tot[:, caps < l * step - 1e-6] = INF
                    nxt = b_ - (lap_idx[b_] if jt == 'slip' else 0)
                    old = new[nxt].copy()
                    np.minimum.at(new, nxt, tot)       # nxt can repeat
                    win = (tot < old) & (tot == new[nxt])
                    jj, gg = np.nonzero(win)
                    if jj.size:
                        pa_[nxt[jj], gg] = a_[jj]
                        pl_[nxt[jj], gg] = l
                        pg_[nxt[jj], gg] = src[jj, gg]
                best = new
                ptrs.append((pa_, pl_, pg_))
            Lb = H - s
            okb = (Lb >= C.min_tube - 1e-9) & (Lb <= C.len_normal_max + 1e-9)
            val, src = place(best, SM)
            tot = val + (CW[N] - CW)[:, None] * Gw[None, :]
            tot[:, BIG] = INF
            tot[~okb, :] = INF
            for g in range(len(G)):
                tot[caps[g] < Lb - 1e-6, g] = INF
            if not np.isfinite(tot).any():
                continue
            a_cur, g_bot = np.unravel_index(int(np.argmin(tot)), tot.shape)
            g_cur = int(src[a_cur, g_bot])       # gauge of the tube above
            lens: List[float] = []
            a_cur = int(a_cur)
            for (pa_, pl_, pg_) in reversed(ptrs):
                l = int(pl_[a_cur, g_cur])
                if l < 0:
                    lens = None
                    break
                lens.append(l * step)
                a_cur, g_cur = int(pa_[a_cur, g_cur]), int(pg_[a_cur, g_cur])
            if lens is not None and a_cur == 0:
                up = tuple(round(x, 2) for x in reversed(lens))
                if up not in out:
                    out.append(up)
        return out

    def _cand_env(self, spec, ss, case_s=None, case_M=None):
        """First-order moment envelope of THIS candidate at stations ss.

        Point (wire / arm) loads are geometry independent; only the shaft
        wind changes with diameter. First-order moments are always smaller
        than the second-order ones the real check uses, so this keeps the
        bound optimistic (a true lower bound on the steel required)."""
        key = (id(spec), None if case_s is None else len(case_s))
        cache = self._env_cache.get(key)
        if cache is None:
            cases = [self.screen_cases[0]] if case_s is not None else self.screen_cases
            grid = np.linspace(0, spec.groundline_rel, 60)
            env = np.zeros_like(grid)
            for c in cases:
                m = build_load_model(spec, self.base, c, ds=2.0)
                vals = []
                for s0 in grid:
                    f = first_order_forces(m, s0)
                    vals.append(math.hypot(f['Mx'], f['My']))
                env = np.maximum(env, np.array(vals))
            cache = (grid, env)
            if len(self._env_cache) > 64:
                self._env_cache.clear()
            self._env_cache[key] = cache
        grid, env = cache
        out = np.interp(ss, grid, env)
        if case_s is not None:
            # scale the candidate envelope to this deflection case
            ref = np.interp(ss, grid, env)
            cas = np.interp(ss, case_s, case_M)
            base_ref = np.interp(ss, self.env_s, self.env_M)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(base_ref > 1e-9, cas / base_ref, 0.0)
            out = ref * np.clip(ratio, 0.0, 1.0)
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
                              self.H, self.emb, C, G, check_thick_len=False)
        if spec is None:
            # The bound only needs a geometry to integrate over. All-max
            # gauges can push the derived taper past its limit and all-min
            # gauges can fail w/t, so also try every uniform gauge (before
            # this, valid pairs such as 007 at 28 / 85.5 in were skipped).
            for gi_ in [0] + list(range(1, len(G) - 1)):
                spec, why = make_spec(Design(d.tip, d.base, d.uppers, tuple([gi_] * d.n)),
                                      self.H, self.emb, C, G, check_thick_len=False)
                if spec is not None:
                    break
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
            M = self._cand_env(spec, ss)
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
                M = self._cand_env(spec, ss, case_s=s_, case_M=np.hypot(Mx_, My_))
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
        if self.bp_on:
            kb = next((i for i in range(idx[-1], len(G)) if self.bp_feasible(d.base, G[i])), None)
            if kb is None:
                return None, ()          # no bottom wall makes the base plate work
            if kb > idx[-1]:
                idx = idx[:-1] + (kb,)
                t = t.copy(); t[-1] = G[kb]
        if C.max_len_by_thick:
            for k, tb in enumerate(lay):
                cap = C.max_length_for_thickness(G[idx[k]])
                if cap is not None and tb['length'] > cap + 1e-6:
                    return None, ()
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
        if self.bp_on and not self.bp_feasible(spec.base_diameter, spec.segments[-1].thickness):
            self.cache[key] = None
            self._log(d, None, 'base plate infeasible')
            return None, 'baseplate'
        self.n_evals += 1
        r = evaluate_candidate(spec, self.base,
                               cases=None if full else self.screen_cases,
                               bend_radius_factor=C.bend_radius_factor,
                               shear_mode=C.shear_mode, lap_stiffness=C.lap_stiffness,
                               spacing=1.0 if full else 2.0, ds=0.25 if full else 0.5,
                               keep_rows=full)
        w = pole_weight(spec, n_steps=40)['total_weight']
        dmax = r.max_defl_usage
        d_ratio = (dmax or 0.0) / C.defl_target if C.defl_target > 0 else 0.0
        s_ratio = r.max_strength / C.strength_target if C.strength_target > 0 else 0.0
        gov_check = 'deflection' if d_ratio > s_ratio else 'strength'
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
        self.log.append(dict(tip=d.tip, base=d.base, tubes='/'.join('auto' if x is None else f"{x:g}" for x in d.uppers),
                             thick='/'.join(f"{self.G[i]:g}" for i in d.ts),
                             weight=round(e.weight, 1) if e else None,
                             strength=round(r.max_strength, 2) if r else None,
                             defl=round(r.max_defl_usage, 2) if (r and r.max_defl_usage) else None,
                             status=status))

    def _defl_tube(self, d: Design) -> int:
        """Tube whose stiffening reduces tip deflection most per lb added."""
        spec, _ = make_spec(d, self.H, self.emb, self.C, self.G, check_thick_len=False)
        best, bi = -1.0, -1
        for k, tb in enumerate(spec.layout()):
            if d.ts[k] >= len(self.G) - 1:
                continue
            if self.C.max_len_by_thick:
                next_t = self.G[d.ts[k] + 1]
                cap = self.C.max_length_for_thickness(next_t)
                if cap is not None and tb['length'] > cap + 1e-6:
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
                if k < 0:
                    return None
            elif why.startswith('tube') and 'w/t' in why:
                k = int(why.split()[1]) - 1
            elif why == 'baseplate':
                k = len(ts) - 1
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
        if self.n_evals >= self.C.max_evaluations:
            self.trunc_reason = f"evaluation cap ({self.C.max_evaluations:,}) reached"
            return True
        if time.time() - self.t0 >= self.C.time_limit_s:
            self.trunc_reason = f"time limit ({self.C.time_limit_s / 60:.0f} min) reached"
            return True
        return False

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
                self.lb_remaining = min(self.lb_remaining, lb)
                break
            # Prune against the BEST weight found so far plus a margin, not
            # the Kth best: a combination whose optimistic lower bound is
            # already heavier than that cannot win, and alternates come from
            # designs that were evaluated anyway.
            best = min((x.weight for x in self.found.values()), default=float('inf'))
            # a fewer-segment design up to X% heavier can still win, so the
            # margin is at least X
            margin = 1 + max(self.C.fab_penalty_pct, NOISE_PCT, self.C.explore_margin_pct) / 100.0
            if lb > min(best * margin, self._kth_weight(K)):
                self.lb_remaining = min(self.lb_remaining, lb)
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
                lays = list(layouts)
                # splice-height layouts and the seed's own tube lengths are
                # tried for every pair, alongside the standard grid
                extra = self._splice_layouts(tip, base)
                if self.seed_uppers is not None and not C.fix_bottom:
                    extra = extra + [self.seed_uppers]
                seen_l = set(lays)
                lays += [u for u in extra if u not in seen_l]
                # quick taper pre-filter (steps between 0 and 2.2 in per joint)
                for up in lays:
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
            # the seed's tube-length layout is a candidate for EVERY (tip,
            # base) pair (laps recomputed, bottom tube takes the remainder);
            # make_spec and sizing apply all the checks
            if (len(ups) + 1 <= C.max_segments
                    and all(C.min_tube - 1e-9 <= L <= C.len_special_max + 1e-9 for L in ups)):
                self.seed_uppers = tuple(round(L, 2) for L in ups)
            self.phase = 'Sizing baseline (seed)'
            self.progress(0.01, "Sizing the baseline geometry (seed)")
            _, seed_why = make_spec(seed, self.H, self.emb, C, self.G)
            e = self.size(seed)
            self._record(e)
            self.baseline_seed = e
            lay_note = (f"its tube layout {' / '.join(f'{L:g}' for L in ups)} ft (+ bottom) is tried "
                        "at every tip/base pair" if self.seed_uppers else
                        "its tube layout is outside the length limits, so it is not tried")
            if e is None:
                self.seed_note = ("baseline geometry could not be made to pass within the constraint set"
                                  + (f" ({seed_why})" if seed_why else "") + f"; {lay_note}")
            elif e.design.ts != idx:
                chg = ", ".join(f"tube {k + 1} {self.G[a]:g} -> {self.G[b]:g} in"
                                for k, (a, b) in enumerate(zip(idx, e.design.ts)) if a != b)
                self.seed_note = f"re-sized to the acceptance limits: {chg}; {lay_note}"
            else:
                self.seed_note = f"passes as-is (same thicknesses); {lay_note}"
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
                    if C.min_tube - 1e-9 <= L <= C.len_special_max + 1e-9:
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
        # the fabrication rules compare the best standard, short-tube and
        # long-tube design at each segment count -> all must be verified
        ids = {id(x) for x in pool}
        by_n: Dict[int, Dict[str, List[Evaluated]]] = {}
        for x in self.found.values():
            cls = length_class(x.spec, C)['cls']
            by_n.setdefault(x.design.n, {}).setdefault(cls, []).append(x)
        for nn, d_ in by_n.items():
            for cls, xs in d_.items():
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
        """Rank fully verified designs.

        Per segment count n:
          * P = lightest standard design, Q = lightest short-tube design.
            The best design without a long tube is Q if it is at least
            NOISE_PCT lighter than P (or no P exists), else P.
          * R = lightest long-tube design. R is used only if it is at least
            X% lighter than that best design without a long tube. If no
            design without a long tube exists at n, R is used as is.
        Across segment counts, from the fewest up: a design with more
        segments replaces the current choice only if it is at least X% lighter
        per extra segment. Within these rules the lightest design wins.
        """
        C = self.C
        X = min(max(C.fab_penalty_pct, 0.0), 99.0)
        f = 1.0 - X / 100.0
        g = 1.0 - NOISE_PCT / 100.0
        cls = {id(e): length_class(e.spec, C)['cls'] for e in cands}
        by_n: Dict[int, List[Evaluated]] = {}
        for e in cands:
            by_n.setdefault(e.design.n, []).append(e)
        lightest = lambda xs: min(xs, key=lambda e: e.weight) if xs else None
        best_at: Dict[int, Evaluated] = {}
        fab_table, dropped = [], []
        has_std, has_nonlong = {}, {}
        for n, es in sorted(by_n.items()):
            P = lightest([e for e in es if cls[id(e)] == 'standard'])
            Q = lightest([e for e in es if cls[id(e)] == 'short'])
            R = lightest([e for e in es if cls[id(e)] == 'long'])
            has_std[n], has_nonlong[n] = P is not None, (P or Q) is not None
            if P and Q:
                ref = Q if Q.weight <= P.weight * g + 1e-9 else P
                d1 = (f"short-tube layout {(1 - Q.weight / P.weight) * 100:.2f}% lighter than standard "
                      + ("(>= noise tolerance) -> short-tube" if ref is Q else
                         f"(< {NOISE_PCT:g}% noise tolerance) -> standard"))
            else:
                ref = P or Q
                d1 = ("standard lengths only" if P else "short-tube layouts only" if Q else
                      "no design without a long tube")
            if R is not None:
                if ref is None:
                    pick, d2 = R, "no design without a long tube at this count -> long tube used"
                else:
                    sav = (1 - R.weight / ref.weight) * 100
                    if R.weight <= ref.weight * f + 1e-9:
                        pick, d2 = R, f"long tube saves {sav:.2f}% >= X = {X:g}% -> long tube used"
                    else:
                        pick, d2 = ref, f"long tube saves {sav:.2f}% < X = {X:g}% -> not used"
                        dropped += [e for e in es if cls[id(e)] == 'long']
            else:
                pick, d2 = ref, ""
            best_at[n] = pick
            fab_table.append({"segments": n,
                              "best standard (lb)": round(P.weight) if P else None,
                              "best short-tube (lb)": round(Q.weight) if Q else None,
                              "best long-tube (lb)": round(R.weight) if R else None,
                              "chosen at this count (lb)": round(pick.weight),
                              "decision": "; ".join(x for x in (d1, d2) if x)})
        # across segment counts
        champ = None
        for n in sorted(best_at):
            b_ = best_at[n]
            if champ is None:
                champ, note = b_, "fewest segments"
            else:
                dn = n - champ.design.n
                need = (1 - f ** dn) * 100
                sav = (1 - b_.weight / champ.weight) * 100
                if b_.weight <= champ.weight * f ** dn + 1e-9:
                    note = (f"{dn} extra segment(s) save {sav:.2f}% >= {need:.2f}% -> "
                            f"{n} segments used")
                    champ = b_
                else:
                    note = (f"{dn} extra segment(s) save {sav:.2f}% < {need:.2f}% -> "
                            f"{champ.design.n} segments kept")
            for row in fab_table:
                if row["segments"] == n:
                    row["across segment counts"] = note
        n0 = min(by_n) if by_n else 0

        def eff(e):
            """Ordering key consistent with the rules above (approximate for
            designs other than the winner, which comes from the rules)."""
            w = e.weight / f ** (e.design.n - n0)
            c = cls[id(e)]
            if c == 'long' and has_nonlong.get(e.design.n):
                w /= f
            if c == 'short' and has_std.get(e.design.n):
                w /= g
            gu = max(e.strength / C.strength_target if C.strength_target > 0 else 0.0,
                     (e.defl or 0) / C.defl_target if C.defl_target > 0 else 0.0)
            return (w, e.weight, gu)

        drop_ids = {id(e) for e in dropped}
        kept = [e for e in cands if id(e) not in drop_ids]
        common = dict(truncated=self.truncated, trunc_reason=self.trunc_reason,
                      lb_remaining=(None if self.lb_remaining == float('inf') else self.lb_remaining), n_evals=self.n_evals,
                      elapsed=time.time() - self.t0, screen_cases=self.screen_cases,
                      history=list(self.history), fab_table=fab_table,
                      baseline=self.baseline_info(), dropped=dropped)
        if not kept:
            return dict(winner=None, alternates=[], kept=[], **common)
        ranked = [champ] + sorted([e for e in kept if e is not champ], key=eff)
        wmin = min(e.weight for e in kept)
        # Alternates must be DISTINCT designs: same diameters, thicknesses
        # and segment count with only tube lengths shifted are collapsed
        # into their best-ranked member (variant count reported).
        groups: Dict[tuple, List[Evaluated]] = {}
        order = []
        for e in ranked:
            k_ = (e.design.tip, e.design.base, e.design.n, e.design.ts)
            if k_ not in groups:
                groups[k_] = []; order.append(k_)
            groups[k_].append(e)
        reps = [groups[k_][0] for k_ in order]
        variants = {id(groups[k_][0]): len(groups[k_]) - 1 for k_ in order}
        bp_rejected = []
        if C.bp_rules is not None and self.emb == 0:
            ok = []
            for e in reps:
                e.baseplate = self.baseplate_check(e)
                (ok if e.baseplate['best'] is not None else bp_rejected).append(e)
                if len(ok) > C.n_alternates:
                    break
            reps = ok
            common['bp_rejected'] = bp_rejected
            if not reps:
                return dict(winner=None, alternates=[], kept=kept, **common)
        win = reps[0]
        alts = reps[1:1 + C.n_alternates]
        self._cls, self._has_std, self._has_nonlong = cls, has_std, has_nonlong
        return dict(winner=win, winner_variants=variants[id(win)],
                    alternates=[(a, self.reason(a, win), variants[id(a)]) for a in alts],
                    kept=kept, wmin=wmin, **common)

    def baseplate_check(self, e: Evaluated) -> dict:
        """Size plate + anchor bolts for a fully verified shaft (post-check)."""
        import dataclasses
        R = dataclasses.replace(self.C.bp_rules, target=self.C.strength_target)
        bot = e.spec.segments[-1]
        return size_baseplate(e.spec.base_diameter, bot.thickness, bot.fy,
                              base_reactions(e.result.cases), R,
                              bend_radius_factor=self.C.bend_radius_factor)

    def baseline_info(self) -> dict:
        b = self.baseline_result
        w = pole_weight(self.baseline_spec)['total_weight']
        s = self.baseline_seed
        return dict(weight=w, strength=b.max_strength, defl=b.max_defl_usage,
                    gov_strength_case=b.gov_strength_case, gov_defl_case=b.gov_defl_case,
                    seed_weight=s.weight if s else None, seed_note=self.seed_note)

    def reason(self, a: Evaluated, w: Evaluated) -> str:
        """Why design a ranks below the recommended design w."""
        C = self.C
        X = min(max(C.fab_penalty_pct, 0.0), 99.0)
        dw = a.weight - w.weight
        pct = dw / w.weight * 100
        ca, cw_ = length_class(a.spec, C), length_class(w.spec, C)
        r = []
        if dw < -1e-6:
            sav = -pct
            dn = a.design.n - w.design.n
            if dn > 0:
                need = (1 - (1 - X / 100) ** dn) * 100
                r.append(f"lighter by {-dw:,.0f} lb ({sav:.2f}%) but {a.design.n} segments vs "
                         f"{w.design.n}: needs >= {need:.2f}% to add {dn} segment(s)")
            elif ca['cls'] == 'long' and cw_['cls'] != 'long':
                r.append(f"lighter by {-dw:,.0f} lb ({sav:.2f}%) but {ca['label']}: needs >= {X:g}%")
            elif ca['cls'] == 'short' and cw_['cls'] == 'standard':
                r.append(f"lighter by {-dw:,.0f} lb ({sav:.2f}%) but {ca['label']}: within the "
                         f"{NOISE_PCT:g}% noise tolerance of the standard layout")
            else:
                r.append(f"lighter by {-dw:,.0f} lb ({sav:.2f}%); ranked by the fabrication rules")
        else:
            r.append(f"heavier by {dw:,.0f} lb (+{pct:.1f}%)")
            if a.design.n != w.design.n:
                r.append(f"{a.design.n} segments vs {w.design.n}")
            if ca['cls'] != 'standard':
                r.append(ca['label'])
        r.append(f"governed by {a.gov_check} ({a.gov_case[:30]})")
        return "; ".join(r)


def summarize(e: Evaluated, C: OptConstraints) -> dict:
    s = e.spec
    lay = s.layout()
    res = {
        "total length (ft)": round(s.total_length, 3),
        "AGL height (ft)": round(s.agl_height, 3),
        "D tip (in)": s.tip_diameter, "D base (in)": round(s.base_diameter, 2),
        "taper (in/ft)": round(s.taper, 5), "segments": e.design.n,
        "tube lengths (ft)": " / ".join(f"{t['length']:g}" for t in lay),
        "thickness (in)": " / ".join(f"{t['thickness']:g}" for t in lay),
        "laps (ft)": " / ".join(f"{t['lap']:g}" for t in lay[:-1]) or "-",
        "fabricated steel (ft)": round(sum(t['length'] for t in lay), 2),
        "joints": " / ".join(t['joint_type'] for t in lay[:-1]) or "-",
        "weight (lb)": round(e.weight, 0),
        "max strength %": round(e.strength, 2),
        "max defl %": round(e.defl, 2) if e.defl is not None else None,
        "governs": e.gov_check, "gov case": e.gov_case,
        "acceptance": ("within tolerance" if (e.strength > C.strength_target + 1e-9 or
                                               (e.defl or 0) > C.defl_target + 1e-9) else "at/below target"),
        "length class": length_class(e.spec, C)['cls'],
    }
    bp = e.baseplate.get('best') if e.baseplate else None
    if bp:
        # Feasibility screen only -- full option list is in the Base Plate tab.
        res["lightest base plate + bolts (lb)"] = round(bp['total_wt'] if bp['total_wt'] is not None
                                                       else bp['plate_wt'])
    if C.max_len_by_thick:
        sym = "≥" if C.thick_len_mode == 'gte' else "="
        res["t-length limits"] = ", ".join(f"t {sym} {tk:g}\" ≤ {ml:g} ft" for tk, ml in sorted(C.max_len_by_thick.items()))
    return res
