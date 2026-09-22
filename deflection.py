"""
deflection.py  --  Phase 3, module 2

Geometrically nonlinear (large-displacement, P-delta) deflection solver for a
tapered 12-sided steel pole, loaded by the LoadModel from loads.py.

Method
    The pole is a cantilever, fixed at the base (base-plate pole) or at the
    ground line (embedded pole -- see ASSUMPTIONS). Nodes are placed every
    `ds` ft along the axis. Each iteration:
      1. Place every load at its DEFLECTED position (shaft element loads at
         the deflected centreline, attachment loads at centreline + arm/vang
         offset rotated with the local pole tilt).
      2. At every node, take the moment of all loads ABOVE it about the
         deflected node position -> bending moment in each plane.
      3. Curvature = M / (E I(s)). Integrate from the fixed end to get the
         slope angle, then position via sin/cos of the slope angle
         (exact for large in-plane rotation, no small-angle assumption).
      4. Repeat until the tip moves less than `tol` in.
    Load directions are fixed in space (wind and wire loads do not follow
    the pole). Axial shortening P/EA is included in the vertical deflection.

Returned section forces are the second-order (P-delta) forces, directly
comparable with PLS-POLE's 'detailed_steel_pole_usages' moments.

Stiffness
    I(s) from geometry.dodecagon_section_properties at mid-wall diameter.
    Inside a slip-joint lap, two walls are present. `lap_stiffness`:
      'outer'  -- female (outer) tube only  [default until validated]
      'sum'    -- female + male tube I summed (full composite)
      'inner'  -- male (inner) tube only (most flexible)
    E = 29,000 ksi unless the XML overrides it.

ASSUMPTIONS (flagged, not yet validated)
    * Embedded poles are taken as FIXED AT THE GROUND LINE (rigid
      foundation). PLS-POLE may model soil rotation differently. Only
      base-plate pole 003 has been validated so far.
    * Arm / vang offsets are rigid (no arm flexibility), per loads.py.
    * Torsion does not couple into bending.

Deflection check
    Limits from the XML load case table ('pole_deflection_check' /
    'pole_deflection_limit_or'). '% Pole Height' uses ABOVE-GROUND height.
    The checked value is the tip deflection from the vertical axis,
    sqrt(dx^2 + dy^2), matching PLS-POLE's 'deflection_from_vertical_axis'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from geometry import dodecagon_section_properties, E_STEEL_KSI
from loads import LoadModel, tubes_at


@dataclass
class DeflectionResult:
    case: str
    s: np.ndarray            # ft below tip, node positions (tip -> fixed end)
    x: np.ndarray            # ft, longitudinal deflection
    y: np.ndarray            # ft, transverse deflection
    z: np.ndarray            # ft, elevation of deflected node above ground line
    dz: np.ndarray           # ft, vertical deflection (+ up)
    thx: np.ndarray          # rad, tilt in x-z plane
    thy: np.ndarray          # rad, tilt in y-z plane
    Mx: np.ndarray           # ft-k, transverse bending (drives y)
    My: np.ndarray           # ft-k, longitudinal bending (drives x)
    P: np.ndarray            # kips, axial (+ compression)
    Vy: np.ndarray           # kips
    Vx: np.ndarray           # kips
    T: np.ndarray            # ft-k
    iterations: int
    converged: bool
    agl_height: float
    # final load state (sorted by key), for exact section forces anywhere
    _key: np.ndarray = None
    _F: np.ndarray = None
    _LX: np.ndarray = None
    _LY: np.ndarray = None
    _LZ: np.ndarray = None

    # --- tip values, inches ---
    @property
    def tip_trans_in(self) -> float:
        return float(self.y[0] * 12.0)

    @property
    def tip_long_in(self) -> float:
        return float(self.x[0] * 12.0)

    @property
    def tip_vert_in(self) -> float:
        return float(self.dz[0] * 12.0)

    @property
    def tip_horiz_ft(self) -> float:
        return float(math.hypot(self.x[0], self.y[0]))

    def section_forces(self, s0: float, include_loads_at_s0: bool = True) -> dict:
        """Exact second-order section forces at s0 ft below the tip.

        A point load applied exactly at s0 is included when
        include_loads_at_s0 is True (PLS-POLE 'Origin' row, just below the
        attachment) and excluded when False ('End' row, just above it).
        """
        eps = 1e-6 if include_loads_at_s0 else -1e-6
        m = self._key < s0 + eps
        x0 = float(np.interp(s0, self.s, self.x))
        y0 = float(np.interp(s0, self.s, self.y))
        z0 = float(np.interp(s0, self.s, self.z))
        Fx, Fy, Fz = self._F[m, 0], self._F[m, 1], self._F[m, 2]
        rx, ry, rz = self._LX[m] - x0, self._LY[m] - y0, self._LZ[m] - z0
        SFx, SFy, SFz = float(np.sum(Fx)), float(np.sum(Fy)), float(np.sum(Fz))
        ty = float(np.interp(s0, self.s, self.thy))
        tx = float(np.interp(s0, self.s, self.thx))
        return dict(Mx=float(np.sum(Fy * rz - Fz * ry)),
                    My=float(np.sum(Fx * rz - Fz * rx)),
                    T=float(np.sum(rx * Fy - ry * Fx)),
                    P=-SFz, Vx=SFx, Vy=SFy,
                    # Local (tilted section) frame -- PLS-POLE convention:
                    # axial TENSION positive, shear perpendicular to axis.
                    axial_local=SFz * math.cos(ty) * math.cos(tx)
                                + SFy * math.sin(ty) + SFx * math.sin(tx),
                    Vy_local=SFy * math.cos(ty) - SFz * math.sin(ty),
                    Vx_local=SFx * math.cos(tx) - SFz * math.sin(tx))

    def at(self, s0: float) -> dict:
        """Interpolated results at s0 ft below the tip."""
        s = self.s
        f = lambda a: float(np.interp(s0, s, a))
        return dict(x_in=f(self.x) * 12, y_in=f(self.y) * 12, z_in=f(self.dz) * 12,
                    Mx=f(self.Mx), My=f(self.My), P=f(self.P),
                    Vx=f(self.Vx), Vy=f(self.Vy), T=f(self.T))


def lap_zone(spec, s):
    for tb in spec.layout():
        if tb['lap_top'] is not None and tb['lap_top'] - 1e-9 <= s <= tb['lap_bottom'] + 1e-9:
            return tb['lap_top'], tb['lap_bottom']
    return s, s


def _stiffness(spec, s_nodes, E_ksi, lap_stiffness):
    """EI (kip-ft^2) and EA (kips) at each node."""
    EI = np.zeros(len(s_nodes))
    EA = np.zeros(len(s_nodes))
    for i, s in enumerate(s_nodes):
        tb = tubes_at(spec, s)          # [(D, t)] for every tube present
        if len(tb) > 1:
            if lap_stiffness == 'sum':
                use = tb
            elif lap_stiffness == 'inner':
                use = [min(tb, key=lambda d: d[0])]
            elif lap_stiffness == 'stiffer':
                use = [max(tb, key=lambda d: d[0] ** 3 * d[1])]
            elif lap_stiffness == 'midpoint':
                # female above the lap mid-point, male below it
                lo, hi = lap_zone(spec, s)
                use = [max(tb, key=lambda d: d[0])] if s < 0.5 * (lo + hi) else \
                      [min(tb, key=lambda d: d[0])]
            else:
                use = [max(tb, key=lambda d: d[0])]
        else:
            use = tb
        I_in4 = sum(dodecagon_section_properties(d, t)['I'] for d, t in use)
        A_in2 = sum(dodecagon_section_properties(d, t)['Ag'] for d, t in use)
        EI[i] = E_ksi * I_in4 / 144.0      # kip-ft^2
        EA[i] = E_ksi * A_in2              # kips
    return EI, EA


def solve_deflection(model: LoadModel, ds: float = 0.25,
                     E_ksi: float = E_STEEL_KSI,
                     lap_stiffness: str = 'midpoint',
                     max_iter: int = 50, tol_in: float = 0.001,
                     second_order: bool = True) -> DeflectionResult:
    spec = model.spec
    H = spec.total_length
    s_fix = spec.groundline_rel              # base (FDN) or ground line (EMB)
    n = max(2, int(round(s_fix / ds)))
    s_nodes = np.linspace(0.0, s_fix, n + 1)  # tip -> fixed end
    u = s_fix - s_nodes                        # height above fixed end
    EI, EA = _stiffness(spec, s_nodes, E_ksi, lap_stiffness)

    # Loads acting on the free (above-fixity) part only.
    els = [e for e in model.elements if e.s_mid < s_fix]
    e_s = np.array([e.s_mid for e in els])
    e_F = np.array([[e.fx, e.fy, -e.fz] for e in els])   # global, z up
    pts = model.points
    p_s = np.array([p.s for p in pts]) if pts else np.zeros(0)
    p_F = np.array([[p.Fx, p.Fy, -p.Fz] for p in pts]) if pts else np.zeros((0, 3))
    p_off = np.array([[p.dx, p.dy, p.dz] for p in pts]) if pts else np.zeros((0, 3))

    # Undeformed state
    x = np.zeros_like(s_nodes); y = np.zeros_like(s_nodes)
    z = u.copy(); thx = np.zeros_like(s_nodes); thy = np.zeros_like(s_nodes)

    # Combined load arrays, sorted by the key that decides "above the
    # section": element loads count when s_mid < s0; point loads count when
    # s_point <= s0 (a load exactly at a node acts on that node's section).
    key = np.concatenate([e_s, p_s - 1e-7])
    order = np.argsort(key, kind='stable')
    key = key[order]
    F_all = np.vstack([e_F, p_F])[order]
    n_e = len(e_s)
    is_pt = (order >= n_e)
    pt_idx = order[is_pt] - n_e
    idx = np.searchsorted(key, s_nodes, side='left')   # count of loads above
    ip = lambda arr, q: np.interp(q, s_nodes, arr)

    def csum(v):
        c = np.concatenate([[0.0], np.cumsum(v)])
        return c[idx]

    it, converged = 0, False
    for it in range(1, max_iter + 1):
        # --- load positions in the current (deflected) shape
        if second_order:
            ex, ey, ez = ip(x, e_s), ip(y, e_s), ip(z, e_s)
            X0, Y0, Z0 = x, y, z
        else:
            ex, ey, ez = np.zeros(n_e), np.zeros(n_e), s_fix - e_s
            X0, Y0, Z0 = np.zeros_like(u), np.zeros_like(u), u
        if len(pts):
            if second_order:
                px0, py0, pz0 = ip(x, p_s), ip(y, p_s), ip(z, p_s)
                ax, ay = ip(thx, p_s), ip(thy, p_s)
            else:
                px0, py0, pz0 = np.zeros(len(pts)), np.zeros(len(pts)), s_fix - p_s
                ax = ay = np.zeros(len(pts))
            # rotate rigid arm/vang offsets with the local pole tilt
            ox = p_off[:, 0] * np.cos(ax) + p_off[:, 2] * np.sin(ax)
            oy = p_off[:, 1] * np.cos(ay) + p_off[:, 2] * np.sin(ay)
            oz = (-p_off[:, 1] * np.sin(ay) - p_off[:, 0] * np.sin(ax)
                  + p_off[:, 2] * np.cos(ay) * np.cos(ax))
            px, py, pz = px0 + ox, py0 + oy, pz0 + oz
        else:
            px = py = pz = np.zeros(0)
        LX = np.concatenate([ex, px])[order]
        LY = np.concatenate([ey, py])[order]
        LZ = np.concatenate([ez, pz])[order]
        Fx, Fy, Fz = F_all[:, 0], F_all[:, 1], F_all[:, 2]   # Fz + up

        SFx, SFy, SFz = csum(Fx), csum(Fy), csum(Fz)
        # bending that curves the pole toward +y (Mx) and +x (My)
        Mx = csum(Fy * LZ) - Z0 * SFy - csum(Fz * LY) + Y0 * SFz
        My = csum(Fx * LZ) - Z0 * SFx - csum(Fz * LX) + X0 * SFz
        T = csum(LX * Fy) - X0 * SFy - csum(LY * Fx) + Y0 * SFx
        P, Vx, Vy = -SFz, SFx, SFy

        # --- integrate curvature from the fixed end up to the tip
        ky, kx = Mx / EI, My / EI
        ey_ax = P / EA                       # axial strain (+ shortening)
        nthx = np.zeros_like(s_nodes); nthy = np.zeros_like(s_nodes)
        nx = np.zeros_like(s_nodes); ny = np.zeros_like(s_nodes)
        nz = np.zeros_like(s_nodes)
        for j in range(n - 1, -1, -1):       # node j is above node j+1
            h = s_nodes[j + 1] - s_nodes[j]
            nthy[j] = nthy[j + 1] + 0.5 * (ky[j] + ky[j + 1]) * h
            nthx[j] = nthx[j + 1] + 0.5 * (kx[j] + kx[j + 1]) * h
            tmy = 0.5 * (nthy[j] + nthy[j + 1])
            tmx = 0.5 * (nthx[j] + nthx[j + 1])
            hl = h * (1.0 - 0.5 * (ey_ax[j] + ey_ax[j + 1]))
            ny[j] = ny[j + 1] + hl * math.sin(tmy)
            nx[j] = nx[j + 1] + hl * math.sin(tmx)
            nz[j] = nz[j + 1] + hl * math.cos(tmy) * math.cos(tmx)

        change = max(abs(ny[0] - y[0]), abs(nx[0] - x[0]), abs(nz[0] - z[0])) * 12
        x, y, z, thx, thy = nx, ny, nz, nthx, nthy
        if not second_order:
            converged = True
            break
        if change < tol_in:
            converged = True
            break

    return DeflectionResult(case=model.case.name, s=s_nodes, x=x, y=y, z=z,
                            dz=z - u, thx=thx, thy=thy, Mx=Mx, My=My, P=P, Vy=Vy, Vx=Vx, T=T,
                            iterations=it, converged=converged,
                            agl_height=spec.agl_height,
                            _key=key, _F=F_all, _LX=LX, _LY=LY, _LZ=LZ)


# --------------------------------------------------------------------------
# Deflection limit check
# --------------------------------------------------------------------------

def deflection_check(res: DeflectionResult, check_type: str,
                     limit: float) -> Optional[dict]:
    """Return a usage dict, or None when the case has no deflection limit.

    check_type values seen in PLS-POLE: 'No Limit', '% Pole Height'.
    Anything else is returned with usage=None and flagged, rather than
    guessed at.
    """
    if not check_type or check_type.strip().lower() == 'no limit' or not limit:
        return None
    actual_ft = res.tip_horiz_ft
    ct = check_type.strip().lower()
    if ct == '% pole height':
        allow_ft = limit / 100.0 * res.agl_height
        return dict(check=check_type, limit=limit, allowable_ft=allow_ft,
                    actual_ft=actual_ft, usage=actual_ft / allow_ft * 100.0)
    return dict(check=check_type, limit=limit, allowable_ft=None,
                actual_ft=actual_ft, usage=None,
                note=f"check type '{check_type}' not implemented")
