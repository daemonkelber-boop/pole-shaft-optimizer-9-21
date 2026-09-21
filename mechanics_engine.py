"""
mechanics_engine.py

Independent ASCE/SEI 48-19 shaft strength check, for validation against
PLS-POLE's own reported results. Built and calibrated against a real
PLS-POLE export (see pole_optimizer/pls_pole_xml_parser.py).

Section property formulas for dodecagonal (12-sided) polygons are from
a published design-guide reference ("Figure B-3. Properties of
dodecagonal (12-sided) sections"), confirmed by the user, page 61 of an
unspecified source document. NOT independently re-derived from ASCE 48-19
itself — flagged here so this isn't mistaken for a code-cited formula.

Calibration findings (from validating against one real section, base of
a 105 ft pole, governing load case):
- P/A and M/S (using the dodecagon Ag/Ix/C formulas) agree with
  PLS-POLE's p_a/m_s to within ~1%.
- Fa (local buckling, Eq 5.2-8/5.2-9) agrees almost exactly with
  PLS-POLE's fa_min WHEN using PLS-POLE's own reported w/t_max. A
  simplified w = D*tan(15deg) (no inside-bend-radius correction)
  overstates w/t and understates Fa by several percent — bend radius
  matters and isn't in this engine yet (BR unknown from the XML export).
- Shear (v_q): using ONLY the transverse shear component (not the
  longitudinal component, not the resultant) reproduces PLS-POLE's v_q
  almost exactly for the one section checked. This is a genuinely
  unexplained finding, not derived from the ASCE 48-19 text itself —
  flagged as UNCONFIRMED / needs broader validation and a code citation
  before being trusted as a general rule.
"""

import math

SQRT_FY_E_CONST = None  # computed per-call, Fy varies by tube


def dodecagon_section_properties(D: float, t: float) -> dict:
    """
    D = flat-to-flat OUTSIDE diameter (in), t = wall thickness (in).
    Formulas per "Figure B-3, Properties of dodecagonal (12-sided)
    sections" (design-guide reference, not ASCE 48-19 itself).

    CALIBRATION FINDING (validated against 10 real PLS-POLE load cases):
    these formulas must be evaluated using the MID-WALL diameter
    (D - t), not the outer flat-to-flat diameter D, to match PLS-POLE's
    reported area/inertia. Using D directly overstated Ag by ~0.9% and I
    by ~2.3%, which produced usage ratios systematically ~2% LOW across
    all 10 validated cases. Using D-t brought Ag and I to within ~0.15%
    of PLS-POLE's own reported values. This matches the standard
    thin-wall tube convention (properties written about the wall
    centerline, not the outer surface) -- makes physical sense, but note
    it is a calibration finding from real-data validation, not something
    stated explicitly on the reference formula sheet itself.
    """
    D_mid = D - t
    a = math.radians(15)
    Ag = 3.22 * D_mid * t
    I = 0.411 * D_mid**3 * t
    r = 0.358 * D_mid
    Cx = 0.518 * (D_mid + t) * math.cos(a)
    Cy = 0.518 * (D_mid + t) * math.sin(a)
    C = math.sqrt(Cx**2 + Cy**2)
    Q_over_It_max = 0.631 / (D_mid * t)
    C_over_J_max = 0.622 * (D_mid + t) / (D_mid**3 * t)
    return {
        'Ag': Ag, 'I': I, 'r': r, 'C': C,
        'Q_over_It_max': Q_over_It_max, 'C_over_J_max': C_over_J_max,
    }


def local_buckling_Fa_dodecagonal(w_over_t: float, Fy: float, E: float = 29000.0) -> tuple:
    """Eq 5.2-8 / 5.2-9, dodecagonal (12-sided), bend angle = 30 deg."""
    param = w_over_t * math.sqrt(Fy / E)
    if param <= 1.41:
        return Fy, "5.2-8"
    elif param <= 2.20:
        return 1.45 * Fy * (1.0 - 0.220 * param), "5.2-9"
    else:
        return None, "out of range (>2.20) -- local buckling eq set not valid, check geometry"


def combined_stress_check(P, Mx, My, V_tran, V_long, Torsion_ftk,
                           D, t, Fy, w_over_t, shear_mode='transverse_only'):
    """
    Full independent check for one section, one load case.
    shear_mode: 'transverse_only' (calibration finding, UNCONFIRMED by
    code text) or 'resultant' (textbook interpretation of the equation).
    Returns dict of all intermediate + final values.
    """
    props = dodecagon_section_properties(D, t)
    Ag, I, C = props['Ag'], props['I'], props['C']

    p_a = P / Ag

    M_res_ftk = math.sqrt(Mx**2 + My**2)
    M_res_ink = M_res_ftk * 12
    m_s = M_res_ink * C / I

    if shear_mode == 'transverse_only':
        V_eff = abs(V_tran)
    elif shear_mode == 'resultant':
        V_eff = math.sqrt(V_tran**2 + V_long**2)
    else:
        raise ValueError("shear_mode must be 'transverse_only' or 'resultant'")
    v_q = V_eff * props['Q_over_It_max']

    Torsion_ink = Torsion_ftk * 12
    t_r = abs(Torsion_ink) * props['C_over_J_max']

    # Worst fiber: bending puts both tension and compression on the section,
    # so the axial stress always adds in magnitude. Matches PLS-POLE's
    # reported 'res' (e.g. p_a=-0.46, m_s=7.90 -> res=8.37).
    res = math.sqrt((abs(p_a) + m_s)**2 + 3 * (v_q + t_r)**2)

    Fa, eq_used = local_buckling_Fa_dodecagonal(w_over_t, Fy)
    usage = (res / Fa * 100) if Fa else None

    return {
        'p_a': p_a, 'm_s': m_s, 'v_q': v_q, 't_r': t_r, 'res': res,
        'Fa': Fa, 'Fa_equation': eq_used, 'usage': usage,
        'Ag': Ag, 'I': I, 'C': C,
    }


# --------------------------------------------------------------------------
# Perimeter-point stress check (PLS-POLE reproduction)
# --------------------------------------------------------------------------
# Stresses are evaluated at 24 points around the 12-sided section: the 12
# flat mid-points (radius D/2) and the 12 vertices (radius D/(2 cos 15)).
# Flats are oriented normal to the transverse (y) and longitudinal (x) axes,
# i.e. flat normals at 0, 30, 60 ... deg measured from +y.
#   normal stress   sigma(phi) = P/A + (Mx*y + My*x)/I      (signed)
#   shear stress    tau(phi)   = tau_max*|sin(phi - beta_V)| + tau_torsion
#   combined        res(phi)   = sqrt(sigma^2 + 3 tau^2)
# Governing = max over the 24 points. At the bending extreme fiber, shear is
# zero; at the neutral axis, bending is zero -- which is what PLS-POLE's
# per-point results show (v_q = 0 at governing rows; res = |p_a| + m_s).
# Sign convention for P: TENSION positive (PLS-POLE convention).

_PHI_FLAT = [math.radians(30 * k) for k in range(12)]
_PHI_VERT = [math.radians(15 + 30 * k) for k in range(12)]


def perimeter_stress_check(P, Mx, My, V_tran, V_long, Torsion_ftk,
                           D, t, Fy, w_over_t, shear_mode='resultant'):
    props = dodecagon_section_properties(D, t)
    Ag, I = props['Ag'], props['I']
    p_a = P / Ag
    Mx_in, My_in = Mx * 12.0, My * 12.0
    if shear_mode == 'transverse_only':
        V, beta = abs(V_tran), 0.0
    else:
        V, beta = math.hypot(V_tran, V_long), math.atan2(V_long, V_tran)
    tau_max = V * props['Q_over_It_max']
    t_r = abs(Torsion_ftk) * 12.0 * props['C_over_J_max']
    r_flat = D / 2.0
    r_vert = D / (2.0 * math.cos(math.radians(15)))

    best = None
    for phi_list, r in ((_PHI_FLAT, r_flat), (_PHI_VERT, r_vert)):
        for phi in phi_list:
            y, x = r * math.cos(phi), r * math.sin(phi)
            sig_b = (Mx_in * y + My_in * x) / I
            sig = p_a + sig_b
            tau = tau_max * abs(math.sin(phi - beta)) + t_r
            res = math.sqrt(sig ** 2 + 3.0 * tau ** 2)
            if best is None or res > best[0]:
                best = (res, sig_b, tau - t_r, phi)
    res, sig_b, v_q, phi = best
    Fa, eq_used = local_buckling_Fa_dodecagonal(w_over_t, Fy)
    usage = (res / Fa * 100) if Fa else None
    return {
        'p_a': p_a, 'm_s': abs(sig_b), 'v_q': v_q, 't_r': t_r, 'res': res,
        'Fa': Fa, 'Fa_equation': eq_used, 'usage': usage,
        'Ag': Ag, 'I': I, 'C': r_flat, 'gov_point_deg': math.degrees(phi),
    }
