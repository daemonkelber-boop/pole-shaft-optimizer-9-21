"""Validate baseplate.py against PLS-POLE: every row of
'Base Plate Results by Bend Line' + the 6.4.2 override.
Usage: python test_baseplate.py <base-plate PLS-POLE export.xml>"""
import math, sys
import numpy as np
from pls_pole_xml_parser import parse_pls_pole_xml
from baseplate import (BasePlateRules, bolt_loads, wedge_matrix, half_capacity_override,
                       bolt_circle, hole_diameter)

p = parse_pls_pole_xml(sys.argv[1])
g = lambda t: [r for inst in p['tables'][t] for r in inst['rows']]
v = lambda r, k: float(r[k]['value'] if isinstance(r[k], dict) else r[k])
bp = g('base_plate_properties')[0]
n, bc, D = int(bp['num_of_bolts']), v(bp, 'bolt_pattern_diam'), None
D = v(g('pole_steel_properties')[-1], 'outer_diam')
t, Fy, w_plate = v(bp, 'plate_thick'), v(bp, 'steel_yield_stress'), v(bp, 'plate_weight') / 1000
Mcap = v(g('pole_steel_properties')[-1], 't_moment_capacity')
ang0 = v([r for r in g('base_plate') if 'bolt_angle' in r][0], 'bolt_angle')
R = BasePlateRules(start_deg=ang0)
summ = {r['load_case']: r for r in g('summary_of_base_plate_usages_by_load_case')}
jr = {i['titledetail']: i['rows'][0] for i in p['tables']['joint_support_reactions']}
b = D * math.tan(math.radians(15))
W = wedge_matrix(n, bc, D, R)
e_m = e_u = e_t = e_ov = 0.0; nrow = 0
for r in g('base_plate_results_by_bend_line'):
    s = summ[r['load_case']]
    BL = bolt_loads(n, bc, [v(s, 'vertical_load')], [v(s, 'x_moment')], [v(s, 'y_moment')], R)[0]
    mx = (v(r, 'start_x') + v(r, 'end_x')) / 2; my = (v(r, 'start_y') + v(r, 'end_y')) / 2
    k = int(round(((math.degrees(math.atan2(my, mx)) - 90) % 360) / 30)) % 12
    M = float(np.abs(BL) @ W[k])
    e_m = max(e_m, abs(M / 12 - v(r, 'bolt_mom_sum')))
    e_u = max(e_u, abs(6 * M / (b * t * t) / Fy * 100 - float(r['usage'])))
    e_t = max(e_t, abs(math.sqrt(6 * M / (b * Fy)) - v(r, 'min_plate_thickness')))
    nrow += 1
for lc, s in summ.items():
    j = jr[lc]
    mx, my, _ = half_capacity_override([v(j, 'x_moment')], [v(j, 'y_moment')], Mcap)
    e_ov = max(e_ov, abs(mx[0] - v(s, 'x_moment')), abs(my[0] - v(s, 'y_moment')))
print(f"{nrow} bend-line rows: max |dM| {e_m:.4f} ft-k, |dusage| {e_u:.4f} %, |dt_min| {e_t:.4f} in")
print(f"6.4.2 override: max |dM| {e_ov:.3f} ft-k (reaction rounding)")
print(f"rules: dBC needs t_pole -- XML dBC {bc}, hole rule {hole_diameter(D)} vs XML {v(bp, 'hole_diam')}")
assert e_m < 0.01 and e_u < 0.02 and e_t < 0.002 and e_ov < 1.0
print("PASS")
