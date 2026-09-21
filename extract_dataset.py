"""
extract_dataset.py

Batch-extracts a training/validation dataset from a folder of PLS-POLE XML
exports, and runs the independent ASCE 48-19 engine against every section
of every load case so agreement can be measured across many poles.

Usage:
    python3 extract_dataset.py <folder_or_files...> --out dataset.xlsx

Produces one Excel workbook with these sheets:
    poles       one row per pole - geometry, weight, governing result
    tubes       one row per tube - length, thickness, slip joint check
    sections    one row per section - geometry + PLS-POLE section properties
    checks      one row per section x load case - forces, PLS-POLE stresses,
                independent engine stresses, and the difference
    summary     agreement statistics per pole and overall

Design notes / conventions confirmed against real files:
  - Tube starts:  s_1 = 0;  s_{i+1} = s_i + L_i - lap_i
    so sum(L) - sum(lap) = pole length.
  - Slip joint required length = ceil(1.1 * [1.5 * female ID] / 0.25) * 0.25
    i.e. CEILING to the next 0.25 ft, not nearest. Verified on 4 real joints.
  - Section properties evaluated at mid-wall diameter (D - t).
  - Inside bend radius BR = 4.5t (configurable).
  - Shaft weight excludes the base plate (base plate optimization is out of
    scope; a pole with a base plate still reports plate weight separately).
"""

import sys, os, glob, math, argparse
from pls_pole_xml_parser import (parse_pls_pole_xml, get_single_table,
                                 get_load_case_instances, get_field)
from mechanics_engine import dodecagon_section_properties, local_buckling_Fa_dodecagonal

STEEL_DENSITY = 490.0   # lb/ft^3
BR_FACTOR = 4.5         # inside bend radius = BR_FACTOR * t
E_STEEL = 29000.0       # ksi

# Standard plate gauges: 3/16 in to 1 in in 1/16 in steps.
GAUGES = [0.1875 + 0.0625 * i for i in range(14)]


def snap_gauge(t: float, tol: float = 0.02):
    """Snap a derived thickness to the nearest standard gauge if it is
    close enough; otherwise return it unchanged (so genuinely odd walls
    aren't silently forced onto the grid)."""
    if t is None:
        return None
    g = min(GAUGES, key=lambda x: abs(x - t))
    return g if abs(g - t) <= tol else t


# ----------------------------------------------------------------- helpers
def slip_joint_required(one_pt_five_x_id: float) -> float:
    """1.1 x [1.5 x female ID], rounded UP to the next 0.25 ft."""
    if not one_pt_five_x_id:
        return 0.0
    return math.ceil(one_pt_five_x_id * 1.1 / 0.25) * 0.25


def w_over_t(D: float, t: float, br_factor: float = BR_FACTOR) -> float:
    """w = 0.268 * (D - t - 2*BR); BR = br_factor * t."""
    w = 0.268 * (D - t - 2 * br_factor * t)
    return w / t


def thickness_from_wt(D: float, wt: float, br_factor: float = BR_FACTOR) -> float:
    """Invert w/t to recover t. t = 0.268D / (w/t + 0.268*(1 + 2*br_factor))."""
    return 0.268 * D / (wt + 0.268 * (1 + 2 * br_factor))


def tube_layout(tubes):
    """Return list of dicts with each tube's start/end rel_dist and properties."""
    out, start = [], 0.0
    for t in sorted(tubes, key=lambda r: get_field(r, 'tube_no')):
        L = get_field(t, 'length')
        lap = get_field(t, 'lap_length', 0.0) or 0.0
        out.append({
            'no': int(get_field(t, 'tube_no')),
            'start': start, 'end': start + L,
            'L': L, 'lap': lap,
            'thk': get_field(t, 'thickness'),
            'Fy': get_field(t, 'yield_stress'),
            'D_top': get_field(t, 'tube_top_diameter'),
            'D_bot': get_field(t, 'tube_bot_diameter'),
            'taper': get_field(t, 'calculated_taper'),
            'weight': get_field(t, 'tube_weight'),
            'req_1_5xID': get_field(t, '_1_5x_diam_lap_length', 0.0) or 0.0,
            'actual_overlap': get_field(t, 'actual_overlap', 0.0) or 0.0,
        })
        start = start + L - lap
    return out


def tube_at(layout, rel_dist, tol=1e-6):
    """Which tube governs at rel_dist. In a lap region the OUTER (lower,
    female) tube governs the outside diameter, so pick the highest-numbered
    tube whose span contains rel_dist."""
    hits = [t for t in layout if t['start'] - tol <= rel_dist <= t['end'] + tol]
    if not hits:
        return min(layout, key=lambda t: min(abs(t['start'] - rel_dist),
                                             abs(t['end'] - rel_dist)))
    return max(hits, key=lambda t: t['no'])


def in_lap(layout, rel_dist, tol=1e-6):
    hits = [t for t in layout if t['start'] - tol <= rel_dist <= t['end'] + tol]
    return len(hits) > 1


# ----------------------------------------------------------------- per-file
def extract_file(path, br_factor=BR_FACTOR):
    p = parse_pls_pole_xml(path)
    fname = os.path.basename(path)

    pp = get_single_table(p, 'steel_pole_properties')[0]
    tubes_raw = get_single_table(p, 'steel_tubes_properties')
    layout = tube_layout(tubes_raw)

    try:
        bp = get_single_table(p, 'base_plate_properties')
        bp_weight = get_field(bp[0], 'plate_weight', 0.0) if bp else 0.0
        bp_dia = get_field(bp[0], 'plate_diam', None) if bp else None
        bp_thk = get_field(bp[0], 'plate_thick', None) if bp else None
    except Exception:
        bp_weight, bp_dia, bp_thk = 0.0, None, None

    summ = get_single_table(p, 'summary_of_steel_pole_usages')[0]
    vlc = get_single_table(p, 'vector_load_cases')

    tube_weight_sum = sum(t['weight'] for t in layout)
    embedded = str(pp.get('base_plate', '')).strip().lower() != 'yes'

    pole = {
        'file': fname,
        'label': pp.get('steel_pole_property_label'),
        'project': p['creator'].get('title', ''),
        'pls_version': p['creator'].get('version', ''),
        'length_ft': get_field(pp, 'length'),
        'shape': pp.get('shape'),
        'n_sides': 12 if '12' in str(pp.get('shape', '')) else None,
        'tip_dia_in': get_field(pp, 'tip_diameter'),
        'base_dia_in': get_field(pp, 'base_diameter'),
        'taper_in_per_ft': layout[0]['taper'] if layout else None,
        'n_tubes': len(layout),
        'has_base_plate': not embedded,
        'embedded': embedded,
        'embed_len_ft': get_field(pp, 'default_embedded_length', 0.0),
        'agl_height_ft': get_field(pp, 'length') - (get_field(pp, 'default_embedded_length', 0.0) or 0.0),
        'shaft_weight_lbs': tube_weight_sum,
        'base_plate_weight_lbs': bp_weight,
        'base_plate_dia_in': bp_dia,
        'base_plate_thk_in': bp_thk,
        'pls_reported_weight_lbs': get_field(summ, 'weight'),
        'pls_gov_usage_pct': get_field(summ, 'maximum_usage'),
        'pls_gov_case': summ.get('load_case'),
        'pls_gov_height_agl_ft': get_field(summ, 'height_agl'),
        'pls_gov_segment': get_field(summ, 'segment_number'),
        'n_load_cases': len(vlc),
    }
    # weight bookkeeping check
    pole['weight_recon_diff_lbs'] = (tube_weight_sum + bp_weight) - (pole['pls_reported_weight_lbs'] or 0)

    # ---- tubes ----
    tube_rows = []
    for t in layout:
        req = slip_joint_required(t['req_1_5xID'])
        tube_rows.append({
            'file': fname, 'tube_no': t['no'],
            'start_rel_ft': t['start'], 'end_rel_ft': t['end'],
            'length_ft': t['L'], 'thickness_in': t['thk'], 'Fy_ksi': t['Fy'],
            'D_top_in': t['D_top'], 'D_bot_in': t['D_bot'],
            'taper_in_per_ft': t['taper'], 'weight_lbs': t['weight'],
            'joint_type': 'flange/butt' if t['req_1_5xID'] == 0 else 'slip',
            'pls_1_5x_female_ID_ft': t['req_1_5xID'],
            'rule_required_lap_ft': req,
            'pls_actual_lap_ft': t['lap'],
            'lap_rule_match': (abs(req - t['lap']) < 1e-6) if t['req_1_5xID'] else None,
            'len_pref_flag': ('53' if abs(t['L'] - 53) < .01 else
                              '57' if abs(t['L'] - 57) < .01 else
                              '60' if abs(t['L'] - 60) < .01 else
                              'over60' if t['L'] > 60.01 else 'other'),
        })

    # ---- sections ----
    ps = [r for r in get_single_table(p, 'pole_steel_properties')
          if r.get('element_label') == 'P']
    sec_rows, geom_by_key = [], {}
    for r in ps:
        rd = get_field(r, 'rel_dist')
        D = get_field(r, 'outer_diam')
        wt_pls = get_field(r, 'w_t_max', 0.0)
        tb = tube_at(layout, rd)
        t_map = tb['thk']
        # PLS-POLE emits a separate section row per tube where tubes overlap,
        # so mapping thickness by elevation picks the wrong wall inside a lap.
        # Each row's own D and w/t identify its wall unambiguously.
        t_der_raw = thickness_from_wt(D, wt_pls, br_factor) if wt_pls else None
        t_der = snap_gauge(t_der_raw)
        t_use = t_der if t_der else t_map
        props = dodecagon_section_properties(D, t_use)
        wt_calc = w_over_t(D, t_use, br_factor) if D and t_use else None
        Fa_calc, eq = local_buckling_Fa_dodecagonal(wt_pls, tb['Fy']) if wt_pls else (None, None)

        row = {
            'file': fname, 'joint_label': r.get('joint_label'),
            'joint_position': r.get('joint_position'),
            'rel_dist_ft': rd,
            'height_agl_ft': (pole['length_ft'] - rd) - (pole['embed_len_ft'] or 0.0),
            'in_lap_region': in_lap(layout, rd),
            'tube_no': tb['no'],
            'D_in': D,
            't_used_in': t_use,
            't_mapped_in': t_map,
            't_from_wt_raw_in': t_der_raw,
            't_map_vs_used_pct': ((t_map / t_use - 1) * 100) if (t_use and t_map) else None,
            'Fy_ksi': tb['Fy'],
            'pls_area_in2': get_field(r, 'area'),
            'calc_area_in2': props['Ag'],
            'area_diff_pct': (props['Ag'] / get_field(r, 'area') - 1) * 100 if get_field(r, 'area') else None,
            'pls_I_in4': get_field(r, 't_moment_inertia'),
            'calc_I_in4': props['I'],
            'I_diff_pct': (props['I'] / get_field(r, 't_moment_inertia') - 1) * 100 if get_field(r, 't_moment_inertia') else None,
            'pls_wt': wt_pls, 'calc_wt': wt_calc,
            'wt_diff': (wt_calc - wt_pls) if (wt_calc and wt_pls) else None,
            'pls_Fa_ksi': get_field(r, 'fa_min'),
            'calc_Fa_ksi': Fa_calc,
            'Fa_diff_pct': (Fa_calc / get_field(r, 'fa_min') - 1) * 100 if (Fa_calc and get_field(r, 'fa_min')) else None,
            'Fa_equation': eq,
            'pls_moment_cap_ftk': get_field(r, 't_moment_capacity'),
        }
        sec_rows.append(row)
        # Key by (rel_dist, D) so overlapping male/female rows stay distinct.
        geom_by_key[(round(rd, 2), round(D, 2))] = row

    # ---- checks: every section x every load case ----
    # The usage table and the section table list the same sections in the
    # same order (verified: equal row counts per file, including poles with
    # overlapping male/female rows at slip joints). Labels can NOT be used
    # to pair them - PLS-POLE names the same location differently in each
    # table (e.g. 'SpliceT' vs '#P:8') - so pair by index and assert that
    # rel_dist agrees.
    usages = get_load_case_instances(p, 'detailed_steel_pole_usages')
    chk_rows = []
    pair_mismatch = 0
    for case, rows in usages.items():
        if len(rows) != len(sec_rows):
            pair_mismatch += 1
            continue
        for idx, u in enumerate(rows):
            g = sec_rows[idx]
            rd = get_field(u, 'rel_dist')
            if abs(rd - g['rel_dist_ft']) > 0.01:
                pair_mismatch += 1
                continue
            D, t, Fy = g['D_in'], g['t_used_in'], g['Fy_ksi']
            if not D or not t:
                continue
            props = dodecagon_section_properties(D, t)
            P = get_field(u, 'axial_force', 0.0)
            Mx = get_field(u, 'trans_mom_local_mx', 0.0)
            My = get_field(u, 'long_mom_local_my', 0.0)
            Vt = get_field(u, 'tran_shear', 0.0)
            Vl = get_field(u, 'long_shear', 0.0)
            T = get_field(u, 'tors_mom', 0.0)

            p_a = P / props['Ag']
            M_res = math.sqrt(Mx ** 2 + My ** 2)
            m_s = M_res * 12 * props['C'] / props['I']
            v_q = abs(Vt) * props['Q_over_It_max']
            t_r = abs(T) * 12 * props['C_over_J_max']
            res = math.sqrt((p_a + m_s) ** 2 + 3 * (v_q + t_r) ** 2)
            Fa, _ = local_buckling_Fa_dodecagonal(g['pls_wt'], Fy) if g['pls_wt'] else (Fy, None)
            allow = min(Fy, Fa) if Fa else Fy
            usage = res / allow * 100

            pls_usage = get_field(u, 'max_usage', None)
            chk_rows.append({
                'file': fname, 'load_case': case,
                'joint_label': u.get('joint_label'), 'rel_dist_ft': rd,
                'height_agl_ft': g['height_agl_ft'],
                'in_lap_region': g['in_lap_region'], 'tube_no': g['tube_no'],
                'D_in': D, 't_in': t,
                'P_kips': P, 'Mx_ftk': Mx, 'My_ftk': My,
                'V_tran_kips': Vt, 'V_long_kips': Vl, 'T_ftk': T,
                'pls_p_a': get_field(u, 'p_a'), 'calc_p_a': p_a,
                'pls_m_s': get_field(u, 'm_s'), 'calc_m_s': m_s,
                'pls_v_q': get_field(u, 'v_q'), 'calc_v_q': v_q,
                'pls_t_r': get_field(u, 't_r'), 'calc_t_r': t_r,
                'pls_res': get_field(u, 'res'), 'calc_res': res,
                'allow_ksi': allow,
                'pls_usage_pct': pls_usage, 'calc_usage_pct': usage,
                'usage_diff_pts': (usage - pls_usage) if pls_usage is not None else None,
            })
    pole['pairing_mismatches'] = pair_mismatch

    return pole, tube_rows, sec_rows, chk_rows


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default='pls_dataset.xlsx')
    ap.add_argument('--br', type=float, default=BR_FACTOR)
    args = ap.parse_args()

    files = []
    for pth in args.paths:
        if os.path.isdir(pth):
            files += sorted(glob.glob(os.path.join(pth, '*.xml')))
        else:
            files.append(pth)
    files = [f for f in files if f.lower().endswith('.xml')]
    if not files:
        print("No XML files found.")
        return

    poles, tubes, sections, checks = [], [], [], []
    for f in files:
        try:
            po, tu, se, ch = extract_file(f, args.br)
            poles.append(po); tubes += tu; sections += se; checks += ch
            print(f"  parsed {os.path.basename(f):<40} {len(se):>3} sections, "
                  f"{po['n_load_cases']:>3} cases, {len(ch):>5} checks")
        except Exception as e:
            print(f"  FAILED {os.path.basename(f)}: {type(e).__name__}: {e}")

    try:
        import pandas as pd
    except ImportError:
        print("pandas required for Excel output")
        return

    dfp, dft = pd.DataFrame(poles), pd.DataFrame(tubes)
    dfs, dfc = pd.DataFrame(sections), pd.DataFrame(checks)

    # summary per pole
    rows = []
    for f, grp in dfc.groupby('file'):
        v = grp['usage_diff_pts'].dropna()
        hi = grp[grp['pls_usage_pct'] > 50]['usage_diff_pts'].dropna()
        sg = dfs[dfs['file'] == f]
        rows.append({
            'file': f, 'n_checks': len(grp),
            'usage_diff_mean_pts': v.mean(), 'usage_diff_min_pts': v.min(),
            'usage_diff_max_pts': v.max(), 'usage_diff_absmax_pts': v.abs().max(),
            'usage_diff_mean_over50pct': hi.mean() if len(hi) else None,
            'area_diff_mean_pct': sg['area_diff_pct'].dropna().mean(),
            'I_diff_mean_pct': sg['I_diff_pct'].dropna().mean(),
            'Fa_diff_absmax_pct': sg['Fa_diff_pct'].dropna().abs().max(),
            'wt_diff_absmax': sg['wt_diff'].dropna().abs().max(),
            't_map_vs_used_absmax_pct': sg['t_map_vs_used_pct'].dropna().abs().max(),
        })
    dfsum = pd.DataFrame(rows)

    with pd.ExcelWriter(args.out, engine='openpyxl') as xl:
        dfp.to_excel(xl, sheet_name='poles', index=False)
        dft.to_excel(xl, sheet_name='tubes', index=False)
        dfs.to_excel(xl, sheet_name='sections', index=False)
        dfc.to_excel(xl, sheet_name='checks', index=False)
        dfsum.to_excel(xl, sheet_name='summary', index=False)

    print(f"\nWrote {args.out}")
    print(f"  poles={len(dfp)}  tubes={len(dft)}  sections={len(dfs)}  checks={len(dfc)}")
    return dfp, dft, dfs, dfc, dfsum


if __name__ == '__main__':
    main()
