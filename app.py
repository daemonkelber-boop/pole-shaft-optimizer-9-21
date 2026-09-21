import streamlit as st
import tempfile, os, hashlib, traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pole Shaft Optimizer", layout="wide")
st.title("🗼 Pole Shaft Optimizer")
st.caption("ASCE 48-19 | 12-sided tapered polygonal steel poles")

# ── SIDEBAR: upload + analysis settings ────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload XML")
    uploaded = st.file_uploader("PLS-POLE XML export", type=["xml"])
    if uploaded:
        st.success(f"Loaded: {uploaded.name}")
    else:
        st.info("Upload a file to begin.")

    st.header("⚙️ Analysis settings")
    br_label = st.radio("Inside bend radius", ["4.5t (matches PLS-POLE)", "4.0t (ASCE 48-19 cap)"],
                        index=0, help="Used for flat width w and w/t -> local buckling Fa.")
    BR = 4.5 if br_label.startswith("4.5") else 4.0
    SHEAR = st.radio("Shear for stress check", ["resultant", "transverse_only"], index=0,
                     help="Shear direction used at the perimeter points. Identical on "
                          "tangent structures with no longitudinal load.")
    LAP = st.radio("Slip-joint lap stiffness (deflection)", ["outer", "inner", "sum"], index=0,
                   help="outer = female tube only (default, validated on 003 to +0.3%); "
                        "inner = male only; sum = both walls composite.")

# ── Cached helpers ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def save_xml(data: bytes) -> str:
    h = hashlib.md5(data).hexdigest()
    path = os.path.join(tempfile.gettempdir(), f"pls_{h}.xml")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path

@st.cache_resource(show_spinner=False)
def load_parsed(path):
    from pls_pole_xml_parser import parse_pls_pole_xml
    return parse_pls_pole_xml(path)

@st.cache_resource(show_spinner=False)
def load_baseline(path):
    from loads import Baseline
    return Baseline.from_xml(path)

def build_spec(p):
    from pls_pole_xml_parser import get_field as gf, get_single_table as gst
    from geometry import PoleSpec, Segment
    prop  = gst(p, 'steel_pole_properties')[0]
    tubes = gst(p, 'steel_tubes_properties')
    conn  = gst(p, 'steel_pole_connectivity')
    is_bp = str(prop.get('base_plate', '')).lower() == 'yes'
    emb = 0.0
    if not is_bp and conn:
        emb = gf(conn[0], 'embed_override') or gf(prop, 'default_embedded_length') or 0.0
    segs = []
    for t in tubes:
        lap = gf(t, 'lap_length') or 0.0
        segs.append(Segment(length=gf(t, 'length'), thickness=gf(t, 'thickness'),
                            fy=gf(t, 'yield_stress') or 65.0,
                            joint_type='slip' if lap > 0 else 'flange'))
    return PoleSpec(label=prop.get('steel_pole_property_label', ''),
                    tip_diameter=gf(prop, 'tip_diameter'),
                    taper=gf(tubes[0], 'calculated_taper'),
                    segments=segs, embedment=emb)

@st.cache_resource(show_spinner=False)
def run_candidate(path, br, shear, lap):
    from strength import evaluate_candidate
    p = load_parsed(path)
    return evaluate_candidate(build_spec(p), load_baseline(path),
                              bend_radius_factor=br, shear_mode=shear,
                              lap_stiffness=lap, spacing=1.0)

def show_error(e):
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())

def gov_index(cases, name):
    return cases.index(name) if name in cases else 0

tmp_path = save_xml(uploaded.getvalue()) if uploaded else None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 1 · XML Parser", "📐 2 · Geometry & Weight", "💨 3 · Loads",
    "🔩 4 · Strength Check", "📏 5 · Deflection", "⚖️ 6 · Optimizer",
])

# ── TAB 1: XML PARSER ──────────────────────────────────────────────────────
with tab1:
    st.header("PLS-POLE XML Parser")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import get_single_table, get_field, LOAD_CASE_TAGNAMES
            p = load_parsed(tmp_path)

            st.subheader("File metadata")
            st.json(p['creator'])

            st.subheader("Table inventory")
            inv = []
            for tagname, instances in sorted(p['tables'].items()):
                kind = ("per-load-case" if tagname in LOAD_CASE_TAGNAMES
                        else "repeated" if any(i['titledetail'] for i in instances)
                        else "single")
                inv.append({"table": tagname, "instances": len(instances),
                            "total rows": sum(i['nrows_actual'] for i in instances),
                            "kind": kind})
            st.dataframe(pd.DataFrame(inv), width="stretch", hide_index=True)

            st.subheader("Pole properties")
            prop1 = get_single_table(p, 'steel_pole_properties')
            if prop1:
                st.json({k: v for k, v in prop1[0].items()})

            st.subheader("Tube properties")
            tubes1 = get_single_table(p, 'steel_tubes_properties')
            if tubes1:
                st.dataframe(pd.DataFrame([{
                    "tube #": get_field(t, 'tube_no'),
                    "length (ft)": get_field(t, 'length'),
                    "thickness (in)": get_field(t, 'thickness'),
                    "D top (in)": get_field(t, 'tube_top_diameter'),
                    "D bot (in)": get_field(t, 'tube_bot_diameter'),
                    "taper (in/ft)": get_field(t, 'calculated_taper'),
                    "lap (ft)": get_field(t, 'lap_length') or 0.0,
                    "weight (lb)": get_field(t, 'tube_weight'),
                    "Fy (ksi)": get_field(t, 'yield_stress'),
                } for t in tubes1]), width="stretch", hide_index=True)

            st.subheader("Load cases")
            lcs1 = get_single_table(p, 'vector_load_cases')
            if lcs1:
                st.dataframe(pd.DataFrame([{
                    "load case": r.get('load_case_description'),
                    "DLF": get_field(r, 'dead_load_factor'),
                    "q_trans (psf)": get_field(r, 'trans_wind_pressure'),
                    "q_long (psf)": get_field(r, 'longit_wind_pressure'),
                    "ice t (in)": get_field(r, 'ice_thick') or 0.0,
                    "defl check": r.get('pole_deflection_check'),
                    "defl limit": get_field(r, 'pole_deflection_limit_or'),
                } for r in lcs1]), width="stretch", hide_index=True)

            st.subheader("Governing usage summary")
            summ1 = get_single_table(p, 'summary_of_steel_pole_usages')
            if summ1:
                st.json({k: v for k, v in summ1[0].items()})
        except Exception as e:
            show_error(e)

# ── TAB 2: GEOMETRY & WEIGHT ───────────────────────────────────────────────
with tab2:
    st.header("Geometry & Weight")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from geometry import DesignLimits, build_sections, validate, describe
            from weight import pole_weight

            spec2 = build_spec(load_parsed(tmp_path))
            limits = DesignLimits(bend_radius_factor=BR)
            lay2 = spec2.layout()

            st.subheader("Pole summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total length", f"{spec2.total_length:.2f} ft")
            c2.metric("AGL height", f"{spec2.agl_height:.2f} ft")
            c3.metric("Tip D", f"{spec2.tip_diameter:.2f} in")
            c4.metric("Base D", f"{spec2.base_diameter:.2f} in")
            c5.metric("Taper", f"{spec2.taper:.5f} in/ft")

            st.subheader("Tube layout")
            st.dataframe(pd.DataFrame([{
                "tube #": tb['tube_no'], "start (ft)": round(tb['start'], 3),
                "end (ft)": round(tb['end'], 3), "length (ft)": round(tb['length'], 3),
                "thickness (in)": tb['thickness'], "D top (in)": round(tb['d_top'], 3),
                "D bot (in)": round(tb['d_bot'], 3), "joint": tb['joint_type'],
                "lap (ft)": round(tb['lap'], 3),
            } for tb in lay2]), width="stretch", hide_index=True)

            st.subheader("Weight breakdown")
            w2 = pole_weight(spec2)
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Total shaft weight", f"{w2['total_weight']:,.0f} lb")
            wc2.metric("Above ground", f"{w2['above_ground_weight']:,.0f} lb")
            wc3.metric("Embedded steel", f"{w2['embedded_weight']:,.0f} lb")
            st.dataframe(pd.DataFrame([{
                "tube #": pt['tube_no'], "length (ft)": round(pt['length'], 3),
                "thickness (in)": pt['thickness'], "D top (in)": round(pt['d_top'], 3),
                "D bot (in)": round(pt['d_bot'], 3), "lap (ft)": round(pt['lap'], 3),
                "weight (lb)": round(pt['weight'], 1),
            } for pt in w2['per_tube']]), width="stretch", hide_index=True)

            st.subheader("Section properties table")
            spacing_val = st.slider("Section spacing (ft)", 1.0, 10.0, 5.0, 0.5)
            secs2 = build_sections(spec2, spacing=spacing_val, bend_radius_factor=BR)
            st.caption(f"w/t computed with inside bend radius = {BR}t")
            st.dataframe(pd.DataFrame([{
                "rel_dist (ft)": round(s['rel_dist'], 3), "ht AGL (ft)": round(s['height_agl'], 3),
                "tube #": s['tube_no'], "in lap": s['in_lap'], "D (in)": round(s['D'], 3),
                "t (in)": s['t'], "w/t": round(s['w_over_t'], 2), "Ag (in²)": round(s['Ag'], 3),
                "I (in⁴)": round(s['I'], 1), "C (in)": round(s['C'], 3),
                "S (in³)": round(s['S'], 3), "below GL": s['below_groundline'],
            } for s in secs2]), width="stretch", hide_index=True)

            hgts2 = [spec2.groundline_rel - s['rel_dist'] for s in secs2]
            cA, cB = st.columns(2)
            with cA:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot([s['D'] for s in secs2], hgts2, color='steelblue', linewidth=2)
                ax.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
                first = True
                for tb in lay2:
                    if tb['lap'] > 0:
                        ax.axhspan(spec2.groundline_rel - tb['end'],
                                   spec2.groundline_rel - (tb['end'] - tb['lap']),
                                   color='orange', alpha=0.3, label='Slip lap' if first else None)
                        first = False
                ax.set_xlabel("Flat-to-flat diameter (in)"); ax.set_ylabel("Height AGL (ft)")
                ax.set_title("Diameter vs height"); ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig); plt.close(fig)
            with cB:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.plot([s['w_over_t'] for s in secs2], hgts2, color='darkorange', linewidth=2)
                ax2.axvline(limits.max_wt, color='red', linestyle='--', linewidth=1,
                            label=f'Limit = {limits.max_wt}')
                ax2.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
                ax2.set_xlabel(f"w/t (BR = {BR}t)"); ax2.set_ylabel("Height AGL (ft)")
                ax2.set_title("w/t vs height"); ax2.legend(); ax2.grid(True, alpha=0.3)
                st.pyplot(fig2); plt.close(fig2)

            st.subheader("Geometry constraint check")
            viols = validate(spec2, limits)
            if viols:
                for v in viols:
                    st.warning(v)
            else:
                st.success("All geometry constraints satisfied.")
            with st.expander("Full pole description (text)"):
                st.text(describe(spec2))
        except Exception as e:
            show_error(e)

# ── TAB 3: LOAD MODEL ─────────────────────────────────────────────────────
with tab3:
    st.header("Load Model")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import get_single_table as _gst3
            from loads import build_load_model, first_order_forces

            p3 = load_parsed(tmp_path)
            spec3 = build_spec(p3)
            base3 = load_baseline(tmp_path)
            gov_case3 = _gst3(p3, 'summary_of_steel_pole_usages')[0].get('load_case')
            cases3 = list(base3.load_cases.keys())
            sel_case3 = st.selectbox("Load case", cases3, index=gov_index(cases3, gov_case3),
                                     key="lc3")
            st.caption(f"Governing case per PLS-POLE: **{gov_case3}**")

            m3 = build_load_model(spec3, base3, sel_case3, ds=0.25)
            lc3 = m3.case

            st.subheader("Load case parameters")
            lp = st.columns(5)
            lp[0].metric("DLF", f"{lc3.dlf:.2f}")
            lp[1].metric("q trans (psf)", f"{lc3.q_trans:.1f}")
            lp[2].metric("q long (psf)", f"{lc3.q_long:.1f}")
            lp[3].metric("Ice t (in)", f"{lc3.ice_t:.3f}")
            lp[4].metric("Ice density", f"{lc3.ice_density:.0f} pcf")

            st.subheader("Base reactions — first-order (no P-delta)")
            fo3 = first_order_forces(m3, spec3.groundline_rel - 1e-9)
            br_ = st.columns(6)
            for col, (k, u) in zip(br_, [('P', 'kips'), ('Vx', 'kips'), ('Vy', 'kips'),
                                         ('Mx', 'ft-k'), ('My', 'ft-k'), ('T', 'ft-k')]):
                col.metric(f"{k} ({u})", f"{fo3[k]:+.2f}")
            st.caption("Second-order (P-delta) forces are in Tab 5 · Deflection.")

            st.subheader("Point loads at attachment points")
            if m3.points:
                st.dataframe(pd.DataFrame([{
                    "label": pl.label, "s (ft)": round(pl.s, 3),
                    "ht AGL (ft)": round(m3.z_of(pl.s), 3),
                    "dx (ft)": round(pl.dx, 3), "dy (ft)": round(pl.dy, 3), "dz (ft)": round(pl.dz, 3),
                    "Fx (kips)": round(pl.Fx, 4), "Fy (kips)": round(pl.Fy, 4), "Fz (kips)": round(pl.Fz, 4),
                } for pl in sorted(m3.points, key=lambda x: x.s)]),
                    width="stretch", hide_index=True)
                tc = st.columns(3)
                tc[0].metric("∑ Fx (kips)", f"{sum(pl.Fx for pl in m3.points):+.3f}")
                tc[1].metric("∑ Fy (kips)", f"{sum(pl.Fy for pl in m3.points):+.3f}")
                tc[2].metric("∑ Fz (kips)", f"{sum(pl.Fz for pl in m3.points):+.3f}")
            else:
                st.info("No attachment point loads for this load case.")

            st.subheader("Shaft element loads")
            st.dataframe(pd.DataFrame([{
                "s_top (ft)": round(e.s_top, 3), "s_bot (ft)": round(e.s_bot, 3),
                "ht AGL (ft)": round(m3.z_of(e.s_mid), 3), "above GL": e.above_ground,
                "D_wind (in)": round(e.D_wind, 3), "D_out (in)": round(e.D_out, 3),
                "Fy wind (k)": round(e.fy, 5), "Fx wind (k)": round(e.fx, 5),
                "Fz DL+ice (k)": round(e.fz, 5),
            } for e in m3.elements]), width="stretch", hide_index=True)
            sc = st.columns(3)
            sc[0].metric("∑ Fy shaft wind (kips)", f"{sum(e.fy for e in m3.elements):+.3f}")
            sc[1].metric("∑ Fx shaft wind (kips)", f"{sum(e.fx for e in m3.elements):+.3f}")
            sc[2].metric("∑ Fz shaft DL+ice (kips)", f"{sum(e.fz for e in m3.elements):+.3f}")

            st.subheader("Load distribution along shaft")
            above3 = [e for e in m3.elements if e.above_ground]
            hts3 = [m3.z_of(e.s_mid) for e in above3]
            fig3, (a3, b3) = plt.subplots(1, 2, figsize=(10, 5))
            a3.barh(hts3, [e.fy * 1000 for e in above3], height=0.2, color='steelblue')
            a3.set_xlabel("Transverse wind per element (lb)"); a3.set_ylabel("Height AGL (ft)")
            a3.set_title("Shaft transverse wind"); a3.grid(True, alpha=0.3)
            b3.barh(hts3, [e.fz * 1000 for e in above3], height=0.2, color='darkorange')
            b3.set_xlabel("Vertical load per element (lb)"); b3.set_ylabel("Height AGL (ft)")
            b3.set_title("Shaft self-weight + ice"); b3.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
        except Exception as e:
            show_error(e)

# ── TAB 4: ASCE 48-19 STRENGTH CHECK ──────────────────────────────────────
with tab4:
    st.header("ASCE 48-19 Strength Check")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (get_single_table as _gst4, get_field as _gf4,
                                             get_load_case_instances as _glci4)
            from geometry import w_over_t as _wot4
            from mechanics_engine import perimeter_stress_check

            p4 = load_parsed(tmp_path)
            spec4 = build_spec(p4)
            cand4 = run_candidate(tmp_path, BR, SHEAR, LAP)
            summ4 = _gst4(p4, 'summary_of_steel_pole_usages')[0]
            gov_case4 = summ4.get('load_case')
            cases4 = list(cand4.cases.keys())
            pls_by_case = {r['load_case']: _gf4(r, 'maximum_usage')
                           for r in _gst4(p4, 'summary_of_steel_pole_usages_by_load_case')}

            # ---- all-cases summary
            st.subheader("All load cases — max strength usage")
            k = st.columns(3)
            k[0].metric("Engine max usage", f"{cand4.max_strength:.2f}%")
            k[1].metric("PLS-POLE max usage", f"{_gf4(summ4, 'maximum_usage'):.2f}%")
            k[2].metric("Governing case (engine)", cand4.gov_strength_case)
            st.dataframe(pd.DataFrame([{
                "load case": c,
                "engine max %": round(cr.max_strength, 2),
                "PLS max %": pls_by_case.get(c),
                "diff (pts)": round(cr.max_strength - pls_by_case[c], 2) if c in pls_by_case else None,
                "at ht AGL (ft)": round(cr.gov_row.get('height_agl', float('nan')), 2),
            } for c, cr in cand4.cases.items()]), width="stretch", hide_index=True)

            st.divider()
            col_a, col_b = st.columns([2, 1])
            with col_a:
                sel_case4 = st.selectbox("Load case (detail)", cases4,
                                         index=gov_index(cases4, gov_case4), key="lc4")
                st.caption(f"Governing case per PLS-POLE: **{gov_case4}**")
            with col_b:
                section_mode = st.radio("Section points", ["PLS-POLE points", "Engine points (1 ft)"],
                                        key="secmode4")
                force_src = st.radio("Forces", ["Engine (P-delta)", "PLS-POLE reported"],
                                     key="force4", disabled=(section_mode != "PLS-POLE points"))

            cr4 = cand4.cases[sel_case4]
            lay4 = spec4.layout()

            def tube_for(rd):
                for i, tb in enumerate(lay4):
                    lo = max(tb['start'], lay4[i - 1]['end'] if i > 0 else tb['start'])
                    hi = min(tb['end'], lay4[i + 1]['start'] if i + 1 < len(lay4) else tb['end'])
                    if lo - 1e-6 <= rd <= hi + 1e-6:
                        return tb
                return lay4[-1]

            rows4 = []
            if section_mode == "PLS-POLE points":
                pls_rows = sorted(_glci4(p4, 'detailed_steel_pole_usages').get(sel_case4, []),
                                  key=lambda r: _gf4(r, 'rel_dist'))
                for r in pls_rows:
                    rd = _gf4(r, 'rel_dist')
                    if rd > spec4.groundline_rel + 1e-6:
                        continue
                    tb = tube_for(rd)
                    D = tb['d_top'] + spec4.taper * (rd - tb['start'])
                    t = tb['thickness']
                    wt = _wot4(D, t, BR)
                    if force_src.startswith("Engine"):
                        f = cr4.defl.section_forces(rd, r.get('joint_position') != 'End')
                        F = (f['axial_local'], f['Mx'], f['My'], f['Vy_local'], f['Vx_local'], f['T'])
                    else:
                        F = tuple(_gf4(r, k) or 0.0 for k in ('axial_force', 'trans_mom_local_mx',
                                  'long_mom_local_my', 'tran_shear', 'long_shear', 'tors_mom'))
                    c = perimeter_stress_check(*F, D, t, tb['fy'], wt, SHEAR)
                    pu = _gf4(r, 'max_usage')
                    rows4.append({
                        'rel_dist (ft)': round(rd, 3), 'ht AGL (ft)': round(spec4.groundline_rel - rd, 3),
                        'side': r.get('joint_position'), 'tube': tb['tube_no'],
                        'D (in)': round(D, 3), 't (in)': t, 'w/t': round(wt, 2),
                        'P (k)': round(F[0], 3), 'Mx (ft-k)': round(F[1], 1), 'My (ft-k)': round(F[2], 1),
                        'PLS Mx': _gf4(r, 'trans_mom_local_mx'),
                        'Fa (ksi)': round(c['Fa'], 2) if c['Fa'] else None, 'Fa eq': c['Fa_equation'],
                        'p_a': round(c['p_a'], 3), 'PLS p_a': _gf4(r, 'p_a'),
                        'm_s': round(c['m_s'], 3), 'PLS m_s': _gf4(r, 'm_s'),
                        'v_q': round(c['v_q'], 3), 'PLS v_q': _gf4(r, 'v_q'),
                        'res': round(c['res'], 3), 'PLS res': _gf4(r, 'res'),
                        'calc usage %': round(c['usage'], 2) if c['usage'] else None,
                        'PLS usage %': pu,
                        'diff (pts)': round(c['usage'] - pu, 2) if (c['usage'] is not None and pu is not None) else None,
                    })
            else:
                for x in cr4.rows:
                    rows4.append({
                        'rel_dist (ft)': round(x['s'], 3), 'ht AGL (ft)': round(x['height_agl'], 3),
                        'side': x['side'], 'tube': x['tube_no'], 'D (in)': round(x['D'], 3),
                        't (in)': x['t'], 'w/t': round(x['w_over_t'], 2), 'P (k)': round(x['P'], 3),
                        'Mx (ft-k)': round(x['Mx'], 1), 'My (ft-k)': round(x['My'], 1),
                        'Fa (ksi)': round(x['Fa'], 2) if x['Fa'] else None, 'Fa eq': x['Fa_equation'],
                        'p_a': round(x['p_a'], 3), 'm_s': round(x['m_s'], 3),
                        'v_q': round(x['v_q'], 3), 'res': round(x['res'], 3),
                        'calc usage %': round(x['usage'], 2) if x['usage'] else None,
                    })

            df4 = pd.DataFrame(rows4)
            if df4.empty:
                st.warning("No check points for this case.")
            else:
                valid4 = df4[df4['calc usage %'].notna()]
                gov4 = valid4.loc[valid4['calc usage %'].idxmax()]
                st.subheader(f"Governing section — {sel_case4}")
                g_ = st.columns(5)
                g_[0].metric("Max usage", f"{gov4['calc usage %']:.2f}%")
                g_[1].metric("At ht AGL", f"{gov4['ht AGL (ft)']:.2f} ft")
                g_[2].metric("D / t", f"{gov4['D (in)']:.2f} / {gov4['t (in)']:.4f}")
                g_[3].metric("Fa (ksi)", f"{gov4['Fa (ksi)']:.2f}")
                g_[4].metric("Fa equation", str(gov4['Fa eq']))

                st.subheader("Section check results")
                st.dataframe(df4, width="stretch", hide_index=True)

                cA, cB = st.columns(2)
                with cA:
                    fig4a, ax4a = plt.subplots(figsize=(6, 6))
                    ax4a.plot(valid4['calc usage %'], valid4['ht AGL (ft)'], color='steelblue',
                              linewidth=2, label='Engine')
                    if 'PLS usage %' in valid4 and valid4['PLS usage %'].notna().any():
                        ax4a.plot(valid4['PLS usage %'], valid4['ht AGL (ft)'], color='darkorange',
                                  linewidth=1.5, linestyle='--', label='PLS-POLE')
                    ax4a.axvline(100, color='red', linestyle='--', linewidth=1, label='100%')
                    ax4a.set_xlabel("Usage (%)"); ax4a.set_ylabel("Height AGL (ft)")
                    ax4a.set_title("Usage vs height"); ax4a.legend(); ax4a.grid(True, alpha=0.3)
                    st.pyplot(fig4a); plt.close(fig4a)
                with cB:
                    fig4b, ax4b = plt.subplots(figsize=(6, 6))
                    for col in ('p_a', 'm_s', 'v_q'):
                        ax4b.plot(valid4[col].abs(), valid4['ht AGL (ft)'], label=f'|{col}|', linewidth=1.5)
                    ax4b.plot(valid4['res'], valid4['ht AGL (ft)'], label='res', linewidth=2,
                              color='black', linestyle='--')
                    ax4b.set_xlabel("Stress at governing point (ksi)"); ax4b.set_ylabel("Height AGL (ft)")
                    ax4b.set_title("Stress components"); ax4b.legend(fontsize=8); ax4b.grid(True, alpha=0.3)
                    st.pyplot(fig4b); plt.close(fig4b)

                if 'diff (pts)' in df4 and df4['diff (pts)'].notna().any():
                    d = df4[df4['diff (pts)'].notna()]
                    st.subheader("Engine vs PLS-POLE usage difference")
                    s_ = st.columns(4)
                    s_[0].metric("Mean diff", f"{d['diff (pts)'].mean():+.3f} pts")
                    s_[1].metric("Max |diff|", f"{d['diff (pts)'].abs().max():.3f} pts")
                    s_[2].metric("Std dev", f"{d['diff (pts)'].std():.3f} pts")
                    s_[3].metric("Points > ±2", f"{int((d['diff (pts)'].abs() > 2).sum())}")
                    fig4c, ax4c = plt.subplots(figsize=(7, 4))
                    ax4c.barh(d['ht AGL (ft)'], d['diff (pts)'], height=0.6,
                              color=['red' if abs(v) > 2 else 'steelblue' for v in d['diff (pts)']])
                    ax4c.axvline(0, color='black', linewidth=0.8)
                    ax4c.set_xlabel("Engine − PLS (usage pts)"); ax4c.set_ylabel("Height AGL (ft)")
                    ax4c.grid(True, alpha=0.3)
                    st.pyplot(fig4c); plt.close(fig4c)
        except Exception as e:
            show_error(e)

# ── TAB 5: DEFLECTION ─────────────────────────────────────────────────────
with tab5:
    st.header("Deflection — second-order (P-delta) solver")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (get_single_table as _gst5, get_field as _gf5,
                                             get_load_case_instances as _glci5)
            from loads import build_load_model as _blm5
            from deflection import solve_deflection

            p5 = load_parsed(tmp_path)
            spec5 = build_spec(p5)
            base5 = load_baseline(tmp_path)
            cand5 = run_candidate(tmp_path, BR, SHEAR, LAP)

            tips = {r['load_case']: r for r in _gst5(p5, 'summary_of_tip_deflections_for_all_load_cases')}
            pdu = ({r['load_case']: r for r in _gst5(p5, 'pole_deflection_usages_for_all_load_cases')}
                   if 'pole_deflection_usages_for_all_load_cases' in p5['tables'] else {})

            st.subheader("Deflection limit check")
            k = st.columns(3)
            if cand5.max_defl_usage is not None:
                pls_max = max((_gf5(r, 'usage') for r in pdu.values()), default=None)
                k[0].metric("Engine max deflection usage", f"{cand5.max_defl_usage:.2f}%")
                k[1].metric("PLS-POLE max deflection usage", f"{pls_max:.2f}%" if pls_max is not None else "—")
                k[2].metric("Governing case", cand5.gov_defl_case)
            else:
                st.info("No load case in this XML has a deflection limit.")
            st.caption(f"Deflection limit basis: above-ground height = {spec5.agl_height:.2f} ft. "
                       f"Lap stiffness: {LAP}.")

            st.subheader("All load cases — tip deflection")
            rows5 = []
            for c, cr in cand5.cases.items():
                r = cr.defl
                T = tips.get(c)
                pt = _gf5(T, 'tran_defl') if T else None
                dc = cr.defl_check
                rows5.append({
                    "load case": c,
                    "trans (in)": round(r.tip_trans_in, 2), "PLS trans": pt,
                    "diff %": round((r.tip_trans_in / pt - 1) * 100, 2) if (pt and abs(pt) > 1) else None,
                    "long (in)": round(r.tip_long_in, 2), "PLS long": _gf5(T, 'long_defl') if T else None,
                    "vert (in)": round(r.tip_vert_in, 2), "PLS vert": _gf5(T, 'vert_defl') if T else None,
                    "limit": f"{dc['limit']} {dc['check']}" if dc else "No Limit",
                    "usage %": round(dc['usage'], 2) if dc and dc.get('usage') is not None else None,
                    "PLS usage %": _gf5(pdu[c], 'usage') if c in pdu else None,
                    "iter": r.iterations, "converged": r.converged,
                })
            st.dataframe(pd.DataFrame(rows5), width="stretch", hide_index=True)

            st.divider()
            cases5 = list(cand5.cases.keys())
            default5 = cand5.gov_defl_case or _gst5(p5, 'summary_of_steel_pole_usages')[0].get('load_case')
            sel5 = st.selectbox("Load case (detail)", cases5, index=gov_index(cases5, default5), key="lc5")
            r5 = cand5.cases[sel5].defl
            gl5 = spec5.groundline_rel

            m_ = st.columns(4)
            m_[0].metric("Tip transverse", f"{r5.tip_trans_in:.2f} in")
            m_[1].metric("Tip longitudinal", f"{r5.tip_long_in:.2f} in")
            m_[2].metric("Tip vertical", f"{r5.tip_vert_in:.2f} in")
            m_[3].metric("Tip from vertical axis", f"{r5.tip_horiz_ft:.3f} ft")

            pls5 = sorted(_glci5(p5, 'detailed_steel_pole_usages').get(sel5, []),
                          key=lambda q: _gf5(q, 'rel_dist'))
            fo5 = solve_deflection(_blm5(spec5, base5, sel5, ds=0.25), lap_stiffness=LAP,
                                   second_order=False)

            cA, cB = st.columns(2)
            with cA:
                fig5, ax5 = plt.subplots(figsize=(6, 6))
                h = gl5 - r5.s
                ax5.plot(r5.y * 12, h, color='steelblue', linewidth=2, label='Engine transverse')
                if np.any(np.abs(r5.x) > 1e-4):
                    ax5.plot(r5.x * 12, h, color='seagreen', linewidth=1.5, label='Engine longitudinal')
                if pls5:
                    ax5.scatter([_gf5(q, 'trans_defl') for q in pls5],
                                [gl5 - _gf5(q, 'rel_dist') for q in pls5],
                                s=12, color='darkorange', zorder=3, label='PLS-POLE transverse')
                ax5.axvline(0, color='gray', linewidth=0.5)
                ax5.set_xlabel("Deflection (in)"); ax5.set_ylabel("Height AGL (ft)")
                ax5.set_title("Deflected shape"); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
                st.pyplot(fig5); plt.close(fig5)
            with cB:
                fig6, ax6 = plt.subplots(figsize=(6, 6))
                ax6.plot(r5.Mx, gl5 - r5.s, color='steelblue', linewidth=2, label='Engine 2nd-order Mx')
                ax6.plot(fo5.Mx, gl5 - fo5.s, color='gray', linewidth=1, linestyle=':', label='1st-order Mx')
                if pls5:
                    ax6.scatter([_gf5(q, 'trans_mom_local_mx') for q in pls5],
                                [gl5 - _gf5(q, 'rel_dist') for q in pls5],
                                s=12, color='darkorange', zorder=3, label='PLS-POLE Mx')
                ax6.set_xlabel("Mx (ft-k)"); ax6.set_ylabel("Height AGL (ft)")
                ax6.set_title("Transverse moment: P-delta effect"); ax6.legend(fontsize=8)
                ax6.grid(True, alpha=0.3)
                st.pyplot(fig6); plt.close(fig6)

            mb2, mb1 = r5.Mx[-1], fo5.Mx[-1]
            st.caption(f"Base Mx: 2nd-order {mb2:.1f} ft-k vs 1st-order {mb1:.1f} ft-k "
                       f"→ P-delta adds {((mb2 / mb1 - 1) * 100) if abs(mb1) > 1e-6 else 0:+.2f}%.")

            with st.expander("Deflected shape table"):
                idx = np.arange(0, len(r5.s), 4)
                st.dataframe(pd.DataFrame({
                    "rel_dist (ft)": np.round(r5.s[idx], 2), "ht AGL (ft)": np.round(gl5 - r5.s[idx], 2),
                    "trans (in)": np.round(r5.y[idx] * 12, 3), "long (in)": np.round(r5.x[idx] * 12, 3),
                    "vert (in)": np.round(r5.dz[idx] * 12, 3),
                    "Mx 2nd (ft-k)": np.round(r5.Mx[idx], 1), "My 2nd (ft-k)": np.round(r5.My[idx], 1),
                    "P (k, +comp)": np.round(r5.P[idx], 3),
                }), width="stretch", hide_index=True)

            if spec5.embedment:
                st.warning("Embedded pole: solver assumes the pole is FIXED at the ground line "
                           "(rigid foundation). Not yet validated against an embedded PLS-POLE model.")
        except Exception as e:
            show_error(e)

# ── TAB 6: OPTIMIZER ───────────────────────────────────────────────────────
with tab6:
    st.header("Minimum-Weight Optimizer")
    st.info("🚧 Next: greedy search, all load cases, strength + deflection.")
