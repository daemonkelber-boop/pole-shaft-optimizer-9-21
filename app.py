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
    LAP = st.radio("Slip-joint lap stiffness (deflection)", ["midpoint", "outer", "inner", "sum"], index=0,
                   help="midpoint (default) = female tube above the lap mid-point, male below; "
                        "conservative by +0.1% to +0.6% mean on tip deflection vs PLS-POLE "
                        "(003, 014, 015). outer = female only (+0.4% to +1.9%); "
                        "inner = male only (slightly unconservative); sum = both walls (too stiff).")

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
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            import json, math as _m, time as _time
            import plotly.graph_objects as go
            from pls_pole_xml_parser import get_single_table as _gst6, get_field as _gf6
            from optimizer import Optimizer, OptConstraints, summarize, length_class
            from weight import pole_weight as _pw

            p6 = load_parsed(tmp_path)
            spec6 = build_spec(p6)
            base6 = load_baseline(tmp_path)
            lay6 = spec6.layout()
            n_j6 = len(lay6) - 1
            half = lambda v: _m.floor(v * 2) / 2
            base_joints = {n_j6 - k: lay6[k]['joint_type'] for k in range(n_j6)}
            base_flange = ",".join(str(k) for k, v in sorted(base_joints.items()) if v == 'flange')

            # ---- baseline (seed) weights -------------------------------------
            w_base = _pw(spec6)['total_weight']
            pls_total = _gf6(_gst6(p6, 'summary_of_steel_pole_usages')[0], 'weight')
            plate_w, plate_note = 0.0, "no base plate"
            if 'base_plate_properties' in p6['tables']:
                lab = _gst6(p6, 'steel_pole_properties')[0].get('steel_pole_property_label')
                rows_bp = [r for inst in p6['tables']['base_plate_properties'] for r in inst['rows']]
                match = [r for r in rows_bp if r.get('pole_property') == lab] or rows_bp[:1]
                if match and str(_gst6(p6, 'steel_pole_properties')[0].get('base_plate', '')).lower() == 'yes':
                    plate_w = _gf6(match[0], 'plate_weight') or 0.0
                    plate_note = f"base plate {plate_w:,.0f} lb removed"
            pls_shaft = (pls_total - plate_w) if pls_total else None

            st.subheader("Starting point (baseline XML, shaft only)")
            b_ = st.columns(3)
            b_[0].metric("Engine shaft weight", f"{w_base:,.0f} lb",
                         help="Steel in all tubes including laps and embedded length. Base plate excluded.")
            b_[1].metric("PLS-POLE shaft weight", f"{pls_shaft:,.0f} lb" if pls_shaft else "—",
                         help=f"PLS-reported pole weight {pls_total:,.1f} lb; {plate_note}." if pls_total else None)
            b_[2].metric("Engine vs PLS", f"{(w_base / pls_shaft - 1) * 100:+.2f}%" if pls_shaft else "—")
            st.caption(f"PLS-POLE reported {pls_total:,.1f} lb; {plate_note}. "
                       "All savings below are measured against the engine shaft weight." if pls_total else "")

            DEF = dict(
                tip_min=10.0, tip_max=half(spec6.tip_diameter + 7), tip_inc=0.5,
                base_min=half(max(spec6.base_diameter - 15, 20)), base_max=half(spec6.base_diameter + 10),
                base_inc=0.5, t_mode="Min / max / increment", t_min=0.1875, t_max=1.0, t_inc=0.0625,
                t_list="0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5", taper_min=0.15, taper_max=0.50,
                max_wt=38.0, max_segments=6, len_preferred=53.0, len_normal_max=57.0,
                len_special_max=60.0, min_tube=15.0,
                fix_bottom=False, bottom_length=40.0, bottom_mode="exactly L", joint_default="slip",
                flange_at=base_flange, slip_at="", lap_factor=1.65, lap_round=0.25,
                slip_clearance=0.125, min_slip_above_gl=0.0, fy=65.0,
                strength_target=100.0, defl_target=100.0, tolerance_pct=0.0,
                long_tube_threshold_pct=2.0, tie_band_pct=1.0,
                n_alternates=10, log_all=False, time_limit_min=15.0, max_evals=200000)
            for k, v in DEF.items():
                st.session_state.setdefault(f"o_{k}", v)
            if st.button("↺ Reset all constraints to defaults"):
                for k, v in DEF.items():
                    st.session_state[f"o_{k}"] = v
                st.rerun()

            with st.form("opt_form"):
                st.markdown("**Diameters and thickness** (tip and base stay on the grid; taper is derived)")
                h = st.columns([1.3, 1, 1, 1])
                h[1].markdown("Minimum"); h[2].markdown("Maximum"); h[3].markdown("Increment")
                r = st.columns([1.3, 1, 1, 1]); r[0].markdown("Top diameter (in)")
                r[1].number_input("tmin", key="o_tip_min", step=0.5, label_visibility="collapsed")
                r[2].number_input("tmax", key="o_tip_max", step=0.5, label_visibility="collapsed")
                r[3].number_input("tinc", key="o_tip_inc", step=0.5, label_visibility="collapsed")
                r = st.columns([1.3, 1, 1, 1]); r[0].markdown("Bottom diameter (in)")
                r[1].number_input("bmin", key="o_base_min", step=0.5, label_visibility="collapsed")
                r[2].number_input("bmax", key="o_base_max", step=0.5, label_visibility="collapsed")
                r[3].number_input("binc", key="o_base_inc", step=0.5, label_visibility="collapsed")
                st.radio("Plate thicknesses to try", ["Min / max / increment", "User list only"],
                         key="o_t_mode", horizontal=True)
                r = st.columns([1.3, 1, 1, 1]); r[0].markdown("Thickness (in)")
                r[1].number_input("thmin", key="o_t_min", step=0.0625, format="%.4f", label_visibility="collapsed")
                r[2].number_input("thmax", key="o_t_max", step=0.0625, format="%.4f", label_visibility="collapsed")
                r[3].number_input("thinc", key="o_t_inc", step=0.0625, format="%.4f", label_visibility="collapsed")
                st.text_input("User thickness list (in, comma separated)", key="o_t_list")
                r = st.columns(4)
                r[0].number_input("Min taper (in/ft)", key="o_taper_min", step=0.01, format="%.4f")
                r[1].number_input("Max taper (in/ft)", key="o_taper_max", step=0.01, format="%.4f")
                r[2].number_input("Max w/t", key="o_max_wt", step=0.5)
                r[3].number_input("Fy (ksi)", key="o_fy", step=5.0)

                st.markdown("**Sections and joints**")
                r = st.columns(5)
                r[0].number_input("Max segments", key="o_max_segments", min_value=1, max_value=10, step=1)
                r[1].number_input("Preferred length (ft)", key="o_len_preferred", step=0.25)
                r[2].number_input("Normal max length (ft)", key="o_len_normal_max", step=0.25)
                r[3].number_input("Long-tube max length (ft)", key="o_len_special_max", step=0.25)
                r[4].number_input("Min tube length (ft)", key="o_min_tube", step=0.25)
                r2 = st.columns([1, 1, 1.4])
                r2[0].checkbox("Fix bottom tube length", key="o_fix_bottom")
                r2[1].number_input("Bottom tube length L (ft)", key="o_bottom_length", step=0.25,
                                   min_value=1.0)
                r2[2].radio("L means", ["exactly L", "at most L"], key="o_bottom_mode", horizontal=True)
                st.caption(
                    "**Fixed bottom tube (off by default).** Some customers require the base section at a "
                    "set length. L is the **fabricated length of the bottom tube, embedment included** - "
                    "on a pole with 20 ft embedment, L = 40 ft means 40 ft of steel with 20 ft above ground. "
                    "The tube directly above then absorbs the remaining height on the 0.25 ft grid and is "
                    "not held to the preferred 53 / 57 ft lengths. If that tube would fall outside the min / "
                    "max tube length, the segment count is infeasible and the optimizer moves to the next "
                    "one automatically. *Example (003, 110 ft, base plate): L = 40 ft has no 2-segment "
                    "solution, so the result is 3 segments, 53 / 27 / 40 ft at 13,379 lb versus 12,798 lb "
                    "unconstrained.*")
                st.number_input("Long-tube weight threshold X (%)", key="o_long_tube_threshold_pct",
                                min_value=0.0, step=0.5)
                st.caption(
                    "**What is X?** Tubes longer than the *normal max* (57 ft default) are harder to "
                    "fabricate, galvanize and ship. For each segment count, the optimizer compares the "
                    "lightest design that uses only standard lengths with the lightest design that uses a "
                    "long tube. The long-tube design is accepted only if it is **more than X% lighter**. "
                    "X = 0 → always take the lighter design; a large X → long tubes are effectively "
                    "never used when a standard layout exists. *Example (003): standard 57/57 ft = 14,199 lb, "
                    "long-tube 54/60 ft = 12,784 lb → 10.0% saving → accepted when X < 10.0.* "
                    "If no standard-length layout is possible at a segment count, long tubes are allowed there.")
                r = st.columns(3)
                r[0].radio("Default joint type", ["slip", "flange"], key="o_joint_default", horizontal=True)
                r[1].text_input("Force FLANGE at joints (1 = lowest)", key="o_flange_at",
                                help="Comma-separated positions counted from the bottom joint. "
                                     "Pre-filled from the baseline.")
                r[2].text_input("Force SLIP at joints (1 = lowest)", key="o_slip_at")
                r = st.columns(4)
                r[0].number_input("Lap factor (× female ID)", key="o_lap_factor", step=0.05, format="%.3f",
                                  help="1.65 = 1.1 × 1.5")
                r[1].number_input("Lap rounding up to (ft)", key="o_lap_round", step=0.25)
                r[2].number_input("Slip clearance (in)", key="o_slip_clearance", step=0.0625, format="%.4f")
                r[3].number_input("Lowest slip joint above GL (ft)", key="o_min_slip_above_gl", step=1.0,
                                  help="Clearance from the ground line up to the BOTTOM OF THE LAP of the lowest slip joint (the female tube's lower end), not the joint itself. Default 0 = no requirement. Example (011): joint at 15.0 ft AGL, lap bottom at 7.5 ft AGL.")

                st.markdown("**Acceptance and ranking**")
                r = st.columns(4)
                r[0].selectbox("Strength usage target (%)", [100.0, 95.0, 90.0, 85.0], key="o_strength_target")
                r[1].number_input("Deflection target (% of XML limit)", key="o_defl_target", step=5.0)
                r[2].number_input("Acceptance tolerance (% of target)", key="o_tolerance_pct",
                                  min_value=0.0, max_value=1.0, step=0.1,
                                  help="Relative. Applies to strength AND deflection. Accept if usage ≤ "
                                       "target × (1 + tolerance/100). E.g. target 100%, tolerance 1 → 101%; "
                                       "target 90%, tolerance 1 → 90.9%. Designs accepted above target are "
                                       "flagged 'within tolerance'.")
                r[3].number_input("Tie band (% of lightest)", key="o_tie_band_pct", step=0.5,
                                  help="Designs within this % of the lightest are ranked by: fewer segments → "
                                       "preferred lengths → weight → lower usage.")
                r = st.columns(3)
                r[0].number_input("Alternates to report", key="o_n_alternates", min_value=0, max_value=30, step=1)
                r[1].number_input("Search time limit (min)", key="o_time_limit_min", min_value=1.0, step=1.0,
                                  help="The search stops here and the result is flagged as truncated. "
                                       "Final all-case verification is never cut short.")
                r[2].number_input("Evaluation cap", key="o_max_evals", min_value=500, step=1000,
                                  help="Hard stop on the number of screening analyses. Leave high and let "
                                       "the time limit govern; lower it only to force a quick run.")
                st.checkbox("Log every candidate tried", key="o_log_all")
                go_run = st.form_submit_button("▶ Run optimizer", type="primary")

            def _pos(txt):
                out = []
                for x in str(txt).replace(";", ",").split(","):
                    x = x.strip()
                    if x:
                        out.append(int(float(x)))
                return out

            if go_run:
                S = st.session_state
                ov = {k: 'flange' for k in _pos(S.o_flange_at)}
                ov.update({k: 'slip' for k in _pos(S.o_slip_at)})
                C6 = OptConstraints(
                    tip_min=S.o_tip_min, tip_max=S.o_tip_max, tip_inc=S.o_tip_inc,
                    base_min=S.o_base_min, base_max=S.o_base_max, base_inc=S.o_base_inc,
                    t_min=S.o_t_min, t_max=S.o_t_max, t_inc=S.o_t_inc,
                    t_list=([float(x) for x in S.o_t_list.split(",") if x.strip()]
                            if S.o_t_mode == "User list only" else None),
                    taper_min=S.o_taper_min, taper_max=S.o_taper_max, max_wt=S.o_max_wt,
                    bend_radius_factor=BR, fy=S.o_fy, max_segments=int(S.o_max_segments),
                    len_preferred=S.o_len_preferred, len_normal_max=S.o_len_normal_max,
                    len_special_max=S.o_len_special_max, min_tube=S.o_min_tube,
                    fix_bottom=bool(S.o_fix_bottom), bottom_length=float(S.o_bottom_length),
                    bottom_mode=('exact' if S.o_bottom_mode == 'exactly L' else 'max'),
                    joint_default=S.o_joint_default, joint_overrides=ov,
                    lap_factor=S.o_lap_factor, lap_round=S.o_lap_round,
                    slip_clearance=S.o_slip_clearance, min_slip_above_gl=S.o_min_slip_above_gl,
                    strength_target=float(S.o_strength_target), defl_target=S.o_defl_target,
                    tolerance_pct=float(S.o_tolerance_pct),
                    long_tube_threshold_pct=float(S.o_long_tube_threshold_pct),
                    tie_band_pct=S.o_tie_band_pct, n_alternates=int(S.o_n_alternates),
                    shear_mode=SHEAR, lap_stiffness=LAP,
                    time_limit_s=float(S.o_time_limit_min) * 60.0,
                    max_evaluations=int(S.o_max_evals))

                run_hdr = st.empty()
                run_hdr.subheader("Running")
                bar = st.progress(0.0, text="Starting")
                live = st.empty()
                holder, last = [], [0.0]

                def _prog(f, m):
                    bar.progress(min(max(f, 0.0), 1.0), text=m)
                    if not holder or _time.time() - last[0] < 1.0:
                        return
                    last[0] = _time.time()
                    o = holder[0]
                    el = _time.time() - o.t0
                    with live.container():
                        c = st.columns(5)
                        c[0].metric("Phase", o.phase)
                        c[1].metric("Elapsed", f"{el / 60:.1f} / {C6.time_limit_s / 60:.0f} min")
                        c[2].metric("Evaluations", f"{o.n_evals:,}")
                        c[3].metric("Passing designs found", f"{len(o.found):,}")
                        if o.best_weight < float('inf'):
                            d = o.best_weight - w_base
                            c[4].metric("Best weight (screened, not yet verified)",
                                        f"{o.best_weight:,.0f} lb",
                                        f"{d:+,.0f} lb ({d / w_base * 100:+.1f}%) vs baseline",
                                        delta_color="inverse")
                        else:
                            c[4].metric("Best weight (screened)", "—")

                opt_real = Optimizer(base6, spec6, C6, _prog, log_all=bool(S.o_log_all))
                holder.append(opt_real)
                res = opt_real.run()
                bar.empty(); live.empty(); run_hdr.empty()
                st.session_state["opt_result"] = dict(res=res, C=C6, log=opt_real.log, file=tmp_path)
                st.session_state.pop("opt_pick", None)

            R = st.session_state.get("opt_result")
            if R and R["file"] == tmp_path:
                res, C6 = R["res"], R["C"]
                B = res["baseline"]
                if res.get("truncated"):
                    st.warning(f"⚠️ Search stopped early — {res.get('trunc_reason') or 'limit reached'} — before all candidate "
                               "combinations were examined. The designs below passed full verification, "
                               "but a lighter design may exist. Raise the limit, or narrow the diameter "
                               "ranges around the best result and re-run.")
                    if res.get("lb_remaining") and res.get("winner"):
                        lbw, ww = res["lb_remaining"], res["winner"].weight
                        if lbw < ww:
                            st.info(f"Bound on what was missed: combinations are searched in lower-bound "
                                    f"order, so no unexamined design can weigh less than **{lbw:,.0f} lb**. "
                                    f"The recommended design is {ww:,.0f} lb, so an unexamined design could "
                                    f"at best be {(1 - lbw / ww) * 100:.1f}% lighter.")
                        else:
                            st.info(f"No unexamined combination can weigh less than {lbw:,.0f} lb, which is "
                                    f"heavier than the recommended {ww:,.0f} lb — the result is optimal for "
                                    "these constraints despite the early stop.")
                else:
                    st.success(f"Search completed in {res['elapsed']:.0f} s ({res['n_evals']:,} screening "
                               "evaluations): every combination in the ranges was sized or ruled out.")
                st.caption(f"Screening cases: {', '.join(res['screen_cases'])}. Every reported design "
                           "was re-checked on ALL load cases, full mesh.")

                st.subheader("Baseline and seed")
                k = st.columns(4)
                k[0].metric("Baseline shaft weight (engine)", f"{B['weight']:,.0f} lb")
                k[1].metric("Baseline max strength / deflection",
                            f"{B['strength']:.2f}% / " + (f"{B['defl']:.2f}%" if B['defl'] is not None else "—"))
                if B['seed_weight'] is not None:
                    k[2].metric("Seed after sizing", f"{B['seed_weight']:,.0f} lb",
                                f"{B['seed_weight'] - B['weight']:+,.0f} lb", delta_color="inverse")
                k[3].metric("Acceptance limit", f"{C6.strength_limit():.2f}% strength")
                st.caption(f"Seed: {B['seed_note']}.")

                win = res["winner"]
                if win is None:
                    st.error("No design satisfied all constraints and load cases. "
                             "Widen the diameter / thickness / taper ranges and re-run.")
                else:
                    sw = summarize(win, C6)
                    st.subheader("Recommended design")
                    k = st.columns(5)
                    k[0].metric("Shaft weight", f"{win.weight:,.0f} lb",
                                f"{win.weight - B['weight']:+,.0f} lb vs baseline", delta_color="inverse")
                    k[1].metric("Saving vs baseline", f"{(1 - win.weight / B['weight']) * 100:.1f}%")
                    k[2].metric("Max strength", f"{win.strength:.2f}%")
                    k[3].metric("Max deflection", f"{win.defl:.2f}%" if win.defl is not None else "—")
                    k[4].metric("Governs", win.gov_check)
                    if sw["acceptance"] == "within tolerance":
                        st.info(f"Accepted within the {C6.tolerance_pct:g}% tolerance (usage above the "
                                f"{C6.strength_target:g}% target).")
                    for f_, v_ in (("tip", sw["D tip (in)"]), ("base", sw["D base (in)"])):
                        lim = (C6.tip_min, C6.tip_max) if f_ == "tip" else (C6.base_min, C6.base_max)
                        if abs(v_ - lim[0]) < 1e-6 or abs(v_ - lim[1]) < 1e-6:
                            st.warning(f"{f_.capitalize()} diameter {v_} in is at the edge of the search "
                                       f"range {lim}. A lighter design may exist outside it.")
                    st.dataframe(pd.DataFrame([{
                        "tube #": t['tube_no'], "length (ft)": t['length'], "thickness (in)": t['thickness'],
                        "D top (in)": round(t['d_top'], 3), "D bot (in)": round(t['d_bot'], 3),
                        "joint below": t['joint_type'], "lap (ft)": t['lap'],
                        "height AGL of tube top (ft)": round(win.spec.groundline_rel - t['start'], 2),
                    } for t in win.spec.layout()]), width="stretch", hide_index=True)
                    st.caption(f"Taper {win.spec.taper:.5f} in/ft (derived). "
                               f"{res.get('winner_variants', 0)} other tube-length arrangements of this "
                               "design also pass; the one shown ranks best on section lengths.")

                # ---- long-tube rule table
                st.subheader(f"Long-tube rule (X = {C6.long_tube_threshold_pct:g}%)")
                st.dataframe(pd.DataFrame(res["long_table"]), width="stretch", hide_index=True)
                st.caption("For each segment count: lightest verified design with standard lengths only vs "
                           "lightest with a tube over the normal max. Long tubes are used only when the "
                           "saving exceeds X.")

                # ---- design scatter
                kept = sorted(res.get("kept", []), key=lambda e: e.weight)
                if win is not None:
                    alts = [a for a, _, _ in res["alternates"]]
                    ranked = [win] + alts
                    others = [e for e in kept if all(e is not r_ for r_ in ranked)]
                    labels = {id(win): "Recommended"}
                    for i, a in enumerate(alts):
                        labels[id(a)] = f"Rank {i + 2}"
                    for e in others:
                        labels[id(e)] = f"Other · {e.weight:,.0f} lb · D {e.design.tip}/{e.design.base}"
                    pickable = ranked + others
                    for e in res.get("dropped", []):
                        labels[id(e)] = f"Excluded (long-tube rule) · {e.weight:,.0f} lb"
                    pickable += list(res.get("dropped", []))
                    gu = lambda e: max(e.strength, e.defl or 0.0)

                    st.subheader("Design map — every fully verified design")
                    fig = go.Figure()
                    wmin, band = res["wmin"], res["band"]
                    fig.add_hrect(y0=wmin, y1=band, fillcolor="green", opacity=0.08, line_width=0,
                                  annotation_text=f"tie band ({C6.tie_band_pct:g}%)",
                                  annotation_position="top left")
                    fig.add_vline(x=C6.strength_target, line_dash="dash", line_color="red",
                                  annotation_text="target")
                    if C6.tolerance_pct > 0:
                        fig.add_vline(x=C6.strength_limit(), line_dash="dot", line_color="orange",
                                      annotation_text="tolerance")
                    colors = {1: "#8c564b", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd", 5: "#ff7f0e", 6: "#17becf"}
                    idx_of = {id(e): i for i, e in enumerate(pickable)}
                    def hover(e):
                        s_ = summarize(e, C6)
                        return (f"<b>{labels[id(e)]}</b><br>{e.weight:,.0f} lb "
                                f"({(e.weight / B['weight'] - 1) * 100:+.1f}% vs baseline)<br>"
                                f"D {s_['D tip (in)']} / {s_['D base (in)']} in, taper {s_['taper (in/ft)']}<br>"
                                f"tubes {s_['tube lengths (ft)']} ft<br>t {s_['thickness (in)']} in<br>"
                                f"strength {e.strength:.2f}%, defl "
                                + (f"{e.defl:.2f}%" if e.defl is not None else "—")
                                + f"<br>governs: {e.gov_check}")
                    for n_ in sorted({e.design.n for e in kept}):
                        for lng in (False, True):
                            grp = [e for e in kept if e.design.n == n_ and
                                   (length_class(e.spec, C6)['n_special'] > 0) == lng]
                            if not grp:
                                continue
                            fig.add_trace(go.Scatter(
                                x=[gu(e) for e in grp], y=[e.weight for e in grp], mode="markers",
                                name=f"{n_} seg" + (" · long tube" if lng else ""),
                                marker=dict(size=9, color=colors.get(n_, "gray"),
                                            symbol="circle-open" if lng else "circle",
                                            line=dict(width=2 if lng else 0.5, color=colors.get(n_, "gray"))),
                                customdata=[idx_of[id(e)] for e in grp],
                                hovertext=[hover(e) for e in grp], hoverinfo="text"))
                    drp = list(res.get("dropped", []))
                    if drp:
                        fig.add_trace(go.Scatter(
                            x=[gu(e) for e in drp], y=[e.weight for e in drp], mode="markers",
                            name="excluded (long-tube rule)",
                            marker=dict(size=8, color="lightgray", symbol="x"),
                            customdata=[idx_of[id(e)] for e in drp],
                            hovertext=[hover(e) for e in drp], hoverinfo="text"))
                    fig.add_trace(go.Scatter(
                        x=[gu(a) for a in alts], y=[a.weight for a in alts], mode="text",
                        text=[str(i + 2) for i in range(len(alts))], textposition="top center",
                        showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(
                        x=[gu(win)], y=[win.weight], mode="markers+text", name="Recommended",
                        marker=dict(size=18, color="gold", symbol="diamond", line=dict(width=1.5, color="black")),
                        text=["1"], textposition="middle center", customdata=[idx_of[id(win)]],
                        hovertext=[hover(win)], hoverinfo="text"))
                    fig.add_trace(go.Scatter(
                        x=[max(B['strength'], B['defl'] or 0)], y=[B['weight']], mode="markers",
                        name="Baseline (XML)", marker=dict(size=18, color="red", symbol="star"),
                        hovertext=[f"<b>Baseline</b><br>{B['weight']:,.0f} lb<br>strength "
                                   f"{B['strength']:.2f}%"], hoverinfo="text"))
                    fig.update_layout(height=520, xaxis_title="Governing usage (%) — max of strength and deflection",
                                      yaxis_title="Shaft weight (lb)", legend=dict(orientation="h", y=-0.2),
                                      margin=dict(t=30))
                    ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="design_map",
                                         width="stretch")
                    st.caption("Click a point to load it into the detail check below. Hollow markers use a "
                               "long tube; grey × were verified but excluded by the long-tube rule.")
                    try:
                        pts = ev.selection.points if ev and ev.selection else []
                    except Exception:
                        pts = []
                    opts = [labels[id(e)] for e in pickable]
                    if pts and pts[0].get("customdata") is not None:
                        cd = pts[0]["customdata"]
                        cd = cd[0] if isinstance(cd, (list, tuple)) else cd
                        st.session_state["opt_pick"] = opts[int(cd)]

                    # ---- convergence
                    st.subheader("Convergence")
                    hist = res.get("history", [])
                    if hist:
                        fig2 = go.Figure()
                        xs = [h_[0] for h_ in hist] + [res["n_evals"]]
                        ys = [h_[2] for h_ in hist] + [hist[-1][2]]
                        fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line_shape="hv",
                                                  name="best screened weight",
                                                  hovertext=[f"{h_[2]:,.0f} lb at eval {h_[0]} ({h_[1]:.0f} s)"
                                                             for h_ in hist] + [""], hoverinfo="text"))
                        fig2.add_hline(y=B['weight'], line_dash="dash", line_color="red",
                                       annotation_text="baseline")
                        fig2.add_hline(y=win.weight, line_dash="dot", line_color="green",
                                       annotation_text="recommended (verified)")
                        fig2.update_layout(height=360, xaxis_title="Screening evaluations",
                                           yaxis_title="Shaft weight (lb)", margin=dict(t=30), showlegend=False)
                        st.plotly_chart(fig2, width="stretch")
                        st.caption("Best weight found during screening (5 governing cases) vs work done. "
                                   "The recommended design can differ slightly: ranking applies the tie band "
                                   "and length preferences, and only fully verified designs are eligible.")

                    # ---- alternates table
                    st.subheader(f"Next {len(res['alternates'])} designs and why they rank lower")
                    st.dataframe(pd.DataFrame([{
                        "rank": i + 2, **{k_: v_ for k_, v_ in summarize(a, C6).items() if k_ != "gov case"},
                        "Δ weight (lb)": round(a.weight - win.weight),
                        "why not preferred": why, "length variants": nv,
                    } for i, (a, why, nv) in enumerate(res["alternates"])]), width="stretch", hide_index=True)

                    # ---- baseline vs recommended
                    st.subheader("Baseline vs recommended")
                    st.dataframe(pd.DataFrame([
                        {"design": "Baseline (XML)", "D tip": spec6.tip_diameter,
                         "D base": round(spec6.base_diameter, 2), "taper": round(spec6.taper, 5),
                         "tubes (ft)": " / ".join(f"{t['length']:g}" for t in lay6),
                         "thickness (in)": " / ".join(f"{t['thickness']:g}" for t in lay6),
                         "weight (lb)": round(B['weight'])},
                        {"design": "Recommended", "D tip": sw["D tip (in)"], "D base": sw["D base (in)"],
                         "taper": sw["taper (in/ft)"], "tubes (ft)": sw["tube lengths (ft)"],
                         "thickness (in)": sw["thickness (in)"], "weight (lb)": sw["weight (lb)"]},
                    ]), width="stretch", hide_index=True)

                    # ---- detail
                    st.subheader("Detail check")
                    if st.session_state.get("opt_pick") not in opts:
                        st.session_state["opt_pick"] = opts[0]
                    pick = st.selectbox("Design", opts, key="opt_pick")
                    e = pickable[opts.index(pick)]
                    rr = e.result
                    st.dataframe(pd.DataFrame([summarize(e, C6)]), width="stretch", hide_index=True)
                    st.dataframe(pd.DataFrame([{
                        "load case": c, "max strength %": round(x.max_strength, 2),
                        "at ht AGL (ft)": round(x.gov_row.get('height_agl', float('nan')), 2),
                        "tip trans (in)": round(x.defl.tip_trans_in, 2),
                        "defl usage %": (round(x.defl_check['usage'], 2)
                                         if x.defl_check and x.defl_check.get('usage') is not None else None),
                    } for c, x in rr.cases.items()]), width="stretch", hide_index=True)
                    cg = rr.cases[rr.gov_strength_case]
                    fig7, ax7 = plt.subplots(figsize=(7, 5))
                    for c, x in rr.cases.items():
                        ax7.plot([q['usage'] for q in x.rows], [q['height_agl'] for q in x.rows],
                                 linewidth=2 if c == cg.case else 0.6,
                                 color='steelblue' if c == cg.case else 'lightgray')
                    ax7.axvline(C6.strength_target, color='red', linestyle='--', linewidth=1)
                    ax7.set_xlabel("Strength usage (%)"); ax7.set_ylabel("Height AGL (ft)")
                    ax7.set_title(f"All load cases (bold = {cg.case[:35]})"); ax7.grid(True, alpha=0.3)
                    st.pyplot(fig7); plt.close(fig7)

                    st.subheader("Export")
                    tube_csv = pd.DataFrame([{
                        "tube": t['tube_no'], "length_ft": t['length'], "thickness_in": t['thickness'],
                        "d_top_in": round(t['d_top'], 3), "d_bot_in": round(t['d_bot'], 3),
                        "joint_below": t['joint_type'], "lap_ft": t['lap'], "fy_ksi": t['fy'],
                    } for t in e.spec.layout()]).to_csv(index=False)
                    ce = st.columns(2)
                    ce[0].download_button("Tube table (CSV) for PLS-POLE re-verification", tube_csv,
                                          file_name=f"{pick.split(' ')[0]}_tubes.csv")
                    cons = {k_: (v_ if not isinstance(v_, tuple) else list(v_))
                            for k_, v_ in C6.__dict__.items()}
                    ce[1].download_button("Constraint set used (JSON)", json.dumps(cons, indent=2, default=str),
                                          file_name="constraints.json")
                    with st.expander("Constraint set used for this run"):
                        st.json(cons)
                if R["log"]:
                    with st.expander(f"Every candidate tried ({len(R['log'])})"):
                        st.dataframe(pd.DataFrame(R["log"]), width="stretch", hide_index=True)
        except Exception as e:
            show_error(e)
