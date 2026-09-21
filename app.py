import streamlit as st
import tempfile, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Pole Shaft Optimizer", layout="wide")
st.title("🗼 Pole Shaft Optimizer")
st.caption("ASCE 48-19 | 12-sided tapered polygonal steel poles")

# ── SINGLE XML UPLOAD (sidebar) ────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload XML")
    uploaded = st.file_uploader("PLS-POLE XML export", type=["xml"])
    if uploaded:
        st.success(f"Loaded: {uploaded.name}")
    else:
        st.info("Upload a file to begin.")

tmp_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tmp.write(uploaded.read())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 1 · XML Parser",
    "📐 2 · Geometry & Weight",
    "💨 3 · Loads",
    "🔩 4 · Strength Check",
    "⚖️ 5 · Optimizer",
])

# ── SHARED HELPER: build PoleSpec from parsed XML ──────────────────────────
def build_spec(p, gf, gst):
    from geometry import PoleSpec, Segment
    prop  = gst(p, 'steel_pole_properties')[0]
    tubes = gst(p, 'steel_tubes_properties')
    conn  = gst(p, 'steel_pole_connectivity')
    is_bp = str(prop.get('base_plate', '')).lower() == 'yes'
    emb   = 0.0
    if not is_bp and conn:
        emb = gf(conn[0], 'embed_override') or \
              gf(prop, 'default_embedded_length') or 0.0
    segs = []
    for t in tubes:
        lap = gf(t, 'lap_length') or 0.0
        segs.append(Segment(
            length=gf(t, 'length'),
            thickness=gf(t, 'thickness'),
            fy=gf(t, 'yield_stress') or 65.0,
            joint_type='slip' if lap > 0 else 'flange',
        ))
    return PoleSpec(
        label=prop.get('steel_pole_property_label', ''),
        tip_diameter=gf(prop, 'tip_diameter'),
        taper=gf(tubes[0], 'calculated_taper'),
        segments=segs,
        embedment=emb,
    )

# ── TAB 1: XML PARSER ──────────────────────────────────────────────────────
with tab1:
    st.header("PLS-POLE XML Parser")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml, get_single_table, get_field,
                LOAD_CASE_TAGNAMES
            )
            p = parse_pls_pole_xml(tmp_path)

            st.subheader("File metadata")
            st.json(p['creator'])

            st.subheader("Table inventory")
            inv = []
            for tagname, instances in sorted(p['tables'].items()):
                kind = ("per-load-case" if tagname in LOAD_CASE_TAGNAMES
                        else "repeated" if any(i['titledetail'] for i in instances)
                        else "single")
                inv.append({
                    "table":      tagname,
                    "instances":  len(instances),
                    "total rows": sum(i['nrows_actual'] for i in instances),
                    "kind":       kind,
                })
            st.dataframe(pd.DataFrame(inv), use_container_width=True, hide_index=True)

            st.subheader("Pole properties")
            prop1 = get_single_table(p, 'steel_pole_properties')
            if prop1:
                st.json({k: v for k, v in prop1[0].items()})

            st.subheader("Tube properties")
            tubes1 = get_single_table(p, 'steel_tubes_properties')
            if tubes1:
                tr = []
                for t in tubes1:
                    tr.append({
                        "tube #":         get_field(t, 'tube_no'),
                        "length (ft)":    get_field(t, 'length'),
                        "thickness (in)": get_field(t, 'thickness'),
                        "D top (in)":     get_field(t, 'tube_top_diameter'),
                        "D bot (in)":     get_field(t, 'tube_bot_diameter'),
                        "taper (in/ft)":  get_field(t, 'calculated_taper'),
                        "lap (ft)":       get_field(t, 'lap_length') or 0.0,
                        "weight (lb)":    get_field(t, 'tube_weight'),
                        "Fy (ksi)":       get_field(t, 'yield_stress'),
                    })
                st.dataframe(pd.DataFrame(tr), use_container_width=True, hide_index=True)

            st.subheader("Load cases")
            lcs1 = get_single_table(p, 'vector_load_cases')
            if lcs1:
                lcr = []
                for r in lcs1:
                    lcr.append({
                        "load case":     r.get('load_case_description'),
                        "DLF":           get_field(r, 'dead_load_factor'),
                        "q_trans (psf)": get_field(r, 'trans_wind_pressure'),
                        "q_long (psf)":  get_field(r, 'longit_wind_pressure'),
                        "ice t (in)":    get_field(r, 'ice_thick') or 0.0,
                    })
                st.dataframe(pd.DataFrame(lcr), use_container_width=True, hide_index=True)

            st.subheader("Governing usage summary")
            summ1 = get_single_table(p, 'summary_of_steel_pole_usages')
            if summ1:
                st.json({k: v for k, v in summ1[0].items()})

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ── TAB 2: GEOMETRY & WEIGHT ───────────────────────────────────────────────
with tab2:
    st.header("Geometry & Weight")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml as _p2, get_single_table as _gst2,
                get_field as _gf2
            )
            from geometry import DesignLimits, build_sections, validate, describe
            from weight import pole_weight

            p2   = _p2(tmp_path)
            spec2 = build_spec(p2, _gf2, _gst2)
            limits = DesignLimits()
            lay2   = spec2.layout()
            secs2  = build_sections(spec2, spacing=5.0)
            hgts2  = [spec2.groundline_rel - s['rel_dist'] for s in secs2]

            st.subheader("Pole summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total length", f"{spec2.total_length:.2f} ft")
            c2.metric("AGL height",   f"{spec2.agl_height:.2f} ft")
            c3.metric("Tip D",        f"{spec2.tip_diameter:.2f} in")
            c4.metric("Base D",       f"{spec2.base_diameter:.2f} in")
            c5.metric("Taper",        f"{spec2.taper:.5f} in/ft")

            st.subheader("Tube layout")
            st.dataframe(pd.DataFrame([{
                "tube #":         tb['tube_no'],
                "start (ft)":     round(tb['start'], 3),
                "end (ft)":       round(tb['end'], 3),
                "length (ft)":    round(tb['length'], 3),
                "thickness (in)": tb['thickness'],
                "D top (in)":     round(tb['d_top'], 3),
                "D bot (in)":     round(tb['d_bot'], 3),
                "joint":          tb['joint_type'],
                "lap (ft)":       round(tb['lap'], 3),
            } for tb in lay2]), use_container_width=True, hide_index=True)

            st.subheader("Weight breakdown")
            w2 = pole_weight(spec2)
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Total shaft weight", f"{w2['total_weight']:,.0f} lb")
            wc2.metric("Above ground",       f"{w2['above_ground_weight']:,.0f} lb")
            wc3.metric("Embedded steel",     f"{w2['embedded_weight']:,.0f} lb")
            st.dataframe(pd.DataFrame([{
                "tube #":         pt['tube_no'],
                "length (ft)":    round(pt['length'], 3),
                "thickness (in)": pt['thickness'],
                "D top (in)":     round(pt['d_top'], 3),
                "D bot (in)":     round(pt['d_bot'], 3),
                "lap (ft)":       round(pt['lap'], 3),
                "weight (lb)":    round(pt['weight'], 1),
            } for pt in w2['per_tube']]), use_container_width=True, hide_index=True)

            st.subheader("Section properties table")
            spacing_val = st.slider("Section spacing (ft)", 1.0, 10.0, 5.0, 0.5)
            secs2s = build_sections(spec2, spacing=spacing_val)
            st.dataframe(pd.DataFrame([{
                "rel_dist (ft)": round(s['rel_dist'], 3),
                "ht AGL (ft)":   round(s['height_agl'], 3),
                "tube #":        s['tube_no'],
                "in lap":        s['in_lap'],
                "D (in)":        round(s['D'], 3),
                "t (in)":        s['t'],
                "w/t":           round(s['w_over_t'], 2),
                "Ag (in²)":      round(s['Ag'], 3),
                "I (in⁴)":       round(s['I'], 1),
                "C (in)":        round(s['C'], 3),
                "S (in³)":       round(s['S'], 3),
                "below GL":      s['below_groundline'],
            } for s in secs2s]), use_container_width=True, hide_index=True)

            st.subheader("Diameter profile")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot([s['D'] for s in secs2], hgts2, color='steelblue', linewidth=2)
            ax.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            for tb in lay2:
                if tb['lap'] > 0:
                    ax.axhline(spec2.groundline_rel - (tb['end'] - tb['lap']),
                               color='orange', linestyle=':', linewidth=1,
                               label='Slip joint' if tb == lay2[0] else '')
            ax.set_xlabel("Flat-to-flat diameter (in)")
            ax.set_ylabel("Height above ground line (ft)")
            ax.set_title("Diameter vs height")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close(fig)

            st.subheader("w/t ratio profile")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot([s['w_over_t'] for s in secs2], hgts2, color='darkorange', linewidth=2)
            ax2.axvline(limits.max_wt, color='red', linestyle='--',
                        linewidth=1, label=f'Limit = {limits.max_wt}')
            ax2.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax2.set_xlabel("w/t"); ax2.set_ylabel("Height above ground line (ft)")
            ax2.set_title("w/t vs height")
            ax2.legend(); ax2.grid(True, alpha=0.3)
            st.pyplot(fig2); plt.close(fig2)

            st.subheader("Geometry constraint check")
            viols = validate(spec2, limits)
            if viols:
                for v in viols: st.warning(v)
            else:
                st.success("All geometry constraints satisfied.")

            with st.expander("Full pole description (text)"):
                st.text(describe(spec2))

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ── TAB 3: LOAD MODEL ─────────────────────────────────────────────────────
with tab3:
    st.header("Load Model")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml as _p3, get_single_table as _gst3,
                get_field as _gf3
            )
            from loads import Baseline, build_load_model, first_order_forces

            p3    = _p3(tmp_path)
            spec3 = build_spec(p3, _gf3, _gst3)
            summ3 = _gst3(p3, 'summary_of_steel_pole_usages')[0]
            base3 = Baseline.from_xml(tmp_path)

            all_cases3  = list(base3.load_cases.keys())
            gov_case3   = summ3.get('load_case', all_cases3[0])
            def3_idx    = all_cases3.index(gov_case3) if gov_case3 in all_cases3 else 0
            sel_case3   = st.selectbox("Load case", all_cases3, index=def3_idx,
                                       key="lc3",
                                       help="Defaults to governing case from PLS-POLE summary")
            st.caption(f"Governing case per PLS-POLE: **{gov_case3}**")

            m3  = build_load_model(spec3, base3, sel_case3, ds=0.25)
            lc3 = m3.case

            st.subheader("Load case parameters")
            lp1, lp2, lp3, lp4, lp5 = st.columns(5)
            lp1.metric("DLF",           f"{lc3.dlf:.2f}")
            lp2.metric("q trans (psf)", f"{lc3.q_trans:.1f}")
            lp3.metric("q long (psf)",  f"{lc3.q_long:.1f}")
            lp4.metric("Ice t (in)",    f"{lc3.ice_t:.3f}")
            lp5.metric("Ice density",   f"{lc3.ice_density:.0f} pcf")

            st.subheader("Base reactions — first-order (no P-delta)")
            fo3 = first_order_forces(m3, spec3.total_length - 1e-9)
            br1, br2, br3, br4, br5, br6 = st.columns(6)
            br1.metric("P (kips)",  f"{fo3['P']:+.3f}")
            br2.metric("Vx (kips)", f"{fo3['Vx']:+.3f}")
            br3.metric("Vy (kips)", f"{fo3['Vy']:+.3f}")
            br4.metric("Mx (ft-k)", f"{fo3['Mx']:+.1f}")
            br5.metric("My (ft-k)", f"{fo3['My']:+.1f}")
            br6.metric("T (ft-k)",  f"{fo3['T']:+.3f}")

            st.subheader("Point loads at attachment points")
            if m3.points:
                st.dataframe(pd.DataFrame([{
                    "label":       pl.label,
                    "s (ft)":      round(pl.s, 3),
                    "ht AGL (ft)": round(m3.z_of(pl.s), 3),
                    "dx (ft)":     round(pl.dx, 3),
                    "dy (ft)":     round(pl.dy, 3),
                    "dz (ft)":     round(pl.dz, 3),
                    "Fx (kips)":   round(pl.Fx, 4),
                    "Fy (kips)":   round(pl.Fy, 4),
                    "Fz (kips)":   round(pl.Fz, 4),
                } for pl in sorted(m3.points, key=lambda x: x.s)]),
                    use_container_width=True, hide_index=True)
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("∑ Fx (kips)", f"{sum(pl.Fx for pl in m3.points):+.3f}")
                tc2.metric("∑ Fy (kips)", f"{sum(pl.Fy for pl in m3.points):+.3f}")
                tc3.metric("∑ Fz (kips)", f"{sum(pl.Fz for pl in m3.points):+.3f}")
            else:
                st.info("No attachment point loads for this load case.")

            st.subheader("Shaft element loads")
            st.dataframe(pd.DataFrame([{
                "s_top (ft)":    round(e.s_top, 3),
                "s_bot (ft)":    round(e.s_bot, 3),
                "ht AGL (ft)":   round(m3.z_of(e.s_mid), 3),
                "above GL":      e.above_ground,
                "D_wind (in)":   round(e.D_wind, 3),
                "D_out (in)":    round(e.D_out, 3),
                "Fy wind (k)":   round(e.fy, 5),
                "Fx wind (k)":   round(e.fx, 5),
                "Fz DL+ice (k)": round(e.fz, 5),
            } for e in m3.elements]), use_container_width=True, hide_index=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("∑ Fy shaft wind (kips)",   f"{sum(e.fy for e in m3.elements):+.3f}")
            sc2.metric("∑ Fx shaft wind (kips)",   f"{sum(e.fx for e in m3.elements):+.3f}")
            sc3.metric("∑ Fz shaft DL+ice (kips)", f"{sum(e.fz for e in m3.elements):+.3f}")

            st.subheader("Load distribution along shaft")
            above3 = [e for e in m3.elements if e.above_ground]
            hts3   = [m3.z_of(e.s_mid) for e in above3]
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 5))
            ax3a.barh(hts3, [e.fy * 1000 for e in above3], height=0.2, color='steelblue')
            ax3a.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax3a.set_xlabel("Transverse wind load per element (lb)")
            ax3a.set_ylabel("Height AGL (ft)"); ax3a.set_title("Shaft transverse wind")
            ax3a.legend(); ax3a.grid(True, alpha=0.3)
            ax3b.barh(hts3, [e.fz * 1000 for e in above3], height=0.2, color='darkorange')
            ax3b.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax3b.set_xlabel("Vertical load per element (lb)")
            ax3b.set_ylabel("Height AGL (ft)"); ax3b.set_title("Shaft self-weight + ice")
            ax3b.legend(); ax3b.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

            st.subheader("First-order moment diagram (transverse)")
            s_vals  = np.linspace(0, spec3.groundline_rel, 60)
            mx_vals = [first_order_forces(m3, s)['Mx'] for s in s_vals]
            ht_vals = [m3.z_of(s) for s in s_vals]
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            ax4.plot(mx_vals, ht_vals, color='steelblue', linewidth=2)
            ax4.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax4.axvline(0, color='gray', linewidth=0.5)
            ax4.fill_betweenx(ht_vals, mx_vals, 0, alpha=0.15, color='steelblue')
            ax4.set_xlabel("First-order Mx (ft-kips)"); ax4.set_ylabel("Height AGL (ft)")
            ax4.set_title(f"Transverse moment diagram — {sel_case3}")
            ax4.legend(); ax4.grid(True, alpha=0.3)
            st.pyplot(fig4); plt.close(fig4)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ── TAB 4: ASCE 48-19 STRENGTH CHECK ──────────────────────────────────────
with tab4:
    st.header("ASCE 48-19 Strength Check")
    if not tmp_path:
        st.info("Upload a PLS-POLE XML file in the sidebar to begin.")
    else:
        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml as _p4, get_single_table as _gst4,
                get_field as _gf4, get_load_case_instances as _glci4
            )
            from geometry import (build_sections as _bs4, w_over_t as _wot4,
                                  thickness_from_wt as _tfw4)
            from loads import Baseline as _BL4, build_load_model as _blm4, first_order_forces as _fof4
            from mechanics_engine import combined_stress_check

            p4    = _p4(tmp_path)
            spec4 = build_spec(p4, _gf4, _gst4)
            summ4 = _gst4(p4, 'summary_of_steel_pole_usages')[0]
            base4 = _BL4.from_xml(tmp_path)
            lay4  = spec4.layout()

            all_cases4  = list(base4.load_cases.keys())
            gov_case4   = summ4.get('load_case', all_cases4[0])
            def4_idx    = all_cases4.index(gov_case4) if gov_case4 in all_cases4 else 0

            col_a, col_b = st.columns([2, 1])
            with col_a:
                sel_case4 = st.selectbox("Load case", all_cases4, index=def4_idx,
                                         key="lc4",
                                         help="Defaults to governing case from PLS-POLE summary")
                st.caption(f"Governing case per PLS-POLE: **{gov_case4}**")
            with col_b:
                section_mode = st.radio("Section points",
                                        ["PLS-POLE points", "Custom spacing"],
                                        key="secmode4")
                if section_mode == "Custom spacing":
                    chk_spacing = st.slider("Spacing (ft)", 1.0, 10.0, 5.0, 0.5, key="sp4")

            shear_mode4 = st.radio(
                "Shear mode", ["transverse_only", "resultant"],
                index=0, horizontal=True,
                help="'transverse_only' matches PLS-POLE calibration. "
                     "'resultant' is the textbook interpretation.",
                key="shear4"
            )

            m4  = _blm4(spec4, base4, sel_case4, ds=0.25)
            usages4   = _glci4(p4, 'detailed_steel_pole_usages')
            pls_rows4 = sorted(usages4.get(sel_case4, []),
                               key=lambda r: _gf4(r, 'rel_dist'))

            # ---- Build check points ----
            if section_mode == "PLS-POLE points":
                check_pts = []
                for r in pls_rows4:
                    rd  = _gf4(r, 'rel_dist')
                    # D: prefer XML field, fall back to spec geometry
                    D   = _gf4(r, 'outer_diameter') or spec4.diameter_at(rd)
                    wt  = _gf4(r, 'w_t_max') or 0.0
                    # t: recover from w/t if available, else use spec
                    t   = (_tfw4(D, wt) if wt > 0 else spec4.thickness_at(rd))
                    # recompute w/t from spec if XML gave zero
                    if wt == 0 and t and t > 0:
                        wt = _wot4(D, t)
                    fy  = next((tb['fy'] for tb in lay4
                                if tb['start'] - 1e-6 <= rd <= tb['end'] + 1e-6), 65.0)
                    check_pts.append({
                        'rd': rd, 'D': D, 't': t, 'wt': wt, 'fy': fy,
                        'pls_usage': _gf4(r, 'max_usage'),
                        'pls_pa':    _gf4(r, 'p_a'),
                        'pls_ms':    _gf4(r, 'm_s'),
                        'pls_vq':    _gf4(r, 'v_q'),
                        'pls_tr':    _gf4(r, 't_r'),
                        'pls_res':   _gf4(r, 'res'),
                        'pls_Fa':    _gf4(r, 'fa_min'),
                        'joint_pos': r.get('joint_position', ''),
                        'P':  _gf4(r, 'axial_force') or 0.0,
                        'Mx': _gf4(r, 'trans_mom_local_mx') or 0.0,
                        'My': _gf4(r, 'long_mom_local_my') or 0.0,
                        'Vt': _gf4(r, 'tran_shear') or 0.0,
                        'Vl': _gf4(r, 'long_shear') or 0.0,
                        'T':  _gf4(r, 'tors_mom') or 0.0,
                    })
            else:
                secs4c = _bs4(spec4, spacing=chk_spacing)
                check_pts = []
                for s in secs4c:
                    if s['below_groundline']:
                        continue
                    rd    = s['rel_dist']
                    fo4c  = _fof4(m4, rd)
                    check_pts.append({
                        'rd': rd, 'D': s['D'], 't': s['t'],
                        'wt': s['w_over_t'], 'fy': s['fy'],
                        'pls_usage': None,
                        'pls_pa': None, 'pls_ms': None,
                        'pls_vq': None, 'pls_tr': None,
                        'pls_res': None, 'pls_Fa': None,
                        'joint_pos': '',
                        'P':  fo4c['P'],  'Mx': fo4c['Mx'], 'My': fo4c['My'],
                        'Vt': fo4c['Vy'], 'Vl': fo4c['Vx'], 'T':  fo4c['T'],
                    })

            # ---- Run strength check ----
            results4 = []
            for cp in check_pts:
                if not cp['D'] or not cp['t'] or cp['t'] <= 0 or not cp['wt']:
                    continue
                chk = combined_stress_check(
                    P=cp['P'], Mx=cp['Mx'], My=cp['My'],
                    V_tran=cp['Vt'], V_long=cp['Vl'], Torsion_ftk=cp['T'],
                    D=cp['D'], t=cp['t'], Fy=cp['fy'], w_over_t=cp['wt'],
                    shear_mode=shear_mode4
                )
                ht = m4.z_of(cp['rd'])
                results4.append({
                    'rel_dist (ft)': round(cp['rd'], 3),
                    'ht AGL (ft)':   round(ht, 3),
                    'joint_pos':     cp['joint_pos'],
                    'D (in)':        round(cp['D'], 3),
                    't (in)':        round(cp['t'], 4),
                    'w/t':           round(cp['wt'], 2),
                    'Fy (ksi)':      cp['fy'],
                    'Fa (ksi)':      round(chk['Fa'], 3) if chk['Fa'] else None,
                    'Fa eq':         chk['Fa_equation'],
                    'p_a (ksi)':     round(chk['p_a'], 4),
                    'm_s (ksi)':     round(chk['m_s'], 4),
                    'v_q (ksi)':     round(chk['v_q'], 4),
                    't_r (ksi)':     round(chk['t_r'], 4),
                    'res (ksi)':     round(chk['res'], 4),
                    'calc usage %':  round(chk['usage'], 2) if chk['usage'] else None,
                    'PLS usage %':   round(cp['pls_usage'], 2) if cp['pls_usage'] else None,
                    'diff (pts)':    round(chk['usage'] - cp['pls_usage'], 2)
                                     if (chk['usage'] and cp['pls_usage']) else None,
                    'PLS p_a':  cp['pls_pa'],  'PLS m_s': cp['pls_ms'],
                    'PLS v_q':  cp['pls_vq'],  'PLS t_r': cp['pls_tr'],
                    'PLS res':  cp['pls_res'],  'PLS Fa':  cp['pls_Fa'],
                })

            df4 = pd.DataFrame(results4)

            if df4.empty:
                st.warning("No valid check points produced. Check the debug expander below.")
                with st.expander("🔍 Debug: raw check points"):
                    st.dataframe(pd.DataFrame(check_pts),
                                 use_container_width=True, hide_index=True)
            else:
                # ---- Governing section ----
                valid4 = df4[df4['calc usage %'].notna()]
                if not valid4.empty:
                    gov4 = valid4.loc[valid4['calc usage %'].idxmax()]
                    st.subheader("Governing section")
                    g1, g2, g3, g4c, g5 = st.columns(5)
                    g1.metric("Max usage",   f"{gov4['calc usage %']:.2f}%")
                    g2.metric("At ht AGL",   f"{gov4['ht AGL (ft)']:.2f} ft")
                    g3.metric("D / t",       f"{gov4['D (in)']:.2f} / {gov4['t (in)']:.4f}")
                    g4c.metric("Fa (ksi)",   f"{gov4['Fa (ksi)']:.3f}")
                    g5.metric("Fa equation", str(gov4['Fa eq']))
                    if gov4['PLS usage %']:
                        d1, d2 = st.columns(2)
                        d1.metric("PLS-POLE usage at same point",
                                  f"{gov4['PLS usage %']:.2f}%")
                        d2.metric("Difference", f"{gov4['diff (pts)']:+.2f} pts")

                # ---- Full results table ----
                st.subheader("Full section check results")
                display_cols = ['rel_dist (ft)', 'ht AGL (ft)', 'D (in)', 't (in)',
                                'w/t', 'Fa (ksi)', 'p_a (ksi)', 'm_s (ksi)',
                                'v_q (ksi)', 't_r (ksi)', 'res (ksi)',
                                'calc usage %', 'PLS usage %', 'diff (pts)']
                st.dataframe(df4[[c for c in display_cols if c in df4.columns]],
                             use_container_width=True, hide_index=True)

                # ---- Usage profile plot ----
                st.subheader("Usage % vs height")
                fig4a, ax4a = plt.subplots(figsize=(7, 6))
                ax4a.plot(valid4['calc usage %'], valid4['ht AGL (ft)'],
                          color='steelblue', linewidth=2, label='Calc (this engine)')
                if valid4['PLS usage %'].notna().any():
                    ax4a.plot(valid4['PLS usage %'], valid4['ht AGL (ft)'],
                              color='darkorange', linewidth=1.5, linestyle='--',
                              label='PLS-POLE reported')
                ax4a.axvline(100, color='red', linestyle='--', linewidth=1, label='100% limit')
                ax4a.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
                ax4a.set_xlabel("Usage (%)")
                ax4a.set_ylabel("Height above ground line (ft)")
                ax4a.set_title(f"Usage profile — {sel_case4}")
                ax4a.legend(); ax4a.grid(True, alpha=0.3)
                st.pyplot(fig4a); plt.close(fig4a)

                # ---- Stress components plot ----
                st.subheader("Stress components vs height")
                fig4b, ax4b = plt.subplots(figsize=(7, 6))
                ax4b.plot(valid4['p_a (ksi)'], valid4['ht AGL (ft)'],
                          label='p_a', linewidth=1.5)
                ax4b.plot(valid4['m_s (ksi)'], valid4['ht AGL (ft)'],
                          label='m_s', linewidth=1.5)
                ax4b.plot(valid4['v_q (ksi)'], valid4['ht AGL (ft)'],
                          label='v_q', linewidth=1.5)
                ax4b.plot(valid4['t_r (ksi)'], valid4['ht AGL (ft)'],
                          label='t_r', linewidth=1.5)
                ax4b.plot(valid4['res (ksi)'], valid4['ht AGL (ft)'],
                          label='resultant', linewidth=2,
                          color='black', linestyle='--')
                ax4b.axhline(0, color='brown', linestyle='--',
                             linewidth=1, label='Ground line')
                ax4b.set_xlabel("Stress (ksi)")
                ax4b.set_ylabel("Height above ground line (ft)")
                ax4b.set_title(f"Stress components — {sel_case4}")
                ax4b.legend(fontsize=8); ax4b.grid(True, alpha=0.3)
                st.pyplot(fig4b); plt.close(fig4b)

                # ---- Engine vs PLS difference plot ----
                if section_mode == "PLS-POLE points":
                    diff_valid = df4[df4['diff (pts)'].notna()]
                    if not diff_valid.empty:
                        st.subheader("Engine vs PLS-POLE usage difference")
                        fig4c, ax4c = plt.subplots(figsize=(7, 5))
                        colors = ['red' if abs(d) > 2 else 'steelblue'
                                  for d in diff_valid['diff (pts)']]
                        ax4c.barh(diff_valid['ht AGL (ft)'],
                                  diff_valid['diff (pts)'],
                                  height=0.4, color=colors)
                        ax4c.axvline(0,  color='black', linewidth=0.8)
                        ax4c.axvline(2,  color='red', linestyle='--',
                                     linewidth=0.8, label='±2 pt band')
                        ax4c.axvline(-2, color='red', linestyle='--', linewidth=0.8)
                        ax4c.axhline(0,  color='brown', linestyle='--',
                                     linewidth=1, label='Ground line')
                        ax4c.set_xlabel("Calc usage % − PLS usage % (pts)")
                        ax4c.set_ylabel("Height above ground line (ft)")
                        ax4c.set_title(f"Engine vs PLS-POLE — {sel_case4}")
                        ax4c.legend(fontsize=8); ax4c.grid(True, alpha=0.3)
                        st.pyplot(fig4c); plt.close(fig4c)

                        diffs = diff_valid['diff (pts)']
                        st1, st2, st3, st4s = st.columns(4)
                        st1.metric("Mean diff",    f"{diffs.mean():+.3f} pts")
                        st2.metric("Max |diff|",   f"{diffs.abs().max():.3f} pts")
                        st3.metric("Std dev",      f"{diffs.std():.3f} pts")
                        st4s.metric("Points > ±2", f"{(diffs.abs() > 2).sum()}")

                # ---- Detailed component comparison ----
                with st.expander("Detailed component comparison (calc vs PLS-POLE)"):
                    comp_cols = ['rel_dist (ft)', 'ht AGL (ft)',
                                 'p_a (ksi)', 'PLS p_a',
                                 'm_s (ksi)', 'PLS m_s',
                                 'v_q (ksi)', 'PLS v_q',
                                 't_r (ksi)', 'PLS t_r',
                                 'res (ksi)', 'PLS res',
                                 'Fa (ksi)',  'PLS Fa',
                                 'calc usage %', 'PLS usage %', 'diff (pts)']
                    st.dataframe(df4[[c for c in comp_cols if c in df4.columns]],
                                 use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ── TAB 5: OPTIMIZER ───────────────────────────────────────────────────────
with tab5:
    st.header("Minimum-Weight Optimizer")
    st.info("🚧 Coming soon")

# ── CLEANUP ────────────────────────────────────────────────────────────────
if tmp_path and os.path.exists(tmp_path):
    os.unlink(tmp_path)
