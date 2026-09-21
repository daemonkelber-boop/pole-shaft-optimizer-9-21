import streamlit as st
import tempfile, os

st.set_page_config(page_title="Pole Shaft Optimizer", layout="wide")
st.title("🗼 Pole Shaft Optimizer")
st.caption("ASCE 48-19 | 12-sided tapered polygonal steel poles")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 1 · XML Parser",
    "📐 2 · Geometry & Weight",
    "💨 3 · Loads",
    "🔩 4 · Strength Check",
    "⚖️ 5 · Optimizer",
])

# ── TAB 1: XML PARSER ──────────────────────────────────────────────────────
with tab1:
    st.header("PLS-POLE XML Parser")
    uploaded = st.file_uploader("Upload a PLS-POLE XML export", type=["xml"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml, get_single_table,
                get_load_case_instances, get_field, LOAD_CASE_TAGNAMES
            )
            p = parse_pls_pole_xml(tmp_path)

            # Creator metadata
            st.subheader("File metadata")
            st.json(p['creator'])

            # Table inventory
            st.subheader("Table inventory")
            import pandas as pd
            rows = []
            for tagname, instances in sorted(p['tables'].items()):
                kind = ("per-load-case" if tagname in LOAD_CASE_TAGNAMES
                        else "repeated" if any(i['titledetail'] for i in instances)
                        else "single")
                rows.append({
                    "table": tagname,
                    "instances": len(instances),
                    "total rows": sum(i['nrows_actual'] for i in instances),
                    "kind": kind,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Pole properties
            st.subheader("Pole properties")
            prop = get_single_table(p, 'steel_pole_properties')
            if prop:
                st.json({k: v for k, v in prop[0].items()})

            # Tube properties
            st.subheader("Tube properties")
            tubes = get_single_table(p, 'steel_tubes_properties')
            if tubes:
                tube_rows = []
                for t in tubes:
                    tube_rows.append({
                        "tube #": get_field(t, 'tube_no'),
                        "length (ft)": get_field(t, 'length'),
                        "thickness (in)": get_field(t, 'thickness'),
                        "D top (in)": get_field(t, 'tube_top_diameter'),
                        "D bot (in)": get_field(t, 'tube_bot_diameter'),
                        "taper (in/ft)": get_field(t, 'calculated_taper'),
                        "lap (ft)": get_field(t, 'lap_length') or 0.0,
                        "weight (lb)": get_field(t, 'tube_weight'),
                        "Fy (ksi)": get_field(t, 'yield_stress'),
                    })
                st.dataframe(pd.DataFrame(tube_rows), use_container_width=True, hide_index=True)

            # Load cases
            st.subheader("Load cases")
            lcs = get_single_table(p, 'vector_load_cases')
            if lcs:
                lc_rows = []
                for r in lcs:
                    lc_rows.append({
                        "load case": r.get('load_case_description'),
                        "DLF": get_field(r, 'dead_load_factor'),
                        "q_trans (psf)": get_field(r, 'trans_wind_pressure'),
                        "q_long (psf)": get_field(r, 'longit_wind_pressure'),
                        "ice t (in)": get_field(r, 'ice_thick') or 0.0,
                    })
                st.dataframe(pd.DataFrame(lc_rows), use_container_width=True, hide_index=True)

            # Governing usage summary
            st.subheader("Governing usage summary")
            summ = get_single_table(p, 'summary_of_steel_pole_usages')
            if summ:
                st.json({k: v for k, v in summ[0].items()})

        except Exception as e:
            st.error(f"Parse error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            os.unlink(tmp_path)
    else:
        st.info("Upload a PLS-POLE XML file to begin.")

# ── TAB 2: GEOMETRY & WEIGHT ───────────────────────────────────────────────
with tab2:
    st.header("Geometry & Weight")
    uploaded2 = st.file_uploader("Upload a PLS-POLE XML export", type=["xml"], key="xml2")

    if uploaded2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            tmp.write(uploaded2.read())
            tmp_path2 = tmp.name

        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml as _parse, get_single_table as _gst,
                get_field as _gf
            )
            from geometry import PoleSpec, Segment, DesignLimits, build_sections, validate, describe
            from weight import pole_weight
            import matplotlib.pyplot as plt
            import numpy as np

            p2 = _parse(tmp_path2)
            prop2  = _gst(p2, 'steel_pole_properties')[0]
            tubes2 = _gst(p2, 'steel_tubes_properties')
            conn2  = _gst(p2, 'steel_pole_connectivity')

            # ---- Reconstruct PoleSpec from XML ----
            tip_d  = _gf(prop2, 'tip_diameter')
            taper  = _gf(tubes2[0], 'calculated_taper')
            is_bp  = str(prop2.get('base_plate', '')).lower() == 'yes'
            emb    = 0.0
            if not is_bp and conn2:
                emb = _gf(conn2[0], 'embed_override') or _gf(prop2, 'default_embedded_length') or 0.0

            segs = []
            for t in tubes2:
                lap = _gf(t, 'lap_length') or 0.0
                jtype = 'slip' if lap > 0 else 'flange'
                segs.append(Segment(
                    length=_gf(t, 'length'),
                    thickness=_gf(t, 'thickness'),
                    fy=_gf(t, 'yield_stress') or 65.0,
                    joint_type=jtype,
                ))

            spec = PoleSpec(
                label=prop2.get('steel_pole_property_label', ''),
                tip_diameter=tip_d,
                taper=taper,
                segments=segs,
                embedment=emb,
            )

            limits = DesignLimits()

            # ---- Summary cards ----
            st.subheader("Pole summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total length", f"{spec.total_length:.2f} ft")
            c2.metric("AGL height",   f"{spec.agl_height:.2f} ft")
            c3.metric("Tip D",        f"{spec.tip_diameter:.2f} in")
            c4.metric("Base D",       f"{spec.base_diameter:.2f} in")
            c5.metric("Taper",        f"{spec.taper:.5f} in/ft")

            # ---- Tube layout table ----
            st.subheader("Tube layout")
            lay = spec.layout()
            lay_rows = []
            for tb in lay:
                lay_rows.append({
                    "tube #":       tb['tube_no'],
                    "start (ft)":   round(tb['start'], 3),
                    "end (ft)":     round(tb['end'], 3),
                    "length (ft)":  round(tb['length'], 3),
                    "thickness (in)": tb['thickness'],
                    "D top (in)":   round(tb['d_top'], 3),
                    "D bot (in)":   round(tb['d_bot'], 3),
                    "joint":        tb['joint_type'],
                    "lap (ft)":     round(tb['lap'], 3),
                })
            st.dataframe(pd.DataFrame(lay_rows), use_container_width=True, hide_index=True)

            # ---- Weight breakdown ----
            st.subheader("Weight breakdown")
            w = pole_weight(spec)
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Total shaft weight", f"{w['total_weight']:,.0f} lb")
            wc2.metric("Above ground",       f"{w['above_ground_weight']:,.0f} lb")
            wc3.metric("Embedded steel",     f"{w['embedded_weight']:,.0f} lb")

            wtube_rows = []
            for pt in w['per_tube']:
                wtube_rows.append({
                    "tube #":        pt['tube_no'],
                    "length (ft)":   round(pt['length'], 3),
                    "thickness (in)": pt['thickness'],
                    "D top (in)":    round(pt['d_top'], 3),
                    "D bot (in)":    round(pt['d_bot'], 3),
                    "lap (ft)":      round(pt['lap'], 3),
                    "weight (lb)":   round(pt['weight'], 1),
                })
            st.dataframe(pd.DataFrame(wtube_rows), use_container_width=True, hide_index=True)

            # ---- Section table ----
            st.subheader("Section properties table")
            spacing_val = st.slider("Section spacing (ft)", 1.0, 10.0, 5.0, 0.5)
            secs = build_sections(spec, spacing=spacing_val)
            sec_rows = []
            for s in secs:
                sec_rows.append({
                    "rel_dist (ft)":  round(s['rel_dist'], 3),
                    "ht AGL (ft)":    round(s['height_agl'], 3),
                    "tube #":         s['tube_no'],
                    "in lap":         s['in_lap'],
                    "D (in)":         round(s['D'], 3),
                    "t (in)":         s['t'],
                    "w/t":            round(s['w_over_t'], 2),
                    "Ag (in²)":       round(s['Ag'], 3),
                    "I (in⁴)":        round(s['I'], 1),
                    "C (in)":         round(s['C'], 3),
                    "S (in³)":        round(s['S'], 3),
                    "below GL":       s['below_groundline'],
                })
            st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

            # ---- Diameter profile plot ----
            st.subheader("Diameter profile")
            fig, ax = plt.subplots(figsize=(7, 5))
            hgts = [spec.groundline_rel - s['rel_dist'] for s in secs]
            dias = [s['D'] for s in secs]
            ax.plot(dias, hgts, color='steelblue', linewidth=2)
            ax.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            for tb in lay:
                if tb['lap'] > 0:
                    lap_ht = spec.groundline_rel - (tb['end'] - tb['lap'])
                    ax.axhline(lap_ht, color='orange', linestyle=':', linewidth=1,
                               label='Slip joint' if tb == lay[0] else '')
            ax.set_xlabel("Flat-to-flat diameter (in)")
            ax.set_ylabel("Height above ground line (ft)")
            ax.set_title("Diameter vs height")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            # ---- w/t profile plot ----
            st.subheader("w/t ratio profile")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            wt_vals = [s['w_over_t'] for s in secs]
            ax2.plot(wt_vals, hgts, color='darkorange', linewidth=2)
            ax2.axvline(limits.max_wt, color='red', linestyle='--',
                        linewidth=1, label=f'Limit = {limits.max_wt}')
            ax2.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax2.set_xlabel("w/t")
            ax2.set_ylabel("Height above ground line (ft)")
            ax2.set_title("w/t vs height")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

            # ---- Constraint check ----
            st.subheader("Geometry constraint check")
            viols = validate(spec, limits)
            if viols:
                for v in viols:
                    st.warning(v)
            else:
                st.success("All geometry constraints satisfied.")

            # ---- Full describe() output ----
            with st.expander("Full pole description (text)"):
                st.text(describe(spec))

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            os.unlink(tmp_path2)
    else:
        st.info("Upload a PLS-POLE XML file to begin.")

# ── TAB 3: LOAD MODEL ─────────────────────────────────────────────────────
with tab3:
    st.header("Load Model")
    uploaded3 = st.file_uploader("Upload a PLS-POLE XML export", type=["xml"], key="xml3")

    if uploaded3:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            tmp.write(uploaded3.read())
            tmp_path3 = tmp.name

        try:
            from pls_pole_xml_parser import (
                parse_pls_pole_xml as _parse3, get_single_table as _gst3,
                get_field as _gf3, get_load_case_instances as _glci3
            )
            from geometry import PoleSpec, Segment
            from loads import Baseline, build_load_model, first_order_forces
            import matplotlib.pyplot as plt
            import numpy as np

            p3      = _parse3(tmp_path3)
            prop3   = _gst3(p3, 'steel_pole_properties')[0]
            tubes3  = _gst3(p3, 'steel_tubes_properties')
            conn3   = _gst3(p3, 'steel_pole_connectivity')
            summ3   = _gst3(p3, 'summary_of_steel_pole_usages')[0]

            # ---- Reconstruct PoleSpec ----
            is_bp3 = str(prop3.get('base_plate', '')).lower() == 'yes'
            emb3   = 0.0
            if not is_bp3 and conn3:
                emb3 = _gf3(conn3[0], 'embed_override') or \
                       _gf3(prop3, 'default_embedded_length') or 0.0
            segs3 = []
            for t in tubes3:
                lap = _gf3(t, 'lap_length') or 0.0
                segs3.append(Segment(
                    length=_gf3(t, 'length'),
                    thickness=_gf3(t, 'thickness'),
                    fy=_gf3(t, 'yield_stress') or 65.0,
                    joint_type='slip' if lap > 0 else 'flange',
                ))
            spec3 = PoleSpec(
                label=prop3.get('steel_pole_property_label', ''),
                tip_diameter=_gf3(prop3, 'tip_diameter'),
                taper=_gf3(tubes3[0], 'calculated_taper'),
                segments=segs3,
                embedment=emb3,
            )

            # ---- Baseline and load case list ----
            base3    = Baseline.from_xml(tmp_path3)
            all_cases = list(base3.load_cases.keys())
            gov_case  = summ3.get('load_case', all_cases[0])

            # Default index = governing case
            default_idx = all_cases.index(gov_case) if gov_case in all_cases else 0
            sel_case = st.selectbox(
                "Load case",
                all_cases,
                index=default_idx,
                help="Defaults to the governing case from PLS-POLE summary"
            )
            st.caption(f"Governing case per PLS-POLE: **{gov_case}**")

            # ---- Build load model ----
            m3  = build_load_model(spec3, base3, sel_case, ds=0.25)
            lc3 = m3.case

            # ---- Load case parameters ----
            st.subheader("Load case parameters")
            lp1, lp2, lp3, lp4, lp5 = st.columns(5)
            lp1.metric("DLF",          f"{lc3.dlf:.2f}")
            lp2.metric("q trans (psf)", f"{lc3.q_trans:.1f}")
            lp3.metric("q long (psf)",  f"{lc3.q_long:.1f}")
            lp4.metric("Ice t (in)",    f"{lc3.ice_t:.3f}")
            lp5.metric("Ice density",   f"{lc3.ice_density:.0f} pcf")

            # ---- Base reactions (first-order) ----
            st.subheader("Base reactions — first-order (no P-delta)")
            fo = first_order_forces(m3, spec3.total_length - 1e-9)
            br1, br2, br3, br4, br5, br6 = st.columns(6)
            br1.metric("P (kips)",  f"{fo['P']:+.3f}")
            br2.metric("Vx (kips)", f"{fo['Vx']:+.3f}")
            br3.metric("Vy (kips)", f"{fo['Vy']:+.3f}")
            br4.metric("Mx (ft-k)", f"{fo['Mx']:+.1f}")
            br5.metric("My (ft-k)", f"{fo['My']:+.1f}")
            br6.metric("T (ft-k)",  f"{fo['T']:+.3f}")

            # ---- Point loads at attachments ----
            st.subheader("Point loads at attachment points")
            if m3.points:
                pt_rows = []
                for pl in sorted(m3.points, key=lambda x: x.s):
                    pt_rows.append({
                        "label":       pl.label,
                        "s (ft)":      round(pl.s, 3),
                        "ht AGL (ft)": round(m3.z_of(pl.s), 3),
                        "dx (ft)":     round(pl.dx, 3),
                        "dy (ft)":     round(pl.dy, 3),
                        "dz (ft)":     round(pl.dz, 3),
                        "Fx (kips)":   round(pl.Fx, 4),
                        "Fy (kips)":   round(pl.Fy, 4),
                        "Fz (kips)":   round(pl.Fz, 4),
                    })
                st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

                # Totals
                tot_Fx = sum(pl.Fx for pl in m3.points)
                tot_Fy = sum(pl.Fy for pl in m3.points)
                tot_Fz = sum(pl.Fz for pl in m3.points)
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("∑ Fx attachment (kips)", f"{tot_Fx:+.3f}")
                tc2.metric("∑ Fy attachment (kips)", f"{tot_Fy:+.3f}")
                tc3.metric("∑ Fz attachment (kips)", f"{tot_Fz:+.3f}")
            else:
                st.info("No attachment point loads for this load case.")

            # ---- Per-element shaft loads ----
            st.subheader("Shaft element loads")
            el_rows = []
            for e in m3.elements:
                el_rows.append({
                    "s_top (ft)":    round(e.s_top, 3),
                    "s_bot (ft)":    round(e.s_bot, 3),
                    "ht AGL (ft)":   round(m3.z_of(e.s_mid), 3),
                    "above GL":      e.above_ground,
                    "D_wind (in)":   round(e.D_wind, 3),
                    "D_out (in)":    round(e.D_out, 3),
                    "Fy wind (k)":   round(e.fy, 5),
                    "Fx wind (k)":   round(e.fx, 5),
                    "Fz DL+ice (k)": round(e.fz, 5),
                })
            df_el = pd.DataFrame(el_rows)
            st.dataframe(df_el, use_container_width=True, hide_index=True)

            # Element load totals
            tot_fy_shaft = sum(e.fy for e in m3.elements)
            tot_fx_shaft = sum(e.fx for e in m3.elements)
            tot_fz_shaft = sum(e.fz for e in m3.elements)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("∑ Fy shaft wind (kips)",   f"{tot_fy_shaft:+.3f}")
            sc2.metric("∑ Fx shaft wind (kips)",   f"{tot_fx_shaft:+.3f}")
            sc3.metric("∑ Fz shaft DL+ice (kips)", f"{tot_fz_shaft:+.3f}")

            # ---- Load distribution plots ----
            st.subheader("Load distribution along shaft")
            above_els = [e for e in m3.elements if e.above_ground]
            hts  = [m3.z_of(e.s_mid) for e in above_els]
            fyw  = [e.fy * 1000 for e in above_els]   # kips -> lbs for clarity
            fzw  = [e.fz * 1000 for e in above_els]

            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 5))

            ax3a.barh(hts, fyw, height=0.2, color='steelblue', label='Trans wind (lb)')
            ax3a.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax3a.set_xlabel("Transverse wind load per element (lb)")
            ax3a.set_ylabel("Height above ground line (ft)")
            ax3a.set_title("Shaft transverse wind")
            ax3a.legend(fontsize=8)
            ax3a.grid(True, alpha=0.3)

            ax3b.barh(hts, fzw, height=0.2, color='darkorange', label='DL + ice (lb)')
            ax3b.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax3b.set_xlabel("Vertical load per element (lb)")
            ax3b.set_ylabel("Height above ground line (ft)")
            ax3b.set_title("Shaft self-weight + ice")
            ax3b.legend(fontsize=8)
            ax3b.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # ---- First-order moment diagram ----
            st.subheader("First-order moment diagram (transverse)")
            s_vals = np.linspace(0, spec3.groundline_rel, 60)
            mx_vals = [first_order_forces(m3, s)['Mx'] for s in s_vals]
            ht_vals = [m3.z_of(s) for s in s_vals]

            fig4, ax4 = plt.subplots(figsize=(7, 5))
            ax4.plot(mx_vals, ht_vals, color='steelblue', linewidth=2)
            ax4.axhline(0, color='brown', linestyle='--', linewidth=1, label='Ground line')
            ax4.axvline(0, color='gray', linewidth=0.5)
            ax4.fill_betweenx(ht_vals, mx_vals, 0, alpha=0.15, color='steelblue')
            ax4.set_xlabel("First-order Mx (ft-kips)")
            ax4.set_ylabel("Height above ground line (ft)")
            ax4.set_title(f"Transverse moment diagram — {sel_case}")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            plt.close(fig4)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            os.unlink(tmp_path3)
    else:
        st.info("Upload a PLS-POLE XML file to begin.")

with tab4:
    st.header("ASCE 48-19 Strength Check")
    st.info("🚧 Coming soon — mechanics_engine.py UI")

with tab5:
    st.header("Minimum-Weight Optimizer")
    st.info("🚧 Coming soon")
