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

# ── TABS 2-5: COMING SOON ──────────────────────────────────────────────────
with tab2:
    st.header("Geometry & Weight")
    st.info("🚧 Coming soon — geometry.py + weight.py UI")

with tab3:
    st.header("Load Model")
    st.info("🚧 Coming soon — loads.py UI")

with tab4:
    st.header("ASCE 48-19 Strength Check")
    st.info("🚧 Coming soon — mechanics_engine.py UI")

with tab5:
    st.header("Minimum-Weight Optimizer")
    st.info("🚧 Coming soon")
