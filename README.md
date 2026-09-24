# Pole Shaft Optimizer

A Streamlit web app for minimum-weight design and ASCE 48-19 strength verification of 12-sided tapered polygonal steel transmission poles. Reads a PLS-POLE XML export file and finds the lightest shaft that passes all load cases.

---

## What it does

Upload a PLS-POLE XML export and the app will:

1. **Parse** the XML — pole geometry, tube layout, load cases, and usage results
2. **Reconstruct geometry & weight** — reproduces PLS-POLE's per-tube weights to ~0.15%
3. **Build the load model** — shaft wind, self-weight, ice, and insulator attachment loads, all recomputed for the candidate geometry
4. **Run a second-order (P-delta) deflection solver** — geometrically nonlinear cantilever, iterates to convergence, validated against PLS-POLE's own deflection output
5. **Check ASCE 48-19 strength** — local buckling (Fa), bending (Fb), combined stress, and shear at section points along the shaft, matching PLS-POLE's perimeter-point stress checks
6. **Search for the minimum-weight shaft** — enumerates (tip diameter, base diameter, tube layout, wall thickness) combinations, screens on governing load cases, then fully verifies the best candidates on all load cases

---

## App tabs

| Tab | Description |
|-----|-------------|
| **PLS-POLE XML Parser** | Raw data browser — file metadata, table inventory, pole properties, load case list, governing usage summary |
| **Geometry & Weight** | Tube layout, section properties (w/t, I, S), weight breakdown, geometry constraint check, taper/diameter plots |
| **Load Model** | Load case parameters, first-order base reactions, point loads at attachments, shaft element loads, load distribution plots |
| **ASCE 48-19 Strength Check** | Max usage per load case, governing section detail, engine vs PLS-POLE usage comparison |
| **Deflection — P-delta solver** | Deflection limit check, tip deflection across all load cases, deflection profile plots |
| **Minimum-Weight Optimizer** | Full shaft optimization: baseline summary, recommended design, design map, convergence log, export results |

---

## Architecture

```
app.py                   Streamlit UI — all six tabs
pls_pole_xml_parser.py   XML parser (windows-1252 encoding, per-load-case tables)
geometry.py              PoleSpec, Segment, section properties, fabrication limits
loads.py                 Baseline (from XML) + LoadModel (recomputed per candidate)
deflection.py            P-delta cantilever solver (large-displacement)
strength.py              ASCE 48-19 section checks — calls mechanics_engine
mechanics_engine.py      Perimeter stress check (Fa, Fb, combined, shear)
weight.py                Steel weight from geometry (490 lb/ft³)
optimizer.py             Minimum-weight search (coarse screen → fine refine → verify)
extract_dataset.py       Utility to extract training/validation data from XML exports
check_all.py             Batch validation script against a set of reference XML files
```

---

## Getting started

### Requirements

```
streamlit
numpy
lxml
matplotlib
pandas
openpyxl
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Input file

Export your model from PLS-POLE via **File → Export XML**. The parser expects the standard PLS-POLE XML format (PLS-POLE Version 21.01+, windows-1252 encoding).

---

## Analysis settings (sidebar)

| Setting | Options | Notes |
|---------|---------|-------|
| **Inside bend radius** | 4.5t (matches PLS-POLE) / 4.0t (ASCE 48-19 cap) | Affects flat width `w` and the w/t → local buckling Fa calculation |
| **Shear for stress check** | resultant / transverse_only | Transverse-only reproduces PLS-POLE on tangent structures; resultant is conservative |
| **Slip-joint lap stiffness** | midpoint / outer / inner / sum | Controls which tube wall carries bending stiffness through a lap. `midpoint` (default) is conservative by +0.1% to +0.6% vs PLS-POLE tip deflection |

---

## Scope and assumptions

- **12-sided (dodecagonal) tapered poles only** — round poles are not supported
- **Slip joints and flange joints** — both modelled; taper is single and continuous
- **Embedment** — fixed input, not optimized; embedded poles taken as fixed at the ground line (rigid foundation)
- **Shaft wind and self-weight** recomputed for each candidate; wire and insulator loads carried over unchanged from the baseline XML
- **Davit arm self-weight and arm wind** are not modelled (minor residual per validation)
- **Base plate** weight is not included in the optimizer objective

---

## Validation

The engine has been verified against real PLS-POLE exports:

- Moment profiles agree to **< 1% mean error** on the governing load case
- Per-tube weights reproduced to **< 0.15%**; total shaft weight to **< 0.4 lb**
- Tip deflection reproduced to within **< 2%** (midpoint lap stiffness, base-plate pole)
- ASCE 48-19 Fa (local buckling) matches PLS-POLE when using the same bend radius factor
