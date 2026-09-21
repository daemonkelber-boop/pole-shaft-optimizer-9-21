"""
test_geometry_weight.py  --  validation gate for Phase 3 modules 1 and 2

Reconstructs each of the three reference poles from a compact PoleSpec and
compares the generated geometry and weight against what PLS-POLE reported.

The specs below contain ONLY design inputs -- tip diameter, taper, segment
lengths, wall thicknesses, joint types, embedment. Everything else
(diameters at every elevation, lap lengths, total length, section
properties, weight) is derived by geometry.py / weight.py. Nothing is
copied from the PLS-POLE output except the expected values used for
comparison.

Run:  python3 test_geometry_weight.py
"""

from geometry import (PoleSpec, Segment, DesignLimits, build_sections,
                      validate, describe, slip_joint_lap, thickness_from_wt,
                      w_over_t)
from weight import pole_weight

# --------------------------------------------------------------------------
# The three reference poles, as design inputs only.
# --------------------------------------------------------------------------

CASES = [
    dict(
        name='001_T316-S-01_105FT_FDN',
        spec=PoleSpec(
            label='001_T316-S-01_105FT_FDN',
            tip_diameter=35.00,
            taper=0.49167,
            embedment=0.0,                      # base plate
            segments=[
                Segment(length=55.00, thickness=0.5000, joint_type='flange'),
                Segment(length=50.00, thickness=0.6875, joint_type='flange',
                        top_diameter_override=62.42),
            ],
        ),
        expect=dict(total_length=105.0, base_diameter=87.00,
                    tube_weights=[14449, 27840], shaft_weight=42289,
                    laps=[0.0, 0.0],
                    pls_reported_weight=49898.8, base_plate_weight=7610),
    ),
    dict(
        name='004_T316-S-02_90FT_FDN',
        spec=PoleSpec(
            label='004_T316-S-02_90FT_FDN',
            tip_diameter=13.00,
            taper=0.32955,
            embedment=0.0,                      # base plate
            segments=[
                Segment(length=57.00, thickness=0.3125, joint_type='slip'),
                Segment(length=37.50, thickness=0.3750, joint_type='flange'),
            ],
        ),
        expect=dict(total_length=90.0, base_diameter=41.91,
                    tube_weights=[4303, 5440], shaft_weight=9743,
                    laps=[4.50, 0.0],
                    pls_reported_weight=11097.4, base_plate_weight=1354),
    ),
    dict(
        name='005_T316-S-03_145FT_EMB',
        spec=PoleSpec(
            label='005_T316-S-03_145FT_EMB',
            tip_diameter=16.00,
            taper=0.29655,
            embedment=20.5,                     # direct buried, FIXED
            segments=[
                Segment(length=53.00, thickness=0.2500, joint_type='slip'),
                Segment(length=50.25, thickness=0.3750, joint_type='slip'),
                Segment(length=26.75, thickness=0.4375, joint_type='slip'),
                Segment(length=32.50, thickness=0.4375, joint_type='flange'),
            ],
        ),
        expect=dict(total_length=145.0, base_diameter=56.50,
                    tube_weights=[3423, 7594, 5821, 7972], shaft_weight=24810,
                    laps=[4.50, 6.25, 6.75, 0.0],
                    pls_reported_weight=24809.7, base_plate_weight=0),
    ),
]


def pct(a, b):
    return (a / b - 1.0) * 100.0 if b else float('nan')


def main():
    limits = DesignLimits()
    all_ok = True

    for case in CASES:
        spec, exp = case['spec'], case['expect']
        print('=' * 74)
        print(describe(spec))

        # ---- laps -------------------------------------------------------
        laps = spec.laps()
        lap_ok = all(abs(a - b) < 1e-6 for a, b in zip(laps, exp['laps']))
        print(f"\n  laps          derived {[round(x,2) for x in laps]}"
              f"  expected {exp['laps']}   {'OK' if lap_ok else 'MISMATCH'}")

        # ---- total length ----------------------------------------------
        tl_ok = abs(spec.total_length - exp['total_length']) < 0.01
        print(f"  total length  {spec.total_length:8.2f} ft  "
              f"expected {exp['total_length']:8.2f}   {'OK' if tl_ok else 'MISMATCH'}")

        # ---- base diameter ---------------------------------------------
        bd_ok = abs(spec.base_diameter - exp['base_diameter']) < 0.06
        print(f"  base dia      {spec.base_diameter:8.2f} in  "
              f"expected {exp['base_diameter']:8.2f}   "
              f"{'OK' if bd_ok else 'MISMATCH'} ({pct(spec.base_diameter, exp['base_diameter']):+.2f}%)")

        # ---- weight ----------------------------------------------------
        w = pole_weight(spec)
        print(f"\n  {'tube':>5} {'derived lb':>12} {'PLS lb':>10} {'diff':>9}")
        tube_ok = True
        for pt, expw in zip(w['per_tube'], exp['tube_weights']):
            d = pct(pt['weight'], expw)
            if abs(d) > 0.5:
                tube_ok = False
            print(f"  {pt['tube_no']:>5} {pt['weight']:>12,.0f} {expw:>10,} {d:>8.2f}%")
        sw_diff = pct(w['total_weight'], exp['shaft_weight'])
        sw_ok = abs(sw_diff) < 0.5
        print(f"  {'TOTAL':>5} {w['total_weight']:>12,.0f} {exp['shaft_weight']:>10,} "
              f"{sw_diff:>8.2f}%   {'OK' if sw_ok else 'MISMATCH'}")

        recon = w['total_weight'] + exp['base_plate_weight']
        print(f"  shaft + base plate = {recon:,.0f} lb   "
              f"PLS reported {exp['pls_reported_weight']:,.1f} lb   "
              f"diff {recon - exp['pls_reported_weight']:+.1f} lb")

        if spec.embedment:
            print(f"  embedded steel {w['embedded_weight']:,.0f} lb "
                  f"({w['embedded_weight']/w['total_weight']*100:.1f}% of shaft), "
                  f"above ground {w['above_ground_weight']:,.0f} lb")
        if w['lap_double_counted_weight']:
            print(f"  steel in laps  {w['lap_double_counted_weight']:,.0f} lb")

        # ---- w/t and geometry limits ------------------------------------
        secs = build_sections(spec, spacing=5.0)
        wt_max = max(s['w_over_t'] for s in secs)
        print(f"\n  max w/t {wt_max:.2f} (limit {limits.max_wt})")

        viol = validate(spec, limits, target_length=exp['total_length'])
        if viol:
            print("  constraint violations:")
            for x in viol:
                print(f"    - {x}")
        else:
            print("  constraint violations: none")

        if spec.embedment:
            slips = [t for t in spec.layout() if t['lap_bottom'] is not None]
            if slips:
                low = max(slips, key=lambda t: t['lap_bottom'])
                clear = spec.groundline_rel - low['lap_bottom']
                print(f"  lowest slip joint bottom: {clear:.2f} ft above ground line "
                      f"(minimum {limits.min_base_joint_above_gl_ft})")

        ok = lap_ok and tl_ok and bd_ok and sw_ok and tube_ok
        all_ok = all_ok and ok
        print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")

    print('=' * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")

    # ---- rule spot checks ------------------------------------------------
    print("\nSlip joint rule, independent spot check:")
    for id_in, expect in [(61.040, 8.50), (31.155, 4.50), (31.220, 4.50),
                          (43.910, 6.25), (48.985, 6.75)]:
        got = slip_joint_lap(id_in / 12.0)
        print(f"  female ID {id_in:7.3f} in -> lap {got:5.2f} ft "
              f"(expected {expect:5.2f})  {'OK' if abs(got-expect)<1e-9 else 'MISMATCH'}")

    print("\nThickness recovered from D and w/t (for reading PLS-POLE tables):")
    for D, wt, t_true in [(87.00, 31.2, 0.6875), (35.00, 16.1, 0.5000),
                          (27.12, 26.4, 0.2500)]:
        got = thickness_from_wt(D, wt)
        print(f"  D={D:6.2f} w/t={wt:5.1f} -> t={got:.4f} "
              f"(actual {t_true})  {pct(got, t_true):+.2f}%")


if __name__ == '__main__':
    main()
