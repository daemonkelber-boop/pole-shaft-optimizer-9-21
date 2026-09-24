"""
test_thick_len_constraint.py

Unit tests for the thickness-dependent section length constraints in optimizer.py
and app.py.
"""

from optimizer import OptConstraints, Design, make_spec, GAUGES
from geometry import Segment, PoleSpec

def test_max_length_for_thickness():
    # 1. Test 'gte' mode
    c_gte = OptConstraints(max_len_by_thick={0.75: 50.0, 0.875: 40.0}, thick_len_mode='gte')
    assert c_gte.max_length_for_thickness(0.5) is None
    assert c_gte.max_length_for_thickness(0.6875) is None
    assert c_gte.max_length_for_thickness(0.75) == 50.0
    assert c_gte.max_length_for_thickness(0.8125) == 50.0
    assert c_gte.max_length_for_thickness(0.875) == 40.0
    assert c_gte.max_length_for_thickness(1.0) == 40.0

    # 2. Test 'exact' mode
    c_exact = OptConstraints(max_len_by_thick={0.75: 50.0, 0.875: 40.0}, thick_len_mode='exact')
    assert c_exact.max_length_for_thickness(0.5) is None
    assert c_exact.max_length_for_thickness(0.75) == 50.0
    assert c_exact.max_length_for_thickness(0.8125) is None
    assert c_exact.max_length_for_thickness(0.875) == 40.0
    assert c_exact.max_length_for_thickness(1.0) is None

    # 3. Test empty constraints
    c_empty = OptConstraints()
    assert c_empty.max_length_for_thickness(0.75) is None

    print("PASS: test_max_length_for_thickness")


def test_make_spec_thickness_length_validation():
    G = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    H = 100.0
    emb = 0.0

    # 2 segments: upper = 55.0 ft, bottom = 45.0 ft (approx without laps, or with slip)
    # Case A: upper has t = 0.75 (index 4 in G), length = 55.0 ft
    d_violating = Design(tip=15.0, base=45.0, uppers=(55.0,), ts=(4, 2))  # top=0.75", bot=0.5"
    
    # Without constraint -> should pass
    c_unconstrained = OptConstraints(tip_min=10.0, tip_max=20.0, base_min=40.0, base_max=50.0)
    spec, why = make_spec(d_violating, H, emb, c_unconstrained, G)
    assert spec is not None, f"Expected pass without constraint, got: {why}"

    # With constraint 0.75" <= 50.0 ft -> should FAIL because top tube is 55.0 ft @ 0.75"
    c_constrained = OptConstraints(
        tip_min=10.0, tip_max=20.0, base_min=40.0, base_max=50.0,
        max_len_by_thick={0.75: 50.0}, thick_len_mode='gte'
    )
    spec, why = make_spec(d_violating, H, emb, c_constrained, G, check_thick_len=True)
    assert spec is None, "Expected failure due to 0.75\" tube exceeding 50 ft"
    assert "length 55.00 ft > max 50 ft for thickness 0.75\"" in why, f"Unexpected reason: {why}"

    # But with check_thick_len=False (as used during lower_bound dummy sizing) -> should pass
    spec_dummy, _ = make_spec(d_violating, H, emb, c_constrained, G, check_thick_len=False)
    assert spec_dummy is not None, "check_thick_len=False should bypass thickness-length validation"

    # Case B: compliant design: upper tube 48.0 ft @ 0.75"
    d_compliant = Design(tip=15.0, base=45.0, uppers=(48.0,), ts=(4, 2))
    spec_ok, why_ok = make_spec(d_compliant, H, emb, c_constrained, G, check_thick_len=True)
    assert spec_ok is not None, f"Expected compliant design to pass, got: {why_ok}"

    print("PASS: test_make_spec_thickness_length_validation")


def test_parse_thick_len():
    # Test the parsing logic from app.py
    def _parse_thick_len(txt):
        out = {}
        if not txt or not str(txt).strip():
            return out, None
        for part in str(txt).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            delim = ':' if ':' in part else ('=' if '=' in part else None)
            if not delim:
                return {}, f"Invalid rule format '{part}'. Use 'thickness: max_length' (e.g. '0.75: 50')."
            lhs, rhs = part.split(delim, 1)
            try:
                t = float(lhs.strip().replace('"', '').replace("in", ""))
                ml = float(rhs.strip().replace("'", "").replace("ft", ""))
                if t <= 0 or ml <= 0:
                    return {}, f"Thickness and length must be positive numbers in '{part}'."
                out[round(t, 4)] = round(ml, 2)
            except ValueError:
                return {}, f"Could not parse numeric values in '{part}'. Use format '0.75: 50'."
        return out, None

    # Clean colon format
    res, err = _parse_thick_len("0.75: 50, 0.875: 40")
    assert err is None
    assert res == {0.75: 50.0, 0.875: 40.0}

    # Units / quotes included
    res2, err2 = _parse_thick_len('0.75": 50ft, 0.875 in = 40 ft')
    assert err2 is None
    assert res2 == {0.75: 50.0, 0.875: 40.0}

    # Empty
    res3, err3 = _parse_thick_len("")
    assert err3 is None and res3 == {}

    # Invalid string
    res4, err4 = _parse_thick_len("0.75-50")
    assert err4 is not None

    print("PASS: test_parse_thick_len")


if __name__ == '__main__':
    test_max_length_for_thickness()
    test_make_spec_thickness_length_validation()
    test_parse_thick_len()
    print("ALL TESTS PASSED!")
