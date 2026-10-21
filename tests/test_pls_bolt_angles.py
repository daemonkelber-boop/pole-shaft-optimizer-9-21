"""PLS-POLE Base Plate dialog export: quadrant bolt angles, plate Fy
rule, and the copy-paste angle text. Run: python -m pytest tests/"""
import pytest

from baseplate import plate_fy, pls_angles_text, pls_bolt_angles


def test_60_on_axis():
    a = pls_bolt_angles(60, "on_axis")
    assert len(a) == 16
    assert a[0] == 0
    assert a[-1] == 90
    assert all(abs((b - c) - 6.0) < 1e-9 for b, c in zip(a[1:], a))


def test_56_on_axis():
    a = pls_bolt_angles(56, "on_axis")
    assert len(a) == 15
    assert a[0] == 0
    assert a[-1] == 90
    step = 360.0 / 56
    assert all(abs((b - c) - step) <= 1e-4 for b, c in zip(a[1:], a))


def test_60_straddle():
    a = pls_bolt_angles(60, "straddle")
    assert len(a) == 15
    assert a[0] == 3
    assert a[-1] == 87


def _unique_positions(angles):
    """Mirror quadrant angles over both axes, dedupe within 1e-6 (mod 360)."""
    pos = sorted(m % 360.0 for a in angles
                 for m in (a, 180.0 - a, 180.0 + a, -a))
    uniq = []
    for x in pos:
        if not uniq or x - uniq[-1] > 1e-6:
            uniq.append(x)
    if len(uniq) > 1 and uniq[0] + 360.0 - uniq[-1] <= 1e-6:
        uniq.pop()
    return uniq


@pytest.mark.parametrize("n", range(8, 81, 4))
@pytest.mark.parametrize("orientation", ["on_axis", "straddle"])
def test_mirroring_reproduces_n_bolts(n, orientation):
    assert len(_unique_positions(pls_bolt_angles(n, orientation))) == n


def test_58_raises():
    with pytest.raises(ValueError):
        pls_bolt_angles(58)


def test_plate_fy_rule():
    assert plate_fy(4.0) == 50
    assert plate_fy(4.25) == 42
    assert plate_fy(5.0) == 42


def test_copy_text_60_on_axis():
    assert pls_angles_text(pls_bolt_angles(60, "on_axis")) == (
        "0\n6\n12\n18\n24\n30\n36\n42\n48\n54\n60\n66\n72\n78\n84\n90")
