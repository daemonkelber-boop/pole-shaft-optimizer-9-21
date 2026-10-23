"""FirstOrderEnvelope must reproduce build_load_model + first_order_forces,
and Optimizer._cand_env must return each geometry's OWN envelope (it used to
be cached under id(spec), and freed specs' ids are reused)."""
import math

import numpy as np
import pytest

from geometry import PoleSpec, Segment
from loads import FirstOrderEnvelope, build_load_model, first_order_forces
from synthetic import synthetic_baseline


SPECS = [
    PoleSpec(tip_diameter=14.0, taper=0.24, embedment=0.0,
             segments=[Segment(53.0, 0.25), Segment(55.0, 0.3125),
                       Segment(57.0, 0.375), Segment(7.0, 0.5, joint_type='none')]),
    PoleSpec(tip_diameter=18.5, taper=0.31, embedment=0.0,
             segments=[Segment(57.0, 0.3125, joint_type='flange'),
                       Segment(60.0, 0.4375, top_diameter_override=36.4),
                       Segment(43.5, 0.5, joint_type='none')]),
    PoleSpec(tip_diameter=11.0, taper=0.2, embedment=16.0,
             segments=[Segment(55.0, 0.1875), Segment(55.0, 0.25),
                       Segment(60.0, 0.3125, joint_type='none')]),
]


def reference(spec, base, cases, st, ds=2.0):
    env = np.zeros(len(st))
    for c in cases:
        m = build_load_model(spec, base, c, ds=ds)
        env = np.maximum(env, [math.hypot(f['Mx'], f['My'])
                               for f in (first_order_forces(m, s0) for s0 in st)])
    return env


@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('ds', [2.0, 0.5])
def test_matches_load_model(spec, ds):
    base = synthetic_baseline(emb=spec.embedment)
    cases = list(base.load_cases)
    st = np.concatenate([np.linspace(0, spec.groundline_rel, 60), [12.0, 36.0, 60.25]])
    got = FirstOrderEnvelope(base, cases, ds=ds)(spec, st)
    ref = reference(spec, base, cases, st, ds)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-9)
    # each case alone too (Mx / My signs are not hidden by the max)
    for c in cases:
        assert np.allclose(FirstOrderEnvelope(base, [c], ds=ds)(spec, st),
                           reference(spec, base, [c], st, ds), rtol=1e-10, atol=1e-9)


def test_cand_env_is_per_geometry():
    """Specs created and freed one after another (same id() in CPython)
    must each get their own envelope."""
    from optimizer import Optimizer
    base = synthetic_baseline()
    opt = Optimizer.__new__(Optimizer)
    opt.base = base
    opt.screen_cases = list(base.load_cases)
    opt._env_cache = {}
    opt._env_fn = {}
    st = np.linspace(0, 150, 40)
    for tip in (12.0, 16.0, 20.0, 24.0):
        spec = PoleSpec(tip_diameter=tip, taper=0.25, embedment=0.0,
                        segments=[Segment(55.0, 0.25), Segment(55.0, 0.3125),
                                  Segment(55.0, 0.375, joint_type='none')])
        got = opt._cand_env(spec, st)
        ref = np.interp(st, np.linspace(0, spec.groundline_rel, 60),
                        reference(spec, base, opt.screen_cases,
                                  np.linspace(0, spec.groundline_rel, 60)))
        assert np.allclose(got, ref, rtol=1e-10)
        del spec
