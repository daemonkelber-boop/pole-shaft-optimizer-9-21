"""Optimizer._eval must return the original failure reason for a cached
failure, so size() keeps thickening instead of treating it as a geometry
failure ('cached fail') and giving up."""
from optimizer import Design, OptConstraints, Optimizer
from synthetic import synthetic_pole


def _opt():
    base, spec = synthetic_pole()
    return Optimizer(base, spec, OptConstraints())


def test_cached_failure_keeps_reason():
    opt = _opt()
    for ts in [(0, 0, 0), (1, 1, 2)]:            # w/t failure, strength failure
        d = Design(16.0, 46.0, (55.0, 57.0), ts)
        e, why = opt._eval(d)
        assert e is None and why
        n = opt.n_evals
        assert opt._eval(d) == (None, why)
        assert opt.n_evals == n                  # served from the cache
    assert any(w.startswith('strength:') for w in opt._fail_why.values())


def test_size_continues_after_cached_failure():
    d = Design(16.0, 46.0, (55.0, 57.0), (0, 0, 0))
    fresh = _opt().size(d)
    assert fresh is not None
    opt = _opt()
    assert opt._eval(d)[0] is None               # failure now cached
    again = opt.size(d)
    assert again is not None and again.design.ts == fresh.design.ts
