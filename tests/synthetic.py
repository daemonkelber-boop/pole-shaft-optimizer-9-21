"""Synthetic loading for optimizer tests (no PLS-POLE XML is committed)."""
from geometry import PoleSpec, Segment
from loads import AttachPoint, Baseline, LoadCase


def synthetic_baseline(H=160.5, emb=0.0) -> Baseline:
    lc = lambda n, dlf, qt, ql, ice: LoadCase(n, dlf, qt, ql, ice, 57.0 if ice else 0.0,
                                              1.0, 'No Limit', 0.0)
    cases = {c.name: c for c in [lc('NESC 250B', 1.5, 4.0, 0.0, 0.5),
                                 lc('Extreme wind', 1.1, 25.0, 0.0, 0.0),
                                 lc('Skewed wind', 1.0, 15.0, 10.0, 0.0),
                                 lc('Broken wire', 1.2, 0.0, 0.0, 0.0)]}
    attach = {
        'OHGW': AttachPoint('OHGW', 0.0, 0.0, 0.5, 0.0),
        'A': AttachPoint('A', 12.0, 6.0, -1.0, 0.0),
        'B': AttachPoint('B', 24.0, 7.5, -1.0, 180.0),
        'C': AttachPoint('C', 36.0, 6.0, -1.0, 90.0),
        'V1': AttachPoint('V1', 60.25, 1.0, 0.0, 270.0),
        'OFF': AttachPoint('OFF', H + 3.0, 0.0, 0.0, 0.0),     # below the shaft
    }
    loads = {
        'NESC 250B': [('OHGW', 0.2, 1.1, 0.9), ('A', 0.3, 3.0, 5.0),
                      ('B', 0.3, 3.0, 5.0), ('C', 0.3, 3.0, 5.0), ('V1', 0.0, 0.0, 2.0)],
        'Extreme wind': [('OHGW', 0.0, 2.0, 0.4), ('A', 0.0, 5.0, 2.0),
                         ('B', 0.0, 5.0, 2.0), ('C', 0.0, 5.0, 2.0), ('OFF', 0.0, 1.0, 0.0)],
        'Skewed wind': [('A', 2.0, 3.0, 2.0), ('C', -1.0, 4.0, 2.5)],
        'Broken wire': [('A', 8.0, 0.0, 3.0), ('B', 0.0, 0.0, 3.0)],
    }
    dav = [('DA1', AttachPoint('davit:DA1', 12.0, 3.0, -0.5, 0.0), 0.15)]
    return Baseline('synthetic', H, emb, 1.1, cases, attach, loads, dav)


def synthetic_pole(H=160.5):
    """Baseline with a deflection-limited case and a 3-tube seed geometry
    of total length H (160.5 ft)."""
    base = synthetic_baseline(H=H)
    base.load_cases['Deflection'] = LoadCase('Deflection', 1.0, 9.0, 0.0, 0.0, 0.0, 1.0,
                                             '% Pole Height', 3.0)
    base.attach_loads['Deflection'] = [('OHGW', 0.0, 0.6, 0.4), ('A', 0.0, 1.5, 2.0),
                                       ('B', 0.0, 1.5, 2.0), ('C', 0.0, 1.5, 2.0)]
    spec = PoleSpec(tip_diameter=16.0, taper=0.2, embedment=0.0,
                    segments=[Segment(55.0, 0.3125), Segment(57.0, 0.375),
                              Segment(H - 103.25, 0.5, joint_type='none')])
    return base, spec
