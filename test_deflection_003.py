import numpy as np, time
from geometry import PoleSpec, Segment
from loads import Baseline, build_load_model
from deflection import solve_deflection, deflection_check
from pls_pole_xml_parser import parse_pls_pole_xml, get_field as g, get_single_table, get_load_case_instances
import sys
XML='003.xml'; mode = sys.argv[1] if len(sys.argv)>1 else 'outer'
SPEC=PoleSpec(label='003',tip_diameter=13.0,taper=0.32955,embedment=0.0,
   segments=[Segment(57.0,0.3125,joint_type='slip'),Segment(57.5,0.375,joint_type='flange')])
base=Baseline.from_xml(XML); p=parse_pls_pole_xml(XML)
tips={r['load_case']:r for r in get_single_table(p,'summary_of_tip_deflections_for_all_load_cases')}
us=get_load_case_instances(p,'detailed_steel_pole_usages')
print(f"lap_stiffness={mode}")
print(f"{'case':42s} {'tipY':>7} {'PLS':>7} {'d%':>6} | {'tipZ':>6} {'PLS':>6} | {'Mbase':>7} {'PLS':>7} {'d%':>6} | it")
t0=time.time(); allM=[]
for case in base.load_cases:
    m=build_load_model(SPEC,base,case,ds=0.25)
    r=solve_deflection(m,lap_stiffness=mode)
    T=tips[case if case in tips else [k for k in tips if k.startswith(case[:20])][0]]
    py=g(T,'tran_defl'); pz=g(T,'vert_defl')
    rows=us[case]; Rb=max(rows,key=lambda q:g(q,'rel_dist'))
    Mb=r.Mx[-1]; Mp=g(Rb,'trans_mom_local_mx')
    for q in rows:
        Mq=g(q,'trans_mom_local_mx')
        if abs(Mq)>50 and not (q['joint_position']=='Origin' and g(q,'rel_dist')>0):
            allM.append((r.at(g(q,'rel_dist'))['Mx']/Mq-1)*100)
    dy=(r.tip_trans_in/py-1)*100 if abs(py)>1 else float('nan')
    print(f"{case[:42]:42s} {r.tip_trans_in:7.2f} {py:7.2f} {dy:+6.2f} | {r.tip_vert_in:6.2f} {pz:6.2f} | {Mb:7.1f} {Mp:7.1f} {(Mb/Mp-1)*100 if abs(Mp)>1 else 0:+6.2f} | {r.iterations}")
a=np.array(allM); print(f"section Mx |M|>50: n={len(a)} mean {a.mean():+.2f}% max|d| {abs(a).max():.2f}%   time {time.time()-t0:.1f}s")
