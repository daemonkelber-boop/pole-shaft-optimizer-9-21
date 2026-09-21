import glob, math, numpy as np
from geometry import PoleSpec, Segment
from loads import Baseline, build_load_model
from pls_pole_xml_parser import parse_pls_pole_xml, get_field as g, get_single_table, get_load_case_instances
for F in sorted(glob.glob('/mnt/user-data/uploads/0*.xml')):
    p=parse_pls_pole_xml(F); name=F.split('/')[-1][:30]
    try:
        base=Baseline.from_xml(F)
        tubes=get_single_table(p,'steel_tubes_properties'); prop=get_single_table(p,'steel_pole_properties')[0]
        segs=[Segment(g(t,'length'),g(t,'thickness'),joint_type='slip' if g(t,'lap_length')>0 else 'flange') for t in tubes]
        spec=PoleSpec(tip_diameter=g(prop,'tip_diameter'),taper=g(tubes[0],'calculated_taper'),segments=segs,embedment=base.embedment)
        H=spec.total_length
        us=get_load_case_instances(p,'detailed_steel_pole_usages')
        su=get_single_table(p,'summary_of_steel_pole_usages')[0]; case=su['load_case']
        m=build_load_model(spec,base,case,ds=0.25)
        rows=sorted(us[case],key=lambda r:g(r,'rel_dist'))
        sp=np.array([g(r,'rel_dist') for r in rows]); yp=np.array([g(r,'trans_defl') for r in rows])/12; xp=np.array([g(r,'long_defl') for r in rows])/12; vp=np.array([g(r,'vert_defl') for r in rows])/12
        def pos(s):
            e=0.25; y=np.interp(s,sp,yp); x=np.interp(s,sp,xp); z=m.z_of(s)+np.interp(s,sp,vp)
            th=math.atan2(np.interp(max(s-e,0),sp,yp)-np.interp(min(s+e,H),sp,yp),2*e); return x,y,z,th
        L=[]
        for e in m.elements:
            x,y,z,_=pos(e.s_mid); L.append((x,y,z,e.fx,e.fy,e.fz,e.s_mid))
        for pl in m.points:
            x,y,z,th=pos(pl.s); L.append((x+pl.dx,y+pl.dy*math.cos(th)+pl.dz*math.sin(th),z-pl.dy*math.sin(th)+pl.dz*math.cos(th),pl.Fx,pl.Fy,pl.Fz,pl.s))
        diffs=[]; big=[]
        Mmax=max(abs(g(r,'trans_mom_local_mx')) for r in rows)
        for r in rows:
            s0=g(r,'rel_dist')
            if r['joint_position']=='Origin' and s0>0: continue
            if s0>spec.groundline_rel+1e-6: continue
            x0,y0,z0,_=pos(s0)
            Mx=sum(Fy*(z-z0)+Fz*(y-y0) for x,y,z,Fx,Fy,Fz,s in L if s<s0-1e-6)
            M=g(r,'trans_mom_local_mx')
            if abs(M)>0.2*Mmax: diffs.append((Mx/M-1)*100)
        d=np.array(diffs)
        print(f"{name:30s} emb={base.embedment:5.1f} pts={len(L)-len(m.elements):2d} gov: {case[:28]:28s} M>20%max: n={len(d):2d} mean {d.mean():+.2f}% max|d| {np.abs(d).max():.2f}%")
    except Exception as ex:
        import traceback; print(name,'ERROR',repr(ex)); traceback.print_exc(limit=2)
