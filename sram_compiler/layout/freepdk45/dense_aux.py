"""Compact source-owned two-gate chains and output latch."""
from .cell_flow import leaves
from .components import CellView
from .dense_logic import make_logic,HEIGHT
from .lib import Builder


def make_aux(layout,component,area_ceiling=True):
    b=Builder(layout,'DENSE_AUX_'+component.identity[:12]);cache={};placed={};terminals={};fingers=[]
    x=0.
    for path,c,m in leaves(component):
        if c.identity not in cache:cache[c.identity]=make_logic(layout,c,area_ceiling)
        v=cache[c.identity];b.place(v.builder,x,0);placed[path]=(v,x,m)
        terminals.update({path+'/'+n:(x+px,py,l) for n,(px,py,l) in v.device_pins.items()})
        fingers.extend(dict(f,device=path+'/'+f['device'],gate_xy_nm=(x+f['gate_xy_nm'][0],f['gate_xy_nm'][1])) for f in v.physical_fingers)
        x+=v.width_nm
    width=x;pins={};points={}
    for path,(v,x,m) in placed.items():
        for p,(px,py,l) in v.pins.items():points.setdefault(m[p],[]).append((x+px,py,l))
    if len(placed)==2:
        a,z=list(placed.values());va,xa,ma=a;vz,xz,mz=z
        if ma['Z']!=mz['A']:raise ValueError('Not a two-gate chain')
        xx=xa+va.output_rail_m2[0];tx,ty,_=vz.pins['A'];tx+=xz;zy=va.pins['Z'][1]
        if 70<=abs(ty-zy)<155:
            # A second via2 would violate spacing to the output via2.
            b.rect('m3',xx-35,min(ty,zy)-35,xx+35,max(ty,zy)+35)
        else:
            b.via(2,xx,ty);b.rect('m2',xx-35,ty-70,xx+35,ty+70)
            b.rect('m3',xx-35,ty-70,xx+35,ty+70)
        b.rect('m3',xx-35,ty-35,tx+35,ty+35)
    elif component.name=='D_LATCH':
        tracks={'D':200.,'EN':500.,'DB':800.,'INT1':200.,'INT2':1100.,'Q':500.,'QB':800.}
        for net,yy in tracks.items():
            sites=points[net]
            for xx,py,l in sites:
                if l!='m3':raise ValueError('Latch access requires M3')
                b.rect('m3',xx-35,min(py,yy)-35,xx+35,max(py,yy)+35)
                b.rect('m3',xx-35,yy-70,xx+35,yy+70);b.via(3,xx,yy);b.sq('m4',xx,yy,140)
            b.rect('m4',min(p[0] for p in sites)-70,yy-70,max(p[0] for p in sites)+70,yy+70)
            if net in component.ports:pins[net]=(sites[0][0],yy,'m4')
    else:raise ValueError('No dense physical plan for auxiliary component')
    for p in component.ports:
        if p not in pins:pins[p]=points[p][0]
    for p,(x,y,l) in pins.items():b.label(l,p,x,y)
    v=CellView(component,b,width,HEIGHT,pins,terminals,[],[],[]);v.physical_fingers=fingers
    return v
