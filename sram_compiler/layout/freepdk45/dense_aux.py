"""Compact source-owned two-gate chains and output latch."""
from .cell_flow import leaves
from .components import CellView
from .dense_logic import make_logic,HEIGHT
from .lib import Builder

# Folded latch M4 tracks, cell-local. A supply bar crossing the shared rail
# (row boundary at HEIGHT) needs 140 nm beyond its 140 nm width on both sides,
# so lower-row tracks end below 1085 nm and upper-row tracks start above 1645.
FOLD_TRACKS={'D':200.,'EN':500.,'DB':800.,'QB':2*HEIGHT-800.,'Q':2*HEIGHT-500.}


def _folded_latch(layout,component,area_ceiling):
    """D_LATCH in two rows: the input gates below, the cross-coupled pair above,
    mirrored onto the shared VDD rail. The upper row shifts so each pair input
    sits over the lower gate driving it: INT1/INT2 are straight M3 lines."""
    b=Builder(layout,'DENSE_AUX_FOLD_'+component.identity[:12]);cache={};terminals={};fingers=[]
    items=list(leaves(component))
    for _,c,_ in items:
        if c.identity not in cache:cache[c.identity]=make_logic(layout,c,area_ceiling)
    pair=[next(i for i in items if i[2].get('Z')==q) for q in ('Q','QB')]
    drivers=[next(i for i in items if i[2].get('Z')==p[2]['A']) for p in pair]
    lower=drivers+[i for i in items if i not in pair and i not in drivers]
    placed={};x=0.
    for path,c,m in lower:placed[path]=(cache[c.identity],x,m,False);x+=cache[c.identity].width_nm
    width=x
    va,xa,_,_=placed[drivers[0][0]]
    x=xa+va.pins['Z'][0]-cache[pair[0][1].identity].pins['A'][0];shift=x
    for path,c,m in pair:placed[path]=(cache[c.identity],x,m,True);x+=cache[c.identity].width_nm
    width=max(width,x)
    top=2*HEIGHT
    def at(x,mirror,px,py):return (x+px,top-py if mirror else py)
    for path,(v,x,m,mirror) in placed.items():
        b.place(v.builder,x,top if mirror else 0.,my=mirror)
        terminals.update({path+'/'+n:(*at(x,mirror,px,py),l) for n,(px,py,l) in v.device_pins.items()})
        fingers.extend(dict(f,device=path+'/'+f['device'],gate_xy_nm=at(x,mirror,*f['gate_xy_nm'])) for f in v.physical_fingers)
    points={}
    for path,(v,x,m,mirror) in placed.items():
        for p,(px,py,l) in v.pins.items():points.setdefault(m[p],[]).append((*at(x,mirror,px,py),l))
    joins=[]
    for p,d in zip(pair,drivers):
        (zx,zy,_),=[pt for pt in points[d[2]['Z']] if pt[1]<HEIGHT]
        (ax,ay,_),=[pt for pt in points[d[2]['Z']] if pt[1]>HEIGHT]
        if abs(zx-ax)>1e-6:raise ValueError('Folded latch inputs are not aligned to their drivers')
        b.rect('m3',zx-35,zy-35,zx+35,ay+35);joins.append(zx)
    pins={}
    for net,yy in FOLD_TRACKS.items():
        sites=points[net]
        for xx,py,l in sites:
            if l!='m3':raise ValueError('Latch access requires M3')
            b.rect('m3',xx-35,min(py,yy)-35,xx+35,max(py,yy)+35)
            b.rect('m3',xx-35,yy-70,xx+35,yy+70);b.via(3,xx,yy);b.sq('m4',xx,yy,140)
        b.rect('m4',min(p[0] for p in sites)-70,yy-70,max(p[0] for p in sites)+70,yy+70)
        if net in component.ports:pins[net]=(sites[0][0],yy,'m4')
    # One VDD rail region across the row boundary; each row keeps its own VSS
    # rail. The VDD access sits midway between the two M3 joins.
    b.rect('m1',shift,HEIGHT-100,x,HEIGHT+100)
    pins['VDD']=(round(sum(joins)/2/2.5)*2.5,HEIGHT,'m1');pins['VSS']=(100.,100.,'m1')
    for p,(px,py,l) in pins.items():b.label(l,p,px,py)
    v=CellView(component,b,width,top,pins,terminals,[],[],[]);v.physical_fingers=fingers
    # Rail points the parent must join to its supplies (the pins included).
    v.supply_rails=[('VSS',100.,100.),('VSS',shift+100,top-100),('VDD',*pins['VDD'][:2])]
    v.fold_shift=shift
    return v


def make_aux(layout,component,area_ceiling=True,folded=False,compact=False):
    """compact=True builds a two-gate chain from the compact decoder's gates,
    so it can share a decoder gate row, its rails and wells."""
    if folded:
        if component.name!='D_LATCH':raise ValueError('Only the output latch has a folded plan')
        return _folded_latch(layout,component,area_ceiling)
    b=Builder(layout,('DENSE_AUX_C_' if compact else 'DENSE_AUX_')+component.identity[:12]);cache={};placed={};terminals={};fingers=[]
    x=0.
    for path,c,m in leaves(component):
        if c.identity not in cache:cache[c.identity]=make_logic(layout,c,area_ceiling,compact=compact)
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
