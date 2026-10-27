"""Reusable CMOS standard-cell views for the frozen TIME_CONTROL source.

The 1.6 um internal row height accommodates 500/250 nm register devices and
540/90 nm NORs without changing their source sizes. Large drivers use balanced
5 nm finger partitions, including unequal last fingers when necessary.
"""
from collections import defaultdict
from math import ceil
from .components import CellView,Component
from .lib import Builder,pya

HEIGHT=1600.
N_BOTTOM=245.
P_TOP=1355.
WELL_SPLIT=680.


def partition(width,count):
    units=round(width/5)
    if abs(units*5-width)>1e-6:raise ValueError('Width must be on the 5 nm device grid')
    q,r=divmod(units,count)
    result=[5.*(q+(i<r)) for i in range(count)]
    if min(result)<90:raise ValueError('Refingering would create subminimum active width')
    return result


def make_logic(layout: pya.Layout,component: Component,row_height: float=1500.,compact: bool=False) -> CellView:
    height=row_height
    n_bottom=255.
    p_top=height-245.
    well_split=640. if height==1500 else 680.
    if component.children or component.passives:raise ValueError('Standard gate must be a MOS leaf')
    ps=[d for d in component.devices if d.model=='PMOS_VTG']
    ns=[d for d in component.devices if d.model=='NMOS_VTG']
    if len(ps)!=len(ns) or len(ps) not in (1,2,3):raise ValueError('Unsupported standard logic')
    for d in component.devices:
        if abs(d.length_m*1e9-50)>1e-6 or d.nodes[3]!=('VDD' if d in ps else 'VSS'):
            raise ValueError('Unsupported channel length or body topology')
    nor=all(set((d.nodes[0],d.nodes[2]))=={'Z','VSS'} for d in ns) and len(ps)>1
    serial=ps if nor else ns;parallel=ns if nor else ps
    supply='VDD' if nor else 'VSS';other_supply='VSS' if nor else 'VDD'
    if any(set((d.nodes[0],d.nodes[2]))!={'Z',other_supply} for d in parallel):raise ValueError('Not CMOS NAND/NOR/INV')
    widths=[{round(d.width_m*1e9,6) for d in ds} for ds in (ps,ns)]
    if any(len(w)!=1 for w in widths):raise ValueError('Uniform width per channel polarity required')
    wp,wn=[w.pop() for w in widths];nf=max(1,ceil(wp/550),ceil(wn/(330 if height==1500 else 380)))
    # Decoder-height rows (1365 nm) keep gate pads clear of the first S/D straps.
    finger=475. if height<1500 else float('inf')
    while (max(partition(wp,nf))+max(partition(wn,nf))>height-695
           or max(partition(wp,nf))>finger or max(partition(wn,nf))>finger):nf+=1
    pw,nw=partition(wp,nf),partition(wn,nf)
    if nor and nf!=1:raise ValueError('Folded NOR series routing not planned')
    node=supply;pending=list(serial);chain=[]
    while pending:
        matches=[d for d in pending if node in (d.nodes[0],d.nodes[2])]
        if len(matches)!=1:raise ValueError('Unsupported series transistor graph')
        d=matches[0];nxt=d.nodes[2] if d.nodes[0]==node else d.nodes[0]
        chain.append((d,node,nxt));pending.remove(d);node=nxt
    if node!='Z':raise ValueError('Series network does not end at Z')
    bygate={d.nodes[1]:d for d in parallel}
    if set(bygate)!={d.nodes[1] for d in serial}:raise ValueError('Complementary gates disagree')
    specs=[];sn=[supply]
    for f in range(nf):
        steps=chain if f%2==0 else [(d,z,a) for d,a,z in reversed(chain)]
        for d,a,z in steps:
            pd,nd=(d,bygate[d.nodes[1]]) if nor else (bygate[d.nodes[1]],d)
            specs.append((pd,nd,f,pw[f],nw[f]));sn.append(z)
    pn=[other_supply if i%2==0 else 'Z' for i in range(len(specs)+1)]
    pnodes,nnodes=(sn,pn) if nor else (pn,sn)
    b=Builder(layout,'CTRL_STD_'+component.identity[:16]+'_H'+str(int(height))+('_C' if compact else ''));count=len(specs)
    # Keep 55 nm well enclosure at both diffusion ends and 152.5 nm
    # between the last gate access and the output via2. Gate pitch and
    # finger partitions stay unchanged; the centre frame retains its gap.
    first=92.5 if compact else 100.
    xs=[first+190*i for i in range(count+1)];out=xs[-1]+(57.5 if compact else 105.);width=out+35
    gy=(n_bottom+max(nw)+p_top-max(pw))/2
    # Decoder-height rows split the wells mid-gap so each diffusion keeps
    # its own well (the fixed 680 nm split would cut a 470 nm P strip).
    if height<1500:well_split=round(gy/5)*5
    # Two opposite contact enclosures on a 65 nm via fit narrow channels.
    def wire(layer,x,y,xx,yy,w=70):b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    def stack(x,y,upper=2,gate=False):
        b.sq('v1',x,y,65)
        for layer in ('m1','m2'):
            if gate:b.rect(layer,x-35,y-67.5,x+35,y+67.5)
            else:b.rect(layer,x-67.5,y-35,x+67.5,y+35)
        if upper==3:
            b.sq('v2',x,y,65)
            if gate:b.rect('m3',x-35,y-67.5,x+35,y+67.5)
            else:b.rect('m3',x-67.5,y-35,x+67.5,y+35)
    b.rect('pwell',0,0,width,well_split);b.rect('nwell',0,well_split,width,height);b.rect('vtg',0,0,width,height)
    bodies={}
    for n,y,implant in [('VSS',100.,'pplus'),('VDD',height-100.,'nplus')]:
        wire('m1',35,y,width-35,y);b.rect('active',55,y-45,145,y+45)
        b.rect(implant,40,y-60,160,y+60);b.cont(100,y);bodies[n]=(100,y,'m1')
    row_sites=[];output_ys=[];internals=defaultdict(list)
    for polarity,nets,wi in [('p',pnodes,3),('n',nnodes,4)]:
        widths=[v[wi] for v in specs];sites=[]
        lo=min(widths);cy=p_top-lo/2 if polarity=='p' else n_bottom+lo/2
        # Folded NAND internal straps have M2 landings at y=235. Keep the
        # output rail and its via1 pads at least 70 nm above those landings.
        rail_y=max(cy,375.) if polarity=='n' and not nor and nf>1 and len(ps)>1 else cy
        for i,w in enumerate(widths):
            left=xs[i]-(37.5 if i==0 else 0);right=xs[i+1]+(37.5 if i==count-1 else 0)
            low,high=(p_top-w,p_top) if polarity=='p' else (n_bottom,n_bottom+w)
            b.rect('active',left,low,right,high)
            b.rect('pplus' if polarity=='p' else 'nplus',left-40,low-40,right+40,high+40)
        for i,net in enumerate(nets):
            x=xs[i];b.cont(x,cy);b.rect('m1',x-35,cy-67.5,x+35,cy+67.5);sites.append((x,cy,'m1'))
            if net in bodies:wire('m1',x,cy,x,bodies[net][1])
            elif net=='Z':wire('m1',x,cy,x,rail_y);stack(x,rail_y)
            else:internals[net].append((x,cy))
        zs=[xs[i] for i,n in enumerate(nets) if n=='Z'];wire('m2',min(zs),rail_y,out,rail_y)
        row_sites.append(sites);output_ys.append(rail_y)
    p_sites,n_sites=row_sites
    wire('m2',out,output_ys[1],out,output_ys[0])
    for net,sites in internals.items():
        if len(sites)==1:continue
        if nor:raise ValueError('Folded NOR routing not supported')
        for x,y in sites:wire('m1',x,y,x,235);stack(x,235,3)
        wire('m3',min(x for x,y in sites),235,max(x for x,y in sites),235)
    inputs=defaultdict(list);terminals={};physical=[]
    for i,(pd,nd,f,pw_i,nw_i) in enumerate(specs):
        x=first+95+190*i;b.rect('poly',x-25,n_bottom-50,x+25,p_top+50);inputs[nd.nodes[1]].append(x)
        for d,w,sites,nodes in [(pd,pw_i,p_sites,pnodes),(nd,nw_i,n_sites,nnodes)]:
            a,z=sites[i],sites[i+1]
            drain,source=(a,z) if d.nodes[0]==nodes[i] else (z,a)
            terminals[f'{d.name}@{f}:d']=drain;terminals[f'{d.name}@{f}:s']=source
            terminals[f'{d.name}@{f}:g']=(x,gy,'poly');terminals[f'{d.name}@{f}:b']=bodies[d.nodes[3]]
            physical.append(dict(device=d.name,finger=f,width_nm=w,length_nm=50.,model=d.model,nodes=d.nodes,gate_xy_nm=(x,gy)))
    pins={}
    for k,(net,points) in enumerate(inputs.items()):
        runs=[]
        for x in points:
            if runs and x-runs[-1][-1]==190:runs[-1].append(x)
            else:runs.append([x])
        access=[]
        for run in runs:
            x=run[0];wire('poly',x,gy,run[-1],gy,50);b.sq('poly',x,gy,75);b.cont(x,gy)
            stack(x,gy,3,gate=True);access.append(x)
        if len(access)>1:
            # 1365 nm decoder rows step the bars by 400 nm to leave the output access free.
            yy=420.+(600 if height>=1500 else 400)*k
            for x in access:
                wire('m3',x,gy,x,yy);b.via(3,x,yy);b.rect('m3',x-35,yy-70,x+35,yy+70);b.sq('m4',x,yy,140)
            wire('m4',min(access),yy,max(access),yy,140);pins[net]=(access[0],yy,'m4')
        else:pins[net]=(access[0],gy,'m3')
    b.via(2,out,gy);b.rect('m2',out-35,gy-70,out+35,gy+70);b.rect('m3',out-35,gy-70,out+35,gy+70)
    pins.update(Z=(out,gy,'m3'),VDD=bodies['VDD'],VSS=bodies['VSS'])
    for net,(x,y,l) in pins.items():b.label(l,net,x,y)
    if set(pins)!=set(component.ports):raise ValueError('Standard-cell pin mismatch')
    v=CellView(component,b,width,height,pins,terminals,[],[],[]);v.physical_fingers=physical
    v.standard_family='NOR'+str(len(ps)) if nor else 'INV' if len(ps)==1 else 'NAND'+str(len(ps))
    v.well_split=well_split
    return v


def make_transmission_gate(layout,component):
    if len(component.devices)!=2:raise ValueError('Transmission gate requires two MOS')
    expected={'PMOS_VTG':(('OUT','CTR_P','IN','VDD'),500.),'NMOS_VTG':(('IN','CTR_N','OUT','VSS'),250.)}
    for d in component.devices:
        nodes,w=expected[d.model]
        if d.nodes!=nodes or abs(d.width_m*1e9-w)>1e-6 or abs(d.length_m*1e9-50)>1e-6:
            raise ValueError('Transmission-gate source mismatch')
    b=Builder(layout,'CTRL_TG_'+component.identity[:16]);width=520.
    b.rect('pwell',0,0,width,680);b.rect('nwell',0,680,width,HEIGHT);b.rect('vtg',0,0,width,HEIGHT)
    for net,y,imp in [('VSS',100.,'pplus'),('VDD',1500.,'nplus')]:
        b.rect('m1',0,y-35,width,y+35);b.rect('active',55,y-45,145,y+45)
        b.rect(imp,40,y-60,160,y+60);b.cont(100,y)
    row_sites={}
    for kind,lo,hi,gate_y in [('n',245.,495.,600.),('p',855.,1355.,755.)]:
        b.rect('active',152.5,lo,417.5,hi);b.rect('nplus' if kind=='n' else 'pplus',112.5,lo-40,457.5,hi+40)
        b.rect('poly',260,195 if kind=='n' else 717.5,310,637.5 if kind=='n' else 1405)
        b.sq('poly',285,gate_y,75);b.cont(285,gate_y)
        for layer in ('m1','m2','m3'):b.rect(layer,217.5,gate_y-35,352.5,gate_y+35)
        for via in (1,2):b.sq('v'+str(via),285,gate_y,65)
        cy=(lo+hi)/2;row_sites[kind]=[(190.,cy,'m1'),(380.,cy,'m1')]
        for x,rail in [(190.,85.),(380.,485.)]:
            b.cont(x,cy);b.rect('m1',x-35,cy-67.5,x+35,cy+67.5);b.sq('v1',x,cy,65)
            b.rect('m2',x-35,cy-67.5,x+35,cy+67.5)
            b.rect('m2',min(x,rail)-35,cy-67.5,max(x,rail)+35,cy+67.5)
    for x in (85.,485.):
        b.rect('m2',x-35,335,x+35,1140);b.via(2,x,800)
        b.rect('m2',x-35,730,x+35,870);b.rect('m3',x-35,730,x+35,870)
    pins=dict(VDD=(100,1500,'m1'),VSS=(100,100,'m1'),IN=(85,800,'m3'),OUT=(485,800,'m3'),CTR_N=(285,600,'m3'),CTR_P=(285,755,'m3'))
    terminals={}
    for d in component.devices:
        row=row_sites['p' if d.model=='PMOS_VTG' else 'n'];nodes={'IN':row[0],'OUT':row[1],**pins}
        # Pin accesses are on the same actual source/drain conductor.
        nodes.update(IN=row[0],OUT=row[1])
        for t,n in zip('dgsb',d.nodes):terminals[f'{d.name}@0:{t}']=nodes[n]
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(component,b,width,HEIGHT,pins,terminals,[],[],[]);v.standard_family='TG'
    return v
