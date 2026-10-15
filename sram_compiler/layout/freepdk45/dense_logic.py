"""Compact source-owned CMOS gates on the OpenRAM 1365 nm row pitch.

Physical refingering preserves total W/L, models and every logical net. Shared
S/D strips, rail taps and local M2/M3 routes replace the generic maze frames.
"""
from collections import defaultdict
import hashlib
import json

from .components import CellView
from .lib import Builder, pya

HEIGHT=1365.
N_BOTTOM=245.
P_TOP=1120.
WELL_SPLIT=640.
VSS_Y=100.
VDD_Y=1265.


def _plan(component):
    if component.children or not component.devices:
        raise ValueError('Dense logic requires a direct-MOS leaf')
    pmos=[d for d in component.devices if d.model=='PMOS_VTG']
    nmos=[d for d in component.devices if d.model=='NMOS_VTG']
    if len(pmos)!=len(nmos) or len(pmos) not in (1,2,3) or len(pmos)+len(nmos)!=len(component.devices):
        raise ValueError('Dense logic supports INV/NAND2/NAND3 topology')
    if any(abs(d.length_m*1e9-50)>1e-6 for d in component.devices):
        raise ValueError('Dense gates require source L=50 nm')
    if any(d.nodes[3]!=('VDD' if d in pmos else 'VSS') for d in component.devices):
        raise ValueError('Unsupported body connection')
    wp={round(d.width_m*1e9,6) for d in pmos};wn={round(d.width_m*1e9,6) for d in nmos}
    if len(wp)!=1 or len(wn)!=1:raise ValueError('Uniform P and N widths required')
    wp,wn=wp.pop(),wn.pop()
    limit=int(min(wp,wn)//90)
    nf=next((n for n in range(1,limit+1) if 90<=wp/n<=425 and 90<=wn/n<=340
        and wp/n+wn/n<=700 and all(abs(w/n/5-round(w/n/5))<1e-6 for w in (wp,wn))),None)
    if nf is None:raise ValueError('No exact legal finger partition for the frozen widths')
    # Walk the series N network from ground toward Z.
    chain=[];node='VSS';remaining=list(nmos)
    while remaining:
        matches=[d for d in remaining if node in (d.nodes[0],d.nodes[2])]
        if len(matches)!=1:raise ValueError('Not a series-N CMOS gate')
        d=matches[0];other=d.nodes[2] if d.nodes[0]==node else d.nodes[0]
        chain.append((d,node,other));remaining.remove(d);node=other
    if node!='Z' or any(set((d.nodes[0],d.nodes[2]))!={'VDD','Z'} for d in pmos):
        raise ValueError('Not an inverter/NAND topology')
    by_gate={d.nodes[1]:d for d in pmos}
    if set(by_gate)!={d.nodes[1] for d in nmos}:raise ValueError('P/N input mismatch')
    if nf>1 and len(chain)>2:raise ValueError('Folded NAND3 needs additional internal routing')
    if nf>1 and len(chain)==2 and (nf>2 or wn/nf<250):
        # Such a folded NAND2 joins its series nodes above the N contacts
        # (make_logic), which needs P + N/2 <= 447.5 nm per finger.
        nf=next((n for n in range(nf,limit+1) if 90<=wn/n and wp/n+wn/n/2<=447.5
                 and all(abs(w/n/5-round(w/n/5))<1e-6 for w in (wp,wn))),None)
        if nf is None:raise ValueError('No folded NAND2 partition with a series-node track')
    gate_specs=[];n_contacts=['VSS'];n_fingers=defaultdict(int);p_fingers=defaultdict(int)
    for repeat in range(nf):
        steps=chain if repeat%2==0 else [(d,b,a) for d,a,b in reversed(chain)]
        for nd,a,b in steps:
            pd=by_gate[nd.nodes[1]]
            gate_specs.append((nd,n_fingers[nd.name],pd,p_fingers[pd.name],a,b))
            n_fingers[nd.name]+=1;p_fingers[pd.name]+=1;n_contacts.append(b)
    p_contacts=['VDD' if i%2==0 else 'Z' for i in range(len(gate_specs)+1)]
    return wp/nf,wn/nf,nf,gate_specs,n_contacts,p_contacts


def make_logic(layout,component,area_ceiling=True):
    wp,wn,nf,gates,ns,ps=_plan(component)
    ident=hashlib.sha256((component.identity+':dense-1365-v1').encode()).hexdigest()
    b=Builder(layout,'DENSE_'+ident[:16])
    count=len(gates);contacts=[100.+190*i for i in range(count+1)]
    gx=[195.+190*i for i in range(count)]
    out_x=contacts[-1]+105
    width=out_x+35
    # A NAND2 folded more than twice, or with narrow N fingers, joins its
    # series nodes above the N contact pads: the legacy track below them
    # would cross VSS straps or output pads.
    high=nf>1 and len(gates)==2*nf and (nf>2 or wn<250)
    nb=N_BOTTOM
    ny=nb+wn/2;py=P_TOP-wp/2;gy=(nb+wn+P_TOP-wp)/2
    if len(gates)==3:gy-=2.5
    track_y=235.
    if high:
        track_y=ny+170;gy=max(gy,ny+340)
        if gy+135>py-70 or gy+87.5>P_TOP-wp:raise ValueError('No dense series-node track for this NAND2 folding')
    paths=[];vias=[];terminals={};physical=[]
    def rect(layer,x0,y0,x1,y1):b.rect(layer,x0,y0,x1,y1)
    def wire(net,layer,x0,y0,x1,y1,w=70):
        h=w/2;rect(layer,min(x0,x1)-h,min(y0,y1)-h,max(x0,x1)+h,max(y0,y1)+h)
        paths.append((net,(x0,y0,layer),(x1,y1,layer)))
    def via(net,n,x,y,gate=False):
        b.via(n,x,y);vias.append((net,x,y,n))
        if n==1:
            if gate:rect('m1',x-35,y-70,x+35,y+70)
            else:rect('m1',x-70,y-35,x+70,y+35)
        else:
            rect('m3',x-35,y-70,x+35,y+70)
        rect('m2',x-35,y-70,x+35,y+70)
    rect('pwell',0,0,width,WELL_SPLIT);rect('nwell',0,WELL_SPLIT,width,HEIGHT)
    rect('vtg',0,0,width,HEIGHT)
    for layer,y0,y1 in [('nplus',nb-40,nb+wn+40),('pplus',P_TOP-wp-40,P_TOP+40)]:
        rect(layer,contacts[0]-77.5,y0,contacts[-1]+77.5,y1)
    rect('active',contacts[0]-37.5,nb,contacts[-1]+37.5,nb+wn)
    rect('active',contacts[0]-37.5,P_TOP-wp,contacts[-1]+37.5,P_TOP)
    for net,y in [('VSS',VSS_Y),('VDD',VDD_Y)]:wire(net,'m1',35,y,width-35,y)
    body={}
    for net,y,implant in [('VSS',VSS_Y,'pplus'),('VDD',VDD_Y,'nplus')]:
        tx=contacts[0]
        rect('active',tx-45,y-45,tx+45,y+45)
        rect(implant,tx-60,y-60,tx+60,y+60)
        b.cont(tx,y);rect('m1',tx-35,y-35,tx+35,y+35)
        body[net]=(tx,y,'m1')
    n_sites={};p_sites={};internal=defaultdict(list)
    for row,netlist,cy,sites in [('n',ns,ny,n_sites),('p',ps,py,p_sites)]:
        for i,net in enumerate(netlist):
            x=contacts[i];b.cont(x,cy)
            half_y=35 if net=='Z' else 70
            rect('m1',x-35,cy-half_y,x+35,cy+half_y)
            sites[i]=(x,cy,'m1')
            if net in ('VDD','VSS'):
                wire(net,'m1',x,cy,x,VDD_Y if net=='VDD' else VSS_Y)
            elif net=='Z':
                via(net,1,x,cy)
            else:internal[net].append((x,cy))
        zs=[contacts[i] for i,n in enumerate(netlist) if n=='Z']
        rect('m2',min(zs)-35,cy-70,out_x+35,cy+70)
        paths.append(('Z',(min(zs),cy,'m2'),(out_x,cy,'m2')))
    wire('Z','m2',out_x,ny,out_x,py)
    for k,(net,sites) in enumerate(internal.items()):
        if len(sites)==1:continue
        track=track_y-k*135
        if track<track_y:raise ValueError('Too many shared series nodes for dense routing')
        for x,y in sites:wire(net,'m1',x,y,x,track)
        wire(net,'m1',min(x for x,_ in sites),track,max(x for x,_ in sites),track)
    inputs=defaultdict(list)
    for i,(nd,nf_i,pd,pf_i,a,c) in enumerate(gates):
        x=gx[i];rect('poly',x-25,nb-50,x+25,P_TOP+50)
        inputs[nd.nodes[1]].append(x)
        for d,f,w,sd,left,right in [(nd,nf_i,wn,ns[i:i+2],n_sites[i],n_sites[i+1]),
                                   (pd,pf_i,wp,ps[i:i+2],p_sites[i],p_sites[i+1])]:
            drain,source=(left,right) if d.nodes[0]==sd[0] else (right,left)
            terminals[f'{d.name}@{f}:d']=drain;terminals[f'{d.name}@{f}:s']=source
            terminals[f'{d.name}@{f}:b']=body[d.nodes[3]]
            physical.append(dict(device=d.name,finger=f,width_nm=w,length_nm=50,model=d.model,
                                 nodes=d.nodes,gate_xy_nm=(x,gy)))
    pins={}
    # Connect adjacent repeated inputs in poly; outer repeated inputs use M3.
    for index,(net,xs) in enumerate(inputs.items()):
        runs=[]
        for x in xs:
            if runs and x-runs[-1][-1]==190:runs[-1].append(x)
            else:runs.append([x])
        access=[]
        for run in runs:
            x=run[0]
            if len(run)>1:rect('poly',x-25,gy-25,run[-1]+25,gy+25)
            b.sq('poly',x,gy,75);b.cont(x,gy)
            via(net,1,x,gy,gate=True);via(net,2,x,gy)
            access.append(x)
        # A second split input joins below the gate row, never on the first's M3.
        joined=sum(1 for p in pins.values() if p[1]!=gy)
        port_y=gy+((220 if not joined else -220) if len(access)>1 else 0)
        for x in access:wire(net,'m3',x,gy,x,port_y)
        if len(access)>1:wire(net,'m3',min(access),port_y,max(access),port_y)
        pins[net]=(access[0],port_y,'m3')
        for i,(nd,ni,pd,pi,_,_) in enumerate(gates):
            if nd.nodes[1]==net:
                terminals[f'{nd.name}@{ni}:g']=(gx[i],gy,'poly')
                terminals[f'{pd.name}@{pi}:g']=(gx[i],gy,'poly')
    for net,y in [('VDD',VDD_Y),('VSS',VSS_Y)]:
        pins[net]=(100,y,'m1')
    via('Z',2,out_x,gy);pins['Z']=(out_x,gy,'m3')
    for net,p in pins.items():b.label(p[2],net,p[0],p[1])
    rect('bnd',0,0,width,HEIGHT)
    if set(pins)!=set(component.ports):raise ValueError('Dense gate port mismatch')
    view=CellView(component,b,width,HEIGHT,pins,terminals,[],paths,vias)
    view.physical_fingers=physical
    view.output_rail_m2=(out_x,ny,py)
    from .openram_core import ASSETS
    reference=json.loads((ASSETS/'area_reference.json').read_text())['cells']
    inputs=len(component.devices)//2
    family='pnand'+str(inputs) if inputs>1 else ('pinv_0' if max(d.width_m for d in component.devices)>540e-9+1e-15 else 'pinv')
    limit=reference['openram_1rw_32x32_freepdk45_'+family]['area_um2']
    # The ceilings come from the 32x32 reference; larger driver classes have none.
    if area_ceiling and b.cell.dbbox().area()>limit+1e-9:raise ValueError('Logic cell exceeds the OpenRAM physical area ceiling')
    return view
