"""Source-owned decoder and wordline drivers on 1365 nm rows.

Each 3-to-8 slice has local M3 input buses. M2 joins its row gates;
M4/M5/M6 distribute address and predecode signals without stretching rows.
Decoder level l occupies its own 3.9 um column. A level-l slice sits at the
first row of the children it enables, so its eight output trunks stay within
that span and later slices reuse the same trunk tracks.
"""
from collections import Counter, defaultdict
import re
from .components import Component, Instance, CellView
from .cell_flow import leaves
from .dense_logic import make_logic, HEIGHT, _plan
from .lib import Builder


def make_decoder(layout, source, specs):
    rows,cols=source.rows,source.cols
    if rows<8 or rows%8 or rows>512:
        raise ValueError('OpenRAM-pitch decoder supports 8 to 512 rows in multiples of 8')
    # The OpenRAM area ceilings describe the 32x32 reference driver classes.
    ceiling=rows<=32 and cols<=32
    items={p:(c,source.instances[p]['pins']) for n,c in specs.items()
           for p,c,_ in leaves(c,path=n)}
    b=Builder(layout,f'DENSE_DECODER_{rows}')
    cache={};placed={};terminals={};fingers=[];paths=[];vias=[]
    def wire(layer,x,y,xx,yy,w=70):
        b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    def via(n,x,y):
        b.via(n,x,y)
        if n<3:
            b.rect(f'm{n}',x-35,y-70,x+35,y+70)
            b.rect(f'm{n+1}',x-35,y-70,x+35,y+70)
        elif n==3:
            b.rect('m3',x-35,y-70,x+35,y+70);b.sq('m4',x,y,140)
        else:
            b.sq(f'm{n}',x,y,145);b.sq(f'm{n+1}',x,y,145)
        vias.append((n,x,y))
    def bus_via(x,y):
        b.via(2,x,y);b.rect('m2',x-70,y-35,x+70,y+35);b.rect('m3',x-35,y-70,x+35,y+70)
    def put(path,x,row):
        c,m=items[path]
        if c.identity not in cache:cache[c.identity]=make_logic(layout,c,ceiling)
        v=cache[c.identity];y=row*HEIGHT;b.place(v.builder,x,y)
        placed[path]=(v,x,y,m)
        terminals.update({path+'/'+n:(px+x,py+y,l) for n,(px,py,l) in v.device_pins.items()})
        fingers.extend(dict(f,device=path+'/'+f['device'],gate_xy_nm=(f['gate_xy_nm'][0]+x,f['gate_xy_nm'][1]+y)) for f in v.physical_fingers)
        return v
    def point(path,p):
        v,x,y,m=placed[path];px,py,l=v.pins[p];return x+px,y+py,l
    def extend_input(path,p,y):
        x,old,_=point(path,p);wire('m3',x,old,x,y)
        b.via(2,x,y);b.rect('m2',x-70,y-35,x+70,y+35);b.rect('m3',x-35,y-70,x+35,y+70)
        return x,y
    def link(a,z):
        va,xa,ya,ma=placed[a];vb,xb,yb,mb=placed[z]
        assert ma['Z']==mb['A']
        px,py,_=point(z,'A');sx=xa+va.output_rail_m2[0]
        if not ya+va.output_rail_m2[1]<=py<=ya+va.output_rail_m2[2]:raise ValueError('Local output access outside rail')
        zy=ya+va.pins['Z'][1]
        if 70<=abs(py-zy)<155:
            # Reuse the output via2; a second one would violate via2 spacing.
            wire('m3',sx,zy,sx,py)
        else:via(2,sx,py)
        wire('m3',sx,py,px,py)
    def tier(root):
        level,index=re.fullmatch(r'.*/XDEC_(\d+)_(\d+)',root).groups()
        return int(level),int(index)
    roots=sorted({p.rsplit('/',1)[0] for p in items if '/XINV_A' in p},key=tier)
    levels=1+max(tier(r)[0] for r in roots)
    final=[r for r in roots if tier(r)[0]==levels-1]
    if final!=[r for r in roots if any(m['Z'].startswith('DEC_WL') for p,(c,m) in items.items() if p.startswith(r+'/'))]:
        raise ValueError('Only the last decoder level may drive wordlines')
    if [tier(r)[1] for r in final]!=list(range(rows//8)):raise ValueError('Final decoder slices must cover every row')
    group_width=3900.;base=300.
    def column_x(level):return base+level*group_width
    def first_row(level,index):return 8*8**(levels-1-level)*index
    main_x=column_x(levels-1)
    chain_start=1610.
    # Dense gate width is 240 nm plus one 190 nm pitch per physical gate.
    chain_width=sum(240+190*len(_plan(items[final[0]+suffix][0])[3]) for suffix in
                    ('/XAND0/XPNAND3','/XAND0/XPINV','/XAND_EN0/XPNAND3','/XAND_EN0/XPINV'))
    bus_defs=[];root_outputs={};driver_start=main_x+chain_start+chain_width
    driver_views=[]
    for p,(c,m) in items.items():
        if p.startswith('XWL_DRV_0/'):
            if c.identity not in cache:cache[c.identity]=make_logic(layout,c,ceiling)
            driver_views.append(cache[c.identity])
    driver_width=sum(v.width_nm for v in driver_views)
    driver_enable_track=driver_start+driver_views[0].width_nm+100
    width=driver_start+driver_width
    pins={};wordlines=[]
    for root in roots:
        level,index=tier(root)
        first=first_row(level,index);ox=column_x(level)
        # Three true input buses, their three complements, and one enable.
        invpaths=[root+f'/XINV_A{i}' for i in (1,2,3)]
        invmaps=[items[p][1] for p in invpaths]
        sample=items[root+'/XAND_EN0/XPNAND3'][1]
        nets=[m['A'] for m in invmaps]+[m['Z'] for m in invmaps]+[sample['B']]
        # Duplicate formal ties (e.g. root A0=VSS) share the same logical net.
        bx=[ox+535+i*150 for i in (0,2,4,1,3,5,6)]
        for i,net in enumerate(nets):wire('m3',bx[i],first*HEIGHT+80,bx[i],(first+8)*HEIGHT-100)
        bus_defs.append((first,ox,nets,bx))
        for i,p in enumerate(invpaths):
            v=put(p,ox,first+i);y=(first+i)*HEIGHT
            xx,yy=extend_input(p,'A',y+80);wire('m2',xx,yy,bx[i],yy);via(2,bx[i],yy)
            sx=ox+v.output_rail_m2[0];yy=y+985
            wire('m2',sx,yy,bx[i+3],yy);via(2,bx[i+3],yy)
        for k in range(8):
            row=first+k;y=row*HEIGHT;x=ox+chain_start
            chain=[root+f'/XAND{k}/XPNAND3',root+f'/XAND{k}/XPINV',
                   root+f'/XAND_EN{k}/XPNAND3',root+f'/XAND_EN{k}/XPINV']
            for p in chain:x+=put(p,x,row).width_nm
            for a,z in zip(chain,chain[1:]):link(a,z)
            m=items[chain[0]][1]
            for port,yy in [('C',400),('B',810),('A',1195)]:
                xx,yy=extend_input(chain[0],port,y+yy);bi=nets.index(m[port])
                wire('m2',bx[bi],yy,xx,yy)
                if bi==0 and k<3:
                    # Beside an input inverter's output rail: vertical landing.
                    b.via(2,bx[bi],yy);b.rect('m2',bx[bi]-35,yy-70,bx[bi]+35,yy+70);b.rect('m3',bx[bi]-35,yy-70,bx[bi]+35,yy+70)
                else:bus_via(bx[bi],yy)
            xx,yy=extend_input(chain[2],'B',y+80)
            wire('m2',bx[6],yy,xx,yy);via(2,bx[6],yy)
            if root not in final:
                xx,oy,_=point(chain[-1],'Z');yy=y+1140
                wire('m3',xx,oy,xx,yy);via(3,xx,yy)
                root_outputs[items[chain[-1]][1]['Z']]=(level,k,xx,yy)
            else:
                dr=[p for p in items if p.startswith(f'XWL_DRV_{row}/')]
                dr.sort(key=lambda p:len(items[p][0].devices),reverse=True)
                for p in dr:x+=put(p,x,row).width_nm
                link(chain[-1],dr[0]);link(dr[0],dr[1])
                xx,yy=extend_input(dr[0],'B',y+80);via(3,xx,yy)
                wx=driver_enable_track
                wire('m4',xx,yy,wx,yy,140);via(4,wx,yy)
                zx,zy,_=point(dr[-1],'Z');yy=y+HEIGHT/2
                wire('m3',zx,zy,zx,yy);wire('m3',zx,yy,width,yy)
                net=items[dr[-1]][1]['Z'];pins[net]=(width,yy,'m3')
                wordlines.append(dict(row=row,net=net,pin_nm=pins[net]))
    # Source chains can vary with driver class. Derive the actual edge.
    actual=max(x+v.width_nm for v,x,y,m in placed.values())
    if actual!=width:raise ValueError(f'Unexpected wordline-driver width: {actual} != {width}')
    # Global address trunks use M5, while local bus feeds use M6.
    inputs=list(dict.fromkeys(n for _,_,ns,_ in bus_defs for n in ns[:3]))
    globals={};extra_vss=[]
    for first,ox,ns,bx in bus_defs:
        for i,n in enumerate(ns[:3]):
            # Missing high address bits tie one or two buses of the first
            # slice to VSS; each such bus keeps its own grounded trunk.
            if n=='VSS' and n in globals and bx[i]!=globals[n] and bx[i] not in extra_vss:extra_vss.append(bx[i])
            globals.setdefault(n,bx[i])
    for n,x in globals.items():
        wire('m5',x,100,x,rows*HEIGHT-100,145);pins[n]=(x,100,'m5')
    for x in extra_vss:wire('m5',x,100,x,rows*HEIGHT-100,145)
    vss_reach=max([globals['VSS']]+extra_vss) if 'VSS' in globals else None
    if vss_reach is not None and any(x<=vss_reach for n,x in globals.items() if n!='VSS'):
        raise ValueError('Grounded address buses must precede every address trunk')
    pre_globals={}
    count=Counter(tier(r)[0] for r in roots)
    for n,(level,k,xx,yy) in root_outputs.items():
        tx=column_x(level)+1810+k*310
        # Disabled predecode branches retain a local unconnected output.
        child=[first for first,_,ns,_ in bus_defs if ns[6]==n]
        if not child:continue
        pre_globals[n]=tx
        wire('m4',xx,yy,tx,yy,140);via(4,tx,yy)
        if count[level]==1:wire('m5',tx,100,tx,rows*HEIGHT-100,145)
        else:
            # Repeated slices reuse a track within their own children's span.
            yc=child[0]*HEIGHT+1140;wire('m5',tx,min(yy,yc),tx,max(yy,yc),145)
    for first,ox,ns,bx in bus_defs:
        for i,n in enumerate(ns[:3]):
            yy=first*HEIGHT+240+300*i
            for level in (3,4):via(level,bx[i],yy)
            if bx[i]!=globals[n] and not (n=='VSS' and bx[i] in extra_vss):raise ValueError('Address bus phase mismatch')
        n=ns[6]
        if n in pre_globals:
            yy=first*HEIGHT+1140;tx=pre_globals[n]
            for level in (3,4,5):via(level,bx[6],yy)
            via(5,tx,yy);wire('m6',tx,yy,bx[6],yy,145)
        elif n=='VDD':
            yy=first*HEIGHT+985
            wire('m1',bx[6],yy,bx[6],first*HEIGHT+1265)
            via(2,bx[6],yy);via(1,bx[6],yy)
        else:raise ValueError('Unknown predecode enable')
    # M1 row supplies are common to every source gate. Two external M2 rails
    # avoid consuming signal tracks or requiring virtual supply connectivity.
    for n,x,phase in [('VDD',50,1265),('VSS',200,100)]:
        wire('m2',x,phase,x,(rows-1)*HEIGHT+phase)
        for r in range(rows):
            yy=r*HEIGHT+phase;wire('m1',x,yy,width,yy)
            b.via(1,x,yy);b.rect('m1',x-70,yy-35,x+70,yy+35)
            b.rect('m2',x-35,yy-70,x+35,yy+70)
        pins[n]=(x,phase,'m2')
        if n in globals:
            gx=vss_reach if n=='VSS' else globals[n];yy=phase
            via(2,x,yy);via(3,x,yy);via(4,x,yy)
            wire('m5',x,yy,gx,yy,145)
    wx=driver_enable_track
    wire('m5',wx,80,wx,(rows-1)*HEIGHT+80,145);pins['wl_en']=(wx,80,'m5')
    # Every leaf, including disabled root branches, is physically retained.
    if set(placed)!=set(items):raise ValueError('Unplaced decoder source leaves')
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    netports=tuple(pins)
    component=Component(b.name,netports,(),tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in c.ports),c) for n,c in specs.items()))
    v=CellView(component,b,width,rows*HEIGHT,pins,terminals,[],paths,vias)
    v.physical_fingers=fingers
    v.placements=[dict(instance=p,x_nm=x,y_nm=y,width_nm=c.width_nm,height_nm=c.height_nm) for p,(c,x,y,m) in placed.items()]
    v.wordlines=wordlines
    v.driver_start_nm=driver_start
    return v
