"""Source-owned decoder and wordline drivers on 1365 nm rows.

Each 3-to-8 slice has local M3 input buses. M2 joins its row gates;
M4/M5/M6 distribute address and predecode signals without stretching rows.
Decoder level l occupies its own 3.9 um column. A level-l slice sits at the
first row of the children it enables, so its eight output trunks stay within
that span and later slices reuse the same trunk tracks.

A split bank builds two halves. Each half places its own rows and the
predecode slices whose children it holds, at the end of their span farthest
from the centre periphery; a slice with children in both halves (level 0)
stays at row 0 of the lower half. Trunks end at the half's band edge, where
every net shared with the other half or the controller has its pin.
"""
from collections import Counter, defaultdict
import re
from .components import Component, Instance, CellView
from .cell_flow import leaves
from .dense_logic import make_logic, HEIGHT, _plan
from .lib import Builder, pya, u


def make_decoder(layout, source, specs, span=None, spare=0.):
    """span=(r0, r1) builds one half of a split bank at local y=(row-r0)*HEIGHT.

    The lower half (r0=0) faces the band with its top edge, the upper half
    (r1=rows) with its bottom edge. spare (a multiple of 100 nm) widens the
    empty space left of the first decoder column for controller gates.
    """
    if spare<0 or spare%100:raise ValueError('Spare decoder width must be a non-negative multiple of 100 nm')
    rows,cols=source.rows,source.cols
    if rows<8 or rows%8 or rows>512:
        raise ValueError('OpenRAM-pitch decoder supports 8 to 512 rows in multiples of 8')
    r0,r1=span or (0,rows)
    if span is not None and (r0%8 or r1%8 or not (r0==0)!=(r1==rows) or not 0<=r0<r1<=rows):
        raise ValueError('A decoder half spans whole 8-row slices from one array edge')
    band=None if span is None else 'top' if r0==0 else 'bottom'
    # The OpenRAM area ceilings describe the 32x32 reference driver classes.
    ceiling=rows<=32 and cols<=32
    every={p:(c,source.instances[p]['pins']) for n,c in specs.items()
           for p,c,_ in leaves(c,path=n)}
    b=Builder(layout,f'DENSE_DECODER_{rows}' if span is None else f'DENSE_DECODER_{rows}_R{r0}_{r1}')
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
    def Y(row):return (row-r0)*HEIGHT
    def put(path,x,row):
        c,m=items[path]
        if c.identity not in cache:cache[c.identity]=make_logic(layout,c,ceiling)
        v=cache[c.identity];y=Y(row);b.place(v.builder,x,y)
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
    every_root=sorted({p.rsplit('/',1)[0] for p in every if '/XINV_A' in p},key=tier)
    levels=1+max(tier(r)[0] for r in every_root)
    def first_row(level,index):return 8*8**(levels-1-level)*index
    def anchor(root):
        """Placement row of a slice in this build, or None for the other half."""
        level,index=tier(root);start=first_row(level,index)
        if span is None:return start
        end=min(start+8*8**(levels-1-level),rows)
        if r0<=start and end<=r1:return end-8 if band=='bottom' else start
        return start if r0==0 and start<r1<end else None
    roots=[r for r in every_root if anchor(r) is not None]
    def owner(p):
        match=re.match(r'(.*/XDEC_\d+_\d+)/',p)
        if match:return match.group(1) in roots
        return r0<=int(re.match(r'XWL_DRV_(\d+)/',p).group(1))<r1
    items={p:v for p,v in every.items() if owner(p)}
    final=[r for r in roots if tier(r)[0]==levels-1]
    if final!=[r for r in roots if any(m['Z'].startswith('DEC_WL') for p,(c,m) in items.items() if p.startswith(r+'/'))]:
        raise ValueError('Only the last decoder level may drive wordlines')
    if [tier(r)[1] for r in final]!=list(range(r0//8,r1//8)):raise ValueError('Final decoder slices must cover every row')
    group_width=3900.;base=300.+spare
    def column_x(level):return base+level*group_width
    main_x=column_x(levels-1)
    chain_start=1610.
    # Dense gate width is 240 nm plus one 190 nm pitch per physical gate.
    chain_width=sum(240+190*len(_plan(items[final[0]+suffix][0])[3]) for suffix in
                    ('/XAND0/XPNAND3','/XAND0/XPINV','/XAND_EN0/XPNAND3','/XAND_EN0/XPINV'))
    bus_defs=[];root_outputs={};driver_start=main_x+chain_start+chain_width
    driver_views=[]
    for p,(c,m) in items.items():
        if p.startswith(f'XWL_DRV_{r0}/'):
            if c.identity not in cache:cache[c.identity]=make_logic(layout,c,ceiling)
            driver_views.append(cache[c.identity])
    driver_width=sum(v.width_nm for v in driver_views)
    driver_enable_track=driver_start+driver_views[0].width_nm+100
    width=driver_start+driver_width
    pins={};wordlines=[]
    for root in roots:
        level,index=tier(root)
        first=anchor(root);ox=column_x(level)
        # Three true input buses, their three complements, and one enable.
        invpaths=[root+f'/XINV_A{i}' for i in (1,2,3)]
        invmaps=[items[p][1] for p in invpaths]
        sample=items[root+'/XAND_EN0/XPNAND3'][1]
        nets=[m['A'] for m in invmaps]+[m['Z'] for m in invmaps]+[sample['B']]
        # Duplicate formal ties (e.g. root A0=VSS) share the same logical net.
        bx=[ox+535+i*150 for i in (0,2,4,1,3,5,6)]
        for i,net in enumerate(nets):wire('m3',bx[i],Y(first)+80,bx[i],Y(first+8)-100)
        bus_defs.append((first,ox,nets,bx))
        for i,p in enumerate(invpaths):
            v=put(p,ox,first+i);y=Y(first+i)
            xx,yy=extend_input(p,'A',y+80);wire('m2',xx,yy,bx[i],yy);via(2,bx[i],yy)
            sx=ox+v.output_rail_m2[0];yy=y+985
            wire('m2',sx,yy,bx[i+3],yy);via(2,bx[i+3],yy)
        for k in range(8):
            row=first+k;y=Y(row);x=ox+chain_start
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
    top=Y(r1)
    # A half's shared trunks end at its band edge (``edge``), where the bank
    # or controller joins them; other trunks span only their own uses.
    edge=None if band is None else top-100 if band=='top' else 100.
    def trunk(x,ys,pin=None):
        if edge is None:
            wire('m5',x,100,x,rows*HEIGHT-100,145)
        else:
            ys=list(ys)+([edge] if pin else [])
            wire('m5',x,min(ys),x,max(ys),145)
        if pin:pins[pin]=(x,100 if edge is None else edge,'m5')
    uses=defaultdict(list)
    for first,ox,ns,bx in bus_defs:
        for i in range(3):uses[bx[i]].append(Y(first)+240+300*i)
    vss_x=set(extra_vss)|({globals['VSS']} if 'VSS' in globals else set())
    for n,x in globals.items():
        # Grounded buses tie to the row-0 VSS rail and need no band access.
        trunk(x,uses[x]+([100.] if n=='VSS' else []),None if n=='VSS' and edge is not None else n)
    for x in extra_vss:trunk(x,uses[x]+[100.])
    vss_reach=max([globals['VSS']]+extra_vss) if 'VSS' in globals else None
    if vss_reach is not None and any(x<=vss_reach for n,x in globals.items() if n!='VSS'):
        raise ValueError('Grounded address buses must precede every address trunk')
    pre_globals={}
    count=Counter(tier(r)[0] for r in roots)
    # Predecode outputs of the whole decoder, for nets shared by both halves.
    producer={every[r+f'/XAND_EN{k}/XPINV'][1]['Z']:(tier(r)[0],k) for r in every_root
              if tier(r)[0]<levels-1 for k in range(8)}
    consumer={every[r+'/XAND_EN0/XPNAND3'][1]['B'] for r in every_root}
    for n,(level,k,xx,yy) in root_outputs.items():
        tx=column_x(level)+1810+k*310
        # Disabled predecode branches retain a local unconnected output.
        child=[first for first,_,ns,_ in bus_defs if ns[6]==n]
        if not child and (edge is None or n not in consumer):continue
        pre_globals[n]=tx
        wire('m4',xx,yy,tx,yy,140);via(4,tx,yy)
        if edge is not None:
            # Enables of the other half's slices continue to the band edge.
            trunk(tx,[yy]+[Y(c)+1140 for c in child],None if child else n)
        elif count[level]==1:wire('m5',tx,100,tx,rows*HEIGHT-100,145)
        else:
            # Repeated slices reuse a track within their own children's span.
            yc=child[0]*HEIGHT+1140;wire('m5',tx,min(yy,yc),tx,max(yy,yc),145)
    if edge is not None:
        for first,ox,ns,bx in bus_defs:
            n=ns[6]
            if n in producer and n not in pre_globals:
                level,k=producer[n];tx=column_x(level)+1810+k*310
                pre_globals[n]=tx;trunk(tx,[Y(first)+1140],n)
    for first,ox,ns,bx in bus_defs:
        for i,n in enumerate(ns[:3]):
            yy=Y(first)+240+300*i
            for level in (3,4):via(level,bx[i],yy)
            if bx[i]!=globals[n] and not (n=='VSS' and bx[i] in extra_vss):raise ValueError('Address bus phase mismatch')
        n=ns[6]
        if n in pre_globals:
            yy=Y(first)+1140;tx=pre_globals[n]
            for level in (3,4,5):via(level,bx[6],yy)
            via(5,tx,yy);wire('m6',tx,yy,bx[6],yy,145)
        elif n=='VDD':
            yy=Y(first)+985
            wire('m1',bx[6],yy,bx[6],Y(first)+1265)
            via(2,bx[6],yy);via(1,bx[6],yy)
        else:raise ValueError('Unknown predecode enable')
    # M1 row supplies are common to every source gate. Two external M2 rails
    # avoid consuming signal tracks or requiring virtual supply connectivity.
    for n,x,phase in [('VDD',50,1265),('VSS',200,100)]:
        wire('m2',x,phase,x,Y(r1-1)+phase)
        for r in range(r0,r1):
            yy=Y(r)+phase;wire('m1',x,yy,width,yy)
            b.via(1,x,yy);b.rect('m1',x-70,yy-35,x+70,yy+35)
            b.rect('m2',x-35,yy-70,x+35,yy+70)
        pins[n]=(x,Y(r1-1)+phase if band=='top' else phase,'m2')
        if n in globals:
            gx=vss_reach if n=='VSS' else globals[n];yy=phase
            via(2,x,yy);via(3,x,yy);via(4,x,yy)
            wire('m5',x,yy,gx,yy,145)
    wx=driver_enable_track
    if band=='top':
        wire('m5',wx,80,wx,top-100,145);pins['wl_en']=(wx,top-100,'m5')
    else:
        wire('m5',wx,80,wx,Y(r1-1)+80,145);pins['wl_en']=(wx,80,'m5')
    # Every leaf, including disabled root branches, is physically retained.
    if set(placed)!=set(items):raise ValueError('Unplaced decoder source leaves')
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    netports=tuple(pins)
    component=Component(b.name,netports,(),tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in c.ports),c) for n,c in specs.items()))
    v=CellView(component,b,width,top,pins,terminals,[],paths,vias)
    v.physical_fingers=fingers
    v.placements=[dict(instance=p,x_nm=x,y_nm=y,width_nm=c.width_nm,height_nm=c.height_nm) for p,(c,x,y,m) in placed.items()]
    v.wordlines=wordlines
    v.driver_start_nm=driver_start
    # Column start of the leftmost slice per row (gates, buses and trunks of
    # a slice stay right of it): the spare space a controller may use.
    v.row_left_nm={}
    for first,ox,ns,bx in bus_defs:
        for row in range(first,first+8):v.row_left_nm[row]=min(v.row_left_nm.get(row,ox),ox)
    # Trunks of other slices cross rows left of that column: a controller's
    # pin accesses there keep clear of each M4-M6 conductor (layer, x0, x1).
    v.row_metal_nm={}
    for row in range(r0,r1):
        box=pya.Box(u(0),u(Y(row)+2.5),u(v.row_left_nm[row]),u(Y(row)+HEIGHT-2.5));found=[]
        for layer in ('m4','m5','m6'):
            region=pya.Region(b.cell.begin_shapes_rec_overlapping(b.li(layer),box)).merged()
            found.extend((layer,pg.bbox().left*layout.dbu*1000,pg.bbox().right*layout.dbu*1000) for pg in region.each())
        v.row_metal_nm[row]=sorted(found,key=lambda t:t[1])
    # Well (nwell or pwell) intervals within 225 nm of each row's bottom and
    # top edge: controller wells in the neighbouring row end clear of them.
    v.row_well_nm={}
    for row in range(r0,r1):
        edges={}
        for side,y0 in (('bottom',Y(row)),('top',Y(row)+HEIGHT-225)):
            box=pya.Box(u(0),u(y0),u(width),u(y0+225))
            region=pya.Region()
            for layer in ('nwell','pwell'):region+=pya.Region(b.cell.begin_shapes_rec_overlapping(b.li(layer),box))
            edges[side]=sorted((pg.bbox().left*layout.dbu*1000,pg.bbox().right*layout.dbu*1000)
                               for pg in (region&pya.Region(box)).merged().each())
        v.row_well_nm[row]=edges
    return v
