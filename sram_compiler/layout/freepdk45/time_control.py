"""TIME_CONTROL assembled from reusable standard cells and digital delays."""
from collections import defaultdict,Counter
from copy import copy,deepcopy
from dataclasses import replace
from math import ceil,floor
from typing import Any, Optional
from .components import Component,CellView,read_component
from .lib import Builder,new_layout,pya,u
from .control_place import StandardLibrary,ControlRouter,GRID
from .control_passives import make_filter,make_cap
from .dense_views import write_view


def units(component,mapping=None,path=''):
    mapping=dict(zip(component.ports,component.ports)) if mapping is None else mapping
    def net(n):return mapping.get(n,path+'/'+n if path else n)
    if component.name=='DFF' or component.devices:
        yield path,component,mapping
        return
    for i in component.children:
        yield from units(i.component,dict(zip(i.component.ports,(net(n) for n in i.nodes))),path+'/'+i.name if path else i.name)


def filters(component,mapping=None,path=''):
    mapping=dict(zip(component.ports,component.ports)) if mapping is None else mapping
    def net(n):return mapping.get(n,path+'/'+n if path else n)
    caps={p.nodes[0]:p for p in component.passives if p.name.startswith('C')}
    resistors=[p for p in component.passives if p.name.startswith('R')]
    if (len(component.passives)!=len(caps)+len(resistors)
            or len(resistors)!=len(caps)
            or {p.nodes[1] for p in resistors}!=set(caps)):
        raise ValueError('Every intentional passive must belong to one supported RC guard')
    for p in component.passives:
        if not p.name.startswith('R'):continue
        cap=caps.get(p.nodes[1])
        if cap is None or cap.nodes[1]!='VSS':raise ValueError('Unsupported intentional RC network')
        yield path+'/'+p.name if path else p.name,p.value,cap.value,{'IN':net(p.nodes[0]),'OUT':net(p.nodes[1]),'VSS':net(cap.nodes[1])}
    for i in component.children:
        yield from filters(i.component,dict(zip(i.component.ports,(net(n) for n in i.nodes))),path+'/'+i.name if path else i.name)


GUARD_MAX_OHM=6000.   # longest folded guard resistor inside its 4.6 x 6 um frame
GUARD_PITCH=(4900.,7000.)
ANTENNA_INPUTS=('pre_far','rwl','sen_far')


def corner_filters(component,mapping=None,path=''):
    """RC guards and capacitor-only settling loads of TIME_CONTROL.

    Yields (path, resistance or None, capacitance, connections). Values are
    historical source values; current controllers contain no intentional RC.
    """
    mapping=dict(zip(component.ports,component.ports)) if mapping is None else mapping
    def net(n):return mapping.get(n,path+'/'+n if path else n)
    caps={p.nodes[0]:p for p in component.passives if p.name.startswith('C')}
    resistors=[p for p in component.passives if p.name.startswith('R')]
    paired={p.nodes[1] for p in resistors}
    for p in resistors:
        cap=caps.get(p.nodes[1])
        if cap is None or cap.nodes[1]!='VSS':raise ValueError('Unsupported intentional RC network')
        yield path+'/'+p.name if path else p.name,p.value,cap.value,{'IN':net(p.nodes[0]),'OUT':net(p.nodes[1]),'VSS':net(cap.nodes[1])}
    for node,cap in caps.items():
        if node in paired:continue
        if cap.nodes[1]!='VSS':raise ValueError('Unsupported intentional capacitor')
        yield path+'/'+cap.name if path else cap.name,None,cap.value,{'TOP':net(node),'VSS':net('VSS')}
    for i in component.children:
        yield from corner_filters(i.component,dict(zip(i.component.ports,(net(n) for n in i.nodes))),path+'/'+i.name if path else i.name)


def _slices(library,component,limit):
    """Parallel finger groups of an oversized gate, each within ``limit`` nm.

    Every slice keeps the source topology and model; slice widths sum to
    the source width, and finger indices continue across slices.
    """
    whole=library.build(component)
    if whole.standard_family not in ('INV','NAND2','NAND3','NOR2'):
        raise ValueError(f'{component.name}: only standard gates can be sliced')
    count=ceil(whole.width_nm/limit)
    while True:
        parts=[]
        for i in range(count):
            devices=[]
            for d in component.devices:
                units5=round(d.width_m*1e9/5);q,r=divmod(units5,count)
                devices.append(replace(d,width_m=5*(q+(i<r))*1e-9))
            parts.append(replace(component,devices=tuple(devices)))
        views=[library.build(c) for c in parts]
        if all(v.width_nm<=limit for v in views):break
        count+=1
    offset=defaultdict(int);result=[]
    for v in views:
        fs=v.physical_fingers
        w=copy(v)
        w.physical_fingers=[dict(f,finger=f['finger']+offset[f['device']]) for f in fs]
        pins={}
        for key,point in v.device_pins.items():
            device,rest=key.split('@',1);finger,term=rest.split(':')
            pins[f'{device}@{int(finger)+offset[device]}:{term}']=point
        w.device_pins=pins
        for f in fs:offset[f['device']]+=1
        result.append(w)
    return result


def make_control(layout: pya.Layout,component: Component,width: float=32500.,exclude: tuple[str,...]=(),
                 pin_plan: Any=None,corner: bool=False,min_height: float=0.,
                 overflow: Optional[tuple[float,float]]=None) -> CellView:
    """Place and route TIME_CONTROL from its frozen source units.

    exclude lists unit-path prefixes that another physical block places, such
    as the data-register array. pin_plan maps each macro-facing net to an
    explicit (x, y, layer) access on the frame edge, replacing the default
    left-side M8 interface grid; M8 entries also reserve M9 above them.

    corner=True fits a fixed-width bank corner: cells start at the left edge,
    oversized gates split into parallel finger slices, and pin_plan may be a
    function of the final frame height. Archived RC decks retain a bottom band.
    overflow=(height, width) reserves a narrower lane above the corner for
    cells sharing the row-decoder floorplan. Routing obeys the same boundary.
    """
    if overflow is not None and (not corner or not 600<overflow[1]<=width or overflow[0]<=0):
        raise ValueError('Decoder overflow requires a positive corner height and a usable lane width')
    library=StandardLibrary(layout)
    b=Builder(layout,('TIME_CONTROL_CORNER_' if corner else 'TIME_CONTROL_LOGIC_' if exclude else 'TIME_CONTROL_STD_')+component.identity[:12])
    if corner:
        inventory=[]
        for p,c,m in units(component):
            if p.startswith(tuple(exclude)):continue
            v=library.build(c)
            limit=(overflow[1] if overflow is not None and v.standard_family!='DFF' else width)-600
            inventory.extend([(p,v,m)] if v.width_nm<=limit else [(p,s,m) for s in _slices(library,c,limit)])
    else:
        inventory=[(p,library.build(c),m) for p,c,m in units(component) if not p.startswith(tuple(exclude))]
    inventory.sort(key=lambda item:item[1].standard_family!='DFF')
    # Keep neighboring bank data bits in neighboring register columns. This
    # reduces crossings in the bus that leaves the controller's right side.
    data=[i for i,item in enumerate(inventory) if item[0].startswith('Xdff_buf_data/') and item[1].standard_family=='DFF']
    if len(data)==32:
        original=[inventory[i] for i in data]
        columns=[original[0:7],original[7:14],original[14:20],original[20:26],original[26:32]]
        ordered=[column[row] for row in range(7) for column in columns if row<len(column)]
        for i,item in zip(data,ordered):inventory[i]=item
    x0=300. if corner else 5200.;vss_x=150. if corner else 5000.
    guards=list(corner_filters(component)) if corner else []
    if any(c!=1e-15 or (r is not None and r>GUARD_MAX_OHM) for _,r,c,_ in guards):
        raise ValueError('Archived RC values exceed the calibrated passive templates')
    guard_views=[make_filter(layout,r,c,diode=True) if r is not None else make_cap(layout)
                 for _,r,c,_ in guards]
    if guard_views and max(v.width_nm for v in guard_views)>width-600:
        raise ValueError('Source passive footprint exceeds the controller frame width')
    shelves=[];guard_slots={}
    for i in sorted(range(len(guards)),key=lambda i:-guard_views[i].height_nm):
        v=guard_views[i]
        row=next((r for r,s in enumerate(shelves) if s['cursor']+v.width_nm<=width-300),None)
        if row is None:
            row=len(shelves);shelves.append(dict(cursor=300.,height=ceil((v.height_nm+1000)/GRID)*GRID))
        guard_slots[i]=(shelves[row]['cursor'],row)
        shelves[row]['cursor']=ceil((shelves[row]['cursor']+v.width_nm+300)/GRID)*GRID
    band=0.
    for shelf in shelves:
        shelf['y']=band;band+=shelf['height']
    prepared=[];cursor=x0;y=band;row_height=0.;row_vdd=0.;row=0;rows=[]
    def row_width(y: float) -> float:
        return overflow[1] if overflow is not None and y>=overflow[0] else width
    cell_top=overflow[0]-1000. if overflow is not None else None
    for index,(path,v,m) in enumerate(inventory):
        tap_space=500. if corner and (any(m[p] in ANTENNA_INPUTS for p in m if p not in ('VDD','VSS','Z'))
                                     or path=='Xpre_unbuf') else 0.
        # An edge-pinned frame starts logic on a new row instead of sharing
        # the last register row.
        family_break=pin_plan is not None and prepared and (prepared[-1][1].standard_family=='DFF')!=(v.standard_family=='DFF')
        boundary=overflow is not None and y<overflow[0] and y+max(row_height,v.height_nm)>cell_top
        if (cursor+v.width_nm+tap_space>row_width(y)-300 and cursor>x0) or family_break or boundary:
            if not row_height:raise ValueError('Source passive band exceeds the available corner height')
            rows.append((y,row_height,row_vdd))
            # Preserve clearance between the register's M5/M6 clock rails
            # as well as the wells at the next row's mixed logic boundary.
            gap=(100 if v.standard_family=='DFF' else 125) if row_height==1800 else 0
            y+=row_height+gap;cursor=x0;row_height=0;row_vdd=0;row+=1
        if overflow is not None and y<overflow[0] and y+v.height_nm>cell_top:
            y=overflow[0]
        x=cursor
        if x+v.width_nm+tap_space>row_width(y)-300:raise ValueError(f'{path} ({v.width_nm} nm) does not fit the {row_width(y)} nm controller row')
        prepared.append((path,v,m,x,y,row,False))
        next_dff=index+1<len(inventory) and inventory[index+1][1].standard_family=='DFF'
        cursor=x+v.width_nm+((190 if next_dff else 290) if v.standard_family=='DFF' else 30)
        cursor+=tap_space
        row_height=max(row_height,v.height_nm);row_vdd=max(row_vdd,v.pins['VDD'][1])
    rows.append((y,row_height,row_vdd))
    passive_locations=[]
    if corner:
        for i,(path,r,c,m) in enumerate(guards):
            v=guard_views[i]
            x,row=guard_slots[i];py=shelves[row]['y']
            prepared.append((path,v,m,x,py,len(rows),True))
            passive_locations.append(dict(instance=path,x_nm=x,y_nm=py,connections=m,source_resistance_ohm=r,
                source_capacitance_f=c,**v.passive_parameters))
    else:
        for i,(path,r,c,m) in enumerate(filters(component)):
            v=make_filter(layout,r,c);x=300.;py=i*6200.
            prepared.append((path,v,m,x,py,len(rows),True))
            passive_locations.append(dict(instance=path,x_nm=x,y_nm=py,connections=m,**v.passive_parameters))
    signal_ports=[p for p in component.ports if p not in ('VDD','VSS')]
    if corner:
        # A 1 um top margin holds the antenna diodes under the top-edge pins.
        height=ceil(max(y+row_height+1000,min_height)/GRID)*GRID
        if callable(pin_plan):pin_plan=pin_plan(height)
    else:
        interface_height=800+400*ceil(len(signal_ports)/2) if pin_plan is None else max(p[1] for p in pin_plan.values())+1000
        height=ceil(max(y+row_height+200,24800.,interface_height)/GRID)*GRID
    print('TIME_CONTROL frame',width,height,'units',len(prepared),flush=True)
    terminals={};physical=[];placed=[];routed=defaultdict(list);access=[];ownership=[];blocked_boxes=[]
    def wire(layer,x,y,xx,yy,w=70):b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    def lift(x,y,layer,target):
        for n in range(min(layer,target),max(layer,target)):
            if n>=7:b.sq(f'v{n}',x,y,400)
            else:b.via(n,x,y)
            for k in (n,n+1):
                if k==1:b.rect('m1',x-70,y-35,x+70,y+35)
                elif k==2:
                    if n==2:b.rect('m2',x-35,y-70,x+35,y+70)
                    else:b.rect('m2',x-70,y-35,x+70,y+35)
                elif k==3:b.rect('m3',x-35,y-70,x+35,y+70)
                else:b.sq(f'm{k}',x,y,400 if k==7 else 405 if k==8 else 145)
    for path,v,m,x,y,row,passive in prepared:
        b.place(v.builder,x,y)
        if v.standard_family=='DFF' and any(
                q[1].standard_family=='DFF' and q[5]==row and q[3]==x+v.width_nm+190
                for q in prepared):
            # Match OpenRAM's register-array abutment: continue the same
            # wells and supply rails through the inter-register access gap.
            b.rect('pwell',x+v.width_nm,y+100,x+v.width_nm+190,y+780)
            b.rect('nwell',x+v.width_nm,y+780,x+v.width_nm+190,y+1700)
            for phase in (200,1600):wire('m1',x+v.width_nm-35,y+phase,x+v.width_nm+225,y+phase)
        if not passive and v.standard_family!='DFF':
            b.rect('pwell',x+v.width_nm,y,x+v.width_nm+30,y+640)
            b.rect('nwell',x+v.width_nm,y+640,x+v.width_nm+30,y+1500)
            for phase in (100,1400):b.rect('m1',x+v.width_nm,y+phase-35,x+v.width_nm+30,y+phase+35)
            if row and rows[row-1][1]==1800 and y-rows[row-1][0]==1900:
                b.rect('pwell',x,y-200,x+v.width_nm+30,y)
        placed.append(dict(instance=path,cell=v.builder.name,family=v.standard_family,x_nm=x,y_nm=y,width_nm=v.width_nm,height_nm=v.height_nm))
        terminals.update({path+'/'+n:(x+px,y+yy,l) for n,(px,yy,l) in v.device_pins.items()})
        fs=getattr(v,'physical_fingers',None)
        if fs is None:fs=[dict(device=d.name,finger=0,width_nm=d.width_m*1e9,length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes) for d in v.component.devices]
        for f in fs:
            f=dict(f,device=path+'/'+f['device'])
            if 'gate_xy_nm' in f:f['gate_xy_nm']=(x+f['gate_xy_nm'][0],y+f['gate_xy_nm'][1])
            physical.append(f)
        composite=v.standard_family in ('DFF','RC_GUARD')
        if passive:blocked_boxes.append((x-100,y-100,x+v.width_nm+100,y+v.height_nm+100,2))
        for p,(px,yy,l) in v.pins.items():
            net=m[p];px+=x;yy+=y
            if p in ('VDD','VSS'):
                if passive:
                    lift(px,yy,int(l[1:]),4)
                    wire('m4',px,yy,px,yy+400,145);wire('m4',px,yy+400,vss_x,yy+400,145)
                    lift(vss_x,yy+400,3,4)
                else:
                    lift(px,yy,int(l[1:]),2)
                    ty=rows[row][0]+(90 if p=='VSS' else rows[row][2])
                    wire('m2',px,yy,px,ty)
                    if v.standard_family=='DFF' and p=='VSS':b.rect('m2',px-70,ty-35,px+70,yy+35)
                continue
            if v.standard_family=='DFF':
                tx=(ceil(px/GRID) if p=='CLK' else round(px/GRID))*GRID;ty=round(yy/GRID)*GRID
                wire('m7',px,yy,tx,yy,400);wire('m7',tx,yy,tx,ty,400)
                lift(tx,ty,7,8);point=(tx,ty,'m8');ownership.append((net,(px,yy,'m7')))
            elif passive:
                tx=round(px/GRID)*GRID;ty=y+(5400 if p in ('IN','TOP') else 6300)
                wire('m6',px,yy,tx,yy,145);wire('m6',tx,yy,tx,ty,145);point=(tx,ty,'m6')
            elif l=='m3':
                names=[k for k in v.pins if k not in ('VDD','VSS','Z')]
                phase=({'IN':100,'OUT':1400,'CTR_N':400,'CTR_P':1050}[p] if v.standard_family=='TG'
                       else 1400 if p=='Z' else 1000 if any(q[2]=='m4' for q in v.pins.values()) else 300+350*names.index(p))
                ty=round((y+phase)/GRID)*GRID;tx=round(px/GRID)*GRID
                wire('m3',px,yy,px,ty);wire('m3',px,ty,tx,ty)
                lift(tx,ty,3,4);point=(tx,ty,'m4')
                if corner and (net in ANTENNA_INPUTS or (path=='Xpre_unbuf' and p=='A')):
                    # A reserved filler protects the observer before upper
                    # routing layers reach its far-end driver. Source MOS
                    # sizes and the far-end feedback path stay fixed.
                    from .control_passives import antenna_diode
                    tapx=x+v.width_nm+250;tapy=y+300
                    b.rect('pwell',x+v.width_nm,y,x+v.width_nm+530,y+640)
                    b.rect('nwell',x+v.width_nm,y+640,x+v.width_nm+530,y+1500)
                    antenna_diode(b,tapx,tapy);lift(tapx,tapy,3,4)
                    wire('m4',tx,ty,tapx,ty,145);wire('m4',tapx,ty,tapx,tapy,145)
                    ownership.append((net,(tapx,tapy,'m4')))
            else:
                tx=round(px/GRID)*GRID;ty=round(yy/GRID)*GRID
                wire(l,px,yy,tx,yy,145);wire(l,tx,yy,tx,ty,145);point=(tx,ty,l)
            routed[net].append(point);access.append((net,point));ownership.append((net,point))
    # Row supplies are below/above the devices on M2, with M3 edge trunks.
    pins={}
    supply_top={}
    for net in ('VSS','VDD'):
        ys=[yy+(90 if net=='VSS' else vdd) for yy,h,vdd in rows]
        if net=='VSS':ys.extend(p['y_nm']+6100 for p in passive_locations)
        sites=[]
        for yy,h,vdd in rows:
            edge=row_width(yy)-100;tx=vss_x if net=='VSS' else edge
            py=yy+(90 if net=='VSS' else vdd)
            wire('m2',vss_x,py,edge,py);lift(tx,py,2,3);sites.append((tx,py))
        for (x1,y1),(x2,y2) in zip(sites,sites[1:]):
            if x1!=x2:
                # Change trunk width in the reserved gap. A via2 on the last
                # wide row can land on an unrelated gate's M3 output escape.
                transition=overflow[0]-800.
                wire('m3',x1,y1,x1,transition);lift(x1,transition,3,4)
                wire('m4',x1,transition,x2,transition,145);lift(x2,transition,3,4)
                ownership.extend((net,(x,transition,'m4')) for x in (x1,x2))
                wire('m3',x2,transition,x2,y2)
            else:wire('m3',x2,y1,x2,y2)
        if net=='VSS':wire('m3',vss_x,min(ys),vss_x,max(ys))
        pins[net]=(*sites[0],'m3');supply_top[net]=sites[-1]
    wire('m3',vss_x,pins['VSS'][1],vss_x,200);lift(vss_x,200,3,4);wire('m4',100,200,vss_x,200,145)
    pins['VSS']=(100,200,'m4')
    vdd_x,vdd_y=row_width(height)-100.,height-(900. if corner else 200.)
    vdd_source_y=supply_top['VDD'][1]
    lower_supply=(overflow is not None and pin_plan is not None
                  and 'VDD' in pin_plan and pin_plan['VDD'][1]<overflow[0])
    if lower_supply:
        # The native M3/M2 rails already join every row. Access their lower
        # trunk beside the bank supply, avoiding a redundant tall metal route.
        vdd_x,vdd_y=width-100.,pin_plan['VDD'][1];vdd_source_y=pins['VDD'][1]
    for net,tx,ty in [('VSS',100.,200.),('VDD',vdd_x,vdd_y)]:
        if net=='VDD':
            if not lower_supply and tx!=supply_top[net][0]:lift(tx,supply_top[net][1],2,3)
            wire('m3',tx,vdd_source_y,tx,ty);lift(tx,ty,3,4)
        point=(tx,ty,'m4');routed[net].append(point);access.append((net,point));ownership.append((net,point))
    # Registers expose their native high-metal D/Q access. Bank routing owns
    # the data/address connections; no redundant perimeter bus is inserted.
    external={p:routed[p][0] for p in signal_ports if p in routed}
    if pin_plan is not None:
        for net,point in pin_plan.items():
            if net!='VDD' and net!='VSS' and net not in routed:raise ValueError(f'Planned controller pin has no source connection: {net}')
            external[net]=point;b.sq(point[2],point[0],point[1],145 if point[2] in ('m5','m6') else 405)
            routed[net].append(point);access.append((net,point));ownership.append((net,point))
            if corner and point[2]=='m5' and (net.startswith('A_dff') or net=='wl_en'):
                # Top-edge pins drive long decoder spines that reach gates only;
                # an N+ tap at the pin keeps them antenna-safe whatever the route.
                px,dy=point[0],point[1]-350.
                b.rect('pwell',px-130,dy-130,px+130,dy+130);b.rect('nplus',px-105,dy-105,px+105,dy+105)
                b.rect('active',px-75,dy-75,px+75,dy+75);b.cont(px,dy);b.rect('m1',px-35,dy-70,px+35,dy+70)
                lift(px,dy,1,5);wire('m5',px,dy,px,point[1],145)
                ownership.extend((net,(px,dy,layer)) for layer in ('m4','m5'))
    interface=[] if pin_plan is not None else [p for p in component.ports if p.startswith('A_dff') or p in
               ('wl_en','s_en','w_en','PRE','sa_iso','rbl','rwl','pre_far','iso_far','sen_far','wen_far')]
    for i,net in enumerate(interface):
        point=(1200.+1800*(i%3),1200.+3600*(i//3),'m8');external[net]=point
        b.sq('m8',point[0],point[1],405);routed[net].append(point);access.append((net,point));ownership.append((net,point))
    for net,x in ([] if pin_plan is not None else [('VSS',1200.),('VDD',3000.)]):
        point=(x,23000.,'m8');external[net]=point
        b.sq('m8',*point[:2],405);routed[net].append(point);access.append((net,point));ownership.append((net,point))
    router=ControlRouter(b,width,height,upper_metal=9,overflow=overflow)
    for layer_index,layer in enumerate(router.layers):
        reg=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
        sites=[(n,pya.Point(u(pt[0]),u(pt[1]))) for n,pt in ownership if pt[2]==layer]
        for pg in reg.each():
            bb=pg.bbox();scale=layout.dbu*1000
            owners={net for net,point in sites if pg.inside(point)}
            if len(owners)>1:raise ValueError(f'Shorted fixed conductor on {layer}: {owners}')
            clearance=1202.5 if layer_index>=5 else 602.5 if layer_index>=3 else 212.5
            expanded=pya.Region(pg).sized(u(clearance))
            for contour in expanded.each():
                box=contour.bbox()
                for xi in range(ceil(box.left*scale/GRID),floor(box.right*scale/GRID)+1):
                    for yi in range(ceil(box.bottom*scale/GRID),floor(box.top*scale/GRID)+1):
                        if not contour.inside(pya.Point(u(xi*GRID),u(yi*GRID))):continue
                        pt=(xi,yi,layer_index)
                        if owners:router.clearance.setdefault(pt,set()).update(owners)
                        else:router.blocked.add(pt)
        for x0,y0,x1,y1,limit in blocked_boxes:
            if layer_index>=limit:continue
            for xi in range(floor(x0/GRID),ceil(x1/GRID)+1):
                for yi in range(floor(y0/GRID),ceil(y1/GRID)+1):router.blocked.add((xi,yi,layer_index))
    for net,pt in access:
        q=router.point(*pt)
        router.blocked.discard(q);router.reserve(q,net)
    integration_ports=[p for p in component.ports if p.startswith(('DIN_dff','A_dff')) or p in ('VDD','VSS','wl_en','s_en','w_en','PRE','sa_iso','rbl','rwl','pre_far','iso_far','sen_far','wen_far')]
    if pin_plan is not None:integration_ports=[n for n,p in pin_plan.items() if p[2]=='m8']
    for net in integration_ports:
        point=external[net]
        if point[2]!='m8':raise ValueError('Macro interface requires an M8 landing')
        router.reserve((round(point[0]/GRID),round(point[1]/GRID),5),net)
    # Internal leaf M6 escapes are already connected conductors. Opening the
    # terminal's final point is sufficient: all global routing starts outside
    # each composite's protected footprint.
    nets=sorted(routed,key=lambda n:(n!='clk_buf',max(p[0] for p in routed[n])-min(p[0] for p in routed[n])+max(p[1] for p in routed[n])-min(p[1] for p in routed[n]),n))
    from .router import RoutingError
    fixed_blocked=set(router.blocked);fixed_clearance={p:set(n) for p,n in router.clearance.items()}
    # Route scarce top-edge address accesses before the interior fills up.
    priority=[n for n in nets if n.startswith('A_dff')] if height>80000. else []
    clock_state=None
    for attempt in range(32):
        router=ControlRouter(b,width,height,upper_metal=9,overflow=overflow);router.blocked=set(fixed_blocked)
        router.clearance={p:set(n) for p,n in fixed_clearance.items()}
        for net,pt in access:
            q=router.point(*pt);router.reserve(q,net)
            if q[2]==0:router.reserve((q[0],q[1],1),net)
        for net,pt in access:
            q=router.point(*pt)
            if router.clearance.get(q,set())-{net}:raise ValueError(f'Conflicting fixed access: {net} {q} {router.clearance[q]}')
        if clock_state is not None:
            # Every retry starts from the same completed clock tree. Copy all
            # routing state while retaining the native, undrawn GDS builder.
            router=deepcopy(clock_state,{id(b):b})
        local=[n for n in nets if n not in component.ports and max(p[0] for p in routed[n])-min(p[0] for p in routed[n])+max(p[1] for p in routed[n])-min(p[1] for p in routed[n])<6000]
        order=['clk_buf']+priority+[n for n in nets if n!='clk_buf' and n not in priority]
        if overflow is not None and height>80000.:
            # Preserve channels for long feedback and interface nets before
            # connecting the small local groups between neighboring gates.
            order=['clk_buf']+priority+[n for n in nets if n!='clk_buf' and n not in priority and n not in local]
            order.extend(n for n in nets if n!='clk_buf' and n not in priority and n in local)
        try:
            for i,net in enumerate(order):
                if net=='clk_buf' and clock_state is not None:continue
                ps=list(dict.fromkeys(router.point(*p) for p in routed[net]))
                limit=router.search_limit
                if net=='clk_buf' or net in priority or (overflow is not None and height>80000. and net not in local):
                    router.search_limit=None
                router.connect(net,ps[0],ps[1:])
                router.search_limit=limit
                if net=='clk_buf' and height>80000.:clock_state=deepcopy(router,{id(b):b})
                if i%100==0:print('TIME_CONTROL routing attempt',attempt,'net',i,'/',len(nets),flush=True)
            break
        except RoutingError as error:
            if error.net=='clk_buf' or (priority and priority[0]==error.net):raise
            if error.net in priority:priority.remove(error.net)
            priority.insert(0,error.net)
            print('Retrying with priority for',error.net,flush=True)
    else:raise ValueError('Controller routing did not converge after 32 deterministic retries')
    router.draw();pins.update(external)
    if corner:
        from .conductor_ownership import port_conductors
        check_points={str(i):point for i,(_,point) in enumerate(ownership)}
        check_aliases={str(i):net.upper() for i,(net,_) in enumerate(ownership)}
        if not guards:
            # Close same-net M4-M6 notches between native bars and grid escapes.
            conductors=port_conductors(b,check_points,check_aliases,minimum_metal=4)
            groups=defaultdict(pya.Region)
            for net,layer,polygon in conductors:
                if layer in (17,19,21):groups[net,layer].insert(polygon)
            for (_,layer),region in groups.items():
                fill=region.merged().sized(u(67.5)).sized(-u(67.5))-region
                if not fill.is_empty():b.cell.shapes(b.ly.layer(layer,0)).insert(fill.merged())
        # Same-net notches between leaf stubs and routed metal: close gaps
        # narrower than the layer spacing (different nets are already wider apart).
        # The radius stays below half the spacing: legal gaps between nets survive.
        for layer,radius in (('m7',197.5),('m8',197.5),('m9',397.5),('m10',397.5)):
            have=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
            fill=have.sized(u(radius)).sized(-u(radius))-have
            if not fill.is_empty():b.cell.shapes(b.li(layer)).insert(fill.merged())
            # Corner-offset necks (two staggered same-net wires) are filled with
            # the bounding box of that polygon's local piece.
            minw=140. if layer in ('m4','m5','m6') else 400. if layer in ('m7','m8') else 800.;space=2*radius+5
            have=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
            counts=lambda r:(r.width_check(u(minw)).count(),r.isolated_check(u(space)).count(),r.notch_check(u(space)).count())
            before=counts(have)
            # A filled elbow can expose another short neck at its new edge.
            # Recheck the geometry, retaining the existing spacing safeguards.
            for repair in range(4):
                if have.width_check(u(minw)).is_empty():break
                for pair in have.width_check(u(minw)).each():
                    window=pya.Region(pair.bbox().enlarged(u(2*minw),u(2*minw)))
                    for pg in have.each():
                        if pg.bbox().overlaps(pair.bbox()) and not (pya.Region(pg)&pya.Region(pair.bbox().enlarged(1,1))).is_empty():
                            local=(pya.Region(pg)&window).bbox()
                            # A merged region hides a new short from spacing
                            # checks. Keep the fill clear of every other wire.
                            foreign=have-pya.Region(pg)
                            if (pya.Region(local).sized(u(space))&foreign).is_empty():
                                b.cell.shapes(b.li(layer)).insert(local)
                have=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
            after=counts(pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged())
            if after[0] or after[1]>before[1] or after[2]>before[2]:
                raise ValueError(f'Controller {layer} width repair failed: width/isolation/notch {before} -> {after}')
        from .conductor_ownership import port_conductors
        from .export import Nets
        aliases=Nets()
        for _,r,_,mapping in guards:
            if r is not None:aliases.union(mapping['IN'].upper(),mapping['OUT'].upper())
        check_points={str(i):point for i,(_,point) in enumerate(ownership)}
        check_aliases={str(i):aliases.find(net.upper()) for i,(net,_) in enumerate(ownership)}
        port_conductors(b,check_points,check_aliases,require_connected=True)
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(component,b,width,height,pins,terminals,[],router.paths,router.vias)
    v.physical_fingers=physical;v.placements=placed;v.passive_locations=passive_locations
    v.standard_cell_inventory=dict(Counter(p['family'] for p in placed));v.standard_family='TIME_CONTROL'
    return v


def build(deck,output):
    c=read_component(deck,'XTIME_CONTROL',allow_passives=True)
    v=make_control(new_layout(),c)
    if v.builder.cell.dbbox().area()>872.20185+1e-9:
        raise ValueError('Controller exceeds the OpenRAM controller plus input-register area budget')
    return write_view(v,output,deck,'XTIME_CONTROL including input registers',
        dict(controller_layout=True,standard_cell_inventory=v.standard_cell_inventory,
             passive_locations=v.passive_locations,standard_row_height_nm=1500,
             waveform_validation='not run',defaults_promotable=False))


def main(argv=None):
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    build(args.deck,args.output_dir)
    print(args.output_dir/'full_macro.gds')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
