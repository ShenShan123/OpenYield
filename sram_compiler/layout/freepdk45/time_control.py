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


def _corner_inventory(library,component,width,exclude):
    inventory=[]
    for p,c,m in units(component):
        if p.startswith(tuple(exclude)):continue
        v=library.build(c)
        limit=width-600
        inventory.extend([(p,v,m)] if v.width_nm<=limit else [(p,s,m) for s in _slices(library,c,limit)])
    return inventory


def centre_lanes_used(component: Component,width: float,exclude: tuple[str,...],
                      band_span: tuple[float,float],lanes: list[tuple],compact_registers: bool=False) -> tuple[int,int]:
    """Lane rows a centre frame fills and the unsliced decoder drivers it
    leaves outside the band, from a placement in a scratch layout."""
    layout=new_layout();library=StandardLibrary(layout,compact_registers=compact_registers)
    inventory=_corner_inventory(library,component,width,exclude)
    inventory.sort(key=lambda item:item[1].standard_family!='DFF')
    prepared,_,_=_place_centre(layout,inventory,{p:c for p,c,_ in units(component)},width,band_span,lanes,300.,
                              local_logic=compact_registers)
    ys=[lane[0] for lane in lanes]
    used=max((ys.index(y)+1 for _,_,_,_,y,row,_ in prepared if row<0),default=0)
    lane=[p for p,_,_,_,_,row,_ in prepared if row<0]
    decoder={p.split('/')[0] for p,_,m,_,_,_,_ in prepared if m.get('Z','').startswith('A_dff') or m.get('Z')=='wl_en'}
    return used,sum(1 for p in set(lane) if p.split('/')[0] in decoder and lane.count(p)==1)


LANE_TAP_SPAN=8      # lane gates further apart in source order than this get input taps
LANE_TAP_REACH=15000.  # ... as do lane gates placed further than this (nm) from their lane driver
WELL_MIN=200.        # Well.4: lane well overlap with a neighbouring row's decoder well
WELL_SPACE=225.      # Well.2: lane well gap to a neighbouring row's decoder well
ACCESS_GAP=300.      # lane pin access (M4 landing, M5 above) or tap wire to a decoder conductor edge
BAND_FILL=1.0        # cell area / band area above which optional gates move to the lanes
CENTRE_TALL=False    # route centre frames below 80 um with the tall-frame order (long nets first)
HISTORY_COST=10      # centre routing: per-step cost for other nets on a blocked net's path
ESCAPE_REACH=3000    # macro input escape stacks slide at most this far from their access
ESCAPE_RADIUS=8000   # ... or stand alone, routed to the access, at most this far away
ESCAPE_LANDING={4:145.,5:145.,6:145.,7:400.,8:405.,9:805.}
ESCAPE_SPACE={4:140.,5:140.,6:140.,7:400.,8:400.,9:800.}
RIPUP_LIMIT=400      # centre rip-ups per attempt before a restart with new priority
HEAL_ROUNDS=3        # centre reroutes of nets whose same-net gaps cannot be filled
HISTORY_RADIUS=(6,6,6,10,10,16)  # M4..M9 grid cells, about each layer's clearance
ANTENNA_LIMIT=250.   # centre routing: lower metal per gate area (Calibre 300, less cell-internal metal)
ANTENNA_ROUNDS=3     # centre reroutes of nets whose gate inputs exceed it
ANTENNA_LIFT=10.     # M4-M7 step cost factor for those reroutes


def _port_loads(component: Component) -> dict[str,tuple[float,bool]]:
    """Port -> (gate area in um2, joined to a diffusion) over the flattened devices."""
    loads=defaultdict(lambda:[0.,False])
    def walk(c,nets):
        for d in c.devices:
            g,sd=nets.get(d.nodes[1],d.nodes[1]),{nets.get(n,n) for n in (d.nodes[0],d.nodes[2])}
            loads[g][0]+=d.width_m*d.length_m*1e12
            for n in sd:loads[n][1]=True
        for child in c.children:
            walk(child.component,{port:nets.get(n,n) for port,n in zip(child.component.ports,child.nodes)})
    walk(component,{p:p for p in component.ports})
    return {p:tuple(loads[p]) for p in component.ports if p in loads}


def _antenna_exposed(router: ControlRouter,loads: dict) -> list[tuple[str,int]]:
    """(net, layer index) pairs whose gate inputs would see more lower metal
    than ANTENNA_LIMIT times their gate area, with no diffusion joined, when
    that routing layer (M5..M9) is etched: Calibre's cumulative check on the
    router's own metal (GRID x width per grid point). loads maps (net, access
    in nm) to (gate area, diffusion) of the pins and taps there."""
    pins=defaultdict(dict)
    for (net,pt),(gate,diode) in loads.items():
        q=router.point(*pt);g,d=pins[net].get(q,(0.,False));pins[net][q]=(g+gate,d or diode)
    edges=defaultdict(list)
    for net,a,b in router.paths:edges[net].append((a,b))
    for net,x,y,up in router.vias:edges[net].append(((x,y,up-1),(x,y,up)))
    exposed=[]
    for net,at in pins.items():
        if not any(g for g,d in at.values() if not d):continue
        for level in range(1,len(router.layers)):
            parent={}
            def find(p):
                parent.setdefault(p,p)
                while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
                return p
            for a,b in edges[net]:
                for q in (a,b):
                    if q[2]<=level:find(q)
                if a[2]<=level and b[2]<=level:parent[find(a)]=find(b)
            for q in at:
                if q[2]<=level:find(q)
            area=defaultdict(float);gate=defaultdict(float);joined=set()
            for q in list(parent):
                if q[2]<level:area[find(q)]+=router.grid*router.widths[q[2]]/1e6
            for q,(g,d) in at.items():
                if q[2]>level:continue
                root=find(q);gate[root]+=g
                if d:joined.add(root)
            if any(g and r not in joined and area[r]>ANTENNA_LIMIT*g for r,g in gate.items()):
                exposed.append((net,level));break
    return exposed


def _on_grid(box: pya.Box) -> pya.Box:
    """box grown outward to the 2.5 nm (5 dbu) manufacturing grid; width-check
    edge pairs end at projected, off-grid points."""
    return pya.Box(box.left//5*5,box.bottom//5*5,-(-box.right//5)*5,-(-box.top//5)*5)


def _neck_fill(have: pya.Region,pair: Any,minw: int,space: int) -> Optional[pya.Box]:
    """The neck of a width-check edge pair widened symmetrically to minw
    (dbu, on the 5 dbu grid), or None if it is not axis-aligned or would come
    within space of metal not joined to it."""
    box=pair.bbox();e=pair.first
    if e.dy()==0:grow=(0,-(-(minw-box.height())//10)*5)
    elif e.dx()==0:grow=(-(-(minw-box.width())//10)*5,0)
    else:return None
    neck=_on_grid(box.enlarged(*grow))
    own=pya.Region([pg for pg in have.each() if pg.bbox().overlaps(box) and not (pya.Region(pg)&pya.Region(box.enlarged(1,1))).is_empty()])
    return neck if (pya.Region(neck).sized(space)&(have-own)).is_empty() else None


def _bridge_owned(b: Builder,router: ControlRouter,obstacles: list) -> None:
    """Close same-net notches between routes and fixed conductors they own.

    Owned bank conductors sit off the routing grid, and their owner may route
    one grid row away without touching them: a gap below the layer spacing
    that no other net can occupy. Fill each such gap, never within spacing of
    foreign metal (the controller's other shapes or other bank conductors).
    """
    drawn=defaultdict(pya.Region)
    for net,a,z in router.paths:
        h=router.widths[a[2]]/2;layer=router.layers[a[2]]
        drawn[net,layer].insert(pya.Box(u(min(a[0],z[0])*GRID-h),u(min(a[1],z[1])*GRID-h),
                                        u(max(a[0],z[0])*GRID+h),u(max(a[1],z[1])*GRID+h)))
    for net,gx,gy,upper in router.vias:
        for k in (upper-1,upper):
            h=router.widths[k]/2;drawn[net,router.layers[k]].insert(pya.Box(u(gx*GRID-h),u(gy*GRID-h),u(gx*GRID+h),u(gy*GRID+h)))
    for layer in ('m4','m5','m6'):
        boxes=[(owner,pya.Box(u(x0),u(y0),u(x1),u(y1))) for l,x0,y0,x1,y1,owner in obstacles if l==layer]
        if not any(owner for owner,_ in boxes):continue
        have=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
        for net in sorted({owner for owner,_ in boxes if owner}):
            own=drawn.get((net,layer),pya.Region()).merged()
            fixed=pya.Region([box for owner,box in boxes if owner==net]).merged()
            if own.is_empty() or (own&fixed.sized(u(140.))).is_empty():continue
            union=(own+fixed).merged()
            fill=(union.sized(u(70.)).sized(-u(70.))-union)&fixed.sized(u(140.))
            if fill.is_empty():continue
            foreign=have.not_interacting(own)+pya.Region([box for owner,box in boxes if owner!=net])
            if not (fill.sized(u(140.)-1)&foreign).is_empty():
                raise ValueError(f'{net}: {layer} notch at a bank conductor crowds foreign metal')
            b.cell.shapes(b.li(layer)).insert(fill.merged())
            have=(have+fill).merged()


def _antenna(path: str,m: dict) -> bool:
    return any(m[p] in ANTENNA_INPUTS for p in m if p not in ('VDD','VSS','Z')) or path=='Xpre_unbuf'


def _place_centre(layout,inventory,sources,width,band_span,lanes,x0,local_logic: bool=False):
    """Fill the full-width band, then decoder lanes, keeping source order.

    Band rows use the controller's 1500/1800 nm rows; DFFs, transmission
    gates, NOR gates and antenna-tapped receivers must fit there, followed by
    the groups that drive decoder nets while space remains. Remaining gates are
    rebuilt on 1365 nm decoder rows and sliced to the widest lane stretch
    clear of decoder conductors; a lane width is the limit of its gates' right
    edges, and gates slide right until their pin accesses clear the
    conductors crossing the row. Returns placements (row -1
    marks a lane gate), the band rows and the lane inputs that get antenna
    taps, keyed by (path, x, y).
    """
    y0,y1=band_span
    order=list(dict.fromkeys(p for p,_,_ in inventory))
    entries={p:[(v,m) for q,v,m in inventory if q==p] for p in order}
    lane_library=StandardLibrary(layout,1365.,compact_registers=local_logic)
    def lane_ready(path):
        try:lane_library.build(sources[path])
        except ValueError:return False
        return True
    fixed=[p for p in order if entries[p][0][0].standard_family in ('DFF','TG') or _antenna(p,entries[p][0][1])
           or not lane_ready(p)]
    # Groups driving decoder nets (address buffers, wl_en) take the bottom
    # rows, beside the block-edge pins of their outputs; the DFF rows above
    # them then carry no decoder net across. Members too wide for a lane
    # (an address buffer's first stage) stay with their group.
    decoder=({p.split('/')[0] for p in order if entries[p][0][1].get('Z','').startswith('A_dff')
              or entries[p][0][1].get('Z')=='wl_en'})
    first=[p for p in order if p.split('/')[0] in decoder and entries[p][0][0].standard_family not in ('DFF','TG')]
    def fill(order):
        prepared=[];rows=[];state=dict(cursor=x0,y=y0,height=0.,vdd=0.,row=0)
        done=0;budget=BAND_FILL*width*(y1-y0);used=0.
        for index,path in enumerate(order):
            saved=dict(state);mark=len(prepared);saved_rows=len(rows)
            area=sum(v.width_nm*v.height_nm for v,_ in entries[path])
            if path not in fixed and used+area>budget:break
            used+=area
            fits=True
            for v,m in entries[path]:
                tap=500. if _antenna(path,m) else 0.
                dff=v.standard_family=='DFF'
                family_break=prepared and (prepared[-1][1].standard_family=='DFF')!=dff
                if (state['cursor']+v.width_nm+tap>width-300 and state['cursor']>x0) or family_break:
                    rows.append((state['y'],state['height'],state['vdd']))
                    gap=((100 if dff else 25) if state['height']==1900 else
                         (100 if dff else 125) if state['height']==1800 else 125 if dff else 0)
                    state.update(y=state['y']+state['height']+gap,cursor=x0,height=0.,vdd=0.,row=state['row']+1)
                if state['y']+v.height_nm>y1-1000 or state['cursor']+v.width_nm+tap>width-300:
                    fits=False;break
                x=state['cursor'];prepared.append((path,v,m,x,state['y'],state['row'],False))
                following=entries[order[index+1]][0][0] if index+1<len(order) else None
                nxt=following is not None and following.standard_family=='DFF'
                state['cursor']=x+v.width_nm+((190 if nxt else 290) if dff else 30)+tap
                state['height']=max(state['height'],v.height_nm);state['vdd']=max(state['vdd'],v.pins['VDD'][1])
            if not fits:
                if path in fixed:raise ValueError(f'{path} must sit in the band, which is full')
                del prepared[mark:];del rows[saved_rows:];state=saved;break
            done=index+1
        if state['height']:rows.append((state['y'],state['height'],state['vdd']))
        if any(p in fixed for p in order[done:]):raise ValueError('Band cells left out of the full band')
        return prepared,rows,done
    def fill_band():
        rest=[p for p in entries if p not in fixed and p not in first]
        try:
            order=first+[p for p in fixed if p not in first]+rest;prepared,rows,done=fill(order)
        except ValueError:
            # Large decoder drivers leave the fixed cells no room: place
            # those first and the drivers while space remains.
            order=fixed+[p for p in first if p not in fixed]+rest;prepared,rows,done=fill(order)
        return order,prepared,rows,done
    order,prepared,rows,done=fill_band()
    if local_logic and done<len(order):
        # NORs must stay in the band. Keep their small combinational partners
        # there too, rather than stretching a local ready/enable net through
        # the entire decoder. Only the remaining buffer/delay cells spill.
        fixed=set(fixed)|{p for p in order if entries[p][0][0].standard_family!='INV' or '/' not in p}
        parents={p.rsplit('/',1)[0] for p in fixed if '/' in p}
        parents.update(p.rsplit('/',1)[0] for p in order
                       if entries[p][0][1].get('Z') in {'clk_buf','PRE','w_en','s_en','sa_iso'} and '/' in p)
        fixed=[p for p in order if p in fixed or ('/' in p and p.rsplit('/',1)[0] in parents)]
        order,prepared,rows,done=fill_band()
    # Lane inputs not driven by a lane gate close in source order (block
    # outputs such as DFF Q, bank nets, far lane gates) usually arrive over
    # M7/M8: each gets a diffusion antenna tap.
    lane=order[done:];block=list(prepared)
    if lane and not lanes:raise ValueError('Controller gates exceed a frame without decoder lanes')
    driver={entries[p][0][1]['Z']:p for p in lane if 'Z' in entries[p][0][1]}
    planned={}
    for k,path in enumerate(lane):
        m=entries[path][0][1]
        planned[path]=[p for p in m if p not in ('VDD','VSS','Z') and m[p] not in ('VDD','VSS')
                       and (m[p] not in driver or abs(lane.index(driver[m[p]])-k)>LANE_TAP_SPAN)]
    def clear(v,x,keep,tapped):
        # The gate's wells (with taps and fill) end WELL_MIN past or WELL_SPACE
        # before a neighbouring row's decoder well: anything between is a well
        # neck or a well spacing error.
        end=x+v.width_nm+500*len(tapped)+30
        if any(kl=='well' and k0-WELL_SPACE<end<k0+WELL_MIN for kl,k0,_ in keep):return False
        # Pin accesses (M4 landing and the M5 reserved above) and M4 tap wires
        # keep ACCESS_GAP from the decoder conductors crossing the lane row.
        spans=[]
        for p,(px,_,l) in v.pins.items():
            if p in ('VDD','VSS'):continue
            tx=round((x+px)/GRID)*GRID;spans+=[('m4',tx,tx),('m5',tx,tx)]
            if l=='m4':spans.append(('m4',x,x+v.width_nm))
        for i,p in enumerate(tapped):
            tx=round((x+v.pins[p][0])/GRID)*GRID;tapx=x+v.width_nm+500*i+250
            spans.append(('m4',min(tx,tapx),max(tx,tapx)))
        return not any(kl==l and a-ACCESS_GAP<k1 and b+ACCESS_GAP>k0 for l,a,b in spans for kl,k0,k1 in keep)
    def spans(lane):
        # Lengths of the lane stretches clear of decoder M4/M5 conductors.
        edges=sorted((k0-ACCESS_GAP,k1+ACCESS_GAP) for kl,k0,k1 in lane[2] if kl in ('m4','m5'))
        out,start=[],x0
        for a,b in edges:
            out.append(min(a,lane[1])-start);start=max(start,b)
        return out+[lane[1]-start]
    # Slices fit the median lane's widest clear stretch, so most rows hold them.
    clear_widths=sorted(max(spans(lane)) for lane in lanes)
    widest=(clear_widths[len(clear_widths)//2] if clear_widths else 0.)+x0
    def place(forced):
        prepared=list(block);lane_taps={};queue=iter(lanes);current=None;cursor=x0
        for path in lane:
            c=sources[path];v=lane_library.build(c)
            # Lower accesses take nearer taps, so the M4 tap wires nest without crossing.
            names=[k for k in v.pins if k not in ('VDD','VSS','Z')]
            taps=sorted((p for p in names if (p in planned[path] or (path,p) in forced) and v.pins[p][2]=='m3'),
                        key=names.index);tap=500.*len(taps)
            views=[v] if v.width_nm+tap<=widest-x0 else _slices(lane_library,c,widest-x0)
            # Slices of one gate stack alone at one x on consecutive rows, so their
            # straps meet no other gate's access. Their inputs need no tap: the
            # stacked gate area keeps the antenna ratio low, and a tap wire would
            # cross the output strap.
            stacked=len(views)>1
            if stacked or views[0].width_nm+tap>widest-x0:taps=[];tap=0.
            if stacked and cursor>x0:cursor=float('inf')
            stack_x=None
            for v in views:
                while True:
                    # Slide right past decoder trunks; a stack keeps its first
                    # slice's x where that row allows it.
                    if current is not None and cursor+v.width_nm+tap<=current[1]:
                        if clear(v,cursor,current[2],taps):break
                        cursor+=GRID;continue
                    current=next(queue,None);cursor=x0 if stack_x is None else stack_x
                    if current is None:raise ValueError('Controller gates exceed the spare decoder lanes')
                m=entries[path][0][1]
                prepared.append((path,v,m,cursor,current[0],-1,False))
                if taps:lane_taps[path,cursor,current[0]]=tuple(taps)
                if stacked and stack_x is None:stack_x=cursor
                cursor=float('inf') if stacked else cursor+v.width_nm+30+tap
        return prepared,lane_taps
    # A lane driver placed far away (across the band, past a stack) needs the
    # tap too: place again with those taps until no new one is needed.
    forced=set()
    while True:
        prepared,lane_taps=place(forced)
        at=defaultdict(list)
        for path,_,_,_,y,_,_ in prepared[len(block):]:at[path].append(y)
        far={(path,p) for path,ys in at.items() if len(ys)==1 for p,n in entries[path][0][1].items()
             if p not in ('VDD','VSS','Z') and p not in planned[path] and n in driver
             and abs(at[driver[n]][0]-ys[0])>LANE_TAP_REACH}
        if far<=forced:break
        forced|=far
    return prepared,rows,lane_taps


def _route_centre(router: ControlRouter,order: list[str],routed: dict,priority: list[str],local: list[str],
                  tall: bool,clock_state: Any,history: dict,attempt: int,total: int,loads: dict) -> None:
    """Route a centre frame in order with local rip-up and reroute.

    A net that cannot connect is cut off by nets routed before it: the
    cheapest path through their (but no fixed) metal names them. They are
    removed, that path costs every other net HISTORY_COST more per step, and
    the blocked net routes before them again. A net that would rip a net
    which ripped it before looks for a path past other nets first, so two
    nets contending for one passage do not take turns indefinitely.
    Afterwards same-net near-misses are healed; a net whose gap cannot be
    filled reroutes with that spot costing it more, up to HEAL_ROUNDS times.
    A net whose gate inputs then exceed the antenna limit reroutes with
    M4-M7 steps costing ANTENNA_LIFT times more, up to ANTENNA_ROUNDS times.
    Raises the RoutingError when no routed net encloses the failure, after
    RIPUP_LIMIT rip-ups or with gaps or antenna exposure left, so the caller
    restarts with a new priority.
    """
    from collections import deque
    import time
    from .router import RoutingError
    queue=deque(order);done=set();ripped=Counter();ripped_by=defaultdict(set)
    state=dict(base=None,ripups=0)
    def snapshot():
        return dict(router.owner),{p:set(n) for p,n in router.clearance.items()}
    if clock_state is not None:state['base']=snapshot();done.add('clk_buf')
    def drain():
        base=state['base']
        while queue:
            net=queue.popleft()
            if net in done:continue
            ps=list(dict.fromkeys(router.point(*p) for p in routed[net]))
            limit=router.search_limit
            if net=='clk_buf' or net in priority or (tall and net not in local):router.search_limit=None
            try:
                router.connect(net,ps[0],ps[1:])
            except RoutingError as error:
                router.search_limit=limit
                if base is None or state['ripups']>=RIPUP_LIMIT:raise
                started=time.time()
                culprits,path=router.culprits(net,ps[0],error.target,*base,ripped=ripped)
                culprits&=done
                if culprits&ripped_by[net]:
                    others,detour=router.culprits(net,ps[0],error.target,*base,ripped=ripped,
                                                  avoid=frozenset(culprits&ripped_by[net]))
                    if others&done:culprits,path=others&done,detour
                if not culprits:raise
                for n in culprits:ripped_by[n].add(net)
                for n in [net,*culprits]:
                    router.rip(n,*base);done.discard(n)
                ripped.update(culprits)
                for x,y,l in path:
                    for cell in ((x,y,l),(x+1,y,l),(x-1,y,l),(x,y+1,l),(x,y-1,l)):
                        costs=history.setdefault(cell,{});costs[net]=costs.get(net,0)+HISTORY_COST
                queue.extendleft(reversed([net,*sorted(culprits,key=order.index)]))
                state['ripups']+=1
                print('TIME_CONTROL rip-up',state['ripups'],net,'past',sorted(culprits),
                      f'({time.time()-started:.0f} s search)',flush=True)
                continue
            router.search_limit=limit;done.add(net)
            if net=='clk_buf':base=state['base']=snapshot()
            if len(done)%100==0:
                print('TIME_CONTROL routing attempt',attempt,'net',len(done),'/',total,'rip-ups',state['ripups'],flush=True)
    heals=antennas=0
    while True:
        drain()
        gaps=router.heal()
        if not gaps:
            exposed=_antenna_exposed(router,loads)
            if not exposed:return
            nets=sorted({n for n,_ in exposed},key=order.index)
            print('TIME_CONTROL antenna reroute:',[(n,router.layers[l]) for n,l in exposed],flush=True)
            if antennas==ANTENNA_ROUNDS:raise RoutingError(nets[0],None,set())
            antennas+=1
            for net in nets:
                router.lift[net]=router.lift.get(net,1.)*ANTENNA_LIFT
                router.rip(net,*state['base']);done.discard(net);queue.append(net)
            continue
        print('TIME_CONTROL same-net gaps to reroute:',sorted({n for n,_,_ in gaps}),flush=True)
        if heals==HEAL_ROUNDS:raise RoutingError(gaps[0][0],gaps[0][1],set())
        heals+=1
        # The gap's neighbourhood costs every net more ('' is no net).
        for net,p,q in gaps:
            for x,y,l in (p,q):
                for dx in range(-2,3):
                    for dy in range(-2,3):
                        costs=history.setdefault((x+dx,y+dy,l),{});costs['']=costs.get('',0)+HISTORY_COST
        for net in sorted({n for n,_,_ in gaps},key=order.index):
            router.rip(net,*state['base']);done.discard(net);queue.append(net)


def make_control(layout: pya.Layout,component: Component,width: float=32500.,exclude: tuple[str,...]=(),
                 pin_plan: Any=None,corner: bool=False,min_height: float=0.,
                 band_span: Optional[tuple[float,float]]=None,lanes: Optional[list[tuple]]=None,
                 profile: Optional[list[tuple[float,float,float]]]=None,
                 obstacles: Optional[list[tuple[str,float,float,float,float,Optional[str]]]]=None,
                 escapes: tuple[str,...]=(),compact_registers: bool=False) -> CellView:
    """Place and route TIME_CONTROL from its frozen source units.

    exclude lists unit-path prefixes that another physical block places, such
    as the data-register array. pin_plan maps each macro-facing net to an
    explicit (x, y, layer) access on the frame edge, replacing the default
    left-side M8 interface grid; M8 entries also reserve M9 above them.
    compact_registers keeps private DFF clocks on M4 and exposes M4/M5 pins
    so bank M6 crossings do not force the registers below the peripheral band.

    corner=True fits a fixed-width bank corner: cells start at the left edge,
    oversized gates split into parallel finger slices, and pin_plan may be a
    function of the final frame height. Archived RC decks retain a bottom band.

    A centre frame gives band_span=(y0, y1), the full-width block between the
    split arrays, and lanes=[(y, width, keepouts), ...], spare decoder rows in
    fill order with the (layer, x0, x1) decoder conductors crossing each.
    Lane gates use the decoder's 1365 nm rows and its M1 supply rails; DFFs,
    transmission gates and antenna-tapped receivers stay in the band. profile
    lists the left-aligned (y0, y1, width) routing region, and obstacles the
    fixed bank conductors inside it as (layer, x0, y0, x1, y1, owner). An
    owner permits access by that net; None blocks every net. pin_plan values
    may then be lists of accesses to one net.

    escapes lists macro input nets whose first access (a register D landing or
    a gate access) is lifted to M10 before routing. The router only sees the
    stack as the net's own fixed metal; the parent draws M10 to the macro edge
    from v.escapes (net -> frame-local point).
    """
    if lanes is not None and (not corner or band_span is None or profile is None):
        raise ValueError('Decoder lanes require a corner-style band frame and a routing profile')
    library=StandardLibrary(layout,compact_registers=compact_registers)
    b=Builder(layout,('TIME_CONTROL_CORNER_' if corner else 'TIME_CONTROL_LOGIC_' if exclude else 'TIME_CONTROL_STD_')+component.identity[:12])
    if corner:
        inventory=_corner_inventory(library,component,width,exclude)
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
    prepared=[];cursor=x0;y=band;row_height=0.;row_vdd=0.;row=0;rows=[];lane_taps={}
    if lanes is not None:
        sources={p:c for p,c,_ in units(component)}
        prepared,rows,lane_taps=_place_centre(layout,inventory,sources,width,band_span,lanes,x0,
                                            local_logic=compact_registers)
        inventory=[]
        y=rows[-1][0];row_height=rows[-1][1]
    for index,(path,v,m) in enumerate(inventory):
        tap_space=500. if corner and (any(m[p] in ANTENNA_INPUTS for p in m if p not in ('VDD','VSS','Z'))
                                     or path=='Xpre_unbuf') else 0.
        # An edge-pinned frame starts logic on a new row instead of sharing
        # the last register row.
        family_break=pin_plan is not None and prepared and (prepared[-1][1].standard_family=='DFF')!=(v.standard_family=='DFF')
        if (cursor+v.width_nm+tap_space>width-300 and cursor>x0) or family_break:
            if not row_height:raise ValueError('Source passive band exceeds the available corner height')
            rows.append((y,row_height,row_vdd))
            # Preserve clearance between the register's M5/M6 clock rails
            # as well as the wells at the next row's mixed logic boundary.
            gap=((100 if v.standard_family=='DFF' else 25) if row_height==1900 else
                 (100 if v.standard_family=='DFF' else 125) if row_height==1800 else 0)
            y+=row_height+gap;cursor=x0;row_height=0;row_vdd=0;row+=1
        x=cursor
        if x+v.width_nm+tap_space>width-300:raise ValueError(f'{path} ({v.width_nm} nm) does not fit the {width} nm controller row')
        prepared.append((path,v,m,x,y,row,False))
        next_dff=index+1<len(inventory) and inventory[index+1][1].standard_family=='DFF'
        cursor=x+v.width_nm+((190 if next_dff else 290) if v.standard_family=='DFF' else 30)
        cursor+=tap_space
        row_height=max(row_height,v.height_nm);row_vdd=max(row_vdd,v.pins['VDD'][1])
    if lanes is None:rows.append((y,row_height,row_vdd))
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
    # (net, access) -> [gate area um2, diffusion joined]: the centre antenna check.
    loads=defaultdict(lambda:[0.,False])
    lane_points=defaultdict(list)
    def wire(layer,x,y,xx,yy,w=70):b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    fixed_cuts=[]
    def lift(x,y,layer,target):
        for n in range(min(layer,target),max(layer,target)):
            if n>=7:b.sq(f'v{n}',x,y,800 if n==9 else 400)
            else:b.via(n,x,y)
            if n>=4:fixed_cuts.append((n,x,y))
            for k in (n,n+1):
                if k==1:b.rect('m1',x-70,y-35,x+70,y+35)
                elif k==2:
                    if n==2:b.rect('m2',x-35,y-70,x+35,y+70)
                    else:b.rect('m2',x-70,y-35,x+70,y+35)
                elif k==3:b.rect('m3',x-35,y-70,x+35,y+70)
                else:b.sq(f'm{k}',x,y,400 if k==7 else 405 if k==8 else 805 if k>=9 else 145)
    escape_access={};escape_points={};escape_modes={}
    def escape(net,x,y,layer):
        # Macro input: its first access continues to M10 (see the docstring)
        # once every fixed access is drawn.
        if net in escapes and net not in escape_access:escape_access[net]=(x,y,layer)
    # A lane gate's wells and rails continue to the next gate in its row,
    # which may have slid right past a decoder trunk. Wells also start at the
    # lane's left end, so neighbouring rows' wells never meet in a neck there.
    lane_cells=sorted((y,x) for _,_,_,x,y,row,_ in prepared if row<0)
    lane_next={(ya,xa):xb for (ya,xa),(yb,xb) in zip(lane_cells,lane_cells[1:]) if ya==yb}
    lane_first={y:x for y,x in reversed(lane_cells)}
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
        if row<0:
            # Decoder-row gates: the gate's own well split and the decoder's rail phases.
            end=max(lane_next.get((y,x),0.),x+v.width_nm+30)
            if lane_first[y]==x and x>x0:
                b.rect('pwell',x0,y,x,y+v.well_split);b.rect('nwell',x0,y+v.well_split,x,y+v.height_nm)
            b.rect('pwell',x+v.width_nm,y,end,y+v.well_split)
            b.rect('nwell',x+v.width_nm,y+v.well_split,end,y+v.height_nm)
            for phase in (100,v.height_nm-100):b.rect('m1',x+v.width_nm,y+phase-35,end,y+phase+35)
        elif not passive and v.standard_family!='DFF':
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
        port_loads=_port_loads(v.component)
        for p,(px,yy,l) in v.pins.items():
            net=m[p];px+=x;yy+=y;tap=False
            if p in ('VDD','VSS'):
                if row<0:continue  # the decoder's M1 rails supply lane gates
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
            if net in ('VDD','VSS') and not passive and v.standard_family!='DFF':
                # A constant input ties to its own gate's rail on M3. Routed as
                # part of the supply net it hangs on an upper-metal branch with
                # no diffusion until the supply joins: an antenna the supply
                # rails never protect (128x128 M8 violation, 2026-10-17).
                if l!='m3':lift(px,yy,int(l[1:]),3)
                if row<0:
                    _,sy,sl=v.pins[net];sy+=y
                    wire('m3',px,yy,px,sy);lift(px,sy,int(sl[1:]),3)
                    # Lane rails join the supply only through the decoder,
                    # after placement: no controller-level ownership check.
                else:
                    sy=rows[row][0]+(90 if net=='VSS' else rows[row][2])
                    wire('m3',px,yy,px,sy);lift(px,sy,2,3)
                    ownership.append((net,(px,yy,'m3')))
                continue
            if v.standard_family=='DFF' and l=='m7':
                tx=(ceil(px/GRID) if p=='CLK' else round(px/GRID))*GRID;ty=round(yy/GRID)*GRID
                wire('m7',px,yy,tx,yy,400);wire('m7',tx,yy,tx,ty,400)
                lift(tx,ty,7,8);point=(tx,ty,'m8');ownership.append((net,(px,yy,'m7')))
                if p=='D':escape(net,tx,ty,8)
            elif passive:
                tx=round(px/GRID)*GRID;ty=y+(5400 if p in ('IN','TOP') else 6300)
                wire('m6',px,yy,tx,yy,145);wire('m6',tx,yy,tx,ty,145);point=(tx,ty,'m6')
            elif l=='m3':
                names=[k for k in v.pins if k not in ('VDD','VSS','Z')]
                multi=any(q[2]=='m4' for q in v.pins.values())
                if v.standard_family=='TG':phase={'IN':100,'OUT':1400,'CTR_N':400,'CTR_P':1050}[p]
                elif row>=0:phase=v.height_nm-100 if p=='Z' else 1000 if multi else 300+350*names.index(p)
                # 1365 nm lane rows: >=300 nm between neighbouring accesses and
                # clear of the 420/820 nm input bars and folded-NAND straps.
                else:phase=v.height_nm-100 if p=='Z' else 1020 if multi else 350+300*names.index(p)
                ty=round((y+phase)/GRID)*GRID;tx=round(px/GRID)*GRID
                wire('m3',px,yy,px,ty)
                # The last input of a compact leaf is 152.5 nm from Z.
                # Snap on M4: an M3 jog toward either neighbouring pin can
                # violate spacing. Fixed-metal reservation sees this jog.
                if p!='Z' and 'Z' in v.pins and 0<x+v.pins['Z'][0]-px<190.:
                    lift(px,ty,3,4);wire('m4',px,ty,tx,ty,145)
                else:
                    wire('m3',px,ty,tx,ty);lift(tx,ty,3,4)
                point=(tx,ty,'m4');escape(net,tx,ty,4)
                tapped=lane_taps.get((path,x,y),())
                if row<0 and p in tapped:
                    # Lane input tap in the space reserved right of the gate.
                    from .control_passives import antenna_diode
                    x1=x+v.width_nm+500*tapped.index(p);tapx=x1+250;tapy=y+300
                    b.rect('pwell',x1,y,x1+530,y+v.well_split);b.rect('nwell',x1,y+v.well_split,x1+530,y+v.height_nm)
                    antenna_diode(b,tapx,tapy);lift(tapx,tapy,3,4)
                    wire('m4',tx,ty,tapx,ty,145);wire('m4',tapx,ty,tapx,tapy,145)
                    ownership.append((net,(tapx,tapy,'m4')));tap=True
                elif corner and (net in ANTENNA_INPUTS or (path=='Xpre_unbuf' and p=='A')):
                    # A reserved filler protects the observer before upper
                    # routing layers reach its far-end driver. Source MOS
                    # sizes and the far-end feedback path stay fixed.
                    from .control_passives import antenna_diode
                    tapx=x+v.width_nm+250;tapy=y+300
                    b.rect('pwell',x+v.width_nm,y,x+v.width_nm+530,y+640)
                    b.rect('nwell',x+v.width_nm,y+640,x+v.width_nm+530,y+1500)
                    antenna_diode(b,tapx,tapy);lift(tapx,tapy,3,4)
                    wire('m4',tx,ty,tapx,ty,145);wire('m4',tapx,ty,tapx,tapy,145)
                    ownership.append((net,(tapx,tapy,'m4')));tap=True
            else:
                tx=round(px/GRID)*GRID;ty=round(yy/GRID)*GRID
                wire(l,px,yy,tx,yy,145);wire(l,tx,yy,tx,ty,145);point=(tx,ty,l)
                if v.standard_family=='DFF' and p=='D':escape(net,tx,ty,int(l[1:]))
            routed[net].append(point);access.append((net,point));ownership.append((net,point))
            gate,diode=port_loads.get(p,(0.,False));loads[net,point][0]+=gate;loads[net,point][1]|=diode or tap
            if row<0:lane_points[path,x,v.width_nm].append((y,p,net,point,l))
    if lanes is not None:
        # Parallel slices of one gate stack in consecutive decoder rows at one
        # x. Straight M4 straps join their accesses; the router reaches each
        # stack through its access nearest the band.
        middle=sum(band_span)/2;stacks=defaultdict(list)
        for (path,x,w),items in lane_points.items():
            if any(l!='m3' for *_,l in items):continue
            stacks[path,x,w].extend(items)
        for (path,x,w),items in stacks.items():
            ys=sorted({y for y,*_ in items})
            runs=[[ys[0]]]
            for y in ys[1:]:
                if abs(y-runs[-1][-1]-1365)<1e-6:runs[-1].append(y)
                else:runs.append([y])
            for run in runs:
                if len(run)<2:continue
                for pin in {p for _,p,*_ in items}:
                    points=[(y,net,point) for y,p,net,point,_ in items if p==pin and y in run]
                    net=points[0][1];xs={pt[0] for _,_,pt in points}
                    if len(xs)!=1 or len(points)!=len(run):raise ValueError(f'{path}: stacked slices disagree on {pin}')
                    tys=[pt[1] for _,_,pt in points];wire('m4',xs.pop(),min(tys),points[0][2][0],max(tys),145)
                    keep=min(points,key=lambda t:abs(t[0]-middle))[2]
                    for _,_,pt in points:
                        if pt!=keep:
                            routed[net].remove(pt);access.remove((net,pt))
                            gate,diode=loads.pop((net,pt));loads[net,keep][0]+=gate;loads[net,keep][1]|=diode
    # Row supplies are below/above the devices on M2, with M3 edge trunks.
    pins={}
    supply_top={}
    for net in ('VSS','VDD'):
        ys=[yy+(90 if net=='VSS' else vdd) for yy,h,vdd in rows]
        if net=='VSS':ys.extend(p['y_nm']+6100 for p in passive_locations)
        sites=[]
        for yy,h,vdd in rows:
            edge=width-100;tx=vss_x if net=='VSS' else edge
            py=yy+(90 if net=='VSS' else vdd)
            wire('m2',vss_x,py,edge,py);lift(tx,py,2,3);sites.append((tx,py))
        for (x1,y1),(x2,y2) in zip(sites,sites[1:]):
            wire('m3',x2,y1,x2,y2)
        if net=='VSS':wire('m3',vss_x,min(ys),vss_x,max(ys))
        pins[net]=(*sites[0],'m3');supply_top[net]=sites[-1]
    base=band_span[0] if lanes is not None else 0.
    wire('m3',vss_x,pins['VSS'][1],vss_x,base+200);lift(vss_x,base+200,3,4);wire('m4',100,base+200,vss_x,base+200,145)
    pins['VSS']=(100,base+200,'m4')
    vdd_x,vdd_y=width-100.,height-(900. if corner else 200.)
    vdd_source_y=supply_top['VDD'][1]
    if lanes is not None:
        # The native M3/M2 rails already join every row. Access their lower
        # trunk beside the bank supply, avoiding a redundant tall metal route.
        vdd_x,vdd_y=width-100.,pin_plan['VDD'][1];vdd_source_y=pins['VDD'][1]
        if compact_registers:
            # The combined write-input row changes the bank rail height.
            # Keep the native M4 supply access clear of fixed signal M4/M5
            # before asking the router to join it to the planned M6 pin.
            fixed={layer:pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged() for layer in ('m4','m5')}
            def supply_clear(ty: float) -> bool:
                probe=pya.Region(pya.Box(u(vdd_x-212.5),u(ty-212.5),u(vdd_x+212.5),u(ty+212.5)))
                if any(limit>1 and x0<=vdd_x<=x1 and y0<=ty<=y1
                       for x0,y0,x1,y1,limit in blocked_boxes):return False
                if any(layer in ('m4','m5') and owner!='VDD'
                       and x0-212.5<vdd_x<x1+212.5 and y0-212.5<ty<y1+212.5
                       for layer,x0,y0,x1,y1,owner in obstacles or ()):return False
                return all((metal & probe).is_empty() for metal in fixed.values())
            candidates=(vdd_y+GRID*k*side for k in range(int(height//GRID)) for side in (1,-1))
            vdd_y=next((ty for ty in candidates if base+300<=ty<=band_span[1]-300 and supply_clear(ty)),None)
            if vdd_y is None:raise ValueError('No clear native VDD access beside the controller rows')
    for net,tx,ty in [('VSS',100.,base+200.),('VDD',vdd_x,vdd_y)]:
        if net=='VDD':
            wire('m3',tx,vdd_source_y,tx,ty);lift(tx,ty,3,4)
        point=(tx,ty,'m4');routed[net].append(point);access.append((net,point));ownership.append((net,point))
    # Registers expose their native high-metal D/Q access. Bank routing owns
    # the data/address connections; no redundant perimeter bus is inserted.
    external={p:routed[p][0] for p in signal_ports if p in routed}
    if pin_plan is not None:
        pin_nwell=pya.Region(b.cell.begin_shapes_rec(b.li('nwell'))).merged()
        for net,planned in pin_plan.items():
            points=planned if isinstance(planned,list) else [planned]
            # A multi-point net outside the controller source is a feed-through
            # (a decoder net joining the two array halves).
            if net!='VDD' and net!='VSS' and net not in routed and len(points)<2:raise ValueError(f'Planned controller pin has no source connection: {net}')
            external[net]=points[0]
            for point in points:
                b.sq(point[2],point[0],point[1],145 if point[2] in ('m5','m6') else 405)
                routed[net].append(point);access.append((net,point));ownership.append((net,point))
                # Planned pins lead to bank drivers or tapped decoder spines.
                loads[net,point][1]=True
                # Block-edge pins (including the 1 um bottom margin) carry the taps.
                edge=lanes is None or band_span[0]-1000<=point[1]<=band_span[1]
                if corner and edge and point[2]=='m5':
                    # Edge pins drive long decoder spines that reach gates only;
                    # an N+ tap at the pin keeps them antenna-safe whatever the route.
                    below=lanes is None or point[1]>sum(band_span)/2
                    px,dy=point[0],point[1]+(-350. if below else 350.)
                    # A shorter band can put its last logic row beside this
                    # tap. Keep the full 225 nm opposite-well clearance; use
                    # the other side of the pin if the inward tap is crowded.
                    def tap_clear(y: float) -> bool:
                        probe=pya.Region(pya.Box(u(px-355.),u(y-355.),u(px+355.),u(y+355.)))
                        return (pin_nwell & probe).is_empty()
                    if lanes is not None and not tap_clear(dy):
                        dy=point[1]+(350. if below else -350.)
                        if not tap_clear(dy):raise ValueError(f'No legal antenna-tap well clearance at {net} {point}')
                    b.rect('pwell',px-130,dy-130,px+130,dy+130);b.rect('nplus',px-105,dy-105,px+105,dy+105)
                    b.rect('active',px-75,dy-75,px+75,dy+75);b.cont(px,dy);b.rect('m1',px-35,dy-70,px+35,dy+70)
                    lift(px,dy,1,5);wire('m5',px,dy,px,point[1],145)
                    # The tap landing is off the routing grid: centre routes
                    # (its own net included) keep full spacing from it.
                    # Grid points within 285 nm (wire half-width, space, landing
                    # half-width) on M4/M5; the pin itself is 350 nm away.
                    if lanes is not None:blocked_boxes.append((ceil((px-284)/GRID)*GRID,ceil((dy-284)/GRID)*GRID,
                                                               floor((px+284)/GRID)*GRID,floor((dy+284)/GRID)*GRID,2))
                    ownership.extend((net,(px,dy,layer)) for layer in ('m4','m5'))
    interface=[] if pin_plan is not None else [p for p in component.ports if p.startswith('A_dff') or p in
               ('wl_en','s_en','w_en','PRE','sa_iso','rbl','rwl','pre_far','iso_far','sen_far','wen_far')]
    for i,net in enumerate(interface):
        point=(1200.+1800*(i%3),1200.+3600*(i//3),'m8');external[net]=point
        b.sq('m8',point[0],point[1],405);routed[net].append(point);access.append((net,point));ownership.append((net,point))
    for net,x in ([] if pin_plan is not None else [('VSS',1200.),('VDD',3000.)]):
        point=(x,23000.,'m8');external[net]=point
        b.sq('m8',*point[:2],405);routed[net].append(point);access.append((net,point));ownership.append((net,point))
    # Escape stacks are fixed metal: register landings first, then gate
    # accesses. A stack that would crowd another stack, an access or a bank
    # conductor slides along a stub on its access layer to the nearest clear
    # point (two neighbouring inputs at 256x256 overlapped on M9). Without a
    # clear stub it stands alone at the nearest clear point and becomes one
    # more access of its net for the router (the boxed-in clock input at 256).
    missing=set(escapes)-set(escape_access)
    if missing:raise ValueError(f'Macro inputs without an M10 escape access: {sorted(missing)}')
    have={k:pya.Region(b.cell.begin_shapes_rec(b.li(f'm{k}'))).merged() for k in range(4,10)}
    def area(x0,y0,x1,y1,grow):return pya.Box(u(min(x0,x1)-grow),u(min(y0,y1)-grow),u(max(x0,x1)+grow),u(max(y0,y1)+grow))
    for net,(x,y,base) in sorted(escape_access.items(),key=lambda t:-t[1][2]):
        near=area(x,y,x,y,ESCAPE_RADIUS+2000)
        local={k:have[k]&pya.Region(near) for k in range(4,10)}
        blocks=[o for o in obstacles or () if o[1]<near.right*layout.dbu*1000 and o[3]>near.left*layout.dbu*1000
                and o[2]<near.top*layout.dbu*1000 and o[4]>near.bottom*layout.dbu*1000]
        def crowded(k,x0,y0,x1,y1,half,own):
            space=ESCAPE_SPACE[k]
            if not (pya.Region(area(x0,y0,x1,y1,half+space))&(local[k]-own)).is_empty():return True
            return any(l==f'm{k}' and ox0<x1+half+space and ox1>x0-half-space and oy0<y1+half+space and oy1>y0-half-space
                       for l,ox0,oy0,ox1,oy1,_ in blocks)
        def clear(ex,ey,lowest):
            return (0<ex<width and 0<ey<height
                    and (profile is None or any(a<=ey<=z and ex<w for a,z,w in profile))
                    and not any(crowded(k,ex,ey,ex,ey,ESCAPE_LANDING[k]/2,pya.Region()) for k in range(lowest,10)))
        mine=local[base].interacting(pya.Region(area(x,y,x,y,2.5)))
        steps=sorted(range(-ESCAPE_REACH,ESCAPE_REACH+1,int(GRID)),key=abs)
        stub=next(((ex,ey) for ex,ey in dict.fromkeys(p for d in steps for p in ((x+d,y),(x,y+d)))
                   if not crowded(base,min(x,ex),min(y,ey),max(x,ex),max(y,ey),ESCAPE_LANDING[base]/2,mine)
                   and clear(ex,ey,base+1)),None)
        if stub is not None:
            ex,ey=stub;escape_modes[net]='on access' if stub==(x,y) else 'stub'
            if (ex,ey)!=(x,y):wire(f'm{base}',x,y,ex,ey,ESCAPE_LANDING[base])
        else:
            reach=int(ESCAPE_RADIUS/GRID)
            spots=sorted(((x+i*GRID,y+j*GRID) for i in range(-reach,reach+1) for j in range(-reach,reach+1)),
                         key=lambda q:(abs(q[0]-x)+abs(q[1]-y),q))
            ex,ey=next((q for q in spots if clear(*q,base)),(None,None))
            if ex is None:raise ValueError(f'No clear M10 escape for {net} within {ESCAPE_RADIUS} nm of {x},{y}')
            point=(ex,ey,f'm{base}');routed[net].append(point);access.append((net,point));escape_modes[net]='routed'
        lift(ex,ey,base,10);escape_points[net]=(ex,ey)
        have[base].insert(area(x,y,ex,ey,ESCAPE_LANDING[base]/2) if stub else area(ex,ey,ex,ey,ESCAPE_LANDING[base]/2))
        for k in range(base+1,10):have[k].insert(area(ex,ey,ex,ey,ESCAPE_LANDING[k]/2))
        ownership.extend((net,(ex,ey,f'm{k}')) for k in range(base,10))
    owned_bank=[o for o in obstacles or () if o[0]=='m6' and o[5] is not None]
    if owned_bank:
        # Fixed DFF clock rails exist before routing. A bank supply crossing
        # through one cannot be repaired by the router; reject it immediately.
        from .conductor_ownership import port_conductors
        # High-metal accesses identify DFF rails and escape stacks; the two
        # native supply accesses identify any fixed power metal. Avoid probing
        # every low-metal gate input in a large controller.
        probes=[(net,point) for net,point in ownership
                if int(point[2][1:])>=6 or (net in ('VDD','VSS') and point[2]=='m4')]
        fixed_points={str(i):point for i,(_,point) in enumerate(probes)}
        fixed_aliases={str(i):net.upper() for i,(net,_) in enumerate(probes)}
        fixed=port_conductors(b,fixed_points,fixed_aliases,minimum_metal=6)
        known=pya.Region()
        for net,layer,polygon in fixed:
            if layer!=21:continue
            native=pya.Region(polygon);known+=native
            for _,x0,y0,x1,y1,owner in owned_bank:
                if net!=owner.upper() and not (native&pya.Region(pya.Box(u(x0),u(y0),u(x1),u(y1)))).is_empty():
                    raise ValueError(f'Fixed controller {net} M6 overlaps bank {owner}; separate the cell placement')
        private=pya.Region(b.cell.begin_shapes_rec(b.li('m6')))-known
        for _,x0,y0,x1,y1,owner in owned_bank:
            if not (private&pya.Region(pya.Box(u(x0),u(y0),u(x1),u(y1)))).is_empty():
                raise ValueError(f'Private controller M6 overlaps bank {owner}; separate the cell placement')
    router=ControlRouter(b,width,height,upper_metal=9,profile=profile)
    for layer_index,layer in enumerate(router.layers):
        reg=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
        sites=[(n,pya.Point(u(pt[0]),u(pt[1]))) for n,pt in ownership if pt[2]==layer]
        # Fixed bank metal blocks every foreign net; known owners can join
        # their own conductors. The clearance is square: a wire
        # ending beside an obstacle corner extends half its width past the
        # last grid point.
        clearance=1202.5 if layer_index>=5 else 602.5 if layer_index>=3 else 212.5
        for o_layer,ox0,oy0,ox1,oy1,owner in obstacles or ():
            if o_layer!=layer:continue
            for xi in range(ceil((ox0-clearance)/GRID),floor((ox1+clearance)/GRID)+1):
                for yi in range(ceil((oy0-clearance)/GRID),floor((oy1+clearance)/GRID)+1):
                    dx=max(0.,ox0-xi*GRID,xi*GRID-ox1);dy=max(0.,oy0-yi*GRID,yi*GRID-oy1)
                    if dx<clearance and dy<clearance:
                        pt=(xi,yi,layer_index)
                        if owner is None:router.blocked.add(pt)
                        else:router.clearance.setdefault(pt,set()).add(owner)
        for pg in reg.each():
            bb=pg.bbox();scale=layout.dbu*1000
            owners={net for net,point in sites if pg.inside(point)}
            if len(owners)>1:raise ValueError(f'Shorted fixed conductor on {layer} at {[v*scale for v in (bb.left,bb.bottom,bb.right,bb.top)]}: {owners}')
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
    if pin_plan is not None:integration_ports=[n for n,p in pin_plan.items() if not isinstance(p,list) and p[2]=='m8']
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
    tall=height>80000. or (profile is not None and CENTRE_TALL)
    priority=[n for n in nets if n.startswith('A_dff')] if tall else []
    if profile is not None and not tall:
        # Centre frames: nets with planned edge pins take straight paths
        # first; local nets then claim their pin access before longer ones.
        priority=[n for n in nets if n in pin_plan and n not in ('VDD','VSS')]
    clock_state=None;history={}
    for attempt in range(32):
        router=ControlRouter(b,width,height,upper_metal=9,profile=profile);router.blocked=set(fixed_blocked)
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
        if profile is not None:
            router.history=history;router.exclude_cuts(fixed_cuts)
            router.fixed_points={router.point(*pt) for _,pt in access}
        local=[n for n in nets if n not in component.ports and max(p[0] for p in routed[n])-min(p[0] for p in routed[n])+max(p[1] for p in routed[n])-min(p[1] for p in routed[n])<6000]
        order=['clk_buf']+priority+[n for n in nets if n!='clk_buf' and n not in priority]
        if profile is not None and tall:
            # Preserve channels for long feedback and interface nets before
            # connecting the small local groups between neighboring gates.
            order=['clk_buf']+priority+[n for n in nets if n!='clk_buf' and n not in priority and n not in local]
            order.extend(n for n in nets if n!='clk_buf' and n not in priority and n in local)
        try:
            if profile is not None:
                _route_centre(router,order,routed,priority,local,tall,clock_state,history,attempt,len(nets),loads)
                break
            for i,net in enumerate(order):
                if net=='clk_buf' and clock_state is not None:continue
                ps=list(dict.fromkeys(router.point(*p) for p in routed[net]))
                limit=router.search_limit
                if net=='clk_buf' or net in priority:
                    router.search_limit=None
                router.connect(net,ps[0],ps[1:])
                router.search_limit=limit
                if net=='clk_buf' and tall:clock_state=deepcopy(router,{id(b):b})
                if i%100==0:print('TIME_CONTROL routing attempt',attempt,'net',i,'/',len(nets),flush=True)
            break
        except RoutingError as error:
            if error.net=='clk_buf' or (priority and priority[0]==error.net):raise
            if error.net in priority:priority.remove(error.net)
            priority.insert(0,error.net)
            if profile is not None:
                # Centre frames: other nets pay to pass the failed net's terminals.
                for point in routed[error.net]:
                    cx,cy,cl=router.point(*point)
                    for l in range(len(router.layers)):
                        r=HISTORY_RADIUS[l]
                        for dx in range(-r,r+1):
                            for dy in range(-r,r+1):
                                cell=history.setdefault((cx+dx,cy+dy,l),{})
                                cell[error.net]=cell.get(error.net,0)+HISTORY_COST
            print('Retrying with priority for',error.net,flush=True)
    else:raise ValueError('Controller routing did not converge after 32 deterministic retries')
    router.draw();pins.update(external)
    if obstacles:_bridge_owned(b,router,obstacles)
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
                            local=_on_grid((pya.Region(pg)&window).bbox())
                            # A merged region hides a new short from spacing
                            # checks. Keep the fill clear of every other wire.
                            foreign=have-pya.Region(pg)
                            if (pya.Region(local).sized(u(space))&foreign).is_empty():
                                b.cell.shapes(b.li(layer)).insert(local)
                have=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
            if not have.width_check(u(minw)).is_empty():
                # The piece's box crowds another wire (a same-net bridge between
                # two offset via landings, narrowed by the offset): widen only
                # the neck itself.
                for pair in have.width_check(u(minw)).each():
                    neck=_neck_fill(have,pair,u(minw),u(space))
                    if neck is not None:b.cell.shapes(b.li(layer)).insert(neck)
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
    v.escapes=escape_points;v.escape_modes=escape_modes
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
