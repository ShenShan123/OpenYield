"""Fold source-owned row logic into shelves with span-limited routing."""
from collections import defaultdict
from math import ceil

from . import bitcell as bc
from .cell_flow import leaves
from .lib import Builder


def ordered_leaves(source,specs,rows):
    """Only auxiliary logic belongs below the pitch-matched decoder strip."""
    for name in ('XRWL','XD_LATCH'):
        for path,leaf,_ in leaves(specs[name],path=name):
            yield path,leaf,source.instances[path]['pins']


def build_rows(library,source,specs,ports,shelf_columns=32):
    """Route ports above each shelf; reuse side tracks with disjoint spans.

    Local nets terminate in their own shelf. Only cross-shelf and boundary
    nets consume M5 side trunks. W/L/NF and all source hierarchy remain fixed.
    """
    width=max(32,shelf_columns)*bc.W
    shelves=[];cursor=0
    for path,leaf,mapping in ordered_leaves(source,specs,source.rows):
        view=library.build(leaf,'row')
        if not shelves or cursor+view.width_nm>width:
            shelves.append([]);cursor=0
        shelves[-1].append((path,view,mapping,cursor));cursor+=view.width_nm
    by_net=defaultdict(set)
    for r,shelf in enumerate(shelves):
        for _,view,mapping,_ in shelf:
            for p in view.pins:by_net[mapping[p]].add(r)
    global_nets={n for n,rs in by_net.items() if len(rs)>1 or n in ports}
    plans=[];y=0
    global_points=defaultdict(list)
    for r,shelf in enumerate(shelves):
        points=defaultdict(list)
        for _,view,mapping,x in shelf:
            for n,(px,py,layer) in view.pins.items():points[mapping[n]].append((x+px,y+py,layer))
        slots=[];tracks={}
        for net in sorted(points,key=lambda n:(n in global_nets,
                max(p[0] for p in points[n])-min(p[0] for p in points[n]),n)):
            lo=0 if net in global_nets else min(p[0] for p in points[net])
            hi=max(p[0] for p in points[net])
            slot=next((i for i,spans in enumerate(slots)
                       if all(hi+300<=a or lo>=b+300 for a,b in spans)),len(slots))
            if slot==len(slots):slots.append([])
            slots[slot].append((lo,hi));tracks[net]=y+(4+slot)*bc.H
            if net in global_nets:global_points[net].append(tracks[net])
        plans.append((shelf,y,points,tracks))
        y+=(5+len(slots))*bc.H
    # Boundary trunks reach the top so the bank can join them without routing
    # through cell interiors. Internal trunks occupy only their actual span.
    side_slots=[];side={}
    for net in sorted(global_points,key=lambda n:(n not in ports,n)):
        lo=400. if net in ports else min(global_points[net])
        hi=y if net in ports else max(global_points[net])
        slot=next((i for i,spans in enumerate(side_slots)
                   if all(hi+300<=a or lo>=b+300 for a,b in spans)),len(side_slots))
        if slot==len(side_slots):side_slots.append([])
        side_slots[slot].append((lo,hi));side[net]=(400+slot*300,lo,hi)
    margin=ceil((800+len(side_slots)*300)/bc.W)*bc.W
    cell=Builder(library.layout,'FOLDED_BANK_ROWS')
    terminals={};placed=[];paths=[];vias=[]
    def via(net,n,x,y):
        cell.via(n,x,y)
        for metal in (n,n+1):
            if metal<=3:cell.rect(f'm{metal}',x-35,y-70,x+35,y+70)
            else:cell.sq(f'm{metal}',x,y,145 if metal==5 else 140)
        vias.append((net,x,y,n))
    def wire(net,layer,x0,y0,x1,y1,width):
        h=width/2
        cell.rect(layer,min(x0,x1)-h,min(y0,y1)-h,max(x0,x1)+h,max(y0,y1)+h)
        paths.append((net,(x0,y0,layer),(x1,y1,layer)))
    for shelf,sy,points,tracks in plans:
        for path,view,mapping,x in shelf:
            x+=margin
            cell.place(view.builder,x,sy)
            placed.append(dict(instance=path,cell=view.builder.name,x_nm=x,y_nm=sy,
                               width_nm=view.width_nm,height_nm=view.height_nm,connections=mapping))
            for n,(px,py,layer) in view.device_pins.items():terminals[path+'/'+n]=(x+px,sy+py,layer)
        for net,pts in points.items():
            track=tracks[net]
            for px,py,layer in pts:
                x=px+margin
                if layer!='m2':raise ValueError('Row leaf interface must be M2')
                for n in (2,3,4):via(net,n,x,py)
                wire(net,'m5',x,py,x,track,140)
                for n in (3,4):via(net,n,x,track)
            lo=side[net][0] if net in side else min(p[0] for p in pts)+margin
            hi=max(p[0] for p in pts)+margin
            wire(net,'m3',lo,track,hi,track,70)
            if net in side:
                for n in (3,4):via(net,n,lo,track)
    pins={}
    for net,(x,lo,hi) in side.items():
        wire(net,'m5',x,lo,x,hi,140)
        if net in ports:pins[net]=(x,y,'m5')
    if set(pins)!=set(ports):raise ValueError('Missing row-logic boundary pins')
    width=max(x+v.width_nm for shelf in shelves for _,v,_,x in shelf)+margin+400
    cell.rect('bnd',0,0,width,y+400)
    return cell,dict(width_nm=width,height_nm=y+400,pins=pins,device_pins=terminals,
        placements=placed,paths=paths,vias=vias,shelf_count=len(shelves),side_track_count=len(side_slots),
        margin_nm=margin)
