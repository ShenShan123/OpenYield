"""Identify public-port metal conductors without treating MOS channels as wires.

This supplies fixed, same-net routing access through verified hierarchical
cells. It does not replace transistor extraction or LVS.
"""
from collections import defaultdict
from typing import Optional
from .lib import pya,u


def port_conductors(builder,ports,aliases,require_connected=False,minimum_metal: int = 7,
                    exclude_bbox_nm: Optional[tuple] = None):
    ly=builder.ly;cell=builder.cell;scale=ly.dbu*1000;step=u(1000)
    excluded=pya.Region(pya.Box(*[u(v) for v in exclude_bbox_nm])) if exclude_bbox_nm else pya.Region()
    nodes=[];parents=[];layers=defaultdict(list);bins=defaultdict(lambda:defaultdict(list))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for layer in [9]+list(range(11,30,2)):
        for pg in (pya.Region(cell.begin_shapes_rec(ly.layer(layer,0)))-excluded).merged().each():
            index=len(nodes);box=pg.bbox();nodes.append((layer,pg,box));parents.append(index);layers[layer].append(index)
            for x in range(box.left//step,box.right//step+1):
                for y in range(box.bottom//step,box.top//step+1):bins[layer][x,y].append(index)
    for cut,lower,upper in [(10,9,11)]+[(12+2*i,11+2*i,13+2*i) for i in range(9)]:
        for via in (pya.Region(cell.begin_shapes_rec(ly.layer(cut,0)))-excluded).merged().each():
            box=via.bbox();vr=pya.Region(via);hits=[]
            for layer in (lower,upper):
                candidates=set()
                for x in range(box.left//step,box.right//step+1):
                    for y in range(box.bottom//step,box.top//step+1):candidates.update(bins[layer][x,y])
                hits.extend(i for i in candidates if nodes[i][2].overlaps(box) and not (pya.Region(nodes[i][1]) & vr).is_empty())
            for i in hits[1:]:parents[root(i)]=root(hits[0])
    owners={}
    for port,net in aliases.items():
        x,y,metal=ports[port];layer=9+2*int(metal[1:]);point=pya.Point(u(x),u(y))
        found=[root(i) for i in layers[layer] if nodes[i][1].inside(point)]
        if len(set(found))!=1:raise ValueError(f'No unique physical conductor for {port}')
        key=found[0]
        if key in owners and owners[key]!=net:raise ValueError(f'Public circuit nets are shorted: {port} ({owners[key]}, {net})')
        owners[key]=net
    if require_connected:
        components=defaultdict(set)
        for key,net in owners.items():components[net].add(key)
        disconnected=[net for net,keys in components.items() if len(keys)>1]
        if disconnected:raise ValueError(f'Public circuit nets are disconnected: {sorted(disconnected)}')
    return [(owners[root(i)],layer,pg) for i,(layer,pg,box) in enumerate(nodes)
            if layer>=9+2*minimum_metal and root(i) in owners]
