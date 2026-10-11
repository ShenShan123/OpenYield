"""Reusable leaf-subcircuit geometry and height-aware compact placement.

Geometry is keyed by the actual compiler MOS dimensions and finger counts.
Electrical nets and instance names stay in the caller's terminal map; no
OpenRAM schematic, sizing policy or technology-specific cell is imported.
"""
from collections import defaultdict
from math import ceil

from . import controller
from .lib import Builder
from .router import GRID, RoutingError


COMMON_BLOCKS = {'DFF','DFF_BUFFER','AND2','AND3','AND4','WORDLINEDRIVER',
                 'PRECHARGE','SENSEAMP','WRITEDRIVER','WRITE_DATA_BUFFER'}


def placement_scope(device, instances):
    """Keep complete common circuits together, including their child gates."""
    parts=device['name'].split('/')[:-1]
    for depth in range(1,len(parts)+1):
        scope='/'.join(parts[:depth])
        if instances[scope]['subcircuit'] in COMMON_BLOCKS:
            return scope
    return '/'.join(parts)


def compact_placement(layout, fingers, instances, columns, pitch, margin):
    """Return every finger's origin, reusable blocks and occupied top edge.

    A block contains one common source circuit, including its child gates,
    or the direct MOS of a remaining leaf subcircuit. Larger folded
    blocks occupy several rows. First-fit shelves of decreasing height avoid
    charging a whole row for one unusually wide transistor.
    """
    groups = defaultdict(list)
    for index, (device, finger, width) in enumerate(fingers):
        groups[placement_scope(device,instances)].append((index, device, finger, width))
    primitives, templates, blocks = {}, {}, []
    for scope, group in groups.items():
        keys = [(d['model'], round(w, 6), round(d['length_m']*1e9, 6))
                for _, d, _, w in group]
        for key in keys:
            if key not in primitives:
                primitives[key] = controller.mos(layout, f'COMPACT_MOS_{len(primitives)}',
                                                 *key, compact_pins=True)
        local_columns = min(len(group), columns)
        signature = instances[scope]['subcircuit'], tuple(keys), local_columns
        if signature not in templates:
            cell = Builder(layout, f'COMPACT_SUBCKT_{len(templates)}')
            offsets, cursor = [], 0.
            for first in range(0, len(group), local_columns):
                row = keys[first:first+local_columns]
                for col, key in enumerate(row):
                    x = col*pitch
                    cell.place(primitives[key][0], x, cursor)
                    offsets.append((x, cursor))
                cursor += ceil((max(primitives[k][2] for k in row)+560)/GRID)*GRID
            templates[signature] = cell, offsets, cursor
        cell, offsets, height = templates[signature]
        blocks.append(dict(scope=scope, subcircuit=instances[scope]['subcircuit'],
                           cell=cell, group=group, offsets=offsets,
                           columns=local_columns, height=height))
    shelves = []
    for block in sorted(blocks, key=lambda b: (-b['height'], -b['columns'], b['scope'])):
        shelf = next((s for s in shelves if s['used']+block['columns'] <= columns), None)
        if shelf is None:
            shelf = dict(used=0, height=block['height'], blocks=[])
            shelves.append(shelf)
        block['column'] = shelf['used']
        shelf['used'] += block['columns']
        shelf['blocks'].append(block)
    positions, manifest, cursor = {}, [], 1120.
    for shelf in shelves:
        for block in shelf['blocks']:
            x, y = margin+block['column']*pitch, cursor
            for (index, _, _, _), (dx, dy) in zip(block['group'], block['offsets']):
                positions[index] = x+dx, y+dy
            manifest.append(dict(instance=block['scope'], subcircuit=block['subcircuit'],
                                 cell=block['cell'].name, x_nm=x, y_nm=y,
                                 width_nm=block['columns']*pitch, height_nm=block['height']))
        cursor += shelf['height']
    return positions, blocks, manifest, primitives, cursor


def route_compact(router, routed, order, attempts=8):
    """Negotiate pin access by moving a caged net ahead of its blockers.

    Every retry restores all physical obstacles and terminal reservations.
    Only route order changes; no clearance or circuit constraint is relaxed.
    """
    owner, blocked = dict(router.owner), set(router.blocked)
    halos = {p:set(nets) for p,nets in router.via_halos.items()}
    history = []
    seen = set()
    for attempt in range(attempts):
        identity = tuple(order)
        if identity in seen:
            raise RuntimeError('Compact routing repeated a failed order; expand placement spacing')
        seen.add(identity)
        router.owner, router.blocked = dict(owner), set(blocked)
        router.paths, router.vias = [], []
        router.via_halos = {p:set(nets) for p,nets in halos.items()}
        try:
            for index,net in enumerate(order):
                points=list(dict.fromkeys(router.point(*p) for p in routed[net]))
                if len(points)>1:
                    router.connect(net,points[0],points[1:])
                if index%50==0:
                    print('routed',index,'/',len(order),net,'attempt',attempt+1,flush=True)
            return history
        except RoutingError as error:
            candidates = [order.index(n) for n in error.blockers if n in order[:index]]
            if not candidates or attempt+1 == attempts:
                raise
            position = min(candidates)
            history.append(dict(net=error.net, terminal=error.target,
                                blockers=sorted(error.blockers), attempt=attempt+1))
            order.remove(error.net)
            order.insert(position,error.net)
            print('retry routing:',error.net,'before',order[position+1],flush=True)


def compact_route_order(routed):
    """Connect small nets before global trunks can enclose local terminals."""
    def key(net):
        points=set(routed[net])
        count=len(points)
        local=count<=12
        span=sum(max(p[axis] for p in points)-min(p[axis] for p in points) for axis in (0,1))
        return not local, net not in ('VDD','VSS'), span if local else -count, count, net
    return sorted(routed,key=key)
