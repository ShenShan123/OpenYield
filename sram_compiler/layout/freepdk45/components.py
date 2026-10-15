"""Circuit-owned, pitch-constrained physical cells for FreePDK45.

Cells own their diffusion, contacts, body taps and local M1/M2/M3 wiring. Their
parents see ports and routing obstructions, never a bag of MOS terminals.
The existing reference generators are independent historical layouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from functools import cached_property
import json
from math import ceil, isfinite
from pathlib import Path

from . import bitcell
from .export import Nets, WIRE_C, WIRE_R, instance_parts, parse, parse_file
from .lib import Builder, pya, u
from .router import Router
from sram_compiler.parasitics import spice_value


LOCAL_GRID = 160.


@dataclass(frozen=True)
class Mos:
    name: str
    nodes: tuple
    model: str
    width_m: float
    length_m: float
    fingers: int = 1


@dataclass(frozen=True)
class Instance:
    name: str
    nodes: tuple
    component: 'Component'


@dataclass(frozen=True)
class Passive:
    name: str
    nodes: tuple
    value: float


@dataclass(frozen=True)
class Component:
    """One schematic and its ordered interface, before instance-level ties."""
    name: str
    ports: tuple
    devices: tuple
    children: tuple = ()
    passives: tuple = ()

    @cached_property
    def identity(self):
        body = (self.ports, tuple((d.name, d.nodes, d.model, d.width_m,
                                  d.length_m, d.fingers) for d in self.devices),
                tuple((i.name, i.nodes, i.component.identity) for i in self.children))
        if self.passives:
            body += (tuple((p.name, p.nodes, p.value) for p in self.passives),)
        return hashlib.sha256(json.dumps(body, separators=(',', ':')).encode()).hexdigest()

    def schematic(self, cell_names=None):
        cell_names = cell_names or {}
        name = cell_names.get(self.identity, 'CELL_' + self.identity[:16])
        lines = [f'.subckt {name} ' + ' '.join(self.ports)]
        for d in self.devices:
            lines.append(f'{d.name} ' + ' '.join(d.nodes) +
                         f' {d.model} W={d.width_m:.15g} L={d.length_m:.15g}' +
                         (f' NF={d.fingers}' if d.fingers != 1 else ''))
        for child in self.children:
            ref = cell_names.get(child.component.identity, 'CELL_' + child.component.identity[:16])
            lines.append(child.name + ' ' + ' '.join(child.nodes) + ' ' + ref)
        for passive in self.passives:
            lines.append(passive.name + ' ' + ' '.join(passive.nodes) + f' {passive.value:.15g}')
        return '\n'.join([*lines, f'.ends {name}', ''])


def read_component(deck, instance, allow_passives=False):
    """Resolve a compiler instance without collapsing its formal port ties.

    Wiring RC is removed within each definition. Intentional passives,
    unresolved sweep expressions and unknown MOS parameters fail explicitly.
    """
    scope = parse_file(deck)
    for name in instance.split('/'):
        matches = [t for t in scope.lines if t[0].upper() == name.upper()]
        if len(matches) != 1 or matches[0][0][0].upper() != 'X':
            raise ValueError(f'Compiler instance not found: {instance}')
        pins, ref = instance_parts(matches[0])
        child = scope.resolve(ref)
        if len(pins) != len(child.pins):
            raise ValueError(f'{name}: port count differs from {ref}')
        scope = child
    cache = {}

    def convert(sub):
        if id(sub) in cache:
            return cache[id(sub)]
        nets = Nets()
        for t in sub.lines:
            if t[0][0].upper() == 'R' and WIRE_R.match(t[0]):
                nets.union(t[1], t[2])
        canonical = {}
        formal_ports = set(sub.pins)
        for members in nets.groups().values():
            matched = formal_ports.intersection(members)
            ports = [n for n in sub.pins if n in matched] if matched else []
            if len(ports) > 1:
                raise ValueError(f'{sub.name}: wiring shorts formal ports {ports}')
            chosen = ports[0] if ports else min(members, key=lambda n: (len(n), n))
            canonical.update((n, chosen) for n in members)
        devices, children, passives = [], [], []
        for t in sub.lines:
            kind = t[0][0].upper()
            if kind == 'M':
                params = dict(p.lower().split('=', 1) for p in t[6:])
                if set(params) - {'w', 'l', 'nf'}:
                    raise ValueError(f'{sub.name}: unsupported MOS parameters in {t[0]}')
                w, length = spice_value(params['w']), spice_value(params['l'])
                nf = spice_value(params.get('nf', '1'))
                if not all(isfinite(v) and v > 0 for v in (w, length, nf)) or nf != int(nf):
                    raise ValueError(f'{sub.name}: invalid geometry in {t[0]}')
                devices.append(Mos(t[0], tuple(canonical.get(n, n) for n in t[1:5]),
                                   t[5], w, length, int(nf)))
            elif kind == 'X':
                pins, ref = instance_parts(t)
                component = convert(sub.resolve(ref))
                if len(pins) != len(component.ports):
                    raise ValueError(f'{sub.name}/{t[0]}: incorrect port count')
                children.append(Instance(t[0], tuple(canonical.get(n, n) for n in pins), component))
            elif (kind == 'R' and WIRE_R.match(t[0])) or (kind == 'C' and WIRE_C.match(t[0])):
                continue
            elif allow_passives and kind in ('R', 'C') and len(t) == 4:
                value = spice_value(t[3])
                if not isfinite(value) or value <= 0:
                    raise ValueError(f'{sub.name}: invalid intentional passive {t[0]}')
                passives.append(Passive(t[0], tuple(canonical.get(n, n) for n in t[1:3]), value))
            else:
                raise ValueError(f'{sub.name}: intentional/unsupported element {t[0]} needs a physical view')
        result = Component(sub.name, tuple(sub.pins), tuple(devices), tuple(children), tuple(passives))
        cache[id(sub)] = result
        return result

    return convert(scope)


class CellRouter(Router):
    """Local M1/M2/M3 routing; historical macro defaults stay unchanged."""
    def __init__(self, builder, bounds, **kwargs):
        super().__init__(builder, bounds, grid_nm=LOCAL_GRID,
                         metal_layers=('m1', 'm2', 'm3'), widths=(70., 70., 70.), **kwargs)

    def point(self, x, y, layer='m2'):
        return super().point(x, y, layer)

    def available(self, p, net):
        if not super().available(p, net):
            return False
        # Via enclosures are longer than a wire segment. Keep a foreign track
        # off the adjacent grid point in that enclosure's long direction.
        for owner, x, y, upper in self.vias:
            if owner==net:
                continue
            if p[2]==0 and upper==1 and p[1]==y and abs(p[0]-x)==1:
                return False
            if p[2] in (1,2) and p[2] in (upper-1,upper) and p[0]==x and abs(p[1]-y)==1:
                return False
        return True

    def neighbors(self, p, net):
        for q,cost in super().neighbors(p,net):
            if q[2]!=p[2]:
                upper=max(p[2],q[2])
                if upper==1:
                    halo=[(p[0]+dx,p[1],0) for dx in (-1,1)]
                    halo += [(p[0],p[1]+dy,1) for dy in (-1,1)]
                else:
                    halo=[(p[0],p[1]+dy,layer) for layer in (1,2) for dy in (-1,1)]
                if any(h in self.owner and self.owner[h]!=net for h in halo):
                    continue
            yield q,cost


@dataclass
class CellView:
    component: Component
    builder: Builder
    width_nm: float
    height_nm: float
    pins: dict
    device_pins: dict
    diffusion_runs: list
    paths: list
    vias: list
    children: tuple = ()
    array_pins: dict = field(default_factory=dict)

    def abstract(self, obstructions=True):
        obstacles={}
        for layer in (('m1','m2','m3','m4','m5') if obstructions else ()):
            region=pya.Region(self.builder.cell.begin_shapes_rec(self.builder.li(layer))).merged()
            if not region.is_empty():
                boxes=[polygon.bbox() for polygon in region.each()]
                obstacles[layer]=[[v*self.builder.ly.dbu*1000 for v in
                                   (box.left,box.bottom,box.right,box.top)] for box in boxes]
        return dict(cell=self.builder.name, schematic_identity=self.component.identity,
                    component=self.component.name, width_nm=self.width_nm, height_nm=self.height_nm,
                    row_span=self.height_nm/bitcell.H, column_span=self.width_nm/bitcell.W,
                    pins=self.pins, device_pins=self.device_pins,
                    diffusion_runs=self.diffusion_runs, local_paths=self.paths,
                    local_vias=self.vias, children=self.children,
                    array_pins=self.array_pins, local_grid_nm=LOCAL_GRID,
                    routing_obstructions_nm=obstacles,
                    geometry_status='generated_unverified')


def diffusion_runs(devices, maximum_gates):
    """Share diffusion only at electrically identical terminals and geometry.

    W is the compiler's total BSIM width; each explicit finger keeps W/NF.
    Splitting a trail creates a diffusion break, never another transistor.
    """
    pending = [(d, f) for d in devices for f in range(d.fingers)]
    result = []
    while pending:
        d, f = pending.pop(0)
        run = [(d, f, False)]
        key = (d.model, d.width_m/d.fingers, d.length_m, d.nodes[3])
        end = d.nodes[0]
        while len(run) < maximum_gates:
            candidate = next(((i, item, item[0].nodes[0] == end) for i, item in enumerate(pending)
                              if (item[0].model, item[0].width_m/item[0].fingers,
                                  item[0].length_m, item[0].nodes[3]) == key
                              and end in (item[0].nodes[0], item[0].nodes[2])), None)
            if candidate is None:
                break
            i, (d, f), reverse = candidate
            pending.pop(i)
            run.append((d, f, reverse))
            end = d.nodes[2] if reverse else d.nodes[0]
        result.append(run)
    return result


class CellLibrary:
    """One technology/pitch context; connectivity is part of every cache key."""
    def __init__(self, layout):
        self.layout, self.views, self.parents = layout, {}, {}

    def build(self, component, axis='column'):
        if axis not in ('row', 'column'):
            raise ValueError('Cell axis must be row or column')
        validate_axis(component, axis)
        key = component.identity, axis, tuple(array_bindings(component,axis).items())
        if key not in self.views:
            if component.children:
                raise ValueError('Build leaf cells before assembling their parent')
            self.views[key] = self._leaf(component, axis)
        return self.views[key]

    def _leaf(self, component, axis):
        # A row view is drawn in the same narrow strip and rotated. Its height
        # is exactly three core rows; larger W/NF extends its width instead.
        strip_width = bitcell.W if axis == 'column' else 3*bitcell.H
        physical_identity=hashlib.sha256(json.dumps((component.identity,axis,
            array_bindings(component,axis),bitcell.W,bitcell.H,LOCAL_GRID),sort_keys=True).encode()).hexdigest()
        name = f'{axis.upper()}_{physical_identity[:16]}'
        raw = Builder(self.layout, name + '_LOCAL')
        maximum_gates = int((strip_width-LOCAL_GRID-100)/(2*LOCAL_GRID))
        runs = diffusion_runs(component.devices, maximum_gates)
        points, terminals, descriptions = {}, {}, []
        cursor = 0.
        def terminal(net, x, y):
            raw.cont(x, y)
            raw.rect('m1', x-70, y-35, x+70, y+35)
            raw.via(1, x, y)
            raw.rect('m2', x-35, y-70, x+35, y+70)
            points.setdefault(net, []).append((x, y, 'm2'))
            return (x, y, 'm2')
        for run in runs:
            first = run[0][0]
            width, length = first.width_m*1e9/first.fingers, first.length_m*1e9
            if (first.model not in ('NMOS_VTG', 'PMOS_VTG') or width < 90 or
                    abs(width/5-round(width/5)) > 1e-6 or length < 50 or
                    abs(length/5-round(length/5)) > 1e-6 or length > 140):
                raise ValueError(f'{component.name}: unsupported frozen MOS geometry {first}')
            pmos = first.model == 'PMOS_VTG'
            body_y = cursor+LOCAL_GRID
            cy = ceil((cursor+3*LOCAL_GRID+width/2)/LOCAL_GRID)*LOCAL_GRID
            gy = ceil((cy+width/2+LOCAL_GRID)/LOCAL_GRID)*LOCAL_GRID
            top = gy+LOCAL_GRID
            well, implant, tap_implant = ('nwell', 'pplus', 'nplus') if pmos else ('pwell', 'nplus', 'pplus')
            raw.rect(well, 0, cursor, strip_width, top)
            raw.rect('vtg', 0, cursor, strip_width, top)
            xlast = LOCAL_GRID+2*LOCAL_GRID*len(run)
            raw.rect('active', LOCAL_GRID-37.5, cy-width/2, xlast+37.5, cy+width/2)
            raw.rect(implant, LOCAL_GRID-77.5, cy-width/2-40, xlast+77.5, cy+width/2+40)
            raw.rect('active', LOCAL_GRID-45, body_y-45, LOCAL_GRID+45, body_y+45)
            raw.rect(tap_implant, LOCAL_GRID-85, body_y-85, LOCAL_GRID+85, body_y+85)
            body = terminal(first.nodes[3], LOCAL_GRID, body_y)
            sd = [run[0][0].nodes[0] if run[0][2] else run[0][0].nodes[2]]
            sd += [d.nodes[2] if reverse else d.nodes[0] for d, _, reverse in run]
            contacts = [terminal(n, LOCAL_GRID+2*LOCAL_GRID*i, cy) for i, n in enumerate(sd)]
            descriptions.append(dict(model=first.model, width_nm=width, length_nm=length,
                                     body=first.nodes[3], contact_nets=sd,
                                     fingers=[(d.name, f, reverse) for d, f, reverse in run]))
            for index, (d, f, reverse) in enumerate(run):
                gx = 2*LOCAL_GRID*(index+1)
                raw.rect('poly', gx-length/2, cy-width/2-50, gx+length/2, gy+37.5)
                raw.sq('poly', gx, gy, 75)
                gate = terminal(d.nodes[1], gx, gy)
                drain, source = (contacts[index], contacts[index+1]) if reverse else (contacts[index+1], contacts[index])
                terminals[f'{d.name}@{f}:d'] = drain
                terminals[f'{d.name}@{f}:s'] = source
                terminals[f'{d.name}@{f}:g'] = gate
                terminals[f'{d.name}@{f}:b'] = body
            # The next strip starts with a full tap/access row. The former
            # two additional empty routing rows duplicated that clearance.
            cursor = top
        if not terminals:
            raise ValueError(f'{component.name}: no transistor geometry')
        pins = {}
        px = (int(strip_width/LOCAL_GRID)-1)*LOCAL_GRID
        for index, port in enumerate(component.ports):
            if port not in points:
                raise ValueError(f'{component.name}: unused port {port} needs an explicit physical contract')
            access_x = (LOCAL_GRID if port=='VSS' else px if port=='VDD' else
                        round(strip_width/(2*LOCAL_GRID))*LOCAL_GRID) if axis=='row' else px
            pins[port] = (access_x, cursor+(index+1)*2*LOCAL_GRID, 'm2')
            raw.rect('m2', access_x-35, pins[port][1]-35, access_x+35, pins[port][1]+35)
            points[port].append(pins[port])
        quantum = bitcell.H if axis == 'column' else bitcell.W
        strip_height = ceil((cursor+(len(pins)+2)*2*LOCAL_GRID)/quantum)*quantum
        router = CellRouter(raw, (1, 0, int((strip_width-35)/LOCAL_GRID), int((strip_height-35)/LOCAL_GRID)),
                            use_tree_heuristic=True)
        # A via1 M1 landing extends 70 nm sideways. Keep at least half of
        # the 65 nm M1 spacing at the cell edge for mirrored neighbours.
        for xi in range(1, router.bounds[2]+1):
            if xi*LOCAL_GRID+70 > strip_width-32.5:
                router.blocked.update((xi,yi,0) for yi in range(router.bounds[1],router.bounds[3]+1))
        for net, items in points.items():
            for x, y, _ in items:
                q = router.point(x, y)
                router.reserve(q, net)
                # Via1 enclosures require clearance from adjacent narrow tracks.
                if (x, y, 'm2') not in pins.values():
                    router.reserve((q[0], q[1], 0), net)
                    router.blocked.update(((q[0]-1, q[1], 0), (q[0]+1, q[1], 0),
                                           (q[0], q[1]-1, 1), (q[0], q[1]+1, 1)))
        for net, p in pins.items():
            q=router.point(*p)
            router.reserve((q[0],q[1],2),net)
            # A parent adds a via2 landing after rotating row cells. Reserve
            # both enclosure orientations on M2, not only access on M3.
            router.blocked.update((q[0]+dx,q[1]+dy,1)
                                  for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)))
            router.blocked.update((q[0]+dx,q[1]+dy,2) for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy)
        if axis=='row':
            for net in ('VDD','VSS'):
                if net not in pins:
                    continue
                x,y,_=pins[net]
                raw.rect('m3',x-35,75,x+35,strip_height-75)
                raw.rect('m2',x-35,y-70,x+35,y+70)
                raw.via(2,x,y)
                raw.rect('m3',x-35,y-70,x+35,y+70)
                for yi in range(1,int(strip_height/LOCAL_GRID)):
                    q=(round(x/LOCAL_GRID),yi,2)
                    router.blocked.discard(q)
                    router.reserve(q,net)
        from .placement import compact_route_order, route_compact
        route_compact(router, points, compact_route_order(points))
        router.draw()
        raw.rect('bnd', 0, 0, strip_width, strip_height)
        cell = Builder(self.layout, name)
        if axis == 'row':
            # x' = strip_height-y, y' = x; transform physical pins as well.
            cell.cell.insert(pya.CellInstArray(raw.cell.cell_index(), pya.Trans(pya.Trans.R90, u(strip_height), 0)))
            transform = lambda p: (strip_height-p[1], p[0], p[2])
            width_nm, height_nm = strip_height, strip_width
        else:
            cell.place(raw)
            transform = lambda p: p
            width_nm, height_nm = strip_width, strip_height
        pins = {n:transform(p) for n,p in pins.items()}
        terminals = {n:transform(p) for n,p in terminals.items()}
        array_pins = {}
        if axis=='column':
            for port, bitline in array_bindings(component,axis).items():
                x,y,_ = pins[port]
                target_x = bitcell.X_BL if bitline=='BL' else bitcell.X_BLB
                target_y = height_nm-140
                cell.rect('m2',x-35,y-70,x+35,y+70)
                cell.via(2,x,y)
                cell.rect('m3',x-35,y-70,x+35,y+70)
                cell.via(3,x,y)
                cell.rect('m4',min(x,target_x)-70,y-70,max(x,target_x)+70,y+70)
                cell.via(4,target_x,y)
                cell.sq('m5',target_x,y,145)
                cell.sq('m4',target_x,y,140)
                cell.rect('m5',target_x-70,y-70,target_x+70,target_y+70)
                pins[port]=(target_x,target_y,'m5')
                array_pins[port]=dict(core_net=bitline,pin=pins[port])
        for n, p in pins.items():
            cell.label(p[2], n, p[0], p[1])
        return CellView(component, cell, width_nm, height_nm, pins, terminals,
                        descriptions, router.paths, router.vias,array_pins=array_pins)


def validate_axis(component, axis):
    """Peripheral roles cannot silently choose a geometry that breaks pitch."""
    name=component.name.upper()
    column = name in ('PRECHARGE','SENSEAMP','WRITEDRIVER','WRITE_DATA_BUFFER') or name.startswith('COLUMNMUX')
    row = name=='WORDLINEDRIVER' or name.startswith('DECODER')
    required='column' if column else 'row' if row else None
    if required is not None and axis!=required:
        raise ValueError(f'{component.name} requires the {required} core-pitch axis')


def array_bindings(component,axis):
    if axis!='column':
        return {}
    if component.name=='SENSEAMP':
        return {'IN':'BL','INB':'BLB'}
    if component.name in ('PRECHARGE','WRITEDRIVER'):
        return {'BL':'BL','BLB':'BLB'}
    return {}
