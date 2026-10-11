"""Size-dependent 6T calibration layouts with compiler-matched periphery.

The dense bitcell tile retains its proven geometry. All peripheral, replica,
register and controller MOS are placed from the flattened compiler source,
including explicit fingers and body contacts. This deliberately spacious
reference is for RC calibration, rather than an area-optimized SRAM layout.
The fixed original 8x8 generator remains available independently.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import ceil
from pathlib import Path

from . import bitcell as bc, controller, edge
from .full_source import flatten, render
from .lib import Builder, new_layout
from .power import add_power_grid
from .router import GRID, Router


def array_tile(layout, rows, cols):
    """Build the full transistor tile; no cells or capacitance are omitted."""
    if rows < 2 or rows % 2 or cols < 2 or cols % 2:
        raise ValueError('The mirrored calibration tile requires even dimensions >= 2')
    tile = Builder(layout, f'ARRAY_{rows}X{cols}')
    cell = bc.make_bitcell(layout)
    bottom = edge.make_tap(layout, 'TAP_BOTTOM', far='periphery')
    top = edge.make_tap(layout, 'TAP_TOP', far='periphery')
    for col in range(cols):
        x, mirror = ((col+1)*bc.W, True) if col % 2 else (col*bc.W, False)
        for row in range(rows):
            tile.place(cell, x, (row+1)*bc.H if row % 2 else row*bc.H,
                       mx=mirror, my=bool(row % 2))
        tile.place(bottom, x, 0, mx=mirror)
        tile.place(top, x, rows*bc.H, mx=mirror, my=True)
    return tile


def reserve_supply_stacks(router, memory_taps, external):
    """Protect fixed M4/M5 landings before any signal can overfly them."""
    fixed_stacks = dict(memory_taps)
    for net in ('VDD', 'VSS'):
        fixed_stacks[net+'_pad'] = external[net][:2]
    for name,(x,y) in fixed_stacks.items():
        net = name.split('_',1)[0]
        center = router.point(x,y,'m5')
        for layer in (1,2):
            router.reserve((center[0],center[1],layer),net)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx or dy:
                        point=(center[0]+dx,center[1]+dy,layer)
                        if router.owner.get(point,net)!=net:
                            raise ValueError(f'Supply stack clearance overlaps a {router.owner[point]} pin')
                        router.blocked.add(point)


def build(deck, output, placement_columns=32, power_tap_count=None, allow_same_net_vias=False,
          placement_pitch_nm=None):
    source = flatten(deck)
    rows, cols = source.rows, source.cols
    if not any(d['name'].startswith('XD_LATCH/') for d in source.mos):
        raise ValueError('Complete reference requires the read&write output latch')
    if placement_columns < 8:
        raise ValueError('Use at least eight placement columns')
    placement_pitch = controller.CELL_WIDTH if placement_pitch_nm is None else placement_pitch_nm
    if (not 0 < placement_pitch < float('inf') or placement_pitch < controller.CELL_WIDTH
            or abs(placement_pitch/GRID-round(placement_pitch/GRID))>1e-8):
        raise ValueError('Placement pitch must fit the device and be a finite multiple of the routing grid')
    if power_tap_count is not None and power_tap_count<1:
        raise ValueError('Supply tap count must be positive')
    core = f'XSRAM_6T_CORE_{rows}x{cols}'
    expected = {'MPGL': ('NMOS_VTG', 135), 'MPGR': ('NMOS_VTG', 135),
                'MPDL': ('NMOS_VTG', 205), 'MPDR': ('NMOS_VTG', 205),
                'MPUL': ('PMOS_VTG', 90), 'MPUR': ('PMOS_VTG', 90)}
    cells = [d for d in source.mos if d['name'].startswith(core+'/')]
    if len(cells) != rows*cols*6:
        raise ValueError('Calibration layout requires every 6T transistor')
    for device in cells:
        model, width = expected[device['name'].rsplit('/', 1)[1]]
        if (device['model'] != model or abs(device['width_m']*1e9-width)>1e-6
                or abs(device['length_m']*1e9-50)>1e-6 or device['fingers']!=1):
            raise ValueError('Bitcell geometry differs from the calibration tile')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    layout = new_layout()
    main = Builder(layout, f'OPENYIELD_{rows}X{cols}')
    routed = defaultdict(list)
    access, placements, locations = [], [], {}
    fingers = [(d, finger, d['width_m']*1e9/d['fingers'])
               for d in source.mos if not d['name'].startswith(core+'/')
               for finger in range(d['fingers'])]
    cache, row_y, cursor = {}, [], 1120.
    for first in range(0, len(fingers), placement_columns):
        height = 0.
        row_y.append(cursor)
        for device, _, width in fingers[first:first+placement_columns]:
            key = device['model'], round(width, 6), round(device['length_m']*1e9, 6)
            if key not in cache:
                cache[key] = controller.mos(layout, f'PERIPHERAL_MOS_{len(cache)}', *key)
            height = max(height, cache[key][2])
        cursor += ceil((height+560)/GRID)*GRID
    margin = 11200.
    for index, (device, finger, width) in enumerate(fingers):
        row, col = divmod(index, placement_columns)
        x, y = margin+col*placement_pitch, row_y[row]
        key = device['model'], round(width, 6), round(device['length_m']*1e9, 6)
        cell, pins, _ = cache[key]
        main.place(cell, x, y)
        placements.append(dict(device=device['name'], finger=finger, x_nm=x, y_nm=y, width_nm=width))
        access_y = {'s':140., 'g':300+width+120., 'd':300+width+400., 'b':140.}
        for terminal, (px, py, layer) in pins.items():
            net = device['nodes']['dgsb'.index(terminal)]
            point = x+px, y+py, layer
            routed[net].append(point)
            access.append((net, x+px, y+access_y[terminal], y+py))
            locations[device['name']+f'@{finger}:{terminal}'] = point
    caps = {p['nodes'][0]:p for p in source.passives if p['kind']=='C'}
    passives, passive_x, passive_y = [], margin, cursor+1120.
    for index, item in enumerate(p for p in source.passives if p['kind']=='R'):
        cell, pins, parameters = controller.passive(layout, f'FILTER_{index}', item['value'])
        main.place(cell, passive_x, passive_y)
        mapping = dict(zip(('in', 'out'), item['nodes']))
        mapping['ground'] = caps[item['nodes'][1]]['nodes'][1]
        box = cell.cell.bbox()
        bounds = [v*layout.dbu*1000 for v in (box.left, box.bottom, box.right, box.top)]
        points = {k:[passive_x+p[0], passive_y+p[1], p[2]] for k,p in pins.items()}
        for terminal, point in points.items():
            routed[mapping[terminal]].append(tuple(point))
        passives.append(dict(name=item['name'], x_nm=passive_x, y_nm=passive_y,
                             bbox_nm=bounds, pins=points, nodes=mapping, **parameters))
        passive_x += ceil((bounds[2]-bounds[0]+1120)/GRID)*GRID
    controller_top = passive_y+6720.
    array_x = margin
    array_y = ceil((controller_top+14000+cols*2*560)/GRID)*GRID
    tile = array_tile(layout, rows, cols)
    main.place(tile, array_x, array_y)
    max_x = max(margin+placement_columns*placement_pitch+11200,
                array_x+cols*bc.W+5600, passive_x+5600)
    max_y = array_y+rows*bc.H+5600
    router = Router(main, (-20, -8, ceil(max_x/GRID), ceil(max_y/GRID)),
                    use_tree_heuristic=True, allow_same_net_vias=allow_same_net_vias)
    # The array and its bitline escapes are unavailable to the generic router.
    for layer in range(3):
        for xi in range(int((array_x-200)/GRID), ceil((array_x+cols*bc.W+200)/GRID)+1):
            for yi in range(int((array_y-edge.TAP_H-200)/GRID),
                            ceil((array_y+rows*bc.H+edge.TAP_H+200)/GRID)+1):
                router.blocked.add((xi, yi, layer))
    for net, x, y0, y1 in access:
        for yi in range(ceil((min(y0,y1)-175)/GRID), int((max(y0,y1)+175)/GRID)+1):
            router.reserve(router.point(x, yi*GRID), net)
    for item in passives:
        x0,y0,x1,y1=item['bbox_nm']
        for layer in range(3):
            for xi in range(int((item['x_nm']+x0)/GRID)-1, ceil((item['x_nm']+x1)/GRID)+2):
                for yi in range(int((item['y_nm']+y0)/GRID)-1, ceil((item['y_nm']+y1)/GRID)+2):
                    router.blocked.add((xi,yi,layer))
        for terminal, point in item['pins'].items():
            x,y,layer=point
            ey=ceil((item['y_nm']+y1+840)/GRID)*GRID
            half=35 if layer=='m3' else 70
            main.rect(layer,x-half,y-70,x+half,ey+70)
            q=router.point(x,y,layer)
            net=item['nodes'][terminal]
            for yi in range(q[1],int(ey/GRID)+1):
                p=(q[0],yi,q[2]);router.blocked.discard(p);router.reserve(p,net)
            routed[net].remove(tuple(point));routed[net].append((x,ey,layer))
    array_pins=source.instances[core]['pins']
    for row in range(rows):
        y=array_y+row*bc.H+bc.Y_WL
        x=array_x-1120-(row%8)*560
        gy=round(y/GRID)*GRID
        main.rect('m2',x-70,y-35,array_x+35,y+35)
        main.via(2,x,y)
        main.rect('m3',x-35,min(y,gy)-70,x+35,max(y,gy)+70)
        net=array_pins[f'WL{row}']
        for yi in range(ceil((min(y,gy)-235)/GRID),int((max(y,gy)+235)/GRID)+1):
            router.blocked.discard((round(x/GRID),yi,0))
            router.reserve((round(x/GRID),yi,0),net)
        routed[net].append((x,gy,'m3'))
    for col in range(cols):
        origin=(col+1)*bc.W if col%2 else col*bc.W
        for polarity,local in (('BL',bc.X_BL),('BLB',bc.X_BLB)):
            x=array_x+origin+(-local if col%2 else local)
            y=array_y-edge.TAP_H
            target_y=array_y-2240-(col*2+(polarity=='BLB'))*560
            gx=round(x/GRID)*GRID
            main.rect('m3',x-35,target_y-70,x+35,y+70)
            main.via(3,x,target_y)
            main.rect('m4',min(x,gx)-70,target_y-70,max(x,gx)+70,target_y+70)
            for xi in range(ceil((x-235)/GRID),int((x+235)/GRID)+1):
                for yi in range(ceil((target_y-175)/GRID),int((y+175)/GRID)+1):
                    router.blocked.add((xi,yi,0))
            routed[array_pins[f'{polarity}{col}']].append((gx,target_y,'m4'))
    memory_taps={}
    for net,offset,local in (('VDD',560,bc.X_VDD),('VSS',1120,0)):
        y=array_y-edge.TAP_H-offset
        xpin=array_x-840
        main.rect('m4',xpin-70,y-70,array_x+cols*bc.W+70,y+70)
        for xi in range(int((xpin-210)/GRID),ceil((array_x+cols*bc.W+210)/GRID)+1):
            for yi in range(ceil((y-210)/GRID),int((y+210)/GRID)+1):
                router.blocked.add((xi,yi,1))
        for col in range(cols):
            x=array_x+col*bc.W+local
            main.rect('m3',x-35,y-70,x+35,array_y+70)
            main.via(3,x,y)
        main.via(3,xpin,y)
        # Align the physical supply landing with the maze and M6 feed grid.
        gy=round(y/GRID)*GRID
        main.rect('m3',xpin-35,min(y,gy)-70,xpin+35,max(y,gy)+70)
        routed[net].append((xpin,gy,'m3'))
        memory_taps[net]=(xpin,gy)
    external={}
    for index, net in enumerate(source.ports):
        if net in ('VDD','VSS'):
            point=margin+index*560,-1120.,'m3'
            routed[net].append(point)
            main.rect('m3',point[0]-35,point[1]-70,point[0]+35,point[1]+70)
        else:
            # Expose the existing register/sense terminal as the macro port.
            # Long artificial leads from every input to a common bottom edge
            # create antenna violations and unrelated access capacitance.
            point=routed[net][0]
        external[net]=point
        main.label('m3',net,point[0],point[1])
    for net,points in routed.items():
        for point in points:
            q=router.point(*point)
            router.blocked.discard(q)
            router.reserve(q,net)
            if q[2] == 0:
                # Reserve a via3 escape on M4 before routing any other net.
                # A later horizontal bus must not trap an unrouted M3 pin.
                router.reserve((q[0],q[1],1),net)
    # Fixed pad/array feeds receive M3-to-M6 stacks after signal routing.
    # Reserve their upper landings now: an M5 overflight would otherwise
    # short that signal to a supply, even when ordinary spacing DRC passes.
    reserve_supply_stacks(router,memory_taps,external)
    order=sorted(routed,key=lambda n:(n not in ('VDD','VSS'),-len(routed[n]),n))
    for index,net in enumerate(order):
        points=list(dict.fromkeys(router.point(*p) for p in routed[net]))
        if len(points)>1:
            router.connect(net,points[0],points[1:])
        if index%50==0:
            print('routed',index,'/',len(order),net,flush=True)
    router.draw()
    fractions=(None if power_tap_count is None else
               tuple((index+.5)/power_tap_count for index in range(power_tap_count)))
    power=add_power_grid(main,router.paths,array_y,array_x,controller_top,external,memory_taps,fractions)
    layout.write(str(output/'full_macro.gds'))
    (output/'source.sp').write_text(render(source, main.name))
    manifest=dict(primary=main.name,rows=rows,cols=cols,ports=source.ports,
                  source_mos=len(source.mos),control_fingers=len(fingers),
                  placements=placements,device_pins=locations,external_pins=external,
                  passives=passives,routed_paths=router.paths,routed_vias=router.vias,
                  aliases=source.aliases,grid_nm=GRID,source_devices=source.mos,
                  source_instances=source.instances,source_passives=source.passives,
                  controller_bbox_nm=[0,0,max_x,controller_top],power_grid=power,
                  allow_same_net_vias=allow_same_net_vias,
                  placement_pitch_nm=placement_pitch,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_deck_sha256=hashlib.sha256(Path(deck).read_bytes()).hexdigest())
    (output/'layout.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'full_macro.gds'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--placement-columns',type=int,default=32)
    parser.add_argument('--power-tap-count',type=int,
                        help='number of upper-metal controller taps per supply; default uses the original four')
    parser.add_argument('--allow-same-net-vias',action='store_true',
                        help='permit adjacent via4 landings on the same electrical net; requires fresh DRC/LVS/xRC')
    parser.add_argument('--placement-pitch-nm',type=float,
                        help='horizontal placement pitch, at least 1680 nm and a multiple of the 280 nm routing grid')
    args=parser.parse_args(argv)
    print(build(args.deck,args.output_dir,args.placement_columns,args.power_tap_count,
                args.allow_same_net_vias,args.placement_pitch_nm))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
