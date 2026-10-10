"""Complete 8x8 macro placement and routing, including TIME_CONTROL."""
from __future__ import annotations

from collections import defaultdict
from math import ceil
from pathlib import Path
import json
import argparse

from . import controller, top as memory
from .lib import Builder, pya, u
from .router import Router, GRID, LAYERS
from .full_source import flatten, render


FEEDBACK = {
    'XPRECHARGE_RBL/ENB_end': ('PRE', 'feedback_pre'),
    'XREPLICA_SENSEAMP/ISO_end': ('sa_iso', 'feedback_iso'),
    'XREPLICA_SENSEAMP/EN_end': ('s_en', 'feedback_sense'),
    'XREPLICA_WDRV_LOAD/EN_end': ('w_en', 'feedback_write'),
    'RWL_far': ('RWL', 'feedback_wordline'),
    'XREPLICA_SENSEAMP/IN_end': ('RBL', 'feedback_bitline'),
}


def build(deck, output, columns=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = flatten(deck)
    if not any(d['name'].startswith('XD_LATCH/') for d in source.mos):
        raise ValueError('Full macro reference requires a read&write source with its output latch')
    ly, array = memory.build()
    main = Builder(ly, 'OPENYIELD_8X8')
    routed_pins = defaultdict(list)
    device_locations, placements, access = {}, [], []
    fingers = []
    for device in source.mos:
        if not device['name'].startswith(('XTIME_CONTROL/', 'XD_LATCH/')):
            continue
        for finger in range(device['fingers']):
            fingers.append((device, finger, device['width_m'] * 1e9 / device['fingers']))
    cell_cache, row_heights = {}, []
    for first in range(0, len(fingers), columns):
        row = fingers[first:first+columns]
        height = 0
        for device, _, width in row:
            key = (device['model'], round(width, 6), round(device['length_m']*1e9, 6))
            if key not in cell_cache:
                cell_cache[key] = controller.mos(ly, f'CONTROL_MOS_{len(cell_cache)}', *key)
            height = max(height, cell_cache[key][2])
        row_heights.append(ceil((height + 560) / GRID) * GRID)
    margin = 5600.
    row_y, cursor = [], 1120.
    for height in row_heights:
        row_y.append(cursor)
        cursor += height
    cell_positions = {}
    for index, (device, finger, width) in enumerate(fingers):
        row, col = divmod(index, columns)
        x, y = margin + col*controller.CELL_WIDTH, row_y[row]
        key = (device['model'], round(width,6), round(device['length_m']*1e9,6))
        cell, pins, _ = cell_cache[key]
        main.place(cell,x,y)
        placements.append(dict(device=device['name'], finger=finger, x_nm=x, y_nm=y, width_nm=width))
        local_top = 300 + width
        access_y = {'s':140., 'g':local_top+120., 'd':local_top+400., 'b':140.}
        for terminal,(px,py,layer) in pins.items():
            index_terminal = 'dgsb'.index(terminal)
            net = device['nodes'][index_terminal]
            original = device['original_nodes'][index_terminal]
            if device['name'].startswith('XTIME_CONTROL/') and original in FEEDBACK:
                net = FEEDBACK[original][1]
            point = (x+px,y+py,layer)
            routed_pins[net].append(point)
            access.append((net,x+px,y+access_y[terminal],y+py))
            device_locations[device['name']+f'@{finger}:{terminal}'] = point
        cell_positions[device['name'],finger] = (x,y)

    passive_locations=[]
    passive_y = cursor + 1120.
    passive_x = margin
    caps = {item['nodes'][0]:item for item in source.passives if item['kind']=='C'}
    for i,item in enumerate(r for r in source.passives if r['kind']=='R'):
        cell,pins,parameters = controller.passive(ly,f'CONTROL_FILTER_{i}',item['value'])
        main.place(cell,passive_x,passive_y)
        mapping = {'in':item['nodes'][0], 'out':item['nodes'][1], 'ground':caps[item['nodes'][1]]['nodes'][1]}
        for terminal,(px,py,layer) in pins.items():
            routed_pins[mapping[terminal]].append((passive_x+px,passive_y+py,layer))
        box=cell.cell.bbox()
        passive_locations.append(dict(name=item['name'],x_nm=passive_x,y_nm=passive_y,
                                      bbox_nm=[v*ly.dbu*1000 for v in (box.left,box.bottom,box.right,box.top)],
                                      pins={k:[passive_x+v[0],passive_y+v[1],v[2]] for k,v in pins.items()},
                                      nodes=mapping,**parameters))
        passive_x += ceil((box.width()*ly.dbu*1000+1120)/GRID)*GRID
    control_top = passive_y + 6720.
    macro_x = margin
    macro_bottom = ceil((control_top+14000)/GRID)*GRID
    macro_y = macro_bottom - memory.Y_BOT
    # Rename the array-facing registered inputs. Root A/DIN remain external.
    rename = {f'A{i}':f'A_dff{i}' for i in range(3)}
    rename.update({f'DIN{i}':f'DIN_dff{i}' for i in range(8)})
    rename.update({f'Q{i}':f'SA_Q{i}' for i in range(8)})
    rename.update({f'QB{i}':f'SA_QB{i}' for i in range(8)})
    rename.update(WL_EN='wl_en',W_EN='w_en',S_EN='s_en',SA_ISO='sa_iso')
    for layer_index in ly.layer_indexes():
        for shape in array.cell.shapes(layer_index).each():
            if shape.is_text() and shape.text.string in rename:
                text=shape.text
                text.string=rename[text.string]
                shape.text=text
    main.place(array,macro_x,macro_y)
    max_x = margin + columns*controller.CELL_WIDTH + 11200
    max_y = macro_bottom + 15400
    router = Router(main,(-20,-8,ceil(max_x/GRID),ceil(max_y/GRID)))
    # Reserve the memory block; access is through explicit external escapes.
    box = array.cell.bbox()
    for layer in range(3):
        for x in range(int((macro_x+box.left*ly.dbu*1000)/GRID)-1,
                       ceil((macro_x+box.right*ly.dbu*1000)/GRID)+2):
            for y in range(int(macro_bottom/GRID)-1,ceil((macro_y+box.top*ly.dbu*1000)/GRID)+2):
                router.blocked.add((x,y,layer))
    for net,x,y0,y1 in access:
        for yi in range(ceil((min(y0,y1)-175)/GRID),int((max(y0,y1)+175)/GRID)+1):
            router.reserve(router.point(x,yi*GRID),net)
    # Passive plate regions are real routing obstructions. Their declared
    # connection points remain accessible from outside the plate.
    for item in passive_locations:
        x0,y0,x1,y1=item['bbox_nm']
        for layer in range(3):
            for xi in range(int((item['x_nm']+x0)/GRID)-1,ceil((item['x_nm']+x1)/GRID)+2):
                for yi in range(int((item['y_nm']+y0)/GRID)-1,ceil((item['y_nm']+y1)/GRID)+2):
                    router.blocked.add((xi,yi,layer))
        for terminal,p in item['pins'].items():
            q=router.point(*p)
            net=item['nodes'][terminal]
            router.blocked.discard(q)
            # Extend each port into the open channel above/below the passive.
            ey=ceil((item['y_nm']+y1+840)/GRID)*GRID
            main.rect(p[2],p[0]-35 if p[2]=='m3' else p[0]-70,p[1]-70,
                      p[0]+35 if p[2]=='m3' else p[0]+70,ey+70)
            for yi in range(q[1],int(ey/GRID)+1):
                qp=(q[0],yi,q[2]);router.blocked.discard(qp);router.reserve(qp,net)
            old=tuple(p)
            routed_pins[net].remove(old)
            routed_pins[net].append((p[0],ey,p[2]))

    # Bottom-facing macro data/sense ports escape on M3 into separate M4 rows.
    macro_ports=[]
    for col in range(8):
        for kind,local in [('DIN',memory.cp.X_DIN),('Q',memory.cp.X_Q),('QB',memory.cp.X_QB)]:
            old=f'{kind}{col}';net=rename[old]
            x=macro_x+memory.gx(col,local+memory.OFF)
            y=macro_y+memory.Y_BOT+20
            target_y=macro_bottom-1120-(col*3+('DIN','Q','QB').index(kind))*280
            main.rect('m3',x-35,target_y-70,x+35,y+70)
            for xi in range(ceil((x-140)/GRID),int((x+140)/GRID)+1):
                for yi in range(ceil((target_y-175)/GRID),int((y+175)/GRID)+1):
                    router.blocked.add((xi,yi,0))
            gx=round(x/GRID)*GRID
            main.via(3,x,target_y)
            main.rect('m4',min(x,gx)-70,target_y-70,max(x,gx)+70,target_y+70)
            macro_ports.append((net,(gx,target_y,'m4')))
    # Left-facing decoder/control ports use M2 outside the memory boundary.
    left=[(f'A{i}',memory.JOIN[f'A{i}']) for i in range(3)]
    left += [('WL_EN',memory.JOIN['WL_EN']),('PRE',memory.Y_CP+memory.cp.ROW_PRE),
             ('SA_ISO',memory.Y_CP+memory.cp.ROW_ISO),('S_EN',memory.Y_CP+memory.cp.ROW_SEN),
             ('W_EN',memory.Y_CP+memory.cp.ROW_WEN)]
    for i,(old,py) in enumerate(left):
        net=rename.get(old,old);y=macro_y+py
        gx=macro_x-1120-i*560
        gy=round(y/GRID)*GRID
        main.rect('m2',gx-35,y-35,macro_x+55,y+35)
        main.via(2,gx,y)
        main.rect('m2',gx-70,y-35,gx+70,y+35)
        main.rect('m3',gx-35,min(y,gy)-70,gx+35,max(y,gy)+70)
        macro_ports.append((net,(gx,gy,'m3')))
    # Feedback routes touch the replica end, and are routed separately from
    # their corresponding driver lead all the way through the controller.
    feedback_points = {
        'feedback_pre':(memory.X_REP+memory.OFF+650,memory.Y_CP+memory.cp.ROW_PRE,'m2'),
        'feedback_iso':(memory.X_REP+memory.OFF+650,memory.Y_CP+memory.cp.ROW_ISO,'m2'),
        'feedback_sense':(memory.X_REP+memory.OFF+650,memory.Y_CP+memory.cp.ROW_SEN,'m2'),
        'feedback_write':(memory.X_REP+memory.OFF+650,memory.Y_CP+memory.cp.ROW_WEN2,'m2'),
        'feedback_wordline':(memory.X_REP+400,memory.Y_TOP-20,'m3'),
        'feedback_bitline':(memory.X_REP+memory.bc.X_BL,memory.Y_TOP-20,'m3'),
    }
    for i,(net,(px,py,layer)) in enumerate(feedback_points.items()):
        x,y=macro_x+px,macro_y+py
        gx=ceil((macro_x+memory.X_END+1120)/GRID)*GRID+i*560
        gy=round(y/GRID)*GRID
        if layer=='m2':
            main.rect('m2',x-35,y-35,gx+35,y+35)
            main.via(2,gx,y)
            main.rect('m2',gx-70,y-35,gx+70,y+35)
            main.rect('m3',gx-35,min(y,gy)-70,gx+35,max(y,gy)+70)
            macro_ports.append((net,(gx,gy,'m3')))
        else:
            if net=='feedback_bitline':
                main.rect('m3',x-35,y-70,x+35,y+630)
                y+=560
                gy=round(y/GRID)*GRID
            main.rect('m3',x-35,y-70,x+35,y+70)
            main.via(3,x,y)
            main.rect('m4',x-70,y-70,gx+70,y+70)
            main.via(3,gx,y)
            main.rect('m3',gx-35,min(y,gy)-70,gx+35,max(y,gy)+70)
            macro_ports.append((net,(gx,gy,'m3')))
    # Connect both array power trunks explicitly to the controller supply mesh.
    for net,px in [('VDD',memory.rp.X_VDD_TRUNK),('VSS',memory.rp.X_VSS_TRUNK)]:
        x=macro_x+px;y=macro_bottom+280
        gy=macro_bottom-(1120 if net=='VDD' else 840);gx=round(x/GRID)*GRID
        main.rect('m3',x-35,gy-70,x+35,y+70)
        main.via(3,x,gy)
        main.rect('m4',min(x,gx)-70,gy-70,max(x,gx)+70,gy+70)
        macro_ports.append((net,(gx,gy,'m4')))
    for net,p in macro_ports:
        q=router.point(*p)
        router.blocked.discard(q)
        routed_pins[net].append(p)
    external={}
    for i,net in enumerate(source.ports):
        p=(margin+i*560,-1120.,'m3')
        routed_pins[net].append(p);external[net]=p
        main.rect('m3',p[0]-35,p[1]-70,p[0]+35,p[1]+70)
        main.label('m3',net,p[0],p[1])
    for net,points in routed_pins.items():
        for p in points:
            router.reserve(router.point(*p),net)
    # Supplies first, then clocks/control fanout, then short local connections.
    order=sorted(routed_pins,key=lambda n:(n not in ('VDD','VSS'),-len(routed_pins[n]),n))
    for i,net in enumerate(order):
        points=list(dict.fromkeys(router.point(*p) for p in routed_pins[net]))
        if len(points)>1:
            router.connect(net,points[0],points[1:])
        if i%50==0:print('routed',i,'/',len(order),net,flush=True)
    router.draw()
    from .power import add_power_grid
    power_grid = add_power_grid(main, router.paths, macro_bottom, macro_x, control_top, external)
    ly.write(str(output/'full_macro.gds'))
    (output/'source.sp').write_text(render(source))
    manifest=dict(primary=main.name,ports=source.ports,source_mos=len(source.mos),
                  control_fingers=len(fingers),placements=placements,device_pins=device_locations,
                  external_pins=external,passives=passive_locations,
                  routed_paths=router.paths,routed_vias=router.vias,
                  aliases=source.aliases,grid_nm=GRID,controller_bbox_nm=[0,0,max_x,control_top])
    manifest['source_devices']=source.mos
    manifest['source_instances']=source.instances
    manifest['source_passives']=source.passives
    manifest['power_grid']=power_grid
    (output/'layout.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'full_macro.gds'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Use a new output directory to preserve layout evidence')
    print(build(args.deck, args.output_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
