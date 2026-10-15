"""Component-based 6T bank; TIME_CONTROL remains external.

The CLI includes row logic, replica circuitry and output latch by default.
The build API and --data-path-only retain the earlier data-path scope.
No RC profile is reused across these different physical boundaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import bitcell as bc, edge
from .cell_flow import assemble, leaves, schematic_tree, write_view
from .components import CellLibrary, Component, Instance, read_component
from .full_source import flatten
from .lib import Builder, new_layout
from .bank_core import array_tile
from sram_compiler.version import VERSION


SCOPE = ('6T bank data path: array, precharge, sense amplifiers, write drivers '
         'and data buffers; external decoded wordlines and controls; excludes '
         'decoder, replica, TIME_CONTROL and output latch')
ROLES = ('XDIN_BUF_', 'XWRITEDRIVER_', 'XSENSEAMP_', 'XPRECHARGE_')
REPLICA_SCOPE = ('6T data and replica core, dummy load row and column periphery; '
                 'external decoded wordlines and controls')


def transform_pin(pin, x, y, mirror=False):
    """The same x reflection used by Builder.place, including layer identity."""
    px, py, layer = pin
    return (x-px if mirror else x+px), y+py, layer


def source_contract(deck, include_replica=False):
    """Bind every physical instance to a validated subset of the actual deck."""
    source = flatten(deck)
    rows, cols = source.rows, source.cols
    if rows < 2 or cols < 2 or rows % 2 or cols % 2:
        raise ValueError('Bank tile requires even row/column counts >= 2')
    core = f'XSRAM_6T_CORE_{rows}x{cols}'
    required = [core] + [role+str(c) for role in ROLES for c in range(cols)]
    if include_replica:
        required += [f'Xsram_{rows}x1_replica_column', 'XPRECHARGE_RBL',
                     'XREPLICA_SENSEAMP','XREPLICA_WDRV_LOAD']
        required += [f'XRWL_LOAD_{c}' for c in range(cols-1)]
    if any(name not in source.instances for name in required):
        raise ValueError('Bank requires a full 6T array and unmuxed column periphery')
    specs = {name: read_component(deck, name) for name in required}
    if include_replica:
        from .bank_core import replica_contract
        replica_contract(source,specs,rows,cols)
        for name,role in [('XPRECHARGE_RBL','XPRECHARGE_'),
                          ('XREPLICA_SENSEAMP','XSENSEAMP_'),('XREPLICA_WDRV_LOAD','XWRITEDRIVER_')]:
            if specs[name].identity != specs[role+'0'].identity:
                raise ValueError('Replica periphery differs from its column physical master')
        expected = {
            'XPRECHARGE_RBL': dict(VDD='VDD',ENB=source.instances['XPRECHARGE_0']['pins']['ENB'],BL='RBL',BLB='RBLB'),
            'XREPLICA_SENSEAMP':dict(VDD='VDD',VSS='VSS',EN=source.instances['XSENSEAMP_0']['pins']['EN'],
                ISO=source.instances['XSENSEAMP_0']['pins']['ISO'],IN='RBL',INB='RBLB',Q='REPLICA_SA_Q',QB='REPLICA_SA_QB'),
            'XREPLICA_WDRV_LOAD':dict(VDD='VDD',VSS='VSS',EN=source.instances['XWRITEDRIVER_0']['pins']['EN'],
                DIN='VSS',BL='RBL',BLB='RBLB')}
        if any(source.instances[n]['pins']!=p for n,p in expected.items()):
            raise ValueError('Unsupported replica peripheral connections')
    core_pins = source.instances[core]['pins']
    expected = {'MPGL': ('NMOS_VTG', 135), 'MPGR': ('NMOS_VTG', 135),
                'MPDL': ('NMOS_VTG', 205), 'MPDR': ('NMOS_VTG', 205),
                'MPUL': ('PMOS_VTG', 90), 'MPUR': ('PMOS_VTG', 90)}
    array_leaves = list(leaves(specs[core]))
    if len(array_leaves) != rows*cols:
        raise ValueError('Bank requires every array cell')
    for path, cell, pins in array_leaves:
        location = re.fullmatch(r'XSRAM_6T_CELL_(\d+)_(\d+)',path)
        if location is None:
            raise ValueError('Unsupported array instance placement')
        row, col = map(int,location.groups())
        if not (0 <= row < rows and 0 <= col < cols) or pins != dict(
                VDD='VDD',VSS='VSS',BL=f'BL{col}',BLB=f'BLB{col}',WL=f'WL{row}'):
            raise ValueError('Array cell connections differ from the physical tile')
        if cell.name != 'SRAM_6T_CELL' or {d.name for d in cell.devices} != set(expected):
            raise ValueError('Unsupported bitcell physical view')
        for d in cell.devices:
            model, width = expected[d.name]
            if (d.model != model or abs(d.width_m*1e9-width) > 1e-6 or
                    abs(d.length_m*1e9-50) > 1e-6 or d.fingers != 1):
                raise ValueError('Bitcell geometry differs from the fixed dense tile')
        # Check connectivity too: matching W/L alone cannot authorize a tile.
        expected_nodes = {
            'MPGL': ('BL','WL','Q','VSS'), 'MPGR': ('BLB','WL','QB','VSS'),
            'MPDL': ('Q','QB','VSS','VSS'), 'MPDR': ('QB','Q','VSS','VSS'),
            'MPUL': ('Q','QB','VDD','VDD'), 'MPUR': ('QB','Q','VDD','VDD')}
        if any(d.nodes != expected_nodes[d.name] for d in cell.devices):
            raise ValueError('Bitcell connectivity differs from the fixed dense tile')
    # A repeated column has a single physical master. Reject mixed geometry
    # and unexpected wiring instead of silently forcing it onto this floorplan.
    shared = {}
    for c in range(cols):
        def pins(role):
            return source.instances[role+str(c)]['pins']
        pc, sa, wr, buf = [pins(r) for r in
                           ('XPRECHARGE_', 'XSENSEAMP_', 'XWRITEDRIVER_', 'XDIN_BUF_')]
        for role in ROLES:
            if specs[role+str(c)].identity != specs[role+'0'].identity:
                raise ValueError('Column variants need an independently planned physical view')
            for supply in ('VDD', 'VSS'):
                if supply in pins(role) and pins(role)[supply] != core_pins[supply]:
                    raise ValueError('Bank supply mapping differs across components')
        for port, core_port in (('BL',f'BL{c}'), ('BLB',f'BLB{c}')):
            if pc[port] != core_pins[core_port] or wr[port] != pc[port]:
                raise ValueError('Column bitline mapping differs from the array')
            if sa['IN' if port == 'BL' else 'INB'] != pc[port]:
                raise ValueError('Sense input mapping differs from the array')
        if wr['DIN'] != buf['Z']:
            raise ValueError('Write driver must connect to its own data buffer')
        for role, port in (('XPRECHARGE_', 'ENB'), ('XSENSEAMP_', 'EN'),
                           ('XSENSEAMP_', 'ISO'), ('XWRITEDRIVER_', 'EN')):
            key = role, port
            shared.setdefault(key, pins(role)[port])
            if shared[key] != pins(role)[port]:
                raise ValueError('Bank controls must be shared across columns')
    selected = [d for d in source.mos if d['name'].split('/')[0] in required]
    if any(p['name'].split('/')[0] in required for p in source.passives):
        raise ValueError('Bank components with intentional passives need another physical view')
    internal = {core_pins[f'{pol}{c}'] for pol in ('BL','BLB') for c in range(cols)}
    internal.update(source.instances[f'XDIN_BUF_{c}']['pins']['Z'] for c in range(cols))
    if len(internal) != 3*cols:
        raise ValueError('Bank bitlines and buffered data must be independent per column')
    boundary = {core_pins[p] for p in core_pins if p in ('VDD','VSS') or p.startswith('WL')}
    for c in range(cols):
        for role, ports in (('XDIN_BUF_',('A',)),('XSENSEAMP_',('EN','ISO','Q','QB')),
                            ('XPRECHARGE_',('ENB',)),('XWRITEDRIVER_',('EN',))):
            boundary.update(source.instances[role+str(c)]['pins'][p] for p in ports)
    if internal & boundary:
        raise ValueError('Internal bank nets cannot be tied to boundary controls or data')
    if include_replica:
        internal.add('RBLB')
    ports = list(dict.fromkeys(n for name in required
                              for n in source.instances[name]['pins'].values() if n not in internal))
    component = Component(f'BANK_DATA_{rows}X{cols}', tuple(ports), (), tuple(
        Instance(name, tuple(source.instances[name]['pins'][p] for p in specs[name].ports), specs[name])
        for name in required))
    return source, core, specs, component, selected


def build(deck, output, include_replica=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    attempt = dict(status='in_progress', scope=REPLICA_SCOPE if include_replica else SCOPE,
                   compiler_version=VERSION)
    def save():
        (output/'attempt.json').write_text(json.dumps(attempt, indent=2)+'\n')
    save()
    try:
        attempt['source_deck_sha256'] = hashlib.sha256(Path(deck).read_bytes()).hexdigest()
        source, core, specs, component, selected = source_contract(deck,include_replica)
        rows, cols = source.rows, source.cols
        ly = new_layout()
        library = CellLibrary(ly)
        bank = Builder(ly, component.name)
        views = {role: assemble(library, specs[role+'0'], 'column') for role in ROLES}
        placed, terminals, port_points = [], {}, defaultdict(list)
        paths, vias = [], []
        array_x, cursor = 2*bc.W, bc.H
        stage_y = {}

        def wire(net, layer, x0, y0, x1, y1, width=140):
            if x0 != x1 and y0 != y1:
                raise ValueError('Bank wires must be Manhattan')
            h = width/2
            bank.rect(layer, min(x0,x1)-h, min(y0,y1)-h,
                      max(x0,x1)+h, max(y0,y1)+h)
            paths.append((net, (x0,y0,layer), (x1,y1,layer)))

        def via(net, number, x, y):
            bank.via(number,x,y)
            for layer in (number, number+1):
                if layer <= 3:
                    bank.rect(f'm{layer}',x-35,y-70,x+35,y+70)
                else:
                    bank.sq(f'm{layer}',x,y,145 if layer>=5 else 140)
            vias.append((net,x,y,number))

        for role in ROLES:
            view = views[role]
            stage_y[role] = cursor
            count=cols+(include_replica and role!='XDIN_BUF_')
            for c in range(count):
                mirror = bool(c % 2)
                x = array_x+(c+1 if mirror else c)*bc.W
                name = role+str(c)
                if c==cols:
                    from .bank_core import SPACER
                    mirror=False
                    x=array_x+cols*bc.W+SPACER
                    name={'XPRECHARGE_':'XPRECHARGE_RBL','XSENSEAMP_':'XREPLICA_SENSEAMP',
                          'XWRITEDRIVER_':'XREPLICA_WDRV_LOAD'}[role]
                mapping = source.instances[name]['pins']
                bank.place(view.builder,x,cursor,mx=mirror)
                pins = {n: transform_pin(p,x,cursor,mirror) for n,p in view.pins.items()}
                placed.append(dict(instance=name,cell=view.builder.name,x_nm=x,y_nm=cursor,
                                   mirror_x=mirror,width_nm=view.width_nm,height_nm=view.height_nm,
                                   connections=mapping,pins=pins))
                for name_local,p in view.device_pins.items():
                    terminals[name+'/'+name_local] = transform_pin(p,x,cursor,mirror)
                for port,p in pins.items():
                    if name=='XREPLICA_WDRV_LOAD' and port=='DIN':
                        # Tie the replica load locally. A bank-wide VSS bus
                        # on the DIN row would short every data-column driver.
                        tx=x+755
                        for px,py,_ in (p,pins['VSS']):
                            via('VSS',2,px,py);via('VSS',3,px,py)
                            wire('VSS','m4',px,py,tx,py)
                            via('VSS',4,tx,py)
                        wire('VSS','m5',tx,p[1],tx,pins['VSS'][1])
                        continue
                    port_points[mapping[port]].append(p)
            cursor += view.height_nm+bc.H
        array_y = cursor+3*bc.H
        core_info={}
        if include_replica:
            from .bank_core import core_tile, replica_terminals
            tile,core_info=core_tile(ly,rows,cols)
            terminals.update(replica_terminals(rows,cols,array_x,array_y))
        else:
            tile = array_tile(ly,rows,cols)
        bank.place(tile,array_x,array_y)
        core_pins = source.instances[core]['pins']
        placed.append(dict(instance=core,cell=tile.name,x_nm=array_x,y_nm=array_y,
                           width_nm=cols*bc.W,height_nm=rows*bc.H,
                           connections=core_pins))
        if include_replica:
            placed[-1].update(width_nm=core_info['width_nm'],height_nm=core_info['height_nm'],
                logical_instances=[core,f'Xsram_{rows}x1_replica_column',
                                   *[f'XRWL_LOAD_{c}' for c in range(cols-1)]])
        # Contact-level array access, reflected with each actual bitcell.
        left = {
            'MPGL': ((155,400),(0,250),(155,200)),
            'MPDL': ((155,200),(717.5+bc.D,120),(155,0)),
            'MPUL': ((447.5+bc.D,200),(717.5+bc.D,120),(447.5+bc.D,0))}
        for row in range(rows):
            for col in range(cols):
                for name, contacts in left.items():
                    for side in ('L','R'):
                        device = name[:-1]+side
                        for term,(px,py) in zip('dgs',contacts):
                            if side=='R':
                                px,py=bc.W-px,bc.H-py
                            if col%2:
                                px=bc.W-px
                            if row%2:
                                py=bc.H-py
                            terminals[f'{core}/XSRAM_6T_CELL_{row}_{col}/{device}@0:{term}'] = (
                                array_x+col*bc.W+px,array_y+row*bc.H+py,'m1')
                        bx = bc.X_VDD if device.startswith('MPU') else 92.5
                        if side=='R' and not device.startswith('MPU'):
                            bx=bc.W-bx
                        if col%2:
                            bx=bc.W-bx
                        terminals[f'{core}/XSRAM_6T_CELL_{row}_{col}/{device}@0:b'] = (
                            array_x+col*bc.W+bx,array_y-edge.TAP_H/2,'m1')
        right = array_x+core_info.get('width_nm',cols*bc.W)
        external = {}
        shared = [core_pins['VDD'],core_pins['VSS']]
        shared += [source.instances[role+'0']['pins'][p] for role,p in
                   (('XPRECHARGE_','ENB'),('XSENSEAMP_','EN'),
                    ('XSENSEAMP_','ISO'),('XWRITEDRIVER_','EN'))]
        if len(set(shared)) != len(shared):
            raise ValueError('Tied bank controls need a different routing plan')
        # M4 buses connect leaf interfaces. Buffer assemblies already own M4
        # and M5, so their supply buses cross on M6 instead.
        for index, net in enumerate(shared):
            trunk_x = bc.H+index*300
            trunk_ys = []
            by_y = defaultdict(list)
            for x,y,layer in port_points[net]:
                by_y[(y,layer)].append(x)
            for (y,layer), xs in by_y.items():
                bus = 'm6' if layer=='m5' else 'm4'
                for x in xs:
                    if layer=='m2':
                        via(net,2,x,y); via(net,3,x,y)
                    elif layer=='m5':
                        via(net,5,x,y)
                    else:
                        raise ValueError('Unsupported column supply/control interface')
                wire(net,bus,trunk_x,y,max(xs),y,145 if bus=='m6' else 140)
                via(net,5 if bus=='m6' else 4,trunk_x,y)
                trunk_ys.append(y)
            if net in (core_pins['VDD'],core_pins['VSS']):
                supply = 'VDD' if net==core_pins['VDD'] else 'VSS'
                y = array_y-edge.TAP_H-(400 if supply=='VDD' else 800)
                xs = ([array_x+c*bc.W+bc.X_VDD for c in range(cols)] if supply=='VDD'
                      else [array_x+c*bc.W for c in range(cols+1)])
                if include_replica:
                    rx=array_x+core_info['replica_x_nm']
                    xs += [rx+bc.X_VDD] if supply=='VDD' else [rx,rx+bc.W]
                for x in xs:
                    wire(net,'m3',x,y,x,array_y,70)
                    via(net,3,x,y)
                wire(net,'m4',trunk_x,y,max(xs),y)
                via(net,4,trunk_x,y)
                trunk_ys.append(y)
            wire(net,'m5',trunk_x,min(trunk_ys),trunk_x,max(trunk_ys))
            external[net] = (trunk_x,min(trunk_ys),'m5')
        # The BL/BLB M5 spine ends at an explicit M3/M4/M5 core landing.
        for c in range(cols+int(include_replica)):
            mirror = bool(c%2)
            origin = array_x+(c+1 if mirror else c)*bc.W
            if c==cols:
                origin=array_x+core_info['replica_x_nm'];mirror=False
            for polarity,local in (('BL',bc.X_BL),('BLB',bc.X_BLB)):
                net = ('RBL' if polarity=='BL' else 'RBLB') if c==cols else core_pins[f'{polarity}{c}']
                x = origin-local if mirror else origin+local
                points = port_points[net]
                if any(px != x or layer!='m5' for px,_,layer in points):
                    raise ValueError('Mirrored component bitlines do not align with the tile')
                y = array_y-edge.TAP_H
                wire(net,'m5',x,min(p[1] for p in points),x,y)
                via(net,3,x,y); via(net,4,x,y)
                if c==cols and polarity=='BL':
                    external[net]=(x,y,'m5')
            if c==cols:
                continue
            net = source.instances[f'XDIN_BUF_{c}']['pins']['Z']
            points = port_points[net]
            start = next(p for p in points if p[2]=='m5')
            end = next(p for p in points if p[2]=='m2')
            x,y,_ = start; ex,ey,_ = end
            wire(net,'m5',x,y,x,ey)
            via(net,4,x,ey)
            wire(net,'m4',x,ey,ex,ey)
            via(net,2,ex,ey); via(net,3,ex,ey)
        for row in range(rows):
            net = core_pins[f'WL{row}']
            y = array_y+row*bc.H+bc.Y_WL
            wire(net,'m2',array_x-300,y,array_x,y,70)
            external[net] = (array_x-300,y,'m2')
        if include_replica:
            x,y,layer=core_info['rwl_pin']
            external['RWL']=(array_x+x,array_y+y,layer)
        for net in component.ports:
            if net not in external:
                if len(port_points[net]) != 1:
                    raise ValueError(f'Unsupported bank boundary connection: {net}')
                external[net] = port_points[net][0]
            x,y,layer = external[net]
            bank.label(layer,net,x,y)
        # Source MOS and physical gate count remain independently inspectable.
        from .lib import pya
        gates = (pya.Region(bank.cell.begin_shapes_rec(bank.li('active'))) &
                 pya.Region(bank.cell.begin_shapes_rec(bank.li('poly')))).merged()
        expected_fingers = sum(d['fingers'] for d in selected)
        expected_terminals = {f'{d["name"]}@{f}:{t}' for d in selected
                              for f in range(d['fingers']) for t in 'dgsb'}
        if set(terminals) != expected_terminals:
            raise ValueError('Physical terminal inventory differs from the bank source')
        expected_gate_area = sum(d['width_m']*d['length_m']*1e18 for d in selected)
        measured_gate_area = gates.area()*(ly.dbu*1000)**2
        if gates.count()!=expected_fingers or abs(measured_gate_area-expected_gate_area)>1e-5:
            raise ValueError('Physical gate geometry differs from the selected source MOS')
        bank.rect('bnd',0,0,right+bc.H,array_y+max(rows*bc.H+2*bc.H,core_info.get('height_nm',0)+bc.H))
        export = new_layout()
        export.create_cell(bank.name).copy_tree(bank.cell)
        export.write(str(output/'full_macro.gds'))
        (output/'source.sp').write_text(schematic_tree(component,{component.identity:bank.name}))
        box = bank.cell.bbox()
        area = box.width()*box.height()*ly.dbu**2
        manifest = dict(schema=1,primary=bank.name,artifact_kind='bank',scope=attempt['scope'],
            rows=rows,cols=cols,compiler_version=VERSION,controller_layout=False,
            included_instances=list(specs),excluded_instances=[n for n in source.instances
                if '/' not in n and n not in specs],
            source_deck_sha256=attempt['source_deck_sha256'],
            source_sha256=hashlib.sha256((output/'source.sp').read_bytes()).hexdigest(),
            layout_sha256=hashlib.sha256((output/'full_macro.gds').read_bytes()).hexdigest(),
            generator_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')},
            core_pitch_nm=dict(row=bc.H,column=bc.W),array_origin_nm=[array_x,array_y],
            stage_origins_y_nm=stage_y,
            replica_integrated_in_core=include_replica,core_geometry=core_info,
            macro_bbox_nm=[v*ly.dbu*1000 for v in (box.left,box.bottom,box.right,box.top)],
            macro_area_um2=area,density_bits_per_um2=rows*cols/area,
            placements=placed,device_pins=terminals,external_pins=external,
            source_devices=selected,source_mos=len(selected),source_fingers=expected_fingers,
            physical_gate_count=gates.count(),physical_gate_area_nm2=measured_gate_area,
            routed_paths=paths,routed_vias=vias,geometry_status='generated_unverified',
            waveform_validation='not run',defaults_promotable=False)
        summary = Path(deck).with_name('summary.json')
        if summary.is_file():
            metadata = json.loads(summary.read_text())
            manifest['compiler_provenance']={k:metadata.get(k) for k in (
                'compiler_version','corner','vdd','temperature','variation_mode',
                'real_cell_mode','seed','model_sha256','interconnect','timing','driver_sizes')}
            manifest['compiler_summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
        (output/'layout.json').write_text(json.dumps(manifest,indent=2)+'\n')
        catalog = []
        for view in [*library.views.values(),*library.parents.values()]:
            catalog.append(write_view(library,view,output/'cells'/view.builder.name,
                                      deck,'column',view.component.name))
        (output/'library.json').write_text(json.dumps(dict(schema=1,cells=catalog),indent=2)+'\n')
        attempt['status']='generated_unverified'
        return output/'full_macro.gds'
    except BaseException as error:
        attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--data-path-only',action='store_true',
                        help='retain the earlier data-array/column-periphery scope')
    parser.add_argument('--legacy-pitch',action='store_true',
                        help='use the historical 1365 nm column / 400 nm row bank')
    parser.add_argument('--with-controller',action='store_true',help='integrate TIME_CONTROL and its data registers')
    parser.add_argument('--controller-floorplan',choices=('corner','beside'),default='corner',
                        help='corner: shared decoder/control block aligned with column registers (any size); '
                             'beside: the earlier 32x32 plan left of the decoder')
    parser.add_argument('--controller-layout-dir',type=Path,help='reuse control logic from a previous --with-controller run (OUTPUT_parts/control_logic)')
    args=parser.parse_args(argv)
    if args.with_controller:
        if args.data_path_only or args.legacy_pitch:parser.error('--with-controller requires the OpenRAM-pitch complete bank')
        if args.controller_floorplan=='corner':
            if args.controller_layout_dir:parser.error('--controller-layout-dir applies to the beside floorplan')
            from .corner_control import generate
            generate(args.deck,args.output_dir)
        else:
            from .macro_control import generate
            generate(args.deck,args.output_dir,args.controller_layout_dir)
        print(args.output_dir/'full_macro.gds')
        return 0
    if args.controller_layout_dir:parser.error('--controller-layout-dir requires --with-controller')
    if args.data_path_only:
        print(build(args.deck,args.output_dir))
    elif args.legacy_pitch:
        from .bank_complete import build as integrated_build
        print(integrated_build(args.deck,args.output_dir))
    else:
        from .dense_bank import build as integrated_build
        integrated_build(args.deck,args.output_dir)
        print(args.output_dir/'full_macro.gds')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
