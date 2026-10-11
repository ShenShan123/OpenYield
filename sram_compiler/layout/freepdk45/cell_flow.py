"""Build small physical cells, then place and route a compiler component.

Row leaf cells use three-row slices; column components occupy one bitcell
column. This is an opt-in component development flow, not a qualified macro.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import ceil
from pathlib import Path

from . import bitcell
from .components import CellLibrary, CellView, read_component, validate_axis
from .lib import Builder, new_layout
from sram_compiler.version import VERSION


def leaves(component, mapping=None, path=''):
    """Flatten instances down to physical cells, preserving their boundaries."""
    mapping = dict(zip(component.ports, component.ports)) if mapping is None else mapping
    def net(n):
        return mapping.get(n, path + '/' + n if path else n)
    if component.devices:
        if component.children:
            raise ValueError(f'{component.name}: mixed direct MOS/child views need a separate leaf component')
        yield path, component, {p: net(p) for p in component.ports}
    for instance in component.children:
        childpath = path + '/' + instance.name if path else instance.name
        yield from leaves(instance.component,
                          dict(zip(instance.component.ports, (net(n) for n in instance.nodes))), childpath)


def assemble(library, component, axis):
    """Place physical cells, then route only their named boundary ports.

    Column parents use five M5 trunks inside one core column. Row parents
    use M5 pin legs and M3 channels at the core row pitch. Routing capacity
    is checked explicitly; dimensions and fingers are never changed to fit.
    """
    if not component.children:
        return library.build(component, axis)
    validate_axis(component, axis)
    array_outputs = [p for p in component.ports if p.startswith('WL') and p[2:].isdigit()]
    if component.name=='WORDLINEDRIVER':
        array_outputs=['Z']
    key=component.identity,axis,tuple(array_outputs)
    if key in library.parents:
        return library.parents[key]
    physical_identity=hashlib.sha256(json.dumps(key).encode()).hexdigest()
    cell = Builder(library.layout, f'{axis.upper()}_ASSEMBLY_{physical_identity[:16]}')
    placed, device_pins, cursor = [], {}, 0.
    routed = defaultdict(list)
    for path, leaf, connections in leaves(component):
        view = library.build(leaf, axis)
        x, y = (cursor, 0.) if axis == 'row' else (0., cursor)
        cell.place(view.builder, x, y)
        placed.append(dict(instance=path, cell=view.builder.name, component=leaf.name,
                           x_nm=x, y_nm=y, width_nm=view.width_nm, height_nm=view.height_nm,
                           connections=connections))
        for name, (px, py, layer) in view.device_pins.items():
            device_pins[path+'/'+name] = x+px, y+py, layer
        for port,(px,py,layer) in view.pins.items():
            routed[connections[port]].append((x+px,y+py,layer))
        cursor += view.width_nm if axis == 'row' else view.height_nm
    if axis=='column' and len(routed)>5:
        raise ValueError('One-column assembly has five upper-metal trunks; use a dedicated physical view for this component')
    access_width=ceil((len(array_outputs)*300+280)/bitcell.W)*bitcell.W if array_outputs else 0
    width = cursor+access_width if axis=='row' else bitcell.W
    row_tracks, occupied = {}, []
    if axis=='row':
        # Local nets reuse channels when their physical spans do not overlap.
        # Carrying every short gate connection to a global trunk creates
        # unnecessary metal area and can violate antenna limits.
        order=sorted(routed,key=lambda n:(n in component.ports,
            max(p[0] for p in routed[n])-min(p[0] for p in routed[n]),n))
        for net in order:
            begin=140. if net in component.ports else min(p[0] for p in routed[net])
            end=max(p[0] for p in routed[net])
            if net in array_outputs:
                end=width-140-array_outputs.index(net)*300
            slot=next((i for i,ranges in enumerate(occupied)
                       if all(end+300<=a or begin>=b+300 for a,b in ranges)),len(occupied))
            if slot==len(occupied):
                occupied.append([])
            occupied[slot].append((begin,end))
            row_tracks[net]=(4+slot)*bitcell.H
    height = (4+len(occupied))*bitcell.H if axis=='row' else cursor
    pins, paths, vias, array_pins = {}, [], [], {}
    for index,net in enumerate(sorted(routed,key=lambda n:(n not in ('VDD','VSS'),n))):
        points=routed[net]
        for x,y,layer in points:
            if layer!='m2':
                raise ValueError('Small-cell interfaces must be on M2')
            # The library reserves this access site on M3 before local routing.
            cell.rect('m2',x-35,y-70,x+35,y+70)
            cell.via(2,x,y)
            cell.rect('m3',x-35,y-70,x+35,y+70)
            cell.via(3,x,y)
            cell.sq('m4',x,y,140)
            vias.extend([(net,x,y,2),(net,x,y,3)])
        if axis=='row':
            track=row_tracks[net]
            for x,y,_ in points:
                cell.via(4,x,y)
                cell.sq('m5',x,y,145)
                cell.rect('m5',x-70,min(y,track)-70,x+70,max(y,track)+70)
                cell.via(4,x,track)
                cell.sq('m5',x,track,145)
                cell.sq('m4',x,track,140)
                cell.via(3,x,track)
                cell.rect('m3',x-35,track-70,x+35,track+70)
                paths.append((net,(x,y,'m5'),(x,track,'m5')))
                vias.extend([(net,x,y,4),(net,x,track,4),(net,x,track,3)])
            start=(140. if net in component.ports else min(p[0] for p in points),track,'m3')
            cell.rect('m3',start[0]-35,track-35,max(p[0] for p in points)+35,track+35)
            paths.append((net,start,(max(p[0] for p in points),track,'m3')))
        else:
            # 300 nm accommodates 145 nm via4 landings and 140 nm spacing.
            track=90.+index*300
            low,high=min(p[1] for p in points),max(p[1] for p in points)
            cell.rect('m5',track-70,low-70,track+70,high+70)
            for x,y,_ in points:
                cell.rect('m4',min(x,track)-70,y-70,max(x,track)+70,y+70)
                cell.via(4,track,y)
                cell.sq('m5',track,y,145)
                cell.sq('m4',track,y,140)
                paths.append((net,(x,y,'m4'),(track,y,'m4')))
                vias.append((net,track,y,4))
            paths.append((net,(track,low,'m5'),(track,high,'m5')))
            start=(track,low,'m5')
        if net in component.ports:
            if net in array_outputs:
                row=int(net[2:]) if net.startswith('WL') else 0
                target=(width-140-array_outputs.index(net)*300,bitcell.Y_WL+row*bitcell.H,'m5')
                cell.rect('m3',start[0]-35,track-35,target[0]+35,track+35)
                cell.via(3,target[0],track)
                cell.rect('m3',target[0]-35,track-70,target[0]+35,track+70)
                cell.via(4,target[0],track)
                cell.sq('m5',target[0],track,145)
                cell.sq('m4',target[0],track,140)
                cell.rect('m5',target[0]-70,min(track,target[1])-70,target[0]+70,max(track,target[1])+70)
                paths.extend([(net,start,(target[0],track,'m3')),
                              (net,(target[0],track,'m5'),target)])
                vias.extend([(net,target[0],track,4),(net,target[0],track,3)])
                start=target
                array_pins[net]=dict(core_row=row,pin=target)
            pins[net]=start
            cell.label(start[2],net,start[0],start[1])
    if set(pins)!=set(component.ports):
        raise ValueError('Assembly does not expose every schematic port')
    cell.rect('bnd',0,0,width,height)
    view=CellView(component,cell,width,height,pins,device_pins,[],paths,vias,tuple(placed),array_pins)
    library.parents[key]=view
    return view


def schematic_tree(component, names):
    definitions = {}
    def visit(c):
        for i in c.children:
            visit(i.component)
        definitions[c.identity] = c.schematic(names)
    visit(component)
    return '\n'.join(definitions.values())


def write_view(library, view, output, deck, axis, scope):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('full_macro.gds','source.sp','layout.json')):
        raise FileExistsError('Cell artifacts already exist; use a new directory')
    # Export only this view and its physical descendants, not other library tops.
    export = new_layout()
    top = export.create_cell(view.builder.name)
    top.copy_tree(view.builder.cell)
    export.write(str(output/'full_macro.gds'))
    names = {view.component.identity: view.builder.name}
    (output/'source.sp').write_text(schematic_tree(view.component, names))
    abstract = view.abstract()
    box = view.builder.cell.bbox()
    bbox = [v*library.layout.dbu*1000 for v in (box.left, box.bottom, box.right, box.top)]
    from .lib import u
    if box.left < 0 or box.bottom < 0 or box.right > u(view.width_nm) or box.top > u(view.height_nm):
        raise ValueError(f'{view.builder.name}: physical geometry exceeds the pitch frame: {bbox}')
    abstract.update(primary=view.builder.name, artifact_kind='component', axis=axis,
                    scope=f'compiler component {scope}; no SRAM waveform or yield qualification',
                    compiler_instance=scope, compiler_version=VERSION,
                    core_pitch_nm=dict(row=bitcell.H, column=bitcell.W),
                    source_deck_sha256=hashlib.sha256(Path(deck).read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256((output/'source.sp').read_bytes()).hexdigest(),
                    layout_sha256=hashlib.sha256((output/'full_macro.gds').read_bytes()).hexdigest(),
                    macro_bbox_nm=bbox,macro_area_um2=box.width()*box.height()*library.layout.dbu**2,
                    routing_retries=getattr(view, 'routing_retries', []),
                    generator_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in Path(__file__).parent.glob('*.py')})
    summary_path=Path(deck).with_name('summary.json')
    if summary_path.is_file():
        summary=json.loads(summary_path.read_text())
        abstract['compiler_provenance']={k:summary.get(k) for k in (
            'compiler_version','rows','cols','cell_type','corner','vdd','temperature',
            'variation_mode','real_cell_mode','seed','model_sha256','interconnect','timing','driver_sizes')}
        abstract['compiler_summary_sha256']=hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (output/'layout.json').write_text(json.dumps(abstract, indent=2)+'\n')
    return abstract


def build(deck, output, instance, axis='row'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve cell evidence')
    output.mkdir(parents=True)
    attempt=dict(status='in_progress',instance=instance,axis=axis,
                 source_deck_sha256=None,
                 compiler_version=VERSION)
    def save():
        (output/'attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    save()
    try:
        attempt['source_deck_sha256']=hashlib.sha256(Path(deck).read_bytes()).hexdigest()
        component = read_component(deck, instance)
        library = CellLibrary(new_layout())
        view = assemble(library, component, axis)
        result = write_view(library, view, output, deck, axis, instance)
        catalog = []
        for leaf_view in library.views.values():
            path = output/'cells'/leaf_view.builder.name
            catalog.append(write_view(library, leaf_view, path, deck, axis, leaf_view.component.name))
        (output/'library.json').write_text(json.dumps(dict(schema=1, axis=axis,
            core_pitch_nm=result['core_pitch_nm'], cells=catalog,
            verification='independent receipts required for each cell and assembly'), indent=2)+'\n')
        attempt['status']='generated_unverified'
    except BaseException as error:
        attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return output/'full_macro.gds'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--instance', required=True, help='compiler instance path, e.g. XWL_DRV_0 or XPRECHARGE_0')
    parser.add_argument('--axis', choices=('row', 'column'), required=True,
                        help='row: three-row leaf height and row-pitch channels; column: fixed one-column width')
    args = parser.parse_args(argv)
    print(build(args.deck, args.output_dir, args.instance, args.axis))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
