"""Integrate source decoder, wordline drivers, replica core and output latch.

TIME_CONTROL and its input registers remain outside the physical bank.
"""
import hashlib
import json
from pathlib import Path

from . import bank, bitcell as bc
from .bank_rows import build_rows
from .bank_decoder import build_decoder
from .cell_flow import schematic_tree, write_view
from .components import CellLibrary, Component, Instance, read_component
from .lib import Builder, new_layout, pya
from sram_compiler.version import VERSION

SCOPE = ('6T bank including decoder, wordline drivers, data and replica core, '
         'dummy loads, column periphery and output latch; TIME_CONTROL and '
         'input registers remain external')


def build(deck, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    attempt=dict(status='in_progress',scope=SCOPE)
    def save():
        (output/'attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    save()
    try:
        attempt['source_deck_sha256']=hashlib.sha256(Path(deck).read_bytes()).hexdigest()
        save()
        bank.build(deck,output/'data_path',include_replica=True)
        data=json.loads((output/'data_path/layout.json').read_text())
        source,core,specs,data_component,_=bank.source_contract(deck,True)
        rows,cols=source.rows,source.cols
        row_names=['XDECODER',*[f'XWL_DRV_{r}' for r in range(rows)],'XRWL','XD_LATCH']
        row_specs={n:read_component(deck,n) for n in row_names}
        row_ports=set(n for name in row_names for n in source.instances[name]['pins'].values())
        decoder_outputs={source.instances['XDECODER']['pins'][f'WL{r}'] for r in range(rows)}
        row_ports-=decoder_outputs
        # Verify driver-to-decoder and array mappings before committing geometry.
        for r in range(rows):
            p=source.instances[f'XWL_DRV_{r}']['pins']
            if (p['A']!=source.instances['XDECODER']['pins'][f'WL{r}'] or
                    p['Z']!=source.instances[core]['pins'][f'WL{r}'] or p['B']!='wl_en'):
                raise ValueError('Unsupported decoder/wordline driver mapping')
        specs.update(row_specs)
        omitted=[n for n in source.instances if '/' not in n and n not in specs]
        if omitted!=['XTIME_CONTROL']:
            raise ValueError(f'Unexpected unplaced bank instances: {omitted}')
        internal={source.instances[core]['pins'][f'WL{r}'] for r in range(rows)}|{'RWL'}
        ports=tuple(dict.fromkeys([p for p in data_component.ports if p not in internal]+
                                 sorted(row_ports-set(data_component.ports))))
        component=Component(f'BANK_{rows}X{cols}',ports,(),tuple(
            Instance(n,tuple(source.instances[n]['pins'][p] for p in c.ports),c) for n,c in specs.items()))
        ly=new_layout();library=CellLibrary(ly)
        decoder,logic_info=build_decoder(library,source,row_specs)
        aux_ports={n for name in ('XRWL','XD_LATCH') for n in source.instances[name]['pins'].values()}
        auxiliary,aux_info=build_rows(library,source,row_specs,aux_ports,max(cols,32))
        parent=Builder(ly,component.name)
        parent.place(auxiliary)
        imported=new_layout();imported.read(str(output/'data_path/full_macro.gds'))
        data_cell=Builder(ly,data['primary']);data_cell.cell.copy_tree(imported.cell(data['primary']))
        decoder_x=max(p[0] for p in aux_info['pins'].values())+800
        decoder_y=data['array_origin_nm'][1]
        dx=decoder_x+logic_info['width_nm']+400-data['array_origin_nm'][0]
        dy=0.
        parent.place(decoder,decoder_x,decoder_y)
        parent.place(data_cell,dx,dy)
        data_pins={n:(x+dx,y+dy,l) for n,(x,y,l) in data['external_pins'].items()}
        decoder_pins={n:(x+decoder_x,y+decoder_y,l) for n,(x,y,l) in logic_info['pins'].items()}
        paths=[];vias=[];wordlines=[]
        for r in range(rows):
            net=source.instances[f'XWL_DRV_{r}']['pins']['Z']
            a,b=decoder_pins[net],data_pins[net]
            expected_y=data['array_origin_nm'][1]+dy+bc.Y_WL+r*bc.H
            if a[1]!=b[1] or a[1]!=expected_y or a[0]>=b[0]:
                raise ValueError('Decoder wordlines must meet the left core edge at exact row pitch')
            parent.rect('m2',a[0]-35,a[1]-35,b[0]+35,b[1]+35)
            paths.append((net,a,b));wordlines.append(dict(net=net,row=r,start_nm=a,end_nm=b))
        def m6_access(net,point):
            x,y,layer=point
            if layer in ('m2','m3'):
                if layer=='m2':
                    parent.rect('m2',x-35,y-70,x+35,y+70);parent.via(2,x,y)
                    vias.append((net,x,y,2))
                parent.rect('m3',x-35,y-70,x+35,y+70);parent.via(3,x,y)
                parent.sq('m4',x,y,140);vias.append((net,x,y,3))
                # A sense output's M2 pin is close to its M5 bitline. Move the
                # upper stack into the middle of its column on M4 first.
                if layer=='m2':
                    core_x=dx+data['array_origin_nm'][0]
                    col=int((x-core_x)//bc.W)
                    gx=core_x+(col+.5)*bc.W
                    parent.rect('m4',min(x,gx)-70,y-70,max(x,gx)+70,y+70)
                    x=gx
                parent.via(4,x,y);parent.sq('m4',x,y,140)
                parent.sq('m5',x,y,145);vias.append((net,x,y,4))
            parent.via(5,x,y)
            parent.sq('m5',x,y,145);parent.sq('m6',x,y,145)
            vias.append((net,x,y,5))
            return x,y
        external={}
        for net in set(data_pins)|set(decoder_pins)|set(aux_info['pins']):
            if net in {link['net'] for link in wordlines}:continue
            points=[p[net] for p in (aux_info['pins'],data_pins,decoder_pins) if net in p]
            root=points[0]
            if len(points)>1:
                if net not in aux_info['pins']:
                    raise ValueError('Shared controls require an auxiliary side trunk')
                x,y,_=root
                for target in points[1:]:
                    px,py=m6_access(net,target)
                    parent.rect('m5',x-70,min(y,py)-70,x+70,max(y,py)+70)
                    m6_access(net,(x,py,'m5'))
                    parent.rect('m6',min(x,px)-72.5,py-72.5,max(x,px)+72.5,py+72.5)
                    paths.extend([(net,(x,y,'m5'),(x,py,'m5')),
                                  (net,(x,py,'m6'),(px,py,'m6'))])
            if net in ports:external[net]=root
        if set(external)!=set(ports):raise ValueError('Missing integrated bank boundary ports')
        for net,(x,y,layer) in external.items():parent.label(layer,net,x,y)
        terminals=dict(aux_info['device_pins'])
        terminals.update({n:(x+decoder_x,y+decoder_y,l) for n,(x,y,l) in logic_info['device_pins'].items()})
        terminals.update({n:(x+dx,y+dy,l) for n,(x,y,l) in data['device_pins'].items()})
        devices=[d for d in source.mos if d['name'].split('/')[0] in specs]
        expected={f'{d["name"]}@{f}:{t}' for d in devices for f in range(d['fingers']) for t in 'dgsb'}
        if set(terminals)!=expected:raise ValueError('Incomplete integrated bank terminal map')
        gates=(pya.Region(parent.cell.begin_shapes_rec(parent.li('active')))&
               pya.Region(parent.cell.begin_shapes_rec(parent.li('poly')))).merged()
        count=sum(d['fingers'] for d in devices)
        area=sum(d['width_m']*d['length_m']*1e18 for d in devices)
        if gates.count()!=count or abs(gates.area()*(ly.dbu*1000)**2-area)>1e-4:
            raise ValueError('Integrated physical gate inventory differs from source')
        width=max(aux_info['width_nm'],dx+data['macro_bbox_nm'][2])
        height=max(aux_info['height_nm'],dy+data['macro_bbox_nm'][3])
        parent.rect('bnd',0,0,width,height)
        export=new_layout();export.create_cell(parent.name).copy_tree(parent.cell)
        export.write(str(output/'full_macro.gds'))
        (output/'source.sp').write_text(schematic_tree(component,{component.identity:parent.name}))
        manifest=dict(schema=2,primary=parent.name,artifact_kind='bank',scope=SCOPE,rows=rows,cols=cols,
            compiler_version=VERSION,controller_layout=False,replica_integrated_in_core=True,
            included_instances=list(specs),excluded_instances=omitted,ports=ports,
            source_devices=devices,source_mos=len(devices),source_fingers=count,
            device_pins=terminals,physical_gate_count=gates.count(),physical_gate_area_nm2=area,
            external_pins=external,macro_bbox_nm=[0,0,width,height],macro_area_um2=width*height/1e6,
            density_bits_per_um2=rows*cols/(width*height/1e6),core_pitch_nm=data['core_pitch_nm'],
            array_origin_nm=[dx+data['array_origin_nm'][0],dy+data['array_origin_nm'][1]],
            core_geometry=data['core_geometry'],data_path_origin_nm=[dx,dy],row_logic=logic_info,auxiliary_logic=aux_info,
            decoder_origin_nm=[decoder_x,decoder_y],
            decoder_bbox_nm=[decoder_x,decoder_y,decoder_x+logic_info['width_nm'],decoder_y+logic_info['height_nm']],
            wordline_links=wordlines,
            data_path_placements=data['placements'],routed_paths=paths,routed_vias=vias,
            geometry_status='generated_unverified',waveform_validation='not run',defaults_promotable=False,
            source_deck_sha256=data['source_deck_sha256'],compiler_provenance=data.get('compiler_provenance'),
            compiler_summary_sha256=data.get('compiler_summary_sha256'),
            source_sha256=hashlib.sha256((output/'source.sp').read_bytes()).hexdigest(),
            layout_sha256=hashlib.sha256((output/'full_macro.gds').read_bytes()).hexdigest(),
            generator_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('*.py')})
        (output/'layout.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for view in library.views.values():
            write_view(library,view,output/'row_cells'/view.builder.name,deck,'row',view.component.name)
        attempt['status']='generated_unverified'
        return output/'full_macro.gds'
    except BaseException as error:
        attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:save()
