"""Export source-owned physical views with explicit geometry and provenance."""
import hashlib,json
from math import fsum
from collections import defaultdict
from pathlib import Path
from .cell_flow import schematic_tree,leaves
from .lib import new_layout,pya
from sram_compiler.version import VERSION


def write_view(view,output,deck,scope,metadata=None):
    source={('/'.join(filter(None,(path,d.name)))):d
            for path,c,_ in leaves(view.component) for d in c.devices}
    physical=getattr(view,'physical_fingers',None)
    if physical is None:
        physical=[dict(device=n,finger=f,width_nm=d.width_m*1e9/d.fingers,
                       length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes)
                  for n,d in source.items() for f in range(d.fingers)]
    totals={n:0. for n in source};seen=set()
    for f in physical:
        d=source[f['device']];key=(f['device'],f['finger'])
        if key in seen or f['width_nm']<=0 or f['model']!=d.model or tuple(f['nodes'])!=d.nodes or abs(f['length_nm']-d.length_m*1e9)>1e-6:
            raise ValueError('Physical finger topology/model/length differs from source')
        seen.add(key);totals[f['device']]+=f['width_nm']
    if any(abs(totals[n]-d.width_m*1e9)>1e-6 for n,d in source.items()):
        raise ValueError('Physical refingering changed total transistor width')
    expected={f'{n}@{f}:{t}' for n,f in seen for t in 'dgsb'}
    if set(view.device_pins)!=expected:raise ValueError('Incomplete physical MOS terminal map')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ly=new_layout();top=ly.create_cell(view.builder.name);top.copy_tree(view.builder.cell)
    ly.write(str(output/'full_macro.gds'))
    (output/'source.sp').write_text(schematic_tree(view.component,{view.component.identity:view.builder.name}))
    large=len(view.device_pins)>COMPACT_ABOVE
    m=view.abstract(obstructions=not large);box=top.dbbox()
    gates=(pya.Region(top.begin_shapes_rec(ly.layer(1,0))) & pya.Region(top.begin_shapes_rec(ly.layer(9,0)))).merged()
    gate_area=gates.area()*(ly.dbu*1000)**2
    if gates.count()!=len(physical) or abs(gate_area-fsum(d.width_m*d.length_m*1e18 for d in source.values()))>1e-4:
        raise ValueError('Physical gate geometry differs from frozen source W/L')
    m.update(placements=getattr(view,'placements',[]),wordline_links=getattr(view,'wordlines',[]),
        local_grid_nm=2.5,source_mos=len(source),source_fingers=sum(d.fingers for d in source.values()),
        reference_assets_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).parent/'assets'/'openram').glob('*') if p.is_file()},
        primary=view.builder.name,artifact_kind='component',scope=scope,compiler_instance=scope,
        compiler_version=VERSION,core_pitch_nm=dict(row=1365,column=705),
        row_span=view.height_nm/1365,column_span=view.width_nm/705,
        macro_bbox_nm=[1000*v for v in (box.left,box.bottom,box.right,box.top)],macro_area_um2=box.area(),
        physical_gate_count=gates.count(),physical_gate_area_nm2=gates.area()*(ly.dbu*1000)**2,
        physical_fingers=physical,
        refingering_policy='physical refingering authorized; logical W/L and topology preserved',
        source_deck_sha256=hashlib.sha256(Path(deck).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((output/'source.sp').read_bytes()).hexdigest(),
        layout_sha256=hashlib.sha256((output/'full_macro.gds').read_bytes()).hexdigest(),
        generator_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    m.update(metadata or {})
    if large:
        # Repeated instances share one relative terminal/finger master.
        m['device_pins'],m['device_pin_groups']=_compact_pins(m['device_pins'])
        m['physical_fingers'],m['physical_finger_groups']=_compact_fingers(m['physical_fingers'])
        m['routing_obstructions_nm']='omitted above %d terminals; use the GDS' % COMPACT_ABOVE
        m['manifest_compaction']='device_pin_groups/physical_finger_groups: masters of offsets from each instance origin; expand with dense_views.expand'
    (output/'layout.json').write_text(json.dumps(m,indent=1 if large else 2)+'\n')
    return m


COMPACT_ABOVE=100000


def _split(key):
    return key.rsplit('/',1) if '/' in key else ('',key)


def _compact_pins(pins):
    groups=defaultdict(dict)
    for key,(x,y,l) in pins.items():
        inst,local=_split(key);groups[inst][local]=(x,y,l)
    masters={};instances=[];rest={}
    for inst,items in groups.items():
        if len(items)<8 or not inst:
            rest.update({(inst+'/'+k if inst else k):v for k,v in items.items()});continue
        ox=min(x for x,y,l in items.values());oy=min(y for x,y,l in items.values())
        sig=tuple(sorted((k,round(x-ox,3),round(y-oy,3),l) for k,(x,y,l) in items.items()))
        mid=masters.setdefault(sig,len(masters))
        instances.append([inst,mid,ox,oy])
    return rest,dict(masters={i:{k:[dx,dy,l] for k,dx,dy,l in sig} for sig,i in masters.items()},instances=instances)


def _compact_fingers(fingers):
    groups=defaultdict(list)
    for f in fingers:
        inst,local=_split(f['device']);groups[inst].append((local,f))
    masters={};instances=[];rest=[]
    for inst,items in groups.items():
        placed=['gate_xy_nm' in f for _,f in items]
        if len(items)<4 or not inst or (any(placed) and not all(placed)):
            rest.extend(f for _,f in items);continue
        ox=min((f['gate_xy_nm'][0] for _,f in items),default=0.) if all(placed) else 0.
        oy=min((f['gate_xy_nm'][1] for _,f in items),default=0.) if all(placed) else 0.
        sig=tuple(sorted((local,f['finger'],f['width_nm'],f['length_nm'],f['model'],tuple(f['nodes']),
                          *((round(f['gate_xy_nm'][0]-ox,3),round(f['gate_xy_nm'][1]-oy,3)) if 'gate_xy_nm' in f else ()))
                         for local,f in items))
        mid=masters.setdefault(sig,len(masters))
        instances.append([inst,mid,ox,oy])
    def record(item):
        d,k,w,ln,mo,no,*xy=item
        f=dict(device=d,finger=k,width_nm=w,length_nm=ln,model=mo,nodes=list(no))
        if xy:f['gate_xy_nm']=list(xy)
        return f
    return rest,dict(masters={i:[record(item) for item in sig] for sig,i in masters.items()},instances=instances)


def expand(m):
    """Full device_pins and physical_fingers of a (possibly compacted) manifest."""
    pins=dict(m['device_pins']);fingers=list(m['physical_fingers'])
    g=m.get('device_pin_groups')
    if g:
        masters={int(k):v for k,v in g['masters'].items()}
        for inst,mid,ox,oy in g['instances']:
            for local,(dx,dy,l) in masters[mid].items():pins[inst+'/'+local]=(ox+dx,oy+dy,l)
    g=m.get('physical_finger_groups')
    if g:
        masters={int(k):v for k,v in g['masters'].items()}
        for inst,mid,ox,oy in g['instances']:
            for f in masters[mid]:
                f=dict(f,device=inst+'/'+f['device'])
                if 'gate_xy_nm' in f:f['gate_xy_nm']=(ox+f['gate_xy_nm'][0],oy+f['gate_xy_nm'][1])
                fingers.append(f)
    return pins,fingers
