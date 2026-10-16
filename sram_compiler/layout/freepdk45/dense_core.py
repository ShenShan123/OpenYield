"""Array and integrated replica at the native OpenRAM cell pitch."""
from .components import Component,Instance,CellView
from .cell_flow import leaves
from .openram_core import make_cell,W,H,Y_WL
from .lib import Builder


def make_core(layout,source,specs,core_name,span=None):
    """span=(r0, r1) builds one half of a split bank at local row row-r0.

    The upper half (r1=rows) keeps the active replica cell and the RWL dummy
    row; the lower half carries passive replica cells and exposes its
    bitlines at the top edge, which faces the centre periphery.
    """
    rows,cols=source.rows,source.cols
    r0,r1=span or (0,rows)
    upper=r1==rows
    replica=f'Xsram_{rows}x1_replica_column'
    names=[core_name,replica]+[f'XRWL_LOAD_{c}' for c in range(cols-1)]
    component=Component(f'DENSE_CORE_{rows}X{cols}' if span is None else f'DENSE_CORE_{rows}X{cols}_R{r0}_{r1}',
        tuple(['VDD','VSS','RBL','RBLB']+(['RWL'] if upper else [])+
        [f'WL{r}' for r in range(r0,r1)]+[f'{p}{c}' for c in range(cols) for p in ('BL','BLB')]),(),
        tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in specs[n].ports),specs[n]) for n in names))
    b=Builder(layout,component.name);cache={};terminals={};placements=[];fingers=[]
    def put(path,c,kind,x,row):
        if kind not in cache:cache[kind]=make_cell(layout,c,kind)
        v=cache[kind]
        if v.component.identity!=c.identity:raise ValueError('Mixed core cell definitions')
        # Global row parity keeps shared rails; r0 is a multiple of eight.
        mirrored=bool(row%2);oy=(row-r0+1 if mirrored else row-r0)*H
        b.place(v.builder,x,oy,my=mirrored)
        terminals.update({path+'/'+n:(x+px,oy-py if mirrored else oy+py,l) for n,(px,py,l) in v.device_pins.items()})
        placements.append(dict(instance=path,x_nm=x,y_nm=oy,mirror_y=mirrored))
        fingers.extend(dict(device=path+'/'+d.name,finger=i,width_nm=d.width_m*1e9/d.fingers,
                            length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes)
                       for d in c.devices for i in range(d.fingers))
        return v
    for path,c,m in leaves(specs[core_name],path=core_name):
        mapped=source.instances[path]['pins'];row=int(mapped['WL'][2:]);col=int(mapped['BL'][2:])
        if r0<=row<r1:put(path,c,'cell',col*W,row)
    rx=(cols+1)*W
    for r,(path,c,m) in enumerate(leaves(specs[replica],path=replica)):
        if not r0<=r<r1:continue
        put(path,c,'replica' if r==rows-1 else 'passive',rx,r)
        if r<rows-1:
            x=rx+W;y=(r-r0+.5)*H
            b.via(2,x,y);b.rect('m2',x-35,y-70,x+35,y+70);b.rect('m3',x-35,y-70,x+35,y+70)
    n=r1-r0
    if upper:
        for col in range(cols-1):put(f'XRWL_LOAD_{col}',specs[f'XRWL_LOAD_{col}'],'dummy',col*W,rows)
    # Data and replica supply rails bridge the reserved single-column channel.
    for r in range(n+1):
        y=r*H;b.rect('m1',cols*W-70,y-35,rx+70,y+35)
    if upper:
        # Shared RWL connects the active replica row to every source dummy load.
        x=rx+500;ya=(n-.5)*H;yb=(n+.5)*H
        b.rect('m3',(cols-1)*W-35,yb-35,x+35,yb+35)
        for y in (ya,yb):
            b.via(3,x,y);b.rect('m3',x-35,y-70,x+35,y+70);b.sq('m4',x,y,140)
        b.rect('m4',x-70,ya-70,x+70,yb+70)
    # A physical VDD trunk joins every alternating row pair and the dummy
    # row. Its 65 nm via1 fits the native 65 nm M1 rail without widening it.
    x=352.5
    for r in range(1,n+2 if upper else n,2):
        y=r*H
        b.sq('v1',x,y,65);b.rect('m1',x-67.5,y-32.5,x+67.5,y+32.5)
        b.rect('m2',x-35,y-70,x+35,y+70);b.via(2,x,y)
        b.rect('m3',x-35,y-70,x+35,y+70);b.via(3,x,y);b.sq('m4',x,y,140)
    b.rect('m4',x-70,H-70,x+70,((n+1) if upper else (n-1))*H+70)
    # Bitline and supply access face the periphery: the lower half's top edge.
    edge=0 if upper else n*H
    pins={'VDD':(352.5,H if upper else (n-1)*H,'m1'),'VSS':(352.5,edge,'m1'),
          'RBL':(rx+185,edge,'m2'),'RBLB':(rx+520,edge,'m2')}
    if upper:pins['RWL']=(rx+500,(n+.5)*H,'m4')
    pins.update({f'WL{r}':(0,(r-r0+.5)*H,'m3') for r in range(r0,r1)})
    pins.update({f'{p}{c}':(c*W+px,edge,'m2') for c in range(cols) for p,px in [('BL',185),('BLB',520)]})
    for net,(x,y,l) in pins.items():b.label(l,net,x,y)
    v=CellView(component,b,(cols+2)*W,(n+1 if upper else n)*H,pins,terminals,[],[],[])
    v.placements=placements
    # A half holds only its own rows; the full view derives fingers from source.
    if span is not None:v.physical_fingers=fingers
    return v
