"""Array and integrated replica at the native OpenRAM cell pitch."""
from .components import Component,Instance,CellView
from .cell_flow import leaves
from .openram_core import make_cell,W,H,Y_WL
from .lib import Builder


def make_core(layout,source,specs,core_name):
    rows,cols=source.rows,source.cols
    replica=f'Xsram_{rows}x1_replica_column'
    names=[core_name,replica]+[f'XRWL_LOAD_{c}' for c in range(cols-1)]
    component=Component(f'DENSE_CORE_{rows}X{cols}',tuple(['VDD','VSS','RBL','RBLB','RWL']+
        [f'WL{r}' for r in range(rows)]+[f'{p}{c}' for c in range(cols) for p in ('BL','BLB')]),(),
        tuple(Instance(n,tuple(source.instances[n]['pins'][p] for p in specs[n].ports),specs[n]) for n in names))
    b=Builder(layout,component.name);cache={};terminals={};placements=[]
    def put(path,c,kind,x,row):
        if kind not in cache:cache[kind]=make_cell(layout,c,kind)
        v=cache[kind]
        if v.component.identity!=c.identity:raise ValueError('Mixed core cell definitions')
        mirrored=bool(row%2);oy=(row+1 if mirrored else row)*H
        b.place(v.builder,x,oy,my=mirrored)
        terminals.update({path+'/'+n:(x+px,oy-py if mirrored else oy+py,l) for n,(px,py,l) in v.device_pins.items()})
        placements.append(dict(instance=path,x_nm=x,y_nm=oy,mirror_y=mirrored))
        return v
    for path,c,m in leaves(specs[core_name],path=core_name):
        mapped=source.instances[path]['pins'];row=int(mapped['WL'][2:]);col=int(mapped['BL'][2:])
        put(path,c,'cell',col*W,row)
    rx=(cols+1)*W
    for r,(path,c,m) in enumerate(leaves(specs[replica],path=replica)):
        put(path,c,'replica' if r==rows-1 else 'passive',rx,r)
        if r<rows-1:
            x=rx+W;y=(r+.5)*H
            b.via(2,x,y);b.rect('m2',x-35,y-70,x+35,y+70);b.rect('m3',x-35,y-70,x+35,y+70)
    for col in range(cols-1):put(f'XRWL_LOAD_{col}',specs[f'XRWL_LOAD_{col}'],'dummy',col*W,rows)
    # Data and replica supply rails bridge the reserved single-column channel.
    for r in range(rows+1):
        y=r*H;b.rect('m1',cols*W-70,y-35,rx+70,y+35)
    # Shared RWL connects the active replica row to every source dummy load.
    x=rx+500;ya=(rows-.5)*H;yb=(rows+.5)*H
    b.rect('m3',(cols-1)*W-35,yb-35,x+35,yb+35)
    for y in (ya,yb):
        b.via(3,x,y);b.rect('m3',x-35,y-70,x+35,y+70);b.sq('m4',x,y,140)
    b.rect('m4',x-70,ya-70,x+70,yb+70)
    # A physical VDD trunk joins every alternating row pair and the dummy
    # row. Its 65 nm via1 fits the native 65 nm M1 rail without widening it.
    x=352.5
    for r in range(1,rows+2,2):
        y=r*H
        b.sq('v1',x,y,65);b.rect('m1',x-67.5,y-32.5,x+67.5,y+32.5)
        b.rect('m2',x-35,y-70,x+35,y+70);b.via(2,x,y)
        b.rect('m3',x-35,y-70,x+35,y+70);b.via(3,x,y);b.sq('m4',x,y,140)
    b.rect('m4',x-70,H-70,x+70,(rows+1)*H+70)
    pins={'VDD':(352.5,H,'m1'),'VSS':(352.5,0,'m1'),
          'RBL':(rx+185,0,'m2'),'RBLB':(rx+520,0,'m2'),'RWL':(rx+500,(rows+.5)*H,'m4')}
    pins.update({f'WL{r}':(0,(r+.5)*H,'m3') for r in range(rows)})
    pins.update({f'{p}{c}':(c*W+px,0,'m2') for c in range(cols) for p,px in [('BL',185),('BLB',520)]})
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(component,b,(cols+2)*W,(rows+1)*H,pins,terminals,[],[],[])
    v.placements=placements
    return v
