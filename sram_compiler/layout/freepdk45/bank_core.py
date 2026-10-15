"""Dense data/replica core with source-matched replica and dummy devices."""
from . import bitcell as bc, edge
from .cell_flow import leaves
from .lib import Builder, pya

# One core row pitch also clears opposite-well corners in the repeated
# column peripherals; the historical 155 nm gap needed custom well fillers.
SPACER = bc.H


def array_tile(layout: pya.Layout, rows: int, cols: int) -> Builder:
    """Build the full transistor tile; no cells or capacitance are omitted."""
    if rows < 2 or rows % 2 or cols < 2 or cols % 2:
        raise ValueError('The mirrored array tile requires even dimensions >= 2')
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


def replica_contract(source, specs, rows, cols):
    name = f'Xsram_{rows}x1_replica_column'
    column = specs[name]
    pins = source.instances[name]['pins']
    expected_pins = dict(VDD='VDD',VSS='VSS',RBL='RBL',RBLB='RBLB',
                         **{f'WL{r}': 'RWL' if r==rows-1 else 'VSS' for r in range(rows)})
    if pins != expected_pins:
        raise ValueError('Unsupported replica-column boundary mapping')
    widths = {'MPGL':135,'MPGR':135,'MPDL':205,'MPDR':205,'MPUL':90,'MPUR':90}
    def check(cell, kind):
        nodes = {'MPGL':('RBL','WL','Q','VSS'), 'MPGR':('RBLB','WL','VDD','VSS'),
                 'MPDL':('Q','VDD','VSS','VSS'), 'MPDR':('VDD','Q','VSS','VSS'),
                 'MPUL':('Q','VDD','VDD','VDD'), 'MPUR':('VDD','Q','VDD','VDD')}
        if kind=='passive':
            nodes.update({n:(v[0],'VSS',v[2],v[3]) for n,v in nodes.items() if n not in ('MPGL','MPGR')})
        if kind=='dummy':
            nodes['MPGL']=('bl_node_noconnect','WL','Q','VSS')
            nodes['MPGR']=('blb_node_noconnect','WL','VDD','VSS')
        if {d.name for d in cell.devices} != set(widths):
            raise ValueError('Replica/dummy device inventory differs from dense view')
        for d in cell.devices:
            model = 'PMOS_VTG' if d.name.startswith('MPU') else 'NMOS_VTG'
            if (d.nodes!=nodes[d.name] or d.model!=model or d.fingers!=1 or
                    abs(d.width_m*1e9-widths[d.name])>1e-6 or abs(d.length_m*1e9-50)>1e-6):
                raise ValueError('Replica/dummy topology or W/L/NF differs from dense view')
    children=list(leaves(column))
    if len(children)!=rows:
        raise ValueError('Replica column must retain every source cell')
    for r,(path,cell,mapping) in enumerate(children):
        kind='replica' if r==rows-1 else 'passive'
        expected_name='Replica_CELL' if kind=='replica' else 'Replica_CELL_PASSIVE'
        if (path!=f'XReplica_CELL_{r}' or cell.name!=expected_name or
                mapping!=dict(VDD='VDD',VSS='VSS',RBL='RBL',RBLB='RBLB',WL=f'WL{r}')):
            raise ValueError('Replica cell placement differs from dense view')
        check(cell,kind)
    for c in range(cols-1):
        name=f'XRWL_LOAD_{c}'
        if source.instances[name]['pins']!=dict(VDD='VDD',VSS='VSS',BL='VDD',BLB='VDD',WL='RWL'):
            raise ValueError('Dummy load boundary differs from dense view')
        check(specs[name],'dummy')


def core_tile(layout, rows, cols):
    tile=Builder(layout,f'CORE_WITH_REPLICA_{rows}X{cols}')
    cells={k:bc.make_bitcell(layout,k) for k in ('cell','replica','passive','dummy')}
    bottom=edge.make_tap(layout,'CORE_BOTTOM',far='periphery')
    upper=edge.make_tap(layout,'CORE_TOP',far='periphery')
    dummy_tap=edge.make_tap(layout,'CORE_DUMMY_TAP',far='array')
    end=edge.make_edge(layout,'CORE_DUMMY_EDGE')
    rx=cols*bc.W+SPACER
    dummy_y=rows*bc.H+edge.TAP_H
    for c in range(cols+1):
        x,mirror=(rx,False) if c==cols else (((c+1)*bc.W,True) if c%2 else (c*bc.W,False))
        for r in range(rows):
            kind=('replica' if r==rows-1 else 'passive') if c==cols else 'cell'
            tile.place(cells[kind],x,(r+1)*bc.H if r%2 else r*bc.H,mx=mirror,my=bool(r%2))
        tile.place(bottom,x,0,mx=mirror)
        tile.place(dummy_tap if c<cols-1 else upper,x,rows*bc.H,mx=mirror,my=True)
        if c<cols-1:
            tile.place(cells['dummy'],x,dummy_y,mx=mirror)
            tile.place(end,x,dummy_y+bc.H,mx=mirror)
    tile.place(edge.make_spacer(layout,rows*bc.H+2*edge.TAP_H,
                               'CORE_REPLICA_SPACER',w=SPACER,y0=-edge.TAP_H),cols*bc.W,0)
    # The replica load row and active replica cell share the actual RWL wire.
    xj=rx+400
    yr=rows*bc.H-bc.Y_WL
    yd=dummy_y+bc.Y_WL
    tile.rect('m2',(cols-1)*bc.W-35,yd-35,xj+70,yd+35)
    for y in (yr,yd):
        tile.via(2,xj,y)
        tile.rect('m2',xj-35,y-70,xj+35,y+70)
        tile.rect('m3',xj-35,y-70,xj+35,y+70)
    tile.rect('m3',xj-35,yr-70,xj+35,yd+70)
    return tile, dict(replica_x_nm=rx,dummy_y_nm=dummy_y,rwl_pin=(xj,yd,'m3'),
        width_nm=rx+bc.W,height_nm=dummy_y+bc.H+100,
        replica_cells=[dict(instance=f'Xsram_{rows}x1_replica_column/XReplica_CELL_{r}',
                           row=r,kind='replica' if r==rows-1 else 'passive',
                           x_nm=rx,y_nm=(r+1 if r%2 else r)*bc.H,mirror_y=bool(r%2))
                       for r in range(rows)],
        dummy_loads=[dict(instance=f'XRWL_LOAD_{c}',column=c,
                         x_nm=(c+1 if c%2 else c)*bc.W,y_nm=dummy_y,mirror_x=bool(c%2))
                     for c in range(cols-1)])


def replica_terminals(rows, cols, x, y):
    """Named physical access for all replica and dummy finger terminals."""
    result={}
    base={'MPGL':((155,400),(0,250),(155,200)),
          'MPDL':((155,200),(717.5+bc.D,120),(155,0)),
          'MPUL':((447.5+bc.D,200),(717.5+bc.D,120),(447.5+bc.D,0))}
    entries=[(f'Xsram_{rows}x1_replica_column/XReplica_CELL_{r}',cols*bc.W+SPACER,
              r*bc.H,False,bool(r%2),'replica' if r==rows-1 else 'passive',-edge.TAP_H/2)
             for r in range(rows)]
    entries += [(f'XRWL_LOAD_{c}',c*bc.W,rows*bc.H+edge.TAP_H,bool(c%2),False,
                 'dummy',rows*bc.H+edge.TAP_H/2) for c in range(cols-1)]
    for path,ox,oy,mx,my,kind,by in entries:
        for name,contacts in base.items():
            for side in ('L','R'):
                device=name[:-1]+side
                for terminal,(px,py) in zip('dgs',contacts):
                    layer='m1'
                    if side=='R':px,py=bc.W-px,bc.H-py
                    if kind=='passive' and terminal=='g' and name!='MPGL':
                        px,py,layer=bc.X_VDD,bc.Y_WL,'m2'
                    # Dummy PG drain diffusion deliberately has no contact.
                    if kind=='dummy' and name=='MPGL' and terminal=='d':layer='active'
                    if mx:px=bc.W-px
                    if my:py=bc.H-py
                    result[f'{path}/{device}@0:{terminal}']=(x+ox+px,y+oy+py,layer)
                bx=bc.X_VDD if name=='MPUL' else (92.5 if side=='L' else bc.W-92.5)
                if mx:bx=bc.W-bx
                result[f'{path}/{device}@0:b']=(x+ox+bx,y+by,'m1')
    return result
