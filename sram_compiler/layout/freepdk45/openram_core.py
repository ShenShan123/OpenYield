"""Source-matched 6T core views at the user-selected OpenRAM pitch.

Upstream BSD-3-Clause GDS masters are immutable assets. Derived passive/dummy
views retain all six MOS devices and change only the required local wiring.
"""
import hashlib
import json
from pathlib import Path

from .components import CellView
from .lib import Builder,new_layout,pya,u

W=705.
H=1365.
X_BL=185.
X_BLB=520.
Y_WL=H/2
ASSETS=Path(__file__).with_name('assets')/'openram'


def make_cell(layout,component,kind='cell'):
    from .dense_contract import core_contract
    core_contract(component,kind)
    if kind not in ('cell','replica','passive','dummy'):raise ValueError('Unsupported core view')
    path=ASSETS/('cell_1rw.gds' if kind=='cell' else 'replica_cell_1rw.gds')
    provenance=json.loads((ASSETS/'provenance.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['sha256'][path.name]:
        raise ValueError('Changed upstream geometry asset')
    imported=new_layout();imported.read(str(path));source=imported.top_cell()
    b=Builder(layout,'ORP_'+kind.upper())
    for layer in imported.layer_indexes():
        info=imported.get_info(layer)
        region=pya.Region(source.begin_shapes_rec(layer)).merged()
        b.cell.shapes(layout.layer(info)).insert(region)
    def remove(layer,box):
        index=b.li(layer);r=pya.Region(b.cell.shapes(index))
        r-=pya.Region(pya.Box(*[u(v) for v in box]));b.cell.shapes(index).clear();b.cell.shapes(index).insert(r.merged())
    def via_small(n,x,y):b.sq(f'v{n}',x,y,65)
    if kind=='passive':
        # Detach the two inverter gate contacts from Q / the tied-high QB.
        for x,y in [(465,900),(240,1050)]:
            remove('cont',(x-32.5,y-32.5,x+32.5,y+32.5))
        remove('m1',(397.5,867.5,485,932.5))
        remove('m1',(220,1017.5,307.5,1082.5))
        for y in (900,1050):
            b.cont(352.5,y);b.rect('m1',285,y-32.5,420,y+32.5)
            via_small(1,352.5,y)
            b.rect('m2',317.5,y-70,387.5,y+70)
        b.rect('m2',317.5,830,387.5,1120)
        for x in (352.5,705):
            b.via(2,x,1050);b.rect('m2',x-35,980,x+35,1120)
            b.rect('m3',x-35,980,x+35,1120)
        b.rect('m3',317.5,1015,740,1085)
        # WL remains a formal port. The replica-column parent makes its
        # source-specified ground tie, preserving independent leaf interfaces.
    if kind=='dummy':
        for x in (267.5,437.5):remove('cont',(x-32.5,312.5,x+32.5,377.5))
        for x in (255,450):remove('v1',(x-32.5,312.5,x+32.5,377.5))
        for box in [(222.5,277.5,300,412.5),(405,277.5,482.5,412.5)]:remove('m1',box)
        # Bitline metal is not a terminal of an uncontacted dummy PG drain.
        index=b.li('m2');r=pya.Region(b.cell.shapes(index)).merged();keep=pya.Region()
        for polygon in r.each():
            box=polygon.bbox()
            if box.left<=u(X_BL)<=box.right or box.left<=u(X_BLB)<=box.right:continue
            keep.insert(polygon)
        b.cell.shapes(index).clear();b.cell.shapes(index).insert(keep)
    # Raise the WL to a centred M3 interface. Mirrored rows retain this phase.
    via_small(1,352.5,140);b.rect('m1',285,107.5,420,172.5)
    b.rect('m2',317.5,70,387.5,210);b.via(2,352.5,140)
    b.rect('m3',317.5,70,387.5,Y_WL+35)
    b.rect('m3',-35,Y_WL-35,W+35,Y_WL+35)
    pins={'VDD':(352.5,H,'m1'),'VSS':(352.5,0,'m1'),'WL':(0,Y_WL,'m3')}
    if kind!='dummy':
        bl,br=('BL','BLB') if kind=='cell' else ('RBL','RBLB')
        pins.update({bl:(X_BL,H/2,'m2'),br:(X_BLB,H/2,'m2')})
    for net,(x,y,layer) in pins.items():b.label(layer,net,x,y)
    # Logical device terminals refer to actual contacts, or active diffusion
    # for deliberately uncontacted dummy drains.
    contacts={
        'MPGL':[(267.5,345,'m1'),(335,160,'m1'),(82.5,345,'m1'),(347.5,0,'m1')],
        'MPGR':[(437.5,345,'m1'),(335,160,'m1'),(622.5,345,'m1'),(347.5,0,'m1')],
        'MPDL':[(187.5,735,'m1'),(465,900,'m1'),(0,735,'m1'),(347.5,0,'m1')],
        'MPDR':[(517.5,735,'m1'),(240,1050,'m1'),(705,735,'m1'),(347.5,0,'m1')],
        'MPUL':[(187.5,1200,'m1'),(465,900,'m1'),(0,1200,'m1'),(347.5,H,'m1')],
        'MPUR':[(517.5,1200,'m1'),(240,1050,'m1'),(705,1200,'m1'),(347.5,H,'m1')]}
    if kind=='passive':
        for d in ('MPDL','MPUL'):contacts[d][1]=(352.5,900,'m1')
        for d in ('MPDR','MPUR'):contacts[d][1]=(352.5,1050,'m1')
    if kind=='dummy':
        for d in ('MPGL','MPGR'):
            x,y,_=contacts[d][0];contacts[d][0]=(x,y,'active')
    terminals={f'{d}@0:{t}':p for d,points in contacts.items() for t,p in zip('dgsb',points)}
    return CellView(component,b,W,H,pins,terminals,[],[],[])
