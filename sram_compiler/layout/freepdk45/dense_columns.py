"""Compact column components derived from source-matched reference geometry."""
import hashlib,json
from pathlib import Path
from .components import CellView
from .lib import Builder,new_layout,pya,u
from .openram_core import ASSETS,W
from .dense_contract import require_devices,PRECHARGE,SENSE


def _asset(layout,filename,name,lower_y=None):
    meta=json.loads((ASSETS/'provenance.json').read_text());path=ASSETS/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256'][filename]:raise ValueError('Changed reference asset')
    src=new_layout();src.read(str(path));top=src.top_cell();b=Builder(layout,name)
    for li in src.layer_indexes():
        r=pya.Region(top.begin_shapes_rec(li)).merged()
        if lower_y is not None:r &= pya.Region(pya.Box(u(-1000),u(lower_y),u(2000),u(10000)))
        b.cell.shapes(layout.layer(src.get_info(li))).insert(r)
    return b


def make_precharge(layout,component):
    require_devices(component,PRECHARGE)
    b=_asset(layout,'precharge.gds','DENSE_PRECHARGE')
    pins={}
    # Retain exact native pin geometry, only map its formal interface names.
    names={'vdd':'VDD','en_bar':'ENB','bl':'BL','br':'BLB'}
    src=new_layout();src.read(str(ASSETS/'precharge.gds'));top=src.top_cell()
    for li in src.layer_indexes():
        for shape in top.shapes(li).each():
            if shape.is_text() and shape.text.string in names:
                t=shape.text;ln=src.get_info(li).layer
                pins[names[t.string]]=(t.x*src.dbu*1000,t.y*src.dbu*1000,{11:'m1',13:'m2',15:'m3'}[ln])
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    if set(pins)!=set(component.ports):raise ValueError('Precharge reference pin mismatch')
    # Ordered source M1/M2 charge BL/BLB and M3 equalizes them; official LVS
    # verifies the source model/W/L/topology rather than assuming a cell name.
    terminals={f'{d.name}@0:{t}':pins[n] for d in component.devices for t,n in zip('dgsb',d.nodes)}
    return CellView(component,b,W,b.cell.dbbox().height()*1000,pins,terminals,[],[],[])


def make_sense(layout,component):
    require_devices(component,SENSE)
    offset=1525.
    raw=_asset(layout,'sense_amp.gds','DENSE_SENSE_RAW',lower_y=offset)
    layer=raw.li('poly');r=pya.Region(raw.cell.shapes(layer))
    r-=pya.Region(pya.Box(u(27.5),u(4000),u(77.5),u(4100)))
    raw.cell.shapes(layer).clear();raw.cell.shapes(layer).insert(r.merged())
    # A separate ISO contact on the split P-access gate conductor.
    raw.sq('poly',32.5,3750,75);raw.cont(32.5,3750)
    raw.rect('m1',-2.5,3680,67.5,3820);raw.via(1,32.5,3750)
    raw.rect('m2',-2.5,3680,67.5,3820);raw.via(2,32.5,3750)
    raw.rect('m3',-2.5,3680,67.5,3820)
    # Access the two original latch nodes before the removed output buffers.
    for net,y in [('Q',4700.),('QB',4490.)]:
        raw.rect('m1',282.5,y-35,422.5,y+35);raw.via(1,352.5,y)
        raw.rect('m2',317.5,y-70,387.5,y+70);raw.via(2,352.5,y)
        raw.rect('m3',317.5,y-70,387.5,y+70)
    b=Builder(layout,'DENSE_SENSE');b.place(raw,0,-offset)
    pins={'VDD':(352.5,3800-offset,'m2'),'VSS':(705,5730-offset,'m2'),
          'IN':(185,2000-offset,'m2'),'INB':(520,2000-offset,'m2'),
          'EN':(352.5,5860-offset,'m1'),'ISO':(32.5,3750-offset,'m3'),
          'Q':(352.5,4700-offset,'m3'),'QB':(352.5,4490-offset,'m3')}
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    coords={
      'M1':[(165,4890),(295,4490),(355,4890),(625,5465)],
      'M2':[(165,4005),(295,4490),(355,4005),(352.5,1625)],
      'M3':[(540,4890),(410,4690),(355,4890),(625,5465)],
      'M4':[(540,4005),(410,4690),(355,4005),(352.5,1625)],
      'M5':[(350,2222.5),(32.5,3750),(165,2222.5),(352.5,1625)],
      'M6':[(355,3192.5),(32.5,3750),(540,3192.5),(352.5,1625)],
      'M7':[(265,5487.5),(357.5,5860),(450,5487.5),(625,5465)]}
    terminals={f'{d}@0:{t}':(x,y-offset,'m1') for d,ps in coords.items() for t,(x,y) in zip('dgsb',ps)}
    return CellView(component,b,W,6010-offset,pins,terminals,[],[],[])
