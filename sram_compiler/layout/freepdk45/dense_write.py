"""Compact paired-column write-driver view with unchanged source W/L.

The two tri-state branches share device strips. The enable/data inverters use
one short common channel; buffer devices are separate physical components.
"""
from .colperi import Col
from .components import CellView
from .lib import Builder,pya,u

WIDTH=1365.
HEIGHT=2565.
Y_BOTTOM=-5195.
Y_EN=-2665.


def make_write(layout,component):
    from .dense_contract import require_devices,WRITE
    require_devices(component,WRITE)
    b=Builder(layout,'DENSE_WRITE_RAW');c=Col(b)
    b.via=lambda n,x,y:b.sq(f'v{n}',x,y,65 if n<3 else 70)
    def m1(x,y,h=70):b.rect('m1',x-32.5,y-h,x+32.5,y+h)
    def supply_via(x,y):
        b.via(1,x,y);b.rect('m1',x-67.5,y-35,x+67.5,y+35)
    wells=[('pwell',-3262.5,-2665),('nwell',-4822.5,-3262.5),('pwell',Y_BOTTOM,-4822.5)]
    for layer,lo,hi in wells:b.rect(layer,-30,lo,1335,hi)
    b.rect('vtg',-30,Y_BOTTOM,1335,-2665)
    for y in (-3710,-4535,-2995,-5160):b.rect('m2',-30,y-35,1335,y+35)
    b.rect('poly',0,Y_EN-25,1305,Y_EN+25)
    b.rect('m2',-30,Y_EN-35,1335,Y_EN+35)
    # N pull-down branches: BL - EN - int2 - DINB - VSS - DIN - int4 - EN - BLB.
    cx=[272.5,462.5,652.5,842.5,1032.5]
    nr=c.strip(cx,[(360,-2740),(360,-2815),(360,-2815),(360,-2740)])
    ns={}
    for i,x in enumerate(cx):
        lo,hi=nr[i];ns[x]=c.conts(x,lo+100 if 0<i<4 else lo,hi)
    for x in (367.5,937.5):c.gate(x,-3150,Y_EN+25)
    c.pcont(1205,Y_EN);c.v1(1205,Y_EN)
    b.rect('m1',1137.5,Y_EN-32.5,1272.5,Y_EN+32.5)
    for x in (462.5,652.5,842.5):c.strap(x,min(ns[x])-67.5,max(ns[x])+67.5)
    supply_via(652.5,-2995)
    # P pull-up branches, with independent inverted enable.
    pr=c.strip(cx,[(720,-3350)]*4);ps={}
    for i,x in enumerate(cx):
        lo,hi=pr[i];ps[x]=c.conts(x,lo,hi-100 if 0<i<4 else hi)
    for x in (557.5,747.5):c.gate(x,-4120,-2765)
    for x in (367.5,937.5):
        c.gate(x,-4192.5,-3300);c.pcont(x,-4192.5);m1(x,-4192.5);c.v1(x,-4192.5)
        b.rect('m2',x-35,-4262.5,x+35,-4122.5)
    for x in (462.5,652.5,842.5):c.strap(x,min(ps[x])-67.5,max(ps[x])+67.5)
    supply_via(652.5,-3710)
    for x in (272.5,1032.5):c.strap(x,min(ps[x])-67.5,max(ns[x])+67.5)
    # DINB and DIN controls in the N/P channel.
    for x in (557.5,747.5):
        c.pcont(x,-3262.5);m1(x,-3262.5);c.v12(x,-3262.5)
    # Two inverters, with exact 360/180 nm source widths.
    cxs=[77.5,302.5,527.5];gxs=[207.5,397.5]
    rp=c.strip(cxs,[(360,-4355)]*2);rn=c.strip(cxs,[(180,-4930)]*2)
    ip={x:c.conts(x,rp[i][0]+(20 if i==1 else 0),rp[i][1]) for i,x in enumerate(cxs)}
    inn={x:c.conts(x,*rn[i]) for i,x in enumerate(cxs)}
    for x in gxs:c.gate(x,-5160,-4305)
    for x in (77.5,527.5):c.strap(x,min(inn[x])-67.5,max(ip[x])+67.5,w=65)
    c.strap(302.5,min(ip[302.5])-67.5,max(ip[302.5])+67.5);supply_via(302.5,-4535)
    c.strap(302.5,-5195,max(inn[302.5])+67.5);supply_via(302.5,-5160)
    # Inverted enable rises directly to the P switches on M1/M2.
    c.strap(77.5,max(ip[77.5]),-4122.5)
    c.v1(80,-4192.5);m1(80,-4192.5)
    b.rect('m2',45,-4262.5,115,-4122.5)
    c.m2(45,1007.5,-4192.5)
    # Inverter gates: M2 access, M3 routing clear of the internal DINB trunk.
    for x in gxs:
        c.pcont(x,-4822.5);m1(x,-4822.5);c.v1(x,-4822.5)
        b.rect('m2',x-35,-4892.5,x+35,-4752.5) if x==207.5 else b.rect('m2',x-70,-4857.5,x+70,-4787.5)
    c.m2(172.5,237.5,-4822.5);b.via(2,220,-4822.5)
    b.rect('m3',185,-4892.5,255,-4752.5)
    b.rect('m3',185,-4857.5,290,-4787.5)
    b.rect('m2',185,-4892.5,255,-4752.5)
    c.m3(255,-4892.5,Y_EN+35);b.via(2,255,Y_EN)
    b.rect('m3',185,Y_EN-35,325,Y_EN+35)
    b.rect('m2',185,Y_EN-35,325,Y_EN+35)
    b.via(2,397.5,-4822.5);c.m3(397.5,-4890,-4755)
    b.rect('m3',362.5,-4857.5,782.5,-4787.5)
    c.m3(747.5,-4822.5,-3192.5)
    # DINB output access uses horizontal landings to avoid consuming another
    # vertical input-routing row in the inverter channel.
    y=-4675.
    c.v1(532.5,y);m1(532.5,y)
    b.rect('m2',462.5,y-35,627.5,y+35);b.via(2,557.5,y)
    b.rect('m3',487.5,y-35,627.5,y+35);c.m3(557.5,y,-3192.5)
    # Independent supplies; taps reside beside the two inverter transistors.
    for kind,x,y in [('p',1205,-2995),('n',105,-3710),('p',1205,-5020)]:
        c.tap(kind,x-50,y-45,x+50,y+45,y,[x]);m1(x,y)
        if y==-5020:
            c.strap(x,-5195,y+67.5);supply_via(x,-5160)
        else:supply_via(x,y)
    for y in (-3710,-4535):
        b.via(2,105,y);c.m3(105,y-70,y+70)
    c.m3(105,-4605,-3640)
    for y in (-2995,-5160):
        b.via(2,1205,y);b.rect('m3',1135,y-35,1275,y+35)
    c.m3(1205,-5160,-2995)
    # Bitline access sites avoid all internal M3 control/supply columns.
    for x,tx in [(272.5,397.5),(1032.5,907.5)]:
        y=-3210.;c.v1(x,y);b.rect('m1',x-67.5,y-35,x+67.5,y+35)
        b.rect('m2',min(x,tx)-35,y-70,max(x,tx)+35,y+70)
        b.via(2,tx,y);c.m3(tx,y-70,y+70)
    # Implant coverage follows the wells, with local body-tap exceptions.
    nw=pya.Region();pw=pya.Region()
    for layer,lo,hi in wells:
        r=pya.Region(pya.Box(u(-30),u(lo),u(1335),u(hi)))
        if layer=='nwell':nw+=r
        else:pw+=r
    nt=pya.Region([pya.Box(*[u(v) for v in q]) for q in c.ntap])
    pt=pya.Region([pya.Box(*[u(v) for v in q]) for q in c.ptap])
    b.cell.shapes(b.li('pplus')).insert(((nw-nt)+pt).merged())
    b.cell.shapes(b.li('nplus')).insert(((pw-pt)+nt).merged())
    cell=Builder(layout,'DENSE_WRITE');cell.place(b,30,-Y_BOTTOM)
    rawpins={'VDD':(105,-3710,'m3'),'VSS':(1205,-5160,'m3'),
             'EN':(255,-4822.5,'m3'),'DIN':(747.5,-4822.5,'m3'),
             'BL':(397.5,-3210,'m3'),'BLB':(907.5,-3210,'m3')}
    pins={n:(x+30,y-Y_BOTTOM,l) for n,(x,y,l) in rawpins.items()}
    for n,(x,y,l) in pins.items():cell.label(l,n,x,y)
    pc=lambda x:(x,ps[x][0])
    nc=lambda x:(x,ns[x][0])
    pbody=(105,-3710);nbody=(1205,-2995);ibody=(1205,-5020)
    t={
      'M1':[(527.5,ip[527.5][0]),(397.5,-4822.5),(302.5,ip[302.5][0]),pbody],
      'M2':[(527.5,inn[527.5][0]),(397.5,-4822.5),(302.5,inn[302.5][0]),ibody],
      'M3':[(77.5,ip[77.5][0]),(207.5,-4822.5),(302.5,ip[302.5][0]),pbody],
      'M4':[(77.5,inn[77.5][0]),(207.5,-4822.5),(302.5,inn[302.5][0]),ibody],
      'M5':[pc(462.5),(557.5,-3262.5),pc(652.5),pbody],
      'M6':[pc(272.5),(367.5,-4192.5),pc(462.5),pbody],
      'M7':[nc(272.5),(1205,Y_EN),nc(462.5),nbody],
      'M8':[nc(462.5),(557.5,-3262.5),nc(652.5),nbody],
      'M9':[pc(842.5),(747.5,-3262.5),pc(652.5),pbody],
      'M10':[pc(1032.5),(937.5,-4192.5),pc(842.5),pbody],
      'M11':[nc(1032.5),(1205,Y_EN),nc(842.5),nbody],
      'M12':[nc(842.5),(747.5,-3262.5),nc(652.5),nbody]}
    terminals={f'{d}@0:{pin}':(x+30,y-Y_BOTTOM,'m1') for d,ps in t.items() for pin,(x,y) in zip('dgsb',ps)}
    return CellView(component,cell,WIDTH,HEIGHT,pins,terminals,[],[],[])
