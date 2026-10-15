"""Folded field-poly guard resistors and M4/M5 overlap capacitors.

These are intentional interconnect devices. LVS filters alone do not qualify
values; terminal-to-terminal resistance and capacitance require xRC checks.
"""
from math import ceil
from .components import Component,Passive,CellView
from .lib import Builder


def antenna_diode(b,x,y):
    """N+ tap in a local p-well on a guard node's M3 riser at (x, y).

    The plate and resistor of a guard drive only gates; this diffusion
    contact satisfies the antenna rule. The LVS deck defines no diode, so
    it is node diffusion, not a device.
    """
    b.rect('pwell',x-130,y-130,x+130,y+130);b.rect('nplus',x-105,y-105,x+105,y+105)
    b.rect('active',x-75,y-75,x+75,y+75);b.cont(x,y)
    b.rect('m1',x-35,y-70,x+35,y+70);b.via(1,x,y);b.rect('m2',x-35,y-70,x+35,y+70)
    b.via(2,x,y);b.rect('m3',x-35,y-70,x+35,y+70)


def make_filter(layout,resistance,capacitance=1e-15,capacitor_side_nm=2800.,diode=False):
    if abs(capacitance-1e-15)>1e-24:raise ValueError('Guard capacitor plan requires 1 fF')
    c=Component('CONTROL_RC',('IN','OUT','VSS'),(),(),
        (Passive('Rguard',('IN','OUT'),resistance),Passive('Cguard',('OUT','VSS'),capacitance)))
    b=Builder(layout,'CTRL_RC_'+c.identity[:16]+('_BLANK' if capacitor_side_nm==0 else '')+('_D' if diode else ''))
    length=round(max(140.,(resistance-335.)*50/7.8+115)/5)*5
    runs=max(2,2*ceil(length/8000));pitch=200.;lead=500.
    span=round((length-2*lead-(runs-1)*pitch)/runs/5)*5
    if not 200<=span<=4000:raise ValueError('Guard resistance outside folded plan')
    startx=300.;left=startx+lead;right=left+span;y0=400.
    def wire(layer,x,y,xx,yy,w=70):
        b.rect(layer,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    wire('poly',startx,y0,left,y0,50)
    for r in range(runs):
        y=y0+r*pitch;wire('poly',left,y,right,y,50)
        if r<runs-1:
            x=right if r%2==0 else left;wire('poly',x,y,x,y+pitch,50)
    endy=y0+(runs-1)*pitch
    total=runs*span+(runs-1)*pitch+2*lead
    endx=startx+total-length
    wire('poly',left,endy,endx,endy,50)
    for x,y,tx in [(startx,y0,4300.),(endx,endy,3900.)]:
        b.rect('poly',x-37.5,y-182.5,x+37.5,y+182.5)
        for dy in (-145.,0.,145.):
            b.cont(x,y+dy);b.via(1,x,y+dy)
        b.rect('m1',x-35,y-215,x+35,y+215)
        b.rect('m2',x-35,y-215,x+35,y+215)
        wire('m2',x,y,tx,y);b.via(2,tx,y)
        b.rect('m2',tx-35,y-70,tx+35,y+70);b.rect('m3',tx-35,y-70,tx+35,y+70)
        wire('m3',tx,y,tx,5700)
    if capacitor_side_nm:
        for layer in ('m4','m5'):b.rect(layer,500,2400,500+capacitor_side_nm,2400+capacitor_side_nm)
    b.via(3,3900,3300);b.rect('m3',3865,3230,3935,3370)
    wire('m4',3300,3300,3900,3300,140)
    if diode:antenna_diode(b,3900.,4600.)
    # Metal6 boundary pins sit above the capacitor plates.
    for x,low in [(4300.,3),(3900.,3),(3200.,5)]:
        if low==5:wire('m5',x,5000,x,5700,145)
        for n in range(low,6):
            b.via(n,x,5700)
            if n==3:b.rect('m3',x-35,5630,x+35,5770)
            else:b.sq(f'm{n}',x,5700,145)
            b.sq(f'm{n+1}',x,5700,145)
    pins={'IN':(4300,5700,'m6'),'OUT':(3900,5700,'m6'),'VSS':(3200,5700,'m6')}
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(c,b,4600,6000,pins,{},[],[],[]);v.standard_family='RC_GUARD'
    v.passive_parameters=dict(target_resistance_ohm=resistance,target_capacitance_f=capacitance,
        poly_length_nm=length,folds=runs,capacitor_side_nm=capacitor_side_nm)
    return v


def make_cap(layout,capacitor_side_nm=2800.):
    """Historical capacitor-only settling load, retained for archived layouts."""
    c=Component('CONTROL_CAP',('TOP','VSS'),(),(),(Passive('Cguard',('TOP','VSS'),1e-15),))
    b=Builder(layout,'CTRL_CAP_'+c.identity[:16])
    for layer in ('m4','m5'):b.rect(layer,500,2400,500+capacitor_side_nm,2400+capacitor_side_nm)
    b.rect('m4',3300,3230,4370,3370);b.via(3,4300,3300);b.rect('m3',4265,3230,4335,5700)
    antenna_diode(b,4300.,4600.)
    for x,low in [(4300.,3),(3200.,5)]:
        if low==5:b.rect('m5',x-72.5,5000,x+72.5,5700)
        for n in range(low,6):
            b.via(n,x,5700)
            if n==3:b.rect('m3',x-35,5630,x+35,5770)
            else:b.sq(f'm{n}',x,5700,145)
            b.sq(f'm{n+1}',x,5700,145)
    pins={'TOP':(4300,5700,'m6'),'VSS':(3200,5700,'m6')}
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(c,b,4600,6000,pins,{},[],[],[]);v.standard_family='RC_GUARD'
    v.passive_parameters=dict(target_resistance_ohm=None,target_capacitance_f=1e-15,
        poly_length_nm=0,folds=0,capacitor_side_nm=capacitor_side_nm)
    return v
