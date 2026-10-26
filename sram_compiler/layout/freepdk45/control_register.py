"""Transmission-gate DFF assembled from the frozen standard-cell topology.

Adjacent data-path connections stay on M3. Two feedback tracks use M4;
clock distribution uses separate M5 legs and M6 rails. The three public
signal pins reach M7 for routing over register rows.
"""
from .components import CellView
from .control_cells import make_logic,make_transmission_gate
from .lib import Builder


def make_register(layout,c,low_metal: bool=False):
    names=['Xinv1_clk','Xinv2_D','Xtg1','Xinv3','Xinv4','Xtg2','Xinv5','Xtg3','Xinv6','Xinv7','Xtg4']
    if c.passives or [i.name for i in c.children]!=names or c.ports!=('VDD','VSS','D','Q','CLK'):
        raise ValueError('Unsupported DFF topology')
    signals=[('CLK','CLKB'),('D','D_b'),('D_b','z1','CLK','CLKB'),
             ('z1','z2'),('z2','z3'),('z3','z1','CLKB','CLK'),
             ('z2','z4'),('z4','z5','CLKB','CLK'),('z5','Q'),
             ('Q','QB'),('QB','z5','CLK','CLKB')]
    for inst,nets in zip(c.children,signals):
        if inst.nodes!=('VDD','VSS',*nets):
            raise ValueError('Unsupported DFF connectivity')
        for device in inst.component.devices:
            expected={'PMOS_VTG':500.,'NMOS_VTG':250.}.get(device.model)
            if expected is None or abs(device.width_m*1e9-expected)>1e-6:
                raise ValueError('DFF view requires frozen 500/250 nm devices')
    b=Builder(layout,('CTRL_DFF_LOW_' if low_metal else 'CTRL_DFF_')+c.identity[:16]);x=0.;dy=100.;placed={};terminals={};physical=[];cache={}
    for i in c.children:
        tg=i.component.name=='TRANSMISSION_GATE'
        key=i.component.identity
        if key not in cache:cache[key]=make_transmission_gate(layout,i.component) if tg else make_logic(layout,i.component,1600.)
        v=cache[key]
        if tg:
            b.rect('pwell',x,dy,x+30,dy+680);b.rect('nwell',x,dy+680,x+30,dy+1600);x+=30
        b.place(v.builder,x,dy);mapping=dict(zip(i.component.ports,i.nodes));placed[i.name]=(v,x,mapping)
        terminals.update({i.name+'/'+n:(xx+x,yy+dy,l) for n,(xx,yy,l) in v.device_pins.items()})
        fs=getattr(v,'physical_fingers',None) or [dict(device=d.name,finger=0,width_nm=d.width_m*1e9,length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes) for d in i.component.devices]
        for f in fs:
            f=dict(f,device=i.name+'/'+f['device'])
            if 'gate_xy_nm' in f:f['gate_xy_nm']=(x+f['gate_xy_nm'][0],dy+f['gate_xy_nm'][1])
            physical.append(f)
        x+=v.width_nm
    width=x+(500. if low_metal else 0.)
    if low_metal:
        b.rect('pwell',x,dy,width,dy+680);b.rect('nwell',x,dy+680,width,dy+1600)
    def point(n,p):
        v,x,m=placed[n];px,py,l=v.pins[p];return x+px,py+dy,l
    def wire(l,x,y,xx,yy,w=70):b.rect(l,min(x,xx)-w/2,min(y,yy)-w/2,max(x,xx)+w/2,max(y,yy)+w/2)
    def lift(x,y,lo,hi):
        for n in range(lo,hi):
            b.via(n,x,y)
            for k in (n,n+1):
                if k<4:b.rect(f'm{k}',x-35,y-70,x+35,y+70)
                else:b.sq(f'm{k}',x,y,400 if k==7 else 145)
    def local(a,p,z,q):
        va,xa,ma=placed[a];vz,xz,mz=placed[z]
        if ma[p]!=mz[q]:raise ValueError('Register local connection differs from source')
        x,y,l=point(a,p);xx,yy,ll=point(z,q);track=min(y,yy)
        wire('m3',x,y,x,track);wire('m3',x,track,xx,track);wire('m3',xx,track,xx,yy)
    for a,p,z,q in [('Xinv2_D','Z','Xtg1','IN'),('Xtg1','OUT','Xinv3','A'),
        ('Xinv3','Z','Xinv4','A'),('Xinv4','Z','Xtg2','IN'),('Xinv5','Z','Xtg3','IN'),
        ('Xtg3','OUT','Xinv6','A'),('Xinv6','Z','Xinv7','A'),('Xinv7','Z','Xtg4','IN')]:local(a,p,z,q)
    for a,p,z,q,phase in [('Xtg1','OUT','Xtg2','OUT',1100),('Xtg3','OUT','Xtg4','OUT',1100),('Xinv3','Z','Xinv5','A',700)]:
        if placed[a][2][p]!=placed[z][2][q]:raise ValueError('Register feedback differs from source')
        pts=[]
        for n,port in [(a,p),(z,q)]:
            x,y,l=point(n,port);yy=dy+phase;wire('m3',x,y,x,yy);lift(x,yy,3,4);pts.append(x)
        wire('m4',min(pts),dy+phase,max(pts),dy+phase,145)
    # The N and P clock contacts use opposite sides of each transmission gate,
    # so their M5 legs never occupy the same column when phases are exchanged.
    clocks=[]
    for n in ('Xtg1','Xtg2','Xtg3','Xtg4'):
        v,ox,m=placed[n]
        for port,phase,tx in [('CTR_N',300,ox+85),('CTR_P',1400,ox+485)]:
            x,y,l=point(n,port);yy=dy+phase;wire('m3',x,y,x,yy);lift(x,yy,3,4)
            wire('m4',x,yy,tx,yy,145);lift(tx,yy,4,5);clocks.append((m[port],tx,yy))
    for p,phase in [('A',300),('Z',1400)]:
        x,y,l=point('Xinv1_clk',p);yy=dy+phase;wire('m3',x,y,x,yy);lift(x,yy,3,5)
        clocks.append((placed['Xinv1_clk'][2][p],x,yy))
    # Column-bank supplies cross on M6. Keep private register clocks on M4
    # and expose low-metal ports so global routing chooses safe lift sites.
    clock_layer='m4' if low_metal else 'm6'
    clock_top=1700 if low_metal else 1600
    for net,xx,yy in clocks:
        if net not in ('CLK','CLKB'):raise ValueError('Unexpected register clock phase')
        ry=dy+(0 if net=='CLK' else clock_top);wire('m5',xx,yy,xx,ry,145)
        lift(xx,ry,4,5) if low_metal else lift(xx,ry,5,6)
    for ry in (dy,dy+clock_top):wire(clock_layer,100,ry,width-100,ry,145)
    pins={}
    for net,inst,port in [('D','Xinv2_D','A'),('Q','Xinv6','Z')]:
        xx,yy,l=point(inst,port);y=dy+700;wire('m3',xx,yy,xx,y);lift(xx,y,3,4 if low_metal else 7)
        if low_metal:pins[net]=(xx,y,'m4')
        else:wire('m7',xx,y,xx,dy+1400,400);pins[net]=(xx,dy+1400,'m7')
    xx=width-200
    if low_metal:
        wire('m5',xx,dy,xx,dy+1200,145);lift(xx,dy,4,5);pins['CLK']=(xx,dy+1200,'m5')
    else:
        wire('m6',xx,dy,xx,dy+1200,145);lift(xx,dy+1200,6,7)
        wire('m7',xx,dy+1200,xx,dy+1400,400);pins['CLK']=(xx,dy+1400,'m7')
    for net,phase in [('VSS',100),('VDD',1500)]:
        wire('m1',35,dy+phase,width-35,dy+phase);pins[net]=(100,dy+phase,'m1')
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    v=CellView(c,b,width,1900 if low_metal else 1800,pins,terminals,[],[],[]);v.physical_fingers=physical;v.standard_family='DFF'
    return v
