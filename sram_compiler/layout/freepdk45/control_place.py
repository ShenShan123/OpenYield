"""Hierarchical placement and metal routing of reusable control standard cells."""
from collections import defaultdict
from math import ceil,floor,hypot
from typing import Optional
from .components import CellView
from .lib import Builder,pya,u
from .router import Router
from .control_cells import HEIGHT,make_logic,make_transmission_gate

GRID=100.


class ControlRouter(Router):
    def __init__(self,builder: Builder,width: float,height: float,upper_metal: int=6,
                 overflow: Optional[tuple[float,float]]=None) -> None:
        super().__init__(builder,(1,1,int(width/GRID)-1,int(height/GRID)-1),
            grid_nm=GRID,metal_layers=tuple(f'm{i}' for i in range(4,upper_metal+1)),
            widths=tuple(145. if i<7 else 405. if i<9 else 805. for i in range(4,upper_metal+1)),
            use_tree_heuristic=True,allow_same_net_vias=True,exact_tree_heuristic=height>80000.,
            search_limit=50000 if height>80000. else None,
            prefer_deeper_paths=overflow is not None and height>80000.)
        self.clearance={};self.cut_centres=set();self.cut_exclusion=set();self.cut_count=0
        self.overflow=overflow
    def reserve(self,p,net):
        super().reserve(p,net)
        radius=3 if p[2]<3 else 8 if p[2]<5 else 16
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                if p[2]<3 and hypot(max(0,abs(dx)*GRID-145),max(0,abs(dy)*GRID-145))>=140:
                    continue
                self.clearance.setdefault((p[0]+dx,p[1]+dy,p[2]),set()).add(net)
    def available(self,p,net):
        if self.overflow is not None and p[1]*GRID>=self.overflow[0] and p[0]*GRID>=self.overflow[1]:return False
        border=5 if p[2]>=5 else 3
        if p[2]>=3 and not (border<=p[0]<=self.bounds[2]-border+1 and border<=p[1]<=self.bounds[3]-border+1):return False
        return super().available(p,net) and not (self.clearance.get(p,set())-{net})
    def neighbors(self,p,net):
        x,y,l=p
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            q=x+dx,y+dy,l
            if self.available(q,net):
                cost=1 if (l%2==0 and dy==0) or (l%2 and dx==0) else 3
                if net=='clk_buf' and l<len(self.layers)-1:cost*=30
                yield q,cost
        for other in (l-1,l+1):
            q=x,y,other
            if not (0<=other<len(self.layers) and self.available(q,net)):continue
            while self.cut_count<len(self.vias):
                _,vx,vy,up=self.vias[self.cut_count];vn=3+up
                self.cut_centres.add((vn,vx,vy));self.cut_count+=1
                radius=3 if vn<7 else 8
                self.cut_exclusion.update((vn,vx+dx,vy+dy) for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)
                    if vn>=7 or hypot(max(0,abs(dx)*GRID-140),max(0,abs(dy)*GRID-140))<160)
            key=(4+min(l,other),x,y)
            if key in self.cut_exclusion and key not in self.cut_centres:continue
            yield q,4
    def draw(self):
        regions=defaultdict(pya.Region)
        def segment(net,layer,x,y,xx,yy):
            h=self.widths[self.layers.index(layer)]/2
            regions[net,layer].insert(pya.Box(u(min(x,xx)-h),u(min(y,yy)-h),u(max(x,xx)+h),u(max(y,yy)+h)))
        for net,a,z in self.paths:
            segment(net,self.layers[a[2]],a[0]*GRID,a[1]*GRID,z[0]*GRID,z[1]*GRID)
        retained=[]
        for net,gx,gy,upper in sorted(set(self.vias)):
            n=3+upper;x,y=gx*GRID,gy*GRID;minimum=300 if n<7 else 840
            cut=140 if n<7 else 400
            same=next((v for v in retained if v[0]==net and v[3]==upper and
                       (v[1],v[2])!=(gx,gy) and
                       hypot(max(0,abs(v[1]-gx)*GRID-cut),max(0,abs(v[2]-gy)*GRID-cut))<minimum-cut),None)
            if same:
                # Replace nearby cuts by one legal cut and same-net metal on
                # both sides. The electrical connection is unchanged.
                for layer in (f'm{n}',f'm{n+1}'):segment(net,layer,x,y,same[1]*GRID,same[2]*GRID)
                continue
            retained.append((net,gx,gy,upper))
        for net,gx,gy,upper in retained:
            x,y=gx*GRID,gy*GRID;n=3+upper
            if n>=7:self.builder.sq(f'v{n}',x,y,400)
            else:self.builder.via(n,x,y)
            for layer in (f'm{n}',f'm{n+1}'):
                h=self.widths[self.layers.index(layer)]/2
                regions[net,layer].insert(pya.Box(u(x-h),u(y-h),u(x+h),u(y+h)))
        for (net,layer),region in regions.items():
            # Fill only sub-spacing notches of the same net. Foreign nets
            # cannot legally occupy these gaps; final DRC still checks them.
            radius=400 if int(layer[1:])>=9 else 200 if int(layer[1:])>=7 else 0
            if radius:region=region.merged().sized(u(radius)).sized(-u(radius))
            self.builder.cell.shapes(self.builder.li(layer)).insert(region.merged())


class StandardLibrary:
    def __init__(self,layout):self.layout=layout;self.views={}
    def build(self,c):
        if c.identity not in self.views:
            if c.name=='DFF':
                from .control_register import make_register
                v=make_register(self.layout,c)
            elif c.name=='TRANSMISSION_GATE':v=make_transmission_gate(self.layout,c)
            elif c.children:v=assemble(self,c)
            else:v=make_logic(self.layout,c)
            self.views[c.identity]=v
        return self.views[c.identity]


def assemble(library,component,row_width=26000.):
    if component.passives:raise ValueError('RC guards require the controller physical plan')
    b=Builder(library.layout,'CTRL_GROUP_'+component.identity[:16])
    child_views=[(i,library.build(i.component)) for i in component.children]
    # DFFs remain single reusable masters, with all eleven constituent cells.
    if component.name=='DFF':row_width=sum(v.width_nm for i,v in child_views)+600
    cursor=300.;row=0;terminals={};physical=[];placed=[];routed=defaultdict(list);access=[]
    def rectwire(layer,x,y,xx,yy,width=70):
        b.rect(layer,min(x,xx)-width/2,min(y,yy)-width/2,max(x,xx)+width/2,max(y,yy)+width/2)
    for inst,v in child_views:
        if cursor+v.width_nm>row_width and cursor>300:row+=1;cursor=300.
        y=300.+row*HEIGHT
        if v.standard_family=='TG':
            b.rect('pwell',cursor,y,cursor+30,y+680);b.rect('nwell',cursor,y+680,cursor+30,y+HEIGHT)
            cursor+=30
        x=cursor;b.place(v.builder,x,y)
        m=dict(zip(inst.component.ports,inst.nodes));placed.append(dict(instance=inst.name,cell=v.builder.name,x_nm=x,y_nm=y,width_nm=v.width_nm,height_nm=v.height_nm))
        terminals.update({inst.name+'/'+n:(x+px,y+py,l) for n,(px,py,l) in v.device_pins.items()})
        fs=getattr(v,'physical_fingers',None)
        if fs is None:
            fs=[dict(device=d.name,finger=0,width_nm=d.width_m*1e9,length_nm=d.length_m*1e9,model=d.model,nodes=d.nodes) for d in v.component.devices]
        for f in fs:
            f=dict(f,device=inst.name+'/'+f['device'])
            if 'gate_xy_nm' in f:f['gate_xy_nm']=(x+f['gate_xy_nm'][0],y+f['gate_xy_nm'][1])
            physical.append(f)
        for p,(px,py,l) in v.pins.items():
            net=m[p];px+=x;py+=y
            if p in ('VDD','VSS'):
                routed[net].append((px,py,l));continue
            if l=='m3':
                if v.standard_family=='TG':phase={'IN':100,'OUT':1400,'CTR_N':400,'CTR_P':1050}[p]
                else:
                    inputs=[k for k in v.pins if k not in ('Z','VDD','VSS')]
                    phase=1400 if p=='Z' else 300+350*inputs.index(p)
                yy=y+phase
                rectwire('m3',px,py,px,yy)
                xx=round(px/GRID)*GRID;yyy=round(yy/GRID)*GRID
                rectwire('m3',px,yy,xx,yy);rectwire('m3',xx,yy,xx,yyy)
                b.via(3,xx,yyy);b.rect('m3',xx-35,yyy-70,xx+35,yyy+70);b.sq('m4',xx,yyy,145)
                point=(xx,yyy,'m4')
            else:
                xx=round(px/GRID)*GRID;yy=round(py/GRID)*GRID
                rectwire(l,px,py,xx,py,145);rectwire(l,xx,py,xx,yy,145);point=(xx,yy,l)
            routed[net].append(point);access.append((net,point))
        cursor+=v.width_nm
    width=ceil((max(p['x_nm']+p['width_nm'] for p in placed)+300)/GRID)*GRID
    height=ceil(((row+1)*HEIGHT+600)/GRID)*GRID
    # Physical M1 row rails and two M2 trunks join every standard-cell tap.
    pins={}
    for net,tx,phase in [('VSS',100.,100.),('VDD',width-100.,HEIGHT-100)]:
        for r in range(row+1):
            yy=300+r*HEIGHT+phase;rectwire('m1',tx,yy,300 if tx>300 else width-300,yy)
            b.sq('v1',tx,yy,65);b.rect('m1',tx-67.5,yy-35,tx+67.5,yy+35);b.rect('m2',tx-35,yy-70,tx+35,yy+70)
        rectwire('m2',tx,300+phase,tx,300+row*HEIGHT+phase)
        pins[net]=(tx,300+phase,'m2')
    # Composite interfaces have reserved upper-metal access at the top edge.
    external={}
    for i,net in enumerate(p for p in component.ports if p not in ('VDD','VSS')):
        point=(800.+400*i,height-100.,'m6');external[net]=point
        routed[net].append(point);access.append((net,point));b.sq('m6',point[0],point[1],145)
    router=ControlRouter(b,width,height)
    # Fixed leaf M4 conductors must remain obstacles, except their own pins.
    for li,layer in enumerate(router.layers):
        reg=pya.Region(b.cell.begin_shapes_rec(b.li(layer))).merged()
        for pg in reg.each():
            bb=pg.bbox();sc=library.layout.dbu*1000
            for xi in range(floor(bb.left*sc/GRID)-1,ceil(bb.right*sc/GRID)+2):
                for yi in range(floor(bb.bottom*sc/GRID)-1,ceil(bb.top*sc/GRID)+2):router.blocked.add((xi,yi,li))
    for net,pt in access:
        q=router.point(*pt)
        for layer in range(q[2],min(3,q[2]+2)):
            qq=q[0],q[1],layer;router.blocked.discard(qq);router.reserve(qq,net)
    for net,pt in access:
        q=router.point(*pt)
        if router.clearance.get(q,set())-{net}:
            raise ValueError(f'Conflicting cell access {net} {q}: {router.clearance[q]}')
    for net in sorted((n for n in routed if n not in ('VDD','VSS')),key=lambda n:(len(routed[n]),n)):
        pts=list(dict.fromkeys(router.point(*p) for p in routed[net]));router.connect(net,pts[0],pts[1:])
        if net in component.ports:pins[net]=external[net]
    router.draw()
    for n,(x,y,l) in pins.items():b.label(l,n,x,y)
    if set(pins)!=set(component.ports):raise ValueError('Composite standard-cell interface incomplete')
    v=CellView(component,b,width,height,pins,terminals,[],router.paths,router.vias)
    v.physical_fingers=physical;v.placements=placed;v.standard_family=component.name
    return v
