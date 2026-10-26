"""Hierarchical placement and metal routing of reusable control standard cells."""
from collections import defaultdict
from math import ceil,floor,hypot
from typing import Optional
from .components import CellView
from .lib import Builder,pya,u
from .router import Router,VIA4_HALO
from .control_cells import HEIGHT,make_logic,make_transmission_gate

GRID=100.


class ControlRouter(Router):
    def __init__(self,builder: Builder,width: float,height: float,upper_metal: int=6,
                 profile: Optional[list[tuple[float,float,float]]]=None) -> None:
        super().__init__(builder,(1,1,int(width/GRID)-1,int(height/GRID)-1),
            grid_nm=GRID,metal_layers=tuple(f'm{i}' for i in range(4,upper_metal+1)),
            widths=tuple(145. if i<7 else 405. if i<9 else 805. for i in range(4,upper_metal+1)),
            use_tree_heuristic=True,allow_same_net_vias=True,exact_tree_heuristic=height>80000.,
            search_limit=50000 if height>80000. else None,
            prefer_deeper_paths=profile is not None and height>80000.)
        self.clearance={};self.cut_centres=set();self.cut_exclusion=set();self.cut_count=0
        # profile: left-aligned (y0, y1, width) rectangles that bound routing.
        self.reach={}
        for y0,y1,w in profile or ():
            for yi in range(ceil(y0/GRID),floor(y1/GRID)+1):
                self.reach[yi]=max(self.reach.get(yi,-1),floor(w/GRID)-1)
        self.profile=profile is not None
        # Congestion history: point -> {net: cost} charged to every other net.
        self.history={}
        # Cut keys (via number, x, y) too close to a fixed cut drawn outside
        # the router (register pin lifts, tapped edge pins): no via there.
        self.fixed_cuts=set()
        # Fixed accesses whose metal is drawn outside the router.
        self.fixed_points=set()
        # Net -> cost factor on M4-M7 steps: long runs of these nets go to
        # M8/M9, which count toward no gate's antenna before the net joins.
        self.lift={}
    def halo(self,p):
        """Grid points within p's same-layer clearance."""
        radius=3 if p[2]<3 else 8 if p[2]<5 else 16
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                if p[2]<3 and hypot(max(0,abs(dx)*GRID-145),max(0,abs(dy)*GRID-145))>=140:
                    continue
                yield (p[0]+dx,p[1]+dy,p[2])
    def reserve(self,p,net):
        super().reserve(p,net)
        for q in self.halo(p):self.clearance.setdefault(q,set()).add(net)
    # Local rip-up and reroute (centre frames). base_owner/base_clearance are
    # the state before any rippable net was routed: fixed accesses and clock.
    def culprits(self,net,source,target,base_owner,base_clearance,penalty=30.,limit=3000000,ripped=None,margin=60,
                 avoid=frozenset()):
        """Routed nets on the cheapest source-target path that may cross
        routed metal, clearance and via cuts, but never fixed ones, the
        clock tree routed before them or nets in avoid. Each crossed point
        costs penalty per net, scaled by how often that net was ripped
        already. The search stays within margin grid steps of the
        source-target box. Returns (nets, path); both are empty without such
        a path."""
        import heapq
        cut_nets=defaultdict(set)
        for other,x,y,upper in self.vias:
            vn=3+upper;radius=3 if vn<7 else 8
            for dx in range(-radius,radius+1):
                for dy in range(-radius,radius+1):
                    if vn<7 and hypot(max(0,abs(dx)*GRID-140),max(0,abs(dy)*GRID-140))>=160:continue
                    cut_nets[vn,x+dx,y+dy].add(other)
        xmin,ymin,xmax,ymax=self.bounds
        wx0,wx1=min(source[0],target[0])-margin,max(source[0],target[0])+margin
        wy0,wy1=min(source[1],target[1])-margin,max(source[1],target[1])+margin
        def soft(q):
            x,y,l=q
            if not (0<=l<len(self.layers) and xmin<=x<=xmax and ymin<=y<=ymax) or q in self.blocked:return None
            if not (wx0<=x<=wx1 and wy0<=y<=wy1):return None
            if self.profile and x>self.reach.get(y,-1):return None
            border=5 if l>=5 else 3
            if l>=3 and not (border<=x<=xmax-border+1 and border<=y<=ymax-border+1):return None
            o=self.owner.get(q);found=set()
            if o is not None and o!=net:
                if q in base_owner:return None
                found.add(o)
            c=self.clearance.get(q,set())-{net}
            if c&base_clearance.get(q,set()):return None
            found|=c;found|=self.via_halos.get(q,set())-{net}
            return None if found&avoid else found
        def h(p):return abs(p[0]-target[0])+abs(p[1]-target[1])+4*abs(p[2]-target[2])
        def crossing(p,q):
            found=soft(q)
            if found is None:return None
            if q[2]!=p[2]:
                if (4+min(p[2],q[2]),p[0],p[1]) in self.fixed_cuts:return None
                others=cut_nets.get((4+min(p[2],q[2]),p[0],p[1]),set())-{net}
                if 'clk_buf' in others or others&avoid:return None  # base vias: the clock tree
                found=found|others
            return found
        dist={source:0.};parent={};heap=[(h(source),0.,source)];seen=0
        while heap and seen<limit:
            _,d,p=heapq.heappop(heap)
            if p==target:break
            if d>dist[p]:continue
            seen+=1;x,y,l=p
            steps=[((x+dx,y+dy,l),1. if (l%2==0 and dy==0) or (l%2 and dx==0) else 3.) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))]
            steps+=[((x,y,o),4.) for o in (l-1,l+1)]
            for q,step in steps:
                found=crossing(p,q)
                if found is None:continue
                nd=d+step+penalty*sum(1+(ripped or {}).get(n,0) for n in found)
                if nd<dist.get(q,float('inf')):
                    dist[q]=nd;parent[q]=p;heapq.heappush(heap,(nd+h(q),nd,q))
        if target not in parent:return set(),[]
        path=[target]
        while path[-1]!=source:path.append(parent[path[-1]])
        nets=set()
        for q,p in zip(path,path[1:]):nets|=crossing(p,q)
        return nets,path
    def rip(self,net,base_owner,base_clearance):
        """Remove net's routing, keeping its fixed accesses."""
        removed=[p for p,n in self.owner.items() if n==net and p not in base_owner]
        for p in removed:del self.owner[p]
        for p in removed:
            for q in self.halo(p):
                c=self.clearance.get(q)
                if c and net in c and net not in base_clearance.get(q,()):c.discard(net)
        for _,x,y,upper in [v for v in self.vias if v[0]==net]:
            if self.via_number(upper)>=4:
                for dx,dy in VIA4_HALO:
                    h=self.via_halos.get((x+dx,y+dy,upper))
                    if h:h.discard(net)
        self.paths=[s for s in self.paths if s[0]!=net]
        self.vias=[v for v in self.vias if v[0]!=net]
        # Cut exclusions rebuild lazily from the remaining vias.
        self.cut_centres=set();self.cut_exclusion=set();self.cut_count=0
    def available(self,p,net):
        if self.profile and p[0]>self.reach.get(p[1],-1):return False
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
                if self.lift and l<len(self.layers)-2:cost*=self.lift.get(net,1)
                if self.history:cost+=sum(c for n,c in self.history.get(q,{}).items() if n!=net)
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
            if (key in self.cut_exclusion and key not in self.cut_centres) or key in self.fixed_cuts:continue
            yield q,4+(sum(c for n,c in self.history.get(q,{}).items() if n!=net) if self.history else 0)
    def exclude_cuts(self,cuts) -> None:
        """Keep router cuts at via spacing from fixed cuts (number, x_nm, y_nm)."""
        for vn,x,y in cuts:
            gx,gy=round(x/GRID),round(y/GRID);radius=3 if vn<7 else 8
            for dx in range(-radius,radius+1):
                for dy in range(-radius,radius+1):
                    if (dx or dy) and (vn>=7 or hypot(max(0,abs(dx)*GRID-140),max(0,abs(dy)*GRID-140))<160):
                        self.fixed_cuts.add((vn,gx+dx,gy+dy))
    def heal(self) -> list:
        """Join same-net points that the router left too close.

        Clearance keeps other nets apart, but one net may pass within two grid
        steps of its own metal without touching it, leaving a gap below the
        140 nm spacing. Such a pair gets an L of free points between them, or
        their whole bounding box. Two cuts of one net that draw() will merge
        fill their bounding box on both layers; those points are claimed here
        or reported. On M7-M9, two ends of one net that would meet within
        spacing or through a sub-width bridge are brought level. Returns the
        gaps left open as (net, p, q).
        """
        def joined(p,q,own):
            # A monotone owned path from p to q inside their bounding box.
            sx=1 if q[0]>p[0] else -1;sy=1 if q[1]>p[1] else -1;seen={p};stack=[p]
            while stack:
                c=stack.pop()
                if c==q:return True
                for n in ((c[0]+sx,c[1],c[2]),(c[0],c[1]+sy,c[2])):
                    if n in own and n not in seen and min(p[0],q[0])<=n[0]<=max(p[0],q[0]) \
                            and min(p[1],q[1])<=n[1]<=max(p[1],q[1]):
                        seen.add(n);stack.append(n)
            return False
        # Only drawn metal counts: an owned point may be a reservation (the M5
        # above a pin access) that no route used.
        drawn=defaultdict(set)
        for net,a,b in self.paths:
            for x in range(min(a[0],b[0]),max(a[0],b[0])+1):
                for y in range(min(a[1],b[1]),max(a[1],b[1])+1):drawn[net].add((x,y,a[2]))
        for net,x,y,upper in self.vias:drawn[net].update(((x,y,upper),(x,y,upper-1)))
        for p in self.fixed_points:drawn[self.owner[p]].add(p)
        left=[]
        cuts=defaultdict(list)
        for net,x,y,upper in self.vias:cuts[net,upper].append((x,y))
        for (net,upper),points in cuts.items():
            n=3+upper;minimum=300 if n<7 else 840;cut=140 if n<7 else 400
            for i,(x,y) in enumerate(points):
                for xx,yy in points[i+1:]:
                    if hypot(max(0,abs(x-xx)*GRID-cut),max(0,abs(y-yy)*GRID-cut))>=minimum-cut:continue
                    box=[(cx,cy,l) for cx in range(min(x,xx),max(x,xx)+1) for cy in range(min(y,yy),max(y,yy)+1)
                         for l in (upper-1,upper)]
                    if all(self.owner.get(c)==net or self.available(c,net) for c in box):
                        for c in box:
                            if self.owner.get(c)!=net:self.reserve(c,net)
                            drawn[net].add(c)
                    else:left.append((net,(x,y,upper),(xx,yy,upper)))
        for l in range(min(3,len(self.layers))):
            nets={n:{p for p in ps if p[2]==l} for n,ps in drawn.items()}
            for net,own in nets.items():
                for p in sorted(own):
                    for dx in range(-2,3):
                        for dy in range(-2,3):
                            q=(p[0]+dx,p[1]+dy,l)
                            if (dx,dy)<=(0,0) or abs(dx)+abs(dy)<2 or q not in own or joined(p,q,own):continue
                            xs=range(p[0],q[0]+(1 if dx>=0 else -1),1 if dx>=0 else -1)
                            ys=range(p[1],q[1]+(1 if dy>=0 else -1),1 if dy>=0 else -1)
                            options=[[(x,p[1],l) for x in xs]+[(q[0],y,l) for y in ys],
                                     [(p[0],y,l) for y in ys]+[(x,q[1],l) for x in xs],
                                     [(x,y,l) for x in xs for y in ys]]
                            for cells in options:
                                if all(c in own or self.owner.get(c)==net or self.available(c,net) for c in cells):
                                    for c in cells:
                                        if c not in own:self.reserve(c,net);own.add(c)
                                        self.paths.append((net,c,c))
                                    for a,b in zip(cells,cells[1:]):
                                        if abs(a[0]-b[0])+abs(a[1]-b[1])==1:self.paths.append((net,a,b))
                                    break
                            else:left.append((net,p,q))
        # M7-M9: wires are as wide as their spacing, so two ends of one net
        # (an M6 jog between two M7 runs) meet within spacing diagonally, or
        # across a gap that draw()'s same-net closing bridges over less than
        # the minimum width. Extending one end level with the other leaves a
        # straight gap that the closing fills at full width; otherwise an L
        # of free points.
        def unsafe(dx,dy,w,space):
            gx,gy=abs(dx)*GRID-w,abs(dy)*GRID-w
            if gx>=0 and gy>=0:return hypot(gx,gy)<space
            if not (dx and dy):return False
            if gx<0 and gy<0:return hypot(gx,gy)<space
            return (gy if gx<0 else gx)<space and -min(gx,gy)<space
        for l in range(3,len(self.layers)):
            w=self.widths[l];space=400. if l<5 else 800.
            for net,ps in drawn.items():
                own={p for p in ps if p[2]==l}
                def corner(p):
                    nb={(dx,dy) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)) if (p[0]+dx,p[1]+dy,l) in own}
                    return len(nb)<2 or (len(nb)==2 and nb not in ({(1,0),(-1,0)},{(0,1),(0,-1)}))
                ends=sorted(p for p in own if corner(p))
                for i,p in enumerate(ends):
                    for q in ends[i+1:]:
                        dx,dy=q[0]-p[0],q[1]-p[1]
                        if not unsafe(dx,dy,w,space) or joined(p,q,own):continue
                        sx=1 if dx>0 else -1;sy=1 if dy>0 else -1
                        options=[[(p[0],y,l) for y in range(p[1],q[1]+sy,sy)],[(x,p[1],l) for x in range(p[0],q[0]+sx,sx)],
                                 [(q[0],y,l) for y in range(q[1],p[1]-sy,-sy)],[(x,q[1],l) for x in range(q[0],p[0]-sx,-sx)],
                                 [(x,p[1],l) for x in range(p[0],q[0]+sx,sx)]+[(q[0],y,l) for y in range(p[1],q[1]+sy,sy)],
                                 [(p[0],y,l) for y in range(p[1],q[1]+sy,sy)]+[(x,q[1],l) for x in range(p[0],q[0]+sx,sx)]]
                        for cells in options:
                            if all(c in own or self.owner.get(c)==net or self.available(c,net) for c in cells):
                                for c in cells:
                                    if c not in own:self.reserve(c,net);own.add(c)
                                for a,b in zip(cells,cells[1:]):
                                    if a!=b:self.paths.append((net,a,b))
                                break
                        else:left.append((net,p,q))
        return left
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
    """Reusable views; row_height=1365 builds gates on the decoder's row rails."""
    def __init__(self,layout,row_height=None,compact_registers: bool=False):
        self.layout=layout;self.views={};self.row_height=row_height;self.compact_registers=compact_registers
    def build(self,c):
        if self.row_height is not None:
            if c.identity not in self.views:self.views[c.identity]=make_logic(self.layout,c,self.row_height)
            return self.views[c.identity]
        if c.identity not in self.views:
            if c.name=='DFF':
                from .control_register import make_register
                v=make_register(self.layout,c,low_metal=self.compact_registers)
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
