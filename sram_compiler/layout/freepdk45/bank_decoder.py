"""Left-side decoder strip with an exact 400 nm wordline interface.

Unchanged three-row leaf views are packed horizontally in eight-row bands.
The complete strip occupies exactly the data core's height. Routing uses the
space above the cells and the free M4/M5 layers, not additional shelf height.
"""
from collections import defaultdict
import heapq
from itertools import count
from math import ceil
import re

from . import bitcell as bc
from .cell_flow import leaves
from .lib import Builder, pya, u
from .router import Router, RoutingError

GRID=80.
Y_PHASE=40.


class PitchRouter(Router):
    def __init__(self,builder,bounds):
        super().__init__(builder,bounds,grid_nm=GRID,metal_layers=('m3','m4','m5','m6'),
                         widths=(70.,140.,145.,145.),
                         use_tree_heuristic=True,allow_same_net_vias=True)
        self.clearance={}
        self.fixed_access=set()
        self.upper_bus_nets=set()

    def point(self,x,y,layer=None):
        return super().point(x,y-Y_PHASE,layer)

    def reserve(self,p,net):
        super().reserve(p,net)
        radius=1 if p[2]==0 else 3
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                self.clearance.setdefault((p[0]+dx,p[1]+dy,p[2]),set()).add(net)

    def available(self,p,net):
        return super().available(p,net) and not (self.clearance.get(p,set())-{net})

    def neighbors(self,p,net):
        x,y,layer=p
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            q=(x+dx,y+dy,layer)
            if self.available(q,net):
                cost=1. if (layer%2==1 and dy==0) or (layer%2==0 and dx==0) else 3.
                if net in self.upper_bus_nets and dx and layer<3:
                    cost=12.
                yield q,cost
        for other in (layer-1,layer+1):
            if not 0<=other<len(self.layers):continue
            q=(x,y,other)
            if not self.available(q,net):continue
            if min(layer,other)==0:
                # M3's via3 enclosure is longer than its wire width.
                halo=[(x,y+d,0) for d in (-2,-1,1,2)]
                if any(self.owner.get(h,net)!=net for h in halo):continue
                if (x,y,0) not in self.fixed_access and any(h in self.blocked for h in halo):continue
            yield q,4.

    def connect(self,net,source,targets):
        # A global net's enclosing box covers most of this wide strip and is
        # a poor search bound. Exact nearest-tree L1 distance remains an
        # admissible lower bound, including the cost of changing metal layers.
        from scipy.spatial import cKDTree
        tree={source};self.reserve(source,net)
        remaining=set(targets)-tree
        serial=count()
        while remaining:
            target=min(remaining,key=lambda p:(abs(p[0]-source[0])+abs(p[1]-source[1]),p))
            nearest=cKDTree([(x,y,4*z) for x,y,z in tree])
            estimates={}
            def heuristic(p):
                if p not in estimates:
                    estimates[p]=nearest.query((p[0],p[1],4*p[2]),p=1)[0]
                return estimates[p]
            frontier=[(heuristic(target),next(serial),target)]
            costs={target:0.};parent={};closed=set();reached=None
            while frontier:
                _,_,p=heapq.heappop(frontier)
                if p in closed:continue
                closed.add(p)
                if p in tree:
                    reached=p;break
                for q,weight in self.neighbors(p,net):
                    cost=costs[p]+weight
                    if cost<costs.get(q,float('inf')):
                        costs[q]=cost;parent[q]=p
                        heapq.heappush(frontier,(cost+heuristic(q),next(serial),q))
            if reached is None:
                blockers=set()
                for p in closed:blockers.update(self.clearance.get(p,set())-{net})
                raise RoutingError(net,target,blockers)
            path=[reached]
            while path[-1]!=target:path.append(parent[path[-1]])
            for p in path:self.reserve(p,net)
            for a,b in zip(path,path[1:]):
                if a[2]==b[2]:self.paths.append((net,a,b))
                else:
                    upper=max(a[2],b[2]);x,y=a[:2]
                    self.vias.append((net,x,y,upper))
                    if upper==1:
                        for d in (-2,-1,1,2):self.clearance.setdefault((x,y+d,0),set()).add(net)
            tree.update(path);remaining-=tree

    def draw(self):
        # Own drawing uses the physical grid and phase throughout. The legacy
        # router's 280 nm same-net stitching is intentionally not used here.
        regions=defaultdict(pya.Region)
        def rect(net,layer,x0,y0,x1,y1):
            regions[net,layer].insert(pya.Box(u(x0),u(y0),u(x1),u(y1)))
        for net,a,b in self.paths:
            x0,y0=a[0]*GRID,a[1]*GRID+Y_PHASE
            x1,y1=b[0]*GRID,b[1]*GRID+Y_PHASE
            half=self.widths[a[2]]/2
            rect(net,self.layers[a[2]],min(x0,x1)-half,min(y0,y1)-half,
                 max(x0,x1)+half,max(y0,y1)+half)
        for net,x,y,upper in set(self.vias):
            x,y=x*GRID,y*GRID+Y_PHASE;n=self.via_number(upper)
            self.builder.via(n,x,y)
            if n==3:
                rect(net,'m3',x-35,y-70,x+35,y+70)
                rect(net,'m4',x-70,y-70,x+70,y+70)
            else:
                half=70 if n==4 else 72.5
                rect(net,f'm{n}',x-half,y-half,x+half,y+half)
                rect(net,f'm{n+1}',x-72.5,y-72.5,x+72.5,y+72.5)
        for (_,layer),region in regions.items():
            self.builder.cell.shapes(self.builder.li(layer)).insert(region.merged())


def build_decoder(library,source,specs):
    rows=source.rows
    if rows<8 or rows%8:
        raise ValueError('Pitch-matched decoder requires a row count divisible by eight')
    items=[]
    for name in ['XDECODER',*[f'XWL_DRV_{r}' for r in range(rows)]]:
        for path,leaf,_ in leaves(specs[name],path=name):
            items.append((path,leaf,source.instances[path]['pins']))
    producers={pins['Z']:(path,leaf,pins) for path,leaf,pins in items}
    consumers=defaultdict(set)
    def trace(net,row,seen):
        if net in seen or net not in producers:return
        seen.add(net)
        path,leaf,pins=producers[net];consumers[path].add(row)
        for p,n in pins.items():
            if p not in ('Z','VDD','VSS'):trace(n,row,seen)
    for r in range(rows):trace(source.instances[f'XWL_DRV_{r}']['pins']['Z'],r,set())
    # Cascaded decoders retain disabled predecode branches in the native
    # source. Keep these transistors even when no data wordline consumes them.
    groups=[[] for _ in range(rows//8)]
    # Source ordering preserves local decoder cones; drivers are inserted
    # immediately after their corresponding decoder output.
    ordered=[];added=set()
    outputs={source.instances['XDECODER']['pins'][f'WL{r}']:r for r in range(rows)}
    for item in items:
        if not item[0].startswith('XDECODER/'):continue
        ordered.append(item)
        r=outputs.get(item[2]['Z'])
        if r is not None:
            ordered.extend(i for i in items if i[0].startswith(f'XWL_DRV_{r}/'));added.add(r)
    if len(added)!=rows:raise ValueError('Cannot order every decoder output and driver')
    for item in ordered:
        used=consumers[item[0]]
        if used:
            group=min(used)//8
        else:
            branch=re.search(r'/XAND(?:_EN)?(\d+)/',item[0])
            group=int(branch[1])%len(groups) if branch else 0
        groups[group].append(item)
    cell=Builder(library.layout,'ROW_PITCH_DECODER')
    placed=[];terminals={};routed=defaultdict(list);occupied=[]
    for group,items in enumerate(groups):
        cursor=1600.
        y=group*8*bc.H+Y_PHASE
        for path,leaf,mapping in items:
            view=library.build(leaf,'row')
            phase=next(iter(view.pins.values()))[0]
            x=ceil((cursor+phase)/GRID)*GRID-phase
            cell.place(view.builder,x,y)
            occupied.append((x,y,x+view.width_nm,y+view.height_nm))
            placed.append(dict(instance=path,cell=view.builder.name,x_nm=x,y_nm=y,
                width_nm=view.width_nm,height_nm=view.height_nm,core_rows=sorted(consumers[path]),
                band_first_row=group*8,band_row_span=8,connections=mapping))
            for name,(px,py,layer) in view.device_pins.items():terminals[path+'/'+name]=(x+px,y+py,layer)
            for p,(px,py,layer) in view.pins.items():
                routed[mapping[p]].append((x+px,y+py,'m3'))
                cell.rect('m2',x+px-35,y+py-70,x+px+35,y+py+70)
                cell.via(2,x+px,y+py)
                cell.rect('m3',x+px-35,y+py-70,x+px+35,y+py+70)
            cursor=x+view.width_nm+640
    width=ceil((max(b[2] for b in occupied)+1600)/GRID)*GRID
    height=rows*bc.H
    router=PitchRouter(cell,(1,1,int(width/GRID)-1,int((height-160)/GRID)))
    # Reserve the complete lower-metal cell footprint. Parents route through
    # named access sites, with M4/M5 available above the leaf geometry.
    for x0,y0,x1,y1 in occupied:
        for xi in range(int((x0-120)/GRID),ceil((x1+120)/GRID)+1):
            for yi in range(int((y0-Y_PHASE-120)/GRID),ceil((y1-Y_PHASE+120)/GRID)+1):
                router.blocked.add((xi,yi,0))
    for net,points in routed.items():
        for p in points:
            q=router.point(*p);router.blocked.discard(q);router.fixed_access.add(q)
            # Protect a complete upper-metal escape before global buses are
            # routed. Otherwise an early M5/M6 bus can cage a later leaf pin.
            for layer in range(len(router.layers)):
                router.reserve((q[0],q[1],layer),net)
    inputs=['VDD','VSS']+[source.instances['XDECODER']['pins'][p]
        for p in specs['XDECODER'].ports if p.startswith('A')]+['wl_en']
    pins={}
    for i,net in enumerate(inputs):
        p=(320.,200.+i*320,'m5');pins[net]=p;routed[net].append(p);router.reserve(router.point(*p),net)
        router.reserve(router.point(p[0],p[1],'m6'),net)
    for r in range(rows):
        net=source.instances[f'XWL_DRV_{r}']['pins']['Z']
        p=(width-320,bc.Y_WL+r*bc.H,'m4')
        routed[net].append(p);router.reserve(router.point(*p),net)
        # A local stack at the right edge exposes the physical core's M2 WL.
        x,y,_=p
        router.reserve(router.point(x,y,'m3'),net)
        for n in (2,3):cell.via(n,x,y)
        for layer in ('m2','m3'):cell.rect(layer,x-35,y-70,x+35,y+70)
        cell.sq('m4',x,y,140)
        pins[net]=(x,y,'m2')
    router.upper_bus_nets={n for n,p in routed.items() if len(p)>4 and n not in ('VDD','VSS')}
    order=sorted(routed,key=lambda n:(n not in ('VDD','VSS'),n not in router.upper_bus_nets,n not in pins,n))
    for i,net in enumerate(order):
        pts=list(dict.fromkeys(router.point(*p) for p in routed[net]))
        if len(pts)>1:router.connect(net,pts[0],pts[1:])
        if i%25==0:print('decoder routed',i,'/',len(order),net,flush=True)
    router.draw()
    for net,(x,y,layer) in pins.items():cell.label(layer,net,x,y)
    cell.rect('bnd',0,0,width,height)
    return cell,dict(width_nm=width,height_nm=height,pins=pins,device_pins=terminals,
        placements=placed,band_rows=8,row_pitch_nm=bc.H,wordline_phase_nm=bc.Y_WL,
        routing_grid_nm=GRID,routing_y_phase_nm=Y_PHASE,routed_paths=router.paths,routed_vias=router.vias)
