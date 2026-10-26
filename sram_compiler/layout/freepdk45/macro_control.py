"""Wires, via stacks and physical checks for the compact bank controller."""
from collections import defaultdict

from .lib import pya, u

GRID=100.
WIDTH={'m1':70.,'m2':70.,'m3':70.,'m4':140.,'m5':145.,'m6':145.,'m7':400.,'m8':405.,'m9':805.,'m10':805.}
MINW={'m1':65.,'m2':70.,'m3':70.,'m4':140.,'m5':140.,'m6':140.,'m7':400.,'m8':400.,'m9':800.,'m10':800.}
SPACE={'m1':65.,'m2':70.,'m3':70.,'m4':140.,'m5':140.,'m6':140.,'m7':400.,'m8':400.,'m9':800.,'m10':800.}
VIA={1:65.,2:70.,3:70.,4:140.,5:140.,6:140.,7:400.,8:400.,9:800.}
LANDING={4:140.,5:145.,6:145.,7:400.,8:405.,9:805.,10:805.}


def snap(v):return round(v/GRID)*GRID


_conductors={}
def bank_conductor(cell,layer,x,y):
    """Bounding box (nm) of the merged bank conductor containing a point."""
    key=cell.cell_index(),layer
    if key not in _conductors:
        _conductors[key]=pya.Region(cell.begin_shapes_rec(cell.layout().layer(*{'m4':(17,0),'m5':(19,0),'m6':(21,0)}[layer]))).merged()
    hits=[pg.bbox() for pg in _conductors[key].each() if pg.inside(pya.Point(u(x),u(y)))]
    if len(hits)!=1:raise ValueError(f'No unique bank {layer} conductor at {x},{y}')
    s=cell.layout().dbu*1000;b=hits[0]
    return b.left*s,b.bottom*s,b.right*s,b.top*s


class Planner:
    """Explicit wires and via stacks; every macro-level shape is recorded."""
    def __init__(self,builder):
        self.b=builder;self.drawn=defaultdict(pya.Region);self.segments=[]
    def rect(self,layer,x0,y0,x1,y1):
        self.b.rect(layer,x0,y0,x1,y1)
        self.drawn[layer].insert(pya.Box(u(min(x0,x1)),u(min(y0,y1)),u(max(x0,x1)),u(max(y0,y1))))
    def wire(self,net,layer,pts,width=None):
        w=(width or WIDTH[layer])/2
        for (x,y),(xx,yy) in zip(pts,pts[1:]):
            if x!=xx and y!=yy:raise ValueError(f'{net}: diagonal {layer} segment')
            self.rect(layer,min(x,xx)-w,min(y,yy)-w,max(x,xx)+w,max(y,yy)+w)
            self.segments.append((net,layer,(x,y),(xx,yy)))
    def stack(self,net,x,y,lo,hi):
        # The cuts and landings of the verified TIME_CONTROL integration.
        for n in range(lo,hi):
            h=VIA[n]/2;self.rect(f'v{n}',x-h,y-h,x+h,y+h)
            for k in (n,n+1):
                if k==1:self.rect('m1',x-67.5,y-35,x+67.5,y+35)
                elif k<4:self.rect(f'm{k}',x-35,y-70,x+35,y+70)
                else:h=LANDING[k]/2;self.rect(f'm{k}',x-h,y-h,x+h,y+h)
        self.segments.append((net,'via',(x,y,lo),(x,y,hi)))
    def check(self,cell):
        """Width/space violations touching planned shapes (Calibre is final)."""
        errors=[];ly=cell.layout()
        for layer,space in SPACE.items():
            mine=self.drawn[layer].merged()
            if mine.is_empty():continue
            full=pya.Region(cell.begin_shapes_rec(ly.layer(*{f'm{i}':(9+2*i,0) for i in range(1,11)}[layer]))).merged()
            near=mine.sized(u(space))
            for pairs in (full.space_check(u(space)),full.width_check(u(MINW[layer]))):
                for pair in pairs.each():
                    if not (pya.Region(pair.bbox()) & near).is_empty():
                        b=pair.bbox();s=ly.dbu*1000
                        errors.append(f'{layer} {b.left*s:.1f},{b.bottom*s:.1f} {b.right*s:.1f},{b.top*s:.1f}')
        if errors:raise ValueError('Planned macro routes violate spacing:\n'+'\n'.join(errors[:40]))
