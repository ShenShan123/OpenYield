"""Deterministic three-metal maze routing for physical calibration references."""
from __future__ import annotations

import heapq
from itertools import count
from math import isfinite

from .lib import Builder

GRID = 280.0
LAYERS = ('m3', 'm4', 'm5')
WIDTHS = (70., 140., 140.)
VIA4_HALO = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if dx or dy)


class RoutingError(RuntimeError):
    """An unroutable terminal and the nets enclosing its reachable region."""
    def __init__(self, net, target, blockers):
        super().__init__(f'Unable to route {net} terminal {target}; expand routing channels')
        self.net, self.target, self.blockers = net, target, blockers


class Router:
    def __init__(self, builder, bounds, use_tree_heuristic=False, allow_same_net_vias=False,
                 join_diagonal_vias=False, grid_nm=GRID, metal_layers=LAYERS, widths=WIDTHS):
        if (len(metal_layers)!=len(widths) or not metal_layers or
                not isfinite(grid_nm) or grid_nm<=0 or any(not isfinite(w) or w<=0 for w in widths)):
            raise ValueError('Invalid routing stack')
        if any(int(b[1:])!=int(a[1:])+1 for a,b in zip(metal_layers,metal_layers[1:])):
            raise ValueError('Routing metals must be consecutive')
        self.grid, self.layers, self.widths = grid_nm, tuple(metal_layers), tuple(widths)
        self.builder = builder
        self.bounds = bounds
        self.owner = {}
        self.paths = []
        self.vias = []
        self.blocked = set()
        self.use_tree_heuristic = use_tree_heuristic
        self.allow_same_net_vias = allow_same_net_vias
        self.via_halos = {}
        self.join_diagonal_vias = join_diagonal_vias
        self.wide_clearance = {}

    def point(self, x, y, layer=None):
        layer = self.layers[0] if layer is None else layer
        if abs(x / self.grid - round(x / self.grid)) > 1e-8 or abs(y / self.grid - round(y / self.grid)) > 1e-8:
            raise ValueError(f'Off routing grid: {x}, {y}')
        return (round(x / self.grid), round(y / self.grid), self.layers.index(layer))

    def via_number(self, upper):
        return int(self.layers[upper-1][1:])

    def reserve(self, point, net):
        if self.owner.get(point, net) != net:
            raise ValueError(f'Overlapping physical terminals at {point}')
        self.owner[point] = net

    def available(self, p, net):
        x, y, layer = p
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin <= x <= xmax and ymin <= y <= ymax and p not in self.blocked
                and (net not in ('VDD','VSS') or not (self.wide_clearance.get(p,set())-{net}))
                and (not self.allow_same_net_vias or not (self.via_halos.get(p, set()) - {net}))
                and self.owner.get(p, net) == net)

    def neighbors(self, p, net):
        x, y, layer = p
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (x + dx, y + dy, layer)
            if self.available(q, net):
                preferred = (layer == 1 and dy == 0) or (layer != 1 and dx == 0)
                yield q, 1. if preferred else 3.
        for other in (layer - 1, layer + 1):
            if not 0 <= other < len(self.layers):
                continue
            q = (x, y, other)
            if not self.available(q, net):
                continue
            if self.via_number(max(layer, other)) >= 4:
                # A 145 nm M5 landing encloses the 140 nm via4. It needs
                # clearance from the adjacent 140 nm M5 tracks.
                halo = [(x+dx,y+dy,max(layer,other)) for dx,dy in VIA4_HALO]
                if any(h in self.blocked or
                       (h in self.owner and (not self.allow_same_net_vias or self.owner[h] != net)) or
                       self.via_halos.get(h, set()) - {net}
                       for h in halo):
                    continue
            yield q, 4.

    def connect(self, net, source, targets):
        tree = {source}
        lower, upper = list(source), list(source)
        def heuristic(point):
            if not self.use_tree_heuristic:
                return 0.
            # Distance to the tree's enclosing box is a lower bound on the
            # nearest branch. Step costs are >=1 and layer changes cost 4.
            return sum((4 if axis==2 else 1)*max(lower[axis]-point[axis],
                       point[axis]-upper[axis],0) for axis in range(3))
        self.reserve(source, net)
        remaining = set(targets) - tree
        serial = count()
        while remaining:
            target = min(remaining, key=lambda p: (abs(p[0]-source[0])+abs(p[1]-source[1]), p))
            frontier = [(heuristic(target), next(serial), target)]
            costs, parent = {target: 0.}, {}
            closed = set()
            reached = None
            while frontier:
                _, _, p = heapq.heappop(frontier)
                if p in closed:
                    continue
                closed.add(p)
                if p in tree:
                    reached = p
                    break
                for q, weight in self.neighbors(p, net):
                    cost = costs[p] + weight
                    if cost < costs.get(q, float('inf')):
                        costs[q] = cost
                        parent[q] = p
                        heapq.heappush(frontier, (cost+heuristic(q), next(serial), q))
            if reached is None:
                blockers = set()
                for x,y,layer in closed:
                    for dx,dy,dl in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                        q=(x+dx,y+dy,layer+dl)
                        owner=self.owner.get(q)
                        if owner is not None and owner!=net:
                            blockers.add(owner)
                        blockers.update(self.via_halos.get(q,set())-{net})
                raise RoutingError(net,target,blockers)
            path = [reached]
            while path[-1] != target:
                path.append(parent[path[-1]])
            for p in path:
                self.reserve(p, net)
            for a, b in zip(path, path[1:]):
                if a[2] == b[2]:
                    self.paths.append((net, a, b))
                else:
                    self.vias.append((net, a[0], a[1], max(a[2], b[2])))
                    if self.via_number(max(a[2], b[2])) >= 4:
                        for dx,dy in VIA4_HALO:
                            halo = (a[0]+dx,a[1]+dy,max(a[2],b[2]))
                            if self.allow_same_net_vias:
                                self.via_halos.setdefault(halo, set()).add(net)
                            elif halo not in self.owner:
                                self.blocked.add(halo)
            tree.update(path)
            for point in path:
                for axis in range(3):
                    lower[axis]=min(lower[axis],point[axis])
                    upper[axis]=max(upper[axis],point[axis])
            remaining -= tree

    def draw(self):
        for net, a, b in self.paths:
            # Supply trunks carry aggregate register/control switching current.
            # Wider M3 needs 90 nm spacing: 190 nm rails fit a 280 nm grid.
            width = 190. if net in ('VDD', 'VSS') and self.layers[a[2]] == 'm3' and self.grid==GRID else self.widths[a[2]]
            half = width / 2
            x0, y0 = a[0] * self.grid, a[1] * self.grid
            x1, y1 = b[0] * self.grid, b[1] * self.grid
            self.builder.rect(self.layers[a[2]], min(x0, x1)-half, min(y0, y1)-half,
                              max(x0, x1)+half, max(y0, y1)+half)
        for _, x, y, upper in set(self.vias):
            x, y = x * self.grid, y * self.grid
            number = self.via_number(upper)
            self.builder.via(number, x, y)
            if number < 3:
                if number==1:
                    self.builder.rect(self.layers[upper-1], x-70, y-35, x+70, y+35)
                else:
                    self.builder.rect(self.layers[upper-1], x-35, y-70, x+35, y+70)
                self.builder.rect(self.layers[upper], x-35, y-70, x+35, y+70)
            elif number == 3:
                self.builder.rect(self.layers[upper-1], x-35, y-70, x+35, y+70)
                self.builder.sq(self.layers[upper], x, y, 140)
            else:
                self.builder.sq(self.layers[upper-1], x, y, 140)
                self.builder.sq(self.layers[upper], x, y, 145)
        # A single maze path can turn back alongside its new via4 landing.
        # Join adjacent pieces of that same net so the 2.5 nm landing growth
        # does not leave a 137.5 nm re-entrant spacing notch.
        existing={(a,b) for _,a,b in self.paths}
        for net,x,y,upper in set(self.vias):
            if self.via_number(upper)!=4:
                continue
            a=(x,y,upper)
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                b=(x+dx,y+dy,upper)
                if self.owner.get(b)!=net:
                    continue
                if dx:
                    xa,xb=(a[0]*GRID-72.5,b[0]*GRID-70) if dx>0 else (b[0]*GRID+70,a[0]*GRID+72.5)
                    self.builder.rect('m5',xa,a[1]*GRID-72.5,xb,a[1]*GRID+72.5)
                else:
                    ya,yb=(a[1]*GRID-72.5,b[1]*GRID-70) if dy>0 else (b[1]*GRID+70,a[1]*GRID+72.5)
                    self.builder.rect('m5',a[0]*GRID-72.5,ya,a[0]*GRID+72.5,yb)
                if (a,b) not in existing and (b,a) not in existing:
                    self.paths.append((net,a,b));existing.add((a,b))
            if not self.join_diagonal_vias:
                continue
            # Calibre uses square corner spacing. A same-net diagonal track
            # can also leave a 137.5 nm notch beside a 145 nm via4 landing.
            for dx,dy in ((-1,-1),(-1,1),(1,-1),(1,1)):
                b=(x+dx,y+dy,2)
                if self.owner.get(b)!=net:
                    continue
                corners=((x+dx,y,2),(x,y+dy,2))
                if any(self.owner.get(p,net)!=net for p in corners):
                    raise ValueError('Foreign route inside a via4 diagonal clearance')
                self.builder.rect('m5',min(x*GRID-72.5,(x+dx)*GRID-70),
                                  min(y*GRID-72.5,(y+dy)*GRID-70),
                                  max(x*GRID+72.5,(x+dx)*GRID+70),
                                  max(y*GRID+72.5,(y+dy)*GRID+70))
