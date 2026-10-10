"""Deterministic three-metal maze routing for physical calibration references."""
from __future__ import annotations

import heapq
from itertools import count

from .lib import Builder

GRID = 280.0
LAYERS = ('m3', 'm4', 'm5')
WIDTHS = (70., 140., 140.)
VIA4_HALO = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if dx or dy)


class Router:
    def __init__(self, builder, bounds):
        self.builder = builder
        self.bounds = bounds
        self.owner = {}
        self.paths = []
        self.vias = []
        self.blocked = set()

    def point(self, x, y, layer='m3'):
        if abs(x / GRID - round(x / GRID)) > 1e-8 or abs(y / GRID - round(y / GRID)) > 1e-8:
            raise ValueError(f'Off routing grid: {x}, {y}')
        return (round(x / GRID), round(y / GRID), LAYERS.index(layer))

    def reserve(self, point, net):
        if self.owner.get(point, net) != net:
            raise ValueError(f'Overlapping physical terminals at {point}')
        self.owner[point] = net

    def available(self, p, net):
        x, y, layer = p
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin <= x <= xmax and ymin <= y <= ymax and p not in self.blocked
                and self.owner.get(p, net) == net)

    def neighbors(self, p, net):
        x, y, layer = p
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (x + dx, y + dy, layer)
            if self.available(q, net):
                preferred = (layer == 1 and dy == 0) or (layer != 1 and dx == 0)
                yield q, 1. if preferred else 3.
        for other in (layer - 1, layer + 1):
            if not 0 <= other < 3:
                continue
            q = (x, y, other)
            if not self.available(q, net):
                continue
            if max(layer, other) == 2:
                # A 145 nm M5 landing encloses the 140 nm via4. It needs
                # clearance from the adjacent 140 nm M5 tracks.
                halo = [(x+dx,y+dy,2) for dx,dy in VIA4_HALO]
                if any(h in self.blocked or h in self.owner for h in halo):
                    continue
            yield q, 4.

    def connect(self, net, source, targets):
        tree = {source}
        self.reserve(source, net)
        remaining = set(targets) - tree
        serial = count()
        while remaining:
            target = min(remaining, key=lambda p: (abs(p[0]-source[0])+abs(p[1]-source[1]), p))
            frontier = [(0., next(serial), target)]
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
                        # The connected tree is often a long bus; a zero
                        # heuristic is deliberate to find the nearest branch.
                        heapq.heappush(frontier, (cost, next(serial), q))
            if reached is None:
                raise RuntimeError(f'Unable to route {net} terminal {target}; expand routing channels')
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
                    if max(a[2], b[2]) == 2:
                        for dx,dy in VIA4_HALO:
                            halo = (a[0]+dx,a[1]+dy,2)
                            if halo not in self.owner:
                                self.blocked.add(halo)
            tree.update(path)
            remaining -= tree

    def draw(self):
        for net, a, b in self.paths:
            # Supply trunks carry aggregate register/control switching current.
            # Wider M3 needs 90 nm spacing: 190 nm rails fit a 280 nm grid.
            width = 190. if net in ('VDD', 'VSS') and a[2] == 0 else WIDTHS[a[2]]
            half = width / 2
            x0, y0 = a[0] * GRID, a[1] * GRID
            x1, y1 = b[0] * GRID, b[1] * GRID
            self.builder.rect(LAYERS[a[2]], min(x0, x1)-half, min(y0, y1)-half,
                              max(x0, x1)+half, max(y0, y1)+half)
        for _, x, y, upper in set(self.vias):
            x, y = x * GRID, y * GRID
            if upper == 1:
                self.builder.via(3, x, y)
                self.builder.rect('m3', x-35, y-70, x+35, y+70)
                self.builder.sq('m4', x, y, 140)
            else:
                self.builder.via(4, x, y)
                self.builder.sq('m4', x, y, 140)
                self.builder.sq('m5', x, y, 145)
        # A single maze path can turn back alongside its new via4 landing.
        # Join adjacent pieces of that same net so the 2.5 nm landing growth
        # does not leave a 137.5 nm re-entrant spacing notch.
        existing={(a,b) for _,a,b in self.paths}
        for net,x,y,upper in set(self.vias):
            if upper!=2:
                continue
            a=(x,y,2)
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                b=(x+dx,y+dy,2)
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
