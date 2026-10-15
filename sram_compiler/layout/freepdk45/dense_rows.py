"""Folded transistor rows for scaled column periphery at the OpenRAM pitch.

A row spans one two-column slot (1410 nm). VSS and VDD rails lie on its
bottom and top boundaries, so mirrored neighbours share rails, wells and
taps. Gate columns sit on a 190 nm pitch; every source/drain position is
contacted, including series nodes, so the layout keeps the source topology
instead of relying on split-gate reduction. Gate contacts occupy fixed
150 nm levels in the P/N gap.

Metal roles inside a row are fixed: M1 straps and rails, M2 horizontal bars
(one per net and region). Composites add M3 vertical tracks and M2-M5 via
stacks at x positions they reserve, so rows never route by search.
"""
from collections import defaultdict
from dataclasses import dataclass, field

from .lib import Builder

SLOT = 1410.
PITCH = 190.
X0 = 100.          # first contact centre
NB = 145.          # N strip bottom above the VSS boundary
EDGE = 145.        # P strip top below the VDD boundary
GAP_EDGE = 132.5   # bar region edge to the nearest gate level (strap/pad clearance)
LEVEL = 150.       # gate-level pitch (split-gate poly gap and M1 pad spacing)
STRIP_LEVEL = 140. # bar pitch inside a strip
P_ONLY_BASE = 105.


@dataclass
class Gate:
    """One finger: source MOS, finger index, width, and the row's net names."""
    mos: object
    finger: int
    width: float
    nets: dict      # source node -> row net
    level: int = None   # gate level override (e.g. one clock at two levels)

    @property
    def gate(self):
        return self.nets[self.mos.nodes[1]]


@dataclass
class RowView:
    builder: Builder
    width: float
    height: float
    bars: dict                       # (net, region, level) -> (y, x0, x1)
    terminals: dict
    fingers: list
    rails: dict = field(default_factory=dict)


def make_row(layout, name, gates, p_sd, n_sd, levels, reach=None, width=SLOT, x0=X0, min_region=None):
    """Build one folded row.

    gates: list of (p Gate | None, n Gate | None) per gate column.
    p_sd, n_sd: row net at each of len(gates)+1 contact positions, or None
    when the row has no such strip. VDD/VSS contacts strap to their rail.
    levels: ('g', net) -> gate level, ('p'|'n', net) -> strip bar level.
    Nets may share a level only with disjoint bar extents (checked).
    reach: net -> extra x positions its bars must cover (tracks or stacks).
    A strip's bar region may exceed its diffusion: straps run from the
    contacts to the bar, so narrow devices still get distinct bar levels.
    """
    reach = reach or {}
    count = len(gates)
    has_p, has_n = p_sd is not None, n_sd is not None
    if (has_p and len(p_sd) != count + 1) or (has_n and len(n_sd) != count + 1):
        raise ValueError(f'{name}: contact list length differs from gate count')
    cx = [x0 + PITCH * i for i in range(count + 1)]
    gx = [x + PITCH / 2 for x in cx[:-1]]
    if cx[-1] + 37.5 + 77.5 > width or cx[0] - 77.5 < 0:
        raise ValueError(f'{name}: {count} gates exceed the {width} nm slot')
    wn = max((n.width for _, n in gates if n), default=0.)
    wp = max((p.width for p, _ in gates if p), default=0.)
    def region(kind, w):
        used = [lv for (k, _), lv in levels.items() if k == kind]
        floor = (min_region or {}).get(kind, 0.)
        return max(w, floor, 75. + STRIP_LEVEL * max(used)) if used else max(w, floor)
    rn, rp = region('n', wn), region('p', wp)
    glevel = lambda g: levels[('g', g.gate)] if g.level is None else g.level
    top_level = max(glevel(g) for pair in gates for g in pair if g)
    base = NB + rn + GAP_EDGE if has_n else P_ONLY_BASE
    gate_y = lambda level: base + LEVEL * level
    pb = gate_y(top_level) + GAP_EDGE
    pt = pb + rp
    height = pt + EDGE
    b = Builder(layout, name)
    terminals, fingers, bars = {}, [], {}

    def rect(layer, a, c, d, e):
        b.rect(layer, min(a, d), min(c, e), max(a, d), max(c, e))

    # Wells, implants, threshold layer and boundary rails (shared by mirroring).
    split = (NB + rn + pb) / 2 if has_n else -100.
    if has_n:
        rect('pwell', 0, -100, width, split)
    rect('nwell', 0, split, width, height + 100)
    rect('vtg', 0, -100 if has_n else 0, width, height + 100)
    rails = {}
    # Rails overhang the slot so boundary supply vias are enclosed even at
    # the outermost slot; neighbours overlap on the same net.
    if has_n:
        rect('m1', -70, -35, width + 70, 35)
        rails['VSS'] = 0.
    rect('m1', -70, height - 35, width + 70, height + 35)
    rails['VDD'] = height
    taps = {}
    for net, y, implant, present in (('VSS', 0., 'pplus', has_n), ('VDD', height, 'nplus', True)):
        if not present:
            continue
        rect('active', X0 - 45, y - 45, X0 + 45, y + 45)
        rect(implant, X0 - 60, y - 60, X0 + 60, y + 60)
        b.cont(X0, y)
        taps[net] = (X0, y, 'm1')

    def strip_range(kind, column):
        p, n = gates[column]
        g = p if kind == 'p' else n
        if g is None:
            return None
        return (pt - g.width, pt) if kind == 'p' else (NB, NB + g.width)

    def strip_y(kind, level):
        return pt - 37.5 - STRIP_LEVEL * level if kind == 'p' else NB + 37.5 + STRIP_LEVEL * level

    # Diffusion per gate column; adjacent columns share contacted S/D.
    for kind, present in (('p', has_p), ('n', has_n)):
        if not present:
            continue
        ys = [strip_range(kind, i) for i in range(count)]
        if any(r is None for r in ys):
            raise ValueError(f'{name}: a {kind} strip needs a device in every column')
        lo = min(r[0] for r in ys); hi = max(r[1] for r in ys)
        rect('nplus' if kind == 'n' else 'pplus', cx[0] - 77.5, lo - 40, cx[-1] + 77.5, hi + 40)
        for i, (y0, y1) in enumerate(ys):
            rect('active', cx[i] - (37.5 if i == 0 else 0), y0,
                 cx[i + 1] + (37.5 if i == count - 1 else 0), y1)
    sites = defaultdict(list)    # (net, region) -> x positions
    for kind, nets in (('p', p_sd), ('n', n_sd)):
        if nets is None:
            continue
        supply = 'VDD' if kind == 'p' else 'VSS'
        for i, net in enumerate(nets):
            adj = [strip_range(kind, j) for j in (i - 1, i) if 0 <= j < count]
            y0 = max(r[0] for r in adj); y1 = min(r[1] for r in adj)
            first, last = y0 + 37.5, y1 - 37.5
            if last < first:
                raise ValueError(f'{name}: diffusion too narrow for a contact')
            # Contacts start at the strip's aligned edge.
            steps = range(int((last - first) // STRIP_LEVEL) + 1)
            ys = sorted(first + STRIP_LEVEL * k for k in steps) if kind == 'n' else \
                sorted(last - STRIP_LEVEL * k for k in steps)
            for y in ys:
                b.cont(cx[i], y)
            if net == supply:
                rail = height if kind == 'p' else 0.
                rect('m1', cx[i] - 35, min(ys[0] - 70, rail), cx[i] + 35, max(ys[-1] + 70, rail))
                continue
            level = levels.get((kind, net))
            if level is None:
                raise ValueError(f'{name}: no strip level for {net}')
            y = strip_y(kind, level)
            # The 70 nm strap encloses contacts and the via1 by 35 nm vertically.
            rect('m1', cx[i] - 35, min(ys[0], y) - 70, cx[i] + 35, max(ys[-1], y) + 70)
            b.via(1, cx[i], y)
            sites[(net, kind, level)].append(cx[i])
    # Gate columns: one poly line for a shared gate, two for split gates.
    pads = defaultdict(list)     # level y -> [(x, net)]
    for i, (p, n) in enumerate(gates):
        x = gx[i]
        y_top = pt + 50 if p else None
        y_bot = NB - 50 if n else None
        if p and n and p.gate == n.gate and glevel(p) == glevel(n):
            y = gate_y(glevel(p))
            rect('poly', x - 25, y_bot, x + 25, y_top)
            contacts = [(y, p.gate)]
        else:
            contacts = []
            if n:
                yn = gate_y(glevel(n))
                rect('poly', x - 25, y_bot, x + 25, yn + 37.5)
                contacts.append((yn, n.gate))
            if p:
                yp = gate_y(glevel(p))
                rect('poly', x - 25, yp - 37.5, x + 25, y_top)
                contacts.append((yp, p.gate))
            if p and n and yp - yn < LEVEL:
                raise ValueError(f'{name}: split gate {p.gate}/{n.gate} levels are out of order')
        for y, net in contacts:
            b.sq('poly', x, y, 75)
            b.cont(x, y)
            b.via(1, x, y)
            pads[y].append((x, net))
            sites[(net, 'g', round((y - base) / LEVEL))].append(x)
        for g, kind in ((p, 'p'), (n, 'n')):
            if g is None:
                continue
            sy0, sy1 = strip_range(kind, i)
            d, gate, s, body = g.mos.nodes
            left = (p_sd if kind == 'p' else n_sd)[i]
            right = (p_sd if kind == 'p' else n_sd)[i + 1]
            dn, sn = g.nets[d], g.nets[s]
            if {left, right} != {dn, sn}:
                raise ValueError(f'{name}: {g.mos.name} finger {g.finger} does not match its diffusion nets')
            yc = (sy0 + sy1) / 2
            dx, sx = (cx[i], cx[i + 1]) if left == dn else (cx[i + 1], cx[i])
            key = f'{g.mos.name}@{g.finger}'
            terminals[key + ':d'] = (dx, yc, 'm1')
            terminals[key + ':s'] = (sx, yc, 'm1')
            terminals[key + ':g'] = (x, gate_y(glevel(g)), 'poly')
            terminals[key + ':b'] = taps['VDD' if kind == 'p' else 'VSS']
            fingers.append(dict(device=g.mos.name, finger=g.finger, width_nm=g.width,
                                length_nm=g.mos.length_m * 1e9, model=g.mos.model,
                                nodes=g.mos.nodes, gate_xy_nm=(x, yc)))
    # Gate M1 landings 140 x 70 enclose contact and via1; same-net
    # neighbours on a level merge into one landing.
    for y, items in pads.items():
        items.sort()
        runs = []
        for x, net in items:
            if runs and x - runs[-1][1] < 210:
                if runs[-1][2] != net:
                    raise ValueError(f'{name}: adjacent gate contacts of {runs[-1][2]} and {net} share a level')
                runs[-1][1] = x
            else:
                runs.append([x, x, net])
        for a, c, _ in runs:
            rect('m1', a - 70, y - 35, c + 70, y + 35)
    # M2 bars extend 70 nm past their end vias (35 nm via enclosure).
    by_y = defaultdict(list)
    for (net, kind, level), xs in sites.items():
        y = gate_y(level) if kind == 'g' else strip_y(kind, level)
        xs = xs + list(reach.get(net, ()))
        lo, hi = min(xs) - 70, max(xs) + 70
        rect('m2', lo, y - 35, hi, y + 35)
        bars[(net, kind, level)] = (y, lo + 70, hi - 70)
        by_y[y].append((lo, hi, net))
    for y, spans in by_y.items():
        spans.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
            if b0 - a1 < 70:
                raise ValueError(f'{name}: bars {an} and {bn} collide at y={y}')
    b.rect('bnd', 0, 0, width, height)
    return RowView(b, width, height, bars, terminals, fingers, rails)


def transform(point, x, y, mirror, height):
    """Row placement: optional mirror about the row's horizontal mid-line."""
    px, py, *rest = point
    return (x + px, y + (height - py if mirror else py), *rest)
