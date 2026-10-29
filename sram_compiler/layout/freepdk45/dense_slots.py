"""Shared column-slot plumbing and compact data-register rows."""
from dataclasses import dataclass, field, replace
from collections import defaultdict
from itertools import product

from .components import Component
from .lib import Builder, pya
from .dense_rows import make_row, Gate, RowView, PITCH, X0 as X0_ROW

SUPPLY_X = 0.       # shared boundary M1-M4 supply stacks


@dataclass
class SlotView:
    builder: Builder
    width: float
    height: float
    terminals: dict            # role-prefixed device terminal -> point
    fingers: list
    pins: dict = field(default_factory=dict)       # role net -> [points]
    supplies: list = field(default_factory=list)   # (net, y) boundary stacks
    m3: list = field(default_factory=list)         # (net, x, y0, y1) audit
    rows: list = field(default_factory=list)


class _Slot:
    """Stack rows, record transformed bars, draw tracks and via stacks."""

    def __init__(self, layout, name):
        self.ly = layout
        self.b = Builder(layout, name)
        self.y = 0.
        self.bars = defaultdict(list)      # net -> [(y, x0, x1)]
        self.terminals, self.fingers, self.rows = {}, [], []
        self.boundaries = []               # (net, y)
        self.m3 = []

    def add(self, row, mirror, prefix_map):
        y, h = self.y, row.height
        from .lib import pya, u
        t = pya.Trans(pya.Trans.M0, u(0), u(y + h)) if mirror else pya.Trans(u(0), u(y))
        self.b.cell.insert(pya.CellInstArray(row.builder.cell.cell_index(), t))
        fy = (lambda v: y + h - v) if mirror else (lambda v: y + v)
        for (net, region, level), (by, x0, x1) in row.bars.items():
            self.bars[net].append((fy(by), x0, x1))
        for key, (px, py, layer) in row.terminals.items():
            self.terminals[key] = (px, fy(py), layer)
        for f in row.fingers:
            gx, gy = f['gate_xy_nm']
            self.fingers.append(dict(f, gate_xy_nm=(gx, fy(gy))))
        for net, ry in row.rails.items():
            self.boundaries.append((net, fy(ry)))
        self.rows.append(dict(cell=row.builder.name, y_nm=y, height_nm=h, mirror=mirror))
        self.y += h

    def rect(self, layer, x0, y0, x1, y1):
        self.b.rect(layer, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def via(self, n, x, y):
        self.b.via(n, x, y)
        for k in (n, n + 1):
            if k < 4:
                self.rect(f'm{k}', x - 35, y - 70, x + 35, y + 70)
            else:
                self.b.sq(f'm{k}', x, y, 140 if k == 4 else 145)

    def track(self, net, x, extra=()):
        """Vertical M3 joining every bar of ``net`` that covers x."""
        ys = [y for y, x0, x1 in self.bars[net] if x0 - 1e-6 <= x <= x1 + 1e-6] + list(extra)
        if not ys:
            raise ValueError(f'No bar of {net} reaches track x={x}')
        for y in ys:
            if y not in extra:
                self.b.via(2, x, y)
        lo, hi = min(ys), max(ys)
        self.rect('m3', x - 35, lo - 70, x + 35, hi + 70)
        self.m3.append((net, x, lo - 70, hi + 70))
        return lo, hi

    def stack(self, net, x):
        """M2-M5 stacks on every bar of ``net`` at x (bitline joins)."""
        ys = sorted(y for y, x0, x1 in self.bars[net] if x0 - 1e-6 <= x <= x1 + 1e-6)
        if not ys:
            raise ValueError(f'No bar of {net} reaches stack x={x}')
        for y in ys:
            # The bar itself encloses via2; M3/M4/M5 get landings.
            self.b.via(2, x, y)
            for n in (3, 4):
                self.via(n, x, y)
            self.m3.append((net, x, y - 70, y + 70))
        return ys

    def supplies(self, x=SUPPLY_X):
        done = set()
        for net, y in self.boundaries:
            if y in done:
                continue
            done.add(y)
            self.b.via(1, x, y)
            for n in (2, 3):
                self.via(n, x, y)
            self.m3.append((net, x, y - 70, y + 70))
        return sorted({(net, y) for net, y in self.boundaries}, key=lambda t: t[1])

    def audit(self):
        """M3 width/space between different nets (Calibre remains final)."""
        for i, (na, xa, a0, a1) in enumerate(self.m3):
            for nb, xb, b0, b1 in self.m3[i + 1:]:
                if na == nb:
                    continue
                dx = max(0., abs(xa - xb) - 70)
                dy = max(0., max(a0, b0) - min(a1, b1))
                if (dx * dx + dy * dy) ** .5 < 70 - 1e-6:
                    raise ValueError(f'M3 {na}@{xa} and {nb}@{xb} are closer than 70 nm')


def _named(mos, prefix, path):
    # The hierarchical source path, so fingers map back to leaf devices.
    return replace(mos, name=prefix + path)


def _devices(component, path=''):
    out = {}
    for d in component.devices:
        out[(path + '/' if path else '') + d.name] = d
    for child in component.children:
        out.update(_devices(child.component, (path + '/' if path else '') + child.name))
    return out


def _even(width, count):
    w = width / count
    if abs(w / 5 - round(w / 5)) > 1e-6 or w < 90:
        raise ValueError(f'{width} nm cannot form {count} legal fingers')
    return w


def _gate_levels(columns, reach):
    """Per-column (P, N) gate levels with the lowest top level.

    columns: (P gate net, N gate net) per gate; split gates need P above N.
    Adjacent pads on one level share a net; a net's bar on a level spans its
    pads and its reach, and different nets' bars keep 70 nm apart.
    """
    xs = [X0_ROW + PITCH * i + PITCH / 2 for i in range(len(columns))]
    def legal(assign):
        spans = defaultdict(lambda: defaultdict(list))
        for x, (lp, ln), (pn, nn) in zip(xs, assign, columns):
            spans[lp][pn].append(x); spans[ln][nn].append(x)
        for level, nets in spans.items():
            bars = sorted((min(v + reach.get(n, [])) - 70, max(v + reach.get(n, [])) + 70, n) for n, v in nets.items())
            if any(b0 - a1 < 70 for (_, a1, _), (b0, _, _) in zip(bars, bars[1:])):
                return False
        for k in range(len(assign) - 1):
            for la, na in zip(assign[k], columns[k]):
                for lb, nb in zip(assign[k + 1], columns[k + 1]):
                    if la == lb and na != nb:
                        return False
        return True
    for top in range(1, 8):
        options = [[(l, l) for l in range(top + 1)] if pn == nn else
                   [(lp, ln) for lp in range(top + 1) for ln in range(lp)] for pn, nn in columns]
        for assign in product(*options):
            if max(max(a) for a in assign) == top and legal(assign):
                return assign
    raise ValueError('No legal gate-level assignment')


def _strip_levels(nets, reach):
    """Interval colouring of one strip's bars (contacts plus reach, 70 nm apart)."""
    sites = defaultdict(list)
    for i, net in enumerate(nets):
        if net not in ('VDD', 'VSS'):
            sites[net].append(X0_ROW + PITCH * i)
    levels, ends = {}, []
    for lo, hi, net in sorted((min(v + reach[n]) - 70, max(v + reach[n]) + 70, n) for n, v in sites.items()):
        level = next((k for k, e in enumerate(ends) if lo - e >= 70), len(ends))
        if level == len(ends):
            ends.append(hi)
        else:
            ends[level] = hi
        levels[net] = level
    return levels


def _register_rows(layout: pya.Layout, name: str, role: str, dff: Component,
                   tracks: dict, width: float) -> tuple[RowView, RowView]:
    """Two source-matched DFF rows with searched gate and strip-bus levels."""
    pre = role + '.'
    kids = {i.name: i for i in dff.children}
    if set(kids) != {'Xinv1_clk', 'Xinv2_D', 'Xtg1', 'Xinv3', 'Xinv4', 'Xtg2',
                     'Xinv5', 'Xtg3', 'Xinv6', 'Xinv7', 'Xtg4'} or dff.ports != ('VDD', 'VSS', 'D', 'Q', 'CLK'):
        raise ValueError('Folded register requires the frozen transmission-gate DFF')
    net = lambda n: n if n in ('VDD', 'VSS', 'CLK') else pre + n

    def gates(inst, level_p=None, level_n=None):
        i = kids[inst]
        nets = {f: net(a) for f, a in zip(i.component.ports, i.nodes)}
        p = [d for d in i.component.devices if d.model == 'PMOS_VTG']
        n = [d for d in i.component.devices if d.model == 'NMOS_VTG']
        if len(p) != 1 or len(n) != 1:
            raise ValueError('Register leaf must be one P and one N device')
        p, n = p[0], n[0]
        mp = replace(p, name=pre + inst + '/' + p.name); mn = replace(n, name=pre + inst + '/' + n.name)
        return (Gate(mp, 0, p.width_m * 1e9, nets, level_p), Gate(mn, 0, n.width_m * 1e9, nets, level_n))
    a_p = ['VDD', net('D_b'), net('z1'), net('z3'), 'VDD', net('z2')]
    a_n = ['VSS', net('D_b'), net('z1'), net('z3'), 'VSS', net('z2')]
    b_p = [net('CLKB'), 'VDD', net('z4'), net('z5'), net('QB'), 'VDD', net('Q')]
    b_n = [net('CLKB'), 'VSS', net('z4'), net('z5'), net('QB'), 'VSS', net('Q')]
    t, clk = tracks[role], tracks['CLK']
    rows = []
    for tag, insts, p_sd, n_sd, keys in (
            ('A', ['Xinv2_D', 'Xtg1', 'Xtg2', 'Xinv4', 'Xinv3'], a_p, a_n, ('CLKB', 'z2', 'D', 'D_b', 'z1', 'z3')),
            ('B', ['Xinv1_clk', 'Xinv5', 'Xtg3', 'Xtg4', 'Xinv7', 'Xinv6'], b_p, b_n, ('CLKB', 'z2', 'Q', 'z4', 'z5', 'QB'))):
        reach = {'CLK': [clk], **{net(k): [t[k]] for k in keys}}
        pairs = [gates(i) for i in insts]
        assign = _gate_levels([(p.gate, n.gate) for p, n in pairs], reach)
        pairs = [(replace(p, level=lp), replace(n, level=ln)) for (p, n), (lp, ln) in zip(pairs, assign)]
        levels = {('g', g.gate): g.level for pair in pairs for g in pair}
        for kind, sd in (('p', p_sd), ('n', n_sd)):
            levels.update({(kind, k): v for k, v in _strip_levels(sd, reach).items()})
        rows.append(make_row(layout, f'{name}_{role}_REG{tag}', pairs, p_sd, n_sd, levels, reach, width=width))
    return tuple(rows)


def make_register_slot(layout: pya.Layout, name: str, dff: Component,
                       roles: tuple[str, ...] = ('L', 'U'), *, geometry: dict) -> SlotView:
    """Two compact data registers; Q tracks meet their own write buffers."""
    geo = geometry
    s = _Slot(layout, name)
    table = geo['register']
    rows = {r: _register_rows(layout, name, r, dff, table, geo['width']) for r in ('L', 'U')}
    plan = [(r, row) for r in ('L', 'U') for row in rows[r]]
    for i, (role, row) in enumerate(plan):
        if role in roles:
            s.add(row, i % 2 == 1, None)
        else:
            s.y += row.height
    top = s.y
    pins = defaultdict(list)
    ys = [y for net in s.bars if net == 'CLK' for y, x0, x1 in s.bars[net]]
    clk = table['CLK']
    s.track('CLK', clk)
    clock_y = min(ys)
    s.b.via(3, clk, clock_y); s.b.sq('m4', clk, clock_y, 140)
    pins['CLK'].append((clk, clock_y, 'm4'))
    for role in roles:
        pre = role + '.'
        t = table[role]
        for key in ('CLKB', 'z2', 'D_b', 'z1', 'z3', 'z4', 'z5', 'QB'):
            s.track(pre + key, t[key])
        # D reaches just below the band: its landing defines the macro's
        # bottom edge, so the data inputs stay accessible from outside.
        s.track(pre + 'D', t['D'], extra=(-70.,))
        pins[pre + 'D'].append((t['D'], -70., 'm3'))
        # Q climbs past the band top to overlap the write slot's input track.
        s.track(pre + 'Q', t['Q'], extra=(top + 370.,))
        pins[pre + 'Q'].append((t['Q'], top + 370., 'm3'))
    supplies = s.supplies(geo['supply_x'])
    s.audit()
    s.b.rect('bnd', 0, 0, geo['width'], top)
    return SlotView(s.b, geo['width'], top, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)
