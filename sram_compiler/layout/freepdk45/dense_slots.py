"""Two-column write and precharge slots for scaled driver classes.

Rows ≤ 32 keep the verified reference cells. Larger row classes enlarge the
precharge, write-driver and data-buffer devices, so they are folded into
``dense_rows`` rows inside a 1410 nm slot (columns c and c+1):

* write slot, bottom to top: both data buffers in one row; then for each
  column an input-inverter row, a BL tristate row and a BLB tristate row;
* precharge slot: two mirrored P-only rows sharing their VDD rail.

Each driver owns fixed M3 track positions that clear the other column's
bitline stacks. BL/BLB bars stack M2-M5 directly onto the column's bitline
x, so the bitline itself joins a driver's rows. Supply rails meet M1-M4
stacks on the slot boundary (x = 0); the bank draws M4 bars along them.
"""
from dataclasses import dataclass, field, replace
from collections import defaultdict
from itertools import product

from .lib import Builder
from .dense_rows import make_row, Gate, SLOT, PITCH, X0 as X0_ROW

# Slot-relative x of the bitline stacks and the gate-net M3 tracks.
TRACKS = {
    'L': dict(BL=185., BLB=520., DIN=840., DINB=700., ENB=980., EN=1120.),
    'U': dict(BL=890., BLB=1225., DIN=352.5, DINB=492.5, ENB=632.5, EN=212.5),
}
BUFFER = dict(A_U=700., A_L=980., X_U=212.5, X_L=1120.)
PRE_TAP = 352.5     # precharge enable via stack onto its M4 bar
SUPPLY_X = 0.       # boundary M1-M4 supply stacks (shared with the neighbour)


def fold_class(rows):
    """Return True when this array needs the folded periphery."""
    return rows > 32


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


def _inv_fingers(wp, wn, limit=3):
    for f in range(limit, 0, -1):
        try:
            return f, _even(wp, f), _even(wn, f)
        except ValueError:
            continue
    raise ValueError('No legal inverter folding')


def _driver_rows(layout, name, role, write, tie_din=False, geo=None):
    """Inverter, BL tristate and BLB tristate rows of one write driver."""
    geo = geo or OPENRAM
    pre = role + '.'
    devs = {n: _named(d, pre, n) for n, d in _devices(write).items()}
    t = geo['tracks'][role]
    net = lambda n: n if n in ('VDD', 'VSS') else pre + n
    nets = {n: net(n) for n in ('DIN', 'DINB', 'EN', 'ENB', 'BL', 'BLB', 'int1', 'int2', 'int3', 'int4', 'VDD', 'VSS')}
    din = 'VSS' if tie_din else nets['DIN']
    gmap = dict(nets, DIN=din)
    # Input inverters: DIN -> DINB (M1/M2), EN -> ENB (M3/M4), one row.
    f, wp, wn = _inv_fingers(devs['M1'].width_m * 1e9, devs['M2'].width_m * 1e9)
    seq_din = [('VDD' if (f - k) % 2 == 0 else 'out') for k in range(f + 1)]
    seq_en = [('VDD' if k % 2 == 0 else 'out') for k in range(f + 1)]
    gates, p_sd, n_sd = [], [], []
    for k, s in enumerate(seq_din):
        p_sd.append('VDD' if s == 'VDD' else nets['DINB']); n_sd.append('VSS' if s == 'VDD' else nets['DINB'])
    for k, s in enumerate(seq_en[1:], 1):
        p_sd.append('VDD' if s == 'VDD' else nets['ENB']); n_sd.append('VSS' if s == 'VDD' else nets['ENB'])
    for k in range(f):
        gates.append((Gate(devs['M1'], k, wp, gmap), Gate(devs['M2'], k, wn, gmap)))
    for k in range(f):
        gates.append((Gate(devs['M3'], k, wp, gmap), Gate(devs['M4'], k, wn, gmap)))
    levels = {('g', din): 0, ('g', nets['EN']): 1, ('p', nets['DINB']): 0, ('n', nets['DINB']): 0,
              ('p', nets['ENB']): 1, ('n', nets['ENB']): 1}
    reach = {nets['DINB']: [t['DINB']], nets['ENB']: [t['ENB']], nets['EN']: [t['EN']]}
    if not tie_din:
        reach[din] = [t['DIN']]
    else:
        reach['VSS'] = [t['DIN']]
    inv = make_row(layout, f'{name}_{role}_INV', gates, p_sd, n_sd, levels, reach, width=geo['width'])
    rows = [inv]
    # Tristates: P VDD-(DINB|DIN)-int-ENB-out, N out-EN-int-(DINB|DIN)-VSS.
    for out, (pa, pb_, na, nb_), (ip, inn), gate in (
            ('BL', ('M5', 'M6', 'M7', 'M8'), ('int1', 'int2'), 'DINB'),
            ('BLB', ('M9', 'M10', 'M11', 'M12'), ('int3', 'int4'), 'DIN')):
        wp = _even(devs[pa].width_m * 1e9, 3); wn = _even(devs[nb_].width_m * 1e9, 3)
        gates = []
        order = [(0, 'in'), (0, 'en'), (1, 'en'), (1, 'in'), (2, 'in'), (2, 'en')]
        for finger, kind in order:
            if kind == 'in':
                gates.append((Gate(devs[pa], finger, wp, gmap), Gate(devs[nb_], finger, wn, gmap)))
            else:
                gates.append((Gate(devs[pb_], finger, wp, gmap), Gate(devs[na], finger, wn, gmap)))
        o, i1, i2 = nets[out], nets[ip], nets[inn]
        p_sd = ['VDD', i1, o, i1, 'VDD', i1, o]
        n_sd = ['VSS', i2, o, i2, 'VSS', i2, o]
        g = din if gate == 'DIN' else nets['DINB']
        levels = {('g', nets['EN']): 0, ('g', g): 1, ('g', nets['ENB']): 2,
                  ('p', i1): 0, ('p', o): 1, ('n', i2): 0, ('n', o): 1}
        reach = {o: [t[out]], nets['EN']: [t['EN']], nets['ENB']: [t['ENB']],
                 g: [t['DIN' if gate == 'DIN' else 'DINB']]}
        if g == 'VSS':
            reach = {k: v for k, v in reach.items() if k != 'VSS'}
            reach['VSS'] = [t['DIN']]
        rows.append(make_row(layout, f'{name}_{role}_{out}', gates, p_sd, n_sd, levels, reach, width=geo['width']))
    return rows, nets


def _buffer_rows(layout, name, buffers, geo=None):
    """Data-buffer rows: one shared row, or one row per column when the
    output inverter is too narrow to fold into two fingers."""
    geo = geo or OPENRAM
    try:
        return [_buffer_row(layout, name, buffers, geo)]
    except ValueError as error:
        if 'legal fingers' not in str(error):
            raise
    rows = []
    for role in ('L', 'U'):
        comp = buffers[role]
        devs = {n: _named(d, role + 'B.', n) for n, d in _devices(comp).items()}
        n = {'A': role + 'B.A', 'X': role + 'B.X', 'Z': role + '.DIN'}
        inv0 = {'A': n['A'], 'Z': n['X'], 'VDD': 'VDD', 'VSS': 'VSS'}
        inv1 = {'A': n['X'], 'Z': n['Z'], 'VDD': 'VDD', 'VSS': 'VSS'}
        g = lambda d, m: Gate(d, 0, d.width_m * 1e9, m)
        gates = [(g(devs['Xinv0/Mpinv_pmos'], inv0), g(devs['Xinv0/Mpinv_nmos'], inv0)),
                 (g(devs['Xinv1/Mpinv_pmos'], inv1), g(devs['Xinv1/Mpinv_nmos'], inv1))]
        p_sd = [n['X'], 'VDD', n['Z']]; n_sd = [n['X'], 'VSS', n['Z']]
        levels = {('g', n['A']): 0, ('g', n['X']): 1}
        for r in ('p', 'n'):
            levels.update({(r, n['X']): 0, (r, n['Z']): 1})
        reach = {n['A']: [geo['buffer']['A_' + role]], n['Z']: [geo['tracks'][role]['DIN']], n['X']: [geo['buffer']['X_' + role]]}
        rows.append(make_row(layout, f'{name}_{role}_BUF', gates, p_sd, n_sd, levels, reach, width=geo['width']))
    return rows


def _buffer_row(layout, name, buffers, geo=None):
    """Both data buffers: column c+1 (U) on the left, column c (L) right
    (swapped when the geometry puts column c's tracks on the left)."""
    geo = geo or OPENRAM
    left, right = ('L', 'U') if geo['buffer_swap'] else ('U', 'L')
    gates = []
    p_sd, n_sd = [], []
    order = []
    widths = {}
    for role in ('U', 'L'):
        comp = buffers.get(role)
        if comp is None:
            raise ValueError('Folded data buffers come in column pairs')
        devs = {n: _named(d, role + 'B.', n) for n, d in _devices(comp).items()}
        i0p, i0n = devs['Xinv0/Mpinv_pmos'], devs['Xinv0/Mpinv_nmos']
        i1p, i1n = devs['Xinv1/Mpinv_pmos'], devs['Xinv1/Mpinv_nmos']
        w1p, w1n = _even(i1p.width_m * 1e9, 2), _even(i1n.width_m * 1e9, 2)
        widths[role] = (i0p, i0n, i1p, i1n, w1p, w1n)
    def m(role):
        b = role + 'B.'
        # The buffer output is the driver's DIN net, joined by one M3 track.
        return {'A': b + 'A', 'Z': role + '.DIN', 'X': b + 'X', 'VDD': 'VDD', 'VSS': 'VSS'}
    # Leaf nets differ per inverter: Xinv0 (A -> Z=X), Xinv1 (A=X -> Z).
    def inv0(role):
        n = m(role); return {'A': n['A'], 'Z': n['X'], 'VDD': 'VDD', 'VSS': 'VSS'}
    def inv1(role):
        n = m(role); return {'A': n['X'], 'Z': n['Z'], 'VDD': 'VDD', 'VSS': 'VSS'}
    i0p, i0n, i1p, i1n, w1p, w1n = widths[left]
    gates += [(Gate(i0p, 0, i0p.width_m * 1e9, inv0(left)), Gate(i0n, 0, i0n.width_m * 1e9, inv0(left))),
              (Gate(i1p, 0, w1p, inv1(left)), Gate(i1n, 0, w1n, inv1(left))),
              (Gate(i1p, 1, w1p, inv1(left)), Gate(i1n, 1, w1n, inv1(left)))]
    i0p, i0n, i1p, i1n, w1p, w1n = widths[right]
    gates += [(Gate(i1p, 0, w1p, inv1(right)), Gate(i1n, 0, w1n, inv1(right))),
              (Gate(i1p, 1, w1p, inv1(right)), Gate(i1n, 1, w1n, inv1(right))),
              (Gate(i0p, 0, i0p.width_m * 1e9, inv0(right)), Gate(i0n, 0, i0n.width_m * 1e9, inv0(right)))]
    U, L = m(left), m(right)
    p_sd = [U['X'], 'VDD', U['Z'], 'VDD', L['Z'], 'VDD', L['X']]
    n_sd = [U['X'], 'VSS', U['Z'], 'VSS', L['Z'], 'VSS', L['X']]
    # Left and right halves share levels with disjoint extents.
    levels = {('g', U['A']): 0, ('g', L['A']): 0, ('g', U['X']): 1, ('g', L['X']): 2}
    for r in ('p', 'n'):
        levels.update({(r, U['X']): 0, (r, L['X']): 0, (r, U['Z']): 1, (r, L['Z']): 1})
    buf, tracks = geo['buffer'], geo['tracks']
    reach = {U['A']: [buf['A_' + left]], L['A']: [buf['A_' + right]],
             U['Z']: [tracks[left]['DIN']], L['Z']: [tracks[right]['DIN']],
             U['X']: [buf['X_' + left]], L['X']: [buf['X_' + right]]}
    return make_row(layout, name + '_BUF', gates, p_sd, n_sd, levels, reach, width=geo['width'])


def make_write_slot(layout, name, writes, buffers, replica=False, geometry=None):
    """writes/buffers: role ('L' column c, 'U' column c+1) -> source component.

    The replica slot carries only the U write-driver load with DIN tied to
    VSS; the empty L rows keep their heights so every slot aligns.
    geometry selects the slot width and track tables (default: OpenRAM pitch).
    """
    geo = geometry or OPENRAM
    s = _Slot(layout, name)
    roles = ('U',) if replica else ('L', 'U')
    rows = {}
    nets = {}
    reference = writes['U']
    for role in ('L', 'U'):
        rows[role], nets[role] = _driver_rows(layout, name, role, writes.get(role, reference),
                                              tie_din=replica, geo=geo)
    bufs = _buffer_rows(layout, name, buffers if not replica else {'U': buffers['U'], 'L': buffers['U']}, geo)
    plan = [('buf', buf, False) for buf in bufs] + [(r, row, k % 2 == 0) for r in ('L', 'U')
                                                   for k, row in enumerate(rows[r])]
    # Mirror alternate rows so neighbours share rails: buf normal, then alternate.
    plan = [(tag, row, i % 2 == 1) for i, (tag, row, _) in enumerate(plan)]
    for tag, row, mirror in plan:
        if (tag == 'buf' or tag == 'L') and replica:
            s.y += row.height
            continue
        s.add(row, mirror, None)
    pins = defaultdict(list)
    for role in roles:
        t, n = geo['tracks'][role], nets[role]
        for out in ('BL', 'BLB'):
            ys = s.stack(n[out], t[out])
            pins[role + '.' + out].append((t[out], min(ys), 'm5'))
        for key in ('DINB', 'ENB'):
            s.track(n[key], t[key])
        lo, hi = s.track(n['EN'], t['EN'])
        # EN meets the bank's w_en M4 bar at the inverter row's EN level.
        en_y = min(y for y, x0, x1 in s.bars[n['EN']])
        s.b.via(3, t['EN'], en_y); s.b.sq('m4', t['EN'], en_y, 140)
        pins[role + '.EN'].append((t['EN'], en_y, 'm4'))
        if replica:
            # DIN gates join the nearest VSS rail through the DIN track.
            vss = [y for net, y in s.boundaries if net == 'VSS']
            ys = [y for y, x0, x1 in s.bars['VSS'] if x0 <= t['DIN'] <= x1]
            low = min(ys)
            rail = min(vss, key=lambda y: abs(y - low))
            s.track('VSS', t['DIN'], extra=(rail,))
            s.b.via(1, t['DIN'], rail); s.b.via(2, t['DIN'], rail)
            s.rect('m2', t['DIN'] - 35, rail - 70, t['DIN'] + 35, rail + 70)
        else:
            s.track(n['DIN'], t['DIN'])
    if not replica:
        for role, key in (('U', 'X_U'), ('L', 'X_L')):
            s.track(role + 'B.X', geo['buffer'][key])
        for role, key in (('U', 'A_U'), ('L', 'A_L')):
            lo, hi = s.track(role + 'B.A', geo['buffer'][key], extra=(-300.,))
            pins[role + 'B.A'].append((geo['buffer'][key], -300., 'm3'))
    supplies = s.supplies(geo['supply_x'])
    s.audit()
    s.b.rect('bnd', 0, 0, geo['width'], s.y)
    v = SlotView(s.b, geo['width'], s.y, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)
    return v


def _precharge_row(layout, name, role, comp, geo=None):
    geo = geo or OPENRAM
    pre = role + '.'
    devs = {n: _named(d, pre, n) for n, d in _devices(comp).items()}
    net = lambda n: n if n == 'VDD' else pre + n
    gmap = {n: net(n) for n in ('BL', 'BLB', 'ENB', 'VDD')}
    w = devs['M1'].width_m * 1e9
    # Each row holds one or two fingers of every device (BL VDD BLB BL ...).
    # BL/BLB bars sit on strip levels 1 and 2, which needs >= 355 nm fingers.
    fingers = 1 if w / 2 < 355 else 2
    wf = _even(w, fingers)
    gates, p_sd = [], ['BL']
    cycle = [('M1', 'VDD'), ('M2', 'BLB'), ('M3', 'BL')]
    for k in range(fingers):
        for dev, nxt in cycle:
            gates.append((Gate(devs[dev], k, wf, gmap), None))
            p_sd.append(nxt)
    p_sd = [gmap[n] for n in p_sd]
    t = geo['tracks'][role]
    levels = {('g', gmap['ENB']): 0, ('p', gmap['BL']): 1, ('p', gmap['BLB']): 2}
    reach = {gmap['BL']: [t['BL']], gmap['BLB']: [t['BLB']], gmap['ENB']: [geo['pre_tap']]}
    # The PRE M4 bar on the gate level stays >= 280 nm from the BL/BLB M4
    # landings: the P region is at least 465 nm even for narrow fingers.
    return make_row(layout, f'{name}_{role}_PC', gates, p_sd, None, levels, reach, width=geo['width'],
                    min_region={'p': 465.})


def make_precharge_slot(layout, name, precharges, replica=False, geometry=None):
    """Two mirrored P-only rows: L (column c) below, U (column c+1) above."""
    geo = geometry or OPENRAM
    s = _Slot(layout, name)
    lower = _precharge_row(layout, name, 'L', precharges.get('L', precharges['U']), geo)
    upper = _precharge_row(layout, name, 'U', precharges['U'], geo)
    if replica:
        s.y += lower.height
    else:
        s.add(lower, False, None)
    s.add(upper, True, None)
    pins = defaultdict(list)
    for role in (('U',) if replica else ('L', 'U')):
        t, tap = geo['tracks'][role], geo['pre_tap']
        for out in ('BL', 'BLB'):
            ys = s.stack(role + '.' + out, t[out])
            pins[role + '.' + out].append((t[out], min(ys), 'm5'))
        (y, x0, x1), = s.bars[role + '.ENB']
        s.b.via(2, tap, y); s.via(3, tap, y)
        s.m3.append((role + '.ENB', tap, y - 70, y + 70))
        pins[role + '.ENB'].append((tap, y, 'm4'))
    supplies = s.supplies(geo['supply_x'])
    s.audit()
    s.b.rect('bnd', 0, 0, geo['width'], s.y)
    return SlotView(s.b, geo['width'], s.y, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)


# Data registers: each source DFF folds into two rows (A: data latch, B:
# clock inverter and output latch) with its frozen 500/250 nm devices.
# Track x per role; Q tracks continue the write slot's buffer-input tracks.
REGISTER_TRACKS = {
    'CLK': 140.,
    'L': dict(CLKB=280., z2=420., D=700., Q=BUFFER['A_L'],
              A=(840., 980., 1120.), B=(700., 840., 1120.)),
    'U': dict(CLKB=280., z2=420., D=560., Q=BUFFER['A_U'],
              A=(700., 840., 1120.), B=(840., 1120., 1260.)),
}


# Compact registers: the same rows with searched track positions and the
# fewest gate and strip bar levels make_row accepts (_gate_levels,
# _strip_levels). Q keeps the write slot's buffer-input tracks. The track set
# came from an annealed search over 35 nm positions that checked every
# candidate with make_row and the slot's M3 audit: 7730 nm per two registers
# against 9750 nm for REGISTER_TRACKS.
COMPACT_REGISTER_TRACKS = {
    'CLK': 140.,
    'L': dict(CLKB=280., z2=700., D=140., Q=BUFFER['A_L'], D_b=420., z1=1050., z3=1225.,
              z4=1190., z5=840., QB=420.),
    'U': dict(CLKB=280., z2=1120., D=560., Q=BUFFER['A_U'], D_b=700., z1=1260., z3=840.,
              z4=560., z5=1260., QB=840.),
}


# Default slot geometry: the OpenRAM 1410 nm column pair.
OPENRAM = dict(width=SLOT, tracks=TRACKS, buffer=BUFFER, pre_tap=PRE_TAP, supply_x=SUPPLY_X,
               register=COMPACT_REGISTER_TRACKS, buffer_swap=False)


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


def _register_rows(layout, name, role, dff, tracks=None, width=SLOT):
    """Two rows of one transmission-gate DFF; nets are role-prefixed.

    tracks (a COMPACT_REGISTER_TRACKS entry) replaces the fixed level plan by
    the fewest legal levels for those track positions.
    """
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
    t = REGISTER_TRACKS[role]
    # Row A: VDD inv2(D) D_b tg1 z1 tg2 z3 inv4(z2) VDD inv3(z1) z2
    a_gates = [gates('Xinv2_D'), gates('Xtg1', 2, 1), gates('Xtg2', 1, 0), gates('Xinv4'), gates('Xinv3')]
    a_p = ['VDD', net('D_b'), net('z1'), net('z3'), 'VDD', net('z2')]
    a_n = ['VSS', net('D_b'), net('z1'), net('z3'), 'VSS', net('z2')]
    a_levels = {('g', 'CLK'): 0, ('g', net('CLKB')): 1, ('g', net('D')): 3, ('g', net('z2')): 4, ('g', net('z1')): 5}
    for r in ('p', 'n'):
        a_levels.update({(r, net('D_b')): 0, (r, net('z1')): 1, (r, net('z3')): 2, (r, net('z2')): 3})
    la, lb = t['A'], t['B']
    a_reach = {'CLK': [REGISTER_TRACKS['CLK']], net('CLKB'): [t['CLKB']], net('z2'): [t['z2']], net('D'): [t['D']],
               net('D_b'): [la[0]], net('z1'): [la[1]], net('z3'): [la[2]]}
    row_a = make_row(layout, f'{name}_{role}_REGA', a_gates, a_p, a_n, a_levels, a_reach)
    # Row B: CLKB inv1(CLK) VDD inv5(z2) z4 tg3 z5 tg4 QB inv7(Q) VDD inv6(z5) Q
    b_gates = [gates('Xinv1_clk', 0, 0), gates('Xinv5'), gates('Xtg3', 1, 0), gates('Xtg4', 2, 1),
               gates('Xinv7'), gates('Xinv6')]
    b_p = [net('CLKB'), 'VDD', net('z4'), net('z5'), net('QB'), 'VDD', net('Q')]
    b_n = [net('CLKB'), 'VSS', net('z4'), net('z5'), net('QB'), 'VSS', net('Q')]
    b_levels = {('g', 'CLK'): 0, ('g', net('CLKB')): 1, ('g', net('z2')): 3, ('g', net('Q')): 4, ('g', net('z5')): 5}
    for r in ('p', 'n'):
        b_levels.update({(r, net('CLKB')): 0, (r, net('z4')): 1, (r, net('z5')): 2, (r, net('QB')): 3, (r, net('Q')): 4})
    b_reach = {'CLK': [REGISTER_TRACKS['CLK']], net('CLKB'): [t['CLKB']], net('z2'): [t['z2']], net('Q'): [t['Q']],
               net('z4'): [lb[0]], net('z5'): [lb[1]], net('QB'): [lb[2]]}
    if tracks is not None:
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
    row_b = make_row(layout, f'{name}_{role}_REGB', b_gates, b_p, b_n, b_levels, b_reach)
    return row_a, row_b


def make_register_slot(layout, name, dff, roles=('L', 'U'), compact=False, geometry=None):
    """Data registers of columns c (L, bottom rows) and c+1 (U, top rows).

    D inputs enter from below on M3, Q outputs leave the top on the write
    slot's buffer-input tracks, and one shared CLK track meets an M4 bar.
    compact uses COMPACT_REGISTER_TRACKS and the fewest legal levels; a
    geometry supplies its own (compact-style) register table and width.
    """
    geo = geometry or OPENRAM
    s = _Slot(layout, name)
    table = geo['register'] if geometry else COMPACT_REGISTER_TRACKS if compact else None
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
    clk = (table or REGISTER_TRACKS)['CLK']
    s.track('CLK', clk)
    clock_y = min(ys)
    s.b.via(3, clk, clock_y); s.b.sq('m4', clk, clock_y, 140)
    pins['CLK'].append((clk, clock_y, 'm4'))
    for role in roles:
        pre = role + '.'
        if table is None:
            t = REGISTER_TRACKS[role]
            t = dict(t, **dict(zip(('D_b', 'z1', 'z3'), t['A'])), **dict(zip(('z4', 'z5', 'QB'), t['B'])))
        else:
            t = table[role]
        for key in ('CLKB', 'z2', 'D_b', 'z1', 'z3', 'z4', 'z5', 'QB'):
            s.track(pre + key, t[key])
        s.track(pre + 'D', t['D'], extra=(-300.,))
        pins[pre + 'D'].append((t['D'], -300., 'm3'))
        # Q climbs past the band top to overlap the write slot's input track.
        s.track(pre + 'Q', t['Q'], extra=(top + 370.,))
        pins[pre + 'Q'].append((t['Q'], top + 370., 'm3'))
    supplies = s.supplies(geo['supply_x'])
    s.audit()
    s.b.rect('bnd', 0, 0, geo['width'], top)
    return SlotView(s.b, geo['width'], top, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)
