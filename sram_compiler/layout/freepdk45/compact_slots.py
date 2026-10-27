"""Column-pair slots at the compact pitch: 2250 nm, columns c (L) and c+1 (U).

Odd columns mirror, so the pair's bitlines sit at 172.5 (BL c), 952.5
(BLB c), 1297.5 (BLB c+1) and 2077.5 nm (BL c+1). Precharge uses two
mirrored diffusion islands in one row; registers reuse the dense_slots
builder with mirror-symmetric track tables.
The sense slot is new: one 2:1 column mux and the sense amplifier of
its group, folded into three dense_rows rows:

* SEL inverters (SEL0 -> SELB0, SEL1 -> SELB1);
* isolated BL and BLB transmission-gate islands sharing select buses;
* the amplifier: P strip IN Q VDD QB INB, N strip Q net1 VSS net1 QB, with
  the tail device folded into two fingers.

The write slot puts data buffers and input inverters in one row of four
diffusion islands, followed by one row containing both BL and BLB tristates
(one full-width finger per device). Both columns share the source EN net.

The replica slots keep the replica column in the U position; its mux input
1, select inputs and DIN are tied to the rails as in the source.
"""
from collections import defaultdict

from .dense_rows import make_row, Gate, RowView
from .dense_rows import X0, PITCH
from .dense_slots import _Slot, SlotView, _devices, _named, _even
from .lib import Builder, pya
from .components import Component

WIDTH = 2250.
_L = dict(BL=172.5, BLB=952.5, DIN=312.5, DINB=452.5, ENB=592.5, EN=732.5)
TRACKS = {'L': _L, 'U': {k: WIDTH - x for k, x in _L.items()}}
BUFFER = dict(A_L=872.5, X_L=140., A_U=WIDTH - 872.5, X_U=WIDTH - 140.)
# Register tracks: the 1410 nm searched set with Q moved to the wider buffer
# inputs; a conflict-repair loop then moved L.z5 and four U tracks (7730 nm).
REGISTER = {
    'CLK': 140.,
    'L': dict(CLKB=280., z2=700., D=140., Q=BUFFER['A_L'], D_b=420., z1=1050., z3=1225.,
              z4=1190., z5=560., QB=420.),
    'U': dict(CLKB=280., z2=1120., D=1680., Q=BUFFER['A_U'], D_b=700., z1=1260., z3=420.,
              z4=560., z5=1855., QB=2030.),
}
GEOMETRY = dict(width=WIDTH, tracks=TRACKS, buffer=BUFFER, pre_tap=WIDTH / 2, supply_x=0.,
                register=REGISTER, buffer_swap=True, buffer_pitch=300.)
# Sense-slot M3 tracks (left half for group input 0 / SA_IN, right half mirrored).
SENSE = dict(SEL0=312.5, SELB0=452.5, SA_IN=592.5, Q=732.5,
             SEL1=WIDTH - 312.5, SELB1=WIDTH - 452.5, SA_INB=WIDTH - 592.5, QB=WIDTH - 732.5,
             EN=WIDTH / 2, ISO=WIDTH / 2)


def _row(layout, name, columns, p_sd, n_sd, levels, reach, min_region=None, pitch=PITCH):
    return make_row(layout, name, columns, p_sd, n_sd, levels, reach, width=WIDTH, min_region=min_region, pitch=pitch)


def _strip_levels(kind, nets, reach):
    """Interval colouring of a strip's bars; only the strip's own rail
    supply straps without a bar (a replica's tied-high N contact needs one)."""
    rail = 'VDD' if kind == 'p' else 'VSS'
    sites = defaultdict(list)
    for i, net in enumerate(nets):
        if net != rail:
            sites[net].append(X0 + PITCH * i)
    levels, ends = {}, []
    for lo, hi, net in sorted((min(v + reach.get(n, [])) - 70, max(v + reach.get(n, [])) + 70, n)
                              for n, v in sites.items()):
        level = next((k for k, e in enumerate(ends) if lo - e >= 70), len(ends))
        if level == len(ends):
            ends.append(hi)
        else:
            ends[level] = hi
        levels[(kind, net)] = level
    return levels


def make_sense_slot(layout: pya.Layout, name: str, mux: Component, sa: Component,
                    replica: bool = False) -> SlotView:
    """mux: COLUMNMUX2 source component, sa: SENSEAMP component.

    Device names keep the M./A. role prefix (mux / amplifier). replica maps
    mux input 0 to the replica bitlines in the U position and ties input 1
    and both selects to the rails, as the source does.
    """
    s = _Slot(layout, name)
    md = {n: _named(d, 'M.', n) for n, d in _devices(mux).items()}
    ad = {n: _named(d, 'A.', n) for n, d in _devices(sa).items()}
    if set(md) != {f'{k}_{i}' for k in ('MInvp', 'MInvn', 'MMuxn_BL', 'MMuxp_BL', 'MMuxn_BLB', 'MMuxp_BLB')
                   for i in (0, 1)} or set(ad) != {f'M{i}' for i in range(1, 8)}:
        raise ValueError('Compact sense slot requires the source COLUMNMUX2 and SENSEAMP')
    # Row nets. Normal: inputs are the pair's bitlines; replica: RBL/RBLB in
    # the U position, input 1 at VDD, SEL0 at VDD and SEL1 at VSS.
    if replica:
        net = dict(BL0='RBL', BL1='VDD', BLB0='RBLB', BLB1='VDD', SEL0='VDD', SEL1='VSS',
                   SELB0='M.SELB0', SELB1='M.SELB1', SA_IN='RBL_MUX', SA_INB='RBLB_MUX')
        bl_x = dict(BL0=TRACKS['U']['BL'], BLB0=TRACKS['U']['BLB'])
    else:
        net = dict(BL0='L.BL', BL1='U.BL', BLB0='L.BLB', BLB1='U.BLB', SEL0='SEL0', SEL1='SEL1',
                   SELB0='M.SELB0', SELB1='M.SELB1', SA_IN='SA_IN', SA_INB='SA_INB')
        bl_x = dict(BL0=TRACKS['L']['BL'], BL1=TRACKS['U']['BL'], BLB0=TRACKS['L']['BLB'], BLB1=TRACKS['U']['BLB'])
    supply = {'VDD', 'VSS'}
    # Supply-held nets with bars meet a rail tie track: the select taps for
    # the selects, the unused L-position bitline x for input 1.
    ties = defaultdict(list)
    if replica:
        ties['VDD'].append(SENSE['SEL0'])
        ties['VSS'].append(SENSE['SEL1'])
    mux_map = {'VDD': 'VDD', 'VSS': 'VSS', 'SA_IN': net['SA_IN'], 'SA_INB': net['SA_INB'],
               'SEL0': net['SEL0'], 'SEL1': net['SEL1'], 'BL0': net['BL0'], 'BL1': net['BL1'],
               'BLB0': net['BLB0'], 'BLB1': net['BLB1'], 'SELB0': net['SELB0'], 'SELB1': net['SELB1']}
    track = lambda key: [SENSE[key]]
    def reach_of(nets):
        out = {}
        for key, n in nets.items():
            if n in supply:
                out[n] = ties[n]
                continue
            out[n] = [bl_x[key]] if key in bl_x else track(key) if key in SENSE else []
        return out
    g = lambda d, f, w, m: Gate(d, f, w, m)
    w135 = lambda d: d.width_m * 1e9
    # 1: SEL inverters
    inv = [(g(md['MInvp_0'], 0, w135(md['MInvp_0']), mux_map), g(md['MInvn_0'], 0, w135(md['MInvn_0']), mux_map)),
           (g(md['MInvp_1'], 0, w135(md['MInvp_1']), mux_map), g(md['MInvn_1'], 0, w135(md['MInvn_1']), mux_map))]
    keys = dict(SEL0=net['SEL0'], SEL1=net['SEL1'], SELB0=net['SELB0'], SELB1=net['SELB1'])
    reach = reach_of(keys)
    p_sd = [net['SELB0'], 'VDD', net['SELB1']]
    n_sd = [net['SELB0'], 'VSS', net['SELB1']]
    # Fixed levels keep normal and replica rows the same height.
    levels = {('g', net['SEL0']): 0, ('g', net['SEL1']): 0}
    for k in ('p', 'n'):
        levels.update({(k, net['SELB0']): 0, (k, net['SELB1']): 0})
    # Two gates need little of the slot width. Spreading them lets both
    # select gates and their disjoint output bars share one level.
    rows = [('inv', _row(layout, f'{name}_SEL', inv, p_sd, n_sd, levels, reach, pitch=400.))]
    # One switch row: separate BL/BLB diffusion islands share the rails,
    # wells and select buses. Each island still contains both source TGs.
    switch = []
    for tag, out in (('BL', 'SA_IN'), ('BLB', 'SA_INB')):
        cols = [(g(md[f'MMuxp_{tag}_0'], 0, w135(md[f'MMuxp_{tag}_0']), mux_map),
                 g(md[f'MMuxn_{tag}_0'], 0, w135(md[f'MMuxn_{tag}_0']), mux_map)),
                (g(md[f'MMuxp_{tag}_1'], 0, w135(md[f'MMuxp_{tag}_1']), mux_map),
                 g(md[f'MMuxn_{tag}_1'], 0, w135(md[f'MMuxn_{tag}_1']), mux_map))]
        sd = [net[tag + '0'], net[out], net[tag + '1']]
        keys = {tag + '0': net[tag + '0'], tag + '1': net[tag + '1'], out: net[out],
                'SEL0': net['SEL0'], 'SEL1': net['SEL1'], 'SELB0': net['SELB0'], 'SELB1': net['SELB1']}
        reach = reach_of(keys)
        levels = {('g', net['SEL0']): 0, ('g', net['SEL1']): 1, ('g', net['SELB0']): 2, ('g', net['SELB1']): 3}
        # Bitline bars stay off strip level 0: their M4 landings must clear
        # the column-edge supply stacks 172.5 nm away.
        for k in ('p', 'n'):
            rail = 'VDD' if k == 'p' else 'VSS'
            levels.update({(k, n): v for n, v in ((net[out], 0), (sd[0], 3 if replica and tag == 'BLB' else 1),
                                                 (sd[2], 2 if replica or tag == 'BL' else 3)) if n != rail})
        # The replica's tied-high P contact needs no bar; keep both slots'
        # four-level regions so their rails abut.
        switch.append(make_row(layout, f'{name}_{tag}', cols, sd, sd, levels, reach,
                               width=WIDTH, x0=100. if tag == 'BL' else 1225.,
                               min_region={'p': 495., 'n': 495.}, pitch=400.))
    rows.append(('mux', _merge_rows(layout, name + '_MUX', switch)))
    # Amplifier
    amap = {'VDD': 'VDD', 'VSS': 'VSS', 'EN': 'A.EN', 'ISO': 'A.ISO', 'IN': net['SA_IN'], 'INB': net['SA_INB'],
            'Q': 'A.Q', 'QB': 'A.QB', 'net1': 'A.net1'}
    m7 = _even(w135(ad['M7']), 2)
    cols = [(g(ad['M5'], 0, w135(ad['M5']), amap), g(ad['M1'], 0, w135(ad['M1']), amap)),
            (g(ad['M2'], 0, w135(ad['M2']), amap), g(ad['M7'], 0, m7, amap)),
            (g(ad['M4'], 0, w135(ad['M4']), amap), g(ad['M7'], 1, m7, amap)),
            (g(ad['M6'], 0, w135(ad['M6']), amap), g(ad['M3'], 0, w135(ad['M3']), amap))]
    p_sd = [amap['IN'], 'A.Q', 'VDD', 'A.QB', amap['INB']]
    n_sd = ['A.Q', 'A.net1', 'VSS', 'A.net1', 'A.QB']
    reach = {amap['IN']: track('SA_IN'), amap['INB']: track('SA_INB'), 'A.Q': track('Q'), 'A.QB': track('QB'),
             'A.EN': track('EN'), 'A.ISO': track('ISO'), 'A.net1': []}
    levels = {('g', 'A.EN'): 0, ('g', 'A.QB'): 1, ('g', 'A.Q'): 2, ('g', 'A.ISO'): 3}
    for k, sd in (('p', p_sd), ('n', n_sd)):
        levels.update(_strip_levels(k, sd, reach))
    rows.append(('amp', _row(layout, f'{name}_AMP', cols, p_sd, n_sd, levels, reach)))
    for i, (tag, row) in enumerate(rows):
        s.add(row, i % 2 == 1, None)
    pins = defaultdict(list)
    # Bitline stacks onto the column M5 lines.
    for key, x in bl_x.items():
        n = net[key]
        ys = s.stack(n, x)
        pins[n].append((x, min(ys), 'm5'))
    # Local tracks, and rail ties for nets the replica holds at a supply.
    spans = {}
    for key in ('SELB0', 'SELB1', 'SA_IN', 'SA_INB'):
        lo, hi = s.track(net[key], SENSE[key])
        if key.startswith('SA_'):
            spans[net[key]] = (SENSE[key], lo, hi)
    for key in ('Q', 'QB'):
        lo, hi = s.track('A.' + key, SENSE[key])
        spans['A.' + key] = (SENSE[key], lo, hi)
    for key in ('SEL0', 'SEL1'):
        n = net[key]
        if n in supply:
            _tie(s, n, SENSE[key])
            continue
        s.track(n, SENSE[key])
        # Distinct select levels keep their M4 bars clear of bitline stacks.
        ys = [y for y, x0, x1 in s.bars[n]]
        y = min(ys) if key == 'SEL0' else max(ys)
        s.b.via(3, SENSE[key], y); s.b.sq('m4', SENSE[key], y, 140)
        pins[n].append((SENSE[key], y, 'm4'))
    for key in ('EN', 'ISO'):
        (y, x0, x1), = s.bars['A.' + key]
        s.b.via(2, SENSE[key], y); s.via(3, SENSE[key], y)
        s.m3.append(('A.' + key, SENSE[key], y - 70, y + 70))
        pins['A.' + key].append((SENSE[key], y, 'm4'))
    supplies = s.supplies(0.)
    # Mux outputs and amplifier outputs are taken where an M4 landing clears
    # every bank-wide M4 bar crossing the slot (rails and control pins).
    bars = sorted([y for _, y in supplies] + [p[1] for k in ('SEL0', 'SEL1', 'A.EN', 'A.ISO') for p in pins.get(k, [])])
    for n, (x, lo, hi) in spans.items():
        pins[n].append((x, _window(lo + 70., hi - 70., bars), 'm3'))
    s.audit()
    s.b.rect('bnd', 0, 0, WIDTH, s.y)
    return SlotView(s.b, WIDTH, s.y, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)


def _window(lo, hi, bars, clear=280.):
    """Grid point in [lo, hi] farthest from every bar centre (at least clear)."""
    best = max((min((abs(y - b) for b in bars), default=1e9), y)
               for y in (lo + 2.5 * k for k in range(int((hi - lo) // 2.5) + 1)))
    if best[0] < clear:
        raise ValueError('No M4 landing window on the track')
    return best[1]


def _merge_rows(layout: pya.Layout, name: str, rows: list[RowView]) -> RowView:
    """Co-planar diffusion islands with common rails and bar levels."""
    b = Builder(layout, name)
    bars, terminals, fingers = {}, {}, []
    height = rows[0].height
    for row in rows:
        if row.height != height or row.rails != rows[0].rails:
            raise ValueError('Merged islands require identical rail and well geometry')
        b.place(row.builder)
        for key, (y, lo, hi) in row.bars.items():
            if key in bars:
                old, a, z = bars[key]
                if old != y:
                    raise ValueError('Merged bars disagree on their level')
                lo, hi = min(lo, a), max(hi, z)
            bars[key] = (y, lo, hi)
        terminals.update(row.terminals)
        fingers.extend(row.fingers)
    # Same-net islands need one continuous bus; check every other bus on
    # that level before drawing the union (including its via enclosure).
    for key, (y, lo, hi) in bars.items():
        for other, (oy, a, z) in bars.items():
            if key[0] != other[0] and y == oy and max(lo, a) - min(hi, z) < 210.:
                raise ValueError(f'Merged bars {key[0]} and {other[0]} collide')
        b.rect('m2', lo - 70., y - 35., hi + 70., y + 35.)
    return RowView(b, WIDTH, height, bars, terminals, fingers, rows[0].rails)


def make_precharge_slot(layout: pya.Layout, name: str, precharges: dict[str, Component],
                        replica: bool = False) -> SlotView:
    """Pitch-aligned precharge pair sharing one PRE bus and one VDD rail.

    Each column keeps its own three-device diffusion island. The U island
    mirrors L about the pair centre, matching the mirrored array bitlines.
    Full-width fingers use the spare horizontal pitch to avoid a second
    row. VDD faces the sense slot so the two bands share their rail. The
    replica retains only U, at the same coordinates as a normal pair.
    """
    s = _Slot(layout, name)
    rows = []
    roles = ('U',) if replica else ('L', 'U')
    for role in roles:
        devs = {n: _named(d, role + '.', n) for n, d in _devices(precharges[role]).items()}
        if set(devs) != {'M1', 'M2', 'M3'}:
            raise ValueError('Compact precharge requires the source three-PMOS cell')
        nets = dict(VDD='VDD', ENB='PRE', BL=role + '.BL', BLB=role + '.BLB')
        order = ('M1', 'M2', 'M3') if role == 'L' else ('M3', 'M2', 'M1')
        sd = ('BL', 'VDD', 'BLB', 'BL') if role == 'L' else ('BL', 'BLB', 'VDD', 'BL')
        gates = [(Gate(devs[n], 0, devs[n].width_m * 1e9, nets), None) for n in order]
        levels = {('g', 'PRE'): 0, ('p', nets['BL']): 1, ('p', nets['BLB']): 2}
        reach = {nets[n]: [TRACKS[role][n]] for n in ('BL', 'BLB')}
        reach['PRE'] = [WIDTH / 2]
        # 115 nm leaves the outer implant/endcap margin; the opposite
        # island is its exact x mirror. Both bars clear the central gap.
        x0 = 115. if role == 'L' else WIDTH - 115. - 3 * PITCH
        rows.append(make_row(layout, name + '_' + role + '_PC', gates,
                             [nets[n] for n in sd], None, levels, reach,
                             width=WIDTH, x0=x0, min_region={'p': 465.}))
    s.add(_merge_rows(layout, name + '_ROW', rows), True, None)
    pins = defaultdict(list)
    for role in roles:
        for net in ('BL', 'BLB'):
            x = TRACKS[role][net]
            ys = s.stack(role + '.' + net, x)
            pins[role + '.' + net].append((x, min(ys), 'm5'))
    (y, _, _), = s.bars['PRE']
    tap = WIDTH / 2
    s.b.via(2, tap, y); s.via(3, tap, y)
    s.m3.append(('PRE', tap, y - 70, y + 70))
    for role in roles:
        pins[role + '.ENB'].append((tap, y, 'm4'))
    supplies = s.supplies(0.)
    s.audit()
    s.b.rect('bnd', 0, 0, WIDTH, s.y)
    return SlotView(s.b, WIDTH, s.y, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)


def _tie(s, net, x):
    """Join every bar of a supply-held net at x to the nearest rail of it."""
    ys = [y for y, x0, x1 in s.bars[net] if x0 - 1e-6 <= x <= x1 + 1e-6]
    rails = [y for n, y in s.boundaries if n == net]
    if not ys or not rails:
        raise ValueError(f'No {net} bar or rail at x={x}')
    rail = min(rails, key=lambda r: min(abs(r - y) for y in ys))
    s.track(net, x, extra=(rail,))
    s.b.via(1, x, rail); s.b.via(2, x, rail)
    s.rect('m2', x - 35, rail - 70, x + 35, rail + 70)


EN_TRACK = _L['EN']       # the write slot's shared EN track


def _write_front_row(layout: pya.Layout, name: str, devs: dict, buffers: dict[str, Component],
                     nets: dict, replica: bool) -> RowView:
    """Buffers and input inverters in four isolated, co-planar islands.

    Single fingers preserve the frozen widths; reversing the left buffer
    lets strip buses use three levels. The replica retains only its input
    inverters, with the same wells and rail positions as a full pair.
    """
    gate_levels = {'LB.X': 0, 'L.DIN': 1, 'LB.A': 2, 'EN': 0,
                   'U.DIN': 1, 'UB.A': 2, 'UB.X': 0}
    strip_levels = {'L.DIN': 0, 'LB.X': 1, 'L.DINB': 2, 'L.ENB': 0,
                    'U.DINB': 1, 'U.ENB': 0, 'UB.X': 2, 'U.DIN': 0}
    reach = {'EN': [EN_TRACK]}
    for role in ('L', 'U'):
        reach.update({role + 'B.A': [BUFFER['A_' + role]], role + 'B.X': [BUFFER['X_' + role]]})
        reach.update({role + '.' + n: [TRACKS[role][n]] for n in ('DIN', 'DINB', 'ENB')})
    # All islands share the largest source width and bar-region height.
    bd = {r: {n: _named(d, r + 'B.', n) for n, d in _devices(buffers[r]).items()} for r in ('L', 'U')}
    region = {kind: max(355., max(d.width_m * 1e9 for r in ('L', 'U')
                                for d in (*bd[r].values(), *(devs[r][f'M{i}'] for i in range(1, 5)))
                                if d.model == model))
              for kind, model in (('p', 'PMOS_VTG'), ('n', 'NMOS_VTG'))}
    rows = []
    for index, (role, kind) in enumerate((('L', 'buf'), ('L', 'inv'), ('U', 'inv'), ('U', 'buf'))):
        if replica and (role == 'L' or kind == 'buf'):
            continue
        if kind == 'buf':
            cols = []
            for stage in ((1, 0) if role == 'L' else (0, 1)):
                inp = role + ('B.A' if stage == 0 else 'B.X')
                out = role + ('B.X' if stage == 0 else '.DIN')
                mapping = dict(A=inp, Z=out, VDD='VDD', VSS='VSS')
                cols.append(tuple(Gate(bd[role][f'Xinv{stage}/Mpinv_{k}'], 0,
                                       bd[role][f'Xinv{stage}/Mpinv_{k}'].width_m * 1e9, mapping)
                                  for k in ('pmos', 'nmos')))
            sd = [role + '.DIN', 'VDD', role + 'B.X'] if role == 'L' else [role + 'B.X', 'VDD', role + '.DIN']
        else:
            cols = [tuple(Gate(devs[role][f'M{i}'], 0, devs[role][f'M{i}'].width_m * 1e9, nets[role])
                          for i in pair) for pair in ((1, 2), (3, 4))]
            sd = [role + '.DINB', 'VDD', role + '.ENB']
        levels = {('g', 'VSS' if replica and n == 'U.DIN' else n): lv for n, lv in gate_levels.items()}
        levels.update({(k, n): lv for n, lv in strip_levels.items() for k in ('p', 'n')})
        n_sd = ['VSS' if n == 'VDD' else n for n in sd]
        if replica:
            reach['VSS'] = [TRACKS['U']['DIN']]
        rows.append(make_row(layout, f'{name}_{role}_{kind.upper()}', cols, sd, n_sd, levels, reach,
                             width=WIDTH, x0=100. + 535. * index, min_region=region, min_gate_level=2))
    return _merge_rows(layout, name + '_FRONT', rows)


def make_write_slot(layout: pya.Layout, name: str, writes: dict[str, Component],
                    buffers: dict[str, Component], replica: bool = False) -> SlotView:
    """writes/buffers: role ('L' column c, 'U' column c+1) -> source component.

    Rows, bottom to top: combined data buffers/input inverters, then
    isolated L/U islands containing both BL and BLB tristates. Gate and strip
    levels are fixed so the replica slot (U only, DIN at VSS) keeps every row
    height. Its upper VSS rail abuts the sense slot's lower VSS rail.
    """
    s = _Slot(layout, name)
    roles = ('U',) if replica else ('L', 'U')
    ref = writes['U']
    devs = {r: {n: _named(d, r + '.', n) for n, d in _devices(writes.get(r, ref)).items()} for r in ('L', 'U')}

    def nets(r: str) -> dict[str, str]:
        n = {k: r + '.' + k for k in ('DIN', 'DINB', 'ENB', 'BL', 'BLB', 'int1', 'int2', 'int3', 'int4')}
        n.update(VDD='VDD', VSS='VSS', EN='EN')
        if replica:
            n['DIN'] = 'VSS'
        return n
    nm = {r: nets(r) for r in ('L', 'U')}
    # A single full-width finger per device lets BL and BLB share the
    # central supply contact in each column island. The two columns remain
    # isolated, with identical rail geometry in the replica's U island.
    rows = []
    for index, r in enumerate(('L', 'U')):
        if r not in roles:
            continue
        n, t = nm[r], TRACKS[r]
        pairs = ((6, 7), (5, 8), (9, 12), (10, 11))
        cols = [tuple(Gate(devs[r][f'M{i}'], 0, devs[r][f'M{i}'].width_m * 1e9, n)
                      for i in pair) for pair in pairs]
        p_sd = [n['BL'], n['int1'], 'VDD', n['int3'], n['BLB']]
        n_sd = [n['BL'], n['int2'], 'VSS', n['int4'], n['BLB']]
        levels = {('g', 'EN'): 0, ('g', n['DINB']): 1,
                  ('g', n['DIN']): 2, ('g', n['ENB']): 3}
        for kind, sd in (('p', p_sd), ('n', n_sd)):
            levels.update({(kind, net): 1 if net == n['BL'] else 2 if net == n['BLB'] else 0
                           for net in sd if net not in ('VDD', 'VSS')})
        reach = {n[key]: [t[key]] for key in ('BL', 'BLB', 'DIN', 'DINB', 'ENB')}
        reach['EN'] = [EN_TRACK]
        rows.append(make_row(layout, f'{name}_{r}_TRI', cols, p_sd, n_sd, levels, reach,
                             width=WIDTH, x0=X0 + 1125. * index))
    tristates = _merge_rows(layout, name + '_TRI', rows)
    front = _write_front_row(layout, name, devs, buffers if not replica else {'U': buffers['U'], 'L': buffers['U']}, nm, replica)
    for i, row in enumerate([front, tristates]):
        s.add(row, i % 2 == 1, None)
    pins = defaultdict(list)
    for r in roles:
        t, n = TRACKS[r], nm[r]
        for out in ('BL', 'BLB'):
            ys = s.stack(n[out], t[out])
            pins[r + '.' + out].append((t[out], min(ys), 'm5'))
        for key in ('DINB', 'ENB'):
            s.track(n[key], t[key])
        if replica:
            _tie(s, 'VSS', t['DIN'])
        else:
            s.track(n['DIN'], t['DIN'])
    s.track('EN', EN_TRACK)
    en_y = min(y for y, x0, x1 in s.bars['EN'])
    s.b.via(3, EN_TRACK, en_y); s.b.sq('m4', EN_TRACK, en_y, 140)
    pins['EN'].append((EN_TRACK, en_y, 'm4'))
    if not replica:
        for r in ('L', 'U'):
            s.track(r + 'B.X', BUFFER['X_' + r])
            s.track(r + 'B.A', BUFFER['A_' + r], extra=(-300.,))
            pins[r + 'B.A'].append((BUFFER['A_' + r], -300., 'm3'))
    supplies = s.supplies(0.)
    s.audit()
    s.b.rect('bnd', 0, 0, WIDTH, s.y)
    return SlotView(s.b, WIDTH, s.y, s.terminals, s.fingers, dict(pins), supplies, s.m3, s.rows)
