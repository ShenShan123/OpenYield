"""Compact point-symmetric 6T core views: 1125 nm columns x 395 nm rows.

The thin-cell topology follows foundry high-density SRAM cells: the right
half (QB side) is drawn explicitly and the left half (Q side) is its
180-degree rotation about the frame centre. Each NMOS strip carries its pass
gate and pull-down in series; both PMOS strips sit in a shared centre n-well.
An inverter gate crosses its own pull-down and pull-up and ends in a poly pad
beyond the other half's PMOS drain end, standing in for a butted contact.
Rows mirror about y=0/H and columns about x=0/W, so frame-edge diffusion
contacts and the column-edge WL poly pads are shared with the neighbours.
Device W/L are the OpenRAM/OpenYield 6T sizes (PD 205, PG 135, PU 90, L 50 nm).

Routing: BL/BLB and VDD on vertical M2, WL on M3 across the row centre, VSS
on M3 rails along both row edges. A passive replica cell ties its inverter
gates to the WL poly; the replica-column parent ties that wordline to VSS.
"""
from .components import CellView
from .lib import Builder, Sym

W = 1125.
H = 395.
CX = W / 2
X_BL = CX - 390       # unmirrored column; mirrored columns swap BL and BLB
X_BLB = CX + 390
Y_WL = H / 2
X_VDD = (CX - 92.5, CX + 92.5)
E = 92.5              # well/implant extension past the frame (overhang + 55)


def bitline_x(col):
    """(BL, BLB) x offsets inside column `col` when odd columns mirror."""
    return (X_BLB, X_BL) if col % 2 else (X_BL, X_BLB)


def draw(b, kind='cell'):
    """kind: cell | replica (QB tied to VDD) | passive (QB to VDD, inverter
    gates on the WL poly) | dummy (replica with uncontacted pass-gate drains)."""
    s = Sym(b, W, H)
    X = lambda v: CX + v
    linked = kind != 'passive'
    has_bl = kind != 'dummy'
    # wells, implants and threshold layer (vertical extension, not twinned)
    b.rect('nwell', X(-185), -E, X(185), H + E)
    b.rect('pwell', 0, -E, X(-185), H + E)
    b.rect('pwell', X(185), -E, W, H + E)
    b.rect('pplus', X(-200), -E, X(200), H + E)
    b.rect('nplus', 0, -E, X(-200), H + E)
    b.rect('nplus', X(200), -E, W, H + E)
    b.rect('vtg', 0, -E, W, H + E)
    b.rect('bnd', 0, 0, W, H)
    # PMOS_R; NMOS_R with the pass gate outer-aligned below the pull-down jog
    s.rect('active', X(40), 170, X(130), H + 37.5)
    s.polygon('active', [(X(340), -37.5), (X(475), -37.5), (X(475), H + 37.5),
                         (X(270), H + 37.5), (X(270), 170), (X(340), 170)])
    # PG_R wordline gate (lower row) with the shared column-edge pad
    s.rect('poly', X(290), 70, X(600), 120)
    s.rect('poly', X(525), 57.5, X(600), 132.5)
    # Inverter-R gate (Q): upper row, pad above PMOS_L's drain end
    s.rect('poly', X(-215), 275, X(525), 325)
    if linked:
        s.rect('poly', X(-215), 282.5, X(-140), 357.5)
    else:
        s.rect('poly', X(-290), 275, X(-215), 325)
    if has_bl:
        s.cont(X(390), 0)
    s.cont(X(307.5), 207.5)
    s.cont(X(437.5), H)
    s.cont(X(77.5), 207.5)
    s.cont(X(92.5), H)
    s.cont(W, 95)
    if linked:
        s.cont(X(-177.5), 320)
    # QB: gate-pad landing, filled riser, PMOS drain and NMOS node bar
    if linked:
        s.rect('m1', X(45), 42.5, X(245), 107.5)
        s.rect('m1', X(45), 107.5, X(210), 240)
    else:
        s.rect('m1', X(45), 140, X(110), 175)
    s.rect('m1', X(45), 240, X(110), 275)
    s.rect('m1', X(45), 175, X(375), 240)
    if kind != 'cell':
        b.rect('m1', X(45), 240, X(110), H)
    if has_bl:
        s.rect('m1', X(357.5), -67.5, X(422.5), 67.5)
    s.rect('m1', X(370), H - 32.5, X(630), H + 32.5)
    s.rect('m1', X(25), H - 32.5, X(160), H + 32.5)
    s.rect('m1', X(495), 62.5, X(630), 127.5)
    for x, y in [(W, H), (X(92.5), H), (W, 95)] + ([(X(390), 0)] if has_bl else []):
        b.sq('v1', x, y, 65)
        b.sq('v1', W - x, H - y, 65)
    if has_bl:
        s.rect('m2', X(355), -67.5, X(425), H)
    s.rect('m2', X(57.5), 0, X(127.5), H + 67.5)
    s.rect('m2', X(495), 62.5, X(630), 230)
    s.rect('m2', X(495), H - 35, X(630), H + 35)
    for x, y in [(W, Y_WL), (W, H)]:
        b.sq('v2', x, y, 65)
        b.sq('v2', W - x, H - y, 65)
    b.rect('m3', -67.5, Y_WL - 35, W + 67.5, Y_WL + 35)
    b.rect('m3', -67.5, H - 35, W + 67.5, H + 35)
    b.rect('m3', -67.5, -35, W + 67.5, 35)
    return b


HS = 460.             # well-strap row height


def draw_strap(b, channel=False):
    """One column of the well-strap row that separates array edges.

    Both strap edges face a cell row's y=0 edge, whose diffusion overhangs
    37.5 nm and whose implants reach 92.5 nm into the strap. The n-tap feeds
    an M1 VDD rail that meets the column's two VDD M2 lines; the column-edge
    p-taps (shared with mirrored neighbours) stack to an M3 VSS rail.
    Bitline and VDD M2 lines pass straight through. A channel column holds
    only the rails over p-well.
    """
    X = lambda v: CX + v
    if channel:
        b.rect('pwell', 0, 0, W, HS)
        b.rect('pplus', 0, E, W, HS - E)    # covers the neighbours' edge p-taps
    else:
        b.rect('nwell', X(-185), 0, X(185), HS)
        b.rect('pwell', 0, 0, X(-185), HS)
        b.rect('pwell', X(185), 0, W, HS)
        b.rect('nplus', X(-200), E, X(200), HS - E)
        b.rect('pplus', 0, E, X(-200), HS - E)
        b.rect('pplus', X(200), E, W, HS - E)
        b.rect('active', X(-60), 120, X(60), 210)
        b.cont(X(0), 165)
        for x in (X(-92.5), X(92.5)):
            b.sq('v1', x, 165, 65)
            b.rect('m2', x - 35, 0, x + 35, HS)
        for x in (X(-390), X(390)):
            b.rect('m2', x - 35, 0, x + 35, HS)
        for x in (0, W):
            b.rect('active', x - 45, HS - 210, x + 45, HS - 120)
            b.cont(x, HS - 165)
            b.rect('m1', x - 67.5, HS - 197.5, x + 67.5, HS - 132.5)
            b.sq('v1', x, HS - 165, 65)
            b.rect('m2', x - 67.5, HS - 200, x + 67.5, HS - 130)
            b.sq('v2', x, HS - 165, 65)
    b.rect('m1', 0, 132.5, W, 197.5)
    b.rect('m3', -67.5, HS - 200, W + 67.5, HS - 130)
    b.rect('bnd', 0, 0, W, HS)
    return b


def make_cell(layout, component, kind='cell'):
    from .dense_contract import core_contract
    core_contract(component, kind)
    if kind not in ('cell', 'replica', 'passive', 'dummy'):
        raise ValueError('Unsupported core view')
    b = draw(Builder(layout, 'CMP_' + kind.upper()), kind)
    pins = {'VDD': (X_VDD[1], H, 'm2'), 'VSS': (W, H, 'm3'), 'WL': (0, Y_WL, 'm3')}
    if kind != 'dummy':
        bl, br = ('BL', 'BLB') if kind == 'cell' else ('RBL', 'RBLB')
        pins.update({bl: (X_BL, Y_WL, 'm2'), br: (X_BLB, Y_WL, 'm2')})
    for net, (x, y, layer) in pins.items():
        b.label(layer, net, x, y)
    # Terminals refer to contacts; dummy pass-gate drains are bare diffusion.
    bl, blb, q, qb = (X_BL, H, 'm1'), (X_BLB, 0, 'm1'), (W - 870, H - 207.5, 'm1'), (870, 207.5, 'm1')
    vss_l, vss_r, vdd_l, vdd_r = (125, 0, 'm1'), (1000, H, 'm1'), (470, 0, 'm1'), (655, H, 'm1')
    wl_l, wl_r = (0, 300, 'm1'), (W, 95, 'm1')
    gate_l, gate_r = ((W - 385, H - 320, 'm1'), (385, 320, 'm1')) if kind != 'passive' else (wl_r, wl_l)
    if kind == 'dummy':
        bl, blb = (X_BL, H, 'active'), (X_BLB, 0, 'active')
    contacts = {
        'MPGL': [bl, wl_l, q, vss_l], 'MPGR': [blb, wl_r, qb, vss_r],
        'MPDL': [q, gate_l, vss_l, vss_l], 'MPDR': [qb, gate_r, vss_r, vss_r],
        'MPUL': [(W - 640, H - 207.5, 'm1'), gate_l, vdd_l, vdd_l],
        'MPUR': [(640, 207.5, 'm1'), gate_r, vdd_r, vdd_r]}
    terminals = {f'{d}@0:{t}': p for d, points in contacts.items() for t, p in zip('dgsb', points)}
    return CellView(component, b, W, H, pins, terminals, [], [], [])
