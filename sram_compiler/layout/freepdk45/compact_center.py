"""TIME_CONTROL between the halves of a split bank at the compact pitch.

Same plan as ``center_control`` (column periphery between the lower and
upper arrays, continuous M5 bitlines, data registers folded into their
column slots below the write band, explicit interface tracks, M10 bridges
for decoder nets of both halves, inputs escaping on M10), with the compact
geometry: 2250 nm column-pair slots, one M6 trunk per control net, the
replica wordline's far end on the core's M4 riser, the mux output RBL_MUX
as the replica sense return, and SEL0/SEL1 as macro inputs. The compact
decoder's 3160 nm bands leave a block between its halves wide enough for
every controller gate, so there are no decoder lanes.
"""
from __future__ import annotations
from math import ceil, floor
from pathlib import Path
import hashlib, json, shutil

from .components import Component, Instance, CellView, read_component
from .full_source import flatten
from .dense_views import write_view, expand
from .dense_aux import make_aux
from .dense_slots import make_register_slot
from .compact_slots import GEOMETRY, WIDTH as SLOT_W
from .compact_bank import build as build_bank, split_row
from .center_control import (_imported, _latch_y, TRUNK_GAP, TOP_GAP, REG_GAP, FRAME_X, RETURN_SPACE,
                             DRIVE_SPACE, BOTTOM_GAP, ROUTER_LAYERS, BRIDGE_PITCH, LATCH_GAP, M10_HALF, M10_SPACE)
from .lib import Builder, new_layout, pya, u
from .time_control import make_control, units, centre_lanes_used
from . import macro_control
from .macro_control import Planner, bank_conductor, snap, GRID

INPUTS_EXTRA = ('SEL0', 'SEL1')     # bank inputs that are not TIME_CONTROL ports


def _register_slot(layout, deck):
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)
    registers = [(p, c, m) for p, c, m in units(tc) if p.startswith('Xdff_buf_data/')]
    return tc, registers, make_register_slot(layout, 'CMP_REGISTER_SLOT', registers[0][1], geometry=GEOMETRY)


def _register_origin(rs) -> tuple[float, float]:
    c_rel = rs.pins['CLK'][0][1]
    clock_y = ((-REG_GAP - rs.height + c_rel) // GRID) * GRID
    return clock_y - c_rel, clock_y


def register_reserve(deck) -> float:
    _, _, rs = _register_slot(new_layout(), deck)
    return BOTTOM_GAP - _register_origin(rs)[0]


def _block(deck, bm, tc):
    """Width and gate band (y span) of the block between the decoder halves,
    as build() derives them from a split bank manifest."""
    _, _, rs = _register_slot(new_layout(), deck)
    reg_y, _ = _register_origin(rs)
    band_top = int((bm['array_origin_nm'][1] - 460. - TOP_GAP) // GRID) * GRID
    width = snap(min(bm['control_trunks_nm'].values()) - TRUNK_GAP - FRAME_X)
    lower = any('lower' in ends for n, ends in bm['split']['decoder_band_pins'].items() if n in tc.ports)
    return width, (reg_y + (1000. if lower else 0.), band_top)


def _fits(tc, width, band):
    try:
        return not centre_lanes_used(tc, width, ('Xdff_buf_data/',), band, [])[0]
    except ValueError:              # gates overflow a frame without decoder lanes
        return False


def build(deck, output, bank_dir):
    source = flatten(deck); rows, cols = source.rows, source.cols
    macro_control._conductors.clear()
    groups = cols // 2
    latch = source.instances['XD_LATCH']['pins']; sense = source.instances[f'XSENSEAMP_{groups - 1}']['pins']
    if latch['D'] != sense['Q'] or latch['EN'] != sense['EN']:
        raise ValueError('The centre plan requires the last-group sense output and enable')
    bank_dir = Path(bank_dir)
    bm = json.loads((bank_dir / 'layout.json').read_text())
    if bm['source_deck_sha256'] != hashlib.sha256(Path(deck).read_bytes()).hexdigest():
        raise ValueError('Bank does not match the source deck')
    if hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest() != bm['layout_sha256']:
        raise ValueError('Bank GDS differs from its layout manifest')
    split = bm.get('split')
    if (set(bm.get('excluded_instances', [])) != {'XTIME_CONTROL', 'XRWL', 'XD_LATCH'} or not split
            or split['decoder_bridges']):
        raise ValueError('Centre integration requires a split compact bank without decoder bridges')
    ly = new_layout(); b = Builder(ly, f'OPENYIELD_COMPACT_{rows}X{cols}'); P = Planner(b)
    bank = _imported(ly, bank_dir, bm); b.place(bank)
    bp = bm['external_pins']; cx, cy = bm['array_origin_nm']
    cy -= 460.                       # upper core origin (array origin minus its bottom strap)
    yb = split['lower_array_top_nm']; cut = split['row']
    endpoints = {}; links = []
    def endpoint(net, point, alias=None):
        endpoints.setdefault(net, []).append((point, alias or net))

    # 1. Data registers folded into their column slots below the write band.
    tc, registers, rs = _register_slot(ly, deck)
    by_q = {m['Q']: (p, c, m) for p, c, m in registers}
    if len(registers) != cols or any(c.identity != registers[0][1].identity for _, c, _ in registers):
        raise ValueError('Every data column needs one identical source register')
    reg_y, clock_y = _register_origin(rs)
    if abs(reg_y - BOTTOM_GAP - yb) > 1e-6:
        raise ValueError('Register band does not match the bank reserve')
    terminals, physical, reg_pins = {}, [], {}
    clock_xs = []
    for pair in range(groups):
        x = cx + pair * SLOT_W; b.place(rs.builder, x, reg_y)
        roles = {}
        for role, col in (('L', 2 * pair), ('U', 2 * pair + 1)):
            q = f'DIN_dff{col}'
            if bp[q][0] != x + rs.pins[role + '.Q'][0][0]:
                raise ValueError(f'{q}: register Q track is not under its buffer input')
            path, c, m = by_q[q]; roles[role] = 'XTIME_CONTROL/' + path
            dx, dy, layer = rs.pins[role + '.D'][0]
            reg_pins[m['D']] = (x + dx, reg_y + dy, layer)
            endpoint(q, (bp[q][0], reg_y + rs.height, 'm3')); endpoint(q, tuple(bp[q]))
        for key, (px, py, layer) in rs.terminals.items():
            role, rest = key.split('.', 1); terminals[roles[role] + '/' + rest] = (x + px, reg_y + py, layer)
        for f in rs.fingers:
            role, rest = f['device'].split('.', 1); gx, gy = f['gate_xy_nm']
            physical.append(dict(f, device=roles[role] + '/' + rest, gate_xy_nm=(x + gx, reg_y + gy)))
        clock_xs.append(x + rs.pins['CLK'][0][0])
    trunks = dict(bm['supply_trunks_nm'])
    for net, y in rs.supplies:
        tx = trunks[net]; yy = reg_y + y
        P.wire(net, 'm4', [(cx, yy), (tx, yy)], 140.); P.stack(net, tx, yy, 4, 6)
    for net, tx in trunks.items():
        x0, y0, x1, y1 = bank_conductor(bank.cell, 'm6', tx, bp[net][1])
        low = min(reg_y + y for n, y in rs.supplies if n == net)
        if low < y0 + 72.5: P.wire(net, 'm6', [(tx, y0 + 72.5), (tx, low)])
        endpoint(net, (tx, low, 'm6'))

    # 2. Interface heights from the bank conductors. Each control net has its
    #    own M6 trunk; return pins are spread at RETURN_SPACE on the block edge.
    ctl = bm['control_trunks_nm']
    def trunk(net):
        return bank_conductor(bank.cell, 'm6', ctl[net], bp[net][1])
    drives = {}
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        _, y0, _, y1 = trunk(net)
        drives[net] = snap((y0 + y1) / 2)
    drives['clk_buf'] = clock_y
    supply_y = {}
    for net, y in bm['write_supplies']:
        supply_y.setdefault(net, y)
    far = {'pre_far': 'PRE', 'wen_far': 'w_en', 'sen_far': 's_en', 'iso_far': 'sa_iso', 'rbl': 'RBL_MUX'}
    band_top = int((cy - TOP_GAP) // GRID) * GRID
    # The mux output returns straight at its track top; SEL0/SEL1 stack to
    # M10 on their trunks. The bar returns spread at RETURN_SPACE around them.
    mx, my, _ = bp['RBL_MUX']
    sel_y = {}
    for net in INPUTS_EXTRA:
        _, sy0, _, sy1 = trunk(net); sel_y[net] = snap((sy0 + sy1) / 2)
    # Router accesses on M7 block a full RETURN_SPACE, so pins keep one grid
    # step more. The mux return jogs on M7 at the replica slot from its pin
    # height to the mux output, so the other returns keep clear of that span.
    lo_y, hi_y = reg_y + 1500., band_top - 400.
    def nearest(y, spans):
        for c in (snap(y) + GRID * k * sign for k in range(int((hi_y - lo_y) // GRID) + 1) for sign in (1, -1)):
            if lo_y <= c <= hi_y and all(c < a - RETURN_SPACE or c > b2 + RETURN_SPACE for a, b2 in spans):
                return c
        raise ValueError('Return pins do not fit the block edge')
    spans = [(y, y) for y in sel_y.values()]
    pin_y = {'rbl': nearest(my, spans)}
    spans.append((min(pin_y['rbl'], my), max(pin_y['rbl'], my)))
    for y, port in sorted((drives[far[p]], p) for p in far if p != 'rbl'):
        pin_y[port] = nearest(y, spans); spans.append((pin_y[port], pin_y[port]))
    ys = sorted(list(drives.values()) + list(supply_y.values()))
    if any(b2 - a < DRIVE_SPACE for a, b2 in zip(ys, ys[1:])):
        raise ValueError(f'Centre drive pins closer than {DRIVE_SPACE} nm: {ys}')

    # 3. The block between the decoder halves holds every controller gate.
    width = snap(min(ctl.values()) - TRUNK_GAP - FRAME_X)
    frame_x = FRAME_X
    ends_by_net = split['decoder_band_pins']
    bridged = sorted((n for n, ends in ends_by_net.items() if len(ends) == 2), key=lambda n: ends_by_net[n]['lower'][0])
    if any(ends_by_net[n]['lower'][0] != ends_by_net[n]['upper'][0] for n in bridged):
        raise ValueError('Decoder halves disagree on a bridged trunk phase')
    decoder_nets = {n: [('lower' if 'lower' in ends else 'upper', *ends['lower' if 'lower' in ends else 'upper'][:2])]
                    for n, ends in ends_by_net.items() if n in tc.ports}
    edge_halves = {half for points in decoder_nets.values() for half, *_ in points}
    band_lo = reg_y + (1000. if 'lower' in edge_halves else 0.)
    used, _ = centre_lanes_used(tc, width, ('Xdff_buf_data/',), (band_lo, band_top), [])
    if used:
        raise ValueError('Controller gates exceed the block between the decoder halves')
    y0 = reg_y
    height = snap(band_top - y0 + 50)
    local = lambda y: y - y0
    profile = [(0., height, width)]
    obstacles = []
    for layer in ROUTER_LAYERS:
        li = ly.layer(*{f'm{i}': (9 + 2 * i, 0) for i in range(4, 10)}[layer])
        box = pya.Box(u(frame_x - 1500), u(y0 - 1500), u(frame_x + width + 1500), u(y0 + height + 1500))
        region = pya.Region(bank.cell.begin_shapes_rec_overlapping(li, box)) & pya.Region(box)
        for pg in region.merged().each():
            for trap in pg.decompose_trapezoids():
                bb = trap.bbox(); s = ly.dbu * 1000
                obstacles.append((layer, bb.left * s - frame_x, bb.bottom * s - y0,
                                  bb.right * s - frame_x, bb.top * s - y0, None))
    edge_y = {'upper': local(band_top) // GRID * GRID - 100., 'lower': local(reg_y) + 100.}
    grid_x = lambda x: min(snap(x - frame_x), width - 200.) + frame_x
    # Off-grid decoder trunks jog to the router grid. Neighbours 300 nm apart
    # jog the same way, so their stubs take heights 300 nm apart, the one on
    # the side the jog points away from nearest the block.
    jog_y = {}
    for half, base, step in (('lower', reg_y - 500., -300.), ('upper', band_top + 500., 300.)):
        runs = []
        for x in sorted(x for points in decoder_nets.values() for h, x, _ in points if h == half):
            if runs and x - runs[-1][-1] <= 300. + 1e-6: runs[-1].append(x)
            else: runs.append([x])
        for run in runs:
            for k, x in enumerate(run if grid_x(run[0]) > run[0] else run[::-1]):
                jog_y[half, x] = base + step * k
    def riser(half, x, y):
        px = grid_x(x); py = edge_y[half] + y0
        if px == x: return [(px, py), (px, y)]
        jy = jog_y[half, x]
        if not min(py, y) + 300. < jy < max(py, y) - 300.: raise ValueError(f'{half} decoder riser at {x} has no jog room')
        return [(px, py), (px, jy), (x, jy), (x, y)]
    for half in ('lower', 'upper'):
        xs = sorted(riser(half, x, y)[0][0] for points in decoder_nets.values() for h, x, y in points if h == half)
        if any(b2 - a < 300 for a, b2 in zip(xs, xs[1:])): raise ValueError(f'{half} decoder pins closer than 300 nm')
    shapes = {}
    for net, points in decoder_nets.items():
        for half, x, y in points:
            path = riser(half, x, y)
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                shapes.setdefault(net, pya.Region()).insert(pya.Box(u(min(ax, bx) - 72.5), u(min(ay, by) - 72.5),
                                                                    u(max(ax, bx) + 72.5), u(max(ay, by) + 72.5)))
    names = sorted(shapes)
    for i, n in enumerate(names):
        grown = shapes[n].merged().sized(u(140.) - 1)
        if any(not (grown & shapes[m]).is_empty() for m in names[i + 1:]):
            raise ValueError(f'Decoder riser {n} violates M5 spacing to another riser')
    for points in decoder_nets.values():
        for half, x, y in points:
            path = riser(half, x, y)
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                obstacles.append(('m5', min(ax, bx) - 72.5 - frame_x, min(ay, by) - 72.5 - y0,
                                  max(ax, bx) + 72.5 - frame_x, max(ay, by) + 72.5 - y0, None))
    # Bridges: stacks on each half's trunk inside the decoder (clear of its
    # band-edge M6 join), M8 jogs to columns over the array, M10 across the band.
    m6 = ly.layer(21, 0)
    def free_m6(x, y):
        box = pya.Box(u(x - 300), u(y - 300), u(x + 300), u(y + 300))
        return pya.Region(bank.cell.begin_shapes_rec_overlapping(m6, box)).is_empty()
    upper_edge = ends_by_net[bridged[0]]['upper'][1] if bridged else cy
    def stack_ys(half):
        # Each stack sits on its own trunk (an enable trunk ends at its slice),
        # 900 nm from every other net's so the M8 jogs keep their spacing.
        spans = {}
        for n in bridged:
            x, y = ends_by_net[n][half][:2]
            _, t0, _, t1 = bank_conductor(bank.cell, 'm5', x, y)
            spans[n] = (t0, t1)
        chosen = {}
        for n in sorted(bridged, key=lambda n: spans[n][1] - spans[n][0]):
            t0, t1 = spans[n]; x = ends_by_net[n][half][0]
            steps = range(int((t1 - t0 - 600) // GRID) + 1)
            cands = [t0 + 400. + GRID * k for k in steps] if half == 'upper' else [t1 - 400. - GRID * k for k in steps]
            for y in cands:
                if all(abs(y - c) >= 900. for c in chosen.values()) and free_m6(x, y):
                    chosen[n] = y; break
            else:
                raise ValueError(f'{n}: no bridge stack position on its {half} trunk')
        return [chosen[n] for n in bridged]
    bridges = [(n, ends_by_net[n]['upper'][0], cx + 1000. + BRIDGE_PITCH * i, yl, yu)
               for i, (n, yl, yu) in enumerate(zip(bridged, stack_ys('lower'), stack_ys('upper')))]
    if bridges and bridges[-1][2] + 1000. > cx + bm['core_geometry']['width_nm']:
        raise ValueError('Decoder bridges exceed the array width')
    tap_y = band_top + 100.
    def blocked(layer, x0, y0_, x1, y1_, half):
        obstacles.append((layer, min(x0, x1) - half - frame_x, min(y0_, y1_) - half - y0,
                          max(x0, x1) + half - frame_x, max(y0_, y1_) + half - y0, None))
    for n, x, xb, yl, yu in bridges:
        for y in (yl, yu):
            for layer, half in (('m5', 72.5), ('m6', 72.5), ('m7', 240.), ('m8', 242.5)): blocked(layer, x, y, x, y, half)
            blocked('m8', x, y, xb, y, 202.5)
        blocked('m5', x, upper_edge, x, tap_y, 72.5); blocked('m4', x, tap_y, x, tap_y, 70.)
    # Replica wordline far end: the core's M4 riser at the RWL load row; M7
    # over the upper array, M9 down over the decoder into the block's top.
    rwl_tap = tuple(bm['rwl_tap_nm'])
    rwl_x = width - 1500.
    rwl_pin_y = edge_y['upper'] + y0
    blocked('m9', rwl_x + frame_x, rwl_pin_y, rwl_x + frame_x, rwl_tap[1], 402.5)
    blocked('m9', rwl_x + frame_x, rwl_tap[1], rwl_x + frame_x, rwl_tap[1], 482.5)
    def plan(h: float) -> dict:
        p = {n: (width - 100., snap(local(y)), 'm6') for n, y in drives.items()}
        p.update({n: (width - 300., snap(local(pin_y[n])), 'm7') for n in far})
        p['rwl'] = (rwl_x, edge_y['upper'], 'm8')
        p.update({n: (width - 100., snap(local(y)), 'm6') for n, y in supply_y.items()})
        for net, points in decoder_nets.items():
            p[net] = [(riser(half, x, y)[0][0] - frame_x, edge_y[half], 'm5') for half, x, y in points]
        return p
    ports = [p for p in source.ports if p in tc.ports or p in ('VDD', 'VSS', 'OUT', 'OUT_B') or p in reg_pins or p in bp]
    inputs = [p for p in ports if p not in ('VDD', 'VSS', 'OUT', 'OUT_B') and p not in reg_pins and p not in bp]
    formal = source.instances['XTIME_CONTROL']['pins']
    if any(p not in tc.ports or formal[p] != p for p in inputs):
        raise ValueError('Macro inputs must be TIME_CONTROL ports of the same name')
    ctrl = make_control(ly, tc, width, exclude=('Xdff_buf_data/',), pin_plan=plan, corner=True,
                        min_height=height, band_span=(local(band_lo), local(band_top)), lanes=[],
                        profile=profile, obstacles=obstacles, escapes=tuple(inputs))
    if any(not 0 <= p['y_nm'] <= ctrl.height_nm - p['height_nm'] for p in ctrl.placements):
        raise ValueError('Controller gates were placed outside the routed frame')
    b.place(ctrl.builder, frame_x, y0)
    cp = {n: (x + frame_x, y + y0, l) for n, (x, y, l) in ctrl.pins.items()}
    terminals.update({'XTIME_CONTROL/' + n: (x + frame_x, y + y0, l) for n, (x, y, l) in ctrl.device_pins.items()})
    for f in ctrl.physical_fingers:
        f = dict(f, device='XTIME_CONTROL/' + f['device'])
        if 'gate_xy_nm' in f: f['gate_xy_nm'] = (f['gate_xy_nm'][0] + frame_x, f['gate_xy_nm'][1] + y0)
        physical.append(f)
    planned = plan(ctrl.height_nm)

    # 4. Drives: one M6 line into each control net's own trunk.
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        x, y, _ = cp[net]
        _, by0, _, by1 = trunk(net)
        ty = y if by1 - by0 > 300 else (by0 + by1) / 2
        if not by0 + 72.5 <= ty <= by1 - 72.5 and by1 - by0 > 145: raise ValueError(f'{net} drive misses its trunk')
        P.wire(net, 'm6', [(x, y), (x + 300, y), (x + 300, ty), (ctl[net], ty)])
        endpoint(net, (x, y, 'm6')); endpoint(net, (ctl[net], ty, 'm6'))
        links.append(dict(controller_port=net, route='M6 into the bank trunk', y_nm=y))
    x, y, _ = cp['clk_buf']
    if abs(y - clock_y) > GRID / 2: raise ValueError('Register clock pin is off its bar')
    P.wire('clk_buf', 'm6', [(x, y), (min(ctl.values()) - 300., y)]); P.stack('clk_buf', min(ctl.values()) - 300., y, 4, 6)
    P.wire('clk_buf', 'm4', [(min(ctl.values()) - 300., y), (max(clock_xs), y)], 140.)
    endpoint('clk_buf', (x, y, 'm6')); endpoint('clk_buf', (clock_xs[0], y, 'm4'))
    links.append(dict(controller_port='clk_buf', route='M6 + M4 register clock bar', y_nm=y))
    for net, y in supply_y.items():
        x, py, _ = cp[net]
        sx = x + 400.
        P.stack(net, sx, py, 4, 6); P.wire(net, 'm6', [(x, py), (sx, py)])
        P.wire(net, 'm4', [(sx, py), (sx, y), (cx, y)], 140.)
        endpoint(net, (x, py, 'm6')); endpoint(net, (cx, y, 'm4'))
    # 5. Decoder nets rise on M5 from a block edge into one decoder half.
    for net, points in decoder_nets.items():
        for (half, tx, ty), (px, py, _) in zip(points, planned[net]):
            px += frame_x; py += y0
            P.wire(net, 'm5', riser(half, tx, ty))
            endpoint(net, (px, py, 'm5')); endpoint(net, (tx, ty, 'm5'))
            links.append(dict(controller_port=net, half=half, route='M5 riser into the decoder', x_nm=px))
    from .control_passives import antenna_diode
    for net, x, xb, yl, yu in bridges:
        for y in (yl, yu):
            P.stack(net, x, y, 5, 8); P.wire(net, 'm8', [(x, y), (xb, y)]); P.stack(net, xb, y, 8, 10)
        P.wire(net, 'm10', [(xb, yl), (xb, yu)])
        P.wire(net, 'm5', [(x, upper_edge), (x, tap_y)]); antenna_diode(b, x, tap_y); P.stack(net, x, tap_y, 3, 5)
        endpoint('BRIDGE_' + net, (x, yl, 'm5'), net); endpoint('BRIDGE_' + net, (x, yu, 'm5'), net)
        links.append(dict(decoder_net=net, route='M5-M8 stacks, M8 jogs and an M10 bridge over the array', x_nm=x, bridge_x_nm=xb))
    # 6. Far-end returns: each bar's right end continues to a stack beside the
    #    VDD trunk, then M7 back; the mux output returns from the replica slot.
    tap = trunks['VDD'] - 300.
    bars = {}
    for port, net in far.items():
        x, ry, _ = cp[port]
        if net == 'RBL_MUX':
            P.stack('RBL_MUX', mx, my, 3, 7)
            P.wire('RETURN_rbl', 'm7', [(x, ry), (mx, ry)] + ([(mx, my)] if ry != my else []))
            endpoint('RETURN_rbl', (x, ry, 'm7'), net); endpoint('RETURN_rbl', (mx, my, 'm3'), net)
            links.append(dict(controller_port=port, bank_net=net, route='replica mux output + M7', y_nm=ry))
            continue
        # The bar of this net that reaches farthest right (into the replica slot).
        ybar, xend = max(((y, bank_conductor(bank.cell, 'm4', ctl[net], y)[2]) for y in bm['control_bars_nm'][net]),
                         key=lambda t: t[1])
        bars[port] = (xend, ybar)
        # Bars may sit closer than an M7 pad pitch (400 + 400 nm), so each
        # return climbs on M5 to its own pin height before lifting to M7.
        P.wire(net, 'm4', [(xend - 70, ybar), (tap, ybar)], 140.)
        if ry != ybar:
            P.stack(net, tap, ybar, 4, 5); P.wire(net, 'm5', [(tap, ybar), (tap, ry)]); P.stack(net, tap, ry, 5, 7)
        else:
            P.stack(net, tap, ybar, 4, 7)
        P.wire('RETURN_' + port, 'm7', [(x, ry), (tap, ry)])
        endpoint('RETURN_' + port, (x, ry, 'm7'), net); endpoint('RETURN_' + port, (xend - 100, ybar, 'm4'), net)
        links.append(dict(controller_port=port, bank_net=net, route='M4 bar end + M7', y_nm=ry))
    # 7. Replica wordline return.
    bank_conductor(bank.cell, 'm4', *rwl_tap)
    x, ry, _ = cp['rwl']
    P.stack('RWL', *rwl_tap, 4, 7); P.wire('RETURN_rwl', 'm7', [rwl_tap, (x, rwl_tap[1])])
    P.stack('RWL', x, rwl_tap[1], 7, 9); P.wire('RETURN_rwl', 'm9', [(x, rwl_tap[1]), (x, ry)]); P.stack('RWL', x, ry, 8, 9)
    endpoint('RETURN_rwl', (x, ry, 'm8'), 'RWL'); endpoint('RETURN_rwl', (*rwl_tap, 'm4'), 'RWL')
    links.append(dict(controller_port='rwl', bank_net='RWL', route='M7 above the array + M9 over the decoder'))

    # 8. Replica wordline driver on the upper decoder; output latch in the
    #    replica slot's register position.
    auxiliaries, positions, apins = {}, {}, {}
    ceiling = rows <= 32 and cols <= 32
    rwl_c = read_component(deck, 'XRWL'); rwl_v = make_aux(ly, rwl_c, ceiling)
    dec_top = bm['upper_decoder_top_nm']
    dec_right = split['upper_decoder_bbox_nm'][2]
    rpx = snap(dec_right - rwl_v.width_nm - 300.); rpy = dec_top
    latch_c = read_component(deck, 'XD_LATCH'); latch_v = make_aux(ly, latch_c, folded=True)
    lpx = cx + groups * SLOT_W + LATCH_GAP
    if lpx + latch_v.width_nm > trunks['VSS'] - 67.5 - 70:
        raise ValueError('Output latch does not fit between the register band and the supply trunks')
    lpy = _latch_y(b.cell, ly, reg_y, rs.supplies, lpx, latch_v, trunks['VSS'] + 72.5)
    for name, v, x, y in (('XRWL', rwl_v, rpx, rpy), ('XD_LATCH', latch_v, lpx, lpy)):
        auxiliaries[name] = v; positions[name] = (x, y); b.place(v.builder, x, y)
        apins[name] = {p: (px + x, py + y, l) for p, (px, py, l) in v.pins.items()}
    rp_ = apins['XRWL']; lp = apins['XD_LATCH']
    ax, ay, _ = rp_['A']; vx, vy, _ = rp_['VDD']
    P.wire('VDD', 'm3', [(ax, ay), (ax, rpy + 1100), (vx, rpy + 1100)]); P.stack('VDD', vx, vy, 1, 3)
    P.wire('VDD', 'm3', [(vx, rpy + 1100), (vx, vy)])
    zx, zy, _ = rp_['Z']
    rx_, ry_, _ = bp['RWL']
    P.wire('RWL', 'm3', [(zx, zy), (zx, ry_), (rx_, ry_)])
    endpoint('RWL_DRIVE', (zx, zy, 'm3'), 'RWL'); endpoint('RWL_DRIVE', (rx_, ry_, 'm3'), 'RWL')
    bx, by, _ = rp_['B']
    spine = tuple(bm['wl_en_spine_nm'])
    bank_conductor(bank.cell, 'm5', *spine)
    P.wire('wl_en', 'm3', [(bx, by), (bx, rpy + 385)]); P.stack('wl_en', bx, rpy + 385, 3, 4)
    P.wire('wl_en', 'm4', [(bx, rpy + 385), (spine[0], rpy + 385)]); P.stack('wl_en', spine[0], rpy + 385, 4, 5)
    P.wire('wl_en', 'm5', [(spine[0], rpy + 385), spine])
    endpoint('wl_en', (bx, by, 'm3'))
    for net, rail_x in (('VSS', 200.), ('VDD', 50.)):
        px, py, _ = rp_[net]
        P.wire(net, 'm1', [(px, py), (rail_x, py)]); P.stack(net, rail_x, py, 1, 2)
        py_probe = split['upper_decoder_bbox_nm'][1] + 1300.     # both rails cross band 0 here
        probe = pya.Box(u(rail_x) - 1, u(py_probe) - 1, u(rail_x) + 1, u(py_probe) + 1)
        rail = pya.Region(bank.cell.begin_shapes_rec(ly.layer(13, 0))).merged().interacting(pya.Region(probe))
        if rail.count() != 1: raise ValueError(f'No unique decoder {net} M2 rail')
        ry1 = rail.bbox().top * ly.dbu * 1000
        P.wire(net, 'm2', [(rail_x, ry1 - 35), (rail_x, py)])
        endpoint(net, (px, py, 'm1'))
    qx, qy, _ = bp[f'SA_Q{groups - 1}']; dx, dy, _ = lp['D']     # qy: the slot's clear M4 window
    ex, ey, _ = lp['EN']; drop = dx - 300.; sy = bars['sen_far'][1]
    bank_conductor(bank.cell, 'm4', drop, sy)
    # The data drop crosses the replica slot on M5: clear of the RBL_MUX
    # landing, the enable drop and the replica bitlines (140 nm M5 spacing).
    m5 = pya.Region(bank.cell.begin_shapes_rec(ly.layer(19, 0))).merged()
    def clear(x):
        box = pya.Box(u(x - 212.5), u(min(dy, qy) - 212.5), u(x + 212.5), u(max(dy, qy) + 212.5))
        return (m5 & pya.Region(box)).is_empty() and all(abs(x - a) >= 285. for a in (mx, drop))
    data_x = next((x for x in (dx + 50. * k for k in range(60)) if clear(x)), None)
    if data_x is None: raise ValueError('No M5 column for the latch data drop')
    P.stack('LATCH_DATA', qx, qy, 3, 4); P.wire('LATCH_DATA', 'm4', [(qx, qy), (data_x, qy)], 140.)
    P.stack('LATCH_DATA', data_x, qy, 4, 5)
    P.wire('LATCH_DATA', 'm5', [(data_x, qy), (data_x, dy)] + ([(dx, dy)] if data_x != dx else []))
    P.stack('LATCH_DATA', dx, dy, 4, 5)
    P.stack('LATCH_ENABLE', drop, sy, 4, 5); P.wire('LATCH_ENABLE', 'm5', [(drop, sy), (drop, ey)])
    P.stack('LATCH_ENABLE', drop, ey, 4, 5); P.wire('LATCH_ENABLE', 'm4', [(drop, ey), (ex, ey)], 140.)
    endpoint('LATCH_DATA', (qx, qy, 'm3'), latch['D']); endpoint('LATCH_DATA', (dx, dy, 'm4'), latch['D'])
    endpoint('LATCH_ENABLE', (drop, sy, 'm4'), latch['EN']); endpoint('LATCH_ENABLE', (ex, ey, 'm4'), latch['EN'])
    for net, px, py in latch_v.supply_rails:
        px += lpx; py += lpy
        if net == 'VDD': P.stack(net, px, py, 1, 4)
        else: P.wire(net, 'm1', [(px, py), (trunks[net], py)]); P.stack(net, trunks[net], py, 1, 6)
        endpoint(net, (px, py, 'm1'))

    # 9a. Macro edge pins: controller inputs and SEL0/SEL1 escape on M10 to
    #     the left, bottom or top edge; latch outputs leave right on M4; the
    #     supply trunks continue on M6 to the top and bottom edges.
    extent = b.cell.dbbox()
    x_left, x_right, y_bottom, y_top = (v * 1000 for v in (extent.left, extent.right, extent.bottom, extent.top))
    macro_pins = {}
    pads = {n: (x + frame_x, y + y0) for n, (x, y) in ctrl.escapes.items()}
    for net in INPUTS_EXTRA:
        tx = ctl[net]; sy = sel_y[net]
        P.stack(net, tx, sy, 6, 10); pads[net] = (tx, sy)
    def m10(path):
        r = pya.Region()
        for (ax_, ay_), (bx_, by_) in zip(path, path[1:]):
            r.insert(pya.Box(u(min(ax_, bx_) - M10_HALF), u(min(ay_, by_) - M10_HALF),
                             u(max(ax_, bx_) + M10_HALF), u(max(ay_, by_) + M10_HALF)))
        return r.merged()
    taken = pya.Region(b.cell.begin_shapes_rec(b.li('m10'))).merged()
    def free(net, path):
        return (m10(path).sized(u(M10_SPACE)) & (taken - m10([pads[net]] * 2))).is_empty()
    ends = {'left': lambda x, y: (x_left + M10_HALF, y), 'bottom': lambda x, y: (x, y_bottom + M10_HALF),
            'top': lambda x, y: (x, y_top - M10_HALF)}
    pad_ys = [y for _, y in pads.values()]; middle = (min(pad_ys) + max(pad_ys)) / 2
    vertical = lambda y: ('bottom', 'top') if y <= middle else ('top', 'bottom')
    escape = {}
    for net in sorted(pads, key=lambda n: pads[n]):
        x, y = pads[net]
        for side in ('left',) + vertical(y):
            path = [(x, y), ends[side](x, y)]
            if free(net, path): escape[net] = (side, path); taken = (taken + m10(path)).merged(); break
    columns = [frame_x + GRID * i for i in range(ceil(M10_HALF / GRID), floor((width - M10_HALF) / GRID) + 1)]
    for net in sorted(set(pads) - set(escape), key=lambda n: pads[n]):
        x, y = pads[net]
        for side in vertical(y):
            path = next(([(x, y), (c, y), ends[side](c, y)] for c in sorted(columns, key=lambda c: abs(c - x))
                         if free(net, [(x, y), (c, y), ends[side](c, y)])), None)
            if path: escape[net] = (side, path); taken = (taken + m10(path)).merged(); break
        else: raise ValueError(f'Macro input {net} has no M10 path to an edge')
    for net, (side, path) in escape.items():
        P.wire(net, 'm10', path)
        endpoint(net, (*pads[net], 'm10')); endpoint(net, (*path[-1], 'm10'))
        macro_pins[net] = [(side, *path[-1], 'm10')]
        links.append(dict(macro_port=net, side=side, route='M10 to the macro edge', path_nm=[list(pt) for pt in path]))
    for port, key in (('OUT', 'Q'), ('OUT_B', 'QB')):
        x, y, layer = lp[key]
        if layer != 'm4': raise ValueError('Latch outputs are expected on M4')
        P.wire('LATCH_' + port, 'm4', [(x, y), (x_right - 70., y)], 140.)
        endpoint('LATCH_' + port, (x, y, 'm4'), latch[key]); endpoint('LATCH_' + port, (x_right - 70., y, 'm4'), latch[key])
        macro_pins[port] = [('right', x_right - 70., y, 'm4')]
    for net, tx in trunks.items():
        _, ty0, _, ty1 = bank_conductor(bank.cell, 'm6', tx, bp[net][1])
        P.wire(net, 'm6', [(tx, y_bottom + 72.5), (tx, ty0 + 72.5)]); P.wire(net, 'm6', [(tx, ty1 - 72.5), (tx, y_top - 72.5)])
        endpoint(net, (tx, y_bottom + 72.5, 'm6')); endpoint(net, (tx, y_top - 72.5, 'm6'))
        macro_pins[net] = [('bottom', tx, y_bottom + 72.5, 'm6'), ('top', tx, y_top - 72.5, 'm6')]
    P.check(b.cell)
    if b.cell.dbbox() != extent: raise ValueError('Macro pins extend the macro boundary')
    from .conductor_ownership import port_conductors
    check_ports, check_aliases = {}, {}
    for net, points in endpoints.items():
        for i, (point, alias) in enumerate(points):
            key = f'{net}:{i}'; check_ports[key] = point; check_aliases[key] = alias.upper()
    lo_y0 = split['lower_array_origin_nm'][1]
    top_rows = rows - cut
    ix0 = max([cx + 2 * SLOT_W] + [xb + 1000 for _, _, xb, _, _ in bridges])
    interior = ([(ix0, cy + 460 + 3160, cx + (cols - 2) * 1125, cy + 460 + (top_rows - 8) * 395),
                 (ix0, lo_y0 + 3160, cx + (cols - 2) * 1125, yb - 460 - 3160)]
                if rows * cols > 4096 else None)
    port_conductors(b, check_ports, check_aliases, require_connected=True, exclude_bbox_nm=interior)

    # 9. Macro ports and the complete source component.
    ports = tuple(ports) + INPUTS_EXTRA; pins = {}
    for p in ports:
        if p in macro_pins: pins[p] = tuple(macro_pins[p][0][1:])
        elif p in reg_pins: pins[p] = reg_pins[p]
        elif p in bp: pins[p] = bp[p]
        else: raise ValueError(f'Macro port {p} has no pin')
        x, y, layer = pins[p]; b.label(layer, p, x, y)
        for _, x, y, layer in macro_pins.get(p, [])[1:]: b.label(layer, p, x, y)
    pin_records = ([dict(port=p, side=side, x_nm=x, y_nm=y, layer=layer) for p in ports for side, x, y, layer in macro_pins.get(p, [])]
                   + [dict(port=p, side='internal', x_nm=pins[p][0], y_nm=pins[p][1], layer=pins[p][2]) for p in ports if p not in macro_pins])
    bank_ports = tuple(bm['pins'])
    specs = {n: read_component(deck, n) for n in bm['included_instances']}
    bc = Component(bm['primary'], bank_ports, (), tuple(Instance(n, tuple(source.instances[n]['pins'][p] for p in c.ports), c)
                                                     for n, c in specs.items()))
    component = Component(b.name, ports, (), (Instance('XBANK', bank_ports, bc),
        Instance('XTIME_CONTROL', tuple(formal[p] for p in tc.ports), tc),
        *(Instance(n, tuple(source.instances[n]['pins'][p] for p in v.component.ports), v.component)
          for n, v in auxiliaries.items())))
    bank_pins, bank_fingers = expand(bm)
    terminals.update({'XBANK/' + n: tuple(p) for n, p in bank_pins.items()})
    physical.extend(dict(f, device='XBANK/' + f['device']) for f in bank_fingers)
    for name, v in auxiliaries.items():
        dx_, dy_ = positions[name]
        terminals.update({name + '/' + n: (x + dx_, y + dy_, l) for n, (x, y, l) in v.device_pins.items()})
        for f in v.physical_fingers:
            f = dict(f, device=name + '/' + f['device'])
            if 'gate_xy_nm' in f: f['gate_xy_nm'] = (f['gate_xy_nm'][0] + dx_, f['gate_xy_nm'][1] + dy_)
            physical.append(f)
    box = b.cell.dbbox()
    v = CellView(component, b, box.width() * 1000, box.height() * 1000, pins, terminals, [], P.segments, [])
    v.physical_fingers = physical
    bank_box = bank.cell.dbbox()
    block = [frame_x, reg_y, frame_x + width, band_top]
    metadata = dict(artifact_kind='bank', rows=rows, cols=cols, controller_layout=True, floorplan='centre',
        core_pitch_nm=dict(row=395, column=1125), row_span=rows, column_span=cols, column_mux=2,
        included_instances=list(bm['included_instances']) + ['XTIME_CONTROL', 'XRWL', 'XD_LATCH'], excluded_instances=[],
        routing='control logic: grid router inside the block between the decoder halves; interface: planned straight tracks',
        split=split, controller_frame_nm=[frame_x, y0, frame_x + width, y0 + ctrl.height_nm],
        controller_block_nm=block, controller_block_area_um2=width * (band_top - reg_y) / 1e6,
        controller_placements=[dict(p, x_nm=p['x_nm'] + frame_x, y_nm=p['y_nm'] + y0) for p in ctrl.placements],
        register_band_nm=[cx, reg_y, cx + groups * SLOT_W, reg_y + rs.height],
        controller_standard_cells=ctrl.standard_cell_inventory, passive_locations=ctrl.passive_locations,
        controller_links=links, auxiliary_positions_nm=positions,
        array_origin_nm=bm['array_origin_nm'], decoder_bbox_nm=bm['decoder_bbox_nm'], core_geometry=bm['core_geometry'],
        bank_bbox_nm=[v_ * 1000 for v_ in (bank_box.left, bank_box.bottom, bank_box.right, bank_box.top)],
        source_devices=len(source.mos), source_passives=source.passives,
        source_artifacts={str(bank_dir): hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest()},
        guard_policy=('far-end feedback and fixed digital delays; no intentional RC guards'
                      if not source.passives else 'archived intentional RC; values require independent extraction'),
        interface_conductor_audit_excluded_bbox_nm=interior,
        macro_pins=pin_records,
        macro_pin_sides={side: [r['port'] for r in pin_records if r['side'] == side]
                         for side in ('left', 'bottom', 'top', 'right', 'internal')},
        output_latch=dict(folded=True, origin_nm=[lpx, lpy], width_nm=latch_v.width_nm, height_nm=latch_v.height_nm),
        waveform_validation='not run', defaults_promotable=False, external_pins=pins,
        compiler_provenance=bm.get('compiler_provenance'))
    return write_view(v, output, deck, 'complete compact split bank with TIME_CONTROL between the decoder halves', metadata)


def generate(deck, output):
    source = flatten(deck)
    cut = split_row(source.rows)
    if cut is None: raise ValueError('The centre floorplan needs at least 16 rows')
    output = Path(output); parts = output.with_name(output.name + '_parts')
    reserve = register_reserve(deck)
    bank_dir = parts / 'bank'
    build_bank(deck, bank_dir, split=cut, reserve=reserve, bridge=False)
    # Large banks are controller-limited: the block between the decoder halves
    # widens (the core moves right) to the narrowest width, to 1 um, whose
    # band holds every controller gate.
    bm = json.loads((bank_dir / 'layout.json').read_text())
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)
    width, band = _block(deck, bm, tc)
    if not _fits(tc, width, band):
        lo, hi = width, 2 * width
        while not _fits(tc, hi, band):
            if hi > 1e6: raise ValueError('Controller gates do not fit a 1 mm block')
            lo, hi = hi, 2 * hi
        while hi - lo > 1000:
            mid = snap((lo + hi) / 2)
            lo, hi = (lo, mid) if _fits(tc, mid, band) else (mid, hi)
        shutil.rmtree(bank_dir)
        build_bank(deck, bank_dir, split=cut, reserve=reserve, bridge=False,
                   core_x=bm['array_origin_nm'][0] + hi - width + GRID)
    return build(deck, output, bank_dir)
