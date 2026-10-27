"""TIME_CONTROL below the shared decoder of a compact column-split bank.

Data registers stay under their own half-arrays with their bottom edges
aligned to the controller. Large controllers use spare lower decoder rows.
The bank uses M1-M6; controller routing and macro input escapes reach M10.
"""
from __future__ import annotations
from math import ceil, floor
from pathlib import Path
from typing import Sequence, Union
import hashlib, json, shutil

from .components import Component, Instance, CellView, read_component
from .full_source import flatten
from .dense_views import write_view, expand
from .dense_aux import make_aux
from .dense_slots import make_register_slot
from .compact_slots import GEOMETRY, WIDTH as SLOT_W
from .compact_bank import build as build_bank
from .lib import Builder, new_layout, pya, u
from .time_control import make_control, units, centre_lanes_used
from . import macro_control
from .macro_control import Planner, bank_conductor, snap, GRID

INPUTS_EXTRA = ('SEL0', 'SEL1')     # bank inputs that are not TIME_CONTROL ports
MIN_COLUMN_FRAME = 20600.          # smallest routed controller frame; larger sources use the fit search
COLUMN_CONTROL_DROP = 0.           # align controller and data-register bottom edges


TRUNK_GAP = 600.
TOP_GAP = 1000.
REG_GAP = 300.
FRAME_X = 35.
RETURN_SPACE = 800.
DRIVE_SPACE = 300.
ROUTER_LAYERS = ('m4', 'm5', 'm6', 'm7', 'm8', 'm9')
LATCH_GAP = 300.
M10_HALF, M10_SPACE = 402.5, 800.


def _imported(ly: pya.Layout, directory: Path, m: dict) -> Builder:
    src = new_layout(); src.read(str(directory / 'full_macro.gds'))
    cell = Builder(ly, m['primary']); cell.cell.copy_tree(src.cell(m['primary']))
    return cell


def _latch_y(cell: pya.Cell, ly: pya.Layout, reg_y: float, supplies: Sequence[tuple[str, float]],
             lpx: float, latch: CellView, right: float) -> float:
    """Bottom of the folded latch: the highest register VDD bar at its row
    boundary whose window holds no other M4, from the EN drop to the edge."""
    m4 = pya.Region(cell.begin_shapes_rec(ly.layer(17, 0))).merged()
    vx, vy, _ = latch.pins['VDD']
    for net, y in sorted(supplies, key=lambda t: -t[1]):
        if net != 'VDD': continue
        bottom = reg_y + y - vy
        box = pya.Region(pya.Box(u(lpx - 300 - 70 - 140), u(bottom - 150), u(right), u(bottom + latch.height_nm + 150)))
        hits = m4 & box
        bar = hits.interacting(pya.Region(pya.Box(u(lpx + vx) - 1, u(reg_y + y) - 1, u(lpx + vx) + 1, u(reg_y + y) + 1)))
        if not bar.is_empty() and (hits - bar).is_empty() and bar.bbox().height() <= u(140): return bottom
    raise ValueError('No register VDD bar with a free window for the output latch')


def _register_slot(layout, deck):
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)
    registers = [(p, c, m) for p, c, m in units(tc) if p.startswith('Xdff_buf_data/')]
    return tc, registers, make_register_slot(layout, 'CMP_REGISTER_SLOT', registers[0][1], geometry=GEOMETRY)


def _register_origin(rs) -> tuple[float, float]:
    c_rel = rs.pins['CLK'][0][1]
    clock_y = ((-REG_GAP - rs.height + c_rel) // GRID) * GRID
    return clock_y - c_rel, clock_y


def _column_lanes(bank: Builder, bm: dict) -> list[tuple]:
    """Unoccupied bottom gate rows of compact decoder bands, from actual metal.

    Mirrored upper rows and rows crossed by lower-metal predecode buses are
    excluded. Controller accesses also avoid every M4/M5 decoder conductor.
    """
    dx, dy, right, top = bm['decoder_bbox_nm']
    frame_x = dx + FRAME_X
    scale = bank.ly.dbu * 1000
    bounds = pya.Box(u(frame_x + 300.), u(dy), u(right), u(top))
    layers = {layer: (pya.Region(bank.cell.begin_shapes_rec_overlapping(bank.li(layer), bounds)) & pya.Region(bounds)).merged()
              for layer in ('active', 'poly', 'm2', 'm3', 'm4', 'm5')}
    lanes = []
    for band in range(bm['rows'] // 8):
        y = dy + band * 3160.
        left = frame_x + 300.
        box = pya.Region(pya.Box(u(left), u(y + 210.), u(right), u(y + 1365.)))
        first = right
        for layer in ('active', 'poly', 'm2', 'm3'):
            hits = layers[layer] & box
            if not hits.is_empty():
                first = min(first, hits.bbox().left * scale)
        width = floor((first - 500. - frame_x) / GRID) * GRID
        if width < 2000.:
            continue
        keep = []
        window = pya.Region(pya.Box(u(left), u(y), u(frame_x + width), u(y + 1365.)))
        for layer in ('m4', 'm5'):
            for pg in (layers[layer] & window).each():
                bb = pg.bbox()
                keep.append((layer, bb.left * scale - frame_x, bb.right * scale - frame_x))
        lanes.append((y, width, keep))
    return lanes


def _fits(tc: Component, width: float, band: tuple[float, float], compact_registers: bool = False,
          lanes: Sequence[tuple] = ()) -> bool:
    try:
        used, _ = centre_lanes_used(tc, width, ('Xdff_buf_data/',), band, lanes, compact_registers)
        return used <= len(lanes)
    except ValueError:              # gates overflow the band and available decoder lanes
        return False


def build(deck: Union[str, Path], output: Union[str, Path], bank_dir: Union[str, Path]) -> dict:
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
    if (set(bm.get('excluded_instances', [])) != {'XTIME_CONTROL', 'XRWL', 'XD_LATCH'}
            or not split or split.get('axis') != 'columns' or 'controller_channel_nm' not in bm):
        raise ValueError('Controller integration requires a compact column-split bank with a controller channel')
    ly = new_layout(); b = Builder(ly, f'OPENYIELD_COMPACT_{rows}X{cols}'); P = Planner(b)
    bank = _imported(ly, bank_dir, bm); b.place(bank)
    bp = bm['external_pins']; cx, cy = bm['array_origin_nm']
    cy -= 460.                       # upper core origin (array origin minus its bottom strap)
    frame_x = bm['decoder_bbox_nm'][0] + FRAME_X
    endpoints = {}; links = []
    def endpoint(net, point, alias=None):
        endpoints.setdefault(net, []).append((point, alias or net))

    # 1. Data registers folded into their column slots below the write band.
    tc, registers, rs = _register_slot(ly, deck)
    by_q = {m['Q']: (p, c, m) for p, c, m in registers}
    if len(registers) != cols or any(c.identity != registers[0][1].identity for _, c, _ in registers):
        raise ValueError('Every data column needs one identical source register')
    reg_y, clock_y = _register_origin(rs)
    terminals, physical, reg_pins = {}, [], {}
    clock_xs = []
    for pair in range(groups):
        x = pair * SLOT_W if pair < groups // 2 else cx + (pair - groups // 2) * SLOT_W
        b.place(rs.builder, x, reg_y)
        roles = {}
        for role, col in (('L', 2 * pair), ('U', 2 * pair + 1)):
            q = f'DIN_dff{col}'
            if bp[q][0] != x + rs.pins[role + '.Q'][0][0]:
                raise ValueError(f'{q}: register Q track is not under its buffer input')
            path, c, m = by_q[q]; roles[role] = 'XTIME_CONTROL/' + path
            dx, dy, layer = rs.pins[role + '.D'][0]
            reg_pins[m['D']] = (x + dx, reg_y + dy, layer)
            # Both M3 accesses have 70 nm end landings. Taller controller
            # bands can separate them beyond the register's fixed Q overhang.
            qx, qy, qlayer = rs.pins[role + '.Q'][0]
            if reg_y + qy + 70. < bp[q][1] - 70.:
                if qlayer != 'm3' or bp[q][2] != 'm3':
                    raise ValueError(f'{q}: register bridge requires aligned M3 accesses')
                P.wire(q, 'm3', [(x + qx, reg_y + qy), tuple(bp[q][:2])])
            endpoint(q, (bp[q][0], reg_y + rs.height, 'm3')); endpoint(q, tuple(bp[q]))
        for key, (px, py, layer) in rs.terminals.items():
            role, rest = key.split('.', 1); terminals[roles[role] + '/' + rest] = (x + px, reg_y + py, layer)
        for f in rs.fingers:
            role, rest = f['device'].split('.', 1); gx, gy = f['gate_xy_nm']
            physical.append(dict(f, device=roles[role] + '/' + rest, gate_xy_nm=(x + gx, reg_y + gy)))
        clock_xs.append(x + rs.pins['CLK'][0][0])
    trunks = dict(bm['supply_trunks_nm'])
    def register_bar(net: str, yy: float) -> None:
        tx = trunks[net]
        left, right = bm['controller_channel_nm']
        if bm.get('left_supply_trunks_nm'):
            lx = bm['left_supply_trunks_nm'][net]
            P.wire(net, 'm4', [(0., yy), (lx, yy)], 140.)
            P.stack(net, lx, yy, 4, 6)
            top = min(y for n, y in bm['write_supplies'] if n == net)
            P.wire(net, 'm6', [(lx, yy), (lx, top)])
            P.wire(net, 'm6', [(right - 1000., yy), (right, yy)])
            P.stack(net, right, yy, 4, 6)
        else:
            P.wire(net, 'm4', [(0., yy), (left, yy)], 140.)
            P.wire(net, 'm6', [(left, yy), (right, yy)])
            for sx in (left, right): P.stack(net, sx, yy, 4, 6)
        P.wire(net, 'm4', [(right, yy), (tx, yy)], 140.)
        P.stack(net, tx, yy, 4, 6)
    for net, y in rs.supplies:
        tx = trunks[net]; yy = reg_y + y
        register_bar(net, yy)
    for net, tx in trunks.items():
        x0, y0, x1, y1 = bank_conductor(bank.cell, 'm6', tx, bp[net][1])
        low = min(reg_y + y for n, y in rs.supplies if n == net)
        if low < y0 + 72.5: P.wire(net, 'm6', [(tx, y0 + 72.5), (tx, low)])
        endpoint(net, (tx, low, 'm6'))
    channel_sites = []
    left, right = bm['controller_channel_nm']
    middle = (left + right) / 2
    supply_site = right - 500. if bm.get('left_supply_trunks_nm') else middle
    channel_sites = [(net, middle, y) for net, ys in bm['control_bars_nm'].items() for y in ys]
    channel_sites += [(net, supply_site, y) for net, y in bm['write_supplies']]
    channel_sites += [(net, supply_site, reg_y + y) for net, y in rs.supplies]
    channel_sites += [(net, supply_site, y) for net, ys in bm['supply_bars_nm'].items() for y in ys]
    P.wire('clk_buf', 'm6', [(left, clock_y), (min(bm['control_trunks_nm'].values()) - 300., clock_y)])
    channel_sites.append(('clk_buf', middle, clock_y))
    channel_m6 = pya.Region(b.cell.begin_shapes_rec(ly.layer(21, 0))).merged()
    channel_sites = [(n, x, y) for n, x, y in channel_sites
                     if any(pg.inside(pya.Point(u(x), u(y))) for pg in channel_m6.each())]

    # 2. Interface heights from the bank conductors. Each control net has its
    #    own M6 trunk; return pins are spread at RETURN_SPACE on the block edge.
    ctl = bm['control_trunks_nm']
    def trunk(net):
        return bank_conductor(bank.cell, 'm6', ctl[net], bp[net][1])
    drives = {}
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        _, y0, _, y1 = trunk(net)
        middle = (y0 + y1) / 2
        drives[net] = snap(min(bm['control_bars_nm'][net], key=lambda y: abs(y - middle)))
    drives['clk_buf'] = clock_y
    supply_y = {}
    for net, y in bm['write_supplies']:
        supply_y.setdefault(net, y)
    supply_y['VDD'] = max(y for net, y in bm['write_supplies'] if net == 'VDD')
    far = {'pre_far': 'PRE', 'wen_far': 'w_en', 'sen_far': 's_en', 'iso_far': 'sa_iso', 'rbl': 'RBL_MUX'}
    band_top = int((cy - TOP_GAP) // GRID) * GRID
    # The mux output returns straight at its track top; SEL0/SEL1 stack to
    # M10 on their trunks. The bar returns spread at RETURN_SPACE around them.
    mx, my, _ = bp['RBL_MUX']
    sel_y = {}
    for net in INPUTS_EXTRA:
        _, sy0, _, sy1 = trunk(net); sel_y[net] = snap((sy0 + sy1) / 2)
    # M10 pads need more separation than their M4 select bars. Extend a
    # select's own M6 trunk within the periphery when the switch row places
    # those bars less than one M10 pad/space pitch apart.
    select_pitch = ceil((2 * M10_HALF + M10_SPACE) / GRID) * GRID
    previous = None
    for net in sorted(sel_y, key=sel_y.get):
        if previous is not None:
            sel_y[net] = max(sel_y[net], previous + select_pitch)
        previous = sel_y[net]
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

    # 3. The controller block holds every gate outside the data-register slots.
    width = snap(min(ctl.values()) - TRUNK_GAP - frame_x)
    ends_by_net = ({n: {'upper': p} for n, p in bp.items() if n in tc.ports and p[2] == 'm5'})
    decoder_nets = {n: [('upper', *ends['upper'][:2])]
                    for n, ends in ends_by_net.items() if n in tc.ports}
    cell_drop = COLUMN_CONTROL_DROP
    band_lo = reg_y - cell_drop
    # The controller placer already keeps its last cell 1 um below the
    # band ceiling. Reserving that margin twice drops a whole gate row
    # after peripheral compaction and needlessly widens large macros.
    band_hi = band_top - cell_drop
    lanes_abs = _column_lanes(bank, bm)
    used, _ = centre_lanes_used(tc, width, ('Xdff_buf_data/',), (band_lo, band_hi), lanes_abs, True)
    if used > len(lanes_abs):
        raise ValueError('Controller gates exceed the block below the shared decoder')
    y0 = reg_y - cell_drop
    highest = max([band_top] + [y + 1365. for y, _, _ in lanes_abs[:used]])
    highest = max(highest, min(band_top + 10000., bm['upper_decoder_top_nm'] - 1000.))
    height = snap(highest - y0 + 50)
    local = lambda y: y - y0
    lanes = [(local(y), w, keep) for y, w, keep in lanes_abs]
    profile = [(0., height, width)]
    if used:
        # Above the peripheral band, only the spare decoder strip provides
        # access to lane pins. Avoid searching the occupied decoder body.
        neck = local(band_top) + 1000.
        lane_width = min(width, max(w for _, w, _ in lanes_abs[:used]) + 1000.)
        profile = [(0., neck, width), (neck, height, lane_width)]
    obstacles = []
    for layer in ROUTER_LAYERS:
        li = ly.layer(*{f'm{i}': (9 + 2 * i, 0) for i in range(4, 10)}[layer])
        box = pya.Box(u(frame_x - 1500), u(y0 - 1500), u(frame_x + width + 1500), u(y0 + height + 1500))
        region = pya.Region(b.cell.begin_shapes_rec_overlapping(li, box)) & pya.Region(box)
        for pg in region.merged().each():
            owners = {n for n, x, y in channel_sites if layer == 'm6' and pg.inside(pya.Point(u(x), u(y)))}
            if len(owners) > 1: raise ValueError(f'Controller channel shorts bank nets: {owners}, {pg.bbox()}')
            for trap in pg.decompose_trapezoids():
                bb = trap.bbox(); s = ly.dbu * 1000
                obstacles.append((layer, bb.left * s - frame_x, bb.bottom * s - y0,
                                  bb.right * s - frame_x, bb.top * s - y0, next(iter(owners), None)))
    edge_y = {'upper': next(y for y in (local(band_top) // GRID * GRID - 500. - GRID * k for k in range(30))
                           if all(abs(y + y0 - yy) >= 350. for _, _, yy in channel_sites))}
    grid_x = lambda x: min(snap(x - frame_x), width - 200.) + frame_x
    # Off-grid decoder trunks jog to the router grid. Neighbours 300 nm apart
    # jog the same way, so their stubs take heights 300 nm apart, the one on
    # the side the jog points away from nearest the block.
    jog_y = {}
    for half, base, step in (('upper', band_top + 500., 300.),):
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
    for half in ('upper',):
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
    for net, points in decoder_nets.items():
        for half, x, y in points:
            path = riser(half, x, y)
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                obstacles.append(('m5', min(ax, bx) - 72.5 - frame_x, min(ay, by) - 72.5 - y0,
                                  max(ax, bx) + 72.5 - frame_x, max(ay, by) + 72.5 - y0, net))
    def blocked(layer, x0, y0_, x1, y1_, half):
        obstacles.append((layer, min(x0, x1) - half - frame_x, min(y0_, y1_) - half - y0,
                          max(x0, x1) + half - frame_x, max(y0_, y1_) + half - y0, None))
    # Select inputs climb from M6 to M10 after controller routing. Reserve
    # every landing first: their M9 pads can crowd routes near the frame
    # edge even when the M10 pads and M7 feedback lines are well separated.
    for net, sy in sel_y.items():
        tx = ctl[net]
        for k in range(6, 10):
            blocked(f'm{k}', tx, sy, tx, sy, macro_control.LANDING[k] / 2)
        _, sy0, _, sy1 = trunk(net)
        anchor = max(sy0 + 72.5, min(sy, sy1 - 72.5))
        if anchor != sy:
            blocked('m6', tx, anchor, tx, sy, 72.5)
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
                        min_height=height, band_span=(local(band_lo), local(band_hi)), lanes=lanes,
                        profile=profile, obstacles=obstacles, escapes=tuple(inputs), compact_registers=True)
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
        endpoint(net, (x, y, 'm6'))
        endpoint(net, (ctl[net], (by0 + by1) / 2, 'm6'))
        links.append(dict(controller_port=net, route='existing M6 bank crossing', y_nm=y))
    x, y, _ = cp['clk_buf']
    if abs(y - clock_y) > GRID / 2: raise ValueError('Register clock pin is off its bar')
    left = bm['controller_channel_nm'][0]
    right = min(ctl.values()) - 300.
    for tx, end in ((left, min(clock_xs)), (right, max(clock_xs))):
        P.stack('clk_buf', tx, clock_y, 4, 6)
        P.wire('clk_buf', 'm4', [(tx, clock_y), (end, clock_y)], 140.)
    endpoint('clk_buf', (x, y, 'm6')); endpoint('clk_buf', (clock_xs[0], clock_y, 'm4'))
    links.append(dict(controller_port='clk_buf', route='M6 + M4 register clock bar', y_nm=y))
    for net, y in supply_y.items():
        x, py, _ = cp[net]
        endpoint(net, (x, py, 'm6')); endpoint(net, (cx, y, 'm4'))
        if net == 'VDD':
            # The native controller VDD rail ends on M4 at this access.
            # Join the bank landing instead of leaving two close pads.
            native = pya.Region(ctrl.builder.cell.begin_shapes_rec(ctrl.builder.li('v3')))
            probe = pya.Region(pya.Box(u(x - frame_x) - 1, u(py - y0) - 1,
                                       u(x - frame_x) + 1, u(py - y0) + 1))
            if not (native & probe).is_empty():
                P.rect('m4', x - 72.5, py - 72.5, bm['controller_channel_nm'][1] + 70., py + 72.5)
    # 5. Decoder nets rise on M5 from a block edge into one decoder half.
    for net, points in decoder_nets.items():
        for (half, tx, ty), (px, py, _) in zip(points, planned[net]):
            px += frame_x; py += y0
            P.wire(net, 'm5', riser(half, tx, ty))
            endpoint(net, (px, py, 'm5')); endpoint(net, (tx, ty, 'm5'))
            links.append(dict(controller_port=net, half=half, route='M5 riser into the decoder', x_nm=px))
    # 6. Far-end returns: each bar's right end continues to a stack beside the
    #    VDD trunk, then M7 back; the mux output returns from the replica slot.
    tap = trunks['VDD'] - 300.
    replica_left = cx + (groups // 2) * SLOT_W
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
        # Each return owns its M5 climb. A shorter sense row can reverse
        # the order of bar and return heights; sharing one x would short
        # overlapping climbs even though the M7 return pins remain spaced.
        metal = pya.Region(b.cell.begin_shapes_rec(b.li('m5'))).merged()
        def clear_return(x: float) -> bool:
            box = pya.Box(u(x - 212.5), u(min(ry, ybar) - 212.5),
                          u(x + 212.5), u(max(ry, ybar) + 212.5))
            return (metal & pya.Region(box)).is_empty()
        tx = next((tap - GRID * k for k in range(int((tap - replica_left) // GRID) + 1)
                   if clear_return(tap - GRID * k)), None)
        if tx is None:
            raise ValueError(f'{port}: no clear M5 return column in the replica slot')
        # Bars may sit closer than an M7 pad pitch (400 + 400 nm), so each
        # return climbs on M5 to its own pin height before lifting to M7.
        P.wire(net, 'm4', [(xend - 70, ybar), (tx, ybar)], 140.)
        if ry != ybar:
            P.stack(net, tx, ybar, 4, 5); P.wire(net, 'm5', [(tx, ybar), (tx, ry)]); P.stack(net, tx, ry, 5, 7)
        else:
            P.stack(net, tx, ybar, 4, 7)
        P.wire('RETURN_' + port, 'm7', [(x, ry), (tx, ry)])
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
    dec_right = bm['decoder_bbox_nm'][2]
    rpx = snap(dec_right - rwl_v.width_nm - 300.); rpy = dec_top
    latch_c = read_component(deck, 'XD_LATCH'); latch_v = make_aux(ly, latch_c, folded=True)
    lpx = cx + (groups // 2) * SLOT_W + LATCH_GAP
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
    for net, offset in (('VSS', 200.), ('VDD', 50.)):
        rail_x = bm['decoder_bbox_nm'][0] + offset
        px, py, _ = rp_[net]
        P.wire(net, 'm1', [(px, py), (rail_x, py)]); P.stack(net, rail_x, py, 1, 2)
        py_probe = (bm['decoder_bbox_nm'][1]) + 1300.
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
        _, sy0, _, sy1 = trunk(net)
        anchor = max(sy0 + 72.5, min(sy, sy1 - 72.5))
        if anchor != sy:
            P.wire(net, 'm6', [(tx, anchor), (tx, sy)])
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
    ends = {'left': lambda x, y: (x_left + M10_HALF, y), 'right': lambda x, y: (x_right - M10_HALF, y),
            'bottom': lambda x, y: (x, y_bottom + M10_HALF),
            'top': lambda x, y: (x, y_top - M10_HALF)}
    pad_ys = [y for _, y in pads.values()]; middle = (min(pad_ys) + max(pad_ys)) / 2
    vertical = lambda y: ('bottom', 'top') if y <= middle else ('top', 'bottom')
    escape = {}
    for net in sorted(pads, key=lambda n: pads[n]):
        x, y = pads[net]
        # Select trunks sit outside the controller frame. Escape them over
        # their own array instead of fencing them behind controller inputs.
        sides = ('right', 'left') + vertical(y) if net in INPUTS_EXTRA else ('left',) + vertical(y)
        for side in sides:
            path = [(x, y), ends[side](x, y)]
            if free(net, path): escape[net] = (side, path); taken = (taken + m10(path)).merged(); break
    escape_columns = [frame_x + GRID * i for i in range(ceil(M10_HALF / GRID), floor((width - M10_HALF) / GRID) + 1)]
    for net in sorted(set(pads) - set(escape), key=lambda n: pads[n]):
        x, y = pads[net]
        for side in vertical(y):
            path = next(([(x, y), (c, y), ends[side](c, y)] for c in sorted(escape_columns, key=lambda c: abs(c - x))
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
    low = reg_y + max(y for net, y in rs.supplies if net == 'VSS')
    high = min(y for net, y in bm['write_supplies'] if net == 'VSS')
    if 0. < high - low < 400.:
        left, right = bm['controller_channel_nm']
        if bm.get('left_supply_trunks_nm'):left = right - 1000.
        P.rect('m6', left - 72.5, low - 72.5, right + 72.5, high + 72.5)
    # Controller routes are on a 100 nm grid; bank crossings are on the
    # device grid. Close sub-rule notches within their joined conductors.
    # Check foreign polygons before adding any metal.
    metal = pya.Region(b.cell.begin_shapes_rec(b.li('m6'))).merged()
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        px, py, _ = cp[net]
        probe = pya.Region(pya.Box(u(px) - 1, u(py) - 1, u(px) + 1, u(py) + 1))
        own = metal.interacting(probe)
        fill = own.sized(u(70.)).sized(-u(70.)) - own
        if not (fill.sized(u(140.)) & (metal - own)).is_empty():
            raise ValueError(f'{net}: crossing notch fill would crowd foreign M6')
        for polygon in fill.each():
            b.cell.shapes(b.li('m6')).insert(polygon)
            P.drawn['m6'].insert(polygon)
        metal = (metal + fill).merged()
    P.check(b.cell)
    if b.cell.dbbox() != extent: raise ValueError('Macro pins extend the macro boundary')
    from .conductor_ownership import port_conductors
    check_ports, check_aliases = {}, {}
    for net, points in endpoints.items():
        for i, (point, alias) in enumerate(points):
            key = f'{net}:{i}'; check_ports[key] = point; check_aliases[key] = alias.upper()
    ix0 = cx + 2 * SLOT_W
    interior = ([(2 * SLOT_W, cy + 460 + 3160, (cols // 2 - 2) * 1125, cy + 460 + (rows - 8) * 395),
                 (ix0, cy + 460 + 3160, cx + (cols // 2 - 2) * 1125, cy + 460 + (rows - 8) * 395)]
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
    block = [frame_x, y0, frame_x + width, band_top]
    metadata = dict(artifact_kind='bank', rows=rows, cols=cols, controller_layout=True, floorplan='column_split',
        core_pitch_nm=dict(row=395, column=1125), row_span=rows, column_span=cols, column_mux=2,
        included_instances=list(bm['included_instances']) + ['XTIME_CONTROL', 'XRWL', 'XD_LATCH'], excluded_instances=[],
        routing=('control logic: grid router below the shared decoder; interface: planned straight tracks'),
        split=split, controller_frame_nm=[frame_x, y0, frame_x + width, y0 + ctrl.height_nm],
        controller_block_nm=block, controller_block_area_um2=width * (band_top - y0) / 1e6,
        controller_cell_drop_nm=cell_drop,
        controller_decoder_lanes_nm=[[frame_x, y, frame_x + w, y + 1365.] for y, w, _ in lanes_abs[:used]],
        controller_placements=[dict(p, x_nm=p['x_nm'] + frame_x, y_nm=p['y_nm'] + y0) for p in ctrl.placements],
        register_band_nm=[cx, reg_y, cx + groups * SLOT_W, reg_y + rs.height],
        controller_standard_cells=ctrl.standard_cell_inventory, passive_locations=ctrl.passive_locations,
        controller_links=links, auxiliary_positions_nm=positions,
        array_origin_nm=bm['array_origin_nm'], decoder_bbox_nm=bm['decoder_bbox_nm'], core_geometry=bm['core_geometry'],
        bank_bbox_nm=[v_ * 1000 for v_ in (bank_box.left, bank_box.bottom, bank_box.right, bank_box.top)],
        source_devices=len(source.mos), source_passives=source.passives,
        source_artifacts={str(bank_dir): hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest()},
        controller_source_artifact=getattr(ctrl, 'source_artifact', None),
        guard_policy=('far-end feedback and fixed digital delays; no intentional RC guards'
                      if not source.passives else 'archived intentional RC; values require independent extraction'),
        interface_conductor_audit_excluded_bbox_nm=interior,
        macro_pins=pin_records,
        macro_pin_sides={side: [r['port'] for r in pin_records if r['side'] == side]
                         for side in ('left', 'bottom', 'top', 'right', 'internal')},
        output_latch=dict(folded=True, origin_nm=[lpx, lpy], width_nm=latch_v.width_nm, height_nm=latch_v.height_nm),
        waveform_validation='not run', defaults_promotable=False, external_pins=pins,
        compiler_provenance=bm.get('compiler_provenance'))
    metadata.update(register_band_nm=[0., reg_y, cx + cols // 2 * 1125., reg_y + rs.height],
        left_supply_layers=bm.get('left_supply_layers', dict(VDD='m5', VSS='m5')),
        left_supply_trunks_nm=bm.get('left_supply_trunks_nm'),
        bank_incremental_routing=bm.get('incremental_routing_provenance'),
        register_bands_nm=[
        [0., reg_y, cols // 2 * 1125., reg_y + rs.height],
        [cx, reg_y, cx + cols // 2 * 1125., reg_y + rs.height]],
        metal_policy=dict(bank_highest_routing_layer='m6', highest_routing_layer='m10',
                          upper_metals_used=True, purpose='TIME_CONTROL routing, feedback and macro input escapes'))
    return write_view(v, output, deck, 'complete compact column-split bank with TIME_CONTROL, RWL driver and output latch', metadata)


def generate(deck: Union[str, Path], output: Union[str, Path]) -> dict:
    source = flatten(deck)
    if source.rows < 16:
        raise ValueError('The integrated controller needs at least 16 rows')
    output = Path(output); parts = output.with_name(output.name + '_parts')
    bank_dir = parts / 'bank'
    build_bank(deck, bank_dir, controller_channel=True)
    # Use spare decoder rows before widening the central channel.
    bm = json.loads((bank_dir / 'layout.json').read_text())
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)
    _, _, rs = _register_slot(new_layout(), deck)
    width = snap(min(bm['control_trunks_nm'].values()) - TRUNK_GAP - bm['decoder_bbox_nm'][0] - FRAME_X)
    band = (_register_origin(rs)[0], int((bm['array_origin_nm'][1] - 460. - TOP_GAP) // GRID) * GRID)
    lane_layout = new_layout()
    lanes = _column_lanes(_imported(lane_layout, bank_dir, bm), bm)
    extra = 0.
    if not _fits(tc, width, band, True, lanes):
        lo, hi = width, 2 * width
        while not _fits(tc, hi, band, True, lanes):
            if hi > 1e6: raise ValueError('Controller gates do not fit a 1 mm block')
            lo, hi = hi, 2 * hi
        while hi - lo > GRID:
            mid = snap((lo + hi) / 2)
            lo, hi = (lo, mid) if _fits(tc, mid, band, True, lanes) else (mid, hi)
        extra = hi - width + GRID
    # Keep five routing-grid tracks beyond the placement minimum for escapes.
    extra = max(extra + 5 * GRID, MIN_COLUMN_FRAME - width)
    if extra:
        shutil.rmtree(bank_dir)
        build_bank(deck, bank_dir, core_x=bm['array_origin_nm'][0] + extra, controller_channel=True)
    return build(deck, output, bank_dir)
