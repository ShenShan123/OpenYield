"""TIME_CONTROL between the two halves of a split bank.

The column periphery sits between the lower and upper arrays
(``dense_bank.build(split=...)``); bitlines cross it on M5. TIME_CONTROL
shares the row-decoder columns, with its left edge on the decoder's:

* A full-width block between the decoder halves, beside the column
  periphery, holds the DFFs, transmission gates and antenna-tapped receivers,
  then as many other gates as fit. Its bottom edge equals the data-register
  band's, its right edge faces the bank's control trunks.
* Remaining gates sit in spare decoder rows above and below that block: the
  predecode columns left of the final decoder level, on the decoder's own
  1365 nm rows and M1 supply rails. Rows holding a predecode slice narrow or
  end the lane. Routing there treats every bank conductor as an obstacle.
* Every data register folds into its column's slot below the write band,
  directly above the lower array; its D input is an internal band pin.
* Interface routes are explicit straight tracks. Every decoder net
  that reaches the band (address buses, wl_en and level-0 enables of the
  other half) rises on M5 from a tapped pin on the facing block edge.

Source topology, device sizing and the core pitch remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json

from .components import Component, Instance, CellView, read_component
from .full_source import flatten
from .dense_views import write_view, expand
from .dense_aux import make_aux
from .dense_slots import make_register_slot
from .lib import Builder, new_layout, pya, u
from .time_control import make_control, units, centre_lanes_used
from . import macro_control
from .macro_control import Planner, bank_conductor, snap, GRID

TRUNK_GAP = 600.     # frame right edge to the bank's control trunks (cx - 150)
TOP_GAP = 1000.      # band top below the core; decoder supply bars run above
REG_GAP = 300.       # register band top below the write band
FRAME_X = 35.        # decoder address spines sit at x = 35 (mod 100): frame grid phase
RETURN_SPACE = 800.  # M7 returns: 400 nm wires, 400 nm spacing
DRIVE_SPACE = 300.   # M6 edge pins
BOTTOM_GAP = 1000.   # lower array top edge to the register band; decoder bars run here
LANE_GAP = 500.      # lane gates to the next decoder slice: same rows, rails and well order
MIN_LANE = 2000.     # narrowest spare decoder row that receives controller gates
LANE_FILL = 0.75     # placed lane rows / available rows that still routes reliably
LANE_MARGIN = 2      # extra routing rows beyond the last placed lane row
ROUTER_LAYERS = ('m4', 'm5', 'm6', 'm7', 'm8', 'm9')
BRIDGE_PITCH = 1700.  # M10 bridges over the array (0.8 um width and spacing, 805 nm M9 landings)


def _imported(ly, directory, m):
    src = new_layout(); src.read(str(directory / 'full_macro.gds'))
    cell = Builder(ly, m['primary']); cell.cell.copy_tree(src.cell(m['primary']))
    return cell


def _register_slot(layout, deck):
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)
    registers = [(p, c, m) for p, c, m in units(tc) if p.startswith('Xdff_buf_data/')]
    return tc, registers, make_register_slot(layout, 'CENTRE_REGISTER_SLOT', registers[0][1], compact=True)


def _register_origin(rs) -> tuple[float, float]:
    """Register-band bottom and its grid-aligned clock bar below the write band."""
    c_rel = rs.pins['CLK'][0][1]
    clock_y = ((-REG_GAP - rs.height + c_rel) // GRID) * GRID
    return clock_y - c_rel, clock_y


def register_reserve(deck) -> float:
    """Write-band bottom to the lower array's top edge."""
    _, _, rs = _register_slot(new_layout(), deck)
    return BOTTOM_GAP - _register_origin(rs)[0]


def _lanes(split: dict, rows: int, cy: float, extra: float = 0.):
    """Spare decoder rows next to the band, outward, their gate widths and
    the decoder M4-M6 conductors (frame-local layer, x0, x1) crossing them.

    A lane ends LANE_GAP before the row's first decoder slice; trunks of other
    slices cross it, and gate pin accesses keep clear of them. ('well', x0, x1)
    entries are the decoder well intervals along the row's edges with its
    neighbours in the same half; lane gate wells must not end beside them.
    extra previews a wider spare column before the bank is rebuilt with it.
    """
    cut = split['row']; yb = split['lower_array_top_nm']
    left = {int(r): x for r, x in split['decoder_row_left_nm'].items()}
    metal = {int(r): v for r, v in split['decoder_row_metal_nm'].items()}
    wells = {int(r): v for r, v in split['decoder_row_well_nm'].items()}
    def lane_width(row):
        return (left[row] + extra - LANE_GAP - FRAME_X) // GRID * GRID
    def keepouts(row):
        # The row above meets this one at its bottom edge, the row below at its top.
        near = [(n, side) for n, side in ((row - 1, 'top'), (row + 1, 'bottom'))
                if n in wells and (n >= cut) == (row >= cut)]
        return (tuple((layer, x0 + extra - FRAME_X, x1 + extra - FRAME_X) for layer, x0, x1 in metal.get(row, ()))
                + tuple(('well', x0 + extra - FRAME_X, x1 + extra - FRAME_X)
                        for n, side in near for x0, x1 in wells[n][side]))
    def run(seq):
        out = []
        for row in seq:
            if lane_width(row) < MIN_LANE: break
            out.append(row)
        return out
    def row_y(row):
        return cy + (row - cut) * 1365. if row >= cut else yb - (cut - row) * 1365.
    return run(range(cut, rows)), run(range(cut - 1, -1, -1)), lane_width, row_y, keepouts


def _fuse(layout: pya.Layout, top: pya.Cell, bank: pya.Cell, name: str) -> str:
    """One physical module for both decoder halves and the controller."""
    instances = []
    for parent, prefix, count in ((bank, 'DENSE_DECODER_', 2), (top, 'TIME_CONTROL_CORNER_', 1)):
        matches = [i for i in parent.each_inst() if layout.cell(i.cell_index).name.startswith(prefix)]
        if len(matches) != count: raise ValueError(f'Expected {count} physical {prefix} blocks')
        instances.extend(matches)
    shared = Builder(layout, name)
    for instance in instances:
        shared.cell.insert(instance.cell_inst)
        instance.delete()
    top.insert(pya.CellInstArray(shared.cell.cell_index(), pya.Trans()))
    return shared.name


def build(deck, output, bank_dir):
    source = flatten(deck); rows, cols = source.rows, source.cols
    macro_control._conductors.clear()
    latch = source.instances['XD_LATCH']['pins']; sense = source.instances[f'XSENSEAMP_{cols-1}']['pins']
    if latch['D'] != sense['Q'] or latch['EN'] != sense['EN']:
        raise ValueError('The centre plan requires the last-column sense output and enable')
    bank_dir = Path(bank_dir)
    bm = json.loads((bank_dir / 'layout.json').read_text())
    digest = hashlib.sha256(Path(deck).read_bytes()).hexdigest()
    if bm['source_deck_sha256'] != digest:
        raise ValueError('Bank does not match the source deck')
    if hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest() != bm['layout_sha256']:
        raise ValueError('Bank GDS differs from its layout manifest')
    split = bm.get('split')
    if (set(bm.get('excluded_instances', [])) != {'XTIME_CONTROL', 'XRWL', 'XD_LATCH'} or not split
            or split['decoder_bridges']):
        raise ValueError('Centre integration requires a split bank without decoder bridges')
    ly = new_layout(); b = Builder(ly, f'OPENYIELD_CENTRE_{rows}X{cols}'); P = Planner(b)
    bank = _imported(ly, bank_dir, bm); b.place(bank)
    bp = bm['external_pins']; cx, cy = bm['array_origin_nm']; bands = bm['periphery_bands_nm']
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
    for pair in range(cols // 2):
        x = cx + pair * 1410; b.place(rs.builder, x, reg_y)
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
    trunks = {n: bp[n][0] for n in ('VDD', 'VSS')}
    for net, y in rs.supplies:
        tx = trunks[net]; yy = reg_y + y
        P.wire(net, 'm4', [(cx, yy), (tx, yy)], 140.); P.stack(net, tx, yy, 4, 6)
    for net, tx in trunks.items():
        x0, y0, x1, y1 = bank_conductor(bank.cell, 'm6', tx, bp[net][1])
        low = min(reg_y + y for n, y in rs.supplies if n == net)
        if low < y0 + 72.5: P.wire(net, 'm6', [(tx, y0 + 72.5), (tx, low)])
        endpoint(net, (tx, low, 'm6'))

    # 2. Interface heights, from the bank conductors themselves.
    trunk_x = cx - 150
    def trunk(net):
        return bank_conductor(bank.cell, 'm6', trunk_x, bp[net][1])
    drives, returns = {}, {}
    for net, far in (('PRE', 'pre_far'), ('w_en', 'wen_far'), ('s_en', 'sen_far'), ('sa_iso', 'iso_far')):
        _, y0, _, y1 = trunk(net)
        drives[net] = snap((y0 + y1) / 2)
        returns[far] = (net, y1 - 72.5 if y1 - y0 > 145 else (y0 + y1) / 2)
    pin_y = {far: (drives['PRE'] if far == 'pre_far' else snap(y)) for far, (net, y) in returns.items()}
    rbl_y = snap((bands['write'][1] + bands['sense'][0]) / 2)
    returns['rbl'] = ('RBL', rbl_y); pin_y['rbl'] = rbl_y
    drives['clk_buf'] = clock_y
    supply_y = {}
    for net, y in bands['write_supplies']:
        supply_y.setdefault(net, y)
    for name, ys, space in (('drive', list(drives.values()) + list(supply_y.values()), DRIVE_SPACE),
                            ('return', list(pin_y.values()), RETURN_SPACE)):
        ys = sorted(ys)
        if any(b2 - a < space for a, b2 in zip(ys, ys[1:])):
            raise ValueError(f'Centre {name} pins closer than {space} nm: {ys}')

    # 3. Shared decoder/control plan: a full-width block between the decoder
    #    halves and lanes in the spare predecode rows above and below it.
    width = snap(trunk_x - TRUNK_GAP - FRAME_X)
    frame_x = FRAME_X
    band_top = int((cy - TOP_GAP) // GRID) * GRID
    upper, lower, lane_width, row_y, keepouts = _lanes(split, rows, cy)
    # Decoder nets end on M5 at each half's band edge. A net in both halves
    # (shared address buses, wl_en, level-0 enables of the upper half) is
    # bridged on M10 over the array. The controller drives each of its nets
    # once, through a tapped pin on the block's lower edge (upper edge for an
    # upper-only net) and a planned M5 riser.
    ends_by_net = split['decoder_band_pins']
    bridged = sorted((n for n, ends in ends_by_net.items() if len(ends) == 2), key=lambda n: ends_by_net[n]['lower'][0])
    if any(ends_by_net[n]['lower'][0] != ends_by_net[n]['upper'][0] for n in bridged):
        raise ValueError('Decoder halves disagree on a bridged trunk phase')
    decoder_nets = {n: [('lower' if 'lower' in ends else 'upper', *ends['lower' if 'lower' in ends else 'upper'][:2])]
                    for n, ends in ends_by_net.items() if n in tc.ports}
    edge_halves = {half for points in decoder_nets.values() for half, *_ in points}
    band_lo = reg_y + (1000. if 'lower' in edge_halves else 0.)
    lanes_abs = [(row_y(r), lane_width(r), keepouts(r)) for r in upper + lower]
    used, _ = centre_lanes_used(tc, width, ('Xdff_buf_data/',), (band_lo, band_top), lanes_abs)
    used_upper = min(used, len(upper)); used_lower = max(0, used - len(upper))
    if used > len(upper) + len(lower):
        raise ValueError('Controller gates exceed the spare decoder lanes')
    # Placement keeps every listed row; routing also uses a small margin.
    up_rows = upper[:max(1, min(len(upper), used_upper + LANE_MARGIN))] if upper else []
    lo_rows = lower[:max(1, min(len(lower), used_lower + LANE_MARGIN))] if lower else []
    lowest = min([row_y(r) for r in lo_rows] + [reg_y])
    y0 = reg_y - ((reg_y - lowest) // GRID + (1 if (reg_y - lowest) % GRID else 0)) * GRID
    highest = max([row_y(r) + 1365. for r in up_rows] + [band_top])
    height = snap(highest - y0 + 50)
    local = lambda y: y - y0
    # Gates sit in the lanes, but routing spans the frame width: over the
    # decoder, every bank conductor is an obstacle.
    profile = [(0., height, width)]
    # The same rows as the fit above, so placement repeats it exactly (slice
    # widths follow the rows' clear stretches) and fills only the framed rows.
    lanes = [(local(row_y(r)), lane_width(r), keepouts(r)) for r in upper + lower]
    # Bank conductors inside the routing region are blockages, including
    # trunks of controller nets: those join only at reserved stub accesses, so
    # no same-net notch forms against bank metal.
    obstacles = []
    for layer in ROUTER_LAYERS:
        li = ly.layer(*{f'm{i}': (9 + 2 * i, 0) for i in range(4, 10)}[layer])
        region = pya.Region()
        for ry0, ry1, rw in profile:
            box = pya.Box(u(frame_x - 1500), u(y0 + ry0 - 1500), u(frame_x + rw + 1500), u(y0 + ry1 + 1500))
            region.insert(pya.Region(bank.cell.begin_shapes_rec_overlapping(li, box)) & pya.Region(box))
        for pg in region.merged().each():
            for trap in pg.decompose_trapezoids():
                bb = trap.bbox(); s = ly.dbu * 1000
                obstacles.append((layer, bb.left * s - frame_x, bb.bottom * s - y0,
                                  bb.right * s - frame_x, bb.top * s - y0, None))
    edge_y = {'upper': local(band_top) // GRID * GRID - 100., 'lower': local(reg_y) + 100.}
    # A pin stays 200 nm inside the block; a trunk beyond that (wl_en beside
    # the drivers) is reached by a jog in the gate-free strip.
    jog_y = {'upper': band_top + 500., 'lower': reg_y - 500.}
    def riser(half, x, y):
        px = min(snap(x - frame_x), width - 200.) + frame_x; py = edge_y[half] + y0
        return [(px, py), (px, y)] if px == x else [(px, py), (px, jog_y[half]), (x, jog_y[half]), (x, y)]
    # Risers cross the lane strips: they block routing there, and their edge
    # pins (each with a diffusion tap) keep 300 nm apart.
    for half in ('lower', 'upper'):
        xs = sorted(riser(half, x, y)[0][0] for points in decoder_nets.values() for h, x, y in points if h == half)
        if any(b2 - a < 300 for a, b2 in zip(xs, xs[1:])): raise ValueError(f'{half} decoder pins closer than 300 nm')
    for points in decoder_nets.values():
        for half, x, y in points:
            path = riser(half, x, y)
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                obstacles.append(('m5', min(ax, bx) - 72.5 - frame_x, min(ay, by) - 72.5 - y0,
                                  max(ax, bx) + 72.5 - frame_x, max(ay, by) + 72.5 - y0, None))
    # Bridges: staggered M5-M8 stacks on each half's trunk (clear of the
    # decoder's M6 feeds), M8 jogs to columns over the array, M10 across the
    # band. The upper trunk gets a diffusion tap in the strip below it.
    m6 = ly.layer(21, 0)
    def free_m6(x, y):
        box = pya.Box(u(x - 300), u(y - 300), u(x + 300), u(y + 300))
        return pya.Region(bank.cell.begin_shapes_rec_overlapping(m6, box)).is_empty()
    def stack_ys(half):
        ys, y, step = [], (cy + 400. if half == 'upper' else yb - 400.), (900. if half == 'upper' else -900.)
        for n in bridged:
            x = ends_by_net[n][half][0]
            while not free_m6(x, y): y += step / 9
            bank_conductor(bank.cell, 'm5', x, y)
            ys.append(y); y += step
        return ys
    bridges = [(n, ends_by_net[n]['upper'][0], cx + 1000. + BRIDGE_PITCH * i, yl, yu)
               for i, (n, yl, yu) in enumerate(zip(bridged, stack_ys('lower'), stack_ys('upper')))]
    if bridges and bridges[-1][2] + 1000. > cx + bm['core_geometry']['width_nm']:
        raise ValueError('Decoder bridges exceed the array width')
    tap_y = cy - 900.
    def blocked(layer, x0, y0_, x1, y1_, half):
        obstacles.append((layer, min(x0, x1) - half - frame_x, min(y0_, y1_) - half - y0,
                          max(x0, x1) + half - frame_x, max(y0_, y1_) + half - y0, None))
    for n, x, xb, yl, yu in bridges:
        for y in (yl, yu):
            for layer, half in (('m5', 72.5), ('m6', 72.5), ('m7', 200.), ('m8', 202.5)): blocked(layer, x, y, x, y, half)
            blocked('m8', x, y, xb, y, 202.5)
        blocked('m5', x, cy + 100., x, tap_y, 72.5); blocked('m4', x, tap_y, x, tap_y, 70.)
    rwl_x = snap(max([lane_width(r) for r in up_rows[:1]] + [0.]) + 1500.)
    if rwl_x > width - 1500: raise ValueError('No replica-wordline drop beside the controller lanes')
    top_rows = rows - cut
    rwl_tap = (cx + (cols + 1) * 705 + 500, cy + (top_rows - .5) * 1365)
    rwl_pin_y = edge_y['upper'] + y0
    # Planned wires over the decoder rows: the replica-wordline return's M7 run
    # and M9 drop, and the wl_en spine up to the replica driver.
    blocked('m9', rwl_x + frame_x, rwl_pin_y, rwl_x + frame_x, rwl_tap[1], 402.5)
    blocked('m8', rwl_x + frame_x, rwl_tap[1], rwl_x + frame_x, rwl_tap[1], 202.5)
    blocked('m7', rwl_x + frame_x, rwl_tap[1], rwl_tap[0], rwl_tap[1], 202.5)
    spine = (split['decoder_band_pins']['wl_en']['upper'][0], cy + (top_rows - 1) * 1365 + 80)
    blocked('m5', spine[0], spine[1], spine[0], cy + top_rows * 1365 + 100., 72.5)
    def plan(h: float) -> dict:
        p = {n: (width - 100., snap(local(y)), 'm6') for n, y in drives.items()}
        p.update({n: (width - 300., snap(local(pin_y[n])), 'm7') for n in returns})
        # The replica-wordline return drops on M8 beside the lanes, over the
        # decoder's final column, into the block's top edge.
        p['rwl'] = (rwl_x, edge_y['upper'], 'm8')
        p.update({n: (width - 100., snap(local(y)), 'm6') for n, y in supply_y.items()})
        for net, points in decoder_nets.items():
            p[net] = [(riser(half, x, y)[0][0] - frame_x, edge_y[half], 'm5') for half, x, y in points]
        return p
    ctrl = make_control(ly, tc, width, exclude=('Xdff_buf_data/',), pin_plan=plan, corner=True,
                        min_height=height, band_span=(local(band_lo), local(band_top)), lanes=lanes,
                        profile=profile, obstacles=obstacles)
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

    # 4. Drives: one straight M6 line into each control trunk.
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        x, y, _ = cp[net]
        _, by0, _, by1 = trunk(net)
        ty = y if by1 - by0 > 300 else (by0 + by1) / 2
        if not by0 + 72.5 <= ty <= by1 - 72.5: raise ValueError(f'{net} drive misses its trunk')
        P.wire(net, 'm6', [(x, y), (x + 300, y), (x + 300, ty), (trunk_x, ty)])
        endpoint(net, (x, y, 'm6')); endpoint(net, (trunk_x, ty, 'm6'))
        links.append(dict(controller_port=net, route='M6 straight into the bank trunk', y_nm=y))
    x, y, _ = cp['clk_buf']
    if abs(y - clock_y) > GRID / 2: raise ValueError('Register clock pin is off its bar')
    P.wire('clk_buf', 'm6', [(x, y), (cx - 300, y)]); P.stack('clk_buf', cx - 300, y, 4, 6)
    P.wire('clk_buf', 'm4', [(cx - 300, y), (max(clock_xs), y)], 140.)
    endpoint('clk_buf', (x, y, 'm6')); endpoint('clk_buf', (clock_xs[0], y, 'm4'))
    links.append(dict(controller_port='clk_buf', route='M6 + M4 register clock bar', y_nm=y))
    for net, y in supply_y.items():
        x, py, _ = cp[net]
        P.wire(net, 'm6', [(x, py), (cx - 450, py)]); P.stack(net, cx - 450, py, 4, 6)
        P.wire(net, 'm4', [(cx - 450, py), (cx - 450, y), (cx, y)], 140.)
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
        P.wire(net, 'm5', [(x, cy + 100.), (x, tap_y)]); antenna_diode(b, x, tap_y); P.stack(net, x, tap_y, 3, 5)
        endpoint('BRIDGE_' + net, (x, yl, 'm5'), net); endpoint('BRIDGE_' + net, (x, yu, 'm5'), net)
        links.append(dict(decoder_net=net, route='M5-M8 stacks, M8 jogs and an M10 bridge over the array', x_nm=x, bridge_x_nm=xb))
    # 6. Far-end returns: bar end past the supply trunks, then M7 straight back.
    tap = snap(max(trunks.values()) + 72.5 + 140 + 72.5 + GRID / 2)
    bars = {}
    for port, (net, y) in returns.items():
        x, ry, _ = cp[port]
        if net == 'RBL':
            rbx = bp['RBL'][0]; bank_conductor(bank.cell, 'm5', rbx, y)
            P.stack('RBL', rbx, ry, 5, 7); P.wire('RETURN_rbl', 'm7', [(x, ry), (rbx, ry)])
            endpoint('RETURN_rbl', (x, ry, 'm7'), 'RBL'); endpoint('RETURN_rbl', (rbx, ry, 'm5'), 'RBL')
            links.append(dict(controller_port=port, bank_net=net, route='M5 replica bitline + M7', y_nm=ry))
            continue
        _, b0, b1, t0 = bank_conductor(bank.cell, 'm4', cx + cols * 705 - 200, y)
        ybar = (b0 + t0) / 2; bars[port] = (b1, ybar)
        P.wire(net, 'm4', [(b1 - 70, ybar), (tap, ybar)], 140.); P.stack(net, tap, ybar, 4, 7)
        path = [(x, ry), (tap, ry)] if abs(ry - ybar) <= 30 else [(x, ry), (tap, ry), (tap, ybar)]
        P.wire('RETURN_' + port, 'm7', path)
        endpoint('RETURN_' + port, (x, ry, 'm7'), net); endpoint('RETURN_' + port, (b1 - 100, ybar, 'm4'), net)
        links.append(dict(controller_port=port, bank_net=net, route='M4 bar end + M7', y_nm=ry))
    # 7. Replica wordline: far replica row, M7 above the upper array, M9 down.
    bank_conductor(bank.cell, 'm4', *rwl_tap)
    x, ry, _ = cp['rwl']
    # The drop is M9 so the bridges' M8 jogs pass under it.
    P.stack('RWL', *rwl_tap, 4, 7); P.wire('RETURN_rwl', 'm7', [rwl_tap, (x, rwl_tap[1])])
    P.stack('RWL', x, rwl_tap[1], 7, 9); P.wire('RETURN_rwl', 'm9', [(x, rwl_tap[1]), (x, ry)]); P.stack('RWL', x, ry, 8, 9)
    endpoint('RETURN_rwl', (x, ry, 'm8'), 'RWL'); endpoint('RETURN_rwl', (*rwl_tap, 'm4'), 'RWL')
    links.append(dict(controller_port='rwl', bank_net='RWL', route='M7 above the array + M9 over the decoder'))

    # 8. Replica wordline driver above the decoder; output latch at the right.
    auxiliaries, positions, apins = {}, {}, {}
    ceiling = rows <= 32 and cols <= 32
    rwl_c = read_component(deck, 'XRWL'); rwl_v = make_aux(ly, rwl_c, ceiling)
    zx_local, zy_local, _ = rwl_v.pins['Z']
    dec_right = split['upper_decoder_bbox_nm'][2]
    rpx = dec_right - rwl_v.width_nm; rpy = cy + (top_rows + .5) * 1365 - zy_local
    latch_c = read_component(deck, 'XD_LATCH'); latch_v = make_aux(ly, latch_c)
    lpx = tap + 300; lpy = bp[f'SA_Q{cols-1}'][1] - 200
    for name, v, x, y in (('XRWL', rwl_v, rpx, rpy), ('XD_LATCH', latch_v, lpx, lpy)):
        auxiliaries[name] = v; positions[name] = (x, y); b.place(v.builder, x, y)
        apins[name] = {p: (px + x, py + y, l) for p, (px, py, l) in v.pins.items()}
    rp_ = apins['XRWL']; lp = apins['XD_LATCH']
    b.rect('pwell', rpx, cy + top_rows * 1365, rpx + rwl_v.width_nm, rpy)
    ax, ay, _ = rp_['A']; vx, vy, _ = rp_['VDD']
    P.wire('VDD', 'm3', [(ax, ay), (ax, rpy + 1100), (vx, rpy + 1100)]); P.stack('VDD', vx, vy, 1, 3)
    P.wire('VDD', 'm3', [(vx, rpy + 1100), (vx, vy)])
    zx, zy, _ = rp_['Z']
    if abs(zy - (cy + (top_rows + .5) * 1365)) > 1e-6: raise ValueError('Replica-driver output is not aligned to the core RWL')
    P.stack('RWL', zx, zy, 3, 6)
    bx, by, _ = rp_['B']
    bank_conductor(bank.cell, 'm5', *spine)
    P.wire('wl_en', 'm3', [(bx, by), (bx, rpy + 385)]); P.stack('wl_en', bx, rpy + 385, 3, 4)
    P.wire('wl_en', 'm4', [(bx, rpy + 385), (spine[0], rpy + 385)]); P.stack('wl_en', spine[0], rpy + 385, 4, 5)
    P.wire('wl_en', 'm5', [(spine[0], rpy + 385), spine])
    endpoint('wl_en', (bx, by, 'm3'))
    for net, rail_x, top_y in (('VSS', 200., cy + (top_rows - 1) * 1365 + 100),
                               ('VDD', 50., cy + (top_rows - 1) * 1365 + 1265)):
        px, py, _ = rp_[net]
        P.wire(net, 'm1', [(px, py), (rail_x, py)]); P.stack(net, rail_x, py, 1, 2); P.wire(net, 'm2', [(rail_x, top_y), (rail_x, py)])
        endpoint(net, (px, py, 'm1'))
    qx, qy, _ = bp[f'SA_Q{cols-1}']; dx, dy, _ = lp['D']
    if abs(dy - qy) > 1e-6: raise ValueError('Latch D track is not aligned to the selected sense output')
    P.stack('LATCH_DATA', qx, qy, 3, 4); P.wire('LATCH_DATA', 'm4', [(qx, qy), (dx, dy)], 140.)
    ex, ey, _ = lp['EN']; sy = bars['sen_far'][1]
    P.wire('LATCH_ENABLE', 'm5', [(tap, sy), (ex, sy), (ex, ey)]); P.stack('LATCH_ENABLE', ex, ey, 4, 5)
    endpoint('LATCH_DATA', (qx, qy, 'm3'), latch['D']); endpoint('LATCH_DATA', (dx, dy, 'm4'), latch['D'])
    endpoint('LATCH_ENABLE', (ex, ey, 'm4'), latch['EN'])
    for net, ty, tx in (('VSS', lpy - 270, trunks['VSS']), ('VDD', lpy + 630, trunks['VDD'])):
        px, py, _ = lp[net]; side = lpx - 200
        P.wire(net, 'm1', [(px, py), (side, py), (side, ty), (tx, ty)]); P.stack(net, tx, ty, 1, 6)
        endpoint(net, (px, py, 'm1'))
    P.check(b.cell)
    from .conductor_ownership import port_conductors
    check_ports, check_aliases = {}, {}
    for net, points in endpoints.items():
        for i, (point, alias) in enumerate(points):
            key = f'{net}:{i}'; check_ports[key] = point; check_aliases[key] = alias.upper()
    lo_x0, lo_y0 = split['lower_array_origin_nm']
    # Large cores leave their interior out of the audit, except the columns
    # the decoder bridges cross.
    ix0 = max([cx + 1410] + [xb + 1000 for _, _, xb, _, _ in bridges])
    interior = ([(ix0, cy + 2730, cx + (cols - 1) * 705, cy + (top_rows - 2) * 1365),
                 (ix0, lo_y0 + 2730, cx + (cols - 1) * 705, yb - 2730)]
                if rows * cols > 4096 else None)
    port_conductors(b, check_ports, check_aliases, require_connected=True, exclude_bbox_nm=interior)

    # 9. Macro ports and the complete source component.
    ports = tuple(source.ports); pins = {}
    for p in ports:
        if p in ('VDD', 'VSS'): pins[p] = (trunks[p], bp[p][1], 'm6')
        elif p in ('OUT', 'OUT_B'): pins[p] = lp['Q' if p == 'OUT' else 'QB']
        elif p in reg_pins: pins[p] = reg_pins[p]
        elif p in bp: pins[p] = bp[p]
        else: pins[p] = cp[p]
        x, y, layer = pins[p]; b.label(layer, p, x, y)
    formal = source.instances['XTIME_CONTROL']['pins']
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
        dx, dy = positions[name]
        terminals.update({name + '/' + n: (x + dx, y + dy, l) for n, (x, y, l) in v.device_pins.items()})
        for f in v.physical_fingers:
            f = dict(f, device=name + '/' + f['device'])
            if 'gate_xy_nm' in f: f['gate_xy_nm'] = (f['gate_xy_nm'][0] + dx, f['gate_xy_nm'][1] + dy)
            physical.append(f)
    box = b.cell.dbbox()
    v = CellView(component, b, box.width() * 1000, box.height() * 1000, pins, terminals, [], P.segments, [])
    v.physical_fingers = physical
    bank_box = bank.cell.dbbox()
    shared_cell = _fuse(ly, b.cell, bank.cell, f'ROW_DECODER_TIME_CONTROL_{rows}X{cols}')
    shared_box = ly.cell(shared_cell).dbbox()
    lane_rows = [p for p in ctrl.placements if p['height_nm'] == 1365]
    lane_extent = {}
    for p in lane_rows:
        y = p['y_nm'] + y0; lane_extent[y] = max(lane_extent.get(y, 0.), p['x_nm'] + p['width_nm'])
    block = [frame_x, reg_y, frame_x + width, band_top]
    metadata = dict(artifact_kind='bank', rows=rows, cols=cols, controller_layout=True, floorplan='centre',
        included_instances=list(bm['included_instances']) + ['XTIME_CONTROL', 'XRWL', 'XD_LATCH'], excluded_instances=[],
        routing='control logic: grid router inside its block and decoder lanes; interface: planned straight tracks',
        split=dict(split, decoder_row_left_nm=None, decoder_row_metal_nm=None, decoder_row_well_nm=None),
        controller_frame_nm=[frame_x, y0, frame_x + width, y0 + ctrl.height_nm],
        controller_block_nm=block, controller_block_area_um2=width * (band_top - reg_y) / 1e6,
        controller_lane_rows_nm=[[frame_x, y, frame_x + x1, y + 1365.] for y, x1 in sorted(lane_extent.items())],
        controller_lane_area_um2=sum(p['width_nm'] * p['height_nm'] for p in lane_rows) / 1e6,
        controller_left_edge_nm=frame_x, decoder_left_edge_nm=0.,
        controller_left_extension_nm=0., controller_bottom_alignment='data-register band',
        controller_bottom_offset_nm=0.,
        routing_profile_nm=[[frame_x, y0 + a, frame_x + w, y0 + c] for a, c, w in profile],
        decoder_control_bbox_nm=[v * 1000 for v in (shared_box.left, shared_box.bottom, shared_box.right, shared_box.top)],
        decoder_control_cell=shared_cell,
        controller_placements=[dict(p, x_nm=p['x_nm'] + frame_x, y_nm=p['y_nm'] + y0) for p in ctrl.placements],
        register_band_nm=[cx, reg_y, cx + cols // 2 * 1410, reg_y + rs.height],
        controller_standard_cells=ctrl.standard_cell_inventory, passive_locations=ctrl.passive_locations,
        controller_links=links, auxiliary_positions_nm=positions,
        array_origin_nm=bm['array_origin_nm'], decoder_bbox_nm=bm['decoder_bbox_nm'], core_geometry=bm['core_geometry'],
        bank_bbox_nm=[v * 1000 for v in (bank_box.left, bank_box.bottom, bank_box.right, bank_box.top)],
        replica_integrated_in_core=True, source_devices=len(source.mos), source_passives=source.passives,
        source_artifacts={str(bank_dir): hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest()},
        guard_policy=('far-end feedback and fixed digital delays; no intentional RC guards'
                      if not source.passives else 'archived intentional RC; values require independent extraction'),
        interface_conductor_audit_excluded_bbox_nm=interior,
        waveform_validation='not run', defaults_promotable=False, external_pins=pins,
        compiler_provenance=bm.get('compiler_provenance'))
    return write_view(v, output, deck, 'complete split bank with TIME_CONTROL between the decoder halves', metadata)


def _required_spare(deck, bm) -> float:
    """Smallest spare decoder width (100 nm steps) whose lanes hold every gate
    within LANE_FILL of the available rows and whose block keeps the unsliced
    decoder drivers (address buffers) beside their DFFs."""
    split = bm['split']; rows = bm['rows']; cx, cy = bm['array_origin_nm']
    tc, _, rs = _register_slot(new_layout(), deck)
    reg_y, _ = _register_origin(rs)
    band_top = int((cy - TOP_GAP) // GRID) * GRID
    def fits(extra):
        upper, lower, lane_width, row_y, keepouts = _lanes(split, rows, cy, extra)
        width = snap(cx + extra - 150 - TRUNK_GAP - FRAME_X)
        lanes = [(row_y(r), lane_width(r), keepouts(r)) for r in upper + lower]
        try:
            # The bottom margin for block-edge taps is assumed present.
            used, spilled = centre_lanes_used(tc, width, ('Xdff_buf_data/',), (reg_y + 1000., band_top), lanes)
        except ValueError as error:
            if 'exceed the spare decoder lanes' not in str(error): raise
            return False
        return used <= LANE_FILL * len(lanes) and not spilled
    if fits(0.): return 0.
    lo, hi = 0, 1
    while not fits(hi * 100.):
        lo, hi = hi, hi * 2
        if hi > 400: raise ValueError('Controller does not fit a 40 um spare decoder column')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if fits(mid * 100.) else (mid, hi)
    return hi * 100.


def generate(deck, output):
    from .dense_bank import build as build_bank, split_row
    source = flatten(deck)
    cut = split_row(source.rows)
    if cut is None: raise ValueError('The centre floorplan needs at least 16 rows')
    output = Path(output); parts = output.with_name(output.name + '_parts')
    reserve = register_reserve(deck)
    bank_dir = parts / 'bank'
    build_bank(deck, bank_dir, include_auxiliary=False, folded=True, split=cut, reserve=reserve, bridge=False)
    spare = _required_spare(deck, json.loads((bank_dir / 'layout.json').read_text()))
    if spare:
        # Small banks have too little spare decoder space; widen its first column.
        bank_dir = parts / f'bank_spare_{int(spare)}'
        build_bank(deck, bank_dir, include_auxiliary=False, folded=True, split=cut, reserve=reserve,
                   bridge=False, spare=spare)
    return build(deck, output, bank_dir)
