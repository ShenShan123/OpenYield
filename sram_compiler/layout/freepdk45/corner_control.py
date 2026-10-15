"""TIME_CONTROL in the bank corner below the decoder, left of the periphery.

The OpenRAM-pitch bank keeps its decoder above the corner and its column
periphery to the right. This plan fills that corner:

* Control logic (TIME_CONTROL minus its data registers) occupies a frame
  whose right edge faces the bank's M6 control trunks and whose top edge
  lies under the decoder's address and wl_en spines. Oversized gates are
  sliced into parallel finger groups. Current controllers use far-end
  feedback and fixed digital delays, without intentional RC guards.
* Every data register folds into its column's slot (``dense_slots``) in a
  band directly below the write drivers. Its Q track continues the write
  slot's buffer-input track; D is a macro pin at the band bottom.
* Interface routes are explicit straight tracks: M6 drives into the bank
  trunks, M5 risers into the decoder, M7 far-end returns at their bar
  heights, an M7/M8 replica-wordline return over the decoder, an M4 register
  clock bar and M4 supply extensions. No routing search is used outside the
  control-logic frame.

The logic's bottom aligns with the data-register band. Excess standard cells
continue upward in a reserved lane of the shared decoder/control floorplan.
The lane keeps the controller's 1500 nm rows clear of the decoder's 1365 nm
rails. Source topology, device sizing and the core pitch remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json
from math import ceil

from .components import Component, Instance, CellView, read_component
from .full_source import flatten
from .dense_views import write_view, expand
from .dense_aux import make_aux
from .dense_slots import make_register_slot
from .lib import Builder, new_layout, pya
from .time_control import make_control, units
from . import macro_control
from .macro_control import Planner, bank_conductor, snap, GRID

CORNER_GAP = 600.    # frame right edge to the bank's control trunks (cx - 150)
TOP_GAP = 1000.      # frame top below the core; decoder supply bars run above
REG_GAP = 300.       # register band top below the write band
RWL_DROP_X = 600.    # M8 replica-wordline return over the decoder (frame-local x)
FRAME_X = 35.        # decoder address spines sit at x = 35 (mod 100): frame grid phase
RETURN_SPACE = 800.  # M7 returns: 400 nm wires, 400 nm spacing
DRIVE_SPACE = 300.   # M6 edge pins
DECODER_GAP = 1000.  # separate the two physical row heights and their rails


def _imported(ly, directory, m):
    src = new_layout(); src.read(str(directory / 'full_macro.gds'))
    cell = Builder(ly, m['primary']); cell.cell.copy_tree(src.cell(m['primary']))
    return cell


def _fuse_decoder_control(layout: pya.Layout, top: pya.Cell, bank: pya.Cell, name: str) -> str:
    """Give decoder gates and controller cells one physical module boundary."""
    instances=[]
    for parent,prefix in ((bank,'DENSE_DECODER_'),(top,'TIME_CONTROL_CORNER_')):
        matches=[i for i in parent.each_inst() if layout.cell(i.cell_index).name.startswith(prefix)]
        if len(matches)!=1:raise ValueError(f'Expected one physical {prefix} block')
        instances.extend(matches)
    shared=Builder(layout,name)
    for instance in instances:
        shared.cell.insert(instance.cell_inst)
        instance.delete()
    top.insert(pya.CellInstArray(shared.cell.cell_index(),pya.Trans()))
    return shared.name


def build(deck, output, bank_dir):
    source = flatten(deck); rows, cols = source.rows, source.cols
    macro_control._conductors.clear()
    latch = source.instances['XD_LATCH']['pins']; sense = source.instances[f'XSENSEAMP_{cols-1}']['pins']
    if latch['D'] != sense['Q'] or latch['EN'] != sense['EN']:
        raise ValueError('The corner plan requires the last-column sense output and enable')
    bank_dir = Path(bank_dir)
    bm = json.loads((bank_dir / 'layout.json').read_text())
    digest = hashlib.sha256(Path(deck).read_bytes()).hexdigest()
    if bm['source_deck_sha256'] != digest:
        raise ValueError('Bank does not match the source deck')
    if hashlib.sha256((bank_dir / 'full_macro.gds').read_bytes()).hexdigest() != bm['layout_sha256']:
        raise ValueError('Bank GDS differs from its layout manifest')
    if set(bm.get('excluded_instances', [])) != {'XTIME_CONTROL', 'XRWL', 'XD_LATCH'} or not bm.get('periphery_bands_nm'):
        raise ValueError('Corner integration requires the folded bank with auxiliary blocks external')
    ly = new_layout(); b = Builder(ly, f'OPENYIELD_CORNER_{rows}X{cols}'); P = Planner(b)
    bank = _imported(ly, bank_dir, bm); b.place(bank)
    bp = bm['external_pins']; cx, cy = bm['array_origin_nm']; bands = bm['periphery_bands_nm']
    endpoints = {}; links = []
    def endpoint(net, point, alias=None):
        endpoints.setdefault(net, []).append((point, alias or net))
    tc = read_component(deck, 'XTIME_CONTROL', allow_passives=True)

    # 1. Data registers folded into their column slots below the write band.
    registers = [(p, c, m) for p, c, m in units(tc) if p.startswith('Xdff_buf_data/')]
    by_q = {m['Q']: (p, c, m) for p, c, m in registers}
    if len(registers) != cols or any(c.identity != registers[0][1].identity for _, c, _ in registers):
        raise ValueError('Every data column needs one identical source register')
    rs = make_register_slot(ly, 'CORNER_REGISTER_SLOT', registers[0][1])
    # The band top stays >= REG_GAP below the write band; its clock bar lands
    # on the routing grid so the controller pin and the bar share one height.
    c_rel = rs.pins['CLK'][0][1]
    clock_y = ((-REG_GAP - rs.height + c_rel) // GRID) * GRID
    reg_y = clock_y - c_rel
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
    # Register supplies: M4 boundary bars to the bank's right supply trunks.
    trunks = {n: bp[n][0] for n in ('VDD', 'VSS')}
    for net, y in rs.supplies:
        tx = trunks[net]; yy = reg_y + y
        P.wire(net, 'm4', [(cx, yy), (tx, yy)], 140.); P.stack(net, tx, yy, 4, 6)
    for net, tx in trunks.items():
        x0, y0, x1, y1 = bank_conductor(bank.cell, 'm6', tx, bp[net][1])
        low = min(reg_y + y for n, y in rs.supplies if n == net)
        P.wire(net, 'm6', [(tx, y0 + 72.5), (tx, low)])
        endpoint(net, (tx, low, 'm6'))

    # 2. Interface heights, from the bank conductors themselves.
    trunk_x = cx - 150
    def trunk(net):
        return bank_conductor(bank.cell, 'm6', trunk_x, bp[net][1])
    drives, returns = {}, {}
    for net, far in (('PRE', 'pre_far'), ('w_en', 'wen_far'), ('s_en', 'sen_far'), ('sa_iso', 'iso_far')):
        _, y0, _, y1 = trunk(net)
        drives[net] = snap((y0 + y1) / 2)
        # Far ends are the replica-reaching (upper) bars; a pin above the
        # frame top would sit on its routing border, so PRE returns from
        # its trunk midpoint and jogs up at the tap.
        returns[far] = (net, y1 - 72.5 if y1 - y0 > 145 else (y0 + y1) / 2)
    pin_y = {far: (drives['PRE'] if far == 'pre_far' else snap(y)) for far, (net, y) in returns.items()}
    rbl_y = snap((bands['write'][1] + bands['sense'][0]) / 2)
    returns['rbl'] = ('RBL', rbl_y); pin_y['rbl'] = rbl_y
    drives['clk_buf'] = clock_y
    # Controller supplies extend the write band's two lowest boundary bars.
    supply_y = {}
    for net, y in bands['write_supplies']:
        supply_y.setdefault(net, y)
    for name, ys, space in (('drive', list(drives.values()) + list(supply_y.values()), DRIVE_SPACE),
                            ('return', list(pin_y.values()), RETURN_SPACE)):
        ys = sorted(ys)
        if any(b2 - a < space for a, b2 in zip(ys, ys[1:])):
            raise ValueError(f'Corner {name} pins closer than {space} nm: {ys}')

    # 3. Shared decoder/control plan: wide corner, narrow upper cell lane.
    available_width = snap(trunk_x - CORNER_GAP - FRAME_X)
    # Small banks have little decoder height; tall banks need more parallel
    # tracks for the feedback nets and the enlarged source buffer slices.
    lane = 10000. if rows<32 else 4000. if rows<=32 else 5000. if rows<=256 else 20000.
    width = max(available_width + lane + DECODER_GAP, 10100.)
    frame_x = FRAME_X + available_width - width
    top = int((cy - TOP_GAP) // GRID) * GRID
    if max(list(drives.values()) + list(pin_y.values())) > top - 500:
        raise ValueError('A corner edge pin lies within 500 nm of the frame top')
    y0 = reg_y
    elbow = snap(top - y0)
    lane_width = snap(-DECODER_GAP - frame_x)
    tops = {f'A_dff{i}': bp[f'A_dff{i}'][0] for i in range(max(1, (rows - 1).bit_length()))}
    tops['wl_en'] = bp['wl_en'][0]
    def plan(height: float) -> dict[str, tuple[float, float, str]]:
        p = {n: (width - 100., snap(y - y0), 'm6') for n, y in drives.items()}
        p.update({n: (width - 300., snap(pin_y[n] - y0), 'm7') for n in returns})
        p.update({n: (snap(x - frame_x), elbow - 100., 'm5') for n, x in tops.items()})
        p['rwl'] = (RWL_DROP_X, height - 300., 'm8')
        p.update({n: (width - 100., snap(y - y0), 'm6') for n, y in supply_y.items()})
        return p
    ctrl = make_control(ly, tc, width, exclude=('Xdff_buf_data/',), pin_plan=plan, corner=True,
                        min_height=elbow+1000., overflow=(elbow, lane_width))
    if y0+ctrl.height_nm>bm['decoder_bbox_nm'][3]:
        raise ValueError('Controller overflow lane exceeds the row-decoder height')
    b.place(ctrl.builder, frame_x, y0)
    cp = {n: (x + frame_x, y + y0, l) for n, (x, y, l) in ctrl.pins.items()}
    terminals.update({'XTIME_CONTROL/' + n: (x + frame_x, y + y0, l) for n, (x, y, l) in ctrl.device_pins.items()})
    for f in ctrl.physical_fingers:
        f = dict(f, device='XTIME_CONTROL/' + f['device'])
        if 'gate_xy_nm' in f: f['gate_xy_nm'] = (f['gate_xy_nm'][0] + frame_x, f['gate_xy_nm'][1] + y0)
        physical.append(f)

    # 4. Drives: one straight M6 line into each control trunk.
    for net in ('PRE', 'w_en', 's_en', 'sa_iso'):
        x, y, _ = cp[net]
        _, by0, _, by1 = trunk(net)
        # A point trunk takes its exact height; the grid pin jogs once outside the frame.
        ty = y if by1 - by0 > 300 else (by0 + by1) / 2
        if not by0 + 72.5 <= ty <= by1 - 72.5: raise ValueError(f'{net} drive misses its trunk')
        P.wire(net, 'm6', [(x, y), (x + 300, y), (x + 300, ty), (trunk_x, ty)])
        endpoint(net, (x, y, 'm6')); endpoint(net, (trunk_x, ty, 'm6'))
        links.append(dict(controller_port=net, route='M6 straight into the bank trunk', y_nm=y))
    # Register clock: M6 to an M4 bar along the register band.
    x, y, _ = cp['clk_buf']
    if abs(y - clock_y) > GRID / 2: raise ValueError('Register clock pin is off its bar')
    P.wire('clk_buf', 'm6', [(x, y), (cx - 300, y)]); P.stack('clk_buf', cx - 300, y, 4, 6)
    P.wire('clk_buf', 'm4', [(cx - 300, y), (max(clock_xs), y)], 140.)
    endpoint('clk_buf', (x, y, 'm6')); endpoint('clk_buf', (clock_xs[0], y, 'm4'))
    links.append(dict(controller_port='clk_buf', route='M6 + M4 register clock bar', y_nm=y))
    # Supplies: M4 extensions of the write band's boundary bars.
    for net, y in supply_y.items():
        x, py, _ = cp[net]
        P.wire(net, 'm6', [(x, py), (cx - 450, py)]); P.stack(net, cx - 450, py, 4, 6)
        P.wire(net, 'm4', [(cx - 450, py), (cx - 450, y), (cx, y)], 140.)
        endpoint(net, (x, py, 'm6')); endpoint(net, (cx, y, 'm4'))
    # 5. Address and wl_en: M5 risers into the decoder spines.
    for net, x in tops.items():
        px, py, _ = cp[net]; tx, ty, tl = bp[net]
        if abs(px - tx) > GRID / 2 or tl != 'm5': raise ValueError(f'{net}: decoder spine is not above its pin')
        # Grid pin to the spine: a straight riser with at most a 50 nm jog.
        P.wire(net, 'm5', [(px, py), (px, ty), (tx, ty)])
        endpoint(net, (px, py, 'm5')); endpoint(net, (tx, ty, 'm5'))
        links.append(dict(controller_port=net, route='M5 riser into the decoder', x_nm=px))
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
        yb = (b0 + t0) / 2; bars[port] = (b1, yb)
        P.wire(net, 'm4', [(b1 - 70, yb), (tap, yb)], 140.); P.stack(net, tap, yb, 4, 7)
        path = [(x, ry), (tap, ry)] if abs(ry - yb) <= 30 else [(x, ry), (tap, ry), (tap, yb)]
        P.wire('RETURN_' + port, 'm7', path)
        endpoint('RETURN_' + port, (x, ry, 'm7'), net); endpoint('RETURN_' + port, (b1 - 100, yb, 'm4'), net)
        links.append(dict(controller_port=port, bank_net=net, route='M4 bar end + M7', y_nm=ry))
    # 7. Replica wordline: far replica row, M7 above the array, M8 down.
    rwl_tap = (cx + (cols + 1) * 705 + 500, cy + (rows - .5) * 1365)
    bank_conductor(bank.cell, 'm4', *rwl_tap)
    x, ry, _ = cp['rwl']
    P.stack('RWL', *rwl_tap, 4, 7); P.wire('RETURN_rwl', 'm7', [rwl_tap, (x, rwl_tap[1])])
    P.stack('RWL', x, rwl_tap[1], 7, 8); P.wire('RETURN_rwl', 'm8', [(x, rwl_tap[1]), (x, ry)])
    endpoint('RETURN_rwl', (x, ry, 'm8'), 'RWL'); endpoint('RETURN_rwl', (*rwl_tap, 'm4'), 'RWL')
    links.append(dict(controller_port='rwl', bank_net='RWL', route='M7 above the array + M8 over the decoder'))

    # 8. Replica wordline driver above the decoder; output latch at the right.
    auxiliaries, positions, apins = {}, {}, {}
    ceiling = rows <= 32 and cols <= 32
    rwl_c = read_component(deck, 'XRWL'); rwl_v = make_aux(ly, rwl_c, ceiling)
    zx_local, zy_local, _ = rwl_v.pins['Z']
    dec_right = bm['decoder_bbox_nm'][2]
    rpx = dec_right - rwl_v.width_nm; rpy = cy + (rows + .5) * 1365 - zy_local
    latch_c = read_component(deck, 'XD_LATCH'); latch_v = make_aux(ly, latch_c)
    lpx = tap + 300; lpy = bp[f'SA_Q{cols-1}'][1] - 200
    for name, v, x, y in (('XRWL', rwl_v, rpx, rpy), ('XD_LATCH', latch_v, lpx, lpy)):
        auxiliaries[name] = v; positions[name] = (x, y); b.place(v.builder, x, y)
        apins[name] = {p: (px + x, py + y, l) for p, (px, py, l) in v.pins.items()}
    rp_ = apins['XRWL']; lp = apins['XD_LATCH']
    b.rect('pwell', rpx, cy + rows * 1365, rpx + rwl_v.width_nm, rpy)
    ax, ay, _ = rp_['A']; vx, vy, _ = rp_['VDD']
    P.wire('VDD', 'm3', [(ax, ay), (ax, rpy + 1100), (vx, rpy + 1100)]); P.stack('VDD', vx, vy, 1, 3)
    P.wire('VDD', 'm3', [(vx, rpy + 1100), (vx, vy)])
    zx, zy, _ = rp_['Z']
    if abs(zy - (cy + (rows + .5) * 1365)) > 1e-6: raise ValueError('Replica-driver output is not aligned to the core RWL')
    P.stack('RWL', zx, zy, 3, 6)
    bx, by, _ = rp_['B']; spine = (bp['wl_en'][0], cy + (rows - 1) * 1365 + 80)
    bank_conductor(bank.cell, 'm5', *spine)
    P.wire('wl_en', 'm3', [(bx, by), (bx, rpy + 385)]); P.stack('wl_en', bx, rpy + 385, 3, 4)
    P.wire('wl_en', 'm4', [(bx, rpy + 385), (spine[0], rpy + 385)]); P.stack('wl_en', spine[0], rpy + 385, 4, 5)
    P.wire('wl_en', 'm5', [(spine[0], rpy + 385), spine])
    endpoint('wl_en', (bx, by, 'm3'))
    for net, rail_x, top_y in (('VSS', 200., cy + (rows - 1) * 1365 + 100), ('VDD', 50., cy + (rows - 1) * 1365 + 1265)):
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
    interior=(cx+1410,cy+2730,cx+(cols-1)*705,cy+(rows-2)*1365) if rows*cols>4096 else None
    port_conductors(b, check_ports, check_aliases, require_connected=True,exclude_bbox_nm=interior)

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
    shared_cell = _fuse_decoder_control(ly,b.cell,bank.cell,f'ROW_DECODER_TIME_CONTROL_{rows}X{cols}')
    shared_box = ly.cell(shared_cell).dbbox()
    corner = dict(x_nm=[0., cx - 150.], y_nm=[reg_y, cy - TOP_GAP])
    frame = [frame_x, y0, frame_x + width, y0+ctrl.height_nm]
    regions = [[frame_x,y0,frame_x+width,y0+elbow],
               [frame_x,y0+elbow,frame_x+lane_width,y0+ctrl.height_nm]]
    control_area = width*elbow+lane_width*(ctrl.height_nm-elbow)
    overlap = sum(max(0.,min(r[2],corner['x_nm'][1])-max(r[0],corner['x_nm'][0]))*
                  max(0.,min(r[3],corner['y_nm'][1])-max(r[1],corner['y_nm'][0])) for r in regions)
    overflow = control_area-overlap
    metadata = dict(artifact_kind='bank', rows=rows, cols=cols, controller_layout=True, floorplan='corner',
        included_instances=list(bm['included_instances']) + ['XTIME_CONTROL', 'XRWL', 'XD_LATCH'], excluded_instances=[],
        routing='control logic: grid router inside its frame; interface: planned straight tracks',
        controller_frame_nm=frame, controller_regions_nm=regions, controller_area_um2=control_area / 1e6,
        corner_nm=corner, corner_area_um2=(corner['x_nm'][1] - corner['x_nm'][0]) * (corner['y_nm'][1] - corner['y_nm'][0]) / 1e6,
        corner_overflow_um2=overflow / 1e6, fits_corner=overflow<=1e-6,
        controller_left_extension_nm=max(0.,FRAME_X-frame_x),
        controller_bottom_alignment='data-register band', controller_bottom_offset_nm=y0-reg_y,
        controller_downward_overflow_nm=max(0.,reg_y-y0),
        decoder_control_bbox_nm=[v*1000 for v in (shared_box.left,shared_box.bottom,shared_box.right,shared_box.top)],
        decoder_control_floorplan_nm=[frame_x,y0,bm['decoder_bbox_nm'][2],bm['decoder_bbox_nm'][3]],
        decoder_control_cell=shared_cell,
        decoder_control_lane_nm=regions[1], decoder_control_lane_area_um2=lane_width*(ctrl.height_nm-elbow)/1e6,
        controller_placements=[dict(p,x_nm=p['x_nm']+frame_x,y_nm=p['y_nm']+y0) for p in ctrl.placements],
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
    return write_view(v, output, deck, 'complete bank with TIME_CONTROL in the decoder/periphery corner', metadata)


def generate(deck, output):
    from .dense_bank import build as build_bank
    output = Path(output); parts = output.with_name(output.name + '_parts')
    bank_dir = parts / 'bank'
    build_bank(deck, bank_dir, include_auxiliary=False, folded=True)
    return build(deck, output, bank_dir)
