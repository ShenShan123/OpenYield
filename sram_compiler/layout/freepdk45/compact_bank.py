"""Compact column-split bank with one shared decoder and a 2:1 mux.

Each full-row half-array has its own 2250 nm column-pair periphery below it.
The right half owns the replica column. Wordlines cross the decoder on M6;
bitlines drop locally on M5. TIME_CONTROL, the RWL driver and output latch
remain external until compact_center integrates them.
"""
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from .cell_flow import leaves
from .components import Component, Instance, CellView, read_component
from .compact_array import make_core
from .compact_core import W, H, HS, bitline_x
from .compact_decoder import make_decoder
from .compact_slots import WIDTH as SLOT_W, make_sense_slot, make_write_slot, make_precharge_slot
from .dense_views import write_view
from .full_source import FullSource, flatten
from .lib import Builder, new_layout

CONTROLS = ('PRE', 'w_en', 's_en', 'sa_iso', 'SEL0', 'SEL1')
EXCLUDED = ('XTIME_CONTROL', 'XRWL', 'XD_LATCH')
DECODER_GAP = 225.             # FreePDK45 opposite-well spacing at array edges


def replica_contract(source: FullSource, specs: dict[str, Component], rows: int, cols: int) -> None:
    name = f'Xsram_{rows}x1_replica_column'
    column = specs[name]
    pins = source.instances[name]['pins']
    expected_pins = dict(VDD='VDD',VSS='VSS',RBL='RBL',RBLB='RBLB',
                         **{f'WL{r}': 'RWL' if r==rows-1 else 'VSS' for r in range(rows)})
    if pins != expected_pins:
        raise ValueError('Unsupported replica-column boundary mapping')
    widths = {'MPGL':135,'MPGR':135,'MPDL':205,'MPDR':205,'MPUL':90,'MPUR':90}
    def check(cell, kind):
        nodes = {'MPGL':('RBL','WL','Q','VSS'), 'MPGR':('RBLB','WL','VDD','VSS'),
                 'MPDL':('Q','VDD','VSS','VSS'), 'MPDR':('VDD','Q','VSS','VSS'),
                 'MPUL':('Q','VDD','VDD','VDD'), 'MPUR':('VDD','Q','VDD','VDD')}
        if kind=='passive':
            nodes.update({n:(v[0],'VSS',v[2],v[3]) for n,v in nodes.items() if n not in ('MPGL','MPGR')})
        if kind=='dummy':
            nodes['MPGL']=('bl_node_noconnect','WL','Q','VSS')
            nodes['MPGR']=('blb_node_noconnect','WL','VDD','VSS')
        if {d.name for d in cell.devices} != set(widths):
            raise ValueError('Replica/dummy device inventory differs from dense view')
        for d in cell.devices:
            model = 'PMOS_VTG' if d.name.startswith('MPU') else 'NMOS_VTG'
            if (d.nodes!=nodes[d.name] or d.model!=model or d.fingers!=1 or
                    abs(d.width_m*1e9-widths[d.name])>1e-6 or abs(d.length_m*1e9-50)>1e-6):
                raise ValueError('Replica/dummy topology or W/L/NF differs from dense view')
    children=list(leaves(column))
    if len(children)!=rows:
        raise ValueError('Replica column must retain every source cell')
    for r,(path,cell,mapping) in enumerate(children):
        kind='replica' if r==rows-1 else 'passive'
        expected_name='Replica_CELL' if kind=='replica' else 'Replica_CELL_PASSIVE'
        if (path!=f'XReplica_CELL_{r}' or cell.name!=expected_name or
                mapping!=dict(VDD='VDD',VSS='VSS',RBL='RBL',RBLB='RBLB',WL=f'WL{r}')):
            raise ValueError('Replica cell placement differs from dense view')
        check(cell,kind)
    for c in range(cols-1):
        name=f'XRWL_LOAD_{c}'
        if source.instances[name]['pins']!=dict(VDD='VDD',VSS='VSS',BL='VDD',BLB='VDD',WL='RWL'):
            raise ValueError('Dummy load boundary differs from dense view')
        check(specs[name],'dummy')


def compact_contract(deck: Union[str, Path]) -> tuple[FullSource, str, dict[str, Component], dict[str, str]]:
    """Bind the muxed source to the compact physical plan."""
    source = flatten(deck)
    rows, cols = source.rows, source.cols
    if rows < 8 or rows % 8 or rows > 512 or cols < 4 or cols % 4 or cols > 512:
        raise ValueError('Compact bank supports 8-512 rows (multiples of 8) and 4-512 columns (multiples of 4)')
    inst = source.instances
    groups = cols // 2
    core = f'XSRAM_6T_CORE_{rows}x{cols}'
    names = [core, f'Xsram_{rows}x1_replica_column'] + [f'XRWL_LOAD_{c}' for c in range(cols - 1)]
    names += ['XDECODER'] + [f'XWL_DRV_{r}' for r in range(rows)]
    names += [f'{p}_{c}' for c in range(cols) for p in ('XPRECHARGE', 'XWRITEDRIVER', 'XDIN_BUF')]
    names += [f'{p}_{g}' for g in range(groups) for p in ('XCOLUMNMUX2', 'XSENSEAMP')]
    names += ['XPRECHARGE_RBL', 'XREPLICA_WDRV_LOAD', 'XREPLICA_MUX', 'XREPLICA_SENSEAMP']
    missing = [n for n in names if n not in inst]
    if missing:
        raise ValueError(f'Compact bank requires a 2:1 column-mux deck; missing {missing[:4]}')
    top = {n for n in inst if '/' not in n}
    if top - set(names) - set(EXCLUDED):
        raise ValueError(f'Unexpected source instances: {sorted(top - set(names) - set(EXCLUDED))[:4]}')
    specs = {n: read_component(deck, n) for n in names}
    for g in range(groups):
        m, sa = inst[f'XCOLUMNMUX2_{g}']['pins'], inst[f'XSENSEAMP_{g}']['pins']
        want = dict(BL0=f'BL{2 * g}', BL1=f'BL{2 * g + 1}', BLB0=f'BLB{2 * g}', BLB1=f'BLB{2 * g + 1}',
                    SEL0='SEL0', SEL1='SEL1', SA_IN=f'SA_IN{g}', SA_INB=f'SA_INB{g}')
        if any(m[k] != v for k, v in want.items()) or sa['IN'] != m['SA_IN'] or sa['INB'] != m['SA_INB']:
            raise ValueError(f'Column group {g} differs from the compact mux plan')
    rm, rs = inst['XREPLICA_MUX']['pins'], inst['XREPLICA_SENSEAMP']['pins']
    want = dict(BL0='RBL', BLB0='RBLB', BL1='VDD', BLB1='VDD', SEL0='VDD', SEL1='VSS',
                SA_IN='RBL_MUX', SA_INB='RBLB_MUX')
    if any(rm[k] != v for k, v in want.items()) or (rs['IN'], rs['INB']) != ('RBL_MUX', 'RBLB_MUX'):
        raise ValueError('Replica mux differs from the compact plan')
    for c in range(cols):
        for p in ('XPRECHARGE', 'XWRITEDRIVER'):
            pins = inst[f'{p}_{c}']['pins']
            if (pins['BL'], pins['BLB']) != (f'BL{c}', f'BLB{c}'):
                raise ValueError(f'{p}_{c} is not on its own column')
    for name, master in [('XPRECHARGE_RBL', 'XPRECHARGE_0'), ('XREPLICA_WDRV_LOAD', 'XWRITEDRIVER_0'),
                         ('XREPLICA_MUX', 'XCOLUMNMUX2_0'), ('XREPLICA_SENSEAMP', 'XSENSEAMP_0')]:
        if specs[name].identity != specs[master].identity:
            raise ValueError(f'{name} differs from its column physical master')
    for kind in ('XPRECHARGE', 'XWRITEDRIVER', 'XDIN_BUF', 'XCOLUMNMUX2', 'XSENSEAMP'):
        ids = {specs[n].identity for n in specs if n.startswith(kind + '_')}
        if len(ids) != 1:
            raise ValueError(f'{kind} instances differ physically')
    replica_contract(source, specs, rows, cols)
    controls = dict(PRE=inst['XPRECHARGE_0']['pins']['ENB'], w_en=inst['XWRITEDRIVER_0']['pins']['EN'],
                    s_en=inst['XSENSEAMP_0']['pins']['EN'], sa_iso=inst['XSENSEAMP_0']['pins']['ISO'],
                    SEL0='SEL0', SEL1='SEL1')
    return source, core, specs, controls


def build(deck: Union[str, Path], output: Union[str, Path], core_x: Optional[float] = None,
          controller_channel: bool = False, decoder_pad: float = 0.) -> dict:
    """Full-row half-arrays beside one shared decoder and muxed periphery.

    ``core_x`` widens the central channel for TIME_CONTROL. With
    ``controller_channel`` the central M4 bars cross on M6, leaving room
    for the controller and separate half-array supply feeds.
    ``decoder_pad`` widens the decoder's empty left region instead, so the
    spare lower decoder rows grow along with the channel.
    """
    source, core_name, specs, controls = compact_contract(deck)
    inst = source.instances
    rows, cols = source.rows, source.cols
    ly = new_layout()
    b = Builder(ly, f'COMPACT_BANK_{rows}X{cols}')
    row_specs = {n: specs[n] for n in ['XDECODER'] + [f'XWL_DRV_{r}' for r in range(rows)]}
    decoder = make_decoder(ly, source, row_specs, pad=decoder_pad)
    core = make_core(ly, source, specs, core_name, column_span=(cols // 2, cols))
    core_left = make_core(ly, source, specs, core_name, column_span=(0, cols // 2))
    wslot = make_write_slot(ly, 'CMP_WRITE_SLOT', {'L': specs['XWRITEDRIVER_0'], 'U': specs['XWRITEDRIVER_1']},
                            {'L': specs['XDIN_BUF_0'], 'U': specs['XDIN_BUF_1']})
    wrep = make_write_slot(ly, 'CMP_WRITE_SLOT_RBL', {'U': specs['XREPLICA_WDRV_LOAD']},
                           {'U': specs['XDIN_BUF_0']}, replica=True)
    sslot = make_sense_slot(ly, 'CMP_SENSE_SLOT', specs['XCOLUMNMUX2_0'], specs['XSENSEAMP_0'])
    srep = make_sense_slot(ly, 'CMP_SENSE_SLOT_RBL', specs['XREPLICA_MUX'], specs['XREPLICA_SENSEAMP'], replica=True)
    pslot = make_precharge_slot(ly, 'CMP_PRECHARGE_SLOT', {'L': specs['XPRECHARGE_0'], 'U': specs['XPRECHARGE_1']})
    prep = make_precharge_slot(ly, 'CMP_PRECHARGE_SLOT_RBL', {'U': specs['XPRECHARGE_RBL']}, replica=True)
    # Bands abut where facing rails share a net. Otherwise 425 nm keeps the
    # facing wells (each overhangs 100 nm) 225 nm apart and the two rails'
    # full-width M4 supply bars clear of each other.
    def gap(lower, upper):
        top = max(lower.supplies, key=lambda t: t[1]) if lower.supplies else None
        bottom = min(upper.supplies, key=lambda t: t[1]) if upper.supplies else None
        if top and bottom and top[0] == bottom[0] and abs(top[1] - lower.height) < 1e-6 and abs(bottom[1]) < 1e-6:
            return 0.
        return 425.
    sa_y = wslot.height + gap(wslot, sslot)
    pc_y = sa_y + sslot.height + gap(sslot, pslot)
    cy = pc_y + pslot.height + 1000.
    dec_w = decoder.width_nm
    # The control trunks only span the periphery band, so they sit below the
    # decoder's right edge. Opposite wells set the minimum array clearance.
    dx = core_left.width_nm + DECODER_GAP
    cx = dx + dec_w + DECODER_GAP if core_x is None else max(dx + dec_w + DECODER_GAP, core_x)
    def column_x(column: int) -> float:
        return column * W if column < cols // 2 else cx + (column - cols // 2) * W
    terminals, fingers, placed, pins = {}, [], [], {}

    def wire(layer, x, y, xx, yy, w=None):
        if controller_channel and layer == 'm4' and y == yy and y < cy:
            left, right = dx - 200., cx - 2100.
            if min(x, xx) < left and max(x, xx) > right:
                wire('m4', min(x, xx), y, left, y, w)
                wire('m4', right, y, max(x, xx), y, w)
                for sx in (left, right):
                    via(4, sx, y); via(5, sx, y)
                wire('m6', left, y, right, y)
                return
        w = w or (70 if int(layer[1:]) < 4 else 140 if layer == 'm4' else 145)
        b.rect(layer, min(x, xx) - w / 2, min(y, yy) - w / 2, max(x, xx) + w / 2, max(y, yy) + w / 2)

    def via(n, x, y):
        b.via(n, x, y)
        for k in (n, n + 1):
            if k < 4:
                b.rect(f'm{k}', x - 35, y - 70, x + 35, y + 70)
            else:
                b.sq(f'm{k}', x, y, 140 if k == 4 else 145)

    def lift(point, upper=4):
        x, y, l = point
        for n in range(int(l[1:]), upper):
            via(n, x, y)
        return x, y

    def add(v, x, y, prefix=''):
        b.place(v.builder, x, y)
        terminals.update({'/'.join(filter(None, (prefix, n))): (x + px, y + py, l) for n, (px, py, l) in v.device_pins.items()})
        fs = getattr(v, 'physical_fingers', None)
        if fs is None:
            fs = [dict(device='/'.join(filter(None, (p, d.name))), finger=i, width_nm=d.width_m * 1e9 / d.fingers,
                       length_nm=d.length_m * 1e9, model=d.model, nodes=d.nodes)
                  for p, c, _ in leaves(v.component) for d in c.devices for i in range(d.fingers)]
        for f in fs:
            f = dict(f, device='/'.join(filter(None, (prefix, f['device']))))
            if 'gate_xy_nm' in f:
                f['gate_xy_nm'] = (x + f['gate_xy_nm'][0], y + f['gate_xy_nm'][1])
            fingers.append(f)
        placed.append(dict(instance=prefix or v.component.name, x_nm=x, y_nm=y, width_nm=v.width_nm, height_nm=v.height_nm))
        return {p: (x + px, y + py, l) for p, (px, py, l) in v.pins.items()}

    def place_slot(view, x, y, roles):
        b.place(view.builder, x, y)
        for key, (px, py, l) in view.terminals.items():
            pre, rest = key.split('.', 1)
            terminals[roles[pre] + '/' + rest] = (x + px, y + py, l)
        for f in view.fingers:
            pre, rest = f['device'].split('.', 1)
            gx, gy = f['gate_xy_nm']
            fingers.append(dict(f, device=roles[pre] + '/' + rest, gate_xy_nm=(x + gx, y + gy)))
        for name in dict.fromkeys(roles.values()):
            placed.append(dict(instance=name, cell=view.builder.name, x_nm=x, y_nm=y,
                               width_nm=view.width, height_nm=view.height))
        return {k: [(x + px, y + py, l) for px, py, l in v] for k, v in view.pins.items()}

    cp = add(core, cx, cy)
    dp = add(decoder, dx, cy + HS)
    left_supply_layers = {}
    cp_left = add(core_left, 0., cy)
    wl_pins, core_pins = dp, cp
    for r in range(rows):
        wire('m3', *wl_pins[f'WL{r}'][:2], *core_pins[f'WL{r}'][:2])
        a, z = dp[f'WL{r}'], cp_left[f'WL{r}']
        lift(a, 6); lift(z, 6)
        wire('m6', *a[:2], *z[:2])
    a, z = cp['RWL'], cp_left['RWL']
    lift(a, 6); lift(z, 6)
    wire('m6', *a[:2], *z[:2])
    # Column periphery: pair p holds columns 2p (L) and 2p+1 (U); the last
    # slot's U role is the replica column.
    bars = defaultdict(set)          # net -> {(x, y)} M4 bar pins
    supply = defaultdict(lambda: defaultdict(list))
    stacks = defaultdict(list)       # bitline x -> slot pin ys
    pairs = cols // 2
    for p in range(pairs + 1):
        x = column_x(2 * p)
        last = p == pairs
        c = 2 * p
        if last:
            w = place_slot(wrep, x, 0., {'U': 'XREPLICA_WDRV_LOAD', 'L': 'XREPLICA_WDRV_LOAD'})
            s = place_slot(srep, x, sa_y, {'M': 'XREPLICA_MUX', 'A': 'XREPLICA_SENSEAMP'})
            q = place_slot(prep, x, pc_y, {'U': 'XPRECHARGE_RBL', 'L': 'XPRECHARGE_RBL'})
            views = ((wrep, 0.), (srep, sa_y), (prep, pc_y))
            pins['RBL_MUX'] = s['RBL_MUX'][0]      # TIME_CONTROL's replica sense input
        else:
            w = place_slot(wslot, x, 0., {'L': f'XWRITEDRIVER_{c}', 'U': f'XWRITEDRIVER_{c + 1}',
                                          'LB': f'XDIN_BUF_{c}', 'UB': f'XDIN_BUF_{c + 1}'})
            s = place_slot(sslot, x, sa_y, {'M': f'XCOLUMNMUX2_{p}', 'A': f'XSENSEAMP_{p}'})
            q = place_slot(pslot, x, pc_y, {'L': f'XPRECHARGE_{c}', 'U': f'XPRECHARGE_{c + 1}'})
            views = ((wslot, 0.), (sslot, sa_y), (pslot, pc_y))
            for role, col in (('L', c), ('U', c + 1)):
                pins[inst[f'XDIN_BUF_{col}']['pins']['A']] = w[role + 'B.A'][0]
            pins[inst[f'XSENSEAMP_{p}']['pins']['Q']] = s['A.Q'][0]
            pins[inst[f'XSENSEAMP_{p}']['pins']['QB']] = s['A.QB'][0]
            for net in ('SEL0', 'SEL1'):
                bars[net].add(s[net][0][:2])
        for view, y0 in views:
            for net, y in view.supplies:
                supply[net][y0 + y].append(x)
        for role in (('U',) if last else ('L', 'U')):
            for net in ('BL', 'BLB'):
                for slot_pins in (w, q):
                    px, py, _ = slot_pins[f'{role}.{net}'][0]
                    stacks[px].append(py)
            bars['w_en'].add(w['EN'][0][:2])
            bars['PRE'].add(q[role + '.ENB'][0][:2])
        keys = ('RBL', 'RBLB') if last else ('L.BL', 'L.BLB', 'U.BL', 'U.BLB')
        for k in keys:
            px, py, _ = s[k][0]
            stacks[px].append(py)
        bars['s_en'].add(s['A.EN'][0][:2])
        bars['sa_iso'].add(s['A.ISO'][0][:2])
    # Bitlines: one M5 line per column from the lowest slot stack to a lift
    # in the core's bottom strap (the core pin is the strap's M2 line).
    expected = {column_x(c) + bitline_x(c)[k] for c in list(range(cols)) + [cols + 1] for k in (0, 1)}
    if set(stacks) != expected:
        raise ValueError('Slot bitline stacks do not match the core columns')
    for x, ys in stacks.items():
        top = cy + 70.
        wire('m5', x, min(ys), x, top)
        lift((x, top, 'm2'), 5)
    for pins_, label in ((cp, 'right'),):
        for c in range(cols // 2, cols):
            for n in ('BL', 'BLB'):
                if abs(pins_[f'{n}{c}'][0] - (column_x(c) + bitline_x(c)[n == 'BLB'])) > 1e-6:
                    raise ValueError(f'{label} core bitline pin off its slot stack')
    for c in range(cols // 2):
        for n in ('BL', 'BLB'):
            if cp_left[f'{n}{c}'][0] != column_x(c) + bitline_x(c)[n == 'BLB']:
                raise ValueError('Left core bitline pin off its slot stack')
    # Control bars: M4 from each slot pin to the net's own M6 trunk.
    trunk_x = {net: cx - 150. - 300. * i for i, net in enumerate(CONTROLS)}
    trunk_y = defaultdict(list)
    for net, sites in bars.items():
        for y in sorted({y for x, y in sites}):
            xs = [x for x, yy in sites if yy == y]
            tx = trunk_x[net]
            wire('m4', min([tx] + xs), y, max([tx] + xs), y)
            via(4, tx, y); via(5, tx, y)
            trunk_y[net].append(y)
    for net, ys in trunk_y.items():
        wire('m6', trunk_x[net], min(ys), trunk_x[net], max(ys))
        pins[controls[net]] = (trunk_x[net], min(ys), 'm6')
    # Supplies: M4 bars on every slot boundary rail, M6 trunks right of the core.
    root = {'VDD': cx + core.width_nm + 600., 'VSS': cx + core.width_nm + 900.}
    left_root = {net: cp_left[net][0] for net in ('VDD', 'VSS')} if controller_channel else {}
    rail_y = defaultdict(list)
    for net, levels in supply.items():
        for y, xs in levels.items():
            if controller_channel:
                # Each half feeds its own rails. Repeated M6 power bridges
                # through the controller would consume its signal lift sites.
                lx = left_root[net]
                wire('m4', min(xs), y, lx, y)
                via(4, lx, y); via(5, lx, y)
                right = cx - 2100.
                wire('m4', right, y, root[net], y)
                via(4, right, y); via(5, right, y)
                wire('m6', right - 1000., y, right, y)
            else:
                wire('m4', min(xs), y, root[net], y)
            via(4, root[net], y); via(5, root[net], y)
            rail_y[net].append(y)
    for net, lx in left_root.items():
        # The left core already joins the decoder supply feed. End below
        # its first M6 wordline crossing and lift onto its existing M4 trunk.
        wire('m6', lx, min(supply[net]), lx, cy)
        via(4, lx, cy); via(5, lx, cy)
    # Core trunks run right one row inside each array: VDD on M6 to the
    # nearer root, VSS on M5 under the VDD root trunk to its own.
    for pins_, base in ((cp, cy + HS + 1700.),):
        for net in ('VDD', 'VSS'):
            x, y, _ = pins_[net]
            yy = base + (300. if net == 'VSS' else 0.)
            layer = 'm6' if net == 'VDD' else 'm5'
            wire('m4', x, y, x, yy)
            via(4, x, yy)
            if net == 'VDD':
                via(5, x, yy)
            wire(layer, x, yy, root[net], yy)
            via(5, root[net], yy)
            rail_y[net].append(yy)
    for net, yy in (('VDD', cy - 600.), ('VSS', cy - 300.)):
        x, y, _ = cp_left[net]
        target = dp[net][0]
        # A fully used three-bit predecode has an input spine only
        # 150 nm from VSS. Its controller riser crosses this feed, so
        # use the clear M3 channel below the core instead of M5.
        layer = ('m3' if controller_channel and net == 'VSS' and any(
            l == 'm5' and abs(px - target) < 285. for px, _, l in dp.values()) else 'm5')
        left_supply_layers[net] = layer
        wire('m4', x, y, x, yy)
        cut = 3 if layer == 'm3' else 4
        via(cut, x, yy)
        wire(layer, x, yy, target, yy)
        via(cut, target, yy)
    # Decoder M2 rails leave each half on its band side, then M4 bars run right.
    halves = [(dp, (('VDD', cy - 600.), ('VSS', cy - 300.)))]
    for pins_, levels in halves:
        for net, yy in levels:
            x, y, _ = pins_[net]
            wire('m2', x, y, x, yy)
            lift((x, yy, 'm2'))
            wire('m4', x, yy, root[net], yy)
            via(4, root[net], yy); via(5, root[net], yy)
            rail_y[net].append(yy)
    for net, ys in rail_y.items():
        wire('m6', root[net], min(ys), root[net], max(ys))
        pins[net] = (root[net], min(ys), 'm6')
    # Bank boundary: nets shared with excluded instances or driven at the top.
    excluded = set()
    for n in EXCLUDED:
        excluded |= set(inst[n]['pins'].values())
    included = set()
    for n in specs:
        included |= set(inst[n]['pins'].values())
    ports = sorted((included & excluded) | {'VDD', 'VSS', 'SEL0', 'SEL1'} | set(pins))
    for p in ports:
        if p not in pins:
            if p in wl_pins:
                pins[p] = wl_pins[p]
            elif p in core_pins:
                pins[p] = core_pins[p]
            else:
                raise ValueError(f'No physical access for bank port {p}')
    pins = {p: pins[p] for p in ports}
    for p, (x, y, l) in pins.items():
        b.label(l, p, x, y)
    comp = Component(b.name, tuple(ports), (), tuple(
        Instance(n, tuple(inst[n]['pins'][p] for p in c.ports), c) for n, c in specs.items()))
    v = CellView(comp, b, root['VSS'] + 100., cy + core.height_nm, pins, terminals, [], [], [])
    v.physical_fingers = fingers
    v.placements = placed
    metadata = dict(schema=1, artifact_kind='bank', rows=rows, cols=cols, controller_layout=False,
                    core_pitch_nm=dict(row=395, column=1125), row_span=rows, column_span=cols,
                    column_mux=2, included_instances=list(specs), excluded_instances=list(EXCLUDED),
                    array_origin_nm=[cx, cy + HS], decoder_bbox_nm=[dx, cy + HS, dx + decoder.width_nm, cy + HS + decoder.height_nm],
                    decoder_clearance_nm=dict(left=DECODER_GAP, right=cx-dx-dec_w,
                                              minimum=DECODER_GAP),
                    decoder_pad_nm=decoder_pad,
                    periphery_bands_nm=dict(write=[0, wslot.height], sense=[sa_y, sa_y + sslot.height],
                                            precharge=[pc_y, pc_y + pslot.height], slot_pitch=SLOT_W, slots=pairs + 1),
                    core_geometry=dict(width_nm=core.width_nm, height_nm=core.height_nm,
                                       replica_x_nm=((cols // 2) + 1) * W),
                    supply_trunks_nm=root, control_trunks_nm={controls[n]: trunk_x[n] for n in CONTROLS},
                    control_bars_nm={controls[n]: ys for n, ys in trunk_y.items()},
                    write_supplies=[list(t) for t in wslot.supplies],
                    external_pins=pins, waveform_validation='not run', defaults_promotable=False)
    rx, ry = core.rwl_riser
    tracks = decoder.wlen_tracks
    first = min(tracks)
    metadata.update(rwl_tap_nm=[cx + rx, cy + ry], rwl_pin_nm=list(cp['RWL']),
                    wl_en_spine_nm=[dx + first, cy + HS + tracks[first][1]],
                    upper_decoder_top_nm=cy + HS + decoder.height_nm)
    half_width, half_height = cols // 2 * W, rows * H
    metadata['split'] = dict(axis='columns', column=cols // 2,
        left_columns=[0, cols // 2], right_columns=[cols // 2, cols], rows=[0, rows],
        left_array_origin_nm=[0., cy + HS], right_array_origin_nm=[cx, cy + HS],
        periphery_origins_nm=[0., cx],
        bitcell_half_array_nm=[half_width, half_height], half_array_aspect_ratio=half_width / half_height,
        decoder='one shared decoder and one source wordline driver per logical row',
        wordlines='M3 within both arrays, M6 crossing the centre decoder',
        bitlines='M2 in each array, local M5 drops to its own column-pair periphery')
    metadata['metal_policy'] = dict(highest_routing_layer='m6',
        core_bitlines='m2', core_wordlines='m3', periphery_bars='m4', bitline_drops='m5',
        decoder_crossings_and_control_supply_trunks='m6', upper_metals_used=False)
    metadata['left_supply_layers'] = left_supply_layers
    if controller_channel:
        metadata['controller_channel_nm'] = [dx - 200., cx - 2100.]
        metadata['supply_bars_nm'] = dict(rail_y)
        metadata['left_supply_trunks_nm'] = left_root
    summary = Path(deck).with_name('summary.json')
    if summary.is_file():
        import hashlib
        import json
        data = json.loads(summary.read_text())
        metadata['compiler_provenance'] = {k: data.get(k) for k in ('compiler_version', 'corner', 'vdd', 'temperature',
            'variation_mode', 'real_cell_mode', 'seed', 'model_sha256', 'interconnect', 'timing', 'driver_sizes')}
        metadata['compiler_summary_sha256'] = hashlib.sha256(summary.read_bytes()).hexdigest()
    return write_view(v, output, deck, 'compact bank: array, decoder and muxed column periphery; '
                      'TIME_CONTROL, RWL driver and output latch external', metadata)
