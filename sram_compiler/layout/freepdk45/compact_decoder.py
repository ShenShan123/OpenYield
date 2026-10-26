"""Source-owned decoder and wordline drivers at the compact 395 nm row pitch.

A 395 nm row cannot hold a gate, so the decoder works in bands of eight
array rows (3160 nm): one 3-to-8 slice per band. Each band holds two 1365 nm
gate rows, row A at its bottom and row B mirrored at its top, VDD sides
facing. Every dense gate keeps its M2 within 220-1055 nm of its own rails,
which leaves room for the slice's seven bus lines (true inputs, complements,
enable) as horizontal M2 between the rows; gate inputs rise or drop to their
line on M3. Column j stacks chains j (row A) and j + 4 (row B): they share
every input line but the first address, and stacked pins on one x reach
lines at least 280 nm apart (C0 below T0; the input inverters 1 and 2 stack
the same way, inverter 3 has row B above it empty). The driver NAND2s take
wl_en from M2 lines along the band edges, so no M4 landing sits among the
wordlines (one line per band boundary serves both rows that face it);
every WL leaves its driver on its own M4 line at its row's y.
Trunks are M5 in each level's left region, where no wordline runs, and each
level reserves only the slots it uses. A predecode slice spreads its chains
over up to four bands of its span.
"""
from collections import defaultdict
from math import ceil
import re

from .components import Component, Instance, CellView
from .cell_flow import leaves
from .dense_logic import make_logic, VSS_Y, VDD_Y, WELL_SPLIT
from .full_source import FullSource
from .lib import Builder, pya
from .compact_core import H as ROW

BAND = 8 * ROW
CELL_M2 = (220., 1055.)    # M2 extent of every decoder gate, from its own rail side
LINE0 = CELL_M2[1] + 105.  # first bus line, 70 nm clear of row A's M2
LINE = 140.
SLOT = 300.                # trunk pitch
# Bottom to top; the last line keeps 70 nm to row B's M2 (BAND - 1055).
LINES = ('C0', 'C2', 'T0', 'EN', 'T1', 'C1', 'T2')
JOGS = ((630., 350.), (BAND - 630., BAND - 350.))  # predecode output M4 jogs of rows A and B


def line_y(name):
    return LINE0 + LINE * LINES.index(name)


def spread(used):
    """(chain columns per gate row, bands) of a slice over `used` free bands."""
    cols = ceil(4 / used)
    return cols, ceil(4 / cols)


def make_decoder(layout: pya.Layout, source: FullSource, specs: dict[str, Component]) -> CellView:
    """One source-owned decoder shared by both full-row column halves."""
    rows = source.rows
    if rows < 8 or rows % 8 or rows > 512:
        raise ValueError('Compact decoder supports 8 to 512 rows in multiples of 8')
    if LINE0 + LINE * (len(LINES) - 1) + 105. > BAND - CELL_M2[1]:
        raise ValueError('Bus lines do not fit between the gate rows')
    every = {p: (c, source.instances[p]['pins']) for n, c in specs.items() for p, c, _ in leaves(c, path=n)}
    b = Builder(layout, f'COMPACT_DECODER_{rows}')
    cache, placed, terminals, fingers = {}, {}, {}, []
    edge_taps = defaultdict(set)
    nbands = rows // 8
    top = nbands * BAND

    def wire(layer, x, y, xx, yy, w=None):
        w = w or (70 if layer in ('m1', 'm2', 'm3') else 140 if layer == 'm4' else 145)
        b.rect(layer, min(x, xx) - w / 2, min(y, yy) - w / 2, max(x, xx) + w / 2, max(y, yy) + w / 2)

    def via(n, x, y):
        b.via(n, x, y)
        for k in (n, n + 1):
            if k < 4:
                b.rect(f'm{k}', x - 35, y - 70, x + 35, y + 70)
            else:
                b.sq(f'm{k}', x, y, 140 if k == 4 else 145)

    def stack(x, y, upper=5):
        """M2 line -> M5 trunk at (x, y)."""
        b.via(2, x, y); b.rect('m3', x - 35, y - 70, x + 35, y + 70)
        for n in range(3, upper):
            via(n, x, y)

    def drop(x, y0, by, net):
        """M3 stub from a bus line to a supply rail of the band. The VDD rails
        lie among the bus lines: their access sits just outside the bus, on an
        M1 stub from the rail."""
        if net == 'VSS':
            options = [(VSS_Y, VSS_Y), (BAND - VSS_Y, BAND - VSS_Y)]
        else:
            options = [(VDD_Y, LINE0 - LINE), (BAND - VDD_Y, line_y(LINES[-1]) + LINE)]
        rail, y1 = min(((by + r, by + a) for r, a in options), key=lambda t: abs(t[1] - y0))
        b.via(2, x, y0); wire('m3', x, y0, x, y1)
        for y in (y0, y1):
            b.rect('m3', x - 35, y - 70, x + 35, y + 70)
        b.via(2, x, y1); b.rect('m2', x - 70, y1 - 35, x + 70, y1 + 35)
        b.via(1, x, y1); b.rect('m1', x - 70, y1 - 35, x + 70, y1 + 35)
        if rail != y1:
            wire('m1', x, y1, x, rail)

    def view_of(path):
        c, m = every[path]
        if c.identity not in cache:
            v = make_logic(layout, c, False, compact=True)
            m2 = v.builder.cell.dbbox_per_layer(v.builder.li('m2'))
            if m2.bottom * 1000 < CELL_M2[0] - 1e-6 or m2.top * 1000 > CELL_M2[1] + 1e-6:
                raise ValueError(f'{path}: gate M2 leaves the {CELL_M2} nm margin of the bus lines')
            cache[c.identity] = v
        return cache[c.identity]

    def put(path, x, by, mirror):
        """Place a gate in row A (mirror False) or row B of the band at by."""
        v = view_of(path)
        y0 = by + BAND if mirror else by
        t = (lambda y: y0 - y) if mirror else (lambda y: y0 + y)
        b.place(v.builder, x, y0, my=mirror)
        if v.edge_tap is not None and 0 < y0 < top:
            tx, ty = v.edge_tap
            edge_taps[(x + tx, y0, ty)].add(mirror)
        placed[path] = (v, x, t, items[path][1], mirror)
        terminals.update({path + '/' + n: (px + x, t(py), l) for n, (px, py, l) in v.device_pins.items()})
        fingers.extend(dict(f, device=path + '/' + f['device'],
                            gate_xy_nm=(f['gate_xy_nm'][0] + x, t(f['gate_xy_nm'][1]))) for f in v.physical_fingers)
        return v

    def point(path, p):
        v, x, t, m, _ = placed[path]
        px, py, l = v.pins[p]
        return x + px, t(py), l

    def rise(path, p, y):
        """M3 stub from an input pin to a line (via2 onto the line)."""
        x, old, _ = point(path, p)
        wire('m3', x, old, x, y)
        b.via(2, x, y); b.rect('m3', x - 35, y - 70, x + 35, y + 70)
        return x

    def link(a, z):
        va, xa, ta, ma, _ = placed[a]; mb = placed[z][3]
        assert ma['Z'] == mb['A']
        px, py, _ = point(z, 'A'); sx = xa + va.output_rail_m2[0]
        if not min(ta(va.output_rail_m2[1]), ta(va.output_rail_m2[2])) <= py <= max(ta(va.output_rail_m2[1]), ta(va.output_rail_m2[2])):
            raise ValueError('Local output access outside rail')
        zy = ta(va.pins['Z'][1])
        if 70 <= abs(py - zy) < 155:
            wire('m3', sx, zy, sx, py)
        else:
            via(2, sx, py)
        wire('m3', sx, py, px, py)

    def tier(root):
        level, index = re.fullmatch(r'.*/XDEC_(\d+)_(\d+)', root).groups()
        return int(level), int(index)

    every_root = sorted({p.rsplit('/', 1)[0] for p in every if '/XINV_A' in p}, key=tier)
    levels = 1 + max(tier(r)[0] for r in every_root)
    span_rows = lambda level: 8 * 8 ** (levels - 1 - level)

    def placement(root):
        level, index = tier(root)
        start = span_rows(level) * index
        end = min(start + span_rows(level), rows)
        return start // 8, min(8, (end - start) // 8)

    def en_of(root):
        return every[root + '/XAND_EN0/XPNAND3'][1]['B']

    cols, multi, enables, ties = {}, defaultdict(bool), defaultdict(set), defaultdict(bool)
    for r in every_root:
        p = placement(r)
        if p:
            level, index = tier(r)
            c, nb = spread(p[1])
            cols[level] = max(cols.get(level, 1), c)
            multi[level] |= nb > 1
            if en_of(r) in ('VDD', 'VSS'):
                ties[level] = True
            else:
                enables[level].add(index % 8)

    roots = [r for r in every_root if placement(r) is not None]

    def owner(p):
        match = re.match(r'(.*/XDEC_\d+_\d+)/', p)
        if match:
            return match.group(1) in roots
        return 0 <= int(re.match(r'XWL_DRV_(\d+)/', p).group(1)) < rows
    items = {p: v for p, v in every.items() if owner(p)}
    final_level = levels - 1
    for r in roots:
        if tier(r)[0] == final_level and placement(r)[1] != 1:
            raise ValueError('A final decoder slice occupies exactly one band')

    # Level regions: trunk slots, the input inverters, then chain columns.
    def chain_paths(root, k):
        chain = [root + f'/XAND{k}/XPNAND3', root + f'/XAND{k}/XPINV',
                 root + f'/XAND_EN{k}/XPNAND3', root + f'/XAND_EN{k}/XPINV']
        if tier(root)[0] == final_level:
            row = (tier(root)[1]) * 8 + k
            dr = sorted((p for p in every if p.startswith(f'XWL_DRV_{row}/')),
                        key=lambda p: len(every[p][0].devices), reverse=True)
            chain += dr
        return chain

    region = {}
    x = 300.
    for level in sorted(cols):
        sample = next(r for r in every_root if tier(r)[0] == level)
        slots = ['T0', 'T1', 'T2'] + (['TIE_VDD'] if ties[level] else [])
        if multi[level]:
            slots += ['C0', 'C1', 'C2']
        slots += [f'E{k}' for k in sorted(enables[level])]
        if level == final_level:
            slots += ['WLEN']         # last: its riser keeps clear of the address trunks' jogs
        slot_x = {s: x + 150 + SLOT * i for i, s in enumerate(slots)}
        inv_x = x + SLOT * len(slots)          # 150 nm past the last trunk
        inv = [view_of(sample + f'/XINV_A{i}') for i in (1, 2, 3)]
        if inv[0] is not inv[1]:
            raise ValueError('Stacked input inverters 1 and 2 must be identical gates')
        inv_w = [v.width_nm for v in inv]
        chain_x = inv_x + inv_w[0] + inv_w[2]  # chains abut the inverters
        chain_w = sum(view_of(p).width_nm for p in chain_paths(sample, 0))
        right = chain_x + cols[level] * chain_w
        region[level] = dict(left=x, slots=slot_x, inv_x=inv_x, inv_w=inv_w, chain_x=chain_x,
                             chain_w=chain_w, right=right)
        x = right + 100
    width = x + 50                             # WL pins 185 nm past the last driver output
    pins, wordlines = {}, []
    boundary = set(source.instances['XDECODER']['pins'].values())
    consumed = {en_of(r) for r in every_root}
    trunks = defaultdict(list)        # (level, slot) -> [(net, y)]
    producer = {}                     # predecode net -> (x, y) of its M4 jog start
    jogs = defaultdict(list)          # (band, row B) -> jog heights in use

    def wlen_y(band, mirror):
        """wl_en line of a gate row: on the band boundary it faces, 105 nm
        inside the decoder at its outer edges."""
        if mirror:
            return (band + 1) * BAND - (105. if band == nbands - 1 else 0.)
        return band * BAND + (105. if band == 0 else 0.)
    for root in roots:
        level, index = tier(root)
        first, used = placement(root)
        ncols, nb = spread(used)
        g = region[level]
        inv = [items[root + f'/XINV_A{i}'][1] for i in (1, 2, 3)]
        en = items[root + '/XAND_EN0/XPNAND3'][1]['B']
        nets = dict(T0=inv[0]['A'], T1=inv[1]['A'], T2=inv[2]['A'],
                    C0=inv[0]['Z'], C1=inv[1]['Z'], C2=inv[2]['Z'], EN=en)
        lookup = {}
        for name, n in nets.items():
            lookup.setdefault(n, name)
        # Stacked chains k, k + 4 share their B, C and EN lines; their A
        # lines (C0 below T0) keep two line positions between the risers.
        for k in range(4):
            ma, mb = items[root + f'/XAND{k}/XPNAND3'][1], items[root + f'/XAND{k + 4}/XPNAND3'][1]
            for port in ('A', 'B', 'C'):
                if ma[port] != mb[port] and LINES.index(lookup[ma[port]]) + 2 > LINES.index(lookup[mb[port]]):
                    raise ValueError(f'{root}: stacked chains {k} and {k + 4} cross on input {port}')
        lo, hi = g['left'] + 150 - 70, g['chain_x'] + cols[level] * g['chain_w']
        for band in range(first, first + nb):
            by = band * BAND
            for name in nets:
                wire('m2', lo, by + line_y(name), hi, by + line_y(name))
            # Inputs: trunks for addresses, complements and enables; rail ties.
            for name, n in nets.items():
                y = by + line_y(name)
                if n in ('VDD', 'VSS'):
                    # A tied address line drops at its own (trunk-free) slot.
                    sx = g['slots'][name] if name.startswith('T') else g['slots']['TIE_VDD']
                    if n == 'VDD' and name.startswith('T'):
                        raise ValueError('Unsupported VDD-tied address bus')
                    drop(sx, y, by, n)
                    continue
                if name.startswith('T'):
                    sx = g['slots'][name]
                elif name.startswith('C'):
                    if nb == 1:
                        continue
                    sx = g['slots'][name]
                else:
                    sx = g['slots'][f'E{index % 8}']
                stack(sx, y)
                trunks[(level, sx)].append((n, y))
            if level == final_level:
                wx = g['slots']['WLEN']
                for y in (wlen_y(band, False), wlen_y(band, True)):
                    if ('wl_en', y) not in trunks[(level, wx)]:
                        wire('m2', wx - 70, y, hi, y)
                        stack(wx, y)
                        trunks[(level, wx)].append(('wl_en', y))
        # Input inverters in the slice's first band: 1 and 3 in row A, 2 above 1.
        by = first * BAND
        spots = {1: (g['inv_x'], False), 2: (g['inv_x'], True), 3: (g['inv_x'] + g['inv_w'][0], False)}
        for i in (1, 2, 3):
            p = root + f'/XINV_A{i}'
            x0, mirror = spots[i]
            put(p, x0, by, mirror)
            rise(p, 'A', by + line_y(f'T{i - 1}'))
            rise(p, 'Z', by + line_y(f'C{i - 1}'))
        for k in range(8):
            mirror = k >= 4
            j = k % 4
            band = first + j // ncols
            by = band * BAND
            x0 = g['chain_x'] + (j % ncols) * g['chain_w']
            chain = chain_paths(root, k)
            xc = x0
            for p in chain:
                xc += put(p, xc, by, mirror).width_nm
            for a, z in zip(chain, chain[1:]):
                link(a, z)
            m = items[chain[0]][1]
            for port in ('A', 'B', 'C'):
                rise(chain[0], port, by + line_y(lookup[m[port]]))
            rise(chain[2], 'B', by + line_y(lookup[items[chain[2]][1]['B']]))
            if level == final_level:
                if items[chain[4]][1]['B'] != 'wl_en':
                    raise ValueError(f'{chain[4]}: driver NAND2 B is not wl_en')
                rise(chain[4], 'B', wlen_y(band, mirror))
                row = index * 8 + k
                zx, zy, _ = point(chain[-1], 'Z')
                y = row * ROW + ROW / 2
                wire('m3', zx, zy, zx, y)
                via(3, zx, y)
                wire('m4', zx, y, width, y)
                b.via(3, width, y); b.rect('m3', width - 35, y - 70, width + 35, y + 70)
                net = items[chain[-1]][1]['Z']
                pins[net] = (width, y, 'm3')
                wordlines.append(dict(row=row, net=net, pin_nm=pins[net]))
            elif items[chain[-1]][1]['Z'] in consumed:
                # Jogs of one band run right side by side: distinct heights.
                free = [y for y in JOGS[mirror] if y not in jogs[(band, mirror)]]
                if not free:
                    raise ValueError(f'{root}: more predecode outputs than jog heights in a gate row')
                jogs[(band, mirror)].append(free[0])
                zx, zy, _ = point(chain[-1], 'Z')
                y = by + free[0]
                wire('m3', zx, zy, zx, y)
                via(3, zx, y)
                producer[items[chain[-1]][1]['Z']] = (zx, y)
    # Facing stretched-driver VSS taps share diffusion at band boundaries.
    # Their 65 nm isolated gap would violate Active.2; joining the same-body
    # taps preserves every MOS and avoids a wider gate row.
    for (tx, y, ty), sides in edge_taps.items():
        if sides != {False, True}:
            raise ValueError('Stretched driver tap lacks its facing neighbour')
        b.rect('active', tx - 45., y - ty + 45., tx + 45., y + ty - 45.)
        b.rect('pplus', tx - 60., y - ty + 60., tx + 60., y + ty - 60.)
    # Predecode outputs: M4 jog to the consumer level's enable slot, M5 trunk.
    consumers = defaultdict(list)
    for (level, sx), uses in trunks.items():
        for n, y in uses:
            consumers[n].append((level, sx, y))
    remote = set()
    for n, (zx, zy) in producer.items():
        uses = consumers.get(n, [])
        if not uses:
            continue                      # disabled predecode branch
        targets = {sx for _, sx, _ in uses}
        if len(targets) > 1:
            raise ValueError(f'{n}: predecode consumers disagree on the trunk slot')
        sx = targets.pop()
        ys = [zy] + [y for _, _, y in uses]
        wire('m4', zx, zy, sx, zy)
        via(4, sx, zy)
        wire('m5', sx, min(ys), sx, max(ys))
        remote.add(n)
    for key in trunks:
        trunks[key] = [(m, y) for m, y in trunks[key] if m not in remote]
    # Address, complement, enable and wl_en trunks.
    wlen_tracks = {}
    for (level, sx), uses in trunks.items():
        by_net = defaultdict(list)
        for n, y in uses:
            by_net[n].append(y)
        for n, ys in by_net.items():
            # Address and wl_en accesses face the controller below.
            external = n in boundary or n == 'wl_en' or (n in consumed and n not in producer)
            if external:
                ys = ys + [100.]
            wire('m5', sx, min(ys), sx, max(ys))
            if external:
                pins[n] = (sx, 100., 'm5')
            if n == 'wl_en':
                wlen_tracks[sx] = (min(ys), max(ys))
    # Supplies: M1 rails along both gate rows of every band, joined by M2 rails.
    for n, x, phases in [('VDD', 50., (VDD_Y, BAND - VDD_Y)), ('VSS', 200., (VSS_Y, BAND - VSS_Y))]:
        ys = sorted(band * BAND + y for band in range(nbands) for y in phases)
        wire('m2', x, ys[0], x, ys[-1])
        for y in ys:
            wire('m1', x, y, width, y)
            b.via(1, x, y); b.rect('m1', x - 70, y - 35, x + 70, y + 35); b.rect('m2', x - 35, y - 70, x + 35, y + 70)
        pins[n] = (x, ys[0], 'm2')
    for band in range(nbands):
        b.rect('nwell', 0, band * BAND + WELL_SPLIT, width, (band + 1) * BAND - WELL_SPLIT)
    if set(placed) != set(items):
        raise ValueError('Unplaced decoder source leaves')
    for n, (x, y, l) in pins.items():
        b.label(l, n, x, y)
    component = Component(b.name, tuple(pins), (), tuple(
        Instance(n, tuple(source.instances[n]['pins'][p] for p in c.ports), c) for n, c in specs.items()))
    v = CellView(component, b, width, top, pins, terminals, [], [], [])
    v.physical_fingers = fingers
    v.placements = [dict(instance=p, x_nm=x, y_nm=t(c.height_nm) if mirror else t(0), width_nm=c.width_nm,
                         height_nm=c.height_nm, mirror_y=mirror)
                    for p, (c, x, t, m, mirror) in placed.items()]
    v.wordlines = wordlines
    v.regions = region
    v.wlen_tracks = wlen_tracks
    return v
