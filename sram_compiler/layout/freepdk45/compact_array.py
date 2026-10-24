"""Array, replica column and RWL loads at the compact point-symmetric pitch.

Odd columns and odd rows mirror, so neighbours share frame-edge contacts and
WL poly pads. Every array edge is a cell row's y=0 edge and meets a well-strap
row. The replica column sits one column slot past the array (the empty slot
holds the VSS/VDD M4 trunks), so the replica periphery takes the U role of
one more column pair. Passive replica wordlines tie to the VSS trunk; the
active replica row and the RWL load row join on an M4 riser right of the
replica column.
"""
from .components import Component, Instance, CellView
from .cell_flow import leaves
from .compact_core import make_cell, draw_strap, bitline_x, W, H, HS, Y_WL
from .lib import Builder

VSS_TRUNK = 300.      # M4 trunk x offsets inside the empty column slot
VDD_TRUNK = 800.
RWL_RISER = 300.      # past the replica column's right edge


def make_core(layout, source, specs, core_name, span=None):
    """span=(r0, r1) builds one half of a split bank (r0 a multiple of 8).

    The upper half (r1=rows) keeps the active replica cell and the RWL load
    row; the lower half exposes its bitlines at the top edge, which faces the
    centre periphery. Local y=0 is the bottom strap's lower edge.
    """
    rows, cols = source.rows, source.cols
    r0, r1 = span or (0, rows)
    upper = r1 == rows
    if r0 % 2 or (r1 - r0) % 2:
        raise ValueError('Compact core halves need even row counts')
    replica = f'Xsram_{rows}x1_replica_column'
    loads = [f'XRWL_LOAD_{c}' for c in range(cols - 1)] if upper else []
    names = [core_name, replica] + loads
    component = Component(
        f'COMPACT_CORE_{rows}X{cols}' if span is None else f'COMPACT_CORE_{rows}X{cols}_R{r0}_{r1}',
        tuple(['VDD', 'VSS', 'RBL', 'RBLB'] + (['RWL'] if upper else []) +
              [f'WL{r}' for r in range(r0, r1)] + [f'{p}{c}' for c in range(cols) for p in ('BL', 'BLB')]), (),
        tuple(Instance(n, tuple(source.instances[n]['pins'][p] for p in specs[n].ports), specs[n]) for n in names))
    b = Builder(layout, component.name)
    n = r1 - r0
    ya = HS                       # first array row
    yt = ya + n * H               # top strap
    yd = yt + HS                  # RWL load row (upper half)
    top = yd + H if upper else yt + HS
    rx = (cols + 1) * W           # replica column (odd slot: mirrored)
    xe = (cols + 2) * W
    cache, terminals, placements, fingers = {}, {}, [], []

    def put(path, c, kind, col, y, my):
        if kind not in cache:
            cache[kind] = make_cell(layout, c, kind)
        v = cache[kind]
        if v.component.identity != c.identity:
            raise ValueError('Mixed core cell definitions')
        mx = bool(col % 2)
        x = (col + 1) * W if mx else col * W
        oy = y + H if my else y
        b.place(v.builder, x, oy, mx=mx, my=my)
        terminals.update({path + '/' + k: (x - px if mx else x + px, oy - py if my else oy + py, l)
                          for k, (px, py, l) in v.device_pins.items()})
        placements.append(dict(instance=path, x_nm=x, y_nm=oy, mirror_x=mx, mirror_y=my))
        fingers.extend(dict(device=path + '/' + d.name, finger=i, width_nm=d.width_m * 1e9 / d.fingers,
                            length_nm=d.length_m * 1e9, model=d.model, nodes=d.nodes)
                       for d in c.devices for i in range(d.fingers))

    row_y = lambda r: ya + (r - r0) * H
    for path, c, m in leaves(specs[core_name], path=core_name):
        mapped = source.instances[path]['pins']
        row, col = int(mapped['WL'][2:]), int(mapped['BL'][2:])
        if r0 <= row < r1:
            put(path, c, 'cell', col, row_y(row), bool(row % 2))
    for r, (path, c, m) in enumerate(leaves(specs[replica], path=replica)):
        if r0 <= r < r1:
            put(path, c, 'replica' if r == rows - 1 else 'passive', cols + 1, row_y(r), bool(r % 2))
    for col, name in enumerate(loads):
        put(name, specs[name], 'dummy', col, yd, False)

    # Straps: one per array edge, across data, empty slot and replica.
    strap = Builder(layout, 'CMP_STRAP')
    draw_strap(strap)
    channel = Builder(layout, 'CMP_STRAP_CHANNEL')
    draw_strap(channel, channel=True)
    for y0 in (0., yt):
        for col in list(range(cols)) + [cols + 1]:
            mx = bool(col % 2)
            b.place(strap, (col + 1) * W if mx else col * W, y0, mx=mx)
        b.place(channel, cols * W, y0)
        for x0 in (-100., xe):     # p-taps on the outer column edges
            b.rect('pwell', x0, y0, x0 + 100, y0 + HS)
            b.rect('pplus', x0, y0 + 92.5, x0 + 100, y0 + HS - 92.5)
    # Empty column slot: p-well under the array rows, rails across it.
    b.rect('pwell', cols * W, ya, rx, yt)
    if upper:
        b.rect('pwell', cols * W, yd, rx, yd + H)
    edges = [ya + k * H for k in range(n + 1)] + ([yd, yd + H] if upper else [])
    vss = [y + HS - 165 for y in (0., yt)] + edges
    for y in vss:
        # From the last load column, so the RWL load row's rails reach too.
        b.rect('m3', (cols - 1) * W, y - 35, rx, y + 35)
    vx, dx = cols * W + VSS_TRUNK, cols * W + VDD_TRUNK
    b.rect('m4', vx - 70, 0, vx + 70, top + 70)
    b.rect('m4', dx - 70, 0, dx + 70, top + 70)
    for y in vss:
        b.sq('v3', vx, y, 65)
    for y0 in (0., yt):
        # Strap VDD rail -> M4: via1 on the rail, M2 stub down to a via2 that
        # clears the strap's VSS M3 rail and the array-edge VSS rail.
        b.sq('v1', dx, y0 + 165, 65)
        b.rect('m2', dx - 35, y0 + 77.5, dx + 35, y0 + 232.5)
        b.sq('v2', dx, y0 + 145, 65)
        b.rect('m3', dx - 67.5, y0 + 110, dx + 67.5, y0 + 180)
        b.sq('v3', dx, y0 + 145, 65)
    # Passive replica wordlines join the VSS trunk; the active one rises to RWL.
    for r in range(r0, r1):
        y = row_y(r) + Y_WL
        if r < rows - 1:
            b.rect('m3', vx - 70, y - 35, rx, y + 35)
            b.sq('v3', vx, y, 65)
    if upper:
        yr, yl, wx = row_y(rows - 1) + Y_WL, yd + Y_WL, xe + RWL_RISER
        b.rect('m3', xe, yr - 35, wx + 70, yr + 35)
        b.rect('m3', (cols - 1) * W, yl - 35, wx + 70, yl + 35)
        for y in (yr, yl):
            b.sq('v3', wx, y, 65)
        b.rect('m4', wx - 70, yr - 70, wx + 70, yl + 70)

    # Bitlines and supplies face the periphery: the lower half's top edge.
    edge = 0. if upper else top
    pins = {'VDD': (dx, edge, 'm4'), 'VSS': (vx, edge, 'm4')}
    rbl, rblb = bitline_x(cols + 1)
    pins.update({'RBL': (rx + rbl, edge, 'm2'), 'RBLB': (rx + rblb, edge, 'm2')})
    if upper:
        pins['RWL'] = (0., yd + Y_WL, 'm3')
    pins.update({f'WL{r}': (0., row_y(r) + Y_WL, 'm3') for r in range(r0, r1)})
    for c in range(cols):
        xbl, xblb = bitline_x(c)
        pins.update({f'BL{c}': (c * W + xbl, edge, 'm2'), f'BLB{c}': (c * W + xblb, edge, 'm2')})
    for net, (x, y, l) in pins.items():
        b.label(l, net, x, y)
    v = CellView(component, b, xe, top, pins, terminals, [], [], [])
    v.placements = placements
    v.array_y = (ya, yt)
    v.rwl_riser = (xe + RWL_RISER, yd + Y_WL) if upper else None
    if span is not None:
        v.physical_fingers = fingers
    return v
