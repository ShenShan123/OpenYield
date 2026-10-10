"""Upper-metal supply feeds for the complete physical calibration macro."""
from .lib import pya, u


def add_power_grid(builder, paths, macro_bottom, macro_x, controller_top, external,
                   memory_taps=None, tap_fractions=None):
    """Join real M3 supply paths to 2 um M7 buses through M6 taps.

    M6 stripes are 280 nm wide and separated by at least 560 nm. New via
    stacks enter only where M4/M5 are empty, so signal routing is preserved.
    All feeds remain physical interconnect with extracted resistance.
    """
    from . import top as memory
    buses = {'VDD': -2800., 'VSS': 1400.}
    bottom, top = -1680., macro_bottom + 1120.
    for net, x in buses.items():
        builder.rect('m7', x-1000, bottom, x+1000, top)
    all_regions = {layer:pya.Region(builder.cell.begin_shapes_rec(builder.li(layer)))
                   for layer in ('m4','m5')}
    points = {'VDD':set(), 'VSS':set()}
    for net, a, b in paths:
        if net in points and a[2] == b[2] == 0:
            points[net].update(((a[0]*280., a[1]*280.), (b[0]*280., b[1]*280.)))
    used_y, taps = [], []

    def stack(x, y):
        builder.sq('m3', x, y, 145)
        builder.sq('m4', x, y, 145)
        builder.sq('m5', x, y, 145)
        for level in (3,4,5):
            builder.via(level, x, y)

    def feed(net, x, y, *, start_y=None):
        start_y = y if start_y is None else start_y
        stack(x, start_y)
        if start_y != y:
            builder.rect('m6', x-140, min(y,start_y)-140, x+140, max(y,start_y)+140)
        bx = buses[net]
        builder.rect('m6', min(bx,x)-140, y-140, max(bx,x)+140, y+140)
        builder.via(6, bx, y)
        used_y.append(y)
        taps.append(dict(net=net, x_nm=x, y_nm=y, start_y_nm=start_y))

    for net, y in [('VDD',-1120.), ('VSS',-560.)]:
        x, sy, _ = external[net]
        feed(net, x, y, start_y=sy)
    # The two memory trunks are 200 nm apart, so their stacks are staggered.
    if memory_taps is None:
        feed('VSS', macro_x+memory.rp.X_VSS_TRUNK, macro_bottom+280.)
        feed('VDD', macro_x+memory.rp.X_VDD_TRUNK, macro_bottom+840.)
    else:
        for net, (x, y) in memory_taps.items():
            feed(net, x, y)
    fractions=(.15,.35,.55,.75) if tap_fractions is None else tuple(tap_fractions)
    if not fractions or any(not 0<f<1 for f in fractions):
        raise ValueError('Supply tap fractions must lie strictly inside the controller height')
    for fraction in fractions:
        for net, offset in [('VDD',0.), ('VSS',1120.)]:
            target = round(controller_top*fraction/280)*280 + offset
            candidates = sorted(points[net],key=lambda p:(abs(p[1]-target),p[0]))
            found = None
            for x,y in candidates:
                if not 1120 < y < controller_top-7000:
                    continue
                if any(abs(y-old)<560 for old in used_y):
                    continue
                guard=pya.Region(pya.Box(u(x-212.5),u(y-212.5),u(x+212.5),u(y+212.5)))
                if any(not (region & guard).is_empty() for region in all_regions.values()):
                    continue
                found=(x,y);break
            if found is None:
                raise ValueError(f'No safe upper supply tap for {net} near row {target}')
            feed(net,*found)
    return dict(metal7_width_nm=2000.,metal6_width_nm=280.,buses_nm=buses,
                taps=taps,minimum_tap_row_spacing_nm=560.)
