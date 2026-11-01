"""Upper-metal supply grid of a complete compact macro (V2.2.8 post-layout fix).

Post-layout transients with the extracted VDD/VSS networks showed 150-950 ohm
from the supply pins to the controller, decoder, wordline drivers and array:
the bank feeds every block through minimum-width M4-M6 bars and single
connection points. This pass adds VDD straps on M8 and VSS straps on M9 (M7
for the top VSS strap, M3 in the controller gap) after all routing, so it never
constrains the controller router:

* a vertical strap pair at the right edge, joined to the root M6 trunks and to
  the right core's M4 supply trunks;
* a vertical strap pair over the left trunks, joined to the left M6 trunks
  and to the left core's M4 supply trunks;
* a horizontal pair across the top of the macro joining both sides;
* the decoder supply spines, fed at their top ends from the top straps;
* a vertical M3 pair in the gap between the controller and the right half, onto
  which every controller row rail exits at its right end through a row of
  via2 (the enable buffers otherwise share one rail's left exit), and supply
  pins at the bottom of that pair and at both ends of the left straps;
* in the decoder, facing band VSS rails merged into 270 nm M1 buses and the
  top and bottom bands' single rails widened under M2 buses that join the
  VSS spine;
* over each half-array, an M4 VSS mesh: a strap every fourth column with a
  via3 on every row-edge M3 VSS rail, stacked to the top M7 VSS strap and,
  in the left half, to the left M9 VSS strap.

Every via, pad and strap is checked against all existing metal and cut
shapes with the larger of the rule spacing and a margin; a blocked site is
skipped and counted, never forced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lib import pya, u

NET_LAYER = {'VDD': 8, 'VSS': 9}
# Square pad size on each metal for a single-cut stack (nm), and the cut size.
PAD = {1: (70, 140), 2: (70, 140), 3: (70, 140), 4: (140, 140), 5: (145, 145), 6: (145, 145),
       7: (410, 410), 8: (410, 410), 9: (800, 800)}
CUT = {1: 70, 2: 70, 3: 70, 4: 140, 5: 140, 6: 140, 7: 400, 8: 400}
# Spacing used against foreign metal: rule spacing, at least as large as any
# wide-metal rule triggered by the grid's own shapes (straps stay <= 0.5/0.8 um).
SPACE = {1: 70, 2: 70, 3: 70, 4: 140, 5: 140, 6: 140, 7: 500, 8: 500, 9: 900}
CUT_SPACE = {1: 80, 2: 80, 3: 80, 4: 160, 5: 160, 6: 160, 7: 440, 8: 440}
STRAP = {8: 500., 9: 800.}
TOP_STRAP = 500.
STEP = 3200.                     # stack pitch per net along a trunk; the other net sits half way
GAP_SPACE = 270.                 # M3 spacing of the 480 nm gap straps (wider than 0.27 um class)
MESH_PITCH = 4                   # bitcell VSS mesh: one M4 strap per four columns


@dataclass
class Grid:
    b: object
    own: dict = field(default_factory=dict)      # (net, layer) -> Region of grid metal
    report: dict = field(default_factory=lambda: dict(stacks={}, skipped={}, straps=[]))

    def local(self, layer: str, x0: float, y0: float, x1: float, y1: float) -> pya.Region:
        """Merged shapes of `layer` touching a window (spatially indexed cell query)."""
        box = pya.Box(u(round(x0 / 2.5) * 2.5), u(round(y0 / 2.5) * 2.5),
                      u(round(x1 / 2.5) * 2.5), u(round(y1 / 2.5) * 2.5))
        return pya.Region(self.b.cell.begin_shapes_rec_touching(self.b.li(layer), box)).merged()

    def at(self, layer: str, x: float, y: float) -> pya.Region:
        """The `layer` metal under a point, from a small window around it."""
        probe = self.box(x - 2.5, y - 2.5, x + 2.5, y + 2.5)
        return self.local(layer, x - 400., y - 400., x + 400., y + 400.).interacting(probe)

    @staticmethod
    def box(x0, y0, x1, y1):
        return pya.Region(pya.Box(u(x0), u(y0), u(x1), u(y1)))

    def clear(self, layer: str, shape: pya.Region, net: str, target: pya.Region | None = None,
              space: float | None = None) -> bool:
        k = int(layer[1:])
        if space is None:
            space = SPACE[k] if layer.startswith('m') else CUT_SPACE[k]
        halo = shape.sized(u(space))
        nm = self.b.ly.dbu * 1000
        bb = halo.bbox()
        near = self.local(layer, bb.left * nm, bb.bottom * nm, bb.right * nm, bb.top * nm)
        mine = self.own.get((net, layer), pya.Region())
        allowed = mine + (target if target is not None else pya.Region())
        foreign = near - near.interacting(allowed) if not allowed.is_empty() else near
        if not (halo & foreign).is_empty():
            return False
        # Same-net metal is exempt only where it merges with the new shape;
        # a near miss is still a spacing violation.
        own = self.own.get((net, layer), pya.Region())
        return (halo & own.not_interacting(shape)).is_empty()

    def draw(self, layer: str, shape: pya.Region, net: str | None = None):
        for polygon in shape.each():
            self.b.cell.shapes(self.b.li(layer)).insert(polygon)
        if net is not None:
            self.own[(net, layer)] = (self.own.get((net, layer), pya.Region()) + shape).merged()

    def stack(self, net: str, x: float, y: float, low: int, high: int, xmax: float | None = None) -> bool:
        """Single-cut stack from metal `low` (existing net metal at the point) to `high`.

        With `xmax`, pads wider than the landing shift left so no new metal
        passes the macro edge; every cut stays inside both of its pads.
        """
        x, y = round(x / 5) * 5., round(y / 5) * 5.
        base = self.at(f'm{low}', x, y)
        if base.is_empty():
            return False

        def centre(width):
            return x if xmax is None else min(x, round((xmax - width / 2) / 2.5) * 2.5)

        shapes, below = [], None
        self.why = None
        for k in range(low, high + 1):
            w, h = PAD[k]
            px = centre(w)
            pad = self.box(px - w / 2, y - h / 2, px + w / 2, y + h / 2)
            if k > low:
                if not self.clear(f'm{k}', pad, net) or not (below - pad).is_empty():
                    self.why = f'm{k}'
                    return False
                shapes.append((f'm{k}', pad))
            if k < high:
                c = CUT[k]
                upper = PAD[k + 1][0]
                # The cut sits as far right as both its pads allow (2.5 nm enclosure).
                cxk = min(x, centre(upper) + upper / 2 - c / 2 - 2.5)
                if k == low:
                    cxk = x if xmax is None or x + c / 2 <= xmax else cxk
                cut = self.box(cxk - c / 2, y - c / 2, cxk + c / 2, y + c / 2)
                landing = base if k == low else pad
                if not (cut - landing).is_empty() or not self.clear(f'v{k}', cut, net):
                    self.why = f'v{k}'
                    return False
                shapes.append((f'v{k}', cut))
                below = cut
        for layer, shape in shapes:
            self.draw(layer, shape, net if layer.startswith('m') else None)
        self.report['stacks'][net] = self.report['stacks'].get(net, 0) + 1
        return True

    def strap(self, net: str, x0: float, y0: float, x1: float, y1: float, layer: str | None = None,
              space: float | None = None) -> bool:
        layer = layer or f'm{NET_LAYER[net]}'
        shape = self.box(x0, y0, x1, y1)
        if not self.clear(layer, shape, net, space=space):
            return False
        self.draw(layer, shape, net)
        self.report['straps'].append(dict(net=net, layer=layer, box_nm=[x0, y0, x1, y1]))
        return True

    def jumper(self, net: str, x0: float, x1: float, y: float) -> bool:
        """Horizontal piece on the net's strap layer joining a stack pad to a strap."""
        k = NET_LAYER[net]
        half = PAD[k][1] / 2
        shape = self.box(min(x0, x1), y - half, max(x0, x1), y + half)
        if not self.clear(f'm{k}', shape, net):
            return False
        self.draw(f'm{k}', shape, net)
        return True

    def tie(self, net: str, x_from: float, x_to: float, y: float) -> bool:
        """M3 jumper from an M2 line at x_from (via2) to an M4 line at x_to (via3)."""
        y = round(y / 5) * 5.
        spine = self.at('m2', x_from, y)
        trunk = self.at('m4', x_to, y)
        if spine.is_empty() or trunk.is_empty():
            return False
        lo, hi = min(x_from, x_to), max(x_from, x_to)
        m3 = (self.box(lo - 35, y - 35, hi + 35, y + 35) + self.box(x_from - 35, y - 70, x_from + 35, y + 70)
              + self.box(x_to - 35, y - 70, x_to + 35, y + 70)).merged()
        v2 = self.box(x_from - 35, y - 35, x_from + 35, y + 35)
        v3 = self.box(x_to - 35, y - 35, x_to + 35, y + 35)
        pad4 = self.box(x_to - 70, y - 70, x_to + 70, y + 70)
        if (not self.clear('m3', m3, net) or not self.clear('v2', v2, net) or not self.clear('v3', v3, net)
                or not (v2 - spine).is_empty() or not (pad4 - trunk).is_empty()):
            return False
        for layer, shape in (('m3', m3), ('v2', v2), ('v3', v3)):
            self.draw(layer, shape, net if layer == 'm3' else None)
        self.report['stacks'][f'{net} spine ties'] = self.report['stacks'].get(f'{net} spine ties', 0) + 1
        return True

    def bridge(self, net: str, x_from: float, y: float, layer_from: int, x_to: float, layer_to: int,
               jump: int) -> bool:
        """Existing net metal (x_from, layer_from) -> jumper on `jump` -> existing (x_to, layer_to)."""
        y = round(y / 5) * 5.
        bases = {}
        for x, k in ((x_from, layer_from), (x_to, layer_to)):
            bases[x] = self.at(f'm{k}', x, y)
            if bases[x].is_empty():
                return False
        shapes = []

        def column(x, k0, k1):
            # Cuts between k0 and k1 at x; pads on every layer strictly between,
            # plus the jump layer end pad. The end at the existing metal is its base.
            lo, hi = min(k0, k1), max(k0, k1)
            for k in range(lo, hi):
                c = CUT[k]
                shapes.append((f'v{k}', self.box(x - c / 2, y - c / 2, x + c / 2, y + c / 2)))
            for k in range(lo, hi + 1):
                if k == k0:
                    continue
                w, h = PAD[k]
                shapes.append((f'm{k}', self.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)))

        column(x_from, layer_from, jump)
        column(x_to, layer_to, jump)
        half = {2: 35, 3: 35, 4: 70, 5: 72.5, 6: 72.5, 9: 400.}[jump]
        shapes.append((f'm{jump}', self.box(min(x_from, x_to) - half, y - half, max(x_from, x_to) + half, y + half)))
        targets = {f'm{layer_from}': bases[x_from], f'm{layer_to}': bases[x_to]}
        merged = {}
        for layer, shape in shapes:
            merged[layer] = (merged.get(layer, pya.Region()) + shape).merged()
        self.why = None
        for layer, shape in merged.items():
            if layer.startswith('m') and layer in targets and not (shape - targets[layer]).is_empty() \
                    and not self.clear(layer, shape - targets[layer], net, targets[layer]):
                self.why = layer
                return False
            if layer.startswith('m') and layer not in targets and not self.clear(layer, shape, net):
                self.why = layer
                return False
            if layer.startswith('v') and not self.clear(layer, shape, net):
                self.why = layer
                return False
        for layer, shape in merged.items():
            self.draw(layer, shape, net if layer.startswith('m') else None)
        key = f'{net} bridges'
        self.report['stacks'][key] = self.report['stacks'].get(key, 0) + 1
        return True

    def bridge_jog(self, net: str, x_from: float, y: float, x_to: float, y_to: float) -> bool:
        """An M2 rail end -> M2 run to x_to -> M2 jog to y_to -> stack M2..M6 onto an M6 trunk.

        The kit charges 88 ohm per minimum via2 and nothing above via2, so the
        M2-M3 step uses two cuts where they fit.
        """
        return self._bridge_jog(net, x_from, y, x_to, y_to, 2) or self._bridge_jog(net, x_from, y, x_to, y_to, 1)

    def _bridge_jog(self, net: str, x_from: float, y: float, x_to: float, y_to: float, cuts: int) -> bool:
        y, y_to = round(y / 5) * 5., round(y_to / 5) * 5.
        rail = self.at('m2', x_from, y)
        trunk = self.at('m6', x_to, y_to)
        if rail.is_empty() or trunk.is_empty():
            return False
        reach = 70. + 75. * (cuts - 1)           # two via2 cuts 150 nm apart (80 nm space)
        m2 = (self.box(min(x_from, x_to) - 35, y - 35, max(x_from, x_to) + 35, y + 35)
              + self.box(x_to - 35, min(y, y_to) - 35, x_to + 35, max(y, y_to) + 35)
              + self.box(x_to - 35, y_to - reach, x_to + 35, y_to + reach)).merged()
        shapes = [('m2', m2 - rail),
                  ('m3', self.box(x_to - 35, y_to - reach, x_to + 35, y_to + reach))]
        for dy in ((-75., 75.) if cuts == 2 else (0.,)):
            shapes.append(('v2', self.box(x_to - 35, y_to + dy - 35, x_to + 35, y_to + dy + 35)))
        for k in (3, 4, 5):
            c = CUT[k]
            shapes.append((f'v{k}', self.box(x_to - c / 2, y_to - c / 2, x_to + c / 2, y_to + c / 2)))
        for k in (4, 5):
            w, h = PAD[k]
            shapes.append((f'm{k}', self.box(x_to - w / 2, y_to - h / 2, x_to + w / 2, y_to + h / 2)))
        for layer, shape in shapes:
            if not self.clear(layer, shape, net, rail if layer == 'm2' else None):
                self.why = layer
                return False
        for layer, shape in shapes:
            self.draw(layer, shape, net if layer.startswith('m') else None)
        key = f'{net} jog bridges ({cuts} via2)'
        self.report['stacks'][key] = self.report['stacks'].get(key, 0) + 1
        return True

    def rail_cuts(self, net: str, x_trunk: float, y: float, direction: float, extra: int = 2,
                  level: int = 2) -> int:
        """Extra via2 cuts where an M2 rail meets its M3 edge trunk.

        A short M3 stub runs along the rail from the trunk; each cut is
        150 nm from the previous one (80 nm space). Returns cuts added.
        """
        y = round(y / 2.5) * 2.5        # rail centres lie on the 2.5 nm grid
        # level 2: M2 rail -> M3 trunk through via2 (controller rows);
        # level 1: M1 rail -> M2 spine through via1 (decoder bands).
        rail_layer, stub_layer, cut_layer = f'm{level}', f'm{level + 1}', f'v{level}'
        rail = self.at(rail_layer, x_trunk + direction * 150., y)
        trunk = self.at(stub_layer, x_trunk, y)
        if rail.is_empty() or trunk.is_empty():
            return 0
        added = 0
        for k in range(1, extra + 1):
            x = x_trunk + direction * 150. * k
            # Metal3.3: M3 extends 35 nm past the cut's far edge (and
            # overlaps the trunk), so the cut has its two opposite extensions.
            far = x + direction * 70.
            stub = self.box(min(x_trunk - 35., far), y - 35, max(x_trunk + 35., far), y + 35)
            cut = self.box(x - 35, y - 35, x + 35, y + 35)
            if not (cut - rail).is_empty() or not self.clear(stub_layer, stub - trunk, net, trunk) \
                    or not self.clear(cut_layer, cut, net):
                break
            self.draw(stub_layer, stub, net)
            self.draw(cut_layer, cut)
            added += 1
        if added:
            key = f'{net} rail {cut_layer} cuts added'
            self.report['stacks'][key] = self.report['stacks'].get(key, 0) + added
        return added

    def gap_exit(self, net: str, x_rail: float, y: float, x_cut: float, cols: int, row: float) -> bool:
        """A controller M2 rail's right end -> M2 run into the gap -> two rows of `cols`
        via2 (at y and y + row) onto the net's M3 gap strap.

        The kit charges 88 ohm per minimum via2, so the M2-M3 step is an array.
        """
        y = round(y / 2.5) * 2.5
        rail = self.at('m2', x_rail, y)
        if rail.is_empty():
            return False
        x_end = rail.bbox().right * self.b.ly.dbu * 1000
        xs = [x_cut + (i - (cols - 1) / 2) * 150. for i in range(cols)]
        ys = sorted((y, y + row))
        m2 = (self.box(x_end - 70., y - 35, xs[-1] + 70., y + 35)
              + self.box(xs[0] - 70., ys[0] - 70., xs[-1] + 70., ys[-1] + 70.)).merged()
        shapes = [('m2', m2 - rail)] + [('v2', self.box(x - 35, v - 35, x + 35, v + 35)) for x in xs for v in ys]
        self.why = None
        for layer, shape in shapes:
            if not self.clear(layer, shape, net, rail if layer == 'm2' else None):
                self.why = layer
                return False
        for layer, shape in shapes:
            self.draw(layer, shape, net if layer.startswith('m') else None)
        key = f'{net} right exits ({cols * 2} via2)'
        self.report['stacks'][key] = self.report['stacks'].get(key, 0) + 1
        return True

    def edge_pin(self, net: str, layer: str, x0: float, x1: float, y_from: float, y_edge: float,
                 side: str) -> bool:
        """Continue a vertical strap from y_from to the macro edge and record a supply pin there."""
        lo, hi = sorted((y_from, y_edge))
        if not self.strap(net, x0, lo, x1, hi, layer):
            self.skip(f'{side} {net} pin')
            return False
        inset = min(250., (x1 - x0) / 2)
        y = y_edge + inset if side == 'bottom' else y_edge - inset
        self.report.setdefault('pins', []).append(dict(net=net, side=side, x_nm=(x0 + x1) / 2, y_nm=y, layer=layer))
        return True

    def fill(self, net: str, layer: str, x0: float, x1: float, y0: float, y1: float, space: float,
             anchors: list, step: float = 200.) -> float:
        """Draw `layer` over y0..y1 along x0..x1 wherever a `step` chunk is clear of foreign
        metal by `space`. Only metal connected to an anchor box (the net's own rail or
        spine) is exempt. Returns the filled length (nm)."""
        filled = 0.
        x = x0
        own = pya.Region()
        for a in anchors:
            own += self.box(*a)
        while x < x1:
            xe = min(x + step, x1)
            chunk = self.box(x, y0, xe, y1)
            near = self.local(layer, x - space - 100., y0 - space - 100., xe + space + 100., y1 + space + 100.)
            if self.clear(layer, chunk, net, near.interacting(own), space):
                self.draw(layer, chunk, net)
                filled += xe - x
            x = xe
        return filled

    def stitch(self, net: str, x0: float, x1: float, y: float, pitch: float = 400.) -> int:
        """via1 cuts at y along x0..x1 where M1 and M2 of the net both cover the cut."""
        y = round(y / 2.5) * 2.5
        added = 0
        x = x0
        while x <= x1:
            cut = self.box(x - 35, y - 35, x + 35, y + 35)
            if ((cut - self.at('m1', x, y)).is_empty() and (cut - self.at('m2', x, y)).is_empty()
                    and self.clear('v1', cut, net)):
                self.draw('v1', cut)
                added += 1
            x += pitch
        return added

    def skip(self, what: str):
        self.report['skipped'][what] = self.report['skipped'].get(what, 0) + 1


def _ys(lo: float, hi: float, step: float = STEP):
    n = max(1, int((hi - lo) // step))
    return [round((lo + (hi - lo) * (i + .5) / n) / 5) * 5. for i in range(n)]


def add_grid(b, bm: dict, left_trunks: dict, root: dict, macro_box: tuple, frame_box: tuple) -> dict:
    """Draw the supply grid; returns a report of straps, stacks and skipped sites."""
    g = Grid(b)
    x_left, y_bottom, x_right, y_top = macro_box
    cx, array_y = bm['array_origin_nm']
    core = bm['core_geometry']
    nc = bm['cols'] // 2
    W = bm['core_pitch_nm']['column']
    core_y0, core_y1 = array_y - 460., array_y - 460. + core['height_nm']
    # Core supply trunks in each half's empty column slot (compact_array.VSS/VDD_TRUNK).
    core_trunks = {'left': {'VSS': nc * W + 300., 'VDD': nc * W + 800.},
                   'right': {'VSS': cx + nc * W + 300., 'VDD': cx + nc * W + 800.}}
    report = g.report

    # Vertical straps. Right: VSS on M9 centred on the right core's VSS trunk,
    # VDD on M8 between it and the root trunks. Left: mirrored outside the
    # left trunks, clear of the controller frame.
    rv = core_trunks['right']['VDD']
    right = {'VSS': (core_trunks['right']['VSS'] - 400., core_trunks['right']['VSS'] + 400.),
             'VDD': (root['VDD'] - 205. - 500. - 500., root['VDD'] - 205. - 500.)}
    lv = left_trunks['VSS']
    left = {'VSS': (lv - 2700., lv - 1900.), 'VDD': (lv - 1300., lv - 800.)}
    # Top straps: the highest clear band below the macro top. A vertical M9
    # riser crosses the decoder, so the top VSS strap is M7 (clear above the
    # periphery's M7 feedback returns), at least 1 um below the VDD strap.
    top_layer = {'VDD': 8, 'VSS': 7}
    top_y = {}
    for net in ('VDD', 'VSS'):
        layer, half = f'm{top_layer[net]}', TOP_STRAP / 2
        ceiling = y_top - 300. - half if net == 'VDD' else top_y['VDD'] - TOP_STRAP - 1000.
        for k in range(120):
            y = round((ceiling - 100. * k) / 5) * 5.
            if g.clear(layer, g.box(left[net][0], y - half, right[net][1], y + half), net):
                top_y[net] = y
                break
        else:
            y = y_top - 300. - half
            band = g.local(layer, left[net][0], y - 8000., right[net][1], y + half)
            nm = g.b.ly.dbu * 1000
            blockers = [[round(v * nm) for v in (q.bbox().left, q.bbox().bottom, q.bbox().right, q.bbox().top)]
                        for q in band.each()][:6]
            raise ValueError(f'No clear top {net} band on {layer}; blockers {blockers}')
    report['top_strap_y_nm'] = top_y
    for side, xs in (('right', right), ('left', left)):
        for net, (x0, x1) in xs.items():
            if not g.strap(net, x0, y_bottom + 200., x1, top_y['VDD'] + TOP_STRAP / 2):
                raise ValueError(f'{side} {net} strap is blocked; the supply grid needs a clear column')
    # Supply pins at both ends of the left straps: otherwise the decoder,
    # controller and left half return through the top strap to the right pins.
    for net, (x0, x1) in left.items():
        layer = f'm{NET_LAYER[net]}'
        g.edge_pin(net, layer, x0, x1, y_bottom + 300., y_bottom, 'bottom')
        g.edge_pin(net, layer, x0, x1, top_y['VDD'] + TOP_STRAP / 2 - 100., y_top, 'top')
    half = TOP_STRAP / 2
    g.strap('VDD', left['VDD'][0], top_y['VDD'] - half, right['VDD'][1], top_y['VDD'] + half)
    g.strap('VSS', left['VSS'][0], top_y['VSS'] - half, right['VSS'][1], top_y['VSS'] + half, layer='m7')
    # The M7 VSS strap joins each M9 VSS strap through M7-M9 stacks.
    for x in (sum(left['VSS']) / 2, sum(right['VSS']) / 2):
        joined = sum(g.stack('VSS', x, top_y['VSS'] + dy, 7, 9) for dy in (-500., 0., 500.))
        if not joined:
            raise ValueError('Top VSS strap cannot join a vertical VSS strap')

    # Joins: (trunk x, y range, landing metal) -> strap x edge.
    def join(net, x, lo, hi, low, strap_edge, label, xmax=None):
        for y in _ys(lo, hi):
            for dy in (0., 200., -200., 400., -400.):
                if g.stack(net, x, y + dy, low, NET_LAYER[net], xmax):
                    if g.jumper(net, x, strap_edge, y + dy):
                        break
            else:
                g.skip(label)

    # Right root M6 trunks (bottom to top edge); VDD and VSS sites interleave.
    join('VDD', root['VDD'], y_bottom + 300., y_top - 300., 6, right['VDD'][1], 'right root VDD', x_right)
    join('VSS', root['VSS'], y_bottom + 300. + STEP / 2, y_top - 300., 6, right['VSS'][1], 'right root VSS',
         x_right)
    # Core M4 trunks in both halves.
    join('VSS', core_trunks['right']['VSS'], core_y0 + 300., core_y1 - 300., 4, right['VSS'][1], 'right core VSS')
    join('VDD', rv, core_y0 + 300. + STEP / 2, core_y1 - 300., 4, right['VDD'][0], 'right core VDD')
    join('VSS', core_trunks['left']['VSS'], core_y0 + 300., core_y1 - 300., 4, left['VSS'][1], 'left core VSS')
    join('VDD', core_trunks['left']['VDD'], core_y0 + 300. + STEP / 2, core_y1 - 300., 4, left['VDD'][1],
         'left core VDD')
    # Decoder M2 supply spines (compact_decoder: x = 50/200 from the decoder
    # origin) tie to the left core's M4 trunks in the M3 gaps between rows.
    dx, dy0, dx1, dy1 = bm['decoder_bbox_nm']
    spines = {'VDD': dx + 50., 'VSS': dx + 200.}
    H = bm['core_pitch_nm']['row']
    for k in range(int((dy1 - dy0) // (H / 4))):
        y = dy0 + k * H / 4
        order = ('VSS', 'VDD') if int((y - dy0) // H) % 2 else ('VDD', 'VSS')
        for net in order:
            if g.tie(net, spines[net], core_trunks['left'][net], y):
                break
    for net in ('VDD', 'VSS'):
        if not g.report['stacks'].get(f'{net} spine ties'):
            g.skip(f'decoder {net} spine')
    # Each band's VSS M1 rail meets the M2 spine through one via1 (88 ohm in
    # the kit deck). Extra cuts on an M2 stub over the rail, inside the
    # 220 nm next to the rail that decoder gates keep free of M2. VDD rails
    # cannot: their stub would cross the VSS spine.
    xs = spines['VSS']
    rail_ys = []
    for q in g.local('v1', xs - 50., dy0, xs + 50., dy1).each():
        bb = q.bbox()
        rail_ys.append(round((bb.top + bb.bottom) / 2 * g.b.ly.dbu * 1000 / 2.5) * 2.5)
        g.rail_cuts('VSS', xs, (bb.top + bb.bottom) / 2 * g.b.ly.dbu * 1000, 1., extra=4, level=1)
    # Band VSS rails are 70 nm M1 from the wordline drivers at the decoder's right
    # end to the spine (20-38 um); a falling wordline lifted its band's held-low
    # neighbours through them (128x128: WL125 0.165 V as WL127 fell). Facing
    # rails 200 nm apart at band boundaries merge into one 270 nm bus (0.09 um
    # wide-metal spacing). The edge bands' single rails widen outward to 270 nm under
    # an M2 bus that joins the VSS spine, stitched by via1.
    rail_ys.sort()
    x_end = dx1 - 100.
    merged = sum(g.fill('VSS', 'm1', xs - 35., x_end, lo - 35., hi + 35., 90.,
                        [(xs, lo - 2.5, x_end, lo + 2.5), (xs, hi - 2.5, x_end, hi + 2.5)])
                 for lo, hi in zip(rail_ys, rail_ys[1:]) if abs(hi - lo - 200.) < 1.)
    report['decoder_vss_m1_merged_um'] = round(merged / 1000., 1)
    if rail_ys:
        # The edge bands' single rails (top and bottom of the decoder) widen away
        # from the decoder under an M2 bus. The bus keeps 90 nm from the band's M2
        # line over the rail (to 30-40 nm past it); a neck joins it to the spine.
        spine = [(xs - 2.5, rail_ys[0], xs + 2.5, rail_ys[-1])]
        for side, rail, sign in (('top', rail_ys[-1], 1.), ('bottom', rail_ys[0], -1.)):
            def band(a, b):
                return tuple(sorted((rail + sign * a, rail + sign * b)))
            widened = g.fill('VSS', 'm1', xs + 300., x_end, *band(35., 235.), 90., [(xs, rail - 2.5, x_end, rail + 2.5)])
            # The neck is minimum width (70 nm rule). At the top the bus starts flush
            # with the neck (no M2 notch); below the decoder it starts 90 nm from the
            # VDD spine (x - 150), inside the VSS spine that continues down there.
            # Below the decoder M2 on the rail reaches 35 nm under it, so that bus sits
            # 10 nm lower to keep its 90 nm halo clear and merge with the spine.
            near, far, cut = (120., 390., 177.5) if side == 'top' else (130., 400., 182.5)
            neck = g.fill('VSS', 'm2', xs - 35., xs + 35., *band(30., near + 10.), 70., spine)
            bus = g.fill('VSS', 'm2', xs - (35. if side == 'top' else 25.), x_end, *band(near, far), 90., spine)
            # The spine's merged M2 must reach into the bus, or the bus only shunts the rail.
            ylo, yhi = sorted((rail, rail + sign * far))
            b0, b1 = band(near, far)
            on_spine = g.local('m2', xs - 50., ylo, xs + 400., yhi).interacting(
                g.box(xs - 2.5, rail - 2.5, xs + 2.5, rail + 2.5))
            joined = not (on_spine & g.box(xs + 200., b0, xs + 300., b1)).is_empty()
            report[f'decoder_{side}_rail'] = dict(m1_widened_um=round(widened / 1000., 1), m2_bus_um=round(bus / 1000., 1),
                                                  spine_neck=bool(neck), bus_on_spine=joined,
                                                  via1=g.stitch('VSS', xs + 400., x_end - 100., rail + sign * cut))
    # Controller rows: VSS M2 rails join one M3 trunk at the frame's left
    # edge and VDD rails one M3 trunk at its right edge (time_control), each
    # with a single bank feed. Bridge both to the adjacent left M6 trunks.
    fx0, fy0, fx1, fy1 = frame_box
    # A strap pair in the gap between the controller frame and the right half,
    # on M3 (M2/M3 are free there; M4, M5 stubs, M7 and M10 cross it), from
    # supply pins at the macro bottom up to the core. Every controller rail
    # also leaves its right end there (below), so the enable buffers no longer
    # share one rail's left exit. 480 nm keeps them below the 0.5 um
    # wide-metal class; they keep its 0.27 um spacing.
    gap = {}
    if cx - fx1 >= 2000.:
        for net, (x0, x1) in {'VDD': (fx1 + 210., fx1 + 690.), 'VSS': (fx1 + 960., fx1 + 1440.)}.items():
            for k in range(200):
                y_hi = round((core_y0 - 300. - 100. * k) / 5) * 5.
                if y_hi >= fy0 + 2000. and g.strap(net, x0, y_bottom, x1, y_hi, 'm3', space=GAP_SPACE):
                    gap[net] = (x0 + x1) / 2
                    report.setdefault('pins', []).append(dict(net=net, side='bottom', x_nm=(x0 + x1) / 2,
                                                              y_nm=y_bottom + 250., layer='m3'))
                    break
            else:
                g.skip(f'gap {net} strap')
    m3 = g.local('m3', fx0, fy0, fx0 + 400., fy1) & g.box(fx0, fy0, fx0 + 400., fy1)
    trunks3 = [q.bbox() for q in m3.each() if q.bbox().height() > q.bbox().width() * 10]
    nm = g.b.ly.dbu * 1000
    if trunks3:
        t = max(trunks3, key=lambda q: q.height())
        xc = (t.left + t.right) / 2 * nm
        # VDD rails sit at fixed heights: bridge them first; VSS can tie
        # anywhere along its M3 trunk and fills the remaining gaps.
        rails = g.local('m2', xc - 100., fy0, xc + 2000., fy1) & g.box(xc - 100., fy0, xc + 2000., fy1)
        found = []
        for q in rails.each():
            bb = q.bbox()
            y = round((bb.top + bb.bottom) / 2 * nm / 2.5) * 2.5
            if bb.width() < bb.height() or bb.left * nm > xc + 100.:
                continue
            # A VSS rail's single via2 joins the M3 edge trunk; a VDD rail has none here.
            net = 'VSS' if not g.local('v2', xc - 50., y - 50., xc + 50., y + 50.).is_empty() else 'VDD'
            found.append((net, bb.left * nm + 40., y))
        # Each row rail reaches its edge trunk through one 88-ohm via2 (kit
        # value); the rail's own bridge, with doubled via2, bypasses it.
        # Extra cuts at each rail's own edge trunk: VSS at the left M3
        # trunk, VDD at the right one (time_control: x = width - 100).
        x_vdd = fx1 - 100.
        for net, _, y in found:
            if net == 'VSS':
                g.rail_cuts('VSS', xc, y, 1.)
            elif not g.local('v2', x_vdd - 50., y - 50., x_vdd + 50., y + 50.).is_empty():
                g.rail_cuts('VDD', x_vdd, y, -1.)
        # VSS rails first: their run to the farther VSS trunk crosses the
        # VDD stack column; VDD rails then jog around them where needed.
        for net in ('VSS', 'VDD'):
            for _, x_rail, y in [f for f in found if f[0] == net]:
                if not any(g.bridge_jog(net, x_rail, y, left_trunks[net], y + dy)
                           for dy in (0., 300., -300., 500., -500., 700., -700., 900., -900.)):
                    g.skip(f'controller {net} rail')
        last = None
        for k in range(int((t.top - t.bottom) * nm // 100)):
            y = t.bottom * nm + 100. + 100. * k
            if last is not None and y - last < 900.:
                continue
            if g.bridge('VSS', xc, y, 3, left_trunks['VSS'], 6, 4):
                last = y
        # Right exits: each rail's M2 continues into the gap and joins its M3
        # gap strap through a 3 x 2 via2 array; the second row points away from
        # the neighbouring rail of the other net, 190 nm away. VSS runs pass
        # under the VDD strap.
        for net, _, y in found:
            if net in gap and not any(g.gap_exit(net, fx1 - 300., y, gap[net], cols,
                                                 150. if net == 'VSS' else -150.) for cols in (3, 2)):
                g.skip(f'controller {net} right exit')
    else:
        g.skip('controller VSS trunk')
    # Left M6 trunks below the core.
    for net in ('VDD', 'VSS'):
        lx = left_trunks[net]
        join(net, lx, y_bottom + 300. + (STEP / 2 if net == 'VDD' else 0.), core_y0 - 300., 6, left[net][1],
             f'left trunk {net}')
    report['array_mesh'] = array_mesh(g, bm, core_y0, top_y['VSS'], left['VSS'])
    return report


def array_mesh(g: Grid, bm: dict, core_y0: float, top_vss_y: float, left_vss: tuple) -> dict:
    """M4 VSS straps over both half-arrays, tying every row-edge M3 VSS rail together.

    A cell row has VSS on 70 nm M3 rails along both row edges (compact_core),
    and each half-array joins them to its M4 trunk in the empty column slot
    at one end only (compact_array). A read sinks the whole row's current
    along that row's two rails: at 128x128 FF 1.1 V -40 C the far-end cells'
    ground rose about 0.35 V and the read flipped them. With a strap at every
    fourth column centre and a via3 on each rail, the row current spreads
    over every rail of the half-array. M4-M7 are otherwise empty over the
    data columns; the bitline drop pads at the bottom sit 250 nm to either
    side of a column centre. Straps also stack to the top M7 VSS strap where
    it crosses them, and left-half straps beside the left M9 VSS strap join
    it, because the left core's M4 trunk lies under the M6 wordline
    crossings and has no stack of its own.
    """
    cx, array_y = bm['array_origin_nm']
    rows, nc = bm['rows'], bm['cols'] // 2
    W, H = bm['core_pitch_nm']['column'], bm['core_pitch_nm']['row']
    yt = array_y + rows * H
    # compact_array.make_core VSS M3 rails: each well-strap row's rail 165 nm below
    # its top edge, every array row edge, and both edges of the RWL load row.
    rails = ([core_y0 + 295.] + [array_y + k * H for k in range(rows + 1)]
             + [yt + 295., yt + 460., yt + 460. + H])
    out = dict(pitch_columns=MESH_PITCH, straps=0, via3=0, m7_stacks=0, m9_joins=0, skipped={})
    counted = dict(g.report['stacks'])      # mesh stacks are reported under array_mesh only

    def skip(what):
        out['skipped'][what] = out['skipped'].get(what, 0) + 1

    for side, x0 in (('left', 0.), ('right', cx)):
        for c in range(1, nc, MESH_PITCH):
            # Column centre on the 5 nm grid that stack() snaps to.
            x = (x0 + c * W + W / 2) // 5 * 5.
            strap = g.box(x - 70., rails[0] - 70., x + 70., rails[-1] + 70.)
            if not g.clear('m4', strap, 'VSS'):
                skip(f'{side} strap')
                continue
            g.draw('m4', strap, 'VSS')
            out['straps'] += 1
            for y in rails:
                cut = g.box(x - 35., y - 35., x + 35., y + 35.)
                if (cut - g.at('m3', x, y)).is_empty() and g.clear('v3', cut, 'VSS'):
                    g.draw('v3', cut)
                    out['via3'] += 1
                else:
                    skip('via3')
            if not g.at('m7', x, top_vss_y).is_empty():
                if g.stack('VSS', x, top_vss_y, 4, 7):
                    out['m7_stacks'] += 1
                else:
                    skip('M7 stack')
            if side == 'left' and left_vss[0] - 1200. <= x <= left_vss[1] + 1200.:
                # M4-M9 stack whose M9 pad runs into the strap (checked as one piece).
                for y in _ys(array_y + 300., yt - 300.):
                    if any(g.bridge('VSS', x, y + dy, 4, sum(left_vss) / 2, 9, 9)
                           for dy in (0., 200., -200., 400., -400.)):
                        out['m9_joins'] += 1
                    else:
                        skip('M9 join')
    g.report['stacks'] = counted
    return out
