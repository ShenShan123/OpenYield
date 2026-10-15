"""Layout helpers for the FreePDK45 8x8 SRAM macro (klayout.db).

All coordinates are nanometres on the 2.5 nm manufacturing grid. The GDS
database unit is 0.5 nm, so every coordinate maps to an integer.
"""
try:
    import pya
except ImportError:
    import klayout.db as pya

DBU_UM = 0.0005  # um per database unit

# GDS layer numbers used by the FreePDK45 KLayout DRC/LVS decks.
LAYERS = {
    "active": (1, 0), "pwell": (2, 0), "nwell": (3, 0),
    "nplus": (4, 0), "pplus": (5, 0), "vtg": (6, 0),
    "poly": (9, 0), "cont": (10, 0),
    "m1": (11, 0), "v1": (12, 0), "m2": (13, 0), "v2": (14, 0),
    "m3": (15, 0), "v3": (16, 0), "m4": (17, 0), "v4": (18, 0),
    "m5": (19, 0), "v5": (20, 0), "m6": (21, 0),
    "v6": (22, 0), "m7": (23, 0),
    "v7": (24, 0), "m8": (25, 0),
    "v8": (26, 0), "m9": (27, 0),
    "v9": (28, 0), "m10": (29, 0),
    "bnd": (235, 0),
}

# Rule values (nm) from the FreePDK45 Calibre/KLayout decks.
CONT = 65.0      # contact size (exact)
VIA = 70.0       # via1..via3 size (KLayout deck requires exactly 70)
VIA4 = 140.0


def u(v):
    """nm -> database units; refuses off-grid values."""
    r = round(v / 2.5)
    if abs(v / 2.5 - r) > 1e-6:
        raise ValueError("off-grid coordinate %r nm" % v)
    return int(r) * 5


class Builder(object):
    """Thin wrapper around a pya.Cell taking nm coordinates."""

    def __init__(self, layout, name):
        self.ly = layout
        self.cell = layout.create_cell(name)
        self.name = name

    def li(self, layer):
        return self.ly.layer(*LAYERS[layer])

    def rect(self, layer, x1, y1, x2, y2):
        xa, xb = sorted((x1, x2))
        ya, yb = sorted((y1, y2))
        if xa == xb or ya == yb:
            raise ValueError("degenerate %s rect in %s" % (layer, self.name))
        self.cell.shapes(self.li(layer)).insert(pya.Box(u(xa), u(ya), u(xb), u(yb)))

    def polygon(self, layer, pts):
        self.cell.shapes(self.li(layer)).insert(
            pya.Polygon([pya.Point(u(x), u(y)) for x, y in pts]))

    def sq(self, layer, x, y, size):
        h = size / 2.0
        self.rect(layer, x - h, y - h, x + h, y + h)

    def cont(self, x, y):
        self.sq("cont", x, y, CONT)

    def via(self, n, x, y):
        self.sq("v%d" % n, x, y, VIA4 if n >= 4 else VIA)

    def label(self, layer, text, x, y):
        self.cell.shapes(self.li(layer)).insert(pya.Text(text, pya.Trans(u(x), u(y))))

    def place(self, child, x=0.0, y=0.0, mx=False, my=False):
        """Instance `child` (a Builder). mx mirrors x (x->-x), my mirrors y (y->-y),
        both together rotate by 180 degrees; then translate by (x, y)."""
        if mx and my:
            t = pya.Trans(pya.Trans.R180, u(x), u(y))
        elif mx:
            t = pya.Trans(pya.Trans.M90, u(x), u(y))
        elif my:
            t = pya.Trans(pya.Trans.M0, u(x), u(y))
        else:
            t = pya.Trans(pya.Trans.R0, u(x), u(y))
        self.cell.insert(pya.CellInstArray(child.cell.cell_index(), t))


class Sym(object):
    """Draws a shape and optionally its 180-degree rotated twin about the
    centre of a W x H frame (used for the point-symmetric bitcell)."""

    def __init__(self, b, w, h):
        self.b, self.w, self.h = b, w, h

    def rect(self, layer, x1, y1, x2, y2, twin=True):
        self.b.rect(layer, x1, y1, x2, y2)
        if twin:
            self.b.rect(layer, self.w - x1, self.h - y1, self.w - x2, self.h - y2)

    def polygon(self, layer, pts, twin=True):
        self.b.polygon(layer, pts)
        if twin:
            self.b.polygon(layer, [(self.w - x, self.h - y) for x, y in pts])

    def cont(self, x, y, twin=True):
        self.b.cont(x, y)
        if twin:
            self.b.cont(self.w - x, self.h - y)

    def via(self, n, x, y, twin=True):
        self.b.via(n, x, y)
        if twin:
            self.b.via(n, self.w - x, self.h - y)


def new_layout():
    ly = pya.Layout()
    ly.dbu = DBU_UM
    return ly
