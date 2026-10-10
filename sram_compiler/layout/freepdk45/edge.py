"""Array-edge cells: well-tap strip, column spacer and top edge.

TAP strip (W x 325, y in [-325, 0]): sits between two bottom-type cell
boundaries (VSS_L / VDD(PU_L) / BLB contacts on y = 0). It is mirror-symmetric
about y = -162.5. One P tap per p-well half (to VSS) and one N tap across the
n-well (to VDD); taps join the adjacent bitcell power metal1.
"""
from .lib import Builder
from .bitcell import W, D, X_VDD, X_BL

TAP_H = 325.0


XW = 347.5 + D / 2    # n-well edge (as in the bitcell)


def _column_wells(b, y1, y2, w=W):
    b.rect("pwell", 0, y1, XW, y2)
    b.rect("nwell", XW, y1, W - XW, y2)
    b.rect("pwell", W - XW, y1, w, y2)
    b.rect("vtg", 0, y1, w, y2)


def _array_implant(b, y1, y2):
    b.rect("nplus", 0, y1, XW, y2)
    b.rect("pplus", XW, y1, W - XW, y2)
    b.rect("nplus", W - XW, y1, W, y2)


def _tap_implant(b, y1, y2):
    b.rect("pplus", 0, y1, XW, y2)
    b.rect("nplus", XW, y1, W - XW, y2)
    b.rect("pplus", W - XW, y1, W, y2)


def make_tap(ly, name="ARRAY_TAP", bitlines=True, far="array"):
    """Tap strip. `bitlines` continues the BL/BLB metal3 through the strip.
    `far` says what faces the y = -325 side: "array" (bitcell / dummy row,
    power metal1 joined across) or "periphery" (column periphery: only the
    VSS metal3 at x = 0, VDD at the centre and the bit lines continue)."""
    b = Builder(ly, name)
    y0, y1 = -TAP_H, 0.0
    yc = -TAP_H / 2.0
    _column_wells(b, y0, y1)
    _array_implant(b, -77.5, 0)
    _tap_implant(b, -247.5, -77.5)
    _array_implant(b, y0, -247.5)
    b.rect("bnd", 0, y0, W, y1)

    # taps: P (left), N (centre), P (right)
    b.rect("active", 55, yc - 45, 292.5, yc + 45)
    b.rect("active", XW + 55, yc - 45, W - XW - 55, yc + 45)
    b.rect("active", W - 292.5, yc - 45, W - 55, yc + 45)
    for x in (92.5, 232.5):
        b.cont(x, yc)
    for x in (X_VDD - 172.5, X_VDD, X_VDD + 172.5):
        b.cont(x, yc)
    for x in (W - 232.5, W - 92.5):
        b.cont(x, yc)

    # left P tap -> VSS landing of both neighbouring cells (x = 0 column)
    lo = y0 if far == "array" else yc
    b.rect("m1", -35, yc - 35, 300, yc + 35)
    b.rect("m1", -35, (y0 - 70) if far == "array" else (yc - 35), 35, 70)
    # N tap -> VDD landings of both neighbouring cells (PU_L sources)
    b.rect("m1", X_VDD - 240, yc - 35, X_VDD + 240, yc + 35)
    b.rect("m1", 415 + D, (y0 + 35) if far == "array" else (yc - 35), 480 + D, -35)
    # right P tap -> own via stack to the VSS metal3 column at x = W
    b.rect("m1", W - 300, yc - 35, W + 35, yc + 35)
    b.rect("m1", W - 35, yc - 70, W + 35, yc + 70)
    b.via(1, W, yc)
    b.rect("m2", W - 35, yc - 70, W + 35, yc + 70)
    b.via(2, W, yc)

    # VSS strap: joins the VSS metal3 at x = 0 to the one at x = W. Without it the
    # x = W column (and its p-well taps) reaches VSS only through the p-well.
    b.rect("m2", -35, yc - 35, W + 35, yc + 35)
    b.rect("m2", -35, y0 if far == "array" else yc - 70, 35, 0)
    b.via(2, 0, yc)

    # metal3 columns continue through the strip
    b.rect("m3", -35, y0 - 70, 35, 70)
    b.rect("m3", X_VDD - 35, y0 - 70, X_VDD + 35, 70)
    b.rect("m3", W - 35, (y0 - 70) if far == "array" else (lo - 70), W + 35, 70)
    if bitlines:
        b.rect("m3", X_BL - 35, y0 - 70, X_BL + 35, 70)
        b.rect("m3", W - X_BL - 35, y0 - 70, W - X_BL + 35, 70)
    return b


def make_spacer(ly, h, name="ARRAY_SPACER", w=150.0, y0=0.0):
    """Gap column between the data array and the replica column: p-well only."""
    b = Builder(ly, name)
    b.rect("pwell", 0, y0, w, y0 + h)
    b.rect("nplus", 0, y0, w, y0 + h)
    b.rect("vtg", 0, y0, w, y0 + h)
    b.rect("bnd", 0, y0, w, y0 + h)
    return b


def make_edge(ly, name="ARRAY_EDGE", h=100.0):
    """Well/implant extension beyond a top-type cell boundary (y in [0, h])."""
    b = Builder(ly, name)
    _column_wells(b, 0, h)
    _array_implant(b, 0, h)
    b.rect("bnd", 0, 0, W, h)
    return b
