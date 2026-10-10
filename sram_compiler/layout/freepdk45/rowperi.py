"""Row periphery: 3-8 decoder slices + word-line drivers, two rows per cell.

A single 400 nm bitcell row cannot hold a series NMOS stack (contact-gate-gate-
contact needs ~455 nm plus spacing), so the row logic uses 800 nm "pair"
standard cells (VSS rail at y = 0, VDD rail at y = 800, metal2 rails), mirrored
pair by pair. Each pair carries two full chains, one per word line:

    NAND3 -> INV -> NAND2(EN = VDD) -> INV -> NAND2(WL_EN) -> INV(540/180) -> WL

  [taps + power trunks][left chain (x-mirrored)][address bus][right chain][WL jog] | array

Template (pair-local y): NMOS slab bottom at 87.5 (W 180 or 90), PMOS slab
[442.5, 712.5] (W 270), gate pads at y = 355 between them. Gate inputs:
  * NAND3 C (VSS side)  <- bus, metal2 at y = 355 (A0 / A0b)
  * NAND3 B             <- bus, metal1 riser + metal2 at y = 215 (A1 / A1b)
  * NAND3 A (Z side)    <- one of two metal3 tracks over the pad (A2 / A2b)
  * NAND2 B inputs      <- metal3 track over the pad (VDD resp. WL_EN)
  * other inputs        <- metal1 from the previous stage output at y = 355
Outputs: right chain WL on metal2 y = 200; left chain WL on metal2 y = 660,
jogged to y = 600 next to the array.
"""
from .lib import Builder

PAIR_H = 800.0
YPAD = 355.0
YN0 = 87.5                 # NMOS slab bottom
YP0, YP1 = 442.5, 712.5    # PMOS slab
YNC = 132.5                # NMOS contacts
YPC_OUT = 560.0            # PMOS contact on output columns
YPC = 612.5                # PMOS contact on columns next to a gate pad
Y_PLINK = 555.0            # metal2 row joining PMOS drains
Y_WL_R = 180.0             # right-chain word line (metal2)
Y_WL_L = 600.0             # left-chain word line (metal4 across the pair, metal2 at the array)
Y_ROW_B = 215.0            # metal2 row for the NAND3 B input

# chain-local x (bus edge at 0)
NAND3_X0, INV1_X0, NAND2A_X0, INV2_X0, NAND2D_X0, INVD_X0 = 120.0, 885.0, 1265.0, 1835.0, 2215.0, 2785.0
X_G3 = NAND3_X0 + 475.0
X_A2, X_A2B = X_G3 - 70.0, X_G3 + 70.0       # metal3 tracks over the NAND3 A pad
X_VDDEN = NAND2A_X0 + 285.0                  # metal3 VDD over the AND2 enable pad
X_WLEN = NAND2D_X0 + 285.0                   # metal3 WL_EN over the driver B pad
CHAIN_W = 3330.0
X_DRV_Z = INVD_X0 + 225.0

BUS_W = 840.0
BUS = {"A0_L": 70.0, "A0b_L": 210.0, "A1": 350.0, "A1b": 490.0, "A0_R": 630.0, "A0b_R": 770.0}


class Std(object):
    """Standard-cell primitives drawn into Builder `b` at x offset `ox`,
    optionally mirrored in x (mx: x -> ox - x)."""

    def __init__(self, b, ox=0.0, mx=False):
        self.b, self.ox, self.mx = b, ox, mx

    def X(self, x):
        return self.ox - x if self.mx else self.ox + x

    def rect(self, layer, x1, y1, x2, y2):
        self.b.rect(layer, self.X(x1), y1, self.X(x2), y2)

    def cont(self, x, y):
        self.b.cont(self.X(x), y)

    def via(self, n, x, y):
        self.b.via(n, self.X(x), y)

    # ---- devices
    def nslab(self, x1, x2, w):
        self.rect("active", x1, YN0, x2, YN0 + w)

    def pslab(self, x1, x2):
        self.rect("active", x1, YP0, x2, YP1)

    def gate(self, x, n=True, p=True):
        y1 = YN0 - 50 if n else YPAD - 25
        y2 = YP1 + 50 if p else YPAD + 25
        self.rect("poly", x - 25, y1, x + 25, y2)

    def pad(self, x, via=False):
        """Gate pad; 65 nm metal1 keeps 65 nm to an output strap 130 nm away.
        With via=True the landing is 70 wide (only where no strap is close)."""
        self.rect("poly", x - 37.5, YPAD - 37.5, x + 37.5, YPAD + 37.5)
        self.cont(x, YPAD)
        w = 35.0 if via else 32.5
        self.rect("m1", x - w, YPAD - 70, x + w, YPAD + 70)
        if via:
            self.via(1, x, YPAD)

    def pad_track(self, x):
        """Gate pad fed from a metal3 track at x (output strap 130 nm to the right)."""
        self.rect("poly", x - 37.5, YPAD - 37.5, x + 37.5, YPAD + 37.5)
        self.cont(x, YPAD)
        self.rect("m1", x - 37.5, YPAD - 70, x + 32.5, YPAD + 70)
        self.via(1, x - 2.5, YPAD)

    def vss(self, x):
        self.cont(x, YNC)
        self.rect("m1", x - 35, -70, x + 35, YNC + 67.5)
        self.via(1, x, 0)

    def vdd(self, x):
        self.cont(x, YPC)
        self.rect("m1", x - 35, YPC - 67.5, x + 35, PAIR_H + 70)
        self.via(1, x, PAIR_H)

    def out(self, x, n=True):
        """Output column: NMOS + PMOS drains joined by a 65 nm strap."""
        if n:
            self.cont(x, YNC)
        self.cont(x, YPC_OUT)
        self.rect("m1", x - 32.5, (YNC if n else YPAD) - 67.5, x + 32.5, YPC_OUT + 67.5)

    def pz(self, x):
        """PMOS-only drain column joined on the metal2 PMOS link row."""
        self.cont(x, YPC)
        self.rect("m1", x - 35, Y_PLINK - 70, x + 35, YPC + 67.5)
        self.via(1, x, Y_PLINK)

    def plink(self, x1, x2):
        self.rect("m1", x2 - 35, Y_PLINK - 70, x2 + 35, Y_PLINK + 70)
        self.via(1, x2, Y_PLINK)
        self.rect("m2", x1 - 70, Y_PLINK - 35, x2 + 70, Y_PLINK + 35)

    def link(self, x1, x2):
        """Metal1 at the pad row from an output strap to the next gate pad."""
        self.rect("m1", x1, YPAD - 32.5, x2, YPAD + 32.5)

    # ---- gates (x0 = first contact column); return pad x positions
    def nand3(self, x0):
        c0, g1, c1, g2, c2, g3, c3 = (x0, x0 + 95, x0 + 190, x0 + 285, x0 + 380, x0 + 475, x0 + 610)
        self.nslab(c0 - 37.5, c3 + 37.5, 180)
        self.pslab(c0 - 37.5, c3 + 37.5)
        for g in (g1, g2, g3):
            self.gate(g)
        self.vss(c0)
        self.vdd(c0)
        self.pz(c1)
        self.vdd(c2)
        self.out(c3)
        self.plink(c1, c3)
        return g1, g2, g3, c3

    def inv(self, x0, wn=90):
        c0, g, z = x0, x0 + 95, x0 + 225
        self.nslab(c0 - 37.5, z + 37.5, wn)
        self.pslab(c0 - 37.5, z + 37.5)
        self.gate(g)
        self.vss(c0)
        self.vdd(c0)
        self.out(z)
        self.pad(g)
        return g, z

    def nand2(self, x0):
        c0, ga, c1, gb, c2 = x0, x0 + 95, x0 + 190, x0 + 285, x0 + 415
        self.nslab(c0 - 37.5, c2 + 37.5, 180)
        self.pslab(c0 - 37.5, c2 + 37.5)
        for g in (ga, gb):
            self.gate(g)
        self.vss(c0)
        self.pz(c0)
        self.vdd(c1)
        self.out(c2)
        self.plink(c0, c2)
        self.pad(ga)
        self.pad_track(gb)
        return ga, gb, c2

    def inv540(self, x0):
        c0, g1, z, g2, c3 = x0, x0 + 95, x0 + 225, x0 + 355, x0 + 450
        self.nslab(c0 - 37.5, z + 37.5, 180)
        self.pslab(c0 - 37.5, c3 + 37.5)
        self.gate(g1)
        self.gate(g2, n=False)
        self.rect("poly", g1, YPAD - 25, g2, YPAD + 25)      # joins the two PMOS fingers
        self.vss(c0)
        self.vdd(c0)
        self.vdd(c3)
        self.out(z)
        self.pad(g1)
        return g1, z

    def track_landing(self, x):
        """via1 (already on the pad, at x - 2.5) -> metal2 landing -> via2 onto
        the metal3 track at x."""
        self.rect("m2", x - 72.5, YPAD - 35, x + 70, YPAD + 35)
        self.via(2, x, YPAD)


def wells(b, x1, x2, ntaps=(), ptaps=()):
    from . import lib
    pya, u = lib.pya, lib.u
    b.rect("pwell", x1, 0, x2, YPAD)
    b.rect("nwell", x1, YPAD, x2, PAIR_H)
    b.rect("vtg", x1, 0, x2, PAIR_H)
    nt = pya.Region([pya.Box(u(a), u(c), u(d), u(e)) for a, c, d, e in ntaps])
    pt = pya.Region([pya.Box(u(a), u(c), u(d), u(e)) for a, c, d, e in ptaps])
    pw = pya.Region(pya.Box(u(x1), 0, u(x2), u(YPAD)))
    nw = pya.Region(pya.Box(u(x1), u(YPAD), u(x2), u(PAIR_H)))
    b.cell.shapes(b.li("nplus")).insert(((pw - pt) + nt).merged())
    b.cell.shapes(b.li("pplus")).insert(((nw - nt) + pt).merged())


def chain(s, wl_side):
    """Draw one decoder + driver chain with primitives `s` (chain-local x)."""
    g1, g2, g3, z3 = s.nand3(NAND3_X0)
    s.pad(g1, via=True)
    s.rect("m2", -35, YPAD - 35, g1 + 70, YPAD + 35)          # C input from the bus
    s.pad(g2)
    s.rect("m1", g2 - 35, Y_ROW_B - 70, g2 + 35, YPAD)       # B input riser
    s.via(1, g2, Y_ROW_B)
    s.rect("m2", -35, Y_ROW_B - 35, g2 + 70, Y_ROW_B + 35)
    s.pad(g3, via=True)                                      # A input: jog drawn per row
    gi, zi = s.inv(INV1_X0)
    s.link(z3, gi)
    ga, gb, za = s.nand2(NAND2A_X0)
    s.link(zi, ga)
    s.track_landing(gb)
    gi2, zi2 = s.inv(INV2_X0)
    s.link(za, gi2)
    gd, gbd, zd = s.nand2(NAND2D_X0)
    s.link(zi2, gd)
    s.track_landing(gbd)
    gz, zw = s.inv540(INVD_X0)
    s.link(zd, gz)
    if wl_side == "right":
        y = Y_WL_R
        s.rect("m1", zw - 35, y - 70, zw + 35, y + 70)
        s.via(1, zw, y)
        s.rect("m2", zw - 70, y - 35, CHAIN_W, y + 35)
    else:
        # climb to metal4 and run over the chain towards the bus
        y = Y_WL_L
        s.rect("m1", zw - 35, y - 70, zw + 35, y + 70)
        s.via(1, zw, y)
        s.rect("m2", zw - 35, y - 70, zw + 35, y + 70)
        s.via(2, zw, y)
        s.rect("m3", zw - 35, y - 70, zw + 35, y + 70)
        s.via(3, zw, y)
        s.rect("m4", -35, y - 70, zw + 70, y + 70)
    # metal3 tracks through the chain
    for x in (X_A2, X_A2B, X_VDDEN, X_WLEN):
        s.rect("m3", x - 35, -70, x + 35, PAIR_H + 70)
    s.via(2, X_VDDEN, PAIR_H)
    # rails
    s.rect("m2", 0, -35, CHAIN_W, 35)
    s.rect("m2", 0, PAIR_H - 35, CHAIN_W, PAIR_H + 35)


def make_chain(ly, wl_side):
    b = Builder(ly, "ROWCHAIN_" + wl_side.upper())
    chain(Std(b), wl_side)
    wells(b, 0, CHAIN_W)
    b.rect("bnd", 0, 0, CHAIN_W, PAIR_H)
    return b


def a2_jog(s, b2):
    """NAND3 A pad -> A2 (b2 = 1) or A2b metal3 track, in chain-local x."""
    if b2:
        s.rect("m2", X_A2 - 70, YPAD - 35, X_G3 + 70, YPAD + 35)
        s.via(2, X_A2, YPAD)
    else:
        s.rect("m2", X_G3 - 70, YPAD - 35, X_A2B + 70, YPAD + 35)
        s.via(2, X_A2B, YPAD)


# ---------------------------------------------------------------- pair cell
TRUNK_W = 600.0
X_VSS_TRUNK, X_VDD_TRUNK = 100.0, 300.0
X_BUSL = TRUNK_W + CHAIN_W            # left edge of the bus
X_CHR = X_BUSL + BUS_W                # right chain origin
X_JOG = X_CHR + CHAIN_W + 70.0        # left WL jog column
X_ARRAY = X_CHR + CHAIN_W + 270.0     # array starts here (relative to row periphery origin)
RP_W = X_ARRAY


def trunk_and_taps(b, x_end):
    """Power trunks (metal3) and well taps at the left end of a pair-height cell."""
    b.rect("m3", X_VSS_TRUNK - 35, -70, X_VSS_TRUNK + 35, PAIR_H + 70)
    b.rect("m3", X_VDD_TRUNK - 35, -70, X_VDD_TRUNK + 35, PAIR_H + 70)
    b.via(2, X_VSS_TRUNK, 0)
    b.via(2, X_VDD_TRUNK, PAIR_H)
    # taps
    b.rect("active", 420, 55, 540, 145)
    b.cont(480, 100)
    b.rect("m1", 445, -70, 515, 167.5)
    b.via(1, 480, 0)
    b.rect("active", 420, 655, 540, 745)
    b.cont(480, 700)
    b.rect("m1", 445, 632.5, 515, PAIR_H + 70)
    b.via(1, 480, PAIR_H)
    b.rect("m2", 0, -35, x_end, 35)
    b.rect("m2", 0, PAIR_H - 35, x_end, PAIR_H + 35)
    return [(390, 625, 570, 775)], [(390, 25, 570, 175)]


X_BUS_TAP = 420.0     # bus-local x of the mid-row well taps


def bus_taps(b, x):
    """P and N well taps at global x (bus region): keeps every row-periphery
    MOS within 5 um of a tap. Returns (ntap, ptap) implant boxes."""
    b.rect("active", x - 60, 55, x + 60, 145)
    b.cont(x, 100)
    b.rect("m1", x - 35, -70, x + 35, 167.5)
    b.via(1, x, 0)
    b.rect("active", x - 60, 655, x + 60, 745)
    b.cont(x, 700)
    b.rect("m1", x - 35, 632.5, x + 35, PAIR_H + 70)
    b.via(1, x, PAIR_H)
    return [(x - 90, 625, x + 90, 775)], [(x - 90, 25, x + 90, 175)]


def make_pair(ly, k, chains):
    """Pair cell for word lines 2k, 2k+1. `chains` = (left_cell, right_cell).
    Instanced mirrored in y when k is odd (then the right chain drives 2k+1)."""
    b = Builder(ly, "ROWPAIR%d" % k)
    left, right = chains
    b.place(left, X_BUSL, 0, mx=True)
    b.place(right, X_CHR, 0)
    mirrored = k % 2 == 1
    r_right = 2 * k + 1 if mirrored else 2 * k
    r_left = 2 * k if mirrored else 2 * k + 1
    for side, r in (("L", r_left), ("R", r_right)):
        b0, b1, b2 = r & 1, (r >> 1) & 1, (r >> 2) & 1
        t0 = BUS[("A0_" if b0 else "A0b_") + side]
        t1 = BUS["A1" if b1 else "A1b"]
        if side == "L":
            b.rect("m2", X_BUSL, YPAD - 35, X_BUSL + t0 + 70, YPAD + 35)
            b.rect("m2", X_BUSL, Y_ROW_B - 35, X_BUSL + t1 + 70, Y_ROW_B + 35)
            a2_jog(Std(b, X_BUSL, mx=True), b2)
        else:
            b.rect("m2", X_BUSL + t0 - 70, YPAD - 35, X_CHR, YPAD + 35)
            b.rect("m2", X_BUSL + t1 - 70, Y_ROW_B - 35, X_CHR, Y_ROW_B + 35)
            a2_jog(Std(b, X_CHR), b2)
        b.via(2, X_BUSL + t0, YPAD)
        b.via(2, X_BUSL + t1, Y_ROW_B)
    for x in BUS.values():
        b.rect("m3", X_BUSL + x - 35, -70, X_BUSL + x + 35, PAIR_H + 70)
    # word lines: right chain on metal2 (y = 180); left chain on metal4 (y = 600)
    # over the bus and the right chain, back down to metal2 next to the array
    b.rect("m2", X_CHR + CHAIN_W - 35, Y_WL_R - 35, X_ARRAY + 35, Y_WL_R + 35)
    b.rect("m4", X_BUSL - 35, Y_WL_L - 70, X_JOG + 70, Y_WL_L + 70)
    b.via(3, X_JOG, Y_WL_L)
    b.rect("m3", X_JOG - 35, Y_WL_L - 70, X_JOG + 35, Y_WL_L + 70)
    b.via(2, X_JOG, Y_WL_L)
    b.rect("m2", X_JOG - 70, Y_WL_L - 35, X_ARRAY + 35, Y_WL_L + 35)
    # rails across the bus; VSS continues into the array edge, VDD stops short
    b.rect("m2", X_BUSL, -35, X_CHR, 35)
    b.rect("m2", X_BUSL, PAIR_H - 35, X_CHR, PAIR_H + 35)
    b.rect("m2", X_CHR + CHAIN_W - 35, -35, X_ARRAY + 35, 35)
    nt, pt = trunk_and_taps(b, TRUNK_W)
    b.rect("m2", TRUNK_W - 35, -35, TRUNK_W + 35, 35)
    wells(b, 0, TRUNK_W, nt, pt)
    wells(b, X_BUSL, X_CHR, *bus_taps(b, X_BUSL + X_BUS_TAP))
    b.rect("bnd", 0, 0, X_ARRAY, PAIR_H)
    return b, r_left, r_right


def make_rwl_driver(ly):
    """RWL driver (A = VDD, B = WL_EN) in the right-chain slot; output on y = 525."""
    b = Builder(ly, "RWLDRV")
    s = Std(b, X_CHR)
    gd, gbd, zd = s.nand2(NAND2D_X0)
    s.track_landing(gbd)
    # A input tied to VDD: pad -> metal2 at y = 355 -> VDD metal3 track (X_VDDEN)
    s.via(1, gd, YPAD)
    s.rect("m1", gd - 35, YPAD - 70, gd + 35, YPAD + 70)
    s.rect("m2", gd - 35, YPAD - 70, gd + 35, YPAD + 70)
    s.rect("m2", X_VDDEN - 70, YPAD - 35, gd + 35, YPAD + 35)
    s.via(2, X_VDDEN, YPAD)
    gz, zw = s.inv540(INVD_X0)
    s.link(zd, gz)
    y, yo = 560.0, 525.0          # via on the output strap; dummy-row word line
    s.rect("m1", zw - 35, y - 70, zw + 35, y + 70)
    s.via(1, zw, y)
    b.rect("m2", X_CHR + zw - 70, y - 35, X_JOG + 35, y + 35)
    b.rect("m2", X_JOG - 35, yo - 35, X_JOG + 35, y + 35)
    b.rect("m2", X_JOG - 35, yo - 35, X_ARRAY + 35, yo + 35)
    for x in (X_VDDEN, X_WLEN):
        s.rect("m3", x - 35, -70, x + 35, PAIR_H + 70)
    s.via(2, X_VDDEN, PAIR_H)
    b.rect("m2", X_CHR + X_VDDEN - 70, -35, X_CHR + CHAIN_W, 35)
    b.rect("m2", X_CHR + X_VDDEN - 70, PAIR_H - 35, X_CHR + CHAIN_W, PAIR_H + 35)
    b.rect("m2", TRUNK_W, -35, X_CHR + X_VDDEN, 35)
    b.rect("m2", TRUNK_W, PAIR_H - 35, X_CHR + X_VDDEN, PAIR_H + 35)
    nt, pt = trunk_and_taps(b, TRUNK_W)
    wells(b, 0, TRUNK_W, nt, pt)
    wells(b, TRUNK_W, X_CHR + CHAIN_W,         # continuous with the tapped trunk wells
          *bus_taps(b, X_BUSL + X_BUS_TAP))
    b.rect("bnd", 0, 0, X_ARRAY, PAIR_H)
    return b


def make_addr_inv_row(ly):
    """Three address inverters (DECODER3_8 INV_A*) in one pair-height row.
    Returns (cell, [(x_in_pad, x_out_strap) per inverter])."""
    b = Builder(ly, "ADDRINV")
    s = Std(b, 0)
    pins = []
    xs = (660.0, 1040.0, 1880.0)
    for x0 in xs:
        g, z = s.inv(x0)
        gi, zo = g - 2.5, z + 10.0     # keep the two metal3 stubs 70 apart
        s.rect("m1", gi - 35, YPAD - 70, gi + 35, YPAD + 70)
        s.via(1, gi, YPAD)
        s.rect("m2", gi - 35, YPAD - 70, gi + 35, YPAD + 70)
        s.via(2, gi, YPAD)
        s.rect("m1", zo - 35, 490, zo + 35, 630)
        s.via(1, zo, 560)
        s.rect("m2", zo - 35, 490, zo + 35, 630)
        s.via(2, zo, 560)
        pins.append((gi, zo))
    nt, pt = trunk_and_taps(b, 2300.0)
    wells(b, 0, 2300.0, nt, pt)
    b.rect("bnd", 0, 0, 2300.0, PAIR_H)
    return b, pins
