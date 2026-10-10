"""Column periphery (one bitline pair, 1305 nm pitch): PRECHARGE, SENSEAMP,
WRITEDRIVER and WRITE_DATA_BUFFER, stacked top (array side, y = 0) to bottom.

Style: vertical poly gates, device W along y. Global control signals (PRE,
SA_ISO, S_EN, W_EN) are horizontal poly bars that run through every column,
each strapped by a metal2 line. Power: horizontal metal2 rails; VSS metal3 at
x = 0 and VDD metal3 at x = 1305 (both shared with the mirrored neighbour).
Metal1: device straps; metal2: horizontal; metal3: vertical
(Q 417.5, DINB 557.5, DIN_buf 747.5, QB 887.5, DIN input 1027.5).

Bitline contacts at the array side: BL metal3 x = 155, BLB x = 1150.

replica=True builds the replica column variant: DIN tied to VSS, no data
buffer, sense-amp outputs internal.
"""
from .lib import Builder

W = 1305.0
EDGE = 30.0          # power metal3 sits on the column boundary, 30 nm outside this cell
Y_ENB = -4192.5     # write-driver ENB gate pads (below the WD PMOS)
X_Q, X_DINB, X_DINBUF, X_QB, X_DIN = 417.5, 557.5, 747.5, 887.5, 1027.5

# global rows (y centres): poly bars with a metal2 strap, and metal2 rails
ROW_PRE = -60.0
ROW_ISO = -760.0
ROW_SEN = -2525.0
ROW_WEN = -2665.0
ROW_WEN2 = -4802.5          # metal2 only (no poly bar under the inverters)
RAIL_VDD = (-270.0, -1285.0, -3710.0, -4535.0)
RAIL_VSS = (-2382.5, -2995.0, -5545.0)
POLY_BARS = (ROW_PRE, ROW_ISO, ROW_SEN, ROW_WEN)
WELLS = (("nwell", 0.0, -1700.0), ("pwell", -1700.0, -3262.5),
         ("nwell", -3262.5, -4985.0), ("pwell", -4985.0, -5645.0))
H = 5645.0


class Col(object):
    def __init__(self, b):
        self.b = b
        self.ntap = []
        self.ptap = []

    # -- devices -------------------------------------------------------
    def strip(self, cx, gates):
        """cx: contact x list; gates: list of (W, y_top) per gap between
        consecutive contacts. Returns per-contact (y_lo, y_hi) ranges in
        which contacts may sit (narrower adjacent slab)."""
        b = self.b
        rng = []
        for i, (w, top) in enumerate(gates):
            x1 = cx[i] - (37.5 if i == 0 else 0.0)
            x2 = cx[i + 1] + (37.5 if i == len(gates) - 1 else 0.0)
            b.rect("active", x1, top - w, x2, top)
        for i in range(len(cx)):
            adj = [gates[j] for j in (i - 1, i) if 0 <= j < len(gates)]
            lo = max(t - w for w, t in adj)
            hi = min(t for w, t in adj)
            rng.append((lo, hi))
        return rng

    def conts(self, x, lo, hi, n=None):
        """Fill contacts at x within active y-range [lo, hi]; returns centres."""
        top, bot = hi - 37.5, lo + 37.5
        k = int((top - bot) // 140) + 1 if n is None else n
        mid = (top + bot) / 2.0
        ys = [mid + (k - 1) * 70.0 - 140.0 * i for i in range(k)]
        for y in ys:
            self.b.cont(x, y)
        return ys

    def gate(self, x, y1, y2):
        self.b.rect("poly", x - 25, y1, x + 25, y2)

    def pcont(self, x, y):
        self.b.rect("poly", x - 37.5, y - 37.5, x + 37.5, y + 37.5)
        self.b.cont(x, y)

    def strap(self, x, y1, y2, w=70.0):
        """Vertical metal1 strap (w wide) from y1 to y2."""
        self.b.rect("m1", x - w / 2, min(y1, y2), x + w / 2, max(y1, y2))

    def v1(self, x, y):
        self.b.via(1, x, y)

    def v12(self, x, y):
        """via1 + metal2 pad (top/bottom enclosure) + via2."""
        self.b.via(1, x, y)
        self.b.rect("m2", x - 35, y - 70, x + 35, y + 70)
        self.b.via(2, x, y)

    def m2(self, x1, x2, y):
        self.b.rect("m2", x1, y - 35, x2, y + 35)

    def m3(self, x, y1, y2):
        self.b.rect("m3", x - 35, min(y1, y2), x + 35, max(y1, y2))

    def bar_contact(self, x, y):
        """Poly bar contact + metal1 landing + via1 onto the metal2 strap."""
        self.pcont(x, y)
        self.b.rect("m1", x - 35, y - 70, x + 35, y + 70)
        self.v1(x, y)

    def tap(self, kind, x1, y1, x2, y2, ycont, xs):
        self.b.rect("active", x1, y1, x2, y2)
        for x in xs:
            self.b.cont(x, ycont)
        (self.ntap if kind == "n" else self.ptap).append((x1 - 30, y1 - 30, x2 + 30, y2 + 30))


def _implants(col, x1, x2):
    """pplus over n-wells, nplus over p-wells, swapped on tap boxes."""
    from . import lib
    pya = lib.pya
    b = col.b
    u = lib.u
    nw, pw = pya.Region(), pya.Region()
    for kind, ya, yb in WELLS:
        r = pya.Region(pya.Box(u(x1), u(yb), u(x2), u(ya)))
        if kind == "nwell":
            nw += r
        else:
            pw += r
    nt = pya.Region([pya.Box(u(a), u(c), u(bb), u(d)) for a, c, bb, d in col.ntap])
    pt = pya.Region([pya.Box(u(a), u(c), u(bb), u(d)) for a, c, bb, d in col.ptap])
    pplus = ((nw - nt) + pt).merged()
    nplus = ((pw - pt) + nt).merged()
    b.cell.shapes(b.li("pplus")).insert(pplus)
    b.cell.shapes(b.li("nplus")).insert(nplus)


def global_rows(b, x1, x2):
    """Wells, poly bars, metal2 straps and rails shared by every column."""
    for kind, ya, yb in WELLS:
        b.rect(kind, x1, yb, x2, ya)
    b.rect("vtg", x1, -H, x2, 0)
    for y in POLY_BARS:
        b.rect("poly", x1, y - 25, x2, y + 25)
    # metal2 straps / rails reach past the column boundary so edge via2s are enclosed
    for y in POLY_BARS + (ROW_WEN2,) + RAIL_VDD + RAIL_VSS:
        b.rect("m2", x1 - EDGE - 70, y - 35, x2 + EDGE + 70, y + 35)


def make_colperi(ly, replica=False, name=None):
    b = Builder(ly, name or ("COLPERI_RBL" if replica else "COLPERI"))
    c = Col(b)
    global_rows(b, 0, W)
    b.rect("bnd", 0, -H, W, 0)
    # power columns on the column boundaries: VSS metal3 left, VDD metal3 right
    c.m3(-EDGE, 0, -H)
    c.m3(W + EDGE, -140, -H)
    for y in RAIL_VSS:
        b.via(2, -EDGE, y)
    for y in RAIL_VDD:
        b.via(2, W + EDGE, y)

    # ================= PRECHARGE (P): BL | eq | BLB | M2 | VDD | M1 | BL | eq | BLB
    cx = [272.5, 462.5, 652.5, 842.5, 1032.5]
    top = -135.0
    rng = c.strip(cx, [(135, top), (270, top), (270, top), (135, top)])
    for xg, yb in ((367.5, -320), (557.5, -455), (747.5, -455), (937.5, -320)):
        c.gate(xg, yb, ROW_PRE - 25)
    ys = {x: c.conts(x, *rng[i]) for i, x in enumerate(cx)}
    c.bar_contact(100, ROW_PRE)
    # VDD
    c.strap(652.5, min(ys[652.5]) - 67.5, max(ys[652.5]) + 67.5)
    c.v1(652.5, -270)
    # BL strap (x 272.5) runs down to the write driver; BLB strap (1032.5) too
    c.strap(462.5, -550, max(ys[462.5]) + 67.5)       # BLB (inner)
    c.strap(842.5, -690, max(ys[842.5]) + 67.5)       # BL (inner)
    c.v1(462.5, -480)
    c.v1(1032.5, -480)
    c.m2(392.5, 1220, -480)                           # BLB link, to metal3 BLB
    b.via(2, 1150, -480)
    c.m3(1150, -550, 70)
    c.v1(842.5, -620)
    c.v1(272.5, -620)
    c.m2(85, 912.5, -620)                             # BL link, to metal3 BL
    b.via(2, 155, -620)
    c.m3(155, -690, 70)
    # N tap for n-well 1 on the VDD rail
    c.tap("n", 55, -315, 155, -225, -270, [105])
    b.rect("m1", 70, -340, 140, -200)
    c.v1(105, -270)
    # array VDD metal3 (x 652.5) lands on the precharge VDD rail
    b.via(2, 652.5, -270)
    c.m3(652.5, -340, 70)

    # ================= SENSE AMP P strip: BL | ISO | Q | QB | VDD | Q | QB | ISO | BLB
    cx = [237.5, 427.5, 652.5, 877.5, 1067.5]
    pb = -1555.0
    rng = c.strip(cx, [(720, pb + 720), (540, pb + 540), (540, pb + 540), (720, pb + 720)])
    ys = {}
    for i, x in enumerate(cx):
        lo, hi = rng[i]
        ys[x] = c.conts(x, lo + 70 if x == 652.5 else lo, hi)   # VDD clears the QB pad
    c.gate(332.5, pb - 50, ROW_ISO + 25)
    c.gate(972.5, pb - 50, ROW_ISO + 25)
    c.bar_contact(100, ROW_ISO)
    # cross-coupled gates run down to the NMOS latch
    n1_top, n1_bot = -1870.0, -2140.0
    y1, y2 = -1642.5, -1782.5
    c.gate(557.5, n1_bot - 50, pb + 540 + 50)
    c.gate(747.5, n1_bot - 50, pb + 540 + 50)
    # VDD
    c.strap(652.5, min(ys[652.5]) - 67.5, max(ys[652.5]) + 67.5)
    c.v1(652.5, -1285)
    # BL / BLB straps (continuous metal1 down to the write driver)
    c.strap(272.5, -720, -135)
    c.strap(237.5, min(ys[237.5]) - 67.5, -690)
    c.strap(1032.5, -720, -135)
    c.strap(1067.5, min(ys[1067.5]) - 67.5, -690)
    qtop = {x: max(ys[x]) + 67.5 for x in (427.5, 877.5)}
    # outputs to metal3 (Q at 417.5, QB at 887.5)
    b.rect("m1", 382.5, -1170, 460, -1030)
    c.v12(417.5, -1100)
    b.rect("m1", 845, -1170, 922.5, -1030)
    c.v12(887.5, -1100)
    # cross-couple: QB gate pad (557.5, y1) -> QB node; Q gate pad (747.5, y2) -> Q node
    c.pcont(557.5, y1)
    b.rect("m1", 525, y1 - 70, 595, y1 + 70)
    c.v1(560, y1)
    b.rect("m1", 845, y1 - 70, 915, y1 + 70)
    c.v1(880, y1)
    c.m2(490, 950, y1)
    c.pcont(747.5, y2)
    b.rect("m1", 710, y2 - 70, 780, y2 + 70)
    c.v1(745, y2)
    b.rect("m1", 390, y2 - 70, 460, y2 + 70)
    c.v1(425, y2)
    c.m2(355, 815, y2)

    # ================= SENSE AMP N latch: Q | QB | net1 | Q | QB
    cx = [427.5, 652.5, 877.5]
    rng = c.strip(cx, [(270, n1_top), (270, n1_top)])
    ys1 = {}
    for i, x in enumerate(cx):
        lo, hi = rng[i]
        ys1[x] = c.conts(x, lo, hi - 70 if x == 652.5 else hi)  # net1 clears the Q pad
    # Q / QB straps (65 wide) from the PMOS down to the NMOS latch
    for x in (427.5, 877.5):
        c.strap(x, min(ys1[x]) - 67.5, qtop[x], w=65)
    # tail: VSS | S_EN | net1 | S_EN | VSS (two 135 fingers)
    n2_top = -2315.0
    cx2 = [462.5, 652.5, 842.5]
    rng = c.strip(cx2, [(135, n2_top), (135, n2_top)])
    ys2 = {x: c.conts(x, *rng[i]) for i, x in enumerate(cx2)}
    for xg in (557.5, 747.5):
        c.gate(xg, ROW_SEN, n2_top + 50)
    c.bar_contact(100, ROW_SEN)
    c.strap(652.5, min(ys2[652.5]) - 67.5, max(ys1[652.5]) + 67.5)    # net1
    for x in (462.5, 842.5):
        c.strap(x, -2452.5, -2312.5)
        c.v1(x, RAIL_VSS[0])
    c.tap("p", 1155, -2427.5, 1255, -2337.5, -2382.5, [1205])
    b.rect("m1", 1170, -2452.5, 1240, -2312.5)
    c.v1(1205, RAIL_VSS[0])

    # ================= WRITE DRIVER N: BL | EN | int2 | DINB | VSS | DIN | int4 | EN | BLB
    cx = [272.5, 462.5, 652.5, 842.5, 1032.5]
    o_top, i_top = -2740.0, -2815.0
    rng = c.strip(cx, [(360, o_top), (360, i_top), (360, i_top), (360, o_top)])
    ysn = {}
    for i, x in enumerate(cx):
        lo, hi = rng[i]
        ysn[x] = c.conts(x, lo + 100 if 0 < i < 4 else lo, hi)
    c.gate(367.5, o_top - 360 - 50, ROW_WEN + 25)
    c.gate(937.5, o_top - 360 - 50, ROW_WEN + 25)
    c.bar_contact(1205, ROW_WEN)
    c.strap(652.5, min(ysn[652.5]) - 67.5, max(ysn[652.5]) + 67.5)    # VSS
    c.v1(652.5, RAIL_VSS[1])
    for x in (462.5, 842.5):                                          # int2 / int4
        c.strap(x, min(ysn[x]) - 67.5, max(ysn[x]) + 67.5)
    # ================= WRITE DRIVER P: BL | ENB | int1 | DINB | VDD | DIN | int3 | ENB | BLB
    p_top = -3350.0
    rng = c.strip(cx, [(720, p_top)] * 4)
    ysp = {}
    for i, x in enumerate(cx):
        lo, hi = rng[i]
        ysp[x] = c.conts(x, lo, hi - 100 if 0 < i < 4 else hi)
    yp = -3262.5
    for xg in (557.5, 747.5):                                         # DINB / DIN
        c.gate(xg, p_top - 720 - 50, i_top + 50)
    for xg in (367.5, 937.5):                                         # ENB
        c.gate(xg, Y_ENB, p_top + 50)
        c.pcont(xg, Y_ENB)
        b.rect("m1", xg - 35, Y_ENB - 70, xg + 35, Y_ENB + 70)
        c.v1(xg, Y_ENB)
    c.strap(652.5, min(ysp[652.5]) - 67.5, max(ysp[652.5]) + 67.5)    # VDD
    c.v1(652.5, RAIL_VDD[2])
    for x in (462.5, 842.5):                                          # int1 / int3
        c.strap(x, min(ysp[x]) - 67.5, max(ysp[x]) + 67.5)
    # BL / BLB straps through the whole write driver
    c.strap(272.5, min(ysp[272.5]) - 67.5, -720)
    c.strap(1032.5, min(ysp[1032.5]) - 67.5, -720)
    # DINB / DIN input pads in the N-P gap, fed from metal3
    for x in (X_DINB, X_DINBUF):
        c.pcont(x, yp)
        b.rect("m1", x - 35, yp - 70, x + 35, yp + 70)
        c.v12(x, yp)
    c.tap("n", 55, -3755, 155, -3665, -3710, [105])
    b.rect("m1", 70, -3780, 140, -3640)
    c.v1(105, RAIL_VDD[2])

    # ================= INVERTERS
    # pair 1: ENB | W_EN | VDD | DIN_buf | DINB  (P 360 / N 180)
    # pair 2: DIN_buf | b0 | VDD | DIN | b0     (P 270 / N 90)
    it, nt = -4355.0, -5240.0
    pairs = [([77.5, 302.5, 527.5], [207.5, 397.5], 360, 180)]
    if not replica:
        pairs.append(([682.5, 907.5, 1132.5], [812.5, 1002.5], 270, 90))
    yr1, yr2, yr3 = ROW_WEN2, -4942.5, -5082.5
    inv = []
    for cxs, gx, wp, wn in pairs:
        rp = c.strip(cxs, [(wp, it), (wp, it)])
        rn = c.strip(cxs, [(wn, nt), (wn, nt)])
        yp_ = {}
        for i, x in enumerate(cxs):
            lo, hi = rp[i]
            yp_[x] = c.conts(x, lo + 20 if i == 1 else lo, hi)
        yn_ = {x: c.conts(x, *rn[i]) for i, x in enumerate(cxs)}
        for xg in gx:
            c.gate(xg, nt - wn - 50, it + 50)
        inv.append((cxs, gx, wp, wn, yp_, yn_))
    # straps: outputs span P->N through the gap; VDD / VSS to the rails
    for cxs, gx, wp, wn, yp_, yn_ in inv:
        for x in (cxs[0], cxs[2]):
            c.strap(x, min(yn_[x]) - 67.5, max(yp_[x]) + 67.5, w=65)
        x = cxs[1]
        c.strap(x, min(yp_[x]) - 67.5, max(yp_[x]) + 67.5)
        c.v1(x, RAIL_VDD[3])
        c.strap(x, RAIL_VSS[2] - 70, max(yn_[x]) + 67.5)
        c.v1(x, RAIL_VSS[2])
    # ENB: eninv output (77.5) -> write-driver ENB pads (metal2 at y -4157.5)
    b.rect("m1", 45, max(inv[0][4][77.5]), 115, Y_ENB + 70)
    c.v1(80, Y_ENB)
    b.rect("m2", 45, Y_ENB - 70, 115, Y_ENB + 70)
    c.m2(45, 1007.5, Y_ENB)
    # W_EN (second bar) -> eninv gate 207.5: pad on the bar row
    c.pcont(207.5, yr1)
    b.rect("m1", 175, yr1 - 70, 245, yr1 + 70)
    c.v1(210, yr1)
    # DINB: wdinv output (527.5) -> metal3 DINB (557.5), row yr3
    b.rect("m1", 495, yr3 - 70, 565, yr3 + 70)
    c.v1(530, yr3)
    c.m2(460, 627.5, yr3)
    b.via(2, X_DINB, yr3)
    c.m3(X_DINB, yr3 - 70, yp + 70)
    # wdinv gate (397.5), row yr2: DIN_buf link on metal2 to metal3 (747.5)
    c.pcont(397.5, yr2)
    b.rect("m1", 360, yr2 - 70, 430, yr2 + 70)
    c.v1(395, yr2)
    c.m2(325, 817.5, yr2)
    b.via(2, X_DINBUF, yr2)
    if not replica:
        c.m3(X_DINBUF, yr2 - 70, yp + 70)
        # buf1 output DIN_buf (strap 682.5) joins the link
        b.rect("m1", 645, yr2 - 70, 715, yr2 + 70)
        c.v1(680, yr2)
        # b0: buf0 output (1132.5) -> buf1 gate (812.5), row yr3
        c.pcont(812.5, yr3)
        b.rect("m1", 780, yr3 - 70, 850, yr3 + 70)
        c.v1(815, yr3)
        b.rect("m1", 1100, yr3 - 70, 1170, yr3 + 70)
        c.v1(1135, yr3)
        c.m2(745, 1205, yr3)
        # DIN input: metal3 (1027.5) -> buf0 gate (1002.5), row yr2
        c.pcont(1002.5, yr2)
        b.rect("m1", 965, yr2 - 70, 1035, yr2 + 70)
        c.v1(1000, yr2)
        c.m2(930, 1097.5, yr2)
        b.via(2, X_DIN, yr2)
        c.m3(X_DIN, -H, yr2 + 70)
    else:
        # replica: write-driver DIN and the DIN inverter input are tied to VSS
        c.m3(X_DINBUF, RAIL_VSS[2] - 70, yp + 70)
        b.via(2, X_DINBUF, RAIL_VSS[2])
    # Q / QB to the bottom edge on metal3
    c.m3(X_Q, -H, -1030)
    c.m3(X_QB, -H, -1030)
    # P tap for the bottom p-well on the VSS rail
    c.tap("p", 480, -5590, 600, -5500, -5545, [540])
    b.rect("m1", 505, -5615, 575, -5475)
    c.v1(540, RAIL_VSS[2])
    _implants(c, 0, W)
    return b


def make_colperi_spacer(ly, w, name="COLPERI_SPACER"):
    b = Builder(ly, name)
    global_rows(b, 0, w)
    c = Col(b)
    _implants(c, 0, w)
    b.rect("bnd", 0, -H, w, 0)
    return b
