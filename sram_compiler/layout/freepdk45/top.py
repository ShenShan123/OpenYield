"""Assemble the SRAM_8X8 macro and run DRC / LVS.

The run module exports a fresh schematic and verifies this fixed geometry.

Floorplan (x right, y up; nm):
  y in [0, 3200]        8 bitcell rows; rows mirror in y, columns mirror in x
  y in [-325, 0]        tap strip T0;   y in [3200, 3525]  tap strip T1
  y in [3525, 3925]     RWL dummy row (7 Dummy_CELL);  y in [3925, 4025] edge
  y < -325              column periphery (PRECHARGE / SENSEAMP / WRITEDRIVER /
                        WRITE_DATA_BUFFER), one per column, mirrored with it
  x < X_ARR             row periphery: 4 pair cells (decoder + WL drivers),
                        RWL driver above, address inverters + pins below
"""
import os
from .lib import Builder, new_layout
from . import bitcell as bc
from . import edge as ed
from . import colperi as cp
from . import rowperi as rp

W, H = bc.W, bc.H
NR, NC = 8, 8
SPACER = 155.0
X_ARR = rp.X_ARRAY
X_SP = X_ARR + NC * W
X_REP = X_SP + SPACER
X_END = X_REP + W
Y_T0 = -ed.TAP_H
Y_CP = Y_T0                       # column periphery top
Y_BOT = Y_CP - cp.H
Y_ARR_TOP = NR * H
Y_DUM = Y_ARR_TOP + ed.TAP_H
Y_EDGE = Y_DUM + H
Y_TOP = Y_EDGE + 100.0
NDUMMY = 7

# corner join rows (global y) for the decoder address nets and WL_EN
JOIN = {"A0": -3460.0, "A0b": -3600.0, "A1": -3740.0, "A1b": -3880.0,
        "A2": -4175.0, "A2b": -4315.0, "WL_EN": -4455.0}
Y_ADDRINV = -2560.0
OFF = (bc.W - cp.W) / 2.0        # column periphery is narrower than the bitcell


def col_x(c):
    """(origin x, mirrored) of data column c."""
    return (X_ARR + (c + 1) * W, True) if c % 2 else (X_ARR + c * W, False)


def gx(c, local):
    x0, m = col_x(c)
    return x0 - local if m else x0 + local


def build():
    ly = new_layout()
    cell = bc.make_bitcell(ly, "cell")
    rep = bc.make_bitcell(ly, "replica")
    pas = bc.make_bitcell(ly, "passive")
    dum = bc.make_bitcell(ly, "dummy")
    t0 = ed.make_tap(ly, "ARRAY_TAP_BOT", far="periphery")
    t1 = ed.make_tap(ly, "ARRAY_TAP_TOP", far="array")
    edge = ed.make_edge(ly)
    cpd = cp.make_colperi(ly)
    cpr = cp.make_colperi(ly, replica=True)
    cps = cp.make_colperi_spacer(ly, SPACER + 2 * (bc.W - cp.W) / 2.0)
    chains = (rp.make_chain(ly, "left"), rp.make_chain(ly, "right"))
    top = Builder(ly, "SRAM_8X8")

    def put(child, x0, mx, y, my):
        top.place(child, x0, y, mx=mx, my=my)

    cols = [col_x(c) for c in range(NC)] + [(X_REP, False)]
    for ci, (x0, mx) in enumerate(cols):
        for r in range(NR):
            if ci == NC:
                child = rep if r == NR - 1 else pas
            else:
                child = cell
            put(child, x0, mx, (r + 1) * H if r % 2 else r * H, r % 2 == 1)
        put(t0, x0, mx, 0.0, False)
        put(t1, x0, mx, Y_ARR_TOP, True)
        put(cpr if ci == NC else cpd, x0 - OFF if mx else x0 + OFF, mx, Y_CP, False)
    for ci in range(NDUMMY):
        x0, mx = cols[ci]
        put(dum, x0, mx, Y_DUM, False)
        put(edge, x0, mx, Y_EDGE, False)
    sp = ed.make_spacer(ly, Y_DUM - Y_T0, w=SPACER, y0=Y_T0)
    top.place(sp, X_SP, 0)
    top.place(cps, X_SP - OFF, Y_CP)
    fill = cp.make_colperi_spacer(ly, 2 * OFF, "COLPERI_FILL")
    for c in range(1, NC):
        top.place(fill, X_ARR + c * W - OFF, Y_CP)

    # ---- RWL: dummy-row word line -> metal3 jumper -> replica cell (row 7)
    y_rwl = Y_DUM + bc.Y_WL
    y_rep = Y_ARR_TOP - bc.Y_WL
    xj = X_REP + 400.0
    top.rect("m2", X_ARR + NDUMMY * W, y_rwl - 35, xj + 70, y_rwl + 35)
    top.via(2, xj, y_rwl)
    top.via(2, xj, y_rep)
    top.rect("m3", xj - 35, y_rep - 70, xj + 35, Y_TOP)
    top.label("m3", "RWL", xj, Y_TOP - 20)
    # RBL pin: replica bit line continued to the top edge
    xr = X_REP + bc.X_BL
    top.rect("m3", xr - 35, Y_DUM - 100, xr + 35, Y_TOP)
    top.label("m3", "RBL", xr, Y_TOP - 20)

    # ---- row periphery
    for k in range(NR // 2):
        pair, rl, rr = rp.make_pair(ly, k, chains)
        if k % 2:
            top.place(pair, 0, (k + 1) * rp.PAIR_H, my=True)
        else:
            top.place(pair, 0, k * rp.PAIR_H)
    rwl = rp.make_rwl_driver(ly)
    top.place(rwl, 0, Y_ARR_TOP)
    inv, pins = rp.make_addr_inv_row(ly)
    top.place(inv, 0, Y_ADDRINV + rp.PAIR_H, my=True)   # VSS rail on top

    # ---- corner: decoder address joins
    trk = {
        "A0": [rp.X_BUSL + rp.BUS["A0_L"], rp.X_BUSL + rp.BUS["A0_R"]],
        "A0b": [rp.X_BUSL + rp.BUS["A0b_L"], rp.X_BUSL + rp.BUS["A0b_R"]],
        "A1": [rp.X_BUSL + rp.BUS["A1"]],
        "A1b": [rp.X_BUSL + rp.BUS["A1b"]],
        "A2": [rp.X_BUSL - rp.X_A2, rp.X_CHR + rp.X_A2],
        "A2b": [rp.X_BUSL - rp.X_A2B, rp.X_CHR + rp.X_A2B],
        "WL_EN": [rp.X_BUSL - rp.X_WLEN, rp.X_CHR + rp.X_WLEN],
    }
    ypin, yout = Y_ADDRINV + rp.PAIR_H - rp.YPAD, Y_ADDRINV + rp.PAIR_H - 560.0
    (g0, z0), (g1, z1), (g2, z2) = pins
    stubs = {"A0": [(g0, ypin)], "A0b": [(z0, yout)], "A1": [(g1, ypin)],
             "A1b": [(z1, yout)], "A2": [(g2, ypin)], "A2b": [(z2, yout)], "WL_EN": []}
    for net, y in JOIN.items():
        xs = trk[net] + [x for x, _ in stubs[net]]
        top.rect("m2", 0, y - 35, max(xs) + 70, y + 35)
        for x in trk[net]:
            top.via(2, x, y)
            top.rect("m3", x - 35, y - 70, x + 35, 0)
        for x, y0 in stubs[net]:
            top.via(2, x, y)
            top.rect("m3", x - 35, y - 70, x + 35, y0 + 70)
        if net in ("A0", "A1", "A2", "WL_EN"):
            top.label("m2", net, 20, y)

    # ---- column-periphery global rows run left to the pins / power trunks
    rows = {"PRE": cp.ROW_PRE, "SA_ISO": cp.ROW_ISO, "S_EN": cp.ROW_SEN, "W_EN": cp.ROW_WEN,
            "W_EN2": cp.ROW_WEN2}
    for name, y in rows.items():
        top.rect("m2", 0, Y_CP + y - 35, X_ARR + 35, Y_CP + y + 35)
        if name != "W_EN2":
            top.label("m2", name, 20, Y_CP + y)
    xw = 500.0
    for y in (cp.ROW_WEN, cp.ROW_WEN2):
        top.via(2, xw, Y_CP + y)
    top.rect("m3", xw - 35, Y_CP + cp.ROW_WEN2 - 70, xw + 35, Y_CP + cp.ROW_WEN + 70)
    for y in cp.RAIL_VDD + cp.RAIL_VSS:
        top.rect("m2", 0, Y_CP + y - 35, X_ARR + 35, Y_CP + y + 35)
        top.rect("m2", X_END - 35, Y_CP + y - 35, X_END + 105, Y_CP + y + 35)
    for y in cp.RAIL_VDD:
        top.via(2, rp.X_VDD_TRUNK, Y_CP + y)
    for y in cp.RAIL_VSS:
        top.via(2, rp.X_VSS_TRUNK, Y_CP + y)
    # power trunks over the full height
    top.rect("m3", rp.X_VSS_TRUNK - 35, Y_BOT, rp.X_VSS_TRUNK + 35, Y_TOP)
    top.rect("m3", rp.X_VDD_TRUNK - 35, Y_BOT, rp.X_VDD_TRUNK + 35, Y_TOP)
    top.label("m3", "VSS", rp.X_VSS_TRUNK, Y_BOT + 20)
    top.label("m3", "VDD", rp.X_VDD_TRUNK, Y_BOT + 20)

    # ---- well extensions at the left / right edges of the column periphery
    for kind, ya, yb in cp.WELLS:
        top.rect(kind, X_ARR - 100, Y_CP + yb, X_ARR + OFF, Y_CP + ya)
        top.rect(kind, X_END - OFF, Y_CP + yb, X_END + 100, Y_CP + ya)
    top.rect("vtg", X_ARR - 100, Y_BOT, X_ARR + OFF, Y_CP)
    top.rect("vtg", X_END - OFF, Y_BOT, X_END + 100, Y_CP)

    # ---- column pins at the bottom edge
    for c in range(NC):
        for name, x in (("DIN", cp.X_DIN), ("Q", cp.X_Q), ("QB", cp.X_QB)):
            top.label("m3", "%s%d" % (name, c), gx(c, x + OFF), Y_BOT + 20)
    top.rect("bnd", 0, Y_BOT, X_END + 100, Y_TOP)
    return ly, top


def write(output):
    os.makedirs(output, exist_ok=True)
    ly, top = build()
    gds = os.path.join(output, "sram8x8.gds")
    ly.write(gds)
    bb = top.cell.bbox()
    print("macro bbox: %.3f x %.3f um" % (bb.width() * ly.dbu, bb.height() * ly.dbu))
    top.cell.flatten(True)
    flat = os.path.join(output, "sram8x8_flat.gds")
    ly.write(flat)
