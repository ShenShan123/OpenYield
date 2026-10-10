"""6T thin bitcell and its replica / dummy variants.

Frame: W x H = 1365 x 400 nm. The cell is point-symmetric about its centre
(left half rotated 180 degrees gives the right half). Neighbours are mirrored:
across x = 0 / W (x-mirror) and across y = 0 / H (y-mirror), so contacts on the
frame edges are shared.

  y = H  : BL contact (PG_L drain), VDD (PU_R), VSS (PD_R)      shared row
  y 270..320 : poly P_high = WL (PG_L) on the left, inverter R gate on the right
  y = 200 : storage-node contacts Q (left) / QB (right)
  y 80..130  : poly P_low = inverter L gate on the left, WL (PG_R) on the right
  y = 0  : VSS (PD_L), VDD (PU_L), BLB contact (PG_R drain)     shared row

Routing: WL on metal2 (y = 200 track), BL/BLB/VDD/VSS vertical on metal3.

Variants (`kind`):
  cell     SRAM_6T_CELL
  replica  Replica_CELL: QB tied to VDD (stores 0), WL track = RWL
  passive  Replica_CELL_PASSIVE: QB tied to VDD, both inverter gates and the
           WL track tied to VSS
  dummy    Dummy_CELL: as replica, but BL/BLB diffusions are left floating
"""
from .lib import Builder, Sym

D = 30.0              # n-active to p-active = 140 nm (Calibre Poly.2 gate space, Implant.1)
W = 1305.0 + 2 * D
H = 400.0
# metal3 track centres (local x)
X_VSS_L, X_BL, X_VDD, X_BLB, X_VSS_R = 0.0, 155.0, W / 2, W - 155.0, W
Y_WL = 200.0          # metal2 word-line track centre
M3_W = 70.0


def make_bitcell(ly, kind="cell", name=None):
    names = {"cell": "SRAM_6T_CELL", "replica": "Replica_CELL",
             "passive": "Replica_CELL_PASSIVE", "dummy": "Dummy_CELL"}
    b = Builder(ly, name or names[kind])
    s = Sym(b, W, H)
    has_bl = kind != "dummy"

    # ---- wells, implants, threshold layer, boundary
    xw = 347.5 + D / 2                # n-well edge, 70 nm from both actives
    b.rect("nwell", xw, 0, W - xw, H)
    b.rect("pplus", xw, 0, W - xw, H)
    s.rect("pwell", 0, 0, xw, H)
    s.rect("nplus", 0, 0, xw, H)
    b.rect("vtg", 0, 0, W, H)
    b.rect("bnd", 0, 0, W, H)

    # ---- active: N strip (PD 205 below the jog, PG 135 above), PU strip 90
    s.polygon("active", [(87.5, -37.5), (292.5, -37.5), (292.5, 210.0),
                         (222.5, 210.0), (222.5, 437.5), (87.5, 437.5)])
    s.rect("active", 402.5 + D, -37.5, 492.5 + D, 237.5)

    # ---- poly
    s.rect("poly", 37.5, 80, 755 + D, 130)        # inverter-L gate (PD_L, PU_L)
    s.rect("poly", 680 + D, 82.5, 755 + D, 157.5) # its contact pad
    s.rect("poly", -37.5, 270, 272.5, 320)        # WL gate of PG_L
    s.rect("poly", -37.5, 212.5, 37.5, 287.5)     # shared WL contact pad

    # ---- contacts
    s.cont(155, 0)          # VSS (PD_L source)      / twin: VSS (PD_R)
    s.cont(155, 200)        # Q                     / QB
    if has_bl:
        s.cont(155, 400)    # BL (PG_L drain)       / BLB
    s.cont(447.5 + D, 0)    # VDD (PU_L source)     / VDD (PU_R)
    s.cont(447.5 + D, 200)  # Q (PU_L drain)        / QB
    s.cont(717.5 + D, 120)  # inverter-L gate       / inverter-R gate
    s.cont(0, 250)          # WL poly               / WL poly

    # ---- metal1
    s.rect("m1", -35, -35, 222.5, 35)             # VSS bar
    s.rect("m1", -35, -70, 35, 70)                # VSS via landing
    s.rect("m1", -35, 140, 35, 317.5)             # WL stub (poly contact -> via1)
    if has_bl:
        s.rect("m1", 85, 365, 225, 435)           # BL landing
    s.rect("m1", 377.5 + D, -35, 517.5 + D, 35)   # VDD landing (PU_L)
    if kind == "passive":
        # Q joins only the two drains; both gates go to the VSS word-line track
        s.rect("m1", 122.5, 167.5, 480 + D, 232.5)
        s.rect("m1", 122.5, 132.5, 187.5, 267.5)
        s.rect("m1", 415 + D, 132.5, 480 + D, 267.5)
        b.rect("m1", 552.5 + D, 165, 622.5 + D, 347.5)
        b.rect("m1", 682.5 + D, 52.5, 752.5 + D, 235)
        b.rect("m1", 552.5 + D, 165, 752.5 + D, 235)
        b.via(1, W / 2, Y_WL)
    else:
        s.rect("m1", 122.5, 132.5, 187.5, 267.5)  # Q contact landing
        s.rect("m1", 122.5, 167.5, 620 + D, 232.5)    # Q bar
        s.rect("m1", 555 + D, 167.5, 620 + D, 347.5)  # Q -> inverter-R gate
    if kind != "cell":
        b.rect("m1", W - 480 - D, 200, W - 415 - D, 400)   # QB strap to VDD (PU_R)

    # ---- via1 / metal2
    s.via(1, 0, 0)                                # VSS
    b.via(1, 0, 210)                              # WL (left pad)
    b.via(1, W, 190)                              # WL (right pad)
    if has_bl:
        s.via(1, 155, 400)                        # BL / BLB
    s.via(1, 447.5 + D, 0)                        # VDD
    s.rect("m2", -35, -70, 35, 70)                # VSS
    b.rect("m2", -35, 165, W + 35, 235)           # word-line track
    b.rect("m2", -35, 140, 35, 280)
    b.rect("m2", W - 35, 120, W + 35, 260)
    if has_bl:
        s.rect("m2", 85, 365, 225, 435)
    s.rect("m2", 377.5 + D, -35, X_VDD + 70, 35)  # VDD jog to the centre track

    # ---- via2 / metal3
    s.via(2, 0, 0)
    if has_bl:
        s.via(2, 155, 400)
    s.via(2, X_VDD, 0)
    if kind == "passive":
        b.via(2, 0, 210)                          # WL track -> VSS
    s.rect("m3", -35, -70, 35, H + 70)            # VSS
    if has_bl:
        s.rect("m3", X_BL - 35, -70, X_BL + 35, H + 70)   # BL / BLB
    b.rect("m3", X_VDD - 35, -70, X_VDD + 35, H + 70)     # VDD
    return b
