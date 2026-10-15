"""Diffusion, contact and metal primitives shared by the dense write driver."""


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
