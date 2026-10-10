"""Physical MOS and passive primitives for the complete control block.

All dimensions passed to these builders are nanometres. Every device has an
explicit well tap; no connectivity is inferred through a well alone.
"""
from math import ceil

from .lib import Builder, new_layout

PITCH_X = 280.0
PITCH_Y = 280.0
CELL_WIDTH = 1680.0


def mos(layout, name, model, width, length=50.0):
    width, length = round(width, 6), round(length, 6)
    if width < 90 or width % 5 or length < 50 or length % 5:
        raise ValueError(f'{name}: illegal MOS dimensions W={width}, L={length}')
    cell = Builder(layout, name)
    pmos = model.startswith('PMOS')
    top = 300 + width
    height = ceil((top + 800) / PITCH_Y) * PITCH_Y
    cell.rect('nwell' if pmos else 'pwell', 0, 0, 840, height)
    cell.rect('vtg', 0, 0, 840, height)
    cell.rect('active', 80, 300, 480, top)
    cell.rect('pplus' if pmos else 'nplus', 40, 260, 520, top + 40)
    cell.rect('active', 655, 300, 745, 390)
    cell.rect('nplus' if pmos else 'pplus', 615, 260, 785, 430)
    cell.rect('poly', 280 - length / 2, 250, 280 + length / 2, top + 185)
    cell.sq('poly', 280, top + 120, 75)
    contacts = {'s': (140, 300 + width / 2), 'g': (280, top + 120),
                'd': (420, 300 + width / 2), 'b': (700, 345)}
    pins = {}
    pin_base = ceil((top + 300) / PITCH_Y) * PITCH_Y
    access = {'s': (280., 140.), 'g': (560., top+120.),
              'd': (840., top+400.), 'b': (1400., 140.)}
    for index, (terminal, (x, y)) in enumerate(contacts.items()):
        gx, gy = access[terminal]
        cell.cont(x, y)
        cell.rect('m1', x - 35, min(y,gy) - 70, x + 35, max(y,gy) + 70)
        cell.via(1, x, gy)
        cell.rect('m2', min(x,gx) - 35, gy - 70, max(x,gx) + 35, gy + 70)
        cell.via(2, gx, gy)
        pin_y = pin_base + index * PITCH_Y
        cell.rect('m3', gx - 35, min(gy,pin_y) - 70, gx + 35, max(gy,pin_y) + 70)
        pins[terminal] = (gx, pin_y, 'm3')
    return cell, pins, max(height, pin_base + 4 * PITCH_Y)


def passive(layout, name, resistance, cap_side_nm=2800., poly_length_nm=None):
    """Field-poly resistor and M4/M5 overlap capacitor, measured by xRC.

    The kit has no recognized intentional R/C devices in its LVS deck. These
    are physical interconnect passives and require a separate value check.
    Nominal target values are provenance, not asserted extracted values.
    """
    cell = Builder(layout, name)
    if poly_length_nm is None:
        poly_length_nm = max(140., (resistance - 177.) * 50. / 7.8 + 115.)
    length = round(poly_length_nm / 5) * 5
    side = round(cap_side_nm / 5) * 5
    x0, x1, y = 280., 280. + length, 560.
    cell.rect('poly', x0 - 37.5, y - 25, x1 + 37.5, y + 25)
    pins = {}
    for terminal, x, offset in [('in', x0, 0), ('out', x1, 1)]:
        # Three parallel contact/via stacks lower the access resistance enough
        # to realize the 0.26 kohm filter without an artificial ideal contact.
        cell.rect('poly', x-37.5, y-182.5, x+37.5, y+182.5)
        for delta in (-145., 0., 145.):
            cell.cont(x, y+delta)
            cell.via(1, x, y+delta)
            cell.via(2, x, y+delta)
        cell.rect('m1', x-35, y-215, x+35, y+215)
        cell.rect('m2', x-35, y-215, x+35, y+215)
        gx = ceil(x / PITCH_X) * PITCH_X
        cell.rect('m2', min(x,gx)-35, y-215, max(x,gx)+35, 910.)
        py = 1120 + offset * PITCH_Y
        cell.rect('m3', x-35, y-215, x+35, y+215)
        cell.rect('m3', min(x,gx)-35, 700., max(x,gx)+35, 840.)
        cell.rect('m3', gx-35, 770., gx+35, py+70)
        pins[terminal] = (gx, py, 'm3')
    # Capacitor output plate is on M4; grounded plate is on M5.
    cx, cy = 280., 2240.
    cell.rect('m4', cx, cy, cx + side, cy + side)
    cell.rect('m5', cx, cy, cx + side, cy + side)
    ox = pins['out'][0]
    cell.rect('m3', ox-35, 1050, ox+35, cy+840+70)
    cell.via(3, ox, cy+840)
    cell.rect('m4', min(ox-70,cx), cy+840-70, max(ox+70,cx+side), cy+840+70)
    gy = cy + side + 280
    cell.rect('m5', 1400-70, cy+840, 1400+70, gy+70)
    pins['ground'] = (1400., gy, 'm5')
    return cell, pins, dict(target_resistance_ohm=resistance, poly_length_nm=length,
                            capacitor_side_nm=side)
