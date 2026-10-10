"""The macro's 8x8 bitcell tile and tap strips, without periphery."""
from pathlib import Path

from .lib import Builder, new_layout
from . import bitcell as bc
from . import edge as ed
from .export import Emitter, parse


def write(output, schematic):
    output = Path(output)
    root = parse(Path(schematic).read_text().splitlines())
    emitter = Emitter()
    index = emitter.visit(root.resolve('SRAM_6T_CORE_8x8'))
    names = emitter.names()
    names[index] = 'ARRAY8X8'
    Path(schematic).write_text('\n'.join(emitter.render(names) + ['.end', '']))
    ly = new_layout()
    cell = bc.make_bitcell(ly, 'cell')
    t0 = ed.make_tap(ly, 'TAP_BOT', far='periphery')
    t1 = ed.make_tap(ly, 'TAP_TOP', far='periphery')
    top = Builder(ly, 'ARRAY8X8')
    for col in range(8):
        x, mirror = ((col + 1) * bc.W, True) if col % 2 else (col * bc.W, False)
        for row in range(8):
            top.place(cell, x, (row + 1) * bc.H if row % 2 else row * bc.H,
                      mx=mirror, my=bool(row % 2))
        top.place(t0, x, 0, mx=mirror)
        top.place(t1, x, 8 * bc.H, mx=mirror, my=True)
        for name, local in (('BL', bc.X_BL), ('BLB', bc.X_BLB)):
            top.label('m3', f'{name}{col}', x - local if mirror else x + local, -ed.TAP_H)
    for row in range(8):
        top.label('m2', f'WL{row}', -20, row * bc.H + bc.Y_WL)
    # In the macro these VDD stripes join through the column periphery. A
    # standalone tile needs an explicit strap: well conductivity must not be
    # used to excuse seven disconnected supply nets during LVS.
    top.rect('m4', 0, -ed.TAP_H - 70, 8 * bc.W, -ed.TAP_H + 70)
    for col in range(8):
        top.via(3, col * bc.W + bc.X_VDD, -ed.TAP_H)
    top.label('m3', 'VSS', 0, -100)
    top.label('m3', 'VDD', bc.X_VDD, -100)
    gds = output / 'array.gds'
    ly.write(str(gds))
    return gds
