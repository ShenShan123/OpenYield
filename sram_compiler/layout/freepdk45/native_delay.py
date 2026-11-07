"""Locally wired short delay chains with unchanged source devices and loads."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .components import CellView, Component
from .lib import Builder

if TYPE_CHECKING:
    from .control_place import StandardLibrary


def delay_stage_count(component: Component) -> int | None:
    if component.ports != ('VDD', 'VSS', 'in', 'out') or component.devices or component.passives:
        return None
    children = {child.name: child for child in component.children}
    count = sum(bool(re.fullmatch(r'Xwinv\d+', name)) for name in children)
    if count not in (4, 8) or set(children) != ({f'Xwinv{i}' for i in range(count)} | {f'Xwload_{i}_0' for i in range(count)}):
        return None
    for name, child in children.items():
        scale = 2 if name.startswith('Xwinv') else 1
        cell = child.component
        if cell.ports != ('VDD', 'VSS', 'A', 'Z') or cell.children or cell.passives or len(cell.devices) != 2:
            return None
        expected = {'NMOS_VTG': 90e-9 * scale, 'PMOS_VTG': 270e-9 * scale}
        if {device.model for device in cell.devices} != expected.keys():
            return None
        if any(abs(device.width_m - expected[device.model]) > 1e-15
               or abs(device.length_m - 50e-9) > 1e-15 or device.fingers != 1 for device in cell.devices):
            return None
    return count


def build_native_delay(library: StandardLibrary, component: Component) -> CellView:
    count = delay_stage_count(component)
    if count is None:
        raise ValueError('Expected a short one-dummy delay chain')
    if library.row_height is not None:
        raise ValueError('Native delay chains require the standard controller band')
    if component.identity in library.views:
        return library.views[component.identity]
    children = {child.name: child for child in component.children}
    b = Builder(library.layout, 'DELAY_NATIVE_' + component.identity[:16])
    pins_by_child = {}
    device_pins = {}
    fingers = []
    x = 0.

    def wire(layer: str, x0: float, y0: float, x1: float, y1: float, width: float) -> None:
        b.rect(layer, min(x0, x1) - width / 2, min(y0, y1) - width / 2,
               max(x0, x1) + width / 2, max(y0, y1) + width / 2)

    for stage in range(count):
        driver, load = children[f'Xwinv{stage}'], children[f'Xwload_{stage}_0']
        if driver.nodes[:2] != ('VDD', 'VSS') or load.nodes[:3] != ('VDD', 'VSS', driver.nodes[3]):
            raise ValueError('Delay-stage source connectivity differs')
        expected_input = 'in' if stage == 0 else children[f'Xwinv{stage-1}'].nodes[3]
        if driver.nodes[2] != expected_input or (stage == count - 1 and driver.nodes[3] != 'out'):
            raise ValueError('Delay-chain stage order differs')
        for child in (driver, load):
            view = library.build(child.component)
            if view.standard_family != 'INV' or view.height_nm != 1500.:
                raise ValueError('Native delay requires standard1500nm inverter rows')
            b.place(view.builder, x, 0.)
            pins = {name: (px + x, py, layer) for name, (px, py, layer) in view.pins.items()}
            pins_by_child[child.name] = pins
            device_pins.update({child.name + '/' + name: (px + x, py, layer)
                                for name, (px, py, layer) in view.device_pins.items()})
            for finger in view.physical_fingers:
                translated = dict(finger, device=child.name + '/' + finger['device'])
                if 'gate_xy_nm' in translated:
                    gx, gy = translated['gate_xy_nm']
                    translated['gate_xy_nm'] = (gx + x, gy)
                fingers.append(translated)
            # Retain every primitive's original supply via and row connection.
            for supply, y in (('VSS', 90.), ('VDD', 1400.)):
                px, py, layer = pins[supply]
                if layer != 'm1':
                    raise ValueError('Unexpected inverter supply layer')
                b.via(1, px, py)
                b.rect('m1', px - 70., py - 35., px + 70., py + 35.)
                b.rect('m2', px - 70., py - 35., px + 70., py + 35.)
                wire('m2', px, py, px, y, 70.)
            end = x + view.width_nm
            b.rect('pwell', end, 0., end + 30., 640.)
            b.rect('nwell', end, 640., end + 30., 1500.)
            for y in (100., 1400.):
                b.rect('m1', end, y - 35., end + 30., y + 35.)
            x = end + 30.
    width = x
    for y in (90., 1400.):
        wire('m2', 0., y, width, y, 70.)

    def landing(child: str, port: str, y: float) -> tuple[float, float]:
        px, py, layer = pins_by_child[child][port]
        if layer != 'm3':
            raise ValueError('Unexpected delay inverter signal layer')
        wire('m3', px, py, px, y, 70.)
        b.via(3, px, y)
        b.rect('m3', px - 35., y - 70., px + 35., y + 70.)
        b.sq('m4', px, y, 145.)
        return px, y

    inputs = [landing(f'Xwinv{i}', 'A', 300.) for i in range(count)]
    outputs = [landing(f'Xwinv{i}', 'Z', 1400.) for i in range(count)]
    loads = [landing(f'Xwload_{i}_0', 'A', 300.) for i in range(count)]
    for stage in range(count):
        ox, oy = outputs[stage]
        lx, ly = loads[stage]
        wire('m4', ox, oy, lx, oy, 145.)
        wire('m4', lx, oy, lx, ly, 145.)
        if stage + 1 < count:
            nx, ny = inputs[stage + 1]
            wire('m4', lx, ly, nx, ny, 145.)
    pins = dict(VDD=(width / 2, 1400., 'm2'), VSS=(width / 2, 90., 'm2'),
                **{'in': (*inputs[0], 'm4'), 'out': (*outputs[-1], 'm4')})
    for name, (px, py, layer) in pins.items():
        b.label(layer, name, px, py)
    view = CellView(component, b, width, 1500., pins, device_pins, [], [], [])
    view.physical_fingers = fingers
    view.standard_family = 'DELAY_CHAIN'
    library.views[component.identity] = view
    return view
