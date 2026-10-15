"""Render native bank GDS previews, including placement and controller detail."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def render(directory: Path, pixels: int = 1600) -> list[Path]:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import klayout.lay as lay
    import klayout.db as db
    manifest = json.loads((directory / 'layout.json').read_text())
    view = lay.LayoutView()
    index = view.load_layout(str(directory / 'full_macro.gds'), False)
    cell = view.cellview(index).layout().cell(manifest['primary'])
    view.select_cell(cell.cell_index(), index)
    view.add_missing_layers()
    view.max_hier()
    view.set_config('background-color', '#07121b')
    view.set_config('grid-visible', 'false')
    view.set_config('text-visible', 'false')
    colors = {1: 0x82b394, 9: 0xe48a82, 10: 0xd2d8df, 11: 0x4dabb8,
              13: 0x6e8ad2, 15: 0x40bfaa, 17: 0xaa83db, 19: 0xd8b767,
              21: 0xdcaacc, 23: 0x75d4b0, 25: 0xffcc7a, 27: 0xeb8ac4, 29: 0x8ccaf2}

    def layers(placement: bool = False) -> None:
        iterator = view.begin_layers()
        while not iterator.at_end():
            layer = iterator.current()
            layer.visible = layer.source_layer in colors and (not placement or layer.source_layer <= 15)
            if layer.visible:
                layer.fill_color = layer.frame_color = colors[layer.source_layer]
                layer.dither_pattern = 0
            iterator.next()

    outputs = []
    box = cell.dbbox()
    def save(name: str, region: db.DBox) -> None:
        ratio = region.width() / region.height()
        width = pixels if ratio >= 1 else max(600, round(pixels * ratio))
        height = pixels if ratio < 1 else max(600, round(pixels / ratio))
        path = directory / name
        view.save_image_with_options(str(path), width, height, 0, 2, 0, region, False)
        outputs.append(path)

    layers()
    save('layout.png', box)
    layers(True)
    save('placement.png', box)
    frame = manifest.get('controller_frame_nm')
    if frame:
        layers()
        x0, y0, x1, y1 = frame
        save('time_control.png', db.DBox((x0-300)/1000, (y0-300)/1000, (x1+300)/1000, (y1+300)/1000))
    shared = manifest.get('decoder_control_bbox_nm')
    if shared:
        layers()
        x0, y0, x1, y1 = shared
        save('decoder_control.png', db.DBox((x0-300)/1000, (y0-300)/1000, (x1+300)/1000, (y1+300)/1000))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-dir', type=Path, required=True)
    parser.add_argument('--pixels', type=int, default=1600)
    args = parser.parse_args(argv)
    if args.pixels < 400 or args.pixels > 4000:
        parser.error('--pixels must be between 400 and 4000')
    for path in render(args.layout_dir, args.pixels):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
