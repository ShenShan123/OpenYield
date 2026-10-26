"""Generate a compact column-split FreePDK45 bank with a 2:1 column mux."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--compact-cell', action='store_true', default=True,
                        help='use the compact 1125 nm column / 395 nm row cell (always enabled)')
    parser.add_argument('--split-columns', action='store_true', default=True,
                        help='place column halves beside one shared decoder (always enabled)')
    parser.add_argument('--with-controller', action='store_true',
                        help='include TIME_CONTROL, data registers, RWL driver and output latch (>=16 rows)')
    args = parser.parse_args(argv)
    if args.with_controller:
        from .compact_center import generate
        generate(args.deck, args.output_dir)
    else:
        from .compact_bank import build
        build(args.deck, args.output_dir)
    print(args.output_dir / 'full_macro.gds')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
