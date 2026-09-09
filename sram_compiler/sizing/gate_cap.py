"""Reproduce the V2.0.4 gate-charge reference used to normalize explicit RC loads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .qualification import read_measurements


def characterize(output_root, xyce):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]

    def run(condition):
        corner, temperature, vdd = condition
        directory = root / corner
        directory.mkdir(exist_ok=True)
        model = project / 'tran_models' / f'models_{corner}.spice'
        deck = directory / 'deck.sp'
        deck.write_text(f'''V2.0.4 unit inverter input charge
.include "{model}"
.TEMP {temperature}
VDD VDD 0 {vdd}
VIN IN 0 PULSE(0 {vdd} 1n 20p 20p 500p 2n)
MP OUT IN VDD VDD PMOS_VTG W=0.27u L=0.05u
MN OUT IN 0 0 NMOS_VTG W=0.09u L=0.05u
CLOAD OUT 0 1f
.TRAN 1p 2n 0 2p
.MEASURE TRAN QR INTEG I(VIN) FROM=1n TO=1.3n
.MEASURE TRAN QF INTEG I(VIN) FROM=1.52n TO=1.9n
.MEASURE TRAN CR PARAM='ABS(QR)/{vdd}'
.MEASURE TRAN CF PARAM='ABS(QF)/{vdd}'
.END
''')
        with (directory / 'xyce.log').open('w') as log:
            subprocess.run([xyce, '-linsolv', 'KLU', '-o', str(deck), str(deck)],
                           stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True,
                           env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
        return {'corner': corner, 'temperature': temperature, 'vdd': vdd,
                **read_measurements(str(deck) + '.mt0')}

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, [('TT', 25, 1.0), ('SS', 125, .9), ('FF', -40, 1.0)]))
    (root / 'unit_cap.json').write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/qualification/unit_cap'))
    parser.add_argument('--xyce', default='Xyce')
    args = parser.parse_args()
    xyce = shutil.which(args.xyce)
    if xyce is None:
        parser.error('Xyce executable not found')
    print(json.dumps(characterize(args.output_dir, xyce), indent=2))


if __name__ == '__main__':
    main()
