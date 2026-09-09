"""Reproduce the V2.0.4 wide-gate transient and total-width DC checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .qualification import read_measurements


def characterize(output_root, xyce):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = Path(__file__).resolve().parents[2] / 'tran_models/models_TT.spice'

    def run(nf):
        results = {}
        for analysis in ('tran', 'dc'):
            directory = root / f'{analysis}_nf{nf}'
            directory.mkdir(exist_ok=True)
            deck = directory / 'deck.sp'
            if analysis == 'tran':
                circuit = f'''VDD VDD 0 1
VIN IN 0 PULSE(0 1 1n 20p 20p 500p 2n)
MP OUT IN VDD VDD PMOS_VTG W=92.34u L=.05u NF={nf}
MN OUT IN 0 0 NMOS_VTG W=61.56u L=.05u NF={nf}
CLOAD OUT 0 550f
.TRAN 1p 2n 0 1p
.MEASURE TRAN FALL_EDGE TRIG V(OUT)=.9 FALL=1 TARG V(OUT)=.1 FALL=1
.MEASURE TRAN RISE_EDGE TRIG V(OUT)=.1 RISE=1 TARG V(OUT)=.9 RISE=1
'''
            else:
                circuit = f'''VD D 0 .5
VG G 0 1
MN D G 0 0 NMOS_VTG W=61.56u L=.05u NF={nf}
.DC VD .5 .5 .1
.MEASURE DC I_ON FIND I(VD) AT=.5
'''
            deck.write_text(f'V2.0.4 gate fingering {analysis}\n.include "{model}"\n.TEMP 27\n{circuit}.END\n')
            with (directory / 'xyce.log').open('w') as log:
                subprocess.run([xyce, '-linsolv', 'KLU', '-o', str(deck), str(deck)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
            suffix = '.mt0' if analysis == 'tran' else '.ms0'
            results[analysis] = {'nf': nf, **read_measurements(str(deck) + suffix)}
        return results

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, (1, 16, 100)))
    transient, dc = [r['tran'] for r in results], [r['dc'] for r in results]
    finite = all(isinstance(value, (int, float)) and math.isfinite(value)
                 for row in transient + dc for key, value in row.items() if key != 'nf')
    passed = (finite and all(0 < row[edge] < .5 * transient[0][edge]
                            for row in transient[1:] for edge in ('FALL_EDGE', 'RISE_EDGE'))
              and all(.98 < row['I_ON'] / dc[0]['I_ON'] < 1.02 for row in dc[1:]))
    reference = {
        'version': 'V2.0.4', 'passed': passed, 'corner': 'TT', 'temperature': 27,
        'nmos_total_width_m': 61.56e-6, 'pmos_total_width_m': 92.34e-6,
        'length_m': .05e-6, 'transient_load_f': 550e-15,
        'dc_vgs_v': 1.0, 'dc_vds_v': .5,
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'xyce_binary_sha256': hashlib.sha256(Path(xyce).read_bytes()).hexdigest(),
        'source': 'https://github.com/Xyce/Xyce/blob/master/src/DeviceModelPKG/OpenModels/N_DEV_MOSFET_B4p82.C',
        'transient': transient, 'dc': dc,
    }
    (root / 'gate_finger_reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False))
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/qualification/gate_fingers'))
    parser.add_argument('--xyce', default='Xyce')
    args = parser.parse_args()
    xyce = shutil.which(args.xyce)
    if xyce is None:
        parser.error('Xyce executable not found')
    result = characterize(args.output_dir, xyce)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
