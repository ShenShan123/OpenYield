"""Sense-amplifier input-offset ensemble for V2.0.5 qualification.

One transient sweep drives 100 independent SA/latch copies. Each copy keeps the
same per-device random parameters throughout all differential trials, so the
decision threshold is bracketed for each sample, not inferred from unrelated
Monte Carlo runs at different input voltages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PySpice.Spice.Netlist import Circuit

from per_device_mc.netlist import specialize_netlist
from per_device_mc.run import load_config
from sram_compiler.sizing.qualification import read_measurements
from sram_compiler.subcircuits.standard_cell import D_latch
from sram_compiler.testbenches.parameter_factor import SenseAmpFactory


def characterize_offset(output_root, xyce, *, corner='SS', temperature=125, vdd=0.9,
                        samples=100, seed=2026, timeout=7200, profile='mid'):
    if samples < 2 or seed < 1:
        raise ValueError('Offset characterization requires at least two samples and a positive seed')
    if profile not in ('mid', 'rail'):
        raise ValueError('Offset profile must be mid or rail')
    suffix = '_rail' if profile == 'rail' else ''
    directory = Path(output_root).resolve() / f'offset_{corner}_{temperature}_{vdd}{suffix}'
    directory.mkdir(parents=True, exist_ok=True)
    logging.getLogger('PySpice').setLevel(logging.ERROR)
    deltas = np.linspace(-0.3, 0.3, 25)
    with (directory / 'generation.log').open('w') as log, redirect_stdout(log):
        cfg = load_config(16, 16, corner)
        sa = cfg.senseamp
        assert sa is not None, 'Sense amplifier configuration was not loaded'
        circuit = Circuit('V2.0.5 local SA offset ensemble')
        model = Path(getattr(cfg.global_config, f'pdk_path_{corner}'))
        circuit.include(model)
        circuit.V('DD', 'VDD', 0, vdd)
        amplifier = SenseAmpFactory(sa.nmos_model.value, sa.pmos_model.value,
                                    sa.nmos_width.value, sa.pmos_width.value, sa.length.value).create()
        latch = D_latch(nmos_model='NMOS_VTG', pmos_model='PMOS_VTG')
        circuit.subcircuit(amplifier)
        circuit.subcircuit(latch)
        for sample in range(samples):
            circuit.X(f'SA{sample}', amplifier.NAME, 'VDD', 0, 'EN', 'EN', 'IN', 'INB', f'Q{sample}', f'QB{sample}')
            circuit.X(f'LATCH{sample}', latch.NAME, 'VDD', 0, f'Q{sample}', 'EN', f'OUT{sample}', f'OUTB{sample}')
        raw = [f'.TEMP {temperature}', '.OPTIONS MEASURE MEASFAIL=1',
               f'VEN EN 0 PULSE(0 {vdd} 1n 10p 10p 1n 4n)']
        for name, sign in [('IN', -1), ('INB', 1)]:
            points = []
            for index, delta in enumerate(deltas):
                value = 0.75 * vdd + sign * delta / 2
                if profile == 'rail':
                    # Actual precharged SRAM bitlines: one stays at VDD while
                    # the data-dependent side develops the differential.
                    value = vdd - max(delta, 0) if sign < 0 else vdd + min(delta, 0)
                points.extend([f'{index * 4e-9:.12g} {value:.12g}',
                               f'{(index + 1) * 4e-9 - 1e-12:.12g} {value:.12g}'])
            raw.append(f'V{name} {name} 0 PWL(' + ' '.join(points) + ')')
        for sample in range(samples):
            for index in range(len(deltas)):
                raw.append(f'.MEASURE TRAN Q_{sample}_{index} FIND V(Q{sample}) AT={index * 4e-9 + 1.9e-9:.12g}')
        raw.extend([f'.TRAN 10p {len(deltas) * 4e-9:.12g}', '.END'])
        deck, audit = specialize_netlist(
            str(circuit) + '\n' + '\n'.join(raw) + '\n', base_model_path=model,
            model_output_path=directory / 'models_local.spice', mc_runs=1, vth_std=0.05,
            deck_base_dir=Path(__file__).resolve().parents[2], audit_path=directory / 'model_audit.csv',
        )
    signature = hashlib.sha256(deck.encode() + (directory / 'models_local.spice').read_bytes() + str(seed).encode()).hexdigest()
    marker = directory / 'complete.json'
    deck_path = directory / 'deck.sp'
    deck_path.write_text(deck)
    if not marker.exists() or json.loads(marker.read_text()).get('signature') != signature:
        for path in directory.glob('deck.sp.*'):
            path.unlink()
        with (directory / 'xyce.log').open('w') as log:
            process = subprocess.run([xyce, '-linsolv', 'KLU', '-randseed', str(seed), '-o', str(deck_path), str(deck_path)],
                                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                                     env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        if process.returncode:
            raise RuntimeError(f'Offset Xyce exit {process.returncode}; see {directory / "xyce.log"}')
        marker.write_text(json.dumps({'signature': signature, 'linear_solver': 'KLU'}))
    measurements = read_measurements(directory / 'deck.sp.mt0')
    offsets, errors = [], []
    for sample in range(samples):
        values = [measurements.get(f'Q_{sample}_{index}') for index in range(len(deltas))]
        if any(value is None or not np.isfinite(value) for value in values):
            errors.append(f'Sample {sample}: missing measurements')
            continue
        low = np.asarray(values) < vdd / 2
        flips = np.flatnonzero(low[1:] != low[:-1])
        if low[0] or not low[-1] or len(flips) != 1:
            errors.append(f'Sample {sample}: unbracketed/nonmonotonic offset')
            continue
        index = flips[0]
        offsets.append(float((deltas[index] + deltas[index + 1]) / 2))
    resolution = float(deltas[1] - deltas[0])
    mean = float(np.mean(offsets)) if offsets else None
    sigma = float(np.std(offsets, ddof=1)) if len(offsets) > 1 else None
    bound = abs(mean) + 3 * sigma + resolution / 2 if sigma is not None and mean is not None else None
    result = {'version': 'V2.0.5', 'corner': corner, 'temperature': temperature, 'vdd': vdd,
              'scoring_version': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'profile': profile, 'linear_solver': json.loads(marker.read_text()).get('linear_solver', 'Xyce default'),
                  'samples': samples, 'seed': seed, 'independent_models': audit['unique_mc_models'],
                  'offset_step': resolution, 'offsets': offsets, 'mean': mean, 'sigma': sigma,
                  'conservative_3sigma_offset': bound, 'errors': errors,
                  'passed': len(offsets) == samples and not errors, 'signature': signature}
    (directory / 'offset_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'outputs/qualification/V2.0.5')
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    xyce = shutil.which(args.xyce)
    if xyce is None or args.samples < 2:
        parser.error('A working Xyce executable and at least two samples are required')
    with ProcessPoolExecutor(4) as pool:
        jobs = [pool.submit(characterize_offset, args.output_dir, xyce, corner=corner,
                            temperature=temperature, vdd=vdd, samples=args.samples, profile=profile)
                for corner, temperature, vdd in [('SS', 125, .9), ('FF', -40, 1.0)]
                for profile in ('mid', 'rail')]
        results = [job.result() for job in jobs]
    print(json.dumps([{key: value for key, value in result.items() if key != 'offsets'}
                      for result in results], indent=2))
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
