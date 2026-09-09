"""Wordline wire model check: the compiler's per-pin stubs (a star) against a distributed line.

    python3 -m sram_compiler.sizing.wordline_model --xyce /path/to/Xyce

For each column count the array's RC wordline driver (fixed-mode scale) drives
real 6T cells whose WL pins carry the compiler's 100 ohm / 1 fF stub. ``star`` is
the compiler netlist (an ideal row net between the cells); ``unit`` chains the
same 100 ohm / 1 fF per column pitch; ``pitch`` chains a metal-pitch estimate
(defaults 1 ohm and 0.1 fF per column). Arrival from the driver input and the
10-90 % slew are measured at the first and at the last cell gate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_pF  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export

from per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.subcircuits.sram_6t_core import Sram6TCell
from sram_compiler.testbenches.parameter_factor import WordlineDriverFactory

from .qualification import read_measurements

MEASURES = ('T_NEAR', 'T_FAR', 'SLEW_NEAR', 'SLEW_FAR', 'T_FAR_FALL')


def build_deck(cols, variant, series_res=0.0, shunt_cap=0.0, vdd=1.0, temperature=25):
    """Deck text and the driver scales; `variant` is star, unit or pitch."""
    if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
        raise ValueError('cols must be a positive integer')
    if variant != 'star' and (series_res <= 0 or shunt_cap <= 0):
        raise ValueError('Distributed variants need positive series_res and shunt_cap')
    cfg = load_config(16, cols, 'TT')
    sizes = resolve_driver_sizes(cfg, mux=False)
    wl, cell_cfg = cfg.wordline_driver, cfg.sram_6t_cell
    with contextlib.redirect_stdout(io.StringIO()):
        driver = WordlineDriverFactory(
            wl.nmos_model.value[0], wl.pmos_model.value[0],
            nand_nmos_width=wl.nmos_width.value[0], nand_pmos_width=wl.pmos_width.value[0],
            inv_nmos_width=wl.nmos_width.value[1], inv_pmos_width=wl.pmos_width.value[1],
            length=wl.length.value, num_cols=cols, w_rc=True,
            inverter_scale=sizes.wl_inv, nand_gate_scale=sizes.wl_nand,
        ).create()
        cell = Sram6TCell(
            cell_cfg.nmos_model.value[0], cell_cfg.pmos_model.value, cell_cfg.nmos_model.value[1],
            cell_cfg.nmos_width.value[0], cell_cfg.pmos_width.value, cell_cfg.nmos_width.value[1],
            cell_cfg.length.value, w_rc=True, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        )
        circuit = Circuit(f'wordline model {variant} {cols} columns')
        circuit.include(cfg.global_config.pdk_path_TT)
        circuit.subcircuit(driver)
        circuit.subcircuit(cell)
        circuit.V('VDD', 'VDD', circuit.gnd, vdd)
        circuit.PulseVoltageSource('A', 'A', circuit.gnd, initial_value=0, pulsed_value=vdd,
                                   delay_time=1e-9, rise_time=20e-12, fall_time=20e-12,
                                   pulse_width=2e-9, period=5e-9)
        # Decoder input A pulses, wl_en (B) stays high: the driver output is WL_0.
        circuit.X('DRV', driver.name, 'VDD', circuit.gnd, 'A', 'VDD', 'WL_0')
        for col in range(cols):
            node = 'WL_0' if variant == 'star' else f'WL_{col}'
            circuit.X(f'C{col}', cell.name, 'VDD', circuit.gnd, 'VDD', 'VDD', node)
            if variant != 'star' and col < cols - 1:
                circuit.R(f'W{col}', f'WL_{col}', f'WL_{col + 1}', series_res @ u_Ohm)
                circuit.C(f'W{col}', f'WL_{col + 1}', circuit.gnd, shunt_cap)
    near, far = 'XC0:WL_end', f'XC{cols - 1}:WL_end'
    circuit.raw_spice += '\n'.join([
        f'.OPTIONS DEVICE TEMP={temperature}', '.OPTIONS MEASURE MEASFAIL=1', '.TRAN 1p 4n',
        f'.MEASURE TRAN T_NEAR TRIG V(A) VAL={0.5 * vdd} RISE=1 TARG V({near}) VAL={0.5 * vdd} RISE=1',
        f'.MEASURE TRAN T_FAR TRIG V(A) VAL={0.5 * vdd} RISE=1 TARG V({far}) VAL={0.5 * vdd} RISE=1',
        f'.MEASURE TRAN SLEW_NEAR TRIG V({near}) VAL={0.1 * vdd} RISE=1 TARG V({near}) VAL={0.9 * vdd} RISE=1',
        f'.MEASURE TRAN SLEW_FAR TRIG V({far}) VAL={0.1 * vdd} RISE=1 TARG V({far}) VAL={0.9 * vdd} RISE=1',
        f'.MEASURE TRAN T_FAR_FALL TRIG V(A) VAL={0.5 * vdd} FALL=1 TARG V({far}) VAL={0.5 * vdd} FALL=1',
        '']) + '\n'
    # PySpice emits raw_spice before the elements, so the terminator goes last.
    return str(circuit) + '.END\n', sizes


def characterize(output_root, xyce, columns=(16, 256), pitch_res=1.0, pitch_cap=1e-16):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    variants = {'star': (0.0, 0.0), 'unit': (100.0, 1e-15), 'pitch': (pitch_res, pitch_cap)}
    jobs = [(cols, name) for cols in columns for name in variants]

    def run(job):
        cols, name = job
        series_res, shunt_cap = variants[name]
        directory = root / f'{cols}_{name}'
        directory.mkdir(exist_ok=True)
        deck, sizes = build_deck(cols, name, series_res, shunt_cap)
        path = directory / 'deck.sp'
        path.write_text(deck)
        environment = dict(os.environ, OMP_NUM_THREADS='1', KOKKOS_NUM_THREADS='1')
        with (directory / 'xyce.log').open('w') as log:
            code = subprocess.run([xyce, '-linsolv', 'KLU', '-o', str(path), str(path)],
                                  stdout=log, stderr=subprocess.STDOUT, env=environment, check=False).returncode
        measures = read_measurements(path.with_name('deck.sp.mt0')) if code == 0 and path.with_name('deck.sp.mt0').exists() else {}
        return {'cols': cols, 'variant': name, 'series_res_ohm': series_res, 'shunt_cap_f': shunt_cap,
                'wl_inv': sizes.wl_inv, 'wl_nand': sizes.wl_nand, 'returncode': code,
                **{key: measures.get(key) for key in MEASURES}}

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    project = Path(__file__).resolve().parents[2]
    summary = {
        'setup': 'TT 25 C 1.0 V; fixed-mode RC wordline driver for 16 x cols; real 6T cells with the '
                 'compiler stub (100 ohm, 1 fF) on every WL pin; bitlines at VDD; A pulses, wl_en high',
        'variants': {'star': 'ideal row net between cells (compiler netlist)',
                     'unit': 'series 100 ohm and shunt 1 fF per column pitch',
                     'pitch': f'series {pitch_res} ohm and shunt {pitch_cap} F per column pitch'},
        'measures': {'T_NEAR': 'A 50 % to first cell gate 50 % (s)', 'T_FAR': 'A 50 % to last cell gate 50 % (s)',
                     'SLEW_NEAR': 'first cell gate 10-90 % (s)', 'SLEW_FAR': 'last cell gate 10-90 % (s)',
                     'T_FAR_FALL': 'A 50 % falling to last cell gate 50 % falling (s)'},
        'model_sha256': hashlib.sha256((project / 'tran_models/models_TT.spice').read_bytes()).hexdigest(),
        'xyce_sha256': hashlib.sha256(Path(xyce).read_bytes()).hexdigest(),
        'results': results,
    }
    (root / 'wordline_model.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'outputs/qualification/wordline_model')
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--cols', default='16,256', help='Comma-separated column counts')
    parser.add_argument('--pitch-res', type=float, default=1.0, help='Series ohm per column pitch for the pitch variant')
    parser.add_argument('--pitch-cap', type=float, default=1e-16, help='Shunt farad per column pitch for the pitch variant')
    args = parser.parse_args()
    xyce = shutil.which(args.xyce)
    if xyce is None:
        parser.error('Xyce executable not found')
    columns = tuple(int(value) for value in args.cols.split(','))
    if any(cols < 1 for cols in columns) or min(args.pitch_res, args.pitch_cap) <= 0:
        parser.error('Column counts and pitch values must be positive')
    summary = characterize(args.output_dir, xyce, columns, args.pitch_res, args.pitch_cap)
    for row in summary['results']:
        cells = ' '.join(f"{key}={row[key] * 1e12:7.1f}ps" if isinstance(row[key], float) else f'{key}=FAILED' for key in MEASURES)
        print(f"{row['cols']:4d} {row['variant']:5s} R={row['series_res_ohm']:5g} C={row['shunt_cap_f']:8g} exit={row['returncode']} {cells}")
    return 0 if all(row['returncode'] == 0 for row in summary['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
