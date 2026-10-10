"""Generate and verify the fixed 8x8 physical calibration reference.

Run from the repository root. Foundry/PDK rule files remain external; generated
layouts, netlists and tool logs belong under outputs/ or a temporary directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from sram_compiler.parasitics import read_dspf
from . import top
from .export import flat_devices, parse

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_command(arguments, directory, log, environment):
    with (directory / log).open('w') as stream:
        subprocess.run(arguments, cwd=directory, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--deck', required=True, type=Path, help='nominal, mode-0 8x8 write deck')
    parser.add_argument('--pdk-dir', type=Path, default=os.environ.get('PDK_DIR'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/rc_calibration/8x8'))
    parser.add_argument('--calibre', default='calibre')
    parser.add_argument('--extract', action='store_true', help='run Calibre DRC, LVS and coupled xRC')
    parser.add_argument('--array-only', action='store_true', help='extract the bitcell tile without periphery')
    args = parser.parse_args(argv)
    deck = args.deck.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; previous extraction evidence is preserved')
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='in_progress', time_utc=datetime.now(timezone.utc).isoformat(),
                   source_deck_sha256=sha256(deck), rows=8, cols=8,
                   cell='SRAM_6T_CELL', variation_mode='nominal', equivalent_mode=0,
                   controller_layout=False, full_layout=False, defaults_promotable=False,
                   scope='array' if args.array_only else 'macro')
    source_summary = deck.with_name('summary.json')
    if source_summary.is_file():
        metadata = json.loads(source_summary.read_text())
        receipt['source_metadata'] = {key: metadata.get(key) for key in
                                      ('compiler_version', 'model_sha256', 'corner', 'vdd', 'temperature',
                                       'variation_mode', 'real_cell_mode', 'seed', 'interconnect', 'time_control')}
    receipt_path = output / 'receipt.json'

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    save()
    try:
        reference = output / 'schematic.sp'
        run_command([sys.executable, '-m', 'sram_compiler.layout.freepdk45.export',
                     str(deck), str(reference)], ROOT,
                    str(output / 'export.log'), dict(os.environ))
        # The layout is intentionally fixed. The official LVS checks widths,
        # lengths and topology against the supplied deck, not a bundled source.
        if '.subckt SRAM_8X8 ' not in reference.read_text():
            raise ValueError('The physical reference supports only an 8x8 6T macro')
        tree = parse(reference.read_text().splitlines())
        inventory = flat_devices(tree.resolve('SRAM_8X8').lines, tree)
        if sum(inventory.values()) != 876 or {key[0] for key in inventory} != {'NMOS_VTG', 'PMOS_VTG'}:
            raise ValueError('The physical reference requires 876 nominal VTG MOS in full mode 0')
        from .lib import pya
        receipt['klayout_library_version'] = getattr(pya, '__version__', 'unknown')
        primary = 'ARRAY8X8' if args.array_only else 'SRAM_8X8'
        if args.array_only:
            from . import array
            gds = array.write(output, reference)
        else:
            top.write(output)
            gds = output / 'sram8x8.gds'
        receipt.update(layout_sha256=sha256(gds), schematic_sha256=sha256(reference),
                       generator_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
        if args.extract:
            if args.pdk_dir is None:
                raise ValueError('--pdk-dir or PDK_DIR is required for Calibre extraction')
            calibre = shutil.which(args.calibre)
            if calibre is None:
                raise ValueError(f'Calibre executable not found: {args.calibre}')
            pdk = args.pdk_dir.resolve()
            rules = pdk / 'ncsu_basekit/techfile/calibre'
            receipt['rule_sha256'] = {n: sha256(rules / n) for n in
                                     ('calibreDRC.rul', 'calibreLVS.rul', 'calibrexRC.rul')}
            receipt['calibre_binary_sha256'] = sha256(Path(calibre).resolve())
            env = dict(os.environ, PDK_DIR=str(pdk))
            layout = [f'LAYOUT PATH "{gds}"', f'LAYOUT PRIMARY "{primary}"', 'LAYOUT SYSTEM GDSII']
            drc = output / 'drc'
            drc.mkdir()
            drc_run = drc / 'runset'
            drc_run.write_text('\n'.join(layout + [
                'DRC RESULTS DATABASE "results" ASCII', 'DRC SUMMARY REPORT "summary"',
                'DRC MAXIMUM RESULTS ALL', f'INCLUDE "{rules / "calibreDRC.rul"}"', '']))
            run_command([calibre, '-drc', str(drc_run)], drc, 'drc.log', env)
            import re
            summary = (drc / 'summary').read_text()
            count = re.search(r'TOTAL DRC Results Generated:\s*(\d+)', summary)
            if count is None or int(count[1]):
                raise ValueError('Calibre DRC did not produce a zero-violation summary')
            receipt['calibre_drc_violations'] = 0
            xrc = output / 'xrc'
            xrc.mkdir()
            runset = xrc / 'runset'
            runset.write_text('\n'.join(layout + [f'SOURCE PATH "{reference}"',
                f'SOURCE PRIMARY "{primary}"', 'SOURCE SYSTEM SPICE',
                'MASK SVDB DIRECTORY "svdb" QUERY XRC',
                'PEX NETLIST "macro.dspf" DSPF SOURCENAMES',
                f'INCLUDE "{rules / "calibrexRC.rul"}"', '']))
            run_command([calibre, '-lvs', '-hier', '-spice', 'layout.sp', str(runset)], xrc, 'lvs.log', env)
            report = xrc / 'mask.lvs.rep'
            text = report.read_text()
            if 'INCORRECT' in text or not re.search(r'\bCORRECT\b', text):
                raise ValueError('Calibre LVS did not certify source/layout equivalence')
            shutil.copyfile(report, output / 'calibre_lvs.report')
            receipt['calibre_lvs'] = 'CORRECT'
            save()
            for stage, flags in [('phdb', ['-phdb']), ('pdb', ['-pdb', '-rcc']), ('fmt', ['-fmt', '-all'])]:
                run_command([calibre, '-xrc', *flags, str(runset)], xrc, stage + '.log', env)
            dspf = xrc / 'macro.dspf'
            parsed = read_dspf(dspf)
            receipt.update(dspf_sha256=sha256(dspf), extraction='distributed R+C+CC',
                           parasitic_nets=len(parsed.nets), ports=list(parsed.ports),
                           capacitance_inventory=parsed.capacitance_inventory())
        receipt['status'] = 'macro_extracted' if args.extract else 'layout_generated_unverified'
        save()
    except Exception as error:
        receipt.update(status='failed', error=str(error))
        save()
        raise
    print(receipt_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
