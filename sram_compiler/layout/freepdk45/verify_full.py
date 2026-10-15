"""Calibre verification and extraction of a generated macro or component."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence

from .reduce_rc import sha256
from sram_compiler.parasitics import read_dspf


def run_command(arguments: Sequence[str], directory: Path, log: str,
                environment: dict[str, str]) -> None:
    with (directory / log).open('w') as stream:
        subprocess.run(arguments, cwd=directory, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-dir',required=True,type=Path)
    parser.add_argument('--pdk-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--drc-only',action='store_true')
    parser.add_argument('--extraction-workers',type=int,
                        help='explicit worker count for xRC phdb/pdb; default leaves Calibre scheduling unchanged')
    args=parser.parse_args(argv)
    if args.extraction_workers is not None and args.extraction_workers<1:
        parser.error('Extraction worker count must be positive')
    layout=args.layout_dir.resolve();output=args.output_dir.resolve();output.mkdir(exist_ok=False,parents=True)
    pdk=args.pdk_dir.resolve();rules=pdk/'ncsu_basekit/techfile/calibre'
    env=dict(os.environ,PDK_DIR=str(pdk))
    gds=layout/'full_macro.gds';source=layout/'source.sp'
    manifest=json.loads((layout/'layout.json').read_text())
    primary=manifest['primary']
    component=manifest.get('artifact_kind')=='component'
    bank=manifest.get('artifact_kind')=='bank'
    rows,cols=(None,None) if component else (manifest.get('rows',8),manifest.get('cols',8))
    scope=manifest['scope'] if component or bank else f'complete {rows}x{cols} 6T macro including controller and output latch'
    receipt=dict(status='in_progress',scope=scope,
                 rows=rows,cols=cols,primary=primary,
                 extraction_workers=args.extraction_workers,
                 controller_layout=manifest.get('controller_layout',not (component or bank)),layout_sha256=sha256(gds),source_sha256=sha256(source),
                 rule_sha256={n:sha256(rules/n) for n in ['calibreDRC.rul','calibreLVS.rul','calibrexRC.rul']},
                 lvs_passive_policy='R SHORT SOURCE; C OPEN SOURCE. Physical passive values require separate xRC verification.',
                 full_release_screen=False)
    if component:
        calibre=shutil.which('calibre')
        receipt.update(artifact_kind='component',core_pitch_nm=manifest['core_pitch_nm'],
                       row_span=manifest['row_span'],column_span=manifest['column_span'],
                       compiler_instance=manifest['compiler_instance'],
                       calibre_executable=calibre,calibre_sha256=sha256(Path(calibre)) if calibre else None)
    if bank:
        calibre=shutil.which('calibre')
        receipt.update(artifact_kind='bank',core_pitch_nm=manifest['core_pitch_nm'],
                       included_instances=manifest['included_instances'],
                       excluded_instances=manifest['excluded_instances'],
                       source_deck_sha256=manifest['source_deck_sha256'],
                       generator_sources_sha256=manifest['generator_sources_sha256'],
                       calibre_executable=calibre,calibre_sha256=sha256(Path(calibre)) if calibre else None)
    def save(): (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    common=[f'LAYOUT PATH "{gds}"',f'LAYOUT PRIMARY {primary}','LAYOUT SYSTEM GDSII']
    try:
        drc=output/'drc';drc.mkdir();runset=drc/'runset'
        runset.write_text('\n'.join(common+['DRC RESULTS DATABASE "results" ASCII','DRC SUMMARY REPORT "summary"','DRC MAXIMUM RESULTS ALL',f'INCLUDE "{rules/"calibreDRC.rul"}"','']))
        run_command(['calibre','-drc',str(runset)],drc,'drc.log',env)
        count=re.search(r'TOTAL DRC Results Generated:\s*(\d+)',(drc/'summary').read_text())
        receipt['drc_violations']=int(count[1]) if count else None;save()
        if receipt['drc_violations']!=0: raise ValueError('DRC violations; inspect results and summary')
        if args.drc_only:
            receipt['status']='drc_passed';save();return 0
        xrc=output/'xrc';xrc.mkdir();runset=xrc/'runset'
        runset.write_text('\n'.join(common+[f'SOURCE PATH "{source}"',f'SOURCE PRIMARY {primary}','SOURCE SYSTEM SPICE',
                         'MASK SVDB DIRECTORY "svdb" QUERY XRC','PEX NETLIST "full_macro.dspf" DSPF SOURCENAMES',
                         f'INCLUDE "{rules/"calibrexRC.rul"}"','LVS FILTER R SHORT SOURCE','LVS FILTER C OPEN SOURCE','']))
        run_command(['calibre','-lvs','-hier','-spice','layout.sp',str(runset)],xrc,'lvs.log',env)
        report=(xrc/'mask.lvs.rep').read_text()
        shutil.copyfile(xrc/'mask.lvs.rep',output/'lvs.report')
        if 'INCORRECT' in report or not re.search(r'\bCORRECT\b',report):raise ValueError('LVS failed; physical passives remain independently checked')
        receipt['lvs']='CORRECT';save()
        for stage,flags in [('phdb',['-phdb']),('pdb',['-pdb','-rcc']),('fmt',['-fmt','-all'])]:
            workers=(['-turbo',str(args.extraction_workers)]
                     if args.extraction_workers is not None and stage!='fmt' else [])
            run_command(['calibre','-xrc',*flags,*workers,str(runset)],xrc,stage+'.log',env)
        dspf=xrc/'full_macro.dspf';parsed=read_dspf(dspf)
        receipt.update(status='extracted',dspf_sha256=sha256(dspf),nets=len(parsed.nets),
                       capacitance=parsed.capacitance_inventory(),extraction='distributed R+C+CC')
        save()
    except Exception as error:
        receipt.update(status='failed',error=str(error));save();raise
    print(output/'receipt.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
