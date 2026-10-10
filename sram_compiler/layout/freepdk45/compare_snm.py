"""Compare compiler DC butterfly curves with layout-derived cell coupons.

Coupons retain the selected cell's MOS geometry and the complete resistor
components incident on its terminals, including remote WL/BL bias access and
VDD/VSS meshes. Other MOS are removed: this is embedded-cell DC stability,
not a full macro operating-state or mismatch qualification. Four inverter
gates are disconnected for the compiler's rotated butterfly stimulus.
Capacitances have no DC current and are omitted from these DC-only decks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from sram_compiler.calibrated_rc import terminal_map
from sram_compiler.parasitics import read_dspf
from sram_compiler.testbenches.snm import (
    extract_classical_snm_from_run, extract_write_snm_from_run,
    split_xyce_prn_runs,
)
from utils.xyce import execute_xyce
from .reduce_rc import reduce, sha256
from .full_source import flatten


OPERATIONS = ('hold_snm', 'read_snm', 'write_snm')


def coupon(network, row, col, operation, model_path):
    rows, cols = network['scope']['rows'], network['scope']['cols']
    if operation not in OPERATIONS or not 0 <= row < rows or not 0 <= col < cols:
        raise ValueError('Invalid DC coupon operation or location')
    prefix = f'XSRAM_6T_CORE_{rows}X{cols}/XSRAM_6T_CELL_{row}_{col}/'
    devices = []
    for entry in network['mos']:
        name=entry[0]
        if name[:2].lower()=='mm':name=name[1:]
        if name[:2].lower()=='mx':name=name[1:]
        name = re.sub(r'@\d+$', '', name).upper()
        if name.startswith(prefix):
            devices.append((name.rsplit('/', 1)[1], list(entry)))
    if len(devices) != 6:
        raise ValueError('Coupon requires all six unfingered physical cell MOS')
    by_name = dict(devices)
    graph = defaultdict(set)
    for a,b,_ in network['resistors']:
        graph[a].add(b);graph[b].add(a)
    selected = {node for _,entry in devices for node in entry[1:5]}
    pending = deque(selected)
    while pending:
        for node in graph[pending.popleft()] - selected:
            selected.add(node);pending.append(node)
    resistors = [(a,b,r) for a,b,r in network['resistors'] if a in selected and b in selected]
    terms = terminal_map(network)
    manifest = network['layout_manifest']

    def bias_terminal(logical, instance):
        candidates = [d for d in manifest['source_devices']
                      if d['name'].upper().startswith(instance.upper()+'/')
                      and d['nodes'][0].upper()==logical.upper()]
        if not candidates:
            raise ValueError(f'Missing remote DC bias terminal for {logical}')
        return terms[min(candidates,key=lambda d:d['name'])['name'].upper()]['D']

    core = f'XSRAM_6T_CORE_{rows}x{cols}'
    instances = manifest.get('source_instances', {})
    wl_logical = (instances[core]['pins'][f'WL{row}'] if core in instances else
                  manifest['aliases'][f'{core}/WL{row}'])
    wl = bias_terminal(wl_logical, f'XWL_DRV_{row}')
    bl = terms[f'XPRECHARGE_{col}/M1']['D']
    blb = terms[f'XPRECHARGE_{col}/M2']['D']
    if not {wl, bl, blb, 'VDD', 'VSS'} <= selected:
        raise ValueError('Remote biases must connect to the physical cell resistor components')
    qout, qbout = by_name['MPDL'][1], by_name['MPDR'][1]
    for name,entry in devices:
        if name in ('MPDL','MPUL'):
            entry[2] = 'QB_FORCE'
        elif name in ('MPDR','MPUR'):
            entry[2] = 'Q_FORCE'
    lines = [f'* {rows}x{cols} {operation} coupon row={row} col={col}',
             f'.include "{Path(model_path).resolve()}"', '.TEMP 25', '.PARAM U=0',
             'VDD_BIAS VDD 0 1', 'VSS_BIAS VSS 0 0',
             f'VWL_BIAS {wl} 0 {0 if operation=="hold_snm" else 1}',
             f'VBL_BIAS {bl} 0 1',
             f'VBLB_BIAS {blb} 0 {0 if operation=="write_snm" else 1}']
    lines += [' '.join(entry) for _,entry in devices]
    lines += [f'Rcoupon_{index} {a} {b} {r:.15g}' for index,(a,b,r) in enumerate(resistors)]
    lines += [f"EV1 V1 0 VOL='U+sqrt(2)*V({qbout})'",
              f"EV2 V2 0 VOL='-U+sqrt(2)*V({qout})'",
              "EQ Q_FORCE 0 VOL='U/sqrt(2)+V(V1)/sqrt(2)'",
              "EQB QB_FORCE 0 VOL='-U/sqrt(2)+V(V2)/sqrt(2)'",
              '.OPTIONS LINSOL TYPE=KLU', '.DC U -0.71 0.71 0.001',
              '.PRINT DC FORMAT=NOINDEX {U} V(V1) V(V2)', '.END', '']
    return '\n'.join(lines), dict(row=row,col=col,mos=6,resistors=len(resistors),
             capacitance_policy='DC-only; all capacitors carry zero steady-state current',
             remote_bias_nodes=dict(wl=wl,bl=bl,blb=blb,vdd='VDD',vss='VSS'),
             output_terminals=dict(q=qout,qb=qbout),
             feedback='inverter gates opened with the compiler rotated butterfly convention',
             other_macro_mos='removed; no active periphery or other-cell DC current')


def curve(path, operation):
    runs=split_xyce_prn_runs(str(path))
    if len(runs)!=1 or len(runs[0])!=1421:
        raise ValueError('DC comparison requires one complete 1421-point sweep')
    frame=runs[0]
    if not np.isfinite(frame[['U','V1','V2']].to_numpy()).all():
        raise ValueError('Nonfinite DC curve')
    if not np.allclose(frame.U.to_numpy(),np.linspace(-.71,.71,1421),rtol=0,atol=1e-8):
        raise ValueError('DC sweep must cover -0.71 to 0.71 V in 1 mV increments')
    result=(extract_write_snm_from_run(frame,vdd=1.) if operation=='write_snm'
            else extract_classical_snm_from_run(frame,allow_endpoint_fallback=False))
    return frame,result


def compare(output):
    output=Path(output)
    results={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,operation in zip(axes,OPERATIONS):
        curves={};metrics={}
        for kind,color,style in [('compiler','#999999','--'),('estimated','#d95f02','-'),('post','#1f78b4',':')]:
            frame,metric=curve(output/kind/operation/'deck.sp.prn',operation)
            curves[kind]=frame;metrics[kind]=metric
            root2=np.sqrt(2.)
            axis.plot((frame.V2-frame.U)/root2,(frame.V2+frame.U)/root2,
                      style,color=color,label=kind,linewidth=1.7)
            axis.plot((frame.V1-frame.U)/root2,(frame.V1+frame.U)/root2,
                      style,color=color,linewidth=1.7)
        truth=curves['post']
        for kind in ('compiler','estimated'):
            values=curves[kind]
            if not np.allclose(values.U,truth.U,rtol=0,atol=1e-12):
                raise ValueError('Mismatched DC sweep grids')
            metrics[kind]['curve_rmse_v']=float(np.sqrt(np.mean(
                (values[['V1','V2']].to_numpy()-truth[['V1','V2']].to_numpy())**2)))
            metrics[kind]['snm_error_mv']=float(1000*abs(metrics[kind]['snm']-metrics['post']['snm']))
        results[operation]=metrics
        axis.set_title(operation.replace('_',' '));axis.set_xlabel('Input / reflected output (V)');axis.set_ylabel('Output / reflected input (V)')
        axis.set_xlim(-.05,1.05);axis.set_ylim(-.05,1.05)
        axis.grid(alpha=.2);axis.legend();axis.set_aspect('equal',adjustable='box')
    figure.suptitle('DC cell coupons: compiler versus reduced RC versus full extracted resistance\nTT · 1 V · 25 °C · remote WL/BL/supply bias; other MOS removed')
    figure.tight_layout();figure.savefig(output/'snm_comparison.png',dpi=180)
    figure.savefig(output/'snm_comparison.pdf');plt.close(figure)
    (output/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--extraction-dir',required=True,type=Path)
    parser.add_argument('--compiler-dir',required=True,type=Path,
                        help='contains hold_snm/read_snm/write_snm/deck.sp')
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--row',type=int)
    parser.add_argument('--col',type=int)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--xyce',default='Xyce')
    parser.add_argument('--timeout',type=int,default=2400)
    args=parser.parse_args(argv)
    content=args.profile.read_bytes()
    profile=json.loads(gzip.decompress(content) if args.profile.suffix=='.gz' else content)
    if profile.get('schema')==2:
        from sram_compiler.per_net_rc import assemble
        assemble(profile,args.profile.parent)
    scope=profile['scope']
    if (scope.get('corner')!='TT' or scope.get('vdd')!=1.
            or scope.get('temperature')!=25 or scope.get('variation_mode')!='nominal'
            or sha256(args.models)!=profile.get('model_sha256')):
        raise ValueError('DC coupons require the recorded nominal TT, 1 V, 25 C model')
    path=args.extraction_dir/'xrc/full_macro.dspf'
    receipt=profile['extraction_receipt']
    if receipt.get('drc_violations')!=0 or receipt.get('lvs')!='CORRECT' or sha256(path)!=receipt['dspf_sha256']:
        raise ValueError('DC reference requires matching verified extraction')
    full=copy.deepcopy(profile);full.update(reduce(read_dspf(path),0))
    row=profile['scope']['rows']-1 if args.row is None else args.row
    col=profile['scope']['cols']-1 if args.col is None else args.col
    args.output_dir.mkdir(parents=True,exist_ok=False)
    provenance=dict(scope=__doc__,rows=profile['scope']['rows'],cols=profile['scope']['cols'],
                    row=row,col=col,corner='TT',vdd=1.,temperature=25,variation_mode='nominal',
                    model_sha256=sha256(args.models),dspf_sha256=sha256(path),
                    profile_sha256=sha256(args.profile),full_release_screen=False)
    directories=[]
    for kind,network in [('compiler',None),('estimated',profile),('post',full)]:
        for operation in OPERATIONS:
            directory=args.output_dir/kind/operation;directory.mkdir(parents=True)
            if network is None:
                source=args.compiler_dir/operation/'deck.sp'
                compiler=flatten(source)
                expected={d['name'].rsplit('/',1)[-1]:(d['model'],d['width_m'],d['length_m'],d['fingers'])
                          for d in profile['layout_manifest']['source_devices']
                          if d['name'].startswith(f'XSRAM_6T_CORE_{scope["rows"]}x{scope["cols"]}/XSRAM_6T_CELL_{row}_{col}/')}
                actual={d['name'].rsplit('/',1)[-1]:(d['model'],d['width_m'],d['length_m'],d['fingers'])
                        for d in compiler.mos}
                if len(compiler.mos)!=6 or actual!=expected:
                    raise ValueError('Compiler SNM device geometry differs from the physical cell')
                text=source.read_text();detail=dict(source_sha256=sha256(source),source='compiler single-cell SNM testbench')
                text=re.sub(r'(?im)^\.OPTIONS\s+LINSOL[^\n]*\n?','',text)
                text=text.replace('\n.END','\n.OPTIONS LINSOL TYPE=KLU\n.END')
            else:
                text,detail=coupon(network,row,col,operation,args.models)
            (directory/'deck.sp').write_text(text)
            (directory/'provenance.json').write_text(json.dumps(dict(provenance,**detail),indent=2)+'\n')
            directories.append(directory.resolve())
    if args.execute:
        solver=shutil.which(args.xyce)
        if solver is None:raise ValueError('Xyce unavailable')
        def run(directory):
            deck=directory/'deck.sp';start=time.monotonic()
            result=execute_xyce(deck,[solver,str(deck)],log_path=directory/'xyce.log',timeout=args.timeout,
                               env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            execution=dict(returncode=result.returncode,elapsed_seconds=time.monotonic()-start,
                           solver_sha256=sha256(solver),deck_sha256=sha256(deck))
            (directory/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
            return result.returncode
        with ThreadPoolExecutor(max_workers=3) as pool:
            codes=list(pool.map(run,directories))
        if any(codes):raise RuntimeError('DC simulation failed; artifacts remain unqualified')
        results=compare(args.output_dir)
        print(json.dumps({op:{k:m['snm'] for k,m in groups.items()} for op,groups in results.items()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
