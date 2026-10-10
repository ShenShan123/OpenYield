"""Calibrate a reduced distributed RC model from a verified complete layout.

Retain all transistor/port terminals and all coupling. Merge only degree-two
wire nodes, preserving their DC series resistance and assigning capacitance
to the nearer endpoint at each elimination. Aliases resolve recursively. This is an estimated capacitance placement,
not the full extractor network; waveform error is measured before promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from sram_compiler.parasitics import read_dspf, spice_value


def sha256(path):
    """Keep RC profile loading independent of KLayout's optional library."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduce(dspf, maximum_series_ohm=20.):
    if not isfinite(maximum_series_ohm) or maximum_series_ohm < 0:
        raise ValueError('maximum_series_ohm must be finite and nonnegative')
    graph=defaultdict(dict);protected=set(dspf.ports);mos=[];capacitors=[]
    for line in dspf.statements:
        tokens=line.split()
        if tokens[0][0].lower()=='m' and not tokens[0].startswith('*'):
            mos.append(tokens);protected.update(tokens[1:5])
    for net in dspf.nets.values():
        protected.update(net.ports)
        for a,b,r in net.resistors:
            if a==b:continue
            if r<=0:raise ValueError('Zero resistance requires explicit connectivity reduction')
            previous=graph[a].get(b)
            value=r if previous is None else 1/(1/previous+1/r)
            graph[a][b]=value;graph[b][a]=value
        capacitors.extend(net.capacitors)
    # A physical ground can be global, not a subcircuit port.
    protected.add(dspf.ground)
    aliases={};pending=deque(graph);eliminated=0
    while pending:
        node=pending.popleft()
        if node in protected or node not in graph or len(graph[node])!=2:continue
        (a,ra),(b,rb)=list(graph[node].items())
        if ra+rb>maximum_series_ohm:continue
        aliases[node]=a if ra<=rb else b
        del graph[a][node];del graph[b][node];del graph[node]
        total=ra+rb
        prior=graph[a].get(b)
        combined=total if prior is None else 1/(1/prior+1/total)
        graph[a][b]=combined;graph[b][a]=combined
        pending.extend((a,b));eliminated+=1
    def canonical(node):
        trail=[]
        while node in aliases:
            trail.append(node);node=aliases[node]
        for old in trail:aliases[old]=node
        return node
    caps=defaultdict(float);discarded=0.
    for a,b,c in capacitors:
        a,b=canonical(a),canonical(b)
        if a==b:discarded+=c;continue
        caps[tuple(sorted((a,b)))]+=c
    edges=[]
    for a,neighbors in graph.items():
        for b,r in neighbors.items():
            if a<b:edges.append((a,b,r))
    return dict(ports=list(dspf.ports),mos=mos,resistors=edges,
                capacitors=[(a,b,c) for (a,b),c in caps.items()],node_aliases=aliases,
                eliminated_wire_nodes=eliminated,discarded_same_node_capacitance_f=discarded,
                maximum_series_ohm=maximum_series_ohm)


def render(profile,name=None):
    if name is None:
        scope=profile.get('scope',{})
        name=f'CALIBRATED_OPENYIELD_{scope.get("rows",8)}X{scope.get("cols",8)}'
    lines=[f'.subckt {name} '+' '.join(profile['ports'])]
    lines.extend(' '.join(m) for m in profile['mos'])
    lines.extend(f'Rcal_{i} {a} {b} {r:.15g}' for i,(a,b,r) in enumerate(profile['resistors']))
    lines.extend(f'Ccal_{i} {a} {b} {c:.15g}' for i,(a,b,c) in enumerate(profile['capacitors']))
    lines.extend([f'.ends {name}',''])
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-dir',required=True,type=Path)
    parser.add_argument('--layout-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--maximum-series-ohm',type=float,default=20.)
    args=parser.parse_args(argv)
    receipt=json.loads((args.extraction_dir/'receipt.json').read_text())
    path=args.extraction_dir/'xrc/full_macro.dspf'
    if receipt.get('status')!='extracted' or receipt.get('drc_violations')!=0 or receipt.get('lvs')!='CORRECT':
        raise ValueError('Full layout DRC/LVS and coupled extraction must pass before calibration')
    if sha256(path)!=receipt['dspf_sha256']:raise ValueError('Extraction hash mismatch')
    if sha256(args.layout_dir/'full_macro.gds') != receipt['layout_sha256']:
        raise ValueError('Layout hash mismatch')
    if sha256(args.layout_dir/'source.sp') != receipt['source_sha256']:
        raise ValueError('Schematic hash mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    dspf=read_dspf(path);profile=reduce(dspf,args.maximum_series_ohm)
    manifest=json.loads((args.layout_dir/'layout.json').read_text())
    source_devices={d['name'].upper():d for d in manifest['source_devices']}
    node_logical={}
    for net_name,net in dspf.nets.items():
        logical=net_name
        for node,instance,terminal in net.pins:
            key=instance[1:] if instance.startswith('MX') else instance
            key=re.sub(r'@\d+$','',key).upper()
            if key in source_devices:
                logical=source_devices[key]['nodes']['dgsb'.index(terminal.lower())]
                break
        for a,b,r in net.resistors:
            node_logical[a]=logical;node_logical[b]=logical
        for p in net.ports:node_logical[p]=logical
        for node,_,_ in net.pins:node_logical[node]=logical
    rows,cols=manifest.get('rows',8),manifest.get('cols',8)
    profile.update(schema=1,scope=dict(rows=rows,cols=cols,cell='SRAM_6T_CELL',mode=0,target_col=cols-1,
                                     variation_mode='nominal',hardware_operation='read&write'),
                   extraction_receipt=receipt,source_dspf_sha256=sha256(path),
                   layout_manifest=manifest,node_logical_net=node_logical,
                   method='Calibrated distributed RC; degree-two series-node reduction and nearest-end capacitance placement',
                   waveform_validated=False)
    (args.output_dir/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (args.output_dir/'estimated_macro.sp').write_text(render(profile))
    print('Eliminated wire nodes',profile['eliminated_wire_nodes'],'R',len(profile['resistors']),'C',len(profile['capacitors']))
    return 0


if __name__=='__main__':raise SystemExit(main())
