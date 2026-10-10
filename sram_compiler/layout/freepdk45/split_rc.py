"""Export separate per-net R/C settings and explicit VDD/VSS power meshes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path


def write(profile, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    nets=defaultdict(lambda: {'resistors':[],'capacitors':[]})
    logical=profile['node_logical_net']
    def owner(node):
        if node not in logical:raise ValueError(f'Missing RC net identity: {node}')
        return logical[node]
    for index,(a,b,r) in enumerate(profile['resistors']):
        name=owner(a)
        if owner(b)!=name:raise ValueError('Resistor crosses independent signal nets')
        nets[name]['resistors'].append((index,a,b,r))
    coupling=[]
    for index,(a,b,c) in enumerate(profile['capacitors']):
        if a=='0' or b=='0':
            name=owner(b if a=='0' else a)
            nets[name]['capacitors'].append((index,a,b,c))
        else:
            na,nb=owner(a),owner(b)
            if na==nb:nets[na]['capacitors'].append((index,a,b,c))
            else:coupling.append((index,a,b,c,na,nb));nets[na];nets[nb]
    metadata={k:copy.deepcopy(v) for k,v in profile.items()
              if k not in ('resistors','capacitors','net_models') and not k.startswith('_')}
    metadata.update(schema=2,model_family='independent per-net RC models and coupled power meshes',
                    resistor_count=len(profile['resistors']),capacitor_count=len(profile['capacitors']))
    directory=output/'nets';directory.mkdir();settings={}
    def save(path,data):
        path.write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for index,name in enumerate(sorted(nets)):
        filename=f'nets/net_{index:04d}.json.gz';data=dict(nets[name],net=name)
        digest=save(output/filename,data)
        settings[name]=dict(model='power_mesh' if name.upper() in ('VDD','VSS') else 'distributed_signal_network',
                            file=filename,sha256=digest,resistance_scale=1.,capacitance_scale=1.,
                            resistor_count=len(data['resistors']),capacitor_count=len(data['capacitors']))
    metadata['net_models']=settings
    digest=save(output/'coupling.json.gz',dict(capacitors=coupling))
    metadata['coupling_model']=dict(file='coupling.json.gz',sha256=digest,capacitor_count=len(coupling))
    save(output/'profile.json.gz',metadata)
    return metadata


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    profile=json.loads(args.profile.read_text());metadata=write(profile,args.output_dir)
    print('Exported',len(metadata['net_models']),'independent RC models, including VDD and VSS meshes')
    return 0


if __name__=='__main__':raise SystemExit(main())
