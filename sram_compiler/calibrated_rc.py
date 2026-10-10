"""Apply a scoped, calibrated distributed RC profile to compiler circuits.

The profile is a reduced network, not the full post-layout DSPF. Its terminal
locations and MOS junction geometry are frozen against a verified layout.
"""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import re

from sram_compiler.layout.freepdk45.reduce_rc import render

ROOT=Path(__file__).resolve().parents[1]


def initial_state(profile,original_conditions,vdd=1.):
    """Resolve the complete schematic's quiescent logic before clock capture.

    All wire nodes receive the same logical guess as their schematic net.
    This prevents startup clock glitches caused by zero-initialized intermediate
    clock-buffer nodes. Values are released for the transient, not clamped.
    """
    from sram_compiler.layout.freepdk45.export import Nets
    manifest=profile['layout_manifest']
    aliases={k.upper():v for k,v in manifest['aliases'].items()}
    values={aliases.get(n.replace(':','/').upper(),n):int(round(float(v)/vdd))
            for n,v in original_conditions.items()}
    values.update(VDD=1,VSS=0,clk=0,csb=0,web=1,PRE=0)
    values.update({f'A{i}':0 for i in range(3)})
    values.update({f'DIN{i}':0 for i in range(8)})
    for _ in range(100):
        nets=Nets()
        for passive in manifest['passives']:
            nets.union(passive['nodes']['in'],passive['nodes']['out'])
        for device in manifest['source_devices']:
            drain,gate,source,body=device['nodes']
            if gate in values and values[gate]==(0 if device['model'].startswith('PMOS') else 1):
                nets.union(drain,source)
        added=0
        for members in nets.groups().values():
            known={values[n] for n in members if n in values}
            if len(known)==1:
                for n in members:
                    if n not in values:values[n]=next(iter(known));added+=1
        if not added:break
    upper={n.upper():v for n,v in values.items()}
    analog = {aliases.get(name.replace(':', '/').upper(), name).upper(): float(value)
              for name, value in original_conditions.items()}
    result={}
    for node,logical in profile['node_logical_net'].items():
        while node in profile['node_aliases']:node=profile['node_aliases'][node]
        result['XPHYSICAL_FULL_MACRO:'+node]=analog.get(
            logical.upper(), vdd*upper.get(logical.upper(),.5))
    for name,value in original_conditions.items():
        result['XPHYSICAL_FULL_MACRO:'+probe_node(profile,name)]=float(value)
    return result


def load_profile(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    data=json.loads(content)
    if data.get('schema') not in (1,2) or not data.get('waveform_validated'):
        raise ValueError('Calibrated RC profile requires verified full-layout waveform comparisons')
    receipt = data.get('extraction_receipt', {})
    if (receipt.get('drc_violations') != 0 or receipt.get('lvs') != 'CORRECT'
            or receipt.get('extraction') != 'distributed R+C+CC'
            or not data.get('waveform_validation', {}).get('passed')):
        raise ValueError('Calibrated RC profile requires passing physical and waveform evidence')
    data['_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if data['schema'] == 2:
        from sram_compiler.per_net_rc import assemble
        data['_network_directory'] = str(path.parent)
        assemble(data, path.parent)
    return data


def validate_source(deck, profile):
    """Reject reuse after any schematic topology, model or geometry change."""
    from tempfile import TemporaryDirectory
    from sram_compiler.layout.freepdk45.full_source import flatten

    with TemporaryDirectory(prefix='openyield-rc-source-') as directory:
        path = Path(directory) / 'source.sp'
        path.write_text(deck)
        source = flatten(path)
        actual = source.mos
    fields = ('name', 'nodes', 'model', 'width_m', 'length_m', 'fingers')
    def signature(devices):
        return sorted(json.dumps({key: device[key] for key in fields},
                                 sort_keys=True) for device in devices)
    if signature(actual) != signature(profile['layout_manifest']['source_devices']):
        raise ValueError('Schematic topology or transistor geometry changed relative to the calibrated RC profile')
    expected_passives = profile['layout_manifest'].get('source_passives')
    if expected_passives is not None:
        passive_fields = ('name', 'kind', 'nodes', 'value')
        def passives(items):
            return sorted(json.dumps({key: item[key] for key in passive_fields},
                                     sort_keys=True) for item in items)
        if passives(source.passives) != passives(expected_passives):
            raise ValueError('Intentional passive sizing changed relative to the calibrated RC profile')


def terminal_map(profile):
    if '_terminal_map' in profile:
        return profile['_terminal_map']
    result={}
    for tokens in profile['mos']:
        name=tokens[0]
        # Calibre prepends its device letter to the original source name.
        if name[:2].lower()=='mm':name=name[1:]
        if name[:2].lower()=='mx':name=name[1:]
        name=re.sub(r'@\d+$','',name).upper()
        result.setdefault(name,dict(zip('DGSB',tokens[1:5])))
    profile['_terminal_map'] = result
    return result


def probe_node(profile,name):
    """Map an original compiler probe to a physical terminal in the model."""
    terminals=terminal_map(profile)
    manifest=profile['layout_manifest']
    if '_source_aliases' not in profile:
        profile['_source_aliases'] = {k.upper():v for k,v in manifest['aliases'].items()}
    aliases=profile['_source_aliases']
    n=name.replace(':','/').upper()

    def device(path,terminal='D'):
        value=terminals.get(path.upper())
        if value is None:raise ValueError(f'Physical RC profile has no device terminal for {path}:{terminal}')
        return value[terminal]

    if n == 'RWL_FAR':
        return device('Xsram_8x1_replica_column/XReplica_CELL_7/MPGL', 'G')
    match = re.fullmatch(r'(PRE|SA_ISO|S_EN|W_EN|WL_EN)_LINE_(FAR|TAP(\d+))', n)
    if match:
        role, _, tap = match.groups()
        index = 7 if tap is None else int(tap)
        paths = dict(PRE=f'XPRECHARGE_{index}/M1',
                     SA_ISO=f'XSENSEAMP_{index}/M5',
                     S_EN=f'XSENSEAMP_{index}/M7',
                     W_EN=f'XWRITEDRIVER_{index}/M3',
                     WL_EN=f'XWL_DRV_{index}/XPNAND2/Mpnand2_nmos1')
        if tap is None:
            paths = dict(PRE='XPRECHARGE_RBL/M1',
                         SA_ISO='XREPLICA_SENSEAMP/M5',
                         S_EN='XREPLICA_SENSEAMP/M7',
                         W_EN='XREPLICA_WDRV_LOAD/M3',
                         WL_EN='XRWL/XPNAND2/Mpnand2_nmos1')
        return device(paths[role], 'G')
    match = re.fullmatch(r'(R?BLB?)(\d*)_PERIPH_TAP([012])', n)
    if match:
        role, column, tap = match.groups()
        replica = role.startswith('R')
        inverted = role.endswith('BLB')
        if tap == '0':
            path = ('XPRECHARGE_RBL' if replica else f'XPRECHARGE_{column}') + ('/M2' if inverted else '/M1')
            terminal = 'D'
        elif tap == '1':
            path = ('XREPLICA_WDRV_LOAD' if replica else f'XWRITEDRIVER_{column}') + ('/M10' if inverted else '/M6')
            terminal = 'D'
        else:
            path = ('XREPLICA_SENSEAMP' if replica else f'XSENSEAMP_{column}') + ('/M6' if inverted else '/M5')
            terminal = 'S'
        return device(path, terminal)
    match = re.fullmatch(r'XDECODER:A(\d+)_LINE_(?:FAR|TAP\d+)', name.upper())
    if match:
        canonical = aliases[n]
        candidates = [d['name'] for d in manifest['source_devices']
                      if d['name'].upper().startswith('XDECODER/')
                      and d['nodes'][1].upper() == canonical.upper()]
        if not candidates:
            raise ValueError(f'Missing physical decoder address terminal: {name}')
        return device(max(candidates), 'G')

    match=re.fullmatch(r'(XSRAM_6T_CORE_8X8/XSRAM_6T_CELL_\d+_\d+)/(Q(?:_END)?|QB(?:_END)?|WL(?:_END)?|BL(?:_END)?|BLB(?:_END)?)',n)
    if match:
        prefix,role=match.groups();role=role.replace('_END','')
        return device(prefix+dict(Q='/MPDL',QB='/MPDR',WL='/MPGL',BL='/MPGL',BLB='/MPGR')[role],
                      'G' if role=='WL' else 'D')
    match=re.fullmatch(r'XSRAM_6T_CORE_8X8/(WL|BL|BLB)(\d+)_(?:FAR|TAP(\d+))',n)
    if match:
        role,index,tap=match.groups();index=int(index);tap=7 if tap is None else int(tap)
        row,col=(index,tap) if role=='WL' else (tap,index)
        return device(f'XSRAM_6T_CORE_8X8/XSRAM_6T_CELL_{row}_{col}/'+('MPGR' if role=='BLB' else 'MPGL'),
                      'G' if role=='WL' else 'D')
    match=re.fullmatch(r'(XPRECHARGE_(?:\d+|RBL))/(ENB_END|BL_END|BLB_END)',n)
    if match:
        prefix,role=match.groups()
        if role=='ENB_END':return device(prefix+'/M1','G')
    match=re.fullmatch(r'(XSENSEAMP_\d+|XREPLICA_SENSEAMP)/(EN_END|ISO_END|IN_END|INB_END)',n)
    if match:
        prefix,role=match.groups()
        return device(prefix+dict(EN_END='/M7',ISO_END='/M5',IN_END='/M5',INB_END='/M6')[role],
                      'G' if role in ('EN_END','ISO_END') else 'S')
    match=re.fullmatch(r'(XWRITEDRIVER_\d+|XREPLICA_WDRV_LOAD)/(EN_END|DIN_END)',n)
    if match:
        prefix,role=match.groups()
        if role=='EN_END':return device(prefix+'/M3','G')
        return device(prefix+'/M1', 'G')
    match=re.fullmatch(r'XTIME_CONTROL/XDFF_BUF_DATA/CLK_LINE_(FAR|TAP(\d+))',n)
    if match:
        col=7 if match[1]=='FAR' else int(match[2])
        return device(f'XTIME_CONTROL/Xdff_buf_data/Xdff_{col}/Xinv1_clk/Mpinv_nmos','G')
    canonical=aliases.get(n,n)
    ports={p.upper():p for p in profile['ports']}
    if canonical.upper() in ports:return ports[canonical.upper()]
    if '_source_candidates' not in profile:
        from collections import defaultdict
        indexed = defaultdict(list)
        for source in manifest['source_devices']:
            if source['name'].upper() not in terminals:
                continue
            for terminal, node in zip('DGSB', source['nodes']):
                rank=(terminal!='D',not source['name'].startswith('XTIME_CONTROL/'),source['name'])
                indexed[node.upper()].append((rank,device(source['name'],terminal)))
        profile['_source_candidates'] = dict(indexed)
    candidates=profile['_source_candidates'].get(canonical.upper(), [])
    if not candidates:raise ValueError(f'Unmapped physical compiler probe: {name}')
    return min(candidates)[1]


def apply_to_deck(deck,profile,probes=()):
    """Replace generated hardware with the scoped calibrated physical model.

    Stimulus sources and analysis directives remain caller-owned. Probe aliases
    are observational resistive shorts, so existing measurement/checker names
    identify the corresponding physical terminals.
    """
    keep=[];depth=0
    for line in deck.splitlines():
        tokens=line.split()
        if not tokens:continue
        key=tokens[0].lower()
        if key=='.subckt':depth+=1;continue
        if key=='.ends':depth-=1;continue
        if depth:continue
        if key[0]=='x' or key[0] in ('r','c'):continue
        if key[0] in ('b','e','g','f','h') and not key.startswith('*'):continue
        if key=='.end':continue
        keep.append(line)
    used=set(probes)
    # Preserve the usual cell observation names even if the caller adds probes
    # after constructing the simulator (as the local waveform checker does).
    for row in range(profile['scope']['rows']):
        for col in range(profile['scope']['cols']):
            prefix = f'XSRAM_6T_CORE_8x8:XSRAM_6T_CELL_{row}_{col}:'
            used.update(prefix + pin for pin in ('Q', 'QB', 'WL', 'WL_end', 'BL', 'BL_end', 'BLB', 'BLB_end'))
            used.add(f'XSRAM_6T_CORE_8x8:WL{row}_tap{col}')
    for col in range(profile['scope']['cols']):
        used.add(f'XPRECHARGE_{col}:ENB_end')
        used.update(f'XWRITEDRIVER_{col}:{pin}' for pin in ('DIN_end', 'EN_end'))
        used.update(f'XSENSEAMP_{col}:{pin}' for pin in ('EN_end', 'ISO_end', 'IN_end', 'INB_end'))
    for line in keep:
        used.update(re.findall(r'\bV\(([^)]+)\)',line,re.I))
    ports={p.upper():p for p in profile['ports']}
    for index,line in enumerate(keep):
        if line.lower().startswith('.ic '):
            def initial(match):
                name=match.group(1)
                if name.upper() in ports:return match.group(0)
                return 'V(XPHYSICAL_FULL_MACRO:'+probe_node(profile,name)+')'
            keep[index]=re.sub(r'\bV\(([^)]+)\)',initial,line,flags=re.I)
    result=render(profile)
    result+='XPHYSICAL_FULL_MACRO '+' '.join(profile['ports'])+' CALIBRATED_OPENYIELD_8X8\n'
    # Read-only compiler decks omit data sources. The physical macro still has
    # its eight data registers and write buffers; hold those inputs at ground.
    driven={l.split()[1].upper() for l in keep if l.split() and l.split()[0][0].upper()=='V' and not l.startswith('.')}
    for port in profile['ports']:
        if port.startswith('DIN') and port.upper() not in driven:
            result+=f'VDATA_HOLD_{port} {port} VSS 0\n'
    for i,name in enumerate(sorted(used)):
        if name.upper() in ports:continue
        node=probe_node(profile,name)
        result+=f'RPHYSICAL_PROBE_{i} {name} XPHYSICAL_FULL_MACRO:{node} 1\n'
    return '\n'.join(keep)+'\n'+result+'\n.END\n'
