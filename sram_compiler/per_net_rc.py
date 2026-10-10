"""Independent extracted signal and power RC models, with shared coupling edges."""
from __future__ import annotations

import gzip
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_network(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f'RC network changed: {Path(path).name}; recalibrate the baseline')
    return json.loads(gzip.decompress(data))


def assemble(profile, directory, overrides=None):
    """Compose separate per-net networks without sharing R/C values or scales.

    Power VDD/VSS are distinct resistive meshes. Coupling edges connect the
    separate networks symmetrically; their scale uses the geometric mean of
    the participating nets' capacitance scales.
    """
    directory = Path(directory)
    overrides = {} if overrides is None else overrides
    unknown = set(overrides)-set(profile['net_models'])
    if unknown:
        raise ValueError(f'Unknown extracted RC nets: {sorted(unknown)}')
    resistors, capacitors, scales = {}, {}, {}
    qualified_settings = True
    for name, settings in profile['net_models'].items():
        chosen = overrides.get(name, {})
        rscale = chosen.get('resistance_scale', settings['resistance_scale'])
        cscale = chosen.get('capacitance_scale', settings['capacitance_scale'])
        if (isinstance(rscale, bool) or isinstance(cscale, bool)
                or not isfinite(rscale) or not isfinite(cscale)
                or rscale <= 0 or cscale <= 0):
            raise ValueError('Per-net R/C scales must be finite and positive')
        scales[name] = cscale
        filename = chosen.get('network_file')
        if filename is None:
            path, digest = directory/settings['file'], settings['sha256']
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = ROOT/path
            digest = chosen['network_sha256']
        qualified_settings &= (digest == settings['sha256']
                               and rscale == settings['resistance_scale']
                               and cscale == settings['capacitance_scale'])
        network = read_network(path, digest)
        if network['net'] != name:
            raise ValueError(f'RC file belongs to a different net: {name}')
        for index,a,b,r in network['resistors']:
            if index in resistors:
                raise ValueError('Duplicate per-net resistor index')
            resistors[index]=(a,b,r*rscale)
        for index,a,b,c in network['capacitors']:
            if index in capacitors:
                raise ValueError('Duplicate per-net capacitor index')
            capacitors[index]=(a,b,c*cscale)
    settings = profile['coupling_model']
    network = read_network(directory/settings['file'], settings['sha256'])
    for index,a,b,c,net_a,net_b in network['capacitors']:
        if index in capacitors:
            raise ValueError('Duplicate coupling capacitor index')
        capacitors[index]=(a,b,c*sqrt(scales[net_a]*scales[net_b]))
    if set(resistors) != set(range(profile['resistor_count'])):
        raise ValueError('Incomplete per-net resistor model')
    if set(capacitors) != set(range(profile['capacitor_count'])):
        raise ValueError('Incomplete per-net capacitor/coupling model')
    profile['resistors']=[resistors[i] for i in range(len(resistors))]
    profile['capacitors']=[capacitors[i] for i in range(len(capacitors))]
    profile['_net_overrides']=dict(overrides)
    profile['_qualified_settings'] = qualified_settings
    return profile
