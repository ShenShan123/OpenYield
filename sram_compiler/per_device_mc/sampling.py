"""Materialize independent local LHS draws before a parallel Xyce transient.

Xyce 7.4's MPI random-expression initialization crashes on large per-model
ensembles in this workspace. Numeric model cards keep the random draws fixed
while MPI evaluates the actual transistor circuit. Every sampled card is saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from contextlib import ExitStack
from pathlib import Path
from statistics import NormalDist

SAMPLER_VERSION = 'per-device-lhs-v1'
_NORMAL = NormalDist()
_MODEL = re.compile(r'^\.model\s+(\S+)', re.M | re.I)
_DRAW = re.compile(r'(vth0|u0|voff)(\s*=\s*)\{AGAUSS\(([^,]+),\s*([^,]+),\s*1\)\}', re.I)


def standard_normal_draws(seed, parameter, samples):
    """A stable per-parameter stream preserves common draws across candidates."""
    identity = hashlib.sha256(f'{seed}:{parameter.upper()}'.encode()).digest()
    rng = random.Random(int.from_bytes(identity, 'big'))
    strata = list(range(samples))
    rng.shuffle(strata)
    return [_NORMAL.inv_cdf(min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0),
                               (stratum + rng.random()) / samples))) for stratum in strata]


def materialize_decks(deck_text, source_model, output_root, seed, samples):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('Sample count must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError('Materialized local sampling requires a positive integer seed')
    source_model, root = Path(source_model).resolve(), Path(output_root).resolve()
    text = source_model.read_text()
    matches = list(_MODEL.finditer(text))
    if not matches:
        raise ValueError('No per-device model cards to sample')
    directories = [root / f'sample_{index:04d}' for index in range(samples)]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    models = [directory / 'models.spice' for directory in directories]
    count = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open('w')) for path in models]
        for index, match in enumerate(matches):
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start():stop]
            parameters = {}
            for draw in _DRAW.finditer(block):
                name, mean, sigma = draw[1].lower(), float(draw[3]), float(draw[4])
                if not math.isfinite(mean) or not math.isfinite(sigma) or sigma < 0:
                    raise ValueError('Invalid Gaussian model parameters')
                parameters[name] = [mean + sigma * value for value in
                                    standard_normal_draws(seed, match[1] + ':' + name, samples)]
            if set(parameters) != {'vth0', 'u0', 'voff'}:
                raise ValueError(f'Model {match[1]} lacks all three local random parameters')
            count += len(parameters)
            for sample, stream in enumerate(streams):
                stream.write(_DRAW.sub(lambda draw: draw[1] + draw[2] + repr(parameters[draw[1].lower()][sample]), block))
    decks = []
    for sample, (directory, model) in enumerate(zip(directories, models)):
        deck = deck_text.replace(str(source_model), str(model))
        if deck == deck_text:
            raise ValueError('Per-device model include is missing from the source deck')
        deck = '\n'.join(line for line in deck.splitlines()
                         if not line.lstrip().lower().startswith(('.sampling', '.options samples'))) + '\n'
        path = directory / 'deck.sp'
        path.write_text(deck)
        decks.append({'sample': sample, 'deck': str(path), 'model': str(model),
                      'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()})
    manifest = {'sampler': SAMPLER_VERSION, 'seed': seed, 'samples': samples,
                'sampled_parameters': count, 'independent_models': len(matches),
                'distribution': 'independent Gaussian marginals; Latin hypercube strata per parameter',
                'stream_identity': 'SHA256(ensemble seed, per-device model name, parameter)',
                'source_model_sha256': hashlib.sha256(source_model.read_bytes()).hexdigest(),
                'sampler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'decks': decks}
    (root / 'sampling.json').write_text(json.dumps(manifest, indent=2))
    return manifest
