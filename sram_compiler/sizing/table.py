"""Read qualification records with exact baseline and physical-context matching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .timing import TimingConfig

DEFAULT_TABLE = Path(__file__).with_name('sizing_table.json')
SCORING_SOURCES = Path(__file__).with_name('scoring_sources.json')


def current_scoring_version():
    """Identity of approved scoring sources and runtime code, without local tools.

    Development runners verify their files against the tracked source manifest
    before producing evidence. Table consumers only need that manifest and the
    compiler, so an ordinary checkout never imports or opens ignored scripts.
    """
    digest = hashlib.sha256()
    digest.update(SCORING_SOURCES.read_bytes())
    for name in ('timing.py', 'table.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / 'sram_compiler').rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def physical_context(w_rc=False, pi_res=100.0, pi_cap=1e-15, real_cell_mode=0, interconnect=None):
    from sram_compiler.interconnect import resolve_interconnect
    result: dict[str, Any] = {'w_rc': bool(w_rc), 'real_cell_mode': int(real_cell_mode)}
    result['interconnect'] = resolve_interconnect(interconnect).to_dict()
    if w_rc:
        result.update(pi_res=float(pi_res), pi_cap=float(pi_cap))
    return result


def record_key(sizes_key, context):
    return hashlib.sha256(json.dumps({'sizes_key': sizes_key, 'physical': context}, sort_keys=True).encode()).hexdigest()


def lookup_record(sizes, table_path=None, context=None):
    path = DEFAULT_TABLE if table_path is None else Path(table_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return None
    table = json.loads(path.read_text())
    if table.get('schema') != 1:
        raise ValueError(f'Unsupported qualification table schema: {path}')
    context = physical_context() if context is None else context
    record = table.get('records', {}).get(record_key(sizes.key, context))
    if record is None:
        return None
    if record.get('scoring_version') != current_scoring_version():
        return None
    if (record.get('sizes_key') != sizes.key or record.get('physical') != context
            or record.get('qualified') is not True or record.get('campaign_complete') is not True
            or record.get('local_mismatch_qualified') is not True):
        raise ValueError('Incomplete or mismatched qualification table record')
    if record.get('variation_policy') != 'full-local-v1' or record.get('local_relative_sigma') != .05:
        raise ValueError('Incomplete or mismatched local process qualification')
    # Reject a record that claims qualification for a different physical vector.
    expected = sizes.to_dict()
    actual = dict(record.get('driver_sizes', {}))
    actual.pop('source', None)
    expected.pop('source', None)
    actual.pop('qualified_context_key', None)
    expected.pop('qualified_context_key', None)
    # JSON represents the immutable tuples as arrays.
    if actual != json.loads(json.dumps(expected)):
        raise ValueError('Qualification table driver vector does not match the resolver')
    try:
        timing = TimingConfig(**record['timing'])
        if min(timing.low_read, timing.low_write, timing.high) <= 0:
            raise ValueError('Qualified records require measured nonzero phases')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid qualified clock configuration') from exc
    return record


def qualified_timing(sizes, table_path=None, context=None):
    record = lookup_record(sizes, table_path, context)
    return None if record is None else TimingConfig(**record['timing'])
