"""Independent physical waveform checks for the V2.2.x phase contract.

V2.2.2 also reports how much of each phase budget a passing trace used: the
read output's lead over the checker deadline, the wordline dwell after a
written cell has flipped, and the bitline restore and precharge-on slack
before the next capture.

V2.2.3 takes the access deadline from the compiler itself, so a runtime
`.MEASURE` pass and a screen pass accept the same traces; scales the sense
bar with VDD instead of fixing it at 0.25 V; and probes more sentinel rows
on arrays too large to probe whole, including the rows adjacent to every
accessed row.

`min_storage_polarity_margin_v` is a reversal bound, not a stability metric:
it is the distance of the worst storage node from a fixed 0.5 VDD boundary,
and that boundary is not the trip point of every cell. The 10T cell is a
Schmitt-trigger cell read through the same access transistors as the 6T, so
it disturbs its storage node further at a correspondingly higher trip point;
comparing the two topologies by this number is not meaningful. What bounds
recovery is the 0.1 VDD rail-error check after the access deadline.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from sram_compiler.testbenches.sram_6t_core_MC_testbench import ACCESS_DEADLINE, cycle_plan

SCORER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# A trace probes every cell up to this many; above it only the rows below.
PROBE_EVERY_CELL_UPTO = 1024
# The bitline differential a read must present at the isolation sampling
# instant, as a fraction of VDD. V2.2.2 used a fixed 0.25 V, which is 28 % of
# a 0.9 V rail and 23 % of a 1.1 V one; the fraction is the 0.9 V bar, so no
# screened point is relaxed. It is a screening floor: the sense amplifier's
# own input offset under local mismatch is not measured by this screen.
SENSE_MARGIN_FRACTION = 0.28


def probed_rows(rows, cols, case, target_row):
    """Rows whose cells are probed and checked for retention and disturb.

    Small arrays probe every cell. Above that, probe the array's quarter
    points and last row, every row an operation addresses, and each addressed
    row's immediate neighbours, which are the rows a wordline or bitline
    excursion would disturb first. The runner and the checker must agree on
    this set, so both call it rather than repeating the rule.
    """
    if rows * cols <= PROBE_EVERY_CELL_UPTO:
        return set(range(rows))
    addressed = {target_row}
    if case.get('next_row') is not None:
        addressed.add(case['next_row'])
    addressed.update(entry['row'] for entry in case.get('pattern', ()))
    chosen = {0, rows // 4, rows // 2, (3 * rows) // 4, rows - 1} | addressed
    chosen.update(neighbour for row in addressed
                  for neighbour in (row - 1, row + 1) if 0 <= neighbour < rows)
    return chosen


def threshold_overlap(time, conditions, start, stop=None):
    """Whether strict threshold intervals intersect on linear trace segments.

    Each condition is (values, threshold, above). Interpolation can expose
    overlap even when both recorded endpoints satisfy the safety rule. This
    cannot detect an excursion absent from the recorded waveform itself.
    """
    begin = np.maximum(0., (start - time[:-1]) / np.diff(time))
    end = np.ones(len(time) - 1)
    if stop is not None:
        end = np.minimum(end, (stop - time[:-1]) / np.diff(time))
    for values, level, above in conditions:
        direction = 1 if above else -1
        left = direction * (values[:-1] - level)
        right = direction * (values[1:] - level)
        slope = right - left
        crossing = np.divide(-left, slope, out=np.zeros_like(left), where=slope != 0)
        begin = np.maximum(begin, np.where(slope > 0, crossing, 0.))
        end = np.minimum(end, np.where(slope < 0, crossing, 1.))
        end[(left <= 0) & (right <= 0)] = 0.
    return bool(np.any(begin < end))


def score(directory, report_name="result.json"):
    """Score a completed single-sample trace; missing signals fail closed.

    Check electrical exclusivity at 0.1/0.9 VDD, pulse counts at 0.1 VDD,
    retained data at 0.1 VDD error, and restored bitlines at 0.02 VDD error.
    The expected row comes from the external address before capture, rather
    than following whichever row the implementation happened to activate.
    """
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    with (directory / 'deck.sp.prn').open() as stream:
        header = stream.readline().upper().split()
        data = np.loadtxt(line for line in stream
                          if line.split() and line.split()[0].isdigit())
    if (data.ndim != 2 or len(data) < 2 or data.shape[1] != len(header)
            or 'TIME' not in header or 'INDEX' not in header):
        raise ValueError('Incomplete or malformed waveform table')
    time = data[:, header.index('TIME')]
    columns = {name: index for index, name in enumerate(header)}

    def signal(node):
        return data[:, columns[f'V({node.upper()})']]

    def at(node, when):
        return float(np.interp(when, time, signal(node)))

    def window(start, stop):
        selected = (time >= start) & (time <= stop)
        if not selected.any():
            raise ValueError('Empty check interval')
        return selected

    def band_entry(node, high, start, stop):
        """Time from which `node` stays within 0.1 VDD of its expected rail until `stop`."""
        values = signal(node)
        outside = (time >= start) & (time <= stop) & (np.abs(values - high * vdd) > .1 * vdd)
        indexes = np.flatnonzero(outside)
        if not len(indexes):
            return start
        last = indexes[-1]
        if last + 1 >= len(time):
            return time[last]
        level = (high - .1) * vdd if high else .1 * vdd
        slope = values[last + 1] - values[last]
        fraction = 1. if slope == 0 else np.clip((level - values[last]) / slope, 0., 1.)
        return time[last] + fraction * (time[last + 1] - time[last])

    def edges(node, level, start, stop, rising):
        values = signal(node)
        selected = (time[:-1] >= start) & (time[1:] <= stop)
        if rising:
            selected &= (values[:-1] < level) & (values[1:] >= level)
        else:
            selected &= (values[:-1] > level) & (values[1:] <= level)
        indexes = np.flatnonzero(selected)
        return (time[indexes] + (level - values[indexes])
                * (time[indexes + 1] - time[indexes])
                / (values[indexes + 1] - values[indexes]))

    checks, metrics = {}, {}
    vdd, period = metadata['vdd'], metadata['period']
    nodes, case = metadata['nodes'], metadata['case']
    rows, cols = metadata['rows'], metadata['cols']
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1:
        raise ValueError('Invalid array dimensions in waveform contract')
    if set(nodes['wl']) != {str(row) for row in range(rows)} or any(not value for value in nodes['wl'].values()):
        raise ValueError('Every row needs physical wordline probes')
    required_roles = ['pre', 'wen', 'sen', 'iso']
    if metadata['operation'] != 'read':
        required_roles.append('data')
    if metadata['operation'] != 'write':
        required_roles.extend(('sense', 'sense_state'))
    for role in required_roles:
        if set(nodes[role]) != {str(col) for col in range(cols)}:
            raise ValueError(f'Incomplete {role} probe map')
    checked_rows = probed_rows(rows, cols, case, metadata['row'])
    required_cells = {f'{row},{col}' for row in checked_rows for col in range(cols)}
    if not required_cells <= set(nodes['cells']):
        raise ValueError('Missing accessed or retention-sentinel cell probes')
    interval, analysis_stop = metadata['sample_interval'], metadata['analysis_stop']
    if not np.isfinite(interval) or interval <= 0 or not np.isfinite(analysis_stop) or analysis_stop <= 0:
        raise ValueError('Invalid waveform sampling contract')
    indexes = data[:, columns['INDEX']]
    expected_samples = round(analysis_stop / interval) + 1
    if abs((expected_samples - 1) * interval - analysis_stop) > max(1e-18, interval * 1e-5):
        raise ValueError('Analysis stop must lie on the recorded output grid')
    checks['index_contiguous'] = bool(np.array_equal(indexes, np.arange(len(data))))
    checks['sample_count'] = bool(len(data) == expected_samples)
    # Xyce can coalesce a grid point with a nearby solver point (observed
    # offsets below 5 fs on a 5 ps grid). A 1% interval allowance admits that
    # jitter, while a missing or shifted sample is still nearly 100% wrong.
    checks['time_grid'] = bool(np.all(np.abs(time - np.arange(len(data)) * interval)
                                      <= max(1e-18, interval * .01)))
    for index, name in enumerate(header):
        previous = header.index(name)
        if previous != index:
            checks[f'duplicate_{name}_{index}'] = bool(
                np.array_equal(data[:, previous], data[:, index])
                and np.isfinite(data[:, index]).all())
    checks['finite_monotonic'] = bool(np.isfinite(data).all() and (np.diff(time) > 0).all())
    span, plan = cycle_plan(metadata['operation'], case.get('select_every', 1))
    checks['complete'] = bool(time[-1] >= 1e-9 + span * period - 1e-14)
    if 'pattern' in case:
        plan = [(entry['op'], entry['data']) for entry in case['pattern']]
    # V2.2.4: the target column is recorded; older records always used the last one.
    target_col = metadata.get('col', metadata['cols'] - 1)
    columns_by_cycle = case.get('column_sequence')
    if columns_by_cycle is not None:
        if (not case.get('mux', False) or len(columns_by_cycle) != len(plan)
                or columns_by_cycle[0] != target_col
                or any(type(col) is not int or not 0 <= col < cols
                       or col // 2 != target_col // 2 for col in columns_by_cycle)):
            raise ValueError('Invalid column_sequence in waveform contract')

    def column_for_cycle(cycle):
        return target_col if columns_by_cycle is None else columns_by_cycle[min(cycle, len(columns_by_cycle) - 1)]

    target = f'{metadata["row"]},{target_col}'
    expected = {key: 0 if key == target else case.get('background', 0) for key in nodes['cells']}
    cycles = 8 if metadata['operation'] == 'read&write' else len(plan)
    sense_margins, capture_slack, driver_slack = [], [], []
    sense_slack, isolation_slack, drive_setup, storage_margins = [], [], [], []
    recovery_slack, output_margin, write_dwell, restore_slack, precharge_slack = [], [], [], [], []
    wordline_peak = np.maximum.reduce([signal(node) for row in nodes['wl'].values() for node in row])
    array = next(iter(nodes['cells'].values()))[0].split(':')[0]

    def retain_polarity(q, qb, value, start, stop):
        """Allow analog read disturb, but never a stored-state reversal.

        Linear extrema occur at recorded samples or interval endpoints. The
        stricter rail-error check still applies after the access deadline.
        """
        if stop <= start:
            return True
        selected = (time > start) & (time < stop)
        rails = []
        for node, high in ((q, value), (qb, 1 - value)):
            values = signal(node)
            samples = values[selected]
            endpoints = np.array([at(node, start), at(node, stop)])
            if not high:
                samples, endpoints = vdd - samples, vdd - endpoints
            rails.append(min(float(np.min(samples, initial=np.inf)), float(np.min(endpoints))))
        margin = min(rails) - .5 * vdd
        storage_margins.append(margin)
        return bool(margin > 0.)

    # Check every linear segment, including capture boundaries and a partial
    # final cycle. Interpolating the maximum of WL endpoints is conservative:
    # it bounds every individual interpolated WL from above.
    capture = 1e-9 + .2 * period
    def safe(*conditions):
        return not threshold_overlap(time, conditions, capture)
    wl_active = (wordline_peak, .1 * vdd, True)
    for col in range(metadata['cols']):
        key = str(col)
        precharge = signal(nodes['pre'][key])
        write = signal(nodes['wen'][key])
        sense = signal(nodes['sen'][key])
        isolation = signal(nodes['iso'][key])
        write_active, sense_active = (write, .1 * vdd, True), (sense, .1 * vdd, True)
        checks[f'continuous_c{col}_isolation'] = all(
            safe((isolation, .9 * vdd, False), enabled)
            for enabled in (write_active, sense_active))
        checks[f'continuous_c{col}_read_write_exclusion'] = safe(write_active, sense_active)
        checks[f'continuous_c{col}_sense_after_wl'] = safe(sense_active, wl_active)
        checks[f'continuous_c{col}_pre_exclusion'] = all(
            safe((precharge, .9 * vdd, False), active)
            for active in (wl_active, write_active, sense_active))
        checks[f'continuous_c{col}_driver_covers_wl'] = safe(
            (write, .9 * vdd, False), wl_active,
            (signal('WE'), .5 * vdd, True), (signal('CS'), .5 * vdd, True))
        checks[f'continuous_c{col}_write_request'] = all(
            safe(write_active, (signal(request), .5 * vdd, False))
            for request in ('WE', 'CS'))

    # Safety also applies to the final partial access. Native mixed traces
    # start a ninth operation; custom PWL patterns hold their final request.
    # Do not require completion/release from an access whose tail is absent.
    retained = expected.copy()
    for cycle in range(max(0, int((time[-1] - 1e-9) / period - .2) + 1)):
        kind, datum = plan[min(cycle, len(plan) - 1)]
        cycle_col = column_for_cycle(cycle)
        start = 1e-9 + (cycle + .2) * period
        stop = min(1e-9 + (cycle + 1.2) * period, time[-1])
        prefix = f'cycle{cycle}_{kind}'
        row = sum((at(f'A{bit}', start - .005 * period) > .5 * vdd) << bit
                  for bit in range(max(1, (metadata['rows'] - 1).bit_length())))
        for row_number, wordlines in nodes['wl'].items():
            for endpoint, node in enumerate(wordlines):
                if kind == 'idle' or int(row_number) != row:
                    checks[f'{prefix}_wl{row_number}_{endpoint}_unselected'] = not threshold_overlap(
                        time, ((signal(node), .1 * vdd, True),), start, stop)
                else:
                    checks[f'{prefix}_wl{row_number}_{endpoint}_at_most_once'] = bool(
                        len(edges(node, .1 * vdd, start, stop, True)) <= 1)
        for node in ('XTIME_CONTROL:wordline_busy', 'XTIME_CONTROL:iso_ready',
                     'XTIME_CONTROL:access_request', 'XTIME_CONTROL:access_settled',
                     'XTIME_CONTROL:read_done', 'RBL_DELAY'):
            checks[f'{prefix}_capture_reset_{node}'] = bool(abs(at(node, start)) <= .1 * vdd)
        checks[f'{prefix}_capture_recovery_ready'] = bool(
            at('XTIME_CONTROL:enables_off', start) >= .9 * vdd)
        for col in range(metadata['cols']):
            key = str(col)
            checks[f'{prefix}_c{col}_capture_idle'] = bool(max(
                at(nodes['wen'][key], start), at(nodes['sen'][key], start),
                at(nodes['iso'][key], start),
                float(np.interp(start, time, wordline_peak))) <= .1 * vdd)
            for role, allowed in (('wen', kind == 'write'), ('sen', kind == 'read'), ('iso', kind != 'idle')):
                node = nodes[role][key]
                if allowed:
                    checks[f'{prefix}_c{col}_{role}_at_most_once'] = bool(
                        len(edges(node, .1 * vdd, start, stop, True)) <= 1)
                else:
                    checks[f'{prefix}_c{col}_{role}_quiet'] = not threshold_overlap(
                        time, ((signal(node), .1 * vdd, True),), start, stop)
        rises, falls = [], []
        if kind == 'write':
            for node in nodes['wl'][str(row)]:
                rises.extend(edges(node, .1 * vdd, start, stop, True))
                falls.extend(edges(node, .1 * vdd, start, stop, False))
            for col in range(metadata['cols']):
                value = int(datum[col]) if isinstance(datum, str) else datum
                write_data = signal(nodes['data'][str(col)])
                checks[f'{prefix}_c{col}_data_stable'] = not any(threshold_overlap(
                    time, (wl_active, (write_data, limit * vdd, above)), start, stop)
                    for limit, above in ((value - .1, False), (value + .1, True)))
        for key, (q, qb) in nodes['cells'].items():
            cell_row, col = map(int, key.split(','))
            if kind == 'write' and cell_row == row:
                before = retain_polarity(q, qb, retained[key], start, min(rises) if rises else stop)
                value = int(datum[col]) if isinstance(datum, str) else datum
                after = retain_polarity(q, qb, value, max(falls) if falls else stop, stop)
                checks[f'{prefix}_cell{key}_logical_retention'] = before and after
                retained[key] = value
            else:
                checks[f'{prefix}_cell{key}_logical_retention'] = retain_polarity(
                    q, qb, retained[key], start, stop)
        sense_start = 1e-9 + (cycle + ACCESS_DEADLINE) * period
        if kind == 'read' and sense_start < stop:
            mux = 2 if case.get('mux', False) else 1
            for col in range(0, metadata['cols'], mux):
                selected_col = col + cycle_col % mux
                value = retained[f'{row},{selected_col}']
                enabled = (signal(nodes['sen'][str(col)]), .1 * vdd, True)
                q, qb = nodes['sense_state'][str(col)]
                checks[f'{prefix}_sense_group{col // mux}_hold'] = all(
                    not threshold_overlap(time, (enabled, (signal(node), limit * vdd, above)),
                                          sense_start, stop)
                    for node, high in ((q, value), (qb, 1 - value))
                    for limit, above in ((high - .1, False), (high + .1, True)))

    for cycle, (kind, datum) in enumerate(plan[:cycles]):
        if 1e-9 + (cycle + ACCESS_DEADLINE) * period > time[-1] + 1e-14:
            continue
        base = 1e-9 + cycle * period
        start, stop = base + .2 * period, min(base + 1.2 * period, time[-1])
        selected = window(start, stop)
        row = sum((at(f'A{bit}', start - .005 * period) > .5 * vdd) << bit
                  for bit in range(max(1, (metadata['rows'] - 1).bit_length())))
        prefix = f'cycle{cycle}_{kind}'
        cycle_col = column_for_cycle(cycle)
        if columns_by_cycle is not None:
            selected_input = cycle_col % 2
            group = (target_col // 2) * 2
            access = window(start, base + .7 * period)
            for input_number in range(2):
                level = vdd if input_number == selected_input else 0.
                for node in (f'SEL{input_number}', f'SEL{input_number}_line_tap{group}'):
                    checks[f'{prefix}_{node}_settled'] = bool(
                        np.max(np.abs(signal(node)[access] - level)) <= .1 * vdd)
        for node in ('XTIME_CONTROL:wordline_busy', 'XTIME_CONTROL:iso_ready',
                     'XTIME_CONTROL:access_request', 'XTIME_CONTROL:access_settled',
                     'XTIME_CONTROL:read_done', 'RBL_DELAY'):
            checks[f'{prefix}_capture_reset_{node}'] = bool(abs(at(node, start)) <= .1 * vdd)
        checks[f'{prefix}_capture_recovery_ready'] = bool(
            at('XTIME_CONTROL:enables_off', start) >= .9 * vdd)

        # Exactly one pulse on the selected row, none on any other row.
        for row_number, wordlines in nodes['wl'].items():
            for endpoint, node in enumerate(wordlines):
                rises = edges(node, .1 * vdd, start, stop, True)
                active_row = kind != 'idle' and int(row_number) == row
                checks[f'{prefix}_wl{row_number}_{endpoint}_pulse'] = bool(len(rises) == int(active_row))
                if active_row:
                    falls = edges(node, .1 * vdd, start, stop, False)
                    checks[f'{prefix}_wl{row_number}_{endpoint}_release'] = bool(len(falls) == 1)
                    if len(falls) == 1:
                        capture_slack.append(float((base + 1.2 * period - falls[0]) * 1e12))

        for col in range(metadata['cols']):
            key = str(col)
            precharge = signal(nodes['pre'][key])
            write = signal(nodes['wen'][key])
            sense = signal(nodes['sen'][key])
            isolation = signal(nodes['iso'][key])
            enabled = selected & ((write > .1 * vdd) | (sense > .1 * vdd))
            checks[f'{prefix}_c{col}_isolate_before_enable'] = bool(np.all(isolation[enabled] >= .9 * vdd))
            checks[f'{prefix}_c{col}_read_write_exclusion'] = bool(
                not np.any(selected & (write > .1 * vdd) & (sense > .1 * vdd)))
            checks[f'{prefix}_c{col}_sense_after_wl'] = bool(
                not np.any(selected & (sense > .1 * vdd) & (wordline_peak > .1 * vdd)))
            pre_excluded = selected & ((wordline_peak > .1 * vdd) | (write > .1 * vdd) | (sense > .1 * vdd))
            checks[f'{prefix}_c{col}_pre_exclusion'] = bool(np.all(precharge[pre_excluded] >= .9 * vdd))
            checks[f'{prefix}_c{col}_capture_idle'] = bool(max(
                at(nodes['wen'][key], start), at(nodes['sen'][key], start),
                at(nodes['iso'][key], start),
                float(np.interp(start, time, wordline_peak))) <= .1 * vdd)
            iso_edges = edges(nodes['iso'][key], .1 * vdd, start, stop, True)
            checks[f'{prefix}_c{col}_isolation_pulse'] = bool(len(iso_edges) == int(kind != 'idle'))
            # Recovery order: both physical enables are off before this cycle's
            # precharge turns on. The exclusion above is a yes/no answer; the
            # superseded root-only observer left 20.55 ps here at 8x256 SS, so
            # record the slack as well. A read has no write-enable crossing and
            # a write has no sense-enable crossing, so the later of the two is
            # whichever one this access used.
            if kind != 'idle':
                enable_off = [crossing for role in ('wen', 'sen')
                              for crossing in edges(nodes[role][key], .1 * vdd, start, stop, False)]
                precharge_on = edges(nodes['pre'][key], .9 * vdd, start, stop, False)
                if enable_off and len(precharge_on) == 1:
                    recovery_slack.append(float((precharge_on[0] - max(enable_off)) * 1e12))

            if kind == 'write':
                active = selected & (wordline_peak > .1 * vdd)
                checks[f'{prefix}_c{col}_driver_covers_wl'] = bool(active.any() and np.min(write[active]) >= .9 * vdd)
                checks[f'{prefix}_c{col}_sense_off'] = bool(np.max(sense[selected]) <= .1 * vdd)
                driver_on = edges(nodes['wen'][key], .1 * vdd, start, stop, True)
                driver_off = edges(nodes['wen'][key], .1 * vdd, start, stop, False)
                checks[f'{prefix}_c{col}_driver_pulse'] = bool(len(driver_on) == len(driver_off) == 1)
                wl_on = edges(nodes['wl'][str(row)][0], .1 * vdd, start, stop, True)
                wl_off = edges(nodes['wl'][str(row)][-1], .1 * vdd, start, stop, False)
                release = edges(nodes['wen'][key], .9 * vdd, start, stop, False)
                driven = edges(nodes['wen'][key], .9 * vdd, start, stop, True)
                if len(wl_off) == len(release) == 1:
                    driver_slack.append(float((release[0] - wl_off[0]) * 1e12))
                if len(wl_on) == len(driven) == 1:
                    drive_setup.append(float((wl_on[0] - driven[0]) * 1e12))
                firing = driver_on
            else:
                checks[f'{prefix}_c{col}_write_off'] = bool(np.max(write[selected]) <= .1 * vdd)
                firing = edges(nodes['sen'][key], .1 * vdd, start, stop, True)
                if kind == 'idle':
                    checks[f'{prefix}_c{col}_sense_off'] = bool(np.max(sense[selected]) <= .1 * vdd)

            if kind == 'read':
                release = edges(nodes['sen'][key], .1 * vdd, start, stop, False)
                checks[f'{prefix}_c{col}_sense_once'] = bool(len(firing) == len(release) == 1)
                if len(firing) == 1:
                    left, right = nodes['sense'][key]
                    sampled = edges(nodes['iso'][key], .5 * vdd, start, stop, True)
                    sample_time = sampled[0] if len(sampled) else firing[0]
                    margin = abs(at(left, sample_time) - at(right, sample_time))
                    sense_margins.append(margin)
                    checks[f'{prefix}_c{col}_sense_margin'] = bool(margin >= SENSE_MARGIN_FRACTION * vdd)
                    wl_off = edges(nodes['wl'][str(row)][-1], .1 * vdd, start, stop, False)
                    if len(wl_off) == 1:
                        sense_slack.append(float((firing[0] - wl_off[0]) * 1e12))
            isolated = edges(nodes['iso'][key], .9 * vdd, start, stop, True)
            if kind != 'idle' and len(isolated) == len(firing) == 1:
                isolation_slack.append(float((firing[0] - isolated[0]) * 1e12))

        if kind == 'write':
            # Write margin: the earliest physical WL endpoint leaves 0.1 VDD
            # this long after the last written cell crossed mid-rail.
            wl_off = [crossing for node in nodes['wl'][str(row)]
                      for crossing in edges(node, .1 * vdd, start, stop, False)]
            for key in expected:
                cell_row, col = map(int, key.split(','))
                if cell_row != row:
                    continue
                value = int(datum[col]) if isinstance(datum, str) else datum
                if expected[key] != value and wl_off and key in nodes['cells']:
                    q, qb = nodes['cells'][key]
                    flips = [crossing for node, rising in ((q, value == 1), (qb, value == 0))
                             for crossing in edges(node, .5 * vdd, start, stop, rising)]
                    if flips:
                        write_dwell.append(float((min(wl_off) - max(flips)) * 1e12))
                expected[key] = value
        # Every probed cell, including untouched rows, must retain its state.
        hold = window(base + ACCESS_DEADLINE * period, stop)
        for key, (q, qb) in nodes['cells'].items():
            value = expected[key]
            error = max(np.max(np.abs(signal(q)[hold] - value * vdd)),
                        np.max(np.abs(signal(qb)[hold] - (1 - value) * vdd)))
            checks[f'{prefix}_cell{key}_retention'] = bool(error <= .1 * vdd)
        if kind == 'read':
            value = expected[f'{row},{cycle_col}']
            checks[f'{prefix}_output'] = bool(np.max(np.abs(signal('OUT')[hold] - value * vdd)) <= .1 * vdd)
            # Access margin: the latched output and every sense group settle
            # on their rail this long before the checker deadline (negative if
            # OUT settles after it). The sense nodes are precharged again once
            # isolation releases, so they are only followed to the deadline.
            deadline = base + ACCESS_DEADLINE * period
            settled = [band_entry('OUT', value, start, stop)]
            mux = 2 if case.get('mux', False) else 1
            for col in range(0, metadata['cols'], mux):
                selected_col = col + cycle_col % mux
                value = expected[f'{row},{selected_col}']
                q, qb = nodes['sense_state'][str(col)]
                error = max(abs(at(q, base + ACCESS_DEADLINE * period) - value * vdd),
                            abs(at(qb, base + ACCESS_DEADLINE * period) - (1 - value) * vdd))
                checks[f'{prefix}_sense_group{col // mux}_data'] = bool(error <= .1 * vdd)
                settled += [band_entry(q, value, start, deadline), band_entry(qb, 1 - value, start, deadline)]
            output_margin.append(float((deadline - max(settled)) * 1e12))
        if base + 1.1 * period <= time[-1]:
            for col in range(metadata['cols']):
                for pin in ('BL', 'BLB'):
                    checks[f'{prefix}_restore_{pin}{col}'] = bool(
                        abs(at(f'{array}:{pin}{col}_far', base + 1.1 * period) - vdd) <= .02 * vdd)
            checks[f'{prefix}_replica_reset'] = bool(at('RBL', base + 1.1 * period) >= .98 * vdd
                                                   and at('RBL_DELAY', base + 1.1 * period) <= .1 * vdd)
        if kind != 'idle' and base + 1.2 * period <= time[-1] + 1e-14:
            # Recovery margin: the last far bitline or RBL back above 0.9 VDD,
            # and the last far precharge terminal turning on, before capture.
            next_capture = base + 1.2 * period
            lines = [f'{array}:{pin}{col}_far' for col in range(metadata['cols']) for pin in ('BL', 'BLB')]
            restored = [crossing for node in lines + ['RBL']
                        for crossing in edges(node, .9 * vdd, base + .7 * period, next_capture, True)]
            if restored:
                restore_slack.append(float((next_capture - max(restored)) * 1e12))
            precharge_on = [crossing for col in range(metadata['cols'])
                            for crossing in edges(nodes['pre'][str(col)], .9 * vdd, base + .7 * period,
                                                  next_capture, False)]
            if precharge_on:
                precharge_slack.append(float((next_capture - max(precharge_on)) * 1e12))

    for name, values in (
        ('min_sense_margin_v', sense_margins),
        ('min_wl_off_before_capture_ps', capture_slack),
        ('min_driver_release_after_wl_ps', driver_slack),
        ('min_wl_off_before_sense_ps', sense_slack),
        ('min_isolation_before_enable_ps', isolation_slack),
        ('min_driver_on_before_wl_ps', drive_setup),
        ('min_enable_off_before_precharge_ps', recovery_slack),
        ('min_storage_polarity_margin_v', storage_margins),
        ('min_read_output_margin_ps', output_margin),
        ('min_write_wl_after_flip_ps', write_dwell),
        ('min_restore_before_capture_ps', restore_slack),
        ('min_precharge_on_before_capture_ps', precharge_slack),
    ):
        if values:
            metrics[name] = min(values)
            if name.endswith('_ps'):
                checks[f'{name}_nonnegative'] = bool(metrics[name] >= 0.)
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        'passed': not failures, 'checks': len(checks), 'failures': failures,
        'metrics': metrics, 'metadata': metadata,
        'scorer_sha256': SCORER_SHA256,
    }
    (directory / report_name).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import sys
    for argument in sys.argv[1:]:
        result = score(argument)
        print(argument, result['passed'], result['checks'], result['metrics'], result['failures'][:20])
