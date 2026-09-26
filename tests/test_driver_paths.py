"""Structural regression checks for qualification circuit paths."""

import io
import unittest
from contextlib import redirect_stdout

from PySpice.Unit import u_Ohm, u_pF

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.subcircuits.decoder import DECODER_CASCADE
from sram_compiler.subcircuits.dummy_row_or_column import Dummy_Cell
from sram_compiler.subcircuits.standard_cell import Pinv
from sram_compiler.subcircuits.time_generate import ReplicaDelayChain, TaperedBuffer
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


class PathTests(unittest.TestCase):
    def test_write_startup_initializes_buffer_and_register_feedback_nodes(self):
        import tempfile
        from pathlib import Path
        for operation in ('read', 'write', 'read&write'):
            for sweep in (False, True):
                with self.subTest(operation=operation, sweep=sweep), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                    if sweep:
                        (Path(temp) / 'param_sweep_models.data').write_text(
                            'pmos_model_pu nmos_model_pd nmos_model_pg\n0 0 0\n')
                    tb = Sram6TCoreMcTestbench(load_config(4, 4, 'TT'), variation_mode='nominal',
                                              sweep_cell=sweep, sim_path=temp)
                    tb.set_vdd(.9)
                    circuit = tb.create_testbench(operation, 3, 3)
                    simulator = circuit.simulator(simulator='xyce-serial', temperature=25)
                    tb.add_meas_and_print(simulator, tb.data_init(), operation)
                    initial = '\n'.join(line for line in str(simulator).splitlines() if line.lower().startswith('.ic')).upper()
                    for col in range(4):
                        if operation == 'read':
                            self.assertNotIn(f'DIN_HOLD{col}', initial)
                            self.assertNotIn('XDFF_BUF_DATA', initial)
                        else:
                            self.assertIn(f'V(DIN_BUF{col})=0', initial)
                            self.assertNotIn('XDIN_HOLD', initial)
                            self.assertIn(f'V(XTIME_CONTROL:XDFF_BUF_DATA:XDFF_{col}:Z5)=0.9', initial)

    def test_generated_decks_start_from_the_precharged_clock_low_state(self):
        """Every bitline pair, RBL/RBLB and the sense nodes start at VDD (V2.2.2).

        With the clock low at t = 0 the controller keeps PRE active and the
        sense pass gates open, so forcing BL, RBL or SA_Q to 0 V described a
        state the circuit is never in; Xyce 7.4 could not solve one per-device
        FF 10T mux read from it (`8x4_10t_mux_FF_read_pd`, V2.2.1 record).
        """
        import tempfile
        for operation in ('read', 'write', 'read&write'):
            for mux in (False, True):
                with self.subTest(operation=operation, mux=mux), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                    tb = Sram6TCoreMcTestbench(load_config(4, 4, 'TT'), variation_mode='nominal',
                                              choose_columnmux=mux, sim_path=temp)
                    tb.set_vdd(.9)
                    circuit = tb.create_testbench(operation, 3, 3)
                    simulator = circuit.simulator(simulator='xyce-serial', temperature=25)
                    tb.add_meas_and_print(simulator, tb.data_init(), operation)
                    initial = ' '.join(line for line in str(simulator).splitlines() if line.lower().startswith('.ic')).upper()
                    for col in range(4):
                        self.assertIn(f'V(BL{col})=0.9V', initial)
                        self.assertIn(f'V(BLB{col})=0.9V', initial)
                        self.assertNotIn(f'V(BL{col})=0V', initial)
                    self.assertIn('V(RBL)=0.9V', initial)
                    self.assertIn('V(RBLB)=0.9V', initial)
                    self.assertNotIn('V(RBL)=0V', initial)
                    for group in range(4 // (2 if mux else 1)):
                        if operation == 'write':
                            self.assertNotIn(f'V(SA_Q{group})', initial)
                        else:
                            self.assertIn(f'V(SA_Q{group})=0.9V', initial)
                            self.assertIn(f'V(SA_QB{group})=0.9V', initial)
                    if operation == 'read':
                        # The target cell stores 0, so a completed read is OUT falling.
                        self.assertIn('V(OUT)=0.9V', initial)
                    # V2.2.4: both request registers start having latched "idle", so
                    # neither output fights its own slave latch at t = 0.
                    self.assertIn('V(CS_BAR)=0.9V', initial)
                    self.assertIn('V(XTIME_CONTROL:XDFF_BUF:QINT)=0.9V', initial)
                    self.assertIn('V(WE)=0V', initial)
                    self.assertIn('V(WE_BAR)=0.9V', initial)
                    self.assertIn('V(XTIME_CONTROL:XDFF_BUF1:QINT)=0.9V', initial)

    def test_wide_buffer_fingers_preserve_total_transistor_width(self):
        with redirect_stdout(io.StringIO()):
            inv = Pinv('NMOS_VTG', 'PMOS_VTG', 20e-6, 30e-6, 50e-9, max_finger_width=2e-6)
            legacy = Pinv('NMOS_VTG', 'PMOS_VTG', 20e-6, 30e-6, 50e-9)
        self.assertEqual(float(inv['Mpinv_nmos'].width), 20e-6)
        self.assertIn('NF=10', str(inv['Mpinv_nmos']))
        self.assertIn('NF=15', str(inv['Mpinv_pmos']))
        self.assertNotIn('NF=', str(legacy))

    def test_delay_chain_odd_counts_and_polarity(self):
        for stages in (1, 5, 9, 11):
            with self.subTest(stages=stages), redirect_stdout(io.StringIO()):
                chain = ReplicaDelayChain(stages=stages)
                lines = str(chain).splitlines()
                drivers = [line for line in lines if line.startswith('Xdinv')]
                self.assertEqual(len(drivers), stages)
                self.assertIn(' in ', drivers[0])
                self.assertIn(' out ', drivers[-1])
                self.assertEqual(sum(line.startswith('Xdload') for line in lines), 4 * stages)
        for stages in (0, 2, True, 3.5):
            with self.assertRaises(ValueError):
                ReplicaDelayChain(stages=stages)

    def test_buffer_effort_adds_stages_at_large_loads(self):
        with redirect_stdout(io.StringIO()):
            small = TaperedBuffer('small', 8, effort_based=True)
            large = TaperedBuffer('large', 4096, effort_based=True)
            legacy = TaperedBuffer('legacy', 4096)
        self.assertEqual(small.n_stages, 2)
        self.assertEqual(large.n_stages, 6)
        self.assertEqual(legacy.n_stages, 4)
        self.assertLessEqual(large.drive_scale ** (1 / large.n_stages), 8)
        with redirect_stdout(io.StringIO()):
            iso = TaperedBuffer('iso', 16, effort_based=True, load_units=64)
        self.assertEqual(iso.n_stages, 2)
        with redirect_stdout(io.StringIO()):
            strong = TaperedBuffer('strong', 16, effort_based=True, load_units=64, fall_strength=2)
        final = list(strong.subcircuits)[-1]
        self.assertEqual(float(final.nmos_width), 0.09e-6 * 16 * 2)
        self.assertEqual(float(final.pmos_width), 0.27e-6 * 16)

    def test_dummy_rc_does_not_create_an_unused_qb_island(self):
        for pin_rc in (False, True):
            with self.subTest(cell_pin_rc=pin_rc), redirect_stdout(io.StringIO()):
                dummy = Dummy_Cell('NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG',
                                   .205e-6, .09e-6, .135e-6, 50e-9,
                                   w_rc=True, cell_pin_rc=pin_rc)
            self.assertNotIn('RR_QB', str(dummy))
            self.assertIn('RR_Q', str(dummy))
            self.assertEqual('RR_WL' in str(dummy), pin_rc)

    def test_decoder_scales_only_final_output_inverters(self):
        with redirect_stdout(io.StringIO()):
            decoder = DECODER_CASCADE('NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', 'PMOS_VTG',
                                      num_rows=64, output_scale=4)
        early = decoder.decoders_by_level[0][0]
        last = decoder.decoders_by_level[-1][0]
        self.assertNotEqual(early.NAME, last.NAME)
        self.assertEqual(float(last.inv_A0.nmos_width), 0.09e-6)
        self.assertIn('w=3.6e-07', str(last.and_for_en[0]))
        self.assertNotIn('w=3.6e-07', str(early.and_for_en[0]))

    def test_matched_replica_and_canonical_read(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(8, 4, 'TT')
            cfg.global_config.sizing = {'mode': 'rules_only', 'replica': {'K': 2, 'N': 5}}
            sizes = resolve_driver_sizes(cfg)
            cfg.sram_6t_cell.nmos_width.value[1] *= 0.8
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False, driver_sizes=sizes)
            deck = str(tb.create_testbench('read', 7, 3))
        self.assertIn('XRWL VDD VSS VDD wl_en_line_far RWL WORDLINEDRIVER', deck)
        self.assertEqual(sum(line.startswith('XRWL_LOAD_') for line in deck.splitlines()), 2)
        replica_instance = next(line for line in deck.splitlines() if line.startswith('Xsram_8x1_replica_column '))
        self.assertEqual([node for node in replica_instance.split() if node.startswith('RWL_tap')],
                         ['RWL_tap2', 'RWL_tap3'])
        self.assertEqual(sum(line.startswith('XWRITEDRIVER_') for line in deck.splitlines()), 4)
        self.assertIn('VDD VSS w_en_line_tap3 VSS BL3_periph_tap1 BLB3_periph_tap1 WRITEDRIVER', deck)
        # V2.1.6: the replica write driver writes a 0 with the real drivers (write slot).
        self.assertIn('XREPLICA_WDRV_LOAD VDD VSS w_en_line_far VSS RBL_periph_tap1 RBLB_periph_tap1 WRITEDRIVER', deck)
        self.assertEqual(sizes.replica_nmos_widths[1], 0.135e-6)

    def test_replica_load_cells_neither_leak_nor_short(self):
        # V2.2.4: only the K cells on the replica wordline may pull RBL. Through
        # V2.2.3 every tied-off load cell also stored 0, and their leakage fired
        # the sense enable early at 512 rows FS/FF 125 C, below the sense bar. A
        # load cell must hold both pass-gate nodes at VDD (no drain-source
        # voltage against the precharged RBL/RBLB) without any static VDD-VSS
        # path; the first V2.2.4 passive 6T cell shorted its right inverter.
        def blocks(deck):
            found, stack = {}, []
            for line in deck.splitlines():
                words = line.split()
                if not words:
                    continue
                if line.lower().startswith('.subckt'):
                    stack.append(words[1])
                    found[words[1]] = []
                elif line.lower().startswith('.ends'):
                    stack.pop()
                elif stack:
                    found[stack[-1]].append(words)
            return found

        def static_levels(body, wordline):
            """DC levels (1/0) implied by the rails, the idle wordline and on devices."""
            levels = {'VDD': 1, 'VSS': 0, 'WL': wordline}
            devices = [(w[0][0].upper(), w[1:4] if w[0][0].upper() == 'M' else w[1:3], w[5] if w[0][0].upper() == 'M' else '')
                       for w in body if w[0][0].upper() in 'MR']
            for _ in range(len(devices) + 1):
                for kind, pins, model in devices:
                    if kind == 'M':
                        drain, gate, source = pins
                        on = levels.get(gate) == (0 if model.upper().startswith('P') else 1)
                        ends = (drain, source)
                    else:
                        on, ends = True, pins
                    if on:
                        for a, b in (ends, ends[::-1]):
                            if a in levels and b not in levels:
                                levels[b] = levels[a]
            shorts = [pins for kind, pins, model in devices if kind == 'M'
                      and levels.get(pins[1]) == (0 if model.upper().startswith('P') else 1)
                      and {levels.get(pins[0]), levels.get(pins[2])} == {0, 1}]
            return levels, shorts

        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for w_rc in (False, True):
                with self.subTest(cell=cell, w_rc=w_rc), redirect_stdout(io.StringIO()):
                    cfg = load_config(8, 4, 'TT')
                    cfg.global_config.sram_cell_type = cell
                    cfg.global_config.sizing = {'mode': 'rules_only', 'replica': {'K': 2, 'N': 5}}
                    tb = Sram6TCoreTestbench(cfg, sram_cell_type=cell, choose_columnmux=False, w_rc=w_rc)
                    deck = str(tb.create_testbench('read', 7, 3))
                found = blocks(deck)
                column = next(name for name in found if name.endswith('_replica_column'))
                # Rows on the replica wordline (the last K) use the driven cell, all others the load.
                cells = {int(w[0].rsplit('_', 1)[1]): w[-1] for w in found[column]
                         if w[0].startswith('XReplica_CELL_')}
                self.assertEqual(sorted(cells), list(range(8)))
                self.assertEqual({cells[row] for row in (6, 7)}, {'Replica_CELL'})
                self.assertEqual({cells[row] for row in range(6)}, {'Replica_CELL_PASSIVE'})
                for name, wordline in (('Replica_CELL', 1), ('Replica_CELL', 0), ('Replica_CELL_PASSIVE', 0)):
                    levels, shorts = static_levels(found[name], wordline)
                    self.assertEqual(shorts, [], f'{name} shorts VDD to VSS')
                    # The storage-side node of each pass gate (gate on WL, one end on a bitline).
                    pass_nodes = {pins[0] if not pins[0].startswith('RBL') else pins[2]
                                  for pins in ([w[1], w[2], w[3]] for w in found[name]
                                               if w[0].upper().startswith('M') and w[2].startswith('WL'))}
                    stored = {levels.get(node) for node in pass_nodes}
                    self.assertEqual(stored, {1} if name.endswith('PASSIVE') else {0, 1}, name)

    def test_replica_bitline_and_wordline_share_the_array_rc_configuration(self):
        def blocks(deck):
            found, stack = {}, []
            for line in deck.splitlines():
                if line.startswith('.subckt '):
                    stack.append(line.split()[1])
                    found.setdefault(stack[-1], [])
                elif line.startswith('.ends'):
                    stack.pop()
                elif stack:
                    found[stack[-1]].append(line)
                else:
                    found.setdefault('__top__', []).append(line)
            return found

        def count(lines, prefix):
            return sum(line.startswith(prefix) for line in lines)

        def instance_subckt(lines, name):
            return next(line.split()[-1] for line in lines if line.split()[:1] == [name])

        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mode in ('lookup', 'rules_only'):
                for mux in (False, True):
                    for w_rc in (True, False):
                        with self.subTest(cell=cell, mode=mode, mux=mux, w_rc=w_rc), \
                                redirect_stdout(io.StringIO()):
                            cfg = load_config(4, 4, 'TT')
                            cfg.global_config.sram_cell_type = cell
                            cfg.global_config.sizing = {'mode': mode}
                            tb = Sram6TCoreTestbench(cfg, sram_cell_type=cell, choose_columnmux=mux,
                                                     w_rc=w_rc, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF)
                            deck = str(tb.create_testbench('read', 3, 3))
                        found = blocks(deck)
                        top = found['__top__']
                        segments = 2 if w_rc else 0
                        # Wordline: the real and replica drivers carry the same output RC.
                        real_driver = instance_subckt(top, 'XWL_DRV_3')
                        replica_driver = instance_subckt(top, 'XRWL')
                        self.assertEqual(count(found[real_driver], 'RR_Z_'), segments)
                        self.assertEqual(count(found[replica_driver], 'RR_Z_'), segments)
                        self.assertEqual(count(found[replica_driver], 'RR_B_' if replica_driver == 'WORDLINEDRIVER' else 'RR_A_'), segments)
                        # Physical wires carry the pin loads; optional series pin
                        # stubs are off by default for real, replica and dummy cells.
                        array = instance_subckt(top, next(
                            line.split()[0] for line in top if line.startswith('XSRAM_') and '_CORE_' in line))
                        cell_block = instance_subckt(found[array], f'X{cell}_3_3')
                        replica_column = instance_subckt(top, next(
                            line.split()[0] for line in top if 'replica_column' in line))
                        replica_cell = instance_subckt(found[replica_column], 'XReplica_CELL_0')
                        stub = 0
                        for block, names in ((cell_block, ('RR_BL_', 'RR_BLB_', 'RR_WL_')),
                                             (replica_cell, ('RR_RBL_', 'RR_RBLB_', 'RR_WL_'))):
                            for name in names:
                                self.assertEqual(count(found[block], name), stub, (block, name))
                        dummy = instance_subckt(top, 'XRWL_LOAD_0')
                        self.assertEqual(count(found[dummy], 'RR_WL_'), stub)
                        # Bitline: replica precharge is the array precharge; the replica
                        # bitline reaches TIME_CONTROL through the sense-amplifier input segments.
                        self.assertEqual(instance_subckt(top, 'XPRECHARGE_RBL'), instance_subckt(top, 'XPRECHARGE_0'))
                        sense = instance_subckt(top, 'XSENSEAMP_0')
                        self.assertEqual(count(found[sense], 'RR_IN_'), segments)
                        replica_sense = instance_subckt(top, 'XREPLICA_SENSEAMP')
                        self.assertEqual(count(found[replica_sense], 'RR_IN_'), segments)
                        self.assertEqual(count(top, 'RR_RBL_SENSE_'), 0)
                        self.assertEqual(count(top, 'CCg_RBL_SENSE_'), 0)
                        time_line = next(line for line in top if line.startswith('XTIME_CONTROL '))
                        sense_node = ('XREPLICA_SENSEAMP:IN_end' if w_rc else
                                      ('RBL_MUX' if mux else 'RBL_periph_tap2'))
                        self.assertIn(f' {sense_node} ', time_line)
                        pre_node = 'XPRECHARGE_RBL:ENB_end' if w_rc else 'PRE_line_far'
                        self.assertIn(f' RWL_far {pre_node} ', time_line)

    def test_rc_precharge_waits_for_the_matched_physical_wordline(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(8, 4, 'TT')
            cfg.global_config.sizing = {'mode': 'rules_only'}
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False, w_rc=True)
            circuit = tb.create_testbench('read', 7, 3)
        time = next(block for block in circuit.subcircuits if block.name == 'TIME_CONTROL')
        self.assertTrue(tb.driver_sizes.replica_precharge_guard)
        self.assertEqual(time.NODES[-5:], ['rwl', 'pre_far', 'iso_far', 'sen_far', 'wen_far'])
        # V2.1.6: the precharge is additionally inhibited by the held write request.
        self.assertIn('clk_bar pre_ready enables_off PRE_UNBUF', str(time))
        self.assertEqual(tb.driver_sizes.precharge_guard_stages, 4)
        self.assertIn('RWL_far XPRECHARGE_RBL:ENB_end XREPLICA_SENSEAMP:ISO_end XREPLICA_SENSEAMP:EN_end XREPLICA_WDRV_LOAD:EN_end TIME_CONTROL', str(circuit['XTIME_CONTROL']))
        # An unmatched replica cannot represent the physical wordline load.
        cfg.global_config.sizing['replica'] = {'matched': False}
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'matched replica'):
            Sram6TCoreTestbench(cfg, choose_columnmux=False, w_rc=True)


if __name__ == '__main__':
    unittest.main()
