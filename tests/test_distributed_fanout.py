"""Decoder and register wire topology; DC logic checks do not qualify timing."""

import contextlib
import io
import unittest

from sram_compiler.interconnect import resolve_interconnect
from sram_compiler.subcircuits.decoder import DECODER_CASCADE
from sram_compiler.subcircuits.standard_cell import AND2, AND3, Pinv
from sram_compiler.subcircuits.time_generate import DataRegister
from sram_compiler.testbenches.parameter_factor import (
    DecoderCascadeFactory, DummyColumnFactory, DummyRowFactory, TimeControlFactory,
)


def wire_config(refinement=1):
    wire = dict(layer='M3', pitch_m=.6e-6, width_m=.1e-6,
                sheet_resistance_ohm=1/6, capacitance_f_per_m=1e-16/.6e-6,
                sections_per_half_pitch=refinement)
    return dict(mode='distributed', wl=dict(wire, layer='M2'), bl=wire)


def line_elements(circuit, prefix, kind):
    return [e for e in circuit.elements if e.name.startswith(f'{kind}wire_{prefix}_')]


def evaluate_decoder(circuit, address):
    """Collapse finite wire resistance at DC and evaluate actual gate pin wiring."""
    parent, gates = {}, []

    def root(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = root(parent[node])
        return parent[node]

    def expand(block, scope, pins):
        def node(name):
            return pins.get(name, f'{scope}:{name}')

        if isinstance(block, (Pinv, AND2, AND3)):
            gates.append((isinstance(block, Pinv), [node(n) for n in block.external_nodes[2:]]))
            return
        definitions = {sub.name: sub for sub in block.subcircuits}
        for element in block.elements:
            nodes = [node(n) for n in element.node_names]
            if element.name.startswith('R'):
                parent[root(nodes[0])] = root(nodes[1])
            elif element.name.startswith('X'):
                child = definitions[element.subcircuit_name]
                expand(child, f'{scope}:{element.name}', dict(zip(child.external_nodes, nodes)))

    expand(circuit, '', {n: n for n in circuit.external_nodes})
    values = {root('VDD'): True, root('VSS'): False}
    values.update({root(f'A{i}'): bool(address & (1 << i)) for i in range(circuit.n_bits)})
    pending = list(gates)
    while pending:
        progressed = False
        for gate in pending[:]:
            invert, nodes = gate
            inputs, output = [root(n) for n in nodes[:-1]], root(nodes[-1])
            if all(n in values for n in inputs):
                values[output] = not values[inputs[0]] if invert else all(values[n] for n in inputs)
                pending.remove(gate)
                progressed = True
        if not progressed:
            raise AssertionError('Decoder contains unconnected or cyclic gate inputs')
    return [values[root(f'WL{i}')] for i in range(circuit.num_rows)]


class DistributedFanoutTests(unittest.TestCase):
    def test_decoder_address_and_predecode_fanout_conserve_row_height(self):
        for rows in (1, 2, 4, 8, 10, 64, 65):
            for refinement in (1, 3):
                with self.subTest(rows=rows, refinement=refinement), contextlib.redirect_stdout(io.StringIO()):
                    decoder = DECODER_CASCADE('N', 'P', 'N', 'P', num_rows=rows,
                                              interconnect=wire_config(refinement))
                    for bit in range(decoder.n_bits):
                        self.assertAlmostEqual(sum(float(e.resistance) for e in
                                                   line_elements(decoder, f'A{bit}_line', 'R')), rows)
                        self.assertAlmostEqual(sum(float(e.capacitance) for e in
                                                   line_elements(decoder, f'A{bit}_line', 'C')) / 1e-16, rows)
                    for level, blocks in enumerate(decoder.decoders_by_level):
                        span = rows / len(blocks)
                        for block in blocks:
                            for net in ('A0', 'A1', 'A2', 'A0b', 'A1b', 'A2b', 'EN'):
                                resistors = line_elements(block, f'{net}_line', 'R')
                                self.assertEqual(len(resistors), 16 * refinement)
                                self.assertAlmostEqual(sum(float(e.resistance) for e in resistors), span)
                                self.assertAlmostEqual(sum(float(e.capacitance) for e in
                                                           line_elements(block, f'{net}_line', 'C')) / 1e-16, span)
                            enable_nodes = [block[f'XAND_EN{i}'].node_names[-2] for i in range(8)]
                            self.assertEqual(len(set(enable_nodes)), 8)
                            self.assertNotIn('EN', enable_nodes)
                        pins = [decoder[f'XDEC_{level}_{i}'].node_names[3:6] for i in range(len(blocks))]
                        for index in range(3):
                            active = [p[index] for p in pins if p[index] != 'VSS']
                            self.assertEqual(len(set(active)), len(active))

    def test_decoder_keeps_binary_address_polarity_including_unused_codes(self):
        for rows in (1, 2, 4, 8, 10, 64, 65):
            for sweep in (False, True):
                dims = dict(inv_nmos_width='inv_n', inv_pmos_width='inv_p',
                            nand_nmos_width='nand_n', nand_pmos_width='nand_p', length='length') if sweep else {}
                with self.subTest(rows=rows, sweep=sweep), contextlib.redirect_stdout(io.StringIO()):
                    decoder = DECODER_CASCADE('N', 'P', 'N', 'P', num_rows=rows,
                                              output_scale=4, interconnect=wire_config(), **dims)
                    for address in range(1 << decoder.n_bits):
                        self.assertEqual(evaluate_decoder(decoder, address),
                                         [address == row for row in range(rows)])

    def test_write_register_clock_has_a_centered_tap_per_column(self):
        for columns in (1, 4, 64, 512):
            for refinement in (1, 3):
                with self.subTest(columns=columns, refinement=refinement), contextlib.redirect_stdout(io.StringIO()):
                    register = DataRegister(num_cols=columns, interconnect=wire_config(refinement))
                    resistors = line_elements(register, 'CLK_line', 'R')
                    self.assertEqual(len(resistors), 2 * columns * refinement)
                    self.assertAlmostEqual(sum(float(e.resistance) for e in resistors), columns)
                    self.assertAlmostEqual(sum(float(e.capacitance) for e in
                                               line_elements(register, 'CLK_line', 'C')) / 1e-16, columns)
                    for col in range(columns):
                        self.assertEqual(register[f'Xdff_{col}'].node_names[-1], f'CLK_line_tap{col}')

    def test_factories_preserve_custom_geometry_through_nested_time(self):
        config = resolve_interconnect(wire_config(3))
        with contextlib.redirect_stdout(io.StringIO()):
            decoder = DecoderCascadeFactory('N', 'P', 'N', 'P', num_rows=10, interconnect=config).create()
            time = TimeControlFactory(num_rows=10, num_cols=4, operation='write', interconnect=config).create()
        self.assertEqual(decoder.interconnect, config)
        register = next(sub for sub in time.subcircuits if sub.name == 'DATA_REGISTER')
        self.assertEqual(register.interconnect, config)
        self.assertEqual(len(line_elements(register, 'CLK_line', 'R')), 24)

    def test_dummy_factories_preserve_the_same_wire_configuration(self):
        config = resolve_interconnect(wire_config(3))
        with contextlib.redirect_stdout(io.StringIO()):
            column = DummyColumnFactory(4, 'N', 'P', 'N', interconnect=config).create()
            row = DummyRowFactory(4, 'N', 'P', 'N', interconnect=config).create()
        self.assertEqual(column.interconnect, config)
        self.assertEqual(row.interconnect, config)

    def test_default_decoder_and_write_register_are_distributed(self):
        with contextlib.redirect_stdout(io.StringIO()):
            decoder = DECODER_CASCADE('N', 'P', 'N', 'P', num_rows=4)
            register = DataRegister(num_cols=4)
        self.assertEqual(decoder.interconnect.mode, 'distributed')
        self.assertEqual(register.interconnect.mode, 'distributed')
        self.assertTrue(line_elements(decoder, 'A0_line', 'R'))
        self.assertTrue(line_elements(register, 'CLK_line', 'R'))


if __name__ == '__main__':
    unittest.main()
