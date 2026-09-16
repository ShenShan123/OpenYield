"""Every construction path retains physical wire taps without opting in."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from sram_compiler.config_yaml.config import GlobalConfig
from sram_compiler.interconnect import InterconnectConfig, load_interconnect, resolve_interconnect
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits import sram_cell_add_equivalent as equivalent
from sram_compiler.subcircuits.dummy_row_or_column import Dummy_Column, Dummy_Row
from sram_compiler.subcircuits.replica_column import ReplicaColumn
from sram_compiler.subcircuits.sram_6t_core import Sram6TCore
from sram_compiler.subcircuits.sram_10t_core import Sram10TCore


class DistributedDefaultsTests(unittest.TestCase):
    def test_omitted_settings_match_explicit_reference_geometry(self):
        reference = resolve_interconnect(load_interconnect(
            'sram_compiler/config_yaml/interconnect_example.yaml'))
        for settings in (None, {}, {'mode': 'distributed'}, InterconnectConfig(),
                         GlobalConfig({}).interconnect):
            with self.subTest(settings=settings):
                resolved = resolve_interconnect(settings)
                self.assertEqual(resolved, reference)
                self.assertFalse(resolved.cell_pin_rc)
                self.assertEqual(physical_context(interconnect=settings),
                                 physical_context(interconnect=reference))

    def test_removed_topology_is_rejected_before_deck_generation(self):
        for options in ({'mode': 'star'}, {'mode': 'star', 'cell_pin_rc': False}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'distributed'):
                resolve_interconnect(options)
        with self.assertRaisesRegex(ValueError, 'distributed'):
            InterconnectConfig(mode='star')

    def test_default_arrays_keep_taps_with_numeric_and_sweep_dimensions(self):
        for core_type in (Sram6TCore, Sram10TCore):
            for sweep in (False, True):
                for w_rc in (False, True):
                    kwargs = dict(pd_width='pd', pu_width='pu', pg_width='pg',
                                  length='length') if sweep else {}
                    if core_type is Sram10TCore:
                        kwargs['fd_nmos_model'] = 'NMOS_VTG'
                        if sweep:
                            kwargs['fd_width'] = 'fd'
                    with self.subTest(core=core_type.__name__, sweep=sweep, w_rc=w_rc), \
                            redirect_stdout(io.StringIO()):
                        core = core_type(2, 3, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG',
                                         w_rc=w_rc, **kwargs)
                    cells = [str(e).split() for e in core.elements if e.name.startswith('XSRAM')]
                    self.assertEqual(len(cells), 6)
                    for row in range(2):
                        for col in range(3):
                            cell = next(parts for parts in cells if parts[0].endswith(f'_{row}_{col}'))
                            self.assertEqual(cell[3:6], [f'BL{col}_tap{row}',
                                                        f'BLB{col}_tap{row}', f'WL{row}_tap{col}'])
                    self.assertIn('Rwire_WL0_', str(core))
                    self.assertNotIn('RR_WL_', str(list(core.subcircuits)[0]))

    def test_replica_has_exact_array_height_and_tapped_bitlines(self):
        with redirect_stdout(io.StringIO()):
            replica = ReplicaColumn(4, 3, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', 'NMOS_VTG')
        cells = [str(e).split() for e in replica.elements if e.name.startswith('XReplica_CELL')]
        self.assertEqual(len(cells), 4)
        self.assertEqual(replica.cell_count, 4)
        for row, cell in enumerate(cells):
            self.assertEqual(cell[3:6], [f'RBL_tap{row}', f'RBLB_tap{row}', f'WL{row}'])
        self.assertNotIn('WL4', replica.NODES)

    def test_dummy_arrays_have_wire_segments_between_consumers(self):
        args = ('NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', .205e-6, .09e-6, .135e-6, 50e-9)
        with redirect_stdout(io.StringIO()):
            column = Dummy_Column(4, *args, w_rc=True)
            row = Dummy_Row(4, *args, w_rc=True)
        column_cells = [str(e).split() for e in column.elements if e.name.startswith('XDummy_CELL')]
        row_cells = [str(e).split() for e in row.elements if e.name.startswith('XDummy_CELL')]
        for index, cell in enumerate(column_cells):
            self.assertEqual(cell[3:5], [f'BL_tap{index}', f'BLB_tap{index}'])
        for index, cell in enumerate(row_cells):
            self.assertEqual(cell[5], f'WL_tap{index}')
        self.assertIn('Rwire_BL_', str(column))
        self.assertIn('Rwire_WL_', str(row))
        self.assertNotIn('RR_WL_', str(list(row.subcircuits)[0]))

    def test_omitted_cells_preserve_local_loads_without_explicit_wire_options(self):
        caps = {'caps': dict(c_wl=1e-16, c_bl=2e-16, c_blb=3e-16,
                             c_wl_bl=1e-17, c_wl_blb=2e-17)}
        for core_type in (Sram6TCore, Sram10TCore):
            for mode in (1, 2, 3, 4):
                args = ('NMOS_VTG',) if core_type is Sram10TCore else ()
                with self.subTest(core=core_type.__name__, mode=mode), redirect_stdout(io.StringIO()), \
                        patch.object(equivalent, '_cached_extraction',
                                     side_effect=lambda tester, name, fn: caps if name == 'caps' else 1e12):
                    core = core_type(3, 3, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', *args,
                                     real_cell_mode=mode, target_row=2, target_col=2, w_rc=True)
                coupling = [str(e).split() for e in core.elements if e.name.startswith('Ceq_WLBL_')]
                self.assertEqual(len(coupling), core._count_total_unused_cells())
                self.assertTrue(all('_tap' in e[1] and '_tap' in e[2] for e in coupling))
                self.assertNotIn('_rc_mid', str(core))
                self.assertNotIn('Ccap_WL', str(core))


if __name__ == '__main__':
    unittest.main()
