"""The wordline model check must build the star and distributed lines it claims."""

import unittest

from sram_compiler.sizing.wordline_model import build_deck


class WordlineModelTests(unittest.TestCase):
    def test_star_and_distributed_decks(self):
        star, sizes = build_deck(8, 'star')
        lines = star.splitlines()
        self.assertEqual(sum(line.startswith('XC') for line in lines), 8)
        self.assertTrue(all(' WL_0 SRAM_6T_CELL' in line for line in lines if line.startswith('XC')))
        self.assertFalse(any(line.startswith('RW') for line in lines))
        self.assertIn('RR_WL_0 WL WL_end', star)      # the compiler stub stays on every cell pin
        self.assertIn('RR_Z_1 Z_seg0 Z_end', star)    # the driver keeps its two output segments
        self.assertEqual(sizes.wl_inv, 2.0)
        dist, _ = build_deck(8, 'pitch', 1.0, 1e-16)
        lines = dist.splitlines()
        self.assertEqual(sum(line.startswith('RW') for line in lines), 7)
        self.assertEqual(sum(line.startswith('CW') for line in lines), 7)
        self.assertIn('XC7 VDD 0 VDD VDD WL_7 SRAM_6T_CELL', dist)
        self.assertIn('TARG V(XC7:WL_end)', dist)
        with self.assertRaises(ValueError):
            build_deck(8, 'unit', 0.0, 0.0)


if __name__ == '__main__':
    unittest.main()
