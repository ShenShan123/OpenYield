"""Parallel execution must preserve local distributions and candidate draws."""

import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist

from sram_compiler.per_device_mc.sampling import materialize_decks, standard_normal_draws


class LocalSamplingTests(unittest.TestCase):
    def test_independent_reproducible_latin_hypercube_streams(self):
        first = standard_normal_draws(17, 'MC_left:vth0', 100)
        self.assertEqual(first, standard_normal_draws(17, 'mc_LEFT:VTH0', 100))
        self.assertNotEqual(first, standard_normal_draws(18, 'MC_left:vth0', 100))
        self.assertNotEqual(first, standard_normal_draws(17, 'MC_right:vth0', 100))
        self.assertNotEqual(first, standard_normal_draws(17, 'MC_left:u0', 100))
        strata = sorted(int(NormalDist().cdf(value) * 100) for value in first)
        self.assertEqual(strata, list(range(100)))

    def test_numeric_models_keep_all_devices_and_mean_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'random.spice'
            blocks = ['.model MC_' + name + ' NMOS\n+ vth0={AGAUSS(0.4, 0.02, 1)} '
                      'u0={AGAUSS(0.04, 0.002, 1)} voff={AGAUSS(-0.1, 0.005, 1)} '
                      'tox=1.234567890123e-9\n' for name in ('left', 'right')]
            model.write_text(''.join(blocks))
            deck = f'Materialization\n.include "{model}"\n.SAMPLING useExpr=true\n.options samples numsamples=3\n.TRAN 1p 1n\n.end\n'
            first = materialize_decks(deck, model, root / 'first', 17, 3)
            self.assertEqual(first['independent_models'], 2)
            self.assertEqual(first['sampled_parameters'], 6)
            self.assertEqual(len(first['decks']), 3)
            for entry in first['decks']:
                text = Path(entry['model']).read_text()
                self.assertNotIn('AGAUSS', text)
                self.assertEqual(text.count('tox=1.234567890123e-9'), 2)
                simulation = Path(entry['deck']).read_text()
                self.assertNotIn('.SAMPLING', simulation)
                self.assertIn('.TRAN 1p 1n', simulation)
                self.assertIn(entry['model'], simulation)
            # Inserting/reordering unrelated devices must not reassign a cell's draw.
            model.write_text(''.join(reversed(blocks)))
            second = materialize_decks(deck, model, root / 'second', 17, 3)
            def by_model(path):
                return {block.split()[0]: block for block in Path(path).read_text().split('.model ')[1:]}
            for a, b in zip(first['decks'], second['decks']):
                self.assertEqual(by_model(a['model']), by_model(b['model']))


if __name__ == '__main__':
    unittest.main()
