"""Route experiments through main_sram.py, importing optional tools on demand."""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

from sram_compiler.version import VERSION


LAYOUT_ACTIONS = {
    'reference': 'run',
    'full': 'full_macro',
    'scaled': 'scaled_macro',
    'compact': 'compact_macro',
    'cells': 'cell_flow',
    'export': 'export',
    'analyze': 'analyze',
    'compare': 'compare',
    'compare-full': 'compare_full',
    'compare-snm': 'compare_snm',
    'verify-full': 'verify_full',
    'reduce-rc': 'reduce_rc',
    'split-rc': 'split_rc',
    'waveform-metrics': 'waveform_metrics',
}
CIRCUIT_ALGORITHMS = (
    'sa', 'pso', 'cmaes', 'random', 'smac', 'cbo', 'roseopt',
    'cpn', 'tssbo', 'mobo', 'moead', 'nsgaii',
)


def call_module(module_name: str, argv: Sequence[str]) -> int:
    """Adapt existing argparse entrances while preserving the caller's argv."""
    previous = sys.argv
    try:
        sys.argv = ['main_sram.py', *argv]
        module = importlib.import_module(module_name)
        result = module.main()
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous


def circuit_optimize(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog='main_sram.py circuit-optimize',
                                     description='Run a circuit-backed sizing optimizer.')
    parser.add_argument('--algorithm', choices=CIRCUIT_ALGORITHMS, default='sa')
    parser.add_argument('--config', help='optimizer YAML (default: size_optimization/config_sram.yaml)')
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_iter is not None and args.max_iter <= 0:
        parser.error('--max-iter must be positive')
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    config = Path(args.config).expanduser() if args.config else root / 'size_optimization/config_sram.yaml'
    module = importlib.import_module(f'size_optimization.demo_{args.algorithm}')
    from size_optimization.exp_utils import seed_set
    seed_set(args.seed)
    options = {'config_path': str(config.resolve())}
    if args.max_iter is not None:
        options['max_iter'] = args.max_iter
    module.main(**options)
    return 0


def layout(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog='main_sram.py layout',
                                     description='FreePDK45 layout and RC calibration tools.')
    parser.add_argument('action', choices=LAYOUT_ACTIONS)
    parser.add_argument('options', nargs=argparse.REMAINDER,
                        help='options for the selected tool; ACTION --help lists them')
    args = parser.parse_args(argv)
    return call_module(f'sram_compiler.layout.freepdk45.{LAYOUT_ACTIONS[args.action]}', args.options)


def main(argv: Sequence[str], *, simulation_defaults=None) -> int:
    """Simulation accepts an explicit subcommand or direct flags for scripting."""
    parser = argparse.ArgumentParser(prog='main_sram.py',
                                     description=f'OpenYield {VERSION}: unified experiment entrance.')
    parser.add_argument('--version', action='version', version=VERSION)
    sub = parser.add_subparsers(dest='command')
    for command, help_text in (
        ('simulate', 'Generate or simulate 6T/10T transient and SNM decks (default).'),
        ('equivalent', 'Compare equivalent modes with the full transistor array.'),
        ('layout', 'Generate, extract, compare and analyze FreePDK45 layouts.'),
        ('optimize', 'Run an offline optimization family.'),
        ('compare', 'Compare offline optimization algorithms.'),
        ('experiment', 'Run the configured offline comparison campaign.'),
        ('circuit-optimize', 'Run a circuit-backed sizing algorithm.'),
        ('circuit-experiment', 'Run the interactive joint or fixed-configuration search.'),
    ):
        sub.add_parser(command, add_help=False, help=help_text)
    values = list(argv)
    commands = set(sub.choices)
    if not values or (values[0].startswith('--') and values[0] not in ('--help', '--version')):
        values.insert(0, 'simulate')
    if values and values[0] in commands:
        command, forwarded = values[0], values[1:]
        if forwarded[:1] == ['--']:
            forwarded = forwarded[1:]
        if command == 'simulate':
            from . import simulation
            return simulation.main(forwarded, defaults=simulation_defaults)
        if command == 'layout':
            return layout(forwarded)
        if command == 'circuit-optimize':
            return circuit_optimize(forwarded)
        if command == 'circuit-experiment':
            interactive = argparse.ArgumentParser(prog='main_sram.py circuit-experiment',
                                                   description='Interactive joint architecture/sizing or fixed-configuration search.')
            interactive.parse_args(forwarded)
            return call_module('size_optimization.experiment', [])
        if command in ('optimize', 'compare'):
            return call_module('size_optimization.openyield_v2.run', [command, *forwarded])
        module = ('sram_compiler.experiments.equivalent' if command == 'equivalent'
                  else 'size_optimization.openyield_v2.run_experiment')
        return call_module(module, forwarded)
    parser.parse_args(values)
    return 0
