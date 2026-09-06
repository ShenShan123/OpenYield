"""Compatibility shim: the configuration loader lives in
``sram_compiler/config_yaml/config.py``.  This module re-exports it so that the
scripts doing ``from config import SRAM_CONFIG`` keep working without a second,
near-identical copy of the loader that could drift.
"""
from sram_compiler.config_yaml.config import (  # noqa: F401
    AttrDict,
    GlobalConfig,
    load_global_config,
    Parameter,
    CircuitConfig,
    ConfigLoader,
    SRAM_CONFIG,
)

__all__ = [
    "AttrDict", "GlobalConfig", "load_global_config",
    "Parameter", "CircuitConfig", "ConfigLoader", "SRAM_CONFIG",
]
