"""Equivalent array modelling, selected as a simulation input.

Mode 0 keeps every array transistor and is the reference: it is the only mode
whose per-device local mismatch covers the whole array.  Modes 1 to 4 keep the
transistors the addressed access touches and replace the remaining cells with
the extracted five-capacitor network and a static-leakage load at the same
local wire tap, which is an approximation (see ``README.md`` and
``sram_compiler/subcircuits/sram_cell_add_equivalent.py`` for the model).

The physical wires are independent of this option: every mode keeps every wire
segment of ``sram_compiler/interconnect.py``.

``global.yaml`` carries the default under ``equivalent:``; an explicit
``real_cell_mode`` argument or command-line option overrides it, exactly as the
``interconnect`` block and ``--interconnect-config`` do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

MODES: dict[int, str] = {
    0: "full transistor array (reference; complete per-device coverage)",
    1: "cross: the target row and the target column stay real",
    2: "the target row stays real",
    3: "the target column stays real",
    4: "only the target cell stays real",
}


@dataclass(frozen=True)
class EquivalentConfig:
    """Resolved equivalent-cell selection for one simulation."""

    mode: int = 0

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in MODES:
            raise ValueError(
                f"equivalent mode must be one of {sorted(MODES)}, got {mode!r}"
            )

    @property
    def approximate(self) -> bool:
        """True when unused cells are replaced by extracted equivalent loads."""
        return self.mode != 0

    def describe(self) -> str:
        return f"mode {self.mode}: {MODES[self.mode]}"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "approximate": self.approximate}


def resolve_equivalent(value: Any = None) -> EquivalentConfig:
    """Normalize a YAML block, an integer mode or an ``EquivalentConfig``.

    ``None`` selects the full transistor array.  A boolean keeps the historical
    ``use_equivalent`` spelling of the core classes (``True`` is the cross mode).
    """
    if isinstance(value, EquivalentConfig):
        return value
    if value is None:
        return EquivalentConfig()
    if isinstance(value, bool):
        return EquivalentConfig(1 if value else 0)
    if isinstance(value, int):
        return EquivalentConfig(value)
    if isinstance(value, Mapping):
        settings = dict(value)
    elif hasattr(value, "__dict__"):
        settings = vars(value).copy()
    else:
        raise ValueError("Equivalent settings must be a mapping, an integer or None")
    unknown = settings.keys() - {"mode"}
    if unknown:
        raise ValueError(f"Unsupported equivalent options: {sorted(unknown)}")
    return EquivalentConfig(settings.get("mode", 0))
