"""Frozen clock periods derived from measured worst-case macro phases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, log2


@dataclass(frozen=True)
class TimingConfig:
    t_period: float
    low_read: float
    low_write: float
    high: float
    margin: float = 0.25
    source: str = "calibrated"

    def __post_init__(self):
        if not isfinite(self.t_period) or self.t_period <= 0:
            raise ValueError('Clock period must be finite and positive')
        if any(not isfinite(value) or value < 0 for value in
               (self.low_read, self.low_write, self.high, self.margin)):
            raise ValueError('Timing phases and margin must be finite and nonnegative')
        minimum = 2 * max(self.low_read, self.low_write, self.high) * (1 + self.margin)
        if self.t_period + 1e-18 < minimum:
            raise ValueError('Clock period does not cover its declared phase margin')

    def to_dict(self):
        return asdict(self)

    def apply(self, testbench):
        from PySpice.Unit import (
            u_ns,  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export
        )
        period = (self.t_period / 1e-9) @ u_ns
        testbench.set_timing_parameters(
            t_rise=0.01 * period, t_fall=0.01 * period, t_pulse=0.5 * period,
            t_period=period, t_delay=1 @ u_ns,
        )


def timing_from_measurements(read, writes, *, margin=0.25):
    """Use SS read and both SS/SF writes; never recalibrate from MC candidates."""
    def positive(metrics, key):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Missing/failed timing calibration measurement {key}') from exc
        if not isfinite(value) or value < 0:
            raise ValueError(f"Invalid timing calibration measurement {key}: {value}")
        return value

    if not isfinite(margin) or margin < 0:
        raise ValueError("Timing margin must be finite and nonnegative")
    if len(writes) != 2:
        raise ValueError("Timing calibration requires SS and SF write phases")
    low_read = positive(read, 'TCLK_WLEN') + positive(read, 'TREAD_TOTAL')
    low_write = max(positive(m, 'TCLK_WLEN') + positive(m, 'TWRITE_TOTAL') for m in writes)
    high = max(max(positive(m, 'TRESTORE'), positive(m, 'TCLK_DEC') if 'TCLK_DEC' in m else 0)
               for m in [read, *writes])
    if not isfinite(high) or max(low_read, low_write, high) <= 0:
        raise ValueError("Invalid calibration phases")
    period = ceil(2 * max(low_read, low_write, high) * (1 + margin) / 50e-12) * 50e-12
    return TimingConfig(period, low_read, low_write, high, margin)


def provisional_period(rows, cols, cell_type):
    """Long calibration clock only: historical phase model plus a factor of two.

    The old fitted coefficients do not qualify the new periphery. Every shipped
    timing record must come from the measured phases above, then pass MC checks.
    """
    read_low_ps = (392 + 1.20 * rows + 9.33 * log2(cols) if cell_type == 'SRAM_10T_CELL'
                   else 390 + 0.946 * rows + 9.06 * log2(cols)) * 2.31
    write_high_ps = (188 + 0.41 * cols + 10.7 * log2(max(rows, 2))) * 2.24
    return max(5e-9, ceil(5 * max(read_low_ps, write_high_ps) / 50) * 50e-12)
