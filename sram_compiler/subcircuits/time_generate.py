"""V2.2.0 TIME_CONTROL: rising-edge capture, clock-high access, clock-low recovery.

Address, data, select and write request are captured once per cycle. Access
starts after the physical precharge has released and a shared setup delay;
the falling clock edge ends it. The write drivers cover the wordline's physical
falling tail. Recovery and precharge finish before the next capture edge, so
no secondary address, request or write-data hold latches are needed.

The replica bitline still times read sensing; it is not a write-completion
detector. The far replica wordline and PRE observers enforce non-overlap.
A clock period must cover BOTH access and recovery, including local mismatch.
See docs/design/PHASED_CONTROL_V2_2_0.md for the timing contract and evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log2
from typing import Optional, Sequence

from PySpice.Unit import u_Ohm, u_pF

from sram_compiler.interconnect import add_tapped_line, resolve_interconnect
from .base_subcircuit import BaseSubcircuit
from .standard_cell import AND2, AND3, PNAND2, PNAND3, PNOR2, Pinv

# FreePDK45 control logic: the unit inverter is 0.09 / 0.27 um at 50 nm.  Every
# enable is sized in multiples of it; the NAND2 of the standard cells is
# 0.18 / 0.27 um.
DEFAULT_NMOS_MODEL = "NMOS_VTG"
DEFAULT_PMOS_MODEL = "PMOS_VTG"
GATE_LENGTH = 0.05e-6
UNIT_NMOS_WIDTH = 0.09e-6
UNIT_PMOS_WIDTH = 0.27e-6
NAND_NMOS_WIDTH = 0.18e-6
NAND_PMOS_WIDTH = 0.27e-6
# Output inverters of the AND gates: 6 units for the gated clocks, 4 units for
# the write and sense enables (they drive up to 32 unit loads directly).
GATED_CLOCK_INVERTER = (0.54e-6, 1.62e-6)   # (nmos, pmos)
ENABLE_INVERTER = (0.36e-6, 1.08e-6)
# Half-unit observers of far wires (replica wordline, far PRE) keep the added load small.
OBSERVER_WIDTHS = (0.045e-6, 0.135e-6)
# Register and transmission-gate devices of the flip-flops.
FLOP_NMOS_WIDTH = 2.5e-07
FLOP_PMOS_WIDTH = 5e-07


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class TaperedBuffer(BaseSubcircuit):
    """Non-inverting inverter chain for a large fan-out.

    The output stage is `drive_scale` unit inverters (0.09 / 0.27 um) and the
    stages are tapered geometrically from the input, so every stage sees a
    fan-out of drive_scale ** (1 / n_stages): two stages up to a scale of 16
    (fan-out <= 4 per stage), four stages above.  `name` must be unique per
    scope because PySpice keeps one subcircuit definition per name.

    With `effort_based` the chain length comes from the total path effort
    (final stage up to eight times its input gate width) and wide gates are
    folded into 2 um fingers.
    """
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, name: str, drive_scale: float = 1.0,
                 nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 length: float = GATE_LENGTH,
                 w_rc: bool = False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 effort_based: bool = False, load_units: Optional[float] = None,
                 fall_strength: float = 1.0) -> None:
        self.NAME = name
        super().__init__(
            nmos_model, pmos_model,
            UNIT_NMOS_WIDTH, UNIT_PMOS_WIDTH, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        drive_scale = max(1.0, float(drive_scale))
        n_stages = 2 if drive_scale <= 16 else 4
        if effort_based:
            # Final stage drives up to eight times its input gate width.
            # Pick an even chain length from that total path effort.
            effort = 8.0 * drive_scale if load_units is None else max(1.0, float(load_units))
            n_stages = max(2, 2 * ceil(log2(effort) / 6.0))
        self.drive_scale = drive_scale
        self.n_stages = n_stages
        prev = 'A'
        for k in range(n_stages):
            s = drive_scale ** ((k + 1) / n_stages)
            n_strength = fall_strength if k == n_stages - 1 else 1.0
            inv = Pinv(nmos_model, pmos_model, 0.09e-6 * s * n_strength, 0.27e-6 * s, length,
                       num=f'_{name}_{k}', max_finger_width=2e-6 if effort_based else None)
            self.subcircuit(inv)
            out = 'Z' if k == n_stages - 1 else f'b{k}'
            self.X(f'inv{k}', inv.NAME, 'VDD', 'VSS', prev, out)
            prev = out


class TransmissionGate(BaseSubcircuit):
    """NMOS/PMOS pass gate: conducts while CTR_N is high and CTR_P is low."""
    NAME = "TRANSMISSION_GATE"
    NODES = ('VDD', 'VSS', 'IN', 'OUT', 'CTR_P', 'CTR_N')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 pmos_width: float = FLOP_PMOS_WIDTH, nmos_width: float = FLOP_NMOS_WIDTH,
                 length: float = GATE_LENGTH,
                 w_rc: bool = False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF) -> None:
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        self.M('transpmos', 'OUT', 'CTR_P', 'IN', 'VDD',
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M('transnmos', 'IN', 'CTR_N', 'OUT', 'VSS',
               model=self.nmos_model, w=self.nmos_width, l=self.length)


class ClockBuffer(BaseSubcircuit):
    """Four-stage tapered clock buffer (1 : 3 : 9 : 27 unit widths times `drive_scale`)."""
    NAME = "CLOCK_BUFFER"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 drive_scale: float = 1.0, fold_gates: bool = False) -> None:
        super().__init__(nmos_model, pmos_model, NAND_NMOS_WIDTH, NAND_PMOS_WIDTH, GATE_LENGTH,
                         w_rc=False)
        drive_scale = max(1.0, float(drive_scale))
        s1 = drive_scale ** 0.25
        s2 = drive_scale ** 0.50
        s3 = drive_scale ** 0.75
        s4 = drive_scale
        fingers = 2e-6 if fold_gates else None
        stages = [
            Pinv(nmos_model, pmos_model, 0.09e-6 * s1, 0.27e-6 * s1, 0.05e-6, num=1, max_finger_width=fingers),
            Pinv(nmos_model, pmos_model, 0.27e-6 * s2, 0.81e-6 * s2, 0.05e-6, num=2, max_finger_width=fingers),
            Pinv(nmos_model, pmos_model, 0.91e-6 * s3, 2.43e-6 * s3, 0.05e-6, num=3, max_finger_width=fingers),
            Pinv(nmos_model, pmos_model, 2.43e-6 * s4, 7.29e-6 * s4, 0.05e-6, num=4, max_finger_width=fingers),
        ]
        for inv in stages:
            self.subcircuit(inv)
        self.X('buf_inv1', stages[0].NAME, 'VDD', 'VSS', 'A', 'zb1_node')
        self.X('buf_inv2', stages[1].NAME, 'VDD', 'VSS', 'zb1_node', 'zb2_node')
        self.X('buf_inv3', stages[2].NAME, 'VDD', 'VSS', 'zb2_node', 'zb3_node')
        self.X('buf_inv4', stages[3].NAME, 'VDD', 'VSS', 'zb3_node', 'Z')


class WordlineRequestNand(BaseSubcircuit):
    """NAND2 of the wordline request and read-not-done, folded like the buffer inverters."""
    NAME = "WORDLINE_REQUEST_NAND"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 nmos_width: float = NAND_NMOS_WIDTH, pmos_width: float = NAND_PMOS_WIDTH,
                 length: float = GATE_LENGTH, max_finger_width: Optional[float] = None) -> None:
        super().__init__(nmos_model, pmos_model, nmos_width, pmos_width, length, w_rc=False)
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length

        def fingers(width):
            if max_finger_width is None:
                return {}
            count = max(1, ceil(round(float(width) / max_finger_width, 12)))
            return {'raw_spice': f'NF={count}'} if count > 1 else {}
        self.M('pmos1', 'Z', 'A', 'VDD', 'VDD', model=pmos_model, w=pmos_width, l=length, **fingers(pmos_width))
        self.M('pmos2', 'Z', 'B', 'VDD', 'VDD', model=pmos_model, w=pmos_width, l=length, **fingers(pmos_width))
        self.M('nmos1', 'Z', 'B', 'net1', 'VSS', model=nmos_model, w=nmos_width, l=length, **fingers(nmos_width))
        self.M('nmos2', 'net1', 'A', 'VSS', 'VSS', model=nmos_model, w=nmos_width, l=length, **fingers(nmos_width))


class WordlineEnableBuffer(BaseSubcircuit):
    """Two-stage wl_en buffer; both stages scale with the wordline-driver load.

    The first stage is a NAND2 of the prepared access request and read_done_bar.
    A read releases WL before sense regeneration; a write releases it at clock
    fall. The 1.35 / 0.45 um output stage scales with the row-driver load
    (V2.0.2: the fixed stage took 280 / 600 ps to switch at 512 rows).
    """
    NAME = "WORDLINE_ENABLE_BUFFER"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 drive_scale: float = 1.0, fold_gates: bool = False) -> None:
        super().__init__(nmos_model, pmos_model, NAND_NMOS_WIDTH, NAND_PMOS_WIDTH, GATE_LENGTH,
                         w_rc=False)
        drive_scale = max(1.0, float(drive_scale))
        fingers = 2e-6 if fold_gates else None
        first = WordlineRequestNand(nmos_model, pmos_model, NAND_NMOS_WIDTH * drive_scale,
                                    NAND_PMOS_WIDTH * drive_scale, max_finger_width=fingers)
        second = Pinv(nmos_model, pmos_model, 0.45e-06 * drive_scale, 1.35e-06 * drive_scale, 0.05e-6,
                      num=2, max_finger_width=fingers)
        self.subcircuit(first)
        self.subcircuit(second)
        self.X('buf_nand1', first.NAME, 'VDD', 'VSS', 'A', 'B', 'zb1_node')
        self.X('buf_inv2', second.NAME, 'VDD', 'VSS', 'zb1_node', 'Z')


class Dff(BaseSubcircuit):
    """Positive-edge master/slave flip-flop from transmission gates (Q follows D).

    The master (tg1, inv3, inv4, tg2) is transparent while CLK is low, the
    slave (tg3, inv6, inv7, tg4) while CLK is high.  The testbench initialises
    the internal nodes ``D_b``, ``z1`` .. ``z5`` and ``QB`` by name.
    """
    NAME = "DFF"
    NODES = ('VDD', 'VSS', 'D', 'Q', 'CLK')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 length: float = GATE_LENGTH) -> None:
        super().__init__(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, length, w_rc=False)
        inv = Pinv(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, 0.05e-6, num=1)
        gate = TransmissionGate(nmos_model=nmos_model, pmos_model=pmos_model)
        self.subcircuit(inv)
        self.subcircuit(gate)
        self.X('inv1_clk', inv.NAME, 'VDD', 'VSS', 'CLK', 'CLKB')
        # master: D_b -> z1 -> z2, held by inv4 / tg2 while CLK is high
        self.X('inv2_D', inv.NAME, 'VDD', 'VSS', 'D', 'D_b')
        self.X('tg1', gate.NAME, 'VDD', 'VSS', 'D_b', 'z1', 'CLK', 'CLKB')
        self.X('inv3', inv.NAME, 'VDD', 'VSS', 'z1', 'z2')
        self.X('inv4', inv.NAME, 'VDD', 'VSS', 'z2', 'z3')
        self.X('tg2', gate.NAME, 'VDD', 'VSS', 'z3', 'z1', 'CLKB', 'CLK')
        # slave: z4 -> z5 -> Q, held by inv7 / tg4 while CLK is low
        self.X('inv5', inv.NAME, 'VDD', 'VSS', 'z2', 'z4')
        self.X('tg3', gate.NAME, 'VDD', 'VSS', 'z4', 'z5', 'CLKB', 'CLK')
        self.X('inv6', inv.NAME, 'VDD', 'VSS', 'z5', 'Q')
        self.X('inv7', inv.NAME, 'VDD', 'VSS', 'Q', 'QB')
        self.X('tg4', gate.NAME, 'VDD', 'VSS', 'QB', 'z5', 'CLK', 'CLKB')


class DffBuffer(BaseSubcircuit):
    """Flip-flop with buffered complementary outputs: Q follows D, QB is its complement.

    TIME_CONTROL registers the active-low chip select and write request with it, so
    the QB output carries the active-high `cs` / `we` and Q the complement.
    """
    NAME = "DFF_BUFFER"
    NODES = ('VDD', 'VSS', 'D', 'Q', 'QB', 'CLK')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 length: float = GATE_LENGTH) -> None:
        super().__init__(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, length, w_rc=False)
        flop = Dff(nmos_model, pmos_model, length=length)
        inv_qb = Pinv(nmos_model, pmos_model, 0.18e-6, 0.54e-6, length=0.05e-6, num=1)   # 2 units
        inv_q = Pinv(nmos_model, pmos_model, 0.36e-6, 1.08e-6, length=0.05e-6, num=2)    # 4 units
        self.subcircuit(flop)
        self.subcircuit(inv_qb)
        self.subcircuit(inv_q)
        self.X('dff', flop.NAME, 'VDD', 'VSS', 'D', 'qint', 'CLK')
        self.X('inv1', inv_qb.NAME, 'VDD', 'VSS', 'qint', 'QB')
        self.X('inv2', inv_q.NAME, 'VDD', 'VSS', 'QB', 'Q')


class ReplicaDelayChain(BaseSubcircuit):
    """Odd unit-inverter chain, four unit loads per stage: rbl -> rbl_delay.

    Inverting, so `rbl_delay` rises once the replica bitline has discharged;
    the stage count `N` (with the replica cell count `K`) sets the sensing
    margin (docs/DRIVER_SIZING_PROPOSAL.md).  Legacy instance names are kept.
    """
    NAME = "REPLICA_DELAY_CHAIN"
    NODES = ('VDD', 'VSS', 'in', 'out')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 stages: int = 9) -> None:
        if isinstance(stages, bool) or not isinstance(stages, int) or stages < 1 or stages % 2 == 0:
            raise ValueError("Replica delay stages must be a positive odd integer")
        self.stages = stages
        super().__init__(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, GATE_LENGTH, w_rc=False)
        inv = Pinv(nmos_model, pmos_model, 0.9e-07, 2.7e-07, length=0.05e-6, num=1)
        self.subcircuit(inv)
        for i in range(self.stages):
            source = 'in' if i == 0 else f'dout_{i}'
            target = 'out' if i == self.stages - 1 else f'dout_{i+1}'
            self.X(f'dinv{i}', inv.NAME, 'VDD', 'VSS', source, target)
            for j in range(4):
                self.X(f'dload_{i}_{j}', inv.NAME, 'VDD', 'VSS', target, f'n_{i}_{j}')


class UnitDelayChain(BaseSubcircuit):
    """Even (non-inverting) unit-inverter chain with `loads_per_stage` dummy loads per stage."""
    NAME = "UNIT_DELAY_CHAIN"
    NODES = ('VDD', 'VSS', 'in', 'out')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 length: float = GATE_LENGTH, stages: int = 4, loads_per_stage: int = 4,
                 w_rc: bool = False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF) -> None:
        stages = max(2, int(stages))
        if stages % 2:
            stages += 1
        self.stages = stages
        self.loads_per_stage = loads_per_stage
        super().__init__(
            nmos_model, pmos_model,
            UNIT_NMOS_WIDTH, UNIT_PMOS_WIDTH, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        inv = Pinv(nmos_model, pmos_model, nmos_width=UNIT_NMOS_WIDTH, pmos_width=UNIT_PMOS_WIDTH,
                   length=length, num='_unit_delay')
        self.subcircuit(inv)
        prev = 'in'
        for i in range(self.stages):
            out = 'out' if i == self.stages - 1 else f'wen_dly_{i + 1}'
            self.X(f'winv{i}', inv.NAME, 'VDD', 'VSS', prev, out)
            for j in range(self.loads_per_stage):
                self.X(f'wload_{i}_{j}', inv.NAME, 'VDD', 'VSS', out, f'wn_{i}_{j}')
            prev = out


class AddressRegister(BaseSubcircuit):
    """One flip-flop per address bit, clustered in TIME_CONTROL before the decoder wires."""
    NAME = "ADDRESS_REGISTER"

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 num_rows: int = 16) -> None:
        self.num_rows = num_rows
        n_bits = address_bits(num_rows)
        self.NODES = (['VDD', 'VSS', 'CLK'] + [f'A{i}' for i in range(n_bits)]
                      + [f'A_dff{i}' for i in range(n_bits)])
        super().__init__(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, GATE_LENGTH, w_rc=False)
        flop = Dff(nmos_model, pmos_model)
        self.subcircuit(flop)
        for i in range(n_bits):
            self.X(f'dff_{i}', flop.NAME, 'VDD', 'VSS', f'A{i}', f'A_dff{i}', 'CLK')


class DataRegister(BaseSubcircuit):
    """One flip-flop per write-data bit; the clock crosses the array width on
    the wordline wire geometry, one tap per column (V2.1.1 distributed wiring)."""
    NAME = "DATA_REGISTER"

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 num_cols: int = 8, interconnect=None) -> None:
        self.num_cols = num_cols
        self.interconnect = resolve_interconnect(interconnect)
        self.NODES = (['VDD', 'VSS', 'CLK'] + [f'DIN{i}' for i in range(num_cols)]
                      + [f'DIN_dff{i}' for i in range(num_cols)])
        super().__init__(nmos_model, pmos_model, FLOP_NMOS_WIDTH, FLOP_PMOS_WIDTH, GATE_LENGTH, w_rc=False)
        flop = Dff(nmos_model, pmos_model)
        self.subcircuit(flop)
        clocks = add_tapped_line(self, 'CLK_line', 'CLK', num_cols, self.interconnect.wl)
        for i in range(num_cols):
            self.X(f'dff_{i}', flop.NAME, 'VDD', 'VSS', f'DIN{i}', f'DIN_dff{i}', clocks[i])


# Gate identities: PySpice keeps one subcircuit definition per name and scope,
# so every gate with its own sizes or role carries its own name.
class WordlineBusyNand(PNAND2):
    NAME = "WORDLINE_BUSY_NAND"


class WriteWindowNor(PNOR2):
    NAME = "WRITE_WINDOW_NOR"


class EnablesOffNor(PNOR2):
    NAME = "ENABLES_OFF_NOR"


class WordlineSetupDelay(UnitDelayChain):
    """Give capture, decode and write-driver assertion a head start."""
    NAME = "WORDLINE_SETUP_DELAY"


class WordlineSetupAnd(AND2):
    NAME = "WORDLINE_SETUP_AND"


class WordlineStartAnd(AND2):
    NAME = "WORDLINE_START_AND"


class WriteReadyNor(PNOR2):
    NAME = "WRITE_READY_NOR"


class PrechargeGuardDelay(UnitDelayChain):
    """Settling delay of the replica-wordline observer before the precharge."""
    NAME = "PRECHARGE_GUARD_DELAY"


class PrechargeGuardAnd(AND2):
    NAME = "PRECHARGE_GUARD_AND"


class PrechargeOffDelay(UnitDelayChain):
    NAME = "PRECHARGE_OFF_DELAY"


class PrechargeAccessAnd(AND2):
    NAME = "PRECHARGE_ACCESS_AND"


class PrechargeOffGuard(BaseSubcircuit):
    """Qualify access assertion after the physical PRE-off edge has settled.

    PRE is active low. Its small inverter observer switches near mid-rail;
    a baseline wire/load RC delay and fixed even chain let the tail settle.
    The raw request directly inhibits the final gate, so this settling delay
    cannot extend an access after the clock-high request ends.  The settled
    ``pre_off_ready`` is exported for waveform verification.
    """
    NAME = "PRECHARGE_OFF_GUARD"
    NODES = ('VDD', 'VSS', 'request', 'pre_far', 'access', 'pre_off_ready')

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 stages: int = 4, access_load: float = 3.5, settling_tau: float = 0.0) -> None:
        if type(stages) is not int or stages < 2 or stages % 2:
            raise ValueError('Precharge-off guard stages must be a positive even integer')
        if not isfinite(access_load) or access_load <= 0:
            raise ValueError('Access-control load must be finite and positive')
        if not isfinite(settling_tau) or settling_tau < 0:
            raise ValueError('Precharge-off settling time must be finite and nonnegative')
        super().__init__(nmos_model, pmos_model, .09e-6, .27e-6, .05e-6, w_rc=False)
        self.stages = stages
        self.access_load = access_load
        self.settling_tau = settling_tau
        observer = Pinv(nmos_model, pmos_model, *OBSERVER_WIDTHS,
                        GATE_LENGTH, num='_pre_off_observer')
        delay = PrechargeOffDelay(nmos_model, pmos_model, stages=stages)
        ready = PNOR2(nmos_model, pmos_model, .09e-6, .54e-6, .05e-6)
        drive_scale = max(6, ceil(access_load / 6.0))
        gate = PrechargeAccessAnd(nmos_model, pmos_model, nmos_model, pmos_model,
                                 inv_nmos_width=.09e-6 * drive_scale,
                                 inv_pmos_width=.27e-6 * drive_scale)
        for sub in (observer, delay, ready, gate):
            self.subcircuit(sub)
        self.X('observe_pre', observer.NAME, 'VDD', 'VSS', 'pre_far', 'pre_on')
        settle_input = 'pre_on'
        if settling_tau:
            # Fast devices cannot make a long metal line settle faster. Add
            # the baseline wire/load time constant only to the delayed branch.
            # A small fixed capacitor avoids loading the half-unit observer;
            # the first inverter's input capacitance adds conservative delay.
            settle_input = 'pre_on_filtered'
            self.R('settle', 'pre_on', settle_input, max(1.0, settling_tau / 1e-15))
            self.C('settle', settle_input, 'VSS', 1e-15)
        self.X('settle_pre', delay.NAME, 'VDD', 'VSS', settle_input, 'pre_on_delayed')
        self.X('ready', ready.NAME, 'VDD', 'VSS', 'pre_on', 'pre_on_delayed', 'pre_off_ready')
        self.X('request_gate', gate.NAME, 'VDD', 'VSS', 'request', 'pre_off_ready', 'access')


class IsolationOffGuard(PrechargeOffGuard):
    """Same settled-high observer, watching far SA input-isolation gates."""
    NAME = "ISOLATION_OFF_GUARD"


class EnableOffGuard(PrechargeOffGuard):
    """Admit recovery only after both far physical enables have released."""
    NAME = "ENABLE_OFF_GUARD"


class FarEnablesOffNor(PNOR2):
    NAME = "FAR_ENABLES_OFF_NOR"


class ReadDoneAnd(AND3):
    NAME = "READ_DONE_AND"


class WritePrepareAnd(AND2):
    NAME = "WRITE_PREPARE_AND"


class IsolationRequestNor(PNOR2):
    NAME = "ISOLATION_REQUEST_NOR"


class IsolationNand(PNAND2):
    NAME = "ISOLATION_NAND"


class GuardedWriteAnd(AND3):
    NAME = "GUARDED_WRITE_AND"


def address_bits(num_rows: int) -> int:
    return ceil(log2(num_rows)) if num_rows > 1 else 1


# ---------------------------------------------------------------------------
# Sizing policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlSizing:
    """Buffer scales of the TIME_CONTROL enables for their loads (unit inverters).

    The loads are supplied by ``resolve_driver_sizes`` (V2.0.4+); the
    fallbacks reproduce the V2.0.2 estimates for direct ``TIME_CONTROL`` callers.
    Every enable drives up to `effort` unit loads per unit of drive directly;
    above ``4 * effort`` a :class:`TaperedBuffer` follows the gate.
    """
    wl_load: float
    pre_load: float
    wen_load: float
    sen_load: float
    iso_load: float
    num_sa: int
    access_load: float
    addr_fanout_units: int
    addr_scale: int
    clk_drive_scale: float
    wl_en_scale: int
    wlb_scale: int
    senb_scale: int
    wen_effort: float
    sen_effort: float
    iso_effort: float
    pre_scale: int
    iso_fall_strength: float

    @property
    def wen_buffered(self) -> bool:
        return not self.wen_load <= 4 * self.wen_effort

    @property
    def sen_buffered(self) -> bool:
        return not self.sen_load <= 4 * self.sen_effort

    def iso_buffered(self, effort_buffers: bool) -> bool:
        return not self.iso_load <= (16 if effort_buffers else 32)

    @property
    def access_drive_scale(self) -> int:
        """The prepared request drives the scaled wordline-request NAND."""
        return max(6, ceil(self.access_load / 6.0))


def resolve_control_sizing(num_rows: int, num_cols: int, operation: str, effort_buffers: bool,
                           precharge_off_guard: bool, num_sa: Optional[int], wl_load: Optional[float],
                           pre_load: Optional[float], wen_load: Optional[float], sen_load: Optional[float],
                           iso_load: Optional[float], sen_effort: Optional[float],
                           access_load: Optional[float]) -> ControlSizing:
    """Fill the load fallbacks and derive every buffer scale (see the docstrings of TIME_CONTROL)."""
    n_bits = address_bits(num_rows)
    writes = operation == 'write' or operation == 'read&write'
    num_sa = num_cols if num_sa is None else int(num_sa)
    wl_load = float(num_rows) if wl_load is None else float(wl_load)
    if pre_load is None:
        pre_load = (num_cols + 1) * 3 * 0.27e-6 * max(0.5, num_rows / 16.0) / 0.36e-6
        if precharge_off_guard:
            pre_load += .5  # The physical far-PRE inverter observer.
    pre_load = float(pre_load)
    if wen_load is None:
        # Write drivers plus the enables-off NOR input (1.75 units, V2.1.8).
        wen_load = num_cols * (3 * 0.18e-6 + 0.36e-6) * max(8, num_rows) / 16.0 / 0.36e-6 + 1.75
    wen_load = float(wen_load)
    # Address buffers: 5 decoder gate inputs per 8 rows, 3-unit output per 8
    # loads; with effort buffers four NAND3 inputs (1.25 units) plus the input
    # inverter, and TaperedBuffer's last stage is one unit times drive_scale.
    addr_fanout_units = 5 * ceil(num_rows / 8.0)
    addr_scale = max(1, ceil(addr_fanout_units / 8.0 / 3.0))
    if effort_buffers:
        addr_fanout_units = 6 * ceil(num_rows / 8.0)
        addr_scale = max(1, ceil(addr_fanout_units / 8.0))
    # Clock buffer: scaled with the flip-flop count relative to a 16x16 array.
    ref_bits = ceil(log2(16))
    clk_dff_count = n_bits + 2   # address + chip-select + write-request registers
    ref_dff_count = ref_bits + 2
    if writes:
        clk_dff_count += num_cols
        ref_dff_count += 16
    clk_drive_scale = max(1.0, clk_dff_count / ref_dff_count)
    # wl_en: one unit per 32 wordline-driver NAND2 inputs keeps the fan-out <= 8.
    wl_en_scale = max(1, ceil(wl_load / (24.0 if effort_buffers else 32.0)))
    # Access request load: the wordline-request NAND2 input (1.25 units per
    # unit of scale) plus the sense request input (V2.1.8: the first wl_en
    # stage was an inverter of `wl_en_scale` units).
    access_load = 1.25 * wl_en_scale + 2.5 if access_load is None else float(access_load)
    # wl_en_bar: two NAND2 inputs per hold latch (address bits + write request)
    # plus the precharge NAND3, at a fan-out of ~5.  With the replica guard the
    # write-request latch and the precharge gate read the wordline observer and
    # wl_en_bar feeds the wordline-busy NAND2 instead (V2.1.9): the count is kept
    # as an upper bound, so the address path is unchanged.
    wlb_scale = max(1, ceil((2 * (n_bits + 1) + 1) / 5.0))
    # s_en_bar: the wordline-request NAND2 input and the write-slot NAND3 input
    # (1.25 units each), at a fan-out of ~4 (V2.1.8).
    senb_scale = max(1, ceil(1.25 * (wl_en_scale + 1) / 4.0))
    wen_effort = 6.0 if effort_buffers else 8.0
    # Sense enable load: footers + the output latch, plus its inverter and the
    # enables-off NOR input (1.75 units) since V2.1.8.
    sen_load = 0.75 * num_sa + 3.5 + senb_scale + 1.75 if sen_load is None else float(sen_load)
    sen_effort = (4.0 if effort_buffers else 8.0) if sen_effort is None else float(sen_effort)
    iso_load = 4.0 * num_sa if iso_load is None else float(iso_load)
    iso_effort = 3.0 if effort_buffers else 8.0
    pre_scale = max(1, ceil(pre_load / 8.0))
    if effort_buffers:
        # Restore transitions include precharge-gate Miller loading.
        pre_scale = max(2, ceil(pre_load / 4.0))
    return ControlSizing(
        wl_load=wl_load, pre_load=pre_load, wen_load=wen_load, sen_load=sen_load, iso_load=iso_load,
        num_sa=num_sa, access_load=access_load, addr_fanout_units=addr_fanout_units,
        addr_scale=addr_scale, clk_drive_scale=clk_drive_scale, wl_en_scale=wl_en_scale,
        wlb_scale=wlb_scale, senb_scale=senb_scale, wen_effort=wen_effort, sen_effort=sen_effort,
        iso_effort=iso_effort,
        pre_scale=pre_scale, iso_fall_strength=2.0 if effort_buffers else 1.0)


# ---------------------------------------------------------------------------
# The control block
# ---------------------------------------------------------------------------
class TIME_CONTROL(BaseSubcircuit):
    """Control-signal generator (see the module docstring for the signal list).

    Ports: VDD, VSS, clk, csb, web, clk_buf, clk_bar, cs_bar, cs, we_bar, we,
    gated_clk_bar, gated_clk_buf, wl_en, A{i}, A_dff{i}, [DIN{i}, DIN_dff{i}],
    rbl, rbl_delay, rbl_delay_bar, s_en, w_en, PRE, sa_iso, [rwl], [pre_far],
    iso_far, sen_far, wen_far.
    The data ports exist for write operations; rwl and pre_far are the physical
    guard inputs. din_hold_load is accepted only for Python-call compatibility.
    """
    NAME = "TIME_CONTROL"

    def __init__(self, nmos_model: str = DEFAULT_NMOS_MODEL, pmos_model: str = DEFAULT_PMOS_MODEL,
                 # Base widths for NAND gate transistors
                 pmos_width: float = NAND_PMOS_WIDTH, nmos_width: float = NAND_NMOS_WIDTH,
                 length: float = GATE_LENGTH, num_rows: int = 16, num_cols: int = 8,
                 w_rc: bool = False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF, operation: str = 'read',
                 num_sa: Optional[int] = None, wl_load: Optional[float] = None,
                 pre_load: Optional[float] = None, wen_load: Optional[float] = None,
                 dc_stages: int = 9, effort_buffers: bool = False,
                 sen_load: Optional[float] = None, iso_load: Optional[float] = None,
                 replica_precharge_guard: bool = False,
                 sen_effort: Optional[float] = None,
                 precharge_guard_stages: int = 0,
                 interconnect=None,
                 precharge_off_guard: bool = False, precharge_off_guard_stages: int = 4,
                 access_load: Optional[float] = None,
                 precharge_off_tau: float = 0.0,
                 din_hold_load: Optional[float] = None,
                 enable_off_tau: Optional[float] = None,
                 isolation_tau: Optional[float] = None,
                 ) -> None:
        """
        num_sa:   number of sense amplifiers driven by s_en (num_cols / mux_in);
                  default num_cols.
        wl_load:  load on wl_en in units of a 0.18/0.27 um NAND2 gate, i.e.
                  num_rows * (wordline-driver NAND2 scale); default num_rows.
        pre_load: load on PRE in unit (0.09/0.27 um) inverter inputs, i.e.
                  (num_cols + 1) * 3 * precharge PMOS width / 0.36 um; default
                  assumes the 0.27 um base width scaled with max(0.5, rows/16).
        wen_load: load on w_en in unit inverter inputs: per column the write
                  driver's EN inverter and two EN-gated NMOS (3 * nmos_w +
                  pmos_w, row-scaled) plus the testbench's latch-enable
                  inverter (on w_en until V2.1.9, on din_hold since V2.1.10;
                  kept as an upper bound) and the enables-off NOR input;
                  default assumes the 0.18/0.36 um base widths scaled with
                  max(8, rows)/16.
        precharge_off_tau: frozen PRE wire/load settling scale in seconds;
                  enabling precharge_off_guard appends the physical pre_far input.
        din_hold_load: obsolete V2.1.10 argument; no hold-enable line is built.
        """
        if (isinstance(precharge_guard_stages, bool) or not isinstance(precharge_guard_stages, int)
                or precharge_guard_stages < 0 or precharge_guard_stages % 2):
            raise ValueError('Precharge guard stages must be a nonnegative even integer')
        if precharge_guard_stages and not replica_precharge_guard:
            raise ValueError('Precharge settling delay requires the replica guard')
        if type(precharge_off_guard) is not bool:
            raise ValueError('Precharge-off guard must be boolean')
        if (type(precharge_off_guard_stages) is not int or precharge_off_guard_stages < 2
                or precharge_off_guard_stages % 2):
            raise ValueError('Precharge-off guard stages must be a positive even integer')
        self.interconnect = resolve_interconnect(interconnect)
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.w_rc = w_rc
        self.operation = operation
        self.writes = operation == 'write' or operation == 'read&write'
        self.effort_buffers = effort_buffers
        self.dc_stages = dc_stages
        self.replica_precharge_guard = replica_precharge_guard
        self.precharge_guard_stages = precharge_guard_stages
        self.precharge_off_guard = precharge_off_guard
        self.precharge_off_guard_stages = precharge_off_guard_stages
        self.precharge_off_tau = precharge_off_tau
        self.enable_off_tau = precharge_off_tau if enable_off_tau is None else enable_off_tau
        self.isolation_tau = precharge_off_tau if isolation_tau is None else isolation_tau
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.n_bits = address_bits(num_rows)
        sizing = resolve_control_sizing(num_rows, num_cols, operation, effort_buffers,
                                        precharge_off_guard, num_sa, wl_load, pre_load, wen_load,
                                        sen_load, iso_load, sen_effort, access_load)
        self.sizing = sizing
        # Loads and scales read back by the testbench and the tests.
        self.num_sa = sizing.num_sa
        self.wl_load = sizing.wl_load
        self.pre_load = sizing.pre_load
        self.wen_load = sizing.wen_load
        self.sen_load = sizing.sen_load
        self.iso_load = sizing.iso_load
        self.access_load = sizing.access_load
        self.NODES = self._ports()
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        # Keep builder order stable: per-device native sampling follows deck
        # order. Forward references connect the physical guards and observers.
        self._add_address_path()
        self._add_data_register()
        self._add_clock_tree()
        self._add_select_and_write_registers()
        self._add_gated_clocks()
        self._add_access_request()
        self._add_wordline_enable()
        self._add_replica_delay()
        self._add_read_completion()
        self._add_wordline_off_guard()
        self._add_wordline_busy()
        self._add_write_window()
        self._add_write_enable()
        self._add_sense_enable()
        self._add_sense_isolation()
        self._add_isolation_guard()
        self._add_precharge()
        self._add_enable_observers()

    # -- ports ---------------------------------------------------------------
    def _ports(self) -> list:
        nodes = ['VDD', 'VSS', 'clk', 'csb', 'web', 'clk_buf', 'clk_bar',
                 'cs_bar', 'cs', 'we_bar', 'we', 'gated_clk_bar', 'gated_clk_buf', 'wl_en']
        nodes.extend([f'A{i}' for i in range(self.n_bits)])
        nodes.extend([f'A_dff{i}' for i in range(self.n_bits)])
        if self.writes:
            nodes.extend([f'DIN{i}' for i in range(self.num_cols)])
            nodes.extend([f'DIN_dff{i}' for i in range(self.num_cols)])
        nodes += ['rbl', 'rbl_delay', 'rbl_delay_bar', 's_en', 'w_en', 'PRE', 'sa_iso']
        if self.replica_precharge_guard:
            nodes.append('rwl')
        if self.precharge_off_guard:
            nodes.append('pre_far')
        nodes.extend(('iso_far', 'sen_far', 'wen_far'))
        return nodes

    # -- gate helpers -----------------------------------------------------------
    def _unit_inverter(self, num: str = '', scale: float = 1.0) -> Pinv:
        return Pinv(self.nmos_model, self.pmos_model,
                    nmos_width=0.09e-6 * scale if scale != 1.0 else 0.09e-6,
                    pmos_width=0.27e-6 * scale if scale != 1.0 else 0.27e-6,
                    length=0.05e-6, num=num)

    def _and_gate(self, cls, inverter: Sequence[float]):
        """AND2/AND3 with the standard NAND stage and an `inverter` (nmos, pmos) output stage."""
        return cls(nmos_model_nand=self.nmos_model, pmos_model_nand=self.pmos_model,
                   nmos_model_inv=self.nmos_model, pmos_model_inv=self.pmos_model,
                   nand_pmos_width=NAND_PMOS_WIDTH, nand_nmos_width=NAND_NMOS_WIDTH,
                   inv_pmos_width=inverter[1], inv_nmos_width=inverter[0],
                   length=0.05e-6, w_rc=self.w_rc,
                   pi_res=self.pi_res, pi_cap=self.pi_cap)

    def _buffered_output(self, instance: str, gate: BaseSubcircuit, inputs: Sequence[str],
                         output: str, buffered: bool, buffer_name: str, buffer_instance: str,
                         drive_scale: int, load_units: float, fall_strength: float = 1.0) -> str:
        """Instantiate `gate` driving `output`, through a TaperedBuffer when the
        load exceeds the gate's direct fan-out.  Returns the gate's own output node."""
        source = f'{output}_unbuf' if buffered else output
        self.X(instance, gate.NAME, 'VDD', 'VSS', *inputs, source)
        if buffered:
            buffer = TaperedBuffer(buffer_name, effort_based=self.effort_buffers, drive_scale=drive_scale,
                                   nmos_model=self.nmos_model, pmos_model=self.pmos_model,
                                   load_units=load_units, fall_strength=fall_strength)
            self.subcircuit(buffer)
            self.X(buffer_instance, buffer.NAME, 'VDD', 'VSS', source, output)
        return source

    # -- builders -----------------------------------------------------------------
    def _add_address_path(self) -> None:
        """Registered address directly buffered to the decoder.

        The old wordline is off during clock-low, before this register updates.
        The access setup delay leaves the decoder time to settle at capture.
        """
        register = AddressRegister(self.nmos_model, self.pmos_model, num_rows=self.num_rows)
        self.subcircuit(register)
        self.X('dff_buf_addr', register.NAME, 'VDD', 'VSS', 'clk_buf',
               *[f'A{i}' for i in range(self.n_bits)], *[f'A_reg{i}' for i in range(self.n_bits)])
        buffer = TaperedBuffer('ADDRESS_BUFFER', drive_scale=self.sizing.addr_scale,
                               nmos_model=self.nmos_model, pmos_model=self.pmos_model,
                               effort_based=self.effort_buffers, load_units=self.sizing.addr_fanout_units)
        self.subcircuit(buffer)
        for i in range(self.n_bits):
            self.X(f'addr_buf_{i}', buffer.NAME,
                   'VDD', 'VSS', f'A_reg{i}', f'A_dff{i}')

    def _add_data_register(self) -> None:
        """DIN{i} -> DIN_dff{i} on clk_buf (write operations only)."""
        if not self.writes:
            return
        register = DataRegister(self.nmos_model, self.pmos_model, num_cols=self.num_cols,
                                interconnect=self.interconnect)
        self.subcircuit(register)
        self.X('dff_buf_data', register.NAME, 'VDD', 'VSS', 'clk_buf',
               *[f'DIN{i}' for i in range(self.num_cols)], *[f'DIN_dff{i}' for i in range(self.num_cols)])

    def _add_clock_tree(self) -> None:
        """clk -> clk_buf (sized for the register count) -> clk_bar."""
        clock_buffer = ClockBuffer(self.nmos_model, self.pmos_model,
                                   drive_scale=self.sizing.clk_drive_scale, fold_gates=self.effort_buffers)
        self.subcircuit(clock_buffer)
        self.X('clkbuf', clock_buffer.NAME, 'VDD', 'VSS', 'clk', 'clk_buf')
        inverter = self._unit_inverter()
        self.subcircuit(inverter)
        self.X('inv_clk_bar', inverter.NAME, 'VDD', 'VSS', 'clk_buf', 'clk_bar')

    def _add_select_and_write_registers(self) -> None:
        """csb -> cs / cs_bar and web -> we / we_bar on the rising clk_buf edge."""
        select = DffBuffer(self.nmos_model, self.pmos_model)
        self.subcircuit(select)
        self.X('dff_buf', select.NAME, 'VDD', 'VSS', 'csb', 'cs_bar', 'cs', 'clk_buf')
        write = DffBuffer(self.nmos_model, self.pmos_model)
        self.subcircuit(write)
        self.X('dff_buf1', write.NAME, 'VDD', 'VSS', 'web', 'we_bar', 'we', 'clk_buf')

    def _add_gated_clocks(self) -> None:
        """Raw selected clocks; clock-high is access and clock-low is recovery."""
        gated_bar = self._and_gate(AND2, GATED_CLOCK_INVERTER)
        self.subcircuit(gated_bar)
        self.X('and2_gated_clk_bar', gated_bar.NAME, 'VDD', 'VSS', 'cs', 'clk_bar', 'gated_clk_bar')
        gated_buf = self._and_gate(AND2, GATED_CLOCK_INVERTER)
        self.subcircuit(gated_buf)
        self.X('and2_gated_clk_buf', gated_buf.NAME, 'VDD', 'VSS', 'cs', 'clk_buf', 'gated_clk_buf')

    def _add_access_request(self) -> None:
        """Clock-high request, admitted only after far PRE has released."""
        self.access_request = 'gated_clk_buf'
        if self.precharge_off_guard:
            guard = PrechargeOffGuard(self.nmos_model, self.pmos_model,
                                      stages=self.precharge_off_guard_stages,
                                      access_load=self.access_load,
                                      settling_tau=self.precharge_off_tau)
            self.subcircuit(guard)
            self.access_request = 'access_request'
            self.X('access_guard', guard.NAME, 'VDD', 'VSS',
                   'gated_clk_buf', 'pre_far', self.access_request, 'pre_off_ready')
        # A write's setup starts from its far physical enable, after isolation.
        # Reads bypass that wait. Delay only assertion; the raw request still
        # releases WL at clock fall, independently of all these delayed paths.
        ready = WriteReadyNor(self.nmos_model, self.pmos_model, .09e-6, .54e-6, GATE_LENGTH)
        inverter = self._unit_inverter('_write_ready')
        start = self._and_gate(WordlineStartAnd, GATED_CLOCK_INVERTER)
        for sub in (ready, inverter, start):
            self.subcircuit(sub)
        self.X('write_ready_nor', ready.NAME, 'VDD', 'VSS', 'we_bar', 'wen_far', 'write_ready_bar')
        self.X('write_ready', inverter.NAME, 'VDD', 'VSS', 'write_ready_bar', 'write_ready')
        self.X('wordline_start', start.NAME, 'VDD', 'VSS', self.access_request, 'write_ready', 'wordline_start')
        delay = WordlineSetupDelay(self.nmos_model, self.pmos_model, stages=8)
        decoder_tau = 0.0
        if self.num_rows > 64:
            # The third decoder level's weak complement inverter drives a
            # full-height wire. Observe equivalent load on both polarities
            # before allowing WL, including a first read or changed address.
            # Existing dummy inputs model the unit decoder's gate fan-out.
            wire_cap = self.num_rows * self.interconnect.bl.capacitance_per_pitch
            for stage in (1, 2):
                delay.C(f'decoder_settle_{stage}', f'wen_dly_{stage}', 'VSS', 2 * wire_cap)
            # Bound the low-bit trunk plus one local complement line. This
            # includes their gate loads, not just the metal's passive R*C.
            trunk_cap = 2 * wire_cap + ceil(self.num_rows / 8.0) * 3e-15
            local_cap = wire_cap + 2.5e-15
            decoder_tau = 2 * self.num_rows * self.interconnect.bl.resistance_per_pitch * (trunk_cap + local_cap)
        scale = self.sizing.access_drive_scale
        output_sizes = (GATED_CLOCK_INVERTER if scale == 6 else
                        (UNIT_NMOS_WIDTH * scale, UNIT_PMOS_WIDTH * scale))
        gate = self._and_gate(WordlineSetupAnd, output_sizes)
        self.subcircuit(delay)
        self.subcircuit(gate)
        setup_input = 'wordline_start'
        setup_tau = max(self.enable_off_tau, decoder_tau)
        if setup_tau:
            # The observer switches near mid-rail; the same frozen enable-wire
            # scale covers the remaining rise to a fully enabled driver. Reads
            # share this conservative preparation interval.
            setup_input = 'wordline_start_filtered'
            self.R('wordline_setup', 'wordline_start', setup_input, max(1.0, setup_tau / 1e-15))
            self.C('wordline_setup', setup_input, 'VSS', 1e-15)
        self.X('wordline_setup_delay', delay.NAME, 'VDD', 'VSS', setup_input, 'access_settled')
        self.X('wordline_setup_gate', gate.NAME, 'VDD', 'VSS', self.access_request,
               'access_settled', 'wordline_request')

    def _add_wordline_enable(self) -> None:
        """Read WL ends at read_done, before sensing; write WL ends at clock fall."""
        buffer = WordlineEnableBuffer(self.nmos_model, self.pmos_model,
                                      drive_scale=self.sizing.wl_en_scale, fold_gates=self.effort_buffers)
        self.subcircuit(buffer)
        self.X('wl_en', buffer.NAME, 'VDD', 'VSS', 'wordline_request', 'read_done_bar', 'wl_en')
        inverter = self._unit_inverter('_wl_en_bar', self.sizing.wlb_scale)
        self.subcircuit(inverter)
        self.X('inv_wl_en_bar', inverter.NAME, 'VDD', 'VSS', 'wl_en', 'wl_en_bar')

    def _add_replica_delay(self) -> None:
        """rbl -> rbl_delay (inverting chain) -> rbl_delay_bar: the replica-timed sense trigger."""
        chain = ReplicaDelayChain(self.nmos_model, self.pmos_model, stages=self.dc_stages)
        self.subcircuit(chain)
        self.X('delaychain', chain.NAME, 'VDD', 'VSS', 'rbl', 'rbl_delay')
        inverter = self._unit_inverter()
        self.subcircuit(inverter)
        self.X('inv_rbl_delay_bar', inverter.NAME, 'VDD', 'VSS', 'rbl_delay', 'rbl_delay_bar')

    def _add_read_completion(self) -> None:
        """Sample/isolate the read, release WL, then enable regeneration.

        RBL stays discharged until clock-low precharge, so read_done cannot
        re-arm the wordline during the same access. It is not used for writes.
        """
        gate = self._and_gate(ReadDoneAnd, ENABLE_INVERTER)
        inverter = self._unit_inverter('_read_done_bar', self.sizing.senb_scale)
        self.subcircuit(gate)
        self.subcircuit(inverter)
        self.X('read_done', gate.NAME, 'VDD', 'VSS', 'rbl_delay', self.access_request, 'we_bar', 'read_done')
        self.X('read_done_bar', inverter.NAME, 'VDD', 'VSS', 'read_done', 'read_done_bar')

    def _add_wordline_off_guard(self) -> None:
        """wordline_off: WL is physically off, as seen on the far replica wordline.

        With terminal RC the physical wordline outlives wl_en, so the far
        replica wordline is observed (half-unit inverter) and its falling tail
        given `precharge_guard_stages` of settling before the bitlines may be
        restored or driven (V2.1.0 / V2.1.1).
        """
        self.wordline_off = 'wl_en_bar'
        if not self.replica_precharge_guard:
            return
        observer = Pinv(self.nmos_model, self.pmos_model, *OBSERVER_WIDTHS,
                        length=GATE_LENGTH, num='_rwl_precharge')
        self.subcircuit(observer)
        self.X('rwl_precharge_guard', observer.NAME, 'VDD', 'VSS', 'rwl', 'rwl_pre_bar')
        self.wordline_off = 'rwl_pre_bar'
        if not self.precharge_guard_stages:
            return
        delay = PrechargeGuardDelay(self.nmos_model, self.pmos_model, stages=self.precharge_guard_stages)
        ready = PrechargeGuardAnd(self.nmos_model, self.pmos_model, self.nmos_model, self.pmos_model)
        self.subcircuit(delay)
        self.subcircuit(ready)
        self.X('precharge_guard_delay', delay.NAME, 'VDD', 'VSS', 'rwl_pre_bar', 'rwl_pre_delayed')
        self.X('precharge_guard_ready', ready.NAME, 'VDD', 'VSS', 'rwl_pre_bar', 'rwl_pre_delayed', 'pre_ready')
        self.wordline_off = 'pre_ready'

    def _add_wordline_busy(self) -> None:
        """Busy from wl_en assertion through the observed physical wordline tail."""
        nand = WordlineBusyNand(self.nmos_model, self.pmos_model, NAND_NMOS_WIDTH, NAND_PMOS_WIDTH, GATE_LENGTH)
        inverter = self._unit_inverter('_wordline_idle')
        self.subcircuit(nand)
        self.subcircuit(inverter)
        self.X('wordline_busy_nand', nand.NAME, 'VDD', 'VSS', 'wl_en_bar', self.wordline_off, 'wordline_busy')
        self.X('inv_wordline_idle', inverter.NAME, 'VDD', 'VSS', 'wordline_busy', 'wordline_idle')

    def _add_write_window(self) -> None:
        """Drivers start before WL, then cover its tail after the falling edge.

        Both terms belong to the SAME captured operation. Their overlap spans
        the entire write access; no delayed handoff to the next cycle is needed.
        Clock-low must finish recovery before the next register capture.
        """
        nor = WriteWindowNor(self.nmos_model, self.pmos_model, .09e-6, .54e-6, GATE_LENGTH)
        inverter = self._unit_inverter('_write_window')
        self.subcircuit(nor)
        self.subcircuit(inverter)
        self.X('write_window_nor', nor.NAME, 'VDD', 'VSS', self.access_request, 'wordline_busy', 'write_window_bar')
        self.X('write_window_inv', inverter.NAME, 'VDD', 'VSS', 'write_window_bar', 'write_window')
        self.write_window = 'write_window'

    def _add_write_enable(self) -> None:
        """Registered write request is stable through access and recovery."""
        gate = self._and_gate(GuardedWriteAnd, ENABLE_INVERTER)
        self.subcircuit(gate)
        s = self.sizing
        self.wen_source = self._buffered_output(
            'w_en', gate, ('we', self.write_window, 'iso_ready'), 'w_en', s.wen_buffered,
            'WRITE_ENABLE_BUFFER', 'w_en_buf', drive_scale=ceil(s.wen_load / s.wen_effort), load_units=s.wen_load)

    def _add_sense_enable(self) -> None:
        """s_en = read_done & wordline_idle & iso_ready: regenerate after release.

        It enables the sense-amplifier footers and the output latch; the input
        pass gates are driven by sa_iso.  Buffered above 32 unit loads (V2.0.2:
        the shared inverter had a fan-out of ~75 and a precharge-coupling bump).
        """
        gate = self._and_gate(AND3, ENABLE_INVERTER)
        self.subcircuit(gate)
        s = self.sizing
        self.sen_source = self._buffered_output(
            's_en', gate, ('read_done', 'wordline_idle', 'iso_ready'), 's_en', s.sen_buffered,
            'SENSE_ENABLE_BUFFER', 's_en_buf', drive_scale=ceil(s.sen_load / s.sen_effort), load_units=s.sen_load)

    def _add_sense_isolation(self) -> None:
        """Isolate before either driver or sense footer turns on; release last.

        The request raises ISO ahead of the enables. Their observed outputs
        then keep it high until both have released; no storage latch is needed.
        """
        write = self._and_gate(WritePrepareAnd, ENABLE_INVERTER)
        request = IsolationRequestNor(self.nmos_model, self.pmos_model, .09e-6, .54e-6, GATE_LENGTH)
        gate = IsolationNand(self.nmos_model, self.pmos_model, .72e-6, 1.08e-6, GATE_LENGTH)
        for sub in (write, request, gate):
            self.subcircuit(sub)
        self.X('write_prepare', write.NAME, 'VDD', 'VSS', 'we', self.access_request, 'write_prepare')
        self.X('isolation_request', request.NAME, 'VDD', 'VSS', 'read_done', 'write_prepare', 'isolation_request_bar')
        s = self.sizing
        self._buffered_output(
            'sa_iso', gate, ('isolation_request_bar', 'enables_off'), 'sa_iso', s.iso_buffered(self.effort_buffers),
            'SENSE_ISOLATION_BUFFER', 'sa_iso_buf', drive_scale=ceil(s.iso_load / s.iso_effort), load_units=s.iso_load,
            fall_strength=s.iso_fall_strength)

    def _add_isolation_guard(self) -> None:
        guard = IsolationOffGuard(self.nmos_model, self.pmos_model,
                                  stages=4, access_load=3.75,
                                  settling_tau=self.isolation_tau)
        self.subcircuit(guard)
        self.X('isolation_guard', guard.NAME, 'VDD', 'VSS', 'VDD', 'iso_far',
               'isolation_observed', 'iso_ready')

    def _add_precharge(self) -> None:
        """Clock-low precharge after physical WL and both enables release.

        Every cycle restores BL/BLB and RBL/RBLB, including write and idle
        cycles. The next request is captured only after this recovery phase.
        """
        nand = PNAND3(nmos_model=self.nmos_model, pmos_model=self.pmos_model,
                      nmos_width=0.27e-6, pmos_width=0.27e-6, length=0.05e-6, w_rc=self.w_rc)
        self.subcircuit(nand)
        self.X('pre_unbuf', nand.NAME, 'VDD', 'VSS', 'clk_bar', self.wordline_off, 'enables_off', 'PRE_UNBUF')
        buffer = TaperedBuffer('PRECHARGE_BUFFER', effort_based=self.effort_buffers, drive_scale=self.sizing.pre_scale,
                               nmos_model=self.nmos_model, pmos_model=self.pmos_model,
                               load_units=self.pre_load)
        self.subcircuit(buffer)
        self.X('pre', buffer.NAME, 'VDD', 'VSS', 'PRE_UNBUF', 'PRE')

    def _add_enable_observers(self) -> None:
        """Observe root and far enables; allow only their settled release.

        A root output can fall while its far wire is still high. Recovery and
        ISO release must wait for the actual peripheral terminals, including
        their RC tail. The root inhibit still responds immediately to assertion.
        """
        inverter = self._unit_inverter('_s_en_bar', self.sizing.senb_scale)
        nor = EnablesOffNor(nmos_model=self.nmos_model, pmos_model=self.pmos_model,
                            nmos_width=0.09e-6, pmos_width=0.54e-6, length=0.05e-6, w_rc=self.w_rc)
        self.subcircuit(inverter)
        self.subcircuit(nor)
        self.X('inv_s_en_bar', inverter.NAME, 'VDD', 'VSS', 's_en', 's_en_bar')
        self.X('enables_off_nor', nor.NAME, 'VDD', 'VSS', 's_en', 'w_en', 'local_enables_off')
        far = FarEnablesOffNor(self.nmos_model, self.pmos_model, .045e-6, .27e-6, GATE_LENGTH)
        guard = EnableOffGuard(self.nmos_model, self.pmos_model, stages=2,
                               access_load=6.5, settling_tau=self.enable_off_tau)
        self.subcircuit(far)
        self.subcircuit(guard)
        self.X('far_enables_off', far.NAME, 'VDD', 'VSS', 'sen_far', 'wen_far', 'far_enables_off')
        self.X('enable_off_guard', guard.NAME, 'VDD', 'VSS', 'local_enables_off',
               'far_enables_off', 'enables_off', 'far_enables_ready')
