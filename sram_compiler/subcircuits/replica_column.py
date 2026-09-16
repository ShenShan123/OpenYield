"""Replica column: the self-timed reference for the sense enable.

`ReplicaColumn` is one extra bitline pair with `num_rows` replica cells on the
same tapped wire geometry as a real column.  Only the last `K` cells are
driven by the replica wordline (the testbench ties the other wordlines to
ground), so a real-row access never discharges RBL in parallel with them.

`ReplicaCell` stores a fixed 0: its left storage node is held low by an
always-on pull-down, its right node is tied to VDD, so an active replica
wordline discharges RBL through the same pass gate and pull-down as an array
cell reading a 0 while RBLB stays high.  Widths and models follow the array
cell through ``resolve_driver_sizes`` (matched replica), which is why the
V2.1.5 10T pull-down resize moved the sense trigger of every 10T array.
Instance names (``XReplica_CELL_{row}``) and node names are part of the
testbench probe contract.
"""
from __future__ import annotations

from typing import Optional

from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Unit import u_Ohm, u_pF

from sram_compiler.interconnect import add_tapped_line, resolve_interconnect
from .base_subcircuit import BaseSubcircuit


class ReplicaCell(BaseSubcircuit):
    """Replica bitcell with a fixed stored 0 (6T or 10T topology)."""
    NAME = 'Replica_CELL'
    NODES = ('VDD', 'VSS', 'RBL', 'RBLB', 'WL')

    def __init__(self,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str, fd_nmos_model: Optional[str],
                 pd_width, pu_width, pg_width, length, fd_width=None,
                 sram_cell_type: str = "SRAM_6T_CELL",
                 w_rc: bool = False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 cell_pin_rc: Optional[bool] = None) -> None:
        super().__init__(
            pd_nmos_model, pu_pmos_model,
            pd_width, pu_width, length,
            w_rc, pi_res, pi_cap
        )
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.fd_nmos_model = fd_nmos_model
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.fd_width = fd_width if fd_width is not None else pg_width   # 10T feedback device, pass-gate width by default
        self.length = length
        self.sram_cell_type = sram_cell_type
        self.cell_pin_rc = w_rc and bool(cell_pin_rc)
        self.w_rc = w_rc

        # Local series stubs on the pins (cell_pin_rc) and on the storage nodes
        # (w_rc), exactly as in the array cells.
        if self.cell_pin_rc:
            bl_node = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node = self.add_rc_networks_to_node(self.NODES[4], 1)
        else:
            bl_node, blb_node, wl_node = self.NODES[2:5]
        if self.w_rc:
            q_node = self.add_rc_networks_to_node('Q', 1)
            qb_node = (self.add_rc_networks_to_node('QB', 1)
                       if self.sram_cell_type == 'SRAM_10T_CELL' else 'QB')
        else:
            q_node, qb_node = 'Q', 'QB'

        if self.sram_cell_type == 'SRAM_10T_CELL':
            self.add_10T_cell(bl_node, blb_node, wl_node, q_node, qb_node)
        else:
            self.add_6T_cell(bl_node, blb_node, wl_node, q_node)

    def add_6T_cell(self, bl_node: str, blb_node: str, wl_node: str, q_node: str) -> None:
        """6T replica: Q held at 0 (PDL gate at VDD), the right node tied to VDD."""
        vdd, vss = self.NODES[0], self.NODES[1]
        # Access transistors: RBL to the stored 0, RBLB to VDD.
        self.M('PGL', bl_node, wl_node, q_node, vss,
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        self.M('PGR', blb_node, wl_node, vdd, vss,
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        # Left inverter with its input tied high: the pull-down holds Q at 0.
        self.M('PDL', q_node, vdd, vss, vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('PUL', q_node, vdd, vdd, vdd,
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        # Right inverter with its output tied to VDD (gate loads of the storage node).
        self.M('PDR', vdd, 'Q', vss, vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('PUR', vdd, 'Q', vdd, vdd,
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)

    def add_10T_cell(self, bl_node: str, blb_node: str, wl_node: str, q_node: str, qb_node: str) -> None:
        """10T (Schmitt-trigger) replica with the stored state fixed to Q=0, QB=1."""
        vdd, vss = self.NODES[0], self.NODES[1]
        q_fix, qb_fix = vss, vdd   # the cross-coupled gate inputs, tied to the stored levels
        # Access transistors
        self.M('AXL', bl_node, wl_node, q_node, vss,
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        self.M('AXR', blb_node, wl_node, qb_node, vss,
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        # Pull-up transistors
        self.M('PL', q_node, qb_fix, vdd, vdd,
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        self.M('PR', qb_node, q_fix, vdd, vdd,
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        # First-stage pull-downs
        self.M('NL1', q_node, qb_fix, 'VNL', vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('NR1', qb_node, q_fix, 'VNR', vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        # Second-stage pull-downs
        self.M('NL2', 'VNL', qb_fix, vss, vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('NR2', 'VNR', q_fix, vss, vss,
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        # Feedback devices
        self.M('NFL', 'VNL', q_node, vdd, vss,
               model=self.fd_nmos_model, w=self.fd_width, l=self.length)
        self.M('NFR', 'VNR', qb_node, vdd, vss,
               model=self.fd_nmos_model, w=self.fd_width, l=self.length)


class ReplicaColumn(SubCircuitFactory):
    """`num_rows` replica cells on tapped RBL / RBLB wires, one wordline pin per row.

    `num_cols` is accepted for symmetry with the array factories; the replica
    column itself is one column wide.
    """

    def __init__(self, num_rows: int, num_cols: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str, fd_nmos_model: Optional[str],
                 pd_width=0.205e-6, pu_width=0.09e-6,
                 pg_width=0.135e-6, length=50e-9, fd_width=None,
                 w_rc: bool = False,
                 sram_cell_type: str = 'SRAM_6T_CELL',
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 interconnect=None) -> None:
        self.interconnect = resolve_interconnect(interconnect)
        self.cell_count = num_rows
        self.NAME = f"sram_{self.cell_count}x1_replica_column"
        self.NODES = (
            'VDD',
            'VSS',
            'RBL',
            'RBLB',
            *[f'WL{i}' for i in range(self.cell_count)],
        )
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.fd_nmos_model = fd_nmos_model
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        self.fd_width = fd_width if fd_width is not None else pg_width
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap
        self.sram_cell_type = sram_cell_type
        # Physical bitline pair: one centred tap per row, the far end past the last cell.
        add_tapped_line(self, 'RBL', 'RBL', num_rows, self.interconnect.bl)
        add_tapped_line(self, 'RBLB', 'RBLB', num_rows, self.interconnect.bl)
        self.build_array()
        self.inst_prefix = "XReplica_Column"

    def build_array(self) -> None:
        """One replica cell per row, each on its own bitline taps and wordline pin."""
        replica_cell = ReplicaCell(
            self.pd_nmos_model, self.pu_pmos_model, self.pg_nmos_model, self.fd_nmos_model,
            self.pd_width, self.pu_width,
            self.pg_width, self.length, self.fd_width,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            sram_cell_type=self.sram_cell_type,
            cell_pin_rc=self.interconnect.cell_pin_rc,
        )
        self.subcircuit(replica_cell)
        for row in range(self.cell_count):
            self.X(
                replica_cell.name + f"_{row}",
                replica_cell.name,
                self.NODES[0],
                self.NODES[1],
                f'RBL_tap{row}',
                f'RBLB_tap{row}',
                f'WL{row}',
            )
