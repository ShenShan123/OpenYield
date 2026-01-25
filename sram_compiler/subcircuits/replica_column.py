from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
import os
# from utils import model_dict2str
from typing import Dict, Any, Union

# ==============================================================================
# 1. Replica Cell (标准单元 - 纯电路拓扑)
# ==============================================================================
class Replica_Cell(BaseSubcircuit):
    """
    Replica Cell 拓扑结构。
    """
    NAME = 'Replica_CELL'
    NODES = ('VDD', 'VSS', 'RBL', 'RBLB', 'WL')

    def __init__(self,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width: float, pu_width: float,
                 pg_width: float, length: float,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 disconnect=False
                 ):
        
        if disconnect:
            self.NAME += '_DISCONNECT'

        # 调用父类初始化 (BaseSubcircuit)
        super().__init__(
            pd_nmos_model, pu_pmos_model,
            pd_width, pu_width, length,
            w_rc, pi_res, pi_cap
        )
        
        # 保存参数
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        self.w_rc = w_rc
        self.disconnect = disconnect

        # 处理节点名称
        if not self.w_rc:
            bl_node = self.NODES[2]
            blb_node = self.NODES[3]
            wl_node = self.NODES[4]
            q_node = 'Q'
            qb_node = 'QB'
        else:
            bl_node = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node = self.add_rc_networks_to_node(self.NODES[4], 1)
            q_node = self.add_rc_networks_to_node('Q', 1)
            qb_node = self.add_rc_networks_to_node('QB', 1)

        self.add_6T_cell(bl_node, blb_node, wl_node, q_node, qb_node)

    def add_6T_cell(self, bl_node, blb_node, wl_node, q_node, qb_node):
        # 处理断开模式
        if self.disconnect:
            data_q = 'QD'
            data_qb = 'QBD'
        else:
            data_q = q_node
            data_qb = qb_node
        
        # Access transistors
        self.M('PGL', bl_node, wl_node, data_q, self.NODES[1], 
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        self.M('PGR', blb_node, wl_node, self.NODES[0], self.NODES[1], 
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        self.M('PDL', data_q, self.NODES[0], self.NODES[1], self.NODES[1], 
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('PUL', data_q, self.NODES[0], self.NODES[0], self.NODES[0], 
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        self.M('PDR', self.NODES[0], 'Q', self.NODES[1], self.NODES[1], 
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('PUR', self.NODES[0], 'Q', self.NODES[0], self.NODES[0], 
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        


class Replica_Column(SubCircuitFactory):
    ###
    # Replica Column SubCircuitFactory class.
    # Configurable number of rows.
    ###

    def __init__(self, num_rows: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width=0.205e-6, pu_width=0.09e-6,
                 pg_width=0.135e-6, length=50e-9,
                 w_rc=False,
                 ):
        self.NAME = f"sram_{num_rows+1}x1_replica_column"
        
        # Define nodes - shared bitlines and individual wordlines
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            'RBL',   # Bitline
            'RBLB',  # Bitline bar
            *[f'WL{i}' for i in range(num_rows+1)],  # Wordlines (0 to num_rows)多生成一行
        )
        
        super().__init__()
        self.num_rows = num_rows
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_pmos_model = pg_nmos_model
        # Transistor Sizes (FreePDK45 uses nanometers)
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        # other config
        self.w_rc = w_rc

        # Build the array
        self.build_array(self.num_rows)        #构建阵列
        # set instance prefix and the name of replica cell
        self.inst_prefix = "XReplica_Column"          #设置实例的前缀

    def build_array(self, num_rows):
        # Generate replica cells
        replica_cell = Replica_Cell(
            self.pd_nmos_model, self.pu_pmos_model, self.pg_pmos_model,
            self.pd_width, self.pu_width,
            self.pg_width, self.length,
            w_rc=self.w_rc,
        )

        # define the cell subcircuit
        self.subcircuit(replica_cell)

        # Instantiate replica cells - sharing the same bitlines but connecting to different wordlines
        for row in range(num_rows+1):
            self.X(
                replica_cell.name + f"_{row}",  # Instance name
                replica_cell.name,              # Subcircuit type
                self.NODES[0],                  # Power net (VDD)
                self.NODES[1],                  # Ground net (VSS)
                self.NODES[2],                  # Shared bitline (BL)
                self.NODES[3],                  # Shared bitline bar (BLB)
                f'WL{row}',                     # Individual wordline connection
            )
