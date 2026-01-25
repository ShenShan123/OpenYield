from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
import os
# from utils import model_dict2str
from typing import Dict, Any, Union

# ==============================================================================
# 1. Dummy Cell (标准虚假单元)
# ==============================================================================
class Dummy_Cell(BaseSubcircuit):
    """
    Dummy Cell 拓扑结构。
    不再包含 param_sweep 逻辑，参数通过 __init__ 直接传入。
    """
    NAME = 'Dummy_CELL'
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')

    def __init__(self,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width, pu_width, pg_width, length,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 disconnect=False
                 ):
        
        if disconnect:
            self.NAME += '_DISCONNECT'

        # 调用父类初始化
        super().__init__(
            pd_nmos_model, pu_pmos_model, 
            pd_width, pu_width, length,
            w_rc, pi_res, pi_cap
        )

        # 保存参数
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        self.w_rc = w_rc
        self.disconnect = disconnect

        
        self.add_dummy_components()

    def add_dummy_components(self):
        # 1. 处理节点 (RC 和 Disconnect)
        if self.w_rc:
            bl_node = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node = self.add_rc_networks_to_node(self.NODES[4], 1)
            # 内部存储节点
            q_node = self.add_rc_networks_to_node('Q', 1)
            qb_node = self.add_rc_networks_to_node('QB', 1)
        else:
            bl_node, blb_node, wl_node = self.NODES[2], self.NODES[3], self.NODES[4]
            q_node, qb_node = 'Q', 'QB'

        # 如果断开连接 (disconnect=True)，使用独立的内部节点名，避免短路
        if self.disconnect:
            data_q = 'QD'
            data_qb = 'QBD'
        else:
            data_q = q_node
            data_qb = qb_node

        # 2. 实例化 6个 晶体管
        # Pull-Down (Driver) NMOS
        self.M('PDL', data_q, self.NODES[0], self.NODES[1], self.NODES[1], 
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)
        self.M('PDR', self.NODES[0], data_q, self.NODES[1], self.NODES[1], 
               model=self.pd_nmos_model, w=self.pd_width, l=self.length)

        # Pull-Up (Load) PMOS
        self.M('PUL', data_q, self.NODES[0], self.NODES[0], self.NODES[0], 
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)
        self.M('PUR', self.NODES[0], data_q, self.NODES[0],self.NODES[0], 
               model=self.pu_pmos_model, w=self.pu_width, l=self.length)

        # Pass-Gate (Access) NMOS
        self.M('PGL','bl_node_noconnect', wl_node, data_q, self.NODES[1], 
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)
        self.M('PGR','blb_node_noconnect', wl_node,self.NODES[0],  self.NODES[1], 
               model=self.pg_nmos_model, w=self.pg_width, l=self.length)

# ==============================================================================
# 2. Dummy Column (列阵列 )
# ==============================================================================
            
class Dummy_Column(SubCircuitFactory):
    """Represents a column of dummy cells sharing bitlines"""
    
    
    def __init__(self, num_rows: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width: float, pu_width: float, pg_width: float, length: float,
                 w_rc=False,
                 disconnect=False,):
        # Set the nodes dynamically
        self.NAME = f"sram_{num_rows+3}x1_Dummy_column"
        # Define nodes - shared bitlines and individual wordlines
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            'BL',   # Bitline
            'BLB',  # Bitline bar
            *[f'WL{i}' for i in range(num_rows+3)],  # Wordlines (0 to num_rows)多生成三行
        )
   
        super().__init__( )
        self.num_rows = num_rows
        self.pg_nmos_model = pg_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pd_nmos_model = pd_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        self.w_rc = w_rc
        self.disconnect = disconnect
   
        
        # Build the array
        self.build_array(self.num_rows)
         # set instance prefix
        self.inst_prefix = "XDummy_Column"  # 设置实例的前缀

    def build_array(self, num_rows: int):
        """Build the dummy cell array with shared bitlines and individual wordlines"""
        dummy_cell = Dummy_Cell(
            self.pd_nmos_model, self.pu_pmos_model, self.pg_nmos_model,
            self.pd_width, self.pu_width,
            self.pg_width, self.length,
            w_rc=self.w_rc,      
            disconnect=self.disconnect,
        )
        # define the cell subcircuit
        self.subcircuit(dummy_cell)
        # Create dummy cells for each row
        for row in range(num_rows + 3):     
            # Add the dummy cell subcircuit
            self.X(
                dummy_cell.name + f"_{row}",
                dummy_cell.name, 
                self.NODES[0], 
                self.NODES[1],
                self.NODES[2],                  # Shared bitline (BL)
                self.NODES[3],                  # Shared bitline bar (BLB)
                f'WL{row}', 
            )

# ==============================================================================
# 3. Dummy Row (行阵列 - 纯电路拓扑)
# ==============================================================================
class Dummy_Row(SubCircuitFactory):
    """Represents a row of dummy cells sharing wordline """
    
    def __init__(self, num_cols: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width: float, pu_width: float, pg_width: float, length: float,
                 w_rc=False,
                 disconnect=False,
                 ):
        # Set the name and nodes
        self.NAME = f"sram_1x{num_cols+1}_Dummy_row"
        
        # Define nodes - shared wordline and individual bitlines
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            *[f'BL{i}' for i in range(num_cols+1)],   # Bitlines
            *[f'BLB{i}' for i in range(num_cols+1)],   # Bitline bars
            'WL',   # Shared wordline
        )
        
        super().__init__()

        self.num_cols = num_cols
        self.pg_nmos_model = pg_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pd_nmos_model = pd_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length

        self.w_rc = w_rc
        self.disconnect = disconnect

        # Build the array
        self.build_array(self.num_cols)
        # set instance prefix
        self.inst_prefix = "XDummy_Row"  # 设置实例的前缀

    def build_array(self, num_cols: int):
        """Build the dummy cell array with shared wordline and individual bitlines"""
        # Create dummy cell instance
        dummy_cell = Dummy_Cell(
            self.pd_nmos_model, self.pu_pmos_model, self.pg_nmos_model,
            self.pd_width, self.pu_width,
            self.pg_width, self.length,
            w_rc=self.w_rc,
            disconnect=self.disconnect, 
        )

        # define the cell subcircuit
        self.subcircuit(dummy_cell)
        # Create dummy cells for each column
        for col in range(num_cols+1):
            # Add the dummy cell subcircuit
            self.X(
                dummy_cell.name + f"_{col}", 
                dummy_cell.name, 
                self.NODES[0], 
                self.NODES[1],
                f'BL{col}', 
                f'BLB{col}', 
                'WL',
            )


             
              
