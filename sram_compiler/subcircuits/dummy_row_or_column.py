from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit

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
                 disconnect=False,
                 cell_pin_rc=None
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
        self.cell_pin_rc = w_rc and bool(cell_pin_rc)
        self.w_rc = w_rc
        self.disconnect = disconnect

        
        self.add_dummy_components()

    def add_dummy_components(self):
        if self.cell_pin_rc:
            bl_node = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node = self.add_rc_networks_to_node(self.NODES[4], 1)
        else:
            bl_node, blb_node, wl_node = self.NODES[2:5]
        q_node = self.add_rc_networks_to_node('Q', 1) if self.w_rc else 'Q'

        # 如果断开连接 (disconnect=True)，使用独立的内部节点名，避免短路
        if self.disconnect:
            data_q = 'QD'
        else:
            data_q = q_node

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
