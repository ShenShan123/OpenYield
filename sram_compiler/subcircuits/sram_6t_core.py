from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
from typing import Dict, Any, Union

class Sram6TCell(BaseSubcircuit):
    NAME = 'SRAM_6T_CELL'
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')

    def __init__(self,
                 pd_model, pu_model, pg_model,
                 pd_width, pu_width, pg_width, length,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 disconnect=False,
                 suffix='',          # 良率模式专用：后缀
                 model_dict=None     # 良率模式专用：模型参数字典
                 ):
        
        if disconnect:
            self.NAME += '_DISCONNECT'
        if suffix:
            self.NAME += suffix

        super().__init__(
            pd_model, pu_model, pd_width, pu_width, length,
            w_rc, pi_res, pi_cap
        )

        self.pd_model = pd_model
        self.pu_model = pu_model
        self.pg_model = pg_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        
        self.w_rc = w_rc
        self.disconnect = disconnect
        self.suffix = suffix
        self.model_dict = model_dict  # 如果不为None，则启用add_usrdefine_mos_model逻辑

        self.add_6T_cell()

    def add_usrdefine_mos_model(self, pdk_model_name, udf_model_name):
        """根据 model_dict 生成 .model 语句,yield模式专有"""
        if not self.model_dict:
            return
            
        model_data = self.model_dict[pdk_model_name]
        mos_type = model_data['type']
        self.raw_spice += f'.model {udf_model_name} {mos_type} '
        params = model_data['parameters']
        for param_name, param_value in params.items():
            if isinstance(param_value, float):
                if abs(param_value) < 1e-3 or abs(param_value) > 1e6:
                    param_str = f"{param_value:.3e}"
                else:
                    param_str = str(param_value)
            else:
                param_str = str(param_value)
            
            # 关键的替换逻辑：用于 Mismatch 分析
            if param_name in ['vth0', 'u0', 'voff']:
                param_str = f"'{param_name}_{udf_model_name}'"

            self.raw_spice += f"{param_name}={param_str} "
        self.raw_spice += "\n"

    def add_6T_cell(self):
        # 1. 节点处理 
        if self.w_rc:
            bl_node = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node = self.add_rc_networks_to_node(self.NODES[4], 1)
            q_node = self.add_rc_networks_to_node('Q', 1)
            qb_node = self.add_rc_networks_to_node('QB', 1)
        else:
            bl_node, blb_node, wl_node = self.NODES[2], self.NODES[3], self.NODES[4]
            q_node, qb_node = 'Q', 'QB'
            
        if self.disconnect:
            data_q, data_qb = 'QD', 'QBD'
        else:
            data_q, data_qb = q_node, qb_node

        # 2. 晶体管实例化 (合并了 Normal 和 Yield 逻辑)
        # 通过判断 model_dict 是否存在，决定是否生成独立模型
        
        # --- Access Transistors (PG) ---
        if self.model_dict:
            # 良率模式：生成带后缀的唯一模型名，并定义模型
            pgl_model = f"{self.pg_model}_PGL{self.suffix}"
            pgr_model = f"{self.pg_model}_PGR{self.suffix}"
            self.add_usrdefine_mos_model(self.pg_model, pgl_model)
            self.add_usrdefine_mos_model(self.pg_model, pgr_model)
        else:
            # 普通模式：直接使用传入的模型名
            pgl_model = self.pg_model
            pgr_model = self.pg_model

        # 这里的 self.pg_width 可以是数字，也可以是字符串(sweep模式)
        self.M('PGL', bl_node, wl_node, data_q, self.NODES[1], model=pgl_model, w=self.pg_width, l=self.length)
        self.M('PGR', blb_node, wl_node, data_qb, self.NODES[1], model=pgr_model, w=self.pg_width, l=self.length)

        # --- Pull-Down Transistors (PD) ---
        if self.model_dict:
            pdl_model = f"{self.pd_model}_PDL{self.suffix}"
            pdr_model = f"{self.pd_model}_PDR{self.suffix}"
            self.add_usrdefine_mos_model(self.pd_model, pdl_model)
            self.add_usrdefine_mos_model(self.pd_model, pdr_model)
        else:
            pdl_model = self.pd_model
            pdr_model = self.pd_model

        self.M('PDL', data_q, 'QB', self.NODES[1], self.NODES[1], model=pdl_model, w=self.pd_width, l=self.length)
        self.M('PDR', data_qb, 'Q', self.NODES[1], self.NODES[1], model=pdr_model, w=self.pd_width, l=self.length)

        # --- Pull-Up Transistors (PU) ---
        if self.model_dict:
            pul_model = f"{self.pu_model}_PUL{self.suffix}"
            pur_model = f"{self.pu_model}_PUR{self.suffix}"
            self.add_usrdefine_mos_model(self.pu_model, pul_model)
            self.add_usrdefine_mos_model(self.pu_model, pur_model)
        else:
            pul_model = self.pu_model
            pur_model = self.pu_model

        self.M('PUL', data_q, 'QB', self.NODES[0], self.NODES[0], model=pul_model, w=self.pu_width, l=self.length)
        self.M('PUR', data_qb, 'Q', self.NODES[0], self.NODES[0], model=pur_model, w=self.pu_width, l=self.length)


class Sram6TCore(SubCircuitFactory):    #构建sram阵列
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###

    def __init__(self, num_rows: int, num_cols: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 pd_width=0.205e-6, pu_width=0.09e-6,pg_width=0.135e-6, length=50e-9,
                 model_dict=None ,
                 w_rc=False,
                 ):
        #  disconnect=False, target_row=None, target_col=None):

        self.NAME = f"SRAM_6T_CORE_{num_rows}x{num_cols}"
        # Define nodes
        self.NODES = (  #动态生成第几列第几行
            'VDD',  # Power supply
            'VSS',  # Ground
            *[f'BL{i}' for i in range(num_cols)],  # Bitlines
            *[f'BLB{i}' for i in range(num_cols)],  # Bitline bars
            *[f'WL{i}' for i in range(num_rows)],  # Wordlines
        )
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        # other config
        self.w_rc = w_rc
        self.model_dict = model_dict

        # Build the array
        self.build_array(num_rows, num_cols)        #构建阵列
        # set instance prefix and the name of 6t cell
        self.inst_prefix = "XSRAM_6T_CELL"          #设置实例的前缀

    def build_array(self, num_rows, num_cols):
        # Generate SRAM cells   #创建单个sram单元子电路
        if self.model_dict is None:
            subckt_6t_cell = Sram6TCell(
                self.pd_nmos_model  , self.pu_pmos_model,self.pg_nmos_model,
                self.pd_width, self.pu_width,
                self.pg_width, self.length,
                w_rc=self.w_rc,
                model_dict=self.model_dict,
                suffix=f"_{num_rows}x{num_cols}" if self.model_dict is not None else ""
            )

            # define the cell subcircuit    #添加cell单元
            self.subcircuit(subckt_6t_cell)

        # Instantiate SRAM cells    #遍历所有行和列，实例化单元子电路
        for row in range(num_rows):
            for col in range(num_cols):

                if self.model_dict is not None:
                    subckt_6t_cell = Sram6TCell(
                        self.pd_nmos_model  , self.pu_pmos_model,self.pg_nmos_model,
                        self.pd_width, self.pu_width,
                        self.pg_width, self.length,
                        w_rc=self.w_rc,
                        model_dict=self.model_dict,
                        suffix=f"_{row}_{col}" #if self.model_dict else ""
                    )                    
                    self.subcircuit(subckt_6t_cell)

                self.X(
                    subckt_6t_cell.name + f"_{row}_{col}"  if self.model_dict is None else subckt_6t_cell.name,#实例的名称
                    subckt_6t_cell.name,                    #引用的子电路类型，即Sram6TCell
                    self.NODES[0],  # Power net             #连接的节点名
                    self.NODES[1],  # Ground net
                    f'BL{col}',  # Connect to column bitline
                    f'BLB{col}',  # Connect to column bitline bar
                    f'WL{row}',  # Connect to row wordline
                )
