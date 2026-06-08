from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
from .sram_cell_add_equivalent import add_10t_equivalent_circuit

class Sram10TCell(BaseSubcircuit):
    NAME = 'SRAM_10T_CELL'
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')  # Adding node for 10T cell

    def __init__(self,
                 pd_model, pu_model, pg_model, fd_model,
                 pd_width, pu_width, pg_width, fd_width, length,
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
        self.fd_model = fd_model  # New field for pass gate transistors
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.fd_width = fd_width  # Width for pass gate transistors
        self.length = length
        
        self.w_rc = w_rc
        self.disconnect = disconnect
        self.suffix = suffix
        self.model_dict = model_dict  # 如果不为None，则启用add_usrdefine_mos_model逻辑

        self.add_10T_cell()

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

    def add_10T_cell(self):
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
        self.M('AXL', bl_node, wl_node, data_q, 'VSS', model=pgl_model, w=self.pg_width, l=self.length)
        self.M('AXR', blb_node, wl_node, data_qb, 'VSS', model=pgr_model, w=self.pg_width, l=self.length)

        # --- Pull-Up Transistors (PU) ---
        if self.model_dict:
            pul_model = f"{self.pu_model}_PUL{self.suffix}"
            pur_model = f"{self.pu_model}_PUR{self.suffix}"
            self.add_usrdefine_mos_model(self.pu_model, pul_model)
            self.add_usrdefine_mos_model(self.pu_model, pur_model)
        else:
            pul_model = self.pu_model
            pur_model = self.pu_model

        self.M('PL', data_q, 'QB', 'VDD', 'VDD', model=pul_model, w=self.pu_width, l=self.length)
        self.M('PR', data_qb, 'Q', 'VDD', 'VDD', model=pur_model, w=self.pu_width, l=self.length)

        # --- First Stage Pull-Down Transistors (PD) ---
        if self.model_dict:
            pdl1_model = f"{self.pd_model}_PDL1{self.suffix}"
            pdr1_model = f"{self.pd_model}_PDR1{self.suffix}"
            self.add_usrdefine_mos_model(self.pd_model, pdl1_model)
            self.add_usrdefine_mos_model(self.pd_model, pdr1_model)
        else:
            pdl1_model = self.pd_model
            pdr1_model = self.pd_model

        self.M('NL1', data_q, 'QB', 'VNL', 'VSS', model=pdl1_model, w=self.pd_width, l=self.length)
        self.M('NR1', data_qb, 'Q', 'VNR', 'VSS', model=pdr1_model, w=self.pd_width, l=self.length)

        # --- Second Stage Pull-Down Transistors (PD) ---
        if self.model_dict:
            pdl2_model = f"{self.pd_model}_PDL2{self.suffix}"
            pdr2_model = f"{self.pd_model}_PDR2{self.suffix}"
            self.add_usrdefine_mos_model(self.pd_model, pdl2_model)
            self.add_usrdefine_mos_model(self.pd_model, pdr2_model)
        else:
            pdl2_model = self.pd_model
            pdr2_model = self.pd_model

        self.M('NL2', 'VNL', 'QB', 'VSS', 'VSS', model=pdl2_model, w=self.pd_width, l=self.length)
        self.M('NR2', 'VNR', 'Q', 'VSS', 'VSS', model=pdr2_model, w=self.pd_width, l=self.length)

        # --- Pass Gate/Fix Logic Transistors (FD) ---
        if self.model_dict:
            fd_l_model = f"{self.fd_model}_FD_L{self.suffix}"
            fd_r_model = f"{self.fd_model}_FD_R{self.suffix}"
            self.add_usrdefine_mos_model(self.fd_model, fd_l_model)
            self.add_usrdefine_mos_model(self.fd_model, fd_r_model)
        else:
            fd_l_model = self.fd_model
            fd_r_model = self.fd_model

        self.M('NFL', 'VNL', data_q, 'VDD', 'VSS', model=fd_l_model, w=self.fd_width, l=self.length)
        self.M('NFR', 'VNR', data_qb, 'VDD', 'VSS', model=fd_r_model, w=self.fd_width, l=self.length)

class Sram10TCore(SubCircuitFactory):    #构建sram阵列
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###
    add_equivalent_circuit = add_10t_equivalent_circuit
    def __init__(self, num_rows: int, num_cols: int,
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str, fd_nmos_model: str,
                 pd_width=0.205e-6, pu_width=0.09e-6, pg_width=0.135e-6, fd_width=0.135e-6, length=50e-9,
                 model_dict=None ,
                 w_rc=False,
                 target_row=0, target_col=0,
                 real_cell_mode=0,
                 write_power_model=False,
                 q_init_val=0,
                 global_config=None,
                 pi_res=None,
                 pi_cap=None,
                 ):
        #  disconnect=False, target_row=None, target_col=None):

        self.NAME = f"SRAM_10T_CORE_{num_rows}x{num_cols}"
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
        self.target_row = target_row
        self.target_col = target_col
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.fd_nmos_model = fd_nmos_model  # New field for pass gate transistors
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.fd_width = fd_width  # Width for pass gate transistors
        self.length = length
        # other config
        self.w_rc = w_rc
        self.real_cell_mode = self._normalize_real_cell_mode(real_cell_mode)
        self.write_power_model = write_power_model
        self.q_init_val = q_init_val
        self.model_dict = model_dict
        self.global_config = global_config
        from .base_subcircuit import BaseSubcircuit
        self.pi_res = pi_res if pi_res is not None else BaseSubcircuit.DEFAULT_PI_RES
        self.pi_cap = pi_cap if pi_cap is not None else BaseSubcircuit.DEFAULT_PI_CAP

        # Build the array
        self.build_array(num_rows, num_cols)        #构建阵列
        # set instance prefix and the name of 10t cell
        self.inst_prefix = "XSRAM_10T_CELL"          #设置实例的前缀


    def build_array(self, num_rows, num_cols):
        # Generate SRAM cells   #创建单个sram单元子电路
        if self.model_dict is None:
            subckt_10t_cell = Sram10TCell(
                self.pd_nmos_model, self.pu_pmos_model, self.pg_nmos_model, self.fd_nmos_model,
                self.pd_width, self.pu_width, self.pg_width, self.fd_width, self.length,
                w_rc=self.w_rc,
                model_dict=self.model_dict,
                suffix=f"_{num_rows}x{num_cols}" if self.model_dict is not None else ""
            )

            # define the cell subcircuit    #添加cell单元
            self.subcircuit(subckt_10t_cell)

        # Instantiate SRAM cells    #遍历所有行和列，实例化单元子电路
        for row in range(num_rows):
            for col in range(num_cols):

                if self.model_dict is not None:
                    subckt_10t_cell = Sram10TCell(
                        self.pd_nmos_model, self.pu_pmos_model, self.pg_nmos_model, self.fd_nmos_model,
                        self.pd_width, self.pu_width, self.pg_width, self.fd_width, self.length,
                        w_rc=self.w_rc,
                        model_dict=self.model_dict,
                        suffix=f"_{row}_{col}" 
                    )                    
                    self.subcircuit(subckt_10t_cell)

                if self._should_instantiate_real_cell(row, col):
                    self.X(
                        subckt_10t_cell.name + f"_{row}_{col}"  if self.model_dict is None else subckt_10t_cell.name,#实例的名称
                        subckt_10t_cell.name,                    #引用的子电路类型，即Sram10TCell
                        self.NODES[0],  # Power net             #连接的节点名
                        self.NODES[1],  # Ground net
                        f'BL{col}',  # Connect to column bitline
                        f'BLB{col}',  # Connect to column bitline bar
                        f'WL{row}',  # Connect to row wordline
                    )
        if self.real_cell_mode != 0:#非全真实模式：对未实例化的单元添加等效电路
            print(f"[DEBUG] generating equivalent circuit for unused cells")
            self.add_equivalent_circuit()  # 添加等效电路

    def _normalize_real_cell_mode(self, mode):
        # 与 OpenYield2.5 对齐：
        #   0 = 全真实（所有 cell）
        #   1 = 十字（目标行或目标列为真实，其余等效）
        #   2 = 仅目标行真实
        #   3 = 仅目标列真实
        #   4 = 仅目标 cell 真实
        # bool 兼容旧 use_equivalent: True→1(十字), False→0(全真实)
        if isinstance(mode, bool):
            return 1 if mode else 0
        mode = int(mode)
        if mode not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Invalid real_cell_mode: {mode}. Valid values are 0-4.")
        return mode

    def _should_instantiate_real_cell(self, row, col):
        if self.real_cell_mode == 0:
            return True
        if self.real_cell_mode == 1:
            return row == self.target_row or col == self.target_col
        if self.real_cell_mode == 2:
            return row == self.target_row
        if self.real_cell_mode == 3:
            return col == self.target_col
        if self.real_cell_mode == 4:
            return row == self.target_row and col == self.target_col
        return False

    def _is_unused_cell(self, row, col):
        return not self._should_instantiate_real_cell(row, col)

    def _count_unused_cells_in_row(self, row):
        return sum(1 for col in range(self.num_cols) if self._is_unused_cell(row, col))

    def _count_unused_cells_in_col(self, col):
        return sum(1 for row in range(self.num_rows) if self._is_unused_cell(row, col))

    def _count_total_unused_cells(self):
        return sum(1 for row in range(self.num_rows) for col in range(self.num_cols)
                   if self._is_unused_cell(row, col))
