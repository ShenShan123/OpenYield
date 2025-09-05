from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from .base_subcircuit import BaseSubcircuit

class Precharge(BaseSubcircuit):    #只需要三个PMOS
    """
    Precharge circuit for SRAM bitlines with dynamically adjusted strength.
    根据行数动态调节强度的SRAM位线预充电路。驱动位线,行数越多,位线越长,需要的驱动能力越强
    """
    NAME = "PRECHARGE"
    # Power, Precharge Enable (0 for assertion), BL, BLB
    NODES = ('VDD', 'ENB', 'BL', 'BLB') 

    def __init__(self, pmos_model_name, base_pmos_width=0.27e-6, length=50e-9, 
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF, num_rows=16,
                 sweep_precharge = False,pmos_modle_choices='NMOS_VTG'
                 ):
        
        super().__init__(
            None, pmos_model_name, 
            base_pmos_width, base_pmos_width, length,
            w_rc, pi_res, pi_cap,
        )

        self.pmos_modle_choices=pmos_modle_choices
        self.num_rows = num_rows
        self.sweep_precharge = sweep_precharge
        self.pmos_width = self.calculate_dynamic_width(base_pmos_width, num_rows)
          # 定义该类需要读取的列名
        self.REQUIRED_COLUMNS = ['pmos_model_precharge']
        # 读取参数文件中的模型名
        self.mos_model_index = self.read_mos_model_from_param_file(self.REQUIRED_COLUMNS)
        self.add_precharge_transistors()#预充电单元函数


    def calculate_dynamic_width(self, base_width, num_rows):
        """
        Dynamically adjust the transistor width based on the number of rows.
        This is a simple linear scaling; you might need a more complex function.
        根据行数动态线性缩放晶体管宽度
        """
        scaling_factor = num_rows / 16  
        scaling_factor = 0.5 if scaling_factor < 0.5 else scaling_factor    #缩放因子至少为0.5

        return base_width * scaling_factor

    def add_precharge_transistors(self):
        if self.w_rc:                                                       #考虑是否添加rc网络
            bl_node = self.add_rc_networks_to_node('BL', 2)
            blb_node = self.add_rc_networks_to_node('BLB', 2)
            enb_node = self.add_rc_networks_to_node('ENB', 1)
        else:
            bl_node = 'BL'
            blb_node = 'BLB'
            enb_node = 'ENB'
        if not self.sweep_precharge:   
            # PMOS transistors to precharge BL and BLB to VDD
            self.M(1, bl_node,  enb_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(2, blb_node, enb_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            # Equalization transistor to reduce the difference between BL and BLB
            self.M(3, bl_node, enb_node, blb_node, 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
        else:
             # PMOS transistors to precharge BL and BLB to VDD
            self.M(1, bl_node,  enb_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[ int(self.mos_model_index['pmos'])],
                w='pmos_width_precharge', l='length_precharge')
            self.M(2, blb_node, enb_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[ int(self.mos_model_index['pmos'])],
                w='pmos_width_precharge', l='length_precharge')
            # Equalization transistor to reduce the difference between BL and BLB
            self.M(3, bl_node, enb_node, blb_node, 'VDD',
                model=self.pmos_modle_choices[ int(self.mos_model_index['pmos'])],
                w='pmos_width_precharge', l='length_precharge')

class WriteDriver(BaseSubcircuit):          #写驱动
    """
    Write driver circuit for SRAM with dynamically adjusted strength.
    同样需要根据行数动态调整晶体管宽度
    """
    NAME = "WRITEDRIVER"
    # VDD, GND, ENable, Data In, BL, BLB, 
    NODES = ('VDD', 'VSS', 'EN', 'DIN', 'BL', 'BLB')  

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_nmos_width=0.18e-6, base_pmos_width=0.36e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 num_rows=16,sweep_writedriver=False,
                 pmos_modle_choices='PMOS_VTG',nmos_modle_choices='NMOS_VTG'
                 ):
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc, pi_res, pi_cap,
        )
        self.num_rows = num_rows
        self.sweep_writedriver= sweep_writedriver
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices

        self.nmos_width = self.calculate_dynamic_width(base_nmos_width, num_rows)
        self.pmos_width = self.calculate_dynamic_width(base_pmos_width, num_rows)

        self.REQUIRED_COLUMNS = ['pmos_model_writedriver', 'nmos_model_writedriver']
        # 读取参数文件中的模型名
        self.mos_model_index = self.read_mos_model_from_param_file(self.REQUIRED_COLUMNS)
        self.add_driver_transistors()#添加写驱动晶体管函数

    # def read_mos_model_from_param_file(self):
    #     """
    #     从 param_sweep_PRECHARGE.txt 中读取 mos_model 的值
    #     """
    #     try:
    #         with open(self.param_file, 'r') as f:
    #             lines = f.readlines()
    #             # 第一行为标题行，第二行为数据行
    #             if len(lines) >= 2:
    #                 header = lines[0].strip().split()
    #                 values = lines[1].strip().split()
    #                 models = {}
    #                 for key in ['pmos_model_writedriver', 'nmos_model_writedriver']:
    #                     if key not in header:
    #                         raise ValueError(f"Missing required column: {key}")
    #                     index = header.index(key)
    #                     models[key.split('_')[0]] = values[index]  # 保留 pmos/nmos作为键
    #                 return models
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"Parameter file '{self.param_file}' not found.")
    #     except Exception as e:
    #         raise ValueError(f"Error parsing parameter file: {e}")
    #     raise ValueError("Could not find 'pmos_model_precharge' in parameter file.")

    def calculate_dynamic_width(self, base_width, num_rows):#动态调整width函数
        """
        Dynamically adjust the transistor width based on the number of rows.
        This is a simple linear scaling; you might need a more complex function.
        """
        #  scaling_factor = 1 + (num_rows - 1) * 0.1  # Example: 10% increase per additional row
        num_rows = 8 if num_rows < 8 else num_rows
        scaling_factor = num_rows / 16
        return base_width * scaling_factor

    def add_driver_transistors(self):
        if self.w_rc:                                           #考虑是否添加rc网络
            d_node = self.add_rc_networks_to_node('DIN', 1)
            db_node = self.add_rc_networks_to_node('DINB', 1)
            bl_node = self.add_rc_networks_to_node('BL', 2)
            blb_node = self.add_rc_networks_to_node('BLB', 2)
            en_node = self.add_rc_networks_to_node('EN', 1)
            enb_node = self.add_rc_networks_to_node('ENB', 1)

        else:
            d_node = 'DIN'
            db_node = 'DINB'
            bl_node = 'BL'
            blb_node = 'BLB'
            en_node = 'EN'
            enb_node = 'ENB'
        if not self.sweep_writedriver:
        # Inverters for enable and data input
            self.M(1, 'DINB', d_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(2, 'DINB', d_node, 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)
            self.M(3, 'ENB', en_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(4, 'ENB', en_node, 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)

            # Tristate for BL
            self.M(5, 'int1', db_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(6, bl_node, enb_node, 'int1', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(7, bl_node, en_node, 'int2', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)
            self.M(8, 'int2', db_node, 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)

            # Tristate for BLB
            self.M(9, 'int3', d_node, 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            self.M(10, blb_node, enb_node, 'int3', 'VDD',
                    model=self.pmos_pdk_model,
                    w=self.pmos_width, l=self.length)
            self.M(11, blb_node, en_node, 'int4', 'VSS',
                    model=self.nmos_pdk_model,
                    w=self.nmos_width, l=self.length)
            self.M(12, 'int4', d_node, 'VSS', 'VSS',
                    model=self.nmos_pdk_model,
                    w=self.nmos_width, l=self.length)
        else:
            self.M(1, 'DINB', d_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_wrd', l='length_wrd')
            self.M(2, 'DINB', d_node, 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_wrd', l='length_wrd')
            self.M(3, 'ENB', en_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_wrd', l='length_wrd')
            self.M(4, 'ENB', en_node, 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_wrd', l='length_wrd')

            # Tristate for BL
            self.M(5, 'int1', db_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_wrd', l='length_wrd')
            self.M(6, bl_node, enb_node, 'int1', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_wrd', l='length_wrd')
            self.M(7, bl_node, en_node, 'int2', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_wrd', l='length_wrd')
            self.M(8, 'int2', db_node, 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_wrd', l='length_wrd')

            # Tristate for BLB
            self.M(9, 'int3', d_node, 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_wrd', l='length_wrd')
            self.M(10, blb_node, enb_node, 'int3', 'VDD',
                    model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                    w='pmos_width_wrd', l='length_wrd')
            self.M(11, blb_node, en_node, 'int4', 'VSS',
                    model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                    w='nmos_width_wrd', l='length_wrd')
            self.M(12, 'int4', d_node, 'VSS', 'VSS',
                    model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                    w='nmos_width_wrd', l='length_wrd')
