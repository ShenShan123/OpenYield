from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from .base_subcircuit import BaseSubcircuit
from numpy.f2py.crackfortran import endifs
import math
from math import ceil, log2

class Pinv(BaseSubcircuit):  # 非门
    """
    CMOS Inverter based on sram_16x4_pinv netlist.
    """
    NAME = "PINV"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.09e-6, length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'MOS_VTG',param_file="sim/param_sweep_model_name.txt"
                 ):
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.pmos_width = base_pmos_width
        self.nmos_width = base_nmos_width
        self.sweep_decoder=sweep_decoder
        self.param_file = param_file
        self.mos_model_index = self.read_mos_model_from_param_file()
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        self.add_inverter_transistors()  # 添加反相器单元函数
    
    def read_mos_model_from_param_file(self):
        """
        从 param_sweep_PRECHARGE.txt 中读取 mos_model 的值
        """
        try:
            with open(self.param_file, 'r') as f:
                lines = f.readlines()
                # 假设第一行为标题行，第二行为数据行
                if len(lines) >= 2:
                    header = lines[0].strip().split()
                    values = lines[1].strip().split()
                    models = {}
                    for key in ['pmos_model_decoder_invp', 'nmos_model_decoder_invn']:
                        if key not in header:
                            raise ValueError(f"Missing required column: {key}")
                        index = header.index(key)
                        models[key.split('_')[0]] = values[index]  # 保留 pmos/nmos作为键
                    return models
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file '{self.param_file}' not found.")
        except Exception as e:
            raise ValueError(f"Error parsing parameter file: {e}")
        raise ValueError("Could not find 'pmos_model_precharge' in parameter file.")

    def add_inverter_transistors(self):  # 反相器即一个pmos+一个nmos
        # Mpinv_pmos Z A vdd vdd pmos_vtg m=1 w=0.27u l=0.05u
        if not self.sweep_decoder:
            self.M(f'pinv_pmos', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
            # Mpinv_nmos Z A gnd gnd nmos_vtg m=1 w=0.09u l=0.05u
            self.M(f'pinv_nmos', 'Z', 'A', 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)
        else:
            self.M(f'pinv_pmos', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_invp', l='length_decoder')
            # Mpinv_nmos Z A gnd gnd nmos_vtg m=1 w=0.09u l=0.05u
            self.M(f'pinv_nmos', 'Z', 'A', 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_invn', l='length_decoder')


class PNAND3(BaseSubcircuit):  # 3输入与非门
    """
    CMOS NAND3
    """
    NAME = "PNAND3"
    NODES = ('VDD', 'VSS', 'A', 'B', 'C', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.18e-6, length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'MOS_VTG',param_file="sim/param_sweep_model_name.txt"
                 ):
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.sweep_decoder=sweep_decoder
        self.param_file = param_file
        self.mos_model_index = self.read_mos_model_from_param_file()
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        self.add_nand3_transistors()

    def read_mos_model_from_param_file(self):
        """
        从 param_sweep_PRECHARGE.txt 中读取 mos_model 的值
        """
        try:
            with open(self.param_file, 'r') as f:
                lines = f.readlines()
                # 假设第一行为标题行，第二行为数据行
                if len(lines) >= 2:
                    header = lines[0].strip().split()
                    values = lines[1].strip().split()
                    models = {}
                    for key in ['pmos_model_decoder_nandp', 'nmos_model_decoder_nandn']:
                        if key not in header:
                            raise ValueError(f"Missing required column: {key}")
                        index = header.index(key)
                        models[key.split('_')[0]] = values[index]  # 保留 pmos/nmos作为键
                    return models
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file '{self.param_file}' not found.")
        except Exception as e:
            raise ValueError(f"Error parsing parameter file: {e}")
        raise ValueError("Could not find 'pmos_model_precharge' in parameter file.")

    def add_nand3_transistors(self):
        if not self.sweep_decoder:
            self.M(f'pnand3_pmos1', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.base_pmos_width, l=self.length)
            self.M(f'pnand3_pmos2', 'Z', 'B', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.base_pmos_width, l=self.length)
            self.M(f'pnand3_pmos3', 'Z', 'C', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.base_pmos_width, l=self.length)

            self.M(f'pnand3_nmos1', 'Z', 'A', 'net1_nand', 'VSS',
                model=self.nmos_pdk_model,
                w=self.base_nmos_width, l=self.length)
            self.M(f'pnand3_nmos2', 'net1_nand', 'B', 'net2_nand', 'VSS',
                model=self.nmos_pdk_model,
                w=self.base_nmos_width, l=self.length)
            self.M(f'pnand3_nmos3', 'net2_nand', 'C', 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.base_nmos_width, l=self.length)
        else:
            self.M(f'pnand3_pmos1', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_nandp', l='length_decoder')
            self.M(f'pnand3_pmos2', 'Z', 'B', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_nandp', l='length_decoder')
            self.M(f'pnand3_pmos3', 'Z', 'C', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_nandp', l='length_decoder')

            self.M(f'pnand3_nmos1', 'Z', 'A', 'net1_nand', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_nandn', l='length_decoder')
            self.M(f'pnand3_nmos2', 'net1_nand', 'B', 'net2_nand', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_nandn', l='length_decoder')
            self.M(f'pnand3_nmos3', 'net2_nand', 'C', 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_nandn', l='length_decoder')


class PNAND2(BaseSubcircuit):  # 单个2输入与非门
    """
    CMOS NAND2 gate based on sram_16x4_pnand2 netlist in OpenRAM.
    Widths can be dynamically scaled based on num_cols.
    不需要晶体管宽度根据列数调整，因为不起驱动作用
    """
    NAME = "PNAND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.18e-6, length=0.05e-6,
                 num_cols=4,  # Number of columns in the SRAM array configuration
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'MOS_VTG',param_file="sim/param_sweep_model_name.txt"
                 ):
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.sweep_decoder=sweep_decoder
        self.param_file = param_file
        self.mos_model_index = self.read_mos_model_from_param_file()
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        self.add_nand2_transistors()

    def read_mos_model_from_param_file(self):
        """
        从 param_sweep_PRECHARGE.txt 中读取 mos_model 的值
        """
        try:
            with open(self.param_file, 'r') as f:
                lines = f.readlines()
                # 假设第一行为标题行，第二行为数据行
                if len(lines) >= 2:
                    header = lines[0].strip().split()
                    values = lines[1].strip().split()
                    models = {}
                    for key in ['pmos_model_decoder_nandp', 'nmos_model_decoder_nandn']:
                        if key not in header:
                            raise ValueError(f"Missing required column: {key}")
                        index = header.index(key)
                        models[key.split('_')[0]] = values[index]  # 保留 pmos/nmos作为键
                    return models
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file '{self.param_file}' not found.")
        except Exception as e:
            raise ValueError(f"Error parsing parameter file: {e}")
        raise ValueError("Could not find 'pmos_model_precharge' in parameter file.")

    def add_nand2_transistors(self):
        if not self.sweep_decoder:
            self.M('pnand2_pmos1', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.base_pmos_width, l=self.length)
            self.M('pnand2_pmos2', 'Z', 'B', 'VDD', 'VDD',
                model=self.pmos_pdk_model,
                w=self.base_pmos_width, l=self.length)

            self.M('pnand2_nmos1', 'Z', 'B', 'net1_nand', 'VSS',
                model=self.nmos_pdk_model,
                w=self.base_nmos_width, l=self.length)
            self.M('pnand2_nmos2', 'net1_nand', 'A', 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.base_nmos_width, l=self.length)
        else:
            self.M('pnand2_pmos1', 'Z', 'A', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_nandp', l='length_decoder')
            self.M('pnand2_pmos2', 'Z', 'B', 'VDD', 'VDD',
                model=self.pmos_modle_choices[int(self.mos_model_index['pmos'])],
                w='pmos_width_decoder_nandp', l='length_decoder')

            self.M('pnand2_nmos1', 'Z', 'B', 'net1_nand', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_nandn', l='length_decoder')
            self.M('pnand2_nmos2', 'net1_nand', 'A', 'VSS', 'VSS',
                model=self.nmos_modle_choices[int(self.mos_model_index['nmos'])],
                w='nmos_width_decoder_nandn', l='length_decoder')

class AND3(BaseSubcircuit):  # ////////三输入与门=与非门+非门

    NAME = "AND3"
    NODES = ('VDD', 'VSS', 'A', 'B', 'C', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 # Base widths for NAND gate transistors
                 base_nand_pmos_width=0.27e-6, base_nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 base_inv_pmos_width=0.27e-6, base_inv_nmos_width=0.09e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'NMOS_VTG'
                 ):

        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.sweep_decoder=sweep_decoder
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        # This is the nand gate
        self.nand_gate = PNAND3(nmos_model_name=nmos_model_name,
                                pmos_model_name=pmos_model_name,
                                base_pmos_width=base_nand_pmos_width,
                                base_nmos_width=base_nand_nmos_width,
                                length=length,sweep_decoder=self.sweep_decoder,
                                pmos_modle_choices = self.pmos_modle_choices,
                                nmos_modle_choices = self.nmos_modle_choices
                                )
        self.subcircuit(self.nand_gate)  # 添加与非门电路

        # This is the inverter for driving WLs
        self.inv_driver = Pinv(nmos_model_name=nmos_model_name,
                                        pmos_model_name=pmos_model_name,
                                        base_pmos_width=base_inv_pmos_width,
                                        base_nmos_width=base_inv_nmos_width,
                                        length=length,sweep_decoder=self.sweep_decoder,
                                        pmos_modle_choices = self.pmos_modle_choices,
                                        nmos_modle_choices = self.nmos_modle_choices
                                        )
        self.subcircuit(self.inv_driver)  # 添加反相器电路

        self.add_and3_components()

    def add_and3_components(self):
        if self.w_rc:  # 字线要考虑是否添加rc网络，
            a_node = self.add_rc_networks_to_node('A', num_segs=2)  # 调用base里的rc网络函数
            b_node = self.add_rc_networks_to_node('B', num_segs=2)  # 4条线每条分成两段加rc
            c_node = self.add_rc_networks_to_node('C', num_segs=2)
            zb_node = self.add_rc_networks_to_node('zb_int', num_segs=2)
            z_node = self.add_rc_networks_to_node('Z', num_segs=2)
        else:
            a_node = 'A'
            b_node = "B"
            c_node = "C"
            zb_node = "zb_int"
            z_node = "Z"

        """ Instantiate the `PNAND3` and `Pinv` gates """  # 实例化
        self.X(f'PNAND3', self.nand_gate.name,
               'VDD', 'VSS', a_node, b_node, c_node, zb_node)
        self.X(f'PINV', self.inv_driver.name,
               'VDD', 'VSS', zb_node, z_node)

class AND2(BaseSubcircuit):  # ////////二输入与门=二输入与非门+非门

    NAME = "AND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 # Base widths for NAND gate transistors
                 base_nand_pmos_width=0.27e-6, base_nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 base_inv_pmos_width=0.27e-6, base_inv_nmos_width=0.09e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'NMOS_VTG'
                 ):

        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.sweep_decoder=sweep_decoder
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        # This is the nand gate
        self.nand_gate = PNAND2(nmos_model_name=nmos_model_name,
                                pmos_model_name=pmos_model_name,
                                base_pmos_width=base_nand_pmos_width,
                                base_nmos_width=base_nand_nmos_width,
                                length=length,sweep_decoder=self.sweep_decoder,
                                pmos_modle_choices = self.pmos_modle_choices,
                                nmos_modle_choices = self.nmos_modle_choices
                                )
        self.subcircuit(self.nand_gate)  # 添加与非门电路

        # This is the inverter for driving WLs
        self.inv_driver = Pinv(nmos_model_name=nmos_model_name,
                                        pmos_model_name=pmos_model_name,
                                        base_pmos_width=base_inv_pmos_width,
                                        base_nmos_width=base_inv_nmos_width,
                                        length=length,sweep_decoder=self.sweep_decoder,
                                        pmos_modle_choices = self.pmos_modle_choices,
                                        nmos_modle_choices = self.nmos_modle_choices
                                        )
        self.subcircuit(self.inv_driver)  # 添加反相器电路

        self.add_and3_components()

    def add_and3_components(self):
        if self.w_rc:  # 字线要考虑是否添加rc网络，
            a_node = self.add_rc_networks_to_node('A', num_segs=2)  # 调用base里的rc网络函数
            b_node = self.add_rc_networks_to_node('B', num_segs=2)  # 4条线每条分成两段加rc
            zb_node = self.add_rc_networks_to_node('zb_int', num_segs=2)
            z_node = self.add_rc_networks_to_node('Z', num_segs=2)
        else:
            a_node = 'A'
            b_node = "B"
            zb_node = "zb_int"
            z_node = "Z"

        """ Instantiate the `PNAND3` and `Pinv` gates """  # 实例化
        self.X(f'PNAND3', self.nand_gate.name,
               'VDD', 'VSS', a_node, b_node, zb_node)
        self.X(f'PINV', self.inv_driver.name,
               'VDD', 'VSS', zb_node, z_node)


class DECODER3_8(BaseSubcircuit):  # 38译码器+使能EN端

    NAME = "DECODER3_8"
    NODES = ('VDD', 'VSS','EN', 'A0', 'A1', 'A2', 'WL0', 'WL1', 'WL2', 'WL3', 'WL4', 'WL5', 'WL6', 'WL7')

    def __init__(self, nmos_model_name, pmos_model_name,
                 # Base widths for NAND gate transistors
                 base_nand_pmos_width=0.27e-6, base_nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 base_inv_pmos_width=0.27e-6, base_inv_nmos_width=0.09e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                 pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'NMOS_VTG'
                 ):

        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.sweep_decoder=sweep_decoder
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        # 创建三个非门用于生成输入的反相
        self.inv_A0 = Pinv(
            nmos_model_name=nmos_model_name,
            pmos_model_name=pmos_model_name,
            base_pmos_width=base_inv_pmos_width,
            base_nmos_width=base_inv_nmos_width,
            length=length,
            sweep_decoder=self.sweep_decoder,
            pmos_modle_choices = self.pmos_modle_choices,
            nmos_modle_choices = self.nmos_modle_choices
        )
        self.subcircuit(self.inv_A0)

        self.inv_A1 = Pinv(
            nmos_model_name=nmos_model_name,
            pmos_model_name=pmos_model_name,
            base_pmos_width=base_inv_pmos_width,
            base_nmos_width=base_inv_nmos_width,
            length=length,
            sweep_decoder=self.sweep_decoder,
            pmos_modle_choices = self.pmos_modle_choices,
            nmos_modle_choices = self.nmos_modle_choices
        )
        self.subcircuit(self.inv_A1)

        self.inv_A2 = Pinv(
            nmos_model_name=nmos_model_name,
            pmos_model_name=pmos_model_name,
            base_pmos_width=base_inv_pmos_width,
            base_nmos_width=base_inv_nmos_width,
            length=length,
            sweep_decoder=self.sweep_decoder,
            pmos_modle_choices = self.pmos_modle_choices,
            nmos_modle_choices = self.nmos_modle_choices
        )
        self.subcircuit(self.inv_A2)

        # 创建八个三输入与门
        self.and_gates = []
        for _ in range(8):
            and_gate = AND3(
                nmos_model_name=nmos_model_name,
                pmos_model_name=pmos_model_name,
                base_nand_pmos_width=base_nand_pmos_width,
                base_nand_nmos_width=base_nand_nmos_width,
                base_inv_pmos_width=base_inv_pmos_width,
                base_inv_nmos_width=base_inv_nmos_width,
                length=length,
                w_rc=w_rc,
                pi_res=pi_res,
                pi_cap=pi_cap,
                sweep_decoder=self.sweep_decoder,
                pmos_modle_choices = self.pmos_modle_choices,
                nmos_modle_choices = self.nmos_modle_choices
            )
            self.subcircuit(and_gate)
            self.and_gates.append(and_gate)
        #创建八个用于使能的与门
        self.and_for_en = []
        for _ in range(8):
            and_for_en = AND2(
                nmos_model_name=nmos_model_name,
                pmos_model_name=pmos_model_name,
                base_nand_pmos_width=base_nand_pmos_width,
                base_nand_nmos_width=base_nand_nmos_width,
                base_inv_pmos_width=base_inv_pmos_width,
                base_inv_nmos_width=base_inv_nmos_width,
                length=length,
                w_rc=w_rc,
                pi_res=pi_res,
                pi_cap=pi_cap,
                sweep_decoder=self.sweep_decoder,
                pmos_modle_choices = self.pmos_modle_choices,
                nmos_modle_choices = self.nmos_modle_choices
            )
            self.subcircuit(and_for_en)
            self.and_for_en.append(and_for_en)

        self.add_decoder_components()

    def add_decoder_components(self):
        # 添加三个非门实例化
        self.X('INV_A1', self.inv_A0.name, 'VDD', 'VSS', 'A0', 'A0b')
        self.X('INV_A2', self.inv_A1.name, 'VDD', 'VSS', 'A1', 'A1b')
        self.X('INV_A3', self.inv_A2.name, 'VDD', 'VSS', 'A2', 'A2b')

        # 定义每个与门的输入组合（真值表）
        input_combinations = [
            ('A0b', 'A1b', 'A2b'),  # WL0: 000
            ('A0b', 'A1b', 'A2'),  # WL1: 001
            ('A0b', 'A1', 'A2b'),  # WL2: 010
            ('A0b', 'A1', 'A2'),  # WL3: 011
            ('A0', 'A1b', 'A2b'),  # WL4: 100
            ('A0', 'A1b', 'A2'),  # WL5: 101
            ('A0', 'A1', 'A2b'),  # WL6: 110
            ('A0', 'A1', 'A2')  # WL7: 111
        ]

        # 实例化八个与门和EN与门
        for i in range(8):
            inputs = input_combinations[i]
            self.X(f"AND{i}", self.and_gates[i].name,
                   'VDD', 'VSS',
                   inputs[0],  # 输入A
                   inputs[1],  # 输入B
                   inputs[2],  # 输入C
                   f'WL{i}_pre'  # 输出
                   )
            self.X(f"AND_EN{i}", self.and_for_en[i].name,
                   'VDD', 'VSS',
                   f'WL{i}_pre',  # 输入A
                   'EN',  # 输入B
                   f'WL{i}'  # 输出
                   )


class DECODER_CASCADE(BaseSubcircuit):
    """级联行地址译码器，支持任意行数"""
    NAME = "DECODER_CASCADE"
    
    def __init__(self, nmos_model_name, pmos_model_name, num_rows=16,
                 base_inv_pmos_width=0.27e-6, base_inv_nmos_width=0.09e-6,
                 base_nand_pmos_width=0.27e-6, base_nand_nmos_width=0.18e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,sweep_decoder=False,
                pmos_modle_choices = 'PMOS_VTG',nmos_modle_choices = 'NMOS_VTG'
                 ):
        # 计算地址位数和级数
        self.num_rows = num_rows
        self.sweep_decoder = sweep_decoder
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices
        self.n_bits = ceil(log2(num_rows)) if num_rows > 1 else 1
        self.n_levels = ceil(self.n_bits / 3.0)
        #NODES = ['VDD', 'VSS','EN'] + [f'A{i}'for i in range(self.n_bits)] + [f'WL{i}' for i in range(num_rows)]

        # 计算每级译码器数量（从最后一级开始）
        level_groups = [0] * self.n_levels
        level_groups[self.n_levels - 1] = ceil(num_rows / 8.0)
        for level in range(self.n_levels - 2, -1, -1):
            level_groups[level] = ceil(level_groups[level + 1] / 8.0)
        self.level_groups = level_groups

         #定义节点：VDD, VSS, 地址线, 字线
        nodes = ['VDD', 'VSS']
        nodes.extend([f'A{i}' for i in range(self.n_bits)])
        nodes.extend([f'WL{i}' for i in range(num_rows)])
        self.NODES = nodes
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        

        # 存储每级译码器和输出节点
        self.decoders_by_level = []
        self.level_output_nodes = [[] for _ in range(self.n_levels)]


        # 创建所有译码器
        for level in range(self.n_levels):
            level_decoders = []
            for decoder_idx in range(self.level_groups[level]):
                # 计算该级译码器使用的地址位
                address_nodes = []
                # 计算地址起始位置：从最高位开始
                start_bit = 3 * (self.n_levels-level-1)
                for bit in range(3):
                    bit_idx = start_bit + bit
                    if bit_idx >= 0 and bit_idx < self.n_bits:
                        address_nodes.append(f'A{bit_idx}')
                    else:
                        address_nodes.append('VSS')  # 地址不足时接地
                # 创建译码器实例
                decoder = DECODER3_8(
                    nmos_model_name=nmos_model_name,
                    pmos_model_name=pmos_model_name,
                    base_inv_pmos_width=base_inv_pmos_width,
                    base_inv_nmos_width=base_inv_nmos_width,
                    base_nand_pmos_width=base_nand_pmos_width,
                    base_nand_nmos_width=base_nand_nmos_width,
                    length=length,
                    w_rc=w_rc,
                    sweep_decoder=self.sweep_decoder,
                    pmos_modle_choices = self.pmos_modle_choices,
                    nmos_modle_choices = self.nmos_modle_choices
                )
                self.subcircuit(decoder)
                level_decoders.append(decoder)

                # 确定使能信号
                if level == 0:
                    enable_signal = 'VDD'
                else:
                    # 从上一级输出节点获取使能
                    if decoder_idx < len(self.level_output_nodes[level - 1]):
                        enable_signal = self.level_output_nodes[level - 1][decoder_idx]
                    else:
                        enable_signal = 'VSS'

                # 确定输出节点
                output_nodes = []
                for out_idx in range(8):
                    if level == self.n_levels - 1:  # 最后级输出字线
                        wl_idx = decoder_idx * 8 + out_idx
                        if wl_idx < num_rows:
                            node_name = f'WL{wl_idx}'
                        else:
                            node_name = f'NC_{level}_{decoder_idx}_{out_idx}'
                    else:  # 中间级输出使能信号
                        node_name = f'EN_{level}_{decoder_idx}_{out_idx}'
                    output_nodes.append(node_name)

                # 保存输出节点（用于下一级使能）
                self.level_output_nodes[level].extend(output_nodes)

                # 实例化译码器
                self.X(f'DEC_{level}_{decoder_idx}', decoder.NAME,
                       'VDD', 'VSS',
                       enable_signal,
                       address_nodes[2], address_nodes[1], address_nodes[0],  # A0, A1, A2
                       *output_nodes
                       )
            self.decoders_by_level.append(level_decoders)



if __name__ == '__main__':
    # 创建3-8译码器实例
    decoder = DECODER_CASCADE(
        nmos_model_name="NMOS_VTG",
        pmos_model_name="PMOS_VTG",
        num_rows=2,
        base_inv_pmos_width=0.27e-6,
        base_inv_nmos_width=0.09e-6,
        base_nand_pmos_width=0.27e-6,
        base_nand_nmos_width=0.18e-6,
        length=0.05e-6,
        w_rc=False
    )

    # 打印网表信息
    print("=" * 80)
    print("CASCADE DECODER NETLIST")
    print("=" * 80)
    print(decoder)
    print("=" * 80)

    # 打印结构信息
    print("\nDecoder Structure:")
    print(f"- Total levels: {decoder.n_levels}")
    for level, count in enumerate(decoder.level_groups):
        print(f"  Level {level}: {count} decoders")
    print(f"Total word lines: {decoder.num_rows}")
