from PySpice.Unit import u_Ohm, u_pF
from dataclasses import replace
from .base_subcircuit import BaseSubcircuit
from math import ceil, log2
from .standard_cell import Pinv,AND3,AND2 # type: ignore
from sram_compiler.interconnect import add_tapped_line, resolve_interconnect


class DECODER3_8(BaseSubcircuit):  # 38译码器+使能EN端

    NAME = "DECODER3_8"
    NODES = ('VDD', 'VSS','EN', 'A0', 'A1', 'A2', 'WL0', 'WL1', 'WL2', 'WL3', 'WL4', 'WL5', 'WL6', 'WL7')

    def __init__(self, nmos_model_inv, pmos_model_inv, nmos_model_nand, pmos_model_nand,
                 # Base widths for NAND gate transistors
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 output_scale=1.0,
                 interconnect=None, row_span=8,
                 ):

        self.interconnect = resolve_interconnect(interconnect)
        self.row_span = row_span
        if output_scale != 1.0:
            self.NAME = "DECODER3_8_OUTPUT_SCALED"
        # Different cascade levels have different physical spans, so they must
        # not overwrite one another's subcircuit definitions in PySpice.
        if row_span != 8:
            self.NAME += f'_ROWS{float(row_span).hex().replace(".", "_").replace("+", "p").replace("-", "m")}'
        super().__init__(
            nmos_model_inv, pmos_model_inv,
            nand_nmos_width, nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        # 创建三个非门用于生成输入的反相
        self.inv_A0 = Pinv(
            nmos_model=nmos_model_inv,
            pmos_model=pmos_model_inv,
            pmos_width=inv_pmos_width,
            nmos_width=inv_nmos_width,
            length=length,
        )
        self.subcircuit(self.inv_A0)

        self.inv_A1 = Pinv(
            nmos_model=nmos_model_inv,
            pmos_model=pmos_model_inv,
            pmos_width=inv_pmos_width,
            nmos_width=inv_nmos_width,
            length=length,
        )
        self.subcircuit(self.inv_A1)

        self.inv_A2 = Pinv(
            nmos_model=nmos_model_inv,
            pmos_model=pmos_model_inv,
            pmos_width=inv_pmos_width,
            nmos_width=inv_nmos_width,
            length=length,
        )
        self.subcircuit(self.inv_A2)    


        # 创建八个三输入与门
        self.and_gates = []
        for _ in range(8):
            and_gate = AND3(
                nmos_model_nand=nmos_model_nand,
                pmos_model_nand=pmos_model_nand,
                nmos_model_inv=nmos_model_inv,
                pmos_model_inv=pmos_model_inv,
                nand_pmos_width=nand_pmos_width,
                nand_nmos_width=nand_nmos_width,
                inv_pmos_width=inv_pmos_width,
                inv_nmos_width=inv_nmos_width,
                length=length,
                w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
            )
            self.subcircuit(and_gate)
            self.and_gates.append(and_gate)
        #创建八个用于使能的与门
        self.and_for_en = []
        def scaled(width):
            if output_scale == 1.0:
                return width
            return f'{{{width}*{output_scale}}}' if isinstance(width, str) else width * output_scale
        for _ in range(8):
            and_for_en = AND2(
                nmos_model_nand=nmos_model_nand,
                pmos_model_nand=pmos_model_nand,
                nmos_model_inv=nmos_model_inv,
                pmos_model_inv=pmos_model_inv,
                nand_pmos_width=nand_pmos_width,
                nand_nmos_width=nand_nmos_width,
                inv_pmos_width=scaled(inv_pmos_width),
                inv_nmos_width=scaled(inv_nmos_width),
                length=length,
                w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
            )
            self.subcircuit(and_for_en)
            self.and_for_en.append(and_for_en)

        self.add_decoder_components()

    def add_decoder_components(self):
        # Place all eight logic outputs across this block's share of the row
        # height, including unused outputs in small/partial decoders. These
        # are geometric routing assumptions, not extracted decoder metal.
        wire = replace(self.interconnect.bl,
                       pitch_m=self.interconnect.bl.pitch_m * self.row_span / 8)
        taps = {net: add_tapped_line(self, f'{net}_line', net, 8, wire)
                for net in ('A0', 'A1', 'A2', 'A0b', 'A1b', 'A2b', 'EN')}
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
                   taps[inputs[0]][i],  # 输入A
                   taps[inputs[1]][i],  # 输入B
                   taps[inputs[2]][i],  # 输入C
                   f'WL{i}_pre'  # 输出
                   )
            self.X(f"AND_EN{i}", self.and_for_en[i].name,
                   'VDD', 'VSS',
                   f'WL{i}_pre',  # 输入A
                   taps['EN'][i],  # 输入B
                   f'WL{i}'  # 输出
                   )


class DECODER_CASCADE(BaseSubcircuit):
    """级联行地址译码器，支持任意行数"""
    NAME = "DECODER_CASCADE"
    
    def __init__(self, nmos_model_inv, pmos_model_inv, nmos_model_nand, pmos_model_nand, num_rows=16,
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6,
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 output_scale=1.0,
                 interconnect=None,
                 ):
        # 计算地址位数和级数
        self.w_rc=w_rc
        self.num_rows = num_rows
        self.interconnect = resolve_interconnect(interconnect)
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
            nmos_model_inv, pmos_model_inv,
            nand_nmos_width, nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        # 存储每级译码器和输出节点
        self.decoders_by_level = []
        self.level_output_nodes = [[] for _ in range(self.n_levels)]

        # Each level occupies the row height, split uniformly among its
        # decoder blocks. Address trunks place one centered tap per block;
        # local true/complement/enable lines distribute to its eight gates.
        address_taps = {}
        for bit in range(self.n_bits):
            level = self.n_levels - 1 - bit // 3
            groups = self.level_groups[level]
            wire = replace(self.interconnect.bl,
                           pitch_m=self.interconnect.bl.pitch_m * num_rows / groups)
            address_taps[bit] = add_tapped_line(self, f'A{bit}_line', f'A{bit}', groups, wire)

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
                        address_nodes.append(address_taps[bit_idx][decoder_idx])
                    else:
                        address_nodes.append('VSS')  # 地址不足时接地
                # 创建译码器实例
                decoder = DECODER3_8(
                    nmos_model_inv=nmos_model_inv,
                    pmos_model_inv=pmos_model_inv,
                    nmos_model_nand=nmos_model_nand,
                    pmos_model_nand=pmos_model_nand,
                    inv_pmos_width=inv_pmos_width,
                    inv_nmos_width=inv_nmos_width,
                    nand_pmos_width=nand_pmos_width,
                    nand_nmos_width=nand_nmos_width,
                    length=length,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    output_scale=output_scale if level == self.n_levels - 1 else 1.0,
                    interconnect=self.interconnect,
                    row_span=num_rows / self.level_groups[level],
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
