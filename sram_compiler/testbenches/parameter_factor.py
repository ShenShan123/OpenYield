from sram_compiler.subcircuits.mux_and_sa import SenseAmp, ColumnMux
from sram_compiler.subcircuits.precharge_and_write_driver import Precharge,WriteDriver
from sram_compiler.subcircuits.wordline_driver import WordlineDriver
from sram_compiler.subcircuits.sram_6t_core import Sram6TCell, Sram6TCore
from sram_compiler.subcircuits.decoder import DECODER_CASCADE
from sram_compiler.subcircuits.dummy_row_or_column import Dummy_Row, Dummy_Column
from sram_compiler.subcircuits.replica_column import Replica_Column
from sram_compiler.subcircuits.time_generate import TIME
def read_mos_model_from_param_file(names):
    """
    Read MOS models from parameter file
    """
    param_file="sram_compiler/param_sweep_data/param_sweep_model_name.txt"
    try:
        with open(param_file, 'r') as f:
            lines = f.readlines()
            # Assume first line is header, second line is data
            if len(lines) >= 2:
                header = lines[0].strip().split()
                values = lines[1].strip().split()
                models = {}
                for key in names:
                    if key not in header:
                        raise ValueError(f"Missing required column: {key}")
                    index = header.index(key)
                    models[key.split('_')[0]] = values[index]  # Keep pmos/nmos as keys
                return models
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file '{param_file}' not found.")
    except Exception as e:
        raise ValueError(f"Error parsing parameter file: {e}")
    
class SenseAmpFactory:
    """
    Factory class to configure and create SenseAmp instances.
    """
    def __init__(self, 
                 nmos_model, pmos_model,
                 nmos_width, pmos_width, length,
                 w_rc=False, 
                 sweep_senseamp=False,
                 pmos_modle_choices=None, nmos_modle_choices=None,
                 ):
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        self.w_rc = w_rc
        self.sweep = sweep_senseamp

        self.pmos_choices = pmos_modle_choices
        self.nmos_choices = nmos_modle_choices
        self.REQUIRED_COLUMNS = ['pmos_model_senseamp', 'nmos_model_senseamp']
        self.mos_model_index = read_mos_model_from_param_file(self.REQUIRED_COLUMNS)
        
    def _get_config(self):
        """Based on whether the parameters are scanned, determine 
        the final dictionary to be transmitted to the circuit"""
        
        if self.sweep:
            # --- sweep model ---
            # Change to string variable name
            nmos_width = 'nmos_width_senseamp'
            pmos_width = 'pmos_width_senseamp'
            length = 'length_senseamp'
            nmos_model = self.nmos_choices[int(self.mos_model_index['nmos'])]
            pmos_model = self.pmos_choices[int(self.mos_model_index['pmos'])]
            
        else:
            # --- yaml model ---
            nmos_width = self.nmos_width
            pmos_width = self.pmos_width
            length = self.length    
            nmos_model = self.nmos_model
            pmos_model = self.pmos_model

        # Return the parameter dictionary
        return {
            'nmos_model': nmos_model,
            'pmos_model': pmos_model,
            'nmos_width': nmos_width,
            'pmos_width': pmos_width,
            'length': length,
            'w_rc': self.w_rc,
        }

    def create(self):
        """Generate and return the SenseAmp instance"""
        config = self._get_config()
        # pass the config dictionary to StandardSenseAmp
        return SenseAmp(**config)

class ColumnMuxFactory:
    """
    Factory for creating StandardColumnMux instances.
    Handles parameter sweeping configuration.
    """
    def __init__(self, 
                 num_in,
                 nmos_model, pmos_model,
                 nmos_width, pmos_width, 
                 length,
                 w_rc=False, 
                 sweep_columnmux=False,
                 use_external_selb=False,
                 pmos_modle_choices=None, nmos_modle_choices=None,
                 ):
        
        self.num_in = num_in
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length

        self.w_rc = w_rc
        self.sweep = sweep_columnmux
        self.use_external_selb = use_external_selb
        
        self.pmos_choices = pmos_modle_choices
        self.nmos_choices = nmos_modle_choices
        self.REQUIRED_COLUMNS = ['pmos_model_columnmux', 'nmos_model_columnmux']
        self.mos_model_index = read_mos_model_from_param_file(self.REQUIRED_COLUMNS)

    def _get_config(self):
        """The parameters ultimately transmitted to the circuit"""
        if self.sweep:
            # --- sweep model ---
            # Change to string variable name
            nmos_width = 'nmos_width_mux'
            pmos_width = 'pmos_width_mux'
            length = 'length_mux'
            nmos_model = self.nmos_choices[int(self.mos_model_index['nmos'])]
            pmos_model = self.pmos_choices[int(self.mos_model_index['pmos'])]
        else:
            # --- yaml model ---
            nmos_width = self.nmos_width
            pmos_width = self.pmos_width
            length = self.length    
            nmos_model = self.nmos_model
            pmos_model = self.pmos_model

        # Return the parameter dictionary
        return {
            'nmos_model': nmos_model,
            'pmos_model': pmos_model,
            'nmos_width': nmos_width,
            'pmos_width': pmos_width,
            'length': length,
            'w_rc': self.w_rc,
            'num_in': self.num_in,
            'use_external_selb': self.use_external_selb,
        }

    def create(self):
        config = self._get_config()
        
        return ColumnMux(**config)  
    
class PrechargeFactory:
    def __init__(self, 
                pmos_model, pmos_width=0.27e-6, length=50e-9, num_rows=16,
                w_rc=False, 
                sweep_precharge=False, 
                pmos_modle_choices=None):
        
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        self.length = length
        self.num_rows = num_rows
        self.w_rc = w_rc
        
        self.sweep = sweep_precharge
        self.pmos_choices = pmos_modle_choices
        self.REQUIRED_COLUMNS = ['pmos_model_precharge']
        self.mos_model_index = read_mos_model_from_param_file(self.REQUIRED_COLUMNS)

    def _calculate_dynamic_width(self):
        #The width of the transistor needs to 
        #be dynamically adjusted according to the number of rows.
        scaling_factor = self.num_rows / 16  
        scaling_factor = 0.5 if scaling_factor < 0.5 else scaling_factor
        return self.pmos_width * scaling_factor

    def _get_config(self):
        
        if self.sweep:
            # --- sweep model ---# Change to string variable name
            pmos_width = 'pmos_width_precharge'
            length = 'length_precharge'
            pmos_model = self.pmos_choices[int(self.mos_model_index['pmos'])]
        else:
            # --- yaml model ---
            pmos_width = self._calculate_dynamic_width()
            length = self.length
            pmos_model = self.pmos_model
            
        return {
            "pmos_model": pmos_model,
            "pmos_width": pmos_width,
            "length": length,
            "w_rc": self.w_rc,
            }
    def create(self):
        config = self._get_config()
        return Precharge(**config)
    
class WriteDriverFactory:
    def __init__(self, nmos_model, pmos_model,
                 nmos_width=0.18e-6, pmos_width=0.36e-6, 
                 length=50e-9, num_rows=16,
                 w_rc=False, 
                 sweep_writedriver=False,
                 pmos_modle_choices=None, nmos_modle_choices=None,
                ):
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        self.num_rows = num_rows
        
        self.w_rc = w_rc
        self.sweep = sweep_writedriver
        
        self.pmos_choices = pmos_modle_choices
        self.nmos_choices = nmos_modle_choices
        self.REQUIRED_COLUMNS = ['pmos_model_writedriver', 'nmos_model_writedriver']
        self.mos_model_index = read_mos_model_from_param_file(self.REQUIRED_COLUMNS)

    def _calculate_dynamic_width(self, base_width):
        """Dynamically adjust the transistor width based on the number of rows.
        This is a simple linear scaling; you might need a more complex function."""
        eff_rows = 8 if self.num_rows < 8 else self.num_rows
        scaling_factor = eff_rows / 16
        return base_width * scaling_factor

    def _get_config(self):
        if self.sweep:
            # --- sweep model ---
            nmos_width = 'nmos_width_wrd'
            pmos_width = 'pmos_width_wrd'
            length = 'length_wrd'
            nmos_model = self.nmos_choices[int(self.mos_model_index['nmos'])]
            pmos_model = self.pmos_choices[int(self.mos_model_index['pmos'])]
        else:
            # --- yaml model ---
            nmos_width = self._calculate_dynamic_width(self.nmos_width)
            pmos_width = self._calculate_dynamic_width(self.pmos_width)
            length = self.length
            nmos_model = self.nmos_model
            pmos_model = self.pmos_model

        return {
            "nmos_model": nmos_model,
            "pmos_model": pmos_model,
            "nmos_width": nmos_width,
            "pmos_width": pmos_width,
            "length": length,
            "w_rc": self.w_rc,
        }
    def create(self):
        config = self._get_config()
        return WriteDriver(**config)
    
class WordlineDriverFactory:
    """
    Factory for creating WordlineDriver instances.
    Responsible for:
      - Calculating dynamic widths based on load (num_cols).
      - Configuring parameters for sweep vs fixed mode.
      - Instantiating the WordlineDriver with the correct sub-blocks (PNAND2, Pinv).
    """
    def __init__(self, nmos_model, pmos_model,
                 # NAND2 base widths (usually fixed or less sensitive)
                 nand_nmos_width=0.18e-6, nand_pmos_width=0.27e-6,
                 # Inverter base widths (sensitive to load)
                 inv_nmos_width=0.09e-6, inv_pmos_width=0.27e-6,
                 length=50e-9, 
                 num_cols=16,  # Load parameter
                 w_rc=False, 
                 sweep_wordlinedriver=False,
                 pmos_modle_choices=None, nmos_modle_choices=None,
                 ):
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        
        self.nand_nmos_width = nand_nmos_width
        self.nand_pmos_width = nand_pmos_width
        self.inv_base_nmos_width = inv_nmos_width
        self.inv_base_pmos_width = inv_pmos_width  
        self.length = length
        self.num_cols = num_cols
        
        self.w_rc = w_rc
        self.sweep = sweep_wordlinedriver
        self.pmos_choices = pmos_modle_choices
        self.nmos_choices = nmos_modle_choices
        self.REQUIRED_COLUMNS1 = ['pmos_model_wld_invp', 'nmos_model_wld_invn']
        self.mos_model_index1 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS1)
        self.REQUIRED_COLUMNS2 = ['pmos_model_wld_nandp', 'nmos_model_wld_nandn']
        self.mos_model_index2 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS2)

    def _calculate_inv_width(self):
        """
        Calculate dynamic width for the output inverter based on column load.
        """
        # Example scaling logic: Scale up if columns > 4
        scale = max(self.num_cols, 4) / 4.0
        # You might want to cap the scale or adjust the formula
        return (self.inv_base_nmos_width * scale, self.inv_base_pmos_width * scale)

    def _get_config(self):
        """
        Prepare the configuration dictionary for WordlineDriver.
        """
        if self.sweep:
            # --- Sweep Mode ---
            # Use string parameter names for SPICE netlist
            # NAND2 parameters
            nand_nmos_width = 'nmos_width_wld_nandn'
            nand_pmos_width = 'pmos_width_wld_nandp'
            inv_nmos_width = 'nmos_width_wld_invn'
            inv_pmos_width = 'pmos_width_wld_invp'
            length = 'length_wld'
            
            # Select model from choices
            nmos_modle_inv = self.nmos_choices[int(self.mos_model_index1['nmos'])]
            pmos_modle_inv = self.pmos_choices[int(self.mos_model_index1['pmos'])]
            nmos_modle_nand = self.nmos_choices[int(self.mos_model_index2['nmos'])]
            pmos_modle_nand = self.pmos_choices[int(self.mos_model_index2['pmos'])]
            
        else:
            # --- yaml Mode ---
            # Calculate numerical values
            nand_nmos_width = self.nand_nmos_width
            nand_pmos_width = self.nand_pmos_width
            
            # Dynamic sizing for Inverter
            inv_nmos_width, inv_pmos_width = self._calculate_inv_width()
            
            length = self.length
            nmos_modle_inv = self.nmos_model
            pmos_modle_inv = self.pmos_model
            nmos_modle_nand = self.nmos_model
            pmos_modle_nand = self.pmos_model

        return {
            'nmos_model_inv': nmos_modle_inv,
            'pmos_model_inv': pmos_modle_inv,
            'nmos_model_nand': nmos_modle_nand,
            'pmos_model_nand': pmos_modle_nand,
            'nand_nmos_width': nand_nmos_width,
            'nand_pmos_width': nand_pmos_width,
            'inv_nmos_width': inv_nmos_width,
            'inv_pmos_width': inv_pmos_width,
            'length': length,
            'w_rc': self.w_rc,
        }

    def create(self):
        """
        Instantiate the WordlineDriver with the configured parameters.
        """
        config = self._get_config()
        return WordlineDriver(**config)
    
class Sram6TCellFactory:
    def __init__(self, 
                 pd_model, pu_model, pg_model,
                 pd_width, pu_width, pg_width, length,
                 w_rc=False, disconnect=False,
                 # 模式控制
                 sweep=False,
                 yield_mode=False, # 用于控制是否传入 model_dict
                 # 附加数据
                 model_dict=None,  
                 suffix='',        
                 pmos_choices=None, nmos_choices=None,
                 ):
        
        self.pd_model = pd_model
        self.pu_model = pu_model
        self.pg_model = pg_model
        self.dims = {'pd': pd_width, 'pu': pu_width, 'pg': pg_width, 'l': length}
        
        self.w_rc = w_rc
        self.disconnect = disconnect
        self.sweep = sweep
        self.yield_mode = yield_mode
        self.model_dict = model_dict
        self.suffix = suffix
        
        self.pmos_choices = pmos_choices
        self.nmos_choices = nmos_choices
        self.REQUIRED_COLUMNS1 = ['pmos_model_pu', 'nmos_model_pd']
        self.mos_model_index1 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS1)
        self.REQUIRED_COLUMNS2 = ['nmos_model_pg']
        self.mos_model_index2 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS2)

    def _get_config(self):
        """给定 StandardSram6T 参数"""
        
        # 1. 确定宽度 (Widths)
        # 如果是 sweep 模式，传入字符串变量名；否则传入数值
        if self.sweep:
            pd_width = 'nmos_width_pd'
            pu_width = 'pmos_width_pu'
            pg_width = 'nmos_width_pg'
            length = 'length'
        else:
            pd_width = self.dims['pd']
            pu_width = self.dims['pu']
            pg_width = self.dims['pg']
            length = self.dims['l']

        # 2. 确定模型 (Models)
        # 如果是 sweep 模式，从 choices 中取模型名；否则用默认名
        if self.sweep:
            pd_model = self.nmos_choices[int(self.mos_model_index1['nmos'])]
            pg_model = self.nmos_choices[int(self.mos_model_index2['nmos'])] 
            pu_model = self.pmos_choices[int(self.mos_model_index1['pmos'])]
        else:
            pd_model = self.pd_model
            pg_model = self.pg_model
            pu_model = self.pu_model

        # 3. 确定是否启用 Yield 模式
        # 只有当开启 yield_mode 且提供了字典时，才传给电路类
        model_dict = self.model_dict if (self.yield_mode and self.model_dict) else None
        suffix = self.suffix if self.yield_mode else ''

        # 4. 实例化
        return  {
            'pd_model': pd_model,
            'pu_model': pu_model,
            'pg_model': pg_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'w_rc': self.w_rc,
            'disconnect': self.disconnect,
            'suffix': suffix,
            'model_dict': model_dict
        }
            
        
    def create(self):
        """
        Instantiate the WordlineDriver with the configured parameters.
        """
        config = self._get_config()
        return Sram6TCell(**config)
    
class Sram6TCoreFactory:
    """
    Sram6TCore 的工厂类。
    负责根据配置（Sweep 或 Fixed）准备参数，并实例化 Sram6TCore。
    """
    def __init__(self, 
                 num_rows: int, num_cols: int,
                 # 默认模型名 (Fixed 模式用)
                 pd_nmos_model: str, pu_pmos_model: str, pg_nmos_model: str,
                 # 默认尺寸 (Fixed 模式用)
                 pd_width=0.205e-6, pu_width=0.09e-6, pg_width=0.135e-6, length=50e-9,
                 w_rc=False,
                 # --- 控制参数 ---
                 sweep_core=False,          # 是否开启扫描模式
                 yield_mode=False,          # 是否开启良率/Mismatch模式
                 # --- Sweep 模式专用 ---
                 pmos_choices=None,         # PMOS 模型列表
                 nmos_choices=None,         # NMOS 模型列表
                 # --- Yield 模式专用 ---
                 model_dict=None            # Mismatch 参数字典
                 ):
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        # 保存基础参数
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        
        self.dims = {
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length
        }
        
        self.w_rc = w_rc
        
        # 保存控制参数
        self.sweep = sweep_core
        self.yield_mode = yield_mode
        self.model_dict = model_dict
        
        self.pmos_choices = pmos_choices
        self.nmos_choices = nmos_choices
        self.REQUIRED_COLUMNS1 = ['pmos_model_pu', 'nmos_model_pd']
        self.mos_model_index1 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS1)
        self.REQUIRED_COLUMNS2 = ['nmos_model_pg']
        self.mos_model_index2 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS2)

    def _get_config(self):
        """
        内部方法：计算最终传给 Sram6TCore 的参数字典。
        根据 self.sweep 决定是传数值还是传字符串变量名。
        """

        # 1. 处理尺寸 (Widths & Length)
        if self.sweep:
            # 扫描模式：使用 SPICE 变量名字符串
            pd_width = 'nmos_width_pd'
            pu_width = 'pmos_width_pu'
            pg_width = 'nmos_width_pg'
            length = 'length'
        else:
            # 固定模式：使用具体数值
            pd_width = self.dims['pd_width']
            pu_width = self.dims['pu_width']
            pg_width = self.dims['pg_width']
            length = self.dims['length']

        # 2. 处理模型 (Models)
        if self.sweep:
            # 扫描模式：从 choices 列表中根据 index 选取
            pd_model = self.nmos_choices[int(self.mos_model_index1['nmos'])]
            pg_model = self.nmos_choices[int(self.mos_model_index2['nmos'])]
            pu_model = self.pmos_choices[int(self.mos_model_index1['pmos'])]
        else:
            # 固定模式：使用初始化时传入的模型名
            pd_model = self.pd_nmos_model
            pg_model = self.pg_nmos_model
            pu_model = self.pu_pmos_model

        # 3. 处理良率/Mismatch (Yield)
        # 只有在 yield_mode=True 且 model_dict 存在时才传递字典
        # 否则传 None，对应 Sram6TCore 里的默认行为
        if self.yield_mode and self.model_dict:
            model_dict = self.model_dict
        else:
            model_dict = None

        return {
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'pd_nmos_model': pd_model,
            'pu_pmos_model': pu_model,
            'pg_nmos_model': pg_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'w_rc': self.w_rc,
            'model_dict': model_dict
        }

    def create(self):
        """
        创建并返回配置好的 Sram6TCore 实例
        """
        config = self._get_config()
        # 将准备好的参数解包传入 Sram6TCore 的构造函数
        return Sram6TCore(**config)
    
class DecoderCascadeFactory:
    """
    DECODER_CASCADE 的工厂类。
    负责准备参数（支持参数扫描或固定值）。
    """
    def __init__(self, 
                 nmos_model_inv, pmos_model_inv, nmos_model_nand, pmos_model_nand,
                 num_rows=16,
                 # 默认尺寸 (Fixed/YAML 模式用)
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6,
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 length=0.05e-6,
                 w_rc=False,
                 # --- 控制参数 ---
                 sweep_decoder=False,       # 是否开启扫描模式
                 # --- Sweep 模式专用 ---
                 pmos_choices=None,         # PMOS 模型列表
                 nmos_choices=None,         # NMOS 模型列表
                 ):
        
        self.num_rows = num_rows
        self.nmos_model_inv = nmos_model_inv
        self.pmos_model_inv = pmos_model_inv
        self.nmos_model_nand = nmos_model_nand
        self.pmos_model_nand = pmos_model_nand
        
        # 保存尺寸参数
        self.dims = {
            'inv_pmos_width': inv_pmos_width,
            'inv_nmos_width': inv_nmos_width,
            'nand_pmos_width': nand_pmos_width,
            'nand_nmos_width': nand_nmos_width,
            'length': length
        }
        
        # 寄生参数
        self.w_rc = w_rc
        
        # Sweep 控制
        self.sweep = sweep_decoder
        self.pmos_model_choices = pmos_choices
        self.nmos_model_choices = nmos_choices
        self.REQUIRED_COLUMNS1 = ['pmos_model_decoder_invp', 'nmos_model_decoder_invn']
        self.mos_model_index1 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS1)
        self.REQUIRED_COLUMNS2 = ['pmos_model_decoder_nandp', 'nmos_model_decoder_nandn']
        self.mos_model_index2 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS2)

    def _get_config(self):
        """
        内部方法：根据模式生成最终的配置字典
        """

        # 1. 处理尺寸 (Widths & Length)
        if self.sweep:
            # --- 扫描模式 ---
            # 使用 SPICE 变量名字符串
            # 区分反相器(Inv)和与非门(Nand)的参数名
            inv_nmos_width = 'nmos_width_decoder_invn'
            inv_pmos_width = 'pmos_width_decoder_invp'
            nand_nmos_width = 'nmos_width_decoder_nandn'
            nand_pmos_width = 'pmos_width_decoder_nandp'
            length = 'length_decoder'
        else:
            # --- 固定/YAML 模式 ---
            # 使用构造函数传入的数值
            inv_nmos_width = self.dims['inv_nmos_width']
            inv_pmos_width = self.dims['inv_pmos_width']
            nand_nmos_width = self.dims['nand_nmos_width']
            nand_pmos_width = self.dims['nand_pmos_width']
            length = self.dims['length']

        # 2. 处理模型 (Models)
        if self.sweep:
            # 从 choices 列表中选取
            pmos_model_inv=self.pmos_model_choices[int(self.mos_model_index1['pmos'])]
            nmos_model_inv=self.nmos_model_choices[int(self.mos_model_index1['nmos'])]
            pmos_model_nand=self.pmos_model_choices[int(self.mos_model_index2['pmos'])]
            nmos_model_nand=self.nmos_model_choices[int(self.mos_model_index2['nmos'])]
        else:
            # 使用固定模型名
            nmos_model_inv = self.nmos_model_inv
            pmos_model_inv = self.pmos_model_inv
            nmos_model_nand = self.nmos_model_nand
            pmos_model_nand = self.pmos_model_nand

        return {
            'num_rows': self.num_rows,
            'inv_nmos_width': inv_nmos_width,
            'inv_pmos_width': inv_pmos_width,
            'nand_nmos_width': nand_nmos_width,
            'nand_pmos_width': nand_pmos_width,
            'length': length,
            'nmos_model_inv': nmos_model_inv,
            'pmos_model_inv': pmos_model_inv,
            'nmos_model_nand': nmos_model_nand,
            'pmos_model_nand': pmos_model_nand,
            'w_rc': self.w_rc
        }

    def create(self):
        """
        实例化并返回 DECODER_CASCADE 对象
        """
        config = self._get_config()
        return DECODER_CASCADE(**config)
            
class DummyColumnFactory:
    def __init__(self, num_rows,
                 pd_nmos_model, pu_pmos_model, pg_nmos_model,
                 pd_width=0.205e-6, pu_width=0.09e-6, pg_width=0.135e-6, length=50e-9,
                 w_rc=False, disconnect=False,
                 ):
        
        self.num_rows = num_rows
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        self.w_rc = w_rc
        self.disconnect = disconnect
        

    def _get_config(self):
        

        pd_width = self.pd_width
        pu_width = self.pu_width
        pg_width = self.pg_width
        length = self.length

        pd_model = self.pd_model
        pg_model = self.pg_model
        pu_model = self.pu_model
        
        return{
            'num_rows': self.num_rows,
            'pd_nmos_model': pd_model,
            'pu_pmos_model': pu_model,
            'pg_nmos_model': pg_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'w_rc': self.w_rc,
            'disconnect': self.disconnect,
        }
        
    def create(self):
        config = self._get_config()
        return Dummy_Column(**config)


class DummyRowFactory:
    def __init__(self, num_cols,
                 pd_nmos_model, pu_pmos_model, pg_nmos_model,
                 pd_width=0.205e-6, pu_width=0.09e-6, pg_width=0.135e-6, length=50e-9,
                 w_rc=False, disconnect=False,

                 ):
        
        self.num_cols = num_cols
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        self.w_rc = w_rc
        self.disconnect = disconnect
        

    def _get_config(self):

            # 固定模式：使用具体数值
        pd_width = self.pd_width
        pu_width = self.pu_width
        pg_width = self.pg_width
        length = self.length

        pd_model = self.pd_nmos_model
        pg_model = self.pg_nmos_model
        pu_model = self.pu_pmos_model
        
        return{
            'num_cols': self.num_cols,
            'pd_nmos_model': pd_model,
            'pu_pmos_model': pu_model,
            'pg_nmos_model': pg_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'w_rc': self.w_rc,
            'disconnect': self.disconnect,
        }
        
    def create(self):
        config = self._get_config()
        return Dummy_Row(**config)     
        
        
class ReplicaColumnFactory:
    def __init__(self, num_rows,
                 pd_nmos_model, pu_pmos_model, pg_nmos_model,
                 pd_width=0.205e-6, pu_width=0.09e-6, pg_width=0.135e-6, length=50e-9,
                 w_rc=False,
                 # Sweep 参数
                 sweep_replica=False,
                 pmos_choices=None, nmos_choices=None,
                 ):
        
        self.num_rows = num_rows
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        self.w_rc = w_rc
        
        self.sweep = sweep_replica
        self.pmos_choices = pmos_choices
        self.nmos_choices = nmos_choices
        self.REQUIRED_COLUMNS1 = ['pmos_model_pu', 'nmos_model_pd']
        self.mos_model_index1 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS1)
        self.REQUIRED_COLUMNS2 = ['nmos_model_pg']
        self.mos_model_index2 = read_mos_model_from_param_file(self.REQUIRED_COLUMNS2)

    def _get_config(self):
        if self.sweep:
            # 扫描模式：使用 SPICE 变量名字符串
            pd_width = 'nmos_width_pd'
            pu_width = 'pmos_width_pu'
            pg_width = 'nmos_width_pg'
            length = 'length'
        else:
            # 固定模式：使用具体数值
            pd_width = self.pd_width
            pu_width = self.pu_width
            pg_width = self.pg_width
            length = self.length

        # 2. 处理模型 (Models)
        if self.sweep:
            # 扫描模式：从 choices 列表中根据 index 选取
            pd_model = self.nmos_choices[int(self.mos_model_index1['nmos'])]
            pg_model = self.nmos_choices[int(self.mos_model_index2['nmos'])]
            pu_model = self.pmos_choices[int(self.mos_model_index1['pmos'])]
        else:
            # 固定模式：使用初始化时传入的模型名
            pd_model = self.pd_nmos_model
            pg_model = self.pg_nmos_model
            pu_model = self.pu_pmos_model
        
        return{
            'num_rows': self.num_rows,
            'pd_nmos_model': pd_model,
            'pu_pmos_model': pu_model,
            'pg_nmos_model': pg_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'w_rc': self.w_rc,
        }
        
    def create(self):
        config = self._get_config()
        return Replica_Column(**config)
    
class TIMEFactory:
    def __init__(self, 
                 nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=0.27e-6, nmos_width=0.18e-6,
                 length=0.05e-6, num_rows=16, num_cols=8,
                 w_rc=False, operation='read'
                 ):

        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        self.nmos_width = nmos_width
        self.length = length
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.w_rc = w_rc
        self.operation = operation
    def create(self):
        return TIME(
            nmos_model=self.nmos_model,
            pmos_model=self.pmos_model,
            pmos_width=self.pmos_width,
            nmos_width=self.nmos_width,
            length=self.length,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            w_rc=self.w_rc,
            operation=self.operation
        )
