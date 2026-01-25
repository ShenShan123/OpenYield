"""
SRAM 扫描配置文件
定义了各个模块的名称、CSV文件名以及对应的参数列表
"""

SWEEP_CONFIGS = {
    'cell': {
        'name': 'SRAM_6T_CELL',
        'csv': 'param_sweep_6t_cell.csv',
        'params': ['pmos_width_pu', 'nmos_width_pd', 'nmos_width_pg', 'length']
    },
    'precharge': {
        'name': 'PRECHARGE',
        'csv': 'param_sweep_precharge.csv',
        'params': ['pmos_width_precharge', 'length_precharge']
    },
    'senseamp': {
        'name': 'SENSEAMP',
        'csv': 'param_sweep_senseamp.csv',
        'params': ['pmos_width_senseamp', 'nmos_width_senseamp', 'length_senseamp']
    },
    'wordlinedriver': {
        'name': 'WORDLINEDRIVER',  
        'csv': 'param_sweep_wordlinedriver.csv',
        'params': ['pmos_width_wld_nandp', 'nmos_width_wld_nandn', 
                   'pmos_width_wld_invp', 'nmos_width_wld_invn', 'length_wld']
    },
    'columnmux': {
        'name': 'COLUMNMUX',
        'csv': 'param_sweep_columnmux.csv',
        'params': ['pmos_width_mux', 'nmos_width_mux', 'length_mux']
    },
    'writedriver': {
        'name': 'WRITEDRIVER',
        'csv': 'param_sweep_writedriver.csv',
        'params': ['pmos_width_wrd', 'nmos_width_wrd', 'length_wrd']
    },
    'decoder': {
        'name': 'DECODER',
        'csv': 'param_sweep_decoder.csv',
        'params': ['pmos_width_decoder_nandp', 'nmos_width_decoder_nandn', 
                   'pmos_width_decoder_invp', 'nmos_width_decoder_invn', 'length_decoder']
    }
}
