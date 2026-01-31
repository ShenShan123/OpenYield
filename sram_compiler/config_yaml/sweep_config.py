"""
SRAM 扫描配置文件
定义了各个模块的名称、CSV文件名以及对应的参数列表
"""

SWEEP_CONFIGS = {
    'cell': {
        'name': 'sram_6t_cell',
        'params': ['pmos_width_pu', 'nmos_width_pd', 'nmos_width_pg', 'length'],
        'model_params': ['pmos_model_pu', 'nmos_model_pd', 'nmos_model_pg']
    },
    'cell_10T': {
        'name': 'sram_10t_cell',
        'params': ['pmos_width_pu', 'nmos_width_pd', 'nmos_width_pg', 'nmos_width_fd', 'length'],
        'model_params': ['pmos_model_pu_10T', 'nmos_model_pd_10T', 'nmos_model_pg_10T', 'nmos_model_fd_10T']
    },
    'precharge': {
        'name': 'precharge',
        'params': ['pmos_width_precharge', 'length_precharge'],
        'model_params': ['pmos_model_precharge']
    },
    'senseamp': {
        'name': 'senseamp',
        'params': ['pmos_width_senseamp', 'nmos_width_senseamp', 'length_senseamp'],
        'model_params': ['pmos_model_senseamp', 'nmos_model_senseamp']
    },
    'wordlinedriver': {
        'name': 'wordlinedriver',  
        'params': ['pmos_width_wld_nandp', 'nmos_width_wld_nandn', 
                   'pmos_width_wld_invp', 'nmos_width_wld_invn', 'length_wld'],
        'model_params': ['pmos_model_wld_nandp', 'nmos_model_wld_nandn', 
                         'pmos_model_wld_invp', 'nmos_model_wld_invn']
    },
    'columnmux': {
        'name': 'columnmux',
        'params': ['pmos_width_mux', 'nmos_width_mux', 'length_mux'],
        'model_params': ['pmos_model_columnmux', 'nmos_model_columnmux']    
    },
    'writedriver': {
        'name': 'writedriver',
        'params': ['pmos_width_wrd', 'nmos_width_wrd', 'length_wrd'],
        'model_params': ['pmos_model_writedriver', 'nmos_model_writedriver']
    },
    'decoder': {
        'name': 'decoder',
        'params': ['pmos_width_decoder_nandp', 'nmos_width_decoder_nandn', 
                   'pmos_width_decoder_invp', 'nmos_width_decoder_invn', 'length_decoder'],
        'model_params': ['pmos_model_decoder_nandp', 'nmos_model_decoder_nandn', 
                         'pmos_model_decoder_invp', 'nmos_model_decoder_invn']
    }
}
