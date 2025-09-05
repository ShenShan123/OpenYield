from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit, Circuit
from PySpice.Unit import u_Ohm, u_pF

class BaseSubcircuit(SubCircuit):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'BASE_SUBCKT'
    # The first and second nodes are always power and ground nodes,VDD and VSS
    NODES = ('VDD', 'VSS')
    
    def __init__(self, #接受晶体管参数和rc参数
                 nmos_model_name, pmos_model_name,
                 base_nmos_width, base_pmos_width, length,
                 w_rc, pi_res, pi_cap,param_file="sram_compiler/param_sweep_data/param_sweep_model_name.txt"
                 ):
        super().__init__(self.NAME, *self.NODES)
        
        self.nmos_pdk_model = nmos_model_name#存储晶体管模型名称
        self.pmos_pdk_model = pmos_model_name
        print(f"\n[DEBUG] Creating {self.name} with models: "
              f"NMOS={self.nmos_pdk_model}, PMOS={self.pmos_pdk_model}")

        # Transistor Sizes (FreePDK45 uses nanometers)
        self.base_nmos_width = base_nmos_width#存储晶体管参数
        self.base_pmos_width = base_pmos_width
        self.length = length

        # use RC?
        self.w_rc = w_rc#存储rc寄生参数
        self.pi_res = pi_res
        self.pi_cap = pi_cap
        self.param_file = param_file
    def add_rc_networks_to_node(self, in_node, num_segs=1, end_name=None):#为指定节点添加rc网络
        ###Add RC networks to the SRAM cell###
        start_node = in_node#每段起始节点名
        end_node = start_node#每段结束节点名

        for i in range(num_segs):
            if num_segs-1 == i:#最后一段
                if end_name:    #如果最后一个节点有名字，就用这个名字
                    end_node = end_name
                else:           #不然就是起点名字+end
                    end_node = in_node + '_end' 
            else:
                end_node = start_node + f'_seg{i}' #中间节点名

            self.R(f'R_{in_node}_{i}',  start_node, end_node, self.pi_res)
            self.C(f'Cg_{in_node}_{i}', end_node, self.gnd, self.pi_cap)
            start_node = end_node   #为下一段更新起始节点
            #输出示例：RR_BL_0 BL BL_end 100Ohm  CCg_BL_0 BL_end 0 0.001pF
        
        return end_node

    def read_mos_model_from_param_file(self,names):
        """
    sim/param_sweep_model_name.txt"
        """
        try:
            with open(self.param_file, 'r') as f:
                lines = f.readlines()
                # 假设第一行为标题行，第二行为数据行
                if len(lines) >= 2:
                    header = lines[0].strip().split()
                    values = lines[1].strip().split()
                    models = {}
                    for key in names:
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
