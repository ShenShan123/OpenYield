from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit, Circuit
from PySpice.Unit import u_Ohm, u_pF

class BaseSubcircuit(SubCircuit):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'BASE_SUBCKT'
    # The first and second nodes are always power and ground nodes,VDD and VSS
    NODES = ('VDD', 'VSS')
    
    def __init__(self, 
                 nmos_model_name, pmos_model_name,
                 nmos_width=0.9e-6, pmos_width=1.8e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 ):
        super().__init__(self.NAME, *self.NODES)
        
        self.nmos_pdk_model = nmos_model_name
        self.pmos_pdk_model = pmos_model_name
        print(f"\n[DEBUG] Creating {self.name} with models: "
              f"NMOS={self.nmos_pdk_model}, PMOS={self.pmos_pdk_model}")

        # Transistor Sizes (FreePDK45 uses nanometers)
        self.base_nmos_width = nmos_width
        self.base_pmos_width = pmos_width
        self.length = length

        # use RC?
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap
        
    def add_rc_networks_to_node(self, in_node, num_segs=1, end_name=None):
        ###Add RC networks to the SRAM cell###
        start_node = in_node
        end_node = start_node

        for i in range(num_segs):
            if num_segs-1 == i:
                if end_name:
                    end_node = end_node
                else:
                    end_node = in_node + '_end' 
            else:
                end_node = start_node + f'_seg{i}'

            self.R(f'R_{in_node}_{i}',  start_node, end_node, self.pi_res)
            self.C(f'Cg_{in_node}_{i}', end_node, self.gnd, self.pi_cap)
            start_node = end_node
        
        return end_node