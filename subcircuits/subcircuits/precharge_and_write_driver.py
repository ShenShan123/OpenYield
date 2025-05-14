from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from subcircuits.base import BaseSubcircuit

class Precharge(BaseSubcircuit):
	"""
	Precharge circuit for SRAM bitlines with dynamically adjusted strength.
	"""
	NAME = "PRECHARGE"
	NODES = ('VDD', 'PRE', 'BL', 'BLB')  # Power, Precharge Enable, BL, BLB

	def __init__(self, pmos_model_name, base_pmos_width=0.27e-6, length=50e-9, num_rows=16):
		
		super().__init__(None, pmos_model_name, 0.0, base_pmos_width, length)
		self.num_rows = num_rows
		self.pmos_width = self.calculate_dynamic_width(base_pmos_width, num_rows)
		self.add_precharge_transistors()

	def calculate_dynamic_width(self, base_width, num_rows):
		"""
		Dynamically adjust the transistor width based on the number of rows.
		This is a simple linear scaling; you might need a more complex function.
		"""
		scaling_factor = num_rows / 16  # Example: 5% increase per additional row
		scaling_factor = 0.5 if scaling_factor < 0.5 else scaling_factor

		return base_width * scaling_factor

	def add_precharge_transistors(self):
		# PMOS transistors to precharge BL and BLB to VDD
		self.M(1, 'BL',  'PRE', 'VDD', 'VDD',
			model=self.pmos_pdk_model,
			w=self.pmos_width, l=self.length)
		self.M(2, 'BLB', 'PRE', 'VDD', 'VDD',
			model=self.pmos_pdk_model,
			w=self.pmos_width, l=self.length)
		# Equalization transistor to reduce the difference between BL and BLB
		self.M(3, 'BL', 'PRE', 'BLB', 'VDD',
			model=self.pmos_pdk_model,
			w=self.pmos_width, l=self.length)

class WriteDriver(BaseSubcircuit):
    """
    Write driver circuit for SRAM with dynamically adjusted strength.
    """
    NAME = "WRITEDRIVER"
    # VDD, GND, ENable, Data In, BL, BLB, 
    NODES = ('VDD', 'VSS', 'EN', 'DIN', 'BL', 'BLB')  

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_nmos_width=0.18e-6, base_pmos_width=0.36e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 num_rows=16):
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.num_rows = num_rows

        self.nmos_width = self.calculate_dynamic_width(base_nmos_width, num_rows)
        self.pmos_width = self.calculate_dynamic_width(base_pmos_width, num_rows)

        self.add_driver_transistors()

    def calculate_dynamic_width(self, base_width, num_rows):
       """
       Dynamically adjust the transistor width based on the number of rows.
       This is a simple linear scaling; you might need a more complex function.
       """
       #  scaling_factor = 1 + (num_rows - 1) * 0.1  # Example: 10% increase per additional row
       scaling_factor = num_rows / 16
       scaling_factor = 0.5 if scaling_factor < 0.5 else scaling_factor
       return base_width * scaling_factor

    def add_driver_transistors(self):
        # Inverters for enable and data input
        self.M(1, 'BLB_bar', 'DIN', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        self.M(2, 'BLB_bar', 'DIN', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)
        self.M(3, 'EN_bar', 'EN', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        self.M(4, 'EN_bar', 'EN', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)

        # Tristate for BL
        self.M(5, 'int1', 'BLB_bar', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        self.M(6, 'BL', 'EN_bar', 'int1', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        self.M(7, 'BL', 'EN', 'int2', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)
        self.M(8, 'int2', 'BLB_bar', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)

        # Tristate for BLB
        self.M(9, 'int3', 'DIN', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        self.M(10, 'BLB', 'EN_bar', 'int3', 'VDD',
                model=self.pmos_pdk_model,
                w=self.pmos_width, l=self.length)
        self.M(11, 'BLB', 'EN', 'int4', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)
        self.M(12, 'int4', 'DIN', 'VSS', 'VSS',
                model=self.nmos_pdk_model,
                w=self.nmos_width, l=self.length)