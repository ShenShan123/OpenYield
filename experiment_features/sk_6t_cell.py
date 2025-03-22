from skidl import Part, Net, generate_netlist, set_default_tool, lib_search_paths, KICAD8
from skidl.pyspice import lib_search_paths

set_default_tool(KICAD8)
# Append your custom path to the library search paths
lib_search_paths[KICAD8].append('/home/shenshan/kicad-symbols')  # Replace with your actual path
# lib_search_paths.spice['MOS'] = 'basic'
# r1 = Part('Simulation_SPICE', 'R')  # Resistor
# Define m1 as an NMOS transistor
m1 = Part('Simulation_SPICE', 'NMOS', W=10e-6, L=1e-6, footprint='Package_TO_SOT_SMD:SOT-23')

# Define m2 as a PMOS transistor (or NMOS, depending on your circuit)
m2 = Part('Simulation_SPICE', 'PMOS', W=20e-6, L=1e-6, footprint='Package_TO_SOT_SMD:SOT-23')

# Generate a netlist or schematic as needed
generate_netlist()