# OpenYield: An Open-Source SRAM Yield Analysis and Optimization Benchmark Suite
![](img/logo-cut-openyield.jpg)
**OpenYield** is a novel and scalable SRAM circuit generator designed to produce diverse, industrially-relevant test cases. It distinguishes itself by incorporating critical second-order effects often overlooked in simpler models, such as:

* **Detailed Parasitics:** Accurate modeling of parasitic capacitances and resistances.
* **Inter-cell Leakage Coupling:** Accounting for leakage current interactions between adjacent memory cells.
* **Peripheral Circuit Variations:** Modeling variations in the behavior of peripheral circuits like sense amplifiers and write drivers.

This enhanced level of detail enables more realistic and reliable yield analysis of SRAM designs.

## Key Features

* **Xyce Integration:** Utilizes the Xyce parallel circuit simulator for transistor-level simulations.
* **Monte Carlo Simulation Support:**
    * Built-in Monte Carlo simulations within Xyce.
    * Support for user-defined Monte Carlo simulations, allowing for custom process parameter generation.
* **Performance Metrics Analysis:** Evaluates critical SRAM performance metrics:
    * Hold Static Noise Margin (SNM)
    * Read Static Noise Margin (SNM)
    * Write Static Noise Margin (SNM)
    * Read Delay
    * Write Delay
* **Output Parsing and Waveform Plotting:** Includes parsers to extract simulation results and tools to visualize signal waveforms.
* **Extensible Design:** The project is under active development with plans to integrate various yield analysis and sizing optimization algorithms.

## Dependencies

* **PySpice:** Required for interfacing with the Xyce simulator. Install using pip:

    ```bash
    pip install PySpice
    ```

## Usage Examples

### 1.  Using the `SRAM_6T_Array_MC_Testbench` Class

The `SRAM_6T_Array_MC_Testbench` class in `testbenches/sram_6t_core_MC_testbench.py` facilitates Monte Carlo simulations of the SRAM core. Here's a basic example of how to instantiate and use it:

```python
from testbenches.sram_6t_core_MC_testbench import SRAM_6T_Array_MC_Testbench

# Create an instance of the testbench
testbench = SRAM_6T_Array_MC_Testbench(
    netlist="sram_6t_core.spice",  # Path to the SRAM cell netlist
    simulator="xyce",            # Simulator to use (currently only 'xyce' supported)
    variation_type="process",     # Type of variation (e.g., "process", "mismatch")
    num_bits=1                    # Number of bits in the SRAM array
)
```
### 2. Using the `run_mc_simulation` Method
The run_mc_simulation method within the        `SRAM_6T_Array_MC_Testbench` class executes Monte Carlo simulations.  Here's an example demonstrating its usage:
```python 
from testbenches.sram_6t_core_MC_testbench import SRAM_6T_Array_MC_Testbench

testbench = SRAM_6T_Array_MC_Testbench(
    netlist="sram_6t_core.spice",
    simulator="xyce",
    variation_type="process",
    num_bits=1
)

# Define the number of Monte Carlo samples
num_samples = 100

# Run the Monte Carlo simulation
mc_results = testbench.run_mc_simulation(num_samples=num_samples)
```

## Important Notes:

* Ensure that you have Xyce installed and configured correctly. OpenYield assumes Xyce is available in your system's PATH.
* The netlist parameter should point to the SPICE netlist file describing your SRAM cell.
* The structure of the mc_results will depend on the specific analyses performed in the Monte Carlo simulation. You'll need to inspect the output to understand how to access the desired metrics.
* Refer to main.py for more complete examples and usage patterns.

## Future Development
This project is actively being developed.  Planned future enhancements include:
* Integration of advanced yield analysis algorithms.
* Implementation of SRAM cell sizing optimization techniques.

## Contributing
Contributions to OpenYield are welcome! Please refer to the contribution guidelines for details on how to get involved.