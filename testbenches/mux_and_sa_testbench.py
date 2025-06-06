import os
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_uW
from subcircuits.mux_and_sa import ColumnMux, SenseAmp
from utils import plot_results, measure_delay, measure_power
# from utils import parse_mt0, analyze_mt0
from testbenches.base_testbench import BaseTestbench
import numpy as np

class SingleColumnMuxSenseAmp_Testbench(BaseTestbench):
    """Testbench for single Column Mux and Sense Amp pair"""
    
    def __init__(self, pdk_path, nmos_model_name, pmos_model_name, 
                 mux_ratio=4, w_rc=False):
        super().__init__(
            f'SingleColumnMuxSenseAmp_{mux_ratio}_Testbench',
            pdk_path, nmos_model_name, pmos_model_name
        )
        self.mux_ratio = mux_ratio
        self.w_rc = w_rc
        
        # # Timing parameters
        # self.t_precharge = 1 @ u_ns
        # self.t_sense_enable = 2 @ u_ns
        # self.t_period = 5 @ u_ns

    def create_testbench(self, target_column):
        """Create testbench with single mux/sense amp pair"""
        circuit = Circuit(self.name)
        circuit.include(self.pdk_path)
        
        # Power supply
        circuit.V('VDD', 'VDD', circuit.gnd, self.vdd)

        # 1. Add Column Mux
        mux = ColumnMux(
            num_in=self.mux_ratio,
            nmos_model_name=self.nmos_model_name,
            pmos_model_name=self.pmos_model_name,
            nmos_width=0.2e-6,
            pmos_width=0.4e-6,
            length=50e-9
        )
        circuit.subcircuit(mux)
        circuit.X('Mux', mux.name, 
                 'VDD', circuit.gnd, 'SA_BL', 'SA_BLB',
                 *[f'SEL{i}' for i in range(self.mux_ratio)],
                 *[f'BL{i}' for i in range(self.mux_ratio)],
                 *[f'BLB{i}' for i in range(self.mux_ratio)])

        # 2. Add Sense Amplifier
        sa = SenseAmp(
            nmos_model_name=self.nmos_model_name,
            pmos_model_name=self.pmos_model_name,
            nmos_width=0.1e-6,
            pmos_width=0.2e-6,
            length=45e-9
        )
        circuit.subcircuit(sa)
        circuit.X('SA', sa.name, 'VDD', circuit.gnd,
                 'SA_BL', 'SA_BLB', 'DOUT', 'DOUTB', 'SAE')

        # 3. Add Control Signals
        self._add_control_signals(circuit, target_column=target_column)
        
        # 4. Add Stimulus
        self._add_stimulus(circuit, target_column)
        
        # 5. Add Load Caps
        circuit.C('CL', 'DOUT', circuit.gnd, 0.01 @ u_pF)
        circuit.C('CLB', 'DOUTB', circuit.gnd, 0.01 @ u_pF)

        return circuit

    def _add_control_signals(self, circuit, target_column):
        """Add control signal generators"""
        # Column Select Signals
        circuit.PulseVoltageSource(
            f'SEL{target_column}', f'SEL{target_column}', circuit.gnd,
            initial_value=0, pulsed_value=self.vdd,
            delay_time=self.t_pulse,
            rise_time=self.t_rise, fall_time=self.t_fall,
            pulse_width=self.t_pulse - 2*self.t_rise,
            period=self.t_period *2,
        )

        # Sense Enable (SE)
        circuit.PulseVoltageSource(
            'SAE', 'SAE', circuit.gnd,
            initial_value=0, pulsed_value=self.vdd,
            delay_time=self.t_pulse*2,
            rise_time=self.t_rise, fall_time=self.t_fall,
            pulse_width=self.t_pulse,
            period=self.t_period *2,
        )

        for i in range(1, self.mux_ratio):
            if target_column != i:
                circuit.V(f'SEL{i}' , f'SEL{i}', circuit.gnd, 0)

    def _add_stimulus(self, circuit, target_column):
        """Add bitline stimulus for all mux inputs"""
        # for i in range(self.mux_ratio):
            # Differential stimulus pairs
        circuit.PieceWiseLinearVoltageSource(
            f'BL{target_column}_stim', f'BL{target_column}', circuit.gnd,
            values=[(0, self.vdd), (self.t_pulse, self.half_vdd*0.5)],
        )
        ## Mimic the leakage current
        circuit.PieceWiseLinearVoltageSource(
            f'BLB{target_column}_stim', f'BLB{target_column}', circuit.gnd,
            values=[(0, self.vdd), (self.t_period*2, 0.9*self.vdd)],
        )

    def run_simulation(self, target_column=0):
        """Run characterization for selected column"""
        circuit = self.create_testbench(target_column)
        simulator = circuit.simulator()

        # Initial conditions
        init_cond = {
            'DOUT': self.vdd @ u_V,
            'DOUTB': 0 @ u_V,
            'SA_BL': 0 @ u_V, 
            'SA_BLB': 0 @ u_V,
            # floating BLs/BLBs
            **{f'BL{i}': self.vdd @ u_V for i in range(self.mux_ratio)},
            **{f'BLB{i}': self.vdd @ u_V for i in range(self.mux_ratio)}
        }

        simulator.initial_condition(**init_cond)
        print(simulator)
        # Run transient simulation
        analysis = simulator.transient(
            step_time=0.01 @ u_ns,
            end_time=self.t_period*2
        )

        # Plot key signals
        plot_nodes = [
            f'BL{target_column}', #f'BLB{target_column}',
            f'SEL{target_column}',
            'SA_BL', 'SA_BLB', 
            'DOUT', 'DOUTB', 'SAE'
        ]
        plot_results(
            analysis, plot_nodes,
            fig_name=f'plots/single_mux_sense_col{target_column}.png'
        )

        # Critical measurements
        metrics = {
            # Measure SEL->BL propagation delay
            'mux_delay': measure_delay(
                analysis.time,
                [analysis[f'SEL{target_column}'], analysis['SA_BLB']],
                trig_val=self.half_vdd,
                targ_val=self.half_vdd,
                targ_edge_type='rise'
            ),

            # Measure SE->DOUT resolution time
            'sense_time': measure_delay(
                analysis.time,
                [analysis['SAE'], analysis['DOUT']],
                trig_val=self.half_vdd,
                targ_val=0.9*self.vdd,
                targ_edge_type='fall'
            ),

            'power': measure_power(analysis['VDD'], analysis['VVDD'])
        }

        print(metrics)

        return {
            'success': True,
            'metrics': metrics,
            'analysis': analysis
        }

if __name__ == '__main__':
    print("===== MUX and SA Debug Session =====")
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTL'
    pmos_model_name = 'PMOS_VTL'

    # Initialize testbench for 8:1 mux with 4 column groups
    mux_sa_tb = SingleColumnMuxSenseAmp_Testbench(
        pdk_path='model_lib/models.spice',
        nmos_model_name='NMOS_VTL',
        pmos_model_name='PMOS_VTL',
        mux_ratio=8,
        w_rc=False,
    )

    # Run characterization for group 0, column 3
    results = mux_sa_tb.run_simulation(target_column=0)

    print(f"Mux Delay: {results['metrics']['mux_delay']:.2e} s")
    print(f"Sense Time: {results['metrics']['sense_time']:.2e} s")
    print(f"Average Power: {results['metrics']['power']:.2e} W")
