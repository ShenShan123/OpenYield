import numpy as np
from typing import Optional
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F
from sram_compiler.config_yaml.config import GlobalConfig
import yaml
import matplotlib.pyplot as plt

image_base_path = "equivalent_modeling/images/"


def last_proportion_mean(arr, p=1 / 6):
    n = max(1, int(len(arr) * p))
    return np.mean(arr[-n:])


class SRAMCellParasiticTester:
    def __init__(
        self,
        cell_type="6T",
        config=None,
        pd_nmos_model="NMOS_VTH",
        pu_pmos_model="PMOS_VTG",
        pg_nmos_model="NMOS_VTH",
        fd_nmos_model=None,
        pd_width=1.8e-7,
        pu_width=1.2e-7,
        pg_width=0.7e-7,
        fd_width=None,
        length=74.0e-9,
        q_init_val=0,
    ):
        self.cell_type = cell_type.upper()
        if self.cell_type not in ("6T", "10T"):
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        
        self.pd_nmos_model = pd_nmos_model
        self.pu_pmos_model = pu_pmos_model
        self.pg_nmos_model = pg_nmos_model
        self.fd_nmos_model = fd_nmos_model

        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.fd_width = fd_width
        self.length = length
        self.q_init_val = q_init_val

        self.subckt_name = "SRAM_6T_CELL" if self.cell_type == "6T" else "SRAM_10T_CELL"

        if self.cell_type == "10T":
            if self.fd_nmos_model is None:
                raise ValueError("fd_nmos_model is required when cell_type='10T'")
            if self.fd_width is None:
                raise ValueError("fd_width is required when cell_type='10T'")

        if config is None:
            try:
                with open("config.yaml", "r") as f:
                    config_data = yaml.safe_load(f)
                self.config = GlobalConfig(config_data)
            except FileNotFoundError:
                self.config = GlobalConfig({})
        else:
            self.config = config

        self.corner = self.config.corner
        self.step_time = 1e-14
        self.end_time = 10000 * self.step_time
        self.rise_time = self.step_time * 500
        self.hold_time = 5000 * self.step_time

        self._init_circuit()

    def _build_sram_cell(self):
        # Lazy import to avoid circular import with sram_6t_core/sram_10t_core.
        if self.cell_type == "6T":
            from .sram_6t_core import Sram6TCell

            return Sram6TCell(
                pd_model=self.pd_nmos_model,
                pu_model=self.pu_pmos_model,
                pg_model=self.pg_nmos_model,
                pd_width=self.pd_width,
                pu_width=self.pu_width,
                pg_width=self.pg_width,
                length=self.length,
            )

        from .sram_10t_core import Sram10TCell

        return Sram10TCell(
            pd_model=self.pd_nmos_model,
            pu_model=self.pu_pmos_model,
            pg_model=self.pg_nmos_model,
            fd_model=self.fd_nmos_model,
            pd_width=self.pd_width,
            pu_width=self.pu_width,
            pg_width=self.pg_width,
            fd_width=self.fd_width,
            length=self.length,
        )

    def _init_circuit(self):
        self.circuit = Circuit(f"SRAM_{self.cell_type}_Cell_Test")
        pdk_path = getattr(self.config, f"pdk_path_{self.corner}")
        self.circuit.include(pdk_path)

        sram_cell = self._build_sram_cell()
        self.circuit.subcircuit(sram_cell)
        self.circuit.X("SRAM1", self.subckt_name, "VDD", "VSS", "BL", "BLB", "WL")

        self.circuit.V("VDD", "VDD", self.circuit.gnd, self.config.vdd @ u_V)
        self.circuit.V("VSS", "VSS", self.circuit.gnd, 0 @ u_V)

    def _get_pwl_waveform(self, voltage_val):
        t0 = 0
        t1 = self.rise_time
        t2 = self.hold_time
        t3 = self.hold_time + self.rise_time
        t4 = self.end_time

        v = str(voltage_val).replace(" ", "")
        return f"PWL({t0} 0 {t1} {v} {t2} {v} {t3} 0 {t4} 0)"

    def _setup_port_source(self, port_name, voltage_val):
        for p in ["BL", "BLB", "WL"]:
            if f"V{p}" in self.circuit._elements:
                self.circuit._elements.pop(f"V{p}", None)

        pwl = self._get_pwl_waveform(voltage_val)
        self.circuit.V(port_name, port_name, self.circuit.gnd, pwl)

        if port_name in ["BL", "BLB"]:
            self.circuit.V("WL", "WL", self.circuit.gnd, 0 @ u_V)

    def get_current(self, source_name, analysis):
        branch_name = source_name.lower()
        if branch_name in analysis.branches:
            return -analysis.branches[branch_name].as_ndarray()

        if source_name in analysis.nodes.keys():
            return -analysis.nodes[source_name].as_ndarray()

        print(
            f"[WARNING] Could not find current for {source_name}. "
            f"Available branches: {list(analysis.branches.keys())}"
        )
        return np.zeros(len(analysis.time))

    def run_sram_transient(self, port_name, voltage_val):
        print(f"[DEBUG] Running SRAM transient simulation for {self.cell_type} {port_name} at {voltage_val}")
        self._setup_port_source(port_name, voltage_val)

        simulator = self.circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )

        vq = self.config.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.config.vdd @ u_V
        simulator.initial_condition(**{f"XSRAM1:Q": vq, f"XSRAM1:QB": vqb})

        print("[DEBUG] Netlist:")
        print(str(simulator))

        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)

        time = analysis.time.as_ndarray()
        voltage = analysis[port_name].as_ndarray()
        current = self.get_current(f"V{port_name}", analysis)
        avg_current = last_proportion_mean(current)

        return {
            "time": time,
            "voltage": voltage,
            "current": current,
            "avg_current": avg_current,
        }

    def get_equivalent_circuit(self, port_name, voltage_val, params):
        circuit = Circuit(f"{port_name}_Equiv")
        pwl = self._get_pwl_waveform(voltage_val)
        circuit.V(port_name, port_name, circuit.gnd, pwl)

        c0 = params.get("c0", 1e-16)
        circuit.C("C0", port_name, circuit.gnd, c0)

        return circuit

    def run_equivalent_transient(self, port_name, voltage_val, params):
        print(f"[DEBUG] Running Equivalent Circuit transient simulation for {port_name}")
        circuit = self.get_equivalent_circuit(port_name, voltage_val, params)
        simulator = circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )

        print("[DEBUG] Netlist:")
        print(str(simulator))

        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)

        time = analysis.time.as_ndarray()
        voltage = analysis[port_name].as_ndarray()
        current = self.get_current(f"V{port_name}", analysis)
        avg_current = last_proportion_mean(current)

        result = {
            "time": time,
            "voltage": voltage,
            "current": current,
            "avg_current": avg_current,
        }

        return result

    def compare_and_plot(self, port_name, ratios, equiv_params):
        results = []

        for ratio in ratios:
            voltage_val = self.config.vdd * ratio @ u_V
            sram_res = self.run_sram_transient(port_name, voltage_val)
            equiv_res = self.run_equivalent_transient(port_name, voltage_val, equiv_params)

            results.append(
                {
                    "ratio": ratio,
                    "voltage_val": voltage_val,
                    "sram": sram_res,
                    "equiv": equiv_res,
                }
            )

        self._plot_comparison(port_name, results)

    def _plot_comparison(self, port_name, results):
        def align_yaxis(ax1, ax2):
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()
            ratio_max = y1_max / y2_max if y2_max != 0 else 1
            ratio_min = y1_min / y2_min if y2_min != 0 else 1
            max_ratio = min(abs(ratio_max), abs(ratio_min))
            ax2.set_ylim(y1_min / max_ratio, y1_max / max_ratio)

        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(8 / 1.5, 3 * n / 1.2))
        if n == 1:
            axes = [axes]

        for i, res in enumerate(results):
            ax1 = axes[i]
            ratio = res["ratio"]
            sram = res["sram"]
            equiv = res["equiv"]

            color = "tab:blue"
            ax1.set_ylabel("Current (uA)", color=color)
            ax1.plot(sram["time"] * 1e9, sram["current"] * 1e6, color=color, linestyle="-", label="SRAM Current")
            ax1.plot(equiv["time"] * 1e9, equiv["current"] * 1e6, color=color, linestyle="--", label="Equiv Current", alpha=0.7, linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, alpha=0.6)

            ax2 = ax1.twinx()
            color = "tab:red"
            ax2.set_ylabel("Voltage (V)", color=color)
            ax2.plot(sram["time"] * 1e9, sram["voltage"], color=color, linestyle="-", label="Voltage")
            if "voltage_mid" in equiv:
                ax2.plot(equiv["time"] * 1e9, equiv["voltage_mid"], color="tab:orange", linestyle="-.", label="Equiv Mid Voltage")
            ax2.tick_params(axis="y", labelcolor=color)

            voltage_val = res["voltage_val"]
            ax1.set_title(f"{port_name} = {voltage_val}")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            align_yaxis(ax1, ax2)

        axes[-1].set_xlabel("Time (ns)")
        plt.tight_layout()
        plt.savefig(image_base_path + port_name + "_equiv_comparison.svg")
        plt.close()

    def get_port_c(self, port_name):
        voltage_val = self.config.vdd * 0.2 @ u_V
        sram_res = self.run_sram_transient(port_name, voltage_val)

        end_index = sum(sram_res["time"] < self.hold_time)
        charge = np.trapz(sram_res["current"][:end_index], sram_res["time"][:end_index])
        c_value = charge / voltage_val
        return c_value

    def get_static_power_r(self):
        print(f"[DEBUG] Running {self.cell_type} SRAM transient simulation for static power estimation")
        for p in ["BL", "BLB", "WL"]:
            if f"V{p}" in self.circuit._elements:
                self.circuit._elements.pop(f"V{p}", None)

        simulator = self.circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )

        vq = self.config.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.config.vdd @ u_V
        simulator.initial_condition(**{f"XSRAM1:Q": vq, f"XSRAM1:QB": vqb})

        print("[DEBUG] Netlist:")
        print(str(simulator))

        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)

        time = analysis.time.as_ndarray()
        voltage = analysis["VDD"].as_ndarray()
        current = self.get_current("VVDD", analysis)
        avg_current = last_proportion_mean(current)

        return {
            "R": self.config.vdd / avg_current,
            "time": time,
            "voltage": voltage,
            "current": current,
            "avg_current": avg_current,
        }


def _build_tester_from_core(core, cell_type):
    kwargs = {
        "cell_type": cell_type,
        "pd_nmos_model": core.pd_nmos_model,
        "pu_pmos_model": core.pu_pmos_model,
        "pg_nmos_model": core.pg_nmos_model,
        "pd_width": core.pd_width,
        "pu_width": core.pu_width,
        "pg_width": core.pg_width,
        "length": core.length,
        "q_init_val": core.q_init_val,
    }

    if cell_type == "10T":
        kwargs["fd_nmos_model"] = core.fd_nmos_model
        kwargs["fd_width"] = core.fd_width

    return SRAMCellParasiticTester(**kwargs)


def _add_equivalent_circuit_impl(core, cell_type):
   
    from .base_subcircuit import BaseSubcircuit
    pi_res = getattr(core, "pi_res", BaseSubcircuit.DEFAULT_PI_RES)
    pi_cap = getattr(core, "pi_cap", BaseSubcircuit.DEFAULT_PI_CAP)

    tester = _build_tester_from_core(core, cell_type)

    vdd_r = tester.get_static_power_r()["R"]
    core.R(
        "res_static_power",
        core.NODES[0],
        core.NODES[1],
        vdd_r / ((core.num_cols - 1) * (core.num_rows - 1)) @ u_Ohm,
    )

    wl_c = tester.get_port_c("WL")
    bl_c = tester.get_port_c("BL")
    blb_c = tester.get_port_c("BLB")

    if core.w_rc:
        for row in range(core.num_rows):
            if row != core.target_row:
                wl_mid_node = f"WL{row}_rc_mid"
                core.C(f"cap_WL{row}", wl_mid_node, core.NODES[1], pi_cap * (core.num_cols - 1))
                core.R(f"res_WL{row}", f"WL{row}", wl_mid_node, pi_res / (core.num_cols - 1))
                core.C(f"cap_WL{row}_cell", f"WL{row}", core.NODES[1], wl_c * (core.num_cols - 1))

        for col in range(core.num_cols):
            if col != core.target_col:
                bl_mid_node = f"BL{col}_rc_mid"
                blb_mid_node = f"BLB{col}_rc_mid"

                core.C(f"cap_BL{col}", bl_mid_node, core.NODES[1], pi_cap * (core.num_rows - 1))
                core.C(f"cap_BLB{col}", blb_mid_node, core.NODES[1], pi_cap * (core.num_rows - 1))
                core.R(f"res_BL{col}", f"BL{col}", bl_mid_node, pi_res / (core.num_rows - 1))
                core.R(f"res_BLB{col}", f"BLB{col}", blb_mid_node, pi_res / (core.num_rows - 1))

                core.C(f"cap_BL{col}_cell", f"BL{col}", core.NODES[1], bl_c * (core.num_rows - 1))
                core.C(f"cap_BLB{col}_cell", f"BLB{col}", core.NODES[1], blb_c * (core.num_rows - 1))


def add_6t_equivalent_circuit(self):
    _add_equivalent_circuit_impl(self, "6T")


def add_10t_equivalent_circuit(self):
    _add_equivalent_circuit_impl(self, "10T")


if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
        config = GlobalConfig(config_data)
    except FileNotFoundError:
        config = GlobalConfig({})

    tester = SRAMCellParasiticTester(cell_type="6T", config=config, corner="TT", q_init_val=0)

    result = tester.get_static_power_r()
    print(f"Static Power R: {result['R'] @ u_Ohm}")

    plt.plot(result["time"] * 1e9, result["current"] * 1e9)
    plt.xlabel("Time (ns)")
    plt.ylabel("Current (nA)")
    plt.title("Static Power Current Estimation")
    plt.savefig(image_base_path + "static_power_current.svg")
    plt.close()

    ratios = [1.0]

    for port in ["BL", "BLB", "WL"]:
        c0 = tester.get_port_c(port)
        equiv_params = {"c0": c0 @ u_F}
        tester.compare_and_plot(port, ratios, equiv_params)
