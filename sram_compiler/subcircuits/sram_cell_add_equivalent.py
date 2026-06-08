"""
Equivalent-circuit model for unused SRAM cells.

Migration from OpenYield2.5/sram_compiler/subcircuits/sram_cell_parasitics.py
Key improvements over the previous version:
  - 5-cap parasitic model: c_bl, c_blb, c_wl, c_wl_bl, c_wl_blb
  - Cross-coupling caps added between WL and BL/BLB mid-nodes
  - Cell caps placed at RC mid-node (physically accurate)
  - WL static-power sweep with piecewise-linear curve fit
  - Both 6T and 10T cell support
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── project root for config fall-back ──────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

image_base_path = "equivalent_modeling/images/"


# ─── colour helpers ──────────────────────────────────────────────────────────
def _build_blue_orange_cmap():
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    default_blue   = default_colors[0]
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_orange",
        ["#021729", "#083457", default_blue, "#32A0FF"],
        N=256,
    )


def last_proportion_mean(arr, p=1 / 6):
    n = max(1, int(len(arr) * p))
    return np.mean(arr[-n:])


# ─── SRAMCellParasiticTester ─────────────────────────────────────────────────
class SRAMCellParasiticTester:
    """Extracts parasitic caps + static power for a single SRAM cell.

    Supports both 6T and 10T topologies.

    Three transient tests extract a 5-element parasitic network
    (c_bl, c_blb, c_wl, c_wl_bl, c_wl_blb).
    """

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

        self.pd_width  = pd_width
        self.pu_width  = pu_width
        self.pg_width  = pg_width
        self.fd_width  = fd_width
        self.length    = length
        self.q_init_val = q_init_val

        self.subckt_name = "SRAM_6T_CELL" if self.cell_type == "6T" else "SRAM_10T_CELL"

        if self.cell_type == "10T":
            if fd_nmos_model is None:
                raise ValueError("fd_nmos_model is required when cell_type='10T'")
            if fd_width is None:
                raise ValueError("fd_width is required when cell_type='10T'")

        # ── config ──────────────────────────────────────────────────────────
        if config is None:
            config = _load_default_config()
        self.config = config
        self.corner = getattr(self.config, "corner", "TT")

        # ── simulation timing ────────────────────────────────────────────────
        self.step_time  = 1e-13
        self.end_time   = 6000 * self.step_time
        self.rise_time  = self.step_time * 2000
        self.hold_time  = 3000 * self.step_time

        self._init_circuit()

    # ── circuit initialisation ────────────────────────────────────────────────
    def _build_sram_cell(self):
        """Lazy-import cell subclass so this module avoids circular imports."""
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

    # ── waveform helpers ──────────────────────────────────────────────────────
    def _get_pwl_waveform(self, voltage_val):
        t0, t1 = 0, self.rise_time
        t2, t3 = self.hold_time, self.hold_time + self.rise_time
        t4 = self.end_time
        v = str(voltage_val).replace(" ", "")
        return f"PWL({t0} 0 {t1} {v} {t2} {v} {t3} 0 {t4} 0)"

    def _remove_port_sources(self):
        for p in ["BL", "BLB", "WL"]:
            self.circuit._elements.pop(f"V{p}", None)

    def _configure_test_sources(self, test_case, voltage_val):
        """Set BL/BLB/WL voltage sources for one of three test cases."""
        self._remove_port_sources()
        pwl = self._get_pwl_waveform(voltage_val)

        if test_case == "bl_drive":
            self.circuit.V("BL",  "BL",  self.circuit.gnd, pwl)
            self.circuit.V("BLB", "BLB", self.circuit.gnd, 0 @ u_V)
            self.circuit.V("WL",  "WL",  self.circuit.gnd, 0 @ u_V)
        elif test_case == "blb_drive":
            self.circuit.V("BLB", "BLB", self.circuit.gnd, pwl)
            self.circuit.V("BL",  "BL",  self.circuit.gnd, 0 @ u_V)
            self.circuit.V("WL",  "WL",  self.circuit.gnd, 0 @ u_V)
        elif test_case == "all_drive":
            self.circuit.V("BL",  "BL",  self.circuit.gnd, pwl)
            self.circuit.V("BLB", "BLB", self.circuit.gnd, pwl)
            self.circuit.V("WL",  "WL",  self.circuit.gnd, pwl)
        else:
            raise ValueError(f"Unknown test case: {test_case}")

    # ── current helper ────────────────────────────────────────────────────────
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

    def _get_charge(self, current, time, end_time=None):
        if end_time is None:
            end_time = self.hold_time
        end_index = np.searchsorted(time, end_time)
        return np.trapz(current[:end_index], time[:end_index])

    # ── main transient simulation ─────────────────────────────────────────────
    def run_sram_transient(self, test_case, voltage_val, measure_sources=None):
        """Run one transient test and return time, voltages, currents."""
        print(f"[DEBUG] Running SRAM transient simulation for {test_case} at {voltage_val}")
        self._configure_test_sources(test_case, voltage_val)

        simulator = self.circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )

        vq  = self.config.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.config.vdd @ u_V
        simulator.initial_condition(**{f"XSRAM1:Q": vq, f"XSRAM1:QB": vqb})

        print("[DEBUG] Netlist:")
        print(str(simulator))

        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)

        time     = analysis.time.as_ndarray()
        voltages = {p: analysis[p].as_ndarray() for p in ["BL", "BLB", "WL"]}
        if measure_sources is None:
            measure_sources = ["VBL", "VBLB", "VWL"]
        currents = {s: self.get_current(s, analysis) for s in measure_sources}

        return {"time": time, "voltages": voltages, "currents": currents}

    # ── 5-cap extraction ──────────────────────────────────────────────────────
    def extract_parasitic_caps(self, voltage_ratio=1.0):
        """Extract 5-element parasitic model via three transient tests.

        Topology:
            BL  ──[c_bl]── GND
            BLB ──[c_blb]── GND
            WL  ──[c_wl]── GND
            WL  ──[c_wl_bl]──  BL
            WL  ──[c_wl_blb]── BLB

        Tests:
          1. BL-drive (BLB=WL=0): VBL and VWL currents → c_bl, c_wl_bl
          2. BLB-drive (BL=WL=0): VBLB and VWL currents → c_blb, c_wl_blb
          3. All-drive (BL=BLB=WL=pwl): VWL current → c_wl
        """
        voltage_val = self.config.vdd * voltage_ratio @ u_V
        v = float(voltage_val)

        # Test 1: BL drive
        r1     = self.run_sram_transient("bl_drive", voltage_val, ["VBL", "VWL"])
        q_bl   = abs(self._get_charge(r1["currents"]["VBL"], r1["time"]))
        q_wl1  = abs(self._get_charge(r1["currents"]["VWL"], r1["time"]))
        c_wl_bl = q_wl1 / v
        c_bl    = max((q_bl - q_wl1) / v, 0.0)

        # Test 2: BLB drive
        r2      = self.run_sram_transient("blb_drive", voltage_val, ["VBLB", "VWL"])
        q_blb   = abs(self._get_charge(r2["currents"]["VBLB"], r2["time"]))
        q_wl2   = abs(self._get_charge(r2["currents"]["VWL"],  r2["time"]))
        c_wl_blb = q_wl2 / v
        c_blb    = max((q_blb - q_wl2) / v, 0.0)

        # Test 3: All-drive → WL gnd cap
        r3    = self.run_sram_transient("all_drive", voltage_val, ["VWL"])
        q_wl3 = abs(self._get_charge(r3["currents"]["VWL"], r3["time"]))
        c_wl  = q_wl3 / v

        return {
            "voltage_val": voltage_val,
            "caps": {
                "c_bl":     c_bl,
                "c_blb":    c_blb,
                "c_wl":     c_wl,
                "c_wl_bl":  c_wl_bl,
                "c_wl_blb": c_wl_blb,
            },
            "actual": {
                "bl_drive":  r1,
                "blb_drive": r2,
                "all_drive": r3,
            },
        }

    # ── equivalent circuit ────────────────────────────────────────────────────
    def get_equivalent_circuit(self, params):
        circuit = Circuit("SRAM_Equivalent")
        circuit.C("CBLGND",  "BL",  circuit.gnd, params.get("c_bl",     1e-16))
        circuit.C("CBLBGND", "BLB", circuit.gnd, params.get("c_blb",    1e-16))
        circuit.C("CWLGND",  "WL",  circuit.gnd, params.get("c_wl",     1e-16))
        circuit.C("CWLBL",   "WL",  "BL",        params.get("c_wl_bl",  1e-16))
        circuit.C("CWLBLB",  "WL",  "BLB",       params.get("c_wl_blb", 1e-16))
        return circuit

    def run_equivalent_test(self, test_case, voltage_val, params, measure_sources=None):
        circuit = self.get_equivalent_circuit(params)
        pwl = self._get_pwl_waveform(voltage_val)

        if test_case == "bl_drive":
            circuit.V("BL",  "BL",  circuit.gnd, pwl);  circuit.V("BLB", "BLB", circuit.gnd, 0 @ u_V);  circuit.V("WL", "WL", circuit.gnd, 0 @ u_V)
        elif test_case == "blb_drive":
            circuit.V("BLB", "BLB", circuit.gnd, pwl);  circuit.V("BL",  "BL",  circuit.gnd, 0 @ u_V);  circuit.V("WL", "WL", circuit.gnd, 0 @ u_V)
        elif test_case == "all_drive":
            circuit.V("BL",  "BL",  circuit.gnd, pwl);  circuit.V("BLB", "BLB", circuit.gnd, pwl);  circuit.V("WL", "WL", circuit.gnd, pwl)
        else:
            raise ValueError(f"Unknown test case: {test_case}")

        simulator = circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )
        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)
        time     = analysis.time.as_ndarray()
        voltages = {p: analysis[p].as_ndarray() for p in ["BL", "BLB", "WL"]}
        if measure_sources is None:
            measure_sources = ["VBL", "VBLB", "VWL"]
        currents = {s: self.get_current(s, analysis) for s in measure_sources}
        return {"time": time, "voltages": voltages, "currents": currents}

    def run_equivalent_suite(self, voltage_ratio, params):
        voltage_val = self.config.vdd * voltage_ratio @ u_V
        return {
            "bl_drive":  self.run_equivalent_test("bl_drive",  voltage_val, params, ["VBL", "VWL"]),
            "blb_drive": self.run_equivalent_test("blb_drive", voltage_val, params, ["VBLB", "VWL"]),
            "all_drive": self.run_equivalent_test("all_drive", voltage_val, params, ["VWL"]),
        }

    def compare_and_plot(self, ratios, params):
        results = {}
        for ratio in ratios:
            actual = self.extract_parasitic_caps(ratio)
            equiv  = self.run_equivalent_suite(ratio, params)
            results[ratio] = {"actual": actual, "equiv": equiv}

        plots = [
            ("bl_drive",  "VBL",  "BL",  "BL"),
            ("bl_drive",  "VWL",  "BL",  "WL (BL drive)"),
            ("blb_drive", "VBLB", "BLB", "BLB"),
            ("blb_drive", "VWL",  "BLB", "WL (BLB drive)"),
            ("all_drive", "VWL",  "WL",  "WL (all drive)"),
        ]
        for case, source, voltage_port, title_suffix in plots:
            self._plot_current_figure(case, source, voltage_port, ratios, results, title_suffix)

    def _plot_current_figure(self, case, source, voltage_port, ratios, results, title_suffix):
        fig, axes = plt.subplots(len(ratios), 1, figsize=(10, 3.5 * len(ratios)))
        if len(ratios) == 1:
            axes = [axes]

        for idx, ratio in enumerate(ratios):
            ax1 = axes[idx]
            actual_case = results[ratio]["actual"]["actual"][case]
            equiv_case  = results[ratio]["equiv"][case]

            ax2 = ax1.twinx()
            ax2.set_zorder(0);  ax1.set_zorder(1);  ax1.patch.set_alpha(0.0)

            ax1.set_ylabel("电流 (µA)", color="tab:blue")
            ax1.plot(actual_case["time"]*1e9, actual_case["currents"][source]*1e6,
                     label=f"实际 {source[1:]} 电流", color="tab:blue")
            ax1.plot(equiv_case["time"]*1e9,  equiv_case["currents"][source]*1e6,
                     linestyle="--", label=f"模型 {source[1:]} 电流", color="tab:cyan")
            ax1.tick_params(axis="y", labelcolor="tab:blue");  ax1.grid(True, alpha=0.3)

            ax2.set_ylabel("电压 (V)", color="tab:orange")
            ax2.plot(actual_case["time"]*1e9, actual_case["voltages"][voltage_port],
                     color="tab:orange", label="驱动电压")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            ax1.set_title(f"驱动电压: {ratio:.1f}V")
            l1, la1 = ax1.get_legend_handles_labels();  l2, la2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, la1 + la2, loc="best")

        axes[-1].set_xlabel("时间 (ns)")
        plt.tight_layout()
        Path(image_base_path).mkdir(parents=True, exist_ok=True)
        file_name = f"simplified_model_{case}_{source.lower()}_comparison.svg"
        fig_path  = image_base_path + file_name
        print(f"[DEBUG] Saving plot to {fig_path}")
        plt.savefig(fig_path);  plt.close()

    # ── static power helpers ──────────────────────────────────────────────────
    def _voltage_value(self, voltage):
        if hasattr(voltage, "value"):
            try:
                return float(voltage.value)
            except Exception:
                pass
        try:
            return float(voltage)
        except Exception:
            return float((voltage @ u_V).value)

    def _resolve_bl_blb_rails(self, bl_voltage, blb_voltage):
        bl_val  = self._voltage_value(bl_voltage)
        blb_val = self._voltage_value(blb_voltage)
        if bl_val >= blb_val:
            return "VDD", "VSS", float(self.config.vdd), 0.0
        return "VSS", "VDD", 0.0, float(self.config.vdd)

    def _configure_static_hold_sources(self, wl_voltage, bl_voltage, blb_voltage, use_rails=False):
        self._remove_port_sources()
        wl_v = wl_voltage @ u_V
        self.circuit.V("WL", "WL", self.circuit.gnd, wl_v)
        if use_rails:
            bl_rail, blb_rail, bl_v_val, blb_v_val = self._resolve_bl_blb_rails(bl_voltage, blb_voltage)
            self.circuit.V("BL",  "BL",  bl_rail,  0 @ u_V)
            self.circuit.V("BLB", "BLB", blb_rail, 0 @ u_V)
            return wl_v, bl_v_val @ u_V, blb_v_val @ u_V
        bl_v  = bl_voltage  @ u_V
        blb_v = blb_voltage @ u_V
        self.circuit.V("BL",  "BL",  self.circuit.gnd, bl_v)
        self.circuit.V("BLB", "BLB", self.circuit.gnd, blb_v)
        return wl_v, bl_v, blb_v

    def run_static_power(self, wl_voltage, bl_voltage, blb_voltage,
                         q_state=1, measure_source="VVDD", use_rails=False):
        wl_v, bl_v, blb_v = self._configure_static_hold_sources(
            wl_voltage, bl_voltage, blb_voltage, use_rails=use_rails
        )
        print(f"[DEBUG] Static power WL={float(wl_v.value):.6g} "
              f"BL={float(bl_v.value):.6g} BLB={float(blb_v.value):.6g} Q_state={q_state}")

        simulator = self.circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )
        vq  = self.config.vdd @ u_V if q_state else 0 @ u_V
        vqb = 0 @ u_V if q_state else self.config.vdd @ u_V
        simulator.initial_condition(**{f"XSRAM1:Q": vq, f"XSRAM1:QB": vqb})
        simulator.circuit.raw_spice += ".PRINT TRAN FORMAT=NOINDEX V(XSRAM1:Q) V(XSRAM1:QB)\n"

        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)
        time    = analysis.time.as_ndarray()
        voltage = analysis["VDD"].as_ndarray()
        current = self.get_current(measure_source, analysis)
        voltages = {}
        for port in ["BL", "BLB", "WL", "Q", "QB", "XSRAM1:Q", "XSRAM1:QB"]:
            if port in analysis.nodes:
                voltages[port] = analysis[port].as_ndarray()

        avg_current = abs(last_proportion_mean(current))
        try:
            max_current = float(np.max(np.abs(current)))
        except Exception:
            max_current = float(avg_current)
        avg_power = avg_current * float(self.config.vdd)

        return {
            "wl_voltage":  float(wl_v.value),
            "bl_voltage":  float(bl_v.value),
            "blb_voltage": float(blb_v.value),
            "q_state":     q_state,
            "time":        time,
            "voltage":     voltage,
            "voltages":    voltages,
            "current":     current,
            "avg_current": avg_current,
            "max_current": max_current,
            "avg_power":   avg_power,
        }

    def get_static_power_r(self):
        print(f"[DEBUG] Running {self.cell_type} SRAM transient for static power estimation")
        for p in ["BL", "BLB", "WL"]:
            self.circuit._elements.pop(f"V{p}", None)

        simulator = self.circuit.simulator(
            temperature=self.config.temperature,
            nominal_temperature=self.config.temperature,
            simulator="xyce-parallel",
        )
        vq  = self.config.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.config.vdd @ u_V
        simulator.initial_condition(**{f"XSRAM1:Q": vq, f"XSRAM1:QB": vqb})
        print("[DEBUG] Netlist:");  print(str(simulator))

        analysis    = simulator.transient(step_time=self.step_time, end_time=self.end_time)
        time        = analysis.time.as_ndarray()
        voltage     = analysis["VDD"].as_ndarray()
        current     = self.get_current("VVDD", analysis)
        avg_current = last_proportion_mean(current)

        return {
            "R":           self.config.vdd / avg_current,
            "time":        time,
            "voltage":     voltage,
            "current":     current,
            "avg_current": avg_current,
        }

    # ── WL static-power sweep + fitting (from OpenYield2.5) ───────────────────
    @staticmethod
    def _piecewise_static_model(v_ratio, p0, p1, p2, p3, p4, p5, p6, p7):
        """Smooth blend of exp and shifted quadratic."""
        sig   = 1.0 / (1.0 + np.exp(-p7 * (v_ratio - p4)))
        left  = p0 * np.exp(p1 * v_ratio) + p6
        right = p2 * (v_ratio - p5) ** 2 + p3
        return (1.0 - sig) * left + sig * right

    def simulate_static_power_wl_sweep(
        self, wl_ratios=None, bl_voltage=0, blb_voltage=None,
        q_state=1, break_on_decrease=True, decrease_threshold=0.02,
        stop_on_drop=False,
    ):
        if wl_ratios is None:
            wl_ratios = np.linspace(0, 1, 100)
        if blb_voltage is None:
            blb_voltage = self.config.vdd

        results          = []
        last_good_current = None
        for wl_ratio in np.array(wl_ratios):
            result = self.run_static_power(
                wl_voltage=wl_ratio * self.config.vdd,
                bl_voltage=bl_voltage,
                blb_voltage=blb_voltage,
                q_state=q_state,
                use_rails=True,
            )
            current = result.get("max_current", result.get("avg_current", 0.0))
            dropped = False
            if break_on_decrease and last_good_current is not None:
                if current < last_good_current * (1 - decrease_threshold):
                    dropped = True
                    print(f"Current decreased >{decrease_threshold*100:.0f}% at WL ratio {wl_ratio:.2f}")
            result["dropped"] = dropped
            results.append(result)
            if dropped and stop_on_drop:
                break
            if not dropped and current > 0:
                last_good_current = current
        return results

    def fit_static_power_vs_wl(
        self, wl_ratios=None, bl_voltage=0, blb_voltage=None, q_state=1,
        fit_label=None, break_on_decrease=True, decrease_threshold=0.02,
    ):
        results = self.simulate_static_power_wl_sweep(
            wl_ratios=wl_ratios, bl_voltage=bl_voltage, blb_voltage=blb_voltage,
            q_state=q_state, break_on_decrease=break_on_decrease,
            decrease_threshold=decrease_threshold,
        )
        if not results:
            raise ValueError("No static power results collected for WL sweep.")

        wl_ratios_all   = np.array([r["wl_voltage"] / self.config.vdd for r in results])
        avg_currents_all = np.array([r["avg_current"] for r in results])
        max_currents_all = np.array([r.get("max_current", r["avg_current"]) for r in results])
        fit_mask = np.array([(not r.get("dropped", False)) and (mc > 0)
                             for r, mc in zip(results, max_currents_all)])
        if np.count_nonzero(fit_mask) < 2:
            raise ValueError("Need at least two non-zero current points to fit.")

        wl_ratios_fit   = wl_ratios_all[fit_mask]
        max_currents_fit = max_currents_all[fit_mask]
        avg_currents_fit = avg_currents_all[fit_mask]

        popt, pcov = curve_fit(
            self._piecewise_static_model, wl_ratios_fit, max_currents_fit,
            p0=(1e-6, 1.0, 1e-6, 0.0, 0.5, 0.5, 0.0, 10.0),
            bounds=(
                [0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf, 0.1],
                [np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, np.inf, 100.0],
            ),
            maxfev=10000,
        )

        def fit_func(x):
            return self._piecewise_static_model(x, *popt)

        if fit_label is None:
            fit_label = (
                f"Fitted: smooth blend; "
                f"p0={popt[0]:.2e}, p1={popt[1]:.3f}, p2={popt[2]:.2e}, "
                f"p3={popt[3]:.2e}, p4={popt[4]:.3f}, p5={popt[5]:.3f}, "
                f"p6={popt[6]:.2e}, p7={popt[7]:.3f}"
            )

        return {
            "results":      results,
            "wl_ratios":    wl_ratios_fit,
            "avg_currents": avg_currents_fit,
            "max_currents": max_currents_fit,
            "fit_mask":     fit_mask,
            "popt":         popt,
            "pcov":         pcov,
            "fit_func":     fit_func,
            "fit_label":    fit_label,
        }

    # ── plotting ──────────────────────────────────────────────────────────────
    def plot_static_power_vs_wl(self, results, filename="static_power_vs_wl.svg",
                                 fit_func=None, fit_label=None):
        Path(image_base_path).mkdir(parents=True, exist_ok=True)
        ratios      = np.array([r["wl_voltage"] / self.config.vdd for r in results])
        max_current = np.array([r.get("max_current", r["avg_current"]) for r in results]) * 1e6

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ratios, max_current, marker="s", color="tab:blue", label="Max Current")
        if fit_func is not None:
            fit_x = np.linspace(ratios.min(), ratios.max(), 200)
            ax.plot(fit_x, fit_func(fit_x), color="tab:orange", linestyle="--", label="Fitted")
        ax.set_xlabel("WL / VDD");  ax.set_ylabel("最大电流 (µA)")
        ax.tick_params(axis="y");  ax.grid(True, alpha=0.3)
        if fit_func is not None:
            ax.legend(loc="upper left", title=fit_label)
        fig.tight_layout()
        save_path = Path(image_base_path) / filename
        fig.savefig(save_path);  plt.close(fig)
        return str(save_path)

    def plot_static_waveform_suite(self, results, filename="static_waveforms.svg",
                                    cmap_name="viridis", x_limits=None, y_limits=None):
        Path(image_base_path).mkdir(parents=True, exist_ok=True)
        ratios = np.array([r["wl_voltage"] / self.config.vdd for r in results])
        norm   = plt.Normalize(vmin=ratios.min(), vmax=ratios.max())
        cmap   = _build_blue_orange_cmap()

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        labels = ["VDD 电流 (µA)", "Q 电压 (V)", "QB 电压 (V)"]

        for result, ratio in zip(results, ratios):
            color   = cmap(norm(ratio))
            time_ns = result["time"] * 1e9
            axes[0].plot(time_ns, result["current"] * 1e6, color=color, linewidth=1)
            q_voltage  = result["voltages"].get("Q") or result["voltages"].get("XSRAM1:Q")
            qb_voltage = result["voltages"].get("QB") or result["voltages"].get("XSRAM1:QB")
            if q_voltage  is not None: axes[1].plot(time_ns, q_voltage,  color=color, linewidth=1)
            if qb_voltage is not None: axes[2].plot(time_ns, qb_voltage, color=color, linewidth=1)

        for ax, label in zip(axes, labels):
            ax.set_ylabel(label);  ax.grid(True, alpha=0.3)
        if x_limits is not None:
            for ax in axes: ax.set_xlim(x_limits)
        if y_limits is not None:
            if "all"     in y_limits: [ax.set_ylim(y_limits["all"]) for ax in axes]
            if "current" in y_limits: axes[0].set_ylim(y_limits["current"])
            if "q"       in y_limits: axes[1].set_ylim(y_limits["q"])
            if "qb"      in y_limits: axes[2].set_ylim(y_limits["qb"])

        axes[2].set_xlabel("Time (ns)")
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);  sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.03)
        cbar.set_label("WL / VDD")
        save_path = Path(image_base_path) / filename
        fig.savefig(save_path, bbox_inches="tight");  plt.close(fig)
        return str(save_path)

    def compute_fit_mse_range(self, fit_info, min_ratio=0.0, max_ratio=0.1):
        if "fit_func" not in fit_info:
            raise ValueError("fit_info must contain a fit_func")
        actual_ratios   = np.array([r["wl_voltage"] / self.config.vdd for r in fit_info["results"]])
        actual_currents = np.array([r.get("max_current", r["avg_current"]) for r in fit_info["results"]])
        mask = (actual_ratios >= min_ratio) & (actual_ratios <= max_ratio)
        if not np.any(mask):
            raise ValueError(f"No fit data within ratio range [{min_ratio}, {max_ratio}]")
        predictions = fit_info["fit_func"](actual_ratios[mask])
        return float(np.mean((actual_currents[mask] - predictions) ** 2))


# ─── config fall-back helper ─────────────────────────────────────────────────
def _load_default_config():
    """Try to load global config from YAML; fall back to default GlobalConfig."""
    try:
        from sram_compiler.config_yaml.config import SRAM_CONFIG
        sc = SRAM_CONFIG()
        sc.load_all_configs(
            global_file=os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/global.yaml"),
            circuit_configs={
                "SRAM_6T_CELL":  os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/sram_6t_cell.yaml"),
                "SRAM_10T_CELL": os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/sram_10t_cell.yaml"),
                "WORDLINEDRIVER": os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/wordline_driver.yaml"),
                "PRECHARGE":     os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/precharge.yaml"),
                "COLUMNMUX":     os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/mux.yaml"),
                "SENSEAMP":      os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/sa.yaml"),
                "WRITEDRIVER":   os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/write_driver.yaml"),
                "DECODER":       os.path.join(_PROJECT_ROOT, "sram_compiler/config_yaml/decoder.yaml"),
            },
        )
        return sc.global_config
    except Exception as e:
        print(f"[EquivCircuit] Could not load config from YAML ({e}), using defaults (TT corner, 1V)")
        from sram_compiler.config_yaml.config import GlobalConfig
        return GlobalConfig({})


# ─── factory helpers ──────────────────────────────────────────────────────────
def _build_tester_from_core(core, cell_type):
    """Build a SRAMCellParasiticTester from a Sram6TCore / Sram10TCore instance."""
    # Prefer config stored on the core (if testbench passed it), else load from YAML
    global_config = getattr(core, "global_config", None)
    if global_config is None:
        global_config = _load_default_config()

    kwargs = {
        "cell_type":      cell_type,
        "config":         global_config,
        "pd_nmos_model":  core.pd_nmos_model,
        "pu_pmos_model":  core.pu_pmos_model,
        "pg_nmos_model":  core.pg_nmos_model,
        "pd_width":       core.pd_width,
        "pu_width":       core.pu_width,
        "pg_width":       core.pg_width,
        "length":         core.length,
        "q_init_val":     core.q_init_val,
    }

    if cell_type == "10T":
        kwargs["fd_nmos_model"] = core.fd_nmos_model
        kwargs["fd_width"]      = core.fd_width

    return SRAMCellParasiticTester(**kwargs)


# ─── WL-controlled static power source (ported from OpenYield2.5) ─────────────
def _add_wl_controlled_static_power(core, tester, rows_with_unused,
                                    wl_c, bl_c, blb_c, wl_bl_c, wl_blb_c):
    """Add per-row WL-voltage-dependent static-power current sources.

    For each row containing unused cells, build a behavioral current source
    BIWL_POWER_{row} VDD VSS I={count * f(V(WL{row})/vdd)} where f() is a
    piecewise-linear lookup of static current vs WL ratio, fitted by sweeping
    WL.  This makes the unused-cell static power respond to WL activity during
    a write, instead of using a single fixed resistor.
    """
    print(f"[DEBUG] Parasitic caps (write_power_model): wl_c={wl_c:.3e}F "
          f"bl_c={bl_c:.3e}F blb_c={blb_c:.3e}F wl_bl_c={wl_bl_c:.3e}F "
          f"wl_blb_c={wl_blb_c:.3e}F")
    try:
        fit_info = tester.fit_static_power_vs_wl(
            wl_ratios=np.linspace(0.0, 1.0, 100),
            bl_voltage=0,
            blb_voltage=tester.config.vdd,
            q_state=1,
            break_on_decrease=True,
        )
        vdd_value = float(tester.config.vdd)
        sim_results = fit_info["results"]
        wl_ratios = np.array([r["wl_voltage"] / tester.config.vdd for r in sim_results])
        avg_currents = np.array([r["avg_current"] for r in sim_results])
        if len(wl_ratios) < 2:
            raise ValueError("WL static power table requires at least two points.")

        # Build a piecewise-linear lookup expression from simulation points.
        # if(x<=x0, y0, if(x<=x1, y0+m0*(x-x0), ... , yn))
        wl_ratio_var = f"(V(WL{{row}})/{vdd_value:.12e})"
        segment_exprs = []
        for i in range(len(wl_ratios) - 1):
            x0 = float(wl_ratios[i])
            x1 = float(wl_ratios[i + 1])
            y0 = float(avg_currents[i])
            y1 = float(avg_currents[i + 1])
            if x1 == x0:
                raise ValueError(f"Duplicate WL ratio point in table at index {i}: {x0}")
            m = (y1 - y0) / (x1 - x0)
            segment_exprs.append((x1, y0, m, x0))

        first_x = float(wl_ratios[0])
        first_y = float(avg_currents[0])
        last_y = float(avg_currents[-1])

        def _build_table_expr(row):
            x_expr = wl_ratio_var.format(row=row)
            expr = f"{last_y:.12e}"
            for x1, y0, m, x0 in reversed(segment_exprs):
                expr = (f"if({x_expr}<={x1:.12e}, "
                        f"({y0:.12e}+{m:.12e}*({x_expr}-{x0:.12e})), "
                        f"{expr})")
            expr = f"if({x_expr}<={first_x:.12e}, {first_y:.12e}, {expr})"
            return expr

        for row in rows_with_unused:
            count_unused = core._count_unused_cells_in_row(row)
            if count_unused == 0:
                continue
            table_expr = _build_table_expr(row)
            core.raw_spice += (
                f"BIWL_POWER_{row} VDD VSS I={{{count_unused:.12e}*({table_expr})}}\n"
            )
    except Exception as e:
        print(f"[WARNING] Failed to add WL power current source model: {e}")


# ─── main equivalent-circuit builder ─────────────────────────────────────────
def _add_equivalent_circuit_impl(core, cell_type):
    """Insert parasitic equivalent circuit for all unused cells in *core*.

    Uses the 5-cap model (c_bl, c_blb, c_wl, c_wl_bl, c_wl_blb) from
    OpenYield2.5.  Cross-coupling caps are added between WL and BL/BLB
    mid-nodes.  Cell caps are placed at the RC mid-node for physical accuracy.
    """
    from .base_subcircuit import BaseSubcircuit
    pi_res = getattr(core, "pi_res", BaseSubcircuit.DEFAULT_PI_RES)
    pi_cap = getattr(core, "pi_cap", BaseSubcircuit.DEFAULT_PI_CAP)

    tester = _build_tester_from_core(core, cell_type)

    # ── 5-cap parasitic extraction ────────────────────────────────────────────
    nominal  = tester.extract_parasitic_caps(0.6)
    wl_c     = nominal["caps"]["c_wl"]
    bl_c     = nominal["caps"]["c_bl"]
    blb_c    = nominal["caps"]["c_blb"]
    wl_bl_c  = nominal["caps"]["c_wl_bl"]
    wl_blb_c = nominal["caps"]["c_wl_blb"]
    print(
        f"[DEBUG] Parasitic caps: c_wl={wl_c:.3e}F  c_bl={bl_c:.3e}F  "
        f"c_blb={blb_c:.3e}F  c_wl_bl={wl_bl_c:.3e}F  c_wl_blb={wl_blb_c:.3e}F"
    )

    # ── unused-cell bookkeeping (mode-aware, matches OpenYield2.5) ─────────────
    rows_with_unused = [r for r in range(core.num_rows)
                        if core._count_unused_cells_in_row(r) > 0]
    cols_with_unused = [c for c in range(core.num_cols)
                        if core._count_unused_cells_in_col(c) > 0]

    write_power_model = getattr(core, "write_power_model", False)

    # ── static power ──────────────────────────────────────────────────────────
    if write_power_model:
        # WL-controlled behavioral current source per row (matches OpenYield2.5):
        # static current of unused cells in a row varies with that row's WL voltage.
        _add_wl_controlled_static_power(core, tester, rows_with_unused, wl_c,
                                        bl_c, blb_c, wl_bl_c, wl_blb_c)
    else:
        total_unused = core._count_total_unused_cells()
        if total_unused > 0:
            vdd_r = tester.get_static_power_r()["R"]
            core.R("res_static_power", core.NODES[0], core.NODES[1],
                   vdd_r / total_unused @ u_Ohm)
            print(f"[DEBUG] Static power: R={vdd_r:.3e}Ω  total_unused={total_unused}")
        else:
            print("[DEBUG] No unused cells; skipping static power resistor.")

    if not core.w_rc:
        return

    # ── RC model for WL lines of rows containing unused cells ──────────────────
    for row in rows_with_unused:
        row_unused = core._count_unused_cells_in_row(row)
        wl_mid = f"WL{row}_rc_mid"
        core.C(f"cap_WL{row}",      wl_mid,     core.NODES[1], pi_cap * row_unused)
        core.R(f"res_WL{row}",      f"WL{row}", wl_mid,        pi_res / row_unused)
        core.C(f"cap_WL{row}_cell", wl_mid,     core.NODES[1], wl_c   * row_unused)

    # ── RC model for BL/BLB lines of cols containing unused cells ──────────────
    for col in cols_with_unused:
        col_unused = core._count_unused_cells_in_col(col)
        bl_mid  = f"BL{col}_rc_mid"
        blb_mid = f"BLB{col}_rc_mid"
        core.C(f"cap_BL{col}",       bl_mid,      core.NODES[1], pi_cap * col_unused)
        core.C(f"cap_BLB{col}",      blb_mid,     core.NODES[1], pi_cap * col_unused)
        core.R(f"res_BL{col}",       f"BL{col}",  bl_mid,        pi_res / col_unused)
        core.R(f"res_BLB{col}",      f"BLB{col}", blb_mid,       pi_res / col_unused)
        core.C(f"cap_BL{col}_cell",  bl_mid,      core.NODES[1], bl_c  * col_unused)
        core.C(f"cap_BLB{col}_cell", blb_mid,     core.NODES[1], blb_c * col_unused)

    # ── Cross-coupling WL↔BL and WL↔BLB (only for actually-unused cells) ──────
    for row in rows_with_unused:
        wl_mid = f"WL{row}_rc_mid"
        for col in cols_with_unused:
            if core._is_unused_cell(row, col):
                bl_mid  = f"BL{col}_rc_mid"
                blb_mid = f"BLB{col}_rc_mid"
                core.C(f"cap_WL{row}_BL{col}",  wl_mid, bl_mid,  wl_bl_c)
                core.C(f"cap_WL{row}_BLB{col}", wl_mid, blb_mid, wl_blb_c)


# ─── public API ───────────────────────────────────────────────────────────────
def add_6t_equivalent_circuit(self):
    """Bound method for Sram6TCore: add equivalent circuits for unused 6T cells."""
    _add_equivalent_circuit_impl(self, "6T")


def add_10t_equivalent_circuit(self):
    """Bound method for Sram10TCore: add equivalent circuits for unused 10T cells."""
    _add_equivalent_circuit_impl(self, "10T")
