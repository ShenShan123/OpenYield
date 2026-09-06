# OpenYield SRAM Circuit & Testbench Review

**Date:** 2026-09-05
**Scope:** 6T / 10T SRAM cores, equivalent-circuit model, timing-generation circuit,
DC (SNM) and transient (read / write / read&write) testbenches, measurement setup,
and simulation-result parsing — across multiple array sizes.
**Commit reviewed:** `578d53d` (branch `main`, clean tree)

---

## 0. Method

Three passes:

1. **Source read** of `sram_compiler/subcircuits/*`, `sram_compiler/testbenches/*`,
   `utils.py`, `config.py`, `main_sram.py`, `equivalent_modeling/main_sram.py`,
   `size_optimization/exp_utils.py`.

2. **Netlist generation** with PySpice 1.5, covering:

   | axis | values exercised |
   |---|---|
   | operation | `read`, `write`, `read&write`, `hold_snm`, `read_snm`, `write_snm` |
   | cell | `SRAM_6T_CELL`, `SRAM_10T_CELL` |
   | column mux | `choose_columnmux = True` / `False` |
   | array size | 2x1, 4x2, 8x4, 12x4, 16x1, 16x12, 16x16, 32x8, 64x32 |

   Automated checks over the generated decks: `.subckt` port count vs. instance node
   count (per hierarchy scope), node-degree / floating-node detection, and inspection of
   every emitted `.MEASURE`, `.PRINT`, `.IC`, `.TRAN`, `.DC`.

3. **Live Xyce runs** in the `openyield` conda environment
   (Xyce 7.4.0-36, Python 3.9.19), 8x4 6T array, `target_row=7`, `target_col=3`,
   `w_rc=False`, `real_cell_mode=0`, `mc_runs=1`, 25 C:
   `read`, `write`, and `read` with the column mux enabled — plus an instrumented
   re-run printing `I(VVDD)`, `V(CS)`, `V(CS_BAR)`, `V(S_EN)`, `V(SA_Q3)`, `V(BL3)`,
   `V(PRE)` to locate the root cause of the bad measurements.

Findings are tagged:

* **[measured]** — reproduced in a live Xyce run; numbers quoted below.
* **[confirmed]** — reproduced in a generated netlist or by executing the code.
* **[static]** — reasoned from source + netlist, not exercised at runtime.

### Headline result

An 8x4 6T array, default settings, produces:

| operation | returned delay | PAVG | PSTC | PDYN |
|---|---|---|---|---|
| `read`  | **-5.108e-09 s** (negative) | 6.83e-05 W | 3.23e-04 W | **-2.55e-04 W** (negative) |
| `write` | 2.069e-10 s | 1.005e-04 W | 3.47e-04 W | **-2.46e-04 W** (negative) |

The read delay is negative, both dynamic-power figures are negative, and the write delay
is 2.2x the value of the correctly-formulated `TWRITE_TOTAL` measure that the code
computes and then discards (206.9 ps reported vs. 94.8 ps measured). Root causes are
findings 11a, 11b and 12 below.

---

## P0 — Netlist is structurally broken in these configurations

### 1. Column mux instantiated with two extra ports  **[confirmed]**

`sram_compiler/testbenches/sram_6t_core_testbench.py:489` builds the mux with
`use_external_selb=False`, so `ColumnMux` declares **10** ports:

```spice
.subckt COLUMNMUX2 VDD VSS SA_IN SA_INB SEL0 SEL1 BL0 BL1 BLB0 BLB1
```

but `sram_6t_core_testbench.py:507` still passes `SELB0 SELB1` in the connection list,
giving **12** nodes:

```spice
XCOLUMNMUX2_0 VDD VSS SA_IN0 SA_INB0 SEL0 SEL1 SELB0 SELB1 BL0 BL1 BLB0 BLB1 COLUMNMUX2
```

Reproduced in every muxed configuration generated (16x16, 32x8, 64x32; `read` and
`read&write`). Xyce rejects a subcircuit instance whose node count does not match the
definition. Even if a tool tolerated it, `BL0`/`BL1` would bind to the `SELB` nets.

Note the mux already generates `SELB` internally from `SEL` (`mux_and_sa.py:66-72`), so
the top-level `VSELB_*` sources at `sram_6t_core_testbench.py:530-545` are redundant in
either resolution.

**Effect: the entire read path with `choose_columnmux=True` cannot simulate.**
**[measured]** — an 8x4 read with `choose_columnmux=True` aborts:

```
Netlist error: Number of nodes for subcircuit instance XCOLUMNMUX2_0 does not
 agree with number of nodes in subcircuit COLUMNMUX2
Netlist error: Number of nodes for subcircuit instance XCOLUMNMUX2_1 does not
 agree with number of nodes in subcircuit COLUMNMUX2
Simulation aborted due to error.  There are 0 MSG_FATAL errors and 2 MSG_ERROR errors
```

---

### 2. `D_LATCH` input is a floating gate-only node  **[confirmed]**

`sram_6t_core_testbench.py:441` hardwires the latch input to `SA_Q{target_col}`, and
`:1009` calls `create_D_latch()` unconditionally for every operation.

* **`write`** (any size, mux on or off): no sense amplifiers exist at all.
  ```spice
  XD_LATCH VDD VSS SA_Q15 S_EN OUT OUT_B D_LATCH
  ```
  `SA_Q15` connects to nothing else. Reproduced in the 16x16, 4x2 and 16x16-10T write decks.
* **`read` + mux**: only `SA_Q0 .. SA_Q{num_cols/mux_in - 1}` exist. With 16 columns and
  `mux_in=2` there are 8 sense amps but the latch asks for `SA_Q15`.
  Reproduced in 16x16 and 16x12 muxed reads.

The correct index is `target_col // mux_in`, which the `.MEASURE` and `.PRINT` code at
`sram_6t_core_MC_testbench.py:180-236` already uses.

Inside `D_LATCH` the `D` node feeds only `Pinv` and `PNAND2` gates
(`standard_cell.py:330-334`), so there is no DC path to ground.

**[measured]** Xyce does **not** abort — it resolves the floating node via gmin and the
8x4 write completes normally. So this is not a hard blocker; it is worse in a sense:
`V(OUT)`, which is printed and used by the `read&write` `TVOUT_PERIOD` measure, is an
undefined value that no one is warned about. Severity: P1, not P0.

This is the default path of `main_sram.py`, which runs `operation = 'write'`.

---

### 3. `choose_columnmux=True` with `num_cols < mux_in` produces no read path  **[confirmed]**

`sram_6t_core_testbench.py:501` and `:576` loop over `range(self.num_cols // self.mux_in)`.
For `num_cols=1`, `mux_in=2` this is `range(0)`.

Generated 16x1 + mux deck contains **zero** `XCOLUMNMUX*` and **zero** `XSENSEAMP*`
instances, while `.MEASURE TSA` and `.MEASURE TREAD_TOTAL` still reference `V(SA_Q0)`.

---

### 4. Malformed `.PRINT` — two signals concatenated without a separator  **[confirmed]**

`sram_compiler/testbenches/sram_6t_core_MC_testbench.py:226` — the f-string concatenation
drops the space:

```spice
.PRINT TRAN FORMAT=NOINDEX V(S_EN) V(WL15) V(DEC_WL15)V(BL15) V(BLB15) ...
```

Present in all `read` decks. The `write` equivalent at `:314` is correctly spaced — only
the read path is affected.

---

### 5. Missing dependency breaks the import chain  **[confirmed]**

`sram_compiler/testbenches/snm.py:8`

```python
from regex import T
```

is an unused stray import of the third-party `regex` package, which is **not listed in
`environment.yml`**. `sram_6t_core_MC_testbench.py:8` imports `snm`, so a clean
environment built from `environment.yml` cannot import the MC testbench at all.

---

## Timing circuits

**The self-timed core is sound.** Verified by hand against the generated netlist:

* `wl_en = cs & ~clk` — access occurs in the clock-low half cycle.
* `s_en  = rbl_delay & gated_clk_bar & we_bar` (`time_generate.py:717-721`).
* `w_en  = rbl_delay_bar & gated_clk_bar & we` (`time_generate.py:700-704`).
* `PRE   = ~(gated_clk_buf & rbl_delay & wl_en_bar)` (`time_generate.py:737-741`).
* The replica bitline correctly self-terminates both the write pulse and the precharge pulse.
* `dff` (`time_generate.py:181-247`) is a correct positive-edge-triggered master-slave
  (master transparent on `CLK=0` via `tg1`, slave on `CLK=1` via `tg3`, `Q = D`).
* `DECODER_CASCADE` is **functionally correct**. The address mapping was hand-verified
  for 2, 4, 8, 12, 16, 32, 64 and 128 rows, including the `A2<-A0` port reversal at
  `decoder.py:218` and the `VSS` tie-off of unused address bits at `decoder.py:180`.

The problems are around that core.

---

### 6. Column-mux `SEL` is a free-running pulse, not derived from the clock  **[confirmed]**

`sram_6t_core_testbench.py:530-545` drives `SEL`/`SELB` from independent pulse sources
with `delay_time = self.t_pulse` (6 ns) and `pulse_width = self.t_pulse` (6 ns):

```spice
VSEL_1  SEL1 VSS PULSE(0V 1.0V 6ns 0.1ns 0.1ns 6ns 10ns)   ; high 6-12 ns
gated_clk_bar  (WL access window)                          ; high 8-13 ns
```

`SEL` turns on 2 ns *before* the access (during precharge) and turns off 1 ns *before* the
access ends. `t_pulse` has no relationship to the clock and is unused elsewhere, so any
change to `t_period`, array size, or PVT slides the select window out of the access window
silently. Every other control signal is generated inside `TIME`; this one is not.

---

### 7. CS start-up clamp fights the DFF through the first capture edge  **[confirmed]**

`sram_6t_core_testbench.py:109`:

```python
release_time = 1.0 @ u_ns + 0.2 * self.t_period + 2 * self.t_rise   # = 3.2 ns
```

The first clock rising edge is at `1 ns + 0.2 * t_period = 3.0 ns`:

```spice
VCLK    clk      VSS PULSE(0V 1.0V 3.0ns 0.1ns 0.1ns 5.0ns 10ns)
VCSINIT cs_init  VSS PULSE(1.0V 0V 3.2ns 0.02ns 0.02ns 20ns 40ns)
```

`MCSINIT_N` (w = 1.44 um) holds `cs` at 0 for 200 ps *after* the DFF has captured `csb=0`
and is driving `cs` high through a 1.08 / 0.36 um inverter. Result: crowbar current plus
~200 ps of skew injected into cycle 1 — the cycle from which every measurement is taken.

**[measured]** The instrumented 8x4 read shows `V(CS) = 0.064 V` while `V(CS_BAR) = 1.000 V`
at `t = 0`, with `I(VVDD)` at **-322.7 uA** versus **-3.0 uA** in a genuinely idle window.
This contention is what corrupts the `PSTC` measurement — see finding 13a.

Separately, `VCSB` pulses high only over 1-2 ns and 11-12 ns, i.e. never at a clock edge,
so `csb` is effectively a no-op stimulus.

---

### 8. Clock period is hardcoded at 10 ns for every array size, with no feedback  **[confirmed]**

`sram_compiler/testbenches/base_testbench.py:31-36` sets `t_period = 10 ns`,
`t_pulse = 6 ns`, `t_rise = t_fall = 0.01 * t_period`.
`set_timing_parameters()` exists at `base_testbench.py:43` but **is never called anywhere
in the repository** (verified by grep).

`.TRAN` is always `1 ns + 2 * t_period`, giving exactly one complete access window
(8-13 ns) for read and write. `run_mc_simulation` computes and prints
`CLK(min) in this size and PVT` (`sram_6t_core_MC_testbench.py:868`) but discards it.

For a large array where the access does not complete inside the 5 ns clock-low window,
the `.MEASURE` statements simply fail and become `0.0` — see finding 14.

The symptom is already acknowledged in the source: `time_generate.py:680` contains a
hardcoded escape hatch for exactly one configuration.

```python
if operation == 'write' and self.num_rows == 16 and self.num_cols == 512:
    wen_delaychain = WenDelayChain(stages=6, loads_per_stage=4, w_rc=w_rc)
```

---

### 9. Replica column is tied to the real wordlines  **[confirmed]**

`sram_6t_core_testbench.py:180-186` connects the replica column's `num_rows + 1` cells to
`RWL, WL0 ... WL{num_rows-1}`:

```spice
Xsram_17x1_replica_column VDD VSS RBL RBLB RWL WL0 WL1 ... WL15 sram_17x1_replica_column
```

Every cell is a full `Replica_CELL` with a live access transistor
(`replica_column.py:78-79`), so on each access **both** the `RWL` replica cell **and** the
selected row's replica cell discharge `RBL`. The replica delay — which sets `s_en`, `w_en`
and precharge termination — is therefore roughly 2x too fast. The error is deterministic
(always exactly two cells), so it is a systematic optimism rather than noise.

In an OpenRAM-style replica column, the non-replica cells are dummies with no access path.

Also: the instance name is hardcoded to `'sram_17x1_replica_column'`
(`sram_6t_core_testbench.py:194`) while the subcircuit name is
`sram_{num_rows+1}x1_replica_column` — these only coincide at 16 rows.

---

### 10. Supply voltage hardcoded to 1.0 V in the stimulus  **[confirmed]**

`VCLK`, `VCSB`, `VWEB`, `VSEL_*` and `VSELB_*` use a literal `1.0 @ u_V` instead of
`self.vdd` (`sram_6t_core_testbench.py:530-545`, `:1046-1093`). Harmless today because
`global.yaml` sets `vdd: 1.0`, but any voltage-scaling study will drive the control path
at 1.0 V while the array runs at a different rail.

---

## Measurement correctness

### 11a. `TSA` and `TREAD_TOTAL` latch onto numerical noise, not the sense event  **[measured]**

This is the cause of the negative read delay. Measured `.mt0` from the 8x4 read:

```
TWL         =  8.286266e-09     <- wordline actually rises at 8.29 ns
TBL         =  8.305074e-09
TSA         = -5.256937e-09     <- negative
TREAD_TOTAL = -4.996334e-09     <- negative
```

Both bad measures share the same target:

```spice
TARG V(SA_Q3)=0.01 FALL=1        ; 0.01 * VDD
```

`add_meas_and_print` sets `.IC V(SA_Q{col}) = 0` (`sram_6t_core_MC_testbench.py:113-118`).
The instrumented re-run shows `V(SA_Q3)` then **parks between 9.07 mV and 9.98 mV for
~3 ns**, i.e. immediately below the 10 mV target threshold, until precharge fires:

```
     TIME      V(SA_Q3)      V(BL3)       V(PRE)
 2.900e-09   9.0703e-03   3.8434e-03   1.0000e+00
 3.000e-09   9.3751e-03   3.9715e-03   1.0000e+00
 3.100e-09   9.6790e-03   4.0992e-03   1.0000e+00
 3.200e-09   9.9821e-03   4.2298e-03   1.0000e+00
 3.300e-09   1.3667e-01   6.7082e-01   2.9770e-02   <- precharge starts
 3.400e-09   9.9610e-01   9.9654e-01   1.9580e-04
```

Any numerical ripple in that band registers as a `FALL=1` crossing at ~3.03 ns, five
nanoseconds *before* the trigger. The real sense event is unambiguous and happens much
later:

```
     TIME      V(S_EN)     V(SA_Q3)      V(BL3)
 8.200e-09  1.3974e-05   1.0080e+00   1.0080e+00
 8.350e-09 -8.8263e-03   4.6267e-01   7.1275e-02
 8.500e-09 -5.3429e-03   3.7140e-01   3.5934e-03
 8.650e-09  9.9999e-01   6.3569e-05   2.0838e-04   <- s_en fires, SA resolves
```

True `TSA` is on the order of 50-100 ps; the deck reports -5.26 ns.

Two independent fixes are needed: the target threshold must not sit inside the band where
`.IC` parks the node (use e.g. `VDD/2` with the correct edge), and the `.IC` value should
not place `SA_Q` on top of the threshold.

Knock-on effect: `parse_mc_measurements` (`utils.py:70-77`) **discards every MC run** when
`num_runs > 1`, because it skips any run with a negative `TSA`, `TS_EN` or `TSWING`. With
`TSA` negative on every run, a real Monte Carlo sweep yields an empty result set.

### 11b. `TS_EN` measures between mismatched edge numbers  **[measured]**

`sram_6t_core_MC_testbench.py:186-188`:

```spice
.meas TRAN Ts_en TRIG V(S_EN)=0.02 RISE=2 TARG V(S_EN)=0.25 RISE=1
```

The trigger uses the **second** rising crossing while the target uses the **first**. In
the 8x4 run this happened to return a plausible `1.146951e-11` (11.5 ps) — `s_en` bounces
across 0.02 V twice on its way up, so `RISE=2` at 0.02 V still lands on the same edge as
`RISE=1` at 0.25 V.

That is luck, not correctness: if `s_en` rises monotonically, `RISE=2` at 0.02 V lands on
the *next* read cycle (~19 ns) and the measure returns roughly -10 ns. For a slew
measurement both crossings must be `RISE=1`.

---

### 12. Delay is a sum of overlapping and unrelated segments; the correct totals are discarded

`sram_6t_core_MC_testbench.py:884-889`:

```python
if operation == 'read':
    delay = TDECODER + TPRCH + TSA + TSWING + TS_EN + TWLDRV
else:
    delay = TDECODER + TWDRV + TWLDRV + TWRITE_Q
```

* `TPRCH` is precharge recovery time and is not part of the read access path.
* `TDECODER` is measured from `A_dff0` (t ~ 3 ns, at the capture edge) while the access
  starts at 8 ns — it is off the critical path entirely. The decoder is purely
  combinational on `A_dff` (`decoder.py:186`, level-0 `EN` tied to `VDD`); gating happens
  in the wordline driver via `WL_EN`.
* `TS_EN` is the negative value from finding 11.

Meanwhile `TREAD_TOTAL` (`wl_en -> SA_Q`, `:200-203`) and `TWRITE_TOTAL`
(`wl_en -> Q@90%`, `:287-290`) are correctly formulated and never used.

**[measured]** The 8x4 write `.mt0` shows the size of the error directly:

```
TDECODER     = 4.293588e-11
TWLDRV       = 1.792476e-11
TWDRV        = 6.920793e-11
TWRITE_Q     = 7.685762e-11
TWRITE_TOTAL = 9.478238e-11    <- correct end-to-end delay, discarded
```

The code returns `TDECODER + TWDRV + TWLDRV + TWRITE_Q = 206.9 ps`, while the
correctly-formulated `TWRITE_TOTAL` is `94.8 ps`. **The write delay is over-reported by
2.2x** because the segments overlap and `TDECODER` is not on the critical path at all.

`TDECODER` additionally only measures anything when bit 0 of `target_row` is 1 — every
entry point uses `target_row = num_rows - 1`, so it works for even row counts and silently
fails for odd ones.

---

### 13. `read&write` returns delay = 0 and power = 0  **[confirmed]**

`add_meas_and_print` for `read&write` emits only `TVOUT_PERIOD` and `PAVG`
(`sram_6t_core_MC_testbench.py:370-379`). Confirmed in the generated deck:

```spice
.meas TRAN TVOUT_PERIOD TRIG V(OUT)=0.5 RISE=1 TARG V(OUT)=0.5 RISE=2
.meas TRAN PAVG AVG {-V(VDD)*I(VVDD)} FROM=1e-09 TO=1.21e-07
```

But the return path (`:884-895`) asks for `TDECODER`, `TWDRV`, `TWLDRV`, `TWRITE_Q`,
`PSTC` and `PDYN`, none of which exist. `_col()` (`:880`) returns `np.array([0.0])` for a
missing column, so `main_sram.py` prints `Delay(0.00 ns), Power(0.00 mW)` with no error.

The `PAVG` window also runs past the end of the simulation:

```spice
.TRAN 1.0000e-11 8.1000e-08                 ; 1 ns + 8 * t_period  = 81 ns
.meas TRAN PAVG AVG ... FROM=1e-09 TO=1.21e-07   ; 1 ns + 12 * t_period = 121 ns
```

(`:335` uses `8*t_period` for `.TRAN`, `:378` uses `12*t_period` for the measure.)

---

### 13a. `PSTC` is measured during the start-up transient, so `PDYN` comes out negative  **[measured]**

`sram_6t_core_MC_testbench.py:204-213` measures "static" power over 0.1-0.9 ns:

```spice
.meas TRAN PSTC AVG {-V(VDD)*I(VVDD)} FROM=1.0e-10 TO=9.0e-10
.meas TRAN PDYN PARAM={PAVG-PSTC}
```

That window is not static. It sits inside the operating-point release and the CS start-up
clamp contention (finding 7). Instrumented 8x4 read:

| window | mean `I(VVDD)` |
|---|---|
| 0.1-0.9 ns (the `PSTC` window) | **-3.228e-04 A** |
| 4.0-7.0 ns (genuinely idle) | **-3.019e-06 A** |
| 2.0-12.0 ns (the `EREAD` window) | -6.816e-05 A |

The `PSTC` window reads **107x** the true quiescent current. At `t = 0` the probe shows
`V(CS) = 0.064 V` against `V(CS_BAR) = 1.000 V` — the clamp NMOS and the DFF output
inverter are both driving `cs`.

Since `PSTC` (3.23e-04 W) exceeds `PAVG` (6.83e-05 W), `PDYN = PAVG - PSTC` is negative in
**both** read and write (-2.55e-04 W and -2.46e-04 W). `main_sram.py:98` reports
`power = pstc + pdyn`, which happens to cancel back to `pavg`, so the top-level number
survives — but `PSTC` and `PDYN` individually are meaningless, and
`size_optimization/exp_utils.py:1160` consumes `pavg` only, so the breakdown is silently
wrong wherever it is used.

The window must be moved into a genuinely quiescent interval (e.g. 4-7 ns for the default
timing), after the clamp releases and before the access.

### 14. `_col()` turns any failed measurement into 0.0

`sram_6t_core_MC_testbench.py:880-881`:

```python
def _col(name, default=0.0):
    return df[name].values if name in df.columns else np.array([default])
```

This is the general form of finding 13. A Xyce `FAILED` measure, or a measure whose
trigger never fires (large array, insufficient clock period, missing node), does not
raise — it drops out of the CSV and contributes `0.0` to the delay sum.

There is no "did the operation actually complete" assertion anywhere in the flow. Combined
with finding 8, this is the mechanism by which a large-array run would report a *better*
delay than a small one.

---

### 14a. `Sram6TCoreMcTestbench` defaults `sweep_senseamp=True`, which then crashes the parser  **[measured]**

`sram_6t_core_MC_testbench.py:20` defaults `sweep_senseamp=True`, while the parent
`Sram6TCoreTestbench.__init__` (`sram_6t_core_testbench.py:16`) defaults the same flag to
`False`. Every other `sweep_*` flag defaults to `False` in both classes.

Any caller that does not pass the flag explicitly therefore silently gets a 3-point
`.STEP data=SENSEAMP` sweep (`sa.yaml:12,20,28` each list three values). Xyce then emits
three result blocks while `num_mc` is still 1, and `split_blocks` hard-fails:

```
ValueError: Auto-split block count (3) does not match specified num_mc (1)
```

Confirmed on an 8x4 read: Xyce produced `.mt0`, `.mt1`, `.mt2` and a `.res` sweep table of
three steps. `main_sram.py` happens to pass `sweep_senseamp=False` explicitly, which is
why this is not seen there.

### 15. `1/2CLK` margin logic is inverted

`sram_6t_core_MC_testbench.py:858-863`:

```python
if std_val > 0.1e-9 or pd.isna(std_val) or std_val == float('inf') ...:
    half_clk_sum += mean_val            # large spread -> ignore the spread
else:
    half_clk_sum += mean_val + std_val  # small spread -> add the spread
```

A large spread should widen the margin, not be discarded. With `mc_runs=1` the `std` is
NaN and the mean-only branch is the intended path, so this only bites in real Monte Carlo.

---

### 16. Waveform plot requests the wrong sense-amp column  **[confirmed]**

`sram_6t_core_MC_testbench.py:773` puts `V(SA_Q{target_col})` into `selected_columns`,
while `.PRINT` (`:232-236`) emits `V(SA_Q{target_col // mux_in})`. In muxed reads no
requested column matches, and `visualize_results` (`utils.py:349-352`) silently falls back
to plotting *all* columns with only a printed warning.

---

## Simulation-result parsing

### 17. `.prn` reader requires an `Index` column that `FORMAT=NOINDEX` removes

`utils.py:165` hard-raises unless the first header is `INDEX`:

```python
if headers[0].upper() != 'INDEX':
    raise ValueError(f"First column must be 'INDEX', got '{headers[0]}'")
```

Yet every `.PRINT` in `add_meas_and_print` specifies `FORMAT=NOINDEX`. `read` and
`read&write` happen to *also* contain one `.PRINT TRAN` without `FORMAT=NOINDEX`
(`sram_6t_core_MC_testbench.py:230`, `:365`).

**`write` has no such line — all of its `.PRINT` statements are `NOINDEX`** (confirmed in
the generated deck).

**[measured]** Xyce 7.4 nevertheless emitted an `Index` column for the write run, so the
write path currently works. But the resolution is **order-dependent**: an instrumented
copy of the *read* deck, with an extra `FORMAT=NOINDEX` `.PRINT` inserted immediately after
`.TRAN` (i.e. before the existing ones), produced a `.prn` with **no** `Index` column and
would have failed `read_prn_with_preprocess` outright.

So the flow depends on the relative ordering of `.PRINT` lines that the code emits in
several different places. It works today by luck, not by design, and will break the moment
a `.PRINT` is added or reordered. Either drop `FORMAT=NOINDEX` everywhere or stop requiring
an `Index` column in `utils.py:165`.

---

### 18. SNM `.prn` splitter appears incompatible with the `.PRINT` it is fed

`snm.py:28-37` requires exactly 4 whitespace-separated fields and `int(parts[0])`:

```python
parts = s.split()
if len(parts) != 4:
    continue
idx = int(parts[0])
```

The SNM deck emits (confirmed):

```spice
.DC U -0.71 0.71 0.001
.PRINT DC FORMAT=NOINDEX {U} V(V1) V(V2)
```

With the index suppressed the first column is a float (either the DC sweep variable or
`{U}`), so `int()` raises, every line is skipped, `runs == []`, and `build_stats_table`
raises `KeyError`. **[needs Xyce run]** to confirm the actual column layout — but the two
sides are clearly written against different assumptions.

---

### 19. `write_snm` extraction takes a global max over the whole sweep

`snm.py:253-281`. With

```
d = V1 - V2 = 2U + sqrt(2) * (V(QBD) - V(QD))
```

swept over `U` in `[-VDD/sqrt(2), +VDD/sqrt(2)]`, the `2U` term dominates at the sweep
ends, so `max|d| / sqrt(2)` is pinned near the sweep bound rather than to any cell
property. Expect a write SNM close to `VDD` and largely independent of transistor sizing.

The hold / read path (`extract_classical_snm_from_run`, `snm.py:87-178`) is correct:
bounded intervals between zero crossings, `max|d|/sqrt(2)` per interval, minimum over
intervals.

---

### 20. `hold_snm` leaves BL/BLB with no DC path  **[confirmed in netlist]**

`create_single_cell_for_snm` (`sram_6t_core_testbench.py:770-776`) adds a source only for
`WL` in the hold case:

```spice
XSRAM_6T_CELL_DISCONNECT VDD VSS BL BLB WL SRAM_6T_CELL_DISCONNECT
VWL_gnd WL VSS 0V
...
.ic V(BL)=1.0V V(BLB)=1.0V
```

`.IC` is not honoured in a `.DC` sweep. With `WL = 0` the access transistors are off, so
BL and BLB float. `read_snm` and `write_snm` correctly use real voltage sources.
**[needs Xyce run]** to confirm whether this fails outright or converges via gmin at an
arbitrary bias.

---

### 21. `parse_mc_measurements` swallows failures

`utils.py:46`:

```python
value = float(raw_value) if '.' in raw_value or 'e' in raw_value.lower() else int(raw_value)
```

A Xyce `FAILED` token raises `ValueError`, is caught at `:49-51`, printed as
"Ignoring invalid line", and the measurement vanishes — becoming NaN or, via finding 14,
`0.0`.

`utils.py:172` uses `comment='E'` in `pd.read_csv`. This works only because Xyce writes
lowercase `e` exponents; it exists to strip the trailing `End of Xyce...` line and will
silently truncate data rows if that ever changes.

---

## Equivalent circuit

### 22. Parasitic capacitances are only inserted when `w_rc=True`

`sram_compiler/subcircuits/sram_cell_add_equivalent.py:837`:

```python
if not core.w_rc:
    return
```

This early return sits **before** all of the BL / BLB / WL capacitance and the WL-BL
coupling capacitors (`:840-874`). With `w_rc=False` and `real_cell_mode != 0`, omitted
cells contribute **only** a static-power resistor and zero loading.

* `real_cell_mode = 1` (cross): target row and target column are fully real, so the impact
  is mostly on power.
* `real_cell_mode = 2 / 3 / 4`: the target bitline or wordline loses most of its load and
  delay becomes strongly optimistic.

No warning is emitted in either case.

---

### 23. Building the testbench silently launches Xyce — up to ~103 extra runs per build  **[confirmed]**

`_add_equivalent_circuit_impl` (`sram_cell_add_equivalent.py:797-812`) calls
`extract_parasitic_caps()` (3 transient simulations) at **netlist-construction** time,
through PySpice's `XyceServer` — a different invocation path from the
`subprocess.run(['Xyce', ...])` used for the main simulation.

Confirmed: constructing a testbench with `real_cell_mode=1` fails with
`FileNotFoundError: [Errno 2] No such file or directory: 'Xyce'` from
`PySpice/Spice/Xyce/Server.py`.

For `operation='write'` it additionally runs
`fit_static_power_vs_wl(wl_ratios=np.linspace(0, 1, 100))` — **100 more transients**
(`:748`), each 6000 steps. In `size_optimization`, the testbench is rebuilt per candidate.

---

### 24. Static-power model failures are swallowed

`_add_wl_controlled_static_power` (`sram_cell_add_equivalent.py:723-793`) wraps its entire
body in `except Exception` and, on failure, prints a warning and adds **no static-power
source at all**.

Also, `curve_fit` is computed at `:749-758` and then unused: the emitted PWL table is
built from `avg_currents` (`:757`, including points flagged `dropped`), while `fit_mask` —
which excludes them — is only used for the discarded fit.

---

### 25. `_PROJECT_ROOT` resolves one directory too high  **[confirmed]**

`sram_cell_add_equivalent.py:25-27`:

```python
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
```

The file lives at `<repo>/sram_compiler/subcircuits/`, so three `..` land one level *above*
the repository root. `_load_default_config()` therefore always fails and falls back to
`GlobalConfig({})` — **vdd = 1.0, temperature = 27, corner = TT, 32x1** — ignoring
`global.yaml` (which sets `temperature: 25`).

Reached whenever `SRAMCellParasiticTester` is used without an explicit `config`.
The 6T/10T testbench paths do pass `global_config`, so this is currently latent.

---

### 26. `get_static_power_r()` runs with BL/BLB/WL floating

`sram_cell_add_equivalent.py:466-492` pops the port sources via `_remove_port_sources()`
and then simulates. `WL` then connects only to MOS gates, with no DC path to ground.
This is the non-`write_power_model` path — i.e. every `read` with `real_cell_mode != 0`.
**[needs Xyce run]**

---

## Lower severity

* **`w_rc` node substitution is inconsistent across blocks.** `Sram6TCell` /
  `Sram10TCell` compute `q_node`/`qb_node` from the RC helper but the cross-coupled gates
  use literal `'Q'` / `'QB'` (`sram_6t_core.py:110-127`) — this happens to place R between
  driver and load gates, which is defensible. But in `SenseAmp` (`mux_and_sa.py:150-161`),
  `ColumnMux`, and `AND2` (vs. `WordlineDriver`), the RC networks end up as dangling stubs
  with no delay effect. Same flag, three different resulting topologies.
* **Sweep mode silently changes the circuit.** `SenseAmp` uses
  `pmos_width_pass = pmos_width * 4/3` normally but `= pmos_width` in sweep mode
  (`mux_and_sa.py:131-136`); `WordlineDriverFactory` scales the output inverter by
  `max(num_cols, 4) / 4` normally but drops the scaling in sweep mode
  (`parameter_factor.py:311-335`).
* **`data_init()` emits `.IC` for all `rows x cols` cells** regardless of
  `real_cell_mode`, so most `.IC` targets do not exist when the equivalent model is on
  (16x16 mode 1: 256 of 480 statements point at nonexistent nodes).
* **`DummyColumnFactory._get_config` (`parameter_factor.py:426-431`)** reads
  `self.pd_model` / `pg_model` / `pu_model`, which `__init__` never sets -> `AttributeError`.
  Dead code today (`create_dummy_column` is commented out at
  `sram_6t_core_testbench.py:1000`).
* **Two near-identical `config.py` files** — repo root and `sram_compiler/config_yaml/` —
  differing only in how `_PROJECT_ROOT` is derived. `main_sram.py` uses one,
  `equivalent_modeling/main_sram.py` uses the other.
* **`demo_run_a_testbench.py`** has hardcoded `/home/lixy/...` paths and passes
  `param_sweep=False`, which is not a parameter of `Sram6TCoreMcTestbench`
  (`sweep_cell` is) -> `TypeError`.
* **10T `process_parameters.vars` in `sram_10t_cell.yaml` is a flat list**, while
  `gen_process_params` requires a 2-D array (`ndim != 2 -> ValueError`,
  `sram_6t_core_MC_testbench.py:546`). Only reached with `custom_mc=True`. The 6T YAML is
  correctly a list of lists.
* **`temperature` is not passed** in `size_optimization/exp_utils.py:1114-1153`, so all
  optimizer simulations run at the 27 C default instead of the 25 C in `global.yaml`.
  `main_sram.py` does pass it.
* **Cell topologies themselves are correct.** The 6T cell, the 10T Schmitt-trigger cell
  (`NL1`/`NL2`/`NFL` local feedback, bulk ties), and both replica cells were verified
  against the generated netlist — no issues found there.

---

## Suggested fix order

### Blocking — nothing measures correctly until these land

| # | Fix | Effect today |
|---|---|---|
| 1 | Drop `SELB*` from the mux instantiation (`sram_6t_core_testbench.py:507`) | muxed read aborts in Xyce |
| 11a | Move the `TSA` / `TREAD_TOTAL` target off the `.IC` parking level | read delay = **-5.1 ns** |
| 13a | Move the `PSTC` window out of the start-up transient | `PDYN` negative in read *and* write |
| 12 | Return `TREAD_TOTAL` / `TWRITE_TOTAL` instead of summing segments | write delay **2.2x** too large |
| 5 | Delete `from regex import T` (`snm.py:8`) | MC testbench unimportable in a clean env |
| 4 | Add the missing space in the read `.PRINT` (`:226`) | malformed `.PRINT` token |
| 14a | Default `sweep_senseamp=False` in the MC testbench (`:20`) | `split_blocks` crash for any caller that omits the flag |

### High — wrong or undefined results, no error raised

* **2** — index the D-latch as `target_col // mux_in`, and skip it entirely for `write`.
* **11b** — `TS_EN` should use `RISE=1` on both crossings.
* **13** — add the missing `read&write` measures, and fix the `PAVG` window (`12*t_period`
  exceeds the `8*t_period` `.TRAN`).
* **14** — make a missing measurement raise instead of defaulting to `0.0`.
* **3** — reject or handle `num_cols < mux_in`.

### Timing generality

* **6** — derive the column-mux `SEL` from `wl_en` instead of the free-running `t_pulse`.
* **7** — release the CS clamp *before* the first clock edge, not 200 ps after.
* **8** — scale `t_period` from the computed `CLK(min)`; remove the hardcoded 16x512 case.
* **9** — make the non-replica cells in the replica column true dummies.
* **10** — use `self.vdd` in the stimulus instead of literal `1.0 V`.

### Parsing

* **17** — settle on one `.prn` format; today it works only because of `.PRINT` ordering.
* **18** — reconcile `split_xyce_prn_runs` with the actual SNM `.prn` column layout.
* **19** — `write_snm` needs a bounded-interval definition, not a global max.
* **20** — give `hold_snm` real BL/BLB sources; `.IC` does not apply to a `.DC` sweep.

### Equivalent circuit

* **22** — decouple the parasitic caps from the `w_rc` flag.
* **23** — cache the extraction; the 100-point WL sweep runs per testbench build.
* **24** — fail loudly instead of silently omitting the static-power source.
* **25** — fix `_PROJECT_ROOT` (one `..` too many).

### Cleanup

Everything in the "Lower severity" section.

---

## Appendix — environment setup

The `openyield` conda environment was created at
`/proj/workarea/user5/miniconda3/envs/openyield` (Python 3.9.19, Xyce 7.4.0-36,
ngspice 41, PySpice 1.5).

Two deviations from `environment.yml` were required:

1. **`python-graphviz==0.20.3` in the `pip:` section is a conda package name.** PyPI has no
   such distribution, so `conda env create` fails at the pip stage with
   `ERROR: Could not find a version that satisfies the requirement python-graphviz==0.20.3`.
   The PyPI name is `graphviz`. Either move it to the conda `dependencies:` list or rename
   it to `graphviz==0.20.3`.
2. **`regex` is missing from the dependency list** but is imported by `snm.py:8`
   (finding 5). It was installed manually so the environment is usable. The correct fix is
   to delete the unused import rather than add the dependency.

Channels were remapped to the Tsinghua mirror for `defaults` and `conda-forge`;
`vlsida-eda` (which hosts `xyce` and `trilinos`) was left on `conda.anaconda.org`, since
that channel is not mirrored. This also avoids the `repo.anaconda.com` Terms-of-Service
prompt.

**Usage note:** the flow shells out to `Xyce` via `subprocess.run(['Xyce', ...])`
(`sram_6t_core_MC_testbench.py:733`), so the environment must be *activated* — having the
interpreter on `PATH` is not enough:

```bash
conda activate openyield
python main_sram.py
```

---

# Part II — V2.0.1 dynamic verification (2026-09-05)

The findings of Part I were fixed in the working tree and then verified by
simulation across array sizes. This part records the method and the
additional defects found only by running the circuits; the complete list of
changes is in `CHANGELOG.md`.

## Method

A scratch driver (`run_one.py`, outside the repository) builds one
`Sram6TCoreMcTestbench` per configuration with `sim_path` in a scratch
directory, runs `run_mc_simulation()`, and then reads the `.prn` waveform
back and scores it automatically per Monte Carlo block:

| operation | checks |
|---|---|
| `read` | WL > 0.9 VDD; target BL discharges below VDD/2 while BLB stays > 0.75 VDD; `s_en` fires inside the access window and shows no glitch before it; `SA_Q`/`SA_QB` resolve to 0/1; `OUT` falls; no read upset (Q bump < 0.4 VDD, Q/QB unchanged at the end of the access); BL and RBL back above 0.95 VDD at the next access; precharge fired |
| `write` | WL > 0.9 VDD; `w_en` asserted and released together with WL; BLB driven below 0.1 VDD, BL above 0.9 VDD; Q > 0.9 VDD and QB < 0.1 VDD at the end of the access and still at the next access; bitlines and RBL restored |
| `read&write` | Q sequence 1,1,0,0,1,1,0,0 at the end of the 8 access windows; `OUT` follows the reads (1,1,0,0,1,1,0 from cycle 2); `OUT` toggles at least 3 times |

Xyce runs single-threaded (`OMP_NUM_THREADS=1`), ~60-90 decks in parallel on
a 96-core host. Runtime per deck (all cells real, no RC): 8x4 ~1-2 min,
16x16 ~5-10 min, 32x32 ~15 min, 64x64 ~1 h, `read&write` 3-4x a `read`.

Sweeps:

1. **Size sweep** (nominal, `real_cell_mode=0`, `w_rc=False`, 25 °C, TT,
   target = last row / last column): 29 sizes — 1x1, 2x1, 2x2, 3x3, 4x2, 4x4,
   5x3, 6x6, 8x4, 8x8, 12x4, 16x1, 16x8, 16x16, 20x10, 32x8, 32x32, 64x16,
   64x64, 100x50, 128x32, 128x128, 256x8, 256x64, 512x4, 16x256, 8x512, 2x128,
   16x512 — × {6T, 10T} × {read, write, read&write} × {mux off, mux on (even
   column counts)} = 318 runs.
2. **Monte Carlo** (`.SAMPLING`, `vth_std=0.05`, fixed seed): 5 samples at
   8x4, 16x16 and 32x8 for read and write, both cells, mux off/on; 3 samples
   of `read&write` at 8x4; plus the seeded hunts used to reproduce and fix
   the write failures (6 samples × 3 seeds at 4x2 with mux, 12 samples at 4x2
   and 8x4).
3. **Equivalent circuit** (`real_cell_mode=1`, `w_rc=True`, the
   `main_sram.py` defaults) at 16x16, 32x32 and 64x64, both cells, read and
   write; `real_cell_mode` 2/3/4 and `q_init_val=1` and other target cells at
   16x16.
4. SNM sanity: hold/read/write SNM on both cells (6T: 0.325 / 0.182 / 0.365 V,
   10T: 0.485 / 0.290 / 0.419 V).

## Defects found by simulation (in addition to Part I)

| # | finding | evidence | fix |
|---|---|---|---|
| D1 | A single run (`mc_runs=1`) was one random process sample: `.SAMPLING` was always emitted and Xyce seeded itself randomly. | Two identical 8x4 read runs returned 298.4 ps and 301.1 ps; muxed `read&write` at 4x2/4x4 and 6x6 randomly raised `TVOUT_PERIOD FAILED`, then passed on re-run. | `.SAMPLING` only for `mc_runs > 1`; `mc_seed` argument; nominal run bit-for-bit repeatable (3 × 301.1321 ps). |
| D2 | Write pulse terminated by the replica bitline was too short for the write path through the column mux; ~5 % sigma samples failed to write. | Seed 11 sample 0, cycle 3: `w_en` 8.272–8.525 ns, BL only reached 0.397 V, Q stayed 1; seed 33 sample 2 cycle 1: BLB min 0.307 V, Q stayed 0; seed 55 sample 0 likewise. | `w_en = gated_clk_bar & we` for the whole wordline phase; seeds 11/33/55 now pass all 6 samples; 12-sample write sweeps pass. |
| D3 | The stand-alone write testbench had no bitline precharge and no sense-amp / mux load: write delay 96 ps vs 133 ps for the same write inside `read&write`. | 8x4 6T: BL started at 0.003 V (`.IC`), BLB floated at –0.012 V after the write; `TWDRV` measured a BL rise that does not exist in a precharged array. | Full column periphery in every transient deck; `TWDRV` on the driven bitline. |
| D4 | Energy window 2 ns .. 2 ns + T measured the start-up precharge (all bitlines charged from the 0 V `.IC`) and excluded the real post-access precharge. | 8x4 6T read: E(2–3.5 ns) = 156 fJ of E(2–12 ns) = 298 fJ; E(13–14 ns, real precharge) = 109 fJ; E(8–18 ns) = 281 fJ. | Window = one period starting at the access. |
| D5 | `read&write` transient ended at 1 ns + 8 T, cutting the 8th access 1 ns after its wordline rose; `PAVG` window started at 1 ns. | `.TRAN` end 81 ns vs 8th access 78–83 ns. | `.TRAN` to 8.5 T, `PAVG` over 4 T from the first access. |
| D6 | A plain `AND2` for the new `w_en` gate silently replaced the gated-clock `AND2` definition inside `TIME` (PySpice keeps one `.subckt` per name and scope). | Generated deck had one `.subckt AND2` with the `w_en` inverter sizes. | `AND2_WEN` subclass with its own name; a scope-aware duplicate-definition check over the decks now reports 0 collisions (also 0 in the original decks). |
| D7 | 10T core ignored the testbench `pi_res` / `pi_cap`. | Factory call without the two keywords. | Passed in both paths. |
| D8 | With the full-phase `w_en` (D2) the data register update at the edge that ends the write races the driver release: the drivers briefly write the next cycle's data. | `read&write` 64x16, both cells: Q = 1.00 at 12.9 ns, W_EN = 0.85 and DIN_dff = 0.02 at 13.2 ns with WL still 1.00, BL pulled to 0.54 V, Q = 0.00 at 13.4 ns; `TVOUT_PERIOD` FAILED. 8x4-32x8 unaffected (weak row-scaled driver). | Per-column transparent-low data hold latch on `w_en_bar` between `DIN_dff` and the write driver; 64x16 sequence 1,1,0,0 restored, seeded hunts still pass. |
| D9 | `w_en` buffer scaled with columns/64 only although the write drivers scale with `max(8,rows)/16`. | `w_en` fall − WL fall: 32x32 +53 ps, 64x16 +40 ps, 256x8 +72 ps, 64x64 with RC +285 ps (W_EN still 0.5 V at 13.8 ns, WL off at 13.54 ns). | Row factor added to the `w_en` scale (`s_en` keeps columns/64). |
| D10 | `TS_EN` took a precharge-coupling bump on `s_en` (0.2-0.3 V with 32-128 sense amps) as the 20 % crossing. | 32x32 and 2x128 no-mux reads: `TS_EN` = 5.35 ns; s_en = 0.226 V at 3.4 ns while `GATED_CLK_BAR` / `WL_EN` stay at 0. | `TD=` (access start) on trigger and target; 32x32 deck gives 74.6 ps. `FROM=` does not restrict TRIG/TARG measures in Xyce 7.4. |
| D11 | Xyce Newton loop stalls on 512-row decks. | 6T 512x4 read (no mux), 10T 512x4 write (mux) and both 10T 512x4 read&write: "Time step too small" at 8.8-9.0 ns, `F:max s`, 21 iterations, ‖F‖ = 1.02e-12 for every step size down to 1e-22 s. `NONLIN-TRAN ABSTOL=1e-10` does not help; `TIMEINT ERROPTION=1` completes the deck but changes converging decks' delays by 2-15 %. | Not changed by default; documented (see solver notes). |

## Solver notes (Xyce 7.4, 512-row decks)

Four of the twelve nominal 512x4 decks (6T read without mux, 10T write with
mux, 10T read&write with and without mux) stop with `Time step too small`
between 8.8 and 9.0 ns, i.e. during the access. The failure history shows the
Newton loop reaching its iteration limit (`F:max s`, 21 iterations) with a
residual of only 1.02e-12 A at every step size down to 1e-22 s — an
oscillating Newton iteration, not a tolerance problem. Options tried on the
6T 512x4 read deck:

| option | result |
|---|---|
| `.OPTIONS NONLIN-TRAN ABSTOL=1e-10` | fails at the same step |
| `.OPTIONS TIMEINT NLNEARCONV=1` | fails at the same step |
| `.OPTIONS NONLIN-TRAN MAXSTEP=60` | fails at the same step |
| `.OPTIONS TIMEINT ERROPTION=1` | completes; on 8x4 / 16x16 decks that converge with the default settings it shifts TREAD_TOTAL by +3 %, TSA by +5..15 %, TS_EN by +4..36 %, TWRITE_TOTAL by +2 %, PAVG by up to −2.3 % |
| `.TRAN ... 0 2e-11` (maximum step 20 ps) | completes the deck (TREAD_TOTAL 709 ps, PAVG 148 µW, 1375 time steps; the muxed 512x4 read that converges with default settings gives 692 ps / 148 µW in 591 steps); on the 8x4 read / write decks it changes every delay measure by < 0.5 % and the energy by < 0.4 % (1038 -> 1856 time steps) |

The maximum-step form is applied automatically as a one-time retry when Xyce
reports `Time step too small`; `Sram6TCoreMcTestbench` also accepts
`t_max_step=<s>` to use it from the start and `xyce_options=[...]` (extra
netlist lines) for anything else.

## Observations that were verified and deliberately left unchanged

* Read delay ≈ 300 ps for every size up to 32 rows: `s_en` = fully
  discharged replica bitline + 9-stage delay chain; BL is at ~0.02 V when the
  sense amplifier is enabled, 250 mV of swing is reached 10–35 ps after WL.
* Precharge is a ~300 ps self-timed pulse; bitlines then float, overshoot to
  1.03–1.07 V (PRE gate coupling) and leak to 0.93–0.95 V by the next access
  on leaky / small-capacitance columns (10T 16x16 RBL, 2x128).
* Write delay is non-monotonic in the row count because the write driver
  width scales with `max(8, rows) / 16` (8x4: 137 ps, 16x1: 72 ps).
* `w_rc=True` with `pi_cap = 1 fF` loads Q/QB of every cell and triples the
  write delay (16x16 6T: 98 → 346 ps).
* One `.SAMPLING` sample of the old 10T 64x16 muxed write deck aborted in Xyce
  with "time step too small" at the start of the access; the nominal run with
  the corrected topology completes (139 ps).

## Appendix B — per-run results of the final sweep (V2.0.1 code)

Every row is one Xyce run of the final code. Delays in ps; "WL rise -> s_en / w_en"
is the time from the target wordline crossing VDD/2 to the sense / write enable
crossing VDD/2; "BL@s_en" is the target bitline voltage when the sense
amplifier is enabled (read) or the minimum BL of the read-1 cycle (`read&write`);
"Q bump" is the read-disturb peak on the stored 0; "TWRITE" is WL rise -> Q 90 %
(`read&write`: write-1 / write-0 cycles).

| array | cell | mux | op | mc | status | delay [ps] (mean±sd) | PAVG [µW] | PSTC [µW] | WL rise→s_en/w_en [ps] | BL@s_en [V] | Q bump [V] | TWRITE [ps] | waveform checks | run [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1 | 10T | off | read | 1 | ok | 288.8 | 20.14 | 0.89 | 245.8 | 0.040 | 0.203 | - | PASS | 37 |
| 1x1 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 21.90 | 0.77 | - | 0.971 | - | 120.8/133.1 | PASS | 130 |
| 1x1 | 10T | off | write | 1 | ok | 132.7 | 21.81 | 0.75 | -16.4 | - | - | 121.4 | PASS | 41 |
| 1x1 | 6T | off | read | 1 | ok | 286.1 | 20.14 | 0.70 | 242.7 | 0.033 | 0.132 | - | PASS | 34 |
| 1x1 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 22.03 | 0.76 | - | 0.973 | - | 118.0/149.3 | PASS | 161 |
| 1x1 | 6T | off | write | 1 | ok | 131.4 | 22.04 | 0.74 | -16.3 | - | - | 120.8 | PASS | 44 |
| 2x1 | 10T | off | read | 1 | ok | 289.9 | 21.46 | 0.72 | 247.3 | 0.044 | 0.203 | - | PASS | 41 |
| 2x1 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 23.13 | 0.80 | - | 0.971 | - | 121.5/134.0 | PASS | 190 |
| 2x1 | 10T | off | write | 1 | ok | 133.6 | 23.13 | 0.78 | -17.0 | - | - | 122.0 | PASS | 51 |
| 2x1 | 6T | off | read | 1 | ok | 287.0 | 21.40 | 0.72 | 244.5 | 0.029 | 0.132 | - | PASS | 48 |
| 2x1 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 23.25 | 0.80 | - | 0.971 | - | 119.2/149.2 | PASS | 179 |
| 2x1 | 6T | off | write | 1 | ok | 132.3 | 23.29 | 0.78 | -17.1 | - | - | 121.2 | PASS | 45 |
| 2x2 | 10T | off | read | 1 | ok | 293.2 | 22.23 | 0.74 | 248.4 | 0.037 | 0.203 | - | PASS | 43 |
| 2x2 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 25.58 | 0.90 | - | 0.969 | - | 119.4/132.6 | PASS | 210 |
| 2x2 | 10T | off | write | 1 | ok | 134.3 | 26.42 | 0.88 | -18.2 | - | - | 119.8 | PASS | 50 |
| 2x2 | 10T | on | read | 1 | ok | 289.9 | 21.73 | 0.74 | 245.9 | 0.036 | 0.189 | - | PASS | 45 |
| 2x2 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 25.20 | 0.90 | - | 0.976 | - | 148.2/166.2 | PASS | 166 |
| 2x2 | 10T | on | write | 1 | ok | 163.8 | 26.11 | 0.87 | -18.5 | - | - | 149.1 | PASS | 49 |
| 2x2 | 6T | off | read | 1 | ok | 290.2 | 22.21 | 0.73 | 245.6 | 0.029 | 0.131 | - | PASS | 55 |
| 2x2 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 25.78 | 0.89 | - | 0.967 | - | 116.5/145.4 | PASS | 210 |
| 2x2 | 6T | off | write | 1 | ok | 131.5 | 26.77 | 0.87 | -18.3 | - | - | 118.0 | PASS | 64 |
| 2x2 | 6T | on | read | 1 | ok | 287.1 | 21.74 | 0.73 | 243.1 | 0.029 | 0.121 | - | PASS | 46 |
| 2x2 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 25.43 | 0.89 | - | 0.975 | - | 157.0/188.2 | PASS | 167 |
| 2x2 | 6T | on | write | 1 | ok | 173.6 | 26.49 | 0.87 | -18.6 | - | - | 159.0 | PASS | 65 |
| 4x2 | 10T | off | read | 1 | ok | 295.4 | 24.10 | 0.85 | 249.0 | 0.035 | 0.204 | - | PASS | 60 |
| 4x2 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 27.50 | 1.01 | - | 0.969 | - | 121.0/133.7 | PASS | 239 |
| 4x2 | 10T | off | write | 1 | ok | 136.3 | 28.31 | 0.99 | -20.8 | - | - | 121.6 | PASS | 57 |
| 4x2 | 10T | on | read | 1 | ok | 292.2 | 23.60 | 0.84 | 246.2 | 0.030 | 0.192 | - | PASS | 70 |
| 4x2 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 27.10 | 1.00 | - | 0.975 | - | 148.8/166.0 | PASS | 251 |
| 4x2 | 10T | on | write | 1 | ok | 166.0 | 28.07 | 0.98 | -20.9 | - | - | 150.4 | PASS | 57 |
| 4x2 | 6T | off | read | 1 | ok | 292.0 | 24.01 | 0.83 | 245.5 | 0.028 | 0.130 | - | PASS | 48 |
| 4x2 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 27.68 | 0.99 | - | 0.965 | - | 115.9/144.2 | PASS | 176 |
| 4x2 | 6T | off | write | 1 | ok | 133.9 | 28.75 | 0.97 | -20.9 | - | - | 118.7 | PASS | 54 |
| 4x2 | 6T | on | read | 1 | ok | 288.8 | 23.57 | 0.82 | 242.8 | 0.028 | 0.122 | - | PASS | 56 |
| 4x2 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 27.33 | 0.99 | - | 0.974 | - | 156.2/187.3 | PASS | 180 |
| 4x2 | 6T | on | write | 1 | ok | 176.1 | 28.43 | 0.97 | -21.0 | - | - | 161.2 | PASS | 75 |
| 3x3 | 10T | off | read | 1 | ok | 297.2 | 23.68 | 0.84 | 249.5 | 0.032 | 0.203 | - | PASS | 49 |
| 3x3 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 28.82 | 1.07 | - | 0.967 | - | 120.1/132.7 | PASS | 246 |
| 3x3 | 10T | off | write | 1 | ok | 135.9 | 30.47 | 1.05 | -20.3 | - | - | 120.6 | PASS | 81 |
| 3x3 | 6T | off | read | 1 | ok | 294.0 | 23.62 | 0.82 | 246.1 | 0.025 | 0.129 | - | PASS | 57 |
| 3x3 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 29.10 | 1.06 | - | 0.961 | - | 114.7/142.2 | PASS | 248 |
| 3x3 | 6T | off | write | 1 | ok | 132.2 | 31.00 | 1.04 | -20.4 | - | - | 115.7 | PASS | 76 |
| 5x3 | 10T | off | read | 1 | ok | 299.4 | 24.50 | 0.93 | 251.2 | 0.032 | 0.203 | - | PASS | 78 |
| 5x3 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 29.53 | 1.17 | - | 0.965 | - | 122.5/134.5 | PASS | 236 |
| 5x3 | 10T | off | write | 1 | ok | 138.2 | 31.34 | 1.15 | -21.1 | - | - | 122.9 | PASS | 92 |
| 5x3 | 6T | off | read | 1 | ok | 295.8 | 24.44 | 0.90 | 247.6 | 0.024 | 0.130 | - | PASS | 52 |
| 5x3 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 29.98 | 1.14 | - | 0.963 | - | 118.7/143.0 | PASS | 262 |
| 5x3 | 6T | off | write | 1 | ok | 134.3 | 31.93 | 1.12 | -21.1 | - | - | 120.5 | PASS | 87 |
| 4x4 | 10T | off | read | 1 | ok | 301.2 | 25.71 | 0.91 | 251.7 | 0.029 | 0.202 | - | PASS | 73 |
| 4x4 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 32.53 | 1.22 | - | 0.962 | - | 121.8/133.5 | PASS | 256 |
| 4x4 | 10T | off | write | 1 | ok | 137.8 | 35.11 | 1.21 | -20.8 | - | - | 122.0 | PASS | 94 |
| 4x4 | 10T | on | read | 1 | ok | 295.4 | 24.76 | 0.90 | 247.1 | 0.032 | 0.189 | - | PASS | 74 |
| 4x4 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 31.77 | 1.22 | - | 0.973 | - | 150.3/167.3 | PASS | 301 |
| 4x4 | 10T | on | write | 1 | ok | 167.6 | 34.57 | 1.19 | -20.9 | - | - | 151.3 | PASS | 98 |
| 4x4 | 6T | off | read | 1 | ok | 297.7 | 25.70 | 0.88 | 248.4 | 0.022 | 0.129 | - | PASS | 52 |
| 4x4 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 32.99 | 1.20 | - | 0.959 | - | 115.8/141.2 | PASS | 231 |
| 4x4 | 6T | off | write | 1 | ok | 133.5 | 36.02 | 1.18 | -20.8 | - | - | 118.2 | PASS | 91 |
| 4x4 | 6T | on | read | 1 | ok | 292.0 | 24.72 | 0.88 | 243.5 | 0.024 | 0.117 | - | PASS | 56 |
| 4x4 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 32.17 | 1.19 | - | 0.971 | - | 158.0/188.3 | PASS | 232 |
| 4x4 | 6T | on | write | 1 | ok | 176.1 | 35.46 | 1.16 | -21.0 | - | - | 161.0 | PASS | 71 |
| 16x1 | 10T | off | read | 1 | ok | 308.2 | 28.31 | 1.37 | 264.3 | 0.033 | 0.207 | - | PASS | 90 |
| 16x1 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 29.80 | 1.46 | - | 0.983 | - | 72.5/71.1 | PASS | 387 |
| 16x1 | 10T | off | write | 1 | ok | 85.6 | 29.89 | 1.44 | -24.4 | - | - | 72.4 | PASS | 120 |
| 16x1 | 6T | off | read | 1 | ok | 301.6 | 28.30 | 1.33 | 257.6 | 0.027 | 0.134 | - | PASS | 119 |
| 16x1 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 29.80 | 1.42 | - | 0.982 | - | 60.4/59.8 | PASS | 352 |
| 16x1 | 6T | off | write | 1 | ok | 72.6 | 29.81 | 1.40 | -24.5 | - | - | 60.5 | PASS | 111 |
| 8x4 | 10T | off | read | 1 | ok | 305.3 | 28.00 | 1.08 | 256.4 | 0.028 | 0.202 | - | PASS | 103 |
| 8x4 | 10T | off | read | 5 | ok | 309.5±3.7 | 27.78 | 1.08 | 263.1 | 0.028 | 0.196 | - | PASS | 450 |
| 8x4 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 34.89 | 1.41 | - | 0.962 | - | 126.8/139.1 | PASS | 347 |
| 8x4 | 10T | off | read&write | 3 | ok | TVOUT_PERIOD 40.00 ns | 34.94 | 1.49 | - | 0.968 | - | 143.9/160.9 | PASS | 927 |
| 8x4 | 10T | off | write | 1 | ok | 142.3 | 37.71 | 1.39 | -22.6 | - | - | 127.1 | PASS | 93 |
| 8x4 | 10T | off | write | 5 | ok | 152.0±7.5 | 37.56 | 1.38 | -24.4 | - | - | 131.4 | PASS | 444 |
| 8x4 | 10T | on | read | 1 | ok | 299.7 | 26.98 | 1.09 | 251.8 | 0.027 | 0.192 | - | PASS | 80 |
| 8x4 | 10T | on | read | 5 | ok | 303.8±3.6 | 26.80 | 1.07 | 258.3 | 0.028 | 0.186 | - | PASS | 454 |
| 8x4 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 34.11 | 1.40 | - | 0.972 | - | 153.9/171.2 | PASS | 296 |
| 8x4 | 10T | on | read&write | 3 | ok | TVOUT_PERIOD 40.00 ns | 34.16 | 1.47 | - | 0.977 | - | 179.4/202.1 | PASS | 897 |
| 8x4 | 10T | on | write | 1 | ok | 171.5 | 37.04 | 1.38 | -22.8 | - | - | 155.6 | PASS | 99 |
| 8x4 | 10T | on | write | 5 | ok | 184.6±10.3 | 37.00 | 1.37 | -24.6 | - | - | 163.0 | PASS | 450 |
| 8x4 | 6T | off | read | 1 | ok | 301.1 | 28.04 | 1.05 | 252.3 | 0.021 | 0.130 | - | PASS | 65 |
| 8x4 | 6T | off | read | 5 | ok | 305.1±3.6 | 27.75 | 1.03 | 258.7 | 0.021 | 0.125 | - | PASS | 380 |
| 8x4 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 35.23 | 1.36 | - | 0.958 | - | 121.1/142.7 | PASS | 334 |
| 8x4 | 6T | off | read&write | 3 | ok | TVOUT_PERIOD 40.00 ns | 35.31 | 1.43 | - | 0.965 | - | 140.9/171.7 | PASS | 610 |
| 8x4 | 6T | off | write | 1 | ok | 138.0 | 38.39 | 1.34 | -22.8 | - | - | 121.8 | PASS | 105 |
| 8x4 | 6T | off | write | 5 | ok | 153.3±14.2 | 38.58 | 1.34 | -24.5 | - | - | 127.1 | PASS | 383 |
| 8x4 | 6T | on | read | 1 | ok | 295.5 | 27.02 | 1.03 | 247.7 | 0.024 | 0.122 | - | PASS | 87 |
| 8x4 | 6T | on | read | 5 | ok | 299.5±3.5 | 26.78 | 1.03 | 254.1 | 0.024 | 0.118 | - | PASS | 325 |
| 8x4 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 34.51 | 1.35 | - | 0.971 | - | 161.3/190.6 | PASS | 323 |
| 8x4 | 6T | on | read&write | 3 | ok | TVOUT_PERIOD 40.00 ns | 34.60 | 1.42 | - | 0.976 | - | 190.6/229.5 | PASS | 793 |
| 8x4 | 6T | on | write | 1 | ok | 180.9 | 37.89 | 1.32 | -23.0 | - | - | 165.3 | PASS | 111 |
| 8x4 | 6T | on | write | 5 | ok | 202.8±20.6 | 38.09 | 1.33 | -24.7 | - | - | 173.4 | PASS | 383 |
| 6x6 | 10T | off | read | 1 | ok | 308.4 | 28.39 | 1.11 | 257.2 | 0.024 | 0.201 | - | PASS | 81 |
| 6x6 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 38.70 | 1.58 | - | 0.957 | - | 124.6/136.4 | PASS | 328 |
| 6x6 | 10T | off | write | 1 | ok | 141.9 | 43.24 | 1.56 | -22.3 | - | - | 124.8 | PASS | 129 |
| 6x6 | 10T | on | read | 1 | ok | 300.6 | 26.87 | 1.09 | 250.8 | 0.026 | 0.189 | - | PASS | 81 |
| 6x6 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 37.52 | 1.56 | - | 0.971 | - | 151.8/169.1 | PASS | 396 |
| 6x6 | 10T | on | write | 1 | ok | 171.2 | 42.32 | 1.53 | -22.5 | - | - | 153.5 | PASS | 120 |
| 6x6 | 6T | off | read | 1 | ok | 304.6 | 28.30 | 1.06 | 253.4 | 0.018 | 0.129 | - | PASS | 94 |
| 6x6 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 39.38 | 1.52 | - | 0.952 | - | 118.4/138.6 | PASS | 351 |
| 6x6 | 6T | off | write | 1 | ok | 136.9 | 44.45 | 1.50 | -22.6 | - | - | 118.9 | PASS | 92 |
| 6x6 | 6T | on | read | 1 | ok | 296.7 | 26.75 | 1.04 | 247.0 | 0.022 | 0.120 | - | PASS | 89 |
| 6x6 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 38.23 | 1.50 | - | 0.970 | - | 158.8/188.4 | PASS | 350 |
| 6x6 | 6T | on | write | 1 | ok | 180.2 | 43.67 | 1.48 | -22.8 | - | - | 162.5 | PASS | 96 |
| 12x4 | 10T | off | read | 1 | ok | 311.4 | 29.78 | 1.53 | 262.6 | 0.024 | 0.204 | - | PASS | 152 |
| 12x4 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 36.29 | 1.86 | - | 0.971 | - | 88.0/90.5 | PASS | 470 |
| 12x4 | 10T | off | write | 1 | ok | 105.8 | 38.74 | 1.83 | -25.0 | - | - | 87.9 | PASS | 151 |
| 12x4 | 10T | on | read | 1 | ok | 305.2 | 28.83 | 1.52 | 257.7 | 0.027 | 0.196 | - | PASS | 150 |
| 12x4 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 35.46 | 1.84 | - | 0.978 | - | 101.8/105.2 | PASS | 461 |
| 12x4 | 10T | on | write | 1 | ok | 118.8 | 38.07 | 1.82 | -25.2 | - | - | 101.9 | PASS | 159 |
| 12x4 | 6T | off | read | 1 | ok | 305.9 | 29.51 | 1.45 | 257.1 | 0.021 | 0.130 | - | PASS | 101 |
| 12x4 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 36.33 | 1.78 | - | 0.970 | - | 76.4/77.6 | PASS | 371 |
| 12x4 | 6T | off | write | 1 | ok | 92.9 | 38.84 | 1.76 | -24.9 | - | - | 76.4 | PASS | 116 |
| 12x4 | 6T | on | read | 1 | ok | 300.4 | 28.53 | 1.44 | 252.6 | 0.023 | 0.124 | - | PASS | 118 |
| 12x4 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 35.55 | 1.77 | - | 0.977 | - | 94.2/97.3 | PASS | 362 |
| 12x4 | 6T | on | write | 1 | ok | 110.4 | 38.30 | 1.74 | -25.1 | - | - | 94.3 | PASS | 121 |
| 8x8 | 10T | off | read | 1 | ok | 315.2 | 31.59 | 1.33 | 261.6 | 0.021 | 0.202 | - | PASS | 107 |
| 8x8 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 45.68 | 1.95 | - | 0.952 | - | 126.2/138.1 | PASS | 431 |
| 8x8 | 10T | off | write | 1 | ok | 146.5 | 52.24 | 1.93 | -23.8 | - | - | 126.5 | PASS | 135 |
| 8x8 | 10T | on | read | 1 | ok | 305.6 | 29.57 | 1.32 | 253.8 | 0.026 | 0.191 | - | PASS | 112 |
| 8x8 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 44.06 | 1.93 | - | 0.968 | - | 153.5/171.2 | PASS | 435 |
| 8x8 | 10T | on | write | 1 | ok | 174.3 | 50.95 | 1.91 | -23.9 | - | - | 154.3 | PASS | 137 |
| 8x8 | 6T | off | read | 1 | ok | 311.2 | 31.40 | 1.23 | 257.6 | 0.015 | 0.128 | - | PASS | 102 |
| 8x8 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 46.49 | 1.86 | - | 0.948 | - | 122.0/139.9 | PASS | 387 |
| 8x8 | 6T | off | write | 1 | ok | 141.2 | 53.75 | 1.84 | -24.1 | - | - | 122.6 | PASS | 134 |
| 8x8 | 6T | on | read | 1 | ok | 301.3 | 29.45 | 1.22 | 249.4 | 0.019 | 0.120 | - | PASS | 100 |
| 8x8 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 44.89 | 1.84 | - | 0.966 | - | 162.1/190.2 | PASS | 392 |
| 8x8 | 6T | on | write | 1 | ok | 183.4 | 52.69 | 1.81 | -24.3 | - | - | 163.8 | PASS | 123 |
| 16x8 | 10T | off | read | 1 | ok | 326.9 | 36.17 | 2.11 | 271.8 | 0.019 | 0.204 | - | PASS | 286 |
| 16x8 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 50.27 | 2.79 | - | 0.971 | - | 76.9/75.6 | PASS | 1528 |
| 16x8 | 10T | off | write | 1 | ok | 98.1 | 55.99 | 2.77 | -24.5 | - | - | 76.9 | PASS | 330 |
| 16x8 | 10T | on | read | 1 | ok | 316.4 | 34.18 | 2.09 | 263.0 | 0.021 | 0.198 | - | PASS | 283 |
| 16x8 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 48.64 | 2.76 | - | 0.979 | - | 84.2/82.7 | PASS | 1238 |
| 16x8 | 10T | on | write | 1 | ok | 104.6 | 54.73 | 2.74 | -24.6 | - | - | 84.1 | PASS | 333 |
| 16x8 | 6T | off | read | 1 | ok | 320.5 | 35.83 | 1.93 | 265.0 | 0.015 | 0.131 | - | PASS | 137 |
| 16x8 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 49.88 | 2.62 | - | 0.966 | - | 65.8/65.1 | PASS | 543 |
| 16x8 | 6T | off | write | 1 | ok | 85.2 | 55.66 | 2.58 | -24.3 | - | - | 65.8 | PASS | 167 |
| 16x8 | 6T | on | read | 1 | ok | 310.4 | 33.80 | 1.90 | 257.0 | 0.019 | 0.127 | - | PASS | 156 |
| 16x8 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 48.25 | 2.59 | - | 0.977 | - | 73.8/72.5 | PASS | 599 |
| 16x8 | 6T | on | write | 1 | ok | 93.3 | 54.36 | 2.57 | -24.5 | - | - | 73.8 | PASS | 197 |
| 20x10 | 10T | off | read | 1 | ok | 336.2 | 40.13 | 2.80 | 278.0 | 0.017 | 0.204 | - | PASS | 405 |
| 20x10 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 57.97 | 3.67 | - | 0.972 | - | 72.9/69.7 | PASS | 2129 |
| 20x10 | 10T | off | write | 1 | ok | 95.3 | 65.96 | 3.65 | -22.7 | - | - | 72.9 | PASS | 519 |
| 20x10 | 10T | on | read | 1 | ok | 324.3 | 37.60 | 2.77 | 268.1 | 0.019 | 0.200 | - | PASS | 457 |
| 20x10 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 55.96 | 3.64 | - | 0.981 | - | 77.3/72.6 | PASS | 1712 |
| 20x10 | 10T | on | write | 1 | ok | 99.3 | 64.43 | 3.62 | -22.8 | - | - | 77.4 | PASS | 531 |
| 20x10 | 6T | off | read | 1 | ok | 328.8 | 39.76 | 2.53 | 271.0 | 0.013 | 0.132 | - | PASS | 288 |
| 20x10 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 57.69 | 3.41 | - | 0.968 | - | 61.1/59.4 | PASS | 1304 |
| 20x10 | 6T | off | write | 1 | ok | 82.6 | 65.44 | 3.39 | -22.7 | - | - | 61.1 | PASS | 353 |
| 20x10 | 6T | on | read | 1 | ok | 316.6 | 37.29 | 2.50 | 261.0 | 0.017 | 0.127 | - | PASS | 279 |
| 20x10 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 55.56 | 3.37 | - | 0.980 | - | 65.3/61.0 | PASS | 1340 |
| 20x10 | 6T | on | write | 1 | ok | 86.8 | 63.83 | 3.35 | -22.8 | - | - | 65.3 | PASS | 370 |
| 2x128 | 10T | off | read | 1 | ok | 462.7 | 158.75 | 3.36 | 312.2 | 0.003 | 0.177 | - | PASS | 728 |
| 2x128 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 513.84 | 13.61 | - | 0.907 | - | 101.4/108.6 | PASS | 6188 |
| 2x128 | 10T | off | write | 1 | ok | 202.3 | 733.99 | 13.59 | -62.6 | - | - | 101.5 | PASS | 2227 |
| 2x128 | 10T | on | read | 1 | ok | 386.4 | 94.82 | 3.13 | 242.0 | 0.007 | 0.159 | - | PASS | 847 |
| 2x128 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 480.35 | 13.36 | - | 0.938 | - | 121.4/137.7 | PASS | 5747 |
| 2x128 | 10T | on | write | 1 | ok | 224.0 | 728.43 | 13.20 | -62.8 | - | - | 122.5 | PASS | 1763 |
| 2x128 | 6T | off | read | 1 | ok | 458.0 | 158.39 | 3.03 | 308.3 | 0.002 | 0.108 | - | PASS | 530 |
| 2x128 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 515.34 | 13.29 | - | 0.904 | - | 89.5/101.9 | PASS | 5189 |
| 2x128 | 6T | off | write | 1 | ok | 190.8 | 740.06 | 13.27 | -62.9 | - | - | 89.6 | PASS | 1986 |
| 2x128 | 6T | on | read | 1 | ok | 382.1 | 94.31 | 2.80 | 238.1 | 0.005 | 0.094 | - | PASS | 570 |
| 2x128 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 488.51 | 13.05 | - | 0.937 | - | 120.1/149.4 | PASS | 6943 |
| 2x128 | 6T | on | write | 1 | ok | 225.1 | 745.19 | 12.88 | -63.1 | - | - | 124.1 | PASS | 1625 |
| 16x16 | 10T | off | read | 1 | ok | 343.9 | 45.32 | 2.99 | 279.4 | 0.013 | 0.202 | - | PASS | 528 |
| 16x16 | 10T | off | read | 5 | ok | 348.3±3.8 | 44.78 | 2.81 | 285.6 | 0.013 | 0.197 | - | PASS | 2907 |
| 16x16 | 10T | off | read | 1 | ok | 554.3 | 94.65 | 2.79 | 443.5 | 0.132 | 0.210 | - | PASS | 169 |
| 16x16 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 74.93 | 4.34 | - | 0.959 | - | 84.9/83.4 | PASS | 2061 |
| 16x16 | 10T | off | write | 1 | ok | 110.9 | 88.78 | 4.31 | -22.8 | - | - | 84.9 | PASS | 665 |
| 16x16 | 10T | off | write | 5 | ok | 114.8±2.8 | 88.43 | 4.17 | -23.1 | - | - | 87.7 | PASS | 2906 |
| 16x16 | 10T | off | write | 1 | ok | 336.2 | 158.52 | 4.65 | -73.7 | - | - | 267.7 | PASS | 462 |
| 16x16 | 10T | on | read | 1 | ok | 327.0 | 41.14 | 2.94 | 265.2 | 0.018 | 0.197 | - | PASS | 534 |
| 16x16 | 10T | on | read | 5 | ok | 330.9±3.8 | 40.81 | 2.76 | 272.4 | 0.018 | 0.191 | - | PASS | 2944 |
| 16x16 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 71.67 | 4.29 | - | 0.972 | - | 89.0/87.6 | PASS | 2096 |
| 16x16 | 10T | on | write | 1 | ok | 116.1 | 86.16 | 4.27 | -22.8 | - | - | 89.3 | PASS | 649 |
| 16x16 | 10T | on | write | 5 | ok | 120.9±3.4 | 85.81 | 4.11 | -23.2 | - | - | 93.4 | PASS | 2805 |
| 16x16 | 6T | off | read | 1 | ok | 337.1 | 44.87 | 2.66 | 273.2 | 0.010 | 0.130 | - | PASS | 329 |
| 16x16 | 6T | off | read | 5 | ok | 341.6±4.0 | 44.33 | 2.52 | 279.4 | 0.010 | 0.125 | - | PASS | 1570 |
| 16x16 | 6T | off | read | 1 | ok | 512.7 | 93.90 | 2.44 | 401.8 | 0.068 | 0.133 | - | PASS | 118 |
| 16x16 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 74.90 | 3.99 | - | 0.952 | - | 73.0/72.8 | PASS | 2063 |
| 16x16 | 6T | off | write | 1 | ok | 98.4 | 89.11 | 3.98 | -22.6 | - | - | 73.1 | PASS | 503 |
| 16x16 | 6T | off | write | 5 | ok | 102.5±3.0 | 88.48 | 3.88 | -22.9 | - | - | 74.5 | PASS | 2204 |
| 16x16 | 6T | off | write | 1 | ok | 334.1 | 165.68 | 4.31 | -73.6 | - | - | 265.5 | PASS | 319 |
| 16x16 | 6T | on | read | 1 | ok | 320.4 | 41.01 | 2.60 | 259.1 | 0.014 | 0.125 | - | PASS | 361 |
| 16x16 | 6T | on | read | 5 | ok | 324.6±3.5 | 40.47 | 2.47 | 265.1 | 0.014 | 0.121 | - | PASS | 2079 |
| 16x16 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 71.22 | 3.94 | - | 0.971 | - | 78.8/76.3 | PASS | 1616 |
| 16x16 | 6T | on | write | 1 | ok | 105.4 | 85.80 | 3.92 | -22.7 | - | - | 78.9 | PASS | 504 |
| 16x16 | 6T | on | write | 5 | ok | 110.4±3.9 | 85.39 | 3.84 | -23.0 | - | - | 82.1 | PASS | 2816 |
| 32x8 | 10T | off | read | 1 | ok | 346.7 | 43.87 | 3.41 | 290.0 | 0.017 | 0.207 | - | PASS | 516 |
| 32x8 | 10T | off | read | 5 | ok | 351.3±4.0 | 43.36 | 3.22 | 296.8 | 0.017 | 0.201 | - | PASS | 3169 |
| 32x8 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 59.29 | 4.19 | - | 0.982 | - | 62.5/54.9 | PASS | 2050 |
| 32x8 | 10T | off | write | 1 | ok | 84.2 | 65.66 | 4.17 | -23.7 | - | - | 62.5 | PASS | 568 |
| 32x8 | 10T | off | write | 5 | ok | 86.6±1.6 | 65.32 | 3.99 | -25.4 | - | - | 64.3 | PASS | 2720 |
| 32x8 | 10T | on | read | 1 | ok | 337.1 | 41.81 | 3.38 | 282.3 | 0.020 | 0.204 | - | PASS | 575 |
| 32x8 | 10T | on | read | 5 | ok | 342.3±4.4 | 41.27 | 3.20 | 290.3 | 0.020 | 0.198 | - | PASS | 2659 |
| 32x8 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 57.69 | 4.16 | - | 0.988 | - | 64.3/57.0 | PASS | 2080 |
| 32x8 | 10T | on | write | 1 | ok | 86.0 | 64.92 | 4.14 | -23.7 | - | - | 64.3 | PASS | 582 |
| 32x8 | 10T | on | write | 5 | ok | 88.3±1.8 | 64.36 | 3.96 | -25.4 | - | - | 65.5 | PASS | 3184 |
| 32x8 | 6T | off | read | 1 | ok | 336.2 | 43.52 | 3.05 | 280.0 | 0.013 | 0.133 | - | PASS | 400 |
| 32x8 | 6T | off | read | 5 | ok | 341.6±4.3 | 43.17 | 2.92 | 287.4 | 0.013 | 0.129 | - | PASS | 1838 |
| 32x8 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 58.87 | 3.83 | - | 0.980 | - | 49.5/45.9 | PASS | 1524 |
| 32x8 | 6T | off | write | 1 | ok | 72.2 | 65.56 | 3.81 | -24.3 | - | - | 50.0 | PASS | 426 |
| 32x8 | 6T | off | write | 5 | ok | 74.8±2.0 | 64.85 | 3.68 | -25.3 | - | - | 52.8 | PASS | 2677 |
| 32x8 | 6T | on | read | 1 | ok | 326.8 | 41.43 | 3.01 | 272.3 | 0.015 | 0.130 | - | PASS | 399 |
| 32x8 | 6T | on | read | 5 | ok | 331.5±4.5 | 41.13 | 2.89 | 279.6 | 0.015 | 0.127 | - | PASS | 1766 |
| 32x8 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 57.43 | 3.80 | - | 0.985 | - | 51.0/46.1 | PASS | 1968 |
| 32x8 | 6T | on | write | 1 | ok | 72.9 | 64.31 | 3.78 | -24.4 | - | - | 50.9 | PASS | 470 |
| 32x8 | 6T | on | write | 5 | ok | 75.8±1.7 | 63.63 | 3.65 | -25.4 | - | - | 53.4 | PASS | 2654 |
| 32x32 | 10T | off | read | 1 | ok | 395.5 | 81.90 | 8.50 | 313.9 | 0.008 | 0.205 | - | PASS | 1383 |
| 32x32 | 10T | off | read | 1 | ok | 769.8 | 231.61 | 7.83 | 625.8 | 0.204 | 0.212 | - | PASS | 466 |
| 32x32 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 174.02 | 11.75 | - | 0.966 | - | 109.4/102.2 | PASS | 7129 |
| 32x32 | 10T | off | write | 1 | ok | 147.8 | 227.37 | 11.73 | -0.6 | - | - | 109.5 | PASS | 1640 |
| 32x32 | 10T | off | write | 1 | ok | 339.9 | 404.14 | 13.31 | -70.6 | - | - | 248.0 | PASS | 806 |
| 32x32 | 10T | on | read | 1 | ok | 363.7 | 72.90 | 8.39 | 286.5 | 0.014 | 0.202 | - | PASS | 1365 |
| 32x32 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 167.27 | 11.64 | - | 0.978 | - | 110.5/102.9 | PASS | 5954 |
| 32x32 | 10T | on | write | 1 | ok | 149.4 | 222.59 | 11.62 | -0.5 | - | - | 110.7 | PASS | 1944 |
| 32x32 | 6T | off | read | 1 | ok | 385.5 | 81.11 | 7.19 | 303.6 | 0.006 | 0.130 | - | PASS | 1007 |
| 32x32 | 6T | off | read | 1 | ok | 695.2 | 229.73 | 6.52 | 552.6 | 0.139 | 0.135 | - | PASS | 344 |
| 32x32 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 174.08 | 10.40 | - | 0.962 | - | 98.6/93.0 | PASS | 5016 |
| 32x32 | 6T | off | write | 1 | ok | 137.3 | 227.80 | 10.38 | -0.7 | - | - | 98.7 | PASS | 1142 |
| 32x32 | 6T | off | write | 1 | ok | 334.0 | 414.49 | 12.00 | -70.7 | - | - | 241.7 | PASS | 759 |
| 32x32 | 6T | on | read | 1 | ok | 354.7 | 72.25 | 7.07 | 276.7 | 0.009 | 0.128 | - | PASS | 904 |
| 32x32 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 167.29 | 10.29 | - | 0.977 | - | 99.5/92.8 | PASS | 4255 |
| 32x32 | 6T | on | write | 1 | ok | 137.9 | 222.98 | 10.27 | -0.7 | - | - | 99.6 | PASS | 1179 |
| 64x16 | 10T | off | read | 1 | ok | 401.1 | 76.26 | 9.21 | 332.4 | 0.017 | 0.209 | - | PASS | 1378 |
| 64x16 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 133.82 | 11.21 | - | 0.986 | - | 94.1/83.6 | PASS | 6430 |
| 64x16 | 10T | off | write | 1 | ok | 124.9 | 166.63 | 11.20 | -3.1 | - | - | 93.9 | PASS | 1991 |
| 64x16 | 10T | on | read | 1 | ok | 382.8 | 72.21 | 9.14 | 316.6 | 0.028 | 0.208 | - | PASS | 1663 |
| 64x16 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 130.80 | 11.15 | - | 0.988 | - | 95.1/84.0 | PASS | 5957 |
| 64x16 | 10T | on | write | 1 | ok | 125.5 | 164.99 | 11.13 | -3.0 | - | - | 94.4 | PASS | 1872 |
| 64x16 | 6T | off | read | 1 | ok | 382.3 | 75.10 | 7.85 | 314.6 | 0.009 | 0.135 | - | PASS | 977 |
| 64x16 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 133.00 | 9.85 | - | 0.986 | - | 83.3/76.8 | PASS | 4014 |
| 64x16 | 6T | off | write | 1 | ok | 113.9 | 165.83 | 9.83 | -3.5 | - | - | 83.2 | PASS | 1100 |
| 64x16 | 6T | on | read | 1 | ok | 366.0 | 71.23 | 7.79 | 300.4 | 0.014 | 0.134 | - | PASS | 1271 |
| 64x16 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 129.74 | 9.80 | - | 0.988 | - | 83.5/75.3 | PASS | 3906 |
| 64x16 | 6T | on | write | 1 | ok | 114.3 | 163.42 | 9.77 | -3.4 | - | - | 83.4 | PASS | 1409 |
| 256x8 | 10T | off | read | 1 | ok | 579.3 | 124.86 | 21.05 | 507.8 | 0.174 | 0.215 | - | PASS | 4249 |
| 256x8 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 202.76 | 23.44 | - | 0.975 | - | 44.1/28.5 | PASS | 8081 |
| 256x8 | 10T | off | write | 1 | ok | 78.1 | 229.16 | 23.43 | -52.4 | - | - | 44.1 | PASS | 4457 |
| 256x8 | 10T | on | read | 1 | ok | 566.0 | 122.77 | 21.02 | 497.9 | 0.194 | 0.215 | - | PASS | 4132 |
| 256x8 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 201.16 | 23.42 | - | 0.970 | - | 44.4/30.2 | PASS | 7741 |
| 256x8 | 10T | on | write | 1 | ok | 78.6 | 228.67 | 23.39 | -52.2 | - | - | 44.4 | PASS | 4234 |
| 256x8 | 6T | off | read | 1 | ok | 526.4 | 122.19 | 18.17 | 455.0 | 0.105 | 0.138 | - | PASS | 2553 |
| 256x8 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 199.80 | 20.56 | - | 0.974 | - | 33.0/26.8 | PASS | 5696 |
| 256x8 | 6T | off | write | 1 | ok | 67.7 | 226.51 | 20.55 | -51.3 | - | - | 33.0 | PASS | 3392 |
| 256x8 | 6T | on | read | 1 | ok | 513.8 | 120.20 | 18.14 | 446.4 | 0.122 | 0.138 | - | PASS | 3013 |
| 256x8 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 198.23 | 20.54 | - | 0.968 | - | 33.2/25.6 | PASS | 5733 |
| 256x8 | 6T | on | write | 1 | ok | 68.4 | 226.02 | 20.52 | -51.3 | - | - | 33.5 | PASS | 3298 |
| 512x4 | 10T | off | read | 1 | ok | 790.7 | 153.32 | 28.20 | 722.5 | 0.318 | 0.229 | - | PASS | 4660 |
| 512x4 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 236.18 | 30.29 | - | 0.970 | - | 39.7/24.9 | PASS | 9691 |
| 512x4 | 10T | off | write | 1 | ok | 79.5 | 248.17 | 30.25 | -137.0 | - | - | 41.7 | PASS | 8720 |
| 512x4 | 10T | on | read | 1 | ok | 782.7 | 152.34 | 28.25 | 714.7 | 0.330 | 0.230 | - | PASS | 4418 |
| 512x4 | 10T | on | write | 1 | ok | 77.2 | 248.99 | 30.25 | -136.9 | - | - | 39.8 | PASS | 4864 |
| 512x4 | 6T | off | read | 1 | ok | 709.3 | 148.28 | 25.00 | 639.0 | 0.238 | 0.146 | - | PASS | 6230 |
| 512x4 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 231.26 | 27.11 | - | 0.970 | - | 32.5/24.1 | PASS | 7642 |
| 512x4 | 6T | off | write | 1 | ok | 70.3 | 245.48 | 27.09 | -136.3 | - | - | 32.5 | PASS | 3870 |
| 512x4 | 6T | on | read | 1 | ok | 692.0 | 148.02 | 25.00 | 624.5 | 0.257 | 0.139 | - | PASS | 4294 |
| 512x4 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 230.49 | 27.08 | - | 0.967 | - | 31.4/24.1 | PASS | 7214 |
| 512x4 | 6T | on | write | 1 | ok | 69.5 | 245.03 | 27.06 | -136.4 | - | - | 31.5 | PASS | 3730 |
| 8x512 | 10T | off | read | 1 | ok | 642.0 | 1034.52 | 31.24 | 239.2 | 0.003 | 0.115 | - | PASS | 6999 |
| 8x512 | 10T | off | write | 1 | ok | 512.0 | 7386.73 | 72.30 | -237.4 | - | - | 163.5 | PASS | 6777 |
| 8x512 | 10T | on | read | 1 | ok | 557.1 | 805.63 | 30.12 | 146.9 | 0.017 | 0.116 | - | PASS | 6155 |
| 8x512 | 10T | on | write | 1 | ok | 509.0 | 7329.41 | 70.34 | -237.5 | - | - | 161.0 | PASS | 6840 |
| 8x512 | 6T | off | read | 1 | ok | 639.7 | 1078.47 | 26.12 | 235.6 | 0.002 | 0.065 | - | PASS | 5516 |
| 8x512 | 6T | off | write | 1 | ok | 509.3 | 7527.39 | 67.37 | -238.0 | - | - | 160.8 | PASS | 5481 |
| 8x512 | 6T | on | read | 1 | ok | 549.7 | 859.51 | 25.00 | 141.1 | 0.006 | 0.065 | - | PASS | 4735 |
| 8x512 | 6T | on | write | 1 | ok | 509.4 | 7502.03 | 65.57 | -238.3 | - | - | 160.6 | PASS | 5669 |
| 16x256 | 10T | off | read | 1 | ok | 542.4 | 444.64 | 29.64 | 306.8 | 0.004 | 0.167 | - | PASS | 5645 |
| 16x256 | 10T | off | write | 1 | ok | 273.0 | 2340.02 | 51.95 | -98.1 | - | - | 87.4 | PASS | 4924 |
| 16x256 | 10T | on | read | 1 | ok | 456.5 | 298.71 | 28.94 | 227.5 | 0.008 | 0.156 | - | PASS | 5605 |
| 16x256 | 10T | on | write | 1 | ok | 275.3 | 2314.72 | 51.10 | -96.4 | - | - | 90.2 | PASS | 4924 |
| 16x256 | 6T | off | read | 1 | ok | 535.1 | 441.69 | 24.46 | 298.9 | 0.002 | 0.100 | - | PASS | 3117 |
| 16x256 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 1573.97 | 46.78 | - | 0.933 | - | 74.5/68.3 | PASS | 11891 |
| 16x256 | 6T | off | write | 1 | ok | 260.7 | 2339.65 | 46.77 | -98.5 | - | - | 74.6 | PASS | 5444 |
| 16x256 | 6T | on | read | 1 | ok | 448.8 | 295.83 | 23.79 | 219.9 | 0.005 | 0.092 | - | PASS | 3945 |
| 16x256 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 1495.50 | 46.10 | - | 0.955 | - | 78.2/71.8 | PASS | 11948 |
| 16x256 | 6T | on | write | 1 | ok | 263.6 | 2302.01 | 46.06 | -97.1 | - | - | 78.2 | PASS | 6683 |
| 64x64 | 10T | off | read | 1 | ok | 504.9 | 210.22 | 29.28 | 395.6 | 0.006 | 0.206 | - | PASS | 4547 |
| 64x64 | 10T | off | read | 1 | ok | 1187.8 | 701.56 | 27.00 | 982.2 | 0.275 | 0.213 | - | PASS | 923 |
| 64x64 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 464.91 | 37.52 | - | 0.974 | - | 108.4/97.8 | PASS | 12031 |
| 64x64 | 10T | off | write | 1 | ok | 172.8 | 612.97 | 37.50 | 9.1 | - | - | 108.5 | PASS | 7047 |
| 64x64 | 10T | off | write | 1 | ok | 304.6 | 1137.04 | 44.67 | -133.7 | - | - | 164.6 | PASS | 1717 |
| 64x64 | 10T | on | read | 1 | ok | 439.7 | 176.42 | 29.03 | 331.1 | 0.021 | 0.205 | - | PASS | 5486 |
| 64x64 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 444.26 | 37.31 | - | 0.982 | - | 108.6/98.0 | PASS | 11922 |
| 64x64 | 10T | on | write | 1 | ok | 172.9 | 605.53 | 37.28 | 9.2 | - | - | 108.7 | PASS | 5534 |
| 64x64 | 6T | off | read | 1 | ok | 491.9 | 202.95 | 23.98 | 378.2 | 0.003 | 0.131 | - | PASS | 3155 |
| 64x64 | 6T | off | read | 1 | ok | 1051.2 | 696.40 | 21.79 | 847.2 | 0.210 | 0.136 | - | PASS | 655 |
| 64x64 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 459.88 | 32.32 | - | 0.972 | - | 97.5/90.9 | PASS | 8280 |
| 64x64 | 6T | off | write | 1 | ok | 161.9 | 610.80 | 32.29 | 9.0 | - | - | 97.6 | PASS | 3799 |
| 64x64 | 6T | off | write | 1 | ok | 296.8 | 1140.59 | 39.46 | -133.7 | - | - | 156.1 | PASS | 1591 |
| 64x64 | 6T | on | read | 1 | ok | 420.7 | 169.66 | 23.72 | 312.5 | 0.009 | 0.131 | - | PASS | 3008 |
| 64x64 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 439.50 | 32.10 | - | 0.981 | - | 97.1/90.0 | PASS | 8407 |
| 64x64 | 6T | on | write | 1 | ok | 160.4 | 601.25 | 32.09 | 9.1 | - | - | 96.8 | PASS | 3780 |
| 128x32 | 10T | off | read | 1 | ok | 510.0 | 181.10 | 30.84 | 417.0 | 0.048 | 0.213 | - | PASS | 6210 |
| 128x32 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 369.16 | 36.69 | - | 0.976 | - | 96.3/83.2 | PASS | 11961 |
| 128x32 | 10T | off | write | 1 | ok | 145.6 | 462.82 | 36.67 | 3.2 | - | - | 96.3 | PASS | 6410 |
| 128x32 | 10T | on | read | 1 | ok | 466.7 | 173.25 | 30.72 | 378.7 | 0.077 | 0.211 | - | PASS | 6147 |
| 128x32 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 363.12 | 36.57 | - | 0.975 | - | 95.6/83.1 | PASS | 11438 |
| 128x32 | 10T | on | write | 1 | ok | 144.7 | 458.87 | 36.54 | 3.3 | - | - | 95.7 | PASS | 6568 |
| 128x32 | 6T | off | read | 1 | ok | 468.3 | 174.78 | 25.40 | 374.8 | 0.021 | 0.136 | - | PASS | 3929 |
| 128x32 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 364.88 | 31.33 | - | 0.976 | - | 86.2/76.6 | PASS | 8093 |
| 128x32 | 6T | off | write | 1 | ok | 134.7 | 459.35 | 31.33 | 3.1 | - | - | 85.8 | PASS | 3649 |
| 128x32 | 6T | on | read | 1 | ok | 433.5 | 166.25 | 25.27 | 346.8 | 0.039 | 0.135 | - | PASS | 3999 |
| 128x32 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 358.57 | 31.23 | - | 0.975 | - | 83.3/77.4 | PASS | 7719 |
| 128x32 | 6T | on | write | 1 | ok | 132.6 | 455.25 | 31.22 | 3.1 | - | - | 83.3 | PASS | 4520 |
| 100x50 | 10T | off | read | 1 | ok | 517.3 | 212.69 | 36.05 | 410.6 | 0.021 | 0.209 | - | PASS | 6907 |
| 100x50 | 10T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 468.99 | 43.99 | - | 0.974 | - | 108.7/98.0 | PASS | 16446 |
| 100x50 | 10T | off | write | 1 | ok | 168.1 | 607.95 | 43.97 | 15.5 | - | - | 108.7 | PASS | 6204 |
| 100x50 | 10T | on | read | 1 | ok | 463.5 | 198.41 | 35.83 | 364.0 | 0.050 | 0.209 | - | PASS | 6293 |
| 100x50 | 10T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 458.41 | 43.79 | - | 0.977 | - | 108.8/97.8 | PASS | 16169 |
| 100x50 | 10T | on | write | 1 | ok | 168.3 | 601.57 | 43.76 | 15.6 | - | - | 108.8 | PASS | 6052 |
| 100x50 | 6T | off | read | 1 | ok | 488.0 | 205.38 | 29.69 | 383.8 | 0.008 | 0.134 | - | PASS | 4339 |
| 100x50 | 6T | off | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 464.19 | 37.57 | - | 0.974 | - | 98.1/89.9 | PASS | 8623 |
| 100x50 | 6T | off | write | 1 | ok | 157.1 | 605.49 | 37.55 | 15.3 | - | - | 98.1 | PASS | 4745 |
| 100x50 | 6T | on | read | 1 | ok | 438.1 | 189.32 | 29.48 | 336.3 | 0.018 | 0.134 | - | PASS | 4213 |
| 100x50 | 6T | on | read&write | 1 | ok | TVOUT_PERIOD 40.00 ns | 453.37 | 37.36 | - | 0.977 | - | 98.1/88.8 | PASS | 8710 |
| 100x50 | 6T | on | write | 1 | ok | 157.9 | 598.41 | 37.34 | 15.4 | - | - | 98.1 | PASS | 5047 |
| 16x512 | 10T | off | read | 1 | ok | 655.9 | 1228.06 | 57.98 | 252.2 | 0.003 | 0.127 | - | PASS | 8766 |
| 16x512 | 10T | off | write | 1 | ok | 472.4 | 7729.31 | 103.01 | -205.7 | - | - | 123.2 | PASS | 12318 |
| 16x512 | 10T | on | read | 1 | ok | 567.2 | 952.70 | 56.56 | 158.2 | 0.043 | 0.126 | - | PASS | 8259 |
| 16x512 | 10T | on | write | 1 | ok | 471.5 | 7676.36 | 101.26 | -205.5 | - | - | 122.0 | PASS | 12626 |
| 16x512 | 6T | off | read | 1 | ok | 649.8 | 1286.87 | 47.64 | 244.9 | 0.002 | 0.072 | - | PASS | 6021 |
| 16x512 | 6T | off | write | 1 | ok | 460.4 | 7830.10 | 92.63 | -206.5 | - | - | 112.1 | PASS | 9074 |
| 16x512 | 6T | on | read | 1 | ok | 559.2 | 1021.48 | 46.35 | 150.9 | 0.015 | 0.072 | - | PASS | 5711 |
| 16x512 | 6T | on | write | 1 | ok | 460.5 | 7790.74 | 91.24 | -206.4 | - | - | 112.1 | PASS | 9965 |

