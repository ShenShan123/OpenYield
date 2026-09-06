# Changelog

## V2.0.1 — 2026-09-05 — transient / Monte Carlo circuit review

Scope of this release: the SRAM compiler circuits (6T and 10T cores, replica
column, timing generator, decoder, wordline driver, precharge, column mux,
sense amplifier, write driver, output latch), the transient testbenches
(`read`, `write`, `read&write`), their measurements, and the Xyce simulation /
result-parsing flow. Optimisation and yield-estimation algorithms were not
reviewed in this round.

Every change below was validated by running the generated netlists in Xyce
7.4 and scoring the waveforms automatically (wordline, bitlines, replica
bitline, precharge, sense enable, write enable, sense-amp outputs, output
latch, target-cell Q/QB) for every configuration. The final sweep covers
29 planned array sizes from 1x1 to 512x4, 16x512 and 100x50 (the 128x128 and
256x64 decks were stopped for run time, see the evidence section), both
cells, column mux on and off, all three operations, nominal and seeded Monte
Carlo. The evidence tables are at the end of this entry; the per-run
netlists, waveforms and Xyce logs were produced by the scratch sweep harness
described in `CIRCUIT_REVIEW.md`.

### Summary of fixes

- **Write pulse too short.** `w_en` was cut by the replica bitline (~250 ps),
  a cell-strength path, while the write path (row-scaled write driver, through
  the column mux) is weaker. Seeded 5 % sigma Monte Carlo samples left the
  bitline at 0.3-0.4 V and the cell kept its old data. `w_en` now spans the
  wordline phase; the hard-coded 16x512 `WenDelayChain` hack is removed.
- **Hold hazard introduced by that fix, caught by the sweep.** At the clock
  edge that ends a write the data register updates before the drivers
  release, so the next cycle's data was briefly written; at 64 rows this
  flipped the freshly written cell (`read&write` 64x16 failed). A per-column
  write-data hold latch fixes it.
- **Write testbench topology.** It had no bitline precharge and no sense-amp
  / mux load, so the stand-alone write delay was 30-40 % optimistic against
  the same write inside the `read&write` sequence. All transient decks now
  carry the full column periphery; `TWDRV` is measured on the driven bitline.
- **Nominal runs were random samples.** Every deck emitted `.SAMPLING` with
  a random seed, so identical calls returned different delays and
  occasionally failed. `mc_runs=1` is now deterministic; `mc_seed` makes
  Monte Carlo sweeps reproducible; the Xyce console log is kept per run.
- **Energy window** measured the start-up charging of the bitlines from 0 V
  (half of the "read energy" on 8x4) instead of a steady-state cycle; it is
  now one full clock period starting at the access. `read&write` averages
  over one 4-cycle pattern and its transient covers the 8th access.
- **Measurement details.** `TS_EN` was corrupted by a precharge-coupling bump
  on `s_en` (now measured from the access phase); the `w_en` buffer was not
  scaled with the row-dependent write-driver size (release lagged the
  wordline by 40-285 ps on 64-256-row arrays); the 10T core ignored the
  testbench RC parameters; a PySpice subcircuit-name collision (one
  definition per name and scope) is avoided with a dedicated `AND2_WEN`.
- **Xyce Newton stall on some 512-row decks** (residual 1e-12 A at every
  step size, not a circuit fault): the flow retries once with a 20 ps maximum
  time step, which keeps results of converging decks within 0.5 %;
  `t_max_step` and `xyce_options` are exposed on `Sram6TCoreMcTestbench`.
- **Static-review fixes carried into this release** (details in
  `CIRCUIT_REVIEW.md`): column-mux port mismatch that aborted every muxed
  read, free-running `SEL` pulse, wrong output-latch index and floating latch
  input on writes, replica column driven by the real wordlines, CS start-up
  clamp fighting the flip-flop, negative read delay and negative dynamic
  power from mis-placed measure thresholds and windows, write delay
  over-reported 2.2x by summing overlapping segments, `FAILED` measures
  silently becoming 0.0, `.prn` / SNM parsing that depended on `.PRINT`
  ordering, equivalent-circuit caps only inserted with `w_rc`, unused `regex`
  import, `python-graphviz` pip name, duplicated `config.py`, `mW` label.

### Not completed / left open

- **Array sizes not finished at release time.** 128x128 and 256x64 (all
  operations): the read transient reached 2.0 ns of 21 ns after 64 min
  (~11 h per read deck, ~45 h per `read&write` deck), so these 24 decks were
  stopped. The `read&write` decks of 8x512, 16x512, 16x256 and 10T 512x4
  (86 ns transients) were still running after 3-5 h and were not waited for;
  their `read` and `write` decks completed and are in the evidence table.
  All 283 completed decks pass every waveform check. The same flow runs them
  offline; use `t_max_step` / `xyce_options` if Xyce stalls.
- **Address-path hold hazard** (pre-existing, not exercised): the access
  window ends at the next capture edge, so a *changed* address would raise a
  second wordline for ~100-250 ps while the old one is still falling. The
  testbenches keep the address constant. A hold latch on `A_dff` (transparent
  while `wl_en` is low) or a delayed register clock would close it.
- **Design choices verified and left as they are** (see Observations below):
  sensing waits for a fully discharged replica bitline plus a 9-stage delay
  chain, so read delay is ~300 ps for every size up to 32 rows; the precharge
  is a ~300 ps self-timed pulse after which the bitlines float and droop to
  ~0.93 V; the `s_en` buffer keeps the columns/64 scaling; the `w_rc=True`
  default of `main_sram.py` puts 1 fF on every cell's Q/QB and triples the
  write delay; the fixed 10 ns clock leaves > 4 ns of margin for every size
  tested; `read` always reads a stored 0 (`read&write` covers both values).
- **Out of scope this round:** optimisation and yield-estimation algorithms,
  the SNM extraction beyond a sanity run (6T hold/read/write 0.325 / 0.182 /
  0.365 V, 10T 0.485 / 0.290 / 0.419 V), and parameter-sweep (`sweep_*`) modes.

### Circuit topology

- **Write testbench uses the full column periphery.** The `write` deck had
  only the write drivers and a precharge on the replica column: the bitlines
  started from the artificial `.IC` state (BL = 0 V, BLB = VDD), floated at
  the written values after the write pulse, and carried no sense-amplifier /
  column-mux load. The same write inside the `read&write` sequence took
  30-40 % longer (8x4 6T: 96 ps stand-alone vs 133 ps in the sequence).
  `create_testbench()` now instantiates precharge (all columns + replica
  column), column mux and sense amplifiers for every transient operation and
  adds the write drivers on top, so a write cycle is
  precharge -> write -> precharge with the real bitline load
  (`sram_6t_core_testbench.py`).
- **Write enable spans the whole wordline phase.** `w_en` was
  `rbl_delay_bar & gated_clk_bar & we`, i.e. the write pulse ended as soon as
  the replica cell had discharged the replica bitline (~250 ps). That path
  (cell pull-down through the pass gate) is stronger than the write path
  (row-scaled write driver, optionally through the column-mux transmission
  gate), so the pulse had ~30 % nominal margin and Monte Carlo samples with a
  weak NMOS left the bitline at 0.3-0.4 V when `w_en` ended: the cell kept its
  old data (4x2 6T with mux, 5 % sigma on vth0/u0/voff: 3 of 6 seeds contained
  failing samples, e.g. seed 11 sample 0 cycle 3: BL min 0.397 V, Q stayed 1).
  `w_en` is now `gated_clk_bar & we` (new `AND2_WEN` gate in
  `time_generate.py`, own subcircuit name so it does not replace the larger
  gated-clock `AND2`), the write drivers stay on as long as the wordline, and
  the hard-coded 16x512 `WenDelayChain` special case that lengthened the pulse
  for one array size is removed. With the fix the same seeds pass every
  sample; 12-sample seeded write sweeps at 4x2 and 8x4 pass with the slowest
  sample at ~250 ps.
- **Write data is held while the write drivers are enabled.** With `w_en`
  spanning the clock-low phase, the drivers are released 150-300 ps after
  the rising clock edge that ends the cycle, but the data register `DIN_dff`
  already changes 100-150 ps after that same edge, so the drivers briefly
  wrote the *next* cycle's data into the still-selected row. Small arrays
  survived because their row-scaled write driver is weak; at 64 rows the 4x
  driver flipped the freshly written cell back (`read&write` 64x16, both
  cells: Q = 1 at 12.9 ns, 0 at 13.4 ns, OUT never rose). A per-column
  transparent-low latch (`D_LATCH`, enable = `w_en_bar`) now feeds the write
  drivers: it is transparent while `w_en` is low and holds while the drivers
  are on, and it opens two gate delays after the drivers tristate, so the
  driver input cannot change while the driver is active
  (`create_write_periphery`). The analogous address-path hazard (the decoder
  output of a *new* address rises while the old wordline is still on) is not
  exercised by the testbenches, which keep the address fixed; see the
  observations below.
- **w_en buffer scaled with its real fan-out.** The `w_en` output inverter
  scaled with columns/64 only, while each write driver it drives scales with
  `max(8, rows)/16`; on 64-256-row arrays `w_en` was released 40-70 ps after
  the wordline (285 ps with RC at 64x64) and the drivers overlapped the start
  of the precharge. The row factor is now included; arrays up to 32 rows are
  unchanged.
- **Write-driver delay measured on the driven bitline.** `TWDRV` targeted
  `V(BL)` rising, which only existed because BL started at 0 V; with
  precharged bitlines a '1' is written by pulling BLB low, so `TWDRV` is now
  `w_en` rise -> `V(BLB) = VDD/2` falling.
- 10T core now receives the testbench `pi_res` / `pi_cap` in both the
  standard and the `custom_mc` path (it silently used the class defaults).
- Carried over from the static review (see `CIRCUIT_REVIEW.md`): column-mux
  instantiated with two extra `SELB` ports (aborted every muxed read in Xyce);
  `SEL` was a free-running pulse unrelated to the clock, now a static level;
  output D-latch used `SA_Q{target_col}` instead of `SA_Q{target_col // mux_in}`
  and was created for pure writes with a floating input; `choose_columnmux`
  with `num_cols % mux_in != 0` produced columns without a sense amplifier
  (now rejected); replica column tied its non-replica cells to the real
  wordlines so two cells discharged RBL on every access (now dummies tied to
  VSS); the CS start-up clamp fought the CS flip-flop for 200 ps after the
  first capture edge and re-engaged in cycle 3 (released before the edge,
  one-shot, slave node initialised); stimulus sources used a literal 1.0 V
  instead of `vdd`; `AND2` RC network was a dangling stub; sense-amp pass
  width and wordline-driver column scaling were dropped in parameter-sweep
  mode; `hold_snm` bitlines had no DC path.

### Measurements

- **Energy window is one full clock period starting at the access.**
  `EREAD` / `EWRITE` were integrated over 2 ns .. 2 ns + T, a window that
  contains the start-up precharge charging every bitline from its 0 V initial
  condition (156 fJ of the 298 fJ "read energy" on an 8x4 6T array) and cuts
  the access off 1 ns before the wordline falls, while the real post-access
  precharge (109 fJ) was excluded. The window is now
  `1 ns + 0.7 T` .. `1 ns + 1.7 T`: wordline access, sensing / writing, the
  self-timed precharge, and the idle time to the next access.
- `TS_EN` (s_en 20 %-80 % rise) is measured only from the start of the
  access phase (`TD=` on trigger and target). When the first precharge fires,
  all bitlines rise together and couple through the sense-amplifier pass
  gates into the weakly held `s_en` net; with 32-128 sense amplifiers that
  bump reaches 0.2-0.3 V and was taken as the 20 % crossing, so `TS_EN` read
  ~5.3 ns on 32x32 and 2x128 arrays.
- `read&write`: `PAVG` averages over one complete write-1 / read / write-0 /
  read pattern (4 T) starting at the first access, and the transient runs to
  `1 ns + 8.5 T` so the 8th access is no longer cut 1 ns after its wordline
  rises. `PSTC` / `PDYN` are now produced for this operation too.
- Carried over from the static review: read delay is `TREAD_TOTAL` (`wl_en`
  -> output latch) and write delay is `TWRITE_TOTAL` (`wl_en` -> Q at 90 %)
  instead of a sum of overlapping segments that over-reported the write delay
  2.2x; `TSA` / `TREAD_TOTAL` target the output latch instead of a 10 mV
  threshold on `SA_Q` that sat on the `.IC` parking level (read delay was
  -5.1 ns); `TS_EN` uses the same edge for trigger and target; `PSTC` is
  measured in a quiescent window (`1 ns + [0.4, 0.65] T`) instead of the
  start-up transient (PDYN was negative); `TDECODER` triggers on the lowest
  set address bit; a `FAILED` measure raises instead of contributing 0.0.

### Simulation control and reproducibility

- **`mc_runs = 1` is the nominal point.** Every deck carried
  `.SAMPLING useExpr=true` with a fresh random seed, so a single run was one
  random process sample: identical calls returned different delay / power
  numbers, `size_optimization` evaluated a noisy objective, and the random
  sample occasionally failed the write (the errors above were first seen as
  random single-run failures). `.SAMPLING` is now only emitted for
  `mc_runs > 1`; a single run evaluates every `AGAUSS(...)` at its mean and
  is bit-for-bit repeatable.
- New `mc_seed=<int>` argument of `Sram6TCoreMcTestbench` writes
  `.options samples numsamples=N seed=S`, so a Monte Carlo sweep is
  reproducible (verified: two seeded 6-sample sweeps give identical results).
- The Xyce console output of every run is kept as `<netlist>.log` next to the
  netlist (it was discarded on success, losing all netlist warnings and the
  random seed).
- Carried over: `sweep_senseamp` defaulted to `True` in the MC testbench
  (3-point `.STEP` that broke the block splitter); `.OPTIONS MEASURE MEASFAIL=1`;
  `1/2 CLK` margin adds the spread instead of discarding it.

### Parsing, configuration, documentation

- Carried over: `.prn` reader accepts both `Index` / no-`Index` layouts (the
  flow depended on `.PRINT` ordering); the SNM splitter handles
  `FORMAT=NOINDEX`; write-SNM uses the constriction beyond the eye of the
  remaining lobe instead of a global maximum pinned to the sweep bound;
  equivalent-circuit parasitic caps are inserted regardless of `w_rc`, the
  extraction is cached per process, static-power model failures propagate,
  `_PROJECT_ROOT` fixed; unused `regex` import removed; `python-graphviz`
  -> `graphviz` in `environment.yml`; `config.py` re-exports
  `sram_compiler/config_yaml/config.py`; `demo_run_a_testbench.py` paths and
  keyword; 10T `process_parameters.vars` made 2-D; temperature passed in
  `size_optimization/exp_utils.py`.
- `main_sram.py` prints power in µW (the value was µW, the label said mW).
- `readme_compiler.md` documents the energy window, the nominal single run,
  `mc_seed` and the write-testbench topology.

### Observations (verified, not changed)

- The read delay is ~300 ps almost independently of array size up to 32
  rows because the sense enable is derived from a fully discharged replica
  bitline plus a 9-stage delay chain: the sense amplifier fires ~250 ps after
  the wordline while the target bitline has already swung to ~0.02 V
  (`vswing` = 250 mV is reached after 10-35 ps). Sensing is therefore
  full-swing and robust but slow; the 10 ns clock leaves >4 ns of margin for
  every size tested.
- The precharge is a self-timed ~300 ps pulse; afterwards the bitlines float
  for the rest of the cycle. They overshoot to 1.03-1.07 V from the `PRE`
  gate coupling and then leak down (to 0.93-0.95 V by the next access on the
  10T 16x16 replica bitline and on the 2-row 128-column array). Functionally
  harmless in the sweep, but bitline levels at the start of an access depend
  on the cycle time.
- The `w_rc=True` default of `main_sram.py` places `pi_cap` (1 fF) on the
  internal storage nodes Q/QB of every cell as well as on BL/BLB/WL; this
  triples the write delay (16x16 6T: 98 ps -> 346 ps) and doubles the power.
  It is a parameter choice, not a code defect, but the default is heavy.
- `read` always reads a stored 0; `read&write` covers both data values.
- The equivalent-circuit model (`real_cell_mode=1`) tracks the all-real array
  with the same RC model within 6-12 % on delay and 5 % on power at 16x16
  (`w_rc=True`, 6T: read 501 vs 533 ps, write 300 vs 263 ps, 98 vs 97 µW;
  10T: read 541 vs 591 ps, write 275 vs 257 ps). The evidence table below
  lists the equivalent-circuit runs next to the no-RC all-real runs; most of
  that difference is the RC model itself (see the `w_rc` note above).
- The end of the access window coincides with the next capture edge, so any
  input that changes at that edge (address, data, chip select) races the
  release of the wordline. The data path is now held (see above); the address
  path is not exercised by the testbenches (the target address is constant),
  but a decoder output of a new address would rise ~250 ps after the edge
  while `WL_EN` is still falling, briefly selecting a second row. A hold
  latch on `A_dff` (transparent while `wl_en` is low) or a delayed register
  clock would close it.
- The `s_en` buffer keeps the authors' columns/64 scaling; it slows the sense
  enable by ~60 ps at 128 columns and lets the precharge-coupling bump on
  `s_en` reach 0.23 V. Not functional, but worth a fan-out-aware size like
  the wordline driver's.
- Xyce convergence at 512 rows: some nominal 512x4 decks stop with "time
  step too small" during the access (the Newton loop oscillates, 21
  iterations with a residual of only 1e-12 A, at every step size). Relaxing
  `ABSTOL`, `NLNEARCONV` or `MAXSTEP` does not help; `ERROPTION=1` completes
  the deck but shifts delays of converging decks by 2-15 % and power by up to
  2.4 %; a `.TRAN` maximum step of 20 ps completes it and keeps converging
  decks within 0.5 % (delay) / 0.4 % (energy) at ~1.8x the time steps.
  `run_mc_simulation()` therefore retries once with that maximum step when
  Xyce reports the failure (new `t_max_step` and `xyce_options` arguments of
  `Sram6TCoreMcTestbench` expose both knobs); details in the solver notes of
  `CIRCUIT_REVIEW.md`.
- A `.SAMPLING` run of the old netlist on 10T 64x16 with mux aborted with a
  Xyce "time step too small" at the start of the access; the nominal run with
  the corrected topology completes (139 ps).

### Evidence

Filled in below from the final sweep (nominal, all cells real, `w_rc=False`,
25 °C, TT, target cell = last row / last column).

#### Size sweep (nominal, all cells real, no RC, mux off / on)

283 of 318 configurations simulated (35 not finished at
release time), 283 completed in Xyce, 283 pass every waveform check.
Errors: none.
Not completed at release time (35): 10T_512x4_m1_rw, 10T_16x256_m0_rw, 10T_16x256_m1_rw, 6T_8x512_m0_rw, 6T_8x512_m1_rw, 10T_8x512_m0_rw, 10T_8x512_m1_rw, 6T_16x512_m0_rw, 6T_16x512_m1_rw, 10T_16x512_m0_rw, 10T_16x512_m1_rw, 6T_128x128_m0_read, 6T_128x128_m0_write, 6T_128x128_m0_rw, 6T_128x128_m1_read, 6T_128x128_m1_write, 6T_128x128_m1_rw, 10T_128x128_m0_read, 10T_128x128_m0_write, 10T_128x128_m0_rw, 10T_128x128_m1_read, 10T_128x128_m1_write, 10T_128x128_m1_rw, 6T_256x64_m0_read, 6T_256x64_m0_write, 6T_256x64_m0_rw, 6T_256x64_m1_read, 6T_256x64_m1_write, 6T_256x64_m1_rw, 10T_256x64_m0_read, 10T_256x64_m0_write, 10T_256x64_m0_rw, 10T_256x64_m1_read, 10T_256x64_m1_write, 10T_256x64_m1_rw.
The 128x128 and 256x64 decks (16k cells, ~100k transistors) were stopped after
64 min at 2.0 ns of the 21 ns read transient (~11 h per read deck, ~45 h per
read&write deck at that rate); the 8x512 / 16x512 / 16x256 / 512x4-10T
read&write decks (8.5 clock cycles, 86 ns) were still running at ~40-70 ns
after 3-5 h. Their read and write decks completed and are in the table.
Delays in ps (read: `wl_en` rise -> `OUT`; write: `wl_en` rise -> Q at 90 %);
`read&write` shows the measured period of `OUT` (40 ns = 4 clock cycles is the
correct write-1 / read / write-0 / read sequence); PAVG at 100 MHz.

| array | cell | read delay [ps] mux off / on | write delay [ps] mux off / on | read&write OUT period mux off / on | PAVG read / write [µW] | waveform checks |
|---|---|---|---|---|---|---|
| 1x1 | 10T | 289 / n/a | 133 / n/a | 40 ns / n/a | 20.1 / 21.8 | PASS (3 runs) |
| 1x1 | 6T | 286 / n/a | 131 / n/a | 40 ns / n/a | 20.1 / 22.0 | PASS (3 runs) |
| 2x1 | 10T | 290 / n/a | 134 / n/a | 40 ns / n/a | 21.5 / 23.1 | PASS (3 runs) |
| 2x1 | 6T | 287 / n/a | 132 / n/a | 40 ns / n/a | 21.4 / 23.3 | PASS (3 runs) |
| 2x2 | 10T | 293 / 290 | 134 / 164 | 40 ns / 40 ns | 22.2 / 26.4 | PASS (6 runs) |
| 2x2 | 6T | 290 / 287 | 132 / 174 | 40 ns / 40 ns | 22.2 / 26.8 | PASS (6 runs) |
| 4x2 | 10T | 295 / 292 | 136 / 166 | 40 ns / 40 ns | 24.1 / 28.3 | PASS (6 runs) |
| 4x2 | 6T | 292 / 289 | 134 / 176 | 40 ns / 40 ns | 24.0 / 28.8 | PASS (6 runs) |
| 3x3 | 10T | 297 / n/a | 136 / n/a | 40 ns / n/a | 23.7 / 30.5 | PASS (3 runs) |
| 3x3 | 6T | 294 / n/a | 132 / n/a | 40 ns / n/a | 23.6 / 31.0 | PASS (3 runs) |
| 5x3 | 10T | 299 / n/a | 138 / n/a | 40 ns / n/a | 24.5 / 31.3 | PASS (3 runs) |
| 5x3 | 6T | 296 / n/a | 134 / n/a | 40 ns / n/a | 24.4 / 31.9 | PASS (3 runs) |
| 4x4 | 10T | 301 / 295 | 138 / 168 | 40 ns / 40 ns | 25.7 / 35.1 | PASS (6 runs) |
| 4x4 | 6T | 298 / 292 | 133 / 176 | 40 ns / 40 ns | 25.7 / 36.0 | PASS (6 runs) |
| 16x1 | 10T | 308 / n/a | 86 / n/a | 40 ns / n/a | 28.3 / 29.9 | PASS (3 runs) |
| 16x1 | 6T | 302 / n/a | 73 / n/a | 40 ns / n/a | 28.3 / 29.8 | PASS (3 runs) |
| 8x4 | 10T | 305 / 300 | 142 / 172 | 40 ns / 40 ns | 28.0 / 37.7 | PASS (6 runs) |
| 8x4 | 6T | 301 / 295 | 138 / 181 | 40 ns / 40 ns | 28.0 / 38.4 | PASS (6 runs) |
| 6x6 | 10T | 308 / 301 | 142 / 171 | 40 ns / 40 ns | 28.4 / 43.2 | PASS (6 runs) |
| 6x6 | 6T | 305 / 297 | 137 / 180 | 40 ns / 40 ns | 28.3 / 44.5 | PASS (6 runs) |
| 12x4 | 10T | 311 / 305 | 106 / 119 | 40 ns / 40 ns | 29.8 / 38.7 | PASS (6 runs) |
| 12x4 | 6T | 306 / 300 | 93 / 110 | 40 ns / 40 ns | 29.5 / 38.8 | PASS (6 runs) |
| 8x8 | 10T | 315 / 306 | 147 / 174 | 40 ns / 40 ns | 31.6 / 52.2 | PASS (6 runs) |
| 8x8 | 6T | 311 / 301 | 141 / 183 | 40 ns / 40 ns | 31.4 / 53.7 | PASS (6 runs) |
| 16x8 | 10T | 327 / 316 | 98 / 105 | 40 ns / 40 ns | 36.2 / 56.0 | PASS (6 runs) |
| 16x8 | 6T | 320 / 310 | 85 / 93 | 40 ns / 40 ns | 35.8 / 55.7 | PASS (6 runs) |
| 20x10 | 10T | 336 / 324 | 95 / 99 | 40 ns / 40 ns | 40.1 / 66.0 | PASS (6 runs) |
| 20x10 | 6T | 329 / 317 | 83 / 87 | 40 ns / 40 ns | 39.8 / 65.4 | PASS (6 runs) |
| 2x128 | 10T | 463 / 386 | 202 / 224 | 40 ns / 40 ns | 158.8 / 734.0 | PASS (6 runs) |
| 2x128 | 6T | 458 / 382 | 191 / 225 | 40 ns / 40 ns | 158.4 / 740.1 | PASS (6 runs) |
| 16x16 | 10T | 344 / 327 | 111 / 116 | 40 ns / 40 ns | 45.3 / 88.8 | PASS (6 runs) |
| 16x16 | 6T | 337 / 320 | 98 / 105 | 40 ns / 40 ns | 44.9 / 89.1 | PASS (6 runs) |
| 32x8 | 10T | 347 / 337 | 84 / 86 | 40 ns / 40 ns | 43.9 / 65.7 | PASS (6 runs) |
| 32x8 | 6T | 336 / 327 | 72 / 73 | 40 ns / 40 ns | 43.5 / 65.6 | PASS (6 runs) |
| 32x32 | 10T | 396 / 364 | 148 / 149 | 40 ns / 40 ns | 81.9 / 227.4 | PASS (6 runs) |
| 32x32 | 6T | 386 / 355 | 137 / 138 | 40 ns / 40 ns | 81.1 / 227.8 | PASS (6 runs) |
| 64x16 | 10T | 401 / 383 | 125 / 125 | 40 ns / 40 ns | 76.3 / 166.6 | PASS (6 runs) |
| 64x16 | 6T | 382 / 366 | 114 / 114 | 40 ns / 40 ns | 75.1 / 165.8 | PASS (6 runs) |
| 256x8 | 10T | 579 / 566 | 78 / 79 | 40 ns / 40 ns | 124.9 / 229.2 | PASS (6 runs) |
| 256x8 | 6T | 526 / 514 | 68 / 68 | 40 ns / 40 ns | 122.2 / 226.5 | PASS (6 runs) |
| 512x4 | 10T | 791 / 783 | 80 / 77 | 40 ns / n/a | 153.3 / 248.2 | PASS (5 runs) |
| 512x4 | 6T | 709 / 692 | 70 / 69 | 40 ns / 40 ns | 148.3 / 245.5 | PASS (6 runs) |
| 8x512 | 10T | 642 / 557 | 512 / 509 | n/a / n/a | 1034.5 / 7386.7 | PASS (4 runs) |
| 8x512 | 6T | 640 / 550 | 509 / 509 | n/a / n/a | 1078.5 / 7527.4 | PASS (4 runs) |
| 16x256 | 10T | 542 / 457 | 273 / 275 | n/a / n/a | 444.6 / 2340.0 | PASS (4 runs) |
| 16x256 | 6T | 535 / 449 | 261 / 264 | 40 ns / 40 ns | 441.7 / 2339.6 | PASS (6 runs) |
| 64x64 | 10T | 505 / 440 | 173 / 173 | 40 ns / 40 ns | 210.2 / 613.0 | PASS (6 runs) |
| 64x64 | 6T | 492 / 421 | 162 / 160 | 40 ns / 40 ns | 202.9 / 610.8 | PASS (6 runs) |
| 128x32 | 10T | 510 / 467 | 146 / 145 | 40 ns / 40 ns | 181.1 / 462.8 | PASS (6 runs) |
| 128x32 | 6T | 468 / 434 | 135 / 133 | 40 ns / 40 ns | 174.8 / 459.4 | PASS (6 runs) |
| 100x50 | 10T | 517 / 464 | 168 / 168 | 40 ns / 40 ns | 212.7 / 607.9 | PASS (6 runs) |
| 100x50 | 6T | 488 / 438 | 157 / 158 | 40 ns / 40 ns | 205.4 / 605.5 | PASS (6 runs) |
| 16x512 | 10T | 656 / 567 | 472 / 471 | n/a / n/a | 1228.1 / 7729.3 | PASS (4 runs) |
| 16x512 | 6T | 650 / 559 | 460 / 460 | n/a / n/a | 1286.9 / 7830.1 | PASS (4 runs) |

#### Monte Carlo (Xyce `.SAMPLING`, `vth_std = 0.05`, seed 2026)

| array | cell | mux | op | samples | delay mean ± sd [ps] | min / max [ps] | PAVG mean [µW] | waveform checks (all samples) |
|---|---|---|---|---|---|---|---|---|
| 8x4 | 10T | off | read | 5 | 309.5 ± 3.7 | 303.9 / 314.1 | 27.8 | PASS |
| 8x4 | 10T | off | read&write | 3 | OUT period 40.00 ns | - | 34.9 | PASS |
| 8x4 | 10T | off | write | 5 | 152.0 ± 7.5 | 143.8 / 166.0 | 37.6 | PASS |
| 8x4 | 10T | on | read | 5 | 303.8 ± 3.6 | 298.3 / 308.4 | 26.8 | PASS |
| 8x4 | 10T | on | read&write | 3 | OUT period 40.00 ns | - | 34.2 | PASS |
| 8x4 | 10T | on | write | 5 | 184.6 ± 10.3 | 172.7 / 203.5 | 37.0 | PASS |
| 8x4 | 6T | off | read | 5 | 305.1 ± 3.6 | 299.6 / 309.7 | 27.8 | PASS |
| 8x4 | 6T | off | read&write | 3 | OUT period 40.00 ns | - | 35.3 | PASS |
| 8x4 | 6T | off | write | 5 | 153.3 ± 14.2 | 139.9 / 180.3 | 38.6 | PASS |
| 8x4 | 6T | on | read | 5 | 299.5 ± 3.5 | 294.2 / 303.9 | 26.8 | PASS |
| 8x4 | 6T | on | read&write | 3 | OUT period 40.00 ns | - | 34.6 | PASS |
| 8x4 | 6T | on | write | 5 | 202.8 ± 20.6 | 182.1 / 241.6 | 38.1 | PASS |
| 16x16 | 10T | off | read | 5 | 348.3 ± 3.8 | 342.9 / 353.3 | 44.8 | PASS |
| 32x8 | 10T | off | read | 5 | 351.3 ± 4.0 | 345.6 / 357.0 | 43.4 | PASS |
| 16x16 | 10T | off | write | 5 | 114.8 ± 2.8 | 111.8 / 119.9 | 88.4 | PASS |
| 32x8 | 10T | off | write | 5 | 86.6 ± 1.6 | 84.9 / 89.4 | 65.3 | PASS |
| 16x16 | 10T | on | read | 5 | 330.9 ± 3.8 | 325.1 / 335.3 | 40.8 | PASS |
| 32x8 | 10T | on | read | 5 | 342.3 ± 4.4 | 336.1 / 347.8 | 41.3 | PASS |
| 16x16 | 10T | on | write | 5 | 120.9 ± 3.4 | 117.4 / 127.4 | 85.8 | PASS |
| 32x8 | 10T | on | write | 5 | 88.3 ± 1.8 | 86.4 / 91.4 | 64.4 | PASS |
| 16x16 | 6T | off | read | 5 | 341.6 ± 4.0 | 335.4 / 346.4 | 44.3 | PASS |
| 32x8 | 6T | off | read | 5 | 341.6 ± 4.3 | 335.1 / 347.4 | 43.2 | PASS |
| 16x16 | 6T | off | write | 5 | 102.5 ± 3.0 | 99.5 / 108.3 | 88.5 | PASS |
| 32x8 | 6T | off | write | 5 | 74.8 ± 2.0 | 72.9 / 78.4 | 64.9 | PASS |
| 16x16 | 6T | on | read | 5 | 324.6 ± 3.5 | 319.1 / 329.0 | 40.5 | PASS |
| 32x8 | 6T | on | read | 5 | 331.5 ± 4.5 | 324.7 / 337.2 | 41.1 | PASS |
| 16x16 | 6T | on | write | 5 | 110.4 ± 3.9 | 106.7 / 118.0 | 85.4 | PASS |
| 32x8 | 6T | on | write | 5 | 75.8 ± 1.7 | 74.1 / 78.9 | 63.6 | PASS |

#### Equivalent-circuit model (`real_cell_mode=1`, `w_rc=True`, `main_sram.py` defaults)

| array | cell | op | all cells real, no RC: delay [ps] / PAVG [µW] | real_cell_mode=1, w_rc=True: delay [ps] / PAVG [µW] | waveform checks |
|---|---|---|---|---|---|
| 16x16 | 10T | read | 344 / 45.3 | 554 / 94.6 | PASS |
| 16x16 | 10T | write | 111 / 88.8 | 336 / 158.5 | PASS |
| 16x16 | 6T | read | 337 / 44.9 | 513 / 93.9 | PASS |
| 16x16 | 6T | write | 98 / 89.1 | 334 / 165.7 | PASS |
| 32x32 | 10T | read | 396 / 81.9 | 770 / 231.6 | PASS |
| 32x32 | 10T | write | 148 / 227.4 | 340 / 404.1 | PASS |
| 32x32 | 6T | read | 386 / 81.1 | 695 / 229.7 | PASS |
| 32x32 | 6T | write | 137 / 227.8 | 334 / 414.5 | PASS |
| 64x64 | 10T | read | 505 / 210.2 | 1188 / 701.6 | PASS |
| 64x64 | 10T | write | 173 / 613.0 | 305 / 1137.0 | PASS |
| 64x64 | 6T | read | 492 / 202.9 | 1051 / 696.4 | PASS |
| 64x64 | 6T | write | 162 / 610.8 | 297 / 1140.6 | PASS |
