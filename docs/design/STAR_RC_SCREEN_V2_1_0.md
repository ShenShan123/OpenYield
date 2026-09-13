# V2.1.0: star-shape RC screen

Screen of the V2.1.0 working tree (on top of commit `0433072`) for star-shape
RC networks. The target is a distributed RC load model on every read and write
path. This screen covers topology only. It changes no code and makes no
waveform, timing or yield claim.

## Summary

Star-shape RC is present in V2.1.0 and remains the default. Enabling
`interconnect.mode: distributed` makes the array wires, replica wires and array-spanning
control lines distributed, but leaves several lumped junctions on the read,
write and address paths (section 3).

## 1. Star topology is the default

| Entry point | Resolution |
|---|---|
| `sram_compiler/config_yaml/global.yaml` | No `interconnect` block; `config.py:61` falls back to `{"mode": "star"}` |
| `InterconnectConfig` | `mode: str = 'star'` (`interconnect.py:73`) |
| `main_sram.py` | `INTERCONNECT_CONFIG = None` (`main_sram.py:52`) keeps the global setting |
| `python3 -m sram_compiler.per_device_mc.run` | Star unless `--interconnect-config` is given |
| Optimizer | `size_optimization/experiment.py:389` and `exp_utils.py:1151` inherit `global_config.interconnect` |
| Yield estimation | Reaches the testbench through `exp_utils`, so it inherits the same setting |

In star mode every array-spanning wire is a single node:

- Array core: each `BL{c}`/`BLB{c}` joins all row cells, each `WL{r}` all column cells.
- Replica column: `RBL`/`RBLB` join all `num_rows + 1` replica cells.
- Top level: `PRE`, `w_en`, `w_en_bar`, `s_en`, `sa_iso`, `WL_EN` and `RWL`
  each join every consumer.
- `cell_pin_rc` defaults to true for star (`interconnect.py:83-84`). With
  `w_rc=True` each cell adds its own 100 Ω / 1 fF WL/BL/BLB branch from the
  shared node (`sram_6t_core.py:76-79`), which is a literal star.
- Equivalent modes 1–4 lump every omitted cell of a row or column into one
  capacitor behind `pi_res / unused` (`sram_cell_add_equivalent.py:924-930`,
  `954-975`).

Measured in the generated 8×4 star `read&write` deck (no mux, `w_rc=True`):

| Scope | Net | Loads joined without wire R |
|---|---|---:|
| Array core | `BL{c}`, `BLB{c}` | 8 cells |
| Array core | `WL{r}` | 4 cells |
| Replica column | `RBL`, `RBLB` | 9 cells |
| Top level | `WL_EN` | 8 wordline drivers |
| Top level | `PRE`, `RWL`, `w_en` | 6 |
| Top level | `s_en`, `sa_iso`, `w_en_bar` | 5 |

The deck contains 49 resistors and no wire-segment resistors.

## 2. Distributed-mode coverage

With `sram_compiler/config_yaml/interconnect_example.yaml` the following are
tapped pi ladders (`interconnect.add_tapped_line()`), and every consumer
connects at its own tap:

- Array `WL{r}`, `BL{c}`, `BLB{c}` (`interconnect.add_array_wires()`, used by
  both the 6T and 10T cores).
- Replica `RBL`/`RBLB` and `RWL`.
- Control lines `PRE`, `w_en`, `w_en_bar`, `s_en`, `sa_iso` and `wl_en`
  (`Sram6TCoreTestbench._add_control_wires()`).
- Equivalent modes attach each omitted cell at its local taps
  (`sram_cell_add_equivalent.py:932-952`).

In the 8×4 distributed deck, neither the array core nor the replica column has
a net with three or more loads. The deck contains 326 resistors, 288 of them
wire segments.

## 3. Lumped junctions remaining in distributed mode

| # | Node | Loads joined without wire R | Path | Source |
|---|---|---|---|---|
| A | `BL{c}`, `BLB{c}` near end | Precharge, write driver, sense amplifier (or column mux) and the array port | Read and write | `sram_6t_core_testbench.py:633-637`, `681-694`, `750-758`, `834-843`, `1223-1226` |
| B | `RBL`, `RBLB` near end | Replica precharge, replica sense amplifier or mux, write-driver load and replica column; TIME as well when `w_rc=False` | Replica timing | `sram_6t_core_testbench.py:639-643`, `150-151`, `673-676`, `730-734`, `791-793` |
| C | Decoder address lines | `create_decoder()` passes `w_rc=False`; in `DECODER_CASCADE` each low address bit reaches every 3-to-8 block with no RC although the lines span the row height | Address to wordline | `sram_6t_core_testbench.py:473`, `decoder.py:234-239` |
| D | Write-data register clock | `DATA_DFF` `CLK` feeds one flip-flop per column | Write data | `time_generate.py:680` |
| E | Column-mux `SEL{i}` | One DC source shared by every mux group | Read (mux only) | `sram_6t_core_testbench.py:700-706` |

Measured loads in the generated decks:

| Deck | Net | Loads |
|---|---|---:|
| 8×4 distributed, no mux, `w_rc=True` | `BL{c}`, `BLB{c}`, `RBL`, `RBLB` | 4 |
| 8×4 distributed, mux, `w_rc=True` | `BL{c}`, `BLB{c}` (mux instead of sense amplifier), `RBL`, `RBLB` | 4 |
| 8×4 distributed, no mux, `w_rc=False` | `RBL` (includes TIME) | 5 |
| 64×4 distributed read | `DECODER_CASCADE` `A0`, `A1`, `A2` | 8 decoder blocks each |
| 8×4 distributed | `DATA_DFF` `CLK` | 4 flip-flops |

Notes:

- A and B follow the documented placement: all bitline periphery connects at
  row zero ([model guide](DISTRIBUTED_RC_MODEL.md), "Configuration and
  topology"). No wire separates the periphery blocks along the column.
- E is a static level during the access and does not affect timing.
- The `w_rc` stubs inside the sense amplifier, write driver, precharge and
  wordline driver are series sections on each pin, not stars. They are fixed
  100 Ω / 1 fF values and do not scale with wire geometry.
- The compiler supplies no extracted wire geometry. The example file uses
  illustrative 1 Ω / 0.1 fF per pitch.

## 4. Closing the gaps

A distributed load model on every read/write path would require:

1. A distributed `interconnect` block as the default in `global.yaml`, with
   extracted geometry.
2. Wire segments between the bitline periphery blocks at the near end
   (A, B).
3. RC on the decoder address and predecode lines (C).

D and E are lower priority. None of these changes is implemented.

## Method and limits

- Code reading: `interconnect.py`, 6T/10T cores, replica and dummy columns,
  precharge/write driver, mux/sense amplifier, wordline driver, decoder, TIME,
  the equivalent-cell builder and the MC testbench probes.
- Deck generation without a simulator. The 8×4 decks use `--target-row 5 --target-col 1`:

  ```bash
  python3 -m sram_compiler.per_device_mc.run --rows 8 --cols 4 --target-row 5 \
    --target-col 1 --operation 'read&write' --variation-mode nominal \
    [--interconnect-config sram_compiler/config_yaml/interconnect_example.yaml]
  python3 -m sram_compiler.per_device_mc.run --rows 64 --cols 4 --target-row 37 \
    --target-col 1 --operation read --variation-mode nominal \
    --interconnect-config sram_compiler/config_yaml/interconnect_example.yaml
  ```

  The mux and `w_rc=False` variants were built in memory with
  `Sram6TCoreMcTestbench(..., variation_mode='nominal')`, setting
  `choose_columnmux` and `interconnect` on the loaded configuration.
- Net fan-out: an ad hoc script parsed each deck scope and flagged every net
  (other than supplies) with three or more MOS terminals or subcircuit pins
  and no resistor. The script was kept in a temporary session directory and is
  not in the repository.
- Only 6T decks were generated. 10T coverage rests on code reading: the 10T
  core uses the same `add_array_wires()` and `cell_wire_nodes()`.
- Xyce was not run. This is a topology screen, so it establishes neither waveform
  correctness nor timing, retention or sensing margin.
