# OpenRAM FreePDK45 reference assets

The immutable `cell_1rw.gds`, `replica_cell_1rw.gds`, and `sense_amp.gds` files
come from `technology/freepdk45/gds_lib/` at OpenRAM commit
`b2b069ce119d1488cbe6883b2240bceb5c7ce29a`. `precharge.gds` exports the precharge
module from the recorded 32×32 run of that checkout. The upstream BSD-3-Clause
license is retained in LICENSE; provenance.json records the source and hashes.
These GDS files are preserved reference assets; the alternative OpenRAM-pitch
bank generators have been removed. Runtime logic still reads the area ceilings.

The six core MOS widths and lengths match the OpenYield source. The placement
pitch is 705 nm per column and 1365 nm per row; native geometry extends beyond
that frame for abutment. The retired sense view removed the reference output buffers
and separates ISO from EN to reproduce OpenYield's seven-MOS source topology.
The retired passive replica and dummy views implemented OpenYield's own source
connections. Source topology/W/L checks and strict LVS cover these adaptations.

`area_reference.json` records recursive physical bounding boxes from the
reference GDS, including its hash. These are area ceilings, not a claim that
the two compilers use identical peripheral circuits. In particular, the latch
uses the reference DFF as an area ceiling, and the two-inverter data buffer uses
the sum of two reference inverter bounding boxes. There is no matched 8×8 bank
area reference. All 32×32 bank, core, decoder and driver comparisons use the
recorded 32×32 reference.
