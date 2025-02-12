# OpenYield
This project is a open-source yield analysis for SRAM circuits.

## What we have
### SRAM circuits
| Type | Components | Technology | Size | RC support |
|----------|----------|----------|----------|----------|
| 6T SRAM cell | 6T cell | TSMC 28nm | 1x1 | No
| 6T Column | 6T cell | TSMC 28nm | 256x1 | No
| 6T Column | cell+SA+drivers | TSMC 28nm | 256x1 | No
| 6T Array | 6T cell | TSMC 28nm | 256x32 | No
| 6T Array | cell+SA+drivers | TSMC 28nm | 256x32 | No
| 8T SRAM cell | 8T cell | TSMC 28nm | 1x1 | No
| 8T Column | 8T cell | TSMC 28nm | 128x1, 256x1 | No
| 8T Column | cell+SA+drivers | TSMC 28nm | 256x1 | No
| 8T Array | 8T cell | TSMC 28nm | 128x32 | Yes
| 8T Array | cell+SA+drivers | TSMC 28nm | 128x32 | No
| 6T Macro | cell+SA+drivers | TSMC 0.18um | 16x2, 128x128 | Yes