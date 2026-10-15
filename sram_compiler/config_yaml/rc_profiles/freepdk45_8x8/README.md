# FreePDK45 8×8 independent RC reference

`profile.json.gz` is a schema-2 manifest with source-device geometry, terminal
identities, scope and passing physical/waveform receipts. `nets/` contains 709
independent networks. The manifest and `interconnect_example.yaml` map readable
logical net names to files; VDD and VSS are distinct power meshes.

Each net file stores `[element_index, node_a, node_b, value]` resistor and
capacitor records. Resistance is in ohms and capacitance in farads.
`coupling.json.gz` stores explicit cross-net capacitor records with both logical
net names. Global indices preserve the validated element ordering when the
separate networks are assembled. SHA-256 bindings cover compressed bytes.

The YAML has separate `resistance_scale` and `capacitance_scale` for every net.
Changing either marks the reference settings unvalidated. Coupling scales by
the geometric mean of its endpoint capacitance scales. Independent VDD/VSS
resistance scales allow separate supply-drop and ground-rise experiments.

Scope: complete 8×8 6T, no mux, target row/column 7, mode 0, nominal TT,
1 V, 25 °C, read/write at 9 ns, with the recorded controller and device models.
The four-access comparison includes final restore/hold. This reference does
not certify other configurations or a full release screen. The legacy generator,
calibration report and raw campaign outputs have been purged; these runtime
networks and their embedded receipts are retained for the original scope.
The [current layout flow](../../../layout/README.md) requires independent
verification for any new geometry.
