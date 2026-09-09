"""Shared OpenYield utilities; legacy ``from utils import ...`` imports remain supported."""

from .area import (
    estimate_bitcell_area,
    estimate_total_area,
    estimate_array_area,
    estimate_array_macro_area,
    estimate_total_macro_area,
    estimate_scaled_array_area,
)

from .measurements import (
    parse_mc_measurements,
    generate_mc_statistics,
    save_mc_results,
)

from .spice import (
    parse_spice_models,
    remove_comments,
    parse_parameters,
    convert_value,
    write_spice_models,
)

from .waveforms import (
    read_prn_with_preprocess,
    split_blocks,
)

from .plotting import (
    visualize_results,
    process_simulation_data,
    plot_delay,
    plot_power,
    plot_rc_delay,
    plot_leak_delay,
    plot_merit_history,
    plot_pareto_frontier,
)

__all__ = [
    "estimate_bitcell_area",
    "estimate_total_area",
    "estimate_array_area",
    "estimate_array_macro_area",
    "estimate_total_macro_area",
    "estimate_scaled_array_area",
    "parse_mc_measurements",
    "generate_mc_statistics",
    "save_mc_results",
    "parse_spice_models",
    "remove_comments",
    "parse_parameters",
    "convert_value",
    "write_spice_models",
    "read_prn_with_preprocess",
    "split_blocks",
    "visualize_results",
    "process_simulation_data",
    "plot_delay",
    "plot_power",
    "plot_rc_delay",
    "plot_leak_delay",
    "plot_merit_history",
    "plot_pareto_frontier",
]
