import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import re
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D 
from typing import Dict, Any, Union
from matplotlib.ticker import MultipleLocator

def parse_mc_measurements(netlist_prefix: str = "simulation",
                         file_suffix: str = 'mt',
                         num_runs: int = 100,
                         missing_value: float = np.nan,
                         value_threshold: float = 1e-30) -> pd.DataFrame:
    """
    Parse Monte Carlo simulation results from multiple output files
    
    Args:
        netlist_prefix: Base name for simulation files
        file_suffix: Suffix pattern for MC result files
        num_runs: Number of Monte Carlo runs
        missing_value: Value to fill for missing measurements
        value_threshold: Minimum absolute value to consider valid
    
    Returns:
        DataFrame containing parsed results with runs as rows
    """
    measurement_cache = {}
    raw_data = []

    def parse_line(line: str) -> tuple:
        """Parse single measurement line with validation"""
        line = line.strip().replace('\t', ' ')
        if not line or '=' not in line:
            return None, None
        if line.startswith(('*', '#', '//')):
            return None, None
        
        try:
            var_part, value_part = line.split('=', 1)
            var_name = var_part.strip()
            raw_value = value_part.split()[0].strip()
            
            # Numeric conversion
            value = float(raw_value) if '.' in raw_value or 'e' in raw_value.lower() else int(raw_value)
            
            if abs(value) < value_threshold:
                return None, None
                
            return var_name, value
        except (ValueError, IndexError) as e:
            print(f"Ignoring invalid line: {line[:50]}... | Error: {str(e)}")
            return None, None

    for run_id in range(num_runs):
        file_path = Path(f"{netlist_prefix}.{file_suffix}{run_id}")
        if not file_path.exists():
            print(f"Warning: Missing file {file_path}")
            continue

        run_data = {"Run": run_id}
        with open(file_path, 'r') as f:
            for line in f:
                var_name, value = parse_line(line)
                if var_name and value is not None:
                    run_data[var_name] = value
                    measurement_cache[var_name] = True
        
        if num_runs>1:
            # 只跑一次的话正常显示，跑多次蒙卡时要检查TSA、TS_EN、TSWING是否为负数，如果是则跳过此次结果，否则会影响结果
            skip_run = False
            for param in ['TSA', 'TS_EN', 'TSWING']:
                if param in run_data and run_data[param] < 0:
                    print(f"Skipping run {run_id} due to negative {param} value: {run_data[param]}")
                    skip_run = True
                    break
            if not skip_run:
                raw_data.append(run_data)
        else:
            raw_data.append(run_data)

    # Build complete dataframe
    all_vars = sorted(measurement_cache.keys())
    clean_data = []
    for entry in raw_data:
        full_entry = {var: entry.get(var, missing_value) for var in all_vars}
        full_entry["Run"] = entry["Run"]
        clean_data.append(full_entry)
    
    return pd.DataFrame(clean_data).set_index('Run')

def generate_mc_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive statistics from MC results
    
    Args:
        df: DataFrame from parse_mc_measurements()
    
    Returns:
        Transposed DataFrame with statistical metrics
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    stats = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # Additional statistical metrics
    stats.loc['cv'] = stats.loc['std'] / stats.loc['mean']  # Coefficient of variation
    stats.loc['range'] = stats.loc['max'] - stats.loc['min']
    stats.loc['skew'] = df.skew()
    stats.loc['kurtosis'] = df.kurtosis()
    
    return stats.T

def save_mc_results(df: pd.DataFrame,
                   stats_df: pd.DataFrame,
                   data_file: str = "mc_results.csv",
                   stats_file: str = "mc_statistics.csv") -> None:
    """
    Save MC results and statistics to CSV files
    
    Args:
        df: Main results DataFrame
        stats_df: Statistics DataFrame
        data_file: Filename for measurement data
        stats_file: Filename for statistics
    """
    df.to_csv(data_file)
    stats_df.to_csv(stats_file)
    print(f"[DEBUG] Saved results to {data_file} and {stats_file}")
    print("\n[DEBUG] Statistical Summary:")
    print(stats_df)


def read_prn_with_preprocess(prn_file_path):
    """Read and preprocess PRN files with Index column
    
    Args:
        prn_file_path (str): Path to PRN file
        
    Returns:
        tuple: (pd.DataFrame, analysis_type)
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: For format errors
    """
    try:
        # Read and preprocess header
        with open(prn_file_path, 'r') as f:
            # Find first non-empty line for column headers
            header_line = ''
            while not header_line.strip():
                header_line = f.readline()
                if not header_line:  # Handle empty files
                    raise ValueError("Empty PRN file")

            # Clean and validate header
            headers = [h.strip() for h in header_line.split()]
            if len(headers) < 2:
                raise ValueError("Invalid header - insufficient columns")
                
            if headers[0].upper() != 'INDEX':
                raise ValueError(f"First column must be 'INDEX', got '{headers[0]}'")

        # Read data with enhanced validation
        df = pd.read_csv(
            prn_file_path,
            sep='\s+',
            skiprows=1,
            header=None,
            names=headers,
            engine='python',
            dtype=np.float64,
            comment='E',
            on_bad_lines='warn'
        )

        # drop the default index
        df = df.set_index(headers[0])
        df = df.reset_index(drop=True)

        # Determine analysis type from first data column
        first_data_col = df.columns[0].upper()
        analysis_type = "tran" if first_data_col == "TIME" else "dc"

        if df.columns[0].upper() != 'TIME' and df.columns[0].upper() != '{U}':
            raise ValueError(f"Wrong x-axis in PRN file: {df.columns[0]}")

        # Data integrity checks
        if df.empty:
            raise ValueError("No valid data rows found")
            
        if df.select_dtypes(exclude=np.number).any().any():
            raise ValueError("Non-numeric data detected")

        return df, analysis_type

    except FileNotFoundError as e:
        raise FileNotFoundError(f"PRN file not found: {prn_file_path}") from e
        
    except pd.errors.ParserError as e:
        raise ValueError(f"Data parsing error: {str(e)}") from e

def split_blocks(df, analysis_type, num_mc):
    """
    Enhanced data splitting function, including strict num_mc validation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing simulation data to be split into blocks
    analysis_type : str
        Type of analysis, must be either "tran" (transient) or "dc"
    num_mc : int
        Expected number of Monte Carlo iterations/blocks
        
    Returns:
    --------
    list of pandas.DataFrame
        A list containing the split dataframes, with length equal to num_mc
        
    Raises:
    -------
    ValueError
        If parameters are invalid or if the actual block count doesn't match num_mc
    """
    # Basic parameter validation
    if not isinstance(num_mc, int) or num_mc <= 0:
        raise ValueError("num_mc must be a positive integer")
    
    if analysis_type == "tran":
        # Time series splitting logic
        time_series = df.iloc[:, 0]
        reset_indices = np.where(np.diff(time_series) < -1e-12)[0] + 1
        
        # Automatically split blocks count
        auto_blocks = np.split(df, reset_indices) if reset_indices.size else [df]
        auto_block_count = len(auto_blocks)
        
        # Block count consistency verification
        if auto_block_count != num_mc:
            raise ValueError(
                f"Auto-split block count ({auto_block_count}) does not match specified num_mc ({num_mc})\n"
                f"Possible reasons: 1. Incorrect simulation count setting 2. Incomplete data 3. Time series anomaly"
            )
        
        # Secondary validation of start times
        for i, blk in enumerate(auto_blocks):
            if abs(blk.iloc[0, 0]) > 1e-12:
                raise ValueError(f"TRAN block {i} start time anomaly: {blk.iloc[0,0]:.2e}s")
        
        return auto_blocks
    
    elif analysis_type == "dc":
        # DC analysis splitting logic
        total_points = len(df)
        if total_points % num_mc != 0:
            raise ValueError(
                f"Data points ({total_points}) cannot be evenly divided by num_mc ({num_mc})\n"
                f"Suggestions: 1. Check simulation settings 2. Verify output options"
            )
        
        dc_blocks = np.array_split(df, num_mc)
        
        # Final block count verification
        if (actual_mc := len(dc_blocks)) != num_mc:
            raise ValueError(
                f"Actual split block count ({actual_mc}) does not match num_mc ({num_mc})\n"
                f"Possible reason: Split anomaly caused by pandas version difference"
            )
        
        return dc_blocks
    
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")

def visualize_results(blocks, analysis_type, output_file, selected_columns=None, y_min=None, y_max=None):
    """
    Visualization function for large-scale Monte Carlo simulations with variable time steps
    
    Parameters:
    -----------
    blocks : list of pandas.DataFrame
        List of data blocks from split_blocks function, each containing simulation data
    analysis_type : str
        Type of analysis, typically "tran" (transient) or "dc"
    output_file : str or Path
        Path to save the output visualization file
        
    Returns:
    --------
    None
        The function saves the visualization to the specified output file
        
    Raises:
    -------
    ValueError
        If the input data blocks are empty
    """
    # Change default font family
    plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace'

    # Set specific font (if installed on your system)
    # plt.rcParams['font.serif'] = ['Times New Roman']  # Or 'Palatino', 'Computer Modern Roman', etc.

    # Change font sizes
    # plt.rcParams['font.size'] = 12          # Base font size
    # plt.rcParams['axes.titlesize'] = 14     # Title font size
    # plt.rcParams['axes.labelsize'] = 14     # Axis label size
    # plt.rcParams['xtick.labelsize'] = 12    # X-tick label size
    # plt.rcParams['ytick.labelsize'] = 12    # Y-tick label size
    # plt.rcParams['legend.fontsize'] = 12    # Legend font size
    # # Set default figure size (width, height) in inches
    plt.rcParams['figure.figsize'] = [12.0, 6.0]

    # plt.rcParams.update({
    #     'font.family': 'serif',           # 设置字体族
    #     'font.sans-serif': 'Century',
    #     'font.size': 20,                  # 基础字体大小
    #     'axes.labelsize': 20,             # 轴标签字体大小
    #     'axes.titlesize': 20,             # 标题字体大小
    #     'xtick.labelsize': 20,            # x轴刻度标签大小
    #     'ytick.labelsize': 20,            # y轴刻度标签大小
    #     'legend.fontsize': 20,            # 图例字体大小
    #     'figure.figsize': [8, 8],         # 图形大小
    #     'figure.dpi': 350,                # 分辨率
    # })

    # Or use built-in style sheets
    plt.style.use('ggplot')  # Options: 'seaborn', 'fivethirtyeight', 'dark_background', etc.

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Basic data validation
    if not blocks:
        raise ValueError("Input data blocks are empty, please check splitting results")
    
    # Get signal list
    base_block = blocks[0]
    x_label = base_block.columns[0]
    #signals = base_block.columns[1:]
    # Filter signals based on selected_columns
    all_signals = base_block.columns[1:]
    if selected_columns is not None:
        # Only plot signals that exist in the data and are requested
        signals = [s for s in selected_columns if s in all_signals]
        if not signals:
            print(f"Warning: None of the selected columns {selected_columns} found in data. Plotting all columns instead.")
            signals = all_signals
    else:
        signals = all_signals

    # Create plot object
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))
    
    # Configure plot parameters
    LINE_ALPHA = 0.9  # Lower transparency to support large-scale data
    LINE_WIDTH = 0.9   # Thin line width for optimized rendering performance
    
    # Process signals in parallel
    for color, signal in zip(colors, signals):
        print(f"[DEBUG] Processing signal: {signal}")
        
        # Collect valid samples
        valid_samples = []
        for blk_idx, blk in enumerate(blocks):
            try:
                # Get raw data directly
                x = blk[x_label].to_numpy()
                y = blk[signal].to_numpy()
                
                # Strict dimension validation
                if len(x) != len(y):
                    print(f"Block {blk_idx} dimension mismatch, skipped: x({len(x)}) vs y({len(y)})")
                    continue
                
                # Plot raw trajectory
                ax.plot(x, y,
                       color=color,
                       alpha=LINE_ALPHA,
                       linewidth=LINE_WIDTH,
                       zorder=1)
                
                valid_samples.append((x, y))
                
            except Exception as e:
                print(f"Failed to process block {blk_idx}: {str(e)}")
                continue
        
        if not valid_samples:
            print(f"No valid data for signal {signal}")
            continue
        
        # Optional: Add representative statistical trajectories
        if len(valid_samples) > 10:
            # Randomly sample some trajectories for highlighting
            for x, y in valid_samples[:10]:
                ax.plot(x, y,
                       color=color,
                       alpha=0.3,
                       linewidth=1,
                       zorder=2)
    
    # Graph decoration
    # ax.set_title(f"{analysis_type.upper()} Monte Carlo Analysis (Variable Steps)", pad=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Voltage (V)")

    # Set y-axis major tick interval to 0.05
    # try:
    #     ax.yaxis.set_major_locator(MultipleLocator(0.05))
    # except Exception:
    #     # Fallback: explicitly set yticks based on provided or current limits
    #     ymin_cur, ymax_cur = ax.get_ylim()
    #     ymin_use = y_min if y_min is not None else ymin_cur
    #     ymax_use = y_max if y_max is not None else ymax_cur
    #     yticks = np.arange(np.floor(ymin_use / 0.05) * 0.05, np.ceil(ymax_use / 0.05) * 0.05 + 1e-9, 0.05)
    #     ax.set_yticks(yticks)

    # # Apply y-axis limits if provided
    # if y_min is not None or y_max is not None:
    #     ymin_cur, ymax_cur = ax.get_ylim()
    #     ymin_set = y_min if y_min is not None else ymin_cur
    #     ymax_set = y_max if y_max is not None else ymax_cur
    #     ax.set_ylim(ymin_set, ymax_set)

    ax.grid(alpha=1.0)
    #ax.set_aspect('equal', adjustable='box')
    # Create legend proxies
    legend_elements = [Line2D([0], [0], color=c, lw=2, label=s) 
                      for s, c in zip(signals, colors)]
    ax.legend(handles=legend_elements,
             loc='upper center',
             bbox_to_anchor=(0.5, -0.15),
             ncol=2,
             frameon=False)
    
    # Optimize output
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')  # Reduce dpi to optimize file size
    plt.close()
    
    print(f"[DEBUG] Waveform file generated: {output_path.resolve()}")
    print(f"[DEBUG] Plot parameters: line alpha={LINE_ALPHA}, line width={LINE_WIDTH}, total samples={sum(len(b) for b in blocks)}")

def process_simulation_data(prn_path, num_mc=None, output="results", selected_columns=None):
    """
    Main processing function for simulation data
    
    Parameters:
    -----------
    prn_path : str or Path
        Path to the .prn simulation output file
    num_mc : int, optional
        Number of Monte Carlo iterations to expect in the data
    output : str, default="results"
        Path to save output visualization files
        
    Returns:
    --------
    bool
        True if processing completed successfully
        
    Raises:
    -------
    Exception
        If any error occurs during data processing
    """
    try:
        # Data loading
        df, analysis_type = read_prn_with_preprocess(prn_path)

        # Data splitting
        data_blocks = split_blocks(df, analysis_type, num_mc)
        # print("data_blocks", data_blocks)
        # assert 0
        # Results visualization
        visualize_results(data_blocks, analysis_type, output, selected_columns, y_min=None, y_max=None)
        # assert 0
        print(f"[DEBUG] Successfully data processed!")
        return True
    
    except Exception as e:
        print(f"Failed data processed: {str(e)}")
        raise


def _normalize_sram_cell_type(cell_type: str) -> str:
    raw = str(cell_type).strip().upper()
    if raw in {"6T", "SRAM_6T", "SRAM_6T_CELL"}:
        return "SRAM_6T_CELL"
    if raw in {"10T", "SRAM_10T", "SRAM_10T_CELL"}:
        return "SRAM_10T_CELL"
    raise ValueError(f"Unsupported cell_type: {cell_type}")


def estimate_bitcell_area(
    # Transistor dimensions (m)
    w_access: float,       # Access NMOS / PG width
    w_pd: float,           # Pull-down NMOS width
    w_pu: float,           # Pull-up PMOS width
    l_transistor: float,   # Transistor length (kept for interface consistency)

    # FreePDK45 Design Rules (default values in meters)
    CPP: float = 0.18e-6,          # Contacted Poly Pitch
    MMP: float = 0.16e-6,          # Minimum Metal Pitch
    diffusion_spacing: float = 0.08e-6,
    gate_contact_spacing: float = 0.05e-6,
    metal_overhang: float = 0.03e-6,

    *,
    w_fd=None,                     # 10T only: feed-forward NMOS width
    cell_type: str = "SRAM_6T_CELL",
) -> float:
    """
    Estimate SRAM bitcell area for 6T / 10T using the same coarse geometry idea.

    6T:
        Keep the existing estimation logic unchanged.

    10T:
        Reuse the 6T methodology, but treat the 10T cell as:
        - the original 6T storage core
        - plus one extra feedback / stacked pull-down branch

        So compared with 6T, the width model gains one extra branch term.
        Height is kept the same as the 6T estimator to preserve the original style.
    """
    cell_type = _normalize_sram_cell_type(cell_type)

    w_access = float(w_access)
    w_pd = float(w_pd)
    w_pu = float(w_pu)
    _ = float(l_transistor)  # currently not used in the geometric area model

    # Shared terms
    access_width = w_access + 2 * gate_contact_spacing
    pd_pu_width = max(w_pd, w_pu) + 2 * diffusion_spacing

    if cell_type == "SRAM_6T_CELL":
        # Keep old 6T logic exactly as before
        width_columns = 3 * CPP  # 2 access + 1 inverter pair
        width_scaling = (access_width + pd_pu_width) / (0.135e-6 + 0.205e-6)
        cell_width = width_columns * width_scaling + 2 * metal_overhang
        cell_height = 2 * MMP + 2 * diffusion_spacing + 4 * metal_overhang
        return cell_width * cell_height

    # 10T branch
    if w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    w_fd = float(w_fd)

    # 10T approximation:
    # - original access branch
    # - original storage-core branch
    # - one extra FD / stacked-PD helper branch
    fd_branch_width = max(w_pd, w_fd) + 2 * diffusion_spacing

    # Compared with 6T's 3-column abstraction, 10T is modeled as one extra branch.
    width_columns = 4 * CPP

    # Reference widths for current 10T default sizing style:
    # pg=135nm, core(max(pd, pu))=90nm, helper(max(pd, fd))=135nm
    width_scaling = (
        access_width + pd_pu_width + fd_branch_width
    ) / (0.135e-6 + 0.09e-6 + 0.135e-6)

    cell_width = width_columns * width_scaling + 2 * metal_overhang

    # Keep the same height model as 6T to preserve the original estimation philosophy.
    cell_height = 2 * MMP + 2 * diffusion_spacing + 4 * metal_overhang

    return cell_width * cell_height

def estimate_total_area(num_rows, num_cols, num_arrays):
    """
    估算多阵列 SRAM 总面积 (µm²)。
    使用 OpenRAM FreePDK45nm 单阵列面积回归公式乘以阵列数。
    """
    return estimate_array_area(num_cols, num_rows) * num_arrays


def estimate_array_area(word_size, num_words):
    width = 31.3547 + word_size * 2.9918 + num_words * 0.0568
    height = 49.9556 + word_size * 1.0801 + num_words * 0.0830
    return width * height

def estimate_array_macro_area(
    num_rows,
    num_cols,
    current_bitcell_area,
    default_bitcell_area,
):
    num_cells = num_rows * num_cols
    base_array_area = estimate_array_area(num_cols, num_rows)
    base_cell_area = default_bitcell_area * num_cells
    periphery_area = base_array_area - base_cell_area

    if periphery_area < 0:
        raise ValueError(
            f"Computed periphery_area is negative: {periphery_area}. "
            "Area models are inconsistent for this architecture."
        )

    return periphery_area + current_bitcell_area * num_cells


def estimate_total_macro_area(
    num_rows,
    num_cols,
    num_arrays,
    current_bitcell_area,
    default_bitcell_area,
):
    array_area = estimate_array_macro_area(
        num_rows=num_rows,
        num_cols=num_cols,
        current_bitcell_area=current_bitcell_area,
        default_bitcell_area=default_bitcell_area,
    )
    return array_area * num_arrays

_DR_CPP = 0.18e-6
_DR_MMP = 0.16e-6
_DR_DIFF_SP = 0.08e-6
_DR_GATE_CT = 0.05e-6
_DR_CONTACT = 0.065e-6
_DR_METAL_OH = 0.03e-6

_REF_PG = 0.135e-6
_REF_PD = 0.205e-6
_REF_PU = 0.090e-6
_REF_FD = 0.135e-6
_REF_L = 50e-9


def _cell_width_metric(
    w_access: float,
    w_pd: float,
    w_pu: float,
    l_transistor: float,
    *,
    cell_type: str = "SRAM_6T_CELL",
    w_fd=None,
) -> float:
    cell_type = _normalize_sram_cell_type(cell_type)

    w_access = float(w_access)
    w_pd = float(w_pd)
    w_pu = float(w_pu)
    l_transistor = float(l_transistor)

    active_pg = w_access + 2 * _DR_GATE_CT
    active_inv = w_pd + w_pu + 2 * _DR_DIFF_SP
    cpp = l_transistor + 2 * _DR_GATE_CT

    if cell_type == "SRAM_6T_CELL":
        return active_pg + active_inv + 3 * cpp

    if w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    w_fd = float(w_fd)
    active_helper = max(w_pd, w_fd) + 2 * _DR_DIFF_SP

    return active_pg + active_inv + active_helper + 3 * cpp


def _cell_height_metric(
    l_transistor: float,
    *,
    cell_type: str = "SRAM_6T_CELL",
) -> float:
    cell_type = _normalize_sram_cell_type(cell_type)
    l_transistor = float(l_transistor)

    gate_pitch = l_transistor + _DR_GATE_CT + _DR_CONTACT

    if cell_type == "SRAM_6T_CELL":
        return 3 * gate_pitch + 2 * _DR_MMP + 2 * _DR_METAL_OH

    return 4 * gate_pitch + 3 * _DR_MMP + 2 * _DR_METAL_OH


_REF_CW_6T = _cell_width_metric(
    _REF_PG,
    _REF_PD,
    _REF_PU,
    _REF_L,
    cell_type="SRAM_6T_CELL",
)
_REF_CH_6T = _cell_height_metric(
    _REF_L,
    cell_type="SRAM_6T_CELL",
)

_REF_CW_10T = _cell_width_metric(
    _REF_PG,
    _REF_PD,
    _REF_PU,
    _REF_L,
    cell_type="SRAM_10T_CELL",
    w_fd=_REF_FD,
)
_REF_CH_10T = _cell_height_metric(
    _REF_L,
    cell_type="SRAM_10T_CELL",
)

_BASE_COL_PITCH_6T = 1.20
_BASE_ROW_PITCH_6T = 0.90

# 用默认 10T 几何尺寸相对默认 6T 的比例，生成 10T 的基准 pitch
_BASE_COL_PITCH_10T = _BASE_COL_PITCH_6T * (_REF_CW_10T / _REF_CW_6T)
_BASE_ROW_PITCH_10T = _BASE_ROW_PITCH_6T * (_REF_CH_10T / _REF_CH_6T)


def estimate_scaled_array_area(
    num_rows: int,
    num_cols: int,
    num_arrays: int,
    w_access: float,
    w_pd: float,
    w_pu: float,
    l_transistor: float,
    sa_max_width: float = 0.54e-6,
    wld_max_width: float = 0.27e-6,
    prc_max_width: float = 0.27e-6,
    *,
    cell_type: str = "SRAM_6T_CELL",
    w_fd=None,
) -> float:
    """
    Estimate total SRAM area in m^2 using the pitch-scaling model.
    Supports both 6T and 10T.
    """
    import math

    cell_type = _normalize_sram_cell_type(cell_type)

    if cell_type == "SRAM_10T_CELL" and w_fd is None:
        raise ValueError("w_fd must be provided when cell_type='SRAM_10T_CELL'")

    if cell_type == "SRAM_10T_CELL":
        ref_cw = _REF_CW_10T
        ref_ch = _REF_CH_10T
        base_col_pitch = _BASE_COL_PITCH_10T
        base_row_pitch = _BASE_ROW_PITCH_10T
    else:
        ref_cw = _REF_CW_6T
        ref_ch = _REF_CH_6T
        base_col_pitch = _BASE_COL_PITCH_6T
        base_row_pitch = _BASE_ROW_PITCH_6T

    w_ratio = _cell_width_metric(
        w_access,
        w_pd,
        w_pu,
        l_transistor,
        cell_type=cell_type,
        w_fd=w_fd,
    ) / ref_cw

    h_ratio = _cell_height_metric(
        l_transistor,
        cell_type=cell_type,
    ) / ref_ch

    col_pitch = base_col_pitch * w_ratio
    row_pitch = base_row_pitch * h_ratio

    ref_wld_w = 0.27e-6
    row_log_factor = 0.60 + 0.40 * math.log2(max(num_rows, 2)) / math.log2(32)
    wld_r = max(float(wld_max_width), 1e-9) / ref_wld_w
    overhead_x = (9.59 + 4.11 * wld_r) * row_log_factor

    ref_sa_w = 0.54e-6
    ref_prc_w = 0.27e-6
    sa_r = max(float(sa_max_width), 1e-9) / ref_sa_w
    prc_r = max(float(prc_max_width), 1e-9) / ref_prc_w
    overhead_y = 5.77 + 5.13 * sa_r + 1.92 * prc_r

    width_um = overhead_x + num_cols * col_pitch
    height_um = overhead_y + num_rows * row_pitch

    return width_um * height_um * num_arrays * 1e-12


def parse_spice_models(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Parse SPICE transistor model library file into Python dictionary."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    models = {}
    
    # Split content by .model statements
    model_sections = re.split(r'\.model\s+', content, flags=re.IGNORECASE)
    
    # Skip first empty section
    for section in model_sections[1:]:
        if section.strip():
            # Split first line to get model name and type
            lines = section.strip().split('\n', 1)
            first_line = lines[0].strip()
            
            # Extract model name and type from first line
            parts = first_line.split()
            if len(parts) >= 2:
                model_name = parts[0]
                model_type = parts[1]
                
                # Get parameter text (rest of first line + remaining lines)
                param_text = ' '.join(parts[2:])
                if len(lines) > 1:
                    param_text += ' ' + lines[1]
                
                # Remove comments and clean up the parameter text
                cleaned_text = remove_comments(param_text)
                
                # Parse parameters
                parameters = parse_parameters(cleaned_text)
                
                models[model_name] = {
                    'name': model_name,
                    'type': model_type,
                    'parameters': parameters
                }
    
    return models

def remove_comments(text: str) -> str:
    """Remove comments (lines starting with * or inline comments)."""
    lines = []
    for line in text.split('\n'):
        # Remove inline comments
        if '*' in line:
            line = line[:line.index('*')]
        line = line.strip()
        if line:
            lines.append(line)
    return ' '.join(lines)

def parse_parameters(text: str) -> Dict[str, Union[float, int, str]]:
    """Parse parameter=value pairs from text."""
    parameters = {}
    
    # Remove continuation characters (+) at the beginning of lines only
    # This preserves + signs in scientific notation like 3.4e+18
    text = re.sub(r'^\+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\n\+', '\n ', text)
    
    # Find all param=value pairs
    for match in re.finditer(r'(\w+)\s*=\s*([^\s]+)', text):
        param_name = match.group(1)
        param_value = convert_value(match.group(2))
        parameters[param_name] = param_value
    
    return parameters

def convert_value(value_str: str) -> Union[float, int, str]:
    """Convert string value to appropriate Python type."""
    try:
        # Try integer first (if no decimal point or scientific notation)
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        return value_str
    
def write_spice_models(models: Dict[str, Dict[str, Any]], filepath: str):
    """Write models dictionary back to a SPICE model file."""
    with open(filepath, 'w') as f:
        for model_name, model_data in models.items():
            # Write model header
            f.write(f".model  {model_data['name']}  {model_data['type']}")
            
            # Write parameters
            params = model_data['parameters']
            param_count = 0
            
            for param_name, param_value in params.items():
                # Start new line every 4 parameters or at the beginning
                if param_count % 4 == 0:
                    f.write('\n+')
                
                # Format parameter value
                if isinstance(param_value, float):
                    # Use scientific notation for very small/large numbers
                    if abs(param_value) < 1e-3 or abs(param_value) > 1e6:
                        param_str = f"{param_value:.3e}"
                    else:
                        param_str = str(param_value)
                else:
                    param_str = str(param_value)
                
                # Write parameter with proper spacing
                f.write(f"{param_name:>12} = {param_str:<26}")
                param_count += 1
            
            # Add blank lines between models
            f.write('\n\n')
        
        f.write('\n')
    
# if __name__ == "__main__":
# # Parse the models from file
#     models = parse_spice_models('model_lib/models.spice')
    
#     # Print results
#     for name, model in models.items():
#         print(f"Model: {name}")
#         print(f"Type: {model['type']}")
#         print(f"Parameters: {len(model['parameters'])}")
        
#         # Show first few parameters
#         for i, (param, value) in enumerate(model['parameters'].items()):
#             if i < 5:
#                 print(f"  {param}: {value}")
#         print()
    
#     # Access specific parameter
#     if 'NMOS_VTG' in models:
#         vth0 = models['NMOS_VTG']['parameters']['vth0']
#         print(f"NMOS_VTG vth0 parameter: {vth0}")
