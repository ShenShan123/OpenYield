import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict
import re
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D 


def plot_results(analysis, signals, fig_name='sram_6t.png'):
    """Plot specified signals from analysis results"""
    time_vector = np.array(analysis.time) * 1e9  # Convert to nanoseconds

    plt.figure(figsize=(12, 5))
    for signal in signals:
        signal_vector = np.array(analysis[signal])
        plt.plot(time_vector, signal_vector, label=signal)

    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)

def find_crossing_time(timestamp, signal, threshold, edge_type='rise'):
    """
    Find the time when the signal crosses the threshold.
    edge_type: 'rise' or 'fall'
    """
    if edge_type == 'rise':
        # Find rising edge crossing
        crossing_indices = np.where(np.diff(np.sign(signal - threshold)))[0]
        crossing_indices = crossing_indices[signal[crossing_indices + 1] > signal[crossing_indices]]
    elif edge_type == 'fall':
        # Find falling edge crossing
        crossing_indices = np.where(np.diff(np.sign(signal - threshold)))[0]
        crossing_indices = crossing_indices[signal[crossing_indices + 1] < signal[crossing_indices]]
    else:
        raise ValueError("edge_type must be 'rise' or 'fall'")
    
    if len(crossing_indices) == 0:
        raise ValueError(f"No {edge_type} crossing found for threshold {threshold}.")
    
    # Interpolate to get the exact crossing time
    t1 = timestamp[crossing_indices[0]]
    t2 = timestamp[crossing_indices[0] + 1]
    v1 = signal[crossing_indices[0]]
    v2 = signal[crossing_indices[0] + 1]
    crossing_time = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
    
    return crossing_time

def measure_delay(timestamp, signals, 
                    trig_val=0.5, targ_val=0.5,
                    trig_edge_type='rise', targ_edge_type='fall'):
    
    assert len(signals) == 2, "Two signals are required for delay measurement."
    # Calculate read delay from simulation results
    # Extract time vector
    timestamp = np.array(timestamp)
    
    # Find trigger time
    trig_signal = np.array(signals[0])
    trig_time = find_crossing_time(timestamp, trig_signal, trig_val, edge_type=trig_edge_type)
    
    # Find target time
    targ_signal = np.array(signals[1])
    targ_time = find_crossing_time(timestamp, targ_signal, targ_val, edge_type=targ_edge_type)  
    
    return float(targ_time - trig_time)

def measure_power(vdd, all_branches):
    assert isinstance(all_branches, dict), f"In measure_power, must be a dict class, but got {type(all_branches)}"
    current = np.zeros(all_branches['vvdd'].shape)
    for b in all_branches.values():
        current += b
    """Calculate average power during sensing"""
    return np.mean(vdd * np.abs(current))

from scipy.interpolate import CubicSpline

def calculate_snm(VQ_sweep, VQB_measured, VQB_sweep, VQ_measured, operation, plot_name):
    theta = np.radians(45)  # Convert degrees to radians
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t],
                                [sin_t, cos_t]])
    
    def rotate_curve(curve, rotation_matrix):
        """Rotate a curve using the given rotation matrix."""
        return np.dot(curve, rotation_matrix.T)  # Apply rotation to all points

    # curve where we swept Q and measured QB
    curve_r_inv = np.column_stack((VQ_sweep, VQB_measured))
    # The cubic spline interpolation requires the x-axis to be monotonic
    sort_idx = np.argsort(VQ_measured)

    # Interpolate the curve_l_inv to align with curve_r_inv w.r.t. VQ_sweep (x-axis)
    f_l_inv = CubicSpline(
        VQ_measured[sort_idx], VQB_sweep[sort_idx], extrapolate=False
    )
    VQB_interp = f_l_inv(VQ_sweep)

    # Filter out NaN values
    mask = ~np.isnan(VQB_interp)
    curve_l_inv = np.column_stack((VQ_sweep[mask], VQB_interp[mask]))
    
    # Rotate both curves
    rotated_curve_r_inv = rotate_curve(curve_r_inv, rotation_matrix)
    rotated_curve_l_inv = rotate_curve(curve_l_inv, rotation_matrix)

    plt.figure(figsize=(8, 8))
    plt.plot(rotated_curve_r_inv[:, 0], rotated_curve_r_inv[:, 1], label='Right Inverter (Q sweep)')
    plt.plot(rotated_curve_l_inv[:, 0], rotated_curve_l_inv[:, 1], label='Left Inverter (QB sweep)')
    plt.xlabel('rotated VQ')
    plt.ylabel('rotated VQB')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Interpolate the rotated curves to align w.r.t. x-axis
    f_r_inv = CubicSpline(
        rotated_curve_r_inv[:, 0], rotated_curve_r_inv[:, 1], extrapolate=False
    )
    f_l_inv = CubicSpline(
        rotated_curve_l_inv[:, 0], rotated_curve_l_inv[:, 1], extrapolate=False
    )
    # Set x-axis range to be the common region of the two curves.
    x = np.linspace(start=max(rotated_curve_l_inv[0, 0], rotated_curve_r_inv[0, 0]), 
                    stop=min(rotated_curve_l_inv[-1, 0], rotated_curve_r_inv[-1, 0]), 
                    num=100)
    # Calculate the SNM.
    snm = f_r_inv(x) - f_l_inv(x)

    # For write SNM
    if operation == 'write_snm':
        snm_name = 'WSNM' 
        # Check the WSNM is valide or not
        if (snm < 0).all() or (snm > 0).all():
            plt.title(f'{snm_name}: {np.abs(snm).min()*1000/np.sqrt(2):.2f}mV, {np.abs(snm).max()*1000/np.sqrt(2):.2f}mV')
        else:
            intersection = np.argwhere(np.diff(np.sign(snm)) != 0).reshape(-1) + 0
            inters_str = ', '.join([f'{x[i]:.2f}' for i in intersection])
            plt.scatter(x[intersection], f_r_inv(x[intersection]), color='red', label='Intersection')
            plt.title(f'Bad WSNM with intersection: {inters_str}')
            print(f"[DEBUG] Bad WSNM with intersection: {inters_str}!!")
        ## final write SNM is obtained at x=0 in this version
        # final_snm = (f_r_inv(0) - f_l_inv(0)) / np.sqrt(2)
        final_snm = snm[snm < 0].min() / np.sqrt(2)

    # For read and hold SNMs
    elif operation == 'read_snm' or operation == 'hold_snm':
        snm_l_lobe = snm[snm > 0].max()
        snm_r_lobe = snm[snm < 0].min()
        snm_name = 'RSNM' if operation == 'read_snm' else 'HSNM'
        plt.title(f'{snm_name}: {snm_l_lobe*1000/np.sqrt(2):.2f}mV, {snm_r_lobe*1000/np.sqrt(2):.2f}mV')
        final_snm = min(snm_l_lobe, snm_r_lobe) /np.sqrt(2)

    plt.savefig(plot_name)
    return np.abs(final_snm)

def plot_butterfly_with_squares(VQ_sweep, VQB_measured, VQB_sweep, VQ_measured, 
                                # snm_values, anchor_points, 
                                filename='butterfly_curve.png'):
    """
    Plot butterfly curve with SNM squares
    
    Args:
        VQ_sweep, VQB_measured: First curve data
        VQB_sweep, VQ_measured: Second curve data
        snm_values: Tuple of (SNM_lobe1, SNM_lobe2)
        points: Unity-gain points from calculate_snm
    """
    plt.figure(figsize=(8, 8))
    
    # Plot butterfly curves
    plt.plot(VQ_sweep, VQB_measured, label='Right Inverter (Q sweep)')
    plt.plot(VQ_measured, VQB_sweep, label='Left Inverter (QB sweep)')
    
    # # Draw SNM squares
    # for p, snm in zip(anchor_points, snm_values):
    #     # Calculate square coordinates
    #     side = snm
    #     square = Rectangle(p, side, side,
    #                      linewidth=1, edgecolor='green' if snm == snm_values[0] else 'purple',
    #                      facecolor='none', linestyle='--',
    #                      label=f'SNM {snm*1000:.2f} mV')
    #     plt.gca().add_patch(square)
    
    plt.xlabel('Q Voltage (V)')
    plt.ylabel('QB Voltage (V)')
    plt.title('Butterfly Curve with SNM Squares')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


def parse_mt0(filename):
    """Parse an HSPICE .mt0 or .ms0 file and return a list of dictionaries for each data entry."""
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Find the .TITLE line index
    title_line_index = None
    for i, line in enumerate(lines):
        if line.startswith('.TITLE'):
            title_line_index = i
            break
    if title_line_index is None:
        raise ValueError("No .TITLE line found in the file.")

    # Collect header lines
    header_lines = []
    i = title_line_index + 1
    while i < len(lines):
        line = lines[i]
        if line.startswith('$') or line.startswith('.'):
            i += 1
            continue
        # Check if the line is the start of data (begins with a number)
        if re.match(r'^\s*\d+', line):
            break
        header_lines.append(line)
        i += 1

    # Process header lines to get column names
    column_names = []
    for hl in header_lines:
        parts = hl.split()
        column_names.extend(parts)
    if not column_names:
        raise ValueError("No column headers found.")

    # Number of header lines determines how many lines per data entry
    num_header_lines = len(header_lines)
    if num_header_lines == 0:
        raise ValueError("Header lines count is zero.")

    # Collect all data lines (skip comments, commands, and empty lines)
    data_lines = []
    while i < len(lines):
        line = lines[i]
        if line.startswith('$') or line.startswith('.') or not line.strip():
            i += 1
            continue
        data_lines.append(line)
        i += 1

    # Split data into chunks of num_header_lines each
    chunks = [data_lines[j:j+num_header_lines] for j in range(0, len(data_lines), num_header_lines)]
    parsed_data = []

    for chunk in chunks:
        if len(chunk) != num_header_lines:
            continue  # Skip incomplete chunks

        elements = []
        # Process each line in the chunk
        first_line_parts = chunk[0].split()
        if not first_line_parts:
            continue  # Skip if first line is empty

        elements.append(first_line_parts[0])  # Index value
        elements.extend(first_line_parts[1:])  # Rest of the first line parts

        # Process remaining lines in the chunk
        for line in chunk[1:]:
            elements.extend(line.split())

        # Convert elements to appropriate data types
        row = []
        for col, val in zip(column_names, elements):
            if col == 'index' or col.endswith('#'):
                # Attempt to convert to integer, fallback to float if necessary
                try:
                    converted = int(val)
                except ValueError:
                    converted = int(float(val))  # Handle cases like '27.0'
                row.append(converted)
            else:
                # Convert to float, if possible
                try:
                    converted = float(val)
                except ValueError:
                    converted = val  # Fallback (unlikely in .mt0 files)
                row.append(converted)

        entry = dict(zip(column_names, row))
        parsed_data.append(entry)

    return parsed_data

def analyze_mt0(filename):
    """Parse the .mt0 file and compute statistics for each column."""
    # Parse the file
    data = parse_mt0(filename)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Compute statistics
    stats = {
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max()
    }
    
    # Combine statistics into a single DataFrame
    stats_df = pd.DataFrame(stats)
    print(stats_df)

    return stats_df

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
    print(f"Saved results to {data_file} and {stats_file}")
    print("\nStatistical Summary:")
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

def visualize_results(blocks, analysis_type, output_file):
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
    plt.rcParams['figure.figsize'] = [8.0, 6.0]

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
    signals = base_block.columns[1:]
    
    # Create plot object
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))
    
    # Configure plot parameters
    LINE_ALPHA = 0.5  # Lower transparency to support large-scale data
    LINE_WIDTH = 0.3   # Thin line width for optimized rendering performance
    
    # Process signals in parallel
    for color, signal in zip(colors, signals):
        print(f"Processing signal: {signal}")
        
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
    ax.grid(alpha=1.0)
    
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
    
    print(f"Visualization file generated: {output_path.resolve()}")
    print(f"Plot parameters: line alpha={LINE_ALPHA}, line width={LINE_WIDTH}, total samples={sum(len(b) for b in blocks)}")

def process_simulation_data(prn_path, num_mc=None, output="results"):
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
        visualize_results(data_blocks, analysis_type, output)
        # assert 0
        print(f"Successfully data processed! Saving results to {output}")
        return True
    
    except Exception as e:
        print(f"Failed data processed: {str(e)}")
        raise