"""Plot Xyce waveforms, SRAM comparisons, and optimizer results."""

from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .waveforms import read_prn_with_preprocess, split_blocks

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

_DEFAULT_PLOT_DIR = Path(__file__).resolve().parents[1] / 'outputs/plots'


def _comparison_style(function):
    @wraps(function)
    def styled(*args, **kwargs):
        with plt.style.context('seaborn-v0_8-whitegrid'), plt.rc_context({
            'font.size': 22, 'axes.labelsize': 24, 'figure.figsize': [8, 8],
            'figure.dpi': 350,
        }):
            return function(*args, **kwargs)
    return styled


def _save_comparison_plot(figname, extension, output_dir, show):
    root = _DEFAULT_PLOT_DIR if output_dir is None else Path(output_dir)
    path = root / (str(figname) + extension)
    figure = plt.gcf()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return path


@_comparison_style
def plot_delay(row, r_delay_mean, r_delay_std, w_delay_mean, w_delay_std,
               labelr, labelw, figname, ylim_b=0.05, *, output_dir=None, show=False):
    # set_default()

    # Convert to nanoseconds for better readability
    r_delay_mean_ns = [x * 1e9 for x in r_delay_mean]
    r_delay_std_ns = [x * 3e9 for x in r_delay_std]
    w_delay_mean_ns = [x * 1e9 for x in w_delay_mean]
    w_delay_std_ns = [x * 3e9 for x in w_delay_std]

    # Set up the figure with a specified size
    plt.figure()

    # Create the plot with error bars for read delay
    plt.errorbar(
        row,
        r_delay_mean_ns,
        yerr=r_delay_std_ns,
        fmt='o-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#1f77b4',
        # ecolor='#ff7f0e',
        label=labelr,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row,
        w_delay_mean_ns,
        yerr=w_delay_std_ns,
        fmt='s-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#2ca02c',
        # ecolor='#d62728',
        label=labelw,
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Delay (ns)', fontsize=24)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks()
    plt.yticks()

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout()
    # Option 1: Adjust subplot parameters directly # Increase left margin
    plt.subplots_adjust(left=0.16)

    # Show log scale on y-axis to better display both datasets
    plt.yscale('log')
    plt.ylim(bottom=ylim_b)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot

    return _save_comparison_plot(figname, '.pdf', output_dir, show)


@_comparison_style
def plot_power(row, r_pavg_mean, r_pavg_std, w_pavg_mean, w_pavg_std,
               labelr, labelw, figname, *, output_dir=None, show=False):
    # Convert to microwatts (μW) for better readability
    r_pavg_mean_uw = [x * 1e6 for x in r_pavg_mean]
    r_pavg_std_uw = [x * 3e6 for x in r_pavg_std]
    w_pavg_mean_uw = [x * 1e6 for x in w_pavg_mean]
    w_pavg_std_uw = [x * 3e6 for x in w_pavg_std]

    # Set up the figure
    plt.figure()

    # Create the plot with error bars for read power
    plt.errorbar(
        row,
        r_pavg_mean_uw,
        yerr=r_pavg_std_uw,
        fmt='o-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#1f77b4',
        # ecolor='#ff7f0e',
        label=labelr
    )

    # Add write power data to the same plot
    plt.errorbar(
        row,
        w_pavg_mean_uw,
        yerr=w_pavg_std_uw,
        fmt='s-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#2ca02c',
        # ecolor='#d62728',
        label=labelw
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Average Power (μW)', fontsize=24)
    # plt.title('SRAM Read and Write Power vs Row Size')

    # Add grid
    plt.grid(True, linestyle='--', alpha=1.0)

    # Add legend
    plt.legend(frameon=False)

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Set y-axis to log scale since there's a large range of values
    plt.yscale('log')

    # Improve layout
    plt.tight_layout()

    # Display the plot
    return _save_comparison_plot(figname, '.pdf', output_dir, show)


@_comparison_style
def plot_rc_delay(row, rc_delay_mean, rc_delay_std, orc_delay_mean, orc_delay_std,
                #   c_delay_mean, c_delay_std,
                  labelrc, labelorc, #labelc,
                  figname, *, output_dir=None, show=False):
    # set_default()

    # Convert to nanoseconds for better readability
    rc_delay_mean_ns = [x * 1e9 for x in rc_delay_mean]
    rc_delay_std_ns = [x * 3e9 for x in rc_delay_std]
    orc_delay_mean_ns = [x * 1e9 for x in orc_delay_mean]
    orc_delay_std_ns = [x * 3e9 for x in orc_delay_std]

    # Set up the figure with a specified size
    plt.figure()

    # Create the plot with error bars for read delay
    plt.errorbar(
        row,
        rc_delay_mean_ns,
        yerr=rc_delay_std_ns,
        fmt='o-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        color='#1f77b4',
        ecolor='#ff7f0e',
        label=labelrc,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row,
        orc_delay_mean_ns,
        yerr=orc_delay_std_ns,
        fmt='s-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        color='#2ca02c',
        ecolor='#d62728',
        label=labelorc,
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Delay (ns)', fontsize=24)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks()
    plt.yticks()

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout(pad=1.5)

    # Show log scale on y-axis to better display both datasets
    plt.yscale('log')
    plt.ylim(bottom=6e-3)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot

    return _save_comparison_plot(figname, '.png', output_dir, show)


@_comparison_style
def plot_leak_delay(row, r_delay_mean, r_delay_std, w_delay_mean, w_delay_std,
               labelr, labelw, figname, ylim_b=0.05, ylim_t=7, *, output_dir=None, show=False):
    # set_default()

    # Convert to nanoseconds for better readability
    r_delay_mean_ns = [x * 1e9 for x in r_delay_mean]
    r_delay_std_ns = [x * 1e9 for x in r_delay_std]
    w_delay_mean_ns = [x * 1e9 for x in w_delay_mean]
    w_delay_std_ns = [x * 1e9 for x in w_delay_std]

    # Set up the figure with a specified size
    plt.figure(figsize=(12, 6))

    # Create the plot with error bars for read delay
    plt.errorbar(
        row,
        r_delay_mean_ns,
        yerr=r_delay_std_ns,
        fmt='o-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#1f77b4',
        # ecolor='#ff7f0e',
        label=labelr,
        alpha=0.7,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row,
        w_delay_mean_ns,
        yerr=w_delay_std_ns,
        fmt='s-',
        linewidth=2,
        capsize=6,
        capthick=2,
        markersize=8,
        # color='#2ca02c',
        # ecolor='#d62728',
        label=labelw,
        alpha=0.7,
    )

    # Set labels and title
    plt.xlabel('VDD (V)', fontsize=22)
    plt.ylabel('Delay (ns)', fontsize=22)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False, fontsize=20)

    # Add grid for better readability
    # plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout()
    # Option 1: Adjust subplot parameters directly # Increase left margin
    # plt.subplots_adjust(left=0.16)

    # Show log scale on y-axis to better display both datasets
    # plt.yscale('log')
    plt.ylim(bottom=ylim_b, top=ylim_t)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot

    return _save_comparison_plot(figname, '.pdf', output_dir, show)


def plot_merit_history(merit_history, algorithm_name, filename):
    """
    Plot Merit function history
    绘制Merit函数历史
    """
    plt.figure(figsize=(10, 6))
    plt.plot(merit_history, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Merit")
    plt.title(f"{algorithm_name} Optimization: Merit vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(pareto_front, algorithm_name, filename):
    """
    Plot Pareto frontier
    绘制Pareto前沿
    """
    if len(pareto_front) == 0:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    snm_vals = [p["min_snm"] for p in pareto_front]
    power_vals = [p["max_power"] for p in pareto_front]
    area_vals = [p["area"] for p in pareto_front]

    ax.scatter(snm_vals, power_vals, area_vals, c="red", s=50)
    ax.set_xlabel("Min SNM (V)")
    ax.set_ylabel("Max Power (W)")
    ax.set_zlabel("Area (m²)")
    ax.set_title(f"{algorithm_name} Pareto Frontier")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
