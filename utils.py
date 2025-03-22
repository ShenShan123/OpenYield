import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict
import re
import pandas as pd
from matplotlib.patches import Rectangle

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

def measure_power(vdd, vvdd_i):
    """Calculate average power during sensing"""
    return np.mean(vdd * vvdd_i)

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

    # For read and hold SNMs
    elif operation == 'read_snm' or operation == 'hold_snm':
        snm_l_lobe = snm[snm > 0].max()
        snm_r_lobe = snm[snm < 0].min()
        snm_name = 'RSNM' if operation == 'read_snm' else 'HSNM'
        plt.title(f'{snm_name}: {snm_l_lobe*1000/np.sqrt(2):.2f}mV, {snm_r_lobe*1000/np.sqrt(2):.2f}mV')

    plt.savefig(plot_name)

    return snm

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


# def plot_butterfly_curve(butterfly_curve, filename='butterfly_curve.png'):
#     """
#     Plot the butterfly curve with SNM annotation and the largest inscribed square.
    
#     Args:
#         butterfly_curve (np.ndarray): Array of (V1, V2) points for the butterfly curve.
#         snm (float): Static noise margin (in volts).
#         vdd (float): Supply voltage (in volts).
#         filename (str): Output filename for saving the plot.
#     """
#     # Extract BL and BLB voltages
#     bl_voltages = butterfly_curve[:, 0]
#     blb_voltages = butterfly_curve[:, 1]
    
#     # Create the plot
#     plt.figure(figsize=(8, 8))
    
#     # Plot the butterfly curve
#     plt.plot(bl_voltages, blb_voltages, 'b-', label='Butterfly Curve')
    
#     # Add labels and title
#     plt.xlabel('BL Voltage (V)')
#     plt.ylabel('BLB Voltage (V)')
#     plt.title('SRAM Butterfly Curve')
#     plt.grid(True)
#     plt.legend()
    
#     # Save the plot
#     plt.savefig(filename)
#     plt.close()


def parse_mt0(filename):
    """Parse an HSPICE .mt0 file and return a list of dictionaries for each data entry."""
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