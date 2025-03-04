import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict
import re
import pandas as pd

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
    plt.grid()
    plt.savefig(fig_name)

def find_crossing_time(time_vector, signal_vector, threshold, edge_type='rise'):
    """
    Find the time when the signal crosses the threshold.
    edge_type: 'rise' or 'fall'
    """
    if edge_type == 'rise':
        # Find rising edge crossing
        crossing_indices = np.where(np.diff(np.sign(signal_vector - threshold)))[0]
        crossing_indices = crossing_indices[signal_vector[crossing_indices + 1] > signal_vector[crossing_indices]]
    elif edge_type == 'fall':
        # Find falling edge crossing
        crossing_indices = np.where(np.diff(np.sign(signal_vector - threshold)))[0]
        crossing_indices = crossing_indices[signal_vector[crossing_indices + 1] < signal_vector[crossing_indices]]
    else:
        raise ValueError("edge_type must be 'rise' or 'fall'")
    
    if len(crossing_indices) == 0:
        raise ValueError(f"No {edge_type} crossing found for threshold {threshold}.")
    
    # Interpolate to get the exact crossing time
    t1 = time_vector[crossing_indices[0]]
    t2 = time_vector[crossing_indices[0] + 1]
    v1 = signal_vector[crossing_indices[0]]
    v2 = signal_vector[crossing_indices[0] + 1]
    crossing_time = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
    
    return crossing_time

def measure_delay(time, signals, 
                    trig_val=0.5, targ_val=0.5,
                    trig_edge_type='rise', targ_edge_type='fall'):
    # Calculate read delay from simulation results
    # Extract time vector
    time_vector = np.array(time)
    
    # Find WL activation time (50% point)
    wl_signal = np.array(signals[0])
    time_wl = find_crossing_time(time_vector, wl_signal, trig_val, edge_type=trig_edge_type)
    
    # Find BL swing time (50% point)
    bl_signal = np.array(signals[1])
    time_bl = find_crossing_time(time_vector, bl_signal, targ_val, edge_type=targ_edge_type)  
    
    return float(time_bl - time_wl)

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