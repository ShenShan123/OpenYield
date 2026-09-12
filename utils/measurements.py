"""Parse Xyce Monte Carlo measurements and summarize their statistics."""

from pathlib import Path

import numpy as np
import pandas as pd

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
        DataFrame with one row per requested run. Missing files and failed or
        invalid measurements remain missing values, never removed samples.
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

            # Xyce writes "FAILED" (with .OPTIONS MEASURE MEASFAIL=1) for a measure whose
            # trigger/target never occurred; keep the column so the caller can see it.
            if raw_value.upper() == 'FAILED':
                print(f"[WARNING] Measurement {var_name} FAILED")
                return var_name, missing_value

            # Numeric conversion
            value = float(raw_value)
            if not np.isfinite(value) or (var_name in ('TSA', 'TS_EN', 'TSWING') and value < 0):
                print(f"[WARNING] Invalid measurement {var_name}: {raw_value}")
                return var_name, missing_value

            if value != 0 and abs(value) < value_threshold:
                # Below the numeric floor the value carries no information, but
                # the sample still exists: report it missing rather than hiding
                # the measurement for this run.
                print(f"[WARNING] Measurement {var_name} below {value_threshold:g}: {raw_value}")
                return var_name, missing_value

            return var_name, value
        except (ValueError, IndexError) as e:
            print(f"Ignoring invalid line: {line[:50]}... | Error: {str(e)}")
            return None, None

    for run_id in range(num_runs):
        file_path = Path(f"{netlist_prefix}.{file_suffix}{run_id}")
        if not file_path.exists():
            print(f"Warning: Missing file {file_path}")
            raw_data.append({"Run": run_id})
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

    return pd.DataFrame(clean_data, columns=['Run', *all_vars]).set_index('Run')

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
    if df.columns.empty or df.isna().all().all():
        # Every requested run is kept as a row now, so an all-missing frame is a
        # simulation failure, not an empty one.  Describing it would publish a
        # statistics file made entirely of NaN.
        raise ValueError(
            f"No measurement of the {len(df)} requested runs was parsed; the simulation "
            f"produced no usable .mt/.ms output (check the Xyce log for a failed "
            f"operating point or netlist error)")

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
