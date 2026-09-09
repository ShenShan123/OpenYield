"""Read and split Xyce transient and DC waveform tables."""

import io

import numpy as np
import pandas as pd

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
        # Read the file once; drop blank lines and the "End of Xyce(TM) Simulation"
        # trailer explicitly (the old `comment='E'` trick relied on Xyce never printing
        # an upper-case exponent).
        with open(prn_file_path, 'r') as f:
            lines = [l for l in f
                     if l.strip() and not l.lstrip().startswith('End of Xyce')]
        if not lines:
            raise ValueError("Empty PRN file")

        # Clean and validate header
        headers = [h.strip() for h in lines[0].split()]
        if len(headers) < 2:
            raise ValueError("Invalid header - insufficient columns")

        # Read data with enhanced validation
        df = pd.read_csv(
            io.StringIO(''.join(lines[1:])),
            sep=r'\s+',
            header=None,
            names=headers,
            engine='python',
            dtype=np.float64,
            on_bad_lines='warn'
        )

        # Xyce prints a leading Index column unless every .PRINT says FORMAT=NOINDEX;
        # accept both layouts and drop the index when it is there.
        if headers[0].upper() == 'INDEX':
            df = df.drop(columns=headers[0])
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
        boundaries = [0, *reset_indices, len(df)]
        auto_blocks = [df.iloc[start:stop] for start, stop in zip(boundaries, boundaries[1:])]
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

        block_size = total_points // num_mc
        dc_blocks = [df.iloc[i * block_size:(i + 1) * block_size] for i in range(num_mc)]

        # Final block count verification
        if (actual_mc := len(dc_blocks)) != num_mc:
            raise ValueError(
                f"Actual split block count ({actual_mc}) does not match num_mc ({num_mc})\n"
                f"Possible reason: Split anomaly caused by pandas version difference"
            )

        return dc_blocks

    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
