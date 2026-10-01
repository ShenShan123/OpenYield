"""Parse and write SPICE transistor model libraries."""

import re
from pathlib import Path
from typing import Any, Dict, Union

from sram_compiler.per_device_mc.netlist import parse_spice_models as _parse_active_models


def parse_spice_models(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Parse SPICE transistor model library file into Python dictionary."""
    # Generated model cards must carry the values Xyce reads: only active
    # .model statements, with comments and later directives excluded.
    return _parse_active_models(Path(filepath))

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
