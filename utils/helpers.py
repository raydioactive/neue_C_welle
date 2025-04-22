import pandas as pd
import json
import numpy as np
import re
import os
import warnings # Ensure warnings is imported if you use warnings.warn

def load_data(file_path):
    """
    Loads experimental data from a CSV file.
    # ... (rest of docstring) ...
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path, comment='#')
    except Exception as e:
        if isinstance(e, pd.errors.ParserError):
             error_msg = f"Error parsing CSV file '{file_path}'. Check file format near the line mentioned in the error. Original error: {e}"
        else:
             error_msg = f"Error reading data file '{file_path}': {e}"
        raise Exception(error_msg) from e

    time_col = 'Time'
    time_col_found = None
    for col in df.columns:
        if col.lower() == 'time':
             time_col_found = col
             break
    if time_col_found:
         time_col = time_col_found
         print(f"Note: Found time column as '{time_col}' (case-insensitive match).")
    else:
         raise ValueError(f"Required 'Time' column not found in data file: {file_path}. Found columns: {list(df.columns)}")

    measurement_cols = [col for col in df.columns if col.lower() != time_col.lower() and not col.lower().endswith('_sd')]
    if not measurement_cols:
        raise ValueError(f"No measurement columns found (excluding Time and _SD columns) in data file: {file_path}")

    error_cols = []
    errors_present = True
    df_cols_lower = {c.lower(): c for c in df.columns}

    for m_col in measurement_cols:
         expected_err_col_lower = f"{m_col.lower()}_sd"
         if expected_err_col_lower in df_cols_lower:
              original_err_col_name = df_cols_lower[expected_err_col_lower]
              error_cols.append(original_err_col_name)
         else:
              errors_present = False
              missing_expected = f"{m_col}_SD"
              print(f"Warning: Error column like '{missing_expected}' not found for measurement '{m_col}'. Will estimate errors.")
              break

    # *** MODIFIED LINE: Change 'metabolites' key to 'measured_variables' ***
    data = {
        'time': df[time_col].values.astype(float),
        'measured_variables': measurement_cols, # Use the key expected by DataFitter
        'measurements': df[measurement_cols].values.astype(float),
        'errors': None # Initialize error key
    }
    # **********************************************************************

    if errors_present:
        data['errors'] = df[error_cols].values.astype(float)
        print(f"Loaded measurement errors from columns: {error_cols}")
    else:
        relative_error = 0.05
        data['errors'] = np.abs(data['measurements']) * relative_error
        data['errors'][data['errors'] == 0] = np.mean(data['errors'][data['errors'] > 0]) * 0.1 if np.any(data['errors'] > 0) else relative_error
        print(f"Warning: Some error columns not found. Using estimated relative error: {relative_error*100}%")

    # Validation
    if np.isnan(data['time']).any() or np.isnan(data['measurements']).any() or np.isnan(data['errors']).any() or \
       np.isinf(data['time']).any() or np.isinf(data['measurements']).any() or np.isinf(data['errors']).any():
         raise ValueError(f"Data file '{file_path}' contains non-finite values (NaN or Inf) after loading. Please check the data.")
    if np.any(data['errors'] < 0):
         raise ValueError(f"Data file '{file_path}' contains negative error values after loading. Errors (standard deviations) must be non-negative.")

    print(f"Successfully loaded data for variables: {data['measured_variables']}") # Adjusted print statement
    return data

def load_parameters(file_path):
    """
    Loads initial parameter values and bounds from a JSON file.
    # ... (rest of docstring remains the same) ...
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file not found at: {file_path}")

    try:
        with open(file_path, 'r') as f:
            parameters = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON in parameter file '{file_path}': {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Error reading parameter file '{file_path}': {e}")

    # Validate parameter structure
    validated_parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError(f"Parameter file '{file_path}' should contain a JSON object (dictionary).")

    for param_id, details in parameters.items():
        # *** ADDED CHECK ***
        # Skip keys starting with underscore (intended as comments)
        if param_id.startswith('_'):
            continue
        # *******************

        if not isinstance(details, dict):
            raise ValueError(f"Invalid structure for parameter '{param_id}' in '{file_path}'. Expected a dictionary.")
        if not all(k in details for k in ['initial_value', 'lower_bound', 'upper_bound']):
            raise ValueError(f"Missing required keys ('initial_value', 'lower_bound', 'upper_bound') for parameter '{param_id}' in '{file_path}'.")

        try:
            initial = float(details['initial_value'])
            lower = float(details['lower_bound'])
            upper = float(details['upper_bound'])
        except (ValueError, TypeError) as e:
             raise ValueError(f"Invalid numerical value for parameter '{param_id}' in '{file_path}': {e}")

        if not (lower <= initial <= upper):
             # Allow initial value to be outside bounds, but warn
             warnings.warn(f"Initial value ({initial}) for parameter '{param_id}' is outside its bounds [{lower}, {upper}] in '{file_path}'. Fitting might start clamped.")
             # Clamp initial value to bounds for safety if needed, depends on optimizer behavior
             # initial = np.clip(initial, lower, upper)

        # Removed warning for negative bounds, as some parameters might allow it.
        # if lower < 0 or upper < 0:
        #     print(f"Warning: Parameter '{param_id}' has negative bounds [{lower}, {upper}]. Ensure this is intended.")

        validated_parameters[param_id] = {'initial_value': initial, 'lower_bound': lower, 'upper_bound': upper}

    if not validated_parameters:
         raise ValueError(f"No valid parameters found in '{file_path}'. Check the file structure.")

    print(f"Successfully loaded initial parameters and bounds for: {list(validated_parameters.keys())}")
    return validated_parameters



def calculate_residuals(simulated_data, experimental_data, errors=None):
    """
    Calculates the weighted residuals between simulated and experimental data.

    Residuals can be weighted by the measurement error (standard deviation)
    if provided, which is common practice in parameter fitting (chi-squared).

    Args:
        simulated_data (np.ndarray): The data generated by the model simulation (time x metabolites).
        experimental_data (np.ndarray): The experimental measurements (time x metabolites).
        errors (np.ndarray, optional): The standard deviations of the experimental
                                       measurements (time x metabolites). If None,
                                       unweighted residuals are calculated.

    Returns:
        np.ndarray: A 1D array of residuals (differences between simulated
                    and experimental values, potentially weighted). Returns flattened array.

    Raises:
        ValueError: If the shapes of the input arrays are incompatible.
    """
    if simulated_data.shape != experimental_data.shape:
        raise ValueError(f"Shape mismatch: simulated data ({simulated_data.shape}) vs experimental data ({experimental_data.shape})")

    if errors is not None:
        if errors.shape != experimental_data.shape:
            raise ValueError(f"Shape mismatch: experimental data ({experimental_data.shape}) vs errors ({errors.shape})")
        # Avoid division by zero or very small errors
        valid_errors = np.maximum(errors, 1e-9) # Set a minimum error floor
        residuals = (simulated_data - experimental_data) / valid_errors
    else:
        # Unweighted residuals
        residuals = simulated_data - experimental_data

    return residuals.flatten() # Return a flattened array for optimization routines


def save_results(results, output_dir="results"):
    """
    Saves various results (parameters, statistics) to files.

    Args:
        results (dict): A dictionary containing results to save. Keys might include
                        'fitted_parameters', 'uncertainty_analysis', 'simulation_output'.
        output_dir (str): The directory to save the results files in. Created if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, data in results.items():
        file_path = os.path.join(output_dir, f"{key}.json")
        try:
            # Convert numpy arrays to lists for JSON serialization if necessary
            serializable_data = convert_numpy_to_serializable(data)
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            print(f"Saved {key} results to {file_path}")
        except TypeError as e:
            print(f"Warning: Could not serialize '{key}' data to JSON: {e}. Skipping save for this key.")
        except Exception as e:
            print(f"Error saving {key} results to {file_path}: {e}")


def convert_numpy_to_serializable(obj):
    """
    Recursively converts numpy arrays within a data structure to lists
    to make it JSON serializable.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    return obj

