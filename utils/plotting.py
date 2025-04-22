# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# Constants for plotting style
COLORS = plt.cm.get_cmap('tab10').colors # Use a standard colormap
LINESTYLES = ['-', '--', ':', '-.']
MARKERS = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']

def plot_results(data, simulation_function, fitted_parameters,
                 uncertainty_results=None, output_dir="results", filename="fit_results.png"):
    """
    Generates plots comparing experimental data with model simulations.

    Args:
        data (dict): Dictionary containing experimental data.
                     Expected keys: 'time', 'measured_variables', 'measurements', 'errors'.
        simulation_function (callable): A function that takes a parameter vector and
                                        returns the simulated model output at the data time points,
                                        matching the structure of data['measurements'].
        fitted_parameters (np.ndarray): The vector of fitted parameter values.
        uncertainty_results (dict, optional): Summary statistics from Monte Carlo
                                               (e.g., {'param_id': {'ci_95_lower': L, 'ci_95_upper': U}, ...}).
                                               Currently used for title/legend, not plotting CIs on simulation.
        output_dir (str): Directory to save the plot.
        filename (str): Name for the output plot file.
    """
    # --- Input Validation ---
    required_data_keys = ['time', 'measured_variables', 'measurements', 'errors']
    # *** MODIFIED: Check for 'measured_variables' instead of 'metabolites' ***
    if not all(k in data for k in required_data_keys):
        missing = set(required_data_keys) - set(data.keys())
        raise ValueError(f"Input 'data' dictionary is missing required keys {missing}.")
    # ************************************************************************

    if not callable(simulation_function):
        raise TypeError("simulation_function must be callable.")
    if not isinstance(fitted_parameters, (list, np.ndarray)):
         raise TypeError("fitted_parameters must be a list or numpy array.")
    fitted_parameters = np.array(fitted_parameters) # Ensure numpy array

    time_points = data['time']
    measured_variables = data['measured_variables']
    measurements = data['measurements']
    errors = data['errors']

    if not (len(time_points) == measurements.shape[0] == errors.shape[0]):
        raise ValueError("Time points, measurements, and errors must have the same number of rows (time steps).")
    if not (len(measured_variables) == measurements.shape[1] == errors.shape[1]):
        raise ValueError("Number of measured_variables must match the number of columns in measurements and errors.")

    # --- Simulation ---
    print("--- Running simulation with fitted parameters for plotting ---")
    try:
        # Simulate model with fitted parameters
        # The simulation_function needs to return the simulated values
        # corresponding *only* to the measured variables, in the *same order* as data['measured_variables']
        # This logic is handled within DataFitter._objective_function currently.
        # We need a way to get the full simulation result mapped correctly.
        # For now, assume simulation_function passed from GUI does this extraction.
        simulated_output = simulation_function(fitted_parameters)
        if simulated_output.shape != measurements.shape:
             warnings.warn(f"Simulation output shape {simulated_output.shape} does not match "
                           f"measurement shape {measurements.shape}. Plotting may fail or be incorrect. "
                           "Ensure the simulation function provided only returns the measured variables "
                           "in the correct order.")
             # Attempt to reshape or slice if possible? Risky. Best to fix the simulation_function passed.
             # If simulation returns full state vector, we need mapping indices here too.
             # Let's assume for now the shapes match.

    except Exception as e:
        print(f"Error during simulation for plotting: {e}")
        # Optionally, plot only data if simulation fails
        simulated_output = np.full_like(measurements, np.nan) # Plot NaNs if simulation fails


    # --- Plotting ---
    num_variables = len(measured_variables)
    # Determine grid size (e.g., aim for roughly square)
    ncols = int(np.ceil(np.sqrt(num_variables)))
    nrows = int(np.ceil(num_variables / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False, sharex=True)
    fig.suptitle("Model Fitting Results", fontsize=16)
    axes_flat = axes.flatten() # Flatten for easy iteration

    for i, var_name in enumerate(measured_variables):
        ax = axes_flat[i]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # Plot experimental data with error bars
        ax.errorbar(time_points, measurements[:, i], yerr=errors[:, i],
                    fmt=marker, linestyle='', label='Data', color=color, capsize=3, markersize=5)

        # Plot simulation results
        # Handle potential NaNs from failed simulation
        valid_sim_indices = ~np.isnan(simulated_output[:, i])
        if np.any(valid_sim_indices):
             ax.plot(time_points[valid_sim_indices], simulated_output[valid_sim_indices, i],
                     linestyle='-', linewidth=2, label='Model Fit', color=color)
        elif np.any(~np.isnan(measurements[:,i])):
             # If sim failed but data exists, note it on plot
              ax.text(0.5, 0.5, "Sim Failed", horizontalalignment='center',
                      verticalalignment='center', transform=ax.transAxes, color='red', fontsize=10)


        ax.set_title(var_name.replace('_', ' '), fontsize=10)
        ax.set_ylabel("Value", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i >= num_variables - ncols: # Only add x-label to bottom row
             ax.set_xlabel("Time", fontsize=9)
        if i == 0: # Add legend to the first plot only
             ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(num_variables, nrows * ncols):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Saving ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show() # Display the plot interactively

# Example usage (for testing purposes, needs appropriate inputs)
if __name__ == '__main__':
    # Create dummy data and inputs for testing plot_results
    print("Testing plotting function...")
    dummy_data = {
        'time': np.linspace(0, 100, 11),
        'measured_variables': ['MetA_Total', 'MetB_C4_FE', 'MetC_Total'],
        'measurements': np.array([[10-i*0.5, 0.1+i*0.05, 5] for i in range(11)]),
        'errors': np.random.rand(11, 3) * 0.5 + 0.1
    }
    dummy_params = np.array([1.0, 2.0])
    def dummy_sim_func(params):
        # Simulate some curves based on params (example)
        k1, k2 = params
        t = dummy_data['time']
        sim_metA = 10 * np.exp(-k1 * t / 50)
        sim_metB = 1.0 - np.exp(-k2 * t / 100)
        sim_metC = np.full_like(t, 5.0)
        # Return in the same order as measured_variables
        return np.vstack([sim_metA, sim_metB, sim_metC]).T

    plot_results(dummy_data, dummy_sim_func, dummy_params)