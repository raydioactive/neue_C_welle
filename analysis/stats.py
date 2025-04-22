# analysis/stats.py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import time
import warnings
import collections # Ensure collections is imported

# Import from other project modules
try:
    # Use relative imports if structure allows, otherwise adjust path as needed
    from fitting.fitter import DataFitter
    from models.metabolism import MetabolicModel
    from models.kinetics import KineticModel
    from analysis.isotopomer import IsotopomerHandler
except ImportError as e:
    print(f"Error importing MonteCarloAnalyzer dependencies: {e}")
    # Depending on usage context, might raise or just warn
    raise


class MonteCarloAnalyzer:
    """
    Performs Monte Carlo uncertainty analysis on fitted parameters.
    """
    def __init__(self, metabolic_model: MetabolicModel, kinetic_model: KineticModel,
                 isotopomer_handler: IsotopomerHandler, original_data: dict,
                 original_parameters_info: dict, fitted_parameters_dict: dict):
        """
        Initializes the MonteCarloAnalyzer.

        Args:
            metabolic_model: The defined metabolic model instance.
            kinetic_model: The defined kinetic model instance.
            isotopomer_handler: The initialized isotopomer handler instance.
            original_data: The original experimental data dictionary.
            original_parameters_info: Dict with parameter bounds and *initial* values (used if fit fails).
            fitted_parameters_dict: Dict of the *point estimates* of parameters from the initial fit.
        """
        self.metabolic_model = metabolic_model
        self.kinetic_model = kinetic_model
        self.isotopomer_handler = isotopomer_handler
        self.original_data = original_data
        self.original_parameters_info = original_parameters_info # Store for bounds and potential initial values
        self.fitted_parameters_dict = fitted_parameters_dict

        # Validate data structure
        required_data_keys = ['time', 'measured_variables', 'measurements', 'errors']
        if not all(k in self.original_data for k in required_data_keys):
            raise ValueError(f"Original data dictionary is missing required keys: {required_data_keys}")

        print("MonteCarloAnalyzer initialized (Isotopomer Tracking Enabled).")


    def _generate_noisy_data(self):
        """Generates one instance of noisy data based on original data and errors."""
        noisy_measurements = np.zeros_like(self.original_data['measurements'])
        measurement_shape = self.original_data['measurements'].shape

        for i in range(measurement_shape[0]): # Time points
             for j in range(measurement_shape[1]): # Variables
                  mean = self.original_data['measurements'][i, j]
                  std_dev = self.original_data['errors'][i, j]
                  if std_dev <= 0:
                       # Handle zero or negative error - maybe use a small fraction of mean?
                       std_dev = abs(mean) * 0.01 if mean != 0 else 1e-6 # Avoid zero std dev
                       # warnings.warn(f"Zero or negative error found for measurement [{i},{j}]. Using small value: {std_dev:.2g}")

                  # Generate noise from normal distribution
                  noisy_measurements[i, j] = np.random.normal(loc=mean, scale=std_dev)

        noisy_data_instance = self.original_data.copy() # Shallow copy is okay
        noisy_data_instance['measurements'] = noisy_measurements
        # Keep original errors for weighting in the objective function if needed by fitter
        # noisy_data_instance['errors'] = self.original_data['errors']
        return noisy_data_instance


    def analyze_uncertainty(self, num_iterations=100, fit_method='least_squares', progress_callback=None):
        """
        Performs Monte Carlo simulation by refitting the model to noisy data replicas.

        Args:
            num_iterations (int): Number of Monte Carlo iterations to perform.
            fit_method (str): Fitting method to use ('least_squares').
            progress_callback (callable, optional): A function to call after each
                                                     iteration for progress updates.
                                                     Expected signature: callback(current_iter, total_iter)

        Returns:
            dict: Dictionary containing results ('parameter_distributions', 'summary_statistics',
                                                'successful_iterations', 'total_iterations').
        """
        print(f"\n--- Starting Monte Carlo Uncertainty Analysis ({num_iterations} iterations, Isotopomer Model) ---")
        start_time = time.time()

        all_fitted_params = [] # Store successful parameter vectors
        successful_fits = 0

        # Use the point estimate as the initial guess for each MC fit
        initial_guess = list(self.fitted_parameters_dict.values())

        for i in range(num_iterations):
            if (i + 1) % 10 == 0 or i == 0: # Log progress periodically
                 print(f"  Iteration {i+1}/{num_iterations}... (Successful fits: {successful_fits}, Time: {time.time() - start_time:.1f}s)")

            # 1. Generate noisy data replica
            noisy_data = self._generate_noisy_data()

            # 2. Initialize a new DataFitter instance for this replica
            #    It's important to re-initialize to reset any internal state if necessary,
            #    though if DataFitter is stateless after __init__, re-using might be faster (but riskier).
            #    Let's re-initialize for safety.
            try:
                # Use the *original* parameter info for bounds, but the *fitted* params as initial guess for MC fits
                mc_fitter = DataFitter(
                    self.metabolic_model, self.kinetic_model, self.isotopomer_handler,
                    noisy_data, self.original_parameters_info # Use original info for structure/bounds
                )
                # Override initial values with the best fit from the original data
                mc_fitter.initial_parameter_values = np.array(initial_guess)

            except Exception as e:
                 warnings.warn(f"MC Iteration {i+1}: Error initializing DataFitter. Skipping. Error: {e}")
                 if progress_callback: progress_callback(i + 1, num_iterations) # Still report progress
                 continue # Skip to next iteration

            # 3. Re-fit the model to the noisy data
            try:
                # Use verbose=0 for MC fits to avoid excessive console output
                fitted_params_mc, fit_result_mc = mc_fitter.fit_data(method=fit_method, verbose=0)

                if fitted_params_mc and fit_result_mc is not None and fit_result_mc.success:
                    # Ensure the order matches the original parameter IDs
                    param_vector = [fitted_params_mc[p_id] for p_id in self.fitted_parameters_dict.keys()]
                    all_fitted_params.append(param_vector)
                    successful_fits += 1
                # else: Fit failed or didn't converge, don't store params for this iteration

            except Exception as e:
                 warnings.warn(f"MC Iteration {i+1}: Error during fitting. Skipping. Error: {e}")
                 # Optionally log traceback here for debugging specific MC failures
                 # import traceback; traceback.print_exc()

            # 4. Report progress via callback (if provided)
            if progress_callback:
                try:
                    progress_callback(i + 1, num_iterations)
                except Exception as cb_e:
                    warnings.warn(f"Progress callback failed: {cb_e}") # Don't let callback crash MC


        end_time = time.time()
        print(f"--- Monte Carlo Analysis completed in {end_time - start_time:.2f} seconds ---")
        print(f"    Successful fits: {successful_fits} / {num_iterations}")

        # 4. Analyze the distribution of fitted parameters
        results = {
            'parameter_distributions': {},
            'summary_statistics': {},
            'successful_iterations': successful_fits,
            'total_iterations': num_iterations
        }

        if successful_fits > 1: # Need at least 2 points for statistics
            param_array = np.array(all_fitted_params)
            param_ids = list(self.fitted_parameters_dict.keys()) # Ensure consistent order

            summary_stats = collections.OrderedDict() # Use OrderedDict to maintain order
            for j, param_id in enumerate(param_ids):
                distribution = param_array[:, j]
                results['parameter_distributions'][param_id] = distribution

                mean_val = np.mean(distribution)
                std_val = np.std(distribution)
                # Calculate 95% confidence interval using percentiles
                ci_lower = np.percentile(distribution, 2.5)
                ci_upper = np.percentile(distribution, 97.5)

                summary_stats[param_id] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper
                }
                print(f"  Parameter '{param_id}': Mean={mean_val:.4g}, Std={std_val:.4g}, 95% CI=[{ci_lower:.4g}, {ci_upper:.4g}]")

            results['summary_statistics'] = summary_stats
        else:
            print("  Not enough successful fits to calculate reliable statistics.")

        return results