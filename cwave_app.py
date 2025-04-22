import sys
import os
import collections

# Import project modules
from models.metabolism import MetabolicModel
from models.kinetics import KineticModel
from analysis.isotopomer import IsotopomerHandler
from fitting.fitter import DataFitter
from analysis.stats import MonteCarloAnalyzer
from utils.plotting import plot_results
from utils.helpers import load_data, load_parameters, save_results

# --- Configuration ---
# Define file paths relative to this script's location
# It's often better to use absolute paths or more robust path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir, "data", "example_labeling_data.csv")
PARAMS_FILE = os.path.join(script_dir, "data", "example_model_params.json")
RESULTS_DIR = os.path.join(script_dir, "results") # Directory to save outputs

# --- Model Definition ---
# This section defines the specific metabolic model to be simulated.
# It should match the system you are studying.
# This example is loosely based on Fig 2 from the PDF, highly simplified.
def define_example_model():
    """
    Defines a simplified example metabolic model for demonstration.
    Includes metabolites, reactions, kinetics, and atom mappings.
    """
    model = MetabolicModel()
    kinetics = KineticModel() # Uses predefined kinetics

    # == Metabolites ==
    # Arguments: metabolite_id, carbon_count, name, compartment, initial_concentration, initial_labeling, is_substrate, is_constant
    # Initial Labeling: Array of fractional enrichment (0 to 1) for each carbon. None defaults to natural abundance (~0.011).
    # Example: 99% U-13C Glucose (assuming 6 carbons)
    glc_labeling = np.full(6, 0.99) # 99% labeled at all 6 carbons
    # natural_abundance = np.full(6, 0.011)

    model.add_metabolite("GLC_ext", 6, name="External Glucose", initial_concentration=5.0, initial_labeling=glc_labeling, is_substrate=True, is_constant=True)
    model.add_metabolite("GLC", 6, name="Brain Glucose", initial_concentration=1.0)
    model.add_metabolite("LAC", 3, name="Lactate", initial_concentration=1.0)
    model.add_metabolite("GLU_g", 5, name="Glial Glutamate", initial_concentration=8.0)
    model.add_metabolite("GLN", 5, name="Glutamine", initial_concentration=4.0)
    model.add_metabolite("GLU_n", 5, name="Neuronal Glutamate", initial_concentration=10.0)
    # Add other metabolites as needed (e.g., pyruvate, TCA intermediates)

    # == Reactions ==
    # Arguments: reaction_id, reactants={'ID': stoich}, products={'ID': stoich}, kinetic_model_id, atom_mapping_str, reversible
    # Atom Mapping Format: "Reactant1[c1,c2,...] + ... -> Product1[Origin1,Origin2,...] + ..."
    # Origin Format: 'ReactantIDCarbonIndex' (e.g., 'GLC1') or '?' for untracked. Indices are 1-based.

    # Glucose Transport (simplified)
    model.add_reaction("Vtrans", {"GLC_ext": 1}, {"GLC": 1}, "MichaelisMenten",
                       atom_mapping_str="GLC_ext[1,2,3,4,5,6] -> GLC[1,2,3,4,5,6]", reversible=True) # Assume reversible for MM form

    # Glycolysis (simplified: Glucose -> 2 Lactate)
    # Mapping: GLC[1,2,3,4,5,6] -> 2 * LAC[3,2,1] (Note reverse order for lactate carbons)
    # This requires careful handling of stoichiometry in labeling, simplified here.
    # We map one GLC to one LAC for labeling demo, rate scales with total flux.
    model.add_reaction("Vgly", {"GLC": 1}, {"LAC": 2}, "MichaelisMenten",
                       atom_mapping_str="GLC[1,2,3,4,5,6] -> LAC[3,2,1] + LAC[6,5,4]", reversible=False) # Simplified mapping for demo

    # Pyruvate Carboxylase (simplified: Lactate -> Glial Glutamate C1-C3)
    # Mapping: LAC[1,2,3] -> GLU_g[?,?,1,2,3] (Assuming carbons 1-3 come from Lac)
    model.add_reaction("Vpc", {"LAC": 1}, {"GLU_g": 1}, "MassAction",
                       atom_mapping_str="LAC[1,2,3] -> GLU_g[?,?,1,2,3]", reversible=False) # Highly simplified

    # Glutamine Synthetase
    # Mapping: GLU_g[1,2,3,4,5] -> GLN[1,2,3,4,5]
    model.add_reaction("Vsyn", {"GLU_g": 1}, {"GLN": 1}, "MichaelisMenten",
                       atom_mapping_str="GLU_g[1,2,3,4,5] -> GLN[1,2,3,4,5]", reversible=False)

    # Glutamate-Glutamine Cycle (Glutaminase in neuron)
    # Mapping: GLN[1,2,3,4,5] -> GLU_n[1,2,3,4,5]
    model.add_reaction("Vcycle", {"GLN": 1}, {"GLU_n": 1}, "MassAction",
                       atom_mapping_str="GLN[1,2,3,4,5] -> GLU_n[1,2,3,4,5]", reversible=False)

    # Glial TCA Activity Placeholder (consumes/produces Glial Glu)
    model.add_reaction("Vtca_g", {"GLU_g": 1}, {"GLU_g": 1}, "MassAction",
                       atom_mapping_str="GLU_g[1,2,3,4,5] -> GLU_g[1,2,3,4,5]", reversible=True) # Placeholder self-loop

    # Neuronal TCA Activity Placeholder (consumes/produces Neuronal Glu)
    model.add_reaction("Vtca_n", {"GLU_n": 1}, {"GLU_n": 1}, "MassAction",
                       atom_mapping_str="GLU_n[1,2,3,4,5] -> GLU_n[1,2,3,4,5]", reversible=True) # Placeholder self-loop

    # Glutamine Efflux/Dilution
    model.add_reaction("Vefflux_gln", {"GLN": 1}, {}, "MassAction",
                       atom_mapping_str="GLN[1,2,3,4,5] -> ?", reversible=False) # Label leaves system


    return model, kinetics

# --- Main Execution Logic ---
def main():
    """
    Main function to run the CWAVE application recreation.
    Orchestrates model definition, data loading, fitting, analysis, and visualization.
    """
    print("--- Running CWAVE Python Recreation ---")

    # 1. Define Model and Kinetics
    try:
        metabolic_model, kinetic_model = define_example_model()
    except Exception as e:
        print(f"Error defining model: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load Data and Parameters
    try:
        print(f"Loading data from: {DATA_FILE}")
        experimental_data = load_data(DATA_FILE)
        print(f"Loading parameters from: {PARAMS_FILE}")
        # This is the full info dict with initial values and bounds
        parameters_info = load_parameters(PARAMS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data and parameter files exist.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, Exception) as e:
        print(f"Error loading data or parameters: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Initialize Handlers
    try:
        isotopomer_handler = IsotopomerHandler(metabolic_model)
        fitter = DataFitter(metabolic_model, kinetic_model, isotopomer_handler,
                            experimental_data, parameters_info)
    except (TypeError, ValueError, RuntimeError) as e:
         print(f"Error initializing handlers or fitter: {e}", file=sys.stderr)
         sys.exit(1)

    # 4. Perform Parameter Fitting
    fitted_params_dict = None
    fit_result_obj = None
    try:
        # Using least_squares is generally recommended for this type of problem
        fitted_params_dict, fit_result_obj = fitter.fit_data(method="least_squares", verbose=1) # verbose=1 for least_squares progress
    except Exception as e:
        print(f"Critical error during fitting process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Attempt to continue if possible, or exit
        # sys.exit(1) # Or try to proceed without fitting results

    # 5. Perform Statistical Analysis (if fitting was successful)
    uncertainty_results = None
    if fitted_params_dict is not None:
        try:
            analyzer = MonteCarloAnalyzer(metabolic_model, kinetic_model, isotopomer_handler,
                                          experimental_data, parameters_info, # Pass original info with bounds
                                          fitted_params_dict)
            # Run Monte Carlo - adjust iterations as needed
            uncertainty_results = analyzer.analyze_uncertainty(num_iterations=100, fit_method="least_squares") # Use fewer iterations for testing
        except Exception as e:
            print(f"Error during Monte Carlo analysis: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Monte Carlo analysis because parameter fitting failed or was skipped.")

    # 6. Visualize Results (if fitting was successful)
    if fitted_params_dict is not None and hasattr(fitter, 'simulation_function_fitted') and fitter.simulation_function_fitted is not None:
        print("Generating results plot...")
        try:
            # The fitter._simulate method needs adaptation to be directly callable for plotting
            # Or we need a dedicated simulation function that takes params and returns mapped output
            # Let's create a wrapper for plotting
            def simulation_for_plot(param_vector):
                 sim_output = fitter._simulate(param_vector)
                 if sim_output is None:
                      # Return shape expected by plotter, filled with NaN
                      return np.full_like(experimental_data['measurements'], np.nan)
                 return sim_output

            plot_results(
                data=experimental_data, # Pass original data for plotting experimental points
                simulation_function=simulation_for_plot, # Pass the simulation wrapper
                fitted_parameters=list(fitted_params_dict.values()), # Pass fitted params as list/vector
                uncertainty_results=uncertainty_results['summary_statistics'] if uncertainty_results else None, # Pass summary stats for CI
                output_dir=RESULTS_DIR
            )
        except Exception as e:
            print(f"Error generating plot: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    else:
        print("Skipping plotting because parameter fitting failed or simulation function is unavailable.")

    # 7. Save Results
    print("Saving results...")
    results_to_save = {
        "fitted_parameters": fitted_params_dict,
        # Add raw fit result object info if needed (might not be JSON serializable)
        # "fit_result_details": repr(fit_result_obj),
        "monte_carlo_summary": uncertainty_results['summary_statistics'] if uncertainty_results else None,
        # Optionally save full parameter distributions if needed (can be large)
        # "monte_carlo_distributions": uncertainty_results['parameter_distributions'] if uncertainty_results else None,
    }
    try:
        save_results(results_to_save, output_dir=RESULTS_DIR)
    except Exception as e:
         print(f"Error saving results: {e}", file=sys.stderr)


    print("\n--- CWAVE Python Recreation Finished ---")

if __name__ == "__main__":
    # Import necessary libraries here if not already imported globally
    # This ensures they are available when the script is run directly
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import scipy # Ensure main libraries are importable
    except ImportError as e:
        print(f"Error: Missing required Python library: {e}. Please install dependencies.", file=sys.stderr)
        print("Typically: pip install numpy scipy pandas matplotlib", file=sys.stderr)
        sys.exit(1)

    main()

