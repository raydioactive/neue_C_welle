# Filename: cwave_app_gui.py
import sys
import os
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading # To run long tasks without freezing the GUI
import traceback

# Import project modules
try:
    from models.metabolism import MetabolicModel
    from models.kinetics import KineticModel
    from analysis.isotopomer import IsotopomerHandler
    from fitting.fitter import DataFitter
    from analysis.stats import MonteCarloAnalyzer # Import the updated analyzer
    from utils.plotting import plot_results
    from utils.helpers import load_data, load_parameters, save_results
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import necessary modules: {e}")
    sys.exit(1)

# Import numerical/plotting libraries
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
     messagebox.showerror("Import Error", f"Missing required library: {e}")
     sys.exit(1)


# --- Default Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(script_dir, "data", "example_labeling_data.csv")
DEFAULT_PARAMS_FILE = os.path.join(script_dir, "data", "example_model_params.json")
DEFAULT_RESULTS_DIR = os.path.join(script_dir, "results")
DEFAULT_MC_ITERATIONS = 100 # Default value for Monte Carlo

# --- Model Definition ---
def define_example_model():
    # (Keep the same model definition as before)
    model = MetabolicModel()
    kinetics = KineticModel()
    glc_labeling = np.full(6, 0.99)
    model.add_metabolite("GLC_ext", 6, name="External Glucose", initial_concentration=5.0, initial_labeling=glc_labeling, is_substrate=True, is_constant=True)
    model.add_metabolite("GLC", 6, name="Brain Glucose", initial_concentration=1.0)
    model.add_metabolite("LAC", 3, name="Lactate", initial_concentration=1.0)
    model.add_metabolite("GLU_g", 5, name="Glial Glutamate", initial_concentration=8.0)
    model.add_metabolite("GLN", 5, name="Glutamine", initial_concentration=4.0)
    model.add_metabolite("GLU_n", 5, name="Neuronal Glutamate", initial_concentration=10.0)
    # Ensure add_reaction stores 'atom_mapping_str' in the reaction dict in metabolism.py
    model.add_reaction("Vtrans", {"GLC_ext": 1}, {"GLC": 1}, "MichaelisMentenReversible", atom_mapping_str="GLC_ext[1,2,3,4,5,6] -> GLC[1,2,3,4,5,6]", reversible=True)
    model.add_reaction("Vgly", {"GLC": 1}, {"LAC": 2}, "MichaelisMenten", atom_mapping_str="GLC[1,2,3,4,5,6] -> LAC[3,2,1] + LAC[6,5,4]", reversible=False)
    model.add_reaction("Vpc", {"LAC": 1}, {"GLU_g": 1}, "MassAction", atom_mapping_str="LAC[1,2,3] -> GLU_g[?,?,1,2,3]", reversible=False)
    model.add_reaction("Vsyn", {"GLU_g": 1}, {"GLN": 1}, "MichaelisMenten", atom_mapping_str="GLU_g[1,2,3,4,5] -> GLN[1,2,3,4,5]", reversible=False)
    model.add_reaction("Vcycle", {"GLN": 1}, {"GLU_n": 1}, "MassAction", atom_mapping_str="GLN[1,2,3,4,5] -> GLU_n[1,2,3,4,5]", reversible=False)
    model.add_reaction("Vtca_g", {"GLU_g": 1}, {"GLU_g": 1}, "MassActionReversible", atom_mapping_str="GLU_g[1,2,3,4,5] -> GLU_g[1,2,3,4,5]", reversible=True)
    model.add_reaction("Vtca_n", {"GLU_n": 1}, {"GLU_n": 1}, "MassActionReversible", atom_mapping_str="GLU_n[1,2,3,4,5] -> GLU_n[1,2,3,4,5]", reversible=True)
    model.add_reaction("Vefflux_gln", {"GLN": 1}, {}, "MassAction", atom_mapping_str="GLN[1,2,3,4,5] -> ?", reversible=False)
    return model, kinetics

# --- GUI Application Class ---
class CwaveAppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CWAVE Python GUI")
        self.geometry("700x630")

        # --- Data Storage ---
        self.data_file_path = tk.StringVar(value=DEFAULT_DATA_FILE)
        self.params_file_path = tk.StringVar(value=DEFAULT_PARAMS_FILE)
        self.results_dir_path = tk.StringVar(value=DEFAULT_RESULTS_DIR)
        self.mc_iterations_var = tk.IntVar(value=DEFAULT_MC_ITERATIONS)

        self.metabolic_model = None
        self.kinetic_model = None
        self.experimental_data = None
        self.parameters_info = None
        self.isotopomer_handler = None
        self.fitter = None
        self.fitted_params_dict = None
        self.fit_result_obj = None
        self.analyzer = None
        self.uncertainty_results = None

        # --- GUI Widgets ---
        self.create_widgets()

    # (log_message method remains the same)
    def log_message(self, message):
        if self.status_text:
             def append_log():
                  self.status_text.insert(tk.END, message + "\n")
                  self.status_text.see(tk.END)
             self.after(0, append_log)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File Selection Frame
        file_frame = ttk.LabelFrame(main_frame, text="Input Files & Output", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_frame, text="Data File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(file_frame, textvariable=self.data_file_path, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_data_file).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(file_frame, text="Params File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(file_frame, textvariable=self.params_file_path, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_params_file).grid(row=1, column=2, padx=5, pady=2)
        ttk.Label(file_frame, text="Results Dir:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(file_frame, textvariable=self.results_dir_path, width=50).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_results_dir).grid(row=2, column=2, padx=5, pady=2)
        file_frame.columnconfigure(1, weight=1)

        # Options Frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        ttk.Label(options_frame, text="MC Iterations:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        # Use validatecommand later if needed for strict integer input
        mc_iter_entry = ttk.Entry(options_frame, textvariable=self.mc_iterations_var, width=10)
        mc_iter_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Control Buttons Frame
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=(5,0))
        self.load_button = ttk.Button(control_frame, text="1. Load", command=self.load_model_and_data)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.fit_button = ttk.Button(control_frame, text="2. Fit", command=self.run_fitting_thread, state=tk.DISABLED)
        self.fit_button.pack(side=tk.LEFT, padx=5)
        self.mc_button = ttk.Button(control_frame, text="3. Monte Carlo", command=self.run_monte_carlo_thread, state=tk.DISABLED)
        self.mc_button.pack(side=tk.LEFT, padx=5)
        self.plot_button = ttk.Button(control_frame, text="4. Plot", command=self.display_plot, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(control_frame, text="5. Save", command=self.save_all_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Progress Bar Frame
        progress_frame = ttk.Frame(main_frame, padding=(5, 2, 5, 5))
        progress_frame.pack(fill=tk.X)
        self.progress_label = ttk.Label(progress_frame, text="Progress:")
        # Progress bar will be configured in show_progress
        self.progressbar = ttk.Progressbar(progress_frame, length=200)

        # Status Text Area
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(5,5))
        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, height=15, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True)

    # (browse_* methods remain the same)
    def browse_data_file(self):
        path = filedialog.askopenfilename(title="Select Data File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialdir=os.path.dirname(self.data_file_path.get()) or script_dir)
        if path: self.data_file_path.set(path)
    def browse_params_file(self):
        path = filedialog.askopenfilename(title="Select Parameters File", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=os.path.dirname(self.params_file_path.get()) or script_dir)
        if path: self.params_file_path.set(path)
    def browse_results_dir(self):
        path = filedialog.askdirectory(title="Select Results Directory", initialdir=self.results_dir_path.get() or script_dir)
        if path: self.results_dir_path.set(path)

    # (set_buttons_state remains the same)
    def set_buttons_state(self, state):
        def update_state():
            self.load_button.config(state=state)
            self.fit_button.config(state=tk.NORMAL if state == tk.NORMAL and self.fitter else tk.DISABLED)
            self.mc_button.config(state=tk.NORMAL if state == tk.NORMAL and self.fitted_params_dict else tk.DISABLED)
            plot_enabled = state == tk.NORMAL and self.fitter and self.fitter.simulation_function_for_plot is not None
            self.plot_button.config(state=tk.NORMAL if plot_enabled else tk.DISABLED)
            self.save_button.config(state=tk.NORMAL if state == tk.NORMAL and (self.fitted_params_dict or self.uncertainty_results) else tk.DISABLED)
        self.after(0, update_state)

    # (run_in_thread remains the same)
    def run_in_thread(self, target_func, on_complete=None, on_start=None):
        self.set_buttons_state(tk.DISABLED)
        if on_start: on_start()
        def thread_target():
            try: target_func()
            finally:
                self.set_buttons_state(tk.NORMAL)
                if on_complete: on_complete()
        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()

    # --- Modified show_progress ---
    def show_progress(self, label_text="Processing...", mode='indeterminate', maximum=100):
        """Shows and starts the progress bar."""
        def action():
            self.progress_label.config(text=label_text)
            self.progressbar.config(mode=mode, maximum=maximum, value=0) # Configure mode/max
            self.progress_label.pack(side=tk.LEFT, padx=(0, 5))
            self.progressbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if mode == 'indeterminate':
                self.progressbar.start(10) # Start animation
            else:
                self.progressbar.stop() # Stop animation if switching from indeterminate
        self.after(0, action) # Schedule GUI update

    # (hide_progress remains the same)
    def hide_progress(self):
        def action():
            self.progressbar.stop()
            self.progressbar.pack_forget()
            self.progress_label.pack_forget()
        self.after(0, action)

    # (load_model_and_data remains the same)
    def load_model_and_data(self):
        # Use the latest working version
        self.log_message("--- Loading Model and Data ---")
        try:
            self.log_message("Defining model...")
            self.metabolic_model, self.kinetic_model = define_example_model()
            self.log_message("Model definition complete.")
            data_path = self.data_file_path.get()
            params_path = self.params_file_path.get()
            self.log_message(f"Loading data from: {data_path}")
            self.experimental_data = load_data(data_path)
            self.log_message(f"Loading parameters from: {params_path}")
            self.parameters_info = load_parameters(params_path)
            self.log_message("Initializing handlers...")
            self.isotopomer_handler = IsotopomerHandler.from_metabolic_model(self.metabolic_model)
            self.fitter = DataFitter(self.metabolic_model, self.kinetic_model, self.isotopomer_handler, self.experimental_data, self.parameters_info)
            self.log_message("Model and data loaded successfully.")
            self.set_buttons_state(tk.NORMAL)
        except Exception as e: # Catch-all for simplified example
            self.log_message(f"Error during loading: {e}")
            self.log_message(traceback.format_exc())
            messagebox.showerror("Loading Error", f"An error occurred:\n{e}")
            self.fitter = None
            self.set_buttons_state(tk.NORMAL)

    # (_perform_fitting remains the same)
    def _perform_fitting(self):
        # Use the latest working version
        if not self.fitter: self.log_message("Error: Model/data not loaded."); return
        self.log_message("\n--- Running Parameter Fitting ---")
        self.fitted_params_dict = None; self.fit_result_obj = None
        try:
            self.fitted_params_dict, self.fit_result_obj = self.fitter.fit_data(method="least_squares", verbose=1)
            if self.fitted_params_dict:
                 self.log_message("Fitting completed successfully.")
                 # (Log parameters and cost as before)
                 self.log_message("Fitted Parameters:")
                 for p_id, val in self.fitted_params_dict.items(): self.log_message(f"  {p_id}: {val:.4g}")
                 if hasattr(self.fit_result_obj, 'cost') and self.fit_result_obj.cost is not None: self.log_message(f"Final Cost (0.5 * sum(residuals^2)): {self.fit_result_obj.cost:.4g}")

            else:
                 self.log_message("Fitting did not converge or failed.")
                 if self.fit_result_obj and hasattr(self.fit_result_obj, 'message'): self.log_message(f"Reason: {self.fit_result_obj.message}")
        except Exception as e:
            self.log_message(f"Critical error during fitting process: {e}")
            self.log_message(traceback.format_exc())
            messagebox.showerror("Fitting Error", f"An error occurred during fitting:\n{e}")


    # --- Modified run_fitting_thread ---
    def run_fitting_thread(self):
        """Starts the fitting process with INDETERMINATE progress bar."""
        self.run_in_thread(
            target_func=self._perform_fitting,
            on_start=lambda: self.show_progress("Fitting parameters...", mode='indeterminate'),
            on_complete=self.hide_progress
        )

    # --- Modified _perform_monte_carlo ---
    def _perform_monte_carlo(self):
        """Internal Monte Carlo logic (to be run in a thread)."""
        if not self.fitter or not self.fitted_params_dict:
            self.log_message("Error: Fitting must be done before Monte Carlo.")
            return

        try:
             num_iterations = self.mc_iterations_var.get()
             if num_iterations <= 0: messagebox.showerror("Input Error", "MC iterations must be positive."); return
        except tk.TclError: messagebox.showerror("Input Error", "Invalid number for MC iterations."); return

        self.log_message(f"\n--- Running Monte Carlo Analysis ({num_iterations} iterations) ---")
        self.uncertainty_results = None

        # --- Define progress callback ---
        def update_mc_progress(current_iter, total_iter):
            # This function will be called from the worker thread
            # We need to schedule the GUI update on the main thread
            def update_gui():
                 # Update progress bar value (ensure maximum is set correctly)
                 self.progressbar['value'] = current_iter
            self.after(0, update_gui) # Schedule update_gui to run in main thread

        try:
            self.analyzer = MonteCarloAnalyzer(
                 metabolic_model=self.metabolic_model,
                 kinetic_model=self.kinetic_model,
                 isotopomer_handler=self.isotopomer_handler,
                 original_data=self.experimental_data,
                 original_parameters_info=self.parameters_info,
                 fitted_parameters_dict=self.fitted_params_dict
            )
            self.log_message(f"Starting Monte Carlo with {num_iterations} iterations...")

            # Pass the callback function to the analyzer
            mc_results = self.analyzer.analyze_uncertainty(
                num_iterations=num_iterations,
                fit_method="least_squares",
                progress_callback=update_mc_progress # Pass the callback here
            )

            self.uncertainty_results = mc_results.get('summary_statistics')
            total_iters = mc_results.get('total_iterations', num_iterations)
            success_iters = mc_results.get('successful_iterations', 'N/A')
            self.log_message(f"Monte Carlo finished. Successful iterations: {success_iters}/{total_iters}")

            if self.uncertainty_results:
                 self.log_message("--- Monte Carlo Results ---")
                 # (Log results as before)
                 for p_id, stats in self.uncertainty_results.items():
                      mean_str = f"{stats.get('mean', 'N/A'):.4g}" if isinstance(stats.get('mean'), (int, float)) else 'N/A'
                      std_str = f"{stats.get('std', 'N/A'):.4g}" if isinstance(stats.get('std'), (int, float)) else 'N/A'
                      ci_low_str = f"{stats.get('ci_95_lower', 'N/A'):.4g}" if isinstance(stats.get('ci_95_lower'), (int, float)) else 'N/A'
                      ci_up_str = f"{stats.get('ci_95_upper', 'N/A'):.4g}" if isinstance(stats.get('ci_95_upper'), (int, float)) else 'N/A'
                      self.log_message(f"  Parameter '{p_id}': Mean={mean_str}, Std={std_str}, 95% CI=[{ci_low_str}, {ci_up_str}]")
            else: self.log_message("Could not calculate uncertainty statistics.")

        except Exception as e:
            self.log_message(f"Error during Monte Carlo analysis: {e}")
            self.log_message(traceback.format_exc())
            messagebox.showerror("Monte Carlo Error", f"An error occurred:\n{e}")

    # --- Modified run_monte_carlo_thread ---
    def run_monte_carlo_thread(self):
        """Starts the Monte Carlo analysis with a DETERMINATE progress bar."""
        try: # Get iterations to set progress bar maximum
             num_iterations = self.mc_iterations_var.get()
             if num_iterations <= 0: messagebox.showerror("Input Error", "MC iterations must be positive."); return
        except tk.TclError: messagebox.showerror("Input Error", "Invalid number for MC iterations."); return

        self.run_in_thread(
            target_func=self._perform_monte_carlo,
            on_start=lambda: self.show_progress("Running Monte Carlo...", mode='determinate', maximum=num_iterations),
            on_complete=self.hide_progress
        )

    # (display_plot remains the same)
    def display_plot(self):
        # Use latest working version
        if not self.fitter or not self.fitted_params_dict or self.fitter.simulation_function_for_plot is None: messagebox.showerror("Plot Error", "Fitting must be done first."); return
        self.log_message("\n--- Generating Results Plot ---")
        try:
             sim_func = self.fitter.simulation_function_for_plot
             if 'measured_variables' not in self.experimental_data: messagebox.showerror("Plot Error", "Data missing 'measured_variables'."); return
             plot_results(data=self.experimental_data, simulation_function=sim_func, fitted_parameters=list(self.fitted_params_dict.values()), uncertainty_results=self.uncertainty_results, output_dir=self.results_dir_path.get())
             self.log_message(f"Plot generated/displayed and saved to '{self.results_dir_path.get()}'.")
        except Exception as e:
            self.log_message(f"Error generating plot: {e}"); self.log_message(traceback.format_exc()); messagebox.showerror("Plot Error", f"Error during plotting:\n{e}")

    # (save_all_results remains the same)
    def save_all_results(self):
        # Use latest working version
        if not self.fitted_params_dict and not self.uncertainty_results: messagebox.showwarning("Save Warning", "No results available."); return
        self.log_message("\n--- Saving Results ---")
        results_to_save = {}
        if self.fitted_params_dict: results_to_save["fitted_parameters"] = self.fitted_params_dict
        if self.uncertainty_results: results_to_save["monte_carlo_summary"] = self.uncertainty_results
        if not results_to_save: self.log_message("No data structure to save."); return
        try:
            save_results(results_to_save, output_dir=self.results_dir_path.get())
            self.log_message(f"Results saved to directory: {self.results_dir_path.get()}")
            messagebox.showinfo("Save Complete", f"Results saved to\n{self.results_dir_path.get()}")
        except Exception as e:
             self.log_message(f"Error saving results: {e}"); self.log_message(traceback.format_exc()); messagebox.showerror("Save Error", f"Error saving results:\n{e}")


# --- Main Execution ---
if __name__ == "__main__":
    app = CwaveAppGUI()
    try: from analysis.isotopomer import IsotopomerHandler
    except ImportError as e: messagebox.showerror("Init Error", f"Import fail: {e}"); sys.exit(1)
    app.mainloop()