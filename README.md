# CWAVE - Python Simulation Tool for 13C Metabolic Labeling Studies

## Overview

CWAVE is a Python-based application designed for simulating and analyzing dynamic 13C labeling experiments in metabolic networks, particularly inspired by studies involving Magnetic Resonance Spectroscopy (MRS) data[cite: 1]. It allows users to define metabolic models, simulate the time course of both metabolite concentrations and positional 13C fractional enrichment (FE), fit model parameters to experimental data, and perform uncertainty analysis[cite: 1, 4, 5].

This tool integrates kinetic modeling with isotopomer tracking to provide insights into metabolic fluxes and pathway activities[cite: 1, 4].

## Features

* **Metabolic Network Definition**:
    * Define metabolites with names, compartments, carbon counts, initial concentrations, and initial 13C labeling patterns (e.g., natural abundance or specific enrichment).
    * Specify metabolites as constant (boundary conditions) or dynamic.
* **Reaction Definition**:
    * Define reactions with reactants, products, and stoichiometry.
    * Assign predefined or custom kinetic rate laws (e.g., Mass Action, Michaelis-Menten)[cite: 1].
    * Specify reaction reversibility.
* **Atom Mapping & Isotopomer Tracking**[cite: 1, 4]:
    * Define atom transitions using explicit mapping strings (e.g., `GLC[1,2,3,4,5,6] -> LAC[3,2,1] + LAC[6,5,4]`)[cite: 1, 4].
    * Supports tracking of positional 13C fractional enrichment (FE) over time by integrating an expanded ODE system that includes labeling states[cite: 1, 4].
    * Handles untracked carbon sources ('?') assuming natural abundance[cite: 4].
* **Kinetic Simulation**:
    * Simulates the dynamic changes in metabolite concentrations and fractional enrichments using numerical integration (`scipy.integrate.solve_ivp` with BDF method).
    * Utilizes defined kinetic models (Mass Action, Michaelis-Menten, Reversible variants, Constant Flux).
* **Parameter Estimation**[cite: 1]:
    * Fits kinetic parameters to experimental time-course data (concentrations and/or fractional enrichments)[cite: 1].
    * Uses least-squares optimization (`scipy.optimize.least_squares`) with log-parameter transformation and bounds handling.
    * Includes options for multi-start optimization to improve the chance of finding a global minimum.
    * Alternative optimization methods (Nelder-Mead Simplex, Simulated Annealing) are implemented but might require adaptation for the least-squares framework used in the main app.
* **Uncertainty Analysis**[cite: 1, 5]:
    * Performs Monte Carlo simulations to estimate parameter uncertainty[cite: 1, 5].
    * Generates noisy data replicas based on experimental errors[cite: 5].
    * Re-fits the model to noisy data to build parameter distributions[cite: 5].
    * Calculates summary statistics like mean, standard deviation, and confidence intervals (e.g., 95% CI via percentiles)[cite: 5].
* **Data Handling**[cite: 1]:
    * Loads experimental time-course data from CSV files. Data format requires a 'Time' column and columns for measured variables (e.g., `MetaboliteID_Total` or `MetaboliteID_C#_FE`) along with corresponding standard deviation columns (suffix `_SD`)[cite: 9].
    * Loads initial parameter guesses and bounds from JSON files. Parameter keys follow the format `ReactionID_ParameterName` (e.g., `Vgly_Vmax`)[cite: 10].
    * Saves fitting results (parameters) and Monte Carlo analysis summaries to JSON files[cite: 1].
* **Visualization**[cite: 1]:
    * Plots experimental data points with error bars against the model simulation fit for measured variables.
    * Generates multi-panel plots for easy comparison across different measured variables.
    * Saves plots to image files (e.g., PNG).
* **Graphical User Interface (GUI)**[cite: 2]:
    * Provides a Tkinter-based GUI (`cwave_app_gui.py`) for easier interaction[cite: 2].
    * Allows users to browse for data and parameter files, set options (like Monte Carlo iterations), and run the workflow steps (Load, Fit, Monte Carlo, Plot, Save) via buttons[cite: 2].
    * Displays status messages and progress bars for long tasks (fitting, Monte Carlo) using threading to prevent freezing[cite: 2].

## Project Structure
```
CWAVE/
├── cwave_app.py              # Main command-line application script 
├── cwave_app_gui.py          # Main GUI application script 
├── models/                   # Model definition modules
│   ├── metabolism.py         # Metabolic model class (metabolites, reactions, atom maps)
│   └── kinetics.py           # Kinetic rate law definitions and management
├── fitting/                  # Parameter fitting modules
│   ├── fitter.py             # DataFitter class (ODE simulation, least-squares optimization)
│   └── methods.py            # Implementations of Simplex and Simulated Annealing (currently separate)
├── analysis/                 # Analysis modules
│   ├── isotopomer.py         # IsotopomerHandler class (parsing mappings, calculating label fluxes) 
│   └── stats.py              # MonteCarloAnalyzer class for uncertainty analysis 
├── utils/                    # Utility modules
│   ├── helpers.py            # Data loading, results saving, residual calculation
│   └── plotting.py           # Results plotting functions
├── data/                     # Example data and parameters
│   ├── example_labeling_data.csv # Example experimental data (conc + FE) 
│   └── example_model_params.json # Example initial parameters and bounds 
├── results/                  # Default output directory for plots and saved results files 
├── build/                    # (Optional) Directory for build artifacts (e.g., PyInstaller)
│   └── cwave_app/
│       ├── warn-cwave_app.txt    # PyInstaller warnings (useful for dependencies) 
│       └── xref-cwave_app.html   # PyInstaller cross-reference (build analysis)
├── requirements.txt          # File listing Python dependencies
└── README.md                 # This file 
```

## Dependencies

The primary dependencies are:

* Python 3.7+
* **NumPy**: For numerical operations and arrays [cite: 1, 4, 5]
* **SciPy**: For ODE solving (`solve_ivp`) and optimization (`least_squares`) [cite: 1]
* **Pandas**: For loading data from CSV files [cite: 1]
* **Matplotlib**: For generating plots [cite: 1]

You should ```pip install -r requirements.txt ``` the requirements file listing these



## Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd CWAVE
    ```
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows PowerShell:
    .\venv\Scripts\Activate.ps1
    # Windows CMD:
    .\venv\Scripts\activate.bat
    # Linux/macOS:
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have created the `requirements.txt` file as shown above)*

## Usage

### 1. Define Your Model

* Modify the `define_example_model` function inside `cwave_app.py` (or `cwave_app_gui.py` if using the GUI) to represent your specific metabolic system[cite: 1, 2].
* Use `model.add_metabolite()` to define each metabolite, specifying its ID, `carbon_count`, `initial_concentration`, `initial_labeling` (as a list or NumPy array of FE per carbon, defaults to natural abundance if `None`), and whether it's `is_constant`[cite: 1].
* Use `model.add_reaction()` to define each reaction, providing its ID, `reactants` and `products` dictionaries (with stoichiometry), the `kinetic_model_id` (must match an ID in `KineticModel`), the `atom_mapping_str`, and `reversible` status[cite: 1].
    * **Atom Mapping Format**: `Reactant1[c1,c2,...] + ... -> Product1[Origin1,Origin2,...] + ...`[cite: 1, 4].
    * Origins can be explicit reactant carbons (e.g., `GLC1`) or `?` for untracked sources[cite: 1, 4]. Positional mapping (product indices referring to the order of reactant carbons listed) is also supported by the parser[cite: 4].

### 2. Prepare Input Data

* **Experimental Data (CSV)**: Create a CSV file (like `data/example_labeling_data.csv`)[cite: 1, 9].
    * Must include a `Time` column[cite: 9].
    * Include columns for each measured variable (e.g., `GLC_Total`, `GLN_C4_FE`). Names must match `MetaboliteID_Total` or `MetaboliteID_C#_FE` format[cite: 9, 10].
    * Include columns for the standard deviation of each measurement, named with an `_SD` suffix (e.g., `GLC_Total_SD`, `GLN_C4_FE_SD`)[cite: 9]. If missing, errors will be estimated.
* **Parameters (JSON)**: Create a JSON file (like `data/example_model_params.json`)[cite: 1, 10].
    * Define parameters to be fitted as keys following the `ReactionID_ParameterName` convention (e.g., `Vtrans_Vmax_fwd`)[cite: 10].
    * For each parameter, provide `initial_value`, `lower_bound`, and `upper_bound`[cite: 10]. Keys starting with `_` are ignored (comments)[cite: 10].

### 3. Run the Application

* **Command Line**:
    ```bash
    python cwave_app.py
    ```
    * Modify the file paths for data (`DATA_FILE`) and parameters (`PARAMS_FILE`) near the top of `cwave_app.py` if needed[cite: 1].
* **GUI**:
    ```bash
    python cwave_app_gui.py
    ```
    * Use the "Browse..." buttons to select your data file, parameter file, and results directory[cite: 2].
    * Adjust Monte Carlo iterations if desired[cite: 2].
    * Follow the numbered buttons (Load, Fit, Monte Carlo, Plot, Save) to execute the workflow[cite: 2].

### 4. Check Results

* Output plots (e.g., `fit_results.png`) and JSON result files (e.g., `fitted_parameters.json`, `monte_carlo_summary.json`) will be saved in the specified results directory (default: `results/`)[cite: 1].

## Important Notes & Limitations

* **Example Model**: The default model in the scripts is a simplified example and **must** be replaced with your actual biological system[cite: 1, 3].
* **Atom Mapping Accuracy**: The accuracy of the isotopomer simulation critically depends on the correctness of the `atom_mapping_str` for each reaction[cite: 3, 4]. Double-check these mappings.
* **ODE Solver**: The simulation uses `scipy.integrate.solve_ivp` (typically with 'BDF' method for potentially stiff systems). Performance and accuracy can depend on solver choice and tolerances (`rtol`, `atol`). Tuning might be required for complex or stiff models[cite: 3].
* **Isotopomer Framework**: This implementation uses positional fractional enrichment (FE)[cite: 1, 4]. More complex labeling patterns or data types (e.g., MS fragmentation) might require more advanced frameworks (like EMU or Cumomer), which are not implemented here[cite: 3].
* **Reversible Reactions**: Label flux handling in reversible reactions assumes the net flux calculated by the kinetic model determines the direction of label transfer for the FE ODEs. Rigorous handling near equilibrium might require further refinement[cite: 3].
* **Experimental Data Correction**: Placeholders for functions like natural abundance correction or NMR resonance overlap handling exist in `analysis/isotopomer.py` but require specific implementation based on experimental details[cite: 3].

## Contributing

(Optional: Add guidelines for contributions if the repository is open for collaboration).

## License

(Optional: Specify the license under which the code is distributed, e.g., MIT, GPL, Apache 2.0).

