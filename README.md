# CWAVE Recreation in Python (with Isotopomer Tracking)

## Description
This project is a Python-based recreation of the core functionalities described for the CWAVE software, originally used for designing and analyzing 13C labeling studies, particularly with MRS data. This version attempts to simulate both total metabolite concentrations and positional 13C fractional enrichment (FE) over time.

## Features Implemented
* **Metabolic Model Definition**: Define metabolites (with carbon counts, initial labeling) and reactions (with stoichiometry, kinetics, and atom mappings).
* **Kinetic Modeling**: Use predefined kinetic rate laws (Mass Action, Michaelis-Menten) or add custom ones.
* **Isotopomer Tracking**: Simulates the time course of positional fractional 13C enrichment based on atom mappings using an expanded ODE system.
* **Parameter Fitting**: Estimates kinetic parameters by fitting the model simulation (including labeling) to experimental time-course data using least-squares optimization.
* **Statistical Analysis**: Performs Monte Carlo simulations to estimate the uncertainty (confidence intervals) of the fitted parameters based on experimental error.
* **Data Handling**: Loads experimental data (including labeling) from CSV and initial parameters/bounds from JSON.
* **Visualization**: Plots experimental data, the model fit (for measured variables), and parameter confidence intervals.
* **Results Saving**: Saves fitted parameters and Monte Carlo analysis summary to JSON files.

## Project Structure
```
CWAVE/
├── cwave_app.py        # Main application script
├── models/
│   ├── metabolism.py   # Metabolic model definition (metabolites, reactions, atom maps)
│   └── kinetics.py     # Kinetic rate law definitions
├── fitting/
│   ├── fitter.py       # Data fitting class (incl. ODE simulation)
│   └── methods.py      # Optimization algorithms (Simplex, SA - optional)
├── analysis/
│   ├── stats.py        # Monte Carlo uncertainty analysis
│   └── isotopomer.py   # Isotopomer labeling flux calculations
├── data/
│   ├── example_labeling_data.csv # Example experimental data (conc + FE)
│   └── example_model_params.json # Example initial parameters and bounds
├── utils/
│   ├── helpers.py      # Data loading, saving, residuals
│   └── plotting.py     # Results plotting functions
├── results/            # Default output directory for plots and results files
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
## Dependencies
* Python 3.7+
* NumPy
* SciPy (for ODE solving and optimization)
* Pandas (for data loading)
* Matplotlib (for plotting)

## Installation
1.  Ensure you have Python 3 installed.
2.  Clone or download this project code into your `CWAVE` directory.
3.  Create a virtual environment (highly recommended):
    ```bash
    # Navigate to the directory containing the CWAVE folder
    python -m venv cwave_env
    # Activate the environment
    # Windows PowerShell:
    .\cwave_env\Scripts\Activate.ps1
    # Windows CMD:
    .\cwave_env\Scripts\activate.bat
    # Linux/macOS:
    source cwave_env/bin/activate
    ```
4.  Install the required packages (while the environment is active):
    ```bash
    pip install -r CWAVE/requirements.txt
    ```
    *(Note: You'll need to create the requirements.txt file - see next step)*

## Usage
1.  **Define Your Model**: Modify the `define_example_model` function inside `CWAVE/cwave_app.py` to represent your specific metabolic system. This includes:
    * Adding metabolites with correct `carbon_count` and `initial_labeling`.
    * Adding reactions with correct `reactants`, `products`, `kinetic_model_id`, and crucially, accurate `atom_mapping_str`.
2.  **Prepare Data**: Create a CSV file (like `CWAVE/data/example_labeling_data.csv`) containing your experimental time course data. It must include:
    * A `Time` column.
    * Columns for each measured variable (e.g., `GLC_Total`, `GLN_C4_FE`). The names must follow the format `MetaboliteID_Total` or `MetaboliteID_C#_FE`.
    * Columns for the standard deviation of each measurement, named with an `_SD` suffix (e.g., `GLC_Total_SD`, `GLN_C4_FE_SD`).
3.  **Prepare Parameters**: Create a JSON file (like `CWAVE/data/example_model_params.json`) defining the parameters to be fitted. For each parameter (named `ReactionID_ParameterName`):
    * Provide an `initial_value`.
    * Provide `lower_bound` and `upper_bound`.
4.  **Run the Application**: Execute the main script from your terminal (make sure your virtual environment is active):
    ```bash
    python CWAVE/cwave_app.py
    ```
5.  **Check Results**: Output plots and JSON result files will be saved in the `CWAVE/results/` directory.

## Important Notes & Limitations
* **Example Model**: The model defined in `cwave_app.py` is a simplified example and must be replaced with your actual biological system.
* **Atom Mappings**: The accuracy of the isotopomer simulation depends entirely on the correctness of the `atom_mapping_str` provided for each reaction. Use `'?'` for untracked carbons.
* **ODE Solver**: The simulation uses `scipy.integrate.solve_ivp`. Performance and accuracy can depend on solver choice (`method='LSODA'`) and tolerances (`rtol`, `atol` in `fitter.py`). Stiff systems are common and may require tuning.
* **Isotopomer Framework**: This implementation uses positional fractional enrichment. More advanced frameworks (EMU, Cumomer) might be necessary for certain analyses or data types (like complex MS fragmentation) but are significantly more complex to implement.
* **Placeholders**: Functions related to natural abundance correction for experimental MS data (`correct_natural_abundance_MFA`) and NMR resonance overlap (`handle_resonance_overlap`) in `analysis/isotopomer.py` are placeholders requiring specific implementation based on experimental details.
* **Reversible Reactions**: The handling of label flux in reversible reactions within the `dFE/dt` calculation might need refinement for rigorous accuracy, especially under conditions where net flux is near zero or rapidly changing direction.
