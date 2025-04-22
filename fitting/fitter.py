# fitting/fitter.py
# -------------------------------------------------------------------------
# Updated 2025-04-21 - Incorporates fixes for ID parsing and features
# from user-provided script header. Corrected fit_data signature.
# -------------------------------------------------------------------------

import collections
import time
import warnings
import re # Ensure re is imported
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Project-local imports
try:
    from models.metabolism import MetabolicModel
    from models.kinetics import KineticModel
    from analysis.isotopomer import IsotopomerHandler
    from utils.helpers import calculate_residuals
except ImportError as e:
    raise ImportError(f"Error importing fitter dependencies: {e}") from e

# Helper – Jacobian sparsity pattern (optional speed-up)
def _build_jacobian_sparsity(n_residuals: int, n_params: int):
    """ Builds a default dense sparsity pattern. """
    try:
        import scipy.sparse as sp
        # Return a structure SciPy can understand, e.g., array, list of lists, or sparse matrix
        # Using None often defaults to dense or SciPy's estimation
        # return sp.csr_matrix(np.ones((n_residuals, n_params), dtype=bool))
        return None # Let least_squares handle it unless a specific pattern is known
    except ImportError:
        # warnings.warn("scipy.sparse not found, cannot use Jacobian sparsity.")
        return None

# Main class
class DataFitter:
    """
    Fits experimental concentration + isotopic-label data to the metabolic model
    using SciPy's least-squares optimiser (with multiple random restarts).

    Key features:
        - log-parameter transform
        - residual σ floor
        - dynamic-pool integrity check
        - multi-start optimization
        - Optional Jacobian sparsity support
    """

    def __init__(
        self,
        metabolic_model: MetabolicModel,
        kinetic_model: KineticModel,
        isotopomer_handler: IsotopomerHandler,
        data: dict,
        parameters_info: dict,
        sigma_floor: float = 0.05,
        n_restarts: int = 1
    ):
        # (Constructor code remains the same as the previous version you provided)
        # ------------- Basic Validation -------------
        if not isinstance(metabolic_model, MetabolicModel): raise TypeError("metabolic_model must be MetabolicModel")
        if not isinstance(kinetic_model, KineticModel): raise TypeError("kinetic_model must be KineticModel")
        if not isinstance(isotopomer_handler, IsotopomerHandler): raise TypeError("isotopomer_handler must be IsotopomerHandler")
        required = ['time', 'measured_variables', 'measurements', 'errors']
        if not all(k in data for k in required): raise ValueError(f"Input 'data' missing keys {required}")
        if not isinstance(parameters_info, dict) or not parameters_info: raise ValueError("parameters_info is empty")
        # ------------- Store Raw Data & Model Components -------------
        self.metabolic_model = metabolic_model
        self.kinetic_model = kinetic_model
        self.isotopomer_handler = isotopomer_handler
        self.data_time = np.asarray(data['time'], dtype=float)
        self.data_measured_variables = data['measured_variables']
        self.data_measurements = np.asarray(data['measurements'], dtype=float)
        self.data_errors = np.asarray(data['errors'], dtype=float)
        # ------------- Residual Weight Floor -------------
        self.sigma_floor_relative = float(sigma_floor)
        if self.data_errors.shape != self.data_measurements.shape: raise ValueError("Shape mismatch: measurements vs errors")
        if len(self.data_time) != self.data_measurements.shape[0]: raise ValueError("Time points mismatch with measurements rows")
        if len(self.data_measured_variables) != self.data_measurements.shape[1]: raise ValueError("Measured variables mismatch with measurements columns")
        # ------------- Parameters (Setup for Log-Transform) -------------
        self.parameters_info = collections.OrderedDict(sorted(parameters_info.items()))
        self.parameter_ids = list(self.parameters_info.keys())
        n_params = len(self.parameter_ids)
        raw_initial_values = np.zeros(n_params); raw_lower_bounds = np.zeros(n_params); raw_upper_bounds = np.zeros(n_params)
        for i, (pid, info) in enumerate(self.parameters_info.items()):
            if not all(k in info for k in ['initial_value', 'lower_bound', 'upper_bound']): raise ValueError(f"Parameter '{pid}' missing keys.")
            lb = float(info['lower_bound']); ub = float(info['upper_bound']); init_val = float(info['initial_value'])
            if lb <= 0 or ub <= 0:
                warnings.warn(f"Param '{pid}' non-positive bounds [{lb}, {ub}]. Adjusting lower>1e-9.", RuntimeWarning)
                lb = max(lb, 1e-9); ub = max(ub, 1e-9)
            if init_val <= 0: warnings.warn(f"Param '{pid}' non-positive initial {init_val}. Adjusting to LB {lb}.", RuntimeWarning); init_val = lb
            if not (lb <= init_val <= ub): warnings.warn(f"Initial {init_val} for '{pid}' outside bounds [{lb}, {ub}]. Clamping.", RuntimeWarning); init_val = np.clip(init_val, lb, ub)
            raw_initial_values[i] = init_val; raw_lower_bounds[i] = lb; raw_upper_bounds[i] = ub
        self.raw_parameter_bounds = (raw_lower_bounds, raw_upper_bounds)
        self.raw_initial_parameter_values = raw_initial_values
        self.log_initial_parameter_values = np.log(self.raw_initial_parameter_values)
        self.log_parameter_bounds = (np.log(raw_lower_bounds), np.log(raw_upper_bounds))
        # ------------- Model Bookkeeping -------------
        self.dynamic_metabolite_ids = self.metabolic_model.get_metabolite_ids(non_constant=True)
        if not self.dynamic_metabolite_ids: raise ValueError("Model has no dynamic metabolites defined")
        # Corrected Integrity Check
        all_model_met_ids = set(self.metabolic_model.metabolites.keys()); miss = []
        for var_name in self.data_measured_variables:
            parts = var_name.split('_'); matched_met_id = None
            for i in range(1, len(parts) + 1):
                potential_met_id = "_".join(parts[:i])
                if potential_met_id in all_model_met_ids: matched_met_id = potential_met_id
            if matched_met_id is None: warnings.warn(f"Cannot map variable '{var_name}'")
            elif matched_met_id not in self.dynamic_metabolite_ids: miss.append(var_name)
        if miss: raise RuntimeError(f"Measured variables refer to non-dynamic metabolites: {miss}.")
        # End Corrected Integrity Check
        self.initial_state_vector = self.metabolic_model.get_initial_state_vector(self.dynamic_metabolite_ids)
        self.S_matrix, self.S_metabolite_ids, self.S_reaction_ids = self.metabolic_model.build_stoichiometric_matrix(metabolite_ids=self.dynamic_metabolite_ids)
        if self.S_metabolite_ids != self.dynamic_metabolite_ids: warnings.warn("Order mismatch: dynamic_metabolite_ids vs S_metabolite_ids.")
        self.constant_metabolite_ids = self.metabolic_model.get_metabolite_ids(constant=True)
        self.constant_concentrations = self.metabolic_model.get_initial_concentrations(self.constant_metabolite_ids)
        self.constant_labeling = {met: self.metabolic_model.metabolites[met]['initial_labeling'] for met in self.constant_metabolite_ids}
        self._map_measured_variables_to_state()
        # Optimisation settings
        self.n_restarts = int(max(1, n_restarts))
        # User feedback
        print("DataFitter initialized:"); print(f"  Dynamic metabolites : {self.dynamic_metabolite_ids}"); print(f"  Parameters to fit   : {self.parameter_ids}"); print(f"  Sigma floor (rel.)  : {self.sigma_floor_relative:.3g}"); print(f"  Multi-start trials: {self.n_restarts}")


    # (_map_measured_variables_to_state remains the same)
    def _map_measured_variables_to_state(self):
        self.measured_data_indices = []
        state_index_map = {}
        current_state_index = 0
        for met_id in self.dynamic_metabolite_ids:
             state_index_map[met_id] = current_state_index
             n_states = self.metabolic_model.get_state_vector_size(met_id)
             current_state_index += n_states
        all_model_met_ids = set(self.metabolic_model.metabolites.keys())
        for var_name in self.data_measured_variables:
            parts = var_name.split('_'); matched_met_id = None; prefix_len = 0
            for i in range(1, len(parts) + 1):
                pid = "_".join(parts[:i])
                if pid in all_model_met_ids: matched_met_id = pid; prefix_len = i
            if matched_met_id is None: raise ValueError(f"Cannot map variable '{var_name}'")
            if matched_met_id not in self.dynamic_metabolite_ids: raise RuntimeError(f"Variable '{var_name}' metabolite '{matched_met_id}' is constant")
            met_start = state_index_map[matched_met_id]
            carb_count = self.metabolic_model.get_metabolite_carbon_count(matched_met_id)
            suffix = parts[prefix_len:]
            if len(suffix) == 1 and suffix[0].lower() == 'total': self.measured_data_indices.append(met_start)
            elif len(suffix) == 2 and suffix[1].lower() == 'fe':
                 cm = re.match(r'C(\d+)', suffix[0], re.IGNORECASE)
                 if not cm: raise ValueError(f"Cannot parse carbon index in '{var_name}'")
                 c_idx = int(cm.group(1))
                 if not (1 <= c_idx <= carb_count): raise ValueError(f"Carbon index out of range in '{var_name}'")
                 self.measured_data_indices.append(met_start + c_idx)
            else: raise ValueError(f"Unknown measured variable format '{var_name}'")
        if len(self.measured_data_indices) != len(self.data_measured_variables): raise RuntimeError("Internal mapping error")
        print(f"Successfully mapped measured variables to state vector indices.")

    # (_map_parameters_to_reaction_dicts remains the same)
    def _map_parameters_to_reaction_dicts(self, parameter_vector):
        param_values = dict(zip(self.parameter_ids, parameter_vector))
        reaction_params = collections.defaultdict(dict)
        sorted_rxn = sorted(self.S_reaction_ids, key=len, reverse=True)
        for pid, val in param_values.items():
            for rxn_id in sorted_rxn:
                prefix = rxn_id + '_'
                if pid.startswith(prefix):
                    kinetic_param_name = pid[len(prefix):]
                    if kinetic_param_name: reaction_params[rxn_id][kinetic_param_name] = val
                    break
        return reaction_params

    # (_unpack_state_vector remains the same)
    def _unpack_state_vector(self, y):
        dynamic_concs = {}; current_labeling = {}; idx = 0
        for met_id in self.dynamic_metabolite_ids:
             n = self.metabolic_model.get_state_vector_size(met_id)
             met_state = y[idx : idx + n]; dynamic_concs[met_id] = met_state[0]; current_labeling[met_id] = met_state[1:]; idx += n
        return dynamic_concs, current_labeling

    # (_define_ode_system remains the same)
    def _define_ode_system(self, t, y, parameter_vector):
        try:
            dyn_conc, curr_label = self._unpack_state_vector(y)
            all_conc = {**dyn_conc, **self.constant_concentrations}; all_label = {**curr_label, **self.constant_labeling}
            reaction_params = self._map_parameters_to_reaction_dicts(parameter_vector)
            reaction_rates = np.zeros(len(self.S_reaction_ids))
            for i, rxn_id in enumerate(self.S_reaction_ids):
                info = self.metabolic_model.reactions[rxn_id]
                reactant_c = {m: max(0, all_conc.get(m, 0)) for m in info['reactants']}
                product_c = ({m: max(0, all_conc.get(m, 0)) for m in info['products']} if info['reversible'] else None)
                params = reaction_params.get(rxn_id, {})
                reaction_rates[i] = self.kinetic_model.calculate_reaction_rate(info['kinetic_model'], params, reactant_c, product_c)
            dCdt = self.S_matrix @ reaction_rates
            prod_fes_all = self.isotopomer_handler.calculate_labeling_fluxes_all(all_label)
            if len(prod_fes_all) != len(self.S_reaction_ids): raise RuntimeError("Label flux length mismatch")
            sum_flux_in_label = collections.defaultdict(lambda: np.zeros(0)); dFE_dt_dict = collections.defaultdict(lambda: np.zeros(0))
            for met in self.dynamic_metabolite_ids: nC = self.metabolic_model.get_metabolite_carbon_count(met); sum_flux_in_label[met] = np.zeros(nC) if nC > 0 else np.zeros(0)
            for j, rxn_id in enumerate(self.S_reaction_ids):
                 info = self.metabolic_model.reactions[rxn_id]; rate = reaction_rates[j]
                 if rate < 0: continue
                 prod_fes = prod_fes_all[j]
                 for met_id, stoich in info['products'].items():
                     if met_id in sum_flux_in_label and met_id in prod_fes:
                         if len(prod_fes[met_id]) == len(sum_flux_in_label[met_id]): sum_flux_in_label[met_id] += np.array(prod_fes[met_id]) * rate * stoich
            for i, met in enumerate(self.dynamic_metabolite_ids):
                conc = dyn_conc[met]; fe = curr_label[met]
                if fe.size == 0: continue
                total_out = sum(abs(self.S_matrix[i, j]) * reaction_rates[j] for j, r in enumerate(self.S_reaction_ids) if self.S_matrix[i, j] < 0)
                if conc > 1e-12: dfe = (sum_flux_in_label[met] - fe * total_out) / conc
                else: dfe = np.zeros_like(fe)
                dfe[(fe <= 1e-9) & (dfe < 0)] = 0; dfe[(fe >= 1.0-1e-9) & (dfe > 0)] = 0
                dFE_dt_dict[met] = dfe
            dydt = np.zeros_like(y); idx = 0
            for i, met in enumerate(self.dynamic_metabolite_ids):
                 n = self.metabolic_model.get_state_vector_size(met)
                 dydt[idx] = dCdt[i]
                 if n > 1: dydt[idx + 1: idx + n] = dFE_dt_dict[met]
                 idx += n
            if not np.all(np.isfinite(dydt)): warnings.warn(f"NaN/Inf in derivatives at t={t}", RuntimeWarning); return np.full_like(y, np.nan)
            return dydt
        except Exception as exc: print(f"Error derivatives t={t}: {exc}"); return np.full_like(y, np.nan)


    # (_simulate remains the same)
    def _simulate(self, log_param_vec):
        raw_param_vec = np.exp(log_param_vec)
        sol = solve_ivp(fun=self._define_ode_system, t_span=(self.data_time[0], self.data_time[-1]), y0=self.initial_state_vector, t_eval=self.data_time, args=(raw_param_vec,), method='BDF', rtol=1e-4, atol=1e-6, min_step=1e-6)
        if sol.status != 0 or not np.all(np.isfinite(sol.y)): return None
        sim_output = sol.y[self.measured_data_indices, :].T
        return sim_output

    # (_objective_residuals remains the same)
    def _objective_residuals(self, log_param_vec):
        sim_out = self._simulate(log_param_vec)
        if sim_out is None: return np.full(self.data_measurements.size, 1e10)
        eff_sigma = np.maximum(self.data_errors, self.sigma_floor_relative * np.abs(self.data_measurements))
        eff_sigma[eff_sigma < 1e-9] = 1e-9
        residuals = calculate_residuals(self.data_measurements, sim_out, eff_sigma)
        return residuals.ravel()

    # --- Modified fit_data ---
    def fit_data(self, method='least_squares', verbose=1, **kwargs):
        """
        Fit parameters with multi-start Least-Squares using log-transformation.
        Handles 'method' argument correctly.
        """
        # Method argument selects high-level strategy (e.g., which scipy function or custom method)
        # Other kwargs are passed down *if appropriate* for the chosen strategy.

        if method != 'least_squares':
             # Placeholder for potentially adding back Simplex, SA etc. later
             # These would likely need a scalar objective function, e.g., sum of squares
             raise NotImplementedError(f"Fitting method '{method}' is not currently implemented. Use 'least_squares'.")

        # --- Proceed with least_squares using log-transform ---
        print(f"\n--- Starting Parameter Fitting ({self.n_restarts} trial(s), Isotopomer Model) using '{method}' ---")
        start_time = time.time()

        log_lb, log_ub = self.log_parameter_bounds

        # Default settings specific to scipy.optimize.least_squares
        ls_kwargs = {'jac':'3-point', 'method': 'trf', 'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5}
        # Update with any *other* kwargs passed, EXCLUDING the 'method' meant for fit_data itself
        kwargs_for_scipy = {k: v for k, v in kwargs.items() if k not in ['method', 'verbose']}
        ls_kwargs.update(kwargs_for_scipy)

        # Jacobian sparsity (optional)
        J_sp = ls_kwargs.pop('jac_sparsity', None) # Allow override via kwargs
        if J_sp is None: J_sp = _build_jacobian_sparsity(self.data_measurements.size, len(self.parameter_ids))
        if J_sp is not None: ls_kwargs['jac_sparsity'] = J_sp

        best_cost = np.inf
        best_result_obj = None
        best_raw_params = None

        for trial in range(self.n_restarts):
            print(f"--- Starting Trial {trial + 1}/{self.n_restarts} ---")
            if trial == 0:
                log_x0 = self.log_initial_parameter_values.copy()
            else:
                log_x0 = np.random.uniform(log_lb, log_ub)
                log_x0 = np.clip(log_x0, log_lb, log_ub)

            try:
                res = least_squares(
                    fun=self._objective_residuals,
                    x0=log_x0,
                    bounds=(log_lb, log_ub),
                    verbose=verbose, # Use verbose passed to fit_data
                    **ls_kwargs # Pass the filtered kwargs
                )

                final_raw_p = np.exp(res.x)
                hit_bounds = [pid for pid, p, low, high in zip(self.parameter_ids, final_raw_p, self.raw_parameter_bounds[0], self.raw_parameter_bounds[1]) if np.isclose(p, low, rtol=1e-3, atol=1e-9) or np.isclose(p, high, rtol=1e-3, atol=1e-9)]
                if hit_bounds: print(f"  [Trial {trial+1}] Parameters at bounds: {hit_bounds}")

                if res.success and res.cost < best_cost:
                    best_cost = res.cost; best_result_obj = res; best_raw_params = final_raw_p
                    print(f"  [Trial {trial+1}] New best cost: {res.cost:.4g}")
                elif res.success: print(f"  [Trial {trial+1}] Cost {res.cost:.4g} (not best)")
                else: print(f"  [Trial {trial+1}] Failed. Status: {res.status}, Msg: {res.message}")

            except Exception as e: print(f"  [Trial {trial+1}] Error: {e}")

        end_time = time.time()
        print(f"\n--- Multi-start Fitting completed in {end_time - start_time:.2f} seconds ---")

        if best_raw_params is None:
            print("Optimization failed to find a solution across all trials.")
            self.simulation_function_fitted = None
            return None, best_result_obj

        fitted_parameters_dict = collections.OrderedDict(zip(self.parameter_ids, best_raw_params))
        print("--- Best Fit Found ---")
        for k, v in fitted_parameters_dict.items(): print(f"  {k:<20s} : {v:.5g}")
        print(f"Final Cost (0.5 * sum(residuals^2)): {best_cost:.4g}")
        if best_result_obj: print(f"Status: {best_result_obj.status}, Msg: {best_result_obj.message}")

        # Store callable simulation function with best fitted params (takes raw params)
        self.simulation_function_fitted = lambda p_vec=best_raw_params: self._simulate(np.log(np.asarray(p_vec)))

        return fitted_parameters_dict, best_result_obj


    # (simulation_function_for_plot property remains the same)
    @property
    def simulation_function_for_plot(self):
        """ Returns a callable simulation function using best fitted parameters (takes raw params)."""
        func = getattr(self, 'simulation_function_fitted', None)
        # The lambda stored already takes raw params, so just return it.
        return func


    # (_dump_flux_balance remains the same)
    def _dump_flux_balance(self, raw_param_vec):
        pass # Implement if needed for debugging