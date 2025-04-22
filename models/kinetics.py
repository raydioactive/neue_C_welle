import numpy as np
import collections

class KineticModel:
    """
    Defines and manages the kinetic models (rate laws) for reactions.

    This class stores different types of rate laws (e.g., Mass Action,
    Michaelis-Menten) and provides methods to calculate reaction rates
    given metabolite concentrations and kinetic parameters.
    """
    def __init__(self):
        """
        Initializes the KineticModel with a dictionary to store predefined
        and potentially custom rate law functions.
        """
        self.rate_laws = collections.OrderedDict() # {model_id: {'function': callable, 'parameter_names': list}}
        self._add_predefined_rate_laws()

    def _add_predefined_rate_laws(self):
        """Adds common kinetic rate laws to the instance."""
        self.add_rate_law(
            model_id="MassAction",
            rate_function=self._mass_action_rate,
            parameter_names=['k'] # Forward rate constant
        )
        self.add_rate_law(
            model_id="MassActionReversible",
            rate_function=self._mass_action_reversible_rate,
            parameter_names=['k_fwd', 'k_rev'] # Forward and reverse rate constants
            # Alternative: ['k_fwd', 'K_eq'] where K_eq = k_fwd / k_rev
        )
        self.add_rate_law(
            model_id="MichaelisMenten",
            rate_function=self._michaelis_menten_rate,
            parameter_names=['Vmax', 'Km_substrate'] # Vmax and Km for the primary substrate
            # Note: Assumes one substrate for simplicity here. More complex MM might be needed.
        )
        self.add_rate_law(
            model_id="MichaelisMentenReversible",
            rate_function=self._michaelis_menten_reversible_rate,
            parameter_names=['Vmax_fwd', 'Km_substrate', 'Vmax_rev', 'Km_product'] # Haldane relationship implies dependency
            # See https://en.wikipedia.org/wiki/Enzyme_kinetics#Reversible_reactions
        )
        self.add_rate_law(
            model_id="ConstantFlux",
            rate_function=self._constant_flux_rate,
            parameter_names=['flux_value'] # A constant flux value
        )
        # Add other common rate laws as needed (e.g., Hill kinetics, inhibition models)

    def add_rate_law(self, model_id, rate_function, parameter_names):
        """
        Adds a kinetic model (rate law) to the collection.

        Args:
            model_id (str): The unique identifier for the kinetic model (e.g., 'MassAction').
            rate_function (callable): A function that calculates the reaction rate.
                                      It must accept 'parameters' (dict) and 'concentrations' (dict)
                                      as arguments, corresponding to the specific reaction.
            parameter_names (list): A list of parameter names (str) expected by this rate law
                                    (e.g., ['k'], ['Vmax', 'Km_substrate']).

        Raises:
            ValueError: If the model_id already exists.
            TypeError: If rate_function is not callable or parameter_names is not a list.
        """
        if model_id in self.rate_laws:
            raise ValueError(f"Kinetic model ID '{model_id}' already exists.")
        if not callable(rate_function):
            raise TypeError(f"rate_function for '{model_id}' must be callable.")
        if not isinstance(parameter_names, list):
             raise TypeError(f"parameter_names for '{model_id}' must be a list.")

        self.rate_laws[model_id] = {
            'function': rate_function,
            'parameter_names': parameter_names
        }
        print(f"Added rate law: {model_id} (Parameters: {parameter_names})")

    def get_rate_law_details(self, model_id):
        """
        Retrieves the details (function, parameter names) for a kinetic model.

        Args:
            model_id (str): The identifier of the kinetic model.

        Returns:
            dict: A dictionary containing 'function' and 'parameter_names'.

        Raises:
            ValueError: If the model_id is not found.
        """
        if model_id not in self.rate_laws:
            raise ValueError(f"Kinetic model ID '{model_id}' not found.")
        return self.rate_laws[model_id]

    def get_required_parameters(self, model_id):
        """
        Gets the list of parameter names required for a specific rate law.

        Args:
            model_id (str): The identifier of the kinetic model.

        Returns:
            list: List of required parameter names (e.g., ['Vmax', 'Km_substrate']).

        Raises:
            ValueError: If the model_id is not found.
        """
        details = self.get_rate_law_details(model_id)
        return details['parameter_names']

    def calculate_reaction_rate(self, model_id, reaction_parameters, reactant_concentrations, product_concentrations=None):
        """
        Calculates the net rate of a reaction using a specified kinetic model.

        Args:
            model_id (str): The identifier of the kinetic model to use (e.g., 'MassAction').
            reaction_parameters (dict): Dictionary of kinetic parameter values for this specific reaction
                                       (e.g., {'k': 0.1} or {'Vmax': 1.0, 'Km_substrate': 0.5}).
                                       Keys must match the expected parameter_names for the model_id.
            reactant_concentrations (dict): Dictionary of concentrations for the reactants involved
                                           in this specific reaction (e.g., {'GLC': 10.0, 'ATP': 2.0}).
            product_concentrations (dict, optional): Dictionary of concentrations for the products.
                                                    Required for reversible reactions. Defaults to None.

        Returns:
            float: The calculated net reaction rate.

        Raises:
            ValueError: If the model_id is not found or if required parameters/concentrations are missing.
            Exception: Propagates exceptions from the rate law function itself (e.g., math errors).
        """
        details = self.get_rate_law_details(model_id)
        rate_function = details['function']
        required_params = details['parameter_names']

        # Check if all required parameters are provided for this reaction
        missing_params = [p for p in required_params if p not in reaction_parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters {missing_params} for kinetic model '{model_id}' in reaction parameters: {reaction_parameters}")

        # Call the specific rate law function
        try:
            # Pass None for products if not needed by the rate law (checked within the law)
            rate = rate_function(reaction_parameters, reactant_concentrations, product_concentrations)
            # Ensure rate is non-negative for irreversible reactions (can happen with complex reversible forms)
            # Note: Reversible forms should inherently handle net direction.
            # if not model_id.endswith("Reversible") and rate < 0:
            #     print(f"Warning: Calculated negative rate ({rate}) for irreversible model '{model_id}'. Clamping to zero.")
            #     rate = 0.0
            return rate
        except Exception as e:
            print(f"Error calculating rate for model '{model_id}' with params {reaction_parameters} and concs {reactant_concentrations}, {product_concentrations}")
            raise e # Re-raise the exception

    # --- Specific Rate Law Implementations ---

    def _mass_action_rate(self, parameters, reactants, products=None):
        """Rate = k * [Reactant1]^stoich1 * [Reactant2]^stoich2 * ..."""
        rate = parameters['k']
        if not reactants: # Handle case with no reactants (e.g., constant influx)
             return rate
        for met_id, concentration in reactants.items():
            # Assumes stoichiometry is handled by how reactants dict is passed (should contain only reactants needed)
            # If stoichiometry needs to be explicit here, the MetabolicModel needs to pass it.
            if concentration < 0: concentration = 0 # Concentrations cannot be negative
            rate *= concentration
        return rate

    def _mass_action_reversible_rate(self, parameters, reactants, products):
        """Rate = k_fwd * Product([Reactants]) - k_rev * Product([Products])"""
        if products is None:
            raise ValueError("Product concentrations required for reversible Mass Action.")

        forward_rate = parameters['k_fwd']
        if reactants:
            for met_id, concentration in reactants.items():
                 if concentration < 0: concentration = 0
                 forward_rate *= concentration
        else: # Handle case with no reactants (e.g., A <=> B+C where A is constant)
             pass # Forward rate depends only on k_fwd

        reverse_rate = parameters['k_rev']
        if products:
            for met_id, concentration in products.items():
                 if concentration < 0: concentration = 0
                 reverse_rate *= concentration
        else: # Handle case with no products (e.g., A+B <=> C where C is constant)
            pass # Reverse rate depends only on k_rev

        return forward_rate - reverse_rate

    def _michaelis_menten_rate(self, parameters, reactants, products=None):
        """Rate = Vmax * [S] / (Km_substrate + [S])"""
        # Assumes the first reactant is the primary substrate 'S'
        if not reactants:
            raise ValueError("Reactant concentration required for Michaelis-Menten.")

        substrate_id = list(reactants.keys())[0] # Get the first reactant ID
        substrate_conc = reactants[substrate_id]
        if substrate_conc < 0: substrate_conc = 0

        vmax = parameters['Vmax']
        km = parameters['Km_substrate']

        if km + substrate_conc <= 1e-9: # Avoid division by zero or near-zero
            return 0.0
        return vmax * substrate_conc / (km + substrate_conc)

    def _michaelis_menten_reversible_rate(self, parameters, reactants, products):
        """Rate = (Vmax_fwd*[S]/Km_s - Vmax_rev*[P]/Km_p) / (1 + [S]/Km_s + [P]/Km_p)"""
        if not reactants or not products:
            raise ValueError("Reactant and product concentrations required for reversible Michaelis-Menten.")

        # Assumes first reactant is S, first product is P
        substrate_id = list(reactants.keys())[0]
        product_id = list(products.keys())[0]
        s_conc = reactants[substrate_id]
        p_conc = products[product_id]
        if s_conc < 0: s_conc = 0
        if p_conc < 0: p_conc = 0

        vmax_fwd = parameters['Vmax_fwd']
        km_s = parameters['Km_substrate']
        vmax_rev = parameters['Vmax_rev']
        km_p = parameters['Km_product']

        # Check for valid Km values
        if km_s <= 1e-9 or km_p <= 1e-9:
             print(f"Warning: Near-zero Km detected (Km_s={km_s}, Km_p={km_p}). Rate calculation might be unstable.")
             # Handle potential division by zero gracefully, maybe return 0 or based on limits
             if km_s <= 1e-9 and km_p <= 1e-9: return 0.0 # Or some other logic
             elif km_s <= 1e-9: km_s = 1e-9
             elif km_p <= 1e-9: km_p = 1e-9


        numerator = (vmax_fwd * s_conc / km_s) - (vmax_rev * p_conc / km_p)
        denominator = 1.0 + (s_conc / km_s) + (p_conc / km_p)

        if denominator <= 1e-9:
            return 0.0 # Avoid division by zero

        return numerator / denominator

    def _constant_flux_rate(self, parameters, reactants=None, products=None):
        """Rate = flux_value (constant)"""
        return parameters['flux_value']

