import collections
from collections import defaultdict # Ensure this is imported
import numpy as np
import re # For parsing atom mappings
import warnings # Ensure this is imported

class MetabolicModel:
    """
    Represents the metabolic network model, updated to include information
    needed for isotopomer tracking (carbon counts, atom mappings).

    Stores metabolites, reactions, stoichiometry, compartments, carbon counts,
    and atom mappings.
    """
    def __init__(self):
        """
        Initializes the MetabolicModel.
        """
        self.metabolites = collections.OrderedDict()
        # {metabolite_id: {'name': str, 'compartment': str, 'initial_concentration': float,
        #                  'initial_labeling': np.ndarray, # Fractional enrichment per carbon
        #                  'is_substrate': bool, 'is_constant': bool, 'carbon_count': int}}
        self.reactions = collections.OrderedDict()
        # {reaction_id: {'name': str, 'reactants': {met_id: stoich}, 'products': {met_id: stoich},
        #                'kinetic_model': str, 'reversible': bool,
        #                'atom_mapping': dict (parsed by this class's parser),
        #                'atom_mapping_str': str (original string for IsotopomerHandler)}}
        self.parameters = collections.OrderedDict() # Still potentially useful, but managed mainly by KineticModel now

    def add_metabolite(self, metabolite_id, carbon_count, name=None, compartment='cytosol',
                       initial_concentration=0.0, initial_labeling=None,
                       is_substrate=False, is_constant=False):
        """
        Adds a metabolite to the model, including carbon count and initial labeling.
        # ... (docstring remains the same) ...
        """
        if metabolite_id in self.metabolites:
            raise ValueError(f"Metabolite ID '{metabolite_id}' already exists.")
        if not isinstance(carbon_count, int) or carbon_count < 0:
            raise ValueError(f"Carbon count for '{metabolite_id}' must be a non-negative integer.")
        if initial_concentration < 0:
            raise ValueError(f"Initial concentration for '{metabolite_id}' cannot be negative.")

        # Handle initial labeling
        natural_abundance = 0.011 # Approximate
        if initial_labeling is None:
            labeling_array = np.full(carbon_count, natural_abundance, dtype=float) if carbon_count > 0 else np.array([], dtype=float)
        else:
            labeling_array = np.array(initial_labeling, dtype=float)
            if labeling_array.shape != (carbon_count,):
                raise ValueError(f"Length of initial_labeling ({len(labeling_array)}) must match carbon_count ({carbon_count}) for metabolite '{metabolite_id}'.")
            if np.any((labeling_array < 0) | (labeling_array > 1)):
                 raise ValueError(f"Initial labeling values for '{metabolite_id}' must be between 0 and 1.")

        self.metabolites[metabolite_id] = {
            "name": name if name else metabolite_id,
            "compartment": compartment,
            "carbon_count": carbon_count,
            "initial_concentration": float(initial_concentration),
            "initial_labeling": labeling_array, # Store as numpy array
            "is_substrate": bool(is_substrate),
            "is_constant": bool(is_constant),
        }
        print(f"Added metabolite: {metabolite_id} (C={carbon_count}, Initial Conc: {initial_concentration:.3g}, Constant: {is_constant})")


    def add_reaction(self, reaction_id, reactants, products, kinetic_model_id,
                     atom_mapping_str, reversible=False, name=None):
        """
        Adds a reaction, including atom mapping information.
        # ... (docstring remains the same) ...
        """
        # Store the original mapping string for the IsotopomerHandler
        if reaction_id in self.reactions:
            raise ValueError(f"Reaction ID '{reaction_id}' already exists.")

        # Validate reactants and products (stoichiometry and existence)
        for met_id, stoich in reactants.items():
            if met_id not in self.metabolites:
                raise ValueError(f"Reactant metabolite ID '{met_id}' not found in model for reaction '{reaction_id}'.")
            if not isinstance(stoich, (int, float)) or stoich <= 0:
                 raise ValueError(f"Reactant stoichiometry for '{met_id}' in reaction '{reaction_id}' must be a positive number.")
        for met_id, stoich in products.items():
             # Allow products to be empty for efflux reactions
            if met_id not in self.metabolites: # products dict is not empty here
                raise ValueError(f"Product metabolite ID '{met_id}' not found in model for reaction '{reaction_id}'.")
            if not isinstance(stoich, (int, float)) or stoich <= 0:
                 raise ValueError(f"Product stoichiometry for '{met_id}' in reaction '{reaction_id}' must be a positive number.")


        # Parse and validate atom mapping string using the model's internal parser
        try:
            # Refined check: Only raise error if RHS is not empty AND not just '?' when products dict is empty
            if "->" in atom_mapping_str:
                 rhs_part = atom_mapping_str.split("->", 1)[1].strip()
                 # Check if RHS defines actual metabolites using regex
                 contains_actual_product = re.search(r'[A-Za-z0-9_]+\[', rhs_part)
                 # Raise error only if RHS looks like it defines a metabolite, but products dict is empty
                 if contains_actual_product and not products:
                      raise ValueError(f"Reaction '{reaction_id}' has defined products (e.g., MET[...]) in atom mapping string RHS but not in the product dictionary.")
                 # Allow RHS to be non-empty (like '?') if products dict IS empty
                 # No error needed here for cases like "A[1] -> ?" with products={}

            # Proceed with parsing and validation
            parsed_mapping = self._parse_atom_mapping(atom_mapping_str, reactants, products)
            self._validate_atom_mapping(parsed_mapping, reactants, products) # Use the fully corrected validator here

        except ValueError as e:
            raise ValueError(f"Invalid atom mapping for reaction '{reaction_id}': {e}")

        self.reactions[reaction_id] = {
            "name": name if name else reaction_id,
            "reactants": reactants,
            "products": products,
            "kinetic_model": kinetic_model_id,
            "reversible": bool(reversible),
            "atom_mapping": parsed_mapping, # Store the parsed mapping dict
            "atom_mapping_str": atom_mapping_str # Store the original string
        }
        print(f"Added reaction: {reaction_id} ({self.get_reaction_equation(reaction_id)})")

    def _parse_atom_mapping(self, mapping_str, reactants, products):
        """Parses the atom mapping string into a structured dictionary."""
        parsed = {'reactants': {}, 'products': {}}
        if not mapping_str:
             if not reactants and not products: return parsed
             else: raise ValueError("Atom mapping string is empty but reactants/products are defined.")

        if "->" not in mapping_str:
             if not products:
                  reactant_part = mapping_str.strip(); product_part = ""
             else: raise ValueError(f"Atom mapping string '{mapping_str}' is missing '->' but products are defined.")
        else:
            reactant_part, product_part = mapping_str.split('->', 1)
            reactant_part = reactant_part.strip(); product_part = product_part.strip()

        # --- Try block for parsing ---
        try:
            # Parse reactants (same as before)
            if reactant_part:
                reactant_matches = re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]', reactant_part)
                if not reactant_matches and reactant_part: raise ValueError(f"Could not parse reactant part: '{reactant_part}'")
                for met_id, carbons_str in reactant_matches:
                    if met_id not in reactants: raise ValueError(f"Metabolite '{met_id}' in mapping string LHS is not listed as a reactant.")
                    try:
                        split_indices = [c.strip() for c in carbons_str.split(',') if c.strip()]
                        carbon_indices = [int(i) for i in split_indices]
                        if len(carbon_indices) != len(split_indices): raise ValueError("contains non-integer indices")
                    except ValueError as ve: raise ValueError(f"Invalid reactant indices '{carbons_str}' for '{met_id}'. Must be comma-separated integers. {ve}")
                    parsed['reactants'][met_id] = carbon_indices
            elif reactants: raise ValueError("Reactants defined but none found in mapping string LHS.")

            # Parse products - ** Corrected Logic **
            if product_part:
                if product_part == '?':
                    # Handle efflux case: RHS is exactly '?'
                    if products: raise ValueError("Mapping string RHS is '?' indicating efflux, but products dictionary is not empty.")
                    parsed['products'] = {} # Correctly set to empty and do nothing else for products
                else:
                    # Handle normal case: RHS is not '?' and not empty
                    product_matches = re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]', product_part)
                    if not product_matches: # Check if regex parsing failed
                        raise ValueError(f"Could not parse product part: '{product_part}'. Expected format like MET[...] or '?'.")

                    temp_prod_origins = defaultdict(list)
                    for met_id, mapping_info_str in product_matches:
                        if met_id not in products: raise ValueError(f"Metabolite '{met_id}' in mapping string RHS is not listed as a product.")
                        origin_list = [origin.strip() for origin in mapping_info_str.split(',') if origin.strip()]
                        if not origin_list: raise ValueError(f"Empty origin list found for product '{met_id}' in mapping.")
                        temp_prod_origins[met_id].extend(origin_list)
                    parsed['products'] = dict(temp_prod_origins)

            elif products: # product_part is empty, but products dict is not
                  raise ValueError("Products defined for reaction but RHS of mapping string is empty.")
            # else: product_part is empty and products is empty, parsed['products'] remains {} (correct)

        # --- End Try block ---
        except Exception as e:
            raise ValueError(f"Could not parse atom mapping string '{mapping_str}'. Error: {e}.") from e

        return parsed


    def _validate_atom_mapping(self, parsed_mapping, reactants, products):
        """Validates the parsed atom mapping against metabolite carbon counts."""
        # (This is the corrected version from the previous step)
        reactant_carbon_map = {} # Map 'MetIDIndex' -> (MetID, Index)
        total_reactant_carbons = 0

        # Check reactant side and build source map
        for met_id, carbon_indices in parsed_mapping.get('reactants', {}).items():
            if met_id not in self.metabolites: # Check metabolite exists
                 raise ValueError(f"Reactant metabolite '{met_id}' not found in model.")
            expected_count = self.metabolites[met_id]['carbon_count']
            if len(carbon_indices) != expected_count:
                raise ValueError(
                    f"Atom mapping for reactant '{met_id}' specifies {len(carbon_indices)} carbons, "
                    f"but metabolite has {expected_count}."
                )
            if not all(1 <= idx <= expected_count for idx in carbon_indices):
                 raise ValueError(
                     f"Atom mapping indices for reactant '{met_id}' are out of bounds "
                     f"(1 to {expected_count}). Found: {carbon_indices}"
                 )
            # Populate source map
            for r_idx in carbon_indices:
                 reactant_carbon_map[f"{met_id}{r_idx}"] = (met_id, r_idx)
                 total_reactant_carbons +=1

        # Check product side
        origin_counts = defaultdict(int) # Needs "from collections import defaultdict" at top
        total_product_carbons = 0
        product_carbons_defined = set() # Track product carbons defined in the mapping

        for met_id, origin_list in parsed_mapping.get('products', {}).items():
            if met_id not in self.metabolites: # Check metabolite exists
                 raise ValueError(f"Product metabolite '{met_id}' not found in model.")
            expected_prod_count = self.metabolites[met_id]['carbon_count']
            # If a product appears multiple times (e.g., A -> B + B), origin_list contains origins for ALL instances
            # We need to validate based on the total expected carbons for all instances of this product
            total_expected_prod_carbons = expected_prod_count * products[met_id] # Stoichiometry matters here
            if len(origin_list) != total_expected_prod_carbons:
                raise ValueError(
                    f"Atom mapping for product '{met_id}' specifies {len(origin_list)} total origins, "
                    f"but stoichiometry ({products[met_id]}) and carbon count ({expected_prod_count}) require {total_expected_prod_carbons}."
                )

            # Check if origins are valid reactant carbons or '?'
            product_pos_counter = 0
            for origin in origin_list:
                total_product_carbons += 1
                # Product position is harder to track simply when stoichiometry > 1
                # product_pos_counter += 1
                # product_carbons_defined.add(f"{met_id}{product_pos_counter}")

                if origin == '?':
                    continue # Untracked carbon is allowed
                if origin not in reactant_carbon_map:
                    # Check if 'origin' just looks like a number (positional index format)
                    if origin.isdigit():
                         # It's likely positional mapping was intended. Issue warning.
                         warnings.warn(f"Atom mapping origin '{origin}' for product '{met_id}' looks like a positional index "
                                       f"instead of an explicit source (e.g., 'ReactantIDCarbonIndex' like '{list(reactant_carbon_map.keys())[0] if reactant_carbon_map else 'N/A'}'). "
                                       f"Assuming positional mapping was intended, but consider revising the parser or mapping string format.",
                                       SyntaxWarning)
                         pass # Allow this format for now
                    else:
                         # If it's not '?' and not in the reactant map and not a digit, it's an error.
                         raise ValueError(
                             f"Atom mapping origin '{origin}' for product '{met_id}' does not correspond "
                             f"to a defined reactant carbon ({list(reactant_carbon_map.keys())}) or '?'.")
                else:
                     # It's a valid reactant carbon source like 'GLC_ext1'
                     origin_counts[origin] += 1

        # Optional: Check total atom balance (allows carbon loss/gain if needed)
        # num_question_marks = sum(o == '?' for origins in parsed_mapping.get('products', {}).values() for o in origins)
        # if total_reactant_carbons != total_product_carbons - num_question_marks:
        #     warnings.warn(f"Atom balance mismatch: {total_reactant_carbons} reactant C atoms vs "
        #                   f"{total_product_carbons - num_question_marks} mapped product C atoms (excluding '?'). "
        #                   f"Check mapping string: {parsed_mapping}") # Provide more context


        # Check if any reactant carbon is used more than once as an origin
        for origin, count in origin_counts.items():
            if count > 1:
                 warnings.warn(
                     f"Reactant carbon '{origin}' is mapped to {count} different product carbons. "
                     f"Ensure this is correct based on reaction stoichiometry/mechanism."
                 )

    def get_metabolite_ids(self, constant=None, non_constant=None):
        """Returns a list of metabolite IDs, optionally filtered by constant flag."""
        ids = list(self.metabolites.keys())
        if constant is True:
            ids = [mid for mid in ids if self.metabolites[mid]['is_constant']]
        if non_constant is True:
            ids = [mid for mid in ids if not self.metabolites[mid]['is_constant']]
        return ids

    def get_reaction_ids(self):
        """Returns a list of reaction IDs."""
        return list(self.reactions.keys())

    def get_initial_state_vector(self, metabolite_ids):
        """
        Returns the initial state vector (concentrations and labeling)
        for a given list of metabolite IDs.
        # ... (docstring remains the same) ...
        """
        initial_state = []
        for met_id in metabolite_ids:
            if met_id not in self.metabolites:
                 raise ValueError(f"Metabolite ID '{met_id}' requested for initial state not found in model.")
            met_info = self.metabolites[met_id]
            initial_state.append(met_info['initial_concentration'])
            initial_state.extend(met_info['initial_labeling']) # Append FE for each carbon
        return np.array(initial_state, dtype=float)

    def get_state_vector_size(self, metabolite_id):
        """Returns the number of state variables for a single metabolite (1 for conc + C for labeling)."""
        if metabolite_id not in self.metabolites:
             raise ValueError(f"Metabolite ID '{metabolite_id}' not found when requesting state vector size.")
        return 1 + self.metabolites[metabolite_id]['carbon_count']

    def get_metabolite_indices(self, metabolite_id, dynamic_metabolite_ids):
        """
        Gets the start and end index for a metabolite's block (conc + labeling)
        within the state vector defined by dynamic_metabolite_ids.
        """
        start_index = 0
        for mid in dynamic_metabolite_ids:
            if mid not in self.metabolites:
                 raise ValueError(f"Metabolite ID '{mid}' in dynamic list not found in model.")
            num_states = self.get_state_vector_size(mid)
            if mid == metabolite_id:
                return start_index, start_index + num_states
            start_index += num_states
        raise ValueError(f"Metabolite ID '{metabolite_id}' not found in dynamic_metabolite_ids list: {dynamic_metabolite_ids}")

    def get_initial_concentrations(self, metabolite_ids):
        """ Gets initial concentrations for a list of metabolite IDs. """
        conc_dict = {}
        for met_id in metabolite_ids:
             if met_id not in self.metabolites:
                  raise ValueError(f"Metabolite ID '{met_id}' not found when requesting initial concentration.")
             conc_dict[met_id] = self.metabolites[met_id]['initial_concentration']
        return conc_dict

    def get_reaction_equation(self, reaction_id):
        """Returns a human-readable string representation of the reaction equation."""
        if reaction_id not in self.reactions:
            raise ValueError(f"Reaction ID '{reaction_id}' not found.")
        reaction = self.reactions[reaction_id]
        def format_side(met_dict):
            return " + ".join([f"{stoich} {met_id}" if stoich != 1 else met_id for met_id, stoich in met_dict.items()]) if met_dict else ""
        reactants_str = format_side(reaction['reactants'])
        products_str = format_side(reaction['products'])
        arrow = "<=>" if reaction['reversible'] else "->"
        # Handle efflux/influx where one side might be empty
        if not reactants_str: return f"{arrow} {products_str}"
        if not products_str: return f"{reactants_str} {arrow}"
        return f"{reactants_str} {arrow} {products_str}"

    def build_stoichiometric_matrix(self, metabolite_ids=None, reaction_ids=None):
        """Builds the stoichiometric matrix (S) for total concentrations."""
        if metabolite_ids is None:
            metabolite_ids = self.get_metabolite_ids(non_constant=True)
        if reaction_ids is None:
            reaction_ids = self.get_reaction_ids()

        num_metabolites = len(metabolite_ids)
        num_reactions = len(reaction_ids)
        s_matrix = np.zeros((num_metabolites, num_reactions))
        metabolite_index_map = {met_id: i for i, met_id in enumerate(metabolite_ids)}

        for j, rxn_id in enumerate(reaction_ids):
            if rxn_id not in self.reactions:
                 raise ValueError(f"Reaction ID '{rxn_id}' for stoichiometric matrix not found in model.")
            reaction = self.reactions[rxn_id]
            for met_id, stoich in reaction['reactants'].items():
                if met_id in metabolite_index_map:
                    s_matrix[metabolite_index_map[met_id], j] -= stoich
            for met_id, stoich in reaction['products'].items():
                if met_id in metabolite_index_map:
                    s_matrix[metabolite_index_map[met_id], j] += stoich
        return s_matrix, metabolite_ids, reaction_ids

    def get_metabolite_carbon_count(self, metabolite_id):
        """Gets the number of carbon atoms for a metabolite."""
        if metabolite_id not in self.metabolites:
            raise ValueError(f"Metabolite ID '{metabolite_id}' not found.")
        # Ensure carbon_count exists, default to 0 if somehow missing
        return self.metabolites[metabolite_id].get('carbon_count', 0)