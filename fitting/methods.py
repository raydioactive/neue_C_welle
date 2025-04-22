import numpy as np
import random
import time
import math

# Nelder-Mead (Simplex) Implementation
# Based on the description from Wikipedia and standard implementations.

def nelder_mead_simplex(objective_function, initial_guess,
                        initial_step=0.1, max_iterations=1000,
                        xtol=1e-6, ftol=1e-6, adaptive=False,
                        no_improve_thr=10e-6, no_improve_break=10,
                        max_time_sec=None, verbose=False):
    """
    Minimizes a function using the Nelder-Mead (Simplex) algorithm.

    Args:
        objective_function (callable): The function to minimize. Must take a list or numpy array
                                       of parameters and return a single scalar value.
        initial_guess (list or np.ndarray): Initial guess for the parameters.
        initial_step (float or list/np.ndarray): Initial step size to create the simplex.
                                                 If float, used for all dimensions.
                                                 If list/array, must match length of initial_guess.
        max_iterations (int): Maximum number of iterations allowed.
        xtol (float): Absolute error in xopt between iterations that is acceptable for convergence.
        ftol (float): Absolute error in func(xopt) between iterations that is acceptable for convergence.
        adaptive (bool): If True, use adaptive parameters for Nelder-Mead steps (rho, chi, psi, sigma).
        no_improve_thr (float): Threshold for function value improvement check.
        no_improve_break (int): Number of consecutive iterations without improvement to trigger termination.
        max_time_sec (float, optional): Maximum allowed execution time in seconds. Defaults to None (no time limit).
        verbose (bool): If True, print progress information during optimization.

    Returns:
        dict: A dictionary containing the optimization results:
              - 'x': The best parameter vector found.
              - 'fun': The value of the objective function at the best parameters.
              - 'nit': The number of iterations performed.
              - 'nfev': The number of function evaluations.
              - 'status': An integer status code (0: success, 1: max iterations, 2: max time, 3: stalled).
              - 'message': A string describing the termination reason.
    """
    start_time = time.time()
    x0 = np.array(initial_guess, dtype=float)
    N = len(x0)

    # --- Adaptive parameters (from Fuchang Gao and Lixing Han, 2012) ---
    if adaptive:
        rho = 1.0
        chi = 1.0 + 2.0 / N
        psi = 0.75 - 1.0 / (2.0 * N)
        sigma = 1.0 - 1.0 / N
    else:
        rho = 1.0   # Reflection coefficient
        chi = 2.0   # Expansion coefficient
        psi = 0.5   # Contraction coefficient
        sigma = 0.5 # Shrink coefficient

    # --- Build initial simplex ---
    if isinstance(initial_step, (int, float)):
        steps = np.full(N, initial_step, dtype=float)
    elif isinstance(initial_step, (list, np.ndarray)) and len(initial_step) == N:
        steps = np.array(initial_step, dtype=float)
    else:
        raise ValueError("initial_step must be a scalar or array matching dimension of initial_guess")

    simplex = [x0]
    f_simplex = [objective_function(x0)]
    nfev = 1

    for i in range(N):
        x_new = np.copy(x0)
        # Create points along axes, handle zero initial guess
        if x_new[i] != 0:
            x_new[i] = x_new[i] * (1 + steps[i]) if steps[i] != 0 else x_new[i] + 0.00025 # Perturb if step is 0
        else:
             x_new[i] = steps[i] if steps[i] != 0 else 0.00025 # Use step directly if initial guess is 0
        simplex.append(x_new)
        f_simplex.append(objective_function(x_new))
        nfev += 1

    simplex = np.array(simplex)
    f_simplex = np.array(f_simplex)

    # --- Iteration loop ---
    iterations = 0
    no_improve_count = 0
    prev_best_f = f_simplex[np.argmin(f_simplex)]

    while iterations < max_iterations:
        iterations += 1

        # Check time limit
        if max_time_sec is not None and (time.time() - start_time) > max_time_sec:
            best_idx = np.argmin(f_simplex)
            return {'x': simplex[best_idx], 'fun': f_simplex[best_idx], 'nit': iterations, 'nfev': nfev,
                    'status': 2, 'message': 'Maximum execution time exceeded.'}

        # 1. Order points by function value (best to worst)
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]

        # Check convergence criteria
        f_std = np.std(f_simplex)
        x_std = np.std([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, N + 1)])

        if f_std < ftol and x_std < xtol:
             return {'x': simplex[0], 'fun': f_simplex[0], 'nit': iterations, 'nfev': nfev,
                     'status': 0, 'message': 'Optimization terminated successfully (xtol and ftol).'}

        # Check for stalling
        if abs(f_simplex[0] - prev_best_f) < no_improve_thr:
            no_improve_count += 1
        else:
            no_improve_count = 0
        prev_best_f = f_simplex[0]

        if no_improve_count >= no_improve_break:
            return {'x': simplex[0], 'fun': f_simplex[0], 'nit': iterations, 'nfev': nfev,
                    'status': 3, 'message': f'Optimization stalled: No improvement for {no_improve_break} iterations.'}


        # 2. Calculate centroid (excluding the worst point)
        centroid = np.mean(simplex[:-1], axis=0)

        # 3. Reflection
        x_refl = centroid + rho * (centroid - simplex[-1])
        f_refl = objective_function(x_refl)
        nfev += 1

        if f_simplex[0] <= f_refl < f_simplex[-2]:
            # Accept reflection
            simplex[-1] = x_refl
            f_simplex[-1] = f_refl
            if verbose: print(f"Iter {iterations}: Reflect")
            continue

        # 4. Expansion
        if f_refl < f_simplex[0]:
            x_exp = centroid + chi * (x_refl - centroid)
            f_exp = objective_function(x_exp)
            nfev += 1
            if f_exp < f_refl:
                # Accept expansion
                simplex[-1] = x_exp
                f_simplex[-1] = f_exp
                if verbose: print(f"Iter {iterations}: Expand")
            else:
                # Accept reflection (better than expansion)
                simplex[-1] = x_refl
                f_simplex[-1] = f_refl
                if verbose: print(f"Iter {iterations}: Reflect (post-expansion)")
            continue

        # 5. Contraction
        if f_refl >= f_simplex[-2]:
            # Try outside contraction if reflection is worse than second worst
            if f_refl < f_simplex[-1]:
                x_cont = centroid + psi * (x_refl - centroid) # Outside contraction
                f_cont = objective_function(x_cont)
                nfev += 1
                if f_cont <= f_refl:
                    simplex[-1] = x_cont
                    f_simplex[-1] = f_cont
                    if verbose: print(f"Iter {iterations}: Outside Contraction")
                    continue
            # Try inside contraction if reflection is worse than worst
            else: # f_refl >= f_simplex[-1]
                x_cont = centroid - psi * (centroid - simplex[-1]) # Inside contraction
                f_cont = objective_function(x_cont)
                nfev += 1
                if f_cont < f_simplex[-1]:
                    simplex[-1] = x_cont
                    f_simplex[-1] = f_cont
                    if verbose: print(f"Iter {iterations}: Inside Contraction")
                    continue

        # 6. Shrink (if contraction failed)
        if verbose: print(f"Iter {iterations}: Shrink")
        x_best = simplex[0]
        for i in range(1, N + 1):
            simplex[i] = x_best + sigma * (simplex[i] - x_best)
            f_simplex[i] = objective_function(simplex[i])
            nfev += 1

    # Reached max iterations
    best_idx = np.argmin(f_simplex)
    return {'x': simplex[best_idx], 'fun': f_simplex[best_idx], 'nit': iterations, 'nfev': nfev,
            'status': 1, 'message': 'Maximum number of iterations reached.'}


# Simulated Annealing Implementation

def simulated_annealing(objective_function, initial_guess, bounds=None,
                        max_iterations=10000, initial_temperature=100.0,
                        cooling_schedule='exponential', cooling_rate=0.95,
                        step_size=0.1, max_time_sec=None, verbose=False):
    """
    Minimizes a function using the Simulated Annealing algorithm.

    Args:
        objective_function (callable): The function to minimize. Must take a list or numpy array
                                       of parameters and return a single scalar value.
        initial_guess (list or np.ndarray): Initial guess for the parameters.
        bounds (list of tuples, optional): List of (min, max) bounds for each parameter.
                                           If None, parameters are unbounded (or step size handles).
        max_iterations (int): Maximum number of iterations allowed.
        initial_temperature (float): Starting temperature for the annealing process.
        cooling_schedule (str): How the temperature decreases ('exponential', 'linear', 'logarithmic').
        cooling_rate (float): Parameter controlling the cooling speed (e.g., alpha for exponential).
        step_size (float or list/np.ndarray): Maximum size of the random step taken to find neighbors.
                                              If float, used for all dimensions.
                                              If list/array, must match length of initial_guess.
        max_time_sec (float, optional): Maximum allowed execution time in seconds. Defaults to None.
        verbose (bool): If True, print progress information.

    Returns:
        dict: A dictionary containing the optimization results:
              - 'x': The best parameter vector found.
              - 'fun': The value of the objective function at the best parameters.
              - 'nit': The number of iterations performed.
              - 'nfev': The number of function evaluations (approx = iterations + 1).
              - 'status': An integer status code (0: success (converged/max iter), 1: max time).
              - 'message': A string describing the termination reason.
    """
    start_time = time.time()
    current_params = np.array(initial_guess, dtype=float)
    N = len(current_params)

    # Validate bounds
    if bounds is not None:
        if len(bounds) != N:
            raise ValueError("Length of bounds must match length of initial_guess.")
        bounds = np.array(bounds, dtype=float)
        # Ensure initial guess is within bounds
        if not np.all((current_params >= bounds[:, 0]) & (current_params <= bounds[:, 1])):
             print("Warning: Initial guess is outside provided bounds. Clamping initial guess.")
             current_params = np.clip(current_params, bounds[:, 0], bounds[:, 1])

    # Validate step size
    if isinstance(step_size, (int, float)):
        steps = np.full(N, step_size, dtype=float)
    elif isinstance(step_size, (list, np.ndarray)) and len(step_size) == N:
        steps = np.array(step_size, dtype=float)
    else:
        raise ValueError("step_size must be a scalar or array matching dimension of initial_guess")


    current_energy = objective_function(current_params)
    nfev = 1
    best_params = np.copy(current_params)
    best_energy = current_energy
    temperature = initial_temperature
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Check time limit
        if max_time_sec is not None and (time.time() - start_time) > max_time_sec:
             return {'x': best_params, 'fun': best_energy, 'nit': iterations, 'nfev': nfev,
                     'status': 1, 'message': 'Maximum execution time exceeded.'}

        # 1. Generate a neighbor solution
        # Random step within [-step_size, +step_size] for each dimension
        neighbor_params = current_params + np.random.uniform(-steps, steps, N)

        # Apply bounds if provided
        if bounds is not None:
            neighbor_params = np.clip(neighbor_params, bounds[:, 0], bounds[:, 1])

        neighbor_energy = objective_function(neighbor_params)
        nfev += 1

        # 2. Calculate change in energy
        delta_e = neighbor_energy - current_energy

        # 3. Decide whether to accept the neighbor
        accept = False
        if delta_e < 0:
            # Always accept better solutions
            accept = True
        else:
            # Accept worse solutions with a probability based on temperature
            probability = math.exp(-delta_e / temperature) if temperature > 1e-9 else 0.0
            if random.random() < probability:
                accept = True

        if accept:
            current_params = neighbor_params
            current_energy = neighbor_energy
            # Update best solution found so far
            if current_energy < best_energy:
                best_params = np.copy(current_params)
                best_energy = current_energy

        # 4. Cool the temperature
        if cooling_schedule == 'exponential':
            temperature *= cooling_rate # alpha (0 < alpha < 1)
        elif cooling_schedule == 'linear':
            temperature -= initial_temperature / max_iterations
        elif cooling_schedule == 'logarithmic':
             # Avoid log(1) = 0 early on
             temperature = initial_temperature / (1 + cooling_rate * math.log(1 + iterations))
        else:
             raise ValueError(f"Unknown cooling schedule: {cooling_schedule}")

        # Ensure temperature doesn't go below zero (for linear) or too close to zero
        temperature = max(temperature, 1e-9)

        if verbose and iterations % 100 == 0:
            print(f"Iter {iterations}: Temp={temperature:.4f}, Current E={current_energy:.4g}, Best E={best_energy:.4g}")

    # Max iterations reached or temperature cooled down
    message = 'Optimization terminated successfully (max iterations or temperature reached).'
    return {'x': best_params, 'fun': best_energy, 'nit': iterations, 'nfev': nfev,
            'status': 0, 'message': message}

