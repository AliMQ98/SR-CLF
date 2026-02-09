import numpy as np
from src.VVdot_Calculations import compute_v_and_v_dot
from src.SymFunctions import (
    contains_symbol,
)
from sympy import symbols, Function
import pygmo as pg
from SystemDynamics import f, G, Q, R
import warnings


def eval_MSE_sol(individual, num_consts, toolbox, true_data, ind2MSE, consts):
    """
    Evaluates the Mean Squared Error (MSE) of a given individual (control function)
    based on Lyapunov function conditions.

    Parameters:
    individual : function
        A callable function representing the candidate Lyapunov function.
    true_data : Dataset
        A dataset containing input values (state variables x1 and x2).

    Returns:
    tuple:
        - MSE : float
            The computed Mean Squared Error with penalties for violations.
        - V_vals : numpy.ndarray
            The evaluated Lyapunov function values over the state space.
    """
    warnings.filterwarnings("ignore")

    x1_vals = true_data.get_input(0)  # Access X1
    x2_vals = true_data.get_input(1)  # Access X2
    x_vals = [x1_vals, x2_vals]
    X1 = true_data.X1
    X2 = true_data.X2

    # Normalization factors (use domain bounds)
    s1 = np.max(np.abs(x1_vals))
    s2 = np.max(np.abs(x2_vals))
    scales = [s1, s2]
    
    # Normalized grid
    X1_norm = X1 / s1
    X2_norm = X2 / s2
    x_vals_norm = [x1_vals/s1, x2_vals/s2]

    # Ensure individual is callable before evaluating
    if not callable(individual):
        # print(f"Error: individual {individual} is not callable")
        return 1e10  # Assign a high error to discard bad individuals

    try:
        # Evaluate V once, then short-circuit if bad
        try:
            V_vals  = individual(X1_norm,  X2_norm,  consts)
            V_val_0 = individual(0.0, 0.0, consts)

            if (not np.isfinite(V_val_0)) or (not np.all(np.isfinite(V_vals))):
                return 1e10, None
            if np.max(np.abs(V_vals)) > 1e6:   # simple magnitude guard
                return 1e10, None
            # Ensure V_vals is a NumPy array (even if scalar)
            V_vals = np.asarray(V_vals)
            if V_vals.ndim == 0:
                # print("V_vals is scalar, broadcasting to full grid")
                V_vals = np.full(X1.shape, V_vals)
        except Exception:
            return 1e10, None  # <<< anything weird: discard immediately
        
        x1, x2, a = symbols("x1 x2, a")

        MSE = 0

        if not contains_symbol(str(ind2MSE), "x1"):
            MSE += 1e6
        if not contains_symbol(str(ind2MSE), "x2"):
            MSE += 1e6

        # Check properness numerically
        proper_func = (X1_norm**2 + X2_norm**2)
        upper_ref_bound = 10 * proper_func
        lower_ref_bound = 0.1 * proper_func
        proper_penalty = np.count_nonzero((upper_ref_bound <= V_vals) | (V_vals <= lower_ref_bound))
        MSE += proper_penalty

        # Compute Lyapunov-related properties
        (
            _,
            V_grad_mag,
            V_dot_vals,
            _,
            _,
            _,
            a_vals,
            _,
            x_norm_squared,
        ) = compute_v_and_v_dot(
            V_vals,
            x_vals,
            true_data,
            scales,
            f,
            G,
            Q,
            R,
            TRQlim=None,
            u_func=None
        )

        # Count violations of Lyapunov function conditions
        V_violations = np.count_nonzero(V_vals <= 0)
        # a_violations = 0  # np.count_nonzero(a_vals >= 0)
        V_dot_violations = np.count_nonzero(V_dot_vals >= -0.004 * (x_norm_squared))

        # Check for invalid values (e.g., NaN or inf) and add penalty
        penalty = 0
        if np.any(np.isnan(V_vals)) or np.any(np.isinf(V_vals)):
            penalty += 1e6  # Add large penalty for invalid V_vals
        if np.any(np.isnan(a_vals)) or np.any(np.isinf(a_vals)):
            penalty += 1e6  # Add large penalty for invalid V_vals
        if np.any(np.isnan(V_dot_vals)) or np.any(np.isinf(V_dot_vals)):
            penalty += 1e6  # Add large penalty for invalid V_dot_vals
        # if np.any(np.isnan(lambda_vals)) or np.any(np.isinf(lambda_vals)):
        #     penalty += 1e6  # Add large penalty for invalid lambda_vals

        V_grad_mag = np.abs(np.mean(V_grad_mag) - 1.6137)
        
        # Compute final MSE including violations and penalties
        MSE += (
            (
                V_violations
                # + a_violations
                + V_dot_violations
                + penalty
                + 1000 * V_val_0**2
            )
            + 1 * V_grad_mag
        )

        # Handle invalid MSE values
        if np.isnan(MSE) or np.isinf(MSE):
            MSE = 1e10  # Assign large penalty

        return MSE, V_vals

    except Exception as e:
        print(f"Error in eval_MSE_sol: {e}")
        return 1e10, None  # Assign a large penalty in case of failure


def eval_MSE_and_tune_constants(individual, num_consts, toolbox, true_data, ind2MSE):
    if num_consts > 0:
        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err, _ = eval_MSE_sol(
                    individual, num_consts, toolbox, true_data, ind2MSE, x
                )
                return [total_err]

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        # algo = pg.algorithm(pg.pso(gen=10))
        algo = pg.algorithm(pg.sea(gen=10))
        pop = pg.population(prb, size=70)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x

    else:
        MSE, _ = eval_MSE_sol(
            individual, num_consts, toolbox, true_data, ind2MSE, consts=[]
        )
        consts = []
    return MSE, consts
