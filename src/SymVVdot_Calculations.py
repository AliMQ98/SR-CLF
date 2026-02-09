from sympy import Matrix, sqrt


# Define the gradient of V(x)
def V_x(V, x_syms):
    """Compute symbolic gradient of V with respect to given symbols."""
    return Matrix([V.diff(xi) for xi in x_syms])


# Compute V and V_dot
def compute_v_and_v_dotSR(V_value, f, G, Q, R, x_syms, u_func):
    """
    Computes the symbolic Lyapunov derivative and control terms.

    Parameters:
        V_value : sympy.Expr
            The candidate Control Lyapunov function V(x).
        u_func : sympy.Expr or None
            User-specified control function u(x) if any.
        f, G, Q, R : callables
            Functions returning sympy Matrix dynamics/cost matrices.
        x_syms : list of sympy.Symbol
            List of state variables (x1, x2, ..., xn).

    Returns:
        tuple:
            - V_value: Value function
            - V_dot: Time derivative
            - lambda_x: Lambda function
            - u_value[1]: Control input (example: second input)
            - u_value2[1]: Alternative control input (without scaling)
            - u_mag: Control magnitude
            - b_norm_squared[0]: Norm squared
            - b_T[1]: Control effectiveness along second input
    """
    # n_states = len(x_syms)

    # Evaluate system dynamics
    f_vector = f(*x_syms)  # shape (n_states, 1)
    G_matrix = G(*x_syms)  # shape (n_states, n_inputs)
    Q_matrix = Q(*x_syms)
    R_matrix = R(*x_syms)

    # State vector
    x_vector = Matrix(x_syms)

    # Gradient of V
    V_grad = V_x(V_value, x_syms)  # shape (n_states, 1)

    # Compute a and b_T
    a = (V_grad.T * f_vector)[0]
    b_T = V_grad.T * G_matrix  # shape (1, n_inputs)

    # Norms
    b_norm_squared = (b_T * R_matrix.inv() * b_T.T)[0]
    x_norm_squared = (x_vector.T * Q_matrix * x_vector)[0]

    # lambda_x
    """
    lambda_x = safe_divideSR(
        sqrt(a**2 + x_norm_squared * b_norm_squared) + a,
        b_norm_squared,
        default=0
    )
    """
    lambda_x = (sqrt(a**2 + x_norm_squared * b_norm_squared) + a) / b_norm_squared

    # Control law
    if u_func is None:
        u_value = -R_matrix.inv() * b_T.T * lambda_x
    else:
        # user-supplied u_func is scalar, maps to second input (index 1)
        n_inputs = G_matrix.shape[1]
        u_entries = [0] * n_inputs
        u_entries[1] = u_func
        u_value = Matrix(u_entries)

    # Alternative control (without lambda_x scaling)
    u_value2 = -R_matrix.inv() * b_T.T

    # Time derivative of V
    V_dot = (V_grad.T * f_vector + V_grad.T * G_matrix * u_value)[0]

    # Control effort
    u_mag = (u_value.T * R_matrix * u_value)[0]

    # Return important terms
    return (
        V_value,
        V_dot,
        lambda_x,
        u_value[1],
        u_value2[1],
        u_mag,
        [b_norm_squared, b_T[0], b_T[1]],
        b_T[0, 1],
        a,
    )
