from sympy import (
    symbols,
    Matrix,
    sin,
    cos,
)

# Define symbolic variables
x1, x2 = symbols("x1 x2")


# Define x(x) as a vector
def xSR(x1, x2):
    return Matrix([x1, x2])


# Define f(x) as a vector
def fSR(x1, x2):
    g, l, b, m = 9.81, 0.5, 0.1, 0.15
    A, B = g / l, -b / (m * l**2)

    f1 = x2
    f2 = A * sin(x1) + B * x2
    return Matrix([f1, f2])


# Define G(x) as a matrix
def GSR(x1, x2):
    G1 = [0, 0]
    G2 = [0, 26.7]
    return Matrix([G1, G2])


# Define Q(x) as a matrix
def QSR(x1, x2):
    Q1 = [1, 0]
    Q2 = [0, 1]
    return Matrix([Q1, Q2])


# Define R(x) as a matrix
def RSR(x1, x2):
    R1 = [1, 0]
    R2 = [0, 1]
    return Matrix([R1, R2])
