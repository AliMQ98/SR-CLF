from sympy import (
    symbols,
    Matrix,
    exp,
)

# Define symbolic variables
x1, x2 = symbols("x1 x2")


# Define x(x) as a vector
def xSR(x1, x2):
    return Matrix([x1, x2])


# Define f(x) as a vector
def fSR(x1, x2):
    f1 = -x2
    f2 = x1 + x2 * (x1**2 -1)

    return Matrix([f1, f2])


# Define G(x) as a matrix
def GSR(x1, x2):
    G1 = [0, 0]
    G2 = [0, x2 * (x1**2 -1)]
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
