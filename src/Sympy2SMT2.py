from sympy import symbols, Function, Integer, Rational, Float
from sympy import sin, cos, tanh, sinh, sqrt, exp, asin

def _real_lit(x):
    s = str(x)
    return s if ("." in s or "e" in s or "E" in s) else f"{s}.0"

def sympy_to_smt(e):
    # Symbols
    if e.is_Symbol:
        return e.name

    # Numbers
    if e.is_Number:
        if isinstance(e, Rational):
            return f"(/ {_real_lit(e.p)} {_real_lit(e.q)})"
        if isinstance(e, Integer):
            return _real_lit(int(e))
        if isinstance(e, Float):
            return _real_lit(e.evalf(17))
        return _real_lit(e.evalf(17))

    # Addition
    if e.is_Add:
        return f"(+ {' '.join(sympy_to_smt(a) for a in e.args)})"

    # Multiplication
    if e.is_Mul:
        return f"(* {' '.join(sympy_to_smt(a) for a in e.args)})"

    # Powers
    if e.is_Pow:
        base, expn = e.as_base_exp()

        # sqrt: exponent = 1/2
        if isinstance(expn, Rational) and expn.p == 1 and expn.q == 2:
            return f"(sqrt {sympy_to_smt(base)})"

        # Integer exponents
        if expn.is_Integer:
            n = int(expn)
            if n == 0:
                return "1.0"
            if n == 1:
                return sympy_to_smt(base)
            if n > 1:
                return f"(* {' '.join(sympy_to_smt(base) for _ in range(n))})"
            if n == -1:
                return f"(/ 1.0 {sympy_to_smt(base)})"
            # n < -1
            return f"(/ 1.0 (* {' '.join(sympy_to_smt(base) for _ in range(-n))}))"

        # Float-looking integer, e.g., 2.0
        if expn.is_Float and float(expn).is_integer():
            n = int(float(expn))
            return sympy_to_smt(base) if n == 1 else f"(* {' '.join(sympy_to_smt(base) for _ in range(n))})"

        raise NotImplementedError(f"Non-integer power: {e}")

    # Elementary functions
    fn_map = {sin: "sin", cos: "cos", tanh: "tanh", sinh: "sinh", sqrt: "sqrt", exp: "exp", asin: "arcsin"}
    if e.func in fn_map:
        return f"({fn_map[e.func]} {' '.join(sympy_to_smt(a) for a in e.args)})"

    if isinstance(e.func, Function):
        return f"({e.func.__name__} {' '.join(sympy_to_smt(a) for a in e.args)})"

    raise NotImplementedError(f"Cannot print: {e} ({type(e)})")


# -----------------------
# SMT2 file writer
# -----------------------
def write_smt2(filename, a=None, b=None, vdot=None, domain=2.0, tol=1e-5, boxes="both"):
    """
    Writes an SMT-LIB2 file.
      - Pass either (a and b) or (vdot).
      - a, b, vdot are SymPy expressions in x1, x2.
      - boxes: "both" | "neg" | "pos"
    """
    if (vdot is None) == (a is None or b is None):
        raise ValueError("Provide either vdot, or both a and b (but not both).")

    lines = []
    lines.append("(set-logic QF_NRA)")
    lines.append("(declare-fun x1 () Real)")
    lines.append("(declare-fun x2 () Real)")

    if vdot is not None:
        lines.append(f"(define-fun vdot ((x1 Real) (x2 Real)) Real {sympy_to_smt(vdot)})")
    else:
        lines.append(f"(define-fun a ((x1 Real) (x2 Real)) Real {sympy_to_smt(a)})")
        lines.append(f"(define-fun b ((x1 Real) (x2 Real)) Real {sympy_to_smt(b)})")

    if boxes == "both":
        lines.append(
            f"(assert (or (and (<= 0.001 x1) (<= x1 {domain}) (<= 0.001 x2) (<= x2 {domain}))"
            f"            (and (<= -{domain} x1) (<= x1 -0.001) (<= -{domain} x2) (<= x2 -0.001))))"
        )
    elif boxes == "neg":
        lines.append(f"(assert (and (<= -{domain} x1) (<= x1 -0.001) (<= -{domain} x2) (<= x2 -0.001)))")
    elif boxes == "pos":
        lines.append(f"(assert (and (<= 0.001 x1) (<= x1 {domain}) (<= 0.001 x2) (<= x2 {domain})))")
    else:
        raise ValueError('boxes must be "both", "neg", or "pos".')

    if vdot is not None:
        lines.append(f"(assert (> (vdot x1 x2) {tol}))")
    else:
        lines.append("(assert (= (b x1 x2) 0.0))")
        lines.append(f"(assert (> (a x1 x2) {tol}))")

    lines.append("(check-sat)")
    lines.append("(exit)")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
