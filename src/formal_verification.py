import os
import uuid
import subprocess
from sympy import symbols, sin, cos, tanh, sqrt, exp
from src.SymVVdot_Calculations import compute_v_and_v_dotSR
from SystemDynamicsSR import fSR, xSR, GSR, QSR, RSR
from src.Sympy2SMT2 import write_smt2


def get_sif():
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.basename(d) == "examples":
            return os.path.join(d, "dreal4_latest.sif")
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError("could not find 'examples' directory")
        d = parent


SIF_PATH = get_sif()


def a_violation_check(expression, domain=2.0, tol=1e-5, delta=1e-3):
    # symbols
    x1_s, x2_s = symbols("x1 x2")
    x_syms = [x1_s, x2_s]

    # compute SymPy expressions
    _V, V_dot, *_rest, b_norm_squared, b1, a = compute_v_and_v_dotSR(
        expression, fSR, GSR, QSR, RSR, x_syms, None
    )
    b_sym = b_norm_squared[2]  # keep your indexing
    a_sym = a
    vdot_sym = V_dot

    # out dir + UNIQUE filenames (safe in parallel)
    out_dir = "SMT_expresions"
    os.makedirs(out_dir, exist_ok=True)
    uniq = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ab_path = os.path.join(out_dir, f"ab_case_{uniq}.smt2")
    vdot_path = os.path.join(out_dir, f"vdot_case_{uniq}.smt2")

    # write SMT2
    write_smt2(ab_path, a=a_sym, b=b_sym, domain=domain, tol=tol, boxes="both")
    write_smt2(vdot_path, vdot=vdot_sym, domain=domain, tol=tol, boxes="both")

    # run dReal
    def run_dreal(path, delta=1e-3):
        proc = subprocess.run(
            ["dreal", "--model", "--precision", str(delta), path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.stdout.strip()

    try:
        out1 = run_dreal(ab_path)
        out2 = run_dreal(vdot_path)
    finally:
        # cleanup (best-effort)
        for p in (ab_path, vdot_path):
            try:
                os.remove(p)
            except OSError:
                pass

    # delta-sat if output is NOT "unsat"
    sat1 = "unsat" not in out1.lower()
    sat2 = "unsat" not in out2.lower()
    score = 1000 if (sat1 or sat2) else 0
    return score, out1, out2
