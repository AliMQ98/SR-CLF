# SymCLF

**SYMBOLIC REGRESSION OF CONTROL LYAPUNOV FUNCTIONS (SymCLF)**  
A symbolic regression framework for discovering Control Lyapunov Functions (CLFs)
using genetic programming to design stabilizing feedback controllers for nonlinear systems.

*Paper reference:*  
Ali M. Qaragoez, Rafal Wisniewski, Alessandro Lucantonio  
"Symbolic Regression of Control Lyapunov Functions", ..., 2026.

---

## Features

- Search for analytic CLFs via multi-island genetic programming (GP).
- Recover stabilizing controllers using Sontag's universal formula.
- Numerical and optional formal verification (dReal) over bounded domains.
- Support for non‑polynomial, interpretable certificates.
- Comparison against classical baselines (LQR, analytical Functions).
- Modular benchmark examples with notebooks and simulation tools.

---

## Requirements

- Linux / macOS.
- Python 3.10+ (conda environment recommended).
- [Flex](https://github.com/cpml-au/Flex) symbolic regression engine.
- Optional: [dReal](https://dreal.github.io/) for δ‑complete formal verification.
- Standard scientific Python stack (installed via `environment.yaml`).

---

## Installation

```bash
# install Flex following its repository instructions
git clone https://github.com/cpml-au/Flex.git
cd Flex
# follow the build/install instructions

# return to this repo
cd /path/to/symclf

# create environment
conda env create -f environment.yaml
conda activate symclf

# install dReal if you plan to use formal verification
# e.g., follow instructions at https://dreal.github.io/
```

> **Tip:** use `conda activate symclf && pip install -e .` if
> you plan to edit the code and want it available as a package.

---

## Quick Start

1. Configure a benchmark in one of the subfolders under `examples/`
   (the defaults already contain four systems).

2. Run symbolic regression:

    ```bash
    python examples/2DExample1/run.py
    ```

   The best candidate CLF will be written to `best_expression.txt` in
   the example directory.

3. Launch Jupyter and open the corresponding notebook:

    ```bash
    jupyter lab
    ```

   Notebooks available:

   - `examples/2DExample1/2DExample1.ipynb`
   - `examples/2DExample1/LQRandROA.ipynb`  
     (each example folder has its own pair)

4. Use notebooks to numerically evaluate the CLF, simulate closed-loop
   trajectories, visualize the region of attraction (ROA) and compute
   costs relative to baselines.

---

## Repository Layout

### API Modules

- `Fitness.fitness` – main fitness evaluation used by GP.
- `VVdot_Calculations.compute_v_and_v_dot` – numeric CLF and derivative.
- `SymVVdot_Calculations` – symbolic routines for formal specs.
- `formal_verification.a_violation_check` – generate/check SMT2 via dReal.
- `NumSol` – simulate trajectories and compute performance costs.

See docstrings in each module for further details.

---

## Development


### Adding a Benchmark

1. Copy an existing example directory (`examples/2DExample1/`)
   and update `config.yaml`, system dynamics (e.g., `SystemDynamics.py`), Fitnees (`Fitness` and `Evaluate`),  and notebooks accordingly.

2. Adjust `run.py` arguments or add a new entry point script if needed.

3. Implement numeric and symbolic evaluation routines if the system
   has special structure.

---

