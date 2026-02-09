# SymCLF

**SYMBOLIC REGRESSION OF CONTROL LYAPUNOV FUNCTIONS (SymCLF)**  
A symbolic regression framework for discovering Control Lyapunov Functions (CLFs) using genetic programming to design stabilizing feedback controllers for nonlinear systems.

## Installation

First, install [Flex](https://github.com/cpml-au/Flex) and its dependencies by following the instructions in its repository.

Then, create a Conda environment using the provided `environment.yaml`:

```bash
conda env create -f environment.yml
conda activate symclf
