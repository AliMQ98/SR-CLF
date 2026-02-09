#!/bin/bash -l
#SBATCH --job-name=my_python_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=102
#SBATCH --mem=88G
#SBATCH --time=02:00:00

# 1) Load conda from Miniforge
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 2) Activate your env
conda activate flex

# 3) Debug info
which python
python --version

# 4) Run your script
python run.py
