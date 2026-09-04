#!/usr/bin/env bash
# Idempotent Cloud Agent setup for SymCLF (SR-CLF).
#
# Sets up a self-contained conda environment named `symclf` with the exact
# stack the code needs: the Flex symbolic-regression engine (pinned to the
# revision this repository was developed against), the cpml-au DEAP/dctkit
# forks, jax 0.5.0, pygmo, and the scientific Python stack. Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
ENV_NAME="symclf"
ENV_FILE="$REPO_ROOT/.cursor/symclf-env.yaml"
FLEX_DIR="${FLEX_DIR:-$HOME/Flex}"
# Last Flex revision whose YAML primitive format matches this repo's configs
# (the commit right before dimension/rank became enum objects upstream).
FLEX_COMMIT="39a0886"

# 1. Install Miniforge (conda + mamba) if not already present.
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "[install] Installing Miniforge into $CONDA_DIR ..."
  tmp_installer="$(mktemp --suffix=.sh)"
  curl -fsSL -o "$tmp_installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$tmp_installer" -b -p "$CONDA_DIR"
  rm -f "$tmp_installer"
else
  echo "[install] Miniforge already present at $CONDA_DIR."
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 2. Create or update the conda environment from the pinned spec.
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[install] Updating existing conda env '$ENV_NAME' ..."
  mamba env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[install] Creating conda env '$ENV_NAME' ..."
  mamba env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

# 3. Install the Flex symbolic-regression engine (pinned, editable).
if [ ! -d "$FLEX_DIR/.git" ]; then
  echo "[install] Cloning Flex into $FLEX_DIR ..."
  git clone https://github.com/cpml-au/Flex.git "$FLEX_DIR"
fi
git -C "$FLEX_DIR" fetch --all --tags --quiet || true
git -C "$FLEX_DIR" checkout --quiet "$FLEX_COMMIT"
pip install -e "$FLEX_DIR"

# 4. Make the environment available in interactive shells.
if ! grep -q "conda activate ${ENV_NAME}" "$HOME/.bashrc" 2>/dev/null; then
  echo "[install] Enabling '$ENV_NAME' auto-activation in ~/.bashrc ..."
  {
    echo ""
    echo "# >>> SymCLF conda setup >>>"
    echo "source \"$CONDA_DIR/etc/profile.d/conda.sh\""
    echo "conda activate ${ENV_NAME}"
    echo "# <<< SymCLF conda setup <<<"
  } >> "$HOME/.bashrc"
fi

echo "[install] Done. Verifying key imports ..."
python - <<'PY'
import flex, deap, dctkit, ray, jax, pygmo, numpy, scipy, sympy, sklearn, matplotlib
from flex.gp.regressor import GPSymbolicRegressor
print("flex OK | deap", deap.__version__, "| jax", jax.__version__)
PY
echo "[install] SymCLF environment ready. Activate with: conda activate ${ENV_NAME}"
