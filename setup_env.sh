#!/bin/bash
#SBATCH --job-name=pllum-setup
#SBATCH --output=out/setup_env_slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=tutorial
#SBATCH --partition=tutorial
#SBATCH --reservation=tutorial-llm

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${SCRATCH:?Run this on Athena with SCRATCH set}"

RUN_DIR="$SCRATCH/pllum-slurm-lab"
VENV_DIR="$RUN_DIR/.venv"
PYTHON_BIN="/net/software/v1/software/Python/3.12.3-GCCcore-13.3.0/bin/python3.12"

mkdir -p "$RUN_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo "Environment ready: $VENV_DIR"
