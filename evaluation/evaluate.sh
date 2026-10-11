#!/bin/bash
#SBATCH --job-name=pllum-eval
#SBATCH --output=out/eval_slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=tutorial
#SBATCH --partition=tutorial
#SBATCH --reservation=tutorial-llm

set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_DIR="$SCRATCH/PLLuM-SLURM-Tutorial"
DATASET_PATH="$REPO_DIR/data/example.test.json"
OUTPUT_DIR="$RUN_DIR/evaluation"

cd "$REPO_DIR"
source "$RUN_DIR/.venv/bin/activate"
export HF_HOME="$RUN_DIR/huggingface"

python -m evaluation.evaluate \
  --adapter "$RUN_DIR/adapter" \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
