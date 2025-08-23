#!/bin/bash
# submit.sh — PyTorch 2.5 + CUDA 12.1 (Apptainer)
#
# Usage (run once to prepare, then submit):
#   cd ~
#   # (If needed) pull the image:
#   # apptainer pull torch_2.5_py312.sif docker://ghcr.io/andrew-xqy/ml-containers:2.5-py3.12-cuda12.1
#   mkdir -p ~/dataset ~/results ~/results/slurm ~/code
#
#   sbatch submit.sh
#
# Quick GPU test (optional):
#   apptainer exec --nv ~/torch_2.5_py312.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

#SBATCH --job-name=beam_image_reconstruction_model_training
#SBATCH --output=results/slurm/training-%j.out
#SBATCH --error=results/slurm/training-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

# Load Apptainer (adjust if your cluster uses a different module name)
module load apps/apptainer

# Paths in your HOME (adjust if yours differ)
IMG="$HOME/torch_2.5_py312.sif"
DATA="$HOME/dataset"
OUT="$HOME/results"
CODE="$HOME/code"

# Ensure output dirs exist (host side)
mkdir -p "$OUT/slurm"

# Run inside the container
srun apptainer exec --nv --writable-tmpfs \
  -B "$DATA:/workspace/dataset" \
  -B "$OUT:/workspace/results" \
  -B "$CODE:/workspace/code" \
  "$IMG" /bin/bash -lc '
    python -m pip install -U pip
    # Optional: install your xflow package if it is under code/xflow
    if [ -d /workspace/code/xflow ]; then
      pip install -e /workspace/code/xflow
    fi
    cd /workspace/code/examples/machine_learning
    python training.py --data /workspace/dataset --out /workspace/results
  '
