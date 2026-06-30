#!/bin/bash
#SBATCH --job-name=convex_hull_workflow
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --hint=nomultithread
#SBATCH --account=drautrmy_0004
#SBATCH -N 1
#SBATCH --output=workflow_%j.log
#SBATCH --error=workflow_%j.err

echo "=== Starting Convex Hull Workflow on Elysium GPU ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# Activate the conda environment
source ~/.bashrc
conda activate speed_02

# Set OMP_NUM_THREADS to match the CPUs requested
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU}

# Navigate to the workflow directory
cd /home/maniykxj/ConvexHull_Workflow

# Run the python workflow script using the input.yaml configuration
echo "Executing workflow.py..."
python -u workflow.py input.yaml

echo ""
echo "=== Workflow finished at $(date) ==="
