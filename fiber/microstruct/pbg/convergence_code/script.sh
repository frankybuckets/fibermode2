#!/usr/bin/bash
#SBATCH --job-name PBG_convergence
#SBATCH --array 0-3
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH --mem MaxMemPerNode
#SBATCH --partition himem
#SBATCH --mail-user piet2@pdx.edu
#SBATCH --mail-type ALL
#SBATCH --output slurm_outputs/
#SBATCH --error slurm_errors/

# Load needed modules.
module load ngsolve/serial
module load gcc-9.2.0
module load intel


# Run the code.
echo "Starting PBG convergence study: `date`"
python3 convergence_study.py
echo "Ending PBG convergence study: `date`"
