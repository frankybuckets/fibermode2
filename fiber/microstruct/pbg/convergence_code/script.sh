#!/usr/bin/bash
#SBATCH --job-name PBG_convergence
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH --mem MaxMemPerNode
#SBATCH --partition himem
#SBATCH --mail-user piet2@pdx.edu
#SBATCH --mail-type ALL
#SBATCH --output=PBG_convergence.log
#SBATCH --error=PBG_convergence.err

echo "Starting at wall clock time:"
date
echo "Running CMT on $SLURM_CPUS_ON_NODE CPU cores"

# Load needed modules.
module load ngsolve/serial
module load gcc-9.2.0
module load intel


# Run the code.
echo "Starting PBG convergence study: "
date
python3 convergence_study.py
echo "Ending PBG convergence study:"
date
