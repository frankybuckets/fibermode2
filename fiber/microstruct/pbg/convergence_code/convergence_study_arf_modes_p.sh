#!/usr/bin/bash
#SBATCH --job-name parfconvstudy
#SBATCH --array 0-2
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH --mem-per-cpu 28G 
#SBATCH --partition himem
#SBATCH --mail-user bqp2@pdx.edu
#SBATCH --mail-type ALL
#SBATCH --output slurm_outputs/parfconvstudy_out_%A-%a
#SBATCH --error slurm_errors/parfconvstudy_err_%A-%a

# Load needed modules.
module load ngsolve/serial
module load gcc-9.2.0
module load intel

# The minimum and maximum polynomial degrees.
pmin=2
pmax=12

# The number of mesh refinements for the p and h studies.
nrefs=("0" "1" "2")

# The PML strength.
alpha=10

# The initial span size.
nspan=5

# The number of quadrature points for FEAST.
npts=4

# The name of fiber.
names=("poletti" "kolyadin")

# The type of mode we wish to compute.
modestrs=("LP01" "LP11" "LP21" "LP02")

# The filename prefix for the reference ARF objects.
prefixes=("poletti_reference", "kolyadin_reference")

# Set the number of refinements based on the task id.
nref=${nrefs[$SLURM_ARRAY_TASK_ID]}

# Run the code.
echo "Starting run of convergence_study_arf_modes.py: `date`"
python3 convergence_study_arf_modes.py $pmin $pmax $nref $alpha $nspan $npts ${names[0]} ${modestrs[1]}
#python3 convergence_study_arf_modes.py $pmin $pmax $nref $alpha $nspan $npts ${names[1]} ${modestrs[0]}
echo "Ending run of convergence_study_arf_modes.py: `date`"
