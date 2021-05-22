#!/usr/bin/bash
#SBATCH --job-name kol_harfconvstudy_linear
#SBATCH --array 0-3
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH --mem MaxMemPerNode
#SBATCH --partition himem
#SBATCH --mail-user piet2@pdx.edu
#SBATCH --mail-type ALL
#SBATCH --output slurm_outputs/kol_harfconvstudy_linear_out_%A-%a
#SBATCH --error slurm_errors/kol_harfconvstudy_linear_err_%A-%a

# Load needed modules.
module load ngsolve/serial
module load gcc-9.2.0
module load intel

# The polynomial degree for the h-refinement study.
ps=("7")

# The number of mesh refinements.
nrefs=("0" "1" "2" "3")

# The PML strength.
alpha=10

# The initial span size.
nspan=5

# The number of quadrature points for FEAST.
npts=4

# The names of fibers.
names=("poletti" "kolyadin")

# The types of modes we wish to compute.
modestrs=("LP01" "LP11" "LP21" "LP02")

# The filename prefixes for the reference ARF objects.
prefixes=("poletti_reference", "kolyadin_reference")

# Set the number of refinements based on the task id.
nref=${nrefs[$SLURM_ARRAY_TASK_ID]}

# Set the polynomial degree.
p=${ps[0]}

# Run the code.
echo "Starting run of convergence_study_arf_modes.py: `date`"
#python3 convergence_study_arf_modes.py $p $p $nref $alpha $nspan $npts ${names[0]} ${modestrs[0]}
python3 convergence_study_arf_modes.py $p $p $nref $alpha $nspan $npts ${names[1]} ${modestrs[0]}
echo "Ending run of convergence_study_arf_modes.py: `date`"
