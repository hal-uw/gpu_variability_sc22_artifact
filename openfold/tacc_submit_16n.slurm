#!/bin/bash
#SBATCH -J openfold-run              # Job name
#SBATCH -o debug/openfold.%j.out     # Name of stdout output file
#SBATCH -e debug/openfold.%j.err     # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 1                   # Total # of nodes
#SBATCH --ntasks-per-node=16
#SBATCH -t 06:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

# Load necessary modules
module load tacc-apptainer
module load cuda

ncu --version
whereis ncu

cd /scratch1/08503/rnjain/new-apps/openfold/container_version/openfold/
apptainer exec --nv /scratch1/00946/zzhang/container/openfold_0.1.sif /home1/apps/cuda/12.2/bin/ncu --target-processes all --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./run.sh
