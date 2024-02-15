#!/bin/bash
#SBATCH -J resnet-across              # Job name
#SBATCH -o debug/resnet-across.%j.out     # Name of stdout output file
#SBATCH -e debug/resnet-across.%j.err     # Name of stderr error file
#SBATCH -p gpu-a100                  # Queue (partition) name
#SBATCH -N 2                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 2                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:40:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
node=$SLURM_JOB_NODELIST
module load tacc-apptainer
echo "Node Number: ${node}"
resnet-singularity-across.sh
