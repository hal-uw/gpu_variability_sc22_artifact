#!/bin/bash
#SBATCH -J resnet-within              # Job name
#SBATCH -o debug/resnet-within.%j.out     # Name of stdout output file
#SBATCH -e debug/resnet-within.%j.err     # Name of stderr error file
#SBATCH -p rtx-dev                 # Queue (partition) name
#SBATCH -x c196-[031-032],c196-[021-022] 
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
node=$SLURM_JOB_NODELIST
module load tacc-apptainer
echo "Node Number: ${node}"
resnet-singularity-multi.sh
