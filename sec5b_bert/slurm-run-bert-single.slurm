#!/bin/bash
#SBATCH -J bert              # Job name
#SBATCH -o debug/mybert.%j.out     # Name of stdout output file
#SBATCH -e debug/mybert.%j.err     # Name of stderr error file
#SBATCH -p gpu-a100                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

node=$SLURM_JOB_NODELIST
module load tacc-apptainer
module load cuda
./bert-singularity-single.sh
