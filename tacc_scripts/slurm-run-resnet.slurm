#!/bin/bash
#SBATCH -J resnet              # Job name
#SBATCH -o debug/myresnet.%j.out     # Name of stdout output file
#SBATCH -e debug/myresnet.%j.err     # Name of stderr error file
#SBATCH -p v100                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -x c002-[001-012],c003-[001-012],c004-[001-012],c005-[001-012],c006-[001-012],c007-[001-012],c009-[001-012]
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=bqtran2@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
node=$SLURM_JOB_NODELIST

echo "Node Number: ${node}"
run-resnet.sh
