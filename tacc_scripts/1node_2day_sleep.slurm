#!/bin/bash
#SBATCH -J test   # job name
#SBATCH -o test.out         # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of nodes
#SBATCH -n 1
#SBATCH -A Deep-Learning-at-Sca
#SBATCH -p v100            # queue (partition) -- normal, development, etc.
#SBATCH -t 48:00:00        # run time (hh:mm:ss) - 2 days
#SBATCH --mail-user=bqtran2@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

sleep 172800

#Include this line to exclude nodes that we are not interested in
#-x c002-[001-012],c008-[001-012],c004-[001-012],c005-[001-012],c006-[001-012],c007-[001-012],c009-[001-012]
