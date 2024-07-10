import os
import subprocess

prefix = """#!/bin/bash
#SBATCH -J bert              # Job name
#SBATCH -o debug/mybert.%j.out     # Name of stdout output file
#SBATCH -e debug/mybert.%j.err     # Name of stderr error file
#SBATCH -p gpu-a100                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=kchen346@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
"""

suffix = """
node=$SLURM_JOB_NODELIST
module load tacc-apptainer
module load cuda
./bert-singularity-single.sh
"""

script = prefix + suffix
script_path = f'slurm-c003-003-run-bert-single.slurm'
print(script_path)
with open(script_path, 'w') as f:
    f.write(script)
# submit this file to the sbatch
result = subprocess.run(['sbatch', script_path],
                        capture_output=True, text=True)

# Check if sbatch command was successful
if result.returncode == 0:
    print(f"Job successfully submitted for node")
else:
    print(f"Failed to submit job for node. Error: {result.stderr}")

if os.path.exists(script_path):
    os.remove(script_path)

print("All jobs have been submitted.")

# 'c302-001', 'c302-002', 'c302-003', 'c302-004', 'c303-001', 'c303-002', 'c303-003', 'c303-004', 
# 'c304-001', 'c305-001', 'c305-002', 'c305-003', 'c305-004', 'c306-001', 'c306-002', 'c306-003', 
# 'c306-004', 'c308-001', 'c308-002', 'c308-003', 'c308-004', 'c309-001', 'c309-002', 'c309-003', 
