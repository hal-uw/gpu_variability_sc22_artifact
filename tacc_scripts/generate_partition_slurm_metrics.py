import os

# Script details
app_name = "bert-single"
out_dir = "{}-metrics-slurm-scripts".format(app_name)

metric_list = [ 
#    'sm_efficiency',
    'dram_utilization',
    'l2_utilization',
    'double_precision_fu_utilization',
    'single_precision_fu_utilization',
    'half_precision_fu_utilization',
    'tensor_precision_fu_utilization',
    'special_fu_utilization',
    'tex_fu_utilization',
    # 'cf_fu_utilization',
    # 'ldst_fu_utilization',
    'stall_memory_dependency',
    'stall_memory_throttle',
    'stall_sleeping',
    'stall_pipe_busy',
    'stall_exec_dependency',
]
# The rest of what the script should have CAUTION ABOUT THE FIRST LINE STARTING IMMEDIATELY AFTER THE """
prefix = """#!/bin/bash
#SBATCH -J APP_NAME-metric   # Job name
#SBATCH -o debug/out/APP_NAME-metric.%j.out     # Name of stdout output file
#SBATCH -e debug/err/APP_NAME-metric.%j.err     # Name of stderr error file
#SBATCH -p v100         # Queue (partition) name
#SBATCH -N 1           # Total # of nodes (must be 1 for serial)
#SBATCH -n 1           # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A Deep-Learning-at-Sca
#SBATCH -t 48:00:00       # Run time (hh:mm:ss)
#SBATCH --mail-user=bqtran2@wisc.edu
#SBATCH --mail-type=all     # Send email at begin and end of job

""".replace("APP_NAME",app_name)

run_app_line = """
echo "Running metric: METRIC"
run-{}-metrics.sh 0 METRIC""".format(app_name)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

#For all the partitions I want to create scripts for
for metric in metric_list:
    file_name = "{}_{}.slurm".format(app_name,metric)
    filepath = os.path.join(out_dir, file_name)
    run_app_with_metric = run_app_line.replace("METRIC",metric)
    with open(filepath,"w") as f:
        f.writelines(prefix)
        f.writelines(run_app_with_metric)