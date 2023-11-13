import os

#Details about where I am slurming
partition_min_limit = 2
partition_max_limit = 9
node_min_limit = 1
node_max_limit = 12
padding_limit = 3

#Detail how much I want to sbatch
partition_start = 2
partition_end = 9
node_start = 5
node_end = 12

#Script details
script_name = "single-gpu-resnet"
out_dir = "{}-slurm-scripts".format(script_name)

#The rest of what the script should have CAUTION ABOUT THE FIRST LINE STARTING IMMEDIATELY AFTER THE """
prefix="""#!/bin/bash
#SBATCH -J single-gpu-resnet   # Job name
#SBATCH -o debug/out/resnet.%j.out     # Name of stdout output file
#SBATCH -e debug/err/resnet.%j.err     # Name of stderr error file
#SBATCH -p v100        # Queue (partition) name
#SBATCH -N 1           # Total # of nodes (must be 1 for serial)
#SBATCH -n 1           # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A Deep-Learning-at-Sca
"""

suffix="""#SBATCH -t 48:00:00       # Run time (hh:mm:ss)
#SBATCH --mail-user=bqtran2@wisc.edu
#SBATCH --mail-type=all     # Send email at begin and end of job

node=$SLURM_JOB_NODELIST
for ((num_run=1; num_run<5; num_run++))
do
  echo "Run Number: ${num_run}"
  run-resnet-single.sh 0 $num_run
  run-resnet-single.sh 1 $num_run
  run-resnet-single.sh 2 $num_run
  run-resnet-single.sh 3 $num_run
done
"""

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

#For all the partitions I want to create scripts for
for cabinet in range(partition_start,partition_end+1):
    cabinet_str = str(cabinet).zfill(padding_limit)
    #For all the nodes I want to create scripts for
    for node in range(node_start,node_end+1):
        node_str = str(node).zfill(padding_limit)
        filename = "{}-c{}-{}.slurm".format(script_name,cabinet_str,node_str)       
        filepath = os.path.join(out_dir,filename)
        #Add all generic exclusions
        exclude_line = ",".join(["c{}-[{}-{}]".format(str(cab).zfill(padding_limit), str(node_min_limit).zfill(padding_limit),str(node_max_limit).zfill(padding_limit)) for cab in range(partition_min_limit,partition_max_limit+1) if cab != cabinet])
        #Additional exclusions within my cabinet
        if node == node_min_limit:
            additional_exclude = ",c{}-[{}-{}]".format(cabinet_str,str(node_min_limit+1).zfill(padding_limit),str(node_max_limit).zfill(padding_limit))
        elif node == node_max_limit:
            additional_exclude = ",c{}-[{}-{}]".format(cabinet_str,str(node_min_limit).zfill(padding_limit),str(node_max_limit-1).zfill(padding_limit))
        #Consider if we exclude +1 on either end
        elif node-1 == node_min_limit:
            additional_exclude = ",c{}-{}".format(cabinet_str,str(node_min_limit).zfill(padding_limit))
            additional_exclude += ",c{}-[{}-{}]".format(cabinet_str,str(node+1).zfill(padding_limit),str(node_max_limit).zfill(padding_limit))
        elif node+1 == node_max_limit:
            additional_exclude = ",c{}-[{}-{}]".format(cabinet_str,str(node_min_limit).zfill(padding_limit),str(node-1).zfill(padding_limit))
            additional_exclude += ",c{}-{}".format(cabinet_str,str(node_max_limit).zfill(padding_limit))
        else:
            additional_exclude = ",c{}-[{}-{}]".format(cabinet_str,str(node_min_limit).zfill(padding_limit),str(node-1).zfill(padding_limit))
            additional_exclude += ",c{}-[{}-{}]".format(cabinet_str,str(node+1).zfill(padding_limit),str(node_max_limit).zfill(padding_limit))
        exclude_line += additional_exclude

        # print(filename,exclude_line)

        #Overwrite to specific files
        with open(filepath,"w") as f:
            f.writelines(prefix)
            f.writelines("#SBATCH -x "+exclude_line+"\n")
            f.writelines(suffix)