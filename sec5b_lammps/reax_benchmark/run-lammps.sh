#! /bin/bash

# Run lammps using the REAXC benchmark with nvprof 
# Output from the MD simulation is logged in log.lammps
# Input configuration values for $x,$y and $z are set in the cmdline below
# ./run-lammps0.sh ${gpu_num} ${run_num} ${node_num}
#                     ${1}       ${2}       ${3}

# Get GPU UUID using nvidia-smi
UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo $UUID
# Get current timestamp of run
ts=`date '+%s'`

# Format of profiling data log - lammps_<GPU-UUID>_run<run-num>_node<node-num>_gpu<device-num>_<timestamp>.csv
echo out/lammps_${UUID}_run${2}_${ts}.csv
touch ../../out/lammps_${UUID}_run${2}_${ts}.csv 

export CUDA_LAUNCH_BLOCKING=1
mpiexec -np 1 -x CUDA_LAUNCH_BLOCKING --bind-to core /usr/local/cuda-11/bin/nvprof  --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ../../out/lammps_${UUID}_run${2}_${ts}.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ../src/lmp_kokkos_cuda_mpi -k on g 1 device ${1} -sf kk -pk kokkos neigh half neigh/qeq full newton on -v x 16 -v y 8 -v z 12 -in in.reaxc.hns -nocite

