#! /bin/bash

# Run lammps using the REAXC benchmark with nvprof 
# Output from the MD simulation is logged in log.lammps
# Input configuration values for $x,$y and $z are set in the cmdline below
# ./run-lammps0.sh ${gpu_num} ${run_num} ${node_num}
#                     ${1}       ${2}       ${3}

UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo $UUID
ts=`date '+%s'`

echo lammps_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv
touch /scratch/08503/rnjain/tacc-lammps-r2/lammps_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv

export CUDA_LAUNCH_BLOCKING=1
mpiexec -np 1 -x CUDA_LAUNCH_BLOCKING --bind-to core /usr/local/cuda-10.1/bin/nvprof --devices ${1} --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file /scratch/08503/rnjain/tacc-lammps-r2/lammps_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ../src/lmp_kokkos_cuda_mpi -k on g 1 device ${1} -sf kk -pk kokkos neigh half neigh/qeq full newton on -v x 8 -v y 16 -v z 16 -in in.reaxc.hns -nocite
