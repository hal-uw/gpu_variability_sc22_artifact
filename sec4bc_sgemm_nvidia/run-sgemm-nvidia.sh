#!/bin/bash

# Usage: ./run-sgemm-nvidia.sh

export CUDA_HOME=/usr/local/cuda-10.1 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Parameter to configure - change these as required!
NUM_KERN=100
DEVICE_ID=0
SIZE=25536

echo "Number of kernels: ${NUM_KERN}"
echo "GPU ID: ${DEVICE_ID}"
echo "Matrix size: ${SIZE}"

# Get UUID of GPU SGEMM kernels are run on
UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo "GPU UUID: ${UUID}"

# Get timestamp at the time of the run
ts=`date '+%s'`

# File name
FILE_NAME=sgemm_nvidia_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.csv
echo "Output file name: ${FILE_NAME}"

# Run application with profiling via nvprof
echo ""
echo "Generating 2 matrices of size ${SIZE}. This will take a few minutes."
#./gen_data ${SIZE}
echo "Completed generating 2 matrices"

echo ""
echo "Running ${NUM_KERN} kernels of SGEMM on GPU ${DEVICE_ID}. This will takes a few minutes."

if [ -d "/usr/loca/cuda-10.1/bin" ]
then
    __PREFETCH=off /usr/local/cuda-10.1/bin/nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME} --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm_nvidia ${SIZE} ${NUM_KERN} ${DEVICE_ID}
    echo "Completed SGEMM. Outputs in current working directory."
elif [ -d "/usr/local/cuda/bin" ]
then
    __PREFETCH=off /usr/local/cuda/bin/nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME} --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm_nvidia ${SIZE} ${NUM_KERN} ${DEVICE_ID}
    echo "Completed SGEMM. Outputs in current working directory."
else 
    echo "Couldn't find CUDA bin directory. Check /usr/local for your CUDA installation and update run-sgemm-nvidia.sh. Aborting."
fi


