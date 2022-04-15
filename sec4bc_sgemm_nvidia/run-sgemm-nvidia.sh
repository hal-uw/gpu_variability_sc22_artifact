#!/bin/bash

# Usage: ./run-sgemm-nvidia.sh

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
echo "Generating 2 matrices of size ${SIZE}"
./gen_data ${SIZE}
echo "Completed generating 2 matrices"

echo ""
echo "Running ${NUM_KERN} of SGEMM on GPU ${DEVICE_ID}. This application takes a few minutes."
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME} --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm_nvidia ${SIZE} ${NUM_KERN} ${DEVICE_ID}
echo "Completed SGEMM. Outputs in ../out"

# Move output csv file to ../out/
mv ${FILE_NAME} ../out/