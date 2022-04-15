#!/bin/bash

# Usage: ./run-sgemm-amd.sh

# Parameter to configure - change these as required!
NUM_KERN=100
DEVICE_ID=0
SIZE=24576

echo "Number of kernels: ${NUM_KERN}"
echo "GPU ID: ${DEVICE_ID}"
echo "Matrix size: ${SIZE}"

# Get UUID of GPU SGEMM kernels are run on
UUID=(`rocm-smi --showuniqueid -d ${1} | grep "GPU" | awk '{print $NF}'`)
echo "GPU UUID: ${UUID}"

# Get timestamp at the time of the run
ts=`date '+%s'`

# Generate data
echo ""
echo "Generating 2 matrices of size ${SIZE}. This will take a few minutes."
./gen_data ${SIZE}
echo "Completed generating 2 matrices"

# Begin SGEMM application
echo ""
echo "Running ${NUM_KERN} kernels of SGEMM on GPU ${DEVICE_ID}. This will takes a few minutes."

# Run rocm-smi in the background and then kill it once rocprof is done
ROCM_SMI_ARGS="-t -c -P -d ${DEVICE_ID} --repeat 0.001"
ROCM_SMI_CMD="python3 rocm_smi.py $ROCM_SMI_ARGS"
$ROCM_SMI_CMD > sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.txt &
ROCM_SMI_PID=$!

# Sleep for 5 before running rocprof
sleep 5

# Begin SGEMM run with profiler rocprof
start="$(__PREFETCH=off rocprof --stats -i metrics.txt --hip-trace --hsa-trace  -o sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.csv ./sgemm_amd ${SIZE} ${NUM_KERN} ${SIZE})"
kern_0_start="$(echo ${start} | sed -n 's/.*Kernel 0 Start Time: //p' | cut -d ' ' -f 1)"

# Rename rocprof output csv file to include the start time of the first kernel
mv sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.csv sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.csv
echo "Output csv file name: sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.csv"

# Rename rocm-smi output text file to include kern_0_start of when rocprof began to associate the text file and csv file
mv sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.txt sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.txt
echo "Output text file name: sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.txt"

# Kill rocm_smi.py
kill $ROCM_SMI_PID
echo "Completed SGEMM. Outputs in ../out"

# Remove extra output files not used in analysis - uncomment if you want to look at these!
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.db 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.copy_stats.csv 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.hip_stats.csv 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.hsa_stats.csv 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.json 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.stats.csv 
rm sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}.sysinfo.txt

# Move outputs of rocm-smi and rocprof to ../out/
mv sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.csv ../out/
mv sgemm_amd_${SIZE}_${NUM_KERN}_${UUID}_${DEVICE_ID}_${ts}_${kern_0_start}.txt ../out/
