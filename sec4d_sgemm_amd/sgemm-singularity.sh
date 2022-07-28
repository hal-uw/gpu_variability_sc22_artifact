#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [[ "$uarch" == *ppc64le* ]]; then
    echo "Base image (sunway513/rocm-dev) only supports AMD64/x86_64  architectures. Aborting." 
else
    singularity pull docker://sunway513/rocm-dev:4.5.2-rocblas-bench
    singularity run --nv rocm-dev_4.5.2-rocblas-bench.sif ./build-sgemm-amd.sh
    singularity run --nv rocm-dev_4.5.2-rocblas-bench.sif ./run-sgemm-amd.sh
fi



