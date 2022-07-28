#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [[ "$uarch" == *ppc64le* ]]; then
    echo "Base image (sunway513/rocm-dev) only supports AMD64/x86_64  architectures. Aborting." 
else
    #singularity pull docker://sunway513/rocm-dev:4.5.2-rocblas-bench
    singularity run --nv docker://sunway513/rocm-dev:4.5.2-rocblas-bench ./build-sgemm-amd.sh
    singularity run --nv docker://sunway513/rocm-dev:4.5.2-rocblas-bench ./run-sgemm-amd.sh
fi



