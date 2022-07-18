#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [[ "$uarch" == *ppc64le* ]]; then
    singularity pull docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04
    singularity run --nv cuda-ppc64le_10.1-devel-ubuntu18.04.sif ./build-sgemm-nvidia.sh
    singularity run --nv cuda-ppc64le_10.1-devel-ubuntu18.04.sif ./run-sgemm-nvidia.sh
else
    singularity pull docker://nvidia/cuda:10.1-devel-ubuntu18.04
    singularity run --nv cuda_10.1-devel-ubuntu18.04.sif ./build-sgemm-nvidia.sh
    singularity run --nv cuda_10.1-devel-ubuntu18.04.sif ./run-sgemm-nvidia.sh
fi



