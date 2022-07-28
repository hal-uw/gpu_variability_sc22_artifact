#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [[ "$uarch" == *ppc64le* ]]; then
    singularity run --nv docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04 ./build-sgemm-nvidia.sh
    sleep 30
    singularity run --nv docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04 ./run-sgemm-nvidia.sh
else
    singularity run --nv docker://nvidia/cuda:10.1-devel-ubuntu18.04 ./build-sgemm-nvidia.sh
    sleep 30
    singularity run --nv docker://nvidia/cuda:10.1-devel-ubuntu18.04 ./run-sgemm-nvidia.sh
fi

