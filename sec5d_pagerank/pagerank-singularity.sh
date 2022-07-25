#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [ ! -f data_dirs/pannotia/pagerank_spmv/data/rajat30/rajat30.mtx ]; then
   echo "Input graph not found. Run fetch-input.sh first. Aborting."
   exit
fi

if [[ "$uarch" == *ppc64le* ]]; then
    singularity pull docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04
    singularity run --nv cuda-ppc64le_10.1-devel-ubuntu18.04.sif ./build-pagerank.sh
    singularity run --nv cuda-ppc64le_10.1-devel-ubuntu18.04.sif ./run-pagerank.sh
else
    singularity pull docker://nvidia/cuda:10.1-devel-ubuntu18.04
    singularity run --nv cuda_10.1-devel-ubuntu18.04.sif ./build-pagerank.sh
    singularity run --nv cuda_10.1-devel-ubuntu18.04.sif ./run-pagerank.sh
fi


