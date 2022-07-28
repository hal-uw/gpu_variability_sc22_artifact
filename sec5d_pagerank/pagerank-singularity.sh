#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [ ! -f data_dirs/pannotia/pagerank_spmv/data/rajat30/rajat30.mtx ]; then
   echo "Input graph not found. Run fetch-input.sh first. Aborting."
   exit
fi

if [[ "$uarch" == *ppc64le* ]]; then
    singularity run --nv docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04 ./build-pagerank.sh
    singularity run --nv docker://nvidia/cuda-ppc64le:10.1-devel-ubuntu18.04 ./run-pagerank.sh
else
    singularity run --nv docker://nvidia/cuda:10.1-devel-ubuntu18.04 ./build-pagerank.sh
    singularity run --nv docker://nvidia/cuda:10.1-devel-ubuntu18.04 ./run-pagerank.sh
fi


