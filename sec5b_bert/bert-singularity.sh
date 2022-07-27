#!/bin/bash

uarch=$(uname -m)

if [[ "$uarch" == *ppc64le* ]]; then
    echo "Base image (nvcr.io/nvidia/pytorch) only supports LINUX/AMD64 and LINUX/ARM64 architectures. Aborting." 
else
    echo "Pulling PyTorch NVIDIA image from Docker registry"
    singularity pull docker://nvcr.io/nvidia/pytorch:22.06-py3
    echo "Loading dataset and running BERT within container"
    singularity run --nv pytorch_22.06-py3.sif scripts/install.sh
    echo "Profiling BERT"  
    singularity run --nv pytorch_22.06-py3.sif scripts/run_pretraining_lamb.sh
    echo "BERT run completed."
fi

