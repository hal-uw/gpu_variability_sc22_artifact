
#!/bin/bash

nvidia-smi -L > uuids/${HOSTNAME}_uuid.txt

uarch=$(uname -m)

if [[ "$uarch" == *ppc64le* ]]; then
    echo "Base image (nvcr.io/nvidia/pytorch) only supports LINUX/AMD64 and LINUX/ARM64 architectures. Aborting." 
else
    echo "Pulling PyTorch NVIDIA image from Docker registry"
    #singularity pull docker://nvcr.io/nvidia/pytorch:22.06-py3
    singularity run --nv docker://nvcr.io/nvidia/pytorch:22.06-py3 scripts/install.sh
    echo "Loading dataset and running BERT within container"
    echo "Profiling BERT" 
    for i in {0..1}; do
        singularity run --nv docker://nvcr.io/nvidia/pytorch:22.06-py3 scripts/run_pretraining_lamb_single.sh ${i}
    done
    echo "BERT run completed."
fi
