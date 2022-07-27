#!/bin/bash

# Install APEX
echo "Installing APEX"
#cd utils/apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd ../..

# Install boto3
echo "Installing boto3"
python -m pip install boto3
echo "Installing h5py"
pip install h5py
pip install tokenizers


# Download dataset 
#if ${1} 
#then
#   ./scripts/create_datasets.sh --output ./data --nproc 8 --download --no-books --format --encode --encode-type bert 
#fi
