#!/bin/bash

# Usage: ./build-sgemm-nvidia.sh

SGEMM_BIN=./sgemm_nvidia
GEN_DATA_BIN=./gen_data
if [[ -f "$SGEMM_BIN" ]]; then
    make clean
fi
make all