#!/bin/bash

# Usage: ./build-sgemm-nvidia.sh

SGEMM_BIN=./sgemm_nvidia
if [[ -f "$SGEMM_BIN" ]]; then
    make clean
fi
make all