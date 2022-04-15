#!/bin/bash

# Usage: ./build-sgemm-amd.sh

SGEMM_BIN=./sgemm_amd
if [[ -f "$SGEMM_BIN" ]]; then
    make clean
fi
make all