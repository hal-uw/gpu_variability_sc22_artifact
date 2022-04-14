#!/bin/bash

echo "cd to src directory for pannotia"
cd src/cuda/pannotia/pagerank_mod
make clean
env VARIANT=SPMV make
cd ../../../../
echo "Compiled pagerank_spmv binary"
