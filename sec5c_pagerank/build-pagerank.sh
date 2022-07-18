#!/bin/bash

echo "cd to src directory for pannotia"
cd src/cuda/pannotia/pagerank_mod
make clean
env VARIANT=SPMV make
cp pagerank_spmv ../../../../
cd ../../../../
echo "Compiled pagerank_spmv binary"

