#!/bin/bash

echo "cd to src directory for pannotia"
cd src/cuda/pannotia/pagerank_mod
make clean
env VARIANT=SPMV make
cp pagerank_spmv ../../../../
cd ../../../../
echo "Compiled pagerank_spmv binary"
echo "Getting input graph from Suite Sparse Matrix Collection"
cd data_dirs/pannotia/pagerank_spmv/data/
wget https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat30.tar.gz
tar -xvzf rajat30.tar.gz
cd ../../../../
echo "Fetched input graph"
echo "Build for PageRank complete, now launch run-pagerank.sh"
