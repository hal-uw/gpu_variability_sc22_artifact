#!/bin/bash

if [ ! -x /usr/bin/wget ]; then 
    echo "Please install wget or set it in your path. Aborting."
fi

echo "Getting input graph from Suite Sparse Matrix Collection"
cd data_dirs/pannotia/pagerank_spmv/data/
#wget https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat30.tar.gz
#tar -xvzf rajat30.tar.gz
#wget https://suitesparse-collection-website.herokuapp.com/MM/ATandT/pre2.tar.gz 
#tar -xvzf pre2.tar.gz 
wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz
tar -xvzf coPapersDBLP.tar.gz
cd ../../../../
echo "Fetched input graph"


