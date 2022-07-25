#!/bin/bash

chmod u+x fetch-input.sh
chmod u+x build-pagerank.sh
chmod u+x run-pagerank.sh
./fetch-input.sh
./build-pagerank.sh
./run-pagerank.sh 0 1 0

# run-pagerank.sh ${gpu_num} ${run_num} ${node_num}
#                 [0/1/2/3]   [user-assigned] [user-assigned, usually used for compute nodes in large clusters]
