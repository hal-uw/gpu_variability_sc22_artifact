#!/bin/bash

# run-pagerank.sh $gpu_num $num_run $node

# Run pagerank with nvprof 
# ./run-pagerank.sh ${gpu_num} ${run_num} ${node_num}
#                      ${1}       ${2}       ${3}
# Assumed for artifact: 0          1-2        0        

UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo $UUID
ts=`date '+%s'`

echo pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv
touch ../../out/pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv

echo running PageRank on graph: rajat30.mtx : file_format MMTranspose
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ../../out/pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ${BINDIR}/release/pagerank_spmv data_dirs/pannotia/pagerank_spmv/data/rajat30.mtx 2

echo waitingfor nvprof to flush all data to log 
wait
echo end PageRank, profiling log in out/pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv
