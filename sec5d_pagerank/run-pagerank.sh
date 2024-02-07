#!/bin/bash

# run-pagerank.sh $gpu_num $num_run $node

export CUDA_HOME=/usr/local/cuda-10.1 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run pagerank with nvprof 
# ./run-pagerank.sh ${gpu_num} ${run_num} ${node_num}
#                      ${1}       ${2}       ${3}
# Assumed for artifact: 0          1-2        0        

UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo $UUID
ts=`date '+%s'`

echo pagerank_${UUID}_run${2}_node_${HOSTNAME}_gpu${1}_${ts}.csv
touch pagerank_${UUID}_run${2}_node_${HOSTNAME}_gpu${1}_${ts}.csv

#echo running PageRank on graph: rajat30.mtx : file_format MMTranspose
echo running PageRank on graph: pre2.mtx : file_format MMTranspose

#if [ -d "/usr/local/cuda-10.1/bin" ]
#then
#    __PREFETCH=off /usr/local/cuda-10.1/bin/nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./pagerank_spmv data_dirs/pannotia/pagerank_spmv/data/rajat30/rajat30.mtx 2
#    echo waitingfor nvprof to flush all data to log 
#    wait
#    echo "Completed PageRank. Output in pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv"
#elif [ -d "/usr/local/cuda/bin" ]
#then
    #__PREFETCH=off /usr/local/cuda/bin/nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./pagerank_spmv data_dirs/pannotia/pagerank_spmv/data/rajat30/rajat30.mtx 2
#nv-nsight-cu-cli --metrics gpu__time_duration.avg --csv ./pagerank_spmv data_dirs/pannotia/pagerank_spmv/data/rajat30/rajat30.mtx 2 | grep '^"' > pagerank_${UUID}_run${2}_node${3}_gpu${1}_${ts}.csv
nv-nsight-cu-cli --metrics gpu__time_duration.avg --csv ./pagerank_spmv data_dirs/pannotia/pagerank_spmv/data/pre2/pre2.mtx 2 | grep '^"' > pagerank_${UUID}_run${2}_node_${HOSTNAME}_gpu${1}_${ts}.csv
echo waitingfor nvprof to flush all data to log 
wait
echo "Completed PageRank. Output in pagerank_${UUID}_run${2}_node_${HOSTNAME}_device${1}_${ts}.csv"
#else 
#    echo "Couldn't find CUDA bin directory. Check /usr/local for your CUDA installation and update run-pagerank.sh. Aborting."
#fi
