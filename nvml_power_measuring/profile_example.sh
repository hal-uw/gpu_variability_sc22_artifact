#!/bin/bash

# Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#Usage: ./profile_six_gpu_power.sh BE_DP_FP_ADD 20000000

rate=100 #NOT NEEDED
samples=-1 #Will continue sampling until end of application

sleep_time=5 #Seconds waiting between pinging for temperature
max_wait_iterations=20  # 5 minutes waiting in total
threshold_temperature_diff=1  # Threshold above starting temperature allowed, adjust as needed

PROFILER="dumpGpuPower"
bm=${1}
###########################################
#1) START WITH THE IN ORDER

DEVID=0
#Do for iterations of typical
GPU_UUID=${UUID_list[${DEVID}]}
idle_temperature=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --id=${DEVID})

echo "Starting profiling"

mkdir -p "${bm}/${HOSTNAME}"

#Launch the profiler in the background
"$PROFILER" -r "$rate" -n "$samples" -d "$DEVID"  >> "${bm}/${HOSTNAME}/${HOSTNAME}_gpu${DEVID}_${GPU_UUID}_${bm}.txt" &
pid_profiler=$!

#Launch the application (in the background as well)
CUDA_VISIBLE_DEVICES=$DEVID $bm >> /dev/null & 
pid=$!
wait $pid
#Send kill signal so profiler can catch and smooth cleanup
kill -2 $pid_profiler
echo "Profiling concluded."

echo "Sleeping..."
#Adaptive sleeping mechanism so next experiment starts at relatively same temperature
wait_counter=0
while true; do
    current_time=$(date +%s%N)
    current_temperature=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --id=${DEVID})
    if ((current_temperature > (idle_temperature + threshold_temperature_diff))); then
        sleep $sleep_time
        ((wait_counter++))
    else
        break
    fi

    # Check if max wait iterations are reached
    if [ $wait_counter -ge $max_wait_iterations ]; then
        break
    fi
done
