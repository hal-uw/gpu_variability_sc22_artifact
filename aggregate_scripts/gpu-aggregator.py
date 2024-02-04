'''
    File name: gpu-aggregator.py
    Author: Prasoon Sinha
    Python Version: 3.7
'''

import pandas as pd
import sys
import numpy as np
import math
import collections
import os
import json
from explorer import read_nvprof_gpu_trace, system_types
from itertools import islice


def chunk_dict(data, SIZE=1100):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def csv_aggregator(df_dict, cluster):

    # Dictionary that will be returned in final_collection
    aggregate_data_dict = {}

    # Initial set up of aggregate_data_dict
    # We realize this initialization isn't necessary, but
    # include it to make it easier to read what keys are in the dict
    aggregate_data_dict['exp'] = []
    aggregate_data_dict['input_size'] = []
    aggregate_data_dict['reps'] = []
    aggregate_data_dict['uuid'] = []
    if cluster == 'summit':
        aggregate_data_dict['cabinet'] = []
        aggregate_data_dict['node'] = []
        aggregate_data_dict['row'] = []
        aggregate_data_dict['col'] = []
        aggregate_data_dict['ts'] = []
    elif cluster == 'tacc':
        aggregate_data_dict['cabinet'] = []
        aggregate_data_dict['node'] = []
        aggregate_data_dict['ts'] = []
    elif cluster == 'vortex':
        aggregate_data_dict['ts'] = []
    elif cluster == 'cloudlab':
        #     aggregate_data_dict['max_freq_set'] = []
        #     aggregate_data_dict['max_pwr_set'] = []
        aggregate_data_dict['ts'] = []
    aggregate_data_dict['kern'] = []
    aggregate_data_dict['kern_min_start'] = []
    aggregate_data_dict['kern_max_start'] = []
    aggregate_data_dict['num_kerns'] = []
    aggregate_data_dict['perf'] = []
    for item, val in system_types.items():
        aggregate_data_dict[item] = []

    for k, v in df_dict.items():
        # Split the name of the csv file
        li = k.split('_')

        # Populate meta data in the dictionary
        aggregate_data_dict['exp'].append(li[0])
        #aggregate_data_dict['input_size'].append(li[1])
        aggregate_data_dict['reps'].append(li[1])
        if cluster == 'summit':
            # Summit has a cabinet, node, row, col
            aggregate_data_dict['uuid'].append(li[3])
            aggregate_data_dict['cabinet'].append(li[4][0:3])
            aggregate_data_dict['node'].append(li[4])
            aggregate_data_dict['row'].append(li[4][0])
            aggregate_data_dict['col'].append(li[4][1:3])
            aggregate_data_dict['ts'].append(li[5])
        elif cluster == 'vortex':
            # Vortex csv files don't have cabinet, node, row, col
            aggregate_data_dict['uuid'].append(li[3])
            aggregate_data_dict['ts'].append(li[4])
        elif cluster == 'tacc':
            aggregate_data_dict['cabinet'].append(li[2][0:4])
            aggregate_data_dict['node'].append(li[2][5:8])
            aggregate_data_dict['ts'].append(li[-1])
            #aggregate_data_dict['uuid'].append(li[3])
            # Time stamp, cabinet, and node missing
        elif cluster == 'cloudlab':
            # aggregate_data_dict['max_freq_set'].append(li[3])
            # aggregate_data_dict['max_pwr_set'].append(li[4])
            aggregate_data_dict['uuid'].append(str(li[3]))
            aggregate_data_dict['ts'].append(li[-1])

        # Kernel name (i.e. volta_sgemm_128x64_nn)
        #print(v.streams)
        kern = [i for i in v.streams if 'Kokkos' in i][0]
        aggregate_data_dict['kern'].append(kern)

        # The GPU that this csv file corresponds to (i.e. Tesla V100-SXM2-16GB (0))
        GPU = v.data[(v.data.Name) == kern].Device.min()

        # Get GPU number within the specified machine that this csv file corresponds to
        GPU_split = GPU.split(' ')
        device_id = int(GPU_split[2][1])

        # Start time of first kernel instance
        min_start = v.data[(v.data.Device == v.devices[device_id]) & (
            v.data.Name == kern)].Start.min()
        aggregate_data_dict['kern_min_start'].append(min_start)

        # Start time of last kernel instance
        max_start = v.data[(v.data.Device == v.devices[device_id]) & (
            v.data.Name == kern)].Start.max()
        aggregate_data_dict['kern_max_start'].append(max_start)

        # Number of kernels for this particular run of the benchmark
        count = v.data[(v.data.Device == v.devices[device_id])
                       & (v.data.Name == kern)].Start.count()
        aggregate_data_dict['num_kerns'].append(count)

        # Filter data only for specific GPU and only kernel data
        a = v.data[(v.data.Device == v.devices[device_id]) & (
            (v.data.Start > min_start) & (v.data.Start < max_start))]

        # Average performance (duration) of all kernel instances in ms
        aggregate_data_dict['perf'].append(
            a[a.Name == kern].Duration.mean() * 1000)

        # Median value for each metric (frequency, memory frequency, power, temperature) for each run of benchmark
        for item, val in system_types.items():
            aggregate_data_dict[item].append(
                a[a.Name == val[0]][item].median())

        # print("Completed aggregation for: " + k)

    return aggregate_data_dict


def handle_data(data_dir, cluster):
    # For all the csv files in the directory, populate dictionary with the file name followed by its path
    base_file_dict = {}
    for f in os.listdir(data_dir):
        if "csv" in f:
            li = f.split(".")
            key = "_".join(i for i in li if i != 'csv')
            base_file_dict[key] = os.path.join(data_dir, f)

    # Print number of csv files in directory
    print("The number of csv files in this directory is: " +
          str(len(base_file_dict)))

    # Read each csv and aggregate
    chunk = 0
    dir = data_dir.split('/')
    for file_dict in chunk_dict(base_file_dict):
        # k (key) - the csv file name
        # v (value) - the csv file path
        df_dict = {}
        for k, v in file_dict.items():
            tmp = read_nvprof_gpu_trace(v)
            # Some files may be empty
            if tmp is not None:
                df_dict[k] = tmp
        collection = csv_aggregator(df_dict, cluster)
        print(collection)
        for key in collection:
            print(key + ": " + str(len(collection[key])))
        aggregated_data = pd.DataFrame(collection)
        chunk += 1
        if cluster == 'summit':
            aggregated_data.to_csv(
                '../chunked-' + cluster + '-aggregations/'+dir[2]+'_agg_'+str(chunk)+".csv")
        elif cluster == 'cloudlab':
            aggregated_data = aggregated_data.sort_values(by=['uuid'])
            aggregated_data.to_csv(
                '../../cuda-sync-test/cloudlab-sync-aggregations.csv')
        print("Completed reading and aggregations for: " + str(chunk * 1100))


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 gpu-aggregator.py [path to data directory] [cluster]")
        exit()

    # Path to directory containing csv files to read and aggregate
    data_dir = sys.argv[1]
    # Cluster - summit, vortex, tacc, cloudlab, etc.
    cluster = sys.argv[2]

    # Read and aggregate data
    handle_data(data_dir, cluster)


if __name__ == "__main__":
    main()
