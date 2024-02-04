'''
    File name: explorer.py
    Authors: Akhil Guliani, Prasoon Sinha, Rutwik Jain
    Date created: 6/6/2020
    Date last modified: 2/16/2022
    Python Version: 3.7
'''

__copyright__ = "Copyright 2020, Akhil Guliani"

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import sqlite3
import os
import sys
import collections
import math
import numpy as np
import tempfile

NVTrace = collections.namedtuple(
    'NVTrace', 'data unit devices streams stream_ids')


def get_power_values(x):
    # covert x into string
    y = str(x)
    return int(y.split('/')[0]) / 1000.


def get_freq_values(x):
    # covert x into string
    y = str(x)
    return int(y.split('/')[0])


def get_memfreq_values(x):
    # covert x into string
    y = str(x)
    return int(y.split('/')[1])


system_types = collections.OrderedDict()
system_types['freq'] = ['[SM/Memory Clock (MHz)]', get_freq_values]
system_types['memfreq'] = ['[SM/Memory Clock (MHz)]', get_memfreq_values]
system_types['temp'] = ['[Temperature (C)]', float]
system_types['pwr'] = ['[Power/Limit (mW)]', get_power_values]


# system_types = {'freq':['[SM/Memory Clock (MHz)]',get_freq_values],
#  'temp':['[Temperature (C)]', float ],
#  'pwr':['[Power/Limit (mW)]',get_power_values]}

def read_nvprof_gpu_trace(trace, tag="", tag_it=False):
    list_of_lines = []
    with open(trace) as f:
        lines = f.readlines()
        for line in lines:
            if "==" in line:
                continue
            arr = line.strip().split(',')
            arr = [i.strip("\"") for i in arr]
            list_of_lines.append(arr)
    if not list_of_lines:
        print("Error, empty file: " + trace)
        print("")
        return None
    titles = list_of_lines[0]
    if tag_it:
        titles = [tag+"_"+t for t in titles]
    data = list_of_lines[2:]
    data = [d for d in data if len(d) == len(titles)]
    df = pd.DataFrame(data=data, columns=titles)
    if 'Grid X' in df.columns:
        df = df.drop(columns=['Grid X', 'Grid Y', 'Grid Z', 'Block X', 'Block Y', 'Block Z', 'Registers Per Thread', 'Static SMem',
                     'Dynamic SMem', 'Size', 'Throughput', 'SrcMemType', 'DstMemType', 'Context', 'Stream', 'Correlation_ID'])
    else:
        print('This file is probably empty: ' + trace)
        return None
    with tempfile.NamedTemporaryFile() as temp_out:
        df.to_csv(temp_out.name)
        df = pd.read_csv(temp_out.name, index_col=0)
    # df.set_index(df.columns[0])
    units = {i: j for i, j in zip(list_of_lines[0], list_of_lines[1])}
    devices = list(df.Device.unique())
    kind = list(df.Name.unique())
    kind_dict = {k: i for i, k in enumerate(kind)}
    df['ones'] = df['Name']
    df.ones = df.ones.replace(kind_dict)

    for k, v in system_types.items():
        #         print(v)
        df[k] = df[(df.Name == v[0])].System.apply(v[1])

    return NVTrace(df, units, devices, kind, kind_dict)


def aggregator(file_id, split_name_len, kernel, device_id, get_blocks=True, combined=True):
    collect = []
    for k, v in df_dict.items():
        if file_id in k:
            li = k.split("_")
            if len(li) > split_name_len:
                out_list = []
                for it in li:
                    out_list.append(it)
                    print(it, end=',')
                try:
                    print([i for i in v.streams if kernel in i][0], end=',')
                    kern = [i for i in v.streams if kernel in i][0]
                    p2p_kern = [i for i in v.streams if 'PtoP' in i][0]
                    out_list.append(kern)
                    min_start = v.data[(v.data.Device == v.devices[device_id]) & (
                        v.data.Name == kern)].Start.min()
                    if combined:
                        max_start = v.data[(v.data.Device == v.devices[device_id]) & (
                            v.data.Name == p2p_kern)].Start.max()
                    else:
                        max_start = v.data[(v.data.Device == v.devices[device_id]) & (
                            v.data.Name == kern)].Start.max()
                    count = v.data[(v.data.Device == v.devices[device_id]) & (
                        v.data.Name == kern)].Start.count()
                    print(min_start, max_start, count, sep=',', end=',')
                    out_list.append(min_start)
                    out_list.append(max_start)
                    out_list.append(count)
                    a = v.data[(v.data.Device == v.devices[device_id]) & (
                        (v.data.Start > min_start) & (v.data.Start < max_start))]
                    print(a[a.Name == kern].Duration.mean(), end=',')
                    print(a[a.Name == kern].Throughput.mean(), end=',')
                    if v.unit['Duration'] == 'ms':
                        out_list.append(a[a.Name == kern].Duration.mean())
                    else:
                        out_list.append(a[a.Name == kern].Duration.mean()*1000)
                    out_list.append(a[a.Name == p2p_kern].Throughput.mean())

                    for item, val in system_types.items():
                        print(a[a.Name == val[0]][item].median(), end=',')
                        out_list.append(a[a.Name == val[0]][item].median())

                    print("\n", end='')
                    if get_blocks:
                        val = int(a['Grid X'].max()) * int(a['Grid Y'].max())
                        out_list.append(val)
                    collect.append(out_list)
                except:
                    print("Error")
    return collect


def aggregator2(df_dict, file_id, split_name_len, kernel, dev_id, get_blocks=True, combined=True, get_dev_id=False):
    collect = []
    for k, v in df_dict.items():
        if v is None:
            continue
        if file_id in k:
            li = k.split("_")
            if len(li) > split_name_len:
                out_list = []
                for it in li:
                    out_list.append(it)
                    print(it, end=',')
                from_id = None
                x = li[-1]
#                 print()
#                 print(x)
#                 print(int(x))
                device_id = int(x) if get_dev_id else dev_id
                try:
                    #                     print(li)
                    #                     print(type(device_id))
                    print([i for i in v.streams if kernel in i][0], end=',')
                    kern = [i for i in v.streams if kernel in i][0]
                    if combined:
                        p2p_kern = [i for i in v.streams if 'PtoP' in i][0]
                    out_list.append(kern)
                    min_start = v.data[(v.data.Device == v.devices[device_id]) & (
                        v.data.Name == kern)].Start.min()

                    if combined:
                        max_start = v.data[(v.data.Device == v.devices[device_id]) & (
                            v.data.Name == p2p_kern)].Start.max()
                    else:
                        max_start = v.data[(v.data.Device == v.devices[device_id]) & (
                            v.data.Name == kern)].Start.max()
                    count = v.data[(v.data.Device == v.devices[device_id]) & (
                        v.data.Name == kern)].Start.count()
                    print(min_start, max_start, count, sep=',', end=',')
                    out_list.append(min_start)
                    out_list.append(max_start)
                    out_list.append(count)
                    a = v.data[(v.data.Device == v.devices[device_id]) & (
                        (v.data.Start > min_start) & (v.data.Start < max_start))]
                    print(a[a.Name == kern].Duration.mean(), end=',')
                    print(a[a.Name == kern].Throughput.mean(), end=',')

                    if v.unit['Duration'] == 'ms':
                        out_list.append(a[a.Name == kern].Duration.mean())
                    else:
                        out_list.append(a[a.Name == kern].Duration.mean()*1000)

                    if combined:
                        out_list.append(
                            a[a.Name == p2p_kern].Throughput.mean())
                    else:
                        out_list.append(-1)

                    for item, val in system_types.items():
                        print(a[a.Name == val[0]][item].median(), end=',')
                        out_list.append(a[a.Name == val[0]][item].median())

                    print("\n", end='')
                    if get_blocks:
                        val = int(a['Grid X'].max()) * int(a['Grid Y'].max())
                        out_list.append(val)
                    collect.append(out_list)
#                 except Exception as e:
#                     print("Error", e)
                except (Exception, ArithmeticError) as e:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(
                        exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(message)
    return collect


def generate_df(collected, col_names):
    collectdf = pd.DataFrame(data=collected, columns=col_names)
    collectdf.sort_values(by='inputSize', inplace=True)
    o = list(collectdf.inputSize.unique())
    print(o)
    o = [int(i) for i in o]
    o.sort()
    o = [str(i) for i in o]
    print(collectdf.columns)
    if 'set_freq' in col_names:
        collectdf.sort_values(by=['set_freq'], inplace=True)
    return collectdf
