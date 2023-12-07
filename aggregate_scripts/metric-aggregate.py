######################
# Last Update: March 2023
# Author: Brandon Tran
#


import os
import csv
from warnings import resetwarnings
import pandas as pd
import time
import datetime
import pprint
import math

write_kernel_duration_files = True
summary_dict_filename = "app_metrics_summary.csv"
num_gpu = 4

# summary_aggregates = ['Min','Max','Avg']
trace_metrics = [ 
    'dram_utilization',
    'l2_utilization',
    'double_precision_fu_utilization',
    'single_precision_fu_utilization',
    'half_precision_fu_utilization',
    'tensor_precision_fu_utilization',
    'special_fu_utilization',
    'tex_fu_utilization',
    'stall_constant_memory_dependency',
    'stall_exec_dependency',
    'stall_inst_fetch',
    'stall_memory_dependency',
    'stall_memory_throttle',
    'stall_not_selected',
    'stall_other',
    'stall_pipe_busy',
    'stall_sync',
    'stall_texture',]
#    dram_utilization|l2_utilization|double_precision_fu_utilization|single_precision_fu_utilization|half_precision_fu_utilization|tensor_precision_fu_utilization|special_fu_utilization|tex_fu_utilization
#    dram_utilization|double_precision_fu_utilization|single_precision_fu_utilization|half_precision_fu_utilization|tensor_precision_fu_utilization|special_fu_utilization|tex_fu_utilization|l2_utilization|stall_constant_memory_dependency|stall_exec_dependency|stall_inst_fetch|stall_memory_dependency|stall_memory_throttle|stall_not_selected|stall_other|stall_pipe_busy|stall_sync|stall_texture

data_detail_dict = {
    # 'sgemm':{
    #     'perf_dir': '../POWER/20220614_SGEMM_TACC_25536',
    #     'metrics_dir': '../METRICS/sgemm/trace',
    # },
    # 'pagerank':{
    #     'perf_dir': '../POWER/20220614_PAGERANK_TACC',
    #     'metrics_dir': '../METRICS/pagerank/trace',
    # },
    # 'lammps':{
    #     'perf_dir': '../POWER/20220615_LAMMPS_TACC',
    #     'metrics_dir': '../METRICS/lammps/trace',
    # },
    # 'resnet_bs64':{
    #     'perf_dir': '../POWER/20230117_RESNET_CLOUDLAB_bs64',
    #     'metrics_dir': '../METRICS/resnet/cloudlab_multigpu_bs64',
    # },
    # 'resnet_bs16':{
    #     'perf_dir': '../POWER/20230117_RESNET_CLOUDLAB_bs16',
    #     'metrics_dir': '../METRICS/resnet/cloudlab_multigpu_bs16',
    # },
    'resnet_single_bs4':{
        'perf_dir': '../POWER/20230118_RESNET_CLOUDLAB_singleGPU_bs16_10iters',
        'metrics_dir': '../METRICS/resnet/cloudlab_single_bs16',
    },
    'bert_bs64':{
        'perf_dir': '../POWER/20230119_BERT_CLOUDLAB_bs64_50iters',
        'metrics_dir': '../METRICS/bert/trace-cloudlab-multi-50iter',
    },
    'bert_bs16':{
        'perf_dir': '../POWER/20230131_BERT_CLOUDLAB_bs16_50iters',
        'metrics_dir': '../METRICS/resnet/cloudlab_multigpu_bs16',
    },
    'bert_single_bs4':{
        'perf_dir': '../POWER/20230119_BERT_CLOUDLAB_single_10iters_bs16',
        'metrics_dir': '../METRICS/bert/trace-cloudlab-single-10iter-bs16',
    },
    'yolov7_bs32': {
        'perf_dir': '../POWER/20230214_YOLO_CLOUDLAB_bs32_50iters',
        'metrics_dir': '../METRICS/yolov7/multigpu_bs32_50iters'
    },
    'yolov7_bs64': {
        'perf_dir': '../POWER/20230307_YOLO_CLOUDLAB_bs64_50iters',
        'metrics_dir': '../METRICS/yolov7/multigpu_bs64_50iters'
    },
}


#
#     'resnet':{
#         'perf_dir': '../POWER/20220629_RESNET_TACC',
#         'metrics_dir': '../METRICS/resnet/trace',
#     },
#     'bert':{
#         'perf_dir': '../POWER/20220710_BERT_TACC_Shiv',
#         'metrics_dir': '../METRICS/bert/trace',
#     },
#     'resnet_single':{
#         'perf_dir': "../POWER/20221101_RESNET_single_gpu_Longhorn_5Iters",
#         'metrics_dir': "../METRICS/single_gpu_resnet/trace-5iters",
#     },
#     'bert_single':{
#         'perf_dir': "../POWER/20221101_BERT_single_gpu_Longhorn_5iters",
#         'metrics_dir': "../METRICS/bert/trace-single-gpu-5iter",
#     },
# }

def get_utilization(x):
    '''
    extracts integer from utilization value such as 'High (10)'
    if the ce is just a number, preserve it as a float

    parameters:
        -x: string
    returns:
        -val: integer extracted 
    '''
    pieces = str(x).split(" ")
    if len(pieces) > 1:
        try:
            val = int(pieces[-1].strip("()"))
        except TypeError:
            print(x)
            raise ValueError("Something went wrong")
    #Other values would be floats
    else:
        val = float(pieces[0])
    return val


def get_dataframe_from_metric_file(metric_file):
    list_of_lines = []
    with open(metric_file) as f:
        # Read the first non-header line as the header
        header_line = f.readline().strip()

        # Skip any additional header lines that start with "=="
        while header_line.startswith("=="):
            header_line = f.readline().strip()

        # Read the remaining CSV data
        reader = csv.reader(f)
        list_of_lines = list(reader)
    # Insert the header line at the beginning of the list
    if not list_of_lines:
        print("Error, empty file: " + metric_file)
        print("")
        return None
    titles = header_line.strip('"').split('","')
    # print(titles)
    data = list_of_lines[1:] #Skip over the extra line for units that we ignore
    df = pd.DataFrame(data=data, columns=titles)
    if 'resnet' in metric_file.lower():
        #The first kernel that appears once per iteration
        skip_list = df.index[df.Kernel.str.contains('volta_scudnn_128x64_relu_medium_nn_v1',regex=False)].tolist()
        #Include the one cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) before this skip target
        df = df.iloc[(skip_list[1]-1):]
    elif 'bert' in metric_file.lower():
        #The last kernel that appears once per iteration
        skip_list = df.index[df.Kernel.str.contains('at::native::amp_update_scale_cuda_kernel(float*, int*, float*, double, double, int)',regex=False)].tolist()
        #Begin after the first iteration
        df = df.iloc[(skip_list[0]+1):]
    elif 'yolo' in metric_file.lower():
        #A kernel that appears one every iteration (it does double in the first...)
        skip_list = df.index[df.Kernel.str.contains('volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nt',regex=False)].tolist()
        #Include the one cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) before this skip target
        df = df.iloc[(skip_list[1]-1):]
    #Get the device_id for the data
    devices = list(df.Device.unique())

    if len(devices) != 1:
        print("We are dealing with multiple devices! Selecting first")
    device = devices[0]
    GPU_split = device.split(' ')
    device_id = int(GPU_split[-1][1])
    return df, device_id

def filter_long_kernels_in_metric_file(metric_file,long_kernels):
    '''
    extracts dataframe out of a metric trace file
    if there are long kernels, filter data to only include those kernels

    parameters:
        -metric_file: path to metric trace file
        -long_kernels: list of kernel names that were considered long (in duration)
    returns:
        -good_df: dataframe from the metric_file with only the long_kernels
        -device_id: device that was measured in the metric_file
    '''
    df, device_id = get_dataframe_from_metric_file(metric_file)
    good_df = df[df['Kernel'].isin(long_kernels)].copy()
    # print(len(good_df))
    # print(good_df.Kernel.to_string(index=False))
    return  good_df,device_id


def clean_utilization_values(app,metric_device_id,exec_type,metric_df,kernel_dur_dict,exec_time):
    '''
    Calculate the weighted value for each metric
    Multiply the kernel value to the
    parameters:
        -df: dataframe extracted from a metric trace file
        -kernel_dur_dict: dictionary mapping kernel name to arithemtic mean kernel duration
    returns:
        -df: dataframe with weighted values for the metrics
        -found_metrics: list of metrics found in metric trace file
        -metric_stats: dictionary calculating simple stats about the metric trace file
    '''
    found_metrics = []
    metric_stats = {}
    metrics_found = [metric for metric in metric_df.columns if metric in trace_metrics]
    normalized_metric_dict = {}
    for metric in metrics_found:
            metric_df[metric] = metric_df[metric].apply(get_utilization)
            metric_stats[metric] = {}
            #print("getting "+metric+" from file")
            # print(df.filter(["Kernel",metric]))
            normalized_kernel_series = pd.Series(dtype='float64')
            for kernel in kernel_dur_dict.keys():
                # print("appylying {} tied to {} to {}".format(kernel_dur_dict[kernel],kernel,metric))
                to_update = (metric_df.Kernel == kernel) 
                # print("\n\nSTART OF COMPARISON FOR KERNEL: {}".format(kernel))              
                # pprint.pprint(metric_df.loc[to_update][metric] * kernel_dur_dict[kernel].values)#good_df.loc[to_update,[metric]])
                # print(len(metric_df.loc[to_update][metric]),len(kernel_dur_dict[kernel]))
                # In the event that the number of instances during a kernel duration
                metric_for_single_kernel = metric_df.loc[to_update][metric]
                kernel_duration_count = len(kernel_dur_dict[kernel])
                metric_kernel_count = len(metric_for_single_kernel)
                if kernel_duration_count != metric_kernel_count:
                    diff_filename = "{}_kernel_count_difference.csv".format(app)
                    if not os.path.isfile(diff_filename):          
                        with open(diff_filename, 'w') as diff_file:
                            print("app,device_id,exec_type,metric,kernel_name,perf_count,metric_count",file=diff_file)
                    with open(diff_filename, 'a+') as diff_file:
                        print('{},{},{},{},"{}",{},{}'.format(app,metric_device_id,exec_type,metric,kernel,kernel_duration_count,metric_kernel_count), file=diff_file)             

                if kernel_duration_count < metric_kernel_count:
                    normalized_kernel = metric_for_single_kernel[:kernel_duration_count].multiply(kernel_dur_dict[kernel].values)
                else:
                    normalized_kernel = metric_for_single_kernel.multiply(kernel_dur_dict[kernel].values[:metric_kernel_count])
                # print(normalized_kernel)
                normalized_kernel_series = pd.concat([normalized_kernel_series, normalized_kernel])
                metric_stats[metric][kernel] = {
                    "min": metric_df[metric_df.Kernel == kernel][metric].min(),
                    "max": metric_df[metric_df.Kernel == kernel][metric].max(),
                    "count": metric_df[metric_df.Kernel == kernel]["Correlation_ID"].count(),
                    "avg_ker_dur": kernel_dur_dict[kernel].mean(),
                    "weighted_value": normalized_kernel.sum(),
                    "final_contribution": normalized_kernel.sum()/exec_time
                }
            found_metrics.append(metric)
            normalized_metric_dict[metric] = normalized_kernel_series
    return pd.DataFrame(normalized_metric_dict),found_metrics,metric_stats


def get_execution_time_from_file(perf_file_path,app=None):
    '''
    Extracts execution_time and average kernel durations
    parameters:
        -perf_file_path: path to nvprof perf_file_path file
        -app: application of the perf_file_path (used to filter kernels)
    returns:
        -exec_time: sum of filtered kernel durations (normalized to milliseconds)
        -kernel_dur_dict: dictionary mapping kernel name to mean duration (normailized to millseconds)
    '''
    list_of_lines = []
    with open(perf_file_path) as f:
        # Read the first non-header line as the header
        header_line = f.readline().strip()

        # Skip any additional header lines that start with "=="
        while header_line.startswith("=="):
            header_line = f.readline().strip()

        # Read the remaining CSV data
        reader = csv.reader(f)
        list_of_lines = list(reader)
    # Insert the header line at the beginning of the list
    if not list_of_lines:
        print("Error, empty file: " + perf_file_path)
        print("")
        return (None,None,None)
    titles = header_line.strip('"').split('","')
    # print(titles)
    duration_unit = list_of_lines[0][1] #The first line is an additional line for units, Second column holds duration
    if duration_unit == 's':
        duration_factor = 1000
    elif duration_unit == 'ms':
        duration_factor = 1
    elif duration_unit == 'us':
        duration_factor = float(1.0/1000)
    elif duration_unit == 'ns':
        duration_factor = float(1.0/1000000)
    else:
        raise ValueError("The duration unit was not what we thought: {}".format(duration_unit))
    data = list_of_lines[1:]
    data = [d for d in data if len(d) == len(titles)]
    df = pd.DataFrame(data=data, columns=titles)
    #Drop unneeded columns from dataframe
    if 'Grid X' in df.columns:
        df = df.drop(columns=['Grid Y', 'Grid Z', 'Block X', 'Block Y', 'Block Z', 'Registers Per Thread', 'Static SMem',
                     'Dynamic SMem', 'Size', 'Throughput', 'SrcMemType', 'DstMemType', 'Context', 'Stream', 'Correlation_ID'])
    else:
        print('This file is probably empty: ' + perf_file_path)
        return (None,None,None)

    #Convert string value to numeric values
    df[["Start", "Duration"]] = df[["Start", "Duration"]].apply(pd.to_numeric)
    df.rename(columns={'Grid X': 'GridX'}, inplace=True)

    #General filter that removes all the sys_metrics (power, temp, freq) as well as [CUDA memcpy]/[CUDA memset]
    df = df[~df.Name.str.contains('[',regex=False)]

    #Filter out the kernels of interest
    if 'sgemm' in app.lower():
        kern = [i for i in df.Name.unique() if app in i][0]
        a = df[df.Name == kern]    
    elif 'pagerank' in app.lower():
        a = df[df.Name.str.contains('pagerank2') | df.Name.str.contains('spmv_csr_scalar_kernel') | df.Name.str.contains('inicsr') | df.Name.str.contains('inibuffer')]
    elif 'lammps' in app.lower(): 
        a = df[df.Name.str.contains('Kokkos')] ### & (df.Duration > 2.000)]
    elif 'resnet' in app.lower():
        a = df[(df.GridX.isnull() == False)] # & (~df.Name.str.contains('CatArrayBatchedCopy',regex=False)) ] ### & (df.Duration > 2.00)]
        #The first kernel that appears once per iteration
        skip_list = a.index[a.Name.str.contains('volta_scudnn_128x64_relu_medium_nn_v1',regex=False)].tolist()
        #Include the one cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) before this skip target
        a = a.iloc[(skip_list[1]-1):]
    elif 'bert' in app.lower():
        a = df[(df.GridX.isnull() == False)] # & (~df.Name.str.contains('CatArrayBatchedCopy',regex=False)) ] ### & (df.Duration > 2.00)]
        #The last kernel that appears once per iteration
        skip_list = a.index[a.Name.str.contains('at::native::amp_update_scale_cuda_kernel(float*, int*, float*, double, double, int)',regex=False)].tolist()
        #Begin after the first iteration
        a = a.iloc[(skip_list[0]+1):]
    elif 'yolo' in app.lower():
        a = df[(df.GridX.isnull() == False)] # & (~df.Name.str.contains('CatArrayBatchedCopy',regex=False)) ] ### & (df.Duration > 2.00)]
        #A kernel that appears one every iteration (it does double in the first...)
        skip_list = a.index[a.Name.str.contains('volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nt',regex=False)].tolist()
        #Include the one cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams) before this skip target
        a = a.iloc[(skip_list[1]-1):]
    else:
        print("Could not identify app {}, using default filter of checking GridX".format(app))
        a = df[(df.GridX.isnull() == False)]
    kernels = a.Name.unique()
    # pprint.pprint(kernels)
    kernel_dur_dict = {}

    for kernel in kernels:
        '''
        kernel_dur_dict[kernel] = df[df.Name == kernel].Duration.mean()
        #Converting seconds to miliseconds
        kernel_dur_dict[kernel] *= duration_factor
        '''
        #Setup have each kerne assigned a list instead 
        kernel_series = a[a.Name == kernel].Duration
        kernel_series *= duration_factor
        kernel_dur_dict[kernel] = kernel_series#list(kernel_series)
    # pprint.pprint(kernel_dur_dict)
    try:
        assert math.isclose(a.Duration.sum()*duration_factor,sum([sum(v) for (k,v) in kernel_dur_dict.items()]), rel_tol= 1e-4)
    except:
        print("The kernel dict and execution time don't match")
        return (None, None, None)

    #Determine the device id of the performance metric
    devices = list(a.Device.unique())
    if len(devices) != 1:
        print("We are dealing with multiple devices! Selecting first")
    device = devices[0] # Tesla V100-SXM2-16GB (3)
    GPU_split = device.split(' ')
    device_id = int(GPU_split[-1][1]) # 3

    return (a.Duration.sum())*duration_factor, kernel_dur_dict, device_id


def get_iter_dur(path,device=0):
    '''
    Extracts execution_time and average kernel durations
    parameters:
        -trace: path to nvprof trace file
        -app: application of the trace (used to filter kernels)
    returns: 
        (train_time, average_iter_dur)
            -train_time: the training time reported at the end of the file [in seconds], default to 0 if not found
            -averate_iter_dur: the average iteration duration (minus the first iteration) [in seconds]
    '''
    with open(path, 'r') as f:

        lines = f.readlines()

        iter_dur_vals = []
        final_data = []

        for line in lines:
            if 'Iteration' in line:
                datas = line.split('I')
                for data in datas:
                    if data == '':
                        continue
                    else:
                        x = data.split(' ')
                        if x[3] == str(device):
                            iter_dur_vals.append(float(x[5].split('\n')[0]))
            elif 'Training time' in line:
                datas = line.split(' ')
                train_time = datas[2].split('\n')[0].split('.')
                x = time.strptime(train_time[0].split(',')[0], '%H:%M:%S')
                seconds = datetime.timedelta(
                    hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                ms = '0.' + train_time[1]
                final_data.append(seconds + float(ms))
                break
        f.close()
        if len(final_data) == 0:
            print("Did not have total training time")
            final_data.append(0)

        #Remove the first iteration of data
        removed_value = iter_dur_vals.pop(0)
        print("Removing {} from iteration calculations".format(removed_value))

        exec_time = sum(iter_dur_vals) / len(iter_dur_vals)
        final_data.append(exec_time)
        return final_data


def single_aggregate(app, perf_dir, metrics_dir):
    '''
    Calculate normalized values for an application
    based on traces that gather values
    at the kernel granularity. Normalize the kernel
    values by the average kernel duration and then
    divide by the execution time.

    parameters:
        -app: string containing name of application
        -power_trace: path to power/performance trace with accurate kernel durations
        -metrics_dir: path to directory containing all metric traces
        -power_iterdur: path to iteration duration text file
    returns:
        -summary_dict: nested dictionary mapping metric to normalized value
        {
            app: {
                metric: value,
                metric: value,
                ...
            },
        }
        -metric_stats: nested dictionary mapping kernel stats
        { 
            app: {
                metric: {
                    "min":
                    "max":
                    "count":
                    "avg_ker_dur":
                    "weighted_value":
                    "final_contribution"
                },
        }
    '''
    app_exec_details = {}

    #Get execution time (sum kernel durations) and average kernel durations
    for perf_file_name in os.listdir(perf_dir):
        if str(perf_file_name).endswith(".csv"):
            perf_file_path = os.path.join(perf_dir,perf_file_name)
            kern_exec_time, kernel_dur_dict, device_id = get_execution_time_from_file(perf_file_path,app)
            if kern_exec_time is not None:
                app_exec_details[device_id] = {}
                app_exec_details[device_id]['kern_exec_time'] = kern_exec_time
                app_exec_details[device_id]['kernel_dur_dict'] = kernel_dur_dict
                print(app,device_id,"Kern_Exec_Time:",kern_exec_time)
        elif 'iterdur' in str(perf_file_name):
            if 'single' in app:
                true_num_gpus = 1
            else:
                true_num_gpus = num_gpu
            for iter_device_id in range(true_num_gpus):
                iterdur_file_path = os.path.join(perf_dir,perf_file_name)
                iter_exec_time = get_iter_dur(iterdur_file_path,iter_device_id)[1]*1000 #Gets running_time, sum_iteration_durations
                app_exec_details[iter_device_id]['iter_exec_time'] = iter_exec_time
                print(app,iter_device_id,"Iter_Exec_Time:",iter_exec_time)

    summary_dict = {}
    metric_stats = {}
    #Go through all metric csv
    for f in os.listdir(metrics_dir):
        if "csv" in f:
            filepath = os.path.join(metrics_dir,f)
            # print("Begin processing {}".format(filepath))
            #Extract metric data only for the kernels listed in our average kernel durations
            metric_data, metric_device_id = get_dataframe_from_metric_file(filepath)
            for exec_type in app_exec_details[metric_device_id]:
                if exec_type == 'kernel_dur_dict':
                    continue
                
                if 'resnet' in app or 'bert' in app or 'yolo' in app:
                    app_device = "{}_{}_{}".format(str(exec_type).replace('_exec_time',''),app,str(metric_device_id))
                else:
                    app_device = "{}_{}".format(app,str(metric_device_id))

                if app_device not in summary_dict:
                    summary_dict[app_device] = {}
                    metric_stats[app_device] = {}
                #Weigh metric values by the average kernel durations
                clean_data,found_metrics,metric_stat = clean_utilization_values(app,metric_device_id,exec_type,metric_data,app_exec_details[device_id]['kernel_dur_dict'],app_exec_details[device_id][exec_type])
                metric_stats[app_device].update(metric_stat)
                for metric in found_metrics:
                    #Sum the weighted kernel durations and then divide by the overall execution time
                    if "utilization" in metric:
                        summary_dict[app_device][metric] = clean_data[metric].sum()/app_exec_details[device_id][exec_type]
                    elif "transactions" in metric:
                        summary_dict[app_device][metric] = clean_data[metric].sum()
                    else: #stalls
                        summary_dict[app_device][metric] = clean_data[metric].sum()/app_exec_details[device_id][exec_type]
                summary_dict[app_device]['exec_time'] = app_exec_details[device_id][exec_type]
            # print("Finished processing {}".format(filepath))
    return summary_dict,metric_stats

def write_output_file(summary_dict,metric_dict):
    if write_kernel_duration_files:
        pass
        for app in metric_dict:
            kernel_stats_filename = "{}_kernel_duration_averages.csv".format(app)
            kernel_stats_file_path = os.path.join("..","output",kernel_stats_filename)
            if app[-1].isdigit():
                app_true = app[:-1]
            else:
                app_true = app
            try:
                with open(kernel_stats_file_path, 'w') as csvfile:
                    csvfile.write("kernel_name,metric,min,max,count,avg_ker_dur,weighted_value,final_contribution\n")
                    for metric in metric_dict[app]:
                        for kernel in metric_dict[app][metric]:
                            kernel_stats = metric_dict[app][metric][kernel]
                            csvfile.write('"{}",{},{},{},{},{},{},{}\n'.format(kernel,metric,kernel_stats["min"],kernel_stats['max'],kernel_stats['count'],kernel_stats["avg_ker_dur"],kernel_stats['weighted_value'],kernel_stats['final_contribution']))
            except IOError:
                print("I/O error: issue with getting {}".format(kernel_stats_filename))

    pprint.pprint(summary_dict)
    summary_dict_file_path = os.path.join("..","output",summary_dict_filename)
    try:
        with open(summary_dict_file_path, 'w') as csvfile:
            csvfile.write('idx,app_name,exec_time,'+','.join(trace_metrics)+'\n')
            idx = 0
            for app in summary_dict:
                app_data_row = "{},{},{},".format(idx,app,summary_dict[app]['exec_time'])
                for metric in trace_metrics:
                    app_data_row += str(summary_dict[app].get(metric,"-1"))
                    app_data_row += ','
                csvfile.write(app_data_row+'\n')
                idx += 1
    except IOError:
        print("I/O error: issue with getting {}".format(summary_dict_filename))

def main():
    if not os.path.isdir(os.path.join('..','output')):
        os.mkdir(os.path.join('..','output'))

    summary_dict = {}
    metric_dict = {}
    for app in data_detail_dict:
        app_summary_dict, metric_stats_dict = single_aggregate(app,data_detail_dict[app]['perf_dir'],data_detail_dict[app]['metrics_dir'])
        summary_dict.update(app_summary_dict)
        metric_dict.update(metric_stats_dict)
    pprint.pprint(summary_dict)
#    pprint.pprint(metric_dict)
    write_output_file(summary_dict,metric_dict)

if __name__ == "__main__":
    main()
