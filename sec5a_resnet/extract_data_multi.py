import datetime
import time
import numpy as np
import pandas as pd
import os
import re

df_data = pd.DataFrame(columns=['exp', 'reps', 'cabinet', 'node', 'device', 'train_time', 'mean_iter_dur', 'med_iter_dur', 'max_iter_dur', 'min_iter_dur'])

def walk_through_files(directory, df):
    # Regular expression pattern to extract fields from the filename
    pattern = r"resnet_multi_iterdur_(\d+)_([a-zA-Z0-9-]+)_run_(\d+).txt"

    # Dictionary to store extracted data

    # Walking through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file name matches the pattern
            match = re.match(pattern, file)
            if match:
                #print("true")
                # Extract data
                iteration_duration, node_id, reps = match.groups()
                cabinet = node_id.split('-')[0]
                node = node_id.split('-')[1]
                model = 'resnet'
                device = 0
                for i in range(0,3):
                    all_data = []
                    device = i
                    all_data.append(model)
                    all_data.append(int(reps))
                    all_data.append(cabinet)
                    all_data.append(int(node))
                    all_data.append(int(device))
                    data_list = get_iter_dur(file, int(device))
                    all_data.extend(data_list)
                    row_to_append = pd.DataFrame(
                        [all_data], columns=df.columns)
                    df = pd.concat([df, row_to_append], ignore_index=True)

    return df


def get_iter_dur(path, device=0):
    with open(path, 'r') as f:

        lines = f.readlines()

        iter_dur_vals = []
        final_data = []
        max_dur = 0
        min_dur = 10000
        median_dur = 0

        for line in lines:
            if 'Iteration' in line:
                datas = line.split('I')
                for data in datas:
                    if data == '':
                        continue
                    else:
                        x = data.split(' ')
                        if x[3] == str(device):
                            this_iter = float(x[5].split('\n')[0])
                            iter_dur_vals.append(this_iter)
                            if this_iter > max_dur:
                                max_dur = this_iter
                            if this_iter < min_dur:
                                min_dur = this_iter
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

        # Remove the first iteration of data
        removed_value = iter_dur_vals.pop(0)
        # print("Removing {} from iteration calculations".format(removed_value))

        exec_time = sum(iter_dur_vals) / len(iter_dur_vals)
        median_dur = np.median(iter_dur_vals)
        final_data.append(exec_time)
        final_data.append(median_dur)
        final_data.append(max_dur)
        final_data.append(min_dur)
        return final_data


df_data = walk_through_files(
    '/work/09732/kchen346/ls6/gpu_variability_sc22_artifact/sec5a_resnet/data_multi', df_data)
print(df_data)
df_data.to_csv(path_or_buf="all_data.csv")

grouped = df_data.groupby(['exp', 'cabinet', 'node', 'device'])

# Compute the mean and median of 'mean_iter_dur' and 'train_time' for each group
aggregated_df = grouped.agg({
    'mean_iter_dur': ['mean', 'median'],
    'train_time': ['mean', 'median']
})
# Flatten the multi-level column names and create new names
aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]

print(aggregated_df)
aggregated_df.to_csv(path_or_buf="aggregated_data.csv")

