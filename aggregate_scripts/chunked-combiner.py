'''
    File name: gpu-aggregator.py
    Author: Prasoon Sinha
    Python Version: 3.7
'''
import os
import sys
import pandas as pd


def read_csv(chunked_dir):
    # For all the csv files in the directory, populate dictionary with the file name followed by its path
    files = []
    for f in os.listdir(chunked_dir):
        files.append(chunked_dir+'/'+f)
        # if raw_dir in f:
        #     files.append(chunked_dir + '/' + f)
    return files


def combine(chunked_files, chunked_dir, csv_name):
    # Combine all files in the dictionary into one dataframe
    combined_df = pd.concat([pd.read_csv(f) for f in chunked_files])
    # Create csv name
    csv_name = chunked_dir + '/' + csv_name
    # Export dataframe to csv
    combined_df.to_csv(csv_name, index=False, header=True)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 chunked_combiner.py [path to data directory] [final csv name, including '.csv']")
        exit()
    chunked_dir = sys.argv[1]
    csv_name = sys.argv[2]
    chunked_files = read_csv(chunked_dir)
    print(chunked_files)
    combine(chunked_files, chunked_dir, csv_name)


if __name__ == "__main__":
    main()
