import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.f2py.crackfortran import skipfuncs

from dfa import dfa

target_dir = 'data_clean'


def read_files(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".netts"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')

            # Only keep 90-150 rows
            array_2d = array_2d[:, 90:150]
            yield array_2d, file

# for array_2d, file in read_files(directory=target_dir):
#     print(array_2d.shape)


def preprocess():
    tr: int = 2  # 2000 ms
    max_frequency = 0.1
    min_frequency = 0.01
    mn = int(np.ceil(1 / (tr * max_frequency)))
    mx = int(np.floor(1 / (tr * min_frequency)))
    counter = 0
    results_dict = {}
    for array_2d, file in read_files(directory=target_dir):
        dfa_results = dfa(x=array_2d.T, max_window_size=mx, min_window_size=mn, return_confidence_interval=True,
                          return_windows=False)
        split_h = len(dfa_results[0])
        # turn results into a 2d array, where each column is a key
        hurst = np.hsplit(np.array(np.array(dfa_results[0])), split_h)
        print(dfa_results[0])
        print(dfa_results[1])
        print(f'length of dfa_results[0]: {len(dfa_results[0])}')
        print(f'length of dfa_results[1]: {len(dfa_results[1])}')
        cis0 = np.array([item[0] for item in dfa_results[1]])
        cis1 = np.array([item[1] for item in dfa_results[1]])
        cis0 = np.hsplit(cis0, split_h)
        cis1 = np.hsplit(cis1, split_h)
        r_squared = np.hsplit(np.array(np.array(dfa_results[2])), split_h)

        array_new = np.hstack((hurst, cis0, cis1, r_squared))
        df = pd.DataFrame(array_new, columns=['hurst', 'cis0', 'cis1', 'r_squared'])
        results_dict[file] = df
        counter += 1

        print(f'processing done for {file}')

    print('finished processing all files, saving to pickle')
    with open('outcome_60TR.pickle', 'wb') as outfile:
        pickle.dump(results_dict, outfile)


preprocess()

# ======================================================================================================================
# pickle_to_read = './pickles/outcome_268.pickle'
# pickle_to_read = './pickles/outcome_60TR.pickle'
pickle_to_read = './pickles/fc_dict.pickle'
# pickle_to_read = './pickles/fc_dict_last_60_TR.pickle'

with open(pickle_to_read, 'rb') as f:
    results_dict = pickle.load(f)
    for key, value in results_dict.items():
        # check how many files contain the string 'rest_01_LPI'
        key_1 = [key for key in results_dict.keys() if 'rest_01_LPI' in key]
        key_2 = [key for key in results_dict.keys() if 'rest_02_LPI' in key]
        key_3 = [key for key in results_dict.keys() if 'rest_03_LPI' in key]
        key_4 = [key for key in results_dict.keys() if 'rest_04_LPI' in key]
        key_5 = [key for key in results_dict.keys() if 'movie_01_LPI' in key]
        key_6 = [key for key in results_dict.keys() if 'movie_02_LPI' in key]
        key_7 = [key for key in results_dict.keys() if 'movie_03_LPI' in key]
        key_8 = [key for key in results_dict.keys() if 'movie_04_LPI' in key]

        # save all results as separate csv
        if '.npy' in key:
            key_new = key.replace('.npy', '')

        value_df = pd.DataFrame(value)
        # value_df.to_csv(f'./data_generated/Hurst_mixed/{key_new}.csv', index=False)
        if 'movie' in key:
            value_df.to_csv(f'./data_generated/FC_mixed/{key_new}.csv', index=False)


    # print the number of files
    print(f'Number of files containing rest_01_LPI is {len(key_1)}')
    print(f'Number of files containing rest_02_LPI is {len(key_2)}')
    print(f'Number of files containing rest_03_LPI is {len(key_3)}')
    print(f'Number of files containing rest_04_LPI is {len(key_4)}')
    print(f'Number of files containing movie_01_LPI is {len(key_5)}')
    print(f'Number of files containing movie_02_LPI is {len(key_6)}')
    print(f'Number of files containing movie_03_LPI is {len(key_7)}')
    print(f'Number of files containing movie_04_LPI is {len(key_8)}')


# ======================================================================================================================
# Update: Rewrite the function to read and filter data files
# ======================================================================================================================
class AggregatedDataGenerator:
    def __init__(self, file_path: str, data_type: str):
        """
        Initialize the AggregatedDataGenerator with the path to the directory
        containing the CSV files.
        """
        self.file_path = file_path
        self.data_type_dict = {
            'hurst': self.process_hurst,
            'fc': self.process_fc,
            'edges': self.process_edges
        }
        self.process_module = self.data_type_dict[data_type]

    def save_results(self, condition_data: dict, missing_columns: list, output_path: str):
        """
        Save the results to csv files.

        Parameters:
            results_dict (dict): Dictionary with condition keys and their respective DataFrames.
            missing_columns (list): List of column names across all data that have missing values.
            missing_columns_per_condition (dict): Dictionary with condition keys and a list of
                                                  missing column names for each.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Save each condition's DataFrame to a separate CSV file
        for condition, df in condition_data.items():
            output_file = os.path.join(output_path, f"{condition}_concatenated.csv")
            df.to_csv(output_file, index=True)
            print(f"Saved {output_file}")

        # Save the missing columns to a CSV file
        missing_df = pd.DataFrame(missing_columns, columns=["missing_column"])
        missing_file = os.path.join(output_path, "missing_columns.csv")
        missing_df.to_csv(missing_file, index=False)
        print(f"Saved {missing_file}")

    def process_hurst(self, full_path: str, key: str, filename: str):
        df = pd.read_csv(full_path, usecols=[0]).T
        df.index = [f"{key}_{os.path.splitext(filename)[0]}"]
        return df

    def process_fc(self, full_path: str, key: str, filename: str):
        df = pd.read_csv(full_path)
        df = df.replace(2.0, pd.NA)
        # # Take the absolute value before averaging FC (positive and negative FC would be retained as absolute FC strength)
        # df = df.abs()
        col_means = df.mean(skipna=True)
        means_df = col_means.to_frame().T
        means_df.index = [f"{key}_{os.path.splitext(filename)[0]}"]
        # Take the absolute value after averaging FC (positive and negative FC would be averaged out)
        means_df = means_df.abs()
        return means_df

    def process_edges(self, full_path: str, key: str, filename: str):
        df = pd.read_csv(full_path)
        arr = df.to_numpy()
        n = arr.shape[0]
        # Extract upper triangle indices (excluding the main diagonal)
        triu_indices = np.triu_indices(n, k=1)
        edges = arr[triu_indices]
        edges_df = pd.DataFrame([edges])
        edges_df = edges_df.abs()
        edges_df.index = [f"{key}_{os.path.splitext(filename)[0]}"]
        return edges_df

    def process(self, keys: list, output_path: str):
        condition_data = {}

        # Process each condition.
        for key in keys:
            dfs = []
            for file in os.listdir(self.file_path):
                if key in file and file.endswith('.csv'):
                    full_path = os.path.join(self.file_path, file)
                    df = self.process_module(full_path, key, file)
                    dfs.append(df)
            condition_data[key] = pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

        # Aggregate data across conditions.
        all_dfs = [df for df in condition_data.values() if not df.empty]
        all_data = pd.concat(all_dfs, axis=0) if all_dfs else pd.DataFrame()
        missing_columns = all_data.columns[all_data.isna().any()].tolist() if not all_data.empty else []

        # Remove columns with missing values from each condition.
        if missing_columns:
            columns_to_keep = all_data.columns[~all_data.isna().any()]
            for key, df in condition_data.items():
                condition_data[key] = df[columns_to_keep] if not df.empty else pd.DataFrame()

        # Save the results to CSV files
        self.save_results(condition_data, missing_columns, output_path)

        return condition_data, missing_columns


# Example Usage
# Define the input file paths
hurst_file_path = './data_generated/Hurst_mixed/'
fc_file_path = './data_generated/FC_mixed/'

# Define the output file path
output_path = './data_generated/Hurst_csv_output'

# Select from the following:
# movie_01: Narrative-Listening Awake
# movie_02: Narrative-Listening Mild Sedation
# movie_03: Narrative-Listening Deep Sedation
# movie_04: Narrative-Listening Recovery
# rest_01: Resting-State Awake
# rest_02: Resting-State Mild Sedation
# rest_03: Resting-State Deep Sedation
# rest_04: Resting-State Recovery
# key = ['movie_01', 'movie_02', 'movie_03', 'movie_04', 'rest_01', 'rest_02', 'rest_03', 'rest_04'] # This is for 8-Condition Contrast
key = ['movie_01', 'rest_01']  # This is for 2-Condition Contrast

# Create an instance of the AggregatedDataGenerator
# Data type can be 'hurst', 'fc', or 'edges'
agg_gen = AggregatedDataGenerator(fc_file_path, 'hurst')

# Process the data
condition_data, missing_all = agg_gen.process(key, output_path)

print("Data for each condition:")
for key, df in condition_data.items():
    print(f"\nCondition: {key}")
    print(df.head())

print("\nMissing columns across all data:", missing_all)


# ======================================================================================================================
# Try to revert last window
# ======================================================================================================================
last_window_all = pd.read_csv('./data_generated/last_60_TR_all.csv', index_col=0)
last_window_awake = pd.read_csv('./data_generated/last_window_awake.csv', header=None)
last_window_mild = pd.read_csv('./data_generated/last_window_mild.csv', header=None)
last_window_deep = pd.read_csv('./data_generated/last_window_deep.csv', header=None)
last_window_recovery = pd.read_csv('./data_generated/last_window_recovery.csv', header=None)


# if rest is in the index, save it as a separate csv
rest_files = [file for file in last_window_all.index if 'rest' in file]
for file in rest_files:
    df = last_window_all.loc[file]
    df.columns = ['Hurst']
    df.to_csv(f'./data_generated/Hurst_last_window/{file}.csv', index=False)

# print the number of files in a directory
dir = './data_generated/Hurst_mixed/'

print(f'Number of files in {dir} is {len([file for file in os.listdir(dir) if file.endswith(".csv")])}')


def read_files(directory: str):
    counter = 0
    for file in os.listdir(directory):
        if file.endswith(".netts") and 'rest' in file:
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')

            # Only keep 90-150 rows
            array_2d = array_2d[:, 90:150]
            if array_2d.shape[1] == 60:
                counter += 1

            print(array_2d.shape)
            print(counter)


read_files(directory=target_dir)




