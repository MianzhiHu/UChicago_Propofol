import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.f2py.crackfortran import skipfuncs
from dfa import dfa
from pathlib import Path

target_dir = 'data_clean'


def read_files(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            # load data as numpy 2d array
            array_2d = np.load(os.path.join(target_dir, file))

            # # Only keep 90-150 rows
            # array_2d = array_2d[:, 90:150]
            yield array_2d, file


def preprocess():
    tr: int = 2  # 2000 ms
    max_frequency = 0.1
    min_frequency = 0.01
    mn = int(np.ceil(1 / (tr * max_frequency)))
    mx = int(np.floor(1 / (tr * min_frequency)))
    counter = 0
    results_dict = {}
    for array_2d_raw, file in read_files(directory=target_dir):
        if array_2d_raw.shape[1] < 60:
            print(f'skipping {file} because it has too few TRs')
            continue
        mask = ~(np.all((array_2d_raw == 0) | np.isnan(array_2d_raw), axis=0))
        array_2d = array_2d_raw[:, mask]
        print(f'array_2d raw shape: {array_2d_raw.shape}; array_2d shape: {array_2d.shape}')
        if array_2d.shape[1] < 50:
            print(f'skipping {file} because it has too few TRs after censoring')
            continue

        dfa_results = dfa(x=array_2d.T, max_window_size=mx, min_window_size=mn, return_confidence_interval=True,
                          return_windows=False)
        split_h = len(dfa_results[0])
        # turn results into a 2d array, where each column is a key
        hurst = np.hsplit(np.array(np.array(dfa_results[0])), split_h)
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
    with open('./data_generated/pickles/outcome_dict.pickle', 'wb') as outfile:
        pickle.dump(results_dict, outfile)


# preprocess()

# ======================================================================================================================
# # pickle_to_read = './data_generated/pickles/outcome_dict.pickle'
# # pickle_to_read = './data_generated/pickles/outcome_60TR.pickle'
# # pickle_to_read = './data_generated/pickles/fc_dict.pickle'
# pickle_to_read = './data_generated/pickles/fc_dict_last_60_TR.pickle'
# last_window = True
# i = 0
# low_r2 = []
#
# with open(pickle_to_read, 'rb') as f:
#     results_dict = pickle.load(f)
#     for key, value in results_dict.items():
#
#         # save all results as separate csv
#         key_new = key.replace('.npy', '')
#
#         value_df = pd.DataFrame(value)
#
#         # check whether the pickle file to read is hurst or fc
#         if 'outcome' in pickle_to_read:
#             # filter out data with low R²
#             mean_r2 = value_df['r_squared'].mean()
#             if mean_r2 < 0.90:
#                 i += 1
#                 low_r2.append(key)
#                 print(f"Skipping {key_new}: mean R² = {mean_r2:.3f} < 0.9")
#                 continue
#             folder_path = 'Hurst_mixed'
#         elif 'fc' in pickle_to_read:
#             folder_path = 'FC_mixed'
#
#         # value_df.to_csv(f'./data_generated/{folder_path}/{key_new}.csv', index=False)
#
#         if last_window:
#             if 'rest' in key:
#                 value_df.to_csv(f'./data_generated/{folder_path}/{key_new}.csv', index=False)
#         else:
#             if 'movie' in key:
#                 value_df.to_csv(f'./data_generated/{folder_path}/{key_new}.csv', index=False)
#
# print(f'{i} files skipped due to low R²')

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
            'Hurst': self.process_hurst,
            'FC': self.process_fc,
            'Edges': self.process_edges
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
hurst_file_path = './data_generated/Hurst/'
fc_file_path = './data_generated/FC/'

# Define the output file path
output_path = Path('./data_generated/Contrasts_Full/')

# Select from the following:
# movie_01: Narrative-Listening Awake
# movie_02: Narrative-Listening Mild Sedation
# movie_03: Narrative-Listening Deep Sedation
# movie_04: Narrative-Listening Recovery
# rest_01: Resting-State Awake
# rest_02: Resting-State Mild Sedation
# rest_03: Resting-State Deep Sedation
# rest_04: Resting-State Recovery

# your three data types and the corresponding file paths
data_types = {
    'Hurst': hurst_file_path,
    'FC':    fc_file_path,
    'Edges': fc_file_path
}

# your four “key‐lists” (you can name these anything you like)
key_sets = {
    '8_condition': ['movie_01','movie_02','movie_03','movie_04', 'rest_01','rest_02','rest_03','rest_04'], # This is for 8-Condition Contrast
    'Effects of Narrative-Listening': ['movie_01','rest_01'], # This is the effect of narrative-listening
    'Effects of Propofol': ['rest_01','rest_02','rest_03','rest_04'], # This is the effect of propofol
    'Effects of Propofol on Narrative-Listening': ['movie_01','movie_02','movie_03','movie_04'], # This is the effect of propofol on narrative-listening
}

all_results = {}

for dtype, file_path in data_types.items():
    agg = AggregatedDataGenerator(file_path, dtype)

    for set_name, keys in key_sets.items():
        # make a subfolder per dtype+set_name
        out_dir = output_path / set_name / dtype
        out_dir.mkdir(parents=True, exist_ok=True)

        # process & save
        condition_data, missing_all = agg.process(keys, str(out_dir))

        # store in memory if you need it later
        all_results[(dtype, set_name)] = (condition_data, missing_all)

        # report
        print(f" → {dtype} / {set_name}:")
        for k, df in condition_data.items():
            print(f"    • {k}: {len(df)} rows")
        print(f"    missing columns: {len(missing_all)}\n")

# ======================================================================================================================
# Check the direction of effects for PLS results
# ======================================================================================================================
# Print the average values for contrast for each condition
contrast_dir = './data_generated/Contrasts'

def calculate_condition_means(contrast_dir):
    """Calculate means for each condition within each measurement and contrast"""
    contrasts = [d for d in os.listdir(contrast_dir) if os.path.isdir(os.path.join(contrast_dir, d))]

    for contrast in contrasts:
        contrast_path = os.path.join(contrast_dir, contrast)
        measurements = [d for d in os.listdir(contrast_path) if os.path.isdir(os.path.join(contrast_path, d))]

        print(f"\nContrast: {contrast}")
        for measurement in measurements:
            measurement_path = os.path.join(contrast_path, measurement)
            print(f"\nMeasurement: {measurement}")

            for file in os.listdir(measurement_path):
                if file.endswith('.csv') and 'missing' not in file:
                    condition = file.split('_concatenated.csv')[0]
                    df = pd.read_csv(os.path.join(measurement_path, file), index_col=0, header=0)
                    mean_value = df.mean().mean()
                    print(f"Condition: {condition}, Mean: {mean_value:.4f}")


# Calculate means for each contrast/measurement/condition
calculate_condition_means(contrast_dir)




