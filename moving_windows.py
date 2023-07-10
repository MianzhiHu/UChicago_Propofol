import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Propofol.dfa import dfa

target_dir = 'data_clean'

# read hurst averages
hurst_averages = pd.read_csv('hurst_averages.csv')

# filter out rows with r_squared < 0.9
hurst_averages = hurst_averages[hurst_averages['r_squared'] < 0.9]

# print the "key" column and save it to a list
keys = []
for key in hurst_averages['key']:
    keys.append(key)

for key in keys:
    # delete the ".npy" extension
    key = key[:-4]
    # print(key)

def read_files(directory: str):
    for file in os.listdir(directory):
        # exclude only files that has elements of the list "keys" in their name
        if file.endswith(".npy") and not any(key in file for key in keys):
            # load data as numpy 2d array
            array_2d = np.load(os.path.join(target_dir, file))
            yield array_2d, file



def read_files_awake(directory: str):
    for file in os.listdir(directory):
        if file.endswith("_01_LPI_000.netts") and not file.startswith("02CB_01_movie") and not file.startswith("08BC_01_movie") and not file.startswith("19AK_01_rest") and not file.startswith("22CY_01_movie"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')
            yield array_2d, file


def read_files_mild(directory: str):
    for file in os.listdir(directory):
        if file.endswith("_02_LPI_000.netts") and not file.startswith("04HD_01_movie") and not file.startswith("10JR_01_movie") and not file.startswith("22CY_01_movie"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')
            yield array_2d, file


def read_files_deep(directory: str):
    for file in os.listdir(directory):
        if file.endswith("_03_LPI_000.netts") and not file.startswith("04HD_01_movie") and not file.startswith("13CA_01_movie") and not file.startswith("17NA_01_movie"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')
            yield array_2d, file


def read_files_recovery(directory: str):
    for file in os.listdir(directory):
        if file.endswith("_04_LPI_000.netts") and not file.startswith("17EK_01_movie") and not file.startswith("17NA_01_movie") and not file.startswith("25JK_01_movie") and not file.startswith("30AQ_01_movie"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')
            yield array_2d, file



def dfa_process(dfa_results):
    split_h = len(dfa_results[0])
    # turn results into a 2d array, where each column is a key
    hurst = np.hsplit(np.array(np.array(dfa_results[0])), split_h)
    cis0 = np.array([item[0] for item in dfa_results[1]])
    cis1 = np.array([item[1] for item in dfa_results[1]])
    cis0 = np.hsplit(cis0, split_h)
    cis1 = np.hsplit(cis1, split_h)
    r_squared = np.hsplit(np.array(np.array(dfa_results[2])), split_h)
    array_new = np.hstack((hurst, cis0, cis1, r_squared))
    df = pd.DataFrame(array_new, columns=['hurst', 'cis0', 'cis1', 'rsquared'])
    return df


def preprocess():
    tr: int = 2  # 2000 ms
    max_frequency = 0.1
    min_frequency = 0.01
    mn = int(np.ceil(1 / (tr * max_frequency)))
    mx = int(np.floor(1 / (tr * min_frequency)))
    print(f'mn: {mn}, mx: {mx}')
    results = {}
    processed_counter = 0
    files_processed = []

    for array_2d, file in read_files(directory=target_dir):
        processed_counter += 1
        files_processed.append(file)
        print(df := pd.DataFrame(array_2d))


        def sliding_window_generator(dataframe: pd.DataFrame, window_size: int, step_size: int):
            """
            dataframes: 2d arrays from netts file
            window_size: number of columns to be included in each window
            step_size: number of columns to be skipped between each window
            number of sliding windows = (len(dataframe) - window_size) / step_size + 1
            """
            for i in range(0, dataframe.shape[1] - window_size + 1, step_size):
                yield dataframe.iloc[:, i:i + window_size]

        for window_number, df_slice in enumerate(sliding_window_generator(df, 60, step_size=1)):
            print(f'working on window {window_number} of file {file}')
            slice_dfa_result = dfa(x=df_slice.to_numpy().T, max_window_size=mx, min_window_size=mn,
                                     return_confidence_interval=True,
                                     return_windows=False)

            rec_result = dfa_process(dfa_results=slice_dfa_result)
            hurst = rec_result['hurst']

            results.setdefault(window_number, []).append(hurst)

        print(f'processing done for {file}')
        print('filling in missing values for counter = ', processed_counter)
        for k, v in results.items():
            if len(v) < processed_counter:
                results[k].append(np.nan)
        print(results)

    print('finished processing all files, saving to pickle')
    with open('full_hurst.pickle', 'wb') as outfile:
        pickle.dump([results, files_processed], outfile)
    print('done')


# preprocess()


if __name__ == '__main__':
    with open('full_hurst.pickle', 'rb') as infile:
        results, files = pickle.load(infile)
        counter = 0
        df_early = pd.DataFrame()  # create an empty DataFrame
        df_mid = pd.DataFrame()  # create an empty DataFrame
        df_late = pd.DataFrame()  # create an empty DataFrame
        # # select only files containing the string "movie_03"
        for file in files:
            if "movie_03" in file:
                counter += 1
                # for the selected files, only take the first 20 windows
                for window in results[counter][:19]:
                    print(window)
                    window_df_early = pd.DataFrame(window)
                    df_early = pd.concat([df_early, window_df_early], axis=1) # append window_df to df
                for window in results[counter][40:59]:
                    print(window)
                    window_df_late = pd.DataFrame(window)
                    df_late = pd.concat([df_late, window_df_late], axis=1)
        # for file in files:
        #     if "movie_02" in file:
        #         counter += 1
        #         # for the selected files, only take the first 20 windows
        #         for window in results[counter][:5]:
        #             print(window)
        #             window_df_early = pd.DataFrame(window)
        #             df_early = pd.concat([df_early, window_df_early], axis=1)
        #         for window in results[counter][41:46]:
        #             print(window)
        #             window_df_late = pd.DataFrame(window)
        #             df_late = pd.concat([df_late, window_df_late], axis=1)
        # for file in files:
        #     if "movie_01" in file:
        #         counter += 1
        #         # for the selected files, only take the first 20 windows
        #         for window in results[counter][:10]:
        #             print(window)
        #             window_df_early = pd.DataFrame(window)
        #             df_early = pd.concat([df_early, window_df_early], axis=1)
        #         for window in results[counter][40:50]:
        #             print(window)
        #             window_df_mid = pd.DataFrame(window)
        #             df_mid = pd.concat([df_mid, window_df_mid], axis=1)
        #         for window in results[counter][79:89]:
        #             print(window)
        #             window_df_late = pd.DataFrame(window)
        #             df_late = pd.concat([df_late, window_df_late], axis=1)
        df_early = df_early.T
        # df_mid = df_mid.T
        df_late = df_late.T
        # print(df_early)
        # print(df_mid)
        # print(df_late)
        # # save the DataFrame to a csv file
        # df_early.to_csv('pls_movie_03_early.csv', index=False, header=False)
        # df_mid.to_csv('pls_movie_03_mid.csv', index=False, header=False)
        # df_late.to_csv('pls_movie_03_late.csv', index=False, header=False)













