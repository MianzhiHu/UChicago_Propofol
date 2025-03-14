import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from partial_least_squares import awake


def sliding_window_pls(dataframe: pd.DataFrame, window_size: int, step_size: int):
    """
    dataframes: 2d arrays from netts file
    window_size: number of columns to be included in each window
    step_size: number of columns to be skipped between each window
    number of sliding windows = (len(dataframe) - window_size) / step_size + 1
    """
    for i in range(0, dataframe.shape[0] - window_size + 1, step_size):
        yield dataframe.iloc[i:i + window_size, :]

if __name__ == '__main__':
    with open('full_hurst.pickle', 'rb') as f:
        results_duo = pickle.load(f)
        results_df_complete = pd.DataFrame.from_dict(results_duo[0], orient='index')
        results_df_complete.columns = results_duo[1]
        results_df_90 = results_df_complete.head(90)
        results_df_90_filtered = results_df_90[results_df_90.columns[results_df_90.columns.str.contains('rest_01')]]
        results_df_90_filtered_02 = results_df_90[results_df_90.columns[results_df_90.columns.str.contains('rest_02')]]
        results_df_90_filtered_03 = results_df_90[results_df_90.columns[results_df_90.columns.str.contains('rest_03')]]
        results_df_90_filtered_04 = results_df_90[results_df_90.columns[results_df_90.columns.str.contains('rest_04')]]
        # # identify cells with NaN values
        # results_df_90_filtered.isnull().any()
        # # print the column 22TK_01_movie_03_LPI_000.npy22TK_01_movie_03_LPI_000.npy
        # results_df_90_filtered['22TK_01_movie_03_LPI_000.npy']
        # # print all the values in the column 22TK_01_movie_03_LPI_000.npy22TK_01_movie_03_LPI_000.npy
        # results_df_90_filtered['22TK_01_movie_03_LPI_000.npy'].values
        windows = sliding_window_pls(results_df_90_filtered, 10, 1)
        window_unnested_list = []
        for window in windows:
            # unnest the window so that values in each cell are represented as a row (for 10x8 matrix, 80 rows)
            window_unnested = window.unstack().reset_index()
            # drop the rows if the whole row is NaN
            window_unnested = window_unnested.drop(window_unnested.columns[[0, 1]], axis=1)
            window_unnested = window_unnested[0].apply(pd.Series)
            window_unnested = window_unnested.dropna(axis=0, how='all')
            window_unnested_list.append(window_unnested)

        # concatenate all the windows_unnested into one dataframe
        window_unnested_df = pd.concat(window_unnested_list, axis=0)

        # get the columns with NaN values
        nan_columns = window_unnested_df.columns[window_unnested_df.isnull().any()].tolist()

        # drop columns with NaN values from all the windows_unnested
        window_unnested_list = [window_unnested.drop(nan_columns, axis=1) for window_unnested in window_unnested_list]

        # for idx, window_unnested in enumerate(window_unnested_list):
        #     # save window_unnested to a CSV file with the window number in the filename
        #     filename = '/content/drive/MyDrive/Data/windows_pls/window_{0}.csv'.format(idx)
        #     window_unnested.to_csv(filename, index=False, header=False)
        #     print('{0} saved'.format(filename))


# now I want the last window of each session in the resting condition
def find_last_window(dataframe: pd.DataFrame):
    # select the last window of each session
    last_window = dataframe.tail(1).dropna(axis=1, how='all')
    # unnest the last window so that values in each cell are represented as a row (for 10x8 matrix, 80 rows)
    last_window_unnested = last_window.unstack().unstack().reset_index()
    # rename the columns
    last_window_unnested.columns = ['index', 'stacked_hurst']
    last_window_unnested = last_window_unnested.explode('stacked_hurst')
    # make rows with the same index into separate columns
    last_window_unnested['columns'] = last_window_unnested.groupby('index').cumcount()
    last_window_clean = last_window_unnested.pivot(index='index', columns='columns', values='stacked_hurst')
    return last_window_clean


last_window_awake = find_last_window(results_df_90_filtered)
last_window_mild = find_last_window(results_df_90_filtered_02)
last_window_deep = find_last_window(results_df_90_filtered_03)
last_window_recovery = find_last_window(results_df_90_filtered_04)

# # remove the columns within effect_of_movie_missing (only for the awake movie vs awake rest comparison)
# last_window_awake_clean = last_window_awake.drop(columns=effect_of_movie_missing)
# print(last_window_awake_clean.shape)

# concatenate all the last windows into one dataframe
last_window_all = pd.concat([last_window_awake, last_window_mild, last_window_deep, last_window_recovery], axis=0)
last_window_all.to_csv('./data_generated/last_60_TR_all.csv')

# save a copy of average
last_window_average = last_window_all.copy()

# take the average
last_window_average = last_window_average.mean(axis=1)

# save as a CSV file
last_window_average.to_csv('./data_generated/last_60_TR_average.csv')

# record the columns being dropped
dropped_columns = last_window_all.columns[last_window_all.isnull().any()].tolist()

# # save the dropped columns to a .npy file
# np.save('./data_generated/last_60_TR_rest_missing.npy', dropped_columns)

# drop the columns with NaN values
last_window_all = last_window_all.dropna(axis=1, how='any')

# separate the concatenated dataframe back into the last windows of each session
last_window_awake_clean = last_window_all.iloc[0:9, :]
last_window_mild_clean = last_window_all.iloc[9:16, :]
last_window_deep_clean = last_window_all.iloc[16:22, :]
last_window_recovery_clean = last_window_all.iloc[22:31, :]

# save the last windows to CSV files
last_window_awake_clean.to_csv('./data_generated/last_window_awake.csv', index=False, header=False)
last_window_mild_clean.to_csv('./data_generated/last_window_mild.csv', index=False, header=False)
last_window_deep_clean.to_csv('./data_generated/last_window_deep.csv', index=False, header=False)
last_window_recovery_clean.to_csv('./data_generated/last_window_recovery.csv', index=False, header=False)

# # plot the boxplot of significant nodes
# nodes_last_window = np.load('./data_generated/nodes_with_hurst_values_last_60_TR.npy').tolist()
# awake_nodes = last_window_awake[nodes_last_window].mean()
# mild_nodes = last_window_mild[nodes_last_window].mean()
# deep_nodes = last_window_deep[nodes_last_window].mean()
# recovery_nodes = last_window_recovery[nodes_last_window].mean()
#
# plt.boxplot([awake_nodes, mild_nodes, deep_nodes, recovery_nodes])
# plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
# plt.ylabel('Hurst exponent')
# plt.xlabel('State')
# plt.title('H of significant nodes in the last window of resting state sessions')
# plt.show()

# nodes_effect_of_movie = np.load('./data_generated/nodes_with_hurst_values_effect_of_movie.npy').tolist()
# awake_rest_nodes = last_window_awake[nodes_effect_of_movie].mean()
# awake_movie_nodes = awake[nodes_effect_of_movie].mean()
#
# plt.boxplot([awake_rest_nodes, awake_movie_nodes])
# plt.xticks([1, 2], ['Rest', 'Narrative Listening'])
# plt.ylabel('Hurst exponent')
# plt.xlabel('Task')
# plt.title('Hurst exponent of significant nodes in the movie versus rest comparison')
# plt.show()






























