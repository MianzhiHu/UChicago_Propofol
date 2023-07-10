import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# with open('outcome_268.pickle', 'rb') as f:
#     results_dict = pickle.load(f)
#     counter = 0
#     rest = {}
#     for key, value in results_dict.items():
#         if 'rest' in key:
#             rest[key] = value
#             counter += 1
#
# with open('rest.pickle', 'wb') as outfile:
#     pickle.dump(rest, outfile)
#     print('files saved to pickle')


def pls_csv(pickle_name: str):
    with open(pickle_name, 'rb') as f:
        results_dict = pickle.load(f)
        counter = 0
        # To convert a dictionary to a list of tuples, use the following:
        list_of_tuples = [(key, value) for key, value in results_dict.items()]
        # discard all the subjects with mean r_squared < 0.9
        list_of_tuples = [item for item in list_of_tuples if item[1]['r_squared'].mean() > 0.9]
        # keep only the hurst values for the subjects with mean r_squared > 0.9
        list_of_tuples = [item[1]['hurst'] for item in list_of_tuples]
        # convert the list of hurst values to a numpy array
        hurst_array = np.array(list_of_tuples)
        # transpose the array so that each row is a subject
        hurst_array = hurst_array.T
        # convert the array to a dataframe
        hurst_df = pd.DataFrame(hurst_array.T)
        print(hurst_df)
        print(hurst_df.shape)
        # convert the dataframe to a csv file
        hurst_df.to_csv(f'pls_{pickle_name}.csv', index=False, header=False)
        print(f'pls_{pickle_name} saved to disk')

# pls_csv('rest.pickle')


# load the csv file
rest_awake = pd.read_csv('./data_generated/last_window_awake.csv', header=None)
awake = pd.read_csv('./data_generated/pls_movie_awake.csv', header=None)
mild = pd.read_csv('./data_generated/pls_movie_mild.csv', header=None)
deep = pd.read_csv('./data_generated/pls_movie_deep.csv', header=None)
recovery = pd.read_csv('./data_generated/pls_movie_recovery.csv', header=None)

awake_1 = pd.read_csv('./data_generated/pls_rest_awake.csv', header=None)
mild_1 = pd.read_csv('./data_generated/pls_rest_mild.csv', header=None)
deep_1 = pd.read_csv('./data_generated/pls_rest_deep.csv', header=None)
recovery_1 = pd.read_csv('./data_generated/pls_rest_recovery.csv', header=None)

# # remove the columns within effect_of_movie_missing (only for the awake movie vs awake rest comparison)
# awake_clean = awake.drop(columns=effect_of_movie_missing)
# print(awake_clean.shape)

# # load the column numbers that need to be kept
# nodes = np.load('./data_generated/nodes_with_hurst_values.npy', allow_pickle=True)
# nodes = nodes.tolist()
#
# # keep only the columns that are in the nodes list
# awake_pls_nodes = awake[nodes]
# mild_pls_nodes = mild[nodes]
# deep_pls_nodes = deep[nodes]
# recovery_pls_nodes = recovery[nodes]
#
# # take the mean value of each row
# awake_pls_nodes_mean = awake_pls_nodes.mean(axis=1)
# mild_pls_nodes_mean = mild_pls_nodes.mean(axis=1)
# deep_pls_nodes_mean = deep_pls_nodes.mean(axis=1)
# recovery_pls_nodes_mean = recovery_pls_nodes.mean(axis=1)

# # take the mean value of each column
# awake_pls_nodes_mean = awake_pls_nodes_mean.mean()
# mild_pls_nodes_mean = mild_pls_nodes_mean.mean()
# deep_pls_nodes_mean = deep_pls_nodes_mean.mean()
# recovery_pls_nodes_mean = recovery_pls_nodes_mean.mean()


# # plot a boxplot
# plt.boxplot([awake_pls_nodes_mean, mild_pls_nodes_mean, deep_pls_nodes_mean, recovery_pls_nodes_mean])
# plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
# plt.ylabel('Hurst Exponent')
# plt.savefig('./graphs/ave_hurst_movie.png', dpi=300)
# # set the size of the figure
# plt.figure(figsize=(10, 30))
# plt.show()

movie_03_early = pd.read_csv('./data_generated/pls_movie_03_early.csv', header=None)
movie_03_late = pd.read_csv('./data_generated/pls_movie_03_late.csv', header=None)

movie_03_early_30 = pd.read_csv('./data_generated/pls_movie_03_early_30.csv', header=None)
movie_03_late_30 = pd.read_csv('./data_generated/pls_movie_03_late_30.csv', header=None)

# # load the csv files
# nodes_03 = np.load('./data_generated/nodes_with_hurst_values_03.npy', allow_pickle=True)
# nodes_03 = nodes_03.tolist()
#
# movie_03_early = movie_03_early[nodes_03]
# movie_03_late = movie_03_late[nodes_03]
#
# movie_03_early_mean = movie_03_early.mean(axis=1)
# movie_03_late_mean = movie_03_late.mean(axis=1)
#
# movie_03_early_mean = movie_03_early_mean.mean()
# movie_03_late_mean = movie_03_late_mean.mean()
#
# x = ['early', 'late']
# y = [movie_03_early_mean, movie_03_late_mean]
# plt.bar(x, y)
# plt.title('Hurst values for PLS latent variable')
# plt.xlabel('state')
# plt.ylabel('Hurst value')
# plt.show()


movie_02_early = pd.read_csv('./data_generated/pls_movie_02_early.csv', header=None)
movie_02_late = pd.read_csv('./data_generated/pls_movie_02_late.csv', header=None)

# # load the csv files
# nodes_02 = np.load('./data_generated/nodes_with_hurst_values_02.npy', allow_pickle=True)
# nodes_02 = nodes_02.tolist()
#
# movie_02_early = movie_02_early[nodes_02]
# movie_02_late = movie_02_late[nodes_02]
#
# movie_02_early_mean = movie_02_early.mean(axis=1)
# movie_02_late_mean = movie_02_late.mean(axis=1)
#
# movie_02_early_mean = movie_02_early_mean.mean()
# movie_02_late_mean = movie_02_late_mean.mean()
#
# x = ['early', 'late']
# y = [movie_02_early_mean, movie_02_late_mean]
# plt.bar(x, y)
# plt.title('Hurst values for PLS latent variable')
# plt.xlabel('state')
# plt.ylabel('Hurst value')
# plt.show()

movie_01_early = pd.read_csv('./data_generated/pls_movie_01_early.csv', header=None)
movie_01_mid = pd.read_csv('./data_generated/pls_movie_01_mid.csv', header=None)
movie_01_late = pd.read_csv('./data_generated/pls_movie_01_late.csv', header=None)

# # load the csv files
# nodes_01 = np.load('./data_generated/nodes_with_hurst_values_01_3.npy', allow_pickle=True)
# nodes_01 = nodes_01.tolist()
#
# movie_01_early = movie_01_early[nodes_01]
# movie_01_mid = movie_01_mid[nodes_01]
#
# movie_01_early_mean = movie_01_early.mean(axis=1)
# movie_01_mid_mean = movie_01_mid.mean(axis=1)
#
# movie_01_early_mean = movie_01_early_mean.mean()
# movie_01_mid_mean = movie_01_mid_mean.mean()
#
# x = ['early', 'late']
# y = [movie_01_early_mean, movie_01_mid_mean]
# plt.bar(x, y)
# plt.title('Hurst values for PLS latent variable')
# plt.xlabel('state')
# plt.ylabel('Hurst value')
# plt.show()


# vertical stack the dataframes
df_movie = pd.concat([awake, mild, deep, recovery], axis=0)
# print(df_movie)
# print(awake.shape, mild.shape, deep.shape, recovery.shape)

df_combined = pd.concat([rest_awake, awake, mild, deep, recovery], axis=0)
# print(df_combined)

df_03 = pd.concat([movie_03_early, movie_03_late], axis=0)
# print(df_03)
# print(movie_03_early.shape, movie_03_late.shape)

df_03_30 = pd.concat([movie_03_early_30, movie_03_late_30], axis=0)
# print(df_03_30)
# print(movie_03_early_30.shape, movie_03_late_30.shape)

df_02 = pd.concat([movie_02_early, movie_02_late], axis=0)
# print(df_02)
# print(movie_02_early.shape, movie_02_late.shape)

df_01 = pd.concat([movie_01_early, movie_01_mid, movie_01_late], axis=0)
# print(df_01)
# print(movie_01_early.shape, movie_01_mid.shape, movie_01_late.shape)

df_everything = pd.concat([awake, mild, deep, recovery, awake_1, mild_1, deep_1, recovery_1], axis=0)
# print(df_everything)
# print(awake.shape, mild.shape, deep.shape, recovery.shape, awake_1.shape, mild_1.shape, deep_1.shape, recovery_1.shape)

# print columns with missing values and save them to a list
def find_missing_values(df):
    nodes_with_missing_values = []
    for column in df.columns:
        if df[column].isnull().values.any():
            # print(column)
            nodes_with_missing_values.append(column)
    return nodes_with_missing_values

df_movie_missing = find_missing_values(df_movie)
df_03_missing = find_missing_values(df_03)
df_03_30_missing = find_missing_values(df_03_30)
df_02_missing = find_missing_values(df_02)
df_01_missing = find_missing_values(df_01)
df_everything_missing = find_missing_values(df_everything)
df_combined_missing = find_missing_values(df_combined)

# remove the columns with missing values
df_everything = df_everything.dropna(axis=1)
# print(df_everything.shape)

df_movie = df_movie.dropna(axis=1)
# print(df_movie.shape)

df_combined = df_combined.dropna(axis=1)
# print(df_combined.shape)

df_03 = df_03.dropna(axis=1)
# print(df_03.shape)

df_03_30 = df_03_30.dropna(axis=1)
# print(df_03_30.shape)

# separate the dataframes into awake, mild, deep and recovery
# awake_clean = df.iloc[:12, :]
# mild_clean = df.iloc[12:24, :]
# deep_clean = df.iloc[24:32, :]
# recovery_clean = df.iloc[32:, :]
# print(awake_clean.shape, mild_clean.shape, deep_clean.shape, recovery_clean.shape)
# movie_awake_clean = df_everything.iloc[:12, :]
# movie_mild_clean = df_everything.iloc[12:24, :]
# movie_deep_clean = df_everything.iloc[24:32, :]
# movie_recovery_clean = df_everything.iloc[32:44,:]
# rest_awake_clean = df_everything.iloc[44:44+16, :]
# rest_mild_clean = df_everything.iloc[44+16:44+16+15, :]
# rest_deep_clean = df_everything.iloc[44+16+15:44+16+15+11, :]
# rest_recovery_clean = df_everything.iloc[44+16+15+11:44+16+15+11+16, :]
# print(movie_awake_clean.shape, movie_mild_clean.shape, movie_deep_clean.shape, movie_recovery_clean.shape)
# print(rest_awake_clean.shape, rest_mild_clean.shape, rest_deep_clean.shape, rest_recovery_clean.shape)

# rest_awake_clean = df_combined.iloc[:9, :]
# movie_awake_clean = df_combined.iloc[9:21, :]
# movie_mild_clean = df_combined.iloc[21:21+12, :]
# movie_deep_clean = df_combined.iloc[21+12:21+12+8, :]
# movie_recovery_clean = df_combined.iloc[21+12+8:21+12+8+12, :]


# early_clean = df_03_30.iloc[:160, :]
# late_clean = df_03_30.iloc[160:, :]

# early_clean = df_03.iloc[:152, :]
# late_clean = df_03.iloc[152:, :]

# early_clean = df.iloc[:120, :]
# mid_clean = df.iloc[120:240, :]
# late_clean = df.iloc[240:, :]

# # save the dataframes to csv files
# awake_clean.to_csv('pls_movie_awake_clean.csv', index=False, header=False)
# mild_clean.to_csv('pls_movie_mild_clean.csv', index=False, header=False)
# deep_clean.to_csv('pls_movie_deep_clean.csv', index=False, header=False)
# recovery_clean.to_csv('pls_movie_recovery_clean.csv', index=False, header=False)

# early_clean.to_csv('pls_movie_03_early_clean.csv', index=False, header=False)
# mid_clean.to_csv('pls_movie_01_mid_clean.csv', index=False, header=False)
# late_clean.to_csv('pls_movie_03_late_clean.csv', index=False, header=False)

# movie_awake_clean.to_csv('pls_movie_awake_everything.csv', index=False, header=False)
# movie_mild_clean.to_csv('pls_movie_mild_everything.csv', index=False, header=False)
# movie_deep_clean.to_csv('pls_movie_deep_everything.csv', index=False, header=False)
# movie_recovery_clean.to_csv('pls_movie_recovery_everything.csv', index=False, header=False)
# rest_awake_clean.to_csv('pls_rest_awake_everything.csv', index=False, header=False)
# rest_mild_clean.to_csv('pls_rest_mild_everything.csv', index=False, header=False)
# rest_deep_clean.to_csv('pls_rest_deep_everything.csv', index=False, header=False)
# rest_recovery_clean.to_csv('pls_rest_recovery_everything.csv', index=False, header=False)

# rest_awake_clean.to_csv('./data_generated/pls_rest_awake_combined.csv', index=False, header=False)
# movie_awake_clean.to_csv('./data_generated/pls_movie_awake_combined.csv', index=False, header=False)
# movie_mild_clean.to_csv('./data_generated/pls_movie_mild_combined.csv', index=False, header=False)
# movie_deep_clean.to_csv('./data_generated/pls_movie_deep_combined.csv', index=False, header=False)
# movie_recovery_clean.to_csv('./data_generated/pls_movie_recovery_combined.csv', index=False, header=False)


# # do the same for the rest files
# rest_awake = pd.read_csv('pls_rest_awake.csv', header=None)
# rest_mild = pd.read_csv('pls_rest_mild.csv', header=None)
# rest_deep = pd.read_csv('pls_rest_deep.csv', header=None)
# rest_recovery = pd.read_csv('pls_rest_recovery.csv', header=None)
#
# # vertical stack the dataframes
# df = pd.concat([rest_awake, rest_mild, rest_deep, rest_recovery], axis=0)
# print(df)
# print(rest_awake.shape, rest_mild.shape, rest_deep.shape, rest_recovery.shape)
#
# # print columns with missing values
# nodes_with_missing_values = df.columns[df.isnull().any()]
# print(nodes_with_missing_values)
#
# # remove the columns with missing values
# df = df.dropna(axis=1)
# print(df)
#
# # separate the dataframes into awake, mild, deep and recovery
# rest_awake_clean = df.iloc[:16, :]
# rest_mild_clean = df.iloc[16:31, :]
# rest_deep_clean = df.iloc[31:42, :]
# rest_recovery_clean = df.iloc[42:, :]
#
# print(rest_awake_clean.shape, rest_mild_clean.shape, rest_deep_clean.shape, rest_recovery_clean.shape)
#
# # save the dataframes to csv files
# rest_awake_clean.to_csv('pls_rest_awake_clean.csv', index=False, header=False)
# rest_mild_clean.to_csv('pls_rest_mild_clean.csv', index=False, header=False)
# rest_deep_clean.to_csv('pls_rest_deep_clean.csv', index=False, header=False)
# rest_recovery_clean.to_csv('pls_rest_recovery_clean.csv', index=False, header=False)

# # load the csv files
# movie = pd.read_csv('pls_movie.csv', header=None)
# rest = pd.read_csv('pls_rest.csv', header=None)
#
# # vertical stack the dataframes
# df = pd.concat([movie, rest], axis=0)
# print(df)
# print(movie.shape, rest.shape)
#
# # print columns with missing values
# nodes_with_missing_values = df.columns[df.isnull().any()]
# print(nodes_with_missing_values)
#
# # add 1 to the column index to get the node number
# nodes_with_missing_values = nodes_with_missing_values + 1
# print(nodes_with_missing_values)
#
# # remove the columns with missing values
# df = df.dropna(axis=1)
# print(df)
#
# # separate the dataframes back into movie and rest
# movie_clean = df.iloc[:44, :]
# rest_clean = df.iloc[44:, :]
# print(movie_clean.shape, rest_clean.shape)
#
# # save the dataframes to csv files
# movie_clean.to_csv('pls_movie_clean.csv', index=False, header=False)
# rest_clean.to_csv('pls_rest_clean.csv', index=False, header=False)

print('Partial_Least_Squares has been read')










