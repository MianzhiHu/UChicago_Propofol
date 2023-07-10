import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_preparation import new_df_edges_movie, new_df_edges_rest, new_df_edges_effect_of_movie, \
    new_df_edges_rest_last_60_TR, new_df_edges_double_two_way, new_df_edges_rest_post_hoc
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from nilearn import plotting, datasets
import nibabel


def upper_triangle_flattened(mat):
    upper_triangular_mat = mat[np.triu_indices(mat.shape[0], k=1)]  # k=1 to exclude the main diagonal
    return upper_triangular_mat.flatten()


brain_nodes_indices = list(range(1, 269))
edges = [(brain_nodes_indices[i], brain_nodes_indices[j]) for i in range(len(brain_nodes_indices))
         for j in range(i + 1, len(brain_nodes_indices))]


hurst_nodes_rest = np.load('./data_generated/nodes_with_hurst_values_last_60_TR.npy')
fc_nodes_rest = np.load('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy')

# take the union set of the nodes
nodes = np.union1d(hurst_nodes_rest, fc_nodes_rest)

# get the combination of the nodes
edges_rest = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

with open('fc_dict_last_60_TR.pickle', 'rb') as f:
    fc_dict = pickle.load(f)

    # for all the files in the dictionary, only keep the upper triangle
    for key, value in fc_dict.items():
        flattened_matrix = upper_triangle_flattened(value)
        df = pd.DataFrame(({'edges': edges, 'values': flattened_matrix}))
        fc_dict[key] = df

    all_data = pd.concat(fc_dict, names=['key'])
    all_data.reset_index(inplace=True)
    pivoted_data = all_data.pivot(index='key', columns='edges', values='values')

    # edges_movie_awake = pivoted_data[pivoted_data.index.str.contains('movie_01_LPI')]
    # edges_movie_mild = pivoted_data[pivoted_data.index.str.contains('movie_02_LPI')]
    # edges_movie_deep = pivoted_data[pivoted_data.index.str.contains('movie_03_LPI')]
    # edges_movie_recovery = pivoted_data[pivoted_data.index.str.contains('movie_04_LPI')]
    # edges_rest_awake = pivoted_data[pivoted_data.index.str.contains('rest_01_LPI')]
    # edges_rest_mild = pivoted_data[pivoted_data.index.str.contains('rest_02_LPI')]
    # edges_rest_deep = pivoted_data[pivoted_data.index.str.contains('rest_03_LPI')]
    # edges_rest_recovery = pivoted_data[pivoted_data.index.str.contains('rest_04_LPI')]

    edges_rest_awake_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_01_LPI')]
    edges_rest_mild_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_02_LPI')]
    edges_rest_deep_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_03_LPI')]
    edges_rest_recovery_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_04_LPI')]

# only keep the edges if they are in the edges rest
edges_rest_awake_post_hoc = edges_rest_awake_last_60_TR[[col for col in edges_rest_awake_last_60_TR.columns if col in edges_rest]]
# print(edges_rest_awake_post_hoc.shape)
edges_rest_mild_post_hoc = edges_rest_mild_last_60_TR[[col for col in edges_rest_mild_last_60_TR.columns if col in edges_rest]]
# print(edges_rest_mild_post_hoc.shape)
edges_rest_deep_post_hoc = edges_rest_deep_last_60_TR[[col for col in edges_rest_deep_last_60_TR.columns if col in edges_rest]]
# print(edges_rest_deep_post_hoc.shape)
edges_rest_recovery_post_hoc = edges_rest_recovery_last_60_TR[[col for col in edges_rest_recovery_last_60_TR.columns if col in edges_rest]]
# print(edges_rest_recovery_post_hoc.shape)



def record_missing_edges(df_1, df_2, df_3, df_4, df_5=None):
    missing_edges = []
    if df_3 is None and df_4 is None:
        df_all = pd.concat([df_1, df_2])
        for i, col in enumerate(df_all.columns):
            if df_all[col].isnull().values.any():
                missing_edges.append(i)

        # now clean the data and return the cleaned data
        df_all = df_all.dropna(axis=1, how='any')
        # convert to absolute value
        df_all = np.abs(df_all)
        df_1 = df_all.iloc[:len(df_1), :]
        df_2 = df_all.iloc[len(df_1):, :]
        return df_1, df_2, missing_edges
    if df_4 is None:
        df_all = pd.concat([df_1, df_2, df_3])
        for i, col in enumerate(df_all.columns):
            if df_all[col].isnull().values.any():
                missing_edges.append(i)

        # now clean the data and return the cleaned data
        df_all = df_all.dropna(axis=1, how='any')
        # convert to absolute value
        df_all = np.abs(df_all)
        df_1 = df_all.iloc[:len(df_1), :]
        df_2 = df_all.iloc[len(df_1):len(df_1) + len(df_2), :]
        df_3 = df_all.iloc[len(df_1) + len(df_2):, :]
        return df_1, df_2, df_3, missing_edges
    if df_5 is not None:
        df_all = pd.concat([df_1, df_2, df_3, df_4, df_5])
        for i, col in enumerate(df_all.columns):
            if df_all[col].isnull().values.any():
                missing_edges.append(i)

        # now clean the data and return the cleaned data
        df_all = df_all.dropna(axis=1, how='any')
        # convert to absolute value
        df_all = np.abs(df_all)
        df_1 = df_all.iloc[:len(df_1), :]
        df_2 = df_all.iloc[len(df_1):len(df_1) + len(df_2), :]
        df_3 = df_all.iloc[len(df_1) + len(df_2):len(df_1) + len(df_2) + len(df_3), :]
        df_4 = df_all.iloc[len(df_1) + len(df_2) + len(df_3):len(df_1) + len(df_2) + len(df_3) + len(df_4), :]
        df_5 = df_all.iloc[len(df_1) + len(df_2) + len(df_3) + len(df_4):, :]
        return df_1, df_2, df_3, df_4, df_5, missing_edges
    else:
        df_all = pd.concat([df_1, df_2, df_3, df_4])
        for i, col in enumerate(df_all.columns):
            if df_all[col].isnull().values.any():
                missing_edges.append(i)

        # now clean the data and return the cleaned data
        df_all = df_all.dropna(axis=1, how='any')
        # convert to absolute value
        df_all = np.abs(df_all)
        df_1 = df_all.iloc[:len(df_1), :]
        df_2 = df_all.iloc[len(df_1):len(df_1) + len(df_2), :]
        df_3 = df_all.iloc[len(df_1) + len(df_2):len(df_1) + len(df_2) + len(df_3), :]
        df_4 = df_all.iloc[len(df_1) + len(df_2) + len(df_3):, :]
        return df_1, df_2, df_3, df_4, missing_edges


# edges_movie_awake_cleaned, edges_movie_mild_cleaned, edges_movie_deep_cleaned, edges_movie_recovery_cleaned, missing_edges_movie = record_missing_edges(
#     edges_movie_awake, edges_movie_mild, edges_movie_deep, edges_movie_recovery)
# edges_rest_awake_cleaned, edges_rest_mild_cleaned, edges_rest_deep_cleaned, edges_rest_recovery_cleaned, missing_edges_rest = record_missing_edges(
#     edges_rest_awake, edges_rest_mild, edges_rest_deep, edges_rest_recovery)
# edges_movie_awake_cleaned_1, edges_rest_awake_cleaned_1, missing_edges_effect_of_movie = record_missing_edges(
#     edges_movie_awake, edges_rest_awake_last_60_TR, None, None)
# edges_rest_awake_last_60_TR_cleaned, edges_rest_mild_last_60_TR_cleaned, edges_rest_deep_last_60_TR_cleaned, edges_rest_recovery_last_60_TR_cleaned, missing_edges_rest_last_60_TR = record_missing_edges(
#     edges_rest_awake_last_60_TR, edges_rest_mild_last_60_TR, edges_rest_deep_last_60_TR, edges_rest_recovery_last_60_TR)
# edges_double_rest_awake_cleaned, edges_double_movie_mild_cleaned, edges_double_movie_deep_cleaned, missing_edges_double = record_missing_edges(
#     edges_rest_awake_last_60_TR, edges_movie_mild, edges_movie_deep, None)
# edges_rest_awake_post_hoc_cleaned, edges_rest_mild_post_hoc_cleaned, edges_rest_deep_post_hoc_cleaned, edges_rest_recovery_post_hoc_cleaned, missing_edges_rest_post_hoc = record_missing_edges(
#     edges_rest_awake_post_hoc, edges_rest_mild_post_hoc, edges_rest_deep_post_hoc, edges_rest_recovery_post_hoc)

# # save the data
# edges_movie_awake_cleaned.to_csv('./data_generated/edges_movie_awake.csv', index=False, header=False)
# edges_movie_mild_cleaned.to_csv('./data_generated/edges_movie_mild.csv', index=False, header=False)
# edges_movie_deep_cleaned.to_csv('./data_generated/edges_movie_deep.csv', index=False, header=False)
# edges_movie_recovery_cleaned.to_csv('./data_generated/edges_movie_recovery.csv', index=False, header=False)

# edges_rest_awake_cleaned.to_csv('./data_generated/edges_rest_awake.csv', index=False, header=False)
# edges_rest_mild_cleaned.to_csv('./data_generated/edges_rest_mild.csv', index=False, header=False)
# edges_rest_deep_cleaned.to_csv('./data_generated/edges_rest_deep.csv', index=False, header=False)
# edges_rest_recovery_cleaned.to_csv('./data_generated/edges_rest_recovery.csv', index=False, header=False)

# edges_movie_awake_cleaned_1.to_csv('./data_generated/edges_movie_awake_1.csv', index=False, header=False)
# edges_rest_awake_cleaned_1.to_csv('./data_generated/edges_rest_awake_1.csv', index=False, header=False)

# edges_rest_awake_last_60_TR_cleaned.to_csv('./data_generated/edges_rest_awake_last_60_TR.csv', index=False, header=False)
# edges_rest_mild_last_60_TR_cleaned.to_csv('./data_generated/edges_rest_mild_last_60_TR.csv', index=False, header=False)
# edges_rest_deep_last_60_TR_cleaned.to_csv('./data_generated/edges_rest_deep_last_60_TR.csv', index=False, header=False)
# edges_rest_recovery_last_60_TR_cleaned.to_csv('./data_generated/edges_rest_recovery_last_60_TR.csv', index=False, header=False)

# edges_double_rest_awake_cleaned.to_csv('./data_generated/edges_double_rest_awake.csv', index=False, header=False)
# edges_double_movie_mild_cleaned.to_csv('./data_generated/edges_double_movie_mild.csv', index=False, header=False)
# edges_double_movie_deep_cleaned.to_csv('./data_generated/edges_double_movie_deep.csv', index=False, header=False)

# edges_rest_awake_post_hoc_cleaned.to_csv('./data_generated/edges_rest_awake_post_hoc.csv', index=False, header=False)
# edges_rest_mild_post_hoc_cleaned.to_csv('./data_generated/edges_rest_mild_post_hoc.csv', index=False, header=False)
# edges_rest_deep_post_hoc_cleaned.to_csv('./data_generated/edges_rest_deep_post_hoc.csv', index=False, header=False)
# edges_rest_recovery_post_hoc_cleaned.to_csv('./data_generated/edges_rest_recovery_post_hoc.csv', index=False, header=False)

# # save the missing edges
# np.save('./data_generated/missing_edges_movie.npy', missing_edges_movie)
# np.save('./data_generated/missing_edges_rest.npy', missing_edges_rest)
# np.save('./data_generated/missing_edges_effect_of_movie.npy', missing_edges_effect_of_movie)
# np.save('./data_generated/missing_edges_rest_last_60_TR.npy', missing_edges_rest_last_60_TR)
# np.save('./data_generated/missing_edges_double.npy', missing_edges_double)
# np.save('./data_generated/missing_edges_rest_post_hoc.npy', missing_edges_rest_post_hoc)

# reset the index of new_df_edges_movie as edges
new_df_edges_movie.index = edges
new_df_edges_rest.index = edges
new_df_edges_effect_of_movie.index = edges
new_df_edges_rest_last_60_TR.index = edges
new_df_edges_double_two_way.index = edges
new_df_edges_rest_post_hoc.index = edges_rest

# re-fit the new_df_edges_rest_post_hoc into the 35778 edges
new_df_edges_rest_post_hoc = new_df_edges_rest_post_hoc.reindex(edges)


new_df_edges_movie_list = new_df_edges_movie['u1'].tolist()
new_df_edges_rest_list = new_df_edges_rest['u1'].tolist()
new_df_edges_effect_of_movie_list = new_df_edges_effect_of_movie['u1'].tolist()
new_df_edges_rest_last_60_TR_list = new_df_edges_rest_last_60_TR['u1'].tolist()
new_df_edges_double_two_way_list = new_df_edges_double_two_way['u1'].tolist()
new_df_edges_rest_post_hoc_list = new_df_edges_rest_post_hoc['u1'].tolist()

nodes_edges_movie = [i for i, x in enumerate(new_df_edges_movie_list) if str(x) != 'nan']
nodes_edges_rest = [i for i, x in enumerate(new_df_edges_rest_list) if str(x) != 'nan']
nodes_edges_effect_of_movie = [i for i, x in enumerate(new_df_edges_effect_of_movie_list) if str(x) != 'nan']
nodes_edges_rest_last_60_TR = [i for i, x in enumerate(new_df_edges_rest_last_60_TR_list) if str(x) != 'nan']
nodes_edges_double_two_way = [i for i, x in enumerate(new_df_edges_double_two_way_list) if str(x) != 'nan']
nodes_edges_rest_post_hoc = [i for i, x in enumerate(new_df_edges_rest_post_hoc_list) if str(x) != 'nan']

new_df_edges_movie = new_df_edges_movie['u1']
new_df_edges_rest = new_df_edges_rest['u1']
new_df_edges_effect_of_movie = new_df_edges_effect_of_movie['u1']
new_df_edges_rest_last_60_TR = new_df_edges_rest_last_60_TR['u1']
new_df_edges_double_two_way = new_df_edges_double_two_way['u1']
new_df_edges_rest_post_hoc = new_df_edges_rest_post_hoc['u1']

new_df_edges_movie_clean = new_df_edges_movie.dropna(axis=0, how='any')
new_df_edges_rest_clean = new_df_edges_rest.dropna(axis=0, how='any')
new_df_edges_effect_of_movie_clean = new_df_edges_effect_of_movie.dropna(axis=0, how='any')
new_df_edges_rest_last_60_TR_clean = new_df_edges_rest_last_60_TR.dropna(axis=0, how='any')
new_df_edges_double_two_way_clean = new_df_edges_double_two_way.dropna(axis=0, how='any')
new_df_edges_rest_post_hoc_clean = new_df_edges_rest_post_hoc.dropna(axis=0, how='any')


def check_min_and_max(pd_series):
    min_value = pd_series.min()
    max_value = pd_series.max()
    print('min value is: ', min_value)
    print('max value is: ', max_value)


# check_min_and_max(new_df_edges_movie_clean)
# check_min_and_max(new_df_edges_rest_clean)
# check_min_and_max(new_df_edges_effect_of_movie_clean)
# check_min_and_max(new_df_edges_rest_last_60_TR_clean)
# check_min_and_max(new_df_edges_double_two_way_clean)
# check_min_and_max(new_df_edges_rest_post_hoc_clean)


def boxplot_the_mean(df_1, df_2, df_3, df_4, nodes_with_values):
    if df_3 is None and df_4 is None:
        df_1 = df_1.iloc[:, nodes_with_values].mean(axis=1)
        df_2 = df_2.iloc[:, nodes_with_values].mean(axis=1)
        plt.boxplot([df_1, df_2])
        plt.xticks([1, 2], ['Rest', 'Narrative Listening'])
        plt.ylabel('Mean of Functional Connectivity')
        plt.xlabel('Sedation Level')
        plt.show()

        print('mean of df_1 is: ', df_1.mean())
        print('mean of df_2 is: ', df_2.mean())

    else:
        df_1 = df_1.iloc[:, nodes_with_values].mean(axis=1)
        df_2 = df_2.iloc[:, nodes_with_values].mean(axis=1)
        df_3 = df_3.iloc[:, nodes_with_values].mean(axis=1)
        df_4 = df_4.iloc[:, nodes_with_values].mean(axis=1)

        plt.boxplot([df_1, df_2, df_3, df_4])
        plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
        plt.ylabel('Mean of Functional Connectivity')
        plt.xlabel('Sedation Level')
        plt.show()


# boxplot_the_mean(edges_movie_awake, edges_movie_mild, edges_movie_deep, edges_movie_recovery, nodes_edges_movie)
# boxplot_the_mean(edges_rest_awake, edges_rest_mild, edges_rest_deep, edges_rest_recovery, nodes_edges_rest)
# boxplot_the_mean(edges_rest_awake_last_60_TR, edges_movie_awake, None, None, nodes_edges_effect_of_movie)
# boxplot_the_mean(edges_rest_awake_last_60_TR, edges_movie_deep, None, None, nodes_edges_double_two_way)
# boxplot_the_mean(edges_rest_awake_post_hoc, edges_rest_mild_post_hoc, edges_rest_deep_post_hoc, edges_rest_recovery_post_hoc, nodes_edges_rest_post_hoc)


# visualize the significant edges
# load the shen 268 atlas
atlas = load_shen_268(1)
coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)


def matrix_generator(df_edges):
    # create a matrix of zeros
    matrix = np.zeros((268, 268))
    # fill the matrix with the edges
    matrix[np.triu_indices(268, 1)] = df_edges
    matrix += np.triu(matrix, 1).T
    # fill all the nan values with 0
    matrix[np.isnan(matrix)] = 0
    # flip the sign of the matrix
    matrix = -matrix
    return matrix


# # plot the matrix
# plt.imshow(matrix, cmap='RdBu_r', vmin=-0.03, vmax=0.03)
# plt.colorbar()
# plt.show()

# plot the matrix with the atlas
# plotting.plot_connectome(matrix_generator(new_df_edges_movie), coordinates, colorbar=True, node_size=0, edge_threshold=0.025)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_rest), coordinates, colorbar=True, node_size=0, edge_threshold=0.020)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_effect_of_movie), coordinates, colorbar=True, node_size=0, edge_threshold=0.020)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_rest_last_60_TR), coordinates, colorbar=True, node_size=0, edge_threshold=0.020)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_double_two_way), coordinates, colorbar=True, node_size=0, edge_threshold="99.9%")
# plt.show()

plotting.plot_connectome(matrix_generator(new_df_edges_rest_post_hoc), coordinates, colorbar=True, node_size=0, edge_threshold="99.9%")
plt.show()

a = matrix_generator(new_df_edges_rest_post_hoc)

