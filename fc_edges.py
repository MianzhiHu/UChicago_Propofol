import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_preparation import new_df_edges_movie, new_df_edges_rest, new_df_edges_effect_of_movie, \
    new_df_edges_rest_last_60_TR, new_df_edges_double_two_way, new_df_edges_rest_post_hoc, new_df_edges_combined, \
    new_df_edges_everything
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

with open('./pickles/fc_dict.pickle', 'rb') as f:
    general_fc = pickle.load(f)
    for key, value in general_fc.items():
        # exclude the diagonal
        df = np.where(value == 2.0, np.nan, value)
        # calculate the average FC
        average = np.nanmean(df, axis=0)
        general_fc[key] = average

    all_general_fc = pd.DataFrame(general_fc).transpose()

with open('./pickles/fc_dict_last_60_TR.pickle', 'rb') as f:
    general_rest_fc = pickle.load(f)
    for key, value in general_rest_fc .items():
        # exclude the diagonal
        df = np.where(value == 2.0, np.nan, value)
        # calculate the average FC
        average = np.nanmean(df, axis=0)
        general_rest_fc [key] = average

    all_general_rest_fc = pd.DataFrame(general_rest_fc).transpose()

# remove columns that contain NaN and record the missing nodes
missing_nodes = all_general_fc.columns[all_general_fc.isnull().any()].tolist()
missing_nodes_last_60 = all_general_fc.columns[all_general_fc.isnull().any()].tolist()
missing_nodes_total = np.union1d(missing_nodes, missing_nodes_last_60)

# drop the missing nodes by column index
all_general_fc = all_general_fc.drop(columns=missing_nodes)
all_general_rest_fc = all_general_rest_fc.drop(columns=missing_nodes_last_60)

# take absolute values
all_general_fc = np.abs(all_general_fc)
all_general_rest_fc = np.abs(all_general_rest_fc)

general_fc_movie_awake = all_general_fc[all_general_fc.index.str.contains('movie_01_LPI')]
general_fc_movie_mild = all_general_fc[all_general_fc.index.str.contains('movie_02_LPI')]
general_fc_movie_deep = all_general_fc[all_general_fc.index.str.contains('movie_03_LPI')]
general_fc_movie_recovery = all_general_fc[all_general_fc.index.str.contains('movie_04_LPI')]
general_fc_rest_awake = all_general_rest_fc[all_general_rest_fc.index.str.contains('rest_01_LPI')]
general_fc_rest_mild = all_general_rest_fc[all_general_rest_fc.index.str.contains('rest_02_LPI')]
general_fc_rest_deep = all_general_rest_fc[all_general_rest_fc.index.str.contains('rest_03_LPI')]
general_fc_rest_recovery = all_general_rest_fc[all_general_rest_fc.index.str.contains('rest_04_LPI')]

# save the data
general_fc_movie_awake.to_csv('./data_generated/general_fc_movie_awake.csv', index=False, header=False)
general_fc_movie_mild.to_csv('./data_generated/general_fc_movie_mild.csv', index=False, header=False)
general_fc_movie_deep.to_csv('./data_generated/general_fc_movie_deep.csv', index=False, header=False)
general_fc_movie_recovery.to_csv('./data_generated/general_fc_movie_recovery.csv', index=False, header=False)
general_fc_rest_awake.to_csv('./data_generated/general_fc_rest_awake.csv', index=False, header=False)
general_fc_rest_mild.to_csv('./data_generated/general_fc_rest_mild.csv', index=False, header=False)
general_fc_rest_deep.to_csv('./data_generated/general_fc_rest_deep.csv', index=False, header=False)
general_fc_rest_recovery.to_csv('./data_generated/general_fc_rest_recovery.csv', index=False, header=False)
general_fc_missing_nodes = np.save('./data_generated/general_fc_missing_nodes.npy', missing_nodes_total)




with open('./pickles/fc_dict.pickle', 'rb') as f:
    fc_dict = pickle.load(f)

    # for all the files in the dictionary, only keep the upper triangle
    for key, value in fc_dict.items():
        flattened_matrix = upper_triangle_flattened(value)
        df = pd.DataFrame(({'edges': edges, 'values': flattened_matrix}))
        fc_dict[key] = df

    all_data = pd.concat(fc_dict, names=['key'])
    all_data.reset_index(inplace=True)
    pivoted_data = all_data.pivot(index='key', columns='edges', values='values')

    edges_movie_awake = pivoted_data[pivoted_data.index.str.contains('movie_01_LPI')]
    edges_movie_mild = pivoted_data[pivoted_data.index.str.contains('movie_02_LPI')]
    edges_movie_deep = pivoted_data[pivoted_data.index.str.contains('movie_03_LPI')]
    edges_movie_recovery = pivoted_data[pivoted_data.index.str.contains('movie_04_LPI')]
    edges_rest_awake = pivoted_data[pivoted_data.index.str.contains('rest_01_LPI')]
    edges_rest_mild = pivoted_data[pivoted_data.index.str.contains('rest_02_LPI')]
    edges_rest_deep = pivoted_data[pivoted_data.index.str.contains('rest_03_LPI')]
    edges_rest_recovery = pivoted_data[pivoted_data.index.str.contains('rest_04_LPI')]

with open('./pickles/fc_dict_last_60_TR.pickle', 'rb') as f:
    fc_dict = pickle.load(f)

    # for all the files in the dictionary, only keep the upper triangle
    for key, value in fc_dict.items():
        flattened_matrix = upper_triangle_flattened(value)
        df = pd.DataFrame(({'edges': edges, 'values': flattened_matrix}))
        fc_dict[key] = df

    all_data = pd.concat(fc_dict, names=['key'])
    all_data.reset_index(inplace=True)
    pivoted_data = all_data.pivot(index='key', columns='edges', values='values')

    edges_rest_awake_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_01_LPI')]
    edges_rest_mild_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_02_LPI')]
    edges_rest_deep_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_03_LPI')]
    edges_rest_recovery_last_60_TR = pivoted_data[pivoted_data.index.str.contains('rest_04_LPI')]

# # only keep the edges if they are in the edges rest
# edges_rest_awake_post_hoc = edges_rest_awake_last_60_TR[[col for col in edges_rest_awake_last_60_TR.columns if col in edges_rest]]
# # print(edges_rest_awake_post_hoc.shape)
# edges_rest_mild_post_hoc = edges_rest_mild_last_60_TR[[col for col in edges_rest_mild_last_60_TR.columns if col in edges_rest]]
# # print(edges_rest_mild_post_hoc.shape)
# edges_rest_deep_post_hoc = edges_rest_deep_last_60_TR[[col for col in edges_rest_deep_last_60_TR.columns if col in edges_rest]]
# # print(edges_rest_deep_post_hoc.shape)
# edges_rest_recovery_post_hoc = edges_rest_recovery_last_60_TR[[col for col in edges_rest_recovery_last_60_TR.columns if col in edges_rest]]
# # print(edges_rest_recovery_post_hoc.shape)


def record_missing_edges(*dfs):
    # Initialize a list to track missing edges
    missing_edges = []

    # Concatenate all dataframes passed to the function
    df_all = pd.concat(dfs)

    # Identify columns with missing values
    for i, col in enumerate(df_all.columns):
        if df_all[col].isnull().values.any():
            missing_edges.append(i)

    # Clean the data: drop columns with any missing values and convert to absolute values
    df_all = df_all.dropna(axis=1, how='any')
    df_all = np.abs(df_all)

    # Split df_all back into individual dataframes based on original lengths
    result_dfs = []
    start_idx = 0
    for df in dfs:
        end_idx = start_idx + len(df)
        # Slice df_all for each original dataframe's length
        result_dfs.append(df_all.iloc[start_idx:end_idx, :])
        start_idx = end_idx

    # Return the cleaned dataframes and the list of columns with missing values before cleaning
    return (*result_dfs, missing_edges)

# edges_movie_awake_cleaned, edges_movie_mild_cleaned, edges_movie_deep_cleaned, edges_movie_recovery_cleaned, missing_edges_movie = record_missing_edges(
#     edges_movie_awake, edges_movie_mild, edges_movie_deep, edges_movie_recovery)
# edges_rest_awake_cleaned, edges_rest_mild_cleaned, edges_rest_deep_cleaned, edges_rest_recovery_cleaned, missing_edges_rest = record_missing_edges(
#     edges_rest_awake, edges_rest_mild, edges_rest_deep, edges_rest_recovery)
# edges_movie_awake_cleaned_1, edges_rest_awake_cleaned_1, missing_edges_effect_of_movie = record_missing_edges(
#     edges_movie_awake, edges_rest_awake_last_60_TR)
# edges_rest_awake_last_60_TR_cleaned, edges_rest_mild_last_60_TR_cleaned, edges_rest_deep_last_60_TR_cleaned, edges_rest_recovery_last_60_TR_cleaned, missing_edges_rest_last_60_TR = record_missing_edges(
#     edges_rest_awake_last_60_TR, edges_rest_mild_last_60_TR, edges_rest_deep_last_60_TR, edges_rest_recovery_last_60_TR)
# edges_double_rest_awake_cleaned, edges_double_movie_mild_cleaned, edges_double_movie_deep_cleaned, missing_edges_double = record_missing_edges(
#     edges_rest_awake_last_60_TR, edges_movie_mild, edges_movie_deep)
# edges_rest_awake_post_hoc_cleaned, edges_rest_mild_post_hoc_cleaned, edges_rest_deep_post_hoc_cleaned, edges_rest_recovery_post_hoc_cleaned, missing_edges_rest_post_hoc = record_missing_edges(
#     edges_rest_awake_post_hoc, edges_rest_mild_post_hoc, edges_rest_deep_post_hoc, edges_rest_recovery_post_hoc)
# edges_rest_awake_combined_cleaned, edges_movie_awake_combined_cleaned, edges_movie_mild_combined_cleaned, edges_movie_deep_combined_cleaned, edges_movie_recovery_combined_cleaned, missing_edges_combined = record_missing_edges(
#     edges_rest_awake_last_60_TR, edges_movie_awake, edges_movie_mild, edges_movie_deep, edges_movie_recovery)
edges_movie_awake_everything, edges_movie_mild_everything, edges_movie_deep_everything, edges_movie_recovery_everything, edges_rest_awake_everything, edges_rest_mild_everything, edges_rest_deep_everything, edges_rest_recovery_everything, missing_edges_everything = record_missing_edges(
    edges_movie_awake, edges_movie_mild, edges_movie_deep, edges_movie_recovery, edges_rest_awake_last_60_TR, edges_rest_mild_last_60_TR, edges_rest_deep_last_60_TR, edges_rest_recovery_last_60_TR)

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

# edges_rest_awake_combined_cleaned.to_csv('./data_generated/edges_rest_awake_combined.csv', index=False, header=False)
# edges_movie_awake_combined_cleaned.to_csv('./data_generated/edges_movie_awake_combined.csv', index=False, header=False)
# edges_movie_mild_combined_cleaned.to_csv('./data_generated/edges_movie_mild_combined.csv', index=False, header=False)
# edges_movie_deep_combined_cleaned.to_csv('./data_generated/edges_movie_deep_combined.csv', index=False, header=False)
# edges_movie_recovery_combined_cleaned.to_csv('./data_generated/edges_movie_recovery_combined.csv', index=False, header=False)

# edges_movie_awake_everything.to_csv('./data_generated/edges_movie_awake_everything.csv', index=False, header=False)
# edges_movie_mild_everything.to_csv('./data_generated/edges_movie_mild_everything.csv', index=False, header=False)
# edges_movie_deep_everything.to_csv('./data_generated/edges_movie_deep_everything.csv', index=False, header=False)
# edges_movie_recovery_everything.to_csv('./data_generated/edges_movie_recovery_everything.csv', index=False, header=False)
# edges_rest_awake_everything.to_csv('./data_generated/edges_rest_awake_everything.csv', index=False, header=False)
# edges_rest_mild_everything.to_csv('./data_generated/edges_rest_mild_everything.csv', index=False, header=False)
# edges_rest_deep_everything.to_csv('./data_generated/edges_rest_deep_everything.csv', index=False, header=False)
# edges_rest_recovery_everything.to_csv('./data_generated/edges_rest_recovery_everything.csv', index=False, header=False)

# # save the missing edges
# np.save('./data_generated/missing_edges_movie.npy', missing_edges_movie)
# np.save('./data_generated/missing_edges_rest.npy', missing_edges_rest)
# np.save('./data_generated/missing_edges_effect_of_movie.npy', missing_edges_effect_of_movie)
# np.save('./data_generated/missing_edges_rest_last_60_TR.npy', missing_edges_rest_last_60_TR)
# np.save('./data_generated/missing_edges_double.npy', missing_edges_double)
# np.save('./data_generated/missing_edges_rest_post_hoc.npy', missing_edges_rest_post_hoc)
# np.save('./data_generated/missing_edges_combined.npy', missing_edges_combined)
# np.save('./data_generated/missing_edges_everything.npy', missing_edges_everything)

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
# check_min_and_max(new_df_edges_combined_clean)


def boxplot_the_mean(*dfs, nodes_with_values):
    for df in dfs:
        df = df[nodes_with_values]
        df = df.mean(axis=1)
        plt.boxplot(df)
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


def matrix_generator(new_df_edges):
    # reset the index of new_df_edges as edges
    new_df_edges.index = edges
    # # in case I need to identify the nodes with values, return this list
    # new_df_edges_list = new_df_edges['u1'].tolist()
    # nodes_edges = [i for i, x in enumerate(new_df_edges_list) if str(x) != 'nan']
    new_df_edges = new_df_edges['u1']
    # create a matrix of zeros
    matrix = np.zeros((268, 268))
    # fill the matrix with the edges
    matrix[np.triu_indices(268, 1)] = new_df_edges
    matrix += np.triu(matrix, 1).T
    # fill all the nan values with 0
    matrix[np.isnan(matrix)] = 0
    # flip the sign of the matrix
    # matrix = -matrix
    return matrix


# # plot the matrix
# plt.imshow(matrix, cmap='RdBu_r', vmin=-0.03, vmax=0.03)
# plt.colorbar()
# plt.show()

# plot the matrix with the atlas
# plotting.plot_connectome(matrix_generator(new_df_edges_movie), coordinates, colorbar=True, node_size=0, edge_threshold="99.7%")
# plt.savefig('./graphs/edges_movie_plot.png', dpi=650)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_rest), coordinates, colorbar=True, node_size=0, edge_threshold="99.8%")
# plt.savefig('./graphs/edges_effect_of_propofol_plot.png', dpi=800)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_effect_of_movie), coordinates, colorbar=True, node_size=0, edge_threshold="99.8%")
# plt.savefig('./graphs/edges_effect_of_movie_plot.png', dpi=650)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_rest_last_60_TR), coordinates, colorbar=True, node_size=0, edge_threshold="99.8%")
# plt.savefig('./graphs/edges_effect_of_propofol_plot.png', dpi=650)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_double_two_way), coordinates, colorbar=True, node_size=0, edge_threshold="99.9%")
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_rest_post_hoc), coordinates, colorbar=True, node_size=0, edge_threshold="99.9%")
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_combined), coordinates, colorbar=True, node_size=0, edge_threshold="99.8%")
# plt.savefig('./graphs/edges_combined_effects_plot.png', dpi=650)
# plt.show()

# plotting.plot_connectome(matrix_generator(new_df_edges_everything), coordinates, colorbar=True, node_size=0, edge_threshold="99.7%")
# plt.savefig('./graphs/edges_everything_plot.png', dpi=650)
# plt.show()


