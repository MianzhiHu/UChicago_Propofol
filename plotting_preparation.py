import scipy.io as sio
import numpy as np
import pandas as pd
from partial_least_squares import df_movie_missing, df_03_missing, df_02_missing, df_01_missing, df_everything_missing, \
    df_03_30_missing, df_combined_missing
from scipy.stats import spearmanr

# read lv_vals file
lv_vals = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie_lv_vals.mat')
lv_vals_movie_03 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_03_lv_vals.mat')
lv_vals_movie_03_30 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_03_30_lv_vals.mat')
lv_vals_movie_02 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_02_lv_vals.mat')
lv_vals_movie_01 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_01_lv_vals.mat')
lv_vals_movie_01_2 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_01_2_lv_vals.mat')
lv_vals_movie_01_3 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_1_3_lv_vals.mat')
lv_vals_movie_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSeverything_lv_vals.mat')
lv_vals_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSlast_60_TR_lv_vals.mat')
lv_vals_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSeffect_of_movie_lv_vals.mat')
lv_vals_double_three_way = sio.loadmat('./data_generated/PLS_outputTaskPLSdouble_lv_vals.mat')
lv_vals_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLStwo-way double_lv_vals.mat')
lv_vals_double_merged = sio.loadmat('./data_generated/PLS_outputTaskPLSmerged_double_lv_vals.mat')
lv_vals_combined = sio.loadmat('./data_generated/PLS_outputTaskPLScombined_lv_vals.mat')

fc_lv_vals_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_movie_lv_vals.mat')
fc_lv_vals_rest = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_rest_lv_vals.mat')
fc_lv_vals_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_rest_last_60_TR_lv_vals.mat')
fc_lv_vals_movie_abs = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_movie_abs_lv_vals.mat')
fc_lv_vals_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_effect_of_movie_lv_vals.mat')
fc_lv_vals_double_three_way = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_three_way_lv_vals.mat')
fc_lv_vals_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_two_way_lv_vals.mat')
fc_lv_vals_double_merged = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_merged_lv_vals.mat')
fc_lv_vals_combined = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_combined_lv_vals.mat')
fc_lv_vals_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_everything_lv_vals.mat')
edges_lv_vals_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_movie_lv_vals.mat')
edges_lv_vals_rest = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest_lv_vals.mat')
edges_lv_vals_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_effect_of_movie_lv_vals.mat')
edges_lv_vals_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest_last_60_TR_lv_vals.mat')
edges_lv_vals_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_double_two_way_lv_vals.mat')
edges_lv_vals_rest_post_hoc = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest_post_hoc_lv_vals.mat')
edges_lv_vals_combined = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_combined_lv_vals.mat')
edges_lv_vals_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_everything_lv_vals.mat')

corr = spearmanr(fc_lv_vals_double_three_way['u1'][:, 0], fc_lv_vals_double_two_way['u1'][:, 0])
corr_1 = spearmanr(fc_lv_vals_double_two_way['u1'][:, 0], fc_lv_vals_double_merged['u1'][:, 0])
corr_2 = spearmanr(fc_lv_vals_double_two_way['u1'][:, 0], lv_vals_double_two_way['u1'][:, 0])

# read bootstrap ratio file
boot_ratio = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie.mat')
boot_ratio_movie_03 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_03.mat')
boot_ratio_movie_03_30 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_03_30.mat')
boot_ratio_movie_02 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_02.mat')
boot_ratio_movie_01 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_01.mat')
boot_ratio_movie_01_2 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_01_2.mat')
boot_ratio_movie_01_3 = sio.loadmat('./data_generated/PLS_outputTaskPLSmovie_1_3.mat')
boot_ratio_movie_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSeverything.mat')
boot_ratio_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSlast_60_TR.mat')
boot_ratio_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSeffect_of_movie.mat')
boot_ratio_double_three_way = sio.loadmat('./data_generated/PLS_outputTaskPLSdouble.mat')
boot_ratio_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLStwo-way double.mat')
boot_ratio_double_merged = sio.loadmat('./data_generated/PLS_outputTaskPLSmerged_double.mat')
boot_ratio_combined = sio.loadmat('./data_generated/PLS_outputTaskPLScombined.mat')

fc_boot_ratio_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_movie.mat')
fc_boot_ratio_rest = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_rest.mat')
fc_boot_ratio_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_rest_last_60_TR.mat')
fc_boot_ratio_movie_abs = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_movie_abs.mat')
fc_boot_ratio_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_effect_of_movie.mat')
fc_boot_ratio_double_three_way = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_three_way.mat')
fc_boot_ratio_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_two_way.mat')
fc_boot_ratio_double_merged = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_double_merged.mat')
fc_boot_ratio_combined = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_combined.mat')
fc_boot_ratio_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSfc_everything.mat')
edges_boot_ratio_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_movie.mat')
edges_boot_ratio_rest = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest.mat')
edges_boot_ratio_effect_of_movie = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_effect_of_movie.mat')
edges_boot_ratio_rest_last_60_TR = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest_last_60_TR.mat')
edges_boot_ratio_double_two_way = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_double_two_way.mat')
edges_boot_ratio_rest_post_hoc = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_rest_post_hoc.mat')
edges_boot_ratio_combined = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_combined.mat')
edges_boot_ratio_everything = sio.loadmat('./data_generated/PLS_outputTaskPLSedges_everything.mat')

# load the missing nodes
double_nan = np.load('./data_generated/double_missing.npy')
fc_movie_nan = np.load('./data_generated/fc_movie_nan.npy')
fc_rest_nan = np.load('./data_generated/fc_rest_nan.npy')
fc_rest_last_60_TR_nan = np.load('./data_generated/fc_rest_nan_last_60_TR.npy')
fc_effect_of_movie_nan = np.load('./data_generated/fc_effect_of_movie_nan.npy')
fc_double_nan = np.load('./data_generated/fc_double_nan.npy')
fc_combined_nan = np.load('./data_generated/fc_combined_nan.npy')
fc_all_nan = np.load('./data_generated/fc_all_nan.npy')
df_last_60_TR_missing = np.load('./data_generated/last_60_TR_rest_missing.npy')
edges_movie_nan = np.load('./data_generated/missing_edges_movie.npy')
edges_rest_nan = np.load('./data_generated/missing_edges_rest.npy')
effect_of_movie_nan = np.load('./data_generated/effect_of_movie_missing.npy')
edges_effect_of_movie_nan = np.load('./data_generated/missing_edges_effect_of_movie.npy')
edges_rest_last_60_TR_nan = np.load('./data_generated/missing_edges_rest_last_60_TR.npy')
edges_double_nan = np.load('./data_generated/missing_edges_double.npy')
edges_rest_post_hoc_nan = np.load('./data_generated/missing_edges_rest_post_hoc.npy')
edges_combined_nan = np.load('./data_generated/missing_edges_combined.npy')
edges_everything_nan = np.load('./data_generated/missing_edges_everything.npy')


# # find the union set between df_movie_missing and df_last_60_TR_missing
# effect_of_movie_missing = np.union1d(df_movie_missing, df_last_60_TR_missing)
#
# # save the union set
# np.save('./data_generated/effect_of_movie_missing.npy', effect_of_movie_missing)

def plot_preparation(lv_vals, boot_ratio, nodes_with_missing_values, keep=False):
    # get the data with only the first column
    u1 = lv_vals['u1'][:, 0]
    # print(u1)

    # get the data
    boot_ratio = boot_ratio['bsrs1']
    # print(boot_ratio)

    # combine the data with their respective columns
    data = np.column_stack((u1, boot_ratio))
    # print(data)

    # name the columns
    df = pd.DataFrame(data, columns=['u1', 'boot_ratio'])
    # print(df)

    if not keep:
        # keep only the rows with an absolute boot_ratio value greater than 3 and set the rest to NAN
        df.loc[abs(df['boot_ratio']) < 3, 'u1'] = np.nan
        # print(df)
    elif keep:
        # skip this step
        pass

    # Create a new dataframe with NaN values for all rows
    new_df = pd.DataFrame(data=np.nan, index=range(len(df) + len(nodes_with_missing_values)), columns=df.columns)

    # Use loc method to insert the deleted rows at their original position
    for i, row in enumerate(nodes_with_missing_values):
        new_df.loc[row] = np.nan

    # Update the values of the remaining rows in the new dataframe
    j = 0
    for i in range(len(new_df)):
        if i not in nodes_with_missing_values:
            new_df.iloc[i, :] = df.iloc[j, :]
            j += 1

    return new_df


new_df = plot_preparation(lv_vals, boot_ratio, df_movie_missing)
new_df_movie_03 = plot_preparation(lv_vals_movie_03, boot_ratio_movie_03, df_03_missing)
new_df_movie_03_30 = plot_preparation(lv_vals_movie_03_30, boot_ratio_movie_03_30, df_03_30_missing)
new_df_movie_02 = plot_preparation(lv_vals_movie_02, boot_ratio_movie_02, df_02_missing)
new_df_movie_01 = plot_preparation(lv_vals_movie_01, boot_ratio_movie_01, df_01_missing)
new_df_movie_01_3 = plot_preparation(lv_vals_movie_01_3, boot_ratio_movie_01_3, df_01_missing)
new_df_movie_everything = plot_preparation(lv_vals_movie_everything, boot_ratio_movie_everything, df_everything_missing)
new_df_rest_last_60_TR = plot_preparation(lv_vals_rest_last_60_TR, boot_ratio_rest_last_60_TR, df_last_60_TR_missing)
new_df_effect_of_movie = plot_preparation(lv_vals_effect_of_movie, boot_ratio_effect_of_movie, effect_of_movie_nan)
new_df_double_three_way = plot_preparation(lv_vals_double_three_way, boot_ratio_double_three_way, double_nan)
new_df_double_two_way = plot_preparation(lv_vals_double_two_way, boot_ratio_double_two_way, double_nan)
new_df_double_merged = plot_preparation(lv_vals_double_merged, boot_ratio_double_merged, double_nan)
new_df_combined = plot_preparation(lv_vals_combined, boot_ratio_combined, df_combined_missing)

new_df_fc_movie = plot_preparation(fc_lv_vals_movie, fc_boot_ratio_movie, fc_movie_nan)
new_df_fc_rest = plot_preparation(fc_lv_vals_rest, fc_boot_ratio_rest, fc_rest_nan)
new_df_fc_rest_last_60_TR = plot_preparation(fc_lv_vals_rest_last_60_TR, fc_boot_ratio_rest_last_60_TR,
                                             fc_rest_last_60_TR_nan)
new_df_fc_movie_abs = plot_preparation(fc_lv_vals_movie_abs, fc_boot_ratio_movie_abs, fc_movie_nan)
new_df_fc_effect_of_movie = plot_preparation(fc_lv_vals_effect_of_movie, fc_boot_ratio_effect_of_movie,
                                             fc_effect_of_movie_nan)
new_df_fc_double_three_way = plot_preparation(fc_lv_vals_double_three_way, fc_boot_ratio_double_three_way,
                                              fc_double_nan)
new_df_fc_double_two_way = plot_preparation(fc_lv_vals_double_two_way, fc_boot_ratio_double_two_way, fc_double_nan)
new_df_fc_double_merged = plot_preparation(fc_lv_vals_double_merged, fc_boot_ratio_double_merged, fc_double_nan)
new_df_fc_combined = plot_preparation(fc_lv_vals_combined, fc_boot_ratio_combined, fc_combined_nan)
new_df_fc_everything = plot_preparation(fc_lv_vals_everything, fc_boot_ratio_everything, fc_all_nan)
new_df_edges_movie = plot_preparation(edges_lv_vals_movie, edges_boot_ratio_movie, edges_movie_nan)
new_df_edges_rest = plot_preparation(edges_lv_vals_rest, edges_boot_ratio_rest, edges_rest_nan)
new_df_edges_effect_of_movie = plot_preparation(edges_lv_vals_effect_of_movie, edges_boot_ratio_effect_of_movie,
                                                edges_effect_of_movie_nan)
new_df_edges_rest_last_60_TR = plot_preparation(edges_lv_vals_rest_last_60_TR, edges_boot_ratio_rest_last_60_TR,
                                                edges_rest_last_60_TR_nan)
new_df_edges_double_two_way = plot_preparation(edges_lv_vals_double_two_way, edges_boot_ratio_double_two_way,
                                               edges_double_nan)
new_df_edges_rest_post_hoc = plot_preparation(edges_lv_vals_rest_post_hoc, edges_boot_ratio_rest_post_hoc,
                                              edges_rest_post_hoc_nan)
new_df_edges_combined = plot_preparation(edges_lv_vals_combined, edges_boot_ratio_combined, edges_combined_nan)
new_df_edges_everything = plot_preparation(edges_lv_vals_everything, edges_boot_ratio_everything, edges_everything_nan)

hurst_effect_of_movie_full = plot_preparation(lv_vals_effect_of_movie, boot_ratio_effect_of_movie, effect_of_movie_nan,
                                              keep=True)

def plot_preparation_for_2ndLV(lv_vals, boot_ratio, nodes_with_missing_values):
    # get the data with only the first column
    u1 = lv_vals['u1'][:, 1]
    # print(u1)

    # get the data
    boot_ratio = boot_ratio['bsrs1']
    # print(boot_ratio)

    # combine the data with their respective columns
    data = np.column_stack((u1, boot_ratio))
    # print(data)

    # name the columns
    df = pd.DataFrame(data, columns=['u1', 'boot_ratio'])
    # print(df)

    # keep only the rows with an absolute boot_ratio value greater than 3 and set the rest to NAN
    df.loc[abs(df['boot_ratio']) < 3, 'u1'] = np.nan
    # print(df)

    # Create a new dataframe with NaN values for all rows
    new_df = pd.DataFrame(data=np.nan, index=range(len(df) + len(nodes_with_missing_values)), columns=df.columns)

    # Use loc method to insert the deleted rows at their original position
    for i, row in enumerate(nodes_with_missing_values):
        new_df.loc[row] = np.nan

    # Update the values of the remaining rows in the new dataframe
    j = 0
    for i in range(len(new_df)):
        if i not in nodes_with_missing_values:
            new_df.iloc[i, :] = df.iloc[j, :]
            j += 1

    print(new_df)
    return new_df


# new_df_movie_01_2 = plot_preparation_for_2ndLV(lv_vals_movie_01_2, boot_ratio_movie_01_2, df_01_missing)


print('plotting_preparation has been read')
