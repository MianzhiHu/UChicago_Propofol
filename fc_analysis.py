from load_ts import read_files_268
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from plotting_preparation import fc_lv_vals_movie, fc_lv_vals_rest, \
    fc_movie_nan, fc_rest_nan, lv_vals, lv_vals_rest_last_60_TR, lv_vals_effect_of_movie, \
    df_movie_missing, df_last_60_TR_missing, effect_of_movie_nan, fc_lv_vals_effect_of_movie, \
    fc_effect_of_movie_nan, fc_lv_vals_rest_last_60_TR
from brainsmash.mapgen.stats import spearmanr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# fc_dict = {}
#
# for array_2d, file in read_files_268(directory='data_clean'):
#     corr = np.corrcoef(array_2d)
#     corr_z = np.arctanh(corr)
#     np.fill_diagonal(corr_z, 2.0000)
#     fc_dict[file] = corr_z
#
#     print(f'file: {file}')
#
# with open('fc_dict.pickle', 'wb') as outfile:
#     pickle.dump(fc_dict, outfile)
#
# print('done')

# # create a new version of the dictionary for the last 60 TRs of rest
# fc_dict_last_60_TR = {}
#
# for array_2d, file in read_files_268(directory='data_clean'):
#     if "rest" in file:
#         corr = np.corrcoef(array_2d[:, 90:150])
#         corr_z = np.arctanh(corr)
#         np.fill_diagonal(corr_z, 2.0000)
#         fc_dict_last_60_TR[file] = corr_z
#
#         print(f'file: {file}')
#
# with open('fc_dict_last_60_TR.pickle', 'wb') as outfile:
#     pickle.dump(fc_dict_last_60_TR, outfile)
#
# print('done')

def scatter_plot():
    with open('fc_dict.pickle', 'rb') as f:
        fc_dict = pickle.load(f)
        # remove the rest condition
        fc_dict = {key: value for key, value in fc_dict.items() if 'rest' not in key}
        for key, value in fc_dict.items():
            # replace the value dataframe with a list of means
            value_mean = np.abs(np.nanmean(value, axis=1))
            fc_dict[key] = value_mean
        df = pd.DataFrame.from_dict(fc_dict, orient='index')

    with open('fc_dict_last_60_TR.pickle', 'rb') as g:
        fc_dict_last_60_TR = pickle.load(g)
        # keep the rest condition
        fc_dict_last_60_TR = {key: value for key, value in fc_dict_last_60_TR.items() if 'rest' in key}
        for key, value in fc_dict_last_60_TR.items():
            # replace the value dataframe with a list of means
            value_mean_rest = np.abs(np.nanmean(value, axis=1))
            fc_dict_last_60_TR[key] = value_mean_rest
        df_rest = pd.DataFrame.from_dict(fc_dict_last_60_TR, orient='index')

    # combine the two dataframes
    df_fc = pd.concat([df, df_rest], axis=0)

    # take the row means and remove all the columns
    df_fc = df_fc.mean(axis=1)



    with open('outcome_268.pickle', 'rb') as h:
        hurst_dict = pickle.load(h)
        hurst_dict = {key: value for key, value in hurst_dict.items() if 'rest' not in key}
        for key, value in hurst_dict.items():
            # replace the value dataframe with a list of means
            hurst_dict[key] = value['hurst']

        df_hurst = pd.DataFrame.from_dict(hurst_dict, orient='index')

    df_hurst_rest = pd.read_csv('./data_generated/last_60_TR_all.csv', index_col=0)
    df_hurst_rest.columns = pd.to_numeric(df_hurst_rest.columns)

    df_hurst = pd.concat([df_hurst, df_hurst_rest], axis=0)
    df_hurst = df_hurst.mean(axis=1)

    # only keep the subjects that have both fc and hurst
    df_fc = df_fc[df_fc.index.isin(df_hurst.index)]

    # combine the two dataframes
    df_fc_reshaped = pd.concat([df_fc, df_hurst], axis=1)

    # set the column names
    df_fc_reshaped.columns = ['FC', 'Hurst']

    # # sort the dataframes by index
    # df_fc = df_fc.sort_index()
    # df_hurst = df_hurst.sort_index()
    #
    # # reshape the dataframes
    # df_fc_reshaped = pd.DataFrame(np.ravel(df_fc), columns=['FC'])
    # df_hurst_reshaped = pd.DataFrame(np.ravel(df_hurst), columns=['Hurst'])

    # # combine the two dataframes
    # df_fc_reshaped['Hurst'] = df_hurst_reshaped['Hurst']
    #
    # # remove the nan values
    # df_fc_reshaped = df_fc_reshaped.dropna()
    #
    # # delete the row if Hurst is above 1
    # df_fc_reshaped = df_fc_reshaped[df_fc_reshaped['Hurst'] < 1]

    return df_fc_reshaped


# df = scatter_plot()
#
# # check the maximum hurst
# print(df['FC'].max())
#
# # plot the scatter plot
# plt.scatter(df['FC'], df['Hurst'])
# plt.xlabel('FC')
# plt.ylabel('Hurst')
# plt.show()
#
# # test linear regression between the two hurst values
# lm = LinearRegression()
# lm.fit(df[['Hurst']], df[['FC']])
#
# # print the coefficients
# print('Intercept: %.3f' % lm.intercept_)
# print('Coefficient: %.3f' % lm.coef_)
# print('R-Squared: %.3f' % lm.score(df[['Hurst']], df[['FC']]))
#
# # print the linear regression equation
# print('y = %.3f + %.3f * x' % (lm.intercept_, lm.coef_))
#
# # calculate the p value of the linear regression
# X = df[['Hurst']]
# y = df[['FC']]
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print(model.summary())


def fc_analysis(subset_keys=None, pickle_file='fc_dict.pickle'):
    with open(pickle_file, 'rb') as f:
        fc_dict = pickle.load(f)

    # Select a subset of correlation matrices based on their file name
    subset_keys = [key for key in fc_dict.keys() if subset_keys in key]
    subset_matrices = [fc_dict[key] for key in subset_keys]

    # Calculate the mean correlation matrix across the subset
    n_matrices = len(subset_matrices)
    sum_matrix = np.sum(subset_matrices, axis=0)
    mean_matrix = sum_matrix / n_matrices

    # Convert the mean correlation matrix to a Pandas DataFrame for easy visualization
    mean_df = pd.DataFrame(mean_matrix)

    # find the nan columns
    # if a column only has one value, it will be nan
    nan_cols = []
    for col in mean_df.columns:
        if len(mean_df[col].unique()) == 2:
            nan_cols.append(col)

    # drop the nan columns
    mean_df_cleaned = mean_df.drop(nan_cols, axis=1).drop(nan_cols, axis=0)

    # take the absolute value of the correlation matrix
    mean_df = mean_df.abs()
    mean_df_cleaned = mean_df_cleaned.abs()

    # # Plot the mean correlation matrix
    # plt.imshow(mean_df_cleaned)
    # plt.colorbar()
    # plt.show()

    return nan_cols, mean_df


fc_movie_nan_1, fc_movie_awake = fc_analysis(subset_keys='movie_01_LPI')
fc_movie_nan_2, fc_movie_mild = fc_analysis(subset_keys='movie_02_LPI')
fc_movie_nan_3, fc_movie_deep = fc_analysis(subset_keys='movie_03_LPI')
fc_movie_nan_4, fc_movie_recovery = fc_analysis(subset_keys='movie_04_LPI')

fc_rest_nan_1, fc_rest_awake = fc_analysis(subset_keys='rest_01_LPI')
fc_rest_nan_2, fc_rest_mild = fc_analysis(subset_keys='rest_02_LPI')
fc_rest_nan_3, fc_rest_deep = fc_analysis(subset_keys='rest_03_LPI')
fc_rest_nan_4, fc_rest_recovery = fc_analysis(subset_keys='rest_04_LPI')

fc_all_nan_1, fc_movie = fc_analysis(subset_keys='movie')
fc_all_nan_2, fc_rest = fc_analysis(subset_keys='rest')

fc_rest_nan_last_60_TR_1, fc_rest_awake_last_60_TR = fc_analysis(subset_keys='rest_01_LPI',
                                                                 pickle_file='fc_dict_last_60_TR.pickle')
fc_rest_nan_last_60_TR_2, fc_rest_mild_last_60_TR = fc_analysis(subset_keys='rest_02_LPI',
                                                                pickle_file='fc_dict_last_60_TR.pickle')
fc_rest_nan_last_60_TR_3, fc_rest_deep_last_60_TR = fc_analysis(subset_keys='rest_03_LPI',
                                                                pickle_file='fc_dict_last_60_TR.pickle')
fc_rest_nan_last_60_TR_4, fc_rest_recovery_last_60_TR = fc_analysis(subset_keys='rest_04_LPI',
                                                                    pickle_file='fc_dict_last_60_TR.pickle')

# # combine the nan columns
# fc_movie_nan = list(set(fc_movie_nan_1 + fc_movie_nan_2 + fc_movie_nan_3 + fc_movie_nan_4))
# fc_combined_nan = list(set(fc_movie_nan_1 + fc_movie_nan_2 + fc_movie_nan_3 + fc_movie_nan_4 + fc_rest_nan_last_60_TR_1))
# fc_rest_nan = list(set(fc_rest_nan_1 + fc_rest_nan_2 + fc_rest_nan_3 + fc_rest_nan_4))
# fc_all_nan = list(set(fc_all_nan_1 + fc_all_nan_2))
# fc_effect_of_movie = list(set(fc_movie_nan_1 + fc_rest_nan_last_60_TR_1))
# fc_rest_nan_last_60_TR = list(set(fc_rest_nan_last_60_TR_1 + fc_rest_nan_last_60_TR_2 + fc_rest_nan_last_60_TR_3 + fc_rest_nan_last_60_TR_4))
# fc_double_nan = list(set(fc_rest_nan_last_60_TR_1 + fc_movie_nan_2 + fc_movie_nan_3))

# # save the nan columns as npy file
# np.save('./data_generated/fc_movie_nan', fc_movie_nan)
# np.save('./data_generated/fc_rest_nan', fc_rest_nan)
# np.save('./data_generated/fc_all_nan', fc_all_nan)
# np.save('./data_generated/fc_effect_of_movie_nan', fc_effect_of_movie)
# np.save('./data_generated/fc_rest_nan_last_60_TR', fc_rest_nan_last_60_TR)
# np.save('./data_generated/fc_double_nan', fc_double_nan)
# np.save('./data_generated/fc_combined_nan', fc_combined_nan)

# load the nan columns
fc_movie_nan = np.load('./data_generated/fc_movie_nan.npy')
fc_rest_nan = np.load('./data_generated/fc_rest_nan.npy')
fc_combined_nan = np.load('./data_generated/fc_combined_nan.npy')
fc_all_nan = np.load('./data_generated/fc_all_nan.npy')
fc_effect_of_movie_nan = np.load('./data_generated/fc_effect_of_movie_nan.npy')
fc_rest_nan_last_60_TR = np.load('./data_generated/fc_rest_nan_last_60_TR.npy')
fc_double_nan = np.load('./data_generated/fc_double_nan.npy')
nodes_with_fc_values = np.load('./data_generated/nodes_with_fc_values.npy', allow_pickle=True).tolist()
nodes_with_fc_values_rest = np.load('./data_generated/nodes_with_fc_values_rest.npy', allow_pickle=True).tolist()
nodes_with_fc_values_effect_of_movie = np.load('./data_generated/nodes_with_fc_values_effect_of_movie.npy',
                                               allow_pickle=True).tolist()
nodes_with_fc_values_rest_last_60_TR = np.load('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy',
                                               allow_pickle=True).tolist()
nodes_with_fc_values_double_three_way = np.load('./data_generated/nodes_with_fc_values_double_three_way.npy',
                                                allow_pickle=True).tolist()

# drop the nan columns
fc_movie_awake_clean = fc_movie_awake.drop(fc_movie_nan, axis=1).drop(fc_movie_nan, axis=0)
fc_movie_mild_clean = fc_movie_mild.drop(fc_movie_nan, axis=1).drop(fc_movie_nan, axis=0)
fc_movie_deep_clean = fc_movie_deep.drop(fc_movie_nan, axis=1).drop(fc_movie_nan, axis=0)
fc_movie_recovery_clean = fc_movie_recovery.drop(fc_movie_nan, axis=1).drop(fc_movie_nan, axis=0)

fc_combined_rest_awake_clean = fc_rest_awake_last_60_TR.drop(fc_combined_nan, axis=1).drop(fc_combined_nan, axis=0)
fc_combined_movie_awake_clean = fc_movie_awake.drop(fc_combined_nan, axis=1).drop(fc_combined_nan, axis=0)
fc_combined_movie_mild_clean = fc_movie_mild.drop(fc_combined_nan, axis=1).drop(fc_combined_nan, axis=0)
fc_combined_movie_deep_clean = fc_movie_deep.drop(fc_combined_nan, axis=1).drop(fc_combined_nan, axis=0)
fc_combined_movie_recovery_clean = fc_movie_recovery.drop(fc_combined_nan, axis=1).drop(fc_combined_nan, axis=0)

# check if there is any nan in the data
fc_combined_rest_awake_clean.isnull().values.any()
fc_combined_movie_awake_clean.isnull().values.any()
fc_combined_movie_mild_clean.isnull().values.any()
fc_combined_movie_deep_clean.isnull().values.any()
fc_combined_movie_recovery_clean.isnull().values.any()

fc_rest_awake_clean = fc_rest_awake.drop(fc_rest_nan, axis=1).drop(fc_rest_nan, axis=0)
fc_rest_mild_clean = fc_rest_mild.drop(fc_rest_nan, axis=1).drop(fc_rest_nan, axis=0)
fc_rest_deep_clean = fc_rest_deep.drop(fc_rest_nan, axis=1).drop(fc_rest_nan, axis=0)
fc_rest_recovery_clean = fc_rest_recovery.drop(fc_rest_nan, axis=1).drop(fc_rest_nan, axis=0)

fc_movie_clean = fc_movie.drop(fc_all_nan, axis=1).drop(fc_all_nan, axis=0)
fc_rest_clean = fc_rest.drop(fc_all_nan, axis=1).drop(fc_all_nan, axis=0)

fc_movie_effect_of_movie = fc_movie_awake.drop(fc_effect_of_movie_nan, axis=1).drop(fc_effect_of_movie_nan, axis=0)
fc_rest_effect_of_movie = fc_rest_awake_last_60_TR.drop(fc_effect_of_movie_nan, axis=1).drop(fc_effect_of_movie_nan, axis=0)


fc_rest_awake_last_60_TR_clean = fc_rest_awake_last_60_TR.drop(fc_rest_nan_last_60_TR, axis=1).drop(
    fc_rest_nan_last_60_TR, axis=0)
fc_rest_mild_last_60_TR_clean = fc_rest_mild_last_60_TR.drop(fc_rest_nan_last_60_TR, axis=1).drop(
    fc_rest_nan_last_60_TR, axis=0)
fc_rest_deep_last_60_TR_clean = fc_rest_deep_last_60_TR.drop(fc_rest_nan_last_60_TR, axis=1).drop(
    fc_rest_nan_last_60_TR, axis=0)
fc_rest_recovery_last_60_TR_clean = fc_rest_recovery_last_60_TR.drop(fc_rest_nan_last_60_TR, axis=1).drop(
    fc_rest_nan_last_60_TR, axis=0)

fc_double_rest_awake = fc_rest_awake_last_60_TR.drop(fc_double_nan, axis=1).drop(
    fc_double_nan, axis=0)
fc_double_movie_mild = fc_movie_mild.drop(fc_double_nan, axis=1).drop(
    fc_double_nan, axis=0)
fc_double_movie_deep = fc_movie_deep.drop(fc_double_nan, axis=1).drop(
    fc_double_nan, axis=0)
# take the average of the two movie fc matrices
fc_double_movie = (fc_double_movie_mild + fc_double_movie_deep) / 2




# # save the cleaned fc matrices as csv files
# fc_movie_awake_clean.to_csv('./data_generated/fc_movie_awake_abs.csv', index=False, header=False)
# fc_movie_mild_clean.to_csv('./data_generated/fc_movie_mild_abs.csv', index=False, header=False)
# fc_movie_deep_clean.to_csv('./data_generated/fc_movie_deep_abs.csv', index=False, header=False)
# fc_movie_recovery_clean.to_csv('./data_generated/fc_movie_recovery_abs.csv', index=False, header=False)

# fc_combined_rest_awake_clean.to_csv('./data_generated/fc_combined_rest_awake.csv', index=False, header=False)
# fc_combined_movie_awake_clean.to_csv('./data_generated/fc_combined_movie_awake.csv', index=False, header=False)
# fc_combined_movie_mild_clean.to_csv('./data_generated/fc_combined_movie_mild.csv', index=False, header=False)
# fc_combined_movie_deep_clean.to_csv('./data_generated/fc_combined_movie_deep.csv', index=False, header=False)
# fc_combined_movie_recovery_clean.to_csv('./data_generated/fc_combined_movie_recovery.csv', index=False, header=False)

# fc_rest_awake_clean.to_csv('./data_generated/fc_rest_awake_clean.csv', index=False, header=False)
# fc_rest_mild_clean.to_csv('./data_generated/fc_rest_mild_clean.csv', index=False, header=False)
# fc_rest_deep_clean.to_csv('./data_generated/fc_rest_deep_clean.csv', index=False, header=False)
# fc_rest_recovery_clean.to_csv('./data_generated/fc_rest_recovery_clean.csv', index=False, header=False)

# fc_movie_clean.to_csv('./data_generated/fc_movie_clean.csv', index=False, header=False)
# fc_rest_clean.to_csv('./data_generated/fc_rest_clean.csv', index=False, header=False)

# fc_movie_effect_of_movie.to_csv('./data_generated/fc_movie_effect_of_movie.csv', index=False, header=False)
# fc_rest_effect_of_movie.to_csv('./data_generated/fc_rest_effect_of_movie.csv', index=False, header=False)

# fc_rest_awake_last_60_TR_clean.to_csv('./data_generated/fc_rest_awake_last_60_TR_clean.csv', index=False, header=False)
# fc_rest_mild_last_60_TR_clean.to_csv('./data_generated/fc_rest_mild_last_60_TR_clean.csv', index=False, header=False)
# fc_rest_deep_last_60_TR_clean.to_csv('./data_generated/fc_rest_deep_last_60_TR_clean.csv', index=False, header=False)
# fc_rest_recovery_last_60_TR_clean.to_csv('./data_generated/fc_rest_recovery_last_60_TR_clean.csv', index=False, header=False)

# fc_double_rest_awake.to_csv('./data_generated/fc_double_rest_awake.csv', index=False, header=False)
# fc_double_movie_mild.to_csv('./data_generated/fc_double_movie_mild.csv', index=False, header=False)
# fc_double_movie_deep.to_csv('./data_generated/fc_double_movie_deep.csv', index=False, header=False)
# fc_double_movie.to_csv('./data_generated/fc_double_movie.csv', index=False, header=False)

# keep the nodes with fc values
fc_movie_awake_pls = fc_movie_awake[nodes_with_fc_values]
fc_movie_mild_pls = fc_movie_mild[nodes_with_fc_values]
fc_movie_deep_pls = fc_movie_deep[nodes_with_fc_values]
fc_movie_recovery_pls = fc_movie_recovery[nodes_with_fc_values]

fc_rest_awake_pls = fc_rest_awake[nodes_with_fc_values_rest]
fc_rest_mild_pls = fc_rest_mild[nodes_with_fc_values_rest]
fc_rest_deep_pls = fc_rest_deep[nodes_with_fc_values_rest]
fc_rest_recovery_pls = fc_rest_recovery[nodes_with_fc_values_rest]

fc_movie_effect_of_movie_pls = fc_movie_effect_of_movie[nodes_with_fc_values_effect_of_movie]
fc_rest_effect_of_movie_pls = fc_rest_effect_of_movie[nodes_with_fc_values_effect_of_movie]

fc_rest_awake_last_60_TR_pls = fc_rest_awake_last_60_TR[nodes_with_fc_values_rest_last_60_TR]
fc_rest_mild_last_60_TR_pls = fc_rest_mild_last_60_TR[nodes_with_fc_values_rest_last_60_TR]
fc_rest_deep_last_60_TR_pls = fc_rest_deep_last_60_TR[nodes_with_fc_values_rest_last_60_TR]
fc_rest_recovery_last_60_TR_pls = fc_rest_recovery_last_60_TR[nodes_with_fc_values_rest_last_60_TR]

fc_double_rest_awake_pls = fc_double_rest_awake[nodes_with_fc_values_double_three_way]
fc_double_movie_mild_pls = fc_double_movie_mild[nodes_with_fc_values_double_three_way]
fc_double_movie_deep_pls = fc_double_movie_deep[nodes_with_fc_values_double_three_way]

# take the mean of the fc matrices
fc_movie_awake_pls_mean = fc_movie_awake_pls.mean(axis=0)
fc_movie_mild_pls_mean = fc_movie_mild_pls.mean(axis=0)
fc_movie_deep_pls_mean = fc_movie_deep_pls.mean(axis=0)
fc_movie_recovery_pls_mean = fc_movie_recovery_pls.mean(axis=0)

fc_rest_awake_pls_mean = fc_rest_awake_pls.mean(axis=0)
fc_rest_mild_pls_mean = fc_rest_mild_pls.mean(axis=0)
fc_rest_deep_pls_mean = fc_rest_deep_pls.mean(axis=0)
fc_rest_recovery_pls_mean = fc_rest_recovery_pls.mean(axis=0)

fc_movie_mean = fc_movie_clean.mean(axis=0)
fc_rest_mean = fc_rest_clean.mean(axis=0)

fc_movie_effect_of_movie_pls_mean = fc_movie_effect_of_movie_pls.mean(axis=0)
fc_rest_effect_of_movie_pls_mean = fc_rest_effect_of_movie_pls.mean(axis=0)

fc_rest_awake_last_60_TR_pls_mean = fc_rest_awake_last_60_TR_pls.mean(axis=0)
fc_rest_mild_last_60_TR_pls_mean = fc_rest_mild_last_60_TR_pls.mean(axis=0)
fc_rest_deep_last_60_TR_pls_mean = fc_rest_deep_last_60_TR_pls.mean(axis=0)
fc_rest_recovery_last_60_TR_pls_mean = fc_rest_recovery_last_60_TR_pls.mean(axis=0)

fc_double_rest_awake_pls_mean = fc_double_rest_awake_pls.mean(axis=0)
fc_double_movie_mild_pls_mean = fc_double_movie_mild_pls.mean(axis=0)
fc_double_movie_deep_pls_mean = fc_double_movie_deep_pls.mean(axis=0)

# # t-test
# print(ttest_rel(fc_movie_awake_pls_mean, fc_movie_mild_pls_mean))
# print(fc_movie_mean.mean())
# print(fc_rest_mean.mean())

# # plot the mean fc matrices
# plt.boxplot([fc_movie_awake_pls_mean, fc_movie_mild_pls_mean, fc_movie_deep_pls_mean, fc_movie_recovery_pls_mean])
# plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
# plt.ylabel('mean fc')
# plt.show()
#
# plt.boxplot([fc_rest_awake_pls_mean, fc_rest_mild_pls_mean, fc_rest_deep_pls_mean, fc_rest_recovery_pls_mean])
# plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
# plt.ylabel('mean fc')
# plt.show()
#
plt.boxplot([fc_rest_effect_of_movie_pls_mean, fc_movie_effect_of_movie_pls_mean])
plt.xticks([1, 2], ['Rest', 'Narrative Listening'])
plt.ylabel('mean fc')
plt.show()
#
# plt.boxplot([fc_rest_awake_last_60_TR_pls_mean, fc_rest_mild_last_60_TR_pls_mean, fc_rest_deep_last_60_TR_pls_mean,
#                 fc_rest_recovery_last_60_TR_pls_mean])
# plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
# plt.ylabel('mean fc')
# plt.show()

# plot the mean fc matrices
plt.boxplot([fc_double_rest_awake_pls_mean, fc_double_movie_mild_pls_mean, fc_double_movie_deep_pls_mean])
plt.xticks([1, 2, 3], ['rest', 'movie_mild', 'movie_deep'])
plt.ylabel('mean fc')
plt.show()


# calculate the spearman correlation
def prepare_for_spearman_r(lv_vals, nodes_with_missing_values):
    u1 = lv_vals['u1'][:, 0]

    df = pd.DataFrame(u1, columns=['u1'])

    new_df = pd.DataFrame(data=np.nan, index=range(268), columns=df.columns)

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


# find the union set for all the missing nodes
nan_all_for_spearman_r = list(set(fc_movie_nan).union(set(fc_rest_nan_last_60_TR)).union(
    set(fc_effect_of_movie_nan).union(set(df_movie_missing)).union(set(df_last_60_TR_missing)).union(
        set(effect_of_movie_nan))))

fc_movie_loadings = prepare_for_spearman_r(fc_lv_vals_movie, nan_all_for_spearman_r)
fc_rest_loadings = prepare_for_spearman_r(fc_lv_vals_rest_last_60_TR, nan_all_for_spearman_r)
fc_effect_of_movie_loadings = prepare_for_spearman_r(fc_lv_vals_effect_of_movie, nan_all_for_spearman_r)
hurst_movie_loadings = prepare_for_spearman_r(lv_vals, nan_all_for_spearman_r)
hurst_rest_loadings = prepare_for_spearman_r(lv_vals_rest_last_60_TR, nan_all_for_spearman_r)
hurst_effect_of_movie_loadings = prepare_for_spearman_r(lv_vals_effect_of_movie, nan_all_for_spearman_r)

# stack the dataframes
loadings = pd.concat([fc_movie_loadings, fc_rest_loadings, fc_effect_of_movie_loadings, hurst_movie_loadings,
                      hurst_rest_loadings, hurst_effect_of_movie_loadings], axis=1).dropna(axis=0, how='any').to_numpy()

# calculate the spearman correlation
corr_movie = stats.spearmanr(loadings[:, 0], loadings[:, 3])
corr_rest = stats.spearmanr(loadings[:, 1], loadings[:, 4])
corr_effect_of_movie = stats.spearmanr(loadings[:, 2], loadings[:, 5])

# now, verify the results using the spin test
# load the surrogates
fc_surrogates_movie = np.load('fc_surrogates_movie.npy')
fc_surrogates_rest = np.load('fc_surrogates_rest.npy')
fc_surrogates_effect_of_movie = np.load('fc_surrogates_effect_of_movie.npy')

# preallocate the lists
pval_movie_surrogates = []
pval_rest_surrogates = []
pval_effect_of_movie_surrogates = []

def spin_test_for_hurst_versus_fc (hurst, fc, surrogates, n, empty_pval_list):
    corr = spearmanr(hurst, fc)[0]
    pvals = np.vstack([(np.abs(corr) < np.abs(spearmanr(fc, surrogates))).sum() / n])
    pvals = np.hstack([max(x, 1 / n) for x in pvals])
    empty_pval_list.append(pvals)

    return corr, empty_pval_list


corr_movie_surrogates, pval_movie_surrogates = spin_test_for_hurst_versus_fc(loadings[:, 0], loadings[:, 3],
                                                                             fc_surrogates_movie, 10000,
                                                                             pval_movie_surrogates)
corr_rest_surrogates, pval_rest_surrogates = spin_test_for_hurst_versus_fc(loadings[:, 1], loadings[:, 4],
                                                                            fc_surrogates_rest, 10000,
                                                                            pval_rest_surrogates)
corr_effect_of_movie_surrogates, pval_effect_of_movie_surrogates = spin_test_for_hurst_versus_fc(loadings[:, 2],
                                                                                                    loadings[:, 5],
                                                                                                    fc_surrogates_effect_of_movie,
                                                                                                    10000,
                                                                                                    pval_effect_of_movie_surrogates)