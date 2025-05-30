import os
import nibabel
import scipy.io as sio
from brainsmash.mapgen.base import Base
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker
from brainsmash.mapgen.stats import spearmanr, pearsonr
from scipy.stats import ttest_ind, ttest_rel
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from pathlib import Path
import numpy as np
from nilearn.plotting import plot_roi,view_img_on_surf, view_markers,view_img,view_connectome, find_parcellation_cut_coords,plot_stat_map
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Ellipse
import pandas as pd

from bar_chart import boot_ratio_path
from plotting_function import contrast_names, plot_preparation

n = 10000
# '8_condition', 'Effects of Narrative-Listening', 'Effects of Propofol', 'Effects of Propofol on Narrative-Listening'
contrast = 'Effects of Propofol on Narrative-Listening'
measurement = 'FC'

contrast_name = contrast_names[contrast]
path = f'./data_generated/PLS_results/PLS_outputTaskPLS{measurement}_{contrast_name}_lv_vals.mat'
boot_ratio_path = f'./data_generated/PLS_results/PLS_outputTaskPLS{measurement}_{contrast_name}.mat'
missing_nodes_path = f'./data_generated/Contrasts/{contrast}/{measurement}/missing_columns.csv'
missing_nodes = pd.read_csv(missing_nodes_path).values.flatten().tolist()

distance_mat = np.load('./data_generated/shen_distance.npy', allow_pickle=True)
kept_terms = np.load('./data_generated/neurosynth_terms.npy', allow_pickle=True)
kept_terms_maps = np.load('./data_generated/neurosynth_terms_parcel_maps.npy', allow_pickle=True)
term_surrogates = np.load('./data_generated/term_surrogates.npy', allow_pickle=True)
term_surrogates_deep = np.load('./data_generated/term_surrogates_deep.npy', allow_pickle=True)
term_surrogates_mild = np.load('./data_generated/term_surrogates_mild.npy', allow_pickle=True)
term_surrogates_mild_pls = np.load('./data_generated/term_surrogates_mild_pls.npy', allow_pickle=True)
term_surrogates_deep_pls = np.load('./data_generated/term_surrogates_deep_pls.npy', allow_pickle=True)



def neurosynth_hurst(file_path, kept_terms_maps, term_surrogates, p_value, df_movie_missing,
                     option='keep'):

    lv_vals = sio.loadmat(file_path)
    # remove the '_lv_vals' from the file name
    boot_ratio_name = os.path.basename(file_path).replace('_lv_vals.mat', '')
    boot_ratio_path = os.path.join('./data_generated/PLS_results', boot_ratio_name)
    boot_ratio = sio.loadmat(boot_ratio_path)
    hurst = lv_vals['u1'][:, 0]

    if option == 'keep':
        hurst = np.array([-x for x in hurst])

    if option == 'remove':
        processed_data = plot_preparation(lv_vals, boot_ratio, missing_nodes, keep=False, bs_thresh=3)
        hurst = processed_data.iloc[:, 0].tolist()
        hurst = [x for i, x in enumerate(hurst) if i not in df_movie_missing]
        hurst = [0 if pd.isna(x) else x for x in hurst]
        hurst = np.array([-x for x in hurst])

    print('hurst', hurst)
    # convert to boolean
    missing_columns = np.isin(np.arange(kept_terms_maps.shape[1]), df_movie_missing)
    # print(missing_columns)

    # remove nodes with missing values from kept_terms_maps
    kept_terms_maps = kept_terms_maps[:, ~missing_columns]

    # remove nodes with missing values from term_surrogates
    term_surrogates = term_surrogates[:, ~missing_columns]

    og_term_corrs = spearmanr(kept_terms_maps, hurst).flatten()
    pvals = np.vstack([(np.abs(og_term_corrs[i])<np.abs(spearmanr(hurst, term_surrogates[i*n:(i+1)*n, :]))).sum()/n for i in range(kept_terms_maps.shape[0])])
    pvals = np.hstack([max(x, 1/n) for x in pvals])
    pvals[np.isnan(og_term_corrs)] = 1.

    discoveries = multipletests(pvals, method='fdr_bh', alpha=p_value)[0]
    discovery_terms = kept_terms[discoveries]
    print(discovery_terms)
    discovery_corrs = og_term_corrs[discoveries]
    print(discovery_corrs)
    order = np.argsort(discovery_corrs)
    discovery_terms = discovery_terms[order]
    discovery_corrs = discovery_corrs[order]

    # Calculate the standard error of the correlation
    standard_error = np.sqrt((1 - np.square(og_term_corrs)) / (len(hurst) - 2))
    standard_error = standard_error[discoveries]
    standard_error = standard_error[order]
    print(standard_error)


    plt.clf()
    sns.set_theme(style='white', font='Arial')
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(1, 2, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    plt.xticks(rotation=60, ha='right')
    colors = ['#1f77b4' if corr < 0 else '#d62728' for corr in discovery_corrs]
    plt.bar(range(len(discovery_corrs)), discovery_corrs, tick_label=discovery_terms, yerr=standard_error[order], color=colors)
    plt.ylabel('correlation with PLS loadings')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_ylim(-.65, .7)

    ax = plt.subplot(1, 2, 1)
    order = np.argsort(og_term_corrs)
    plt.plot(range(len(pvals)), og_term_corrs[order], color='red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    neg_corr = [x for x in discovery_corrs if x < 0]
    pos_corr = [x for x in discovery_corrs if x > 0]
    x_start = len(neg_corr)
    rec_width = 116 - len(discovery_terms) - 1
    non_sig_corr = [x for x in og_term_corrs if x < np.min(pos_corr) and x > np.max(neg_corr)]
    y_start = np.min(non_sig_corr)
    rec_length = abs(np.min(non_sig_corr)) + abs(np.max(non_sig_corr))
    r = plt.Rectangle((x_start, y_start), rec_width, rec_length, color='gray', alpha=.5)
    ax.add_patch(r)
    plt.text(10, .62, r'$p_{null} > .01$', size=13)
    c = Ellipse((5, .63), 6, .04, color='gray', alpha=.5)
    ax.add_patch(c)
    plt.ylabel('correlation with PLS loadings')
    plt.xlabel('Neurosynth terms')
    ax.set_ylim(-.65, .65)
    plt.tight_layout()
    plt.savefig(f'./graphs/neurosynth_hurst_{measurement}_{contrast_name}.png', dpi=600)



if __name__ == '__main__':
    neurosynth_hurst(path, kept_terms_maps, term_surrogates, 0.01, missing_nodes)

    # remove
    # neurosynth_hurst(new_df_effect_of_movie, kept_terms_maps, term_surrogates, 0.01, df_movie_missing, option='remove')
    # p = neurosynth_hurst(new_df, kept_terms_maps, term_surrogates, 0.01, df_movie_missing, option='remove')

    # overlap_mask = np.array(overlap_mask)
    # movie_new = np.array(movie_new)
    # movie_new = np.nan_to_num(movie_new)
    # rest_new = np.array(rest_new)
    # rest_new = np.nan_to_num(rest_new)
    #
    # for data in [overlap_mask, movie_new, rest_new]:
    #     og_term_corrs = spearmanr(kept_terms_maps, data).flatten()
    #     pvals = np.vstack([(np.abs(og_term_corrs[i])<np.abs(spearmanr(data, term_surrogates[i*n:(i+1)*n, :]))).sum()/n for i in range(kept_terms_maps.shape[0])])
    #     pvals = np.hstack([max(x, 1/n) for x in pvals])
    #     pvals[np.isnan(og_term_corrs)] = 1.
    #
    #     discoveries = multipletests(pvals, method='fdr_bh', alpha=.05)[0]
    #     discovery_terms = kept_terms[discoveries]
    #     print(discovery_terms)
    #     discovery_corrs = og_term_corrs[discoveries]
    #     print(discovery_corrs)






# # compute the correlation between the first loadings and the rest of the loadings
# # load the matrix
# lv_vals_mild_map = sio.loadmat('./data_generated/u1_df_mild.mat')
# lv_vals_mild_map = lv_vals_mild_map['u1_df_mild']
# lv_vals_mild_first = lv_vals_mild_map[:, 0]
#
# lv_vals_deep_map = sio.loadmat('./data_generated/u1_df_deep.mat')
# lv_vals_deep_map = lv_vals_deep_map['u1_df_deep']
# lv_vals_deep_first = lv_vals_deep_map[:, 0]
#
#
# # preallocate a dataframe to store the results
# df_mild_corr: list = []
# df_mild_pval: list = []
# df_deep_corr: list = []
# df_deep_pval: list = []
#
#
# for i in range(lv_vals_mild_map.shape[1]):
#     corr = spearmanr(lv_vals_mild_first, lv_vals_mild_map[:,i]).flatten()
#     df_mild_corr.append(corr)
#     pvals = np.vstack([(np.abs(corr)<np.abs(spearmanr(lv_vals_mild_map[:,i],term_surrogates_mild_pls))).sum()/n])
#     pvals = np.hstack([max(x,1/n) for x in pvals])
#     df_mild_pval.append(pvals)
#
#
# df_mild_corr = np.array(df_mild_corr)
# df_mild_pval = np.array(df_mild_pval)
#
#
# for i in range(lv_vals_deep_map.shape[1]):
#     corr_deep = spearmanr(lv_vals_deep_first, lv_vals_deep_map[:,i]).flatten()
#     df_deep_corr.append(corr_deep)
#     pvals_deep = np.vstack([(np.abs(corr_deep)<np.abs(spearmanr(lv_vals_deep_map[:,i],term_surrogates_deep_pls))).sum()/n])
#     pvals_deep = np.hstack([max(x,1/n) for x in pvals_deep])
#     df_deep_pval.append(pvals_deep)
#
#
# df_deep_corr = np.array(df_deep_corr)
# df_deep_pval = np.array(df_deep_pval)
#
#
# # plot df_mild_corr and df_mild_pval together
# plt.plot(df_mild_corr, label='correlation')
# plt.ylabel('Spatial Correlation With First Window - Mild')
# plt.xlabel('Window Number')
# ax = plt.twinx()
# ax.plot(df_mild_pval, color='red', label='p-value')
# ax.axhline(0.05, color='black', linestyle='--', label='p = 0.05')
# plt.ylabel('p-value')
# plt.legend()
# plt.savefig('./graphs/PLS_mild_correlation.png')
# plt.show()
#
#
# plt.plot(df_deep_corr, label='correlation')
# plt.ylabel('Spatial Correlation With First Window - Deep')
# plt.xlabel('Window Number')
# ax = plt.twinx()
# ax.plot(df_deep_pval, color='red', label='p-value')
# ax.axhline(0.05, color='black', linestyle='--', label='p = 0.05')
# plt.ylabel('p-value')
# plt.legend()
# plt.savefig('./graphs/PLS_deep_correlation.png')
# plt.show()

# now, calculate the same for hurst values
# def read_hurst (directory: str):
#     for file in os.listdir(directory):
#         if file.endswith(".csv"):
#             file_path = os.path.join(directory, file)
#             df = pd.read_csv(file_path, header=None, index_col=None)
#             for i in nan_columns_mild:
#                 df.insert(i, f'NaN_{i}', np.nan)
#             # if df has less than 120 rows, add NaNs to the end
#             if df.shape[0] < 120:
#                 for i in range(120 - df.shape[0]):
#                     df.loc[df.shape[0]] = np.nan
#             df = df.to_numpy()
#             yield df
#
# def read_hurst_deep (directory: str):
#     for file in os.listdir(directory):
#         if file.endswith(".csv"):
#             file_path = os.path.join(directory, file)
#             df = pd.read_csv(file_path, header=None, index_col=None)
#             for i in nan_columns:
#                 df.insert(i, f'NaN_{i}', np.nan)
#             # if df has less than 120 rows, add NaNs to the end
#             if df.shape[0] < 80:
#                 for i in range(80 - df.shape[0]):
#                     df.loc[df.shape[0]] = np.nan
#             df = df.to_numpy()
#             yield df
#
#
# hurst_mild_corr: list = []
# hurst_mild_pval: list = []


# for df in read_hurst("./data_generated/windows_pls_mild"):
#     #  matrix to matrix correlation
#     corr = spearmanr(window_0_mild, df)
#     corr = corr[np.diag_indices_from(corr)]
    # # row to row correlation
    # for i in range(df.shape[0]):
    #     corr = spearmanr(window_0_mild[i,:], df[i,:]).flatten()
    #     hurst_mild_corr.append(corr)

    # array to array correlation
    # corr = spearmanr(window_0_mild.flatten(), df.flatten()).flatten()
    # hurst_mild_corr.append(corr)
    # calculate p-values for each correlation
    # perm_corr = np.zeros(n)
    # for i in range(n):
    #     print(i)
    #     perm_y = np.random.permutation(window_0_mild.flatten())
    #     perm_corr[i] = spearmanr(perm_y, df.flatten()).flatten()
    # pval = (np.abs(corr) < np.abs(perm_corr)).sum() / n
    # pval = max(pval, 1 / n)
    # hurst_mild_pval.append(pval)


# hurst_mild_corr = np.array(hurst_mild_corr)
# hurst_mild_pval = np.array(hurst_mild_pval)

# matrix to matrix correlation
# hurst_mild_corr = hurst_mild_corr.mean(axis=1)

# # row to row correlation
# hurst_mild_corr = np.split(hurst_mild_corr, 81)
# hurst_mild_corr = np.array([np.mean(x) for x in hurst_mild_corr])
# hurst_mild_corr = np.array(hurst_mild_corr)



# plt.plot(hurst_mild_corr, label='correlation')
# plt.ylabel('Hurst Value Correlation With First Window - Mild')
# plt.xlabel('Window Number')
# plt.ylim([0.3, 1])
# ax = plt.twinx()
# ax.plot(hurst_mild_pval, color='red', label='p-value')
# ax.set_ylim([0, 0.05])
# plt.ylabel('p-value')
# plt.legend()
# plt.show()

# hurst_deep_corr: list = []
# hurst_deep_pval: list = []
#
# for df in read_hurst_deep("./data_generated/windows_pls"):
#     corr = spearmanr(window_0_deep.flatten(), df.flatten()).flatten()
#     hurst_deep_corr.append(corr)
#     # calculate p-values for each correlation
#     perm_corr = np.zeros(n)
#     for i in range(n):
#         print(i)
#         perm_y = np.random.permutation(window_0_deep.flatten())
#         perm_corr[i] = spearmanr(perm_y, df.flatten()).flatten()
#     pval = (np.abs(corr) < np.abs(perm_corr)).sum() / n
#     pval = max(pval, 1 / n)
#     hurst_deep_pval.append(pval)
#
# hurst_deep_corr = np.array(hurst_deep_corr)
# hurst_deep_pval = np.array(hurst_deep_pval)
#
# plt.plot(hurst_deep_corr, label='correlation')
# plt.ylabel('Hurst Value Correlation With First Window - Deep')
# plt.xlabel('Window Number')
# plt.ylim([0.3, 1])
# ax = plt.twinx()
# ax.plot(hurst_deep_pval, color='red', label='p-value')
# ax.set_ylim([0, 0.05])
# plt.ylabel('p-value')
# plt.legend()
# plt.show()

# # perform a t-test on the correlation values
# ttest_ind(hurst_mild_corr, hurst_deep_corr)

# # perform a paired t-test on the correlation values
# ttest_rel(hurst_mild_corr, hurst_deep_corr)

# hurst_deep_corr_p: list = []
#
# window_0_deep_p = pd.read_csv('./data_generated/windows_pls/window_0.csv', header=None, index_col=None)
# window_0_deep_p = window_0_deep_p.to_numpy()
#
# for df in read_hurst_deep("./data_generated/windows_pls"):
#     if df.shape[0] == 80:
#         corr = pearsonr(window_0_deep_p.flatten(), df.flatten())[0]
#     else:
#         window_0_subset = window_0_deep_p[:df.shape[0],:]
#         corr = pearsonr(window_0_subset.flatten(), df.flatten())[0]
#     hurst_deep_corr_p.append(corr)
#
# hurst_deep_corr_p = np.array(hurst_deep_corr_p)
#
# plt.plot(hurst_deep_corr_p, label='correlation')
# plt.ylabel('Spatial Correlation With First Window - Deep')
# plt.xlabel('Window Number')
# plt.show()
#
# hurst_mild_corr_p: list = []
#
# window_0_mild_p = pd.read_csv('./data_generated/windows_pls_mild/window_0.csv', header=None, index_col=None)
# window_0_mild_p = window_0_mild_p.to_numpy()
#
# for df in read_hurst("./data_generated/windows_pls_mild"):
#     if df.shape[0] == 120:
#         corr = pearsonr(window_0_mild_p.flatten(), df.flatten())[0]
#     else:
#         window_0_subset = window_0_mild_p[:df.shape[0],:]
#         corr = pearsonr(window_0_subset.flatten(), df.flatten())[0]
#     hurst_mild_corr_p.append(corr)
#
# hurst_mild_corr_p = np.array(hurst_mild_corr_p)
#
# plt.plot(hurst_mild_corr_p, label='correlation')
# plt.ylabel('Spatial Correlation With First Window - Mild')
# plt.xlabel('Window Number')
# plt.show()



