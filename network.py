import numpy as np
import pandas as pd
import nibabel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from nilearn import plotting
from nilearn.datasets import load_mni152_template
from nilearn.image import new_img_like
from scipy.stats import stats, tukey_hsd, hypergeom
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from brain_plotting import hurst, hurst_movie_03, hurst_movie_02, hurst_movie_01_3, fc_movie, fc_rest, fc_movie_abs, \
    hurst_last_60_TR, fc_double_two_way, fc_rest_last_60_TR, hurst_double_two_way, hurst_effect_of_movie, fc_effect_of_movie
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests


def find_network(file_path: str, hurst=None, condition: str = None):
    node_numbers = np.load(file_path)  # load node numbers
    network_label = pd.read_csv('./atlasTransform/atlasTransform/data/shen_268/shen_268_parcellation_networklabels.csv')
    # filter the network label DataFrame to only include rows for the specified node numbers
    network_label_filtered = network_label[network_label['Node'].isin(node_numbers+1)]
    network_name = pd.read_csv('./atlasTransform/atlasTransform/data/shen_268/network_descriptions.csv')
    network_label_filtered['Network'] = network_label_filtered['Network'].map(network_name.set_index('Network')['name'])
    print(network_label_filtered)
    # count the number of nodes in each network
    value_count = network_label_filtered['Network'].value_counts()

    # create a dictionary to store the p-values for each network
    p_values = {}
    # iterate through each network
    for network in value_count.index:
        # calculate the p-value for the hypergeometric test
        p = hypergeom.sf(value_count[network] - 1, len(network_label_filtered),
                         len(network_label_filtered) - value_count[network], value_count[network])
        # store the p-value in the dictionary
        p_values[network] = p
    # correct the p-values using the Benjamini-Hochberg procedure
    p_values_corrected = multipletests(list(p_values.values()), method='fdr_bh')[1]
    # create a dictionary to store the corrected p-values for each network
    p_values_corrected_dict = {}
    # iterate through each network
    for i in range(len(p_values_corrected)):
        # store the corrected p-value in the dictionary
        p_values_corrected_dict[list(p_values.keys())[i]] = p_values_corrected[i]
    # create a dictionary to store the number of nodes in each network
    network_node_count = {}
    # iterate through each network
    for network in value_count.index:
        # store the number of nodes in the network in the dictionary
        network_node_count[network] = value_count[network]
    # create a dictionary to store the p-values and number of nodes in each network
    network_p_values = {}
    # iterate through each network
    for network in value_count.index:
        # check if the denominator is zero
        denominator = len(network_label_filtered[network_label_filtered['Network'] == network])
        if denominator == 0:
            # set the p-value to NaN if the denominator is zero
            p_value = np.nan
        else:
            # get the corrected p-value from the dictionary
            p_value = p_values_corrected_dict[network]
        # store the p-value and number of nodes in the network in the dictionary
        network_p_values[network] = [p_value, network_node_count[network]]
    # create a DataFrame from the dictionary
    network_p_values_df = pd.DataFrame.from_dict(network_p_values, orient='index', columns=['p-value', 'Node Count'])
    # sort the DataFrame by the p-values
    network_p_values_df = network_p_values_df.sort_values(by=['p-value'])
    # print the DataFrame
    print(network_p_values_df)

    bl_filtered = []
    for value in hurst:
        if not np.isnan(value):
            bl_filtered.append(value)
    absolute_bl = [abs(value) for value in bl_filtered]
    network_label_filtered['Brain Loading'] = absolute_bl
    # group the network label DataFrame by network
    network_label_grouped = network_label_filtered.groupby('Network')
    bl_values = [group['Brain Loading'].values for name, group in network_label_grouped]
    # conduct a one-way ANOVA
    f, p = stats.f_oneway(*bl_values)
    print('F value: ' + str(f))
    print('P value: ' + str(p))
    # return the both within group and between group degrees of freedom
    df_within = len(bl_filtered) - len(bl_values)
    df_between = len(bl_values) - 1
    print('Degrees of freedom between groups: ' + str(df_between))
    print('Degrees of freedom within groups: ' + str(df_within))
    # conduct a Tukey's HSD test
    # posthoc = tukey_hsd(*bl_values)
    posthoc = sp.posthoc_tukey(network_label_filtered, val_col='Brain Loading', group_col='Network')
    network_bl = network_label_filtered.groupby('Network')['Brain Loading'].mean()
    # network_bl = network_bl.sort_values(ascending=False)

    # plot the results
    network_order = ['medial frontal', 'frontoparietal', 'default mode', 'subcortical-cerebellar', 'motor', 'primary visual', 'secondary visual', 'visual association']
    value_count = value_count.reindex(network_order)
    network_bl = network_bl.reindex(network_order)

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    ax[0].bar(value_count.index, value_count.values)
    ax[0].set_title('Number of Nodes in Each Network for ' + condition)
    ax[0].set_xlabel('Network')
    ax[0].set_ylabel('Number of Nodes')
    ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_xticklabels(rotation=45, ha='right', labels=value_count.index)
    fig.subplots_adjust(bottom=0.3)
    ax[1].bar(network_bl.index, network_bl.values)
    # manually code the order of the x-axis labels
    ax[1].set_xticks(range(len(value_count.index)))
    ax[1].set_title('Mean Absolute Brain Loadings for Each Network for ' + condition)
    ax[1].set_xlabel('Network')
    ax[1].set_ylabel('Mean Absolute Brain Loadings')
    ax[1].set_ylim(bottom=0.05)
    # # set the number of ticks on the y-axis to start at 0.07 and end at 0.12
    # ax[1].yaxis.set_ticks(np.arange(0.05, 0.119, 0.01))
    ax[1].set_xticklabels(rotation=45, ha='right', labels=network_bl.index)
    fig.subplots_adjust(bottom=0.3)
    plt.savefig('./graphs/brain_loading_' + condition + '.png')
    plt.show()

    # # plot the number of nodes in each network against the mean absolute brain loading
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.scatter(value_count.values, network_bl.values)
    # ax.set_title('Number of Nodes vs. Mean Absolute Brain Loading for ' + condition)
    # ax.set_xlabel('Number of Nodes')
    # ax.set_ylabel('Mean Absolute Brain Loading')
    # plt.show()

    return posthoc


posthoc_all = find_network('./data_generated/nodes_with_hurst_values.npy', hurst, 'Narrative Listening + Propofol - H')
# posthoc_mild = find_network('./data_generated/nodes_with_hurst_values_02.npy', hurst_movie_02, 'Mild Sedation')
# posthoc_deep = find_network('./data_generated/nodes_with_hurst_values_03.npy', hurst_movie_03, 'Deep Sedation')
# find_network('./data_generated/nodes_with_hurst_values_01_3.npy', hurst_movie_01_3, 'awake')
posthoc_rest = find_network('./data_generated/nodes_with_hurst_values_last_60_TR.npy', hurst_last_60_TR, 'Propofol - H')
posthoc_effect_of_movie = find_network('./data_generated/nodes_with_hurst_values_effect_of_movie.npy', hurst_effect_of_movie, 'Narrative Listening - H')


posthoc_fc_movie = find_network('./data_generated/nodes_with_fc_values.npy', fc_movie, 'Narrative Listening + Propofol - FC')
# posthoc_fc_rest = find_network('./data_generated/nodes_with_fc_values_rest.npy', fc_rest, 'Resting State')
# posthoc_fc_movie_abs = find_network('./data_generated/nodes_with_fc_values_abs.npy', fc_movie_abs, 'Narrative Listening')
# posthoc_fc_double_two_way = find_network('./data_generated/nodes_with_fc_values_double_two_way.npy', fc_double_two_way, 'Propofol + Narrative Listening')
posthoc_fc_rest_last_60_TR = find_network('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy', fc_rest_last_60_TR, 'Propofol - FC')
# posthoc_hurst_double_two_way = find_network('./data_generated/nodes_with_hurst_double_two_way.npy', hurst_double_two_way, 'Propofol + Narrative Listening')
posthoc_fc_effect_of_movie = find_network('./data_generated/nodes_with_fc_values_effect_of_movie.npy', fc_effect_of_movie, 'Narrative Listening - FC')

# # plot the atlas
# atlas = load_shen_268(1)
# dr = atlas.get_data().astype(float)
# labels = np.unique(dr)
# labels = labels[labels != 0]
# new_labels = pd.read_csv('./atlasTransform/atlasTransform/data/shen_268/shen_268_parcellation_networklabels.csv')
# new_labels = new_labels['Network'].values
# for i, label in enumerate(labels):
#     dr[dr == label] = new_labels[i]
# new_atlas = new_img_like(atlas, dr)
#
#
# # plot the atlas
# fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 0.05]})
# cmap = ListedColormap(['#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3', '#FF00FF'])
# plotting.plot_roi(new_atlas, axes=ax[0], title='Shen 268 atlas', draw_cross=False, colorbar=False, cmap=cmap, bg_img=load_mni152_template(resolution=1), display_mode='ortho', cut_coords=(0, 0, 0), annotate=False, black_bg=False)
# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax[1], orientation='horizontal')
# cbar.set_ticks(np.arange(0, 0.8, 0.1))
# cbar.mappable.set_clim(vmin=0, vmax=0.8)
# cbar.set_ticklabels(['medial frontal', 'frontoparietal', 'default mode', 'subcortical-cerebellar', 'motor', 'primary visual', 'secondary visual', 'visual association'], rotation=45, size=15)
# fig.subplots_adjust(bottom=0.20)
# plt.show()

# posthoc = sp.posthoc_tukey(network, val_col='Brain Loading', group_col='Network')
# pvalues = posthoc.where(np.triu(np.ones(posthoc.shape), k=1).astype(bool))
# comparisons = [(posthoc.index[i], posthoc.columns[j]) for i, j in np.transpose(np.where(pvalues.notna()))]
# pvalues = pvalues.stack().tolist()
# # correct the p-values for multiple comparisons
# final_p_values = multipletests(pvalues, method='bonferroni')[1]
#
# network_label_grouped = network.groupby('Network')
# bl_values = [group['Brain Loading'].values for name, group in network_label_grouped]
# posthoc_scipy = tukey_hsd(*bl_values)
