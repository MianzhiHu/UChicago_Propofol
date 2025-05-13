from itertools import product
from matplotlib import pyplot as plt
from nilearn import datasets, surface, plotting
import nibabel
import numpy as np
import scipy.io as sio
import os
from plotting_function import process_contrast, brain_plotting, floor_to, ceil_to
import pandas as pd

measurements = ['FC', 'Hurst', 'Edges'] # 'Hurst', 'FC', 'Edges'
contrasts    = ['8_condition', 'Effects of Narrative-Listening', 'Effects of Propofol',
                'Effects of Propofol on Narrative-Listening']
# contrasts    = ['8_condition']


# if you need to reverse some combos but not others
reverse_map = {
    ('Hurst', '8_condition'): True,
    ('FC', '8_condition'): True,
    ('Edges', '8_condition'): True
}

keep_map = {
    ('Hurst', '8_condition'): False,
    ('FC', '8_condition'): False,
    ('Edges', '8_condition'): False
}

# for data in [hurst_effect_of_movie_full, hurst_rest_full, fc_rest_full, fc_effect_of_movie_full]:
#     print(min([x for x in data if str(x) != 'nan']))
#     print(max([x for x in data if str(x) != 'nan']))

def mask_generator(data1, data2, goal='overlap', method='percentage', threshold=0.5, std=None):
    if method == 'percentage':
        threshold_data1 = max(data1) * threshold
        threshold_data2 = max(data2) * threshold
    elif method == 'std':
        threshold_data1 = np.nanmean(data1) + std * np.nanstd(data1)
        threshold_data2 = np.nanmean(data2) + std * np.nanstd(data2)
    elif method == 'median':
        # select top 25% of the data
        threshold_data1 = np.nanpercentile(data1, 75)
        threshold_data2 = np.nanpercentile(data2, 75)

    if goal == 'overlap':
        mask_data1 = [1 if x > threshold_data1 else 0 for x in data1]
        mask_data2 = [1 if x > threshold_data2 else 0 for x in data2]
        overlap_mask = [i * j for i, j in zip(mask_data1, mask_data2)]
        print(f'overlap: {sum(overlap_mask)}')
        return overlap_mask

    elif goal == 'divergence':
        # the goal is to find the nodes for each individual data set that are above the threshold for their own data,
        # but below the threshold for the other data set
        mask_data1 = [1 if x > threshold_data1 else 0 for x in data1]
        mask_data2 = [1 if x < threshold_data2 else 0 for x in data2]
        divergence_mask = [i * j for i, j in zip(mask_data1, mask_data2)]
        print(f'divergence: {sum(divergence_mask)}')

        # use the mask to find the nodes that are above the threshold for data1, but below the threshold for data2
        divergence_nodes = [i for i, x in enumerate(divergence_mask) if x == 1]
        data1_new = [x if i in divergence_nodes else np.nan for i, x in enumerate(data1)]
        return data1_new
    return None



# overlap_mask = mask_generator(hurst_effect_of_movie_full, hurst_rest_full, goal='overlap', method='median')
# movie_new = mask_generator(hurst_effect_of_movie_full, hurst_rest_full, goal='divergence', method='median')
# rest_new = mask_generator(hurst_rest_full, hurst_effect_of_movie_full, goal='divergence', method='median')
#
# # for fc
# fc_overlap_mask = mask_generator(fc_effect_of_movie_full, fc_rest_full, goal='overlap', method='median')
# fc_movie_new = mask_generator(fc_effect_of_movie_full, fc_rest_full, goal='divergence', method='median')
# fc_rest_new = mask_generator(fc_rest_full, fc_effect_of_movie_full, goal='divergence', method='median')

# create the output directory
all_u1_data       = {}
all_min_loading = {}
all_max_loading = {}
all_significant   = {}

for m, c in product(measurements, contrasts):
    rev = reverse_map.get((m,c), False)
    keep = keep_map.get((m,c), False)
    u1_vals, min_loading, max_loading, sigs = process_contrast(m, c, reverse=rev, keep=keep, bs_thresh=3)
    all_u1_data[(m,c)]     = u1_vals
    all_min_loading[(m,c)] = min_loading
    all_max_loading[(m,c)] = max_loading
    all_significant[(m,c)] = sigs

if __name__ == '__main__':
    # # now plot the data
    # for m, c in product(measurements, contrasts):
    #     u1_vals = all_u1_data[(m,c)]
    #     min_loading = all_min_loading[(m,c)]
    #     max_loading = all_max_loading[(m,c)]
    #     sigs = all_significant[(m,c)]
    #     if len(u1_vals) == 0:
    #         print(f"Skipping {m}|{c}: no valid data.")
    #         continue
    #
    #     # find the missing columns
    #     missing_path = f'./data_generated/Contrasts/{c}/{m}/missing_columns.csv'
    #     missing_df = pd.read_csv(missing_path)
    #     missing_columns = missing_df['missing_column'].tolist()
    #
    #     if min_loading is None and max_loading is None:
    #         print(f"Skipping {m}|{c}: no valid data.")
    #         continue
    #
    #     if min_loading > 0:
    #         col = 'Reds'
    #         vmin = 0
    #         vmax = ceil_to(max_loading, 0.01)
    #     elif max_loading < 0:
    #         u1_vals = [-x for x in u1_vals]
    #         col = 'Blues'
    #         vmin = 0
    #         vmax = floor_to(min_loading, 0.01) * -1
    #     else:
    #         col = 'RdBu_r'
    #         vmin = floor_to(min_loading, 0.01)
    #         vmax = ceil_to(max_loading, 0.01)
    #
    #     print(f'max color: {vmax}, min color: {vmin}')
    #     brain_plotting(u1_vals, f'{m}_{c}', vmin, vmax, col, missing_columns, threshold=1e-32)
    #     plt.clf()

    # plot the data
    measurement_spec = 'FC'
    contrast_spec = 'Effects of Propofol on Narrative-Listening'
    u1_vals = all_u1_data[(measurement_spec, contrast_spec)]
    u1_vals = [(-x if not pd.isna(x) else x) for x in u1_vals]
    min_loading = all_min_loading[(measurement_spec, contrast_spec)]
    max_loading = all_max_loading[(measurement_spec, contrast_spec)]
    print(f'min_loading: {min_loading}, max_loading: {max_loading}')
    brain_plotting(u1_vals, f'{measurement_spec}_{contrast_spec}', 0, 0.135, 'Reds', threshold=1e-32)

    # # plot using nilearn
    # visual_attention = nibabel.load('./Neurosynth/visual perception_association-test_z_FDR_0.01.nii.gz')
    # stress = nibabel.load('./Neurosynth/stress_association-test_z_FDR_0.01.nii.gz')
    # fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 0.03]})
    # plotting.plot_glass_brain(stress, colorbar=False, cmap='Blues', axes=ax[1, 0], display_mode='lr')
    # ax[1, 0].set_title('stress', pad=20, size=25)
    # # add the other plot
    # plotting.plot_glass_brain(visual_attention, colorbar=False, cmap='Reds', axes=ax[0, 0], display_mode='lr')
    # ax[0, 0].set_title('visual attention', pad=20, size=25)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    # cax = fig.add_subplot(1, 30, 30)
    # cax1 = fig.add_subplot(1, 30, 29)
    # cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', norm=norm, orientation='vertical')
    # cb.set_label('z-score', size=25)
    # cb1 = matplotlib.colorbar.ColorbarBase(cax1, cmap='Blues', norm=norm, orientation='vertical')
    # cb1.set_ticks([])
    # plt.delaxes(ax[0, 1])
    # plt.delaxes(ax[1, 1])
    # plt.savefig('./graphs/Neurosynth_2_terms.png', dpi=600)
    # plt.show()


    # # similarly, plot the other three terms
    # thought = nibabel.load('./Neurosynth/thought_association-test_z_FDR_0.01.nii.gz')
    # updating = nibabel.load('./Neurosynth/updating_association-test_z_FDR_0.01.nii.gz')
    # empathy = nibabel.load('./Neurosynth/empathy_association-test_z_FDR_0.01.nii.gz')
    # fig, ax = plt.subplots(3, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1, 0.03]})
    # plotting.plot_glass_brain(thought, colorbar=False, cmap='Blues', axes=ax[0, 0], display_mode='lr')
    # ax[0, 0].set_title('thought', pad=20, size=25)
    # # add the other plot
    # plotting.plot_glass_brain(updating, colorbar=False, cmap='Blues', axes=ax[1, 0], display_mode='lr')
    # ax[1, 0].set_title('updating', pad=20, size=25)
    # # add the other plot
    # plotting.plot_glass_brain(empathy, colorbar=False, cmap='Reds', axes=ax[2, 0], display_mode='lr')
    # ax[2, 0].set_title('empathy', pad=20, size=25)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    # cax = fig.add_subplot(1, 30, 30)
    # cax1 = fig.add_subplot(1, 30, 29)
    # cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', norm=norm, orientation='vertical')
    # cb.set_label('z-score', size=25)
    # cb1 = matplotlib.colorbar.ColorbarBase(cax1, cmap='Blues', norm=norm, orientation='vertical')
    # cb1.set_ticks([])
    # plt.delaxes(ax[0, 1])
    # plt.delaxes(ax[1, 1])
    # plt.delaxes(ax[2, 1])
    # plt.savefig('./graphs/Neurosynth_3_terms.png', dpi=600)
    # plt.show()

    # # do the same for the other two terms
    # adaptation = nibabel.load('./Neurosynth/adaptation_association-test_z_FDR_0.01.nii.gz')
    # belief = nibabel.load('./Neurosynth/belief_association-test_z_FDR_0.01.nii.gz')
    # fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 0.03]})
    # plotting.plot_glass_brain(adaptation, colorbar=False, cmap='Reds', axes=ax[0, 0], display_mode='lr')
    # ax[0, 0].set_title('adaptation', pad=20, size=25)
    # # add the other plot
    # plotting.plot_glass_brain(belief, colorbar=False, cmap='Blues', axes=ax[1, 0], display_mode='lr')
    # ax[1, 0].set_title('belief', pad=20, size=25)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    # cax = fig.add_subplot(1, 30, 30)
    # cax1 = fig.add_subplot(1, 30, 29)
    # cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', norm=norm, orientation='vertical')
    # cb.set_label('z-score', size=25)
    # cb1 = matplotlib.colorbar.ColorbarBase(cax1, cmap='Blues', norm=norm, orientation='vertical')
    # cb1.set_ticks([])
    # plt.delaxes(ax[0, 1])
    # plt.delaxes(ax[1, 1])
    # plt.show()

    # # plot for specific network
    # node_numbers = np.load('./data_generated/nodes_with_hurst_values.npy')  # load node numbers
    # network_label = pd.read_csv('./atlasTransform/atlasTransform/data/shen_268/shen_268_parcellation_networklabels.csv')
    # # filter the network label DataFrame to only include rows for the specified node numbers
    # network_label['Node'] = network_label['Node'] - 1
    # network_label_filtered = network_label[network_label['Node'].isin(node_numbers)]
    # network_label_filtered = network_label_filtered[network_label_filtered['Network'] == 4]
    # # save the node numbers of the nodes
    # nodes_sc = network_label_filtered['Node'].tolist()
    # # in hurst, keep only the values of the nodes in the SC network, and replace the rest with NaN
    # hurst_sc = [np.nan if i not in nodes_sc else hurst[i] for i in range(268)]



