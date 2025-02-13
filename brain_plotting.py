import matplotlib
from matplotlib import pyplot as plt, colors
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from nilearn import datasets, surface, plotting
import nibabel
import numpy as np
from plotting_preparation import new_df, new_df_movie_03, new_df_movie_02, new_df_movie_01, new_df_movie_01_3, \
    new_df_movie_everything, new_df_movie_03_30, new_df_fc_movie, new_df_fc_rest, new_df_fc_movie_abs, \
    new_df_fc_effect_of_movie, new_df_rest_last_60_TR, new_df_effect_of_movie, new_df_fc_rest_last_60_TR, \
    new_df_double_three_way, new_df_double_two_way, new_df_double_merged, new_df_fc_double_three_way, \
    new_df_fc_double_two_way, new_df_fc_double_merged, new_df_combined, new_df_fc_combined, new_df_fc_everything, \
    hurst_effect_of_movie_full, hurst_rest_full, fc_rest_full, fc_effect_of_movie_full, general_fc_everything, \
    general_fc_movie
from make_side_by_side_surf_plots import make_side_by_side_surf_plots, make_side_by_side_surf_plots_left
import pandas as pd

# select only the first column of the new_df dataframe and save it to a list
hurst = new_df.iloc[:, 0].tolist()
hurst_movie_03 = new_df_movie_03.iloc[:, 0].tolist()
hurst_movie_03_30 = new_df_movie_03_30.iloc[:, 0].tolist()
hurst_movie_02 = new_df_movie_02.iloc[:, 0].tolist()
hurst_movie_01 = new_df_movie_01.iloc[:, 0].tolist()
# hurst_movie_01_2 = new_df_movie_01_2.iloc[:, 0].tolist()
hurst_movie_01_3 = new_df_movie_01_3.iloc[:, 0].tolist()
hurst_everything = new_df_movie_everything.iloc[:, 0].tolist()
hurst_last_60_TR = new_df_rest_last_60_TR.iloc[:, 0].tolist()
hurst_effect_of_movie = new_df_effect_of_movie.iloc[:, 0].tolist()
hurst_double_three_way = new_df_double_three_way.iloc[:, 0].tolist()
hurst_double_two_way = new_df_double_two_way.iloc[:, 0].tolist()
hurst_double_merged = new_df_double_merged.iloc[:, 0].tolist()
hurst_combined = new_df_combined.iloc[:, 0].tolist()
hurst_effect_of_movie_full = hurst_effect_of_movie_full.iloc[:, 0].tolist()
hurst_rest_full = hurst_rest_full.iloc[:, 0].tolist()

fc_movie = new_df_fc_movie.iloc[:, 0].tolist()
fc_rest = new_df_fc_rest.iloc[:, 0].tolist()
fc_rest_last_60_TR = new_df_fc_rest_last_60_TR.iloc[:, 0].tolist()
fc_movie_abs = new_df_fc_movie_abs.iloc[:, 0].tolist()
fc_effect_of_movie = new_df_fc_effect_of_movie.iloc[:, 0].tolist()
fc_double_three_way = new_df_fc_double_three_way.iloc[:, 0].tolist()
fc_double_two_way = new_df_fc_double_two_way.iloc[:, 0].tolist()
fc_double_merged = new_df_fc_double_merged.iloc[:, 0].tolist()
fc_combined = new_df_fc_combined.iloc[:, 0].tolist()
fc_everything = new_df_fc_everything.iloc[:, 0].tolist()
fc_effect_of_movie_full = fc_effect_of_movie_full.iloc[:, 0].tolist()
fc_rest_full = fc_rest_full.iloc[:, 0].tolist()

general_fc_everything = general_fc_everything.iloc[:, 0].tolist()
general_fc_movie = general_fc_movie.iloc[:, 0].tolist()

# # save the node numbers where the hurst values are not NaN
# nodes_with_hurst_values = [i for i, x in enumerate(hurst) if str(x) != 'nan']
# nodes_with_hurst_values_03 = [i for i, x in enumerate(hurst_movie_03) if str(x) != 'nan']
# nodes_with_hurst_values_02 = [i for i, x in enumerate(hurst_movie_02) if str(x) != 'nan']
# nodes_with_hurst_values_01_3 = [i for i, x in enumerate(hurst_movie_01_3) if str(x) != 'nan']
# nodes_with_hurst_values_everything = [i for i, x in enumerate(hurst_everything) if str(x) != 'nan']
# nodes_with_hurst_values_last_60_TR = [i for i, x in enumerate(hurst_last_60_TR) if str(x) != 'nan']
# nodes_with_hurst_values_effect_of_movie = [i for i, x in enumerate(hurst_effect_of_movie) if str(x) != 'nan']
# nodes_with_hurst_double_three_way = [i for i, x in enumerate(hurst_double_three_way) if str(x) != 'nan']
# nodes_with_hurst_double_two_way = [i for i, x in enumerate(hurst_double_two_way) if str(x) != 'nan']
# nodes_with_hurst_combined = [i for i, x in enumerate(hurst_combined) if str(x) != 'nan']
# nodes_with_hurst_everything = [i for i, x in enumerate(hurst_everything) if str(x) != 'nan']

# nodes_with_fc_values = [i for i, x in enumerate(fc_movie) if str(x) != 'nan']
# nodes_with_fc_values_rest = [i for i, x in enumerate(fc_rest) if str(x) != 'nan']
# nodes_with_fc_values_rest_last_60_TR = [i for i, x in enumerate(fc_rest_last_60_TR) if str(x) != 'nan']
# nodes_with_fc_values_abs = [i for i, x in enumerate(fc_movie_abs) if str(x) != 'nan']
# nodes_with_fc_values_effect_of_movie = [i for i, x in enumerate(fc_effect_of_movie) if str(x) != 'nan']
# nodes_with_fc_values_double_three_way = [i for i, x in enumerate(fc_double_three_way) if str(x) != 'nan']
# nodes_with_fc_values_double_two_way = [i for i, x in enumerate(fc_double_two_way) if str(x) != 'nan']
# nodes_with_fc_values_combined = [i for i, x in enumerate(fc_combined) if str(x) != 'nan']
# nodes_with_fc_values_everything = [i for i, x in enumerate(fc_everything) if str(x) != 'nan']

# # save the list as a .npy file
# np.save('./data_generated/nodes_with_hurst_values.npy', nodes_with_hurst_values)
# np.save('./data_generated/nodes_with_hurst_values_03.npy', nodes_with_hurst_values_03)
# np.save('./data_generated/nodes_with_hurst_values_02.npy', nodes_with_hurst_values_02)
# np.save('./data_generated/nodes_with_hurst_values_01_3.npy', nodes_with_hurst_values_01_3)
# np.save('./data_generated/nodes_with_hurst_values_last_60_TR.npy', nodes_with_hurst_values_last_60_TR)
# np.save('./data_generated/nodes_with_fc_values.npy', nodes_with_fc_values)
# np.save('./data_generated/nodes_with_fc_values_rest.npy', nodes_with_fc_values_rest)
# np.save('./data_generated/nodes_with_fc_values_abs.npy', nodes_with_fc_values_abs)
# np.save('./data_generated/nodes_with_fc_values_effect_of_movie.npy', nodes_with_fc_values_effect_of_movie)
# np.save('./data_generated/nodes_with_hurst_values_effect_of_movie.npy', nodes_with_hurst_values_effect_of_movie)
# np.save('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy', nodes_with_fc_values_rest_last_60_TR)
# np.save('./data_generated/nodes_with_fc_values_double_three_way.npy', nodes_with_fc_values_double_three_way)
# np.save('./data_generated/nodes_with_fc_values_double_two_way.npy', nodes_with_fc_values_double_two_way)
# np.save('./data_generated/nodes_with_hurst_double_three_way.npy', nodes_with_hurst_double_three_way)
# np.save('./data_generated/nodes_with_hurst_double_two_way.npy', nodes_with_hurst_double_two_way)
# np.save('./data_generated/nodes_with_hurst_combined.npy', nodes_with_hurst_combined)
# np.save('./data_generated/nodes_with_fc_values_combined.npy', nodes_with_fc_values_combined)
# np.save('./data_generated/nodes_with_fc_values_everything.npy', nodes_with_fc_values_everything)
# np.save('./data_generated/nodes_with_hurst_values_everything.npy', nodes_with_hurst_everything)


# check the range of the hurst values discarding NaN values
df_of_interest = general_fc_movie
print(min([x for x in df_of_interest if str(x) != 'nan']))
print(max([x for x in df_of_interest if str(x) != 'nan']))
print(len([x for x in df_of_interest if str(x) != 'nan']))

# convert the list to negative values
hurst = [-x for x in hurst]
hurst_movie_03 = [-x for x in hurst_movie_03]
hurst_movie_03_30 = [-x for x in hurst_movie_03_30]
hurst_movie_02 = [-x for x in hurst_movie_02]
hurst_movie_01_3 = [-x for x in hurst_movie_01_3]
hurst_everything = [-x for x in hurst_everything]
hurst_last_60_TR = [-x for x in hurst_last_60_TR]
hurst_effect_of_movie = [-x for x in hurst_effect_of_movie]
hurst_double_three_way = [-x for x in hurst_double_three_way]
hurst_double_two_way = [-x for x in hurst_double_two_way]
hurst_double_merged = [-x for x in hurst_double_merged]
hurst_combined = [-x for x in hurst_combined]
hurst_effect_of_movie_full = [-x for x in hurst_effect_of_movie_full]
hurst_rest_full = [-x for x in hurst_rest_full]

fc_movie = [-x for x in fc_movie]
fc_movie_abs = [-x for x in fc_movie_abs]
fc_effect_of_movie = [-x for x in fc_effect_of_movie]
fc_combined = [-x for x in fc_combined]
fc_everything = [-x for x in fc_everything]
fc_effect_of_movie_full = [-x for x in fc_effect_of_movie_full]

general_fc_everything = [-x for x in general_fc_everything]
general_fc_movie = [-x for x in general_fc_movie]

for data in [hurst_effect_of_movie_full, hurst_rest_full, fc_rest_full, fc_effect_of_movie_full]:
    print(min([x for x in data if str(x) != 'nan']))
    print(max([x for x in data if str(x) != 'nan']))


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


overlap_mask = mask_generator(hurst_effect_of_movie_full, hurst_rest_full, goal='overlap', method='median')
movie_new = mask_generator(hurst_effect_of_movie_full, hurst_rest_full, goal='divergence', method='median')
rest_new = mask_generator(hurst_rest_full, hurst_effect_of_movie_full, goal='divergence', method='median')

# for fc
fc_overlap_mask = mask_generator(fc_effect_of_movie_full, fc_rest_full, goal='overlap', method='median')
fc_movie_new = mask_generator(fc_effect_of_movie_full, fc_rest_full, goal='divergence', method='median')
fc_rest_new = mask_generator(fc_rest_full, fc_effect_of_movie_full, goal='divergence', method='median')

def brain_plotting (df, title, vmin, vmax, cmap, nodes_with_missing_values=None):
    # check if df has 268 elements
    if len(df) == 268:
        pass
    else:
        # create a list of 268 nan values
        nan_list = [np.nan] * 268
        # replace the nan values with the values of the df
        j = 0
        for i in range(268):
            if i not in nodes_with_missing_values:
                nan_list[i] = df[j]
                j += 1
        # replace the df with the new list
        df = nan_list
    fsaverage = datasets.fetch_surf_fsaverage()
    atlas = load_shen_268(1)
    dr = atlas.get_fdata()
    dd = dr.copy().astype('float')
    labels = np.unique(dr)
    for i in np.array(list(range(268))):
        dd[dr == labels[i+1]] = df[i]
    new_image_atl = nibabel.Nifti1Image(dd, atlas.affine)
    texture = surface.vol_to_surf(new_image_atl, fsaverage.pial_right)
    make_side_by_side_surf_plots(title,texture,vmin=vmin,vmax=vmax,cmap=cmap)


def brain_plotting_left (df, title, vmin, vmax, cmap, nodes_with_missing_values=None):
    # check if df has 268 elements
    if len(df) == 268:
        pass
    else:
        # create a list of 268 nan values
        nan_list = [np.nan] * 268
        # replace the nan values with the values of the df
        j = 0
        for i in range(268):
            if i not in nodes_with_missing_values:
                nan_list[i] = df[j]
                j += 1
        # replace the df with the new list
        df = nan_list
    fsaverage = datasets.fetch_surf_fsaverage()
    atlas = load_shen_268(1)
    dr = atlas.get_fdata()
    dd = dr.copy().astype('float')
    labels = np.unique(dr)
    for i in np.array(list(range(268))):
        dd[dr == labels[i+1]] = df[i]
    new_image_atl = nibabel.Nifti1Image(dd, atlas.affine)
    texture = surface.vol_to_surf(new_image_atl, fsaverage.pial_left)
    make_side_by_side_surf_plots_left(title, texture, vmin=vmin, vmax=vmax, cmap=cmap)


# if __name__ == '__main__':
    # brain_plotting(hurst, 'brain loadings', 0, 0.15, 'Reds')
    # brain_plotting(hurst_sc, 'brain loadings', 0, 0.15, 'Reds')
    # brain_plotting(hurst_movie_03, 'brain loadings - deep', -0.3, 0.3, 'RdBu_r')
    # brain_plotting(hurst_movie_03_30, 'brain loadings_03_30', -0.3, 0.3, 'RdBu_r')
    # brain_plotting(hurst_movie_02, 'brain loadings - mild', -0.3, 0.3, 'RdBu_r')
    # brain_plotting(hurst_movie_01, 'brain loadings_01', -0.03, 0.15, 'Reds')
    # brain_plotting(hurst_movie_01_2, 'brain loadings_01_2', -0.2, 0.2, 'RdBu_r')
    # brain_plotting(hurst_movie_01_3, 'brain loadings_01_3', -0.2, 0.2, 'RdBu_r')
    # brain_plotting(hurst_everything, 'brain loadings - all', 0, 0.15, 'Reds')
    # brain_plotting(fc_movie, 'brain loadings - fc', -0.25, 0.25, 'RdBu_r')
    # brain_plotting(fc_rest, 'brain loadings - fc', 0, 0.25, 'Blues')
    # brain_plotting(fc_rest_last_60_TR, 'Effect of Propofol - FC', 0, 0.2, 'Blues')
    # brain_plotting(fc_movie_abs, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting(fc_effect_of_movie, 'Effect of Narrative Listening - FC', vmin=0, vmax=0.2, cmap='Blues')
    # brain_plotting(fc_double_three_way, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting(fc_double_two_way, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting(fc_double_merged, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting(fc_everything, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting(hurst_last_60_TR, 'Effect of Propofol - Hurst', 0, 0.16, 'Blues')
    # brain_plotting(hurst_effect_of_movie, 'Effect of Narrative Listening - Hurst', vmin=0, vmax=0.15, cmap='Blues')
    # brain_plotting(hurst_double_three_way, 'brain loadings - double three way', 0, 0.2, 'Blues')
    # brain_plotting(hurst_double_two_way, 'brain loadings - double two way', 0, 0.2, 'Blues')
    # brain_plotting(hurst_double_merged, 'brain loadings - double merged', 0, 0.2, 'Blues')
    # brain_plotting(hurst_combined, 'Combined Effects - Hurst', 0, 0.15, 'Blues')
    # brain_plotting(fc_combined, 'Combined Effects - FC', 0, 0.20, 'Blues')
    # brain_plotting(fc_everything, 'Everything - FC', 0, 0.20, 'Blues')

    # brain_plotting(hurst_effect_of_movie_full, 'Effect of Narrative Listening - Hurst', 0, 0.2, 'Reds')
    # brain_plotting(hurst_rest_full, 'Effect of Propofol - Hurst', 0, 0.2, 'Blues')
    #
    # brain_plotting(fc_effect_of_movie_full, 'Effect of Narrative Listening - FC', 0, 0.2, 'Reds')
    # brain_plotting(fc_rest_full, 'Effect of Propofol - FC', 0, 0.2, 'Blues')
    #
    # brain_plotting(overlap_mask, 'Difference in Hurst Exponents', 0, 1, 'Greens')
    # brain_plotting(movie_new, 'Difference in Hurst Exponents', 0, .15, 'Reds')
    # brain_plotting(rest_new, 'Difference in Hurst Exponents', 0, .2, 'Blues')
    #
    # brain_plotting(fc_overlap_mask, 'Difference in FC', 0, 1, 'Greens')
    # brain_plotting(fc_movie_new, 'Difference in FC', 0, .2, 'Reds')
    # brain_plotting(fc_rest_new, 'Difference in FC', 0, .2, 'Blues')
    # brain_plotting(general_fc_everything, 'Everything - general FC', 0, 0.11, 'Blues')
    # brain_plotting(general_fc_movie, 'Movie - general FC', 0, 0.14, 'Reds')

    # LEFT HEMISPHERE PLOTTING
    # brain_plotting_left(hurst, 'brain loadings', 0, 0.15, 'Reds')
    # brain_plotting_left(hurst_sc, 'brain loadings', 0, 0.15, 'Reds')
    # brain_plotting_left(hurst_movie_03, 'brain loadings - deep_left', -0.3, 0.3, 'RdBu_r')
    # brain_plotting_left(hurst_movie_02, 'brain loadings - mild_left', -0.3, 0.3, 'RdBu_r')
    # brain_plotting_left(hurst_movie_01_3, 'brain loadings_01_3', -0.2, 0.2, 'RdBu_r')
    # brain_plotting_left(hurst_everything, 'brain loadings everything', 0, 0.15, 'Reds')
    # brain_plotting_left(fc_movie, 'brain loadings - fc', -0.25, 0.25, 'RdBu_r')
    # brain_plotting_left(fc_rest, 'brain loadings - fc', 0, 0.25, 'Blues')
    # brain_plotting_left(fc_rest_last_60_TR, 'Effect of Propofol - FC', 0, 0.2, 'Blues')
    # brain_plotting_left(fc_movie_abs, 'brain loadings - fc', 0, 0.2, 'Reds')
    # brain_plotting_left(fc_effect_of_movie, 'Effect of Narrative Listening - FC (left)', 0, 0.2, 'Blues')
    # brain_plotting_left(fc_double_three_way, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting_left(fc_double_two_way, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting_left(fc_double_merged, 'brain loadings - fc', 0, 0.2, 'Blues')
    # brain_plotting_left(hurst_last_60_TR, 'Effect of Propofol - Hurst', 0, 0.16, 'Blues')
    # brain_plotting_left(hurst_effect_of_movie, 'Effect of Narrative Listening - Hurst (left)', 0, 0.15, 'Blues')
    # brain_plotting_left(hurst_double_three_way, 'brain loadings - double three way', 0, 0.2, 'Blues')
    # brain_plotting_left(hurst_double_two_way, 'brain loadings - double two way', 0, 0.2, 'Blues')
    # brain_plotting_left(hurst_double_merged, 'brain loadings - double merged', 0, 0.2, 'Blues')
    # brain_plotting_left(hurst_combined, 'Combined Effects - Hurst', 0, 0.15, 'Blues')
    # brain_plotting_left(fc_combined, 'Combined Effects - FC', 0, 0.20, 'Blues')
    # brain_plotting_left(fc_everything, 'Everything - FC', 0, 0.20, 'Blues')
    #
    # brain_plotting_left(hurst_effect_of_movie_full, 'Effect of Narrative Listening - Hurst', 0, 0.2, 'Reds')
    # brain_plotting_left(hurst_rest_full, 'Effect of Propofol - Hurst', 0, 0.2, 'Blues')
    #
    # brain_plotting_left(fc_effect_of_movie_full, 'Effect of Narrative Listening - FC', 0, 0.2, 'Reds')
    # brain_plotting_left(fc_rest_full, 'Effect of Propofol - FC', 0, 0.2, 'Blues')
    #
    # brain_plotting_left(overlap_mask, 'Difference in Hurst Exponents', 0, 1, 'Greens')
    # brain_plotting_left(movie_new, 'Difference in Hurst Exponents', 0, .15, 'Reds')
    # brain_plotting_left(rest_new, 'Difference in Hurst Exponents', 0, .2, 'Blues')
    #
    # brain_plotting_left(fc_overlap_mask, 'Difference in FC', 0, 1, 'Greens')
    # brain_plotting_left(fc_movie_new, 'Difference in FC', 0, .2, 'Reds')
    # brain_plotting_left(fc_rest_new, 'Difference in FC', 0, .2, 'Blues')

    # brain_plotting_left(general_fc_everything, 'Everything - general FC (left)', 0, 0.11, 'Blues')
    # brain_plotting_left(general_fc_movie, 'Movie - general FC (left)', 0, 0.14, 'Reds')


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



