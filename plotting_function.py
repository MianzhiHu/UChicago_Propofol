import os
import matplotlib
import scipy.io as sio
import nibabel
import numpy as np
import pandas as pd
from nilearn import plotting, datasets, surface, image
from matplotlib import pyplot as plt, colors
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
import math

fsaverage = datasets.fetch_surf_fsaverage()
atlas = load_shen_268(1)
dr = atlas.get_fdata()
dd = dr.copy().astype('float')
labels = np.unique(dr)

contrast_names = {
    '8_condition': '8_condition',
    'Effects of Narrative-Listening': 'EffectsOfNarrative_Listening',
    'Effects of Propofol': 'EffectsOfPropofol',
    'Effects of Propofol on Narrative-Listening': 'EffectsOfPropofolOnNarrative_Listening'
}

def plot_preparation(lv_vals, boot_ratio, nodes_with_missing_values, keep=False, bs_thresh=3):
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
        df.loc[abs(df['boot_ratio']) < bs_thresh, 'u1'] = np.nan
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


def plot_preparation_for_2ndLV(lv_vals, boot_ratio, nodes_with_missing_values, bs_thresh=3):
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
    df.loc[abs(df['boot_ratio']) < bs_thresh, 'u1'] = np.nan
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


def process_contrast(measurement, contrast, reverse=True, keep=False, bs_thresh=3):
    base = './data_generated'
    c_name = contrast_names[contrast]
    lv_vals_path   = f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{c_name}_lv_vals.mat'
    boot_ratio_path= f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{c_name}.mat'
    missing_path   = f'{base}/Contrasts/{contrast}/{measurement}/missing_columns.csv'

    # load
    lv_vals = sio.loadmat(lv_vals_path)
    boot_ratio = sio.loadmat(boot_ratio_path)
    missing_nodes = pd.read_csv(missing_path).values.flatten().tolist()

    # process
    processed_data = plot_preparation(lv_vals, boot_ratio, missing_nodes, keep=keep, bs_thresh=bs_thresh)
    u1_data = processed_data.iloc[:, 0].tolist()

    # optionally reverse
    if reverse:
        u1_data = [(-x if not pd.isna(x) else x) for x in u1_data]

    # filter out NaNs
    valid = [x for x in u1_data if not pd.isna(x)]

    if len(valid) == 0:
        print(f'No valid data for {measurement} | {contrast}')
        return u1_data, None, None, processed_data

    significant_nodes = [i for i, x in enumerate(u1_data) if not pd.isna(x)]
    min_loading = min(valid)
    max_loading = max(valid)

    # save
    out_dir = f'{base}/Significant_nodes'
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'{out_dir}/{measurement}_{contrast}_significant_nodes.npy'
    np.save(out_fname, significant_nodes)

    # stats
    print(f'[{measurement} | {contrast}]  min={min_loading:.4f}, max={max_loading:.4f}, count={len(valid)}')

    return u1_data, min_loading, max_loading, significant_nodes


def floor_to(x, step):
    """
    Round x down toward −∞ to the nearest multiple of `step`.
    """
    return math.floor(x / step) * step


def ceil_to(x, step):
    """
    Round x up (toward +∞) to the nearest multiple of `step`.
    """
    return math.ceil(x / step) * step

def masking_texture(texture):
    """
    Mask the texture with the mask
    """
    # create a mask of the same shape as the texture
    mask = np.isnan(texture) | (texture == 0.0)
    texture = np.ma.masked_array(texture, mask=mask)
    return texture


def make_side_by_side_surf_plots(name, texture_left, texture_right, vmin=None, vmax=None, cmap=None, outlines_texture=None,
                                 outlines_labes=None, threshold=None):
    plt.clf()
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(figsize=(10, 10), ncols=2, nrows=2, subplot_kw={"projection": "3d"})
    # Left hemisphere
    # medial view
    plotting.plot_surf(fsaverage.pial_left, texture_left, hemi='left', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_left, view='medial', alpha=1, bg_on_data=True,
                       darkness=.4, axes=axes[0, 0], threshold=threshold)
    # lateral view
    plotting.plot_surf(fsaverage.pial_left, texture_left, hemi='left', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_left, view='lateral', alpha=1, bg_on_data=True,
                       darkness=.4, axes=axes[0, 1], threshold=threshold)
    # Right hemisphere
    # medial view
    plotting.plot_surf(fsaverage.pial_right, texture_right, hemi='right', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_right, view='medial', alpha=1, bg_on_data=True,
                       darkness=.4, axes=axes[1, 0], threshold=threshold)
    # lateral view
    plotting.plot_surf(fsaverage.pial_right, texture_right, hemi='right', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_right, view='lateral', alpha=1, bg_on_data=True,
                       darkness=.4, axes=axes[1, 1], threshold=threshold)
    # adjust the space of subplots
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.9, wspace=0.001, hspace=0.02)
    ## customize color bar
    #cmap = cmap # color map
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) # the max and min of values in colorbar
    cb_ax = fig.add_axes([0.2, 0.49, 0.6, 0.02]) # add axes for colorbar
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='horizontal')
    # cb.set_label(label=name,size=16) # customize colorbar label font
    cb.ax.tick_params(labelsize=14) # customize colorbar tick font
    # make the colorbar ticks to have only 5 ticks
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    if outlines_texture is not None:
        for lab in outlines_labes:
            try:
                plotting.plot_surf_contours(fsaverage.pial_right, outlines_texture
                                            , figure=fig, axes=axes[1], levels=[lab], colors=['k'])
            except:
                pass
    if outlines_texture is not None:
        for lab in outlines_labes:
            try:
                plotting.plot_surf_contours(fsaverage.pial_right, outlines_texture
                                            , figure=fig, axes=axes[0], levels=[lab], colors=['k'])
            except:
                pass
    # output the figure
    plt_name = name + '.png'
    plt_dir = './graphs/'
    plt_path = os.path.join(plt_dir, plt_name)
    plt.savefig(plt_path, dpi=600)
    plt.show()


def brain_plotting(df, title, vmin, vmax, cmap, nodes_with_missing_values=None, threshold=None):
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
    local_dd = dr.copy()
    for i in np.array(list(range(268))):
        local_dd[dr == labels[i+1]] = df[i]
    new_image_atl = nibabel.Nifti1Image(local_dd, atlas.affine)
    texture_right = masking_texture(surface.vol_to_surf(new_image_atl, fsaverage.pial_right))
    texture_left = masking_texture(surface.vol_to_surf(new_image_atl, fsaverage.pial_left))
    make_side_by_side_surf_plots(title,texture_left, texture_right, vmin=vmin, vmax=vmax, cmap=cmap, threshold=threshold)


