import matplotlib
import os
from matplotlib import pyplot as plt
from nilearn import plotting, datasets, surface, image

fsaverage = datasets.fetch_surf_fsaverage()
def make_side_by_side_surf_plots(name,texture,vmin=None,vmax=None, cmap='RdBu', outlines_texture =None, outlines_labes=None):
    plt.clf()
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(figsize=(10, 7), ncols=2, subplot_kw={"projection": "3d"})
    ## medial view
    display = plotting.plot_surf(fsaverage.pial_right, texture, hemi='right', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_right, view='medial', alpha=1, bg_on_data=True,
                       darkness=.4,axes=axes[0])
    ## lateral view
    plotting.plot_surf(fsaverage.pial_right, texture, hemi='right', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_right, view='lateral', alpha=1, bg_on_data=True,
                       darkness=.4,axes=axes[1])
    # adjust space of subplots
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.9, wspace=0.001, hspace=0.02)
    ## customize color bar
    #cmap = cmap # color map
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) # the max and min of values in colorbar
    cb_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02]) # add axes for colorbar
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


# define a function for the left hemisphere
def make_side_by_side_surf_plots_left(name,texture,vmin=None,vmax=None, cmap='RdBu', outlines_texture =None, outlines_labes=None):
    plt.clf()
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(figsize=(10, 7), ncols=2, subplot_kw={"projection": "3d"})
    ## medial view
    display = plotting.plot_surf(fsaverage.pial_left, texture, hemi='left', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_left, view='medial', alpha=1, bg_on_data=True,
                       darkness=.4,axes=axes[0])
    ## lateral view
    plotting.plot_surf(fsaverage.pial_left, texture, hemi='left', colorbar=False, cmap=cmap, vmin=vmin,
                       vmax=vmax, bg_map=fsaverage.sulc_left, view='lateral', alpha=1, bg_on_data=True,
                       darkness=.4,axes=axes[1])
    # adjust space of subplots
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.9, wspace=0.001, hspace=0.02)
    ## customize color bar
    #cmap = cmap # color map
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) # the max and min of values in colorbar
    cb_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02]) # add axes for colorbar
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='horizontal')
    # cb.set_label(label=name,size=16) # customize colorbar label font
    cb.ax.tick_params(labelsize=14) # customize colorbar tick font
    # make the colorbar ticks to have only 5 ticks
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    if outlines_texture is not None:
        for lab in outlines_labes:
            try:
                plotting.plot_surf_contours(fsaverage.pial_left, outlines_texture
                                            , figure=fig, axes=axes[1], levels=[lab], colors=['k'])
            except:
                pass
    if outlines_texture is not None:
        for lab in outlines_labes:
            try:
                plotting.plot_surf_contours(fsaverage.pial_left, outlines_texture
                                            , figure=fig, axes=axes[0], levels=[lab], colors=['k'])
            except:
                pass
    # output the figure
    plt_name = name + '_left.png'
    plt_dir = './graphs/'
    plt_path = os.path.join(plt_dir, plt_name)
    plt.savefig(plt_path, dpi=600)
    plt.show()