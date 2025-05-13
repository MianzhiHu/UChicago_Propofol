import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from brain_plotting import all_u1_data
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from nilearn import plotting, datasets
import nibabel

# define some variables
contrast = '8_condition'
measurement = 'Edges'
list_nodes = all_u1_data[(measurement, contrast)]

brain_nodes_indices = list(range(1, 269))
edges = [(brain_nodes_indices[i], brain_nodes_indices[j]) for i in range(len(brain_nodes_indices))
         for j in range(i + 1, len(brain_nodes_indices))]

# visualize the significant edges
atlas = load_shen_268(1)
coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)


def matrix_generator(new_df_edges, reverse=False):
    if reverse:
        new_df_edges = [-1 * i for i in new_df_edges]
    # # create a matrix of zeros
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
adj = matrix_generator(list_nodes, reverse=True)

# decide the threshold
num_node = np.count_nonzero(~np.isnan(list_nodes))
if num_node > 1000:
    # take only 25% of the edges
    num_node = int(num_node * 0.25)
thr = 1 - (num_node / len(list_nodes))
thr = f'{thr:.16%}'
print(f'Threshold set to {thr}')
print(f'Max: {np.nanmax(list_nodes)}, Min: {np.nanmin(list_nodes)}')
positive_edges = sum(1 for x in list_nodes if x > 0)
negative_edges = sum(1 for x in list_nodes if x < 0)
print(f'Positive edges: {positive_edges}, Negative edges: {negative_edges}')

fig, ax = plt.subplots(figsize=(10, 10), nrows=2, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.5})
plotting.plot_connectome(adj, coordinates, colorbar=False, node_size=0, edge_cmap='RdBu_r',
                         display_mode='lr', axes=ax[0], edge_threshold=thr)
plotting.plot_connectome(adj, coordinates, colorbar=False, node_size=0, edge_cmap='RdBu_r',
                         display_mode='yz', axes=ax[1], edge_threshold=thr)
norm = mpl.colors.Normalize(vmin=np.min(adj), vmax=np.max(adj))  # the max and min of values in colorbar
cb_ax = fig.add_axes([0.2, 0.49, 0.6, 0.02])  # add axes for colorbar
cmap = plt.get_cmap('RdBu')
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='horizontal')
# cb.set_label(label=name,size=16) # customize colorbar label font
cb.ax.tick_params(labelsize=14)  # customize colorbar tick font
cb.locator = mpl.ticker.MaxNLocator(nbins=8)
cb.update_ticks()
fig.subplots_adjust(hspace=0.5)
plt.savefig(f'./graphs/edges_{measurement}_{contrast}_plot.png', dpi=800)
plt.show()

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


