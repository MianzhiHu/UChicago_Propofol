import numpy as np
from scipy.stats import hypergeom

nodes_hurst_effect_of_movie = np.load('./data_generated/nodes_with_hurst_values_effect_of_movie.npy')
nodes_fc_effect_of_movie = np.load('./data_generated/nodes_with_fc_values_effect_of_movie.npy')
nodes_hurst_effect_of_propofol = np.load('./data_generated/nodes_with_hurst_values_last_60_TR.npy')
nodes_fc_effect_of_propofol = np.load('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy')
nodes_hurst_combined = np.load('./data_generated/nodes_with_hurst_combined.npy')
nodes_fc_combined = np.load('./data_generated/nodes_with_fc_values_combined.npy')


def overlap_calculator(list1, list2):
    overlap = np.intersect1d(list1, list2)
    percentage_overlap = len(overlap) / len(list1) * 100
    return percentage_overlap


overlap_hurst_movie = overlap_calculator(nodes_hurst_effect_of_movie, nodes_hurst_combined)
overlap_fc_movie = overlap_calculator(nodes_fc_effect_of_movie, nodes_fc_combined)
overlap_hurst_propofol = overlap_calculator(nodes_hurst_effect_of_propofol, nodes_hurst_combined)
overlap_fc_propofol = overlap_calculator(nodes_fc_effect_of_propofol, nodes_fc_combined)
x = overlap_calculator(nodes_fc_effect_of_propofol, nodes_fc_effect_of_movie)

# calculate the union set of nodes
union_nodes_hurst = np.union1d(nodes_hurst_effect_of_movie, nodes_hurst_effect_of_propofol)
union_nodes_fc = np.union1d(nodes_fc_effect_of_movie, nodes_fc_effect_of_propofol)

# calculate the intersection set of nodes
overlap_hurst = overlap_calculator(union_nodes_hurst, nodes_hurst_combined)
overlap_fc = overlap_calculator(union_nodes_fc, nodes_fc_combined)

# now conduct hypergeometric tests
def hypergeometric_test(list1, list2):
    M = 268
    n = len(list1)
    N = len(list2)
    x = len(np.intersect1d(list1, list2))
    pval = hypergeom.sf(x - 1, M, n, N)
    print(f'p-value: {pval}')
    return pval


pval_hurst = hypergeometric_test(union_nodes_hurst, nodes_hurst_combined)
pval_fc = hypergeometric_test(union_nodes_fc, nodes_fc_combined)

# hypergeometric test for individual effects
pval_hurst_movie = hypergeometric_test(nodes_hurst_effect_of_movie, nodes_hurst_combined)
pval_fc_movie = hypergeometric_test(nodes_fc_effect_of_movie, nodes_fc_combined)
pval_hurst_propofol = hypergeometric_test(nodes_hurst_effect_of_propofol, nodes_hurst_combined)
pval_fc_propofol = hypergeometric_test(nodes_fc_effect_of_propofol, nodes_fc_combined)