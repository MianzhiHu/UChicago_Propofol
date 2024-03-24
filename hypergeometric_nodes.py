import numpy as np
from scipy.stats import hypergeom

nodes_hurst_effect_of_movie = np.load('./data_generated/nodes_with_hurst_values_effect_of_movie.npy')
nodes_fc_effect_of_movie = np.load('./data_generated/nodes_with_fc_values_effect_of_movie.npy')
nodes_hurst_effect_of_propofol = np.load('./data_generated/nodes_with_hurst_values_last_60_TR.npy')
nodes_fc_effect_of_propofol = np.load('./data_generated/nodes_with_fc_values_rest_last_60_TR.npy')
nodes_hurst_combined = np.load('./data_generated/nodes_with_hurst_combined.npy')
nodes_fc_combined = np.load('./data_generated/nodes_with_fc_values_combined.npy')
nodes_hurst_nl = np.load('./data_generated/nodes_with_hurst_values.npy')
nodes_fc_nl = np.load('./data_generated/nodes_with_fc_values.npy')
nodes_hurst_everything = np.load('./data_generated/nodes_with_hurst_values_everything.npy')
nodes_fc_everything = np.load('./data_generated/nodes_with_fc_values_everything.npy')


def overlap_calculator(list1, list2):
    overlap = np.intersect1d(list1, list2)
    print(f'overlap: {len(overlap)}')
    percentage_overlap = len(overlap) / len(np.union1d(list1, list2)) * 100
    print(f'percentage overlap: {percentage_overlap}')
    return percentage_overlap
    # return overlap




overlap_hurst_movie = overlap_calculator(nodes_hurst_combined, nodes_hurst_effect_of_movie)
overlap_fc_movie = overlap_calculator(nodes_fc_effect_of_movie, nodes_fc_combined)
overlap_hurst_propofol = overlap_calculator(nodes_hurst_effect_of_propofol, nodes_hurst_combined)
overlap_fc_propofol = overlap_calculator(nodes_fc_effect_of_propofol, nodes_fc_combined)
overlap_everything = overlap_calculator(nodes_fc_everything, nodes_hurst_everything)
print(overlap_calculator(nodes_fc_everything, nodes_fc_nl))
print(overlap_calculator(nodes_fc_effect_of_propofol, nodes_fc_nl))


# calculate the union set of nodes
union_nodes_hurst = np.union1d(nodes_hurst_effect_of_movie, nodes_hurst_effect_of_propofol)
union_nodes_hurst_3 = np.union1d(union_nodes_hurst, nodes_hurst_nl)
union_nodes_fc = np.union1d(nodes_fc_effect_of_movie, nodes_fc_effect_of_propofol)
union_nodes_fc_3 = np.union1d(union_nodes_fc, nodes_fc_nl)
union = np.union1d(nodes_hurst_nl, nodes_hurst_effect_of_movie)
union_1 = np.union1d(nodes_hurst_nl, nodes_hurst_effect_of_propofol)
union_fc = np.union1d(nodes_fc_nl, nodes_fc_effect_of_movie)


# calculate the intersection set of nodes
overlap_hurst = overlap_calculator(union_nodes_hurst, nodes_hurst_combined)
overlap_fc = overlap_calculator(union_nodes_fc, nodes_fc_combined)
overlap_union = overlap_calculator(union, nodes_hurst_combined)
overlap_union_fc = overlap_calculator(union_fc, nodes_fc_combined)
overlap_x = overlap_calculator(nodes_fc_combined, union_fc)

list_1 = [nodes_hurst_effect_of_movie, nodes_hurst_effect_of_propofol, union_nodes_hurst, nodes_hurst_nl]
list_2 = [nodes_hurst_combined, nodes_hurst_nl]

list_3 = [nodes_fc_effect_of_movie, nodes_fc_effect_of_propofol, union_nodes_fc, nodes_fc_nl]
list_4 = [nodes_fc_combined, nodes_fc_nl]

for i in list_1:
    for j in list_2:
        overlap_calculator(j, i)

for i in list_3:
    for j in list_4:
        overlap_calculator(j, i)

for i in list_1[0:1]:
    overlap1 = np.intersect1d(i, list_2[0])
    overlap2 = np.intersect1d(i, list_2[1])


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
pval_union = hypergeometric_test(union, nodes_hurst_combined)
pval_union_fc = hypergeometric_test(union_fc, nodes_fc_combined)

# hypergeometric test for individual effects
pval_hurst_movie = hypergeometric_test(nodes_hurst_effect_of_movie, nodes_hurst_combined)
pval_fc_movie = hypergeometric_test(nodes_fc_effect_of_movie, nodes_fc_combined)
pval_fc_between = hypergeometric_test(nodes_fc_effect_of_movie, nodes_fc_effect_of_propofol)
pval_hurst_propofol = hypergeometric_test(nodes_hurst_effect_of_propofol, nodes_hurst_combined)
pval_hurst_between = hypergeometric_test(nodes_fc_effect_of_movie, nodes_fc_effect_of_propofol)
pval_fc_propofol = hypergeometric_test(nodes_fc_effect_of_propofol, nodes_fc_combined)
pval_x = hypergeometric_test(nodes_hurst_combined, union)
pval_everything = hypergeometric_test(nodes_hurst_everything, nodes_fc_everything)
print(hypergeometric_test(nodes_fc_effect_of_movie, nodes_fc_nl))

for i in list_1:
    for j in list_2:
        hypergeometric_test(j, i)

for i in list_3:
    for j in list_4:
        hypergeometric_test(j, i)