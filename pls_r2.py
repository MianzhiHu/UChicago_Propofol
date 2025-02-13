import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import pinv
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from pls_data_paths import (hurst_everything_list, fc_everything_list, edges_everything_list,
                            hurst_effect_of_movie_list, fc_effect_of_movie_list, edges_effect_of_movie_list,
                            hurst_effect_of_propofol_list, fc_effect_of_propofol_list, edges_effect_of_propofol_list,
                            hurst_interaction_list, fc_interaction_list, edges_interaction_list)


# read in the results
path = './data_generated/PLS_outputTaskPLShurst_propofol_movie.mat'
path_everything = './data_generated/PLS_outputTaskPLSeverything.mat'
path_fc_everything = './data_generated/PLS_outputTaskPLSfc_everything.mat'
# path_fc_everything = './data_generated/PLS_outputTaskPLSgeneral_fc_everything.mat'
path_edges_everything = './data_generated/PLS_outputTaskPLSedges_everything.mat'

path_effect_of_movie = './data_generated/PLS_outputTaskPLSeffect_of_movie.mat'
path_fc_effect_of_movie = './data_generated/PLS_outputTaskPLSfc_effect_of_movie.mat'
path_edges_effect_of_movie = './data_generated/PLS_outputTaskPLSedges_effect_of_movie.mat'

path_effect_of_propofol = './data_generated/PLS_outputTaskPLSlast_60_TR.mat'
path_fc_effect_of_propofol = './data_generated/PLS_outputTaskPLSfc_rest_last_60_TR.mat'
path_edges_effect_of_propofol = './data_generated/PLS_outputTaskPLSedges_rest_last_60_TR.mat'

path_interactions = './data_generated/PLS_outputTaskPLShurst_propofol_movie.mat'
path_fc_interactions = './data_generated/PLS_outputTaskPLSfc_movie.mat'
# path_fc_interactions = './data_generated/PLS_outputTaskPLSgeneral_fc_movie.mat'
path_edges_interactions = './data_generated/PLS_outputTaskPLSedges_movie.mat'


# function to calculate r2 for pls results
def r2_pls(result_path):

    # read in the results
    usc = sio.loadmat(result_path)['result']['usc'][0][0][:, 0]
    vsc = sio.loadmat(result_path)['result']['vsc'][0][0][:, 0]

    # calculate the pearson r
    r = pearsonr(usc, vsc)[0]
    r2 = r**2
    print('R2:', r2)

    return r2

# calculate the r2
print('R2 for Hurst in 2*4 PLS model')
r2_h_everthing = r2_pls(path_everything)

# fc everything
print('R2 for FC in 2*4 PLS model')
r2_fc_everything = r2_pls(path_fc_everything)

# edges everything
print('R2 for Edges in 2*4 PLS model')
r2_edges_everything = r2_pls(path_edges_everything)

# hurst movie
print('R2 for Hurst in effect of movie model')
r2_h_effect_of_movie = r2_pls(path_effect_of_movie)

# fc effect of movie
print('R2 for FC in effect of movie model')
r2_fc_effect_of_movie = r2_pls(path_fc_effect_of_movie)

# fc effect of movie
print('R2 for Edges in effect of movie model')
r2_edges_effect_of_movie = r2_pls(path_edges_effect_of_movie)

# hurst propofol
print('R2 for Hurst in effect of propofol model')
r2_hurst_propofol = r2_pls(path_effect_of_propofol)

# fc propofol
print('R2 for FC in effect of propofol model')
r2_fc_propofol = r2_pls(path_fc_effect_of_propofol)

# edges propofol
print('R2 for Edges in effect of propofol model')
r2_edges_propofol = r2_pls(path_edges_effect_of_propofol)

# hurst interactions
print('R2 for Hurst in interactions model')
r2_hurst_interactions = r2_pls(path_interactions)

# fc interactions
print('R2 for FC in interactions model')
r2_fc_interactions = r2_pls(path_fc_interactions)

# edges interactions
print('R2 for Edges in interactions model')
r2_edges_interactions = r2_pls(path_edges_interactions)

# add all