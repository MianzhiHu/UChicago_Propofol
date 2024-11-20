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
path_edges_everything = './data_generated/PLS_outputTaskPLSedges_everything.mat'

path_effect_of_movie = './data_generated/PLS_outputTaskPLSeffect_of_movie.mat'
path_fc_effect_of_movie = './data_generated/PLS_outputTaskPLSfc_effect_of_movie.mat'
path_edges_effect_of_movie = './data_generated/PLS_outputTaskPLSedges_effect_of_movie.mat'

path_effect_of_propofol = './data_generated/PLS_outputTaskPLSlast_60_TR.mat'
path_fc_effect_of_propofol = './data_generated/PLS_outputTaskPLSfc_rest_last_60_TR.mat'
path_edges_effect_of_propofol = './data_generated/PLS_outputTaskPLSedges_rest_last_60_TR.mat'

path_interactions = './data_generated/PLS_outputTaskPLShurst_propofol_movie.mat'
path_fc_interactions = './data_generated/PLS_outputTaskPLSfc_movie.mat'
path_edges_interactions = './data_generated/PLS_outputTaskPLSedges_movie.mat'


# function to calculate r2 for pls results
def r2_pls(result_path, data_paths):

    # read in the results
    u = sio.loadmat(result_path)['result']['u'][0][0]
    v = sio.loadmat(result_path)['result']['v'][0][0]
    usc = sio.loadmat(result_path)['result']['usc'][0][0]
    vsc = sio.loadmat(result_path)['result']['vsc'][0][0]

    # create the dummy variable for the condition
    row_counts = sio.loadmat(result_path)['result']['num_subj_lst'][0][0][0]
    # Create a list of conditions dynamically based on the length of row_counts
    conditions = ['condition' + str(i + 1) for i in range(len(row_counts))]

    # Create a flat list of condition labels
    condition_labels = [condition for condition, count in zip(conditions, row_counts) for _ in range(count)]

    # create a dummy variable for the condition
    y_dummy = pd.get_dummies(condition_labels)

    # Load the data from the CSV files and stack them
    data_list = [pd.read_csv(path, header=None) for path in data_paths]
    data = pd.concat(data_list, axis=0, ignore_index=True)

    # Get data mean-centered
    data_mean = np.mean(data, axis=0)
    data_centered = (data - data_mean) / np.std(data, axis=0)

    # run PLS
    pls = PLSRegression(n_components=len(conditions)).fit(data_centered, y_dummy)
    x_weight = pls.x_weights_
    y_weight = pls.y_weights_
    x_scores = pls.x_scores_
    y_scores = pls.y_scores_
    y_pred = pls.predict(data_centered)

    # validity check, we want to make sure that the PLS here is equivalent to the one in MATLAB
    issues = []

    corr_x_weight = pearsonr(x_weight[:, 0], u[:, 0])[0]
    corr_y_weight = pearsonr(y_weight[:, 0], v[:, 0])[0]
    corr_x_scores = pearsonr(x_scores[:, 0], usc[:, 0])[0]
    corr_y_scores = pearsonr(y_scores[:, 0], vsc[:, 0])[0]
    spr_x_weight = spearmanr(x_weight[:, 0], u[:, 0])[0]
    spr_y_weight = spearmanr(y_weight[:, 0], v[:, 0])[0]
    spr_x_scores = spearmanr(x_scores[:, 0], usc[:, 0])[0]
    spr_y_scores = spearmanr(y_scores[:, 0], vsc[:, 0])[0]

    # Define a dictionary to map variable names to their corresponding correlation values
    correlations = {
        'corr_x_weight': corr_x_weight,
        'corr_y_weight': corr_y_weight,
        'corr_x_scores': corr_x_scores,
        'corr_y_scores': corr_y_scores,
        'spr_x_weight': spr_x_weight,
        'spr_y_weight': spr_y_weight,
        'spr_x_scores': spr_x_scores,
        'spr_y_scores': spr_y_scores
    }

    # Initialize a list to store the failing conditions
    issues = [f"{name}: {value}" for name, value in correlations.items() if abs(value) < 0.90]

    # If any issues were found, print them
    if issues:
        issue_report = ', '.join(issues)
        print(f"The PLS results are not equivalent to the ones in MATLAB! "
              f"The following correlations are below the threshold of 0.90: {issue_report}")

    # Calculate the r2
    r2 = r2_score(y_dummy, y_pred)
    print(f"R^2: {r2}")

    return [y_dummy, y_pred]

# calculate the r2
print('R2 for Hurst in 2*4 PLS model')
r2_h_everthing = r2_pls(path_everything, hurst_everything_list)

# fc everything
print('R2 for FC in 2*4 PLS model')
r2_fc_everything = r2_pls(path_fc_everything, fc_everything_list)

# edges everything
print('R2 for Edges in 2*4 PLS model')
r2_edges_everything = r2_pls(path_edges_everything, edges_everything_list)

# hurst movie
print('R2 for Hurst in effect of movie model')
r2_h_effect_of_movie = r2_pls(path_effect_of_movie, hurst_effect_of_movie_list)

# fc effect of movie
print('R2 for FC in effect of movie model')
r2_fc_effect_of_movie = r2_pls(path_fc_effect_of_movie, fc_effect_of_movie_list)

# fc effect of movie
print('R2 for Edges in effect of movie model')
r2_edges_effect_of_movie = r2_pls(path_edges_effect_of_movie, edges_effect_of_movie_list)

# hurst propofol
print('R2 for Hurst in effect of propofol model')
r2_hurst_propofol = r2_pls(path_effect_of_propofol, hurst_effect_of_propofol_list)

# fc propofol
print('R2 for FC in effect of propofol model')
r2_fc_propofol = r2_pls(path_fc_effect_of_propofol, fc_effect_of_propofol_list)

# edges propofol
print('R2 for Edges in effect of propofol model')
r2_edges_propofol = r2_pls(path_edges_effect_of_propofol, edges_effect_of_propofol_list)

# hurst interactions
print('R2 for Hurst in interactions model')
r2_hurst_interactions = r2_pls(path_interactions, hurst_interaction_list)

# fc interactions
print('R2 for FC in interactions model')
r2_fc_interactions = r2_pls(path_fc_interactions, fc_interaction_list)

# edges interactions
print('R2 for Edges in interactions model')
r2_edges_interactions = r2_pls(path_edges_interactions, edges_interaction_list)

# add all