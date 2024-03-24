import numpy as np
import pandas as pd
import scipy.io as sio


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
def r2_pls(result_path, latent_variable=1, PLS2 = False, data=None):
    """
    Calculate the r2 for PLS results.
    :param result_path: path to the PLS results
    :param data: the data used to calculate the r2
    :return: r2
    """
    column_index = latent_variable - 1

    # read in the results
    vsc = sio.loadmat(result_path)['result']['vsc'][0][0][:, column_index]
    usc = sio.loadmat(result_path)['result']['usc'][0][0][:, column_index]
    s = sio.loadmat(result_path)['result']['s']

    if PLS2:
        var_explained_x = np.var(usc, ddof=1)
        total_var_x = np.var(data, ddof=1).sum()
        r2_x = var_explained_x / total_var_x
        print(f"R^2 for the first latent variable: {r2_x}")
        return r2_x

    else:
        row_counts = sio.loadmat(result_path)['result']['num_subj_lst'][0][0][0]
        # print(row_counts)

        # Create a list of conditions dynamically based on the length of row_counts
        conditions = ['condition' + str(i + 1) for i in range(len(row_counts))]

        # Create a flat list of condition labels
        condition_labels = [condition for condition, count in zip(conditions, row_counts) for _ in range(count)]

        # create a dummy variable for the condition
        y_dummy = pd.get_dummies(condition_labels)

        # Calculate explained variance by the first latent variable
        var_explained = np.var(vsc, ddof=1)

        # Calculate total variance in Y
        total_var = np.var(y_dummy, ddof=1).sum()

        # Calculate R^2 for the first latent variable
        r2 = var_explained / total_var

        print(f"R^2 for the first latent variable: {r2}")
        return r2


# calculate the r2
r2_h_everthing = r2_pls(path_everything)

# fc everything
r2_fc_everything = r2_pls(path_fc_everything)

# edges everything
r2_edges_everything = r2_pls(path_edges_everything)

# fc movie
r2_effect_of_movie = r2_pls(path_effect_of_movie)

# fc effect of movie
r2_fc_effect_of_movie = r2_pls(path_fc_effect_of_movie)

# fc effect of movie
r2_edges_effect_of_movie = r2_pls(path_edges_effect_of_movie)

# hurst propofol
r2_hurst_propofol = r2_pls(path_effect_of_propofol)

# fc propofol
r2_fc_propofol = r2_pls(path_fc_effect_of_propofol)

# edges propofol
r2_edges_propofol = r2_pls(path_edges_effect_of_propofol)

# hurst interactions
r2_hurst_interactions = r2_pls(path_interactions)

# fc interactions
r2_fc_interactions = r2_pls(path_fc_interactions)

# edges interactions
r2_edges_interactions = r2_pls(path_edges_interactions)









