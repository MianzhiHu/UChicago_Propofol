import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr
from partial_least_squares import df_movie
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# read in the results
s = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie.mat')['result']['s'][0][0]
v = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie.mat')['result']['v'][0][0]
u = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie.mat')['result']['u'][0][0]
row_counts = sio.loadmat('./data_generated/PLS_outputTaskPLShurst_propofol_movie.mat')['result']['num_subj_lst'][0][0][
    0]

# Create a list of conditions dynamically based on the length of row_counts
conditions = ['condition' + str(i + 1) for i in range(len(row_counts))]

# Create a flat list of condition labels
condition_labels = [condition for condition, count in zip(conditions, row_counts) for _ in range(count)]

# create a dummy variable for the condition
y_dummy = pd.get_dummies(condition_labels)

# standardize the data
scaler = StandardScaler()
df_movie = scaler.fit_transform(df_movie)

n_components = len(row_counts)
PLS = PLSRegression(n_components=n_components)
PLS.fit(df_movie, y_dummy)

y_pred = PLS.predict(df_movie)
r2 = r2_score(y_pred, y_dummy)

# Using the pseudoinverse in case X' is not square or invertible
X_transpose_pinv = np.linalg.pinv(df_movie.T)

# Now, compute Y = (X')^+ * U * S * V' where (X')^+ is the pseudoinverse of X'
s = np.diag(s[:, 0])
predicted_scores = X_transpose_pinv @ u @ s @ v.T

# transform the predicted scores to 0-1
predicted_scores = (predicted_scores - np.min(predicted_scores)) / (np.max(predicted_scores) - np.min(predicted_scores))

# calculate whether the correlation between the predicted scores here
# and in the original PLS is high enough to indicate convergence
correlation = []
for i in range(predicted_scores.shape[0]):
    correlation.append(pearsonr(predicted_scores[i, :], y_pred[i, :])[0])

# raise an error if the correlation is not high enough
if np.mean(correlation) < 0.8:
    raise ValueError('The mean correlation is {}.'.format(np.mean(correlation)) +
                     'This is too low to indicate model convergence. Please check the data.')

r2 = r2_score(y_dummy, predicted_scores)
