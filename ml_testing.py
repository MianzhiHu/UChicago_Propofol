from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from partial_least_squares import df_everything, df_everything_missing
from brain_plotting import brain_plotting, brain_plotting_left

# THIS IS BULLSHIT
# FUCK MACHINE LEARNING

# pure testing
X = df_everything.iloc[:, :-1]
y = df_everything.iloc[:, -1]

# df_movie['label'] = onehotencoder.fit_transform(df_movie[['label']])
# X = df_movie.iloc[:, :-1]
# y = df_movie.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr', penalty='l1', solver='liblinear')
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('Coefficients: \n', regressor.coef_)
print('Accuracy: \n', regressor.score(X_test, y_test))
print('f1 score: \n', f1_score(y_test, y_pred, average='weighted'))

coefficients = regressor.coef_[2]

# coefficients = [abs(number) for number in coefficients]
brain_plotting(coefficients, 'logistic_regression_coefficients', vmin=-0.6, vmax=0.6, cmap='RdBu_r', nodes_with_missing_values=df_everything_missing)
brain_plotting_left(coefficients, 'logistic_regression_coefficients', vmin=-0.6, vmax=0.6, cmap='RdBu_r', nodes_with_missing_values=df_everything_missing)




# build a svm model
regressor_svm = OneVsRestClassifier(svm.SVC(kernel='linear', C=0.1, probability=True, random_state=42))
regressor_svm.fit(X_train, y_train)

y_pred_svm = regressor_svm.predict(X_test)

print('R-squared: \n', regressor_svm.score(X_test, y_test))
print('f1 score: \n', f1_score(y_test, y_pred_svm, average='weighted'))
print('Precision: \n', precision_score(y_test, y_pred_svm, average='weighted'))
print('Recall: \n', recall_score(y_test, y_pred_svm, average='weighted'))
print('Accuracy: \n', accuracy_score(y_test, y_pred_svm))

binary_classifiers = regressor_svm.estimators_
coefficients_svm = [clf.coef_ for clf in binary_classifiers]
coefficients_svm = np.array(coefficients_svm).squeeze()

coefficients_svm_movie_deep = coefficients_svm[2]

# # calculate pvalues for coefficients
vif = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
std_err = np.sqrt(vif)
t_values = coefficients_svm_movie_deep/std_err
p_values = 2 * (1 - norm.cdf(abs(t_values)))

# coefficients_svm = [abs(number) for number in coefficients_svm]

brain_plotting(coefficients_svm_movie_deep, 'svm_coefficients', vmin=-0.06, vmax=0.06, cmap='RdBu_r', nodes_with_missing_values=df_everything_missing)
brain_plotting_left(coefficients_svm_movie_deep, 'svm_coefficients', vmin=-0.06, vmax=0.06, cmap='RdBu_r', nodes_with_missing_values=df_everything_missing)
