import pandas as pd
import numpy as np
from partial_least_squares import mild, deep
import matplotlib.pyplot as plt


rest_awake = pd.read_csv('./data_generated/last_window_awake.csv', header=None)
movie_mild = mild
movie_deep = deep
nodes_with_values = np.load('./data_generated/nodes_with_hurst_double_three_way.npy')

# find the missing nodes
double_all = pd.concat([rest_awake, movie_mild, movie_deep], axis=0)
double_missing = double_all.columns[double_all.isnull().any()].tolist()
# np.save('./data_generated/double_missing.npy', double_missing)

# drop the missing nodes
double_all = double_all.dropna(axis=1, how='any')

# separate the concatenated dataframe back into the last windows of each session
rest_awake_clean = double_all.iloc[0:9, :]
movie_mild_clean = double_all.iloc[9:21, :]
movie_deep_clean = double_all.iloc[21:, :]
movie_sedated_clean = double_all.iloc[9:, :]

# # save the cleaned dataframes
# rest_awake_clean.to_csv('./data_generated/double_rest_awake.csv', header=False, index=False)
# movie_mild_clean.to_csv('./data_generated/double_movie_mild.csv', header=False, index=False)
# movie_deep_clean.to_csv('./data_generated/double_movie_deep.csv', header=False, index=False)
# movie_sedated_clean.to_csv('./data_generated/double_movie_sedated.csv', header=False, index=False)

# draw a boxplot
rest_awake = rest_awake[nodes_with_values]
movie_mild = movie_mild[nodes_with_values]
movie_deep = movie_deep[nodes_with_values]

# take the mean of each node
rest_awake_mean = rest_awake.mean(axis=1)
movie_mild_mean = movie_mild.mean(axis=1)
movie_deep_mean = movie_deep.mean(axis=1)

# draw a boxplot
plt.figure()
plt.boxplot([rest_awake_mean, movie_mild_mean, movie_deep_mean])
plt.xticks([1, 2, 3], ['Rest-Awake', 'Movie-Mild', 'Movie-Deep'])
plt.ylabel('Mean Hurst Exponent')
plt.title('Mean Hurst Exponent of Double Effect')
plt.show()

