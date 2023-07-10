import pandas as pd
import numpy as np
import pprint

# load the wavelet output
wavelet_df = pd.read_csv('wavelet_output.csv')

# Calculate the mean of each column
wavelet_df_mean = wavelet_df.mean(axis=0)

# stack the mean values into a dataframe with their column names
wavelet_df_mean = pd.DataFrame(wavelet_df_mean)

# name the column
wavelet_df_mean.columns = ['hurst_average']

# save the dataframe to a csv file
wavelet_df_mean.to_csv('wavelet_df_mean.csv', index=True, header=True)