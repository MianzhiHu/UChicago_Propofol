import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM

# read in to data
df = pd.read_csv('participant_avg.csv')

# now i want to conduct a repeated measures ANOVA using scipy
# group by condition
df['rd_condition'] = df['rd_condition'].astype('str')
grouped = df.groupby('rd_condition')

# create a list of dataframes
dataframes = [group for _, group in grouped]

# conduct the repeated measures ANOVA
anova = AnovaRM(df, 'avg_rating', 'participant', within=['rd_condition'])
results = anova.fit()






