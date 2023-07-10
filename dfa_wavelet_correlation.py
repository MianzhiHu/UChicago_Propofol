import pandas as pd
from scipy.stats import pearsonr, levene
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('./data_generated/full.csv')

# test the correlation between the two hurst values
corr, p = pearsonr(df['hurst_average.x'], df['hurst_average.y'])
print('Correlation between the two hurst values: %.3f' % corr)
print('p-value: %.3f' % p)

# print the standard deviation of the two hurst values
print('Standard Deviation of DFA: %.3f' % df['hurst_average.x'].std())
print('Standard Deviation of WLBMFA: %.3f' % df['hurst_average.y'].std())

# do a levene test to see if the variances are equal
stat, p = levene(df['hurst_average.x'], df['hurst_average.y'])
print('Levene Test Statistic: %.3f' % stat)
print('p-value: %.3f' % p)


# find the f statistic and degrees of freedom
n = len(df['hurst_average.x'])
f = (corr**2) / ((1 - corr**2) / (n - 2))
print('F-Statistic: %.3f' % f)
print('Degrees of Freedom: %d' % (n - 2))

# test linear regression between the two hurst values
lm = LinearRegression()
lm.fit(df[['hurst_average.x']], df[['hurst_average.y']])

# print the coefficients
print('Intercept: %.3f' % lm.intercept_)
print('Coefficient: %.3f' % lm.coef_)
print('R-Squared: %.3f' % lm.score(df[['hurst_average.x']], df[['hurst_average.y']]))
print('p-value: %.3f' % p)

# print the linear regression equation
print('y = %.3f + %.3f * x' % (lm.intercept_, lm.coef_))

# plot the linear regression
plt.scatter(df['hurst_average.x'], df['hurst_average.y'], s=5)
plt.plot(df['hurst_average.x'], lm.predict(df[['hurst_average.x']]), color='red')
# add a label for the linear regression equation
plt.text(0.65, 0.9, 'y = %.3f + %.3f * x' % (lm.intercept_, lm.coef_))
plt.title('Linear Regression - DFA vs WLBMFA')
plt.xlim(0.4, 1)
plt.ylim(0.4, 1)
plt.xlabel('Hurst Value (DFA)')
plt.ylabel('Hurst Value (WLBMFA)')
# add an x=y line
plt.plot([0.4, 1], [0.4, 1], color='black', linestyle='--')
# add a legend
plt.legend(['Data Points', 'Linear Regression', 'x=y'])
plt.show()
