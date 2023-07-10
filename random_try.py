import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

A = np.array([2, 4, 6, 8, 10])
B = np.array([10, 100, 1000, 10000, 100000])
C = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])

# plot A vs B
plt.plot(A, B)
plt.show()

# make x as the log of A
x = np.log10(A)
y = np.log10(B)
z = np.log10(C)

# plot x vs y
plt.plot(x, y)
# plot the line of best fit
plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
# set x range
plt.xlim(0, 5)
plt.show()

# estimate the slope of the line
slope, intercept = np.polyfit(x, y, 1)
print(slope, intercept)

# plot f(x) = x^2 + y^2
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
F = X**2 + Y**2
plt.contour(X, Y, F, [9])
plt.show()


