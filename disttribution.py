import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate synthetic data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 3 * x + 7 + np.random.normal(0, 2, x.size)  # Linear data with Gaussian noise

# Define a linear model
def linear_model(x, m, b):
    return m * x + b

# Fit the model to the data
params, _ = curve_fit(linear_model, x, y)
m_est, b_est = params

# Calculate the predicted values and residuals
y_pred = linear_model(x, m_est, b_est)
residuals = y - y_pred

# Plot the observed data and the fitted model
plt.figure(figsize=(10, 6))

# Scatter plot of observed data
plt.subplot(2, 2, 1)
plt.scatter(x, y, label='Observed data')
plt.plot(x, y_pred, color='red', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Observed Data and Fitted Model')
plt.legend()

# Residual plot
plt.subplot(2, 2, 2)
plt.scatter(x, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Histogram of residuals
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=20, edgecolor='black', density=True)
mu, std = np.mean(residuals), np.std(residuals)
xmin, xmax = plt.xlim()
x_hist = np.linspace(xmin, xmax, 100)
p = np.exp(-((x_hist - mu) ** 2) / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
plt.plot(x_hist, p, 'k', linewidth=2)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Histogram of Residuals')

plt.tight_layout()
plt.show()
