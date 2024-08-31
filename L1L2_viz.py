import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define regularization strengths
alphas = np.logspace(-5, 5, 50)

# Initialize lists to store coefficients
lasso_coefs = []
ridge_coefs = []

# Fit models with different regularization strengths
for alpha in alphas:
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    
    lasso.fit(X_scaled, y)
    ridge.fit(X_scaled, y)
    
    lasso_coefs.append(lasso.coef_)
    ridge_coefs.append(ridge.coef_)

# Convert lists to numpy arrays
lasso_coefs = np.array(lasso_coefs)
ridge_coefs = np.array(ridge_coefs)

# Plot the results
plt.figure(figsize=(12, 5))

# L1 (Lasso) plot
plt.subplot(121)
plt.semilogx(alphas, lasso_coefs)
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients')

# L2 (Ridge) plot
plt.subplot(122)
plt.semilogx(alphas, ridge_coefs)
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Coefficients')

plt.tight_layout()
plt.show()