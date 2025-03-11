#Clear Workspace before each run.
import numpy as np
from scipy import linalg

def Psi(z):
    zeta = 0.11  # Establish Tuning Parameter
    Psi = (1 - np.exp(-zeta * z)) / zeta
    return Psi

def Psi_dx(z):
    zeta = 0.11  # Establish tuning parameter
    Psi_dx = np.exp(-zeta * z)
    return Psi_dx

def Theta_0(n, W, X, Py, Q):
    I = np.eye(n)  # Identity matrix
    Pwx = X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W  # Matrix multiplication
    
    # Smallest right singular vectors using SVD
    _, _, V = linalg.svd((I - Pwx) @ W @ Py)
    Theta0 = V[:, -Q:]  # Select the last Q columns
    
    return Theta0

# Establish Matrices and Parameters
tolerance = 1e-14
max_iteration = 10000
RSSold = 2
RSS = 10
iter = 0
K = 2  # Number of classes

# Data
# Synthetic X Matrix
X = np.random.randint(0, 11, size=(100, 40))
X = X / np.linalg.norm(X, axis=0)

# Synthetic Y Matrix - Replace with Target Variable
num_samples = X.shape[0]
num_classes = K  # Classes: 0, 1
Y_labels = np.random.randint(0, 2, size=(num_samples,))

# Convert to dummy variable (one-hot encoding)
Y = np.zeros((num_samples, num_classes))
for i in range(num_samples):
    Y[i, Y_labels[i]] = 1 

n = X.shape[0]
Q = K - 1
betaj = np.ones((X.shape[1], Q))
Thetaj = np.ones((K, Q))
D = (1/n) * (Y.T @ Y)

# ROSDA Loop
while (abs(RSSold - RSS)/RSS > tolerance) and (iter < max_iteration):
    z = np.zeros(n)
    for i in range(n):
        z[i] = np.linalg.norm(Y[i, :] @ Thetaj - X[i, :] @ betaj)**2  # Use default L2 norm
    
    w = np.zeros(n)
    for i in range(n):
        w[i] = Psi_dx(z[i])
    
    W = np.diag(w)
    RSSold = RSS
    
    # betaj calculation
    betaj = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y @ Thetaj
    
    # SVD decomposition
    Py, _, _ = linalg.svd(Y)
    I = np.eye(n)
    Theta0 = Theta_0(n, W, X, Py, Q)
    Thetaj = (1/np.sqrt(n)) * np.linalg.inv(D) @ Y.T @ Py @ Theta0
    
    # Recalculate z using theta(j) and beta(j)
    for i in range(n):
        z[i] = np.linalg.norm(Y[i, :] @ Thetaj - X[i, :] @ betaj)**2  # Use default L2 norm
    
    RSS = (1/n) * sum(Psi(z[i]) for i in range(n))
    iter += 1

# Predict classes by finding which Thetak is closest to X_iB
XB = X @ betaj
predicted_classes = np.zeros((n, K))
for i in range(n):
    distances = np.abs(Thetaj - XB[i, :])
    total_distances = np.sum(distances, axis=1)
    index = np.argmin(total_distances)
    predicted_classes[i, index] = 1

# Note: Functions Psi_dx and Theta_0 need to be defined based on the original paper
# References:
# [1] https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
# [2] https://numpy.org/doc/stable/reference/generated/numpy.diag.html
# [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
# [4] Python operators follow standard mathematical notation
# [5] For finding closest value: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html