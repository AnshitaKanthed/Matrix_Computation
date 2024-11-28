# QR Decomposition
# Task: Decompose a given matrix into an orthogonal matrix 
# 𝑄
# Q and an upper triangular matrix 
# 𝑅
# R.
# Visualization: Show heatmaps of the original matrix, 
# 𝑄
# Q, and 
# 𝑅
# R.
# Code:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Generate a random matrix
A = np.random.rand(5, 3)

# Perform QR decomposition
Q, R = np.linalg.qr(A)

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.heatmap(A, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Original Matrix (A)")

plt.subplot(1, 3, 2)
sns.heatmap(Q, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Orthogonal Matrix (Q)")

plt.subplot(1, 3, 3)
sns.heatmap(R, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Upper Triangular Matrix (R)")

plt.tight_layout()
plt.show()