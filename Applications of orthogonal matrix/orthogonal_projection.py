# Orthogonal Projections
# Task: Project a set of vectors onto an orthogonal basis.
# Visualization: Show the original vectors and their projections

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# Define basis and vectors
basis = np.array([[1, 0], [0, 1]])  # Standard basis
vectors = np.array([[2, 1], [1, -1]])

# Project vectors onto the basis
projections = (vectors @ basis.T) @ basis

# Visualize
plt.figure(figsize=(6, 6))
for i in range(vectors.shape[0]):
    plt.quiver(0, 0, vectors[i, 0], vectors[i, 1], angles='xy', scale_units='xy', scale=1, color='r', label=f"Vector {i+1}" if i == 0 else "")
    plt.quiver(0, 0, projections[i, 0], projections[i, 1], angles='xy', scale_units='xy', scale=1, color='b', label=f"Projection {i+1}" if i == 0 else "")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.title("Orthogonal Projections")
plt.show()