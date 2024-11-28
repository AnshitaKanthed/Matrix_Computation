import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a 2D rotation matrix
theta = np.radians(45)  # Rotate 45 degrees
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

# Define some 2D points
points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

# Apply rotation
rotated_points = points @ rotation_matrix.T

# Visualize
plt.figure(figsize=(6, 6))

# Change here: Use points[:, 0] and points[:, 1] for X and Y as well
# to specify the starting points of the arrows
plt.quiver(points[:, 0], points[:, 1], points[:, 0], points[:, 1], color='r', angles='xy', scale_units='xy', scale=1, label="Original")  
plt.quiver(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 0], rotated_points[:, 1], color='b', angles='xy', scale_units='xy', scale=1, label="Rotated") 

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid()
plt.legend()
plt.show()