import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example 3x3 orthogonal matrix (90-degree rotation about the z-axis)
A = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])  # 3D Rotation matrix about the z-axis

# 1. Visualizing A inverse equals A transpose
def visualize_inverse_equals_transpose(matrix):
    inverse = np.linalg.inv(matrix)
    transpose = matrix.T

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("1. A Inverse Equals A Transpose")

    ax[0].imshow(inverse, cmap="Blues")
    ax[0].set_title("Inverse of A")
    for (i, j), val in np.ndenumerate(inverse):
        ax[0].text(j, i, f'{val:.2f}', ha='center', va='center', color="black")

    ax[1].imshow(transpose, cmap="Greens")
    ax[1].set_title("Transpose of A")
    for (i, j), val in np.ndenumerate(transpose):
        ax[1].text(j, i, f'{val:.2f}', ha='center', va='center', color="black")

    plt.show()

# 2. Visualizing norm preservation (in 3D)
def visualize_norm_preservation(matrix, vector):
    transformed_vector = np.dot(matrix, vector)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='blue', label='Original Vector')
    ax.quiver(0, 0, 0, transformed_vector[0], transformed_vector[1], transformed_vector[2], color='red', label='Transformed Vector')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("2. Norm Preservation (3D)")
    plt.show()

# 3. Visualizing determinant (unit cube transformation)
def visualize_determinant(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the unit cube (vertices)
    cube = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])

    # Apply the transformation (rotation) to the cube vertices
    transformed_cube = np.dot(cube, matrix.T)

    # Plot the original and transformed cubes
    ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], color='blue', label='Original Cube')
    ax.scatter(transformed_cube[:, 0], transformed_cube[:, 1], transformed_cube[:, 2], color='red', label='Transformed Cube')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3. Determinant (Cube Transformation)")
    plt.show()

# 4. Visualizing eigenvalues (for 3x3 matrix)
def visualize_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("4. Eigenvalues Visualization")

    for i in range(len(eigenvalues)):
        ax.quiver(0, 0, 0, eigenvectors[0, i], eigenvectors[1, i], eigenvectors[2, i], color='blue', label=f'Eigenvector {i+1} (λ={eigenvalues[i]:.2f})')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# 5. Visualizing orthonormality (in 3D)
def visualize_orthonormality(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("5. Orthonormal Rows and Columns")

    # Plot rows of the matrix
    for i in range(3):
        ax.quiver(0, 0, 0, matrix[i, 0], matrix[i, 1], matrix[i, 2], color='blue', label=f'Row {i+1}')
    
    # Plot columns of the matrix (transposed)
    for i in range(3):
        ax.quiver(0, 0, 0, matrix[0, i], matrix[1, i], matrix[2, i], color='red', linestyle='dashed', label=f'Column {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Visualizing properties
vector = np.array([1, 1, 0])  # Example 3D vector for visualization

visualize_inverse_equals_transpose(A)
visualize_norm_preservation(A, vector)
visualize_determinant(A)
visualize_eigenvalues(A)
visualize_orthonormality(A)
