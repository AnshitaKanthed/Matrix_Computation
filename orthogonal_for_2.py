import numpy as np
import matplotlib.pyplot as plt

# Example orthogonal matrix (90-degree rotation)
A = np.array([
    [0, -1],
    [1,  0]
    # [np.sqrt(2)/2, -np.sqrt(2)/2],
    # [np.sqrt(2)/2,  np.sqrt(2)/2]
    
])  # Rotation matrix

# 1. Visualizing A inverse equals A transpose
def visualize_inverse_equals_transpose(matrix):
    inverse = np.linalg.inv(matrix)
    transpose = matrix.T

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
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

# 2. Visualizing norm preservation
def visualize_norm_preservation(matrix, vector):
    transformed_vector = np.dot(matrix, vector)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid()

    ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector')
    ax.quiver(0, 0, transformed_vector[0], transformed_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label='Transformed Vector')
    
    ax.legend()
    ax.set_title("2. Norm Preservation")
    plt.show()

# 3. Visualizing determinant
def visualize_determinant(matrix):
    fig, ax = plt.subplots()
    ax.set_title("3. Determinant (Area)")

    # Plot original unit square
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    transformed_square = np.dot(matrix, unit_square.T).T

    ax.plot(unit_square[:, 0], unit_square[:, 1], 'b--', label='Original Square')
    ax.plot(transformed_square[:, 0], transformed_square[:, 1], 'r-', label='Transformed Square')

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend()
    plt.show()

# 4. Visualizing eigenvalues
def visualize_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    fig, ax = plt.subplots()
    ax.set_title("4. Eigenvalues Visualization")
    
    ax.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], angles='xy', scale_units='xy', scale=1, color='blue', label=f'Eigenvector 1 (λ={eigenvalues[0]:.2f})')
    ax.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Eigenvector 2 (λ={eigenvalues[1]:.2f})')
    
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid()
    ax.legend()
    plt.show()

# 5. Visualizing orthonormality
def visualize_orthonormality(matrix):
    fig, ax = plt.subplots()
    ax.set_title("5. Orthonormal Rows and Columns")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    rows = matrix
    cols = matrix.T
    ax.quiver(0, 0, rows[0, 0], rows[0, 1], angles='xy', scale_units='xy', scale=1, color='blue', label='Row 1')
    ax.quiver(0, 0, rows[1, 0], rows[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Row 2')
    ax.quiver(0, 0, cols[0, 0], cols[1, 0], angles='xy', scale_units='xy', scale=1, color='red', linestyle='dashed', label='Column 1')
    ax.quiver(0, 0, cols[0, 1], cols[1, 1], angles='xy', scale_units='xy', scale=1, color='orange', linestyle='dashed', label='Column 2')

    ax.legend()
    plt.show()

# Visualizing properties
vector = np.array([1, 0])

visualize_inverse_equals_transpose(A)
visualize_norm_preservation(A, vector)
visualize_determinant(A)
visualize_eigenvalues(A)
visualize_orthonormality(A)
