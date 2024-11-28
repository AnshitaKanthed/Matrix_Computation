#Doolittle details
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function for LU Decomposition
def luDecomposition(mat, n):
    lower = [[0 for x in range(n)] for y in range(n)]
    upper = [[0 for x in range(n)] for y in range(n)]

    step_snapshots = []  # To store intermediate steps for visualization

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += lower[i][j] * upper[j][k]
            upper[i][k] = mat[i][k] - sum

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                lower[i][i] = 1  # Diagonal as 1
            else:
                sum = 0
                for j in range(i):
                    sum += lower[k][j] * upper[j][i]
                lower[k][i] = (mat[k][i] - sum) / upper[i][i]

        # Save current state of L and U
        step_snapshots.append((np.array(lower), np.array(upper)))

    return np.array(lower), np.array(upper), step_snapshots

# Visualization Helper Functions
def visualize_matrix(matrix, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()

def visualize_reconstruction(A, L, U):
    reconstructed_A = np.dot(L, U)
    difference = np.abs(A - reconstructed_A)

    print("Reconstructed A:")
    print(reconstructed_A)

    # Heatmaps
    visualize_matrix(A, "Original Matrix (A)")
    visualize_matrix(reconstructed_A, "Reconstructed Matrix (L x U)")
    visualize_matrix(difference, "Difference Matrix (|A - L x U|)")

def visualize_sparsity(matrix, title):
    sparsity_pattern = np.where(matrix == 0, 0, 1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(sparsity_pattern, annot=False, cmap="Greens", cbar=False)
    plt.title(f"Sparsity Pattern: {title}")
    plt.show()

def visualize_steps(steps):
    for idx, (L, U) in enumerate(steps):
        print(f"Step {idx + 1}:")
        visualize_matrix(L, f"Step {idx + 1} - Lower Triangular Matrix (L)")
        visualize_matrix(U, f"Step {idx + 1} - Upper Triangular Matrix (U)")

# Driver Code
mat = np.array([[8, -6, 2],
                [-6, 7, -4],
                [2, -4, 3]])
n = len(mat)

# Perform LU Decomposition
L, U, steps = luDecomposition(mat, n)

# Visualize Results
print("Lower Triangular Matrix (L):")
print(L)
print("\nUpper Triangular Matrix (U):")
print(U)

visualize_matrix(L, "Lower Triangular Matrix (L)")
visualize_matrix(U, "Upper Triangular Matrix (U)")

# Validate Reconstruction
visualize_reconstruction(mat, L, U)

# Sparsity Patterns
visualize_sparsity(L, "Lower Triangular Matrix (L)")
visualize_sparsity(U, "Upper Triangular Matrix (U)")

# Step-by-Step Evolution
visualize_steps(steps)