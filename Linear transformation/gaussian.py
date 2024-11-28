import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
N = 3  # Size of the matrix

# Function to visualize matrix states side-by-side (before and after)
def visualize_initial_and_final(mat_initial, mat_final):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Initial Matrix
    sns.heatmap(np.array(mat_initial), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black', ax=ax[0])
    ax[0].set_title("Initial Matrix")
    ax[0].set_xlabel("Columns")
    ax[0].set_ylabel("Rows")

    # Final Matrix
    sns.heatmap(np.array(mat_final), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black', ax=ax[1])
    ax[1].set_title("Final Matrix (RREF)")
    ax[1].set_xlabel("Columns")
    ax[1].set_ylabel("Rows")

    plt.tight_layout()
    plt.show()

# Function to visualize the pivot movement
def visualize_pivot_path(pivot_steps, mat):
    # Create an animation or plot showing pivot steps
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(mat), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black', ax=ax)

    # Highlight the pivots
    for step in pivot_steps:
        i, j = step
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))

    ax.set_title("Pivot Movement")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.show()

# Function to visualize condition number
def visualize_condition_number(condition_numbers):
    plt.plot(condition_numbers, marker='o', color='b')
    plt.title("Condition Number Over Time")
    plt.xlabel("Step")
    plt.ylabel("Condition Number")
    plt.grid(True)
    plt.show()

# Function to visualize solution error (residuals)
def visualize_residuals(original_matrix, solution):
    b = original_matrix[:, -1]

    # Calculate the residual using the correct dimensions
    residual = np.dot(original_matrix[:, :-1], solution) - b

    plt.bar(range(len(residual)), residual, color='g', alpha=0.7)
    plt.title("Residuals (Error in Solution)")
    plt.xlabel("Equation")
    plt.ylabel("Residual")
    plt.show()


# Function to perform Gaussian elimination
def gaussianElimination(mat):
    mat = np.array(mat, dtype=float)
    mat_initial = np.array(mat, dtype=float)  # Store the initial matrix for visualization
    condition_numbers = []
    pivot_steps = []

    # Reduction into RREF
    singular_flag = forwardElim(mat, condition_numbers, pivot_steps)

    # If matrix is singular
    if singular_flag != -1:
        print("Singular Matrix.")
        if mat[singular_flag][N]:
            print("Inconsistent System.")
        else:
            print("May have infinitely many solutions.")
        return

    # Get solution to system and print it using backward substitution
    solution = backSub(mat)

    # Visualizations
    visualize_initial_and_final(mat_initial, mat)  # Before and After Comparison
    visualize_pivot_path(pivot_steps, mat_initial)  # Track pivot movement
    visualize_condition_number(condition_numbers)  # Condition number over time
    # visualize_residuals(mat_initial, solution)  # Residual plot

# Function for elementary operation of swapping two rows
def swap_row(mat, i, j):
    for k in range(N + 1):
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp

# Function to reduce matrix to row echelon form (RREF)
def forwardElim(mat, condition_numbers, pivot_steps):
    for k in range(N):
        # Initialize maximum value and index for pivot
        i_max = k
        v_max = mat[i_max][k]

        # Find greater amplitude for pivot if any
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                i_max = i

        # If a principal diagonal element is zero, the matrix is singular
        if not mat[k][i_max]:
            return k  # Matrix is singular

        # Swap the greatest value row with current row
        if i_max != k:
            swap_row(mat, k, i_max)
            pivot_steps.append((k, k))  # Record pivot movement

        for i in range(k + 1, N):
            # Factor f to set current row kth element to 0, and subsequently the kth column
            f = mat[i][k] / mat[k][k]

            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * f

            mat[i][k] = 0  # Filling lower triangular matrix with zeros

        # Condition number tracking (norm of the matrix)
        cond_number = np.linalg.cond(mat[:, :-1])
        condition_numbers.append(cond_number)

    return -1

# Function to calculate the values of the unknowns using backward substitution
def backSub(mat):
    x = [None for _ in range(N)]  # An array to store solution

    # Start calculating from the last equation up to the first
    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]

        # Initialize j to i+1 since the matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        # Divide the RHS by the coefficient of the unknown being calculated
        x[i] = x[i] / mat[i][i]

    print("\nSolution for the system:")
    for i in range(N):
        print(f"{x[i]:.8f}")

    return x

# Driver code
mat = [[2.0, -2.0, 0.0, -6.0],
       [1.0, -1.0, 1.0, 1.0],
       [0.0, 3.0, -2.0, -5.0]]

gaussianElimination(mat)
