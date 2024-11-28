# Image Compression
# Task: Use an orthogonal matrix (via SVD) to approximate an image with reduced rank.
# Visualization: Display the original image and approximations at different ranks.
# code:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import data, color
from skimage.transform import resize

# Load and preprocess an image
image = color.rgb2gray(data.astronaut())
image = resize(image, (100, 100))

# Perform SVD
U, S, Vt = np.linalg.svd(image, full_matrices=False)

# Reconstruct image with reduced ranks
ranks = [5, 20, 50]
plt.figure(figsize=(12, 4))
for i, rank in enumerate(ranks):
    approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    plt.subplot(1, len(ranks), i + 1)
    plt.imshow(approx, cmap='gray')
    plt.title(f"Rank-{rank} Approximation")
    plt.axis('off')

plt.suptitle("Image Compression using SVD")
plt.show()