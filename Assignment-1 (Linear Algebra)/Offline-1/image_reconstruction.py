import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform Singular Value Decomposition (SVD)
def low_rank_approximation(A, k):
    U, S, Vt = np.linalg.svd(A)
    A_k = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return A_k

# Load and resize image
image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#find k values from the original (n,m) matrix
k_values = list(range(1, min(image.shape[0], image.shape[1]), max(1, (min(image.shape[0], image.shape[1]) // 15))))

image_dimensions = (500, 500)
resized_image = cv2.resize(gray_image, image_dimensions)

fig, axs = plt.subplots(3, 4, figsize=(15, 16))

for k, ax in zip(k_values, axs.flatten()):
    approximation = low_rank_approximation(resized_image, k)

    ax.imshow(approximation, cmap='gray')
    ax.set_title(f'n_components = {k}')

plt.tight_layout()
#plt.show()
#Save plot as pdf
pdf_path = "output.pdf"
plt.savefig(pdf_path)