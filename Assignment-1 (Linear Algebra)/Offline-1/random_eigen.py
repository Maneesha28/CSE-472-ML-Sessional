import numpy as np

def generate_random_invertible_matrix(n):
    A = np.random.randint(-10, 10,  size=(n, n))
    
    while np.linalg.det(A) == 0:
        A = np.random.randint(-10, 10, size=(n, n))
    
    return A

def eigen_decomposition_reconstruction(eigenvalues, eigenvectors):
    A_reconstructed = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    return A_reconstructed

def check_reconstruction_success(A, A_reconstructed):
    if(np.allclose(A, A_reconstructed)):
        print("Reconstruction successful!")
    else:
        print("Reconstruction failed!")
    
    #np.set_printoptions(suppress=True, precision=4)
    print("Original matrix:\n", A, "\n")
    print("Reconstructed matrix:\n", np.real(np.round(A_reconstructed)))


n = int(input("Enter the dimension of the matrix: "))

A = generate_random_invertible_matrix(n)

eigenvalues, eigenvectors = np.linalg.eig(A)

A_reconstructed = eigen_decomposition_reconstruction(eigenvalues, eigenvectors)

check_reconstruction_success(A, A_reconstructed)

