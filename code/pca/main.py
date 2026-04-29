import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils

def center_data(Y):
    """
    Center your original data
    Args:
         Y (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.repeat(mean_vector, Y.shape[0])
 
    mean_matrix = np.reshape(mean_matrix, Y.shape, order='F')
    
    X = Y - mean_matrix
    return X

def perform_PCA(X, eigenvecs, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        X (ndarray): original data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvecs (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector 
                            is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    
    V = eigenvecs[:,:k]
    Xred = X @ V
    return Xred

imgs = utils.load_images('/Users/truthixify/logs/learning-journal/code/pca/data/')
height, width = imgs[0].shape

print(f'\nYour dataset has {len(imgs)} images of size {height}x{width} pixels\n')
plt.imshow(imgs[0], cmap='gray')