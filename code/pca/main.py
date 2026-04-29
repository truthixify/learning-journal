import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils

def get_cov_matrix(X):
    """ Calculate covariance matrix from centered data X
    Args:
        X (np.ndarray): centered data matrix
    Outputs:
        cov_matrix (np.ndarray): covariance matrix
    """

    cov_matrix = X.T @ X 
    cov_matrix = 1/(X.shape[0] - 1) * cov_matrix
    
    return cov_matrix

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

imgs_flatten = np.array([im.reshape(-1) for im in imgs])
X = center_data(imgs_flatten)
cov_matrix = get_cov_matrix(X)

np.random.seed(7)
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=35)

print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')

Xred2 = perform_PCA(X, eigenvecs, 2)
