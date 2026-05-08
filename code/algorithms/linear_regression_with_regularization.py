import pandas as pd
import numpy as np

np.random.seed(3)

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_y)

def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    return {'W': W, 'b': b}

def forward_propagation(X, parameters):
    W = parameters['W']
    b = parameters['b']
    Z = np.matmul(W, X) + b
    Y_hat = Z
    return Y_hat

def compute_cost(Y_hat, Y, parameters, lambd=0, reg='none'):
    m = Y.shape[1]
    W = parameters['W']
    
    # Base MSE cost
    cost = np.sum((Y_hat - Y) ** 2) / (2 * m)
    
    # Regularization term
    if reg == 'l2':
        cost += (lambd / (2 * m)) * np.sum(np.square(W))
    elif reg == 'l1':
        cost += (lambd / m) * np.sum(np.abs(W))
    
    return cost

def backward_propagation(Y_hat, X, Y, parameters, lambd=0, reg='none'):
    m = Y_hat.shape[1]
    W = parameters['W']
    
    dZ = Y_hat - Y
    dW = (1/m) * np.dot(dZ, X.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    # Add regularization gradient (bias is NOT regularized)
    if reg == 'l2':
        dW += (lambd / m) * W
    elif reg == 'l1':
        dW += (lambd / m) * np.sign(W)
    
    return {'dW': dW, 'db': db}

def update_parameters(parameters, grads, learning_rate=1.2):
    W = parameters['W'] - learning_rate * grads['dW']
    b = parameters['b'] - learning_rate * grads['db']
    return {'W': W, 'b': b}

def nn_model(X, Y, num_iterations=10, learning_rate=1.2, lambd=0, reg='none', print_cost=False):
    n_x, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_y)
    
    for i in range(num_iterations):
        Y_hat = forward_propagation(X, parameters)
        cost = compute_cost(Y_hat, Y, parameters, lambd, reg)
        grads = backward_propagation(Y_hat, X, Y, parameters, lambd, reg)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost:
            print(f"Cost after iteration {i}: {cost:.6f}")
    
    return parameters