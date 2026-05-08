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

    parameters = {'W': W,
                  'b': b}
    
    return parameters

def forward_propagation(X, parameters):
    W = parameters['W']
    b = parameters['b']

    Z = np.matmul(W, X) + b

    Y_hat = Z

    return Y_hat

def compute_cost(Y_hat, Y):
    m = Y.shape[1]

    cost = np.sum((Y_hat - Y)**2)/(2*m)

    return cost

def backward_propagation(Y_hat, X, Y):
    m = Y_hat.shape[1]

    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    grads = {'dW': dW,
             'db': db}
    
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    W = parameters['W']
    b = parameters['b']
    dW = grads['dW']
    db = grads['db']

    W = W - learning_rate * dW
    b = b - learning_rate * db

    parameters = {"W": W,
                  "b": b}
    
    return parameters

def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    n_x, n_y = layer_sizes(X, Y)
    
    parameters = initialize_parameters(n_x, n_y)

    for i in range(0, num_iterations):
        Y_hat = forward_propagation(X, parameters)

        cost = compute_cost(Y_hat, Y)

        grads = backward_propagation(Y_hat, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost:
            print("Cost after iterations %i: %f" %(i, cost))

    return parameters

def predict(X, Y, parameters, X_pred):
    W = parameters['W']
    b = parameters['b']

    if isinstance(X, pd.Series):
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_pred_norm  = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(np.mean(X)).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)

    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)

    return Y_pred[0]
