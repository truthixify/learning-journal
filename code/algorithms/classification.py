import pandas as pd
import numpy as np

np.random.seed(3)

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x, n_y)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

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

    A = sigmoid(Z)

    return A

def compute_cost(A, Y):
    m = Y.shape[1]

    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)

    return cost

def backward_propagation(A, X, Y):
    m = X.shape[1]

    dZ = A - Y
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
        A = forward_propagation(X, parameters)

        cost = compute_cost(A, Y)

        grads = backward_propagation(A, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost:
            print("Cost after iterations %i: %f" %(i, cost))

    return parameters

def predict(X, parameters):
    A = forward_propagation(X, parameters)
    predictions = A > 0.5

    return predictions

m = 3000
X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))

parameters = nn_model(X, Y, num_iterations=1000, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")