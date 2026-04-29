import numpy as np
import pandas as pd


def initialize_parameters(n_x):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    
    W = np.random.randn(1, n_x) * 0.01
    b = np.zeros((1, 1))


    assert (W.shape == (1, n_x))
    assert (b.shape == (1, 1))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m), where n_x is the dimension input (in our example is 2) and m is the number of training samples
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (1, m)
    """

    # Retrieve each parameter from the dictionary "parameters".
    W = parameters['W']
    b = parameters['b']

    # Implement Forward Propagation to calculate Z.
    Z = W @ X + b
    Y_hat = Z

    return Y_hat

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

def nn_model(X, Y, num_iterations=1000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """

    n_x = X.shape[0]

    # Initialize parameters
    parameters = initialize_parameters(n_x)

    # Loop
    for i in range(0, num_iterations):

        # Forward propagation, Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)

        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        
        # Parameters update.
        parameters = train_nn(parameters, Y_hat, X, Y, learning_rate = 0.001) 
        
        # Print the cost every iteration.
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(X, parameters):
    W = parameters['W']
    b = parameters['b']

    Z = W @ X + b
    return Z

df = pd.read_csv("/Users/truthixify/logs/learning-journal/code/nn/data/toy_dataset.csv")

X = np.array(df[['x1','x2']]).T
Y = np.array(df['y']).reshape(1,-1)

parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)
y_hat = predict(X,parameters)
df['y_hat'] = y_hat[0]

for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")