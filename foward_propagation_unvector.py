import numpy as np
from planar_utils import *


def forward_propagation(X, Y, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    A1 = np.zeros(2)
    for i, x in enumerate(X):
        print('x: ', x)
        for j, w in enumerate(W1):
            Z1 = np.multiply(w, x) + b1[j]
            A1[j] = np.tanh((np.sum(Z1)))

        
       
        Z2 = np.multiply(W2, A1) 
        A2 = sigmoid(np.sum(Z1))
        print('A1: ', A1)
        print('A2: ', A2)
        print('Z2: ',Z2)
        print('W1: ', W1)
        print('W2: ', W2)

        cache = {"Z1": Z1, "A2": A2, "A1": A1}
    
    return A2, cache
