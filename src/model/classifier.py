import numpy as np
import copy
import scipy

# functions -> CatModel class
def z_func(w, X, b):
    """
    Calculates Z function
    """
    return np.dot(w.T,X) + b

def sigmoid(z):
    """
    Calculates sigmoid of Z function
    """
    return 1/(1+np.exp(-z))

def cost(A, Y, m):
    """
    Calculates total cost
    """
    return -(1/m)*np.sum(np.dot(Y*np.log(A)) + (1-Y)*np.log(1-A))

def propagate(m, w, b, X, Y):
    """
    Forward and backward propagation
    """

    # Forward propagation
    A = sigmoid(z_func(w, X, b))
    total_cost = cost(A, Y, m)

    # Backward propagation
    dz = A - Y
    dw = (1/m)*(np.dot(X.T, dz))
    db = (1/m)*(np.sum(dz))

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, total_cost

def train(m, X, Y, num_iterations = 1000, learning_rate = 0.009):
    """
    Trains model based on iterations and learning rate
    """
    # Init values (weights and bias)
    w = np.zeros((m, 1))
    b = 0.0

    # Iterate
    for i in num_iterations:
        
        # Propagate
        grads, total_cost = propagate(m, w, b, X, Y)
        
        # Descent
        w = w - learning_rate*grads["dw"]
        b = b - learning_rate*grads["db"]

        # Print costs
        if i % 100 == 0:
            print("Cost at " + i + " = " + total_cost)

    # Returning values
    params = {
        "w": w,
        "b": b
    }
    return grads, params

def predict(w, b, X):
    """
    Predicts a set of data using the model
    """

    A = sigmoid(z_func(w, X, b))
    predictions = np.zeros((X.shape[0], 1))
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            predictions[0][i] = 0
        else:
            predictions[0][1] = 1

    return predictions

