import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def tanh(z):
    return(2/(1+np.exp(-2*z)))

def column_wise_shuffle(data):
    return(np.random.shuffle(data.T))

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_derivative(z):
    return(sigmoid(z) * (1-sigmoid(z)))

def tanh_derivative(Z):
    dZ = (1 - np.square(tanh(Z)))
    return dZ

def relu_derivative(Z):
    A = relu(Z)
    dZ = np.int64(A > 0)
    return dZ