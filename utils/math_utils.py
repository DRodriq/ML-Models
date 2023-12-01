import numpy as np

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def tanh(Z):
    A = np.tanh(Z)
    return A

def relu(Z):
    A = np.maximum(0, Z)
    return A

def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_derivative(Z):
    return(sigmoid(Z) * (1-sigmoid(Z)))

def tanh_derivative(Z):
    return(1-np.square(tanh(Z)))

def sigmoid_gradient(dA, Z):
    A = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

def tanh_gradient(dA, Z):
    A = tanh(Z)
    dZ = dA * (1 - np.square(A))
    return dZ

def relu_gradient(dA, Z):
    A = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))
    return dZ

def scalar_initialization(dim1, dim2, scalar=0.01):
    return(np.random.randn(dim1, dim2) * scalar)

def xavier_initialization(dim1, dim2, numerator=1):
    return(np.random.randn(dim1, dim2) * np.sqrt(numerator/dim2))