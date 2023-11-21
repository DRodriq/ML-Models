import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def tanh(z):
    return(2/(1+np.exp(-2*z)))

def column_wise_shuffle(data):
    return(np.random.shuffle(data.T))