import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import lr_utils, math_utils


class FF_NeuralNetwork():

    def __init__(self, layer_dims, activation_func="sigmoid", log = "standard"):
        self.weights = []
        self.bias = []
        self.pre_acts = []
        self.act_function = activation_func
        for i in range(1, len(layer_dims)):
            w = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            b = np.zeros((layer_dims[i],1))
            self.weights.append(w)
            self.bias.append(b)
        self.act_function = activation_func
        self.log_level = log
 
    def forward_propogation(self, X):
        """
            X -- A (len of feature vector, num examples) matrix
            The neural network should have been initialized with the first layer
            dimension being the len of the feature vector
            returns A, a (num examples, 1) array of floats interpreted as the output
        """
        assert(X.shape[0] == self.weights[0].shape[1])
        A = X
        cache = []
        for i in range(len(self.weights)):
            if(self.log_level == "debug"): 
                print("{} shape A {} shape W {} shape b {}".format(i,np.shape(A), np.shape(self.weights[i]), np.shape(self.bias[i])))
            Z = np.dot(self.weights[i], A) + self.bias[i]
            if(self.act_function == "sigmoid"):
                A = math_utils.sigmoid(Z)
            if(self.act_function == "tanh"):
                A = math_utils.tanh(Z)
        if(self.log_level == "debug"):
            print("{} shape A {}".format(i,np.shape(A)))
        return A

    def train(self, training_set, labels):
        classifications = self.forward_propogation(training_set)
        cost = self.compute_cost(classifications, labels)

    @staticmethod
    def compute_cost(classifications, labels):
        m = labels.shape[1]
        cost = -1/m * np.sum(labels * np.log(classifications) + (1 - labels) * np.log(1 - classifications))    
        cost = np.squeeze(cost)
        return cost

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w_cpy = self.weights.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        if(self.act_function == "sigmoid"):
            A = math_utils.sigmoid(np.dot(w_cpy.T, X) + self.bias)
        if(self.act_function == "tanh"):
            A = math_utils.tanh(np.dot(w_cpy.T, X) + self.bias)
        
        for i in range(A.shape[1]):
            
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] > .5:
                Y_prediction[0,i] = True
            else:
                Y_prediction[0,i] = False
        return Y_prediction
    
    def score_predictions(self, pred, Y):
        pred = pred[0]
        Y = Y[0]
        total = len(pred)
        correct = 0
        for i in range(total):
            if(pred[i] == Y[i]):
                correct = correct +1
        per_score = correct/total*100
        print("Scored {} out of {}, or {}%".format(correct, total, per_score))
        return per_score


if __name__ == '__main__':
    data = lr_utils.import_data("cats", do_log=True)
    input_size = data.get("Flattened Training Set").shape[0]
    nn = FF_NeuralNetwork([input_size, 20, 7, 5, 1], activation_func="sigmoid", log="debug")
    flattened_Set = data.get("Flattened Training Set")
    nn.forward_propogation(flattened_Set)
