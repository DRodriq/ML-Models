import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import lr_utils, math_utils
import random


class FF_NeuralNetwork():

    def __init__(self, layer_dims, activation_func="sigmoid", log = "standard", learning_rate = 0.001):
        self.weights = []
        self.bias = []
        self.pre_acts = []
        self.act_function = activation_func
        self.learning_rate = learning_rate
        self.log_level = log

        if(self.log_level == "debug"):
            print("************** NN Initialization LOG **************")
            print("Activation Function:", self.act_function)
            print("Learning Rate:", str(self.learning_rate))

        for i in range(1, len(layer_dims)):
            w = np.random.randn(layer_dims[i], layer_dims[i - 1]) *.001
            b = np.zeros((layer_dims[i],1))
            self.weights.append(w)
            self.bias.append(b)
            if(self.log_level == "debug"):
                print("Weight {}: {}".format(i, w))
                print("Layer {} dimensions {}".format(i, np.shape(self.weights[-1])))
                print("Bias {} dimensions {}".format(i, np.shape(self.bias[-1])))

        if(self.log_level == "debug"):
            print("************** NN Initialization LOG **************\n")
 
    def forward_propogation(self, X):
        """
            X -- A (len of feature vector, num examples) matrix
            The neural network should have been initialized with the first layer
            dimension being the len of the feature vector
            returns A, a (num examples, 1) array of floats interpreted as the output
        """
        if(self.log_level == "debug"):
            print("************** Forward Prop LOG **************")
        assert(X.shape[0] == self.weights[0].shape[1])
        X = np.reshape(X,(12288,1))
        A = X
        caches = []
        for i in range(len(self.weights)):
            cache = []
            A_prev = A
            if(self.log_level == "debug"): 
                print("{} shape A {} shape W {} shape b {}".format(i+1,np.shape(A_prev), np.shape(self.weights[i]), np.shape(self.bias[i])))
            Z = np.dot(self.weights[i], A_prev) + self.bias[i]
            A = self.activation(Z)
            print("{}: A_prev: {}".format(i, A_prev))
            print("{}: Z: {}".format(i, Z))
            print("{}: A: {}".format(i, A))
            cache = (Z, A_prev, self.weights[i], self.bias[i])
            caches.append(cache)
            if(self.log_level == "debug"):
                print("L{} : A_Prev dimensions {}".format(i+1, np.shape(A_prev)))
                print("L{} : W dimensions {}".format(i+1, np.shape(cache[2])))
                print("L{} : Z dimensions {}".format(i+1, np.shape(cache[0])))
                print("L{} : A dimensions {}".format(i+1, np.shape(A)))
        return A, caches

    def activation(self, Z):
        if(self.act_function == "sigmoid"):
            A = math_utils.sigmoid(Z)
        if(self.act_function == "tanh"):
            A = math_utils.tanh(Z)
        if(self.act_function == "relu"):
            A = math_utils.relu(Z)            
        return A
    
    def derivative(self, Z):
        if(self.act_function == "sigmoid"):
            dZ = math_utils.sigmoid_derivative(Z)
        if(self.act_function == "tanh"):
            dZ = math_utils.tanh_derivative(Z)
        if(self.act_function == "relu"):
            dZ = math_utils.relu_derivative(Z)
        return dZ

    def train(self, training_set, labels, iterations = 100):
        for i in range(iterations):
            cache = []
            classifications, cache = self.forward_propogation(training_set)
            gradients = self.backward_propagation(classifications, cache, labels)
            self.update_parameters(gradients)
            if(self.log_level == "debug"):
                cost = self.compute_cost(classifications, labels)
                print("\nCost after {} runs: {}\n".format(i+1,cost))

    @staticmethod
    def compute_cost(classifications, labels):
        m = labels.shape[1]
        cost = -1/m * np.sum(labels * np.log(classifications) + (1 - labels) * np.log(1 - classifications))    
        cost = np.squeeze(cost)
        return cost
    
    def backward_propagation(self, AL, caches, Y):
        """
            caches -- A list of caches from the forward propagation function
            cost -- The cost computed by the compute_cost function
            returns gradients -- A dictionary containing the gradients of the weights and biases
        """
        if(self.log_level == "debug"):
            print("\n************** BackProp LOG **************")
        gradients = {}
        m = len(caches)
       # Y = Y.reshape(A.shape)
        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        for i in reversed(range(m)):
            (Z, A, W, b) = caches[i]
            dZ = self.derivative(Z) * dA
            gradients["dW" + str(i + 1)] = (1/m) * np.dot(dZ, A.T)
            gradients["db" + str(i + 1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if(self.log_level == "debug"):
                print("Layer", i+1)
                print("Shape of W : {}".format(np.shape(W)))
                print("Shape of A : {}".format(np.shape(A)))
                print("Shape of dA : {}".format(np.shape(dA)))
                print("Shape of dZ : {}".format(np.shape(dZ)))
                print("Shape of {} : {}".format("dW" + str(i + 1), np.shape(gradients.get("dW" + str(i + 1)))))
                print("Shape of {} : {}".format("db" + str(i + 1), np.shape(gradients.get("db" + str(i + 1)))))
            dA = np.dot(W.T, dZ)
        return gradients
        
    def update_parameters(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (gradients.get("dW" + str(i+1)) * self.learning_rate)
            self.bias[i] = self.bias[i] - (gradients.get("db" + str(i+1)) * self.learning_rate)

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
        A, caches = self.forward_propogation(X)
        return A
    
    def score_predictions(self, pred, Y):
        total = len(pred)
        correct = 0
        print(pred)
        for i in range(total):
            prediction = 0
            if(pred[i] >= .5):
                prediction = 1
            if(prediction == Y[i]):
                correct = correct +1
        per_score = correct/total*100
        print("Scored {} out of {}, or {}%".format(correct, total, per_score))
        return per_score


if __name__ == '__main__':
    data = lr_utils.import_data("cats", do_log=True)
    input_size = data.get("Flattened Training Set").shape[0]
    nn = FF_NeuralNetwork([input_size, 20, 7, 7, 5, 1], activation_func="sigmoid", log="debug", learning_rate=.03)
    flattened_Set = data.get("Flattened Training Set")
    labels = data.get("Training Set Labels")
    #nn.train(flattened_Set, labels, iterations=10)
    ex = random.randint(0,200)
    print(np.shape(flattened_Set[:,ex]))
    A, cache = nn.forward_propogation(flattened_Set[:,0])
    print(A)
    print(labels[:,ex])
    #preds = nn.predict(data.get("Flattened Test Set"))
    #score = nn.score_predictions(preds, data.get("Test Set Labels"))


