import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import math_utils
from models import model_stage

class FF_NeuralNetwork():
    def __init__(self, layers_dims, **kwargs):        
        self.parameters = {}
        self.layer_dims = layers_dims
        self.hidden_activation_fn = "tanh"
        self.output_activation_fn = "sigmoid"
        self.learning_rate = .03
        self.weight_init_type = "scalar"
        if('hidden_fn' in kwargs):
            self.hidden_activation_fn = kwargs["hidden_fn"]
        if('output_fn' in kwargs):
            self.output_activation_fn = kwargs["output_fn"]
        if('lrn_rate' in kwargs):
            self.learning_rate = kwargs["lrn_rate"]
        if('weight_init' in kwargs):
            self.weight_init_type = kwargs["weight_init"]
        self.init_parameters(layers_dims)

    def init_parameters(self, layers_dims):
        L = len(layers_dims)            
        for l in range(1, L):   
            self.parameters["W" + str(l)] = self.weight_init(layers_dims[l], layers_dims[l - 1]) 
            self.parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
            assert self.parameters["W" + str(l)].shape == (
                layers_dims[l], layers_dims[l - 1])
            assert self.parameters["b" + str(l)].shape == (layers_dims[l], 1)   

    def weight_init(self, dim1, dim2):
        if(self.weight_init_type == "scalar"):
            return(math_utils.scalar_initialization(dim1,dim2))
        if(self.weight_init_type == "xavier"):
            return(math_utils.xavier_initialization(dim1,dim2))
        
    def forward_prop_calc(self, A_prev, l, activation_fn):
        W = self.parameters["W" + str(l)]
        b = self.parameters["b" + str(l)]
        linear_cache = (A_prev, W, b)
        Z = np.dot(W, A_prev) + b
        activation_cache = Z
        A = self.activation(Z, activation_fn)
        assert A.shape == (W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)
        return A, cache

    def activation(self, Z, activation_fn):
        if activation_fn == "sigmoid":
            A = math_utils.sigmoid(Z)
        elif activation_fn == "tanh":
            A = math_utils.tanh(Z)
        elif activation_fn == "relu":
            A = math_utils.relu(Z)
        return A

    def forward_propogation(self, X):
        A = X                           
        caches = []                     
        L = len(self.parameters) // 2        
        for l in range(1, L):
            A_prev = A
            A, cache = self.forward_prop_calc(A_prev, l, self.hidden_activation_fn)
            caches.append(cache)
        AL, cache = self.forward_prop_calc(A, L, self.output_activation_fn)
        caches.append(cache)
        assert AL.shape == (1, X.shape[1])
        return AL, caches

    @staticmethod
    def compute_cost(AL, y):
        m = y.shape[1]              
        cost = - (1 / m) * np.sum(
            np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
        return cost
    
    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters["W" + str(l)] = self.parameters[
                "W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters[
                "b" + str(l)] - self.learning_rate * grads["db" + str(l)]

    # define helper functions that will be used in L-model back-prop
    def linear_backword(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation_fn):
        linear_cache, activation_cache = cache
        dZ = self.gradient(dA, activation_cache, activation_fn)
        dA_prev, dW, db = self.linear_backword(dZ, linear_cache)
        return dA_prev, dW, db

    def gradient(self, dA, activation_cache, activation_fn):
        if activation_fn == "sigmoid":
            dZ = math_utils.sigmoid_gradient(dA, activation_cache)

        elif activation_fn == "tanh":
            dZ = math_utils.tanh_gradient(dA, activation_cache)

        elif activation_fn == "relu":
            dZ = math_utils.relu_gradient(dA, activation_cache) 
        return dZ

    def backpropogation(self, AL, y, caches):
        y = y.reshape(AL.shape)
        L = len(caches)
        grads = {}

        dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
            "db" + str(L)] = self.linear_activation_backward(
                dAL, caches[L - 1], self.output_activation_fn)

        for l in range(L - 1, 0, -1):
            current_cache = caches[l - 1]
            grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
                "db" + str(l)] = self.linear_activation_backward(
                    grads["dA" + str(l)], current_cache, self.hidden_activation_fn)

        return grads

    # Define the multi-layer model using all the helper functions we wrote before
    def train(self, X, Y, num_iterations=3000, sig=None, log=False):
        costs = []
        progress_steps = (num_iterations - (num_iterations % 50)) / 50
        percent_finished = 0
        for i in range(num_iterations):
            for batch_num in range(len(X)):
                AL, caches = self.forward_propogation(X[batch_num])
                grads = self.backpropogation(AL, Y[batch_num], caches)
                cost = self.compute_cost(AL, Y[batch_num])
                self.update_parameters(grads)
            costs.append(cost)
            if(sig is not None):
                if(i > (progress_steps*percent_finished)):
                    percent_finished = percent_finished + 1
                    sig.emit(percent_finished*2, cost)
            if(log and i%100 == 0):
                print("Cost after {} iterations: {}".format(i, cost))
        return costs

    def test_model(self, X_test, Y, log=False):
        probs, cache = self.forward_propogation(X_test)
        labels = (probs >= 0.5) * 1
        accuracy = round(np.mean(labels == Y) * 100,2)
        if(log):
            print(f"The accuracy rate is: {accuracy:.2f}%.")
        return accuracy
    
    def get_parameters(self):
        hyper_params = {}
        hyper_params.update({"Model Type": "Feed Forward Neural Network"})
        hyper_params.update({"Layer Dimensions": str(self.layer_dims)})
        hyper_params.update({"Learning Rate": str(self.learning_rate)})
        hyper_params.update({"Hidden Layer Activation Function": str(self.hidden_activation_fn)})
        hyper_params.update({"Output Layer Activation Function": str(self.output_activation_fn)})
        hyper_params.update({"Weight Initialization Type": str(self.weight_init_type)})
        return(hyper_params)


if __name__ == '__main__':
    stage = model_stage.ModelStage(console_log=True)

    did_load, log = stage.set_dataset("cats", 1)
    assert(did_load==True)
    print(log)

    did_set, log = stage.set_model(model_type="ff_neuralnet", layer_dims=[12288,5,5,1], hidden_fn="tanh", output_fn="sigmoid", lrn_rate=.03, weight_init="xavier")
    print(log)

    did_train, log = stage.do_train(training_iterations=2000)
    print(log)

    did_test, log = stage.do_test()
    print(log)

    stage.save_results()




