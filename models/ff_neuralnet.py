import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import math_utils
from models import model_stage
from models import ml_model

class FF_NeuralNetwork():
    def __init__(self, layers_dims, **kwargs):        
        self.weights = []
        self.bias = []
        self.act_fns = ["tanh" for i in range(len(layers_dims)-2)] + ["sigmoid"]
        self.learning_rate = .03
        self.weight_init_type = "scalar"
        self.cost_function = math_utils.compute_logistic_cost
        if('act_fns' in kwargs):
            self.act_fns = kwargs["act_fns"]
        if('lrn_rate' in kwargs):
            self.learning_rate = kwargs["lrn_rate"]
        if('weight_init' in kwargs):
            self.weight_init_type = kwargs["weight_init"]
        if('cost_fn' in kwargs):
            self.cost_function = kwargs["cost_fn"]
        self.init_parameters(layers_dims)

    def init_parameters(self, layers_dims):
        L = len(layers_dims)            
        for l in range(1, L):   
            self.weights.append(self.weight_init(layers_dims[l], layers_dims[l - 1]))
            self.bias.append(np.zeros((layers_dims[l], 1)))
            assert self.weights[l-1].shape == (
                layers_dims[l], layers_dims[l - 1])
            assert self.bias[l-1].shape == (layers_dims[l], 1)   

    def weight_init(self, dim1, dim2):
        if(self.weight_init_type == "scalar"):
            return(math_utils.scalar_initialization(dim1,dim2))
        if(self.weight_init_type == "xavier"):
            return(math_utils.xavier_initialization(dim1,dim2))
        
    def forward_propogation(self, X):
        A = X                           
        caches = []                     
        L = len(self.weights) 
        for l in range(0, L):
            A_prev = A
            W = self.weights[l]
            b = self.bias[l]
            act_fn = self.act_fns[l]
            cache = (A_prev, W, b)
            #A, cache = self.forward_prop_calc(A_prev, l, self.hidden_activation_fn)
            Z, A = self.forward_prop(A_prev, W, b, act_fn)
            cache = (cache, Z)
            caches.append(cache)
        #AL, cache = self.forward_prop_calc(A, L, self.output_activation_fn)
        #caches.append(cache)
        assert A.shape == (1, X.shape[1])
        return A, caches

    def forward_prop(self, X, W, b, act_fn):
        Z, A = math_utils.forward_prop(X, W, b, act_fn)
        return Z,A
    
    def update_parameters(self, grads):
        L = len(self.weights)
        for l in range(0, L):
            self.weights[l] = self.weights[l] - (self.learning_rate * grads["dW" + str(l+1)])
            self.bias[l] = self.bias[l] - (self.learning_rate * grads["db" + str(l+1)])

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
                dAL, caches[L - 1], self.act_fns[-1])

        for l in range(L-1, 0, -1):
            current_cache = caches[l - 1]
            grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
                "db" + str(l)] = self.linear_activation_backward(
                    grads["dA" + str(l)], current_cache, self.act_fns[l-1])

        return grads
    
    def add_node_to_layer(self, L):
        pass


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
        layer_dims = []
        layer_dims = [W.shape[1] for W in self.weights]
        layer_dims = layer_dims + [self.weights[-1].shape[0]]
        hyper_params.update({"Layer Dimensions": str(layer_dims)})
        hyper_params.update({"Learning Rate": str(self.learning_rate)})
        hyper_params.update({"Activation Functions": str(self.act_fns)})
        hyper_params.update({"Weight Initialization Type": str(self.weight_init_type)})
        return(hyper_params)
    
    def compute_cost(self, AL, Y):
        cost = self.cost_function(AL,Y)
        return cost
    
    def print_characteristics(self):
        for l in range(0, len(self.weights)):
            W = self.weights[l]
            b = self.bias[l]
            act_fn = self.act_fns[l]
            print("L_{}: {}".format(l,W.shape))
            print("b_{}: {}".format(l,b.shape))
            print("act_fn_{}:{}".format(l, act_fn))
        return


if __name__ == '__main__':
    stage = model_stage.ModelStage(console_log=True)

    did_load, log = stage.set_dataset("cats", 5)

    assert(did_load==True)
    print(log)

    did_set, log = stage.set_model(model_type="ff_neuralnet", layer_dims=[12288,5,5,1], act_fns=["tanh","tanh","sigmoid"], lrn_rate=.03, weight_init="scalar")
    print(log)

    stage.model.print_characteristics()

    did_train, log = stage.do_train(training_iterations=2000)
    print(log)

    did_test, log = stage.do_test()
    print(log)

    stage.save_results()
