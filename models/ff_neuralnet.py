import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import sys_utils, math_utils
import time
import datetime

class FF_NeuralNetwork():
    def __init__(self, layers_dims, hidden_activation_func="tanh", output_activation_func="sigmoid", learning_rate=0.01, init_type="scalar", log ="none"):             
        self.parameters = {}
        self.layer_dims = layers_dims
        self.hidden_activation_fn = hidden_activation_func
        self.output_activation_fn = output_activation_func
        self.learning_rate = learning_rate
        self.init_type = init_type
        self.log_level = log
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
        if(self.init_type == "scalar"):
            return(math_utils.scalar_initialization(dim1,dim2))
        if(self.init_type == "xavier"):
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
    def train(self, X, Y, num_iterations=3000, sig=None):
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
            if(self.log_level == "standard" and i%100 == 0):
                print("Cost after {} iterations: {}".format(i, cost))
        return costs

    def test_model(self, X_test, Y):
        probs, cache = self.forward_propogation(X_test)
        labels = (probs >= 0.5) * 1
        accuracy = np.mean(labels == Y) * 100
        if(self.log_level == "standard" or self.log_level == "debug"):
            print(f"The accuracy rate is: {accuracy:.2f}%.")
        return accuracy
    
    def get_parameters(self):
        hyper_params = {}
        hyper_params.update({"Model Type": "Feed Forward Neural Network"})
        hyper_params.update({"Layer Dimensions": str(self.layer_dims)})
        hyper_params.update({"Learning Rate": str(self.learning_rate)})
        hyper_params.update({"Hidden Layer Activation Function": str(self.hidden_activation_fn)})
        hyper_params.update({"Output Layer Activation Function": str(self.output_activation_fn)})
        hyper_params.update({"Weight Initialization Type": str(self.init_type)})
        return(hyper_params)

    @staticmethod
    def post_process(data_set, num_iter, ds_size, mb_size, hyper_params, train_acc, test_acc, costs, exec_time):
        layers_dims = hyper_params.get("Layer Dimensions")
        learning_rate = hyper_params.get("Learning Rate")
        hl_act_fn = hyper_params.get("Hidden Layer Activation Function")
        output_act_fn = hyper_params.get("Output Layer Activation Function")
        weight_init = hyper_params.get("Weight Initialization Type")

        now = datetime.datetime.now()
        timestamp = now.strftime("%m/%d/%Y-%H:%M:%S")
        file_friendly_ts = now.strftime("%m-%d-%Y-%H_%M-%S")
        proj_dir = os.path.dirname(os.path.realpath(__file__))
        results_folder = proj_dir + "\\..\\results\\"
        results_file = results_folder + "logs\\ff_nn.log"
        costs_file_name = "plots\\nn_costs-" + file_friendly_ts + ".png"
        costs_file = results_folder + costs_file_name
        plt.plot(costs)
        plt.title(costs_file_name.replace(".png", ''))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig(costs_file)
        f = open(results_file, "a")
        entry_title = "********** {} Run **********\n".format(timestamp)
        nn_params = "NN Dimensions: {}\nLearning Rate: {}\nHidden Layer Act Fn: {}\nOutput Layer Act Fn: {}\nWeight Initialization: {}\n".format(
                    str(layers_dims), str(learning_rate), hl_act_fn, output_act_fn, weight_init
        )
        test_run_stats = "Data Set: {}\nIterations {}\nDataset Size: {}\nMiniBatch Size: {}\nTraining Accuracy: {}\nTest Accuracy: {}\nCost: {}\nExecution Time: {}\nIterations/Sec: {}\n".format(
                data_set, str(num_iter), str(ds_size), str(mb_size), str(train_acc), str(test_acc), str(round(costs[-1],2)), str(exec_time), str(round(num_iter/exec_time,2)))
        other_results_info = "Costs Plot File: {}\n".format(costs_file_name)
        f.write(entry_title)
        f.write(nn_params)
        f.write(test_run_stats)
        f.write(other_results_info)
        f.write("********************************************\n")

def run(data_set, nn_dims, act_fn, init_type, lrn_rate, training_iterations, batches=1):
    # data settings
    data_set = "cats"
    data = sys_utils.import_data(data_set, batches)
    #info = sys_utils.format_ds_info(data_set)
    #print(info)

    X_train = data.get("Flattened Training Set")
    X_train_batched = data.get("Batched Training Set")
    y_train = data.get("Training Set Labels")
    y_train_batched = data.get("Batched Training Labels")
    X_test = data.get("Flattened Test Set")
    y_test = data.get("Test Set Labels")

    # Setting hypers and test run params
    nn_dimensions = [X_train_batched[0].shape[0]] + nn_dims
    num_iterations = training_iterations

    nn = FF_NeuralNetwork(nn_dimensions, hidden_activation_func=act_fn, init_type=init_type, learning_rate=lrn_rate, log="standard")
    start = time.time()
    costs = nn.train(X_train_batched, y_train_batched, num_iterations)
    execution_time = time.time() - start
    print("Execution Time: ", execution_time)

    training_accuracy = nn.test_model(X_train, y_train)
    test_accuracy = nn.test_model(X_test, y_test)

    hyper_params = nn.get_parameters()
    mini_batch_size = X_train_batched[0].shape[1]
    ds_size = X_train.shape[0]
    nn.post_process(
        data_set, num_iterations, ds_size, mini_batch_size,
        hyper_params, round(training_accuracy,2), round(test_accuracy,2), 
        costs, round(execution_time,2)
    )

if __name__ == '__main__':
    run("cats", [7, 5, 5, 1], "tanh", "xavier", 0.03, 2000, 5)



