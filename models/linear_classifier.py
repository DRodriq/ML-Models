import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import sys_utils, math_utils
import time
import datetime

class LinearClassifier():

    def __init__(self, activation_func="sigmoid", learn_rate=0.009):
        self.weights = np.zeros((2,1))
        self.bias = 0.0
        self.act_function = activation_func
        self.learning_rate = learn_rate

    def init_parameters(self, data_shape):
        self.weights = np.zeros((data_shape,1))

    def forward_propogation(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        grads -- dictionary containing the gradients of the weights and bias
                (dw -- gradient of the loss with respect to w, thus same shape as w)
                (db -- gradient of the loss with respect to b, thus same shape as b)
        cost -- negative log-likelihood cost for logistic regression
        """
        m = X.shape[1]

        if(self.act_function == "sigmoid"):
            A = math_utils.sigmoid(np.dot((self.weights).T, X) + self.bias)
        if(self.act_function == "tanh"):
            A = math_utils.tanh(np.dot((self.weights).T, X) + self.bias)
        cost = self.compute_cost(A,Y)

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)
        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    @staticmethod
    def compute_cost(AL, y):
        m = y.shape[1]              
        cost = - (1 / m) * np.sum(
            np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
        return cost

    def train(self, X, Y, num_iterations=100, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        costs = []
        
        for i in range(num_iterations):
            # Cost and gradient calculation 
            grads, cost = self.propagate(X,Y)
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            self.weights = self.weights - self.learning_rate*(dw)
            self.bias = self.bias - self.learning_rate*(db)

            costs.append(cost)

        grads = {"dw": dw,
                "db": db}
        return costs

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
    
    @staticmethod
    def post_process(costs, act_fn, learning_rate, data_set, num_iters, train_acc, test_acc, exec_time):
        now = datetime.datetime.now()
        timestamp = now.strftime("%m/%d/%Y-%H:%M:%S")
        file_friendly_ts = now.strftime("%m-%d-%Y-%H_%M-%S")
        proj_dir = os.path.dirname(os.path.realpath(__file__))
        results_folder = proj_dir + "\\..\\results\\"
        results_file = results_folder + "logs\\linear_classifier.log"
        costs_file_name = "plots\\lc_costs-" + file_friendly_ts + ".png"
        costs_file = results_folder + costs_file_name
        #plt.plot(costs)
        #plt.title(costs_file_name.replace(".png", ''))
        #plt.xlabel("Iteration")
        #plt.ylabel("Cost")
        #plt.savefig(costs_file)
        f = open(results_file, "a")
        entry_title = "********** {} Run **********\n".format(timestamp)
        nn_params = "Learning Rate: {}\nActivation Function: {}\n".format(
                    str(learning_rate), act_fn
        )
        test_run_stats = "Data Set: {}\nIterations {}\nTraining Accuracy: {}\nTest Accuracy: {}\nCost: {}\nExecution Time: {}\nIterations/Sec: {}\n".format(
                data_set, str(num_iters), str(train_acc), str(test_acc), str(round(costs[-1],2)), str(exec_time), str(round(num_iters/exec_time,2)))
        #other_results_info = "Costs Plot File: {}\n".format(costs_file_name)
        f.write(entry_title)
        f.write(nn_params)
        f.write(test_run_stats)
        #f.write(other_results_info)
        f.write("********************************************\n")        


if __name__ == '__main__':
    data_set = "cats"
    num_iterations = 1000
    learning_rate = 0.09
    data = sys_utils.import_data(data_set, do_log=True)

    X_train = data.get("Flattened Training Set")
    y_train = data.get("Training Set Labels")
    X_test = data.get("Flattened Test Set")
    y_test = data.get("Test Set Labels")

    X_train = X_train / np.amax(X_train)
    X_test = X_test / np.amax(X_test)

    linear_classifier = LinearClassifier(activation_func="sigmoid", learn_rate=learning_rate)
    linear_classifier.init_weights(data.get("Flattened Training Set").shape[0])
    start = time.time()
    costs = linear_classifier.train(X_train, y_train, num_iterations)
    execution_time = time.time() - start
    print("Execution Time: ", execution_time)

    predictions = linear_classifier.predict(X_train)
    training_set_score = linear_classifier.score_predictions(predictions, y_train)
    predictions = linear_classifier.predict(X_test)
    test_set_score = linear_classifier.score_predictions(predictions, y_test)

    linear_classifier.post_process(
        costs, linear_classifier.act_function, linear_classifier.learning_rate, 
        data_set, num_iterations, training_set_score, test_set_score, execution_time
        )