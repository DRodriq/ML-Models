import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import lr_utils, math_utils


class LinearClassifier():

    def __init__(self, activation_func="sigmoid"):
        self.weights = np.zeros((2,1))
        self.bias = 0.0
        self.act_function = activation_func

    def init_weights(self, data_shape):
        self.weights = np.zeros((data_shape,1))

    def propagate(self, X, Y):
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

        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)
        cost = np.squeeze(np.array(cost))
        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    def train(self, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
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
            
            self.weights = self.weights - learning_rate*(dw)
            self.bias = self.bias - learning_rate*(db)

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        grads = {"dw": dw,
                "db": db}
        return grads, costs

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
    
    linear_classifier = LinearClassifier(activation_func="sigmoid")
    linear_classifier.init_weights(data.get("Flattened Training Set").shape[0])
    linear_classifier.train(data.get("Flattened Training Set"), data.get("Training Set Labels"))
    predictions = linear_classifier.predict(data.get("Flattened Test Set"))

    linear_classifier.score_predictions(predictions, data.get("Test Set Labels"))
