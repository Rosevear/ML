from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils


class Regressor(object):
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)
        self.cost_data = None

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            self.params = parameters
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        y = ytrain[:, np.newaxis]
        #self.weights = np.dot(np.dot(np.transpose(Xless), np.linalg.inv(np.dot(Xless, np.transpose(Xless))/numsamples) / numsamples), y) / numsamples
        #Solves with respect to w for the equation Xless * w = y: it computes the pseudo inverse, using singular values internally, for the matri Xlessx, avoiding the original singular matrix error.
        self.weights = np.linalg.lstsq(Xless, y)[0]

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = parameters
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples + (self.params['regwgt'] * np.identity(np.shape(Xless)[1]))), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def reset(self, parameters):
        Regressor.reset(self, parameters)


class LassoLinearRegression(Regressor):
    """
    Linear Regression with lasso regularization (l1 regularization)
    """
    def __init__( self, parameters={} ):
        self.params = parameters
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        self.weights = np.zeros((len(self.params['features']), 1))
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        y = ytrain[:, np.newaxis]

        cur_cost = float("inf")
        tolerance = 10 ** -5
        XTranspose = np.transpose(Xless)
        XX = np.dot(XTranspose, Xless) / numsamples
        Xy = np.dot(XTranspose, y) / numsamples
        step_size = 1 / (2 * np.linalg.norm(XX))
        new_cost = self.cost(self.weights, Xless, y, self.params['regwgt'], numsamples)

        while abs(new_cost - cur_cost) > tolerance:
            cur_cost = new_cost
            self.weights = self.weights - (step_size * np.dot(XX, self.weights)) + (step_size * Xy)
            self.weights = self.prox(self.weights, self.params['regwgt'], step_size)
            new_cost = self.cost(self.weights, Xless, y, self.params['regwgt'], numsamples)
        self.weights = np.ndarray.flatten(self.weights)

    def cost(self, weights, X, y, reg, num_samples):
        "Return the current cost given the weights, features, target values and regularization parameter"
        left_side = np.transpose(np.subtract(np.dot(X, weights), y))
        right_side = np.subtract(np.dot(X, weights), y)
        #Note: We don't need the np.linalg.norm function here, as the equation
        #used is already the sum of squared errors, as described on page 57 of the course notes
        #However, we still divide by the num samples and multiply by 1/2 to be consistent with MSE
        cost = 0.5 * (np.dot(left_side, right_side) / num_samples) + reg * np.linalg.norm(weights, 1)
        print("cost: " + str(cost))
        return cost

    def prox(self, weights, reg, step_size):
        "Proximal operator to handle non-smooth sections of the cost function"
        for i in range(len(weights)):
            if weights[i] > step_size * reg:
                weights[i] = weights[i] - (reg * step_size)
            elif abs(weights[i]) <= (step_size * reg):
                weights[i] = 0
            elif weights[i] < (-1 * step_size * reg):
                weights[i] = weights[i] + (step_size * reg)
        return weights

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def reset(self, parameters):
         Regressor.reset(self, parameters)

class StochasticLinearRegression(Regressor):
    """
    Linear Regression implemented via stochastic gradient descent
    """
    def __init__( self, parameters={} ):
        self.params = parameters
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.cost_data = []
        self.weights = np.zeros((len(self.params['features']), 1))
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        y = ytrain[:, np.newaxis]
        epochs = 1000

        for epoch in range(epochs):
            #Shuffle the data, making sure to maintain the proper correspondence between the features and targets
            data_set = np.append(Xless, y, axis=1)
            np.random.shuffle(data_set)
            Xless = data_set[:, 0:data_set.shape[1] - 1]
            y = data_set[:, -1, np.newaxis]
            for t in range(numsamples):
                gradient = np.dot(np.transpose(Xless[t, :][np.newaxis, :]), np.subtract(np.dot(Xless[t, :], self.weights), y[t, np.newaxis]))
                step_size = 0.01 / (epoch + 1)
                self.weights = self.weights - (step_size * gradient)
                cur_cost = self.cost(self.weights, Xless[t, :], y[t])
        #Format the array properly for the error function
        self.weights = np.ndarray.flatten(self.weights)

    def cost(self, weights, X, y):
        "Return the current cost given the weights, features, target values and regularization parameter"
        cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(X, weights), y)) ** 2))
        print("cost: " + str(cost))
        self.cost_data.append(cost)
        return cost

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def reset(self, parameters):
        Regressor.reset(self, parameters)

class BatchLinearRegression(Regressor):
    """
    Linear Regression implemented via batch gradient descent
    """

    def __init__( self, parameters={} ):
        self.params = parameters
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.cost_data = []
        self.weights = np.zeros((len(self.params['features']), 1))
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        y = ytrain[:, np.newaxis]

        cur_cost = float("inf")
        tolerance = 10 ** -6
        new_cost = self.cost(self.weights, Xless, y, self.params['regwgt'], numsamples)
        while abs(new_cost - cur_cost) > tolerance:
            cur_cost = new_cost
            gradient = np.dot(np.transpose(Xless), np.subtract(np.dot(Xless, self.weights), y)) / numsamples #+ (2 * self.params['regwgt'] * self.weights)
            step_size = self.line_search(self.weights, new_cost, gradient, Xless, y, self.params['regwgt'], numsamples)
            self.weights = self.weights - (step_size * gradient)
            new_cost = self.cost(self.weights, Xless, y, self.params['regwgt'], numsamples)

        #Format properly for the error function
        self.weights = np.ndarray.flatten(self.weights)

    def cost(self, weights, X, y, reg, num_samples):
        "Return the current cost given the weights, features, target values and regularization parameter"
        cost = 0.5 * ((np.linalg.norm(np.subtract(np.dot(X, weights), y)) ** 2) / num_samples)
        print("cost: " + str(cost))
        self.cost_data.append(cost)
        return cost

    def line_search(self, weights, cost, gradient, X, y, reg, numsamples):

        step_size_max = 0.05
        step_size_reducer = 0.1
        tolerance = 10e-4
        max_iterations = 100

        cur_step_size = step_size_max
        cur_weights = weights
        cur_obj = cost
        for i in range(max_iterations):
            cur_weights = weights - (cur_step_size * gradient)
            new_cost = self.cost(cur_weights, X, y, reg, numsamples)
            if new_cost < (cur_obj - tolerance):
                return cur_step_size
            else:
                cur_step_size = step_size_reducer * cur_step_size
                cur_obv = new_cost
        return 0

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def reset(self, parameters):
         Regressor.reset(self, parameters)
