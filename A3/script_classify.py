from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import utilities as util

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    classalgs = {'Random': algs.Classifier(),
                 #'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 #'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 #'Linear Regression': algs.LinearRegressionClass({'regularizer': 'l2', 'regwgt': 0.0}),
                 #'Logistic Regression': algs.LogitReg({'regularizer': 'l2', 'regwgt': 0.0}),
                'Neural Network': algs.NeuralNet({'epochs': 150, 'transfer': 'sigmoid', 'stepsize': 0.1, 'nh': 32}),
                 #'Kernel Logistic Regression': algs.KernelLogitReg({'kernel': 'None', 'num_centers': 10, 'regularizer': 'None', 'regwgt': 0.0}),
                 #'Linear Kernel Logistic Regression': algs.KernelLogitReg({'kernel': 'linear', 'num_centers': 100, 'regularizer': 'l2', 'regwgt': 0.0}),
                 #'Hamming Kernel Logistic Regression': algs.KernelLogitReg({'kernel': 'hamming', 'num_centers': 100, 'regularizer': 'l2', 'regwgt': 0.0})
                }
    numalgs = len(classalgs)

    parameters = (
        #{'regwgt': 0.0, 'nh': 4},
        #{'regwgt': 0.01, 'nh': 8},
        #{'regwgt': 0.05, 'nh': 16},
        #{'regwgt': 0.1, 'nh': 32},
        {'regwgt': 0.0, 'nh': 32}, #These are the best parameters for logistic regression and neural networks, respectively.
     )

    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        #trainset, testset = dtl.load_census(trainsize,testsize)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        best_standard_error = util.stdev(errors[learnername][0,:]) / math.sqrt(numruns)
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            standard_error = util.stdev(errors[learnername][p,:]) / math.sqrt(numruns)
            if aveerror < besterror:
                besterror = aveerror
                best_standard_error = standard_error
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
        print('Standard Error for ' + learnername + ': ' + str(standard_error))
