from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs
import utilities as util

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def l2err(prediction,ytest):
    """ l2 error  """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return 0.5 * l2err(predictions, ytest) ** 2 / ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 5

    regressionalgs = {#'Random': algs.Regressor(),
                #'Range': algs.RangePredictor(),
                #'Mean': algs.MeanPredictor(),
                'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(),
                'LassoLinearRegression': algs.LassoLinearRegression(),
                'StochasticLinearRegression': algs.StochasticLinearRegression(),
                'BatchLinearRegression': algs.BatchLinearRegression(),
             }
    numalgs = len(regressionalgs)

    #The assignment asks for only one specific regualrization condition, so we can comment out the others
    parameters = (
         #{'regwgt': 0.00, 'features': range(1, 6)},
         #{'regwgt': 0.00, 'features': range(50)},
         #{'regwgt': 0.00, 'features': range(385)},
         {'regwgt': 0.01, 'features': range(385)},
        #   {'regwgt': 0.10, 'features': range(385)},
        #   {'regwgt': 0.25, 'features': range(385)},
        #   {'regwgt': 0.50, 'features': range(385)},
        #   {'regwgt': 0.75, 'features': range(385)},
        #   {'regwgt': 1.0, 'features': range(385)},
          )

    numparams = len(parameters)

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams, numruns))

    #To hold the cost per iteration on each run, to be used for graphing later
    SGD_cost_data = []
    BGD_cost_data = []

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize, testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                #learner.params = params
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])

                #Save the training error for gradient descent, for graphing later
                if learnername == "StochasticLinearRegression":
                    SGD_cost_data.append(learner.cost_data)
                elif learnername == "BatchLinearRegression":
                    BGD_cost_data.append(learner.cost_data)

                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(predictions, testset[1])
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
    print("\n")
    for learnername in regressionalgs:
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
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print("Test set average errors:")
        print ('Average Error for ' + learnername + ': ' + str(besterror))
        print('Standard Error for ' + learnername + ': ' + str(standard_error))

    #Average the error per iteration over 5 runs up to the BGD run with the shortest number of iterations
    num_iterations = min([len(run) for run in BGD_cost_data])
    SGD_avg_cost_data = [sum(run)/len(run) for run in zip(*SGD_cost_data)]
    BGD_avg_cost_data = [sum(run)/len(run) for run in zip(*BGD_cost_data)]

    print "\nPlotting the averaged cost over iterations for SGD and BGD"
    plt.ylabel('Error')
    plt.xlabel("Num Iterations")
    plt.axis([0, num_iterations, 0, 1000])
    plt.plot(SGD_avg_cost_data, 'r-', label="SGD")
    plt.plot(BGD_avg_cost_data, 'b-', label="BGD")
    plt.legend(loc='center', bbox_to_anchor=(0.6,0.25))
    plt.show()
    print "\nFinished!"
