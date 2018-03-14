python version: 2.7.12
numpy version: 1.11.0

NOTE: The features and number of classes are hard coded attributes of
most classes, except for NaiveBayes, because we switch between using and not
using the Ones column.

Question notes and answers
1 a) Adding the column of ones makes little to no difference to the final result.
This is a rather intuitive result, however, since the point of Naive Bayes is to
model the p(y|x) by modeling the p(x|y)p(y). Therefore, if we simply add the same
feature to every instance of every class, which is what adding a column of ones
does, this should not impact the parameter estimation since this feature will be
present throughout the entire data set, and won't provide any help in discriminating
between classes.

Also, note that my implementation did not require making any modifications after
adding the ones column to get it to run again, so I'm not sure what the
implementation problem, if any, should have been.

See code for the implementation details.

b) See code for the implementation details.

c) See code for the implementation details.

d) As can be observed from the results below, the neural network had the best average
error, while random classification performed the worst.

Interestingly, however, linear regression performed marginally better than the other
classification methods, on average, although it also has a significantly larger
standard error compared to the other algorithms and so we should be less confident
that its results are truly indicative of a superior performance.

All results below were averaged over 5 runs
Best parameters for Random: {}
Average error for Random: 50.032 +- 0.334022753716
Standard Error for Random: 0.191728975379
Best parameters for Naive Bayes: {'usecolumnones': False}
Average error for Naive Bayes: 25.54 +- 0.301622280344
Standard Error for Naive Bayes: 0.337223961189
Best parameters for Linear Regression: {'regwgt': 0.0, 'regularizer': 'l2'}
Average error for Linear Regression: 24.924 +- 0.412733327949
Standard Error for Linear Regression: 1.130214139
Best parameters for Logistic Regression: {'regwgt': 0.0, 'regularizer': 'l2'}
Average error for Logistic Regression: 26.564 +- 0.446081606884
Standard Error for Logistic Regression: 0.506975344568
Best parameters for Neural Network: {'nh': 32, 'transfer': 'sigmoid', 'epochs': 150, 'stepsize': 0.1}
Average error for Neural Network: 22.668 +- 0.38581238964
Standard Error for Neural Network: 0.431351364899
Best parameters for Naive Bayes Ones: {'usecolumnones': True}
Average error for Naive Bayes Ones: 25.564 +- 0.27870557942
Standard Error for Naive Bayes Ones: 0.311602310646

2 a) NOTE: These results took very long to run, so be aware of this if you choose to run...
Linear kernel logistic regression displayed a much higher degree of variance in
terms of the results delivered. On the one hand it performed similarly well to they
algorithms mentioned earlier, on a different it performed markedly worse.
This may be due to the variance involved in selecting the centers used, different
runs may have yielded different training sets randomly selected from the data,
which had a significant impact on the quality of the learned model.

Both sets of results below were averaged over 5 runs.
First set of runs:
Best parameters for Random: {}
Average error for Random: 50.344 +- 0.278188425352
Standard Error for Random: 0.311024114821
Best parameters for Linear Kernel Logistic Regression: {'kernel': 'linear', 'regwgt': 0.01, 'num_centers': 100, 'regularizer': 'l2'}
Average error for Linear Kernel Logistic Regression: 46.496 +- 0.202733322372
Standard Error for Linear Kernel Logistic Regression: 0.226662745064

Second set of runs
Average error for Random: 50.296 +- 0.414898541815
Standard Error for Random: 0.463870671632
Best parameters for Linear Kernel Logistic Regression: {'kernel': 'linear', 'regwgt': 0.01, 'num_centers': 100, 'regularizer': 'l2'}
Average error for Linear Kernel Logistic Regression: 24.184 +- 0.267096986131
Standard Error for Linear Kernel Logistic Regression: 0.298623508787

b) NOTE: These results took very long to run, so be aware of this if you choose to run...
Also, the data set needs to be switched to the census one by uncommenting it and commenting out the susysubset data set.
As can be observed below, the hamming kernel significantly outperformed the random
predictor, both in terms of average error ad standard error. The margins of errors
for both statistics are also significantly smaller.

The results below are averaged over 5 runs.
Best parameters for Hamming Kernel Logistic Regression: {'kernel': 'hamming', 'regwgt': 0.01, 'num_centers': 100, 'regularizer': 'l2'}
Average error for Hamming Kernel Logistic Regression: 16.464 +- 0.174667684475
Standard Error for Hamming Kernel Logistic Regression: 0.19528440798
Best parameters for Random: {}
Average error for Random: 50.3 +- 0.268208873828
Standard Error for Random: 0.299866637024
