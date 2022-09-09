# version 1.2

from abc import ABC, abstractmethod, abstractproperty
from math import exp, log

import numpy as np

##################################################################################################################
# ACTIVATION FUNCTIONS
##################################################################################################################

class Activation(ABC):
    '''
    An abstract class that implements an activation function
    '''

    @abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the activation function with input x
        :param x: input to activation function
        '''
        return x

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the activation function with input x
        :param x: input to activation function
        '''
        return x

class Identity(Activation):
    '''
    Implements the identity activation function (i.e. g(x) = x)
    '''

    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of applying the Identity activation function (i.e. returns the input)
        :param x: input to the activation function
        '''
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the identity function with input x (i.e. 1s)
        :param x: input to identity function
        '''
        return np.ones(x.shape)

class Sigmoid(Activation):
    '''
    Implements the sigmoid activation function
    :attr k: Parameter of the sigmoid function that controls its steepness around the origin
    '''

    def __init__(self, k: float=1.):
        '''
        :param k: Parameter of the sigmoid function that controls its steepness around the origin
        '''
        self.k = k
        super(Sigmoid, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the sigmoid function with input x
        :param x: input to sigmoid function
        '''

        n, d = x.shape
        value = np.ones((n, d))
        for i in range(n):
            for j in range(d):
                value[i][j] = 1.0 / (1.0 + exp(-x[i][j] * self.k))
        return value

    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the sigmoid function with input x
        :param x: input to sigmoid function
        '''
        n, d = x.shape
        value = np.ones((n, d))
        for i in range(n):
            for j in range(d):
                value[i][j] = self.k * exp(-x[i][j] * self.k) / (1 + exp(-x[i][j] * self.k)) ** 2
        return value

class Tanh(Activation):
    '''
    Implements the hyperbolic tangent activation function
    '''

    def __init__(self):
        super(Tanh, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the hyperbolic tangent function with input x
        :param x: input to activation function
        '''
        n, d = x.shape
        value = np.ones((n, d))
        for i in range(n):
            for j in range(d):
                value[i][j] = (exp(x[i][j]) - exp(-x[i][j])) / (exp(x[i][j]) + exp(-x[i][j]))
        return value

    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the hyperbolic tangent function with input x
        :param x: input to hyperbolic tangent function
        '''
        n, d = x.shape
        value = np.ones((n, d))
        for i in range(n):
            for j in range(d):
                value[i][j] = 4 / (exp(x[i][j]) + exp(-x[i][j])) ** 2
        return value

class ReLU(Activation):
    '''
    Implements the rectified linear unit activation function
    '''

    def __init__(self):
        super(ReLU, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the ReLU function with input x
        :param x: input to ReLU function
        '''
        # n, d = x.shape
        # value = np.ones((n, d))
        # for i in range(n):
        #     for j in range(d):
        #         value[i][j] = max(0, x[i][j])
        # return value
        return x * (x > 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the ReLU function with input x
        Set the derivative to 0 at x=0.
        :param x: input to ReLU function
        '''
        # n, d = x.shape
        # value = np.ones((n, d))
        # for i in range(n):
        #     for j in range(d):
        #         value[i][j] = 1 if x[i][j] > 0 else 0
        # return value
        return 1 * (x > 0)

class LeakyReLU(Activation):
    '''
    Implements the leaky rectified linear unit activation function
    :attr k: Parameter of leaky ReLU function corresponding to its slope in the negative domain
    '''

    def __init__(self, k=0.1):
        '''
        :param k: Parameter of leaky ReLU function corresponding to its slope in the negative domain
        '''
        self.k = k
        super(LeakyReLU, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the Leaky ReLU function with input x
        :param x: input to Leaky ReLU function
        '''
        # n, d = x.shape
        # value = np.ones((n, d))
        # for i in range(n):
        #     for j in range(d):
        #         value[i][j] = x[i][j] if x[i][j] > 0 else self.k * x[i][j]
        # return value
        return x * (x > 0) + self.k * x * (x < 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        '''
        Returns the result of evaluating the derivative of the leaky ReLU function with input x
        Set the derivative to k at x=0.
        :param x: input to leaky ReLU function
        '''
        # n, d = x.shape
        # value = np.ones((n, d))
        # for i in range(n):
        #     for j in range(d):
        #         value[i][j] = 1 if x[i][j] > 0 else self.k
        # return value
        return 1 * (x > 0) + self.k * (x < 0)

##################################################################################################################
# LOSS FUNCTIONS
##################################################################################################################

class Loss(ABC):
    '''
    Abstract class for a loss function
    '''

    @abstractmethod
    def value(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the value of the loss function for n provided predictions and targets, averaged across n examples
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        return y_hat

    @abstractmethod
    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the derivative of the loss function with respect to the predictions, averaged across n examples
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        return y_hat

class CrossEntropy(Loss):
    '''
    Implements the binary cross entropy loss function
    '''

    def value(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the binary cross entropy loss function for n predictions and targets, averaged across n examples
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        n, _ = y_hat.shape
        value = 0
        for i in range(n):
            p = y[i]
            p_hat = y_hat[i]
            value += - p * log(p_hat, exp(1)) - (1 - p) * log(1 - p_hat, exp(1))
        return value / n

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the derivative of the binary cross entropy loss function with respect to the predictions,
        averaged across n examples
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        n, d = y_hat.shape
        value = np.zeros((n, d))
        for i in range(n):
            p = y[i]
            p_hat = y_hat[i]
            value[i] = - p / p_hat + (1-p) / (1-p_hat)
        return value / n


class MeanSquaredError(Loss):
    '''
    Implements the mean squared error loss function
    '''

    def value(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the mean squared error loss function for n predictions and targets, averaged across n examples
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        n, _ = y.shape
        sum = 0 
        for i in range(n):
            sum += (y_hat[i] - y[i]) ** 2

        return sum / n

        # n, _ = y.shape
        # return np.sum((y_hat - y) ** 2 / n)

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Computes the derivative of the mean squared error loss function with respect to the predictions
        :param y_hat: Neural network predictions, with shape (n, 1)
        :param y: Targets, with shape (n, 1)
        '''
        n, _ = y.shape

        return 2 * (y_hat - y) / n


##################################################################################################################
# METRICS
##################################################################################################################

def accuracy(y_hat: np.ndarray, y: np.ndarray, classification_threshold=0.5) -> float:
    '''
    Computes the accuracy of predictions, given the targets. Assumes binary classification (i.e. targets are either 0
    or 1). The predicted class of an example is 1 if the predicted probability is greater than or equal to the
    classification threshold, and 0 otherwise.
    :param y_hat: Neural network predictions, with shape (n, 1). Note that these are probabilities.
    :param y: Targets, with shape (n, 1)
    :param classification_threshold: Classification threshold for binary classification
    '''
    n, _ = y.shape
    count = 0 
    for i in range(n):
        if y_hat[i] >= classification_threshold and y[i] == 1:
            count += 1
        elif y_hat[i] < classification_threshold and y[i] == 0:
            count += 1

    return count / n

def mean_absolute_error(y_hat: np.ndarray, y: np.ndarray) -> float:
    '''
    Computes the mean absolute error between the predictions and the targets. This metric is useful for regression
    problems.
    :param y_hat: Neural network predictions, with shape (n, 1). These should be real numbers.
    :param y: Targets, with shape (n, 1). These should be real numbers.
    '''
    n, _ = y.shape
    sum = 0 
    for i in range(n):
        sum += abs(y_hat[i] - y[i])

    return sum / n