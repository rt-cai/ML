import numpy as np
from scipy.optimize.optimize import approx_fprime

"""
Implementation of function objects.
Function objects encapsulate the behaviour of an objective function that we optimize.
Simply put, implement evaluate(w, X, y) to get the numerical values corresponding to:
f, the function value (scalar) and
g, the gradient (vector).

Function objects are used with optimizers to navigate the parameter space and
to find the optimal parameters (vector). See optimizers.py.
"""

class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        IMPORTANT: w is assumed to be a 1d-array, hence shaping will have to be handled.
        """
        raise NotImplementedError

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(w, lambda w: self.evaluate(w, X, y)[0], epsilon=1e-6)
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient
        if np.max(np.abs(difference) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')

class FunObjLeastSquares(FunObj):
    
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """

        # Prediction is linear combination
        y_hat = X@w
        # Residual is difference between prediction and ground truth
        residuals = y_hat - y
        # Squared residuals gives us the objective function value
        f = 0.5 * np.sum(residuals ** 2)
        # Analytical gradient, written in mathematical form first
        # and then translated into Python
        g = X.T@X@w - X.T@y
        return f, g

class FunObjRobustRegression(FunObj):
        
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """

        n, d = X.shape

        # Calculate the function value
        f = 0
        for i in range(n):
            # Tip: when you have two terms, it's useful to call them "left" and "right".
            # Believe or not, having two terms show up in your functions is extremely common.
            left = np.exp(w@X[i,:] - y[i])
            right = np.exp(y[i] - w@X[i,:])
            f += np.log(left + right)

        # Calculate the gradient value
        r = np.zeros(n)
        for i in range(n):
            left = np.exp(w@X[i,:] - y[i])
            right = np.exp(y[i] - w@X[i,:])
            r[i] = (left - right) / (left + right)
        g = X.T@r

        return f, g

class FunObjLogReg(FunObj):

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """ 
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res
    
        return f, g

class FunObjLogRegL2(FunObj):

    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of L2-regularized logistics regression objective.
        """ 

        """YOUR CODE HERE FOR Q2.1"""
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply
        w2 = np.sum(w*w)  # ||w||^2 

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy/2*w2

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res + self.lammy*w
        
        return f, g

        
class FunObjLogRegL0(FunObj):

    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression objective.
        """ 
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy * len(w)
        
        # We cannot differentiate the "length" function
        g = None
        return f, g

class FunObjSoftmax(FunObj):

    def evaluate(self, w, X, y):
        n, d = X.shape
        k = len(np.unique(y))
        W = w.reshape([k, d])

        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you will want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.
        f = 0
        for i in range(n):
            wx = -W[y[i]]@X[i]
            lg = np.log(np.sum(np.exp(W@X[i].T),axis=0))
            f = f + wx + lg

        g = np.zeros((k, d))
        for c in range(k):
            # I(yi=c)
            I = np.zeros(n)
            i = np.where(y==c)
            I[i] = 1
        
            for i in range(n):
                # p(yi=c|w,xi)
                p = np.exp(W[c]@X[i])/np.sum(np.exp(W@X[i]))
                g[c]=g[c]+X[i]*(p-I[i])
        g = g.flatten()
        
        return f, g
       