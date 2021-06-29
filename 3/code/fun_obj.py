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
        """
        raise NotImplementedError

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(w.flatten(), lambda w: self.evaluate(w.reshape((d,1)),X,y)[0], epsilon=1e-6)
        implemented_gradient = self.evaluate(w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
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

        """YOUR CODE HERE FOR Q2.3"""
        f = np.sum(np.log(np.exp(X@w - y)+np.exp(y-X@w)))
        g = X.T@((((np.exp(X@w - y)-np.exp(y-X@w))))/(np.exp(X@w - y)+np.exp(y-X@w)))
        return f, g
        