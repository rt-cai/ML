import numpy as np
from numpy.linalg import solve

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,v):
        """YOUR CODE HERE FOR Q2.1"""
        self.w=solve(X.T@v@X,X.T@v@y)

class LinearModelGradientDescent(LeastSquares):
    """
    Generic linear model optimizing custom function objects.
    A combination of: 
    (1) optimizer and 
    (2) function object 
    prescribes the behaviour of the parameters, although prediction is 
    always performed exactly the same: y_hat = X @ w.

    See optimizers.py for optimizers.
    See fun_obj.py for function objects, which must implement evaluate()
    and return f and g values corresponding to current parameters.
    """

    def __init__(self, fun_obj, optimizer):
        self.fun_obj = fun_obj
        self.optimizer = optimizer

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        w = np.zeros((d, 1))
        self.optimizer.set_parameters(w)

        # Correctness check for input function object
        self.fun_obj.check_correctness(w, X, y)

        # Use gradient descent to optimize w
        """YOUR CODE HERE FOR Q2.4.2"""
        # TODO: Collect training information so you can visualize it.
        # Hint: should we have a return statement for fit()?
        for i in range(100):
            f, g, w, break_yes = self.optimizer.step()
            if break_yes:
                break
        self.w = w

# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        """YOUR CODE HERE FOR Q3.1"""
        Z = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q3.1"""
        Z = np.concatenate((X_hat, np.ones((X_hat.shape[0], 1))), axis=1)
        return Z@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        """YOUR CODE HERE FOR Q3.2"""
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        """YOUR CODE HERE FOR Q3.2"""
        Z = self.__polyBasis(X)
        return Z@self.w


    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.transpose(X.T)
        Z = np.power(Z, np.arange(self.p+1))
        return Z

