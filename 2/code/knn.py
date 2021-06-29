"""
Implementation of k-nearest neighbours classifier
"""

import utils
import numpy as np

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, X_hat):
        # [[X to X_hat_1][X to X_hat_2]]
        dists = utils.euclidean_dist_squared(self.X, X_hat)
        # sort down, [[smallest][second smallest]]
        sort = np.argsort(dists,axis=0)
        # [[X_hat_1 nearest to X_i, X_hat_2 nearest to X_i]]
        nearest = sort[0:self.k,:]
        n, d = X_hat.shape
        y_hat = np.zeros(n)
        for i, y in enumerate(y_hat):
            y_hat[i] = utils.mode(self.y[nearest[:,i]])
        return y_hat