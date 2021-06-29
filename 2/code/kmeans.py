import numpy as np
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        #add: record iter #
        iter=0
        
        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                if np.any(y==kk): # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y==kk].mean(axis=0)

            
            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break
            self.means = means #new add
            print('Iter #: %d'%iter)
            obj_error=self.error(X)
            print('Objective error: %.3f' % obj_error)
            iter +=1
        self.means = means
        self.obj_error =obj_error

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)
    

    def error(self,X):
        means=self.means
        #calcute (x_i-w_j)^2
        dist = euclidean_dist_squared(X, means)
        dist[np.isnan(dist)] = np.inf
        #find the closed mean to xi, axis=1
        dist_sort=np.sort(dist, axis=1)
        #sum over cloest mean, axis=0
        error=np.sum(dist_sort,axis=0)[0]
        return error
        
        
        