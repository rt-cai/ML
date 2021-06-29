from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
from scipy import stats

class RandomTree(DecisionTree):
        
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]
        
        DecisionTree.fit(self, bootstrap_X, bootstrap_y)

class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        
    def fit(self,X,y):
        trees = []
        for i in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y)
            trees.append(tree)
        self.trees = trees
        
    def predict(self, X):
        predictions = []
        for i in range(self.num_trees):
            # [[results using first tree][]]
            predictions.append(self.trees[i].predict(X))
        return stats.mode(predictions, axis=0)[0]
        