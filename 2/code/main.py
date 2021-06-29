# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports
from sklearn.tree import DecisionTreeClassifier

# our code
from naive_bayes import NaiveBayes
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
import utils
from sklearn.neighbors import KNeighborsClassifier
from random_tree import RandomForest

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)

        y_hat = model.predict(X_test)
        err_test = np.mean(y_hat != y_test)
        print("Training error: {:.3f}".format(err_train))
        print("Testing error: {:.3f}".format(err_test))

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        """YOUR CODE HERE FOR Q1.1"""
        err_train = np.zeros(15)
        err_test = np.zeros(15)
        for depth in range(1,16):
            model = DecisionTree(max_depth=depth,stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_hat = model.predict(X)
            err_train[depth-1] = np.mean(y_hat != y)
            
            y_hat = model.predict(X_test)
            err_test[depth-1] = np.mean(y_hat != y_test)
        depths = np.arange(1,16)
        plt.plot(depths, err_train, label="err_train")
        plt.plot(depths, err_test, label="err_test")

        plt.xlabel("Depth of tree")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_1_tree_errors.pdf")
        plt.savefig(fname)


    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        """YOUR CODE HERE FOR Q1.2"""
        X_first,X_second = np.split(X,2)
        y_first,y_second = np.split(y,2)
        best_depth_first=0
        best_error_first = 1
        best_depth_second=0
        best_error_second = 1
        for depth in range(1,16):
            # first half train, second half validation
            model = DecisionTree(max_depth=depth,stump_class=DecisionStumpInfoGain)
            model.fit(X_first, y_first)
            y_hat = model.predict(X_second)
            error = np.mean(y_hat != y_second)
            if(error < best_error_first):
                best_error_first = error
                best_depth_first = depth
                
            
            # second half train, first half validation
            model = DecisionTree(max_depth=depth,stump_class=DecisionStumpInfoGain)
            model.fit(X_second, y_second)
            y_hat = model.predict(X_first)
            error = np.mean(y_hat != y_first)
            if(error < best_error_second):
                best_error_second = error
                best_depth_second = depth
        print("first: %d"%best_depth_first)
        print("first: %f"%best_error_first)
        print("second: %d"%best_depth_second)
        print("first: %f"%best_error_second)

    
    elif question == '1.3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        n, d = X.shape

        """YOUR CODE HERE FOR Q1.3"""
        X_first,X_second = np.split(X, [15])
        y_first,y_second = np.split(y, [15])
        best_depth_first = 0
        best_error_first = 1
        best_depth_second=0
        best_error_second = 1
        for depth in range(1,16):
            model = DecisionTree(max_depth=depth,stump_class=DecisionStumpInfoGain)
            model.fit(X_second, y_second)
            y_hat = model.predict(X_first)
            error = np.mean(y_hat != y_first)
            if(error < best_error_first):
                best_error_first = error
                best_depth_first = depth
                
            model = DecisionTree(max_depth=depth,stump_class=DecisionStumpInfoGain)
            model.fit(X_first, y_first)
            y_hat = model.predict(X_second)
            error = np.mean(y_hat != y_second)
            if(error < best_error_second):
                best_error_second = error
                best_depth_second = depth
        print("first depth: %f"%best_depth_first)
        print("first error: %f"%best_error_first)
        print("second depth: %f"%best_depth_second)
        print("second error: %f"%best_error_second)



    elif question == '2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        """YOUR CODE HERE FOR Q2"""
        for i in [1, 3, 10]:
            Etest = np.array([])
            Etrain = np.array([])
            model = KNN(k = i)
            model.fit(X, y)
            # training error
            y_train = model.predict(X)
            Etrain = np.append(Etrain, np.mean(y_train != y))

            # testing error
            y_hat = model.predict(X_test)
            Etest = np.append(Etest, np.mean(y_hat != y_test))
            print("The training error of ", i, "-NN is", Etrain)
            print("The testing error of ", i, "-NN is", Etest)

    elif question == '3.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        """YOUR CODE HERE FOR Q3.2"""
        print(wordlist[40])
        print(wordlist[1 == X[400]])
        print(groupnames[y[400]])

    elif question == '3.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
 
        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)
        print("Naive Bayes (ours) training error: {:.3f}".format(err_train))
        
        y_hat = model.predict(X_valid)
        err_valid = np.mean(y_hat != y_valid)
        print("Naive Bayes (ours) validation error: {:.3f}".format(err_valid))

    elif question == '3.4':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
 
        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        
        """YOUR CODE HERE FOR Q3.4"""
            

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

        """YOUR CODE FOR Q4"""
        print("Random Forest")
        evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))

    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic_rerun.png")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        
        """YOUR CODE HERE FOR Q5.1"""
        min_error = 1
        miny = None
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            y = model.predict(X)
            error = model.error
            if(error < min_error):
                min_error = error
                miny = y
        plt.scatter(X[:,0], X[:,1], c=miny, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic_rerun.png")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))
        print("Error: %f" % min_error)

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']

        """YOUR CODE HERE FOR Q5.2"""
        err = np.zeros(10)
        for k in range(1, 11):
            min_error = 1
            for i in range(50):
                model = Kmeans(k=k)
                model.fit(X)
                y = model.predict(X)
                error = model.obj_error
                if(error < min_error):
                    min_error = error
            err[k-1] = min_error
            
        plt.plot(np.arange(1, 11), err)
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "5.2.pdf")
        plt.savefig(fname)

    else:
        print("Unknown question: {:s}".format(question))
