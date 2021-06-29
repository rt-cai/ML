# basics
import argparse
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch
from fun_obj import FunObjLeastSquares, FunObjRobustRegression
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# our code
import linear_models
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "2":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_models.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "2.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # TODO: Finish WeightedLeastSquares in linear_models.py
        model = linear_models.WeightedLeastSquares()
        
        """YOUR CODE HERE FOR Q2.1"""
        v = np.ones(500)
        v[-100:] = 0.1 
        v = np.diag(v)
        
        model = linear_models.WeightedLeastSquares()
        model.fit(X,y,v)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="2.1.pdf")

    elif question == "2.4":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        fun_obj = FunObjLeastSquares()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")



    elif question == "2.4.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        """YOUR CODE HERE FOR Q2.4.1"""
        # TODO: Finish FunObjRobustRegression in fun_obj.py.
        fun_obj = FunObjRobustRegression()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        model.fit(X,y)
        print(model.w)
        
        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="2.4.1.pdf")

    elif question == "2.4.2":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        """YOUR CODE HERE FOR Q2.4.2"""
        descent_errors = np.zeros(100)
        line_errors = np.zeros(100)
        for i in range(1, 101):
            fun_obj = FunObjRobustRegression()
            optimizer = OptimizerGradientDescent(fun_obj, X, y, max_evals=i, verbose=False)
            model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
            model.fit(X,y)
            yhat = model.predict(X)
            descent_errors[i-1] = np.mean((yhat - y)**2)
            
            fun_obj = FunObjRobustRegression()
            optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=i, verbose=False)
            model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
            model.fit(X,y)
            yhat = model.predict(X)
            line_errors[i-1] = np.mean((yhat - y)**2)
            
        num_evals = np.arange(1, 101)
        plt.plot(num_evals, descent_errors, label="OptimizerGradientDescent")
        plt.plot(num_evals, line_errors, label="OptimizerGradientDescentLineSearch")
        
        plt.xlabel("Number of Iteration")
        plt.ylabel("Trainging Error")
        plt.legend()
        fname = os.path.join("..", "figs", "2.4.1.pdf")
        plt.savefig(fname)
            

    elif question == "3":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        # Fit least-squares model
        model = linear_models.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,X_test,y_test,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "3.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        # TODO: Finish LeastSquaresBias in linear_models.py.
        model = linear_models.LeastSquaresBias()
        model.fit(X,y)

        """YOUR CODE HERE FOR Q3.1"""
        model = linear_models.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,X_test,y_test,title="Least Squares with bias",filename="3.1.pdf")

    elif question == "3.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        trainError = np.zeros(101)
        testError = np.zeros(101)

        for p in range(101):
            print("p={:d}".format(p))

            # TODO: Finish LeastSquaresPoly in linear_models.py
            model = linear_models.LeastSquaresPoly(p)
            model.fit(X, y)

            yhat = model.predict(X)
            trainError[p] = np.mean((yhat - y)**2)
            
            if X_test is not None and y_test is not None:
                yhat = model.predict(X_test)
                testError[p] = np.mean((yhat - y_test)**2)
                
        p_value = np.arange(0, 101)
        plt.plot(p_value, trainError, label="trainError")
        plt.plot(p_value, testError, label="testError")
        
        plt.xlabel("p")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "3.2.pdf")
        plt.savefig(fname)
        
        print("Training Error: %s" % trainError)
        print("Test Error: %s" % testError)

    else:
        print("Unknown question: %s" % question)

