import argparse
from fun_obj import FunObjLogReg, FunObjLogRegL0, FunObjLogRegL2, FunObjSoftmax
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch, OptimizerGradientDescentLineSearchProximalL1
import numpy as np
import matplotlib.pyplot as plt
import utils
import linear_models
import os
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=400, verbose=True)
        model = linear_models.LogRegClassifier(fun_obj, optimizer)
        model.fit(X,y)

        print("LogReg Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogReg Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("# nonZeros: {:d}".format((model.w != 0).sum()))
        print("# iterations: {:d}".format(len(model.fs)))

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        # TODO: Complete FunObjLogRegL2.evaluate()
        fun_obj = FunObjLogRegL2(1)
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=400, verbose=False)
        model = linear_models.LogRegClassifier(fun_obj, optimizer)
        model.fit(X,y)

        print("LogRegL2 Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogRegL2 Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("# nonZeros: {:d}".format((model.w != 0).sum()))
        print("# iterations: {:d}".format(len(model.fs)))
        
    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        """YOUR CODE HERE FOR Q2.2"""
        lammys = [0.01, 0.1, 1, 10]
        for l in lammys:
            fun_obj = FunObjLogReg()
            optimizer = OptimizerGradientDescentLineSearchProximalL1(l, fun_obj, X, y, max_evals=400, verbose=False)
            model = linear_models.LogRegClassifier(fun_obj, optimizer)
            model.fit(X,y)
            
            print("lambda = %f:"%l)
            print("LogRegL1 Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
            print("LogRegL1 Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
            print("# nonZeros: {:d}".format((model.w != 0).sum()))
            print("# iterations: {:d}".format(len(model.fs)))
        

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        # TODO: Complete LogRegClassifierForwardSelection.fit()
        local_fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(local_fun_obj, X, y, max_evals=400, verbose=False)
        global_fun_obj = FunObjLogRegL0(1)
        model = linear_models.LogRegClassifierForwardSelection(global_fun_obj, optimizer)
        model.fit(X,y)

        print("LogReg Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogReg Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("# nonZeros: {:d}".format((model.w != 0).sum()))
        
    elif question == "2.5":
        w = np.arange(-1., 3., 0.1)
        
        f = 1/2*((w-2)**2+1)+abs(w)**(1.0/2)
        minw = -1+0.1*np.argmin(f)
        
        print(minw)
        plt.figure()
        plt.plot(w, f)
        plt.xlabel("w")
        plt.ylabel("f(w)")
        fname = os.path.join("..", "figs", "2.5.4.pdf")
        plt.savefig(fname)
        
        
        
        f = 1/2*((w-2)**2+1)+10*abs(w)**(1.0/2)
        minw = -1+0.1*np.argmin(f)
        
        print(minw)
        plt.figure()
        plt.plot(w, f)
        plt.xlabel("w")
        plt.ylabel("f(w)")
        fname = os.path.join("..", "figs", "2.5.5.pdf")
        plt.savefig(fname)
        

    elif question == "3":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        model = linear_models.LeastSquaresClassifier()
        model.fit(X, y)

        print("LeastSquaresClassifier Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LeastSquaresClassifier Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))

        print(np.unique(model.predict(X)))


    elif question == "3.2":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        # TODO: Complete LogRegClassifierOneVsAll.fit()
        fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=500, verbose=False)
        model = linear_models.LogRegClassifierOneVsAll(fun_obj, optimizer)
        model.fit(X, y)

        print("LogRegClassifierOneVsAll Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogRegClassifierOneVsAll Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print(np.unique(model.predict(X)))

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        # TODO: Complete FunObjSoftmax.evaluate()
        # TODO: Complete MulticlassLogRegClassifier.fit()
        # TODO: Complete MulticlassLogRegClassifier.predict()
        fun_obj = FunObjSoftmax()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=500, verbose=True)
        model = linear_models.MulticlassLogRegClassifier(fun_obj, optimizer)
        model.fit(X, y)

        print("Softmax Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("Softmax Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))

    elif question == "3.5":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        """YOUR CODE HERE FOR Q3.5"""
        model=LogisticRegression(multi_class='ovr', C=9999, fit_intercept=False)
        model.fit(X, y)
        
        print("one-vs-all Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("one-vs-all Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("one-vs-all # nonZeros: %d" % (model.coef_ != 0).sum())
              
              
        model=LogisticRegression(multi_class='multinomial', C=9999,fit_intercept=False)
        model.fit(X, y)
        
        print("Softmax Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("Softmax Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("Softmax # nonZeros: %d" % (model.coef_ != 0).sum())
