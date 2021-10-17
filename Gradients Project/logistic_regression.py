import numpy as np
import math
from variable import Variable
class LogisticRegression():
    def __init__(self, learning_rate = 0.01, epochs = 100, verbose = True):
        self.lr = learning_rate
        self.verbose = verbose
        self.iters = epochs
        pass

    def fit(self, X, y, threshold):
        m, n = np.shape(X)
        w = [Variable(name = f"w_{i}") for i in range(n)]
        b = Variable(name='b')
        yh = [(1/(1 + Variable.exp(sum(np.array(w) * X[i]) + b))) for i in range(m)]
        cost = (0 - 1) * sum([y[i] * Variable.log(yh[i]) + (1 - y[i]) * Variable.log(1 - yh[i]) for i in range(m)])

        self.w_iter = np.zeros(n)
        self.b_iter = 0
        
        cost_list = []
        for wq in range(self.iters):
            dict_values = {'b':self.b_iter}
            dict_values.update({f'w_{i}':self.w_iter[i] for i in range(n)})

            step = cost.gradient(dict_values)

            self.b_iter -= self.lr * cost.gradient(dict_values)[0]
            self.w_iter -= self.lr * cost.gradient(dict_values)[1:]

            loss = cost.evaluate(dict_values)
            
            if self.verbose:
                print("cost after ", wq, "iterations is - ", loss)
        print(f"model summary: cost - {loss}", f"weights - {self.w_iter}",f"bias - {self.b_iter}")


    def predict(self, X):
        m, n = np.shape(X)
        predictions = []

        for i in range(m):
            predictions.append(1/(1+math.exp(sum(np.array(self.w_iter)*dataset[i]) + self.b_iter)))

        return predictions