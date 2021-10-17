import numpy as np
import math
import variable as Variable

def LogisticRegression():
    def __init__(self, learning_rate = 0.01, epochs = 10000, verbose = True):
        self.lr = learning_rate
        self.verbose = verbose
        self.iters = epochs
        # self.w_iter = np.zeros(dim, 1)
        # self.b_iter = 0
        # self.w_iter_vals = {"w_iter_{i}":self.w_iter[i] for i in range(dim)}
        # self.b_iter_vals = {"b_iter":self.b_iter}
    
    def fit(self, X, Y):
        # the code in this method is simply following the logistic regression method outlined here: https://www.youtube.com/watch?v=t6MVuMavbBY
        m, n = np.shape(X)
        w = [Variable(name = "w_{i}") for i in range(n)]
        b = Variable(name = "b")
        y_hat = []
        for i in range(m):
            y_hat.append(1/(1+Variable.exp(sum((np.array(w) * X[i]) + b))))
        cost = (0 - 1) * sum([Y[i] * Variable.log(y_hat[i]) + (1 - Y[i]) * Variable.log(1 - y_hat[i]) for i in range(m)])
        self.w_iter = np.zeros(self.n, 1)
        self.b_iter = 0

        cost_list = []
        for i in range(self.iters):
        #    all_vars_coeffs = self.w_iter_vals
        #    all_vars_coeffs.update(self.b_iter_vals)
           all_vars_coeffs = {"w_iter_{i}":self.w_iter[i] for i in range(n)}
           all_vars_coeffs.update({"b_iter":self.b_iter})
           # gradient descent
           grad = cost.gradient(all_vars_coeffs)
           self.w_iter = self.w_iter - self.lr * grad[0: len(impr)-1]
           self.b_iter = self.b_iter - self.lr * grad[-1]
           # updating cost
           cost_list.append(cost.evaluate(all_vars_coeffs))
           cost_updated_val = cost.evaluate(all_vars_coeffs)
           if self.verbose:
               print("cost after ", i, "iterations is: ", cost_updated_val)
            
        # return sum(cost_list)/len(cost_list)

    def predict(self, X):
        m, n = np.shape(X)
        vals = []
        for i in range(m):
            pred = 1/(1+np.exp(sum((np.array(self.w_iter) * X[i]) + self.b_iter)))
            vals.append(pred)
        return vals