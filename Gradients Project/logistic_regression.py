import numpy as np
import math
from variable import Variable

class LogisticRegression():
    def __init__(self, learning_rate = 0.001, epochs = 50, verbose = False):
        self.lr = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        pass
    
    def sigmoid(self, w, val, b, dims):
        act_func = []
        for i in range(dims):
            lin_comb = sum(c * x for c, x in zip(w, val[i]))
            act_func.append((1 / (1 + Variable.exp((lin_comb) + b))))
        return act_func
    
    def loss_function(self, pred, true, dims):
        neg_avg = ((0 - 1) / dims)
        return neg_avg * sum([true[i] * Variable.log(pred[i]) + (1 - true[i]) * Variable.log(1 - pred[i]) for i in range(dims)])

    def fit(self, X, y):
        m = np.shape(X)[0]
        n = np.shape(X)[1]
        w = [Variable(name = i) for i in range(n)]
        b = Variable(name = "bias")
        h = self.sigmoid(w, X, b, m)
        cost = self.loss_function(h, y, m)
        self.w_iter = np.zeros(n)
        self.b_iter = 0
        
        cost_list = []
        for wq in range(self.epochs):
            all_vars_coeffs = {i : self.w_iter[i] for i in range(n)}
            all_vars_coeffs.update({"bias" : self.b_iter})
            self.w_iter = self.w_iter - self.lr * cost.gradient(all_vars_coeffs)[0 : len(cost.gradient(all_vars_coeffs)) - 1]
            self.b_iter = self.b_iter - self.lr * cost.gradient(all_vars_coeffs)[-1]
            cost_list.append(cost.evaluate(all_vars_coeffs))
            if self.verbose:
                if wq <= 9:
                    print(f"[{wq}]     ", "loss:", cost_list[wq])
                elif wq <= 99:
                    print(f"[{wq}]    ", "loss:", cost_list[wq])
                elif wq <= 999:
                    print(f"[{wq}]   ", "loss:", cost_list[wq])
                elif wq <= 9999:
                    print(f"[{wq}]  ", "loss:", cost_list[wq])
                else:
                    print(f"[{wq}] ", "loss:", cost_list[wq])
        print(f"LogisticRegressor(loss_score={cost_list[self.epochs-1]},", f"weights={self.w_iter},", f"bias={self.b_iter})")

    def predict(self, X):
        vals = []
        for i in range(np.shape(X)[0]):
            lin_comb = sum(c * x for c, x in zip(self.w_iter, X[i]))
            vals.append((1 / (1 + math.exp((lin_comb) + self.b_iter))))
        return vals