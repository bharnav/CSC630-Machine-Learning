'''
Here I define the Variable class: a class useful for taking gradients as well as simple arithmetic.
Help from Michael Huang, William Yue, and Andrew Wen
'''

import numpy as np
import math

class Variable():
    '''
    creates an identity matrix, for which all values are 0s, except for the index for self.name, which has a value of 1
    '''

    def grads(self, values):
        #defining an array fill of 0s, for the length of values
        grad = [0]*len(values)
        # getting the position of the current variable in our dictionary values
        # sorts the keys in the dictionary, values
        pos = [i for i, key in enumerate(values.keys()) if key == self.name]
        # if the key is not in values
        if len(pos) == 0:
            raise ValueError('key {self.name} not found')
            
        # finds the position of the current variable, in the sorted keys of the dictionary values
        # sets the index of the current variable to a value of 1
        grad[pos[0]] = 1
        return np.array(grad)
        
    def __init__(self, name=None, evaluate=None, gradient=None):
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate
            
        if gradient == None:
            self.gradient = lambda values: Variable.grads(self, values)
            # self.gradient = lambda values: np.array(list(map((lambda x: int(x == self.name)), sorted(list(values.keys())))))
        else:
            self.gradient = gradient
            
        if name != None:
            self.name = name
    
    def __call__(self, **kwargs): # makes it such that you don't need to call evaluate, and instead run: z.grad(x_1= 3, x_2= 1, x_3= 7)
        return self.evaluate(kwargs)
    
    def grad(self, **kwargs): # allows you to run gradients by calling .grad()
        return self.gradient(kwargs)
    
    def __add__(self, other):
        # adds a variable to a scalar
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) + other, gradient = lambda values: self.gradient(values))
        # adds a variable to a variable
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values), gradient = lambda values: self.gradient(values) + other.gradient(values))
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        # multiplies a variable with a scalar
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) * other, gradient = lambda values: other * self.gradient(values))
        # multiplies a variable with a variable
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values), gradient = lambda values: other.evaluate(values) * self.gradient(values) + other.gradient(values) * self.evaluate(values))
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        # raises a variable to the power of a scalar
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) ** other, gradient = lambda values: (other) * (self.evaluate(values) ** (other - 1)) * self.gradient(values))
        # raises a variable to the power of another variable
        return Variable(evaluate = lambda values: self.evaluate(values) ** other.evaluate(values), gradient = lambda values: other.evaluate(values) * (self.evaluate(values) ** (other.evaluate(values) - 1)) * self.gradient(values) + math.log(self.evaluate(values)) * (self.evaluate(values) ** other.evaluate(values)) * other.gradient(values))

    def __rpow__(self, other):
        # raises a scalar to the power of a vaiable
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: other ** self.evaluate(values), gradient = lambda values: math.log(other) * other ** self.evaluate(values) * self.gradient(values))
        
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def __sub__(self, other):
         return self + (other * -1)
    
    def __rsub__(self, other):
        return (self * -1) + other

    @staticmethod
    def exp(var):
        # raising e to the power of a scalar
        if isinstance(var, (int, float)):
            return Variable(evaluate = math.e ** var, gradient = lambda values: np.zeros(len(values)))
        # raising e to the power of a variable
        return Variable(evaluate = lambda values: math.e ** var.evaluate(values), gradient = lambda values: (math.e ** var.evaluate(values)) * var.gradient(values))
    @staticmethod
    def log(var):
        # taking a log of a scalar
        if isinstance(var, (int, float)):
            return Variable(evaluate = math.log(var), gradient = lambda values: np.zeros(len(values)))
        # taking a log of a variable
        return Variable(evaluate = lambda values: math.log(var.evaluate(values)), gradient = lambda values: (var.evaluate(values) ** -1) * var.gradient(values))