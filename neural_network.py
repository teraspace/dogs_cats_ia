import random
import numpy as np 
class NeuralNetwork():
    parameters = {}
    
    def __init__(self, X_dataset, Y_dataset, n_h, mode):
        n_x = X_dataset.T.shape[0]
        n_y = Y_dataset.T.shape[0]
        W1 = np.random.randn(n_h, n_x ) * 0.01
        W2 = np.random.randn(n_y, n_h) * 0.01
        b1 = np.zeros((n_h,1))
        b2 = np.zeros((n_y,1))
        parameters = { "W1": W1,
                       "b1": b1,
                       "W2": W2,
                       "b2": b2,
                       "mode": mode
                     }
        self.set_parameters(parameters)
        
    def sigmoid(self, s):
        return 1 / (1 + np.power(np.e, -s))            
    def sigmoid_derivative(self, s):
        return s * (1 - s)
    def tanh(self,s):
        return np.tanh(s)
    def tanh_derivative(self, s):
        return 1. - (np.power(s,2))
    def relu(self, s):
        return np.maximum(0,s)
    def relu_derivative(self, s):
        dZ = np.array(s, copy=True)
        dZ[s <= 0] = 0
        return dZ
    def axon(self, s, mode):
            a = {                
                'sigmoid_derivative': self.sigmoid_derivative,
                'tanh': self.tanh,
                'sigmoid': self.sigmoid,
                'tanh_derivative': self.tanh_derivative,
                'relu': self.relu,
                'relu_derivative': self.relu_derivative
            }
            return a[mode](s)
            
    
    def set_parameters(self, parameters):
        self.parameters = parameters
    def get_parameters(self):
        return self.parameters
    def get_learning(self):
        return get_learning
    def set_learning(self, learning):
        self.learning = learning
    def think(self, x, w, b, mode):
        d = np.dot(w, x) + b 
        a =  self.axon(d, mode)
        return a
    def learn(self, a1, a2, X, Y):
        W1 = self.get_parameters()['W1']
        W2 = self.get_parameters()['W2']
        b1 = self.get_parameters()['b1']
        b2 = self.get_parameters()['b2']
        mode = self.get_parameters()['mode']
        m = X.shape[1]
        dZ2 = a2 - Y
        dW2 = 1 / m *(np.dot(dZ2,a1.T))
        db2 = 1 / m *(np.sum(dZ2,axis = 1,keepdims = True))
        
        dZ1 = np.dot(W2.T,dZ2) * self.axon(a1, mode + '_derivative')
        dW1 = 1 / m *(np.dot(dZ1,X.T))
        db1 = 1 / m *(np.sum(dZ1,axis = 1,keepdims = True))
        learning = {  "dZ1": dZ1,
                      "dW1": dW1,
                      "db1": db1,
                      "dZ2": dZ2,
                      "dW2": dW2,
                      "db2": db2,
                      "mode": mode
                   }        
        self.set_learning(learning)
        return learning
    def trainer(self, lessons, learning_rate, X_dataset, Y_dataset, log=False):
        W1 = self.get_parameters()['W1']
        W2 = self.get_parameters()['W2']
        b1 = self.get_parameters()['b1']
        b2 = self.get_parameters()['b2']
        mode = self.get_parameters()['mode']
        for m in range(0, lessons):
            a1 = self.think( X_dataset, W1, b1, mode )
            a2 = self.think(a1, W2, b2, mode)
            learning = self.learn(a1, a2, X_dataset, Y_dataset)
            W1 = W1 - learning_rate * learning['dW1']
            b1 = b1 - learning_rate * learning['db1']
            W2 = W2 - learning_rate * learning['dW2']
            b2 = b2 - learning_rate * learning['db2']
        parameters = { "W1": W1,
               "b1": b1,
               "W2": W2,
               "b2": b2,
               "mode": mode
             }
        self.set_parameters(parameters)
        return parameters
    def test(self, x):
        W1 = self.get_parameters()['W1']
        W2 = self.get_parameters()['W2']
        b1 = self.get_parameters()['b1']
        b2 = self.get_parameters()['b2']
        mode = self.get_parameters()['mode']
        a1 = self.think( x, W1, b1, mode )
        a2 = self.think(a1, W2, b2, mode)
        return a2
    
    
    class HiddenLayer():
        def __init__(self, n_h, mode):
            l = {
                
            }