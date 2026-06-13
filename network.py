import numpy as np
from layers import Dense, ReLU
from losses import Softmax_CrossEntropy

class Network:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def backward(self, dvalues):
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs
    def parameters(self):
        trainable = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                trainable.append(layer)
        return trainable