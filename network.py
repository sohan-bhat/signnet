import numpy as np


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
    def save(self, path):
        params = {}
        for i, layer in enumerate(self.parameters()):
            params[f"w{i}"] = layer.weights
            params[f"b{i}"] = layer.biases
        np.savez(path, **params)
    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.parameters()):
            layer.weights = data[f"w{i}"]
            layer.biases = data[f"b{i}"]