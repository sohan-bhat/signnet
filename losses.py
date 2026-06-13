import numpy as np

class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output =  exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

class CrossEntropy:
    def forward(self, predicted, y_true):
        clipped_pred =  np.clip(predicted, 1e-7, 1 - 1e-7)
        samples = len(clipped_pred)
        if len(y_true.shape) == 1:
            probabilities = clipped_pred[range(samples), y_true]
        else:
            probabilities = np.sum(clipped_pred * y_true, axis=1, keepdims=True)
        loss = -np.log(probabilities)
        return np.mean(loss)
        
class Softmax_CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()
        self.loss = CrossEntropy()
    def forward(self, inputs, y_true):
        self.y_true = y_true
        self.probabilities = self.softmax.forward(inputs)
        loss = self.loss.forward(self.probabilities, y_true)
        return loss
    def backward(self):
        samples = len(self.probabilities)
        copy = self.probabilities.copy()
        if len(self.y_true.shape) == 1:
            copy[range(samples), self.y_true] -= 1
        else:
            copy -= self.y_true
        self.dinputs = copy / samples