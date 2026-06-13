import numpy as np
from data import get_data
from layers import Dense, ReLU
from losses import Softmax_CrossEntropy
from network import Network

X_train, y_train, X_test, y_test = get_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

layers = [Dense(3072, 128), ReLU(), Dense(128, 43)]
network = Network(layers)
cross_entropy = Softmax_CrossEntropy()

def train(learning_rate, epochs):
    for _ in range(epochs):
        output = network.forward(X_train)
        loss = cross_entropy.forward(output, y_train)
        cross_entropy.backward()
        network.backward(cross_entropy.dinputs)
        for layer in network.parameters():
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases
        print(loss)

train(0.01, 10)
    