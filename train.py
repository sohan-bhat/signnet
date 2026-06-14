import numpy as np
from data import get_data
from layers import Dense, ReLU
from losses import Softmax_CrossEntropy
from network import Network

print("Loading data...")
X_train, y_train, X_test, y_test = get_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("Building network...")
layers = [Dense(3072, 128), ReLU(), Dense(128, 43)]
network = Network(layers)
cross_entropy = Softmax_CrossEntropy()

def train(learning_rate, epochs, batch_size):
    print(f"Training: {epochs} epochs, batch {batch_size}, lr {learning_rate}\n")
    print(f"{'Epoch':>6} | {'Train':>7} | {'Test':>7} | {'Loss':>8}")
    print("-" * 38)
    for i in range(epochs):
        loss = 0
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        for start in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[start:start+batch_size]
            y_batch = y_shuffled[start:start+batch_size]
            output = network.forward(X_batch)
            loss = cross_entropy.forward(output, y_batch)
            cross_entropy.backward()
            network.backward(cross_entropy.dinputs)
            for layer in network.parameters():
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases
        train_preds = np.argmax(network.forward(X_train), axis=1)
        test_preds = np.argmax(network.forward(X_test), axis=1)
        train_acc = np.mean(train_preds == y_train)*100
        test_acc = np.mean(test_preds == y_test)*100
        print(f"{i+1:>6} | {train_acc:>6.2f}% | {test_acc:>6.2f}% | {loss:>8.4f}")

train(0.05, 100, 64)
network.save("model.npz")