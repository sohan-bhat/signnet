import numpy as np
from data import get_data
from layers import Conv, Dense, Flatten, MaxPool, ReLU
from losses import Softmax_CrossEntropy
from network import Network

print("Loading data...")
X_train, y_train, X_test, y_test = get_data() 

print("Building network...")
layers = [
    Conv(32, 3, 3), ReLU(), MaxPool(2),
    Conv(64, 3, 32), ReLU(), MaxPool(2),
    Flatten(),
    Dense(2304, 128), ReLU(),
    Dense(128, 43)
]
network = Network(layers)
network.load("models/model.npz")
cross_entropy = Softmax_CrossEntropy()

def evaluate(X, y, batch_size=64):
    correct = 0
    for start in range(0, X.shape[0], batch_size):
        X_batch = X[start:start+batch_size]
        y_batch = y[start:start+batch_size]
        preds = np.argmax(network.forward(X_batch), axis=1)
        correct += np.sum(preds == y_batch)
    return correct / X.shape[0] * 100
print(evaluate(X_test, y_test))


def train(learning_rate, epochs, batch_size):
    print(f"Training: {epochs} epochs, batch {batch_size}, lr {learning_rate}\n")
    print(f"{'Epoch':>6} | {'Train':>7} | {'Test':>7} | {'Loss':>8}")
    print("-" * 38)
    max_acc = 0
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
        train_acc = evaluate(X_train, y_train)
        test_acc = evaluate(X_test, y_test)
        print(f"{i+1:>6} | {train_acc:>6.2f}% | {test_acc:>6.2f}% | {loss:>8.4f}")
        learning_rate *= 0.95
        if test_acc > max_acc == 0:
            max_acc =  test_acc
            network.save("model.npz")
train(0.005, 50, 64)
network.save("model.npz")