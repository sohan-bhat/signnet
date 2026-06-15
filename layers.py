import numpy as np
from numpy.ma.core import zeros

# kind of complex so comments for procedure
class Conv:
    # initialize filters (stored as weights for a uniform interface) and one bias per filter
    def __init__(self, num_filters, filter_size, num_channels):
        self.weights = np.random.randn(num_filters, filter_size, filter_size, num_channels) * 0.1
        self.biases = np.zeros(num_filters)

    def im2col(self, inputs, out_height, out_width, filter_size):
        batch, height, width, channels = inputs.shape
        cols = np.zeros((batch, out_height, out_width, filter_size, filter_size, channels))
        for i in range(filter_size):
            for j in range(filter_size):
                cols[:, :, :, i, j, :] = inputs[:, i:i+out_height, j:j+out_width, :]
        return cols.reshape(batch * out_height * out_width, -1)

    def col2im(self, dcols, inputs_shape, out_height, out_width, filter_size):
        batch, height, width, channels = inputs_shape
        dcols_reshaped = dcols.reshape(batch, out_height, out_width, filter_size, filter_size, channels)
        dinputs = np.zeros(inputs_shape)
        for i in range(filter_size):
            for j in range(filter_size):
                dinputs[:, i:i+out_height, j:j+out_width, :] += dcols_reshaped[:, :, :, i, j, :]
        return dinputs

    def forward(self, inputs):
        self.inputs = inputs
        batch, height, width, channels = inputs.shape
        num_filters, filter_size = self.weights.shape[0], self.weights.shape[1]
        out_height = height - filter_size + 1
        out_width = width - filter_size + 1
        self.cols = self.im2col(inputs, out_height, out_width, filter_size)
        filters_flat = self.weights.reshape(num_filters, -1)
        out = self.cols @ filters_flat.T + self.biases
        self.output = out.reshape(batch, out_height, out_width, num_filters)
        return self.output

    def backward(self, dvalues):
        dvalues_flat = dvalues.reshape(-1, dvalues.shape[-1])
        self.dbiases = np.sum(dvalues_flat, axis=0)
        dweights_flat = self.cols.T @ dvalues_flat
        self.dweights = dweights_flat.T.reshape(self.weights.shape)
        filters_flat = self.weights.reshape(self.weights.shape[0], -1)
        dcols = dvalues_flat @ filters_flat
        _, out_height, out_width, _ = dvalues.shape
        filter_size = self.weights.shape[1]
        self.dinputs = self.col2im(dcols, self.inputs.shape, out_height, out_width, filter_size)
        return self.dinputs

class MaxPool:
    # size of each block to downsample (2 means 2x2 -> 1)
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    # slide non-overlapping windows, shrinks feature maps by half
    def forward(self, inputs):
        self.inputs = inputs
        batch, height, width, channels = inputs.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size
        output = np.zeros((batch, out_height, out_width, channels))
        for h in range(out_height):
            for w in range(out_width):
                hpool_size = h*self.pool_size
                wpool_size = w*self.pool_size
                window = inputs[:,hpool_size:hpool_size+self.pool_size, wpool_size:wpool_size+self.pool_size,:]
                output[:, h, w, :] = np.max(window, axis=(1,2)) # take the strongest activation in each window
        self.output = output
        return output
    # route gradient only to the position that was the max
    def backward(self, dvalues):
        self.dinputs = np.zeros_like(self.inputs)
        _, out_height, out_width, _ = dvalues.shape
        for h in range(out_height):
            for w in range(out_width):
                hpool_size = h*self.pool_size
                wpool_size = w*self.pool_size
                window = self.inputs[:,hpool_size:hpool_size+self.pool_size, wpool_size:wpool_size+self.pool_size,:]
                mask = (window == np.max(window, axis=(1,2), keepdims=True)) # mark where the max was in each window
                grad = dvalues[:,h, w,:][:,None, None,:]
                self.dinputs[:, hpool_size:hpool_size+self.pool_size, wpool_size:wpool_size+self.pool_size, :] += mask * grad

# helper to flatten
class Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.inputs > 0)

conv = Conv(8, 3, 3)
input = np.random.randn(2, 32, 32, 3)
grad = np.random.randn(2, 30, 30, 8)
conv.forward(input)
conv.backward(grad)