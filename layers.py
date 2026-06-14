import numpy as np
from numpy.ma.core import zeros

# kind of complex so comments for procedure
class Conv:
    # initialize filters and one bias per filter
    def __init__(self, num_filters, filter_size, num_channels):
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) * 0.1
        self.biases = np.zeros(num_filters)
    # slide each filter over the image, dot-product each patch, build feature maps
    def forward(self, inputs):
        self.inputs = inputs
        batch, height, width, channels = inputs.shape
        num_filters, filter_size, _, _ = self.filters.shape
        out_height = height - filter_size + 1
        out_width = width - filter_size + 1
        output = np.zeros((batch, out_height, out_width, num_filters))
        for h in range(out_height):
            for w in range(out_width):
                patch = inputs[:,h:h+filter_size, w:w+filter_size,:]
                conv = np.tensordot(patch, self.filters, axes=([1,2,3], [1,2,3])) + self.biases # multiply patch against all filters and sum over spatial+channels
                output[:, h, w, :] = conv
        self.output = output
        return output
    # compute gradients for filters, biases, and inputs by sliding back over each position
    def backward(self, dvalues):
        self.dbiases = np.sum(dvalues, axis=(0,1,2))
        filter_size = self.filters.shape[1]
        _, out_height, out_width, _ = dvalues.shape
        self.dfilters = np.zeros_like(self.filters)
        self.dinputs = np.zeros_like(self.inputs)
        for h in range(out_height):
            for w in range(out_width):
                patch = self.inputs[:,h:h+filter_size, w:w+filter_size,:]
                grad = dvalues[:,h, w,:]
                accum_dfilters = np.tensordot(grad, patch, axes=([0],[0])) # how each filter should change, from patches weighted by incoming gradient
                self.dfilters += accum_dfilters
                accum_dinputs = np.tensordot(grad, self.filters, axes=([1],[0]))  # gradient passed back to previous layer, through the filters
                self.dinputs[:,h:h+filter_size, w:w+filter_size,:] += accum_dinputs

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