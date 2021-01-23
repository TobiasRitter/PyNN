from abc import ABC, abstractclassmethod
import numpy as np


class Layer(ABC):
    @abstractclassmethod
    def forward():
        pass

    @abstractclassmethod
    def backward():
        pass


class ReLU(Layer):
    def forward(self, inputs):
        self.cache = inputs
        return inputs * (inputs > 0)

    def backward(self, d_out):
        inputs = self.cache
        return d_out * (inputs > 0)


class Sigmoid(Layer):
    def forward():
        pass

    def backward():
        pass


class Softmax(Layer):
    def forward():
        pass

    def backward():
        pass


class Affine(Layer):
    def forward(self, inputs, weights, bias):
        self.cache = (inputs, weights)
        return inputs @ weights + bias

    def backward(self, d_out):
        inputs, weights = self.cache
        d_inputs = d_out @ weights.transpose()
        d_weights = inputs.transpose() @ d_out
        n = d_out[:, 0]
        d_bias = np.ones_like(n) @ d_out
        return d_inputs, d_weights, d_bias


class Dropout(Layer):
    def forward(self, inputs):
        pass

    def backward():
        pass
