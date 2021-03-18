from abc import ABC, abstractclassmethod
import numpy as np


class Layer(ABC):
    @abstractclassmethod
    def forward(self, inputs):
        pass

    @abstractclassmethod
    def backward(self, d_out):
        pass


class ReLU(Layer):
    def forward(self, inputs):
        self.cache = inputs
        return inputs * (inputs > 0)

    def backward(self, d_out):
        inputs = self.cache
        return d_out * (inputs > 0)


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
    def __init__(self, prob: float) -> None:
        super().__init__()
        self.p = prob

    def forward(self, inputs):
        mask = np.random.binomial(1, self.p, size=inputs.shape)
        self.cache = mask
        return inputs * mask / (1-self.p)

    def backward(self, d_out):
        mask = self.cache
        return d_out * mask / (1-self.p)
