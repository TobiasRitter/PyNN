from layers import Layer
from scipy.special import softmax
import numpy as np


class CategoricalCrossEntropy(Layer):
    def forward(self, logits, labels):
        probs = softmax(logits, axis=1)
        self.cache = (probs, labels)
        loss = -np.sum(np.sum(labels*np.log(probs)))/probs.shape[0]
        return loss

    def backward(self, d_out=1.0):
        probs, labels = self.cache
        N = labels.shape[0]
        d_logits = (probs-labels)*d_out/N
        d_labels = None
        return d_logits, d_labels
