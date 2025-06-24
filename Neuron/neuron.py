import numpy as np

class Neuron:
    def __init__(self, input_size, learning_rate=0.1, activation=None):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.activation = activation

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        # Add clipping to prevent overflow
        x = np.clip(x, -20, 20)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Ensure input is 1D array
        if isinstance(x, np.ndarray) and x.ndim > 1:
            x = x.flatten()
            
        self.input = x
        z = np.dot(self.input, self.weights) + self.bias
        self.z = z

        if self.activation == 'relu':
            self.output = self.relu(z)
        else:
            self.output = self.sigmoid(z)

        return self.output

    def backward(self, error_gradient):
        if self.activation == 'relu':
            delta = error_gradient * self.relu_derivative(self.z)
        else:
            delta = error_gradient * self.sigmoid_derivative(self.output)

        # Ensure delta is a scalar
        delta = np.atleast_1d(delta)[0]
        
        # Calculate gradients
        weights_gradient = delta * self.input
        bias_gradient = delta

        # Update weights and bias
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient

        # Return gradient for previous layer
        return delta * self.weights