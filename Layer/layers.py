from Neuron.neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, num_neurons, input_size, learning_rate=0.1, activation='sigmoid'):
        self.neurons = [Neuron(input_size, learning_rate, activation) for _ in range(num_neurons)]
        self.num_neurons = num_neurons
        self.input_size = input_size

    def forward(self, x):
        # Store input for backpropagation
        self.inputs = x
        
        # Process input through each neuron
        self.outputs = np.array([neuron.forward(x) for neuron in self.neurons])
        return self.outputs

    def backward(self, error_gradients):
        # Ensure error_gradients is an array
        if np.isscalar(error_gradients):
            error_gradients = np.array([error_gradients])
            
        # Make sure error_gradients has the right shape
        if len(error_gradients) != self.num_neurons and len(error_gradients) == 1:
            # Broadcast the error to all neurons if there's only one error value
            error_gradients = np.array([error_gradients[0]] * self.num_neurons)
        
        # Initialize input gradients with the right shape
        input_gradients = np.zeros(self.input_size)
        
        # Backpropagate through each neuron
        for i, neuron in enumerate(self.neurons):
            # Get the error gradient for this neuron
            gradient = error_gradients[i] if i < len(error_gradients) else error_gradients[0]
            
            # Backpropagate
            grad = neuron.backward(gradient)
            
            # Accumulate gradients
            if isinstance(grad, np.ndarray):
                if len(grad) == len(input_gradients):
                    input_gradients += grad
                else:
                    # Handle case where grad is of wrong size
                    grad_resized = np.zeros_like(input_gradients)
                    min_size = min(len(grad), len(input_gradients))
                    grad_resized[:min_size] = grad[:min_size]
                    input_gradients += grad_resized
            else:
                # Handle scalar gradient
                input_gradients += np.ones_like(input_gradients) * grad
                
        return input_gradients