from Network.network import NeuralNetwork
import numpy as np

# Simple XOR dataset inputs and labels
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR outputs
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Create network with learning rate 0.1
nn = NeuralNetwork(learning_rate=0.1)

# Train for 5000 epochs
nn.train(X, Y, epochs=5000)

# Test trained network
for x in X:
    output = nn.forward(x)
    print(f"Input: {x} -> Output: {output}")
