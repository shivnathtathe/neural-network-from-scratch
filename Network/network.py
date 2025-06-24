from Layer.layers import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.1):
        # Simpler architecture specifically for XOR
        self.layer1 = Layer(num_neurons=4, input_size=2, learning_rate=learning_rate, activation='relu')
        self.layer2 = Layer(num_neurons=1, input_size=4, learning_rate=learning_rate, activation='sigmoid')

    def forward(self, x):
        # Ensure input is the right shape
        x = np.array(x, dtype=float)
        
        # Forward pass through the network
        hidden_output = self.layer1.forward(x)
        final_output = self.layer2.forward(hidden_output)
        return final_output

    def train(self, X, Y, epochs=5000, print_interval=100):
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for x, y_true in zip(X, Y):
                # Forward pass
                y_pred = self.forward(x)
                
                # Calculate error and loss
                error = y_true - y_pred
                loss = np.mean(error**2)
                total_loss += loss
                
            
                # Gradient of MSE loss with respect to output
                grad_output = -2 * error 
                
                # Backpropagate through output layer
                grad_layer2 = self.layer2.backward(grad_output)
                
                # Backpropagate through hidden layer
                self.layer1.backward(grad_layer2)
            
            # Save loss for plotting
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            # Print progress
            if epoch % print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                # Print sample weights
                if epoch % (print_interval * 10) == 0:
                    print(f"Sample weights layer 1 neuron 0: {self.layer1.neurons[0].weights}")
        
        return losses