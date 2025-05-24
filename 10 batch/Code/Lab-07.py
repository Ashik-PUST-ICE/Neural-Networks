import numpy as np

# Sigmoid activation function and its derivative (for training)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input and target datasets
X_input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

D_target = np.array([[0], [0], [1], [1]])

# Neural network parameters
input_layer_size = 3
output_layer_size = 1
learning_rate = 0.1
max_epochs = 10000

# Initialize weights with random values
np.random.seed(42)
weights = np.random.randn(input_layer_size, output_layer_size)

# Training the neural network with SGD
for epoch in range(max_epochs):
    error_sum = 0

    for i in range(len(X_input)):
        # Forward pass
        input_data = X_input[i]
        target_data = D_target[i]

        net_input = np.dot(input_data, weights)
        predicted_output = sigmoid(net_input)

        # Calculate error
        error = target_data - predicted_output
        error_sum += np.abs(error)

        # Update weights using the delta learning rule
        weight_update = learning_rate * error * sigmoid_derivative(predicted_output) * input_data
        weights += weight_update[:, np.newaxis]  # Update weights for each input separately

    # Check for convergence
    if error_sum < 0.01:
        print("Converged in {} epochs.".format(epoch + 1))
        break

# Test data
test_data = X_input

# Use the trained model to recognize target function
print("Target Function Test:")
for i in range(len(test_data)):
    input_data = test_data[i]
    net_input = np.dot(input_data, weights)
    predicted_output = sigmoid(net_input)

    print(f"Input: {input_data} -> Output: {np.round(predicted_output)}")