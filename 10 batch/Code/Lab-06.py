import numpy as np

# Sigmoid activation function and its derivative (for training)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR function dataset with binary inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

targets = np.array([[0], [1], [1], [0]])

# Neural network parameters
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1
learning_rate = 0.1
momentum_factor = 0.9
max_epochs = 10000

# Initialize weights and biases with random values
np.random.seed(42)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
bias_hidden = np.random.randn(hidden_layer_size)

weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size)
bias_output = np.random.randn(output_layer_size)

# Initialize previous weight updates with zeros for momentum
prev_weight_input_hidden_update = np.zeros((input_layer_size, hidden_layer_size))
prev_bias_hidden_update = np.zeros(hidden_layer_size)

prev_weight_hidden_output_update = np.zeros((hidden_layer_size, output_layer_size))
prev_bias_output_update = np.zeros(output_layer_size)

# Training the neural network with backpropagation and momentum
for epoch in range(max_epochs):
    # Forward pass
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = targets - predicted_output

    # Backpropagation
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_delta = output_delta.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases with momentum
    weight_input_hidden_update = inputs.T.dot(hidden_delta) * learning_rate
    bias_hidden_update = np.sum(hidden_delta, axis=0) * learning_rate

    weight_hidden_output_update = hidden_layer_output.T.dot(output_delta) * learning_rate
    bias_output_update = np.sum(output_delta, axis=0) * learning_rate

    weights_input_hidden += weight_input_hidden_update + momentum_factor * prev_weight_input_hidden_update
    bias_hidden += bias_hidden_update + momentum_factor * prev_bias_hidden_update

    weights_hidden_output += weight_hidden_output_update + momentum_factor * prev_weight_hidden_output_update
    bias_output += bias_output_update + momentum_factor * prev_bias_output_update

    # Store previous updates for momentum
    prev_weight_input_hidden_update = weight_input_hidden_update
    prev_bias_hidden_update = bias_hidden_update

    prev_weight_hidden_output_update = weight_hidden_output_update
    prev_bias_output_update = bias_output_update

    # Calculate mean squared error for convergence check
    mse = np.mean(error ** 2)
    if mse < 1e-6:
        print("Converged in {} epochs.".format(epoch + 1))
        break

# Test the XOR function with the trained neural network
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

hidden_layer_input = np.dot(test_inputs, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

print("Predicted outputs:")
print(predicted_output)

# Round the predicted outputs to get binary values (0 or 1)
predicted_binary = np.round(predicted_output).astype(int)
print("Predicted binary outputs:")
print(predicted_binary)