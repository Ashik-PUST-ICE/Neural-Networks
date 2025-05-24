# Perceptron training function
def perceptron_train(inputs, targets, learning_rate=0.1, max_epochs=1000):
    num_inputs = inputs.shape[1]
    num_samples = inputs.shape[0]

    # Initialize weights and bias
    weights = np.random.randn(num_inputs)
    bias = np.random.randn()

    convergence_curve = []

    for epoch in range(max_epochs):
        misclassified = 0

        for i in range(num_samples):
            net_input = np.dot(inputs[i], weights) + bias
            predicted = 1 if net_input >= 0 else 0

            if predicted != targets[i]:
                misclassified += 1
                update = learning_rate * (targets[i] - predicted)
                weights += update * inputs[i]
                bias += update

        accuracy = (num_samples - misclassified) / num_samples
        convergence_curve.append(accuracy)

        if misclassified == 0:
            print("Converged in {} epochs.".format(epoch + 1))
            break

    return weights, bias, convergence_curve

# Generate random linearly separable data points
def generate_data(n_samples):
    np.random.seed(42)
    inputs = np.random.rand(n_samples, 2) * 10
    targets = np.sum(inputs, axis=1) >= 10
    targets = targets.astype(int)
    return inputs, targets

# Main function
if __name__ == "__main__":
    # Generate linearly separable data
    n_samples = 100
    inputs, targets = generate_data(n_samples)

    # Training the perceptron
    weights, bias, convergence_curve = perceptron_train(inputs, targets)

    # Decision boundary line
    x = np.linspace(0, 10, 100)
    y = (-weights[0] * x - bias) / weights[1]

    # Plot convergence curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(convergence_curve) + 1), convergence_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Convergence Curve')
    plt.grid()
    plt.show()

    # Plot the decision boundary line and data points
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Decision Boundary')
    plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], label='Class 1', color='blue')
    plt.scatter(inputs[targets == 0][:, 0], inputs[targets == 0][:, 1], label='Class 0', color='red')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.grid()
    plt.show()