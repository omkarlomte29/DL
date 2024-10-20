# Import necessary libraries
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Step 1: Initialize the dataset
# Input data (XOR problem as an example)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected output (XOR truth table)
y = np.array([[0], [1], [1], [0]])

# Step 2: Initialize weights and biases for a 3-layer network
# Randomly initialize weights and biases
input_neurons = 2  # Two inputs (for XOR)
hidden_neurons1 = 4  # First hidden layer neurons
hidden_neurons2 = 4  # Second hidden layer neurons
output_neurons = 1  # Single output (0 or 1 for XOR)

# Initialize weights with random values
weights_input_hidden1 = np.random.uniform(size=(input_neurons, hidden_neurons1))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden_neurons1, hidden_neurons2))
weights_hidden2_output = np.random.uniform(size=(hidden_neurons2, output_neurons))

# Initialize biases with random values
bias_hidden1 = np.random.uniform(size=(1, hidden_neurons1))
bias_hidden2 = np.random.uniform(size=(1, hidden_neurons2))
bias_output = np.random.uniform(size=(1, output_neurons))

# Step 3: Set hyperparameters
learning_rate = 0.1
epochs = 10000  # Number of iterations

# Step 4: Implement forward and backward propagation
for epoch in range(epochs):
    # Forward propagation
    # Step 4.1: Input to the first hidden layer
    hidden_layer1_input = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    # Step 4.2: First hidden layer to second hidden layer
    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    # Step 4.3: Second hidden layer to output layer
    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Step 5: Calculate the loss (mean squared error)
    error = y - predicted_output
    loss = np.mean(np.square(error))

    # Backward propagation
    # Step 5.1: Output layer gradients (error term * derivative of sigmoid)
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Step 5.2: Gradient for second hidden layer
    error_hidden_layer2 = d_predicted_output.dot(weights_hidden2_output.T)
    d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer2_output)

    # Step 5.3: Gradient for first hidden layer
    error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
    d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer1_output)

    # Step 6: Update the weights and biases using gradient descent
    weights_hidden2_output += hidden_layer2_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_hidden1_hidden2 += hidden_layer1_output.T.dot(d_hidden_layer2) * learning_rate
    bias_hidden2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * learning_rate

    weights_input_hidden1 += X.T.dot(d_hidden_layer1) * learning_rate
    bias_hidden1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Step 7: Print the final predicted output after training
print("Final predictions after training:")
print(predicted_output)

# Step 8: Compare with actual outputs
print("\nActual outputs:")
print(y)
