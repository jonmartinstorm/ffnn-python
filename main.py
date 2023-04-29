import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x, weights, biases, activation_function):
    return activation_function(np.dot(x, weights) + biases)

def binary_cross_entropy(y_true, y_predicted):
    return -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def backwards_pass(x, y_true, y_pred, hidden_layer1_output, hidden_layer2_output, weights2, weights3):
    # Compute gradients for the output layer (use sigmoid_derivative)
    dL_dy = -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred)))
    dy_dz3 = sigmoid_derivative(np.dot(hidden_layer2_output, weights3) + biases3)
    dL_dz3 = dL_dy * dy_dz3

    # Compute gradients for the second hidden layer (use relu_derivative)
    dz3_dh2 = weights3
    dL_dh2 = np.dot(dL_dz3, dz3_dh2.T)
    dh2_dz2 = relu_derivative(np.dot(hidden_layer1_output, weights2) + biases2)
    dL_dz2 = dL_dh2 * dh2_dz2

    # Compute gradients for the first hidden layer (use relu_derivative)
    dz2_dh1 = weights2
    dL_dh1 = np.dot(dL_dz2, dz2_dh1.T)
    dh1_dz1 = relu_derivative(np.dot(x, weights1) + biases1)
    dL_dz1 = dL_dh1 * dh1_dz1

    # Compute gradients for the weights and biases
    dL_dw1 = np.outer(x, dL_dz1)
    dL_db1 = dL_dz1
    dL_dw2 = np.outer(hidden_layer1_output, dL_dz2)
    dL_db2 = dL_dz2
    dL_dw3 = np.outer(hidden_layer2_output, dL_dz3)
    dL_db3 = dL_dz3

    return dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3

def gradient_descent_update(weights, biases, dL_dweights, dL_dbias, lr):
    weights -= lr * dL_dweights
    biases -= lr * dL_dbias
    return weights, biases

np.random.seed(42)

learning_rate = 0.01
epochs = 1000

input_size = 2
hidden_layer1_size = 3
hidden_layer2_size = 2
output_size = 1

# Weights and biases tensors
weights1 = np.random.randn(input_size, hidden_layer1_size)
biases1 = np.random.randn(hidden_layer1_size)

weights2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
biases2 = np.random.randn(hidden_layer2_size)

weights3 = np.random.randn(hidden_layer2_size, output_size)
biases3 = np.random.randn(output_size)

# print(f"weights1: \n{weights1}")
# print(f"biases1: \n{biases1}")
# print()

# print(f"weights2: \n{weights2}")
# print(f"biases2: \n{biases2}")
# print()

# print(f"weights3: \n{weights3}")
# print(f"biases3: \n{biases3}")
# print()

# Test input
x = np.array([0.5, 0.7])
yt = np.array([1])

# Forward pass
hidden_layer1_output = forward_pass(x, weights1, biases1, relu)
hidden_layer2_output = forward_pass(hidden_layer1_output, weights2, biases2, relu)
output = forward_pass(hidden_layer2_output, weights3, biases3, sigmoid)

print("Input:", x)
print("Output:", output)


# training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer1_output = forward_pass(x, weights1, biases1, relu)
    hidden_layer2_output = forward_pass(hidden_layer1_output, weights2, biases2, relu)
    y_pred = forward_pass(hidden_layer2_output, weights3, biases3, sigmoid)

    # compute loss
    loss = binary_cross_entropy(yt, y_pred)

    # Backwards pass
    dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3 = backwards_pass(x, yt, y_pred, hidden_layer1_output, hidden_layer2_output, weights2, weights3)

    # Update weights and biases
    weights1, biases1 = gradient_descent_update(weights1, biases1, dL_dw1, dL_db1, learning_rate)
    weights2, biases2 = gradient_descent_update(weights2, biases2, dL_dw2, dL_db2, learning_rate)
    weights3, biases3 = gradient_descent_update(weights3, biases3, dL_dw3, dL_db3, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.3f}")

print("Trained weights & biases:")
print("weights1:\n", weights1)
print("biases1:\n", biases1)
print("weights2:\n", weights2)
print("biases2:\n", biases2)
print("weights3:\n", weights3)
print("biases3:\n", biases3)


