# ffnn-python
Implementing a small Feed Forward Neural Network just to understand the basics

## ChatGPT hints
> So when implementing a Feed forward neural network it’s just a tensor of weights and biases?

When implementing a feed-forward neural network, the weights and biases are indeed represented as tensors. However, the overall structure of a neural network is more than just these tensors. Let's break down the components of a feed-forward neural network:

1. **Layers**: A feed-forward neural network typically consists of an input layer, one or more hidden layers, and an output layer. Each layer is composed of a set of neurons (also called nodes) that are connected to the neurons in the adjacent layers.

2. **Weights**: The connections between neurons in adjacent layers have associated weights. These weights determine the strength of the connections between neurons. Weights can be represented as tensors, where each element in the tensor corresponds to the weight between two connected neurons. For example, in a fully connected network, the weights connecting layer i to layer i+1 can be stored in a matrix (rank-2 tensor) W_i, where each element W_ij represents the weight between neuron j in layer i and neuron i in layer i+1.

3. **Biases**: In addition to the weights, each neuron has an associated bias term. The bias allows the neuron to produce different outputs even if the input remains the same. Biases can be represented as vectors (rank-1 tensors), where each element in the vector corresponds to the bias term of a neuron in a specific layer.

4. **Activation functions**: Neurons in a neural network apply an activation function to their inputs. Activation functions introduce non-linearity into the network, enabling it to learn complex patterns and representations. Common activation functions include sigmoid, ReLU (rectified linear unit), and tanh.

5. **Loss function**: To train a neural network, you need a loss function that quantifies the difference between the predicted outputs and the actual target values. The goal of training is to minimize this loss function by adjusting the weights and biases.

5. **Optimization algorithm**: An optimization algorithm, such as gradient descent or one of its variants (e.g., stochastic gradient descent, Adam), is used to update the weights and biases during training based on the gradients of the loss function.

In summary, while tensors are used to represent weights and biases in a feed-forward neural network, the network's overall structure also includes layers, activation functions, a loss function, and an optimization algorithm.

> So we represent the layers weights and biases with a tensor and then use activation and optimization algorithms on that. 

Yes, that's correct. In a feed-forward neural network, the weights and biases for each layer are represented using tensors. Specifically, the weights connecting two adjacent layers can be represented as a matrix (rank-2 tensor) and the biases for each layer can be represented as a vector (rank-1 tensor).

During the forward pass of the neural network, the input is passed through each layer, and the output from the previous layer is multiplied by the weights and added to the biases of the current layer. The result is then passed through an activation function, which introduces non-linearity and determines the output of the current layer. This process is repeated for all layers in the network.

During the training phase, an optimization algorithm is used to adjust the weights and biases based on the gradients of the loss function. This process, called backpropagation, involves computing the partial derivatives of the loss function with respect to the weights and biases, and then updating them using the optimization algorithm (e.g., gradient descent, stochastic gradient descent, Adam) to minimize the loss.

So, the feed-forward neural network consists of layers with weights and biases represented as tensors, and the training process involves using activation functions during the forward pass and optimization algorithms during the backpropagation to adjust the weights and biases to minimize the loss function.

> Let’s create a really small neural network. Two inputs two hidden layers with 3 and 2 neurons. And one input. Write the tensors for it in python 

Here's a simple implementation of a neural network with two input nodes, two hidden layers (with 3 and 2 neurons), and one output node using Python and NumPy:

```python
import numpy as np

# Initialize random seed for reproducibility
np.random.seed(42)

# Layer sizes
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

print("Weights 1:\n", weights1)
print("Biases 1:\n", biases1)

print("Weights 2:\n", weights2)
print("Biases 2:\n", biases2)

print("Weights 3:\n", weights3)
print("Biases 3:\n", biases3)

```

This code creates three weight matrices and three bias vectors for the connections between the input layer and the first hidden layer, between the first and second hidden layers, and between the second hidden layer and the output layer. The weight matrices and bias vectors are initialized with random values from a normal distribution.

The resulting tensors are:

 - Weights 1: (2, 3) matrix for connections between the input layer and the first hidden layer.
 - Biases 1: (3,) vector for biases of the first hidden layer neurons.
 - Weights 2: (3, 2) matrix for connections between the first hidden layer and the second hidden layer.
 - Biases 2: (2,) vector for biases of the second hidden layer neurons.
 - Weights 3: (2, 1) matrix for connections between the second hidden layer and the output layer.
 - Biases 3: (1,) vector for biases of the output layer neuron.

