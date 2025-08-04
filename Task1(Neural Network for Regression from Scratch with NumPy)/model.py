# model.py

import numpy as np

# =============================================================================
# Layers
# =============================================================================

class Dense:
    """A fully-connected layer."""
    def __init__(self, input_size, output_size):
        """
        Initializes weights and biases.
        Weights are initialized using He initialization for ReLU activations.
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.inputs = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, inputs):
        """
        Performs the forward pass: Z = XW + b
        """
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.biases

    def backward(self, grad_output):
        """
        Performs the backward pass.
        Calculates gradients for weights, biases, and the input to this layer.
        """
        # Gradient of the loss with respect to the weights
        self.grad_weights = np.dot(self.inputs.T, grad_output)
        
        # Gradient of the loss with respect to the biases
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient of the loss with respect to the input (to be passed to the previous layer)
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input

# =============================================================================
# Activation Functions
# =============================================================================

class ReLU:
    """ReLU activation function."""
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        """
        Applies the ReLU function: max(0, x)
        """
        self.inputs = inputs
        return np.maximum(0, self.inputs)

    def backward(self, grad_output):
        """
        Computes the gradient of the loss with respect to the input of ReLU.
        The derivative of ReLU is 1 for x > 0 and 0 for x <= 0.
        """
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input

class Sigmoid:
    """Sigmoid activation function."""
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        """
        Applies the Sigmoid function: 1 / (1 + exp(-x))
        """
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        """
        Computes the gradient of the loss with respect to the input of Sigmoid.
        The derivative is sigmoid(x) * (1 - sigmoid(x)).
        """
        return grad_output * self.output * (1 - self.output)

# =============================================================================
# Loss Function
# =============================================================================

class MeanSquaredError:
    """Mean Squared Error loss."""
    def forward(self, y_pred, y_true):
        """
        Calculates the loss: L = (1/N) * sum((y_pred - y_true)^2)
        """
        return np.mean(np.power(y_pred - y_true, 2))

    def backward(self, y_pred, y_true):
        """
        Computes the gradient of the loss with respect to the predictions.
        d(L)/d(y_pred) = (2/N) * (y_pred - y_true)
        """
        n_samples = y_true.shape[0]
        return 2 * (y_pred - y_true) / n_samples

# =============================================================================
# Optimizer
# =============================================================================

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):
        """
        Updates the parameters of all layers with gradients.
        """
        for layer in self.layers:
            # Check if the layer has weights and biases (i.e., it's a Dense layer)
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.biases -= self.learning_rate * layer.grad_biases