import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def mse(predicted, target):
    """Mean squared error."""
    return np.mean((predicted - target) ** 2)


def classify(output, threshold=0.5):
    """Classifies based on a threshold."""
    return np.where(output >= threshold, 1, 0)


class Perceptron:
    """Perceptron class for learning the XOR problem."""

    def __init__(self, learning_rate=0.01, num_inputs=2, num_hidden=10, num_outputs=1):
        """
        Initializes the perceptron with random weights.

        Args:
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.01.
            num_inputs (int, optional): Number of input features. Defaults to 2.
            num_hidden (int, optional): Number of hidden neurons. Defaults to 2.
            num_outputs (int, optional): Number of output neurons. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.weights_hidden = 2 * np.random.rand(num_inputs, num_hidden) - 1  # Initialize between -1 and 1
        self.weights_output = 2 * np.random.rand(num_hidden, num_outputs) - 1

    def train(self, inputs, targets, epochs=100):
        """
        Trains the perceptron on the given data for a specified number of epochs.

        Args:
            inputs (np.ndarray): Training data inputs.
            targets (np.ndarray): Training data targets.
            epochs (int, optional): Number of training epochs. Defaults to 100.

        Returns:
            history (dict): Training history containing MSE and classification error per epoch.
        """
        history = {'mse_hidden': [], 'mse_output': [], 'classification_error': [], 'weights_hidden': []}
        for epoch in range(epochs):
            # Forward pass (hidden layer)
            hidden_outputs = sigmoid(np.dot(inputs, self.weights_hidden))

            # Forward pass (output layer)
            output = sigmoid(np.dot(hidden_outputs, self.weights_output))

            # Backpropagation: output layer
            output_error = output * (1 - output) * (targets - output)  # Delta rule with sigmoid derivative
            self.weights_output += self.learning_rate * np.dot(hidden_outputs.T, output_error)

            # Backpropagation: hidden layer
            hidden_error = hidden_outputs * (1 - hidden_outputs) * np.dot(output_error, self.weights_output.T)
            self.weights_hidden += self.learning_rate * np.dot(inputs.T, hidden_error)

            # Calculate and store errors

            history['mse_hidden'].append((hidden_error ** 2).sum(axis=0))
            history['weights_hidden'].append((self.weights_hidden**2).mean(axis=0))
            history['mse_output'].append(mse(output, targets))
            history['classification_error'].append(np.mean(classify(output) != targets))

        print('weights_hidden', self.weights_hidden)
        return history



    def predict(self, inputs):
        """
        Predicts the output for given inputs.

        Args:
            inputs (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted outputs.
        """
        hidden_outputs = sigmoid(np.dot(inputs, self.weights_hidden))
        return sigmoid(np.dot(hidden_outputs, self.weights_output))


def solve_xor(data):
    """
    Solves the XOR problem for given data using logical operations.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Predicted outputs.
    """
    output = np.logical_xor(data[:, 0], data[:, 1]).astype(int)
    return output.reshape(-1, 1)


def perform_tests(num_tests, num_epochs, learning_rate):
    accuracies = []

    for _ in range(num_tests):
        # Generate random dataset for XOR problem
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = solve_xor(inputs)

        # Create and train the perceptron
        perceptron = Perceptron(learning_rate=learning_rate)
        history = perceptron.train(inputs, targets, epochs=num_epochs)

        # Evaluate accuracy
        predictions = perceptron.predict(inputs)
        accuracy = np.mean((predictions >= 0.5) == targets)
        accuracies.append(accuracy)

    return accuracies
