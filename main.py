import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
#np.random.seed(213)

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

    def __init__(self, learning_rate=0.1, num_inputs=2, num_hidden=2, num_outputs=1):
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

    def train(self, inputs, targets, epochs=10000):
        """
        Trains the perceptron on the given data for a specified number of epochs.

        Args:
            inputs (np.ndarray): Training data inputs.
            targets (np.ndarray): Training data targets.
            epochs (int, optional): Number of training epochs. Defaults to 100.

        Returns:
            history (dict): Training history containing MSE and classification error per epoch.
        """
        history = {'mse_hidden': [], 'mse_output': [], 'classification_error': []}
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
            history['mse_hidden'].append(mse(hidden_outputs, np.dot(inputs, self.weights_hidden)))
            history['mse_output'].append(mse(output, targets))
            history['classification_error'].append(np.mean(classify(output) != targets))

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


# Define training data (XOR problem)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
import time
learning_rate = 0.05
start = time.time()
# Create and train the perceptron
perceptron = Perceptron(learning_rate=learning_rate)
history = perceptron.train(inputs, targets, epochs=50000)
end = time.time()

# Plot MSE and classification error
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(history['mse_hidden'], label='Hidden Layer MSE')
plt.plot(history['mse_output'], label='Output Layer MSE')
plt.title('Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.subplot(212)
plt.plot(history['classification_error'])
plt.xlabel('Epoch')
plt.ylabel('Classification Error')
plt.title('Classification Error over Training')
plt.tight_layout()
plt.show()

# Generate artificial data for prediction
num_samples = 5000
random_data = np.random.randint(0, 2, size=(num_samples, 2))

# Predict on the generated data
predictions = perceptron.predict(random_data)

# Solve XOR problem for the generated data
xor_output = solve_xor(random_data)

# Calculate statistics
true_positives = np.sum((predictions >= 0.5) & (xor_output == 1))
true_negatives = np.sum((predictions < 0.5) & (xor_output == 0))
false_positives = np.sum((predictions >= 0.5) & (xor_output == 0))
false_negatives = np.sum((predictions < 0.5) & (xor_output == 1))

#print statistics
print('Time taken:', end - start)
print('True Positives:', true_positives)
print('True Negatives:', true_negatives)
print('False Positives:', false_positives)
print('False Negatives:', false_negatives)

#print learning rate, accuracy and classification error and number of epochs
print('Learning Rate:', perceptron.learning_rate)
print('Accuracy:', (true_positives + true_negatives) / num_samples)
print('Classification Error:', (false_positives + false_negatives) / num_samples)
print('Number of Epochs:', len(history['classification_error']))


# Labels for the bars
labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']

# Heights of the bars
values = [true_positives, true_negatives, false_positives, false_negatives]

# Create bars
plt.bar(labels, values, color=['green', 'blue', 'red', 'orange'])

# Add title and labels
plt.title('Statistics of Predictions')
plt.xlabel('Outcome')
plt.ylabel('Count')

# Show plot
plt.show()

# Plot the predictions
plt.figure(figsize=(8, 6))
plt.scatter(random_data[:, 0], random_data[:, 1], c=predictions, cmap='coolwarm', label='Predicted Output')
#plt.scatter(random_data[:, 0], random_data[:, 1], c=xor_output, cmap='viridis', marker='x', label='XOR Output')
plt.title('Predictions on Random Data ')
plt.xlabel('Points')
plt.ylabel('Output')
plt.legend()
plt.colorbar(label='Output')
plt.show()


# Define a function to conduct multiple tests
def multiple_tests(num_tests=60, num_epochs=50000, learning_rate=0.05):
    accuracies = []

    for _ in range(num_tests):
        # Generate random dataset for XOR problem
        inputs = np.random.randint(0, 2, size=(4, 2))
        targets = solve_xor(inputs)

        # Create and train the perceptron
        perceptron = Perceptron(learning_rate=learning_rate)
        history = perceptron.train(inputs, targets, epochs=num_epochs)

        # Evaluate accuracy
        predictions = perceptron.predict(inputs)
        accuracy = np.mean((predictions >= 0.5) == targets)
        accuracies.append(accuracy)

    return accuracies

# Perform multiple tests
num_tests = 60
accuracies = multiple_tests(num_tests=num_tests)

# Plot the accuracies
plt.figure(figsize=(8, 6))
plt.bar(range(1, num_tests + 1), accuracies, color='skyblue')
plt.xlabel('Test')
plt.ylabel('Accuracy')
plt.title('Accuracy of Perceptron on XOR Problem ({} tests)'.format(num_tests))
plt.ylim(0, 1)
plt.show()
