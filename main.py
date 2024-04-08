import numpy as np
import matplotlib.pyplot as plt
import time
from perceptron_functions import Perceptron, solve_xor, perform_tests

# Part 1: Data Creation
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])


# Part 2: Perceptron Training and Evaluation
learning_rate = 1
epochs = 500
start = time.time()

perceptron = Perceptron(learning_rate=learning_rate)
history = perceptron.train(inputs, targets, epochs=epochs)

end = time.time()

predictions = perceptron.predict(inputs)
xor_output = solve_xor(inputs)

print('inputs', inputs)
print('Predictions:', predictions)

true_positives = np.sum((predictions >= 0.5) & (xor_output == 1))
true_negatives = np.sum((predictions < 0.5) & (xor_output == 0))
false_positives = np.sum((predictions >= 0.5) & (xor_output == 0))
false_negatives = np.sum((predictions < 0.5) & (xor_output == 1))

print('Time taken:', end - start)
print('True Positives:', true_positives)
print('True Negatives:', true_negatives)
print('False Positives:', false_positives)
print('False Negatives:', false_negatives)

print('Learning Rate:', perceptron.learning_rate)
print('Accuracy:', (true_positives + true_negatives) / 4)
print('Classification Error:', (false_positives + false_negatives) / 4)
print('Number of Epochs:', len(history['classification_error']))

# Part 3: Plotting
plt.figure(figsize=(10, 12))

# Plot MSE
plt.subplot(311)
plt.plot(history['mse_hidden'], label='Hidden Layer MSE')
plt.title('Mean Squared Error over all inputs in hidden layer')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.subplot(312)
plt.plot(history['mse_output'], label='Output Layer MSE')
plt.title('Mean Squared Error Output Layer')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

# Plot classification error
plt.subplot(313)
plt.plot(history['classification_error'])
plt.xlabel('Epoch')
plt.ylabel('Classification Error')
plt.title('Classification Error over Training')

# Plot predictions
# plt.subplot(313)
# plt.scatter(random_data[:, 0], random_data[:, 1], c=predictions, cmap='coolwarm', label='Predicted Output')
# plt.title('Predictions on Random Data')
# plt.xlabel('Points')
# plt.ylabel('Output')
# plt.legend()
# plt.colorbar(label='Output')


plt.tight_layout()
plt.show()

# Perform multiple tests
num_tests = 10
accuracies = perform_tests(num_tests=num_tests, num_epochs=epochs, learning_rate=learning_rate)

# Plot the accuracies
plt.figure(figsize=(8, 6))
plt.bar(range(1, num_tests + 1), accuracies, color='skyblue')
plt.xlabel('Test')
plt.ylabel('Accuracy')
plt.title('Accuracy of Perceptron on XOR Problem ({} tests)'.format(num_tests))
plt.ylim(0, 1)
plt.show()
