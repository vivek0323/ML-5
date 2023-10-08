#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

# weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

#  Step activation function
def step_activation(z):
    return 1 if z >= 0 else 0

#  variables for tracking epochs and errors
epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        # weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  predicted output
        predicted = step_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for AND Gate Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

# Bi-Polar Step activation function
def bipolar_step_activation(z):
    return 1 if z > 0 else -1

#  variables for tracking epochs and errors
epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        #  weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  predicted output
        predicted = bipolar_step_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for Bi-Polar Step Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")



# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

#  Sigmoid activation function
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

#  variables for tracking epochs and errors
epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        #  the weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  the predicted output
        predicted = sigmoid_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for Sigmoid Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

#  ReLU activation function
def relu_activation(z):
    return max(0, z)

#  variables for tracking epochs and errors
epochs = 0
errors = []

while True:
    error_sum = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        
        #  the weighted sum
        z = W0 + W1 * xi[0] + W2 * xi[1]
        
        #  the predicted output
        predicted = relu_activation(z)
        
        #  the error
        error = target - predicted
        error_sum += error ** 2
        
        # Update weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # Check for convergence condition or maximum epochs
    if error_sum <= 0.002 or epochs >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for ReLU Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# AND gate input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND gate output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75

#  Bi-Polar Step activation function
def bipolar_step_activation(z):
    return 1 if z > 0 else -1

# List of learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#  a list to store the number of iterations for each learning rate
iterations_needed = []

for learning_rate in learning_rates:
    W0, W1, W2 = 10, 0.2, -0.75
    
    #  variables for tracking epochs and errors
    epochs = 0
    
    while True:
        error_sum = 0
        for i in range(len(X)):
            xi = X[i]
            target = y[i]
            
            #  weighted sum
            z = W0 + W1 * xi[0] + W2 * xi[1]
            
            #  predicted output
            predicted = bipolar_step_activation(z)
            
            #  error
            error = target - predicted
            error_sum += error ** 2
            
            #  weights and bias
            W0 += learning_rate * error
            W1 += learning_rate * error * xi[0]
            W2 += learning_rate * error * xi[1]
        
        epochs += 1
        
        # Check for convergence condition or maximum epochs
        if error_sum <= 0.002 or epochs >= 1000:
            break
    
    iterations_needed.append(epochs)

# Plot learning rates vs. iterations needed
plt.plot(learning_rates, iterations_needed, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations Needed for Convergence')
plt.title('Learning Rate vs. Iterations for Convergence')
plt.grid(True)
plt.show()

# Print the number of iterations needed for each learning rate
for i, rate in enumerate(learning_rates):
    print(f"Learning Rate {rate}: Iterations Needed = {iterations_needed[i]}")


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# XOR gate input 
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR gate output 
y_xor = np.array([0, 1, 1, 0])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
alpha = 0.05

# Max number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epochs = []
errors = []

for epoch in range(max_epochs):
    error_sum = 0.0
    for i in range(len(X_xor)):
        # Calculate the weighted sum
        weighted_sum = W0 + W1 * X_xor[i][0] + W2 * X_xor[i][1]
        
        # Apply Step activation function
        if weighted_sum > 0:
            prediction = 1
        else:
            prediction = 0
        
        error = y_xor[i] - prediction
        
        # Update weights and bias
        W0 += alpha * error
        W1 += alpha * error * X_xor[i][0]
        W2 += alpha * error * X_xor[i][1]
        
        # Add squared error to the sum
        error_sum += error ** 2
    
    # Append epoch and error values for plotting
    epochs.append(epoch)
    errors.append(error_sum / len(X_xor))
    
    # Check for convergence
    if error_sum / len(X_xor) <= 0.002:
        break

# Plot epochs against error values
plt.plot(epochs, errors, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Epochs vs. Error for XOR Gate')
plt.grid(True)
plt.show()

# Print the converged weights
print("Converged Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# XOR gate input data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR gate output data
y_xor = np.array([0, 1, 1, 0])

# Initialize weights and bias
W0, W1, W2 = 10, 0.2, -0.75

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epochs = []
errors = []

for epoch in range(max_epochs):
    error_sum = 0.0
    for i in range(len(X_xor)):
        # Calculate the weighted sum
        weighted_sum = W0 + W1 * X_xor[i][0] + W2 * X_xor[i][1]
        
        # Apply Bi-Polar Step activation function
        if weighted_sum > 0:
            prediction = 1
        else:
            prediction = -1
        
        # Calculate error
        error = y_xor[i] - prediction
        
        # Update weights and bias
        W0 += alpha * error
        W1 += alpha * error * X_xor[i][0]
        W2 += alpha * error * X_xor[i][1]
        
        # Add squared error to the sum
        error_sum += error ** 2
    
    # Append epoch and error values for plotting
    epochs.append(epoch)
    errors.append(error_sum / len(X_xor))
    
    # Check for convergence
    if error_sum / len(X_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("Bi-Polar Step Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epochs, errors)
plt.title('Epochs vs. Error (Bi-Polar Step Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# XOR gate input data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR gate output data
y_xor = np.array([0, 1, 1, 0])

# Initialize weights and bias
W0, W1, W2 = 10, 0.2, -0.75

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epochs = []
errors = []

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(max_epochs):
    error_sum = 0.0
    for i in range(len(X_xor)):
        # Calculate the weighted sum
        weighted_sum = W0 + W1 * X_xor[i][0] + W2 * X_xor[i][1]
        
        # Apply Sigmoid activation function
        prediction = sigmoid(weighted_sum)
        
        # Calculate error
        error = y_xor[i] - prediction
        
        # Update weights and bias
        delta = alpha * error * prediction * (1 - prediction)
        W0 += delta
        W1 += delta * X_xor[i][0]
        W2 += delta * X_xor[i][1]
        
        # Add squared error to the sum
        error_sum += error ** 2
    
    # Append epoch and error values for plotting
    epochs.append(epoch)
    errors.append(error_sum / len(X_xor))
    
    # Check for convergence
    if error_sum / len(X_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("Sigmoid Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epochs, errors)
plt.title('Epochs vs. Error (Sigmoid Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# XOR gate input data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR gate output data
y_xor = np.array([0, 1, 1, 0])

# Initialize weights and bias
W0, W1, W2 = 10, 0.2, -0.75

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epochs = []
errors = []

# ReLU activation function
def relu(x):
    return max(0, x)

for epoch in range(max_epochs):
    error_sum = 0.0
    for i in range(len(X_xor)):
        # Calculate the weighted sum
        weighted_sum = W0 + W1 * X_xor[i][0] + W2 * X_xor[i][1]
        
        # Apply ReLU activation function
        prediction = relu(weighted_sum)
        
        # Calculate error
        error = y_xor[i] - prediction
        
        # Update weights and bias
        delta = alpha * error
        W0 += delta
        W1 += delta * X_xor[i][0]
        W2 += delta * X_xor[i][1]
        
        # Add squared error to the sum
        error_sum += error ** 2
    
    # Append epoch and error values for plotting
    epochs.append(epoch)
    errors.append(error_sum / len(X_xor))
    
    # Check for convergence
    if error_sum / len(X_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("ReLU Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epochs, errors)
plt.title('Epochs vs. Error (ReLU Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[13]:


import numpy as np

# Initialize weights and bias with random values
W_candies, W_mangoes, W_milk_packets, bias = np.random.rand(4)

# Learning rate
alpha = 0.1

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input features (Candies, Mangoes, Milk Packets)
X = np.array([
    [20, 6, 1],
    [16, 3, 2],
    [27, 9, 3],
    [19, 11, 0],
    [24, 8, 2],
    [15, 12, 1],
    [15, 4, 2],
    [18, 8, 2],
    [21, 1, 4],
    [24, 19, 8]
])

# Corresponding target labels (High Value Tx)
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Training the perceptron
for _ in range(1000):  # Adjust the number of epochs as needed
    total_error = 0
    for i in range(len(X)):
        # Compute the weighted sum of inputs
        weighted_sum = (
            W_candies * X[i][0] +
            W_mangoes * X[i][1] +
            W_milk_packets * X[i][2] +
            bias
        )
        
        # Apply sigmoid activation function
        prediction = sigmoid(weighted_sum)
        
        # Calculate the error
        error = y[i] - prediction
        total_error += error ** 2
        
        # Update weights and bias
        W_candies += alpha * error * prediction * (1 - prediction) * X[i][0]
        W_mangoes += alpha * error * prediction * (1 - prediction) * X[i][1]
        W_milk_packets += alpha * error * prediction * (1 - prediction) * X[i][2]
        bias += alpha * error
    
    # Check for convergence (adjust the error threshold as needed)
    if total_error < 0.01:
        break

# Classify new data point
def classify(candies, mangoes, milk_packets):
    weighted_sum = (
        W_candies * candies +
        W_mangoes * mangoes +
        W_milk_packets * milk_packets +
        bias
    )
    prediction = sigmoid(weighted_sum)
    return "High Value" if prediction >= 0.5 else "Low Value"

# Example 
new_transaction = [18, 7, 3]
result = classify(*new_transaction)
print(f"New transaction {new_transaction} is classified as {result}")


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Input features (Candies, Mangoes, Milk Packets)
X = np.array([
    [20, 6, 1],
    [16, 3, 2],
    [27, 9, 3],
    [19, 11, 0],
    [24, 8, 2],
    [15, 12, 1],
    [15, 4, 2],
    [18, 8, 2],
    [21, 1, 4],
    [24, 19, 8]
])

# Corresponding target labels (High Value Tx)
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Create and train a perceptron model
class Perceptron:
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_function(linear_output)

                # Update weights and bias
                self.weights += self.learning_rate * (y[i] - prediction) * X[i]
                self.bias += self.learning_rate * (y[i] - prediction)

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            predictions.append(self.step_function(linear_output))
        return np.array(predictions)

perceptron = Perceptron()
perceptron.fit(X, y)

# Create and train a logistic regression model
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(X, y)

# Generate predictions for each transaction
perceptron_predictions = perceptron.predict(X)
logistic_regression_predictions = logistic_regression.predict(X)

# Calculate accuracy for both models
perceptron_accuracy = accuracy_score(y, perceptron_predictions)
logistic_regression_accuracy = accuracy_score(y, logistic_regression_predictions)

# Plot transactions with actual and predicted labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title(f'Actual Labels (Perceptron Accuracy: {perceptron_accuracy:.2f})')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=perceptron_predictions, cmap=plt.cm.Paired)
plt.title(f'Predicted Labels (Perceptron Accuracy: {perceptron_accuracy:.2f})')

plt.show()


# In[15]:


import numpy as np

# AND gate input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate output data
y = np.array([0, 0, 0, 1])

# Initialize weights and biases
np.random.seed(0)
input_size = 2
hidden_size = 2
output_size = 1

# Weights and biases for the input layer to the hidden layer
W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.zeros(hidden_size)

# Weights and biases for the hidden layer to the output layer
W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.zeros(output_size)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hyperparameters
learning_rate = 0.05
max_iterations = 1000
convergence_error = 0.002

# Training the neural network
for i in range(max_iterations):
    # Forward propagation
    layer1_input = np.dot(X, W1) + b1
    layer1_output = sigmoid(layer1_input)

    layer2_input = np.dot(layer1_output, W2) + b2
    layer2_output = sigmoid(layer2_input)

    # Calculate error
    error = y.reshape(-1, 1) - layer2_output

    # Backpropagation
    delta2 = error * sigmoid_derivative(layer2_output)
    dW2 = np.dot(layer1_output.T, delta2)
    db2 = np.sum(delta2, axis=0)

    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(layer1_output)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0)

    # Update weights and biases
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
    W1 += learning_rate * dW1
    b1 += learning_rate * db1

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {i + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, W1) + b1), W2) + b2)
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[16]:


import numpy as np

# XOR gate input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR gate output data
y = np.array([[0], [1], [1], [0]])

# Hyperparameters
learning_rate = 0.05
input_size = 2
hidden_size = 2
output_size = 1
max_iterations = 10000
convergence_error = 0.002

# Initialize weights with small random values
np.random.seed(0)
W1 = 2 * np.random.random((input_size, hidden_size)) - 1
W2 = 2 * np.random.random((hidden_size, output_size)) - 1

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation
for i in range(max_iterations):
    # Forward propagation
    layer1_input = np.dot(X, W1)
    layer1_output = sigmoid(layer1_input)

    layer2_input = np.dot(layer1_output, W2)
    layer2_output = sigmoid(layer2_input)

    # Calculate errors
    error2 = y - layer2_output
    error2_delta = error2 * sigmoid_derivative(layer2_output)

    error1 = error2_delta.dot(W2.T)
    error1_delta = error1 * sigmoid_derivative(layer1_output)

    # Update weights
    W2 += layer1_output.T.dot(error2_delta) * learning_rate
    W1 += X.T.dot(error1_delta) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error2))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {i + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, W1)), W2))
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[17]:


import numpy as np

# XOR gate input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR gate output data (two nodes for each output)
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Hyperparameters
learning_rate = 0.05
input_size = 2
hidden_size = 2
output_size = 2  # Two output nodes
max_iterations = 10000
convergence_error = 0.002

# Initialize weights with small random values
np.random.seed(0)
W1 = 2 * np.random.random((input_size, hidden_size)) - 1
W2 = 2 * np.random.random((hidden_size, output_size)) - 1

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation
for i in range(max_iterations):
    # Forward propagation
    layer1_input = np.dot(X, W1)
    layer1_output = sigmoid(layer1_input)

    layer2_input = np.dot(layer1_output, W2)
    layer2_output = sigmoid(layer2_input)

    # Calculate errors
    error2 = y - layer2_output
    error2_delta = error2 * sigmoid_derivative(layer2_output)

    error1 = error2_delta.dot(W2.T)
    error1_delta = error1 * sigmoid_derivative(layer1_output)

    # Update weights
    W2 += layer1_output.T.dot(error2_delta) * learning_rate
    W1 += X.T.dot(error1_delta) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error2))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {i + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, W1)), W2))
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Define the AND gate input and output data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Define the XOR gate input and output data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create an MLPClassifier for the AND gate
and_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000)

# Create an MLPClassifier for the XOR gate
xor_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000)

# Train the AND gate classifier
and_classifier.fit(X_and, y_and)

# Train the XOR gate classifier
xor_classifier.fit(X_xor, y_xor)

# Predict for AND gate inputs
and_predictions = and_classifier.predict(X_and)

# Predict for XOR gate inputs
xor_predictions = xor_classifier.predict(X_xor)

# Confusion matrix for AND gate
cm_and = confusion_matrix(y_and, and_predictions)

# Confusion matrix for XOR gate
cm_xor = confusion_matrix(y_xor, xor_predictions)

# Define a function to plot the decision boundary
def plot_decision_boundary(classifier, X, y, title):
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, marker='o')
    plt.title(title)
    plt.show()

# Plot decision boundary for AND gate
plot_decision_boundary(and_classifier, X_and, y_and, "AND Gate Decision Boundary")

# Plot decision boundary for XOR gate
plot_decision_boundary(xor_classifier, X_xor, y_xor, "XOR Gate Decision Boundary")

# Calculate accuracy for AND and XOR gates
accuracy_and = accuracy_score(y_and, and_predictions)
accuracy_xor = accuracy_score(y_xor, xor_predictions)

print("AND Gate Accuracy:", accuracy_and)
print("XOR Gate Accuracy:", accuracy_xor)


# In[19]:


pip install pandas scikit-learn


# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load training data from the uploaded Excel file
train_data = pd.read_excel('training (2) (1).xlsx')

# Drop rows with missing 'input'
train_data.dropna(subset=['input'], inplace=True)

# Load testing data from the uploaded Excel file
test_data = pd.read_excel('testing (2) (1).xlsx')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training text data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['input'])  # Use 'input' or the correct column name in training

# Transform the testing text data
X_test_tfidf = tfidf_vectorizer.transform(test_data['Equation'])  # Use 'Equation' or the correct column name in testing

# Define the labels
y_train = train_data['Classification']
y_test = test_data['Classification']

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='grey')
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




