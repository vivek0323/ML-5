#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[ ]:




