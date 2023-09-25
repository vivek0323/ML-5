#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# AND  input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND  output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

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
        
        # predicted output
        predicted = step_activation(z)
        
        # the error
        error = target - predicted
        error_sum += error ** 2
        
        # weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    # convergence condition or maximum epochs
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

# AND  input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# AND  output 
y = np.array([0, 0, 0, 1])

#  weights and bias
W0, W1, W2 = 10, 0.2, -0.75
learning_rate = 0.05

#  Bi-Polar Step activation function
def bipolar_step_activation(z):
    return 1 if z > 0 else -1

# variables for tracking epochs and errors
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
        
        # the error
        error = target - predicted
        error_sum += error ** 2
        
        #  weights and bias
        W0 += learning_rate * error
        W1 += learning_rate * error * xi[0]
        W2 += learning_rate * error * xi[1]
    
    epochs += 1
    errors.append(error_sum)
    
    #  convergence condition or maximum epochs
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


# In[ ]:




