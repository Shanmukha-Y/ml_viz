import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Generate x values
x = np.linspace(-5, 5, 1000)

# Set up the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

# Plot Sigmoid
axs[0].plot(x, sigmoid(x), label='Sigmoid')
axs[0].plot(x, sigmoid_derivative(x), label='Derivative')
axs[0].set_title('Sigmoid')
axs[0].legend()
axs[0].grid(True)

# Plot ReLU
axs[1].plot(x, relu(x), label='ReLU')
axs[1].plot(x, relu_derivative(x), label='Derivative')
axs[1].set_title('ReLU')
axs[1].legend()
axs[1].grid(True)

# Plot Tanh
axs[2].plot(x, tanh(x), label='Tanh')
axs[2].plot(x, tanh_derivative(x), label='Derivative')
axs[2].set_title('Tanh')
axs[2].legend()
axs[2].grid(True)

# Plot Leaky ReLU
axs[3].plot(x, leaky_relu(x), label='Leaky ReLU')
axs[3].plot(x, leaky_relu_derivative(x), label='Derivative')
axs[3].set_title('Leaky ReLU')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()