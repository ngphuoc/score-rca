import numpy as np
import matplotlib.pyplot as plt

# Define input range
x = np.linspace(-10, 10, 400)

# Activation function definitions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

def mish(x):
    # Mish activation: x * tanh(softplus(x)), where softplus(x) = ln(1+exp(x))
    return x * np.tanh(np.log1p(np.exp(x)))

# Compute outputs for each activation
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_swish = swish(x)
y_mish = mish(x)

# Plot all activations
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_elu, label='ELU')
plt.plot(x, y_swish, label='Swish')
plt.plot(x, y_mish, label='Mish')

plt.xlabel("Input")
plt.ylabel("Activation")
plt.title("Popular Activation Functions")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("fig/act.png")

