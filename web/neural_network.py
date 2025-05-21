import numpy as np
from scipy.optimize import leastsq

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.A1 = 1 / (1 + np.exp(-self.Z1))
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def cost_function(self, X, y):
        y_hat = self.forward(X)
        return 0.5 * np.sum((y_hat - y) ** 2)

    def cost_function_prime(self, X, y):
        y_hat = self.forward(X)
        delta2 = np.multiply(-(y - y_hat), y_hat * (1 - y_hat))
        dJdW2 = np.dot(self.A1.T, delta2)
        delta1 = np.dot(delta2, self.W2.T) * (self.A1 * (1 - self.A1))
        dJdW1 = np.dot(X.T, delta1)
        return dJdW1, dJdW2

    def flatten_weights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def unflatten_weights(self, flat_weights):
        input_size = self.W1.shape[0]
        hidden_size = self.W1.shape[1]
        self.W1 = flat_weights[:input_size * hidden_size].reshape(input_size, hidden_size)
        self.W2 = flat_weights[input_size * hidden_size:].reshape(hidden_size, 1)

    def train(self, X, y, n_iter=500):
        def error_function(flat_weights, X, y):
            self.unflatten_weights(flat_weights)
            y_hat = self.forward(X)
            return (y_hat - y).ravel()

        initial_weights = self.flatten_weights()
        optimal_weights, success = leastsq(error_function, initial_weights, args=(X, y), maxfev=n_iter)
        self.unflatten_weights(optimal_weights)
