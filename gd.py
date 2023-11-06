import numpy as np

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 arrays,size of each array - 1
y = 4 + 3 * X + np.random.randn(100, 1)  # 100 by 1 array of samples with normal distribution with mean 4 and sd 3x
theta = np.random.randn(2, 1)  # initilaizing weight(slope) and bias(intercept) randomly.


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)



def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    history = []

    for _ in range(iterations):
        y_pred = X.dot(theta)
        loss = mean_squared_error(y, y_pred)
        gradient = (1/m) * X.T.dot(y_pred - y) # taking partial derivative
        theta = theta - learning_rate * gradient
        history.append(loss)

    return theta, history

X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
learning_rate = 0.01
iterations = 1000

theta_final, loss_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

print("Final Parameters (Theta):", theta_final)


import matplotlib.pyplot as plt

plt.plot(range(iterations), loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

