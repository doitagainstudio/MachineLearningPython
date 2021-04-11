###################################
#    Coursera Machine Learning    #
#          Python Edition         #
#              Ex. 1              #
#        Linear Regression        #
#   coded by Cristiano Esposito   #
#              v.1.0              #
###################################

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# define function to normalize values
def normalize(X):
    mu = X.mean()  # media
    sigma = X.std()  # deviazione standard
    X_norm = (X - X.mean()) / X.std()
    return mu, sigma, X_norm


# define Cost Function
def costFunction(X, y, theta):
    m = len(y)
    predictions = X @ theta
    sqrErrors = np.power((predictions - y), 2)
    J = (1 / (2 * m)) * np.sum(sqrErrors)
    return J


# define Gradient Descent function
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X @ theta
        theta = theta - ((alpha / m) * (X.T @ (predictions - y)))
        J = costFunction(X, y, theta)
        J_history.append(J)
    return theta, J_history


# define Normal Equation
def normalEqn(X, y):
    theta = ((np.linalg.inv(X.T @ X)) @ X.T) @ y
    return theta


# read data from external file
df = pd.read_csv('./data/ex1data2.txt', header=None)
X_start = df.iloc[:, 0:2]
y = df.iloc[:, [2]]

# wrangling data for initial cost function
mu, sigma, X_norm = normalize(X_start)
X_norm.insert(0, '', 1)
X = X_norm.values
y = y.values
n = X.shape[1]
theta = np.zeros((n, 1))
print('Initial cost: \n', costFunction(X, y, theta), '\n')

# ask user to input learning rate alpha and number of iterations for Gradient Descent
alpha = float(input('Input learning rate alpha: \n'))
num_iters = int(input('Input number of iterations: \n'))
# compute theta using Gradient Descent
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
print('Theta found using Gradient Descent: \n', theta, '\n')
# plot cost J on iterations
plt.plot(list(range(num_iters)), J_history, '-r')
plt.show()

# compute theta using Normal Equation:
# we should not normalize values
X_nEqn = X_start
X_nEqn.insert(0, '', 1)
X_nEqn = X_nEqn.values
theta_nEqn = normalEqn(X_nEqn, y)
print('Theta found using Normal Eqn: \n', theta_nEqn, '\n')

# ask user to input new values for a price prediction
size = float(input('Input house size: \n'))
nRooms = int(input('Input number of rooms: \n'))
X_predict = np.array((size, nRooms))
# normalize values using previous mu and sigma
X_predictNorm = np.divide((np.subtract(X_predict, mu)), sigma)
X_predictNorm = X_predictNorm.values
# adding intercept to values array
X_predictNorm = np.insert(X_predictNorm, 0, 1)  # for Gradient Descent
X_pred = np.insert(X_predict, 0, 1)  # for Normal Equation

# Compute price prediction
y_predictGrad = np.round(X_predictNorm @ theta, 2)
for item in y_predictGrad:
    print("Price prediction using Gradient Discent: \n", item)

y_predictNormEqn = np.round(X_pred @ theta_nEqn, 2)
for item in y_predictNormEqn:
    print("Price prediction using Normal Equation: \n", item)
