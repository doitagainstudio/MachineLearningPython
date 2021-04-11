###################################
#    Coursera Machine Learning    #
#         Python Edition          #
#             Ex. 2a              #
#       Logistic Regression       #
#   coded by Cristiano Esposito   #
#              v.1.0              #
###################################

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# define plot visualization function
def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='y')
    plt.title("Exams Results")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(['Admitted', 'Not Admitted'])
    plt.show()


# define function to plot decision boundary
def plotDecisionBoundary(theta, X, y):
    # compute values for decision boundary line
    x_values = [np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]
    y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary', c='g')
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='y', label='Not admitted')
    plt.title("Exams Results")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


# define Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# define Cost Function and Gradient for Logistic Regression
def lrcostFunction(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = 1 / m * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return J


# define Gradient for Logistic Regression
def gradient(theta, X, y):
    m = len(y)
    return (1 / m) * np.dot(X.T, (sigmoid(np.dot(X, theta))) - y)


# define prediction function
def predict(theta, X):
    h = sigmoid(np.dot(X, theta))
    p = (h >= 0.5)
    return p


# import data from external file
X1, X2, y = np.loadtxt('./data/ex2data1.txt', delimiter=',', unpack=True)
X1 = X1.reshape(-1, 1)
X2 = X2.reshape(-1, 1)
y = y.reshape(-1, 1)
X = np.concatenate((X1, X2), axis=1)

# show scatter plot
plotData(X, y)

# define size of matrix X
(m, n) = np.shape(X)

# add intercept to training data
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Compute cost with initial theta (zeros)
initial_theta = np.zeros(((n + 1), 1))
cost = lrcostFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Gradient at initial theta (zeros): \n', grad)
print('Cost at initial theta (zeros): ', cost)

# Compute cost with non-zero test theta
test_theta = [[-24], [0.2], [0.2]]
cost = lrcostFunction(test_theta, X, y)
grad = gradient(test_theta, X, y)
print('Cost at test theta: ', cost)
print('Gradient at test theta: \n', grad)

# Compute optimal theta using optimization function fmin_tnc
# where func is our cost function and fprime is the gradient function
result = opt.fmin_tnc(func=lrcostFunction, fprime=gradient, x0=initial_theta, args=(X, y.flatten()))
theta = result[0]
print('Optimal theta found: \n', theta)
theta_opt = theta.reshape(-1, 1)
print('Cost for optimal theta is: \n', lrcostFunction(theta_opt, X, y))

# Ask user to input new exam results to predict an admission
print('Insert new exam scores to predict admission.')
exam1_new = int(input('Exam1 score: \n'))
exam2_new = int(input('Exam2 score: \n'))
scores = np.array((1, exam1_new, exam2_new))
prob = sigmoid(np.dot(scores, theta))
print('For a student with score ', exam1_new, ' on exam 1 and score ', exam2_new, ' on exam 2, we predict an '
                                                                                  'admission probability of ',
      round(prob * 100, 2), '%')

# Check for train accurancy
p = predict(theta, X)
print('Train Accurancy: ', np.mean((p == y.flatten()) * 100))

# Plot DecisionBoundary
plotDecisionBoundary(theta, X[:, 1:3], y)
