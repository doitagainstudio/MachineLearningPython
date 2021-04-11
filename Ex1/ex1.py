###################################
#    Coursera Machine Learning    #
#          Python Edition         #
#              Ex. 1              #
#        Linear Regression        #
#   coded by Cristiano Esposito   #
#              v.1.0              #
###################################

# importiamo le librerie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# definiamo la funzione per la normalizzazione dei valori
def normalize(X):
    mu = X.mean()  # media
    sigma = X.std()  # deviazione standard
    X_norm = (X - X.mean()) / X.std()
    return mu, sigma, X_norm


# definiamo la Cost Function
def costFunction(X, y, theta):
    m = len(y)
    predictions = X @ theta
    sqrErrors = np.power((predictions - y), 2)
    J = (1 / (2 * m)) * np.sum(sqrErrors)
    return J


# definiamo la funzione Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X @ theta
        theta = theta - ((alpha / m) * (X.T @ (predictions - y)))
        J = costFunction(X, y, theta)
        J_history.append(J)
    return theta, J_history


# definiamo la Normal Equation
def normalEqn(X, y):
    theta = ((np.linalg.inv(X.T @ X)) @ X.T) @ y
    return theta


# importiamo i dati leggendo da file esterno
df = pd.read_csv('./data/ex1data2.txt', header=None)
X_start = df.iloc[:, 0:2]
y = df.iloc[:, [2]]

# prepariamo i dati per la prima esecuzione della Cost Function
mu, sigma, X_norm = normalize(X_start)
X_norm.insert(0, '', 1)
X = X_norm.values
y = y.values
n = X.shape[1]
theta = np.zeros((n, 1))
print('Costo iniziale: \n', costFunction(X, y, theta), '\n')

# chiediamo all'utente di inserire alpha e numero di iterazioni per il Gradient Discent
alpha = float(input('Inserire il rateo di apprendimento alpha: \n'))
num_iters = int(input('Inserire il numero di iterazioni: \n'))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
print('Calcoliamo i parametri theta con Gradient Descent: \n', theta, '\n')

# raffiguriamo l'andamento del costo in funzione di theta
plt.plot(list(range(num_iters)), J_history, '-r')
plt.show()

# calcoliamo i parametri theta con la Normal Equation:
# in questo caso i dati iniziali NON vanno normalizzati
X_nEqn = X_start
X_nEqn.insert(0, '', 1)
X_nEqn = X_nEqn.values
theta_nEqn = normalEqn(X_nEqn, y)
print('Calcoliamo i parametri theta con Normal Eqn: \n', theta_nEqn, '\n')

# chiediamo all'utente di fornirci nuove variabili iniziali per ottenere una previsione
# con il modello appena creato
size = float(input('Inserire dimensione appartamento: \n'))
nRooms = int(input('Inserire numero di stanze: \n'))
X_predict = np.array((size, nRooms))
# normalizziamo le variabili attraverso i parametri mu e sigma ottenuti in precedenza
X_predictNorm = np.divide((np.subtract(X_predict, mu)), sigma)
X_predictNorm = X_predictNorm.values
# inseriamo i dati dell'intercetto negli array costruiti
X_predictNorm = np.insert(X_predictNorm, 0, 1)  # per Gradient Descent
X_pred = np.insert(X_predict, 0, 1)  # per Normal Equation

# Calcoliamo e visualizziamo le previsioni
y_predictGrad = np.round(X_predictNorm @ theta, 2)
for item in y_predictGrad:
    print("La previsione di prezzo con Gradient Discent è: \n", item)

y_predictNormEqn = np.round(X_pred @ theta_nEqn, 2)
for item in y_predictNormEqn:
    print("La previsione di prezzo con Normal Equation è: \n", item)
