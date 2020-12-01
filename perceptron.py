import numpy as np
import matplotlib.pyplot as plt
import time

# To do:
# Passo 0: Inicializar o vector de peso e o enviesamento com zeros (ou pequenos valores aleatórios).
# Passo 1: Calcular uma combinação linear das características e pesos de entrada.
# Passo 2: Aplicar a função sigmoide, que retorna valores binários
# Passo 3: Calcular actualizações de peso usando as regras do perceptron.
# Passo 4: Actualizar os pesos e o enviesamento.

class perceptron():
    """ Perceptron: Neurónio artificial para classificação binária"""

    def __init__ (self, n_iterations = 100, learning_rate = 0.01):
        # Iniciar pesos e enviesamento.
        self.iterations = n_iterations
        self.learning = learning_rate
        self.activation_function = self.sigmoid
        self.weights= None


    def fit(self, X, y):        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.05
        y_real = np.array(y)

        sum_error = 0.0

        for _ in range(self.iterations):
          for index, xi in  enumerate(X):
            predicted_value = self.predict(xi)
            update = self.learning * (y_real[index] - predicted_value)

            error = y_real[index] - predicted_value
            sum_error = error ** 2
            self.weights += update * xi 
            self.bias += update


    def predict(self, X):
        activation = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(activation)
        return y_predicted       


    def sigmoid(self, X):
         return 1/(1+np.exp(-X))


    def scale(self, X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom

     def __call__(self):
         return [self.weights, self.bias]