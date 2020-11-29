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
        self.activatio_function = self.sigmoid
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):        
        pass

    def predict(self, X):
        pass        
    
    def sigmoid(self, X):
        return np.where(1/(1+np.exp(-x)))