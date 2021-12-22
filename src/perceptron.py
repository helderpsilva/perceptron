# Importação das Livrarias
import numpy as np
import matplotlib.pyplot as plt

# TODO:
# Passo 0: Inicializar o vector de peso e o enviesamento com zeros (ou pequenos valores aleatórios).
# Passo 1: Calcular uma combinação linear das características e pesos de entrada.
# Passo 2: Aplicar a função sigmoide, que retorna valores binários
# Passo 3: Calcular actualizações de peso usando as regras do perceptron.
# Passo 4: Actualizar os pesos e o enviesamento.


class perceptron:
    """
    Perceptron: Nerónio artificial para classificação binária
    ...
    Parâmetros:

    n_interações -- número de épocas do algoritmo
    learning_rate -- taxa de aprendizagem
    min_error -- mínima alteração de erro

    """

    def __init__(self, n_iterations=100, learning_rate=0.01, min_error=0):
        # Iniciar pesos e enviesamento.
        self.iterations = n_iterations
        self.learning = learning_rate
        self.activation_function = self.sigmoid
        self.weights = None
        self.min_error = min_error
        # self.prob = 0.5

    def fit(self, X, y):
        # Função de Treino
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.5
        y_real = np.array(y)

        count = 0
        sum_error = 0
        error_list = []
        error_dif = 0

        for _ in range(self.iterations):
            for index, xi in enumerate(X):
                predicted_value = self.predict(xi)
                update = self.learning * (y_real[index] - predicted_value)

                self.weights += update * xi
                self.bias += update

                error = y_real[index] - predicted_value
                sum_error += error ** 2

            error_list.append(sum_error)
            try:
                error_dif = abs(error_list[-2] - error_list[-1])
            except:
                error_dif = error_list[0]
                pass
            count += 1

            print(f"Iteração n. {count} com erro de {error_dif}")
            if error_dif <= self.min_error:
                break

    def predict(self, X):
        # Função de Previsão
        activation = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(activation)
        # y_predicted = np.where(self.activation_function(activation)>self.prob,1,0)
        return y_predicted

    def sigmoid(self, X):
        # Função de Ativação (0-1)
        return 1 / (1 + np.exp(-X))

    def scale(self, X, x_min, x_max):
        # Função de normalização dos dados.
        nom = (X - X.min(axis=0)) * (x_max - x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom == 0] = 1
        return x_min + nom / denom
