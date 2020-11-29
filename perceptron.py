import numpy as np
import matplotlib.pyplot as plt
import time

# To do:
# Passo 0: Inicializar o vector de peso e o enviesamento com zeros (ou pequenos valores aleatórios).
# Passo 1: Calcular uma combinação linear das características e pesos de entrada.
# Passo 2: Aplicar a função sigmoide, que retorna valores binários
# Passo 3: Calcular actualizações de peso usando as regras do perceptron.
# Passo 4: Actualizar os pesos e o enviesamento.


# Dataset
# Gerar cluster aleatório normalmente distribuido  
def sample_generator():
    dados = {}
    dados["x1"] = np.random.normal(loc=3.0, size=25)
    dados["y1"] = np.random.normal(loc=2.0, size=25)
    dados["x2"] = np.random.normal(loc=9.0, size=25)
    dados["y2"] = np.random.normal(loc=7.0, size=25)
    return dados

dados = sample_generator()

# visualizar dados
plt.scatter(dados["x1"],dados["y1"],color='#023047', marker="o",s=25)
plt.scatter(dados["x2"],dados["y2"],color='#fb8500', marker="o",s=25)
plt.show()