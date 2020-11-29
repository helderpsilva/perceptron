import numpy as np
from perceptron import perceptron

# Utilizador deverá escolher:
# 1. Numero de iterações
# 2. Minima alteração do erro
# 3. Probabilidade de o utilizador escolher uma ou outra class



# Dataset
# Gerar cluster aleatório normalmente distribuido  
def sample_generator():
    dados = {}
    dados["x1"] = np.random.normal(loc=3.0, size=25)
    dados["y1"] = np.random.normal(loc=2.0, size=25)
    dados["x2"] = np.random.normal(loc=9.0, size=25)
    dados["y2"] = np.random.normal(loc=7.0, size=25)
    return dados

# dados = sample_generator()
# visualizar dados
# plt.scatter(dados["x1"],dados["y1"],color='#023047', marker="o",s=25)
# plt.scatter(dados["x2"],dados["y2"],color='#fb8500', marker="o",s=25)
# plt.show()




