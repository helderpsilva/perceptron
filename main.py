import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron

# Utilizador deverá escolher:
# 1. Numero de iterações
# 2. Minima alteração do erro
# 3. Probabilidade de o utilizador escolher uma ou outra class

# Dataset
# Gerar cluster aleatório normalmente distribuido  
def sample_generator(sample_size=20):
    x = (np.random.normal(loc=3.0, size=int(sample_size/2)).tolist()) + (np.random.normal(loc=9.0, size=int(sample_size/2)).tolist())
    y = (np.random.normal(loc=2.0, size=int(sample_size/2)).tolist()) + (np.random.normal(loc=7.0, size=int(sample_size/2)).tolist())
    label = [1 if i > int(sample_size/2-1)  else 0 for i in range(sample_size)]
    return x, y, label

sample = 350
x,y,label = sample_generator(sample)

data = np.column_stack((x, y))
data_label = np.array(label)


p = perceptron(8,0.01)
standard_data = p.scale(data,0,1)
p.fit(standard_data, data_label)
predicted = p.predict(standard_data)

predicted_adjusted = [1 if predicted[i] > 0.5 else 0 for i in range(sample)]

plt.scatter(data[:,0], data[:,1],c=predicted_adjusted, marker="o",s=25)
plt.show()

# dados = sample_generator()
# visualizar dados
# plt.scatter(dados["x1"],dados["y1"],color='#023047', marker="o",s=25)
# plt.scatter(dados["x2"],dados["y2"],color='#fb8500', marker="o",s=25)
# plt.show()




