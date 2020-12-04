import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron


# Dataset
# Gerar cluster aleatório normalmente distribuido  

def sample_generator(sample_size=20):
    x = (np.random.normal(loc=3.0, size=int(sample_size/2)).tolist()) + (np.random.normal(loc=9.0, size=int(sample_size/2)).tolist())
    y = (np.random.normal(loc=2.0, size=int(sample_size/2)).tolist()) + (np.random.normal(loc=7.0, size=int(sample_size/2)).tolist())
    label = [1 if i > int(sample_size/2-1)  else 0 for i in range(sample_size)]
    return x, y, label

sample = 100
x,y,label = sample_generator(sample)

data = np.column_stack((x, y))
data_label = np.array(label)


# Interação com o utilizador:
# 1. Numero de iterações
# 2. Minima alteração do erro
# 3. Probabilidade do utilizador escolher uma ou outra classe

def user_imputs():
# O programa corre o máximo número de interações ou o menor erro, consoante o que ocorrer primeiro
    print ("\n 1.Número de iterações (default = 100)") 
    n_iterations = input()
    if n_iterations == "":
        no_compliance = False
        n_iterations = 100
    else:
        no_compliance = True
    while no_compliance:
        try:
            n_iterations = int(n_iterations)
            no_compliance = False
            break
        except ValueError:
            print("Por favor, indique um número inteiro")
            n_iterations = input()
    
    print ("2. Mínima alteração de erro (default = 0)") 
    min_error = input()
    if min_error == "":
        no_compliance = False
        min_error = 0
    else:
        no_compliance = True
    while no_compliance:
        try:
            min_error = float(min_error)
            no_compliance = False
            break
        except ValueError:
            print("Por favor, indique um número")
            min_error = input()
    
    print ("3. Probabilidade a partir da qual o algoritmo escolhe uma ou outra classe (default = 0.5)")
    prob = input()
    if prob == "":
        no_compliance = False
        prob = 0.5
    else:
        no_compliance = True
    while no_compliance:
        try:
            prob = float(prob)
            if prob < 0 or prob > 1:
                raise ValueError
            else:
                no_compliance = False
                break
            break 
        except ValueError:
            print("Por favor, indique um número entre 0 e 1")
            prob = input()

    return n_iterations, min_error, prob

n_iterations, min_error, prob = user_imputs()

# Inicializar uma nova instância do perceptron com os dados obtidos do utilizador.
p = perceptron(n_iterations,0.01, min_error)

# Normalização dos dados.
standard_data = p.scale(data,0,1)

# Treino do modelo.
fitted = p.fit(standard_data, data_label)

# Gerar previsão.
predicted = p.predict(standard_data)
predicted_adjusted = [1 if predicted[i] > prob else 0 for i in range(sample)]

# Gerar visualização gráfica da previsão obtida.
x = np.linspace(min(standard_data[:,0]), max(standard_data[:,0]), num=50)
m = (-(p.bias / p.weights[1])/(p.bias / p.weights[0]))*x + (- p.bias / p.weights[1])


plt.scatter(standard_data[:,0], standard_data[:,1],c=predicted_adjusted, marker="o",s=25)
plt.plot(x,m)
plt.show()

