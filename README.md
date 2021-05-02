<br/>
<p align="center">
        <img width="60%" src="/img/logo.svg" alt="PERCEPTRON">
    </a>
</p>

<br/>


<p align="center">
  <a href="#Instalação">Instalação</a> •
  <a href="#Utilização">Utilização</a> •
  <a href="#Contribuições">Contribuições</a>
</p>

> **Perceptron** também designado por neurônio artificial é uma função matemática de classificação binária (aprendizagem supervisionada). Análogo à unidade básica do sistema nervoso, esta recebe diversos imputs que são processados gerando um único outpout.
> 

<p align="center">
        <img width="65%" src="/img/esquema.svg" alt="Esquema Perceptron">
    </a>
</p>

## Instalação

#### Recomendações
1. Windows ou macOS

Instalar [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) - Utilizar os seguintes comandos para criar um ambiente com todas as bibliotecas necessárias ao projeto.

```bash
# criar o ambiente virtual
$ conda env create -f environment.yaml

# Ativar o ambiente virtual
$ conda activate perceptron

# Remover o ambiente virtual
$ conda env remove -n perceptron
```
Caso ocorra algum problema durante a instalação, deverá recorrer a `pip instal numpy`e `pip instal matplotlib` para instalar as bibliotecas manualmente.

## Utilização

> O **Perceptron** poderá ser testado correndo o comando `python main.py` no diretório do programa. 

Em alternativa, é possível incorporar o módulo `perceptron` no seu código, de acordo com o exemplo apresentado.

```python

import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron 

# Criar uma nova instância do objeto perceptron. 
# n_iterations - numero de épocas de treino do modelo. 
# learning_rate - taxa de aprendizagem. 
# min error - mínima alteracao do erro entre iterações

p = perceptron(n_iterations = 100, learning_rate = 0.01, min_error = 0.01)

# Normalização das variáves (entre 0-1).
normalized_train = p.scale(train_data,0,1)
normalized_test = p.scale(test_data,0,1) 

# Treinar o modelo.
fit = p.fit(normalized_train,train_labels) 

# Gerar previsão.
predicted = p.predict(normalized_test) 

# Visualização gráfica das previsões.
b = np.linspace(min(normalized_test[:,0]), max(normalized_test[:,0]), num=50)
m = (-(p.bias / p.weights[1])/(p.bias / p.weights[0])) * b + (-p.bias / p.weights[1]) 

plt.scatter(normalized_test[:,0] normalized test[:,1], c = predicted, s=25)
plt.plot(b,m) 
plt.show() 

```

#### Bibliotecas utilizadas:
* Numpy
* Matplotlib

## Contribuições
Criado por [Carla M. Lemos](https://github.com/CarlaMLemos) e [Hélder P. Silva ](https://github.com/helderpsilva)

