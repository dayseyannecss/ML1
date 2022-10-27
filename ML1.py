from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split

iris = datasets.load_iris()
digits = datasets.load_digits()
print("Iris dataset")

print(iris.target_names)
print(iris.data)

#IMPRIMIR DESCRIÇÃO DO DATASET IRIS
print(iris['DESCR'])

#MPRIMIR resultado  DO DATASET IRIS
print(iris['target'])

print("Digito dataset")
print(digits.data)

#IMPRIMIR DESCRIÇÃO DO DATASET DIGITOS

print(digits['DESCR'])

#MPRIMIR resultado  DO DATASET DIGITOS
print(digits['target'])

#Dividindo o dataset IRIS
X_trainiRIS, X_testiRIS, y_trainIRIS, y_testIRIS = train_test_split(iris["data"], iris['target'], random_state=0)

print("Treino Iris: {}".format(X_trainiRIS.shape))
print("Teste Iris: {}".format(X_testiRIS.shape))

plt.xlabel('x - Features')

plt.ylabel('y - Target')

plt.title('Gráfico Iris')

#soma dos features é igual a featuresall

featuresAll=[]

for observation in iris.data:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])

#Plotando o gráfico de dispersão (Relação entre comprimento e largura sépala)
plt.plot(featuresAll, iris.target)

plt.show()

#Gráfico de Dispersão com Conjunto de Dados Iris (Relação entre o Comprimento e a Largura da Pétala)



#Dividindo o dataset Digito
X_trainDig, X_testDig, y_trainDig, y_testDig = train_test_split(digits["data"], digits['target'], random_state=0)


print("Treino digits: {}".format(X_trainDig.shape))
print("Teste digits: {}".format(X_testDig.shape))

plt.xlabel('x - axis')

plt.ylabel('y - axis')

plt.title('Gráfico Digits')

plt.plot(digits.data, digits.target)

plt.show()