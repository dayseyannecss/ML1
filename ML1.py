from sklearn import datasets
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split

iris = datasets.load_iris()
digits = datasets.load_digits()
print("Iris dataset")

print(iris.target_names)
print(iris.data)

#IMPRIMIR DESCRIÇÃO DO DATASET IRIS
print(iris['DESCR'])


print("Digito dataset")
print(digits.data)

#IMPRIMIR DESCRIÇÃO DO DATASET DIGITOS

print(digits['DESCR'])
