from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
digits.target
digits.images[0]
print(digits.target)
print(digits.images[0])
