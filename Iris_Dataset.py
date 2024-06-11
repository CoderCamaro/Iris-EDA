"""EDA (Exploratory Data Analyis) using the Iris dataset found on the sklearn toy datasets."""

from sklearn.datasets import load_iris 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Loading data about flowers.
iris = load_iris()


# Analyzing the flower features.
X = iris["data"]
print(X.dtype)
print(X.shape)
print(X.size)
print(X.ndim)

print()

# Analyzing the flower types.
y = iris["target"]
print(y.dtype)
print(y.shape)
print(y.size)
print(y.ndim)

# Creating a scatter plot based on speal length and flower type.
plt.scatter(X[:,0],y,c=y)
plt.xlabel("Sepal length (cm)")
plt.ylabel("Flower type")
plt.title("Flower types based on sepal length")
plt.grid()
plt.legend()

plt.figure()

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Flower types based on sepal length and sepal width")
plt.grid()

plt.figure()

plt.scatter(X[:,2],X[:,3],c=y)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Flower types based on petal length and petal width")
plt.grid()

df = pd.DataFrame(X,columns = ["Sepal length(cm)", "Speal width(cm)", "Petal length(cm)", "Petal width(cm)"])

plt.figure()

sns.pairplot(df,corner=True)