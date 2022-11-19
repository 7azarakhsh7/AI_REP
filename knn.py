import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

iris = load_iris()
# print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:, 0], iris.data[:, 3], c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[3])
plt.show()

p = iris.data
q = iris.target

print(p.shape)
print(q.shape)

p_train, p_test, q_train, q_test = train_test_split(p, q, test_size=0.2)

print(p_train.shape)
print(p_test.shape)
print(q_train.shape)
print(q_test.shape)

k_range = range(1, 10)
scores = {}
scores_list = []
for k in range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(p_train, q_train)
    q_pred = knn.predict(p_test)
    scores[k] = metrics.accuracy_score(q_test, q_pred)
    sc = metrics.accuracy_score(q_test, q_pred)
    scores_list.append(sc)
fin_max = max(scores, key=scores.get)
print(fin_max)

print(scores)
plt.plot(k_range, scores_list)
plt.xlabel('value of k for knn')
plt.ylabel('testing accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=fin_max)
knn.fit(p, q)

# 0 = setosa , 1 = versicolor , 2 = virginica

classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
x_new = [[3, 4, 5, 2],
         [5, 4, 2, 2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])

print("accuracy is :", scores[fin_max])
