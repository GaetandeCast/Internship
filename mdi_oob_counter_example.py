import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X = np.array([
    [0],
    [0],
    [1],
])
y = np.array([
    [0],
    [1],
    [0],
])

clf = DecisionTreeClassifier()
clf.fit(X, y)
plt.figure(figsize=(5,5))
tree.plot_tree(clf)
plt.savefig("mdi_oob_counter_example.png")