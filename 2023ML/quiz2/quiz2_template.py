import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# ========================================================================
# dataset

"""
n_tot = 800
n = int(n_tot/2)
# two moons, not really linearly separable
X, y = make_moons(n_tot, noise=0.15, random_state=0)

# divide data into training and testing
np.random.seed(42)
order = np.random.permutation(n_tot)
train = order[:n]
test = order[n:]

Xtr = X[train, :]
ytr = y[train]
Xtst = X[test, :]
ytst = y[test]

np.save("quiz2_datafiles/Xtr.npy", Xtr)
np.save("quiz2_datafiles/Xtst.npy", Xtst)
np.save("quiz2_datafiles/ytr.npy", ytr)
np.save("quiz2_datafiles/ytst.npy", ytst)
"""

Xtr = np.load("quiz2_datafiles/Xtr.npy")
Xtst = np.load("quiz2_datafiles/Xtst.npy")
ytr = np.load("quiz2_datafiles/ytr.npy")
ytst = np.load("quiz2_datafiles/ytst.npy")
n = len(ytr)

plt.figure()
colors = ["g", "b"]
for (X, y) in [(Xtr, ytr), (Xtst, ytst)]:
    for ii in range(2):
        class_indices = np.where(y==ii)[0]
        plt.scatter(X[class_indices, 0], X[class_indices, 1], c=colors[ii])
plt.title("full dataset")
plt.show()

# ========================================================================
# classifier

# The perceptron algorithm will be encountered later in the course
# How exactly it works is not relevant yet, it's enough to just know it's a binary classifier
from sklearn.linear_model import Perceptron as binary_classifier

# It can be used like this:
bc = binary_classifier()
bc.fit(Xtr, ytr)  # this is how to train the classifier on training data
preds = bc.predict(Xtst)  # this is how to obtain predictions on test data

