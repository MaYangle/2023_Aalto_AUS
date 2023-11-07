

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.special import expit  # 导入expit函数

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output

mdata,ndim = X.shape

## to convert the {0,1} output into {-1,+1}
y = 2*y - 1

## hyperparameters of the learning task of Question 3
## list of step sizes


## number of iteration
iteration = 50

nfold = 5         ## number of folds

np.random.seed(12345)  ## fix the random seed to avoid random fluctuation of the results

## to split the data into 5-folds we need
cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)

## normalization
## scaling the rows by maximum absolute value, L infinite norm of columns
X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

# Define an empty list to store the maximum average scores and their corresponding step sizes
max_scores = []
best_etas = []
#define a function to implement logistic regression algorithm
#and return predicted probabilities and weight vector

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
neta = 10   ## number of different step size
eta0 = 0.1  ## first step size
leta = [ eta0*(i+1) for i in range(neta)]
# Define a function to implement logistic regression algorithm
def logistic_regression(X_train, y_train, X_test, eta, iteration):
    w = np.zeros(X_train.shape[1])

    for i in range(iteration):
        z = np.dot(X_train, w)
        prob = sigmoid(-y_train * z)  # Use the sigmoid function
        gradient = -np.dot(X_train.T, prob * y_train) / len(y_train)
        w -= eta * gradient

    # Calculate the predicted probabilities for the test set
    probs = expit(np.dot(X_test, w))

    return probs, w



# Loop over each step size
for eta in leta:
    max_avg_score = -1  # Initialize with a negative value

    fold_scores = []

    # Loop over each fold
    for train_index, test_index in cselection.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Call the logistic regression function with the training set and test set and the current step size and iteration
        probs, w = logistic_regression(X_train, y_train, X_test, eta, iteration)

        # Compute the ROC-AUC score for the validation set and append it to the fold score list
        score = roc_auc_score(y_test, probs)
        fold_scores.append(score)

    # Compute the average ROC-AUC score for the current step size and append it to the step score list
    avg_score = np.mean(fold_scores)
    print(avg_score)
    # Check if the current average score is greater than the current maximum
    if avg_score > max_avg_score:
        max_avg_score = avg_score
        best_eta = eta  # Update the best step size

    max_scores.append(max_avg_score)
    best_etas.append(best_eta)

# Find the maximum average score and the corresponding step size
max_score = max(max_scores)
best_eta = best_etas[max_scores.index(max_score)]

# Print the results
print(f'The maximum average score is {max_score:.2f} and the corresponding step size is {best_eta:.2f}.')