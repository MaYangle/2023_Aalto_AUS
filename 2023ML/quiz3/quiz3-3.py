
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
import math

# load the data

X, y = load_breast_cancer(return_X_y=True)  ## X input, y output



mdata,ndim = X.shape



## to convert the {0,1} output into {-1,+1}

y = 2*y - 1



## hyperparameters of the learning task of Question 3

## list of step sizes

leta = [0.1 * (i+1) for i in range(10)]



## number of iteration

iteration = 50



nfold = 5         ## number of folds



np.random.seed(12345)  ## fix the random seed to avoid random fluctuation of the results



## to split the data into 5-folds we need

cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)



## normalization

## scaling the rows by maximum absolute value, L infinite norm of columns

X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))
# your codes begin #

# Lists to store evaluation results
max_avg_score = -1  # Initialize with a negative value
best_eta = None

def logistic(y_i, w, x_i):  #for the positive feedback
    dot_product = np.dot(w, x_i)
    exponential = math.e**(y_i * dot_product)
    results = 1 / (1 + exponential)
    return results

def negative_logistic(y_i, w, x_i): #for the negative
    dot_product = np.dot(w, x_i)
    exponential = math.e**(-1 * y_i * dot_product)
    results = 1 / (1 + exponential)
    return results

# Loop over each step size
for eta in leta:
    fold_scores = []

    # Perform k-fold cross-validation
    for train_index, test_index in cselection.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w = np.zeros(ndim)

        for _ in range(iteration):
            for input_x, output_y in zip(X_train, y_train):
                delta_w = -1 * logistic(output_y, w, input_x) * output_y * input_x
                w = w - eta * delta_w

        pr_y_positive = []
        for x_i in X_test:
            pr = negative_logistic(1, w, x_i)
            pr_y_positive.append(pr)

        pr_y_positive = [1 if label > 0.5 else -1 for label in pr_y_positive]

        # Compute the ROC-AUC score for the validation set and append it to the fold score list
        score = roc_auc_score(y_test, pr_y_positive)
        fold_scores.append(score)

    # Compute the average ROC-AUC score for the current step size
    avg_score = np.mean(fold_scores)

    # Check if the current average score is greater than the current maximum
    if avg_score > max_avg_score:
        max_avg_score = avg_score
        best_eta = eta  # Update the best step size

# Print the results
print(f'The maximum average score is {max_avg_score:.2f} and the corresponding step size is {best_eta:.2f}.')
