
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



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

```

