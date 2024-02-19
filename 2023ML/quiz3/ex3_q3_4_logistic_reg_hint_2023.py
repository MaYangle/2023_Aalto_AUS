## ####################################################
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
## ###################################################
## ###################################################
def main(iworkmode):

  iscenario = 0  ## =0 step size enumration Question 3,
                 ## =1 iteration number enumeration, Question 4

  # load the data
  X, y = load_breast_cancer(return_X_y=True)  ## X input, y output

  print(X.shape, y.shape)
  mdata,ndim = X.shape

  ## to convert the {0,1} output into {-1,+1}
  y = 2*y - 1

  ## hyperparameters of the learning problem

  if iscenario == 0:  ## Question 3, step size enumeration
    ## list of eta, the stepsize or learning rate is enumerated
    neta = 10   ## number of different step size
    eta0 = 0.1  ## first setp size
    leta = [ eta0*(i+1) for i in range(neta)]  ## list of step sizes

    ## number of iteration
    iteration =50

  elif iscenario == 1: ## Question 4, iteration number enumeration
    ## list of iteration numbers
    niteration = 10  ## number of different iteration
    iteration0 = 10  ## first iteration number
    literation = [ iteration0*(i+1) for i in range(niteration)]

    ## step size
    eta = 0.1


  nfold = 5         ## number of folds

  np.random.seed(12345)

  ## split the data into 5-folds
  cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)

  ## normalization
  ## scaling the rows by maximum absolute value, L infinite norm of columns
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

  def logistic_regression(X_train, y_train, X_test, eta, iteration):
      w = np.zeros(X_train.shape[1])

      for i in range(iteration):
          # 计算整个训练集上的梯度
          gradient = np.dot(X_train.T, sigmoid(np.dot(X_train, w)) - y_train) / len(X_train)

          # 更新权重
          w = w - eta * gradient

      # 计算测试集的预测概率
      probs = expit(np.dot(X_test, w))

      return probs, w

  # loop over each step size in the list
  for eta in leta:
      # define an empty list to store the ROC-AUC score for each fold
      fold_scores = []
      # use KFold function to split the data into 5 folds and loop over each fold
      for train_index, test_index in cselection.split(X):
          # get the training and validation sets from the current fold
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]
          # call the logistic regression function with the training set and test set and the current step size and iteration
          probs, w = logistic_regression(X_train, y_train, X_test, eta, iteration)
          # compute the ROC-AUC score for the validation set and append it to the fold score list
          score = roc_auc_score(y_test, probs)
          fold_scores.append(score)
      # compute the average ROC-AUC score for the current step size and append it to the step score list
      avg_score = np.mean(fold_scores)

      # Check if the current average score is greater than the current maximum
      if avg_score > max_avg_score:
          max_avg_score = avg_score
          best_eta = eta  # 更新最佳步长

  # Print the results
  print(f'The maximum average score is {max_avg_score:.2f} and the corresponding step size is {best_eta:.2f}.')

## ####################################################
## ###################################################
# if __name__ == "__main__":
#   if len(sys.argv)==1:
#     iworkmode=0
#   elif len(sys.argv)>=2:
#     iworkmode=eval(sys.argv[1])
#   main(iworkmode)
#
#
