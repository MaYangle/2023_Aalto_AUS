## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score

## ##############################################################

class svm_primal_cls:
  """
  Class implementing the primal SVM algorith:
  "Stochastic gradient descent algorithm for soft-margin SVM"
  """

  def __init__(self, C = 1000, eta = 0.1, xlambda = 0.01, nitermax = 10):

    self.C = C        ## Penalty coefficient 
    self.eta = eta    ## stepsize
    self.nitermax = nitermax  ## nunumber of iteration
    self.xlambda = xlambda    ## penalty parameter 1/C, see Slide 17,Lecture 6
    self.w = None             ## weight parameters
    
    return

  ## ---------------------------------------------------------
  def fit(self,X,y):
    """
    Task: to train the support Vector Machine
          by Stochastic gradient descent algorithm
    Input:  X      2d array of input examples in the rows
            y      1d(vector) array of +1,-1 labels
    Modifies: self.w  weight vector         
    """

    m,n = X.shape
    
    self.w = np.zeros(n)
    ## primal training algorithm
    for iter in range(nitermax):   
      for i in range(m):          ## load the data examples
        x = X[i]
        y = y[i]
        ## primal training step
    
    return
  ## ------------------------------------------
  def predict(self,Xtest):
    """
    Task: to predict the labels for the given examples based on the self.w
    Input:  Xtest    2d array of input examples in the rows
    Output: y    1d array of predicted labels
    """

    ## predictor function
    
    
    return(y)

## ##############################################################

def main():

  # load the data
  X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
  print(X.shape, y.shape)
  ## to convert the {0,1} output into {-1,+1}
  y = 2 * y -1

  mdata,ndim=X.shape   ## size of the data 

  ## normalize the input variables
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

  ## fix the random number generator  
  np.random.seed(12345) 
    
  ## number of iteration
  nitermax_primal = 200        
  ## fixed hyper-parameters for the primal algorithm
  eta = 0.1  ## stepsize
    
  ## list of penalty constats used in the cross validation
  lC = [100, 200, 500, 1000, 2000]
  nC = len(lC) 

  ## Nested cross-validation
  nfold_outer = 5   ## number of folds in the outer loop
  nfold_inner = 4   ## number of folds in the inner loop
  
  ## methods applied
  nmethod = 2 ## 0 svm_primal, 1 svc_linear

  ## split the data into 5-folds
  cselection_outer = KFold(n_splits=nfold_outer, random_state=None, \
    shuffle=False)

  ## run the cross-validation
  ## outer loop
  ifold = 0
  for index_train, index_test in cselection_outer.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]
    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]
    print('Training size:',mtrain)
    print('Test size:',mtest)    

    ## process the hyper parameters
    for iC in range(nC):

      C = lC[iC]  
      
      ## Xtrain, ytrain contains the data to split in the validation

      ## Initialize the learners
      ## xlambda = 1/C, Slide 17, Lecture 6
      csvm_primal = svm_primal_cls(C = C, eta = eta, xlambda = 1/C , \
        nitermax = nitermax_primal) 

      ## sklearn scv method
      svc_lin = SVC(C = C, kernel = 'linear')

      ## split the training into folds
      cselection_inner = KFold(n_splits=nfold_inner, random_state=None, \
        shuffle=False)

      ## inner loop 
      ifold_in = 0
      for index_in_train, index_in_test in cselection_inner.split(Xtrain):

        ## Only the training data is used!!!
        Xtrain_in = Xtrain[index_in_train]
        ytrain_in = ytrain[index_in_train]
        Xtest_in = Xtrain[index_in_test]
        ytest_in = ytrain[index_in_test]
        mtrain_in = Xtrain_in.shape[0]
        mtest_in = Xtest_in.shape[0]
      
        ## stochastic gradient primal descent
        ## training
        ## prediction
        ## compute the F1 score for the primal
        
        ## svc linear
        ## training
        ## prediction
        ## compute the F1 score for the svc


        ifold_in += 1

      ## end of inner loop

      ## compute the mean F1 score on the validation sets 
      ## for each learner for a given C value.     

    ## end of the C selection loop

    ## select the best C value with the highest mean F1 score for each method 
    ## run the methods with those values: C_primal, C_svc    

    ##  initialize the methods with the best C values
    csvm_primal = svm_primal_cls(C = C_primal, eta = eta, \
      xlambda = 1/C_primal, nitermax = nitermax_primal)
    
    svc_lin = SVC(C = C_svc, kernel = 'linear')
        
    ## stochastic gradient primal descent
    ## training
    ## prediction
    ## compute the F1 score for the primal
    
    ## svc linear
    ## training
    ## prediction
    ## compute the F1 score for the svc
      
    ifold += 1

  ## finally compute the mean of the F1 scores on 5 test sets processed
  ## in the outer loop for the two methods.     

  ## compute the standard deviations of the F1 scores on 5 test sets processed
  ## in the outer loop for the two methods.     

  ## compute the ration of standard deviations of the two methods
  ## ratio = std of primal / std of the svc.
  
    
  return

## ####################################################
## ###################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)

