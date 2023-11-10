import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score

class svm_primal_cls:
    def __init__(self, C=1000, eta=0.1, xlambda=0.01, nitermax=10):
        self.C = C
        self.eta = eta
        self.nitermax = nitermax
        self.xlambda = xlambda
        self.w = None

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)

        for iter in range(self.nitermax):
            for i in range(m):
                x = X[i]
                y_i = y[i]

                decision = np.dot(self.w, x)     # weight multiple x

                if y_i * decision <= 1:          # 错误分类 正类别乘正数 负类别乘负数 都应该大于1
                    gradient = self.xlambda * self.w - y_i * x
                # 如果样本点没有被正确分类，这行代码计算梯度。梯度是损失函数相对于权重向量的变化率。self.xlambda
                # 是正则化参数，用于控制模型的复杂度，self.w
                # 是当前的权重向量，x是输入样本。这里采用的是合页损失函数（hinge loss），其中正类别点的损失是 0，负类别点的损失是
                # 1 - y_i * decision。因此，梯度计算包括两部分：正则化项和损失项。
                else:
                    gradient = self.xlambda * self.w
                self.w -= self.eta * gradient  #通过减去梯度乘以学习率，更新权重向量，以减小损失函数的值。

    def predict(self, Xtest):
        m_test = Xtest.shape[0] #计算输入数据集 Xtest 的样本数量，存储在 m_test 变量中。
        y_pred = np.zeros(m_test)

        for i in range(m_test):
            x = Xtest[i]
            decision = np.dot(self.w, x)
            y_pred[i] = 1 if decision > 0 else -1 #根据决策函数的值，将数据点分配到正类别（1）或负类别（-1）。
            # 如果 decision 大于零，将类别标签设置为 1，否则设置为 -1。循环继续，处理下一个数据点，直到所有数据点都被预测完毕。

        return y_pred


def main():
  X, y = load_breast_cancer(return_X_y=True)
  y = 2 * y - 1

  mdata, ndim = X.shape
  X /= np.outer(np.ones(mdata), np.max(np.abs(X), 0))

  np.random.seed(12345)
  nitermax_primal = 200
  eta = 0.1
  lC = [100, 200, 500, 1000, 2000]
  nC = len(lC)
  nfold_outer = 5
  nfold_inner = 4
  nmethod = 2

  cselection_outer = KFold(n_splits=nfold_outer, random_state=None, shuffle=False)

  f1_scores_primal = []  #两个列表用来储存 primal和svc的 分数
  f1_scores_svc = []

  ifold = 0
  for index_train, index_test in cselection_outer.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]
    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]
    print('Training size:', mtrain)
    print('Test size:', mtest)

    mean_f1_scores_primal = []  ##
    mean_f1_scores_svc = []

    for iC in range(nC):
      C = lC[iC]
      csvm_primal = svm_primal_cls(C=C, eta=eta, xlambda=1 / C, nitermax=nitermax_primal)
      svc_lin = SVC(C=C, kernel='linear')

      cselection_inner = KFold(n_splits=nfold_inner, random_state=None, shuffle=False)

      ifold_in = 0
      f1_scores_primal_C = []  # Collect f1 scores for the current C
      f1_scores_svc_C = []  # Collect f1 scores for the current C
      for index_in_train, index_in_test in cselection_inner.split(Xtrain):
        Xtrain_in = Xtrain[index_in_train]
        ytrain_in = ytrain[index_in_train]
        Xtest_in = Xtrain[index_in_test]
        ytest_in = ytrain[index_in_test]
        mtrain_in = Xtrain_in.shape[0]
        mtest_in = Xtest_in.shape[0]

        csvm_primal.fit(Xtrain_in, ytrain_in)
        y_pred_primal = csvm_primal.predict(Xtest_in)
        f1_score_primal = f1_score(ytest_in, y_pred_primal)
        f1_scores_primal_C.append(f1_score_primal)

        svc_lin.fit(Xtrain_in, ytrain_in)
        y_pred_svc = svc_lin.predict(Xtest_in)
        f1_score_svc = f1_score(ytest_in, y_pred_svc)
        f1_scores_svc_C.append(f1_score_svc)

        ifold_in += 1

      mean_f1_scores_primal.append(np.mean(f1_scores_primal_C))  # Append mean f1 score for the current C
      mean_f1_scores_svc.append(np.mean(f1_scores_svc_C))  # Append mean f1 score for the current C

    best_C_primal = lC[np.argmax(mean_f1_scores_primal)]
    best_C_svc = lC[np.argmax(mean_f1_scores_svc)]

    csvm_primal_best = svm_primal_cls(C=best_C_primal, eta=eta, xlambda=1 / best_C_primal, nitermax=nitermax_primal)
    svc_lin_best = SVC(C=best_C_svc, kernel='linear')

    csvm_primal_best.fit(Xtrain, ytrain)
    y_pred_primal_best = csvm_primal_best.predict(Xtest)
    f1_score_primal_best = f1_score(ytest, y_pred_primal_best)

    svc_lin_best.fit(Xtrain, ytrain)
    y_pred_svc_best = svc_lin_best.predict(Xtest)
    f1_score_svc_best = f1_score(ytest, y_pred_svc_best)

    f1_scores_primal.append(f1_score_primal_best)  # Append the best f1 score for the current fold
    f1_scores_svc.append(f1_score_svc_best)  # Append the best f1 score for the current fold

    ifold += 1

  mean_f1_primal = np.mean(f1_scores_primal)
  mean_f1_svc = np.mean(f1_scores_svc)
  std_f1_primal = np.std(f1_scores_primal)
  std_f1_svc = np.std(f1_scores_svc)
  std_ratio = std_f1_primal / std_f1_svc
  print("std_ratio:", std_ratio)


if __name__ == "__main__":
  main()