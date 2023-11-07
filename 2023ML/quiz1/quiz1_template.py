
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

"""
The data in this exercise is a subset of the much used Breast cancer Wisconsin dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Documentation states the class distribution as: 212 - Malignant, 357 - Benign
"""

# 加载数据集
X = np.load("quiz1_X.npy")
y = np.load("quiz1_y.npy")

n = len(y)
print("There are %d samples in total in the dataset" % n)
print("The shape of X:", X.shape)

print("Unique labels in y:", np.unique(y))
print("Counts of labels in y:", Counter(y))

# 打乱数据集的顺序，并将其分为训练集和测试集

# # shuffle the data and randomly divide it to training and testing
# # give the random generator a seed to get reproducable results
np.random.seed(0)
order = np.random.permutation(n)

# --------------------------------------------------------------------------------------------------------------------
# Note! Random number generator has changed between numpy versions X and Y! In this course we use up-to-date versions.
# Check that the generated order should be the same as what is given in the materials:
# order = np.load("quiz1_sample_order.npy")
# --------------------------------------------------------------------------------------------------------------------


tr_samples = order[:int(0.5*n)]
tst_samples = order[int(0.5*n):]
print("The data is divided into %d training and %d test samples" % (len(tr_samples), len(tst_samples)))
Xtr = X[tr_samples, :]
Xtst = X[tst_samples, :]
ytr = y[tr_samples]
ytst = y[tst_samples]

# 创建线性回归模型和线性支持向量机模型对象
lr = LinearRegression(fit_intercept=False)
svm = LinearSVC(dual=False)

# 使用训练集数据训练模型
lr.fit(Xtr, ytr)
svm.fit(Xtr, ytr)


# # this is a helper function for transforming continuous labels to binary ones
# # works with both 0&1 and -1&1 labels
def get_classification_labels_from_regression_predictions(unique_labels, y_pred):
    assert len(unique_labels) == 2  # this function is meant only for binary classification

    meanval = np.mean(unique_labels)

    transformed_predictions = np.zeros(len(y_pred))
    transformed_predictions[y_pred < meanval] = np.min(unique_labels)
    transformed_predictions[y_pred >= meanval] = np.max(unique_labels)

    return transformed_predictions

# 使用测试集数据对模型进行预测，并计算预测准确率
y_pred_lr = lr.predict(Xtst)
y_pred_lr = get_classification_labels_from_regression_predictions(np.unique(y), y_pred_lr)
accuracy_lr = accuracy_score(ytst, y_pred_lr)

y_pred_svm = svm.predict(Xtst)
accuracy_svm = accuracy_score(ytst, y_pred_svm)

# 计算两个模型的测试准确率之差
accuracy_difference = accuracy_svm - accuracy_lr

print("The difference between the test accuracies of the model is", accuracy_difference)
print("THe answer of svm" , accuracy_svm)
print("THe answer of lr" , accuracy_lr)
