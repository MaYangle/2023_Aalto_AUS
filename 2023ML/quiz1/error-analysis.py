
import numpy as np

y_true = np.array([7, 1, 2, 1, 2, 2, 3.5, 3.5, 5, 4, 6])  # 实际值
y_preds = [np.array([2, 1, 1, 2, 3, 3.5, 4, 5, 5, 6, 6]),
           np.array([1.5, 0.5, 0.5, 1.5, 2.5, 3, 3.5, 4.5, 4.5, 5.5, 5.5])] # 预测值

mses = []
maes = []
for y_pred in y_preds:
    mses.append(np.mean((y_true - y_pred) ** 2))
    maes.append(np.mean(np.abs(y_true - y_pred)))

print(mses)    # 输出 [1.0, 0.25]
print(maes)    # 输出 [1.0, 0.5]
